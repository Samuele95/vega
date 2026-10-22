"""
cdp — a headless Chrome driven over the Chrome DevTools Protocol, standard library only
(BUILD-PLAN R-H3; D6 §10.1). `smoke_page.py` drives the page through it, and so do the lanes'
page stage and `smoke_show.py --serve`, through `smoke_page`.

    find_chrome(explicit=None) -> str | None     explicit, then $CHROME, then the usual names
    Browser(chrome, profile, viewport=(1440, 900))
        .page(viewport=None, mobile=False, scheme=None, reduced_motion=True) -> Page
        .close()
    Page
        .call(method, params=None) -> dict       one command of this page's session
        .events(methods=()) -> list[dict]        the events heard since the last read (all, or
                                                 those methods), each {method, params}
        .eval(expression, timeout=60)            Runtime.evaluate, awaited, by value
        .wait(expression, seconds) -> value      polls until truthy (None on timeout)
        .emulate(width, height, mobile=False, scheme=None, reduced_motion=True)
        .goto(url)                               Page.navigate (the load is the caller's to await)
        .screenshot(path, clip=None)             a PNG of the page (or of `clip`)
        .problems() -> list[str]                 what the page's own console and runtime said
                                                 was wrong: an uncaught exception, a console
                                                 error or assert, a log entry of level error

The browser runs with `--remote-debugging-pipe` (fd 3 in, fd 4 out, NUL-framed JSON): no port is
opened, so nothing else on the machine can reach it, and two drivers never race for one. Each
page is a target of its own, attached with `flatten`, with `Page`, `Runtime`, `Log` and
`Network` enabled from the start, so an exception thrown while the page loads is heard. A fresh
profile directory per browser; `TMPDIR` is moved under /tmp when the scratch path is too long
for Chrome's singleton socket (108 bytes). The page runs in a 1440×900 window under reduced
motion unless told otherwise.

Errors: `CdpError` (the browser died, a command failed or timed out, the page threw while
evaluating). Exit and printing are the callers'.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

__all__ = ["CHROME_CANDIDATES", "CdpError", "find_chrome", "Browser", "Page"]

CHROME_CANDIDATES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
VIEWPORT = (1440, 900)
READ_CHUNK = 1 << 16
ERROR_LEVELS = ("error", "assert")
_SPAWN = threading.RLock()       # browsers started from several threads (smoke_page --jobs) fork one at a time


class CdpError(Exception):
    """The browser could not do what it was asked."""


def find_chrome(explicit: str | None = None) -> str | None:
    """A Chrome executable: `explicit`, then $CHROME, then the usual names on PATH."""
    for c in (explicit, os.environ.get("CHROME"), *CHROME_CANDIDATES):
        if c and (Path(c).is_file() or shutil.which(c)):
            return str(Path(c)) if Path(c).is_file() else shutil.which(c)
    return None


class Browser:
    """One headless Chrome over its debugging pipe."""

    def __init__(self, chrome: str, profile: Path | None = None, viewport: tuple = VIEWPORT,
                 extra_args: tuple = ()):
        self._own_profile = profile is None
        self.profile = Path(profile) if profile else Path(tempfile.mkdtemp(prefix="cdp-profile-"))
        with _SPAWN:
            to_chrome, self._in = os.pipe()
            self._out, from_chrome = os.pipe()

        def fds():                      # in the child: the pipe ends to fds 3 and 4, in that order
            inbound, outbound = to_chrome, from_chrome
            if outbound == 3:               # moved aside first, or the first dup2 would close it
                outbound = os.dup(outbound)
            os.dup2(inbound, 3)
            os.dup2(outbound, 4)
            os.set_inheritable(3, True)     # dup2 onto itself keeps close-on-exec
            os.set_inheritable(4, True)

        args = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                "--no-sandbox", "--disable-extensions", "--disable-background-networking", "--mute-audio",
                "--remote-debugging-pipe", f"--user-data-dir={self.profile}",
                f"--window-size={viewport[0]},{viewport[1]}", *extra_args, "about:blank"]
        env = dict(os.environ)
        if len(tempfile.gettempdir()) > 60 and Path("/tmp").is_dir():
            env["TMPDIR"] = "/tmp"
        with _SPAWN:                        # one fork at a time: the fd numbering is the process's
            # no pass_fds=(3, 4): Popen would make the parent's fds 3 and 4 inheritable before fds()
            # runs, and fail (EBADF) when another thread has just closed what held those numbers (a
            # browser of smoke_page --jobs or page_smoke closing); Python's own fds are not
            # inheritable, so close_fds=False hands Chrome only 0-2 and the two fds() sets up
            self.proc = subprocess.Popen(args, close_fds=False, preexec_fn=fds, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, env=env)
            os.close(to_chrome)
            os.close(from_chrome)
        self._buf, self._id, self._events, self.viewport = b"", 0, {}, viewport
        self.version = self.call("Browser.getVersion", timeout=60).get("product", "")

    # -- the pipe ------------------------------------------------------------------------
    def call(self, method: str, params: dict | None = None, session: str | None = None,
             timeout: float = 60) -> dict:
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        if session:
            msg["sessionId"] = session
        try:
            os.write(self._in, json.dumps(msg).encode("utf-8") + b"\0")
        except OSError as exc:
            raise CdpError(f"the browser closed its pipe ({exc})") from None
        deadline = time.time() + timeout
        while True:
            reply = self._read(deadline, method)
            if reply.get("id") == self._id:
                if "error" in reply:
                    raise CdpError(f"{method}: {reply['error'].get('message', reply['error'])}")
                return reply.get("result", {})
            if "method" in reply:
                self._events.setdefault(reply.get("sessionId"), []).append(
                    {"method": reply["method"], "params": reply.get("params", {})})

    def _read(self, deadline: float, method: str) -> dict:
        while b"\0" not in self._buf:
            if time.time() > deadline:
                raise CdpError(f"the browser did not answer {method} in time")
            if self.proc.poll() is not None:
                raise CdpError(f"the browser exited ({self.proc.returncode})")
            ready, _, _ = select.select([self._out], [], [], 0.5)
            if not ready:
                continue
            chunk = os.read(self._out, READ_CHUNK)
            if not chunk:
                raise CdpError("the browser closed its pipe")
            self._buf += chunk
        raw, self._buf = self._buf.split(b"\0", 1)
        return json.loads(raw)

    def pump(self, seconds: float = 0.0) -> None:
        """Reads what the browser has sent (events) for up to `seconds`."""
        deadline = time.time() + seconds
        while True:
            while b"\0" in self._buf:
                raw, self._buf = self._buf.split(b"\0", 1)
                msg = json.loads(raw)
                if "method" in msg:
                    self._events.setdefault(msg.get("sessionId"), []).append(
                        {"method": msg["method"], "params": msg.get("params", {})})
            left = deadline - time.time()
            ready, _, _ = select.select([self._out], [], [], max(0.0, min(left, 0.2)))
            if ready:
                chunk = os.read(self._out, READ_CHUNK)
                if not chunk:
                    return
                self._buf += chunk
            elif left <= 0:
                return

    def drain(self, session: str) -> list:
        return self._events.pop(session, [])

    # -- pages ---------------------------------------------------------------------------
    def page(self, viewport: tuple | None = None, mobile: bool = False, scheme: str | None = None,
             reduced_motion: bool = True) -> "Page":
        target = self.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = self.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        page = Page(self, session, target)
        for domain in ("Page", "Runtime", "Log", "Network"):
            page.call(f"{domain}.enable")
        w, h = viewport or self.viewport
        page.emulate(w, h, mobile=mobile, scheme=scheme, reduced_motion=reduced_motion)
        return page

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.call("Browser.close", timeout=10)
        with contextlib.suppress(Exception):
            self.proc.wait(10)
        if self.proc.poll() is None:
            self.proc.kill()
            with contextlib.suppress(Exception):
                self.proc.wait(10)
        for fd in (self._in, self._out):
            with contextlib.suppress(OSError):
                os.close(fd)
        if self._own_profile:
            shutil.rmtree(self.profile, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class Page:
    """One tab of the browser, with its own session and its own events."""

    def __init__(self, browser: Browser, session: str, target: str):
        self.browser, self.session, self.target = browser, session, target
        self._heard: list = []

    def call(self, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        return self.browser.call(method, params, session=self.session, timeout=timeout)

    def events(self, methods: tuple = ()) -> list:
        """The events heard since the last read (every one is kept for problems())."""
        self.browser.pump(0)
        fresh = self.browser.drain(self.session)
        self._heard.extend(fresh)
        return [e for e in fresh if not methods or e["method"] in methods]

    def heard(self) -> list:
        self.events()
        return list(self._heard)

    def eval(self, expression: str, timeout: float = 60):
        got = self.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True},
                        timeout=timeout)
        if "exceptionDetails" in got:
            d = got["exceptionDetails"]
            raise CdpError(f"the page threw: {(d.get('exception') or {}).get('description') or d.get('text')}")
        return (got.get("result") or {}).get("value")

    def wait(self, expression: str, seconds: float, every: float = 0.25):
        deadline = time.time() + seconds
        while time.time() < deadline:
            value = self.eval(expression)
            if value:
                return value
            time.sleep(every)
        return None

    def emulate(self, width: int, height: int, mobile: bool = False, scheme: str | None = None,
                reduced_motion: bool = True) -> None:
        self.call("Emulation.setDeviceMetricsOverride", {"width": int(width), "height": int(height),
                                                         "deviceScaleFactor": 1, "mobile": bool(mobile)})
        features = [{"name": "prefers-reduced-motion", "value": "reduce" if reduced_motion else "no-preference"}]
        if scheme:
            features.append({"name": "prefers-color-scheme", "value": scheme})
        self.call("Emulation.setEmulatedMedia", {"features": features})

    def goto(self, url: str) -> None:
        got = self.call("Page.navigate", {"url": url})
        if got.get("errorText"):
            raise CdpError(f"navigation to {url} failed: {got['errorText']}")

    def screenshot(self, path: Path, clip: dict | None = None) -> None:
        params = {"format": "png", "captureBeyondViewport": True}
        if clip:
            params["clip"] = {**clip, "scale": 1}
        shot = self.call("Page.captureScreenshot", params)
        Path(path).write_bytes(base64.b64decode(shot["data"]))

    def problems(self) -> list[str]:
        """What the page's runtime and console reported as wrong, as text, in order."""
        return [r["text"] for r in self.problem_records()]

    def problem_records(self) -> list[dict]:
        """The same, each {kind: exception | console | log, text, url (the resource, for a log
        entry about one)}."""
        out = []
        for e in self.heard():
            p = e["params"]
            if e["method"] == "Runtime.exceptionThrown":
                d = p.get("exceptionDetails") or {}
                text = ((d.get("exception") or {}).get("description") or d.get("text") or "")[:400]
                out.append({"kind": "exception", "text": f"exception: {text}", "url": None})
            elif e["method"] == "Runtime.consoleAPICalled" and p.get("type") in ERROR_LEVELS:
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args") or [])[:400]
                out.append({"kind": "console", "text": f"console.{p['type']}: {text}", "url": None})
            elif e["method"] == "Log.entryAdded" and (p.get("entry") or {}).get("level") == "error":
                entry = p["entry"]
                out.append({"kind": "log", "text": f"log ({entry.get('source')}): {entry.get('text', '')[:300]} {entry.get('url') or ''}".strip(),
                            "url": entry.get("url")})
        return out

    def responses(self) -> list[tuple[str, int]]:
        """(url, status) of every response the page received."""
        return [(e["params"]["response"]["url"], int(e["params"]["response"]["status"]))
                for e in self.heard() if e["method"] == "Network.responseReceived"]
