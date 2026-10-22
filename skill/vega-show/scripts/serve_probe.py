"""
serve_probe — the served walk (contract §10 gates SV1–SV5; D5 §7.8; BUILD-PLAN R-H3): the
bundle's own service started with the skills off the path, its showcase slot verified, every
served file and every report or brief link fetched at its pinned digest, and the page walked by
its own probe through the one page driver (`smoke_page.drive`, `smoke_page_checks.judge`).

    ServedPageProbe(bundle, python, port, chrome).run(views="all") -> dict   vega/show-smoke@1

The service is `<python> -B -m serve.app --port P --torch-threads 2`, run in the bundle with a
clean environment (no PYTHONPATH, CPU, no bytecode written), polled on `/health` until ready.
The gates, each a named result in `checks` beside the page driver's own P-checks:

| gate | holds that |
|---|---|
| SV1 | `/health` is ready, `/health.showcase` and `/v1/showcase` say verified, `GET /` is the entry and every served file answers 200 with its pinned digest |
| SV2 | the probe ran and every view rendered with no page error (P-PROBE, P-ERR, P-VIS); the network view is available and an explanation answered |
| SV3 | no placeholder is unresolved in the visible text (P-TXT) |
| SV4 | every code-note link of the story and every link the probe followed answers 200 with the digest `SHOWCASE.json` pins for it |
| SV5 | the alert colour appears exactly when a class role is `alert`, and `data-motif` is `page.json`'s motif (P-ROLE, P-MOTIF) |

A slot that is not verified fails SV1 and the page is not walked (the service serves its console
in its place), so SV2, SV3 and SV5 fail with it. `status` is `pass` when every gate and every
P-check passed, `fail` otherwise; the service and
the browser that could not run make it `not-run` with `why_not_run` (the CLI's to report).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping
from urllib.parse import urldefrag, urljoin, urlsplit

import cdp
import smoke_page
from smoke_page_checks import judge
from vega_show_core import ShowPaths, digest_bytes, now

__all__ = ["ServedPageProbe", "NotRun", "SCHEMA"]

SCHEMA = "vega/show-smoke@1"
READY_S = 300
SHOTS = (("start", "/"), ("results", "/#results"))
GATE_CHECKS = {"SV2": ("P-PROBE", "P-ERR", "P-VIS"), "SV3": ("P-TXT",), "SV5": ("P-ROLE", "P-MOTIF")}


class NotRun(Exception):
    """The walk could not run: no browser, no service packages, a service that never came up."""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fetch(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or b""


def _json(url: str):
    status, body = fetch(url)
    return json.loads(body) if status == 200 else None


class Service:
    """The bundle's own service on a port, the skills off its path."""

    def __init__(self, bundle: Path, python: str, port: int | None) -> None:
        self.bundle, self.python, self.port = bundle, python, port or _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = bundle.parent / f"{bundle.name}.smoke-show.log"
        self.proc = None

    def env(self) -> dict:
        return {"PATH": os.pathsep.join([str(Path(self.python).parent), "/usr/bin", "/bin"]),
                "HOME": "/tmp", "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "",
                "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}

    def __enter__(self) -> "Service":
        missing = subprocess.run([self.python, "-B", "-c", "import fastapi, uvicorn"], env=self.env(),
                                 capture_output=True, text=True, cwd=self.bundle)
        if missing.returncode != 0:
            raise NotRun(f"the service's packages are not installed for {self.python} (requirements-serve.txt)")
        self._fh = open(self.log, "wb")
        self.proc = subprocess.Popen([self.python, "-B", "-m", "serve.app", "--port", str(self.port),
                                      "--torch-threads", "2"], cwd=self.bundle, env=self.env(),
                                     stdout=self._fh, stderr=subprocess.STDOUT)
        self._until_ready()
        return self

    def _until_ready(self) -> None:
        deadline = time.time() + READY_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise NotRun(f"the service exited {self.proc.returncode} before it was ready (log {self.log})")
            with contextlib.suppress(OSError, ValueError):
                if (_json(self.base + "/health") or {}).get("ready"):
                    return
            time.sleep(0.5)
        raise NotRun(f"the service was not ready in {READY_S} s (log {self.log})")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._fh.close()


def _gate(gid: str, problems: list[str]) -> dict:
    return {"id": gid, "ok": not problems, "problems": problems}


class ServedPageProbe:
    """One bundle's showcase, served and walked."""

    def __init__(self, bundle: Path, python: str, port: int | None, chrome: str | None) -> None:
        self.paths = ShowPaths.of(Path(bundle).resolve())
        self.python, self.port, self.chrome = python, port, chrome
        self.raw = self.paths.manifest.read_bytes()
        self.doc = json.loads(self.raw)
        self.page = json.loads(self.paths.page_config.read_text(encoding="utf-8"))
        self.story = json.loads(self.paths.story.read_text(encoding="utf-8"))

    def run(self, views: str = "all") -> dict:
        if not self.chrome:
            raise NotRun("no headless browser (checked --chrome, $CHROME and the usual names)")
        with Service(self.paths.bundle, self.python, self.port) as svc:
            sv1 = self._slot(svc.base)
            links, sv4 = self._links(svc.base, [])
            if not sv1["ok"]:               # the service falls back to its console: nothing to walk
                why = ["not walked: the showcase slot is not verified"]
                gates = [sv1] + [_gate(g, why) for g in ("SV2", "SV3")] + [_gate("SV4", sv4), _gate("SV5", why)]
                return self._record({}, gates, links, [], svc.log)
            result, checks = self._walk(svc.base, views)
            more, sv4b = self._links(svc.base, (result["probe"].get("links") or []), {l["href"] for l in links})
            shots = self._screenshots(svc.base)
        gates = [sv1, self._sv2(result["probe"], checks), self._from_checks("SV3", checks),
                 _gate("SV4", sv4 + sv4b), self._sv5(result["probe"], checks)]
        return self._record(result["probe"], gates + checks, links + more, shots, svc.log)

    # -------------------------------------------------------------------------------- SV1
    def _slot(self, base: str) -> dict:
        out, health = [], _json(base + "/health") or {}
        slot = _json(base + "/v1/showcase") or {}
        if not (health.get("showcase") or {}).get("verified"):
            out.append(f"/health showcase is not verified: {(health.get('showcase') or {}).get('reason')}")
        if not slot.get("verified"):
            out.append(f"/v1/showcase is not verified: {slot.get('reason')}")
        if slot.get("digest") not in (None, digest_bytes(self.raw)):
            out.append("/v1/showcase serves another SHOWCASE.json than the one on disk")
        status, body = fetch(base + "/")
        entry = self.paths.showcase / self.doc["entry"]
        if status != 200 or body != entry.read_bytes():
            out.append(f"GET / is not the showcase entry {self.doc['entry']} (status {status})")
        files = self.doc.get("files") or {}
        for rel in self.doc.get("served") or []:
            status, body = fetch(f"{base}/showcase/{rel}")
            if status != 200 or digest_bytes(body) != files.get(rel):
                out.append(f"showcase/{rel} answered {status} with {digest_bytes(body)[:19]}…, pinned {str(files.get(rel))[:19]}…")
        return _gate("SV1", out)

    # -------------------------------------------------------------------------------- SV4
    def _pinned_of(self, path: str) -> str | None:
        """The showcase file a served URL path answers with."""
        report = self.doc.get("report") or {}
        if path == "/report.pdf" and report.get("pdf"):
            return report["pdf"]["file"]
        return path[len("/showcase/"):] if path.startswith("/showcase/") else None

    def _story_links(self) -> list[dict]:
        out = []
        for step, note in sorted((self.story.get("code_notes") or {}).items()):
            link = (note or {}).get("link") or {}
            if link.get("href"):
                out.append({"step": step, "href": link["href"]})
        if (self.story.get("report") or {}).get("status") == "linked":
            out.append({"step": "report", "href": "/report.pdf"})
        return out

    def _links(self, base: str, probe_links, seen=frozenset()) -> tuple[list[dict], list[str]]:
        found, problems = [], []
        wanted = probe_links or self._story_links()
        for link in wanted:
            if link["href"] in seen:
                continue
            seen = seen | {link["href"]}
            path = urlsplit(urljoin(base + "/", urldefrag(link["href"])[0])).path
            pinned_rel = self._pinned_of(path)
            status, body = fetch(base + path)
            got, pinned = digest_bytes(body), (self.doc.get("files") or {}).get(pinned_rel or "")
            ok = status == 200 and pinned is not None and got == pinned
            found.append({"step": link.get("step"), "href": link["href"], "path": path, "status": status,
                          "digest": got, "pinned": pinned, "ok": ok})
            if not ok:
                problems.append(f"link {link['href']} of {link.get('step')} answered {status} with "
                                f"{got[:19]}…, pinned {str(pinned)[:19]}…")
        return found, problems

    # -------------------------------------------------------------------------------- the walk
    def _served(self, base: str):
        manifest = json.loads(self.paths.bundle_manifest.read_text(encoding="utf-8"))
        allowed = ("/favicon.ico",) + (() if manifest.get("monitor") else ("/v1/monitor",))
        docs = {"schema": _json(base + "/v1/schema"), "attribution": _json(base + "/showcase/data/attribution.json")}
        return smoke_page.Served(self.paths.bundle.name, base, self.page.get("lane") or {}, lambda: [],
                                 allowed, lambda: docs, True)

    def _walk(self, base: str, views: str) -> tuple[dict, list[dict]]:
        opts = types.SimpleNamespace(viewport="1440x900", mode=views, locale="en", mobile=False,
                                     motion="reduced", shots=None)
        served = self._served(base)
        result = smoke_page.drive(self.chrome, served, opts)
        return result, judge(result, served)

    def _screenshots(self, base: str) -> list[str]:
        self.paths.smoke.mkdir(parents=True, exist_ok=True)
        out = []
        with cdp.Browser(self.chrome) as browser:
            for name, path in SHOTS:
                page = browser.page()          # a tab of its own: a hash alone does not reload
                page.goto(base + path)
                page.wait("!!(window.__showcase && window.__showcase.ready)", 120)
                page.eval("(async () => { await window.__showcase.ready; return true; })()", timeout=180)
                time.sleep(1.0)
                target = self.paths.smoke / f"{name}.png"
                page.screenshot(target)
                out.append(str(target.relative_to(self.paths.bundle)))
        return out

    # -------------------------------------------------------------------------------- SV2, SV3, SV5
    @staticmethod
    def _from_checks(gid: str, checks: list[dict]) -> dict:
        names = GATE_CHECKS[gid]
        return _gate(gid, [f"{c['id']}: {p}" for c in checks if c["id"] in names for p in c["problems"]])

    def _sv2(self, probe: Mapping, checks: list[dict]) -> dict:
        out = self._from_checks("SV2", checks)["problems"]
        net, explain = probe.get("network") or {}, probe.get("explain") or {}
        if not net.get("available"):
            out.append(f"the network view is not available: {net.get('why')}")
        if not explain.get("ok"):
            out.append("no explanation answered on the walk (explain.ok is false)")
        return _gate("SV2", out)

    def _sv5(self, probe: Mapping, checks: list[dict]) -> dict:
        out = self._from_checks("SV5", checks)["problems"]
        roles = ((self.story.get("class_roles") or {}).get("roles") or {}).values()
        alert = any(r == "alert" for r in roles)
        used = bool((probe.get("palette") or {}).get("alert_used"))
        if used != alert:
            out.append(f"the alert colour is {'used' if used else 'not used'} and "
                       f"{'a' if alert else 'no'} class role is alert")
        want, drawn = (self.page.get("motif") or {}).get("id"), (probe.get("motif") or {}).get("data_motif")
        if drawn != want:
            out.append(f"data-motif is {drawn!r}, page.json's motif is {want!r}")
        return _gate("SV5", out)

    def _record(self, probe: Mapping, checks: list[dict], links: list[dict], shots: list[str], log: Path) -> dict:
        views = probe.get("views") or {}
        return {"schema": SCHEMA, "status": "pass" if all(c["ok"] for c in checks) else "fail",
                "why_not_run": None, "checked_at": now(), "bundle": self.paths.bundle.name,
                "showcase_digest": digest_bytes(self.raw), "verified": checks[0]["ok"],
                "views": sorted(k for k, v in views.items() if v.get("rendered")),
                "views_off": {k: v.get("off") for k, v in views.items() if v.get("off")},
                "page_errors": len(probe.get("page_errors") or []) + len(probe.get("late_errors") or []),
                "placeholders_unresolved": list(probe.get("placeholders_unresolved") or []),
                "links": links, "network": probe.get("network"), "explain": probe.get("explain"),
                "palette": probe.get("palette"), "motif": probe.get("motif"),
                "checks": checks, "screenshots": shots, "chrome": self.chrome, "service_log": str(log),
                "probe_digest": "sha256:" + hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()}


def not_run_record(bundle: Path, why: str) -> dict:
    paths = ShowPaths.of(Path(bundle).resolve())
    raw = paths.manifest.read_bytes() if paths.manifest.is_file() else b""
    return {"schema": SCHEMA, "status": "not-run", "why_not_run": why, "checked_at": now(),
            "bundle": paths.bundle.name, "showcase_digest": digest_bytes(raw) if raw else None,
            "verified": None, "views": [], "page_errors": None, "placeholders_unresolved": [],
            "links": [], "checks": [], "screenshots": []}
