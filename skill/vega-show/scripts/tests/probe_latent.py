#!/usr/bin/env python3
"""
probe_latent.py — the latent renderers' probe on real bundles (BUILD-PLAN W12, D4 P7): the showcase
of each lane's bundle built and served by the bundle's own service, the page driven in a headless
Chrome, and what the latent area drew read back from the live page.

    python3 probe_latent.py --lanes MATRIX [--only L07,L08,L09] [--stub-briefs] [--work DIR]
                            [--template DIR] [--chrome EXE] [--json OUT] [--shots DIR]

MATRIX is a matrix directory (vega/scripts/tests/matrix.sh lanes --to pack --matrix MATRIX) whose
`lanes/<id>/work` holds the lane's packed bundle and model page. For each lane, in a copy of its work
directory: the fact set is extracted, the plan drafted unattended, authored and resolved, and the showcase built from the page
template (--template: another template directory, e.g. a patched copy); the bundle's service then
starts on a free port and Chrome (over --remote-debugging-pipe, reduced motion, 1440x900) opens
`/?probe=views`. Once the probe's walk is written the driver shows the network view and reads the
latent area (js/net/renderers/latent.js `latentReading()`, imported from the served page), then the
try view (the walk answered its first preset) and reads it again. --shots writes a PNG of each of the
two canvases per lane (<lane>-network.png, <lane>-try.png), for a reader to look at.

Stage briefs. A lane with `report: none` (L07, L08, L09) has no report, so the showcase links the
stage briefs (brief-link mode, fork S1), and the build refuses a showcase that links none (PF6).
The briefs are written by the agentic stages (vega/references/brief-contract.md); the mechanical lane
matrix writes none. Without --stub-briefs a lane whose pack and train briefs are absent is not
probed (exit 2, the missing briefs named). With --stub-briefs the probe writes a STAND-IN brief in
the lane's copy for each missing one (its frontmatter says `stand_in: probe_latent.py`, its body says
no stage wrote it), prints a WARN line per brief and records them in --json (`stand_in_briefs`): the
showcase then links those stand-ins, not what the lane produced. What is probed here is the network
and try canvases, which read no brief.

What is asserted (every lane): the probe is ready, every view of the tour is reached and rendered,
no page error before or after the walk; the network canvas has ink: in every segment it laid out,
or over the whole chain when it is drawn whole (whether or not a waist is marked). Per lane, by what its family draws:

    L07  pca-floor   the waist: bottleneckOf is the one dense stage; the network canvas marks that
                     column as the waist (net.caption.bottleneck) and it has ink
    L08  vae         the latent code: a latent segment of z rows with ink on the network canvas; on
                     the try canvas one answer's μ and σ (z numbers each, every σ > 0) and its KL
    L09  deep-svdd   the sphere: the head's exported centre (<head>.centre) as long as the head's
                     output, on both canvases; one answer's squared distance recomputed from the
                     head's output and that centre equals the service's within 1e-4·max(1, |d|)

Exit 0: every lane passes. 1: an assertion failed (each printed). 2: a lane, the browser or a tool
could not run.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import autoauthor  # noqa: E402
import build_showcase  # noqa: E402
import draft_plan  # noqa: E402
import extract_facts  # noqa: E402
import resolve_story  # noqa: E402

LANES = ("L07", "L08", "L09")
EXPECT = {"L07": "bottleneck", "L08": "latent", "L09": "sphere"}
CHROME_CANDIDATES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
VIEWPORT = (1440, 900)
READY_S, PROBE_S, VIEW_S = 300, 240, 20
SPHERE_TOL = 1e-4


class Unavailable(Exception):
    """A lane, the service or the browser could not run (exit 2)."""


def run(main, argv, cwd: Path) -> tuple[int, str]:
    """A command line in-process from `cwd`: (exit, what it printed)."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.chdir(cwd):
        code = main([str(a) for a in argv])
    return code, printed.getvalue()


# ---------------------------------------------------------------------------- the showcase
def lane_copy(matrix: Path, lane: str, work: Path) -> tuple[Path, Path, Path]:
    """(work copy, bundle, model page) of a packed lane."""
    source = matrix / "lanes" / lane / "work"
    if not source.is_dir():
        raise Unavailable(f"{source} is not there: run the lane through pack first")
    copy = work / lane
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(source, copy, symlinks=True)
    bundles = [p for p in sorted((copy / "bundle").iterdir()) if (p / "MANIFEST.json").is_file()]
    models = sorted(copy.glob("*.pack.html"))
    if not bundles or not models:
        raise Unavailable(f"{source}: no packed bundle or model page")
    return copy, bundles[0], models[0]


def missing_briefs(copy: Path, bundle: Path) -> list[Path]:
    """The stage briefs brief-link mode links that the lane did not write (a no-report lane run by
    the mechanical matrix writes none)."""
    targets = [bundle / "docs" / "pack-brief.md"]
    targets += [c / "docs" / "train-brief.md" for c in sorted((copy / "vega-runs").glob("*")) if c.is_dir()]
    return [p for p in targets if not p.exists()]


def stand_in_briefs(copy: Path, bundle: Path, allowed: bool) -> list[str]:
    """Without `allowed`, refuses (Unavailable) a lane with a missing brief; with it, writes a
    stand-in for each, marked as such, and returns their paths relative to the copy."""
    missing = missing_briefs(copy, bundle)
    if missing and not allowed:
        raise Unavailable("the lane wrote no stage brief for brief-link mode (the build refuses with PF6): "
                          + ", ".join(str(p.relative_to(copy)) for p in missing)
                          + " — run the lane's stages with their briefs, or pass --stub-briefs")
    for path in missing:
        stage = path.stem.split("-")[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nstage: {stage}\nstand_in: probe_latent.py\n---\n# {stage} brief (stand-in)\n\n"
                        "No stage wrote this brief: the lane ran without a report and without briefs, and the "
                        "latent probe wrote this stand-in so that a brief-link showcase could be built.\n",
                        encoding="utf-8")
    return [str(p.relative_to(copy)) for p in missing]


def build(copy: Path, bundle: Path, model: Path, template: Path | None) -> None:
    """extract, draft, author, resolve, build: <bundle>/showcase. A --template is given to every step
    that pins or reads the page template (draft, resolve, build), so the plan pins the template the
    showcase is built from (V1)."""
    plan, pin = copy / "show-plan", (["--template", template] if template else [])
    steps = [(extract_facts.main, ["--model", model, "--bundle", bundle, "--out", plan / "facts.json"]),
             (draft_plan.main, ["--facts", plan / "facts.json", "--out", plan / "draft.json", "--unattended", *pin]),
             (autoauthor.main, ["--plan", plan / "draft.json", "--facts", plan / "facts.json",
                                "--out", plan / "story-plan.json"]),
             (resolve_story.main, ["--plan", plan / "story-plan.json", "--facts", plan / "facts.json",
                                   "--out", plan / "story.json", *pin]),
             (build_showcase.main, ["--bundle", bundle, "--model", model, "--plan", plan / "story-plan.json",
                                    "--story", plan / "story.json", "--facts", plan / "facts.json",
                                    "--out", bundle / "showcase", "--unattended", *pin])]
    for main, argv in steps:
        code, printed = run(main, argv, copy)
        if code != 0:
            raise Unavailable(f"{main.__module__} exited {code}: {printed.strip()[-1500:]}")


# ---------------------------------------------------------------------------- the service
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def service(bundle: Path):
    """The bundle's own service on a free port, until it is ready: yields its base URL."""
    port = free_port()
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]), "HOME": "/tmp",
           "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    log = open(bundle.parent / f"{bundle.name}.serve.log", "wb")
    proc = subprocess.Popen([sys.executable, "-m", "serve.app", "--port", str(port), "--torch-threads", "2"],
                            cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + READY_S
        while True:
            if proc.poll() is not None:
                raise Unavailable(f"the service exited {proc.returncode} (log {log.name})")
            try:
                with urllib.request.urlopen(base + "/health", timeout=10) as r:
                    if json.loads(r.read()).get("ready"):
                        break
            except OSError:
                pass
            if time.time() > deadline:
                raise Unavailable(f"the service was not ready in {READY_S} s (log {log.name})")
            time.sleep(.5)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


# ---------------------------------------------------------------------------- the browser (CDP pipe)
def find_chrome(explicit: str | None) -> str | None:
    for c in [explicit, os.environ.get("CHROME"), *CHROME_CANDIDATES]:
        if c and (Path(c).is_file() or shutil.which(c)):
            return shutil.which(c) or c
    return None


class Browser:
    """A headless Chrome driven over --remote-debugging-pipe (fd 3 in, fd 4 out, NUL-framed JSON)."""

    def __init__(self, chrome: str, profile: Path):
        to_chrome, self._in = os.pipe()
        self._out, from_chrome = os.pipe()

        def fds():
            os.dup2(to_chrome, 3)
            os.dup2(from_chrome, 4)

        args = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                "--remote-debugging-pipe", f"--user-data-dir={profile}", "--force-prefers-reduced-motion",
                f"--window-size={VIEWPORT[0]},{VIEWPORT[1]}", "about:blank"]
        self.proc = subprocess.Popen(args, pass_fds=(3, 4), preexec_fn=fds, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        os.close(to_chrome)
        os.close(from_chrome)
        self._buf, self._id, self.session = b"", 0, None

    def call(self, method: str, params: dict | None = None, timeout: float = 60, session: bool = True) -> dict:
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        if session and self.session:
            msg["sessionId"] = self.session
        os.write(self._in, json.dumps(msg).encode() + b"\0")
        deadline = time.time() + timeout
        while True:
            while b"\0" not in self._buf:
                if time.time() > deadline:
                    raise Unavailable(f"the browser did not answer {method} in {timeout} s")
                chunk = os.read(self._out, 1 << 16)
                if not chunk:
                    raise Unavailable("the browser closed its pipe")
                self._buf += chunk
            raw, self._buf = self._buf.split(b"\0", 1)
            reply = json.loads(raw)
            if reply.get("id") == self._id:
                if "error" in reply:
                    raise Unavailable(f"{method}: {reply['error']}")
                return reply.get("result", {})

    def open(self, url: str) -> None:
        target = self.call("Target.createTarget", {"url": "about:blank"}, session=False)["targetId"]
        self.session = self.call("Target.attachToTarget", {"targetId": target, "flatten": True}, session=False)["sessionId"]
        for method, params in (("Page.enable", {}), ("Runtime.enable", {}),
                               ("Emulation.setDeviceMetricsOverride", {"width": VIEWPORT[0], "height": VIEWPORT[1],
                                                                       "deviceScaleFactor": 1, "mobile": False}),
                               ("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]}),
                               ("Page.navigate", {"url": url})):
            self.call(method, params)

    def evaluate(self, expression: str, timeout: float = 60):
        result = self.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True},
                           timeout=timeout)
        if "exceptionDetails" in result:
            raise Unavailable(f"the page raised: {json.dumps(result['exceptionDetails'])[:800]}")
        return result.get("result", {}).get("value")

    def wait(self, expression: str, seconds: float):
        deadline = time.time() + seconds
        while time.time() < deadline:
            value = self.evaluate(expression)
            if value:
                return value
            time.sleep(.25)
        return None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.call("Browser.close", session=False, timeout=10)
        with contextlib.suppress(Exception):
            self.proc.wait(10)
        if self.proc.poll() is None:
            self.proc.kill()
        for fd in (self._in, self._out):
            with contextlib.suppress(OSError):
                os.close(fd)


PROBE_OUT = "(() => { const n = document.getElementById('showcase-probe'); return n && n.textContent ? n.textContent : null; })()"
LATENT = ("(async () => { const main = document.querySelector('script[type=module][src$=\"js/main.js\"]');"
          " const m = await import(new URL('net/renderers/latent.js', new URL('.', main.src)).href);"
          " return m.latentReading(); })()")


def show_view(browser: Browser, view: str, canvas: str) -> None:
    """Shows a view of the tour and waits for its canvas's drawing (the drawings' state, S.draw)."""
    browser.evaluate(f"(() => {{ const b = document.querySelector('nav.index button[data-view=\"{view}\"]'); if (b) b.click(); return !!b; }})()")
    ready = (f"(async () => {{ const main = document.querySelector('script[type=module][src$=\"js/main.js\"]');"
             f" const {{ S }} = await import(new URL('core/state.js', new URL('.', main.src)).href);"
             f" const v = document.querySelector('.view[data-view=\"{view}\"]');"
             f" return !!(v && !v.hidden && S.draw && S.draw.{canvas} && S.draw.{canvas}.drawing); }})()")
    if not browser.wait(ready, VIEW_S):
        raise Unavailable(f"the {view} view drew no canvas in {VIEW_S} s")
    time.sleep(.5)


def shoot(browser: Browser, canvas_id: str, path: Path | None) -> None:
    """A PNG of one canvas of the page as it is drawn now (nothing when path is None)."""
    if path is None:
        return
    box = browser.evaluate(f"(() => {{ const c = document.getElementById('{canvas_id}'); if (!c) return null;"
                           " c.scrollIntoView({block: 'center'}); const r = c.getBoundingClientRect();"
                           " return {x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height}; })()")
    if not box or box["width"] <= 0 or box["height"] <= 0:
        return
    shot = browser.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True, "clip": {**box, "scale": 1}})
    path.write_bytes(base64.b64decode(shot["data"]))


def drive(chrome: str, base: str, work: Path, shots: tuple[Path, str] | None = None) -> dict:
    """The probe's output and the latent readings of the network and the try canvases (and, with
    shots = (directory, lane), a PNG of each canvas)."""
    profile = Path(tempfile.mkdtemp(prefix="chrome-", dir=work))
    browser = Browser(chrome, profile)
    png = (lambda view: shots[0] / f"{shots[1]}-{view}.png") if shots else (lambda view: None)
    try:
        browser.open(base + "/?probe=views")
        out = browser.wait(PROBE_OUT, PROBE_S)
        if not out:
            raise Unavailable(f"the probe wrote nothing in {PROBE_S} s")
        probe = json.loads(out)
        show_view(browser, "network", "net")
        network = browser.evaluate(LATENT)
        shoot(browser, "net", png("network"))
        show_view(browser, "try", "tryNet")
        tried = browser.evaluate(LATENT)
        shoot(browser, "net-try", png("try"))
        late = json.loads(browser.evaluate(PROBE_OUT) or "{}").get("late_errors", [])
        return {"probe": probe, "network": network, "try": tried, "late_errors": late}
    finally:
        browser.close()
        shutil.rmtree(profile, ignore_errors=True)


# ---------------------------------------------------------------------------- the assertions
def page_problems(result: dict) -> list[str]:
    probe, out = result["probe"], []
    if not probe.get("ready"):
        out.append(f"the page never became ready: {probe.get('error')}")
    for view, v in (probe.get("views") or {}).items():
        if not (v.get("reached") and v.get("rendered")):
            out.append(f"view {view}: reached {v.get('reached')}, rendered {v.get('rendered')}")
    errors = probe.get("page_errors", []) + probe.get("late_errors", []) + result.get("late_errors", [])
    out += [f"page error: {e}" for e in errors]
    net = (result.get("network") or {}).get("canvases", {}).get("network")
    if not net:
        out.append("the network view drew no canvas")
    elif net["drawn"] == "segments":
        out += [f"segment {i} ({s['kind']} {s['stage']}) has no ink" for i, s in enumerate(net["segments"]) if s["pixels"] <= 0]
    else:
        if not net.get("ink", 0) > 0:
            out.append("the chain drawn whole has no ink")
        if net["waist"]["pixels"] <= 0 and net["waist"]["stage"] is not None:
            out.append("the chain's waist column has no ink")
    return out


def bottleneck_problems(result: dict) -> list[str]:
    reading = result["network"]
    b, net = reading["bottleneck"], reading["canvases"]["network"]
    if not b or b["kind"] != "dense":
        return [f"bottleneckOf names no dense waist: {b}"]
    out = []
    if net["waist"]["stage"] != b["stage"]:
        out.append(f"the network canvas marks {net['waist']['stage']} as its waist, not {b['stage']} ({b['units']} units)")
    if not net["waist"]["caption"]:
        out.append("no column caption names the waist (net.caption.bottleneck)")
    if net["waist"]["pixels"] <= 0:
        out.append("the waist column has no ink")
    return out


def latent_problems(result: dict) -> list[str]:
    b = result["network"]["bottleneck"]
    if not b or b["kind"] != "latent":
        return [f"bottleneckOf names no latent stage: {b}"]
    out, z = [], b["units"]
    drawn = result["network"]["canvases"]["network"]["latent"]
    if len(drawn) != 1 or drawn[0]["pixels"] <= 0 or drawn[0]["z"] != z:
        out.append(f"the network canvas's latent segment: {drawn}")
    answered = (result["try"]["canvases"].get("try") or {}).get("latent") or []
    if len(answered) != 1 or not answered[0]["answered"]:
        return out + [f"the try canvas's latent segment was not answered: {answered}"]
    a = answered[0]
    if not a["mu"] or len(a["mu"]) != z or not a["sigma"] or len(a["sigma"]) != z or min(a["sigma"]) <= 0:
        out.append(f"one answer's μ and σ are not z = {z} numbers with σ > 0: {a}")
    if not isinstance(a["kl"], (int, float)) or a["kl"] < 0:
        out.append(f"one answer's KL is {a['kl']}")
    if a["pixels"] <= 0:
        out.append("the try canvas's latent segment has no ink")
    return out


def sphere_problems(result: dict) -> list[str]:
    s = result["network"]["sphere"]
    if not s or not s["centre"] or len(s["centre"]) != s["units"]:
        return [f"the head exports no centre as long as its output: {s}"]
    out = []
    for name, reading in (("network", result["network"]), ("try", result["try"])):
        drawn = (reading["canvases"].get("try" if name == "try" else "network") or {}).get("sphere")
        if not drawn or drawn["centre"] != s["units"] or drawn["pixels"] <= 0:
            out.append(f"the {name} canvas's sphere: {drawn}")
    answer = ((result["try"]["canvases"].get("try") or {}).get("sphere") or {}).get("answer")
    if not answer or answer["network"] is None:
        out.append(f"no answered distance to recompute: {answer}")
    elif answer["gap"] > SPHERE_TOL * max(1.0, abs(answer["network"])):
        out.append(f"the distance from the exported centre is {answer['page']}, the service's {answer['network']}")
    return out


CHECKS = {"bottleneck": bottleneck_problems, "latent": latent_problems, "sphere": sphere_problems}


def probe_lane(matrix: Path, lane: str, work: Path, chrome: str, opts: argparse.Namespace) -> dict:
    copy, bundle, model = lane_copy(matrix, lane, work)
    stand_ins = stand_in_briefs(copy, bundle, opts.stub_briefs)
    for rel in stand_ins:
        print(f"WARN: {lane}: no stage brief at {rel}; the showcase links a stand-in (--stub-briefs)")
    build(copy, bundle, model, Path(opts.template) if opts.template else None)
    with service(bundle) as base:
        result = drive(chrome, base, work, (Path(opts.shots), lane) if opts.shots else None)
    problems = page_problems(result)
    if result.get("network"):
        problems += CHECKS[EXPECT[lane]](result)
    return {"lane": lane, "expect": EXPECT[lane], "problems": problems, "stand_in_briefs": stand_ins, "result": result}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", required=True, help="the matrix directory with lanes/<id>/work packed")
    ap.add_argument("--only", default=",".join(LANES), help="comma list of lanes (default L07,L08,L09)")
    ap.add_argument("--stub-briefs", action="store_true",
                    help="write a marked stand-in for each stage brief a no-report lane lacks (default: refuse the lane)")
    ap.add_argument("--work", help="where the lane copies go (default: a temporary directory)")
    ap.add_argument("--template", help="the page template the showcases are built from (default: installed)")
    ap.add_argument("--chrome", help="the Chrome executable (default $CHROME, then the usual names)")
    ap.add_argument("--json", help="write every lane's result here")
    ap.add_argument("--shots", help="write a PNG of the network and the try canvases of each lane here")
    a = ap.parse_args(argv)
    # every step runs from inside its lane copy (run() changes directory), so the paths given are
    # made absolute here, against the directory the probe was started from
    for name in ("lanes", "work", "template", "json", "shots"):
        if getattr(a, name):
            setattr(a, name, str(Path(getattr(a, name)).resolve()))
    chrome = find_chrome(a.chrome)
    if not chrome:
        print("ERROR: no Chrome found (set $CHROME)")
        return 2
    lanes = [x.strip() for x in a.only.split(",") if x.strip()]
    unknown = [x for x in lanes if x not in EXPECT]
    if unknown:
        print(f"ERROR: no latent check for {', '.join(unknown)} (known: {', '.join(LANES)})")
        return 2
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="probe-latent-"))
    work.mkdir(parents=True, exist_ok=True)
    if a.shots:
        Path(a.shots).mkdir(parents=True, exist_ok=True)
    results, code = [], 0
    for lane in lanes:
        try:
            r = probe_lane(Path(a.lanes), lane, work, chrome, a)
        except Unavailable as exc:
            print(f"ERROR: {lane}: {exc}")
            results.append({"lane": lane, "error": str(exc)})
            code = max(code, 2)
            continue
        results.append(r)
        for p in r["problems"]:
            print(f"FAIL: {lane} ({r['expect']}): {p}")
        if r["problems"]:
            code = max(code, 1)
        else:
            print(f"OK: {lane} ({r['expect']}): {summary(r)}")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return code


def summary(r: dict) -> str:
    reading, views = r["result"]["network"], r["result"]["probe"].get("views", {})
    net = reading["canvases"]["network"]
    ink = net.get("ink", 0)
    return (f"{len(views)} views rendered, no page error; drawn as {net['drawn']} ({ink} px); "
            f"bottleneck {reading['bottleneck']}; sphere centre {len((reading['sphere'] or {}).get('centre') or [])}")


if __name__ == "__main__":
    sys.exit(main())
