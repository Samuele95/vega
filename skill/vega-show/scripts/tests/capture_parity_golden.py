#!/usr/bin/env python3
"""
capture_parity_golden.py — the page template's parity with the hand-built page, on the CIC R7 bundle.

    python3 capture_parity_golden.py [--check] [--template DIR] [--strict]      # compare (default)
    python3 capture_parity_golden.py --write [--strict]                          # capture the golden
        [--upstream URL | --bundle DIR] [--golden PATH] [--python EXE] [--chrome EXE]

--write renders the hand-built page of commit 0d06aea (vega-forge/…/serving/static/index.html)
with the bundle's identity, exactly as pack.py rendered it into R7 (and checks the result against
the page R7 itself serves), hooks the template's probe (js/core/probe.js) into it, serves it
through parity_server.py in front of R7's own running service, drives it in headless Chrome
(`?probe=all`, parity_browser.py) and writes the facts to golden/cic-parity.json.

--check serves the template (vega-show/assets/page by default) the same way, drives it with the
same probe and compares. The facts compared (`compare` in the golden):

  guide         the reading guide's texts, step by step
  choices       the number of decisions
  experiments   the tried variants: their count and their order
  weights_map   every cell of the weights map, both readings (value to 3 significant digits)
  stepper       for every preset, the z of every step of "how the network computed it"

Both runs must also be clean: the page booted, every view rendered, no uncaught error, rejected
promise, failed resource, CSP refusal or console.error (during the walk or after it, until the
browser's time budget ends), no answer of 400 or more from the server, and every URL the page
fetched same-origin and under health, v1/ or showcase/.

The upstream is R7's own service: --upstream when it runs already, otherwise the bundle in
--bundle (default $CIDS_ROOT/bundle/cicids2017-label-R7; CIDS_ROOT is required when neither is
given) is started read-only on a free port with --python (default: this interpreter; it needs
torch, fastapi and uvicorn). Chrome: $CHROME, else google-chrome, chromium.

Exit 0: parity (or, without --strict, no browser or no bundle: WARN). 1: a fact differs or a run
is not clean. 2: the capture could not run (with --strict: no browser, no bundle, CIDS_ROOT unset).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urljoin

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import parity_browser  # noqa: E402
import parity_server  # noqa: E402

REPO = HERE.parents[2]
TEMPLATE = REPO / "vega-show" / "assets" / "page"
GOLDEN = HERE / "golden" / "cic-parity.json"
LEGACY_COMMIT = "0d06aea"
LEGACY_PATH = "vega-forge/scripts/backends/pytorch/templates/serving/static/index.html"
CIDS_ROOT = os.environ.get("CIDS_ROOT") or None
BUNDLE = Path(CIDS_ROOT) / "bundle" / "cicids2017-label-R7" if CIDS_ROOT else None
COMPARE = ("guide", "choices", "experiments", "weights_map", "stepper")


class Unavailable(Exception):
    """The capture cannot run here (no browser, no bundle, no git): WARN, or ERROR under --strict."""


def say(level: str, msg: str):
    print(f"{level}: {msg}", flush=True)


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---- the upstream: R7's own service -------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str, timeout: float = 30) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class Upstream:
    """R7's service: the one given, or one started from the bundle and stopped at the end."""

    def __init__(self, url: str | None, bundle: Path | None, python: str, workdir: Path):
        self.proc, self.log = None, workdir / "upstream.log"
        if url:
            self.url = url.rstrip("/")
            return
        if bundle is None:
            raise Unavailable("CIDS_ROOT not set (set it, or pass --bundle or --upstream)")
        if not (bundle / "serve" / "app.py").is_file():
            raise Unavailable(f"no bundle service at {bundle} (set CIDS_ROOT or pass --bundle/--upstream)")
        port = free_port()
        self.url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        with open(self.log, "wb") as log:
            self.proc = subprocess.Popen([python, "-m", "serve.app", "--bundle", str(bundle), "--port", str(port)],
                                         cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT)

    def wait_ready(self, seconds: float = 240):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if self.proc is not None and self.proc.poll() is not None:
                raise SystemExit(f"ERROR: the bundle service exited ({self.proc.returncode}):\n{self.tail()}")
            try:
                status, body = get(self.url + "/health", timeout=5)
                if status == 200 and json.loads(body).get("ready"):
                    return json.loads(body)
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        raise SystemExit(f"ERROR: {self.url}/health not ready after {seconds:.0f} s\n{self.tail()}")

    def tail(self) -> str:
        return self.log.read_text(errors="replace")[-2000:] if self.log.is_file() else ""

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def identity_of(upstream: Upstream) -> tuple[dict, dict]:
    status, body = get(upstream.url + "/v1/info")
    if status != 200:
        raise SystemExit(f"ERROR: {upstream.url}/v1/info answered {status}")
    info = json.loads(body)
    ident = (info.get("serve") or {}).get("identity")
    if not ident:
        raise SystemExit("ERROR: the upstream's /v1/info carries no serve.identity")
    return info, ident


# ---- the hand-built page ------------------------------------------------------------------------
def legacy_page(ident: dict) -> tuple[str, dict]:
    """0d06aea's page rendered as pack.py rendered it (the same four placeholders, the same escaping)."""
    try:
        text = subprocess.run(["git", "-C", str(REPO), "show", f"{LEGACY_COMMIT}:{LEGACY_PATH}"],
                              check=True, capture_output=True, text=True).stdout
        blob = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{LEGACY_COMMIT}:{LEGACY_PATH}"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Unavailable(f"cannot read {LEGACY_COMMIT}:{LEGACY_PATH} from git: {exc}")
    for key, value in (("{{TITLE_HTML}}", html.escape(ident["title"])),
                       ("{{DESCRIPTION_HTML}}", html.escape(ident["description"])),
                       ("{{NOUNS}}", ident["nouns"]), ("{{NOUN}}", ident["noun"])):
        text = text.replace(key, value)
    return text, {"commit": LEGACY_COMMIT, "path": LEGACY_PATH, "blob": blob,
                  "rendered_digest": sha256(text.encode("utf-8"))}


def legacy_dir(page: str, workdir: Path) -> Path:
    """A template directory for the parity server: the hand-built page with the probe hooked in."""
    try:
        page = parity_browser.hook_probe(page)
    except ValueError as exc:
        raise SystemExit(f"ERROR: the hand-built page: {exc}")
    d = workdir / "legacy"
    (d / "js" / "core").mkdir(parents=True)
    (d / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(TEMPLATE / "js" / "core" / "probe.js", d / "js" / "core" / "probe.js")
    return d


# ---- one run: serve, drive, check clean ------------------------------------------------------------
def run_page(template: Path, upstream: Upstream, chrome: str, workdir: Path, name: str) -> tuple[dict, list[str]]:
    srv = parity_server.start(template, upstream.url)
    try:
        probe = parity_browser.drive(chrome, srv.url, workdir)
    finally:
        srv.stop()
    problems = clean(probe, srv.requests, srv.url)
    (workdir / f"probe-{name}.json").write_text(json.dumps(probe, indent=1, ensure_ascii=False), encoding="utf-8")
    return probe, problems


def clean(probe: dict, requests: list, origin: str) -> list[str]:
    out = []
    for url in probe.get("fetched") or []:       # (/favicon.ico: the browser asks for it, not the page)
        if isinstance(url, dict):                # vega/page-probe@1 (W15): {url relative to the page, status, schema}
            url = urljoin(origin + "/", url.get("url") or "")
        parts = urlsplit(url)
        if f"{parts.scheme}://{parts.netloc}" != origin or not (
                parts.path in ("/health", "/favicon.ico") or parts.path.startswith(("/v1/", "/showcase/"))):
            out.append(f"fetched outside health, v1/ and showcase/: {url}")
    if probe.get("error"):
        out.append(f"the probe failed: {probe['error']}")
    if not probe.get("ready"):
        out.append("the page did not boot (ready: false)")
    for vid, v in (probe.get("views") or {}).items():
        if not (v.get("reached") and v.get("rendered")):
            out.append(f"view {vid}: reached {v.get('reached')}, rendered {v.get('rendered')}")
    # vega/page-probe@1 states each error as "kind: text" (W15); the v0 probe wrote {kind, text}
    said = lambda e: f"({e.get('kind')}): {e.get('text')}" if isinstance(e, dict) else f"({e})"  # noqa: E731
    for e in probe.get("page_errors") or []:
        out.append(f"page error {said(e)}")
    for e in probe.get("late_errors") or []:
        out.append(f"page error after the walk {said(e)}")
    for line in probe.get("browser_console_errors") or []:
        out.append(f"browser console: {line}")
    for method, path, status in requests:
        if status >= 400:
            out.append(f"HTTP {status} for {method} {path}")
    facts = probe.get("parity") or {}
    missing = [k for k in COMPARE if not facts.get(k)]
    if missing:
        out.append(f"parity facts missing or empty: {', '.join(missing)}")
    return out


# ---- the comparison -----------------------------------------------------------------------------
def compared(facts: dict) -> dict:
    """The facts the golden pins, reduced to what is compared."""
    return {
        "guide": [{"n": s.get("n"), "t": s.get("t")} for s in facts.get("guide") or []],
        "choices": facts.get("choices"),
        "experiments": facts.get("experiments"),
        "weights_map": facts.get("weights_map"),
        "stepper": [{"id": p.get("id"), "z": [s.get("z") for s in p.get("steps") or []]}
                    for p in facts.get("stepper") or []],
    }


def diff(want, got, path: str = "", out: list | None = None, limit: int = 40) -> list[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(want, dict) and isinstance(got, dict):
        for k in sorted(set(want) | set(got)):
            diff(want.get(k), got.get(k), f"{path}.{k}" if path else k, out, limit)
    elif isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            out.append(f"{path}: {len(want)} entries in the golden, {len(got)} on the page")
        for i, (a, b) in enumerate(zip(want, got)):
            diff(a, b, f"{path}[{i}]", out, limit)
    elif want != got:
        out.append(f"{path}: golden {want!r}, page {got!r}")
    return out


def summary(facts: dict) -> str:
    wm = facts.get("weights_map") or {}
    cells = sum(len(r["cells"]) for m in wm.values() if m for r in m["rows"])
    zs = sum(1 for p in facts.get("stepper") or [] for z in p["z"] if z is not None)
    return (f"{len(facts.get('guide') or [])} guide texts, {facts.get('choices')} decisions, "
            f"{(facts.get('experiments') or {}).get('count')} variants, {cells} map cells in "
            f"{len(wm)} readings, {zs} stepper z over {len(facts.get('stepper') or [])} presets")


# ---- the two modes ------------------------------------------------------------------------------
def write_golden(args, upstream: Upstream, chrome: str, workdir: Path) -> int:
    info, ident = identity_of(upstream)
    page, source = legacy_page(ident)
    status, served = get(upstream.url + "/")
    source["served_equal"] = status == 200 and served == page.encode("utf-8")
    if not source["served_equal"]:
        say("WARN", "the bundle's own page differs from 0d06aea rendered with its identity; the golden is 0d06aea's")
    probe, problems = run_page(legacy_dir(page, workdir), upstream, chrome, workdir, "legacy")
    for p in problems:
        say("FAIL", f"hand-built page: {p}")
    if problems:
        return 1
    facts = compared(probe["parity"])
    golden = {
        "schema": "vega/page-parity@1",
        "about": "Facts of the hand-built page (0d06aea) on the CIC R7 bundle that the page template must reproduce "
                 "(W01 closing test; vega-show/scripts/tests/capture_parity_golden.py --check).",
        "source": source,
        "upstream": {k: info.get(k) for k in ("slug", "run", "model_version")},
        "captured": {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "chrome": parity_browser.chrome_version(chrome), "probe": probe["probe"],
                     "viewport": parity_browser.VIEWPORT.replace(",", "x"), "reduced_motion": True},
        "compare": list(COMPARE),
        "facts": facts,
    }
    args.golden.parent.mkdir(parents=True, exist_ok=True)
    args.golden.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    say("OK", f"golden written to {args.golden}: {summary(facts)}")
    return 0


def check(args, upstream: Upstream, chrome: str, workdir: Path) -> int:
    if not args.golden.is_file():
        raise SystemExit(f"ERROR: no golden at {args.golden} (capture it with --write)")
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    info, _ = identity_of(upstream)
    if info.get("model_version") != golden["upstream"]["model_version"]:
        raise SystemExit(f"ERROR: the upstream serves {info.get('model_version')}, the golden was captured on "
                         f"{golden['upstream']['model_version']}")
    probe, problems = run_page(args.template, upstream, chrome, workdir, "template")
    for p in problems:
        say("FAIL", f"template: {p}")
    facts = compared(probe.get("parity") or {})
    diffs = [] if problems else diff({k: golden["facts"][k] for k in golden["compare"]},
                                     {k: facts[k] for k in golden["compare"]})
    for d in diffs:
        say("FAIL", f"differs from the golden: {d}")
    if problems or diffs:
        say("FAIL", f"parity probe on {golden['upstream'].get('run')}: {len(problems)} run problem(s), "
                    f"{len(diffs)} difference(s) (probe output kept in {workdir})")
        return 1
    say("OK", f"parity with {golden['source']['commit']} on {golden['upstream'].get('run')}: {summary(facts)}; "
              f"{len(probe['views'])} views rendered, no page error, no HTTP error")
    return 0


def _parse(argv) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="capture the golden from 0d06aea's page")
    mode.add_argument("--check", action="store_true", help="compare the template with the golden (default)")
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--upstream", default=None, help="R7's service, when it runs already")
    ap.add_argument("--bundle", type=Path, default=BUNDLE,
                    help="R7's bundle, started when --upstream is absent (default: $CIDS_ROOT/bundle/cicids2017-label-R7)")
    ap.add_argument("--python", default=sys.executable, help="the interpreter that runs the bundle's service")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the work directory (probe outputs, service log)")
    ap.add_argument("--strict", action="store_true",
                    help="a missing browser, bundle or CIDS_ROOT is an error, not a WARN")
    return ap.parse_args(argv)


def _run(a: argparse.Namespace) -> int:
    workdir = Path(tempfile.mkdtemp(prefix="vega-parity-"))
    upstream = None
    try:
        chrome = parity_browser.find_chrome(a.chrome)
        if not chrome:
            raise Unavailable(f"no headless browser (checked $CHROME, {', '.join(parity_browser.CHROME_CANDIDATES)})")
        upstream = Upstream(a.upstream, a.bundle, a.python, workdir)
        upstream.wait_ready()
        rc = write_golden(a, upstream, chrome, workdir) if a.write else check(a, upstream, chrome, workdir)
        a.keep = a.keep or rc != 0
        return rc
    except Unavailable as exc:
        say("ERROR" if a.strict else "WARN", f"{exc}; parity not checked")
        return 2 if a.strict else 0
    finally:
        if upstream is not None:
            upstream.stop()
        if a.keep:
            say("INFO", f"work directory kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None) -> int:
    return _run(_parse(argv))


if __name__ == "__main__":
    sys.exit(main())
