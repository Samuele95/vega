#!/usr/bin/env python3
"""
smoke_page.py — the page driver (BUILD-PLAN R-H3, W15; D4 §17.5): the showcase page opened in a
headless Chrome (`cdp.py`), walked by its own probe (`?probe=all`, js/core/probe.js) and held to
the page's checks. One driver for the fixture matrix, a bundle's served showcase (`smoke_show.py
--serve`) and the lanes' page stage (`vega/scripts/tests/page_smoke.py`, which runs it four times
with the leak checks).

    python3 smoke_page.py --fixture CASE|all [--fixtures DIR] [--jobs N] [--strict] [--json OUT]
    python3 smoke_page.py --bundle B [--showcase B/showcase] [--strict] [--json OUT]
        --fixture only: [--template DIR] (another page template, e.g. a patched copy)
        common: [--mode all|views] [--locale en|qps-ploc] [--viewport 1440x900] [--mobile]
                [--motion reduced|full] [--chrome EXE] [--shots DIR] [--work DIR]

`--fixture` serves each case of the page fixture matrix with `tests/serve_page_fixture.py`; the
cases are built first by `tests/make_page_fixtures.py` into the work directory (about 15 s for
all 35) unless `--fixtures` names a directory that holds them. `--bundle` starts the bundle's own
service (`python -m serve.app` in the bundle, a clean environment, CPU) with its showcase slot,
which must report itself verified (`/v1/showcase`). Either way the page is opened at
`/?probe=<mode>` and its output read back once written, then again after a pause, for the errors
heard after the walk.

The checks (each a named result; D4 §17.5, D6 §10.4's page-level ones, W10's per-segment ink, W12's
latent readings):

    P-PROBE   the probe ran: ready, no error of its own, its output vega/page-probe@1 (validated
              with jsonschema; without it a `WARN:`, and a failure under --strict)
    P-ERR     no page error before or after the walk, no exception, console error or error log
              heard by the browser, no HTTP status >= 400 but a 404 where one is expected (a
              favicon; the monitor route of a bundle that ships none); a fixture case asked nothing
              it cannot answer (the monitor of a case that ships one included)
    P-VIS     every view of the tour reached and rendered (a view off the tour has its reason); the
              network view's two disclosures (what was tried, the weights) rendered when offered
    P-TXT     no unresolved placeholder, no missing strings key, no `undefined`/`NaN`/`null`/
              `[object Object]` and no placeholder token (`{row.many}`) in a view's visible text
              (the context's last step and the results' limits and breakdown opened), no bar's
              value cut off at the viewport's width, and the results view saying the verdict the
              story records (earned, not earned, beaten, not assessable)
    P-LANE    the lane the page read equals the one expected (the matrix's, or page.json's)
    P-MOTIF   the motif on the field (`data-motif`) is page.json's
    P-ROLE    with no alert role: no `.role-alert` element and no alert-tinted field mark
    P-NET     the network drawn: every segment it laid out has ink in every drawing mode (the
              whole chain when drawn whole), the drawing raised no warning, and the "every weight"
              switch is offered whenever a stage kind's renderer declares that level
    P-AXIS    the weights map on real rows has the lane's weights axis for columns: the binary
              sides, the classes, labels or levels, the steps ahead, the target's bins (ascending,
              contiguous), the strata of a reconstruction or anomaly reading (/v1/schema's output
              and the showcase's attribution)
    P-STEP    every preset walked answered, its explanation exact or degraded only as declared,
              every stepper step within max(1e-3, 1e-3·|z|) of the network, the output step's
              decision (the labels at or above their thresholds, a binary's side) the answer's own,
              the attribution's method named and a completeness gap it states shown, no check the
              page could not make, no explainer that fell back to the generic one
    P-LATENT  the latent families' readings: an autoencoder's or PCA floor's waist marked and
              inked; a VAE's code drawn with its μ, σ (> 0) and KL on an answer; a deep SVDD's
              exported centre drawn and one answer's distance recomputed within 1e-4·max(1, |d|)
    P-MANY    one many-at-once source answered rows, and the drift readings the service answered
              (POST /v1/monitor, 200) are shown
    P-CODE    the code walk read, every step shown, none broken
    P-FETCH   every URL fetched same-origin, under health, v1/ and showcase/
    P-THEME   the theme control flips the theme and back
    P-LINKS   every report link of the code notes answers 200

Without a browser: `WARN:` and exit 0, or exit 2 under `--strict` (as smoke_bundle.py). Exit 0:
every check passed. 1: a check failed (each printed `FAIL:`). 2: a case could not be run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
TESTS = HERE / "tests"
sys.path.insert(0, str(HERE))

import cdp  # noqa: E402
from smoke_page_checks import judge  # noqa: E402

SCHEMA = "vega/page-smoke@1"
MATRIX = TESTS / "page-matrix.json"
PROBE_S, LATE_S, READY_S = 300, 2.0, 300
PROBE_OUT = "(() => { const n = document.getElementById('showcase-probe'); return n && n.textContent ? n.textContent : null; })()"


class Unrunnable(Exception):
    """A case, the service or the browser could not run (exit 2)."""


# ============================================================================ the targets

@dataclass(frozen=True)
class Served:
    """A page served at `url`: a fixture case or a bundle's service; `expect` its lane, `misses`
    what a fixture server could not answer, `allowed` the path prefixes the page may meet answered
    404, `documents` the service's /v1/schema and the showcase's attribution ({schema,
    attribution}), `strict` the run's --strict."""

    name: str
    url: str
    expect: dict
    misses: Callable = lambda: []
    allowed: tuple = ()
    documents: Callable = lambda: {}
    strict: bool = False


@contextlib.contextmanager
def fixture_target(case_dir: Path, expect: dict, template: Path | None = None):
    sys.path.insert(0, str(TESTS))
    import serve_page_fixture
    srv = serve_page_fixture.start(case_dir, template)
    monitor = (case_dir / serve_page_fixture.MONITOR_FILE).is_file()
    docs = lambda: {"schema": srv.case.read("schema.json"), "attribution": srv.case.read("attribution.json")}  # noqa: E731
    try:
        yield Served(case_dir.name, srv.url, expect, lambda: srv.misses,
                     ("/favicon.ico",) + (() if monitor else ("/v1/monitor",)), docs)
    finally:
        srv.stop()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _get_or_none(url: str):
    """A document the service may not serve (404): None."""
    try:
        return _get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


@contextlib.contextmanager
def bundle_target(bundle: Path):
    """The bundle's own service on a free port, until it is ready and its slot verified."""
    port, env = _free_port(), {"PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
                               "HOME": "/tmp", "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "",
                               "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    log = open(bundle.parent / f"{bundle.name}.smoke-page.log", "wb")
    proc = subprocess.Popen([sys.executable, "-m", "serve.app", "--port", str(port), "--torch-threads", "2"],
                            cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        _until_ready(proc, base, log.name)
        slot = _get(base + "/v1/showcase")
        if not slot.get("verified"):
            raise Unrunnable(f"the showcase slot is not verified: {slot.get('reason')}")
        page = _get(base + "/showcase/page.json")
        monitor = bool((json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8")).get("monitor") or {}))
        docs = {"schema": _get(base + "/v1/schema"), "attribution": _get_or_none(base + "/showcase/data/attribution.json")}
        yield Served(bundle.name, base, page.get("lane") or {}, lambda: [],
                     ("/favicon.ico",) + (() if monitor else ("/v1/monitor",)), lambda: docs)
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def _until_ready(proc, base: str, log: str) -> None:
    deadline = time.time() + READY_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise Unrunnable(f"the service exited {proc.returncode} (log {log})")
        with contextlib.suppress(OSError, ValueError):
            if _get(base + "/health").get("ready"):
                return
        time.sleep(0.5)
    raise Unrunnable(f"the service was not ready in {READY_S} s (log {log})")


# ============================================================================ the drive

def drive(chrome: str, served: Served, opts) -> dict:
    """The probe's output, what the browser heard, the responses, the fixture's misses."""
    w, h = (int(v) for v in opts.viewport.lower().split("x"))
    query = f"?probe={opts.mode}" + (f"&locale={opts.locale}" if opts.locale != "en" else "")
    with cdp.Browser(chrome, viewport=(w, h)) as browser:
        page = browser.page(viewport=(w, h), mobile=opts.mobile, reduced_motion=opts.motion == "reduced")
        page.goto(served.url + "/" + query)
        raw = page.wait(PROBE_OUT, PROBE_S, every=0.5)
        if not raw:
            raise Unrunnable(f"{served.name}: the probe wrote nothing in {PROBE_S} s; the page said {page.problems()[:5]}")
        time.sleep(LATE_S)
        probe = json.loads(page.eval(PROBE_OUT) or raw)
        if opts.shots:
            page.screenshot(Path(opts.shots) / f"{served.name}-{opts.mode}.png")
        return {"probe": probe, "console": page.problem_records(), "responses": page.responses(), "misses": served.misses()}


# ============================================================================ the runs

def run_one(chrome: str, ctx, opts) -> dict:
    with ctx as served:
        served = dataclasses.replace(served, strict=bool(opts.strict))
        result = drive(chrome, served, opts)
        checks = judge(result, served)
    p = result["probe"]
    return {"name": served.name, "url": served.url, "checks": checks, "ok": all(c["ok"] for c in checks),
            "summary": {"views": sorted(k for k, v in (p.get("views") or {}).items() if v.get("rendered")),
                        "presets": ((p.get("sub_views") or {}).get("stepper") or {}).get("presets"),
                        "max_gap": (p.get("stepper") or {}).get("max_gap"), "segments": (p.get("network") or {}).get("segments")},
            "probe": p}


def fixture_cases(opts) -> list[tuple[Path, dict]]:
    root = Path(opts.fixtures) if opts.fixtures else Path(opts.work) / "fixtures"
    if not opts.fixtures:
        sys.path.insert(0, str(TESTS))
        import make_page_fixtures
        if make_page_fixtures.main(["--out", str(root)] + ([] if opts.fixture == "all" else ["--cases", opts.fixture])) != 0:
            raise Unrunnable("make_page_fixtures.py could not build every case")
    expect = {c["id"]: c["expect"] for c in json.loads(MATRIX.read_text(encoding="utf-8"))["cases"]}
    found = sorted((d for d in root.iterdir() if (d / "case.json").is_file()), key=lambda d: d.name)
    chosen = [d for d in found if opts.fixture == "all" or opts.fixture in (d.name, d.name.split("-")[0].lstrip("0"))]
    if not chosen:
        raise Unrunnable(f"no fixture case {opts.fixture!r} under {root}")
    return [(d, expect.get(d.name, {})) for d in chosen]


def report(results: list[dict]) -> int:
    code = 0
    for r in results:
        if r.get("error"):
            print(f"ERROR: {r['name']}: {r['error']}")
            code = max(code, 2)
            continue
        for c in r["checks"]:
            for p in c["problems"]:
                print(f"FAIL: {r['name']}: {c['id']}: {p}")
        if r["ok"]:
            s = r["summary"]
            print(f"OK: {r['name']}: {len(r['checks'])} checks; views {', '.join(s['views'])}; {s['presets']} preset(s), "
                  f"max stepper gap {s['max_gap']}, {s['segments']} segment(s) inked")
        else:
            code = max(code, 1)
    return code


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixture", help="a case of the page fixture matrix (id or number), or all")
    src.add_argument("--bundle", help="a bundle whose service serves its showcase")
    ap.add_argument("--showcase", help="the bundle's showcase (it must be <bundle>/showcase: the slot serves that one)")
    ap.add_argument("--fixtures", help="where make_page_fixtures.py wrote the cases (default: build them into --work)")
    ap.add_argument("--template", help="the page template the fixtures are served with (default: the installed one)")
    ap.add_argument("--work", help="the work directory (default: a temporary one)")
    ap.add_argument("--jobs", type=int, default=3, help="fixture cases driven at once (default 3)")
    ap.add_argument("--mode", choices=("all", "views"), default="all")
    ap.add_argument("--locale", choices=("en", "qps-ploc"), default="en")
    ap.add_argument("--viewport", default="1440x900")
    ap.add_argument("--mobile", action="store_true")
    ap.add_argument("--motion", choices=("reduced", "full"), default="reduced")
    ap.add_argument("--chrome")
    ap.add_argument("--shots", help="write a PNG of each page here")
    ap.add_argument("--json", help="write the results (vega/page-smoke@1) here")
    ap.add_argument("--strict", action="store_true", help="no browser is an error (exit 2), not a WARN")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    chrome = cdp.find_chrome(a.chrome)
    if not chrome:
        print(f"{'ERROR' if a.strict else 'WARN'}: no headless browser (checked --chrome, $CHROME, {', '.join(cdp.CHROME_CANDIDATES)}); the page was not driven")
        return 2 if a.strict else 0
    a.work = a.work or tempfile.mkdtemp(prefix="smoke-page-")
    Path(a.work).mkdir(parents=True, exist_ok=True)
    if a.shots:
        Path(a.shots).mkdir(parents=True, exist_ok=True)
    try:
        results = run_all(chrome, a)
    except Unrunnable as exc:
        print(f"ERROR: {exc}")
        return 2
    if a.json:
        write_json(Path(a.json), chrome, a, results)
    return report(results)


def write_json(path: Path, chrome: str, a, results: list[dict]) -> None:
    doc = {"schema": SCHEMA, "chrome": chrome, "mode": a.mode, "locale": a.locale, "viewport": a.viewport, "results": results}
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def run_all(chrome: str, a) -> list[dict]:
    if a.bundle:
        bundle = Path(a.bundle).resolve()
        if a.showcase and Path(a.showcase).resolve() != bundle / "showcase":
            raise Unrunnable(f"--showcase {a.showcase} is not {bundle / 'showcase'}: the slot serves the bundle's own")
        return [_guarded(chrome, bundle.name, lambda: bundle_target(bundle), a)]
    cases = fixture_cases(a)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        template = Path(a.template) if a.template else None
        futures = [pool.submit(_guarded, chrome, d.name, (lambda d=d, e=e: fixture_target(d, e, template)), a) for d, e in cases]
        return [f.result() for f in futures]


def _guarded(chrome: str, name: str, make, a) -> dict:
    try:
        return run_one(chrome, make(), a)
    except (Unrunnable, cdp.CdpError, OSError) as exc:
        return {"name": name, "error": f"{type(exc).__name__}: {exc}", "ok": False, "checks": []}


if __name__ == "__main__":
    sys.exit(main())
