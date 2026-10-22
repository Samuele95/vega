#!/usr/bin/env python3
"""
smoke_show.py — the stage's smokes (D5 §7.8, §9.2; contract §10 SV1–SV5; BUILD-PLAN R-H3).

    python3 smoke_show.py OUT.html [--strict] [--chrome EXE] [--json]
    python3 smoke_show.py --serve B [--strict] [--python PY] [--port N] [--chrome EXE] [--json]
    python3 smoke_show.py --mock FIXTURE_DIR --showcase S [--strict] [--chrome EXE] [--json]

`OUT.html` — the model page with its `showcase` layer: the *Showcase* tab opened in a headless
Chrome and held to its nine checks (`smoke_show_layer.py`).

`--serve B` — the served walk: the bundle's own service (`<python> -B -m serve.app`, the skills
off its path, CPU) with its showcase slot, `/v1/showcase` verified, every served file and every
report or brief link at its pinned digest, every view walked by the page's probe through the one
page driver (`smoke_page.py`), no page error, no unresolved placeholder, the alert colour exactly
when a class role is `alert`, the motif page.json names (`serve_probe.py`, gates SV1–SV5). It
writes `B/showcase/.smoke/smoke.json` (`vega/show-smoke@1`) and the screenshots `start.png` and
`results.png` beside it; `apply_show_layer.py` reads the walk from there.

`--mock FIXTURE_DIR --showcase S` — the coverage matrix's walk without a service: a case of the
page fixture matrix (a directory `tests/make_page_fixtures.py` wrote, I10's recorded payloads)
served by `tests/serve_page_fixture.py` with the showcase's `story.json`, `page.json` and
attribution in place of the case's own, walked and judged by the page driver.

Without a browser (or, for `--serve`, without the packages the service runs on): `WARN: served
page not checked` and exit 0 — `--serve` records `status: not-run` with the reason — or exit 2
under `--strict`. Exit 0: every check passed. 1: a check failed (each printed `FAIL:`). 2: the
inputs could not be read, or the walk could not run under `--strict`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cdp  # noqa: E402
from vega_show_core import ShowPaths, read_json, write_json  # noqa: E402

MOCK_FILES = (("story.json", "story.json"), ("page.json", "page.json"),
              ("attribution.json", "data/attribution.json"))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("out", nargs="?", help="the model page carrying the showcase layer")
    mode.add_argument("--serve", metavar="B", help="a bundle whose service serves its showcase")
    mode.add_argument("--mock", metavar="FIXTURE_DIR", help="a page fixture case (make_page_fixtures.py)")
    ap.add_argument("--showcase", help="with --mock: the showcase whose story and page are walked")
    ap.add_argument("--strict", action="store_true", help="a walk that cannot run is an error (exit 2)")
    ap.add_argument("--python", default=sys.executable, help="the interpreter the service runs on")
    ap.add_argument("--port", type=int, help="the service's port (default: a free one)")
    ap.add_argument("--chrome", help="the headless browser (default: $CHROME, then the usual names)")
    ap.add_argument("--json", action="store_true", help="print the result document")
    return ap


def _report(checks: list[dict], what: str) -> int:
    for c in checks:
        for p in c["problems"]:
            print(f"FAIL: {what}: {c['id']}: {p}")
    if all(c["ok"] for c in checks):
        print(f"OK: {what}: {len(checks)} check(s) passed")
        return 0
    return 1


def _no_browser(a, why: str) -> int:
    level = "ERROR" if a.strict else "WARN"
    print(f"{level}: served page not checked — {why}")
    return 2 if a.strict else 0


# ------------------------------------------------------------------------------- OUT.html
def run_layer(a, chrome: str) -> int:
    import smoke_show_layer
    out = Path(a.out)
    if not out.is_file():
        print(f"ERROR: {out} not found")
        return 2
    checks = smoke_show_layer.check_layer(chrome, out)
    if a.json:
        print(json.dumps({"schema": "vega/show-layer-smoke@1", "file": str(out), "checks": checks}, indent=1))
    return _report(checks, out.name)


# ------------------------------------------------------------------------------- --serve
# what a re-walk changes without changing what the walk found
WALK_CIRCUMSTANCE = ("checked_at", "probe_digest", "service_log")


def _recorded_walk(paths, doc) -> dict | None:
    """The walk already recorded, when this one found the same thing on the same showcase: a
    re-walk is a reading, and the stage brief pins `.smoke/smoke.json` by digest — the
    orchestrator's own `--serve` after the brief must not move it."""
    old, _ = read_json(paths.smoke / "smoke.json", "smoke.json")
    if not isinstance(old, dict):
        return None
    found = [{k: v for k, v in d.items() if k not in WALK_CIRCUMSTANCE} for d in (old, doc)]
    return old if found[0] == found[1] else None


def run_serve(a, chrome: str | None) -> int:
    import serve_probe
    paths = ShowPaths.of(Path(a.serve).resolve())
    if not paths.manifest.is_file():
        print(f"ERROR: no built showcase at {paths.showcase} (SHOWCASE.json) — run build_showcase.py first")
        return 2
    try:
        doc = serve_probe.ServedPageProbe(paths.bundle, a.python, a.port, chrome).run()
    except serve_probe.NotRun as exc:
        doc = serve_probe.not_run_record(paths.bundle, str(exc))
    except (OSError, ValueError, cdp.CdpError) as exc:
        print(f"ERROR: the served walk could not run: {type(exc).__name__}: {exc}")
        return 2
    doc = _recorded_walk(paths, doc) or doc
    write_json(paths.smoke / "smoke.json", doc)
    if a.json:
        print(json.dumps(doc, indent=1, ensure_ascii=False))
    if doc["status"] == "not-run":
        return _no_browser(a, f"the served walk did not run: {doc['why_not_run']}")
    code = _report(doc["checks"], paths.bundle.name)
    print(f"{'OK' if code == 0 else 'ERROR'}: served walk {doc['status']}: views {', '.join(doc['views'])}; "
          f"{doc['page_errors']} page error(s); {len(doc['placeholders_unresolved'])} unresolved placeholder(s); "
          f"{len(doc['links'])} link(s); {paths.smoke / 'smoke.json'}")
    return code


# ------------------------------------------------------------------------------- --mock
def mock_case(case: Path, showcase: Path, work: Path) -> Path:
    """The fixture case with the showcase's story, page config and attribution in its place."""
    target = work / case.name
    shutil.copytree(case, target)
    for name, rel in MOCK_FILES:
        src = showcase / rel
        if src.is_file():
            shutil.copyfile(src, target / name)
        elif name == "attribution.json":
            (target / name).unlink(missing_ok=True)
        else:
            raise FileNotFoundError(f"the showcase has no {rel}")
    if (showcase / "briefs").is_dir():
        shutil.rmtree(target / "briefs", ignore_errors=True)
        shutil.copytree(showcase / "briefs", target / "briefs")
    return target


def run_mock(a, chrome: str) -> int:
    import smoke_page
    from smoke_page_checks import judge
    case, showcase = Path(a.mock), Path(a.showcase or "")
    if not (case / "case.json").is_file() or not (showcase / "story.json").is_file():
        print("ERROR: --mock needs a fixture case (case.json) and --showcase a built showcase (story.json)")
        return 2
    page = json.loads((showcase / "page.json").read_text(encoding="utf-8"))
    opts = types.SimpleNamespace(viewport="1440x900", mode="all", locale="en", mobile=False, motion="reduced", shots=None)
    with tempfile.TemporaryDirectory(prefix="smoke-show-mock-") as tmp:
        merged = mock_case(case, showcase, Path(tmp))
        with smoke_page.fixture_target(merged, page.get("lane") or {}) as served:
            served = smoke_page.dataclasses.replace(served, strict=bool(a.strict))
            result = smoke_page.drive(chrome, served, opts)
            checks = judge(result, served)
    if a.json:
        print(json.dumps({"schema": "vega/show-mock@1", "case": case.name, "checks": checks}, indent=1))
    return _report(checks, f"{case.name} with {showcase}")


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.mock and not a.showcase:
        print("ERROR: --mock needs --showcase S")
        return 2
    chrome = cdp.find_chrome(a.chrome)
    if a.serve:
        return run_serve(a, chrome)
    if not chrome:
        return _no_browser(a, f"no headless browser (checked --chrome, $CHROME, {', '.join(cdp.CHROME_CANDIDATES)})")
    try:
        return run_mock(a, chrome) if a.mock else run_layer(a, chrome)
    except (OSError, ValueError, cdp.CdpError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())
