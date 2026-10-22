#!/usr/bin/env python3
"""
test_coverage_matrix.py — complete coverage, through the stage (D5 §12.1 step 6; BUILD-PLAN H09).

    python3 -m pytest -q vega-show/scripts/tests/test_coverage_matrix.py
    python3 test_coverage_matrix.py [--jobs N] [--cells SUBSTRING] [--fixtures DIR | --build-fixtures]
    VEGA_SHOW_FIXTURES=<dir> python3 -m pytest -q …/test_coverage_matrix.py      # with the mock walks

**The build.** Every admitted (family, task, sample shape, goal) cell of the vega-forge catalog —
`coverage_fixtures.cells()`, the catalog's `admission` rows, the closed definition of complete
coverage — is packed as a coverage-fixture bundle and taken through the stage as the lanes take a
real one: `extract_facts` → `draft_plan --unattended` → `autoauthor` → `resolve_story` →
`build_showcase --unattended` → `build_showcase --check`. Each cell must build (exit 0) and check
`pass`, its `page.json` lane must be the cell's (shape, graph level) with the view key and link
the catalogs derive for its task, and no refusal may be swallowed: a cell that fails says the
step, its exit and its last lines. The reconstruction and anomaly cells read their row scores
from the demonstration sample (`page.json.reference`), so their bundle carries one: a demo
sample in `examples/demo.json`, a `predict.py` whose model answers a row score, and the serving
contract's `tables.frame` (`attribution_build.py`'s `reference_scores`); no cell ships an
inspector, so no cell has an attribution (the build's WARN), which is the coverage bundle's limit,
not the stage's.

**The mock walks.** With the page fixtures present (`--fixtures DIR` or `$VEGA_SHOW_FIXTURES`, a
directory `make_page_fixtures.py` wrote; `--build-fixtures` builds them), every fixture case is
walked by `smoke_show.py --mock CASE --showcase S --strict`, `S` the case's own story, page and
attribution (a showcase built from a coverage cell is not walked over a fixture: its network is not
the fixture's, which the walk's P-NET, P-STEP and P-CODE rightly refuse). And every cell's lane —
(view key, link, shape, graph level) — must be one a walked case has: the walks span what the
build spans.

Exit 0 (the script) when every cell built and checked and, with fixtures, every walk passed and
every lane was walked.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
sys.path[:0] = [str(TESTS), str(SCRIPTS)]

import coverage_fixtures as cf  # noqa: E402

JOBS = min(8, os.cpu_count() or 2)
SCORED = {"unsupervised": "reconstruction", "anomaly": "anomaly-reconstruction"}
FAKE_PREDICT = '''"""A coverage-fixture predictor: verifies nothing, answers a row score, ships no inspection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def verify(bundle, roles=()):
    return None


class _Model:
    def predict(self, rows, history=None):
        return {"score": [0.1 * (i + 1) for i in range(len(rows))]}


def load(bundle):
    return _Model()


def load_inspector(bundle):
    raise LookupError("the coverage fixture ships no inspection")


if __name__ == "__main__":
    sys.exit(0 if "--check" in sys.argv else 2)
'''
FAKE_CONTRACT = '''"""The serving contract's request frame, as the coverage fixture needs it: rows as sent."""


class _Tables:
    @staticmethod
    def frame(rows):
        return list(rows)


class ApiContract:
    tables = _Tables()

    @classmethod
    def from_bundle(cls, bundle):
        return cls()
'''


def _reading(cell: dict) -> str | None:
    kind = cell["task"].partition("/")[0]
    if kind == "anomaly" and cell["family"] == "deep-svdd":
        return "anomaly-svdd"
    return SCORED.get(kind)


def cell_docs(cell: dict) -> dict:
    """The coverage bundle of a cell, its service declaring what the page requires, and (scored
    readings) a demonstration sample whose rows the predictor scores."""
    import page_template
    docs = cf.cell_docs(cell)
    serve = docs["MANIFEST.json"]["serve"]
    serve["endpoints"] = sorted(set(page_template.REQUIRED_ENDPOINTS)
                                | {"GET /", "GET /v1/showcase", "GET /showcase/{path:path}"})
    reading = _reading(cell)
    if reading:
        docs["MANIFEST.json"]["output"] = {**docs["MANIFEST.json"]["output"], "reading": reading,
                                           "field_kinds": {"score": {"kind": "row_score"}}}
        rows = [{name: (0.1 * i if role == "numeric" else "a") for name, role in cf.FEATURES} for i in range(6)]
        docs["examples/demo.json"] = {"strata": [{"id": "all", "kind": "group", "name": "all"}],
                                      "items": [{"stratum": "all", "rows": [row]} for row in rows]}
    return docs


def _run(main, argv, cwd: Path) -> tuple[int, str]:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed), contextlib.chdir(cwd):
        code = main([str(a) for a in argv])
    return code, printed.getvalue()


def _bundle(root: Path, cell: dict) -> tuple[Path, Path]:
    model, bundle = cf.write_bundle(root, cell_docs(cell))
    (bundle / "predict.py").write_text(FAKE_PREDICT, encoding="utf-8")
    (bundle / "serve").mkdir(exist_ok=True)
    (bundle / "serve" / "__init__.py").write_text("", encoding="utf-8")
    (bundle / "serve" / "contract.py").write_text(FAKE_CONTRACT, encoding="utf-8")
    for path in (bundle / "docs" / "pack-brief.md", root / "vega-runs/demo/docs/train-brief.md"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nstage: {path.stem.split('-')[0]}\n---\n# brief\n", encoding="utf-8")
    return model, bundle


def run_cell(cell: dict) -> tuple[list[str], dict | None]:
    """(what is wrong with the cell through the stage, its page.json lane)."""
    import autoauthor, build_showcase, draft_plan, extract_facts, resolve_story
    with tempfile.TemporaryDirectory(prefix="coverage-cell-") as tmp:
        root = Path(tmp) / "p"
        root.mkdir()
        model, bundle = _bundle(root, cell)
        plan, showcase = root / "plan", bundle / "showcase"
        steps = (("extract", extract_facts.main, ["--model", model, "--bundle", bundle, "--out", plan / "facts.json"]),
                 ("draft", draft_plan.main, ["--facts", plan / "facts.json", "--out", plan / "draft.json", "--unattended"]),
                 ("author", autoauthor.main, ["--plan", plan / "draft.json", "--facts", plan / "facts.json",
                                              "--out", plan / "story-plan.json"]),
                 ("resolve", resolve_story.main, ["--plan", plan / "story-plan.json", "--facts", plan / "facts.json",
                                                  "--out", plan / "story.json"]),
                 ("build", build_showcase.main, ["--bundle", bundle, "--model", model, "--plan", plan / "story-plan.json",
                                                 "--story", plan / "story.json", "--facts", plan / "facts.json",
                                                 "--out", showcase, "--unattended"]),
                 ("check", build_showcase.main, ["--check", showcase]))
        for name, main, argv in steps:
            code, printed = _run(main, argv, root)
            if code != 0:
                return [f"{name} exited {code}: {' | '.join(printed.strip().splitlines()[-3:])}"], None
        facts = json.loads((plan / "facts.json").read_text(encoding="utf-8"))
        lane = json.loads((showcase / "page.json").read_text(encoding="utf-8"))["lane"]
    want = {"shape": cell["shape"], "level": cell["level"], "view_key": facts["task"]["view_key"]}
    bad = [f"page.json lane.{k} {lane.get(k)!r} ≠ {v!r}" for k, v in want.items() if lane.get(k) != v]
    return bad, lane


def run_cells(cells: list[dict], jobs: int = JOBS) -> dict:
    """{cell label: (problems, lane)} over every cell, `jobs` processes at once."""
    labels = [label(c) for c in cells]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        return dict(zip(labels, pool.map(run_cell, cells)))


def label(cell: dict) -> str:
    return "/".join(str(cell[k]) for k in ("family", "shape", "level", "task", "variant") if cell[k] is not None)


def lane_key(lane: dict) -> tuple:
    return lane.get("view_key"), lane.get("link"), lane.get("shape"), lane.get("level")


# =============================================================================== the mock walks

def fixture_cases(fixtures: Path) -> list[Path]:
    return sorted(d for d in Path(fixtures).iterdir() if (d / "case.json").is_file())


def mock_walk(case: Path, scratch: Path) -> tuple[int, str]:
    """`smoke_show.py --mock` of a fixture case with the case's own showcase."""
    showcase = scratch / case.name / "showcase"
    (showcase / "data").mkdir(parents=True)
    for name, rel in (("story.json", "story.json"), ("page.json", "page.json"), ("attribution.json", "data/attribution.json")):
        if (case / name).is_file():
            shutil.copyfile(case / name, showcase / rel)
    done = subprocess.run([sys.executable, str(SCRIPTS / "smoke_show.py"), "--mock", str(case), "--showcase",
                           str(showcase), "--strict"], capture_output=True, text=True, timeout=900)
    return done.returncode, done.stdout + done.stderr


def walk_all(fixtures: Path, jobs: int = 3) -> dict:
    """{case name: (exit, output)}; and every case's lane, read from its case.json."""
    with tempfile.TemporaryDirectory(prefix="coverage-mock-") as tmp, \
            concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        cases = fixture_cases(fixtures)
        return dict(zip((c.name for c in cases), pool.map(lambda c: mock_walk(c, Path(tmp)), cases)))


def walked_lanes(fixtures: Path) -> set:
    out = set()
    for case in fixture_cases(fixtures):
        lane = json.loads((case / "page.json").read_text(encoding="utf-8"))["lane"]
        out.add(lane_key(lane))
    return out


def build_fixtures(out: Path) -> Path:
    import make_page_fixtures
    if make_page_fixtures.main(["--out", str(out)]) != 0:
        raise RuntimeError("make_page_fixtures.py failed")
    return out


# =============================================================================== pytest

@pytest.fixture(scope="module")
def results() -> dict:
    return run_cells(cf.cells())


def test_every_admitted_cell_builds_and_checks(results):
    assert len(results) == len(cf.cells()) and results, "no admitted cell"
    failed = {k: v[0] for k, v in results.items() if v[0]}
    assert not failed, "\n".join(f"{k}: {'; '.join(v)}" for k, v in sorted(failed.items()))


FIXTURES = os.environ.get("VEGA_SHOW_FIXTURES")
needs_fixtures = pytest.mark.skipif(not FIXTURES, reason="VEGA_SHOW_FIXTURES (make_page_fixtures.py --out DIR) is not set")


@needs_fixtures
def test_every_fixture_case_passes_the_mock_walk():
    walks = walk_all(Path(FIXTURES))
    failed = {k: out for k, (code, out) in walks.items() if code != 0}
    assert walks and not failed, "\n".join(f"{k}: {out[-1500:]}" for k, out in sorted(failed.items()))


@needs_fixtures
def test_every_cell_lane_is_walked(results):
    walked = walked_lanes(Path(FIXTURES))
    unwalked = sorted({str(lane_key(lane)) for _, lane in results.values() if lane and lane_key(lane) not in walked})
    assert not unwalked, f"lanes the build spans and no fixture case walks: {unwalked}"


# =============================================================================== the command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--cells", help="only the cells whose label holds this substring")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--fixtures", help="the page fixtures (make_page_fixtures.py --out DIR): walk them too")
    src.add_argument("--build-fixtures", action="store_true", help="build the page fixtures, then walk them")
    a = ap.parse_args(argv)
    cells = [c for c in cf.cells() if not a.cells or a.cells in label(c)]
    results = run_cells(cells, a.jobs)
    failed = 0
    for name, (bad, _) in sorted(results.items()):
        if bad:
            failed += 1
            print(f"FAIL: {name}: {'; '.join(bad)}")
    print(f"{'OK' if not failed else 'ERROR'}: {len(results) - failed} of {len(results)} admitted cells built and checked "
          "(extract → draft → autoauthor → resolve → build → check)")
    fixtures = Path(a.fixtures) if a.fixtures else None
    with tempfile.TemporaryDirectory(prefix="coverage-fixtures-") as tmp:
        if a.build_fixtures:
            fixtures = build_fixtures(Path(tmp))
        if fixtures:
            failed += _walks(fixtures, results)
    return 1 if failed else 0


def _walks(fixtures: Path, results: dict) -> int:
    walks, bad = walk_all(fixtures), 0
    for name, (code, out) in sorted(walks.items()):
        if code != 0:
            bad += 1
            print(f"FAIL: mock walk {name} exited {code}: {' | '.join(out.strip().splitlines()[-4:])}")
    print(f"{'OK' if not bad else 'ERROR'}: {len(walks) - bad} of {len(walks)} fixture cases pass smoke_show --mock")
    walked = walked_lanes(fixtures)
    unwalked = sorted({str(lane_key(lane)) for _, lane in results.values() if lane and lane_key(lane) not in walked})
    for lane in unwalked:
        print(f"FAIL: the lane {lane} is built by a cell and walked by no fixture case")
    if not unwalked:
        print(f"OK: every lane the cells build ({len({lane_key(l) for _, l in results.values() if l})}) is walked")
    return bad + len(unwalked)


if __name__ == "__main__":
    sys.exit(main())
