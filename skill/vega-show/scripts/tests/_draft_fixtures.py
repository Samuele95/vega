"""
_draft_fixtures — what `test_draft.py` and `test_carry.py` share: the CIDS fact set (header
included), a draft run through the command line in-process, a memory file with `show:` stances,
the proposers' small inputs, the partition check and the tests' runner (skips counted apart).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))
sys.path.insert(0, str(TESTS))

import coverage_fixtures as cf  # noqa: E402
import draft_plan  # noqa: E402
from extract_facts import ExtractContext, FactExtractor, ModelPage, report_facts  # noqa: E402
from fact_sink import Catalogs  # noqa: E402
from memory_stances import MemoryStances  # noqa: E402
from records import BundleRecords  # noqa: E402
from vega_show_core import TaskShape  # noqa: E402

CIDS = Path(os.environ.get("CIDS_ROOT", Path.home() / "Documenti" / "cids"))
CIDS_BUNDLE = CIDS / "bundle" / "cicids2017-label-R7"
CIDS_MODEL = (CIDS / "analysis"
              / "cicids2017.domain.analysis.shaped-v2.vega-r2.train.eval.pack.html")
CIDS_PLAN_V1 = CIDS / "vega-runs" / "20260925T214854Z" / "story-plan.json"
ROW = {"family": "mlp-plain", "shape": "row", "level": None, "variant": None}
DAY = "2026-09-30"


class Skip(Exception):
    """An input of the test is absent on this machine (the CIDS bundle): a skip, never a pass."""


def skip(reason: str) -> None:
    """Skip the running test: `pytest.skip` under pytest, `Skip` (counted apart) otherwise."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest
        pytest.skip(reason)
    raise Skip(reason)


# --------------------------------------------------------------------------- helpers

def memory(tmp: Path, bodies: Sequence[str], dataset: str = "demo") -> Path:
    """A memory file whose § Network stances holds `bodies` for `dataset` (and, outside it or
    in a comment, `show:` lines a draft must not read)."""
    path = tmp / "memory.md"
    lines = "".join(f"{DAY} | {dataset} | {b}\n" for b in bodies)
    path.write_text(f"# memory\n\n## Dataset stances\n{DAY} | {dataset} | show: motif grid (user)"
                    f"\n\n## Network stances\n<!-- {DAY} | {dataset} | show: motif links (user) -->"
                    f"\n{lines}", encoding="utf-8")
    return path


def draft_cli(facts: Path, out: Path, extra: Sequence = ()) -> tuple[int, dict | None, str]:
    """`draft_plan.py --facts <facts> --out <out> <extra>` in-process: (exit, plan, printed)."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = draft_plan.main(["--facts", str(facts), "--out", str(out), *map(str, extra)])
    plan = json.loads(out.read_text(encoding="utf-8")) if code == 0 else None
    return code, plan, printed.getvalue()


def draft(tmp: Path, docs: dict, extra: Sequence = ()) -> tuple[int, dict | None, str, dict]:
    """Extract a bundle of `docs` under `tmp` and draft its plan: (exit, plan, printed, facts)."""
    code, facts, printed = cf.run(tmp, docs)
    assert code == 0, printed
    code, plan, printed = draft_cli(tmp / "plan" / "facts.json", tmp / "plan" / "draft.json", extra)
    return code, plan, printed, facts


def cell(task_kind: str) -> dict:
    """The records of the mlp-plain row cell of a task (`coverage_fixtures.cell_docs`)."""
    return cf.cell_docs({**ROW, "task": task_kind})


def task(fields: Mapping | None = None) -> TaskShape:
    """A multiclass row task with `fields` changed."""
    base = {"kind": "multiclass", "goal": "predict", "sample_shape": "row", "family": "mlp-plain",
            "view_key": "classes", "primary": "f1_macro", "direction": "max"}
    return TaskShape(**{**base, **(fields or {})})


def stances(bodies: Sequence[str]) -> MemoryStances:
    """The stances of `bodies`, each a line of the dataset `demo` in § Network stances."""
    text = "## Network stances\n" + "".join(f"{DAY} | demo | {b}\n" for b in bodies)
    return MemoryStances.from_text(text, ["demo"])


def partition_problems(groups: list[dict], shaped: list[str], low: int, high: int) -> list[str]:
    """What keeps `groups` from being a partition of `shaped` into `low`–`high` groups."""
    members = [m for g in groups for m in g["members"]]
    bad = [] if sorted(members) == sorted(shaped) and len(members) == len(set(members)) else [
        f"not a partition: {len(members)} members of {len(shaped)} inputs"]
    bad += [] if low <= len(groups) <= high else [f"{len(groups)} groups, not {low}–{high}"]
    ids = [g.get("suggested_id", g.get("id")) for g in groups]
    return bad + ([f"ids {ids}"] if len(set(ids)) != len(ids) else [])


def cids_facts(tmp: Path) -> Path:
    """The CIDS fact set, header included, written to `tmp/facts.json` (extract_facts's CLI
    stops at PF4 on this pre-v2.3 bundle, which has no showcase slot)."""
    store = BundleRecords.open(CIDS_BUNDLE)
    args = type("Args", (), {"bundle": str(CIDS_BUNDLE), "report": None, "label_roles": None})()
    with contextlib.chdir(CIDS):
        report = report_facts(args, CIDS_MODEL, store)
    page = ModelPage.open(CIDS_MODEL)
    header = {"extracted_at": f"{DAY}T00:00:00Z",
              "bundle": {"dir": str(CIDS_BUNDLE), "manifest_digest": store.manifest_digest(),
                         "slug": store.manifest().get("slug"), "run": store.manifest().get("run"),
                         "model_version": store.manifest().get("model_version")},
              "model": {"path": str(CIDS_MODEL), "turtle_digest": page.turtle_digest(),
                        "layers": page.layers()}}
    extractor = FactExtractor(store, report, Catalogs.load(), ExtractContext(header, ()))
    doc = extractor.extract().to_json()
    (tmp / "facts.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return tmp / "facts.json"


def need_cids() -> None:
    """Skip the running test when the CIDS bundle or its model page is absent."""
    if not (CIDS_BUNDLE / "MANIFEST.json").is_file() or not CIDS_MODEL.is_file():
        skip(f"{CIDS_BUNDLE} or its model page is absent (set CIDS_ROOT)")


def run_tests(namespace: dict) -> int:
    """Run every `test_*` of `namespace` in name order: `ok`, `skip` (counted apart), `FAIL`."""
    tests = [(n, f) for n, f in sorted(namespace.items()) if n.startswith("test_")]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Skip as exc:
            skipped += 1
            print(f"  skip  {name} — {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    passed = len(tests) - failed - skipped
    print(f"\n{passed} of {len(tests)} passed" + (f", {skipped} skipped" if skipped else ""))
    return 1 if failed else 0
