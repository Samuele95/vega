#!/usr/bin/env python3
"""
test_no_planned.py — the backend has nothing left planned (D6 §15.5; gate A, BUILD-PLAN §3.3).

    python3 -m pytest -q vega-forge/scripts/tests/test_no_planned.py

1. coverage.json's `planned` holds no capability: every list in it (blocks, families,
   metrics, ...) is empty;
2. no catalog block is `backend_status: planned`, and the backend implements every catalog
   block type (coverage.json `blocks`);
3. every admission cell of catalog.json (family x sample shape x graph level x task, 127 cells)
   compiles its catalog default through the compiler's functional core (compile_spec) without
   a B2 marker: the family's block types, the family's or the task's default loss (the base
   task's for a sequence task), the default optimizer and schedule, the task's primary and
   secondary metrics. Only B2 is read: the other rules judge a whole chain, which a catalog
   default is not.

It fails on the tree as B03-B05 left it (the eight B-graph-recurrent blocks and nine families
are still planned) and becomes a gate when B14 empties `planned` (W3). $VEGA_BACKENDS_DIR points
it at another backends directory (the negative RG3 runs it on a copy whose `planned.blocks`
is seeded).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import compile_spec  # noqa: E402

CATALOG = SCRIPTS / "catalog.json"
BACKENDS = Path(os.environ.get("VEGA_BACKENDS_DIR") or SCRIPTS / "backends")
BACKEND = "pytorch"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage() -> dict:
    return _json(BACKENDS / BACKEND / "coverage.json")


def _cells(catalog: dict) -> list[tuple[str, str, str | None, str]]:
    """(family, shape, level, task) for every admission cell, the named variants included (a variant
    compiles the same default: `gcn x graph/node x multiclass [transductive]`)."""
    out = []
    for row in catalog["admission"]["rows"]:
        out += [(fam, row["shape"], row.get("level"), task) for fam in row["families"] for task in row["tasks"]]
        out += [(v["family"], row["shape"], row.get("level"), f"{v['task']} [{v['variant']}]")
                for v in row.get("variants") or []]
    return out


def _task_parts(catalog: dict, task: str) -> tuple[dict, dict | None]:
    """The task row and, for `kind/base`, the base row a sequence task reads its loss and metrics
    from (`anomaly/labelled` is the anomaly row with or without a label)."""
    kind, _, base = task.split(" [")[0].partition("/")
    base_row = catalog["tasks"].get(base) if kind.startswith("sequence-") else None
    return catalog["tasks"][kind], base_row


def _segment(composition: str, shape: str, level: str | None) -> str:
    """The part of a family's composition that one shape and level build: a composition written
    per shape (`row: ...; window: ...; graph, level node: ...`) or per level (`... -> head at level
    node; ... -> graph_pool(mean) -> head at level graph`, the second continuing the first)."""
    segments = [s.strip() for s in composition.split(";")]
    for seg in segments if len(segments) > 1 else []:
        head = seg.split(":")[0]
        if ":" in seg and head.startswith(shape) and ("level" not in head or f"level {level}" in head):
            return seg
        if level and seg.endswith(f"at level {level}"):
            return f"{segments[0]} {seg}" if seg.startswith("...") else seg
    return composition


def block_types(catalog: dict, family: str, shape: str, level: str | None) -> list[str]:
    """The family's block types this shape and level build: all of them but those its composition
    names only for another shape or level (a segment, or a `(+ graph_pool(...) ... at level graph)`
    clause, of another level)."""
    row = catalog["families"][family]
    part = re.sub(r"\(\+\s*(.*?)\s+at level (\w+)\)", lambda m: m.group(1) if m.group(2) == level else "",
                  _segment(row["composition"], shape, level))
    named = lambda t, text: re.search(rf"\b{re.escape(t)}\b", text)  # noqa: E731
    return [t for t in row["block_types"] if named(t, part) or not named(t, row["composition"])]


def default_spec(catalog: dict, cell: tuple[str, str, str | None, str]) -> dict:
    """The catalog default of one cell as a netspec the compiler reads."""
    family, shape, level, task = cell
    row, base = _task_parts(catalog, task)
    own = [m for m in row.get("secondary") or [] if "base task" not in m]
    source = base or row
    loss = catalog["families"][family].get("loss") or source["default_loss"]
    kind, _, base_kind = task.split(" [")[0].partition("/")
    return {"schema": compile_spec.SCHEMA, "input": {}, "seed": 0,
            "task": {"kind": kind, **({"base_kind": base_kind} if base else {})},
            "data_contract": {}, "family": {"name": family},
            "blocks": [{"type": t} for t in block_types(catalog, family, shape, level)],
            "loss": {"name": loss}, "metrics": {"primary": source["primary"],
                                                "secondary": list(source.get("secondary") or []) + own},
            "optimizer": dict(catalog["defaults"]["optimizer"]["value"]),
            "schedule": dict(catalog["defaults"]["schedule"]["value"]), "regularization": [],
            "early_stopping": {}, "init": {}, "training": {}, "backend": {"name": BACKEND}, "outputs": {}}


def test_planned_is_empty():
    planned = _coverage().get("planned") or {}
    left = {k: v for k, v in planned.items() if isinstance(v, list) and v}
    assert not left, "coverage.json still plans " + "; ".join(f"{k}: {', '.join(map(str, v))}"
                                                              for k, v in sorted(left.items()))


def test_no_catalog_block_is_planned():
    blocks = _json(CATALOG)["blocks"]
    planned = sorted(b for b, row in blocks.items() if (row or {}).get("backend_status") == "planned")
    missing = sorted(set(blocks) - set(_coverage().get("blocks") or []))
    assert not planned, f"catalog blocks marked backend_status: planned: {', '.join(planned)}"
    assert not missing, f"catalog block types the {BACKEND} backend does not implement: {', '.join(missing)}"


def test_every_admission_cell_compiles_its_default_without_b2():
    catalog, failures = _json(CATALOG), []
    cells = _cells(catalog)
    assert len(cells) >= 127, f"the admission table holds {len(cells)} cells, fewer than the 127 of §4"
    for cell in cells:
        markers = compile_spec.compile_spec(default_spec(catalog, cell), backend=BACKEND,
                                            catalog_path=CATALOG, backends_dir=BACKENDS)
        b2 = sorted({m["message"] for m in markers.as_json() if m.get("rule") == "B2"})
        family, shape, level, task = cell
        if b2:
            failures.append(f"{family} x {shape}{'/' + level if level else ''} x {task}: {'; '.join(b2)}")
    assert not failures, f"{len(failures)} of {len(cells)} admission cells hit B2:\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
