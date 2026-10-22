#!/usr/bin/env python3
"""
test_walk_anchors.py — every walk step's code anchor resolves in a rendered package of each sample
shape (BUILD-PLAN W15, R-E8; D4 P9/P13).

    python3 -m pytest -q test_walk_anchors.py

The walk is `page_config.build_walk` (root, store, build, one step per distinct block type in
forward order, forward, link, shape, run, explain, route), each step's anchor copied verbatim from
the backend's explain catalog (a file pattern, `at` patterns, an optional `after`). The packages
are the ones a bundle of each network carries, rendered from the live templates
(`make_page_fixtures.code_view`: the forge's package in full, the serving package with its
`inspection/` modules in the Predictor's module, the composition root `predict.py`, the route
modules `serve/*.py`) for every network of the page fixture matrix (`page-matrix.json`): rows,
windows, graph nodes and whole graphs, every family, every one of the 33 block types. Each walk is
resolved by the code room's rule written independently in Python
(`assets/page/tests/fixtures/code-view/make_code_views.py` `resolve`; `test_code_room.py` holds the
page's own resolution to it), and:

- no step of any walk is left out — not for its file, not for its anchor, not for its pattern;
- a block type's step lands in the forge's block modules (`model/blocks_{core,seq,graph,gen}.py`;
  a `custom` block's in its own module, `model/custom/<stem>.py`, which the package ships),
  `root` in the composition root, `run` in the Predictor's `predict.py`, `store` in the store,
  `build` and `forward` and `link` in `network.py`, `shape` in the Recipe's module, `explain` in
  the inspector, `route` in the predict routes;
- the matrix's networks take every block type of the forge catalog, so every block's anchor is
  resolved at least once.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
REPO = TESTS.parents[2]
FIXTURES = SCRIPTS.parent / "assets" / "page" / "tests" / "fixtures" / "code-view"
EXPLAIN = REPO / "vega-forge" / "scripts" / "backends" / "pytorch" / "explain-catalog.json"
CATALOG = REPO / "vega-forge" / "scripts" / "catalog.json"
for _p in (str(TESTS), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_page_fixtures as mpf  # noqa: E402
from page_config import build_walk  # noqa: E402

FIXED = {"root": r"^predict\.py$", "store": r"/model/store\.py$", "build": r"/model/network\.py$",
         "forward": r"/model/network\.py$", "link": r"/model/network\.py$", "shape": r"/preprocess\.py$",
         "run": r"/serving/predict\.py$", "explain": r"/serving/inspection/inspector\.py$",
         "route": r"^serve/routes_predict\.py$"}
BLOCK_FILE = re.compile(r"/model/blocks_(core|seq|graph|gen)\.py$")
CUSTOM_FILE = re.compile(r"/model/custom/[^/]+\.py$")       # a custom block's own module (R-B9)


def _module(name: str, path: Path):
    """A module from its file, leaving no bytecode beside it (the fixtures belong to the template)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    kept, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = kept
    return mod


RULE = _module("walk_anchor_rule", FIXTURES / "make_code_views.py")
CASES = mpf.load_matrix()


@functools.lru_cache(maxsize=1)
def zoo():
    return mpf.load_zoo()


@functools.lru_cache(maxsize=None)
def resolved(case_id: str) -> tuple:
    """(the network's spec, its walk, the walk resolved against the bundle's rendered code)."""
    case = next(c for c in CASES if c["id"] == case_id)
    spec = mpf.case_spec(case, zoo())
    explain = json.loads(EXPLAIN.read_text(encoding="utf-8"))
    walk = build_walk(spec, explain)
    with tempfile.TemporaryDirectory(prefix="walk-anchors-") as tmp:
        view = mpf.code_view(spec, Path(tmp))
    return spec, walk, RULE.resolve(walk, view["files"], view["language"])


@pytest.mark.parametrize("case_id", [c["id"] for c in CASES])
def test_every_walk_step_resolves_in_the_rendered_package(case_id):
    spec, walk, got = resolved(case_id)
    assert got["missing"] == [], f"{case_id}: walk steps left out: {got['missing']}"
    assert [s["id"] for s in got["steps"]] == [s["id"] for s in walk], case_id
    paths = {s["id"]: s["path"] for s in got["steps"]}
    for sid, pattern in FIXED.items():
        assert re.search(pattern, paths[sid]), f"{case_id}: step {sid} landed in {paths[sid]}"
    blocks = [s["id"] for s in walk if s["id"] not in FIXED]
    assert blocks, case_id
    off = {b: paths[b] for b in blocks if not (CUSTOM_FILE if b == "custom" else BLOCK_FILE).search(paths[b])}
    assert off == {}, f"{case_id}: block steps outside the block modules: {off}"
    for step in got["steps"]:
        assert 1 <= step["from"] <= step["to"], step


def test_each_sample_shape_is_rendered_and_every_block_type_is_resolved():
    shapes, types = set(), set()
    for case in CASES:
        spec, walk, got = resolved(case["id"])
        sample = spec["data_contract"]["sample"]
        shapes.add(sample["shape"] + (f"/{sample['level']}" if sample.get("level") else ""))
        types.update(s["id"] for s in got["steps"] if s["id"] not in FIXED)
    assert shapes == {"row", "window", "graph/node", "graph/graph"}
    catalog = set(json.loads(CATALOG.read_text(encoding="utf-8"))["blocks"])
    assert types == catalog, f"block types the matrix never resolves: {sorted(catalog - types)}"


def test_the_rule_reports_an_anchor_the_package_does_not_hold():
    """The independent rule is not vacuous: a block anchor moved to a name no module defines, and
    a file pattern no rendered file matches, are each reported."""
    spec, walk, _ = resolved(CASES[0]["id"])
    with tempfile.TemporaryDirectory(prefix="walk-anchors-") as tmp:
        files = mpf.code_view(spec, Path(tmp))["files"]
    moved = json.loads(json.dumps(walk))
    block = next(s for s in moved if s["id"] not in FIXED)
    block["code"]["at"] = [r"^class NoSuchBlock\b"]
    next(s for s in moved if s["id"] == "store")["code"]["file"] = r"(^|/)no_such_store\.py$"
    got = RULE.resolve(moved, files, "python")
    assert {(m["id"], m["why"]) for m in got["missing"]} == {(block["id"], "anchor"), ("store", "file")}
