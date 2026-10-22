#!/usr/bin/env python3
"""
test_page_config.py — the `page.json` builders (`page_config.py`, W06) and their gate BD2
(`check_page_config`), on the CIC R7 records and on synthetic records for each view key.

    python3 test_page_config.py           # or: python3 -m pytest test_page_config.py

- the CIC records (`$CIDS_ROOT/bundle/cicids2017-label-R7`, skipped when absent): the lane, the
  metric and its direction, the recipe ops, the walk ids and the explain subset of the R7 bundle;
- synthetic records for every view key, both anomaly score sources, window, graph node and graph
  level samples and a spatial row: the lane table, the metric direction and bounds, the walk
  ids, the views, the reference; every page validates against `page-1.schema.json`;
- the rules one at a time: the lane over every catalog task, `METRIC_BOUNDS` against
  `coverage.json`, the recipe ops, the walk, the views, the motif source, the reference, the
  refusals, and the negatives: PN7 (`motif packets`), a link outside R-C1, a lane against another
  network (PG10's cross-check, done early).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))
sys.path.insert(0, str(TESTS))

import _page_fixtures as fx  # noqa: E402
from page_config import (METRIC_BOUNDS, REFERENCE_QUANTILES, Catalogs, build_page_config,  # noqa: E402
                         build_walk, check_page_config, metric_record, page_records,
                         recipe_summary, reference_record, views_available)
from page_derive import lane_record  # noqa: E402
from records import BundleRecords  # noqa: E402
from vega_show_core import CATALOG, ShowError  # noqa: E402

CATS = Catalogs.load()
HEAD, TAIL = CATALOG["walk"]["head"], CATALOG["walk"]["tail"]
SCHEMA = json.loads((TESTS.parent.parent / "references" / "schemas" / "page-1.schema.json").read_text())
VIEW_KEYS = set(SCHEMA["properties"]["lane"]["properties"]["view_key"]["enum"])
CIDS_BUNDLE = (Path(os.environ.get("CIDS_ROOT", Path.home() / "Documenti" / "cids"))
               / "bundle" / "cicids2017-label-R7")


def _raises(fn, *args, match: str = "") -> str:
    try:
        fn(*args)
    except ShowError as exc:
        assert match in str(exc), f"{match!r} not in {exc}"
        return str(exc)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not refuse")


def _case_page(case: str, **rec) -> dict:
    spec = fx.netspec(case)
    (_, _, _, _, _, _, _, spatial), (view_key, _, _, _, modality) = fx.CASES[case]
    base = CATALOG["motifs"]["by_modality"][modality]["base"]
    scores = {"scores": [0.5, 0.1, 0.9, 0.3, 2.0], "source": "examples/demo.json (train)"}
    records = fx.records(spatial=spatial, scores=scores if view_key in ("reconstruction", "anomaly") else None,
                         **rec)
    story = fx.story(case, spec, CATS.direction(spec["metrics"]["primary"]))
    with tempfile.TemporaryDirectory() as tmp:
        return build_page_config(fx.write_plan(Path(tmp), fx.plan(modality, base)), spec, records, story, CATS)


def _problems(page, network=None) -> list[str]:
    return check_page_config(page, network, CATS)


# --------------------------------------------------------------------------- the CIC records

def test_cic_records():
    if not (CIDS_BUNDLE / "MANIFEST.json").is_file():
        print(f"    (skipped: {CIDS_BUNDLE} is absent)")
        return
    store = BundleRecords.open(CIDS_BUNDLE)
    spec = json.loads((CIDS_BUNDLE / "netspec.json").read_text(encoding="utf-8"))
    presets = json.loads((CIDS_BUNDLE / "examples" / "presets.json").read_text(encoding="utf-8"))
    records = page_records(store, examples={"schema": "vega/examples@2", **presets})
    story = fx.story("classes", spec, "max")
    stance = "2026-09-29 | cicids2017 | show: motif lanes (user)"
    with tempfile.TemporaryDirectory() as tmp:
        paths = fx.write_plan(Path(tmp), fx.plan("row", "points", "lanes", stance))
        page = build_page_config(paths, spec, records, story, CATS)
    assert _problems(page) == [], _problems(page)
    assert page["lane"] == {"task": "multiclass", "base": None, "view_key": "classes", "shape": "row",
                            "level": None, "link": "softmax", "family": "mlp-plain", "goal": "predict",
                            "score_source": None, "source": page["lane"]["source"]}
    assert page["metric"]["primary"] == "f1_macro" and page["metric"]["direction"] == "max"
    assert page["metric"]["bounds"] == [0, 1] and "mcc" in page["metric"]["secondary"]
    assert page["motif"] == {"id": "lanes", "base": "points", "modality": "row", "source": "show:2026-09-29",
                             "reason": "recorded stance: show: motif lanes"}
    assert page["identity"]["title"] == store.manifest()["serve"]["identity"]["title"]
    assert page["bundle"]["manifest_digest"] == store.manifest_digest()
    assert [s["id"] for s in page["walk"]] == [*HEAD, "dense", "activation", "dropout", "head", *TAIL]
    assert list(page["explain"]["blocks"]) == ["dense", "activation", "dropout", "head"]
    assert list(page["explain"]["links"]) == ["softmax"] and "gelu" in page["explain"]["activations"]
    ops = {(o["op"], o["step"]): o["columns"] for o in page["recipe"]["ops"]}
    assert ops[("indicator", "S9")] == 2 and ops[("encode", "S12")] == 1 and ("log1p", "S11b") in ops
    assert ops[("robust_scale", "S13")] == spec["data_contract"]["input_dim"]
    assert page["views"] == {"order": CATALOG["views"]["tour"], "off": {}} and page["reference"] is None


# --------------------------------------------------------------------------- every view key

def test_synthetic_records_for_each_view_key():
    view_keys = set()
    for case, (spec_row, lane_row) in fx.CASES.items():
        page = _case_page(case)
        assert _problems(page) == [], (case, _problems(page))
        assert json.loads(json.dumps(page, allow_nan=False)) == page, case
        view_key, link, goal, source, modality = lane_row
        lane = page["lane"]
        assert (lane["view_key"], lane["link"], lane["goal"], lane["score_source"]) == \
            (view_key, link, goal, source), (case, lane)
        assert lane["base"] == spec_row[1] and page["motif"]["modality"] == modality, case
        assert lane["level"] == (spec_row[6] or {}).get("level"), case
        primary = spec_row[3]
        assert page["metric"]["direction"] == CATS.direction(primary), case
        assert page["metric"]["bounds"] == list(METRIC_BOUNDS[primary]), case
        blocks = [b["type"] for b in fx.netspec(case)["blocks"]]
        middle = [s["id"] for s in page["walk"]][len(HEAD):-len(TAIL)]
        assert middle[0] == blocks[0] and set(middle) >= set(blocks), (case, middle)
        assert (page["reference"] is not None) == (view_key in ("reconstruction", "anomaly")), case
        view_keys.add(view_key)
    assert view_keys == VIEW_KEYS


def test_lane_over_every_catalog_task():
    forge = CATS.forge
    for kind in forge["tasks"]:
        bases = [b for b, k in forge["sequence_map"].items() if k == kind] or [None]
        for base in bases:
            for loss in ("svdd", "vae") if kind == "anomaly" else ("mse",):
                spec = fx.netspec("classes")
                spec["task"] = {"kind": kind, **({"base_kind": base} if base else {})}
                spec["loss"] = {"name": loss}
                lane = lane_record(spec, CATS)
                assert lane["link"] in forge["inspection"]["links"], (kind, base, lane)
                assert lane["view_key"] in VIEW_KEYS, (kind, base, lane)
    for kind in forge["graph_level"]["tasks"]:
        spec = fx.netspec("graph-level")
        spec["task"] = {"kind": kind}
        assert lane_record(spec, CATS)["level"] == "graph"
    spec = fx.netspec("graph-level")
    spec["task"] = {"kind": "multilabel"}
    _raises(lane_record, spec, CATS, match="graph-level")
    spec = fx.netspec("sequence-classes")
    spec["task"]["base_kind"] = "regression"
    _raises(lane_record, spec, CATS, match="base_kind")


# --------------------------------------------------------------------------- the metric

def test_metric_bounds_cover_the_backend_metrics():
    assert set(METRIC_BOUNDS) == set(CATS.coverage["metrics"])
    for name, bounds in METRIC_BOUNDS.items():
        directed = CATS.direction(name) in ("max", "min")
        assert (bounds is not None) == directed, name
        if bounds is not None:
            assert isinstance(bounds[0], (int, float)) and (bounds[1] is None or bounds[0] < bounds[1])


def test_metric_direction_is_the_catalogs():
    spec = fx.netspec("classes")
    assert metric_record(spec, CATS)["direction"] == "max"
    flipped = dict(CATS.forge["metric_direction"], f1_macro="min")
    cats = dataclasses.replace(CATS, forge=MappingProxyType({**CATS.forge, "metric_direction": flipped}))
    assert metric_record(spec, cats)["direction"] == "min"
    spec["metrics"]["selection"]["mode"] = "max"
    _raises(metric_record, spec, cats, match="contradicts")
    spec = fx.netspec("value")
    spec["metrics"]["primary"] = "residual_quantiles"
    _raises(metric_record, spec, CATS, match="direction")
    spec = fx.netspec("value")
    spec["metrics"]["secondary"] = ["rmse", "no_such_metric", "mae", "mae"]
    _raises(metric_record, spec, CATS, match="no_such_metric")
    page = _case_page("classes")
    page["metric"]["direction"] = "min"
    assert any("metric.direction" in p for p in _problems(page))


# --------------------------------------------------------------------------- recipe, walk, views

def test_recipe_ops():
    got = recipe_summary(fx.SHAPE, fx.netspec("classes"))
    assert [(o["op"], o["columns"], o["step"]) for o in got["ops"]] == fx.EXPECTED_OPS
    assert got["source"] == "shape:manifest.steps"
    shape = copy.deepcopy(fx.SHAPE)
    shape["recipe"]["steps"] += [
        {"id": "S6", "op": "transform", "columns": ["d"], "params": {"kind": "yeo-johnson"}},
        {"id": "S7", "op": "winsorize", "columns": ["a"], "params": {}},
        {"id": "S8", "op": "clip", "columns": ["gone"], "params": {}},
        {"id": "S9", "op": "graph_edges", "columns": [], "params": {"method": "knn"}},
        {"id": "S10", "op": "scale", "columns": [], "params": {"kind": "none"}}]
    shape["manifest"]["steps"] += [{"id": f"S{i}"} for i in range(6, 11)]
    ops = [(o["op"], o["columns"], o["step"]) for o in recipe_summary(shape, fx.netspec("graph-node"))["ops"]]
    assert ops[len(fx.EXPECTED_OPS):] == [("other", 1, "S6"), ("clip", 1, "S7"), ("graph_edges", 0, "S9")]
    assert all(o in CATALOG["page"]["recipe_ops"] for o, _, _ in ops)
    lag = {"recipe": {"steps": [{"id": "L1", "op": "lag", "columns": ["a"], "params": {"lags": [1]}}]},
           "manifest": {"steps": [{"id": "L1", "op": "lag", "columns_added": ["a_missing"]}]}}
    assert recipe_summary(lag, fx.netspec("horizon"))["ops"] == [{"op": "window", "columns": 1, "step": "L1"}]


def test_walk_ids():
    walk = build_walk(fx.netspec("labels"), CATS.explain)
    ids = [s["id"] for s in walk]
    assert ids == [*HEAD, "dense", "residual", "layernorm", "activation", "dropout", "head", *TAIL]
    fixed = {s["id"]: s for s in CATS.explain["walk_fixed"]}
    assert walk[0]["code"] == {**{"after": None, "to": None}, **fixed["root"]["code"]}
    assert walk[4]["code"]["at"] == CATS.explain["blocks"]["residual"]["code"]["at"]
    spec = fx.netspec("classes")
    spec["blocks"][0]["type"] = "no_such_block"
    _raises(build_walk, spec, CATS.explain, match="no_such_block")
    broken = {**CATS.explain, "walk_fixed": [s for s in CATS.explain["walk_fixed"] if s["id"] != "route"]}
    _raises(build_walk, fx.netspec("classes"), broken, match="route")


def test_views_availability():
    story = fx.story("classes", fx.netspec("classes"), "max")
    serve = fx.records()["manifest"]["serve"]
    examples = fx.records()["examples"]
    assert views_available(examples, story, serve) == {"order": CATALOG["views"]["tour"], "off": {}}
    thin = {**story, "sections": [{"id": "network"}, {"id": "results"}]}
    assert views_available(examples, thin, serve)["off"] == {"context": "views.off.no_section"}
    assert views_available(None, story, serve)["off"] == {"try": "views.off.no_examples"}
    no_csv = {**serve, "endpoints": [r for r in serve["endpoints"] if r != "POST /v1/predict/csv"]}
    got = views_available({"test": {"rows": [{"a": 1}]}}, story, no_csv)
    assert got == {"order": ["context", "network", "try", "results"], "off": {"many": "views.off.no_route"}}
    _raises(views_available, examples, story, {**serve, "present": False}, match="--serve")
    _raises(views_available, examples, story, {"present": True}, match="endpoints")
    page = _case_page("classes", examples=False)
    assert page["views"]["off"] == {"try": "views.off.no_examples"} and _problems(page) == []
    page["views"]["order"] = ["network", "context", "many", "results"]
    assert any("views.order" in p for p in _problems(page))


# --------------------------------------------------------------------------- motif, reference

def test_motif_source():
    spec, story = fx.netspec("classes"), fx.story("classes", fx.netspec("classes"), "max")
    with tempfile.TemporaryDirectory() as tmp:
        def motif(doc):
            return build_page_config(fx.write_plan(Path(tmp), doc), spec, fx.records(), story, CATS)["motif"]
        assert motif(fx.plan("row", "points"))["source"] == "vega:default"
        assert motif(fx.plan("row", "points", "grid"))["source"] == "user:S3"
        stanced = motif(fx.plan("row", "points", "lanes", "2026-09-29 | orders | show: motif lanes (user)"))
        assert stanced["source"] == "show:2026-09-29" and stanced["id"] == "lanes"
        for doc, match in ((fx.plan("row", "points", "grid", "2026-09-29 | x | show: motif lanes (user)"), "stance"),
                           (fx.plan("window", "traces"), "modality"),
                           ({**fx.plan("row", "points"), "status": "draft"}, "authored")):
            _raises(lambda: motif(doc), match=match)


def test_pn7_motif_packets_is_refused():
    page = _case_page("classes")
    page["motif"]["id"] = "packets"
    problems = _problems(page)
    assert any(p.startswith("ERROR: BD2 motif.id: 'packets' is not admissible") for p in problems), problems
    page["motif"]["id"] = "links"
    assert [p for p in _problems(page) if "motif" in p] == [
        "ERROR: BD2 motif/id: not vega/page@1: 'links' is not one of ['points', 'lanes', 'grid']",
        "ERROR: BD2 motif.id: 'links' is not admissible for the modality row (points, lanes, grid) — R-E4"]
    window = _case_page("horizon")
    assert window["motif"]["id"] == "traces" and _problems(window) == []


def test_reference():
    lane = {"view_key": "anomaly", "link": "distance"}
    got = reference_record(lane, {"scores": [4.0, 0.0, 1.0, 2.0, 3.0], "source": "demo"})
    assert got == {"kind": "distance", "quantiles": {"0.5": 2.0, "0.9": 3.6, "0.99": 3.96}, "n": 5, "source": "demo"}
    assert list(got["quantiles"]) == list(REFERENCE_QUANTILES)
    assert reference_record({"view_key": "reconstruction", "link": "reconstruction"},
                            {"scores": [1.0]})["kind"] == "reconstruction_error"
    assert reference_record({"view_key": "classes", "link": "softmax"}, None) is None
    _raises(reference_record, lane, None, match="row scores")
    _raises(reference_record, lane, {"scores": [1.0, float("nan")]}, match="not finite")
    page = _case_page("anomaly-svdd")
    page["reference"]["kind"] = "reconstruction_error"
    assert any("reference.kind" in p for p in _problems(page))


# --------------------------------------------------------------------------- refusals, negatives

def test_story_and_template_refusals():
    spec = fx.netspec("classes")
    with tempfile.TemporaryDirectory() as tmp:
        paths = fx.write_plan(Path(tmp), fx.plan("row", "points"))
        story = fx.story("binary", spec, "max")
        _raises(build_page_config, paths, spec, fx.records(), story, CATS, match="another network")
        story = fx.story("classes", spec, "max")
        story["code_notes"]["nowhere"] = {"note": "x"}
        _raises(build_page_config, paths, spec, fx.records(), story, CATS, match="nowhere")
        story = fx.story("classes", spec, "max")
        other = dataclasses.replace(CATS, template=MappingProxyType({"version": "2.3.1",
                                                                     "manifest_digest": "sha256:" + "3" * 64}))
        _raises(build_page_config, paths, spec, fx.records(), story, other, match="V1")
        installed = dataclasses.replace(CATS, template=MappingProxyType(dict(fx.TEMPLATE_PIN)))
        page = build_page_config(paths, spec, fx.records(), story, installed)
        assert page["template"] == fx.TEMPLATE_PIN
        _raises(build_page_config, paths, spec, fx.records(scores=None) | {"shape": None}, story, CATS,
                match="shape")


def test_link_outside_the_vocabulary_and_lane_mismatch():
    page = _case_page("binary")
    page["lane"]["link"] = "logistic"
    assert any("lane.link: 'logistic' is not in the one link vocabulary" in p for p in _problems(page))
    page = _case_page("binary")
    page["lane"]["link"] = "softmax"
    assert any(p.startswith("ERROR: BD2 lane.link: 'softmax', but task binary") for p in _problems(page))


def test_lane_against_the_network():
    page = _case_page("binary")
    network = {"schema": "vega/network-view@2", "family": {"name": "mlp-plain", "source": None},
               "task": {"kind": "binary", "base": None, "link": "sigmoid"},
               "sample": {"shape": "row", "level": None}}
    assert _problems(page, network) == []
    page["lane"].update(task="multiclass", view_key="classes", link="softmax")
    problems = _problems(page, network)
    assert any("lane.task: 'multiclass', the network says 'binary'" in p for p in problems), problems
    assert any("lane.link: 'softmax', the network says 'sigmoid'" in p for p in problems)
    assert any("network" in p for p in _problems(_case_page("binary"), {"schema": "vega/network-view@1"}))
    assert any("not an object" in p for p in _problems(_case_page("binary"), ["not", "a", "view"]))
    page = _case_page("reconstruction")
    page["reference"]["quantiles"] = {"0.5": "high"}
    assert any("reference" in p for p in _problems(page))


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
