#!/usr/bin/env python3
"""
test_facts_cids.py — the fact extractor (`extract_facts.py` and its five collectors).

    python3 test_facts_cids.py            # or: python3 -m pytest test_facts_cids.py

- **The golden CIDS facts** (D5 §12.3), when the CIC-IDS2017 bundle is present (`CIDS_ROOT`,
  default `~/Documenti/cids`): `source_rows`, `step_s4_rows_after`, `block_b1_units`,
  `run_r7_primary`, the selection, the inputs R6 and R8 removed and no other (the floor rung R1
  removed none), the `robustness_not_run` limit, the pre-v2.3 bundle's missing `ci_level` (no
  fact, `WARN:`), the `runs` group as D5 §6.2 shapes it (R6 `without Destination Port`, R8
  `without degenerate_flow`, each on its run and its change; every label passing the digit rule)
  and the `sizes` group, and **fact recovery**: every record ref the hand-authored v1 story plan
  bound is a fact's ref. The campaign ledger itself (`vega-runs/20260925T214854Z`) gives the
  same inputs to the same runs. Without the bundle these are *skipped*, never passed
  (`pytest.skip`; under `main()` a `skip` line, counted apart from the passes).
- **`features_tested`** (contract §2.3): a value equal to an input's name, or an item the run
  adds that names it (R6's mask); never the prose of an object the run replaced (R1's floor).
  Colliding record ids (`S1`/`s1`, `H1-5`/`H1_5`) are a `WARN:` naming both refs, no traceback.
- **Fact-level coverage**: every admitted (family, task, shape, goal) cell of the vega-forge
  catalog gives a complete fact set (`coverage_fixtures.py`).
- **The gates**, through the CLI: D5's seeded defects 23 (the bundle layer describes another
  manifest, PF1), 36 (an unknown family, PF7) and 37 (an unknown task kind, PF7), each an
  `ERROR:` with exit 2 and no fact set written; an unknown block type or sample shape, an altered
  or unparsable record, no network primary, a fact set the schema rejects (PF2), an unreadable
  model page (PF1), a failing `predict.py --check` (PF3), a bundle without a showcase slot (PF4),
  a stale role map or a `--report` with no compiled report (PF6) and a failed round trip (PF8).
- **The report**: a current role map binds `report_<role>_number/_page` facts (`linked`); no
  report is `brief-links` with the S1 marker; a stage brief that exists is pinned by digest.
- **The reserved values**: `eval_network_<primary>` is `{value, ci, n}`; `eval_ci_level` is read
  when the evaluation records it.
- **Non-finite values** never give a traceback: in `MANIFEST.json` (the identity) or the bundle
  layer's manifest a gated refusal naming the path (PF2, PF1); in a fact's value, an estimate's
  `n`, an analysis relation or a value a group copies, a `WARN:` and the value dropped.
- **The markers file** is appended to when it is a JSON list, created when absent, and left as
  it is, with a `WARN:`, when it is anything else; without `jsonschema` nothing is written.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))
sys.path.insert(0, str(TESTS))

import coverage_fixtures as cf  # noqa: E402
from extract_facts import ExtractContext, FactExtractor, ModelPage, report_facts  # noqa: E402
from fact_sink import Catalogs  # noqa: E402
from facts_campaign import (decision_of, features_named, run_changes,  # noqa: E402
                            run_label)
from records import BundleRecords  # noqa: E402
from vega_show_core import digest_file  # noqa: E402

CIDS = Path(os.environ.get("CIDS_ROOT", Path.home() / "Documenti" / "cids"))
CIDS_BUNDLE = CIDS / "bundle" / "cicids2017-label-R7"
CIDS_MODEL = CIDS / "analysis" / "cicids2017.domain.analysis.shaped-v2.vega-r2.train.eval.pack.html"
CIDS_CAMPAIGN = CIDS / "vega-runs" / "20260925T214854Z"
CIDS_PLAN_V1 = CIDS_CAMPAIGN / "story-plan.json"
CIDS_REMOVED = {"R6": ["Destination Port"], "R8": ["degenerate_flow"]}
V1_FILES = {"geometry": "records/geometry.json", "shape": "records/shape.json",
            "architecture": "records/architecture.json", "training": "records/training.json",
            "evaluation": "records/evaluation.json", "netspec": "netspec.json"}
BINARY = {"family": "mlp-residual", "shape": "row", "level": None, "task": "binary", "variant": None}


class Skip(Exception):
    """An input of the test is absent on this machine (the CIDS bundle): a skip, never a pass."""


def _skip(reason: str) -> None:
    """Skip the running test: `pytest.skip` under pytest, `Skip` (counted apart) under `main()`."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest
        pytest.skip(reason)
    raise Skip(reason)


class _Args:
    def __init__(self, bundle, report=None, label_roles=None):
        self.bundle, self.report, self.label_roles = str(bundle), report, label_roles


def _cids_facts() -> tuple[dict, tuple]:
    store = BundleRecords.open(CIDS_BUNDLE)
    with contextlib.chdir(CIDS):
        report = report_facts(_Args(CIDS_BUNDLE), CIDS_MODEL, store)
    extractor = FactExtractor(store, report, Catalogs.load(), ExtractContext(
        {"extracted_at": "2026-09-30T00:00:00Z"}, ()))
    return extractor.extract().to_json(), extractor.markers


# --------------------------------------------------------------------------- golden CIDS facts

def test_the_cids_golden_facts():
    if not (CIDS_BUNDLE / "MANIFEST.json").is_file() or not CIDS_MODEL.is_file():
        _skip(f"{CIDS_BUNDLE} or its model page is absent (set CIDS_ROOT)")
    assert ModelPage.open(CIDS_MODEL).layer_data("bundle")["manifest"] == \
        json.loads((CIDS_BUNDLE / "MANIFEST.json").read_text(encoding="utf-8")), "PF1 holds"
    doc, markers = _cids_facts()
    facts, groups = doc["facts"], doc["groups"]
    assert facts["source_rows"] == {"value": 2830743, "ref": "records/geometry.json#source.rows",
                                    "kind": "count", "format": "count"}
    assert facts["step_s4_rows_after"]["value"] == 2522362
    assert facts["step_s4_rows_after"]["ref"] == "records/shape.json#manifest.steps[id=S4].rows_after.all"
    assert facts["block_b1_units"]["value"] == 256
    assert facts["block_b1_units"]["ref"] == "netspec.json#blocks[id=B1].params.units"
    store = BundleRecords.open(CIDS_BUNDLE)
    assert facts["run_r7_primary"]["value"] == store.read("records/training.json#experiments.runs[id=R7].primary")
    assert groups["selection"]["run"] == "R7" and facts["selected_primary"]["value"] == facts["run_r7_primary"]["value"]
    r6 = [d for d in groups["decisions"] if "R6" in d["tested_by"]]
    assert r6 and all("Destination Port" in d["features_tested"] for d in r6), r6
    r8 = [d for d in groups["decisions"] if "R8" in d["tested_by"]]
    assert r8 and all("degenerate_flow" in d["features_tested"] for d in r8), r8
    tested = {n for d in groups["decisions"] for n in d["features_tested"]}
    assert tested == {"Destination Port", "degenerate_flow"}, tested
    runs = {r["id"]: r for r in store.read("records/training.json#experiments.runs")}
    inputs = [f["name"] for f in store.read("netspec.json#data_contract.features")]
    assert features_named(runs["R1"], [*inputs, "floor"]) == [], "the floor rung removed no input"
    assert "robustness_not_run" in {lim["id"] for lim in groups["limits"]}
    primary = facts["eval_network_f1_macro"]
    assert set(primary["value"]) == {"value", "ci", "n"} and primary["kind"] == "ci_metric"
    assert "eval_ci_level" not in facts and any("no ci_level" in m for m in markers)
    assert doc["task"]["kind"] == "multiclass" and doc["task"]["primary"] == "f1_macro"
    assert cf.sizes_problems(doc, classes=True) == [] and groups["sizes"]["parts"]["test"] == "test_rows"


def test_the_cids_runs_group():
    """D5 §6.2's `runs` on the real campaign: R6 removed Destination Port and R8 the
    degenerate_flow flag — each run lists it, on the change that did it, and is labelled
    `without …`; no other run names an input; every label passes the digit rule."""
    if not (CIDS_BUNDLE / "MANIFEST.json").is_file():
        _skip(f"{CIDS_BUNDLE} is absent (set CIDS_ROOT)")
    doc, _ = _cids_facts()
    runs = {r["id"]: r for r in doc["groups"]["runs"]}
    assert {rid: r["features"] for rid, r in runs.items() if r["features"]} == CIDS_REMOVED
    for rid, removed in CIDS_REMOVED.items():
        assert runs[rid]["label"] == f"without {removed[0]}", runs[rid]
        assert [d.get("features") for d in runs[rid]["delta"]] == [removed], runs[rid]["delta"]
    assert runs["R0"]["label"] is None and runs["R0"]["reference"] and runs["R7"]["selected"]
    assert runs["R1"]["label"] == "family {run_r1_delta_0_from} → {run_r1_delta_0_to}"
    assert runs["R9"]["label"] == "B1, B4 units {run_r9_delta_0_from} → {run_r9_delta_0_to}"
    assert runs["R5"]["label"] == "grad_clip {run_r5_delta_0_from} → none"
    assert doc["facts"]["run_r9_delta_0_to"]["value"] == 512
    assert cf.runs_problems(doc) == [], cf.runs_problems(doc)[:3]


def test_the_cids_campaign_ledger():
    """The campaign's own ledger (`experiments.json` of the run directory the bundle records):
    the same inputs to the same runs, and the same `without …` labels."""
    ledger = CIDS_CAMPAIGN / "experiments.json"
    if not ledger.is_file() or not (CIDS_BUNDLE / "netspec.json").is_file():
        _skip(f"{ledger} or {CIDS_BUNDLE} is absent (set CIDS_ROOT)")
    spec = json.loads((CIDS_BUNDLE / "netspec.json").read_text(encoding="utf-8"))
    inputs = [f["name"] for f in spec["data_contract"]["features"]]
    blocks = frozenset(b["id"] for b in spec["blocks"])
    runs = json.loads(ledger.read_text(encoding="utf-8"))["runs"]
    named = {r["id"]: features_named(r, inputs) for r in runs}
    assert {rid: n for rid, n in named.items() if n} == CIDS_REMOVED, named
    templates = Catalogs.load().show["delta_label"]
    for run in (r for r in runs if r["id"] in CIDS_REMOVED):
        changes = run_changes(run, "", blocks, inputs)
        label = run_label(changes, [(None, None)] * len(changes), templates, named[run["id"]])
        assert label == f"without {CIDS_REMOVED[run['id']][0]}", label


def test_the_cids_fact_recovery():
    """Every record ref the hand-authored v1 plan bound, the extractor now finds."""
    if not CIDS_PLAN_V1.is_file() or not (CIDS_BUNDLE / "MANIFEST.json").is_file():
        _skip(f"{CIDS_PLAN_V1} or {CIDS_BUNDLE} is absent (set CIDS_ROOT)")
    values = json.loads(CIDS_PLAN_V1.read_text(encoding="utf-8"))["values"]
    wanted = {f"{V1_FILES[src]}#{path}" for src, _, path in
              (v.partition(":") for v in values.values()) if src in V1_FILES}
    doc, _ = _cids_facts()
    found = {f["ref"] for f in doc["facts"].values()}
    assert len(wanted) >= 30 and wanted <= found, f"not extracted: {sorted(wanted - found)}"


# --------------------------------------------------------------------------- coverage

def test_fact_coverage_of_every_admitted_cell():
    failed = []
    for cell in cf.cells():
        with tempfile.TemporaryDirectory() as tmp:
            bad = cf.check_cell(cell, Path(tmp))
        failed += [f"{cell}: {bad}"] if bad else []
    assert len(cf.cells()) >= 120 and not failed, failed[:3]


# --------------------------------------------------------------------------- the gates

def test_negative_23_the_bundle_layer_describes_another_manifest():
    printed = cf.refused(cf.cell_docs(BINARY), "PF1",
                       layer_manifest=lambda m: {**m, "run": "R9"})
    assert "describes another bundle" in printed


def test_negative_36_an_unknown_family():
    docs = cf.cell_docs(BINARY)
    docs["netspec.json"]["family"]["name"] = "mlp-quantum"
    assert "'mlp-quantum' is not in the vega-forge catalog" in cf.refused(docs, "PF7")


def test_negative_37_an_unknown_task_kind():
    docs = cf.cell_docs(BINARY)
    docs["netspec.json"]["task"]["kind"] = "survival"
    assert "task kind 'survival'" in cf.refused(docs, "PF7")


def test_unknown_block_type_and_sample_shape_are_refused():
    docs = cf.cell_docs(BINARY)
    docs["netspec.json"]["blocks"][2]["params"] = {"inner": [{"type": "capsule"}]}
    docs["netspec.json"]["blocks"][2]["type"] = "residual"
    assert "capsule" in cf.refused(docs, "PF7")
    docs = cf.cell_docs(BINARY)
    docs["netspec.json"]["data_contract"]["sample"] = {"shape": "image"}
    assert "sample shape 'image'" in cf.refused(docs, "PF7")


def test_every_refusal_names_its_gate():
    """The refusals that printed a bare `ERROR:` name a gate now (contract §10)."""
    docs = cf.cell_docs(BINARY)
    del docs["records/evaluation.json"]["metrics"]["models"]["network"]["pr_auc"]
    assert "no network estimate of the task's primary" in cf.refused(docs, "PF2")
    docs = cf.cell_docs(BINARY)
    docs["MANIFEST.json"]["slug"] = 5
    assert "do not validate vega/show-facts@1" in cf.refused(docs, "PF2")
    broken = lambda b: (b / "MANIFEST.json").write_text("{", encoding="utf-8")  # noqa: E731
    assert "is not JSON" in cf.refused(cf.cell_docs(BINARY), "PF2", after=broken)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "no-report").mkdir()
        printed = cf.refused(cf.cell_docs(BINARY), "PF6", extra=["--report", f"{tmp}/no-report"])
        assert "names no compiled report" in printed
        _, bundle = cf.write_bundle(Path(tmp) / "absent", cf.cell_docs(BINARY))
        code, _, printed = cf.extract(Path(tmp) / "absent" / "gone.pack.html", bundle,
                                      Path(tmp) / "facts.json")
        assert code == 2 and printed.startswith("ERROR: PF1 ") and "could not be read" in printed, \
            printed


def test_preflight_gates_pf2_pf3_pf4_pf8():
    alter = lambda b: (b / "records" / "training.json").write_text("{}")  # noqa: E731
    assert "altered after pack" in cf.refused(cf.cell_docs(BINARY), "PF2", after=alter)
    docs = cf.cell_docs(BINARY)
    del docs["records/training.json"]
    assert "records/training.json" in cf.refused(docs, "PF2")
    failing = lambda b: (b / "predict.py").write_text("raise SystemExit('round trip broke')\n")  # noqa: E731
    assert "round trip broke" in cf.refused(cf.cell_docs(BINARY), "PF3", after=failing)
    docs = cf.cell_docs(BINARY)
    del docs["MANIFEST.json"]["serve"]["showcase_slot"]
    assert "--serve" in cf.refused(docs, "PF4")
    docs = cf.cell_docs(BINARY)
    docs["MANIFEST.json"]["roundtrip"]["status"] = "fail"
    cf.refused(docs, "PF8")


# --------------------------------------------------------------------------- report and briefs

def test_a_current_role_map_binds_report_facts():
    with tempfile.TemporaryDirectory() as tmp:
        roles = cf.write_report(Path(tmp) / "docs")
        code, doc, printed = cf.run(Path(tmp), cf.cell_docs(BINARY),
                                  ["--report", str(roles.parent), "--label-roles", str(roles)])
        assert code == 0, printed
        facts, report = doc["facts"], doc["groups"]["report"]
        assert report["status"] == "linked" and report["roles_source"]["origin"] == "document-project"
        assert facts["report_net_forward_number"] == {"value": "3.1", "ref": "report:net.forward.number",
                                                      "kind": "report_number", "format": "verbatim"}
        assert facts["report_pack_slice_page"]["value"] == 4
        assert report["roles"]["net.forward"]["page"] == "report_net_forward_page"
        assert report["pdf"]["digest"] == digest_file(roles.parent / "main.pdf")


def test_a_stale_role_map_is_pf6():
    with tempfile.TemporaryDirectory() as tmp:
        roles = cf.write_report(Path(tmp) / "docs", digest="sha256:" + "1" * 64)
        code, doc, printed = cf.run(Path(tmp), cf.cell_docs(BINARY), ["--label-roles", str(roles)])
        assert code == 2 and printed.startswith("ERROR: PF6 ") and "recompile" in printed, printed


def test_brief_links_and_briefs():
    with tempfile.TemporaryDirectory() as tmp:
        brief = Path(tmp) / "vega-runs" / "demo" / "docs" / "train-brief.md"
        brief.parent.mkdir(parents=True)
        brief.write_text("# train\n", encoding="utf-8")
        code, doc, printed = cf.run(Path(tmp), cf.cell_docs(BINARY))
        assert code == 0, printed
        assert doc["groups"]["report"]["status"] == "brief-links"
        assert "WARN: no report — code notes link to the stage briefs" in printed
        assert doc["groups"]["briefs"] == {"train": {"path": str(brief), "digest": digest_file(brief)}}


# --------------------------------------------------------------------------- values and runs

def test_reserved_values_and_markers_file():
    with tempfile.TemporaryDirectory() as tmp:
        docs = cf.cell_docs(BINARY)
        model, bundle = cf.write_bundle(Path(tmp), docs)
        out = bundle / "showcase.building" / "plan" / "facts.json"
        code, doc, printed = cf.extract(model, bundle, out)
        assert code == 0, printed
        assert doc["facts"]["eval_network_pr_auc"]["value"] == {"value": 0.7, "ci": [0.6, 0.8], "n": 180}
        assert doc["facts"]["eval_ci_level"]["ref"] == "records/evaluation.json#metrics.ci_level"
        assert doc["task"]["outcomes"] == ["no", "yes"] and doc["bundle"]["slug"] == "demo-mlp-residual"
        markers = json.loads((out.parent.parent / ".work" / "markers.json").read_text(encoding="utf-8"))
        assert any(m.endswith("facts → " + str(out)) for m in markers), markers
        del docs["records/evaluation.json"]["metrics"]["ci_level"]
        code, doc, printed = cf.run(Path(tmp) / "again", docs)
        assert code == 0 and "eval_ci_level" not in doc["facts"] and "no ci_level" in printed


def test_json_result():
    with tempfile.TemporaryDirectory() as tmp:
        code, doc, printed = cf.run(Path(tmp), cf.cell_docs(BINARY), ["--json"])
        result = json.loads(printed)
        assert code == 0 and result["status"] == "ok" and result["facts"] == len(doc["facts"])
        docs = cf.cell_docs(BINARY)
        docs["netspec.json"]["task"]["kind"] = "survival"
        code, _, printed = cf.run(Path(tmp) / "refused", docs, ["--json"])
        result = json.loads(printed)
        assert code == 2 and result["status"] == "refused" and result["markers"][0].startswith("ERROR: PF7")


def test_run_changes_decisions_and_features_tested():
    run = {"id": "R6", "delta": [
        {"path": "blocks", "from": [{"id": "B1", "type": "dense", "params": {"units": 8}}],
         "to": [{"id": "B0", "type": "custom", "rationale": "zeroes Destination Port (column 0)"},
                {"id": "B1", "type": "dense", "params": {"units": 16}}]},
        {"path": "regularization[kind=dropout]/value", "from": 0.1, "to": 0.0}]}
    changes = run_changes(run, "records/training.json#experiments.runs[id=R6]", frozenset({"B1"}))
    assert [(c.path, c.decision, c.change) for c in changes] == [
        ("blocks[id=B1]/params/units", "B1", "changed"), ("blocks[id=B0]", "family", "added"),
        ("regularization[kind=dropout]/value", "regularization[dropout]", "changed")]
    assert changes[0].refs[1] == "records/training.json#experiments.runs[id=R6].delta.0.to[id=B1].params.units"
    assert features_named(run, ["Destination Port", "Flow Duration"]) == ["Destination Port"]
    assert decision_of("blocks/B3/params/inner/0/params/units", {"B3"}) == "B3"
    assert decision_of("data_contract/sample/context") is None


def test_features_tested_is_a_value_or_an_added_item():
    """An input is tested by a value equal to its name, or by an item the run adds that names
    it; never by the prose of an object the run replaced, nor by the name of a choice."""
    inputs = ["floor", "linear", "alpha"]
    assert features_named(cf.FLOOR_RUNG, inputs) == []
    assert features_named(cf.MASK_RUNG, inputs) == ["floor"]
    listed = {"delta": [{"path": "blocks[id=B0]/params/columns", "from": ["alpha"],
                         "to": ["alpha", "floor"]}]}
    assert features_named(listed, inputs) == ["floor"]
    prose = {"delta": [{"path": "blocks[id=B2]/params/comment", "from": "x",
                        "to": "drops the floor"}]}
    assert features_named(prose, inputs) == []
    assert features_named({"delta": [{"path": "family/name", "from": "mlp", "to": "linear"}]},
                          inputs) == []


def test_the_floor_rung_tests_no_input_called_floor():
    """Through the CLI: an input named `floor` and the floor rung give no `features_tested`; a
    masking rung that adds a block zeroing it tests it, on the `family` decision alone."""
    docs = cf.rename_input(cf.cell_docs(BINARY), "gamma", "floor")
    runs = docs["records/training.json"]["experiments"]["runs"]
    runs[1] = cf.FLOOR_RUNG
    with tempfile.TemporaryDirectory() as tmp:
        code, doc, printed = cf.run(Path(tmp), docs)
        assert code == 0, printed
        decisions = doc["groups"]["decisions"]
        assert "R1" in next(d for d in decisions if d["id"] == "family")["tested_by"], decisions
        assert all(d["features_tested"] == [] for d in decisions), decisions
        runs.append(cf.MASK_RUNG)
        code, doc, printed = cf.run(Path(tmp) / "masked", docs)
        assert code == 0, printed
        tested = {d["id"]: d["features_tested"] for d in doc["groups"]["decisions"]
                  if d["features_tested"]}
        assert tested == {"family": ["floor"]}, tested
        runs = {r["id"]: r for r in doc["groups"]["runs"]}
        assert runs["R3"]["features"] == ["floor"] and runs["R3"]["label"] == "without floor"
        assert runs["R3"]["delta"][0]["features"] == ["floor"], runs["R3"]["delta"]
        assert runs["R1"]["features"] == [] and runs["R1"]["label"].startswith("family {run_r1")
        assert all("features" not in d for d in runs["R1"]["delta"]), runs["R1"]["delta"]
        assert cf.runs_problems(doc) == [], cf.runs_problems(doc)


def test_colliding_record_ids_are_warned_not_a_traceback():
    """Two record ids that spell one fact key give the first id's fact and a `WARN:` naming
    both refs (contract: `ERROR:` exit 2, or `WARN:` and a skip — never a traceback)."""
    docs = cf.cell_docs(BINARY)
    steps = docs["records/shape.json"]["manifest"]["steps"]
    steps.append({**steps[0], "id": "s1"})
    runs = docs["records/training.json"]["experiments"]["runs"]
    runs += [{**runs[2], "id": "H1-5"}, {**runs[2], "id": "H1_5", "primary": 0.44}]
    blocks = docs["netspec.json"]["blocks"]
    blocks[0]["id"] = "B1-a"
    blocks.append({**blocks[0], "id": "B1_a"})
    with tempfile.TemporaryDirectory() as tmp:
        code, doc, printed = cf.run(Path(tmp), docs)
    assert code == 0 and "Traceback" not in printed, printed
    for key, first, second in (("step_s1_columns_removed_n", "steps[id=S1]", "steps[id=s1]"),
                               ("run_h1_5_primary", "runs[id=H1-5]", "runs[id=H1_5]"),
                               ("block_b1_a_alternatives_n", "blocks[id=B1-a]", "blocks[id=B1_a]")):
        assert first in doc["facts"][key]["ref"], doc["facts"][key]
        warned = [m for m in printed.splitlines() if m.startswith(f"WARN: fact key '{key}'")]
        assert warned and first in warned[0] and second in warned[0], (key, printed)
    assert doc["facts"]["run_h1_5_primary"]["value"] == 0.45


def test_the_breakdown_check_sees_a_misaligned_trivial_side():
    """The coverage check's breakdown rule is not vacuous: a trivial vector of another length
    is no trivial side (a `WARN:`), and the check names it."""
    cell = {**BINARY, "task": "multiclass"}
    docs = cf.cell_docs(cell)
    trivial = docs["records/evaluation.json"]["metrics"]["models"]["trivial"]
    trivial["f1_per_class"]["values"] = trivial["f1_per_class"]["values"][:2]
    with tempfile.TemporaryDirectory() as tmp:
        code, doc, printed = cf.run(Path(tmp), docs)
    assert code == 0 and "no trivial side" in printed, printed
    bad = cf.breakdown_problems(doc, cf.breakdown_of("multiclass", None))
    assert bad and "trivial" in bad[0], bad


def test_unspellable_class_names_are_warned():
    docs = cf.cell_docs({**BINARY, "task": "multiclass"})
    docs["netspec.json"]["task"]["classes"][1] = "Web Attack"
    docs["netspec.json"]["data_contract"]["class_balance"]["train"]["Web Attack"] = 0.2
    with tempfile.TemporaryDirectory() as tmp:
        code, doc, printed = cf.run(Path(tmp), docs)
    assert code == 0 and "'Web Attack'" in printed and "train_share_calm" in doc["facts"]
    assert not any(re.fullmatch(r"train_share_web_attack", k) for k in doc["facts"])


def test_run_labels_from_the_catalog_templates():
    """Each change path's first segment picks its `delta_label` template; scalars are the
    change's fact keys, an absent value `none`, a list `…`; a parameter the template does not
    name is put before its value; equal block changes are one part."""
    templates = Catalogs.load().show["delta_label"]
    run = {"delta": [
        {"path": "schedule/params/warmup_epochs", "from": 3, "to": 0},
        {"path": "loss/params/weight", "from": [0.1, 0.9], "to": None},
        {"path": "blocks[id=B2]/params/p", "from": 0.1, "to": 0.3},
        {"path": "blocks[id=B5]/params/p", "from": 0.1, "to": 0.3},
        {"path": "blocks", "from": [], "to": [{"id": "B9", "type": "dropout"}]},
        {"path": "training/batch_size", "from": 64, "to": 128}]}
    changes = run_changes(run, "", frozenset({"B2", "B5"}))
    sides = [(f"k{i}_from", f"k{i}_to") if isinstance(c.before, (int, float)) else (None, None)
             for i, c in enumerate(changes)]
    assert run_label(changes, sides, templates) == (
        "schedule warmup_epochs {k0_from} → {k0_to}; loss weight … → none; "
        "B2, B5 p {k2_from} → {k2_to}; B9 type none → dropout; "
        "training batch_size {k5_from} → {k5_to}")
    assert run_label([], [], templates) is None


def test_non_finite_values_are_refused_or_dropped():
    """A NaN or an infinity in a record is a gated refusal naming its path, or a `WARN:` and
    the value dropped — never a traceback (JSON cannot spell it; `facts.json` holds none)."""
    nan = float("nan")

    def put(where, key, value=nan):
        return lambda docs: where(docs).__setitem__(key, value)

    refusals = (("PF2", "MANIFEST.json#serve.identity.weight",
                 put(lambda d: d["MANIFEST.json"]["serve"]["identity"], "weight", float("inf"))),
                ("PF2", "records/evaluation.json#metrics.models.network.pr_auc",
                 put(lambda d: d["records/evaluation.json"]["metrics"]["models"]["network"]["pr_auc"], "n")))
    for gate, where, edit in refusals:
        docs = cf.cell_docs(BINARY)
        edit(docs)
        assert where in cf.refused(docs, gate)
    printed = cf.refused(cf.cell_docs(BINARY), "PF1", layer_manifest=lambda m: {**m, "x": nan})
    assert "#layer-bundle-data.manifest.x" in printed and "non-finite" in printed, printed
    docs = cf.cell_docs(BINARY)
    for edit in (put(lambda d: d["records/analysis.json"]["modules"]["relations"]["evidence"]["pairs"][0], "spearman"),
                 put(lambda d: d["records/training.json"]["experiments"]["runs"][2], "eligible"),
                 put(lambda d: d["records/training.json"]["experiments"]["runs"][2]["delta"][0], "to"),
                 put(lambda d: d["records/evaluation.json"]["metrics"]["models"]["network"]["roc_auc"], "n"),
                 put(lambda d: d["netspec.json"]["data_contract"]["features"][0], "origin", -float("inf"))):
        edit(docs)
    with tempfile.TemporaryDirectory() as tmp:
        code, doc, printed = cf.run(Path(tmp), docs)
        written = (Path(tmp) / "plan" / "facts.json").read_text(encoding="utf-8")
    assert code == 0 and "Traceback" not in printed and "NaN" not in written, printed
    groups = doc["groups"]
    assert groups["inputs"]["relations"] == [] and groups["runs"][2]["eligible"] is None
    assert groups["inputs"]["origin"]["alpha"] is None and "eval_network_roc_auc" not in doc["facts"]
    assert "run_r2_delta_0_to" not in doc["facts"] and "…" in groups["runs"][2]["label"]
    for needle in ("pairs.0.spearman is not a finite number", "groups.runs.2.eligible: a non-finite",
                   "groups.inputs.origin.alpha: a non-finite", "no fact run_r2_delta_0_to",
                   "roc_auc is an estimate with a non-finite"):
        assert needle in printed, (needle, printed)


def test_the_markers_file_is_never_silently_replaced():
    """A markers file that is not a JSON list is left byte for byte, with a `WARN:`; a list is
    appended to; an absent one is created."""
    for content in ('{"note": "a hand-kept object"}', "{ not json", "null"):
        with tempfile.TemporaryDirectory() as tmp:
            markers = Path(tmp) / "markers.json"
            markers.write_text(content, encoding="utf-8")
            code, _, printed = cf.run(Path(tmp), cf.cell_docs(BINARY), ["--markers", str(markers)])
            assert code == 0 and markers.read_text(encoding="utf-8") == content, printed
            assert f"WARN: the markers file {markers} is not a JSON list" in printed, printed
    with tempfile.TemporaryDirectory() as tmp:
        markers = Path(tmp) / "markers.json"
        markers.write_text('["WARN: kept", NaN]', encoding="utf-8")
        code, _, printed = cf.run(Path(tmp), cf.cell_docs(BINARY), ["--markers", str(markers)])
        assert code == 0 and markers.read_text(encoding="utf-8") == '["WARN: kept", NaN]'
        assert f"WARN: the markers could not be appended to {markers}" in printed, printed
    with tempfile.TemporaryDirectory() as tmp:
        markers = Path(tmp) / "work" / "markers.json"
        code, _, printed = cf.run(Path(tmp), cf.cell_docs(BINARY), ["--markers", str(markers)])
        first = json.loads(markers.read_text(encoding="utf-8"))
        cf.run(Path(tmp) / "again", cf.cell_docs(BINARY), ["--markers", str(markers)])
        second = json.loads(markers.read_text(encoding="utf-8"))
        assert code == 0 and first and second[:len(first)] == first and len(second) > len(first)


def test_no_jsonschema_writes_nothing():
    """Without `jsonschema` the fact set cannot be validated: an `ERROR:`, exit 2, nothing
    written — never a traceback, never an unvalidated fact set."""
    saved = sys.modules.get("jsonschema")
    sys.modules["jsonschema"] = None  # `import jsonschema` now raises ImportError
    try:
        with tempfile.TemporaryDirectory() as tmp:
            code, doc, printed = cf.run(Path(tmp), cf.cell_docs(BINARY))
            assert not (Path(tmp) / "plan" / "facts.json").exists()
    finally:
        if saved is None:
            del sys.modules["jsonschema"]
        else:
            sys.modules["jsonschema"] = saved
    assert code == 2 and doc is None and printed.startswith("ERROR: jsonschema"), printed


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
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


if __name__ == "__main__":
    sys.exit(main())
