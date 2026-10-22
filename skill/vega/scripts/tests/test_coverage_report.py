#!/usr/bin/env python3
"""
test_coverage_report.py — the coverage report's own proof (D6 §0, §12.8, §13; BUILD-PLAN T12).

    python3 -m pytest -q test_coverage_report.py

The matrices here are synthetic: each lane's observed.json is written from what the lane DECLARES
(so every entry the registry spans is observed once), then one thing is seeded wrong at a time.
What is proved:

1. `--declared-only` passes on the tree's registry; on release, the motif only G00 declares is
   deferred to nightly, reported, and not covered.
2. A matrix whose every release lane passed through page, with a passing case for every new gate,
   is PASS; the JSON is `vega/coverage-report@1` and the markdown carries the grids and the gates.
3. A lane that stopped before the stage that proves a dimension does not cover it: the entry is
   uncovered, with the lane and where it stopped; a stage that passed but did not run a check
   (`not_run`) proves nothing past itself; a missing observed.json is said.
4. A report-only dimension (metrics) only warns; an observed entry the catalog does not hold fails.
5. Gates: a case `C11a` covers `C11` and never `C1`; a grandfathered family key covers its
   lettered rules; a vega-show gate is covered by a case its catalog row names; a failing case
   covers nothing; a suite that left no result is named; a malformed result is a problem.
6. `--tier pr` lets the registry's other lanes stand in by declaration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import coverage_gates  # noqa: E402
import coverage_md  # noqa: E402
import coverage_report as cr  # noqa: E402
from lanekit import contracts, registry  # noqa: E402
from lanekit.stages import stage_plan  # noqa: E402

SKILLS = HERE.parents[2]
LANES = registry.load_lanes()
SHOW = json.loads((SKILLS / contracts.SHOW_CATALOG).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- synthetic matrices
def observed_of(lane: registry.Lane, last: str | None = None, not_run: dict | None = None) -> dict:
    """An observed.json whose values are the lane's declarations, every stage of its plan passed
    (up to `last`, the rest blocked)."""
    d = cr.declared_values(lane)
    one = lambda dim: next(iter(d[dim]), None)  # noqa: E731
    graph = lane.raw.get("graph") or {}
    stages, halted = [], False
    for stage in stage_plan(lane):
        status = "blocked" if halted else "pass"
        stages.append({"id": stage, "status": status, "seconds": 1.0, "log": None,
                       "markers": {"ok": 1, "warn": 0, "error": 0}, "detail": "",
                       "not_run": list((not_run or {}).get(stage, []))})
        halted = halted or stage == last
    if last:
        stages[[s["id"] for s in stages].index(last)]["status"] = "fail"
        stages[[s["id"] for s in stages].index(last)]["detail"] = "seeded"
    report = {"stub": "linked", "real": "linked", "none": "brief-links"}[lane.raw["report"]]
    return {"schema": contracts.LANE_OBSERVED, "lane": lane.id, "verdict": "fail" if last else "pass",
            "stages": stages, "git": {"commit": None, "dirty": False},
            "observed": {"family": lane.family, "task": lane.task, "goal": lane.goal, "shape": lane.shape,
                         "graph": {"level": graph.get("level"), "protocol": graph.get("protocol")} if graph else None,
                         "protocol": graph.get("protocol"), "blocks": lane.flat_block_types(),
                         "activations": sorted(d["activations"]), "loss": one("losses"),
                         "optimizer": one("optimizers"), "schedule": one("schedules"), "init": one("init_schemes"),
                         "metrics_finite": ["accuracy"], "verdict": one("verdicts"),
                         "showcase": {"motif": one("motifs"), "report": report,
                                      "class_roles": [[c, r] for c, r in (lane.raw.get("roles") or {}).items()]},
                         "page": {"views": sorted(d["views"]), "themes": ["day", "night"], "widths": [390, 1440]}}}


def every_new_gate_case() -> list[dict]:
    """One passing case per new gate of every table, in the suite its table reads."""
    cases = {}
    for table in coverage_gates.gate_tables(SKILLS):
        suite = table.suites[0]
        for gate in table.ids:
            if gate not in table.baseline:
                cases.setdefault(suite, []).append({"id": gate, "name": f"{gate}  seeded", "ok": True, "detail": ""})
    return [{"schema": "vega/negatives-result@1", "script": s, "work": "/w", "cases": c,
             "passed": len(c), "failed": 0} for s, c in cases.items()]


def write_matrix(root: Path, docs: dict, negatives: list[dict] | None = None) -> Path:
    for lane_id, doc in docs.items():
        (root / "lanes" / lane_id).mkdir(parents=True, exist_ok=True)
        (root / "lanes" / lane_id / "observed.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "negatives").mkdir(parents=True, exist_ok=True)
    for n, doc in enumerate(negatives if negatives is not None else every_new_gate_case()):
        (root / "negatives" / f"suite{n}.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


def release_docs(**seed) -> dict:
    return {l.id: observed_of(l, **(seed.get(l.id) or {})) for l in registry.lanes_for(LANES, None, "release")}


def build(root: Path, tier: str | None = "release") -> dict:
    return cr.Report(SKILLS, root, tier, False).build()


# --------------------------------------------------------------------------- 1. declared
def test_the_registry_spans_the_catalogs_by_declaration():
    doc = cr.Report(SKILLS, None, None, True).build()
    assert doc["verdict"] == "PASS", doc["uncovered"]
    assert "metrics" not in doc["dimensions"], "a lane declares no metric"
    assert doc["gates"] == {}


def test_on_release_the_golden_lanes_motif_is_deferred_not_covered():
    doc = cr.Report(SKILLS, None, "release", True).build()
    cell = doc["dimensions"]["motifs"]["entries"]["lanes"]
    assert (cell["covered"], cell["deferred"], cell["declared_by"]) == (False, "nightly", ["G00"])
    assert any("motifs:lanes" in w and "nightly" in w for w in doc["warnings"])
    assert doc["verdict"] == "PASS"
    # the verdict states are a must dimension: no release lane earns its cost, which the report
    # says (G00, nightly, does) instead of leaving the state unmentioned (gate C)
    earned = doc["dimensions"]["verdicts"]["entries"]["earned"]
    assert (earned["covered"], earned["deferred"], earned["declared_by"]) == (False, "nightly", ["G00"])
    assert doc["dimensions"]["verdicts"]["must"] and doc["dimensions"]["verdicts"]["proved_at"] == "page"


def test_a_verdict_state_no_lane_reaches_fails_the_report(tmp_path):
    docs = release_docs()
    for doc in docs.values():
        if doc["observed"].get("verdict") == "not_assessable":
            doc["observed"]["verdict"] = "beaten_by_baseline"
    report = build(write_matrix(tmp_path, docs))
    assert report["verdict"] == "FAIL"
    assert any(u["dimension"] == "verdicts" and u["entry"] == "not_assessable" for u in report["uncovered"])


# --------------------------------------------------------------------------- 2. a green matrix
def test_a_green_release_matrix_passes_and_renders(tmp_path):
    doc = build(write_matrix(tmp_path, release_docs()))
    assert doc["verdict"] == "PASS", doc["uncovered"][:5]
    assert doc["schema"] == "vega/coverage-report@1"
    assert doc["dimensions"]["families"]["entries"]["gin"] == {"covered": True, "lanes": ["L22"]}
    assert doc["cross"]["family_x_task"]["gru"] == ["forecast"]
    assert all(g["covered"] for t in doc["gates"].values() for g in t.values() if g["new"])
    md = coverage_md.render(doc)
    for needle in ("# Coverage — PASS", "### families — 21/21", "## Family × task", "## Gates", "### vega_show"):
        assert needle in md
    assert cr.say(doc) == 0


def test_the_cli_writes_both_files_and_exits_on_the_verdict(tmp_path):
    root = write_matrix(tmp_path / "m", release_docs())
    out = tmp_path / "out"
    assert cr.main(["--matrix", str(root), "--tier", "release", "--json", str(out / "c.json"),
                    "--md", str(out / "c.md"), "--skills", str(SKILLS)]) == 0
    assert json.loads((out / "c.json").read_text())["verdict"] == "PASS"
    docs = release_docs(L22={"last": "page"})
    assert cr.main(["--matrix", str(write_matrix(tmp_path / "red", docs)), "--tier", "release"]) == 1


# --------------------------------------------------------------------------- 3. where a lane stopped
def test_a_lane_that_failed_before_page_does_not_prove_its_family(tmp_path):
    doc = build(write_matrix(tmp_path, release_docs(L22={"last": "inspect"})))
    assert doc["verdict"] == "FAIL"
    gin = next(u for u in doc["uncovered"] if (u["dimension"], u["entry"]) == ("families", "gin"))
    assert "L22 stopped at inspect fail" in gin["why"]
    graph_level = doc["dimensions"]["graph_levels"]["entries"]["graph"]
    assert not graph_level["covered"] and "L22" in graph_level["stopped"]
    # the train-proved dimensions still hold: L22 passed train
    assert doc["dimensions"]["losses"]["entries"]["bce_with_logits"]["covered"]


def test_a_stage_that_ran_without_a_check_proves_nothing_past_itself(tmp_path):
    docs = release_docs()
    docs["L20"] = observed_of(LANES["L20"], not_run={"eval": ["the seal re-read — provider absent"]})
    doc = build(write_matrix(tmp_path, docs))
    gat = doc["dimensions"]["families"]["entries"]["gat"]
    assert not gat["covered"]
    assert "passed without running" in gat["stopped"]["L20"]
    assert doc["dimensions"]["losses"]["entries"]["cross_entropy"]["covered"], "train passed and ran everything"


def test_a_lane_without_observed_json_is_said(tmp_path):
    docs = release_docs()
    del docs["L22"]
    doc = build(write_matrix(tmp_path, docs))
    gin = next(u for u in doc["uncovered"] if u["entry"] == "gin")
    assert "declared by L22, which left no observed.json" in gin["why"]


def test_a_lane_whose_observation_lacks_what_it_declares_is_said(tmp_path):
    docs = release_docs()
    docs["L04"]["observed"]["showcase"] = None          # a stage that wrote no show result
    doc = build(write_matrix(tmp_path, docs))
    grid = next(u for u in doc["uncovered"] if (u["dimension"], u["entry"]) == ("motifs", "grid"))
    assert "the observed.json of L04 does not hold it (observed motifs: none)" in grid["why"]


# --------------------------------------------------------------------------- 4. report-only, unknown
def test_metrics_only_warn_and_an_entry_the_catalog_lost_fails(tmp_path):
    docs = release_docs()
    docs["L05"]["observed"]["blocks"].append("hyperconv")
    doc = build(write_matrix(tmp_path, docs))
    assert any(u["entry"] == "hyperconv" and "does not hold" in u["why"] for u in doc["uncovered"])
    assert doc["dimensions"]["metrics"]["must"] is False
    assert not any(u["dimension"] == "metrics" for u in doc["uncovered"])
    assert any(w.startswith("WARN: metrics:") for w in doc["warnings"])


# --------------------------------------------------------------------------- 5. gates
def _cases(*specs, suite="vega-forge/scripts/tests/run_negatives.py"):
    cases = [{"id": i, "name": n or f"{i}  x", "ok": ok, "detail": "" if ok else "missed"} for i, n, ok in specs]
    doc = {"schema": "vega/negatives-result@1", "script": suite, "work": "/w", "cases": cases,
           "passed": sum(1 for c in cases if c["ok"]), "failed": sum(1 for c in cases if not c["ok"])}
    out = coverage_gates.Cases(suites={suite})
    for c in cases:
        (out.passing if c["ok"] else out.failing).append(coverage_gates.Case(c["id"], c["name"], suite))
    return out, doc


def _table(name):
    return next(t for t in coverage_gates.gate_tables(SKILLS) if t.name == name)


def test_a_lettered_case_covers_its_gate_and_never_a_shorter_id():
    cases, _ = _cases(("C11a", "", True))
    cover = coverage_gates.table_cover(_table("conformance"), cases, {})
    assert cover["C11"]["covered"] and cover["C11"]["cases"] == ["C11a"]
    assert not cover["C1"]["covered"] and not cover["C1"]["new"]


def test_a_grandfathered_family_key_covers_its_lettered_rules():
    legacy = {"vega-forge/compile_spec.py#T1": ["T1  a monitor on train is refused"]}
    cases, _ = _cases((None, "T1  a monitor on train is refused", True))
    cover = coverage_gates.table_cover(_table("compile_spec"), cases, legacy)
    assert cover["T1a"]["covered"] and cover["T1d"]["covered"]
    assert cover["T1a"]["grandfathered"]["keys"] == ["T1"]
    assert not cover["T1e"]["covered"], "a new rule is not covered by the old family's cases"


def test_a_grandfathered_case_counts_for_its_own_table_only():
    legacy = {"vega-forge/compile_spec.py#C1": ["C1  a vega:default citing a catalog row that does not exist"]}
    cases, _ = _cases(("C1", "C1  a vega:default citing a catalog row that does not exist", True))
    assert not coverage_gates.table_cover(_table("conformance"), cases, legacy)["C1"]["covered"]


def test_a_show_gate_is_covered_by_a_case_its_row_names_from_any_suite():
    row = next(r for r in SHOW["gates"] if r["cases"] and r["cases"][0].startswith("SS"))
    cases, _ = _cases((row["cases"][0], "", True), suite="vega-pack/scripts/tests/run_negatives_showcase.py")
    assert coverage_gates.table_cover(_table("vega_show"), cases, {})[row["id"]]["covered"]


def test_a_failing_case_covers_nothing_and_a_missing_suite_is_named():
    cases, _ = _cases(("C12a", "", False))
    gates, uncovered, _ = coverage_gates.cover([_table("conformance"), _table("shaper_graph")], cases, {})
    assert not gates["conformance"]["C12"]["covered"]
    sg = next(u for u in uncovered if u["entry"] == "SG1")
    assert "no result from dataset-shaper/scripts/tests/run_negatives_graph.py" in sg["why"]


def test_a_malformed_result_is_a_problem_not_a_pass(tmp_path):
    _, doc = _cases(("C11a", "", True))
    doc["passed"] = 7
    (tmp_path / "bad.json").write_text(json.dumps(doc), encoding="utf-8")
    read = coverage_gates.read_negatives(tmp_path)
    assert read.passing == [] and read.problems and "bad.json" in read.problems[0]


def test_case_ids_are_read_from_names_when_the_suite_wrote_none():
    assert coverage_gates.case_id({"id": None, "name": "SG5b an edges part edited"}) == "SG5b"
    assert coverage_gates.case_id({"id": None, "name": "D5-21 the digit rule"}) == "D5-21"
    assert coverage_gates.case_id({"id": None, "name": "rules: a second step"}) is None


def test_the_rule_tables_are_read_from_the_code():
    assert {"T1e", "G2", "W2", "W3", "R2"} <= set(_table("compile_spec").ids) - coverage_gates.COMPILE_SPEC_BASELINE
    assert _table("conformance").ids == tuple(f"C{i}" for i in range(15))
    assert len(_table("vega_show").ids) == len(SHOW["gates"])


# --------------------------------------------------------------------------- 6. pr
def test_on_pr_the_lanes_that_did_not_run_stand_in_by_declaration(tmp_path):
    pr = {l.id: observed_of(l) for l in registry.lanes_for(LANES, None, "pr")}
    doc = build(write_matrix(tmp_path, pr), tier="pr")
    assert doc["verdict"] == "PASS", doc["uncovered"][:4]
    cell = doc["dimensions"]["families"]["entries"]["bi-gru"]
    assert cell["covered"] and cell["declared"] == ["L13"]
    on_release = build(tmp_path, tier="release")
    assert on_release["verdict"] == "FAIL", "on release a lane that did not run proves nothing"
