#!/usr/bin/env python3
"""
test_campaign_integrity.py — what makes a campaign a record nobody can bend (A46 track T).

    python3 -m pytest test_campaign_integrity.py [-q]

1. **The expectation is a check** (`expectations.py`): the closed grammar parses, refuses what
   is outside it, and judges in the metric's direction — the primary by the selection mode, a
   loss and the train/valid gap downwards — with the margin as a fraction of the reference; a
   check whose reference has not run is pending (null) with the reason, never a guess; a
   pending row settles when its reference lands (`rejudge`), and a study's record is judged
   on its best trial.
2. **A dead delta is named** (`live_paths.py`): from `spec_snapshot.json`'s `read_paths` and
   `_resolved.regularization.shadowed_paths` as the forge's R2 writes them — a declaration no
   code reads, a shadowed live path, a whole-list edit that changes a dead kind, an index
   path; a live declaration and a path outside R2's domain are not dead; a snapshot from
   before R2 is judged by the pre-R2 templates (only grad_clip of the declaration is read),
   a directory with no snapshot by the forge's own derivation on its netspec.json.
3. **plan_ladder plans once**: a directory holding `campaign.json` (or a `ladder.json`) is
   refused and left byte for byte as it was; a plan that fails writes neither record; a dead
   default-test rung refuses the plan (`ERROR: dead delta — …`, nothing written), a dead
   search dimension is dropped, and the same rung is kept when the snapshot says the code
   reads it; every rung written carries its rule's `expect_check` with its text.
4. **The ledger's verdict is the check's**: `apply_training_layer.expect_problems` refuses a
   flipped `expect_held`, a rung or a study without a check, and passes a ledger the trainer
   wrote; a trial's row inherits and is not judged on its own.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
SPIKE = SKILLS / "vega-forge" / "scripts" / "tests" / "spike"
FORGE = SKILLS / "vega-forge" / "scripts"
for _p in (str(SCRIPTS), str(SPIKE), str(FORGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import apply_training_layer as ATL  # noqa: E402
import live_paths as LP  # noqa: E402
import plan_ladder as PL  # noqa: E402
from expectations import ExpectCheck, ExpectError, check_for, judge_row, rejudge  # noqa: E402
from vega_train_core import Delta, load_catalog  # noqa: E402

CATALOG = load_catalog()[0]
SPEC_MAX = {"metrics": {"primary": "pr_auc", "selection": {"mode": "max"}}}
SPEC_MIN = {"metrics": {"primary": "mae", "selection": {"mode": "min"}}}


def _row(run_id, primary, gap=0.1, valid_loss=0.5, best_epoch=5, diagnostics=(),
         name="pr_auc"):
    return {"id": run_id, "primary_name": name, "primary": primary,
            "valid": {name: primary, "loss": valid_loss}, "train": {"loss": valid_loss - gap},
            "gap": {"loss": gap}, "best_epoch": best_epoch, "diagnostics": list(diagnostics)}


# --------------------------------------------------------------------------- 1. the grammar

@pytest.mark.parametrize("raw, needle", [
    ({"relation": "about_the_same", "than": "R0"}, "is not one of"),
    ({"relation": "worse"}, "names the run it compares with"),
    ({"relation": "worse", "than": "R0", "metric": "valid.f1"}, "metric 'valid.f1'"),
    ({"relation": "worse", "than": "R0", "margin": -0.1}, "is not a fraction >= 0"),
    ({"relation": "worse", "than": "R0", "when": "always"}, "the grammar does not have"),
    ({"relation": "learns", "than": "R0"}, "compares with nothing"),
    ({"any": []}, "at least one check"),
    ("valid pr_auc below R0", "is an object"),
])
def test_a_check_outside_the_grammar_is_refused(raw, needle):
    with pytest.raises(ExpectError) as exc:
        ExpectCheck.parse(raw)
    assert needle in str(exc.value)


def test_a_check_never_compares_a_run_with_itself():
    with pytest.raises(ExpectError, match="compares R0 with itself"):
        ExpectCheck.parse({"relation": "worse", "than": "R0"}, run_id="R0")


def test_the_primary_is_judged_in_the_selection_mode():
    rows = [_row("R0", 0.70), _row("R1", 0.60)]
    worse = ExpectCheck.parse({"relation": "worse", "than": "R0", "margin": 0.05})
    assert judge_row(worse, rows[1], rows, SPEC_MAX)[0] is True       # 0.60 < 0.70 − 0.035
    assert judge_row(worse, _row("R1", 0.68), rows, SPEC_MAX)[0] is False   # inside the band
    # a minimised primary: a larger error is the worse one
    rows_min = [_row("R0", 2.0, name="mae"), _row("R1", 3.0, name="mae")]
    assert judge_row(worse, rows_min[1], rows_min, SPEC_MIN)[0] is True
    better = ExpectCheck.parse({"relation": "better", "than": "R0"})
    assert judge_row(better, rows_min[1], rows_min, SPEC_MIN)[0] is False


def test_a_wider_gap_is_worse_and_any_is_a_disjunction():
    check, _ = check_for(None, "vega:ladder/default-test", CATALOG, "R2")
    rows = [_row("R0", 0.70, gap=0.05)]
    held, ev = judge_row(check, _row("R2", 0.71, gap=0.20), rows, SPEC_MAX)   # gap widened
    assert held is True and "gap.loss" in ev
    held, ev = judge_row(check, _row("R2", 0.75, gap=0.01), rows, SPEC_MAX)   # neither
    assert held is False and "did not hold" in ev


def test_the_baseline_learns_unless_it_diverged_or_never_improved():
    check, _ = check_for(None, "netspec", CATALOG, "R0")
    assert judge_row(check, _row("R0", 0.7), [], SPEC_MAX)[0] is True
    assert judge_row(check, _row("R0", 0.7, diagnostics=["D1"]), [], SPEC_MAX)[0] is False
    assert judge_row(check, _row("R0", 0.7, best_epoch=1), [], SPEC_MAX)[0] is False
    held, ev = judge_row(check, _row("R0", 0.7, best_epoch=None), [], SPEC_MAX)
    assert held is None and "no best epoch" in ev


def test_a_check_whose_reference_has_not_run_is_pending_then_settles():
    floor = check_for(None, "vega:ladder/floor", CATALOG, "R1")[0]
    ledger = {"runs": [dict(_row("R1", 0.5), expect_check=floor.as_json("pr_auc"),
                            expect_held=None)]}
    assert rejudge(ledger, SPEC_MAX) == 0
    assert ledger["runs"][0]["expect_held"] is None
    assert "R0 has not run yet" in ledger["runs"][0]["expect_evidence"]
    ledger["runs"].append(dict(_row("R0", 0.7), expect_check={"relation": "learns"},
                               expect_held=None))
    assert rejudge(ledger, SPEC_MAX) == 2
    assert [r["expect_held"] for r in ledger["runs"]] == [True, True]


def test_a_study_is_judged_on_its_best_trial():
    ledger = {"runs": [_row("R0", 0.70)],
              "hpo": [{"study": "H1", "best": "H1-3", "best_value": 0.66,
                       "expect_check": {"relation": "not_worse", "than": "R0"},
                       "expect_held": None}]}
    rejudge(ledger, SPEC_MAX)
    assert ledger["hpo"][0]["expect_held"] is False
    assert "H1 0.66 against R0 0.7" in ledger["hpo"][0]["expect_evidence"]


def test_every_rule_has_a_default_and_a_user_rung_states_its_own():
    for source in ("netspec", "netspec:optimizer/alternatives[0]", "vega:ladder/floor",
                   "vega:ladder/recurrent-floor", "vega:ladder/default-test",
                   "vega:ladder/capacity", "vega:ladder/hpo", "train:diagnostic/D4"):
        check, why = check_for(None, source, CATALOG, "RX")
        assert check is not None, (source, why)
    check, why = check_for(None, "user:turn3", CATALOG, "RX")
    assert check is None and "states no expect_check" in why
    check, why = check_for({"relation": "better", "than": "R0"}, "user:turn3", CATALOG, "RX")
    assert check is not None


# --------------------------------------------------------------------------- 2. dead deltas

#: a spec as R2 sees it: weight decay declared (its optimizer copy shadowed), a dropout that
#: covers B3, a label smoothing under a loss that takes none, gradient clipping
SPEC = {"metrics": {"primary": "pr_auc", "selection": {"mode": "max"}},
        "optimizer": {"name": "adamw", "params": {"lr": 0.001, "weight_decay": 0.0001}},
        "loss": {"name": "bce_with_logits", "params": {}},
        "blocks": [{"id": "B3", "type": "dropout", "params": {"p": 0.2}}],
        "regularization": [{"kind": "weight_decay", "value": 0.0001},
                           {"kind": "dropout", "value": 0.2, "where": ["B3"]},
                           {"kind": "label_smoothing", "value": 0.1},
                           {"kind": "grad_clip", "value": 1.0}]}
#: what the forge's R2 records for it (vega_regularization.snapshot's shape)
SNAPSHOT = {"read_paths": ["regularization/0/value", "regularization/1/value",
                           "regularization/3/value", "regularization[kind=dropout]/value",
                           "regularization[kind=grad_clip]/value",
                           "regularization[kind=weight_decay]/value"],
            "_resolved": {"regularization": {
                "read_paths": [], "live_values": {},
                "shadowed_paths": ["blocks[id=B3]/params/p", "optimizer/params/weight_decay",
                                   "training/grad_clip"]}}}


def _dead(raw_delta, snapshot=SNAPSHOT):
    return LP.dead_edits(Delta.of(raw_delta), SPEC, LP.of_snapshot(snapshot))


def test_a_read_declaration_is_live_and_other_sections_are_not_judged():
    assert _dead([{"path": "regularization[kind=weight_decay]/value", "to": 0.0}]) == []
    assert _dead([{"path": "regularization[kind=dropout]/value", "to": 0.4}]) == []
    assert _dead([{"path": "optimizer/params/lr", "to": 0.01}]) == []
    assert _dead([{"path": "blocks/B3", "to": {"id": "B3", "type": "dropout"}}]) == []


@pytest.mark.parametrize("delta, needle", [
    ([{"path": "regularization[kind=label_smoothing]/value", "to": 0.0}],
     "no emitted code reads regularization[kind=label_smoothing]/value"),
    ([{"path": "regularization/2/value", "to": 0.0}], "no emitted code reads regularization/2"),
    ([{"path": "optimizer/params/weight_decay", "to": 0.0}],
     "a regularization[] declaration covers it"),
    ([{"path": "blocks/B3/params/p", "to": 0.5}], "the change belongs on the declaration"),
    ([{"path": "regularization", "to": [{"kind": "weight_decay", "value": 0.0001},
                                        {"kind": "dropout", "value": 0.2, "where": ["B3"]},
                                        {"kind": "label_smoothing", "value": 0.3},
                                        {"kind": "grad_clip", "value": 1.0}]}],
     "label_smoothing changed by the edit on regularization"),
])
def test_a_dead_delta_is_named(delta, needle):
    dead = _dead(delta)
    assert dead and needle in dead[0][1], dead


def test_the_forges_own_r2_record_reads_the_same():
    """The record `vega-forge/scripts/vega_regularization.snapshot` writes into
    spec_snapshot.json (top-level `read_paths`, `_resolved.regularization`): every path it
    reads is live here, every path it shadows is dead, and so is the declaration it reads not."""
    import vega_regularization as VR
    snapshot = {"read_paths": VR.read_paths(SPEC), "_resolved": {"regularization":
                                                                 VR.snapshot(SPEC)}}
    paths = LP.of_snapshot(snapshot)
    assert paths.recorded
    for read in VR.read_paths(SPEC):
        assert LP.dead_path(read, paths, spec=SPEC) is None, read
    for shadowed in VR.shadowed_paths(SPEC):
        assert LP.dead_path(shadowed, paths, spec=SPEC), shadowed
    assert LP.dead_path("regularization[kind=label_smoothing]/value", paths, spec=SPEC)
    assert LP.dead_path("regularization[kind=weight_decay]/value", paths, spec=SPEC) is None


def test_a_directory_without_a_snapshot_is_judged_by_the_forges_derivation(tmp_path):
    (tmp_path / "netspec.json").write_text(json.dumps(SPEC), encoding="utf-8")
    paths = LP.of_campaign(tmp_path)
    assert not paths.recorded and "R2 derivation on netspec.json" in paths.source
    assert LP.dead_path("regularization[kind=weight_decay]/value", paths, spec=SPEC) is None
    assert LP.dead_path("optimizer/params/weight_decay", paths, spec=SPEC)
    assert LP.dead_path("regularization[kind=label_smoothing]/value", paths, spec=SPEC)


def test_a_snapshot_before_r2_reads_only_grad_clip_of_the_declaration():
    before = LP.of_snapshot({"_resolved": {}})
    assert not before.recorded and "pre-R2" in before.source
    assert _dead([{"path": "regularization[kind=grad_clip]/value", "to": None}], {}) == []
    assert _dead([{"path": "regularization[kind=weight_decay]/value", "to": 0.0}], {})
    assert _dead([{"path": "optimizer/params/weight_decay", "to": 0.0}], {}) == []


def test_a_dotted_or_mapped_record_reads_the_same():
    mapped = LP.of_snapshot({"read_paths": {"regularization[kind=weight_decay]/value":
                                            ["optimizer.params.weight_decay"]}})
    assert "optimizer/params/weight_decay" in mapped.derived
    dead = LP.dead_edits(Delta.of([{"path": "optimizer/params/weight_decay", "to": 0.0}]),
                         SPEC, mapped)
    assert dead and "derives it from regularization[kind=weight_decay]/value" in dead[0][1]


# --------------------------------------------------------------------------- 3. plan_ladder

@pytest.fixture(scope="module")
def window(tmp_path_factory):
    import make_spike_fixtures as msf
    return msf.make("window", tmp_path_factory.mktemp("spike"), "full")


def _stub_campaign(fixture, root: Path, spec_edit=None, snapshot=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(fixture.spec).read_text(encoding="utf-8"))
    if spec_edit:
        spec_edit(spec)
    (root / "netspec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    shutil.copytree(fixture.package_dir, root / fixture.package)
    (root / "model.html").write_text("<!doctype html><html><head><title>stub</title></head>"
                                     "<body></body></html>", encoding="utf-8")
    if snapshot is not None:
        (root / "spec_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    return root


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}


def _plan(camp, *extra):
    said = []
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = PL.main(["--campaign", str(camp), "--model", str(camp / "model.html"),
                        "--budget", "3", *extra])
    said = buf.getvalue()
    return code, said


def _with_dropout_default(spec):
    """The window spec with a dropout declaration the forge marked a default to test."""
    spec["regularization"] = [r for r in spec.get("regularization") or []
                              if r.get("kind") != "dropout"]
    spec["regularization"].append({"kind": "dropout", "value": 0.2, "source": "vega:default",
                                   "catalog": "defaults/dropout", "rationale": "r",
                                   "alternatives": [{"value": 0.4, "when": "train >> valid"}]})


def test_a_directory_holding_a_campaign_is_refused_and_left_as_it_was(window, tmp_path):
    camp = _stub_campaign(window, tmp_path / "camp")
    code, said = _plan(camp, "--no-hpo", "--unattended")
    assert code == 0, said
    before = _tree(camp)
    code, said = _plan(camp, "--no-hpo", "--unattended")
    assert code == 2
    assert "already holds campaign.json" in said and "--resume" in said
    assert _tree(camp) == before
    (camp / "campaign.json").unlink()                     # a ladder nobody planned here
    code, said = _plan(camp, "--no-hpo")
    assert code == 2 and "already exists without a campaign.json" in said


def test_a_failed_plan_writes_neither_record(window, tmp_path):
    camp = _stub_campaign(window, tmp_path / "camp")
    code, said = _plan(camp, "--rules", "capacity", "--no-hpo")   # nothing to resize here
    assert code == 2 and "every rung was dropped" in said
    assert not (camp / "campaign.json").exists() and not (camp / "ladder.json").exists()
    code, said = _plan(camp, "--no-hpo", "--unattended")          # and it is plannable still
    assert code == 0, said


def test_every_rung_written_carries_its_rules_expect_check(window, tmp_path):
    camp = _stub_campaign(window, tmp_path / "camp")
    code, said = _plan(camp, "--hpo", "3", "--unattended")
    assert code == 0, said
    ladder = json.loads((camp / "ladder.json").read_text(encoding="utf-8"))
    for rung in ladder["runs"]:
        check = rung.get("expect_check")
        assert check and check.get("text"), rung["id"]
        ExpectCheck.parse(check, rung["id"])
    by_id = {r["id"]: r for r in ladder["runs"]}
    assert by_id["R0"]["expect_check"]["relation"] == "learns"
    assert by_id["R1"]["expect_check"]["relation"] == "worse"
    assert by_id["H1"]["expect_check"]["relation"] == "not_worse"


def test_a_dead_default_refuses_the_plan_and_a_live_one_is_kept(window, tmp_path):
    entry = {"path": "regularization[kind=dropout]/value", "observation": "o"}
    spec = json.loads(Path(window.spec).read_text(encoding="utf-8"))
    _with_dropout_default(spec)
    dead_said = []
    camp = _stub_campaign(window, tmp_path / "dead", _with_dropout_default,
                          snapshot={"read_paths": ["regularization[kind=grad_clip]/value"]})
    ladder, err = PL.plan(camp, spec, {"defaults_to_test": [entry]}, CATALOG, budget_min=3,
                          unattended=False, rules={"baseline", "default-test"}, hpo_trials=0,
                          out=dead_said.append)
    assert ladder is None, dead_said
    assert err.startswith("dead delta — no emitted code reads regularization[kind=dropout]")
    live_said = []
    camp = _stub_campaign(window, tmp_path / "live", _with_dropout_default,
                          snapshot={"read_paths": ["regularization[kind=dropout]/value"]})
    ladder, err = PL.plan(camp, spec, {"defaults_to_test": [entry]}, CATALOG, budget_min=3,
                          unattended=False, rules={"baseline", "default-test"}, hpo_trials=0,
                          out=live_said.append)
    assert err is None, live_said
    assert [r["delta"][0]["path"] for r in ladder["runs"][1:]] == [entry["path"]]


def test_the_cli_refuses_a_dead_delta_and_writes_nothing(window, tmp_path):
    camp = _stub_campaign(window, tmp_path / "camp",
                          snapshot={"read_paths": ["regularization[kind=grad_clip]/value"]})
    model = camp / "model.html"
    layer = {"defaults_to_test": [{"path": "regularization[kind=weight_decay]/value",
                                   "observation": "o"}]}
    model.write_text('<!doctype html><html><head><title>stub</title></head><body>'
                     '<script id="layer-architecture-data" type="application/json">'
                     + json.dumps(layer) + '</script></body></html>', encoding="utf-8")
    code, said = _plan(camp, "--no-hpo", "--rules", "baseline,default-test")
    assert code == 2, said
    assert "ERROR: dead delta — no emitted code reads regularization[kind=weight_decay]/value" \
        in said
    assert not (camp / "campaign.json").exists() and not (camp / "ladder.json").exists()


def test_a_dead_search_dimension_is_dropped_from_the_space(window):
    spec = json.loads(Path(window.spec).read_text(encoding="utf-8"))
    _with_dropout_default(spec)
    dead = LP.of_snapshot({"read_paths": ["regularization[kind=grad_clip]/value"]})
    space, dropped = PL.build_space(spec, CATALOG, read=dead)
    assert "regularization[kind=dropout]/value" not in [d["path"] for d in space["dimensions"]]
    assert any("dead delta" in note for note in dropped), dropped
    live = LP.of_snapshot({"read_paths": ["regularization[kind=dropout]/value"]})
    space, _ = PL.build_space(spec, CATALOG, read=live)
    assert "regularization[kind=dropout]/value" in [d["path"] for d in space["dimensions"]]


# --------------------------------------------------------------------------- 4. the verdict

def _ledger():
    rows = [dict(_row("R0", 0.70), kind="baseline", source="netspec"),
            dict(_row("R1", 0.50), kind="ablation", source="vega:ladder/floor"),
            dict(_row("H1-1", 0.72), kind="hpo", source="train:hpo/H1", expect_check=None,
                 expect_held=None)]
    rows[0]["expect_check"] = {"relation": "learns"}
    rows[1]["expect_check"] = {"relation": "worse", "than": "R0", "margin": 0.05}
    for r in rows[:2]:
        r["expect_held"] = None
    ledger = {"runs": rows, "ranking": {"by": "valid.pr_auc", "mode": "max"},
              "hpo": [{"study": "H1", "best_value": 0.72, "expect_held": None,
                       "expect_check": {"relation": "not_worse", "than": "R0"}}]}
    rejudge(ledger, SPEC_MAX)
    return ledger


def test_the_ledger_the_scripts_wrote_passes():
    ledger = _ledger()
    assert [r["expect_held"] for r in ledger["runs"]] == [True, True, None]
    assert ATL.expect_problems(ledger) == []


def test_a_flipped_verdict_is_refused():
    ledger = _ledger()
    ledger["runs"][1]["expect_held"] = False
    problems = ATL.expect_problems(ledger)
    assert len(problems) == 1 and "'R1' says expect_held false" in problems[0]
    assert "never edited" in problems[0]


def test_a_rung_or_study_without_a_check_is_refused():
    ledger = _ledger()
    ledger["runs"][0].pop("expect_check")
    ledger["hpo"][0]["expect_check"] = None
    problems = " | ".join(ATL.expect_problems(ledger))
    assert "'R0' carries no expect_check" in problems and "'H1' carries no" in problems


def test_a_ledger_without_a_ranking_mode_cannot_be_checked():
    ledger = _ledger()
    ledger.pop("ranking")
    assert "carries no mode" in ATL.expect_problems(ledger)[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------- the seal's exception

def _graph_spec(protocol, exposure=None):
    sample = {"shape": "graph", "level": "node", "protocol": protocol, "exposure": exposure}
    return {"data_contract": {"sample": sample}}


def test_a_transductive_campaign_discloses_its_test_feature_exposure(tmp_path):
    """run-contract §1: under `transductive` the test nodes' features are in the training graph;
    campaign.json carries the spec's declared exposure, and the apply gate refuses a layer that
    drops it — or one that claims an exposure an inductive campaign never took."""
    declared = {"kind": "features", "rows": "test", "source": "user:turn3"}
    assert PL.test_feature_exposure(_graph_spec("inductive")) is None
    assert PL.test_feature_exposure({"data_contract": {"sample": {"shape": "row"}}}) is None
    record = PL.test_feature_exposure(_graph_spec("transductive", declared))
    assert record["protocol"] == "transductive" and record["declared"] == declared
    (tmp_path / "netspec.json").write_text(json.dumps(_graph_spec("transductive", declared)))
    camp = {"run_dir": str(tmp_path), "exposure": record}
    assert ATL.exposure_problems(camp) == []
    assert "transductive" in " ".join(ATL.exposure_problems(dict(camp, exposure=None)))
    assert "does not repeat" in " ".join(ATL.exposure_problems(
        dict(camp, exposure=dict(record, declared={"kind": "features"}))))
    (tmp_path / "netspec.json").write_text(json.dumps(_graph_spec("inductive")))
    assert ATL.exposure_problems(dict(camp, exposure=None)) == []
    assert "not transductive" in " ".join(ATL.exposure_problems(camp))
    assert ATL.exposure_problems({"exposure": None}) == []


def test_the_mechanics_are_spliced_beside_an_authored_one_step(tmp_path):
    """The T5 round 3 rehearsal: the authoring file carried `mechanics.one_step` (SKILL step 5
    asks for it) and the apply spliced the loss, the update rule and the schedule only when the
    authoring file held no `mechanics` at all — the layer shipped without them. The campaign's
    mechanics are spliced whatever the authoring file holds; its `one_step` is kept."""
    camp = tmp_path / "camp"
    (camp / "runs" / "R0").mkdir(parents=True)
    for name, doc in (("campaign.json", {"schema": "vega/campaign@1"}), ("ladder.json", {"runs": []}),
                      ("experiments.json", {"runs": [], "selection": {"run": "R0"}})):
        (camp / name).write_text(json.dumps(doc), encoding="utf-8")
    (camp / "netspec.json").write_text(json.dumps({
        "loss": {"name": "cross_entropy", "formula": "L = -log p_y", "params": {}},
        "optimizer": {"name": "adamw", "update_rule": "AdamW: θ ← θ − η m̂/(√v̂+ε)", "params": {}},
        "schedule": {"name": "none", "params": {}}, "metrics": {"primary": "f1_macro"}}),
        encoding="utf-8")
    (camp / "runs" / "R0" / "first_step.json").write_text(json.dumps({"head": "B9"}),
                                                       encoding="utf-8")
    lines = []
    doc, problems = ATL.splice({"from_campaign": str(camp), "mechanics": {"one_step": "mine"}},
                               lines.append, tmp_path)
    assert not problems, problems
    mech = doc["mechanics"]
    assert mech["loss"] == "L = -log p_y" and mech["update_rule"].startswith("AdamW")
    assert mech["one_step"] == "mine" and mech["one_step_record"] == {"head": "B9"}
