"""
test_closure_e_eval.py — the contract promises /vega-eval now keeps in code (PLAN-AMENDMENTS A46,
track E), each proved without a trained campaign; `run_negatives.py` seeds each against one.

    python3 -m pytest -q vega-eval/scripts/tests/test_closure_e_eval.py

  * **T3 — a user slice's literal is coerced to the column's dtype** (eval-contract §2): the
    dtype the shaper's manifest records, else the train part's, so `region=4` measures a code the
    design matrix holds as 4.0 (the text comparison lost it); what can be checked without a test
    row — no value, not a feature, a literal the dtype cannot hold — is refused before the seal.
  * **S3 — an after-eval campaign's first look is a further exposure** (§1): the threshold value,
    the slices and a narrowed baseline set are refused before the seal, the record of earlier
    looks must reconcile with the previous campaign's seal, and the new seal counts them all.
  * **B5 — `campaign-baseline` is predicted through the seal** (§4): what is checked before the
    child process runs (an R0 row, its checkpoint, its digest), and that it is a compared model
    and never the verdict's strongest fitted baseline.
  * **M1 — `mape`'s gate** (§3): |y| > 0.1 on 99 % of the train rows, decided on train.
  * **B6 — `lens` carries the analysis's recorded hyper-parameters** (§4), on the rendered
    template, with the ones that did not carry named.
  * **A1–A3 — the apply script** (§6): a `threshold.metrics_at_threshold` metric without a CI, a
    test number marked `_upstream`, and a gated metric published with a value are refused.
"""
from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
FORGE_TESTS = SCRIPTS.parents[1] / "vega-forge" / "scripts" / "tests"
sys.path.insert(0, str(SCRIPTS))

import apply_evaluation_layer as apply  # noqa: E402
import campaign_baseline  # noqa: E402
import earlier_looks  # noqa: E402
import evaluate  # noqa: E402
import metric_gates  # noqa: E402
import user_slices as US  # noqa: E402
from eval_package import lens_protocol, task_classes  # noqa: E402
from vega_eval_core import Resampler, Scoring, Seal  # noqa: E402


def _args(**kw):
    base = {"again": None, "threshold": None, "slices": None, "baselines": None}
    base.update(kw)
    return types.SimpleNamespace(**base)


def _write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


LEDGER = {"runs": [{"id": "R0"}, {"id": "H1-3"}], "selection": {"run": "H1-3"}}
BASELINES = ("trivial", "linear", "tree", "lens", "campaign-baseline")
DTYPES = {"region": "Int64", "dist_km": "float64", "promo": "boolean", "zip": "object"}
FEATURES = ["region", "dist_km", "promo", "zip"]


# =========================================================================== T3: user slices

@pytest.mark.parametrize("dtype, family", [
    ("Int64", "integer"), ("int64", "integer"), ("uint8", "integer"), ("float64", "float"),
    ("double", "float"), ("boolean", "bool"), ("bool", "bool"), ("datetime64[ns]", "datetime"),
    ("timestamp[ns]", "datetime"), ("object", "string"), ("large_string", "string")])
def test_a_dtype_is_read_in_pandas_and_arrow_words(dtype, family):
    assert US.dtype_family(dtype) == family


@pytest.mark.parametrize("literal, dtype, want", [
    ("4", "Int64", 4), ("4.0", "int64", 4), ("1", "float64", 1.0), ("0.25", "double", 0.25),
    ("true", "boolean", True), ("0", "bool", False), ("007", "object", "007")])
def test_the_literal_is_coerced_to_the_column_dtype(literal, dtype, want):
    got = US.coerce(literal, dtype)
    assert got == want and type(got) is type(want)


@pytest.mark.parametrize("literal, dtype", [("4.5", "Int64"), ("abc", "float64"),
                                            ("nan", "float64"), ("maybe", "boolean")])
def test_a_literal_the_dtype_cannot_hold_is_refused(literal, dtype):
    with pytest.raises(ValueError):
        US.coerce(literal, dtype)


def test_a_coerced_slice_matches_the_value_the_design_matrix_holds():
    """The K2 symptom: the design matrix holds every column as float64, so a code 4 reads 4.0;
    a text comparison (`str(4.0) == "4"`) lost the slice, a coerced one measures it."""
    frame = pd.DataFrame({"region": [4.0, 4.0, 1.0, 4.0], "zip": ["007", "7", "007", "x"]})
    (region, zipped), problems = US.declare(["region=4", "zip=007"], DTYPES, FEATURES)
    assert problems == ()
    assert (frame["region"].astype(str) == "4").sum() == 0          # what the text compare did
    assert region.mask(frame).tolist() == [True, True, False, True]
    assert zipped.mask(frame).tolist() == [True, False, True, False]
    assert region.name == "region=4" and region.value == 4


def test_what_can_be_checked_without_a_test_row_is_refused_before_the_seal(tmp_path):
    declared, problems = US.declare(["region", "nosuch=1", "region=4.5", "dist_km=2.5"],
                                    DTYPES, FEATURES)
    assert [d.name for d in declared] == ["dist_km=2.5"]
    said = " | ".join(problems)
    assert "gives no value" in said and "'nosuch' is not one of the features" in said
    assert "not an integer" in said and "Int64" in said
    errs, ctx = evaluate.pre_seal_checks(_args(slices=["region=abc"]), {}, LEDGER, "H1-3",
                                         tmp_path / "test_seal.json", BASELINES, [],
                                         dtypes=DTYPES, features=FEATURES)
    assert len(errs) == 1 and "region=abc" in errs[0] and "before the look" in errs[0]
    assert not (tmp_path / "test_seal.json").exists()
    errs, ctx = evaluate.pre_seal_checks(_args(slices=["region=4"]), {}, LEDGER, "H1-3",
                                         tmp_path / "test_seal.json", BASELINES, [],
                                         dtypes=DTYPES, features=FEATURES)
    assert errs == [] and ctx.user_slices[0].value == 4 and ctx.exposure == 1


def test_the_dtype_comes_from_the_manifest_then_the_train_part(tmp_path):
    train = tmp_path / "train.parquet"
    pd.DataFrame({"region": pd.array([1, 2], dtype="Int64"), "extra": [0.5, 1.5]}) \
        .to_parquet(train)
    got = US.dtypes_of({"output_schema": {"region": {"dtype": "Int64"}}}, train)
    assert got["region"] == "Int64" and US.dtype_family(got["extra"]) == "float"


@dataclass(frozen=True)
class _Options:
    threshold: object = 0.5


class _MeanScore:
    """A metrics module whose one metric is the mean score of the rows it is given."""

    @staticmethod
    def metric(name, y_true, y_score, options=None, task=None):
        return float(np.mean(y_score))


def test_a_declared_slice_is_measured_in_the_one_look():
    n = 40
    frame = pd.DataFrame({"region": [4.0 if i % 2 else 1.0 for i in range(n)]})
    y, score = (np.arange(n) // 2) % 2, np.linspace(0, 1, n)     # both classes in the slice
    scoring = Scoring(_MeanScore, "binary", "pr_auc", "max", ("pr_auc",), _Options(), "binary")
    declared, _ = US.declare(["region=4"], DTYPES, FEATURES)
    markers = []
    out = evaluate.build_slices(frame, {}, None, y, score, scoring, Resampler(n, 20, 7),
                                declared, markers)
    row = next(s for s in out if s["name"] == "region=4")
    assert row["n"] == n // 2 and row["source"] == "user:turn"
    assert row["metrics"]["pr_auc"]["value"] == pytest.approx(np.mean(score[1::2]), abs=1e-4)
    # a raw item (a caller that did not declare it) is coerced on the frame's own dtype
    out = evaluate.build_slices(frame, {}, None, y, score, scoring, Resampler(n, 20, 7),
                                ["region=4"], markers)
    assert any(s["name"] == "region=4" and s["n"] == n // 2 for s in out)


# =========================================================================== S3: after-eval

def _previous(tmp_path, exposures=1, baselines=BASELINES):
    prev = tmp_path / "vega-runs" / "old"
    _write(prev / "campaign.json", {"schema": "vega/campaign@1"})
    _write(prev / "test_seal.json", {"run": "R0", "exposures": exposures, "token": "t"})
    _write(prev / "eval" / "R0" / "baselines.json",
           {"baselines": [{"name": b} for b in baselines]})
    camp = tmp_path / "vega-runs" / "new"
    camp.mkdir(parents=True)
    return prev, camp


def _after(prev, n):
    return {"after_eval": {"previous_campaign": str(prev), "exposures": n}}


def test_an_after_eval_campaign_reconciles_its_earlier_looks(tmp_path):
    prev, camp = _previous(tmp_path, exposures=2)
    looks, err = earlier_looks.read(_after(prev, 2), camp)
    assert err is None and looks.exposures == 2 and looks.run == "R0"
    assert looks.baselines == BASELINES
    assert earlier_looks.read({}, camp) == (None, None)
    for record, needle in ((_after(prev, 3), "records 2"),
                           (_after(tmp_path / "gone", 2), "not found"),
                           ({"after_eval": {"previous_campaign": str(prev), "exposures": 0}},
                            "at least once"),
                           ({"after_eval": {"exposures": 1}}, "names no previous_campaign"),
                           ({"after_eval": "yes"}, "not the record")):
        looks, err = earlier_looks.read(record, camp)
        assert looks is None and needle in err, (record, err)


def test_the_first_look_of_an_after_eval_campaign_is_a_further_exposure(tmp_path):
    prev, camp = _previous(tmp_path, exposures=2)
    seal = camp / "test_seal.json"

    def refused(args, campaign=None, baselines=BASELINES):
        errs, _ = evaluate.pre_seal_checks(args, campaign or _after(prev, 2), LEDGER, "H1-3",
                                           seal, baselines, [], dtypes=DTYPES,
                                           features=FEATURES)
        return " | ".join(errs)

    assert "knowing the answer" in refused(_args(threshold="0.42"))
    assert "after-eval campaign" in refused(_args(threshold="0.42"))
    assert "fitted to the test part" in refused(_args(slices=["region=4"]))
    said = refused(_args(), baselines=("trivial", "linear"))
    assert "narrower" in said and "lens" in said and "tree" in said
    prev2, camp2 = _previous(tmp_path / "w", exposures=2, baselines=("trivial", "linear", "tree"))
    errs, _ = evaluate.pre_seal_checks(_args(), _after(prev2, 2), LEDGER, "H1-3",
                                       camp2 / "test_seal.json", BASELINES, [], dtypes=DTYPES,
                                       features=FEATURES)
    said = " | ".join(errs)
    assert "wider" in said and "lens" in said and "campaign-baseline" in said
    assert "records 2" in refused(_args(), campaign=_after(prev, 1))
    errs, ctx = evaluate.pre_seal_checks(_args(threshold="0.5"), _after(prev, 2), LEDGER,
                                         "H1-3", seal, BASELINES, [], dtypes=DTYPES,
                                         features=FEATURES)
    assert errs == [] and ctx.exposure == 3 and ctx.looked
    assert not seal.exists()


def test_a_further_look_keeps_the_threshold_mode_the_last_one_used(tmp_path):
    """T1: valid-optimum and 0.5 are both allowed after a look, but switching between them is a
    choice made knowing the answer — a further look keeps the last look's mode."""
    from threshold_choice import CATALOG_HALF, requested_mode
    assert requested_mode({"applies": True, "source": "vega:default", "why": CATALOG_HALF}) == "0.5"
    assert requested_mode({"applies": True, "source": "vega:default",
                           "why": "no labelled valid part was available to optimise on"}) \
        == "valid-optimum"
    assert requested_mode({"applies": True, "source": "valid-optimum"}) == "valid-optimum"
    assert requested_mode({"applies": True, "source": "user:turn"}) == "valid-optimum"
    assert requested_mode({"applies": False}) is None and requested_mode(None) is None
    prev, camp = _previous(tmp_path, exposures=1)
    _write(prev / "eval" / "R0" / "threshold.json",
           {"applies": True, "source": "vega:default", "why": CATALOG_HALF, "value": 0.5})
    seal = camp / "test_seal.json"

    def refusals(args):
        errs, _ = evaluate.pre_seal_checks(args, _after(prev, 1), LEDGER, "H1-3", seal,
                                           BASELINES, [], dtypes=DTYPES, features=FEATURES)
        return " | ".join(errs)

    said = refusals(_args())
    assert "keeps the mode" in said and "--threshold 0.5" in said
    assert refusals(_args(threshold="0.5")) == ""
    _write(prev / "eval" / "R0" / "threshold.json", {"applies": True, "source": "valid-optimum"})
    assert "keeps the mode" in refusals(_args(threshold="0.5"))
    assert refusals(_args(threshold="valid-optimum")) == "" and refusals(_args()) == ""
    _write(prev / "eval" / "R0" / "threshold.json",
           {"applies": True, "source": "user:turn", "value": 0.3})
    assert "keeps the mode" in refusals(_args(threshold="0.5"))     # never re-chosen after a look
    assert refusals(_args()) == ""
    assert not seal.exists()

def test_the_seal_counts_the_earlier_looks_and_keeps_them(tmp_path):
    record = {"previous_campaign": "vega-runs/old", "exposures": 2}
    seal = Seal.create(tmp_path, "H1-3", tmp_path / "t.parquet", 10, "first look", "user",
                       after_eval=record)
    assert seal.exposures == 3 and seal.as_json()["after_eval"] == record
    seal.write(tmp_path / "test_seal.json")
    back, err = Seal.load(tmp_path / "test_seal.json")
    assert err is None and back.after_eval == record and back.exposures == 3
    again = back.again("H1-3", "a reviewer asked", "user")
    assert again.exposures == 4 and again.after_eval == record
    assert "after_eval" not in Seal.create(tmp_path, "R0", "t", 1, "r", "user").as_json()


# =========================================================================== B5: campaign-baseline

def test_campaign_baseline_says_why_before_any_child_runs(tmp_path):
    no_r0 = campaign_baseline.predict(tmp_path, tmp_path / "s", {"runs": [{"id": "H1"}]},
                                      [0, 1], tmp_path)
    assert no_r0[0] is None and "no R0 run" in no_r0[1]
    missing = campaign_baseline.predict(tmp_path, tmp_path / "s", LEDGER, [0, 1], tmp_path)
    assert missing[0] is None and "no checkpoint" in missing[1]
    ckpt = tmp_path / "runs" / "R0" / "checkpoints" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    ledger = {"runs": [{"id": "R0", "checkpoint_digest": "sha256:" + "0" * 64}]}
    changed = campaign_baseline.predict(tmp_path, tmp_path / "s", ledger, [0, 1], tmp_path)
    assert changed[0] is None and "weights changed" in changed[1]


def test_campaign_baseline_is_compared_and_never_the_strongest_fitted():
    rows = ({"name": "trivial", "fitted": True}, {"name": "tree", "fitted": True},
            {"name": "lens", "fitted": False},
            {"name": "campaign-baseline", "fitted": False, "predicted": True})
    measured = evaluate.Measured(per_model={}, scorers={}, baselines=rows)
    assert measured.fitted == ["trivial", "tree"]
    assert measured.compared == ["trivial", "tree", "campaign-baseline"]


# =========================================================================== M1: mape's gate

def test_mape_is_reported_only_when_the_train_target_is_away_from_zero():
    far = metric_gates.decided("regression", ["rmse", "mape"], np.linspace(1, 9, 200))
    assert list(far) == ["mape"] and far["mape"].applies and far["mape"].share == 1.0
    near = np.r_[np.full(10, 0.05), np.linspace(1, 9, 190)]          # 95 % above 0.1
    gate = metric_gates.decided("regression", ["mape"], near)["mape"]
    assert not gate.applies and gate.share == 0.95 and "99%" in gate.why
    assert gate.as_json()["part"] == "train" and "withheld" in gate.note
    assert metric_gates.decided("binary", ["mape"], near) == {}
    assert not metric_gates.decided("regression", ["mape"], None)["mape"].applies
    assert metric_gates.GATES["mape"]["abs_y_above"] == 0.1


# =========================================================================== B6: lens

def test_the_lens_protocol_is_read_from_either_record():
    own = lens_protocol({"family": "HistGradientBoosting",
                         "hyper_parameters": {"max_depth": 3}})
    assert own == {"family": "HistGradientBoosting", "hyper_parameters": {"max_depth": 3},
                   "source": "analysis:modules/importance"}
    evidence = {"ran": True, "evidence": {
        "models": [{"name": "logistic-l2", "cv_score": 0.84, "params": {"C": 0.5}},
                   {"name": "gradient-boosting", "cv_score": 0.87}],
        "permutation_importance": [{"feature": "qty", "model": "logistic-l2"}]}}
    got = lens_protocol(evidence)
    assert got["family"] == "logistic-l2" and got["hyper_parameters"] == {"C": 0.5}
    evidence["evidence"]["permutation_importance"] = []
    assert lens_protocol(evidence)["family"] == "gradient-boosting"
    assert lens_protocol({"ran": False, "skipped_because": "no partition"}) is None
    assert lens_protocol(None) is None


def _rendered():
    sys.path.insert(0, str(FORGE_TESTS))
    import test_baselines as TB  # noqa: E402 — renders the live templates once
    return TB


def test_lens_fits_with_the_recorded_hyper_parameters():
    TB = _rendered()
    sp, spec, _, _ = TB.row_fixture("binary")
    ctx = TB.D.BaselineContext(protocol={"family": "gradient-boosting", "hyper_parameters": {
        "max_depth": 2, "max_iter": 7, "n_estimators": 5, "random_state": 3}})
    lens = TB.fitted(spec, sp, "lens", ctx)
    assert lens.fitted and lens.model.max_depth == 2 and lens.model.max_iter == 7
    record = lens.record()
    assert record["delegate"] == "tree" and record["family"] == "gradient-boosting"
    assert record["hyper_parameters"]["applied"] == {"max_depth": 2, "max_iter": 7}
    assert set(record["hyper_parameters"]["not_carried"]) == {"n_estimators", "random_state"}
    assert lens.model.random_state == 7                       # the spec's seed, not the analysis's
    linear = TB.fitted(spec, sp, "lens", TB.D.BaselineContext(
        protocol={"family": "logistic-l2", "hyper_parameters": {"C": 0.25}}))
    assert linear.record()["delegate"] == "linear" and linear.model.C == 0.25
    bad = TB.made(spec, "lens", TB.D.BaselineContext(
        protocol={"family": "gradient-boosting", "hyper_parameters": {"max_depth": -3}}))
    with pytest.raises(ValueError, match="max_depth"):
        bad.fit(bad.design_of(sp, sp.load_part(spec, "train")))


# =========================================================================== A1–A3: the apply script

def _layer(threshold=None, metrics_extra=None, baselines=None):
    est = {"value": 0.5, "ci": [0.4, 0.6], "n": 100}
    metrics = {"task": "binary", "task_effective": "binary", "primary": "pr_auc", "mode": "max",
               "models": {"network": {"pr_auc": dict(est)}, "trivial": {"pr_auc": dict(est)}},
               "differences": {"trivial": {"value": 0.0, "ci": [-0.1, 0.1]}},
               "verdict": {"earned": False, "strongest_baseline": "trivial", "diff": 0.0,
                           "ci": [-0.1, 0.1], "beaten_by_baseline": False},
               "upstream": {"valid": {"pr_auc": {"value": 0.7, "_upstream": True}}}}
    metrics.update(metrics_extra or {})
    return {"schema": apply.SCHEMA, "seal": {"exposures": 1, "test": {}}, "run": {"id": "R0"},
            "verdict": metrics["verdict"], "metrics": metrics,
            "threshold": threshold or {"applies": True, "value": 0.5, "source": "valid-optimum",
                                       "metrics_at_threshold": {"f1": dict(est)}},
            "baselines": {"baselines": baselines or [
                {"name": "trivial", "fitted": True, "valid": dict(est), "test": dict(est)},
                {"name": "linear", "fitted": True, "valid": dict(est), "test": dict(est)},
                {"name": "campaign-baseline", "fitted": False, "predicted": True,
                 "_upstream": True, "valid": {"value": 0.7, "_upstream": True},
                 "test": dict(est)}]},
            "slices": {"slices": []}, "robustness": {"checks": []},
            "notebook": {"executed": True}, "readings": {"verdict": "short"}}


def test_a_well_formed_layer_raises_none_of_these():
    errs = " | ".join(apply.validate(_layer(), print))
    assert "metrics_at_threshold" not in errs and "_upstream" not in errs
    assert "withheld" not in errs


def test_a_threshold_metric_without_its_interval_is_refused():
    layer = _layer(threshold={"applies": True, "value": 0.5, "source": "valid-optimum",
                              "metrics_at_threshold": {"f1": {"value": 0.61, "n": 100}}})
    errs = apply.validate(layer, print)
    assert any("threshold/metrics_at_threshold/f1" in e and "confidence interval" in e
               for e in errs), errs


def test_a_baseline_threshold_is_its_own_and_never_chosen_on_test():
    layer = _layer()
    own = {"at": 0.41, "per_label": None, "scale": "probability", "source": "valid-optimum",
           "why": "its own valid scores"}
    layer["baselines"]["baselines"][1]["decision_threshold"] = dict(own)
    errs = " | ".join(apply.validate(layer, print))
    assert "threshold source" not in errs and "decision_threshold" not in errs, errs
    layer["baselines"]["baselines"][1]["decision_threshold"]["source"] = "test-optimum"
    assert any("baseline 'linear'" in e and "names the test part" in e
               for e in apply.validate(layer, print))
    layer["baselines"]["baselines"][1]["decision_threshold"]["source"] = "user:turn"
    layer["seal"]["exposures"] = 2
    assert any("a baseline's threshold source is a user value on exposure 2" in e
               for e in apply.validate(layer, print))


def test_a_test_number_marked_upstream_is_refused():
    layer = _layer()
    layer["baselines"]["baselines"][2]["test"] = {"value": 0.7, "_upstream": True}
    errs = apply.validate(layer, print)
    assert any("baselines/baselines[2]/test is marked _upstream" in e for e in errs), errs


def test_a_metric_its_gate_withheld_is_not_published():
    gates = {"gates": {"mape": {"applies": False, "why": "|y| > 0.1 on only 80% of rows"}}}
    layer = _layer(metrics_extra=gates)
    layer["metrics"]["models"]["network"]["mape"] = {"value": 0.3, "ci": [0.2, 0.4]}
    errs = apply.validate(layer, print)
    assert any("models/network/mape" in e and "withheld" in e for e in errs), errs
    layer["metrics"]["models"]["network"]["mape"] = {"value": None, "note": "withheld"}
    assert not apply.gate_problems(layer)


def test_the_analysis_layer_is_found_where_the_spec_names_it(tmp_path, monkeypatch):
    """The T5 round 3 rehearsal: `input.model` was `data/<stem>.html` (relative to the project,
    as every `input` path), and the locator looked for its file name beside the campaign and in
    the working directory only, so the lens baseline, every slice and the robustness checks
    that read the analysis layer found none. The path as the spec states it comes first."""
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    camp = project / "vega-runs" / "2026-10-11T00-42Z"
    camp.mkdir(parents=True)
    layer = {"schema": "data-lens/analysis@1", "findings": [{"id": "F1"}]}
    (project / "data" / "lc.domain.analysis.shaped.html").write_text(
        '<html><script id="layer-analysis-data" type="application/json">'
        + json.dumps(layer) + "</script></html>", encoding="utf-8")
    monkeypatch.chdir(project)
    spec = {"input": {"model": "data/lc.domain.analysis.shaped.html"}}
    assert evaluate._analysis(camp, spec) == layer
    # a copy beside the campaign (a campaign moved with its model) still comes first
    moved = {"schema": "data-lens/analysis@1", "findings": [{"id": "F9"}]}
    (camp.parent / "lc.domain.analysis.shaped.html").write_text(
        '<html><script id="layer-analysis-data" type="application/json">'
        + json.dumps(moved) + "</script></html>", encoding="utf-8")
    assert evaluate._analysis(camp, spec) == moved
    assert evaluate._analysis(camp, {"input": {"model": "data/missing.html"}}) is None
    assert evaluate._analysis(camp, {"input": {}}) is None


def test_the_classes_are_read_through_the_tasks_label_set():
    """Final rehearsal round 2: the package's `Task` holds the spec's `LabelSet`
    (`task.labels`), no bare `classes` tuple; a package emitted before still carries one."""
    import types
    new = types.SimpleNamespace(labels=types.SimpleNamespace(classes=("water", "crop")))
    old = types.SimpleNamespace(classes=("a", "b"))
    assert task_classes(new) == ("water", "crop") and task_classes(old) == ("a", "b")

