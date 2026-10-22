"""
The machine verdict and its consistency check (eval-contract §2, §5, §6; BUILD-PLAN A30).

A verdict is read off the interval of the paired difference against the strongest fitted
baseline — or, on a primary that lives on each model's own score scale (an unlabelled anomaly
task's `anomaly_score_mean`), it is **not assessable**: `earned` null with its reason, no
baseline, no difference, no interval. `evaluate.py` writes it; `apply_evaluation_layer.py`
refuses a verdict that disagrees with its own record and a reading that says otherwise.

    python3 -m pytest -q vega-eval/scripts/tests/test_verdict_assessable.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import apply_evaluation_layer as apply  # noqa: E402
import evaluate  # noqa: E402

REASON = evaluate.NOT_ASSESSABLE


def _metrics(kind="anomaly", primary="anomaly_score_mean", mode="min", verdict=None, differences=None):
    return {"task": kind, "task_effective": kind, "primary": primary, "mode": mode,
            "differences": differences or {},
            "verdict": verdict if verdict is not None else evaluate.unassessed_verdict(REASON)}


def _earned(diff, ci, earned, beaten=None, strongest="tree"):
    out = {"earned": earned, "strongest_baseline": strongest, "diff": diff, "ci": list(ci)}
    if beaten is not None:
        out["beaten_by_baseline"] = beaten
    return out


# --------------------------------------------------------------------------- evaluate.py
@pytest.mark.parametrize("kind, primary, assessable", [
    ("anomaly", "anomaly_score_mean", False),       # unlabelled: each model's own scale
    ("anomaly", "pr_auc", True),                    # labelled: a ranking of the label, scale-free
    ("binary", "pr_auc", True),
    ("unsupervised", "reconstruction_mse", True),   # every model reconstructs the same inputs
    ("regression", "rmse", True)])
def test_only_a_scale_bound_primary_is_not_assessable(kind, primary, assessable):
    assert (evaluate.not_assessable(kind, primary) is None) == assessable


def test_the_not_assessable_verdict_is_neither_a_win_nor_a_loss_and_says_why():
    v = evaluate.unassessed_verdict(REASON)
    assert v == {"earned": None, "strongest_baseline": None, "diff": None, "ci": [],
                 "beaten_by_baseline": None, "reason": REASON}
    assert "anomaly_score_mean" in REASON and "no paired difference" in REASON
    # the reason is prose the layer's readings quote: no digit a reading would have to match
    assert not any(ch.isdigit() for ch in REASON)
    assert "earn" not in REASON.lower() and "worth" not in REASON.lower()


def test_the_writer_and_the_check_read_one_table_of_scale_bound_primaries():
    """evaluate.py writes the verdict by SCALE_BOUND_PRIMARIES and the apply script checks it by
    its own copy (neither imports the other): the two copies are pinned equal here."""
    assert apply.SCALE_BOUND_PRIMARIES == evaluate.SCALE_BOUND_PRIMARIES


# --------------------------------------------------------------------------- apply: the record
def test_the_unlabelled_anomaly_verdict_passes_the_check():
    assert apply.verdict_problems(evaluate.unassessed_verdict(REASON), _metrics()) == []


def test_a_paired_verdict_on_a_scale_bound_primary_is_refused():
    verdict = _earned(2.6289, (2.3124, 2.8369), False, beaten=True)
    problems = apply.verdict_problems(verdict, _metrics(verdict=verdict))
    assert problems and "not assessable" in problems[0] and "own score" in problems[0]


@pytest.mark.parametrize("kind, primary", [("binary", "pr_auc"), ("anomaly", "pr_auc"),
                                           ("regression", "rmse")])
def test_a_not_assessable_verdict_elsewhere_is_refused(kind, primary):
    problems = apply.verdict_problems(evaluate.unassessed_verdict(REASON),
                                      _metrics(kind, primary, "max"))
    assert len(problems) == 1 and "only a primary that lives on each model's own" in problems[0]


def test_a_not_assessable_verdict_carries_its_reason_and_nothing_measured():
    bare = dict(evaluate.unassessed_verdict(REASON), reason="")
    assert any("carries no reason" in p for p in apply.verdict_problems(bare, _metrics()))
    named = dict(evaluate.unassessed_verdict(REASON), strongest_baseline="tree", diff=0.4)
    assert any("no paired difference" in p for p in apply.verdict_problems(named, _metrics()))
    diffs = _metrics(differences={"tree": {"value": 0.4, "ci": [0.1, 0.6]}})
    assert any("no paired difference" in p
               for p in apply.verdict_problems(evaluate.unassessed_verdict(REASON), diffs))


@pytest.mark.parametrize("mode, ci, earned, beaten, ok", [
    ("max", (0.01, 0.05), True, False, True),
    ("max", (0.01, 0.05), False, False, False),     # the interval excludes 0: earned
    ("max", (-0.01, 0.05), True, False, False),     # covers 0: not earned
    ("max", (-0.01, 0.05), False, False, True),
    ("max", (-0.09, -0.02), False, True, True),      # excludes 0 the other way: beaten
    ("max", (-0.09, -0.02), False, False, False),
    ("min", (-0.05, -0.01), True, False, True),      # a loss-like primary: lower is better
    ("min", (-0.05, -0.01), False, True, False),
    ("min", (0.02, 0.06), False, True, True)])
def test_earned_and_beaten_follow_from_the_interval(mode, ci, earned, beaten, ok):
    verdict = _earned(round(sum(ci) / 2, 4), ci, earned, beaten)
    problems = apply.verdict_problems(verdict, _metrics("binary", "pr_auc", mode, verdict=verdict))
    assert (problems == []) == ok, problems


def test_an_earned_verdict_names_a_baseline_and_carries_an_interval():
    none = {"earned": False, "strongest_baseline": None, "diff": None, "ci": []}
    assert apply.verdict_problems(none, _metrics("binary", "pr_auc", "max", verdict=none)) == []
    bold = dict(none, earned=True)
    assert "no fitted baseline" in apply.verdict_problems(bold, _metrics("binary", "pr_auc", "max"))[0]
    bare = _earned(0.1, (), False)
    assert "carries no interval" in apply.verdict_problems(bare, _metrics("binary", "pr_auc", "max"))[0]
    odd = dict(none, earned="yes")
    assert "true, false, or null" in apply.verdict_problems(odd, _metrics("binary", "pr_auc", "max"))[0]


# --------------------------------------------------------------------------- apply: the reading
def _doc(verdict_text):
    est = {"value": 0.8123, "ci": [0.7011, 0.9134], "n": 300}
    metrics = _metrics()
    metrics["models"] = {"network": {"anomaly_score_mean": est},
                         "trivial": {"anomaly_score_mean": {"value": 1.2, "ci": [1.1, 1.3], "n": 300}}}
    readings = {"verdict": verdict_text,
                "calibration": "No calibration metric was recorded for an unlabelled anomaly score here.",
                "slices": "The evaluation cut no slice of the test part the primary can be read on.",
                "robustness": "Every robustness check was skipped, each with the reason its record gives.",
                "limits": "One test part and one seed bound what this evaluation can claim about the network."}
    return {"schema": apply.SCHEMA, "seal": {"exposures": 1}, "run": {"id": "R0"},
            "verdict": copy.deepcopy(metrics["verdict"]), "metrics": metrics,
            "baselines": {"baselines": [{"name": "trivial", "fitted": True,
                                         "test": metrics["models"]["trivial"]["anomaly_score_mean"]},
                                        {"name": "linear", "fitted": False, "why": "not fitted here",
                                         "test": None}]},
            "threshold": {"applies": False, "value": None, "scale": "score", "source": "vega:default",
                          "why": "no label and no expected anomaly rate: the reading answers the score"},
            "slices": {"slices": [], "worst": None}, "robustness": {"checks": []},
            "notebook": {"executed": True}, "readings": readings, "markers": []}


def test_a_not_assessable_reading_says_so_and_passes():
    text = ("The network scores anomaly_score_mean 0.8123 [0.7011, 0.9134] on the 300 held-out rows. "
            "The verdict is not assessable: whether the network earned its cost cannot be read off "
            "scores that live on different scales. No paired difference is read here.")
    assert apply.validate(_doc(text), lambda *_: None) == []


def test_a_not_assessable_verdict_read_as_a_loss_or_a_win_is_refused():
    base = "The network scores anomaly_score_mean 0.8123 [0.7011, 0.9134] on the 300 held-out rows. "
    silent = apply.validate(_doc(base + "Its scores are its own, on the test part."), lambda *_: None)
    assert any("does not say so" in e for e in silent), silent
    loss = apply.validate(_doc(base + "The verdict is not assessable. The network has NOT earned its cost."),
                          lambda *_: None)
    assert any("reads as not earned" in e for e in loss), loss
    win = apply.validate(_doc(base + "The verdict is not assessable. The network earned its cost."),
                         lambda *_: None)
    assert any("reads as earned" in e for e in win), win


def _paired_doc(verdict_text):
    """A regression layer whose verdict is earned against tree, off its interval."""
    doc = _doc("placeholder")
    net = {"value": 0.8123, "ci": [0.7011, 0.9134], "n": 300}
    tree = {"value": 0.7612, "ci": [0.6508, 0.8702], "n": 300}
    diff = {"value": 0.0511, "ci": [0.0101, 0.0934], "excludes_zero": True}
    verdict = _earned(0.0511, (0.0101, 0.0934), True, beaten=False)
    doc["metrics"] = dict(_metrics("regression", "r2", "max", verdict=verdict,
                                   differences={"tree": diff}),
                          models={"network": {"r2": net}, "tree": {"r2": tree}})
    doc["verdict"] = copy.deepcopy(verdict)
    doc["baselines"] = {"baselines": [
        {"name": "trivial", "fitted": True, "test": {"value": 0.0, "ci": [0.0, 0.0], "n": 300}},
        {"name": "tree", "fitted": True, "test": tree}]}
    doc["threshold"] = {"applies": False, "value": None, "source": "vega:default",
                        "why": "a regression reading decides by no threshold"}
    doc["readings"]["verdict"] = verdict_text
    return doc


PAIRED = ("The network scores r2 0.8123 [0.7011, 0.9134] on the 300 held-out rows, "
          "0.0511 [0.0101, 0.0934] above tree. ")


def test_an_assessable_reading_as_the_record_says_passes():
    assert apply.validate(_paired_doc(PAIRED + "The network earned its cost."), lambda *_: None) == []


def test_an_assessable_verdict_called_not_assessable_is_refused():
    """'not assessable' is §5's wording for a scale-bound primary only: on a verdict read off its
    interval it is refused, and a sentence carrying it is still read for its earned claim."""
    hedged = apply.validate(_paired_doc(PAIRED + "The verdict is not assessable, but the network "
                                        "earned its cost."), lambda *_: None)
    assert any("calls the verdict not assessable" in e for e in hedged), hedged
    assert any("reads as not earned but verdict.earned is True" in e for e in hedged), hedged
    bare = apply.validate(_paired_doc(PAIRED + "The verdict is not assessable."), lambda *_: None)
    assert len(bare) == 1 and "calls the verdict not assessable" in bare[0], bare
