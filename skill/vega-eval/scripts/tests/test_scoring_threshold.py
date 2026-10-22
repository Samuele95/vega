#!/usr/bin/env python3
"""
test_scoring_threshold.py — every metric this stage reports is read at the threshold the run
chose, on the scale it was chosen on, one per label where the task has labels.

    python3 test_scoring_threshold.py [-v]

This test exists because the defect it guards was invisible on the acceptance fixture. That
campaign is a `binary` task whose slice metrics — pr_auc, roc_auc, log_loss, brier, ece — take
no threshold at all, so when `build_slices` and `build_robustness` were reading at the metrics
module's own `threshold=0.5` default while the headline was read at the valid-optimum 0.70885,
every number in the acceptance run was still correct and all 42 smoke checks and 28 negatives
passed. For a `multilabel` or `anomaly` campaign the layer would have published two different
readings of the same run side by side.

So the property is asserted directly rather than through a campaign: a fake metrics module that
returns the threshold it was given (`options.threshold`, the package's `MetricOptions`), driven
through the real `Scoring`, the real `build_slices` and the real `build_robustness`. Whatever
those functions report is what they were called with — one value, or one per label.

The v2.3 half (BUILD-PLAN R-C2, R-C3) runs the real threshold chooser and the real stored
decision over the backend's own `metrics.py`, rendered from the template:

  * **scale** — a binary or multilabel threshold is a probability and σ is applied to the logit
    unconditionally, even when every logit lies inside [0, 1] (E18); an anomaly threshold lives
    on the score's own scale (the valid quantile at the expected rate, or its f1 optimum), and
    with neither a rate nor a label nothing is chosen (`applies` false, no stored decision);
  * **per label** — the multilabel chooser gives each label the threshold that maximises that
    label's own f1 on valid, keyed by class name, and a label valid cannot place takes the one
    uniform threshold (`value`); the metrics read them in class order;
  * the stored `prediction` is σ(logit) ≥ threshold, and `threshold.json` records `applies`,
    `value`, `scale` and `per_label`; a `--threshold` the reading cannot use is refused before
    the seal; `metrics.json.ci_level` is the level the intervals are drawn at (0.95).

Exit 0 when every check passes.
"""
from __future__ import annotations

import importlib
import math
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402

import evaluate as E                                                      # noqa: E402
import threshold_choice as T                                              # noqa: E402
from evaluate import build_robustness, build_slices, estimates_for        # noqa: E402
from vega_eval_core import CI_LEVEL, Resampler, Scoring                   # noqa: E402

VERBOSE = "-v" in sys.argv
CHOSEN = 0.6875                      # a threshold no default produces by accident, and
                                     # one that survives the records' 4-place rounding
PER_LABEL = (0.25, 0.6875, 0.8125)   # three labels, three different thresholds
FORGE_TESTS = HERE.parents[2] / "vega-forge" / "scripts" / "tests"
TEMPLATES = FORGE_TESTS.parent / "backends" / "pytorch" / "templates"
failures = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    if VERBOSE:
        print(f"  ok    {msg}")


def check_that(label, condition, detail=""):
    if condition:
        ok(label)
    else:
        fail(f"{label}{': ' + detail if detail else ''}")


@dataclass(frozen=True)
class Options:
    """The shape of the package domain's MetricOptions that Scoring moves: a frozen value with
    a threshold (one value, or one per label)."""

    threshold: object = 0.5


def _code(threshold) -> float:
    """One number that says which threshold a call was given: the value itself, or for a
    per-label tuple a weighted sum no single value can produce by accident."""
    if isinstance(threshold, tuple):
        return float(sum((k + 1) * 10.0 * v for k, v in enumerate(threshold)))
    return float(threshold)


class ThresholdEcho:
    """A metrics module whose every metric answers with the threshold it was handed.

    A metric that is genuinely threshold-free in the real module is still asked here, because
    the question under test is not "does this metric use a threshold" but "was this call given
    the run's threshold or did something drop it on the way".
    """

    @staticmethod
    def metric(name, y_true, y_score, options=None, task=None):
        if name == "vector_metric":
            return [1.0, 2.0, 3.0]
        return _code(options.threshold) if options is not None else -1.0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def scoring_with_threshold(threshold=CHOSEN):
    return Scoring(ThresholdEcho, "multilabel", "f1_micro", "max",
                   ("f1_micro", "f1_macro", "hamming"), Options(),
                   "multilabel").at_threshold(threshold)


def fixture(n=120):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "region": rng.integers(0, 3, n).astype(float),
        "dist_km": rng.normal(10, 3, n),
    })
    y = rng.integers(0, 2, n)
    score = rng.random(n)
    return frame, y, score


def reported_numbers(node, out=None):
    """Every `value` a record tree holds, which for ThresholdEcho is every threshold used."""
    out = [] if out is None else out
    if isinstance(node, dict):
        if "value" in node and isinstance(node.get("value"), (int, float)):
            out.append(float(node["value"]))
        for v in node.values():
            reported_numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            reported_numbers(v, out)
    return out


def check(label, numbers, want=CHOSEN):
    numbers = [v for v in numbers if v is not None]
    if not numbers:
        fail(f"{label}: reported no numbers at all, so the property is untested here")
        return
    # the records round to 4 places: the echo's number is compared at that rounding
    wrong = sorted({v for v in numbers if abs(v - round(_code(want), 4)) > 1e-9})
    if wrong:
        fail(f"{label}: {len(wrong)} distinct threshold(s) other than the run's "
             f"{want} reached the metrics module: {wrong[:4]} "
             f"(-1.0 means no threshold was passed at all)")
    else:
        ok(f"{label}: all {len(numbers)} number(s) read at {want}")


def read_at_the_run_threshold(threshold):
    frame, y, score = fixture()
    scoring = scoring_with_threshold(threshold)
    resampler = Resampler(len(y), 20, 7)
    analysis = {"modules": {"segments": {}, "drift": {"columns": ["dist_km"]},
                            "importance": {}}, "context": {}}
    spec = {"metrics": {"primary": "f1_micro"},
            "data_contract": {"features": [{"name": "region", "role": "nominal"},
                                           {"name": "dist_km", "role": "numeric"}]}}
    tag = "per label" if isinstance(threshold, tuple) else "one value"

    check(f"the headline metrics ({tag})", reported_numbers(
        {k: v.as_json() for k, v in estimates_for(resampler, scoring, y, score).items()}),
        threshold)

    markers = []
    slices = build_slices(frame, spec, analysis, y, score, scoring, resampler, [], markers)
    if not slices:
        fail("build_slices produced no slice, so nothing about its threshold was tested")
    check(f"the slices ({tag})", reported_numbers(slices), threshold)

    markers = []
    checks = build_robustness(frame, spec, analysis, y, lambda f: score, scoring, resampler,
                              frame, markers)
    measured = [c for c in checks if c.get("status") != "skipped"]
    if not measured:
        fail("build_robustness measured nothing, so nothing about its threshold was tested")
    check(f"the robustness checks ({tag})", reported_numbers(measured), threshold)

    # a window or graph sample: no re-scoring, the perturbation said skipped, and what is still
    # measured (the drift sides) is read at the same threshold
    markers = []
    checks = build_robustness(frame, spec, analysis, y, None, scoring, resampler, None, markers,
                              base_score=score, perturbation_skipped="windowed inputs are "
                              "skipped and said so")
    skipped = [c for c in checks if c.get("check") == "importance-perturbation"]
    check_that(f"a window sample's perturbation is skipped with its reason ({tag})",
               bool(skipped) and skipped[0].get("status") == "skipped"
               and "windowed" in skipped[0].get("why", ""), str(skipped))
    check(f"a window sample's drift checks ({tag})",
          reported_numbers([c for c in checks if c.get("status") != "skipped"]), threshold)


def the_control():
    """A Scoring that was never given a threshold must show it, so a green result above cannot
    be an artefact of the echo module."""
    _, y, score = fixture()
    resampler = Resampler(len(y), 20, 7)
    bare = Scoring(ThresholdEcho, "multilabel", "f1_micro", "max", ("f1_micro",))
    bare_numbers = reported_numbers(
        {k: v.as_json() for k, v in estimates_for(resampler, bare, y, score).items()})
    if not bare_numbers or any(abs(v - CHOSEN) < 1e-9 for v in bare_numbers):
        fail("the control did not fail: a Scoring with no threshold reported the run's "
             "threshold anyway, so this test cannot detect the defect it exists for")
    else:
        ok(f"control: a Scoring with no threshold reports {bare_numbers[0]}, not {CHOSEN}")
    unlabelled = estimates_for(resampler, scoring_with_threshold(), None, score)
    check("an unlabelled part's metrics (no truth, still resampled)",
          reported_numbers({k: v.as_json() for k, v in unlabelled.items()}))


# ================================================================ the real metrics module

def real_metrics():
    """The backend's metrics.py and domain.py, rendered from the templates."""
    sys.path.insert(0, str(FORGE_TESTS))
    import render_templates as rt
    work = tempfile.mkdtemp(prefix="eval-threshold-")
    rec = rt.render(TEMPLATES, work, "reference", "reference", only=["metrics.py"])
    sys.path.insert(0, work)
    importlib.invalidate_caches()
    package = rec["package"]
    return (importlib.import_module(f"{package}.model.metrics"),
            importlib.import_module(f"{package}.domain"))


def real_scoring(M, D, effective, primary, names=()):
    return Scoring(M, effective, primary, "max", (primary, *names), D.MetricOptions(), effective)


def f1_at(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def brute_force(y, p):
    """The f1 optimum over the same candidate grid the chooser draws (99 quantiles of σ)."""
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    scores = [f1_at(y, p, t) for t in grid]
    return float(grid[int(np.argmax(scores))]), max(scores)


def binary_threshold_is_a_probability(M, D):
    """E18: logits that all lie inside [0, 1] are still logits — the grid, the chosen value and
    the stored decision all read σ of them."""
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 400)
    z = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.12, 400), 0.05, 0.95)
    scoring = real_scoring(M, D, "binary", "f1")
    choice = T.choose_threshold(scoring, y, z, None, effective="binary")
    want, _ = brute_force(y, M.sigmoid(z))
    check_that("binary: the threshold is on the probability scale",
               choice.applies and choice.scale == "probability", str(choice))
    check_that("binary: the chosen threshold is the f1 optimum of σ(logit), not of the logit",
               abs(choice.value - want) < 1e-12, f"{choice.value} vs {want}")
    check_that("binary: a logit inside [0, 1] is still read through σ (the value is above "
               "σ(min logit))", choice.value >= float(M.sigmoid(z.min())) - 1e-12,
               f"{choice.value} < σ({z.min()})")
    look = types.SimpleNamespace(
        threshold=choice, effective="binary", adapters=types.SimpleNamespace(metrics=M),
        parts=types.SimpleNamespace(score_net=z, scored=types.SimpleNamespace(row_score=None)))
    stored = E._decision(look)
    check_that("binary: the stored prediction is σ(logit) ≥ threshold",
               np.array_equal(stored, (M.sigmoid(z) >= choice.value).astype(int)))
    check_that("binary: ...which is not the logit read as a probability",
               not np.array_equal(stored, (z >= choice.value).astype(int)))
    record = choice.record("binary", "binary", {})
    check_that("binary: threshold.json carries applies, value, scale and per_label",
               {"applies", "value", "scale", "per_label"} <= set(record)
               and record["per_label"] is None and record["scale"] == "probability")
    at = scoring.at_threshold(choice.metric_threshold())
    check_that("binary: the metrics are read at the chosen threshold",
               abs(at.point("f1", y, z) - f1_at(y, M.sigmoid(z), choice.value)) < 1e-12)


def multilabel_per_label(M, D):
    """Each label its own f1 optimum, keyed by class name, read in class order; a label valid
    holds one value of takes the uniform threshold."""
    rng = np.random.default_rng(5)
    n, classes = 600, ("red", "green", "blue", "grey")
    y = np.zeros((n, 4), dtype=np.int64)
    z = np.zeros((n, 4))
    for j, shift in enumerate((-2.0, 0.0, 2.0)):
        y[:, j] = rng.integers(0, 2, n)
        z[:, j] = 1.5 * (y[:, j] - 0.5) + shift + rng.normal(0, 1.0, n)
    z[:, 3] = rng.normal(0, 1.0, n)                  # 'grey' never occurs on valid
    scoring = real_scoring(M, D, "multilabel", "f1_micro")
    choice = T.choose_threshold(scoring, y, z, None, effective="multilabel", classes=classes)
    check_that("multilabel: per_label is keyed by class name",
               choice.per_label is not None and list(choice.per_label) == list(classes),
               str(choice.per_label))
    for j, name in enumerate(classes[:3]):
        want, _ = brute_force(y[:, j], M.sigmoid(z[:, j]))
        check_that(f"multilabel: {name!r}'s threshold is that label's own f1 optimum",
                   abs(choice.per_label[name] - want) < 1e-12,
                   f"{choice.per_label[name]} vs {want}")
    distinct = len({round(choice.per_label[c], 9) for c in classes[:3]})
    check_that("multilabel: labels with different score offsets get different thresholds",
               distinct == 3, str(choice.per_label))
    check_that("multilabel: a label valid cannot place takes the uniform value",
               abs(choice.per_label["grey"] - choice.value) < 1e-12
               and "grey" in choice.why)
    check_that("multilabel: the scale is probability", choice.scale == "probability")
    tup = choice.metric_threshold(classes)
    check_that("multilabel: the metrics read the per-label thresholds in class order",
               tup == tuple(choice.per_label[c] for c in classes), str(tup))
    at = scoring.at_threshold(tup)
    decided = (M.sigmoid(z) >= np.asarray(tup)).astype(int)
    tp = int(((decided == 1) & (y == 1)).sum())
    fp = int(((decided == 1) & (y == 0)).sum())
    fn = int(((decided == 0) & (y == 1)).sum())
    check_that("multilabel: f1_micro is read at each label's own threshold",
               abs(at.point("f1_micro", y, z) - 2 * tp / (2 * tp + fp + fn)) < 1e-12)
    record = choice.record("multilabel", "multilabel", {})
    check_that("multilabel: threshold.json per_label is keyed by class name",
               record["per_label"] == {c: choice.per_label[c] for c in classes})
    user = T.choose_threshold(scoring, y, z, "0.5", effective="multilabel", classes=classes)
    check_that("multilabel: --threshold 0.5 is one value for every label (no per_label)",
               user.per_label is None and user.value == 0.5 and user.source == "vega:default")


def anomaly_on_the_score_scale(M, D):
    rng = np.random.default_rng(3)
    n = 500
    y = (rng.random(n) < 0.1).astype(np.int64)
    s = np.abs(rng.normal(3.0, 1.0, n)) + 4.0 * y        # scores well outside [0, 1]
    scoring = real_scoring(M, D, "anomaly", "pr_auc", ("f1",))
    rated = T.choose_threshold(scoring, y, s, None, effective="anomaly", rate=0.1,
                               rate_source="the train part's positive share")
    check_that("anomaly: with an expected rate, the valid quantile on the score's scale",
               rated.applies and rated.scale == "score"
               and abs(rated.value - float(np.quantile(s, 0.9))) < 1e-12, str(rated))
    labelled = T.choose_threshold(scoring, y, s, None, effective="anomaly")
    grid = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    want = float(grid[int(np.argmax([f1_at(y, s, t) for t in grid]))])
    check_that("anomaly: without a rate, the f1 optimum on the score's own scale (> 1)",
               labelled.applies and labelled.scale == "score"
               and abs(labelled.value - want) < 1e-12 and labelled.value > 1.0, str(labelled))
    at = scoring.at_threshold(labelled.value)
    check_that("anomaly: f1 is read at the score-scale threshold",
               abs(at.point("f1", y, s) - f1_at(y, s, labelled.value)) < 1e-12)
    blind = T.choose_threshold(scoring, None, s, None, effective="anomaly")
    check_that("anomaly: no rate and no label — no threshold applies",
               not blind.applies and blind.value is None and blind.scale is None, str(blind))
    look = types.SimpleNamespace(
        threshold=blind, effective="anomaly", adapters=types.SimpleNamespace(metrics=M),
        parts=types.SimpleNamespace(score_net=s, scored=types.SimpleNamespace(row_score=s)))
    check_that("anomaly: no threshold, no stored prediction column (never NaN)",
               E._decision(look) is None)
    look = types.SimpleNamespace(
        threshold=rated, effective="anomaly", adapters=types.SimpleNamespace(metrics=M),
        parts=types.SimpleNamespace(score_net=s, scored=types.SimpleNamespace(row_score=s)))
    check_that("anomaly: the stored prediction is row_score ≥ threshold",
               np.array_equal(E._decision(look), (s >= rated.value).astype(int)))


def the_rest(M, D):
    scoring = real_scoring(M, D, "multiclass", "f1_macro")
    none = T.choose_threshold(scoring, np.array([0, 1, 2]), np.eye(3), None,
                              effective="multiclass", applies=False)
    check_that("multiclass: argmax decides, no threshold applies",
               not none.applies and none.value is None and none.metric_threshold() is None)
    record = none.record("multiclass", "multiclass", {})
    check_that("multiclass: threshold.json says applies false with a null value",
               record["applies"] is False and record["value"] is None
               and record["scale"] is None)
    for value, effective, refused in (("abc", "binary", True), ("1.5", "binary", True),
                                      ("nan", "binary", True), ("0.7", "binary", False),
                                      ("1.5", "anomaly", False), ("0.5", "binary", False)):
        got = T.check_threshold_arg(value, effective)
        check_that(f"--threshold {value} on a {effective} reading is "
                   f"{'refused' if refused else 'accepted'} before the seal",
                   (got is not None) == refused, str(got))
    check_that("the intervals are drawn at the recorded level 0.95",
               math.isclose(CI_LEVEL, 0.95) and math.isclose(Resampler(4, 2, 7).ci_level, 0.95))


def main():
    print("scoring threshold — every reported number is read at the run's threshold, on its "
          "scale")
    read_at_the_run_threshold(CHOSEN)
    read_at_the_run_threshold(PER_LABEL)
    the_control()
    M, D = real_metrics()
    binary_threshold_is_a_probability(M, D)
    multilabel_per_label(M, D)
    anomaly_on_the_score_scale(M, D)
    the_rest(M, D)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("OK: the headline, the slices and the robustness checks are all read at the run's "
          "threshold — one value or one per label, on the probability or the score scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
