#!/usr/bin/env python3
"""
test_metrics.py — `templates/metrics.py`, proved on the *rendered* module (the template as the
emitter writes it into `<package>/model/metrics.py`, beside the rendered `domain.py`).

    python3 test_metrics.py [-v] [-k <substring>]
    python3 -m pytest test_metrics.py

What is proved (build plan WP B09, R-C3, R-C7; backend design §11.2):

  logits    E18 — a binary (and a multilabel) logit vector that happens to lie inside [0, 1]
            is read as logits: σ is applied unconditionally, the threshold is a probability.
            Multiclass scores are softmaxed even when their rows sum to one. Where the v2.2
            heuristic did apply σ (some logit outside [0, 1]), every binary number is the one
            v2.2 computed, bit for bit (the golden test compares those epochs). The rank metrics
            read the raw score, so σ's saturation at 1.0 ties nothing.
  mase      against a hand computation with the TRAIN target's seasonal scale (per series,
            pooled); without that scale it is unavailable, never guessed from the scored part.
  anomaly   `anomaly_score_mean` (mode min) on an unlabelled part; the anomaly threshold is on
            the score's own scale; a score-scale threshold under a binary reading is refused.
  labels    per-label `pr_auc` (the catalog's multilabel breakdown) and its macro scalar;
            per-label thresholds; the ordinal reading (R-C6 decision, class probabilities).
  tasks     every metric the forge and eval catalogs name for a task computes on that task's
            score contract shapes.
  surface   no `**kw` (nor `*args`) in a public function, five parameters at most, every
            `METRICS` function is `(y_true, y_score, options=None)`; C7 and C11–C13 on the
            rendered module; numpy and scikit-learn only; `METRICS` keys ⊆
            `catalog.metric_direction`, equal to `coverage.json.metrics`.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import sklearn.metrics as skm

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
TEMPLATE = TEMPLATES / "metrics.py"
CATALOG = SCRIPTS / "catalog.json"
COVERAGE = BACKEND / "coverage.json"
EVAL_CATALOG = SKILLS / "vega-eval" / "scripts" / "catalog.json"
VERBOSE = "-v" in sys.argv

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import conformance  # noqa: E402
import render_templates as rt  # noqa: E402
from vega_core import Level  # noqa: E402


# --------------------------------------------------------------------------- the rendered module

_WORK = tempfile.TemporaryDirectory(prefix="metrics-")
_REC = rt.render(TEMPLATES, _WORK.name, "reference", "reference", only=["metrics.py"])
PACKAGE = _REC["package"]
PKG_DIR = Path(_REC["root"])
sys.path.insert(0, _WORK.name)
M = importlib.import_module(f"{PACKAGE}.model.metrics")
D = importlib.import_module(f"{PACKAGE}.domain")
RENDERED = (PKG_DIR / "model" / "metrics.py").read_text(encoding="utf-8")


def _opts(**kw):
    return D.MetricOptions(**kw)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _v22():
    """c849f8b's v2.2 metrics.py, read from git (`render_templates.baseline_source`), without
    its package import."""
    src = rt.baseline_source("templates/metrics.py")
    assert "from ..domain import NetSpec" in src, "c849f8b's v2.2 metrics.py is not the expected one"
    mod = types.ModuleType("metrics_v22_c849f8b")
    exec(compile(src.replace("from ..domain import NetSpec", "NetSpec = dict"),
                 f"{rt.BASELINE_COMMIT}:templates/metrics.py", "exec"), mod.__dict__)
    return mod


def _hand_ece(p, y, bins):
    """By hand: the bins (lo, hi] of the confidence of the 0.5 decision, each weighted by its
    share of rows, |accuracy − mean confidence| summed."""
    p, y = [float(v) for v in p], [int(v) for v in y]
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        rows = [i for i, v in enumerate(p) if lo < max(v, 1 - v) <= hi]
        if rows:
            hit = sum(int((p[i] >= 0.5) == (y[i] == 1)) for i in rows) / len(rows)
            conf = sum(max(p[i], 1 - p[i]) for i in rows) / len(rows)
            total += len(rows) / len(p) * abs(hit - conf)
    return total


def _binary(n=240, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = (rng.normal(size=n) * 2.0 + (y - 0.5) * 1.5).astype(np.float32)   # a head's float32
    return y, z


# --------------------------------------------------------------------------- E18: logits always

def test_e18_binary_logits_inside_the_unit_interval_are_logits():
    """The regression case: every logit lies in [0, 1], which v2.2 read as probabilities."""
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    z = np.array([0.05, 0.9, 0.3, 0.6, 0.45, 0.2, 0.8, 0.55])
    p = _sig(z)
    decided = (p >= 0.5).astype(int)            # σ(z) ≥ 0.5 ⇔ z ≥ 0: every row positive here
    assert decided.all()
    assert M.metric("log_loss", y, z) == skm.log_loss(y, p, labels=[0, 1])
    assert M.metric("brier", y, z) == float(np.mean((p - y) ** 2))
    assert M.metric("f1", y, z) == skm.f1_score(y, decided, zero_division=0)
    assert M.metric("accuracy", y, z) == skm.accuracy_score(y, decided)
    assert M.metric("specificity", y, z) == 0.0
    assert math.isclose(M.metric("ece", y, z), _hand_ece(p, y, 15), rel_tol=1e-12)
    # ... and not the numbers the heuristic produced
    assert M.metric("log_loss", y, z) != skm.log_loss(y, z, labels=[0, 1])
    assert M.metric("f1", y, z) != skm.f1_score(y, (z >= 0.5).astype(int))
    old = _v22()
    assert old.metric("log_loss", y, z) == skm.log_loss(y, z, labels=[0, 1])   # E18 itself
    assert old.metric("f1", y, z) != M.metric("f1", y, z)


def test_e18_multilabel_logits_inside_the_unit_interval_are_logits():
    y = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
    z = np.array([[0.9, 0.1, 0.4], [0.2, 0.7, 0.3], [0.6, 0.8, 0.1], [0.3, 0.2, 0.9]])
    decided = (_sig(z) >= 0.5).astype(int)
    assert decided.all()
    assert M.metric("hamming", y, z) == skm.hamming_loss(y, decided)
    assert M.metric("f1_micro", y, z) == skm.f1_score(y, decided, average="micro")
    assert M.metric("subset_accuracy", y, z) == 0.0
    assert M.metric("log_loss", y, z) == skm.log_loss(y.reshape(-1), _sig(z).reshape(-1),
                                                      labels=[0, 1])


def test_multiclass_scores_are_softmaxed_even_when_rows_sum_to_one():
    """v2.2 read a logit matrix whose rows summed to 1 as probabilities (`allclose` guess)."""
    y = np.array([2, 0, 1, 2, 1])
    z = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4],
                  [0.25, 0.5, 0.25]])
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    assert math.isclose(M.metric("log_loss", y, z), skm.log_loss(y, p, labels=[0, 1, 2]),
                        rel_tol=1e-12)
    assert M.metric("log_loss", y, z) != skm.log_loss(y, z, labels=[0, 1, 2])
    # a baseline's log-probabilities read as the probabilities they are the log of
    q = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4],
                  [0.3, 0.4, 0.3]])
    assert math.isclose(M.metric("log_loss", y, np.log(q)),
                        skm.log_loss(y, q, labels=[0, 1, 2]), rel_tol=1e-12)
    expected_brier = np.mean(np.sum((q - np.eye(3)[y]) ** 2, axis=1))
    assert math.isclose(M.metric("brier", y, np.log(q)), expected_brier, rel_tol=1e-12)


def test_binary_numbers_equal_v22_where_it_applied_sigmoid():
    """Where some logit lies outside [0, 1] v2.2 applied σ, and every binary number is the one it
    computed, bit for bit — the epochs the golden test compares (D1 §17.3)."""
    y, z = _binary()
    assert z.min() < 0.0 or z.max() > 1.0
    p = _sig(z.astype(np.float64))
    names = ["roc_auc", "pr_auc", "log_loss", "brier", "ece", "accuracy", "balanced_accuracy",
             "precision", "recall", "f1", "f1_macro", "f1_micro", "f1_weighted",
             "f1_per_class", "mcc", "specificity"]
    for threshold in (0.5, 0.3):
        decided = (M.sigmoid(z) >= threshold).astype(int)
        hand = {"roc_auc": skm.roc_auc_score(y, p), "pr_auc": skm.average_precision_score(y, p),
                "log_loss": skm.log_loss(y, M.sigmoid(z), labels=[0, 1]),
                "brier": float(np.mean((M.sigmoid(z) - y) ** 2)),
                "accuracy": skm.accuracy_score(y, decided),
                "f1": skm.f1_score(y, decided, zero_division=0),
                "mcc": skm.matthews_corrcoef(y, decided)}
        new, unavailable = M.compute(names, y, z, _opts(threshold=threshold))
        assert not unavailable, unavailable
        for name, value in hand.items():
            assert new[name] == value, (name, threshold, new[name], value)
        old = _v22()
        for name in names:
            assert new[name] == old.metric(name, y, z, threshold=threshold), (name, threshold)


def test_rank_metrics_read_the_raw_score():
    """σ saturates to 1.0 above ~36.7: ranking by σ would tie rows the logits order."""
    y = np.array([0, 1, 0, 1, 0, 1, 0])
    z = np.array([40.0, 41.0, 42.0, 45.0, -3.0, 0.0, 39.0])
    assert len(set(_sig(z[[0, 1, 2, 3]]))) == 1          # tied under σ
    assert M.metric("roc_auc", y, z) == skm.roc_auc_score(y, z)
    assert M.metric("roc_auc", y, z) != skm.roc_auc_score(y, _sig(z))
    assert M.metric("pr_auc", y, z) == skm.average_precision_score(y, z)
    top = np.argsort(-z, kind="stable")[:3]
    assert M.metric("precision_at_k", y, z, _opts(k=3)) == y[top].sum() / 3


# --------------------------------------------------------------------------- mase

def _train_scale(train, season):
    """By hand: the mean absolute seasonal difference of the train target, per series (a
    difference never crosses a series), pooled over every series' differences."""
    diffs = []
    for values in train.values():
        for t in range(season, len(values)):
            diffs.append(abs(values[t] - values[t - season]))
    return sum(diffs) / len(diffs)


def test_mase_against_a_hand_computation_with_the_train_side_seasonal_scale():
    train = {"a": [10.0, 12.0, 15.0, 11.0, 13.0, 18.0, 12.0, 14.0, 20.0],
             "b": [3.0, 4.0, 2.0, 5.0, 5.0, 3.0, 6.0]}
    season = 3
    scale = _train_scale(train, season)
    assert math.isclose(scale, (1 + 1 + 3 + 1 + 1 + 2 + 2 + 1 + 1 + 1) / 10)
    truth = np.array([[13.0, 16.0], [11.0, 21.0], [4.0, 7.0], [6.0, 2.0]])     # (N, H=2)
    pred = np.array([[12.0, 17.5], [14.0, 18.0], [4.5, 5.0], [6.0, 3.0]])
    errors = [abs(t - p) for row_t, row_p in zip(truth.tolist(), pred.tolist())
              for t, p in zip(row_t, row_p)]
    expected = (sum(errors) / len(errors)) / scale
    got = M.metric("mase", truth, pred, _opts(season=season, mase_scale=scale))
    assert math.isclose(got, expected, rel_tol=1e-12), (got, expected)
    # the scale is the TRAIN side's: the scored part's own seasonal scale would give another
    in_sample = np.mean(np.abs(truth.reshape(-1)[1:] - truth.reshape(-1)[:-1]))
    assert not math.isclose(got, (sum(errors) / len(errors)) / in_sample)


def test_mase_without_the_train_scale_is_unavailable_not_guessed():
    truth, pred = np.array([[1.0, 2.0], [3.0, 5.0]]), np.array([[1.5, 2.0], [2.0, 5.0]])
    values, unavailable = M.compute(["mase", "smape"], truth, pred, _opts(season=2))
    assert "mase" not in values and "mase_scale" in unavailable["mase"], unavailable
    assert "smape" in values
    for bad in (0.0, float("nan"), -1.0):
        _, unavailable = M.compute(["mase"], truth, pred, _opts(mase_scale=bad))
        assert "scale" in unavailable["mase"], (bad, unavailable)


def test_forecast_readings():
    truth = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0]])
    pred = np.array([[1.5, 2.0, 2.0], [2.0, 3.0, 4.0]])
    assert M.metric("per_horizon_error", truth, pred) == [0.25, 0.5, 0.5]
    assert M.metric("mae", truth, pred) == float(np.mean(np.abs(pred - truth)))


# --------------------------------------------------------------------------- anomaly

def test_anomaly_score_mean_reads_no_truth():
    s = np.array([0.4, 2.5, 0.1, 7.0, 0.3])
    assert M.metric("anomaly_score_mean", None, s) == float(np.mean(s))
    assert M.metric("anomaly_score_mean", None, s.reshape(-1, 1)) == float(np.mean(s))
    values, unavailable = M.compute(["anomaly_score_mean", "pr_auc", "reconstruction_error"],
                                    None, s, task="anomaly")
    assert values == {"anomaly_score_mean": float(np.mean(s))}
    assert set(unavailable) == {"pr_auc", "reconstruction_error"}
    assert all("label" in why for why in unavailable.values()), unavailable
    _, unavailable = M.compute(["anomaly_score_mean"], None, np.ones((5, 3)))
    assert "one anomaly score per row" in unavailable["anomaly_score_mean"]
    direction = json.loads(CATALOG.read_text(encoding="utf-8"))["metric_direction"]
    assert direction["anomaly_score_mean"] == "min"


def test_anomaly_threshold_is_on_the_score_scale():
    """threshold.json scale "score": an anomaly decides score ≥ t, never σ(score) ≥ t."""
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 120)
    s = np.abs(rng.normal(size=120)) * 2.0 + y * 3.0         # a reconstruction error: ≥ 0
    t = float(np.quantile(s, 0.8))
    decided = (s >= t).astype(int)
    opts = _opts(threshold=t)
    assert M.metric("f1", y, s, opts, "anomaly") == skm.f1_score(y, decided, zero_division=0)
    assert M.metric("recall", y, s, opts, "anomaly") == skm.recall_score(y, decided)
    assert M.metric("pr_auc", y, s, opts, "anomaly") == skm.average_precision_score(y, s)
    # the same threshold under a binary reading is the scale mismatch, refused
    _, unavailable = M.compute(["f1"], y, s, opts)
    assert "outside [0, 1]" in unavailable["f1"], unavailable
    # an anomaly score is no probability
    _, unavailable = M.compute(["log_loss", "brier", "ece"], y, s, opts, "anomaly")
    assert set(unavailable) == {"log_loss", "brier", "ece"}, unavailable


# --------------------------------------------------------------------------- labels and classes

def test_multilabel_pr_auc_is_the_per_label_list():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, (80, 4))
    y[:, 2] = 0                                           # a label that never occurs
    z = rng.normal(size=(80, 4)) + y
    per_label = M.metric("pr_auc", y, z)
    assert isinstance(per_label, list) and len(per_label) == 4
    for c in (0, 1, 3):
        assert per_label[c] == skm.average_precision_score(y[:, c], z[:, c])
        assert per_label[c] == skm.average_precision_score(y[:, c], _sig(z[:, c]))
    assert math.isnan(per_label[2])
    assert M.metric("pr_auc_macro", y, z) == float(np.mean([per_label[c] for c in (0, 1, 3)]))
    assert M.metric("f1_per_class", y, z) == [
        float(v) for v in skm.f1_score(y, (_sig(z) >= 0.5).astype(int), average=None,
                                       zero_division=0)]


def test_multilabel_thresholds_one_per_label():
    y = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]])
    z = np.array([[0.2, -0.3, 1.0], [-1.0, 0.1, 0.5], [0.6, 0.9, -2.0], [-0.1, -0.2, 0.7],
                  [0.9, 0.3, -0.4]])
    t = (0.5, 0.55, 0.65)
    decided = (_sig(z) >= np.array(t)).astype(int)
    assert M.metric("f1_micro", y, z, _opts(threshold=t)) == skm.f1_score(y, decided,
                                                                         average="micro")
    assert M.metric("hamming", y, z, _opts(threshold=t)) == skm.hamming_loss(y, decided)
    _, unavailable = M.compute(["f1_micro"], y, z, _opts(threshold=(0.5, 0.5)))
    assert "one per label" in unavailable["f1_micro"], unavailable


def _ordinal(n=300, seed=7):
    """A rank-consistent head: one shared score, ordered thresholds (K = 4 classes)."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n) * 1.5
    y = np.digitize(latent + rng.normal(size=n) * 0.7, [-1.0, 0.5, 2.0])
    z = latent[:, None] - np.array([-1.0, 0.5, 2.0])[None, :]
    return y, z


def test_ordinal_reading():
    y, z = _ordinal()
    decided = (_sig(z) >= 0.5).sum(axis=1)                 # R-C6: the cumulative count
    above = _sig(z)
    proba = np.hstack([1 - above[:, :1], above[:, :-1] - above[:, 1:], above[:, -1:]])
    assert np.all(proba >= 0) and np.allclose(proba.sum(axis=1), 1.0)
    got = M.metric("accuracy", y, z, None, "ordinal")
    assert got == skm.accuracy_score(y, decided)
    assert got != M.metric("accuracy", y, z)               # the multiclass (argmax) reading
    assert M.metric("mae_ordinal", y, z) == float(np.mean(np.abs(decided - y)))
    assert M.metric("qwk", y, z) == skm.cohen_kappa_score(y, decided, weights="quadratic")
    assert math.isclose(M.metric("log_loss", y, z, None, "ordinal"),
                        skm.log_loss(y, proba, labels=[0, 1, 2, 3]), rel_tol=1e-12)
    assert math.isclose(M.metric("brier", y, z, None, "ordinal"),
                        float(np.mean(np.sum((proba - np.eye(4)[y]) ** 2, axis=1))),
                        rel_tol=1e-12)
    from scipy.stats import spearmanr
    assert math.isclose(M.metric("spearman", y, z, None, "ordinal"),
                        spearmanr(y, above.sum(axis=1)).correlation, rel_tol=1e-12)
    assert M.metric("f1_per_class", y, z, None, "ordinal") == [
        float(v) for v in skm.f1_score(y, decided, labels=[0, 1, 2, 3], average=None,
                                       zero_division=0)]
    _, unavailable = M.compute(["mae_ordinal"], y, z, None, "multiclass")
    assert "cumulative logits" in unavailable["mae_ordinal"]


def test_spearman_shares_ranks_between_ties():
    y = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0])
    p = np.array([1.2, 1.9, 2.4, 2.8, 4.4, 3.9, 5.5])
    from scipy.stats import spearmanr
    assert math.isclose(M.metric("spearman", y, p), spearmanr(y, p).correlation, rel_tol=1e-12)


def test_multiclass_readings():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 5, 200)
    z = rng.normal(size=(200, 5)) + np.eye(5)[y] * 1.5
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    assert M.metric("accuracy", y, z) == skm.accuracy_score(y, z.argmax(axis=1))
    assert M.metric("top_k", y, z, _opts(k=2)) == skm.top_k_accuracy_score(y, z, k=2)
    assert math.isclose(M.metric("roc_auc", y, z),
                        skm.roc_auc_score(y, p, multi_class="ovr", average="macro"),
                        rel_tol=1e-12)
    per_class = M.metric("pr_auc", y, z)
    assert per_class == [skm.average_precision_score(y == c, p[:, c]) for c in range(5)]
    # a class absent from the truth: its per-class values stay aligned, in class order
    keep = y != 3
    assert len(M.metric("f1_per_class", y[keep], z[keep])) == 5
    assert math.isnan(M.metric("pr_auc", y[keep], z[keep])[3])


# --------------------------------------------------------------------------- every task

def _eval_catalog():
    return json.loads(EVAL_CATALOG.read_text(encoding="utf-8")) if EVAL_CATALOG.is_file() \
        else None


def _task_names(kind):
    """Every metric the forge catalog and (when present) the eval catalog name for a task."""
    known = set(M.METRICS)
    forge = json.loads(CATALOG.read_text(encoding="utf-8"))["tasks"][kind]
    names = [forge.get("primary"), (forge.get("primary_alt") or {}).get("metric"),
             *(forge.get("secondary") or [])]
    evaluation = _eval_catalog()
    if evaluation is not None:
        row = evaluation["tasks"].get(kind) or {}
        names += list(row.get("secondary") or [])
        names += [w for w in str(row.get("primary", "")).replace(",", " ").split() if w in known]
    return sorted({n for n in names if n and n in known})


def _task_cases():
    rng = np.random.default_rng(21)
    n = 300
    yb, zb = _binary(n, 1)
    ym = rng.integers(0, 5, n)
    zm = rng.normal(size=(n, 5)) + np.eye(5)[ym] * 2.0
    yl = rng.integers(0, 2, (n, 3))
    zl = rng.normal(size=(n, 3)) + (yl - 0.5) * 2.0
    yr = rng.normal(size=n) * 3.0 + 10.0
    yo, zo = _ordinal(n, 2)
    x = rng.normal(size=(n, 4))
    ya = rng.integers(0, 2, n)
    sa = np.abs(rng.normal(size=n)) + ya * 2.0
    yf = rng.normal(size=(n, 3)) + 5.0
    return {
        "binary": ("binary", yb, zb, _opts()),
        "multiclass": ("multiclass", ym, zm, _opts(k=3)),
        "multilabel": ("multilabel", yl, zl, _opts()),
        "regression": ("regression", yr, yr + rng.normal(size=n) * 0.5, _opts()),
        "ordinal": ("ordinal", yo, zo, _opts()),
        "unsupervised": ("unsupervised", x, x + rng.normal(size=x.shape) * 0.2, _opts()),
        "anomaly": ("anomaly", ya, sa, _opts(threshold=float(np.quantile(sa, 0.5)), rate=0.5)),
        "sequence-classification": ("binary", yb, zb, _opts()),
        "sequence-regression": ("regression", yr, yr * 0.9 + 1.0, _opts()),
        "forecast": ("forecast", yf, yf + rng.normal(size=yf.shape) * 0.3,
                     _opts(season=24, mase_scale=0.8)),
    }


def _finite(value):
    if isinstance(value, dict):
        value = list(value.values())
    return all(math.isfinite(v) for v in (value if isinstance(value, list) else [value]))


def test_every_task_computes_every_metric_the_catalogs_name():
    for kind, (effective, y, s, opts) in _task_cases().items():
        names = _task_names(kind)
        assert names, kind
        values, unavailable = M.compute(names, y, s, opts, effective)
        assert not unavailable, (kind, unavailable)
        bad = [n for n, v in values.items() if not _finite(v)]
        assert not bad, (kind, bad, {n: values[n] for n in bad})
        if VERBOSE:
            print(f"    {kind}: {', '.join(names)}")


def test_a_task_value_and_for_task_bind_the_reading():
    y, z = _ordinal()
    task = D.Task(kind="ordinal", labels=("a", "b", "c", "d"))
    bound = M.for_task(task)
    assert bound("accuracy", y, z) == M.metric("accuracy", y, z, None, "ordinal")
    assert bound("accuracy", y, z, D.MetricOptions()) == M.metric("accuracy", y, z, None,
                                                                    D.TaskKind.ORDINAL)
    scored = D.ScoredPart(score=z, truth=y, row_ids=np.arange(len(y)))
    record = D.Metrics.from_predictions(D.Part.VALID, scored, ["accuracy", "mae_ordinal"],
                                        bound, D.MetricOptions())
    assert record["accuracy"] == M.metric("accuracy", y, z, None, task)
    sequence = D.Task(kind="sequence-classification", base="binary", labels=("n", "p"))
    yb, zb = _binary(50, 4)
    assert M.for_task(sequence)("f1", yb, zb) == M.metric("f1", yb, zb, None, "binary")
    for bad in ("sequence-regression", "not-a-kind"):
        try:
            M.for_task(bad)
        except ValueError:
            continue
        raise AssertionError(f"for_task({bad!r}) was not refused")


def test_value_metrics_refuse_a_label_reading():
    yb, zb = _binary(40, 2)
    _, unavailable = M.compute(["rmse", "r2"], yb, zb, None, "binary")
    assert set(unavailable) == {"rmse", "r2"}
    _, unavailable = M.compute(["accuracy"], np.ones(5), np.ones(5), None, "regression")
    assert "regression" in unavailable["accuracy"]


def test_r2_of_a_reconstruction_is_per_feature():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3)) * np.array([1.0, 5.0, 0.1])
    r = x + rng.normal(size=x.shape) * 0.05
    assert M.metric("r2", x, r) == skm.r2_score(x, r)
    assert M.metric("per_feature_error", x, r) == [float(v) for v in np.mean((r - x) ** 2, 0)]
    assert M.metric("reconstruction_error", x, r) == float(np.mean(np.mean((r - x) ** 2, 1)))


# --------------------------------------------------------------------------- the entry points

def test_unknown_names_and_mistaken_calls():
    y, z = _binary(30, 5)
    try:
        M.metric("f1_per_label", y, z)
        raise AssertionError("an unknown metric was computed")
    except ValueError as exc:
        assert "f1_per_label" in str(exc) and "anomaly_score_mean" in str(exc)
    values, unavailable = M.compute(["f1", "f1_per_label"], y, z)
    assert "f1" in values and "f1_per_label" in unavailable
    for call in (lambda: M.metric("f1", y, z, threshold=0.3),          # v2.2's keyword knobs
                 lambda: M.metric("f1", y, z, {"threshold": 0.3}),
                 lambda: M.compute(["f1"], y, z, {"threshold": 0.3})):
        try:
            call()
        except TypeError:
            continue
        raise AssertionError("a keyword knob was accepted")
    try:
        M.compute(["f1"], y, z, None, "sequence-classification")
        raise AssertionError("a sequence kind without its base was read")
    except ValueError as exc:
        assert "base" in str(exc)


def test_names_for():
    spec = {"metrics": {"primary": "f1_micro", "secondary": ["pr_auc", "hamming"]}}
    assert M.names_for(spec) == ["f1_micro", "pr_auc", "hamming"]
    assert M.names_for({"metrics": {"primary": None, "secondary": []}}) == []
    # final rehearsal round 2 (improve, 3 of 3 audits): a name the package's task does not
    # compute is the caller's error, refused where the names are derived — never a record of
    # a missing number two hundred epochs later
    import pytest
    with pytest.raises(ValueError, match="'f1_macroo'"):
        M.names_for({"metrics": {"primary": "f1_macroo", "secondary": []}})


def test_the_multilabel_metrics_refuse_another_reading():
    """hamming and subset_accuracy read a k-hot truth: on a multiclass score they refuse
    (recorded unavailable in a pass), where hamming used to answer 1 - accuracy."""
    import pytest
    y, z = np.array([0, 1, 2, 1]), np.eye(3)[[0, 2, 2, 1]] * 4.0
    for name in ("hamming", "subset_accuracy"):
        with pytest.raises(ValueError, match="multiclass reading"):
            M.metric(name, y, z, None, "multiclass")
    values, unavailable = M.compute(["accuracy", "hamming"], y, z, None, "multiclass")
    assert values == {"accuracy": 0.75} and "hamming" in unavailable


# --------------------------------------------------------------------------- the surface

def _public_defs(tree):
    """Every def a caller can reach: module-level functions and the methods of module-level
    classes, `_prefixed` ones and dunders excluded (a nested def is local)."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            yield node.name, node
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    yield f"{node.name}.{item.name}", item


def test_no_keyword_knobs_in_a_public_function():
    for text, where in ((TEMPLATE.read_text(encoding="utf-8"), "template"),
                        (RENDERED, "rendered")):
        tree = ast.parse(text)
        public = dict(_public_defs(tree))
        assert {"metric", "compute", "for_task", "anomaly_score_mean"} <= set(public)
        for name, fn in public.items():
            a = fn.args
            assert a.kwarg is None and a.vararg is None, f"{where}: {name} has *args/**kwargs"
            count = len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs)
            assert count <= 5, f"{where}: {name} takes {count} parameters"
    assert list(inspect.signature(M.metric).parameters) == ["name", "y_true", "y_score",
                                                            "options", "task"]
    assert list(inspect.signature(M.compute).parameters) == ["names", "y_true", "y_score",
                                                             "options", "task"]
    for name, fn in M.METRICS.items():
        params = inspect.signature(fn).parameters
        assert list(params) == ["y_true", "y_score", "options"], (name, list(params))
        assert params["options"].default is None, name


def test_rendered_module_passes_c7_and_the_size_limits():
    pkg, err = conformance.Package.read(PKG_DIR, {}, {})
    assert err is None, err
    bad = [m for m in conformance.c7_no_module_level_mutable_state(pkg)
           if m.level is Level.ERROR]
    assert not bad, [m.line() for m in bad]
    findings = conformance.size_findings(ast.parse(RENDERED), "model/metrics.py")
    assert not findings, [f.line() if hasattr(f, "line") else str(f) for f in findings]


def test_numpy_and_scikit_learn_only():
    """The evaluator loads stored predictions, not a network: no framework import, and the
    package is reached only through `domain` (C4's implicit arrow)."""
    tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add(("." * node.level) + (node.module or ""))
    assert roots <= {"__future__", "dataclasses", "typing", "numpy", "sklearn", "..domain"}, \
        roots
    assert "{{" not in RENDERED


def test_metrics_keys_and_the_catalogs():
    """METRICS ⊆ catalog.metric_direction (R-C7), = coverage.json.metrics (the backend declares
    what it implements), and every metric a task row of the forge or the eval catalog names is
    a METRICS key (eval-contract §3: metrics.py implements all of them)."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    directions = catalog["metric_direction"]
    missing = sorted(set(M.METRICS) - set(directions))
    assert not missing, f"METRICS without a metric_direction row: {missing}"
    assert directions["anomaly_score_mean"] == "min"
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    assert set(coverage["metrics"]) == set(M.METRICS), \
        sorted(set(coverage["metrics"]) ^ set(M.METRICS))
    named = set()
    for row in catalog["tasks"].values():
        alt = row.get("primary_alt") or {}
        named |= {n for n in (row.get("primary"), alt.get("metric")) if n
                  and not str(n).startswith("as the base")}
        named |= set(row.get("secondary") or [])
    evaluation = _eval_catalog()
    if evaluation is not None:
        for row in evaluation["tasks"].values():
            named |= set(row.get("secondary") or [])
            named |= set(row.get("threshold_dependent") or [])
    assert named <= set(M.METRICS), sorted(named - set(M.METRICS))
    assert "f1_per_label" not in M.METRICS                   # R-C7: the breakdown is pr_auc
    # the dict literal the catalog syncs read with ast (forge and show)
    literal = [n for n in ast.parse(TEMPLATE.read_text(encoding="utf-8")).body
               if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "METRICS"
                                                     for t in n.targets)]
    assert len(literal) == 1 and isinstance(literal[0].value, ast.Dict)
    assert [k.value for k in literal[0].value.keys] == list(M.METRICS)


# --------------------------------------------------------------------------- runner


def test_a_compute_pass_prepares_each_family_once():
    """The T5 round 2 improve pass: `compute` re-derived the float64 scores and the decision
    once per metric name. A pass now prepares each family (decision, probabilities, ranking)
    once and every name reads it; the numbers are the per-name ones, and a lone `metric`
    call still prepares its own."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 90)
    z = rng.normal(size=(90, 3)).astype(np.float32)
    calls = []
    reading = M.reading_for("multiclass")
    real = type(reading).decide
    type(reading).decide = lambda self, s, r: calls.append(1) or real(self, s, r)
    try:
        names = ["accuracy", "balanced_accuracy", "f1_macro", "f1_micro", "precision"]
        values, unavailable = M.compute(names, y, z, None, "multiclass")
        assert len(calls) == 1 and not unavailable, (calls, unavailable)
        for name in names:
            assert values[name] == M.metric(name, y, z, None, "multiclass"), name
        assert len(calls) == 1 + len(names)
    finally:
        type(reading).decide = real


def main(argv) -> int:
    pick = argv[argv.index("-k") + 1] if "-k" in argv else ""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and pick in n]
    failed = []
    for name, fn in tests:
        try:
            fn()
            if VERBOSE:
                print(f"  ok    {name}")
        except Exception as exc:                            # a failing test is reported, not raised
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"\n{len(failed)} of {len(tests)} metric test(s) failed")
        return 1
    print(f"\n{len(tests)} metric test(s) passed on the rendered metrics.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


# --------------------------------------------------------------------------- one record shape

def test_one_unavailable_policy_for_every_producer_of_a_record():
    """The T5 round 1 audit: the Trainer's record kept a failed metric as None while the
    Evaluator's `measure` dropped it, two shapes of the one Metrics artifact. `record` is the
    shape both write: every metric asked for, a failed one None, the reasons under
    `_unavailable`. Final rehearsal round 2 (improve, 2 of 3 audits): the Evaluator recorded
    through `for_task(task, unavailable)`, a closure mutating its out-parameter; it calls
    `compute` now, the Trainer's pass, and `for_task` is pure (a failure raises)."""
    import inspect
    import pytest
    s = np.array([0.4, 2.5, 0.1, 7.0, 0.3])
    names = ["anomaly_score_mean", "pr_auc"]
    computed, unavailable = M.compute(names, None, s, task="anomaly")
    trainer = M.record(computed, unavailable)
    assert set(trainer) == set(names) | {"_unavailable"} and trainer["pr_auc"] is None
    assert trainer["_unavailable"] == unavailable
    assert M.record(computed, {}) == computed, "a full pass carries no _unavailable"
    assert list(inspect.signature(M.for_task).parameters) == ["task"]
    with pytest.raises(M.METRIC_FAILURES):
        M.for_task("anomaly")("pr_auc", None, s)        # pure: a failure raises


def test_the_kind_tables_are_keyed_on_the_domain_enum():
    """The T5 round 3 audit (finding 10): the closed set of task kinds is `TaskKind`, so every
    table metrics.py keys on a kind is keyed on its members — a mistyped kind is an
    AttributeError at import, never a silent table miss."""
    tables = {n: getattr(M, n) for n in ("_SEQUENCE_KINDS", "_READINGS")}
    for name, table in tables.items():
        keys = list(table)
        assert keys and all(type(k) is D.TaskKind for k in keys), (name, keys)
    # the per-kind readings are one table of strategies, each answering for its own kind (the
    # final-round audit: no parallel kind tables beside the readout)
    for kind, reading in M._READINGS.items():
        assert reading.kind is kind and M.reading_for(kind) is reading, kind
    for gone in ("_DECIDE", "_PROBA", "_RANKED", "_SCORE_RANK", "_PLAIN_AVERAGE", "_ONE_COLUMN",
                 "_CLASSES", "_LABEL_KINDS"):
        assert not hasattr(M, gone), gone
