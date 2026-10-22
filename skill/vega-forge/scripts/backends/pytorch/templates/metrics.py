"""
The metric functions — the backend's one implementation.

Emitted by /vega-forge into `{{PACKAGE}}/{{MODULE_MODEL}}/metrics.py` (a `kind: support` file of
the Model's module, used by the Trainer, the Evaluator and the Predictor), so the trainer's valid
numbers, the evaluator's test numbers and the notebooks' numbers are the same code
(backend-contract §1: "no second implementation, torchmetrics included"). numpy and
scikit-learn only — nothing here imports the framework, because the evaluator loads stored
predictions, not a network.

Entry points (C12: five parameters at most, no keyword knobs):

    metric(name, y_true, y_score, options=None, task=None) -> float | list | dict
    compute(names, y_true, y_score, options=None, task=None) -> (values, unavailable)
    record(values, unavailable) -> the record every producer of a `Metrics` writes
    for_task(task) -> `metric` bound to one task, for `Metrics.from_predictions(..., metric)`

What "unavailable" means is decided once, here: a metric whose own computation fails (one of
`METRIC_FAILURES`) is recorded by name with its reason, never guessed and never raised, and
the record (`record`) carries every metric asked for — a failed one with the value None — and
the reasons under `_unavailable`. The Trainer's per-epoch record, its `evaluate` and the
Evaluator's `Metrics` all go through it, so a consumer reads the same key set for the same
spec whoever produced the record.

Every function of `METRICS` takes `(y_true, y_score, options)`. `options` is the domain's
`MetricOptions`: `threshold`, `k`, `rate`, `bins` and `mase_scale` are read here; `season` and
`quantiles` travel with them for the scale's provenance and the quantile readings, and no
metric reads them (the `mase` scale arrives computed). `y_score` is what `TaskReadout.score`
returns (backend design §3.4) — logits, per-row scores or values — never a thresholded label:
the threshold is the evaluator's decision and arrives in `options`.

How a score is read (this fixes E18: nothing is ever guessed from the values):

  binary      a logit, always: the probability is σ(score), the threshold a probability;
  multilabel  one logit per label, σ per label; the threshold is a probability, one for
              every label or one per label (a sequence in class order);
  multiclass  logits (or a baseline's log-probabilities): softmax, always;
  ordinal     K−1 cumulative logits: the class is Σ_j [σ(z_j) ≥ 0.5] (R-C6), the class
              probabilities p0 = 1−σ(z0), pj = σ(z_{j−1})−σ(z_j), p_{K−1} = σ(z_{K−2});
  anomaly     a per-row anomaly score, not a probability: its threshold is on the score's
              own scale (`threshold.json` scale "score");
  values      regression, forecast, unsupervised: the score is the value.

The rank metrics (`roc_auc`, `pr_auc`, `precision_at_k`, `recall_at_k`) read a one-column
score raw: σ is monotone, so the ranking is σ's, and the raw score keeps apart the rows σ
saturates to 1.0.

Each effective kind's reading of its score is one `ScoreReading` (`reading_for(kind)`, the one
table): its decision, the probabilities it means, what the rank metrics order by, the plain
average of a precision, recall or F1, and the two inverses the chain needs — the score whose
decision a reduced train prediction is (the Trainer's record) and the head a probability is
written as (a baseline's). `TaskReadout.scores` hands the network's code the same object, so no
second table of per-kind semantics exists outside this one and the readout's.

`task` names the effective kind: a `Task` (read through `.effective`), a `TaskKind` or its
value. Without it the kind follows from the shapes the score contract fixes: a one-column score
is a binary logit, a score shaped like a two-dimensional truth is multilabel, any other matrix
is multiclass. An ordinal and an anomaly reading cannot be told from their shapes (K−1
cumulative logits look like logits, a per-row score looks like a logit), so their callers pass
`task` — `for_task(spec task)` for the port's `Metrics.from_predictions`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from ..domain import MetricOptions, NetSpec, RowIdKind, ScoredPart, TaskKind, host_array

__all__ = ["METRICS", "METRIC_FAILURES", "ScoreReading", "ScoringPass", "clip", "compute",
           "for_task", "joined", "logit", "metric", "names_for", "reading_for", "record",
           "sigmoid", "softmax"]


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


#: what a probability is clipped to before a log is taken of it
_EPS = 1e-7


def clip(p):
    """Probabilities clipped to `[1e-7, 1 − 1e-7]`: what a head built from them takes the log
    of."""
    return np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)


def logit(p):
    """`log(p / (1 − p))` of the clipped probabilities."""
    p = clip(p)
    return np.log(p) - np.log1p(-p)


# ------------------------------------------------------------------ how a call reads its score

_SEQUENCE_KINDS = (TaskKind.SEQUENCE_CLASSIFICATION, TaskKind.SEQUENCE_REGRESSION)
# >>> task: binary multilabel ordinal
#: the logit magnitude a decided 0/1 is fed to the metrics as: σ(±10) sits on its side of any
#: probability threshold in (4.6e-5, 1 − 4.6e-5), so the metric's decision is the reduced one
_DECIDED_LOGIT = 10.0
# <<< task
# >>> task: regression forecast ordinal unsupervised
#: the residual percentiles eval-contract §3 reports for regression
_RESIDUAL_PERCENTILES = (5, 25, 50, 75, 95)
# <<< task


@dataclass(frozen=True)
class _Reading:
    """How one call reads its score: the effective task kind (None: read from the shapes) and
    the knobs of the caller's `MetricOptions` the metrics use."""

    kind: Any = None
    threshold: Any = 0.5
    k: Any = None
    rate: Any = None
    bins: int = 15
    mase_scale: Any = None
    # within one `compute` pass, the scores' preparations each metric family shares (the
    # decision, the probabilities, the ranking), made once for the pass; None elsewhere
    memo: Any = field(default=None, compare=False, repr=False)

    def prepared(self, tag: str, make):
        """`make()`, once per `compute` pass (every call, outside one)."""
        if self.memo is None:
            return make()
        if tag not in self.memo:
            self.memo[tag] = make()
        return self.memo[tag]

    def resolve(self, y, s) -> "ScoreReading":
        """The reading of this call's kind — the caller's, else the one the shapes fix —
        checked against the score's dimensions."""
        reading = reading_for(self.kind if self.kind is not None else _shaped_kind(y, s))
        if reading.rank is not None and s.ndim != reading.rank:
            raise ValueError(f"a {reading.kind.value} score has {reading.rank} dimension(s) "
                             f"(score contract), got shape {tuple(s.shape)}")
        return reading

    # >>> task: binary multilabel
    def probability_threshold(self, width: int):
        """The decision threshold in probability space: one value, or one per label."""
        t = np.asarray(self.threshold, dtype=np.float64)
        if t.ndim > 1 or (t.ndim == 1 and len(t) != width):
            raise ValueError(f"a threshold is one value or one per label ({width}); got "
                             f"{self.threshold!r}")
        if not np.all((t >= 0.0) & (t <= 1.0)):
            raise ValueError(f"threshold {self.threshold!r} is outside [0, 1]: a binary or "
                             "multilabel threshold is a probability; a threshold on an anomaly "
                             "score's scale needs task='anomaly'")
        return float(t) if t.ndim == 0 else t
    # <<< task

    # >>> task: anomaly
    def score_threshold(self) -> float:
        """The decision threshold on an anomaly score's own scale."""
        t = np.asarray(self.threshold, dtype=np.float64)
        if t.ndim != 0 or not np.isfinite(t):
            raise ValueError("an anomaly threshold is one finite value on the score's scale; "
                             f"got {self.threshold!r}")
        return float(t)
    # <<< task


def _shaped_kind(y, s) -> TaskKind:
    if s.ndim == 1:
        return TaskKind.BINARY
    if y is not None and np.ndim(y) == 2 and np.shape(y) == s.shape:
        return TaskKind.MULTILABEL
    return TaskKind.MULTICLASS


def _kind_of(task) -> TaskKind | None:
    """The effective kind a metric reads. A `Task` reads through `.effective`; a sequence kind
    without its base is refused rather than guessed."""
    if task is None:
        return None
    kind = TaskKind(getattr(task, "effective", task))
    if kind in _SEQUENCE_KINDS:
        raise ValueError(f"a {kind.value} task is read as its base (task.base_kind): pass the "
                         "Task or its effective kind")
    return kind


def _reading(options=None, task=None) -> _Reading:
    """The one normalisation of what a caller passed as `options` (and `task`)."""
    if isinstance(options, _Reading):
        return options if task is None else replace(options, kind=_kind_of(task))
    if isinstance(options, Mapping):
        raise TypeError("options is a MetricOptions value (domain.py), not a mapping: a metric "
                        "takes no keyword knobs (C12)")
    o = MetricOptions() if options is None else options
    return _Reading(kind=_kind_of(task), threshold=o.threshold, k=o.k, rate=o.rate,
                    bins=o.bins, mase_scale=o.mase_scale)


def _only(reading: "ScoreReading", defined: bool, name: str, reads: str) -> None:
    if not defined:
        raise ValueError(f"{name} is not defined on a {reading.kind.value} reading (it reads "
                         f"{reads})")


# ------------------------------------------------------------------ arrays

def _1d(a):
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _scores(y_score):
    """The score as float64; a one-column matrix is its column (a (N, 1) head is one score)."""
    s = np.asarray(y_score, dtype=np.float64)
    return s.reshape(-1) if (s.ndim == 2 and s.shape[1] == 1) else s


def _as_float(y_score):
    """The scores as one float64 array, as given (no reshape), for a pass of many metrics; what
    cannot be read so is left to each metric, which records why it is unavailable."""
    try:
        return None if y_score is None else np.asarray(y_score, dtype=np.float64)
    except (TypeError, ValueError):
        return y_score


def _truth(y_true, name):
    if y_true is None:
        raise ValueError(f"{name} compares against a label, and this part has none")
    return np.asarray(y_true)


# >>> task: binary multiclass multilabel ordinal anomaly
def _onehot(y, k):
    idx = np.asarray(y).reshape(-1).astype(int)
    if idx.size and (idx.min() < 0 or idx.max() >= k):
        raise ValueError(f"a class index outside [0, {k}) in the truth")
    return np.eye(k, dtype=int)[idx]
# <<< task


def _sk():
    """scikit-learn's metrics module, imported on first use (the package imports no framework
    it does not run); a call site names the function it reads as an attribute of it."""
    import sklearn.metrics as M
    return M


# >>> task: ordinal
# ------------------------------------------------------------------ cumulative logits

def _cumulative(s):
    """K−1 cumulative logits as a matrix (a two-class ordinal head has one column)."""
    s = np.asarray(s, dtype=np.float64)
    return s.reshape(len(s), -1)


def _ordinal_pred(s):
    """Cumulative logits -> a class index: the number of thresholds the row passes (R-C6)."""
    return (sigmoid(_cumulative(s)) >= 0.5).sum(axis=-1).astype(int)


def _ordinal_proba(s):
    """The class probabilities: P(y = j) = P(y > j−1) − P(y > j), with P(y > −1) = 1 and
    P(y > K−1) = 0. The rank-consistent head keeps them non-negative; any other input is
    clipped and renormalised rather than read as a negative probability."""
    above = sigmoid(_cumulative(s))
    n = len(above)
    p = np.hstack([np.ones((n, 1)), above]) - np.hstack([above, np.zeros((n, 1))])
    p = np.clip(p, 0.0, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _ordinal_expected(s):
    """E[y] = Σ_j P(y > j): the ordinal's continuous reading, for rank correlation."""
    return sigmoid(_cumulative(s)).sum(axis=1)
# <<< task


# ------------------------------------------------------------------ the readings

# >>> task: binary multiclass ordinal
_CLASS_LABEL_METRICS = frozenset({
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "f1_macro", "f1_micro",
    "f1_weighted", "f1_per_class", "mcc", "specificity"})
# <<< task
# >>> task: regression forecast
_VALUE_METRICS = frozenset({
    "rmse", "mae", "median_ae", "r2", "mape", "spearman", "residual_quantiles", "smape",
    "mase", "per_horizon_error"})
# <<< task


class ScoreReading:
    """How the metrics read one effective kind's score (backend design §3.4). A subclass per
    kind; `reading_for(kind)` is the one table. The flags say which metric families the score
    admits: `one_column` (one score per row and a 0/1 truth), `classes` (a class-index truth
    over K classes), `per_label` (a k-hot truth), `label_score` (the score is no value)."""

    kind: TaskKind = None
    rank: int | None = None              # the score's dimensions (score contract); None: any
    one_column = classes = per_label = label_score = False
    average = "binary"                   # a plain precision / recall / f1's average
    label_only = frozenset()             # the metrics a reduced prediction alone computes

    def decide(self, s, r: _Reading):
        """The decision (label metrics), under the call's threshold."""
        raise ValueError(f"a {self.kind.value} score is decided by no threshold")

    def proba(self, s):
        """The probabilities the score means (probability metrics)."""
        raise ValueError(f"a {self.kind.value} score means no probability")

    def ranked(self, s):
        """What the rank metrics order by: the raw score where it is one per row or per
        label, the class probabilities where the classes compete for one row."""
        raise ValueError(f"a {self.kind.value} score ranks nothing")

    def class_scores(self, s):
        """Per-class scores ordered as the classes' likelihood (`top_k`)."""
        raise ValueError(f"a {self.kind.value} score has no classes to rank")

    def n_classes(self, s) -> int:
        return 2

    def decided_input(self, pred, truth, n_classes: int) -> tuple:
        """`(truth, score)` whose metric reading is the reduced prediction `pred` — what the
        Trainer's train-side record feeds a label metric (the reduction is its decision)."""
        return truth, pred

    def head_of_proba(self, p):
        """The head a model of probabilities `p` (n, K) writes, as the network's would be —
        what a probability baseline answers through the readout (the inverse of `proba`)."""
        raise ValueError(f"a {self.kind.value} head is written from no probability")


# >>> task: binary
class _Logit(ScoreReading):
    """binary: a logit, always — σ is the probability, the threshold a probability."""

    kind, rank, one_column, label_score = TaskKind.BINARY, 1, True, True
    label_only = _CLASS_LABEL_METRICS

    def decide(self, s, r):
        p = sigmoid(s)
        return (p >= r.probability_threshold(1)).astype(int)

    def proba(self, s):
        return sigmoid(s)

    def ranked(self, s):
        return s

    def decided_input(self, pred, truth, n_classes):
        return truth, np.where(pred.reshape(-1) > 0, _DECIDED_LOGIT, -_DECIDED_LOGIT)

    def head_of_proba(self, p):
        return logit(np.asarray(p, dtype=np.float64)[:, 1:])
# <<< task


# >>> task: multilabel
class _LabelLogits(ScoreReading):
    """multilabel: one logit per label, σ per label; one threshold, or one per label."""

    kind, rank, per_label, label_score, average = TaskKind.MULTILABEL, 2, True, True, "micro"
    label_only = frozenset({"hamming", "subset_accuracy"})

    def decide(self, s, r):
        p = sigmoid(s)
        return (p >= r.probability_threshold(p.shape[-1] if p.ndim == 2 else 1)).astype(int)

    def proba(self, s):
        return sigmoid(s)

    def ranked(self, s):
        return s

    def n_classes(self, s) -> int:
        return int(_width(s))

    def decided_input(self, pred, truth, n_classes):
        return truth, np.where(pred > 0, _DECIDED_LOGIT, -_DECIDED_LOGIT)

    def head_of_proba(self, p):
        return logit(p)
# <<< task


# >>> task: multiclass
class _Softmax(ScoreReading):
    """multiclass: logits (or a baseline's log-probabilities) — softmax, always."""

    kind, rank, classes, label_score, average = TaskKind.MULTICLASS, 2, True, True, "macro"
    label_only = _CLASS_LABEL_METRICS - {"specificity"}

    def decide(self, s, r):
        return s.argmax(axis=-1)

    def proba(self, s):
        return softmax(s)

    def ranked(self, s):
        return softmax(s)

    def class_scores(self, s):
        return s

    def n_classes(self, s) -> int:
        return int(_width(s))

    def decided_input(self, pred, truth, n_classes):
        return truth, np.eye(n_classes, dtype=np.int8)[pred.reshape(-1).astype(np.int64)]

    def head_of_proba(self, p):
        return np.log(clip(p))
# <<< task


# >>> task: ordinal
class _Cumulative(ScoreReading):
    """ordinal: K−1 cumulative logits; the class is Σ_j [σ(z_j) ≥ 0.5] (R-C6)."""

    kind, classes, average = TaskKind.ORDINAL, True, "macro"
    label_only = _CLASS_LABEL_METRICS | {"mae_ordinal", "qwk"}

    def decide(self, s, r):
        return _ordinal_pred(s)

    def proba(self, s):
        return _ordinal_proba(s)

    def ranked(self, s):
        return _ordinal_proba(s)

    def class_scores(self, s):
        return _ordinal_proba(s)

    def n_classes(self, s) -> int:
        return int(_width(s)) + 1

    def decided_input(self, pred, truth, n_classes):
        ranks = np.arange(max(1, n_classes - 1))[None, :]
        return truth, np.where(ranks < pred.reshape(-1, 1), _DECIDED_LOGIT, -_DECIDED_LOGIT)

    def head_of_proba(self, p):
        p = np.asarray(p, dtype=np.float64)
        return logit(np.cumsum(p[:, ::-1], axis=1)[:, ::-1][:, 1:])     # P(y > j)
# <<< task


# >>> task: anomaly
class _AnomalyScore(ScoreReading):
    """anomaly: a per-row anomaly score, not a probability — its threshold is on the score's
    own scale (`threshold.json` scale "score")."""

    kind, rank, one_column, label_score = TaskKind.ANOMALY, 1, True, True
    label_only = frozenset({"roc_auc", "pr_auc", "precision_at_k", "recall_at_k",
                            "anomaly_score_mean"})

    def decide(self, s, r):
        return (_1d(s) >= r.score_threshold()).astype(int)

    def ranked(self, s):
        return s
# <<< task


# >>> task: regression forecast
class _Values(ScoreReading):
    """regression, forecast: the score is the value."""

    label_only = _VALUE_METRICS

    def __init__(self, kind: TaskKind):
        self.kind = kind
# <<< task


# >>> task: unsupervised
class _Reconstruction(ScoreReading):
    """unsupervised: the score is the reconstruction of the numeric input."""

    kind = TaskKind.UNSUPERVISED
    label_only = frozenset({"reconstruction_error"})

    def decided_input(self, pred, truth, n_classes):
        """Each row's numeric MSE m as a one-column reconstruction √m of a zero input, so
        `reconstruction_error` reads mean(m) (to float64 rounding) with its own code."""
        m = np.asarray(pred, dtype=np.float64).reshape(-1, 1)
        return np.zeros_like(m), np.sqrt(np.maximum(m, 0.0))
# <<< task


#: the reading of each effective kind this package carries (the one table)
_READINGS = {
    # >>> task: binary
    TaskKind.BINARY: _Logit(),
    # <<< task
    # >>> task: multilabel
    TaskKind.MULTILABEL: _LabelLogits(),
    # <<< task
    # >>> task: multiclass
    TaskKind.MULTICLASS: _Softmax(),
    # <<< task
    # >>> task: ordinal
    TaskKind.ORDINAL: _Cumulative(),
    # <<< task
    # >>> task: anomaly
    TaskKind.ANOMALY: _AnomalyScore(),
    # <<< task
    # >>> task: regression
    TaskKind.REGRESSION: _Values(TaskKind.REGRESSION),
    # <<< task
    # >>> task: forecast
    TaskKind.FORECAST: _Values(TaskKind.FORECAST),
    # <<< task
    # >>> task: unsupervised
    TaskKind.UNSUPERVISED: _Reconstruction(),
    # <<< task
}


def reading_for(kind) -> ScoreReading:
    """The reading of an effective kind (a `Task`, a `TaskKind` or its value)."""
    kind = _kind_of(kind)
    reading = _READINGS.get(kind)
    if reading is None:
        raise ValueError(f"this package reads a {' or '.join(k.value for k in _READINGS)} score, "
                         f"not a {kind.value if kind else None} one")
    return reading


# >>> task: multiclass multilabel ordinal
def _width(s) -> int:
    """The score's columns per row (a one-column head has one)."""
    s = np.asarray(s, dtype=np.float64)
    return s.reshape(len(s), -1).shape[1]
# <<< task


# >>> task: binary multiclass multilabel ordinal anomaly
def _decided(y_true, y_score, options, name):
    """(truth, decision, reading) for a label metric (once per `compute` pass)."""
    r = _reading(options)
    return r.prepared("decided", lambda: _decide(y_true, y_score, r, name))


def _decide(y_true, y_score, r, name):
    s, y = _scores(y_score), _truth(y_true, name)
    reading = r.resolve(y, s)
    decision = _defined(reading, name, lambda: reading.decide(s, r))
    return (y if reading.per_label else y.reshape(-1)), decision, reading


def _defined(reading: ScoreReading, name: str, step):
    """A reading's step; its refusal is said with the metric's name and the reading's kind."""
    try:
        return step()
    except ValueError as exc:
        raise ValueError(f"{name} is not defined on a {reading.kind.value} reading ({exc})") \
            from exc


def _probabilities(y_true, y_score, options, name):
    """(truth, probabilities, reading) for a probability metric (once per `compute` pass)."""
    r = _reading(options)
    return r.prepared("probabilities", lambda: _probability(y_true, y_score, r, name))


def _probability(y_true, y_score, r, name):
    s, y = _scores(y_score), _truth(y_true, name)
    reading = r.resolve(y, s)
    p = _defined(reading, name, lambda: reading.proba(s))
    return (y if reading.per_label else y.reshape(-1)), p, reading


def _ranked(y_true, y_score, options, name):
    """(truth, ranking score, reading) for a rank metric; a class truth becomes its indicator
    (once per `compute` pass)."""
    r = _reading(options)
    return r.prepared("ranked", lambda: _rank(y_true, y_score, r, name))


def _rank(y_true, y_score, r, name):
    s, y = _scores(y_score), _truth(y_true, name)
    reading = r.resolve(y, s)
    ranked = _defined(reading, name, lambda: reading.ranked(s))
    if reading.classes:
        y = _onehot(y, ranked.shape[1])
    elif ranked.ndim == 1:
        y = y.reshape(-1)
    return y, ranked, reading


def _per_column(fn, y, s):
    """One value per column, in column order; nan where the column's truth holds one value
    only (the metric is undefined there, not zero)."""
    out = []
    for c in range(y.shape[1]):
        col = np.asarray(y[:, c], dtype=np.float64)
        defined = 0.0 < col.sum() < len(col)
        out.append(float(fn(col, s[:, c])) if defined else float("nan"))
    return out


def _macro(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


# ------------------------------------------------------------------ ranking

def roc_auc(y_true, y_score, options=None):
    """Area under the ROC curve. One column (binary, anomaly): on the raw score. Otherwise the
    macro mean over the labels (multilabel) or classes (one-vs-rest on the class probability)
    whose truth holds both values."""
    y, s, _ = _ranked(y_true, y_score, options, "roc_auc")
    if s.ndim == 1:
        return float(_sk().roc_auc_score(_1d(y), s))
    return _macro(_per_column(_sk().roc_auc_score, y, s))


def pr_auc(y_true, y_score, options=None):
    """Average precision. One column (binary, anomaly): a number, on the raw score. Multilabel:
    the per-label list, in class order (the catalog's breakdown; nan where a label never or
    always occurs); multiclass and ordinal: the per-class list, one-vs-rest on the class
    probability. `pr_auc_macro` is its scalar."""
    y, s, _ = _ranked(y_true, y_score, options, "pr_auc")
    if s.ndim == 1:
        return float(_sk().average_precision_score(_1d(y), s))
    return _per_column(_sk().average_precision_score, y, s)


def pr_auc_macro(y_true, y_score, options=None):
    """The mean of `pr_auc`'s per-label (per-class) values over the defined ones."""
    y, s, _ = _ranked(y_true, y_score, options, "pr_auc_macro")
    if s.ndim == 1:
        return float(_sk().average_precision_score(_1d(y), s))
    return _macro(_per_column(_sk().average_precision_score, y, s))


def _top(y_true, y_score, options, name):
    """(truth, the rows ranked highest, k) of a one-score-per-row reading."""
    y, s, _ = _ranked(y_true, y_score, options, name)
    if s.ndim != 1:
        raise ValueError(f"{name} ranks one score per row; got shape {tuple(s.shape)}")
    y = _1d(y)
    k = _k_from(y, _reading(options))
    return y, np.argsort(-s, kind="stable")[:k], k


def _k_from(y, r):
    """k: the caller's, else the rate times the rows, else the number of positives."""
    n = len(y)
    if r.k:
        k = int(r.k)
    elif r.rate:
        k = max(1, int(round(float(r.rate) * n)))
    else:
        k = max(1, int(y.sum()))
    return min(k, n)


def precision_at_k(y_true, y_score, options=None):
    y, top, k = _top(y_true, y_score, options, "precision_at_k")
    return float(y[top].sum() / k)


def recall_at_k(y_true, y_score, options=None):
    y, top, _ = _top(y_true, y_score, options, "recall_at_k")
    positives = y.sum()
    if not positives:
        return float("nan")
    return float(y[top].sum() / positives)


# ------------------------------------------------------------------ probabilistic

def log_loss(y_true, y_score, options=None):
    """Cross-entropy of the probabilities the score means (per label cell for multilabel)."""
    y, p, reading = _probabilities(y_true, y_score, options, "log_loss")
    if reading.classes:
        return float(_sk().log_loss(y, p, labels=list(range(p.shape[-1]))))
    return float(_sk().log_loss(y.reshape(-1), p.reshape(-1), labels=[0, 1]))


def brier(y_true, y_score, options=None):
    """Mean squared gap between probability and outcome: per row over the classes for
    multiclass and ordinal, per label cell for multilabel."""
    y, p, reading = _probabilities(y_true, y_score, options, "brier")
    if reading.classes:
        return float(np.mean(np.sum((p - _onehot(y, p.shape[1])) ** 2, axis=1)))
    return float(np.mean((p - np.asarray(y, dtype=np.float64)) ** 2))


def ece(y_true, y_score, options=None):
    """Expected calibration error — the gap between confidence and accuracy, bin by bin.
    Binary and multilabel (each label cell): the confidence of the 0.5 decision; multiclass and
    ordinal: the top class probability."""
    y, p, reading = _probabilities(y_true, y_score, options, "ece")
    if reading.classes:
        conf = p.max(axis=-1)
        correct = (p.argmax(axis=-1) == y).astype(float)
    else:
        p, y = p.reshape(-1), y.reshape(-1)
        conf = np.maximum(p, 1.0 - p)
        correct = ((p >= 0.5).astype(int) == y).astype(float)
    return _calibration_gap(conf, correct, _reading(options).bins)


def _calibration_gap(conf, correct, bins):
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


# ------------------------------------------------------------------ decisions

def accuracy(y_true, y_score, options=None):
    """The share of rows decided right (multilabel: every label of the row right)."""
    y, pred, _ = _decided(y_true, y_score, options, "accuracy")
    return float(_sk().accuracy_score(y, pred))


def balanced_accuracy(y_true, y_score, options=None):
    y, pred, reading = _decided(y_true, y_score, options, "balanced_accuracy")
    _only(reading, reading.one_column or reading.classes, "balanced_accuracy",
          "one decision per row")
    return float(_sk().balanced_accuracy_score(y, pred))


def _prf(y_true, y_score, options, which, average):
    name = which if average is None else f"{which}_{average}"
    y, pred, reading = _decided(y_true, y_score, options, name)
    M = _sk()
    fn = {"precision": M.precision_score, "recall": M.recall_score, "f1": M.f1_score}[which]
    return float(fn(y, pred, average=average or reading.average, zero_division=0))


def precision(y_true, y_score, options=None):
    return _prf(y_true, y_score, options, "precision", None)


def recall(y_true, y_score, options=None):
    return _prf(y_true, y_score, options, "recall", None)


def f1(y_true, y_score, options=None):
    return _prf(y_true, y_score, options, "f1", None)


def f1_macro(y_true, y_score, options=None):
    return _prf(y_true, y_score, options, "f1", "macro")


def f1_micro(y_true, y_score, options=None):
    return _prf(y_true, y_score, options, "f1", "micro")


def f1_weighted(y_true, y_score, options=None):
    return _prf(y_true, y_score, options, "f1", "weighted")


def f1_per_class(y_true, y_score, options=None):
    """F1 per class in class order (per label for multilabel); 0 for a class never seen or
    decided."""
    y, pred, reading = _decided(y_true, y_score, options, "f1_per_class")
    labels = None if reading.per_label else list(range(reading.n_classes(_scores(y_score))))
    values = _sk().f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return [float(v) for v in values]


def mcc(y_true, y_score, options=None):
    y, pred, reading = _decided(y_true, y_score, options, "mcc")
    _only(reading, reading.one_column or reading.classes, "mcc", "one decision per row")
    return float(_sk().matthews_corrcoef(y, pred))


def specificity(y_true, y_score, options=None):
    y, pred, reading = _decided(y_true, y_score, options, "specificity")
    _only(reading, reading.one_column, "specificity", "one 0/1 decision per row")
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def top_k(y_true, y_score, options=None):
    """The share of rows whose true class is among the k most probable (k = options.k, else
    3)."""
    r, s = _reading(options), _scores(y_score)
    y = _truth(y_true, "top_k").reshape(-1)
    reading = r.resolve(y, s)
    _only(reading, reading.classes, "top_k", "classes competing for a row")
    p = reading.class_scores(s)
    k = int(min(r.k or 3, p.shape[-1]))
    top = np.argsort(-p, axis=-1)[:, :k]
    return float(np.mean([yi in row for yi, row in zip(y, top)]))


# >>> task: multilabel
def hamming(y_true, y_score, options=None):
    """The share of label cells decided wrong (a k-hot truth: a multilabel reading only)."""
    y, pred, reading = _decided(y_true, y_score, options, "hamming")
    _only(reading, reading.per_label, "hamming", "a k-hot truth")
    return float(_sk().hamming_loss(y, pred))


def subset_accuracy(y_true, y_score, options=None):
    """The share of rows whose every label is decided right (a multilabel reading only)."""
    y, pred, reading = _decided(y_true, y_score, options, "subset_accuracy")
    _only(reading, reading.per_label, "subset_accuracy", "a k-hot truth")
    return float(np.mean(np.all(y == pred, axis=-1)))
# <<< task
# <<< task


# >>> task: regression forecast ordinal unsupervised
# ------------------------------------------------------------------ values

def _value_pair(y_true, y_score, options, name):
    """(truth, value) as float64 arrays. An ordinal's value is its decided class; a score that
    is a logit or an anomaly score is no value and is refused."""
    r = _reading(options)
    if r.kind is not None and reading_for(r.kind).label_score:
        raise ValueError(f"{name} reads values; a {r.kind.value} score is not one")
    y = np.asarray(_truth(y_true, name), dtype=np.float64)
    # >>> task: ordinal
    if r.kind is TaskKind.ORDINAL:
        return y.reshape(-1), _ordinal_pred(y_score).astype(np.float64)
    # <<< task
    return y, np.asarray(y_score, dtype=np.float64)


def _values(y_true, y_score, options, name):
    y, p = _value_pair(y_true, y_score, options, name)
    return _1d(y), _1d(p)


def rmse(y_true, y_score, options=None):
    y, p = _values(y_true, y_score, options, "rmse")
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y_true, y_score, options=None):
    y, p = _values(y_true, y_score, options, "mae")
    return float(np.mean(np.abs(p - y)))


def median_ae(y_true, y_score, options=None):
    y, p = _values(y_true, y_score, options, "median_ae")
    return float(np.median(np.abs(p - y)))


def r2(y_true, y_score, options=None):
    """R². A matrix truth (a reconstruction, a horizon) is scored per column and averaged —
    the unsupervised row's "r2 of the reconstruction per feature"."""
    y, p = _value_pair(y_true, y_score, options, "r2")
    if y.ndim == 2 and y.shape == p.shape and y.shape[1] > 1:
        return float(_sk().r2_score(y, p))
    return float(_sk().r2_score(_1d(y), _1d(p)))


def mape(y_true, y_score, options=None):
    y, p = _values(y_true, y_score, options, "mape")
    m = np.abs(y) > 1e-9
    return float(np.mean(np.abs((y[m] - p[m]) / y[m]))) if m.any() else float("nan")


def spearman(y_true, y_score, options=None):
    """Spearman's ρ: Pearson's correlation of the average ranks. An ordinal is ranked by its
    expected class Σ_j σ(z_j), which keeps apart the rows its decision ties."""
    y, p = _spearman_pair(y_true, y_score, options)
    ry, rp = _average_ranks(y), _average_ranks(p)
    if ry.std() == 0 or rp.std() == 0:
        return float("nan")
    return float(np.corrcoef(ry, rp)[0, 1])


def _spearman_pair(y_true, y_score, options):
    """(truth, value) as spearman ranks them; an ordinal by its expected class."""
    # >>> task: ordinal
    if _reading(options).kind is TaskKind.ORDINAL:
        return _1d(_truth(y_true, "spearman")), _ordinal_expected(y_score)
    # <<< task
    return _values(y_true, y_score, options, "spearman")


def _average_ranks(a):
    """0-based ranks; tied values share the mean of their ranks."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def residual_quantiles(y_true, y_score, options=None):
    """The residual (prediction − truth) at the 5th, 25th, 50th, 75th and 95th percentiles."""
    y, p = _values(y_true, y_score, options, "residual_quantiles")
    r = p - y
    return {str(q): float(np.percentile(r, q)) for q in _RESIDUAL_PERCENTILES}
# <<< task


# >>> task: ordinal
# ------------------------------------------------------------------ ordinal

def _ordinal_decided(y_true, y_score, options, name):
    kind = _reading(options).kind
    if kind not in (None, TaskKind.ORDINAL):
        raise ValueError(f"{name} reads cumulative logits (an ordinal head), not a "
                         f"{kind.value} score")
    return np.asarray(_truth(y_true, name)).reshape(-1), _ordinal_pred(y_score)


def mae_ordinal(y_true, y_score, options=None):
    y, pred = _ordinal_decided(y_true, y_score, options, "mae_ordinal")
    return float(np.mean(np.abs(pred - y)))


def qwk(y_true, y_score, options=None):
    y, pred = _ordinal_decided(y_true, y_score, options, "qwk")
    return float(_sk().cohen_kappa_score(y.astype(int), pred, weights="quadratic"))
# <<< task


# ------------------------------------------------------------------ reconstruction / anomaly

# >>> task: unsupervised
def reconstruction_error(y_true, y_score, options=None):
    """The mean over rows of the per-row mean squared error (truth = the numeric input)."""
    y = np.asarray(_truth(y_true, "reconstruction_error"), dtype=np.float64)
    p = np.asarray(y_score, dtype=np.float64)
    return float(np.mean(np.mean((p - y) ** 2, axis=tuple(range(1, y.ndim)) or (0,))))


def per_feature_error(y_true, y_score, options=None):
    """The mean squared error of each input column, in column order."""
    y = np.asarray(_truth(y_true, "per_feature_error"), dtype=np.float64)
    p = np.asarray(y_score, dtype=np.float64)
    return [float(v) for v in np.atleast_1d(np.mean((p - y) ** 2, axis=0))]
# <<< task


# >>> task: anomaly
def anomaly_score_mean(y_true, y_score, options=None):
    """The mean per-row anomaly score (`TaskReadout.score`: the reconstruction's numeric MSE
    plus its mean nominal cross-entropy, or ‖f(x) − c‖² for svdd) — the anomaly primary when no
    label exists (R-C7, mode min: on held-out data of the kind it was fitted to, a tighter
    model of normal scores lower). It reads no truth."""
    s = _scores(y_score)
    if s.ndim != 1:
        raise ValueError(f"anomaly_score_mean reads one anomaly score per row; got shape "
                         f"{tuple(s.shape)}")
    if not len(s):
        raise ValueError("anomaly_score_mean has no scored row to average")
    return float(np.mean(s))
# <<< task


# ------------------------------------------------------------------ forecast

# >>> task: forecast
def smape(y_true, y_score, options=None):
    y, p = _values(y_true, y_score, options, "smape")
    d = (np.abs(y) + np.abs(p)) / 2.0
    m = d > 1e-12
    return float(np.mean(np.abs(y[m] - p[m]) / d[m])) if m.any() else float("nan")


def mase(y_true, y_score, options=None):
    """Mean absolute scaled error: the mean absolute error over every row and horizon step,
    divided by `options.mase_scale` — the mean absolute seasonal difference (lag = season) of
    the TRAIN target, per series and pooled (a forecast window's DataSource gives it). Without
    that scale it is unavailable, never estimated from the part being scored."""
    scale = _reading(options).mase_scale
    if scale is None:
        raise ValueError("mase needs the train target's seasonal scale (options.mase_scale); "
                         "none was given, and the scored part is not the train part")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"mase's seasonal scale must be finite and > 0, got {scale!r}: a "
                         "constant train target has no scale")
    y, p = _values(y_true, y_score, options, "mase")
    return float(np.mean(np.abs(y - p)) / scale)
# <<< task


def per_horizon_error(y_true, y_score, options=None):
    """The mean absolute error of each horizon step, in step order."""
    y = np.asarray(_truth(y_true, "per_horizon_error"), dtype=np.float64)
    p = np.asarray(y_score, dtype=np.float64)
    if y.ndim == 1:
        return [float(np.mean(np.abs(p - y)))]
    return [float(np.mean(np.abs(p[:, h] - y[:, h]))) for h in range(y.shape[1])]


#: the metrics of the task kind this package carries, by name
METRICS = {
    # >>> task: binary multiclass multilabel ordinal anomaly
    "roc_auc": roc_auc, "pr_auc": pr_auc, "log_loss": log_loss, "brier": brier, "ece": ece,
    "accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "precision": precision,
    "recall": recall, "f1": f1, "f1_macro": f1_macro, "f1_micro": f1_micro,
    "f1_weighted": f1_weighted, "f1_per_class": f1_per_class, "mcc": mcc,
    "specificity": specificity, "top_k": top_k, "pr_auc_macro": pr_auc_macro,
    "precision_at_k": precision_at_k, "recall_at_k": recall_at_k,
    # <<< task
    # >>> task: multilabel
    "hamming": hamming, "subset_accuracy": subset_accuracy,
    # <<< task
    # >>> task: regression forecast ordinal unsupervised
    "rmse": rmse, "mae": mae, "median_ae": median_ae, "r2": r2, "mape": mape,
    "spearman": spearman, "residual_quantiles": residual_quantiles,
    # <<< task
    # >>> task: ordinal
    "mae_ordinal": mae_ordinal, "qwk": qwk,
    # <<< task
    # >>> task: unsupervised
    "reconstruction_error": reconstruction_error, "per_feature_error": per_feature_error,
    # <<< task
    # >>> task: anomaly
    "anomaly_score_mean": anomaly_score_mean,
    # <<< task
    # >>> task: forecast
    "smape": smape, "mase": mase,
    # <<< task
    "per_horizon_error": per_horizon_error,
}


# ------------------------------------------------------------------ the scoring pass

def joined(parts):
    """The batches' host arrays in order, as one; None when there are none or any batch had
    none — the one join of a pass's arrays (the Trainer's record joins through it too)."""
    if not parts or any(p is None for p in parts):
        return None
    return np.concatenate(parts, axis=0)


class ScoringPass:
    """The one scoring pass (backend design §11.1): a model's scores over a part's batches, in
    the order they come, with the truth each is measured against (`TaskReadout.truth`; the
    pass is given the spec's readout) and, for the reconstruction and anomaly readings, the
    per-row score. The Trainer's valid numbers and its predictions, and the Evaluator's scored
    parts, are all made by it:

      * `add(model, batch)` scores through the Model port's scoring surface (`ScoredModel`:
        the network and every baseline alike);
      * `add_head(out, batch, model)` reads a head the caller already computed (the Trainer's
        valid pass, whose one forward also gives the loss).
    """

    def __init__(self, readout):
        self._readout = readout
        self._scores, self._truths, self._rows = [], [], []

    def add(self, model, batch) -> None:
        self._keep(model.score(batch), batch,
                   model.row_score(batch) if self._readout.per_row else None)

    def add_head(self, out, batch, model) -> None:
        self._keep(self._readout.score(out, batch, model), batch,
                   self._readout.row_score(out, batch, model))

    def _keep(self, score, batch, row) -> None:
        self._scores.append(host_array(score))
        self._truths.append(host_array(self._readout.truth(batch)))
        self._rows.append(host_array(row))

    @property
    def empty(self) -> bool:
        """Whether no batch was scored."""
        return not self._scores

    @property
    def score(self):
        """The scores in batch order ((0,) when nothing was scored)."""
        scores = joined(self._scores)
        return np.zeros((0,)) if scores is None else scores

    @property
    def truth(self):
        """The truths in batch order, or None when any batch had none."""
        return joined(self._truths)

    def scored(self, part) -> ScoredPart:
        """The pass as the scored `part` (a held part: its row ids, their kind, its unscored
        rows); a pass whose samples are not the part's is refused."""
        score, row_ids = self.score, np.asarray(part.row_ids)
        if len(score) != len(row_ids):
            raise ValueError(f"{len(score)} sample(s) scored for {len(row_ids)} row id(s) of the "
                             "part; the loader and the part disagree on what a sample is")
        kind = RowIdKind(part.row_id_kind)
        return ScoredPart(score=score, truth=self.truth, row_ids=row_ids, row_id_kind=kind,
                          unscored=tuple(getattr(part, "unscored", ()) or ()),
                          row_score=joined(self._rows),
                          graph_ids=row_ids if kind is RowIdKind.GRAPH_ID else None)



def names_for(spec):
    """The metric names this spec asks for, primary first — derived once, here, beside the
    functions that compute them. The Trainer and the Evaluator both call it, so the valid
    numbers and the test numbers are a list of the same names as well as the same code; two
    derivations of one list is how a valid-set report and a test report drift apart while
    both look right (SOLID-S)."""
    if isinstance(spec, NetSpec):
        names = list(spec.metric_names)
    else:
        node = spec.get("metrics") or {}
        names = [n for n in [node.get("primary")] + list(node.get("secondary") or []) if n]
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError(f"the spec asks for metric(s) {', '.join(map(repr, unknown))} this "
                         f"package's task does not compute (it computes {', '.join(sorted(METRICS))}"
                         "): a caller's error, refused before a fit or a scoring pass, never "
                         "recorded as a missing number (EJ-49)")
    return names


#: the failure modes a metric function itself has. A metric that cannot be computed on these
#: inputs is *recorded* as unavailable (backend-contract §1); anything else is a bug in the
#: caller and must not be filed as a missing number (EJ-77).
METRIC_FAILURES = (ValueError, KeyError, TypeError, ZeroDivisionError, IndexError,
                   NotImplementedError, ArithmeticError)


def compute(names, y_true, y_score, options=None, task=None):
    """{name: value} for the names that computed, plus {name: why} for the ones that did not.

    One decision about what "unavailable" means, in one place: the Trainer's per-epoch record,
    its `evaluate`, and the Evaluator's `Metrics` all report the same thing the same way, and
    a change to the policy is a change to this function (SOLID-S). A `task` the metrics cannot
    read (an unknown kind, a sequence kind without its base) is the caller's error and raises.
    """
    # the scores as float64 once, and each family's preparation once, for the whole pass
    reading = replace(_reading(options, task), memo={})
    y_score, y_true = _as_float(y_score), None if y_true is None else np.asarray(y_true)
    values, unavailable = {}, {}
    for name in names:
        value = _attempt(metric, name, (y_true, y_score, reading), unavailable)
        if name not in unavailable:
            values[name] = value
    return values, unavailable


def _attempt(fn, name, args, unavailable):
    """`fn(name, *args)`; on one of the metric's own failures, the reason under
    `unavailable[name]` and None (the one place that decides it)."""
    try:
        return fn(name, *args)
    except METRIC_FAILURES as exc:
        unavailable[name] = str(exc)[:120]
        return None


def record(values: Mapping, unavailable: Mapping) -> dict:
    """The record of a metric pass: every metric asked for (`values`, then each unavailable
    one as None) and, when any is unavailable, the reasons under `_unavailable`."""
    out = dict(values)
    out.update({name: None for name in unavailable})
    if unavailable:
        out["_unavailable"] = dict(unavailable)
    return out


def metric(name, y_true, y_score, options=None, task=None):
    """The one entry point. Raises on an unknown name rather than returning a plausible
    number for something the caller did not ask for (EJ-49: check parameters, name the
    offender)."""
    fn = METRICS.get(name)
    if fn is None:
        raise ValueError(f"unknown metric {name!r}; this backend implements "
                         f"{sorted(METRICS)}")
    return fn(y_true, y_score, _reading(options, task))


def for_task(task):
    """`metric` bound to one task's effective kind: the `metric` parameter of
    `Metrics.from_predictions(part, scored, names, metric, options)`, which calls it as
    `(name, y_true, y_score, options)`. It is pure: a metric's own failure raises (a pass that
    records the unavailable ones is `compute`, the one place that decides it)."""
    kind = _kind_of(task)

    def bound(name, y_true, y_score, options=None):
        return metric(name, y_true, y_score, _reading(options, kind))

    return bound
