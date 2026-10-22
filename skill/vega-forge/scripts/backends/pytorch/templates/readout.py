"""
{{PACKAGE}}.{{MODULE_MODEL}}.readout — what the head's raw outputs mean, for this spec's task.

Emitted by /vega-forge from `backends/pytorch/templates/readout.py` (a `kind: support` file of
the Model's module, used by the Model, the Trainer, the Evaluator and the Predictor). A design
defect here is fixed in that template and the package re-emitted; never in this copy.

`TaskReadout.for_spec(spec)` is the **one source of output semantics** (BUILD-PLAN R-B7; D1
§3.4, §7.5): the network's `score` and `predict_proba`, the trainer's epoch accumulator, the
evaluator, the baselines' conversions and the serving Predictor all read a head through it, so
no two of them can disagree on what a logit, a cumulative logit or a reconstruction means.

| effective kind | head | `score` | `proba` | `reduce` | link |
|---|---|---|---|---|---|
# >>> task: binary
| binary | (B, 1) | (B,) logit, always | σ(z) | z > 0, int32 | sigmoid |
# <<< task
# >>> task: multiclass
| multiclass | (B, K) | (B, K) logits | softmax | argmax, int32 | softmax |
# <<< task
# >>> task: multilabel
| multilabel | (B, K) | (B, K) logits | σ per label | z > 0, int8 (B, K) | sigmoid_each |
# <<< task
# >>> task: regression
| regression | (B, 1) / (B, Q) | (B,) value (q=0.5 column) | = score | float32 | identity |
# <<< task
# >>> task: ordinal
| ordinal | (B, K−1) | (B, K−1) cum. logits | (B, K) class probs | Σ_j [z_j ≥ 0] | cumulative |
# <<< task
# >>> task: forecast
| forecast | (B, H) / (B, H·Q) | (B, H) (median column) | = score | (B, H) float32 | horizon |
# <<< task
# >>> task: unsupervised
| unsupervised | (B, R) | (B, n_numeric) values | = score | row numeric MSE | reconstruction |
# <<< task
# >>> task: anomaly
| anomaly, recon / vae | (B, R) | (B,) MSE + mean nominal CE | = score | = score | reconstruction |
# <<< task
# >>> task: anomaly
| anomaly, svdd | (B, z) | (B,) ‖f(x) − c‖², c the buffer | = score | = score | distance |
# <<< task

A `sequence-*` task reads through its `task.base_kind` (R-A9). Binary and multilabel scores
are logits, σ applied unconditionally (R-C3). `reduce` is the train accumulator's prediction:
`z > 0` for binary and multilabel (v2.2's: the golden history stays bit-identical), R-C6's
`Σ_j [σ(z_j) ≥ 0.5]` for ordinal, whose probabilities are clipped at 0 and renormalised
(`clipped` names the rows, for the serving warning). The head is `forward`'s output, the loss
input: `R = recon_dim` = `[n_numeric values | card_1 logits | …]` (contract order), Q quantiles
horizon-major (step h at h·Q + q). `row_score` is R-C2's per-row score of the reconstruction
and anomaly readings. Every method is pure (R-B4); `reduce` is fixed-shape, on the outputs'
device, with no host sync (CUDA-graph safe). `scores` is the reading's numpy half (metrics).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import torch
# >>> task: unsupervised anomaly
from torch.nn import functional as F
# <<< task

from ..domain import NetSpec, Objective, SampleSpec, Task, TaskKind
from .metrics import ScoreReading, reading_for

@dataclass(frozen=True)
class _Layout:
    """Beyond the task: class count, quantiles (loss `quantile` only), horizon, recon layout."""

    n_classes: int
    quantiles: tuple
    horizon: int
    n_numeric: int
    cardinalities: tuple

    @staticmethod
    def of(spec: NetSpec, task: Task) -> "_Layout":
        window = SampleSpec.from_spec(spec).window
        contract = spec.feature_contract
        return _Layout(len(task.labels), spec.loss.quantiles,
                       int(window.horizon) if window else 0, len(contract.numeric_columns),
                       contract.cardinalities)

    @property
    def median(self) -> int:
        """The column of the quantile nearest 0.5 (the point estimate)."""
        return min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))

    @property
    def recon_dim(self) -> int:
        return self.n_numeric + sum(self.cardinalities)


# >>> task: unsupervised anomaly
def _numeric_error(out, batch, layout):
    """(B,) mean squared error of the numeric reconstruction, 0 without numeric columns."""
    n = layout.n_numeric
    if n == 0:
        return out.new_zeros(out.shape[:-1])
    x_num = batch["x_num"][..., :n].to(out.dtype)
    return (out[..., :n] - x_num).pow(2).mean(dim=-1)


def _row_error(out, batch, layout):
    """(B,) numeric MSE + the mean cross-entropy of the nominal logits over their columns."""
    err = _numeric_error(out, batch, layout)
    if not layout.cardinalities:
        return err
    x_cat = batch.get("x_cat")
    if x_cat is None:
        raise ValueError(f"the reconstruction reads {len(layout.cardinalities)} nominal "
                         "column(s) from x_cat, and the batch carries none")
    terms, at = [], layout.n_numeric
    for i, card in enumerate(layout.cardinalities):
        logits = out[..., at: at + card]
        terms.append(F.cross_entropy(logits, x_cat[..., i].long(), reduction="none"))
        at += card
    return err + torch.stack(terms, dim=-1).mean(dim=-1)
# <<< task


class _Reading:
    """One reading of a head: `score` is the head itself, `proba` and `reduce` the score."""

    LINK = "identity"
    ROW_SCORED = False            # the score is a per-row score (the anomaly readings)
    PER_ROW = False               # `row_score` gives one (the anomaly and reconstruction ones)

    def __init__(self, layout: _Layout):
        self.layout = layout

    def head_width(self):
        return None

    def score(self, out, batch, model):
        return out

    def proba(self, out, batch, model):
        return self.score(out, batch, model)

    def reduce(self, out, batch, model):
        return self.score(out, batch, model)

    def row_score(self, out, batch, model):
        return self.score(out, batch, model) if self.ROW_SCORED else None

    def truth(self, batch):
        return batch.get("y")

    def clipped(self, out):
        return None

    def reduced_shape(self) -> tuple:
        return ()


# >>> task: binary
class _Binary(_Reading):
    LINK = "sigmoid"

    def head_width(self):
        return 1

    def score(self, out, batch, model):
        return out[..., 0]

    def proba(self, out, batch, model):
        return torch.sigmoid(out[..., 0])

    def reduce(self, out, batch, model):
        return (out[..., 0] > 0).to(torch.int32)
# <<< task


# >>> task: multiclass
class _Multiclass(_Reading):
    LINK = "softmax"

    def head_width(self):
        return self.layout.n_classes

    def proba(self, out, batch, model):
        return torch.softmax(out, dim=-1)

    def reduce(self, out, batch, model):
        return out.argmax(dim=-1).to(torch.int32)
# <<< task


# >>> task: multilabel
class _Multilabel(_Reading):
    LINK = "sigmoid_each"

    def head_width(self):
        return self.layout.n_classes

    def proba(self, out, batch, model):
        return torch.sigmoid(out)

    def reduce(self, out, batch, model):
        return (out > 0).to(torch.int8)

    def reduced_shape(self):
        return (self.layout.n_classes,)
# <<< task


# >>> task: regression
class _Regression(_Reading):
    LINK = "identity"

    def head_width(self):
        return len(self.layout.quantiles) or 1

    def score(self, out, batch, model):
        return out[..., self.layout.median if self.layout.quantiles else 0]

    def reduce(self, out, batch, model):
        return self.score(out, batch, model).to(torch.float32)
# <<< task


# >>> task: ordinal
class _Ordinal(_Reading):
    """K−1 cumulative logits, `σ(z_j) = P(y > j)`: `p_0 = 1 − σ(z_0)`, `p_j = σ(z_{j−1}) −
    σ(z_j)`, `p_{K−1} = σ(z_{K−2})`, clipped at 0 and renormalised (bundle-contract §4)."""

    LINK = "cumulative"

    def head_width(self):
        return self.layout.n_classes - 1

    def _raw(self, out):
        above = torch.sigmoid(out)
        edge = above.new_ones(above.shape[:-1] + (1,))
        return torch.cat([edge, above], dim=-1) - torch.cat([above, edge * 0], dim=-1)

    def proba(self, out, batch, model):
        p = self._raw(out).clamp(min=0.0)
        return p / p.sum(dim=-1, keepdim=True)

    def clipped(self, out):
        return (self._raw(out) < 0).any(dim=-1)

    def reduce(self, out, batch, model):
        return (out >= 0).sum(dim=-1).to(torch.int32)
# <<< task


# >>> task: forecast
class _Forecast(_Reading):
    """H steps ahead; under a quantile loss (B, H·Q) horizon-major, each step's median read."""

    LINK = "horizon"

    def head_width(self):
        return self.layout.horizon * (len(self.layout.quantiles) or 1)

    def score(self, out, batch, model):
        if not self.layout.quantiles:
            return out
        steps = out.reshape(out.shape[0], self.layout.horizon, len(self.layout.quantiles))
        return steps[..., self.layout.median]

    def reduce(self, out, batch, model):
        return self.score(out, batch, model).to(torch.float32)

    def reduced_shape(self):
        return (self.layout.horizon,)
# <<< task


# >>> task: unsupervised
class _Reconstruction(_Reading):
    """Unsupervised: score the numeric reconstruction, reduce its row MSE, row_score R-C2's."""

    LINK = "reconstruction"
    PER_ROW = True

    def head_width(self):
        return self.layout.recon_dim

    def score(self, out, batch, model):
        return out[..., : self.layout.n_numeric]

    def reduce(self, out, batch, model):
        return _numeric_error(out, batch, self.layout).to(torch.float32)

    def row_score(self, out, batch, model):
        return _row_error(out, batch, self.layout)

    def truth(self, batch):
        return batch.get("x_num")
# <<< task


# >>> task: anomaly
class _AnomalyRecon(_Reading):
    LINK = "reconstruction"
    ROW_SCORED = PER_ROW = True

    def head_width(self):
        return self.layout.recon_dim

    def score(self, out, batch, model):
        return _row_error(out, batch, self.layout)


class _AnomalySvdd(_Reading):
    """‖f(x) − c‖², c the model's `svdd_center` buffer (R-B6), carried by the checkpoint."""

    LINK = "distance"
    ROW_SCORED = PER_ROW = True

    def score(self, out, batch, model):
        centre = model.objective_center() if model is not None else None
        if centre is None:
            raise ValueError("the svdd reading measures the distance to the model's centre "
                             "(model.objective_center()), and none was given")
        return (out - centre.to(out.dtype)).pow(2).sum(dim=-1)
# <<< task


# (the package carries its spec's task kind and no other: `domain.Task` names every kind, and
# a spec of a kind this table does not hold is refused below, by kind and objective)
_BY_KIND = MappingProxyType({
    # >>> task: binary
    (TaskKind.BINARY, Objective.PREDICT): _Binary,
    # <<< task
    # >>> task: multiclass
    (TaskKind.MULTICLASS, Objective.PREDICT): _Multiclass,
    # <<< task
    # >>> task: multilabel
    (TaskKind.MULTILABEL, Objective.PREDICT): _Multilabel,
    # <<< task
    # >>> task: regression
    (TaskKind.REGRESSION, Objective.PREDICT): _Regression,
    # <<< task
    # >>> task: ordinal
    (TaskKind.ORDINAL, Objective.PREDICT): _Ordinal,
    # <<< task
    # >>> task: forecast
    (TaskKind.FORECAST, Objective.PREDICT): _Forecast,
    # <<< task
    # >>> task: unsupervised
    (TaskKind.UNSUPERVISED, Objective.RECONSTRUCTION): _Reconstruction,
    (TaskKind.UNSUPERVISED, Objective.VAE): _Reconstruction,
    # <<< task
    # >>> task: anomaly
    (TaskKind.ANOMALY, Objective.RECONSTRUCTION): _AnomalyRecon,
    (TaskKind.ANOMALY, Objective.VAE): _AnomalyRecon,
    (TaskKind.ANOMALY, Objective.SVDD): _AnomalySvdd,
    # <<< task
})


class TaskReadout:
    """The reading of one spec's head: its task, and the strategy `_BY_KIND` selects."""

    def __init__(self, task: Task, reading: _Reading):
        self._task, self._reading = task, reading

    @staticmethod
    def for_spec(spec: NetSpec) -> "TaskReadout":
        """By the effective kind and the objective (`loss.name`), with the head's layout."""
        spec = NetSpec.require(spec)
        task = Task.from_spec(spec)
        strategy = _BY_KIND.get((task.effective, task.objective))
        if strategy is None:
            raise ValueError(f"no reading for a {task.effective.value} task trained with the "
                             f"{task.objective.value} objective (loss "
                             f"{spec.loss.name!r}); the catalog's "
                             "losses_allowed rows name the pairs that exist")
        return TaskReadout(task, strategy(_Layout.of(spec, task)))

    @property
    def task(self) -> Task:
        return self._task

    @property
    def kind(self) -> TaskKind:
        """The effective kind (a sequence task's base)."""
        return self._task.effective

    @property
    def objective(self) -> Objective:
        return self._task.objective

    @property
    def link(self) -> str:
        """The link of the reading, in the catalog's `inspection.links` vocabulary (R-C1)."""
        return self._reading.LINK

    def score(self, out, batch=None, model=None):
        """What the metrics read (D1 §3.4 "score")."""
        return self._reading.score(out, batch, model)

    def proba(self, out, batch=None, model=None):
        """What `predict_proba` returns: the distribution, or the score where there is none."""
        return self._reading.proba(out, batch, model)

    def reduce(self, out, batch=None, model=None):
        """The fixed-shape per-row prediction the train accumulator keeps."""
        return self._reading.reduce(out, batch, model)

    def row_score(self, out, batch=None, model=None):
        """(B,) the per-row score of the reconstruction and anomaly readings, else None."""
        return self._reading.row_score(out, batch, model)

    def clipped(self, out):
        """(B,) bool: the ordinal rows whose probabilities needed clipping, else None."""
        return self._reading.clipped(out)

    def reduced_shape(self) -> tuple:
        """The per-row shape of `reduce`: () | (K,) | (H,)."""
        return self._reading.reduced_shape()

    def label_only(self) -> frozenset:
        """The metrics computable from `reduce` alone (the rest need the score)."""
        return self.scores.label_only

    @property
    def scores(self) -> ScoreReading:
        """This reading's numpy half, as the metrics read host arrays (`metrics.reading_for`)."""
        return reading_for(self._task)

    @property
    def per_row(self) -> bool:
        """Whether the reading gives a per-row score (`row_score` is not None)."""
        return self._reading.PER_ROW

    def truth(self, batch):
        """What a score is measured against: the label, the input for a reconstruction."""
        return self._reading.truth(batch)

    def check_head(self, width: int, where: str = "the head") -> None:
        """Refuse a head whose width is not the one this reading reads."""
        wanted = self._reading.head_width()
        if wanted is not None and int(width) != int(wanted):
            raise ValueError(f"{where} outputs {width} value(s) per sample; a "
                             f"{self.kind.value} reading under the {self.objective.value} "
                             f"objective reads {wanted} (the catalog's task row forces it)")


__all__ = ["TaskReadout"]
