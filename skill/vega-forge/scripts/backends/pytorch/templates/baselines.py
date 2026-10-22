"""
{{PACKAGE}}.{{MODULE_BASELINES}}.baselines — the baselines of eval-contract §4 (map role
`baseline`): models the network is measured against, fitted on the shaped features it saw and
scored through the Evaluator's one scoring path as the backend's `ScoredModel` — never the
{{PORT_MODEL}} port, which nothing serves through them. Emitted by /vega-forge from
`backends/pytorch/templates/baselines.py`; fix the template and re-emit (code-contract §4).

A baseline is a recipe (`Baseline`) with a *form* composed into it (`BaselineForm`: how it
fits, refuses and reads a part) — one class, never a subclass per kind (EJ-18). `fit(design)`
changes nothing and returns a `FittedBaseline` or a `RefusedBaseline` saying why. A head is
written as the network's (backend design §3.4) and read by the `TaskReadout` (R-B7):
probabilities through the reading's own inverse (`scores.head_of_proba`), values repeated over
the quantiles. Here: the recipe, its outcomes and the trio's forms (`TRIO`: the prior; logistic
and ridge, boosted trees over an `EstimatorPair`); `baselines_extra.py` adds the rest and
`build`, and this file imports nothing of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ..domain import (BaselineContext, DesignMatrix, DesignView, NetSpec, SampleSpec, Task,
                      TaskKind, host_array)
from .readout import TaskReadout

__all__ = ["LINEAR", "SEED", "TREE", "TRIO", "Baseline", "BaselineForm", "BaselineSetup",
           "EstimatorPair", "FittedBaseline", "FittedForm", "RefusedBaseline", "design_rows",
           "estimated_form", "full_proba", "unconfigured", "view_for"]

#: the seed every estimator of the family is built with (eval-contract §4: one seed, stated)
SEED = 7
# >>> task: regression forecast
_VALUE = frozenset({TaskKind.REGRESSION, TaskKind.FORECAST})
# <<< task


# ------------------------------------------------------------------ a batch as a design matrix

def design_rows(batch, view: DesignView) -> np.ndarray:
    """One batch in `view`, row for row what `ShapedParts.design_matrix` builds from the part
    the batch came from (float32; the targets, or the graphs, in batch order): the one design
    layout, `DesignMatrix.rows_of`."""
    view, x, target = DesignView(view), DesignMatrix.rows_of(batch), batch.get("target_index")
    # >>> shape: graph
    graph = batch.get("graph_index")
    if (graph is not None) != (view is DesignView.GRAPH_POOLED):
        raise ValueError(f"the {view.value} design view does not read this batch: graph level "
                         "reads graph-pooled, every other sample flat (or neighbour-mean)")
    if graph is not None:
        return DesignMatrix.pooled(x, graph, int(batch["num_graphs"])).astype(np.float32)
    if view is DesignView.NEIGHBOUR_MEAN:
        if target is None or batch.get("edge_index") is None:
            raise ValueError("the neighbour-mean design view reads a node-level graph batch")
        x = np.concatenate([x, DesignMatrix.neighbour_mean(x, batch["edge_index"])], axis=1)
        return x[host_array(target)].astype(np.float32)
    # <<< shape
    if view is not DesignView.FLAT:
        raise ValueError(f"the {view.value} design view reads a graph batch; this package's "
                         "samples are read flat")
    return (x if target is None else x[host_array(target)]).astype(np.float32)


def view_for(spec: NetSpec) -> DesignView:
    """The design view a baseline reads unless its form names another: graph-pooled at graph
    level, flat otherwise (a node-level graph target by its own features — structure-blind)."""
    sample = SampleSpec.from_spec(NetSpec.require(spec))
    return DesignView.GRAPH_POOLED if sample.is_graph_level else DesignView.FLAT


# ------------------------------------------------------------------ what every form reads

@dataclass(frozen=True)
class BaselineSetup:
    """What every form of a baseline reads: the spec, its task and sample, the context (the
    bottleneck, the protocol, the season, the horizon), the reading its head is read by, and
    the head's layout (numeric width, nominal cardinalities, quantile count)."""

    spec: NetSpec
    context: BaselineContext
    readout: TaskReadout
    n_numeric: int
    cards: tuple
    quantiles: int

    @staticmethod
    def of(spec: NetSpec, context: BaselineContext, reading: NetSpec) -> "BaselineSetup":
        contract = spec.feature_contract
        return BaselineSetup(spec, context, TaskReadout.for_spec(reading),
                             len(contract.numeric_columns), contract.cardinalities,
                             len(spec.loss.quantiles))

    @property
    def task(self) -> Task:
        return self.readout.task

    @property
    def kind(self) -> TaskKind:
        """The effective kind (a sequence task's base)."""
        return self.task.effective

    @property
    def sample(self) -> SampleSpec:
        return SampleSpec.from_spec(self.spec)

    def class_head(self, p) -> np.ndarray:
        """The head a model of the class probabilities `p` (n, K) writes: the reading's own
        inverse of its probabilities (a logit, log-probabilities, cumulative logits)."""
        return self.readout.scores.head_of_proba(p)

    def value_head(self, point) -> np.ndarray:
        """Values as the head: (n,) or (n, H), each repeated over the quantiles (h·Q + q)."""
        point = np.asarray(point, dtype=np.float64)
        return np.repeat(point.reshape(len(point), -1), max(self.quantiles, 1), axis=1)

    def no_label(self) -> str:
        return (f"the train part carries no label {self.task.label!r} to fit a "
                f"{self.kind.value} baseline on")


# ------------------------------------------------------------------ forms

@dataclass(frozen=True)
class FittedForm:
    """What a form fitted: the head it answers for a design (`head`), the statistics or the
    estimators it holds (`state`), the batch a reconstruction
    reading compares the head against (`inputs`, else None), whether the head is itself the
    score (`raw`: an isolation forest's per-row score, read by no readout) and what the form
    notes of its fit (`notes`, the lens's split of its hyper-parameters)."""

    head: Callable[[np.ndarray], np.ndarray]
    state: Any
    inputs: Callable[[np.ndarray], Any] | None = None
    raw: bool = False
    notes: Mapping = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class BaselineForm:
    """How one baseline fits and reads, as values: `fit(setup, x, y)` gives a `FittedForm` or
    the reason it cannot fit; `refusal(setup)` refuses before anything is fitted; `view` and
    `sample_view` say how the baseline reads a part; `reading` is the spec its head is read
    under; `notes(fitted)` adds to its record. A form is composed into a `Baseline`."""

    fit: Callable[[BaselineSetup, np.ndarray, Any], "FittedForm | str"]
    refusal: Callable[[BaselineSetup], "str | None"] = lambda setup: None
    view: Callable[[BaselineSetup], DesignView] = lambda setup: view_for(setup.spec)
    sample_view: Callable[[BaselineSetup, Any], Any] = lambda setup, part: part
    reading: Callable[[NetSpec], NetSpec] = lambda spec: spec
    notes: Callable[["FittedForm | None"], dict] = lambda fitted: {}


def unconfigured(estimator) -> tuple:
    """The `configure` of an estimating form that sets nothing: (the estimator, no note)."""
    return estimator, None


# ------------------------------------------------------------------ the recipe and its outcomes

class Baseline:
    """One baseline, unfitted: its name, the spec, the form composed into it. `fit(design)`
    returns a `FittedBaseline` or a `RefusedBaseline` and changes nothing here."""

    def __init__(self, name: str, spec: NetSpec, form: BaselineForm,
                 context: BaselineContext | None = None):
        spec = NetSpec.require(spec)
        self._name, self._form = str(name), form
        self._setup = BaselineSetup.of(spec, context if context is not None
                                       else BaselineContext(), form.reading(spec))

    @property
    def name(self) -> str:
        return self._name

    @property
    def view(self) -> DesignView:
        return self._form.view(self._setup)

    def sample_view(self, dataset):
        """The part this baseline reads: the part itself, or the form's view of it."""
        return self._form.sample_view(self._setup, dataset)

    def design_of(self, data, dataset) -> DesignMatrix:
        """`dataset` as this baseline's design matrix, through the DataSource port."""
        return data.design_matrix(self._setup.spec, self.sample_view(dataset), self.view)

    def fit(self, design: DesignMatrix) -> "FittedBaseline | RefusedBaseline":
        """Fit on a design matrix in this baseline's view; a refusal is returned, never
        raised."""
        x, y = np.asarray(design.x, dtype=np.float64), host_array(design.y)
        outcome = self._form.refusal(self._setup) or (
            None if len(x) else "the train part holds no sample")
        if outcome is None:
            try:
                outcome = self._form.fit(self._setup, x, y)
            except ImportError as exc:
                outcome = f"scikit-learn is not installed ({exc})"
        if isinstance(outcome, str):
            return RefusedBaseline(self._name, outcome, self.view,
                                   MappingProxyType(self._form.notes(None)))
        return FittedBaseline(self._name, self._setup, self._form, outcome)

    def record(self) -> dict:
        """The row of the baseline not yet fitted."""
        return {"name": self._name, "fitted": False, "why": None, "view": self.view.value,
                **self._form.notes(None)}


@dataclass(frozen=True)
class RefusedBaseline:
    """A baseline that could not be fitted, and why: it keeps its row in the evaluation (a
    baseline that quietly disappeared reads as one the network beat)."""

    name: str
    why: str
    view: DesignView
    notes: Mapping = field(default_factory=lambda: MappingProxyType({}))
    fitted = False

    def record(self) -> dict:
        return {"name": self.name, "fitted": False, "why": self.why, "view": self.view.value,
                **dict(self.notes)}


class FittedBaseline:
    """A fitted baseline, as the Evaluator scores it (`ScoredModel`: `score`, `head`,
    `row_score`, `sample_view`): its head read by the spec's `TaskReadout`, never written."""

    fitted = True
    why = None

    def __init__(self, name: str, setup: BaselineSetup, form: BaselineForm, fitted: FittedForm):
        self._name, self._setup, self._form, self._fitted = name, setup, form, fitted

    @property
    def name(self) -> str:
        return self._name

    @property
    def view(self) -> DesignView:
        return self._form.view(self._setup)

    def sample_view(self, dataset):
        return self._form.sample_view(self._setup, dataset)

    def score(self, batch):
        """The score per the score contract, read off this baseline's head."""
        return self._read(design_rows(batch, self.view), lambda r: r.score)

    def row_score(self, batch):
        """The per-row score of a reconstruction or anomaly reading, else None."""
        return self._read(design_rows(batch, self.view), lambda r: r.row_score)

    def head(self, batch):
        """The raw head, as the network's head would hold it (the stored `score_k` columns)."""
        return self._fitted.head(np.asarray(design_rows(batch, self.view), dtype=np.float64))

    def record(self) -> dict:
        return {"name": self._name, "fitted": True, "why": None, "view": self.view.value,
                **self._form.notes(self._fitted)}

    def _read(self, x, reading):
        """The head for the design `x`, read by the task's `TaskReadout` through `reading` (a
        function that picks `score` or `row_score`, never a name); a raw head is its own score."""
        x = np.asarray(x, dtype=np.float64)
        head = self._fitted.head(x)
        if self._fitted.raw:
            return head
        inputs = self._fitted.inputs(x) if self._fitted.inputs is not None else None
        out = reading(self._setup.readout)(
            torch.from_numpy(np.ascontiguousarray(head, dtype=np.float64)), inputs, None)
        return None if out is None else out.detach().cpu().numpy()


# ------------------------------------------------------------------ the trio's forms

def full_proba(proba, seen, k: int) -> np.ndarray:
    """Probabilities over the classes an estimator saw, as (n, k) over every class."""
    out = np.zeros((len(proba), int(k)))
    out[:, np.asarray(seen).astype(np.int64)] = proba
    return out


def _prior_fit(setup: BaselineSetup, x, y):
    if y is None:
        return setup.no_label()
    # >>> task: regression forecast
    if setup.kind in _VALUE:
        mean = float(np.mean(y))
        steps = setup.sample.window.horizon if setup.kind is TaskKind.FORECAST else 1
        return FittedForm(lambda d: setup.value_head(np.full((len(d), steps), mean)),
                          {"mean": mean})
    # <<< task
    # >>> task: multilabel
    if setup.kind is TaskKind.MULTILABEL:
        prior = np.asarray(y, dtype=np.float64).mean(axis=0)
        return FittedForm(lambda d: setup.class_head(np.tile(prior, (len(d), 1))),
                          {"prior": prior})
    # <<< task
    counts = np.bincount(y.astype(np.int64).reshape(-1), minlength=len(setup.task.labels))
    prior = counts / counts.sum()
    return FittedForm(lambda d: setup.class_head(np.tile(prior, (len(d), 1))), {"prior": prior})


@dataclass(frozen=True)
class EstimatorPair:
    """How an estimating baseline builds its estimators: its classifier, and its regressor
    for one output or several (`outputs`)."""

    name: str
    classifier: Callable[[], Any]
    regressor: Callable[[bool], Any]


def _logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000)


def _ridge(outputs: bool):
    from sklearn.linear_model import RidgeCV
    return RidgeCV(alphas=np.logspace(-3, 3, 13))


def _boosted_classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(early_stopping=True, random_state=SEED)


def _boosted_regressor(outputs: bool):
    from sklearn.ensemble import HistGradientBoostingRegressor
    one = HistGradientBoostingRegressor(early_stopping=True, random_state=SEED)
    if not outputs:
        return one
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(one)


#: logistic regression and ridge, on the features the network saw, in this view
LINEAR = EstimatorPair("linear", _logistic, _ridge)
#: gradient-boosted trees — the honest tabular contender
TREE = EstimatorPair("tree", _boosted_classifier, _boosted_regressor)


# >>> task: multilabel
def _label_proba(model, x) -> np.ndarray:
    """One label's P(1): its prevalence (a float), or the fitted classifier's."""
    if isinstance(model, float):
        return np.full(len(x), model)
    return model.predict_proba(x)[:, list(model.classes_).index(1)]
# <<< task


def estimated_form(pair: EstimatorPair, configure=unconfigured) -> BaselineForm:
    """`linear` / `tree`: one estimator per kind from the pair, each passed through
    `configure` (`estimator -> (estimator, note)`: the lens's recorded hyper-parameters) — a
    concern composed in, never a subclass per concern (EJ-18)."""
    return BaselineForm(fit=lambda setup, x, y: _estimated_fit(pair, configure, setup, x, y))


def _estimated_fit(pair: EstimatorPair, configure, setup: BaselineSetup, x, y):
    if y is None:
        return setup.no_label()
    notes = {}

    def made(estimator):
        estimator, note = configure(estimator)
        if note is not None:
            notes["split"] = note
        return estimator

    # >>> task: regression forecast
    if setup.kind in _VALUE:
        multi = y.ndim > 1 and y.shape[1] > 1
        model = made(pair.regressor(multi)).fit(x, y if multi else y.reshape(-1))
        return FittedForm(lambda d: setup.value_head(model.predict(d)), model,
                          notes=MappingProxyType(notes))
    # <<< task
    # >>> task: multilabel
    if setup.kind is TaskKind.MULTILABEL:
        models = [float(np.mean(col)) if len(np.unique(col)) < 2
                  else made(pair.classifier()).fit(x, col.astype(np.int64))
                  for col in (y[:, j] for j in range(y.shape[1]))]
        return FittedForm(lambda d: setup.class_head(np.stack(
            [_label_proba(m, d) for m in models], axis=1)), models, notes=MappingProxyType(notes))
    # <<< task
    seen = np.unique(y)
    if len(seen) < 2:
        return (f"the train part holds one class only "
                f"({setup.task.labels.at(int(seen[0])).name!r}); a classifier needs two")
    model = made(pair.classifier()).fit(x, y.astype(np.int64).reshape(-1))
    k = len(setup.task.labels)
    return FittedForm(lambda d: setup.class_head(full_proba(model.predict_proba(d),
                                                            model.classes_, k)),
                      model, notes=MappingProxyType(notes))


#: the supervised trio by name, each maker taking the same `configure` (`unconfigured`: none):
#: `trivial` is the floor every metric is read against (the prior, the prevalence, the train
#: mean); baselines_extra.build adds the rest
TRIO = MappingProxyType({"trivial": lambda configure: BaselineForm(fit=_prior_fit),
                         "linear": lambda configure: estimated_form(LINEAR, configure),
                         "tree": lambda configure: estimated_form(TREE, configure)})
