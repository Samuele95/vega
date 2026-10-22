"""
{{PACKAGE}}.{{MODULE_BASELINES}}.baselines_extra — the baseline forms a task or a shape adds to
the supervised trio (map role `baseline`, beside `baselines.py`, whose recipe and forms they
compose), and `build`, which names every baseline of the family (/vega-eval's one entry). The
dependency points one way, from here to `baselines.py`.

Emitted by /vega-forge from `backends/pytorch/templates/baselines_extra.py`. A defect is fixed in
the template and re-emitted, never in this copy (code-contract §4).

# >>> task: unsupervised anomaly
  * **The reconstruction forms of the trio** (`RECONSTRUCTING`: `trivial`, `linear`, `tree` of an
    unsupervised or anomaly task): the mean row, PCA at the bottleneck z (`pca-floor`), each the
    network's reconstruction head `[numeric | nominal log-priors]`, its per-row score its error
    (a baseline has no hypersphere centre); the tree is boosted trees on a two-valued label, else
    an isolation forest (higher `-score_samples`, more anomalous) — none for unsupervised.
# <<< task
  * `lens` — the algorithm /data-lens's importance module recorded, re-fitted on the shaped
    features: only the protocol carries over (the analysis fitted pre-shaping columns) — its
    family picks the `tree` or the `linear` form `build` gives this task, and its recorded
    hyper-parameters are set on every estimator that form builds (`record()` says which carried).
# >>> shape: window
  * `naive` — persistence over the horizon: a forecast repeats the anchor's target, a sequence
    regression the row before it. The history is each step's last numeric channel (the
    autoregressive one, else the labels `sample_view` appends); no step at or after the target
    is read.
  * `seasonal-naive` — the same, one season back: step h repeats the target `s·⌈h/s⌉` rows
    before it (`sample.season`, or the context's), when the window reaches that far.
# <<< shape
# >>> shape: graph
  * `graph-linear` — linear on `[x ‖ mean of the in-neighbours' x]`, one fixed propagation
    (SGC-style): at node level, the contender learned message passing must beat.
# <<< shape

Every maker takes the same `configure` (an estimating form passes its estimators through it;
the others ignore it), so `build` constructs every baseline one way. The package holds its own
shape's and task's forms; one built where it has no definition is refused, and says why.
"""
from __future__ import annotations

# >>> shape: window
import math
# <<< shape
from dataclasses import asdict, dataclass, replace
# >>> task: unsupervised anomaly
from types import MappingProxyType
# <<< task

# >>> task: unsupervised anomaly
import numpy as np
import torch
# <<< task
# >>> shape: window
# >>> task: binary multiclass multilabel regression ordinal forecast
import numpy as np
# <<< task
# <<< shape

from ..domain import BaselineContext, NetSpec
# >>> task: unsupervised anomaly
from ..domain import Task, TaskKind
# <<< task
# >>> shape: window
# >>> task: binary multiclass multilabel regression ordinal forecast
from ..domain import TaskKind
# <<< task
# <<< shape
# >>> shape: graph
from ..domain import DesignView
from .baselines import LINEAR, estimated_form, view_for
# <<< shape
from .baselines import TRIO, Baseline, BaselineForm, FittedForm, unconfigured
# >>> task: unsupervised anomaly
from .baselines import SEED, TREE, full_proba
from .metrics import clip, logit
# <<< task


# >>> task: unsupervised anomaly
#: the losses whose reading a reconstruction baseline shares; svdd's distance is not one
_RECON_LOSSES = frozenset({"reconstruction", "vae"})
# <<< task
# >>> shape: window
#: the task kinds persistence has a meaning for: the next values of a series, or its label
_PERSISTENT = frozenset({TaskKind.FORECAST, TaskKind.SEQUENCE_REGRESSION})
# <<< shape


# >>> task: unsupervised anomaly
# ------------------------------------------------------------------ reconstruction

def _reconstructing(spec: NetSpec) -> NetSpec:
    """The spec the reconstruction forms' heads are read under: the spec itself, except an
    anomaly trained under svdd, whose baselines reconstruct."""
    if Task.from_spec(spec).effective is not TaskKind.ANOMALY or spec.loss.name in _RECON_LOSSES:
        return spec
    d = spec.to_dict()
    d["loss"] = {"name": "reconstruction", "params": {}}
    return NetSpec.from_dict(d)


def _numeric(setup, x) -> np.ndarray:
    return x[:, :x.shape[1] - len(setup.cards)]


def _inputs(setup):
    """The row design as the batch the reconstruction reading compares against."""
    def inputs(x):
        batch = {"x_num": torch.from_numpy(np.ascontiguousarray(_numeric(setup, x)))}
        if setup.cards:
            batch["x_cat"] = torch.from_numpy(x[:, x.shape[1] - len(setup.cards):]
                                              .astype(np.int64))
        return batch
    return inputs


def _priors(setup, x) -> list:
    """The nominal columns' train frequencies, as the head's log-priors."""
    codes = x[:, x.shape[1] - len(setup.cards):].astype(np.int64)
    return [np.log(clip(np.bincount(np.clip(codes[:, i], 0, c - 1), minlength=c) / len(x)))
            for i, c in enumerate(setup.cards)]


def _recon_head(priors, numeric) -> np.ndarray:
    n = len(numeric)
    return np.concatenate([np.asarray(numeric, dtype=np.float64)]
                          + [np.tile(p, (n, 1)) for p in priors], axis=1)


def _mean_row_fit(setup, x, y):
    priors, mean = _priors(setup, x), _numeric(setup, x).mean(axis=0)
    return FittedForm(lambda d: _recon_head(priors, np.tile(mean, (len(d), 1))),
                      {"priors": priors, "mean": mean}, inputs=_inputs(setup))


def _pca_fit(setup, x, y):
    from sklearn.decomposition import PCA
    numeric = _numeric(setup, x)
    d = numeric.shape[1]
    if d == 0:
        return "PCA reconstructs the numeric columns, and this spec has none"
    priors = _priors(setup, x)
    k = int(setup.context.z or max(1, min(d - 1, d // 2)))
    pca = PCA(n_components=min(k, d, len(numeric))).fit(numeric)
    return FittedForm(lambda z: _recon_head(priors, pca.inverse_transform(pca.transform(
        _numeric(setup, z)))), {"priors": priors, "pca": pca}, inputs=_inputs(setup))


def _forest_refusal(setup):
    if setup.kind is TaskKind.UNSUPERVISED:
        return "a tree baseline is not defined for a reconstruction objective"
    return None


def _forest_fit(configure, setup, x, y):
    """Boosted trees (through `configure`) when train holds both label values, else an
    isolation forest; the head is the per-row score, the score and the row score alike."""
    if y is not None and len(np.unique(y)) > 1:
        classifier, note = configure(TREE.classifier())
        model = classifier.fit(x, np.asarray(y).astype(np.int64).reshape(-1))
        return FittedForm(lambda d: logit(full_proba(model.predict_proba(d), model.classes_,
                                                     2)[:, 1]), model, raw=True,
                          notes=MappingProxyType({"split": note} if note is not None else {}))
    from sklearn.ensemble import IsolationForest
    forest = IsolationForest(random_state=SEED).fit(x)
    return FittedForm(lambda d: -forest.score_samples(d), forest, raw=True)


def _mean_row(configure) -> BaselineForm:
    """`trivial` for a reconstruction or an anomaly: the mean row, and its error."""
    return BaselineForm(fit=_mean_row_fit, reading=_reconstructing)


def _pca(configure) -> BaselineForm:
    """`linear` for a reconstruction or an anomaly: PCA at the bottleneck width z."""
    return BaselineForm(fit=_pca_fit, reading=_reconstructing)


def _forest(configure) -> BaselineForm:
    """`tree` for an anomaly: boosted trees, or an isolation forest without two labels."""
    return BaselineForm(fit=lambda setup, x, y: _forest_fit(configure, setup, x, y),
                        refusal=_forest_refusal)
# <<< task


# ------------------------------------------------------------------ lens

@dataclass(frozen=True)
class _HyperSplit:
    """The recorded hyper-parameters as one estimator took them: applied, not carried (why)."""

    applied: dict
    not_carried: dict


def _split(params: dict, est) -> tuple:
    """(`est` with the `params` it takes set — through a multi-output wrapper's estimator —,
    the split); the seed is never carried, and an invalid value fails at fit, saying why."""
    names, applied, not_carried = set(est.get_params(deep=True)), {}, {}
    for key, value in (params or {}).items():
        nested = f"estimator__{key}"
        if key == "random_state":
            not_carried[key] = "the baselines fit under the spec's seed"
        elif key in names or nested in names:
            applied[key if key in names else nested] = value
        else:
            not_carried[key] = f"{type(est).__name__} takes no {key!r}"
    split = _HyperSplit(applied={k.rsplit("__", 1)[-1]: v for k, v in applied.items()},
                        not_carried=not_carried)
    return est.set_params(**applied), split


def _lens_form(spec: NetSpec, context: BaselineContext) -> BaselineForm:
    """The importance model /data-lens recorded: the `tree` form for a tree, forest or boosting
    family, else the `linear` one (the task's form, as `build` gives it), its estimators
    configured with the recorded `hyper_parameters`; refused without a recorded protocol."""
    protocol = context.protocol_dict()
    family = str(protocol.get("family", ""))
    tree = any(word in family.lower() for word in ("tree", "boost", "forest"))
    delegate = "tree" if tree else "linear"
    wanted = dict(protocol.get("hyper_parameters") or {})
    form = _makers(spec)[delegate](lambda est: _split(wanted, est))

    def refusal(setup):
        if not protocol:
            return ("the analysis layer records no importance protocol to re-fit; "
                    "/data-lens's importance module was not run")
        return form.refusal(setup)

    def notes(fitted: FittedForm | None) -> dict:
        split = (fitted.notes.get("split") if fitted is not None else None)
        if split is None:
            why = ("not fitted yet" if fitted is None else
                   f"the {delegate} form of this task builds no estimator that takes them")
            split = _HyperSplit(applied={}, not_carried={k: why for k in wanted})
        return {"delegate": delegate, "family": family, "hyper_parameters": asdict(split)}

    return replace(form, refusal=refusal, notes=notes)


# >>> shape: window
# ------------------------------------------------------------------ windows

def _with_channel(x_num, y):
    """`x_num` (M, F) with the target appended as its last column (M, F + 1)."""
    import torch           # noqa: PLC0415  (a held window's tensors; the rows here are arrays)
    if torch.is_tensor(x_num):
        column = torch.as_tensor(y).reshape(-1, 1).to(device=x_num.device, dtype=x_num.dtype)
        return torch.cat([x_num, column], dim=1)
    x_num = np.asarray(x_num)
    return np.concatenate([x_num, np.asarray(y, dtype=x_num.dtype).reshape(-1, 1)], axis=1)


def _target_history(setup, dataset):
    """A window part with the target's history as its last numeric channel: the part itself
    when the spec's autoregressive channel already is it, else the part's labels appended (a
    window reads them at its own steps, all at or before its anchor)."""
    window = setup.sample.window
    if window is None or window.autoregressive or getattr(dataset, "anchors", None) is None:
        return dataset
    if dataset.y is None:
        raise ValueError("a persistence baseline reads the target's history, and this part "
                         "carries no label")
    return replace(dataset, x_num=_with_channel(dataset.x_num, dataset.y))


def _persistence_refusal(name: str):
    def refusal(setup):
        if setup.sample.window is None:
            return (f"{name} repeats a window's past target, and this spec's sample is "
                    f"{setup.sample.shape.value}")
        if setup.task.kind not in _PERSISTENT:
            return (f"{name} applies to a forecast or a sequence regression; this task is "
                    f"{setup.task.kind.value}")
        return None
    return refusal


def _season(setup):
    declared = setup.context.season
    if declared is None and setup.sample.window is not None:
        declared = setup.sample.window.season
    return None if declared is None else int(declared)


def _seasonal_refusal(setup):
    why = _persistence_refusal("seasonal-naive")(setup)
    if why is not None:
        return why
    season, length = _season(setup), setup.sample.window.length
    if season is None or season < 1:
        return "sample.season is not set: there is no season to repeat"
    reach = length if setup.task.kind is TaskKind.FORECAST else length - 1
    if season > reach:
        return (f"season {season} reaches further back than the window's {length} rows hold "
                f"(at most {reach} for this task)")
    return None


def _positions(setup, season=None) -> tuple:
    """The window step each output repeats: the anchor's for every horizon step (forecast), or
    the step before the anchor (a sequence regression: the anchor's own label is the target);
    one season back of either with `season`."""
    window = setup.sample.window
    if setup.task.kind is TaskKind.FORECAST:
        if season is None:
            return (window.length - 1,) * int(window.horizon)
        return tuple(window.length - 1 + h - season * math.ceil(h / season)
                     for h in range(1, int(window.horizon) + 1))
    return (window.length - 2,) if season is None else (window.length - 1 - season,)


def _step_width(setup, name: str, x) -> int:
    """F′, the numeric width of one window step in a flat design, checked to carry the target
    channel (the features, then the target)."""
    length = setup.sample.window.length
    numeric = x.shape[1] - length * len(setup.cards)
    if numeric % length or numeric // length != setup.n_numeric + 1:
        raise ValueError(
            f"the {name} baseline reads the target's history as the last numeric channel of "
            f"each of the window's {length} steps ({setup.n_numeric} feature(s) + the target); "
            f"this design holds {numeric} numeric column(s) — build it from the baseline's own "
            "view (design_of, sample_view)")
    return numeric // length


def _persistence_fit(name: str, seasonal: bool):
    def fit(setup, x, y):
        """Nothing is estimated: the fitted state is the window step each output repeats."""
        steps = np.asarray(_positions(setup, _season(setup) if seasonal else None),
                           dtype=np.int64)

        def head(d):
            width = _step_width(setup, name, d)
            point = d[:, [int(t) * width + width - 1 for t in steps]]
            if setup.kind is TaskKind.ORDINAL:
                k = len(setup.task.labels)
                ranks = np.clip(np.rint(point[:, 0]), 0, k - 1).astype(np.int64)
                return setup.class_head(np.eye(k)[ranks])
            return setup.value_head(point if setup.task.kind is TaskKind.FORECAST
                                    else point[:, 0])
        return FittedForm(head, {"steps": steps})
    return fit


def _naive(configure) -> BaselineForm:
    """Persistence over a window: the target at the anchor's step (or the step before it)."""
    return BaselineForm(fit=_persistence_fit("naive", False),
                        refusal=_persistence_refusal("naive"), sample_view=_target_history)


def _seasonal_naive(configure) -> BaselineForm:
    """Seasonal persistence: step h repeats the target `s·⌈h/s⌉` rows before it."""
    return BaselineForm(fit=_persistence_fit("seasonal-naive", True), refusal=_seasonal_refusal,
                        sample_view=_target_history)
# <<< shape


# >>> shape: graph
# ------------------------------------------------------------------ graphs

def _graph_linear_refusal(setup):
    if not setup.sample.is_node_level:
        level = setup.sample.graph.level.value if setup.sample.graph else None
        return ("graph-linear propagates once over a node-level graph's edges; this sample is "
                f"{setup.sample.shape.value}" + (f" at level {level}" if level else ""))
    return None


def _graph_linear(configure) -> BaselineForm:
    """The linear form over one fixed propagation: `[x ‖ mean_{N(i)} x]` per target node."""
    return replace(estimated_form(LINEAR, configure), refusal=_graph_linear_refusal,
                   view=_neighbour_view)


def _neighbour_view(setup) -> DesignView:
    """Neighbour-mean at node level; the sample's own view elsewhere (refused there)."""
    return DesignView.NEIGHBOUR_MEAN if setup.sample.is_node_level else view_for(setup.spec)
# <<< shape


__all__ = ("build",)


def _makers(spec: NetSpec) -> dict:
    """Every form maker this package builds, by name: the trio — the reconstruction forms of it
    for an unsupervised or anomaly task — and the forms its sample shape adds."""
    makers = dict(TRIO)
    # >>> task: unsupervised anomaly
    if Task.from_spec(spec).effective in (TaskKind.UNSUPERVISED, TaskKind.ANOMALY):
        makers.update({"trivial": _mean_row, "linear": _pca, "tree": _forest})
    # <<< task
    # >>> shape: window
    makers.update({"naive": _naive, "seasonal-naive": _seasonal_naive})
    # <<< shape
    # >>> shape: graph
    makers.update({"graph-linear": _graph_linear})
    # <<< shape
    return makers


def build(spec: NetSpec, which=("trivial", "linear", "tree", "lens"),
          context: BaselineContext | None = None) -> list:
    """The baselines /vega-eval asks for, unfitted, in the order asked (a reconstruction or
    anomaly task gets the reconstruction forms of the trio); /vega-eval fits and scores them."""
    spec, context = NetSpec.require(spec), context if context is not None else BaselineContext()
    makers = _makers(spec)
    unknown = [n for n in which if n != "lens" and n not in makers]
    if unknown:
        raise ValueError(f"unknown baseline(s) {unknown}; this package builds "
                         f"{sorted([*makers, 'lens'])}, its sample shape's and task's "
                         "(campaign-baseline is the campaign's stored R0 run, read by "
                         "/vega-eval, not a model to fit)")
    return [Baseline(n, spec, _lens_form(spec, context) if n == "lens"
                     else makers[n](unconfigured), context) for n in which]
