"""
inspection.inspector — `NetworkInspector`, the facade of the inspection surface: the network
behind one Predictor, read-only — its stage graph and weights (`vega/network-view@2`), what one
sample does as it passes through (`vega/explanation@2`), how many samples attribute at once
(`vega/attribution-table@1`) and whether all of it can be trusted (the status)
(inspect-contract §6.2, §6.3; D3 §9, §10.3; build plan R-D15, R-F5).

    NetworkInspector(model, spec, answers, output, tolerance)
      model       the served network, read through its Model port alone (`_Network`)
      answers     the Predictor (`_Answers`): `prepare(df_raw, history=None)` — the one raw ->
                  batch path — and `predict(df_raw, history=None)`, the served answer frame;
                  `predictions(...)`, when it has one, each row's warnings apart (a warning may
                  hold the frame's `; `); `info()`, when it has one, is the bundle's manifest
                  (the model version and `MANIFEST.serve.inspection`: enabled, weights, the
                  pinned network digest)
      output      MANIFEST.output (vega/output@2); tolerance MANIFEST.roundtrip.tolerance.value

    open(example)              build and check, once (idempotent); the `InspectionStatus`
    network(weights) / network_bytes(weights)   the body built at open, as a dict / its bytes
    tensor(tensor_id)          one tensor, base64-le, from the live parameters
    explain(query)             an `ExplainQuery` -> the `explanation` body (at node level,
                               with `sample.graph.names` when the graph names its nodes)
    attribute(rows, targets=None, method="grad_x_input", steps=32, history=None)
                               the attribution table of every sample the rows make
    status()                   the status as a dict (`/health.inspection`)
    close()                    the hooks removed; open() may run again

**Opening** (`open(example)`, an `ExplainQuery` over example rows — the bundle's own
`examples/rows.csv` through `predict.load_inspector`) runs in order and stops at the first
failure, which leaves the inspector not ready with the reason (`why`) and its hooks removed; a
failure never raises, since the service stays ready to predict: the manifest must not switch
inspection off; the stage graph is built over the example's prepared batch
(`graph.StageGraph.build`); the targets (`targets.TaskTargets.of`: a svdd network without its
persisted centre fails here); the hooks attached, once (`capture.HookSet`); the network bytes
for both weights modes, each digest equal to the one pack pinned when it pinned one ("the
network the service reads is not the one packed"); every sample of the example the network
answers with finite numbers (up to 16) explained at `detail: "summary"`, each fidelity within
the tolerance; the node checks measured (`recompute.ok: false` is reported, never a failure).

**Afterwards** the inspector only reads: each explanation runs its own `Capture` on the calling
thread, the permanent hooks write nothing without one, no parameter gradient is ever taken, so
`/v1/predict` and any number of explanations share the model. A method called on an inspector
that is not ready raises `InspectionUnavailable` naming why (the routes' 404); a request the
task cannot explain is a `query.QueryError` (422). A hand-built `ExplainQuery`, and each
selector handed to `attribute`, is checked as `QueryRules.parse` checks a request (a selector's
form — `{"threshold": 0.5}`, `True` or `"0"` is no threshold index —, the method, the detail)
before the task checks it. `explanation.FidelityError` and `interpolate.AttributionError` are
the explanation's own defects (500). The weights mode the bundle allows
(`MANIFEST.serve.inspection.weights`) bounds what is published: a bundle packed with
`--inspect-weights summary` answers `network("full")` and `tensor(id)` with
`InspectionUnavailable` / `codec.TensorUnavailable`.
"""
from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass, field, replace
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

from .attribution import Attributor
from .attribution_table import AttributionTable
from .capture import HookSet
from .codec import WEIGHTS_MODES, TensorUnavailable, num
from .explanation import (DEFAULT_TOLERANCE, Explained, Explainer, FidelityError, Served,
                          answered_rows, on_device)
from .graph import StageGraph
from .interpolate import AttributionError
from .query import DETAILS, METHODS, ExplainQuery, QueryError, QueryRules, target_form
from .targets import TaskTargets
from .view import canonical, network_view

if TYPE_CHECKING:                        # an annotation only: the domain's value, no code
    from ...domain import NetworkAnatomy

__all__ = ["EXAMPLE_SAMPLES", "FidelityError", "InspectionStatus", "InspectionSurface",
           "InspectionUnavailable", "NetworkInspector"]

EXAMPLE_SAMPLES = 16              # the example samples open() explains
OFF = "this bundle ships without inspection"
NOT_OPEN = "the inspector has not been opened on example rows"
NOT_PACKED = "the network the service reads is not the one packed"
NO_SAMPLE = ("the example rows make no sample the network answers with finite numbers, so no "
             "explanation can be checked against the served answer")
WITHHELD = "this service does not publish raw weights"
# what an opening that cannot finish raises: a refusal of the model, the records or the rows,
# never a reason to stop serving predictions
REFUSALS = (ValueError, RuntimeError, LookupError, TypeError, ArithmeticError)


class InspectionUnavailable(LookupError):
    """The inspector cannot answer: not opened, not ready (why), or weights withheld (404)."""


class _Answers(Protocol):
    """What the inspector reads from its Predictor (or from a harness standing in for one):
    `prepare` returns the `Prepared` value (`batch`, `index`: the `SampleIndex`,
    `sample_of_row`, `n_samples`); `predict` the answer frame, one row per row of `df_raw` in
    its order, a row's warnings joined as one text in `warnings`. A Predictor also has
    `predictions` (its `Prediction.warnings` a tuple), read where the joined text is ambiguous
    (`explanation.Served.of`); a harness without it has its text split."""

    def prepare(self, df_raw, history=None): ...

    def predict(self, df_raw, history=None): ...


@runtime_checkable
class _Network(Protocol):
    """All the inspection reads of the served network: three methods of its Model port (the
    blueprint's name, in `ports.py`) — the raw head of a batch, the network's view, the counts."""

    def head(self, batch: dict): ...
    def anatomy(self) -> NetworkAnatomy | None: ...
    def param_count(self) -> tuple: ...


@dataclass(frozen=True)
class InspectionStatus:
    """The inspection status (inspect-contract §6.2): `/health.inspection`."""

    enabled: bool
    ready: bool
    why: str | None
    weights: str
    network_digest: str | None = None
    pinned: bool = False
    coverage: Mapping | None = None
    fidelity: Mapping | None = None
    recompute: Mapping | None = None

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "ready": self.ready, "why": self.why,
                "weights": self.weights, "network_digest": self.network_digest,
                "pinned": self.pinned, "coverage": copy.deepcopy(self.coverage),
                "fidelity": copy.deepcopy(self.fidelity),
                "recompute": copy.deepcopy(self.recompute)}


@dataclass(frozen=True)
class _Bundle:
    """What the bundle's manifest says of inspection: the model version, whether inspection
    is on, the weights mode it allows, the network digests pack pinned (mode -> digest)."""

    model_version: str | None = None
    enabled: bool = True
    weights: str = "full"
    pins: Mapping = field(default_factory=lambda: MappingProxyType({}))

    @staticmethod
    def of(answers) -> "_Bundle":
        info = getattr(answers, "info", None)
        manifest = dict(info() or {}) if callable(info) else {}
        record = dict((manifest.get("serve") or {}).get("inspection") or {})
        weights = record.get("weights") or "full"
        if weights not in WEIGHTS_MODES:
            raise ValueError(f"MANIFEST.serve.inspection.weights is {weights!r}, not one of "
                             f"{list(WEIGHTS_MODES)}")
        pins = record.get("network_digest") or {}
        pins = {"full": pins} if isinstance(pins, str) else \
            {m: str(pins[m]) for m in WEIGHTS_MODES if pins.get(m)}
        return _Bundle(manifest.get("model_version"), record.get("enabled", True) is not False,
                       weights, MappingProxyType(pins))


@dataclass(frozen=True, eq=False)
class _Open:
    """An opened inspector's parts: graph, targets, hooks, explainer, table, bytes per mode."""

    graph: StageGraph
    targets: TaskTargets
    hooks: HookSet
    explainer: Explainer
    table: AttributionTable
    bytes: Mapping


@dataclass(frozen=True)
class InspectionSurface:
    """What a service reads of this surface, published by the inspector it opened: the explain
    rules and the query, and the refusals each answer maps (422, 404, 500 for an unfaithful
    explanation, 500 for a failed attribution) — so no caller reaches into the modules."""

    rules: type
    query: type
    refused: tuple
    absent: tuple
    unfaithful: tuple
    failed: tuple


class NetworkInspector:
    """The network behind one Predictor, read-only (see the module docstring)."""

    surface = InspectionSurface(QueryRules, ExplainQuery, (QueryError,),
                                (InspectionUnavailable, TensorUnavailable), (FidelityError,),
                                (AttributionError,))

    def __init__(self, model: _Network, spec, answers: _Answers, output: Mapping,
                 tolerance: float):
        if not isinstance(model, _Network):
            raise TypeError(f"a {type(model).__name__} is not the Model port the inspector reads")
        self._model, self._spec, self._answers = model, spec, answers
        self._output = dict(output or {})
        self._tolerance = float(DEFAULT_TOLERANCE if tolerance is None else tolerance)
        if not self._tolerance > 0:
            raise ValueError(f"the fidelity tolerance is {tolerance!r}; a positive number is "
                             "expected (MANIFEST.roundtrip.tolerance.value)")
        self._bundle = _Bundle.of(answers)
        self._lock = threading.Lock()
        self._open, self._tried = None, False
        self._status = self._closed(NOT_OPEN if self._bundle.enabled else OFF)

    # -- opening -----------------------------------------------------------------------------
    def open(self, example) -> InspectionStatus:
        """Build and check the inspector on `example` (an `ExplainQuery` over example rows),
        once: a second call returns the first call's status."""
        with self._lock:
            if not self._tried:
                self._open, self._status = self._opened(example)
                self._tried = True
            return self._status

    def close(self) -> None:
        """Remove the hooks; the inspector is not open until `open` runs again."""
        with self._lock:
            if self._open is not None:
                self._open.hooks.detach()
            self._open, self._tried = None, False
            self._status = self._closed(NOT_OPEN if self._bundle.enabled else OFF)

    def _opened(self, example) -> tuple:
        if not self._bundle.enabled:
            return None, self._closed(OFF)
        hooks = None
        try:
            prepared = self._answers.prepare(example.rows, example.history)
            if not int(prepared.n_samples):
                raise ValueError(NO_SAMPLE)
            graph = StageGraph.build(self._model, self._spec,
                                     (on_device(prepared.batch, self._model), prepared.index),
                                     self._output)
            targets = TaskTargets.of(self._spec, self._output, self._model)
            hooks = HookSet.attach(self._model, graph)
            parts = self._parts(graph, targets, hooks)
            fidelity, recompute = self._self_test(parts, prepared, example)
        except REFUSALS as exc:
            if hooks is not None:
                hooks.detach()
            return None, self._closed(str(exc) or type(exc).__name__)
        return parts, InspectionStatus(
            True, True, None, self._bundle.weights,
            json.loads(parts.bytes["full"])["network_digest"], bool(self._bundle.pins),
            _coverage(graph), fidelity, recompute)

    def _parts(self, graph, targets, hooks) -> _Open:
        attributor = Attributor(graph, targets)
        bodies = {mode: network_view(graph, mode, self._bundle.model_version)
                  for mode in WEIGHTS_MODES}
        for mode, pinned in self._bundle.pins.items():
            if bodies[mode]["network_digest"] != pinned:
                raise ValueError(f"{NOT_PACKED}: its {mode} digest is "
                                 f"{bodies[mode]['network_digest']}, MANIFEST.serve.inspection "
                                 f"pinned {pinned}")
        return _Open(graph, targets, hooks,
                     Explainer(self._model, graph, targets, attributor, self._tolerance),
                     AttributionTable(self._model, graph, targets, attributor),
                     {mode: canonical(body) for mode, body in bodies.items()})

    def _self_test(self, parts: _Open, prepared, example) -> tuple:
        """Every example sample (up to 16) explained at summary detail: (fidelity, recompute)."""
        finite = parts.explainer.finite(prepared)
        pairs = [(k, row) for k, row in answered_rows(prepared) if k in finite][:EXAMPLE_SAMPLES]
        if not pairs:
            raise ValueError(NO_SAMPLE)
        served = Served.of(self._answers, example.rows, example.history)
        query = replace(example, target=None, method="grad_x_input", detail="summary")
        worst, checks, held = 0.0, [], True
        for k, row in pairs:
            body, ok = parts.explainer.run(prepared, served, Explained(k, row), query)
            fid = body["fidelity"]
            worst = max(worst, fid["head_max_abs_diff"], fid["served_max_abs_diff"] or 0.0)
            checks.append(body["recompute"]["worst"] or 0.0)
            held = held and ok
        return ({"samples": len(pairs), "max_abs_diff": num(worst),
                 "tolerance": num(self._tolerance)},
                {"worst": num(max(checks)), "ok": held})

    def _closed(self, why: str) -> InspectionStatus:
        return InspectionStatus(self._bundle.enabled, False, why, self._bundle.weights)

    def _ready(self) -> _Open:
        opened = self._open
        if opened is None:
            raise InspectionUnavailable(self._status.why)
        return opened

    # -- reading -----------------------------------------------------------------------------
    @property
    def weights(self) -> str:
        """The weights mode the bundle publishes (`full` or `summary`)."""
        return self._bundle.weights

    def status(self) -> dict:
        """The status (inspect-contract §6.2) as `/health.inspection` carries it."""
        return self._status.to_dict()

    def network_bytes(self, weights: str = "full") -> bytes:
        """The canonical `vega/network-view@2` bytes built at open, for a weights mode."""
        opened = self._ready()
        if weights not in WEIGHTS_MODES:
            raise ValueError(f"weights mode {weights!r} is not one of {list(WEIGHTS_MODES)}")
        if weights == "full" and self._bundle.weights != "full":
            raise InspectionUnavailable(WITHHELD)
        return opened.bytes[weights]

    def network(self, weights: str = "full") -> dict:
        """The `vega/network-view@2` body, a fresh copy per call."""
        return json.loads(self.network_bytes(weights))

    def tensor(self, tensor_id: str) -> dict:
        """One tensor as base64-le from the live parameters (`codec.TensorUnavailable`: an
        unknown id, or weights the bundle withholds)."""
        return self._ready().graph.codec.fetch(str(tensor_id), self._bundle.weights)

    def explain(self, query) -> dict:
        """The `explanation` of the sample `query.at` points at (`query.QueryError` when the
        query is not one `QueryRules.parse` builds, the task cannot explain the target, or no
        sample answers the row)."""
        opened = self._ready()
        query = _checked(query)
        opened.targets.validate(query.target)
        prepared = self._answers.prepare(query.rows, query.history)
        k = query.sample(prepared.index)
        served = Served.of(self._answers, query.rows, query.history)
        body = opened.explainer.explain(prepared, served, Explained(k, int(query.at)), query)
        return _named(body, getattr(self._answers, "context", None))

    def attribute(self, rows, targets=None, method: str = "grad_x_input", steps: int = 32,
                  history=None) -> dict:
        """The `vega/attribution-table@1` of every sample `rows` make (with `history` before
        them): `targets` one selector per row of `rows` (None, or a None entry: the task's
        default), each checked as a request's (`query.QueryError` naming it and its row)."""
        opened = self._ready()
        prepared = self._answers.prepare(rows, history)
        served = Served.of(self._answers, rows, history)
        return opened.table.build(prepared, served, targets, method, steps)


def _checked(query):
    """`query` as `QueryRules.parse` would have built it: `at` an integer (no bool), its
    target's form (one known selector, a name as a string, an index as an integer that is no
    bool), its method and detail; a `QueryError` on the field at fault otherwise."""
    if isinstance(query.at, bool) or not isinstance(query.at, Integral):
        raise QueryError("at", f"`at` is an integer index into `rows`, not {query.at!r}")
    if query.method not in METHODS:
        raise QueryError("method", f"`method` is one of {', '.join(METHODS)}, not "
                                   f"{query.method!r}")
    if query.detail not in DETAILS:
        raise QueryError("detail", f"`detail` is one of {', '.join(DETAILS)}, not "
                                   f"{query.detail!r}")
    return replace(query, target=target_form(query.target))


def _named(body: dict, context) -> dict:
    """At node level, `sample.graph.names`: the column that names a node and, for each
    reference node the explanation mentions, its value there (`GraphContext.reference_names`),
    so a reader names a node by the data's own key and never by its index in the assembled
    graph. Left out when the graph names no node (knn, spatial) or the answer is graph level."""
    graph = body["sample"].get("graph")
    key = getattr(context, "node_key", None)
    if not graph or graph.get("level") != "node" or key is None:
        return body
    names = context.reference_names(_mentioned(body))
    graph["names"] = {"column": key, "of": {str(i): v for i, v in names.items()}}
    return body


def _mentioned(body: dict) -> set:
    """Every node id an explanation names: the attribution's neighbours and edges, each node
    payload's kept ids (`nodes` form) and the messages its extras carry."""
    att = body.get("attribution") or {}
    ids = {n["node"] for n in att.get("neighbours") or ()}
    ids.update(v for e in att.get("edges") or () for v in (e["src"], e["dst"]))
    for part in (body.get("nodes") or {}).values():
        forms = [part.get(f) for f in ("in", "out", "grad_out", "relevance")]
        ids.update(i for f in forms if isinstance(f, dict) and f.get("form") == "nodes"
                   for i in f.get("ids") or ())
        messages = (part.get("extra") or {}).get("messages")
        if isinstance(messages, list):
            ids.update(m[end] for m in messages if isinstance(m, dict) for end in ("src", "dst")
                       if isinstance(m.get(end), int))
    return ids


def _coverage(graph) -> dict:
    return {"nodes": len(graph.nodes), "types": sorted({n.type for n in graph.nodes}),
            "opaque": [n.id for n in graph.nodes if n.view == "opaque"]}
