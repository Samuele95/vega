"""
inspection.view — the body of `GET /v1/network`, schema `vega/network-view@2`, around the stage
graph (inspect-contract §4, §5.3; build plan R-B7, R-C1, R-F7): the `task`, `sample` and
`inputs` sections, the header, and the canonical bytes and digest.

- **task** — the kind and base, the link (`TaskReadout.link`, the one source of output
  semantics, which must be the target table's), the head's outputs (`kind` and `names`, §4.5),
  `positive`, `threshold` and its scale, `quantiles`, `decision_rule`, the selectable target
  kinds and the default one (§4.1: `TARGETS`, with the loss rows `BY_LOSS`). The bundle's
  `MANIFEST.output` (`vega/output@2`), when given, supplies the classes, the positive class and
  the decision threshold; without it they are the spec's, and a binary or multilabel decision
  is at σ = 0.5 (the logit's sign).
- **inputs** — `numeric` and `nominal` are `data_contract.features[].name`, numeric then
  nominal, as the network reads them; `order` appends the autoregressive target (`past_target`,
  the last channel of `input.num`); `width` counts the numeric channels and each nominal's
  embedding dims (1 for a nominal not embedded); `axis_labels` names the channels, the window's
  steps `t−(L−1) … t−1, t` and the tokens (`[cls]` then the features for a tokenizer, the steps
  for a sequence encoder).
- **network_digest** — the sha256 of the canonical serialisation (`sort_keys`, separators
  `(",", ":")`, `allow_nan=False`) of the body without that key; two builds of one network
  give the same bytes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType

from . import registry
from .readers import GraphError
from ...domain import FeatureContract, SampleSpec, Task

__all__ = ["BY_LOSS", "SCHEMA", "TARGETS", "Inputs", "canonical", "network_digest",
           "network_view", "task_section"]

SCHEMA = "vega/network-view@2"
INSPECTION = "vega/inspection@1"
# effective kind -> (link, decision rule, default target, selectable targets): §4.1
TARGETS = MappingProxyType({
    "binary": ("sigmoid", "threshold", "binary_logit", ("binary_logit",)),
    "multiclass": ("softmax", "argmax", "logit", ("logit",)),
    "multilabel": ("sigmoid_each", "per_label_threshold", "label_logit", ("label_logit",)),
    "regression": ("identity", "none", "value", ("value",)),
    "ordinal": ("cumulative", "cumulative_count", "expected_rank",
                ("expected_rank", "threshold_logit")),
    "unsupervised": ("reconstruction", "none", "reconstruction_error",
                     ("reconstruction_error", "feature_error")),
    "anomaly": ("reconstruction", "threshold", "reconstruction_error",
                ("reconstruction_error", "feature_error")),
    "forecast": ("horizon", "none", "forecast_step", ("forecast_step",))})
BY_LOSS = MappingProxyType({
    ("regression", "quantile"): ("identity", "none", "quantile", ("value", "quantile")),
    ("anomaly", "svdd"): ("distance", "threshold", "svdd_distance", ("svdd_distance",))})
THRESHOLD_SCALE = MappingProxyType({"binary": "probability", "multilabel": "probability",
                                    "anomaly": "score"})
SEQUENCE_ENCODERS = frozenset({"transformer_encoder", "attention"})


def _data(spec) -> dict:
    return spec.to_dict() if callable(getattr(spec, "to_dict", None)) else dict(spec)


@dataclass(frozen=True)
class Inputs:
    """What the network reads, from the contract: the sample shape, the numeric and nominal
    columns (with cardinalities), the autoregressive target, the window length."""

    shape: str
    numeric: tuple
    nominal: tuple
    cards: tuple
    past: str | None = None
    length: int | None = None
    graph_level: bool = False

    @staticmethod
    def of(spec) -> "Inputs":
        data = _data(spec)
        contract, sample = FeatureContract.from_spec(data), SampleSpec.from_spec(data)
        nominal = tuple(contract.nominal_columns)
        window = sample.window
        return Inputs(sample.shape.value, tuple(contract.numeric_columns), nominal,
                      contract.cardinalities,
                      contract.label if window is not None and window.autoregressive else None,
                      int(window.length) if window is not None else None,
                      sample.is_graph_level)

    @property
    def feature_names(self) -> tuple:
        """The channels of `input.num`: the numeric columns, then the autoregressive target."""
        return self.numeric + ((self.past,) if self.past else ())

    @property
    def num_shape(self) -> tuple:
        return ((self.length,) if self.length else ()) + (len(self.feature_names),)

    @property
    def cat_shape(self) -> tuple:
        return ((self.length,) if self.length else ()) + (len(self.nominal),)

    def steps(self) -> list | None:
        if not self.length:
            return None
        return [f"t−{self.length - 1 - k}" for k in range(self.length - 1)] + ["t"]


# ------------------------------------------------------------------------------ the task

def _row(kind: str, loss) -> tuple:
    return BY_LOSS.get((kind, loss)) or TARGETS[kind]


def _classes(task, output) -> list:
    return [str(c) for c in (output.get("classes") or task.labels.classes or ())]


def _positive(task, output, kind):
    if kind not in ("binary", "anomaly") or not (output.get("classes") or task.labels.classes):
        return None
    if output.get("positive") is not None:
        return str(output["positive"])
    return str(task.positive) if task.positive is not None else _classes(task, output)[-1]


def _outputs(kind, task, output, parts) -> dict:
    """`{kind, names}` of the head (§4.5); `parts` = (data, anatomy, quantiles)."""
    data, anatomy, quantiles = parts
    names = _classes(task, output)
    qnames = [f"q{q:g}" for q in quantiles or ()]
    if kind == "forecast":
        horizon = int(((data.get("data_contract") or {}).get("sample") or {}).get("horizon") or 1)
        steps = [f"t+{h}" for h in range(1, horizon + 1)]
        return {"kind": "horizon",
                "names": [f"{s}/{q}" for s in steps for q in qnames] if qnames else steps}
    if kind == "regression":
        return {"kind": "quantiles", "names": qnames} if qnames else \
            {"kind": "value", "names": ["value"]}
    if task.objective.value == "svdd":
        return {"kind": "embedding", "names": [f"z{j}" for j in range(int(anatomy.shapes[-1][-1]))]}
    if kind in ("unsupervised", "anomaly"):
        inputs = Inputs.of(data)
        return {"kind": "reconstruction", "names": list(inputs.numeric + inputs.nominal)}
    table = {"binary": ("score", [_positive(task, output, kind)]),
             "multiclass": ("classes", names), "multilabel": ("labels", names),
             "ordinal": ("thresholds", [f">{c}" for c in names[:-1]])}
    return dict(zip(("kind", "names"), table[kind]))


def task_section(spec, model, output=None) -> dict:
    """`network.task` (§4.1, §5.3) of `model` (read through its Model port's `anatomy()`)
    under `spec`, the bundle's MANIFEST.output supplying classes, the positive class and the
    threshold when given."""
    data, output, anatomy = _data(spec), dict(output or {}), model.anatomy()
    task = Task.from_spec(data)
    kind, loss = task.effective.value, (data.get("loss") or {}).get("name")
    link, rule, default, selectable = _row(kind, loss)
    if anatomy.readout.link != link:
        raise GraphError(f"the network's readout links {anatomy.readout.link!r}, the target "
                         f"table {link!r} for a {kind} task under the {loss} loss")
    raw_q = ((data.get("loss") or {}).get("params") or {}).get("quantiles") \
        if loss == "quantile" else None
    quantiles = [float(q) for q in (raw_q if isinstance(raw_q, (list, tuple)) else [raw_q])] \
        if raw_q is not None else None
    threshold = output["threshold"] if "threshold" in output else \
        (0.5 if kind in ("binary", "multilabel") else None)
    return {"kind": task.kind.value, "base": task.base.value if task.base else None,
            "link": link, "outputs": _outputs(kind, task, output, (data, anatomy, quantiles)),
            "positive": _positive(task, output, kind),
            "threshold": None if threshold is None else float(threshold),
            "threshold_scale": output.get("threshold_scale", THRESHOLD_SCALE.get(kind)),
            "quantiles": quantiles, "decision_rule": rule, "targets": list(selectable),
            "default_target": default}


# ------------------------------------------------------------------------------ the sections

def sample_section(spec) -> dict:
    sample = SampleSpec.from_spec(_data(spec))
    w, g = sample.window, sample.graph
    return {"shape": sample.shape.value,
            "length": int(w.length) if w else None, "stride": int(w.stride) if w else None,
            "horizon": int(w.horizon) if w and w.is_forecast else None,
            "autoregressive": bool(w.autoregressive) if w else None,
            "level": g.level.value if g else None, "directed": bool(g.directed) if g else None,
            "self_loops": bool(g.self_loops) if g else None,
            "protocol": g.protocol.value if g else None,
            "batching": g.batching.value if g else None}


def _nominal_dims(graph) -> tuple:
    """({column: dim} embedded by an `embedding`, {column: d} read as a tokenizer's token)."""
    embedded, tokens = {}, {}
    for node in graph.nodes:
        module = graph.module(node.id)
        if node.type == "embedding":
            embedded.update({str(c): int(e.embedding_dim)
                             for c, e in zip(module.columns, module.embeddings)})
        elif node.type == "feature_tokenizer":
            tokens.update({c: int(module.weight.shape[1]) for c in graph.inputs.nominal})
    return embedded, tokens


def inputs_section(graph) -> dict:
    i = graph.inputs
    embedded, tokens = _nominal_dims(graph)
    types = {n.type for n in graph.nodes}
    token = (["[cls]", *i.numeric, *i.nominal] if "feature_tokenizer" in types else
             i.steps() if types & SEQUENCE_ENCODERS and i.shape == "window" else None)
    return {"numeric": list(i.numeric),
            "nominal": [{"name": c, "cardinality": k, "dim": embedded.get(c, tokens.get(c))}
                        for c, k in zip(i.nominal, i.cards)],
            "past_target": i.past, "order": [*i.numeric, *i.nominal, *([i.past] if i.past else [])],
            "width": len(i.feature_names) + sum(embedded.get(c, 1) for c in i.nominal),
            "edges": {"weighted": bool(graph.weighted)} if i.shape == "graph" else None,
            "axis_labels": {"feature": [*i.numeric, *([{"name": i.past, "past": True}]
                                                     if i.past else [])],
                            "step": i.steps(), "token": token}}


def canonical(body) -> bytes:
    """The canonical bytes of a body: sorted keys, no spaces, no NaN."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def network_digest(body) -> str:
    """sha256 of the canonical body without its `network_digest`."""
    rest = {k: v for k, v in body.items() if k != "network_digest"}
    return "sha256:" + hashlib.sha256(canonical(rest)).hexdigest()


def network_view(graph, weights: str = "full", model_version: str | None = None) -> dict:
    """The `vega/network-view@2` body of a stage graph under a weights mode."""
    data = graph.spec
    family = data.get("family") or {}
    total, trainable = graph.params
    body = {"schema": SCHEMA, "available": True,
            "model_version": str(model_version or graph.spec_digest),
            "spec_digest": graph.spec_digest, "weights": weights,
            "family": {"name": str(family.get("name") or "user-defined"),
                       "source": None if family.get("source") is None else
                       str(family["source"])},
            "task": copy.deepcopy(graph.task), "sample": sample_section(data),
            "inputs": inputs_section(graph), "params": {"total": total, "trainable": trainable},
            **graph.to_dict(weights), "limits": dict(graph.codec.limits),
            "inspection": {"schema": INSPECTION, "readers": registry.readers_digest()}}
    body["network_digest"] = network_digest(body)
    return body
