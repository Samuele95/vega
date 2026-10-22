"""
inspect_recompute.py — EI2, EI3 and EI4 of D6 §8.2 as pure functions over one served network
(`vega/network-view@2`) and one `detail: "exact"` explanation (`vega/explanation@2`), with the
batch context a graph needs. No HTTP here: `inspect_check.py` fetches, this module judges, and
the unit test feeds it payloads it made itself (and a copy with one weight perturbed).

    view = NetworkView(network, fetch)           the served nodes, edges and tensors (a
                                                 `deferred` tensor through `fetch(id)`)
    exact = ExactPass.of(explanation)            the exact tensors of every node, decoded
    Recompute(view, exact, context).run() -> RecomputeResult
        .ei2   [NodeCheck]  oracle(weights from the view, the node's captured input) against
                            its captured output (dense/head: W·x + b = pre, act(pre) = post,
                            and the head's train-fitted affine A37: out = post·scale + shift)
        .ei3   [NodeCheck]  each node's captured input is its predecessor's captured output
                            (concat and parallel: the join; residual: skip + branch)
        .ei4   [NodeCheck]  the last node's output is the model's raw output of the row

Tolerances (D6 §8.2): EI2 `|oracle − captured| ≤ 1e-5·max(1, |v|) + 1e-4·|v|` (float32 model,
float64 oracle); EI3 and EI4 compare captured float32 tensors, `≤ 1e-6·max(1, |v|)`.

The graph context. At node level the exact tensors are the explained node's rows only, while
a conv reads its neighbours: the context (`GraphContext`, from the served batch) carries every
node's inputs and the edges, and the node-level oracle chain runs the network's top-level
stages over all nodes in float64 — a conv is then judged at the explained node from the chain's
input rows, and the chain is held to the captured input at that node (EI3). At graph level the
exact tensors carry every node of the explained graph, and the context gives its edges.
"""
from __future__ import annotations

import base64
import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

import oracles as O
from check_kit import decode

__all__ = ["ExactPass", "GraphContext", "NetworkView", "NodeCheck", "Recompute",
           "RecomputeResult", "EI2_ATOL", "EI2_RTOL", "CHAIN_TOL", "perturbed"]

EI2_RTOL, EI2_ATOL = 1e-4, 1e-5
CHAIN_TOL = 1e-6
SYNTHETIC = frozenset({"input", "output", "residual_add", "parallel_merge"})
NO_TENSORS = frozenset({"edge_dropout"})
# dense children whose node output is the linear map alone (readers_group: a decoder layer
# carries the activation that follows it, which the next layer reads; a residual projection)
BARE_LINEAR = frozenset({"decoder_layer", "projection"})


# --------------------------------------------------------------------------- the view
class NetworkView:
    """A served network: nodes by id, their children and incoming edges, the tensors decoded
    on first use (`deferred` ones through `fetch`)."""

    def __init__(self, network: Mapping, fetch: Callable[[str], Mapping] | None = None):
        self.network = network
        self._nodes = {n["id"]: n for n in network.get("nodes") or ()}
        self._order = [n["id"] for n in network.get("nodes") or ()]
        self._tensors = dict(network.get("tensors") or {})
        self._fetch, self._cache = fetch, {}
        self.into: dict[str, list] = {}
        for e in network.get("edges") or ():
            self.into.setdefault(e["to"], []).append((e["from"], e["kind"]))

    @property
    def ids(self) -> list:
        return list(self._order)

    def node(self, node_id: str) -> Mapping:
        return self._nodes[node_id]

    def children(self, node_id: str) -> list:
        return [i for i in self._order if self._nodes[i].get("parent") == node_id]

    def is_descendant(self, node_id: str, ancestor: str) -> bool:
        parent = self._nodes[node_id].get("parent")
        while parent is not None:
            if parent == ancestor:
                return True
            parent = self._nodes[parent].get("parent")
        return False

    def top_level(self) -> list:
        return [i for i in self._order if self._nodes[i].get("parent") is None and
                self._nodes[i]["type"] not in ("input", "output")]

    def tensor(self, tensor_id: str) -> np.ndarray:
        if tensor_id not in self._cache:
            payload = self._tensors[tensor_id]
            if payload.get("encoding") == "deferred" and self._fetch is not None:
                payload = self._fetch(tensor_id)
            self._cache[tensor_id] = decode(payload)
        return self._cache[tensor_id]

    def tensors_of(self, node_id: str) -> dict:
        """{name: array} of the tensors registered under `<node>.<name>`."""
        head = node_id + "."
        return {tid[len(head):]: self.tensor(tid) for tid, t in self._tensors.items()
                if t.get("node") == node_id}

    def tensor_ids(self) -> list:
        return list(self._tensors)


@dataclass(frozen=True)
class ExactPass:
    """The exact tensors of one explanation, decoded: {node id: {key: array}}."""
    nodes: Mapping
    levels: Mapping

    @staticmethod
    def of(explanation: Mapping) -> "ExactPass":
        nodes, levels = {}, {}
        for node_id, payload in (explanation.get("nodes") or {}).items():
            levels[node_id] = payload.get("level")
            nodes[node_id] = {k: decode(v) for k, v in (payload.get("exact") or {}).items()
                              if v is not None}
        return ExactPass(nodes, levels)

    def get(self, node_id: str, key: str):
        return (self.nodes.get(node_id) or {}).get(key)


@dataclass(frozen=True)
class GraphContext:
    """The served batch of a graph explanation: every node's inputs, the edges, and which
    nodes the explained sample is (the target at node level, the graph's nodes at graph
    level), as positions in the batch."""
    x_num: np.ndarray | None
    x_cat: np.ndarray | None
    edge_index: np.ndarray
    edge_weight: np.ndarray | None
    level: str
    target: int | None = None
    graph_nodes: tuple = ()

    def local_edges(self) -> tuple:
        """The explained graph's edges renumbered to its node rows (graph level)."""
        nodes = np.asarray(self.graph_nodes, dtype=np.int64)
        where = {int(n): k for k, n in enumerate(nodes)}
        src, dst = self.edge_index
        keep = np.isin(src, nodes) & np.isin(dst, nodes)
        local = np.stack([[where[int(s)] for s in src[keep]], [where[int(d)] for d in dst[keep]]]) \
            if keep.any() else np.zeros((2, 0), dtype=np.int64)
        weight = None if self.edge_weight is None else self.edge_weight[keep]
        return local.astype(np.int64), weight


@dataclass(frozen=True)
class NodeCheck:
    node: str
    type: str
    check: str
    gap: float
    bound: float
    ok: bool
    detail: str = ""


@dataclass
class RecomputeResult:
    ei2: list = field(default_factory=list)
    ei3: list = field(default_factory=list)
    ei4: list = field(default_factory=list)

    def failures(self, which: str) -> list:
        return [c for c in getattr(self, which) if not c.ok]


def _own_activation(node: Mapping, params: Mapping) -> str:
    """The activation a dense node's own output carries: none for a bare linear (a decoder
    layer carries the activation that follows it, a projection none)."""
    return "none" if node.get("role") in BARE_LINEAR else str(params.get("activation", "none"))


def _thresholds(weights: Mapping) -> np.ndarray:
    """An ordinal head's ordered offsets θ_j = first − Σ_{i<j} softplus(gap_i)."""
    gaps = np.asarray(weights["gaps"], dtype=np.float64).reshape(-1)
    steps = np.concatenate([[0.0], np.logaddexp(0.0, gaps)]).cumsum()
    return float(np.asarray(weights["first"]).reshape(-1)[0]) - steps


def _bound(want, rtol: float, atol: float) -> float:
    want = np.abs(np.asarray(want, dtype=np.float64))
    if not want.size:
        return atol
    return atol * max(1.0, float(want.max())) + rtol * float(want.max())


def _compare(got, want, rtol: float, atol: float) -> tuple[float, float, bool]:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    bound = _bound(want, rtol, atol)
    if got.shape != want.shape:
        return math.inf, bound, False
    if not got.size:
        return 0.0, bound, True
    gap = np.abs(got - want)
    if not np.isfinite(gap).all():
        return math.inf, bound, False
    worst = gap - (atol * np.maximum(1.0, np.abs(want).max()) + rtol * np.abs(want))
    return float(gap.max()), bound, bool((worst <= 0).all())


def perturbed(network: Mapping, view: NetworkView, tensor_id: str, index: tuple,
              delta: float) -> dict:
    """A deep copy of a network payload with one weight moved by `delta` — re-encoded as the
    tensor travelled (a `deferred` one inlined as base64-le): the check's own sensitivity
    proof, a perturbed weight must fail EI2 at its node."""
    doc = copy.deepcopy(dict(network))
    payload = doc["tensors"][tensor_id]
    values = np.array(view.tensor(tensor_id), dtype=np.float32)
    values[index] += np.float32(delta)
    if payload.get("encoding") == "json":
        payload["data"] = [float(f"{float(v):.9g}") for v in values.reshape(-1)]
    else:
        payload.update(encoding="base64-le", data=base64.b64encode(
            values.astype("<f4").tobytes()).decode("ascii"))
    return doc


# --------------------------------------------------------------------------- EI2–EI4
class Recompute:
    """EI2, EI3 and EI4 over one exact explanation (see the module docstring)."""

    def __init__(self, view: NetworkView, exact: ExactPass, context: GraphContext | None = None,
                 raw_output=None):
        self.view, self.exact, self.context = view, exact, context
        self.raw_output = raw_output
        self.result = RecomputeResult()
        self._chain_in: dict = {}

    def run(self) -> RecomputeResult:
        if self.context is not None and self.context.level == "node":
            self._node_chain()
        for node_id in self.view.ids:
            self._ei2(node_id)
            self._ei3(node_id)
        self._ei4()
        return self.result

    # -- EI2 -------------------------------------------------------------------------------
    def _ei2(self, node_id: str) -> None:
        node = self.view.node(node_id)
        kind = node["type"]
        if kind in SYNTHETIC or kind in NO_TENSORS or self.exact.levels.get(node_id) != "exact":
            return
        out = self.exact.get(node_id, "out")
        if out is None:
            return
        if kind in O.SHAPE_ONLY:
            declared = node["config"].get("out_shape")
            ok = bool(np.isfinite(out).all()) and (not declared or
                                                   list(out.shape[-len(declared):]) == declared)
            self._add("ei2", node_id, kind, "custom: declared shape, finite", 0.0, 0.0, ok)
            return
        try:
            got = self._oracle_out(node_id, node)
        except Exception as exc:  # an oracle that cannot run on the payload is a failure
            self._add("ei2", node_id, kind, f"the oracle could not run: {type(exc).__name__}: "
                      f"{exc}", math.inf, 0.0, False)
            return
        if kind in ("dense", "head"):
            self._units(node_id, node, got)
            return
        gap, bound, ok = _compare(got, out, EI2_RTOL, EI2_ATOL)
        self._add("ei2", node_id, kind, "oracle(in) = out", gap, bound, ok)

    def _units(self, node_id: str, node: Mapping, oracle_out) -> None:
        """dense / head: W·x + b = pre (the `.linear` call), pre to post through the
        activation — an ordinal head adds its ordered thresholds, a regression-family head
        ends on its train-fitted affine (A37) — and the oracle end to end = out."""
        kind = node["type"]
        _, weights, params = O.view_block(self.view, node_id)
        x, how = self._input(node_id, node)
        pre, post, out = (self.exact.get(node_id, k) for k in ("pre", "post", "out"))
        scale, shift = weights.get("target_scale"), weights.get("target_shift")
        if pre is not None:
            gap, bound, ok = _compare(self._pick(O.affine(weights, x), how), pre, EI2_RTOL,
                                      EI2_ATOL)
            self._add("ei2", node_id, kind, "W·x + b = pre", gap, bound, ok)
            want = pre + _thresholds(weights) if "gaps" in weights else \
                O.activation(_own_activation(node, params), pre)
            want = want if scale is None else want * scale + shift
            if post is not None:
                gap, bound, ok = _compare(want, post, EI2_RTOL, EI2_ATOL)
                self._add("ei2", node_id, kind, "act(pre) = post" +
                          (" · scale + shift" if scale is not None else ""), gap, bound, ok)
        if node.get("role") in BARE_LINEAR:
            oracle_out = self._pick(O.affine(weights, x), how)
        want = oracle_out if scale is None else oracle_out * scale + shift
        gap, bound, ok = _compare(want, out, EI2_RTOL, EI2_ATOL)
        self._add("ei2", node_id, kind, "oracle(in) = out", gap, bound, ok)

    def _oracle_out(self, node_id: str, node: Mapping):
        kind, weights, params = O.view_block(self.view, node_id)
        x, how = self._input(node_id, node)
        if kind == "encoder_layer":
            y = O.encoder_layer_any(weights, x, params, 0)
        elif kind == "transformer_encoder":
            y = x
            for k in range(int(params.get("layers") or len(self.view.children(node_id)))):
                y = O.encoder_layer_any(weights, y, params, k)
        else:
            y = O.ORACLES[kind](weights, x, params)
        return self._pick(y, how)

    def _pick(self, y, how: str):
        """The explained sample's rows of an oracle output: `one` — the batch of one the
        oracle ran on; `rows` — a graph's node rows as they are; `chain` — the explained
        node's row of an output over every node of the batch."""
        y = np.asarray(y, dtype=np.float64)
        if how == "chain":
            return y[self.context.target]
        return y if how == "rows" else y[0]

    def _input(self, node_id: str, node: Mapping) -> tuple:
        """(what the oracle reads for this node, how its output is picked), per the node's
        call (inspect-contract §3.2)."""
        call, ctx = node.get("call"), self.context
        x = self.exact.get(node_id, "in")
        if call == "graph":
            return self._graph_input(node_id, x)
        if call == "gpool":
            rows = np.asarray(x, dtype=np.float64)
            return (rows, np.zeros(rows.shape[0], dtype=np.int64), 1), "one"
        if call == "concat":
            return (x[None], self._emb_source(node_id)), "one"
        if call == "tokens":
            ids = self.exact.get("input.cat", "out")
            return (x[None], None if ids is None else ids[None]), "one"
        if ctx is not None and ctx.level == "graph" and node.get("per") == "node":
            return x, "rows"
        return x[None], "one"

    def _graph_input(self, node_id: str, x) -> tuple:
        ctx = self.context
        if ctx is None:
            raise ValueError("a graph conv needs the served batch (its edges)")
        if ctx.level == "graph":
            edges, weight = ctx.local_edges()
            return (np.asarray(x, dtype=np.float64), edges, weight), "rows"
        return (self._chain_in[node_id], ctx.edge_index, ctx.edge_weight), "chain"

    def _emb_source(self, node_id: str):
        for src, kind in self.view.into.get(node_id, ()):
            if self.view.node(src)["type"] == "embedding":
                out = self.exact.get(src, "out")
                return None if out is None else out[None]
        return None

    # -- the node-level chain ----------------------------------------------------------------
    def _node_chain(self) -> None:
        """Every top-level stage over every node of the batch, in float64, from the served
        inputs; `_chain_in[id]` is what a graph conv reads."""
        ctx, running, emb = self.context, self.context.x_num, None
        running = None if running is None else np.asarray(running, dtype=np.float64)
        for node_id in self.view.top_level():
            node = self.view.node(node_id)
            kind, weights, params = O.view_block(self.view, node_id)
            call = node.get("call")
            if call == "emb":
                emb = O.ORACLES["embedding"](weights, ctx.x_cat, params)
            elif call == "concat":
                running = O.ORACLES["concat"](weights, (running, emb), params)
            elif call == "tokens":
                running = O.ORACLES[kind](weights, (running, ctx.x_cat), params)
            elif call == "graph":
                self._chain_in[node_id] = running
                running = O.ORACLES[kind](weights, (running, ctx.edge_index, ctx.edge_weight),
                                          params)
            elif call == "tensor":
                running = self._tensor_stage(kind, weights, params, running)

    @staticmethod
    def _tensor_stage(kind, weights, params, x):
        y = O.ORACLES[kind](weights, x, params)
        if kind == "head" and weights.get("target_scale") is not None:
            y = y * weights["target_scale"] + weights["target_shift"]
        return y

    # -- EI3 -------------------------------------------------------------------------------
    def _value_for(self, src: str, dst: str):
        """What `dst` reads from `src`: a group node feeds its descendants its own input; a
        decoder layer feeds the next the activation it carries applied to its output."""
        if self.view.is_descendant(dst, src):
            return self.exact.get(src, "in")
        out = self.exact.get(src, "out")
        node = self.view.node(src)
        if out is not None and node.get("role") == "decoder_layer":
            return O.activation(str(node["config"].get("activation", "none")), out)
        return out

    def _ei3(self, node_id: str) -> None:
        node = self.view.node(node_id)
        kind = node["type"]
        into = self.view.into.get(node_id, ())
        if not into or self.exact.levels.get(node_id) != "exact":
            return
        if kind == "residual_add":
            self._join(node_id, kind, into, "sum")
        elif kind == "parallel_merge":
            self._join(node_id, kind, into, node["config"].get("merge", "concat"))
        elif kind == "output":
            src = next(s for s, k in into if k == "readout")
            self._link(node_id, kind, self.exact.get(node_id, "out"), self._value_for(src, node_id),
                       f"out = {src}.out")
        else:
            self._reads(node_id, node, into)
        if node.get("call") == "graph" and self.context is not None and \
                self.context.level == "node" and node_id in self._chain_in:
            self._link(node_id, kind, self.exact.get(node_id, "in"),
                       self._chain_in[node_id][self.context.target],
                       "the served batch's chain = in at the explained node")

    def _reads(self, node_id: str, node: Mapping, into) -> None:
        data = [s for s, k in into if k in ("data", "skip", "branch", "embed")]
        mine = self.exact.get(node_id, "in")
        if mine is None or not data:
            return
        if node.get("call") in ("concat", "tokens") and "input.num" in data:
            data = ["input.num"]
        for src in data[:1]:
            self._link(node_id, node["type"], mine, self._value_for(src, node_id),
                       f"in = {src}.{'in' if self.view.is_descendant(node_id, src) else 'out'}")

    def _join(self, node_id: str, kind: str, into, how: str) -> None:
        parts = [self._value_for(s, node_id) for s, k in into if k in ("data", "skip", "merge")]
        out = self.exact.get(node_id, "out")
        if out is None or any(p is None for p in parts) or not parts:
            return
        want = np.sum(parts, axis=0) if how == "sum" else np.concatenate(parts, axis=-1)
        self._link(node_id, kind, out, want, f"out = {how} of {len(parts)} input(s)")

    def _link(self, node_id: str, kind: str, got, want, what: str) -> None:
        if got is None or want is None:
            return
        gap, bound, ok = _compare(got, want, 0.0, CHAIN_TOL)
        self._add("ei3", node_id, kind, what, gap, bound, ok)

    # -- EI4 -------------------------------------------------------------------------------
    def _ei4(self) -> None:
        readout = [s for s, k in self.view.into.get("output", ()) if k == "readout"]
        if not readout or self.raw_output is None:
            return
        last = readout[0]
        out = self.exact.get(last, "out")
        if out is None:
            return
        raw = np.asarray(self.raw_output, dtype=np.float64).reshape(np.shape(out))
        gap, bound, ok = _compare(out, raw, 0.0, CHAIN_TOL)
        self._add("ei4", last, self.view.node(last)["type"],
                  "the last node's out = the model's raw output", gap, bound, ok)

    def _add(self, which: str, node_id: str, kind: str, what: str, gap: float, bound: float,
             ok: bool) -> None:
        getattr(self.result, which).append(NodeCheck(node_id, kind, which.upper(), gap, bound,
                                                     ok, what))
