"""
inspection.parts — what each node of a captured pass contributes to an explanation
(inspect-contract §5.4 `nodes` and `recompute`; build plan R-F5).

`NodeParts(cap).build(node_id, level)` is the node's parts for `reduce.Budget.place`:

- at level `full`: `in`, `out` and `grad_out` (∂s/∂out) of the explained sample in their
  reduced forms, `relevance` = out ⊙ grad_out (reduced by sums), `extra` — the reader's
  `probe` — and `recompute` — the reader's check;
- at level `exact`: the same, plus `exact` — full `in`, `out` and `grad_out` of the sample as
  base64-le tensors and the reader's extra exact tensors (§3.1's `exact` column);
- at level `summary`: per-axis norms and sums of the same four tensors, the check's value, and
  no probe.

At graph node level a node tensor is reduced over the explained node's neighbourhood (the
`nodes` form: the kept ids and hops, each node's norm, the target's values); at graph level
over the explained graph's nodes. The edge stream (`input.edges`, an `edge_dropout`) carries no
per-sample tensor: its parts are null but for its check. A probe and a check are computed once
per node and pass, whatever the levels the budget asks for.

`place(budget)` places every node in graph order under the explanation's `Budget` — the same
request gives the same bytes — and `report()` is the explanation's `recompute` object: the
nodes with a check, the worst max |Δ| and the node that has it (the first in graph order on a
tie). `chain(network_chain)` is the explanation's `chain` when the network has a dense chain
(§5.3): `inputs` (what the first layer reads: the concat output), and per chain layer `pre`
(z, its `:pre` call), `post` (what the next layer reads — the activation and the inactive
dropouts applied; the head's output for the last) and `grad_pre` (∂s/∂z), so a reader
recomputes `W·post[l−1] + b = pre[l]` with the published weights.
"""
from __future__ import annotations

import torch

from . import registry
from .codec import exact_set, num, nums
from .probes import axes_of, f64, node_rows

__all__ = ["EDGE_TYPES", "NodeParts"]

EDGE_TYPES = frozenset({"edge_dropout"})


class NodeParts:
    """The node parts of one captured, differentiated pass (see the module docstring)."""

    def __init__(self, cap):
        self.cap = cap

    def _tensors(self, node) -> tuple:
        """(in, out, grad_out) of a node over the whole batch; nulls on the edge stream."""
        if node.type in EDGE_TYPES or node.id == "input.edges":
            return None, None, None
        out = self.cap.value(node.id)
        return self.cap.value_in(node.id), out, self.cap.grad_of(out)

    def probe(self, node_id: str) -> dict:
        node = self.cap.graph.node(node_id)
        return self.cap.memo((node_id, "probe"), lambda: registry.reader(node.type).probe(
            self.cap, node))

    def recompute(self, node_id: str):
        node = self.cap.graph.node(node_id)
        value = self.cap.memo((node_id, "recompute"), lambda: registry.reader(
            node.type).recompute(self.cap, node))
        return None if value is None else float(value)

    def _reduced(self, t, node, side: str, kind: str = "value"):
        if t is None:
            return None
        cap, focus = self.cap, self.cap.focus
        if focus.level == "node" and cap.per(node, side) == "node":
            return cap.reducer.nodes(f64(t), focus.nodes, focus.hops, focus.target)
        return cap.reducer.reduce(f64(cap.take(t, node, side)), kind)

    def _summary(self, t, node, side: str):
        if t is None:
            return None
        s = f64(self.cap.take(t, node, side))
        return self.cap.reducer.summarise(s, self._axes(s, node, side))

    def _axes(self, s, node, side: str) -> tuple:
        return axes_of(s, node.axes if side == "out" else (), node_rows(self.cap, node, side))

    def build(self, node_id: str, level: str) -> dict:
        """The node's parts at `level` (`exact`, `full` or `summary`)."""
        node = self.cap.graph.node(node_id)
        x, y, g = self._tensors(node)
        rel = None if y is None or g is None else f64(y) * f64(g)
        parts = {"recompute": self.recompute(node_id)}
        if level == "summary":
            parts.update({"in": self._summary(x, node, "in"), "out": self._summary(y, node, "out"),
                          "grad_out": self._summary(g, node, "out"),
                          "relevance": self._summary(rel, node, "out")})
            return parts
        parts.update({"in": self._reduced(x, node, "in"), "out": self._reduced(y, node, "out"),
                      "grad_out": self._reduced(g, node, "out"),
                      "relevance": self._reduced(rel, node, "out", "relevance"),
                      "extra": self.probe(node_id)})
        if level == "exact":
            parts["exact"] = self._exact(node, (x, y, g))
        return parts

    def _exact(self, node, tensors) -> dict:
        found = {}
        for key, t, side in zip(("in", "out", "grad_out"), tensors, ("in", "out", "out")):
            s = self.cap.take(t, node, side) if t is not None else None
            if s is not None:
                found[key] = (s, self._axes(s, node, side))
        found.update(registry.reader(node.type).exact(self.cap, node))
        return exact_set(found)

    def place(self, budget) -> dict:
        """{node id: payload} of every node, in graph order, under `budget`."""
        return {n.id: budget.place(n.id, lambda level, i=n.id: self.build(i, level))
                for n in self.cap.graph.nodes}

    def report(self) -> dict:
        """The explanation's `recompute`: {nodes, worst, worst_node}."""
        checked = [(n.id, self.recompute(n.id)) for n in self.cap.graph.nodes]
        checked = [(i, v) for i, v in checked if v is not None]
        if not checked:
            return {"nodes": 0, "worst": None, "worst_node": None}
        worst = max(checked, key=lambda pair: pair[1])
        return {"nodes": len(checked), "worst": num(worst[1]), "worst_node": worst[0]}

    def chain(self, network_chain):
        """The explanation's `chain` (see the module docstring), or None without one."""
        if not network_chain:
            return None
        cap, xs, pre, grad = self.cap, [], [], []
        for layer in network_chain["layers"]:
            node = cap.graph.node(layer["node"])
            x, z = cap.pre(node.id)
            g = cap.grad_of(z)
            xs.append(f64(cap.take(x, node, "in")))
            pre.append(f64(cap.take(z, node)))
            grad.append(torch.zeros_like(pre[-1]) if g is None else f64(cap.take(g, node)))
        last = cap.graph.node(network_chain["layers"][-1]["node"])
        post = xs[1:] + [f64(cap.take(cap.value(last.id), last))]
        return {"inputs": nums(xs[0]), "pre": [nums(t) for t in pre],
                "post": [nums(t) for t in post], "grad_pre": [nums(t) for t in grad]}
