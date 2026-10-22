"""
inspection.attribution — how the explained scalar `s` spreads over the inputs of the explained
sample (inspect-contract §5.4 "Attribution", §5.5; D3 §8.3, §9.3; build plan I05).

`Attributor(graph, targets)` attributes for one served network: `graph` is its stage graph
(`graph.StageGraph`), `targets` its `targets.TaskTargets`.

- **`gradient_x_input(cap, target)`** reads a `capture.Capture` that ran the explained batch and
  was differentiated once for `s` (the target's scalar of sample `cap.focus.index`, computed over
  `cap.batch`): a = x ⊙ ∂s/∂x over the numeric channels (the leaf `cap.batch["x_num"]`). A
  nominal column is attributed at its embedding output, since an id is not differentiable:
  a_c = Σ_d e_{c,d}·∂s/∂e_{c,d}, its `grad` the norm ‖∂s/∂e_c‖ — the column's slice of an
  `embedding` stage, or its token of a `feature_tokenizer`; a nominal no stage embeds is not
  read by the network and carries 0. The edge relevances are ∂s/∂w at the capture's edge-weight
  leaf. `completeness` is null: the method makes no completeness claim.
- **`integrated(model, batch, target, k, steps)`**: the baseline x₀ is 0 in shaped space and the
  zero vector at each embedding output; `steps` interpolants at the midpoint α run as one batch
  (`interpolate.Path`); a = (x − x₀) ⊙ mean_α ∂s/∂x(x₀ + α(x − x₀)), and the same dot product at
  each embedding output; `grad` is the path's mean gradient; graph edges are held at their
  weight (1 on an unweighted graph), and their relevance is ∂s/∂w at x. `completeness = {sum:
  Σa, target_minus_baseline: s(x) − s(x₀), gap: |difference|}`; both ends run by the same path,
  whose α = 1 end must reproduce the network's own forward. A row or window sample runs alone;
  a node-level one over the whole graph with its target alone; a graph-level one as its graph
  alone. On a graph, `steps` × nodes is bounded by `ig_batch_max`.
- **`table(model, batch, targets, method, steps)`** is the batch API's per-sample part (§5.5):
  the batch runs once (gradient × input) or as its interpolants (integrated gradients), with
  one gradient for the batch — at graph node level, where the samples share one graph, one per
  target — each sample with its own target, no node payloads.

`total` is Σ over every attribution (every node of a graph). The payloads by sample shape are
`attribution_parts.Layout`'s (row, window) and `attribution_graph.Around`'s (graph).
"""
from __future__ import annotations

import torch

from .attribution_graph import Around, edges_back
from .attribution_parts import TOP_FEATURES, Layout, Shares, Site, nominal_sites
from .codec import num
from .focus import Focus
from .interpolate import EDGES, NUMERIC, AttributionError, Path, midpoints
from .reduce import Reducer

__all__ = ["MATRIX_MAX", "METHODS", "TOP_FEATURES", "AttributionError", "Attributor", "Site",
           "nominal_sites"]

METHODS = ("grad_x_input", "integrated_gradients")
MATRIX_MAX = 65536          # a window's steps × channels travel whole up to this many cells
_ENDS = (0.0, 1.0)          # the path's two ends, run together: s(x₀) and s(x)


def _sample(batch, k: int) -> dict:
    """Row or window sample `k` alone, as a batch of one."""
    return {key: batch[key][k:k + 1] for key in (NUMERIC, "x_cat") if batch.get(key) is not None}


def _one_graph(batch, members) -> tuple:
    """(the explained graph alone — its nodes renumbered in order, its edges, one graph —, the
    mask of the batch's edges it keeps): graph level."""
    rows = Layout.rows(batch)
    device = batch["edge_index"].device
    idx = torch.as_tensor(list(members), dtype=torch.long, device=device)
    renum = torch.full((rows,), -1, dtype=torch.long, device=device)
    renum[idx] = torch.arange(idx.numel(), device=device)
    edge_index = batch["edge_index"]
    keep = (renum[edge_index[0]] >= 0) & (renum[edge_index[1]] >= 0)
    one = {key: batch[key].index_select(0, idx) for key in (NUMERIC, "x_cat")
           if batch.get(key) is not None}
    one.update({"edge_index": renum[edge_index[:, keep]], "num_graphs": 1,
                "graph_index": torch.zeros(idx.numel(), dtype=torch.long, device=device)})
    if batch.get(EDGES) is not None:
        one[EDGES] = batch[EDGES][keep]
    return one, keep


def _unplaced(shares, grad):
    return shares, grad


class Attributor:
    """The attributions of one served network (see the module docstring)."""

    def __init__(self, graph, targets, limits=None):
        self.graph, self.targets = graph, targets
        self.limits = dict(limits or graph.codec.limits)
        self.layout = Layout.of(graph, self.limits)
        self.around = Around(self.layout)
        self.sites = nominal_sites(graph)

    # -- one explanation ------------------------------------------------------------------------
    def gradient_x_input(self, cap, target) -> dict:
        """The gradient × input `attribution` of a captured, differentiated pass."""
        x = cap.batch.get(NUMERIC)
        embedded = [(site, cap.value(site.node), cap.grad_of(cap.value(site.node)))
                    for site in self.sites]
        shares = Shares.of(x, cap.grad_of(x), cap.batch.get("x_cat"), embedded)
        w = cap.batch.get(EDGES)
        edges = (cap.batch.get("edge_index"), cap.grad_of(w) if w is not None else None)
        return self._payload("grad_x_input", target, shares, cap.focus, edges)

    def integrated(self, model, batch, target, k: int, steps: int) -> dict:
        """The integrated-gradient `attribution` of sample `k` of `batch` (the prepared batch
        the network answers) for `target`, at `steps` interpolants."""
        self._check_steps(steps)
        focus = Focus.of(self.graph, batch, k)
        sample, place = self._alone(batch, focus)
        self._bound(steps, sample)
        path = Path(model, self.layout.graphed)
        mids = path.run(sample, midpoints(steps, sample.get(NUMERIC)))
        grads = mids.gradients(self.targets.scalar(target, mids.head, mids.scoring).sum())
        (s0, s1), at_x = self._ends(model, path, sample, target)
        shares, wgrad = place(self._shares(mids, sample, grads), at_x.get(EDGES))
        body = self._payload("integrated_gradients", target, shares, focus,
                             (batch.get("edge_index"), wgrad))
        delta = num(s1 - s0)             # the gap is the difference of the published numbers
        body["completeness"] = {"sum": body["total"], "target_minus_baseline": delta,
                                "gap": num(abs(body["total"] - delta))}
        return body

    def neighbourhood(self, edge_index, targets, hops: int, totals=None) -> tuple:
        """(ids, hops) of the nodes whose features reach `targets` in at most `hops` message
        steps: nearer hops first and, within a hop, the larger |totals[id]| first (the walk's
        order without totals), at most `graph_nodes_max`."""
        return self.around.kept(self.around.reach(edge_index, targets, hops), totals)

    # -- the batch API --------------------------------------------------------------------------
    def table(self, model, batch, targets, method: str = "grad_x_input", steps=None) -> list:
        """[{target_value, attribution}] per sample of `batch`, `targets` one `ExplainTarget`
        per sample: `attribution` a list over `inputs.order` (row), `{per_step, per_feature}`
        (window; `per_feature` over `inputs.order`, a nominal summed over its steps) or
        `{self, hops}` (graph; at graph level `self` sums the graph's nodes)."""
        if method not in METHODS:
            raise ValueError(f"attribution method {method!r} is not one of {list(METHODS)}")
        path = Path(model, self.layout.graphed)
        now = path.run(batch, torch.ones(1))
        plain = self._plain(model, batch)
        now.check(plain, 0)
        if len(targets) != int(plain.shape[0]):
            raise ValueError(f"{len(targets)} target(s) for the batch's {int(plain.shape[0])} "
                             "sample(s): one per sample")
        run = now
        if method == "integrated_gradients":
            steps = int(self.limits["ig_steps_default"]) if steps is None else steps
            self._check_steps(steps)
            self._bound(steps, batch)
            run = path.run(batch, midpoints(steps, batch.get(NUMERIC)))
        values = self._values(targets, plain, batch)
        return [{"target_value": v, "attribution": e}
                for v, e in zip(values, self._entries(run, batch, targets))]

    # -- the path -------------------------------------------------------------------------------
    def _check_steps(self, steps) -> None:
        top = int(self.limits["ig_steps_max"])
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= top:
            raise ValueError(f"integrated gradients take 1 to {top} steps, not {steps!r}")

    def _bound(self, steps: int, batch) -> None:
        """`steps` × nodes ≤ `ig_batch_max` on a graph (inspect-contract §7)."""
        top, nodes = int(self.limits["ig_batch_max"]), Layout.rows(batch)
        if self.layout.graphed and steps * nodes > top:
            raise AttributionError(f"integrated gradients at {steps} steps over {nodes} "
                                   f"node(s) interpolate {steps * nodes} nodes, above "
                                   f"ig_batch_max ({top})")

    def _alone(self, batch, focus) -> tuple:
        """(the batch the explained sample needs, how its shares and edge gradients go back
        onto the explained batch's rows and edges)."""
        rows = Layout.rows(batch)
        if focus.shape != "graph":
            return _sample(batch, focus.index), lambda shares, grad: (
                shares.placed((focus.index,), rows), grad)
        if focus.level == "node":
            target = torch.as_tensor([focus.target], dtype=torch.long,
                                     device=batch["edge_index"].device)
            return {**batch, "target_index": target}, _unplaced
        one, keep = _one_graph(batch, focus.nodes)
        return one, lambda shares, grad: (shares.placed(focus.nodes, rows),
                                          edges_back(grad, keep))

    def _shares(self, run, batch, grads) -> Shares:
        """The shares of `batch`'s rows from a run's gradients (one copy's rows)."""
        embedded = [(site, run.base(site.stage), grads.get(site.stage)) for site in self.sites
                    if site.stage in run.leaves]
        return Shares.of(batch.get(NUMERIC), grads.get(NUMERIC), batch.get("x_cat"), embedded)

    def _ends(self, model, path, sample, target) -> tuple:
        """((s(x₀), s(x)), the gradients at x): the path's two ends, the one at x held to the
        network's own forward."""
        ends = path.run(sample, torch.tensor(_ENDS), weights=self.layout.graphed)
        ends.check(self._plain(model, sample), 1)
        s = self.targets.scalar(target, ends.head, ends.scoring)
        half = int(s.shape[0]) // 2
        grads = ends.gradients(s[half:].sum(), copy=1)
        return (float(s[:half].detach().sum()), float(s[half:].detach().sum())), grads

    @staticmethod
    def _plain(model, batch):
        """The network's own head on `batch` (the Model port's `head`: no gradient)."""
        return model.head(batch)

    # -- the payloads ---------------------------------------------------------------------------
    def _payload(self, method, target, shares, focus, edges) -> dict:
        if focus.shape == "row":
            body = self.layout.row(shares, focus.index)
        elif focus.shape == "window":
            body = self.layout.window(shares, focus.index, Reducer(self.limits), MATRIX_MAX)
        else:
            body = self.around.payload(shares, focus, edges)
        total = body.pop("total")
        return {"method": method, "unit": target.unit, **body, "total": total,
                "completeness": None}

    def _values(self, targets, plain, batch) -> list:
        """Each sample's target value on the network's own head."""
        with torch.no_grad():
            per = {t: self.targets.scalar(t, plain, batch) for t in dict.fromkeys(targets)}
        return [num(per[t][i]) for i, t in enumerate(targets)]

    def _entries(self, run, batch, targets) -> list:
        """Each sample's table entry from one run of the batch's copies."""
        n = len(targets)
        per = {t: self.targets.scalar(t, run.head, run.scoring) for t in dict.fromkeys(targets)}
        rows = [per[t][i::n] for i, t in enumerate(targets)]      # sample i's rows of each copy
        if self.layout.graphed and not self.layout.graph_level:
            return [self.around.node_entry(
                self._shares(run, batch, run.gradients(r.sum(), retain=True)),
                self.layout.target_of(batch, i), batch["edge_index"])
                for i, r in enumerate(rows)]
        shares = self._shares(run, batch, run.gradients(torch.stack(rows).sum()))
        return [self.layout.entry(shares, batch, i) for i in range(n)]
