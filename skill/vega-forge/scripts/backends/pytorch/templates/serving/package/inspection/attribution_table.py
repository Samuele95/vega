"""
inspection.attribution_table — `vega/attribution-table@1`: how many samples of a prepared request
attribute at once, the library path `/vega-show` builds `vega/attribution@2` from
(inspect-contract §5.5; D3 §9.3; build plan R-D9, R-D15).

`AttributionTable(model, graph, targets, attributor).build(prepared, served, selectors, method,
steps)` reads the batch `Predictor.prepare()` returned and the Predictor's answer frame for the
same rows (`explanation.Served`):

- **targets** — each answered sample takes the selector of the request row it answers
  (`selectors`: one per row of the request, a mapping such as `{"class": <the true class>}` or
  None, or no list at all for every sample's default), resolved by `TaskTargets.choose` on the
  network's own head. Each selector is checked first as `QueryRules.parse` checks a request's
  (its form: a name is a string, an index an integer that is no bool, so `{"threshold": 0.5}`,
  `True` or `"0"` is refused, never read as an index), then against the task: the `QueryError`
  that names the selector (`target.<selector>`) and the row (`targets[i]`).
  Every target of one table is of one unit (the table's `unit`), so the rows' totals compare.
- **chunks** — at most `CHUNK` samples at a time, each chunk one run (gradient × input) or one
  run of its interpolants (integrated gradients) and one gradient, no node payloads
  (`Attributor.table`): rows and windows are sliced by sample; at graph node level the chunk
  keeps the whole graph and a slice of `target_index`; at graph level it keeps its graphs, their
  nodes renumbered in order and their edges.
- **samples** — `{row, prediction, target_value, attribution}` per answered sample, in sample
  order: `row` the request row it answers, `prediction` the served frame's row for it,
  `attribution` a list over `inputs.order` (row), `{per_step, per_feature}` (window) or `{self,
  hops}` (graph).
"""
from __future__ import annotations

from typing import Mapping

import torch

from .explanation import answered_rows, on_device
from .query import QueryError, target_form
from .targets import KINDS

__all__ = ["CHUNK", "SCHEMA", "AttributionTable"]

SCHEMA = "vega/attribution-table@1"
CHUNK = 256                     # samples per forward and gradient


def _rows_chunk(batch, a: int, b: int, n: int) -> dict:
    """Samples a…b−1 of a row or window batch: every per-sample tensor sliced."""
    return {key: value[a:b] if torch.is_tensor(value) and value.dim() and
            int(value.shape[0]) == n else value for key, value in batch.items()}


def _graphs_chunk(batch, a: int, b: int) -> dict:
    """Graphs a…b−1 of a graph-level batch: their nodes in order, renumbered, their edges."""
    groups = batch["graph_index"]
    keep = (groups >= a) & (groups < b)
    nodes = torch.nonzero(keep).flatten()
    position = torch.cumsum(keep.long(), 0) - 1
    edges = torch.nonzero(keep[batch["edge_index"][0]]).flatten()
    out = {key: batch[key].index_select(0, nodes) for key in ("x_num", "x_cat")
           if batch.get(key) is not None}
    out.update({"edge_index": position[batch["edge_index"][:, edges]],
                "graph_index": groups.index_select(0, nodes) - a, "num_graphs": int(b - a)})
    if batch.get("edge_weight") is not None:
        out["edge_weight"] = batch["edge_weight"].index_select(0, edges)
    return out


class AttributionTable:
    """The attribution tables of one served network (see the module docstring)."""

    def __init__(self, model, graph, targets, attributor):
        self.model, self.graph, self.targets, self.attributor = model, graph, targets, attributor

    def build(self, prepared, served, selectors, method: str, steps) -> dict:
        """The `vega/attribution-table@1` body of every answered sample of `prepared`."""
        batch = on_device(prepared.batch, self.model)
        pairs = answered_rows(prepared)
        if len(pairs) != int(prepared.n_samples):
            raise ValueError(f"{int(prepared.n_samples)} sample(s) answer {len(pairs)} request "
                             "row(s): every sample of a prepared request answers a row")
        chosen = self._chosen(batch, pairs, selectors, len(served.frame))
        entries = []
        for a in range(0, len(pairs), CHUNK):
            b = min(len(pairs), a + CHUNK)
            entries += self.attributor.table(self.model, self._chunk(batch, a, b, len(pairs)),
                                             chosen[a:b], method, steps)
        return {"schema": SCHEMA, "method": method, "unit": self._unit(chosen),
                "shape": self.graph.inputs.shape,
                "inputs": list(self.attributor.layout.order),
                "targets": [{"kind": t.kind, "index": t.index, "name": t.name, "sign": t.sign,
                             "why": t.why, "unit": t.unit} for t in chosen],
                "samples": [{"row": int(row), "prediction": served.prediction(row),
                             "target_value": entry["target_value"],
                             "attribution": entry["attribution"]}
                            for (_, row), entry in zip(pairs, entries)]}

    def _chosen(self, batch, pairs, selectors, n_rows: int) -> list:
        """Each sample's target: its row's selector, else the task's default."""
        if selectors is not None and len(selectors) != n_rows:
            raise ValueError(f"{len(selectors)} target selector(s) for the {n_rows} request "
                             "row(s): one per row (None: the task's default)")
        selectors = None if selectors is None else [self._checked(i, s)
                                                    for i, s in enumerate(selectors)]
        if not pairs:
            return []
        head = self.model.head(batch)                  # the Model port's: no gradient
        rows = head.reshape(head.shape[0], -1)
        unfit = [row for k, row in pairs if not bool(torch.isfinite(rows[k]).all())]
        if unfit:
            raise ValueError(f"request row(s) {unfit[:8]} are answered with numbers that are not "
                             "finite (their warnings say why: a window's lags lack history), so "
                             "no attribution of them can be published; send their history")
        return [self.targets.choose(head, batch, None if selectors is None else selectors[row],
                                    k) for k, row in pairs]

    def _checked(self, i: int, selector):
        """Row `i`'s selector as `QueryRules.parse` reads a request's target, admitted by the
        task (None: the task's default)."""
        if selector is None:
            return None
        if not isinstance(selector, Mapping):
            raise ValueError(f"target selector {i} is a {type(selector).__name__}; a selector "
                             "is a mapping such as {\"class\": <name>}, or None")
        try:
            selector = target_form(selector)
            self.targets.validate(selector)
        except QueryError as exc:
            raise QueryError(exc.field, f"targets[{i}]: {exc}") from exc
        return selector

    def _unit(self, chosen) -> str:
        units = sorted({t.unit for t in chosen})
        if len(units) > 1:
            raise ValueError(f"the targets are in {', '.join(units)}: one table explains "
                             "targets of one unit, so its rows' totals compare")
        if units:
            return units[0]
        return KINDS[self.targets.default_kind()][1]

    def _chunk(self, batch, a: int, b: int, n: int) -> dict:
        """Samples a…b−1 of the batch, as the network reads them."""
        if a == 0 and b == n:
            return batch
        inputs = self.graph.inputs
        if inputs.shape != "graph":
            return _rows_chunk(batch, a, b, n)
        if inputs.graph_level:
            return _graphs_chunk(batch, a, b)
        targets = batch.get("target_index")
        if targets is None:
            targets = torch.arange(n, device=batch["edge_index"].device)
        return {**batch, "target_index": targets[a:b]}
