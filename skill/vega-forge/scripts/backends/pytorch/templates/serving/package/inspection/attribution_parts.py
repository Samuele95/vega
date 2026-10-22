"""
inspection.attribution_parts — the numbers an attribution is made of and their layout per
sample shape (inspect-contract §5.4, §5.5; D3 §8.3; build plan I05), for
`attribution.Attributor`.

- **`Site`** — one nominal column the network reads (`nominal_sites(graph)`): its index in
  `x_cat`, the stage that embeds it, and where its embedding sits in that stage's output —
  `lo:hi` of the last axis of an `embedding` (one table per `x_cat` column, in order, as the
  block reads them), or its token of a `feature_tokenizer`, counted from the end (the nominal
  tokens close the token axis, one per nominal in contract order). `Site.of(t)` slices it
  from the stage's output, or from the path's leaf of its nominal tokens alike.
- **`Shares`** — the attribution of a batch's rows (samples, or a graph's nodes), float64 on
  the host: `x`, `grad` and `attr` = x ⊙ grad over the numeric channels (…, F); `ids`,
  `nominal` (Σ_d e_d·g_d at each column's embedding) and `nominal_grad` (‖g‖) per nominal
  column (…, C) — per step on a window. `placed(rows, n)` puts the shares of a subset of a
  batch's rows back at their rows.
- **`Layout`** — the network's inputs as the payloads name them: `order` (`inputs.order`: the
  numeric columns, the nominal columns, then the autoregressive target), `channels` (those of
  `input.num`: the numeric columns, then the autoregressive target), the window's step
  labels, the graph level and the number of graph convs; `row` and `window` are the
  explanation's payloads (§5.4) and `entry` a sample's table entry (§5.5) for the row, window
  and graph-level shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from .codec import num, nums

__all__ = ["TOP_FEATURES", "Layout", "Shares", "Site", "host64", "nominal_sites"]

TOP_FEATURES = 5            # a graph neighbour's features reported, the largest |a| first


def host64(t):
    """A detached float64 copy on the host."""
    return t.detach().to(device="cpu", dtype=torch.float64)


@dataclass(frozen=True)
class Site:
    """One nominal column the network reads (see the module docstring)."""

    column: int
    node: str
    stage: int
    lo: int = 0
    hi: int = 0
    token: int | None = None

    def of(self, t):
        """The column's embedding in `t`, a stage output (or leaf) over a batch's rows."""
        return t[:, self.token] if self.token is not None else t[..., self.lo:self.hi]


def nominal_sites(graph) -> tuple:
    """Every nominal column an `embedding` or a `feature_tokenizer` stage embeds."""
    sites = []
    for node in graph.top_level():
        module = graph.module(node.id)
        tables = list(getattr(module, "embeddings", None) or ()) if module is not None else []
        if node.call == "emb":
            lo = 0
            for i, table in enumerate(tables):
                width = int(table.embedding_dim)
                sites.append(Site(i, node.id, int(node.stage), lo, lo + width))
                lo += width
        elif node.call == "tokens":
            sites += [Site(j, node.id, int(node.stage), token=j - len(tables))
                      for j in range(len(tables))]
    return tuple(sites)


@dataclass(frozen=True, eq=False)
class Shares:
    """The attribution of a batch's rows (see the module docstring)."""

    x: torch.Tensor
    grad: torch.Tensor
    attr: torch.Tensor
    ids: torch.Tensor
    nominal: torch.Tensor
    nominal_grad: torch.Tensor

    @staticmethod
    def of(x, grad, ids, embedded) -> "Shares":
        """`x` and `grad` the numeric inputs and ∂s/∂x (None: s does not read them), `ids`
        the nominal ids, `embedded` = [(site, e, ∂s/∂e)] over the same rows."""
        lead = tuple((x if x is not None else ids).shape[:-1])
        xs = host64(x) if x is not None else torch.zeros(lead + (0,), dtype=torch.float64)
        g = host64(grad) if grad is not None else torch.zeros_like(xs)
        c = int(ids.shape[-1]) if ids is not None else 0
        dot = torch.zeros(lead + (c,), dtype=torch.float64)
        square = torch.zeros(lead + (c,), dtype=torch.float64)
        for site, e, ge in embedded:
            if e is None or ge is None:
                continue
            v, gv = host64(site.of(e)), host64(site.of(ge))
            dot[..., site.column] += (v * gv).sum(dim=-1)
            square[..., site.column] += (gv * gv).sum(dim=-1)
        kept = ids.detach().cpu().long() if ids is not None else \
            torch.zeros(lead + (0,), dtype=torch.long)
        return Shares(xs, g, xs * g, kept, dot, torch.sqrt(square))

    def placed(self, rows, n: int) -> "Shares":
        """These shares, over a subset of an n-row batch, at their `rows`; zeros elsewhere."""
        index = torch.as_tensor(list(rows), dtype=torch.long)

        def put(t):
            out = t.new_zeros((int(n), *t.shape[1:]))
            out[index] = t
            return out
        return Shares(put(self.x), put(self.grad), put(self.attr), put(self.ids),
                      put(self.nominal), put(self.nominal_grad))


@dataclass(frozen=True)
class Layout:
    """The network's inputs as the attribution payloads name them (see the module
    docstring)."""

    shape: str
    numeric: tuple
    nominal: tuple
    past: str | None
    step_labels: tuple
    graph_level: bool
    convs: int
    limits: Mapping

    @staticmethod
    def of(graph, limits) -> "Layout":
        i = graph.inputs
        convs = sum(1 for node in graph.top_level() if node.call == "graph")
        return Layout(str(i.shape), tuple(i.numeric), tuple(i.nominal), i.past,
                      tuple(i.steps() or ()), bool(i.graph_level), int(convs),
                      MappingProxyType(dict(limits)))

    @property
    def graphed(self) -> bool:
        return self.shape == "graph"

    @property
    def order(self) -> tuple:
        """`inputs.order`: the numeric columns, the nominal columns, the autoregressive
        target."""
        return self.numeric + self.nominal + ((self.past,) if self.past else ())

    @property
    def channels(self) -> tuple:
        """The channels of `input.num`: the numeric columns, then the autoregressive target."""
        return self.numeric + ((self.past,) if self.past else ())

    def ordered(self, numeric, nominal):
        """Values over `order` from the numeric channels (…, F) and the nominal columns
        (…, C)."""
        n = len(self.numeric)
        return torch.cat([numeric[..., :n], nominal.to(numeric.dtype), numeric[..., n:]],
                         dim=-1)

    @staticmethod
    def rows(batch) -> int:
        """The input rows of a batch (samples, or a graph's nodes)."""
        lead = batch.get("x_num") if batch.get("x_num") is not None else batch["x_cat"]
        return int(lead.shape[0])

    @staticmethod
    def target_of(batch, i: int) -> int:
        """The node answering sample `i` at graph node level."""
        targets = batch.get("target_index")
        return int(i) if targets is None else int(targets[int(i)])

    def labels(self, steps: int) -> list:
        if len(self.step_labels) == steps:
            return list(self.step_labels)
        return [f"t−{steps - 1 - j}" for j in range(steps - 1)] + ["t"]

    # -- the explanation's payloads (§5.4) ---------------------------------------------------
    def row(self, sh: Shares, k: int) -> dict:
        n, x = len(self.numeric), sh.x[k]
        attr = self.ordered(sh.attr[k], sh.nominal[k])
        values = [*nums(x[:n]), *(int(v) for v in sh.ids[k].tolist()), *nums(x[n:])]
        return {"shape": "row",
                "features": {"names": list(self.order), "values": values,
                             "attribution": nums(attr),
                             "grad": nums(self.ordered(sh.grad[k], sh.nominal_grad[k]))},
                "total": num(attr.sum())}

    def window(self, sh: Shares, k: int, reducer, matrix_max: int) -> dict:
        a, an = sh.attr[k], sh.nominal[k]                    # (L, F), (L, C)
        steps, width = int(a.shape[0]), int(a.shape[1])
        per_step = a.sum(dim=-1) + an.sum(dim=-1)
        nominal = None if not self.nominal else {
            "names": list(self.nominal), "values": sh.ids[k].t().tolist(),
            "attribution": nums(an.t())}
        return {"shape": "window", "steps": steps, "step_labels": self.labels(steps),
                "features": {"names": list(self.channels)},
                "matrix": nums(a) if steps * width <= matrix_max else reducer.heat(a, "sum"),
                "per_step": nums(per_step), "per_feature": nums(a.sum(dim=0)),
                "nominal": nominal, "total": num(per_step.sum())}

    # -- the table's entries (§5.5) ----------------------------------------------------------
    def entry(self, sh: Shares, batch, i: int):
        """Sample `i`'s entry: row, window, or a graph of a graph-level batch."""
        if self.shape == "row":
            return nums(self.ordered(sh.attr[i], sh.nominal[i]))
        if self.shape == "window":
            a, an = sh.attr[i], sh.nominal[i]
            return {"per_step": nums(a.sum(dim=-1) + an.sum(dim=-1)),
                    "per_feature": nums(self.ordered(a.sum(dim=0), an.sum(dim=0)))}
        members = torch.nonzero(batch["graph_index"].cpu() == int(i)).flatten()
        per = self.ordered(sh.attr, sh.nominal).index_select(0, members)
        return {"self": nums(per.sum(dim=0)), "hops": {"0": num(per.sum())}}
