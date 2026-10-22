"""
inspection.wiring — how the stage graph is placed before it is measured (inspect-contract §3.2
steps 1–5; build plan R-B1, R-B2): the context readers expand into (`Build`), the placement of
the inputs, of every top-level block with the incoming edges its stage kind implies and of the
output (`Wiring`), and the one forward pass that measures every node module (`ShapeProbe`).

The incoming edges per stage kind (the network's `_STEP` table, read by name):

| kind | the network calls | incoming edges |
|---|---|---|
| `emb` | `e = m(x_cat)` (h untouched) | from `input.cat`, kind `embed` |
| `concat` | `h = m(x_num, e)` | from `input.num` and the last `emb` exit |
| `tokens` | `h = m(x_num, x_cat)` | from `input.num` and `input.cat` |
| `tensor` | `h = m(h)` | from the running exit |
| `gpool` | `h = m(h, graph_index, num_graphs)` | from the running exit |
| `graph` | `h = m(h, edge_index, edge_weight)` | the running exit; `edges` from the edge source |
| `gdrop` | `edge_index, edge_weight = m(…)` | `edges` from the edge source, which it becomes |

The edge source is `input.edges`, then the last `gdrop` node.

`Wiring` also tracks the labels of the channels each top-level node reads — the feature names
into the first stages, `<column>·e<j>` for an embedding's dims after `concat`, kept through
the blocks that keep channels (dropout, activation, the norms, graph_pool) — which a reader
labels its columns with (a recurrent block's layer-0 input columns, the tokenizer's tokens).

`ShapeProbe` is the measure of §3.2 step 6: one no-grad forward of the example batch (the
anatomy's `forward`, the network's own, read through the Model port) inside a
`Capture(mode="shapes")`, active on the building thread only, with a hook (`capture.hook`) on
every node module; each records the per-sample shapes of its first call's first tensor in and
out. Every hook is removed before `run` returns, so no hook outlives the build (the
inspector's permanent hooks are `capture.HookSet`'s, attached once the graph exists).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from . import registry
from .capture import Capture, CaptureError, hook
from .codec import TENSOR_ID
from .readers import GraphError, Place

__all__ = ["Build", "Draft", "ShapeProbe", "Wiring"]

RESERVED = frozenset({"input.num", "input.cat", "input.edges", "output"})
KEEPS_CHANNELS = frozenset({"dropout", "activation", "batchnorm", "layernorm", "graph_pool"})


@dataclass(frozen=True, eq=False)
class Draft:
    """A placed node before it is measured: type, place, module, config, the node whose
    output shape it takes (a synthetic node), its top-level stage, its call."""

    type: str
    place: Place
    module: Any
    config: dict
    shape_from: str | None
    stage: int | None
    call: str


class Build:
    """The graph while it is placed: drafts in graph order, edges `(from, to, kind)`, and the
    shapes known without the measure: `fixed[node] = (in, out)`, either of which may read
    "measured" (the probe's value is kept)."""

    def __init__(self):
        self.drafts, self.edges, self.fixed = {}, [], {}
        self.stage, self.call = None, "io"

    def node(self, node_type, place, module, config, shape_from=None) -> str:
        if place.node in self.drafts or not TENSOR_ID.match(place.node):
            raise GraphError(f"node id {place.node!r} is taken or is not an inspection id "
                             f"({TENSOR_ID.pattern})")
        call = self.call if place.parent is None else "inner"
        self.drafts[place.node] = Draft(node_type, place, module, dict(config or {}),
                                        shape_from, self.stage, call)
        return place.node

    def expand(self, block, module, place) -> str:
        """The block's reader places it (and a composite its children); returns its exit."""
        kind = str((block or {}).get("type"))
        return registry.reader(kind).expand(self, block, module, place)

    def edge(self, source, target, kind) -> None:
        self.edges.append((source, target, kind))


class Wiring:
    """Places the inputs, the top-level blocks and the output (see the module docstring)."""

    STEP = MappingProxyType({"emb": "_emb", "concat": "_concat", "tokens": "_tokens",
                             "tensor": "_tensor", "gpool": "_tensor", "graph": "_graph",
                             "gdrop": "_gdrop"})

    def __init__(self, build: Build, inputs):
        self.b, self.i = build, inputs
        self.run, self.emb, self.edges_src = "input.num", None, "input.edges"
        self.labels = {"input.num": inputs.feature_names, "input.cat": inputs.nominal,
                       "input.edges": ()}
        self.running = self.joined = inputs.feature_names
        self.emb_labels = ()

    def place(self, anatomy, blocks, task) -> None:
        """Every node and edge, from the spec's blocks and the network's `anatomy`."""
        self._inputs()
        ids = [str(i) for i in anatomy.ids]
        for i, (block, module) in enumerate(zip(blocks, anatomy.stages)):
            node = f"{ids[i]}@{i}" if ids.count(ids[i]) > 1 or ids[i] in RESERVED else ids[i]
            kind = _kind_of(anatomy.kinds[i], block, ids[i])
            self.b.stage, self.b.call = i, kind
            exit_ = self.b.expand(block, module, Place(node, f"stages.{i}", None, None, ids[i]))
            if kind == "gdrop":         # the stage leaves the running tensor it did not touch
                self.b.fixed[node] = (None, tuple(int(d) for d in anatomy.shapes[i]))
            getattr(self, self.STEP[kind])(node, exit_, (block, module))
        self.b.stage, self.b.call = None, "io"
        self.b.node("output", Place("output", "output"), None,
                    {"link": task["link"], "outputs": task["outputs"]}, self.run)
        self.b.edge(self.run, "output", "readout")

    def _inputs(self) -> None:
        streams = [("input.num", "x_num", "numeric", self.i.num_shape)]
        if self.i.nominal:
            streams.append(("input.cat", "x_cat", "nominal", self.i.cat_shape))
        if self.i.shape == "graph":
            streams.append(("input.edges", "edge_index", "edges", ()))
        for node, key, stream, shape in streams:
            self.b.node("input", Place(node, key), None, {"stream": stream})
            self.b.fixed[node] = (None, tuple(shape))

    def _emb(self, node, exit_, parts) -> None:
        module = parts[1]
        self.b.edge("input.cat", node, "embed")
        self.emb, self.labels[node] = exit_, self.i.nominal
        self.emb_labels = tuple(f"{c}·e{j}" for c, e in zip(module.columns, module.embeddings)
                                for j in range(int(e.embedding_dim)))

    def _concat(self, node, exit_, parts) -> None:
        self.b.edge("input.num", node, "data")
        if self.emb is not None:
            self.b.edge(self.emb, node, "data")
        self.b.fixed[node] = (tuple(self.i.num_shape), "measured")
        self.labels[node] = self.i.feature_names
        self.running = self.joined = self.i.feature_names + self.emb_labels
        self.run = exit_

    def _tokens(self, node, exit_, parts) -> None:
        self.b.edge("input.num", node, "data")
        if self.i.nominal:
            self.b.edge("input.cat", node, "data")
        self.b.fixed[node] = (tuple(self.i.num_shape), "measured")
        self.labels[node] = self.i.feature_names + self.i.nominal
        self.running, self.run = None, exit_

    def _tensor(self, node, exit_, parts) -> None:
        self.b.edge(self.run, node, "data")
        self.labels[node] = self.running
        if parts[0].get("type") not in KEEPS_CHANNELS:
            self.running = None
        self.run = exit_

    def _graph(self, node, exit_, parts) -> None:
        self.b.edge(self.run, node, "data")
        self.b.edge(self.edges_src, node, "edges")
        self.labels[node], self.running, self.run = self.running, None, exit_

    def _gdrop(self, node, exit_, parts) -> None:
        self.b.edge(self.edges_src, node, "edges")
        self.edges_src = exit_


def _kind_of(kind, block, block_name) -> str:
    """The network's stage kind of a block, which must be the view table's call."""
    kind = str(getattr(kind, "value", kind))
    wanted = registry.reader(str(block.get("type"))).CALL
    if kind != wanted:
        raise GraphError(f"block {block_name}: the network calls it as {kind!r}, the view table "
                         f"as {wanted!r}")
    return kind


class ShapeProbe:
    """The measured (in, out) shape of every placed node module (see the module docstring)."""

    def __init__(self, drafts):
        self._modules = [(n, d.module) for n, d in drafts.items() if d.module is not None]

    def run(self, anatomy, batch) -> dict:
        cap = Capture(None, "shapes")
        handles = [m.register_forward_hook(hook(n)) for n, m in self._modules]
        try:
            with cap, torch.no_grad():
                anatomy.forward(batch)
        except CaptureError as exc:
            raise GraphError(f"the network cannot be measured: {exc}") from exc
        except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise GraphError(f"the example batch does not run through the network: {exc}") \
                from exc
        finally:
            for handle in handles:
                handle.remove()
        return cap.shapes()
