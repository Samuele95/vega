"""
inspection.graph — the stage graph: every block of the served network as a typed node, the edges
the data takes between them, each node's shapes and parameter payload, and the dense chain
(inspect-contract §3.2, §5.3; build plan R-B1, R-B2, R-B5, R-B6, R-F7).

`StageGraph.build(model, spec, example, output=None)` reads the Predictor's own network once,
when the inspector opens, through its Model port — `anatomy()` (the stages, ids, kinds,
shapes, readout, identity, mode, centre, forward) and `param_count()` — and the blocks'
inspection surface; the network has no tracing method of its own (R-B5):

1. **alignment** — `len(spec.blocks) == len(anatomy.stages) == len(ids) == len(kinds)`, else
   `the network is not the spec's`; a network in training mode is refused (the inspector never
   switches a served model);
2. **placement** (`wiring.Wiring`) — the input nodes, block `i` as node `<id>` (`<id>@<i>` when
   top-level ids collide) at path `stages.<i>` with the edges of its stage kind, the output;
3. **shapes** (`wiring.ShapeProbe`) — one no-grad forward of the example batch: the batch, node
   or graph axis is dropped and named by `per` (`sample`, `node`, `graph`); a top-level
   `out_shape` must equal `anatomy.shapes[i]`. Axes: `["unit"]` for a vector, `["step",
   "channel"]` for a sequence (`"token"` after a tokenizer), `["output"]` for the head and the
   output, `["feature"]` (`["step", "feature"]`) for the inputs;
4. **params** — `count` / `trainable` sum the node module's parameters (a group's include its
   children's) and the top-level counts must sum to `model.param_count()`; each reader's payload
   registers its tensors in node order, and under the svdd objective the persisted centre is
   the buffer `<head id>.centre`; a regression-family head's fitted target affine (A37) the
   buffers `<head id>.target_shift` and `<head id>.target_scale`;
5. **code** — `{file, class, anchor}` of the module's class under the package root, or null
   (a torch module, a synthetic node).

`chain()` is the dense chain of §5.3: a row sample whose top-level blocks are embedding,
concat, dense, activation, dropout and head only, each dense or head a `W·x + b` layer that
chains width to width from the concat width, each with at most one activation (its own or one
block before the next layer); a rank-consistent ordinal head is not such a layer. `to_dict`
gives the body's `nodes`, `edges`, `chain` and `tensors`; `view.network_view` the whole body.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import registry
from .codec import TensorCodec, num
from .readers import GraphError, Site
from .view import Inputs, task_section
from .wiring import Build, ShapeProbe, Wiring

__all__ = ["GraphError", "StageEdge", "StageGraph", "StageNode"]

CHAIN_TYPES = frozenset({"embedding", "concat", "dense", "activation", "dropout", "head"})
MEASURED = "measured"


@dataclass(frozen=True, eq=False)
class StageNode:
    """One node of `network.nodes` (inspect-contract §5.3); `stage` is its top-level index."""

    id: str
    block: str | None
    type: str
    view: str
    parent: str | None
    children: tuple
    path: str
    role: str | None
    call: str
    in_shape: tuple | None
    out_shape: tuple
    axes: tuple
    per: str
    config: Mapping
    params: Mapping
    code: Mapping | None
    stage: int | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "block": self.block, "type": self.type, "view": self.view,
                "parent": self.parent, "children": list(self.children), "role": self.role,
                "call": self.call, "path": self.path,
                "in_shape": None if self.in_shape is None else list(self.in_shape),
                "out_shape": list(self.out_shape), "axes": list(self.axes), "per": self.per,
                "config": copy.deepcopy(dict(self.config)),
                "params": copy.deepcopy(dict(self.params)),
                "formula": registry.reader(self.type).FORMULA,
                "code": None if self.code is None else dict(self.code)}


@dataclass(frozen=True)
class StageEdge:
    source: str
    target: str
    kind: str          # data | skip | embed | branch | merge | edges | readout

    def to_dict(self) -> dict:
        return {"from": self.source, "to": self.target, "kind": self.kind}


def _package_root(anatomy) -> tuple:
    """(the directory holding the network's top-level package, that package's name): code
    files are named relative to the directory, and only files of the package are named. The
    anatomy is a value of the package's own `domain`, so its module names the package."""
    name = type(anatomy).__module__.split(".")[0]
    top = sys.modules.get(name)
    where = getattr(top, "__file__", None)
    if where:
        return Path(where).resolve().parent.parent, name
    paths = list(getattr(top, "__path__", ()) or ())
    return (Path(paths[0]).resolve().parent if paths else None), name


def _anatomy(model):
    """The network's anatomy, read through its Model port: a staged network in eval mode."""
    anatomy = model.anatomy()
    if anatomy is None:
        raise GraphError("the model's anatomy() is None: it is not a staged network (a "
                         "baseline), and the inspector reads the bundle's Network")
    if anatomy.training:
        raise GraphError("the inspector reads a network in eval mode, and this one is "
                         "training; it never switches a served model")
    return anatomy


def _source(cls):
    """The file a class is defined in: its module's, else its methods' (a custom block's
    module is loaded from its file, outside `sys.modules`)."""
    where = getattr(sys.modules.get(cls.__module__), "__file__", None)
    if where is None:
        fn = next((v for v in vars(cls).values() if hasattr(v, "__code__")), None)
        where = getattr(getattr(fn, "__code__", None), "co_filename", None)
    return where


def _code(module, package) -> dict | None:
    root, name = package
    if module is None or root is None:
        return None
    cls = type(module)
    where = _source(cls)
    try:
        rel = Path(where).resolve().relative_to(root) if where else None
    except ValueError:
        rel = None
    if rel is None or rel.parts[0] != name:
        return None
    return {"file": rel.as_posix(), "class": cls.__name__, "anchor": f"class {cls.__name__}("}


class _Finish:
    """Placed drafts and measured shapes -> StageNodes, in graph order."""

    def __init__(self, wiring: Wiring, recorded: Mapping, package: tuple):
        self.b, self.w, self.recorded, self.package = wiring.b, wiring, recorded, package
        self.codec, self.shapes, self.children = TensorCodec(), {}, {}
        for node, d in self.b.drafts.items():
            if d.place.parent is not None:
                self.children.setdefault(d.place.parent, []).append(node)
        calls = [(d.stage, d.call) for d in self.b.drafts.values() if d.stage is not None]
        self.gpool = min((s for s, c in calls if c == "gpool"), default=None)
        self.tokens = min((s for s, c in calls if c == "tokens"), default=None)

    def nodes(self) -> tuple:
        return tuple(self._node(n, d) for n, d in self.b.drafts.items())

    def shape(self, node) -> tuple:
        """(in, out) of a node: fixed, measured, or taken from `shape_from`."""
        if node not in self.shapes:
            d = self.b.drafts[node]
            fixed = self.b.fixed.get(node, (MEASURED, MEASURED))
            seen = self.recorded.get(node)
            if d.shape_from is not None:
                out = self.shape(d.shape_from)[1]
                seen = (out if d.type == "output" else None, out)
            if MEASURED in fixed and (seen is None or seen[1] is None):
                raise GraphError(f"node {node} ({d.type}) did not run on the example batch; "
                                 "its shapes cannot be read")
            self.shapes[node] = tuple(seen[k] if fixed[k] == MEASURED else fixed[k]
                                      for k in (0, 1))
        return self.shapes[node]

    def _node(self, node, d) -> StageNode:
        in_shape, out_shape = self.shape(node)
        reader = registry.reader(d.type)
        labels = self.w.labels.get(node)
        site = Site(self.codec, node, in_shape, out_shape,
                    None if labels is None else tuple(labels), d.config,
                    tuple(self.children.get(node, ())))
        params = list(d.module.parameters()) if d.module is not None else []
        counts = {"count": int(sum(p.numel() for p in params)),
                  "trainable": int(sum(p.numel() for p in params if p.requires_grad))}
        return StageNode(node, d.place.block, d.type, reader.VIEW, d.place.parent, site.children,
                         d.place.path, d.place.role, d.call, in_shape, out_shape,
                         self._axes(node, d, out_shape), self._per(node, d), dict(d.config),
                         {**counts, **reader.params(d.module, site)},
                         _code(d.module, self.package), d.stage)

    def _axes(self, node, d, out) -> tuple:
        if d.type == "input":
            return ("feature",) if len(out) == 1 else ("step", "feature")[:len(out)]
        if d.type in ("head", "output") and len(out) == 1:
            return ("output",)
        if len(out) <= 1:
            return ("unit",) * len(out)
        tokens = self.tokens is not None and d.stage is not None and d.stage >= self.tokens
        return ("token" if tokens else "step",) * (len(out) - 1) + ("channel",)

    def _per(self, node, d) -> str:
        if self.w.i.shape != "graph":
            return "sample"
        pooled = self.gpool is not None and (d.type == "output" or (
            d.stage is not None and d.stage >= self.gpool))
        return "graph" if node == "input.edges" or pooled else "node"


def _chain_layer(node, module, width):
    lin = getattr(module, "linear", module)
    out = int(node.out_shape[0])
    if tuple(int(d) for d in lin.weight.shape) != (out, int(width)):
        return None
    act = node.params["activation"]
    return {"node": node.id, "in": int(width), "out": out,
            "activation": None if act == "none" else act, "dropout": 0.0,
            "weight": node.params["weight"], "bias": node.params["bias"],
            "params": int(node.params["count"])}


def _fold(layers, node) -> bool:
    """An embedding, concat, activation or dropout folded into the chain, or False."""
    if node.type in ("embedding", "concat"):
        return not layers
    if node.type == "dropout":
        if layers:
            kept = (1.0 - layers[-1]["dropout"]) * (1.0 - float(node.params["p"]))
            layers[-1]["dropout"] = num(1.0 - kept)
        return True
    name = node.params["name"]
    if name == "none":
        return True
    if not layers or layers[-1]["activation"] is not None:
        return False
    layers[-1]["activation"] = name
    return True


class StageGraph:
    """The stage graph of one network (see the module docstring); built once, then read."""

    def __init__(self, parts: Mapping):
        self._p = dict(parts)
        self._by_id = {n.id: n for n in self._p["nodes"]}

    @staticmethod
    def build(model, spec, example, output=None) -> "StageGraph":
        """`example` is `(batch, SampleIndex)` or the batch; `output` is MANIFEST.output."""
        data = spec.to_dict() if callable(getattr(spec, "to_dict", None)) else dict(spec)
        blocks = [dict(b) for b in data.get("blocks") or ()]
        anatomy = _anatomy(model)
        if not len(blocks) == len(anatomy.stages):
            raise GraphError(f"the network is not the spec's: {len(blocks)} blocks, "
                             f"{len(anatomy.stages)} stages")
        inputs, task = Inputs.of(data), task_section(data, model, output)
        wiring = Wiring(Build(), inputs)
        wiring.place(anatomy, blocks, task)
        batch = example[0] if isinstance(example, (tuple, list)) else example
        finish = _Finish(wiring, ShapeProbe(wiring.b.drafts).run(anatomy, batch),
                         _package_root(anatomy))
        graph = StageGraph({"nodes": finish.nodes(), "codec": finish.codec, "task": task,
                            "edges": tuple(StageEdge(*e) for e in wiring.b.edges),
                            "modules": {n: d.module for n, d in wiring.b.drafts.items()},
                            "spec": data, "inputs": inputs, "joined": tuple(wiring.joined),
                            "weighted": None if inputs.shape != "graph" else
                            batch.get("edge_weight") is not None,
                            "params": tuple(int(v) for v in model.param_count()),
                            "spec_digest": str(anatomy.spec_digest)})
        graph._check(anatomy)
        graph._p["chain"] = graph._dense_chain()
        head = graph.top_level()[-1].id
        if anatomy.centre is not None and anatomy.centre_set:     # R-B6: set once, by the loss
            graph.codec.add(anatomy.centre, f"{head}.centre", head, "buffer", ("unit",))
        module = graph.module(head)                         # A37: fitted once, by the Trainer
        if bool(getattr(module, "target_fitted", False)):
            for name in ("target_shift", "target_scale"):
                graph.codec.add(getattr(module, name), f"{head}.{name}", head, "buffer", ("unit",))
        return graph

    def _check(self, anatomy) -> None:
        """The shapes and counts are the network's own, and every edge joins two nodes."""
        top = self.top_level()
        for node, shape in zip(top, anatomy.shapes):
            if tuple(node.out_shape) != tuple(int(d) for d in shape):
                raise GraphError(f"node {node.id}: out_shape {list(node.out_shape)}, the "
                                 f"network's shape {list(shape)}")
        counted = (sum(n.params["count"] for n in top), sum(n.params["trainable"] for n in top))
        if counted != self.params:
            raise GraphError(f"the nodes count {counted} parameters (total, trainable), the "
                             f"network {self.params}")
        loose = [e for e in self.edges if e.source not in self._by_id or
                 e.target not in self._by_id]
        if loose:
            raise GraphError(f"edge {loose[0].to_dict()} joins a node the graph does not hold")

    def _dense_chain(self):
        top = self.top_level()
        if self.inputs.shape != "row" or not top or top[-1].type != "head" or \
                any(n.type not in CHAIN_TYPES for n in top):
            return None
        width, layers = len(self._p["joined"]), []
        for node in top:
            if node.type in ("dense", "head"):
                layer = _chain_layer(node, self.module(node.id), width)
                if layer is None:
                    return None
                layers.append(layer)
                width = layer["out"]
            elif not _fold(layers, node):
                return None
        return {"eligible": True, "linear_reading": True, "inputs": list(self._p["joined"]),
                "layers": layers}

    # ---------------------------------------------------------------- reading

    @property
    def nodes(self) -> tuple:
        return self._p["nodes"]

    @property
    def edges(self) -> tuple:
        return self._p["edges"]

    @property
    def codec(self) -> TensorCodec:
        return self._p["codec"]

    @property
    def task(self) -> dict:
        return copy.deepcopy(self._p["task"])

    @property
    def spec(self) -> dict:
        return copy.deepcopy(self._p["spec"])

    @property
    def inputs(self) -> Inputs:
        return self._p["inputs"]

    @property
    def weighted(self):
        """Whether the example's graph carries edge weights (None off a graph sample)."""
        return self._p["weighted"]

    @property
    def params(self) -> tuple:
        """(total, trainable) parameters of the network."""
        return self._p["params"]

    @property
    def spec_digest(self) -> str:
        return self._p["spec_digest"]

    def node(self, node_id: str) -> StageNode:
        if node_id not in self._by_id:
            raise KeyError(f"no node {node_id!r} in this stage graph")
        return self._by_id[node_id]

    def module(self, node_id: str):
        """The module a node reads (None for a synthetic node)."""
        self.node(node_id)
        return self._p["modules"][node_id]

    def top_level(self) -> tuple:
        """The spec's blocks, in order."""
        return tuple(n for n in self.nodes if n.parent is None and n.stage is not None)

    def chain(self):
        return copy.deepcopy(self._p.get("chain"))

    def to_dict(self, weights: str = "full") -> dict:
        """`nodes`, `edges`, `chain` and `tensors` of the body, under a weights mode."""
        return {"nodes": [n.to_dict() for n in self.nodes],
                "edges": [e.to_dict() for e in self.edges], "chain": self.chain(),
                "tensors": self.codec.tensors(weights)}
