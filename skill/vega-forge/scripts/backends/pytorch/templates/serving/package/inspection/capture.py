"""
inspection.capture — what one explanation reads from the served network, without writing to it
(inspect-contract §1 R1–R6, §3.2 step 6; build plan R-B4, R-B5, R-F5).

- **`HookSet.attach(model, graph)`** runs once, when the inspector opens, on the stage modules
  the Model port's `anatomy()` names: one hook per stage-graph node with a module (key: the
  node id), one on the `.linear` of a dense or head node (`<id>:pre`: the pre-activation z), one
  on the `.rnn` of a recurrent node (`<id>:seq`: its own sequence output); `run` keeps the head
  the network returns (key `output`, gathered at the targets at node level). The hooks stay
  for the life of the service; an explanation registers none. A hook reads the calling
  thread's active `Capture` — a thread-local, the module's only state — and returns at once
  without one, so a `/v1/predict` thread never captures; it returns None, never altering an
  output.
- **`Capture(graph, mode)`** is one pass. Mode `shapes` is the stage graph's measure (the
  graph is being built, so it is None): each key keeps the per-sample shapes of its first
  call's first tensor in and out (`shapes()`). Mode `values` is one explanation:
  `run(model, batch, index)` feeds the anatomy's `forward` a copy of the batch whose leaves
  require a gradient — `x_num` and, on a graph, the edge weights (ones when the batch has none,
  which every conv reads as the unweighted graph) — under `torch.enable_grad()`, the capture
  active on the calling thread only, and returns the head; `focus` is then sample `index`
  (`focus.Focus`). Each key keeps its first call of the pass: the first positional tensor
  (`input`), the other positional arguments (`extras`: the embedding a concat joins, the
  `x_cat` a tokenizer reads, a graph call's `edge_index` and `edge_weight`), the output (its
  first element when a tuple; `output(key + "#state")` is the rest: an LSTM's (h_n, c_n), an
  edge dropout's weights). `differentiate(s)` takes ∂s/∂t for the leaves and every captured
  tensor in one `torch.autograd.grad(..., allow_unused=True)` call, writing no parameter's
  `.grad`; the inspector never calls `.backward()`, `.train()` or `.zero_grad()`.
  `suspended()` pauses the capture while a probe re-calls a submodule (R5); `memo` keeps what
  a probe computed once for this pass.
- **Node values** (`_Reading`). `value(id)` / `value_in(id)` / `grad(id)` read a node's
  tensors over the whole batch: a module node's from its hook; a synthetic node's by the
  stage graph — `input.num` and `input.cat` are the fed batch's (`input.edges` has no
  per-sample value), `output` is the network's return, a `residual_add` is what the next
  repeat reads (the residual block's output after its last repeat), a `parallel_merge` is the
  parallel block's output. `feeds(id)` gives the tensors on a node's incoming edges — a
  group's entry edge carries the group's input. `take(t, node, side)` slices the explained
  sample out of any of them.

A Capture belongs to one thread and one pass; the network, the stage graph and the hooks are
shared, read-only, by every thread.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import NamedTuple

import torch
from torch import nn

from .focus import SAMPLE, Focus
from .reduce import Reducer

__all__ = ["MODEL", "MODES", "PRE", "SEQ", "STATE", "Capture", "CaptureError", "HookSet",
           "active", "hook"]

_ACTIVE = threading.local()       # the capture the calling thread's hooks write to
MODEL, PRE, SEQ, STATE = "output", ":pre", ":seq", "#state"
MODES = ("values", "shapes")
UNITS = frozenset({"dense", "head"})
RECURRENT = frozenset({"rnn", "gru", "lstm"})


class CaptureError(RuntimeError):
    """An explanation cannot be read: a capture already active on the thread, a network in
    training mode, a pass run or differentiated twice, a node the pass never reached, a
    recomputation of another shape than the output it checks."""


def active():
    """The capture active on the calling thread, or None."""
    return getattr(_ACTIVE, "capture", None)


def hook(key: str):
    """The forward hook of one key: it hands the call to the calling thread's capture."""
    def forward_hook(module, args, output):
        cap = getattr(_ACTIVE, "capture", None)
        if cap is not None and not cap.paused:
            cap.put(key, args, output)
    return forward_hook


def _first_shape(values):
    """The per-sample shape (leading axis dropped) of the first tensor among `values`."""
    for v in values:
        if torch.is_tensor(v):
            return tuple(int(d) for d in v.shape[1:])
        if isinstance(v, (tuple, list)):
            found = _first_shape(v)
            if found is not None:
                return found
    return None


def _inner(node, module):
    """The submodule a node's second hook sits on (`:pre` / `:seq`), or None."""
    name = "linear" if node.type in UNITS else "rnn" if node.type in RECURRENT else None
    inner = getattr(module, name, None) if name else None
    return inner if isinstance(inner, nn.Module) and inner is not module else None


class HookSet:
    """The permanent forward hooks of one served network (R1; see the module docstring)."""

    def __init__(self, handles, keys):
        self._handles, self._keys = list(handles), tuple(keys)

    @staticmethod
    def sites(model, graph) -> tuple:
        """(key, module) of every hook the inspector keeps on `model`'s stages for `graph`."""
        owned = {id(m) for stage in model.anatomy().stages for m in stage.modules()}
        sites = []
        for node in graph.nodes:
            module = graph.module(node.id)
            if module is None:
                continue
            if id(module) not in owned:
                raise CaptureError(f"node {node.id}: its module is not one of this network's; "
                                   "the stage graph was built over another network")
            sites.append((node.id, module))
            inner = _inner(node, module)
            if inner is not None:
                sites.append((node.id + (PRE if node.type in UNITS else SEQ), inner))
        return tuple(sites)

    @staticmethod
    def attach(model, graph) -> "HookSet":
        """Hook `model` for `graph` — once, when the inspector opens."""
        if model.anatomy().training:
            raise CaptureError("the inspector hooks a network in eval mode, and this one is "
                               "training; it never switches a served model")
        sites = HookSet.sites(model, graph)
        return HookSet([m.register_forward_hook(hook(key)) for key, m in sites],
                       [key for key, _ in sites])

    @property
    def keys(self) -> tuple:
        return self._keys

    @property
    def count(self) -> int:
        return len(self._handles)

    def detach(self) -> None:
        """Remove every hook (the service is closing, or a test is done with the model)."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


class _Seen(NamedTuple):
    input: object
    extras: tuple
    output: object
    state: tuple


class _Reading:
    """A captured pass read by stage-graph node (see the module docstring, "Node values");
    `Capture` holds the pass (`graph`, `focus`, `_batch`, `seen`, `has`, `grad_of`)."""

    def module(self, node_id: str):
        """The module a node reads (None for a synthetic node)."""
        return self.graph.module(node_id)

    def pre(self, node_id: str) -> tuple:
        """(x, z) of a dense or head node's linear map: its `:pre` call, or the node's own call
        when the node is the linear itself (a decoder layer, a residual projection)."""
        key = node_id + PRE
        if self.has(key):
            rec = self.seen(key)
            return rec.input, rec.output
        return self.value_in(node_id), self.value(node_id)

    def seq(self, node_id: str):
        """A recurrent node's own sequence output (the `:seq` call)."""
        return self.seen(node_id + SEQ).output

    def value(self, node_id: str):
        """A node's output over the whole batch (None for the edge stream)."""
        node = self.graph.node(node_id)
        if self.module(node_id) is not None:
            return self.seen(node_id).output
        if node.type == "input":
            return None if node.id == "input.edges" else self._batch.get(node.path)
        if node.type == "output":
            return self.seen(MODEL).output
        if node.type == "parallel_merge":
            return self.value(node.parent)
        if node.type == "residual_add":
            return self._after_add(node)
        raise CaptureError(f"node {node.id}: no value rule for a synthetic {node.type!r} node")

    def value_in(self, node_id: str):
        """A module node's first input over the whole batch; None for a synthetic node."""
        if self.module(node_id) is None:
            return None
        return self.seen(node_id).input

    def grad(self, node_id: str):
        """∂s/∂(the node's output) over the whole batch, or None."""
        return self.grad_of(self.value(node_id))

    def _edges(self) -> tuple:
        """({node: [(source, kind)]} incoming, {node: [target]} outgoing), in edge order."""
        if self._into is None:
            into, out = {}, {}
            for e in self.graph.edges:
                into.setdefault(e.target, []).append((e.source, e.kind))
                out.setdefault(e.source, []).append(e.target)
            self._into = (into, out)
        return self._into

    def _after_add(self, node):
        nxt = [t for t in self._edges()[1].get(node.id, ())
               if self.graph.node(t).parent == node.parent and self.module(t) is not None]
        return self.value_in(nxt[0]) if nxt else self.value(node.parent)

    def feeds(self, node_id: str) -> list:
        """[(source, kind, tensor)] on a node's incoming edges; an edge out of the node's own
        group (its entry) carries the group's input."""
        parent = self.graph.node(node_id).parent
        return [(src, kind, self.value_in(src) if src == parent else self.value(src))
                for src, kind in self._edges()[0].get(node_id, ())]

    def per(self, node, side: str = "out") -> str:
        """What one row of a node's tensor is: the head the network returns is per sample; a
        graph pool reads nodes and writes graphs."""
        if node.type == "output":
            return SAMPLE
        if side == "in" and node.type == "graph_pool":
            return "node"
        return node.per

    def take(self, t, node, side: str = "out"):
        """The explained sample's slice of `t`, one of `node`'s tensors (`side` in | out)."""
        return self.focus.take(t, self.per(node, side))


class Capture(_Reading):
    """One pass over the served network (see the module docstring)."""

    def __init__(self, graph, mode: str = "values"):
        if mode not in MODES:
            raise CaptureError(f"capture mode {mode!r} is not one of {list(MODES)}")
        if mode == "values" and graph is None:
            raise CaptureError("an explanation's capture reads a stage graph")
        self.graph, self.mode, self.focus = graph, mode, None
        self.reducer = Reducer(graph.codec.limits) if graph is not None else Reducer()
        self._seen, self._memo, self._paused = {}, {}, False
        self._anatomy = self._batch = self._grads = self._into = None

    def __enter__(self) -> "Capture":
        if active() is not None:
            raise CaptureError("a capture is already active on this thread; one explanation "
                               "runs one pass at a time")
        _ACTIVE.capture = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ACTIVE.capture = None

    @property
    def paused(self) -> bool:
        return self._paused

    @contextmanager
    def suspended(self):
        """The capture paused while a probe re-calls a submodule (R5)."""
        was, self._paused = self._paused, True
        try:
            yield self
        finally:
            self._paused = was

    def put(self, key: str, args, output) -> None:
        """Keep one hooked call — the first of the pass under `key`."""
        if key in self._seen:
            return
        args = tuple(args)
        if self.mode == "shapes":
            self._seen[key] = (_first_shape(args), _first_shape((output,)))
            return
        first = args[0] if args and torch.is_tensor(args[0]) else None
        if isinstance(output, (tuple, list)):
            out, state = (output[0] if output else None), tuple(output[1:])
        else:
            out, state = output, ()
        self._seen[key] = _Seen(first, args[1:], out, state)

    def shapes(self) -> dict:
        """{key: (in, out)} per-sample shapes of a `shapes` pass."""
        if self.mode != "shapes":
            raise CaptureError("shapes() reads a capture in mode 'shapes'")
        return dict(self._seen)

    def run(self, model, batch, index: int = 0):
        """The network's head on `batch`, every hooked call kept; `focus` is sample `index`."""
        if self.mode != "values":
            raise CaptureError("run() explains a pass in mode 'values'")
        anatomy = model.anatomy()
        if anatomy.training:
            raise CaptureError("the inspector reads a network in eval mode, and this one is "
                               "training; it never switches a served model")
        if self._anatomy is not None:
            raise CaptureError("a Capture reads one pass; open another for the next one")
        fed = self._leaves(batch)
        self.focus = Focus.of(self.graph, fed, index)
        with self, torch.enable_grad():
            head = anatomy.forward(fed)
            hooked = bool(self._seen)
            self.put(MODEL, (fed,), head)
        if not hooked:
            raise CaptureError("the network carries no inspection hooks: HookSet.attach(model, "
                               "graph) runs once, when the inspector opens")
        self._anatomy, self._batch = anatomy, fed
        return head

    def _leaves(self, batch) -> dict:
        fed = dict(batch)
        x = fed.get("x_num")
        if torch.is_tensor(x) and x.is_floating_point():
            fed["x_num"] = x.detach().clone().requires_grad_(True)
        edges = fed.get("edge_index")
        if self.graph.inputs.shape == "graph" and torch.is_tensor(edges):
            w = fed.get("edge_weight")
            dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float32
            base = w.detach().clone() if torch.is_tensor(w) else \
                torch.ones(int(edges.shape[1]), dtype=dtype, device=edges.device)
            fed["edge_weight"] = base.requires_grad_(True)
        return fed

    def differentiate(self, s) -> None:
        """∂s/∂t for the leaves and every captured tensor, parameters untouched (R2)."""
        if self._anatomy is None:
            raise CaptureError("differentiate a pass after it ran")
        if self._grads is not None:
            raise CaptureError("a pass is differentiated once")
        if not torch.is_tensor(s) or s.numel() != 1 or not s.requires_grad:
            raise CaptureError("the explained target is one number the pass computed")
        wanted = self._differentiable()
        grads = torch.autograd.grad(s.reshape(()), wanted, allow_unused=True) if wanted else ()
        self._grads = {id(t): (t, g) for t, g in zip(wanted, grads)}

    def _differentiable(self) -> tuple:
        pool = [self._batch.get("x_num"), self._batch.get("edge_weight")]
        for rec in self._seen.values():
            pool += [rec.input, rec.output, *rec.extras]
        found = {}
        for t in pool:
            if torch.is_tensor(t) and t.is_floating_point() and t.requires_grad:
                found.setdefault(id(t), t)
        return tuple(found.values())

    @property
    def anatomy(self):
        """The anatomy of the network this pass ran (its readout, its centre)."""
        return self._anatomy

    @property
    def batch(self) -> dict:
        """The batch the network ran (the leaves in place of `x_num` and the edge weights)."""
        return self._batch

    def has(self, key: str) -> bool:
        return key in self._seen

    def seen(self, key: str) -> _Seen:
        rec = self._seen.get(key)
        if rec is None or self.mode != "values":
            raise CaptureError(f"{key!r} did not run in this pass; it cannot be explained")
        return rec

    def input(self, key: str):
        """The first positional tensor of a key's call."""
        return self.seen(key).input

    def output(self, key: str):
        """A key's output — its first element when a tuple — or, as `<key>#state`, the rest."""
        if key.endswith(STATE):
            return self.seen(key[:-len(STATE)]).state
        return self.seen(key).output

    def extras(self, key: str) -> tuple:
        """The other positional arguments of a key's call."""
        return self.seen(key).extras

    def grad_of(self, t):
        """∂s/∂t for a tensor of this pass (None: not differentiated, or s does not read t)."""
        if t is None or self._grads is None:
            return None
        hit = self._grads.get(id(t))
        return hit[1] if hit is not None and hit[0] is t else None

    def memo(self, key, compute):
        """`compute()` once per key for this pass (a probe's shared work)."""
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]
