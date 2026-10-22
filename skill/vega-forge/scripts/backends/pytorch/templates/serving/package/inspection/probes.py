"""
inspection.probes — what one node did with the explained sample, and the check that the
published numbers reproduce it (inspect-contract §3.1 `probe`, `exact` and `recompute`; build
plan R-B4, R-F5).

A probe is stateless, one instance per node type, bound to the type's reader
(`readers._Reader.PROBE`). It reads a captured pass (`capture.Capture`), never the network's
state, and writes nothing — neither to the network nor to a captured tensor:

- `probe(cap, node)` is the node payload's `extra`, keyed exactly by `KEYS` (the view table's
  `probe` column); a key with nothing to say for this node is null. Tensors travel in the
  reduced forms of `reduce.Reducer`.
- `recompute(cap, node)` is the node's check (`CHECKS`; the view table's `recompute` column):
  max |Δ| between the node's output on the explained sample and its recomputation from the
  node's inputs and the module's parameters — None where no check applies.
- `exact(cap, node)` is `{key: (tensor, axes)}` for the keys of `EXACT`, the tensors a
  `detail: "exact"` explanation carries beyond `in`, `out` and `grad_out`.

Recomputations run in float64 on detached copies (`f64`) — a module method a probe calls
(an activation, the ordinal thresholds, the attention) runs under `torch.no_grad()`, so no
autograd graph reaches a parameter — and call no hooked module; a probe that re-calls a
submodule does it inside `cap.suspended()` (R5). Here:
the base, the helpers and the probes of the unit, table, join, norm, pointwise, reshaping and
pooling blocks; `probes_group.py` the latent block and the synthetic nodes; `probes_seq.py`
the sequence blocks; `probes_graph.py` the graph blocks.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .capture import CaptureError

__all__ = ["ACTIVATION", "BATCHNORM", "CONCAT", "EMBEDDING", "IDENTITY", "LAYERNORM", "NO_PROBE",
           "POOL", "RESHAPE", "UNITS", "Probe", "axes_of", "f64", "gap", "node_rows", "norm_last",
           "reduced", "sample_io"]


def f64(t):
    """A float64 copy of a tensor, detached from every graph (None stays None)."""
    return None if t is None else t.detach().to(dtype=torch.float64, copy=True)


def gap(a, b) -> float:
    """max |a − b| over two tensors of one shape (0 for empty ones)."""
    a, b = f64(a), f64(b)
    if tuple(a.shape) != tuple(b.shape):
        raise CaptureError(f"a recomputation of shape {list(a.shape)} checks an output of "
                           f"shape {list(b.shape)}")
    return float((a - b).abs().max()) if a.numel() else 0.0


def norm_last(t):
    """L2 norms over the last axis (a scalar for a vector, one per step for a sequence)."""
    return torch.sqrt((t * t).sum(dim=-1))


def axes_of(t, preferred, nodes: bool = False) -> tuple:
    """Axis names for a sample tensor: the node's own (`node` before them for the node rows of
    an explained graph), else `axis<i>`."""
    preferred = (("node",) if nodes else ()) + tuple(preferred or ())
    if t.dim() == len(preferred):
        return preferred
    return tuple(f"axis{i}" for i in range(t.dim()))


def node_rows(cap, node, side: str = "out") -> bool:
    """Whether a node's sample tensor is the node rows of an explained graph."""
    return cap.focus.level == "graph" and cap.per(node, side) == "node"


def sample_io(cap, node) -> tuple:
    """(in, out) of the explained sample through a node, as float64 copies."""
    return (f64(cap.take(cap.value_in(node.id), node, "in")),
            f64(cap.take(cap.value(node.id), node)))


def reduced(cap, t):
    """A tensor's reduced form (None stays None)."""
    return None if t is None else cap.reducer.reduce(t)


class Probe:
    """A node type with no probe and no check (a group, an opaque block); the base of all."""

    KEYS = ()
    EXACT = ()
    CHECKS = False

    def probe(self, cap, node) -> dict:
        return dict.fromkeys(self.KEYS)

    def recompute(self, cap, node):
        return None

    def exact(self, cap, node) -> dict:
        return {}


NO_PROBE = Probe()


class _Units(Probe):
    """`dense`, `head`: z = W·x + b (the `:pre` call), the node output, ∂s/∂z. The check is
    W·in + b = pre, and pre to the output through the activation (an ordinal head adds its
    thresholds, a regression-family head ends on its target affine · scale + shift, A37); a
    bare linear node (a decoder layer, a projection) outputs z itself."""

    KEYS = EXACT = ("pre", "post", "grad_pre")
    CHECKS = True

    def _parts(self, cap, node) -> tuple:
        x, z = cap.pre(node.id)
        y = cap.value(node.id)
        return x, z, y, cap.grad_of(z)

    def probe(self, cap, node):
        _, z, y, g = self._parts(cap, node)
        return {"pre": reduced(cap, f64(cap.take(z, node))),
                "post": reduced(cap, f64(cap.take(y, node))),
                "grad_pre": reduced(cap, f64(cap.take(g, node)))}

    def recompute(self, cap, node):
        x, z, y, _ = self._parts(cap, node)
        module = cap.module(node.id)
        lin = getattr(module, "linear", module)
        x, z, y = f64(cap.take(x, node, "in")), f64(cap.take(z, node)), f64(cap.take(y, node))
        with torch.no_grad():
            affine = F.linear(x, f64(lin.weight), f64(lin.bias))
            post = z + f64(module.thresholds()) if callable(getattr(module, "thresholds", None)) \
                else module.act(z) if isinstance(getattr(module, "act", None), nn.Module) else z
            if getattr(module, "affine", False):
                post = post * f64(module.target_scale) + f64(module.target_shift)
        return max(gap(affine, z), gap(post, y))

    def exact(self, cap, node):
        _, z, y, g = self._parts(cap, node)
        out = {}
        for key, t in (("pre", z), ("post", y), ("grad_pre", g)):
            s = cap.take(t, node)
            if s is not None:
                out[key] = (s, axes_of(s, node.axes, node_rows(cap, node)))
        return out


class _Embedding(Probe):
    """Per nominal column: its `id` in the sample, its `vector`, ∂s/∂vector (`grad`) and the
    `attribution` Σ vector ⊙ grad (per step on a window); the check is E_c[id] = vector."""

    KEYS = ("id", "vector", "grad", "attribution")
    CHECKS = True

    def _columns(self, cap, node):
        module, ids = cap.module(node.id), cap.take(cap.value_in(node.id), node, "in")
        e = cap.value(node.id)
        vec, grad = f64(cap.take(e, node)), f64(cap.take(cap.grad_of(e), node))
        off = 0
        for c, (column, emb) in enumerate(zip(module.columns, module.embeddings)):
            width = int(emb.embedding_dim)
            yield (str(column), ids[..., c], emb, vec[..., off:off + width],
                   None if grad is None else grad[..., off:off + width])
            off += width

    def probe(self, cap, node):
        out = {key: {} for key in self.KEYS}
        for column, ids, _, vec, grad in self._columns(cap, node):
            out["id"][column] = ids.tolist()
            out["vector"][column] = reduced(cap, vec)
            out["grad"][column] = reduced(cap, grad)
            out["attribution"][column] = None if grad is None else \
                reduced(cap, (vec * grad).sum(dim=-1))
        return out

    def recompute(self, cap, node):
        return max((gap(f64(emb.weight)[ids.long()], vec)
                    for _, ids, emb, vec, _ in self._columns(cap, node)), default=0.0)


class _Concat(Probe):
    """The check: the numeric stream and the embeddings joined = the output."""

    CHECKS = True

    def recompute(self, cap, node):
        joined = [f64(cap.take(t, node, "in")) for t in
                  (cap.value_in(node.id), *cap.extras(node.id)[:1]) if t is not None]
        out = f64(cap.take(cap.value(node.id), node))
        return gap(torch.cat(joined, dim=-1) if len(joined) > 1 else joined[0], out)


class _BatchNorm(Probe):
    """The check: scale·in + shift (the running statistics of an eval pass) = the output."""

    CHECKS = True

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        bn = cap.module(node.id).bn
        mean, var = f64(bn.running_mean), f64(bn.running_var)
        gamma = f64(bn.weight) if bn.weight is not None else torch.ones_like(mean)
        beta = f64(bn.bias) if bn.bias is not None else torch.zeros_like(mean)
        scale = gamma / torch.sqrt(var + float(bn.eps))
        return gap(x * scale + (beta - scale * mean), y)


class _LayerNorm(Probe):
    """The sample's `mean` and `std` = √(var + ε) (per step on a sequence); the check is the
    formula."""

    KEYS = ("mean", "std")
    CHECKS = True

    def _stats(self, cap, node) -> tuple:
        x, y = sample_io(cap, node)
        ln = cap.module(node.id).ln
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(((x - mean) ** 2).mean(dim=-1, keepdim=True) + float(ln.eps))
        return x, y, ln, mean, std

    def probe(self, cap, node):
        _, _, _, mean, std = self._stats(cap, node)
        return {"mean": reduced(cap, mean.squeeze(-1)), "std": reduced(cap, std.squeeze(-1))}

    def recompute(self, cap, node):
        x, y, ln, mean, std = self._stats(cap, node)
        z = (x - mean) / std
        if ln.weight is not None:
            z = z * f64(ln.weight)
        if ln.bias is not None:
            z = z + f64(ln.bias)
        return gap(z, y)


class _Identity(Probe):
    """`dropout` (inactive at inference): the output is the input."""

    CHECKS = True

    def recompute(self, cap, node):
        return gap(*sample_io(cap, node))


class _Activation(Probe):
    """The check: act(in) = the output."""

    CHECKS = True

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        with torch.no_grad():
            return gap(cap.module(node.id).act(x), y)


class _Reshape(Probe):
    """`flatten`, `reshape`: the input's values, in order, are the output's."""

    CHECKS = True

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        return gap(x.reshape(y.shape), y) if x.numel() == y.numel() else \
            float(abs(x.numel() - y.numel()))


def _steps(node) -> bool:
    """Whether a node's per-sample input is a sequence (steps × channels)."""
    return node.in_shape is not None and len(node.in_shape) == 2


class _Pool(Probe):
    """`winners` — per input step, how many channels it won (max) — and `cls`; the check is
    the pooling formula (a vector, or a graph's node rows, passes through)."""

    KEYS = ("winners", "cls")
    CHECKS = True

    def probe(self, cap, node):
        kind = str(cap.module(node.id).kind)
        winners = None
        if kind == "max" and _steps(node):
            x, _ = sample_io(cap, node)
            winners = torch.bincount(x.argmax(dim=0), minlength=x.shape[0]).tolist()
        return {"winners": winners, "cls": kind == "cls" and _steps(node)}

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        if not _steps(node):
            return gap(x, y)
        kind = str(cap.module(node.id).kind)
        read = x.mean(dim=0) if kind == "mean" else x.max(dim=0).values if kind == "max" \
            else x[0]
        return gap(read, y)


UNITS = _Units()
EMBEDDING = _Embedding()
CONCAT = _Concat()
BATCHNORM = _BatchNorm()
LAYERNORM = _LayerNorm()
IDENTITY = _Identity()
ACTIVATION = _Activation()
RESHAPE = _Reshape()
POOL = _Pool()
