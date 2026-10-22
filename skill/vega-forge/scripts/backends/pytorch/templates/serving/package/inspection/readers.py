"""
inspection.readers — how one block type becomes stage-graph nodes, its config and its parameter
payload (inspect-contract §3, §5.3; build plan R-B1, R-B3, R-B6, R-F7).

A reader is stateless, one instance per type of the view table (`registry.READERS`). It reads a
block through its inspection surface (`coverage.json` `inspection.surface`) only, and never
writes to the module it reads.

- `expand(ctx, block, module, place)` adds the block's node — a composite's children too,
  through `ctx.expand` — and returns the node the block's output leaves from (its exit). A
  group node is the entry of its children.
- `config(block, module)` is the block's params **as the module holds them** (dropout `p` from
  `.drop.p`); `checked` refuses a spec param that says otherwise (`GraphError`: the network is
  not the spec's). Values of different JSON kinds are not compared (`padding: "same"` resolves
  to a number).
- `params(module, site)` is the §5.3 payload: tensors registered through `site.tensor` as
  `<node id>.<name>` in node order, derived vectors inline — at most `json_list_max` numbers,
  else the 64 largest entries and a summary.
- `probe(cap, node)`, `recompute(cap, node)` and `exact(cap, node)` read one explanation's
  captured pass (`capture.Capture`): the node payload's `extra` (§3.1's `probe` column), the
  node's check — max |Δ| between its output and its recomputation, None where §3.1 names none
  — and the tensors a `detail: "exact"` explanation adds beyond `in`, `out` and `grad_out`.
  The reader hands them to its type's probe (`PROBE`, from `probes*.py`), which reads the
  capture only and never writes to the network or to a captured tensor.

Derived numbers are float64 numpy without a BLAS reduction (`np.add.reduce`, no `norm`, no
matrix product; the helpers live in `derived.py`): one network always gives the same bytes.
Here: the base and the unit, table, norm, pointwise and reshaping readers; `readers_group.py`
the composites, the latent and opaque blocks and the synthetic nodes; `readers_seq.py` the
sequence readers; `readers_graph.py` the graph readers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from . import probes as P
from .codec import num, nums, summary
from .derived import (ACT_NAMES, PCA_MAX, SAFE_NAME, act_name, arr, fro, norms, pca2,
                      table_entry, tensor_name)

__all__ = ["ACT_NAMES", "CORE", "PCA_MAX", "SAFE_NAME", "GraphError", "Place", "Site",
           "act_name", "arr", "block_id", "fro", "norms", "pca2", "table_entry", "tensor_name"]

TOP_K = 64                  # a long derived vector keeps its 64 largest entries
_UNSET = object()


class GraphError(ValueError):
    """The network cannot be described: a block type without a reader, a network that is not
    the spec's, a shape or a count that does not add up. The inspector does not open."""


@dataclass(frozen=True)
class Place:
    """Where a node sits: its id, its module path under the network, its parent, its role and
    the spec block (top level) it belongs to."""

    node: str
    path: str
    parent: str | None = None
    role: str | None = None
    block: str | None = None

    def child(self, suffix: str, attr: str, role: str | None = None) -> "Place":
        """A child node `<id>.<suffix>` at module path `<path>.<attr>`."""
        return Place(f"{self.node}.{suffix}", f"{self.path}.{attr}", self.node, role,
                     self.block)


@dataclass(frozen=True)
class Site:
    """What a reader writes one node's payload with: the codec, the node's id and shapes, the
    labels of its input channels (when the graph knows them), its config and its children."""

    codec: Any
    node: str
    in_shape: tuple | None
    out_shape: tuple
    labels: tuple | None = None
    config: Mapping = field(default_factory=dict)
    children: tuple = ()

    def tensor(self, t, name: str, role: str, axes) -> str | None:
        """Register `t` as `<node>.<name>`; None stays None (a block without that tensor)."""
        if t is None:
            return None
        return self.codec.add(t, f"{self.node}.{name}", self.node, role, axes)

    def vector(self, values):
        """Inline numbers, or the 64 largest entries and a summary past `json_list_max`."""
        a = np.asarray(values, dtype=np.float64).reshape(-1)
        if a.size <= self.codec.limits["json_list_max"]:
            return nums(a)
        order = np.argsort(-np.abs(a), kind="stable")[:TOP_K]
        return {"top": [[int(i), num(a[i])] for i in order], "summary": summary(a)}


def _json_kind(v) -> str:
    kinds = ((bool, "bool"), ((int, float), "number"), ((list, tuple), "list"), (Mapping, "map"))
    return next((k for t, k in kinds if isinstance(v, t)), type(v).__name__)


def _agree(said, held) -> bool:
    """Whether a spec value and the module's agree; values of different kinds are not
    compared, and a missing activation is `none`."""
    said, held = ("none" if said is None else said), ("none" if held is None else held)
    kind = _json_kind(said)
    if kind != _json_kind(held):
        return True
    if kind == "number":
        return float(said) == float(held)
    if kind == "list":
        return len(said) == len(held) and all(map(_agree, said, held))
    if kind == "map":
        said, held = ({str(k): v for k, v in m.items()} for m in (said, held))
        return set(said) == set(held) and all(_agree(said[k], held[k]) for k in said)
    return said == held


def block_id(block) -> str:
    block = block or {}
    return str(block.get("id") or block.get("type"))


class _Reader:
    """One block type: its view-table row and how it is read (see the module docstring)."""

    TYPE = ""
    VIEW = ""
    CALL = "tensor"
    FORMULA = ""
    PAYLOAD = ()
    ALIASES = MappingProxyType({})       # config key -> an older spec param name
    PROBE = P.NO_PROBE                   # what an explanation reads of the node (probes*.py)

    def expand(self, ctx, block, module, place: Place) -> str:
        ctx.node(self.TYPE, place, module, self.checked(block, module))
        return place.node

    def config(self, block, module) -> dict:
        return {}

    def checked(self, block, module) -> dict:
        """`config`, refused when a spec param says otherwise."""
        held = self.config(block, module)
        params = dict((block or {}).get("params") or {})
        for key, value in held.items():
            said = params.get(key, params.get(self.ALIASES.get(key, key), _UNSET))
            if said is not _UNSET and not _agree(said, value):
                raise GraphError(f"block {block_id(block)} ({self.TYPE}): the spec says {key} "
                                 f"= {said!r}, the module holds {value!r}; the network is not "
                                 "the spec's")
        return held

    def params(self, module, site: Site) -> dict:
        """By default the payload keys the config holds (a block of plain attributes)."""
        held = self.config(None, module)
        return {k: held[k] for k in self.PAYLOAD if k in held}

    def probe(self, cap, node) -> dict:
        """The node payload's `extra`: its type's probe keys, null where nothing applies."""
        return self.PROBE.probe(cap, node)

    def recompute(self, cap, node):
        """max |Δ| of the node's check on the explained sample, or None (no check)."""
        return self.PROBE.recompute(cap, node)

    def exact(self, cap, node) -> dict:
        """{key: (tensor, axes)}: the extra tensors of a `detail: "exact"` node."""
        return self.PROBE.exact(cap, node)


def _linear(module):
    return getattr(module, "linear", module)


def _units(module, site: Site) -> dict:
    lin = _linear(module)
    w = arr(lin.weight)
    act = act_name(module.act) if hasattr(module, "act") else \
        str(site.config.get("activation", "none"))
    return {"weight": site.tensor(lin.weight, "weight", "weight", ("out", "in")),
            "bias": site.tensor(lin.bias, "bias", "bias", ("out",)),
            "row_norms": site.vector(norms(w, 1)),
            "col_norms": site.vector(norms(w, 0)),
            "activation": act}


class _Dense(_Reader):
    TYPE, VIEW, FORMULA = "dense", "units", "z = W·x + b ; a = act(z)"
    PROBE = P.UNITS
    PAYLOAD = ("weight", "bias", "row_norms", "col_norms", "activation")

    def config(self, block, module):
        lin = _linear(module)
        return {"units": int(lin.out_features), "bias": lin.bias is not None,
                "activation": act_name(getattr(module, "act", None))}

    def params(self, module, site):
        return _units(module, site)


def _cumulative(module) -> bool:
    return callable(getattr(module, "thresholds", None))


class _Head(_Dense):
    """A head; under an ordinal task the rank-consistent head: `weight` the shared score
    (1 × in), `bias` null, `thresholds` the K−1 ordered offsets it adds."""

    TYPE = "head"
    FORMULA = "z = W·x + b ; ordinal: z_j = w·x + θ_j with θ_0 ≥ θ_1 ≥ …"
    PAYLOAD = _Dense.PAYLOAD + ("thresholds",)

    def config(self, block, module):
        if _cumulative(module):
            return {"out": int(module.gaps.numel()) + 1, "activation": "none",
                    "cumulative": True}
        held = super().config(block, module)
        return {"out": held.pop("units"), **held, "cumulative": False}

    def params(self, module, site):
        payload = _units(module, site)
        payload["thresholds"] = None
        if _cumulative(module):
            site.tensor(module.first, "first", "bias", ("threshold",))
            site.tensor(module.gaps, "gaps", "bias", ("gap",))
            with torch.no_grad():
                payload["thresholds"] = nums(module.thresholds())
        return payload


class _Embedding(_Reader):
    TYPE, VIEW, CALL, FORMULA = "embedding", "table", "emb", "e_c = E_c[id_c]"
    PROBE = P.EMBEDDING
    PAYLOAD = ("tables",)

    def config(self, block, module):
        cols = [str(c) for c in module.columns]
        return {"columns": cols,
                "dims": {c: int(e.embedding_dim) for c, e in zip(cols, module.embeddings)},
                "cardinalities": {c: int(e.num_embeddings)
                                  for c, e in zip(cols, module.embeddings)}}

    def params(self, module, site):
        return {"tables": [table_entry(site, c, e.weight, k) for k, (c, e)
                           in enumerate(zip(module.columns, module.embeddings))]}


class _Concat(_Reader):
    TYPE, VIEW, CALL, FORMULA = "concat", "join", "concat", "h = [x_num ; e_1 ; … ; e_m]"
    PROBE = P.CONCAT
    PAYLOAD = ("parts",)

    def params(self, module, site):
        numeric = int(site.in_shape[-1]) if site.in_shape else 0
        parts = [{"stream": "numeric", "width": numeric}]
        if int(site.out_shape[-1]) > numeric:
            parts.append({"stream": "nominal", "width": int(site.out_shape[-1]) - numeric})
        return {"parts": parts}


class _BatchNorm(_Reader):
    TYPE, VIEW, FORMULA = "batchnorm", "affine", "y = γ·(x − μ)/√(σ² + ε) + β"
    PROBE = P.BATCHNORM
    PAYLOAD = ("gamma", "beta", "running_mean", "running_var", "eps", "scale", "shift")

    def config(self, block, module):
        momentum = module.bn.momentum
        return {"eps": float(module.bn.eps),
                "momentum": None if momentum is None else float(momentum)}

    def params(self, module, site):
        bn = module.bn
        mean, var = arr(bn.running_mean), arr(bn.running_var)
        gamma = arr(bn.weight) if bn.weight is not None else np.ones_like(mean)
        beta = arr(bn.bias) if bn.bias is not None else np.zeros_like(mean)
        scale = gamma / np.sqrt(var + float(bn.eps))
        return {"gamma": site.tensor(bn.weight, "gamma", "weight", ("unit",)),
                "beta": site.tensor(bn.bias, "beta", "bias", ("unit",)),
                "running_mean": site.tensor(bn.running_mean, "running_mean", "buffer",
                                            ("unit",)),
                "running_var": site.tensor(bn.running_var, "running_var", "buffer", ("unit",)),
                "eps": num(bn.eps), "scale": site.vector(scale),
                "shift": site.vector(beta - scale * mean)}


class _LayerNorm(_Reader):
    TYPE, VIEW, FORMULA = "layernorm", "affine", "y = γ·(x − mean x)/√(var x + ε) + β"
    PROBE = P.LAYERNORM
    PAYLOAD = ("gamma", "beta", "eps")

    def config(self, block, module):
        return {"eps": float(module.ln.eps)}

    def params(self, module, site):
        ln = module.ln
        return {"gamma": site.tensor(ln.weight, "gamma", "weight", ("unit",)),
                "beta": site.tensor(ln.bias, "beta", "bias", ("unit",)), "eps": num(ln.eps)}


class _Dropout(_Reader):
    TYPE, VIEW, FORMULA = "dropout", "pointwise", "y = x (inactive at inference)"
    PROBE = P.IDENTITY
    PAYLOAD = ("p", "active_at_inference")

    def config(self, block, module):
        return {"p": float(module.drop.p)}

    def params(self, module, site):
        return {"p": num(module.drop.p), "active_at_inference": False}


class _Activation(_Reader):
    TYPE, VIEW, FORMULA = "activation", "pointwise", "a = act(z)"
    PROBE = P.ACTIVATION
    PAYLOAD = ("name",)

    def config(self, block, module):
        return {"name": act_name(module.act)}


class _Flatten(_Reader):
    TYPE, VIEW, FORMULA = "flatten", "reshape", "(L,d) → (L·d)"
    PROBE = P.RESHAPE


class _Reshape(_Reader):
    TYPE, VIEW, FORMULA, PAYLOAD = "reshape", "reshape", "(n) → shape", ("shape",)
    PROBE = P.RESHAPE

    def config(self, block, module):
        return {"shape": [int(s) for s in module.shape]}


class _Pool(_Reader):
    TYPE, VIEW, FORMULA, PAYLOAD = "pool", "pool", "mean_t x_t | max_t x_t | x_0", ("kind",)
    PROBE = P.POOL

    def config(self, block, module):
        return {"kind": str(module.kind)}


CORE = (_Dense(), _Head(), _Embedding(), _Concat(), _BatchNorm(), _LayerNorm(), _Dropout(),
        _Activation(), _Flatten(), _Reshape(), _Pool())
