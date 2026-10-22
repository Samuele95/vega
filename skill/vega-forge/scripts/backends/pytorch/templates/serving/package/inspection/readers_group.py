"""
inspection.readers_group — the readers of the composite blocks (`sequential`, `residual`,
`parallel`, `decoder`), of the latent and opaque blocks (`reparameterize`, `custom`) and of the
synthetic nodes (`input`, `output`, `residual_add`, `parallel_merge`) (inspect-contract §3.1,
§3.2 steps 3–5, §5.3; build plan R-B1, R-B9).

The nested ids are the contract's: `sequential` `<id>.<k>`; `residual` `<id>.r<rep>.<k>`, its
projection `<id>.r<rep>.proj` (a `dense`, role `projection`) and its add `<id>.r<rep>.add` fed
by the branch (`data`) and by the skip path (`skip`); `parallel` branches `<id>.b<k>` (a
composite branch nests further, `<id>.b<k>.<j>`) and the merge `<id>.merge` (`branch` edges
out of the group, `merge` edges into the merge); `decoder` layers `<id>.l<k>` (a `dense`, role
`decoder_layer`, carrying the activation that follows it). A composite's inner spec blocks
are read by their own readers, so a nested block is checked against its spec params as a
top-level one is. The synthetic readers are placed by the graph (`graph.py`), never expanded
from a spec block. Their probes are `probes_group.py`'s; a composite and `custom` have none (a
composite is read through its children, user code is opaque).
"""
from __future__ import annotations

from torch import nn

from . import probes_group as PG
from .readers import SAFE_NAME, GraphError, _Reader, act_name, block_id, fro

__all__ = ["GROUP", "inner_pairs", "series"]


def inner_pairs(block, modules) -> list:
    """The inner spec blocks of a composite, paired with the modules the composite holds."""
    inner = list(((block or {}).get("params") or {}).get("inner") or ())
    modules = list(modules)
    if len(inner) != len(modules):
        raise GraphError(f"block {block_id(block)}: the spec lists {len(inner)} inner "
                         f"block(s), the module holds {len(modules)}; the network is not the "
                         "spec's")
    return list(zip(inner, modules))


def series(ctx, entry: str, items) -> str:
    """Chain `(block, module, place)` items after `entry` with data edges; the last exit."""
    last = entry
    for block, module, place in items:
        ctx.edge(last, place.node, "data")
        last = ctx.expand(block, module, place)
    return last


class _Reparameterize(_Reader):
    TYPE, VIEW, PAYLOAD = "reparameterize", "latent", ("z",)
    PROBE = PG.REPARAMETERIZE
    FORMULA = "z = μ at inference ; KL = −½ Σ (1 + log σ² − μ² − σ²)"

    def config(self, block, module):
        return {"z": int(module.z)}


class _Custom(_Reader):
    """User code: read as an opaque module — its class and its parameters, never verified."""

    TYPE, VIEW, FORMULA = "custom", "opaque", "declared by user code (opaque)"
    PAYLOAD = ("class", "parameters", "declared_out_shape", "verified")

    def config(self, block, module):
        declared = ((block or {}).get("params") or {}).get("out_shape")
        return {"class": type(module).__name__,
                "out_shape": [int(d) for d in declared] if declared else None}

    def params(self, module, site):
        parameters = []
        for k, (name, p) in enumerate(module.named_parameters()):
            ref = f"p.{name}" if SAFE_NAME.match(name.replace(".", "_")) else f"p.@{k}"
            axes = tuple(f"d{j}" for j in range(p.dim()))
            parameters.append({"name": name, "shape": [int(d) for d in p.shape],
                               "l2": fro(p), "tensor": site.tensor(p, ref, "weight", axes)})
        return {"class": type(module).__name__, "parameters": parameters,
                "declared_out_shape": site.config.get("out_shape"), "verified": False}


class _Sequential(_Reader):
    TYPE, VIEW, FORMULA, PAYLOAD = "sequential", "group", "f_n ∘ … ∘ f_1", ("children",)

    def config(self, block, module):
        return {"blocks": len(module.inner)}

    def expand(self, ctx, block, module, place):
        ctx.node(self.TYPE, place, module, self.checked(block, module))
        return series(ctx, place.node, [(b, m, place.child(str(k), f"inner.{k}")) for k, (b, m)
                                        in enumerate(inner_pairs(block, module.inner))])

    def params(self, module, site):
        return {"children": list(site.children)}


class _Residual(_Reader):
    """`<id>.r<rep>.<k>` inner nodes, `<id>.r<rep>.proj` (a dense, role projection) when the
    width changes, `<id>.r<rep>.add` fed by the branch (data) and the skip path (skip)."""

    TYPE, VIEW, FORMULA, PAYLOAD = "residual", "group", "x ← P·x + f(x)", ("repeats",)

    def config(self, block, module):
        return {"repeat": len(module.repeats)}

    def expand(self, ctx, block, module, place):
        ctx.node(self.TYPE, place, module, self.checked(block, module))
        exit_ = place.node
        for rep, (mods, proj) in enumerate(zip(module.repeats, module.projections)):
            exit_ = self._repeat(ctx, (block, mods, proj), place, rep, exit_)
        return exit_

    def _repeat(self, ctx, parts, place, rep, entry):
        block, mods, proj = parts
        last = series(ctx, entry, [(b, m, place.child(f"r{rep}.{k}", f"repeats.{rep}.{k}"))
                                   for k, (b, m) in enumerate(inner_pairs(block, mods))])
        add = place.child(f"r{rep}.add", f"repeats.{rep}")
        skip = entry
        if isinstance(proj, nn.Linear):
            p = place.child(f"r{rep}.proj", f"projections.{rep}", "projection")
            ctx.node("dense", p, proj, {"units": int(proj.out_features), "activation": "none",
                                        "bias": proj.bias is not None})
            ctx.edge(entry, p.node, "skip")
            skip = p.node
        ctx.node("residual_add", add, None, {}, last)
        ctx.edge(last, add.node, "data")
        ctx.edge(skip, add.node, "skip")
        return add.node

    def params(self, module, site):
        repeats = []
        for rep in range(len(module.repeats)):
            head = f"{site.node}.r{rep}."
            mine = [c for c in site.children if c.startswith(head)]
            proj = f"{head}proj"
            repeats.append({"inner": [c for c in mine if c[len(head):].isdigit()],
                            "projection": proj if proj in mine else None,
                            "add": f"{head}add"})
        return {"repeats": repeats}


class _Parallel(_Reader):
    """Branches `<id>.b<k>` (a composite branch nests as `<id>.b<k>.<j>`) and the merge
    `<id>.merge`."""

    TYPE, VIEW, FORMULA = "parallel", "group", "[f_1(x) ; … ; f_n(x)] | Σ_i f_i(x)"
    PAYLOAD = ("branches", "merge")

    def config(self, block, module):
        return {"merge": str(module.merge), "branches": len(module.branches)}

    def expand(self, ctx, block, module, place):
        ctx.node(self.TYPE, place, module, self.checked(block, module))
        merge = place.child("merge", "branches")
        for k, (b, m) in enumerate(inner_pairs(block, module.branches)):
            branch = place.child(f"b{k}", f"branches.{k}")
            ctx.edge(place.node, branch.node, "branch")
            ctx.edge(ctx.expand(b, m, branch), merge.node, "merge")
        ctx.node("parallel_merge", merge, None, {"merge": str(module.merge)}, place.node)
        return merge.node

    def params(self, module, site):
        return {"branches": [c for c in site.children if c != f"{site.node}.merge"],
                "merge": str(module.merge)}


class _Decoder(_Reader):
    """Each linear layer is a child `<id>.l<k>`: a dense, role `decoder_layer`, carrying the
    activation that follows it (none after the last)."""

    TYPE, VIEW, PAYLOAD = "decoder", "group", ("children",)
    FORMULA = "x̂ = W_k·act(… act(W_1 z + b_1) …) + b_k"

    def config(self, block, module):
        lins = [m for m in module.net if isinstance(m, nn.Linear)]
        acts = [act_name(m) for m in module.net if not isinstance(m, nn.Linear)]
        held = {"widths": [int(m.out_features) for m in lins[:-1]],
                "out": int(lins[-1].out_features)}
        if acts:                        # a decoder of one layer holds no activation
            held["activation"] = acts[0]
        return held

    def expand(self, ctx, block, module, place):
        held = self.checked(block, module)
        ctx.node(self.TYPE, place, module, held)
        last = place.node
        linears = [(j, m) for j, m in enumerate(module.net) if isinstance(m, nn.Linear)]
        for k, (j, lin) in enumerate(linears):
            child = place.child(f"l{k}", f"net.{j}", "decoder_layer")
            act = held.get("activation", "none") if k < len(linears) - 1 else "none"
            ctx.node("dense", child, lin, {"units": int(lin.out_features), "activation": act,
                                           "bias": lin.bias is not None})
            ctx.edge(last, child.node, "data")
            last = child.node
        return last

    def params(self, module, site):
        return {"children": list(site.children)}


class _Input(_Reader):
    TYPE, VIEW, CALL, PAYLOAD = "input", "io", "io", ("columns", "stream")
    PROBE = PG.INPUT
    FORMULA = "x = the shaped inputs (numeric, nominal ids, edges)"

    def params(self, module, site):
        return {"columns": [str(c) for c in (site.labels or ())],
                "stream": str(site.config.get("stream"))}


class _Output(_Reader):
    TYPE, VIEW, CALL, FORMULA, PAYLOAD = "output", "io", "io", "ŷ = link(z)", ("link", "outputs")
    PROBE = PG.OUTPUT

    def params(self, module, site):
        return {"link": site.config.get("link"), "outputs": dict(site.config.get("outputs"))}


class _ResidualAdd(_Reader):
    TYPE, VIEW, CALL, FORMULA = "residual_add", "merge", "inner", "y = P·x + f(x)"
    PROBE = PG.RESIDUAL_ADD


class _ParallelMerge(_Reader):
    TYPE, VIEW, CALL, PAYLOAD = "parallel_merge", "merge", "inner", ("merge",)
    PROBE = PG.PARALLEL_MERGE
    FORMULA = "[b_1 ; … ; b_n] | Σ_i b_i"

    def params(self, module, site):
        return {"merge": str(site.config.get("merge"))}


GROUP = (_Reparameterize(), _Custom(), _Sequential(), _Residual(), _Parallel(), _Decoder(),
         _Input(), _Output(), _ResidualAdd(), _ParallelMerge())
