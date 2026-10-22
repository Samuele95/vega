"""
rules_blocks — the compiler's rules on the blocks: the grammar, the family, the shape walk and
the backend's coverage.

Each is a pure `Context -> markers` function that `compile_spec.py` registers in `RULES`:

  G1a every block type is known                                family-catalog.md §3
  G1b a family outside the catalog needs known blocks; a catalog family is admitted on its
      sample shape and task                                    family-catalog.md §2, §7
  G1c residual widths or a projection                          family-catalog.md §3
  G1d custom blocks are user-sourced and declared              netspec-contract §2
  G2  the torch-free shape walk (vega_core.ShapeWalk)          family-catalog.md §3
  B2  the backend covers what the spec uses                    backend-contract.md §3

**Admission (G1b).** The admission table (`catalog.json` `admission`, family-catalog §7) is
the closed definition of what the coverage matrix runs; the compiler refuses by name the
combinations it lists as not admitted. Three of them have their own rule, which says so in its
own words: `gin` at level node (W2), `bi-*` on forecast or under a stream serving mode (W3),
a window family on a split that is not `time` (W1). G1b refuses the other two — a
reconstruction family on a window or graph sample, a graph family on an unsupervised or
anomaly task — quoting the catalog's sentence.

**B2 wording (D1 §14.1).** A catalog block in the backend's `planned.blocks` is refused as
*planned in backend <name> (build obligation …)*; one absent from both lists as *not covered*.
"""
from __future__ import annotations

import re
from typing import Mapping

from vega_core import ShapeWalk, error, nearest, ok, warn

__all__ = ["known_blocks", "open_family", "residual_widths", "custom_blocks", "shape_walk",
           "coverage"]

_RECONSTRUCTION_TASKS = ("unsupervised", "anomaly")


def known_blocks(ctx):
    out = []
    for path, block, _ in ctx.blocks:
        if not isinstance(block, Mapping):
            out.append(error("must be an object", rule="G1a", clause="netspec-contract §2",
                             where=path))
            continue
        kind = block.get("type")
        if ctx.catalog.block(kind) is None:
            hint = nearest(kind, ctx.catalog.block_types)
            out.append(error(f"block type {kind!r} is not in the catalog"
                             + (f" (nearest known type: {hint!r})" if hint else ""),
                             rule="G1a", clause="family-catalog.md §3", where=path))
    return out


def _unknown_family(ctx):
    if not ctx.family_name or ctx.family_row is not None:
        return []
    unknown = sorted({b.get("type") for _, b, _ in ctx.blocks
                      if isinstance(b, Mapping) and ctx.catalog.block(b.get("type")) is None})
    if unknown:
        return [error(f"family {ctx.family_name!r} is not in the catalog and its blocks are "
                      f"not all known ({', '.join(unknown)}) — a family outside the catalog "
                      "is accepted only when every block is", rule="G1b",
                      clause="family-catalog.md §2")]
    return [ok(f"family {ctx.family_name!r} is not a catalog row; accepted as user-defined "
               "because every block it uses is known", rule="G1b",
               clause="family-catalog.md §2")]


def _width_of(block):
    """The width a block leaves behind, when it declares one on its face. The compiler does
    not carry the builder's full shape algebra — the emitter's smoke pass is the authority on
    shapes, and duplicating it here would be a second implementation to keep in sync."""
    if not isinstance(block, Mapping):
        return None
    params = block.get("params") or {}
    kind = block.get("type")
    if kind == "dense":
        return params.get("units")
    if kind == "head":
        return params.get("out")
    if kind in ("rnn", "gru", "lstm"):
        hidden = params.get("hidden")
        return None if hidden is None else hidden * (2 if params.get("bidirectional") else 1)
    return None


def residual_widths(ctx):
    """`x + f(x)` needs equal widths or a projection (family-catalog §3).

    Two things are checkable here without the shape algebra: the inner chain must not change
    width between its own width-setting blocks, and — where the preceding sibling declares a
    width on its face — the inner chain must end at that width. Everything else is the smoke
    pass's, which fails loudly on a real batch.
    """
    out = []
    top = list(ctx.spec.get("blocks") or [])
    for i, block in enumerate(top):
        if not isinstance(block, Mapping) or block.get("type") != "residual":
            continue
        path = f"blocks[{block.get('id', i)}]"
        params = block.get("params") or {}
        if params.get("projection") == "auto":
            continue
        widths = [w for w in (_width_of(inner) for inner in (params.get("inner") or []))
                  if w is not None]
        if not widths:
            continue
        if len(set(widths)) > 1:
            out.append(error(f"the inner blocks change width ({widths}) and projection is not "
                             "'auto' — x + f(x) needs equal widths or a projection",
                             rule="G1c", clause="family-catalog.md §3 (residual)", where=path))
            continue
        before = next((_width_of(b) for b in reversed(top[:i]) if _width_of(b) is not None),
                      None)
        if before is not None and before != widths[0]:
            out.append(error(f"the block enters at width {before} and its inner chain ends at "
                             f"{widths[0]}, with projection {params.get('projection')!r} — "
                             "x + f(x) needs equal widths, or projection: auto to insert the "
                             "dense that reconciles them", rule="G1c",
                             clause="family-catalog.md §3 (residual)", where=path))
    return out


def custom_blocks(ctx):
    out = []
    for path, block, _ in ctx.blocks:
        if not isinstance(block, Mapping) or block.get("type") != "custom":
            continue
        if not str(block.get("source", "")).startswith("user:"):
            out.append(error(f"a custom block may only carry a user: source, not "
                             f"{block.get('source')!r} — it is code nobody else gated",
                             rule="G1d", clause="netspec-contract §2", where=path))
        params = block.get("params") or {}
        for key in ("code_ref", "out_shape"):
            if not params.get(key):
                out.append(error(f"a custom block must declare params.{key}", rule="G1d",
                                 clause="netspec-contract §2", where=path))
    return out


def _not_admitted(ctx, needle, fallback):
    """The catalog's own sentence for a combination it does not admit."""
    for reason in (ctx.catalog.admission.get("not_admitted") or ()):
        if needle in str(reason):
            return str(reason)
    return fallback


def _reconstruction_families(catalog) -> frozenset:
    """The families the admission table admits only on reconstruction tasks."""
    out = set()
    for row in catalog.admission.get("rows") or ():
        tasks = [str(t) for t in (row.get("tasks") or ())]
        if tasks and all(t.split("/")[0] in _RECONSTRUCTION_TASKS for t in tasks):
            out.update(row.get("families") or ())
    return frozenset(out)


def _admission(ctx):
    family, shape = ctx.family_name, ctx.sample_shape
    if family in _reconstruction_families(ctx.catalog) and shape in ("window", "graph"):
        reason = _not_admitted(ctx, "reconstruction family",
                               "a reconstruction family on window or graph")
        return [error(f"family {family!r} on a {shape} sample is not admitted "
                      f"(family-catalog §7: {reason}) — the catalog has no sequence or graph "
                      "decoder", rule="G1b", clause="family-catalog.md §7")]
    if family in ctx.catalog.families_on("graph") and \
            ctx.effective_kind in _RECONSTRUCTION_TASKS:
        reason = _not_admitted(ctx, "graph family", "a graph family on unsupervised or anomaly")
        return [error(f"family {family!r} on the {ctx.effective_kind} task is not admitted "
                      f"(family-catalog §7: {reason})", rule="G1b",
                      clause="family-catalog.md §7")]
    return []


def open_family(ctx):
    """G1b: an unknown family needs known blocks; a catalog family must be admitted."""
    if not ctx.family_name:
        return []
    if ctx.family_row is not None:
        return _admission(ctx)
    return _unknown_family(ctx)


# -- G2 the shape walk -------------------------------------------------------

def shape_walk(ctx):
    walk = ShapeWalk.for_spec(ctx.spec, ctx.catalog.data.get("blocks") or {})
    found = walk.walk(ctx.spec.get("blocks") or [])
    if not found:
        return [ok(f"the shape walk over the {walk.shape} sample finds every axis where the "
                   "next block reads it", rule="G2", clause="family-catalog.md §3 (shape rules)")]
    return [error(f.message, rule="G2", clause="family-catalog.md §3 (shape rules)",
                  where=f.where) for f in found]


# -- B2 backend coverage -----------------------------------------------------

def _uncovered_block(ctx, kind, path, name):
    planned = (ctx.coverage.get("planned") or {})
    if kind in (planned.get("blocks") or ()):
        match = re.search(r"build obligation [A-Za-z0-9_\-]+", str(planned.get("note") or ""))
        why = f" ({match.group(0)})" if match else ""
        return error(f"block {kind!r} is planned in backend {name}{why} and not built yet — "
                     "the spec cannot be emitted until the backend covers it",
                     rule="B2", clause="backend-contract.md §3", where=path)
    return error(f"block {kind!r} is not covered by the {name} backend: its coverage.json "
                 "neither lists it nor plans it", rule="B2", clause="backend-contract.md §3",
                 where=path)


def _sample_uses(ctx) -> tuple:
    """(label, value, coverage key) for what the sample asks of a backend (D1 §14.2): the
    shape, the graph batching and every recurrent block's return. A backend's coverage.json
    that does not declare a key makes no claim about it, and it is not checked."""
    uses = [("sample shape", ctx.sample_shape, "sample_shapes")]
    if ctx.sample_shape == "graph" and ctx.sample.get("batching"):
        uses.append(("graph batching", ctx.sample.get("batching"), "graph_batching"))
    for _, block, _ in ctx.blocks:
        if isinstance(block, Mapping) and block.get("type") in ("rnn", "gru", "lstm"):
            ret = (block.get("params") or {}).get("return", "last")
            uses.append((f"recurrent return ({block.get('id', block.get('type'))})", ret,
                         "recurrent_returns"))
    return tuple(uses)


def coverage(ctx):
    cov = ctx.coverage
    if not cov:
        return []
    name = cov.get("name", ctx.backend_name)
    out = []
    for path, block, _ in ctx.blocks:
        if not isinstance(block, Mapping):
            continue
        kind = block.get("type")
        if ctx.catalog.block(kind) is not None and kind not in (cov.get("blocks") or []):
            out.append(_uncovered_block(ctx, kind, path, name))
    checks = (("loss", (ctx.spec.get("loss") or {}).get("name"), "losses"),
              ("optimizer", (ctx.spec.get("optimizer") or {}).get("name"), "optimizers"),
              ("schedule", (ctx.spec.get("schedule") or {}).get("name"), "schedules"))
    checks += tuple(u for u in _sample_uses(ctx) if u[2] in cov)
    for label, value, field_name in checks:
        if value and value not in (cov.get(field_name) or []):
            out.append(error(f"{label} {value!r} is not covered by the {name} backend "
                             f"({', '.join(sorted(cov.get(field_name) or []))})", rule="B2",
                             clause="backend-contract.md §3"))
    for metric in sorted(ctx.declared_metrics - {"loss"}):
        if metric not in (cov.get("metrics") or []):
            out.append(error(f"metric {metric!r} is not implemented by the {name} backend's "
                             "metrics.py", rule="B2", clause="backend-contract.md §1"))
    return out + _loss_params(ctx, cov, name)


def _loss_params(ctx, cov, name) -> list:
    """`loss.params` against the keys the backend's loss takes (`coverage.json` `loss_params`,
    the losses' own `PARAMS`): a key the loss does not take is refused here, by name, because
    the backend's `build_loss` refuses it at the first step of a fit — after the smoke pass,
    which builds the network and not the loss. A key a weighed alternative carries (the same
    value, a key the alternative's loss takes and this one does not: a ladder merge) is what
    `build_loss` drops with a warning, and is warned here the same way."""
    takes = cov.get("loss_params")
    loss = ctx.spec.get("loss") or {}
    if not isinstance(takes, Mapping) or loss.get("name") not in takes:
        return []
    own, params = set(takes[loss["name"]]), dict(loss.get("params") or {})
    carried = set()
    for alt in loss.get("alternatives") or ():
        other = set(takes.get((alt or {}).get("name")) or ()) if (alt or {}).get("name") != \
            loss["name"] else set()
        carried |= {k for k, v in ((alt or {}).get("params") or {}).items()
                    if k in other and k not in own and params.get(k) == v}
    unknown = sorted(k for k, v in params.items() if v is not None and k not in own | carried)
    out = []
    if unknown:
        out.append(error(f"the {loss['name']} loss of the {name} backend takes no parameter "
                         f"{', '.join(map(repr, unknown))}; it takes "
                         f"{', '.join(sorted(own)) or 'none'} — build_loss would refuse the "
                         "spec at the first step of a fit", rule="B2",
                         clause="backend-contract.md §3", where="loss.params"))
    if carried & set(params):
        out.append(warn(f"loss.params {', '.join(sorted(carried & set(params)))}: not "
                        f"{loss['name']} parameters but a weighed alternative's "
                        "(loss.alternatives); the backend drops them", rule="B2",
                        clause="backend-contract.md §3", where="loss.params"))
    return out
