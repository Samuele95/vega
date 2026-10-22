"""
rules_regularization — the compiler's rule R2: one regularization declaration, live paths
derived from it (netspec-contract §3). A pure `Context -> markers` function that
`compile_spec.py` registers in `RULES`; the derivation itself is `vega_regularization.py`'s,
shared with `compile_spec.py --out` and the backend's `spec_snapshot.json`.

  ERROR  a kind no emitted code implements; a value out of its range (dropout and label
         smoothing in [0, 1), weight decay >= 0, a clipping bound > 0 or null — null only for
         grad_clip); two declarations of one kind (two dropout declarations covering one
         block) with different values; a live path set beside the declaration that covers it
         with a different value — the code reads the declaration, so the copy is a lie, and a
         ladder delta on either would be refused or dead
  WARN   the same with equal values (redundant: a delta on the declaration would then land
         beside a stale copy and its rung be refused — drop the copy); a declaration no
         emitted code reads (a dropout `where` that names no block, or covers no block that
         reads a rate; label smoothing under a loss that takes none) — it moves no weight; a
         `dropout` block with no `p` that no declaration covers (the backend's default trains)
  OK     what the emitted code reads (the paths `spec_snapshot.json` lists as `read_paths`)
"""
from __future__ import annotations

import json
import math

import vega_regularization as R
from vega_core import Level, error, ok, warn

__all__ = ["regularization"]

_CLAUSE = "netspec-contract §3"
_RANGES = {"dropout": "in [0, 1)", "label_smoothing": "in [0, 1)", "weight_decay": ">= 0",
           "grad_clip": "> 0, or null (no clipping)"}


def _err(message, where="regularization"):
    return error(message, rule="R2", clause=_CLAUSE, where=where)


def _wrn(message, where="regularization"):
    return warn(message, rule="R2", clause=_CLAUSE, where=where)


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) or (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
        and not isinstance(a, bool) and not isinstance(b, bool) and float(a) == float(b))


def _in_range(kind, value) -> bool:
    if value is None:
        return kind == "grad_clip"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    if kind in ("dropout", "label_smoothing"):
        return 0.0 <= value < 1.0
    return value > 0.0 if kind == "grad_clip" else value >= 0.0


def _declaration_markers(spec, decls, sites):
    out = []
    for d in decls:
        where = f"regularization[{d.index}] ({d.kind})"
        if d.kind not in R.KINDS:
            out.append(_err(f"regularization[kind={d.kind}]: no emitted code reads this kind; "
                            f"the backend implements {', '.join(R.KINDS)}", where))
            continue
        if not _in_range(d.kind, d.value):
            out.append(_err(f"value {d.value!r} is out of range: {d.kind} must be "
                            f"{_RANGES[d.kind]}", where))
        if d.kind == "label_smoothing" and not R.loss_takes_smoothing(spec):
            out.append(_wrn(f"label smoothing is declared but the loss "
                            f"{(spec.get('loss') or {}).get('name')!r} takes none "
                            f"({', '.join(R.LABEL_SMOOTHING_LOSSES)} does): the declaration "
                            "moves no weight", where))
        if d.kind == "dropout":
            out.extend(_dropout_reach(spec, d, sites, where))
    return out


def _dropout_reach(spec, d, sites, where):
    known = {str(b.get("id")) for b in _walk(spec.get("blocks")) if b.get("id") is not None}
    out = [_wrn(f"where names {w!r}, which is no block of this spec", where)
           for w in d.where or () if w not in known]
    if not any(R.covering(s, [d]) for s in sites):
        out.append(_wrn("the dropout declaration covers no block that reads a rate (a dropout "
                        "block, a transformer_encoder, a recurrent block of two or more "
                        "layers): it moves no weight", where))
    return out


def _walk(blocks):
    for b in blocks or []:
        if isinstance(b, dict):
            yield b
            yield from _walk((b.get("params") or {}).get("inner"))


def _duplicates(decls, sites):
    out = []
    for kind in R.GLOBAL_LIVE:
        same = [d for d in decls if d.kind == kind]
        for d in same[1:]:
            if _same(d.value, same[0].value):
                out.append(_wrn(f"{kind} is declared twice (regularization[{same[0].index}] and "
                                f"[{d.index}]) with one value; keep one declaration"))
            else:
                out.append(_err(f"conflicting declarations: {kind} is {same[0].value!r} in "
                                f"regularization[{same[0].index}] and {d.value!r} in "
                                f"regularization[{d.index}]; one kind, one declaration"))
    for site in sites:
        cover = R.covering(site, decls)
        values = {json.dumps(c.value) for c in cover}
        if len(values) > 1:
            out.append(_err(f"conflicting declarations: {site.path} is covered by "
                            + ", ".join(f"regularization[{c.index}] ({c.value!r})" for c in cover)
                            + "; a block's dropout has one declaration", site.path))
    return out


def _copy_marker(path, held, decl):
    where = f"regularization[{decl.index}] ({decl.kind})"
    if _same(held, decl.value):
        return _wrn(f"{path} repeats the declared {decl.kind} ({held!r}): the code reads the "
                    f"declaration, so drop the copy — a ladder delta on "
                    f"{decl.selector} would land beside it and its rung be refused", where)
    return _err(f"{path} is {held!r} but regularization[{decl.index}] declares {decl.kind} "
                f"{decl.value!r}: the emitted code reads the declaration, so the copy is never "
                "read — set the value once, in regularization[]", where)


def _copies(spec, decls, sites):
    out = []
    for kind, keys in R.GLOBAL_LIVE.items():
        decl = next((d for d in decls if d.kind == kind), None)
        node = spec
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if decl is not None and node is not None and (
                kind != "label_smoothing" or R.loss_takes_smoothing(spec)):
            out.append(_copy_marker("/".join(keys), node, decl))
    for site in sites:
        cover = R.covering(site, decls)
        if cover and site.has_value:
            out.append(_copy_marker(site.path, site.value, cover[0]))
        elif not cover and not site.has_value and site.path.endswith("/params/p"):
            out.append(_wrn(f"{site.path}: a dropout block with no p that no "
                            "regularization[kind=dropout] covers — the backend's default rate "
                            "trains; declare it", site.path))
    return out


def regularization(ctx):
    spec = ctx.spec
    decls, sites = R.declarations(spec), R.dropout_sites(spec)
    out = (_declaration_markers(spec, decls, sites) + _duplicates(decls, sites)
           + _copies(spec, decls, sites))
    if not any(m.level is Level.ERROR for m in out):
        out.append(ok(f"regularization: {len(decls)} declaration(s); the emitted code reads "
                      f"{', '.join(R.read_paths(spec)) or 'none'}", rule="R2", clause=_CLAUSE))
    return out
