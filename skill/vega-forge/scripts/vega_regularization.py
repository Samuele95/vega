"""
vega_regularization — the one regularization declaration and the live paths derived from it
(netspec-contract §3, rule R2). Pure, standard library only: the compiler's R2 rule
(`rules_regularization.py`), `compile_spec.py --out` and the backend's `spec_snapshot.json`
(`read_paths`) read the same derivation.

`regularization[]` is the only place dropout, weight decay, label smoothing and gradient
clipping are *declared*. Each kind has live paths — where a setting sits when nothing declares
it, and where the derivation writes the declared value:

  weight_decay     optimizer/params/weight_decay
  label_smoothing  loss/params/label_smoothing          (a loss that takes it: cross_entropy)
  grad_clip        training/grad_clip
  dropout          a `dropout` block's params/p, and params/dropout of a `transformer_encoder`
                   or of an `rnn`/`gru`/`lstm` with num_layers >= 2 (one layer reads none) —
                   every such block inside a block that `where` names (a composite covers its
                   `inner`), or every such block of the network when `where` is absent

The emitted code reads the declaration (`domain.Regularization`, its twin in the backend's
templates): the optimiser's weight decay, the loss's label smoothing, the clipping bound and
every dropout rate the builder sets come from `regularization[kind=…].value` when a declaration
covers them, and from the live path otherwise. So a ladder delta on the declaration moves the
weights, and a delta on a live path a declaration covers moves nothing: that path is
*shadowed*, never in `read_paths`.

Paths are written in /vega-train's delta syntax, blocks in the selector form
(`blocks[id=B3]/params/inner/3/params/p`; a block without an id by its index).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["KINDS", "GLOBAL_LIVE", "DROPOUT_PARAM", "LABEL_SMOOTHING_LOSSES", "Declaration",
           "Site", "declarations", "dropout_sites", "covering", "read_paths", "shadowed_paths",
           "resolve", "live_values", "snapshot", "loss_takes_smoothing"]

#: the kinds a declaration may name, and the live path of each kind that is not per block
GLOBAL_LIVE = {"weight_decay": ("optimizer", "params", "weight_decay"),
               "label_smoothing": ("loss", "params", "label_smoothing"),
               "grad_clip": ("training", "grad_clip")}
KINDS = tuple(GLOBAL_LIVE) + ("dropout",)
#: the block param a dropout declaration sets, per block type
DROPOUT_PARAM = {"dropout": "p", "rnn": "dropout", "gru": "dropout", "lstm": "dropout",
                 "transformer_encoder": "dropout"}
_RECURRENT = ("rnn", "gru", "lstm")
#: the losses whose params take label_smoothing (backends/pytorch/losses.py PARAMS)
LABEL_SMOOTHING_LOSSES = ("cross_entropy",)


@dataclass(frozen=True)
class Declaration:
    """One entry of `regularization[]`: its index, kind, value and `where` (None: everywhere)."""

    index: int
    kind: str
    value: Any
    where: tuple | None

    @property
    def path(self) -> str:
        return f"regularization/{self.index}/value"

    @property
    def selector(self) -> str:
        return f"regularization[kind={self.kind}]/value"


@dataclass(frozen=True)
class Site:
    """One block param a dropout declaration can set: where it sits, what it holds now."""

    path: str                 # the delta path
    keys: tuple               # its address in the spec dict, for writing
    ids: tuple                # the ids of the block and its ancestors (what `where` matches)
    block: str                # the block's id, or its position, for messages
    has_value: bool
    value: Any


def declarations(spec) -> tuple:
    rows = (spec or {}).get("regularization") or []
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        where = row.get("where")
        if where is not None:
            where = tuple(str(w) for w in (where if isinstance(where, (list, tuple)) else [where]))
        out.append(Declaration(i, str(row.get("kind")), row.get("value"), where))
    return tuple(out)


def _dropout_key(block) -> str | None:
    t = block.get("type")
    params = block.get("params") or {}
    if t in _RECURRENT and int(params.get("num_layers", params.get("layers", 1)) or 1) < 2:
        return None
    return DROPOUT_PARAM.get(t)


def dropout_sites(spec) -> tuple:
    """Every block param of the spec that reads a dropout rate, composites' `inner` included."""
    out = []

    def walk(blocks, keys, path, ancestors):
        for i, block in enumerate(blocks or []):
            if not isinstance(block, Mapping):
                continue
            bid = block.get("id")
            here = (f"{path}/{i}" if path else
                    (f"blocks[id={bid}]" if bid is not None else f"blocks/{i}"))
            ids = ancestors + ((str(bid),) if bid is not None else ())
            params = block.get("params") or {}
            key = _dropout_key(block)
            if key is not None:
                name = str(bid) if bid is not None else here
                out.append(Site(f"{here}/params/{key}", keys + (i, "params", key), ids, name,
                                key in params, params.get(key)))
            inner = params.get("inner")
            if isinstance(inner, (list, tuple)):
                walk(inner, keys + (i, "params", "inner"), f"{here}/params/inner", ids)

    walk((spec or {}).get("blocks"), ("blocks",), "", ())
    return tuple(out)


def covering(site: Site, decls) -> tuple:
    """The dropout declarations that cover `site`, in spec order."""
    return tuple(d for d in decls if d.kind == "dropout"
                 and (d.where is None or set(d.where) & set(site.ids)))


def loss_takes_smoothing(spec) -> bool:
    return ((spec or {}).get("loss") or {}).get("name") in LABEL_SMOOTHING_LOSSES


def _reads(spec, decl: Declaration, sites) -> bool:
    """Whether the emitted code reads this declaration's value."""
    if decl.kind == "dropout":
        return any(decl in covering(s, [decl]) for s in sites)
    if decl.kind == "label_smoothing":
        return loss_takes_smoothing(spec)
    return decl.kind in GLOBAL_LIVE


def _first(decls, kind):
    return next((d for d in decls if d.kind == kind), None)


def read_paths(spec) -> list:
    """The regularization paths the emitted code reads, sorted: each declaration it reads
    (`regularization[kind=<k>]/value` for the first of its kind, `regularization/<i>/value`
    for each), and every live path no declaration covers."""
    decls, sites = declarations(spec), dropout_sites(spec)
    out = set()
    for kind in KINDS:
        read = [d for d in decls if d.kind == kind and _reads(spec, d, sites)]
        if read and read[0] is _first(decls, kind):
            out.add(read[0].selector)
        out.update(d.path for d in read)
    for kind, keys in GLOBAL_LIVE.items():
        if _first(decls, kind) is None and (kind != "label_smoothing"
                                            or loss_takes_smoothing(spec)):
            out.add("/".join(keys))
    out.update(s.path for s in sites if not covering(s, decls))
    return sorted(out)


def shadowed_paths(spec) -> list:
    """The live paths a declaration covers: a delta there moves no weight (a dead delta)."""
    decls = declarations(spec)
    out = {"/".join(GLOBAL_LIVE[d.kind]) for d in decls if d.kind in GLOBAL_LIVE}
    out.update(s.path for s in dropout_sites(spec) if covering(s, decls))
    return sorted(out)


def _set(doc, keys, value):
    node = doc
    for k in keys[:-1]:
        if isinstance(node, dict) and k not in node:
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def live_values(spec) -> dict:
    """{live path: the value the emitted code trains with} for every covered live path."""
    decls = declarations(spec)
    out = {}
    for kind, keys in GLOBAL_LIVE.items():
        first = _first(decls, kind)
        if first is not None and (kind != "label_smoothing" or loss_takes_smoothing(spec)):
            out["/".join(keys)] = first.value
    for site in dropout_sites(spec):
        cover = covering(site, decls)
        if cover:
            out[site.path] = cover[0].value
    return out


def resolve(spec) -> dict:
    """A copy of `spec` with the declared values written into their live paths (what
    `compile_spec.py --out` writes as the resolved spec, and what the emitted code builds)."""
    doc = copy.deepcopy(dict(spec))
    decls = declarations(doc)
    for kind, keys in GLOBAL_LIVE.items():
        first = _first(decls, kind)
        if first is not None and (kind != "label_smoothing" or loss_takes_smoothing(doc)):
            _set(doc, keys, first.value)
    for site in dropout_sites(doc):
        cover = covering(site, decls)
        if cover:
            _set(doc, site.keys, cover[0].value)
    return doc


def snapshot(spec) -> dict:
    """`spec_snapshot.json`'s `_resolved.regularization`: what trains, and what is dead."""
    return {"read_paths": read_paths(spec), "shadowed_paths": shadowed_paths(spec),
            "live_values": live_values(spec)}
