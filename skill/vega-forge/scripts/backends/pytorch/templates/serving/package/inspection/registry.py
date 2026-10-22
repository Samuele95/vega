"""
inspection.registry — `READERS`, the constant table from a node type to its reader: the 33
catalog block types and the 5 synthetic ones (`input`, `output`, `residual_add`,
`parallel_merge`, `encoder_layer`) (inspect-contract §3.1; build plan R-B1, R-F7).

`reader(type)` is the one lookup the stage graph makes. A type without a reader is refused —
`no inspection reader for block type '<t>'` — and the inspector does not open; `/vega-pack`
turns the same refusal into `ERROR:` before a bundle ships. `readers_digest()` is the
`inspection.readers` digest of `/v1/network`: the sha256 of the view table the readers carry
(type, view, call, formula, payload keys), so a page can tell which reader table drew it.
"""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

from .readers import CORE, GraphError
from .readers_graph import GRAPH
from .readers_group import GROUP
from .readers_seq import SEQ

__all__ = ["READERS", "reader", "readers_digest", "view_table"]

READERS = MappingProxyType({r.TYPE: r for r in (*CORE, *GROUP, *SEQ, *GRAPH)})


def reader(node_type: str):
    """The reader of one node type, or the refusal that names the type."""
    found = READERS.get(str(node_type))
    if found is None:
        raise GraphError(f"no inspection reader for block type {str(node_type)!r}")
    return found


def view_table() -> dict:
    """type -> {view, call, formula, payload}: the view table the readers carry."""
    return {t: {"view": r.VIEW, "call": r.CALL, "formula": r.FORMULA,
                "payload": list(r.PAYLOAD)} for t, r in sorted(READERS.items())}


def readers_digest() -> str:
    """sha256 of the canonical JSON of `view_table()`."""
    text = json.dumps(view_table(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
