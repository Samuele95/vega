"""
inspect_network.py — NI1–NI7 of D6 §8.1 over one served `vega/network-view@2` payload and the
bundle it describes (NI8, the inspection-off answer, needs a service of its own and lives in
`inspect_check.py`). Each function returns `(ok, detail)` observations.

    ni1(answer)                         a 200 body with `available: true`
    ni2(view, netspec)                  the top-level node ids are the packed netspec's block
                                        ids, in order; nested ids follow inspect-contract §3.2's
                                        grammar under their parent; the edges form a DAG from
                                        the inputs to the output
    ni3(view, smoke, where)             each top-level out_shape = the packed network's smoke
                                        per_block[id].out_shape (BundleRecords.packed_smoke)
    ni4(view, smoke, match, where)      Σ leaf params = params.total = smoke.params_total =
                                        Σ numel of the weights.pt entries the weights pair with
    ni5(view, match)                    every tensor decodes to its declared shape; the
                                        published tensors are weights.pt's, byte for byte, and
                                        cover it (the drawing is this network)
    ni6(network, netspec, bundle)       inputs = the feature contract's columns (numeric then
                                        nominal; a window's length; a graph's edges); outputs =
                                        the task's output list
    ni7(first, second, pin)             two GETs give the same bytes; the body's digest is its
                                        canonical digest and the one pack pinned
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping

import numpy as np

from lanekit import contracts as C

__all__ = ["canonical_digest", "ni1", "ni2", "ni3", "ni4", "ni5", "ni6", "ni7"]

NODE_ID = re.compile(C.NODE_ID)


def ni1(status: int, body) -> list:
    ok = status == 200 and isinstance(body, Mapping) and body.get("available") is True
    return [(ok, f"GET {C.NETWORK_ROUTE} answered {status}, available = "
                 f"{(body or {}).get('available') if isinstance(body, Mapping) else None}")]


# --------------------------------------------------------------------------- NI2
def _acyclic(ids, edges) -> bool:
    indeg = {i: 0 for i in ids}
    out: dict = {}
    for e in edges:
        indeg[e["to"]] = indeg.get(e["to"], 0) + 1
        out.setdefault(e["from"], []).append(e["to"])
    ready = [i for i, d in indeg.items() if d == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for nxt in out.get(node, ()):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    return seen == len(indeg)


def _reaches(edges, start: str) -> set:
    out: dict = {}
    for e in edges:
        out.setdefault(e["from"], []).append(e["to"])
    seen, stack = {start}, [start]
    while stack:
        for nxt in out.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def ni2(view, netspec: Mapping) -> list:
    blocks = [str(b["id"]) for b in netspec.get("blocks") or ()]
    top = view.top_level()
    out = [(top == blocks, f"top-level nodes {top} = the packed netspec's blocks {blocks}")]
    bad_ids = [i for i in view.ids if not NODE_ID.match(i)]
    nested = [i for i in view.ids if view.node(i).get("parent") is not None and
              not i.startswith(str(view.node(i)["parent"]) + ".")]
    out.append((not bad_ids and not nested,
                f"node ids follow the id grammar ({bad_ids[:3]}) and nest under their parent "
                f"({nested[:3]})"))
    edges = view.network.get("edges") or []
    inputs = [i for i in view.ids if view.node(i)["type"] == "input"]
    reach = set().union(*[_reaches(edges, i) for i in inputs]) if inputs else set()
    orphans = [i for i in view.ids if i not in reach and view.node(i)["type"] != "input"]
    out.append((_acyclic(view.ids, edges) and "output" in reach and not orphans,
                f"the edges form a DAG from {inputs} to the output (unreached: {orphans[:4]})"))
    return out


# --------------------------------------------------------------------------- NI3, NI4
def ni3(view, smoke: Mapping | None, where: str) -> list:
    per_block = {str(b["id"]): list(b.get("out_shape") or ())
                 for b in (smoke or {}).get("per_block") or ()}
    if not per_block:
        return [(False, f"no smoke per_block of the packed network: {where}")]
    out = []
    for node_id in view.top_level():
        got = list(view.node(node_id).get("out_shape") or ())
        want = per_block.get(node_id)
        out.append((got == want, f"{node_id} out_shape {got} = smoke {want} ({where})"))
    return out


def _leaves(view) -> list:
    return [i for i in view.ids if not view.children(i)]


def ni4(view, smoke: Mapping | None, match, where: str) -> list:
    leaves = sum(int((view.node(i).get("params") or {}).get("count") or 0) for i in _leaves(view))
    total = int((view.network.get("params") or {}).get("total") or -1)
    smoke_total = (smoke or {}).get("params_total")
    stored = None if match is None else match.params
    ok = leaves == total == smoke_total and (stored is None or stored == total)
    return [(ok, f"Σ leaf params {leaves} = params.total {total} = smoke.params_total "
                 f"{smoke_total} ({where}) = Σ numel of the matched weights.pt parameters "
                 f"{stored}")]


# --------------------------------------------------------------------------- NI5
def ni5(view, match) -> list:
    out = []
    for tid in view.tensor_ids():
        payload = view.network["tensors"][tid]
        if payload.get("encoding") == "withheld":
            out.append((False, f"{tid} is withheld: the weights mode is summary"))
            continue
        arr = view.tensor(tid)
        shape = [int(s) for s in payload.get("shape") or ()]
        out.append((arr is not None and list(arr.shape) == shape and bool(np.isfinite(arr).all()),
                    f"{tid} decodes ({payload.get('encoding')}) to its shape {shape}"))
    out.append((not match.unmatched and not match.uncovered and
                match.published_digest == match.stored_digest,
                f"the published tensors are weights.pt's: {len(match.pairs)} paired, unmatched "
                f"{list(match.unmatched)[:4]}, uncovered {list(match.uncovered)[:4]}, digest "
                f"{match.published_digest[:19]} = {match.stored_digest[:19]}"))
    return out


# --------------------------------------------------------------------------- NI6
def _features(netspec: Mapping) -> tuple:
    feats = (netspec.get("data_contract") or {}).get("features") or ()
    numeric = [f["name"] for f in feats if f.get("role", "numeric") != "nominal"]
    nominal = [f["name"] for f in feats if f.get("role") == "nominal"]
    return numeric, nominal


def _outputs_ok(outputs: Mapping, output: Mapping) -> tuple:
    """(ok, what) of `task.outputs` against MANIFEST.output."""
    kind, names = outputs.get("kind"), [str(n) for n in outputs.get("names") or ()]
    classes = [str(c) for c in output.get("classes") or ()]
    head_out = int(output.get("head_out") or len(names))
    want = {"classes": lambda: names == classes,
            "score": lambda: names == [str(output.get("positive") or (classes or ["1"])[-1])],
            "labels": lambda: names == classes,
            "value": lambda: names == ["value"],
            "quantiles": lambda: len(names) == len(output.get("quantiles") or ()),
            "thresholds": lambda: len(names) == len(classes) - 1,
            "horizon": lambda: len(names) == head_out and names[0].startswith("t+1"),
            "reconstruction": lambda: len(names) >= 1,
            "embedding": lambda: len(names) == head_out}.get(kind)
    return (want is not None and want()), f"outputs {kind} {names[:6]}"


def ni6(network: Mapping, netspec: Mapping, sample: Mapping, output: Mapping) -> list:
    inputs = network.get("inputs") or {}
    numeric, nominal = _features(netspec)
    got_nom = [n["name"] for n in inputs.get("nominal") or ()]
    out = [(list(inputs.get("numeric") or ()) == numeric and got_nom == nominal,
            f"inputs numeric {len(numeric)} then nominal {nominal} = the feature contract")]
    shape = sample.get("shape", "row")
    view_sample = network.get("sample") or {}
    if shape == "window":
        steps = (inputs.get("axis_labels") or {}).get("step") or ()
        out.append((view_sample.get("length") == sample.get("length") == len(steps),
                    f"window: length {view_sample.get('length')} and {len(steps)} step labels"))
    if shape == "graph":
        out.append((inputs.get("edges") is not None, "graph: the edge input is declared"))
    ok, what = _outputs_ok((network.get("task") or {}).get("outputs") or {}, output)
    out.append((ok, what + f" = MANIFEST.output ({output.get('reading')})"))
    return out


# --------------------------------------------------------------------------- NI7
def canonical_digest(body: Mapping) -> str:
    """inspect-contract §5.3: sha256 of the canonical body without `network_digest`."""
    rest = {k: v for k, v in body.items() if k != C.NETWORK_DIGEST_FIELD}
    text = json.dumps(rest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def ni7(first: bytes, second: bytes, pin: str | None, status: Mapping | None) -> list:
    body = json.loads(first.decode("utf-8"))
    digest = body.get(C.NETWORK_DIGEST_FIELD)
    out = [(first == second, f"two GETs: {len(first)} and {len(second)} bytes, identical"),
           (digest == canonical_digest(body), f"network_digest {str(digest)[:19]} is the "
                                               "body's canonical digest")]
    if pin is not None:
        out.append((digest == pin, f"network_digest = MANIFEST pin {pin[:19]}"))
        pinned = (status or {}).get("pinned")
        out.append((pinned is True, f"/health.inspection.pinned = {pinned}"))
    return out
