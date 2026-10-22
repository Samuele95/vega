"""
inspection.reduce — what one captured tensor of the explained sample becomes in an
explanation, and the number budget that bounds a whole explanation (inspect-contract §5.4;
build plan R-F5).

`Reducer.reduce(t, kind)` takes a per-sample tensor (the batch axis already removed) and
returns its **reduced form**:

- `scalar` (0-D): `{form, value}`;
- `vector` (1-D, d ≤ `vector_max`): `{form, n, values}`;
- `vector` (1-D, d > `vector_max`): `{form, n, top: [[i, v]] (the 64 largest |v|), summary}`;
- `sequence` (2-D, steps × width; higher ranks fold their leading axes into steps):
  `{form, steps, width, per_step: {norm, mean}, per_channel: {mean, absmax}, heat}`;
- `attention` (kind `attention`; (heads ×) q × k): `{form, heads, q, k, maps: [heat per head]}`;
- `nodes` (`Reducer.nodes(t, ids, hops, target)`, a graph's N × d):
  `{form, ids, hop, norm, target: {id, values}}`;
- `summary` (`Reducer.summarise(t, axes)`, a node at level `summary`):
  `{form, norms: {axis: [≤ 64]}, sums: {axis: [≤ 64]}}`.

A heat map (`heat`) is `{rows, cols, row_edges, col_edges, values}`: the matrix block-reduced to
at most `heat_max` blocks per axis — the identity on an axis of at most `heat_max` — with
block `i` of an axis of length `L` > `heat_max` covering `[⌊i·L/m⌋, ⌊(i+1)·L/m⌋)`. Blocks
are averaged, and **summed** for kind `relevance` (`out ⊙ grad_out`), so a relevance map keeps
its total. The statistics keep their names' meaning for every kind (`norm` is the L2 norm,
`mean` the mean): a relevance sum over a step is its `mean` times `width`.

`Budget(detail)` places the node payloads of one explanation in graph order. `place(node_id,
build)` asks `build(level)` for the node's parts at the requested level — `full` (`exact` under
`detail: "exact"`) — and keeps it when its numbers fit what remains of the limit
(`explain_float_budget`, or `explain_exact_float_budget` for `exact`); otherwise it rebuilds the
node at `summary`, and failing that marks it `omitted` with a `why`. Under `exact`, once a node's
exact tensors do not fit, it and every later node are summarised. `detail: "summary"` puts every
node at `summary`. `report()` is the explanation's `budget` object, and `degraded` says whether
any node was placed below the requested level. The order is fixed, so the same request gives
the same bytes.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .codec import LIMITS, CodecError, host, num, nums, summary

__all__ = ["DETAILS", "KINDS", "LEVELS", "PARTS", "TOP_K", "Budget", "Reducer", "block_edges",
           "count", "node_payload"]

KINDS = ("value", "relevance", "attention")
DETAILS = ("full", "summary", "exact")
LEVELS = ("exact", "full", "summary", "omitted")
PARTS = ("in", "out", "grad_out", "relevance", "extra", "recompute", "exact")
TOP_K = 64          # a long vector keeps its 64 largest |v|
AXIS_MAX = 64       # a summary-level node holds at most 64 numbers per axis


def block_edges(length: int, limit: int) -> list:
    """The block boundaries of one axis: the identity when `length` ≤ `limit`."""
    if length <= limit:
        return list(range(length + 1))
    return [(i * length) // limit for i in range(limit + 1)]


def _blocks(a: np.ndarray, edges: list, axis: int) -> np.ndarray:
    """Block sums of `a` along `axis` (float64)."""
    if a.shape[axis] == 0:
        return a
    return np.add.reduceat(a, np.asarray(edges[:-1], dtype=np.intp), axis=axis)


def _sizes(edges: list) -> np.ndarray:
    return np.diff(np.asarray(edges, dtype=np.float64))


def _as64(t) -> np.ndarray:
    return np.asarray(host(t), dtype=np.float64)


class Reducer:
    """The reduced forms of §5.4. Stateless: one instance serves every explanation."""

    def __init__(self, limits=LIMITS):
        self._limits = limits

    def reduce(self, t, kind: str = "value") -> dict:
        """The reduced form of one per-sample tensor."""
        if kind not in KINDS:
            raise CodecError(f"reduction kind {kind!r} is not one of {list(KINDS)}")
        a = _as64(t)
        if kind == "attention":
            return self.attention(a)
        if a.ndim == 0:
            return {"form": "scalar", "value": num(a)}
        if a.ndim == 1:
            return self.vector(a)
        if a.ndim > 2:
            a = a.reshape(-1, a.shape[-1])
        return self.sequence(a, kind)

    def vector(self, t) -> dict:
        a = _as64(t).reshape(-1)
        if a.size <= self._limits["vector_max"]:
            return {"form": "vector", "n": int(a.size), "values": nums(a)}
        order = np.argsort(-np.abs(a), kind="stable")[:TOP_K]
        return {"form": "vector", "n": int(a.size),
                "top": [[int(i), num(a[i])] for i in order], "summary": summary(a)}

    def sequence(self, t, kind: str = "value") -> dict:
        a = _as64(t)
        if a.ndim != 2:
            raise CodecError(f"a sequence is steps × width, not a {a.ndim}-D tensor")
        steps, width = a.shape
        return {"form": "sequence", "steps": int(steps), "width": int(width),
                "per_step": {"norm": nums(np.sqrt(np.add.reduce(a * a, axis=1))),
                             "mean": nums(_mean(a, axis=1))},
                "per_channel": {"mean": nums(_mean(a, axis=0)),
                                "absmax": nums(_absmax(a, axis=0))},
                "heat": self.heat(a, "sum" if kind == "relevance" else "mean")}

    def attention(self, t) -> dict:
        """Attention weights — `(k,)` for a pool, `(q, k)` for one head, `(heads, q, k)` — one
        mean heat map per head."""
        a = _as64(t)
        if a.ndim in (1, 2):                # one query over k steps (a pool), or one head
            a = a.reshape((1,) * (3 - a.ndim) + a.shape)
        if a.ndim != 3:
            raise CodecError(f"attention weights are (heads,) q × k, not a {a.ndim}-D tensor")
        return {"form": "attention", "heads": int(a.shape[0]), "q": int(a.shape[1]),
                "k": int(a.shape[2]), "maps": [self.heat(m, "mean") for m in a]}

    def nodes(self, t, ids, hops, target: int) -> dict:
        """Graph nodes (N × d): the kept neighbourhood — `ids` with their `hops`, ordered
        nearer hops first (stable: the caller's order within a hop) and capped at
        `graph_nodes_max` — each node's norm, and the target node's values."""
        a = _as64(t)
        if a.ndim == 0:
            raise CodecError("graph nodes are a tensor with a node axis, not a scalar")
        a = a.reshape(a.shape[0], -1)
        ids, hops = [int(i) for i in ids], [int(h) for h in hops]
        if len(ids) != len(hops):
            raise CodecError(f"{len(ids)} node id(s) with {len(hops)} hop(s)")
        bad = [i for i in [*ids, int(target)] if not 0 <= i < a.shape[0]]
        if bad or min(hops, default=0) < 0:
            raise CodecError(f"node id(s) {bad} outside the {a.shape[0]} node(s) captured, "
                             "or a negative hop")
        order = sorted(range(len(ids)), key=hops.__getitem__)[:self._limits["graph_nodes_max"]]
        ids, hops = [ids[k] for k in order], [hops[k] for k in order]
        norms = np.sqrt(np.add.reduce(a[ids] * a[ids], axis=1)) if ids else np.zeros(0)
        return {"form": "nodes", "ids": ids, "hop": hops, "norm": nums(norms),
                "target": {"id": int(target), "values": nums(a[int(target)])}}

    def heat(self, matrix, how: str = "mean") -> dict:
        """A block mean (or block sum, `how="sum"`) of a 2-D matrix, ≤ `heat_max` per axis."""
        a = _as64(matrix)
        if a.ndim != 2 or how not in ("mean", "sum"):
            raise CodecError(f"a heat map reduces a 2-D matrix by mean or sum, not a {a.ndim}-D "
                             f"one by {how!r}")
        limit = self._limits["heat_max"]
        rows, cols = block_edges(a.shape[0], limit), block_edges(a.shape[1], limit)
        v = _blocks(_blocks(a, rows, 0), cols, 1)
        if how == "mean" and v.size:
            v = v / np.outer(_sizes(rows), _sizes(cols))
        return {"rows": len(rows) - 1, "cols": len(cols) - 1, "row_edges": rows,
                "col_edges": cols, "values": nums(v) if v.size else [[] for _ in rows[1:]]}

    def summarise(self, t, axes=None) -> dict:
        """A node's `summary` form: per axis, the L2 norm and the sum over the other axes of
        each index, block-reduced to at most 64 numbers (a 0-D tensor is axis `value`)."""
        a = _as64(t)
        names = list(axes) if axes is not None else [f"axis{i}" for i in range(a.ndim)]
        if a.ndim == 0:
            a, names = a.reshape(1), (names or ["value"])
        if len(names) != a.ndim or len(set(names)) != len(names):
            raise CodecError(f"axis names {names} do not name the {a.ndim} axes once each")
        norms, sums = {}, {}
        for i, name in enumerate(names):
            other = tuple(j for j in range(a.ndim) if j != i)
            edges = block_edges(a.shape[i], AXIS_MAX)
            sq = np.add.reduce(a * a, axis=other) if other else a * a
            s = np.add.reduce(a, axis=other) if other else a
            norms[str(name)] = nums(np.sqrt(_blocks(sq, edges, 0)))
            sums[str(name)] = nums(_blocks(s, edges, 0))
        return {"form": "summary", "norms": norms, "sums": sums}


def _mean(a: np.ndarray, axis: int) -> np.ndarray:
    n = a.shape[axis]
    return np.add.reduce(a, axis=axis) / n if n else np.zeros(a.shape[1 - axis])


def _absmax(a: np.ndarray, axis: int) -> np.ndarray:
    return np.abs(a).max(axis=axis) if a.shape[axis] else np.zeros(a.shape[1 - axis])


def count(value) -> int:
    """The numbers a payload holds: every JSON number (not a boolean), and the element count
    of each base64 tensor."""
    if value is None or isinstance(value, (bool, str)):
        return 0
    if isinstance(value, (int, float, np.number)):
        return 1
    if isinstance(value, Mapping):
        if value.get("encoding") == "base64-le" and isinstance(value.get("data"), str):
            return math.prod(int(d) for d in value["shape"])
        return sum(count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count(v) for v in value)
    raise CodecError(f"a {type(value).__name__} is not a JSON value")


def node_payload(level: str, parts: Mapping, why: str | None = None) -> dict:
    """One node payload of `explanation.nodes`: every key present, the unset ones null."""
    if level not in LEVELS:
        raise CodecError(f"level {level!r} is not one of {list(LEVELS)}")
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise CodecError(f"node payload parts {unknown} are not among {list(PARTS)}")
    if (level == "exact") != (parts.get("exact") is not None):
        raise CodecError("a node carries `exact` tensors exactly when its level is exact")
    if level == "omitted" and not why:
        raise CodecError("an omitted node says why")
    out = {"level": level, "why": why}
    out.update({k: parts.get(k) for k in PARTS})
    out["extra"] = dict(parts.get("extra") or {})
    if out["recompute"] is not None:
        out["recompute"] = num(out["recompute"])
    return out


class Budget:
    """The number budget of one explanation (§5.4): nodes placed in graph order."""

    def __init__(self, detail: str = "full", limits=LIMITS):
        if detail not in DETAILS:
            raise CodecError(f"detail {detail!r} is not one of {list(DETAILS)}")
        self.detail = detail
        self.limit = int(limits["explain_exact_float_budget" if detail == "exact"
                                else "explain_float_budget"])
        self.used = 0
        self._summarised, self._omitted = [], []
        self._exact_spent = False

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    @property
    def degraded(self) -> bool:
        """True when some node sits below the requested level (summarised or omitted)."""
        return bool(self._summarised or self._omitted)

    def place(self, node_id: str, build) -> dict:
        """The node's payload at the highest level that fits; `build(level)` gives its parts."""
        why = None
        if self._exact_spent:
            why = f"summarised: the exact budget of {self.limit} numbers ran out before this node"
        else:
            payload = node_payload(self.detail, build(self.detail))
            need = count(payload)
            if need <= self.remaining:
                return self._keep(node_id, payload, need, None)
            self._exact_spent = self.detail == "exact"
            why = (f"summarised: level {self.detail} needs {need} numbers and {self.remaining} "
                   f"of the budget's {self.limit} are left")
        if self.detail != "summary":
            payload = node_payload("summary", build("summary"))
            need = count(payload)
            if need <= self.remaining:
                return self._keep(node_id, payload, need, why)
        self._omitted.append(node_id)
        return node_payload("omitted", {}, why=(
            f"omitted: level summary needs {need} numbers and {self.remaining} of the "
            f"budget's {self.limit} are left"))

    def _keep(self, node_id: str, payload: dict, need: int, why) -> dict:
        self.used += need
        if why is not None:
            self._summarised.append(node_id)
            payload["why"] = why
        return payload

    def report(self) -> dict:
        """The explanation's `budget` object."""
        return {"limit": self.limit, "used": self.used, "summarised": list(self._summarised),
                "omitted": list(self._omitted)}
