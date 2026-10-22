"""
inspection.codec — tensors, numbers and limits of the inspection surface (inspect-contract
§5.1, §5.2, §7; build plan R-F5, R-F8).

Every number the inspector publishes passes through here, so the page, the story and the
checks read one numeric convention:

- `num(v)` rounds a number to **9 significant digits**, which round-trips float32 exactly
  (`np.float32(num(v)) == v` for every finite float32 `v`); `nums(values)` does it for an
  array and keeps its nesting. `NaN` and `±inf` are refused with `CodecError`: the service
  emits no non-finite JSON, so a non-finite weight or activation is a defect, not a value.
- `summary(values)` is the §5.2 summary (`n`, `mean`, `std`, `min`, `max`, `l2`, `abs_q` —
  the |v| quantiles 50, 90, 99, 100 — and a 32-bin `hist`), computed in float64 numpy over
  the row-major values with no BLAS call, so the same tensor gives the same bytes whatever
  its memory layout or device.
- `exact(tensor, axes)` is one `detail: "exact"` tensor: base64 of the little-endian
  float32 (or int64, for `edge_index`) values, row-major; `exact_set(parts)` builds a node's
  `exact` object; `decode(payload)` reads any tensor payload back.
- `TensorCodec` holds the tensors of one network view. Readers `add` them in node order and
  get their ids back; `tensors(weights)` encodes them all — `json` when n ≤ `json_list_max`,
  else `base64-le` when n ≤ `inline_tensor_max` and the running inline total (every float
  inlined so far, JSON lists included) stays ≤ `inline_total_max`, else `deferred`; every
  tensor is `withheld` under weights mode `summary`. `fetch(id, weights)` answers the tensor
  route (`base64-le`, from the live parameters, through an 8-entry LRU).

Torch tensors are read through `.detach()` without importing torch, so the module needs
numpy only and runs on any array-like.
"""
from __future__ import annotations

import base64
import copy
import math
import re
import threading
from collections import OrderedDict
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

__all__ = ["ABS_Q", "EXACT_KEYS", "HIST_BINS", "LIMITS", "ROLES", "SIGNIFICANT_DIGITS",
           "WEIGHTS_MODES", "CodecError", "TensorCodec", "TensorUnavailable", "b64", "decode",
           "exact", "exact_set", "host", "num", "nums", "summary"]

# inspect-contract §7 (the machine copy is catalog.json `inspection.limits`)
LIMITS = MappingProxyType({
    "inline_tensor_max": 262_144,     # floats per tensor inlined in /v1/network
    "inline_total_max": 1_048_576,    # floats inlined in /v1/network overall
    "json_list_max": 1_024,           # tensors this small travel as a JSON list
    "vector_max": 4_096,              # a captured vector longer than this is summarised
    "heat_max": 64,                   # heat maps are block-reduced to at most 64 x 64
    "graph_nodes_max": 256,           # nodes reported around an explained node
    "explain_float_budget": 262_144,  # numbers in one `full` explanation
    "explain_exact_float_budget": 4_194_304,  # numbers in one `exact` explanation
    "explain_rows_max": 4_096,        # raw rows (window) or nodes (graph) per request
    "explain_edges_max": 65_536,      # edges of one explained graph
    "ig_steps_default": 32, "ig_steps_max": 128, "ig_batch_max": 65_536,
})
SIGNIFICANT_DIGITS = 9
TENSOR_ID = re.compile(r"^[A-Za-z0-9_.@:-]{1,128}$")
ROLES = ("weight", "bias", "table", "buffer")
WEIGHTS_MODES = ("full", "summary")
ENCODINGS = ("json", "base64-le", "deferred", "withheld")
EXACT_KEYS = ("in", "out", "grad_out", "pre", "post", "grad_pre", "hidden", "alpha",
              "edge_index")
EXACT_REQUIRED = ("in", "out", "grad_out")
WIRE = MappingProxyType({"float32": "<f4", "int64": "<i8"})      # the bytes on the wire
NATIVE = MappingProxyType({"float32": np.float32, "int64": np.int64})
ABS_Q = (0.5, 0.9, 0.99, 1.0)
HIST_BINS = 32
FETCH_CACHE = 8


class CodecError(ValueError):
    """A number or tensor the inspection surface refuses to publish (a defect upstream)."""


class TensorUnavailable(LookupError):
    """The tensor route cannot answer: an unknown id, or weights withheld (a 404, not a defect)."""


def num(v) -> float:
    """One number at 9 significant digits; a non-finite value is refused."""
    x = float(v)
    if not math.isfinite(x):
        raise CodecError(f"a non-finite number ({x}) is a defect: the inspection surface "
                         "emits no NaN or infinity")
    return float(f"{x:.{SIGNIFICANT_DIGITS}g}")


def nums(values):
    """`num` over an array, keeping its nesting (a 0-D array gives one float)."""
    a = np.asarray(host(values), dtype=np.float64)
    _require_finite(a)
    flat = [float(f"{x:.{SIGNIFICANT_DIGITS}g}") for x in a.ravel().tolist()]
    return np.asarray(flat, dtype=np.float64).reshape(a.shape).tolist()


def host(tensor) -> np.ndarray:
    """A numpy view of a torch tensor or an array-like, its precision kept (half and bfloat16
    tensors are read as float32, which numpy can hold)."""
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        if t.is_floating_point() and t.element_size() < 4:
            t = t.float()
        a = t.numpy()
    else:
        a = np.asarray(tensor)
    if a.dtype.kind not in "biuf":
        raise CodecError(f"a tensor of dtype {a.dtype} is not a real number array")
    return a


def _require_finite(a: np.ndarray, what: str = "a tensor") -> None:
    if a.dtype.kind == "f" and a.size and not np.isfinite(a).all():
        bad = int(np.size(a) - np.count_nonzero(np.isfinite(a)))
        raise CodecError(f"{what} holds {bad} non-finite value(s): a defect, the inspection "
                         "surface emits no NaN or infinity")


def _flat64(values) -> np.ndarray:
    """The row-major float64 values: one element order, whatever the memory layout."""
    return np.ascontiguousarray(host(values), dtype=np.float64).ravel()


def summary(values, what: str = "a tensor") -> dict:
    """§5.2's summary, in float64, rounded by `num`; deterministic byte for byte."""
    a = _flat64(values)
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "l2": 0.0,
                "abs_q": [0.0] * len(ABS_Q),
                "hist": {"lo": 0.0, "hi": 0.0, "counts": [0] * HIST_BINS}}
    _require_finite(a, what)
    mean = float(np.add.reduce(a)) / n
    std = math.sqrt(float(np.add.reduce((a - mean) ** 2)) / n)
    lo, hi = float(a.min()), float(a.max())
    return {"n": n, "mean": num(mean), "std": num(std), "min": num(lo), "max": num(hi),
            "l2": num(math.sqrt(float(np.add.reduce(a * a)))),
            "abs_q": [num(q) for q in np.quantile(np.abs(a), ABS_Q)],
            "hist": {"lo": num(lo), "hi": num(hi), "counts": _hist(a, lo, hi)}}


def _hist(a: np.ndarray, lo: float, hi: float) -> list:
    if hi > lo:
        counts, _ = np.histogram(a, bins=HIST_BINS, range=(lo, hi))
        return [int(c) for c in counts]
    return [int(a.size)] + [0] * (HIST_BINS - 1)   # one value: every sample in the first bin


def b64(values, dtype: str = "float32") -> str:
    """Base64 of the little-endian `dtype` values, row-major."""
    if dtype not in WIRE:
        raise CodecError(f"dtype {dtype!r} is not one of {sorted(WIRE)}")
    wire = np.ascontiguousarray(host(values), dtype=WIRE[dtype])
    _require_finite(wire)
    return base64.b64encode(wire.tobytes()).decode("ascii")


def decode(payload) -> np.ndarray:
    """A tensor payload (§5.2, or an exact tensor) back to its array; refuses one without data."""
    shape = tuple(int(d) for d in payload["shape"])
    dtype = payload.get("dtype", "float32")
    if dtype not in WIRE:
        raise CodecError(f"dtype {dtype!r} is not one of {sorted(WIRE)}")
    encoding = payload.get("encoding")
    if encoding == "json":
        return np.asarray(payload["data"], dtype=NATIVE[dtype]).reshape(shape)
    if encoding != "base64-le":
        raise CodecError(f"a {encoding!r} tensor carries no data to decode")
    raw = np.frombuffer(base64.b64decode(payload["data"], validate=True), dtype=WIRE[dtype])
    if raw.size != math.prod(shape):
        raise CodecError(f"the data holds {raw.size} values, the shape {list(shape)} "
                         f"{math.prod(shape)}")
    return raw.astype(NATIVE[dtype]).reshape(shape)


def _axes(axes, ndim: int) -> list:
    axes = [str(a) for a in (axes or ())]
    if len(axes) != ndim:
        raise CodecError(f"{len(axes)} axis name(s) {axes} for a {ndim}-D tensor")
    return axes


def exact(tensor, axes) -> dict:
    """One `exact` tensor: shape, axes, dtype (int64 for integer ids), base64-le data."""
    a = host(tensor)
    dtype = "int64" if a.dtype.kind in "biu" else "float32"
    return {"shape": [int(d) for d in a.shape], "axes": _axes(axes, a.ndim), "dtype": dtype,
            "encoding": "base64-le", "data": b64(a, dtype)}


def exact_set(parts) -> dict:
    """A node's `exact` object from {key: (tensor, axes) | None}; `in`, `out` and `grad_out`
    are always present (null when the node has none)."""
    unknown = sorted(set(parts) - set(EXACT_KEYS))
    if unknown:
        raise CodecError(f"exact tensors {unknown} are not among {list(EXACT_KEYS)}")
    out = {k: None for k in EXACT_REQUIRED}
    for key in EXACT_KEYS:
        if parts.get(key) is not None:
            tensor, axes = parts[key]
            out[key] = exact(tensor, axes)
    return out


class _Entry(NamedTuple):
    tensor: object          # the live tensor, read again on every fetch
    node: str
    role: str
    shape: tuple
    axes: tuple
    n: int
    summary: dict


class TensorCodec:
    """The tensors of one network view, in the order the readers add them (node order)."""

    def __init__(self, limits=LIMITS):
        self._limits = MappingProxyType(dict(limits))
        self._entries: dict = {}
        self._lock = threading.Lock()
        self._cache: OrderedDict = OrderedDict()

    @property
    def limits(self):
        return self._limits

    def add(self, tensor, tensor_id: str, node: str, role: str, axes) -> str:
        """Register one tensor as `<node>.<name>`; returns its id, which a payload cites."""
        if not TENSOR_ID.match(str(tensor_id)) or not str(tensor_id).startswith(f"{node}."):
            raise CodecError(f"tensor id {tensor_id!r} must match {TENSOR_ID.pattern} and read "
                             f"'<node id>.<name>' for node {node!r}")
        if role not in ROLES:
            raise CodecError(f"tensor role {role!r} is not one of {list(ROLES)}")
        if tensor_id in self._entries:
            raise CodecError(f"tensor id {tensor_id!r} is registered twice")
        a = host(tensor)
        self._entries[tensor_id] = _Entry(tensor, node, role, tuple(int(d) for d in a.shape),
                                          tuple(_axes(axes, a.ndim)), int(a.size),
                                          summary(a, f"tensor {tensor_id}"))
        return tensor_id

    def ids(self) -> tuple:
        return tuple(self._entries)

    def tensors(self, weights: str = "full") -> dict:
        """Every tensor's payload, in registration order, under the §5.2 encoding rule."""
        _mode(weights)
        out, inline = {}, 0
        for tensor_id, entry in self._entries.items():
            encoding = self._encoding(entry.n, inline, weights)
            if encoding in ("json", "base64-le"):
                inline += entry.n
            out[tensor_id] = self.encode(tensor_id, encoding)
        return out

    def _encoding(self, n: int, inline: int, weights: str) -> str:
        if weights == "summary":
            return "withheld"
        if n <= self._limits["json_list_max"]:
            return "json"
        if n <= self._limits["inline_tensor_max"] and \
                inline + n <= self._limits["inline_total_max"]:
            return "base64-le"
        return "deferred"

    def encode(self, tensor_id: str, encoding: str) -> dict:
        """One registered tensor's payload under the encoding given, read from the live tensor."""
        entry = self._entry(tensor_id)
        if encoding not in ENCODINGS:
            raise CodecError(f"encoding {encoding!r} is not one of {list(ENCODINGS)}")
        data = None
        if encoding == "json":
            data = nums(np.asarray(host(entry.tensor), dtype=np.float32).reshape(-1))
        elif encoding == "base64-le":
            data = b64(entry.tensor, "float32")
        return {"id": tensor_id, "node": entry.node, "role": entry.role,
                "shape": list(entry.shape), "axes": list(entry.axes), "dtype": "float32",
                "encoding": encoding, "data": data, "summary": copy.deepcopy(entry.summary)}

    def fetch(self, tensor_id: str, weights: str = "full") -> dict:
        """The tensor route: one tensor as `base64-le`, through an 8-entry LRU. Every answer
        is the caller's own copy: editing it never reaches the cache or another caller."""
        if _mode(weights) == "summary":
            raise TensorUnavailable("this service does not publish raw weights")
        with self._lock:
            hit = self._cache.get(tensor_id)
            if hit is not None:
                self._cache.move_to_end(tensor_id)
                return copy.deepcopy(hit)
        payload = self.encode(tensor_id, "base64-le")
        with self._lock:
            self._cache[tensor_id] = payload
            while len(self._cache) > FETCH_CACHE:
                self._cache.popitem(last=False)
        return copy.deepcopy(payload)

    def _entry(self, tensor_id: str) -> _Entry:
        entry = self._entries.get(tensor_id)
        if entry is None:
            raise TensorUnavailable(f"no tensor {tensor_id!r} in this network")
        return entry


def _mode(weights: str) -> str:
    if weights not in WEIGHTS_MODES:
        raise CodecError(f"weights mode {weights!r} is not one of {list(WEIGHTS_MODES)}")
    return weights
