#!/usr/bin/env python3
"""
test_inspect_codec.py — the inspection codec and reducers (`templates/serving/package/
inspection/{codec,reduce}.py`), proved on the *rendered* modules (build plan I01; inspect-contract
§5.1, §5.2, §5.4, §7; R-F5, R-F8).

    python3 test_inspect_codec.py            (or: python3 -m pytest test_inspect_codec.py -q)

What is proved, and why it matters:

  numbers    `num` keeps 9 significant digits and every finite float32 survives the JSON round
             trip bit for bit (8 digits would not: the test shows a float32 that 8 digits
             lose); `NaN` and `±inf` are refused wherever a number is published.
  base64     `base64-le` is the little-endian float32 bytes, row-major, whatever the tensor's
             layout or origin (torch or numpy), and decodes to the tensor bit for bit; exact
             `int64` tensors (`edge_index`) too.
  encodings  `json` ≤ `json_list_max` < `base64-le` ≤ `inline_tensor_max` < `deferred`, the
             running inline total (JSON lists included) capped at `inline_total_max` in node
             order, `withheld` under weights mode `summary` — at the real limits and at small
             ones; the tensor route (`fetch`) is `base64-le`, reads the live tensor, keeps an
             8-entry LRU and refuses unknown ids and withheld weights; every payload handed
             out is the caller's own copy, so editing one never reaches the cache.
  summaries  deterministic byte for byte: the same values from torch, numpy, a transposed view,
             a Fortran-ordered copy, a second call and a fresh interpreter with one BLAS/OMP
             thread give the same bytes; the statistics are the ones §5.2 names.
  heat       block means (block sums for relevance) at most `heat_max` per axis, the identity
             at or under it, block edges ⌊i·L/m⌋, checked against a naive loop.
  forms      every reduced form of §5.4 and every node payload validates against
             `explanation-2.schema.json`; every tensor payload against `network-view-2`; a
             graph neighbourhood keeps its nearer hops first under the `graph_nodes_max` cap.
  budget     `full` degrades a node to `summary` and then to `omitted` (per node, the limit
             exactly reachable); `summary` puts every node at `summary`; `exact` summarises the
             node that does not fit and every later node, and sets `degraded`.
  contract   the limits, digits, roles, id grammar, weights modes and detail levels equal
             catalog.json `inspection`; both modules pass C11–C13 at `error`, hold no
             module-level mutable state (C7) and import only numpy and the standard library.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import base64
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates" / "serving" / "package"
FILES = (TEMPLATES / "inspection" / "codec.py", TEMPLATES / "inspection" / "reduce.py")
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
SCHEMAS = SCRIPTS.parent / "references" / "schemas"
PACKAGE = "codec_pkg"
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conformance  # noqa: E402
import render_templates  # noqa: E402

_RENDER = tempfile.TemporaryDirectory(prefix="inspect-codec-")


def _render():
    """The inspection templates rendered into `codec_pkg` (the Predictor's module), imported."""
    pipeline = render_templates.reference_pipeline()
    pipeline["system"]["package"] = PACKAGE
    rec = render_templates.render(TEMPLATES, _RENDER.name, pipeline=pipeline, spec="reference",
                                  into="Predictor", recursive=True)
    assert "serving/inspection/codec.py" in rec["files"], rec["files"]
    sys.path.insert(0, _RENDER.name)
    return (importlib.import_module(f"{PACKAGE}.serving.inspection.codec"),
            importlib.import_module(f"{PACKAGE}.serving.inspection.reduce"))


C, R = _render()


def _validator(schema_file: str, ref: str) -> Draft202012Validator:
    doc = json.loads((SCHEMAS / schema_file).read_text(encoding="utf-8"))
    return Draft202012Validator({"$ref": f"#/$defs/{ref}", "$defs": doc["$defs"]})


REDUCED = _validator("explanation-2.schema.json", "reduced")
NODE = _validator("explanation-2.schema.json", "node_payload")
EXACT_TENSOR = _validator("explanation-2.schema.json", "exact_tensor")
HEAT = _validator("explanation-2.schema.json", "heat")
TENSOR = _validator("network-view-2.schema.json", "tensor")
SUMMARY = _validator("network-view-2.schema.json", "summary")
BUDGET = Draft202012Validator(json.loads((SCHEMAS / "explanation-2.schema.json")
                                         .read_text(encoding="utf-8"))["properties"]["budget"])


def _valid(validator, doc):
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    assert not errors, "; ".join(f"{list(e.path)}: {e.message[:200]}" for e in errors[:5])


def _dumps(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _bits(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------------------- the contract

def test_constants_equal_the_catalog():
    ins = CATALOG["inspection"]
    assert dict(C.LIMITS) == ins["limits"]
    assert C.SIGNIFICANT_DIGITS == ins["numbers"]["significant_digits"] == 9
    assert list(C.ROLES) == ins["tensor_roles"]
    assert C.TENSOR_ID.pattern == ins["tensor_id_pattern"]
    assert list(C.WEIGHTS_MODES) == ins["weights_modes"]
    assert list(R.DETAILS) == ins["detail"]
    limits_schema = json.loads((SCHEMAS / "network-view-2.schema.json").read_text())["$defs"]
    assert set(limits_schema["limits"]["required"]) == set(C.LIMITS)
    exact_keys = json.loads((SCHEMAS / "explanation-2.schema.json").read_text())["$defs"]
    assert list(exact_keys["exact"]["properties"]) == list(C.EXACT_KEYS)
    assert set(exact_keys["node_payload"]["required"]) == {"level", "why", *R.PARTS}
    assert set(exact_keys["node_payload"]["properties"]["level"]["enum"]) == set(R.LEVELS)


# ------------------------------------------------------------------------------- numbers

SPECIAL = np.array([0.0, -0.0, 1.0, -1.0, 0.1, 1 / 3, np.pi, 65504.0, 16777217.0,
                    np.finfo(np.float32).max, -np.finfo(np.float32).max,
                    np.finfo(np.float32).tiny, np.finfo(np.float32).eps, 1 + 2 ** -23,
                    1.4e-45, 1.1754942e-38, 3.4028233e38], dtype=np.float32)


def _all_kinds_of_float32(n=200_000, seed=1):
    """Random bit patterns (every exponent, subnormals included) plus the special values."""
    raw = _rng(seed).integers(0, 2 ** 32, size=n, dtype=np.uint64).astype(np.uint32)
    f = raw.view(np.float32)
    return np.concatenate([SPECIAL, f[np.isfinite(f)]])


def _floats(n, seed):
    """Exactly `n` finite float32 values of every exponent."""
    return _all_kinds_of_float32(2 * n + 64, seed)[:n]


def test_nine_digits_round_trip_float32_exactly():
    values = _all_kinds_of_float32()
    back = np.asarray(json.loads(json.dumps(C.nums(values))), dtype=np.float32)
    assert np.array_equal(_bits(back), _bits(values)), "a float32 lost bits through 9 digits"
    for v in SPECIAL:
        assert _bits(np.float32(C.num(v)))[()] == _bits(v)[()]
    eight = np.asarray([float(f"{float(v):.8g}") for v in values], dtype=np.float32)
    assert not np.array_equal(_bits(eight), _bits(values)), "8 digits would do: the test is blunt"


def test_nums_keeps_nesting_and_refuses_non_finite():
    assert C.nums(np.float32(0.5)) == 0.5
    assert C.nums(torch.tensor([[1.0, 2.0], [3.0, 4.0]])) == [[1.0, 2.0], [3.0, 4.0]]
    assert C.num(np.float32(0.1)) == 0.100000001
    for bad in (float("nan"), float("inf"), -np.inf):
        with pytest.raises(C.CodecError):
            C.num(bad)
    with pytest.raises(C.CodecError):
        C.nums([1.0, float("nan")])
    with pytest.raises(C.CodecError):
        C.summary(np.array([1.0, np.inf], dtype=np.float32))
    with pytest.raises(C.CodecError):
        C.b64(np.array([np.nan], dtype=np.float32))
    with pytest.raises(C.CodecError):
        C.TensorCodec().add(torch.tensor([1.0, float("inf")]), "B1.bias", "B1", "bias", ["out"])
    with pytest.raises(C.CodecError):
        R.Reducer().reduce(torch.tensor([[0.0, float("nan")]]))


# ------------------------------------------------------------------------------- base64

def test_base64_le_is_little_endian_float32_row_major():
    assert C.b64(np.array([1.0], dtype=np.float32)) == "AACAPw=="
    m = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert base64.b64decode(C.b64(m)) == np.array([0, 1, 2, 3, 4, 5], dtype="<f4").tobytes()
    assert C.b64(m.T) == C.b64(np.ascontiguousarray(m.T))


@pytest.mark.parametrize("shape", [(), (1,), (7,), (5, 3), (4, 3, 2), (2, 1, 3, 5)])
def test_base64_round_trip_equals_the_float32_tensor(shape):
    values = _all_kinds_of_float32(4096, seed=len(shape))
    a = values[:math.prod(shape)].reshape(shape)
    for t in (a, torch.from_numpy(a.copy()), np.asfortranarray(a)):
        payload = {"shape": list(shape), "dtype": "float32", "encoding": "base64-le",
                   "data": C.b64(t)}
        back = C.decode(payload)
        assert back.shape == shape and back.dtype == np.float32
        assert np.array_equal(_bits(back), _bits(a))
    if len(shape) == 2:                                    # a strided view reads row-major
        t = torch.from_numpy(np.ascontiguousarray(a.T)).T
        assert not t.is_contiguous()
        assert np.array_equal(_bits(C.decode({"shape": list(shape), "encoding": "base64-le",
                                              "data": C.b64(t)})), _bits(a))


def test_exact_tensors_float32_and_int64():
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(3))
    e = C.exact(x, ["step", "channel"])
    _valid(EXACT_TENSOR, e)
    assert e["dtype"] == "float32" and e["shape"] == [3, 4]
    assert np.array_equal(C.decode(e), x.numpy())
    edges = torch.tensor([[0, 1, 2, 2 ** 40], [1, 2, 0, 5]], dtype=torch.long)
    ei = C.exact(edges, ["pair", "edge"])
    _valid(EXACT_TENSOR, ei)
    assert ei["dtype"] == "int64" and np.array_equal(C.decode(ei), edges.numpy())
    assert base64.b64decode(ei["data"])[:8] == (0).to_bytes(8, "little")
    with pytest.raises(C.CodecError):
        C.exact(x, ["only-one-axis"])
    both = C.exact_set({"in": (x, ["step", "channel"]), "alpha": (x[0], ["key"])})
    assert set(both) == {"in", "out", "grad_out", "alpha"} and both["out"] is None
    with pytest.raises(C.CodecError):
        C.exact_set({"weights": (x, ["a", "b"])})
    for enc in ("deferred", "withheld"):
        with pytest.raises(C.CodecError):
            C.decode({"shape": [1], "encoding": enc, "data": None})


# ------------------------------------------------------------------------------- encodings

def _codec_with(sizes, limits=None, weights="full"):
    codec = C.TensorCodec(limits) if limits else C.TensorCodec()
    for i, n in enumerate(sizes):
        codec.add(_floats(n, seed=i), f"B{i}.weight", f"B{i}", "weight",
                  ["unit"])
    return {k: v["encoding"] for k, v in codec.tensors(weights).items()}, codec


def test_encoding_switches_at_the_real_limits():
    lim = C.LIMITS
    sizes = [lim["json_list_max"], lim["json_list_max"] + 1, lim["inline_tensor_max"],
             lim["inline_tensor_max"] + 1]
    got, codec = _codec_with(sizes)
    assert list(got.values()) == ["json", "base64-le", "base64-le", "deferred"]
    for i, payload in enumerate(codec.tensors().values()):
        _valid(TENSOR, payload)
        assert payload["shape"] == [sizes[i]] and payload["summary"]["n"] == sizes[i]
        if payload["encoding"] == "json":               # 9-digit JSON lists, bit for bit
            payload = json.loads(json.dumps(payload))
        if payload["encoding"] != "deferred":
            want = _floats(sizes[i], seed=i)
            assert np.array_equal(_bits(C.decode(payload)), _bits(want))
        else:
            assert payload["data"] is None
    quarter = lim["inline_total_max"] // 4             # four fill the inline total exactly
    got, _ = _codec_with([quarter] * 4 + [lim["json_list_max"] + 1, lim["json_list_max"]])
    assert list(got.values()) == ["base64-le"] * 4 + ["deferred", "json"]


def test_encoding_running_total_counts_json_lists_in_node_order():
    small = {**C.LIMITS, "json_list_max": 4, "inline_tensor_max": 16, "inline_total_max": 20}
    assert list(_codec_with([4, 16, 5, 3], small)[0].values()) == \
        ["json", "base64-le", "deferred", "json"]
    assert list(_codec_with([4, 4, 16], small)[0].values()) == ["json", "json", "deferred"]
    assert list(_codec_with([16, 17, 4], small)[0].values()) == ["base64-le", "deferred", "json"]
    got, codec = _codec_with([4, 16, 100], small, weights="summary")
    assert set(got.values()) == {"withheld"}
    for payload in codec.tensors("summary").values():
        _valid(TENSOR, payload)
        assert payload["data"] is None and payload["summary"]["n"] > 0
    with pytest.raises(C.CodecError):
        codec.tensors("raw")


def test_tensor_ids_roles_and_axes_are_checked():
    codec = C.TensorCodec()
    w = torch.zeros(2, 3)
    assert codec.add(w, "B3.r0.1.weight", "B3.r0.1", "weight", ["out", "in"]) == "B3.r0.1.weight"
    assert codec.add(w[0], "B1@2.table.zip", "B1@2", "table", ["dim"]) == "B1@2.table.zip"
    for tid, node, role, axes in [("B1 weight", "B1", "weight", ["o", "i"]),
                                  ("B1." + "w" * 126, "B1", "weight", ["o", "i"]),
                                  ("B2.weight", "B1", "weight", ["o", "i"]),
                                  ("B3.r0.1.weight", "B3.r0.1", "weight", ["o", "i"]),
                                  ("B4.weight", "B4", "gain", ["o", "i"]),
                                  ("B5.weight", "B5", "weight", ["o"])]:
        with pytest.raises(C.CodecError):
            codec.add(w, tid, node, role, axes)
    assert codec.ids() == ("B3.r0.1.weight", "B1@2.table.zip")
    centre = C.TensorCodec()
    centre.add(torch.ones(4), "B9.centre", "B9", "buffer", ["unit"])
    _valid(TENSOR, centre.tensors()["B9.centre"])


def test_fetch_is_base64_from_the_live_tensor_through_an_lru_of_eight():
    codec = C.TensorCodec()
    live = [torch.full((3,), float(i)) for i in range(10)]
    for i, t in enumerate(live):
        codec.add(t, f"B{i}.bias", f"B{i}", "bias", ["unit"])
    first = codec.fetch("B0.bias")
    _valid(TENSOR, first)
    assert first["encoding"] == "base64-le" and np.array_equal(C.decode(first), [0, 0, 0])
    live[0].fill_(7.0)                                  # a cached answer is kept ...
    assert np.array_equal(C.decode(codec.fetch("B0.bias")), [0, 0, 0])
    for i in range(1, 9):                               # ... until 8 others evict it
        codec.fetch(f"B{i}.bias")
    assert np.array_equal(C.decode(codec.fetch("B0.bias")), [7, 7, 7])
    with pytest.raises(C.TensorUnavailable, match="no tensor"):
        codec.fetch("B99.bias")
    with pytest.raises(C.TensorUnavailable, match="does not publish raw weights"):
        codec.fetch("B1.bias", "summary")
    answers, lock = [], threading.Lock()

    def worker():
        for i in range(10):
            body = _dumps(codec.fetch(f"B{i}.bias"))
            with lock:
                answers.append((i, body))
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(body == _dumps(codec.fetch(f"B{i}.bias")) for i, body in answers)


def test_payloads_are_the_callers_own_copies():
    """Editing an answer never reaches the cache, the registered summary or another caller."""
    codec = C.TensorCodec()
    codec.add(torch.arange(6.0).reshape(2, 3), "B1.weight", "B1", "weight", ["out", "in"])
    want_fetch, want_view = _dumps(codec.fetch("B1.weight")), _dumps(codec.tensors())
    got = codec.fetch("B1.weight")                      # a cache hit
    got["summary"]["hist"]["counts"][0] = -1
    got["summary"]["abs_q"].append(0.0)
    got["shape"].append(9)
    view = codec.tensors()
    view["B1.weight"]["summary"]["mean"] = 1e9
    view["B1.weight"]["data"].append(0.0)
    assert _dumps(codec.fetch("B1.weight")) == want_fetch
    assert _dumps(codec.tensors()) == want_view


# ------------------------------------------------------------------------------- summaries

def _values(seed=5, shape=(37, 29)):
    return (_rng(seed).standard_normal(shape) * 3).astype(np.float32)


def test_summary_is_the_contract_statistics():
    a = _values()
    s = C.summary(a)
    _valid(SUMMARY, s)
    f = a.astype(np.float64).ravel()
    assert s["n"] == f.size
    for key, want in (("mean", f.mean()), ("std", f.std()), ("min", f.min()), ("max", f.max()),
                      ("l2", np.sqrt((f * f).sum()))):
        assert s[key] == pytest.approx(want, rel=1e-8, abs=1e-12), key
    assert s["abs_q"] == pytest.approx(np.quantile(np.abs(f), [0.5, 0.9, 0.99, 1.0]), rel=1e-8)
    assert len(s["hist"]["counts"]) == 32 and sum(s["hist"]["counts"]) == f.size
    assert (s["hist"]["lo"], s["hist"]["hi"]) == (s["min"], s["max"])
    flat = C.summary(np.full(10, 2.5, dtype=np.float32))
    assert flat["hist"]["counts"] == [10] + [0] * 31 and flat["std"] == 0.0
    empty = C.summary(np.zeros((0, 3), dtype=np.float32))
    _valid(SUMMARY, empty)
    assert empty["n"] == 0 and empty["hist"]["counts"] == [0] * 32


def test_summaries_are_deterministic_byte_for_byte():
    a = _values()
    want = _dumps(C.summary(a))
    views = [a, a.copy(), np.asfortranarray(a), torch.from_numpy(a.copy()),
             torch.from_numpy(np.ascontiguousarray(a.T)).T, a.astype(np.float64)]
    for v in views:
        assert _dumps(C.summary(v)) == want
    assert _dumps(C.summary(a)) == want                 # a second call
    code = ("import json, sys, numpy as np\n"
            f"sys.path.insert(0, {_RENDER.name!r})\n"
            f"from {PACKAGE}.serving.inspection import codec\n"
            "a = (np.random.default_rng(5).standard_normal((37, 29)) * 3).astype(np.float32)\n"
            "sys.stdout.write(json.dumps(codec.summary(a), sort_keys=True, separators=(',', ':'),"
            " allow_nan=False))\n")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, check=True)
    assert out.stdout == want, "a fresh interpreter with one thread computed other bytes"


def test_network_tensors_are_deterministic_byte_for_byte():
    def build():
        g = torch.Generator().manual_seed(11)
        codec = C.TensorCodec()
        for i, shape in enumerate([(64, 9), (64,), (2000, 3), (300_000,)]):
            codec.add(torch.randn(shape, generator=g), f"B{i}.weight", f"B{i}", "weight",
                      ["unit"] * len(shape))
        return codec
    one, two = build(), build()
    for weights in ("full", "summary"):
        assert _dumps(one.tensors(weights)) == _dumps(two.tensors(weights))


# ------------------------------------------------------------------------------- heat

def _naive_heat(a, limit, how):
    def edges(n):
        return list(range(n + 1)) if n <= limit else [i * n // limit for i in range(limit + 1)]
    re_, ce = edges(a.shape[0]), edges(a.shape[1])
    out = np.zeros((len(re_) - 1, len(ce) - 1))
    for i in range(len(re_) - 1):
        for j in range(len(ce) - 1):
            block = a[re_[i]:re_[i + 1], ce[j]:ce[j + 1]].astype(np.float64)
            out[i, j] = block.sum() if how == "sum" else block.mean()
    return re_, ce, out


@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (64, 64), (65, 64), (100, 3), (3, 100),
                                   (200, 300), (129, 1000)])
@pytest.mark.parametrize("how", ["mean", "sum"])
def test_heat_block_reductions(shape, how):
    a = _values(seed=shape[0] * 7 + shape[1], shape=shape)
    h = R.Reducer().heat(a, how)
    _valid(HEAT, h)
    re_, ce, want = _naive_heat(a, 64, how)
    assert (h["row_edges"], h["col_edges"]) == (re_, ce)
    assert (h["rows"], h["cols"]) == (len(re_) - 1, len(ce) - 1)
    assert h["rows"] <= 64 and h["cols"] <= 64
    assert np.allclose(np.asarray(h["values"]), want, rtol=1e-7, atol=1e-9)
    if shape[0] <= 64 and shape[1] <= 64:               # the identity at or under heat_max
        assert np.array_equal(np.asarray(h["values"], dtype=np.float32), a)
    if how == "sum":                                    # a relevance map keeps its total
        assert np.asarray(h["values"]).sum() == pytest.approx(a.astype(np.float64).sum(),
                                                              rel=1e-6, abs=1e-6)


def test_heat_edges_and_empty_axes():
    assert R.block_edges(65, 64)[:3] == [0, 1, 2] and R.block_edges(65, 64)[-2:] == [63, 65]
    assert R.block_edges(10, 64) == list(range(11))
    e = R.block_edges(1000, 64)
    assert e[0] == 0 and e[-1] == 1000 and all(b > a for a, b in zip(e, e[1:]))
    empty = R.Reducer().heat(np.zeros((0, 5)))
    _valid(HEAT, empty)
    assert empty["rows"] == 0 and empty["values"] == []
    with pytest.raises(C.CodecError):
        R.Reducer().heat(np.zeros(5))
    with pytest.raises(C.CodecError):
        R.Reducer().heat(np.zeros((2, 2)), "max")


# ------------------------------------------------------------------------------- reduced forms

def test_reduced_forms_validate_and_mean_what_they_say():
    r, rng = R.Reducer(), _rng(9)
    forms = {
        "scalar": r.reduce(torch.tensor(1.5)),
        "vector": r.reduce(rng.standard_normal(4096).astype(np.float32)),
        "top": r.reduce(rng.standard_normal(4097).astype(np.float32)),
        "sequence": r.reduce(rng.standard_normal((12, 5)).astype(np.float32)),
        "folded": r.reduce(rng.standard_normal((3, 4, 5)).astype(np.float32)),
        "attention": r.reduce(torch.softmax(torch.randn(2, 70, 70), -1), "attention"),
        "pool": r.reduce(torch.softmax(torch.randn(12), 0), "attention"),
        "nodes": r.nodes(rng.standard_normal((40, 6)).astype(np.float32), [3, 1, 7], [0, 1, 2], 3),
        "summary": r.summarise(rng.standard_normal((100, 7)).astype(np.float32),
                               ["step", "channel"]),
        "relevance": r.reduce(rng.standard_normal((80, 3)).astype(np.float32), "relevance"),
    }
    for name, form in forms.items():
        _valid(REDUCED, form)
    assert forms["scalar"] == {"form": "scalar", "value": 1.5}
    assert forms["vector"]["n"] == 4096 and len(forms["vector"]["values"]) == 4096
    assert forms["top"]["n"] == 4097 and len(forms["top"]["top"]) == 64
    assert forms["folded"]["steps"] == 12 and forms["folded"]["width"] == 5
    assert (forms["attention"]["heads"], forms["attention"]["q"]) == (2, 70)
    assert forms["attention"]["maps"][0]["rows"] == 64
    assert (forms["pool"]["q"], forms["pool"]["k"]) == (1, 12)
    assert all(len(v) <= 64 for part in ("norms", "sums") for v in forms["summary"][part].values())


def test_vector_top_is_the_largest_magnitudes_with_stable_ties():
    a = np.zeros(5000, dtype=np.float32)
    a[[4999, 10, 20]] = [-9.0, 9.0, 9.0]
    a[100:170] = 1.0
    top = R.Reducer().vector(a)["top"]
    assert top[:3] == [[10, 9.0], [20, 9.0], [4999, -9.0]]
    assert [i for i, _ in top[3:]] == list(range(100, 161))


def test_sequence_statistics_and_relevance_sums():
    a = _values(seed=21, shape=(80, 3))
    f = a.astype(np.float64)
    seq = R.Reducer().sequence(a)
    assert np.allclose(seq["per_step"]["norm"], np.sqrt((f * f).sum(1)), rtol=1e-8)
    assert np.allclose(seq["per_step"]["mean"], f.mean(1), rtol=1e-8, atol=1e-12)
    assert np.allclose(seq["per_channel"]["mean"], f.mean(0), rtol=1e-8)
    assert np.allclose(seq["per_channel"]["absmax"], np.abs(f).max(0), rtol=1e-8)
    rel = R.Reducer().reduce(a, "relevance")
    assert np.asarray(rel["heat"]["values"]).sum() == pytest.approx(f.sum(), rel=1e-6)
    sizes = np.outer(np.diff(seq["heat"]["row_edges"]), np.diff(seq["heat"]["col_edges"]))
    assert np.average(seq["heat"]["values"], weights=sizes) == pytest.approx(f.mean(), rel=1e-6)
    with pytest.raises(C.CodecError):
        R.Reducer().reduce(a, "gradient")


def test_nodes_are_capped_and_summaries_keep_totals():
    rng = _rng(4)
    a = rng.standard_normal((600, 4)).astype(np.float32)
    ids = list(range(599, -1, -1))
    form = R.Reducer().nodes(a, ids, [1] * 600, 5)
    assert form["ids"] == ids[:256] and len(form["norm"]) == 256
    assert form["target"] == {"id": 5, "values": C.nums(a[5])}
    with pytest.raises(C.CodecError):
        R.Reducer().nodes(a, [600], [0], 0)
    with pytest.raises(C.CodecError):
        R.Reducer().nodes(a, [1], [-1], 0)
    far = list(range(300)) + [7, 8]                      # the cap keeps the nearer hops first
    hops = [2] * 300 + [1, 0]
    near = R.Reducer().nodes(a, far, hops, 8)
    assert near["ids"][:3] == [8, 7, 0] and near["hop"][:3] == [0, 1, 2]
    assert near["ids"][2:] == list(range(254)), "the order within a hop is the caller's"
    t = rng.standard_normal((300, 5)).astype(np.float64)
    s = R.Reducer().summarise(t, ["step", "channel"])
    for axis in ("step", "channel"):
        assert sum(s["sums"][axis]) == pytest.approx(t.sum(), rel=1e-6)
        assert math.sqrt(sum(v * v for v in s["norms"][axis])) == \
            pytest.approx(np.sqrt((t * t).sum()), rel=1e-6)
    assert R.Reducer().summarise(np.float32(2.0)) == {"form": "summary", "norms": {"value": [2.0]},
                                                      "sums": {"value": [2.0]}}
    with pytest.raises(C.CodecError):
        R.Reducer().summarise(t, ["step"])


# ------------------------------------------------------------------------------- node payloads

def test_node_payload_has_every_key_and_refuses_what_the_schema_refuses():
    p = R.node_payload("full", {"out": R.Reducer().reduce(torch.ones(3)), "recompute": 1e-9})
    _valid(NODE, p)
    assert set(p) == {"level", "why", *R.PARTS} and p["extra"] == {} and p["exact"] is None
    x = torch.ones(2, 2)
    exact = R.node_payload("exact", {"exact": C.exact_set({"in": (x, ["a", "b"]),
                                                          "pre": (x[0], ["unit"])})})
    _valid(NODE, exact)
    for level, parts, why in [("full", {"gradient": 1}, None), ("exact", {}, None),
                              ("full", {"exact": {"in": None, "out": None, "grad_out": None}},
                               None), ("omitted", {}, None), ("partial", {}, None),
                              ("full", {"recompute": float("nan")}, None)]:
        with pytest.raises(C.CodecError):
            R.node_payload(level, parts, why)


# ------------------------------------------------------------------------------- the budget

def _parts(k):
    """Parts holding exactly `k` numbers (a vector form of k − 1 values)."""
    return {"out": {"form": "vector", "n": k - 1, "values": [0.0] * (k - 1)}}


def _node(sizes, calls):
    def build(level):
        calls.append(level)
        return _parts(sizes[level])
    return build


def test_budget_degrades_to_summary_then_omitted():
    b, calls = R.Budget("full"), []
    limit = C.LIMITS["explain_float_budget"]
    plan = [("A", {"full": 200_000, "summary": 50}, "full"),
            ("B", {"full": 100_000, "summary": 100}, "summary"),
            ("C", {"full": 1_000, "summary": 10}, "full"),            # per node, not sticky
            ("D", {"full": limit - 201_100, "summary": 10}, "full"),  # fills it exactly
            ("E", {"full": 10, "summary": 5}, "omitted"),
            ("F", {"full": 1, "summary": 1}, "omitted")]
    nodes = {}
    for node_id, sizes, level in plan:
        nodes[node_id] = b.place(node_id, _node(sizes, calls))
        _valid(NODE, nodes[node_id])
        assert nodes[node_id]["level"] == level, node_id
    assert calls == ["full", "full", "summary", "full", "full", "full", "summary", "full",
                     "summary"]
    assert nodes["B"]["why"].startswith("summarised") and nodes["A"]["why"] is None
    assert nodes["E"]["why"].startswith("omitted") and nodes["E"]["out"] is None
    report = b.report()
    _valid(BUDGET, report)
    assert report == {"limit": limit, "used": limit, "summarised": ["B"], "omitted": ["E", "F"]}
    assert b.degraded and b.used == sum(R.count(p) for p in nodes.values())


def test_budget_summary_detail_puts_every_node_at_summary():
    b, calls = R.Budget("summary"), []
    for node_id in ("A", "B"):
        assert b.place(node_id, _node({"summary": 30}, calls))["level"] == "summary"
    assert calls == ["summary", "summary"] and not b.degraded
    assert b.report() == {"limit": C.LIMITS["explain_float_budget"], "used": 60,
                          "summarised": [], "omitted": []}
    tiny = R.Budget("summary", {**C.LIMITS, "explain_float_budget": 40})
    assert tiny.place("A", _node({"summary": 30}, []))["level"] == "summary"
    assert tiny.place("B", _node({"summary": 30}, []))["level"] == "omitted"
    assert tiny.degraded and tiny.report()["omitted"] == ["B"]
    with pytest.raises(C.CodecError):
        R.Budget("everything")


def test_exact_budget_summarises_the_rest_and_sets_degraded():
    limit = C.LIMITS["explain_exact_float_budget"]
    assert R.Budget("exact").limit == limit == 4_194_304
    g = torch.Generator().manual_seed(2)
    big = torch.randn(1000, 1000, generator=g)
    rest = torch.randn(limit - 3_000_000 + 1, generator=g)         # one number too many
    calls = []

    def exact_node(tensors, name):
        def build(level):
            calls.append((name, level))
            if level == "exact":
                return {"exact": C.exact_set(tensors), "recompute": 0.0}
            return {"out": R.Reducer().summarise(tensors["out"][0], tensors["out"][1])}
        return build
    b = R.Budget("exact")
    a = b.place("A", exact_node({"in": (big, ["r", "c"]), "out": (big, ["r", "c"]),
                                 "grad_out": (big, ["r", "c"])}, "A"))
    assert a["level"] == "exact" and b.used == 3_000_001 and not b.degraded
    assert R.count(a["exact"]["in"]) == 1_000_000
    bn = b.place("B", exact_node({"out": (rest, ["unit"])}, "B"))
    cn = b.place("C", exact_node({"out": (torch.ones(3), ["unit"])}, "C"))
    assert (bn["level"], cn["level"]) == ("summary", "summary")
    assert ("C", "exact") not in calls, "a node after the spent exact budget was built exact"
    assert bn["exact"] is None and "exact budget" in cn["why"]
    for p in (a, bn, cn):
        _valid(NODE, p)
    assert b.degraded and b.report()["summarised"] == ["B", "C"]
    fits = R.Budget("exact")
    assert fits.place("A", exact_node({"out": (torch.ones(8), ["unit"])}, "A"))["level"] == "exact"
    assert not fits.degraded


def test_explanations_are_deterministic_byte_for_byte():
    def explain():
        g = torch.Generator().manual_seed(7)
        r, b = R.Reducer(), R.Budget("full", {**C.LIMITS, "explain_float_budget": 5_000})
        nodes = {}
        for i, shape in enumerate([(4097,), (30, 70), (2, 9, 9), (), (500, 20)]):
            t = torch.randn(shape, generator=g)
            axes = [f"a{j}" for j in range(t.dim())]
            nodes[f"B{i}"] = b.place(f"B{i}", lambda lvl, t=t, axes=axes: {
                "out": r.reduce(t) if lvl == "full" else r.summarise(t, axes),
                "relevance": r.reduce(t * t, "relevance") if lvl == "full" else None})
        return _dumps({"nodes": nodes, "budget": b.report(), "degraded": b.degraded})
    assert explain() == explain()


# ------------------------------------------------------------------------------- the gate

STDLIB = {"__future__", "base64", "collections", "copy", "math", "re", "threading", "types",
          "typing"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_size_discipline_state_and_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(TEMPLATES).as_posix()
    assert conformance.size_findings(tree, rel) == [], "C11–C13 must hold at error"
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if isinstance(node.value, conformance.MUTABLE_NODES) and names != ["__all__"]:
                assert all(n.lstrip("_").isupper() for n in names), f"C7: {names} at {rel}"
        if isinstance(node, ast.Import):
            roots = {a.name.split(".")[0] for a in node.names}
            assert roots <= STDLIB | {"numpy", "torch"}, f"{rel} imports {roots}"
        if isinstance(node, ast.ImportFrom) and not node.level:
            assert node.module.split(".")[0] in STDLIB | {"numpy", "torch"}, \
                f"{rel} imports {node.module}"
    assert "{{" not in path.read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
