#!/usr/bin/env python3
"""
test_export_parity.py — the ModelStore's two halves on the emitted package (WP B14; D1 §13,
BUILD-PLAN §2.4): `ModelRegistry` (`<model>/store.py`, the adapter) and `TorchExporter` +
`BatchSignature` (`<model>/export.py`, the `ArtifactExporter`).

    python3 -m pytest -q vega-forge/scripts/tests/test_export_parity.py

Every package is emitted by the real emitter from the live templates (`render_templates.py
--full`, which is exactly `emit()`), one per sample shape, from the reference specs. What it
proves:

1. **TorchScript parity for row, window, graph node and graph level.** The network is exported
   on one batch and the traced module reproduces it within 1e-5 on that batch and on a batch of
   another size — other rows, windows, nodes, edges, targets and graphs (at graph level
   `num_graphs` travels as a 0-d tensor, so it is not frozen into the trace). A weighted graph
   exports `edge_weight` in its place; a nominal column exports `x_cat`. The maximum reductions
   (gat_conv's attention softmax, sage_conv's `max` aggregator, graph_pool's `max` readout)
   trace too, isolated nodes and edgeless batches included.
2. **The positional order** is backend-contract §1's per shape, the optional inputs (`x_cat`,
   `edge_weight`) kept exactly when the example batch carries them; the signature travels inside
   the TorchScript file, and a batch the export cannot take is refused by name.
3. **The export records what the exporter warned** (`torch.jit.trace` is deprecated from torch
   2.9 on) and leaves no half-written file when it fails.
4. **ONNX is attempted at opset 18**: with `onnx` installed every sample shape (row, window,
   graph node plain, weighted and with maximum aggregations, graph level with mean and max
   readouts) exports a file that declares opset 18 with the signature's inputs, and with
   `onnxruntime` it answers the network within tolerance on two batch sizes (graph messages
   with repeated destinations, isolated nodes); the ScatterElements lowerings that make the
   graph reductions exact are registered for the export call only. Without `onnx` the export
   fails with exactly the missing-module error (nothing else passes for it) and leaves no file,
   and `/vega-pack`'s export phase ships the bundle with a `WARN:` and a `failed` record rather
   than a broken file.
5. **ModelRegistry** (C14: one class): `save` → `load` restores parameters **and buffers** (the
   one-class centre `svdd_center` / `svdd_center_set`, the batch-norm statistics), the weights
   digest covers the buffers, a tampered file, a pinned digest that does not match, a checkpoint
   of another network and a spec of another digest are refused by name; `version()` reads a
   stored file's identity without building anything.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import warnings
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parents[1]
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
PACK_SCRIPTS = REPO / "vega-pack" / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))

import render_templates as rt  # noqa: E402

ATOL = 1e-5
_KEEP = []


# ================================================================ the emitted packages

def _package(tag: str, spec: dict):
    """The full v2.3 package of `spec` (the real emitter over the live templates), as package
    `b14_<tag>`, imported; returns a namespace of its modules and the spec."""
    work = tempfile.TemporaryDirectory(prefix=f"b14-export-{tag}-")
    _KEEP.append(work)
    package = f"b14_{tag.replace('-', '_')}"
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = package
    rec = rt.render(TEMPLATES, Path(work.name), pipeline, spec, full=True)
    sys.path.insert(0, work.name)
    importlib.invalidate_caches()
    ns = types.SimpleNamespace(root=Path(rec["root"]), spec=spec, package=package,
                               files=rec["files"])
    for key, dotted in (("domain", "domain"), ("network", "model.network"),
                        ("export", "model.export"), ("store", "model.store")):
        setattr(ns, key, importlib.import_module(f"{package}.{dotted}"))
    return ns


def _row_spec():
    return rt.reference_spec("row")                  # 4 numerics + zone (5), binary


def _window_spec():
    return rt.reference_spec("window")               # 2 numerics + the autoregressive channel


def _node_spec():
    return rt.reference_spec("graph")                # gcn, level node, multiclass


def _level_spec():
    return rt.reference_spec("graph-level")          # gin + graph_pool, binary


def _attention_spec():
    """Node level, every aggregation a maximum: gat_conv's attention softmax (a max per
    destination, `scatter_reduce` without the initial values) and sage_conv's `max`
    aggregator (an isolated node reads 0)."""
    spec = _node_spec()
    spec["family"] = {"name": "gat", "source": "user:reference"}
    spec["blocks"] = [
        {"id": "B1", "type": "gat_conv", "params": {"out": 8, "heads": 2}},
        {"id": "B2", "type": "activation", "params": {"name": "relu"}},
        {"id": "B3", "type": "sage_conv", "params": {"out": 16, "aggregator": "max"}},
        {"id": "B4", "type": "head", "params": {"out": 4, "activation": "none"}}]
    return spec


def _level_max_spec():
    """Graph level with graph_pool's `max` readout."""
    spec = _level_spec()
    spec["blocks"] = [dict(b, params=dict(b["params"], readout="max"))
                      if b["type"] == "graph_pool" else b for b in spec["blocks"]]
    return spec


@pytest.fixture(scope="module")
def pkgs():
    return types.SimpleNamespace(row=_package("row", _row_spec()),
                                 window=_package("window", _window_spec()),
                                 node=_package("node", _node_spec()),
                                 level=_package("level", _level_spec()),
                                 attention=_package("attention", _attention_spec()),
                                 level_max=_package("level-max", _level_max_spec()))


def _net(ns, **change):
    """The package's network, built by its one constructor from the spec (with `change`)."""
    net = ns.network.Network.build(ns.domain.NetSpec.from_dict(dict(ns.spec, **change)))
    return net.eval()


# ================================================================ batches of each shape

def _g(seed):
    return torch.Generator().manual_seed(seed)


def _row_batch(n, seed):
    g = _g(seed)
    return {"x_num": torch.randn(n, 4, generator=g),
            "x_cat": torch.randint(0, 5, (n, 1), generator=g)}


def _window_batch(n, seed):
    return {"x_num": torch.randn(n, 12, 3, generator=_g(seed))}


def _edges(n_nodes, n_edges, g, lo=0):
    src = torch.randint(lo, lo + n_nodes, (n_edges,), generator=g)
    dst = torch.randint(lo, lo + n_nodes, (n_edges,), generator=g)
    return torch.stack([src, dst])


def _node_batch(n, e, t, seed, weighted=False):
    g = _g(seed)
    batch = {"x_num": torch.randn(n, 3, generator=g), "edge_index": _edges(n, e, g),
             "target_index": torch.randperm(n, generator=g)[:t]}
    if weighted:
        batch["edge_weight"] = torch.rand(e, generator=g) + 0.5
    return batch


def _level_batch(sizes, seed):
    """Graphs of the given node counts, a few edges inside each, relabelled as GraphCollate
    does (nodes of graph k contiguous, graph_index k)."""
    g = _g(seed)
    edges, index, first = [], [], 0
    for k, size in enumerate(sizes):
        edges.append(_edges(size, 2 * size, g, lo=first))
        index += [k] * size
        first += size
    return {"x_num": torch.randn(first, 3, generator=g), "edge_index": torch.cat(edges, dim=1),
            "graph_index": torch.tensor(index), "num_graphs": len(sizes)}


def _export(ns, net, batch, tmp_path, fmt="torchscript"):
    path = tmp_path / f"{ns.package}.{fmt}"
    written = ns.export.TorchExporter().export(net, ns.spec, fmt, path, batch)
    return path, written


def _parity(ns, net, path, batches):
    exporter = ns.export.TorchExporter()
    for batch in batches:
        result = exporter.parity(net, path, batch, atol=ATOL)
        assert result["pass"], result
        assert result["max_abs_diff"] <= ATOL


# ================================================================ 1. TorchScript parity

def test_torchscript_parity_row(pkgs, tmp_path):
    net = _net(pkgs.row)
    path, written = _export(pkgs.row, net, _row_batch(5, 1), tmp_path)
    assert written["inputs"] == ["x_num", "x_cat"] and written["format"] == "torchscript"
    _parity(pkgs.row, net, path, [_row_batch(5, 1), _row_batch(17, 2), _row_batch(1, 3)])


def test_torchscript_parity_window(pkgs, tmp_path):
    net = _net(pkgs.window)
    path, written = _export(pkgs.window, net, _window_batch(6, 1), tmp_path)
    assert written["inputs"] == ["x_num"]
    _parity(pkgs.window, net, path, [_window_batch(6, 1), _window_batch(3, 2),
                                     _window_batch(11, 3)])


def test_torchscript_parity_graph_node(pkgs, tmp_path):
    net = _net(pkgs.node)
    path, written = _export(pkgs.node, net, _node_batch(20, 50, 6, 1), tmp_path)
    assert written["inputs"] == ["x_num", "edge_index", "target_index"]
    _parity(pkgs.node, net, path, [_node_batch(20, 50, 6, 1), _node_batch(35, 90, 11, 2),
                                   _node_batch(8, 0, 8, 3)])


def test_torchscript_parity_graph_node_weighted(pkgs, tmp_path):
    """A weighted edges part: `edge_weight` is exported in its place and read by every conv."""
    net = _net(pkgs.node)
    path, written = _export(pkgs.node, net, _node_batch(20, 50, 6, 1, weighted=True), tmp_path)
    assert written["inputs"] == ["x_num", "edge_index", "edge_weight", "target_index"]
    _parity(pkgs.node, net, path, [_node_batch(20, 50, 6, 1, weighted=True),
                                   _node_batch(30, 70, 9, 4, weighted=True)])


def test_torchscript_parity_graph_level(pkgs, tmp_path):
    net = _net(pkgs.level)
    path, written = _export(pkgs.level, net, _level_batch([4, 6, 5], 1), tmp_path)
    assert written["inputs"] == ["x_num", "edge_index", "graph_index", "num_graphs"]
    _parity(pkgs.level, net, path, [_level_batch([4, 6, 5], 1), _level_batch([3, 7, 5, 8, 2], 2),
                                    _level_batch([9], 3)])


def test_torchscript_parity_graph_maxima(pkgs, tmp_path):
    """The maximum reductions (gat_conv's attention softmax, sage_conv's `max` aggregator,
    graph_pool's `max` readout) trace too, isolated nodes and edgeless batches included."""
    net = _net(pkgs.attention)
    path, _ = _export(pkgs.attention, net, _node_batch(20, 50, 6, 1), tmp_path)
    _parity(pkgs.attention, net, path, [_node_batch(20, 50, 6, 1), _node_batch(35, 12, 11, 2),
                                        _node_batch(8, 0, 8, 3)])
    net = _net(pkgs.level_max)
    path, _ = _export(pkgs.level_max, net, _level_batch([4, 6, 5], 1), tmp_path)
    _parity(pkgs.level_max, net, path, [_level_batch([4, 6, 5], 1), _level_batch([3, 7, 1], 2)])


def test_the_export_is_the_networks_forward_not_the_readout(pkgs, tmp_path):
    """The traced graph answers the raw head (the loss input), aligned with `y`: at node level
    the targets' rows, gathered once."""
    net = _net(pkgs.node)
    batch = _node_batch(20, 50, 6, 1)
    path, _ = _export(pkgs.node, net, batch, tmp_path)
    module = torch.jit.load(str(path))
    signature = pkgs.node.export.BatchSignature.for_sample(net.sample, batch)
    got = module(*signature.inputs(batch))
    assert tuple(got.shape) == (6, 4)
    assert torch.allclose(got, net(batch), atol=ATOL)


# ================================================================ 2. the positional order

def test_the_signature_per_shape(pkgs):
    """Each shape's signature, read by the package of that shape: a package carries its spec's
    sample shape only, and the row package refuses a window spec by name."""
    sig = pkgs.row.export.BatchSignature.for_sample
    D = pkgs.row.domain
    assert sig(D.SampleSpec.from_spec(_row_spec()), _row_batch(2, 0)).keys == ("x_num", "x_cat")
    assert sig(_row_spec(), {"x_num": torch.zeros(2, 4)}).keys == ("x_num",)
    with pytest.raises(D.UnsupportedShapeError, match="'window'"):
        sig(_window_spec(), _window_batch(2, 0))
    sig = pkgs.window.export.BatchSignature.for_sample
    assert sig(_window_spec(), _window_batch(2, 0)).keys == ("x_num",)
    sig = pkgs.node.export.BatchSignature.for_sample
    node = sig(_node_spec(), _node_batch(5, 6, 2, 0, weighted=True))
    assert node.keys == ("x_num", "edge_index", "edge_weight", "target_index")
    assert node.dynamic_axes() == {"x_num": {0: "nodes"}, "edge_index": {1: "edges"},
                                   "edge_weight": {0: "edges"}, "target_index": {0: "targets"},
                                   "out": {0: "targets"}}
    sig = pkgs.level.export.BatchSignature.for_sample
    level = sig(_level_spec(), _level_batch([2, 3], 0))
    assert level.keys == ("x_num", "edge_index", "graph_index", "num_graphs")
    assert level.dynamic_axes()["out"] == {0: "graphs"} and "num_graphs" not in \
        level.dynamic_axes()
    args = level.inputs(_level_batch([2, 3], 0))
    assert args[-1].dim() == 0 and args[-1].dtype == torch.long and int(args[-1]) == 2


def test_the_signature_refuses_what_it_cannot_take(pkgs):
    E = pkgs.node.export
    with pytest.raises(ValueError, match="target_index"):
        E.BatchSignature.for_sample(_node_spec(), {"x_num": torch.zeros(3, 3),
                                                   "edge_index": torch.zeros(2, 0).long()})
    sig = E.BatchSignature.for_sample(_node_spec(), _node_batch(5, 6, 2, 0))
    with pytest.raises(ValueError, match="edge_weight"):
        sig.inputs(_node_batch(5, 6, 2, 0, weighted=True))
    with pytest.raises(ValueError, match="optional"):
        E.BatchSignature("node", ("x_num", "target_index", "edge_index"))
    with pytest.raises(ValueError, match="unknown batch form"):
        E.BatchSignature("cube", ("x_num",))


def test_the_signature_travels_inside_the_torchscript_file(pkgs, tmp_path):
    net = _net(pkgs.level)
    path, _ = _export(pkgs.level, net, _level_batch([3, 4], 1), tmp_path)
    extra = {"vega-signature.json": ""}
    torch.jit.load(str(path), _extra_files=extra)
    saved = json.loads(extra["vega-signature.json"])
    assert saved == {"form": "graph", "inputs": ["x_num", "edge_index", "graph_index",
                                                 "num_graphs"], "output": "graphs"}
    stranger = tmp_path / "stranger.pt"
    torch.jit.trace(torch.nn.Linear(3, 1), torch.zeros(2, 3)).save(str(stranger))
    with pytest.raises(ValueError, match="vega-signature.json"):
        pkgs.level.export.TorchExporter().parity(net, stranger, _level_batch([3], 2))


# ================================================================ 3. warnings, failures

def test_the_export_records_the_exporters_warnings(pkgs, tmp_path):
    net = _net(pkgs.row)
    _, written = _export(pkgs.row, net, _row_batch(4, 1), tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.jit.trace(torch.nn.Linear(2, 1), torch.zeros(1, 2))
    deprecated = [w for w in caught if "jit" in str(w.message).lower()
                  and issubclass(w.category, FutureWarning)]
    if deprecated:                                   # torch >= 2.9 says it is deprecated
        assert any("torch.jit.trace" in w and w.startswith("FutureWarning")
                   for w in written["warnings"]), written["warnings"]
    assert all(isinstance(w, str) for w in written["warnings"])


def test_a_failed_export_leaves_no_file(pkgs, tmp_path):
    E = pkgs.row.export

    class Broken(torch.nn.Module):
        sample = pkgs.row.domain.SampleSpec.from_spec(_row_spec())

        def forward(self, batch):
            raise RuntimeError("a block the tracer refuses")

    path = tmp_path / "broken.pt"
    with pytest.raises(RuntimeError, match="the tracer refuses"):
        E.TorchExporter().export(Broken(), None, "torchscript", path, _row_batch(2, 0))
    assert not path.exists()
    with pytest.raises(ValueError, match="unsupported export format"):
        E.TorchExporter().export(_net(pkgs.row), _row_spec(), "savedmodel", path,
                                 _row_batch(2, 0))
    with pytest.raises(ValueError, match="one real batch"):
        E.TorchExporter().export(_net(pkgs.row), _row_spec(), "torchscript", path, None)


def test_exporting_leaves_the_callers_network_as_it_was(pkgs, tmp_path):
    """The export and the parity check run on the CPU in eval mode and hand the network back in
    its own mode (and on its own device)."""
    net = _net(pkgs.row).train()
    path, _ = _export(pkgs.row, net, _row_batch(4, 1), tmp_path)
    assert net.training
    pkgs.row.export.TorchExporter().parity(net, path, _row_batch(4, 1))
    assert net.training and next(net.parameters()).device.type == "cpu"


def test_parity_reports_a_mismatch_as_a_failure_not_a_crash(pkgs, tmp_path):
    net = _net(pkgs.row)
    path, _ = _export(pkgs.row, net, _row_batch(4, 1), tmp_path)
    other = _net(pkgs.row, seed=99)
    result = pkgs.row.export.TorchExporter().parity(other, path, _row_batch(4, 1), atol=ATOL)
    assert result["pass"] is False and result["max_abs_diff"] > ATOL


# ================================================================ 4. ONNX at opset 18

def _missing_module(exc):
    """The name of the module whose absence `exc` reports (ModuleNotFoundError itself or the
    one it was raised from, as torch.onnx's OnnxExporterError('Module onnx is not
    installed!') is), else None."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ModuleNotFoundError):
            return exc.name
        exc = exc.__cause__ or exc.__context__
    return None


def _onnx_available():
    try:
        import onnx  # noqa: F401
    except ImportError:
        return False
    return True


def test_onnx_without_the_onnx_module_fails_on_that_and_only_that(pkgs, tmp_path):
    """Without `onnx` the export raises the missing-module failure — not any exception, so a
    fault inside the ONNX writer (a bad keyword, a wrong opset argument) still fails here —
    and leaves no file behind."""
    if _onnx_available():
        pytest.skip("onnx is installed: the opset-18 export runs in the tests below")
    net = _net(pkgs.row)
    path = tmp_path / "model.onnx"
    with pytest.raises(Exception) as info:
        pkgs.row.export.TorchExporter().export(net, _row_spec(), "onnx", path, _row_batch(4, 1))
    assert _missing_module(info.value) in ("onnx", "onnxscript"), repr(info.value)
    assert not path.exists(), "a failed ONNX export left a file behind"


# case -> (the package, the traced batch, a batch of another size, the inputs exported); the
# maxima cases carry isolated nodes (a max over no message) and a graph of one node
_ONNX_CASES = {
    "row": ("row", lambda: _row_batch(4, 1), lambda: _row_batch(9, 2), ["x_num", "x_cat"]),
    "window": ("window", lambda: _window_batch(6, 1), lambda: _window_batch(3, 2), ["x_num"]),
    "node": ("node", lambda: _node_batch(20, 50, 6, 1), lambda: _node_batch(35, 90, 11, 2),
             ["x_num", "edge_index", "target_index"]),
    "node-weighted": ("node", lambda: _node_batch(20, 50, 6, 1, weighted=True),
                      lambda: _node_batch(30, 70, 9, 4, weighted=True),
                      ["x_num", "edge_index", "edge_weight", "target_index"]),
    "node-maxima": ("attention", lambda: _node_batch(20, 50, 6, 1),
                    lambda: _node_batch(35, 12, 11, 2), ["x_num", "edge_index", "target_index"]),
    "level": ("level", lambda: _level_batch([4, 6, 5], 1),
              lambda: _level_batch([3, 7, 5, 8, 2], 2),
              ["x_num", "edge_index", "graph_index", "num_graphs"]),
    "level-max": ("level_max", lambda: _level_batch([4, 6, 5], 1),
                  lambda: _level_batch([3, 7, 1], 2),
                  ["x_num", "edge_index", "graph_index", "num_graphs"])}


def _onnx_case(pkgs, case, tmp_path):
    """`case` exported to ONNX: (namespace, network, file, export record, [traced batch, other
    batch], the inputs expected)."""
    key, traced, other, inputs = _ONNX_CASES[case]
    ns = getattr(pkgs, key)
    net = _net(ns)
    path, written = _export(ns, net, traced(), tmp_path, fmt="onnx")
    return ns, net, path, written, [traced(), other()], inputs


@pytest.mark.parametrize("case", sorted(_ONNX_CASES))
def test_onnx_is_exported_at_opset_18(pkgs, tmp_path, case):
    """With `onnx` installed (the matrix and nightly environments): every sample shape exports
    at opset 18 — the graph convolutions, the attention softmax, the maximum aggregations and
    the graph pooling included — with the positional inputs of the signature, and, with
    `onnxruntime`, answers the network within tolerance on the traced batch and on a batch of
    another size (graph messages with repeated destinations, isolated nodes)."""
    if not _onnx_available():
        pytest.skip("onnx is not installed here; the missing-module test above runs instead")
    import onnx
    ns, net, path, written, batches, inputs = _onnx_case(pkgs, case, tmp_path)
    assert written["opset"] == 18 and written["inputs"] == inputs
    model = onnx.load(str(path))
    assert max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")) == 18
    names = [i.name for i in model.graph.input]
    # the graph's inputs keep the signature's order; an input the trace folded away may be
    # absent (the runner feeds by name), and the parity below proves nothing was frozen
    assert names[0] == "x_num" and [k for k in inputs if k in names] == names, names
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return
    exporter = ns.export.TorchExporter()
    for batch in batches:
        result = exporter.parity(net, path, batch, atol=ATOL, fmt="onnx")
        assert result["pass"], result


def test_the_onnx_scatter_lowerings_are_scoped_to_the_export(pkgs, monkeypatch):
    """The ScatterElements lowerings of `index_add` / `scatter_reduce` are registered at opset
    18 for the export call only, and unregistered afterwards whether it succeeded or failed."""
    E = pkgs.node.export
    live = {}
    monkeypatch.setattr(torch.onnx, "register_custom_op_symbolic",
                        lambda name, fn, opset: live.__setitem__(name, (fn, opset)))
    monkeypatch.setattr(torch.onnx, "unregister_custom_op_symbolic",
                        lambda name, opset: live.pop(name))
    with E._scatter_lowerings():
        assert set(live) == {"aten::index_add", "aten::scatter_reduce"}
        assert {opset for _, opset in live.values()} == {E.ONNX_OPSET}
    assert live == {}
    with pytest.raises(RuntimeError, match="the export failed"):
        with E._scatter_lowerings():
            raise RuntimeError("the export failed")
    assert live == {}


def test_the_pack_export_phase_ships_a_failed_format_with_a_warning(pkgs, tmp_path):
    """`vega-pack/scripts/export_pack.py` calls the bundle's TorchExporter: a passing format is
    recorded with its inputs and warnings, a failing one with WARN and no file."""
    sys.path.insert(0, str(PACK_SCRIPTS))
    try:
        import export_pack
    finally:
        sys.path.remove(str(PACK_SCRIPTS))
    net = _net(pkgs.row)
    ctx = types.SimpleNamespace(paths=types.SimpleNamespace(root=tmp_path), markers=[])
    (tmp_path / "exports").mkdir()
    parts = (pkgs.row.export.TorchExporter(), pkgs.row.domain.NetSpec.from_dict(_row_spec()),
             net)
    ok = export_pack._export_one("torchscript", parts, _row_batch(6, 1), ctx, ATOL)
    assert ok["status"] == "pass" and ok["inputs"] == ["x_num", "x_cat"], ok
    assert (tmp_path / ok["file"]).is_file()
    json.dumps(ok, allow_nan=False)

    class NoOnnx:
        def export(self, *a, **kw):
            raise ModuleNotFoundError("No module named 'onnx'")

    failed = export_pack._export_one("onnx", (NoOnnx(), parts[1], net), _row_batch(6, 1),
                                     ctx, ATOL)
    assert failed["status"] == "failed" and failed["file"] is None
    assert any(m.startswith("WARN: export onnx failed") for m in ctx.markers), ctx.markers


# ================================================================ 5. ModelRegistry

def _svdd_spec():
    spec = _row_spec()
    spec["task"] = {"kind": "anomaly"}
    spec["loss"] = {"name": "svdd", "params": {}}
    spec["metrics"] = {"primary": "anomaly_score_mean", "secondary": []}
    spec["blocks"] = [
        {"id": "B1", "type": "embedding", "params": {"columns": ["zone"], "dims": {"zone": 3}}},
        {"id": "B2", "type": "concat", "params": {}},
        {"id": "B3", "type": "dense", "params": {"units": 8, "activation": "relu"}},
        {"id": "B4", "type": "batchnorm", "params": {}},
        {"id": "B5", "type": "head", "params": {"out": 4, "activation": "none", "bias": False}}]
    return spec


def test_the_store_is_one_class(pkgs):
    S = pkgs.row.store
    assert S.__all__ == ["ModelRegistry"]
    public = [n for n in vars(S) if not n.startswith("_") and callable(getattr(S, n))
              and getattr(getattr(S, n), "__module__", "") == S.__name__]
    assert public == ["ModelRegistry"], public


def test_save_and_load_restore_parameters_and_buffers(pkgs, tmp_path):
    ns = _package("svdd", _svdd_spec())
    spec, Network = ns.domain.NetSpec.from_dict(ns.spec), ns.network.Network
    net = Network.build(spec)
    net.svdd_center.copy_(torch.arange(4.0))       # what the svdd loss writes once
    net.svdd_center_set.fill_(True)
    net.train()
    net(_row_batch(16, 5))                          # moves the batch-norm running statistics
    net.eval()
    registry = ns.store.ModelRegistry()
    version = registry.save(net, tmp_path / "best.pt", spec=spec, epoch=3)
    assert version.epoch == 3 and version.spec_digest == spec.digest
    assert version.weights_digest.startswith("sha256:") and version.digest.startswith("sha256:")
    back = registry.load(tmp_path / "best.pt", spec=spec, build=Network.build)
    assert not back.training
    assert torch.equal(back.svdd_center, torch.arange(4.0)) and bool(back.svdd_center_set)
    for name, value in net.state_dict().items():
        assert torch.equal(back.state_dict()[name], value), name
    batch = _row_batch(7, 6)
    assert torch.equal(back.score(batch), net.score(batch))
    assert registry.load(version, spec=spec, build=Network.build) is not None
    again = registry.version(tmp_path / "best.pt")
    assert again == version


def test_the_weights_digest_covers_the_buffers(pkgs, tmp_path):
    ns = _package("svdd-digest", _svdd_spec())
    spec, Network = ns.domain.NetSpec.from_dict(ns.spec), ns.network.Network
    registry = ns.store.ModelRegistry()
    net = Network.build(spec)
    a = registry.save(net, tmp_path / "a.pt", spec=spec)
    net.svdd_center.copy_(torch.ones(4))
    b = registry.save(net, tmp_path / "b.pt", spec=spec)
    assert a.weights_digest != b.weights_digest


def test_the_store_refuses_what_it_cannot_verify(pkgs, tmp_path):
    ns = pkgs.row
    spec, Network = ns.domain.NetSpec.from_dict(ns.spec), ns.network.Network
    registry = ns.store.ModelRegistry()
    net = Network.build(spec)
    version = registry.save(net, tmp_path / "best.pt", spec=spec)
    with pytest.raises(ValueError, match="checkpoint not found"):
        registry.load(tmp_path / "none.pt", spec=spec, build=Network.build)
    # the spec and the builder are required parameters (finding 6 of the T5 round 3 audit):
    # a call without them is refused by the signature, before anything is read
    with pytest.raises(TypeError, match="spec"):
        registry.load(version, build=Network.build)
    with pytest.raises(TypeError, match="build"):
        registry.load(version, spec=spec)
    with pytest.raises(ValueError, match="digest mismatch"):
        registry.load(tmp_path / "best.pt", expected_digest="sha256:" + "0" * 64, spec=spec,
                      build=Network.build)
    other = ns.domain.NetSpec.from_dict(dict(ns.spec, seed=8))
    with pytest.raises(ValueError, match="trained under spec"):
        registry.load(version, spec=other, build=Network.build)
    blob = torch.load(tmp_path / "best.pt", weights_only=True)
    first = next(iter(blob["state_dict"]))
    blob["state_dict"][first] = blob["state_dict"][first] + 1
    torch.save(blob, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="refusing to load"):
        registry.load(tmp_path / "tampered.pt", spec=spec, build=Network.build)
    blob.pop("weights_digest")
    torch.save(blob, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="no weights digest"):
        registry.load(tmp_path / "old.pt", spec=spec, build=Network.build)


def test_a_checkpoint_of_another_network_is_refused_by_name(pkgs, tmp_path):
    ns = _package("svdd-other", _svdd_spec())
    spec, Network = ns.domain.NetSpec.from_dict(ns.spec), ns.network.Network
    registry = ns.store.ModelRegistry()
    net = Network.build(spec)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("svdd_")}
    lacking = types.SimpleNamespace(state_dict=lambda: state)
    registry.save(lacking, tmp_path / "lacking.pt", spec=spec)
    with pytest.raises(ValueError, match="missing svdd_center"):
        registry.load(tmp_path / "lacking.pt", spec=spec, build=Network.build)


def test_a_checkpoint_is_saved_and_loaded_with_its_spec_or_not_at_all(pkgs, tmp_path):
    """The T5 round 1 audit: `save(spec=None)` wrote `spec_digest: None` and `load` then
    skipped the spec check, so such a checkpoint loaded under any spec that builds the same
    shapes. A checkpoint's identity is its weights and its spec, together."""
    ns = _package("svdd-spec", _svdd_spec())
    spec, Network = ns.domain.NetSpec.from_dict(ns.spec), ns.network.Network
    registry, net = ns.store.ModelRegistry(), Network.build(spec)
    with pytest.raises(TypeError, match="NetSpec"):
        registry.save(net, tmp_path / "nospec.pt", None)
    assert not (tmp_path / "nospec.pt").exists()
    state = net.state_dict()
    torch.save({"state_dict": state, "spec_digest": None, "epoch": None,
                "weights_digest": ns.store._weights_digest(state)}, tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="records no spec digest"):
        registry.load(tmp_path / "legacy.pt", spec=spec, build=Network.build)
    with pytest.raises(ValueError, match="spec_digest"):
        registry.version(tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="spec_digest"):
        ns.domain.Checkpoint.of(tmp_path / "x.pt", "sha256:" + "0" * 64,
                                weights_digest="sha256:" + "1" * 64)


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
