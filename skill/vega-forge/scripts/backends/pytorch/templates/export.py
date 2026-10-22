"""
{{PACKAGE}}.{{MODULE_MODEL}}.export — the network exported to another runtime (support,
implements `ArtifactExporter`; used by the ModelStore's callers).

Emitted by /vega-forge from `backends/pytorch/templates/export.py`. A defect is fixed in the
template and re-emitted, never in this copy (code-contract §4).

Exporting is not storing: the ModelStore puts and gets a model by the digest of its weights,
and this file converts a model to another format and checks the conversion (EJ-21, SOLID-I) —
so a bundle carries the store without having to carry the exporter. What it writes is
`/vega-pack`'s: the bundle's `exports/`, pinned by `MANIFEST.exports` (the map's `output`).

**The positional form of a batch** (`BatchSignature.for_sample`, D1 §13). A tracer and an ONNX
graph take positional tensors, not the batch mapping the Model port reads, so the order is fixed
per sample shape (backend-contract §1), the optional inputs kept when the example batch carries
them:

    row          (x_num, x_cat?)
    window       (x_num[B, L, F'], x_cat[B, L, C]?)
    graph node   (x_num, x_cat?, edge_index, edge_weight?, target_index)
    graph level  (x_num, x_cat?, edge_index, edge_weight?, graph_index, num_graphs)

`num_graphs` travels as a 0-d int64 tensor (a traced graph reads it as a value, so a batch of
another size runs through the same export), and the axes that vary between batches — the rows,
the nodes, the edges, the targets — are the dynamic axes of the ONNX graph. `_Wrapped` rebuilds
the batch mapping from the positional tensors, so the exported graph is the network's own
`forward` (outputs aligned with `batch["y"]`: at node level the targets' rows).

**Formats.** `torchscript` by `torch.jit.trace` — the signature is saved inside the file
(`_extra_files`), so the parity check and any caller read the order from the artifact, not from
a guess; `torch.jit.trace` is deprecated in torch ≥ 2.9 and says so with a `FutureWarning`,
which the export records (a `torch.export` path is left for the next release). `onnx` at opset
18 (graph convolutions need `ScatterElements` with a reduction); its input names are the
signature's keys. The graph reductions — `index_add` (sums, degrees) and `scatter_reduce`
(maxima, the attention softmax) — are lowered to `ScatterElements` for the export call only
(`_scatter_lowerings`): the TorchScript-based exporter's own `index_add` keeps one message of
the several a destination receives, and it refuses `scatter_reduce` without the initial
values. An export that fails raises — the caller records the failure as `WARN:` and ships
without it (bundle-contract §7) — and a parity check reports the worst absolute difference
against the network on the same batch.

    TorchExporter().export(model, spec, fmt, path, batch) -> {format, file, inputs, opset, warnings}
    TorchExporter().parity(model, exported_path, batch, atol=1e-5, fmt="torchscript")
        -> {max_abs_diff, atol, pass[, why]}   (max_abs_diff None when not comparable)
"""
from __future__ import annotations

import inspect
import json
import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from ..domain import BatchForm, BatchKey, ExportFormat, NetSpec, SampleSpec
# >>> shape: graph
from ..domain import SampleShape
# <<< shape

__all__ = ["BatchSignature", "TorchExporter", "FORMATS", "ONNX_OPSET"]

FORMATS = tuple(f.value for f in ExportFormat)
ONNX_OPSET = 18

# the signature carried inside a TorchScript export
_EXTRA = "vega-signature.json"

_X_NUM, _X_CAT = BatchKey.X_NUM.value, BatchKey.X_CAT.value
# >>> shape: graph
_EDGE_INDEX, _EDGE_WEIGHT = BatchKey.EDGE_INDEX.value, BatchKey.EDGE_WEIGHT.value
_TARGET_INDEX, _GRAPH_INDEX = BatchKey.TARGET_INDEX.value, BatchKey.GRAPH_INDEX.value
_NUM_GRAPHS = BatchKey.NUM_GRAPHS.value
# <<< shape

# the positional order of each form this package's sample has (backend-contract §1; D1 §13)
_ORDER = MappingProxyType({
    # >>> shape: row
    BatchForm.ROW: (_X_NUM, _X_CAT),
    # <<< shape
    # >>> shape: window
    BatchForm.WINDOW: (_X_NUM, _X_CAT),
    # <<< shape
    # >>> shape: graph
    BatchForm.NODE: (_X_NUM, _X_CAT, _EDGE_INDEX, _EDGE_WEIGHT, _TARGET_INDEX),
    BatchForm.GRAPH: (_X_NUM, _X_CAT, _EDGE_INDEX, _EDGE_WEIGHT, _GRAPH_INDEX, _NUM_GRAPHS),
    # <<< shape
})
_OPTIONAL = frozenset({_X_CAT,
                       # >>> shape: graph
                       _EDGE_WEIGHT,
                       # <<< shape
                       })

# the axis of each input that varies between batches, per form; and the output's
_AXES = MappingProxyType({
    # >>> shape: row
    BatchForm.ROW: ((_X_NUM, 0, "rows"), (_X_CAT, 0, "rows")),
    # <<< shape
    # >>> shape: window
    BatchForm.WINDOW: ((_X_NUM, 0, "windows"), (_X_CAT, 0, "windows")),
    # <<< shape
    # >>> shape: graph
    BatchForm.NODE: ((_X_NUM, 0, "nodes"), (_X_CAT, 0, "nodes"), (_EDGE_INDEX, 1, "edges"),
                     (_EDGE_WEIGHT, 0, "edges"), (_TARGET_INDEX, 0, "targets")),
    BatchForm.GRAPH: ((_X_NUM, 0, "nodes"), (_X_CAT, 0, "nodes"), (_EDGE_INDEX, 1, "edges"),
                      (_EDGE_WEIGHT, 0, "edges"), (_GRAPH_INDEX, 0, "nodes")),
    # <<< shape
})
_OUT_AXIS = MappingProxyType({
    # >>> shape: row
    BatchForm.ROW: "rows",
    # <<< shape
    # >>> shape: window
    BatchForm.WINDOW: "windows",
    # <<< shape
    # >>> shape: graph
    BatchForm.NODE: "targets", BatchForm.GRAPH: "graphs",
    # <<< shape
})


def _form(sample: SampleSpec) -> BatchForm:
    """row | window | node | graph — the shape, with a graph's level."""
    # >>> shape: graph
    if sample.shape is SampleShape.GRAPH:
        return BatchForm.NODE if sample.is_node_level else BatchForm.GRAPH
    # <<< shape
    return BatchForm(sample.shape.value)


@dataclass(frozen=True)
class BatchSignature:
    """The positional inputs of an exported network: which batch keys, in which order, and
    which of their axes vary between batches."""

    form: BatchForm
    keys: tuple

    def __post_init__(self):
        if self.form not in tuple(f.value for f in _ORDER):
            raise ValueError(f"unknown batch form {self.form!r}; one of {', '.join(_ORDER)}")
        object.__setattr__(self, "form", BatchForm(self.form))
        allowed = _ORDER[self.form]
        required = tuple(k for k in allowed if k not in _OPTIONAL)
        keys = tuple(self.keys)
        if [k for k in allowed if k in keys] != list(keys) or not set(required) <= set(keys):
            raise ValueError(f"a {self.form} batch is exported as {allowed} (optional: "
                             f"{sorted(_OPTIONAL & set(allowed))}), not {keys}")
        object.__setattr__(self, "keys", keys)

    @staticmethod
    def for_sample(sample, batch=None) -> "BatchSignature":
        """The signature of `sample` (a SampleSpec, or the spec it is read from): every key the
        form requires, and each optional one (`x_cat`, `edge_weight`) that `batch` carries."""
        sample = sample if isinstance(sample, SampleSpec) else SampleSpec.from_spec(sample)
        form = _form(sample)
        present = {k for k, v in (batch or {}).items() if v is not None}
        missing = [k for k in _ORDER[form] if k not in _OPTIONAL and batch is not None
                   and k not in present]
        if missing:
            raise ValueError(f"a {form} batch needs {', '.join(missing)} to be exported; the "
                             "example batch has none")
        keys = tuple(k for k in _ORDER[form] if k not in _OPTIONAL or k in present)
        return BatchSignature(form, keys)

    @staticmethod
    def from_dict(d) -> "BatchSignature":
        return BatchSignature(str(d["form"]), tuple(d["inputs"]))

    def as_dict(self) -> dict:
        return {"form": self.form.value, "inputs": list(self.keys),
                "output": _OUT_AXIS[self.form]}

    def inputs(self, batch) -> tuple:
        """The batch as positional tensors, in the signature's order (`num_graphs` a 0-d
        int64 tensor); an optional input the signature does not take must be absent."""
        stray = sorted(k for k in _OPTIONAL & set(_ORDER[self.form])
                       if k not in self.keys and batch.get(k) is not None)
        if stray:
            raise ValueError(f"the batch carries {', '.join(stray)}, which this export was not "
                             "traced with; export it again from a batch that has it")
        missing = [k for k in self.keys if batch.get(k) is None]
        if missing:
            raise ValueError(f"the batch lacks {', '.join(missing)}, which this export takes")
        return tuple(_tensor(k, batch[k]) for k in self.keys)

    def batch_of(self, args) -> dict:
        """The batch mapping a network reads, rebuilt from positional tensors."""
        return dict(zip(self.keys, args))

    def dynamic_axes(self) -> dict:
        """{input: {axis: name}} for the ONNX graph, the output `out` included."""
        axes = {}
        for key, axis, name in _AXES[self.form]:
            if key in self.keys:
                axes.setdefault(key, {})[axis] = name
        axes["out"] = {0: _OUT_AXIS[self.form]}
        return axes


def _tensor(key, value):
    # >>> shape: graph
    if key == _NUM_GRAPHS:
        return torch.as_tensor(int(value), dtype=torch.long)
    # <<< shape
    return value if torch.is_tensor(value) else torch.as_tensor(value)


class _Wrapped(torch.nn.Module):
    """The network over positional inputs: a tracer or an ONNX exporter does not take the
    batch mapping the Model port reads."""

    def __init__(self, model, signature: BatchSignature):
        super().__init__()
        self.model = model
        self.signature = signature

    def forward(self, *args):
        return self.model(self.signature.batch_of(args))


def _sample_of(model, spec):
    """The sample the network was built for: its own (`Network.sample`), else the spec's."""
    sample = getattr(model, "sample", None)
    if isinstance(sample, SampleSpec):
        return sample
    if spec is None:
        raise ValueError("the export needs the NetSpec (or a Network that carries its sample) "
                         "to fix the positional input order")
    return SampleSpec.from_spec(spec)


def _recorded(fn):
    """fn()'s value and the warnings it raised, as text (the jit deprecation included)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    notes = []
    for w in caught:
        text = f"{w.category.__name__}: {str(w.message).splitlines()[0][:200]}"
        if text not in notes:
            notes.append(text)
    return value, notes


def _torchscript(wrapped, args, path, signature):
    traced = torch.jit.trace(wrapped, args, strict=False, check_trace=False)
    traced.save(str(path), _extra_files={_EXTRA: json.dumps(signature.as_dict())})


# >>> shape: graph
# The scatters of a graph network, lowered to ScatterElements with the reduction opset 18 defines.
# The TorchScript-based exporter's own `index_add` keeps one of the duplicate indices every
# graph's messages carry (its file answers wrong), and it refuses `scatter_reduce` with
# include_self=False (max aggregation, the attention softmax); both are replaced for the export
# call only.
_REDUCTION = MappingProxyType({"sum": "add", "prod": "mul", "amax": "max", "amin": "min"})
_IDENTITY = MappingProxyType({"add": 0.0, "mul": 1.0, "max": -math.inf, "min": math.inf})


def _filled(g, value, like, shape=None):
    """A tensor of `like`'s dtype filled with `value`, of `shape`'s shape (default `like`'s)."""
    scalar = g.op("CastLike", g.op("Constant", value_t=torch.tensor(value)), like)
    return g.op("Expand", scalar, g.op("Shape", like if shape is None else shape))


def _index_add(g, *args):
    """aten::index_add(self, dim, index, source, alpha) -> ScatterElements(reduction=add)."""
    sh = torch.onnx.symbolic_helper
    target, dim, index, source = args[:4]
    alpha = args[4] if len(args) > 4 else None
    if alpha and sh._scalar(sh._maybe_get_scalar(alpha)) != 1:
        return sh._unimplemented("index_add", "alpha != 1", target)
    dim, rank = sh._get_const(dim, "i", "dim"), sh._get_tensor_rank(source)
    if rank is None:
        return sh._unimplemented("index_add", "a source of unknown rank", source)
    dim = dim % rank
    axes = [a for a in range(rank) if a != dim]          # index (k,) broadcast to the source
    wide = sh._unsqueeze_helper(g, index, axes) if axes else index
    wide = g.op("Expand", wide, g.op("Shape", source))
    return g.op("ScatterElements", target, wide, source, axis_i=dim, reduction_s="add")


def _scatter_reduce(g, *args):
    """aten::scatter_reduce(self, dim, index, src, reduce, include_self) -> ScatterElements with
    the reduction; without include_self a position no element reaches keeps `self`, and the
    others start from the reduction's identity."""
    sh = torch.onnx.symbolic_helper
    target, dim, index, src, reduce, include_self = args
    dim = sh._get_const(dim, "i", "dim")
    mode = _REDUCTION.get(sh._get_const(reduce, "s", "reduce"))
    if mode is None:
        return sh._unimplemented("scatter_reduce", "reduce='mean'", target)
    if not sh._get_const(include_self, "b", "include_self"):
        zero = _filled(g, 0.0, target)
        reached = g.op("ScatterElements", zero, index, _filled(g, 1.0, src, shape=index),
                       axis_i=dim, reduction_s="add")
        target = g.op("Where", g.op("Greater", reached, zero),
                      _filled(g, _IDENTITY[mode], target), target)
    return g.op("ScatterElements", target, index, src, axis_i=dim, reduction_s=mode)


_LOWERINGS = MappingProxyType({"aten::index_add": _index_add,
                               "aten::scatter_reduce": _scatter_reduce})


@contextmanager
def _scatter_lowerings():
    """The lowerings registered at ONNX_OPSET for the `with` body only (the ones registered are
    unregistered however the body or a later registration ends)."""
    registered = []
    try:
        for name, fn in _LOWERINGS.items():
            torch.onnx.register_custom_op_symbolic(name, fn, ONNX_OPSET)
            registered.append(name)
        yield
    finally:
        for name in registered:
            torch.onnx.unregister_custom_op_symbolic(name, ONNX_OPSET)
# <<< shape


def _onnx(wrapped, args, path, signature):
    options = {"input_names": list(signature.keys), "output_names": ["out"],
               "dynamic_axes": signature.dynamic_axes(), "opset_version": ONNX_OPSET}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False          # the TorchScript-based exporter: dynamic_axes
    # >>> shape: graph
    with _scatter_lowerings():
        torch.onnx.export(wrapped, args, str(path), **options)
    # <<< shape
    # >>> shape: row window
    torch.onnx.export(wrapped, args, str(path), **options)
    # <<< shape


_WRITERS = MappingProxyType({ExportFormat.TORCHSCRIPT: _torchscript, ExportFormat.ONNX: _onnx})


def _loaded_torchscript(path):
    """(the loaded module, the signature saved inside it)."""
    extra = {_EXTRA: ""}
    module = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
    if not extra[_EXTRA]:
        raise ValueError(f"{path} carries no {_EXTRA}; it was not exported by this backend's "
                         "TorchExporter, so its input order is unknown")
    return module, BatchSignature.from_dict(json.loads(extra[_EXTRA]))


def _run_torchscript(path, batch):
    module, signature = _loaded_torchscript(path)
    with torch.no_grad():
        return module(*signature.inputs(_on_cpu(batch)))


def _run_onnx(path, batch):
    import onnxruntime  # noqa: PLC0415  (only needed when an onnx parity is asked for)
    session = onnxruntime.InferenceSession(str(path))
    names = [i.name for i in session.get_inputs()]
    feeds = {k: _tensor(k, v).cpu().numpy() for k, v in batch.items()
             if k in names and v is not None}
    return torch.as_tensor(session.run(None, feeds)[0])


_RUNNERS = MappingProxyType({ExportFormat.TORCHSCRIPT: _run_torchscript,
                             ExportFormat.ONNX: _run_onnx})


def _on_cpu(batch):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}


@contextmanager
def _cpu_eval(model):
    """The model in eval mode on the CPU for the call — an export is traced where it will be
    loaded, and a traced graph keeps the device its tensors were made on — then back on its
    device and in its mode: exporting never changes the caller's network."""
    first = next(iter(model.parameters()), None)
    device, training = (first.device if first is not None else None), model.training
    try:
        yield model.eval().cpu()
    finally:
        if device is not None:
            model.to(device)
        model.train(training)


def _check_format(fmt) -> ExportFormat:
    if fmt not in FORMATS:
        raise ValueError(f"unsupported export format {fmt!r}; this backend exports "
                         f"{' and '.join(FORMATS)}")
    return ExportFormat(fmt)


class TorchExporter:
    """The ArtifactExporter of the PyTorch backend: TorchScript and ONNX, with a parity check
    against the network on a real batch."""

    def export(self, model, spec: NetSpec | None, fmt: str, path, batch=None) -> dict:
        """Write `model` (in eval mode, on the CPU) in `fmt` to `path`, traced on `batch` — one
        real batch of the sample's shape. Returns what was written and the warnings the
        exporter raised."""
        fmt = _check_format(fmt)
        if batch is None:
            raise ValueError("export needs one real batch to trace with")
        signature = BatchSignature.for_sample(_sample_of(model, spec), batch)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        args = signature.inputs(_on_cpu(batch))
        try:
            with _cpu_eval(model) as net, torch.no_grad():
                _, notes = _recorded(lambda: _WRITERS[fmt](_Wrapped(net, signature), args, p,
                                                            signature))
        except BaseException:
            p.unlink(missing_ok=True)         # no half-written artefact is left to ship
            raise
        return {"format": fmt.value, "file": str(p), "inputs": list(signature.keys),
                "opset": ONNX_OPSET if fmt is ExportFormat.ONNX else None, "warnings": notes}

    def parity(self, model, exported_path, batch, atol: float = 1e-5,
               fmt: str = "torchscript") -> dict:
        """The exported artefact's outputs against the network's, on `batch` (which may differ
        in size from the batch the export was traced on)."""
        fmt = _check_format(fmt)
        batch = _on_cpu(batch)
        with _cpu_eval(model) as net, torch.no_grad():
            want = net(batch)
        got = torch.as_tensor(_RUNNERS[fmt](exported_path, batch))
        if tuple(got.shape) != tuple(want.shape):
            return {"max_abs_diff": None, "atol": float(atol), "pass": False,
                    "why": f"the export answers shape {tuple(got.shape)}, the network "
                           f"{tuple(want.shape)}"}
        diff = float((want - got.to(want.dtype)).abs().max()) if want.numel() else 0.0
        if not math.isfinite(diff):
            return {"max_abs_diff": None, "atol": float(atol), "pass": False,
                    "why": "a non-finite difference: one side answers NaN or inf"}
        return {"max_abs_diff": diff, "atol": float(atol), "pass": diff <= float(atol)}
