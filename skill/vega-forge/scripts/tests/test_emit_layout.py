#!/usr/bin/env python3
"""
test_emit_layout.py — the emit table of v2.3, the one generation since gate A (WPs B04, B14).

    python3 test_emit_layout.py [-v] [-k <substring>] [--work <dir>]
    python3 -m pytest test_emit_layout.py            (the same tests, pytest's tmp_path)

What it proves (BUILD-PLAN §2.4, B04's and B14's closing tests; D1 §4.1; R-B9):

1. **One generation, and the cutover kept the network.** `v2.3` is the default and the only
   generation; `v2.2` is refused by name, and so is a template that still carries a v2.2 splice
   placeholder. The forge fixture's orders package, emitted from the live sources, builds the
   network c849f8b's emitter built from the same spec — the same parameters, in the same order,
   with the same seeded values, and the same forward pass (the reference is c849f8b's own
   `backends/pytorch/`, read out of git) — and B01's smoke golden (parameter count, blocks).
2. **The v2.3 file list.** Generation `v2.3` over stub templates and stub splice sources emits
   exactly the files of D1 §4.1 for a row, a window and a graph spec (node and graph level), a
   generative row spec with a `custom` block, and a pipeline without an Evaluator: each file
   under its `when` predicate, nothing more.
3. **Every file is mapped.** On every emitted package the mechanical gate's C0–C10 hold (C3:
   every file has an entry and `unmapped` is empty; C4: the imports the emitter generates run
   along the blueprint's arrows), and the package imports and builds.
4. **The unmapped guard still refuses a stray file.**
5. **Custom blocks ship** (R-B9): the `code_ref` module is copied to `<model>/custom/<stem>.py`
   with a `kind: support` entry and its digest, and the builder finds it relative to the package.
   A relative `code_ref` resolves against the campaign root only, never the run directory.
6. **The splice contract**: block segments routed by `SEGMENT_FILE`; the prelude spliced whole
   into `blocks_core` only, the other block files opening with its imports, so a class it
   defines is one class in the package; package-relative imports resolved (`from .blocks_core
   import Dense`, `from ..domain import NetSpec`); the tables read whether assigned plainly
   or with an annotation; and each way of getting it wrong refused with a message that names
   the fix. A refused re-emit leaves the previous package as it was.
7. **render_templates.py** renders a template set partially (layered, `--into`, `--splice`,
   `--define`) and fully (exactly `emit_with(..., EmitOptions(generation="v2.3", ...))`,
   refusing partial options).
8. **The live sources** emit a mapped package that passes the whole mechanical gate (C0–C14,
   C11–C14 at `warn`) for every reference shape and the orders spec — the baselines of the
   Model port importing each other (C5, PLAN-AMENDMENTS A14), a baseline importing a primary
   adapter still refused; `ports.py` carries the v2.3 fallback signatures (at most five
   parameters, no `Inspectable`) when the blueprint gives none.
9. **One sample shape per package.** `select_shapes` keeps the `# >>> shape:` sections of the
   shapes carried (nested ones too), drops every marker line and the blank lines a section left
   out would double, and refuses an unknown shape, a stray close and an unclosed section;
   every template of the backend parses for every shape and for each one alone. An emitted
   package carries its spec's shape and no other: `domain.CARRIED_SHAPES` is that shape, no
   file names a shape module the package does not hold (the old row package's lazy imports of
   `windows.py`, `graphs.py` and `graph_engines.py` are what C4 now refuses, seeded), and a
   spec of any other shape is refused by name — `UnsupportedShapeError` from `NetSpec.from_dict`
   and `SampleSpec.from_spec`, before the DataSource, the epoch sources or the Predictor's
   assemblers see it — while the package's own shape still loads.

The stub backend (`write_stub_backend`) needs no torch: it is the *layout* of the real sources
(segment markers, `SEGMENTS_FOR`, `SEGMENT_FILE`, the templates' names and placeholders), which
is what the emit table reads.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))

import render_templates as rt  # noqa: E402
import vega_regularization  # noqa: E402

emit = rt.load_emit()

BASELINE_COMMIT = "c849f8b"
PACKAGE = "orders_pipeline"


# ================================================================ the stub backend

_GENERIC = '''"""
{{{{PACKAGE}}}} — stub of the template `{name}` (spec {{{{SPEC_DIGEST}}}}).
"""
from __future__ import annotations

TEMPLATE = "{name}"
'''

_STUB_TEMPLATES = {
    "domain.py": '''"""
{{PACKAGE}}.domain — stub value objects for test_emit_layout.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskKind(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    FORECAST = "forecast"


@dataclass(frozen=True)
class NetSpec:
    blocks: tuple
    shape: str
    loss: str

    @staticmethod
    def require(spec):
        if isinstance(spec, NetSpec):
            return spec
        sample = (spec.get("data_contract") or {}).get("sample") or {}
        return NetSpec(tuple(spec.get("blocks") or ()), sample.get("shape", "row"),
                       (spec.get("loss") or {}).get("name", "mse"))


@dataclass(frozen=True)
class FeatureContract:
    manifest_digest: str


@dataclass(frozen=True)
class Checkpoint:
    digest: str


ModelVersion = Checkpoint


@dataclass(frozen=True)
class Metrics:
    values: tuple


@dataclass(frozen=True)
class Prediction:
    score: float


class SealedPartError(RuntimeError):
    """The test part is sealed."""


class Part(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DesignView(str, Enum):
    FLAT = "flat"


@dataclass(frozen=True)
class _Value:
    value: object = None


BatchPlan = DesignMatrix = DriftReading = FitResult = GraphExposure = _Value
RunContext = SamplePart = ScoredPart = MetricOptions = NetworkAnatomy = _Value


def _private_helper(value):
    return value
''',
    "network.py": '''"""
{{PACKAGE}}.{{MODULE_MODEL}}.network — stub {{PORT_MODEL}} adapter (role: Model).

{{RESPONSIBILITY_MODEL}} ({{SOURCE_MODEL}}); spec {{SPEC_DIGEST}}; blocks: {{BLOCK_TYPES}}.
"""
from __future__ import annotations

from ..domain import NetSpec
from .builder import NetworkBuilder


class _Net:
    def __init__(self, stages):
        self.stages = tuple(stages)


def build_model(spec, run_dir=None):
    return _Net(NetworkBuilder(NetSpec.require(spec), run_dir).stages())
''',
    "data.py": '''"""
{{PACKAGE}}.{{MODULE_INGESTION}}.data — stub {{PORT_DATASOURCE}} adapter.
"""
from __future__ import annotations

from ..domain import SealedPartError


def load_part(spec, part):
    if str(part) == "test":
        raise SealedPartError("the test part is sealed")
    return part


def load_sealed_part(spec, token):
    return token
''',
    # the stubs that take the arrows the real templates take (C4 holds an arrow no import
    # takes): the Trainer and the Evaluator read the Model module, a graph Trainer the
    # DataSource module's graph views
    "loop.py": '''"""
{{PACKAGE}}.{{MODULE_TRAINING}}.loop — stub {{PORT_TRAINER}} adapter.
"""
from __future__ import annotations

from ..{{MODULE_MODEL}} import readout as _readout

TEMPLATE = "loop.py"
READS = (_readout,)
''',
    "evaluate.py": '''"""
{{PACKAGE}}.{{MODULE_EVALUATION}}.evaluate — stub {{PORT_EVALUATOR}} adapter.
"""
from __future__ import annotations

from ..{{MODULE_MODEL}} import readout as _readout

TEMPLATE = "evaluate.py"
READS = (_readout,)
''',
    "graph_engines.py": '''"""
{{PACKAGE}}.{{MODULE_TRAINING}}.graph_engines — stub graph engines.
"""
from __future__ import annotations

from ..{{MODULE_INGESTION}} import graphs as _graphs

TEMPLATE = "graph_engines.py"
READS = (_graphs,)
''',
    "preprocess.py": '''"""
{{PACKAGE}}.{{MODULE_FEATURES}}.preprocess — stub {{PORT_FEATURETRANSFORMER}} adapter.
"""
from __future__ import annotations


def transform(df_raw, history=None):
    return df_raw
''',
}

STUB_TEMPLATE_NAMES = (
    "domain.py", "network.py", "readout.py", "assemble.py", "metrics.py", "baselines.py",
    "baselines_extra.py", "store.py", "export.py", "data.py", "frames.py", "windows.py",
    "graphs.py", "loop.py", "optim.py", "record.py", "engines.py", "graph_engines.py",
    "evaluate.py", "preprocess.py", "predict.py", "monitor.py")

STUB_BLOCKS = '''"""
Stub block library for test_emit_layout.py: the segment layout of blocks.py, without torch.
"""
from __future__ import annotations

# --8<-- prelude
import math

# a table the segments write to, through the base class: one table in the package
REGISTRY = {}


class Block:
    """The base class every block shares: one class in the whole package."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        REGISTRY[cls.__name__] = cls


def _act(name):
    return name or "none"


def _prod(xs):
    return math.prod(int(x) for x in xs)
# -->8-- prelude


# --8<-- dense
class Dense(Block):
    def __init__(self, units, activation="none"):
        self.units, self.act = int(units), _act(activation)
# -->8-- dense


# --8<-- embedding
class Embedding:
    def __init__(self, columns):
        self.columns = tuple(columns)
# -->8-- embedding


# --8<-- concat
class Concat:
    width = 0
# -->8-- concat


# --8<-- simple
class Dropout:
    def __init__(self, p=0.1):
        self.p = float(p)


class Activation:
    def __init__(self, name="relu"):
        self.name = _act(name)
# -->8-- simple


# --8<-- head
class Head:
    def __init__(self, out):
        self.out = int(out)
# -->8-- head


# --8<-- recurrent
class Recurrent(Block):
    def __init__(self, kind, hidden):
        self.kind, self.hidden = kind, int(hidden)
        self.act = _act("tanh")
# -->8-- recurrent


# --8<-- attention_pool
class AttentionPool:
    def __init__(self, dim):
        self.dim = int(dim)
# -->8-- attention_pool


# --8<-- graph_ops
def _scatter_mean(values, index):
    return values
# -->8-- graph_ops


# --8<-- graph_conv
class GCNConv(Block):
    def __init__(self, hidden):
        self.hidden = int(hidden)

    def forward(self, x, edge_index, edge_weight=None):
        return _scatter_mean(x, edge_index)


class GINConv:
    def __init__(self, hidden):
        self.mlp = Dense(hidden, "relu")
# -->8-- graph_conv


# --8<-- graph_pool
class GraphPool:
    def __init__(self, readout="mean"):
        self.readout = readout
# -->8-- graph_pool


# --8<-- generative
class Decoder(Block):
    def __init__(self, out):
        self.head = Head(out)


class Reparameterize:
    def __init__(self, z):
        self.z = int(z)
# -->8-- generative


SEGMENTS_FOR = {
    "dense": ["dense"], "embedding": ["embedding"], "concat": ["concat"],
    "dropout": ["simple"], "activation": ["simple"], "head": ["head"],
    "rnn": ["recurrent"], "gru": ["recurrent"], "lstm": ["recurrent"],
    "attention_pool": ["attention_pool"],
    "gcn_conv": ["graph_ops", "graph_conv"], "gin_conv": ["graph_ops", "graph_conv", "dense"],
    "graph_pool": ["graph_pool"],
    "decoder": ["generative", "head"], "reparameterize": ["generative"],
    "custom": [],
}

SEGMENT_FILE = {
    "prelude": "blocks_core", "dense": "blocks_core", "embedding": "blocks_core",
    "concat": "blocks_core", "simple": "blocks_core", "head": "blocks_core",
    "recurrent": "blocks_seq", "attention_pool": "blocks_seq",
    "graph_ops": "blocks_graph", "graph_conv": "blocks_graph", "graph_pool": "blocks_graph",
    "generative": "blocks_gen",
}
'''

STUB_BUILDER = '''"""
Stub builder for test_emit_layout.py.
"""
from __future__ import annotations

# --8<-- prelude
import importlib.util
from pathlib import Path
# -->8-- prelude


# --8<-- builder
class NetworkBuilder:
    _DISPATCH = {"dense": "_dense", "embedding": "_embedding", "concat": "_concat",
                 "dropout": "_dropout", "activation": "_activation", "head": "_head",
                 "gru": "_recurrent", "gcn_conv": "_gcn", "gin_conv": "_gin",
                 "graph_pool": "_graph_pool", "decoder": "_decoder",
                 "reparameterize": "_reparameterize", "custom": "_custom"}

    def __init__(self, spec, run_dir=None):
        self._spec = NetSpec.require(spec)
        self._run_dir = run_dir

    def stages(self):
        return tuple(getattr(self, self._DISPATCH[b["type"]])(b.get("params") or {})
                     for b in self._spec.blocks)

    def _dense(self, p):
        return Dense(p["units"], p.get("activation", "none"))

    def _embedding(self, p):
        return Embedding(p["columns"])

    def _concat(self, p):
        return Concat()

    def _dropout(self, p):
        return Dropout(p.get("p", 0.1))

    def _activation(self, p):
        return Activation(p.get("name"))

    def _head(self, p):
        return Head(p["out"])

    def _recurrent(self, p):
        return Recurrent("gru", p["hidden"])

    def _gcn(self, p):
        return GCNConv(p["out"])

    def _gin(self, p):
        return GINConv(p["out"])

    def _graph_pool(self, p):
        return GraphPool(p.get("readout", "mean"))

    def _decoder(self, p):
        return Decoder(p["out"])

    def _reparameterize(self, p):
        return Reparameterize(p["z"])

    def _custom(self, p):
        # R-B9: the code_ref module ships inside the package, beside this file
        path = Path(__file__).resolve().parent / "custom" / Path(p["code_ref"]).name
        spec = importlib.util.spec_from_file_location(f"custom_{path.stem}", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build((_prod([2, 3]),), p)
# -->8-- builder
'''

STUB_LOSSES = '''"""
Stub losses for test_emit_layout.py.
"""
from __future__ import annotations


# --8<-- losses
import math


class MSE:
    name = "mse"

    def formula(self):
        return "L = mean[(y - f(x))^2]"


class BCEWithLogits:
    name = "bce_with_logits"

    def formula(self):
        return f"L = -[y log s(z) + (1-y) log(1-s(z))], e = {math.e:.3f}"


LOSSES = {"mse": MSE, "bce_with_logits": BCEWithLogits, "cross_entropy": MSE}


def build_loss(spec):
    spec = NetSpec.require(spec)
    return LOSSES.get(spec.loss, MSE)()
# -->8-- losses
'''

CUSTOM_GAIN = '''"""A custom block: y = x * g (the lane asset custom_gain.py, stubbed)."""


def build(in_shape, params):
    return ("gain", tuple(in_shape), params.get("out_shape"))
'''


def write_stub_backend(root):
    """A backend directory (templates/, blocks.py, builder.py, losses.py) with the layout of
    the real v2.3 sources and no torch. Returns its emit.Sources."""
    root = Path(root)
    (root / "templates").mkdir(parents=True, exist_ok=True)
    for name in STUB_TEMPLATE_NAMES:
        text = _STUB_TEMPLATES.get(name) or _GENERIC.format(name=name)
        (root / "templates" / name).write_text(text, encoding="utf-8")
    (root / "blocks.py").write_text(STUB_BLOCKS, encoding="utf-8")
    (root / "builder.py").write_text(STUB_BUILDER, encoding="utf-8")
    (root / "losses.py").write_text(STUB_LOSSES, encoding="utf-8")
    return emit.Sources.at(root)


# ================================================================ helpers

def files_of(root):
    root = Path(root)
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def _drawn_for(package_root, pipeline):
    """The reference world model as this package's blueprint draws it (`rt.scoped_pipeline`,
    against the spec the package was emitted from); any other pipeline as given."""
    snapshot = Path(package_root).parent / "spec_snapshot.json"
    if pipeline != rt.reference_pipeline() or not snapshot.is_file():
        return pipeline
    return rt.scoped_pipeline(pipeline, json.loads(snapshot.read_text(encoding="utf-8")))


def gate_errors(package_root, pipeline):
    """The mechanical gate's C0–C10 errors on a package (conformance.py, the gate's own code)."""
    import conformance
    pipeline = _drawn_for(package_root, pipeline)
    bmap = json.loads((Path(package_root) / "blueprint-map.json").read_text(encoding="utf-8"))
    markers = conformance.check(package_root, bmap, pipeline)
    rules = {f"C{i}" for i in range(0, 11)}
    return [f"{m.rule}: {m.message}" for m in markers.errors if (m.rule or "C0") in rules]


def c3_errors(package_root, pipeline):
    import conformance
    pipeline = _drawn_for(package_root, pipeline)
    bmap = json.loads((Path(package_root) / "blueprint-map.json").read_text(encoding="utf-8"))
    pkg, err = conformance.Package.read(package_root, bmap, pipeline)
    assert err is None, err
    return [m.message for m in conformance.c3_every_file_is_mapped(pkg)
            if str(m.level).lower().endswith("error")]


@contextlib.contextmanager
def importable(out, package=PACKAGE):
    """Import the emitted package from `out`, and forget it afterwards."""
    def purge():
        for name in list(sys.modules):
            if name == package or name.startswith(package + "."):
                del sys.modules[name]
    purge()
    sys.path.insert(0, str(out))
    try:
        yield
    finally:
        sys.path.remove(str(out))
        purge()


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def raises(fn, *needles):
    """Call fn; it must raise EmitError (or RenderError) whose message contains every needle."""
    try:
        fn()
    except (emit.EmitError, rt.RenderError) as exc:
        msg = str(exc)
        missing = [n for n in needles if n not in msg]
        assert not missing, f"the refusal does not name {missing}: {msg}"
        return msg
    raise AssertionError(f"no refusal (expected one naming {needles})")


def generative_custom_spec():
    spec = rt.reference_spec("row")
    spec["blocks"] = [
        {"id": "B1", "type": "dense", "params": {"units": 8, "activation": "relu"}},
        {"id": "B2", "type": "reparameterize", "params": {"z": 4}},
        {"id": "B3", "type": "decoder", "params": {"out": 5}},
        {"id": "B4", "type": "custom", "source": "user:stance/custom-gain",
         "params": {"code_ref": "custom/custom_gain.py", "out_shape": [5]}},
        {"id": "B5", "type": "head", "params": {"out": 1}}]
    return spec


# the D1 §4.1 file list, for the reference world model (every port drawn)
COMMON = {
    "__init__.py", "domain.py", "ports.py", "blueprint-map.json",
    "ingestion/__init__.py", "ingestion/data.py", "ingestion/frames.py",
    "features/__init__.py", "features/preprocess.py",
    "model/__init__.py", "model/network.py", "model/builder.py", "model/blocks_core.py",
    "model/readout.py", "model/assemble.py", "model/metrics.py", "model/baselines.py",
    "model/baselines_extra.py", "model/store.py", "model/export.py",
    "training/__init__.py", "training/loop.py", "training/losses.py", "training/optim.py",
    "training/record.py", "training/engines.py",
    "evaluation/__init__.py", "evaluation/evaluate.py",
    "serving/__init__.py", "serving/predict.py", "serving/monitor.py",
}
GRAPH = {"ingestion/graphs.py", "training/graph_engines.py", "model/blocks_graph.py"}
CASES = {
    "row": (lambda: rt.reference_spec("row"), COMMON),
    "window": (lambda: rt.reference_spec("window"),
               COMMON | {"ingestion/windows.py", "model/blocks_seq.py"}),
    "graph": (lambda: rt.reference_spec("graph"), COMMON | GRAPH),
    "graph-level": (lambda: rt.reference_spec("graph-level"), COMMON | GRAPH),
    "generative-custom": (generative_custom_spec,
                          COMMON | {"model/blocks_gen.py", "model/custom/custom_gain.py"}),
}


def emit_stub(tmp, spec, pipeline=None, sources=None, name="run"):
    src = sources or write_stub_backend(Path(tmp) / "stub-backend")
    out = Path(tmp) / name
    (out / "custom").mkdir(parents=True, exist_ok=True)
    (out / "custom" / "custom_gain.py").write_text(CUSTOM_GAIN, encoding="utf-8")
    # the reference world model drawn for this spec's package (its shape's and kind's ports,
    # the arrows its code takes), as a blueprint of that package would draw it
    rec = emit.emit_with(spec, pipeline or rt.scoped_pipeline(rt.reference_pipeline(), spec), out,
                         emit.EmitOptions(generation="v2.3", sources=src, campaign_root=out))
    return out, rec


_FIXTURE = {}
_SESSION = {"root": None}      # the runner's session directory (pytest: the first tmp_path)


def orders_fixture(tmp):
    """The forge fixture (make_fixture.py): (spec, pipeline), built once per session in a
    directory that outlives the test that built it (the spec names its shaped parts)."""
    if "spec" not in _FIXTURE:
        work = Path(tempfile.mkdtemp(prefix="orders-fixture-", dir=_SESSION["root"] or tmp))
        r = subprocess.run([sys.executable, str(HERE / "make_fixture.py"), str(work)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"make_fixture.py failed:\n{r.stdout}\n{r.stderr}"
        spec = json.loads((work / "netspec.json").read_text(encoding="utf-8"))
        pipeline = json.loads(Path(spec["input"]["pipeline"]).read_text(encoding="utf-8"))
        _FIXTURE.update(spec=spec, pipeline=pipeline, work=work)
    return json.loads(json.dumps(_FIXTURE["spec"])), json.loads(json.dumps(_FIXTURE["pipeline"]))


def baseline_emitter(tmp):
    """c849f8b's `backends/pytorch/` emitter (`render_templates.baseline_emitter`), or None
    with the reason when no reachable repository holds the commit."""
    try:
        return rt.baseline_emitter(tmp, BASELINE_COMMIT), None
    except rt.RenderError as exc:
        return None, str(exc)


# ================================================================ 1. one generation

def test_v23_is_the_only_generation():
    assert emit.EMIT_GENERATION == "v2.3"
    assert emit.GENERATIONS == ("v2.3",)
    raises(lambda: emit.emit({}, {"x": 1}, "/nonexistent", generation="v2.4"),
           "unknown emit generation", "v2.3")
    raises(lambda: emit.emit({}, {"x": 1}, "/nonexistent", generation="v2.2"),
           "v2.2", "removed at gate A", "PackageAdapters")
    raises(lambda: emit.emit_with({}, {"x": 1}, "/nonexistent",
                                  emit.EmitOptions(generation="v2.2")),
           "v2.2", "removed at gate A")
    assert emit.sources_for().templates == emit.HERE / "templates"
    assert not hasattr(emit, "FROZEN_V22") and not hasattr(emit, "inline_blocks")


def test_the_backend_registry_loads_the_emitter(tmp_path):
    """`backends.load()` — how emit_backend.py and every stage reach the emitter — executes
    emit.py without registering it in sys.modules; the emitter must work that way."""
    import backends
    mod = backends.load("pytorch")
    assert mod.EMIT_GENERATION == "v2.3" and hasattr(mod, "emit")
    src = mod.Sources.at(tmp_path / "stub")
    write_stub_backend(tmp_path / "stub")
    rec = mod.emit_with(rt.reference_spec("graph"), rt.scoped_pipeline(rt.reference_pipeline(),
                        rt.reference_spec("graph")), tmp_path / "out",
                        mod.EmitOptions(sources=src))
    assert "model/blocks_graph.py" in rec["files"] and rec["generation"] == "v2.3"
    spec, pipeline = orders_fixture(tmp_path)
    out = tmp_path / "orders"
    assert mod.emit(spec, pipeline, out, run_dir=out)["generation"] == "v2.3"


def test_a_v22_placeholder_in_a_template_is_refused(tmp_path):
    for key, where in (("BLOCKS", "blocks_"), ("LOSSES", "losses.py"), ("PRELUDE", "prelude"),
                       ("BUILDER", "builder.py")):
        src = write_stub_backend(tmp_path / f"stale-{key}")
        loop = src.templates / "loop.py"
        loop.write_text(loop.read_text(encoding="utf-8") + "\n# {{" + key + "}}\n",
                        encoding="utf-8")
        raises(lambda: emit_stub(tmp_path, rt.reference_spec("row"), sources=src,
                                 name=f"stale-{key}"),
               "templates/loop.py", "{{" + key + "}}", where, "D1 §4.1")
        assert not (tmp_path / f"stale-{key}" / PACKAGE).exists()


def _orders_network(root, spec, package=PACKAGE):
    """(state dict items, forward outputs on a seeded batch, eval and train) of the network the
    package at `root` builds from `spec` — through the Model adapter's own constructor,
    `build_model` (v2.3: `Network.build`)."""
    import torch
    with importable(root, package):
        import importlib
        net = importlib.import_module(f"{package}.model.network")
        ns = importlib.import_module(f"{package}.domain").NetSpec.from_dict(spec)
        model = net.build_model(ns)
        feats = spec["data_contract"]["features"]
        n_num = sum(1 for f in feats if f.get("role") == "numeric")
        cards = [int(f["cardinality"]) for f in feats if f.get("role") == "nominal"]
        g = torch.Generator().manual_seed(11)
        batch = {"x_num": torch.randn(32, n_num, generator=g),
                 "x_cat": torch.stack([torch.randint(0, c, (32,), generator=g) for c in cards],
                                      dim=1)}
        outs = []
        for mode in ("eval", "train"):
            getattr(model, mode)()
            torch.manual_seed(3)
            outs.append(model(batch).detach().clone())
        return [(k, v.clone()) for k, v in model.state_dict().items()], outs


def test_the_orders_network_is_the_one_c849f8b_built(tmp_path):
    """The cutover changed the layout, not the network: the live v2.3 package of the orders
    spec and c849f8b's own emission build the same parameters, in the same order, with the
    same seeded values, and answer the same forward pass (D1 §17.3's premise). c849f8b reads
    the regularization live paths, not the declaration (rule R2 came after it), so it builds
    from the spec resolved (`vega_regularization.resolve`: the declared rates written where
    its builder reads them) — the same network the v2.3 package builds from the declaration."""
    import torch
    spec, pipeline = orders_fixture(tmp_path)
    out = tmp_path / "v23"
    rec = emit.emit(spec, pipeline, out, run_dir=out)
    assert rec["block_files"] == {"blocks_core": ["dense", "embedding", "concat", "norm",
                                                  "simple", "composite", "head"]}
    assert not gate_errors(out / PACKAGE, pipeline)
    ref, why = baseline_emitter(tmp_path)
    if ref is None:
        raise AssertionError(f"cannot build the {BASELINE_COMMIT} reference: {why}")
    ref_out = tmp_path / "ref"
    resolved = vega_regularization.resolve(spec)
    ref.emit(resolved, pipeline, ref_out, run_dir=ref_out)
    mine, mine_out = _orders_network(out, spec)
    theirs, theirs_out = _orders_network(ref_out, resolved)
    assert [k for k, _ in mine] == [k for k, _ in theirs]
    differ = [k for (k, a), (_, b) in zip(mine, theirs) if not torch.equal(a, b)]
    assert not differ, f"the seeded initial weights moved: {differ}"
    assert all(torch.equal(a, b) for a, b in zip(mine_out, theirs_out))


def test_the_v22_sources_are_read_from_git():
    """`render_templates.baseline_source` gives c849f8b's own backend files — what the
    equivalence tests of losses, data rows and metrics compare against; the tree carries no
    copy of the v2.2 generation (removed at gate A)."""
    assert not (SCRIPTS / "backends" / "pytorch" / "generations").exists()
    for rel in ("losses.py", "templates/data.py", "templates/metrics.py"):
        try:
            text = rt.baseline_source(rel)
        except rt.RenderError as exc:
            raise AssertionError(f"cannot read {BASELINE_COMMIT}:{rel}: {exc}") from None
        assert text.strip(), rel
    try:
        rt.baseline_source("no/such/file.py")
    except rt.RenderError as exc:
        assert "no/such/file.py" in str(exc) and "VEGA_SOURCE_REPO" in str(exc)
    else:
        raise AssertionError("a path the commit does not hold was read")


def test_the_orders_network_equals_the_smoke_golden(tmp_path):
    """B01's capture of c849f8b's smoke: the parameter count and every block's shape and
    parameters, from the live package."""
    import torch
    golden = json.loads((HERE / "golden" / "orders-smoke.json").read_text(encoding="utf-8"))
    spec, pipeline = orders_fixture(tmp_path)
    out = tmp_path / "golden"
    emit.emit(spec, pipeline, out, run_dir=out)
    with importable(out):
        import importlib
        network = importlib.import_module(f"{PACKAGE}.model.network")
        ns = importlib.import_module(f"{PACKAGE}.domain").NetSpec.from_dict(spec)
        model = network.Network.build(ns)
        assert model.param_count() == (golden["params_total"], golden["params_trainable"])
        feats = spec["data_contract"]["features"]
        n_num = sum(1 for f in feats if f.get("role") == "numeric")
        cards = [int(f["cardinality"]) for f in feats if f.get("role") == "nominal"]
        bs = golden["batch_size"]
        batch = {"x_num": torch.randn(bs, n_num),
                 "x_cat": torch.stack([torch.randint(0, c, (bs,)) for c in cards], dim=1)}
        assert model.block_shapes(batch) == golden["per_block"]


# ================================================================ 2–3. the v2.3 file list

def _check_case(tmp_path, name):
    make_spec, expected = CASES[name]
    spec = make_spec()
    pipeline = rt.scoped_pipeline(rt.reference_pipeline(), spec)
    out, rec = emit_stub(tmp_path, spec, pipeline, name=name)
    root = out / PACKAGE
    got = files_of(root)
    assert got == expected, (f"{name}: extra {sorted(got - expected)}, "
                             f"missing {sorted(expected - got)}")
    assert (out / "spec_snapshot.json").is_file() and (out / "requirements.txt").is_file()
    assert rec["generation"] == "v2.3" and rec["blueprint_map"]["generation"] == "v2.3"
    # every file is mapped, and the rest of the mechanical gate holds on the stubs
    assert not c3_errors(root, pipeline)
    errors = gate_errors(root, pipeline)
    assert not errors, f"{name}: {errors}"
    _check_map(rec["blueprint_map"], expected)
    _check_builds(out, spec)
    return out, rec


def _check_map(bmap, expected):
    """One entry per file, each a support entry or a role, with the D1 §4.1 fields."""
    rows = {e["file"]: e for e in bmap["entries"]}
    assert set(rows) == expected - {"blueprint-map.json"}
    assert bmap["unmapped"] == []
    for rel, row in rows.items():
        assert row.get("kind") == "support" or row.get("role"), rel
    for stem in ("builder", "blocks_core"):
        row = rows[f"model/{stem}.py"]
        assert row["kind"] == "support" and row["used_by"] == ["Model"], row
        assert row["source"] == "backend-contract §1" and row["segments"], row
    assert rows["model/blocks_core.py"]["segments"][0] == "prelude"
    assert rows["model/readout.py"]["used_by"] == ["Model", "Trainer", "Evaluator", "Predictor"]
    assert rows["model/export.py"]["implements"] == "ArtifactExporter"
    assert rows["model/baselines_extra.py"]["role"] == "baseline"
    assert rows["training/losses.py"]["segments"] == ["losses"]


def _check_builds(out, spec):
    """The package imports, and the adapter builds every block of the spec through the
    builder."""
    with importable(out):
        import importlib
        network = importlib.import_module(f"{PACKAGE}.model.network")
        net = network.build_model(spec)
        assert len(net.stages) == len(spec["blocks"])
        losses = importlib.import_module(f"{PACKAGE}.training.losses")
        assert losses.build_loss(spec).formula().startswith("L =")


def test_v23_row_layout(tmp_path):
    out, rec = _check_case(tmp_path, "row")
    assert rec["sample_shape"] == "row"
    assert rec["block_files"] == {"blocks_core": ["dense", "embedding", "concat", "simple",
                                                  "head"]}


def test_v23_window_layout(tmp_path):
    out, rec = _check_case(tmp_path, "window")
    assert rec["block_files"]["blocks_seq"] == ["recurrent"]
    seq = (out / PACKAGE / "model" / "blocks_seq.py").read_text(encoding="utf-8")
    # the prelude's imports open the file, those it reads only (the stub's recurrent segment
    # reads no `math`: T5 round 2, a spliced file opened with every import of its source);
    # its definitions come from blocks_core
    assert "class Recurrent(Block)" in seq and "\nimport math\n" not in seq
    assert "def _act(" not in seq and "class Block" not in seq
    assert "from .blocks_core import Block, _act\n" in seq, seq
    row = next(e for e in rec["blueprint_map"]["entries"] if e["file"] == "model/blocks_seq.py")
    assert row["segments"] == ["recurrent"]


def all_block_files_spec():
    """A spec whose blocks land in all four block files."""
    spec = rt.reference_spec("row")
    spec["blocks"] = [
        {"id": "B1", "type": "dense", "params": {"units": 8, "activation": "relu"}},
        {"id": "B2", "type": "gru", "params": {"hidden": 8}},
        {"id": "B3", "type": "gcn_conv", "params": {"out": 8}},
        {"id": "B4", "type": "decoder", "params": {"out": 5}},
        {"id": "B5", "type": "head", "params": {"out": 1}}]
    return spec


def test_v23_the_prelude_is_defined_once(tmp_path):
    """The prelude is spliced whole into blocks_core only (SEGMENT_FILE routes it there); the
    other block files open with its imports and import its definitions, so a base class the
    prelude defines is one class, and a table the segments write to is one table."""
    spec = all_block_files_spec()
    out, rec = emit_stub(tmp_path, spec)
    model = out / PACKAGE / "model"
    assert set(rec["block_files"]) == set(emit.BLOCK_FILES)
    core = (model / "blocks_core.py").read_text(encoding="utf-8")
    assert "class Block:" in core and "REGISTRY = {}" in core and "def _prod(" in core
    for stem in ("blocks_seq", "blocks_graph", "blocks_gen"):
        text = (model / f"{stem}.py").read_text(encoding="utf-8")
        for defined in ("class Block", "REGISTRY", "def _act(", "def _prod("):
            assert defined not in text, (stem, defined)
        assert "from .blocks_core import Block" in text, (stem, text)
    assert not c3_errors(out / PACKAGE, rt.reference_pipeline())
    assert not gate_errors(out / PACKAGE, rt.reference_pipeline())
    with importable(out):
        import importlib
        mods = {s: importlib.import_module(f"{PACKAGE}.model.{s}") for s in emit.BLOCK_FILES}
        base = mods["blocks_core"].Block
        for stem in ("blocks_seq", "blocks_graph", "blocks_gen"):
            assert mods[stem].Block is base, stem
        net = importlib.import_module(f"{PACKAGE}.model.network").build_model(spec)
        assert all(isinstance(s, base) for s in net.stages[:4]), net.stages
        assert {"Dense", "Recurrent", "GCNConv", "Decoder"} <= set(mods["blocks_core"].REGISTRY)
        assert net.stages[1].act == "tanh"            # _act, imported from blocks_core


def test_v23_graph_node_layout(tmp_path):
    out, rec = _check_case(tmp_path, "graph")
    graph = (out / PACKAGE / "model" / "blocks_graph.py").read_text(encoding="utf-8")
    # GINConv names Dense, whose segment this spec does not use: left unresolved on purpose
    assert "import Dense" not in graph and "class GINConv" in graph


def test_v23_graph_level_layout(tmp_path):
    out, rec = _check_case(tmp_path, "graph-level")
    graph = (out / PACKAGE / "model" / "blocks_graph.py").read_text(encoding="utf-8")
    assert "from .blocks_core import Block, Dense\n" in graph, graph[:1200]
    assert rec["block_files"]["blocks_graph"] == ["graph_ops", "graph_conv", "graph_pool"]


def test_v23_generative_row_with_a_custom_block(tmp_path):
    out, rec = _check_case(tmp_path, "generative-custom")
    gen = (out / PACKAGE / "model" / "blocks_gen.py").read_text(encoding="utf-8")
    assert "from .blocks_core import Block, Head\n" in gen, gen[:1200]


def test_v23_without_an_evaluator(tmp_path):
    pipeline = rt.reference_pipeline()
    pipeline["ports"] = [p for p in pipeline["ports"] if p["role"] != "Evaluator"]
    out, rec = emit_stub(tmp_path, rt.reference_spec("row"), pipeline)
    expected = COMMON - {"evaluation/__init__.py", "evaluation/evaluate.py",
                         "model/baselines.py", "model/baselines_extra.py"}
    assert files_of(out / PACKAGE) == expected
    assert not gate_errors(out / PACKAGE, pipeline)
    readout = next(e for e in rec["blueprint_map"]["entries"] if e["file"] == "model/readout.py")
    assert readout["used_by"] == ["Model", "Trainer", "Predictor"]


def test_v23_re_emit_removes_only_its_own_stale_files(tmp_path):
    """A re-emit after the spec changed shape drops the files the earlier emission wrote and
    this one does not (by the earlier map), and nothing else."""
    src = write_stub_backend(tmp_path / "stub")
    out, _ = emit_stub(tmp_path, generative_custom_spec(), sources=src, name="same")
    cache = out / PACKAGE / "model" / "custom" / "__pycache__"
    cache.mkdir()                                        # as an import of the package leaves
    (cache / "custom_gain.cpython-312.pyc").write_bytes(b"")
    keep = out / PACKAGE / "model" / "notes.txt"         # not the emitter's: never touched
    keep.write_text("mine", encoding="utf-8")
    out, rec = emit_stub(tmp_path, rt.reference_spec("graph"), sources=src, name="same")
    assert files_of(out / PACKAGE) == (COMMON | GRAPH | {"model/notes.txt"})
    assert not (out / PACKAGE / "model" / "custom").exists()
    keep.unlink()
    assert not c3_errors(out / PACKAGE, rt.reference_pipeline())


def test_v23_a_refused_re_emit_leaves_the_package_as_it_was(tmp_path):
    """The unmapped guard runs before anything is written, and stale files go only after the
    new map is in place: a refused re-emit removes nothing its predecessor's map still lists."""
    src = write_stub_backend(tmp_path / "stub")
    out, _ = emit_stub(tmp_path, generative_custom_spec(), sources=src, name="same")
    root = out / PACKAGE
    before = {f: (root / f).read_bytes() for f in files_of(root)}
    assert "model/blocks_gen.py" in before and "model/custom/custom_gain.py" in before
    real = emit._blueprint_map_v23

    def drop_builder(*a, **kw):
        doc = real(*a, **kw)
        doc["entries"] = [e for e in doc["entries"] if e["file"] != "model/builder.py"]
        return doc

    with patched(emit, "_blueprint_map_v23", drop_builder):
        raises(lambda: emit_stub(tmp_path, rt.reference_spec("graph"), sources=src,
                                 name="same"), "model/builder.py", "no map entry")
    assert {f: (root / f).read_bytes() for f in files_of(root)} == before


def test_v23_generated_imports(tmp_path):
    out, _ = emit_stub(tmp_path, rt.reference_spec("graph-level"))
    model = out / PACKAGE / "model"
    builder = (model / "builder.py").read_text(encoding="utf-8")
    assert "from ..domain import NetSpec\n" in builder
    assert "from .blocks_core import" in builder and "from .blocks_graph import" in builder
    # the builder's own imports (its source's prelude) come first, the emitter's after them
    assert builder.index("from pathlib import Path") < builder.index("from ..domain import")
    assert builder.index("from ..domain import") < builder.index("class NetworkBuilder")
    # names of segments this spec does not use are not imported from anywhere
    assert "Recurrent" not in builder.split("class NetworkBuilder")[0]
    losses = (out / PACKAGE / "training" / "losses.py").read_text(encoding="utf-8")
    assert "from ..domain import NetSpec\n" in losses
    core = (model / "blocks_core.py").read_text(encoding="utf-8")
    assert "\nfrom ." not in core                      # the core imports no sibling
    assert core.startswith(f'"""\n{PACKAGE}.model.blocks_core — spliced by /vega-forge')
    assert "from __future__ import annotations" in core.split("import math")[0]


def test_a_module_publishes_what_crosses_its_boundary(tmp_path):
    """Final rehearsal round 2 (architect, 2 of 3 audits): the model module published only
    Network and ModelRegistry while the training and evaluation modules read its readout and
    metrics through files behind it, and the exporter's output had no home. The module's
    `__init__.py` now publishes what crosses its boundary, the other modules import it from
    there, and the map gives the exporter's output its owner."""
    spec = rt.reference_spec("row")
    out = tmp_path / "live-row"
    pkg = Path(emit.emit(spec, rt.scoped_pipeline(rt.reference_pipeline(), spec), out,
                         run_dir=out)["root"])
    with importable(out):
        import importlib
        model = importlib.import_module(f"{PACKAGE}.model")
        published = set(model.__all__)
        assert published == {"Network", "ModelRegistry", "build_model", "TaskReadout",
                             "AssembledBatch", "SampleAssembler", "BatchSignature",
                             "TorchExporter", "metrics"}, published   # types and one library
        assert model.metrics is importlib.import_module(f"{PACKAGE}.model.metrics")
        assert model.TaskReadout is importlib.import_module(f"{PACKAGE}.model.readout").TaskReadout
        assert model.build_model == model.Network.build
        assert "frames" not in importlib.import_module(f"{PACKAGE}.ingestion").__all__
    for rel in ("training/loop.py", "evaluation/evaluate.py"):
        text = (pkg / rel).read_text(encoding="utf-8")
        assert "from ..model import TaskReadout, metrics as _metrics\n" in text, rel
        assert "from ..model.readout" not in text, rel
    entries = json.loads((pkg / "blueprint-map.json").read_text(encoding="utf-8"))["entries"]
    export = next(e for e in entries if e["file"] == "model/export.py")
    assert export["output"]["owner"] == "/vega-pack" and "exports/" in export["output"]["where"]


# ================================================================ 4. the unmapped guard

def test_the_unmapped_guard_refuses_a_stray_file(tmp_path):
    raises(lambda: emit._refuse_unmapped(["model/stray.py"], {"entries": []}),
           "model/stray.py", "no map entry")

    # v2.3: the map loses the builder's entry -> the file it describes is a stray
    real = emit._blueprint_map_v23

    def drop_builder(*a, **kw):
        doc = real(*a, **kw)
        doc["entries"] = [e for e in doc["entries"] if e["file"] != "model/builder.py"]
        return doc

    with patched(emit, "_blueprint_map_v23", drop_builder):
        raises(lambda: emit_stub(tmp_path, rt.reference_spec("row"), name="stray-v23"),
               "model/builder.py", "no map entry")

    # the live sources: the map loses the exporter's entry
    spec, pipeline = orders_fixture(tmp_path)

    def drop_export(*a, **kw):
        doc = real(*a, **kw)
        doc["entries"] = [e for e in doc["entries"] if e["file"] != "model/export.py"]
        return doc

    with patched(emit, "_blueprint_map_v23", drop_export):
        raises(lambda: emit.emit(spec, pipeline, tmp_path / "stray-live"),
               "model/export.py", "no map entry")
    assert not (tmp_path / "stray-live" / PACKAGE).exists()


# ================================================================ 5. custom blocks ship

def test_custom_block_module_ships_in_the_package(tmp_path):
    out, rec = emit_stub(tmp_path, generative_custom_spec())
    shipped = out / PACKAGE / "model" / "custom" / "custom_gain.py"
    assert shipped.read_bytes() == (out / "custom" / "custom_gain.py").read_bytes()
    row = next(e for e in rec["blueprint_map"]["entries"]
               if e["file"] == "model/custom/custom_gain.py")
    assert row["kind"] == "support" and row["used_by"] == ["Model"] and row["blocks"] == ["B4"]
    assert row["code_ref"] == "custom/custom_gain.py" and row["digest"].startswith("sha256:")
    # the builder resolves it against the package: move the run dir's copy away and build
    (out / "custom" / "custom_gain.py").unlink()
    with importable(out):
        import importlib
        net = importlib.import_module(f"{PACKAGE}.model.network").build_model(
            generative_custom_spec())
        assert net.stages[3] == ("gain", (6,), [5])


def test_custom_block_refusals(tmp_path):
    src = write_stub_backend(tmp_path / "stub")
    spec = generative_custom_spec()
    spec["blocks"][3]["params"]["code_ref"] = "custom/absent.py"
    raises(lambda: emit_stub(tmp_path, spec, sources=src, name="absent"),
           "B4", "custom/absent.py", "not found", "R-B9")

    # one module, cited by two blocks: copied once, both blocks named
    spec = generative_custom_spec()
    spec["blocks"].insert(4, {"id": "B4b", "type": "custom", "source": "user:x",
                              "params": {"code_ref": "custom/custom_gain.py", "out_shape": [5]}})
    out, rec = emit_stub(tmp_path, spec, sources=src, name="twice")
    row = next(e for e in rec["blueprint_map"]["entries"] if "custom/" in e["file"])
    assert row["blocks"] == ["B4", "B4b"]

    # two modules with one stem would overwrite each other: refused
    other = tmp_path / "elsewhere" / "custom_gain.py"
    other.parent.mkdir(parents=True)
    other.write_text("def build(in_shape, params):\n    return None\n", encoding="utf-8")
    spec["blocks"][4]["params"]["code_ref"] = str(other)
    raises(lambda: emit_stub(tmp_path, spec, sources=src, name="clash"),
           "B4", "B4b", "custom_gain", "rename one")


def test_custom_code_ref_resolves_against_the_campaign_root_only(tmp_path):
    """netspec-contract §2: a relative code_ref resolves against the campaign root (emit's
    run_dir), never against the run directory the package is emitted into — a stale copy
    left in a rung's directory is not the block's code."""
    src = write_stub_backend(tmp_path / "stub")
    spec, pipeline = generative_custom_spec(), rt.reference_pipeline()
    campaign, rung = tmp_path / "campaign", tmp_path / "campaign" / "rungs" / "R1"
    for where, text in ((campaign, CUSTOM_GAIN), (rung, "# a stale copy\n" + CUSTOM_GAIN)):
        (where / "custom").mkdir(parents=True)
        (where / "custom" / "custom_gain.py").write_text(text, encoding="utf-8")

    def emit_rung(campaign_root):
        return emit.emit_with(spec, pipeline, rung, emit.EmitOptions(
            generation="v2.3", sources=src, campaign_root=campaign_root))

    emit_rung(campaign)
    shipped = rung / PACKAGE / "model" / "custom" / "custom_gain.py"
    assert shipped.read_text(encoding="utf-8") == CUSTOM_GAIN
    # the campaign root lacks it: the rung's copy is no fallback, and nothing is written
    (campaign / "custom" / "custom_gain.py").unlink()
    shipped.unlink()
    raises(lambda: emit_rung(campaign), "B4", "not found",
           str(campaign / "custom" / "custom_gain.py"), "campaign root")
    assert not shipped.exists()
    raises(lambda: emit_rung(None), "B4", "no campaign root", "netspec-contract §2")
    # the CLI names the flag, and has no default that points at --out
    spec_path = tmp_path / "netspec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(json.dumps(pipeline), encoding="utf-8")
    r = subprocess.run([sys.executable, str(emit.HERE / "emit.py"), "--spec", str(spec_path),
                        "--pipeline", str(pipe_path), "--out", str(rung), "--generation", "v2.3",
                        "--sources", str(tmp_path / "stub")], capture_output=True, text=True)
    assert r.returncode == 2 and "no campaign root" in r.stdout, r.stdout + r.stderr
    assert "--campaign-root" in r.stdout


# ================================================================ 6. the splice contract

def _variant(tmp_path, name, **edits):
    """A stub backend with one source or template edited: blocks/builder/losses=(old, new),
    or drop=<template name>."""
    src = write_stub_backend(tmp_path / name)
    for key in ("blocks", "builder", "losses"):
        if key in edits:
            old, new = edits[key]
            path = src.source(key)
            text = path.read_text(encoding="utf-8")
            assert old in text, (key, old)
            path.write_text(text.replace(old, new), encoding="utf-8")
    if "drop" in edits:
        (src.templates / edits["drop"]).unlink()
    if "domain" in edits:
        old, new = edits["domain"]
        path = src.templates / "domain.py"
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    return src


def test_splice_refusals_on_the_tables(tmp_path):
    row = rt.reference_spec("row")

    src = _variant(tmp_path, "no-segment-file", blocks=("SEGMENT_FILE = {", "_UNUSED = {"))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="a"), "SEGMENT_FILE", "B03")

    src = _variant(tmp_path, "unrouted", blocks=('"concat": "blocks_core", ', ""))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="b"),
           "'concat'", "SEGMENT_FILE", "blocks_core, blocks_seq, blocks_graph, blocks_gen")

    src = _variant(tmp_path, "fifth-file", blocks=('"concat": "blocks_core"',
                                                   '"concat": "blocks_extra"'))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="c"), "blocks_extra", "D1 §4.1")

    # the prelude goes whole into blocks_core, the one block file every package has
    src = _variant(tmp_path, "prelude-elsewhere", blocks=('"prelude": "blocks_core"',
                                                          '"prelude": "blocks_seq"'))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="c2"),
           "prelude", "'blocks_seq'", "blocks_core", "exists once")
    src = _variant(tmp_path, "prelude-unrouted", blocks=('"prelude": "blocks_core", ', ""))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="c3"),
           "prelude", "'no file'", "route it to 'blocks_core'")

    src = _variant(tmp_path, "unmarked", blocks=('"concat": ["concat"]', '"concat": ["joiner"]'))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="d"), "joiner", "does not mark")

    # a bare annotation assigns nothing: the table is still missing
    src = _variant(tmp_path, "bare-annotation",
                   blocks=("SEGMENT_FILE = {", "SEGMENT_FILE: dict\n_UNUSED = {"))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="l"), "SEGMENT_FILE", "B03")


def test_splice_refusals_on_names_and_files(tmp_path):
    row = rt.reference_spec("row")

    src = _variant(tmp_path, "unknown-name",
                   builder=("return Concat()", "return Frobnicator()"))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="e"),
           "model/builder.py", "Frobnicator", "its own imports")

    # a framework import the segment forgot is not guessed either
    src = _variant(tmp_path, "no-import", losses=("# --8<-- losses\nimport math\n",
                                                  "# --8<-- losses\n"))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="f"),
           "training/losses.py", "math")

    # domain.py's private helpers are not the package's to import
    src = _variant(tmp_path, "private",
                   builder=("return Concat()", "return _private_helper(Concat())"))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="g"), "_private_helper")

    # the core's head names Recurrent (blocks_seq) and the recurrent block names Head
    # (blocks_core): each file would import the other
    src = write_stub_backend(tmp_path / "cycle")
    text = src.blocks.read_text(encoding="utf-8")
    text = text.replace("        self.out = int(out)\n",
                        "        self.out = int(out)\n        self.rnn = Recurrent('gru', out)\n")
    text = text.replace("        self.kind, self.hidden = kind, int(hidden)\n",
                        "        self.kind, self.hidden = kind, int(hidden)\n"
                        "        self.head = Head(hidden)\n")
    src.blocks.write_text(text, encoding="utf-8")
    raises(lambda: emit_stub(tmp_path, rt.reference_spec("window"), sources=src, name="h"),
           "cycle", "blocks_core.py", "blocks_seq.py")

    src = _variant(tmp_path, "no-readout", drop="readout.py")
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="i"),
           "model/readout.py", "readout.py", "D1 §4.1", "restore it")

    src = _variant(tmp_path, "bad-segment", blocks=("class Concat:\n    width = 0\n",
                                                    "class Concat(:\n    width = 0\n"))
    raises(lambda: emit_stub(tmp_path, row, sources=src, name="j"), "does not parse")

    spec = rt.reference_spec("row")
    spec["data_contract"]["sample"]["shape"] = "cube"
    raises(lambda: emit_stub(tmp_path, spec, name="k"), "cube", "row, window, graph")


def test_the_tables_may_be_annotated(tmp_path):
    """`SEGMENTS_FOR: dict[str, list[str]] = {...}` is the same table as the plain assignment."""
    src = _variant(tmp_path, "annotated", blocks=("SEGMENT_FILE = {",
                                                  "SEGMENT_FILE: dict[str, str] = {"))
    text = src.blocks.read_text(encoding="utf-8").replace(
        "SEGMENTS_FOR = {", "SEGMENTS_FOR: dict[str, list[str]] = {")
    src.blocks.write_text(text, encoding="utf-8")
    assert emit._module_literal(src.blocks, "SEGMENT_FILE")["recurrent"] == "blocks_seq"
    out, rec = emit_stub(tmp_path, rt.reference_spec("window"), sources=src)
    assert rec["block_files"]["blocks_seq"] == ["recurrent"]
    assert not c3_errors(out / PACKAGE, rt.reference_pipeline())


def test_v23_real_block_files_share_one_prelude(tmp_path):
    """Every block file of the real library (the live `blocks.py`): the prelude's helpers
    exist once, in blocks_core, and the other files import them from there."""
    import torch
    spec, pipeline = orders_fixture(tmp_path)
    wide = json.loads(json.dumps(spec))
    wide["blocks"] = [{"id": f"W{i}", "type": t, "params": {}}
                      for i, t in enumerate(("dense", "conv1d", "gru", "attention", "decoder",
                                             "gcn_conv", "graph_pool", "head"))]
    wide_out = tmp_path / "v23-wide"
    rec = emit.emit(wide, pipeline, wide_out)
    assert set(rec["block_files"]) == set(emit.BLOCK_FILES)
    for stem in ("blocks_seq", "blocks_gen", "blocks_graph"):
        text = (wide_out / PACKAGE / "model" / f"{stem}.py").read_text(encoding="utf-8")
        assert "def _act(" not in text and "def _prod(" not in text, (stem, text[:1500])
    with importable(wide_out):
        import importlib
        core = importlib.import_module(f"{PACKAGE}.model.blocks_core")
        for stem in ("blocks_seq", "blocks_gen", "blocks_graph"):
            mod = importlib.import_module(f"{PACKAGE}.model.{stem}")
            for name in ("_ACT", "_act", "_prod"):
                if hasattr(mod, name):
                    assert getattr(mod, name) is getattr(core, name), (stem, name)
            assert mod.nn is torch.nn                    # the prelude's imports open it
        builder = (wide_out / PACKAGE / "model" / "builder.py").read_text(encoding="utf-8")
        assert "from ..domain import" in builder and "from .blocks_core import" in builder


def _gate(root, pipeline, *extra):
    """conformance.py's CLI on a package: (exit code, output)."""
    import tempfile as _tf
    with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(pipeline, fh)
    r = subprocess.run([sys.executable, str(SCRIPTS / "conformance.py"), "--package", str(root),
                        "--pipeline", fh.name, *extra], capture_output=True, text=True)
    Path(fh.name).unlink()
    return r.returncode, r.stdout + r.stderr


def test_v23_on_the_live_sources(tmp_path):
    """The live sources emit, for every reference shape and the orders spec, a mapped package
    that passes the whole mechanical gate with C11–C14 at warn (conformance exits 0)."""
    cases = [(shape, rt.reference_spec(shape),
              rt.scoped_pipeline(rt.reference_pipeline(), rt.reference_spec(shape)))
             for shape in ("row", "window", "graph", "graph-level")]
    cases.append(("orders", *orders_fixture(tmp_path)))
    for name, spec, pipeline in cases:
        out = tmp_path / f"live-{name}"
        rec = emit.emit(spec, pipeline, out, run_dir=out)
        root = Path(rec["root"])
        assert not c3_errors(root, pipeline), name
        code, text = _gate(root, pipeline)
        assert code == 0, f"{name}: conformance exited {code}:\n{text}"
        assert "OK: C14" in text, text


def test_the_baselines_of_one_port_may_import_each_other(tmp_path):
    """C5 with PLAN-AMENDMENTS A14: model/baselines_extra.py imports model/baselines.py (role
    baseline, one port; one way since the T5 round 1 audit) and the gate passes; a baseline
    that imports a primary adapter is still refused."""
    pipeline = rt.reference_pipeline()
    out = tmp_path / "c5"
    rec = emit.emit(rt.reference_spec("row"), pipeline, out)
    root = Path(rec["root"])
    extra = (root / "model" / "baselines_extra.py").read_text(encoding="utf-8")
    base = (root / "model" / "baselines.py").read_text(encoding="utf-8")
    assert "from .baselines import" in extra and "import baselines_extra" not in base
    assert not [e for e in gate_errors(root, pipeline) if e.startswith("C5")]
    marker = "from ..domain import "
    (root / "model" / "baselines.py").write_text(
        base.replace(marker, "from ..training.loop import Trainer  # a primary adapter\n"
                     + marker, 1), encoding="utf-8")
    errors = [e for e in gate_errors(root, pipeline) if e.startswith("C5")]
    assert any("training/loop.py" in e and "never on another adapter" in e for e in errors), \
        errors


def test_ports_fall_back_to_the_v23_signatures(tmp_path):
    """A blueprint with no signature[] gets the v2.3 fallback, with the WARN: each method within
    five parameters beside self, Model.score, the DataSource's design_matrix / exposure /
    target_scale, Trainer.fit(..., run=), the Evaluator's scored — and no Inspectable
    protocol (R-B5)."""
    import ast
    pipeline = rt.reference_pipeline()
    for port in pipeline["ports"]:
        port.pop("signature", None)
    markers = []
    text = emit.ports_py(PACKAGE, pipeline, emit.plan(rt.reference_spec("row"), pipeline),
                         markers)
    assert len(markers) == len(pipeline["ports"]) and all(m.startswith("WARN:") for m in markers)
    tree = ast.parse(text)
    methods = {f"{c.name}.{f.name}": f for c in tree.body if isinstance(c, ast.ClassDef)
               for f in c.body if isinstance(f, ast.FunctionDef)}
    for name, fn in methods.items():
        params = [a for a in fn.args.args + fn.args.kwonlyargs if a.arg not in ("self", "cls")]
        assert len(params) <= 5 and not fn.args.vararg and not fn.args.kwarg, name
    for need in ("Model.score", "DataSource.design_matrix", "DataSource.exposure",
                 "DataSource.target_scale", "DataSource.load_sealed_part", "Evaluator.scored",
                 "Recipe.edges_for", "Recipe.edges_within", "ArtifactExporter.export"):
        assert need in methods, need
    fit = methods["Trainer.fit"]
    assert [a.arg for a in fit.args.args] == ["self", "model", "spec", "data", "callbacks", "run"]
    assert "Inspectable" not in text and "describe" not in text
    imported = text.split("if TYPE_CHECKING:")[1].split("\n\n")[0]
    for name in ("RunContext", "SamplePart", "ScoredPart", "DesignView", "GraphExposure"):
        assert name in imported, name
    assert all(len(line) <= 99 for line in text.splitlines() if "import" in line)


def _row_domain():
    """The pytorch domain.py template with the row shape's sections kept (the emitter's cut)."""
    path = emit.HERE / "templates" / "domain.py"
    return emit.select_shapes(path.read_text(encoding="utf-8"), ("row",), str(path))


def _signed(pipeline, role, *methods):
    for port in pipeline["ports"]:
        if port["role"] == role:
            port["signature"] = list(methods)
    return pipeline


def test_ports_import_every_held_name_a_signature_reads():
    """T5 round 2: a §3 that annotated with held names (MetricOptions, Versions, BatchKey,
    EpochRecord, FeatureContract) beyond the old fixed PORT_TYPES rendered a ports.py that did
    not resolve. Given the rendered domain.py, every name it holds imports, and so does a
    typing name the signatures read (Sequence), and the file resolves."""
    import ast
    pipeline = _signed(rt.reference_pipeline(), "Evaluator",
                       "def metrics(self, scored: ScoredPart, options: MetricOptions) -> Metrics: ...",
                       "def contract(self) -> FeatureContract: ...")
    _signed(pipeline, "Predictor", "def predict(self, rows) -> list[Prediction]: ...",
            "def versions(self) -> Versions: ...")
    _signed(pipeline, "DataSource",
            "def iter_batches(self, which: Part, keys: Sequence[BatchKey]) -> Iterator[dict]: ...")
    _signed(pipeline, "Trainer", "def history(self) -> Iterator[EpochRecord]: ...")
    plan_ = emit.plan(rt.reference_spec("row"), pipeline)
    text = emit.ports_py(PACKAGE, pipeline, plan_, [], _row_domain(), "row")
    imported = text.split("if TYPE_CHECKING:")[1].split("\n\n")[0]
    for name in ("MetricOptions", "Versions", "BatchKey", "EpochRecord", "FeatureContract",
                 "ScoredPart", "Metrics", "Prediction", "Part"):
        assert name in imported, (name, imported)
    assert "Sequence" in text.split("if TYPE_CHECKING:")[0]
    free, annotation_only = emit.free_names(ast.parse(text))
    assert not (free | annotation_only), free | annotation_only
    assert all(len(line) <= 99 for line in text.splitlines() if "import" in line)
    # without a typing name beyond Any/Iterator the head is the one it always was
    plain = emit.ports_py(PACKAGE, rt.reference_pipeline(),
                          emit.plan(rt.reference_spec("row"), rt.reference_pipeline()), [],
                          _row_domain(), "row")
    assert "from typing import TYPE_CHECKING, Any, Iterator, Protocol\n" in plain
    # final rehearsal round 2 (improve, 3 of 3 audits): no Protocol is @runtime_checkable — an
    # isinstance test against one checks names, never signatures
    assert "runtime_checkable" not in plain.split('"""', 2)[2]


def test_a_framework_type_in_a_signature_is_refused_as_the_blueprints():
    """T5 round 2: §3 annotated with DataFrame and Tensor; the refusal blamed the templates.
    It now names the blueprint, each port and name, the vocabulary and the fix."""
    pipeline = _signed(rt.reference_pipeline(), "Predictor",
                       "def predict(self, rows: DataFrame) -> list[Prediction]: ...")
    _signed(pipeline, "Model", "def score(self, batch: dict[BatchKey, Tensor]) -> Tensor: ...")
    plan_ = emit.plan(rt.reference_spec("row"), pipeline)
    try:
        emit.ports_py(PACKAGE, pipeline, plan_, [], _row_domain(), "row")
    except emit.EmitError as exc:
        msg = str(exc)
    else:
        raise AssertionError("a DataFrame annotation was rendered")
    assert "the blueprint's port signatures" in msg and "(Predictor) names DataFrame" in msg, msg
    assert "(Model) names Tensor" in msg and "BatchKey" not in msg.split("names Tensor")[0][-40:]
    assert "held_artifacts.py --shape row" in msg and "MetricOptions" in msg
    assert "/blueprint --concern" in msg and "fix the template" not in msg


def test_a_port_method_without_self_or_unparsable_is_refused_as_the_blueprints():
    pipeline = _signed(rt.reference_pipeline(), "FeatureTransformer",
                       "def transform(df_raw, history=None): ...")
    plan_ = emit.plan(rt.reference_spec("row"), pipeline)
    raises(lambda: emit.ports_py(PACKAGE, pipeline, plan_, [], _row_domain(), "row"),
           ".transform", "`self`")
    pipeline = _signed(rt.reference_pipeline(), "Monitor", "def check(self, rows) -> : ...")
    plan_ = emit.plan(rt.reference_spec("row"), pipeline)
    raises(lambda: emit.ports_py(PACKAGE, pipeline, plan_, [], _row_domain(), "row"),
           "does not parse", "Monitor")


def test_the_role_table_is_the_adapters_interface_in_every_shape():
    """Every port method of the role table (the fallback, and what port_surface.py prints) is
    a public method of the role's adapter with the same parameters, in every shape."""
    for shape in emit.SAMPLE_SHAPES:
        for role in emit.ROLE_SIGNATURES:
            surface = emit.adapter_surface(role, shape)
            for name, params in emit._role_port(role, shape).items():
                assert name in surface, (shape, role, name)
                assert surface[name].params == params, (shape, role, name, surface[name])


def _drawn(pipeline, role, edit):
    for port in pipeline["ports"]:
        if port["role"] == role:
            port["signature"] = edit(list(port["signature"]))
    return pipeline


def test_a_port_that_is_not_its_adapters_interface_is_refused_before_writing(tmp_path):
    """T5 round 2: §3 drew iter_batches(self, spec, part, plan) while the adapter implements
    (spec, part, batch_size, columns, plan), and the Evaluator and the Registry without the
    methods the stages call. The emitter refuses each, by port and method, as the blueprint's."""
    narrow = "def iter_batches(self, spec: NetSpec, part: Part, plan: BatchPlan | None = None) -> Iterator[dict]: ..."
    pipeline = _drawn(rt.reference_pipeline(), "DataSource",
                      lambda sig: [narrow if "iter_batches" in m else m for m in sig])
    msg = raises(lambda: emit_stub(tmp_path, rt.reference_spec("row"), pipeline),
                 "DataSource.iter_batches(spec, part, plan=…)",
                 "implements iter_batches(spec, part, batch_size=…, columns=…, plan=…)",
                 "port_surface.py --shape row", "/blueprint --concern")
    assert "fix the template" not in msg
    assert not (tmp_path / "run" / PACKAGE).exists()
    pipeline = _drawn(rt.reference_pipeline(), "Evaluator",
                      lambda sig: [m for m in sig if "measure" not in m and "heads" not in m])
    raises(lambda: emit_stub(tmp_path, rt.reference_spec("row"), pipeline, name="b"),
           "omits heads, measure", "Evaluator port carries")
    pipeline = _drawn(rt.reference_pipeline(), "ModelStore",
                      lambda sig: sig + ["def versions(self) -> list: ..."])
    raises(lambda: emit_stub(tmp_path, rt.reference_spec("row"), pipeline, name="c"),
           "Registry.versions", "ModelRegistry has no public method versions")
    # a method of the adapter beside the port may be declared, with its own parameters
    pipeline = _drawn(rt.reference_pipeline(), "Model",
                      lambda sig: sig + ["def named_stages(self) -> Any: ..."])
    out, _rec = emit_stub(tmp_path, rt.reference_spec("row"), pipeline, name="d")
    assert "def named_stages(self) -> Any" in (out / PACKAGE / "ports.py").read_text(encoding="utf-8")
    # the Trainer port is fit, evaluate and predict: the adapter publishes no helper beside them
    pipeline = _drawn(rt.reference_pipeline(), "Trainer",
                      lambda sig: sig + ["def lrs_for(self, spec: NetSpec, steps_per_epoch: int = 1) -> list: ..."])
    raises(lambda: emit_stub(tmp_path, rt.reference_spec("row"), pipeline, name="d2"),
           "Trainer has no public method lrs_for")
    # a graph recipe's edge entry points belong to a graph port only
    pipeline = _drawn(rt.reference_pipeline(), "FeatureTransformer",
                      lambda sig: sig + ["def edges_within(self, df_raw, history=None) -> Any: ..."])
    raises(lambda: emit_stub(tmp_path, rt.reference_spec("row"), pipeline, name="e"),
           "edges_within")


def test_port_surface_prints_the_role_table_and_the_backends_protocols():
    for shape in emit.SAMPLE_SHAPES:
        r = subprocess.run([sys.executable, str(emit.HERE.parents[1] / "port_surface.py"),
                            "--shape", shape, "--json"], capture_output=True, text=True,
                           timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        doc = json.loads(r.stdout)
        for role in emit.ROLE_SIGNATURES:
            port = [m for m in emit._role_lines(role, shape)
                    if not (role == "FeatureTransformer" and "edges_" in m)]
            assert sorted(doc[role]["port"]) == sorted(port), (shape, role)
        assert set(doc["backend"]) == set(emit.BACKEND_PROTOCOLS)
        edges = [l for l in doc["FeatureTransformer"]["beside"] if "edges_" in l]
        assert bool(edges) == (shape == "graph"), (shape, edges)
        # the arrows the code needs: what C4 would otherwise refuse (T5 round 2's net check)
        assert "Model" in doc["arrows"]["Trainer"] and "Model" in doc["arrows"]["Evaluator"]
        assert ("DataSource" in doc["arrows"]["Trainer"]) == (shape == "graph")
    r = subprocess.run([sys.executable, str(emit.HERE.parents[1] / "port_surface.py")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "OK: the port surface of a row package" in r.stdout
    assert "Trainer — adapter Trainer" in r.stdout and "def version(self, path)" in r.stdout


def test_prune_imports_keeps_what_a_splice_reads():
    body = ("import math\nimport importlib.util\nfrom dataclasses import dataclass, replace\n"
            "from typing import (Any, Iterator,\n                    Mapping)\n\n\n"
            "def f(x: Mapping) -> Any:\n    return replace(x) if importlib.util else None\n")
    out = emit.prune_imports(body)
    assert "import math" not in out and "import importlib.util\n" in out
    assert "from dataclasses import replace\n" in out and "dataclass," not in out
    assert "from typing import Any, Mapping\n" in out and "Iterator" not in out
    compile(out, "pruned", "exec")


def test_a_package_carries_its_task_kinds_code_only(tmp_path):
    """T5 round 2's gate: a multiclass row package shipped every task's readings, losses and
    reconstruction baselines. The `task` sections are cut to the spec's effective kind."""
    spec = rt.reference_spec("row")
    out, rec = emit_stub(tmp_path, spec)
    kind = emit.effective_task(spec)
    text = emit.select_tasks("a\n# >>> task: forecast\nb\n# <<< task\n"
                             "# >>> task: %s regression\nc\n# <<< task\n" % kind, (kind,))
    assert text == "a\nc\n"
    raises(lambda: emit.select_tasks("# >>> task: nope\n# <<< task\n", None, "t.py"),
           "t.py:1", "names nope")
    raises(lambda: emit.effective_task({"task": {"kind": "sequence-regression"}}),
           "effective None")
    real, _ = emit_stub(tmp_path, spec, sources=emit.Sources.at(emit.HERE), name="real")
    model, training = real / PACKAGE / "model", real / PACKAGE / "training"
    readout = (model / "readout.py").read_text(encoding="utf-8")
    losses = (training / "losses.py").read_text(encoding="utf-8")
    assert kind == "binary" and "class _Binary(_Reading)" in readout
    assert "class _Forecast" not in readout and "class _AnomalySvdd" not in readout
    assert "class BCEWithLogits" in losses and "class Quantile" not in losses
    assert "class SVDD" not in losses and "# >>> task" not in readout + losses
    extra = (model / "baselines_extra.py").read_text(encoding="utf-8")
    assert "def _mean_row(" not in extra and "def _naive(" not in extra
    assert "def _graph_linear(" not in extra and "def _lens_form(" in extra
    # the metrics carry the package's reading and its metric families, no other kind's
    metrics = (model / "metrics.py").read_text(encoding="utf-8")
    assert "class _Logit(ScoreReading)" in metrics and "class _Softmax" not in metrics
    assert "def roc_auc(" in metrics and "# >>> task" not in metrics
    assert not [f for f in ("def rmse(", "def qwk(", "def mase(", "def reconstruction_error(")
                if f in metrics]
    assert emit.effective_task({"task": {"kind": "sequence-classification",
                                         "base_kind": "multiclass"}}) == "multiclass"


def test_emit_cli(tmp_path):
    src_root = tmp_path / "stub"
    write_stub_backend(src_root)
    spec_path = tmp_path / "netspec.json"
    spec_path.write_text(json.dumps(rt.reference_spec("window")), encoding="utf-8")
    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(json.dumps(rt.scoped_pipeline(rt.reference_pipeline(),
                                                       rt.reference_spec("window"))),
                         encoding="utf-8")
    base = [sys.executable, str(emit.HERE / "emit.py"), "--spec", str(spec_path),
            "--pipeline", str(pipe_path)]
    r = subprocess.run(base + ["--out", str(tmp_path / "cli"), "--generation", "v2.3",
                               "--sources", str(src_root)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "generation v2.3" in r.stdout
    assert (tmp_path / "cli" / PACKAGE / "model" / "blocks_seq.py").is_file()
    r = subprocess.run(base + ["--out", str(tmp_path / "cli2"), "--generation", "v2.4"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout.startswith("ERROR:") and "v2.4" in r.stdout
    r = subprocess.run(base + ["--out", str(tmp_path / "cli2"), "--generation", "v2.2"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "removed at gate A" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(base + ["--out", str(tmp_path / "cli3"), "--generation", "v2.3",
                               "--sources", str(tmp_path / "nowhere")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout.startswith("ERROR:"), r.stdout + r.stderr


# ================================================================ 7. render_templates.py

def test_render_templates_full_is_the_emitter(tmp_path):
    src = write_stub_backend(tmp_path / "stub")
    spec = rt.reference_spec("graph")
    rec = rt.render(src.templates, tmp_path / "full", "reference", spec, full=True,
                    sources=str(tmp_path / "stub"))
    assert rec["mode"] == "full" and rec["generation"] == "v2.3"
    out, _ = emit_stub(tmp_path, spec, sources=src, name="direct")
    a, b = tmp_path / "full" / PACKAGE, out / PACKAGE
    assert files_of(a) == files_of(b)
    assert all((a / f).read_bytes() == (b / f).read_bytes() for f in files_of(a))
    # --full is exactly the emitter: a partial option is refused, never dropped
    for option in ({"defines": {"ENV": "X"}}, {"only": ["domain.py"]}, {"into": "Predictor"},
                   {"splice": True}, {"recursive": True}):
        flag = rt._PARTIAL_FLAGS[next(iter(option))]
        raises(lambda: rt.render(src.templates, tmp_path / "refused", "reference", spec,
                                 full=True, sources=str(tmp_path / "stub"), **option),
               "--full", flag)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert rt.main([str(src.templates), str(tmp_path / "refused"), "--full", "--sources",
                        str(tmp_path / "stub"), "--define", "ENV=X"]) == 2
    assert "--define" in printed.getvalue()
    assert not (tmp_path / "refused").exists()


def test_render_templates_full_resolves_custom_code_against_the_campaign_root(tmp_path):
    src = write_stub_backend(tmp_path / "stub")
    campaign = tmp_path / "campaign"
    (campaign / "custom").mkdir(parents=True)
    (campaign / "custom" / "custom_gain.py").write_text(CUSTOM_GAIN, encoding="utf-8")
    spec_path = campaign / "netspec.json"
    spec_path.write_text(json.dumps(generative_custom_spec()), encoding="utf-8")
    out = tmp_path / "rendered"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        # the default campaign root is the directory of the --spec file
        assert rt.main([str(src.templates), str(out), "--full", "--spec", str(spec_path),
                        "--sources", str(tmp_path / "stub")]) == 0, printed.getvalue()
        assert (out / PACKAGE / "model" / "custom" / "custom_gain.py").is_file()
        # a reference spec has no file, so no default root; and a partial render copies none
        assert rt.main([str(src.templates), str(tmp_path / "p"), "--campaign-root",
                        str(campaign)]) == 2
    assert "--campaign-root applies to --full only" in printed.getvalue()


def _render_forge_set(tmp_path):
    """The stub forge set, rendered partially for the ingestion adapter of a window spec."""
    src = write_stub_backend(tmp_path / "stub")
    out = tmp_path / "partial"
    rt.render(src.templates, out, "reference", "window",
              only=["domain.py", "frames.py", "windows.py", "data.py"])
    return src, out


def test_render_templates_partial(tmp_path):
    src, out = _render_forge_set(tmp_path)
    root = out / PACKAGE
    assert files_of(root) == {"__init__.py", "domain.py", "ports.py", "blueprint-map.json",
                              "ingestion/__init__.py", "ingestion/data.py",
                              "ingestion/frames.py", "ingestion/windows.py"}
    assert not c3_errors(root, rt.reference_pipeline())
    assert "{{" not in (root / "ingestion" / "frames.py").read_text(encoding="utf-8")
    with importable(out):
        import importlib
        data = importlib.import_module(f"{PACKAGE}.ingestion.data")
        raised = False
        try:
            data.load_part(None, "test")
        except Exception as exc:   # noqa: BLE001 — the rendered domain's SealedPartError
            raised = type(exc).__name__ == "SealedPartError"
        assert raised


def test_render_templates_layers_a_set_the_table_does_not_know(tmp_path):
    """A file the table does not know: refused without --into, placed with it; layered over
    the package a first render left."""
    src, out = _render_forge_set(tmp_path)
    root = out / PACKAGE
    serving = tmp_path / "serving-set"
    (serving / "inspection").mkdir(parents=True)
    (serving / "codec.py").write_text('"""{{PACKAGE}} codec, monitor in {{MONITOR_MODULE}}."""\n',
                                      encoding="utf-8")
    (serving / "inspection" / "readers.py").write_text('"""{{PACKAGE}} readers ({{ENV}})."""\n',
                                                       encoding="utf-8")
    raises(lambda: rt.render(serving, out, only=["codec.py"]), "codec.py", "--into")
    raises(lambda: rt.render(serving, out, into="Predictor", recursive=True), "ENV", "--define")
    rt.render(serving, out, into="Predictor", recursive=True, defines={"ENV": "ORDERS"})
    got = files_of(root)
    assert {"serving/codec.py", "serving/inspection/readers.py", "serving/__init__.py",
            "ingestion/frames.py"} <= got, got
    assert not c3_errors(root, rt.reference_pipeline())
    assert "monitor in serving" in (root / "serving" / "codec.py").read_text(encoding="utf-8")
    assert "(ORDERS)" in (root / "serving" / "inspection" / "readers.py").read_text(
        encoding="utf-8")


def test_render_templates_splice(tmp_path):
    """--splice adds the spliced support files over the stub sources."""
    src = write_stub_backend(tmp_path / "stub")
    out = tmp_path / "spliced"
    rt.render(src.templates, out, "reference", "graph-level", only=["network.py"],
              splice=True, sources=str(tmp_path / "stub"))
    got = files_of(out / PACKAGE)
    assert {"model/network.py", "model/builder.py", "model/blocks_core.py",
            "model/blocks_graph.py", "training/losses.py"} <= got, got
    with importable(out):
        import importlib
        net = importlib.import_module(f"{PACKAGE}.model.network").build_model(
            rt.reference_spec("graph-level"))
        assert len(net.stages) == 3


def test_render_templates_cli(tmp_path):
    src = write_stub_backend(tmp_path / "stub")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert rt.main([str(src.templates), str(tmp_path / "cli"), "--only", "domain.py",
                        "--json"]) == 0
        assert rt.main([str(tmp_path / "no-such-set"), str(tmp_path / "cli")]) == 2
    first, _, second = printed.getvalue().partition("ERROR:")
    assert json.loads(first)["files"] == ["__init__.py", "blueprint-map.json", "domain.py",
                                          "ports.py"]
    assert "no-such-set" in second and "not a directory" in second


def test_reference_specs_have_their_shapes():
    for shape, want in (("row", "row"), ("window", "window"), ("graph", "graph"),
                        ("graph-level", "graph"), ("reference", "row")):
        spec = rt.reference_spec(shape)
        assert emit.sample_shape(spec) == want
        assert emit.plan(spec, rt.reference_pipeline())["package"] == PACKAGE


# ================================================================ one sample shape per package

_SECTIONED = """head
# >>> shape: row
row only
# <<< shape


# >>> shape: window graph
window or graph
    # >>> shape: graph
    graph inside
    # <<< shape
# <<< shape


tail
"""


def test_select_shapes_keeps_the_carried_sections():
    assert emit.select_shapes(_SECTIONED, ("row",)) == "head\nrow only\n\n\ntail\n"
    assert emit.select_shapes(_SECTIONED, ("window",)) == "head\n\n\nwindow or graph\n\n\ntail\n"
    assert emit.select_shapes(_SECTIONED, ("graph",)) == \
        "head\n\n\nwindow or graph\n    graph inside\n\n\ntail\n"
    every = emit.select_shapes(_SECTIONED)
    assert every == emit.select_shapes(_SECTIONED, emit.SAMPLE_SHAPES)
    assert ">>>" not in every and "<<<" not in every and "graph inside" in every
    # prose that quotes a marker is not one
    quoted = "text that names `# >>> shape: row` inline\n"
    assert emit.select_shapes(quoted, ("graph",)) == quoted


def test_select_shapes_refuses_a_malformed_section():
    for text, words in (("# >>> shape: rows\nx\n# <<< shape\n", ("rows", "row, window, graph")),
                        ("# >>> shape:\nx\n# <<< shape\n", ("no shape",)),
                        ("x\n# <<< shape\n", ("closes no section",)),
                        ("# >>> shape: row\nx\n", ("never closed",))):
        try:
            emit.select_shapes(text, ("row",), "t.py")
        except emit.EmitError as exc:
            assert all(w in str(exc) for w in words) and "t.py:" in str(exc), str(exc)
        else:
            raise AssertionError(f"select_shapes accepted {text!r}")
    try:
        emit.select_shapes("x\n", ("rows",))
    except emit.EmitError as exc:
        assert "rows" in str(exc)
    else:
        raise AssertionError("select_shapes accepted the shape 'rows'")


def test_every_template_reads_for_every_shape():
    """Every backend template parses with every shape's sections and with each shape's alone
    (the bundle's CLIs, `serving/bundle/`, take the pack's single-brace fills and are read by
    test_serving_templates.py)."""
    import ast
    import re
    templates = SCRIPTS / "backends" / "pytorch" / "templates"
    problems = []
    for path in sorted(templates.rglob("*.py")):
        if path.relative_to(templates).parts[:2] == ("serving", "bundle"):
            continue
        text = re.sub(r"\{\{[A-Z_]+\}\}", "x", path.read_text(encoding="utf-8"))
        for shapes in (None, *((s,) for s in emit.SAMPLE_SHAPES)):
            try:
                ast.parse(emit.select_shapes(text, shapes, path.name))
            except (SyntaxError, emit.EmitError) as exc:
                problems.append(f"{path.relative_to(templates)} ({shapes}): {exc}")
    assert not problems, "\n".join(problems)


_OTHER_SHAPES = {"row": ("window", "graph", "graph-level"), "window": ("row", "graph"),
                 "graph": ("row", "window"), "graph-level": ("row", "window")}


def _refusals(domain, spec, other):
    """The refusals of a spec of the shape `other` at the two places a spec is read."""
    out = []
    for read in (domain.NetSpec.from_dict, domain.SampleSpec.from_spec):
        try:
            read(rt.reference_spec(other))
        except domain.UnsupportedShapeError as exc:
            out.append(str(exc))
    return out


def test_a_package_carries_its_spec_shape_only(tmp_path):
    """Each reference shape's package: its shape and no other, in `CARRIED_SHAPES`, in the
    files and in the refusals — and the shape it carries still loads."""
    import ast
    for name, others in _OTHER_SHAPES.items():
        spec = rt.reference_spec(name)
        pipeline = rt.scoped_pipeline(rt.reference_pipeline(), spec)
        shape = emit.sample_shape(spec)
        out = tmp_path / f"carried-{name}"
        rec = emit.emit(spec, pipeline, out)
        root = Path(rec["root"])
        assert not [e for e in gate_errors(root, pipeline) if e.startswith("C4")], name
        texts = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
                 for p in root.rglob("*.py")}
        assert not [r for r, text in texts.items() if "# >>> shape" in text or "# <<< shape" in text]
        strategies = {"row": {"_RowSamples", "StreamedRows", "_stream", "_streamed"},
                      "window": {"_WindowSamples", "_windows", "_held_host"},
                      "graph": {"_NodeSamples", "_GraphSamples", "_graphs", "_graph_design"}}
        defined = {n.name for text in texts.values() for n in ast.walk(ast.parse(text))
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        for other, names in strategies.items():
            assert (names <= defined) if other == shape else not (names & defined), \
                (name, other, names & defined)
        absent = {"window": ("windows",), "graph": ("graphs", "graph_engines")}
        for other, modules in absent.items():
            if other == shape:
                continue
            assert not [r for r, text in texts.items() for m in modules
                        if f"from .{m} import" in text or f".{m} import" in text], (name, other)
        with importable(out):
            import importlib
            domain = importlib.import_module(f"{PACKAGE}.domain")
            assert {s.value for s in domain.CARRIED_SHAPES} == {shape}, name
            assert domain.SampleSpec.from_spec(domain.NetSpec.from_dict(spec)).shape.value == shape
            for other in others:
                refused = _refusals(domain, spec, other)
                assert len(refused) == 2, (name, other, refused)
                want = emit.sample_shape(rt.reference_spec(other))
                assert all(f"sample shape is {want!r}" in r and f"carries the {shape} sample "
                           "shape only" in r and "/vega-forge" in r for r in refused), refused


def test_c4_refuses_an_import_of_a_module_the_package_does_not_hold(tmp_path):
    """The seeded negative: the row package with the old dispatch's lazy imports put back (a
    window and a graph module it does not hold) is refused by C4, line by line; a lazy import
    of a module it does hold, and `from . import <a module it holds>`, are not."""
    pipeline = rt.reference_pipeline()
    rec = emit.emit(rt.reference_spec("row"), pipeline, tmp_path / "seeded")
    root = Path(rec["root"])
    assert not [e for e in gate_errors(root, pipeline) if e.startswith("C4")]
    data = root / "ingestion" / "data.py"
    data.write_text(data.read_text(encoding="utf-8") + (
        "\n\ndef _windows(spec, part, frames):\n"
        "    from .windows import WindowParts      # the old dispatch's lazy import\n"
        "    return WindowParts(frames).part(spec, part)\n"
        "\n\ndef _frames_again():\n"
        "    from .frames import PartFrames        # a module the package holds\n"
        "    from . import frames                  # likewise\n"
        "    return PartFrames, frames\n"), encoding="utf-8")
    engines = root / "training" / "engines.py"
    engines.write_text(engines.read_text(encoding="utf-8") + (
        "\n\ndef _graph(spec, data, device, hints):\n"
        "    from .graph_engines import graph_source\n"
        "    return graph_source(spec, data, device, hints)\n"), encoding="utf-8")
    errors = [e for e in gate_errors(root, pipeline) if e.startswith("C4")]
    assert len(errors) == 2, errors
    assert any("'ingestion.windows'" in e and "does not hold" in e for e in errors), errors
    assert any("'training.graph_engines'" in e and "does not hold" in e for e in errors), errors
    code, text = _gate(root, pipeline)
    assert code == 2 and "ingestion/data.py:" in text and "training/engines.py:" in text, text


# ================================================================ the runner

def _run(argv):
    verbose = "-v" in argv
    pick = argv[argv.index("-k") + 1] if "-k" in argv else None
    work = Path(argv[argv.index("--work") + 1]) if "--work" in argv else None
    if work:
        work.mkdir(parents=True, exist_ok=True)
    _SESSION["root"] = Path(tempfile.mkdtemp(prefix="emit-layout-session-", dir=work))
    tests = [(n, f) for n, f in globals().items()
             if n.startswith("test_") and callable(f) and (pick is None or pick in n)]
    failed = []
    for name, fn in tests:
        tmp = Path(tempfile.mkdtemp(prefix=f"{name[:40]}-", dir=work))
        try:
            fn(tmp) if fn.__code__.co_argcount else fn()
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            if verbose:
                import traceback
                traceback.print_exc()
        else:
            print(f"  ok    {name}")
            if not work:
                shutil.rmtree(tmp, ignore_errors=True)
    if not work:
        shutil.rmtree(_SESSION["root"], ignore_errors=True)
    if failed:
        print(f"\n{len(failed)} of {len(tests)} test(s) failed")
        return 1
    print(f"\nthe emit table holds: {len(tests)} test(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run(sys.argv[1:]))
