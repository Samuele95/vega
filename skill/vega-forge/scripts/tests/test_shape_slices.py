#!/usr/bin/env python3
"""
test_shape_slices.py — an emitted package carries its spec's sample shape's world model, and
nothing of the others (the T5 round 1 audit, finding 1: a row package's `domain.py` held ~850
lines of window and graph values and plans it refuses to run).

    python3 -m pytest -q test_shape_slices.py

For a row, a window, a node-level and a graph-level spec, the full emission (`render_full`, the
emitter itself):
  * `domain.py` defines its shape's values and plans, and none of another shape's;
  * every module of the package imports, in a fresh interpreter (a section cut in the wrong
    place is an ImportError there);
  * the emitter's own check (`refuse_unresolved`) refuses a module that reads a name it neither
    binds nor imports, or imports from `domain` a name the rendered domain does not define.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render_templates as rt  # noqa: E402

emit = rt.load_emit()
TEMPLATES = HERE.parent / "backends" / "pytorch" / "templates"

WINDOW_ONLY = {"WindowSpec", "WindowPlan", "WindowLayout", "SeriesRows", "SeriesBlock",
               "WindowedPart", "WindowContext", "UnscoredRow", "UnscoredReason"}
GRAPH_ONLY = {"GraphSpec", "GraphView", "GraphEdges", "QueryEdges", "GraphGroups",
              "InferenceGraph", "GraphReference", "GraphPart", "GraphLevel", "GraphBatching",
              "GraphExposure", "GraphProtocol"}
EVERY_SHAPE = {"NetSpec", "SampleSpec", "SamplePart", "TensorPart", "Checkpoint", "Metrics"}
EXPECT = {"row": (set(), WINDOW_ONLY | GRAPH_ONLY),
          "window": (WINDOW_ONLY, GRAPH_ONLY),
          "graph": (GRAPH_ONLY, WINDOW_ONLY),
          "graph-level": (GRAPH_ONLY, WINDOW_ONLY)}

IMPORT_ALL = """
import importlib, sys
from pathlib import Path
root, package = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root))
bad = []
for f in sorted((root / package).rglob("*.py")):
    parts = f.relative_to(root).with_suffix("").parts
    name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    try:
        importlib.import_module(name)
    except Exception as exc:
        bad.append(f"{name}: {type(exc).__name__}: {exc}")
print("\\n".join(bad))
sys.exit(1 if bad else 0)
"""


@pytest.mark.parametrize("shape", sorted(EXPECT))
def test_the_package_carries_its_shape_only(shape, tmp_path):
    rec = rt.render(TEMPLATES, tmp_path, rt.reference_pipeline(), rt.reference_spec(shape),
                    full=True)
    root = Path(rec["root"]) if isinstance(rec, dict) else tmp_path
    pkg = next(p for p in root.iterdir() if (p / "domain.py").is_file()) \
        if not (root / "domain.py").is_file() else root
    domain = emit.defined_names(ast.parse((pkg / "domain.py").read_text(encoding="utf-8")))
    held, absent = EXPECT[shape]
    assert held | EVERY_SHAPE <= domain, sorted((held | EVERY_SHAPE) - domain)
    assert not (absent & domain), sorted(absent & domain)
    assert not {"Precision", "Boundary", "TrafficClass", "ResidentPart"} & domain
    run = subprocess.run([sys.executable, "-c", IMPORT_ALL, str(pkg.parent), pkg.name],
                         capture_output=True, text=True, timeout=600,
                         env={"PATH": str(Path(sys.executable).parent), "CUDA_VISIBLE_DEVICES": ""})
    assert run.returncode == 0, run.stdout + run.stderr


def test_a_section_cut_in_the_wrong_place_is_refused_at_emission():
    domain = "class NetSpec:\n    pass\n"
    with pytest.raises(emit.EmitError, match="reads 'WindowSpec'"):
        emit.refuse_unresolved({"domain.py": domain,
                                "m/a.py": "def f():\n    return WindowSpec\n"})
    with pytest.raises(emit.EmitError, match="imports 'GraphPart' from domain"):
        emit.refuse_unresolved({"domain.py": domain,
                                "m/b.py": "from ..domain import NetSpec, GraphPart\n"})
    with pytest.raises(emit.EmitError, match="reads 'GraphReference'"):
        emit.refuse_unresolved({"domain.py": domain,
                                "m/c.py": "def f(r: GraphReference):\n    return r\n"})
    emit.refuse_unresolved({"domain.py": domain, "m/d.py": "from ..domain import NetSpec\n"})



LAZY = """
import sys
sys.path.insert(0, sys.argv[1])
import importlib
training = importlib.import_module(sys.argv[2] + ".training")
assert "torch" not in sys.modules, "importing a module imports no framework"
assert training.__all__ == ["Trainer"], training.__all__
assert training.Trainer.__module__ == sys.argv[2] + ".training.loop"
ingestion = importlib.import_module(sys.argv[2] + ".ingestion")
assert ingestion.__all__ == ["ShapedParts"]
try:
    training.Nope
except AttributeError as exc:
    assert "Nope" in str(exc)
else:
    raise SystemExit("an unknown name must raise AttributeError")
"""


def test_a_module_re_exports_its_adapter_by_name_on_first_use(tmp_path):
    """The T5 round 1 audit, finding 7: every module `__init__.py` was a docstring, so a
    consumer bound to the file layout (`<pkg>.training.loop`). Each module now re-exports the
    adapter of the port it owns (PEP 562, so no framework is imported until it is used)."""
    rt.render(TEMPLATES, tmp_path, rt.reference_pipeline(), rt.reference_spec("row"), full=True)
    pkg = next(p for p in tmp_path.iterdir() if (p / "domain.py").is_file())
    run = subprocess.run([sys.executable, "-c", LAZY, str(tmp_path), pkg.name],
                         capture_output=True, text=True, timeout=600,
                         env={"PATH": str(Path(sys.executable).parent), "CUDA_VISIBLE_DEVICES": ""})
    assert run.returncode == 0, run.stdout + run.stderr



RECORD_AND_BUILDER = """
import sys, json
sys.path.insert(0, sys.argv[1])
import importlib
pkg = sys.argv[2]
spec = json.loads(sys.argv[3])
domain = importlib.import_module(pkg + ".domain")
record = importlib.import_module(pkg + ".training.record")
builder = importlib.import_module(pkg + ".model.builder")
ns = domain.NetSpec.from_dict(spec)
for mode, ok in (("max", True), ("min", True), ("maximise", False)):
    bad = dict(spec, early_stopping=dict(spec.get("early_stopping") or {}, mode=mode))
    try:
        record.EarlyStopping.for_spec(domain.NetSpec.from_dict(bad), ["loss", "pr_auc"])
    except ValueError as exc:
        assert not ok and "mode" in str(exc), (mode, exc)
    else:
        assert ok, mode
assert builder._CompositeBlocks not in builder.NetworkBuilder.__mro__
"""


def test_the_selection_direction_is_checked_and_the_composites_are_held(tmp_path):
    """The T5 round 1 improve pass: an early-stopping `mode` other than max/min silently
    minimised (refused now), and `_CompositeBlocks` was a mixin calling down into its
    subclass (now a part the builder holds, handed the builder's `one` and `make`)."""
    spec = rt.reference_spec("row")
    rt.render(TEMPLATES, tmp_path, rt.reference_pipeline(), spec, full=True)
    pkg = next(p for p in tmp_path.iterdir() if (p / "domain.py").is_file())
    import json
    run = subprocess.run([sys.executable, "-c", RECORD_AND_BUILDER, str(tmp_path), pkg.name,
                          json.dumps(spec)], capture_output=True, text=True, timeout=600,
                         env={"PATH": str(Path(sys.executable).parent), "CUDA_VISIBLE_DEVICES": ""})
    assert run.returncode == 0, run.stdout + run.stderr



SURFACES = """
import sys, json
sys.path.insert(0, sys.argv[1])
import importlib
pkg, spec = sys.argv[2], json.loads(sys.argv[3])
ports = importlib.import_module(pkg + ".ports")
network = importlib.import_module(pkg + ".model.network")
extra = importlib.import_module(pkg + ".model.baselines_extra")
domain = importlib.import_module(pkg + ".domain")
net = network.build_model(domain.NetSpec.from_dict(spec))
import inspect
def conforms(obj, proto):
    # every method `proto` declares, on `obj`, with the declared parameters (the Protocols are
    # static interfaces, not @runtime_checkable: isinstance would check names only)
    for name, member in vars(proto).items():
        if name.startswith("_") or not callable(member):
            continue
        impl = getattr(obj, name, None)
        if not callable(impl):
            return False
        want = [p for p in inspect.signature(member).parameters if p != "self"]
        got = list(inspect.signature(impl).parameters)
        if got[:len(want)] != want and not any(
                p.kind is inspect.Parameter.VAR_POSITIONAL
                for p in inspect.signature(impl).parameters.values()):
            return False
    return True
assert conforms(net, ports.TrainedNetwork) and conforms(net, ports.ScoredModel)
import numpy as np
ns = domain.NetSpec.from_dict(spec)
width, rng = len(ns.feature_contract.columns), np.random.default_rng(0)
design = domain.DesignMatrix(x=rng.normal(size=(60, width)), y=np.arange(60) % 2,
                             row_ids=np.arange(60))
for baseline in extra.build(ns, ("trivial", "linear", "tree")):
    assert not conforms(baseline, ports.ScoredModel), type(baseline)    # a recipe
    fitted = baseline.fit(design)
    assert fitted.fitted and conforms(fitted, ports.ScoredModel), type(fitted)
"""

UNDECLARED = ("ids", "target_affine_pending", "kl", "objective_center", "row_score",
              "sample_view")


def test_what_the_trainer_and_the_evaluator_read_of_a_model_is_declared(tmp_path):
    """The T5 round 1 audit (finding 4): the Model port named four methods while the Trainer,
    its record and losses and the Evaluator read nine more members through `getattr` defaults
    (a skipped target affine, an unkeyed norm). ports.py now declares the two surfaces
    (TrainedNetwork, ScoredModel), the network satisfies both and every baseline the scoring
    one, and the adapters read the members directly (a `hasattr` that refuses a model by name
    is a check, not a default)."""
    spec = rt.reference_spec("row")
    rt.render(TEMPLATES, tmp_path, rt.reference_pipeline(), spec, full=True)
    pkg = next(p for p in tmp_path.iterdir() if (p / "domain.py").is_file())
    for rel in ("training/loop.py", "training/record.py", "training/losses.py",
                "evaluation/evaluate.py"):
        tree = ast.parse((pkg / rel).read_text(encoding="utf-8"))
        reads = [c.args[1].value for c in ast.walk(tree) if isinstance(c, ast.Call)
                 and getattr(c.func, "id", None) == "getattr" and len(c.args) > 1
                 and isinstance(c.args[1], ast.Constant) and c.args[1].value in UNDECLARED]
        assert not reads, (rel, reads)
    import json
    run = subprocess.run([sys.executable, "-c", SURFACES, str(tmp_path), pkg.name,
                          json.dumps(spec)], capture_output=True, text=True, timeout=600,
                         env={"PATH": str(Path(sys.executable).parent), "CUDA_VISIBLE_DEVICES": ""})
    assert run.returncode == 0, run.stdout + run.stderr


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))


SELECTION_AND_HOOKS = """
import sys, json
sys.path.insert(0, sys.argv[1])
import importlib
pkg = sys.argv[2]
record = importlib.import_module(pkg + ".training.record")
loop = importlib.import_module(pkg + ".training.loop")
engines = importlib.import_module(pkg + ".training.engines")
stop = record.EarlyStopping("pr_auc", "max", 3, 0.0)
assert stop.measured({"pr_auc": 0.5, "loss": 1.0}) == 0.5
try:
    stop.measured({"pr_auc": None, "loss": 1.0, "_unavailable": {"pr_auc": "one class in y"}})
except ValueError as exc:
    assert "valid.pr_auc" in str(exc) and "one class in y" in str(exc), exc
else:
    raise SystemExit("an unavailable monitored metric must not count as 'did not improve'")
assert stop.measured({"pr_auc": None}, nan=True) is None      # a NaN epoch: the loop stops
seen = []
class Given:
    def on_epoch(self, record):
        seen.append(record)
hooks = engines.CallbackHooks(Given())
hooks.on_epoch({"epoch": 1}); hooks.on_first_step({}); hooks.on_step({})
assert seen == [{"epoch": 1}], seen
try:
    hooks.on_epch({"epoch": 2})
except AttributeError:
    pass
else:
    raise SystemExit("a misspelled hook must fail, not resolve to a no-op")
import dataclasses, torch
assert dataclasses.fields(loop._Fit) and loop._Fit.__dataclass_params__.frozen
class Net(torch.nn.Module):
    ids = ("d1", "head")
    def __init__(self):
        super().__init__()
        self.stages = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)])
    def named_stages(self):
        return tuple(zip(self.ids, self.stages))
got = [(b, n) for b, n, _p in record.GradientProbe.by_block(Net())]
assert got == [("d1", "stages.0.weight"), ("d1", "stages.0.bias"), ("head", "stages.1.weight"),
               ("head", "stages.1.bias")], got
net = Net(); net.extra = torch.nn.Parameter(torch.zeros(1))
try:
    list(record.GradientProbe.by_block(net))
except ValueError as exc:
    assert "extra" in str(exc), exc
else:
    raise SystemExit("a parameter outside every stage must not be filed under a guessed block")
assert loop._Best.__dataclass_params__.frozen
"""


def test_an_unmeasured_monitor_is_refused_and_a_misnamed_hook_fails(tmp_path):
    """The T5 round 2 improve pass: an unavailable monitored metric counted as "did not
    improve" (patience expired with no checkpoint), `_Callbacks.__getattr__` turned any name
    into a no-op, and `GradientProbe.block_of` parsed a block back out of a dotted name. The fit's collaborators are frozen and the selection's keep is returned
    per epoch (architect, the training module's state)."""
    spec = rt.reference_spec("row")
    rt.render(TEMPLATES, tmp_path, rt.reference_pipeline(), spec, full=True)
    pkg = next(p for p in tmp_path.iterdir() if (p / "domain.py").is_file())
    run = subprocess.run([sys.executable, "-c", SELECTION_AND_HOOKS, str(tmp_path), pkg.name],
                         capture_output=True, text=True, timeout=600,
                         env={"PATH": str(Path(sys.executable).parent), "CUDA_VISIBLE_DEVICES": ""})
    assert run.returncode == 0, run.stdout + run.stderr
