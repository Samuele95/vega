#!/usr/bin/env python3
"""
test_model_seam.py — the serving seam is typed: the Predictor and the inspection package reach
the network through the Model port alone (code-contract §1, §2; the bundle-gate audit that found
them reading `__call__`, `training`, `parameters`, `buffers`, `spec_digest` and the stages past
the port the blueprint drew).

    python3 -m pytest test_model_seam.py -q        (or: python3 test_model_seam.py)

1. **The role table.** The Model port carries `head` (the raw head of an assembled batch, no
   gradient) and `anatomy` (the network as data: `domain.NetworkAnatomy`, or None for a model
   with no stages) beside its four scoring methods, for every sample shape and task kind; the
   Network implements both with the port's parameters, so `port_surface.py` lists them as
   port methods and a blueprint that omits one is refused by the emitter.
2. **The emitted package.** `ports.py`'s Model Protocol declares them; the Network is an
   instance, and so is a `PortOnly` facade over it (`port_fixtures`); its anatomy is the
   spec's — the stages aligned with the blocks, the digest, the mode read when asked — and a
   misaligned one is refused; every baseline is an instance whose anatomy is None.
3. **The code.** An AST read of every serving package template (`serving/package/*.py`, the
   inspection package included), where the network is named `model` (a parameter, `self.model`,
   `self._model`): no call of it, no attribute of it but a port method, no `getattr` /
   `hasattr` / `setattr` / `vars` / `dir` on it. The read is live: a seeded `model.training`,
   `model(batch)` and `getattr(model, "stages")` are each found.
4. **The answers.** A Predictor over the facade answers bit for bit as one over the network
   (the Predictor, inspector and zoo fixtures hand over the facade, so those suites run the
   serving and inspection paths with nothing but the port).
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
PACKAGE_TEMPLATES = TEMPLATES / "serving" / "package"
for _p in (str(HERE), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_templates as rt  # noqa: E402
from port_fixtures import model_port  # noqa: E402

SEAM = ("head", "anatomy")
#: what may be done to the network beside calling a port method: hand it on, test its type
REFLECTION = frozenset({"getattr", "hasattr", "setattr", "delattr", "vars", "dir"})
NETWORK_FIELDS = frozenset({"model", "_model"})


# --------------------------------------------------------------------------- 1. the role table

def test_the_model_port_carries_the_serving_seam_for_every_shape_and_task():
    emit = rt.load_emit()
    for shape in ("row", "window", "graph"):
        surface = emit.adapter_surface("Model", shape)
        for task in emit.TASK_KINDS:
            declared = [emit._declared(m).name for m in emit._role_lines("Model", shape, task)]
            assert set(SEAM) <= set(declared), (shape, task, declared)
        for name in SEAM:
            assert surface[name].port, (shape, name)
        assert surface["head"].params == (("batch", "positional", False),)
        assert surface["anatomy"].params == ()
    assert model_port()[-2:] == SEAM


def test_a_blueprint_that_omits_the_seam_is_refused():
    emit = rt.load_emit()
    spec, pipeline = rt.reference_spec("row"), rt.reference_pipeline()
    model = next(p for p in pipeline["ports"] if p["role"] == "Model")
    model["signature"] = [m for m in model["signature"] if "anatomy" not in m]
    with pytest.raises(emit.EmitError, match="omits anatomy"):
        emit._refuse_port_drift(emit.plan(spec, pipeline), "row")


# --------------------------------------------------------------------------- 2. the package

PACKAGE = """
import dataclasses, importlib, json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[4])
pkg, spec = sys.argv[2], json.loads(sys.argv[3])
from port_fixtures import PortOnly
ports = importlib.import_module(pkg + ".ports")
domain = importlib.import_module(pkg + ".domain")
network = importlib.import_module(pkg + ".model.network")
extra = importlib.import_module(pkg + ".model.baselines_extra")
bmap = json.loads((__import__("pathlib").Path(sys.argv[1]) / pkg / "blueprint-map.json")
                  .read_text())
name = next(e["port"] for e in bmap["entries"] if e.get("role") == "Model")
port = getattr(ports, name)
for method in ("head", "anatomy"):
    assert callable(getattr(port, method, None)), (name, method)
net = network.build_model(domain.NetSpec.from_dict(spec)).eval()
def conforms(obj, proto):
    # the port's methods, each callable on obj (the Protocols are static interfaces, not
    # @runtime_checkable: final rehearsal round 2)
    return all(callable(getattr(obj, n, None)) for n, v in vars(proto).items()
               if callable(v) and not n.startswith("_"))
assert conforms(net, port) and conforms(PortOnly(net), port)
a = net.anatomy()
assert isinstance(a, domain.NetworkAnatomy), type(a)
assert a.spec_digest == domain.NetSpec.from_dict(spec).digest and a.training is False
assert a.ids == tuple(b["id"] for b in spec["blocks"])
assert len(a.stages) == len(a.kinds) == len(a.shapes) == len(a.types) == len(spec["blocks"])
assert all(isinstance(k, domain.StageKind) for k in a.kinds)
assert a.readout is net.readout and a.sample == net.sample and a.centre is None
assert net.train().anatomy().training and not net.eval().anatomy().training
assert a == net.anatomy() and hash(a) == hash(net.anatomy()), "a view's identity is its network"
assert a.forward.__self__ is a, "the forward is the view's, which calls the network"
class _Misaligned:
    ids, kinds, types, shapes, stages = ("a",), (), (), (), ()
try:
    domain.NetworkAnatomy(_Misaligned())
except ValueError as exc:
    assert "aligned" in str(exc)
else:
    raise AssertionError("a misaligned anatomy was built")
for baseline in extra.build(domain.NetSpec.from_dict(spec), ("trivial", "linear", "tree")):
    assert not conforms(baseline, port) and not hasattr(baseline, "anatomy"), type(baseline)
print("ok")
"""


def test_the_emitted_port_declares_the_seam_and_its_implementations_satisfy_it(tmp_path):
    spec = rt.reference_spec("row")
    rt.render(TEMPLATES, tmp_path, rt.reference_pipeline(), spec, full=True)
    pkg = next(p for p in tmp_path.iterdir() if (p / "domain.py").is_file())
    tree = ast.parse((pkg / "ports.py").read_text(encoding="utf-8"))
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    assert {f.name for f in model.body if isinstance(f, ast.FunctionDef)} >= set(SEAM)
    run = subprocess.run([sys.executable, "-c", PACKAGE, str(tmp_path), pkg.name,
                          json.dumps(spec), str(HERE)], capture_output=True, text=True,
                         timeout=600, env={"PATH": str(Path(sys.executable).parent),
                                           "CUDA_VISIBLE_DEVICES": ""})
    assert run.returncode == 0 and run.stdout.strip() == "ok", run.stdout + run.stderr


# --------------------------------------------------------------------------- 3. the code

def _is_network(node) -> bool:
    """`model`, `self.model`, `self._model`: how the serving templates name the network."""
    if isinstance(node, ast.Name):
        return node.id == "model"
    return (isinstance(node, ast.Attribute) and node.attr in NETWORK_FIELDS
            and isinstance(node.value, ast.Name) and node.value.id == "self")


def reaches_past_the_port(tree, port=None) -> list:
    """[(line, what)] of every reach to the network that is not a call of a port method."""
    port = frozenset(port or model_port())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_network(node.func):
            found.append((node.lineno, "calls the network"))
        elif isinstance(node, ast.Attribute) and _is_network(node.value) \
                and node.attr not in port:
            found.append((node.lineno, f"reads .{node.attr}"))
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) in REFLECTION \
                and node.args and _is_network(node.args[0]):
            found.append((node.lineno, f"{node.func.id}() on the network"))
    return sorted(found)


SERVING_FILES = sorted(p.relative_to(PACKAGE_TEMPLATES).as_posix()
                       for p in PACKAGE_TEMPLATES.rglob("*.py"))


@pytest.mark.parametrize("rel", SERVING_FILES)
def test_the_serving_code_reaches_the_network_through_the_port_alone(rel):
    text = (PACKAGE_TEMPLATES / rel).read_text(encoding="utf-8")
    tree = ast.parse(re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: m.group(1).lower(), text))
    assert reaches_past_the_port(tree) == [], rel


def test_the_code_check_is_live():
    seeded = ast.parse("def f(model, batch):\n"
                       "    if model.training:\n"
                       "        return model(batch)\n"
                       "    stages = getattr(model, 'stages')\n"
                       "    return model.head(batch), model.anatomy().stages, stages\n")
    assert reaches_past_the_port(seeded) == [(2, "reads .training"), (3, "calls the network"),
                                             (4, "getattr() on the network")]
    assert SERVING_FILES and "predict.py" in SERVING_FILES and \
        "inspection/inspector.py" in SERVING_FILES


# --------------------------------------------------------------------------- 4. the answers

@pytest.mark.parametrize("which", ["binary", "anomaly-svdd", "window", "graph-knn"])
def test_a_predictor_over_the_port_answers_as_one_over_the_network(which):
    import predictor_fixtures as pf
    pkg = pf.package()
    case = (pf.window_case() if which == "window" else
            pf.graph_case(which) if which.startswith("graph") else pf.row_case(which))
    rows = case.rows.head(24).reset_index(drop=True)
    direct = pkg.predict.Predictor(case.model, case.shaped.module.transform, case.records(pkg),
                                   case.context)
    through = case.predictor(pkg)
    assert type(through.model).__name__ == "PortOnly"
    pd.testing.assert_frame_equal(direct.predict(rows), through.predict(rows))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
