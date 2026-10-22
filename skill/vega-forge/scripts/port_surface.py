#!/usr/bin/env python3
"""
port_surface.py — the interface each blueprint port is held to (code-contract §1), listed from
the backend's adapters.

    python3 port_surface.py [--shape row|window|graph] [--task <kind>] [--backend pytorch] [--json]

The emitter renders a pipeline blueprint's `ports[].signature` verbatim into `ports.py`, and
the code that implements each port is the backend's adapter of the port's role — a fixed class
(`ShapedParts`, `Network`, `ModelRegistry`, `Trainer`, `Evaluator`, `Predictor`, `Monitor`;
D1 §4.2) whose methods the blueprint does not write. So a port's methods are not free: for each
role, this prints

  * the role's **port methods** — the declarations a port of that role carries, exactly as the
    role table writes them (`ROLE_SIGNATURES`), each one a method the adapter implements with
    these parameters, and each one a method the chain's stages call through the port;
  * the adapter's **other public methods** — a port may also declare one of them, with the
    parameters shown;
  * the package's own interfaces `ports.py` declares beside the ports (`ModelBuilder`,
    `TrainedNetwork`, `TrainingData`, `Callbacks`, `ScoredModel`, `ArtifactExporter`), which
    are the backend's and not blueprint ports — the Trainer, Evaluator and ModelStore port
    methods are typed with them;
  * the module arrows the backend's code needs, role to role (the Trainer's and the
    Evaluator's code read the Model module's readout and metrics; a graph Trainer reads the
    DataSource module's graph views): §4's `depends_on` draws each, or C4 refuses the package.

`/blueprint --mode pipeline` runs this before it draws §3 and copies each port's methods from
it (the port's name is the blueprint's; where a line says `-> Any` the blueprint may name a
type `held_artifacts.py` lists, never a framework type). The emitter refuses, by port and
method, a port that omits a port method, declares a method the adapter does not have, or
changes a parameter's name, order, kind or default — a call written against such a port binds
its arguments to the wrong parameters (code-contract §1). The shape is the shaped data's sample
(`row`, `window`, `graph`; default `row`): the adapters carry that shape's code only, and a port
method only some shapes carry (a graph's `exposure`, a forecast window's `target_scale`) is
listed for those shapes only. The task is the partition's task kind, read through a sequence
task to its base (`binary`, `multiclass`, `regression`, `forecast`, ...): a method only some
kinds carry (`train_targets`, what a regression-family head's affine is fitted on) is listed
for those kinds only; without `--task`, every kind's. For one shape, a port names the sample
part by the class that shape's package holds (`TensorPart` for rows), the name §2 draws.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES = ("row", "window", "graph")
ROLES = ("DataSource", "FeatureTransformer", "Model", "ModelStore", "Trainer", "Evaluator",
         "Predictor", "Monitor")


def _emitter(backend: str):
    """The backend's emitter module, loaded by path."""
    path = HERE / "backends" / backend / "emit.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"port_surface_emit_{backend}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declaration(name: str, params: tuple, static: bool) -> str:
    """A method line built from the adapter's parameters (no annotations: the adapter's own
    names the ports module may not import); `self` first, as every port method takes it."""
    parts = ["self"]
    var = any(k == "var" and n.startswith("*") and not n.startswith("**") for n, k, _ in params)
    for pname, kind, default in params:
        if kind == "keyword" and not var and "*" not in parts:
            parts.append("*")
        parts.append(pname + ("=..." if default else ""))
    note = "   # a static method of the adapter" if static else ""
    return f"def {name}({', '.join(parts)}): ...{note}"


def surface(emit, shape: str, task: str | None = None) -> dict:
    """{role: {adapter, file, port: [declarations], beside: [declarations]}, "backend":
    {protocol: [declarations]}}."""
    out = {}
    for role in ROLES:
        methods = emit.adapter_surface(role, shape, task=task)
        port_lines = {emit._declared(m).name: m for m in emit._role_lines(role, shape, task)}
        rel, cls = emit.ADAPTER_TEMPLATES.get(role, (None, None))
        module_file = emit.ROLE_FILES[role][0]
        out[role] = {
            "adapter": cls or "the recipe's transform-only module (functions)",
            "file": f"<{emit.ROLE_FILES[role][2]}>/{module_file}",
            "status": emit.ROLE_FILES[role][3],
            "port": [port_lines[n] for n, m in methods.items() if m.port],
            "beside": [_declaration(n, m.params, m.static) for n, m in methods.items()
                       if not m.port]}
    ports = emit.ports_py("package", {}, {"entries": []}, [])
    tree = ast.parse(ports)
    backend = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in emit.BACKEND_PROTOCOLS:
            backend[node.name] = [ast.unparse(f).split("\n")[0] + " ..."
                                  for f in node.body if isinstance(f, ast.FunctionDef)]
    out["backend"] = backend
    out["arrows"] = {role: list(roles) for role, roles in emit.role_arrows(shape).items()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="pytorch")
    ap.add_argument("--shape", choices=SHAPES, default="row",
                    help="the sample shape of the shaped data (default: row)")
    ap.add_argument("--task", default=None,
                    help="the partition's task kind, effective (default: every kind's methods)")
    ap.add_argument("--json", action="store_true", help="print the surface as JSON")
    a = ap.parse_args(argv)
    try:
        emit = _emitter(a.backend)
    except FileNotFoundError as exc:
        print(f"ERROR: no emitter for backend {a.backend!r} at {exc}")
        return 2
    if a.task is not None and a.task not in emit.TASK_KINDS:
        print(f"ERROR: --task {a.task!r} is not one of {', '.join(emit.TASK_KINDS)} (a sequence "
              "task is read as its base kind)")
        return 2
    found = surface(emit, a.shape, a.task)
    if a.json:
        print(json.dumps(found, indent=1, ensure_ascii=False))
        return 0
    for role in ROLES:
        r = found[role]
        print(f"{role} — adapter {r['adapter']} in {r['file']} ({r['status']})")
        print("  port methods (declare each, as written):")
        for line in r["port"]:
            print(f"    {line}")
        if r["beside"]:
            print("  the adapter's other public methods (a port may declare one, with these "
                  "parameters):")
            for line in r["beside"]:
                print(f"    {line}")
    print("ports.py also declares the package's own interfaces, which are not blueprint ports "
          "(code-contract §2):")
    for name, lines in found["backend"].items():
        print(f"  {name}: " + "; ".join(l.removeprefix("def ").split("(")[0] for l in lines))
    print("the module arrows the package's code needs (§4: the module owning the first role "
          "depends_on the module owning each other):")
    for role, roles in found["arrows"].items():
        if roles:
            print(f"  {role} -> {', '.join(roles)}")
    kind = f"{a.task} " if a.task else ""
    print(f"OK: the port surface of a {kind}{a.shape} package ({a.backend}): each port of a role "
          "declares the role's port methods with these parameters; the emitter refuses any "
          "other drawing by port and method")
    return 0


if __name__ == "__main__":
    sys.exit(main())
