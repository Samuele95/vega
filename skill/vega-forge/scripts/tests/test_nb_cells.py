#!/usr/bin/env python3
"""
test_nb_cells.py — the notebooks `nb_emit.py` writes drive the v2.3 adapter classes
(BUILD-PLAN B20; D1 §12.1, §16; R-D16).

    python3 -m pytest test_nb_cells.py [-q]          (or: python3 test_nb_cells.py)

**Static.** Every stage's notebook is generated from records written here — a forge run of each
sample shape (row, window, graph node, graph level), a campaign, an evaluation, a row bundle and
a window bundle — over packages rendered from the v2.3 templates (`render_templates.py`), and
every generated code cell is parsed. From the emitted package a cell imports only v2.3 names:
each port's adapter *class* from the module its role names in `blueprint-map.json`
(`vega_adapters.ADAPTER_CLASSES`: `Network`, `ShapedParts`, `Trainer`, `ModelRegistry`), the
domain's values and support modules; never an adapter *module* (v2.2's `network`, `data`,
`loop`, `store`), never a v2.2 module function (`build_model`, `resolve_device`; `build_loss`
and `resolved_lrs` only from the Trainer module's support files `losses.py` and `optim.py`), never `fit(..., store=)`, never a metric called with keyword knobs. Each stage
makes the v2.3 calls its sections stand on: the forge notebook `Network.build`, `ShapedParts()`,
the Trainer module's `device_for` / `build_loss` / `resolved_lrs` / `build_optimizer` (the
port has no static helpers), printing every
batch key's shape, with the degree histogram for a graph sample and only then; the train replay
`Trainer(ModelRegistry()).fit(..., run=RunContext(...))` through the classes of the run's
own package; the eval notebook `TaskReadout.for_spec` and `metric(name, y, score, options,
task)`; the pack notebook `predict.load(HERE)` and `predict.check_examples(HERE)`, and on a
window bundle `predict_one(row, history=…)`. The imports follow the map: a blueprint that puts
the ModelStore in its own `registry` module is imported from there. What cannot be served is
refused (exit 2): a v2.2 map, an evaluation without `vega/predictions-layout@1`, a bundle whose
CLI has no `check_examples`, a window bundle without `sample.history_rows`.

**Executed** (nbconvert, a real kernel, CPU). The forge notebook on the orders fixture (its
numbers against the golden smoke B01 captured on the pre-refactor tree) and on a window, a
graph-node and a graph-level fixture, each ending in its assertion cell; the train notebook
replaying a fitted run that re-emitted its own package — under a cosine schedule whose curve
depends on `max_epochs`, so a replay that shortened the spec would not match — to
`VEGA-REPLAY: ok`; the eval notebook recomputing an R-C2 predictions file to
`VEGA-EVAL: sealed` for a binary logit (orders, trained), a forecast's H steps (window) and a
multiclass node classifier's K logits (graph); and the pack notebook, from inside a row and a
window bundle, to `VEGA-PACK: reproduced` through a stand-in CLI with D2 §9's `load` /
`check_examples` surface whose single-row answer on a window bundle refuses any history but
`MANIFEST.sample.history_rows` rows (the real CLI is S05's; a real bundle's notebook is the W4
integration check's).

Transitional: until B14's `ModelRegistry` lands in the store template, the rendered store
module is given a thin class over its `save`/`load` so the replay's ModelStore port can be
driven; once the template carries the class, the template's own is what runs.

The fits and the smoke numbers are computed in a child process (`--worker`), so this file's
process state (thread count, imported packages) never touches the numbers a kernel reproduces.
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
NB_EMIT = SCRIPTS / "nb_emit.py"
GOLDEN_SMOKE = HERE / "golden" / "orders-smoke.json"
for _p in (str(HERE), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_templates as rt  # noqa: E402
from vega_adapters import ADAPTER_CLASSES  # noqa: E402

#: the v2.2 module-level adapter functions the v2.3 classes replaced
V22_FUNCTIONS = frozenset({"build_model", "resolve_device", "build_loss", "resolved_lrs"})
# the two of them v2.3 keeps as the Trainer module's support functions, by their support file
TRAINER_SUPPORT = {"build_loss": "losses", "resolved_lrs": "optim", "device_for": "optim"}
#: the ports whose adapter module a cell may name only through its class
ADAPTER_ROLES = ("Model", "DataSource", "Trainer", "ModelStore", "Evaluator")
RUN_TS = "2026-09-30T08-00Z"


# =========================================================================== records

def _render(run_dir: Path, spec: dict, pipeline: dict) -> Path:
    """The v2.3 template set rendered as the package of `pipeline` into `run_dir`."""
    rec = rt.render(TEMPLATES, run_dir, pipeline, spec, splice=True)
    return Path(rec["root"])


def _write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def _pipeline(package: str, registry_module: bool = False) -> dict:
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = package
    if registry_module:
        for module in pipeline["modules"]:
            if module["name"] == "model":
                module["ports_owned"] = ["Model"]
        pipeline["modules"].append({"name": "registry", "paradigm": "OOP",
                                    "ports_owned": ["Registry"], "depends_on": ["domain"]})
    return pipeline


#: the decision families compile_spec writes (netspec-contract §1), with the spec's keys
DECISION_FAMILIES = {"architecture": ["family", "blocks"], "objective": ["loss", "metrics"],
                     "optimization": ["optimizer", "schedule", "training", "init"],
                     "regularization": ["regularization", "early_stopping"]}


def _traced(spec: dict, package: str) -> dict:
    """A reference spec as the compiler hands it on: every decision carries its source, the
    families name the decisions, and `outputs.package` names the package."""
    spec = copy.deepcopy(spec)
    spec["decision_families"] = copy.deepcopy(DECISION_FAMILIES)
    spec["outputs"] = {"package": package}
    for key in ("family", "loss", "metrics", "optimizer", "schedule", "training", "init",
                "early_stopping"):
        if isinstance(spec.get(key), dict):
            spec[key].setdefault("source", "vega:default")
    for node in (spec.get("blocks") or []) + (spec.get("regularization") or []):
        node.setdefault("source", "vega:default")
    return spec


def _forge_run(root: Path, spec: dict, package: str, registry_module=False) -> Path:
    """A forge run directory `<root>/vega-runs/<ts>/` with the package, the spec and a smoke
    record carrying what the emitter's smoke pass carries (`forward_ok`)."""
    run = root / "vega-runs" / RUN_TS
    spec = _traced(spec, package)
    _render(run, spec, _pipeline(package, registry_module))
    _write(run / "netspec.json", spec)
    _write(run / "smoke.json", {"forward_ok": True, "sample": dict(
        spec["data_contract"].get("sample") or {"shape": "row"})})
    return run


def _campaign_records(camp: Path, spec: dict, package: str, selected="R0",
                      own_package=False) -> None:
    """campaign.json, ladder.json, experiments.json and one run's history, as /vega-train
    writes them (the values are the notebook's to read, not to check, in the static half)."""
    _write(camp / "campaign.json", {
        "input": {"package": package, "model": "model.html", "netspec_digest": "sha256:0"},
        "backend": {"name": "pytorch", "version": "2"}, "device": "cpu", "seed": 7,
        "determinism": {"tolerance": 0.0}, "budget": {"spent_min": 1, "wall_clock_min": 2},
        "mode": "attended"})
    _write(camp / "ladder.json", {"runs": [{"id": selected, "kind": "baseline", "delta": [],
                                            "source": "spec", "hypothesis": "the spec"}]})
    _write(camp / "experiments.json", {
        "runs": [{"id": selected, "status": "ok", "epochs": 1, "best_epoch": 1,
                  "primary": 0.5}],
        "selection": {"run": selected, "rule": "max valid primary"},
        "ranking": {"order": [selected]}})
    run = camp / "runs" / selected
    _write(run / "netspec.resolved.json", spec)
    (run / "history.jsonl").write_text(json.dumps({"epoch": 1, "train": {"loss": 1.0},
                                                   "valid": {"loss": 1.0}}) + "\n")
    if own_package:
        shutil.copytree(camp / package, run / package)


def _write_predictions(path: Path, frame: pd.DataFrame, task_effective: str) -> None:
    """`predictions.test.parquet` with the R-C2 key-value metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({b"vega.row_id_kind": b"raw_position", b"vega.task_effective":
                 task_effective.encode(), b"vega.truth_shape": b"[]",
                 b"vega.score_shape": b"[]"})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)


def _eval_records(camp: Path, run_id: str, layout="vega/predictions-layout@1") -> None:
    frame = pd.DataFrame({"row_id": [3, 5, 8, 9], "y": [0, 1, 0, 1],
                          "score": [-1.2, 0.7, -0.3, 1.9], "prediction": [0, 1, 0, 1],
                          "score_trivial": [-0.5] * 4})
    eval_dir = camp / "eval" / run_id
    _write_predictions(eval_dir / "predictions.test.parquet", frame, "binary")
    doc = {"schema": "vega/metrics@1", "run": run_id, "task": "binary", "primary": "pr_auc",
           "mode": "max", "n_test": 4, "n_boot": 10, "boot_seed": 0, "ci_level": 0.95,
           "models": {}, "differences": {}, "verdict": {},
           "upstream": {"valid": {"pr_auc": {"value": 0.5, "_upstream": True}}}}
    if layout:
        doc["predictions_layout"] = layout
    _write(eval_dir / "metrics.json", doc)
    for name, body in (("baselines.json", {"baselines": []}), ("slices.json", {"slices": []}),
                       ("robustness.json", {"checks": []}), ("curves.json", {"curves": {}}),
                       ("threshold.json", {"applies": True, "value": 0.5,
                                           "scale": "probability", "per_label": None})):
        _write(eval_dir / name, body)
    _write(camp / "test_seal.json", {"run": run_id, "exposures": 1, "test": {}})


BUNDLE_CLI = '''"""The bundle's command line (a stand-in with the D2 §9 surface)."""


def load(bundle_dir=None):
    raise NotImplementedError


def check_examples(bundle_dir=None, predictor=None):
    raise NotImplementedError
'''


def _bundle(root: Path, package: str, sample: dict, cli: str = BUNDLE_CLI) -> Path:
    bundle = root / "bundle"
    (bundle / package / "features").mkdir(parents=True)
    (bundle / package / "__init__.py").write_text("")
    _write(bundle / package / "blueprint-map.json", {"entries": [
        {"file": "features/preprocess.py", "role": "FeatureTransformer"}]})
    (bundle / "predict.py").write_text(cli)
    _write(bundle / "MANIFEST.json", {
        "slug": "fx", "run": "R0", "package": package,
        "files": {f"{package}/features/preprocess.py": "sha256:0"},
        "roundtrip": {"status": "pass", "rows": 3, "max_abs_diff": 0.0},
        "examples": {"kind": "test", "source": "the test part"}, "sample": sample,
        "output": {"kind": "binary"}, "chain": {}, "usage": {"cli": "", "module": ""}})
    _write(bundle / "schema.json", {"input": [], "output": {}})
    (bundle / "examples").mkdir()
    (bundle / "examples" / "rows.csv").write_text("a,b\n1,2\n")
    pd.DataFrame({"__row__": [0]}).to_parquet(bundle / "examples" / "expected.parquet")
    return bundle


def _emit(stage: str, source: Path, out: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(NB_EMIT), "--stage", stage, "--from",
                           str(source), "--out", str(out), *map(str, extra)],
                          capture_output=True, text=True, timeout=300)


def _emitted(stage: str, source: Path, out: Path, *extra) -> dict:
    proc = _emit(stage, source, out, *extra)
    assert proc.returncode == 0, f"nb_emit --stage {stage} failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(out.read_text(encoding="utf-8"))


# =========================================================================== the cell reader

class Cells:
    """A notebook's code cells, parsed, and what they import from and call on the package."""

    def __init__(self, nb: dict, package: str, bmap: dict | None):
        self.sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.trees = [ast.parse(src) for src in self.sources]
        self.package = package
        files = {e.get("role"): e.get("file") for e in (bmap or {}).get("entries") or []
                 if e.get("role") and e.get("file")}
        self.adapter_modules = {f"{package}.{files[r][:-3].replace('/', '.')}": r
                                for r in ADAPTER_ROLES if r in files}

    def nodes(self, kind):
        return [n for tree in self.trees for n in ast.walk(tree) if isinstance(n, kind)]

    @property
    def imports(self) -> list:
        """(module, name) for every `from <package>... import name` and `import <package>...`."""
        out = []
        for node in self.nodes(ast.ImportFrom):
            if node.module and node.module.split(".")[0] == self.package:
                out += [(node.module, a.name) for a in node.names]
        for node in self.nodes(ast.Import):
            out += [(a.name, None) for a in node.names if a.name.split(".")[0] == self.package]
        return out

    @property
    def calls(self) -> list:
        return self.nodes(ast.Call)

    def called(self) -> set:
        return {_dotted(c.func) for c in self.calls} - {None}

    def names(self) -> set:
        found = {n.id for n in self.nodes(ast.Name)} | {n.attr for n in self.nodes(ast.Attribute)}
        return found | {n.name for n in self.nodes(ast.FunctionDef)}

    def cell_with(self, needle: str) -> str | None:
        return next((src for src in self.sources if needle in src), None)


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, ast.Call):
        base = _dotted(node.func)
        return f"{base}()" if base else None
    return None


def _assert_v23_only(cells: Cells, where: str) -> None:
    """The rules every generated notebook obeys: v2.3 names only."""
    classes = set(ADAPTER_CLASSES.values())
    for module, name in cells.imports:
        full = module if name is None else f"{module}.{name}"
        assert full not in cells.adapter_modules, (
            f"{where}: `{full}` imports an adapter module — v2.2's module-level adapters; a "
            f"v2.3 cell imports the {ADAPTER_CLASSES[cells.adapter_modules[full]]} class")
        if name is not None and module in cells.adapter_modules:
            want = ADAPTER_CLASSES[cells.adapter_modules[module]]
            assert name == want, (f"{where}: `from {module} import {name}` — the "
                                  f"{cells.adapter_modules[module]} module's adapter is {want}")
    # v2.3's Trainer module keeps the loss and the schedule in its support files (`losses.py`,
    # `optim.py`), the functions the loop itself calls: imported from there they are not v2.2's
    # adapter-module functions, and the port carries no static copy of them (no Trainer.<x>_for)
    support = {name for module, name in cells.imports
               if name in TRAINER_SUPPORT and module not in cells.adapter_modules
               and module.rsplit(".", 1)[-1] == TRAINER_SUPPORT[name]}
    stale = (V22_FUNCTIONS & cells.names()) - support
    assert not stale, f"{where}: v2.2 module functions named: {sorted(stale)}"
    statics = {n for n in cells.called() if n.startswith("Trainer.") and n.endswith("_for")}
    assert not statics, f"{where}: the Trainer port has no static helpers: {sorted(statics)}"
    for call in cells.calls:
        name = _dotted(call.func) or ""
        kws = {k.arg for k in call.keywords}
        if name.endswith(".fit"):
            assert "run" in kws and not kws & {"store", "budget", "checkpoint_dir"}, (
                f"{where}: {name}(...) keywords {sorted(k for k in kws if k)} — v2.3's fit "
                "takes run=RunContext(...)")
        if name == "metric" or name.endswith(".metric"):
            assert None not in kws and not call.keywords, (
                f"{where}: {name} called with keyword knobs; a metric takes "
                "(name, y, score, options, task)")
            assert len(call.args) == 5, f"{where}: {name} takes 5 positional arguments"
    for module, name in cells.imports:
        if name in classes:
            assert module in cells.adapter_modules, (
                f"{where}: {name} imported from {module}, which the map gives no port role")


def _map(run: Path, package: str) -> dict:
    return json.loads((run / package / "blueprint-map.json").read_text(encoding="utf-8"))


# =========================================================================== static: forge

@pytest.mark.parametrize("shape", ["row", "window", "graph", "graph-level"])
def test_forge_cells_drive_the_v23_classes(tmp_path, shape):
    package = f"nb_forge_{shape.replace('-', '_')}"
    spec = rt.reference_spec(shape)
    run = _forge_run(tmp_path, spec, package)
    nb = _emitted("forge", run, run / "engineer.ipynb")
    cells = Cells(nb, package, _map(run, package))
    _assert_v23_only(cells, f"engineer.ipynb ({shape})")
    called = cells.called()
    for need in ("Network.build", "ShapedParts", "device_for", "build_loss",
                 "resolved_lrs", "build_optimizer", "data.load_part", "data.make_loader",
                 "model.block_shapes", "metrics.names_for"):
        assert need in called, f"engineer.ipynb ({shape}) never calls {need}"
    imported = {name for _, name in cells.imports}
    assert {"Network", "ShapedParts", "Trainer", "SealedPartError"} <= imported
    batch_cell = cells.cell_with("batch.items()")
    assert batch_cell and "shape_of(value)" in batch_cell, \
        "the batch cell prints every batch key's shape"
    degree = cells.cell_with("torch.bincount")
    if shape.startswith("graph"):
        assert degree and "train.edge_index" in degree, \
            f"a {shape} spec's notebook draws the train view's degree histogram"
    else:
        assert degree is None, f"a {shape} spec's notebook has no degree histogram"
    seal = cells.cell_with("Part.TEST")
    assert seal and "except SealedPartError" in seal and "except Exception" not in seal, \
        "the seal check accepts only the DataSource's own refusal"


def test_imports_follow_the_blueprint_map(tmp_path):
    """A blueprint that gives the ModelStore its own module is imported from that module."""
    package = "nb_registry"
    spec = rt.reference_spec("row")
    run = _forge_run(tmp_path, spec, package, registry_module=True)
    _campaign_records(run, spec, package, own_package=True)
    nb = _emitted("train", run, run / "train.ipynb")
    cells = Cells(nb, package, _map(run, package))
    _assert_v23_only(cells, "train.ipynb (registry module)")
    assert (f"{package}.registry.store", "ModelRegistry") in cells.imports
    replay = cells.cell_with("REPLAY_EPOCHS =")
    assert '"registry.store", "ModelRegistry"' in replay, \
        "the replay re-imports the store class from the module the map names"


# =========================================================================== static: train

def test_train_replay_drives_the_v23_trainer(tmp_path):
    package = "nb_train"
    spec = rt.reference_spec("row")
    run = _forge_run(tmp_path, spec, package)
    _campaign_records(run, spec, package, selected="R1", own_package=True)
    nb = _emitted("train", run, run / "train.ipynb", "--replay-epochs", 2)
    cells = Cells(nb, package, _map(run, package))
    _assert_v23_only(cells, "train.ipynb")
    imported = {name for _, name in cells.imports}
    assert {"Network", "ShapedParts", "Trainer", "ModelRegistry", "RunContext"} <= imported
    assert "device_for" in cells.called()
    fits = [c for c in cells.calls if (_dotted(c.func) or "").endswith(".fit")]
    assert len(fits) == 1
    owner = fits[0].func.value                  # trainer_cls(store_cls())
    assert isinstance(owner, ast.Call) and _dotted(owner.func) == "trainer_cls"
    store = owner.args[0] if owner.args else None
    assert isinstance(store, ast.Call) and _dotted(store.func) == "store_cls", \
        "the replay's Trainer is given a ModelStore instance"
    run_kw = next(k.value for k in fits[0].keywords if k.arg == "run")
    assert isinstance(run_kw, ast.Call) and _dotted(run_kw.func) == "run_cls"
    assert not any(k.arg == "store" for k in run_kw.keywords), \
        "the store is the Trainer's collaborator, not a field of the run"
    assert {k.arg for k in run_kw.keywords} >= {"checkpoint_dir"}
    assert "network_cls.build" in cells.called()
    replay = cells.cell_with("REPLAY_EPOCHS = 2")
    for role, cls in (("Model", "Network"), ("DataSource", "ShapedParts"),
                      ("Trainer", "Trainer"), ("ModelStore", "ModelRegistry")):
        assert f'"{cls}")' in replay, f"the re-emitted run's {role} class is re-imported"
    # the spec is replayed as the run resolved it: the schedule reads max_epochs
    assert "max_epochs" not in "".join(
        line for line in replay.splitlines() if not line.lstrip().startswith("#")), \
        "the replay must not rewrite the resolved spec's max_epochs"
    assert "raise _Enough()" in replay


# =========================================================================== static: eval

def test_eval_cells_read_the_head_through_the_readout(tmp_path):
    package = "nb_eval"
    spec = rt.reference_spec("row")
    run = _forge_run(tmp_path, spec, package)
    _campaign_records(run, spec, package)
    _eval_records(run, "R0")
    nb = _emitted("eval", run, run / "evaluate.ipynb")
    cells = Cells(nb, package, _map(run, package))
    _assert_v23_only(cells, "evaluate.ipynb")
    called = cells.called()
    for need in ("TaskReadout.for_spec", "MetricOptions", "M.metric", "READOUT.score"):
        assert need in called, f"evaluate.ipynb never calls {need}"
    metric_calls = [c for c in cells.calls if _dotted(c.func) == "M.metric"]

    def recorded(arg):
        """OPTIONS (threshold.json's), or a baseline's own (OPTIONS_OF, built from its
        baselines.json decision_threshold), falling back to OPTIONS."""
        return _dotted(arg) == "OPTIONS" or (
            isinstance(arg, ast.Call) and _dotted(arg.func) == "OPTIONS_OF.get"
            and len(arg.args) == 2 and _dotted(arg.args[1]) == "OPTIONS")
    assert all(recorded(c.args[3]) and _dotted(c.args[4]) == "TASK"
               for c in metric_calls), "every metric reads the recorded options and the task"
    source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'BASELINES.get("baselines")' in source and '"decision_threshold"' in source, \
        "each baseline's own threshold is read from baselines.json"
    imported = {(m, n) for m, n in cells.imports}
    layout_map = _map(run, package)
    data_file = next(e["file"] for e in layout_map["entries"] if e.get("role") == "DataSource")
    door = f"{package}.{data_file[:-3].replace('/', '.')}"
    assert not any(m.startswith(f"{package}.ingestion") or m == door for m, _ in imported), \
        "the eval notebook imports nothing of the DataSource's module"
    assert not {n for _, n in imported} & set(ADAPTER_CLASSES.values()), \
        "the eval notebook builds no network and opens no part"
    assert repr((door, f"{package}.ingestion")) in cells.cell_with("DOORS ="), \
        "the seal check looks for the DataSource module the map names"


# =========================================================================== static: pack

@pytest.mark.parametrize("window", [False, True])
def test_pack_cells_call_the_bundles_own_comparison(tmp_path, window):
    package = "nb_pack"
    sample = ({"shape": "window", "length": 4, "stride": 1, "horizon": 1, "lag_depth": 2,
               "history_rows": 5} if window else {"shape": "row"})
    bundle = _bundle(tmp_path, package, sample)
    nb = _emitted("pack", bundle, bundle / "predict.ipynb")
    cells = Cells(nb, package, None)
    _assert_v23_only(cells, f"predict.ipynb ({sample['shape']})")
    called = cells.called()
    assert {"predict.load", "predict.check_examples"} <= called
    checks = [c for c in cells.calls if _dotted(c.func) == "predict.check_examples"]
    assert [_dotted(c.args[0]) for c in checks] == ["HERE"], "check_examples(HERE), once"
    assert cells.cell_with("reproduced = bool(check[\"passed\"])"), \
        "the equality verdict is check_examples' own"
    assert not cells.cell_with("expected.parquet\")"), \
        "no second copy of the example comparison in the notebook"
    asks = [c for c in cells.calls if _dotted(c.func) == "predictor.predict_one"]
    assert asks, "the notebook answers a single row"
    with_history = [c for c in asks if any(k.arg == "history" for k in c.keywords)]
    if window:
        assert with_history == asks, "a window bundle's single row is sent with its history"
        cell = cells.cell_with("HISTORY_ROWS =")
        assert 'MANIFEST["sample"]["history_rows"]' in cell
    else:
        assert not with_history, "a row bundle's single row needs no history"
    md = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert f"`{package}/features/preprocess.py`" in md and "orders_pipeline" not in md


# =========================================================================== static: refusals

def test_a_v22_package_is_refused(tmp_path):
    package = "nb_old"
    spec = rt.reference_spec("row")
    run = _forge_run(tmp_path, spec, package)
    _campaign_records(run, spec, package)
    _eval_records(run, "R0")
    bmap = _map(run, package)
    bmap.pop("generation", None)
    _write(run / package / "blueprint-map.json", bmap)
    for stage in ("forge", "train", "eval"):
        proc = _emit(stage, run, tmp_path / f"{stage}.ipynb")
        assert proc.returncode == 2, f"--stage {stage} on a v2.2 map: exit {proc.returncode}"
        assert "generation v2.2" in proc.stdout and "Re-emit" in proc.stdout, proc.stdout
        assert not (tmp_path / f"{stage}.ipynb").exists()


def test_a_map_without_a_port_role_is_refused(tmp_path):
    package = "nb_norole"
    spec = rt.reference_spec("row")
    run = _forge_run(tmp_path, spec, package)
    bmap = _map(run, package)
    for entry in bmap["entries"]:
        if entry.get("role") == "ModelStore":
            entry.pop("role")
    _write(run / package / "blueprint-map.json", bmap)
    proc = _emit("forge", run, tmp_path / "e.ipynb")
    assert proc.returncode == 2 and "ModelStore" in proc.stdout, proc.stdout


def test_an_evaluation_without_the_v23_layout_is_refused(tmp_path):
    package = "nb_oldeval"
    spec = rt.reference_spec("row")
    run = _forge_run(tmp_path, spec, package)
    _campaign_records(run, spec, package)
    _eval_records(run, "R0", layout=None)
    proc = _emit("eval", run, tmp_path / "e.ipynb")
    assert proc.returncode == 2 and "vega/predictions-layout@1" in proc.stdout, proc.stdout


def test_a_bundle_without_check_examples_is_refused(tmp_path):
    bundle = _bundle(tmp_path, "nb_oldpack", {"shape": "row"},
                     cli="def load(bundle_dir=None):\n    return None\n")
    proc = _emit("pack", bundle, tmp_path / "p.ipynb")
    assert proc.returncode == 2 and "check_examples" in proc.stdout, proc.stdout


def test_a_window_bundle_without_history_rows_is_refused(tmp_path):
    bundle = _bundle(tmp_path, "nb_nohist", {"shape": "window", "length": 4})
    proc = _emit("pack", bundle, tmp_path / "p.ipynb")
    assert proc.returncode == 2 and "history_rows" in proc.stdout, proc.stdout


# =========================================================================== executed

def _execute(nb_path: Path, cwd: Path) -> str:
    """nbconvert in place (the acceptance scripts' call); the text every cell printed."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""          # a replay on the device the records were made on
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                           "--inplace", "--ExecutePreprocessor.timeout=900", nb_path.name],
                          cwd=str(cwd), capture_output=True, text=True, env=env, timeout=1200)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    errors = [f"{o.get('ename')}: {o.get('evalue')}" for c in nb["cells"]
              for o in c.get("outputs") or [] if o.get("output_type") == "error"]
    assert proc.returncode == 0 and not errors, (
        f"{nb_path.name} did not execute:\n{errors}\n{proc.stderr[-3000:]}")
    return "\n".join("".join(o.get("text") or []) for c in nb["cells"]
                     for o in c.get("outputs") or [] if o.get("output_type") == "stream")


def _worker(mode: str, **args) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                           json.dumps(args)], capture_output=True, text=True, env=env,
                          timeout=900)
    assert proc.returncode == 0, f"worker {mode} failed:\n{proc.stdout}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _registry_ready(package_root: Path) -> None:
    """Transitional (until B14's ModelRegistry is in the store template): give the rendered
    store module a thin ModelRegistry over its own `save`/`load`, so the replay's ModelStore
    port has the class the contract names. A template that has the class is left alone."""
    bmap = json.loads((package_root / "blueprint-map.json").read_text(encoding="utf-8"))
    store = package_root / next(e["file"] for e in bmap["entries"]
                                if e.get("role") == "ModelStore")
    text = store.read_text(encoding="utf-8")
    if "class ModelRegistry" in text:
        return
    store.write_text(text + '''

class ModelRegistry:
    """The ModelStore port as a class, over this module's save/load (test stand-in)."""

    def save(self, model, path, spec=None, epoch=None):
        return save(model, path, spec=spec, epoch=epoch)

    def load(self, ref, spec=None, build=None):
        return load(ref, spec=spec, build=build)
''', encoding="utf-8")


@pytest.fixture(scope="module")
def orders(tmp_path_factory):
    """The orders fixture (make_fixture.py) rendered as a v2.3 forge run."""
    work = tmp_path_factory.mktemp("nb-orders")
    shutil.copytree(HERE / "fixtures" / "dot-claude", work / ".claude")
    shutil.copy2(HERE / "architecture-layer.authoring.json", work)
    r = subprocess.run([sys.executable, str(HERE / "make_fixture.py"), str(work)],
                       capture_output=True, text=True, cwd=str(work), timeout=600)
    assert r.returncode == 0, f"make_fixture.py failed:\n{r.stdout}\n{r.stderr}"
    spec = json.loads((work / "netspec.json").read_text(encoding="utf-8"))
    pipeline = json.loads(Path(spec["input"]["pipeline"]).read_text(encoding="utf-8"))
    run = work / "vega-runs" / RUN_TS
    root = _render(run, spec, pipeline)
    _registry_ready(root)
    shutil.copy2(work / "netspec.json", run / "netspec.json")
    shutil.copy2(work / "architecture-layer.authoring.json", run / "architecture-layer.json")
    return work, run, spec, root.name


def test_the_forge_notebook_executes_on_orders(orders):
    """The v2.3 notebook reproduces the numbers the golden smoke captured on the pre-refactor
    tree: parameters per block, output shapes, the loss formula."""
    work, run, spec, package = orders
    smoke = json.loads(GOLDEN_SMOKE.read_text(encoding="utf-8"))
    smoke.pop("_capture", None)
    smoke["sample"] = {"shape": "row"}
    _write(run / "smoke.json", smoke)
    _emitted("forge", run, run / "engineer.ipynb")
    text = _execute(run / "engineer.ipynb", run)
    assert "18 checks: the notebook's numbers are the layer's numbers" in text, text[-2000:]
    assert "seal holds: SealedPartError" in text


def _window_fixture(root: Path) -> dict:
    """Three series of 40 steps split by time; the reference forecast GRU spec on them."""
    import data_fixtures as fx
    rows = [{"series": f"s{s}", "ts": t, "x1": float(s) + 0.1 * t}
            for t in range(40) for s in range(3)]
    frame = pd.DataFrame(rows)
    frame.insert(2, "x0", np.sin(np.arange(len(frame)) / 5.0))
    frame["target"] = frame["x0"] * 2.0 + frame["x1"]
    pos = np.arange(len(frame))
    parts = {"train": pos[frame.ts < 28], "valid": pos[(frame.ts >= 28) & (frame.ts < 34)],
             "test": pos[frame.ts >= 34]}
    man = fx.write_shaped(root / "shaped", {k: frame.iloc[v] for k, v in parts.items()}, parts)
    spec = rt.reference_spec("window")
    spec["data_contract"]["sample"].update({"length": 6, "horizon": 3})
    spec["training"]["batch_size"] = 16
    return fx.spec_on(spec, man)


def _graph_fixture(root: Path, level: str) -> dict:
    """60 nodes with a symmetric edge part (node level), or 12 small graphs (graph level)."""
    import data_fixtures as fx
    rng = np.random.default_rng(5)
    if level == "node":
        n = 60
        frame = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n),
                              "x2": rng.normal(size=n),
                              "community": [("a", "b", "c", "d")[i % 4] for i in range(n)]})
        perm = rng.permutation(n)
        parts = {"train": np.sort(perm[:36]), "valid": np.sort(perm[36:48]),
                 "test": np.sort(perm[48:])}
        pairs = {(int(min(a, b)), int(max(a, b)))
                 for a, b in rng.integers(0, n, size=(160, 2)) if a != b}
    else:
        rows = []
        for g in range(12):
            for _ in range(3 + g % 3):
                rows.append({"molecule": f"m{g:02d}", "x0": rng.normal(), "x1": rng.normal(),
                             "x2": rng.normal(), "y": g % 2})
        frame = pd.DataFrame(rows)
        n = len(frame)
        part_of_graph = {f"m{g:02d}": ("train" if g < 8 else "valid" if g < 10 else "test")
                         for g in range(12)}
        pos = np.arange(n)
        parts = {p: pos[frame.molecule.map(part_of_graph).to_numpy() == p]
                 for p in ("train", "valid", "test")}
        pairs = set()
        for g in range(12):
            members = np.flatnonzero(frame.molecule.to_numpy() == f"m{g:02d}")
            pairs |= {(int(a), int(b)) for a, b in zip(members, members[1:])}
    part_of = {int(p): k for k, v in parts.items() for p in v}
    edges = pd.DataFrame([{"src_pos": s, "dst_pos": d, "src_part": part_of[s],
                           "dst_part": part_of[d], "weight": None}
                          for a, b in sorted(pairs) for s, d in ((a, b), (b, a))])
    man = fx.write_shaped(root / "shaped", {k: frame.iloc[v] for k, v in parts.items()},
                          parts, edges)
    spec = rt.reference_spec("graph" if level == "node" else "graph-level")
    for block in spec["blocks"]:
        # the builder reads a graph convolution's width as `out` (render_templates' reference
        # specs still say `hidden`: a B14 follow-up recorded in the W1 summary)
        if block["type"].endswith("_conv") and "hidden" in block["params"]:
            block["params"]["out"] = block["params"].pop("hidden")
    if level == "graph":
        spec["training"]["batch_size"] = 3
    return fx.spec_on(spec, man)


@pytest.mark.parametrize("shape", ["window", "graph", "graph-level"])
def test_the_forge_notebook_executes_on_every_shape(tmp_path, shape):
    work = tmp_path / "project"
    spec = (_window_fixture(work) if shape == "window" else
            _graph_fixture(work, "node" if shape == "graph" else "graph"))
    package = f"nb_x_{shape.replace('-', '_')}"
    spec = _traced(spec, package)
    run = work / "vega-runs" / RUN_TS
    _registry_ready(_render(run, spec, _pipeline(package)))
    _write(run / "netspec.json", spec)
    smoke = _worker("smoke", run=str(run), package=package)
    assert smoke["forward_ok"]
    _emitted("forge", run, run / "engineer.ipynb")
    text = _execute(run / "engineer.ipynb", run)
    assert "checks: the notebook's numbers are the layer's numbers" in text, text[-2000:]
    assert "seal holds: SealedPartError" in text
    keys = {"window": ("x_num", "y"), "graph": ("edge_index", "target_index"),
            "graph-level": ("edge_index", "graph_index", "num_graphs")}[shape]
    for key in keys:
        assert f"  {key} " in text, f"the batch cell prints {key}"
    if shape != "window":
        assert "in-degree  min" in text and (run / "degree.png").is_file()


@pytest.fixture(scope="module")
def campaign(orders):
    """The orders run as a campaign: R0 on the root package, R1 (selected) on a package it
    re-emitted, both under a warm-up-free cosine over 4 epochs — so each epoch's rate depends on
    `max_epochs` — with the histories the v2.3 Trainer recorded."""
    work, run, spec, package = orders
    camp = run
    records = _worker("campaign", camp=str(camp), package=package)
    return camp, package, records


def test_the_train_notebook_replays_the_selected_run(campaign):
    camp, package, records = campaign
    _emitted("train", camp, camp / "train.ipynb", "--replay-epochs", 2)
    text = _execute(camp / "train.ipynb", camp)
    assert "replaying against the package this run re-emitted" in text
    assert "VEGA-REPLAY: ok R1 epochs=2" in text, text[-3000:]
    assert "checks: the notebook's numbers are the ledger's numbers" in text
    assert list((camp / "replay" / "R1").glob("*.pt")), \
        "the replay's checkpoints went through the ModelStore port into its scratch"


#: a bundle CLI with the D2 §9 surface the pack notebook calls (S05 writes the real one): a
#: predictor answering one row per row, whose single-row answer on a window bundle refuses a
#: history shorter than MANIFEST.sample.history_rows, and the one example comparison
CONTRACT_CLI = '''"""Stand-in bundle CLI: D2 §9's load / check_examples surface, nothing more."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


class _Predictor:
    def __init__(self, manifest):
        self.sample = manifest.get("sample") or {"shape": "row"}

    def predict(self, df_raw, history=None):
        before = 0 if history is None else len(history)
        out = pd.DataFrame({"score": [0.25] * len(df_raw), "warnings": [""] * len(df_raw)})
        if self.sample["shape"] == "window":
            need = int(self.sample["length"]) - 1
            out["answered"] = [before + i >= need for i in range(len(df_raw))]
            out.loc[~out["answered"], "warnings"] = "window incomplete"
        return out

    def predict_one(self, row, history=None):
        if self.sample["shape"] == "window":
            rows = int(self.sample["history_rows"])
            if history is None or len(history) != rows:
                raise ValueError(f"a window row is sent with {rows} rows of history")
        return self.predict(pd.DataFrame([row]), history).iloc[0].to_dict()


def load(bundle_dir=None):
    root = Path(bundle_dir or ROOT)
    return _Predictor(json.loads((root / "MANIFEST.json").read_text()))


def check_examples(bundle_dir=None, predictor=None):
    return {"rows": 3, "max_abs_diff": 0.0, "tolerance": 1e-05, "decisions_match": True,
            "labels_match": True, "passed": True,
            "evidence": "authenticated against MANIFEST.files"}
'''


def _contract_bundle(root: Path, window: bool) -> Path:
    """A bundle whose records, manifest and digests are complete enough for every section of
    predict.ipynb, around the stand-in CLI."""
    import hashlib
    package = "nb_bundle"
    sample = ({"shape": "window", "length": 4, "stride": 1, "horizon": 1, "lag_depth": 2,
               "history_rows": 5} if window else {"shape": "row"})
    bundle = _bundle(root, package, sample, cli=CONTRACT_CLI)
    n = 9 if window else 4
    pd.DataFrame({"a": np.arange(n, dtype=float), "zone": ["x", "y"] * (n // 2) + ["x"] * (n % 2)}
                 ).to_csv(bundle / "examples" / "rows.csv", index=False)
    _write(bundle / "schema.json", {"input": [{"name": "a", "dtype": "float",
                                               "range_seen": [0, 9]},
                                              {"name": "zone", "dtype": "string",
                                               "allowed": ["x", "y"]}],
                                    "output": {"score": "the logit"}, "ignored": []})
    evaluation = {"seal": {"run": "R0", "exposures": 1, "test": {}},
                  "verdict": {"earned": True, "strongest_baseline": "trivial"},
                  "metrics": {"primary": "pr_auc",
                              "models": {"network": {"pr_auc": {"value": 0.7}}}}}
    _write(bundle / "records" / "evaluation.json", evaluation)
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    files = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[path.relative_to(bundle).as_posix()] = \
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.update({
        "campaign": "vega-runs/x", "created_at": "2026-09-30T08:00:00Z", "python": "3",
        "backend": {"name": "pytorch", "version": "2"}, "files": files,
        "roundtrip": {"status": "pass", "rows": 3, "max_abs_diff": 0.0,
                      "shaper_transform_check": "pass"},
        "blueprint": {"conformance": {"mechanical": "pass"}},
        "chain": {"evaluation": {"verdict": {"earned": True, "strongest_baseline": "trivial"},
                                 "primary": {"name": "pr_auc", "value": 0.7},
                                 "seal_exposures": 1}},
        "usage": {"cli": "python3 predict.py --input rows.csv",
                  "module": "from predict import load"},
        "serve": {"present": False, "mode": "batch"}, "exports": [], "stances": [],
        "include": "serving", "entrypoints": ["predict.py"]})
    _write(bundle / "MANIFEST.json", manifest)
    return bundle


@pytest.mark.parametrize("window", [False, True])
def test_the_pack_notebook_runs_on_the_bundle_contract(tmp_path, window):
    """predict.ipynb executes from inside a bundle through the D2 §9 surface — the stand-in
    CLI refuses a window row sent without exactly `history_rows` rows of history."""
    bundle = _contract_bundle(tmp_path, window)
    _emitted("pack", bundle, bundle / "predict.ipynb")
    text = _execute(bundle / "predict.ipynb", bundle)
    assert "VEGA-PACK: reproduced" in text, text[-3000:]
    assert "checks: this bundle is the one the manifest and the card describe" in text
    assert "3 compared row(s)" in text and "evidence: authenticated against MANIFEST.files" in text
    if window:
        assert "history: the 5 row(s) before row 5 (MANIFEST.sample.history_rows = 5)" in text
        assert "row(s) answered" in text


@pytest.mark.parametrize("shape", ["window", "graph"])
def test_the_eval_notebook_reads_every_head(tmp_path, shape):
    """The recomputation over a multi-column head and truth: a forecast's H steps (the window
    fixture) and a multiclass node classifier's K logits (the graph fixture)."""
    work = tmp_path / "project"
    spec = _window_fixture(work) if shape == "window" else _graph_fixture(work, "node")
    package = f"nb_e_{shape}"
    spec = _traced(spec, package)
    camp = work / "vega-runs" / RUN_TS
    _registry_ready(_render(camp, spec, _pipeline(package)))
    _write(camp / "netspec.json", spec)
    _write(camp / "runs" / "R0" / "netspec.resolved.json", spec)
    _write(camp / "campaign.json", {"input": {"package": package}})
    done = _worker("evaluation", camp=str(camp), package=package, run="R0")
    assert done["kind"] == ("forecast" if shape == "window" else "multiclass")
    _emitted("eval", camp, camp / "evaluate.ipynb")
    text = _execute(camp / "evaluate.ipynb", camp)
    assert "VEGA-EVAL: sealed" in text, text[-3000:]
    assert f"read as {done['kind']}" in text and "0 disagreement(s)" in text


def test_the_eval_notebook_recomputes_the_stored_predictions(campaign):
    camp, package, records = campaign
    _worker("evaluation", camp=str(camp), package=package, run="R1")
    _emitted("eval", camp, camp / "evaluate.ipynb")
    text = _execute(camp / "evaluate.ipynb", camp)
    assert "VEGA-EVAL: sealed" in text, text[-3000:]
    assert "checks: the notebook's numbers are the evaluation's numbers" in text


# =========================================================================== the worker
#
# Run as `test_nb_cells.py --worker <mode> <json>` in a child process: the numbers a kernel
# will reproduce are made in a process whose state this test file's imports cannot touch.

def _import(package_dir: Path, package: str):
    from vega_adapters import PackageAdapters
    return PackageAdapters.load(package_dir, package)


def _first_batch(adapters, spec):
    from importlib import import_module
    part = import_module(adapters.package.__name__ + ".domain").Part
    train = adapters.data.load_part(spec, part.TRAIN)
    return next(iter(adapters.data.make_loader(train, spec, shuffle=False)))


def worker_smoke(run: str, package: str) -> dict:
    """smoke.json as the emitter's smoke pass writes it (D1 §16): the counts, the shapes, the
    loss formula, the schedule and the sample."""
    import torch
    run = Path(run)
    adapters = _import(run / package, package)
    raw = json.loads((run / "netspec.json").read_text(encoding="utf-8"))
    spec = adapters.package.domain.NetSpec.from_dict(raw)
    model = adapters.build(spec)
    batch = _first_batch(adapters, spec)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    total, trainable = model.param_count()
    smoke = {"params_total": total, "params_trainable": trainable,
             "per_block": model.block_shapes(batch), "forward_ok": True,
             "out_shape": [int(s) for s in out.shape[1:]],
             "loss_formula": adapters.loss_for(spec).formula(),
             "schedule_lrs": adapters.lrs_for(spec, 1),
             "device": "cpu", "sample": {"shape": raw["data_contract"]["sample"]["shape"]}}
    _write(run / "smoke.json", smoke)
    return smoke


class _Recorder:
    def __init__(self, path: Path):
        self.path, self.records = path, []

    def on_epoch(self, record):
        row = record.as_dict()
        self.records.append(row)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def _fit_run(camp: Path, package: str, run_id: str, spec_raw: dict, own: bool) -> dict:
    run_dir = camp / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if own:
        shutil.copytree(camp / package, run_dir / package, dirs_exist_ok=True)
    _write(run_dir / "netspec.resolved.json", spec_raw)
    adapters = _import((run_dir if own else camp) / package, package)
    spec = adapters.package.domain.NetSpec.from_dict(spec_raw)
    recorder = _Recorder(run_dir / "history.jsonl")
    if recorder.path.exists():
        recorder.path.unlink()
    model = adapters.build(spec)
    run_cls = adapters.package.domain.RunContext
    adapters.trainer.fit(model, spec, adapters.data, callbacks=recorder,
                         run=run_cls(checkpoint_dir=str(run_dir)))
    primary = spec_raw["metrics"]["primary"]
    values = [(r.get("valid") or {}).get(primary) for r in recorder.records]
    best = 1 + max(range(len(values)), key=lambda i: -math.inf if values[i] is None else values[i])
    return {"id": run_id, "status": "ok", "epochs": len(recorder.records), "best_epoch": best,
            "primary": values[best - 1], "kind": "baseline" if run_id == "R0" else "ablation",
            "stopped_by": "max_epochs", "diagnostics": [], "eligible": True,
            "params": model.param_count()[0]}


def worker_campaign(camp: str, package: str) -> dict:
    camp = Path(camp)
    base = json.loads((camp / "netspec.json").read_text(encoding="utf-8"))
    base["training"] = dict(base["training"], max_epochs=4, device="cpu")
    base["schedule"] = {"name": "cosine", "params": {}, "source": "vega:default"}
    other = copy.deepcopy(base)
    other["optimizer"]["params"]["lr"] = 0.002
    rows = [_fit_run(camp, package, "R0", base, own=False),
            _fit_run(camp, package, "R1", other, own=True)]
    _write(camp / "campaign.json", {
        "input": {"package": package, "model": "orders.domain.analysis.shaped.vega.html",
                  "netspec_digest": "sha256:0"},
        "backend": {"name": "pytorch", "version": "2"}, "device": "cpu", "seed": base["seed"],
        "determinism": {"tolerance": 0.0}, "budget": {"spent_min": 1, "wall_clock_min": 2},
        "mode": "attended"})
    _write(camp / "ladder.json", {"runs": [
        {"id": "R0", "kind": "baseline", "delta": [], "source": "spec",
         "hypothesis": "the spec as engineered"},
        {"id": "R1", "kind": "ablation", "family": "optimization",
         "delta": [{"path": "optimizer.params.lr", "to": 0.002}], "source": "vega:ladder",
         "hypothesis": "a larger step converges sooner"}]})
    _write(camp / "experiments.json", {
        "runs": rows, "selection": {"run": "R1", "rule": "max valid pr_auc"},
        "ranking": {"by": "valid.pr_auc", "mode": "max", "order": ["R1", "R0"]}})
    return {"runs": rows}


def _ci(samples, alpha=0.05):
    clean = sorted(float(s) for s in samples if s is not None and np.isfinite(s))
    if len(clean) < 2:
        return None
    lo = clean[max(0, int((alpha / 2) * len(clean)) - 1)]
    hi = clean[min(len(clean) - 1, int((1 - alpha / 2) * len(clean)))]
    return [round(lo, 4), round(hi, 4)]


def _held_head(adapters, spec, model, part):
    """(raw head, truth, row ids) of a held part, batch by batch in stored order — the raw head
    is the network's forward output (every head_out column), as R-C2 stores it."""
    import torch
    heads, truths = [], []
    model.eval()
    with torch.no_grad():
        for batch in adapters.data.make_loader(part, spec, shuffle=False):
            heads.append(model(batch).double().numpy())
            truths.append(np.asarray(batch["y"]))
    return np.concatenate(heads), np.concatenate(truths), np.asarray(part.row_ids)


def _trivial(kind: str, train_y, width: int) -> np.ndarray:
    """The trivial baseline's score row, in the score contract: a prior logit (binary), log
    class priors (multiclass), the train target's mean in every column (values: D1 §11.3's
    "mean of train target" — a window part holds the target per row, not per window)."""
    y = np.asarray(train_y, dtype=np.float64)
    if kind == "binary":
        p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return np.array([math.log(p / (1 - p))])
    if kind == "multiclass":
        counts = np.bincount(y.astype(int), minlength=width) + 1.0
        return np.log(counts / counts.sum())
    return np.full(width, float(y.mean()))


def _columns(frame: dict, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.ndim == 1:
        frame[name] = values
    else:
        for k in range(values.shape[1]):
            frame[f"{name}_{k}"] = values[:, k]


def worker_evaluation(camp: str, package: str, run: str) -> dict:
    """An evaluation of run `run` as /vega-eval writes it under R-C2 — the valid part stands in
    for the sealed test part (this test opens no seal). The run's checkpoint when it has one
    (`best.pt`), else the network as the spec builds it: what is checked is the notebook's
    reading of the record, not how good the network is."""
    import torch
    camp = Path(camp)
    run_dir = camp / "runs" / run
    own = run_dir / package
    adapters = _import(own if own.is_dir() else camp / package, package)
    dom, metrics = adapters.package.domain, adapters.metrics
    raw = json.loads((run_dir / "netspec.resolved.json").read_text(encoding="utf-8"))
    spec = dom.NetSpec.from_dict(raw)
    readout = adapters.readout.for_spec(spec)
    kind = readout.kind.value
    checkpoint = run_dir / "best.pt"
    model = (adapters.store.load(str(checkpoint), spec=spec, build=adapters.build)
             if checkpoint.is_file() else adapters.build(spec))
    head, y, row_ids = _held_head(adapters, spec, model,
                                  adapters.data.load_part(spec, dom.Part.VALID))
    network = readout.score(torch.tensor(head)).numpy()
    train_y = adapters.data.load_part(spec, dom.Part.TRAIN).y
    trivial = np.tile(_trivial(kind, train_y, head.shape[1]), (len(y), 1))
    trivial = trivial[:, 0] if network.ndim == 1 else trivial
    threshold = 0.4 if kind == "binary" else None
    options = dom.MetricOptions(threshold=0.5 if threshold is None else threshold)
    frame = {"row_id": row_ids}
    _columns(frame, "y", y)
    _columns(frame, "score", head if head.shape[1] > 1 else head[:, 0])
    if threshold is not None:
        frame["prediction"] = (metrics.sigmoid(network) >= threshold).astype(int)
    _columns(frame, "score_trivial", trivial)
    eval_dir = camp / "eval" / run
    _write_predictions(eval_dir / "predictions.test.parquet", pd.DataFrame(frame), kind)

    names = metrics.names_for(spec)
    primary = names[0]
    direction = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
    mode = direction["metric_direction"][primary]
    scores = {"network": network, "trivial": trivial}
    n_boot, seed = 60, 11
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]

    def value(name, model_name, idx=None):
        sel = slice(None) if idx is None else idx
        try:
            v = metrics.metric(name, y[sel], scores[model_name][sel], options, readout.task)
        except metrics.METRIC_FAILURES:
            return None
        return None if isinstance(v, (list, dict)) or not np.isfinite(v) else float(v)

    models = {}
    for model_name in scores:
        models[model_name] = {}
        for name in names:
            v = value(name, model_name)
            entry = {"value": None if v is None else round(v, 4)}
            if name == primary and v is not None:
                entry["ci"] = _ci([value(primary, model_name, idx) for idx in draws])
            models[model_name][name] = entry
    diffs = [None if (a is None or b is None) else a - b
             for a, b in ((value(primary, "network", i), value(primary, "trivial", i))
                          for i in draws)]
    diff_value = round(value(primary, "network") - value(primary, "trivial"), 4)
    diff_ci = _ci(diffs)
    earned = bool(diff_ci) and (diff_ci[0] > 0 if mode == "max" else diff_ci[1] < 0)
    _write(eval_dir / "metrics.json", {
        "schema": "vega/metrics@1", "run": run, "task": raw["task"]["kind"], "primary": primary,
        "mode": mode, "n_test": len(y), "n_boot": n_boot, "boot_seed": seed,
        "ci_level": 0.95, "predictions_layout": "vega/predictions-layout@1",
        "models": models, "differences": {"trivial": {"value": diff_value, "ci": diff_ci}},
        "verdict": {"strongest_baseline": "trivial", "diff": diff_value, "ci": diff_ci,
                    "earned": earned},
        "upstream": {"valid": {primary: {"value": 0.5, "_upstream": True}}}})
    _write(eval_dir / "threshold.json", {
        "schema": "vega/threshold@1", "task": kind, "value": threshold,
        "applies": threshold is not None,
        "scale": "probability" if threshold is not None else None, "per_label": None,
        "source": "valid-optimum" if threshold is not None else None})
    _write(eval_dir / "baselines.json", {"baselines": [
        {"name": "trivial", "fitted": True, "valid": {},
         "test": models["trivial"][primary]}]})
    _write(eval_dir / "slices.json", {"primary": primary, "worst": None, "slices": []})
    _write(eval_dir / "robustness.json", {"checks": []})
    _write(eval_dir / "curves.json", {"curves": {}})
    _write(camp / "test_seal.json", {"schema": "vega/test-seal@1", "run": run,
                                     "exposures": 1, "opened_at": "2026-09-30T08:00:00Z",
                                     "test": {"path": "(the valid part, standing in)"},
                                     "history": []})
    return {"rows": len(y), "primary": primary, "kind": kind}


WORKERS = {"smoke": worker_smoke, "campaign": worker_campaign, "evaluation": worker_evaluation}


def _main(argv) -> int:
    if len(argv) >= 3 and argv[0] == "--worker":
        result = WORKERS[argv[1]](**json.loads(argv[2]))
        print(json.dumps(result, default=str))
        return 0
    return pytest.main([str(Path(__file__).resolve()), "-q", *argv])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
