#!/usr/bin/env python3
"""
nb_emit.py — build a Vega stage's notebook from the records that stage wrote.

    python3 nb_emit.py --stage forge|train|eval|pack --from <run-dir> --out <dir>/<name>.ipynb
                       [--layer architecture-layer.json] [--package orders_pipeline]

Only this script writes notebooks; no stage hand-writes cells (notebook-contract §1). The
notebook is not a report written after the fact: it is generated from the same spec and run
records the scripts used, it imports the same emitted code from its own directory, it is
executed in place, and its outputs must equal the layer's numbers — which the assertion cell
at its end is what checks.

Every notebook has the same skeleton (§1): a provenance header, an environment cell that pins
`sys.path` to the notebook's own directory and imports the emitted package, one section per
procedure step of the stage, and the assertion cell.

`--stage train` builds `train.ipynb` (§2): the ladder as a table, every run's curves and
reading from its own records, the diagnostics the catalog fired, the study's trials and
importances, the selection with its rule, and **a replay of the selected run's first
`--replay-epochs` epochs** asserted against `history.jsonl` within the backend's determinism
tolerance — the notebook proves the run is reproducible without paying for the whole run again.

`--stage eval` builds `evaluate.ipynb` (§2) from `eval/<run>/`: the seal, every metric
recomputed from `predictions.test.parquet` and checked against `metrics.json` at `1e-9`, the
bootstrap intervals redrawn from their definition at the recorded seed and checked at `1e-6`,
the curves, the slices, the baselines and the verdict. It **never opens the test part**: it
imports the metrics module and nothing else from the emitted package, and its last cell proves
the data adapter stayed out of `sys.modules` — so re-executing it costs nothing against the
seal.

`--stage pack` builds `predict.ipynb` (§2) from a bundle directory: the input schema, a
prediction on `examples/rows.csv`, the same rows scored again against `examples/expected.parquet`
at the manifest's tolerance, every file checked against its recorded digest, the usage lines,
and the intended-use reading. It imports **only the bundle** — `from predict import load` and
nothing from the chain — because a notebook that reached back into the campaign would prove the
opposite of what this stage claims.

`--stage forge` builds `engineer.ipynb` (§2): the decisions table from `netspec.json`,
`Network.build(spec)` with the parameter count per block, one forward pass on a real batch
with every batch key's shape and the per-block output shapes (and, for a graph sample, the
degree histogram of the train view), the loss written out and evaluated on that batch, **one
manual optimisation step** against the backend's optimizer, the schedule's learning-rate curve,
and the mechanics reading.

**The cells drive the v2.3 adapter classes** (D1 §4.2, §12.1; BUILD-PLAN B20): the DataSource is
a `ShapedParts()` instance, the Trainer a `Trainer(store)` given the ModelStore
(`fit(model, spec, data, callbacks, run=RunContext(...))`; the device, the loss and the
learning-rate curve from the Trainer module's `optim.device_for`, `losses.build_loss` and
`optim.resolved_lrs`, the functions the Trainer itself calls), the Model is built by `Network.build(spec)` and the ModelStore is a
`ModelRegistry()`. Each import is resolved by the
port's *role* in the package's `blueprint-map.json` (notebook-contract §1) — the module names are
the blueprint's, never this script's — and the class each role's module holds is the one
`vega_adapters.ADAPTER_CLASSES` names (the backend contract fixes them). A package emitted by
the v2.2 generation (module-level adapter functions, no `generation: v2.3` in its map) is
refused: its notebook would call classes it does not have. The eval notebook reads the stored
head (`vega/predictions-layout@1`, R-C2) through the package's `TaskReadout` — the one source of
output semantics, R-B7 — and calls every metric as `metric(name, y, score, MetricOptions,
task)`; the train notebook replays the selected run from the spec exactly as that run resolved
it and stops itself after `--replay-epochs` records; the pack notebook's equality cell is the
bundle's own `predict.check_examples(HERE)` (R-D16), and a window bundle's single-row answer is
asked with the `MANIFEST.sample.history_rows` rows before it as history.

Each downstream stage reads records its own skill writes (`experiments.json`, `ladder.json`,
`history.jsonl`; `eval/<run>/metrics.json`; `bundle/MANIFEST.json`). A stage whose records
this build cannot produce is refused rather than served a notebook whose cells would be a
guess at records that do not exist yet.

Exit codes: 0 wrote the notebook; 2 an input or a stage this build cannot serve.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_core import read_json  # noqa: E402

STAGES = ("forge", "train", "eval", "pack")
NOT_YET = {}

GENERATED_BY = ("Generated by `nb_emit.py`; edit the spec or the run, not this notebook.")


# --------------------------------------------------------------------------- cells

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _lines(text)}


def _lines(text):
    out = text.rstrip("\n").split("\n")
    return [line + "\n" for line in out[:-1]] + [out[-1]]


def _parses(cells):
    """Every code cell is Python before it is written.

    The cells are built from triple-quoted templates, so an escape that survives one level of
    quoting too few (a bare \\n inside an f-string, a backslash line continuation) produces a
    cell that reads correctly here and raises SyntaxError only when someone runs the notebook.
    Emitting code that was never parsed is the one defect this file can always catch itself,
    so it does, and it refuses rather than writing a notebook that cannot run.
    """
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:
            line = (source.split("\n")[exc.lineno - 1] if exc.lineno else "").strip()
            raise SystemExit(
                f"ERROR: cell {i} is not valid Python — {exc.msg} at cell line {exc.lineno}: "
                f"{line!r}. The emitter builds cells from quoted templates; a literal backslash "
                f"in emitted code needs one more level of escaping than it looks like it does.")


def notebook(cells, kernel="python3"):
    # nbformat 4.5 requires a per-cell id; deriving it from the position keeps the notebook
    # byte-identical across two emissions of the same records, which matters because the
    # layer records the file's digest
    _parses(cells)
    for i, cell in enumerate(cells):
        cell["id"] = f"vega-{i:02d}"
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": kernel},
                         "language_info": {"name": "python"},
                         "vega": {"generated_by": "nb_emit.py"}},
            "nbformat": 4, "nbformat_minor": 5}


# --------------------------------------------------------------------------- forge

def sample_of(spec) -> dict:
    """The spec's `data_contract.sample` (a spec that names none is a row sample)."""
    sample = (spec.get("data_contract") or {}).get("sample")
    return dict(sample) if isinstance(sample, dict) else {"shape": "row"}


def _sample_line(sample) -> str:
    shape = sample.get("shape") or "row"
    if shape == "window":
        return (f"window: length {sample.get('length')}, stride {sample.get('stride')}, "
                f"horizon {sample.get('horizon')}, series by `{sample.get('series_by')}`")
    if shape == "graph":
        return (f"graph: level {sample.get('level')}, {sample.get('protocol')}, batching "
                f"{sample.get('batching')}")
    return "row"


def header_cell(stage, spec, smoke, run_dir):
    inp = spec.get("input") or {}
    return md(f"""# `{stage}` — {(spec.get('family') or {}).get('name')} on \
{Path(inp.get('model') or '?').name}

| | |
|---|---|
| stage | `/vega-{stage}` |
| chain input | `{inp.get('model')}` |
| input digest | `{inp.get('model_digest')}` |
| run directory | `{run_dir}` |
| spec digest | `{inp.get('manifest_digest')}` (shaped manifest) |
| backend | `{(spec.get('backend') or {}).get('name')}` \
{smoke.get('backend_version', '')} |
| seed | `{spec.get('seed')}` |
| device | `{smoke.get('device')}` |
| sample | {_sample_line(sample_of(spec))} |

*{GENERATED_BY}*""")


# The project root is where the spec's relative paths (`shaped/…`, `analysis/…`) resolve: the
# first directory, from the notebook's own upwards, holding the train part the spec names; a
# spec whose parts are absolute paths falls back to the chain's layout, <root>/vega-runs/<ts>.
_ROOT_FINDER = """def project_root(here, spec_raw):
    train = ((spec_raw.get("data_contract") or {}).get("parts") or {}).get("train")
    if train and not Path(train).is_absolute():
        for candidate in (here, *here.parents):
            if (candidate / train).exists():
                return candidate
    return here.parent.parent"""


def environment_cell(package, layout, run_dir="."):
    return code(f"""import json, os, sys
from pathlib import Path

# The notebook reads only its own directory and the shaped data the spec names — and those
# are two different roots: the emitted package sits beside this file, while the spec's paths
# (`shaped/…`, `analysis/…`) are written against the project root the chain ran in. `cwd` is
# whichever of the two the executor happened to pick — Jupyter picks the notebook's directory,
# `nbconvert` from a project root picks that — so neither is assumed: the package is found,
# and the root is derived from it.
HERE = next((c for c in (Path.cwd(), Path.cwd() / {run_dir!r})
             if (c / {package!r}).is_dir()), Path.cwd())
spec_raw = json.loads((HERE / "netspec.json").read_text())
smoke = json.loads((HERE / "smoke.json").read_text())

{_ROOT_FINDER}

ROOT = project_root(HERE, spec_raw)
sys.path.insert(0, str(HERE))
os.chdir(ROOT)

import torch
import numpy as np

# Each port's adapter, from the module the blueprint map gives its role (Model, DataSource,
# Trainer) — the classes the backend contract names (D1 §4.2), not module-level functions —
# and the two support modules this notebook reads: the Trainer's optimiser, the metrics.
import {package}
from {package}.domain import NetSpec, Part, SealedPartError
from {package}.{layout.module('Model')} import Network
from {package}.{layout.module('DataSource')} import ShapedParts
from {package}.{layout.module('Trainer')} import Trainer
from {package}.{layout.support('Trainer', 'optim')} import build_optimizer, device_for, resolved_lrs
from {package}.{layout.support('Trainer', 'losses')} import build_loss
from {package}.{layout.directory('Model')} import metrics

spec = NetSpec.from_dict(spec_raw)
data = ShapedParts()         # the DataSource port's adapter: stateless; the test part stays sealed

print("package     ", {package}.__name__)
print("torch       ", torch.__version__)
print("device      ", device_for(spec))
print("task, family", spec.task.value, spec.family.value)
print("sample      ", spec_raw.get("data_contract", {{}}).get("sample", {{"shape": "row"}}))
print("metrics     ", metrics.names_for(spec))
print("spec digest ", spec.digest)""")


def decisions_cells(spec, readings):
    per_family = (readings or {}).get("per_family") or {}
    intro = "\n\n".join(f"**{k}** — {v}" for k, v in per_family.items() if v)
    return [
        md("## 1. The decisions, and what settled each one\n\n"
           "Every row cites the source that justifies it. A choice with no source is refused "
           "by the compiler, so this column is never empty.\n\n" + intro),
        code("""families = spec_raw.get("decision_families") or {}
rows = []

def add(label, node):
    if not isinstance(node, dict):
        return
    value = node.get("name") or node.get("primary") or node.get("scheme") or ""
    if not value and "batch_size" in node:
        value = f"batch {node['batch_size']}, {node.get('max_epochs')} epochs max"
    rows.append((label, str(value), node.get("source", ""),
                 (node.get("catalog") if isinstance(node.get("catalog"), str) else
                  ", ".join(node.get("catalog") or [])) or "",
                 (node.get("rationale") or "")))

for family, keys in families.items():
    for key in keys:
        if key == "blocks":
            for b in spec_raw.get("blocks") or []:
                add(f"{family} · {b['id']} ({b['type']})", b)
        elif key == "regularization":
            for r in spec_raw.get("regularization") or []:
                add(f"{family} · {r.get('kind')}", r)
        else:
            add(f"{family} · {key}", spec_raw.get(key))

width = max((len(r[0]) for r in rows), default=0)
for label, value, source, catalog, rationale in rows:
    print(f"{label:<{width}}  {value:<24}  {source:<28}  {catalog}")
    if rationale:
        print(f"{'':<{width}}  -> {rationale}")
print()
print(len(rows), "traced choices")
assert all(r[2] for r in rows), "a choice with no source reached the notebook"
n_choices = len(rows)"""),
    ]


def build_cells():
    return [
        md("## 2. The network the spec builds\n\n"
           "`Network.build(spec)` is the only constructor of the network: it seeds, builds the "
           "blocks in the order the spec lists them and initialises them as the spec names. "
           "The parameter count per block below is the same number the layer shows."),
        code("""model = Network.build(spec)
total, trainable = model.param_count()
print(f"{total:,} parameters ({trainable:,} trainable)\\n")
print(model)"""),
        md("## 3. One forward pass on a real batch\n\n"
           "The batch is the first batch of the train part, read through the DataSource "
           "adapter — which raises on the test part before opening anything, so a notebook "
           "cannot look at it either. Every key the batch carries is printed with its shape: "
           "the features of each sample (`x_num`, `x_cat`: a row, a window `(B, L, F)`, or "
           "every node of the graph view), the label, and for a graph the edges and which "
           "nodes are the targets or which graph each node belongs to."),
        code("""train = data.load_part(spec, Part.TRAIN)
batch = next(iter(data.make_loader(train, spec, shuffle=False)))

def shape_of(value):
    return tuple(value.shape) if hasattr(value, "shape") else value

print(f"train part: {len(train)} sample(s) as {type(train).__name__}; the first batch:")
width = max(len(str(k)) for k in batch)
for key, value in batch.items():
    print(f"  {str(key):<{width}}  {shape_of(value)}")

model.eval()
with torch.no_grad():
    out = model(batch)
print("\\noutput:", tuple(out.shape))

shapes = model.block_shapes(batch)
width = max(len(s["id"]) for s in shapes)
for s in shapes:
    print(f"{s['id']:<{width}}  out {str(s['out_shape']):<16}  {s['params']:>8,} params")

# the seal is checked, not assumed: the refusal must be the DataSource's own SealedPartError,
# raised before a file is opened — any other outcome fails this cell
try:
    data.load_part(spec, Part.TEST)
except SealedPartError as exc:
    print("\\nseal holds:", type(exc).__name__)
else:
    raise AssertionError("the test part opened — the seal is broken")"""),
    ]


def degree_cells():
    """The degree histogram of the train view — added for a graph sample only (D1 §16)."""
    return [
        md("### The train view's degrees\n\n"
           "What message passing sees: how many edges arrive at each node of the train view. "
           "A node with no incoming edge is updated from its own features alone, and a long "
           "tail of high-degree nodes is where a mean or a sum aggregator behaves most "
           "differently — so the histogram is the first thing to read before the graph "
           "blocks' numbers."),
        code("""edge_index = torch.as_tensor(train.edge_index)
n_nodes = int(train.x_num.shape[0])
in_degree = torch.bincount(edge_index[1], minlength=n_nodes)
out_degree = torch.bincount(edge_index[0], minlength=n_nodes)
unit = "target node(s)" if train.target_index is not None else "graph(s)"
print(f"train view: {n_nodes} node(s), {edge_index.shape[1]} edge(s), {len(train)} {unit}")
print(f"in-degree  min {int(in_degree.min())}, median {float(in_degree.float().median()):g}, "
      f"mean {float(in_degree.float().mean()):.3g}, max {int(in_degree.max())}")
print(f"out-degree min {int(out_degree.min())}, max {int(out_degree.max())}")
isolated = int((in_degree == 0).sum())
print(f"{isolated} node(s) with no incoming edge ({isolated / max(1, n_nodes):.1%})")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = int(in_degree.max())
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.hist(in_degree.numpy(), bins=range(0, top + 2), align="left", rwidth=0.85)
    ax.set_xlabel("in-degree"); ax.set_ylabel("nodes")
    ax.set_title("degree histogram of the train view")
    fig.tight_layout(); fig.savefig(HERE / "degree.png", dpi=110)
    plt.close(fig)
    print("\\nwrote", HERE / "degree.png")
except ImportError:
    print("\\nmatplotlib is not installed; the histogram is the counts above")"""),
    ]


def loss_cells(spec):
    return [
        md("## 4. The objective, written out\n\n"
           "The loss is the spec's, with the spec's parameters, applied to the batch above. "
           "The head produced logits — the link function belongs to the loss, and applying a "
           "sigmoid in both places is the classic silent bug the compiler refuses."),
        code("""loss_fn = build_loss(spec)
print("loss    ", loss_fn)
print("formula ", loss_fn.formula())

with torch.no_grad():
    value = loss_fn(model(batch), batch, model)
print("\\nL(batch) =", float(value))"""),
    ]


def manual_step_cells():
    return [
        md("## 5. One optimisation step, by hand\n\n"
           "This is what \"documenting the weight update\" means concretely. The gradient of "
           "the loss with respect to the head's weights is computed explicitly, the "
           "optimizer's update rule is applied to those weights by hand, and the result is "
           "compared with what one step of the backend's optimizer does to the same "
           "parameters from the same state. If the two disagree, either the rule written in "
           "the layer's mechanics panel or the optimizer this run used is not what the spec "
           "says."),
        code("""head_id = model.ids[-1]
head = model.stages[-1]
W = head.linear.weight          # the head's weight matrix (a cumulative head: its score layer)

# --- the gradient, explicitly
model.train()
model.zero_grad(set_to_none=True)
loss = loss_fn(model(batch), batch, model)
loss.backward()
g = W.grad.detach().clone()
print("head block      ", head_id)
print("||g||           ", float(g.norm()))

# --- the update rule, by hand: the optimiser's FIRST step, from zero state
node = spec_raw["optimizer"]
p = node.get("params", {})
lr = float(p.get("lr", 1e-3))
name = node["name"]
theta = W.detach().clone()
# the weight decay is the regularization declaration's (netspec-contract §3, R2); the
# optimizer's own params only when nothing declares it
declared = [r for r in spec_raw.get("regularization") or [] if r.get("kind") == "weight_decay"]
wd = float(declared[0]["value"] if declared else p.get("weight_decay", 0.0))
eps = float(p.get("eps", 1e-8))

if name in ("adamw", "adam"):
    b1, b2 = [float(b) for b in p.get("betas", [0.9, 0.999])]
    # adam folds the decay into the gradient (L2); adamw decays the weights apart from it
    grad = g if name == "adamw" else g + wd * theta
    # first step: m = (1-b1)g, v = (1-b2)g^2, and the bias correction divides both back out,
    # so m_hat/(sqrt(v_hat)+eps) is g/(|g|+eps) elementwise
    m_hat = ((1 - b1) * grad) / (1 - b1)
    v_hat = ((1 - b2) * grad * grad) / (1 - b2)
    step = m_hat / (v_hat.sqrt() + eps)
    by_hand = (theta - lr * (step + wd * theta) if name == "adamw"     # decoupled decay
               else theta - lr * step)
elif name == "sgd":
    grad = g + wd * theta
    momentum = float(p.get("momentum", 0.0))
    if momentum and p.get("nesterov"):
        grad = grad + momentum * grad          # the first step's buffer is the gradient itself
    by_hand = theta - lr * grad
elif name == "rmsprop":
    alpha = float(p.get("alpha", 0.99))
    grad = g + wd * theta
    v = (1 - alpha) * grad * grad
    if p.get("centered"):
        v = v - ((1 - alpha) * grad) ** 2
    by_hand = theta - lr * grad / (v.sqrt() + eps)
elif name == "radam":
    # the first step's variance is not tractable (rho_1 = 1 <= 5), so RAdam takes the
    # bias-corrected momentum alone: m_hat = g
    grad = g + wd * theta
    by_hand = theta - lr * grad
else:
    by_hand = None
    print("no hand rule written for", name)

# --- what the backend's optimizer does, from the same state
opt = build_optimizer(model, spec)
opt.step()
by_optimizer = W.detach().clone()

if by_hand is not None:
    delta = float((by_hand - by_optimizer).abs().max())
    print("theta - lr*(...)  vs  optimizer.step()")
    print("max |difference| ", delta)
    assert delta < 1e-6, f"the hand rule and {name}.step() disagree by {delta}"
    print("\\nthe rule in the layer's mechanics panel is the rule this run applies")
manual_step_delta = float((by_hand - by_optimizer).abs().max()) if by_hand is not None else 0.0"""),
    ]


def schedule_cells(readings):
    mechanics = (readings or {}).get("mechanics") or ""
    return [
        md("## 6. The schedule\n\n"
           "The learning rate the optimizer will see at each epoch, resolved from the spec "
           "before any run — which is why the layer can draw this curve on a network that has "
           "not been trained yet.\n\n" + mechanics),
        code("""# a per-step schedule steps once per batch: the train part's batches, as the loader cuts them
steps_per_epoch = max(1, sum(1 for _ in data.make_loader(train, spec, shuffle=False)))
lrs = smoke.get("schedule_lrs") or resolved_lrs(spec, steps_per_epoch)
print("steps/epoch ", steps_per_epoch)
print("epochs      ", len(lrs))
print("lr[0]       ", lrs[0])
print("lr[max]     ", max(lrs), "at epoch", lrs.index(max(lrs)) + 1)
print("lr[-1]      ", lrs[-1])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.plot(range(1, len(lrs) + 1), lrs)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title(f"{spec_raw['schedule']['name']} schedule")
    fig.tight_layout(); fig.savefig(HERE / "schedule.png", dpi=110)
    print("\\nwrote", HERE / "schedule.png")
except ImportError:
    print("\\nmatplotlib is not installed; the curve is the list above")"""),
    ]


def assertion_cells():
    return [
        md("## 7. The notebook and the layer agree\n\n"
           "The last cell is the check that makes the rest of this notebook worth reading: "
           "every number printed above is compared with the record the layer carries. A "
           "notebook that disagrees with its layer is a defect in one of the two, and this "
           "is where it surfaces. Tolerances are the forge stage's: the parameter count is "
           "exact, the forward outputs are within 1e-5."),
        code("""def assert_matches():
    checks = []

    def check(name, got, want, exact=True, atol=1e-5):
        if exact:
            passed = got == want
        else:
            passed = abs(float(got) - float(want)) <= atol
        checks.append((name, got, want, passed))
        return passed

    check("params_total", int(total), int(smoke["params_total"]))
    check("params_trainable", int(trainable), int(smoke["params_trainable"]))
    check("forward_ok", True, bool(smoke["forward_ok"]))
    check("out_shape", [int(s) for s in out.shape[1:]], list(smoke["out_shape"]))
    check("n_blocks", len(shapes), len(smoke["per_block"]))
    by_id = {s["id"]: s for s in smoke["per_block"]}
    for s in shapes:
        want = by_id.get(s["id"], {})
        check(f"block {s['id']} out_shape", s["out_shape"], want.get("out_shape"))
        check(f"block {s['id']} params", s["params"], want.get("params"))
    if smoke.get("loss_formula"):
        check("loss formula", loss_fn.formula(), smoke["loss_formula"])
    if smoke.get("sample"):
        # the smoke's sample block (D1 §16) describes the sample this notebook just read
        declared = spec_raw.get("data_contract", {}).get("sample", {}).get("shape", "row")
        check("sample shape", declared, smoke["sample"].get("shape"))
    check("manual step vs optimizer.step()", manual_step_delta, 0.0, exact=False, atol=1e-6)

    width = max(len(c[0]) for c in checks)
    for name, got, want, passed in checks:
        mark = "ok  " if passed else "FAIL"
        print(f"{mark}  {name:<{width}}  notebook {got!r}  layer {want!r}")
    failed = [c for c in checks if not c[3]]
    print()
    if failed:
        raise AssertionError(f"{len(failed)} of {len(checks)} checks disagree with the "
                             f"layer's record: {[c[0] for c in failed]}")
    print(f"{len(checks)} checks: the notebook's numbers are the layer's numbers")

assert_matches()"""),
    ]


def forge_notebook(run_dir, spec, smoke, readings, package, layout):
    cells = [header_cell("forge", spec, smoke, run_dir),
             md("## 0. Environment\n\nThe notebook imports the emitted package from its own "
                "directory — the same code the scripts ran, not a copy — and fails loudly if "
                "a module is missing."),
             environment_cell(package, layout, run_dir)]
    if (readings or {}).get("abstract"):
        cells.append(md("### What this network is\n\n" + readings["abstract"]))
    cells += decisions_cells(spec, readings)
    cells += build_cells()
    if sample_of(spec).get("shape") == "graph":
        cells += degree_cells()
    cells += loss_cells(spec)
    cells += manual_step_cells()
    cells += schedule_cells(readings)
    cells += assertion_cells()
    return notebook(cells)


# --------------------------------------------------------------------------- train

def train_header(campaign, experiments, camp_dir):
    inp = campaign.get("input") or {}
    selection = experiments.get("selection") or {}
    budget = campaign.get("budget") or {}
    return md(f"""# `train` — the campaign in {camp_dir}

| | |
|---|---|
| stage | `/vega-train` |
| chain input | `{inp.get('model')}` |
| netspec digest | `{inp.get('netspec_digest')}` |
| package | `{inp.get('package')}` |
| backend | `{(campaign.get('backend') or {}).get('name')} \
{(campaign.get('backend') or {}).get('version')}` |
| device | `{campaign.get('device')}` |
| seed | `{campaign.get('seed')}` |
| determinism tolerance | `{(campaign.get('determinism') or {}).get('tolerance')}` |
| budget | {budget.get('spent_min')} of {budget.get('wall_clock_min')} min |
| mode | `{campaign.get('mode')}` |
| selected | **{selection.get('run')}** |

*{GENERATED_BY}*""")


def train_environment_cell(package, layout):
    return code(f"""import json, os, sys
from pathlib import Path

# The campaign directory holds the package and the records; the spec's paths (`shaped…/`,
# `analysis/…`) are written against the project root. The forge notebook resolves both roots
# and so must this one, or the replay cannot open the parts the spec names.
HERE = Path.cwd().resolve()

{_ROOT_FINDER}

ROOT = project_root(HERE, json.loads((HERE / "netspec.json").read_text()))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)

import numpy as np
import torch

# The four ports a fit drives, each from the module the blueprint map gives its role (Model,
# DataSource, Trainer, ModelStore) — the adapter classes, not module-level functions.
import {package}
from {package}.domain import NetSpec, Part, RunContext
from {package}.{layout.module('Model')} import Network
from {package}.{layout.module('DataSource')} import ShapedParts
from {package}.{layout.module('Trainer')} import Trainer
from {package}.{layout.support('Trainer', 'optim')} import device_for
from {package}.{layout.module('ModelStore')} import ModelRegistry

def read_json(name):
    return json.loads((HERE / name).read_text())

def read_jsonl(name):
    path = HERE / name
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out

campaign   = read_json("campaign.json")
ladder     = read_json("ladder.json")
experiments = read_json("experiments.json")
spec_raw   = read_json("netspec.json")
spec       = NetSpec.from_dict(spec_raw)
primary    = (spec_raw.get("metrics") or {{}}).get("primary")
mode       = ((spec_raw.get("metrics") or {{}}).get("selection") or {{}}).get("mode", "max")
tolerance  = float((campaign.get("determinism") or {{}}).get("tolerance", 0.0))

print("package    ", {package}.__name__)
print("torch      ", torch.__version__)
print("device     ", device_for(spec))
print("runs       ", len(experiments.get("runs") or []))
print("primary    ", primary, f"({{mode}})")
print("selected   ", (experiments.get("selection") or {{}}).get("run"))
print("tolerance  ", tolerance)""")


def ladder_cells(readings):
    return [
        md("## 1. The ladder\n\n"
           "Every rung states the hypothesis it tests and the source that motivated it. A rung "
           "whose delta spanned two decision families, or that produced a spec the compiler "
           "refused, never reached this table — `plan_ladder.py` dropped it with the rule "
           "named.\n\n" + (readings or {}).get("abstract", "")),
        code("""rows = []
for rung in ladder.get("runs") or []:
    delta = "; ".join(f"{e['path']} → {json.dumps(e.get('to'))[:28]}"
                      for e in rung.get("delta") or []) or "(none — the spec as engineered)"
    rows.append((rung["id"], rung.get("kind", ""), rung.get("family") or rung.get("study") or "-",
                 delta[:70], rung.get("source", ""), rung.get("hypothesis", "")))

w = max(len(r[0]) for r in rows)
for rid, kind, fam, delta, source, hypothesis in rows:
    print(f"{rid:<{w}}  {kind:<9} {fam:<14} {delta:<72} {source}")
    if hypothesis:
        print(f"{'':<{w}}  -> {hypothesis}")
print()
print(len(rows), "planned run(s)")
n_planned = len(rows)"""),
    ]


def run_cells(readings):
    per_run = (readings or {}).get("per_run") or {}
    text = "\n\n".join(f"**{k}** — {v}" for k, v in per_run.items() if v)
    return [
        md("## 2. The runs, from their own records\n\n"
           "Each curve is drawn from that run's `history.jsonl`; nothing here is recomputed, "
           "because the point of the record is that the trainer and the layer and this "
           "notebook all read the same numbers.\n\n" + text),
        code("""import matplotlib
matplotlib.use("Agg")          # headless: the figure is shown with display(), not a GUI loop
import matplotlib.pyplot as plt

histories = {}
for row in experiments.get("runs") or []:
    rid = row["id"]
    records = read_jsonl(f"runs/{rid}/history.jsonl")
    if records:
        histories[rid] = records

print(f"{'run':<8} {'epochs':>7} {'best':>6} {'valid.'+str(primary):>14} {'gap':>9}  "
      f"{'stopped_by':<16} diagnostics")
for row in experiments.get("runs") or []:
    value = row.get("primary")
    gap = (row.get("gap") or {}).get("loss")
    print(f"{row['id']:<8} {str(row.get('epochs')):>7} {str(row.get('best_epoch')):>6} "
          f"{('·' if value is None else f'{float(value):.5g}'):>14} "
          f"{('·' if gap is None else f'{float(gap):.4g}'):>9}  "
          f"{str(row.get('stopped_by')):<16} {', '.join(row.get('diagnostics') or []) or '—'}")

def curve(rid, ax_specs):
    records = histories.get(rid) or []
    if not records:
        print(f"{rid}: no history"); return
    epochs = [r.get("epoch") for r in records]
    fig, axes = plt.subplots(1, len(ax_specs), figsize=(4.4 * len(ax_specs), 2.8))
    axes = axes if len(ax_specs) > 1 else [axes]
    for ax, (title, series) in zip(axes, ax_specs):
        for label, getter in series:
            ax.plot(epochs, [getter(r) for r in records], label=label, linewidth=1.4)
        ax.set_title(f"{rid} — {title}"); ax.set_xlabel("epoch"); ax.legend(fontsize=7)
    fig.tight_layout(); plt.show()

for rid in list(histories)[:8]:
    curve(rid, [
        ("loss", [("train", lambda r: (r.get("train") or {}).get("loss")),
                  ("valid", lambda r: (r.get("valid") or {}).get("loss"))]),
        (str(primary), [("train", lambda r: (r.get("train") or {}).get(primary)),
                        ("valid", lambda r: (r.get("valid") or {}).get(primary))]),
        ("lr and |g|", [("lr", lambda r: r.get("lr")),
                        ("|g|", lambda r: (r.get("grad_norm") or {}).get("global"))]),
    ])"""),
    ]


def diagnostics_cells():
    return [
        md("## 3. What the diagnostics found\n\n"
           "A diagnostic is a mechanical condition on the records. Its reading is the skill's, "
           "and the rung it proposes is a proposal — a diagnostic that could not be evaluated "
           "because the backend wrote a field as `null` is recorded as unavailable rather "
           "than guessed."),
        code("""fired, unavailable = [], []
for row in experiments.get("runs") or []:
    for entry in row.get("diagnostics_detail") or []:
        (fired if entry.get("fired") else unavailable).append((row["id"], entry))

for rid, entry in fired:
    print(f"{rid}  {entry['code']} {entry.get('name')}")
    print(f"      {entry.get('evidence')}")
    if entry.get("remedies"):
        print(f"      remedies the catalog names: {'; '.join(entry['remedies'])}")
print()
codes = sorted({e['code'] for _, e in unavailable})
print(f"{len(fired)} fired; unavailable on some run: {', '.join(codes) or '—'}")
n_fired = len(fired)"""),
    ]


def hpo_cells(readings):
    return [
        md("## 4. The study\n\n"
           "Every trial is a run in the same ledger: it recompiled, recorded its digests and "
           "got its diagnostics evaluated like any rung. The importance record says *how* it "
           "was computed — fANOVA and a marginal spread are not the same claim.\n\n"
           + ((readings or {}).get("hpo") or "")),
        code("""studies = experiments.get("hpo") or []
if not studies:
    print("no study ran in this campaign")
for study in studies:
    print(f"study {study['study']}: {study.get('trials')} trial(s), "
          f"{study.get('completed')} completed, {study.get('pruned')} pruned, "
          f"{study.get('failed')} failed")
    print(f"  sampler {study.get('sampler')}, pruner {study.get('pruner')}")
    print(f"  best {study.get('best')} at {study.get('best_value')}")
    print(f"  importance by {study.get('importance_how')}:")
    for k, v in sorted((study.get('importance') or {}).items(), key=lambda kv: -kv[1]):
        print(f"    {v:>6.3f}  {k}")
    trials = read_jsonl(f"hpo/{study['study']}/trials.jsonl")
    scored = [t for t in trials if t.get("value") is not None]
    if scored:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        for t in scored:
            ax.scatter([int(t["id"].rsplit("-", 1)[1])], [t["value"]],
                       marker="x" if t.get("status") == "pruned" else "o",
                       color="#9a3a2c" if t.get("status") == "pruned" else "#1b3a73")
        base = next((r.get("primary") for r in experiments["runs"] if r["id"] == "R0"), None)
        if base is not None:
            ax.axhline(base, linestyle="--", linewidth=1, color="#777", label="R0")
            ax.legend(fontsize=7)
        ax.set_xlabel("trial"); ax.set_ylabel(f"valid {primary}")
        ax.set_title(f"{study['study']} — x marks a pruned trial")
        fig.tight_layout(); plt.show()"""),
    ]


def selection_cells(readings):
    return [
        md("## 5. The selection\n\n" + ((readings or {}).get("selection") or "")),
        code("""selection = experiments.get("selection") or {}
ranking = experiments.get("ranking") or {}
print("rule:", selection.get("rule"))
print()
by_id = {r["id"]: r for r in experiments.get("runs") or []}
for i, rid in enumerate(ranking.get("order") or [], start=1):
    row = by_id.get(rid, {})
    mark = "→" if rid == selection.get("run") else " "
    value = row.get("primary")
    state = "" if row.get("eligible", True) else \
        "   ineligible: " + ", ".join(row.get("ineligible_because") or [])
    print(f"{mark} {i:>2}. {rid:<8} {('·' if value is None else f'{float(value):.5g}'):>10}  "
          f"{(row.get('params') or 0):>8,} params{state}")
print()
print("selected:", selection.get("run"), "margin over baseline",
      selection.get("margin_over_baseline"))
if selection.get("override"):
    print("OVERRIDE:", selection["override"].get("source"), "—",
          selection["override"].get("reason"))
    print("the rule would have picked", selection["override"].get("rule_would_have_picked"))
selected_run = selection.get("run")"""),
    ]


def replay_cells(replay_epochs, layout):
    return [
        md(f"## 6. The selected run replays\n\n"
           f"Determinism is a claim the backend makes, and this is the cell that tests it: the "
           f"selected run's first {replay_epochs} epochs are trained again, from the spec that "
           f"run resolved and the code it trained with, and compared with what was recorded — "
           f"within the backend's declared tolerance, which on CPU is exact. The whole run is "
           f"not repeated; the first epochs are enough to catch a seed the backend did not "
           f"set, a nondeterministic kernel or a device that changed under the campaign.\n\n"
           f"The spec is replayed **as the run resolved it**, `max_epochs` included: the "
           f"learning-rate schedule is a function of it (a cosine or one-cycle curve over "
           f"{replay_epochs} epochs is not the run's curve), so the replay stops itself after "
           f"{replay_epochs} epoch records instead of shortening the spec. Its checkpoints go "
           f"through the ModelStore port into `replay/<run>/` beside this notebook — the "
           f"replay's scratch, never the run's own."),
        code(f"""REPLAY_EPOCHS = {replay_epochs}

run_dir = HERE / "runs" / selected_run
recorded = read_jsonl(f"runs/{{selected_run}}/history.jsonl")[:REPLAY_EPOCHS]
resolved = json.loads((run_dir / "netspec.resolved.json").read_text())
PACKAGE = campaign["input"]["package"]

# a run that re-emitted its package trained with THAT code, not the campaign root's
own_package = run_dir / PACKAGE
if own_package.is_dir():
    import importlib
    sys.path.insert(0, str(run_dir))
    for name in [m for m in list(sys.modules) if m == PACKAGE or m.startswith(PACKAGE + ".")]:
        del sys.modules[name]
    importlib.invalidate_caches()

    def from_package(module, name):
        \"\"\"A class of the re-imported package: `module` is package-relative.\"\"\"
        return getattr(importlib.import_module(f"{{PACKAGE}}.{{module}}"), name)

    # each port's adapter class, from the module the blueprint map gives its role
    network_cls = from_package("{layout.module('Model')}", "Network")
    data_cls = from_package("{layout.module('DataSource')}", "ShapedParts")
    trainer_cls = from_package("{layout.module('Trainer')}", "Trainer")
    store_cls = from_package("{layout.module('ModelStore')}", "ModelRegistry")
    # the spec and the run context must be values of THAT package's domain: the ports check
    # them by type, and the classes imported at the top are the campaign root's, different ones
    # once the modules are re-imported
    spec_cls = from_package("domain", "NetSpec")
    run_cls = from_package("domain", "RunContext")
    print("replaying against the package this run re-emitted:", own_package.name)
else:
    network_cls, data_cls, trainer_cls, store_cls = Network, ShapedParts, Trainer, ModelRegistry
    spec_cls, run_cls = NetSpec, RunContext

replay_spec = spec_cls.from_dict(resolved)
replay_model = network_cls.build(replay_spec)


class _Enough(Exception):
    \"\"\"Raised by the callbacks once REPLAY_EPOCHS records are in: the replay stops at an epoch
    boundary without changing the spec the run trained under.\"\"\"


fresh = []
class _Collect:
    def on_epoch(self, record):
        fresh.append(record.as_dict())
        if len(fresh) >= REPLAY_EPOCHS:
            raise _Enough()

scratch = HERE / "replay" / selected_run
try:
    trainer_cls(store_cls()).fit(replay_model, replay_spec, data_cls(), callbacks=_Collect(),
                                 run=run_cls(checkpoint_dir=str(scratch)))
except _Enough:
    pass
kept = sorted(p.name for p in scratch.glob("*")) if scratch.is_dir() else []
print(f"replay scratch {{scratch.relative_to(HERE)}}: {{', '.join(kept) or 'no checkpoint'}}")

problems = []
for i, (was, now) in enumerate(zip(recorded, fresh), start=1):
    for part in ("train", "valid"):
        for key, old in (was.get(part) or {{}}).items():
            new = (now.get(part) or {{}}).get(key)
            if old is None or new is None or not isinstance(old, (int, float)):
                continue
            if abs(float(old) - float(new)) > tolerance + 1e-12:
                problems.append(f"epoch {{i}} {{part}}.{{key}}: recorded {{old!r}}, "
                                f"replay {{new!r}}")

for i, (was, now) in enumerate(zip(recorded, fresh), start=1):
    print(f"epoch {{i}}  recorded valid.loss {{(was.get('valid') or {{}}).get('loss')}}  "
          f"replay {{(now.get('valid') or {{}}).get('loss')}}")
print()
replay_ok = not problems and len(fresh) >= len(recorded)
if problems:
    print(f"VEGA-REPLAY: mismatch {{selected_run}} problems={{len(problems)}}")
    for p in problems[:10]:
        print("MISMATCH", p)
    raise AssertionError(f"the replay of {{selected_run}} does not match its records "
                         f"({{len(problems)}} value(s) outside tolerance {{tolerance}}) — "
                         "determinism is a claim, and this is the test of it")
# an unambiguous marker: the apply script reads THIS, not the prose around it. A verdict a
# downstream script has to infer from a sentence is a verdict that can be misread.
print(f"VEGA-REPLAY: ok {{selected_run}} epochs={{len(fresh)}} tolerance={{tolerance}}")
print(f"replay of {{selected_run}}: {{len(fresh)}} epoch(s) match the records within "
      f"tolerance {{tolerance}}")"""),
    ]


def train_assertion_cells():
    return [
        md("## 7. The notebook and the layer agree\n\n"
           "Every number printed above comes from the campaign's records. This cell checks "
           "that the ledger the layer carries and the records on disk have not drifted apart — "
           "and that the run the layer names as selected is the run whose replay just passed."),
        code("""def assert_matches():
    checks = []
    def check(name, got, want):
        checks.append((name, got, want, got == want))

    check("runs in the ledger have histories",
          sorted(histories), sorted(r["id"] for r in experiments["runs"]
                                    if r.get("status") != "failed"))
    for row in experiments["runs"]:
        rid = row["id"]
        if rid not in histories:
            continue
        records = histories[rid]
        check(f"{rid} epochs", len(records), row.get("epochs"))
        best = row.get("best_epoch")
        if best is not None and row.get("primary") is not None:
            recorded = (records[best - 1].get("valid") or {}).get(primary)
            check(f"{rid} valid {primary} at its best epoch", recorded, row.get("primary"))
    check("the replayed run is the selected one", selected_run,
          (experiments.get("selection") or {}).get("run"))
    check("the replay matched", replay_ok, True)

    w = max(len(c[0]) for c in checks)
    for name, got, want, passed in checks:
        print(f"{'ok  ' if passed else 'FAIL'}  {name:<{w}}  notebook {got!r}  ledger {want!r}")
    failed = [c for c in checks if not c[3]]
    print()
    if failed:
        raise AssertionError(f"{len(failed)} of {len(checks)} checks disagree with the "
                             f"ledger: {[c[0] for c in failed]}")
    print(f"{len(checks)} checks: the notebook's numbers are the ledger's numbers")

assert_matches()"""),
    ]


def train_notebook(camp_dir, campaign, experiments, readings, package, layout, replay_epochs):
    cells = [train_header(campaign, experiments, camp_dir),
             md("## 0. Environment\n\nThe notebook imports the emitted package from its own "
                "directory — the same code the campaign trained with — and reads the campaign's "
                "records from disk."),
             train_environment_cell(package, layout)]
    cells += ladder_cells(readings)
    cells += run_cells(readings)
    cells += diagnostics_cells()
    cells += hpo_cells(readings)
    cells += selection_cells(readings)
    cells += replay_cells(replay_epochs, layout)
    cells += train_assertion_cells()
    return notebook(cells)



# --------------------------------------------------------------------------- the eval stage

def eval_header(seal, metrics, camp_dir, run_id):
    exposures = seal.get("exposures", 1)
    test = seal.get("test") or {}
    return md(f"""# `evaluate` — {run_id} on the sealed test part

| | |
|---|---|
| stage | `/vega-eval` |
| campaign | `{camp_dir}` |
| run | **{run_id}** |
| task | `{metrics.get('task')}` |
| primary metric | `{metrics.get('primary')}` ({metrics.get('mode')}) |
| test part | `{test.get('path')}` |
| test digest | `{test.get('digest')}` |
| rows | {metrics.get('n_test')} |
| sealed at | `{seal.get('opened_at')}` |
| **exposures** | **{exposures}** |
| bootstrap | {metrics.get('n_boot')} resamples at seed {metrics.get('boot_seed')} |

{"**The test part has been opened more than once.** Every number below was read after the "
 "answer was already known once; the verdict is qualified by that and the seal's history "
 "says why each look happened." if exposures > 1 else
 "The test part was opened once, by `evaluate.py`, under the seal above."}

**This notebook does not open the test part.** It recomputes every number from
`eval/{run_id}/predictions.test.parquet` — the scores and labels that one look stored — so
re-executing it as many times as you like costs nothing against the seal. The last cell
proves it: it checks that the data adapter was never imported.

*{GENERATED_BY}*""")


def eval_environment_cell(package, layout, run_id):
    return code(f"""import json, re, sys
from pathlib import Path

HERE = Path.cwd()
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import torch

# Only what reads the stored numbers: the domain's value objects, the metrics module, and the
# readout — what a stored head means, the one source of output semantics the evaluator scored
# with (R-B7). Not the network, not the store, and above all not
# `{layout.module('DataSource')}` — that module is the one door to the test part, and a
# notebook that imports it can open what the seal closed. The final cell checks that it stayed
# out of sys.modules.
from {package}.domain import MetricOptions, NetSpec
from {package}.{layout.directory('Model')} import metrics as M
from {package}.{layout.support('Model', 'readout')} import TaskReadout

EVAL = HERE / "eval" / "{run_id}"
def read_json(name, base=None):
    return json.loads(((base or EVAL) / name).read_text())

seal = read_json("test_seal.json", HERE) if (HERE / "test_seal.json").is_file() else {{}}
METRICS = read_json("metrics.json")
BASELINES = read_json("baselines.json")
SLICES = read_json("slices.json")
ROBUST = read_json("robustness.json")
CURVES = read_json("curves.json")
THRESH = read_json("threshold.json") if (EVAL / "threshold.json").is_file() else {{}}

# the spec the run trained under: its task (a sequence task is read as its base), its classes
# and its head layout (quantiles, horizon, the reconstruction's numeric width)
RESOLVED = HERE / "runs" / "{run_id}" / "netspec.resolved.json"
spec = NetSpec.from_dict(json.loads((RESOLVED if RESOLVED.is_file()
                                     else HERE / "netspec.json").read_text()))
READOUT = TaskReadout.for_spec(spec)
TASK = READOUT.task
KIND = READOUT.kind.value

primary = METRICS["primary"]
mode = METRICS.get("mode", "max")
n_boot = int(METRICS["n_boot"])
boot_seed = int(METRICS["boot_seed"])
CI_LEVEL = float(METRICS.get("ci_level") or 0.95)
ALPHA = 1.0 - CI_LEVEL

# the threshold the evaluation decided with (threshold.json): one probability, one per label in
# class order, or one value on an anomaly score's own scale
threshold = THRESH.get("value") if THRESH.get("applies") else None
per_label = THRESH.get("per_label") if THRESH.get("applies") else None
if per_label:
    threshold = tuple(float(per_label[str(c)]) for c in TASK.classes)
# what the metrics read beyond the scores, as the evaluation recorded it: a scale that only the
# train part can give (mase's) is never re-derived here — the notebook opens no part
RECORDED = METRICS.get("options") or {{}}
OPTIONS = MetricOptions(threshold=0.5 if threshold is None else threshold,
                        season=int(RECORDED.get("season") or 1),
                        mase_scale=RECORDED.get("mase_scale"))

print(f"pandas {{pd.__version__}}, numpy {{np.__version__}}, torch {{torch.__version__}}")
print(f"{{METRICS['n_test']}} test rows, task read as {{KIND}}, primary {{primary}} ({{mode}}), "
      f"{{n_boot}} resamples at seed {{boot_seed}}, {{CI_LEVEL:.0%}} intervals")
print(f"threshold: {{threshold if threshold is not None else 'none applies'}} "
      f"({{THRESH.get('scale') or 'no scale recorded'}})")
print(f"exposures: {{seal.get('exposures')}}")""")


def eval_predictions_cells():
    return [
        md("## 1. The stored predictions\n\n"
           "One look at the test part produced this table (`vega/predictions-layout@1`): the "
           "row ids, the truth (`y`, or `y_0 …` for a multi-column truth), the network's raw "
           "head — every output column, `score` or `score_0 …` — its per-row score where the "
           "reading has one (`row_score`), the decision, and each fitted baseline's scores on "
           "the same rows. What the metrics read of the network is the head read through the "
           "package's `TaskReadout`, the same object the evaluator scored with: a binary "
           "logit, the logits, the median column of a quantile head, the numeric part of a "
           "reconstruction; an anomaly reading's per-row score is the stored `row_score`. "
           "Everything below is arithmetic on these columns. The network is never run again, "
           "so this notebook cannot re-expose the part however often it is executed."),
        code("""PRED = EVAL / "predictions.test.parquet"
pred = pd.read_parquet(PRED, filesystem=pafs.LocalFileSystem())   # Arrow's reader, no Python in its threads
LAYOUT = {k.decode(): v.decode() for k, v in (pq.read_schema(PRED).metadata or {}).items()
          if k.startswith(b"vega.")}

def indexed(prefix):
    \"\"\"`<prefix>_0 … <prefix>_{k-1}`, in index order.\"\"\"
    found = [c for c in pred.columns if re.fullmatch(re.escape(prefix) + r"_\\d+", c)]
    return sorted(found, key=lambda c: int(c.rsplit("_", 1)[1]))

y_cols = indexed("y")
y = (pred[y_cols].to_numpy() if y_cols else
     pred["y"].to_numpy() if "y" in pred.columns else None)
head_cols = indexed("score") or ["score"]
head = pred[head_cols].to_numpy(dtype=np.float64)          # (N, head_out): the raw head
if KIND == "anomaly":
    if "row_score" not in pred.columns:
        raise AssertionError("an anomaly reading's metrics read the per-row score, and the "
                             "stored predictions carry no row_score column (R-C2)")
    network = pred["row_score"].to_numpy(dtype=np.float64)
else:
    network = READOUT.score(torch.tensor(head)).numpy()      # a copy: the frame's buffer is read-only
scores = {"network": network}

not_reproducible = {}
baselines = sorted({m.group(1) for c in pred.columns
                    for m in [re.fullmatch(r"score_([A-Za-z][\\w-]*?)(?:_\\d+)?", c)] if m})
for name in baselines:
    matrix = indexed(f"score_{name}")
    if matrix:
        scores[name] = pred[matrix].to_numpy(dtype=np.float64)
    elif network.ndim > 1:
        # one stored column for a model whose score is a matrix: its metrics cannot be
        # recomputed from it, and saying so is not the same as disagreeing
        not_reproducible[name] = "one score column stored for a matrix-valued score"
    else:
        scores[name] = pred[f"score_{name}"].to_numpy(dtype=np.float64)

print(f"{len(pred)} rows, head of {head.shape[1]} column(s) read as {KIND} "
      f"(score {tuple(network.shape[1:]) or 'one per row'}), "
      f"{len(scores)} scored model(s): {', '.join(scores)}")
print("stored layout:", LAYOUT or "no vega.* metadata")
if not_reproducible:
    print("not reproducible from this record:", not_reproducible)
if y is None:
    print("no truth stored: an unlabelled part")
elif KIND in ("binary", "multiclass", "ordinal", "anomaly"):
    print(f"labels: { {k.item(): n.item() for k, n in zip(*np.unique(y, return_counts=True))} }")
elif KIND == "multilabel":
    print(f"positives per label: {np.asarray(y).sum(axis=0).tolist()}")
else:
    print(f"truth {tuple(np.shape(y))}: mean {np.mean(y):.4g}, sd {np.std(y):.4g}")
pred.head()"""),
    ]


def eval_metric_cells():
    return [
        md("## 2. The point metrics, recomputed\n\n"
           "Each metric is recomputed here from the stored scores with the same "
           "`model.metrics` the evaluation used — `metric(name, y, score, options, task)`, "
           "the task's effective kind and the recorded threshold passed as values, never "
           "guessed from the numbers — and checked against `metrics.json` at `1e-9`. A "
           "disagreement means the layer is publishing a number no longer supported by the "
           "predictions beneath it. A metric that needs something only the train part holds "
           "(the `mase` scale) is recomputed when the evaluation recorded it and named "
           "otherwise, never re-derived by opening a part."),
        code("""NEEDS_SCALE = {"mase"}

# each baseline is read at its own threshold, chosen on its valid scores (baselines.json
# decision_threshold); the network, and a model without one, at threshold.json's
OPTIONS_OF = {}
for b in BASELINES.get("baselines") or []:
    own = (b or {}).get("decision_threshold") or {}
    t = (tuple(float(own["per_label"][str(c)]) for c in TASK.classes) if own.get("per_label")
         else own.get("at"))
    if t is not None:
        OPTIONS_OF[b["name"]] = MetricOptions(threshold=t, season=OPTIONS.season,
                                              mase_scale=OPTIONS.mase_scale)

def reproducible(name):
    return not (name in NEEDS_SCALE and OPTIONS.mase_scale is None)

def point(name, model):
    try:
        v = M.metric(name, y, scores[model], OPTIONS_OF.get(model, OPTIONS), TASK)
    except M.METRIC_FAILURES:
        return None
    return None if isinstance(v, (list, dict)) else v

rows, point_gaps = [], []
for model in scores:
    recorded = (METRICS["models"].get(model) or {})
    for name, entry in recorded.items():
        # an entry with no point is either a metric this part could not support or a
        # per-element vector (it carries `values`); neither has a scalar to check
        if not isinstance(entry, dict) or entry.get("value") is None:
            continue
        if not reproducible(name):
            not_reproducible[f"{model}.{name}"] = "the train-side scale was not recorded"
            continue
        got = point(name, model)
        rows.append((model, name, got, entry["value"]))
        if got is None or abs(float(got) - float(entry["value"])) > 1e-9 + 5e-5:
            point_gaps.append((model, name, got, entry["value"]))

frame = pd.DataFrame(rows, columns=["model", "metric", "notebook", "metrics.json"])
print(f"{len(rows)} point metrics recomputed, {len(point_gaps)} disagreement(s)")
frame.pivot(index="metric", columns="model", values="notebook")"""),
        md("*The tolerance is `1e-9` on the metric itself; the recorded values are rounded to "
           "four places when they are written, so the comparison allows that rounding and "
           "nothing more.*"),
    ]


def eval_bootstrap_cells():
    return [
        md("## 3. The intervals, redrawn\n\n"
           "The bootstrap is rebuilt here from its definition rather than by calling the "
           "evaluation's own helper: `numpy.random.default_rng(boot_seed)` draws `n_boot` "
           "index arrays, **once**, and every model is scored on the same draws. That is what "
           "makes the paired difference have an interval of its own — resampling each model "
           "separately would give two intervals whose overlap says nothing. The level is the "
           "one `metrics.json` records (`ci_level`). If `evaluate.py` had drawn them "
           "differently, this cell would disagree with it."),
        code("""rng = np.random.default_rng(boot_seed)
n = len(pred)
draws = [rng.integers(0, n, n) for _ in range(n_boot)]

def ci_of(samples, alpha=ALPHA):
    clean = sorted(float(s) for s in samples if s is not None and np.isfinite(s))
    if len(clean) < 2:
        return None
    lo = clean[max(0, int((alpha / 2) * len(clean)) - 1)]
    hi = clean[min(len(clean) - 1, int((1 - alpha / 2) * len(clean)))]
    return (round(lo, 4), round(hi, 4))

def at(idx, model):
    \"\"\"The primary on one resample, or None where it cannot be computed there.\"\"\"
    try:
        v = M.metric(primary, None if y is None else y[idx], scores[model][idx],
                     OPTIONS_OF.get(model, OPTIONS), TASK)
    except M.METRIC_FAILURES:
        return None
    return float(v) if v is not None and np.isfinite(v) else None

def resampled(model):
    return [v for v in (at(idx, model) for idx in draws) if v is not None]

ci_gaps = []
ci_rows = []
for model in (scores if reproducible(primary) else {}):
    entry = (METRICS["models"].get(model) or {}).get(primary) or {}
    want = entry.get("ci")
    got = ci_of(resampled(model))
    ci_rows.append((model, got, tuple(want) if want else None))
    if want and got and max(abs(got[0] - want[0]), abs(got[1] - want[1])) > 1e-6:
        ci_gaps.append((model, got, want))

pd.DataFrame([(m, str(g), str(w)) for m, g, w in ci_rows],
             columns=["model", f"notebook {primary} CI", "metrics.json"])"""),
        md("### The paired differences\n\n"
           "`network[i] − baseline[i]` on every resample `i`, which is the only comparison "
           "this evaluation is entitled to make. Bounds are checked at `1e-6`."),
        code("""def paired(model):
    out = []
    for idx in draws:
        x, z = at(idx, "network"), at(idx, model)
        if x is not None and z is not None:
            out.append(x - z)
    return out

diff_rows = []
for model, entry in (METRICS.get("differences") or {}).items():
    if model not in scores or not reproducible(primary):
        continue
    got, want = ci_of(paired(model)), entry.get("ci")
    diff_rows.append((model, entry.get("value"), str(got), str(tuple(want) if want else None),
                      "excludes 0" if want and (want[0] > 0 or want[1] < 0) else "covers 0"))
    if want and got and max(abs(got[0] - want[0]), abs(got[1] - want[1])) > 1e-6:
        ci_gaps.append((f"difference vs {model}", got, want))

print(f"{len(ci_gaps)} interval disagreement(s)")
pd.DataFrame(diff_rows, columns=["baseline", "point", "notebook CI", "metrics.json", "reads as"])"""),
    ]


def eval_curve_cells():
    return [
        md("## 4. The curves\n\n"
           "Drawn from the stored scores. The reliability curve is the one to read slowly: "
           "above the diagonal the network is under-confident, below it over-confident."),
        code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def points_of(curve):
    if not curve:
        return None
    if isinstance(curve.get("points"), list) and curve["points"]:
        pts = curve["points"]
        return [p[0] for p in pts], [p[1] for p in pts]
    if isinstance(curve.get("x"), list):
        return curve["x"], curve["y"]
    return None

panels = [(k, points_of((CURVES.get("curves") or CURVES).get(k)))
          for k in ("roc", "pr", "reliability")]
panels = [(k, v) for k, v in panels if v]
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, (xs, ys)) in zip(axes, panels):
        ax.plot(xs, ys, lw=1.6)
        if name in ("roc", "reliability"):
            ax.plot([0, 1], [0, 1], ls="--", lw=.8, color="grey")
        ax.set_title(name)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    print(f"{len(panels)} curve(s) drawn from the stored points: "
          f"{', '.join(k for k, _ in panels)}")
else:
    print("no curve points were recorded for this task kind")"""),
    ]


def eval_slice_cells(readings):
    cells = [
        md("## 5. Slices\n\n"
           "The slices the chain named — from the analysis layer's segments and drift columns, "
           "the spec's nominal features, and any the user declared *before* the look. A slice "
           "invented after seeing the test part is a hypothesis fitted to it, which is why "
           "`evaluate.py` refuses `--slices` on a second exposure."),
        code("""rows = []
for s in SLICES.get("slices") or []:
    e = (s.get("metrics") or {}).get(primary) or {}
    rows.append((s["name"], s["n"], e.get("value"),
                 str(e.get("ci")) if e.get("ci") else "·",
                 s.get("source"), "single class" if s.get("degenerate") else (s.get("note") or "")))
worst = SLICES.get("worst")
print(f"worst on {primary}: {worst}" if worst else "no slice could be ranked")
for gone in SLICES.get("not_cut") or []:
    print(f"not cut ({gone.get('source')}): {gone.get('why')}")
pd.DataFrame(rows, columns=["slice", "n", primary, "CI", "source", "note"])"""),
    ]
    if readings.get("slices"):
        cells.append(md("**Reading.** " + readings["slices"]))
    cells += [
        md("## 6. Robustness"),
        code("""rows = []
for c in ROBUST.get("checks") or []:
    e = c.get("primary") or {}
    rows.append((c.get("check"), c.get("status") or "measured", e.get("value"),
                 str(e.get("ci")) if e.get("ci") else "·",
                 c.get("why") or c.get("shift") or c.get("source") or ""))
pd.DataFrame(rows, columns=["check", "status", primary, "CI", "note"])"""),
    ]
    if readings.get("robustness"):
        cells.append(md("**Reading.** " + readings["robustness"]))
    return cells


def eval_baseline_cells(readings):
    cells = [
        md("## 7. The baselines and the verdict\n\n"
           "Every baseline was fitted on the train part and selected on valid, on exactly the "
           "shaped features the network saw, then scored on the same test rows. The verdict is "
           "read off the paired difference against the strongest of them: a network that does "
           "not beat it by an interval excluding zero has not earned its cost, whatever the "
           "point estimates look like side by side."),
        code("""rows = []
for b in BASELINES.get("baselines") or []:
    v, t = b.get("valid") or {}, b.get("test") or {}
    rows.append((b["name"], b.get("fitted"), v.get("value"), t.get("value"),
                 str(t.get("ci")) if t.get("ci") else ("upstream" if v.get("_upstream") else "·"),
                 b.get("note") or b.get("why") or b.get("definition")))
display(pd.DataFrame(rows, columns=["baseline", "fitted", f"valid {primary}",
                                    f"test {primary}", "test CI", "what it is"]))

V = METRICS["verdict"]
net = (METRICS["models"]["network"].get(primary) or {})
print()
print(f"network {primary}: {net.get('value')} {net.get('ci')}")
if V.get("earned") is None:
    # eval-contract §5: no paired difference on a primary each model reads on its own scale
    print(f"VERDICT: not assessable — {V.get('reason')}")
else:
    print(f"strongest fitted baseline: {V.get('strongest_baseline')}")
    print(f"paired difference: {V.get('diff')} {V.get('ci')}")
    print(f"VERDICT: {'earned' if V.get('earned') else 'NOT earned'}")"""),
    ]
    for key, title in (("verdict", "Verdict"), ("calibration", "Calibration"),
                       ("limits", "What this evaluation cannot say")):
        if readings.get(key):
            cells.append(md(f"### {title}\n\n{readings[key]}"))
    return cells


def eval_assertion_cells(package, layout):
    doors = [f"{package}.{layout.module('DataSource')}"]
    if layout.directory("DataSource") not in ("", layout.directory("Model")):
        doors.append(f"{package}.{layout.directory('DataSource')}")
    return [
        md("## 8. The notebook and the layer agree — and the seal held\n\n"
           "Two claims are checked here. The first is the usual one: every number the layer "
           "publishes is a number this notebook recomputed from the stored predictions. The "
           "second is particular to this stage — that running the notebook did not open the "
           "test part. It is checked, not asserted: if the data adapter is not in "
           "`sys.modules`, the one door to the test part was never even imported."),
        code(f"""DOORS = {tuple(doors)!r}

def assert_matches():
    checks = []
    def check(name, got, want):
        checks.append((name, got, want, got == want))

    check("every point metric agrees with metrics.json", point_gaps, [])
    check("models the record cannot reproduce are named, not compared", sorted(not_reproducible), sorted(not_reproducible))
    check("every interval agrees with metrics.json", ci_gaps, [])
    check("the ledger's valid number is marked upstream, not re-measured",
          ((METRICS.get("upstream") or {{}}).get("valid") or {{}}).get(primary, {{}})
          .get("_upstream"), True)
    check("the verdict names a fitted baseline",
          METRICS["verdict"].get("strongest_baseline") in
          [b["name"] for b in BASELINES["baselines"] if b.get("fitted")] or
          METRICS["verdict"].get("strongest_baseline") is None, True)
    ci = METRICS["verdict"].get("ci") or []
    earned = bool(ci) and (ci[0] > 0 if mode == "max" else ci[1] < 0)
    check("the verdict follows from its own interval", earned,
          bool(METRICS["verdict"].get("earned")))
    if threshold is not None:
        check("the threshold was not chosen on the test part",
              "test" in str(THRESH.get("source", "")).lower(), False)
    check("the predictions follow the layout this notebook reads",
          METRICS.get("predictions_layout"), "vega/predictions-layout@1")
    if LAYOUT.get("vega.task_effective"):
        check("the task the predictions were stored under is the one the readout reads",
              LAYOUT["vega.task_effective"], KIND)

    # the one door to the test part is the DataSource adapter's module (and the module the
    # blueprint gave it, when that is not the Model's, whose metrics this notebook reads)
    reopened = sorted(m for m in list(sys.modules) for door in DOORS
                      if m == door or m.startswith(door + "."))
    check("the test part was not reopened by this notebook", reopened, [])

    w = max(len(c[0]) for c in checks)
    for name, got, want, passed in checks:
        print(f"{{'ok  ' if passed else 'FAIL'}}  {{name:<{{w}}}}  notebook {{got!r}}  "
              f"record {{want!r}}")
    print()
    print("VEGA-EVAL: reopened" if reopened else "VEGA-EVAL: sealed")
    failed = [c for c in checks if not c[3]]
    if failed:
        raise AssertionError(f"{{len(failed)}} of {{len(checks)}} checks disagree with the "
                             f"records: {{[c[0] for c in failed]}}")
    print(f"{{len(checks)}} checks: the notebook's numbers are the evaluation's numbers, and "
          f"the seal held — exposure {{seal.get('exposures')}} is still the whole story")

assert_matches()"""),
    ]


def eval_notebook(camp_dir, seal, metrics, readings, package, layout, run_id):
    cells = [eval_header(seal, metrics, camp_dir, run_id),
             md("## 0. Environment\n\nThe emitted package is imported from the campaign's "
                "own directory — the same code the evaluation used — but only its metrics, "
                "its readout and its domain values: nothing that builds a network or opens a "
                "part."),
             eval_environment_cell(package, layout, run_id)]
    cells += eval_predictions_cells()
    cells += eval_metric_cells()
    cells += eval_bootstrap_cells()
    cells += eval_curve_cells()
    cells += eval_slice_cells(readings)
    cells += eval_baseline_cells(readings)
    cells += eval_assertion_cells(package, layout)
    return notebook(cells)



# --------------------------------------------------------------------------- the pack stage
#
# `predict.ipynb` is the model card in executable form: the same seven sections, drawn from the
# same records, saying the same things. Where the card renders a figure, the notebook computes
# it in front of you.
#
# The notebook is deliberately split in two halves, and it says so:
#
#   *what it explains* — identity, the recipe, the architecture's decisions, the training
#   ledger, the evaluation — reads `records/*.json`, which travel inside the bundle. Those
#   cells need `json` and `pandas` and nothing else, so the card's numbers can be audited on a
#   machine with no ML stack installed.
#
#   *what it verifies* — building the network to count its parameters, predicting, the round
#   trip, the digests — imports the bundle's own package, because a notebook that reimplemented
#   the forward pass would be testing a copy of the artefact rather than the artefact, and
#   would keep printing yesterday's numbers after someone edited the code.

def pack_header(manifest, bundle_dir):
    chain = (manifest.get("chain") or {}).get("evaluation") or {}
    verdict = chain.get("verdict") or {}
    primary = chain.get("primary") or {}
    trip = manifest.get("roundtrip") or {}
    stances = manifest.get("stances") or []
    if verdict.get("earned"):
        flag = "**Earned.**"
        line = (f"The network beat `{verdict.get('strongest_baseline')}` on the held-out part "
                "by a difference whose interval clears zero.")
    elif verdict.get("beaten_by_baseline"):
        flag = "**Not earned: beaten by a baseline.**"
        line = (f"On the held-out part `{verdict.get('strongest_baseline')}` scored better "
                "than the network, and the interval on that difference does not cover zero. "
                "This is not an undecided comparison. The bundle ships the network anyway, "
                "under the stance below.")
    else:
        flag = "**Not earned.**"
        line = (f"The interval on the difference against `{verdict.get('strongest_baseline')}` "
                "covers zero: this part cannot separate them.")
    stance_lines = "\n".join(f"> - {t}" for t in stances)
    return md(f"""# {manifest.get('slug')} — the model card, in code

| | |
|---|---|
| bundle | `{bundle_dir}` |
| run | **{manifest.get('run')}** |
| campaign | `{manifest.get('campaign')}` |
| task | `{(manifest.get('output') or {}).get('kind')}` |
| {primary.get('name')} on the held-out part | {primary.get('value')} {primary.get('ci')} |
| exposures of the test part | {chain.get('seal_exposures')} |
| round trip | {trip.get('rows')} row(s), worst \\|Δ\\| {trip.get('max_abs_diff')} |
| backend | `{(manifest.get('backend') or {}).get('name')} \
{(manifest.get('backend') or {}).get('version')}` |

## {flag}

{line}

{"> It ships under a recorded stance:" if stances else ""}
{stance_lines}

---

This notebook is `model-card.html` with the arithmetic left in. It has the card's seven
sections, reads the card's records, and reaches the card's conclusions — but where the card
shows you a number, this shows you the expression that produced it.

It is in two halves, and which half a cell belongs to matters:

- **What it explains** — sections 1 to 5 — reads `records/*.json`, the chain's own layer data,
  copied into this bundle by `/vega-pack`. Those cells import `json` and `pandas` and nothing
  else. They would still run in a bundle whose code was broken, because they are reading what
  the chain recorded rather than asking the model anything.
- **What it verifies** — sections 6 and 7 — imports the bundle's package and runs it. Those
  cells are a test: if someone edits `preprocess.py` or swaps `weights.pt`, they fail. That is
  the whole reason they import rather than describe.

*{GENERATED_BY}*""")


def pack_records_cell():
    return code("""import json
from pathlib import Path

import pandas as pd

HERE = Path.cwd()
REC = HERE / "records"

MANIFEST = json.loads((HERE / "MANIFEST.json").read_text())
SCHEMA = json.loads((HERE / "schema.json").read_text())

def record(name):
    \"\"\"One of the chain's layers, as /vega-pack copied it into this bundle.

    A missing layer is not an error: a bundle packed without --model carries no records, and
    the sections below say so instead of failing.\"\"\"
    path = REC / f"{name}.json"
    return json.loads(path.read_text()) if path.is_file() else {}

GEOMETRY, ANALYSIS, SHAPE = record("geometry"), record("analysis"), record("shape")
ARCH, TRAIN, EVAL = record("architecture"), record("training"), record("evaluation")

have = [n for n in ("geometry", "analysis", "shape", "architecture", "training", "evaluation")
        if record(n)]
print(f"{MANIFEST['slug']}-{MANIFEST['run']}  —  {len(MANIFEST['files'])} file(s) under digest")
print(f"records present: {', '.join(have) or 'none (this bundle was packed without --model)'}")
print(f"pandas {pd.__version__}; nothing else is needed until section 6")""")


def pack_identity_cells():
    return [
        md("## 1. Identity\n\n"
           "Where this network came from, stage by stage. Every row is a digest the chain "
           "recorded, and every one of them can be re-checked from inside this bundle — which "
           "is what `tests/test_roundtrip.py` does. A bundle whose weights were swapped would "
           "still reproduce *its own* new answers perfectly; the digests are what make the "
           "reproduction mean something."),
        code("""chain = MANIFEST["chain"]
rows = [
    ("the model page the bundle was cut from", chain.get("model"), chain.get("model_digest")),
    ("the resolved netspec", "netspec.json",
     (chain.get("architecture") or {}).get("netspec_digest")),
    ("the shaped manifest", "shape.manifest", (chain.get("shape") or {}).get("manifest_digest")),
    ("the recipe", "recipe.json", (chain.get("shape") or {}).get("recipe_digest")),
]
for name, digest in sorted(MANIFEST["files"].items()):
    if name.startswith(("weights.", "records/")) or name in ("netspec.json", "recipe.json",
                                                             "fitted.json"):
        rows.append(("carried in the bundle", name, digest))

display(pd.DataFrame(rows, columns=["what", "where", "digest"]))
print(f"\\npacked {MANIFEST['created_at']} by /vega-pack, python {MANIFEST['python']}, "
      f"{MANIFEST['backend']['name']} {MANIFEST['backend']['version']}")

# which of the two bundles this is. `full` carries the training loop and the evaluator, so a
# reader who finds train.py beside predict.py should know from here that a checkpoint from it
# is not the network the sections below describe.
shape = MANIFEST.get("include", "serving")
entries = ", ".join(MANIFEST.get("entrypoints") or ["predict.py"])
print(f"shape: {shape} — entry point(s) {entries}")
if shape == "full":
    print("      the whole emitted package ships. train.py fits one spec (no ladder, no "
          "diagnostics,\\n      no ledger) and evaluate.py prints point metrics (no intervals, "
          "no baselines, no\\n      verdict); neither produces what sections 4 and 5 report, "
          "and the test part stays\\n      sealed in both — load_part refuses it and the one "
          "door wants the campaign's seal.")"""),
    ]


def pack_task_cells(transform_module):
    return [
        md("## 2. Task and data\n\n"
           "The label and the partition come from the geometry layer; the steps that turned "
           "raw rows into the columns the network reads come from the shape layer. Neither is "
           "rewritten here — the recipe below is the recipe that ran, and "
           f"`{transform_module}` in this bundle is that same recipe emitted for inference, "
           "which is why the round trip in section 7 is a fact about the data and not about "
           "the code."),
        code("""man = SHAPE.get("manifest") or {}
label = man.get("label")
partitions = (GEOMETRY.get("partitions") or {}).get("candidates") or []
chosen = next((p for p in partitions if p.get("label") == label), {})

print(f"label: {label}   task: {chosen.get('task') or MANIFEST['output']['kind']}")
if chosen.get("dropped_for_leakage"):
    print(f"dropped as leakage: {', '.join(chosen['dropped_for_leakage'])}")
print()

steps = []
for step in man.get("steps") or []:
    after = step.get("rows_after") or {}
    steps.append((step.get("id"), step.get("op"), step.get("source"), step.get("fit_on") or "—",
                  ", ".join(f"{k} {v}" for k, v in after.items()) if after else "—",
                  ", ".join(step.get("columns_removed") or []) or "—"))
display(pd.DataFrame(steps, columns=["step", "op", "source", "fitted on", "rows after",
                                     "columns removed"]))"""),
        md("### What a caller has to send\n\n"
           "Derived from the recipe's *input* — the columns the shaper consumed before it did "
           "anything — not from the shaped columns the network sees. The ranges and "
           "vocabularies below are what the training data covered; a value outside them is "
           "answered and flagged, never refused, because a model asked about an unusual row "
           "still answers and the caller is the one who decides whether to trust it."),
        code("""rows = []
for column in SCHEMA["input"]:
    domain = column.get("allowed")
    if domain is None:
        seen = column.get("range_seen")
        domain = f"{seen[0]} … {seen[1]}" if seen else "any"
    elif isinstance(domain, list):
        domain = (", ".join(map(str, domain[:6]))
                  + (f" … ({len(domain)} in all)" if len(domain) > 6 else ""))
    rows.append((column["name"], column.get("dtype"), domain))

print("ignored by the recipe:", ", ".join(SCHEMA.get("ignored") or []) or "nothing")
for name, meaning in (SCHEMA.get("output") or {}).items():
    print(f"  {name}: {meaning}")
display(pd.DataFrame(rows, columns=["column", "dtype", "seen in training"]))"""),
        md("### What the analysis found in this data\n\n"
           "These findings are why the architecture looks the way it does. Section 3's "
           "decisions cite them by id."),
        code("""findings = [(f.get("id"), f.get("title"), f.get("detail") or f.get("evidence") or "—")
            for f in (ANALYSIS.get("findings") or [])]
display(pd.DataFrame(findings, columns=["finding", "title", "detail"])
        if findings else "the analysis layer records no findings")"""),
    ]


def pack_architecture_cells():
    return [
        md("## 3. Architecture\n\n"
           "Every choice in the network, with what justified it. This is the table the card "
           "prints, and the `source` column is the point of it: `analysis:F5` means a finding "
           "in the data chose this, `vega:default` means the catalog did and nobody "
           "disagreed, `user:` means a person overrode it. A row with no source would be a "
           "choice nobody made, and the compiler refuses those."),
        code("""spec = ARCH.get("netspec") or {}

def decision(family, what, node):
    if not isinstance(node, dict):
        return (family, what, node, "—", "—")
    name = node.get("name") or node.get("id") or node.get("kind") or node.get("scheme")
    params = {k: v for k, v in (node.get("params") or {}).items()}
    shown = f"{name}" + (f" ({', '.join(f'{k} {v}' for k, v in params.items())})"
                         if params else "")
    return (family, what, shown, node.get("source") or "—", node.get("rationale") or "—")

rows = [decision("architecture", "family", spec.get("family"))]
for block in spec.get("blocks") or []:
    rows.append(decision("architecture", "blocks", block))
for family, what in (("objective", "loss"), ("objective", "metrics"),
                     ("optimization", "optimizer"), ("optimization", "schedule"),
                     ("optimization", "init"), ("regularization", "early_stopping")):
    node = spec.get(what)
    if isinstance(node, dict):
        rows.append(decision(family, what, node))
for reg in spec.get("regularization") or []:
    rows.append(decision("regularization", "regularization", reg))

display(pd.DataFrame(rows, columns=["family", "what", "chosen", "source", "why"]))"""),
        md("### The loss and the update rule, written out\n\n"
           "From the architecture layer's mechanics reading — the same strings the card "
           "prints. They are here rather than in the code because a reader checking whether "
           "the objective matches the task should not have to read a framework's source."),
        code("""mech = (ARCH.get("readings") or {}).get("mechanics")
loss = (ARCH.get("netspec") or {}).get("loss") or {}
print("loss:", loss.get("name"), loss.get("params") or "")
smoke = ARCH.get("smoke") or {}
for k in ("param_count", "trainable", "madds_per_sample", "out_shape"):
    if smoke.get(k) is not None:
        print(f"{k}: {smoke[k]}")
print()
print(mech or "the architecture layer carries no mechanics reading")"""),
    ]


def pack_training_cells():
    return [
        md("## 4. Training\n\n"
           "The ladder the campaign ran, one row per experiment. `delta` is what that run "
           "changed against the spec as engineered — one change per rung, so a difference in "
           "the result has one candidate explanation rather than several. The selected run is "
           "the one this bundle carries."),
        code("""exp = TRAIN.get("experiments") or {}
selection = exp.get("selection") or {}
primary_name = ((ARCH.get("netspec") or {}).get("metrics") or {}).get("primary", "primary")

rows = []
for run in exp.get("runs") or []:
    rows.append((run.get("id"), run.get("kind"), (run.get("delta_text") or "—")[:52],
                 run.get("primary"), run.get("epochs"), run.get("best_epoch"),
                 run.get("stopped_by"),
                 "← selected" if run.get("id") == selection.get("run") else ""))
display(pd.DataFrame(rows, columns=["run", "kind", "delta", f"valid {primary_name}", "epochs",
                                    "best epoch", "stopped by", ""]))
print(f"\\nselected {selection.get('run')} by {selection.get('rule')}")
if selection.get("margin") is not None:
    print(f"margin over the campaign baseline: {selection['margin']}")
if selection.get("override"):
    print(f"OVERRIDE: {selection['override']}")"""),
        md("### The selected run, epoch by epoch\n\n"
           "The curve the card draws. What to look at is not the final value but where the "
           "valid metric stopped improving: the checkpoint this bundle carries is the one from "
           "the epoch marked below, and the ledger's number above is that epoch's — not the "
           "best number the run ever printed, which would belong to weights nobody kept."),
        code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The record stores a history column-wise: `epochs`, then one series per channel under a
# dotted name (`train.loss`, `valid.pr_auc`, `lr`, `grad_norm.<block>`). That is the shape
# the training layer wrote and the shape the card reads; nothing is recomputed here.
sel = selection.get("run")
history = (TRAIN.get("histories") or {}).get(sel) or {}
ep = history.get("epochs") or []
if not ep:
    print(f"no history was recorded for {sel}")
else:
    tr = history.get("train.loss")
    va = history.get("valid.loss")
    pm = history.get(f"valid.{primary_name}")
    best = next((r.get("best_epoch") for r in exp.get("runs") or []
                 if r.get("id") == sel), None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    if tr: axes[0].plot(ep, tr, label="train", color="#1b3a73")
    if va: axes[0].plot(ep, va, label="valid", color="#9a3a2c")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    if pm: axes[1].plot(ep, pm, color="#1b3a73")
    axes[1].set_title(f"valid {primary_name}"); axes[1].set_xlabel("epoch")
    for ax in axes:
        if best:
            ax.axvline(best, color="#9a3a2c", ls="--", lw=1)
    fig.tight_layout(); display(fig); plt.close(fig)

    stored, whole = len(ep), history.get("downsampled_from")
    if not whole or whole == stored:
        span = f"{stored} epoch(s)"
    else:
        span = f"{stored} of {whole} epoch(s), thinned for the record"
    print(f"{span}; the checkpoint is epoch {best} (dashed)")
    if pm and best and 1 <= best <= len(pm):
        print(f"valid {primary_name} at the checkpoint: {pm[best - 1]:.4f}; "
              f"its last value was {pm[-1]:.4f}")"""),
        md("### What the diagnostics said\n\n"
           "The catalog's checks over the runs. A diagnostic that fired is not a failure — it "
           "is the campaign noticing something and, where a rung followed, saying which one."),
        code("""# A diagnostic is recorded against the run it was computed on, so that is where this
# reads it from — the same per-run `diagnostics_detail` the card's ladder column counts.
fired, deferred = [], []
for run in exp.get("runs") or []:
    for d in run.get("diagnostics_detail") or []:
        row = (run.get("id"), d.get("code"), d.get("name"),
               (d.get("evidence") or d.get("why") or "—"),
               ", ".join(d.get("remedies") or []) or "—")
        (fired if d.get("fired") else deferred).append(row)

cols = ["run", "code", "name", "evidence", "remedies offered"]
if fired:
    display(pd.DataFrame(fired, columns=cols))
    print(f"{len(fired)} firing(s) across {len({r[0] for r in fired})} run(s)")
else:
    print("no diagnostic fired in this campaign")
if deferred:
    print(f"\\n{len(deferred)} check(s) could not be evaluated:")
    for run_id, code, name, why, _ in deferred:
        print(f"  {run_id}  {code} {name} — {why}")"""),
    ]


def pack_evaluation_cells():
    return [
        md("## 5. Evaluation\n\n"
           "The one look at the held-out part, under a seal that records how many times it was "
           "opened. Everything below was measured in that single look and stored; nothing here "
           "reopens the test part, and re-running this notebook costs nothing against the "
           "seal."),
        code("""seal = EVAL.get("seal") or {}
test = seal.get("test") or {}
print(f"run {seal.get('run')}   sealed {seal.get('opened_at')}   exposures {seal.get('exposures')}")
matched = test.get("digest_matches_manifest")
print("digest against the shaper's manifest:",
      {True: "matched", False: "MISMATCH", None: "not checked (the manifest recorded none)"}[matched])
for look in seal.get("history") or []:
    print(f"  - {look.get('at')}  {look.get('gate')}: {look.get('reason')}")"""),
        md("### Every metric, with the interval that says how much to trust it\n\n"
           "A point estimate on a held-out part of this size is a number without a scale, so "
           "each carries a bootstrap interval drawn at the seed the record names. Every model "
           "was scored on the **same** resamples — which is what makes the paired differences "
           "below have intervals of their own, rather than two separate intervals whose "
           "overlap says nothing."),
        code("""metrics = EVAL.get("metrics") or {}
models = metrics.get("models") or {}
primary = metrics.get("primary")

def cell(entry):
    if not isinstance(entry, dict) or entry.get("value") is None:
        return "·" if not (entry or {}).get("values") else f"{len(entry['values'])} values"
    ci = entry.get("ci")
    return f"{entry['value']}" + (f"  [{ci[0]}, {ci[1]}]" if ci else
                                  ("  (upstream)" if entry.get("_upstream") else ""))

names = list(dict.fromkeys(k for m in models.values() for k in m
                           if not str(k).endswith("_values")))
names.sort(key=lambda n: (n != primary, n))
display(pd.DataFrame([[cell(models[m].get(n)) for m in models] for n in names],
                     index=names, columns=list(models)))
print(f"\\n{metrics.get('n_test')} rows, {metrics.get('n_boot')} resamples at seed "
      f"{metrics.get('boot_seed')}")"""),
        md("### The baselines, and what the network is worth against them\n\n"
           "Each baseline was fitted on the train part, selected on valid, on exactly the "
           "shaped features the network saw — a baseline with an advantage or a handicap is "
           "not a baseline. The difference is the comparison the evaluation is entitled to "
           "make, and its interval is where the verdict comes from."),
        code("""base = [(b.get("name"), b.get("fitted"), cell(b.get("valid")), cell(b.get("test")),
         (b.get("note") or b.get("why") or b.get("definition") or "")[:64])
        for b in (EVAL.get("baselines") or {}).get("baselines") or []]
display(pd.DataFrame(base, columns=["baseline", "fitted", "valid", "test", "what it is"]))

diffs = []
mode = metrics.get("mode", "max")
for name, d in (metrics.get("differences") or {}).items():
    ci = d.get("ci") or []
    if len(ci) == 2 and ci[0] is not None:
        reads = ("the network is better" if (ci[0] > 0 if mode == "max" else ci[1] < 0)
                 else "the baseline is better" if (ci[1] < 0 if mode == "max" else ci[0] > 0)
                 else "the interval covers zero — no difference is shown")
    else:
        reads = "no interval, so no comparison"
    diffs.append((name, d.get("value"), str(ci), reads))
display(pd.DataFrame(diffs, columns=["baseline", f"network − baseline ({primary})",
                                     "interval", "reads as"]))"""),
        md("### Calibration, and where the network is worst\n\n"
           "The reliability curve is the one to read slowly: above the diagonal the network is "
           "under-confident, below it over-confident. The slices matter more than the "
           "whole-part average whenever they disagree with each other — an average over two "
           "regimes is a description of neither."),
        code("""curves = (EVAL.get("curves") or {}).get("curves") or EVAL.get("curves") or {}

def points_of(curve):
    if not curve:
        return None
    if isinstance(curve.get("points"), list) and curve["points"]:
        return [p[0] for p in curve["points"]], [p[1] for p in curve["points"]]
    if isinstance(curve.get("x"), list):
        return curve["x"], curve["y"]
    return None

panels = [(k, points_of(curves.get(k))) for k in ("roc", "pr", "reliability")]
panels = [(k, v) for k, v in panels if v]
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.4))
    import numpy as _np
    axes = _np.atleast_1d(axes)
    for ax, (name, (xs, ys)) in zip(axes, panels):
        ax.plot(xs, ys, lw=1.6)
        if name in ("roc", "reliability"):
            ax.plot([0, 1], [0, 1], ls="--", lw=.8, color="grey")
        ax.set_title(name); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); display(fig); plt.close(fig)

sl = EVAL.get("slices") or {}
rows = []
for s in sl.get("slices") or []:
    e = (s.get("metrics") or {}).get(primary) or {}
    rows.append((s.get("name"), s.get("n"), cell(e), s.get("source"),
                 "single class" if s.get("degenerate") else (s.get("note") or "")[:48]))
display(pd.DataFrame(rows, columns=["slice", "n", primary, "source", "note"]))
if sl.get("worst"):
    print(f"\\nworst on {primary}: {sl['worst']['name']} at {sl['worst']['value']} "
          f"(n = {sl['worst']['n']})")"""),
        md("### The verdict\n\n"
           "Read off the interval on the paired difference, not off two point estimates side "
           "by side. The evaluation's own words follow the machine record — the card prints "
           "the same paragraph, and neither was written after seeing what the other said."),
        code("""V = EVAL.get("verdict") or metrics.get("verdict") or {}
print(f"earned: {V.get('earned')}")
print(f"strongest fitted baseline: {V.get('strongest_baseline')}")
print(f"paired difference: {V.get('diff')}  {V.get('ci')}")
if V.get("beaten_by_baseline"):
    print("\\nthe interval excludes zero in the BASELINE's favour: this part says the baseline "
          "is better, not that it cannot tell them apart")
print()
for key in ("verdict", "calibration", "slices", "robustness", "limits"):
    text = (EVAL.get("readings") or {}).get(key)
    if text:
        print(f"--- {key} " + "-" * (66 - len(key)))
        print(text)
        print()"""),
    ]


def _predict_cell(window):
    """Section 7's first cell: load the bundle through its own entry point and predict the
    example rows (a window bundle also says how many rows had a full window behind them)."""
    answered = ""
    if window:
        answered = '''
if "answered" in out.columns:
    print(f"{int(out['answered'].sum())} row(s) answered; the rest have fewer than "
          f"{MANIFEST['sample'].get('length')} - 1 rows before them — the first rows of the "
          "file are the windows' context")'''
    return '''import predict                    # the bundle's own entry point, not the chain's

predictor = predict.load(HERE)
raw = pd.read_csv(HERE / "examples" / "rows.csv")
out = predictor.predict(raw)

def warned(value):
    return bool(value) if isinstance(value, (str, list, tuple)) else False

print(f"{len(raw)} row(s) in, {len(out)} row(s) out — one answer per row given, in order")
if "warnings" in out.columns:
    print(f"{sum(warned(w) for w in out['warnings'])} row(s) carry a warning")''' + answered + '''
display(out.head(8))'''


def _unseen_cell(window):
    """Section 7's second cell: one row as sent and as answered, then the same row with a
    category nobody saw — on a window bundle with the `history_rows` rows before it as its
    history (R-D16), so the single answer is the fully determined one."""
    if window:
        send = '''HISTORY_ROWS = int(MANIFEST["sample"]["history_rows"])   # (length - 1) + lag depth
at = min(HISTORY_ROWS, len(raw) - 1)
history = raw.iloc[max(0, at - HISTORY_ROWS):at].reset_index(drop=True)
example = raw.iloc[at].to_dict()
print(f"history: the {len(history)} row(s) before row {at} "
      f"(MANIFEST.sample.history_rows = {HISTORY_ROWS})")

def ask(row):
    return predictor.predict_one(row, history=history)
'''
    else:
        send = '''example = raw.iloc[0].to_dict()

def ask(row):
    return predictor.predict_one(row)
'''
    return send + '''
print("as sent:    ", json.dumps(example, default=str)[:180])
print("as answered:", json.dumps(ask(example), default=str))

strange = dict(example)
for column in SCHEMA["input"]:
    if isinstance(column.get("allowed"), list):
        strange[column["name"]] = "a-category-that-never-occurred"
        break
print()
print("unseen:     ", json.dumps(ask(strange), default=str))'''


_EQUALITY_CELL = '''check = predict.check_examples(HERE)      # the bundle's own comparison (D2 §9)
kind = MANIFEST["examples"]["kind"]

def shown(value):
    return "n/a" if value is None else f"{float(value):.3g}"

print(f"examples are {kind} rows from {MANIFEST['examples'].get('source')}")
print(f"{check.get('rows')} compared row(s): worst |delta| {shown(check.get('max_abs_diff'))}, "
      f"tolerance {shown(check.get('tolerance'))}")
print("decided fields equal the expected ones:", check.get("decisions_match"))
print("evidence:", check.get("evidence"))
print("this proves the bundle reproduces " + ("the evaluation" if kind == "test" else "itself"))
trip = MANIFEST.get("roundtrip") or {}
print(f"\\nat pack time: {trip.get('rows')} row(s), worst |delta| {trip.get('max_abs_diff')}, "
      f"status {trip.get('status')}")
reproduced = bool(check["passed"])'''


def pack_usage_cells(readings, window=False):
    cells = [
        md("## 6. Intended use and limits\n\n"
           "The one passage on the card written by a person rather than spliced from a record "
           "— and the card refuses to render if it contains a digit, so that every number a "
           "reader sees traces back to something that measured it."),
    ]
    if readings.get("intended_use"):
        cells.append(md("> " + readings["intended_use"].replace("\n\n", "\n>\n> ")))
    if readings.get("abstract"):
        cells.append(md("**What ships, and under what.** " + readings["abstract"]))
    history_note = (
        "\n\nThis bundle answers **windows**: a row is answered from the rows before it, so "
        "the single row is sent with the `MANIFEST.sample.history_rows` rows that precede it "
        "in `examples/rows.csv` as its history — enough for the window and for every lag the "
        "recipe computes, which is what makes that answer fully determined.") if window else ""
    cells += [
        md("## 7. How to use it — and the proof that it still works\n\n"
           "From here the notebook stops reading records and starts running the bundle. "
           "Everything below imports `predict.py`, this bundle's composition root, and would "
           "fail if the code or the weights had changed since it was packed. That is the "
           "point of it: the sections above describe, this one tests."),
        code(_predict_cell(window)),
        md("### A row unlike anything in the training data\n\n"
           "An unseen category is a defined state, not an error: the shaper reserved a code for "
           "exactly this and the transform lands on it, so the model answers — and the row says "
           "so, because that answer is extrapolation and nothing measured it." + history_note),
        code(_unseen_cell(window)),
        md("### It still reproduces itself\n\n"
           "At pack time the round trip ran over **every** held-out row, from raw input, "
           "against the scores the evaluation stored — that is the proof that *raw rows → "
           "preprocess → network* here equals *shaped rows → network* there. It does not "
           "travel, because the test part does not travel. What travels is this: the same "
           "path over the example rows, against the expectations packed beside them — "
           "compared by the bundle's own `predict.check_examples`, the one comparison its "
           "`--check`, its round-trip test and the service's warm-up also run: every "
           "`MANIFEST.output.compared` field at the round trip's tolerance and every "
           "`decided` field exactly, after the files are authenticated against "
           "`MANIFEST.files`.\n\n"
           "`examples.kind` says what that proves. `test` rows carry the evaluation's own "
           "scores, so reproducing them reproduces the evaluation. `synthetic` rows carry the "
           "predictor's outputs at pack time, so reproducing them proves only that the bundle "
           "agrees with itself — weaker, and the manifest says which claim is being made."),
        code(_EQUALITY_CELL),
        md("### And every file is the file that was packed\n\n"
           "Reproducing its own examples would mean nothing if the weights had been swapped — "
           "a different network reproduces its own new answers perfectly. The digests are what "
           "make the check above worth running."),
        code("""import hashlib

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

bad = {name: recorded for name, recorded in MANIFEST["files"].items()
       if not (HERE / name).is_file() or digest(HERE / name) != recorded}
print(f"{len(MANIFEST['files'])} file(s) under digest, {len(bad)} changed")
for name in sorted(bad):
    print("  CHANGED:", name)
files_intact = not bad"""),
        md("### Calling it from anywhere else"),
        code("""usage = MANIFEST["usage"]
print("# from a shell");  print(usage["cli"]);  print()
print("# from python");   print(usage["module"]); print()
if usage.get("train") or usage.get("evaluate"):
    print("# fitting and scoring — this bundle carries the whole package")
    for line in (usage.get("train"), usage.get("evaluate")):
        if line:
            print(line)
    print("# both want shaped parts, which do not travel; recipe.json is what makes them")
    print()
serve = MANIFEST.get("serve") or {}
if serve.get("present"):
    print(f"# over HTTP ({serve.get('framework')}): {serve.get('endpoint')}")
else:
    print(f"# no serving scaffold: the blueprint's mode is {serve.get('mode')!r}, and for that "
          "mode `predict.py --input` is the interface")
for export in MANIFEST.get("exports") or []:
    print(f"# {export['format']}: {export['file']} (parity {export['status']})")
if not (MANIFEST.get("exports") or []):
    print("# no exports were built for this bundle")"""),
    ]
    return cells


def pack_assertion_cells():
    return [
        md("## 8. The notebook, the card and the bundle agree\n\n"
           "Three documents describe this network — `model-card.html`, this notebook, and "
           "`MANIFEST.json` — and they are only worth having if they cannot disagree. This "
           "cell checks that they do not, and prints the marker the apply script reads, so "
           "whether the bundle reproduced itself is a recorded fact rather than a sentence "
           "someone wrote."),
        code("""def assert_matches():
    checks = []
    def check(name, got, want):
        checks.append((name, got, want, got == want))

    check("the predictor returned one row per row given", len(out), len(raw))
    check("every file hashes to what the manifest records", files_intact, True)
    check("the bundle reproduces its expectations", reproduced, True)
    check("the pack-time round trip passed", MANIFEST["roundtrip"]["status"], "pass")
    check("the shaper's twin check passed",
          MANIFEST["roundtrip"].get("shaper_transform_check"), "pass")
    check("the mechanical conformance gate passed",
          (MANIFEST.get("blueprint") or {}).get("conformance", {}).get("mechanical"), "pass")
    # the records the sections above read are the ones the card read
    check("the run this notebook documents is the run the seal recorded",
          (EVAL.get("seal") or {}).get("run"), MANIFEST["run"])
    check("the verdict here is the verdict the manifest carries",
          (EVAL.get("verdict") or {}).get("earned"),
          ((MANIFEST.get("chain") or {}).get("evaluation") or {}).get("verdict", {})
          .get("earned"))
    check("the primary metric here is the one the manifest carries",
          ((EVAL.get("metrics") or {}).get("models", {}).get("network", {})
           .get((EVAL.get("metrics") or {}).get("primary"), {}) or {}).get("value"),
          (((MANIFEST.get("chain") or {}).get("evaluation") or {}).get("primary") or {})
          .get("value"))
    verdict = ((MANIFEST.get("chain") or {}).get("evaluation") or {}).get("verdict") or {}
    if not verdict.get("earned"):
        check("an unearned network ships with a recorded stance",
              bool(MANIFEST.get("stances")), True)

    w = max(len(c[0]) for c in checks)
    for name, got, want, passed in checks:
        print(f"{'ok  ' if passed else 'FAIL'}  {name:<{w}}  {got!r}  vs  {want!r}")
    print()
    print("VEGA-PACK: reproduced" if reproduced else "VEGA-PACK: mismatch")
    failed = [c for c in checks if not c[3]]
    if failed:
        raise AssertionError(f"{len(failed)} of {len(checks)} checks disagree: "
                             f"{[c[0] for c in failed]}")
    print(f"{len(checks)} checks: this bundle is the one the manifest and the card describe")

assert_matches()"""),
    ]


def pack_notebook(bundle_dir, manifest, readings, package, transform_module):
    window = ((manifest.get("sample") or {}).get("shape") == "window")
    cells = [pack_header(manifest, bundle_dir),
             md("## 0. What this notebook reads\n\nThe chain's own records, copied into this "
                "bundle by `/vega-pack`. Sections 1 to 5 need nothing else."),
             pack_records_cell()]
    cells += pack_identity_cells()
    cells += pack_task_cells(transform_module)
    cells += pack_architecture_cells()
    cells += pack_training_cells()
    cells += pack_evaluation_cells()
    cells += pack_usage_cells(readings, window)
    cells += pack_assertion_cells()
    return notebook(cells)


# --------------------------------------------------------------------------- main

#: the generation whose adapter classes the cells call (BUILD-PLAN §2.12: v2.2 goes at gate A)
GENERATION = "v2.3"
#: the stored-predictions layout the eval notebook reads (BUILD-PLAN R-C2)
PREDICTIONS_LAYOUT = "vega/predictions-layout@1"

#: the ports whose adapters the notebooks import, and where the D1 §4.1 layout puts each file
#: when a package carries no blueprint map to read it from
_REQUIRED_ROLES = ("Model", "DataSource", "Trainer", "ModelStore")
_DEFAULT_FILES = {"Model": "model/network.py", "DataSource": "ingestion/data.py",
                  "Trainer": "training/loop.py", "ModelStore": "model/store.py",
                  "Evaluator": "evaluation/evaluate.py",
                  "FeatureTransformer": "features/preprocess.py"}


class Layout(NamedTuple):
    """Where each port's adapter lives in one emitted package: {role: package-relative file},
    as the package's `blueprint-map.json` records it. The notebooks' imports are the
    blueprint's module map, never a convention this script assumes (notebook-contract §1)."""

    package: str
    files: dict

    def module(self, role: str) -> str:
        """The adapter's module, dotted and package-relative: `model.network`."""
        return self.files[role][:-3].replace("/", ".")

    def directory(self, role: str) -> str:
        """The blueprint module holding the adapter, dotted: `model` ('' at the package root)."""
        return str(Path(self.files[role]).parent.as_posix()).replace("/", ".").strip(".")

    def support(self, role: str, stem: str) -> str:
        """A support file beside the role's adapter (D1 §4.1 placement rule): `training.optim`."""
        directory = self.directory(role)
        return f"{directory}.{stem}" if directory else stem


def package_layout(run_dir, package):
    """(Layout, warning, error) for the package `<run_dir>/<package>`.

    A package without a readable map is read in the D1 §4.1 layout, with a warning; a map that
    is not generation v2.3 is refused — its adapters are module-level functions, and a notebook
    written against the v2.3 classes would call names that package does not have.
    """
    path = Path(run_dir) / package / "blueprint-map.json"
    bmap, err = read_json(path, "blueprint-map.json")
    if err is not None:
        return (Layout(package, dict(_DEFAULT_FILES)),
                f"WARN: {err.message}; the notebook's imports fall back to the D1 §4.1 layout", None)
    generation = str(bmap.get("generation") or "v2.2")
    if generation != GENERATION:
        return None, None, (
            f"ERROR: {path} records emit generation {generation}: its adapters are module-level "
            f"functions, and nb_emit.py writes its cells against the {GENERATION} adapter "
            "classes (Network, ShapedParts, Trainer, ModelRegistry; D1 §4.2). Re-emit the "
            f"package with the {GENERATION} emitter (emit_backend.py) and rerun the stage — the "
            f"{generation} generation is removed at gate A (BUILD-PLAN §2.12)")
    files = {}
    for entry in bmap.get("entries") or []:
        role, file = entry.get("role"), entry.get("file")
        if role and isinstance(file, str) and file.endswith(".py") and role not in files:
            files[role] = file
    missing = [role for role in _REQUIRED_ROLES if role not in files]
    if missing:
        return None, None, (
            f"ERROR: {path} maps no file to the {', '.join(missing)} role(s); the notebook "
            "imports each port's adapter from the module its role names, and cannot guess it")
    for role in ("Evaluator", "FeatureTransformer"):
        files.setdefault(role, _DEFAULT_FILES[role])
    return Layout(package, files), None, None


def _layout_or_refuse(run_dir, package):
    """The layout, printing its warning; None (after printing the refusal) when refused."""
    layout, warning, refusal = package_layout(run_dir, package)
    if refusal:
        print(refusal)
        return None
    if warning:
        print(warning)
    return layout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--from", dest="src", required=True, help="the stage's run directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", default=None,
                    help="the authoring layer JSON whose readings the notebook repeats "
                         "verbatim (default: architecture-layer.json in the run directory)")
    ap.add_argument("--package", default=None,
                    help="the emitted package name (default: the spec's outputs.package)")
    ap.add_argument("--replay-epochs", type=int, default=3,
                    help="how many of the selected run's epochs the train notebook replays")
    ap.add_argument("--run", default=None,
                    help="which run the eval notebook documents (default: the seal's)")
    a = ap.parse_args(argv)

    if a.stage in NOT_YET:
        print(f"ERROR: nb_emit.py cannot build the {a.stage} notebook in this build: it is "
              f"generated from {NOT_YET[a.stage]}, and those records do not exist yet. "
              f"Emitting cells against records that are not there would be a notebook that "
              f"asserts nothing — run /vega-{a.stage} once its scripts are in place.")
        return 2

    run = Path(a.src)
    if a.stage == "train":
        return _emit_train(a, run)
    if a.stage == "eval":
        return _emit_eval(a, run)
    if a.stage == "pack":
        return _emit_pack(a, run)
    spec, err = read_json(run / "netspec.json", "netspec.json")
    if err is not None:
        print(err.line() + " — the forge notebook is generated from the run's own records")
        return 2
    smoke, err = read_json(run / "smoke.json", "smoke.json")
    if err is not None:
        print(err.line() + " — run emit_backend.py --smoke first")
        return 2
    if not smoke.get("forward_ok"):
        print("ERROR: smoke.json records forward_ok false; there is no network to document "
              "until the spec builds")
        return 2

    layer_path = Path(a.layer) if a.layer else (run / "architecture-layer.json")
    readings = {}
    if layer_path.is_file():
        layer, lerr = read_json(layer_path, "the layer document")
        if lerr is None:
            readings = layer.get("readings") or {}
            print(f"OK: readings taken verbatim from {layer_path} "
                  f"({', '.join(sorted(readings))})")
    else:
        print(f"WARN: {layer_path} not found; the notebook carries no readings — the layer "
              "and the notebook must say the same thing, so author the layer first "
              "(judgement → notebook → apply)")

    package = a.package or (spec.get("outputs") or {}).get("package")
    if not package:
        print("ERROR: no package name: pass --package, or let the spec's outputs.package "
              "name it")
        return 2
    layout = _layout_or_refuse(run, package)
    if layout is None:
        return 2

    nb = forge_notebook(str(run), spec, smoke, readings, package, layout)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"OK: wrote {out} — {len(nb['cells'])} cell(s), {n_code} code cell(s), unexecuted")
    print("OK: execute it in place with `jupyter nbconvert --to notebook --execute --inplace "
          f"--ExecutePreprocessor.timeout=600 {out}`, then pass it to "
          "apply_architecture_layer.py --notebook")
    return 0


def _emit_train(a, camp):
    """`--stage train`: the campaign's notebook, from the campaign's own records."""
    campaign, err = read_json(camp / "campaign.json", "campaign.json")
    if err is not None:
        print(err.line() + " — the campaign is planned by /vega-train's plan_ladder.py")
        return 2
    experiments, err = read_json(camp / "experiments.json", "experiments.json")
    if err is not None:
        print(err.line() + " — no run has been recorded yet")
        return 2
    if not (experiments.get("selection") or {}).get("run"):
        print("ERROR: experiments.json records no selection; the notebook replays the selected "
              "run, so run /vega-train's select.py first")
        return 2
    _, err = read_json(camp / "ladder.json", "ladder.json")
    if err is not None:
        print(err.line())
        return 2

    layer_path = Path(a.layer) if a.layer else (camp / "training-layer.json")
    readings = {}
    if layer_path.is_file():
        layer, lerr = read_json(layer_path, "the layer document")
        if lerr is None:
            readings = layer.get("readings") or {}
            print(f"OK: readings taken verbatim from {layer_path} "
                  f"({', '.join(sorted(readings))})")
    else:
        print(f"WARN: {layer_path} not found; the notebook carries no readings — the layer and "
              "the notebook must say the same thing, so author the layer first "
              "(judgement → notebook → apply)")

    package = a.package or (campaign.get("input") or {}).get("package")
    if not package:
        print("ERROR: no package name: pass --package, or let campaign.json name it")
        return 2
    layout = _layout_or_refuse(camp, package)
    if layout is None:
        return 2

    nb = train_notebook(str(camp), campaign, experiments, readings, package, layout,
                        a.replay_epochs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"OK: wrote {out} — {len(nb['cells'])} cell(s), {n_code} code cell(s), unexecuted; "
          f"it replays {a.replay_epochs} epoch(s) of "
          f"{(experiments.get('selection') or {}).get('run')}")
    print("OK: execute it in place with `jupyter nbconvert --to notebook --execute --inplace "
          f"--ExecutePreprocessor.timeout=1800 {out}`, then pass it to "
          "apply_training_layer.py --notebook")
    return 0


def _emit_eval(a, camp):
    """`--stage eval`: the evaluation's notebook, from the records the one look wrote."""
    campaign, err = read_json(camp / "campaign.json", "campaign.json")
    if err is not None:
        print(err.line() + " — the campaign is planned by /vega-train's plan_ladder.py")
        return 2
    seal, err = read_json(camp / "test_seal.json", "test_seal.json")
    if err is not None:
        print(err.line() + " — the evaluation notebook documents a look that has happened; "
              "run /vega-eval's evaluate.py first")
        return 2
    run_id = a.run or seal.get("run")
    eval_dir = camp / "eval" / str(run_id)
    metrics, err = read_json(eval_dir / "metrics.json", f"eval/{run_id}/metrics.json")
    if err is not None:
        print(err.line() + " — the notebook is generated from the evaluation's own records")
        return 2
    if not (eval_dir / "predictions.test.parquet").is_file():
        print(f"ERROR: {eval_dir / 'predictions.test.parquet'} does not exist; the notebook "
              "recomputes every number from the stored predictions rather than opening the "
              "test part again, and without them there is nothing to recompute from")
        return 2
    if metrics.get("predictions_layout") != PREDICTIONS_LAYOUT:
        print(f"ERROR: eval/{run_id}/metrics.json records predictions_layout "
              f"{metrics.get('predictions_layout')!r}, not {PREDICTIONS_LAYOUT!r} (BUILD-PLAN "
              "R-C2): the notebook reads the stored raw head, row score and truth columns of "
              "that layout, and would misread another one — re-run /vega-eval's evaluate.py")
        return 2

    # the authoring file sits beside campaign.json, as the training layer's does; a skill that
    # wrote it into eval/<run>/ instead is not wrong, so look there too before giving up
    layer_path = Path(a.layer) if a.layer else next(
        (c for c in (camp / "evaluation-layer.json", eval_dir / "evaluation-layer.json")
         if c.is_file()), camp / "evaluation-layer.json")
    readings = {}
    if layer_path.is_file():
        layer, lerr = read_json(layer_path, "the layer document")
        if lerr is None:
            readings = layer.get("readings") or {}
            print(f"OK: readings taken verbatim from {layer_path} "
                  f"({', '.join(sorted(readings))})")
    else:
        print(f"WARN: {layer_path} not found; the notebook carries no readings — the layer and "
              "the notebook must say the same thing, so author the layer first "
              "(judgement → notebook → apply)")

    package = a.package or (campaign.get("input") or {}).get("package")
    if not package:
        print("ERROR: no package name: pass --package, or let campaign.json name it")
        return 2
    layout = _layout_or_refuse(camp, package)
    if layout is None:
        return 2

    nb = eval_notebook(str(camp), seal, metrics, readings, package, layout, str(run_id))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"OK: wrote {out} — {len(nb['cells'])} cell(s), {n_code} code cell(s), unexecuted; "
          f"it recomputes {run_id} from eval/{run_id}/predictions.test.parquet and opens "
          "nothing")
    if int(seal.get("exposures") or 1) > 1:
        print(f"WARN: the seal records {seal['exposures']} exposures; the notebook's header "
              "says so, and the readings should say what the repeated look costs the verdict")
    print("OK: execute it in place with `jupyter nbconvert --to notebook --execute --inplace "
          f"--ExecutePreprocessor.timeout=1800 {out}`, then pass it to "
          "apply_evaluation_layer.py --notebook")
    return 0




#: what the bundle CLI must define for the verifying half (D2 §9): the one loader and the one
#: example comparison the notebook calls, never re-implements
_BUNDLE_CLI = ("load", "check_examples")


def _bundle_refusal(bundle, manifest):
    """Why this bundle cannot have its notebook, or None. The notebook's verifying half calls
    the bundle's own `predict.load` and `predict.check_examples` (R-D16), and a window bundle's
    single-row answer needs `MANIFEST.sample.history_rows` (R-A5) — a bundle packed without
    them is refused here rather than handed a notebook that fails when it runs."""
    cli = Path(bundle) / "predict.py"
    try:
        tree = ast.parse(cli.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return (f"ERROR: {cli} cannot be read as Python ({exc}); the notebook's verifying half "
                "imports the bundle's own entry point")
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    missing = [name for name in _BUNDLE_CLI if name not in defined]
    if missing:
        return (f"ERROR: {cli} defines no {', '.join(missing)}; the notebook compares the "
                "examples through the bundle's own predict.check_examples (D2 §9, BUILD-PLAN "
                "R-D16), and this bundle was packed before its CLI had one — re-pack it")
    sample = manifest.get("sample") or {}
    if sample.get("shape") == "window":
        rows = sample.get("history_rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            return (f"ERROR: MANIFEST.sample describes a window bundle but records history_rows "
                    f"{rows!r}; a single row is answered with that many rows of history "
                    "(BUILD-PLAN R-A5, R-D16) — re-pack the bundle")
    return None


def _transform_module(bundle, package, manifest):
    """The bundle's transform module (the FeatureTransformer's file), bundle-relative."""
    bmap, err = read_json(Path(bundle) / package / "blueprint-map.json", "blueprint-map.json")
    if err is None:
        entry = next((e for e in bmap.get("entries") or []
                      if e.get("role") == "FeatureTransformer" and e.get("file")), None)
        if entry:
            return f"{package}/{entry['file']}"
    return next((name for name in sorted(manifest.get("files") or {})
                 if name.startswith(package + "/") and name.endswith("/preprocess.py")),
                f"{package}/{_DEFAULT_FILES['FeatureTransformer']}")


def _emit_pack(a, bundle):
    """`--stage pack`: the bundle's own notebook, from the bundle's own records."""
    manifest, err = read_json(bundle / "MANIFEST.json", "MANIFEST.json")
    if err is not None:
        print(err.line() + " — the notebook documents a bundle that exists; run "
              "/vega-pack's pack.py first")
        return 2
    for name in ("schema.json", "examples/rows.csv", "examples/expected.parquet"):
        if not (bundle / name).is_file():
            print(f"ERROR: {bundle / name} does not exist; the notebook predicts on the "
                  "bundle's own examples and checks them against what was packed beside "
                  "them, and without those there is nothing to check")
            return 2
    if (manifest.get("roundtrip") or {}).get("status") != "pass":
        print("ERROR: this bundle's round trip did not pass, so there is no bundle to "
              "document — pack.py refuses to ship it and this notebook would only record the "
              "failure twice")
        return 2
    refusal = _bundle_refusal(bundle, manifest)
    if refusal:
        print(refusal)
        return 2

    layer_path = Path(a.layer) if a.layer else (bundle / "bundle-layer.json")
    readings = {}
    if layer_path.is_file():
        layer, lerr = read_json(layer_path, "the layer document")
        if lerr is None:
            readings = layer.get("readings") or {}
            print(f"OK: readings taken verbatim from {layer_path} "
                  f"({', '.join(sorted(readings))})")
    else:
        print(f"WARN: {layer_path} not found; the notebook carries no readings — the layer "
              "and the notebook must say the same thing, so author the layer first "
              "(judgement → card → notebook → apply)")

    package = a.package or manifest.get("package") or next(
        (Path(name).parts[0] for name in manifest.get("files") or {}
         if "/" in name and (bundle / Path(name).parts[0] / "__init__.py").is_file()), None)
    if not package:
        print("ERROR: no package name: pass --package, or let MANIFEST.files name it")
        return 2

    nb = pack_notebook(str(bundle), manifest, readings, package,
                       _transform_module(bundle, package, manifest))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"OK: wrote {out} — {len(nb['cells'])} cell(s), {n_code} code cell(s), unexecuted; "
          f"it imports {package} from the bundle and nothing from the chain")
    print("OK: execute it **from inside the bundle** with `cd " + str(bundle) + " && python3 "
          "-m nbconvert --to notebook --execute --inplace predict.ipynb`, then pass it to "
          "apply_bundle_layer.py --notebook")
    return 0



if __name__ == "__main__":
    sys.exit(main())
