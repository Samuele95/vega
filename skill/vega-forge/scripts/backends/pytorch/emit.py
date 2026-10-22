"""
The PyTorch backend's emitter: `emit(spec, pipeline, out_dir)`.

It writes the package `code-contract.md` §2 describes — one sub-package per blueprint module,
`ports.py`, `domain.py`, the adapters, `blueprint-map.json`, `spec_snapshot.json`,
`requirements.txt` — from the templates, where the *Effective Java* rules of §4 live. The
emitted code is the template plus the spec: a design defect is fixed in the template and
re-emitted, never in a run's copy.

**One template generation, `v2.3`** (the layout of D1 §4.1). The v2.2 layout — the block
implementations spliced into the Model adapter, the loss into the Trainer adapter, the adapters
as module-level functions — was the default until gate A, and is removed (BUILD-PLAN §2.12,
rule 1; WP B14). A template that still carries one of its splice placeholders (`{{PRELUDE}}`,
`{{BLOCKS}}`, `{{BUILDER}}`, `{{LOSSES}}`) is refused with the file the content now lives in.
The tests that hold the v2.3 code equal to v2.2 read c849f8b's sources from git
(`render_templates.baseline_source`, `render_templates.baseline_emitter`).

The package is the adapters from `templates/`, and beside each adapter the support files
`SUPPORT_FILES` lists, each with a `when` predicate on the spec. Three kinds of support file are
spliced rather than templated, because they must stand alone once /vega-pack copies the package
into a bundle: `<model>/blocks_{core,seq,graph,gen}.py` (the segments of `blocks.py` the spec
uses, each routed by `SEGMENT_FILE`), `<model>/builder.py` (the `builder` segment of
`builder.py`) and `<training>/losses.py` (the `losses` segment of `losses.py`). A `custom`
block's `code_ref` module is copied to `<model>/custom/<stem>.py` (R-B9), so the bundle's
package tree carries — and pins — the code the spec runs.

**One sample shape per package.** The support files of a shape (`windows.py`, `graphs.py`,
`graph_engines.py`) are emitted for a spec of that shape only, so a template's code for some
shapes sits in `# >>> shape: <shape> ...` / `# <<< shape` sections (`select_shapes`), and the
emitter keeps the spec's shape's: an emitted adapter dispatches to the shapes its package
carries and to no module the package does not contain. `domain.CARRIED_SHAPES` is the same
selection, and `NetSpec.from_dict` / `SampleSpec.from_spec` refuse a spec of another shape by
name (`UnsupportedShapeError`). The sections are additive — each is the code one shape adds — so
the template unselected is the program for every shape that a unit test's partial render reads.

**The splice contract.** A spliced file is: a generated docstring;
`from __future__ import annotations`; an opening taken from the source's `prelude` segment when
the source has one; the segments; and the package-relative imports the emitter adds after the
leading import block.

- *Every definition exists once.* `blocks.py`'s prelude is spliced whole into the file its
  `SEGMENT_FILE` names for it, which must be `blocks_core.py` (the one block file every package
  carries). `blocks_seq.py`, `blocks_graph.py` and `blocks_gen.py` open with the prelude's
  top-level `import` statements only, and import its definitions (`_act`, `_prod`, a shared base
  class, a registry) from `blocks_core` like any other sibling name — so a class the prelude
  defines is one class in the whole package, and a table a segment mutates is one table.
  `builder.py`'s and `losses.py`'s preludes are spliced whole into the one file each feeds.
- *A segment carries its own standard-library and framework imports* (at its top, or in its
  source's `prelude`); it never carries a package-relative import, because the source is also a
  runnable module where such an import cannot execute.
- *The emitter resolves the package-relative imports* from the names the file uses and leaves
  free: a name a sibling spliced file of the same module defines (`from .blocks_core import
  Dense, _act`) or a public name `domain.py` defines (`from ..domain import NetSpec`). A name
  defined in a segment this spec does not use is left unresolved on purpose (the branch that
  names it is never taken for this spec); any other unresolved name, and an import cycle between
  spliced files, is an `EmitError` naming it.
- *The tables are literals.* `SEGMENTS_FOR` and `SEGMENT_FILE` are module-level dict literals
  (plain or annotated assignments), read with `ast` — the emitter never imports torch.

The emitter invents nothing. A port the blueprint did not draw is an error naming
`/blueprint --concern`, not a file the emitter writes on its own authority; a file the emitter
would write with no map entry is refused before anything is written (the unmapped guard, C3).
The map's artifact roster is the blueprint's (`vega_roster`): every `artifacts[]` entry is a
name the emitted `domain.py` defines or is declared external with the stage that owns it (the
map's `external_artifacts`); one with no home is refused by name, never dropped.
A re-emit into a package removes the files its previous map listed and this emission no longer
writes (a `blocks_seq.py` left by a spec that dropped its recurrent block), and nothing else —
once the new files and the new map are in place.

A `custom` block's relative `code_ref` resolves against the **campaign root** (`run_dir`, CLI
`--campaign-root`), never against the run directory the package is emitted into
(netspec-contract §2): a stale copy left in a rung's directory is not the block's code.

    python3 emit.py --spec netspec.json --pipeline pipeline.json --out <run-dir>
                    [--campaign-root <dir>] [--sources <dir>]

(the stage's CLI is `emit_backend.py`, which calls `emit()` and smokes the package.)
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, NamedTuple

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"

NAME = "pytorch"

# the one template generation (v2.2 was removed at gate A, WP B14)
EMIT_GENERATION = "v2.3"
GENERATIONS = ("v2.3",)
# the v2.2 splice placeholders a v2.3 template may not carry, and where their content lives now
_V22_PLACEHOLDERS = {
    "PRELUDE": "<model>/blocks_core.py (the `prelude` segment of blocks.py)",
    "BLOCKS": "<model>/blocks_{core,seq,graph,gen}.py (the segments SEGMENT_FILE routes)",
    "BUILDER": "<model>/builder.py (the `builder` segment of builder.py)",
    "LOSSES": "<training>/losses.py (the `losses` segment of losses.py)",
}

# role -> (emitted filename, template, default module name, status)
ROLE_FILES = {
    "DataSource": ("data.py", "data.py", "ingestion", "implemented"),
    "FeatureTransformer": ("preprocess.py", "preprocess.py", "features", "pending: /vega-pack"),
    "Model": ("network.py", "network.py", "model", "implemented"),
    "ModelStore": ("store.py", "store.py", "model", "implemented"),
    "Trainer": ("loop.py", "loop.py", "training", "implemented"),
    "Evaluator": ("evaluate.py", "evaluate.py", "evaluation", "implemented"),
    "Predictor": ("predict.py", "predict.py", "serving", "pending: /vega-pack"),
    "Monitor": ("monitor.py", "monitor.py", "serving", "pending: /vega-pack"),
}

# the roles this stage cannot proceed without (code-contract §1: `net` needs these four)
REQUIRED_ROLES = ("DataSource", "Model", "Trainer", "ModelStore")

# The FALLBACK interface per role, used only when the blueprint carries no `signature` for a
# port and always with a WARN. The blueprint's §3 is the source of a port's interface; this
# table is what the emitter falls back to when the hand-off predates that field. Every method is
# the v2.3 adapter's (D1 §9–§11, backend-contract §1), with at most five parameters besides
# `self` (C12): the rest travel in the domain's parameter objects (`BatchPlan`, `RunContext`).
ROLE_SIGNATURES = {
    # typed with the domain's value objects (EJ-64): a port that says `Any` cannot tell an
    # adapter what crosses it, and every adapter then re-derived a NetSpec from a dict
    "DataSource": ["def load_part(self, spec: NetSpec, part: Part) -> SamplePart: ...",
                   "def iter_batches(self, spec: NetSpec, part: Part, batch_size: int | None = "
                   "None, columns=None, plan: BatchPlan | None = None) -> Iterator[dict]: ...",
                   "def batch_count(self, spec: NetSpec, part: Part, batch_size: int | None = "
                   "None, plan: BatchPlan | None = None) -> int: ...",
                   "def load_resident(self, spec: NetSpec, part: Part, device) -> SamplePart: ...",
                   "def make_loader(self, dataset: SamplePart, spec: NetSpec, shuffle: bool = "
                   "False, seed: int | None = None, batch_size: int | None = None) "
                   "-> Iterator[dict]: ...",
                   "def load_sealed_part(self, spec: NetSpec, token: str, seal_path: str = "
                   "\"test_seal.json\") -> SamplePart: ...",
                   "def design_matrix(self, spec: NetSpec, dataset: SamplePart, view: DesignView "
                   "= ...) -> DesignMatrix: ...",
                   "def exposure(self, spec: NetSpec) -> GraphExposure | None: ...",
                   "def target_scale(self, spec: NetSpec, season: int = 1) -> float | None: ...",
                   "def train_targets(self, spec: NetSpec) -> Any: ..."],
    "FeatureTransformer": ["def transform(self, df_raw, history=None, with_unseen: bool = False) "
                           "-> Any: ...",
                           # a graph recipe's (the shaper's transform-only module)
                           "def edges_for(self, df_raw, reference, history=None) -> Any: ...",
                           "def edges_within(self, df_raw, history=None) -> Any: ..."],
    "Model": ["def score(self, batch: dict) -> Any: ...",
              "def predict_proba(self, batch: dict) -> Any: ...",
              "def param_count(self) -> tuple[int, int]: ...",
              "def block_shapes(self, batch: dict) -> list[dict]: ...",
              # what serving and inspection read, and all they read (the serving seam): the
              # raw head of an assembled batch, and the network as data — its stages, reading,
              # identity, mode, centre and forward (None for a model with no stages: a baseline)
              "def head(self, batch: dict) -> Any: ...",
              "def anatomy(self) -> NetworkAnatomy | None: ..."],
    "ModelStore": ["def save(self, model: TrainedNetwork, path, spec: NetSpec, *, "
                   "epoch: int | None = None) -> Checkpoint: ...",
                   "def load(self, ref: Checkpoint | str, spec: NetSpec, build: ModelBuilder, *, "
                   "expected_digest: str | None = None, run_dir=None) -> TrainedNetwork: ...",
                   # what a bundle stamps as model_version without building the network
                   "def version(self, path) -> Checkpoint: ..."],
    # the parameters a port of the backend's types reads are typed with the backend's own
    # interfaces, which ports.py declares beside the ports: the network a fit trains
    # (`TrainedNetwork`), the data it may read — the DataSource without its sealed door
    # (`TrainingData`) —, the callbacks it calls (`Callbacks`), a model the Evaluator scores
    # (`ScoredModel`), the builder a checkpoint is restored through (`ModelBuilder`)
    "Trainer": ["def fit(self, model: TrainedNetwork, spec: NetSpec, data: TrainingData, "
                "callbacks: Callbacks | None = None, run: RunContext | None = None) "
                "-> FitResult: ...",
                "def evaluate(self, model: TrainedNetwork, spec: NetSpec, loader, loss_fn=None) "
                "-> dict: ...",
                "def predict(self, model: TrainedNetwork, spec: NetSpec, part: SamplePart) "
                "-> ScoredPart: ..."],
    "Evaluator": ["def scored(self, spec: NetSpec, dataset: SamplePart, model: ScoredModel) "
                  "-> ScoredPart: ...",
                  "def score_part(self, spec: NetSpec, part: Part, checkpoint=None, "
                  "model: ScoredModel | None = None) -> Metrics: ...",
                  "def score_sealed_test(self, spec: NetSpec, token: str, checkpoint=None, "
                  "model: ScoredModel | None = None, seal_path: str = \"test_seal.json\") "
                  "-> Metrics: ...",
                  # what /vega-eval and the bundle call beside scoring: the metrics recomputed
                  # from stored predictions, the per-head scores, the metric names, the options
                  "def measure(self, spec: NetSpec, part: Part, scored: ScoredPart) -> Metrics: ...",
                  "def heads(self, spec: NetSpec, dataset: SamplePart, model: ScoredModel) "
                  "-> Any: ...",
                  "def metric_names(self, spec: NetSpec) -> list: ...",
                  "def with_options(self, options: MetricOptions) -> Any: ..."],
    "Predictor": ["def predict(self, df_raw, history=None) -> Any: ...",
                  "def predictions(self, df_raw, history=None) -> tuple[Prediction, ...]: ...",
                  "def predict_one(self, row: dict, history=None) -> dict: ..."],
    "Monitor": ["def check(self, batch, predictions=None) -> tuple[DriftReading, ...]: ...",
                "def report(self, batch, predictions=None) -> tuple[DriftReading, ...]: ..."],
}

# the domain names ports.py imports for its annotations (under TYPE_CHECKING: ports import no
# code) when the rendered domain.py is not at hand — the value objects the fallback signatures
# name. Given the rendered domain.py, the vocabulary is every name it holds (`vega_roster.
# held_vocabulary`, the list `held_artifacts.py` prints for the blueprint): a §3 signature may
# annotate with any of them, as the blueprint SKILL.md patch tells it to
PORT_TYPES = ("BatchPlan", "Checkpoint", "DesignMatrix", "DesignView", "DriftReading",
              "EpochRecord", "FitResult", "GraphExposure", "MetricOptions", "Metrics",
              "ModelVersion", "NetSpec", "NetworkAnatomy", "Part",
              "Prediction", "ResidentPart", "RunContext", "SamplePart", "ScoredPart",
              "TensorPart")

# the `typing` names a port signature may annotate with beyond the ones ports.py always imports
# (Any, Iterator): imported only when a rendered signature reads one
PORT_TYPING = ("Callable", "Iterable", "Literal", "Mapping", "Optional", "Sequence", "Union")

# The adapter each role's port is held against (D1 §4.2: the class names are the backend's):
# the template that holds it and its class — for a role /vega-pack fills, the class it fills the
# role with. The FeatureTransformer's adapter is the shaper's transform-only module of
# functions (C10 reads it structurally), so its surface is the role table's own.
ADAPTER_TEMPLATES = {
    "DataSource": ("data.py", "ShapedParts"), "Model": ("network.py", "Network"),
    "ModelStore": ("store.py", "ModelRegistry"), "Trainer": ("loop.py", "Trainer"),
    "Evaluator": ("evaluate.py", "Evaluator"),
    "Predictor": ("serving/package/predict.py", "Predictor"),
    "Monitor": ("serving/package/monitor.py", "Monitor"),
}

# the package's own interfaces ports.py declares beside the blueprint's ports (code-contract §2)
BACKEND_PROTOCOLS = ("ModelBuilder", "TrainedNetwork", "TrainingData", "Callbacks", "ScoredModel",
                     "ArtifactExporter")

# what the head of ports.py always imports from typing
_PORTS_HEAD = frozenset({"TYPE_CHECKING", "Any", "Iterator", "Protocol"})

BOUNDARY_OF_ROLE = {
    "DataSource": ["shaped"], "FeatureTransformer": ["raw", "shaped"],
    "Model": ["features", "predictions"], "Predictor": ["raw", "predictions"],
    "Monitor": ["predictions"],
}


class EmitError(ValueError):
    """Raised with a message the stage prints as `ERROR:` — always naming the fix."""


# --------------------------------------------------------------------------- the sources

# The value types below are NamedTuples, not dataclasses: `backends.load()` executes this file
# without registering it in `sys.modules`, and a dataclass under PEP 563 looks its module up
# there while the class is being built.

class Sources(NamedTuple):
    """Where one generation's templates and splice sources live."""

    templates: Path          # the adapter and support templates
    blocks: Path             # `blocks.py`: the block segments, SEGMENTS_FOR (+ SEGMENT_FILE)
    builder: Path            # `builder.py`: the `builder` segment
    losses: Path             # `losses.py`: the `losses` segment

    @staticmethod
    def at(root, templates=None) -> "Sources":
        """The layout of a backend directory: `templates/`, `blocks.py`, `builder.py`,
        `losses.py` under `root` (`templates` overrides the first)."""
        root = Path(root)
        return Sources(Path(templates) if templates else root / "templates",
                       root / "blocks.py", root / "builder.py", root / "losses.py")

    def source(self, key) -> Path:
        return {"blocks": self.blocks, "builder": self.builder, "losses": self.losses}[key]


def _check_generation(generation) -> str:
    """The generation asked for (the default when None), or an EmitError naming the one there
    is — v2.2 by name, since a caller that still asks for it predates the cutover."""
    generation = generation or EMIT_GENERATION
    if generation in GENERATIONS:
        return generation
    why = (" — the v2.2 layout was removed at gate A (WP B14): its module-level adapters are "
           "the v2.3 adapter classes now (Network, ShapedParts, Trainer, Evaluator, "
           "ModelRegistry), reached through vega_core.PackageAdapters"
           if generation == "v2.2" else "")
    raise EmitError(f"unknown emit generation {generation!r}; this emitter knows "
                    f"{', '.join(GENERATIONS)} (the default is {EMIT_GENERATION}){why}")


def sources_for(generation=None) -> Sources:
    """The templates and splice sources a generation reads by default: this backend's own
    `templates/`, `blocks.py`, `builder.py` and `losses.py`."""
    _check_generation(generation)
    return Sources.at(HERE)


# --------------------------------------------------------------------------- splicing

_SEG = r"#\s*--8<--\s*{name}\s*\n([\s\S]*?)\n\s*#\s*-->8--\s*{name}\b"
# an opening marker is a comment line of its own (prose that quotes one, as the sources'
# docstrings do, is not a marker)
_SEG_OPEN = re.compile(r"^[ \t]*#[ \t]*--8<--[ \t]*(\w+)[ \t]*$", re.M)


def segment(path, name):
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(_SEG.format(name=re.escape(name)), text)
    if not m:
        raise EmitError(f"segment {name!r} not found in {path}; the emitter splices by the "
                        "`# --8<-- name` markers and one is missing or malformed")
    return m.group(1).rstrip()


def segment_names(path):
    """The segments a splice source marks, in file order."""
    text = Path(path).read_text(encoding="utf-8")
    return [m.group(1) for m in _SEG_OPEN.finditer(text)]


def block_types(spec):
    """Every block type the spec uses, composites' inner included, in first-seen order."""
    seen, order = set(), []

    def walk(bs):
        for b in bs or []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t and t not in seen:
                seen.add(t)
                order.append(t)
            walk((b.get("params") or {}).get("inner"))

    walk(spec.get("blocks"))
    return order


def _blocks(spec):
    """Every block of the spec, composites' inner included, depth first."""
    out = []

    def walk(bs):
        for b in bs or []:
            if isinstance(b, dict):
                out.append(b)
                walk((b.get("params") or {}).get("inner"))

    walk(spec.get("blocks"))
    return out


def _module_literal(path, name, hint=""):
    """A module-level literal assignment of a splice source, read without importing it (the
    block sources import torch; the emitter must not)."""
    if not Path(path).is_file():
        raise EmitError(f"the splice source {path} does not exist; a backend directory holds "
                        "templates/, blocks.py, builder.py and losses.py")
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise EmitError(f"{path} does not parse ({exc.msg}, line {exc.lineno}); the emitter "
                        "reads its tables and segments from it") from None
    for node in tree.body:
        value = _assigned_value(node, name)
        if value is None:
            continue
        try:
            return ast.literal_eval(value)
        except ValueError as exc:
            raise EmitError(f"{name} in {path} is not a literal ({exc}); the emitter reads "
                            "it without importing the module, so it must be written out "
                            "as a dict literal") from None
    raise EmitError(f"{name} not found in {Path(path).name}{hint}")


def _assigned_value(node, name):
    """The value `node` assigns to the module-level name `name` (`X = ...` or `X: T = ...`),
    or None when it assigns something else."""
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                            for t in node.targets):
        return node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
            and node.target.id == name:
        return node.value            # None for a bare annotation, which assigns nothing
    return None


# --------------------------------------------------------------------------- the world model

def _port_by_role(pipeline, role):
    return next((p for p in (pipeline.get("ports") or [])
                 if isinstance(p, dict) and p.get("role") == role), None)


def _module_of_port(pipeline, port_name, default):
    for m in pipeline.get("modules") or []:
        if isinstance(m, dict) and port_name in (m.get("ports_owned") or []):
            return m.get("name")
    return default


def _refuse_unstated_filling(pipeline):
    """A port whose adapter a later stage fills (the FeatureTransformer, the Predictor and the
    Monitor: `/vega-pack`, ROLE_FILES) says so in the blueprint — `"filled_by": "/vega-pack"`
    on its `ports[]` entry — and no other port does: the baseline then states as pending what
    the map records as pending, rather than asserting a port, and its module's arrows, the
    package does not yet realize. Refused by port, as the blueprint's §3."""
    problems = []
    for port in pipeline.get("ports") or []:
        if not isinstance(port, dict) or port.get("role") not in ROLE_FILES:
            continue
        status = ROLE_FILES[port["role"]][3]
        want = status.split(":", 1)[1].strip() if status.startswith("pending:") else None
        if port.get("filled_by") != want:
            problems.append(f"{port.get('name')} ({port['role']}) "
                            + (f"says filled_by {port.get('filled_by')!r}; its adapter is filled "
                               f"by {want}" if want else
                               f"says filled_by {port.get('filled_by')!r}; this stage implements "
                               "its adapter"))
    if problems:
        raise EmitError("the blueprint's ports do not say which stage fills each adapter: "
                        + "; ".join(problems) + ". A port a later stage fills carries "
                        "`\"filled_by\": \"<stage>\"` in pipeline.json.ports[] (and §3 and §4 say its "
                        "adapter and its module's arrows are realized then); every other port "
                        "carries none. Fix it in the blueprint (`/blueprint --concern`)")


def plan(spec, pipeline):
    """What will be emitted where, before anything is written."""
    package = ((pipeline.get("system") or {}).get("package")
               or (pipeline.get("system") or {}).get("name", "pipeline").replace("-", "_"))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", package):
        raise EmitError(f"pipeline.json's system.package {package!r} is not a Python "
                        "identifier; the blueprint names the package")

    missing = [r for r in REQUIRED_ROLES if _port_by_role(pipeline, r) is None]
    if missing:
        raise EmitError(
            "the blueprint draws no port with role " + " / ".join(repr(r) for r in missing) +
            ", which this stage needs. The emitter does not invent a port — a port nobody "
            "gated is a designer's choice nobody made. Run `/blueprint <model> --mode "
            "pipeline --concern \"the " + ", ".join(missing) + " port\"` and re-run.")

    _refuse_unstated_filling(pipeline)
    entries, modules = [], {}
    for role, (fname, tmpl, default_mod, status) in ROLE_FILES.items():
        port = _port_by_role(pipeline, role)
        if port is None:
            continue
        mod = _module_of_port(pipeline, port.get("name"), default_mod)
        modules.setdefault(mod, [])
        entries.append({"role": role, "port": port.get("name"), "module": mod,
                        "file": f"{mod}/{fname}", "template": tmpl, "status": status,
                        "responsibility": port.get("responsibility", ""),
                        "signature": list(port.get("signature") or []),
                        "source": f"blueprint:ports/{port.get('name')}",
                        "boundary": BOUNDARY_OF_ROLE.get(role)})
    model_mod = next((e["module"] for e in entries if e["role"] == "Model"), "model")
    eval_mod = next((e["module"] for e in entries if e["role"] == "Evaluator"), None)
    # The `baseline`-role files are scored models built from the Model module's readout, so
    # they live in the module that owns the Model port, not in the module of the contract that
    # asked for them (the Evaluator's: a module the blueprint may mark FP, and whose arrows do
    # not reach the readout's internals twice).
    baseline_mod = model_mod if eval_mod else None
    if baseline_mod:
        modules.setdefault(baseline_mod, [])
    return {"package": package, "entries": entries, "modules": sorted(modules),
            "model_module": model_mod, "eval_module": eval_mod,
            "baseline_module": baseline_mod,
            "module_names": {e["role"]: e["module"] for e in entries}}


# --------------------------------------------------------------------------- rendering

def _refuse_v22_placeholders(text, where):
    """A v2.2 splice placeholder in a template is refused, naming where its content lives in
    the D1 §4.1 layout: the template predates the cutover and would render without the code it
    expected to receive."""
    found = [k for k in _V22_PLACEHOLDERS if "{{" + k + "}}" in text]
    if found:
        raise EmitError(
            f"{where} carries the v2.2 splice placeholder(s) "
            + ", ".join("{{" + k + "}}" for k in found)
            + "; the emitter splices no code into an adapter any more (D1 §4.1): "
            + "; ".join(f"{{{{{k}}}}} is {_V22_PLACEHOLDERS[k]}" for k in found)
            + " — remove the placeholder and import from that file")


# --------------------------------------------------------------------------- shape sections

# A template's code for some sample shapes only sits between two comment lines of their own:
#     # >>> shape: window graph
#     ...
#     # <<< shape
# (prose that quotes a marker, as this comment does, is not one: a marker is the whole line)
_SHAPE_OPEN = re.compile(r"^[ \t]*# >>> shape:(?P<names>[^\n]*?)[ \t]*$")
_SHAPE_CLOSE = re.compile(r"^[ \t]*# <<< shape[ \t]*$")
# and a template's code for some task kinds only, the same way:
#     # >>> task: forecast regression
#     ...
#     # <<< task
_TASK_OPEN = re.compile(r"^[ \t]*# >>> task:(?P<names>[^\n]*?)[ \t]*$")
_TASK_CLOSE = re.compile(r"^[ \t]*# <<< task[ \t]*$")

# and builder.py's code for some block segments only (the dispatch of the block types a
# package carries), the same way:
#     # >>> segment: recurrent conv
#     ...
#     # <<< segment
_SEGMENT_OPEN = re.compile(r"^[ \t]*# >>> segment:(?P<names>[^\n]*?)[ \t]*$")
_SEGMENT_CLOSE = re.compile(r"^[ \t]*# <<< segment[ \t]*$")
# the names a `segment` section may carry: blocks.py's block segments and `custom` (a custom
# block's code ships beside the builder, R-B9)
BLOCK_SEGMENTS = ("dense", "embedding", "concat", "norm", "simple", "composite", "conv",
                  "recurrent", "attention", "attention_pool", "generative", "graph_ops",
                  "graph_conv", "graph_pool", "graph_edges", "head", "custom")

# the task kinds a `task` section names: a spec's effective kind (a sequence task's base)
TASK_KINDS = ("binary", "multiclass", "multilabel", "regression", "ordinal", "forecast",
              "unsupervised", "anomaly")


def _section_names(axis, found, where, lineno):
    names = tuple(s for s in re.split(r"[\s,]+", found.strip()) if s)
    unknown = [s for s in names if s not in axis.names]
    if not names or unknown:
        raise EmitError(f"{where}:{lineno}: a `# >>> {axis.word}:` section names "
                        f"{', '.join(unknown) or 'no ' + axis.word}; it names one or more of "
                        f"{', '.join(axis.names)}")
    return names


def effective_task(spec) -> str:
    """The spec's effective task kind: `task.kind`, or a `sequence-*` task's `base_kind`."""
    task = spec.get("task") or {}
    kind = task.get("kind")
    if str(kind).startswith("sequence-"):
        kind = task.get("base_kind")
    if kind not in TASK_KINDS:
        raise EmitError(f"task {task.get('kind')!r} (effective {kind!r}) is not one of "
                        f"{', '.join(TASK_KINDS)}; the compiler checks it before this point")
    return kind


def _trailing_blanks(lines) -> int:
    n = 0
    while n < len(lines) and not lines[-1 - n].strip():
        n += 1
    return n


def select_shapes(text, shapes=None, where="a template"):
    """The template with the code of the sample shapes `shapes` carries, and no other.

    A `# >>> shape: <shape> ...` line opens a section kept when one of its shapes is carried,
    a `# <<< shape` line closes the innermost open one, and both marker lines are always
    dropped: a section inside another is kept when both are, and every one is closed. A
    section left out takes the blank lines after it when blank lines precede it, so a package
    reads as if written for its shape. `shapes=None` carries every shape — the template as it
    stands, which is a program for every shape (the sections are additive: each is the code
    one shape adds), and what a unit test's partial render reads. The emitter passes the
    spec's own shape, so a package holds no dispatch to a shape whose support files it does not
    contain (D1 §4.1); `domain.CARRIED_SHAPES` is itself such a set, and refuses a spec of any
    other shape by name.
    """
    return _select_sections(text, _axes()["shape"], shapes, where)


def select_tasks(text, tasks=None, where="a template"):
    """The template with the code of the task kinds `tasks` carries, and no other: the same
    rule as `select_shapes` over `# >>> task: <kind> ...` / `# <<< task` sections. A task's
    kind is the spec's effective kind (`effective_task`: a `sequence-*` task's base), so a
    package holds the readings, losses and baselines of its own task only — the code of every
    other is not dead weight a reader and the gate must audit. `tasks=None` carries every kind
    (a unit test's partial render)."""
    return _select_sections(text, _axes()["task"], tasks, where)


class _Axis(NamedTuple):
    """One axis templates are cut along: its marker words and the names a section may carry."""

    word: str
    names: tuple
    opener: Any
    closer: Any


def _axes():
    return {"shape": _Axis("shape", SAMPLE_SHAPES, _SHAPE_OPEN, _SHAPE_CLOSE),
            "task": _Axis("task", TASK_KINDS, _TASK_OPEN, _TASK_CLOSE),
            "segment": _Axis("segment", BLOCK_SEGMENTS, _SEGMENT_OPEN, _SEGMENT_CLOSE)}


def select_segments(text, segments=None, where="a splice"):
    """builder.py's builder segment with the dispatch of the block segments `segments`
    carries, and no other: the same rule as `select_shapes` over `# >>> segment: <name> ...` /
    `# <<< segment` sections. The emitter passes the segments of blocks.py it spliced (and
    `custom` when a custom block ships), so the builder names no block class the package does
    not hold and refuses an uncarried block type by name. `segments=None` carries them all
    (the runnable source, a unit test's partial render)."""
    return _select_sections(text, _axes()["segment"], segments, where)


def carried_segments(ctx) -> tuple:
    """The block segments the package's builder dispatches to: the ones spliced into its block
    files, and `custom` when the spec has a custom block."""
    found = {seg for segs in ctx.block_files.values() for seg in segs}
    if "custom" in block_types(ctx.spec):
        found.add("custom")
    return tuple(s for s in BLOCK_SEGMENTS if s in found) or ("head",)


def _select_sections(text, axis, carried_names, where):
    names_all = axis.names
    carried = set(names_all if carried_names is None else carried_names)
    if carried - set(names_all) or not carried:
        raise EmitError(f"select_{axis.word}s: {sorted(carried)} is not a set of "
                        f"{axis.word}s ({', '.join(names_all)})")
    out, opened = [], []                   # opened: (line, kept) of each open section
    cut = None          # after a section left out: (blank lines before it, blank lines since)
    for lineno, line in enumerate(text.splitlines(keepends=True), 1):
        found = axis.opener.match(line)
        if found:
            names = _section_names(axis, found.group("names"), where, lineno)
            opened.append((lineno, bool(carried & set(names))))
        elif axis.closer.match(line):
            if not opened:
                raise EmitError(f"{where}:{lineno}: `# <<< {axis.word}` closes no section")
            _, kept = opened.pop()
            if not kept:          # the blank lines between two sections left out go with them
                cut = (_trailing_blanks(out) if cut is None else cut[0], [])
        elif all(kept for _, kept in opened):
            if cut is not None and not line.strip():
                cut[1].append(line)
                continue
            if cut is not None:            # the blank run across the cut: the longer side's
                before, after = cut
                out.extend(after[:max(0, len(after) - before)])
                cut = None
            out.append(line)
    if opened:
        raise EmitError(f"{where}:{opened[-1][0]}: the `# >>> {axis.word}:` section opened here "
                        f"is never closed (`# <<< {axis.word}`)")
    return "".join(out)


# --------------------------------------------------------------------------- compat names

# A v2.2 name kept for one release sits between two comment lines of its own:
#     # >>> compat: TrafficClass
#     ...
#     # <<< compat
# and is emitted only when the blueprint still uses the name (an artifact, or a word of a port
# signature): a package for a blueprint that does not carries no other project's vocabulary.
_COMPAT_OPEN = re.compile(r"^[ \t]*# >>> compat:[ \t]*(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*$")
_COMPAT_CLOSE = re.compile(r"^[ \t]*# <<< compat[ \t]*$")


def blueprint_words(pipeline) -> frozenset:
    """Every name the blueprint uses: its artifacts' names and the words of its port
    signatures."""
    words = {str(a.get("name")) for a in (pipeline.get("artifacts") or []) if isinstance(a, dict)}
    for port in pipeline.get("ports") or []:
        for line in (port.get("signature") or []) if isinstance(port, dict) else ():
            words |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(line)))
    return frozenset(words)


def select_compat(text, used=None, where="a template"):
    """The template with the compat names `used` names (None: all of them, the template as it
    stands) and no other; the marker lines are always dropped, and a section is never nested."""
    out, name, keep = [], None, True
    for lineno, line in enumerate(text.splitlines(keepends=True), 1):
        found = _COMPAT_OPEN.match(line)
        if found:
            if name is not None:
                raise EmitError(f"{where}:{lineno}: a `# >>> compat:` section inside another")
            name = found.group("name")
            keep = used is None or name in used
            if not keep:
                while out and not out[-1].strip():
                    out.pop()
            continue
        if _COMPAT_CLOSE.match(line):
            if name is None:
                raise EmitError(f"{where}:{lineno}: `# <<< compat` closes no section")
            name, keep = None, True
            continue
        if keep:
            out.append(line)
    if name is not None:
        raise EmitError(f"{where}: the `# >>> compat: {name}` section is never closed")
    return "".join(out)


def _render(text, mapping, where="a template", shapes=None, used=None, tasks=None):
    """`text` with the sections of `shapes` (default: every shape), the compat names `used`
    names (default: all) and every placeholder of `mapping` filled; a placeholder left over is
    refused."""
    _refuse_v22_placeholders(text, where)
    text = select_compat(select_tasks(select_shapes(text, shapes, where), tasks, where), used,
                         where)
    for k, v in mapping.items():
        text = text.replace("{{" + k + "}}", str(v))
    leftover = sorted(set(re.findall(r"\{\{([A-Z_]+)\}\}", text)))
    if leftover:
        raise EmitError(f"template placeholders left unrendered: {', '.join(leftover)}")
    return text


render_text = _render


# what the Trainer, its record and its losses read of the network, by the task kinds that read
# it (None: every package's): the members a package of another kind does not declare (SOLID-I)
TRAINED_NETWORK = (
    ("ids: tuple", None), ("training: bool", None),
    ("target_affine_pending: bool", ("regression", "forecast")),
    ("svdd_center_set: Any", ("anomaly",)),
    ("def __call__(self, batch: dict) -> Any: ...", None),
    ("def to(self, device) -> Any: ...", None),
    ("def train(self, mode: bool = True) -> Any: ...", None),
    ("def eval(self) -> Any: ...", None),
    ("def parameters(self) -> Iterator: ...", None),
    ("def named_parameters(self) -> Iterator: ...", None),
    ("def modules(self) -> Iterator: ...", None),
    ("def named_stages(self) -> tuple: ...", None),
    ("def fit_target_affine(self, affine) -> bool: ...", ("regression", "forecast")),
    ("def kl(self, batch=None) -> Any: ...", ("unsupervised", "anomaly")),
    ("def objective_center(self) -> Any: ...", ("anomaly",)),
)


def _trained_network(task, model_port=None) -> list:
    """The `TrainedNetwork` Protocol of a `task` package (None: every kind's members): the Model
    port (`model_port`, the blueprint's class name; None: no Model port drawn) and the
    training-time surface beyond it."""
    members = [m for m, kinds in TRAINED_NETWORK if task is None or kinds is None or task in kinds]
    fields = [m for m in members if not m.startswith("def ")]
    bases = f"{model_port}, Protocol" if model_port else "Protocol"
    return (["", "", f"class TrainedNetwork({bases}):",
             '    """The network the Model module builds, as a fit trains it, a store saves and',
             "    restores it and serving reads it: the Model port, and beyond it what the",
             "    Trainer, its record and its losses read — the stages and their ids (per-block",
             "    norms, the head), the module surface a fit drives (device, mode, parameters,",
             "    the forward) and what the package's task kind adds — a regression head's",
             "    target affine (A37), the VAE's KL, the one-class centre.",
             "",
             "    Not a blueprint port: the training-time surface of the Model port's one",
             "    network (`Network`); a baseline, never trained here, carries none of it",
             "    (SOLID-I). `fit` is annotated with it and reads nothing it does not declare.",
             '    """', ""]
            + ["    " + m for m in fields] + [""]
            + ["    " + m for m in members if m.startswith("def ")])


# what a fit reads of the DataSource, by the shapes and task kinds that carry the member (None:
# every package's): the DataSource without its sealed door (`load_sealed_part`) and without
# what only the Evaluator and the baselines read (`make_loader`, `design_matrix`)
TRAINING_DATA = (
    ("def load_part(self, spec: NetSpec, part: Part) -> SamplePart: ...", None, None),
    ("def iter_batches(self, spec: NetSpec, part: Part, batch_size: int | None = None, "
     "columns=None, plan: BatchPlan | None = None) -> Iterator[dict]: ...", None, None),
    ("def batch_count(self, spec: NetSpec, part: Part, batch_size: int | None = None, "
     "plan: BatchPlan | None = None) -> int: ...", None, None),
    ("def load_resident(self, spec: NetSpec, part: Part, device) -> SamplePart: ...", None, None),
    ("def target_scale(self, spec: NetSpec, season: int = 1) -> float | None: ...", ("window",),
     None),
    ("def train_targets(self, spec: NetSpec) -> Any: ...", None, ("regression", "forecast")),
)


def _training_data(shape, task) -> list:
    """The `TrainingData` Protocol of a `shape` package of the `task` kind (None: every one's)."""
    members = [m for m, shapes, kinds in TRAINING_DATA
               if (shape is None or shapes is None or shape in shapes)
               and (task is None or kinds is None or task in kinds)]
    return (["", "", "class TrainingData(Protocol):",
             '    """What a fit reads of the DataSource: the train and valid parts, streamed or',
             "    held, and what the package's shape and task add (a forecast window's seasonal",
             "    scale, a regression-family head's train targets). The DataSource's adapter",
             "    implements it; its sealed door to the test part is not part of it, so the",
             "    Trainer is handed no way to the test part (SOLID-I).",
             "",
             "    Not a blueprint port: the Trainer port's view of the DataSource port.",
             '    """', ""]
            + ["    " + (re.sub(r"\bSamplePart\b", SHAPE_PARTS[shape], m)
                         if shape in SHAPE_PARTS else m) for m in members])


def _known(lines, domain) -> list:
    """The backend interface's lines with each domain name the package's `domain.py` does not
    hold read as `Any` (a test's stub domain; a rendered domain holds them all)."""
    held = set(_held_names(domain))
    if not held:
        return lines
    loose = sorted({n for line in lines for n in re.findall(r"\b[A-Z][A-Za-z]+\b", line)
                    if n in PORT_TYPES + ("GraphPart", "WindowedPart") and n not in held})
    return [re.sub(rf"\b({'|'.join(loose)})\b", "Any", line) if loose else line
            for line in lines]


_CALLBACKS = ["", "", "class Callbacks(Protocol):",
              '    """The callback surface a caller of `fit` writes against (backend-contract',
              "    §1): the epoch record, `first_step.json`'s numbers after the run's first",
              "    optimiser step, and every `grad_sample_every`-th step's pre-clip global",
              "    gradient norm. Each hook is optional: the Trainer calls the ones the caller",
              "    defines.",
              "",
              "    Not a blueprint port: the Trainer port's parameter type.",
              '    """', "",
              "    def on_epoch(self, record: EpochRecord) -> None: ...",
              "    def on_first_step(self, record: dict) -> None: ...",
              "    def on_step(self, record: dict) -> None: ..."]


def _ports_py(package, pipeline, plan_, markers=None, domain=None, shape=None, task=None):
    """One Protocol per blueprint port, docstring = the blueprint's responsibility and source
    (EJ-20: every port is an interface; adapters depend on ports, never on each other).

    The methods come from the blueprint: `pipeline.json.ports[].signature` is the port's own
    interface, so §3 decides what a port declares and the emitter only renders it. A blueprint
    that carries no `signature` for a port falls back to the role table below, and says so with
    a WARN — the fallback is a stated gap, never a silent default, because a signature the
    emitter invents is an interface nobody gated.
    """
    markers = markers if markers is not None else []
    lines = [f'"""',
             f"{package}.ports — one Protocol per port the blueprint drew.",
             "",
             "Emitted by /vega-forge. The NAMES are the blueprint's; the ROLES are Vega's",
             "closed vocabulary, and the conformance check keys on the role while the code",
             "carries the name (code-contract §1). Every adapter depends on these Protocols",
             "and on `domain`, never on another adapter (EJ-20, EJ-64). They are static",
             "interfaces, not `@runtime_checkable`: an `isinstance` test against a Protocol",
             "checks names, never signatures, so the package offers none.",
             '"""',
             "from __future__ import annotations",
             "",
             "from typing import TYPE_CHECKING, Any, Iterator, Protocol",
             "",
             "if TYPE_CHECKING:  # the annotations are strings (PEP 563); ports import no code"]
    head, lines = lines, [""]
    for e in plan_["entries"]:
        cls = re.sub(r"[^A-Za-z0-9]", "", e["port"]) or e["role"]
        methods = [str(m) for m in (e.get("signature") or []) if str(m).strip()]
        if not methods:
            methods = list(_role_lines(e["role"], shape, task) if e["role"] in ROLE_SIGNATURES
                           else ["def __call__(self, *args, **kw) -> Any: ..."])
            markers.append(
                f"WARN: the blueprint draws no signature[] for port {e['port']!r} (role "
                f"{e['role']}); the emitter fell back to its role table. Add `signature` to "
                f"pipeline.json.ports[] so §3 decides the interface")
        lines += ["",
                  f"class {cls}(Protocol):",
                  f'    """{e["responsibility"] or e["role"]}',
                  "",
                  f'    Role: {e["role"]}.  Source: {e["source"]}.',
                  f'    Adapter: {e["file"]} ({e["status"]}).',
                  '    """', ""]
        for m in methods:
            lines.append("    " + m)
    ports_body = "\n".join(l for l in lines if l is not None)
    # Two Protocols that are not blueprint ports and are marked as such: `ModelBuilder` is the
    # callable three adapters already take as a parameter, and `ArtifactExporter` is the export
    # half the ModelStore port was carrying — a store's job is to put and get by digest, and
    # exporting to another format is a second responsibility with its own implementations.
    # There is no third: the serving and inspection code read the network through the Model
    # port itself — its `head` and its `anatomy()` (BUILD-PLAN R-B5) — so no `Inspectable` or
    # served-network protocol is emitted beside it.
    lines += ["", "", "class ModelBuilder(Protocol):",
              '    """The Model module\'s `build_model(spec, run_dir=None)`, as a type.',
              "",
              "    Not a blueprint port: it is the constructor the Model port's adapter",
              "    exposes, and the Trainer, the Evaluator and the ModelStore take it as a",
              "    parameter rather than importing a sibling adapter (EJ-20, EJ-64).",
              '    """', "",
              "    def __call__(self, spec, run_dir=None) -> Any: ..."]
    # The Model port is what serving, inspection and the Evaluator's scoring call (the
    # blueprint's methods). The Trainer and the Evaluator read more of what they hold, and say
    # so by type: the training-time surface of the one network the Model module builds, and
    # the scoring surface every Model of the package (the network, each baseline) carries.
    model_port = next((re.sub(r"[^A-Za-z0-9]", "", e["port"]) or e["role"]
                       for e in plan_["entries"] if e["role"] == "Model"), None)
    lines += _trained_network(task, model_port)
    lines += _known(_training_data(shape, task) + _CALLBACKS, domain)
    lines += ["", "", "class ScoredModel(Protocol):",
              '    """What the Evaluator reads of every model it scores, the network and each',
              "    fitted baseline alike: the score (the Model port's `score`, a baseline's",
              "    own), the part as the model reads it (`sample_view`: the part itself, or a",
              "    baseline's view with the target's history), the per-row score of an",
              "    unsupervised task (`row_score`) and the raw head (`head`: what the stored",
              "    `score_k` columns hold).",
              "",
              "    Not a blueprint port: the scoring surface the network and the baselines",
              "    share (a baseline implements no port: nothing serves through it), declared",
              "    as the one surface the Evaluator's scoring paths (`scored`, `heads`) read,",
              "    with no type test (a model of another shape fails where it is called).",
              '    """', "",
              "    def score(self, batch: dict) -> Any: ...",
              "    def sample_view(self, dataset) -> Any: ...",
              "    def row_score(self, batch: dict) -> Any: ...",
              "    def head(self, batch: dict) -> Any: ..."]
    lines += ["", "", "class ArtifactExporter(Protocol):",
              '    """Export a trained model to another runtime, and check the parity of the',
              "    export against the original.",
              "",
              "    Not a blueprint port: split out of the ModelStore role so that storing by",
              "    digest and converting to another format are two interfaces, which is what",
              "    lets a bundle carry the first without the second (EJ-21, SOLID-I). The Model",
              "    module's `export.py` implements it (`TorchExporter`); the positional input",
              "    order of an export is `export.BatchSignature.for_sample(sample, batch)`.",
              '    """', "",
              "    def export(self, model, spec, fmt: str, path, batch=None) -> dict: ...",
              "    def parity(self, model, exported_path, batch, atol: float = 1e-5, "
              "fmt: str = \"torchscript\") -> dict: ..."]
    # the domain names the rendered signatures annotate with, and no other: the blueprint's
    # port signatures are held to the package's vocabulary (a name it lacks is refused, by
    # port); the backend's own interfaces import what the package's domain.py holds
    body = "\n".join(l for l in lines if l is not None)
    read = _port_annotations(body, plan_)
    ported = _port_annotations(ports_body, plan_) - set(BACKEND_PROTOCOLS)
    _refuse_selfless(plan_)
    held = _held_names(domain)
    known = set(held) if held else set(PORT_TYPES)
    used = [t for t in PORT_TYPES if re.search(rf"\b{t}\b", ports_body)]
    used += [t for t in held if t in ported and t not in used]
    used += [t for t in sorted(read - ported) if t in known and t not in used]
    typing_used = [t for t in PORT_TYPING if t in read]
    _refuse_unknown_port_types(ported - set(used) - set(typing_used) - _PORTS_HEAD, plan_,
                               held, shape)
    if typing_used:
        names = sorted(["Any", "Iterator", "Protocol"] + typing_used,
                       key=str.lower)
        at = next(i for i, l in enumerate(head) if l.startswith("from typing import "))
        head[at:at + 1] = _import_lines("typing", ["TYPE_CHECKING"] + names)
    head += _import_lines(".domain", used, indent="    ") if used else ["    pass"]
    return "\n".join(head) + "\n" + body + "\n"


class SurfaceMethod(NamedTuple):
    """One public method of a role's adapter: its parameters beyond `self` as (name, kind,
    has a default) — kind `positional`, `keyword` or `var` — whether it is a static method, and
    whether the role's port carries it (ROLE_SIGNATURES)."""

    name: str
    params: tuple
    static: bool
    port: bool


def _parameters(fn, drop_self=True) -> tuple:
    a = fn.args
    pos = list(a.posonlyargs) + list(a.args)
    first_default = len(pos) - len(a.defaults)
    out = [(p.arg, "positional", i >= first_default) for i, p in enumerate(pos)]
    if a.vararg:
        out.append(("*" + a.vararg.arg, "var", False))
    out += [(p.arg, "keyword", d is not None) for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg:
        out.append(("**" + a.kwarg.arg, "var", False))
    if drop_self and out and out[0][0] in ("self", "cls"):
        out = out[1:]
    return tuple(out)


def _declared(method) -> ast.FunctionDef:
    return ast.parse("class _P:\n    " + str(method)).body[0].body[0]


def _role_port(role, shape, task=None) -> dict:
    """{name: parameters} of the role's port as the role table declares it, for one shape (and
    one task kind)."""
    return {(fn := _declared(m)).name: _parameters(fn) for m in _role_lines(role, shape, task)}


# the port methods a sample shape or a task kind adds (SOLID-I: no client depends on a method
# its package cannot carry): (role, method) -> (the shapes, the task kinds) that carry it, None
# for every one. A recipe's `edges_for` / `edges_within` are a graph recipe's entry points; the
# DataSource's `exposure` reads what a graph's training saw of the test part, `target_scale`
# a forecast window's seasonal scale, `train_targets` the train targets a regression-family
# head's affine is fitted on (TargetAffine.applies: an effective regression or forecast)
SCOPED_METHODS = {
    ("FeatureTransformer", "edges_for"): (("graph",), None),
    ("FeatureTransformer", "edges_within"): (("graph",), None),
    ("DataSource", "exposure"): (("graph",), None),
    ("DataSource", "target_scale"): (("window",), None),
    ("DataSource", "train_targets"): (None, ("regression", "forecast")),
}
# the sample part a port names, by the shape the package carries (`SamplePart` is their union)
SHAPE_PARTS = {"row": "TensorPart", "window": "WindowedPart", "graph": "GraphPart"}


def _scoped(role, name, shape, task) -> bool:
    """Whether a package of `shape` and `task` (None: every one) carries the role's method."""
    shapes, tasks = SCOPED_METHODS.get((role, name), (None, None))
    return (shape is None or shapes is None or shape in shapes) and \
        (task is None or tasks is None or task in tasks)


def _role_lines(role, shape=None, task=None) -> list:
    """The role table's declarations for one shape and task kind (None: every one's): a method
    `SCOPED_METHODS` gives to some shapes or kinds is the port's in those packages only — a
    graph recipe's edge entry points, a graph's exposure, a forecast's seasonal scale, a
    regression-family head's train targets — and, for one shape, the part the port reads or
    returns is named by the class that shape's package holds (`SHAPE_PARTS`), the artifact a
    blueprint draws, rather than by the union."""
    out = []
    for m in ROLE_SIGNATURES.get(role, ()):
        if not _scoped(role, _declared(m).name, shape, task):
            continue
        out.append(re.sub(r"\bSamplePart\b", SHAPE_PARTS[shape], m) if shape in SHAPE_PARTS else m)
    return out


def adapter_surface(role, shape, templates=None, task=None) -> dict:
    """{name: SurfaceMethod} — the public methods of the role's adapter in a `shape` package
    of the `task` kind (None: every kind's), properties and `_private` names aside, read from
    its template with that shape's and kind's sections kept. A port the blueprint draws for the
    role declares methods of this surface, each with these parameters (code-contract §1)."""
    port = {n: p for n, p in _role_port(role, shape, task).items() if not n.startswith("edges_")}
    if role not in ADAPTER_TEMPLATES:     # the recipe module: its transform, and a graph
        return {(fn := _declared(m)).name:  # recipe's edge entry points beside it
                SurfaceMethod(fn.name, _parameters(fn), False, fn.name in port)
                for m in _role_lines(role, shape, task)}
    rel, cls = ADAPTER_TEMPLATES[role]
    node = None
    # the template set the emission renders, else (a set without the adapter: a test's stub
    # templates) the backend's own — the surface is the adapter's either way
    for base in dict.fromkeys([Path(templates or HERE / "templates"), HERE / "templates"]):
        path = base / rel
        if not path.is_file():
            continue
        text = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: m.group(1).lower(),
                      path.read_text(encoding="utf-8"))
        try:
            tree = ast.parse(select_tasks(select_shapes(text, (shape,), str(path)),
                                          (task,) if task else None, str(path)))
        except SyntaxError:
            continue
        node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls),
                    None)
        if node is not None:
            break
    if node is None:
        raise EmitError(f"{rel} defines no class {cls}: the {role} adapter the role table names")
    out = {}
    for fn in node.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name.startswith("_"):
            continue
        decorators = {d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
                      for d in fn.decorator_list}
        if decorators & {"property", "setter", "cached_property"}:
            continue
        static = "staticmethod" in decorators
        out[fn.name] = SurfaceMethod(fn.name, _parameters(fn, drop_self=not static), static,
                                     fn.name in port)
    return out


_MODULE_ROLE = {"MODEL": "Model", "STORE": "ModelStore", "INGESTION": "DataSource",
                "TRAINING": "Trainer", "EVALUATION": "Evaluator", "FEATURES": "FeatureTransformer",
                "SERVING": "Predictor", "MONITOR": "Monitor", "BASELINES": "Model"}


def role_arrows(shape, templates=None) -> dict:
    """{role: (roles its module's code imports from)} for a `shape` package: what the
    blueprint's §4 must draw as `depends_on` arrows between the modules owning those roles
    (the implicit arrows to `domain` and `ports` aside) — read from the role's adapter template,
    its support files for that shape and, for a role /vega-pack fills, the serving package's."""
    base = Path(templates or HERE / "templates")
    out = {}
    for role, (_f, template, _m, _s) in ROLE_FILES.items():
        names = [template] + [sf.template for sf in SUPPORT_FILES.get(role, ())
                              if sf.template and (sf.template != "graph_engines.py"
                                                  or shape == "graph")]
        # /vega-pack renders the serving package into the Predictor's module, except
        # `package/monitor.py`, the Monitor's own adapter (vega-pack serving_templates.py)
        package = sorted((base / "serving" / "package").glob("*.py"))
        if role == "Predictor":
            names += [f"serving/package/{f.name}" for f in package if f.name != "monitor.py"]
        elif role == "Monitor":
            names += ["serving/package/monitor.py"]
        found = set()
        for name in names:
            path = base / name if (base / name).is_file() else HERE / "templates" / name
            if not path.is_file():
                continue
            text = select_shapes(path.read_text(encoding="utf-8"), (shape,), str(path))
            found |= {_MODULE_ROLE[m] for m in re.findall(r"from \.\.\{\{MODULE_([A-Z]+)\}\}",
                                                           text) if m in _MODULE_ROLE}
        out[role] = tuple(sorted(found - {role}))
    return out


def module_arrows(pipeline, shape, templates=None) -> dict:
    """{module: the modules its `depends_on` draws} for a `shape` package of this blueprint: an
    arrow from the module owning a role to the module owning each role its code imports
    (`role_arrows`), and an arrow to every module that owns no port (the domain's) — the
    arrows §4 draws, no other (C4 refuses an import against them, and an arrow no import
    takes)."""
    owner = {}
    for port in pipeline.get("ports") or []:
        for mod in pipeline.get("modules") or []:
            if port.get("name") in (mod.get("ports_owned") or []):
                owner[port.get("role")] = mod.get("name")
    domains = {m.get("name") for m in pipeline.get("modules") or [] if not m.get("ports_owned")}
    roles_of = {m.get("name"): {r for r, o in owner.items() if o == m.get("name")}
                for m in pipeline.get("modules") or []}
    needs = role_arrows(shape, templates)
    out = {}
    for name, roles in roles_of.items():
        targets = {owner[t] for r in roles for t in needs.get(r, ()) if t in owner}
        out[name] = sorted((targets | (domains if roles else set())) - {name})
    return out


def _shown(params) -> str:
    out, star = [], False
    for n, k, d in params:
        if k == "keyword" and not star:
            out.append("*")
            star = True
        out.append(n + ("=…" if d else ""))
    return ", ".join(out)


def _refuse_port_drift(plan_, shape, templates=None, task=None):
    """Each port the blueprint drew declares its role's port methods and only methods its
    adapter implements, each with the adapter's own parameters — names, order, kinds and which
    ones have a default. A port that drops, renames or reorders a parameter is an interface the
    adapter does not keep: a call written against the port binds its arguments to the wrong
    parameters, and nothing at run time says so. Refused by port and method, as the
    blueprint's §3, with what the adapter carries (`port_surface.py`)."""
    problems = []
    for e in plan_["entries"]:
        methods = [m for m in (e.get("signature") or []) if str(m).strip()]
        if not methods:
            continue                      # the role table's fallback, with its WARN
        surface = adapter_surface(e["role"], shape, templates, task)
        cls = (ADAPTER_TEMPLATES.get(e["role"]) or (None, "the recipe module"))[1]
        drawn = {}
        for m in methods:
            fn = _declared(m)
            drawn[fn.name] = _parameters(fn)
            have = surface.get(fn.name)
            if have is None:
                problems.append(f"{e['port']}.{fn.name} — {cls} has no public method "
                                f"{fn.name} (it has {', '.join(sorted(surface))})")
            elif have.params != drawn[fn.name]:
                problems.append(f"{e['port']}.{fn.name}({_shown(drawn[fn.name])}) — {cls} "
                                f"implements {fn.name}({_shown(have.params)})")
        missing = [n for n, m in surface.items() if m.port and n not in drawn]
        if missing:
            problems.append(f"{e['port']} omits {', '.join(missing)}, which the {e['role']} "
                            "port carries and the chain's stages call")
    if problems:
        raise EmitError(
            "the blueprint's port signatures are not the interfaces the package's adapters "
            "implement: " + "; ".join(problems) + ". Each port declares its role's port methods "
            "(and may add an adapter method beside them), each with the adapter's parameters, "
            "as `python3 <vega-forge>/scripts/port_surface.py --shape " + str(shape) + "` prints "
            "them; fix it in the blueprint's §3 and pipeline.json (`/blueprint --concern`), "
            "never in the emitted code")


def _held_names(domain):
    """The names the rendered `domain.py` holds an artifact under (value objects, enums,
    aliases), in its order; () when the domain is not at hand."""
    if not isinstance(domain, str):
        return ()
    return tuple(h.name for h in _roster_module().held_vocabulary(domain))


def _port_annotations(body, plan_):
    """The names the rendered port signatures read (annotations and defaults) and bind nowhere.
    A signature that does not parse is the blueprint's §3, refused by port."""
    try:
        tree = ast.parse(body, filename="ports.py")
    except SyntaxError as exc:
        bad = [e["port"] for e in plan_["entries"]
               if any(_unparsable(m) for m in (e.get("signature") or []))]
        raise EmitError(f"a port signature of the blueprint does not parse ({exc.msg}): "
                        f"{', '.join(bad) or 'ports.py'} — each `pipeline.json.ports[].signature` "
                        "entry is one Python method declaration (`def m(self, …) -> T: ...`); "
                        "fix it in the blueprint's §3 (`/blueprint --concern`)") from None
    free, annotation_only = free_names(tree)
    return set(free | annotation_only)


def _refuse_selfless(plan_):
    """A port method is an instance method: its first parameter is `self` (a Protocol method
    without it binds the caller's first argument to the instance). Refused by port and method,
    as the blueprint's §3."""
    bad = []
    for e in plan_["entries"]:
        for m in e.get("signature") or []:
            fn = ast.parse("class _P:\n    " + str(m)).body[0].body[0]
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            first = (fn.args.posonlyargs + fn.args.args)[:1]
            static = any(isinstance(d, ast.Name) and d.id == "staticmethod"
                         for d in fn.decorator_list)
            if not static and (not first or first[0].arg not in ("self", "cls")):
                bad.append(f"{e['port']}.{fn.name}")
    if bad:
        raise EmitError("a port method of the blueprint does not take `self` first: "
                        f"{', '.join(bad)} — each `pipeline.json.ports[].signature` entry is an "
                        "instance method (`def m(self, …) -> T: ...`); fix it in the blueprint's "
                        "§3 (`/blueprint --concern`)")


def _unparsable(method):
    try:
        ast.parse("class _P:\n    " + str(method))
    except SyntaxError:
        return True
    return False


def _refuse_unknown_port_types(unknown, plan_, held, shape):
    """A §3 signature that names a type ports.py cannot import is the blueprint's defect, named
    as such: the port, the name, and the vocabulary it may use (code-contract §1)."""
    if not unknown:
        return
    by_port = []
    for e in plan_["entries"]:
        text = "\n".join(str(m) for m in (e.get("signature") or []))
        names = sorted(n for n in unknown if re.search(rf"\b{re.escape(n)}\b", text))
        if names:
            by_port.append(f"{e['port']} ({e['role']}) names {', '.join(names)}")
    vocab = ", ".join(held) if held else ", ".join(PORT_TYPES)
    raise EmitError(
        "the blueprint's port signatures name types the package's ports.py cannot import: "
        + ("; ".join(by_port) or ", ".join(sorted(unknown)))
        + ". A §3 signature annotates with builtins, the typing names "
        + ", ".join(("Any", "Iterator") + PORT_TYPING)
        + " and the names the package's domain.py holds (`held_artifacts.py"
        + (f" --shape {shape}" if shape else "") + f"`: {vocab}); a framework or library type "
        "(a DataFrame, a tensor, an array) stays unannotated, because ports import no code. "
        "Fix it in the blueprint's §3 and pipeline.json (`/blueprint --concern`), never in the "
        "emitted code")


ports_py = _ports_py


def _init_py(package, plan_, pipeline):
    """EJ-15: the package `__init__.py` exports the adapters and the ports, nothing else."""
    names = ", ".join(f'"{e["role"]}"' for e in plan_["entries"])
    return (f'"""\n{package} — the pipeline the blueprint drew, with the network /vega-forge\n'
            f'compiled from the spec.\n\nModules: {", ".join(plan_["modules"])}.\n'
            f'The world model is `blueprint-map.json`; the spec is `spec_snapshot.json`.\n"""\n'
            "from __future__ import annotations\n\n"
            "from . import domain, ports  # noqa: F401\n\n"
            f"__all__ = [\"domain\", \"ports\"]\n")


init_py = _init_py


def _module_init(package, module, files, exports=None):
    """`<module>/__init__.py`: the module's public API — the adapter of each port it owns and
    every name that crosses the module's boundary (a support file another module's role reads,
    an interface of the backend the module implements) — re-exported by name (EJ-15: a
    consumer binds to the module, not to a file the emitter may re-slice), loaded on first use
    (PEP 562), so importing the module imports no framework. A function library is published
    as itself (`metrics`), beside its names."""
    doc = f'{package}.{module} — the adapters this module owns: {", ".join(sorted(files))}.'
    if not exports:
        return f'"""{doc}"""\n'
    by_file: dict = {}
    for name, src in exports.items():
        by_file.setdefault(src.lstrip("."), []).append(name)
    named = "; ".join(f"{', '.join(names)} ({stem}.py)" for stem, names in by_file.items())
    table = ",\n            ".join(f'"{name}": "{src}"' for name, src in exports.items())
    return (f'"""{doc}\n\nIts public API, re-exported by name and loaded on first use (PEP 562): '
            f'{named}.\n"""\n'
            "from __future__ import annotations\n\n"
            "import importlib\n\n"
            f"_EXPORTS = {{{table}}}\n"
            "__all__ = list(_EXPORTS)\n\n\n"
            "def __getattr__(name):\n"
            "    if name in _EXPORTS:\n"
            "        module = importlib.import_module(_EXPORTS[name], __name__)\n"
            '        return module if _EXPORTS[name] == "." + name else getattr(module, name)\n'
            '    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n')


def module_exports(plan_, files, rows=()) -> dict:
    """{module: {name: ".<file stem>"}}: the one public class of each port adapter file the
    emitter wrote (C14 holds an adapter module to one class at its top level) and its
    `build_model` alias, by module; for the function-shaped role (the FeatureTransformer,
    whose module functions ARE the port, C10) its public functions — so every module that owns
    a port declares its public API. Then what crosses a module's boundary from its support
    files (`rows`, the map's support entries): a file a role of another module reads
    (`used_by`), or one that implements an interface of the backend (`implements`), publishes
    the classes its `__all__` lists — and a library of functions (a function in `__all__`) is
    published as itself, by its stem, never name by name (its classes included) — so an arrow
    the blueprint draws
    lands on the module's API, never on a file behind it, and the API stays the types and
    the one library that cross."""
    out = {}
    for e in plan_["entries"]:
        text = files.get(e["file"])
        if not isinstance(text, str):
            continue
        body = ast.parse(text).body
        classes = [n.name for n in body if isinstance(n, ast.ClassDef) and not n.name.startswith("_")]
        names = classes if len(classes) == 1 else (
            [] if classes else [n.name for n in body
                                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")])
        names += [t.id for n in body if isinstance(n, ast.Assign) for t in n.targets
                  if isinstance(t, ast.Name) and t.id == "build_model"]
        for name in names:
            out.setdefault(e["module"], {})[name] = "." + Path(e["file"]).stem
    owner = {e["role"]: e["module"] for e in plan_["entries"]}
    for row in rows:
        text = files.get(row.get("file"))
        crosses = row.get("implements") in BACKEND_PROTOCOLS or any(
            owner.get(r) not in (None, row.get("module")) for r in row.get("used_by") or ())
        if row.get("kind") != "support" or not crosses or not isinstance(text, str):
            continue
        body, stem = ast.parse(text).body, Path(row["file"]).stem
        public = _dunder_all(body)
        functions = {n.name for n in body if isinstance(n, ast.FunctionDef)}
        table = out.setdefault(row["module"], {})
        if functions & set(public):
            table.setdefault(stem, "." + stem)
            continue
        classes = {n.name for n in body if isinstance(n, ast.ClassDef)}
        for name in public:
            if name in classes:
                table.setdefault(name, "." + stem)
    return out


def _dunder_all(body) -> list:
    """The names a module's `__all__` lists (a list or tuple of strings), in its order."""
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__"
                                                for t in node.targets):
            try:
                return [str(n) for n in ast.literal_eval(node.value)]
            except ValueError:
                return []
    return []


module_init = _module_init


def _requirements(spec, coverage):
    reqs = dict(coverage.get("requirements") or {})
    lines = [f"{k}{v}" for k, v in reqs.items()]
    return "\n".join(sorted(lines)) + "\n"


def _spec_snapshot(spec, coverage, lrs=None):
    """The spec verbatim, the backend's resolved defaults, and `read_paths` — the
    regularization paths the emitted code reads (netspec-contract §3, R2: each declaration it
    reads and every live path no declaration covers; `/vega-train` refuses a rung whose delta
    lands on a path outside them). `_resolved.regularization` adds the live paths a
    declaration shadows and the value each covered live path trains with."""
    snap = json.loads(json.dumps(spec))
    regularization = _regularization().snapshot(spec)
    snap["read_paths"] = regularization["read_paths"]
    snap["_resolved"] = {
        "backend": NAME,
        "determinism": coverage.get("determinism"),
        "optimizer_defaults": {"eps": 1e-8, "amsgrad": False},
        "norm_defaults": {"eps": 1e-5, "momentum": 0.1},
        "schedule_lrs": lrs or [],
        "regularization": regularization,
    }
    return snap


def _regularization():
    """The skill's `vega_regularization` (scripts/, two levels up): the one derivation of the
    live paths, which the compiler's R2 rule reads too."""
    scripts = str(HERE.parent.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import vega_regularization
    return vega_regularization


def _canonical_digest(obj):
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# the backend's own value objects, stamped apart from the blueprint's (backend-contract §1)
BACKEND_ARTIFACTS = ("EpochRecord", "FitResult", "StopReason")


def _roster_module():
    """The skill's `vega_roster` (scripts/, two levels up): the one derivation of the artifact
    roster, which the conformance gate's C6 reads back."""
    scripts = str(HERE.parent.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import vega_roster
    return vega_roster


def artifact_roster(pipeline, domain_text):
    """The blueprint's artifacts against the `domain.py` about to be written: every
    `artifacts[]` entry is held (a name domain.py defines) or declared external with its owner
    (code-contract §3). The roster is derived from the blueprint, never from a list of the
    emitter's: an artifact with no home, or an external one domain.py defines, is refused by
    name — dropping it from the map would be a divergence nobody declared."""
    vr = _roster_module()
    try:
        defined = vr.defined_names(domain_text)
    except SyntaxError as exc:
        raise EmitError(f"domain.py does not parse ({exc}); the artifact roster is read from "
                        "it") from None
    found, problems = vr.roster(pipeline, defined)
    if problems:
        raise EmitError("the blueprint's artifacts are not accounted for (code-contract §3): "
                        + "; ".join(problems))
    return found


def _adapter_row(e):
    row = {"file": e["file"], "module": e["module"], "port": e["port"], "role": e["role"],
           "source": e["source"], "status": e["status"]}
    if e["boundary"]:
        row["boundary"] = e["boundary"]
    return row


def _common_rows(plan_, roster, domain_text=None):
    """The support rows every package carries: domain and ports (the `__init__` files are
    `_init_rows`). `roster` is `artifact_roster(...)`: the domain row lists every blueprint
    artifact domain.py holds, in the blueprint's order, and every other name it holds (value
    objects, enums, aliases: `vega_roster.held_vocabulary` of `domain_text`) as the backend's
    own, so each value object of the package is declared once; the ports row lists the
    blueprint's ports and, apart, the package's own interfaces ports.py declares beside them."""
    if isinstance(domain_text, str):
        own = [n for n in _held_names(domain_text) if n not in roster.held]
    else:
        own = [a for a in BACKEND_ARTIFACTS if a not in roster.held]
    rows = [
        {"file": "domain.py", "module": "domain", "kind": "support",
         # the blueprint's artifacts domain.py holds, and — stamped apart — the backend's own
         # value objects the blueprint did not list (one source per name)
         "artifacts": list(roster.held),
         "backend_artifacts": own,
         "backend_artifacts_source": "backend-contract §1",
         "source": "blueprint:artifacts"},
        {"file": "ports.py", "module": "domain", "kind": "support",
         "ports": [e["port"] for e in plan_["entries"]], "source": "blueprint:ports",
         "backend_protocols": list(BACKEND_PROTOCOLS),
         "backend_protocols_source": "code-contract §2"}]
    return rows


def _init_rows(modules):
    rows = [{"file": "__init__.py", "module": "domain", "kind": "support",
             "source": "code-contract §2"}]
    for mod in modules:
        rows.append({"file": f"{mod}/__init__.py", "module": mod, "kind": "support",
                     "source": "code-contract §2"})
    return rows


def _map_document(spec, pipeline, package, entries, roster):
    # The Monitor stub points the filling stage at the concerns the blueprint recorded, so the
    # map carries them and the pointer resolves.
    concerns = [c for c in (pipeline.get("monitoring") or []) if isinstance(c, dict)]
    if concerns:
        for row in entries:
            if row.get("role") == "Monitor":
                row["concerns"] = concerns
    return {"schema": "vega/blueprint-map@1",
            "blueprint": (spec.get("input") or {}).get("pipeline"),
            # the digest of the hand-off this package was emitted against, computed from the
            # hand-off itself rather than copied from whatever the spec happened to stamp: the
            # map's whole claim is "this code is that blueprint's", and a claim about a file
            # named only by a mutable path is unattributable (EJ-49)
            "pipeline_digest": _canonical_digest(pipeline),
            "package": package,
            "entries": entries,
            "entrypoints": [
                {"file": "predict.py", "calls": "Predictor", "stage": "/vega-pack"},
                {"file": "serve/app.py", "calls": "Predictor", "stage": "/vega-pack"},
                {"file": "tests/test_roundtrip.py", "calls": "Predictor", "stage": "/vega-pack"}],
            # the blueprint's artifacts held outside the package, each with the stage that
            # owns it and where it lives (code-contract §3) — not emitted into domain.py
            "external_artifacts": roster.external_rows(),
            "unmapped": []}


def _port_signature(plan_, role) -> list:
    """The signature lines of the port of `role`: the blueprint's, else the role table's."""
    for e in plan_["entries"]:
        if e["role"] == role:
            drawn = [str(m) for m in (e.get("signature") or []) if str(m).strip()]
            return drawn or list(ROLE_SIGNATURES.get(role, ()))
    return list(ROLE_SIGNATURES.get(role, ()))


def transformer_stub(plan_) -> dict:
    """The FeatureTransformer stub's functions, generated from its port's signature (the
    shaper's transform-only module is a function module whose public functions ARE the port,
    C10): one function per port method, `self` dropped, each raising the pending message, so
    the stub states the contract /vega-pack fills — every method and parameter of it."""
    defs, names = [], []
    for line in _port_signature(plan_, "FeatureTransformer"):
        try:
            fn = ast.parse(line.strip()).body[0]
        except (SyntaxError, IndexError):
            continue
        if not isinstance(fn, ast.FunctionDef):
            continue
        args = fn.args
        if args.posonlyargs or args.args:
            first = (args.posonlyargs or args.args)[0]
            if first.arg in ("self", "cls"):
                if args.posonlyargs:
                    args.posonlyargs = args.posonlyargs[1:]
                else:
                    args.args = args.args[1:]
        fn.returns, fn.decorator_list = None, []
        fn.body = [ast.parse("raise NotImplementedError(_PENDING)").body[0]]
        defs.append(ast.unparse(ast.fix_missing_locations(fn)))
        names.append(fn.name)
    return {"TRANSFORMER_STUBS": "\n\n\n".join(defs) or "def transform(df_raw, history=None):\n"
                                                         "    raise NotImplementedError(_PENDING)",
            "TRANSFORMER_NAMES": ", ".join(f'"{n}"' for n in names or ["transform"]),
            "TRANSFORMER_SIGNATURE": "\n".join("    " + l.strip() for l in
                                               _port_signature(plan_, "FeatureTransformer"))
            or "    (the blueprint draws no signature for this port)"}


def common_mapping(spec, pipeline, plan_):
    """The placeholders every template may use: the package, the spec's identity, the module
    of each role, and each drawn port's name, responsibility and source."""
    mods = plan_["module_names"]
    common = {
        "PACKAGE": plan_["package"],
        # the NetSpec's own identity, canonical-JSON as domain.NetSpec.digest computes it
        # — not the dataset model's digest, which names a different artifact
        "SPEC_DIGEST": _canonical_digest(spec),
        "BLOCK_TYPES": ", ".join(block_types(spec)) or "(none)",
        "SERVING_MODE": (pipeline.get("serving") or {}).get("mode", "(not declared)"),
        "N_MONITORS": len(pipeline.get("monitoring") or []),
        "MODULE_MODEL": mods.get("Model", "model"),
        "MODULE_STORE": mods.get("ModelStore", mods.get("Model", "model")),
        "MODULE_INGESTION": mods.get("DataSource", "ingestion"),
        "MODULE_TRAINING": mods.get("Trainer", "training"),
        "MODULE_EVALUATION": mods.get("Evaluator", "evaluation"),
        "MODULE_FEATURES": mods.get("FeatureTransformer", "features"),
        "MODULE_SERVING": mods.get("Predictor", "serving"),
        "MODULE_MONITOR": mods.get("Monitor", "serving"),
        "MODULE_BASELINES": plan_.get("baseline_module") or mods.get("Model", "model"),
    }
    common.update(transformer_stub(plan_))
    for e in plan_["entries"]:
        common[f"PORT_{e['role'].upper()}"] = e["port"]
        common[f"RESPONSIBILITY_{e['role'].upper()}"] = e["responsibility"] or "(no responsibility line in the blueprint)"
        common[f"SOURCE_{e['role'].upper()}"] = e["source"]
    for role in ROLE_FILES:
        common.setdefault(f"PORT_{role.upper()}", f"({role} — not drawn by the blueprint)")
        common.setdefault(f"RESPONSIBILITY_{role.upper()}", "(not drawn by the blueprint)")
        common.setdefault(f"SOURCE_{role.upper()}", "(not drawn by the blueprint)")
    return common


def _coverage():
    return json.loads((HERE / "coverage.json").read_text(encoding="utf-8"))


def _refuse_unmapped(written, bmap):
    """The unmapped guard (C3): every file the emitter wrote has a map entry."""
    mapped = {e["file"] for e in bmap["entries"]} | {"blueprint-map.json"}
    unmapped = sorted(f for f in written if f not in mapped)
    if unmapped:
        raise EmitError(f"the emitter wrote files with no map entry ({', '.join(unmapped)}); "
                        "a file nobody mapped is a designer's choice the emitter is not "
                        "allowed to make")


class _Writer:
    """Writes package files and remembers which (package-relative) it wrote."""

    def __init__(self, root):
        self.root = Path(root)
        self.written = []

    def write(self, rel, content):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
        self.written.append(rel)


def _write_beside(out, spec, coverage, lrs):
    (out / "spec_snapshot.json").write_text(
        json.dumps(_spec_snapshot(spec, coverage, lrs), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (out / "requirements.txt").write_text(_requirements(spec, coverage), encoding="utf-8")


# --------------------------------------------------------------------------- emit

class EmitOptions(NamedTuple):
    """How a package is emitted, beside what (the spec, the pipeline) and where (out_dir)."""

    generation: str | None = None   # None: EMIT_GENERATION (v2.3, the only one)
    sources: Any = None             # the templates and splice sources (`Sources`); None: ours
    campaign_root: Any = None       # emit's `run_dir`: what a custom code_ref resolves against
    lrs: Any = None                 # the schedule's learning rates, stamped in spec_snapshot.json


def emit(spec, pipeline, out_dir, run_dir=None, *, generation=None):
    """Write the package. Returns a record the caller prints and the layer carries.

    `generation` is accepted for the callers that name it and must be `v2.3` (the only one;
    `v2.2` is refused, naming the cutover). `emit_with` takes the other options (other
    templates and splice sources, the schedule's learning rates).

    `run_dir` is the **campaign root** (the name is kept for the stage's callers): the one
    directory a `custom` block's relative `code_ref` resolves against — never `out_dir`, the
    run directory the package goes to (netspec-contract §2). When the forge stage emits, the
    two are the same directory; a rung that re-emits into its own run directory passes the
    campaign root.
    """
    return emit_with(spec, pipeline, out_dir,
                     EmitOptions(generation=generation, campaign_root=run_dir))


def emit_with(spec, pipeline, out_dir, options: EmitOptions):
    """`emit()` over explicit `EmitOptions`: `sources=Sources.at(dir)` replaces the backend's
    templates and splice sources (the unit tests emit the layout over stub templates this way,
    and `render_templates.py --full` over any template set)."""
    generation = _check_generation(options.generation)
    opts = options._replace(generation=generation,
                            sources=options.sources or sources_for(generation))
    if not pipeline:
        raise EmitError("the blueprint's pipeline.json is required: the emitted code lives in "
                        "its module map. Run `/blueprint <model> --mode pipeline` first.")
    return _emit_v23(spec, pipeline, Path(out_dir), opts)


# --------------------------------------------------------------------------- v2.3: the table

SAMPLE_SHAPES = ("row", "window", "graph")
BLOCK_FILES = ("blocks_core", "blocks_seq", "blocks_graph", "blocks_gen")
# the block file that carries blocks.py's prelude whole: the one every package has (D1 §4.1)
PRELUDE_FILE = "blocks_core"


class _Ctx(NamedTuple):
    """What the `when` predicates read: the spec, the plan, the sample shape, the block files
    the spec's block types need (stem -> segments) and the effective task kind (None: every
    kind's code — a partial render's)."""

    spec: dict
    plan: dict
    shape: str
    block_files: dict
    task: Any = None


def _always(ctx):
    return True


def _window(ctx):
    return ctx.shape == "window"


def _graph(ctx):
    return ctx.shape == "graph"


def _evaluator_drawn(ctx):
    return ctx.plan.get("eval_module") is not None


def _uses_blocks(stem):
    def when(ctx):
        return stem in ctx.block_files
    when.__name__ = f"uses_{stem}"
    return when


class SupportFile(NamedTuple):
    """One file the emitter writes beside a role's adapter (D1 §4.1).

    Exactly one of `template` (rendered from the generation's templates) and `source`
    (spliced from a splice source: `segments` of it, or — for `blocks` — the segments
    `SEGMENT_FILE` routes to this file) is set. `role` is set for the `baseline` files, the
    models the Evaluator scores (`ScoredModel`, no blueprint port); every other file is a
    `kind: support` map entry.
    """

    name: str
    template: Any = None           # str | None
    source: Any = None             # "blocks" | "builder" | "losses" | None
    segments: tuple = ()
    when: Callable = _always
    used_by: tuple = ()
    role: Any = None               # "baseline" | None
    cite: str = "backend-contract §1"
    extra: tuple = ()


_EVAL_CONTRACT = "vega-eval/references/eval-contract.md"

# host role -> the files emitted after that role's adapter, in its module (D1 §4.1)
SUPPORT_FILES = {
    "Model": (
        SupportFile("builder.py", source="builder", segments=("builder",), used_by=("Model",)),
        SupportFile("blocks_core.py", source="blocks", used_by=("Model",)),
        SupportFile("blocks_seq.py", source="blocks", when=_uses_blocks("blocks_seq"),
                    used_by=("Model",)),
        SupportFile("blocks_graph.py", source="blocks", when=_uses_blocks("blocks_graph"),
                    used_by=("Model",)),
        SupportFile("blocks_gen.py", source="blocks", when=_uses_blocks("blocks_gen"),
                    used_by=("Model",)),
        SupportFile("readout.py", template="readout.py",
                    used_by=("Model", "Trainer", "Evaluator", "Predictor")),
        SupportFile("assemble.py", template="assemble.py", used_by=("Predictor",),
                    extra=(("placement", "in the Model's module, not the Predictor's: the "
                            "assembler is the network's own cut of a batch (the DataSource's, "
                            "inverted for served rows), so it is pinned with the network "
                            "/vega-eval scored, and /vega-pack's serving re-emission "
                            "(--re-emit-serving) cannot change how a served batch is cut"),)),
        SupportFile("metrics.py", template="metrics.py",
                    # the metrics, and the numpy half of each reading the readout hands out
                    # (`TaskReadout.scores`): read wherever the readout is
                    used_by=("Model", "Trainer", "Evaluator", "Predictor"),
                    cite=f"{_EVAL_CONTRACT} §3"),
        SupportFile("baselines.py", template="baselines.py", when=_evaluator_drawn,
                    role="baseline", cite=f"{_EVAL_CONTRACT} §4"),
        SupportFile("baselines_extra.py", template="baselines_extra.py", when=_evaluator_drawn,
                    role="baseline", cite=f"{_EVAL_CONTRACT} §4"),
        SupportFile("export.py", template="export.py", used_by=("ModelStore",),
                    extra=(("implements", "ArtifactExporter"),
                           # what an export writes has a home, and it is not the package's
                           ("output", {"artifact": "the exported network (TorchScript, ONNX) "
                                       "and its parity record", "owner": "/vega-pack",
                                       "where": "<bundle>/exports/ (model.pt, model.onnx, "
                                       "parity.json), each pinned by MANIFEST.exports",
                                       "source": "vega-pack/references/bundle-contract.md "
                                       "§1, §7"}))),
    ),
    "DataSource": (
        SupportFile("frames.py", template="frames.py", used_by=("DataSource",)),
        SupportFile("windows.py", template="windows.py", when=_window, used_by=("DataSource",)),
        SupportFile("graphs.py", template="graphs.py", when=_graph, used_by=("DataSource",)),
    ),
    "Trainer": (
        SupportFile("losses.py", source="losses", segments=("losses",), used_by=("Trainer",)),
        SupportFile("optim.py", template="optim.py", used_by=("Trainer",)),
        SupportFile("record.py", template="record.py", used_by=("Trainer",)),
        SupportFile("engines.py", template="engines.py", used_by=("Trainer",)),
        SupportFile("graph_engines.py", template="graph_engines.py", when=_graph,
                    used_by=("Trainer",)),
    ),
}


def sample_shape(spec):
    sample = (spec.get("data_contract") or {}).get("sample") or {}
    shape = sample.get("shape", "row")
    if shape not in SAMPLE_SHAPES:
        raise EmitError(f"data_contract.sample.shape {shape!r} is not one of "
                        f"{', '.join(SAMPLE_SHAPES)}; the compiler checks it against the "
                        "shaper's manifest before this point")
    return shape


def _drawn(plan_):
    return [e["role"] for e in plan_["entries"]]


def _file_row(sf, rel, mod, plan_):
    """The map entry of one SUPPORT_FILES file."""
    if sf.role:
        # a baseline implements no blueprint port: it is a model the Evaluator scores (the
        # backend's `ScoredModel`), and the map says so rather than listing it as another
        # implementation of the Model port, which nothing serves through it
        return {"file": rel, "module": mod, "role": sf.role, "implements": "ScoredModel",
                "used_by": [r for r in ("Evaluator",) if r in _drawn(plan_)],
                "source": sf.cite, "status": "implemented"}
    drawn = _drawn(plan_)
    row = {"file": rel, "module": mod, "kind": "support",
           "used_by": [r for r in sf.used_by if r in drawn], "source": sf.cite}
    row.update(dict(sf.extra))
    return row


def v23_layout(plan_):
    """Template name -> (package-relative path, map entry) for every template the v2.3 table
    renders — the adapters of the drawn ports and the templated support files, whatever their
    `when` says. `render_templates.py` places a template set by it."""
    out = {}
    for e in plan_["entries"]:
        out[e["template"]] = (e["file"], _adapter_row(e))
    for host, files in SUPPORT_FILES.items():
        mod = plan_["module_names"].get(host)
        if mod is None:
            continue
        for sf in files:
            if sf.template:
                rel = f"{mod}/{sf.name}"
                out[sf.template] = (rel, _file_row(sf, rel, mod, plan_))
    return out


def _template(sources, name, rel):
    path = sources.templates / name
    if not path.is_file():
        raise EmitError(
            f"generation v2.3 renders {rel} from the template {name}, which {sources.templates} "
            "does not contain (D1 §4.1); every template of the table is part of the backend — "
            "restore it, or point `sources` at a template set that has it")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- v2.3: splicing

def block_plan(spec, blocks_path):
    """stem -> the segments of `blocks.py` the spec's block types need there, in file order
    (the prelude aside: it opens `PRELUDE_FILE` whole and every other block file with its
    imports only). `blocks_core` is always present (it carries the prelude and the head)."""
    seg_map = _module_literal(blocks_path, "SEGMENTS_FOR")
    seg_file = _module_literal(
        blocks_path, "SEGMENT_FILE",
        hint="; generation v2.3 routes each block segment to the file SEGMENT_FILE names "
             f"({', '.join(BLOCK_FILES)}) — D1 §4.1, WP B03")
    marked = segment_names(blocks_path)
    if "prelude" not in marked:
        raise EmitError(f"{Path(blocks_path).name} marks no `prelude` segment; it opens "
                        f"{PRELUDE_FILE}.py, and its imports open every other block file")
    home = str(seg_file.get("prelude", "")).removesuffix(".py")
    if home != PRELUDE_FILE:
        raise EmitError(
            f"{Path(blocks_path).name}'s SEGMENT_FILE routes the prelude to "
            f"{home or 'no file'!r}; "
            f"generation v2.3 splices it whole into {PRELUDE_FILE}.py, the one block file every "
            "package carries, and opens the other block files with its imports only, so each "
            f"of its definitions exists once (D1 §4.1) — route it to {PRELUDE_FILE!r}")
    wanted = set()
    for t in block_types(spec):
        if t not in seg_map:
            raise EmitError(f"block type {t!r} has no segment mapping in blocks.py; the "
                            "compiler should have refused it before this point")
        wanted.update(s for s in seg_map[t] if s != "prelude")
    unmarked = sorted(wanted - set(marked))
    if unmarked:
        raise EmitError(f"SEGMENTS_FOR names segment(s) {', '.join(unmarked)}, which "
                        f"{Path(blocks_path).name} does not mark with `# --8<-- <name>`")
    groups = {"blocks_core": []}
    for s in (s for s in marked if s in wanted):
        stem = str(seg_file.get(s, "")).removesuffix(".py")
        if stem not in BLOCK_FILES:
            raise EmitError(
                f"segment {s!r} is routed to {stem or 'no file'!r} by blocks.py's SEGMENT_FILE; "
                f"generation v2.3 splices block segments into {', '.join(BLOCK_FILES)} only "
                "(D1 §4.1) — give it one of those")
        groups.setdefault(stem, []).append(s)
    return {stem: tuple(segs) for stem, segs in groups.items()}


_DUNDERS = frozenset({"__file__", "__name__", "__doc__", "__spec__", "__package__",
                      "__loader__", "__builtins__", "__annotations__", "__dict__", "__path__",
                      "__cached__", "__qualname__", "__module__", "__class__"})
_BUILTINS = frozenset(dir(builtins)) | _DUNDERS


def defined_names(tree):
    """The names a module defines at its top level (def, class, assignment), including inside
    top-level if/try/with blocks — not the names it imports."""
    out = set()

    def visit(stmts):
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    out.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(node, ast.AnnAssign) and node.value is not None \
                    and isinstance(node.target, ast.Name):
                out.add(node.target.id)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for part in ("body", "orelse", "finalbody"):
                    visit(getattr(node, part, None) or [])
                for handler in getattr(node, "handlers", None) or []:
                    visit(handler.body)

    visit(tree.body)
    return out


class _Names(ast.NodeVisitor):
    """Loads, bindings and annotation-only loads of a module, scope-insensitively: a name bound
    anywhere counts as bound (generous on purpose — the check exists to catch a missing import,
    not to replace the interpreter's scoping)."""

    def __init__(self):
        self.bound, self.loads, self.annotations = set(), set(), set()
        self._in_annotation = 0

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            (self.annotations if self._in_annotation else self.loads).add(node.id)
        else:
            self.bound.add(node.id)

    def _annotation(self, node):
        if node is not None:
            self._in_annotation += 1
            self.visit(node)
            self._in_annotation -= 1

    def visit_arg(self, node):
        self.bound.add(node.arg)
        self._annotation(node.annotation)

    def _function(self, node):
        self.bound.add(node.name)
        for dec in node.decorator_list:
            self.visit(dec)
        self.visit(node.args)
        self._annotation(node.returns)
        for stmt in node.body:
            self.visit(stmt)

    visit_FunctionDef = visit_AsyncFunctionDef = _function

    def visit_ClassDef(self, node):
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._annotation(node.annotation)
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_Import(self, node):
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name != "*":
                self.bound.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest:
            self.bound.add(node.rest)
        self.generic_visit(node)


def free_names(tree):
    """(names read at run time and bound nowhere, names read only in annotations)."""
    v = _Names()
    v.visit(tree)
    free = v.loads - v.bound - _BUILTINS
    annotation_only = v.annotations - v.loads - v.bound - _BUILTINS
    return free, annotation_only


def _parse(text, where):
    try:
        return ast.parse(text, filename=where)
    except SyntaxError as exc:
        raise EmitError(f"{where} does not parse ({exc.msg}, line {exc.lineno}); a spliced "
                        "segment must be complete top-level Python on its own") from None


def _library_names(sources):
    """Every name a segment of a splice source defines — a name the builder may reference for
    a block type this spec does not use, so its segment is legitimately absent."""
    names = set()
    for key in ("blocks", "builder", "losses"):
        path = sources.source(key)
        if not path.is_file():
            continue
        for seg in segment_names(path):
            try:
                names |= defined_names(ast.parse(segment(path, seg)))
            except SyntaxError:
                continue          # reported when (if) the segment is spliced
    return names


def _import_lines(module, names, indent=""):
    """`from <module> import <names>`, wrapped in parentheses at 99 columns."""
    line = f"{indent}from {module} import {', '.join(names)}"
    if len(line) <= 99:
        return [line]
    lines, cur = [], f"{indent}from {module} import ("
    indent = " " * len(cur)
    for i, n in enumerate(names):
        piece = n + (", " if i < len(names) - 1 else ")")
        if len(cur) + len(piece.rstrip()) > 99 and not cur.endswith("("):
            lines.append(cur.rstrip())
            cur = indent
        cur += piece
    lines.append(cur)
    return lines


def _insert_after_imports(body, tree, block):
    """The body with `block` (lines) inserted after its leading import statements, one blank
    line before it and two after (PEP 8's spacing before top-level code)."""
    if not block:
        return body
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = node.end_lineno
        else:
            break
    lines = body.split("\n")
    head, tail = lines[:last], lines[last:]
    while tail and not tail[0].strip():
        tail = tail[1:]
    return "\n".join(head + ([""] if head else []) + block + ["", ""] + tail)


class _Splice(NamedTuple):
    rel: str
    source: Path
    segments: tuple        # spliced whole, in order (the prelude first when it opens the file)
    body: str
    tree: ast.Module
    defines: frozenset
    prelude_imports: bool  # opens with the imports of the source's prelude (and nothing else)


def prelude_imports(prelude, where="prelude"):
    """The top-level `import` statements of a prelude segment, as source, in order: what a
    block file other than `PRELUDE_FILE` opens with. Its definitions are not copied — the file
    imports them from `PRELUDE_FILE`, so each exists once in the package."""
    tree = _parse(prelude, where)
    return "\n".join(ast.get_source_segment(prelude, node) for node in tree.body
                     if isinstance(node, (ast.Import, ast.ImportFrom)))


def _splice_body(rel, sf, path, ctx):
    """(segments spliced whole, whether the prelude's imports open it, the body text): the
    task kind's sections kept, and the prelude's imports the file does not read dropped — a
    spliced file opens with what it uses, not with every import its source file needs."""
    has_prelude = "prelude" in segment_names(path)
    if sf.source != "blocks":
        segs = (("prelude",) if has_prelude else ()) + sf.segments
        body = "\n\n\n".join(segment(path, s) for s in segs)
        if sf.source == "builder":
            body = select_segments(body, carried_segments(ctx), rel)
        return segs, False, prune_imports(select_tasks(body, (ctx.task,) if ctx.task else None,
                                                       rel), rel)
    stem = Path(rel).stem
    if stem == PRELUDE_FILE:
        segs = ("prelude",) + ctx.block_files.get(stem, ())
        return segs, False, prune_imports("\n\n\n".join(segment(path, s) for s in segs), rel)
    segs = ctx.block_files.get(stem, ())
    opening = prelude_imports(segment(path, "prelude"), f"{path.name}: prelude")
    parts = ([opening] if opening else []) + [segment(path, s) for s in segs]
    return segs, True, prune_imports("\n\n\n".join(parts), rel)


def prune_imports(body, where="a splice"):
    """`body` without the top-level imports it never reads: each `import` / `from … import`
    keeps the names the body loads (an annotation counts) and goes when it keeps none."""
    tree = _parse(body, where)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    lines = body.splitlines(keepends=True)
    for node in reversed(tree.body):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or \
                (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            continue
        kept = [a for a in node.names if (a.asname or a.name).split(".")[0] in used]
        if len(kept) == len(node.names):
            continue
        indent = lines[node.lineno - 1][:node.col_offset]
        if not kept:
            text = []
        elif isinstance(node, ast.Import):
            text = [indent + "import " + ", ".join(
                a.name + (f" as {a.asname}" if a.asname else "") for a in kept) + "\n"]
        else:
            module = "." * node.level + (node.module or "")
            text = [l + "\n" for l in _import_lines(module, [
                a.name + (f" as {a.asname}" if a.asname else "") for a in kept], indent)]
        lines[node.lineno - 1:node.end_lineno] = text
    return "".join(lines)


def _splice_bodies(ctx, sources, spliced):
    """(rel, SupportFile) -> _Splice, the raw bodies before the imports are resolved."""
    out = []
    for rel, sf in spliced:
        path = sources.source(sf.source)
        if not path.is_file():
            raise EmitError(f"generation v2.3 splices {rel} from {path}, which does not exist")
        segs, opened, body = _splice_body(rel, sf, path, ctx)
        what = ", ".join((("prelude imports",) if opened else ()) + segs)
        tree = _parse(body, f"{rel} (from {path.name}: {what})")
        out.append(_Splice(rel, path, segs, body, tree, frozenset(defined_names(tree)), opened))
    return out


def _resolve_imports(splices, domain_public, library):
    """rel -> {provider module: names}, plus the cycle check between spliced siblings."""
    by_rel = {s.rel: s for s in splices}
    imports = {}
    for s in splices:
        mod = s.rel.split("/")[0]
        # a block file provides before the builder or the losses: they are one library split
        # in files, and the block names are what the other two read
        siblings = sorted((o for o in splices if o.rel != s.rel and o.rel.split("/")[0] == mod),
                          key=lambda o: Path(o.rel).stem not in BLOCK_FILES)
        free, annotation_only = free_names(s.tree)
        wanted, unresolved = {}, []
        for n in sorted(free | annotation_only):
            owner = next((o for o in siblings if n in o.defines), None)
            if owner is not None:
                wanted.setdefault("." + Path(owner.rel).stem, set()).add(n)
            elif n in domain_public:
                wanted.setdefault("..domain", set()).add(n)
            elif n in free and n not in library:
                unresolved.append(n)
        if unresolved:
            raise EmitError(
                f"{s.rel} (spliced from {s.source.name}: {', '.join(s.segments)}) uses "
                f"{', '.join(unresolved)}, which neither its own imports, domain.py nor a "
                "sibling spliced file defines. A segment carries its own standard-library and "
                "framework imports (at its top, or in the source's `prelude` segment); the "
                "emitter adds only the package-relative ones")
        imports[s.rel] = wanted
    _refuse_cycles({rel: {f"{rel.split('/')[0]}/{m[1:]}.py" for m in w if m != "..domain"}
                    for rel, w in imports.items()}, by_rel)
    return imports


def _refuse_cycles(edges, by_rel):
    state = {}

    def visit(rel, path):
        if state.get(rel) == 1:
            cycle = path[path.index(rel):] + [rel]
            raise EmitError("the spliced files import each other in a cycle ("
                            + " -> ".join(cycle) + "); move the shared definition into the "
                            "file both sides import")
        if state.get(rel) == 2:
            return
        state[rel] = 1
        for nxt in sorted(edges.get(rel, ())):
            if nxt in by_rel:
                visit(nxt, path + [rel])
        state[rel] = 2

    for rel in sorted(edges):
        visit(rel, [])


def _splice_header(package, s, digest):
    dotted = s.rel[:-3].replace("/", ".")
    segments = "Segments: " + ", ".join(s.segments) + "."
    if s.prelude_imports:
        segments += (f" Opens with the imports of the `prelude` segment; the prelude's "
                     f"definitions live in {PRELUDE_FILE}.py, once.")
    segments = textwrap.fill(segments, width=96)
    return (f'"""\n{package}.{dotted} — spliced by /vega-forge from `{s.source.name}`.\n\n'
            f"{segments}\n"
            "Emitted code: a defect is fixed in the source segment and re-emitted, never in "
            "this copy.\nThe package-relative imports are the emitter's, resolved from the "
            "names this file uses.\n\n"
            f"Spec digest: {digest}\n"
            '"""\n'
            "from __future__ import annotations\n\n")


def spliced_files(ctx, sources, spliced, domain_text, digest):
    """rel -> (text, segments) for every spliced support file, imports resolved."""
    domain_tree = _parse(domain_text, "domain.py (rendered)")
    domain_public = {n for n in defined_names(domain_tree) if not n.startswith("_")}
    splices = _splice_bodies(ctx, sources, spliced)
    imports = _resolve_imports(splices, domain_public, _library_names(sources))
    package = ctx.plan["package"]
    out = {}
    for s in splices:
        wanted = imports[s.rel]
        order = sorted(wanted, key=lambda m: (m != "..domain", m))
        block = [line for m in order for line in _import_lines(m, sorted(wanted[m]))]
        text = (_splice_header(package, s, digest)
                + _insert_after_imports(s.body, s.tree, block).rstrip() + "\n")
        _parse(text, s.rel)
        out[s.rel] = (text, s.segments)
    return out


# --------------------------------------------------------------------------- v2.3: custom code

def _resolve_code_ref(code_ref, campaign_root, block_id):
    """The module a `custom` block's code_ref names: an absolute path as is, a relative one
    against the campaign root only — never against the run directory the package is emitted
    into, where a stale copy may lie (netspec-contract §2)."""
    ref = Path(code_ref)
    if ref.is_absolute():
        path = ref
    elif campaign_root is None:
        raise EmitError(f"custom block {block_id}'s code_ref {code_ref!r} is relative and no "
                        "campaign root was given; a code_ref resolves against the campaign root, "
                        "never against a run's own directory (netspec-contract §2) — pass it as "
                        "run_dir (emit.py: --campaign-root)")
    else:
        path = Path(campaign_root) / ref
    if not path.is_file():
        raise EmitError(f"custom block {block_id}'s code_ref {code_ref!r} was not found at "
                        f"{path}; the module is part of the spec's code and ships inside the "
                        "package (R-B9) — put it under the campaign root (netspec-contract §2) "
                        "and emit with that root as run_dir")
    return path


def custom_files(spec, model_mod, campaign_root):
    """[(rel, bytes, map entry)] — each `custom` block's code_ref module, copied (R-B9)."""
    found = {}
    for b in _blocks(spec):
        if b.get("type") != "custom":
            continue
        bid = b.get("id") or "custom"
        code_ref = (b.get("params") or {}).get("code_ref")
        if not code_ref:
            raise EmitError(f"custom block {bid} declares no params.code_ref; the compiler "
                            "(G1d) should have refused it before this point")
        if Path(code_ref).suffix != ".py":
            raise EmitError(f"custom block {bid}'s code_ref {code_ref!r} is not a .py module "
                            "exposing build(in_shape, params) -> module")
        src = _resolve_code_ref(code_ref, campaign_root, bid)
        stem = Path(code_ref).stem
        if stem in found and found[stem]["path"].resolve() != src.resolve():
            raise EmitError(f"custom blocks {found[stem]['ids'][0]} and {bid} name two different "
                            f"modules with the stem {stem!r} ({found[stem]['path']}, {src}); both "
                            f"would be copied to {model_mod}/custom/{stem}.py — rename one")
        found.setdefault(stem, {"path": src, "code_ref": code_ref, "ids": []})["ids"].append(bid)
    out = []
    for stem, f in found.items():
        rel = f"{model_mod}/custom/{stem}.py"
        data = f["path"].read_bytes()
        out.append((rel, data, {
            "file": rel, "module": model_mod, "kind": "support", "used_by": ["Model"],
            "source": "netspec-contract §2 (custom block code_ref, R-B9)",
            "code_ref": f["code_ref"], "blocks": f["ids"],
            "digest": "sha256:" + hashlib.sha256(data).hexdigest()}))
    return out


# --------------------------------------------------------------------------- v2.3: emit

def _blueprint_map_v23(spec, pipeline, plan_, draft):
    """The v2.3 map: the adapters, the SUPPORT_FILES entries (baselines included), the common
    support rows, the custom modules — one entry per written file — and the blueprint's
    artifacts, each held by the domain.py about to be written or declared external."""
    rows = draft.rows
    roster = artifact_roster(pipeline, draft.files["domain.py"])
    entries = [_adapter_row(e) for e in plan_["entries"]]
    entries += [r for r in rows if r.get("role")]
    entries += _common_rows(plan_, roster, draft.files["domain.py"])
    entries += [r for r in rows if not r.get("role")]
    entries += _init_rows(plan_["modules"])
    doc = _map_document(spec, pipeline, plan_["package"], entries, roster)
    doc["generation"] = "v2.3"
    return doc


def _previously_emitted(root):
    """The files an earlier emission into this package wrote, by its own map — the only files
    the emitter may remove. A re-emit after a spec change (a gru swapped for an mlp) would
    otherwise leave `blocks_seq.py` behind, a file no entry maps (C3)."""
    path = root / "blueprint-map.json"
    if not path.is_file():
        return set()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {e["file"] for e in doc.get("entries") or []
            if isinstance(e, dict) and isinstance(e.get("file"), str)}


def _remove_stale(root, stale):
    """Remove those files, and a directory they leave empty (bytecode caches aside)."""
    base = root.resolve()
    for rel in sorted(stale):
        p = (root / rel).resolve()
        if base not in p.parents or not p.is_file():
            continue
        p.unlink()
        for parent in p.parents:
            if parent == base or any(c.name != "__pycache__" for c in parent.iterdir()):
                break
            shutil.rmtree(parent)


class _Draft:
    """The package being emitted, before anything is written: the text (or bytes) of each file
    by package path, the map rows of the support files, the files still to splice and the
    markers the stage prints."""

    def __init__(self):
        self.files, self.rows, self.spliced, self.markers = {}, [], [], []


def _v23_render(draft, ctx, pipeline, src, common):
    """The templated files: domain, ports, the adapters of the drawn ports and the SUPPORT_FILES
    whose `when` holds (the spliced ones are only listed here)."""
    plan_ = ctx.plan
    package = plan_["package"]
    shapes = (ctx.shape,)            # the package carries its spec's sample shape, no other
    tasks = (ctx.task,) if ctx.task else None           # and its task kind's code, no other
    draft.files["domain.py"] = _render(_template(src, "domain.py", "domain.py"), common,
                                       "templates/domain.py", shapes, blueprint_words(pipeline),
                                       tasks=tasks)
    draft.files["ports.py"] = _ports_py(package, pipeline, plan_, draft.markers,
                                        draft.files["domain.py"], ctx.shape, ctx.task)
    draft.files["__init__.py"] = _init_py(package, plan_, pipeline)
    for e in plan_["entries"]:
        draft.files[e["file"]] = _render(_template(src, e["template"], e["file"]), common,
                                         f"templates/{e['template']}", shapes, tasks=tasks)
        for sf in SUPPORT_FILES.get(e["role"], ()):
            if not sf.when(ctx):
                continue
            rel = f"{e['module']}/{sf.name}"
            draft.rows.append(_file_row(sf, rel, e["module"], plan_))
            if sf.source:
                draft.spliced.append((rel, sf))
            else:
                draft.files[rel] = _render(_template(src, sf.template, rel), common,
                                           f"templates/{sf.template}", shapes, tasks=tasks)


def refuse_unresolved(files, spliced=()):
    """Every templated module resolves what it reads once its shape sections are cut: no name
    is read (at run time, or in an annotation) that the file neither binds nor imports, and
    every name it imports from `domain` is one the rendered `domain.py` defines. A section cut
    in the wrong place is refused here, by file and name — never met as an ImportError or a
    NameError inside a bundle. (The spliced files have their own check, `spliced_files`.)"""
    domain = defined_names(_parse(files["domain.py"], "domain.py"))
    problems = []
    for rel, text in sorted(files.items()):
        if not (isinstance(text, str) and rel.endswith(".py")) or rel in spliced:
            continue
        tree = _parse(text, rel)
        free, annotation_only = free_names(tree)
        for name in sorted(free | annotation_only):
            problems.append(f"{rel} reads {name!r}, which it neither binds nor imports")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and \
                    (node.module or "").split(".")[-1] == "domain":
                for alias in node.names:
                    if alias.name != "*" and alias.name not in domain:
                        problems.append(f"{rel} imports {alias.name!r} from domain, which this "
                                        "package's domain.py does not define")
    if problems:
        raise EmitError("the rendered package does not resolve (a shape or compat section cut "
                        "in the wrong place — fix the template): " + "; ".join(problems))


def _v23_splice(draft, ctx, src, digest):
    """The spliced support files, their imports resolved; each map row names its segments."""
    done = spliced_files(ctx, src, draft.spliced, draft.files["domain.py"], digest)
    for row in draft.rows:
        if row["file"] in done:
            draft.files[row["file"]], segs = done[row["file"]]
            row["segments"] = list(segs)


def _v23_finish(draft, ctx, campaign_root):
    """The custom blocks' modules (R-B9) and the modules' `__init__.py` files."""
    plan_ = ctx.plan
    for rel, data, row in custom_files(ctx.spec, plan_["model_module"], campaign_root):
        draft.files[rel] = data
        draft.rows.append(row)
    per_module = {}
    for rel in draft.files:
        if "/" in rel:
            mod, _, inner = rel.partition("/")
            per_module.setdefault(mod, []).append(inner)
    exports = module_exports(plan_, draft.files, draft.rows)
    for mod in plan_["modules"]:
        draft.files[f"{mod}/__init__.py"] = _module_init(plan_["package"], mod,
                                                         per_module.get(mod, []),
                                                         exports.get(mod))


def _v23_write(root, files, bmap):
    """Write the files, then the map, then remove the files the previous map listed and this
    emission does not write. The unmapped guard has passed before this is called, so a refusal
    never leaves a package whose old map lists files already removed."""
    root.mkdir(parents=True, exist_ok=True)
    previous = _previously_emitted(root)
    w = _Writer(root)
    for rel, content in files.items():
        w.write(rel, content)
    w.write("blueprint-map.json", json.dumps(bmap, indent=2, ensure_ascii=False) + "\n")
    _remove_stale(root, previous - set(w.written))
    return w.written


def _emit_v23(spec, pipeline, out, opts):
    src = opts.sources
    coverage = _coverage()
    plan_ = plan(spec, pipeline)
    common = common_mapping(spec, pipeline, plan_)
    ctx = _Ctx(spec, plan_, sample_shape(spec), block_plan(spec, src.blocks),
               effective_task(spec))
    _refuse_port_drift(plan_, ctx.shape, src.templates, ctx.task)
    draft = _Draft()
    _v23_render(draft, ctx, pipeline, src, common)
    refuse_unresolved(draft.files, {rel for rel, _ in draft.spliced})
    _v23_splice(draft, ctx, src, common["SPEC_DIGEST"])
    _v23_finish(draft, ctx, opts.campaign_root)
    bmap = _blueprint_map_v23(spec, pipeline, plan_, draft)
    # the unmapped guard (C3) on what is about to be written, before anything is
    _refuse_unmapped(list(draft.files) + ["blueprint-map.json"], bmap)
    root = out / plan_["package"]
    written = _v23_write(root, draft.files, bmap)
    _write_beside(out, spec, coverage, opts.lrs)
    return {"package": plan_["package"], "root": str(root), "files": sorted(written),
            "blueprint_map": bmap, "modules": plan_["modules"], "markers": draft.markers,
            "backend": NAME, "coverage": coverage, "generation": "v2.3",
            "sample_shape": ctx.shape,
            "block_files": {k: list(v) for k, v in ctx.block_files.items()}}


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--pipeline", required=True, help="the blueprint run's pipeline.json")
    ap.add_argument("--out", required=True, help="the run directory")
    ap.add_argument("--generation", default=EMIT_GENERATION,
                    help=f"the template generation: {', '.join(GENERATIONS)} (the only one; "
                         "v2.2 was removed at gate A)")
    ap.add_argument("--campaign-root", "--run-dir", dest="campaign_root", default=None,
                    help="the campaign root, the one directory a custom block's relative "
                         "code_ref resolves against (netspec-contract §2) — never --out unless "
                         "you pass the same path (the forge stage's run directory IS the "
                         "campaign root; a rung's is not). Default: none, so a spec with a "
                         "relative code_ref is refused until it is given")
    ap.add_argument("--sources", default=None,
                    help="a backend directory (templates/, blocks.py, builder.py, losses.py) "
                         "replacing the backend's own")
    a = ap.parse_args(argv)
    try:
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
        pipeline = json.loads(Path(a.pipeline).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read the spec or the pipeline: {exc}")
        return 2
    try:
        rec = emit_with(spec, pipeline, a.out, EmitOptions(
            generation=a.generation, sources=Sources.at(a.sources) if a.sources else None,
            campaign_root=a.campaign_root))
    except EmitError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"OK: emitted {rec['package']} (generation {rec['generation']}) into {a.out} — "
          f"{len(rec['files'])} file(s) across {len(rec['modules'])} module(s)")
    for m in rec["markers"]:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
