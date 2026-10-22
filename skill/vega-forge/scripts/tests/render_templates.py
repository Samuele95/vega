#!/usr/bin/env python3
"""
render_templates.py — render a template set into an importable package, for unit tests.

    python3 render_templates.py <templates-dir> <out> [--pipeline reference|<path>]
            [--spec reference|row|window|graph|graph-level|<path>] [--full]
            [--only a.py,b.py] [--shapes row,window,graph] [--splice]
            [--sources <backend-dir>]
            [--into <Role>] [--recursive] [--define KEY=VALUE ...]
            [--campaign-root <dir>] [--json] [-v]

"Rendered" in the build plan means this: the templates, with the placeholders a real emission
fills, written into a package `<out>/<package>/` laid out by the v2.3 emit table (D1 §4.1), so
a unit test can import the modules (`sys.path.insert(0, <out>)`) and run the conformance checks
on them without a campaign, a shaper or a blueprint run.

Two modes:

- **partial** (the default) renders the files of the set one by one — every top-level `*.py`,
  or the ones `--only` names — each to the place the v2.3 table gives it (`network.py` ->
  `<model>/network.py`, `frames.py` -> `<ingestion>/frames.py`, ...). A file the table does not
  know goes into the module of `--into <Role>` (the serving templates:
  `--into Predictor --recursive` puts `package/inspection/*.py` under
  `<serving>/inspection/`); without `--into` it is refused, never guessed. The scaffolding an
  import needs is added: `domain.py` (from the set, else from `--sources`'s templates),
  `ports.py`, the `__init__.py` files and `blueprint-map.json` (one entry per file, so C3 holds).
  `--splice` adds the spliced support files (`builder.py`, `blocks_*.py`, `losses.py`) from the
  splice sources in `--sources`. `--shapes` names the sample shapes the rendered files carry
  (the templates' `# >>> shape:` sections kept, `emit.select_shapes`); by default every shape,
  the templates as they stand — so a render that leaves out a shape's support file
  (`windows.py`, `graphs.py`) names the shapes whose files it holds, as the emitter does. Rendering into an existing `<out>` adds to the package there:
  a serving set can be layered over a forge set.
- **full** (`--full`) is exactly the emitter: `emit_with(spec, pipeline, out,
  EmitOptions(generation="v2.3", sources=<the set + the splice sources>))`, so every template
  of D1 §4.1 must be present. It
  takes none of the partial options (`--only`, `--splice`, `--into`, `--recursive`,
  `--define`): given one, it refuses rather than drop it. A custom block's relative `code_ref`
  resolves against `--campaign-root` (default: the directory of a `--spec` file), never
  against `<out>` (netspec-contract §2).

The pipeline `reference` is the forge fixture's world model (`make_fixture.py`): package
`orders_pipeline`, modules ingestion / features / model / training / evaluation / serving, all
eight ports, each port with its `signature[]` and each port-owning module `OOP` — both from the
one world model every fixture renders (`lanekit.pipeline`, BUILD-PLAN R-G2), so a rendered
`ports.py` is the blueprint's, never the emitter's fallback. The specs `row`, `window`, `graph`
(node level) and `graph-level` are small netspecs of each sample shape (D1 §3.2); `reference` is
`row`. `--define KEY=VALUE` supplies a placeholder the set uses and the emitter does not know
(the serving templates' `{{ENV}}`, ...). With `--into <Role>` every file of the set goes into
that role's module, even one whose name the emit table also knows (the serving set's
`graphs.py` is not the ingestion's).

Exit codes: 0 rendered; 2 an input or rendering error (the message names the fix).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, NamedTuple

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
LANEKIT_DIR = SCRIPTS.parent.parent / "vega" / "scripts" / "tests"
if str(LANEKIT_DIR) not in sys.path:
    sys.path.insert(0, str(LANEKIT_DIR))
from lanekit.pipeline import MODULES as _WORLD_MODULES, PORTS as _WORLD_PORTS  # noqa: E402
from lanekit.pipeline import port_signatures as _world_signatures  # noqa: E402

# ------------------------------------------------------------------------ the reference inputs

# the forge fixture's world model (tests/make_fixture.py), without its run-specific digest
REFERENCE_PIPELINE = {
    "system": {"name": "orders-late-pipeline", "package": "orders_pipeline",
               "source": "[spec §1]"},
    "input": {"shape_manifest_digest": "sha256:" + "0" * 64},
    "posture": {"family": "neural", "source": "[analysis:importance]"},
    "serving": {"mode": "batch"},
    "training": {"protocol": "fixed split (the shaper's)"},
    "monitoring": [{"name": "dist_km drift", "source": "[analysis:F3]"}],
    "feature_contract": {"input_dim": 9, "label": "late"},
    "ports": [
        {"name": "DataSource", "role": "DataSource",
         "responsibility": "yield the parts by name; the test part only through the seal",
         "source": "[shape:phase/split]"},
        {"name": "Recipe", "role": "FeatureTransformer",
         "responsibility": "raw -> shaped with the fitted recipe; no fitting at inference",
         "source": "[shape:manifest]", "filled_by": "/vega-pack"},
        {"name": "Model", "role": "Model",
         "responsibility": "features -> outputs; parameters as one value",
         "source": "[geometry:partitions/late]"},
        {"name": "Trainer", "role": "Trainer",
         "responsibility": "fit a Model on a DataSource under a spec; emit an epoch record",
         "source": "[analysis:importance]"},
        {"name": "Evaluator", "role": "Evaluator",
         "responsibility": "one sealed look; metrics with intervals; baselines under one "
                           "protocol", "source": "[analysis:modules/drift]"},
        {"name": "Registry", "role": "ModelStore",
         "responsibility": "checkpoints and specs as versioned artifacts, by digest",
         "source": "[DESIGNER'S CHOICE — reproducibility]"},
        {"name": "Predictor", "role": "Predictor",
         "responsibility": "raw rows -> predictions through the Recipe and the Model",
         "source": "[DESIGNER'S CHOICE — serving]", "filled_by": "/vega-pack"},
        {"name": "Monitor", "role": "Monitor",
         "responsibility": "watch what the analysis found fragile", "source": "[analysis:F3]",
         "filled_by": "/vega-pack"}],
    "modules": [
        {"name": "domain", "paradigm": "FP", "ports_owned": [], "depends_on": []},
        {"name": "ingestion", "paradigm": "FP", "ports_owned": ["DataSource"],
         "depends_on": ["domain"]},
        {"name": "features", "paradigm": "FP", "ports_owned": ["Recipe"],
         "depends_on": ["domain"]},
        {"name": "model", "paradigm": "OOP", "ports_owned": ["Model", "Registry"],
         "depends_on": ["domain"]},
        {"name": "training", "paradigm": "FP", "ports_owned": ["Trainer"],
         "depends_on": ["domain", "model"]},
        {"name": "evaluation", "paradigm": "FP", "ports_owned": ["Evaluator"],
         "depends_on": ["domain", "model"]},
        {"name": "serving", "paradigm": "OOP", "ports_owned": ["Predictor", "Monitor"],
         "depends_on": ["domain", "model"]}],
    "artifacts": [
        {"name": "NetSpec", "kind": "value", "identity": "digest"},
        {"name": "FeatureContract", "kind": "value", "identity": "manifest digest"},
        {"name": "Checkpoint", "kind": "value", "identity": "digest"},
        {"name": "Metrics", "kind": "value", "identity": "run + seal exposure"},
        {"name": "Prediction", "kind": "value", "identity": "none"},
        {"name": "Campaign", "kind": "entity", "identity": "run directory",
         "external": {"owner": "/vega-train",
                      "where": "the campaign directory (campaign.json, runs/<id>/)"}}],
    "anchors": ["EJ-17", "EJ-20", "EJ-2", "EJ-1", "EJ-10", "EJ-15", "EJ-18", "EJ-34", "EJ-64"],
}


def _world_model(pipeline):
    """Each port's `signature[]` and each module's paradigm, from the one world model: the
    ports of a row package of a classification task (the reference spec's); `scoped_pipeline`
    gives a spec of another shape or kind the methods its package carries."""
    by_role = _world_signatures("row", "binary")
    signatures = {name: by_role[role] for name, role, _resp, _src, _sig in _WORLD_PORTS}
    paradigms = {name: paradigm for name, paradigm, _owned, _deps in _WORLD_MODULES}
    for port in pipeline["ports"]:
        port["signature"] = signatures[port["name"]]
    for module in pipeline["modules"]:
        module["paradigm"] = paradigms[module["name"]]
    return pipeline


_world_model(REFERENCE_PIPELINE)


def reference_pipeline():
    return copy.deepcopy(REFERENCE_PIPELINE)


def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric", "origin": "S1"}


def _nominal(name, cardinality):
    return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": cardinality,
            "unseen_category_id": cardinality - 1, "origin": "S2"}


def _spec(family, task, sample, features, blocks, loss, primary):
    return {
        "schema": "vega/netspec@1",
        "input": {"model": "reference.html", "pipeline": "reference/pipeline.json",
                  "manifest_digest": "sha256:" + "0" * 64},
        "seed": 7,
        "task": task,
        "data_contract": {"source": "shape:manifest/output_schema",
                          "parts": {"train": "shaped/train.parquet",
                                    "valid": "shaped/valid.parquet", "test": "sealed"},
                          "features": features,
                          "input_dim": sum(1 for f in features if f["role"] != "key"),
                          "sample": sample},
        "family": {"name": family, "source": "user:reference"},
        "blocks": blocks,
        "loss": loss,
        "metrics": {"primary": primary, "secondary": [], "source": "vega:default"},
        "optimizer": {"name": "adamw", "params": {"lr": 0.001, "weight_decay": 0.0001}},
        "schedule": {"name": "none", "params": {}},
        "regularization": [],
        "early_stopping": {"monitor": f"valid.{primary}", "mode": "max", "patience": 5},
        "training": {"max_epochs": 3, "batch_size": 32},
        "init": {"scheme": "kaiming_uniform", "bias": "zeros"},
        "backend": {"name": "pytorch"},
    }


def _row():
    feats = [_num(f"x{i}") for i in range(4)] + [_nominal("zone", 5)]
    blocks = [
        {"id": "B1", "type": "embedding", "params": {"columns": ["zone"], "dims": {"zone": 3}}},
        {"id": "B2", "type": "concat", "params": {}},
        {"id": "B3", "type": "dense", "params": {"units": 16, "activation": "relu"}},
        {"id": "B4", "type": "dropout", "params": {"p": 0.1}},
        {"id": "B5", "type": "head", "params": {"out": 1, "activation": "none"}}]
    return _spec("mlp-plain",
                 {"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1},
                 {"shape": "row"}, feats, blocks,
                 {"name": "bce_with_logits", "params": {}}, "pr_auc")


def _window():
    feats = [_num("x0"), _num("x1")]
    sample = {"shape": "window", "length": 12, "stride": 1, "horizon": 3,
              "series_by": "series", "order_by": "ts", "context": "previous-part",
              "autoregressive": True, "season": 12}
    blocks = [
        {"id": "B1", "type": "gru", "params": {"hidden": 16, "return": "last"}},
        {"id": "B2", "type": "head", "params": {"out": 3, "activation": "none"}}]
    return _spec("gru", {"kind": "forecast", "label": "target", "horizon": 3}, sample, feats,
                 blocks, {"name": "mse", "params": {}}, "mae")


def _graph(level):
    feats = [_num(f"x{i}") for i in range(3)]
    sample = {"shape": "graph", "edges": "shaped/edges.parquet", "level": level,
              "graph_id": "molecule" if level == "graph" else None,
              "directed": False, "self_loops": False, "protocol": "inductive",
              "batching": "graphs" if level == "graph" else "full", "neighbours": None,
              "exposure": None}
    if level == "graph":
        blocks = [
            {"id": "B1", "type": "gin_conv", "params": {"out": 16}},
            {"id": "B2", "type": "graph_pool", "params": {"readout": "mean"}},
            {"id": "B3", "type": "head", "params": {"out": 1, "activation": "none"}}]
        return _spec("gin", {"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1},
                     sample, feats, blocks, {"name": "bce_with_logits", "params": {}},
                     "roc_auc")
    blocks = [
        {"id": "B1", "type": "gcn_conv", "params": {"out": 16}},
        {"id": "B2", "type": "activation", "params": {"name": "relu"}},
        {"id": "B3", "type": "dropout", "params": {"p": 0.1}},
        {"id": "B4", "type": "gcn_conv", "params": {"out": 16}},
        {"id": "B5", "type": "head", "params": {"out": 4, "activation": "none"}}]
    return _spec("gcn", {"kind": "multiclass", "label": "community",
                         "classes": ["a", "b", "c", "d"]},
                 sample, feats, blocks, {"name": "cross_entropy", "params": {}}, "f1_macro")


REFERENCE_SPECS = {"row": _row, "window": _window, "graph": lambda: _graph("node"),
                   "graph-level": lambda: _graph("graph")}


def reference_spec(shape="row"):
    """A small netspec of one sample shape: row | window | graph | graph-level."""
    if shape == "reference":
        shape = "row"
    if shape not in REFERENCE_SPECS:
        raise ValueError(f"no reference spec {shape!r}; known: reference, "
                         f"{', '.join(REFERENCE_SPECS)}")
    return REFERENCE_SPECS[shape]()


# ------------------------------------------------------------------------ the emitter

def load_emit():
    """`backends/pytorch/emit.py`, imported by path (it is not a package module)."""
    name = "vega_render_emit"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(BACKEND / "emit.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class RenderError(ValueError):
    """Printed as `ERROR:`; the message names the fix."""


# the commit whose emitter the v2.2-equivalence tests compare against (BUILD-PLAN header)
BASELINE_COMMIT = "c849f8b"
REPO = SCRIPTS.parents[1]
BACKEND_PATH = "vega-forge/scripts/backends/pytorch"


def _source_repos():
    """The git checkouts that may hold the baseline commit: this tree, then the repository
    `$VEGA_SOURCE_REPO` names (the lane program runs the tests in a copy of the skills tree,
    which is no git checkout)."""
    return [str(REPO)] + [r for r in [os.environ.get("VEGA_SOURCE_REPO")] if r]


def _git(args, what, text=False):
    failures = []
    for repo in _source_repos():
        try:
            return subprocess.run(["git", "-C", repo] + args, capture_output=True, check=True,
                                  text=text).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            failures.append(f"{repo}: {exc}")
    raise RenderError(f"{what} failed ({'; '.join(failures)}); set VEGA_SOURCE_REPO to a "
                      "checkout that holds the commit")


def baseline_source(relpath, commit=BASELINE_COMMIT):
    """The text of `commit`'s `backends/pytorch/<relpath>` (`losses.py`,
    `templates/data.py`, `templates/metrics.py`, ...) read out of git — the v2.2 generation's
    sources, which the tree no longer carries once gate A removes `generations/v2.2/`. A
    `RenderError` names the checkouts tried when none holds the commit."""
    return _git(["show", f"{commit}:{BACKEND_PATH}/{relpath}"],
                f"git show {commit}:{BACKEND_PATH}/{relpath}", text=True)


def baseline_emitter(work, commit=BASELINE_COMMIT):
    """`commit`'s own `backends/pytorch/emit.py` — the v2.2 layout, which the live emitter no
    longer knows (removed at gate A, WP B14) — read out of git into `work` with today's
    `coverage.json` (the emit table only reads its requirements), and imported. It is what a
    test that proves "the network did not move across the cutover" emits its reference package
    with. The repository is this checkout, else the one `$VEGA_SOURCE_REPO` names (the lane
    program runs the tests in a copy of the skills tree, which is no git checkout); a
    `RenderError` names both when neither holds the commit."""
    ref = Path(work) / f"emit-{commit}"
    blob = _git(["archive", commit, BACKEND_PATH], f"git archive {commit}")
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(ref, filter="data")
    backend = ref / "vega-forge" / "scripts" / "backends" / "pytorch"
    shutil.copyfile(BACKEND / "coverage.json", backend / "coverage.json")
    spec = importlib.util.spec_from_file_location(f"vega_emit_{commit}", str(backend / "emit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_json(arg, what):
    try:
        return json.loads(Path(arg).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"{what} could not be read ({arg}): {exc}") from None


def load_pipeline(arg):
    if arg in (None, "reference"):
        return reference_pipeline()
    return _read_json(arg, "pipeline.json")


def load_spec(arg):
    if arg is None or arg in ("reference",) or arg in REFERENCE_SPECS:
        return reference_spec(arg or "row")
    return _read_json(arg, "the spec")


# ------------------------------------------------------------------------ partial rendering

#: a template that imports another of the set by its module: rendering the first renders the
#: second beside it (the readout reads its numpy half, the metrics' reading of the same kind)
_BROUGHT = {"readout.py": ("metrics.py",)}


def _selected(templates_dir, only, recursive):
    if only:
        names = [n.strip() for n in only if n.strip()]
        names += [b for n in list(names) for b in _BROUGHT.get(n, ())
                  if b not in names and (templates_dir / b).is_file()]
        missing = [n for n in names if not (templates_dir / n).is_file()]
        if missing:
            raise RenderError(f"--only names {', '.join(missing)}, which {templates_dir} does "
                              "not contain")
        return names
    pattern = "**/*.py" if recursive else "*.py"
    return sorted(p.relative_to(templates_dir).as_posix()
                  for p in templates_dir.glob(pattern)
                  if p.is_file() and "__pycache__" not in p.parts)


def _destination(name, layout, plan_, into):
    """(package-relative path, map entry) for one file of the set: the role `--into` names
    when given (it wins over the emit table: a serving set's `graphs.py` is not the
    ingestion's; `domain.py` stays at the package root), else the place the v2.3 table gives
    the name. A file the table places as the adapter of the very role `--into` names stays
    that adapter (`monitor.py --into Monitor` is the Monitor port's file, not a support file)."""
    top = "/" not in name
    known = layout.get(name) if top else None
    if known and (into is None or name == "domain.py" or known[1].get("role") == into):
        return known
    if into is None:
        raise RenderError(
            f"{name} is not a file of the v2.3 emit table (D1 §4.1), so its place in the "
            "package is not known; pass --into <Role> to render it into that role's module")
    mod = plan_["module_names"].get(into)
    if mod is None:
        raise RenderError(f"--into {into}: the pipeline draws no port with that role")
    rel = f"{mod}/{name}"
    return rel, {"file": rel, "module": mod, "kind": "support", "used_by": [into],
                 "source": f"render_templates.py --into {into} (a test render)"}


def _mapping(emit, spec, pipeline, plan_, defines):
    mapping = emit.common_mapping(spec, pipeline, plan_)
    # the one serving-package placeholder that is a pure function of the world model
    mapping["MONITOR_MODULE"] = plan_["module_names"].get("Monitor", "serving")
    mapping.update(defines or {})
    return mapping


def _existing_entries(root):
    path = root / "blueprint-map.json"
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {e["file"]: e for e in doc.get("entries") or [] if e.get("file")}


def _render_one(emit, text, mapping, name, shapes=None):
    try:
        return emit.render_text(text, mapping, name, shapes)
    except emit.EmitError as exc:
        raise RenderError(f"{name}: {exc} — pass an unknown placeholder with --define "
                          "KEY=VALUE") from None


class PartialOptions(NamedTuple):
    """What a partial render takes beside the set, the output, the pipeline and the spec."""

    only: Any = None            # file names of the set (default: every top-level *.py)
    splice: bool = False        # add the spliced support files from `sources`
    sources: Any = None         # a backend directory (default: the live backends/pytorch)
    into: Any = None            # the role whose module takes a file the table does not know
    recursive: bool = False     # also the set's sub-directories
    defines: Any = None         # {placeholder: value} the emitter does not know
    shapes: Any = None          # the sample shapes the files carry (default: every shape)


class _Partial:
    """One partial render in progress: the package's files (by package path) and map entries."""

    def __init__(self, emit, spec, pipeline, plan_, root):
        self.emit, self.spec, self.pipeline, self.plan = emit, spec, pipeline, plan_
        self.root = root
        self.files, self.markers = {}, []
        self.entries = _existing_entries(root)
        self.roster = None              # the artifact roster, once domain.py is known


def render_partial(templates_dir, out, pipeline, spec, opts=PartialOptions()):
    emit = load_emit()
    templates_dir, out = Path(templates_dir), Path(out)
    src = emit.Sources.at(opts.sources) if opts.sources else emit.sources_for("v2.3")
    try:
        plan_ = emit.plan(spec, pipeline)
    except emit.EmitError as exc:
        raise RenderError(str(exc)) from None
    r = _Partial(emit, spec, pipeline, plan_, out / plan_["package"])
    mapping = _mapping(emit, spec, pipeline, plan_, opts.defines)
    _add_templated(r, templates_dir, src, mapping, opts)
    if opts.splice:
        _add_spliced(r, src, mapping["SPEC_DIGEST"])
    return _write_partial(r)


def _add_templated(r, templates_dir, src, mapping, opts):
    """The set's files, each where the v2.3 table puts it; domain.py, ports.py and the package
    `__init__.py` beside them."""
    emit, plan_ = r.emit, r.plan
    layout = emit.v23_layout(plan_)
    layout["domain.py"] = ("domain.py", {})     # its row is the roster's, set below
    for name in _selected(templates_dir, opts.only, opts.recursive):
        rel, row = _destination(name, layout, plan_, opts.into)
        r.files[rel] = _render_one(emit, (templates_dir / name).read_text(encoding="utf-8"),
                                   mapping, name, opts.shapes)
        r.entries[rel] = row
    if "domain.py" not in r.files and not (r.root / "domain.py").is_file():
        domain_src = templates_dir / "domain.py"
        if not domain_src.is_file():
            domain_src = src.templates / "domain.py"
        r.files["domain.py"] = _render_one(emit, domain_src.read_text(encoding="utf-8"),
                                           mapping, "domain.py", opts.shapes)
    # the roster is the blueprint's against the domain.py this package holds (code-contract §3)
    domain_text = r.files.get("domain.py") or (r.root / "domain.py").read_text(encoding="utf-8")
    r.files["ports.py"] = emit.ports_py(plan_["package"], r.pipeline, plan_, r.markers,
                                        domain_text, emit.sample_shape(r.spec))
    r.files["__init__.py"] = emit.init_py(plan_["package"], plan_, r.pipeline)
    try:
        r.roster = emit.artifact_roster(r.pipeline, domain_text)
    except emit.EmitError as exc:
        raise RenderError(str(exc)) from None
    for row in emit._common_rows(plan_, r.roster, domain_text):
        if row["file"] == "domain.py" or row["file"] not in r.entries:
            r.entries[row["file"]] = row


def _add_spliced(r, src, digest):
    emit, plan_ = r.emit, r.plan
    domain_text = r.files.get("domain.py") or (r.root / "domain.py").read_text(encoding="utf-8")
    try:
        ctx = emit._Ctx(r.spec, plan_, emit.sample_shape(r.spec),
                        emit.block_plan(r.spec, src.blocks))
        spliced = []
        for host, supports in emit.SUPPORT_FILES.items():
            mod = plan_["module_names"].get(host)
            for sf in supports:
                if mod is None or not sf.source or not sf.when(ctx):
                    continue
                rel = f"{mod}/{sf.name}"
                spliced.append((rel, sf))
                r.entries[rel] = emit._file_row(sf, rel, mod, plan_)
        done = emit.spliced_files(ctx, src, spliced, domain_text, digest)
    except emit.EmitError as exc:
        raise RenderError(f"--splice: {exc}") from None
    for rel, (text, segs) in done.items():
        r.files[rel] = text
        r.entries[rel]["segments"] = list(segs)


def _write_partial(r):
    emit, package, root, entries = r.emit, r.plan["package"], r.root, r.entries
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in r.files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    per_module = {}
    for rel in entries:
        if "/" in rel and not rel.endswith("/__init__.py"):
            mod, _, inner = rel.partition("/")
            per_module.setdefault(mod, []).append(inner)
    for row in emit._init_rows(sorted(per_module)):
        entries[row["file"]] = row
    # the modules publish what crosses their boundaries, as the emitter's do (the other
    # modules' templates import those names from the module)
    texts = {rel: r.files.get(rel) or (root / rel).read_text(encoding="utf-8")
             for rel in entries if rel in r.files or (root / rel).is_file()}
    exports = emit.module_exports(r.plan, texts, list(entries.values()))
    for mod, inner in per_module.items():
        (root / mod).mkdir(parents=True, exist_ok=True)
        (root / mod / "__init__.py").write_text(
            emit.module_init(package, mod, inner, exports.get(mod)), encoding="utf-8")
    doc = emit._map_document(r.spec, r.pipeline, package, list(entries.values()), r.roster)
    doc["generation"] = "v2.3"
    doc["rendered_by"] = "render_templates.py (partial)"
    (root / "blueprint-map.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    written = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    try:
        emit._refuse_unmapped(written, doc)
    except emit.EmitError as exc:
        raise RenderError(str(exc)) from None
    return {"package": package, "root": str(root), "files": written, "blueprint_map": doc,
            "markers": r.markers, "mode": "partial"}


# ------------------------------------------------------------------------ full rendering

def render_full(templates_dir, out, pipeline, spec, *, sources=None, campaign_root=None):
    """Exactly the emitter, generation v2.3, over the set and the splice sources of `sources`.
    `campaign_root` is what a custom block's relative code_ref resolves against (emit's
    `run_dir`); it never defaults to `out` (netspec-contract §2)."""
    emit = load_emit()
    base = emit.Sources.at(sources) if sources else emit.sources_for("v2.3")
    src = emit.Sources(Path(templates_dir), base.blocks, base.builder, base.losses)
    try:
        rec = emit.emit_with(spec, pipeline, out, emit.EmitOptions(
            generation="v2.3", sources=src, campaign_root=campaign_root))
    except emit.EmitError as exc:
        raise RenderError(str(exc)) from None
    rec["mode"] = "full"
    return rec


# the partial-render options and their flags: --full is exactly the emitter, which takes none
_PARTIAL_FLAGS = {"only": "--only", "splice": "--splice", "into": "--into",
                  "recursive": "--recursive", "defines": "--define", "shapes": "--shapes"}


def render(templates_dir, out, pipeline=None, spec=None, *, full=False, sources=None,
           campaign_root=None, **partial):
    """The library entry: returns the record (package, root, files, blueprint_map, ...).
    `partial` takes the fields of `PartialOptions` (not with `full`)."""
    pipeline = pipeline if isinstance(pipeline, dict) else load_pipeline(pipeline)
    spec = spec if isinstance(spec, dict) else load_spec(spec)
    pipeline = scoped_pipeline(pipeline, spec)
    templates_dir = Path(templates_dir)
    if not templates_dir.is_dir():
        raise RenderError(f"the template set {templates_dir} is not a directory")
    if full:
        given = [_PARTIAL_FLAGS.get(k, k) for k, v in partial.items() if v]
        if given:
            raise RenderError(f"--full is exactly the emitter, which takes no {', '.join(given)}: "
                              "render partially to use them, or drop them")
        return render_full(templates_dir, out, pipeline, spec, sources=sources,
                           campaign_root=campaign_root)
    if campaign_root:
        raise RenderError("--campaign-root applies to --full only: a partial render copies no "
                          "custom block module")
    try:
        opts = PartialOptions(sources=sources, **partial)
    except TypeError as exc:
        raise RenderError(f"unknown render option: {exc}") from None
    return render_partial(templates_dir, out, pipeline, spec, opts)


def scoped_pipeline(pipeline, spec):
    """`pipeline` with each port's `signature[]` and each module's `depends_on` the ones a
    blueprint draws for this spec's package (`shape_pipeline` on its sample shape and task)."""
    emit = load_emit()
    try:
        shape, task = emit.sample_shape(spec), emit.effective_task(spec)
    except Exception:  # noqa: BLE001 — a spec the emitter refuses is refused by the emitter
        return pipeline
    return shape_pipeline(pipeline, shape, task)


def shape_pipeline(pipeline, shape, task=None):
    """`pipeline` drawn for a `shape` package of the `task` kind (None: every kind's methods):
    a port method only some sample shapes or task kinds carry (the emitter's `SCOPED_METHODS`:
    a graph's exposure, a forecast window's seasonal scale, a regression-family head's train
    targets) given to the packages that carry it only, and each module's `depends_on` the arrows
    that shape's code takes (`emit.module_arrows`: a graph Trainer reads the DataSource module's
    views) — the one reference world model every reference spec is rendered against."""
    emit = load_emit()
    out = copy.deepcopy(pipeline)
    for port in out.get("ports") or []:
        if not (isinstance(port, dict) and port.get("signature")):
            continue
        role = port.get("role")
        kept = [m for m in port["signature"]
                if emit._scoped(role, emit._declared(m).name, shape, task)]
        have = {emit._declared(m).name for m in kept}
        kept += [m for m in emit._role_lines(role, None, None)
                 if (role, emit._declared(m).name) in emit.SCOPED_METHODS
                 and emit._declared(m).name not in have
                 and emit._scoped(role, emit._declared(m).name, shape, task)]
        port["signature"] = kept
    arrows = emit.module_arrows(out, shape)
    for module in out.get("modules") or []:
        if isinstance(module, dict) and module.get("name") in arrows:
            module["depends_on"] = list(arrows[module["name"]])
    return out


def _defines(pairs):
    defines = {}
    for d in pairs:
        key, sep, value = d.partition("=")
        if not sep or not key:
            raise RenderError(f"--define {d!r} is not KEY=VALUE")
        defines[key] = value
    return defines


def _campaign_root(a):
    """--campaign-root, else the directory of a --spec file (where the forge writes
    netspec.json: the campaign root), else none (the reference specs have no custom block)."""
    if a.campaign_root:
        return a.campaign_root
    if a.full and a.spec not in ("reference", *REFERENCE_SPECS):
        return str(Path(a.spec).resolve().parent)
    return None


def _parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("templates_dir")
    ap.add_argument("out")
    ap.add_argument("--pipeline", default="reference")
    ap.add_argument("--spec", default="reference")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated file names of the set")
    ap.add_argument("--shapes", default=None,
                    help="comma-separated sample shapes the rendered files carry (default: all)")
    ap.add_argument("--splice", action="store_true")
    ap.add_argument("--sources", default=None,
                    help="a backend directory (templates/, blocks.py, builder.py, losses.py); "
                         "default: vega-forge/scripts/backends/pytorch")
    ap.add_argument("--into", default=None)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--define", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--campaign-root", default=None,
                    help="--full: the directory a custom block's relative code_ref resolves "
                         "against (default: the directory of --spec when it is a file)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print the emitter's markers (ports.py's signature fallbacks)")
    return ap


def main(argv=None):
    a = _parser().parse_args(argv)
    try:
        rec = render(a.templates_dir, a.out, a.pipeline, a.spec, full=a.full,
                     sources=a.sources, campaign_root=_campaign_root(a),
                     only=a.only.split(",") if a.only else None, splice=a.splice,
                     into=a.into, recursive=a.recursive, defines=_defines(a.define),
                     shapes=tuple(a.shapes.split(",")) if a.shapes else None)
    except RenderError as exc:
        print(f"ERROR: {exc}")
        return 2
    if a.json:
        print(json.dumps({k: rec[k] for k in ("package", "root", "files", "mode")}, indent=2))
        return 0
    print(f"OK: rendered {len(rec['files'])} file(s) of {a.templates_dir} into {rec['root']} "
          f"({rec['mode']})")
    for m in (rec.get("markers") or []) if a.verbose else []:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
