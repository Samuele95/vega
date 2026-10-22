"""
predictor_fixtures.py — what test_predictor.py proves the Predictor and the bundle CLI on (build
plan S05; D2 §4.6, §9; R-A2, R-D15).

Not a test file (the unit tier collects test_*.py). It holds:

  packages  `render(package)`: a package rendered by `render_templates.py` as a bundle carries
            it — the Model side (domain, network, readout, assemble, store and the spliced
            blocks of every type the specs below use) and the serving package templates in
            the Predictor's module (`records, guards, heads, values, samples, graphs` and
            `predict.py`, the Predictor adapter, at the port's file), with the Monitor adapter
            beside them.
  data      the shaper's real outputs: a row recipe written here (four numerics, a text zone),
            and samples_fixtures' lagged series and the three graph fixtures, each through
            `shape.py` and its `--transform-only` module.
  cases     one spec per reading over rows (all nine row readings) and per shape (a forecast
            and a sequence task over windows, two node-level graphs, a graph-level set), each
            with the records vega-pack writes for it (`output_record`, `sample_record`,
            `input_schema.derive`), a seeded network and the evaluation's side of the round
            trip: the stored `vega/predictions-layout@1` rows computed from the shaped part
            (rows) or the sealed view (node level), never through the Predictor.
  bundles   `Bundle.build(case, package)`: a bundle directory as pack lays it out — the
            package, the CLI rendered from `bundle/predict.py` (the placeholders serving_pack
            fills), netspec.json, fitted.json, schema.json, the weights saved through the
            rendered store, the reference graph (graph_pack), the monitor reference,
            examples/, MANIFEST.json with `files` and `roles` (bundle_manifest's own) — that
            the CLI loads.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
SERVING_TEMPLATES = TEMPLATES / "serving"
PACKAGE_TEMPLATES = SERVING_TEMPLATES / "package"
SKILLS = SCRIPTS.parent.parent
PACK = SKILLS / "vega-pack" / "scripts"
for _p in (str(SCRIPTS), str(HERE), str(BACKEND), str(PACK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_templates as rt  # noqa: E402
from port_fixtures import PortOnly, model_port, port_only  # noqa: E402,F401
import samples_fixtures as sf  # noqa: E402  (the shaper runs, the window and graph specs)
import bundle_manifest  # noqa: E402  (vega-pack: file roles and pins, as pack writes them)
import bundle_slice  # noqa: E402  (vega-pack: the module of each role, for the CLI)
import graph_pack  # noqa: E402
import input_schema  # noqa: E402
import output_record as orec  # noqa: E402
from pack_context import Campaign  # noqa: E402
from vega_pack_core import Tolerance  # noqa: E402

_WORK = tempfile.TemporaryDirectory(prefix="predictor-")
WORK = Path(_WORK.name)
MODEL_SIDE = ("domain.py", "network.py", "readout.py", "assemble.py", "store.py", "preprocess.py")
SUPPORT_SET = ("records.py", "guards.py", "heads.py", "values.py", "samples.py")
PROB = {"applies": True, "value": 0.41, "scale": "probability", "source": "valid-optimum"}
PER_LABEL = {"applies": True, "value": 0.5, "scale": "probability", "source": "valid-optimum",
             "per_label": {"red": 0.3, "green": 0.6}}


# --------------------------------------------------------------------------- the package

def _wide_spec():
    """A spec whose blocks splice every segment the networks below build from."""
    s = rt.reference_spec("graph")
    s["blocks"] = [{"id": f"W{i}", "type": t, "params": {}} for i, t in enumerate(
        ("embedding", "concat", "dense", "dropout", "gru", "gcn_conv", "sage_conv", "gat_conv",
         "gin_conv", "graph_pool", "activation", "head"))]
    return s


def _pipeline(package):
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = package
    return pipeline


def _monitor_template():
    """The Monitor adapter a bundle of this tree carries: the serving package's (S06) when the
    tree has it, else the completed v2.2 one pack renders until then."""
    own = PACKAGE_TEMPLATES / "monitor.py"
    return (PACKAGE_TEMPLATES, own.name) if own.is_file() else (SERVING_TEMPLATES, "monitor.py")


def render(package) -> Path:
    """`WORK/pkgs/<package>/<package>`: the Model side and the serving set, rendered."""
    out, spec, pipeline = WORK / "pkgs" / package, _wide_spec(), _pipeline(package)
    shutil.rmtree(out, ignore_errors=True)
    rt.render(TEMPLATES, out, pipeline, spec, only=list(MODEL_SIDE), splice=True)
    rec = rt.render(PACKAGE_TEMPLATES, out, pipeline, spec, only=list(SUPPORT_SET),
                    into="Predictor")
    # the two adapters where the emit table puts the Predictor and the Monitor ports' files,
    # with their adapter rows (C14 reads those)
    rt.render(PACKAGE_TEMPLATES, out, pipeline, spec, only=["predict.py"])
    folder, name = _monitor_template()
    rt.render(folder, out, pipeline, spec, only=[name])
    # the serving set's graphs.py shares its name with the ingestion's graph adapter of the
    # emit table: rendered with the emitter's own mapping into the Predictor's module and
    # mapped there, as pack renders `package/*.py` (samples_fixtures' way)
    emit, root = rt.load_emit(), Path(rec["root"])
    mapping = emit.common_mapping(spec, pipeline, emit.plan(spec, pipeline))
    text = (PACKAGE_TEMPLATES / "graphs.py").read_text(encoding="utf-8")
    (root / "serving" / "graphs.py").write_text(emit.render_text(text, mapping), encoding="utf-8")
    bmap = json.loads((root / "blueprint-map.json").read_text(encoding="utf-8"))
    row = next(e for e in bmap["entries"] if e["file"] == "serving/samples.py")
    bmap["entries"].append(dict(row, file="serving/graphs.py"))
    (root / "blueprint-map.json").write_text(json.dumps(bmap, indent=2) + "\n", encoding="utf-8")
    return root


class Package:
    """A rendered package, imported: its modules by the names the tests read them by."""

    def __init__(self, name):
        self.name, self.root = name, render(name)
        sys.path.insert(0, str(self.root.parent))
        self.domain = self._module("domain")
        self.network = self._module("model.network")
        self.store = self._module("model.store")
        self.records = self._module("serving.records")
        self.samples = self._module("serving.samples")
        self.values = self._module("serving.values")
        self.graphs = self._module("serving.graphs")
        self.predict = self._module("serving.predict")

    def _module(self, rel):
        return importlib.import_module(f"{self.name}.{rel}")

    def build(self, spec):
        """A seeded network of `spec`, in eval mode, as the store would restore it."""
        torch.manual_seed(0)
        return self.network.Network.build(self.domain.NetSpec.from_dict(spec)).eval()

    def save(self, model, path, spec):
        """The weights through the rendered ModelStore adapter (the class of v2.3, the module
        functions before it); returns the checkpoint value."""
        registry = getattr(self.store, "ModelRegistry", None)
        saver = registry().save if registry is not None else self.store.save
        return saver(model, path, spec=self.domain.NetSpec.from_dict(spec))


@functools.lru_cache(maxsize=None)
def package(name="s05_pkg") -> Package:
    return Package(name)


# --------------------------------------------------------------------------- the row data

ZONES = ("n", "e", "w")


def _rows_recipe() -> Path:
    """Four numerics and a text zone (three values), a numeric label; the recipe splits at
    random, encodes the zone ordinally and scales the numerics."""
    rng = np.random.default_rng(3)
    n = 240
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) * (i + 1) for i in range(4)})
    frame["zone"] = rng.choice(ZONES, size=n)
    frame["y"] = rng.normal(size=n)
    csv = WORK / "rows.csv"
    frame.to_csv(csv, index=False)

    def step(sid, op, columns, params):
        return {"id": sid, "op": op, "columns": columns, "params": params, "source": "user:test",
                "rationale": f"test step {op}", "alternatives": [],
                "consequences": {"rows": "none", "columns": "none", "downstream": {}},
                "reversible": True}
    recipe = {"schema": "dataset-shaper/recipe@1",
              "input": {"path": str(csv), "rows": n, "columns": 6, "layers_read": [],
                        "digest": "sha256:" + hashlib.sha256(csv.read_bytes()).hexdigest()},
              "seed": 7, "goal": "custom", "target": "y",
              "outputs": {"dir": "shaped/", "format": "parquet", "files": [],
                          "recipe": "shaped/recipe.json", "manifest": "shaped/manifest.json",
                          "lineage": "shaped/lineage.json",
                          "reproduce": "shaped/reproduce_rows.py"},
              "steps": [step("S1", "select_partition", [], {"label": "y", "task": "regression"}),
                        step("S2", "split", [], {"kind": "random", "test": 0.25,
                                                 "valid": 0.15}),
                        step("S3", "encode", ["zone"], {"strategy": "ordinal"}),
                        step("S4", "scale", [], {"kind": "standard",
                                                 "columns": ["x0", "x1", "x2", "x3"]})]}
    path = WORK / "rows.recipe.json"
    path.write_text(json.dumps(recipe, indent=1), encoding="utf-8")
    return path


@functools.lru_cache(maxsize=None)
def shaped(name):
    """The shaper's outputs: `rows` (written here) or one of samples_fixtures' fixtures. The
    transform module reads its fitted parameters at import, from VEGA_FITTED when set, else
    beside itself: it is imported with the variable unset (a bundle's CLI loaded earlier in
    this process leaves it pointing at that bundle's parameters)."""
    saved = os.environ.pop("VEGA_FITTED", None)
    try:
        return sf.Shaped("rows", _rows_recipe()) if name == "rows" else sf.shaped(name)
    finally:
        if saved is not None:
            os.environ["VEGA_FITTED"] = saved


def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric", "origin": "S4"}


def _zone():
    return {"name": "zone", "dtype": "int64", "role": "nominal", "cardinality": len(ZONES) + 1,
            "unseen_category_id": len(ZONES), "origin": "S3"}


ROW_FEATURES = [_num(f"x{i}") for i in range(4)] + [_zone()]
RECON_DIM = 4 + len(ZONES) + 1          # the numerics and one output per zone id (recon_dim)


def _row_spec(task, head, loss):
    s = rt.reference_spec("row")
    s["family"]["name"] = "user-defined"
    s["task"] = task
    s["data_contract"].update(features=ROW_FEATURES, input_dim=len(ROW_FEATURES),
                              sample={"shape": "row"})
    s["blocks"] = [{"id": "B1", "type": "embedding",
                    "params": {"columns": ["zone"], "dims": {"zone": 2}}},
                   {"id": "B2", "type": "concat", "params": {}},
                   {"id": "B3", "type": "dense", "params": {"units": 8, "activation": "relu"}},
                   {"id": "H", "type": "head", "params": {"out": head, "activation": "none"}}]
    s["loss"] = loss if isinstance(loss, dict) else {"name": loss, "params": {}}
    return s


def _quantile(qs):
    return {"name": "quantile", "params": {"quantiles": qs}}


# reading -> (task, head_out, loss, threshold.json); the threshold "median" is the median of
# the evaluation's own per-row scores (so both decisions occur)
ROW_READINGS = {
    "binary": ({"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1}, 1,
               "bce_with_logits", PROB),
    "multiclass": ({"kind": "multiclass", "label": "y", "classes": ["a", "b", "c"]}, 3,
                   "cross_entropy", None),
    "multilabel": ({"kind": "multilabel", "label": "y", "classes": ["red", "green", "blue"],
                    "label_format": {"kind": "delimited", "sep": "|"}}, 3, "bce_with_logits",
                   PER_LABEL),
    "ordinal": ({"kind": "ordinal", "label": "y", "classes": ["low", "mid", "high", "top"]}, 3,
                "cumulative_link", None),
    "regression": ({"kind": "regression", "label": "y"}, 1, "mse", None),
    "quantile": ({"kind": "regression", "label": "y"}, 3, _quantile([0.1, 0.5, 0.9]), None),
    "reconstruction": ({"kind": "unsupervised"}, RECON_DIM, "reconstruction", None),
    "anomaly-reconstruction": ({"kind": "anomaly", "label": "y", "classes": [0, 1],
                                "positive": 1}, RECON_DIM, "reconstruction", "median"),
    "anomaly-svdd": ({"kind": "anomaly"}, 5, "svdd", "median"),
}


# --------------------------------------------------------------------------- the evaluation side

def _stored(model, batch, row_ids):
    """`vega/predictions-layout@1` for one batch, as the evaluator writes it: the raw head
    (`score` / `score_<k>`) and the readout's per-row score (the decision is `_decided`'s)."""
    readout = model.readout
    with torch.no_grad():
        out = model(batch)
        row = readout.row_score(out, batch, model)
    raw = out.detach().double().numpy().reshape(len(row_ids), -1)
    stored = pd.DataFrame({"row_id": np.asarray(row_ids)})
    if raw.shape[1] == 1:
        stored["score"] = raw[:, 0].astype(np.float32)
    else:
        for k in range(raw.shape[1]):
            stored[f"score_{k}"] = raw[:, k].astype(np.float32)
    if row is not None:
        stored["row_score"] = row.detach().double().numpy().astype(np.float32)
    return stored


def _decided(stored, reading, threshold):
    """The evaluation's own stored decision of a threshold reading (sigmoid of the logit on
    the probability scale, the per-row score on the score scale)."""
    if threshold is None or not threshold.get("applies", True):
        return stored
    value = float(threshold["value"])
    if reading == "binary":
        p = 1.0 / (1.0 + np.exp(-stored["score"].to_numpy(np.float64)))
        return stored.assign(prediction=(p >= value).astype(np.int64))
    if reading.startswith("anomaly-"):
        return stored.assign(prediction=(stored["row_score"].to_numpy(np.float64) >= value)
                             .astype(np.int64))
    return stored


def _row_batch(part):
    """The shaped test part as the evaluation reads it: the contract's numerics (float32) and
    nominal ids (int64), in the contract's order."""
    names = [f["name"] for f in ROW_FEATURES if f["role"] != "nominal"]
    return {"x_num": torch.tensor(part[names].to_numpy(np.float32, copy=True)),
            "x_cat": torch.tensor(part[["zone"]].to_numpy(np.int64, copy=True))}


# --------------------------------------------------------------------------- cases

@dataclass
class Case:
    """One reading or shape: the spec, the shaped data, the model, the records, the request
    rows and what the evaluation stored for them."""

    name: str
    spec: dict
    shaped: object
    model: object
    threshold: dict | None
    manifest: dict                       # {"output", "sample"} as pack writes them
    schema: dict
    rows: pd.DataFrame                   # the raw request rows (the test part, label dropped)
    stored: pd.DataFrame | None          # the evaluation's layout for `rows`, row for row
    context: object = None               # a graph case's GraphContext
    graph_root: Path | None = None

    @property
    def reading(self):
        return self.manifest["output"]["reading"]

    def records(self, pkg, schema=None):
        return pkg.records.ServingRecords.of(self.spec, self.manifest,
                                             self.schema if schema is None else schema,
                                             "sha256:" + "5a" * 32)

    def predictor(self, pkg, model=None):
        """The case's Predictor, its network handed over as the Model port alone."""
        return pkg.predict.Predictor(port_only(self.model if model is None else model),
                                     self.shaped.module.transform, self.records(pkg),
                                     self.context)


def _campaign(s, spec):
    return Campaign(dir=WORK, doc={}, ledger={}, seal={}, run_id="R0", eval_dir=WORK,
                    metrics={}, threshold=None, shaped_dir=s.dir, shaped_manifest=s.manifest,
                    recipe=s.recipe, spec=spec, raw=s.raw, fitted=s.fitted)


def _schema(s, spec, sample):
    """schema.json's input half, as pack derives it (its OK lines kept off the test output)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return input_schema.derive(_campaign(s, spec), sample, [])


def _request(s, schema):
    """The test part's raw rows as a caller sends them: the ignored columns dropped, but for
    the raw key columns a graph recipe attaches a query by (`GRAPH["query_columns"]`: the
    schema lists them as ignored identities until input_schema serves them, W2's open need)."""
    rows = s.rows("test")
    keys = set((getattr(s.module, "GRAPH", None) or {}).get("query_columns") or ())
    return rows.drop(columns=[c for c in schema.get("ignored") or ()
                              if c in rows.columns and c not in keys])


def _median_threshold(stored):
    return {"applies": True, "value": float(np.median(stored["row_score"].to_numpy(np.float64))),
            "scale": "score", "source": "valid-optimum"}


def _svdd_centre(model):
    centre = model.objective_center()
    with torch.no_grad():
        centre.copy_(torch.linspace(-0.5, 0.5, centre.numel()))
        model.svdd_center_set.fill_(True)


@functools.lru_cache(maxsize=None)
def row_case(reading) -> Case:
    pkg, s = package(), shaped("rows")
    task, head, loss, threshold = ROW_READINGS[reading]
    spec = _row_spec(task, head, loss)
    model = pkg.build(spec)
    if reading == "anomaly-svdd":
        _svdd_centre(model)
    part = s.part("test")
    stored = _stored(model, _row_batch(part), np.arange(len(part)))
    threshold = _median_threshold(stored) if threshold == "median" else threshold
    stored = _decided(stored, orec.resolve_reading(spec), threshold)
    sample = orec.sample_record(spec, s.recipe)
    manifest = {"output": orec.output_record(spec, threshold, task.get("label")),
                "sample": sample}
    schema = _schema(s, spec, sample)
    return Case(reading, spec, s, model, threshold, manifest, schema, _request(s, schema),
                stored)


def _window(kind):
    if kind != "sequence-binary":
        return sf.window_spec(length=4, autoregressive=kind == "forecast", kind=kind)
    spec = sf.window_spec(length=4, autoregressive=False, kind="sequence-classification")
    spec["task"] = {"kind": "sequence-classification", "base_kind": "binary", "label": "target",
                    "classes": [0, 1], "positive": 1}
    spec["loss"] = {"name": "bce_with_logits", "params": {}}
    return spec


WINDOW_THRESHOLDS = {"sequence-binary": PROB}


@functools.lru_cache(maxsize=None)
def window_case(kind="forecast") -> Case:
    """A window bundle over the lagged series: a forecast (autoregressive, H=2), a sequence
    task with a regression base, or one with a binary base (a decision on windows). Its
    `stored` is None: the served windows are proved against the network on the prepared batch
    and against the answered rule (S04 proves the batch is the evaluation's)."""
    pkg, s = package(), shaped("series")
    spec = _window(kind)
    sample = orec.sample_record(spec, s.recipe)
    threshold = WINDOW_THRESHOLDS.get(kind)
    manifest = {"output": orec.output_record(spec, threshold, "target"), "sample": sample}
    schema = _schema(s, spec, sample)
    return Case(f"window-{kind}", spec, s, pkg.build(spec), threshold, manifest, schema,
                _request(s, schema), None)


def _graph_root(name, spec):
    root = WORK / "graph-roots" / name
    shutil.rmtree(root, ignore_errors=True)
    s = shaped(name)
    meta = graph_pack.write_reference(root, s.sources(spec))
    return root, meta


def _sealed_node_view(s, spec):
    """The evaluation's sealed test view (reference train ∪ valid, queries attached reference
    -> query) as the evaluator scores it: the domain's GraphSpec.view over the shaped parts
    and the edges part — the batch the stored predictions of a node-level test came from."""
    pkg = package()
    order = [np.asarray(s.split[p], dtype=np.int64) for p in ("train", "valid", "test")]
    ids = pd.Series(np.arange(sum(map(len, order))), index=np.concatenate(order))
    part = pd.read_parquet(s.dir / Path(s.manifest["outputs"]["files"]["edges"]["path"]).name)
    weight = part["weight"].to_numpy(dtype="float64", na_value=np.nan)
    edges = pkg.domain.GraphEdges(src=ids.loc[part["src_pos"]].to_numpy(),
                                  dst=ids.loc[part["dst_pos"]].to_numpy(),
                                  weight=None if np.isnan(weight).all() else weight)
    view = pkg.domain.GraphSpec.from_spec(spec).view(tuple(map(len, order)), edges, "test")
    names = [f["name"] for f in spec["data_contract"]["features"]]
    x = pd.concat([s.part(p) for p in ("train", "valid", "test")], ignore_index=True)
    batch = {"x_num": torch.as_tensor(x[names].to_numpy(np.float32)[view.node_ids]),
             "edge_index": torch.as_tensor(np.asarray(view.edge_index, dtype=np.int64)),
             "target_index": torch.as_tensor(np.asarray(view.target_index, dtype=np.int64))}
    if view.edge_weight is not None:
        batch["edge_weight"] = torch.as_tensor(np.asarray(view.edge_weight, dtype=np.float32))
    return batch


GRAPH_THRESHOLDS = {"graph-keypair": PROB, "graph-knn": None, "graph-level": PROB}


@functools.lru_cache(maxsize=None)
def graph_case(name) -> Case:
    """A graph bundle: `graph-keypair` (binary, node level), `graph-knn` (regression, node
    level), `graph-level` (binary, graph level). A node-level case's `stored` is the network
    on the sealed test view (R-A2: the network gathers the targets once)."""
    pkg, s, spec = package(), shaped(name), sf.GRAPHS[name]()
    threshold = GRAPH_THRESHOLDS[name]
    root, meta = _graph_root(name, spec)
    context = pkg.graphs.GraphContext.load(root, s.module,
                                           pkg.domain.NetSpec.from_dict(spec).feature_contract)
    model = pkg.build(spec)
    stored = None
    if name != "graph-level":
        n = len(s.split["test"])
        stored = _decided(_stored(model, _sealed_node_view(s, spec), np.arange(n)),
                          orec.resolve_reading(spec), threshold)
    sample = orec.sample_record(spec, s.recipe, meta)
    label = spec["task"]["label"]
    manifest = {"output": orec.output_record(spec, threshold, label), "sample": sample}
    schema = _schema(s, spec, sample)
    return Case(name, spec, s, model, threshold, manifest, schema, _request(s, schema), stored,
                context, root)


# --------------------------------------------------------------------------- bundles

def _json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _cli_text(pkg_dir, package_name, env) -> str:
    """`bundle/predict.py` with the placeholders serving_pack fills (`{package}`, `{env}`,
    `{mod_<role>}` from the package's own map)."""
    text = (SERVING_TEMPLATES / "bundle" / "predict.py").read_text(encoding="utf-8")
    text = text.replace("{package}", package_name).replace("{env}", env)
    for key, module in bundle_slice.bundle_modules(pkg_dir).items():
        text = text.replace("{mod_" + key + "}", module)
    return text


@dataclass
class Bundle:
    """A bundle directory as pack lays it out, for one case, under its own package name (one
    bundle per process: the Recipe reads its fitted parameters at import)."""

    root: Path
    package: str
    case: Case

    @staticmethod
    def build(case: Case, name: str, examples: str = "row") -> "Bundle":
        """`examples`: `row` (v2.3: `__row__` with the compared and decided fields),
        `positional` (a v2.2-shaped set: the compared head columns, row for row) or
        `window` (context rows first, `__row__` at the answered rows)."""
        root = WORK / "bundles" / name
        shutil.rmtree(root, ignore_errors=True)
        pkg_dir = render(name)
        shutil.copytree(pkg_dir, root / name)
        shutil.copyfile(case.shaped.module_path, root / name / "features" / "preprocess.py")
        shutil.copyfile(case.shaped.dir / "fitted.json", root / "fitted.json")
        (root / "predict.py").write_text(_cli_text(root / name, name, "S05TEST"),
                                         encoding="utf-8")
        _json(root / "netspec.json", case.spec)
        _json(root / "schema.json", case.schema)
        if case.graph_root is not None:
            shutil.copytree(case.graph_root / "graph", root / "graph")
        bundle = Bundle(root, name, case)
        version = bundle._weights()
        bundle._examples(examples)
        _json(root / "monitor-reference.json", bundle._monitor_reference())
        bundle.write_manifest(version)
        return bundle

    def _weights(self) -> str:
        """The case's model saved through this bundle's own store; the weights digest."""
        sys.path.insert(0, str(self.root))
        try:
            store = importlib.import_module(f"{self.package}.model.store")
            domain = importlib.import_module(f"{self.package}.domain")
            registry = getattr(store, "ModelRegistry", None)
            saver = registry().save if registry is not None else store.save
            ref = saver(self.case.model, self.root / "weights.pt",
                        spec=domain.NetSpec.from_dict(self.case.spec))
        finally:
            sys.path.remove(str(self.root))
            for mod in [m for m in sys.modules if m.split(".")[0] == self.package]:
                del sys.modules[mod]
        return ref.weights_digest

    def _examples(self, kind) -> None:
        ex = self.root / "examples"
        ex.mkdir(parents=True, exist_ok=True)
        rows, expected = EXAMPLES[kind](self.case)
        rows.to_csv(ex / "rows.csv", index=False)
        expected.to_parquet(ex / "expected.parquet", index=False)

    def _monitor_reference(self) -> dict:
        """One psi check on the first numeric input, measured on the train part as pack
        measures it (a reference both the v2.2 and the v2.3 Monitor read)."""
        name = next(c["name"] for c in self.case.schema["input"]
                    if c.get("dtype") != "string" and c.get("role") != "autoregressive")
        train = pd.to_numeric(self.case.shaped.rows("train")[name], errors="coerce").dropna()
        edges = np.quantile(train, np.linspace(0, 1, 6))
        edges[0], edges[-1] = -1e12, 1e12
        counts = np.histogram(train, bins=edges)[0]
        return {"schema": "vega/monitor-reference@1", "concerns": [], "plan": "test",
                "checks": [{"id": "p1", "kind": "psi", "column": name, "bins": edges.tolist(),
                            "reference": (counts / counts.sum()).tolist(), "min_rows": 10,
                            "concern": "[test]"}], "note": ""}

    def write_manifest(self, version=None, changes=None) -> dict:
        """MANIFEST.json over the files on disk now: `files` and `roles` as bundle_manifest
        writes them, the output and sample records, the round-trip tolerance."""
        path = self.root / "MANIFEST.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        files = bundle_manifest.pinned_files(self.root)
        tolerance = Tolerance.for_scores(_scores(self.case)).as_json()
        doc = {"schema": "vega/bundle@1", "package": self.package,
               "input_schema": "schema.json",
               "model_version": version or old.get("model_version"),
               "output": self.case.manifest["output"], "sample": self.case.manifest["sample"],
               "roundtrip": {"tolerance": tolerance}, "files": files,
               "roles": bundle_manifest.file_roles(files, self.package)}
        doc.update(changes or {})
        _json(path, doc)
        return doc


def _scores(case):
    if case.stored is None:
        return [1.0]
    cols = [c for c in case.stored.columns if c.startswith("score") or c == "row_score"]
    return case.stored[cols].to_numpy().ravel().tolist()


def _row_examples(case, n=12):
    """v2.3: the first n test rows, `__row__`, and the compared and decided fields the
    evaluation's stored rows give (`reader.from_stored`), not this build's answers."""
    pkg = package()
    reader = pkg.values.reader_for(case.records(pkg))
    theirs = reader.from_stored(case.stored.head(n)).reset_index(drop=True)
    out = case.manifest["output"]
    keep = list(dict.fromkeys([f for f, _ in out["compared"]] + [f for f, _ in out["decided"]]))
    expected = theirs[keep].assign(__row__=np.arange(n))
    return case.rows.head(n).reset_index(drop=True), expected


def _positional_examples(case, n=12):
    """A v2.2-shaped set (examples_pack before S10): no `__row__`; the compared head columns
    read off the stored scores, row for row (and a class reading's `label`)."""
    rows, expected = _row_examples(case, n)
    out = case.manifest["output"]
    cols = [f for f, _ in out["compared"]] + (["label"] if "label" in expected else [])
    return rows, expected[cols]


def _window_examples(case, n=8):
    """Context rows first: each series' last `length − 1` valid rows, then its first test
    rows; `__row__` names the answered rows, the expected values the network's on the
    prepared batch (the served path's own batch; S04 proves it is the evaluation's)."""
    pkg, s = package(), case.shaped
    past = pd.concat([s.rows("train"), s.rows("valid")])
    station = "A"
    context = past[past["station"] == station].sort_values("ts").tail(3)
    test = case.rows[case.rows["station"] == station].sort_values("ts").head(n)
    rows = pd.concat([context.drop(columns=[c for c in context.columns
                                            if c not in test.columns]), test],
                     ignore_index=True)
    predictor = case.predictor(pkg)
    prepared = predictor.prepare(rows)
    answered = np.flatnonzero(prepared.sample_of_row >= 0)
    with torch.no_grad():
        out = case.model(prepared.batch)
    reader = predictor.reader
    theirs = reader.frame(reader.head(out, prepared.batch, case.model))
    at = prepared.sample_of_row[answered]
    fields = [f for f, _ in case.manifest["output"]["compared"]]
    expected = theirs.iloc[at][fields].reset_index(drop=True).assign(__row__=answered)
    return rows, expected


EXAMPLES = {"row": _row_examples, "positional": _positional_examples,
            "window": _window_examples}


def load_cli(bundle: Bundle):
    """The bundle's `predict.py`, imported as a user would from the bundle directory (under a
    module name of its own, so several bundles share this process). The CLI serves one bundle
    per process and says so when VEGA_FITTED names another's parameters; each test bundle has
    its own package, so the variable is pointed back at this bundle's before it is used."""
    os.environ["VEGA_FITTED"] = str((bundle.root / "fitted.json").resolve())
    name = f"predict_{bundle.package}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, bundle.root / "predict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
