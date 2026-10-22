"""
_inspector_fixtures.py — what test_inspect_inspector.py proves the inspector facade on (build
plan I06; inspect-contract §5.3–§6.3; D3 §9, §10.3; R-D15).

Not a test file (the unit tier collects test_*.py). It holds:

  package   `package()`: a package rendered as a bundle carries it — predictor_fixtures' Model
            side and serving set (the Predictor at the port's file, its support files) with
            every template of `serving/package/inspection/` rendered into the Predictor's
            module (`inspection/__init__.py` included), as pack renders `package/inspection/*`.
  cases     one Predictor per sample shape over the shaper's real outputs: rows through an
            `mlp-plain` network for each of the nine row readings, windows through an `lstm`
            (an autoregressive forecast and a sequence task with a binary base), a node-level
            graph through a `gcn` (binary) and a graph-level set through a `gin` (binary). Each
            case's records carry what pack writes (`output_record`, `sample_record`, the input
            schema), the model version and the round-trip tolerance.
  bundles   `bundle(case, name)`: predictor_fixtures' bundle directory for a case, with the
            inspection modules rendered into its package and MANIFEST.json re-pinned — what
            `predict.load_inspector` opens.
  schemas   `EXPLANATION`, `NETWORK` (the K1 schema files) and `TABLE`, the
            `vega/attribution-table@1` body of inspect-contract §5.5 as a JSON Schema, written
            here from the contract's text (no schema file carries it yet).
"""
from __future__ import annotations

import copy
import functools
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import predictor_fixtures as pf  # noqa: E402  (the rendered packages, the shaped data, bundles)
import samples_fixtures as sf  # noqa: E402  (the window and graph specs)
import graph_pack  # noqa: E402  (vega-pack, on sys.path through predictor_fixtures)
import output_record as orec  # noqa: E402
import render_templates as rt  # noqa: E402
from vega_pack_core import Tolerance  # noqa: E402

SCRIPTS = pf.SCRIPTS
PACKAGE_TEMPLATES = pf.PACKAGE_TEMPLATES
INSPECTION = PACKAGE_TEMPLATES / "inspection"
SCHEMAS = SCRIPTS.parent / "references" / "schemas"
EXPLANATION = json.loads((SCHEMAS / "explanation-2.schema.json").read_text(encoding="utf-8"))
NETWORK = json.loads((SCHEMAS / "network-view-2.schema.json").read_text(encoding="utf-8"))
MODEL_VERSION = "sha256:" + "6e" * 32
PACKAGE = "i06_pkg"


# --------------------------------------------------------------------------- the package

def inspection_names() -> list:
    """Every inspection template, package-relative (`inspection/<name>.py`)."""
    return sorted(f"inspection/{p.name}" for p in INSPECTION.glob("*.py"))


BLOCK_TYPES = ("embedding", "concat", "dense", "activation", "dropout", "lstm", "gru", "pool",
               "edge_dropout", "gcn_conv", "sage_conv", "gat_conv", "gin_conv", "graph_pool",
               "head")


def wide_spec() -> dict:
    """A spec whose blocks splice every segment the networks below build from."""
    s = rt.reference_spec("graph")
    s["blocks"] = [{"id": f"W{i}", "type": t, "params": {}} for i, t in enumerate(BLOCK_TYPES)]
    return s


def add_inspection(pkg_root: Path, package_name: str) -> None:
    """The inspection templates rendered into `<pkg_root>/serving/inspection/` (mapped as
    support files of the Predictor), as pack's `render_package` puts `package/inspection/*`."""
    rt.render(PACKAGE_TEMPLATES, Path(pkg_root).parent, pf._pipeline(package_name),
              wide_spec(), only=inspection_names(), into="Predictor")


def render(package_name) -> Path:
    """predictor_fixtures' package (the Model side, the serving set, the Monitor) over the
    blocks of `BLOCK_TYPES`, with the inspection modules."""
    out, spec, pipeline = pf.WORK / "i06-pkgs" / package_name, wide_spec(), \
        pf._pipeline(package_name)
    shutil.rmtree(out, ignore_errors=True)
    rt.render(pf.TEMPLATES, out, pipeline, spec, only=list(pf.MODEL_SIDE), splice=True)
    rec = rt.render(PACKAGE_TEMPLATES, out, pipeline, spec, only=list(pf.SUPPORT_SET),
                    into="Predictor")
    rt.render(PACKAGE_TEMPLATES, out, pipeline, spec, only=["predict.py"])
    folder, name = pf._monitor_template()
    rt.render(folder, out, pipeline, spec, only=[name])
    emit, root = rt.load_emit(), Path(rec["root"])
    mapping = emit.common_mapping(spec, pipeline, emit.plan(spec, pipeline))
    text = (PACKAGE_TEMPLATES / "graphs.py").read_text(encoding="utf-8")
    (root / "serving" / "graphs.py").write_text(emit.render_text(text, mapping), encoding="utf-8")
    bmap = json.loads((root / "blueprint-map.json").read_text(encoding="utf-8"))
    row = next(e for e in bmap["entries"] if e["file"] == "serving/samples.py")
    bmap["entries"].append(dict(row, file="serving/graphs.py"))
    (root / "blueprint-map.json").write_text(json.dumps(bmap, indent=2) + "\n", encoding="utf-8")
    add_inspection(root, package_name)
    return root


class Package:
    """The rendered package with the inspection modules, imported by the names tests read."""

    def __init__(self, name):
        self.name, self.root = name, render(name)
        sys.path.insert(0, str(self.root.parent))
        mod = lambda m: importlib.import_module(f"{name}.{m}")  # noqa: E731
        self.domain, self.network = mod("domain"), mod("model.network")
        self.records, self.predict = mod("serving.records"), mod("serving.predict")
        self.graphs, self.values = mod("serving.graphs"), mod("serving.values")
        self.inspection = mod("serving.inspection")
        self.inspector = mod("serving.inspection.inspector")
        self.query = mod("serving.inspection.query")
        self.codec = mod("serving.inspection.codec")
        self.view = mod("serving.inspection.view")
        self.capture = mod("serving.inspection.capture")
        self.table = mod("serving.inspection.attribution_table")
        self.explanation = mod("serving.inspection.explanation")

    def build(self, spec):
        """A seeded network of `spec` in eval mode, from this package's own Model module."""
        torch.manual_seed(0)
        return self.network.Network.build(self.domain.NetSpec.from_dict(spec)).eval()


@functools.lru_cache(maxsize=None)
def package() -> Package:
    return Package(PACKAGE)


# --------------------------------------------------------------------------- the specs

def _b(bid, block_type, **params):
    return {"id": bid, "type": block_type, "params": params}


def mlp_plain(task, head, loss) -> dict:
    """The `mlp-plain` composition over the row features: embedding -> concat -> (dense ->
    activation -> dropout) x 2 -> head."""
    s = pf._row_spec(task, head, loss)
    s["family"] = {"name": "mlp-plain", "source": "user:test_inspect_inspector"}
    s["blocks"] = [_b("B1", "embedding", columns=["zone"], dims={"zone": 2}), _b("B2", "concat"),
                   _b("B3", "dense", units=8, activation="none"),
                   _b("B4", "activation", name="gelu"), _b("B5", "dropout", p=0.1),
                   _b("B6", "dense", units=6, activation="none"),
                   _b("B7", "activation", name="relu"), _b("B8", "dropout", p=0.1),
                   _b("H", "head", out=head, activation="none")]
    return s


def lstm(kind) -> dict:
    """The `lstm` composition over the lagged series: embedding -> concat -> lstm(return all)
    -> pool(mean) -> head; `forecast` (autoregressive, H = 2) or `sequence-binary`."""
    s = pf._window(kind)
    out = s["blocks"][-1]["params"]["out"]
    s["family"] = {"name": "lstm", "source": "user:test_inspect_inspector"}
    s["blocks"] = [_b("E", "embedding", columns=["zone"], dims={"zone": 2}), _b("C", "concat"),
                   _b("R", "lstm", hidden=8, **{"return": "all"}), _b("P", "pool", kind="mean"),
                   _b("H", "head", out=out, activation="none")]
    return s


def gcn() -> dict:
    """The `gcn` composition over the key-pair graph, node level, binary: edge_dropout ->
    (gcn_conv -> activation -> dropout) -> gcn_conv -> head."""
    s = sf.keypair_spec()
    s["family"] = {"name": "gcn", "source": "user:test_inspect_inspector"}
    s["blocks"] = [_b("G0", "edge_dropout", p=0.1), _b("G1", "gcn_conv", out=16),
                   _b("A1", "activation", name="relu"), _b("D1", "dropout", p=0.1),
                   _b("G2", "gcn_conv", out=8), _b("H", "head", out=1, activation="none")]
    return s


# --------------------------------------------------------------------------- the cases

def _manifest(spec, threshold, label, sample, scores) -> dict:
    return {"output": orec.output_record(spec, threshold, label), "sample": sample,
            "model_version": MODEL_VERSION,
            "roundtrip": {"tolerance": Tolerance.for_scores(scores).as_json()}}


@functools.lru_cache(maxsize=None)
def row_case(reading) -> pf.Case:
    """An `mlp-plain` Predictor case of one row reading (predictor_fixtures' readings)."""
    pkg, s = package(), pf.shaped("rows")
    task, head, loss, threshold = pf.ROW_READINGS[reading]
    spec = mlp_plain(task, head, loss)
    model = pkg.build(spec)
    if reading == "anomaly-svdd":
        pf._svdd_centre(model)
    part = s.part("test")
    stored = pf._stored(model, pf._row_batch(part), np.arange(len(part)))
    threshold = pf._median_threshold(stored) if threshold == "median" else threshold
    stored = pf._decided(stored, orec.resolve_reading(spec), threshold)
    sample = orec.sample_record(spec, s.recipe)
    cols = [c for c in stored.columns if c.startswith("score") or c == "row_score"]
    manifest = _manifest(spec, threshold, task.get("label"), sample,
                         stored[cols].to_numpy().ravel().tolist())
    schema = pf._schema(s, spec, sample)
    return pf.Case(f"row-{reading}", spec, s, model, threshold, manifest, schema,
                   pf._request(s, schema), stored)


@functools.lru_cache(maxsize=None)
def window_case(kind) -> pf.Case:
    """An `lstm` Predictor case over windows: `forecast` or `sequence-binary`."""
    pkg, s = package(), pf.shaped("series")
    spec = lstm(kind)
    sample = orec.sample_record(spec, s.recipe)
    threshold = pf.WINDOW_THRESHOLDS.get(kind)
    manifest = _manifest(spec, threshold, "target", sample, [8.0])
    schema = pf._schema(s, spec, sample)
    return pf.Case(f"window-{kind}", spec, s, pkg.build(spec), threshold, manifest, schema,
                   pf._request(s, schema), None)


def _graph_root(name, spec) -> tuple:
    root = pf.WORK / "i06-graph-roots" / name
    return root, graph_pack.write_reference(root, pf.shaped(name).sources(spec))


@functools.lru_cache(maxsize=None)
def graph_case(name) -> pf.Case:
    """A graph Predictor case: `graph-keypair` (gcn, node level, binary) or `graph-level`
    (gin, graph level, binary). A node-level case's `stored` is the network on the sealed test
    view, as the evaluation stores it."""
    pkg, s = package(), pf.shaped(name)
    spec = gcn() if name == "graph-keypair" else sf.level_spec()
    threshold = pf.GRAPH_THRESHOLDS[name]
    root, meta = _graph_root(name, spec)
    context = pkg.graphs.GraphContext.load(root, s.module,
                                           pkg.domain.NetSpec.from_dict(spec).feature_contract)
    model = pkg.build(spec)
    stored = None
    if name != "graph-level":
        stored = pf._decided(pf._stored(model, pf._sealed_node_view(s, spec),
                                        np.arange(len(s.split["test"]))),
                             orec.resolve_reading(spec), threshold)
    sample = orec.sample_record(spec, s.recipe, meta)
    manifest = _manifest(spec, threshold, spec["task"]["label"], sample, [8.0])
    schema = pf._schema(s, spec, sample)
    return pf.Case(name, spec, s, model, threshold, manifest, schema, pf._request(s, schema),
                   stored, context, root)


def fresh_model(case):
    """The case's network built again (the same seed, so the same weights; the svdd centre
    set as the case's): a model no inspector has hooked — a deep copy would carry the hooks."""
    model = package().build(case.spec)
    if case.manifest["output"]["reading"] == "anomaly-svdd":
        pf._svdd_centre(model)
    return model


def predictor(case, manifest=None, model=None):
    """A Predictor of this package over a case (its records, or `manifest` in their place)."""
    pkg = package()
    records = pkg.records.ServingRecords.of(case.spec, manifest or case.manifest, case.schema,
                                            MODEL_VERSION)
    return pkg.predict.Predictor(pf.port_only(case.model if model is None else model),
                                 case.shaped.module.transform, records, case.context)


def with_serve(case, inspection: dict) -> dict:
    """The case's manifest with `serve.inspection` set."""
    manifest = copy.deepcopy(case.manifest)
    manifest["serve"] = {"inspection": inspection}
    return manifest


def window_rows(case, n=8, context=None):
    """A window request of series A: its last `context` rows of the past parts (default
    `MANIFEST.sample.history_rows`: every answer fully determined), then its first `n` test
    rows."""
    s = case.shaped
    context = case.manifest["sample"]["history_rows"] if context is None else context
    past = pd.concat([s.rows("train"), s.rows("valid")])
    test = case.rows[case.rows["station"] == "A"].sort_values("ts").head(n)
    context = past[past["station"] == "A"].sort_values("ts").tail(context)
    return pd.concat([context.drop(columns=[c for c in context.columns
                                                if c not in test.columns]), test],
                        ignore_index=True)


def example_query(case, rows=None):
    """The query `open()` explains: the rows of a case, as `predict._example_query` builds it."""
    q = package().query
    rows = (window_rows(case) if case.spec["data_contract"]["sample"]["shape"] == "window"
            else case.rows.head(12)) if rows is None else rows
    rows = rows.reset_index(drop=True)
    return q.ExplainQuery(rows, None, len(rows) - 1, detail="summary")


def opened(case, manifest=None, model=None):
    """(the Predictor, its inspector opened on the case's example rows)."""
    p = predictor(case, manifest, model)
    inspector = p.inspector()
    inspector.open(example_query(case))
    return p, inspector


# --------------------------------------------------------------------------- bundles

def bundle(case, name, examples="row") -> pf.Bundle:
    """predictor_fixtures' bundle for `case`, the inspection modules rendered into its
    package, MANIFEST.json re-pinned over the files now on disk."""
    b = pf.Bundle.build(case, name, examples=examples)
    shutil.rmtree(b.root / name)
    shutil.copytree(render(name), b.root / name)
    shutil.copyfile(case.shaped.module_path, b.root / name / "features" / "preprocess.py")
    b.write_manifest()
    return b


# --------------------------------------------------------------------------- the table schema

_NUMS = {"type": "array", "items": {"type": "number"}}
TABLE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "vega/attribution-table@1 (inspect-contract §5.5)",
    "type": "object", "additionalProperties": False,
    "required": ["schema", "method", "unit", "shape", "inputs", "targets", "samples"],
    "properties": {
        "schema": {"const": "vega/attribution-table@1"},
        "method": {"enum": ["grad_x_input", "integrated_gradients"]},
        "unit": {"enum": ["distance", "error", "logit", "rank", "target"]},
        "shape": {"enum": ["row", "window", "graph"]},
        "inputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "targets": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["kind", "index", "name", "sign", "why", "unit"],
            "properties": {"kind": EXPLANATION["$defs"]["target_kind"],
                           "index": {"type": ["integer", "null"]},
                           "name": {"type": ["string", "null"]}, "sign": {"enum": [1, -1]},
                           "why": {"enum": ["predicted", "requested", "default"]},
                           "unit": {"enum": ["distance", "error", "logit", "rank", "target"]}}}},
        "samples": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["row", "prediction", "target_value", "attribution"],
            "properties": {
                "row": {"type": "integer", "minimum": 0},
                "prediction": {"type": "object", "required": ["warnings"],
                               "properties": {"warnings": {"type": "array",
                                                           "items": {"type": "string"}}}},
                "target_value": {"type": "number"},
                "attribution": {"oneOf": [
                    _NUMS,
                    {"type": "object", "additionalProperties": False,
                     "required": ["per_step", "per_feature"],
                     "properties": {"per_step": _NUMS, "per_feature": _NUMS}},
                    {"type": "object", "additionalProperties": False,
                     "required": ["self", "hops"],
                     "properties": {"self": {"oneOf": [{"type": "null"}, _NUMS]},
                                    "hops": {"type": "object",
                                             "additionalProperties": {"type": "number"}}}}]}}}}}}
