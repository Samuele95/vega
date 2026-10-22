"""
samples_fixtures.py — what test_samples.py proves the serving sample assemblers on: the rendered
package (domain, network, readout, assemble and the spliced blocks, with the serving templates
`samples.py` and `graphs.py` in the Predictor's module), the shaper's real outputs (the three
graph fixtures of dataset-shaper and a small lagged window recipe written here, each through
`shape.py` and its `--transform-only` module), the specs of each, the serving records, the
graph bundles `graph_pack.write_reference` writes and a pack context over them.

Imported by test_samples.py (not a test file itself: the unit tier collects test_*.py).
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
SKILLS = SCRIPTS.parent.parent
SHAPER = SKILLS / "dataset-shaper" / "scripts" / "shape.py"
FIXTURES = SKILLS / "dataset-shaper" / "fixtures"
PACK = SKILLS / "vega-pack" / "scripts"
PACKAGE = "samples_pkg"
for _p in (str(SCRIPTS), str(HERE), str(BACKEND), str(PACK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import graph_pack  # noqa: E402
import output_record  # noqa: E402
import render_templates as rt  # noqa: E402
from pack_context import Campaign, PackContext, PackExit  # noqa: E402
from vega_pack_core import BundlePaths  # noqa: E402

_WORK = tempfile.TemporaryDirectory(prefix="samples-")
WORK = Path(_WORK.name)
PIPELINE = rt.reference_pipeline()
PIPELINE["system"]["package"] = PACKAGE


def _wide_spec():
    """A spec whose blocks splice every segment the networks below build from."""
    s = rt.reference_spec("graph")
    s["blocks"] = [{"id": f"W{i}", "type": t, "params": {}} for i, t in enumerate(
        ("embedding", "concat", "gru", "gcn_conv", "sage_conv", "gat_conv", "gin_conv",
         "graph_pool", "activation", "dense", "head"))]
    return s


def _render():
    """The model side (domain, network, readout, assemble, the spliced blocks) and the two
    serving templates in the Predictor's module. `graphs.py` shares its name with the
    ingestion's graph adapter of the v2.3 emit table, where render_templates would place any
    `graphs.py`; the serving set's copy is rendered with the emitter's own mapping into the
    Predictor's module and mapped there, as pack renders `package/*.py`."""
    out, spec = WORK / "pkg", _wide_spec()
    rt.render(TEMPLATES, out, PIPELINE, spec,
              only=["domain.py", "network.py", "readout.py", "assemble.py"], splice=True)
    rec = rt.render(TEMPLATES / "serving" / "package", out, PIPELINE, spec,
                    only=["samples.py"], into="Predictor")
    emit, root = rt.load_emit(), Path(rec["root"])
    mapping = emit.common_mapping(spec, PIPELINE, emit.plan(spec, PIPELINE))
    text = (TEMPLATES / "serving" / "package" / "graphs.py").read_text(encoding="utf-8")
    (root / "serving" / "graphs.py").write_text(emit.render_text(text, mapping), encoding="utf-8")
    entries = rec["blueprint_map"]["entries"]
    row = next(e for e in entries if e["file"] == "serving/samples.py")
    entries.append(dict(row, file="serving/graphs.py"))
    (root / "blueprint-map.json").write_text(json.dumps(rec["blueprint_map"], indent=2) + "\n",
                                             encoding="utf-8")
    return rec


RENDERED = _render()
ROOT = Path(RENDERED["root"])
sys.path.insert(0, str(WORK / "pkg"))
D = importlib.import_module(f"{PACKAGE}.domain")
A = importlib.import_module(f"{PACKAGE}.model.assemble")
N = importlib.import_module(f"{PACKAGE}.model.network")
S = importlib.import_module(f"{PACKAGE}.serving.samples")
G = importlib.import_module(f"{PACKAGE}.serving.graphs")


# --------------------------------------------------------------------------- the shaper

def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          cwd=str(SKILLS), timeout=900)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    return proc.stdout


class Shaped:
    """One recipe through shape.py and --transform-only; the outputs and the module, loaded."""

    def __init__(self, name, recipe_path):
        self.name, self.dir = name, WORK / "shaped" / name
        _run(SHAPER, "--recipe", recipe_path, "--out-dir", self.dir, "--format", "parquet")
        self.module_path = self.dir / "preprocess.py"
        _run(SHAPER, "--transform-only", "--recipe", recipe_path, "--fitted",
             self.dir / "fitted.json", "--emit", self.module_path)
        self.recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.fitted = json.loads((self.dir / "fitted.json").read_text(encoding="utf-8"))
        path = Path(self.recipe["input"]["path"])
        self.raw = pd.read_csv(path if path.is_absolute() else SKILLS / path, low_memory=False)
        self.split = self.fitted["__split_raw__"]
        spec = importlib.util.spec_from_file_location(f"preprocess_{name}", self.module_path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module          # a bundle imports its Recipe by name
        spec.loader.exec_module(self.module)

    def rows(self, part):
        return self.raw.iloc[np.asarray(self.split[part], dtype=np.int64)].reset_index(drop=True)

    def part(self, part):
        recorded = Path(self.manifest["outputs"]["files"][part]["path"])
        return pd.read_parquet(self.dir / recorded.name)

    def sources(self, spec):
        return graph_pack.ReferenceSources(self.dir, self.manifest, self.split, self.raw, spec)


SERIES_LENGTHS = {"A": 40, "B": 35, "C": 45}


def _series_recipe():
    """Three series (station), ordered by ts, a nominal zone and a numeric target; the recipe
    lags x by 1 and 2 within each station, encodes zone ordinally and scales."""
    rng = np.random.default_rng(0)
    rows = [{"station": s, "ts": 1000 + 10 * t + k, "x": rng.normal(),
             "zone": rng.choice(["n", "e", "w"]), "target": rng.normal()}
            for k, (s, n) in enumerate(SERIES_LENGTHS.items()) for t in range(n)]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    csv = WORK / "series.csv"
    frame.to_csv(csv, index=False)

    def step(sid, op, columns, params):
        return {"id": sid, "op": op, "columns": columns, "params": params, "source": "user:test",
                "rationale": f"test step {op}", "alternatives": [],
                "consequences": {"rows": "none", "columns": "none", "downstream": {}},
                "reversible": True}
    recipe = {"schema": "dataset-shaper/recipe@1",
              "input": {"path": str(csv), "rows": len(frame), "columns": 5, "layers_read": [],
                        "digest": "sha256:" + hashlib.sha256(csv.read_bytes()).hexdigest()},
              "seed": 7, "goal": "custom", "target": "target",
              "outputs": {"dir": "shaped/", "format": "parquet", "files": [],
                          "recipe": "shaped/recipe.json", "manifest": "shaped/manifest.json",
                          "lineage": "shaped/lineage.json",
                          "reproduce": "shaped/reproduce_series.py"},
              "steps": [step("S1", "select_partition", [], {"label": "target",
                                                            "task": "regression"}),
                        step("S2", "split", [], {"kind": "time", "time_column": "ts",
                                                 "test": 0.25, "valid": 0.15}),
                        step("S3", "lag", ["x"], {"lags": [1, 2], "group_by": "station",
                                                  "time_column": "ts"}),
                        step("S4", "encode", ["zone"], {"strategy": "ordinal"}),
                        step("S5", "scale", [], {"kind": "standard",
                                                 "columns": ["x", "x_lag1", "x_lag2"]})]}
    path = WORK / "series.recipe.json"
    path.write_text(json.dumps(recipe, indent=1), encoding="utf-8")
    return path


def _fixture(name) -> dict:
    return json.loads((FIXTURES / f"{name}.recipe.json").read_text(encoding="utf-8"))


def _graph_step(recipe) -> dict:
    return next(s for s in recipe["steps"] if s["op"] == "graph_edges")


def _variant(name) -> Path:
    """Two variants of B05's fixtures: `graph-spatial` (the sensors joined by their 5 nearest
    by raw coordinates, as dataset-shaper's own spatial variant) and `graph-keypair-weighted`
    (the feeder links weighed by the input's line capacity) — `raw:<column>` beside the
    same column as shaped, and weighted edges."""
    if name == "graph-spatial":
        r = _fixture("graph-knn")
        geom = {"id": "S0", "op": "parse_geometry", "columns": ["lat", "lon"],
                "params": {"kind": "latlon", "lat": "lat", "lon": "lon", "crs": "EPSG:4326"},
                "source": "user:the sensors report WGS84 degrees", "rationale": "the CRS",
                "alternatives": [], "consequences": {"rows": "none", "columns": "none",
                                                     "downstream": "none"}, "reversible": True}
        repro = dict(geom, id="S3b", op="spatial_reproject", params={"to_crs": "EPSG:32632"},
                     source="user:metres for the distance", rationale="a metric CRS")
        split = next(i for i, s in enumerate(r["steps"]) if s["op"] == "split")
        r["steps"] = [geom] + r["steps"][:split + 1] + [repro] + r["steps"][split + 1:]
        _graph_step(r)["params"] = {"method": "spatial", "k": 5, "symmetric": True,
                                    "weight": "distance"}
    else:
        r = _fixture("graph-keypair")
        _graph_step(r)["params"]["weight"] = "line_cap_mw"
    path = WORK / f"{name}.recipe.json"
    path.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return path


@functools.lru_cache(maxsize=None)
def shaped(name) -> Shaped:
    if name == "series":
        return Shaped(name, _series_recipe())
    if name in ("graph-spatial", "graph-keypair-weighted"):
        return Shaped(name, _variant(name))
    return Shaped(name, FIXTURES / f"{name}.recipe.json")


# --------------------------------------------------------------------------- specs

def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric", "origin": "S1"}


def _nominal(name, cardinality):
    return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": cardinality,
            "unseen_category_id": cardinality - 1, "origin": "S2"}


def _spec(family, features, sample, task, blocks, loss="mse"):
    s = rt.reference_spec("row")
    s["family"]["name"] = family
    s["task"], s["blocks"] = task, blocks
    s["data_contract"].update(features=features, input_dim=len(features), sample=sample)
    s["loss"] = {"name": loss, "params": {}}
    return s


def _head(out=1):
    return {"id": "H", "type": "head", "params": {"out": out, "activation": "none"}}


def window_spec(length=4, autoregressive=True, kind="forecast"):
    sample = {"shape": "window", "length": length, "stride": 3, "horizon": 2,
              "series_by": "station", "order_by": "ts", "context": "previous-part",
              "autoregressive": autoregressive}
    task = {"kind": "forecast", "label": "target", "horizon": 2}
    if kind != "forecast":
        sample.update(horizon=None, autoregressive=False)
        task = {"kind": kind, "label": "target", "base_kind": "regression"}
    features = [_num("x"), _num("x_lag1"), _num("x_lag2"), _nominal("zone", 4)]
    return _spec("gru", features, sample, task,
                 [{"id": "B1", "type": "gru", "params": {"hidden": 8, "return": "last"}},
                  _head(2 if kind == "forecast" else 1)])


def row_spec():
    s = window_spec()
    s["data_contract"]["sample"] = {"shape": "row"}
    s["task"] = {"kind": "regression", "label": "target"}
    s["blocks"] = [{"id": "B1", "type": "dense", "params": {"units": 4}}, _head()]
    return s


def _graph_sample(s, level="node", protocol="inductive"):
    g = s.manifest["graph"]
    out = {"shape": "graph", "edges": str(s.dir / Path(s.manifest["outputs"]["files"]["edges"]
                                                       ["path"]).name),
           "level": level, "graph_id": g["graph_id"], "directed": g["directed"],
           "self_loops": g["self_loops"], "protocol": protocol,
           "batching": "graphs" if level == "graph" else "full", "neighbours": None,
           "exposure": None}
    if protocol == "transductive":
        out["exposure"] = {"kind": "features", "rows": "test", "source": "user:t1"}
    return out


def keypair_spec(protocol="inductive", name="graph-keypair"):
    s = shaped(name)
    names = ["load_mw", "gen_mw", "voltage_pu", "line_cap_mw"] + [f"zone=Z{i}" for i in range(1, 7)]
    blocks = [{"id": "B1", "type": "gcn_conv", "params": {"out": 16}},
              {"id": "B2", "type": "activation", "params": {"name": "relu"}},
              {"id": "B3", "type": "sage_conv", "params": {"out": 8}}, _head()]
    return _spec("gcn", [_num(n) for n in names], _graph_sample(s, protocol=protocol),
                 {"kind": "binary", "label": "overloaded", "classes": [0, 1], "positive": 1},
                 blocks, "bce_with_logits")


def knn_spec():
    s = shaped("graph-knn")
    names = ["lat", "lon", "lanes", "speed_limit", "flow_vph", "occupancy"]
    blocks = [{"id": "B1", "type": "gat_conv", "params": {"out": 4, "heads": 2}},
              {"id": "B2", "type": "activation", "params": {"name": "relu"}}, _head()]
    return _spec("gat", [_num(n) for n in names], _graph_sample(s),
                 {"kind": "regression", "label": "mean_speed"}, blocks)


def level_spec():
    s = shaped("graph-level")
    features = [_nominal("element", 6), _num("charge"), _num("aromatic"), _num("degree")]
    blocks = [{"id": "B1", "type": "embedding", "params": {"columns": ["element"],
                                                           "dims": {"element": 3}}},
              {"id": "B2", "type": "concat", "params": {}},
              {"id": "B3", "type": "gin_conv", "params": {"out": 16}},
              {"id": "B4", "type": "graph_pool", "params": {"readout": "mean"}}, _head()]
    return _spec("gin", features, _graph_sample(s, level="graph"),
                 {"kind": "binary", "label": "soluble", "classes": [0, 1], "positive": 1},
                 blocks, "bce_with_logits")


def spatial_spec():
    s = shaped("graph-spatial")
    names = ["lat", "lon", "lanes", "speed_limit", "flow_vph", "occupancy"]
    blocks = [{"id": "B1", "type": "gcn_conv", "params": {"out": 8}},
              {"id": "B2", "type": "activation", "params": {"name": "relu"}}, _head()]
    return _spec("gcn", [_num(n) for n in names], _graph_sample(s),
                 {"kind": "regression", "label": "mean_speed"}, blocks)


GRAPHS = {"graph-keypair": keypair_spec, "graph-knn": knn_spec, "graph-level": level_spec,
          "graph-spatial": spatial_spec,
          "graph-keypair-weighted": lambda: keypair_spec(name="graph-keypair-weighted")}
NODE_GRAPHS = ("graph-keypair", "graph-knn", "graph-spatial", "graph-keypair-weighted")


class Records:
    """The serving records as the Predictor holds them: the spec and MANIFEST.sample."""

    def __init__(self, spec, recipe=None):
        self.spec = D.NetSpec.from_dict(spec)
        self.sample = output_record.sample_record(spec, recipe or {})


@functools.lru_cache(maxsize=None)
def bundle(name):
    """A graph bundle's root holding the reference graph_pack writes, and its context."""
    s, spec = shaped(name), GRAPHS[name]()
    root = WORK / "bundles" / name
    meta = graph_pack.write_reference(root, s.sources(spec))
    context = G.GraphContext.load(root, s.module, D.NetSpec.from_dict(spec).feature_contract)
    return root, meta, context


def assembler(name):
    s, spec = shaped(name), GRAPHS[name]()
    return S.assembler_for(Records(spec), s.module.transform, bundle(name)[2])



# --------------------------------------------------------------------------- pack

def _context(name, spec, tag):
    """A PackContext over a shaped fixture, as pack.py builds it before the phases."""
    s = shaped(name)
    root = WORK / "pack" / tag
    shutil.rmtree(root, ignore_errors=True)
    paths = BundlePaths(root=root / "bundle", package="pkg")
    (paths.pkg / "features").mkdir(parents=True)
    shutil.copyfile(s.module_path, paths.pkg / "features" / "preprocess.py")
    campaign = Campaign(dir=root, doc={}, ledger={}, seal={}, run_id="R0", eval_dir=root,
                        metrics={}, threshold=None, shaped_dir=s.dir, shaped_manifest=s.manifest,
                        recipe=s.recipe, spec=spec, raw=s.raw, fitted=s.fitted)
    return PackContext(options=None, paths=paths, campaign=campaign, package="pkg", ident={},
                       markers=[])


def _exit(fn):
    try:
        fn()
    except PackExit as refusal:
        return refusal.code, " ".join(refusal.lines)
    raise AssertionError("expected a PackExit")


def _raises(kind, needle, fn):
    try:
        fn()
    except kind as exc:
        assert needle in str(exc), f"{kind.__name__} raised without {needle!r}: {exc}"
        return
    raise AssertionError(f"expected {kind.__name__} mentioning {needle!r}")
