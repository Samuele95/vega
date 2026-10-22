"""
_page_fixtures.py — synthetic chain records for `test_page_config.py`: one netspec per view key
and sample shape (the D4 §17.4 families, reduced to the blocks each covers), the story and the
authored plan that agree with it, a shape record, the examples and the serve block. A helper
of the test; it runs nothing on import.
"""
from __future__ import annotations

import copy
from pathlib import Path

from page_derive import Catalogs
from vega_show_core import ShowPaths, write_json

ALL_ROUTES = ["GET /health", "GET /v1/examples", "GET /v1/network", "GET /v1/schema",
              "POST /v1/explain", "POST /v1/predict", "POST /v1/predict/csv"]
# the installed template's identity (TEMPLATE.json, written from W15 on), else a stand-in: a plan
# pins the template it was drafted against, and page_config refuses another (V1)
TEMPLATE_PIN = dict(Catalogs.load().template or {"version": "2.3.0", "manifest_digest": "sha256:" + "2" * 64})
MANIFEST_DIGEST = "sha256:" + "1" * 64
FEATURES = [{"name": n, "dtype": "float32", "role": "numeric", "origin": "S5"}
            for n in ("a", "b", "c", "d")] + [
    {"name": "zone", "dtype": "int64", "role": "nominal", "cardinality": 5, "origin": "S4"},
    {"name": "a_missing", "dtype": "float32", "role": "numeric", "origin": "S2"}]


def _b(kind, **params):
    return {"type": kind, "params": params}


MLP = [_b("dense", units=16, activation="relu"), _b("head", out=3, activation="none")]
RESIDUAL = [_b("dense", units=16, activation="none"),
            _b("residual", repeat=1, inner=[_b("dense", units=16, activation="none"), _b("layernorm"),
                                            _b("activation", name="gelu"), _b("dropout", p=0.1)]),
            _b("head", out=3, activation="none")]
EMBED = [_b("embedding", columns=["zone"], dim=4), _b("concat"),
         _b("dense", units=8, activation="silu"), _b("head", out=1, activation="none")]
AUTOENC = [_b("dense", units=4, activation="relu"), _b("decoder", widths=[8]),
           _b("head", out=10, activation="none")]
VAE = [_b("dense", units=8, activation="relu"), _b("reparameterize", z=2), _b("decoder"),
       _b("head", out=10, activation="none")]
GRU = [_b("gru", hidden=8), _b("head", out=3, activation="none")]
CNN = [_b("conv1d", channels=4, kernel=3), _b("pool1d"), _b("flatten"), _b("head", out=3)]
RNN = [_b("rnn", hidden=6, bidirectional=True), _b("head", out=2)]
GCN = [_b("gcn_conv", out=8), _b("gcn_conv", out=8), _b("head", out=3)]
GIN = [_b("gin_conv", out=8), _b("graph_pool", reduce="mean"), _b("head", out=1)]
WINDOW = {"shape": "window", "length": 6, "stride": 1}
NODE = {"shape": "graph", "level": "node", "edges": "shaped/x.edges.parquet"}
GRAPH = {"shape": "graph", "level": "graph", "graph_id": "g", "edges": "shaped/x.edges.parquet"}

# name: (task kind, base, loss, primary, family, blocks, sample, spatial) and the lane expected:
# (view key, link, goal, score source, modality)
CASES = {
    "classes": (("multiclass", None, "cross_entropy", "f1_macro", "mlp-plain", MLP, None, False),
                ("classes", "softmax", "predict", None, "row")),
    "binary": (("binary", None, "bce_with_logits", "pr_auc", "mlp-plain", EMBED, None, False),
               ("binary", "sigmoid", "predict", None, "row")),
    "labels": (("multilabel", None, "bce_with_logits", "f1_micro", "mlp-residual", RESIDUAL, None, False),
               ("labels", "sigmoid_each", "predict", None, "row")),
    "value": (("regression", None, "mse", "rmse", "mlp-plain", MLP, None, False),
              ("value", "identity", "predict", None, "row")),
    "value-quantile": (("regression", None, "quantile", "mae", "mlp-plain", MLP, None, True),
                       ("value", "identity", "predict", None, "spatial-row")),
    "ordinal": (("ordinal", None, "cumulative_link", "mae_ordinal", "mlp-plain", MLP, None, False),
                ("ordinal", "cumulative", "predict", None, "row")),
    "reconstruction": (("unsupervised", None, "reconstruction", "reconstruction_error", "autoencoder",
                        AUTOENC, None, False), ("reconstruction", "reconstruction", "reconstruct", None, "row")),
    "anomaly-vae": (("anomaly", None, "vae", "anomaly_score_mean", "vae", VAE, None, False),
                    ("anomaly", "reconstruction", "anomaly", "reconstruction", "row")),
    "anomaly-svdd": (("anomaly", None, "svdd", "pr_auc", "deep-svdd", MLP, None, False),
                     ("anomaly", "distance", "anomaly", "distance", "row")),
    "horizon": (("forecast", None, "mse", "mase", "gru", GRU, WINDOW, False),
                ("horizon", "horizon", "predict", None, "window")),
    "sequence-classes": (("sequence-classification", "multiclass", "cross_entropy", "f1_macro", "cnn1d",
                          CNN, WINDOW, False), ("classes", "softmax", "predict", None, "window")),
    "sequence-ordinal": (("sequence-regression", "ordinal", "cumulative_link", "mae_ordinal", "rnn",
                          RNN, WINDOW, False), ("ordinal", "cumulative", "predict", None, "window")),
    "graph-node": (("multiclass", None, "cross_entropy", "f1_macro", "gcn", GCN, NODE, False),
                   ("classes", "softmax", "predict", None, "graph")),
    "graph-level": (("regression", None, "huber", "rmse", "gin", GIN, GRAPH, False),
                    ("value", "identity", "predict", None, "graph")),
}


def netspec(case: str) -> dict:
    (kind, base, loss, primary, family, blocks, sample, _), _lane = CASES[case]
    task = {"kind": kind, "label": "y", "source": "geometry:partitions/chosen"}
    if base:
        task["base_kind"] = base
    return {"schema": "vega/netspec@1", "task": task,
            "data_contract": {"features": copy.deepcopy(FEATURES), "sample": copy.deepcopy(sample or {"shape": "row"})},
            "family": {"name": family, "source": "blueprint:posture"},
            "blocks": [{"id": f"B{i + 1}", **copy.deepcopy(b)} for i, b in enumerate(blocks)],
            "loss": {"name": loss, "params": {}},
            "metrics": {"primary": primary, "secondary": ["mae"] if primary != "mae" else ["rmse"],
                        "selection": {"monitor": f"valid.{primary}"}},
            "backend": {"name": "pytorch"}}


def story(case: str, spec: dict, direction: str) -> dict:
    (kind, base, _, primary, family, _, sample, _), (view_key, _, goal, _, _) = CASES[case]
    sample = sample or {"shape": "row"}
    return {"schema": "vega/story@2",
            "task": {"kind": kind, "base": base, "goal": goal, "sample_shape": sample["shape"],
                     "graph_level": sample.get("level"), "family": family, "view_key": view_key,
                     "classes": None, "outcomes": None, "primary": primary, "direction": direction},
            "sections": [{"id": s} for s in ("context", "network", "experiments", "results")],
            "code_notes": {"root": {"note": "The composition root.", "link": None, "ref": None}}}


def plan(modality: str, base: str, chosen: str | None = None, stance: str | None = None) -> dict:
    return {"schema": "vega/story-plan@2", "status": "authored", "template": dict(TEMPLATE_PIN),
            "proposals": {"modality": modality,
                          "motif": {"modality": modality, "base": base, "admissible": [base],
                                    "stance": stance, "fork": None if stance else "S3"}},
            "authored": {"motif": chosen or base}}


def write_plan(root: Path, doc: dict) -> ShowPaths:
    paths = ShowPaths.of(root)
    write_json(paths.plan, doc)
    return paths


SHAPE = {"recipe": {"steps": [
    {"id": "S1", "op": "retype", "columns": ["zone"], "params": {"to": "nominal"}},
    {"id": "S2", "op": "impute", "columns": ["a"], "params": {"strategy": "median", "indicator": True}},
    {"id": "S3", "op": "transform", "columns": ["b", "c"], "params": {"kind": "signed-log1p"}},
    {"id": "S4", "op": "encode", "columns": ["zone"], "params": {"strategy": "ordinal"}},
    {"id": "S5", "op": "scale", "columns": [], "params": {"kind": "standard", "columns": "numeric"}}]},
    "manifest": {"steps": [
        {"id": "S1", "op": "retype", "columns_added": []},
        {"id": "S2", "op": "impute", "columns_added": ["a_missing"]},
        {"id": "S3", "op": "transform", "columns_added": []},
        {"id": "S4", "op": "encode", "columns_added": []},
        {"id": "S5", "op": "scale", "columns_added": [],
         "parameters_fitted": {"kind": "standard", "fitted": {n: {"mean": 0, "std": 1}
                                                              for n in ("a", "b", "c", "d", "a_missing")}}}]}}
EXPECTED_OPS = [("impute", 1, "S2"), ("indicator", 1, "S2"), ("log1p", 2, "S3"),
                ("encode", 1, "S4"), ("standard_scale", 5, "S5")]


def records(examples=True, routes=None, scores=None, spatial=False) -> dict:
    presets = [{"id": "typical-0", "kind": "typical_correct", "group": "typical", "expected": None,
                "facts": {"right": 3, "of": 4}, "row": {"a": 1}}]
    return {"manifest": {"schema": "vega/bundle@2", "files": {},
                         "serve": {"present": True, "mode": "online", "endpoints": list(routes or ALL_ROUTES),
                                   "identity": {"title": "synthetic lane", "description": "A test lane."}}},
            "manifest_digest": MANIFEST_DIGEST, "shape": copy.deepcopy(SHAPE),
            "analysis": {"context": {"spatial": {"crs": "EPSG:4326"} if spatial else None}},
            "examples": {"schema": "vega/examples@2", "presets": presets} if examples else None,
            "reference_scores": scores}
