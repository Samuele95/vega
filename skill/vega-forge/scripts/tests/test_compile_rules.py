#!/usr/bin/env python3
"""
test_compile_rules.py — the v2.3 compiler rules, the shape walk and the adapter resolver, each
rule proved by its positive case and by seeded negatives (build plan B11; D1 §14.1, §14.4, §12.1).

    python3 test_compile_rules.py [-v] [-k <substring>]
    python3 -m pytest test_compile_rules.py

The chain each spec compiles against is synthetic and small — a shaped manifest, a blueprint
pipeline and a model's layers, written as values — one per sample shape: a row (binary, an
embedded nominal), a window (a three-step forecast with a nominal, a lag step grouped by the
series, the autoregressive channel), a window sequence task, a graph at level node (gcn, full
batch) and one at level graph (gin, graphs batching). The graph and recurrent blocks are still
`planned` in the shipped `coverage.json` until B14 flips it, so the positive cases compile
against a copy of that file with the planned blocks covered; B2's own refusal of a planned block
is proved against the shipped one.

What is proved:

  positives  every base chain compiles with no ERROR and no WARN;
  W1  autoregressive only on forecast, horizon iff forecast, length/stride/season, the context
      and its source, series_by (the lag key, the unattended WARN, a user: null, never a
      feature, in the output schema) and its series_source through P1, the time split;
  W2  a graph family or block off a graph sample, the edges part (equal normalised paths:
      a bare file name, another directory or an absolute spelling refused), the graph block's
      level/graph_id/directed, graph_pool iff level graph, gin at level graph, the protocol and
      its exposure, the batching and its neighbours, a non-graph family's WARN (not the floor's);
  W3  a bidirectional block on a forecast, under a stream serving mode, a bi-* family there;
  G2  the step axis unreduced at the head, an embedding before the tokenizer, a conv after
      graph_pool, a sequence block on rows, an axis reader with no axis;
  T1b recon_dim, the svdd z, forecast H×Q, the sequence base, ordinal k−1;
  T1e base_kind, classes, label_format, positive;
  D2a keys out of input_dim, declared keys; D2b a nominal uncovered, the tokenizer's cover;
  R1  "full" iff batching full, batch at most the training targets;
  B2  a planned block named as planned; G1b the admission table's refusals; G1a–G1d each
      emitted under its own id;
  D1e input.pipeline_digest missing, stale or computed over the file's bytes, and the emitter's
      digest as its one definition; D1b a manifest digest that differs between the blueprint,
      the spec and the shape layer, as an ERROR;
  P1  geometry:partitions/<label> and partitions/chosen resolved by candidate label (with a
      key after the label) and typing/<col> by column, as the shaper resolves them; an unknown
      label, an unknown key, an unknown column and an abstained choice refused;
  ShapeWalk on composites and reshapes; ChainInputs' split and series key;
  PackageAdapters on a rendered v2.3 package (classes instantiated, the Evaluator wired) and on
  an emitted v2.2 one (modules), and its refusals.

Exit 0 when every test passes.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))

import compile_spec as C  # noqa: E402
from vega_core import (AdapterLoadError, Catalog, ChainInputs, Level,  # noqa: E402
                       PackageAdapters, ShapeWalk, SourceResolver, canonical_digest)

VERBOSE = "-v" in sys.argv
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
SHIPPED = json.loads((SCRIPTS / "backends" / "pytorch" / "coverage.json")
                     .read_text(encoding="utf-8"))

_WORK = tempfile.TemporaryDirectory(prefix="compile-rules-")
WORK = Path(_WORK.name)


def _covered_backends() -> Path:
    """A backends dir whose pytorch coverage lists the planned blocks as built (post-B14)."""
    root = WORK / "backends-covered" / "pytorch"
    root.mkdir(parents=True)
    cov = copy.deepcopy(SHIPPED)
    cov["blocks"] = list(cov["blocks"]) + [b for b in (cov.get("planned") or {}).get("blocks", [])
                                           if b not in cov["blocks"]]
    cov["planned"] = {"blocks": [], "families": []}
    (root / "coverage.json").write_text(json.dumps(cov), encoding="utf-8")
    return root.parent


COVERED = _covered_backends()


# --------------------------------------------------------------------------- the chains

def _choice(row, *alts):
    """A choice's provenance: vega:default citing a catalog row, with its alternatives."""
    return {"source": "vega:default", "catalog": row, "rationale": f"the {row} row",
            "alternatives": [{"when": a} for a in alts] or [{"when": "an ablation says so"}]}


def _block(bid, type_, **params):
    structural = type_ in ("head", "concat")
    prov = ({"source": "vega:default", "catalog": f"blocks/{type_}", "rationale": "structural",
             "alternatives": []} if structural else _choice(f"blocks/{type_}"))
    return {"id": bid, "type": type_, "params": params, **prov}


def _feature(name, nominal=False, cardinality=None):
    if nominal:
        return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": cardinality,
                "unseen_category_id": cardinality - 1}
    return {"name": name, "dtype": "float32", "role": "numeric"}


def _spec(task, family, features, input_dim, sample, blocks, loss, primary, secondary=()):
    return {
        "schema": "vega/netspec@1",
        "input": {"model": "demo.shaped.html", "manifest": "shaped/manifest.json"},
        "seed": 7,
        "task": {**task, "source": "user:turn-1"},
        "data_contract": {"source": "shape:manifest/output_schema",
                          "parts": {"train": "shaped/demo.train.parquet",
                                    "valid": "shaped/demo.valid.parquet", "test": "sealed"},
                          "features": features, "input_dim": input_dim, "sample": sample},
        "family": {"name": family, **_choice(f"families/{family}")},
        "blocks": blocks,
        "loss": {"name": loss, "params": {}, **_choice(f"tasks/{task['kind']}")},
        "metrics": {"primary": primary, "secondary": list(secondary),
                    "selection": {"monitor": f"valid.{primary}"},
                    **_choice(f"tasks/{task['kind']}")},
        "optimizer": {"name": "adamw", "params": {"lr": 0.001}, **_choice("defaults/optimizer")},
        "schedule": {"name": "cosine", "params": {}, **_choice("defaults/schedule")},
        "regularization": [],
        "early_stopping": {"monitor": f"valid.{primary}", "patience": 5,
                           **_choice("defaults/early_stopping")},
        "init": {"scheme": "kaiming_uniform", **_choice("defaults/init")},
        "training": {"batch_size": 32, "max_epochs": 20, "budget": {"wall_clock_min": 5},
                     **_choice(["defaults/batch_size", "defaults/max_epochs"])},
        "backend": {"name": "pytorch", "hints": {}},
        "outputs": {"run_dir": "vega-runs/demo/", "package": "demo_pipeline"},
    }


def _schema(features=(), keys=(), label=None):
    out = {k: {"dtype": "str", "role": "key"} for k in keys}
    out.update({f: {"dtype": "float64", "role": "feature"} for f in features})
    if label:
        out[label] = {"dtype": "int64", "role": "label"}
    return out


def _files(train, valid, test, edges=None):
    out = {p: {"path": f"shaped/demo.{p}.parquet", "rows": n}
           for p, n in (("train", train), ("valid", valid), ("test", test))}
    if edges:
        out["edges"] = {"path": "shaped/demo.edges.parquet", "rows": edges}
    return {"files": out}


def _encode(sid, column, unseen):
    return {"id": sid, "op": "encode",
            "parameters_fitted": {"fitted": {column: {"unseen_category_id": unseen}}}}


def _split(kind, by=None):
    return {"id": "S1", "op": "split", "parameters_fitted": {"kind": kind, "by": by}}


def _pipeline(input_dim, label, mode="batch"):
    return {"system": {"package": "demo_pipeline"}, "posture": {"family": "neural"},
            "serving": {"mode": mode}, "feature_contract": {"input_dim": input_dim,
                                                            "label": label},
            "ports": [{"name": r, "role": r} for r in
                      ("DataSource", "Model", "Trainer", "ModelStore", "Evaluator")]}


@dataclass
class Chain:
    """A spec and the chain it compiles against: all plain values, mutated by the negatives."""

    spec: dict
    manifest: dict
    pipeline: dict
    layers: dict = field(default_factory=dict)
    #: stamp `input.pipeline_digest` from the pipeline as it stands when compiled, as an author
    #: does; the D1e tests turn it off to seed a missing or a stale digest
    stamp: bool = True

    def html(self) -> str:
        return "\n".join(f'<script id="layer-{k}-data" type="application/json">'
                         f"{json.dumps(v)}</script>" for k, v in self.layers.items())

    def compile(self, backends=COVERED):
        spec = self.spec
        if self.stamp and self.pipeline:
            spec = {**spec, "input": {**(spec.get("input") or {}),
                                      "pipeline_digest": canonical_digest(self.pipeline)}}
        return C.compile_spec(spec, model_html=self.html(), pipeline=self.pipeline,
                              manifest=self.manifest, backends_dir=backends)

    @property
    def sample(self) -> dict:
        return self.spec["data_contract"]["sample"]


def row_chain() -> Chain:
    feats = [_feature(n) for n in ("a", "b", "c")] + [_feature("zip", True, 5)]
    blocks = [_block("B1", "embedding", columns=["zip"], dims={"zip": 4}),
              _block("B2", "concat"), _block("B3", "dense", units=16),
              _block("B4", "activation", name="relu"), _block("B5", "dropout", p=0.1),
              _block("B6", "head", out=1, activation="none")]
    spec = _spec({"kind": "binary", "label": "late", "classes": [0, 1], "positive": 1},
                 "mlp-plain", feats, 4, {"shape": "row"}, blocks, "bce_with_logits", "pr_auc",
                 ("roc_auc",))
    manifest = {"output_schema": _schema(("a", "b", "c", "zip"), label="late"),
                "outputs": _files(400, 100, 100),
                "steps": [_split("stratified", "late"), _encode("S2", "zip", 4)]}
    return Chain(spec, manifest, _pipeline(4, "late"))


def _window_sample(**over):
    sample = {"shape": "window", "length": 24, "stride": 1, "horizon": 3,
              "series_by": "station", "order_by": "ts", "season": 24,
              "context": "previous-part", "context_source": "vega:default",
              "context_catalog": "defaults/window_context",
              "autoregressive": True, "autoregressive_source": "vega:default",
              "autoregressive_catalog": "defaults/autoregressive",
              "series_source": "shape:S2", "source": "analysis:context.time"}
    sample.update(over)
    return {k: v for k, v in sample.items() if v is not ...}


def window_chain() -> Chain:
    """A three-step forecast over stations: a nominal, a lag grouped by station."""
    feats = [_feature("temp"), _feature("wind"), _feature("weather", True, 4)]
    blocks = [_block("B1", "embedding", columns=["weather"], dims={"weather": 2}),
              _block("B2", "concat"), _block("B3", "gru", hidden=16, **{"return": "all"}),
              _block("B4", "attention", heads=2, dim=16, pool=True),
              _block("B5", "head", out=3, activation="none")]
    spec = _spec({"kind": "forecast", "label": "demand"}, "gru", feats, 3, _window_sample(),
                 blocks, "mse", "mase", ("smape",))
    manifest = {"output_schema": _schema(("temp", "wind", "weather"), ("station", "ts"),
                                         "demand"),
                "outputs": _files(600, 150, 150),
                "steps": [_split("time", None),
                          {"id": "S2", "op": "lag", "parameters_fitted": {"lags": [1]}},
                          _encode("S3", "weather", 3)]}
    layers = {"analysis": {"findings": [], "context": {"time": {"column": "ts"}}},
              "shape": {"recipe": {"steps": [{"id": "S2", "op": "lag",
                                              "params": {"group_by": "station",
                                                         "time_column": "ts"}}]}}}
    return Chain(spec, manifest, _pipeline(3, "demand"), layers)


def sequence_chain() -> Chain:
    """sequence-classification over binary, a bidirectional LSTM on complete windows."""
    ch = window_chain()
    ch.spec["task"] = {"kind": "sequence-classification", "base_kind": "binary",
                       "label": "alarm", "classes": [0, 1], "positive": 1,
                       "source": "user:turn-1"}
    ch.spec["data_contract"]["sample"] = _window_sample(
        horizon=..., autoregressive=..., autoregressive_source=...,
        autoregressive_catalog=..., season=...)
    ch.spec["family"] = {"name": "bi-lstm", **_choice("families/bi-lstm")}
    ch.spec["blocks"] = [
        _block("B1", "embedding", columns=["weather"], dims={"weather": 2}),
        _block("B2", "concat"),
        _block("B3", "lstm", hidden=16, bidirectional=True, **{"return": "last"}),
        _block("B4", "dropout", p=0.1), _block("B5", "head", out=1, activation="none")]
    ch.spec["loss"].update(name="bce_with_logits", catalog="tasks/sequence-classification")
    ch.spec["metrics"].update(primary="pr_auc", secondary=["roc_auc"],
                              selection={"monitor": "valid.pr_auc"})
    ch.spec["early_stopping"]["monitor"] = "valid.pr_auc"
    ch.manifest["output_schema"] = _schema(("temp", "wind", "weather"), ("station", "ts"),
                                           "alarm")
    ch.pipeline["feature_contract"]["label"] = "alarm"
    return ch


def _graph_sample(level="node", **over):
    sample = {"shape": "graph", "edges": "shaped/demo.edges.parquet", "level": level,
              "graph_id": "molecule_id" if level == "graph" else None,
              "directed": False, "self_loops": False,
              "protocol": "inductive", "protocol_source": "vega:default",
              "protocol_catalog": "defaults/graph_protocol",
              "batching": "graphs" if level == "graph" else "full",
              "batching_source": "vega:default", "batching_catalog": "defaults/graph_batching",
              "neighbours": None, "exposure": None, "source": "shape:manifest/graph"}
    sample.update(over)
    return sample


def _graph_block(level, graph_id=None):
    return {"step": "S7", "method": "key_pair", "level": level, "graph_id": graph_id,
            "directed": False, "symmetric": True, "self_loops": False,
            "n_nodes": {"train": 420, "valid": 60, "test": 120}, "n_edges": 1188,
            "edges": "outputs.files.edges"}


def graph_chain() -> Chain:
    """gcn at level node, inductive, full batch."""
    feats = [_feature(n) for n in ("load", "gen", "volt")]
    blocks = [_block("B1", "gcn_conv", out=32), _block("B2", "activation", name="relu"),
              _block("B3", "dropout", p=0.5), _block("B4", "gcn_conv", out=32),
              _block("B5", "head", out=1, activation="none")]
    spec = _spec({"kind": "binary", "label": "overloaded", "classes": ["ok", "overloaded"],
                  "positive": "overloaded"}, "gcn", feats, 3, _graph_sample(), blocks,
                 "bce_with_logits", "pr_auc")
    spec["training"]["batch_size"] = "full"
    manifest = {"output_schema": _schema(("load", "gen", "volt"), label="overloaded"),
                "outputs": _files(420, 60, 120, edges=1188), "graph": _graph_block("node"),
                "steps": [_split("random")]}
    return Chain(spec, manifest, _pipeline(3, "overloaded"))


def graph_level_chain() -> Chain:
    """gin at level graph: molecules grouped by molecule_id, an embedded element."""
    feats = [_feature("charge"), _feature("element", True, 5)]
    blocks = [_block("B1", "embedding", columns=["element"], dims={"element": 4}),
              _block("B2", "concat"), _block("B3", "gin_conv", out=32),
              _block("B4", "activation", name="relu"), _block("B5", "gin_conv", out=32),
              _block("B6", "graph_pool", readout="sum"),
              _block("B7", "head", out=1, activation="none")]
    spec = _spec({"kind": "binary", "label": "soluble", "classes": [0, 1], "positive": 1},
                 "gin", feats, 2, _graph_sample("graph"), blocks, "bce_with_logits", "pr_auc")
    spec["training"]["batch_size"] = 16
    manifest = {"output_schema": _schema(("charge", "element"), ("molecule_id",), "soluble"),
                "outputs": _files(450, 63, 134, edges=1174),
                "graph": _graph_block("graph", "molecule_id"),
                "steps": [_split("group", "molecule_id"), _encode("S5", "element", 4)]}
    return Chain(spec, manifest, _pipeline(2, "soluble"))


BASES = {"row": row_chain, "window": window_chain, "sequence": sequence_chain,
         "graph": graph_chain, "graph-level": graph_level_chain}


# --------------------------------------------------------------------------- assertions

def _lines(markers) -> str:
    return "\n".join(m.line(verbose=True) for m in markers)


def expect_ok(chain, backends=COVERED, allow_warn=()):
    markers = chain.compile(backends)
    assert markers.ok, f"expected the spec to compile:\n{_lines(markers.errors)}"
    stray = [m for m in markers.warns if not any(a in m.message for a in allow_warn)]
    assert not stray, f"expected no WARN:\n{_lines(stray)}"
    return markers


def expect(chain, rule, needle, level=Level.ERROR, backends=COVERED):
    """The spec is refused (for an ERROR) by `rule`, in words that contain `needle`."""
    markers = chain.compile(backends)
    hits = [m for m in markers if m.level is level and m.rule == rule
            and needle.lower() in m.message.lower()]
    assert hits, (f"expected {level} from {rule} mentioning {needle!r}; got:\n"
                  f"{_lines(markers)}")
    if level is Level.ERROR:
        assert not markers.ok
    return markers


def seeded(base, mutate):
    chain = BASES[base]()
    mutate(chain)
    return chain


# --------------------------------------------------------------------------- positives

def test_every_base_chain_compiles_clean():
    for name, make in BASES.items():
        markers = expect_ok(make())
        rules = {m.rule for m in markers.at(Level.OK)}
        assert {"T1e", "D2a", "G2"} <= rules, f"{name}: missing OK markers ({sorted(rules)})"
        if name in ("window", "sequence"):
            assert "W1" in rules, f"{name}: W1 did not say the window is declared"
        if name.startswith("graph"):
            assert "W2" in rules, f"{name}: W2 did not say the graph sample agrees"


def test_rule_registry_names_the_v23_rules():
    ids = [r.id for r in C.RULES]
    assert len(ids) == len(set(ids)), f"duplicate rule ids: {ids}"
    for rid in ("T1b", "T1e", "D2a", "D2b", "G1b", "G2", "W1", "W2", "W3", "R1", "B2", "D1e"):
        assert rid in ids, f"{rid} is not registered"


# --------------------------------------------------------------------------- W1

def test_w1_autoregressive_only_on_forecast():
    expect(seeded("sequence", lambda c: c.sample.update(autoregressive=True)),
           "W1", "forecast only")
    expect(seeded("row", lambda c: c.sample.update(autoregressive=True)),
           "W1", "forecast only")
    expect_ok(seeded("window", lambda c: c.sample.update(
        autoregressive=False, autoregressive_source="user:stance/exogenous-only",
        autoregressive_catalog=None)))


def test_w1_window_numbers():
    expect(seeded("window", lambda c: c.sample.update(length=1)), "W1", "integer >= 2")
    expect(seeded("window", lambda c: c.sample.update(stride=0)), "W1", "integer >= 1")
    expect(seeded("window", lambda c: c.sample.pop("horizon")), "W1", "horizon is required")
    expect(seeded("sequence", lambda c: c.sample.update(horizon=3)), "W1", "forecast-only")
    expect(seeded("window", lambda c: c.sample.update(season=1)), "W1", "season")


def test_w1_context_and_its_source():
    expect(seeded("window", lambda c: c.sample.update(context="rolling")), "W1",
           "previous-part")
    expect(seeded("window", lambda c: c.sample.pop("context_source")), "P1",
           "context carries no source")
    expect(seeded("window", lambda c: c.sample.update(context_catalog="defaults/nonesuch")),
           "P1", "catalog row")


def test_w1_series_by_follows_the_shaping_record():
    expect(seeded("window", lambda c: c.sample.update(series_by=None,
                                                      series_source="vega:default")),
           "W1", "names the series key 'station'")
    expect_ok(seeded("window", lambda c: c.sample.update(series_by=None,
                                                         series_source="user:turn-4")))
    expect(seeded("window", lambda c: c.sample.update(series_by="wind",
                                                      series_source="vega:default")),
           "W1", "resolves to 'station'")

    def no_key(c):
        c.sample.update(series_by=None)
        c.sample.pop("series_source")
        c.manifest["steps"] = [s for s in c.manifest["steps"] if s["op"] != "lag"]
        c.layers.pop("shape")
    markers = expect(seeded("window", no_key), "W1", "windows may cross series",
                     level=Level.WARN)
    assert markers.ok, "the unattended null series_by is a WARN, not a refusal"


def test_w1_the_series_source_resolves():
    # the series fork's own answer goes through P1 like every other fork source
    expect(seeded("window", lambda c: c.sample.update(series_source="bogus")), "P1",
           "is not one of")
    expect(seeded("window", lambda c: c.sample.update(series_source="shape:S99")), "P1",
           "does not resolve")
    expect(seeded("window", lambda c: c.sample.update(series_source="vega:default")), "P1",
           "series_catalog")
    expect_ok(seeded("window", lambda c: c.sample.update(
        series_source="vega:default", series_catalog="defaults/window_series")))


def test_w1_key_columns_are_not_features():
    def station_as_feature(c):
        c.spec["data_contract"]["features"].append(_feature("station"))
    expect(seeded("window", station_as_feature), "W1", "is a key column")
    expect(seeded("window", lambda c: c.sample.update(order_by="clock")), "W1",
           "not a column of the shaped output schema")


def test_w1_time_split_and_the_shape():
    expect(seeded("window", lambda c: c.manifest["steps"].__setitem__(0, _split("random"))),
           "W1", "needs a time split")
    expect(seeded("window", lambda c: c.spec["data_contract"].update(sample={"shape": "row"})),
           "W1", "needs windows")


# --------------------------------------------------------------------------- W2

def test_w2_graph_family_and_blocks_off_a_graph():
    expect(seeded("graph", lambda c: c.spec["data_contract"].update(sample={"shape": "row"})),
           "W2", "is a graph family")
    expect(seeded("row", lambda c: c.spec["blocks"].insert(
        2, _block("Bx", "gcn_conv", out=16))), "W2", "is a graph block")


def test_w2_the_edges_part_and_the_graph_block():
    expect(seeded("graph", lambda c: c.sample.update(edges="shaped/other.edges.parquet")),
           "W2", "is not the manifest's edges part")
    expect(seeded("graph", lambda c: c.sample.update(directed=True)), "W2", "sample.directed")

    def no_edges(c):
        c.manifest["outputs"]["files"].pop("edges")
        c.manifest.pop("graph")
    expect(seeded("graph", no_edges), "W2", "no edges part")
    # equal paths: one normalised spelling is the manifest's, a bare name or a tail is not
    expect_ok(seeded("graph", lambda c: c.sample.update(
        edges="./shaped/sub/../demo.edges.parquet")))
    for other in ("demo.edges.parquet", "elsewhere/demo.edges.parquet",
                  "/work/demo/shaped/demo.edges.parquet"):
        expect(seeded("graph", lambda c, e=other: c.sample.update(edges=e)), "W2",
               "is not the manifest's edges part")


def test_w2_graph_pool_iff_level_graph_and_gin_at_level_graph():
    expect(seeded("graph", lambda c: c.spec["blocks"].insert(
        4, _block("Bp", "graph_pool", readout="mean"))), "W2", "graph_pool at level node")
    expect(seeded("graph-level", lambda c: c.spec["blocks"].pop(5)), "W2",
           "needs a graph_pool")

    def gin_at_node(c):
        c.spec["family"] = {"name": "gin", **_choice("families/gin")}
        for b in c.spec["blocks"]:
            if b["type"] == "gcn_conv":
                b["type"] = "gin_conv"
    expect(seeded("graph", gin_at_node), "W2", "requires level 'graph'")
    expect(seeded("graph-level", lambda c: c.sample.update(graph_id=None)), "W2",
           "needs sample.graph_id")


def test_w2_protocol_and_exposure():
    expect(seeded("graph", lambda c: c.sample.update(protocol="transductive")), "W2",
           "transductive without a user: source")
    expect(seeded("graph", lambda c: c.sample.update(
        protocol="transductive", protocol_source="user:turn-5", protocol_catalog=None)),
        "W2", "needs sample.exposure")
    expect_ok(seeded("graph", lambda c: c.sample.update(
        protocol="transductive", protocol_source="user:turn-5", protocol_catalog=None,
        exposure={"kind": "features", "rows": "test", "source": "user:turn-5"})))
    expect(seeded("graph", lambda c: c.sample.update(
        exposure={"kind": "features", "rows": "test", "source": "user:turn-5"})),
        "W2", "exposure is set on an inductive protocol")
    expect(seeded("graph", lambda c: c.sample.pop("protocol")), "W2", "sample.protocol")


def test_w2_batching_and_neighbours():
    def sampled(neighbours):
        def mutate(c):
            c.sample.update(batching="sampled", neighbours=neighbours)
            c.spec["training"]["batch_size"] = 64
        return mutate
    expect(seeded("graph", sampled([10])), "W2", "one positive fan-out per conv block (2)")
    expect_ok(seeded("graph", sampled([10, 5])))
    expect(seeded("graph", lambda c: c.sample.update(batching="graphs")), "W2",
           "level-graph batching")
    expect(seeded("graph", lambda c: c.sample.update(neighbours=[10, 5])), "W2",
           "fan-out, and only its")


def test_w2_a_non_graph_family_on_a_graph_warns_but_the_floor_does_not():
    def mlp(c):
        c.spec["family"] = {"name": "mlp-plain", **_choice("families/mlp-plain")}
        c.spec["blocks"] = [_block("B1", "dense", units=16), _block("B2", "head", out=1,
                                                                    activation="none")]
    markers = expect(seeded("graph", mlp), "W2", "edges are ignored", level=Level.WARN)
    assert markers.ok

    def floor(c):
        c.spec["family"] = {"name": "linear", **_choice("families/linear")}
        c.spec["blocks"] = [_block("B1", "head", out=1, activation="none")]
    expect_ok(seeded("graph", floor))

    def floor_level(c):
        c.spec["family"] = {"name": "linear", **_choice("families/linear")}
        c.spec["blocks"] = [_block("B1", "embedding", columns=["element"], dims={"element": 4}),
                            _block("B2", "concat"), _block("B3", "graph_pool", readout="mean"),
                            _block("B4", "head", out=1, activation="none")]
    expect_ok(seeded("graph-level", floor_level))


# --------------------------------------------------------------------------- W3

def test_w3_no_bidirectional_block_on_a_forecast():
    def bidirectional(c):
        c.spec["blocks"][2]["params"]["bidirectional"] = True
    expect(seeded("window", bidirectional), "W3", "task.kind forecast")


def test_w3_no_bidirectional_block_under_a_stream_serving_mode():
    expect_ok(sequence_chain())
    expect(seeded("sequence", lambda c: c.pipeline["serving"].update(mode="stream")), "W3",
           "serving.mode stream")


def test_w3_a_bi_family_on_a_forecast_is_refused_by_name():
    expect(seeded("window", lambda c: c.spec.update(
        family={"name": "bi-gru", **_choice("families/bi-gru")})), "W3", "forbidden")


# --------------------------------------------------------------------------- G2

def test_g2_the_step_axis_is_reduced_before_the_head():
    expect(seeded("window", lambda c: c.spec["blocks"].pop(3)), "G2",
           "reaches the head unreduced")
    expect_ok(seeded("window", lambda c: c.spec["blocks"].__setitem__(
        3, _block("B4", "pool", kind="mean"))))

    def last_then_attention_pool(c):
        c.spec["blocks"][2]["params"]["return"] = "last"
        c.spec["blocks"][3] = _block("B4", "attention_pool", hidden=16)
    expect(seeded("window", last_then_attention_pool), "G2", "reads a step or token axis")


def test_g2_no_embedding_before_the_tokenizer():
    def ft(c, embed):
        c.spec["family"] = {"name": "ft-transformer", **_choice("families/ft-transformer")}
        head = [_block("B3", "feature_tokenizer", d=16),
                _block("B4", "transformer_encoder", d=16, heads=2, layers=1),
                _block("B5", "pool", kind="cls"), _block("B6", "head", out=1,
                                                         activation="none")]
        lead = [_block("B1", "embedding", columns=["zip"], dims={"zip": 4}),
                _block("B2", "concat")] if embed else []
        c.spec["blocks"] = lead + head
    expect(seeded("row", lambda c: ft(c, True)), "G2", "precedes the feature_tokenizer")
    # without the embedding the tokenizer covers the nominal (D2b) and the walk is clean
    expect_ok(seeded("row", lambda c: ft(c, False)))


def test_g2_no_conv_after_graph_pool():
    def conv_after(c):
        c.spec["blocks"].insert(6, _block("Bz", "gin_conv", out=32))
    expect(seeded("graph-level", conv_after), "G2", "follows graph_pool")


def test_g2_no_sequence_block_on_rows():
    expect(seeded("row", lambda c: c.spec["blocks"].insert(
        2, _block("Bc", "conv1d", channels=8, kernel=3))), "G2", "is a sequence block")
    expect(seeded("row", lambda c: c.spec["blocks"].insert(
        2, _block("Bt", "transformer_encoder", d=16, heads=2, layers=1))), "G2",
        "reads a step or token axis")


# --------------------------------------------------------------------------- T1b, T1e

def _reconstruction(c, family="autoencoder", kind="unsupervised", loss="reconstruction", out=8):
    c.spec["task"] = {"kind": kind, "label": None, "source": "user:turn-1"}
    c.spec["family"] = {"name": family, **_choice(f"families/{family}")}
    c.spec["blocks"] = [_block("B1", "embedding", columns=["zip"], dims={"zip": 4}),
                        _block("B2", "concat"), _block("B3", "dense", units=4),
                        _block("B4", "decoder", widths=[8]),
                        _block("B5", "head", out=out, activation="none")]
    primary = "reconstruction_error" if kind == "unsupervised" else "anomaly_score_mean"
    c.spec["loss"].update(name=loss, catalog=f"tasks/{kind}")
    c.spec["metrics"].update(primary=primary, secondary=[],
                             selection={"monitor": f"valid.{primary}"}, catalog=f"tasks/{kind}")
    c.spec["early_stopping"]["monitor"] = f"valid.{primary}"
    c.manifest["output_schema"].pop("late")
    c.pipeline["feature_contract"]["label"] = None


def test_t1b_recon_dim_under_reconstruction():
    # 3 numeric columns + the embedded zip's 5 categories = 8
    expect_ok(seeded("row", _reconstruction))
    expect(seeded("row", lambda c: _reconstruction(c, out=4)), "T1b", "requires 8")
    expect(seeded("row", lambda c: _reconstruction(c, kind="anomaly", out=4)), "T1b",
           "requires 8")


def test_t1b_svdd_z_is_any_positive_width():
    def svdd(out):
        def mutate(c):
            _reconstruction(c, family="deep-svdd", kind="anomaly", loss="svdd", out=out)
            c.spec["blocks"] = [_block("B1", "embedding", columns=["zip"], dims={"zip": 4}),
                                _block("B2", "concat"), _block("B3", "dense", units=16),
                                _block("B4", "head", out=out, activation="none", bias=False)]
        return mutate
    expect_ok(seeded("row", svdd(6)))
    expect(seeded("row", svdd(0)), "T1b", "positive integer")


def test_t1b_forecast_quantiles_and_sequence_base_and_ordinal():
    def quantile(c):
        c.spec["loss"].update(name="quantile", params={"quantiles": [0.1, 0.5, 0.9]})
    expect(seeded("window", quantile), "T1b", "requires 9")
    expect_ok(seeded("window", lambda c: (quantile(c), c.spec["blocks"][4]["params"]
                                          .update(out=9))))
    expect(seeded("sequence", lambda c: c.spec["blocks"][4]["params"].update(out=2)), "T1b",
           "base task 'binary'")

    def ordinal(c):
        c.spec["task"].update(kind="ordinal", classes=[0, 1, 2, 3], positive=None)
        c.spec["loss"].update(name="cumulative_link", catalog="tasks/ordinal")
        c.spec["metrics"].update(primary="mae_ordinal", secondary=[],
                                 selection={"monitor": "valid.mae_ordinal"},
                                 catalog="tasks/ordinal")
        c.spec["early_stopping"]["monitor"] = "valid.mae_ordinal"
        c.spec["blocks"][-1]["params"]["out"] = 4
    expect(seeded("row", ordinal), "T1b", "requires 3")


def test_t1e_task_fields():
    expect(seeded("sequence", lambda c: c.spec["task"].pop("base_kind")), "T1e",
           "needs task.base_kind")
    expect(seeded("sequence", lambda c: c.spec["task"].update(base_kind="regression")),
           "T1e", "is not a base of sequence-classification")
    expect(seeded("row", lambda c: c.spec["task"].update(base_kind="binary")), "T1e",
           "required iff the kind starts with sequence-")
    expect(seeded("row", lambda c: c.spec["task"].update(positive=7)), "T1e",
           "is not one of task.classes")
    expect(seeded("row", lambda c: c.spec["task"].pop("classes")), "T1e", "needs task.classes")

    def multilabel(fmt):
        def mutate(c):
            c.spec["task"] = {"kind": "multilabel", "label": "tags", "classes": ["a", "b", "c"],
                              "source": "user:turn-1", **({"label_format": fmt} if fmt else {})}
            c.spec["blocks"][-1]["params"]["out"] = 3
            c.spec["loss"]["catalog"] = "tasks/multilabel"
            c.spec["metrics"].update(primary="f1_micro", secondary=[],
                                     selection={"monitor": "valid.f1_micro"},
                                     catalog="tasks/multilabel")
            c.spec["early_stopping"]["monitor"] = "valid.f1_micro"
            c.manifest["output_schema"]["tags"] = c.manifest["output_schema"].pop("late")
            c.pipeline["feature_contract"]["label"] = "tags"
        return mutate
    expect(seeded("row", multilabel(None)), "T1e", "label_format")
    expect(seeded("row", multilabel({"kind": "names"})), "T1e", "kind: delimited")
    expect_ok(seeded("row", multilabel({"kind": "delimited", "sep": "|"})))

    def anomaly_labelled(c):
        _reconstruction(c, kind="anomaly")
        c.spec["task"].update(label="fault", classes=["ok", "fault"])
        c.manifest["output_schema"]["fault"] = {"dtype": "int64", "role": "label"}
        c.pipeline["feature_contract"]["label"] = "fault"
        c.spec["metrics"].update(primary="pr_auc", selection={"monitor": "valid.pr_auc"})
        c.spec["early_stopping"]["monitor"] = "valid.pr_auc"
    expect(seeded("row", anomaly_labelled), "T1e", "needs task.positive")
    expect_ok(seeded("row", lambda c: (anomaly_labelled(c),
                                       c.spec["task"].update(positive="fault"))))


def test_t1c_reads_the_base_row():
    expect(seeded("sequence", lambda c: c.spec["loss"].update(name="cross_entropy")), "T1c",
           "base binary")


# --------------------------------------------------------------------------- D2a, D2b

def test_d2a_keys_are_not_counted_as_features():
    expect(seeded("window", lambda c: c.spec["data_contract"].update(input_dim=5)), "D2a",
           "key column(s) station, ts")
    expect(seeded("window", lambda c: c.spec["data_contract"].update(keys=["station"])),
           "D2a", "keys are derived")
    expect_ok(seeded("window", lambda c: c.spec["data_contract"].update(
        keys=["ts", "station"])))
    expect(seeded("graph-level", lambda c: c.spec["data_contract"].update(input_dim=3)),
           "D2a", "key column(s) molecule_id")


def test_d2b_every_nominal_is_embedded_tokenized_or_one_hot():
    def uncovered(c):
        c.spec["blocks"] = c.spec["blocks"][2:]
    expect(seeded("row", uncovered), "D2b", "invents an order that does not exist")

    def one_hot(c):
        uncovered(c)
        c.manifest["steps"][1]["parameters"] = {"scheme": "one_hot"}
    expect_ok(seeded("row", one_hot))


# --------------------------------------------------------------------------- R1

def test_r1_full_iff_batching_full():
    expect(seeded("row", lambda c: c.spec["training"].update(batch_size="full")), "R1",
           "goes with batching full")
    expect(seeded("graph", lambda c: c.spec["training"].update(batch_size=64)), "R1",
           "goes with batching full")
    expect(seeded("graph", lambda c: (c.sample.update(batching="sampled", neighbours=[5, 5]))),
           "R1", "goes with batching full")


def test_r1_batch_at_most_the_training_targets():
    expect(seeded("row", lambda c: c.spec["training"].update(batch_size=4096)), "R1",
           "exceeds the 400 training rows")
    expect(seeded("window", lambda c: (c.sample.update(stride=4),
                                       c.spec["training"].update(batch_size=200))),
           "R1", "exceeds the 150 training windows")
    expect(seeded("graph-level", lambda c: c.spec["training"].update(batch_size=512)), "R1",
           "training graphs at most")

    def sampled(c):
        c.sample.update(batching="sampled", neighbours=[5, 5])
        c.spec["training"]["batch_size"] = 500
    expect(seeded("graph", sampled), "R1", "exceeds the 420 training seed nodes")
    expect(seeded("row", lambda c: c.spec["training"].update(batch_size=0)), "R1",
           "positive integer")


# --------------------------------------------------------------------------- B2, G1b

def _backends_planning(blocks, obligation="B-graph-recurrent") -> Path:
    """A backends dir whose pytorch coverage lists `blocks` as planned under `obligation` (the
    shipped coverage plans nothing since gate A)."""
    root = WORK / f"backends-planning-{'-'.join(blocks)}" / "pytorch"
    if not root.is_dir():
        root.mkdir(parents=True)
        cov = json.loads((COVERED / "pytorch" / "coverage.json").read_text(encoding="utf-8"))
        cov["blocks"] = [b for b in cov["blocks"] if b not in blocks]
        cov["planned"] = {"note": f"build obligation {obligation}: not built yet",
                          "blocks": list(blocks), "families": []}
        (root / "coverage.json").write_text(json.dumps(cov), encoding="utf-8")
    return root.parent


def test_b2_names_a_planned_block_as_planned():
    expect(graph_chain(), "B2", "planned in backend pytorch (build obligation "
           "B-graph-recurrent)", backends=_backends_planning(["gcn_conv"]))
    assert not (SHIPPED.get("planned") or {}).get("blocks"), "the shipped backend plans nothing"
    expect(window_chain(), "B2", "not covered by the pytorch backend",
           backends=_backends_without("gru"))


def _backends_without(block, key="blocks") -> Path:
    root = WORK / f"backends-no-{key}-{block}" / "pytorch"
    if not root.is_dir():
        root.mkdir(parents=True)
        cov = json.loads((COVERED / "pytorch" / "coverage.json").read_text(encoding="utf-8"))
        cov[key] = [b for b in cov[key] if b != block]
        (root / "coverage.json").write_text(json.dumps(cov), encoding="utf-8")
    return root.parent


def test_b2_checks_what_the_sample_asks_of_the_backend():
    expect(graph_chain(), "B2", "sample shape 'graph' is not covered",
           backends=_backends_without("graph", "sample_shapes"))
    expect(graph_chain(), "B2", "graph batching 'full' is not covered",
           backends=_backends_without("full", "graph_batching"))
    expect(window_chain(), "B2", "recurrent return (B3) 'all' is not covered",
           backends=_backends_without("all", "recurrent_returns"))


def test_b2_refuses_a_loss_parameter_the_backend_loss_does_not_take():
    """Final rehearsal round 2: a spec wrote `loss.params.class_weights` for cross_entropy, whose
    backend name is `weight`; the spec compiled, the smoke pass (which builds no loss) passed,
    and `build_loss` would have refused it at the first step of the fit. B2 names the key."""
    def multiclass_weights(key):
        def mutate(c):
            c.spec["task"].update(kind="multiclass", classes=["a", "b", "c"], positive=None)
            c.spec["loss"].update(name="cross_entropy", catalog="tasks/multiclass",
                                  params={key: [1.0, 2.0, 0.5]})
        return mutate
    expect(seeded("row", multiclass_weights("class_weights")), "B2",
           "takes no parameter 'class_weights'; it takes label_smoothing, weight")
    markers = seeded("row", multiclass_weights("weight")).compile(COVERED)
    assert not [m for m in markers if m.rule == "B2"], _lines(markers)
    # a key a weighed alternative carries (a ladder merge) is warned, as build_loss drops it
    def carried(c):
        c.spec["loss"].update(params={"gamma": 2.0},
                              alternatives=[{"name": "focal", "params": {"gamma": 2.0}}])
    expect(seeded("row", carried), "B2", "a weighed alternative", level=Level.WARN)
    expect(seeded("row", lambda c: c.spec["loss"].update(params={"pos_weight": 3.0, "gama": 2})),
           "B2", "takes no parameter 'gama'")


def test_b2_loss_params_are_the_backend_losses_own():
    """coverage.json `loss_params` is the losses' own PARAMS, read out of losses.py."""
    import ast
    tree = ast.parse((SCRIPTS / "backends" / "pytorch" / "losses.py").read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name == "_Loss":
            continue
        values = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                target = stmt.targets[0]
                names = [e.id for e in target.elts] if isinstance(target, ast.Tuple) else [target.id]
                vals = (stmt.value.elts if isinstance(target, ast.Tuple) else [stmt.value])
                for k, v in zip(names, vals):
                    try:
                        values[k] = ast.literal_eval(v)
                    except ValueError:
                        pass
        if "name" in values:
            found[values["name"]] = sorted(values.get("PARAMS") or ())
    assert {k: sorted(v) for k, v in SHIPPED["loss_params"].items()} == found, found
    assert sorted(SHIPPED["loss_params"]) == sorted(SHIPPED["losses"])


def test_g1b_refuses_what_the_admission_table_does_not_admit():
    def autoencoder_on_window(c):
        c.spec["family"] = {"name": "autoencoder", **_choice("families/autoencoder")}
    expect(seeded("window", autoencoder_on_window), "G1b", "not admitted")

    def gcn_on_anomaly(c):
        c.spec["task"] = {"kind": "anomaly", "label": None, "source": "user:turn-1"}
    expect(seeded("graph", gcn_on_anomaly), "G1b", "graph family on unsupervised or anomaly")


def test_g1_markers_carry_their_specific_ids():
    """The G1 family is emitted as G1a–G1d (netspec-contract §3), never as a bare G1."""
    expect(seeded("row", lambda c: c.spec["blocks"][2].update(type="densse")), "G1a",
           "nearest known type: 'dense'")

    def unknown_family(c):
        c.spec["family"] = {"name": "my-net", "source": "user:turn-2", "rationale": "mine",
                            "alternatives": []}
        c.spec["blocks"][2]["type"] = "densse"
    expect(seeded("row", unknown_family), "G1b", "not all known")

    def widening_residual(c):
        c.spec["blocks"].insert(4, _block("Br", "residual", projection=None, inner=[
            {"type": "dense", "params": {"units": 16}},
            {"type": "dense", "params": {"units": 8}}]))
    expect(seeded("row", widening_residual), "G1c", "change width")

    def default_custom(c):
        custom = _block("Bc", "custom", code_ref="custom/g.py", out_shape=[16])
        c.spec["blocks"].insert(4, custom)
    markers = expect(seeded("row", default_custom), "G1d", "may only carry a user: source")
    assert not [m for m in markers if m.rule == "G1"], _lines(markers)


# --------------------------------------------------------------------------- ShapeWalk

def _walk(shape, blocks, level=None):
    return ShapeWalk.for_spec({"data_contract": {"sample": {"shape": shape, "level": level}}},
                              CATALOG["blocks"]).walk(blocks)


# the memory template's Network stances section, as domain-forge ships it: its examples sit in
# an HTML comment under the heading (the T5 rehearsal, round 2, met them read as stances)
TEMPLATE_STANCES = """## Network stances
<!-- Written by the Vega sub-chain (/vega-forge, /vega-train, /vega-eval, /vega-pack); read by all four.
     One line per fork resolved, per campaign, per evaluation, per bundle:
       YYYY-MM-DD | <dataset> | family: mlp-residual (user); dropout B3: 0.2; loss: bce_with_logits pos_weight 4.9
       YYYY-MM-DD | <dataset> | declined: ft-transformer (too heavy for 420 rows)
       YYYY-MM-DD | <dataset> | campaign 2026-09-05T11-03Z: ladder R0\u2013R3 + H1 approved; selected H1-14, valid pr_auc 0.74 -->
"""


def test_m2_reads_no_stance_out_of_the_templates_comment():
    """The template's illustrative lines are a comment, not stances: a fresh memory has none,
    so a multiclass spec's cross_entropy is not refused against the comment's binary loss, and a
    stance written below the comment is still read and still honoured (M2)."""
    memory = WORK / "m2-memory" / "domain-forge-memory.md"
    memory.parent.mkdir(parents=True, exist_ok=True)
    memory.write_text("# /domain-forge project memory\n\n" + TEMPLATE_STANCES
                      + "\n## Shaping stances\n<!-- YYYY-MM-DD | <dataset> | split: 70/10/20 -->\n",
                      encoding="utf-8")
    assert C.read_stances(memory) == (), C.read_stances(memory)
    m2 = next(r.check for r in C.RULES if r.id == "M2")

    class Ctx:
        inputs = type("I", (), {"stances": C.read_stances(memory)})
        family_name = "mlp-plain"
        spec = {"loss": {"name": "cross_entropy", "source": "vega:default"},
                "family": {"name": "mlp-plain", "source": "vega:default"}}
    assert m2(Ctx) == [], m2(Ctx)
    line = "2026-10-10 | landcover | loss: focal (declared)"
    memory.write_text("# /domain-forge project memory\n\n" + TEMPLATE_STANCES + f"- {line}\n",
                      encoding="utf-8")
    assert C.read_stances(memory) == (line,), C.read_stances(memory)
    Ctx.inputs = type("I", (), {"stances": C.read_stances(memory)})
    found = m2(Ctx)
    assert len(found) == 1 and "'focal'" in str(found[0]), found


def test_m2_reads_a_stance_key_not_a_word_inside_prose():
    """The T5 round 3 rehearsal: /vega-eval wrote `eval: linear is better calibrated (log_loss
    0.2995 vs 0.3331, ...)` to Network stances, and M2 read `loss 0` out of `log_loss 0.2995`,
    so the published spec no longer compiled against its own memory. A stance is a `key:
    value` pair: `loss: <name>`, `declined: <family>`."""
    m2 = next(r.check for r in C.RULES if r.id == "M2")
    prose = ("2026-10-11 | landcover | eval: linear is better calibrated than the network "
             "(log_loss 0.2995 vs 0.3331, ece 0.0362 vs 0.0447); the margin declined to 0.01")

    class Ctx:
        inputs = type("I", (), {"stances": (prose,)})
        family_name = "mlp-plain"
        spec = {"loss": {"name": "cross_entropy", "source": "vega:default"},
                "family": {"name": "mlp-plain", "source": "vega:default"}}
    assert m2(Ctx) == [], m2(Ctx)
    Ctx.inputs = type("I", (), {"stances": (prose, "2026-10-11 | landcover | declined: mlp-plain")})
    found = m2(Ctx)
    assert len(found) == 1 and "declined" in str(found[0]), found


def test_shape_walk_reads_window_only_from_the_catalog():
    walk = ShapeWalk.for_spec({"data_contract": {"sample": {"shape": "row"}}},
                              CATALOG["blocks"])
    assert walk.window_only == frozenset(k for k, v in CATALOG["blocks"].items()
                                         if v.get("window_only")), walk.window_only
    assert {"conv1d", "rnn", "gru", "lstm", "attention_pool"} <= walk.window_only


def test_shape_walk_composites_reshape_and_custom():
    # L05's composition: parallel -> reshape [4, 16] -> pool -> custom [16] -> head
    l05 = [{"id": "B1", "type": "parallel", "params": {"merge": "concat", "inner": [
               {"type": "dense", "params": {"units": 32}},
               {"type": "sequential", "params": {"inner": [
                   {"type": "dense", "params": {"units": 32}},
                   {"type": "layernorm", "params": {}}]}}]}},
           {"id": "B2", "type": "reshape", "params": {"shape": [4, 16]}},
           {"id": "B3", "type": "pool", "params": {"kind": "mean"}},
           {"id": "B4", "type": "custom", "params": {"code_ref": "custom/g.py",
                                                     "out_shape": [16]}},
           {"id": "B5", "type": "head", "params": {"out": 1}}]
    assert _walk("row", l05) == ()
    unpooled = [b for b in l05 if b["id"] not in ("B3", "B4")]
    kinds = [f.kind for f in _walk("row", unpooled)]
    assert kinds == ["unreduced"], kinds
    split = [{"id": "P", "type": "parallel", "params": {"inner": [
        {"type": "gru", "params": {"hidden": 8, "return": "all"}},
        {"type": "gru", "params": {"hidden": 8, "return": "last"}}]}},
        {"id": "H", "type": "head", "params": {"out": 1}}]
    assert "parallel_axes" in [f.kind for f in _walk("window", split)]
    residual = [{"id": "R", "type": "residual", "params": {"repeat": 2, "inner": [
        {"type": "gru", "params": {"hidden": 8, "return": "last"}}]}},
        {"id": "H", "type": "head", "params": {"out": 1}}]
    found = _walk("window", residual)
    assert [f.kind for f in found][:1] == ["residual_axes"], found
    assert found[0].where == "blocks[R]"


def test_shape_walk_names_inner_blocks_as_walk_blocks_does():
    blocks = [{"id": "S", "type": "sequential", "params": {"inner": [
        {"type": "dense", "params": {"units": 8}},
        {"type": "conv1d", "params": {"channels": 4, "kernel": 3}}]}},
        {"id": "H", "type": "head", "params": {"out": 1}}]
    found = _walk("row", blocks)
    assert [f.where for f in found] == ["blocks[S].inner.blocks[1]"], found


# --------------------------------------------------------------------------- D1e, D1b digests

def test_d1e_the_pipeline_digest_is_the_emitters_canonical_digest():
    import importlib.util
    spec_ = importlib.util.spec_from_file_location(
        "forge_emit_digest", SCRIPTS / "backends" / "pytorch" / "emit.py")
    emit = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(emit)
    doc = {"b": [1, {"é": 2}], "a": None}
    assert canonical_digest(doc) == emit._canonical_digest(doc)
    ch = row_chain()
    ch.stamp = False
    ch.spec["input"]["pipeline_digest"] = canonical_digest(ch.pipeline)
    markers = expect_ok(ch)
    assert any(m.rule == "D1e" for m in markers.at(Level.OK)), _lines(markers)


def test_d1e_a_missing_or_stale_pipeline_digest_is_refused():
    ch = row_chain()
    ch.stamp = False
    expect(ch, "D1e", "input.pipeline_digest is missing")
    ch.spec["input"]["pipeline_digest"] = canonical_digest(ch.pipeline)
    ch.pipeline["serving"]["mode"] = "online"              # the blueprint moved on
    expect(ch, "D1e", "digests to " + canonical_digest(ch.pipeline))
    # the file's bytes are not the definition: an indented pipeline.json digests otherwise
    import hashlib
    ch.spec["input"]["pipeline_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(ch.pipeline, indent=2).encode()).hexdigest()
    expect(ch, "D1e", "not of the file's bytes")


def _shape_layer(digest):
    return {"schema": "dataset-shaper/shape@1", "manifest_digest": digest}


def test_d1b_one_manifest_digest_or_an_error():
    good, other = "sha256:" + "1" * 64, "sha256:" + "2" * 64

    def chain(drawn, cited, recorded):
        ch = row_chain()
        ch.pipeline["input"] = {"shape_manifest_digest": drawn}
        ch.spec["input"]["manifest_digest"] = cited
        if recorded:
            ch.layers["shape"] = _shape_layer(recorded)
        return ch

    expect_ok(chain(good, good, good))
    expect_ok(chain(good, good, None))
    expect(chain(other, good, None), "D1b", "blueprint older than the shape")
    expect(chain(other, good, good), "D1b", "the shape layer records " + good)
    expect(chain(good, other, good), "D1b", "copy the shape layer's manifest_digest")


# --------------------------------------------------------------------------- P1 geometry

_GEOMETRY = {"typing": [{"column": "zip", "kind": "nominal"}],
             "partitions": {"chosen": "late", "provenance": "user-chosen",
                            "candidates": [{"label": "late", "task": "binary classification",
                                            "dropped_for_leakage": ["delivered_days"]},
                                           {"label": "cover", "task": "multiclass "
                                                                      "classification"}]}}


def _resolver(geometry=_GEOMETRY):
    inputs = ChainInputs.of(model_html='<script id="layer-geometry-data" type="application/'
                                       f'json">{json.dumps(geometry)}</script>')
    return SourceResolver(inputs, Catalog(CATALOG))


def test_p1_geometry_partitions_resolve_by_candidate_label():
    r = _resolver()
    for src in ("geometry:partitions/late", "geometry:partitions/cover",
                "geometry:partitions/chosen", "geometry:partitions/chosen/task",
                "geometry:partitions/late.dropped_for_leakage",
                "geometry:partitions/cover/task", "geometry:typing/zip"):
        assert r.well_formed(src), src
        resolved, detail = r.resolve(src)
        assert resolved, f"{src}: {detail}"
    # the label is read as a candidate's, never as a key of `partitions`: `chosen` names one
    assert "'late'" in r.resolve("geometry:partitions/chosen")[1]


def test_p1_geometry_partitions_refuse_what_is_not_there():
    r = _resolver()
    for src, needle in (("geometry:partitions/nonesuch", "no partition candidate labelled"),
                        ("geometry:partitions/late.nonesuch", "has no 'nonesuch'"),
                        ("geometry:partitions/cover/dropped_for_leakage", "has no"),
                        ("geometry:typing/nonesuch", "typing of 'nonesuch'")):
        resolved, detail = r.resolve(src)
        assert not resolved and needle in detail, f"{src}: {resolved} {detail}"
    abstained = {**_GEOMETRY, "partitions": {**_GEOMETRY["partitions"], "chosen": None,
                                             "provenance": "abstained"}}
    resolved, detail = _resolver(abstained).resolve("geometry:partitions/chosen")
    assert not resolved and "chose no partition" in detail, detail
    # through the compiler: the task cites a label the geometry does not hold
    ch = row_chain()
    ch.layers["geometry"] = _GEOMETRY
    head = next(b for b in ch.spec["blocks"] if b["type"] == "head")
    head.update(source="geometry:partitions/late", rationale="one logit for the binary label")
    head.pop("catalog", None)
    expect_ok(ch)
    head["source"] = "geometry:partitions/overloaded"
    expect(ch, "P1", "no partition candidate labelled 'overloaded'")


# --------------------------------------------------------------------------- ChainInputs

def test_chain_inputs_read_the_executed_split_and_the_series_key():
    ch = window_chain()
    inputs = ChainInputs.of(model_html=ch.html(), manifest=ch.manifest)
    assert inputs.split_kind() == "time"
    assert inputs.series_key() == ("station", "shape:S2")
    ch.manifest["steps"][0] = {"id": "S1", "op": "split",
                               "parameters_fitted": {"kind": "time", "by": "region"}}
    ch.manifest["steps"].pop(1)
    inputs = ChainInputs.of(model_html="", manifest=ch.manifest)
    assert inputs.series_key() == ("region", "shape:S1")
    assert ChainInputs.of(manifest=row_chain().manifest).series_key() == (None, None)
    assert Catalog(CATALOG).is_floor_on("linear", "graph")
    assert "gcn" in Catalog(CATALOG).families_on("graph")


# --------------------------------------------------------------------------- PackageAdapters

_ADAPTER_STUBS = {
    "network.py": '''"""stub Model adapter"""
class Network:
    built = []

    @staticmethod
    def build(spec, run_dir=None):
        Network.built.append(spec)
        return ("network", spec)


build_model = Network.build
''',
    "data.py": '"""stub DataSource adapter"""\nclass ShapedParts:\n    pass\n',
    "loop.py": '"""stub Trainer adapter"""\nclass Trainer:\n    def __init__(self, store=None):\n'
               '        self.store = store\n',
    "store.py": '"""stub ModelStore adapter"""\nclass ModelRegistry:\n    pass\n',
    "evaluate.py": '''"""stub Evaluator adapter"""
class Evaluator:
    def __init__(self, data, store, build):
        self.ports = (data, store, build)

    @classmethod
    def restoring(cls, data, store, build):
        return cls(data, store, build)
''',
    "readout.py": '"""stub readout"""\nclass TaskReadout:\n    pass\n',
    "metrics.py": '"""stub metrics"""\nMETRICS = {}\n',
    "baselines.py": '"""stub baselines"""\nNAME = "baselines"\n',
    "baselines_extra.py": '"""stub baselines, extra"""\nNAME = "extra"\n',
}


def _rendered_v23(tag, drop=(), override=None):
    import render_templates as rt
    stubs = dict(_ADAPTER_STUBS, **(override or {}))
    src = WORK / f"set-{tag}"
    src.mkdir()
    for name, text in stubs.items():
        if name not in drop:
            (src / name).write_text(text, encoding="utf-8")
    out = WORK / f"out-{tag}"
    record = rt.render_partial(src, out, rt.reference_pipeline(), rt.reference_spec("row"))
    return Path(record["root"]), record["package"]


def test_package_adapters_instantiate_a_v23_package_by_role():
    root, package = _rendered_v23("ok")
    a = PackageAdapters.load(root, package)
    assert a.generation == "v2.3"
    assert type(a.data).__name__ == "ShapedParts" and type(a.trainer).__name__ == "Trainer"
    assert type(a.store).__name__ == "ModelRegistry"
    assert a.trainer.store is a.store, "the Trainer writes its checkpoints through the store"
    assert type(a.evaluator).__name__ == "Evaluator"
    assert a.evaluator.ports == (a.data, a.store, a.build), "the Evaluator's ports are wired"
    assert a.build("spec") == ("network", "spec")
    assert a.readout.__name__ == "TaskReadout"
    assert a.metrics.__name__ == f"{package}.model.metrics"
    assert a.baselines.NAME == "baselines", "the baselines module, not baselines_extra"
    assert set(a.roles) == {"Model", "DataSource", "Trainer", "ModelStore", "Evaluator"}


def test_package_adapters_reimport_a_re_emitted_package():
    root, package = _rendered_v23("first")
    first = PackageAdapters.load(root, package)
    second = PackageAdapters.load(root, package)
    assert first.network_module is not second.network_module, "the import was not fresh"


def test_package_adapters_refuse_what_the_map_does_not_hold():
    root, package = _rendered_v23("noclass",
                                  override={"loop.py": '"""no class"""\ndef fit():\n    pass\n'})
    _raises(AdapterLoadError, "holds no class 'Trainer'", lambda: PackageAdapters.load(root,
                                                                                      package))
    root, package = _rendered_v23("noreadout", drop=("readout.py",))
    _raises(AdapterLoadError, "readout.py is missing", lambda: PackageAdapters.load(root,
                                                                                    package))
    root, package = _rendered_v23("nomap")
    bmap = json.loads((root / "blueprint-map.json").read_text(encoding="utf-8"))
    bmap["entries"] = [e for e in bmap["entries"] if e.get("role") != "Trainer"]
    (root / "blueprint-map.json").write_text(json.dumps(bmap), encoding="utf-8")
    _raises(AdapterLoadError, "maps no file to the Trainer role",
            lambda: PackageAdapters.load(root, package))
    _raises(AdapterLoadError, "blueprint-map.json cannot be read",
            lambda: PackageAdapters.load(WORK / "nowhere" / package, package))


def test_package_adapters_drive_an_emitted_v22_package_as_modules():
    import render_templates as rt
    out = WORK / "v22"
    record = rt.baseline_emitter(WORK / "c849f8b").emit(rt.reference_spec("row"),
                                                       rt.reference_pipeline(), out)
    root, package = Path(record["root"]), record["package"]
    a = PackageAdapters.load(root, package)
    assert a.generation == "v2.2" and a.readout is None
    assert a.data.__name__ == f"{package}.ingestion.data"
    assert a.trainer.__name__ == f"{package}.training.loop"
    assert a.build is a.network_module.build_model
    assert a.evaluator is not None and a.baselines is not None


# --------------------------------------------------------------------------- runner

def _raises(kind, needle, fn):
    try:
        fn()
    except kind as exc:
        assert needle in str(exc), f"{kind.__name__} raised without {needle!r}: {exc}"
        return
    raise AssertionError(f"expected {kind.__name__} mentioning {needle!r}")


def main(argv) -> int:
    pick = argv[argv.index("-k") + 1] if "-k" in argv else ""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and pick in n]
    failed = []
    for name, fn in tests:
        try:
            fn()
            if VERBOSE:
                print(f"  ok    {name}")
        except Exception as exc:                            # a failing test is reported, not raised
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    shutil.rmtree(WORK, ignore_errors=True)
    if failed:
        print(f"\n{len(failed)} of {len(tests)} compiler rule test(s) failed")
        return 1
    print(f"\n{len(tests)} compiler rule test(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
