"""
The netspec composer's unit tests (BUILD-PLAN T02, R-A15): the P1 fields each sample shape
carries, `training.batch_size: "full"` iff full batching, the data contract's keys, the
sources of the defaulted choices, the memory stances a composed spec needs, and the budget of
the lanes file a lane came from. The catalog readings that need no lane are
test_lanekit_catalog.py's.

    python3 -m pytest -q vega/scripts/tests/test_lanekit_spec.py

A spec is composed here over a shaped directory written by the test itself (a manifest with
its output schema, the recipe copy the shaper leaves beside it, a train part), so the rules are
proved without running the shaper; `compose_check.py` runs the same composer over the real
shaper's output for every lane of `lanes.json` and hands each spec to `compile_spec.py`.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import registry, spec  # noqa: E402
from lanekit.fixture import memory_for  # noqa: E402
from lanekit.recipes import embedded_columns  # noqa: E402

LANES = registry.load_lanes()
LANES_DEFAULTS = registry.load_document()["defaults"]
COMPOSED = [lane for lane in LANES.values() if lane.driver == "lanekit"]
WINDOW_P1 = ("length", "stride", "series_by", "order_by", "context", "autoregressive", "season",
             "source", "context_source", "autoregressive_source", "series_source")
GRAPH_P1 = ("edges", "level", "graph_id", "directed", "self_loops", "protocol", "protocol_source",
            "batching", "batching_source", "neighbours", "exposure", "source")
TRAIN_ROWS = 240


# --------------------------------------------------------------------------- a shaped directory
def _schema_for(raw: dict) -> tuple[dict, list[str], dict]:
    """(output_schema, embedded nominals, encode fitted) the shaper would write for a lane:
    three numeric features, each nominal the network embeds (ordinal, with an unseen id), the
    lane's key columns and its label."""
    schema: dict[str, dict] = {}
    window, graph = raw.get("window") or {}, raw.get("graph") or {}
    if raw["shape"] == "window":
        if window.get("series_by"):
            schema[window["series_by"]] = {"dtype": "string", "role": "feature", "origin": "S01"}
        schema[window["order_by"]] = {"dtype": "datetime64[us]", "role": "feature", "origin": "S02"}
    if raw["shape"] == "graph" and graph.get("graph_id"):
        schema[graph["graph_id"]] = {"dtype": "string", "role": "feature", "origin": "input"}
    for i in range(1, 4):
        schema[f"x{i}"] = {"dtype": "float64", "role": "feature", "origin": "S09"}
    nominals = list(embedded_columns(raw.get("blocks") or []))
    if any(b.get("type") == "feature_tokenizer" for b in raw.get("blocks") or []):
        nominals += ["n1", "n2"]
    fitted = {}
    for name in dict.fromkeys(nominals):
        schema[name] = {"dtype": "Int64", "role": "feature", "origin": "S08"}
        fitted[name] = {"mapping": {"a": 0, "b": 1, "c": 2}, "n_categories": 3, "unseen_category_id": 3}
    label = (raw.get("target") or {}).get("column")
    if label:
        classes = (raw.get("target") or {}).get("classes")
        dtype = "int64" if classes and all(str(c).isdigit() for c in classes) else (
            "string" if classes else "float64")
        schema[label] = {"dtype": dtype, "role": "label", "origin": "input"}
    return schema, list(dict.fromkeys(nominals)), fitted


def _train_frame(raw: dict, schema: dict, train_rows: int):
    """The train part: the numeric features, three series (a window lane with a series key),
    ten nodes a graph (a graph lane with a graph id), and the label cycling through its classes."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    frame = pd.DataFrame({c: rng.normal(size=train_rows) for c in schema if c.startswith("x")})
    window, graph = raw.get("window") or {}, raw.get("graph") or {}
    if raw["shape"] == "window" and window.get("series_by"):
        frame[window["series_by"]] = np.repeat(["s0", "s1", "s2"], train_rows // 3 + 1)[:train_rows]
    if raw["shape"] == "graph" and graph.get("graph_id"):
        frame[graph["graph_id"]] = [f"G{i // 10:03d}" for i in range(train_rows)]
    label = (raw.get("target") or {}).get("column")
    classes = (raw.get("target") or {}).get("classes")
    if label:
        frame[label] = [classes[i % len(classes)] for i in range(train_rows)] if classes else rng.normal(size=train_rows)
    return frame


def shaped_dir(tmp: Path, lane, *, graph_block: dict | None = None, recipe_steps: list | None = None,
               train_rows: int = TRAIN_ROWS) -> tuple[dict, dict, Path]:
    """Write `<tmp>/shaped/{manifest.json, recipe.json, lane.train.parquet}` and a pipeline;
    returns (manifest, fitted, pipeline path)."""
    raw = lane.raw if hasattr(lane, "raw") else lane
    shaped = tmp / "shaped"
    shaped.mkdir(parents=True, exist_ok=True)
    schema, _, encoded = _schema_for(raw)
    train = shaped / "lane.train.parquet"
    _train_frame(raw, schema, train_rows).to_parquet(train)
    steps = [{"id": "S08", "op": "encode", "parameters_fitted": {"strategy": "ordinal", "fitted": encoded}}]
    files = {"train": {"path": str(train), "rows": train_rows},
             "valid": {"path": str(shaped / "lane.valid.parquet"), "rows": train_rows // 4},
             "test": {"path": str(shaped / "lane.test.parquet"), "rows": train_rows // 4}}
    manifest = {"schema": "dataset-shaper/manifest@1", "label": (raw.get("target") or {}).get("column"),
                "steps": steps, "output_schema": schema, "outputs": {"dir": str(shaped), "files": files}}
    if raw["shape"] == "graph":
        graph = raw["graph"]
        files["edges"] = {"path": str(shaped / "lane.edges.parquet"), "rows": 3 * train_rows}
        manifest["graph"] = graph_block or {"level": graph["level"], "graph_id": graph.get("graph_id"),
                                            "directed": False, "self_loops": False,
                                            "method": graph["method"]}
    (shaped / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (shaped / "recipe.json").write_text(json.dumps({"steps": recipe_steps or []}), encoding="utf-8")
    pipeline = tmp / "pipeline.json"
    pipeline.write_text(json.dumps({"system": {"package": f"{raw['generator']}_pipeline"}}), encoding="utf-8")
    return manifest, {}, pipeline


def _compose(tmp: Path, lane, **kw) -> dict:
    manifest, fitted, pipeline = shaped_dir(tmp, lane, **kw)
    return spec.compose(lane, manifest, fitted, "sha256:" + "0" * 64, pipeline)


def _variant(lane_id: str, **changes) -> dict:
    """A lane's declaration with some fields changed (nested dicts merged one level deep)."""
    raw = copy.deepcopy(dict(LANES[lane_id].raw))
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return raw


# --------------------------------------------------------------------------- every lane
def _check_window(raw: dict, sample: dict, keys: list) -> None:
    missing = [k for k in WINDOW_P1 if k not in sample]
    assert not missing, f"window sample lacks {missing}"
    w = raw["window"]
    assert (sample["length"], sample["stride"]) == (w["length"], w["stride"])
    assert ("horizon" in sample) == (raw["task"] == "forecast")
    if raw["task"] == "forecast":
        assert sample["horizon"] == w["horizon"]
    assert (sample["series_by"], sample["order_by"], sample["season"]) == (w["series_by"], w["order_by"], w["season"])
    assert sample["autoregressive"] is (raw["task"] == "forecast" and w["autoregressive"] is not False)
    assert keys == [k for k in (w["series_by"], w["order_by"]) if k]


def _check_graph(raw: dict, sample: dict, keys: list, edges: str) -> None:
    missing = [k for k in GRAPH_P1 if k not in sample]
    assert not missing, f"graph sample lacks {missing}"
    g = raw["graph"]
    assert sample["edges"] == edges
    assert (sample["level"], sample["protocol"], sample["batching"]) == (g["level"], g["protocol"], g["batching"])
    assert (sample["neighbours"] is not None) == (g["batching"] == "sampled")
    if g["batching"] == "sampled":
        assert sample["neighbours"] == g["neighbours"]
    assert (sample["exposure"] is not None) == (g["protocol"] == "transductive")
    assert keys == ([g["graph_id"]] if g.get("graph_id") else [])


@pytest.mark.parametrize("lane", COMPOSED, ids=[lane.id for lane in COMPOSED])
def test_every_lane_carries_the_p1_fields_of_its_shape(tmp_path, lane):
    doc = _compose(tmp_path, lane)
    dc, sample = doc["data_contract"], doc["data_contract"]["sample"]
    raw = lane.raw
    assert sample["shape"] == raw["shape"]
    if raw["shape"] == "row":
        assert sample == {"shape": "row"} and dc["keys"] == []
    if raw["shape"] == "window":
        _check_window(raw, sample, dc["keys"])
    if raw["shape"] == "graph":
        _check_graph(raw, sample, dc["keys"], dc_manifest_edges(tmp_path))
    assert (doc["training"]["batch_size"] == "full") == (sample.get("batching") == "full")
    features = [f["name"] for f in dc["features"]]
    assert not set(features) & set(dc["keys"]), "a key column is never a feature"
    assert dc["input_dim"] == len(features)


def dc_manifest_edges(tmp: Path) -> str:
    manifest = json.loads((tmp / "shaped" / "manifest.json").read_text(encoding="utf-8"))
    return manifest["outputs"]["files"]["edges"]["path"]


@pytest.mark.parametrize("lane", COMPOSED, ids=[lane.id for lane in COMPOSED])
def test_every_choice_is_sourced_and_every_user_choice_is_a_stance(tmp_path, lane):
    doc = _compose(tmp_path, lane)
    nodes = [doc["family"], doc["loss"], doc["metrics"], doc["optimizer"], doc["schedule"], doc["init"],
             doc["early_stopping"], doc["training"], *doc["regularization"], *doc["blocks"], doc["task"]]
    for node in nodes:
        assert node.get("source"), f"unsourced choice {node}"
        if node["source"] == "vega:default" and node is not doc["task"]:
            assert node.get("catalog"), f"a vega:default choice names its catalog row: {node}"
    memory = memory_for(lane, doc)
    for key in ("loss", "optimizer", "schedule"):
        if doc[key]["source"] == f"user:stance/{key}":
            assert f"{key}: {doc[key]['name']} (user)" in memory
    assert f"family: {doc['family']['name']} (user)" in memory


# --------------------------------------------------------------------------- windows
def test_window_forecast_defaults_are_sourced_to_the_catalog(tmp_path):
    raw = _variant("L10", window={"autoregressive": None, "context": None})
    sample = _compose(tmp_path, raw)["data_contract"]["sample"]
    assert sample["autoregressive"] is True
    assert (sample["autoregressive_source"], sample["autoregressive_catalog"]) == ("vega:default", "defaults/autoregressive")
    assert sample["context"] == "previous-part"
    assert (sample["context_source"], sample["context_catalog"]) == ("vega:default", "defaults/window_context")


def test_window_forecast_without_the_target_channel_is_the_lanes_stance(tmp_path):
    sample = _compose(tmp_path, LANES["L11"])["data_contract"]["sample"]
    assert sample["autoregressive"] is False
    assert sample["autoregressive_source"] == LANES["L11"].window["autoregressive_source"]
    assert "autoregressive_catalog" not in sample


def test_window_series_source_cites_the_lag_step_that_names_the_series(tmp_path):
    steps = [{"id": "S05", "op": "split", "params": {"kind": "time", "time_column": "ts"}},
             {"id": "S06", "op": "lag", "params": {"lags": [1, 2], "group_by": "station", "time_column": "ts"}}]
    sample = _compose(tmp_path, LANES["L10"], recipe_steps=steps)["data_contract"]["sample"]
    assert sample["series_source"] == "shape:S06"


def test_window_series_the_shaping_record_never_names_is_a_stance(tmp_path):
    assert _compose(tmp_path, LANES["L11"])["data_contract"]["sample"]["series_source"] == "user:stance/series"
    single = _compose(tmp_path / "one", LANES["L14"])
    assert single["data_contract"]["sample"]["series_source"] == "user:stance/one-series"
    assert "series: none — the rows are one single series (user)" in memory_for(LANES["L14"], single)


def test_window_sequence_task_has_no_horizon_and_no_target_channel(tmp_path):
    sample = _compose(tmp_path, LANES["L12"])["data_contract"]["sample"]
    assert "horizon" not in sample
    assert sample["autoregressive"] is False and sample["autoregressive_source"] == "vega:default"


def test_window_autoregressive_on_a_sequence_task_is_refused(tmp_path):
    raw = _variant("L12", window={"autoregressive": True})
    with pytest.raises(spec.SpecError, match="autoregressive is a forecast channel"):
        _compose(tmp_path, raw)


def test_window_keys_leave_the_features_and_the_input_dim(tmp_path):
    doc = _compose(tmp_path, LANES["L10"])
    dc = doc["data_contract"]
    schema = json.loads((tmp_path / "shaped" / "manifest.json").read_text(encoding="utf-8"))["output_schema"]
    assert dc["keys"] == ["station", "ts"]
    assert dc["input_dim"] == len(schema) - 1 - len(dc["keys"])


def test_window_batch_counts_windows_not_rows(tmp_path):
    raw = _variant("L14")                      # one series, length 16
    doc = _compose(tmp_path, raw, train_rows=400)
    windows = 400 - (16 - 1)
    assert doc["training"]["batch_size"] == spec.batch_rule(windows)
    assert "windows" in doc["training"]["rationale"]


# --------------------------------------------------------------------------- graphs
def test_graph_full_batching_is_the_full_batch_and_nothing_is_sampled(tmp_path):
    doc = _compose(tmp_path, LANES["L19"])
    sample = doc["data_contract"]["sample"]
    assert sample["batching"] == "full" and doc["training"]["batch_size"] == "full"
    assert (sample["batching_source"], sample["batching_catalog"]) == ("vega:default", "defaults/graph_batching")
    assert sample["neighbours"] is None and sample["exposure"] is None
    assert (sample["protocol_source"], sample["protocol_catalog"]) == ("vega:default", "defaults/graph_protocol")


def test_graph_directed_and_self_loops_are_copied_from_the_manifest(tmp_path):
    block = {"level": "node", "graph_id": None, "directed": True, "self_loops": True, "method": "key_pair"}
    sample = _compose(tmp_path, LANES["L19"], graph_block=block)["data_contract"]["sample"]
    assert (sample["directed"], sample["self_loops"]) == (True, True)
    assert sample["source"] == "shape:manifest/graph"


def test_graph_sampled_batching_carries_one_fanout_per_conv_and_counts_seed_nodes(tmp_path):
    doc = _compose(tmp_path, LANES["L21"])
    sample = doc["data_contract"]["sample"]
    convs = [b for b in doc["blocks"] if b["type"] in spec.CONV_TYPES]
    assert sample["neighbours"] == [6, 4] and len(sample["neighbours"]) == len(convs)
    assert sample["batching_source"].startswith("user:")
    assert doc["training"]["batch_size"] == spec.batch_rule(TRAIN_ROWS)
    assert "seed nodes" in doc["training"]["rationale"]
    assert "sample batching: sampled (user)" in memory_for(LANES["L21"], doc)


def test_graph_transductive_protocol_discloses_its_exposure_with_a_user_source(tmp_path):
    doc = _compose(tmp_path, LANES["L23"])
    sample = doc["data_contract"]["sample"]
    source = LANES["L23"].graph["protocol_source"]
    assert sample["protocol"] == "transductive" and sample["protocol_source"] == source
    assert sample["exposure"] == {"kind": "features", "rows": "test", "source": source}
    assert "graph protocol: transductive (user)" in memory_for(LANES["L23"], doc)


def test_graph_level_batches_graphs_and_keeps_the_graph_id_a_key(tmp_path):
    doc = _compose(tmp_path, LANES["L22"])
    sample, dc = doc["data_contract"]["sample"], doc["data_contract"]
    assert sample["level"] == "graph" and sample["batching"] == "graphs"
    assert sample["graph_id"] == "molecule_id" and dc["keys"] == ["molecule_id"]
    graphs = TRAIN_ROWS // 10                 # shaped_dir groups ten nodes per graph
    assert doc["training"]["batch_size"] == spec.batch_rule(graphs)


@pytest.mark.parametrize("batching", ["full", "sampled"])
def test_batch_size_is_full_iff_the_batching_is_full(tmp_path, batching):
    raw = _variant("L21", graph={"batching": batching, "neighbours": [6, 4] if batching == "sampled" else None})
    doc = _compose(tmp_path, raw)
    assert (doc["training"]["batch_size"] == "full") == (batching == "full")
    assert isinstance(doc["training"]["batch_size"], str if batching == "full" else int)


def test_a_graph_lane_on_a_manifest_without_a_graph_block_is_refused(tmp_path):
    manifest, fitted, pipeline = shaped_dir(tmp_path, LANES["L19"])
    manifest.pop("graph")
    with pytest.raises(spec.SpecError, match="no graph block"):
        spec.compose(LANES["L19"], manifest, fitted, "sha256:" + "0" * 64, pipeline)


# --------------------------------------------------------------------------- task, head, contract
def test_multilabel_task_carries_its_label_format_and_named_classes(tmp_path):
    task = _compose(tmp_path, LANES["L03"])["task"]
    assert task["label_format"] == {"kind": "delimited", "sep": "|"}
    assert task["classes"] == ["billing", "bug", "account", "shipping"]


def test_sequence_task_names_its_base_kind(tmp_path):
    task = _compose(tmp_path, LANES["L17"])["task"]
    assert (task["kind"], task["base_kind"]) == ("sequence-regression", "ordinal")


def test_reconstruction_head_is_recon_dim(tmp_path):
    doc = _compose(tmp_path, LANES["L06"])
    head = next(b for b in doc["blocks"] if b["type"] == "head")
    dc = doc["data_contract"]
    numeric = [f for f in dc["features"] if f["role"] == "numeric"]
    nominal = [f for f in dc["features"] if f["role"] == "nominal"]
    assert head["params"]["out"] == len(numeric) + sum(f["cardinality"] for f in nominal)


def test_unlabelled_anomaly_reads_the_score_mean(tmp_path):
    doc = _compose(tmp_path, LANES["L08"])
    assert doc["metrics"]["primary"] == "anomaly_score_mean"
    assert doc["metrics"]["selection"] == {"monitor": "valid.anomaly_score_mean", "mode": "min"}
    assert doc["task"]["source"] == "user:stance/goal" and "label" in doc["task"]
    assert "anomaly_rate" not in doc["task"]                      # A8: not a netspec-contract §1 field
    assert "goal: anomaly without a label, expected anomaly rate 0.03 (user)" in memory_for(LANES["L08"], doc)


def test_embedded_nominal_cardinality_is_the_unseen_id_plus_one(tmp_path):
    dc = _compose(tmp_path, LANES["L04"])["data_contract"]
    color = next(f for f in dc["features"] if f["name"] == "color")
    assert (color["dtype"], color["cardinality"], color["unseen_category_id"]) == ("int64", 4, 3)


# --------------------------------------------------------------------------- sources and budget
def test_the_batch_part_of_training_cites_the_catalog_rule_it_applies(tmp_path):
    # A8: the batch size carries the contract's source; no field outside netspec-contract §1
    training = _compose(tmp_path, LANES["L05"])["training"]
    assert training["source"] == "vega:default" and training["catalog"][0] == "defaults/batch_size"
    assert "clip(" in training["rationale"] and "user:stance/lane-budget" in training["rationale"]
    full = _compose(tmp_path / "full", LANES["L19"])
    sample = full["data_contract"]["sample"]
    assert full["training"]["batch_size"] == "full" and full["training"]["source"] == sample["batching_source"]
    if sample["batching_source"] == "vega:default":
        assert full["training"]["catalog"][0] == "defaults/graph_batching"


def test_the_budget_is_the_lanes_file_the_lane_came_from(tmp_path):
    doc = registry.load_document()
    doc["defaults"] = {**doc["defaults"], "max_epochs": 5, "patience": 2}
    lanes_file = tmp_path / "lanes.json"
    lanes_file.write_text(json.dumps(doc), encoding="utf-8")
    lane = registry.load_lanes(lanes_file)["L05"]
    manifest, fitted, pipeline = shaped_dir(tmp_path / "a", lane)
    composed = spec.Composer.for_lanes_file(lanes_file).compose(lane, manifest, fitted, "sha256:" + "0" * 64, pipeline)
    assert (composed["training"]["max_epochs"], composed["early_stopping"]["patience"]) == (5, 2)
    default = _compose(tmp_path / "b", lane)
    assert (default["training"]["max_epochs"], default["early_stopping"]["patience"]) == (
        LANES_DEFAULTS["max_epochs"], LANES_DEFAULTS["patience"])
