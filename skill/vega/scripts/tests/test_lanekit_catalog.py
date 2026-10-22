"""
The composer's readings of the forge catalog that need no lane (BUILD-PLAN T02): each family's
default composition within its `block_types` and holding the types its `composition` requires,
a minimal spec for every block, the batch rule, and the W1 gate's deferral of what the backend
still plans (compose_check.py); and the other side of the catalog, the fields netspec-contract
§1 admits: no composed spec of any lane carries another (amendment A8).

    python3 -m pytest -q vega/scripts/tests/test_lanekit_catalog.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import registry, spec  # noqa: E402
from test_lanekit_spec import COMPOSED, _compose  # noqa: E402

LANES = registry.load_lanes()


# --------------------------------------------------------------------------- family defaults
@pytest.mark.parametrize("family", sorted(spec.CATALOG["families"]))
def test_family_default_uses_only_its_catalog_types_and_every_required_one(family):
    shape = spec.CATALOG["families"][family]["sample_shape"]
    contract = spec.Contract.synthetic(shape, nominals={"c1": 5}, level="graph" if family == "gin" else None)
    target = spec.synthetic_target("binary" if shape == "row" else "sequence-classification/binary"
                                    if shape == "window" else "binary", family)
    if family in ("autoencoder", "vae", "pca-floor"):
        target = spec.synthetic_target("unsupervised", family)
    if family == "deep-svdd":
        target = spec.synthetic_target("anomaly", family)
    types = registry.flatten_types(spec.FAMILY_BLOCKS[family](target, contract))
    allowed = set(spec.CATALOG["families"][family]["block_types"])
    assert set(types) <= allowed, f"{family}: {sorted(set(types) - allowed)} outside block_types"
    assert spec.REQUIRED_TYPES[family] <= set(types), f"{family}: lacks {sorted(spec.REQUIRED_TYPES[family] - set(types))}"


def test_every_family_has_a_default_and_a_required_set():
    families = set(spec.CATALOG["families"])
    assert set(spec.FAMILY_BLOCKS) == families and set(spec.REQUIRED_TYPES) == families


@pytest.mark.parametrize("block_type", sorted(spec.CATALOG["blocks"]))
def test_every_catalog_block_has_a_minimal_spec_that_holds_it(block_type):
    doc = spec.minimal_block_spec(block_type)
    assert block_type in registry.flatten_types(doc["blocks"])
    sample = doc["data_contract"]["sample"]
    assert (doc["training"]["batch_size"] == "full") == (sample.get("batching") == "full")
    assert doc["blocks"][-1]["type"] == "head"


def test_family_spec_takes_a_sequence_task_with_its_base():
    doc = spec.family_spec("lstm-attention", "sequence-regression/ordinal", spec.Contract.synthetic("window"))
    assert (doc["task"]["kind"], doc["task"]["base_kind"]) == ("sequence-regression", "ordinal")
    head = doc["blocks"][-1]
    assert head["params"]["out"] == 3         # K - 1 over the synthetic four levels


def test_batch_rule_is_the_catalog_rule():
    assert spec.batch_rule(1680) == 256       # 1680 / 8 = 210 -> 256
    assert spec.batch_rule(40) == 16          # clipped at 16
    assert spec.batch_rule(100000) == 512     # clipped at 512
    assert spec.batch_rule(10) == 10          # never above the targets


def test_required_types_are_read_from_the_catalog_composition():
    assert spec.required_types("mlp-plain") == {"concat", "dense", "activation", "dropout", "head"}
    assert spec.required_types("mlp-residual") == {"concat", "dense", "residual", "activation", "dropout", "head"}
    assert spec.required_types("gru") == {"gru", "head"}                 # attention(pool) or pool(mean)
    assert spec.required_types("gat") == {"gat_conv", "head"}            # [..] x (depth - 1) runs zero times
    assert spec.required_types("gin") == {"gin_conv", "activation", "dropout", "graph_pool", "head"}
    assert spec.required_types("graphsage") == {"sage_conv", "activation", "dropout", "head"}
    assert spec.required_types("deep-svdd") == {"concat", "sequential", "dense", "head"}
    assert spec.required_types("linear") == {"concat", "head"}           # the row branch
    for family, row in spec.CATALOG["families"].items():
        assert spec.REQUIRED_TYPES[family] <= set(row["block_types"]), family


def test_compose_check_defers_only_what_the_backend_still_plans():
    import compose_check

    refusal = {"level": "ERROR", "rule": "B2", "where": "blocks[3]",
               "message": "'gat_conv' is planned in backend pytorch; the spec cannot be emitted"}
    lane = LANES["L20"]
    errors, deferred, _ = compose_check.classify([refusal], lane, {"gat_conv"})
    assert (errors, len(deferred)) == ([], 1)
    errors, deferred, _ = compose_check.classify([refusal], lane, set())   # after B14 empties `planned`
    assert (len(errors), deferred) == (1, [])
    other = {**refusal, "rule": "B1"}
    assert compose_check.classify([other], lane, {"gat_conv"})[0], "only B2 is ever deferred"


# --------------------------------------------------------------------------- the contract's fields
# netspec-contract §1's fields, section by section (A8: no other field in a composed spec)
CHOICE = {"source", "catalog", "rationale", "alternatives"}
CONTRACT_FIELDS = {
    "": {"schema", "input", "seed", "task", "data_contract", "family", "blocks", "loss", "metrics", "optimizer",
         "schedule", "regularization", "early_stopping", "init", "training", "backend", "decision_families",
         "outputs"},
    "input": {"model", "model_digest", "layers_read", "shaped_dir", "manifest", "manifest_digest", "pipeline",
              "pipeline_digest"},
    "task": {"kind", "base_kind", "label", "classes", "label_format", "positive", "source"},
    "data_contract": {"source", "parts", "features", "input_dim", "keys", "sample", "class_balance"},
    "feature": {"name", "dtype", "role", "cardinality", "unseen_category_id", "origin", "role_source"},
    "sample.row": {"shape"},
    "sample.window": {"shape", "length", "stride", "horizon", "series_by", "order_by", "context", "autoregressive",
                      "season", "source", "context_source", "context_catalog", "autoregressive_source",
                      "autoregressive_catalog", "series_source"},
    "sample.graph": {"shape", "edges", "level", "graph_id", "directed", "self_loops", "protocol", "protocol_source",
                     "protocol_catalog", "batching", "batching_source", "batching_catalog", "neighbours",
                     "exposure", "source"},
    "family": {"name"} | CHOICE, "block": {"id", "type", "params"} | CHOICE,
    "loss": {"name", "params", "formula"} | CHOICE, "metrics": {"primary", "secondary", "selection"} | CHOICE,
    "optimizer": {"name", "params", "update_rule"} | CHOICE, "schedule": {"name", "params"} | CHOICE,
    "regularization": {"kind", "value", "where"} | CHOICE,
    "early_stopping": {"monitor", "mode", "patience", "min_delta", "params"} | CHOICE,
    "init": {"scheme", "bias"} | CHOICE,
    "training": {"batch_size", "max_epochs", "device", "precision", "num_workers", "budget"} | CHOICE,
    "backend": {"name", "min_version", "hints"}, "outputs": {"run_dir", "package", "code", "notebook"}}


def _sections(doc: dict):
    yield "", doc
    for key in ("input", "task", "data_contract", "family", "loss", "metrics", "optimizer", "schedule",
                "early_stopping", "init", "training", "backend", "outputs"):
        yield key, doc[key]
    yield "sample." + doc["data_contract"]["sample"]["shape"], doc["data_contract"]["sample"]
    yield from (("feature", f) for f in doc["data_contract"]["features"])
    yield from (("block", b) for b in doc["blocks"])
    yield from (("regularization", r) for r in doc["regularization"])


@pytest.mark.parametrize("lane", COMPOSED, ids=lambda lane: lane.id)
def test_a_composed_spec_holds_only_the_contract_fields(tmp_path, lane):
    doc = _compose(tmp_path, lane)
    extra = {where or "top": sorted(set(node) - CONTRACT_FIELDS[where]) for where, node in _sections(doc)
             if set(node) - CONTRACT_FIELDS[where]}
    assert not extra, extra
