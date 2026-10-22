"""
test_lane_checks.py — the live lane checks' own proof (BUILD-PLAN T08): `oracles.py` re-exports
the one oracle module and adapts its shape rule; the recompute engine (EI2–EI4), the
explanation checks (EI5–EI7, EI9), the network checks (NI2, NI5, NI7), the receptive-field
checks (EI11) and the serving invariants (SA4–SA11) pass on what the backend really emits and
fail on a seeded defect — above all, a perturbed weight in a copied payload fails EI2 at its
node and only there.

The payloads are the inspection zoo's (vega-forge/scripts/tests/test_inspect_zoo.py, I10): a
real rendered package per case, its `NetworkInspector`'s `/v1/network` bytes and its
`detail: "exact"` explanation — the very bodies the live routes answer — and the case's
assembled batch for the graph context. The live runs (L01, L10, L19, L22) are the lanes'
`serve_check` and `inspect` stages.

    python3 -m pytest -q test_lane_checks.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
FORGE_TESTS = HERE.parents[2] / "vega-forge" / "scripts" / "tests"
for _p in (str(HERE), str(FORGE_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import block_oracles  # noqa: E402
import inspect_explain as X  # noqa: E402
import inspect_network as N  # noqa: E402
import inspect_receptive as RF  # noqa: E402
import oracles as O  # noqa: E402
import serving_check as SC  # noqa: E402
from check_kit import BundleRecords, Report  # noqa: E402
from inspect_context import StoredWeights, keys_pair  # noqa: E402
from inspect_recompute import (ExactPass, GraphContext, NetworkView, Recompute,  # noqa: E402
                               perturbed)
from lanekit import contracts as C  # noqa: E402
from lanekit.registry import load_lanes  # noqa: E402

# one case per family and graph level: every block type the oracle module covers is met
CASES = ("mlp-plain/row/multiclass/nominals", "mlp-residual/row/ordinal/nominals",
         "extra/mlp-residual/row/regression:layernorm", "linear/row/multilabel/nominals",
         "ft-transformer/row/binary/nominals", "autoencoder/row/anomaly:labelled/nominals",
         "vae/row/unsupervised/nominals", "deep-svdd/row/anomaly:labelled/nominals",
         "pca-floor/row/unsupervised/nominals", "gru/window/forecast/stride-1",
         "bi-lstm/window/sequence-classification:multilabel/stride-1",
         "rnn/window/sequence-regression:regression/stride-1",
         "cnn1d/window/sequence-classification:binary/stride-1",
         "lstm-attention/window/sequence-regression:ordinal/stride-1",
         "transformer-seq/window/forecast/stride-1",
         "extra/gru/window/forecast:attention-pool", "extra/mlp-plain/row/regression:quantile",
         "gcn/graph-node/binary/inductive-full", "gat/graph-node/multiclass/inductive-full",
         "graphsage/graph-node/regression/inductive-full",
         "extra/graphsage/graph-node/regression:normalize", "gin/graph-graph/binary/graphs",
         "gat/graph-graph/regression/graphs", "linear/graph-graph/multiclass/graphs")


@pytest.fixture(scope="module")
def zoo():
    import torch
    torch.set_num_threads(2)
    before = list(sys.path)
    import test_inspect_zoo as Z
    # the zoo puts vega-forge's test dirs first on sys.path, and its matrix fixtures load
    # vega-forge's own `run_matrix`: in a shared pytest process this program's modules of the
    # same name must still win, so the zoo's dirs go last and its `run_matrix` is forgotten
    sys.path[:] = before + [p for p in sys.path if p not in before]
    loaded = sys.modules.get("run_matrix")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).parent != HERE:
        del sys.modules["run_matrix"]
    return Z


class Payloads:
    """One zoo case as the live check sees it."""

    def __init__(self, Z, case_id: str):
        self.run = Z.run(case_id)
        self.network = self.run.network
        self.body = self.run.body("exact")
        _, self.batch, self.k = self.run.plain(self.run.query(detail="exact"))
        self.view = NetworkView(self.network, self.run.inspector.tensor)

    def context(self):
        b = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
             for k, v in self.batch.items()}
        if "edge_index" not in b:
            return None
        weight = b.get("edge_weight")
        common = dict(x_num=b.get("x_num"), x_cat=b.get("x_cat"), edge_index=b["edge_index"],
                      edge_weight=None if weight is None else weight.astype(np.float64))
        if self.run.spec["data_contract"]["sample"]["level"] == "node":
            targets = b.get("target_index")
            return GraphContext(level="node", target=int(targets[self.k]) if targets is not None
                                else self.k, **common)
        nodes = tuple(int(i) for i in np.flatnonzero(b["graph_index"] == self.k))
        return GraphContext(level="graph", graph_nodes=nodes, **common)

    def recompute(self, view=None):
        return Recompute(view or self.view, ExactPass.of(self.body), self.context(),
                         self.body["outputs"]["raw"]).run()


# --------------------------------------------------------------------------- oracles.py
def test_oracles_reexports_the_one_oracle_module():
    assert O.ORACLES is block_oracles.ORACLES
    assert O.shape_rule is not block_oracles.shape_rule          # the per-sample adapter
    assert len(O.BLOCK_TYPES) == 33 and O.SHAPE_ONLY == {"custom"}


def test_shape_rule_reads_per_sample_shapes_and_the_lane():
    lanes = load_lanes()
    l01, l10 = lanes["L01"], lanes["L10"]
    assert O.shape_rule("embedding", {"columns": ["zip", "region"],
                                      "dims": {"zip": 12, "region": 4}}, (2,), l01) == (16,)
    assert O.shape_rule("concat", {}, (7,), l01) == (23,)               # 7 numeric + 16
    assert O.shape_rule("concat", {}, (24, 9), l10) == (24, 12)         # per step
    assert O.shape_rule("gru", {"hidden": 32, "return": "last"}, (24, 12), l10) == (32,)
    assert O.shape_rule("graph_pool", {"readout": "sum"}, (32,), None) == (32,)
    residual = {"repeat": 2, "inner": [{"type": "dense", "params": {"units": 8}},
                                       {"type": "activation", "params": {"name": "relu"}}]}
    assert O.shape_rule("residual", residual, (5,), None) == (8,)       # netspec `inner`
    assert O.shape_rule("decoder", {"widths": [16]}, (4,), {"recon_dim": 10}) == (10,)
    with pytest.raises(KeyError, match="recon_dim"):
        O.shape_rule("decoder", {"widths": [16]}, (4,), None)


# --------------------------------------------------------------------------- EI2–EI9 on the zoo
@pytest.mark.parametrize("case_id", CASES)
def test_the_checks_hold_on_what_the_backend_emits(zoo, case_id):
    p = Payloads(zoo, case_id)
    result = p.recompute()
    for which in ("ei2", "ei3", "ei4"):
        assert not result.failures(which), (which, result.failures(which)[:3])
    assert result.ei2 and result.ei3 and result.ei4
    exact = ExactPass.of(p.body)
    positive = p.run.manifest["output"].get("positive")
    for name, observations in (("EI5", X.ei5(p.view, exact, p.body)),
                               ("EI6", X.ei6(p.network, p.body)),
                               ("EI7", X.ei7(p.network, p.body)),
                               ("EI9", X.ei9(p.body, positive, p.network))):
        assert observations and all(ok for ok, _ in observations), (name, observations)


def test_the_cases_meet_every_block_type_with_an_oracle(zoo):
    met = set()
    for case_id in CASES:
        p = Payloads(zoo, case_id)
        met |= {c.type for c in p.recompute().ei2}
    missing = set(O.ORACLES) - met - {"sequential", "parallel", "positional",
                                      "transformer_encoder", "edge_dropout", "flatten",
                                      "reshape", "pool1d"}
    assert not missing, missing


@pytest.mark.parametrize("case_id", ("mlp-residual/row/ordinal/nominals",
                                     "gru/window/forecast/stride-1",
                                     "gcn/graph-node/binary/inductive-full",
                                     "gin/graph-graph/binary/graphs"))
def test_a_perturbed_weight_in_a_copied_payload_fails_ei2_at_its_node(zoo, case_id):
    p = Payloads(zoo, case_id)
    assert not p.recompute().failures("ei2")
    exact = ExactPass.of(p.body)
    for node in ("head", "first"):
        node_id = X.head_of(p.view) if node == "head" else next(
            c.node for c in p.recompute().ei2 if c.type in ("dense", "gcn_conv", "gin_conv",
                                                            "gru"))
        weights = p.view.tensors_of(node_id)
        name = "weight" if "weight" in weights else sorted(weights)[0]
        tensor_id = f"{node_id}.{name}"
        x = exact.get(node_id, "in")
        column = 0 if x is None else int(np.argmax(np.abs(x).reshape(-1)[
            -p.view.tensor(tensor_id).shape[-1]:]))
        index = (0, column) if p.view.tensor(tensor_id).ndim == 2 else \
            (0,) * p.view.tensor(tensor_id).ndim
        copied = perturbed(p.network, p.view, tensor_id, index, 0.75)
        failed = {c.node for c in
                  p.recompute(NetworkView(copied, p.run.inspector.tensor)).failures("ei2")}
        # at graph node level a later conv reads its neighbours from the float64 chain, which
        # runs the copied weights: it fails too, and nothing that is not a graph conv does
        later = {i for i in failed - {node_id} if p.view.node(i).get("call") == "graph"}
        level = (p.context() or GraphContext(None, None, None, None, "")).level
        assert node_id in failed and failed - {node_id} == (later if level == "node" else set()), \
            (tensor_id, failed)
    assert p.network == p.run.network                                 # the copy only moved


# --------------------------------------------------------------------------- NI2, NI5, NI7
def test_network_checks_pass_and_catch_a_broken_drawing(zoo):
    p = Payloads(zoo, "mlp-residual/row/binary/nominals")
    assert all(ok for ok, _ in N.ni2(p.view, p.run.spec))
    broken = copy.deepcopy(p.network)
    broken["edges"].append({"from": "output", "to": broken["nodes"][2]["id"], "kind": "data"})
    assert not all(ok for ok, _ in N.ni2(NetworkView(broken), p.run.spec))
    reordered = dict(p.run.spec, blocks=list(reversed(p.run.spec["blocks"])))
    assert not N.ni2(p.view, reordered)[0][0]
    body = p.run.network_bytes
    assert all(ok for ok, _ in N.ni7(body, body, json.loads(body)["network_digest"],
                                      {"pinned": True}))
    moved = json.dumps(dict(json.loads(body), family={"name": "x"})).encode()
    assert not all(ok for ok, _ in N.ni7(moved, moved, None, None))


def test_weights_match_pairs_the_payload_with_the_state_dict(zoo):
    p = Payloads(zoo, "gru/window/forecast/stride-1")
    state = {k: v.detach().cpu().numpy() for k, v in p.run.model.state_dict().items()
             if v.is_floating_point()}
    published = {t: p.view.tensor(t) for t in p.view.tensor_ids()}
    roles = {t: p.network["tensors"][t]["role"] for t in published}
    match = StoredWeights(state).match(published, roles)
    assert all(ok for ok, _ in N.ni5(p.view, match)), match
    some = next(t for t in published if p.network["tensors"][t]["role"] == "weight")
    moved = dict(published, **{some: published[some] + 1.0})
    assert not N.ni5(p.view, StoredWeights(state).match(moved, roles))[-1][0]


# --------------------------------------------------------------------------- seeded defects
def test_explanation_checks_catch_seeded_defects(zoo):
    p = Payloads(zoo, "mlp-plain/row/multilabel/nominals")
    exact = ExactPass.of(p.body)
    head = X.head_of(p.view)
    bad = ExactPass({**exact.nodes, head: {**exact.nodes[head],
                                           "grad_out": exact.get(head, "grad_out") + 0.5}},
                    exact.levels)
    assert not X.ei5(p.view, bad, p.body)[0][0]
    renormalised = copy.deepcopy(p.body)
    read = np.asarray(renormalised["outputs"]["read"])
    renormalised["outputs"]["read"] = list(read / read.sum())
    assert not all(ok for ok, _ in X.ei9(renormalised))
    feats = renormalised["attribution"]["features"]
    feats["values"][0], feats["attribution"][0] = 0.0, 0.25
    assert not all(ok for ok, _ in X.ei7(p.network, renormalised))


def test_window_rows_follow_the_series_layout():
    rows = [{"s": "a" if i % 2 else "b", "t": i} for i in range(10)]
    history = [{"s": "a", "t": -3}, {"s": "b", "t": -2}, {"s": "a", "t": -1}]
    req = {"rows": rows, "history": history}
    assert RF.window_rows(req, 9, {"series_by": "s", "order_by": "t"}, 4) == [3, 5, 7, 9]
    assert RF.window_rows(req, 3, {"series_by": "s", "order_by": "t"}, 4) == [-3, -1, 1, 3]
    told = {"sample": {"row": 3, "span": [-3, 3]}, "inputs": {"raw": {"s": "a"},
                                                              "shaped": [[0]] * 4}}
    assert all(ok for ok, _ in RF.ei11_window(told, req, {"series_by": "s", "order_by": "t"}, 4))
    told["sample"]["span"] = [-1, 3]
    assert not RF.ei11_window(told, req, {"series_by": "s", "order_by": "t"}, 4)[0][0]


class _Reference:
    size, meta = 3, {"method": "key_pair", "directed": False}

    def has_edge(self, s, d):
        return {s, d} == {0, 1}

    def key_columns(self):
        return ("key", "to")

    def raw(self, ref, column):
        return {"key": f"K{ref}", "to": f"K{ref + 1}"}[column]


def test_receptive_field_of_a_node_query():
    request = {"rows": [{"key": "Q", "to": "K1"}]}
    good = {"sample": {"row": 0, "graph": {"target_node": 3}},
            "attribution": {"neighbours": [{"node": 3, "hop": 0}, {"node": 1, "hop": 1},
                                           {"node": 0, "hop": 2}],
                            "edges": [{"src": 1, "dst": 3}, {"src": 0, "dst": 1}]}}
    assert all(ok for ok, _ in RF.ei11_node(good, request, _Reference()))
    sent_by_query = copy.deepcopy(good)
    sent_by_query["attribution"]["edges"].append({"src": 3, "dst": 2})
    assert not all(ok for ok, _ in RF.ei11_node(sent_by_query, request, _Reference()))
    unpaired = copy.deepcopy(good)
    unpaired["attribution"]["edges"][0] = {"src": 2, "dst": 3}
    assert not all(ok for ok, _ in RF.ei11_node(unpaired, request, _Reference()))
    assert keys_pair({"key": "A", "to": "B"}, {"key": "B", "to": None}, ("key", "to"))


# --------------------------------------------------------------------------- SA4–SA11
def _check(output: dict, sample=None) -> SC.ServingCheck:
    bundle = BundleRecords(Path("."), {"output": output, "sample": sample or {"shape": "row"}},
                           {}, {}, {})
    return SC.ServingCheck(bundle, None, Report(C.SERVING_CHECK, "t", "u", "b",
                                                C.SERVING_CHECK_IDS))


def _holds(observations) -> bool:
    return bool(observations) and all(ok for ok, _ in observations)


def test_serving_invariants_hold_and_catch_seeded_defects():
    binary = _check({"positive": "1", "threshold": 0.4, "classes": ["0", "1"]})
    good = {"label": "1", "probabilities": {"0": 0.45, "1": 0.55},
            "logits": {"1": float(np.log(0.55 / 0.45))}}
    assert _holds(binary._sa4(good))
    assert not _holds(binary._sa4({**good, "label": "0"}))
    assert not _holds(binary._sa4({**good, "logits": {"1": 3.0}}))
    multi = _check({"classes": ["a", "b", "c"]})
    assert _holds(multi._sa5({"label": "b", "probabilities": {"a": .2, "b": .5, "c": .3}}))
    assert not _holds(multi._sa5({"label": "a", "probabilities": {"a": .2, "b": .5, "c": .3}}))
    assert not _holds(multi._sa5({"label": "b", "probabilities": {"a": .3, "b": .5, "c": .3}}))
    labels = _check({"classes": ["x", "y"], "threshold": 0.5, "threshold_per_label": {"y": 0.9}})
    assert _holds(labels._sa6({"labels": ["x"], "probabilities": {"x": .7, "y": .8}}))
    assert not _holds(labels._sa6({"labels": ["x", "y"], "probabilities": {"x": .7, "y": .8}}))
    ordinal = _check({"classes": ["1", "2", "3"], "fields": ["cum_1", "cum_2"],
                      "field_kinds": {"cum_1": {"kind": "cumulative", "of": "1"},
                                      "cum_2": {"kind": "cumulative", "of": "2"}}})
    fine = {"label": "2", "probabilities": {"1": .3, "2": .4, "3": .3},
            "cumulative": {"1": .7, "2": .3}}
    assert _holds(ordinal._sa8(fine))
    assert not _holds(ordinal._sa8({**fine, "label": "3"}))
    assert not _holds(ordinal._sa8({**fine, "cumulative": {"1": .3, "2": .7}}))
    forecast = _check({"horizons": [1, 2]})
    assert _holds(forecast._sa9({"values": {"h1": 1.0, "h2": 2.0}}))
    assert not _holds(forecast._sa9({"values": {"h1": 1.0}}))
    anomaly = _check({"threshold": 2.0})
    assert _holds(anomaly._sa11({"score": 3.0, "anomalous": True}))
    assert not _holds(anomaly._sa11({"score": 1.0, "anomalous": True}))
    assert not _holds(_check({"threshold": None})._sa11({"score": 1.0, "anomalous": False}))
    assert not _holds(_check({})._sa10({"reconstruction_error": -1.0}))
    assert not _holds(_check({})._sa7({"value": None}))


def test_sa14_holds_a_malformed_request_to_no_500_and_no_confident_answer():
    output = {"field_kinds": {"p_0": {"kind": "probability", "of": "0"},
                              "p_1": {"kind": "probability", "of": "1"},
                              "label": {"kind": "class", "of": None}}}
    answered = {"predictions": [{"answered": True, "label": "1", "probabilities": {"0": .4, "1": .6}}]}
    assert SC.sa14_empty_cell(200, answered, "load", output)[0]
    # gate C (L19): answered true, no warning, probability null — the decision read from NaN
    confident = {"predictions": [{"answered": True, "warnings": [], "label": "0",
                                  "probabilities": {"0": None, "1": None}}]}
    assert not SC.sa14_empty_cell(200, confident, "load", output)[0]
    unanswered = {"predictions": [{"answered": False, "warnings": ["not answered: … (load empty "
                                                                   "or non-finite) …"]}]}
    assert SC.sa14_empty_cell(200, unanswered, "load", output)[0]
    assert not SC.sa14_empty_cell(200, {"predictions": [{"answered": False, "warnings": []}]},
                                  "load", output)[0]
    assert SC.sa14_empty_cell(422, {"errors": [{"column": "load"}]}, "load", output)[0]
    assert not SC.sa14_empty_cell(500, {}, "load", output)[0]
    # gate C (L10): a null series key or an unreadable time answered 500
    assert SC.sa14_key(422, {"errors": [{"column": "station"}]}, "station", True)[0]
    assert not SC.sa14_key(500, {}, "station", True)[0]
    assert not SC.sa14_key(200, {}, "station", True)[0], "a null key is never answered"
    assert SC.sa14_key(200, {}, "ts", False)[0]
    assert not SC.sa14_key(500, {}, "ts", False)[0]
    assert not SC.sa14_key(422, {"errors": [{"column": None}]}, "ts", False)[0]


def test_json_values_follow_the_field_kinds():
    answer = {"probabilities": {"a": 0.1}, "values": {"h1": {"0.5": 2.0}, "h2": 3.0},
              "labels": ["x"], "embedding": [0.5, 0.25], "label": "a"}
    assert SC.json_value(answer, "probability", "a") == 0.1
    assert SC.json_value(answer, "horizon", "1/0.5") == 2.0
    assert SC.json_value(answer, "horizon", 2) == 3.0
    assert SC.json_value(answer, "flag", "x") == 1 and SC.json_value(answer, "flag", "y") == 0
    assert SC.json_value(answer, "embedding", 1) == 0.25
    assert SC.json_value(answer, "class", None) == "a"
    assert SC.json_value(answer, "decision", None) is None
    assert SC.same_cell("1", 1.0) and SC.same_cell("True", True) and SC.same_cell("", None)
    assert not SC.same_cell("0.5", 0.50001)


def test_the_packed_smoke_of_a_selected_rung_is_the_records_own():
    """A44(b): pack writes a rung's or trial's own smoke into records/architecture.json with
    `smoke_of`; the checks read it there, not the forged spec's, whatever the blocks."""
    forged, run = {"params_total": 10, "per_block": []}, {"params_total": 99, "per_block": []}
    def bundle(arch, run_id="H1-3"):
        return BundleRecords(Path("/nonexistent/lane/bundle/x"), {"run": run_id}, {},
                             {"blocks": [{"id": "B1", "params": {"units": 64}}]}, arch)
    shipped = {"netspec": {"blocks": [{"id": "B1", "params": {"units": 8}}]}, "smoke": run,
               "smoke_of": {"run": "H1-3"}}
    assert bundle(shipped).packed_smoke() == (run, C.ARCHITECTURE_RECORD)
    other = {**shipped, "smoke": forged, "smoke_of": {"run": "R9"}}
    assert bundle(other).packed_smoke()[0] is None, "a smoke of another run is not this network's"
    same = {"netspec": {"blocks": [{"id": "B1", "params": {"units": 64}}]}, "smoke": forged}
    assert bundle(same, "R0").packed_smoke() == (forged, C.ARCHITECTURE_RECORD)
