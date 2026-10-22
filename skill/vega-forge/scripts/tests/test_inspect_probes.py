#!/usr/bin/env python3
"""
test_inspect_probes.py — the capture, the block probes and their recompute checks
(`templates/serving/package/inspection/{capture, focus, parts, probes, probes_seq,
probes_graph, probes_group, derived}.py` and the readers' `probe`, `recompute`, `exact`),
proved on *rendered* packages (build plan I03; inspect-contract §1 R1–R7, §3.1, §5.4; R-B4,
R-B8, R-F5).

    python3 test_inspect_probes.py            (or: python3 -m pytest test_inspect_probes.py -q)

One package is rendered per case of `_inspect_fixtures.py` — the catalog's 21 families and
user-defined compositions, every one of the 33 block types and 5 synthetic ones among them —
and each is explained through a `Capture` pass under the permanent hooks the inspector
attaches once. On each:

  table      each reader's probe keys, exact tensors and check are catalog.json's
             `inspection.views[type].{probe, exact, recompute}`.
  payloads   every node payload validates `explanation-2.schema.json`'s `node_payload`, at
             `full`, `summary` and `exact`; `extra` carries exactly the type's probe keys; a
             node has a check exactly when §3.1 names one, and every check is within
             `1e-4·max(1, |out|)`.
  closing    dense and head: `W·in + b = pre` with the published weights (the ordinal head's
             shared score and thresholds); the chain arrays recompute `W·post[l−1] + b =
             pre[l]`; recurrent: the step-by-step unroll equals the module's sequence output
             within 1e-5 and the float64 oracle's output; attention rows sum to 1 (attention,
             encoder layers, attention pool, GAT per destination); `attention_pool` is R-B8's
             `α = softmax_t(q·tanh(W h_t + b))`, `y = Σ_t α_t h_t` from the published
             tensors; graph messages (and the root and bias) reproduce the conv output at the
             targets within 1e-4, and their relevances add up to the target's first-order
             share; a target no edge reaches (sage's root apart, gcn and gat without self
             loops) is explained at `full` and `exact` with the root alone or no message.
  capture    focus per sample shape (row, window, graph node level through `target_index`,
             graph level through `graph_index`); the shapes pass of the graph build is a
             `Capture(mode="shapes")`; refusals (training network, a second capture on the
             thread, a pass run or differentiated twice, a target the pass did not compute).
  hygiene    after any explanation no parameter holds `.grad`, the hooks are the ones
             attached once (none per request), the model's state and mode are unchanged and
             `Reparameterize.kl` was never written; a plain forward under the hooks is the
             forward without them; the same request gives the same bytes; the budget
             summarises, omits and degrades deterministically.
  modules    C11–C13 at `error`, no module-level mutable state, imports of numpy, torch,
             the standard library and the package only, no dataset literal.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import copy
import subprocess
import sys

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator

import _inspect_fixtures as fx
from _inspect_fixtures import CASES, CATALOG, EXPLANATION, INSPECTION, REPO, case, explain

import block_oracles as O  # noqa: E402
import conformance  # noqa: E402

VIEWS = CATALOG["inspection"]["views"]
MINE = ("capture.py", "focus.py", "parts.py", "probes.py", "probes_seq.py", "probes_graph.py",
        "probes_group.py", "derived.py", "readers.py", "readers_seq.py", "readers_graph.py",
        "readers_group.py", "wiring.py")
NODE_PAYLOAD = Draft202012Validator({"$ref": "#/$defs/node_payload",
                                     "$defs": EXPLANATION["$defs"]})
CHAIN = Draft202012Validator({**EXPLANATION["properties"]["chain"],
                              "$defs": EXPLANATION["$defs"]})


def _valid(validator, value, where):
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
    assert not errors, f"{where}: " + "; ".join(f"{list(e.path)}: {e.message[:200]}"
                                                  for e in errors[:4])


def _index(c):
    """A sample other than the first when the batch holds one (the focus must follow it)."""
    return 1 if c.samples() > 1 else 0


def _decode(c, payload):
    return None if payload is None else c.codec.decode(payload)


def _tensor(c, tensor_id):
    """A published tensor, decoded from the live parameters (whatever its body encoding)."""
    return c.codec.decode(c.graph.codec.fetch(tensor_id)).astype(np.float64)


def _scale(value) -> float:
    a = np.abs(np.asarray(value, dtype=np.float64))
    return max(1.0, float(a.max()) if a.size else 0.0)


# ================================================================ the view table

def test_the_probe_table_is_the_catalogs():
    reg = case("mlp-plain").registry
    assert set(reg.READERS) == set(VIEWS)
    for t, row in VIEWS.items():
        probe = reg.READERS[t].PROBE
        assert list(probe.KEYS) == row["probe"], t
        assert list(probe.EXACT) == row["exact"], t
        assert probe.CHECKS == (row["recompute"] is not None), t


def test_every_type_is_explained_by_the_cases():
    seen = {n.type for name in CASES for n in case(name).graph.nodes}
    assert seen == set(VIEWS), sorted(set(VIEWS) ^ seen)


# ================================================================ payloads

@pytest.mark.parametrize("name", list(CASES))
def test_node_payloads_validate_and_carry_the_type_keys(name):
    c = case(name)
    section, cap = explain(c, _index(c))
    assert list(section["nodes"]) == [n.id for n in c.graph.nodes]
    for node in c.graph.nodes:
        payload = section["nodes"][node.id]
        _valid(NODE_PAYLOAD, payload, node.id)
        assert payload["level"] == "full", node.id
        assert set(payload["extra"]) == set(VIEWS[node.type]["probe"]), node.id
        assert (payload["recompute"] is not None) == (VIEWS[node.type]["recompute"] is not None)
        if node.type in ("input", "output") or payload["out"] is None:
            continue
        assert payload["out"]["form"] in ("scalar", "vector", "sequence", "nodes"), node.id
    _valid(CHAIN, section["chain"], "chain")
    assert (section["chain"] is None) == (c.graph.chain() is None)


def _checks_hold(c, section, cap) -> tuple:
    """Every node check of a section within `1e-4·max(1, |out|)`; (worst, its node)."""
    worst, worst_node = -1.0, None
    for node in c.graph.nodes:
        value = section["nodes"][node.id]["recompute"]
        if value is None:
            continue
        out = cap.take(cap.value(node.id), node) if node.type not in ("edge_dropout",) else None
        bound = 1e-4 * (_scale(out.detach().numpy()) if torch.is_tensor(out) else 1.0)
        assert value <= bound, f"{node.id} ({node.type}): recompute {value} > {bound}"
        if value > worst:
            worst, worst_node = value, node.id
    return worst, worst_node


@pytest.mark.parametrize("name", list(CASES))
def test_every_check_holds_within_the_contract_tolerance(name):
    c = case(name)
    section, cap = explain(c, _index(c))
    worst, worst_node = _checks_hold(c, section, cap)
    report = section["recompute"]
    assert report["nodes"] == sum(1 for p in section["nodes"].values()
                                  if p["recompute"] is not None)
    assert report["worst_node"] == worst_node and report["worst"] == pytest.approx(worst)


@pytest.mark.parametrize("name", list(CASES))
def test_exact_nodes_carry_the_full_sample_tensors(name):
    c = case(name)
    section, cap = explain(c, _index(c), "exact")
    assert section["degraded"] is False and section["detail"] == "exact"
    for node in c.graph.nodes:
        payload = section["nodes"][node.id]
        _valid(NODE_PAYLOAD, payload, node.id)
        assert payload["level"] == "exact", node.id
        exact = payload["exact"]
        extra = {k for k, v in exact.items() if v is not None} - {"in", "out", "grad_out"}
        assert extra <= set(VIEWS[node.type]["exact"]), (node.id, extra)
        if VIEWS[node.type]["exact"]:
            assert extra == set(VIEWS[node.type]["exact"]), (node.id, extra)
        value = cap.value(node.id)
        if value is not None and node.type != "edge_dropout":
            got = _decode(c, exact["out"])
            want = cap.take(value, node).detach().numpy()
            assert got.shape == want.shape and np.array_equal(got, want.astype(got.dtype)), \
                node.id
            assert exact["out"]["axes"] and len(exact["out"]["axes"]) == got.ndim


def test_summary_detail_puts_every_node_at_summary():
    c = case("mlp-residual")
    section, _ = explain(c, 0, "summary")
    for node_id, payload in section["nodes"].items():
        _valid(NODE_PAYLOAD, payload, node_id)
        assert payload["level"] == "summary" and payload["extra"] == {}, node_id
        assert payload["out"] is None or payload["out"]["form"] == "summary"
    assert section["budget"]["summarised"] == [] and section["degraded"] is False


def test_the_budget_summarises_then_omits_and_says_so():
    c = case("gru")
    full, _ = explain(c, 0)
    limits = dict(c.graph.codec.limits, explain_float_budget=full["budget"]["used"] - 1)
    small, _ = explain(c, 0, limits=limits)
    again, _ = explain(c, 0, limits=limits)
    assert fx.canonical(small) == fx.canonical(again)
    levels = [p["level"] for p in small["nodes"].values()]
    assert levels == ["full"] * (len(levels) - 1) + ["summary"] and small["degraded"] is True
    assert small["budget"]["summarised"] == ["output"] and small["budget"]["used"] <= limits[
        "explain_float_budget"]
    assert small["nodes"]["output"]["why"].startswith("summarised")
    third = dict(limits, explain_float_budget=full["budget"]["used"] // 3)
    cut, _ = explain(c, 0, limits=third)
    assert cut["budget"]["omitted"] == ["R", "A"] and cut["degraded"] is True
    tiny = dict(limits, explain_float_budget=4)
    starved, _ = explain(c, 0, limits=tiny)
    assert starved["budget"]["omitted"]
    for node_id in starved["budget"]["omitted"]:
        payload = starved["nodes"][node_id]
        _valid(NODE_PAYLOAD, payload, node_id)
        assert payload["level"] == "omitted" and payload["why"].startswith("omitted")


def test_an_exact_budget_that_runs_out_summarises_the_rest():
    c = case("lstm")
    section, _ = explain(c, 0, "exact")
    limits = dict(c.graph.codec.limits,
                  explain_exact_float_budget=section["budget"]["used"] // 2)
    short, _ = explain(c, 0, "exact", limits=limits)
    levels = [p["level"] for p in short["nodes"].values()]
    first = levels.index("summary")
    assert set(levels[:first]) == {"exact"} and "exact" not in levels[first:]
    assert short["degraded"] is True


# ================================================================ closing: units and the chain

def _units(c):
    return [n for n in c.graph.nodes if n.type in ("dense", "head")]


@pytest.mark.parametrize("name", list(CASES))
def test_dense_and_head_nodes_recompute_pre_from_the_published_weights(name):
    c = case(name)
    section, cap = explain(c, _index(c), "exact")
    for node in _units(c):
        exact = section["nodes"][node.id]["exact"]
        x, z = _decode(c, exact["in"]).astype(np.float64), _decode(c, exact["pre"])
        w = _tensor(c, node.params["weight"])
        b = _tensor(c, node.params["bias"]) if node.params["bias"] else 0.0
        affine = x @ w.T + b
        assert np.max(np.abs(affine - z)) <= 1e-5 * _scale(z), node.id
        post, out = _decode(c, exact["post"]), _decode(c, exact["out"])
        assert np.array_equal(post, out), node.id
        if node.params.get("thresholds") is not None:     # the rank-consistent head
            theta = np.asarray(node.params["thresholds"])
            assert np.max(np.abs(z + theta - out)) <= 1e-5 * _scale(out), node.id
        grad_pre = _decode(c, exact["grad_pre"])
        assert grad_pre.shape == z.shape


@pytest.mark.parametrize("name", [n for n in CASES if case(n).graph.chain() is not None])
def test_the_chain_recomputes_layer_by_layer(name):
    c = case(name)
    section, cap = explain(c, _index(c))
    chain, net = section["chain"], c.graph.chain()
    assert len(chain["pre"]) == len(chain["post"]) == len(chain["grad_pre"]) == \
        len(net["layers"])
    assert len(chain["inputs"]) == len(net["inputs"])
    prev = np.asarray(chain["inputs"])
    for k, layer in enumerate(net["layers"]):
        w = _tensor(c, layer["weight"])
        b = _tensor(c, layer["bias"]) if layer["bias"] else 0.0
        pre = np.asarray(chain["pre"][k])
        assert np.max(np.abs(prev @ w.T + b - pre)) <= 1e-5 * _scale(pre), layer["node"]
        prev = np.asarray(chain["post"][k])
    head = section["head"]
    assert np.allclose(chain["post"][-1], head, atol=1e-6)
    assert chain["grad_pre"][-1] == [1.0] + [0.0] * (len(head) - 1)


# ================================================================ closing: recurrent

def _recurrent(c):
    return [n for n in c.graph.nodes if n.type in ("rnn", "gru", "lstm")]


@pytest.mark.parametrize("name", [n for n in CASES if _recurrent(case(n))])
def test_the_recurrent_unroll_is_the_modules_sequence_output(name):
    c = case(name)
    section, cap = explain(c, _index(c), "exact")
    for node in _recurrent(c):
        payload = section["nodes"][node.id]
        unroll = payload["extra"]["unroll"]
        assert unroll["max_abs_diff"] <= 1e-5, (node.id, unroll["max_abs_diff"])
        per = node.params["per_layer"]
        assert [(u["layer"], u["direction"]) for u in unroll["per_layer"]] == \
            [(p["layer"], p["direction"]) for p in per]
        gates = len(node.params["gates"])
        assert all(u["gates"]["cols"] == gates for u in unroll["per_layer"])
        assert all((u["cell_norm"] is None) == (node.type != "lstm")
                   for u in unroll["per_layer"])
        hidden = _decode(c, payload["exact"]["hidden"])
        dirs = node.params["directions"]
        top = np.concatenate(list(hidden[-dirs:]), axis=-1)
        seq = cap.take(cap.seq(node.id), node).detach().numpy()
        assert np.max(np.abs(top - seq)) <= 1e-5, node.id
        module = cap.module(node.id)
        weights, params = O.weights_of(module, node.type)
        x = _decode(c, payload["exact"]["in"])[None]
        oracle = O.ORACLES[node.type](weights, x, params)[0]
        assert np.max(np.abs(oracle - _decode(c, payload["exact"]["out"]))) <= 1e-5, node.id


# ================================================================ closing: attention

def _typed(c, *types):
    return [n for n in c.graph.nodes if n.type in types]


@pytest.mark.parametrize("name", ["gru", "ft-transformer", "transformer-seq", "user-attention",
                                  "lstm-attention", "gru-attention"])
def test_attention_rows_sum_to_one(name):
    c = case(name)
    section, _ = explain(c, _index(c), "exact")
    nodes = _typed(c, "attention", "encoder_layer", "attention_pool")
    assert nodes
    for node in nodes:
        payload = section["nodes"][node.id]
        alpha = _decode(c, payload["exact"]["alpha"]).astype(np.float64)
        assert np.all(alpha >= 0.0)
        assert np.max(np.abs(alpha.sum(axis=-1) - 1.0)) <= 1e-5, node.id
        maps = payload["extra"]["weights"]
        assert maps["form"] == "attention" and maps["k"] == alpha.shape[-1]
        if node.type == "attention":
            assert maps["heads"] == alpha.shape[0] == node.params["heads"]
        elif node.type == "encoder_layer":
            assert maps["heads"] == alpha.shape[0] == node.params["self_attn"]["heads"]
        else:
            assert (maps["heads"], maps["q"]) == (1, 1) and alpha.ndim == 1
    cls = section["nodes"].get("X.layer0", {}).get("extra", {}).get("cls_row")
    if name == "ft-transformer":
        alpha = _decode(c, section["nodes"]["X.layer0"]["exact"]["alpha"])
        assert np.allclose(np.asarray(cls), alpha[:, 0, :], atol=1e-8)
    elif "X.layer0" in section["nodes"]:
        assert cls is None


@pytest.mark.parametrize("name", ["lstm-attention", "gru-attention"])
def test_attention_pool_is_rb8_from_the_published_tensors(name):
    c = case(name)
    section, _ = explain(c, _index(c), "exact")
    for node in _typed(c, "attention_pool"):
        payload = section["nodes"][node.id]
        h = _decode(c, payload["exact"]["in"]).astype(np.float64)
        w, b = _tensor(c, f"{node.id}.score"), _tensor(c, f"{node.id}.score_bias")
        q = _tensor(c, f"{node.id}.query")
        s = (np.tanh(h @ w.T + b) @ q.T)[:, 0]
        alpha = np.exp(s - s.max())
        alpha /= alpha.sum()
        exact_alpha = _decode(c, payload["exact"]["alpha"])
        assert np.max(np.abs(alpha - exact_alpha)) <= 1e-6, node.id
        y = (alpha[:, None] * h).sum(axis=0)
        assert np.max(np.abs(y - _decode(c, payload["exact"]["out"]))) <= 1e-5, node.id
        oracle = O.attention_pool_weights({"score": w, "score_bias": b, "query": q}, h[None])[0]
        assert np.allclose(oracle, alpha, atol=1e-12)


def test_gat_alpha_sums_to_one_per_destination():
    c = case("gat")
    section, cap = explain(c, _index(c), "exact")
    for node in _typed(c, "gat_conv"):
        exact = section["nodes"][node.id]["exact"]
        alpha = _decode(c, exact["alpha"]).astype(np.float64)
        ends = _decode(c, exact["edge_index"])
        assert exact["edge_index"]["dtype"] == "int64" and ends.shape == (2, alpha.shape[0])
        assert alpha.shape[1] == node.params["heads"]
        for i in np.unique(ends[1]):
            assert np.max(np.abs(alpha[ends[1] == i].sum(axis=0) - 1.0)) <= 1e-5, (node.id, i)
        assert set(np.unique(ends[1])) == set(cap.focus.nodes)


# ================================================================ closing: graph messages

def _messages(c, index=None):
    section, cap = explain(c, _index(c) if index is None else index)
    return section, cap


def _node_in(cap, node):
    return cap.value_in(node.id).detach().numpy().astype(np.float64)


def _targets(cap):
    return list(cap.focus.nodes) if cap.focus.level == "graph" else [cap.focus.target]


def _relevance_check(cap, node, msgs, out_minus_bias):
    """Σ relevance over the messages (the root included) = ⟨∂s/∂out_t, out_t − b⟩ for a conv
    whose output is its sum (no map after the sum)."""
    g = cap.grad(node.id).detach().numpy().astype(np.float64)
    for t in _targets(cap):
        mine = [m["relevance"] for m in msgs if m["dst"] == t]
        want = float(np.dot(g[t], out_minus_bias[t]))
        assert abs(sum(mine) - want) <= 1e-4 * max(1.0, abs(want)), (node.id, t)


@pytest.mark.parametrize("name", ["gcn", "user-sage", "user-noloop"])
def test_gcn_and_sage_messages_reproduce_the_output_at_the_target(name):
    c = case(name)
    section, cap = _messages(c)
    for node in _typed(c, "gcn_conv", "sage_conv"):
        msgs = section["nodes"][node.id]["extra"]["messages"]
        h, out = _node_in(cap, node), cap.value(node.id).detach().numpy().astype(np.float64)
        bias_id = node.params["bias"]
        b = _tensor(c, bias_id) if bias_id else 0.0
        if node.type == "gcn_conv":
            w = _tensor(c, node.params["weight"])
            terms = [(m, m["coef"] * (w @ h[m["src"]])) for m in msgs]
        else:
            wn = _tensor(c, node.params["neigh"])
            wr = _tensor(c, node.params["root"]) if node.params["root"] else None
            terms = [(m, wr @ h[m["src"]] if m["root"] else m["coef"] * (wn @ h[m["src"]]))
                     for m in msgs]
        for t in _targets(cap):
            got = sum(v for m, v in terms if m["dst"] == t) + b
            if node.type == "sage_conv" and node.params["normalize"]:
                got = got / max(np.linalg.norm(got), 1e-12)
            assert np.max(np.abs(got - out[t])) <= 1e-4 * _scale(out[t]), (node.id, t)
            assert all(abs(m["norm"] - np.linalg.norm(v)) <= 1e-4 * _scale(v)
                       for m, v in terms if m["dst"] == t)
        if node.type == "gcn_conv" or not node.params["normalize"]:
            _relevance_check(cap, node, msgs, out - b)


@pytest.mark.parametrize("name", ["gat", "user-noloop"])
def test_gat_messages_reproduce_the_output_at_the_targets(name):
    c = case(name)
    section, cap = _messages(c)
    for node in _typed(c, "gat_conv"):
        msgs = section["nodes"][node.id]["extra"]["messages"]
        heads, concat = node.params["heads"], node.params["concat"]
        w = _tensor(c, node.params["weight"])
        b = _tensor(c, node.params["bias"]) if node.params["bias"] else 0.0
        h, out = _node_in(cap, node), cap.value(node.id).detach().numpy().astype(np.float64)
        for t in _targets(cap):
            acc = np.zeros((heads, w.shape[0] // heads))
            for m in (m for m in msgs if m["dst"] == t):
                wx = (w @ h[m["src"]]).reshape(heads, -1)
                acc += np.asarray(m["coef"])[:, None] * wx
            got = (acc.reshape(-1) if concat else acc.mean(axis=0)) + b
            assert np.max(np.abs(got - out[t])) <= 1e-4 * _scale(out[t]), (node.id, t)
        _relevance_check(cap, node, msgs, out - b)


def test_gin_messages_are_the_mlp_input_at_the_target():
    c = case("gin")
    section, cap = _messages(c)
    for node in _typed(c, "gin_conv"):
        payload = section["nodes"][node.id]["extra"]
        h = _node_in(cap, node)
        module = cap.module(node.id)
        for t in _targets(cap):
            agg = sum(m["coef"] * h[m["src"]] for m in payload["messages"] if m["dst"] == t)
            with torch.no_grad():
                got = module.mlp(torch.from_numpy(agg).float()).double().numpy()
            out = cap.value(node.id)[t].detach().numpy()
            assert np.max(np.abs(got - out)) <= 1e-4 * _scale(out), (node.id, t)
            selfs = [m for m in payload["messages"] if m["dst"] == t and m["src"] == t]
            eps = float(module.eps.detach()[0])
            assert len(selfs) == 1 and selfs[0]["hop"] == 0 and \
                selfs[0]["coef"] == pytest.approx(1.0 + eps)
        pre = payload["pre_mlp"]
        assert pre["form"] == "sequence" and pre["steps"] == len(_targets(cap))


def _isolated(batch) -> dict:
    """The batch plus one node that no edge reaches, explained alone (`target_index`)."""
    out, n = dict(batch), int(batch["x_num"].shape[0])
    out["x_num"] = torch.cat([batch["x_num"], batch["x_num"][:1] + 0.5])
    if "x_cat" in batch:
        out["x_cat"] = torch.cat([batch["x_cat"], batch["x_cat"][:1]])
    out["target_index"] = torch.tensor([n])
    return out


@pytest.mark.parametrize("detail", ["full", "exact"])
@pytest.mark.parametrize("name", ["graphsage", "user-sage", "user-noloop"])
def test_a_target_no_edge_reaches_is_explained(name, detail):
    """No in-edge and no self loop (sage keeps the root apart; gcn and gat without
    `self_loops`): the conv's messages are the root alone, or none, and every check holds."""
    c = case(name)
    batch = _isolated(c.batch)
    target = int(batch["target_index"][0])
    section, cap = explain(c, 0, detail, batch=batch)
    with torch.no_grad():
        assert np.allclose(section["head"], c.model(batch)[0].numpy())
    assert cap.focus.target == target
    for node in c.graph.nodes:
        _valid(NODE_PAYLOAD, section["nodes"][node.id], f"{name}/{detail}/{node.id}")
    for node in _typed(c, "gcn_conv", "sage_conv", "gat_conv"):
        msgs = section["nodes"][node.id]["extra"]["messages"]
        assert all(m["root"] and m["src"] == m["dst"] == target for m in msgs), node.id
        rooted = node.type == "sage_conv" and bool(node.params["root"])
        assert len(msgs) == (1 if rooted else 0), node.id
    _checks_hold(c, section, cap)


def test_messages_are_ordered_by_hop_then_relevance_and_capped():
    c = case("gcn")
    section, cap = _messages(c)
    msgs = section["nodes"]["G2"]["extra"]["messages"]
    hops = [m["hop"] for m in msgs]
    assert hops == sorted(hops) and hops[0] == 0
    ones = [abs(m["relevance"]) for m in msgs if m["hop"] == 1]
    assert ones == sorted(ones, reverse=True)
    limits = dict(c.graph.codec.limits, graph_nodes_max=2)
    graph = copy.copy(c.graph)
    graph._p = dict(graph._p, codec=_Limited(c.graph.codec, limits))
    cap2 = c.capture.Capture(graph)
    head = cap2.run(c.model, c.batch, _index(c))
    cap2.differentiate(fx.scalar(head, _index(c)))
    capped = c.registry.reader("gcn_conv").probe(cap2, graph.node("G2"))["messages"]
    assert capped == msgs[:2]


class _Limited:
    """A codec whose limits differ (the graph's `codec.limits` is what the probes read)."""

    def __init__(self, codec, limits):
        self._codec, self.limits = codec, limits

    def __getattr__(self, name):
        return getattr(self._codec, name)


def test_graph_pool_shares_and_the_edge_stream():
    c = case("gin")
    section, cap = _messages(c)
    share = section["nodes"]["P"]["extra"]["node_share"]
    x = cap.take(cap.value_in("P"), c.graph.node("P"), "in").detach().numpy()
    norms = np.linalg.norm(x, axis=1)
    assert share["nodes"] == list(cap.focus.nodes)
    assert np.allclose(share["share"], norms / norms.sum(), atol=1e-8)
    g = case("gcn")
    sec, _ = _messages(g)
    drop = sec["nodes"]["G0"]
    assert (drop["in"], drop["out"], drop["grad_out"], drop["recompute"]) == \
        (None, None, None, 0.0)
    assert sec["nodes"]["input.edges"]["extra"] == {"values": None}


# ================================================================ capture and focus

def test_focus_per_sample_shape():
    row = case("mlp-plain")
    section, cap = explain(row, 3)
    assert (cap.focus.shape, cap.focus.index) == ("row", 3)
    with torch.no_grad():
        head = row.model(row.batch)[3]
    assert np.allclose(section["head"], head.numpy())
    win = case("lstm")
    _, cap = explain(win, 2)
    x = cap.take(cap.value("input.num"), win.graph.node("input.num"))
    assert tuple(x.shape) == (6, 4) and torch.equal(x.detach(), win.batch["x_num"][2])
    node = case("gcn")
    _, cap = explain(node, 2)
    assert (cap.focus.level, cap.focus.target) == ("node", 5)
    assert cap.focus.nodes[0] == 5 and cap.focus.hops[0] == 0
    assert set(cap.focus.hops) <= {0, 1, 2}
    level = case("gat")
    _, cap = explain(level, 2)
    assert (cap.focus.level, cap.focus.nodes, cap.focus.target) == ("graph", (7, 8, 9), None)


def test_a_node_level_batch_without_targets_answers_every_node():
    c = case("graphsage")
    batch = {k: v for k, v in c.batch.items() if k != "target_index"}
    section, cap = explain(c, 4, batch=batch)
    assert (cap.focus.target, c.samples(batch)) == (4, 10)
    with torch.no_grad():
        assert np.allclose(section["head"], c.model(batch)[4].numpy())


def test_explaining_another_sample_reads_another_slice():
    c = case("mlp-residual")
    a, _ = explain(c, 0)
    b, _ = explain(c, 5)
    assert a["nodes"]["B3"]["out"] != b["nodes"]["B3"]["out"]
    assert fx.canonical(a) == fx.canonical(explain(c, 0)[0])


def test_the_graph_build_measures_through_a_shapes_capture():
    c = case("mlp-residual")
    cap = c.capture.Capture(None, "shapes")
    key = "B3"
    handle = c.graph.module(key).register_forward_hook(c.capture.hook(key))
    try:
        with cap, torch.no_grad():
            c.model(c.batch)
    finally:
        handle.remove()
    assert cap.shapes()[key] == ((7,), (16,))
    assert set(cap.shapes()) >= {key, "B3:pre"}             # the permanent hooks record too
    assert "output" not in cap.shapes()     # no hook on the network: `run` keeps its head
    with pytest.raises(c.capture.CaptureError, match="mode 'values'"):
        cap.run(c.model, c.batch)
    with pytest.raises(c.capture.CaptureError, match="reads a stage graph"):
        c.capture.Capture(None)
    with pytest.raises(c.capture.CaptureError, match="not one of"):
        c.capture.Capture(c.graph, "tensors")


def test_the_capture_keeps_calls_state_and_extras():
    c = case("bi-lstm")
    _, cap = explain(c, 0)
    h_n, c_n = cap.output("R:seq#state")[0]
    assert tuple(h_n.shape) == (2, 8, 8) and tuple(c_n.shape) == (2, 8, 8)
    assert cap.input("R") is cap.value_in("R") and cap.extras("R") == ()
    g = case("gcn")
    _, cap = explain(g, 0)
    ids, weights = cap.extras("G1")
    assert torch.equal(ids, g.batch["edge_index"]) and weights.requires_grad
    assert cap.grad_of(cap.batch["edge_weight"]) is not None
    assert cap.grad("input.num") is not None and cap.grad("input.cat") is None
    with pytest.raises(g.capture.CaptureError, match="did not run"):
        cap.seen("nowhere")


def test_capture_refusals():
    c = case("mlp-plain")
    trained = copy.deepcopy(c.model).train()
    with pytest.raises(c.capture.CaptureError, match="eval mode"):
        c.capture.Capture(c.graph).run(trained, c.batch)
    with pytest.raises(c.capture.CaptureError, match="eval mode"):
        c.capture.HookSet.attach(trained, c.graph)
    cap = c.capture.Capture(c.graph)
    with pytest.raises(c.capture.CaptureError, match="after it ran"):
        cap.differentiate(torch.ones((), requires_grad=True))
    head = cap.run(c.model, c.batch, 0)
    with pytest.raises(c.capture.CaptureError, match="one pass"):
        cap.run(c.model, c.batch, 0)
    with pytest.raises(c.capture.CaptureError, match="one number"):
        cap.differentiate(head[0:2].sum(dim=-1))
    cap.differentiate(fx.scalar(head, 0))
    with pytest.raises(c.capture.CaptureError, match="differentiated once"):
        cap.differentiate(fx.scalar(head, 0))
    with pytest.raises(IndexError):
        c.capture.Capture(c.graph).run(c.model, c.batch, 99)
    outer = c.capture.Capture(c.graph)
    with outer:
        assert c.capture.active() is outer
        with pytest.raises(c.capture.CaptureError, match="already active"):
            c.capture.Capture(c.graph).run(c.model, c.batch, 0)
    assert c.capture.active() is None
    other = case("mlp-residual")
    with pytest.raises(c.capture.CaptureError, match="not one of this network's"):
        c.capture.HookSet.sites(other.model, c.graph)


def test_a_suspended_capture_records_nothing():
    c = case("mlp-plain")
    cap = c.capture.Capture(c.graph)
    head = cap.run(c.model, c.batch, 0)
    before = dict(cap._seen)
    with cap, cap.suspended(), torch.no_grad():
        assert cap.paused
        c.model(c.batch)
    assert not cap.paused and cap._seen == before and head is cap.value("output")


# ================================================================ hygiene

def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("name", list(CASES))
def test_no_grad_no_hook_and_no_state_after_explanations(name):
    c = case(name)
    before, hooks = _state(c.model), fx.hooks_on(c.model)
    assert hooks == c.hooks.count
    for index in range(min(3, c.samples())):
        for detail in ("full", "exact", "summary"):
            explain(c, index, detail)
    assert all(p.grad is None for p in c.model.parameters())
    assert fx.hooks_on(c.model) == hooks and not c.model.training
    after = c.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(getattr(m, "kl", None) is None for m in c.model.modules()
               if type(m).__name__ == "Reparameterize")
    assert c.capture.active() is None


def test_the_hooks_are_the_contracts():
    c = case("gru-attention")
    keys = set(c.hooks.keys)
    assert keys >= {"R", "R:seq", "A", "H", "H:pre", "E", "C", "D"}
    assert "output" not in keys          # the hooks sit on the stages the port's anatomy names
    assert not any(k.startswith("input") for k in keys)
    d = case("autoencoder")
    assert {"D", "D.l0", "D.l1", "D.l2", "S", "S.0", "S.0:pre"} <= set(d.hooks.keys)
    assert "D.l0:pre" not in d.hooks.keys          # a decoder layer is the linear itself


@pytest.mark.parametrize("name", ["mlp-residual", "lstm", "gat", "ft-transformer"])
def test_a_plain_forward_under_the_hooks_is_the_forward_without_them(name):
    c = case(name)
    bare = copy.deepcopy(c.model)
    for m in bare.modules():
        m._forward_hooks.clear()
    with torch.no_grad():
        assert torch.equal(c.model(c.batch), bare(c.batch))
    section, _ = explain(c, 0)
    with torch.no_grad():
        assert torch.equal(c.model(c.batch), bare(c.batch))
    assert np.allclose(section["head"], bare(c.batch)[0].detach().numpy(), atol=1e-6)


def test_detached_hooks_leave_the_model_bare():
    c = fx._render("detach", fx.Path(fx._WORK.name), CASES["linear"])
    assert fx.hooks_on(c.model) == c.hooks.count > 0
    c.hooks.detach()
    assert fx.hooks_on(c.model) == 0 and c.hooks.count == 0
    with pytest.raises(c.capture.CaptureError, match="carries no inspection hooks"):
        c.capture.Capture(c.graph).run(c.model, c.batch, 0)
    assert c.capture.active() is None


# ================================================================ modules

STDLIB = {"__future__", "contextlib", "copy", "dataclasses", "math", "re", "threading",
          "types", "typing"}


@pytest.mark.parametrize("file", MINE)
def test_size_discipline_state_and_imports(file):
    path = INSPECTION / file
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert conformance.size_findings(tree, file) == [], "C11–C13 must hold at error"
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if isinstance(node.value, conformance.MUTABLE_NODES) and names != ["__all__"]:
                assert all(n.lstrip("_").isupper() for n in names), f"C7: {names} in {file}"
        if isinstance(node, ast.Import):
            roots = {a.name.split(".")[0] for a in node.names}
            assert roots <= STDLIB | {"numpy", "torch"}, f"{file} imports {roots}"
        if isinstance(node, ast.ImportFrom) and not node.level:
            assert node.module.split(".")[0] in STDLIB | {"numpy", "torch"}, \
                f"{file} imports {node.module}"
        if isinstance(node, ast.ImportFrom) and node.level > 1:
            assert (node.level, node.module) == (3, "domain"), f"{file} reaches {node.module}"
    assert "{{" not in text
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute)}
    assert not calls & {"backward", "train", "zero_grad", "register_forward_pre_hook"}, file
    if file != "wiring.py":        # the graph build's measure hooks, and removes, its own
        assert "register_forward_hook" not in calls or file == "capture.py", file


IMMUTABLE_CALLS = {"frozenset", "MappingProxyType", "compile", "object"}   # object(): a sentinel


def test_the_capture_holds_one_thread_local_and_no_other_state():
    for file in MINE:
        tree = ast.parse((INSPECTION / file).read_text(encoding="utf-8"))
        for n in tree.body:
            if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)):
                continue
            func = n.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            targets = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if (file, targets) == ("capture.py", ["_ACTIVE"]):
                assert name == "local"
                continue
            assert name in IMMUTABLE_CALLS or name.lstrip("_")[0].isupper(), (file, targets, name)
    text = (INSPECTION / "capture.py").read_text(encoding="utf-8")
    assert text.count("threading.local()") == 1


def test_the_templates_carry_no_dataset_literal():
    lint = REPO / "vega" / "scripts" / "tests" / "lint_templates.py"
    run = subprocess.run([sys.executable, str(lint), *[str(INSPECTION / f) for f in MINE]],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
