#!/usr/bin/env python3
"""
test_inspect_attribution.py — the attribution of an explanation and of the batch API
(`templates/serving/package/inspection/{attribution, attribution_parts, attribution_graph,
interpolate}.py`), proved on *rendered* packages (build plan I05; inspect-contract §5.4, §5.5;
D3 §8.3; D6 EI6, EI7).

    python3 test_inspect_attribution.py        (or: python3 -m pytest test_inspect_attribution.py -q)

The packages are `_inspect_fixtures.py`'s (the catalog's families and the user compositions),
plus two compositions only this file needs: a tokenizer read without a LayerNorm
(`user-tokens`) and a node-level graph network with no graph conv (`user-nodeonly`, which reads
no edge weight). Every expected number is computed apart from the modules under test: the
gradients by an autograd pass over a *copy* of the network, its embedding outputs kept by the
test's own hooks; s(x₀) by a copy whose embedding tables are zeroed, fed zeros; the hops by the
test's own walk; the edge relevances by autograd and by float64 finite differences.

  schema      every payload validates `explanation-2.schema.json`'s `attribution`, for both
              methods, per shape (row, window, graph at both levels), and is finite JSON.
  gradient    a = x ⊙ ∂s/∂x per numeric channel; an embedded nominal Σ_d e_d·∂s/∂e_d with ‖∂s/∂e‖
              as its grad (an `embedding`'s slice, a tokenizer's token); `total`; no
              completeness; a linear score's total is z − b (EI7); an input at exactly 0
              carries 0.
  integrated  completeness gap ≤ 1e-2·max(1, |s(x) − s(x₀)|) at 64 steps on row, window and
              graph networks (both levels, embeddings, the tokenizer, the reconstruction and
              svdd targets); s(x) − s(x₀) equals the independent one; a linear score's
              integrated gradients are its gradient × input; where the network is scale
              invariant at the baseline (a LayerNorm fed a zero-bias map or zero tokens) the gap
              is reported as it is, never hidden; the path at α = 1 is the network's forward,
              and a network whose forward is not its stages is refused.
  window      nominals attributed per step (`x_cat (B,L,C)`): ids and attributions per column
              and step; per_step, per_feature, total; the matrix as a heat map of block sums
              beyond `MATRIX_MAX`.
  graph       node level: the neighbours and their hops are the walk back from the target over
              the convs, nearer hops first then larger |total|, capped at `graph_nodes_max`;
              `hops`, `outside`, `self`; each listed edge's relevance is ∂s/∂w_e, every edge
              whose message reaches the target listed; graph level: the explained graph's nodes
              at hop 0, no self; no edge weight read → `edges` null.
  table       the batch API's entries equal the explanations' for the same samples and targets
              (one target per sample, both methods, every shape and level); target values.
  hygiene     no parameter `.grad`, no hook added, the network's state and mode unchanged,
              concurrent plain forwards byte-identical; refusals; C11–C13 at `error`, no
              module-level mutable state, imports, no hook or training call, no dataset literal.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import math
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _inspect_fixtures as fx  # noqa: E402

import conformance  # noqa: E402  (on sys.path through _inspect_fixtures)

INSPECTION = fx.INSPECTION
REPO = fx.REPO
MINE = ("attribution.py", "attribution_parts.py", "attribution_graph.py", "interpolate.py")
ATTRIBUTION = Draft202012Validator({"$defs": fx.EXPLANATION["$defs"],
                                    "$ref": "#/$defs/attribution"})
fx.CASES.setdefault("user-tokens", fx._spec("user-defined", [
    fx._b("T", "feature_tokenizer", d=4), fx._b("F", "flatten"),
    fx._b("D", "dense", units=8, activation="tanh"), fx._b("H", "head", out=3)],
    {"kind": "multiclass", "label": "y", "classes": fx.ABC},
    loss={"name": "cross_entropy", "params": {}}))
fx.CASES.setdefault("user-nodeonly", fx._spec("user-defined", fx.EMB + [
    fx._b("D", "dense", units=8, activation="relu"), fx._b("H", "head", out=1)],
    {"kind": "regression", "label": "y"}, sample=fx.NODE, loss={"name": "mse", "params": {}}))

ROWS = ["mlp-plain", "linear", "autoencoder", "deep-svdd", "user-tokens", "user-mixed"]
WINDOWS = ["lstm", "gru", "cnn1d", "bi-gru", "gru-attention"]
NODES = ["gcn", "graphsage", "user-noloop", "user-nodeonly"]
LEVELS = ["gat", "gin"]
EVERY = ROWS + WINDOWS + NODES + LEVELS
METHODS = ("grad_x_input", "integrated_gradients")


# ================================================================ the kit

class Kit:
    """One rendered case with its targets and attributor (the svdd centre set, as the loss
    sets it once in training)."""

    def __init__(self, name):
        self.c = c = fx.case(name)
        mod = lambda m: importlib.import_module(f"{c.package}.serving.inspection.{m}")  # noqa
        self.attr, self.interp, self.focus = mod("attribution"), mod("interpolate"), mod("focus")
        self.targets_mod = mod("targets")
        if c.model.readout.objective.value == "svdd" and not bool(c.model.svdd_center_set):
            with torch.no_grad():
                c.model.svdd_center.copy_(torch.linspace(-0.5, 0.5, c.model.svdd_center.numel()))
                c.model.svdd_center_set.fill_(True)
        self.targets = self.targets_mod.TaskTargets.of(c.spec, None, c.model)
        self.at = self.attr.Attributor(c.graph, self.targets)

    @property
    def shape(self) -> str:
        return self.c.graph.inputs.shape

    def attributor(self, **limits):
        return self.attr.Attributor(self.c.graph, self.targets,
                                    dict(self.c.graph.codec.limits, **limits))

    def explain(self, k, batch=None, request=None):
        """(a captured pass differentiated for sample k's target, the target)."""
        c = self.c
        cap = c.capture.Capture(c.graph)
        head = cap.run(c.model, c.batch if batch is None else batch, k)
        target = self.targets.choose(head, cap.batch, request, k)
        cap.differentiate(self.targets.scalar(target, head, cap.batch)[k])
        return cap, target

    def gxi(self, k, batch=None, request=None, at=None):
        cap, target = self.explain(k, batch, request)
        return (at or self.at).gradient_x_input(cap, target), target

    def ig(self, k, target, steps=64, batch=None, at=None):
        return (at or self.at).integrated(self.c.model, self.c.batch if batch is None else batch,
                                          target, k, steps)


_KITS = {}


def kit(name) -> Kit:
    if name not in _KITS:
        _KITS[name] = Kit(name)
    return _KITS[name]


# ================================================================ independent references

def _embedders(model):
    """(stage index, module, kind) of every stage that embeds nominal ids."""
    return [(i, m, k.value) for i, (k, m) in enumerate(zip(model.kinds, model.stages))
            if k.value in ("emb", "tokens") and len(getattr(m, "embeddings", None) or ())]


def _nominal_parts(module, kind, out):
    """[(x_cat column, its embedding in `out`)] — the test's own reading of the blocks."""
    if kind == "emb":
        parts, lo = [], 0
        for i, table in enumerate(module.embeddings):
            parts.append((i, out[..., lo:lo + table.embedding_dim]))
            lo += table.embedding_dim
        return parts
    n = len(module.embeddings)
    return [(j, out[:, out.shape[1] - n + j]) for j in range(n)]


def reference(k: Kit, target, batch, index):
    """∂s/∂x, ∂s/∂w, and per nominal column (Σ e·g, ‖g‖), for sample `index`, by a copy of the
    network with the test's hooks: {x, g, w_grad, nominal: (B.., C), nominal_grad}."""
    model = copy.deepcopy(k.c.model)
    kept = {}

    def keep(stage):
        def hook(module, args, out):
            out.retain_grad()
            kept[stage] = out
        return hook
    for stage, module, _ in _embedders(model):
        module.register_forward_hook(keep(stage))
    fed = dict(batch)
    fed["x_num"] = batch["x_num"].detach().clone().requires_grad_(True)
    if k.shape == "graph":
        w = batch.get("edge_weight")
        base = w.detach().clone() if w is not None else torch.ones(batch["edge_index"].shape[1])
        fed["edge_weight"] = base.requires_grad_(True)
    head = model(fed)
    k.targets.scalar(target, head, fed)[index].backward()
    lead = tuple(batch["x_num"].shape[:-1])
    c = int(batch["x_cat"].shape[-1]) if batch.get("x_cat") is not None else 0
    dot, sq = torch.zeros(lead + (c,), dtype=torch.float64), torch.zeros(lead + (c,),
                                                                            dtype=torch.float64)
    for stage, module, kind in _embedders(model):
        out = kept[stage]
        for (col, e), (_, g) in zip(_nominal_parts(module, kind, out),
                                    _nominal_parts(module, kind, out.grad)):
            dot[..., col] += (e.double() * g.double()).sum(-1).detach()
            sq[..., col] += (g.double() ** 2).sum(-1)
    return {"x": batch["x_num"].double(), "g": fed["x_num"].grad.double(),
            "w_grad": fed["edge_weight"].grad if k.shape == "graph" else None,
            "nominal": dot, "nominal_grad": sq.sqrt()}


def s_at_baseline(k: Kit, target, batch, index) -> float:
    """s(x₀): a copy with zeroed embedding tables, fed zeros in shaped space."""
    model = copy.deepcopy(k.c.model)
    with torch.no_grad():
        for _, module, _ in _embedders(model):
            for table in module.embeddings:
                table.weight.zero_()
        zero = dict(batch, x_num=torch.zeros_like(batch["x_num"]))
        return float(k.targets.scalar(target, model(zero), zero)[index])


def s_at(k: Kit, target, batch, index) -> float:
    with torch.no_grad():
        return float(k.targets.scalar(target, k.c.model(batch), batch)[index])


def in_order(k: Kit, numeric, nominal):
    """Values over inputs.order from numeric channels and nominal columns (the test's own)."""
    n = len(k.c.graph.inputs.numeric)
    return torch.cat([numeric[..., :n], nominal, numeric[..., n:]], dim=-1)


def walk_back(edge_index, target, hops) -> dict:
    """{node: hop} of every node within `hops` message steps of `target` (the test's own)."""
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    dist, frontier = {int(target): 0}, {int(target)}
    for h in range(1, hops + 1):
        frontier = {j for j, i in edges if i in frontier and j not in dist}
        dist.update({j: h for j in frontier})
    return dist


def close(got, want, tol=1e-5):
    got, want = torch.as_tensor(got, dtype=torch.float64), torch.as_tensor(want,
                                                                            dtype=torch.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    scale = max(1.0, float(want.abs().max())) if want.numel() else 1.0
    worst = float((got - want).abs().max()) if want.numel() else 0.0
    assert worst <= tol * scale, f"max |Δ| {worst:.3g} > {tol:g}·{scale:.3g}"


def _consistent(comp):
    """gap = |sum − target_minus_baseline|, up to the 9 significant digits each carries."""
    diff = abs(comp["sum"] - comp["target_minus_baseline"])
    assert abs(comp["gap"] - diff) <= 5e-9 * diff + 1e-15 * abs(comp["sum"]), comp


def _valid(payload):
    errors = sorted(ATTRIBUTION.iter_errors(payload), key=lambda e: list(e.path))
    assert not errors, [f"{list(e.path)}: {e.message[:160]}" for e in errors[:3]]
    json.dumps(payload, allow_nan=False)


# ================================================================ schema

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", EVERY)
def test_every_payload_validates_the_schema(name, method):
    k = kit(name)
    payload, target = k.gxi(1)
    if method == "integrated_gradients":
        payload = k.ig(1, target, steps=16)
        assert set(payload["completeness"]) == {"sum", "target_minus_baseline", "gap"}
    else:
        assert payload["completeness"] is None
    _valid(payload)
    assert (payload["method"], payload["unit"], payload["shape"]) == \
        (method, target.unit, k.shape)


# ================================================================ gradient × input

@pytest.mark.parametrize("name", EVERY)
def test_gradient_x_input_is_the_networks_own_gradient(name):
    k, index = kit(name), 1
    payload, target = k.gxi(index)
    ref = reference(k, target, k.c.batch, index)
    a, an = ref["x"] * ref["g"], ref["nominal"]
    if k.shape == "row":
        close(payload["features"]["attribution"], in_order(k, a[index], an[index]))
        close(payload["features"]["grad"], in_order(k, ref["g"][index], ref["nominal_grad"][index]))
        want_total = in_order(k, a[index], an[index]).sum()
    elif k.shape == "window":
        close(payload["matrix"], a[index])
        close(payload["nominal"]["attribution"] if payload["nominal"] else [],
              an[index].t() if payload["nominal"] else [])
        want_total = a[index].sum() + an[index].sum()
    else:
        focus = k.focus.Focus.of(k.c.graph, k.c.batch, index)
        per = in_order(k, a, an)
        if payload["self"] is not None:
            close(payload["self"], per[focus.target])
        for nb in payload["neighbours"]:
            close(nb["total"], per[nb["node"]].sum())
        want_total = per.sum()
    close(payload["total"], want_total)


def test_nominal_columns_are_attributed_at_their_embedding():
    for name in ("mlp-plain", "user-tokens"):          # an embedding's slice, a tokenizer token
        k = kit(name)
        payload, target = k.gxi(2)
        ref = reference(k, target, k.c.batch, 2)
        n = len(k.c.graph.inputs.numeric)
        names = payload["features"]["names"]
        assert names[n:] == list(k.c.graph.inputs.nominal)
        assert payload["features"]["values"][n:] == k.c.batch["x_cat"][2].tolist()
        close(payload["features"]["attribution"][n:], ref["nominal"][2])
        close(payload["features"]["grad"][n:], ref["nominal_grad"][2])
        assert abs(payload["features"]["attribution"][n]) > 0


def test_a_nominal_no_stage_embeds_carries_nothing():
    k = kit("vae")                       # nominal zone in the contract, no embedding block
    payload, _ = k.gxi(0)
    n = len(k.c.graph.inputs.numeric)
    assert payload["features"]["attribution"][n:] == [0.0]
    assert payload["features"]["grad"][n:] == [0.0]


def test_a_linear_score_is_complete():
    """EI7: on a linear network the attributions sum to z − b (grad × input and integrated
    gradients alike), and the two methods agree."""
    k = kit("linear")
    head_module = k.c.model.stages[-1]
    for index in range(4):
        payload, target = k.gxi(index)
        with torch.no_grad():
            z = k.c.model(k.c.batch)[index, target.index]
        b = head_module.linear.bias[target.index].detach()
        close(payload["total"], float(z - b))
        ig = k.ig(index, target)
        close(ig["features"]["attribution"], payload["features"]["attribution"])
        assert ig["completeness"]["gap"] <= 1e-5 * max(1.0, abs(float(z)))
        close(ig["completeness"]["target_minus_baseline"], float(z - b))


def test_an_input_at_zero_carries_nothing():
    k = kit("mlp-plain")
    batch = dict(k.c.batch, x_num=k.c.batch["x_num"].clone())
    batch["x_num"][3, 2] = 0.0
    payload, _ = k.gxi(3, batch=batch)
    assert payload["features"]["attribution"][2] == 0.0
    assert payload["features"]["grad"][2] != 0.0


# ================================================================ integrated gradients

COMPLETE = ["mlp-plain", "linear", "autoencoder", "deep-svdd", "user-tokens", "lstm", "gru",
            "cnn1d", "bi-gru", "gcn", "graphsage", "user-nodeonly", "gat", "gin"]


@pytest.mark.parametrize("name", COMPLETE)
def test_integrated_gradients_are_complete(name):
    """The gap ≤ 1e-2·max(1, |Δ|) at 64 steps, and Δ = s(x) − s(x₀) as computed apart."""
    k = kit(name)
    for index in range(3):
        _, target = k.gxi(index)
        payload = k.ig(index, target, steps=64)
        comp = payload["completeness"]
        delta = s_at(k, target, k.c.batch, index) - s_at_baseline(k, target, k.c.batch, index)
        close(comp["target_minus_baseline"], delta, 1e-4)
        assert comp["sum"] == payload["total"]
        assert comp["gap"] <= 1e-2 * max(1.0, abs(delta)), (index, comp)
        _consistent(comp)


@pytest.mark.parametrize("name", ["ft-transformer", "mlp-residual", "transformer-seq",
                                  "user-sage"])
def test_the_gap_is_reported_as_it_is(name):
    """A LayerNorm fed a zero-bias map (or the tokenizer's zero tokens, or a transformer-seq
    encoder's) and an L2-normalised sage conv are scale invariant at the zero baseline: s jumps
    within α ≈ √eps, which 64 midpoints cannot resolve. The gap says so — it is
    |Σa − (s(x) − s(x₀))| with both ends the network's own (A31: the known exceptions)."""
    k = kit(name)
    for index in range(3):
        _, target = k.gxi(index)
        comp = k.ig(index, target)["completeness"]
        delta = s_at(k, target, k.c.batch, index) - s_at_baseline(k, target, k.c.batch, index)
        close(comp["target_minus_baseline"], delta, 1e-4)
        _consistent(comp)


def test_integrated_gradients_average_the_path():
    """a = x ⊙ mean_α ∂s/∂x(αx) at the midpoints, recomputed by the network's own forward on
    the interpolants (a network without embeddings: the interpolants are plain batches)."""
    k, index, steps = kit("rnn"), 2, 8
    _, target = k.gxi(index)
    payload = k.ig(index, target, steps=steps)
    x = k.c.batch["x_num"][index:index + 1]
    grads = []
    for m in range(steps):
        xa = ((m + 0.5) / steps * x).detach().requires_grad_(True)
        s = k.targets.scalar(target, k.c.model({"x_num": xa}), {"x_num": xa})[0]
        grads.append(torch.autograd.grad(s, xa)[0][0].double())
    close(payload["matrix"], x[0].double() * torch.stack(grads).mean(0))


def test_the_path_is_the_networks_forward():
    for name in EVERY + ["ft-transformer", "mlp-residual", "transformer-seq", "user-attention"]:
        k = kit(name)
        path = k.interp.Path(k.c.model, k.shape == "graph")
        run = path.run(k.c.batch, torch.ones(1))
        with torch.no_grad():
            plain = k.c.model(k.c.batch)
        close(run.head.detach(), plain, 1e-6)
        run.check(plain, 0)


def test_a_network_whose_forward_is_not_its_stages_is_refused():
    k = kit("mlp-plain")
    other = copy.deepcopy(k.c.model)
    forward = other.forward
    other.forward = lambda batch: forward(batch) + 1.0
    _, target = k.gxi(0)
    with pytest.raises(k.interp.AttributionError, match="not the network's"):
        k.at.integrated(other, k.c.batch, target, 0, 8)
    with pytest.raises(k.interp.AttributionError, match="not the network's"):
        k.at.table(other, k.c.batch, [target] * 8)


# ================================================================ windows

def test_window_nominals_are_attributed_per_step():
    k, index = kit("lstm"), 3
    batch = k.c.batch
    steps = int(batch["x_num"].shape[1])
    for method in METHODS:
        payload, target = k.gxi(index)
        if method == "integrated_gradients":
            payload = k.ig(index, target, steps=16)
        nominal = payload["nominal"]
        assert nominal["names"] == ["zone"]
        assert nominal["values"] == [batch["x_cat"][index, :, 0].tolist()]
        assert len(nominal["attribution"]) == 1 and len(nominal["attribution"][0]) == steps
        matrix = torch.tensor(payload["matrix"], dtype=torch.float64)
        per_nominal = torch.tensor(nominal["attribution"], dtype=torch.float64).sum(0)
        close(payload["per_step"], matrix.sum(1) + per_nominal)
        close(payload["per_feature"], matrix.sum(0))
        close(payload["total"], matrix.sum() + per_nominal.sum())
        assert payload["steps"] == steps == len(payload["step_labels"])
        assert payload["step_labels"][-1] == "t"
        assert payload["features"]["names"] == list(k.c.graph.inputs.numeric)
    ref = reference(k, target, batch, index)
    close(k.gxi(index)[0]["nominal"]["attribution"], ref["nominal"][index].t())
    assert torch.count_nonzero(ref["nominal"][index]) == steps     # every step carries its own


def test_the_autoregressive_channel_is_a_window_feature():
    k = kit("gru")
    payload, _ = k.gxi(0)
    inputs = k.c.graph.inputs
    assert inputs.past and payload["features"]["names"] == [*inputs.numeric, inputs.past]
    assert len(payload["matrix"][0]) == len(inputs.numeric) + 1
    assert payload["nominal"] is None


def test_a_large_window_matrix_is_a_heat_map(monkeypatch):
    k = kit("lstm")
    whole, _ = k.gxi(1)
    monkeypatch.setattr(k.attr, "MATRIX_MAX", 5)
    small, _ = k.gxi(1)
    _valid(small)
    heat = small["matrix"]
    assert set(heat) == {"rows", "cols", "row_edges", "col_edges", "values"}
    close(sum(map(sum, heat["values"])), sum(map(sum, whole["matrix"])))
    assert small["per_step"] == whole["per_step"]


# ================================================================ graphs

def test_the_node_neighbourhood_is_the_walk_back_over_the_convs():
    k = kit("gcn")
    batch, convs = k.c.batch, 2
    for index in range(3):
        payload, target = k.gxi(index)
        t = int(batch["target_index"][index])
        dist = walk_back(batch["edge_index"], t, convs)
        assert payload["target_node"] == t and payload["level"] == "node"
        assert {n["node"]: n["hop"] for n in payload["neighbours"]} == dist
        order = [(n["hop"], -abs(n["total"])) for n in payload["neighbours"]]
        assert order == sorted(order) and payload["neighbours"][0]["node"] == t
        ref = reference(k, target, batch, index)
        per = in_order(k, ref["x"] * ref["g"], ref["nominal"])
        close(payload["self"], per[t])
        for h in {str(v) for v in dist.values()}:
            close(payload["hops"][h], sum(float(per[j].sum()) for j, v in dist.items()
                                          if str(v) == h))
        assert abs(payload["outside"]) <= 1e-6
        for nb in payload["neighbours"]:
            top = sorted(range(per.shape[1]), key=lambda f: (-abs(float(per[nb["node"], f])), f))
            assert [f for f, _ in nb["top"]] == top[:5]


def test_the_neighbourhood_is_capped_nearer_hops_first():
    k = kit("gcn")
    at = k.attributor(graph_nodes_max=3)
    whole, _ = k.gxi(1)
    capped, _ = k.gxi(1, at=at)
    kept = capped["neighbours"]
    assert len(kept) == 3 and kept[0]["hop"] == 0
    rank = sorted(whole["neighbours"], key=lambda n: (n["hop"], -abs(n["total"])))
    assert [n["node"] for n in kept] == [n["node"] for n in rank[:3]]
    assert capped["hops"] == whole["hops"]
    close(capped["outside"], whole["total"] - sum(n["total"] for n in kept))
    nodes = {n["node"] for n in kept}
    assert all(e["src"] in nodes and e["dst"] in nodes for e in capped["edges"])
    assert at.neighbourhood(k.c.batch["edge_index"], (2,), 2)[0][:1] == (2,)


def test_each_edge_carries_its_relevance():
    k = kit("gcn")
    batch = k.c.batch
    for index in range(3):
        for method in METHODS:
            payload, target = k.gxi(index)
            if method == "integrated_gradients":
                payload = k.ig(index, target, steps=8)
            t = int(batch["target_index"][index])
            dist = walk_back(batch["edge_index"], t, 2)
            ref = reference(k, target, batch, index)["w_grad"]
            src, dst = batch["edge_index"].tolist()
            want = sorted((dist[i] + 1, e) for e, (j, i) in enumerate(zip(src, dst))
                          if i in dist and j in dist and dist[i] < 2)
            assert [(e["hop"], e["src"], e["dst"]) for e in payload["edges"]] == \
                [(h, src[e], dst[e]) for h, e in want]
            close([e["relevance"] for e in payload["edges"]], ref[[e for _, e in want]])


def test_edge_relevance_agrees_with_finite_differences():
    k, index = kit("gcn"), 0
    payload, target = k.gxi(index)
    model = copy.deepcopy(k.c.model).double()
    batch = dict(k.c.batch, x_num=k.c.batch["x_num"].double(),
                 edge_weight=k.c.batch["edge_weight"].double())

    def s(w):
        fed = dict(batch, edge_weight=w)
        with torch.no_grad():
            return float(k.targets.scalar(target, model(fed), fed)[index])
    for edge in payload["edges"][:4]:
        e = [j for j, (a, b) in enumerate(zip(*batch["edge_index"].tolist()))
             if (a, b) == (edge["src"], edge["dst"])][0]
        step = torch.zeros_like(batch["edge_weight"])
        step[e] = 1e-6
        slope = (s(batch["edge_weight"] + step) - s(batch["edge_weight"] - step)) / 2e-6
        assert math.isclose(edge["relevance"], slope, rel_tol=1e-3, abs_tol=1e-5), (edge, slope)


def test_a_network_that_reads_no_edge_weight_has_no_edges():
    k = kit("user-nodeonly")
    for method in METHODS:
        payload, target = k.gxi(1)
        if method == "integrated_gradients":
            payload = k.ig(1, target, steps=8)
        t = int(k.c.batch["target_index"][1])
        assert payload["edges"] is None
        assert [(n["node"], n["hop"]) for n in payload["neighbours"]] == [(t, 0)]
        assert list(payload["hops"]) == ["0"]


@pytest.mark.parametrize("name", LEVELS)
def test_a_graph_level_sample_is_its_graphs_nodes(name):
    k = kit(name)
    batch = k.c.batch
    for index in range(3):
        members = torch.nonzero(batch["graph_index"] == index).flatten().tolist()
        for method in METHODS:
            payload, target = k.gxi(index)
            if method == "integrated_gradients":
                payload = k.ig(index, target, steps=16)
            assert (payload["level"], payload["target_node"], payload["self"]) == \
                ("graph", None, None)
            assert sorted(n["node"] for n in payload["neighbours"]) == members
            assert {n["hop"] for n in payload["neighbours"]} == {0}
            totals = [abs(n["total"]) for n in payload["neighbours"]]
            assert totals == sorted(totals, reverse=True)
            close(payload["hops"]["0"], payload["total"])
            assert abs(payload["outside"]) <= 1e-6 * max(1.0, abs(payload["total"]))
            inside = {(a, b) for a, b in zip(*batch["edge_index"].tolist())
                      if a in members and b in members}
            assert {(e["src"], e["dst"]) for e in payload["edges"]} == inside
            assert {e["hop"] for e in payload["edges"]} <= {1}
        ref = reference(k, target, batch, index)
        close(k.gxi(index)[0]["total"], (ref["x"] * ref["g"]).sum())
        edges = k.gxi(index)[0]["edges"]
        pos = {(a, b): e for e, (a, b) in enumerate(zip(*batch["edge_index"].tolist()))}
        close([e["relevance"] for e in edges], ref["w_grad"][[pos[(e["src"], e["dst"])]
                                                             for e in edges]])


# ================================================================ the batch API

def _requests(k: Kit, n):
    """One selector per sample, cycling through the task's classes where it has them."""
    outputs = k.targets.outputs
    if outputs["kind"] == "classes":
        return [{"class": outputs["names"][i % len(outputs["names"])]} for i in range(n)]
    return [None] * n


@pytest.mark.parametrize("name", ["mlp-residual", "user-tokens", "lstm", "gru", "gcn", "gin"])
def test_the_table_agrees_with_the_explanations(name):
    k = kit(name)
    samples = k.c.samples()
    requests = _requests(k, samples)
    with torch.no_grad():
        head = k.c.model(k.c.batch)
    targets = [k.targets.choose(head, k.c.batch, requests[i], i) for i in range(samples)]
    for method in METHODS:
        table = k.at.table(k.c.model, k.c.batch, targets, method, 16)
        assert len(table) == samples
        for i in (0, samples - 1):
            payload, target = k.gxi(i, request=requests[i])
            assert target == targets[i]
            if method == "integrated_gradients":
                payload = k.ig(i, target, steps=16)
            assert table[i]["target_value"] == k.targets.value(target, head, k.c.batch, i)
            _agrees(k, table[i]["attribution"], payload, i)


def _agrees(k: Kit, entry, payload, i):
    if k.shape == "row":
        close(entry, payload["features"]["attribution"])
    elif k.shape == "window":
        close(entry["per_step"], payload["per_step"])
        nominal = torch.tensor(payload["nominal"]["attribution"]).sum(1) \
            if payload["nominal"] else torch.zeros(0)
        close(entry["per_feature"], in_order(k, torch.tensor(payload["per_feature"]), nominal))
    elif payload["level"] == "node":
        close(entry["self"], payload["self"])
        assert set(entry["hops"]) == set(payload["hops"])
        close([entry["hops"][h] for h in sorted(entry["hops"])],
              [payload["hops"][h] for h in sorted(payload["hops"])])
    else:
        close(entry["hops"]["0"], payload["total"])
        close(sum(entry["self"]), payload["total"])


def test_the_table_refuses_what_it_cannot_attribute():
    k = kit("mlp-plain")
    _, target = k.gxi(0)
    with pytest.raises(ValueError, match="one per sample"):
        k.at.table(k.c.model, k.c.batch, [target] * 3)
    with pytest.raises(ValueError, match="not one of"):
        k.at.table(k.c.model, k.c.batch, [target] * 8, "shap")
    with pytest.raises(ValueError, match="steps"):
        k.at.table(k.c.model, k.c.batch, [target] * 8, "integrated_gradients", 0)


# ================================================================ hygiene

def test_refusals():
    k = kit("gcn")
    _, target = k.gxi(0)
    for steps in (0, 129, True, 2.0):
        with pytest.raises(ValueError, match="steps"):
            k.at.integrated(k.c.model, k.c.batch, target, 0, steps)
    with pytest.raises(k.interp.AttributionError, match="ig_batch_max"):
        k.attributor(ig_batch_max=100).integrated(k.c.model, k.c.batch, target, 0, 11)
    training = copy.deepcopy(k.c.model).train()
    with pytest.raises(k.interp.AttributionError, match="training"):
        k.at.integrated(training, k.c.batch, target, 0, 8)
    cap = k.c.capture.Capture(k.c.graph)
    with cap, pytest.raises(k.interp.AttributionError, match="outside an explanation"):
        k.at.integrated(k.c.model, k.c.batch, target, 0, 8)


@pytest.mark.parametrize("name", ["mlp-plain", "ft-transformer", "lstm", "gcn", "gat"])
def test_nothing_is_written_to_the_network(name):
    k = kit(name)
    model = k.c.model
    state = {key: v.clone() for key, v in model.state_dict().items()}
    hooks = fx.hooks_on(model)
    _, target = k.gxi(0)
    k.ig(0, target, steps=8)
    k.at.table(model, k.c.batch, [target] * k.c.samples(), "integrated_gradients", 4)
    assert all(p.grad is None for p in model.parameters())
    assert fx.hooks_on(model) == hooks and not model.training
    assert all(torch.equal(state[key], v) for key, v in model.state_dict().items())


@pytest.mark.parametrize("name", ["mlp-plain", "lstm"])
def test_concurrent_predictions_are_untouched(name):
    k = kit(name)
    with torch.no_grad():
        want = k.c.model(k.c.batch).numpy().tobytes()
    _, target = k.gxi(0)
    seen, errors = [], []

    def explain():
        try:
            for _ in range(4):
                k.ig(0, target, steps=16)
        except Exception as exc:          # pragma: no cover - reported below
            errors.append(exc)

    def predict():
        for _ in range(40):
            with torch.no_grad():
                seen.append(k.c.model(k.c.batch).numpy().tobytes())
    threads = [threading.Thread(target=explain), threading.Thread(target=explain),
               threading.Thread(target=predict)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(seen) == 40 and all(s == want for s in seen)


STDLIB = {"__future__", "dataclasses", "math", "types", "typing"}


@pytest.mark.parametrize("file", MINE)
def test_size_discipline_state_and_imports(file):
    text = (INSPECTION / file).read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert conformance.size_findings(tree, file) == [], "C11–C13 must hold at error"
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if isinstance(node.value, conformance.MUTABLE_NODES) and names != ["__all__"]:
                assert all(n.lstrip("_").isupper() for n in names), f"C7: {names} in {file}"
            if isinstance(node.value, ast.Call):
                func = node.value.func
                called = func.attr if isinstance(func, ast.Attribute) else func.id
                assert called in {"MappingProxyType", "frozenset"} or called[0].isupper(), \
                    (file, names, called)
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= STDLIB | {"numpy", "torch"}
        if isinstance(node, ast.ImportFrom) and not node.level:
            assert node.module.split(".")[0] in STDLIB | {"numpy", "torch"}, node.module
        if isinstance(node, ast.ImportFrom) and node.level > 1:
            assert (node.level, node.module) == (3, "domain"), node.module
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute)}
    assert not calls & {"backward", "train", "eval", "zero_grad", "register_forward_hook",
                        "register_forward_pre_hook", "load_state_dict", "copy_"}, file
    assert "{{" not in text and "threading" not in text


def test_the_templates_carry_no_dataset_literal():
    lint = REPO / "vega" / "scripts" / "tests" / "lint_templates.py"
    run = subprocess.run([sys.executable, str(lint), *[str(INSPECTION / f) for f in MINE]],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
