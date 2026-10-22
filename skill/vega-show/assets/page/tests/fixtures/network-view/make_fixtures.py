#!/usr/bin/env python3
"""
make_fixtures.py — the vega/network-view@2 fixtures of the page's net/ unit tests (graph.test.js).

    python3 make_fixtures.py            # write every fixture next to this file
    python3 make_fixtures.py --check    # validate every *.json here against network-view-2.schema.json

Each fixture is a small, hand-composed stage graph in D3's form (inspect-contract.md §3, §5): the
nodes a reader of the named block types adds (composites with their children and synthetic merge
nodes, the input and output io nodes), the data edges, the tensors (json lists of 9 significant
digits up to json_list_max floats, base64 little-endian float32 above it, R-F8) with their
summaries, the dense chain when D3 §5.4 makes one eligible, the task readout and the sample. The
weights are seeded, so the files are reproducible byte for byte. The names are neutral (x0, k0, …):
a fixture belongs to no lane, and the template lint scans this directory too.

The shapes the unit tests read:
  mlp-plain      row, multiclass: dense, activation, dropout, dense (own gelu), head — a pure dense chain
  linear         row, regression: the head alone
  mlp-residual   row, binary: dense, residual(dense, batchnorm, activation, dropout), head
  embedding      row, multiclass with two nominal columns: embedding, concat, dense, head (chain, not pure)
  autoencoder    row, unsupervised: sequential(dense, dense), dense (the waist), decoder, head
  parallel       row, regression: dense, parallel(dense | sequential(dense, layernorm); concat), custom, head
  lstm-window    window, multiclass: lstm, dropout, head
  gcn-node       graph at node level, multiclass: edge_dropout, gcn_conv, activation, dropout, gcn_conv, head
  ordinal        row, ordinal: dense, the rank-consistent head (one score row, k−1 thresholds) — a pure chain
  svdd           row, anomaly under the svdd loss: dense, batchnorm, activation, head without bias, the centre buffer
  forecast       window, forecast under the quantile loss: flatten, dense, head of horizon × quantiles units
  recon-nominal  row, unsupervised with a nominal input: embedding, concat, dense, head rebuilding recon_dim units
  residual-deep  row, regression: dense, residual of 3 repeats (dense, activation; the first with a projection),
                 layernorm, dropout, head — the canvas's overflow rule (D4 §7.3) and the skip arcs
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]

CATALOG = REPO / "vega-forge" / "scripts" / "catalog.json"
JSON_LIST_MAX = 1024


def sig9(v) -> float:
    return float(f"{float(v):.9g}")


def digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def summary(a: np.ndarray) -> dict:
    x = a.astype(np.float64).ravel()
    lo, hi = float(x.min()), float(x.max())
    counts, _ = np.histogram(x, bins=32, range=(lo, hi if hi > lo else lo + 1.0))
    q = np.quantile(np.abs(x), [0.5, 0.9, 0.99, 1.0])
    return {"n": int(x.size), "mean": sig9(x.mean()), "std": sig9(x.std()), "min": sig9(lo), "max": sig9(hi),
            "l2": sig9(np.sqrt((x * x).sum())), "abs_q": [sig9(v) for v in q],
            "hist": {"lo": sig9(lo), "hi": sig9(hi), "counts": [int(c) for c in counts]}}


class Net:
    """One network-view@2 document under construction."""

    def __init__(self, name: str, seed: int, catalog: dict):
        self.name, self.rng, self.views = name, np.random.default_rng(seed), catalog["inspection"]["views"]
        self.nodes, self.edges, self.tensors = [], [], {}

    # ---- tensors
    def tensor(self, node: str, role: str, arr: np.ndarray, axes: list[str], suffix: str | None = None) -> str:
        tid = f"{node}.{suffix or role}"
        a = arr.astype(np.float32)
        if a.size <= JSON_LIST_MAX:
            enc, data = "json", [sig9(v) for v in a.ravel()]
        else:
            enc, data = "base64-le", base64.b64encode(a.astype("<f4").tobytes()).decode("ascii")
        self.tensors[tid] = {"id": tid, "node": node, "role": role, "shape": list(a.shape), "axes": axes,
                             "dtype": "float32", "encoding": enc, "data": data, "summary": summary(a)}
        return tid

    def weights(self, node: str, n_in: int, n_out: int, bias: bool = True) -> tuple[str, str | None, int, np.ndarray]:
        w = (self.rng.standard_normal((n_out, n_in)) / np.sqrt(n_in)).astype(np.float32)
        wid = self.tensor(node, "weight", w, ["out", "in"])
        bid = self.tensor(node, "bias", (self.rng.standard_normal(n_out) * 0.1).astype(np.float32), ["out"]) if bias else None
        return wid, bid, n_in * n_out + (n_out if bias else 0), w

    # ---- nodes
    def node(self, nid: str, type_: str, *, parent=None, children=(), role=None, call=None, path="",
             in_shape=None, out_shape=(), axes=("unit",), per="sample", config=None, params=None, block=None, code=True):
        view = self.views[type_]
        cls = {"dense": "Dense", "head": "Head"}.get(type_, "".join(p.title() for p in type_.split("_")))
        self.nodes.append({
            "id": nid, "block": block if block is not None else (None if type_ in ("input", "output", "residual_add", "parallel_merge", "encoder_layer") else nid.split(".")[0]),
            "type": type_, "view": view["view"], "parent": parent, "children": list(children), "role": role,
            "call": call or view["call"], "path": path, "in_shape": None if in_shape is None else list(in_shape),
            "out_shape": list(out_shape), "axes": list(axes), "per": per, "config": config or {},
            "params": {"count": 0, "trainable": 0, **(params or {})}, "formula": view["formula"],
            "code": {"file": "model/blocks_core.py", "class": cls, "anchor": f"class {cls}("} if code else None})
        return self.nodes[-1]

    def dense(self, nid: str, n_in: int, n_out: int, *, act="none", type_="dense", **kw):
        wid, bid, count, w = self.weights(nid, n_in, n_out)
        return self.node(nid, type_, in_shape=[n_in], out_shape=[n_out], config={"units" if type_ == "dense" else "out": n_out, "activation": act, "bias": True},
                         params={"count": count, "trainable": count, "weight": wid, "bias": bid, "activation": act,
                                 "row_norms": [sig9(v) for v in np.linalg.norm(w, axis=1)],
                                 "col_norms": [sig9(v) for v in np.linalg.norm(w, axis=0)]}, **kw)

    def pointwise(self, nid: str, type_: str, width, **kw):
        cfg = {"name": kw.pop("name")} if type_ == "activation" else {"p": kw.pop("p")}
        params = {"name": cfg["name"]} if type_ == "activation" else {"p": cfg["p"], "active_at_inference": False}
        shape = width if isinstance(width, list) else [width]
        return self.node(nid, type_, in_shape=shape, out_shape=shape, config=cfg, params=params, **kw)

    def norm(self, nid: str, type_: str, width: int, **kw):
        g = (1 + self.rng.standard_normal(width) * 0.1).astype(np.float32)
        b = (self.rng.standard_normal(width) * 0.1).astype(np.float32)
        params = {"count": 2 * width, "trainable": 2 * width, "gamma": [sig9(v) for v in g], "beta": [sig9(v) for v in b], "eps": 1e-5}
        if type_ == "batchnorm":
            mean = (self.rng.standard_normal(width) * 0.2).astype(np.float32)
            var = (0.5 + self.rng.random(width)).astype(np.float32)
            scale = g / np.sqrt(var + 1e-5)
            params.update(running_mean=self.tensor(nid, "buffer", mean, ["unit"], "running_mean"),
                          running_var=self.tensor(nid, "buffer", var, ["unit"], "running_var"),
                          scale=[sig9(v) for v in scale], shift=[sig9(v) for v in b - scale * mean])
        return self.node(nid, type_, in_shape=[width], out_shape=[width], config={"eps": 1e-5}, params=params, **kw)

    def edge(self, a: str, b: str, kind: str = "data"):
        self.edges.append({"from": a, "to": b, "kind": kind})

    def chain(self, a: str, *rest: str):
        for x, y in zip((a,) + rest, rest):
            self.edge(x, y)

    def group_count(self, nid: str):
        """A composite's params.count: the sum over its descendants."""
        byid = {n["id"]: n for n in self.nodes}

        def total(i):
            n = byid[i]
            return (n["params"]["count"] if not n["children"] else 0) + sum(total(c) for c in n["children"])
        node = byid[nid]
        node["params"]["count"] = node["params"]["trainable"] = sum(total(c) for c in node["children"])

    # ---- the document
    def document(self, *, family, task, sample, inputs, chain=None) -> dict:
        roots = [n for n in self.nodes if n["parent"] is None and n["type"] not in ("input", "output")]
        total = sum(n["params"]["count"] for n in roots)
        body = {
            "schema": "vega/network-view@2", "available": True, "model_version": digest(self.name + ":model"),
            "spec_digest": digest(self.name + ":spec"), "weights": "full",
            "family": {"name": family, "source": "catalog"}, "task": task, "sample": sample, "inputs": inputs,
            "params": {"total": total, "trainable": total}, "nodes": self.nodes, "edges": self.edges,
            "chain": chain, "tensors": self.tensors, "limits": json.loads(CATALOG.read_text())["inspection"]["limits"],
            "inspection": {"schema": "vega/inspection@1", "readers": digest("readers@1")},
        }
        canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {**body, "network_digest": digest(canon)}


# ---- the task, sample and inputs blocks ---------------------------------------------------------
def task(kind, link, out_kind, names, rule, targets, *, base=None, positive=None, threshold=None, scale=None, quantiles=None):
    return {"kind": kind, "base": base, "link": link, "outputs": {"kind": out_kind, "names": names}, "positive": positive,
            "threshold": threshold, "threshold_scale": scale, "quantiles": quantiles, "decision_rule": rule,
            "targets": targets, "default_target": targets[0]}


def sample(shape="row", *, length=None, stride=None, horizon=None, level=None, directed=None, self_loops=None,
           protocol=None, batching=None, autoregressive=None):
    return {"shape": shape, "length": length, "stride": stride, "horizon": horizon, "autoregressive": autoregressive,
            "level": level, "directed": directed, "self_loops": self_loops, "protocol": protocol, "batching": batching}


def inputs(numeric, nominal=(), *, step=None, edges=None):
    nominal = list(nominal)
    order = list(numeric) + [c["name"] for c in nominal]
    width = len(numeric) + sum(c["dim"] or 0 for c in nominal)
    return {"numeric": list(numeric), "nominal": nominal, "past_target": None, "order": order, "width": width,
            "edges": edges, "axis_labels": {"feature": order, "step": step, "token": None}}


def names(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def io_in(net: Net, nid: str, stream: str, columns, out_shape, axes=("feature",), per="sample"):
    return net.node(nid, "input", call="io", in_shape=None, out_shape=out_shape, axes=axes, per=per,
                    params={"columns": list(columns), "stream": stream}, block=None, code=False)


def io_out(net: Net, link: str, out_kind: str, labels, per="sample"):
    return net.node("output", "output", call="io", in_shape=[len(labels)], out_shape=[len(labels)], axes=("output",), per=per,
                    params={"link": link, "outputs": {"kind": out_kind, "names": list(labels)}}, block=None, code=False)


def chain_layer(net: Net, nid: str, activation, dropout) -> dict:
    n = next(x for x in net.nodes if x["id"] == nid)
    return {"node": nid, "in": n["in_shape"][0], "out": n["out_shape"][0], "activation": activation, "dropout": dropout,
            "weight": n["params"]["weight"], "bias": n["params"]["bias"], "params": n["params"]["count"]}


# ---- the fixtures -------------------------------------------------------------------------------
def mlp_plain(cat):
    net, xs, ks = Net("mlp-plain", 11, cat), names("x", 8), names("k", 3)
    io_in(net, "input.num", "numeric", xs, [8])
    net.dense("B1", 8, 40, path="stages.0")
    net.pointwise("B2", "activation", 40, name="relu", path="stages.1")
    net.pointwise("B3", "dropout", 40, p=0.1, path="stages.2")
    net.dense("B4", 40, 30, act="gelu", path="stages.3")
    net.dense("B5", 30, 3, type_="head", path="stages.4", axes=("output",))
    io_out(net, "softmax", "classes", ks)
    net.chain("input.num", "B1", "B2", "B3", "B4", "B5")
    net.edge("B5", "output", "readout")
    chain = {"eligible": True, "linear_reading": True, "inputs": xs,
             "layers": [chain_layer(net, "B1", "relu", 0.1), chain_layer(net, "B4", "gelu", 0.0), chain_layer(net, "B5", None, 0.0)]}
    return net.document(family="mlp-plain", task=task("multiclass", "softmax", "classes", ks, "argmax", ["logit"]),
                        sample=sample(), inputs=inputs(xs), chain=chain)


def linear(cat):
    net, xs = Net("linear", 12, cat), names("x", 8)
    io_in(net, "input.num", "numeric", xs, [8])
    net.dense("B1", 8, 1, type_="head", path="stages.0", axes=("output",))
    io_out(net, "identity", "value", ["y"])
    net.chain("input.num", "B1")
    net.edge("B1", "output", "readout")
    chain = {"eligible": True, "linear_reading": True, "inputs": xs, "layers": [chain_layer(net, "B1", None, 0.0)]}
    return net.document(family="linear", task=task("regression", "identity", "value", ["y"], "none", ["value"]),
                        sample=sample(), inputs=inputs(xs), chain=chain)


def mlp_residual(cat):
    net, xs = Net("mlp-residual", 13, cat), names("x", 8)
    io_in(net, "input.num", "numeric", xs, [8])
    net.dense("B1", 8, 6, act="relu", path="stages.0")
    net.node("B2", "residual", children=["B2.r0.0", "B2.r0.1", "B2.r0.2", "B2.r0.3", "B2.r0.add"], path="stages.1",
             in_shape=[6], out_shape=[6], config={"repeat": 1, "projection": "auto"},
             params={"repeats": [{"inner": ["B2.r0.0", "B2.r0.1", "B2.r0.2", "B2.r0.3"], "projection": None}]})
    net.dense("B2.r0.0", 6, 6, parent="B2", path="stages.1.repeats.0.0", block="B2")
    net.norm("B2.r0.1", "batchnorm", 6, parent="B2", path="stages.1.repeats.0.1", block="B2")
    net.pointwise("B2.r0.2", "activation", 6, name="relu", parent="B2", path="stages.1.repeats.0.2", block="B2")
    net.pointwise("B2.r0.3", "dropout", 6, p=0.2, parent="B2", path="stages.1.repeats.0.3", block="B2")
    net.node("B2.r0.add", "residual_add", parent="B2", call="inner", path="stages.1", in_shape=[6], out_shape=[6], block="B2", code=False)
    net.group_count("B2")
    net.dense("B3", 6, 1, type_="head", path="stages.2", axes=("output",))
    io_out(net, "sigmoid", "score", ["yes"])
    net.chain("input.num", "B1", "B2.r0.0", "B2.r0.1", "B2.r0.2", "B2.r0.3", "B2.r0.add", "B3")
    net.edge("B1", "B2.r0.add", "skip")
    net.edge("B3", "output", "readout")
    return net.document(family="mlp-residual", task=task("binary", "sigmoid", "score", ["yes"], "threshold", ["binary_logit"],
                                                         positive="yes", threshold=0.42, scale="probability"),
                        sample=sample(), inputs=inputs(xs))


def embedding(cat):
    net, xs, ks = Net("embedding", 14, cat), names("x", 5), names("k", 3)
    nominal = [{"name": "c0", "cardinality": 4, "dim": 3}, {"name": "c1", "cardinality": 3, "dim": 2}]
    io_in(net, "input.num", "numeric", xs, [5])
    io_in(net, "input.cat", "nominal", ["c0", "c1"], [2])
    tables = []
    for c in nominal:
        t = net.rng.standard_normal((c["cardinality"], c["dim"])).astype(np.float32)
        tables.append({"column": c["name"], "cardinality": c["cardinality"], "dim": c["dim"],
                       "table": net.tensor("B1", "table", t, ["id", "dim"], f"table.{c['name']}"),
                       "row_norms": [sig9(v) for v in np.linalg.norm(t, axis=1)], "pca2": None, "unseen_row": c["cardinality"] - 1})
    n_emb = sum(c["cardinality"] * c["dim"] for c in nominal)
    net.node("B1", "embedding", call="emb", path="stages.0", in_shape=[2], out_shape=[5], config={"columns": ["c0", "c1"], "dims": {"c0": 3, "c1": 2}},
             params={"count": n_emb, "trainable": n_emb, "tables": tables})
    net.node("B2", "concat", call="concat", path="stages.1", in_shape=None, out_shape=[10], config={},
             params={"parts": [{"stream": "numeric", "width": 5}, {"stream": "B1", "width": 5}]})
    net.dense("B3", 10, 6, act="relu", path="stages.2")
    net.dense("B4", 6, 3, type_="head", path="stages.3", axes=("output",))
    io_out(net, "softmax", "classes", ks)
    net.edge("input.cat", "B1", "embed")
    net.edge("input.num", "B2")
    net.edge("B1", "B2")
    net.chain("B2", "B3", "B4")
    net.edge("B4", "output", "readout")
    chain = {"eligible": True, "linear_reading": True, "inputs": xs + ["c0·e0", "c0·e1", "c0·e2", "c1·e0", "c1·e1"],
             "layers": [chain_layer(net, "B3", "relu", 0.0), chain_layer(net, "B4", None, 0.0)]}
    return net.document(family="mlp-plain", task=task("multiclass", "softmax", "classes", ks, "argmax", ["logit"]),
                        sample=sample(), inputs=inputs(xs, nominal), chain=chain)


def autoencoder(cat):
    net, xs = Net("autoencoder", 15, cat), names("x", 8)
    io_in(net, "input.num", "numeric", xs, [8])
    net.node("B1", "sequential", children=["B1.0", "B1.1"], path="stages.0", in_shape=[8], out_shape=[4], config={}, params={"children": ["B1.0", "B1.1"]})
    net.dense("B1.0", 8, 6, act="relu", parent="B1", path="stages.0.inner.0", block="B1")
    net.dense("B1.1", 6, 4, act="relu", parent="B1", path="stages.0.inner.1", block="B1")
    net.group_count("B1")
    net.dense("B2", 4, 2, path="stages.1")
    net.node("B3", "decoder", children=["B3.l0", "B3.l1"], path="stages.2", in_shape=[2], out_shape=[6], config={"widths": [4, 6], "activation": "relu"},
             params={"children": ["B3.l0", "B3.l1"]})
    net.dense("B3.l0", 2, 4, act="relu", parent="B3", role="decoder_layer", path="stages.2.net.0", block="B3")
    net.dense("B3.l1", 4, 6, act="relu", parent="B3", role="decoder_layer", path="stages.2.net.2", block="B3")
    net.group_count("B3")
    net.dense("B4", 6, 8, type_="head", path="stages.3", axes=("output",))
    io_out(net, "reconstruction", "reconstruction", xs)
    net.chain("input.num", "B1.0", "B1.1", "B2", "B3.l0", "B3.l1", "B4")
    net.edge("B4", "output", "readout")
    return net.document(family="autoencoder", task=task("unsupervised", "reconstruction", "reconstruction", xs, "none",
                                                        ["reconstruction_error", "feature_error"]),
                        sample=sample(), inputs=inputs(xs))


def parallel(cat):
    net, xs = Net("parallel", 16, cat), names("x", 6)
    io_in(net, "input.num", "numeric", xs, [6])
    net.dense("B1", 6, 4, act="relu", path="stages.0")
    net.node("B2", "parallel", children=["B2.b0", "B2.b1", "B2.merge"], path="stages.1", in_shape=[4], out_shape=[6],
             config={"merge": "concat"}, params={"branches": ["B2.b0", "B2.b1"], "merge": "concat"})
    net.dense("B2.b0", 4, 3, act="tanh", parent="B2", path="stages.1.branches.0", block="B2")
    net.node("B2.b1", "sequential", parent="B2", children=["B2.b1.0", "B2.b1.1"], path="stages.1.branches.1", in_shape=[4], out_shape=[3],
             params={"children": ["B2.b1.0", "B2.b1.1"]}, block="B2")
    net.dense("B2.b1.0", 4, 3, act="relu", parent="B2.b1", path="stages.1.branches.1.inner.0", block="B2")
    net.norm("B2.b1.1", "layernorm", 3, parent="B2.b1", path="stages.1.branches.1.inner.1", block="B2")
    net.group_count("B2.b1")
    net.node("B2.merge", "parallel_merge", parent="B2", call="inner", path="stages.1", in_shape=[6], out_shape=[6], config={"merge": "concat"},
             params={"merge": "concat"}, block="B2", code=False)
    net.group_count("B2")
    net.node("B3", "custom", path="stages.2", in_shape=[6], out_shape=[6], config={"code_ref": "custom/b3.py", "out_shape": [6]},
             params={"count": 6, "trainable": 6, "class": "Gate", "parameters": [{"name": "g", "shape": [6], "l2": 1.5}],
                     "declared_out_shape": [6], "verified": False})
    net.dense("B4", 6, 1, type_="head", path="stages.3", axes=("output",))
    io_out(net, "identity", "value", ["y"])
    net.chain("input.num", "B1")
    net.edge("B1", "B2.b0", "branch")
    net.edge("B1", "B2.b1.0", "branch")
    net.chain("B2.b1.0", "B2.b1.1")
    net.edge("B2.b0", "B2.merge", "merge")
    net.edge("B2.b1.1", "B2.merge", "merge")
    net.chain("B2.merge", "B3", "B4")
    net.edge("B4", "output", "readout")
    return net.document(family="mlp-plain", task=task("regression", "identity", "value", ["y"], "none", ["value"]),
                        sample=sample(), inputs=inputs(xs))


def lstm_window(cat):
    net, xs, ks, L, H = Net("lstm-window", 17, cat), names("x", 5), names("k", 3), 6, 8
    steps = [f"t−{L - 1 - i}" if i < L - 1 else "t" for i in range(L)]
    io_in(net, "input.num", "numeric", xs, [L, 5], axes=("step", "feature"))
    w_ih = net.tensor("B1", "weight", net.rng.standard_normal((4 * H, 5)).astype(np.float32), ["gate_out", "in"], "l0.w_ih")
    w_hh = net.tensor("B1", "weight", net.rng.standard_normal((4 * H, H)).astype(np.float32), ["gate_out", "hidden"], "l0.w_hh")
    b_ih = net.tensor("B1", "bias", net.rng.standard_normal(4 * H).astype(np.float32), ["gate_out"], "l0.b_ih")
    b_hh = net.tensor("B1", "bias", net.rng.standard_normal(4 * H).astype(np.float32), ["gate_out"], "l0.b_hh")
    count = 4 * H * (5 + H + 2)
    net.node("B1", "lstm", path="stages.0", in_shape=[L, 5], out_shape=[H], config={"hidden": H, "layers": 1, "bidirectional": False, "return": "last"},
             params={"count": count, "trainable": count, "cell": "lstm", "hidden": H, "layers": 1, "directions": 1, "gates": ["i", "f", "g", "o"],
                     "return": "last", "per_layer": [{"layer": 0, "direction": "forward", "w_ih": w_ih, "w_hh": w_hh, "b_ih": b_ih, "b_hh": b_hh,
                                                      "gate_norms": {g: {"ih": 1.0, "hh": 1.0, "b": 0.5} for g in "ifgo"},
                                                      "input_columns": {g: [1.0] * 5 for g in "ifgo"}, "rho_hh": {g: 0.9 for g in "ifgo"}}]})
    net.pointwise("B2", "dropout", H, p=0.2, path="stages.1")
    net.dense("B3", H, 3, type_="head", path="stages.2", axes=("output",))
    io_out(net, "softmax", "classes", ks)
    net.chain("input.num", "B1", "B2", "B3")
    net.edge("B3", "output", "readout")
    return net.document(family="lstm", task=task("sequence-classification", "softmax", "classes", ks, "argmax", ["logit"], base="multiclass"),
                        sample=sample("window", length=L, stride=1), inputs=inputs(xs, step=steps))


def gcn_node(cat):
    net, xs, ks = Net("gcn-node", 18, cat), names("x", 6), names("k", 3)
    io_in(net, "input.num", "numeric", xs, [6], per="node")
    io_in(net, "input.edges", "edges", [], [2], axes=("unit",), per="node")
    net.node("B1", "edge_dropout", call="gdrop", path="stages.0", in_shape=[2], out_shape=[2], per="node", config={"p": 0.1},
             params={"p": 0.1, "active_at_inference": False})
    def conv(nid, path, n_in, n_out):
        w = net.tensor(nid, "weight", (net.rng.standard_normal((n_out, n_in)) / np.sqrt(n_in)).astype(np.float32), ["out", "in"])
        b = net.tensor(nid, "bias", (net.rng.standard_normal(n_out) * 0.1).astype(np.float32), ["out"])
        c = n_in * n_out + n_out
        net.node(nid, "gcn_conv", call="graph", path=path, in_shape=[n_in], out_shape=[n_out], per="node", config={"out": n_out},
                 params={"count": c, "trainable": c, "weight": w, "bias": b, "normalize": True, "self_loops": True})
    conv("B2", "stages.1", 6, 8)
    net.pointwise("B3", "activation", 8, name="relu", path="stages.2", per="node")
    net.pointwise("B4", "dropout", 8, p=0.2, path="stages.3", per="node")
    conv("B5", "stages.4", 8, 8)
    net.dense("B6", 8, 3, type_="head", path="stages.5", axes=("output",), per="node")
    io_out(net, "softmax", "classes", ks, per="node")
    net.edge("input.edges", "B1", "edges")
    net.chain("input.num", "B2", "B3", "B4", "B5", "B6")
    net.edge("B1", "B2", "edges")
    net.edge("B1", "B5", "edges")
    net.edge("B6", "output", "readout")
    return net.document(family="gcn", task=task("multiclass", "softmax", "classes", ks, "argmax", ["logit"]),
                        sample=sample("graph", level="node", directed=False, self_loops=True, protocol="inductive", batching="full"),
                        inputs=inputs(xs, edges={"weighted": False}))


def ordinal(cat):
    net, xs, ks = Net("ordinal", 19, cat), names("x", 8), names("k", 4)
    io_in(net, "input.num", "numeric", xs, [8])
    net.dense("B1", 8, 6, act="relu", path="stages.0")
    w = (net.rng.standard_normal((1, 6)) / np.sqrt(6)).astype(np.float32)
    net.tensor("B2", "bias", np.array([0.9], dtype=np.float32), ["threshold"], "first")
    net.tensor("B2", "bias", np.array([0.4, 0.7], dtype=np.float32), ["gap"], "gaps")
    thresholds = [sig9(0.9), sig9(0.9 - 0.4), sig9(0.9 - 0.4 - 0.7)]
    net.node("B2", "head", path="stages.1", in_shape=[6], out_shape=[3], axes=("output",), config={"out": 3, "activation": "none", "bias": True},
             params={"count": 9, "trainable": 9, "weight": net.tensor("B2", "weight", w, ["out", "in"]), "bias": None, "activation": "none",
                     "row_norms": [sig9(np.linalg.norm(w))], "col_norms": [sig9(abs(v)) for v in w[0]], "thresholds": thresholds})
    steps = [f">{k}" for k in ks[:-1]]
    io_out(net, "cumulative", "thresholds", steps)
    net.chain("input.num", "B1", "B2")
    net.edge("B2", "output", "readout")
    return net.document(family="mlp-plain", task=task("ordinal", "cumulative", "thresholds", steps, "cumulative_count", ["expected_rank", "threshold_logit"]),
                        sample=sample(), inputs=inputs(xs))


def svdd(cat):
    net, xs = Net("svdd", 20, cat), names("x", 8)
    io_in(net, "input.num", "numeric", xs, [8])
    net.dense("B1", 8, 6, path="stages.0")
    net.norm("B2", "batchnorm", 6, path="stages.1")
    net.pointwise("B3", "activation", 6, name="relu", path="stages.2")
    wid, _, count, w = net.weights("B4", 6, 4, bias=False)
    net.node("B4", "head", path="stages.3", in_shape=[6], out_shape=[4], axes=("output",), config={"out": 4, "activation": "none", "bias": False},
             params={"count": count, "trainable": count, "weight": wid, "bias": None, "activation": "none", "thresholds": None,
                     "row_norms": [sig9(v) for v in np.linalg.norm(w, axis=1)], "col_norms": [sig9(v) for v in np.linalg.norm(w, axis=0)]})
    net.tensor("B4", "buffer", (net.rng.standard_normal(4) * 0.3).astype(np.float32), ["unit"], "centre")
    zs = names("z", 4)
    io_out(net, "distance", "embedding", zs)
    net.chain("input.num", "B1", "B2", "B3", "B4")
    net.edge("B4", "output", "readout")
    return net.document(family="deep-svdd", task=task("anomaly", "distance", "embedding", zs, "threshold", ["svdd_distance"],
                                                      threshold=0.8, scale="score"),
                        sample=sample(), inputs=inputs(xs))


def forecast(cat):
    net, xs, L, H, qs = Net("forecast", 21, cat), names("x", 3), 6, 3, [0.1, 0.5, 0.9]
    steps = [f"t−{L - 1 - i}" if i < L - 1 else "t" for i in range(L)]
    io_in(net, "input.num", "numeric", xs, [L, 3], axes=("step", "feature"))
    net.node("B1", "flatten", path="stages.0", in_shape=[L, 3], out_shape=[L * 3], config={})
    net.dense("B2", L * 3, 8, act="relu", path="stages.1")
    out = [f"t+{h}/q{q:g}" for h in range(1, H + 1) for q in qs]
    net.dense("B3", 8, H * len(qs), type_="head", path="stages.2", axes=("output",))
    io_out(net, "horizon", "horizon", out)
    net.chain("input.num", "B1", "B2", "B3")
    net.edge("B3", "output", "readout")
    return net.document(family="mlp-plain", task=task("forecast", "horizon", "horizon", out, "none", ["forecast_step"], quantiles=qs),
                        sample=sample("window", length=L, stride=1, horizon=H, autoregressive=False), inputs=inputs(xs, step=steps))


def recon_nominal(cat):
    net, xs = Net("recon-nominal", 22, cat), names("x", 4)
    nominal = [{"name": "c0", "cardinality": 3, "dim": 2}]
    io_in(net, "input.num", "numeric", xs, [4])
    io_in(net, "input.cat", "nominal", ["c0"], [1])
    t = net.rng.standard_normal((3, 2)).astype(np.float32)
    tables = [{"column": "c0", "cardinality": 3, "dim": 2, "table": net.tensor("B1", "table", t, ["id", "dim"], "table.c0"),
               "row_norms": [sig9(v) for v in np.linalg.norm(t, axis=1)], "pca2": None, "unseen_row": 2}]
    net.node("B1", "embedding", call="emb", path="stages.0", in_shape=[1], out_shape=[2], config={"columns": ["c0"], "dims": {"c0": 2}},
             params={"count": 6, "trainable": 6, "tables": tables})
    net.node("B2", "concat", call="concat", path="stages.1", in_shape=None, out_shape=[6], config={},
             params={"parts": [{"stream": "numeric", "width": 4}, {"stream": "nominal", "width": 2}]})
    net.dense("B3", 6, 4, act="relu", path="stages.2")
    net.dense("B4", 4, 7, type_="head", path="stages.3", axes=("output",))
    rebuilt = xs + ["c0"]
    io_out(net, "reconstruction", "reconstruction", rebuilt)
    net.edge("input.cat", "B1", "embed")
    net.edge("input.num", "B2")
    net.edge("B1", "B2")
    net.chain("B2", "B3", "B4")
    net.edge("B4", "output", "readout")
    return net.document(family="autoencoder", task=task("unsupervised", "reconstruction", "reconstruction", rebuilt, "none",
                                                        ["reconstruction_error", "feature_error"]),
                        sample=sample(), inputs=inputs(xs, nominal))


def residual_deep(cat):
    net, xs = Net("residual-deep", 23, cat), names("x", 8)
    io_in(net, "input.num", "numeric", xs, [8])
    net.dense("B1", 8, 5, act="relu", path="stages.0")
    repeats, kids = [], []
    for r in range(3):
        inner, proj = [f"B2.r{r}.0", f"B2.r{r}.1"], ("B2.r0.proj" if r == 0 else None)
        repeats.append({"inner": inner, "projection": proj})
        kids += inner + ([proj] if proj else []) + [f"B2.r{r}.add"]
    net.node("B2", "residual", children=kids, path="stages.1", in_shape=[5], out_shape=[6],
             config={"repeat": 3, "projection": "auto"}, params={"repeats": repeats})
    for r in range(3):
        net.dense(f"B2.r{r}.0", 5 if r == 0 else 6, 6, parent="B2", path=f"stages.1.repeats.{r}.0", block="B2")
        net.pointwise(f"B2.r{r}.1", "activation", 6, name="relu", parent="B2", path=f"stages.1.repeats.{r}.1", block="B2")
        if r == 0:
            net.dense("B2.r0.proj", 5, 6, parent="B2", role="projection", path="stages.1.repeats.0.proj", block="B2")
        net.node(f"B2.r{r}.add", "residual_add", parent="B2", call="inner", path="stages.1", in_shape=[6], out_shape=[6], block="B2", code=False)
    net.group_count("B2")
    net.norm("B3", "layernorm", 6, path="stages.2")
    net.pointwise("B4", "dropout", 6, p=0.3, path="stages.3")
    net.dense("B5", 6, 1, type_="head", path="stages.4", axes=("output",))
    io_out(net, "identity", "value", ["y"])
    net.chain("input.num", "B1", "B2.r0.0", "B2.r0.1", "B2.r0.add", "B2.r1.0", "B2.r1.1", "B2.r1.add",
              "B2.r2.0", "B2.r2.1", "B2.r2.add", "B3", "B4", "B5")
    net.chain("B1", "B2.r0.proj")
    net.edge("B2.r0.proj", "B2.r0.add", "skip")
    net.edge("B2.r0.add", "B2.r1.add", "skip")
    net.edge("B2.r1.add", "B2.r2.add", "skip")
    net.edge("B5", "output", "readout")
    return net.document(family="mlp-residual", task=task("regression", "identity", "value", ["y"], "none", ["value"]),
                        sample=sample(), inputs=inputs(xs))


FIXTURES = {"mlp-plain": mlp_plain, "linear": linear, "mlp-residual": mlp_residual, "embedding": embedding,
            "autoencoder": autoencoder, "parallel": parallel, "lstm-window": lstm_window, "gcn-node": gcn_node,
            "ordinal": ordinal, "svdd": svdd, "forecast": forecast, "recon-nominal": recon_nominal,
            "residual-deep": residual_deep}


def check() -> int:
    import jsonschema
    schema = json.loads((REPO / "vega-forge" / "references" / "schemas" / "network-view-2.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    bad = 0
    for name in FIXTURES:
        doc = json.loads((HERE / f"{name}.json").read_text())
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        for e in errors[:5]:
            print(f"FAIL: {name}.json {'/'.join(map(str, e.path))}: {e.message[:200]}")
        bad += bool(errors)
        if not errors:
            print(f"OK: {name}.json validates vega/network-view@2 ({len(doc['nodes'])} nodes, {len(doc['tensors'])} tensors)")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--check", action="store_true", help="validate the fixtures against the schema")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    cat = json.loads(CATALOG.read_text())
    for name, make in FIXTURES.items():
        (HERE / f"{name}.json").write_text(json.dumps(make(cat), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {name}.json")
    return check()


if __name__ == "__main__":
    sys.exit(main())
