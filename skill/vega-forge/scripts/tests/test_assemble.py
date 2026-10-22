#!/usr/bin/env python3
"""
test_assemble.py — `templates/assemble.py`, the serving side's `SampleAssembler`, proved on the
*rendered* module (the template placed and filled by `render_templates.py`, beside the rendered
`domain.py`, as the emitter writes them into every package).

    python3 test_assemble.py [-v] [-k <substring>]
    python3 -m pytest test_assemble.py

What is proved (build plan WP B10, R-A3, R-A5–R-A7):

  windows  the assembled windows equal the ones the domain plan cuts for the evaluation (the
           previous part as context, the dense scoring plan) row for row, the autoregressive
           channel and `x_cat (B, L, C)` included; every request row with L-1 predecessors of
           its series in history + rows is answered, a forecast's last rows too; a row with
           too little history is unscored `insufficient_context`; `SampleIndex` rows are the
           request order and spans the first and last position read (history negative); the
           history is read under `context: none` too, an empty one is none, and without
           `order_by` a series is in request order.
  nodes    the assembled node batch is the sealed test view's batch, tensor for tensor; a toy
           GNN of the real graph blocks answers each query alone identically to all together,
           and in any request order (no query sends a message).
  graphs   the request's graphs grouped by `graph_id` in first-seen order, `graph_index`,
           `num_graphs`, `SampleIndex.graphs`; the batch equals the evaluator's graph-level
           view; a toy GIN answers each graph alone as all together (to float reassociation).
  refusals every misuse names what is wrong (a missing column, a raw value in a feature
           column, a stray history, a missing or mismatched reference graph, a transductive
           protocol, a cross-graph edge, a frame that is not a DataFrame, ...).
  module   C11–C13 on the rendered file; C4 (imports along the arrows: only `domain`) and C7.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import importlib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
PACKAGE = "assemble_pkg"
VERBOSE = "-v" in sys.argv
for _p in (str(SCRIPTS), str(HERE), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import blocks as B  # noqa: E402
import conformance  # noqa: E402
import render_templates as rt  # noqa: E402
from vega_core import Level  # noqa: E402

_WORK = tempfile.TemporaryDirectory(prefix="assemble-")
PIPELINE = rt.reference_pipeline()
PIPELINE["system"]["package"] = PACKAGE
RENDERED = rt.render(BACKEND / "templates", _WORK.name, PIPELINE, rt.reference_spec("row"),
                     only=["assemble.py"])
ROOT = Path(RENDERED["root"])
sys.path.insert(0, _WORK.name)
D = importlib.import_module(f"{PACKAGE}.domain")
A = importlib.import_module(f"{PACKAGE}.model.assemble")


# --------------------------------------------------------------------------- specs and data

def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric", "origin": "S1"}


def _zone(cardinality):
    return {"name": "zone", "dtype": "int64", "role": "nominal", "cardinality": cardinality,
            "unseen_category_id": cardinality - 1, "origin": "S2"}


def _spec(shape, features=None, sample=None, task=None):
    s = rt.reference_spec(shape)
    if features is not None:
        s["data_contract"]["features"] = features
        s["data_contract"]["input_dim"] = len(features)
    s["data_contract"]["sample"].update(sample or {})
    s["task"] = task or s["task"]
    return D.NetSpec.from_dict(s)


def window_spec(length=4, horizon=2, autoregressive=True, kind="forecast", **sample):
    sample.update(length=length, horizon=horizon, autoregressive=autoregressive)
    task = None
    if kind != "forecast":
        sample.update(horizon=None, autoregressive=False)
        task = {"kind": kind, "base_kind": "binary", "label": "target", "classes": [0, 1]}
    return _spec("window", [_num("x0"), _num("x1"), _zone(5)], sample, task)


def _series_frame(seed, lengths, start=0):
    """Rows of several series (`series`, `ts` from `start`), shuffled, with an out-of-range
    and a missing nominal id among them."""
    rng = np.random.default_rng(seed)
    keys = np.concatenate([[s] * n for s, n in lengths.items()])
    ts = np.concatenate([np.arange(n) + start for n in lengths.values()])
    zone = rng.integers(0, 5, size=len(keys)).astype(float)
    zone[:2] = (9, np.nan)
    frame = pd.DataFrame({"x0": rng.normal(size=len(keys)), "x1": rng.normal(size=len(keys)),
                          "zone": zone, "series": keys, "ts": ts,
                          "target": rng.normal(size=len(keys))})
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def _series_rows(frame, positions):
    return D.SeriesRows(raw_pos=positions, series=frame["series"].to_numpy(dtype=object),
                        order=frame["ts"].to_numpy())


def _unseen(zone):
    z = np.nan_to_num(np.asarray(zone, dtype=float), nan=-1)
    return np.where((z < 0) | (z >= 5), 4, z).astype(np.int64)


# --------------------------------------------------------------------------- windows

def _evaluated(spec, rows, history):
    """What the evaluation cuts for `rows` as the part and `history` as its previous part: the
    domain plan (dense, the spec's own horizon) over the flat [history ; rows] arrays."""
    window = D.WindowSpec.from_spec(spec)
    at_h = np.arange(-len(history), 0)
    layout = window.layout(_series_rows(rows, np.arange(len(rows))), _series_rows(history, at_h))
    plan = window.plan(layout.blocks, dense=True)
    cols = ["x0", "x1", "target"] if window.autoregressive else ["x0", "x1"]
    flat = layout.take(history[cols].to_numpy(np.float32), rows[cols].to_numpy(np.float32))
    cat = layout.take(_unseen(history["zone"])[:, None], _unseen(rows["zone"])[:, None])
    held = D.WindowedPart(x_num=torch.from_numpy(flat), x_cat=torch.from_numpy(cat), y=None,
                          anchors=torch.as_tensor(plan.anchors),
                          targets=torch.as_tensor(plan.targets), row_ids=plan.row_ids,
                          length=plan.length)
    return plan, held.gather(torch.arange(len(plan)))


def _predecessors(rows, history):
    """Per request row: how many rows of its series precede it in history + rows."""
    before = rows.groupby("series")["ts"].rank(method="first").to_numpy() - 1
    count = history["series"].value_counts()
    return before + rows["series"].map(count).fillna(0).to_numpy()


def test_window_batches_equal_the_domain_plan():
    spec = window_spec()
    rows = _series_frame(1, {"s1": 9, "s2": 6, "s3": 11}, start=100)
    history = _series_frame(2, {"s1": 7, "s2": 5, "s3": 12, "s9": 3})
    got = A.SampleAssembler(spec).assemble(rows, history)
    plan, want = _evaluated(spec, rows, history)
    assert got.batch["x_num"].shape[1:] == (4, 3) and got.batch["x_cat"].shape[1:] == (4, 1)
    assert set(got.batch) == {"x_num", "x_cat"}
    for w, row in enumerate(plan.row_ids.tolist()):
        k = int(got.sample_of_row[row])
        assert k >= 0 and got.index.rows[k] == row
        assert torch.equal(got.batch["x_num"][k], want["x_num"][w]), row
        assert torch.equal(got.batch["x_cat"][k], want["x_cat"][w]), row
    answered = set(got.index.rows)
    beyond = {u.raw_pos for u in plan.unscored if u.reason is D.UnscoredReason.HORIZON_BEYOND_PART}
    assert answered == set(plan.row_ids.tolist()) | beyond and len(beyond) == 2 * 3
    assert answered == set(np.flatnonzero(_predecessors(rows, history) >= 3).tolist())
    assert got.unscored == () and len(got) == len(rows)


def test_too_little_history_is_unscored_insufficient_context():
    spec = window_spec(length=4)
    rows = _series_frame(3, {"s1": 6, "s2": 5, "s3": 4}, start=50)
    history = _series_frame(4, {"s1": 8, "s2": 1})             # s2 one row short, s3 none
    for past in (history, None):
        got = A.SampleAssembler(spec).assemble(rows, past)
        short = _predecessors(rows, past if past is not None else history.iloc[0:0]) < 3
        assert [u.raw_pos for u in got.unscored] == np.flatnonzero(short).tolist()
        assert {u.reason for u in got.unscored} == {D.UnscoredReason.INSUFFICIENT_CONTEXT}
        assert np.array_equal(got.sample_of_row < 0, short)
        assert len(got) == int((~short).sum()) == got.batch["x_num"].shape[0]
    alone = A.SampleAssembler(spec).assemble(rows, None)
    assert len(alone.unscored) == 3 * 3                        # the first L-1 of each series


def test_window_index_is_request_order_with_spans():
    spec = window_spec(length=3, horizon=1)
    rows = _series_frame(5, {"a": 5, "b": 4}, start=10)
    history = _series_frame(6, {"a": 1, "b": 4})
    got = A.SampleAssembler(spec).assemble(rows, history)
    assert list(got.index.rows) == sorted(got.index.rows)
    frame = pd.concat([history.assign(at=np.arange(-len(history), 0)),
                       rows.assign(at=np.arange(len(rows)))]).sort_values(["series", "ts"])
    for k, (first, last) in enumerate(got.index.spans):
        assert last == got.index.rows[k] and got.index.sample_of(last) == k
        series = frame[frame["series"] == rows.loc[last, "series"]]["at"].tolist()
        assert first == series[series.index(last) - 2]
    assert any(first < 0 for first, _ in got.index.spans)       # read from the history
    assert got.index.shape is D.SampleShape.WINDOW


def test_autoregressive_channel_and_its_refusals():
    rows = _series_frame(7, {"s1": 6}, start=10)
    history = _series_frame(8, {"s1": 4})
    got = A.SampleAssembler(window_spec(length=3)).assemble(rows, history)
    k = got.index.sample_of(int(rows["ts"].idxmax()))
    last = rows.sort_values("ts")["target"].to_numpy(np.float32)[-3:]
    assert np.array_equal(got.batch["x_num"][k, :, -1].numpy(), last)    # past, up to the anchor
    plain = A.SampleAssembler(window_spec(length=3, autoregressive=False)).assemble(rows, history)
    assert plain.batch["x_num"].shape[-1] == 2
    seq = A.SampleAssembler(window_spec(length=3, kind="sequence-classification"))
    assert seq.assemble(rows.drop(columns="target"), history).batch["x_num"].shape[-1] == 2
    assembler = A.SampleAssembler(window_spec(length=3))
    _raises(ValueError, "'target'", lambda: assembler.assemble(rows.drop(columns="target"),
                                                               history))
    _raises(ValueError, "history rows lack the target", lambda: assembler.assemble(
        rows, history.drop(columns="target")))
    gap = rows.assign(target=rows["target"].where(rows.index != 2))
    _raises(ValueError, "missing at rows position 2", lambda: assembler.assemble(gap, history))
    _raises(ValueError, "['x1']", lambda: assembler.assemble(rows.drop(columns="x1"), history))
    _raises(ValueError, "no graph", lambda: assembler.assemble(rows, history, pd.DataFrame()))
    _raises(ValueError, "series column is missing", lambda: assembler.assemble(
        rows.assign(series=rows["series"].where(rows.index != 1)), history))


def test_an_empty_history_is_no_history():
    spec = window_spec(length=3)
    rows = _series_frame(11, {"s1": 6, "s2": 4}, start=30)
    none = A.SampleAssembler(spec).assemble(rows)
    for empty in (rows.iloc[0:0], pd.DataFrame()):
        got = A.SampleAssembler(spec).assemble(rows, empty)
        assert got.index == none.index and got.unscored == none.unscored
        assert all(torch.equal(got.batch[k], none.batch[k]) for k in none.batch)


def test_the_history_is_read_whatever_the_context_and_order_by_absent_is_request_order():
    """`context: none` speaks of training's parts: a request's history is its series' past
    either way (MANIFEST.sample.rule). Without `order_by` a series is in request order."""
    rows = _series_frame(9, {"s1": 7, "s2": 5}, start=20)
    history = _series_frame(10, {"s1": 3, "s2": 1})
    base = A.SampleAssembler(window_spec(length=3)).assemble(rows, history)
    blind = A.SampleAssembler(window_spec(length=3, context="none")).assemble(rows, history)
    assert blind.index == base.index and blind.unscored == base.unscored
    assert all(torch.equal(blind.batch[k], base.batch[k]) for k in base.batch)
    in_order = rows.sort_values("ts", kind="stable").reset_index(drop=True)
    past = history.sort_values("ts", kind="stable").reset_index(drop=True)
    stored = A.SampleAssembler(window_spec(length=3, order_by=None)).assemble(in_order, past)
    sorted_ = A.SampleAssembler(window_spec(length=3)).assemble(in_order, past)
    assert stored.index == sorted_.index
    assert all(torch.equal(stored.batch[k], sorted_.batch[k]) for k in sorted_.batch)
    shuffled = A.SampleAssembler(window_spec(length=3, order_by=None)).assemble(rows, history)
    assert not torch.equal(shuffled.batch["x_num"], base.batch["x_num"])   # no silent sort


# --------------------------------------------------------------------------- nodes

SIZES = (40, 15, 12)            # train, valid, test


def node_spec(protocol="inductive"):
    sample = {"protocol": protocol}
    if protocol == "transductive":
        sample["exposure"] = {"kind": "features", "rows": "test", "source": "user:t1"}
    return _spec("graph", [_num("x0"), _num("x1"), _num("x2"), _zone(4)], sample)


def _node_world(seed=0, weighted=True):
    """Global features over [train | valid | test] and an edges part touching every part."""
    rng = np.random.default_rng(seed)
    total = sum(SIZES)
    pairs = rng.integers(0, total, size=(70, 2))
    pairs = np.concatenate([pairs, pairs[:, ::-1], [[3, 3], [60, 60], [44, 58]]])
    pairs = pairs[~np.isin(pairs, [total - 1]).any(axis=1)]    # the last test node: no edge
    weight = rng.uniform(0.2, 1.0, size=len(pairs)) if weighted else None
    edges = D.GraphEdges(src=pairs[:, 0], dst=pairs[:, 1], weight=weight)
    x = pd.DataFrame({"x0": rng.normal(size=total), "x1": rng.normal(size=total),
                      "x2": rng.normal(size=total), "zone": rng.integers(0, 4, size=total)})
    return x, edges


def _served(x, edges, spec):
    """The bundle's view of the same world: the reference graph (train ∪ valid, rows stored in
    shuffled `ref` order), the test rows as a request and their `edges_for` links."""
    n_ref = SIZES[0] + SIZES[1]
    order = np.random.default_rng(1).permutation(n_ref)
    nodes = x.iloc[:n_ref].assign(ref=np.arange(n_ref)).iloc[order]
    inside = (edges.src < n_ref) & (edges.dst < n_ref)
    weight = edges.weight[inside] if edges.weight is not None else np.full(inside.sum(), np.nan)
    ref_edges = pd.DataFrame({"src_ref": edges.src[inside], "dst_ref": edges.dst[inside],
                              "weight": weight})
    reference = D.GraphReference.from_frames(nodes, ref_edges, spec.feature_contract)
    fwd = (edges.src < n_ref) & (edges.dst >= n_ref)
    bwd = (edges.dst < n_ref) & (edges.src >= n_ref)
    k = np.flatnonzero(fwd | bwd)
    links = pd.DataFrame({"query_pos": np.where(fwd, edges.dst, edges.src)[k] - n_ref,
                          "ref_pos": np.where(fwd, edges.src, edges.dst)[k],
                          "weight": (edges.weight[k] if edges.weight is not None
                                     else np.full(len(k), np.nan))})
    return reference, x.iloc[n_ref:].reset_index(drop=True), links


def test_node_batch_is_the_sealed_test_view():
    spec = node_spec()
    for weighted in (True, False):
        x, edges = _node_world(weighted=weighted)
        reference, rows, links = _served(x, edges, spec)
        got = A.SampleAssembler(spec, reference).assemble(rows, None, links)
        view = D.GraphSpec().view(SIZES, edges, "test")
        feats = x.iloc[view.node_ids]
        want = D.GraphPart(
            x_num=torch.tensor(feats[["x0", "x1", "x2"]].to_numpy(np.float32)),
            x_cat=torch.tensor(feats[["zone"]].to_numpy(np.int64)),
            edge_index=torch.from_numpy(view.edge_index), target_index=torch.from_numpy(
                view.target_index), y=None, row_ids=view.node_ids[view.target_index],
            edge_weight=None if view.edge_weight is None else torch.from_numpy(view.edge_weight)
        ).as_batch()
        assert set(got.batch) == set(want), (set(got.batch), set(want))
        for key in want:
            assert torch.equal(got.batch[key], want[key]), key
        assert got.index.rows == tuple(range(SIZES[2])) and got.unscored == ()
        assert got.sample_of_row.tolist() == list(range(SIZES[2]))


class _ToyGNN(nn.Module):
    """The real graph blocks (gcn, sage, gat) over [x_num | embedded zone], gathered at the
    targets after the head (R-A2)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(4, 2)
        self.c1 = B.GCNConv(5, 8)
        self.c2 = B.SAGEConv(8, B.SageConfig(out=8))
        self.c3 = B.GATConv(8, B.GatConfig(out=4, heads=2))
        self.head = nn.Linear(8, 3)

    def forward(self, batch):
        e, w = batch["edge_index"], batch.get("edge_weight")
        h = torch.cat([batch["x_num"], self.emb(batch["x_cat"]).flatten(1)], dim=1)
        h = torch.relu(self.c2(torch.relu(self.c1(h, e, w)), e, w))
        return self.head(self.c3(h, e, w)).index_select(0, batch["target_index"])


def _same(a, b):
    """Equal up to float reassociation. A graph-level batch of one small graph runs its dense
    layers over a handful of rows, where the CPU kernels sum in another order than over the
    whole request (last-bit differences, measured below 1e-7); a message leaking between graphs
    moves an answer by orders more. The node level needs no tolerance: see below."""
    return torch.allclose(a, b, rtol=0.0, atol=1e-6)


def _into(batch, node):
    """The messages into `node`: (sources, weights), in edge order."""
    mask = batch["edge_index"][1] == node
    return batch["edge_index"][0][mask].tolist(), batch["edge_weight"][mask].tolist()


def _alone(rows, links, q):
    mine = links[links["query_pos"] == q].assign(query_pos=0)
    return rows.iloc[[q]], mine


def test_node_answers_do_not_depend_on_the_other_rows():
    """Each query alone, all together and in another order give identical outputs (not merely
    close): every batch holds the whole reference graph first, so the dense layers run over at
    least R rows in every case, and a query node's messages are the same edges in the same
    order (`_into`) — the inference-graph rule makes the answer a function of the row alone."""
    spec = node_spec()
    x, edges = _node_world(seed=3)
    reference, rows, links = _served(x, edges, spec)
    assembler, model = A.SampleAssembler(spec, reference), _ToyGNN().eval()
    whole, n_ref = assembler.assemble(rows, None, links).batch, reference.n_nodes
    assert int(whole["edge_index"][0].max()) < n_ref            # no query sends a message
    with torch.no_grad():
        together = model(whole)
        for q in range(len(rows)):
            one_row, one_link = _alone(rows, links, q)
            one = assembler.assemble(one_row, None, one_link)
            assert _into(one.batch, n_ref) == _into(whole, n_ref + q), q
            assert torch.equal(model(one.batch)[0], together[q]), q        # identical
        perm = np.random.default_rng(2).permutation(len(rows))
        inverse = np.argsort(perm)
        moved = links.assign(query_pos=inverse[links["query_pos"].to_numpy()])
        shuffled = model(assembler.assemble(rows.iloc[perm], None, moved).batch)
        assert torch.equal(shuffled, together[torch.as_tensor(perm)])
    assert len(links[links["query_pos"] == SIZES[2] - 1]) == 0          # answered, no neighbour


def test_node_level_refusals():
    spec = node_spec()
    x, edges = _node_world()
    reference, rows, links = _served(x, edges, spec)
    assembler = A.SampleAssembler(spec, reference)
    _raises(ValueError, "reference graph", lambda: A.SampleAssembler(spec))
    _raises(ValueError, "inductive graphs only", lambda: A.SampleAssembler(
        node_spec("transductive"), reference))
    thin = D.GraphReference(x_num=reference.x_num[:, :2], x_cat=reference.x_cat,
                            edge_index=reference.edge_index, edge_weight=reference.edge_weight)
    _raises(ValueError, "2 numeric and 1 nominal", lambda: A.SampleAssembler(spec, thin))
    _raises(ValueError, "no history", lambda: assembler.assemble(rows, rows, links))
    _raises(ValueError, "disagree on weights", lambda: assembler.assemble(
        rows, None, links.assign(weight=np.nan)))
    partial = links.assign(weight=links["weight"].where(links.index != 0))
    _raises(ValueError, "no weight at edge 0", lambda: assembler.assemble(rows, None, partial))
    _raises(ValueError, "leaves the", lambda: assembler.assemble(rows.iloc[:2], None, links))
    _raises(ValueError, "query_pos", lambda: assembler.assemble(rows, None, links.drop(
        columns="query_pos")))
    _raises(TypeError, "NetSpec", lambda: A.SampleAssembler(spec.to_dict(), reference))
    _raises(ValueError, "no graph", lambda: A.SampleAssembler(window_spec(), reference))


# --------------------------------------------------------------------------- whole graphs

def graph_spec():
    return _spec("graph-level", [_num("x0"), _num("x1"), _num("x2"), _zone(4)])


def _graphs(seed=0):
    """Five graphs (a cycle each, weighted), the rows of graph g contiguous."""
    rng = np.random.default_rng(seed)
    sizes = {"g1": 4, "g2": 6, "g3": 3, "g4": 5, "g5": 4}
    ids = np.concatenate([[g] * n for g, n in sizes.items()])
    x = pd.DataFrame({"x0": rng.normal(size=len(ids)), "x1": rng.normal(size=len(ids)),
                      "x2": rng.normal(size=len(ids)), "zone": rng.integers(0, 4, len(ids)),
                      "molecule": ids})
    src, dst, lo = [], [], 0
    for n in sizes.values():
        ring = np.arange(n) + lo
        src += list(ring) + list(np.roll(ring, 1))
        dst += list(np.roll(ring, 1)) + list(ring)
        lo += n
    return x, np.array([src, dst]), rng.uniform(0.5, 1.0, len(src))


def _within(edge_index, weight, keep):
    """edges_within's frame for the rows `keep` (a boolean mask over all rows)."""
    local = np.cumsum(keep) - 1
    inside = keep[edge_index[0]] & keep[edge_index[1]]
    return pd.DataFrame({"src_pos": local[edge_index[0][inside]],
                         "dst_pos": local[edge_index[1][inside]], "weight": weight[inside]})


def test_graph_batch_groups_the_request_by_graph_id():
    x, edge_index, weight = _graphs()
    keep = x["molecule"].isin(["g2", "g3", "g4"]).to_numpy()
    rows = x[keep].reset_index(drop=True)
    got = A.SampleAssembler(graph_spec()).assemble(rows, None, _within(edge_index, weight, keep))
    groups = D.GraphSpec.graphs(rows["molecule"].to_numpy(dtype=object))
    # the evaluator's view of the same graphs as the sealed test part (the others as train)
    gid = np.empty(len(keep), dtype=np.int64)
    gid[np.argsort(keep, kind="stable")] = np.arange(len(keep))
    part = D.GraphEdges(src=gid[edge_index[0]], dst=gid[edge_index[1]], weight=weight)
    view = D.GraphSpec(level="graph", graph_id="molecule", batching="graphs").view(
        (int((~keep).sum()), 0, int(keep.sum())), part, "test")
    assert got.batch["num_graphs"] == 3 and got.index.graphs == ("g2", "g3", "g4")
    assert got.index.rows == (-1, -1, -1) and got.sample_of_row.tolist() == groups.graph_index.tolist()
    assert torch.equal(got.batch["graph_index"], torch.as_tensor(groups.graph_index))
    assert torch.equal(got.batch["edge_index"], torch.as_tensor(view.edge_index))
    assert torch.equal(got.batch["edge_weight"], torch.as_tensor(view.edge_weight))
    assert got.batch["x_cat"].shape == (len(rows), 1) and "target_index" not in got.batch


def test_graph_answers_do_not_depend_on_the_other_graphs():
    x, edge_index, weight = _graphs(seed=4)
    everything = np.ones(len(x), dtype=bool)
    assembler = A.SampleAssembler(graph_spec())
    torch.manual_seed(0)
    conv, pool, head = B.GINConv(3, 8), B.GraphPool("mean"), nn.Linear(8, 2)

    def run(batch):
        h = conv(batch["x_num"], batch["edge_index"], batch.get("edge_weight"))
        return head(pool(h, batch["graph_index"], batch["num_graphs"]))
    with torch.no_grad():
        together = run(assembler.assemble(x, None, _within(edge_index, weight, everything)).batch)
        for g, name in enumerate(["g1", "g2", "g3", "g4", "g5"]):
            keep = (x["molecule"] == name).to_numpy()
            alone = assembler.assemble(x[keep].reset_index(drop=True), None,
                                       _within(edge_index, weight, keep))
            assert alone.index.graphs == (name,) and _same(run(alone.batch)[0], together[g])


def test_graph_level_refusals():
    x, edge_index, weight = _graphs()
    edges = _within(edge_index, weight, np.ones(len(x), dtype=bool))
    assembler = A.SampleAssembler(graph_spec())
    across = pd.concat([edges, pd.DataFrame({"src_pos": [0], "dst_pos": [5], "weight": [1.0]})])
    _raises(ValueError, "joins graph 'g1' and graph 'g2'", lambda: assembler.assemble(
        x, None, across))
    _raises(ValueError, "outside the 22 row(s)", lambda: assembler.assemble(
        x, None, pd.DataFrame({"src_pos": [0], "dst_pos": [22], "weight": [1.0]})))
    _raises(ValueError, "no history", lambda: assembler.assemble(x, x, edges))
    _raises(ValueError, "['molecule']", lambda: assembler.assemble(x.drop(columns="molecule")))
    _raises(ValueError, "graph_id column is missing at row 3", lambda: assembler.assemble(
        x.assign(molecule=x["molecule"].where(x.index != 3))))
    reference = D.GraphReference(x_num=np.zeros((2, 3), np.float32), x_cat=None,
                                 edge_index=np.zeros((2, 0), np.int64))
    _raises(ValueError, "whole graph", lambda: A.SampleAssembler(graph_spec(), reference))
    alone = assembler.assemble(x.iloc[:4])
    assert "edge_weight" not in alone.batch and alone.batch["edge_index"].shape == (2, 0)


# --------------------------------------------------------------------------- rows, empty, module

def test_row_batch_and_its_refusals():
    spec = D.NetSpec.from_dict(rt.reference_spec("row"))
    rows = pd.DataFrame({f"x{i}": np.arange(5.0) + i for i in range(4)})
    rows["zone"] = [0, 4, 7, -1, np.nan]
    got = A.SampleAssembler(spec).assemble(rows)
    assert set(got.batch) == {"x_num", "x_cat"} and got.batch["x_num"].dtype == torch.float32
    assert got.batch["x_cat"].tolist() == [[0], [4], [4], [4], [4]]      # unseen id: 4
    assert np.array_equal(got.batch["x_num"].numpy(), rows[[f"x{i}" for i in range(4)]]
                          .to_numpy(np.float32))
    assert got.index == D.SampleIndex.of_rows(5) and got.sample_of_row.tolist() == [0, 1, 2, 3, 4]
    assembler = A.SampleAssembler(spec)
    _raises(ValueError, "reads only its rows", lambda: assembler.assemble(rows, rows))
    _raises(ValueError, "no graph", lambda: assembler.assemble(rows, None, pd.DataFrame()))
    _raises(ValueError, "['x2']", lambda: assembler.assemble(rows.drop(columns="x2")))
    raw = rows.assign(zone=["a", "b", "c", "d", "e"])
    _raises(ValueError, "rows' column(s) ['zone'] do not all read as numbers", lambda:
            assembler.assemble(raw))
    _raises(ValueError, "['x0', 'x1', 'x2', 'x3']", lambda: assembler.assemble(
        rows.assign(x1=rows["x1"].astype(object).where(rows.index != 2, "n/a"))))
    _raises(TypeError, "shaped is a NoneType", lambda: assembler.assemble(None))
    _raises(TypeError, "shaped is a ndarray", lambda: assembler.assemble(rows.to_numpy()))
    _raises(TypeError, "history is a list", lambda: assembler.assemble(rows, [1, 2]))
    assert len(assembler.assemble(rows, rows.iloc[0:0])) == 5              # empty history: none


def test_each_nominal_column_maps_its_own_unseen_id():
    kind = {"name": "kind", "dtype": "int64", "role": "nominal", "cardinality": 3,
            "unseen_category_id": 0, "origin": "S3"}
    spec = _spec("window", [_num("x0"), _zone(5), kind],
                 {"length": 3, "horizon": 1, "autoregressive": False})
    rows = pd.DataFrame({"x0": np.arange(6.0), "zone": [0, 7, np.nan, 2, 4, 1],
                         "kind": pd.array([2, 5, 1, -3, None, 0], dtype="Int64"),
                         "series": ["a"] * 6, "ts": np.arange(6)})
    got = A.SampleAssembler(spec).assemble(rows)
    assert got.batch["x_cat"].shape == (4, 3, 2) and got.batch["x_cat"].dtype == torch.int64
    assert got.batch["x_cat"][0].tolist() == [[0, 2], [4, 0], [4, 1]]      # zone 4, kind 0
    assert got.batch["x_cat"][:, -1].tolist() == [[4, 1], [2, 0], [4, 0], [1, 0]]
    assert got.batch["x_num"][:, :, 0].tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_an_empty_request_is_an_empty_batch():
    x, edges = _node_world()
    reference, rows, _ = _served(x, edges, node_spec())
    empty_rows = pd.DataFrame({f"x{i}": [] for i in range(4)} | {"zone": []})
    assert len(A.SampleAssembler(D.NetSpec.from_dict(rt.reference_spec("row")))
               .assemble(empty_rows)) == 0
    window = A.SampleAssembler(window_spec()).assemble(_series_frame(1, {"s": 3}).iloc[0:0])
    assert window.batch["x_num"].shape == (0, 4, 3) and window.index.spans == ()
    node = A.SampleAssembler(node_spec(), reference).assemble(rows.iloc[0:0])
    assert node.batch["target_index"].shape == (0,) and len(node.batch["x_num"]) == 55
    graph = A.SampleAssembler(graph_spec()).assemble(_graphs()[0].iloc[0:0])
    assert graph.batch["num_graphs"] == 0 and graph.index.graphs == ()


def test_rendered_module_holds_the_contract():
    text = (ROOT / "model" / "assemble.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert conformance.size_findings(tree, "model/assemble.py") == []
    assert len(text.splitlines()) <= 400
    pkg, err = conformance.Package.read(ROOT, RENDERED["blueprint_map"], PIPELINE)
    assert err is None, err
    for check in (conformance.c4_imports_follow_the_arrows,
                  conformance.c7_no_module_level_mutable_state):
        bad = [m.line() for m in check(pkg) if m.level is Level.ERROR]
        assert not bad, bad
    inner = [(n.level, n.module) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             and n.level]
    assert inner == [(2, "domain")], inner
    public = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
              and not n.name.startswith("_")}
    assert public == {"SampleAssembler", "AssembledBatch"}, public
    for pattern in (r"\bIDS\b", r"(?i)\btraffic\b", r"(?i)\bintrusion", r"(?i)\bflows?\b"):
        assert not re.search(pattern, text), pattern


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
    if failed:
        print(f"\n{len(failed)} of {len(tests)} assembler test(s) failed")
        return 1
    print(f"\n{len(tests)} assembler test(s) passed on the rendered assemble.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
