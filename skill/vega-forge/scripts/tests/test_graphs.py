#!/usr/bin/env python3
"""
test_graphs.py — the graph half of the DataSource (`templates/graphs.py` through
`ShapedParts`), proved on the rendered package (build plan B08; backend design §9.4, §9.5,
§17.1; R-A3 the inference-graph rule).

    python3 test_graphs.py [-v] [-k <substring>]
    python3 -m pytest test_graphs.py

The node-level fixture is 60 raw rows, each part a scattered set of raw positions stored in a
scrambled order, whose `x0` is the row's raw position — so every node of a view names the raw
row it came from — and a symmetric, weighted edges part with every pair of parts in it. What is
proved:

  * edge resolution through `fitted.json.__split_raw__` (every end maps back to its raw
    position; an unknown position and a wrong part name are refused, each named), and on a
    dedupe-shifted real-shaper fixture, where `__split__` and `__split_raw__` disagree;
  * the inductive train view: train nodes only, no test (or valid) node, no edge touching one;
  * the valid view: reference = train with its train–train edges, queries = valid, attached by
    reference → query edges only — no query ↔ query, no query → reference edge;
  * the sealed test view: reference = train ∪ valid with every edge among them, test queries
    attached reference → query only; no test–test edge in any view;
  * `exposure()` is (0, false) inductive without reading a file, (n > 0, true) transductive;
    a transductive spec without a `user:` exposure is refused before anything is opened; the
    transductive training view reads the test part's feature columns only (never its label);
    the inductive train and valid views never open the test part;
  * the neighbour sampler: deterministic per epoch seed, the fan-out bound per hop, seeds-first
    relabelling (`target_index = 0…S−1`), every sampled edge an edge of the view;
  * `GraphCollate`: whole graphs in batch order, edge indices offset by the node count so far;
  * graph level: row ids are graph ids, one label per graph; a graph straddling parts, an edge
    between two graphs and a graph whose rows disagree on the label each raise naming the id;
  * the design matrices (flat, neighbour-mean, graph-pooled), `make_loader` per level, the seal
    for graph samples, and C8/C11–C13 on the rendered files.

Exit 0 when every test passes.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data_fixtures as fx

NS = fx.ingestion("graphs_pkg", "graph", only=("frames.py", "graphs.py", "data.py"))
D = NS.domain
N = 60
CLASSES = ["a", "b", "c", "d"]


def node_frame(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({"x0": np.arange(n, dtype=np.float64), "x1": rng.normal(size=n),
                         "zone": rng.integers(0, 4, size=n),
                         "community": [CLASSES[i % 4] for i in range(n)]})


def node_parts(n: int = N) -> dict:
    """Scattered raw positions per part (train 36, valid 12, test 12), scrambled order."""
    rng = np.random.default_rng(2)
    perm = rng.permutation(n)
    return {"train": perm[:36], "valid": perm[36:48], "test": perm[48:]}


def edge_frame(parts: dict, weighted: bool = True, n: int = N) -> pd.DataFrame:
    """Random undirected pairs, both directions written; every pair of parts represented."""
    rng = np.random.default_rng(4)
    part_of = {int(p): name for name, pos in parts.items() for p in pos}
    pairs = set()
    while len(pairs) < 140:
        a, b = (int(v) for v in rng.integers(0, n, size=2))
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    rows = []
    for a, b in sorted(pairs):
        w = float(rng.uniform(0.5, 2.0))
        for s, d in ((a, b), (b, a)):
            rows.append({"src_pos": s, "dst_pos": d, "src_part": part_of[s],
                         "dst_part": part_of[d], "weight": w if weighted else None})
    return pd.DataFrame(rows).sort_values(["src_pos", "dst_pos"]).reset_index(drop=True)


def node_fixture(protocol: str = "inductive", exposure=None, weighted: bool = True, **sample):
    frame, parts = node_frame(), node_parts()
    edges = edge_frame(parts, weighted)
    root = Path(tempfile.mkdtemp(prefix="graphs-fx-"))
    man = fx.write_shaped(root, {k: frame.iloc[v] for k, v in parts.items()}, parts, edges)
    spec = fx.rt.reference_spec("graph")
    spec["data_contract"]["features"] = [fx.numeric("x0"), fx.numeric("x1"),
                                         fx.nominal("zone", 5)]
    spec["data_contract"]["input_dim"] = 3
    spec["data_contract"]["sample"].update({"protocol": protocol, "exposure": exposure, **sample})
    spec = D.NetSpec.from_dict(fx.spec_on(spec, man))
    return NS.data.ShapedParts(), spec, root, (frame, parts, edges)


USER_EXPOSURE = {"kind": "features", "rows": "test", "source": "user:turn4"}


def pairs_of(part):
    """The view's edges as (raw source, raw destination) pairs (x0 is the raw position)."""
    x0 = part.x_num[:, 0].numpy().astype(np.int64)
    ei = part.edge_index.numpy()
    return list(zip(x0[ei[0]].tolist(), x0[ei[1]].tolist()))


def edge_set(edges, parts, a: tuple, b: tuple):
    """The edges part's pairs whose source part is in `a` and destination part in `b`."""
    keep = edges.src_part.isin(a) & edges.dst_part.isin(b)
    return set(zip(edges.src_pos[keep].tolist(), edges.dst_pos[keep].tolist()))


# --------------------------------------------------------------------------- edges

def test_edge_resolution_through_split_raw():
    frame, parts, edges = node_frame(), node_parts(), edge_frame(node_parts())
    table = NS.graphs.EdgeTable.resolve(edges, parts)
    raw = np.concatenate([parts["train"], parts["valid"], parts["test"]])
    assert table.sizes == (36, 12, 12)
    assert np.array_equal(raw[table.edges.src], edges.src_pos.to_numpy())
    assert np.array_equal(raw[table.edges.dst], edges.dst_pos.to_numpy())
    assert np.allclose(table.edges.weight, edges.weight.to_numpy())
    wrong = edges.copy()
    wrong.loc[0, "src_part"] = "test" if wrong.loc[0, "src_part"] != "test" else "train"
    fx.raises(lambda: NS.graphs.EdgeTable.resolve(wrong, parts), "ValueError", "edge 0",
              f"src_pos {int(wrong.src_pos[0])}", "the edges part names")
    lost = edges.copy()
    lost.loc[3, "dst_pos"] = 999
    fx.raises(lambda: NS.graphs.EdgeTable.resolve(lost, parts), "ValueError", "edge 3",
              "dst_pos 999", "__split_raw__")
    unweighted = NS.graphs.EdgeTable.resolve(edge_frame(parts, weighted=False), parts)
    assert unweighted.edges.weight is None


# --------------------------------------------------------------------------- the views (R-A3)

def test_inductive_train_view_holds_no_test_node_or_edge():
    sp, spec, _, (frame, parts, edges) = node_fixture()
    train = sp.load_part(spec, "train")
    x0 = set(train.x_num[:, 0].numpy().astype(np.int64).tolist())
    assert x0 == set(parts["train"].tolist()), "the train view holds a non-train node"
    assert set(train.row_ids.tolist()) == set(parts["train"].tolist())
    assert set(pairs_of(train)) == edge_set(edges, parts, ("train",), ("train",))
    labels = [CLASSES.index(frame.community[r]) for r in train.row_ids]
    assert train.y.tolist() == labels and train.row_id_kind.value == "raw_position"


def test_valid_view_attaches_valid_queries_to_the_train_reference_only():
    sp, spec, _, (frame, parts, edges) = node_fixture()
    valid = sp.load_part(spec, "valid")
    x0 = valid.x_num[:, 0].numpy().astype(np.int64)
    assert set(x0.tolist()) == set(parts["train"].tolist()) | set(parts["valid"].tolist())
    targets = x0[valid.target_index.numpy()]
    assert np.array_equal(targets, valid.row_ids) and set(targets) == set(parts["valid"])
    got = pairs_of(valid)
    assert len(got) == len(set(got)), "an edge is listed twice"
    train, qset = set(parts["train"].tolist()), set(parts["valid"].tolist())
    reference = [(s, d) for s, d in got if s in train and d in train]
    attached = [(s, d) for s, d in got if (s, d) not in set(reference)]
    assert set(reference) == edge_set(edges, parts, ("train",), ("train",))
    assert all(s in train and d in qset for s, d in attached), "a query sends a message"
    want = {(s, d) for s, d in edge_set(edges, parts, ("train",), ("valid",))}
    assert set(attached) == want, "a train-valid link is missing or oriented wrongly"


def test_sealed_test_view_reference_is_train_and_valid():
    sp, spec, root, (frame, parts, edges) = node_fixture()
    test = sp.load_sealed_part(spec, "tok", str(fx.seal(root, "tok")))
    got = pairs_of(test)
    ref = set(parts["train"].tolist()) | set(parts["valid"].tolist())
    queries = set(parts["test"].tolist())
    inside = {(s, d) for s, d in got if s in ref and d in ref}
    assert inside == edge_set(edges, parts, ("train", "valid"), ("train", "valid"))
    assert all(s in ref and d in queries for s, d in set(got) - inside), "test sends a message"
    assert {(s, d) for s, d in got if d in queries} == edge_set(edges, parts, ("train", "valid"),
                                                                 ("test",))
    assert set(test.row_ids.tolist()) == queries and len(test) == 12
    assert test.exposure.as_dict() == {"protocol": "inductive", "edges_into_test": 0,
                                       "features_seen_in_training": False}


def test_weights_follow_their_edges():
    sp, spec, _, (frame, parts, edges) = node_fixture()
    valid = sp.load_part(spec, "valid")
    want = {(s, d): w for s, d, w in zip(edges.src_pos, edges.dst_pos, edges.weight)}
    got = valid.edge_weight.numpy()
    assert all(abs(want[p] - w) < 1e-6 for p, w in zip(pairs_of(valid), got))


# --------------------------------------------------------------------------- protocol and seal

class _Spy:
    """A PartFrames that records what is opened (part, columns or 'features-only')."""

    def __init__(self):
        base = NS.frames.PartFrames
        opened = self.opened = []

        class Spy(base):
            def open(self, spec, part, columns=None):
                opened.append((str(D.Part(part).value), tuple(columns or ())))
                return base.open(self, spec, part, columns)

            def open_features_only(self, spec, part):
                opened.append((str(D.Part(part).value), "features-only"))
                return base.open_features_only(self, spec, part)

        self.reader = Spy()


def test_inductive_views_never_open_the_test_part():
    spy = _Spy()
    _, spec, _, _ = node_fixture()
    sp = NS.data.ShapedParts(spy.reader)
    sp.load_part(spec, "train")
    sp.load_part(spec, "valid")
    assert spy.opened and all(p != "test" for p, _ in spy.opened), spy.opened


def test_exposure_inductive_reads_nothing_transductive_counts():
    sp, spec, _, _ = node_fixture()
    dead = spec.to_dict()
    dead["input"]["manifest"] = "/nonexistent/manifest.json"
    assert sp.exposure(D.NetSpec.from_dict(dead)).as_dict() == {
        "protocol": "inductive", "edges_into_test": 0, "features_seen_in_training": False}
    sp, spec, _, (frame, parts, edges) = node_fixture("transductive", USER_EXPOSURE)
    exposure = sp.exposure(spec)
    assert exposure.features_seen_in_training is True
    assert exposure.edges_into_test == int((edges.dst_part == "test").sum()) > 0


def test_transductive_reads_test_features_only_and_undeclared_is_refused():
    spy = _Spy()
    _, spec, _, (frame, parts, edges) = node_fixture("transductive", USER_EXPOSURE)
    sp = NS.data.ShapedParts(spy.reader)
    train = sp.load_part(spec, "train")
    assert ("test", "features-only") in spy.opened
    assert not [o for o in spy.opened if o[0] == "test" and o[1] != "features-only"]
    assert train.x_num.shape[0] == N and len(train) == 36, "transductive: all nodes, train targets"
    assert len(pairs_of(train)) == len(edges)
    test_cols = spy.reader.open_features_only(spec, "test").frame.columns.tolist()
    assert "community" not in test_cols and test_cols == ["x0", "x1", "zone"]
    _, undeclared, _, _ = node_fixture("inductive")
    bad = undeclared.to_dict()
    bad["data_contract"]["sample"].update({"protocol": "transductive", "exposure": None})
    bad["input"]["manifest"] = "/nonexistent/manifest.json"
    fx.raises(lambda: NS.data.ShapedParts().load_part(D.NetSpec.from_dict(bad), "train"),
              "SealedPartError", "exposure")
    fx.raises(lambda: spy.reader.open_features_only(undeclared, "test"), "SealedPartError",
              "transductive")


def test_seal_for_graph_samples():
    sp, spec, root, _ = node_fixture()
    dead = spec.to_dict()
    dead["input"]["manifest"] = "/nonexistent/manifest.json"
    dead = D.NetSpec.from_dict(dead)
    for call in (lambda: sp.load_part(dead, "test"), lambda: sp.iter_batches(dead, "test"),
                 lambda: sp.load_resident(dead, "test", "cpu")):
        fx.raises(call, "SealedPartError", "test part")
    fx.raises(lambda: sp.load_sealed_part(spec, "x", str(root / "none.json")), "SealedPartError",
              "no seal")
    fx.raises(lambda: NS.frames.PartFrames().open(spec, "test"), "SealedPartError",
              "load_sealed_part")


# --------------------------------------------------------------------------- sampling, collation

def test_neighbour_sampler_is_seeded_bounded_and_seeds_first():
    sp, spec, _, _ = node_fixture(batching="sampled", neighbours=[3])
    train = sp.load_part(spec, "train")
    one = NS.graphs.NeighbourSampler(train, (3,), 8)
    first, again = list(one.epoch(17)), list(one.epoch(17))
    assert len(first) == len(one) == 5
    for a, b in zip(first, again):
        assert all(torch.equal(a[k], b[k]) for k in a), "the same epoch seed moved the batch"
    assert any(not torch.equal(a["y"], b["y"]) for a, b in zip(first, one.epoch(18)))
    graph = set(pairs_of(train))
    x0 = train.x_num[:, 0].numpy().astype(np.int64)
    degree = np.bincount(train.edge_index[1].numpy(), minlength=len(x0))
    for b in first:
        s = len(b["target_index"])
        assert torch.equal(b["target_index"], torch.arange(s))
        seeds = b["x_num"][:s, 0].numpy().astype(np.int64)
        position = {int(v): i for i, v in enumerate(x0)}
        assert b["y"].tolist() == [int(train.y[train.target_index.tolist().index(position[v])])
                                   for v in seeds]
        bx0 = b["x_num"][:, 0].numpy().astype(np.int64)
        src, dst = b["edge_index"].numpy()
        assert all((int(bx0[u]), int(bx0[v])) in graph for u, v in zip(src, dst))
        into = np.bincount(dst, minlength=len(bx0))
        assert (into[s:] == 0).all(), "one hop: only the seeds receive edges"
        want = np.where(degree[[position[v] for v in seeds]] > 0, 3, 0)
        assert np.array_equal(into[:s], want), "each seed draws exactly its fan-out"
    two = NS.graphs.NeighbourSampler.for_spec(
        D.NetSpec.from_dict({**spec.to_dict(), "training": {"batch_size": 8, "max_epochs": 1}}),
        train)
    deep = NS.graphs.NeighbourSampler(train, (3, 2), 8)
    for b in deep.epoch(3):
        into = np.bincount(b["edge_index"][1].numpy(), minlength=len(b["x_num"]))
        assert into.max() <= 5 and (into[len(b["target_index"]):] <= 2).all()
    assert len(two) == 5


def graph_level_fixture(frame=None, parts=None, edges=None):
    rng = np.random.default_rng(8)
    if frame is None:
        rows, g = [], 0
        for g in range(12):
            for i in range(3 + g % 3):
                rows.append({"molecule": f"m{g:02d}", "x0": 0.0, "x1": rng.normal(),
                             "zone": int(rng.integers(0, 4)), "y": g % 2})
        frame = pd.DataFrame(rows)
        frame["x0"] = np.arange(len(frame), dtype=np.float64)
    graph_of = frame.molecule.to_numpy()
    if parts is None:
        mol = sorted(set(graph_of))
        part_of = {m: ("train" if i < 8 else "valid" if i < 10 else "test")
                   for i, m in enumerate(mol)}
        pos = np.arange(len(frame))
        parts = {p: pos[[part_of[m] == p for m in graph_of]] for p in ("train", "valid", "test")}
    if edges is None:
        part_of_pos = {int(p): k for k, v in parts.items() for p in v}
        rows = []
        for m in sorted(set(graph_of)):
            members = np.flatnonzero(graph_of == m)
            for a, b in zip(members, members[1:]):
                for s, d in ((a, b), (b, a)):
                    rows.append({"src_pos": int(s), "dst_pos": int(d),
                                 "src_part": part_of_pos[int(s)],
                                 "dst_part": part_of_pos[int(d)], "weight": None})
        edges = pd.DataFrame(rows)
    root = Path(tempfile.mkdtemp(prefix="graphs-level-"))
    man = fx.write_shaped(root, {k: frame.iloc[v] for k, v in parts.items()}, parts, edges)
    spec = fx.rt.reference_spec("graph-level")
    spec["data_contract"]["features"] = [fx.numeric("x0"), fx.numeric("x1"),
                                         fx.nominal("zone", 5)]
    spec["data_contract"]["input_dim"] = 3
    spec["training"]["batch_size"] = 3
    spec = D.NetSpec.from_dict(fx.spec_on(spec, man))
    return NS.data.ShapedParts(), spec, root, (frame, parts, edges)


def test_graph_level_part_and_collate_offsets():
    sp, spec, root, (frame, parts, edges) = graph_level_fixture()
    train = sp.load_part(spec, "train")
    assert train.row_id_kind.value == "graph_id" and train.num_graphs == 8
    assert list(train.row_ids) == [f"m{g:02d}" for g in range(8)]
    assert train.y.tolist() == [g % 2 for g in range(8)]
    collate = NS.graphs.GraphCollate(train, 3)
    b = collate.collate([2, 0])
    sizes = [int((train.graph_index == g).sum()) for g in (2, 0)]
    x0 = frame.x0.to_numpy()
    want = np.concatenate([x0[frame.molecule == "m02"], x0[frame.molecule == "m00"]])
    assert np.array_equal(b["x_num"][:, 0].numpy(), want), "nodes not in batch order"
    assert b["graph_index"].tolist() == [0] * sizes[0] + [1] * sizes[1] and b["num_graphs"] == 2
    src, dst = b["edge_index"].numpy()
    slot = b["graph_index"].numpy()
    assert (slot[src] == slot[dst]).all(), "an edge crosses graphs after collation"
    assert (src[slot[src] == 1] >= sizes[0]).all(), "the second graph's edges are not offset"
    assert len(src) == 2 * (sizes[0] - 1) + 2 * (sizes[1] - 1)
    assert b["y"].tolist() == [0, 0]
    seen = [g for batch in sp.make_loader(train, spec, shuffle=True, seed=5)
            for g in batch["y"].tolist()]
    assert len(seen) == 8 and len(list(sp.make_loader(train, spec))) == 3
    test = sp.load_sealed_part(spec, "t", str(fx.seal(root, "t")))
    assert list(test.row_ids) == ["m10", "m11"]


def test_graph_level_refusals_name_the_graph():
    frame = graph_level_fixture()[3][0]
    parts = graph_level_fixture()[3][1]
    moved = {k: v.copy() for k, v in parts.items()}
    stray = int(np.flatnonzero(frame.molecule.to_numpy() == "m03")[0])
    moved["train"] = moved["train"][moved["train"] != stray]
    moved["valid"] = np.concatenate([moved["valid"], [stray]])
    sp, spec, _, _ = graph_level_fixture(frame, moved)
    fx.raises(lambda: sp.load_part(spec, "train"), "ValueError", "'m03'", "straddles")
    _, _, _, (_, _, edges) = graph_level_fixture()
    a, b = (int(np.flatnonzero(frame.molecule.to_numpy() == m)[0]) for m in ("m04", "m05"))
    cross = pd.concat([edges, pd.DataFrame([{"src_pos": a, "dst_pos": b, "src_part": "train",
                                             "dst_part": "train", "weight": None}])])
    sp, spec, _, _ = graph_level_fixture(frame, parts, cross)
    fx.raises(lambda: sp.load_part(spec, "train"), "ValueError", "'m04'", "'m05'")
    mixed = frame.copy()
    mixed.loc[mixed.molecule == "m06", "y"] = [0, 1, 0]
    sp, spec, _, _ = graph_level_fixture(mixed, parts)
    fx.raises(lambda: sp.load_part(spec, "train"), "ValueError", "'m06'", "different labels")


# --------------------------------------------------------------------------- design, loaders

def test_design_matrices_and_loaders():
    sp, spec, _, (frame, parts, edges) = node_fixture()
    valid = sp.load_part(spec, "valid")
    flat = sp.design_matrix(spec, valid)
    assert flat.x.shape == (12, 3) and np.array_equal(flat.row_ids, valid.row_ids)
    assert np.array_equal(flat.x[:, 0], valid.row_ids.astype(np.float32))
    mean = sp.design_matrix(spec, valid, "neighbour-mean")
    train = set(parts["train"].tolist())
    for i, r in enumerate(valid.row_ids):
        sources = [s for s, d in edge_set(edges, parts, ("train",), ("valid",)) if d == r]
        want = np.mean(sources) if sources else 0.0
        assert abs(mean.x[i, 3] - want) < 1e-4 and all(s in train for s in sources)
    fx.raises(lambda: sp.design_matrix(spec, valid, "graph-pooled"), "ValueError", "node level")
    batches = list(sp.make_loader(valid, spec))
    assert len(batches) == 1 and len(batches[0]["target_index"]) == 12
    assert len(list(sp.make_loader(valid, spec, batch_size=5))) == 3
    gsp, gspec, _, (gframe, _, _) = graph_level_fixture()
    part = gsp.load_part(gspec, "train")
    pooled = gsp.design_matrix(gspec, part, "graph-pooled")
    rows = gframe.iloc[np.sort(np.concatenate([np.flatnonzero(gframe.molecule == f"m{g:02d}")
                                               for g in range(8)]))]
    agg = rows.groupby("molecule")["x1"].agg(["mean", "max", "sum"])
    assert pooled.x.shape == (8, 9)
    assert np.allclose(pooled.x[:, [1, 4, 7]], agg.to_numpy(), atol=1e-5)
    fx.raises(lambda: gsp.design_matrix(gspec, part), "ValueError", "graph-pooled")


# --------------------------------------------------------------------------- the real shaper

def test_dedupe_shifted_split_raw_resolution():
    """A key_pair graph shaped by shape.py after a `dedupe` removed rows at the head of the
    file: `__split__` (post-dedupe positions) no longer equals `__split_raw__`, and every edge
    end still resolves to the raw row it names (the `tag` column carries the raw position)."""
    work = Path(tempfile.mkdtemp(prefix="graphs-dedupe-"))
    rng = np.random.default_rng(9)
    n = 80
    keys = [f"B{i:03d}" for i in range(n)]
    base = pd.DataFrame({"bus_key": keys,
                         "feeder_to": [keys[i - 1 - (i % 3)] if i % 4 else "" for i in range(n)],
                         "load": rng.normal(size=n).round(4),
                         "overloaded": (rng.uniform(size=n) > 0.6).astype(int)})
    frame = pd.concat([base.iloc[:6], base], ignore_index=True)     # six leading duplicates
    frame.insert(0, "tag", np.arange(len(frame), dtype=np.float64))
    csv = work / "grid.csv"
    frame.to_csv(csv, index=False)
    steps = [fx.step("S1", "dedupe", [], {"on": ["bus_key", "feeder_to", "load", "overloaded"],
                                           "keep": "last"}, "analysis:F1"),
             fx.step("S2", "drop_identity", ["bus_key", "feeder_to"], {},
                     "geometry:typing/bus_key"),
             fx.step("S3", "select_partition", ["overloaded"],
                     {"label": "overloaded", "task": "binary"}, "geometry:partitions/overloaded"),
             fx.step("S4", "split", [], {"kind": "random", "test": 0.2, "valid": 0.2},
                     "shaper:default"),
             fx.step("S5", "graph_edges", ["bus_key", "feeder_to"],
                     {"method": "key_pair", "columns": ["bus_key", "feeder_to"],
                      "symmetric": True, "self_loops": False, "weight": None}, "analysis:F4/T1")]
    man = fx.run_shaper(work, csv, fx.recipe(csv, steps, "overloaded"))
    import json
    fitted = json.loads((man.parent / "fitted.json").read_text())
    assert fitted["__split__"] != fitted["__split_raw__"], "the dedupe shifted nothing"
    spec = fx.rt.reference_spec("graph")
    spec["task"] = {"kind": "binary", "label": "overloaded", "classes": [0, 1], "positive": 1}
    spec["data_contract"]["features"] = [fx.numeric("tag"), fx.numeric("load")]
    spec["data_contract"]["input_dim"] = 2
    spec = D.NetSpec.from_dict(fx.spec_on(spec, man))
    reader = NS.frames.PartFrames()
    table = NS.graphs.EdgeTable.load(spec, reader)
    shaped = man.parent
    tags = np.concatenate([pd.read_parquet(next(shaped.glob(f"*.{p}.parquet")))["tag"].to_numpy()
                           for p in ("train", "valid", "test")])
    part_edges = pd.read_parquet(next(shaped.glob("*.edges.parquet")))
    assert len(part_edges) > 20
    assert np.array_equal(tags[table.edges.src], part_edges.src_pos.to_numpy())
    assert np.array_equal(tags[table.edges.dst], part_edges.dst_pos.to_numpy())
    train = NS.data.ShapedParts(reader).load_part(spec, "train")
    x_tag = train.x_num[:, 0].numpy().astype(np.int64)
    got = set(zip(x_tag[train.edge_index[0].numpy()].tolist(),
                  x_tag[train.edge_index[1].numpy()].tolist()))
    assert got == edge_set(part_edges, None, ("train",), ("train",))
    assert np.array_equal(train.row_ids, x_tag[train.target_index.numpy()])


def test_template_discipline():
    """C11–C13 on the rendered graph files; C8 finds the seal on the adapter class."""
    import conformance
    files = [NS.root / "ingestion" / n for n in ("data.py", "frames.py", "graphs.py")]
    _, findings = conformance.size_only([str(f) for f in files])
    assert not findings, [f.message() for f in findings]
    pkg, err = conformance.Package.read(NS.root, NS.blueprint_map, NS.pipeline)
    assert err is None, err
    marks = conformance.c8_datasource_seals_the_test(pkg)
    assert all(m.level.name == "OK" for m in marks), [m.line() for m in marks]


def main(argv) -> int:
    return fx.main_for(globals(), argv, "graph")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
