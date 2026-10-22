#!/usr/bin/env python3
"""
test_samples.py — the serving sample assemblers (`templates/serving/package/samples.py`), the
graph context (`graphs.py`) and the reference graph pack writes (`vega-pack/scripts/
graph_pack.py`), proved on the *rendered* modules (placed by `render_templates.py` beside the
rendered domain, assembler and network) over the *real* shaper: every fixture goes through
`dataset-shaper/scripts/shape.py` and its `--transform-only` module, as a bundle runs it
(the rendering, the fixtures, the specs and the pack context are `samples_fixtures.py`'s).

    python3 test_samples.py [-v] [-k <substring>]
    python3 -m pytest test_samples.py

What is proved (build plan WP S04; R-A3–R-A7, R-A10):

  windows  a request row is answered iff length-1 rows of its series precede it (history
           included), else `window incomplete` naming how many it has; `history ++ rows` is
           transformed as one frame (R-A6), so lags run from the history into the rows and the
           windows are that frame cut by series; `x_cat (B, L, C)`; the autoregressive channel
           is the raw target; the lag warning until `history_rows` rows of history; an answer
           does not depend on how the rows were split between history and request; a forecast
           without its target column is refused naming it.
  rows     the transform (history feeding the lags) then one sample per row.
  graphs   pack's reference files carry no raw position; the served node batch is the sealed
           test view (reference train ∪ valid, queries attached reference -> query); graph
           answers are identical alone and together through a real network; graph level groups
           by `graph_id`; history is refused; a row with no link says so; a tampered or unpinned
           reference file is refused (N-S9).
  pack     `graph_pack.write(ctx)` writes vega/graph-reference@1 and records the counts; a
           transductive protocol is refused (exit 2, `inductive graphs only`); a transform that
           does not serve the graph (exit 3), a part that is not the manifest's (exit 2); the
           1,000,000-node warning.
  module   C11–C13, C4 and C7 on the rendered files; no foreign literal (lint_templates.py).

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parent.parent.parent
LINT = SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
VERBOSE = "-v" in sys.argv
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from samples_fixtures import (  # noqa: E402
    A, D, G, GRAPHS, N, NODE_GRAPHS, PACK, PIPELINE, RENDERED, ROOT, S, SERIES_LENGTHS, TEMPLATES,
    WORK, Records, _context, _exit, _raises, assembler, bundle, keypair_spec, knn_spec,
    level_spec, row_spec, shaped, window_spec)
import conformance  # noqa: E402
import graph_pack  # noqa: E402
import output_record  # noqa: E402
from vega_core import Level  # noqa: E402
from vega_pack_core import digest_file  # noqa: E402


# --------------------------------------------------------------------------- windows

def _predecessors(rows, history):
    """Per request row: the rows of its series before it in history + rows (by ts)."""
    both = pd.concat([history.assign(_r=-1), rows.assign(_r=np.arange(len(rows)))])
    both = both.sort_values(["station", "ts"], kind="stable")
    both["_k"] = both.groupby("station").cumcount()
    return both[both["_r"] >= 0].set_index("_r").sort_index()["_k"].to_numpy()


def _history(s, per_series):
    """The last `per_series[station]` rows of train ∪ valid of each station, shuffled."""
    past = pd.concat([s.rows("train"), s.rows("valid")]).sort_values("ts", kind="stable")
    parts = [past[past["station"] == st].tail(n) for st, n in per_series.items() if n]
    out = pd.concat(parts) if parts else past.iloc[0:0]
    return out.sample(frac=1, random_state=3).reset_index(drop=True)


def test_window_rows_are_answered_iff_their_series_has_length_minus_one_predecessors():
    s = shaped("series")
    spec = window_spec()
    rows = s.rows("test")
    for per in ({"A": 5, "B": 1, "C": 0}, {"A": 0, "B": 0, "C": 0}, {"A": 3, "B": 2, "C": 9}):
        history = _history(s, per)
        got = S.assembler_for(Records(spec), s.module.transform).assemble(rows, history)
        k = _predecessors(rows, history)
        assert np.array_equal(got.answered, k >= 3), per
        assert got.n_samples == int((k >= 3).sum()) == got.batch["x_num"].shape[0]
        for i in np.flatnonzero(k < 3):
            note = ("window incomplete: needs 3 preceding rows, "
                    f"{int(k[i])} available — send them as history")
            assert got.warnings[i][0] == note, (i, got.warnings[i])
        assert all(not any(w.startswith("window incomplete") for w in got.warnings[i])
                   for i in np.flatnonzero(k >= 3))


def test_window_batch_is_the_concatenated_transform_cut_by_series():
    """R-A6: history ++ rows transformed as one frame, then each answered row's window is its
    series' last L rows up to it — the lag features read the history, the autoregressive
    channel is the raw target, x_cat is (B, L, C)."""
    s = shaped("series")
    rows, history = s.rows("test"), _history(s, {"A": 6, "B": 4, "C": 2})
    got = S.assembler_for(Records(spec := window_spec()), s.module.transform).assemble(rows,
                                                                                      history)
    frame = pd.concat([history, rows], ignore_index=True)
    want = s.module.transform(frame, history=frame.iloc[0:0])
    want["target"], want["_at"] = frame["target"].to_numpy(), np.arange(-len(history), len(rows))
    assert got.batch["x_cat"].dim() == 3 and got.batch["x_cat"].shape[1:] == (4, 1)
    assert got.batch["x_num"].shape[1:] == (4, 4)                   # x, x_lag1, x_lag2, target
    assert list(got.shaped.index) == list(range(-len(history), len(rows)))
    for i in np.flatnonzero(got.answered):
        series = want[want["station"] == rows.loc[i, "station"]]
        series = pd.concat([series[series["_at"] < 0].sort_values("ts", kind="stable"),
                            series[series["_at"] >= 0].sort_values("ts", kind="stable")])
        at = series.index[series["_at"] == i][0]
        window = series.loc[:at].tail(4)
        k = got.sample_of_row[i]
        x_num = window[["x", "x_lag1", "x_lag2", "target"]].to_numpy(np.float32)
        assert np.array_equal(got.batch["x_num"][k].numpy(), x_num, equal_nan=True), i
        assert got.batch["x_cat"][k, :, 0].tolist() == window["zone"].astype(int).tolist(), i
        assert got.index.spans[k] == (int(window["_at"].iloc[0]), i)
    # the lags run from the history into the rows: every request row's lags are the ones the
    # whole input gives it (the evaluation's part restarts them at the part's start)
    full = pd.concat([_history(s, SERIES_LENGTHS), rows], ignore_index=True)
    whole = S.assembler_for(Records(spec), s.module.transform).assemble(
        rows, full.iloc[:-len(rows)])
    alone = s.module.transform(s.raw, history=s.raw.iloc[0:0])
    pos = np.asarray(s.split["test"], dtype=np.int64)
    lags = whole.shaped.loc[0:, ["x_lag1", "x_lag2"]].to_numpy()
    assert np.allclose(lags, alone.iloc[pos][["x_lag1", "x_lag2"]].to_numpy()) and \
        not np.isnan(lags).any()
    assert np.isnan(s.part("test")[["x_lag2"]].to_numpy()).any()


def test_window_notes_name_the_missing_history_and_the_unseen_category():
    s = shaped("series")
    spec = window_spec()
    first = {st: None for st in SERIES_LENGTHS}
    rows = s.rows("test").assign(zone=lambda f: f["zone"].where(f.index != 4, "q"))
    for st in first:
        first[st] = rows[rows["station"] == st].sort_values("ts").index.tolist()
    adapter = S.assembler_for(Records(spec, s.recipe), s.module.transform)
    assert adapter.history_rows == 3 + 2
    lag = ("lag features at the start of the window lack history (2 rows) — send 5 rows of "
           "history for a fully determined answer")
    short = adapter.assemble(rows, _history(s, {st: 3 for st in SERIES_LENGTHS}))
    full = adapter.assemble(rows, _history(s, {st: 5 for st in SERIES_LENGTHS}))
    for st, order in first.items():
        assert [lag in short.warnings[i] for i in order[:3]] == [True, True, False], st
        assert all(short.answered[order]) and all(full.answered[order])
    assert not any(lag in w for w in full.warnings)
    assert full.warnings[4][-1] == "unseen category in zone"
    k = full.sample_of_row[4]
    assert full.batch["x_cat"][k, -1, 0].item() == 3                 # the unseen id


def test_a_window_answer_does_not_depend_on_how_rows_are_split():
    """The same rows answered as one request, or with the earlier ones sent as history:
    one frame is transformed either way, so the answered windows are the same tensors."""
    s = shaped("series")
    adapter = S.assembler_for(Records(window_spec()), s.module.transform)
    rows = s.rows("test").sort_values("ts", kind="stable").reset_index(drop=True)
    history = _history(s, {st: 5 for st in SERIES_LENGTHS})
    together = adapter.assemble(rows, history)
    later = adapter.assemble(rows.iloc[12:], pd.concat([history, rows.iloc[:12]]))
    for i in range(12, len(rows)):
        a, b = together.sample_of_row[i], later.sample_of_row[i - 12]
        assert a >= 0 and b >= 0
        for key in ("x_num", "x_cat"):
            assert np.array_equal(together.batch[key][a].numpy(), later.batch[key][b].numpy(),
                                  equal_nan=True), (i, key)


def test_a_forecast_without_its_target_column_is_refused_naming_it():
    s = shaped("series")
    adapter = S.assembler_for(Records(window_spec()), s.module.transform)
    rows, history = s.rows("test"), _history(s, {"A": 4, "B": 4, "C": 4})
    _raises(ValueError, "the rows lack the target column 'target'",
            lambda: adapter.assemble(rows.drop(columns="target"), history))
    _raises(ValueError, "the history rows lack the target column 'target'",
            lambda: adapter.assemble(rows, history.drop(columns="target")))
    plain = S.assembler_for(Records(window_spec(autoregressive=False)), s.module.transform)
    assert plain.assemble(rows.drop(columns="target"), history).batch["x_num"].shape[-1] == 3
    seq = S.assembler_for(Records(window_spec(kind="sequence-regression")), s.module.transform)
    assert seq.assemble(rows.drop(columns="target")).batch["x_num"].shape[-1] == 3


def test_window_refusals():
    s = shaped("series")
    adapter = S.assembler_for(Records(window_spec()), s.module.transform)
    rows, history = s.rows("test"), _history(s, {"A": 4, "B": 4, "C": 4})
    _raises(KeyError, "the history rows do not carry x", lambda: adapter.assemble(
        rows, history.drop(columns="x")))
    _raises(KeyError, "the rows do not carry zone", lambda: adapter.assemble(
        rows.drop(columns="zone"), history))
    _raises(TypeError, "rows is a list", lambda: adapter.assemble([1, 2]))
    _raises(TypeError, "history is a ndarray", lambda: adapter.assemble(rows, np.zeros(3)))
    _raises(ValueError, "no reference graph", lambda: S.assembler_for(
        Records(window_spec()), s.module.transform, bundle("graph-keypair")[2]))
    _raises(TypeError, "transform is a str", lambda: S.assembler_for(Records(window_spec()), "t"))
    stated = Records(window_spec())
    stated.sample = {"shape": "row"}
    _raises(ValueError, "not one bundle's", lambda: S.assembler_for(stated, s.module.transform))
    empty = adapter.assemble(rows, history.iloc[0:0])
    assert np.array_equal(empty.sample_of_row, adapter.assemble(rows).sample_of_row)


# --------------------------------------------------------------------------- rows

def test_row_assembler_is_the_transform_then_the_sample_assembler():
    s = shaped("series")
    spec = row_spec()
    adapter = S.assembler_for(Records(spec), s.module.transform)
    rows = s.rows("test").assign(zone=lambda f: f["zone"].where(f.index != 2, "q"))
    history = pd.concat([s.rows("train"), s.rows("valid")])
    got = adapter.assemble(rows, history)
    shaped_rows = s.module.transform(rows, history=history)
    want = A.SampleAssembler(D.NetSpec.from_dict(spec)).assemble(shaped_rows.reset_index(drop=True))
    assert all(torch.equal(got.batch[k], want.batch[k]) for k in want.batch)
    assert got.sample_of_row.tolist() == list(range(len(rows))) and got.n_samples == len(rows)
    assert got.warnings[2] == ("unseen category in zone",) and got.warnings[0] == ()
    _raises(ValueError, "lag step", lambda: adapter.assemble(rows))  # the transform's refusal


# --------------------------------------------------------------------------- graphs

def _as_held(values):
    """Values as a list, a missing one None (NaN and None alike)."""
    series = pd.Series(values)
    return series.astype(object).where(series.notna(), None).tolist()


def _raw_positions(s):
    return np.concatenate([np.asarray(s.split[p], dtype=np.int64) for p in ("train", "valid")])


def test_reference_files_carry_no_raw_position():
    for name in GRAPHS:
        s, spec = shaped(name), GRAPHS[name]()
        root, meta, _ = bundle(name)
        positions = _raw_positions(s)
        assert not np.array_equal(np.sort(positions), np.arange(len(positions))), name
        nodes = pd.read_parquet(root / "graph" / "reference-nodes.parquet")
        edges = pd.read_parquet(root / "graph" / "reference-edges.parquet")
        features = [f["name"] for f in spec["data_contract"]["features"]]
        extra = [c for c in meta["edges_for_columns"] if c not in features]
        assert list(nodes.columns) == ["ref", *features, *extra], (name, list(nodes.columns))
        assert list(edges.columns) == ["src_ref", "dst_ref", "weight"]
        assert nodes["ref"].tolist() == list(range(len(positions)))
        for path in ("reference-nodes.parquet", "reference-edges.parquet"):
            pandas_meta = json.loads(pq.read_schema(root / "graph" / path).metadata[b"pandas"])
            assert pandas_meta["index_columns"] == [], (name, path)
        for frame in (nodes, edges):
            for column in frame.columns:
                values = frame[column].to_numpy()
                if values.dtype.kind in "iu":
                    assert not set(values.tolist()) >= set(positions.tolist()), (name, column)
        for column, source in (s.manifest["graph"].get("edges_for") or {}).get("raw", {}).items():
            held = s.raw[source].to_numpy()[positions]           # uncast: as the input held it
            assert _as_held(nodes[column]) == _as_held(held), (name, column)
            assert nodes[column].to_numpy().dtype.kind == held.dtype.kind, (name, column)
        assert set(meta) == set(graph_pack.REFERENCE["meta_keys"])
        assert meta["n_nodes"] == len(nodes) == sum(len(s.split[p]) for p in ("train", "valid"))
        assert meta["parts"] == ["train", "valid"] and meta["n_edges"] == len(edges)
        text = json.dumps(meta)
        assert "__split" not in text and "src_pos" not in text


def _sealed_view(s, spec):
    """The evaluation's sealed test view of the same graph (R-A3), built from the shaped parts
    and the edges part — independently of anything pack wrote."""
    order = [np.asarray(s.split[p], dtype=np.int64) for p in ("train", "valid", "test")]
    ids = pd.Series(np.arange(sum(map(len, order))), index=np.concatenate(order))
    part = pd.read_parquet(s.dir / Path(s.manifest["outputs"]["files"]["edges"]["path"]).name)
    weight = part["weight"].to_numpy(dtype="float64", na_value=np.nan)
    edges = D.GraphEdges(src=ids.loc[part["src_pos"]].to_numpy(),
                         dst=ids.loc[part["dst_pos"]].to_numpy(),
                         weight=None if np.isnan(weight).all() else weight)
    view = D.GraphSpec.from_spec(spec).view(tuple(map(len, order)), edges, "test")
    names = [f["name"] for f in spec["data_contract"]["features"]]
    x = pd.concat([s.part(p) for p in ("train", "valid", "test")], ignore_index=True)
    return view, x[names].to_numpy(np.float32)[view.node_ids]


def _edge_set(index, weight):
    w = np.full(index.shape[1], -1.0) if weight is None else np.asarray(weight, np.float32)
    return sorted(zip(index[0].tolist(), index[1].tolist(), np.round(w, 6).tolist()))


def test_served_node_batch_is_the_sealed_test_view():
    for name in NODE_GRAPHS:
        s, spec = shaped(name), GRAPHS[name]()
        got = assembler(name).assemble(s.rows("test"))
        view, x = _sealed_view(s, spec)
        assert np.allclose(got.batch["x_num"].numpy(), x, atol=1e-6), name
        assert torch.equal(got.batch["target_index"], torch.as_tensor(view.target_index))
        served = _edge_set(got.batch["edge_index"].numpy(),
                           got.batch.get("edge_weight") if "edge_weight" in got.batch else None)
        assert served == _edge_set(view.edge_index, view.edge_weight), name
        assert int(got.batch["edge_index"][0].max()) < view.n_reference   # no query sends
        assert got.index.rows == tuple(range(len(s.rows("test"))))


def _network(spec):
    torch.manual_seed(0)
    return N.Network.build(D.NetSpec.from_dict(spec)).eval()


def test_graph_answers_are_identical_alone_and_together():
    for name in NODE_GRAPHS:
        s, model, adapter = shaped(name), _network(GRAPHS[name]()), assembler(name)
        rows = s.rows("test")
        with torch.no_grad():
            together = model(adapter.assemble(rows).batch)
            for q in range(0, len(rows), 7):
                alone = model(adapter.assemble(rows.iloc[[q]]).batch)
                assert torch.equal(alone[0], together[q]), (name, q)
            perm = np.random.default_rng(5).permutation(len(rows))
            moved = model(adapter.assemble(rows.iloc[perm]).batch)
            assert torch.equal(moved, together[torch.as_tensor(perm)]), name
    s, model, adapter = shaped("graph-level"), _network(level_spec()), assembler("graph-level")
    rows = s.rows("test")
    with torch.no_grad():
        whole = adapter.assemble(rows)
        together = model(whole.batch)
        for g, gid in enumerate(whole.index.graphs):
            one = adapter.assemble(rows[rows["molecule_id"].astype(str) == str(gid)])
            assert one.index.graphs == (gid,)
            # a graph-level batch of one small graph runs its dense layers over a handful of
            # rows, which the CPU kernels may sum in another order (last bits); a message
            # leaking between graphs would move the answer by orders more
            assert torch.allclose(model(one.batch)[0], together[g], rtol=0.0, atol=1e-6), gid


def test_graph_level_groups_the_request_by_graph_id():
    s = shaped("graph-level")
    rows = s.rows("test")
    got = assembler("graph-level").assemble(rows)
    ids = rows["molecule_id"].astype(str)
    assert list(map(str, got.index.graphs)) == list(dict.fromkeys(ids))
    assert got.n_samples == got.batch["num_graphs"] == ids.nunique()
    assert np.array_equal(got.sample_of_row, got.batch["graph_index"].numpy())
    names = np.asarray(got.index.graphs, dtype=object)[got.sample_of_row]
    assert list(map(str, names)) == ids.tolist() and got.index.rows == (-1,) * got.n_samples
    src, dst = got.batch["edge_index"].numpy()
    assert (ids.to_numpy()[src] == ids.to_numpy()[dst]).all()


def test_a_graph_sample_takes_no_history_and_a_lonely_row_says_so():
    s = shaped("graph-keypair")
    adapter = assembler("graph-keypair")
    rows = s.rows("test")
    _raises(ValueError, "a graph sample takes no history: the reference graph", lambda:
            adapter.assemble(rows, rows.iloc[:2]))
    _raises(ValueError, "a whole graph", lambda: assembler("graph-level").assemble(
        shaped("graph-level").rows("test"), shaped("graph-level").rows("valid")))
    assert adapter.assemble(rows, rows.iloc[0:0]).n_samples == len(rows)
    lonely = rows.iloc[[0]].assign(bus_key="nobody", feeder_to=np.nan)
    got = adapter.assemble(pd.concat([rows.iloc[[1]], lonely], ignore_index=True))
    assert got.warnings[1] == ("no neighbour in the reference graph — answered from the row's "
                               "own features",) and got.warnings[0] == ()
    assert got.answered.all() and got.n_samples == 2


def test_graph_context_refuses_a_tampered_or_unpinned_reference():
    name = "graph-keypair"
    src, _, _ = bundle(name)
    s, spec = shaped(name), keypair_spec()
    contract = D.NetSpec.from_dict(spec).feature_contract
    root = WORK / "tamper"
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(src, root)
    rels = ["graph/reference-nodes.parquet", "graph/reference-edges.parquet", "graph/graph.json"]
    pins = {rel: digest_file(root / rel) for rel in rels}
    (root / "MANIFEST.json").write_text(json.dumps({"files": pins}), encoding="utf-8")
    assert G.GraphContext.load(root, s.module, contract).n_reference == 480
    nodes = pd.read_parquet(root / rels[0])
    nodes.loc[3, "load_mw"] = 9.5
    nodes.to_parquet(root / rels[0], index=False)
    _raises(ValueError, "does not match its digest", lambda: G.GraphContext.load(
        root, s.module, contract))
    (root / "MANIFEST.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    _raises(ValueError, "is not pinned", lambda: G.GraphContext.load(root, s.module, contract))
    (root / "MANIFEST.json").unlink()
    (root / rels[1]).unlink()
    _raises(FileNotFoundError, "reference-edges.parquet", lambda: G.GraphContext.load(
        root, s.module, contract))
    graph_module = shaped("graph-level").module
    _raises(ValueError, "no edges_for()", lambda: G.GraphContext.load(
        src, type("Recipe", (), {})(), contract))
    _raises(ValueError, "packed for another transform", lambda: G.GraphContext.load(
        src, graph_module, contract))
    _raises(ValueError, "is level node", lambda: S.assembler_for(
        Records(level_spec()), graph_module.transform, bundle(name)[2]))
    _raises(ValueError, "reference graph", lambda: S.assembler_for(
        Records(spec), s.module.transform))


# --------------------------------------------------------------------------- pack

def test_pack_phase_writes_the_reference_and_its_counts():
    for name in GRAPHS:
        spec = GRAPHS[name]()
        ctx = _context(name, spec, name)
        record = graph_pack.write(ctx)["sample.reference"]
        meta = json.loads((ctx.paths.root / "graph" / "graph.json").read_text(encoding="utf-8"))
        stated = output_record.sample_record(spec, {}, shaped(name).manifest["graph"])
        assert (record["nodes"], record["edges"]) == (meta["n_nodes"], meta["n_edges"])
        assert (record["nodes"], record["edges"]) == (stated["reference"]["nodes"],
                                                     stated["reference"]["edges"]), name
        assert record["files"] == stated["reference"]["files"]
        assert ctx.markers[-1].startswith("OK: graph/ — vega/graph-reference@1")
    assert graph_pack.write(_context("series", window_spec(), "window")) == {}


def test_a_transductive_protocol_is_refused():
    code, text = _exit(lambda: graph_pack.write(_context("graph-keypair",
                                                         keypair_spec("transductive"), "tr")))
    assert code == 2 and text.startswith("transductive graph protocol") and \
        text.endswith("/vega-pack v2.3 serves inductive graphs only"), text
    assert not (WORK / "pack" / "tr" / "bundle" / "graph").exists()


def test_pack_refuses_what_would_not_serve_the_graph():
    ctx = _context("graph-keypair", keypair_spec(), "wrong-level")
    shutil.copyfile(shaped("graph-level").module_path, ctx.paths.pkg / "features" / "preprocess.py")
    code, text = _exit(lambda: graph_pack.write(ctx))
    assert code == 3 and "serves no level-node graph through edges_for()" in text, text
    ctx = _context("graph-keypair", keypair_spec(), "columns")
    module = ctx.paths.pkg / "features" / "preprocess.py"
    module.write_text(module.read_text(encoding="utf-8").replace(
        "EDGES_FOR_COLUMNS = ['raw:bus_key', 'raw:feeder_to']",
        "EDGES_FOR_COLUMNS = ['raw:bus_key']"), encoding="utf-8")
    code, text = _exit(lambda: graph_pack.write(ctx))
    assert code == 3 and "edges_for columns" in text, text
    spec = keypair_spec()
    spec["data_contract"]["sample"]["directed"] = True
    code, text = _exit(lambda: graph_pack.write(_context("graph-keypair", spec, "directed")))
    assert code == 2 and "says directed True" in text, text
    s = shaped("graph-keypair")
    manifest = json.loads(json.dumps(s.manifest))
    manifest["outputs"]["files"]["valid"]["digest"] = "sha256:" + "0" * 64
    sources = graph_pack.ReferenceSources(s.dir, manifest, s.split, s.raw, keypair_spec())
    code, text = _exit(lambda: graph_pack.write_reference(WORK / "digest", sources))
    assert code == 2 and "not the file the shaped manifest recorded" in text, text


def test_a_reference_above_the_threshold_warns_and_notes_it():
    saved = graph_pack.WARN_ABOVE
    graph_pack.WARN_ABOVE = 100
    try:
        ctx = _context("graph-knn", knn_spec(), "large")
        graph_pack.write(ctx)
    finally:
        graph_pack.WARN_ABOVE = saved
    meta = json.loads((ctx.paths.root / "graph" / "graph.json").read_text(encoding="utf-8"))
    assert any(m.startswith("WARN: the reference graph holds 480 nodes, above 100")
               for m in ctx.markers), ctx.markers
    assert "480 reference nodes, above 100" in meta["notes"][-1]


# --------------------------------------------------------------------------- module

def test_rendered_modules_hold_the_contract():
    pkg, err = conformance.Package.read(ROOT, RENDERED["blueprint_map"], PIPELINE)
    assert err is None, err
    for check in (conformance.c4_imports_follow_the_arrows,
                  conformance.c7_no_module_level_mutable_state):
        bad = [m.line() for m in check(pkg) if m.level is Level.ERROR]
        assert not bad, bad
    imports = {"samples.py": {(2, "domain"), (2, "model"), (1, "graphs")},
               "graphs.py": {(2, "domain")}}
    publics = {"samples.py": {"Prepared", "HeadOutput", "RowAssembler", "WindowAssembler",
                              "GraphAssembler", "assembler_for", "KeyRefused"},
               "graphs.py": {"GraphContext"}}
    for name, want in imports.items():
        text = (ROOT / "serving" / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        assert conformance.size_findings(tree, f"serving/{name}") == [], name
        assert len(text.splitlines()) <= 400
        inner = {(n.level, n.module) for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.level}
        assert inner == want, (name, inner)
        public = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                  and not n.name.startswith("_")}
        assert public == publics[name], (name, public)
    files = [TEMPLATES / "serving" / "package" / "samples.py",
             TEMPLATES / "serving" / "package" / "graphs.py", PACK / "graph_pack.py"]
    proc = subprocess.run([sys.executable, str(LINT), *map(str, files)], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_head_output_and_prepared_values():
    head = S.HeadOutput(raw=np.arange(3.0), row_score=[1, 2, 3])
    assert head.raw.shape == (3, 1) and head.row_score.dtype == np.float64
    _raises(ValueError, "2 row score(s) for 3", lambda: S.HeadOutput(np.zeros((3, 2)), [1, 2]))
    _raises(ValueError, "(n_samples, head_out)", lambda: S.HeadOutput(np.zeros((2, 2, 2))))


# --------------------------------------------------------------------------- runner

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
        print(f"\n{len(failed)} of {len(tests)} sample test(s) failed")
        return 1
    print(f"\n{len(tests)} sample test(s) passed on the rendered samples.py and graphs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
