#!/usr/bin/env python3
"""
test_vertical_slice.py — milestone M1 (BUILD-PLAN B15, gate A): what the evaluation scores and
what the serving assembler answers are one computation, per `row_id`, on a window forecast, a
node-level graph and a graph-level set.

    python3 -m pytest test_vertical_slice.py [-q] [-k "not slow"]
    python3 test_vertical_slice.py [-k NAME]                   (without pytest)

Each fixture of `make_spike_fixtures.py` (raw rows through the real `shape.py`, a netspec over
the shaped manifest that the compiler's core admits, the package emitted by the v2.3 emitter)
goes through the chain's two halves:

1. **the evaluation**: `Network.build(spec)` → `Trainer().fit(model, spec, ShapedParts())` for
   3 epochs → under a `test_seal.json` as /vega-eval writes it, `Evaluator.score_sealed_test`
   (the metrics), and the same door's part scored row by row (`load_sealed_part` →
   `Evaluator.scored`, whose metrics must be `score_sealed_test`'s) with the raw head beside it
   (`Evaluator.heads`);
2. **the serving side**: the same raw test rows (`fitted.json __split_raw__`) through the emitted
   transform module and the package's `SampleAssembler`, as a Predictor runs them —
   * window: `history ++ rows` transformed as one frame with an empty lag history (R-A6), the
     history being each series' last `history_rows = (L − 1) + lag_depth` rows of the previous
     parts, the target copied from the raw rows as the autoregressive channel (R-A7);
   * node level: the reference graph (train ∪ valid, `vega/graph-reference@1`'s dense `ref` ids,
     the shaped features, the edges among them) and the query links `edges_for` returns (R-A3,
     R-A4) — once from the spike's own reading of R-A4, once from the files vega-pack's
     `graph_pack.write_reference` ships;
   * graph level: the whole test graphs, with `edges_within`'s edges —

   then `model.score` and the raw head on the assembled batch.

The two are compared per `row_id` (the anchor's or the node's raw position, the graph id) at
the round-trip tolerance (`vega-pack/scripts/catalog.json` `roundtrip`: |Δ| ≤ 1e-6 ·
max(1, max|score|), fp32 — the tolerance a bundle's round trip will hold these answers to), on
the score and on every head column. Every stored prediction must be answered. On the window
with a lag step the first `(L − 1) + lag_depth` test rows of each series are R-A6's boundary:
the evaluation's lags restart at the part start and the served ones read the history, so they
are excluded from the comparison — and the rows whose *input windows* differ between the two
sides are exactly that boundary, no more, no fewer; without the lag step the boundary is empty
and every row compares. The node-level views are the same graph (nodes, edges, each query's
in-degree); the graph-level graphs the same graphs.

`test_slow_gcn_beats_the_structure_blind_floor` is the learning check (`slow`: four fits of 30
epochs, a few seconds on a CPU at this size): at 30 epochs the gcn's best valid primary beats
the structure-blind linear floor's (R-A12, the floor as the ladder builds it) on both node-level
fixtures. `-k "not slow"` or `VEGA_SKIP_SLOW=1` skips it. Every fixture is the full emit of its
spec: the file-by-file fallback of before gate A is retired (A29, B18), and `VEGA_SPIKE_EMIT`
(default `full`; `auto` is accepted and means the same) no longer changes what is rendered.
`VEGA_SPIKE_WORK=<dir>` makes and keeps the fixtures there (default: a temporary directory,
removed at exit).
"""
from __future__ import annotations

import atexit
import copy
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_spike_fixtures as msf  # noqa: E402

PACK_SCRIPTS = msf.SKILLS / "vega-pack" / "scripts"
PACK_CATALOG = PACK_SCRIPTS / "catalog.json"
TOKEN = "spike-seal-token"
SLOW_EPOCHS = 30
NODE_FIXTURES = (("node-keypair", "key_pair"), ("node-knn", "knn"))
_WORK: list = []            # the work directory, made once per process
_SLICES: dict = {}          # fixture name -> its Slice (made, emitted, fitted once)
_SIDES: dict = {}           # fixture name -> (evaluated Side, served Side)
_SCRIPT: list = []          # non-empty when run as a script (a skip is then not pytest's)


def tolerance(scores) -> float:
    """The round-trip tolerance (bundle-contract §7) over the evaluation's scores."""
    rule = json.loads(PACK_CATALOG.read_text(encoding="utf-8"))["roundtrip"]
    assert rule["rule"] == "1e-6 * max(1, max|score|)", rule
    top = float(np.nanmax(np.abs(np.asarray(scores, dtype=np.float64)))) if np.size(scores) else 0
    return float(rule["value"]) * max(1.0, top)


# ================================================================ the fixtures, fitted

def work_dir() -> Path:
    if not _WORK:
        named = os.environ.get("VEGA_SPIKE_WORK")
        if named:
            Path(named).mkdir(parents=True, exist_ok=True)
            _WORK.append(Path(named))
        else:
            made = Path(tempfile.mkdtemp(prefix="vega-spike-"))
            atexit.register(shutil.rmtree, made, True)
            _WORK.append(made)
    return _WORK[0]


def _module_of(package: str, rel: str) -> str:
    return f"{package}." + rel.removesuffix(".py").replace("/", ".")


def import_package(fixture: msf.Fixture) -> types.SimpleNamespace:
    """The emitted package's modules, found by their blueprint-map entries (the adapters by
    role, the assembler as the Model module's support file) and imported."""
    root = Path(fixture.package_dir)
    entries = json.loads((root / "blueprint-map.json").read_text(encoding="utf-8"))["entries"]
    by_role = {e["role"]: e["file"] for e in entries if e.get("role")}
    model_mod = by_role["Model"].split("/")[0]
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    importlib.invalidate_caches()
    name = fixture.package
    load = importlib.import_module
    return types.SimpleNamespace(
        domain=load(f"{name}.domain"), network=load(_module_of(name, by_role["Model"])),
        data=load(_module_of(name, by_role["DataSource"])),
        loop=load(_module_of(name, by_role["Trainer"])),
        evaluate=load(_module_of(name, by_role["Evaluator"])),
        assemble=load(f"{name}.{model_mod}.assemble"))


def load_transform(fixture: msf.Fixture):
    """The shaper's transform-only module (it finds fitted.json beside itself)."""
    spec = importlib.util.spec_from_file_location(f"{fixture.package}_preprocess",
                                                  fixture.transform)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Slice:
    """One fixture: made, its package imported, its network fitted (3 epochs)."""

    fixture: msf.Fixture
    pkg: types.SimpleNamespace
    spec: object            # the package's NetSpec
    spec_dict: dict
    model: object
    fit: object             # FitResult
    raw: pd.DataFrame
    split: dict             # __split_raw__, as int64 arrays
    transform: types.ModuleType


def slice_of(name: str) -> Slice:
    if name not in _SLICES:
        fixture = msf.make(name, work_dir(), os.environ.get("VEGA_SPIKE_EMIT", "full"))
        pkg = import_package(fixture)
        spec_dict = json.loads(Path(fixture.spec).read_text(encoding="utf-8"))
        spec = pkg.domain.NetSpec.from_dict(spec_dict)
        model = pkg.network.Network.build(spec)
        fit = pkg.loop.Trainer().fit(model, spec, pkg.data.ShapedParts())
        split = json.loads(Path(fixture.fitted).read_text(encoding="utf-8"))["__split_raw__"]
        _SLICES[name] = Slice(fixture, pkg, spec, spec_dict, model, fit,
                              pd.read_csv(fixture.raw, low_memory=False),
                              {k: np.asarray(v, dtype=np.int64) for k, v in split.items()},
                              load_transform(fixture))
    return _SLICES[name]


def _manifest(sl: Slice) -> dict:
    return json.loads(Path(sl.fixture.manifest).read_text(encoding="utf-8"))


# ================================================================ the evaluation side

@dataclass
class Side:
    """One side's answers: per sample its row id, score and raw head; the batch the samples
    were cut from (the evaluation's: a window part's, gathered whole), the unscored rows, the
    evaluation's part and its row-id kind."""

    row_ids: np.ndarray
    score: np.ndarray
    head: np.ndarray
    batch: dict | None
    unscored: tuple = ()
    part: object = None
    row_id_kind: str = ""

    def by_id(self) -> dict:
        return {_key(r): i for i, r in enumerate(self.row_ids)}


def _key(row_id):
    """A row id as a dict key: a raw position as an int, a graph id as its text."""
    if isinstance(row_id, (np.integer, int)):
        return int(row_id)
    return str(row_id)


def _numpy(value):
    detach = getattr(value, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(value)


def evaluated(sl: Slice) -> Side:
    """The sealed test part, as /vega-eval scores it: the metrics of `score_sealed_test`, and
    the same part through the same door scored row by row (their metrics must agree)."""
    seal = Path(sl.fixture.root) / "test_seal.json"
    seal.write_text(json.dumps({"token": TOKEN, "opened": False}), encoding="utf-8")
    data = sl.pkg.data.ShapedParts()
    evaluator = sl.pkg.evaluate.Evaluator.over(data)
    metrics = evaluator.score_sealed_test(sl.spec, TOKEN, model=sl.model, seal_path=str(seal))
    part = data.load_sealed_part(sl.spec, TOKEN, str(seal))
    scored = evaluator.scored(sl.spec, part, sl.model)
    again = evaluator.measure(sl.spec, sl.pkg.domain.Part.TEST, scored).as_dict()
    assert json.dumps(again, sort_keys=True, default=str) == json.dumps(
        metrics.as_dict(), sort_keys=True, default=str), (again, metrics.as_dict())
    head = evaluator.heads(sl.spec, part, sl.model)
    window = sl.fixture.sample["shape"] == "window"
    batch = part.gather(torch.arange(len(part))) if window else None
    return Side(np.asarray(scored.row_ids), np.asarray(scored.score), np.asarray(head), batch,
                tuple(scored.unscored), part, getattr(scored.row_id_kind, "value",
                                                      str(scored.row_id_kind)))


# ================================================================ the serving side

def _served(sl: Slice, assembled) -> tuple:
    """(score, raw head) of an assembled batch: `model.score`, and one eval forward."""
    model = sl.model
    score = _numpy(model.score(assembled.batch))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            head = _numpy(model(dict(assembled.batch)))
    finally:
        model.train(was)
    return score, head


def _test_rows(sl: Slice) -> pd.DataFrame:
    return sl.raw.iloc[sl.split["test"]].reset_index(drop=True)


def history_rows(sl: Slice) -> int:
    """R-A5's `MANIFEST.sample.history_rows = (length − 1) + lag_depth`."""
    return int(sl.fixture.sample["length"]) - 1 + int(sl.fixture.lag_depth)


def history_for(sl: Slice, rows: pd.DataFrame, per_series: int | None = None) -> pd.DataFrame:
    """R-A6's history: per series, the last `per_series` raw rows (default `history_rows`;
    0: all of them) of the previous parts (train then valid), in `order_by` order."""
    sample = sl.fixture.sample
    need = history_rows(sl) if per_series is None else int(per_series)
    prev = sl.raw.iloc[np.concatenate([sl.split["train"], sl.split["valid"]])]
    series, order = sample.get("series_by"), sample.get("order_by")
    prev = prev.assign(__order=pd.to_datetime(prev[order])).sort_values("__order", kind="stable")
    groups = [g for _, g in prev.groupby(prev[series].astype(str), sort=False)] \
        if series else [prev]
    tails = [g.tail(need) if need else g for g in groups]
    assert not need or all(len(t) == need for t in tails), \
        "a previous part is shorter than the history"
    wanted = set(rows[series].astype(str)) if series else None
    return pd.concat([t for t in tails if wanted is None or
                      str(t[series].iloc[0]) in wanted]).drop(columns="__order")


def served_window(sl: Slice, per_series: int | None = None) -> Side:
    """The test rows as a window Predictor serves them (R-A5, R-A6, R-A7), with
    `history_for(sl, rows, per_series)` as the history."""
    rows = _test_rows(sl)
    history = history_for(sl, rows, per_series).reset_index(drop=True)
    frame = pd.concat([history, rows], ignore_index=True, sort=False)
    shaped = sl.transform.transform(frame, history=frame.iloc[0:0])
    keep = [c for c in (*sl.fixture.keys, sl.fixture.label)
            if c not in shaped.columns and c in frame.columns]
    shaped = shaped.reset_index(drop=True).assign(**{c: frame[c].to_numpy() for c in keep})
    n_h = len(history)
    got = sl.pkg.assemble.SampleAssembler(sl.spec).assemble(
        shaped.iloc[n_h:].reset_index(drop=True), shaped.iloc[:n_h].reset_index(drop=True))
    score, head = _served(sl, got)
    ids = sl.split["test"][np.asarray(got.index.rows, dtype=np.int64)]
    return Side(ids, score, head, got.batch, tuple(got.unscored))


def reference_frames(sl: Slice) -> tuple:
    """vega/graph-reference@1 as R-A4 reads: the train then valid nodes (each part in
    `__split_raw__` order) by dense `ref` id with their shaped features as the network reads
    them, the edges among them by `ref`; and the frame `edges_for` reads (`pos` = `ref`, each
    `raw:<column>` from the input, a fitted column as shaped, float32)."""
    files = _manifest(sl)["outputs"]["files"]
    parts = [pd.read_parquet(files[k]["path"]) for k in ("train", "valid")]
    shaped = pd.concat(parts, ignore_index=True)
    positions = np.concatenate([sl.split["train"], sl.split["valid"]])
    nodes = {"ref": np.arange(len(shaped), dtype=np.int64)}
    for feat in sl.spec_dict["data_contract"]["features"]:
        values = shaped[feat["name"]]
        if feat["role"] == "nominal":
            ids = values.astype("float64").fillna(-1).to_numpy(copy=True)
            ids[(ids < 0) | (ids >= int(feat["cardinality"]))] = int(feat["unseen_category_id"])
            nodes[feat["name"]] = ids.astype(np.int64)
        else:
            nodes[feat["name"]] = values.to_numpy(dtype=np.float32, na_value=np.nan)
    edges = pd.read_parquet(files["edges"]["path"])
    inside = edges["src_part"].isin(["train", "valid"]) & edges["dst_part"].isin(["train",
                                                                                   "valid"])
    kept = edges[inside.to_numpy()]
    ref_of = pd.Series(np.arange(len(positions), dtype=np.int64), index=positions)
    ref_edges = pd.DataFrame({"src_ref": ref_of.loc[kept["src_pos"]].to_numpy(),
                              "dst_ref": ref_of.loc[kept["dst_pos"]].to_numpy(),
                              "weight": kept["weight"].to_numpy(dtype="float64",
                                                                na_value=np.nan)})
    mod = sl.transform
    query_ref = pd.DataFrame({"pos": nodes["ref"]})
    for name in mod.EDGES_FOR_COLUMNS:
        column = mod.GRAPH["reference"]["raw"].get(name)
        query_ref[name] = (sl.raw[column].to_numpy()[positions] if column else
                           shaped[name].to_numpy(dtype=np.float32))
    return pd.DataFrame(nodes), ref_edges, query_ref


def _graph_pack():
    """vega-pack's `graph_pack` (the writer of the reference a graph bundle ships)."""
    if str(PACK_SCRIPTS) not in sys.path:
        sys.path.append(str(PACK_SCRIPTS))
    return importlib.import_module("graph_pack")


def packed_frames(sl: Slice, root: Path) -> tuple:
    """The reference as a bundle carries it: `graph_pack.write_reference` into `root`, read
    back as the serving side's `GraphContext.load` reads it — the nodes and edges files, and
    the frame `edges_for` reads (`ref` renamed `pos`, the columns graph.json names)."""
    gp = _graph_pack()
    split = json.loads(Path(sl.fixture.fitted).read_text(encoding="utf-8"))["__split_raw__"]
    sources = gp.ReferenceSources(Path(sl.fixture.shaped), _manifest(sl), split, sl.raw,
                                  sl.spec_dict)
    meta = gp.write_reference(root, sources)
    nodes = pd.read_parquet(root / gp.FILES["nodes"])
    edges = pd.read_parquet(root / gp.FILES["edges"])
    order = nodes.sort_values("ref", kind="stable").reset_index(drop=True)
    query_ref = order[["ref", *meta["edges_for_columns"]]].rename(columns={"ref": "pos"})
    return nodes, edges, query_ref


def served_nodes(sl: Slice, frames=None) -> Side:
    """The test rows as query nodes against the reference graph (R-A3, R-A4): `frames` is
    (nodes, edges, the frame `edges_for` reads), by default `reference_frames(sl)`."""
    rows = _test_rows(sl)
    nodes, ref_edges, query_ref = frames if frames is not None else reference_frames(sl)
    reference = sl.pkg.domain.GraphReference.from_frames(nodes, ref_edges,
                                                         sl.spec.feature_contract)
    links = sl.transform.edges_for(rows, query_ref)
    shaped = sl.transform.transform(rows).reset_index(drop=True)
    got = sl.pkg.assemble.SampleAssembler(sl.spec, reference).assemble(shaped, None, links)
    score, head = _served(sl, got)
    ids = sl.split["test"][np.asarray(got.index.rows, dtype=np.int64)]
    return Side(ids, score, head, got.batch, tuple(got.unscored))


def served_graphs(sl: Slice) -> Side:
    """The test graphs, whole, with their `edges_within` edges."""
    rows = _test_rows(sl)
    shaped = sl.transform.transform(rows).reset_index(drop=True)
    keep = [c for c in sl.fixture.keys if c not in shaped.columns]
    shaped = shaped.assign(**{c: rows[c].to_numpy() for c in keep})
    got = sl.pkg.assemble.SampleAssembler(sl.spec).assemble(
        shaped, None, sl.transform.edges_within(rows))
    score, head = _served(sl, got)
    return Side(np.asarray(got.index.graphs, dtype=object), score, head, got.batch,
                tuple(got.unscored))


SERVED = {"window": served_window, "node": served_nodes, "graph": served_graphs}


def served(sl: Slice) -> Side:
    sample = sl.fixture.sample
    kind = sample["shape"] if sample["shape"] != "graph" else sample["level"]
    return SERVED[kind](sl)


def sides(name: str) -> tuple:
    """(slice, evaluated, served) of a fixture, each computed once."""
    sl = slice_of(name)
    if name not in _SIDES:
        _SIDES[name] = (evaluated(sl), served(sl))
    return (sl, *_SIDES[name])


# ================================================================ the comparison

@dataclass(frozen=True)
class Agreement:
    compared: int
    excluded: int
    unanswered: tuple
    max_abs_diff: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return not self.unanswered and self.max_abs_diff <= self.tolerance


def agree(ev: Side, sv: Side, excluded=frozenset()) -> Agreement:
    """Every stored prediction answered; score and raw head equal per row id, outside the
    excluded rows, within the round-trip tolerance."""
    at = sv.by_id()
    ids = [_key(r) for r in ev.row_ids]
    unanswered = tuple(r for r in ids if r not in at)
    pairs = [(i, at[r]) for i, r in enumerate(ids) if r in at and r not in excluded]
    worst = 0.0
    for mine, theirs in ((ev.score, sv.score), (ev.head, sv.head)):
        a = np.asarray(mine, dtype=np.float64).reshape(len(ev.row_ids), -1)
        b = np.asarray(theirs, dtype=np.float64).reshape(len(sv.row_ids), -1)
        assert a.shape[1] == b.shape[1], (a.shape, b.shape)
        if pairs:
            e, s = (np.asarray(p, dtype=np.int64) for p in zip(*pairs))
            worst = max(worst, float(np.max(np.abs(a[e] - b[s]))))
    return Agreement(len(pairs), sum(r in excluded for r in ids), unanswered, worst,
                     tolerance(ev.score))


def lag_boundary(sl: Slice) -> dict:
    """R-A6: per series, the first `(L − 1) + lag_depth` test rows (in `order_by` order) when
    the recipe lags; none otherwise. {series: [raw positions]}."""
    sample, depth = sl.fixture.sample, sl.fixture.lag_depth
    if not depth:
        return {}
    rows = sl.raw.iloc[sl.split["test"]].assign(__pos=sl.split["test"])
    rows = rows.assign(__order=pd.to_datetime(rows[sample["order_by"]]))
    n = int(sample["length"]) - 1 + int(depth)
    return {str(k): g.sort_values("__order", kind="stable")["__pos"].head(n).tolist()
            for k, g in rows.groupby(rows[sample["series_by"]].astype(str), sort=False)}


def window_inputs_differ(ev: Side, sv: Side) -> set:
    """The row ids whose input window (`x_num (L, F′)`, `x_cat (L, C)`) is not the same on the
    two sides (float32 values, compared at 1e-6)."""
    at, differ = sv.by_id(), set()
    for i, r in enumerate(ev.row_ids):
        j = at.get(_key(r))
        if j is None:
            continue
        for key in ("x_num", "x_cat"):
            a, b = ev.batch.get(key), sv.batch.get(key)
            if a is None and b is None:
                continue
            a, b = _numpy(a[i]).astype(np.float64), _numpy(b[j]).astype(np.float64)
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-6):
                differ.add(_key(r))
    return differ


def _fail(name: str, sl: Slice, got: Agreement) -> str:
    return (f"{name} ({sl.fixture.emission} emission): {got.compared} compared, "
            f"{got.excluded} excluded, {len(got.unanswered)} unanswered "
            f"{list(got.unanswered[:5])}, max |Δ| {got.max_abs_diff:.3g} > tol {got.tolerance:.3g}")


# ================================================================ the tests

def _ops(sl: Slice) -> list:
    return [s["op"] for s in json.loads(Path(sl.fixture.recipe).read_text(encoding="utf-8"))
            ["steps"]]


def test_the_fixtures_are_the_three_shapes_the_plan_names():
    """(a) 3 series × 300 steps, one nominal, a lag step, forecast H = 3, autoregressive,
    previous-part context; (b) node level, key_pair and knn edges, gcn, inductive; (c) 60
    graphs, gin, graph level — each shaped by the real shape.py, each fitted 3 epochs."""
    w = slice_of("window")
    assert w.raw.groupby("station").size().tolist() == [300, 300, 300]
    nominal = [f["name"] for f in w.spec_dict["data_contract"]["features"]
               if f["role"] == "nominal"]
    assert nominal == ["weather"] and "lag" in _ops(w) and "lag" not in _ops(
        slice_of("window-nolag"))
    s = w.fixture.sample
    assert (s["horizon"], s["autoregressive"], s["context"], s["series_by"]) == (
        3, True, "previous-part", "station") and w.spec_dict["task"]["kind"] == "forecast"
    for name, method in NODE_FIXTURES:
        n = slice_of(name)
        graph = _manifest(n)["graph"]
        assert (graph["method"], graph["level"], n.fixture.sample["protocol"],
                n.spec_dict["family"]["name"]) == (method, "node", "inductive", "gcn")
    g = slice_of("graph-level")
    assert g.raw["molecule_id"].nunique() == 60 and g.spec_dict["family"]["name"] == "gin"
    assert g.fixture.sample["level"] == "graph" and "graph_pool" in [
        b["type"] for b in g.spec_dict["blocks"]]
    for name in msf.FIXTURES:
        sl = slice_of(name)
        assert sl.fit.epochs_run == msf.EPOCHS, (name, sl.fit)
        assert str(_manifest(sl).get("schema", "")).startswith("dataset-shaper/"), name


def test_every_spike_spec_is_one_the_compiler_admits():
    """Each netspec passes compile_spec's core over its pipeline and manifest; the only
    refusal recorded instead of raised is B2's for a block coverage.json still lists as
    planned (A7(a)), so from gate A on — `planned` empty — the compile is clean."""
    planned = msf.planned_blocks()
    for name in msf.FIXTURES:
        sl = slice_of(name)
        manifest = _manifest(sl)
        pipeline = json.loads(Path(sl.fixture.pipeline).read_text(encoding="utf-8"))
        notes = msf.compile_check(sl.spec_dict, manifest, pipeline)
        assert notes == sl.fixture.compile_notes, (name, notes)
        named = {m.group(1) for m in map(msf._PLANNED.search, notes) if m}
        assert len(named) == len(notes) and named <= planned, (name, notes, planned)
        if not planned:
            assert not notes, (name, notes)


def _window_case(name: str) -> tuple:
    sl, ev, sv = sides(name)
    sample = sl.fixture.sample
    per_series = sl.raw.iloc[sl.split["test"]][sample["series_by"]].value_counts()
    reasons = {}
    for u in ev.unscored:
        reasons[u.reason.value] = reasons.get(u.reason.value, 0) + 1
    assert reasons == {"horizon_beyond_part": int(sample["horizon"]) * len(per_series)}, reasons
    assert len(ev.row_ids) == len(sl.split["test"]) - len(ev.unscored)
    assert ev.row_id_kind == "raw_position", ev.row_id_kind
    assert not sv.unscored, f"the history left {len(sv.unscored)} served row(s) unanswered"
    assert len(sv.row_ids) == len(sl.split["test"])
    boundary = lag_boundary(sl)
    excluded = frozenset(p for rows in boundary.values() for p in rows)
    return sl, ev, sv, boundary, excluded


def test_window_with_a_lag_agrees_outside_the_r_a6_boundary():
    """(a): 3 series, a nominal, a lag step (depth 2), forecast H = 3, autoregressive,
    previous-part context. The rows whose input windows differ are exactly the first
    (L − 1) + lag_depth test rows of each series; every other stored row agrees."""
    sl, ev, sv, boundary, excluded = _window_case("window")
    sample, depth = sl.fixture.sample, sl.fixture.lag_depth
    assert depth == 2 and sample["autoregressive"] and sample["context"] == "previous-part"
    expected = int(sample["length"]) - 1 + depth
    assert sorted(len(v) for v in boundary.values()) == [expected] * 3, boundary
    assert window_inputs_differ(ev, sv) == set(excluded), "R-A6's boundary is not what differs"
    got = agree(ev, sv, excluded)
    assert got.excluded == 3 * expected and got.compared == len(ev.row_ids) - got.excluded
    assert got.ok, _fail("window", sl, got)
    x = _numpy(ev.batch["x_num"])
    assert x.shape[1:] == (int(sample["length"]),
                           len(sl.spec.feature_contract.numeric_columns) + 1), x.shape
    assert _numpy(ev.batch["x_cat"]).shape[1:] == (int(sample["length"]), 1)


def test_window_without_the_lag_step_agrees_on_every_row():
    """(a) without the lag step: R-A6's boundary is empty, every stored row agrees."""
    sl, ev, sv, boundary, excluded = _window_case("window-nolag")
    assert sl.fixture.lag_depth == 0 and not boundary and not excluded
    assert not window_inputs_differ(ev, sv)
    got = agree(ev, sv)
    assert got.excluded == 0 and got.compared == len(ev.row_ids)
    assert got.ok, _fail("window-nolag", sl, got)


def test_window_history_rows_is_the_history_a_request_needs():
    """R-A5's `history_rows = (L − 1) + lag_depth` is exactly enough: served with that many
    previous rows per series, every test row gets the answer it gets with the whole previous
    parts as history; one row fewer changes an answer on the lag fixture (the lag_depth term
    is needed: the served lags flow from the history) and leaves a row unanswered without the
    lag step (L − 1 context rows are needed)."""
    for name in ("window", "window-nolag"):
        sl, _ev, sv = sides(name)
        whole = served_window(sl, per_series=0)
        got = agree(whole, sv)
        assert got.compared == len(sl.split["test"]) and got.ok, _fail(name, sl, got)
        short = agree(whole, served_window(sl, per_series=history_rows(sl) - 1))
        assert not short.ok, f"{name}: history_rows − 1 rows answer every row alike"
        if sl.fixture.lag_depth:
            assert not short.unanswered and short.max_abs_diff > short.tolerance, short
        else:       # the first test row of each series lacks one context row
            series = _test_rows(sl)[sl.fixture.sample["series_by"]].nunique()
            assert len(short.unanswered) == series, short


def _in_degree(batch, targets) -> np.ndarray:
    dst = _numpy(batch["edge_index"])[1]
    return np.bincount(dst, minlength=int(np.max(targets)) + 1)[targets]


def _node_case(name: str, method: str) -> None:
    sl, ev, sv = sides(name)
    graph = _manifest(sl)["graph"]
    assert graph["method"] == method and graph["level"] == "node"
    assert sl.fixture.sample["protocol"] == "inductive"
    assert len(ev.row_ids) == len(sl.split["test"]) and not ev.unscored
    assert ev.row_id_kind == "raw_position", ev.row_id_kind
    assert _numpy(ev.part.x_num).shape == _numpy(sv.batch["x_num"]).shape, \
        "the sealed test view and the served graph have different node counts"
    assert _numpy(ev.part.edge_index).shape == _numpy(sv.batch["edge_index"]).shape, \
        "the sealed test view and the served graph have different edge counts"
    at = sv.by_id()
    order = np.asarray([at[_key(r)] for r in ev.row_ids])
    ev_deg = _in_degree({"edge_index": ev.part.edge_index}, _numpy(ev.part.target_index))
    sv_deg = _in_degree(sv.batch, _numpy(sv.batch["target_index"]))
    assert np.array_equal(ev_deg, sv_deg[order]), "a query node's links differ"
    got = agree(ev, sv)
    assert got.compared == len(sl.split["test"]) and got.ok, _fail(name, sl, got)


def test_node_level_key_pair_agrees_per_row():
    """(b), key_pair edges, gcn, inductive: the sealed test view (reference train ∪ valid,
    test nodes attached reference → query) is the served graph, and every node agrees."""
    _node_case("node-keypair", "key_pair")


def test_node_level_knn_agrees_per_row():
    """(b), knn edges (directed, weighted, fitted on train): the same, through the fitted
    index `edges_for` checks against its matrix digest."""
    _node_case("node-knn", "knn")


def _same_values(a: pd.Series, b: pd.Series) -> bool:
    """Equal value for value: numbers exactly (a missing one missing on both sides), any
    other column as text."""
    try:
        x = a.to_numpy(dtype="float64", na_value=np.nan)
        y = b.to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError):
        return len(a) == len(b) and list(map(str, a)) == list(map(str, b))
    return x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=True))


def test_node_level_agrees_through_the_reference_files_pack_writes():
    """(b) again, the reference read from what a graph bundle ships: vega-pack's
    `graph_pack.write_reference` (graph/reference-{nodes,edges}.parquet, graph.json) read back
    as the serving side reads it. The files hold the spike's R-A4 reference, value for value,
    and the answers through them agree with the evaluation per row."""
    for name, _ in NODE_FIXTURES:
        sl, ev, _sv = sides(name)
        with tempfile.TemporaryDirectory(prefix="vega-spike-ref-") as tmp:
            packed = packed_frames(sl, Path(tmp))
            mine = reference_frames(sl)
            extra = [c for c in packed[0].columns if c not in mine[0].columns]
            assert extra == [c for c in sl.transform.EDGES_FOR_COLUMNS
                             if c not in mine[0].columns], (name, extra)
            for label, a, b in zip(("nodes", "edges", "edges_for frame"), mine, packed):
                b = b.drop(columns=extra) if label == "nodes" else b
                assert list(a.columns) == list(b.columns), (name, label, a.columns, b.columns)
                differ = [c for c in a.columns if not _same_values(a[c], b[c])]
                assert not differ, f"{name}: the packed {label} differ in {differ}"
            got = agree(ev, served_nodes(sl, packed))
        assert got.compared == len(sl.split["test"]) and got.ok, _fail(name, sl, got)


def test_graph_level_agrees_per_graph():
    """(c), 60 graphs, gin, graph level: each test graph, collated 8 at a time in the
    evaluation and sent whole in one request, gets the same answer."""
    sl, ev, sv = sides("graph-level")
    gid = sl.fixture.sample["graph_id"]
    test_graphs = set(sl.raw.iloc[sl.split["test"]][gid].astype(str))
    assert ev.row_id_kind == "graph_id", ev.row_id_kind
    assert set(map(str, ev.row_ids)) == test_graphs == set(map(str, sv.row_ids))
    got = agree(ev, sv)
    assert got.compared == len(test_graphs) == 12 and got.ok, _fail("graph-level", sl, got)
    assert int(sv.batch["num_graphs"]) == len(test_graphs)
    edges = pd.read_parquet(_manifest(sl)["outputs"]["files"]["edges"]["path"])
    assert _numpy(sv.batch["edge_index"]).shape[1] == int((edges["src_part"] == "test").sum())


def _variant(sl: Slice, family: dict, blocks: list, epochs: int) -> dict:
    """The spec with another family and blocks (a ladder rung's delta), at `epochs`; a
    regularization entry whose blocks the rung removed goes with them."""
    d = copy.deepcopy(sl.spec_dict)
    d["family"], d["blocks"] = family, blocks
    d["training"] = {**d["training"], "max_epochs": epochs}
    ids = {b["id"] for b in blocks}
    kept = []
    for entry in d["regularization"]:
        where = [w for w in entry.get("where") or () if w in ids]
        if entry.get("where") is not None and not where:
            continue
        kept.append({**entry, **({"where": where} if "where" in entry else {})})
    d["regularization"] = kept
    return d


def _best(sl: Slice, spec_dict: dict) -> float:
    spec = sl.pkg.domain.NetSpec.from_dict(spec_dict)
    model = sl.pkg.network.Network.build(spec)
    return float(sl.pkg.loop.Trainer().fit(model, spec, sl.pkg.data.ShapedParts()).best_value)


def _floor(sl: Slice, epochs: int) -> tuple:
    """(gcn, floor) best valid primaries at `epochs`: the spike's gcn, and the linear floor as
    the ladder builds it (R-A12: the spec's embedding/concat and head only — structure-blind
    at node level), a spec the compiler admits; same data, same budget."""
    gcn = _variant(sl, sl.spec_dict["family"], sl.spec_dict["blocks"], epochs)
    floor = _variant(sl, msf.floor_family(sl.spec_dict["family"]["name"]),
                     msf.floor_blocks(sl.spec_dict["blocks"]), epochs)
    pipeline = json.loads(Path(sl.fixture.pipeline).read_text(encoding="utf-8"))
    msf.compile_check(floor, _manifest(sl), pipeline)
    assert [b["type"] for b in floor["blocks"]] == ["head"], floor["blocks"]
    return _best(sl, gcn), _best(sl, floor)


def _skip_slow() -> bool:
    return os.environ.get("VEGA_SKIP_SLOW") == "1"


def test_slow_gcn_beats_the_structure_blind_floor():
    """At 30 epochs the gcn's best valid primary beats the structure-blind linear floor's on
    both node-level fixtures (`slow`; VEGA_SKIP_SLOW=1 skips it)."""
    if _skip_slow():
        _skip("slow: VEGA_SKIP_SLOW=1")
    for name, _ in NODE_FIXTURES:
        gcn, floor = _floor(slice_of(name), SLOW_EPOCHS)
        assert gcn > floor, f"{name}: gcn {gcn:.4f} does not beat the floor {floor:.4f}"


def _skip(reason: str) -> None:
    if _SCRIPT:
        raise _Skipped(reason)
    import pytest
    pytest.skip(reason)


class _Skipped(Exception):
    """A test skipped outside pytest."""


# ================================================================ without pytest

def main(argv) -> int:
    _SCRIPT.append(True)
    pick = argv[argv.index("-k") + 1] if "-k" in argv else ""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and pick in n]
    failed, skipped = [], []
    for name, fn in tests:
        start = time.monotonic()
        try:
            fn()
            print(f"  ok    {name}  ({time.monotonic() - start:.1f} s)")
        except _Skipped as exc:
            skipped.append(name)
            print(f"  skip  {name}: {exc}")
        except Exception as exc:                        # a failing test is reported, not raised
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    passed = len(tests) - len(failed) - len(skipped)
    print(f"\n{passed} of {len(tests)} spike test(s) passed, {len(skipped)} skipped, "
          f"{len(failed)} failed (fixtures under {work_dir()})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
