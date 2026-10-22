#!/usr/bin/env python3
"""
test_determinism.py — the v2.3 `Trainer` over every epoch source: two fits of one spec are the
same fit, the train-side metrics read the head through `TaskReadout.reduce`, and the CUDA-graph
path replays the step it captured (BUILD-PLAN B12; D1 §10, §17.1).

    python3 -m pytest test_determinism.py [-q]        (or: python3 test_determinism.py)

The templates `loop.py`, `optim.py`, `record.py`, `engines.py` and `graph_engines.py` are
rendered with `render_templates.py`, beside the v2.3 DataSource, Model and metrics (and the
spliced builder, blocks and losses), into one package. The shaped datasets are small and
written the way the shaper writes them (`data_fixtures.write_shaped`). What is proved:

1. **Determinism on the CPU** (backend-contract §1, tolerance 0): two fits of one spec give the
   same history (every field but the wall clock) and the same weights, for a window cell
   (forecast, GRU, a nominal, the autoregressive channel, `previous-part` context — held on the
   host and held resident), and for graph cells at node level (`full`, `sampled`) and at graph
   level (`graphs`); another seed gives another history.
2. **E8**: a binary spec held resident, at learning rate 0 (the weights never move, so the
   epoch's predictions are the initial model's): the train-side `f1` and `accuracy` are those of
   the initial logits read as logits (`z > 0`), not of an argmax over one column (every row class
   0, v2.2's resident reduction); the streamed pass gives the same numbers. **E9**: for every
   effective kind, the accumulator's label metrics from `reduce` equal the metrics of the score
   they were reduced from; a metric the reduction cannot carry is reported unavailable; a
   reduction whose per-sample shape is not the reading's `reduced_shape()` is refused.
3. **The run context and the record**: the checkpoint goes through the ModelStore port at every
   improvement; `on_step` every `grad_sample_every` steps with the pre-clip norm; a budget stops
   the run at the epoch boundary; `onecycle` steps once per batch of the source's own count
   (streamed and held); the epoch sources the dispatch table picks, and what they record in
   `shuffle` (the window and graph `sampling` facts, the ignored `cuda_graph` hint on a graph);
   a graph pass out of memory is an error naming `sample.batching` (R-A11); `evaluate` and
   `predict` on a held part (row ids included).
4. **The CUDA-graph path** on a window cell (resident + `cuda_graph`), when CUDA is present
   (skipped with the reason otherwise): the step is captured and replayed (`shuffle.graph`), the
   history agrees with the eager resident pass within the CUDA tolerance, and a per-step
   schedule is refused by name; on a row cell (the CIC R7 posture) the replayed steps are
   counted and sampled every k steps like eager ones, and reach the train metrics.
5. **The size discipline**: the rendered trainer files meet C11–C13, and `loop.py`'s top level
   is C14's (the one class `Trainer`, `_prefixed` helpers, imports, constants).
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
for _p in (str(HERE), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_fixtures as fx  # noqa: E402
import render_templates as rt  # noqa: E402

PACKAGE = "b12_trainer"
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
RENDERED = ("domain.py", "network.py", "readout.py", "metrics.py", "frames.py", "windows.py",
            "graphs.py", "data.py", "loop.py", "optim.py", "record.py", "engines.py",
            "graph_engines.py")
TRAINER_FILES = ("loop.py", "optim.py", "record.py", "engines.py", "graph_engines.py")
CUDA_TOLERANCE = 1e-4          # relative, captured vs eager on one CUDA device
CLASSES = ["a", "b", "c", "d"]


# ================================================================ the rendered package

def _purge(prefix):
    for name in [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]:
        del sys.modules[name]


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    """Every template a fit reads, rendered with a spec that uses every block type (so every
    block file is spliced), imported."""
    out = tmp_path_factory.mktemp("b12-trainer")
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = PACKAGE
    wide = rt.reference_spec("row")
    wide["blocks"] = [{"id": f"W{i}", "type": t, "params": {}}
                      for i, t in enumerate(CATALOG["blocks"])]
    rec = rt.render(TEMPLATES, out, pipeline, wide, only=list(RENDERED), splice=True)
    assert {f"training/{n}" for n in TRAINER_FILES} <= set(rec["files"]), rec["files"]
    sys.path.insert(0, str(out))
    importlib.invalidate_caches()
    try:
        mods = {"domain": "domain", "network": "model.network", "readout": "model.readout",
                "metrics": "model.metrics", "data": "ingestion.data", "loop": "training.loop",
                "optim": "training.optim", "losses": "training.losses", "record": "training.record",
                "engines": "training.engines", "graph_engines": "training.graph_engines"}
        ns = types.SimpleNamespace(root=Path(rec["root"]),
                                   **{k: importlib.import_module(f"{PACKAGE}.{v}")
                                      for k, v in mods.items()})
        yield ns
    finally:
        sys.path.remove(str(out))
        _purge(PACKAGE)


class _Collect:
    def __init__(self):
        self.records, self.first, self.steps = [], [], []

    def on_epoch(self, record):
        self.records.append(record.as_dict())

    def on_first_step(self, record):
        self.first.append(dict(record))

    def on_step(self, record):
        self.steps.append(dict(record))


def fit(pkg, spec: dict, run=None, model=None, store=None):
    """One fit: (records less the wall clock, FitResult, the model, the callbacks)."""
    ns = pkg.domain.NetSpec.from_dict(spec)
    model = model if model is not None else pkg.network.Network.build(ns)
    collect = _Collect()
    result = pkg.loop.Trainer(store).fit(model, ns, pkg.data.ShapedParts(), callbacks=collect,
                                         run=run)
    records = [{k: v for k, v in r.items() if k != "time_s"} for r in collect.records]
    return records, result, model, collect


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in sa)


def _finite(records, *paths):
    for r in records:
        for part, key in paths:
            value = r[part][key]
            assert value is not None and math.isfinite(value), (r["epoch"], part, key, value)


# ================================================================ the shaped datasets

def _common(spec, epochs=2, primary=None, mode=None):
    spec["training"]["max_epochs"] = epochs
    spec["training"]["device"] = "cpu"
    if primary:
        spec["metrics"]["primary"] = primary
        spec["early_stopping"]["monitor"] = f"valid.{primary}"
    if mode:
        spec["early_stopping"]["mode"] = mode
    return spec


def row_fixture(tmp) -> dict:
    """200 rows: four numerics, one nominal, a binary label from a noisy linear rule."""
    rng = np.random.default_rng(11)
    n = 200
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    frame["zone"] = rng.integers(0, 5, size=n)
    score = frame.x0 - 0.5 * frame.x1 + 0.3 * (frame.zone == 2) + rng.normal(0, 0.5, n)
    frame["y"] = (score > 0.3).astype(np.int64)
    perm = rng.permutation(n)
    parts = {"train": perm[:120], "valid": perm[120:160], "test": perm[160:]}
    man = fx.write_shaped(tmp / "rows", {k: frame.iloc[v] for k, v in parts.items()}, parts)
    spec = rt.reference_spec("row")
    spec["data_contract"]["features"] = [fx.numeric(f"x{i}") for i in range(4)] + \
        [fx.nominal("zone", 5)]
    spec["data_contract"]["input_dim"] = 5
    spec["metrics"]["secondary"] = ["f1", "accuracy", "roc_auc"]
    spec["regularization"] = [{"kind": "grad_clip", "value": 1.0}]
    spec["training"]["batch_size"] = 16
    return _common(fx.spec_on(spec, man))


def window_fixture(tmp) -> dict:
    """Three series × 60 steps, time-split 40/10/10: a forecast of H = 2 over L = 6, a nominal,
    the autoregressive channel, previous-part context, mase against the train seasonal scale."""
    rng = np.random.default_rng(5)
    rows = []
    for t in range(60):
        for s in range(3):
            rows.append({"series": f"s{s}", "ts": t, "x0": math.sin(t / 4 + s),
                         "x1": float(s) + rng.normal(0, 0.1), "zone": (t + s) % 4,
                         "target": math.sin(t / 4 + s) * (1 + s) + rng.normal(0, 0.05)})
    frame = pd.DataFrame(rows)
    pos = np.arange(len(frame))
    parts = {"train": pos[frame.ts < 40], "valid": pos[(frame.ts >= 40) & (frame.ts < 50)],
             "test": pos[frame.ts >= 50]}
    parts = {k: np.random.default_rng(3).permutation(v) for k, v in parts.items()}
    man = fx.write_shaped(tmp / "windows", {k: frame.iloc[v] for k, v in parts.items()}, parts)
    spec = rt.reference_spec("window")
    dc = spec["data_contract"]
    dc["features"] = [fx.numeric("x0"), fx.numeric("x1"), fx.nominal("zone", 4)]
    dc["input_dim"] = 3
    dc["sample"].update({"length": 6, "stride": 1, "horizon": 2, "season": 4})
    spec["task"]["horizon"] = 2
    spec["blocks"] = [
        {"id": "E", "type": "embedding", "params": {"columns": ["zone"], "dims": {"zone": 2}}},
        {"id": "C", "type": "concat", "params": {}},
        {"id": "G", "type": "gru", "params": {"hidden": 8, "return": "last"}},
        {"id": "H", "type": "head", "params": {"out": 2, "activation": "none"}}]
    spec["metrics"]["secondary"] = ["rmse", "mase"]
    spec["training"]["batch_size"] = 16
    return _common(fx.spec_on(spec, man), primary="mae", mode="min")


def _community_frame(n=90):
    rng = np.random.default_rng(2)
    community = np.arange(n) % 4
    frame = pd.DataFrame({f"x{i}": rng.normal(community * (i + 1) * 0.3, 1.0) for i in range(3)})
    frame["community"] = [CLASSES[c] for c in community]
    return frame, community


def node_fixture(tmp, batching="full") -> dict:
    """90 nodes of four communities, mostly intra-community edges (both directions)."""
    frame, community = _community_frame()
    n = len(frame)
    perm = np.random.default_rng(4).permutation(n)
    parts = {"train": perm[:54], "valid": perm[54:72], "test": perm[72:]}
    part_of = {int(p): name for name, pos in parts.items() for p in pos}
    rng, pairs = np.random.default_rng(6), set()
    while len(pairs) < 220:
        a, b = (int(v) for v in rng.integers(0, n, size=2))
        if a != b and (community[a] == community[b] or rng.random() < 0.15):
            pairs.add((min(a, b), max(a, b)))
    edges = pd.DataFrame([{"src_pos": s, "dst_pos": d, "src_part": part_of[s],
                           "dst_part": part_of[d], "weight": None}
                          for a, b in sorted(pairs) for s, d in ((a, b), (b, a))])
    man = fx.write_shaped(tmp / f"nodes-{batching}",
                          {k: frame.iloc[v] for k, v in parts.items()}, parts, edges)
    spec = rt.reference_spec("graph")
    spec["data_contract"]["features"] = [fx.numeric(f"x{i}") for i in range(3)]
    spec["data_contract"]["input_dim"] = 3
    spec["blocks"] = [
        {"id": "B1", "type": "gcn_conv", "params": {"out": 16}},
        {"id": "B2", "type": "activation", "params": {"name": "relu"}},
        {"id": "B3", "type": "dropout", "params": {"p": 0.1}},
        {"id": "B4", "type": "gcn_conv", "params": {"out": 16}},
        {"id": "B5", "type": "head", "params": {"out": 4, "activation": "none"}}]
    sample = spec["data_contract"]["sample"]
    sample["batching"] = batching
    if batching == "sampled":
        sample["neighbours"] = [4, 3]
        spec["training"]["batch_size"] = 16
    else:
        spec["training"]["batch_size"] = "full"
    spec["metrics"]["secondary"] = ["accuracy"]
    return _common(fx.spec_on(spec, man))


def graphs_fixture(tmp) -> dict:
    """40 graphs of 5–8 nodes: a cycle (label 1) or a chain (label 0), split by whole graphs."""
    rng = np.random.default_rng(8)
    rows, edges, parts = [], [], {"train": [], "valid": [], "test": []}
    for g in range(40):
        size, label = int(rng.integers(5, 9)), g % 2
        part = "train" if g < 24 else "valid" if g < 32 else "test"
        first = len(rows)
        for k in range(size):
            rows.append({"molecule": f"m{g}", "x0": rng.normal(), "x1": float(k) / size,
                         "x2": rng.normal(label, 0.5), "y": label})
            parts[part].append(first + k)
        links = [(first + k, first + k + 1) for k in range(size - 1)]
        links += [(first + size - 1, first)] if label else []
        for a, b in links:
            edges += [{"src_pos": s, "dst_pos": d, "src_part": part, "dst_part": part,
                       "weight": None} for s, d in ((a, b), (b, a))]
    frame = pd.DataFrame(rows)
    parts = {k: np.asarray(v) for k, v in parts.items()}
    man = fx.write_shaped(tmp / "graphs", {k: frame.iloc[v] for k, v in parts.items()}, parts,
                          pd.DataFrame(edges))
    spec = rt.reference_spec("graph-level")
    spec["data_contract"]["features"] = [fx.numeric(f"x{i}") for i in range(3)]
    spec["data_contract"]["input_dim"] = 3
    spec["blocks"] = [
        {"id": "B1", "type": "gin_conv", "params": {"out": 16}},
        {"id": "B2", "type": "graph_pool", "params": {"readout": "mean"}},
        {"id": "B3", "type": "head", "params": {"out": 1, "activation": "none"}}]
    spec["training"]["batch_size"] = 8
    spec["metrics"]["secondary"] = ["accuracy"]
    return _common(fx.spec_on(spec, man))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b12-data")
    return types.SimpleNamespace(row=row_fixture(tmp), window=window_fixture(tmp),
                                 full=node_fixture(tmp, "full"),
                                 sampled=node_fixture(tmp, "sampled"),
                                 graphs=graphs_fixture(tmp))


def _with(spec, **changes):
    """A deep copy of `spec` with `section.key` changes (`backend.hints` takes a dict)."""
    spec = copy.deepcopy(spec)
    for dotted, value in changes.items():
        section, _, key = dotted.partition("__")
        spec.setdefault(section, {})[key] = value
    return spec


# ================================================================ 1. determinism on the CPU

CELLS = {
    "window-held-host": ("window", {}),
    "window-held-resident": ("window", {"resident_tensors": True}),
    "graph-node-full": ("full", {}),
    "graph-node-sampled": ("sampled", {}),
    "graph-level": ("graphs", {}),
}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_two_cpu_fits_are_one_fit(pkg, data, cell):
    name, hints = CELLS[cell]
    spec = _with(getattr(data, name), backend__hints=hints)
    a, ra, ma, _ = fit(pkg, spec)
    b, rb, mb, _ = fit(pkg, spec)
    assert len(a) == 2 and a == b, cell
    assert (ra.best_epoch, ra.best_value, ra.stopped_by) == \
        (rb.best_epoch, rb.best_value, rb.stopped_by)
    assert _same_weights(ma, mb), cell
    _finite(a, ("train", "loss"), ("valid", "loss"))
    other, _, _, _ = fit(pkg, dict(spec, seed=spec["seed"] + 1))
    assert other[0]["train"]["loss"] != a[0]["train"]["loss"], f"{cell}: the seed moves nothing"


def test_the_window_record_names_its_source_and_samples(pkg, data):
    host, _, _, _ = fit(pkg, data.window)
    shuffle = host[0]["shuffle"]
    assert shuffle["kind"] == "global-randperm" and shuffle["device"] == "cpu"
    assert shuffle["seed"] == data.window["seed"] + 1 and shuffle["graph"] is False
    # per series: 40 train rows, the first L−1 = 5 without context (no part before train) and
    # the last H = 2 whose horizon runs past the part's end are unscored; 33 windows remain
    assert shuffle["sampling"] == {"windows": 99, "unscored": 21} and shuffle["rows"] == 99
    assert set(host[0]["train"]) >= {"loss", "mae", "rmse", "mase"}
    assert host[0]["valid"]["mase"] is not None and host[0]["train"]["mase"] is not None


def test_the_graph_records_name_their_sources(pkg, data):
    full, _, _, _ = fit(pkg, _with(data.full, backend__hints={"cuda_graph": True}))
    assert full[0]["shuffle"]["kind"] == "full-graph"
    assert full[0]["shuffle"]["capture"] == "not applicable to graph samples"
    facts = full[0]["shuffle"]["sampling"]
    assert set(facts) == {"targets", "nodes", "edges"}
    assert facts["targets"] == 54 and facts["nodes"] == 54 and facts["edges"] > 0
    sampled, _, _, _ = fit(pkg, data.sampled)
    s = sampled[0]["shuffle"]
    assert s["kind"] == "neighbour-sampled" and s["neighbours"] == [4, 3]
    assert set(s["sampling"]) == {"targets", "nodes", "edges"}
    assert s["sampling"]["targets"] == 54 and s["sampling"]["nodes"] >= 54
    graphs, _, _, _ = fit(pkg, data.graphs)
    g = graphs[0]["shuffle"]
    assert g["kind"] == "graph-minibatches"
    # backend-contract §1's graph facts; at graph level the targets are the train graphs
    assert set(g["sampling"]) == {"targets", "nodes", "edges"} and g["sampling"]["targets"] == 24
    assert g["sampling"]["nodes"] > 24 and g["sampling"]["edges"] > 0
    assert "capture" not in g


# ================================================================ 2. E8 and E9

def test_binary_resident_train_metrics_read_logits_e8(pkg, data):
    """At lr 0 the epoch's predictions are the initial model's; `z > 0`, not argmax."""
    spec = _with(data.row, optimizer__params={"lr": 0.0, "weight_decay": 0.0},
                 backend__hints={"resident_tensors": True}, training__max_epochs=1)
    spec["blocks"] = [b for b in spec["blocks"] if b["type"] != "dropout"]
    ns = pkg.domain.NetSpec.from_dict(spec)
    initial = pkg.network.Network.build(ns)
    part = pkg.data.ShapedParts().load_part(ns, pkg.domain.Part.TRAIN)
    with torch.no_grad():
        logits = initial.eval()(part.gather(torch.arange(len(part)))).reshape(-1).numpy()
    y = part.y.numpy()
    decided = (logits > 0).astype(int)
    assert 0 < decided.sum() < len(decided), "the initial model must predict both classes"
    task = pkg.domain.Task.from_spec(ns)
    want = {n: pkg.metrics.metric(n, y, logits, None, task) for n in ("f1", "accuracy")}
    assert want["f1"] > 0.0                     # v2.2's argmax over one column gives 0
    resident, _, _, _ = fit(pkg, spec)
    streamed, _, _, _ = fit(pkg, _with(spec, backend__hints={}))
    for records in (resident, streamed):
        train = records[0]["train"]
        assert {n: train[n] for n in want} == want
        assert train["_unavailable"]["roc_auc"] == pkg.record.UNAVAILABLE
    assert resident[0]["shuffle"]["kind"] == "global-randperm"
    assert streamed[0]["shuffle"]["kind"] == "buffered"


def _spec_for(task, loss, features=None, sample=None):
    spec = rt.reference_spec("row")
    spec["task"], spec["loss"] = task, loss
    if features is not None:
        spec["data_contract"]["features"] = features
        spec["data_contract"]["input_dim"] = len(features)
    if sample is not None:
        spec["data_contract"]["sample"] = sample
    return spec


class _Centred:
    """A model stub for the svdd reading: the centre `objective_center()` returns."""

    def __init__(self, centre):
        self._centre = centre

    def objective_center(self):
        return self._centre


READINGS = {
    "binary": ({"kind": "binary", "label": "y", "classes": [0, 1]},
               {"name": "bce_with_logits"}, 1, ["f1", "accuracy", "mcc", "roc_auc"]),
    "multiclass": ({"kind": "multiclass", "label": "y", "classes": CLASSES},
                   {"name": "cross_entropy"}, 4, ["f1_macro", "accuracy", "balanced_accuracy",
                                                  "log_loss"]),
    "multilabel": ({"kind": "multilabel", "label": "y", "classes": ["p", "q", "r"],
                    "label_format": {"kind": "delimited", "sep": "|"}},
                   {"name": "bce_with_logits"}, 3, ["hamming", "subset_accuracy", "pr_auc"]),
    "ordinal": ({"kind": "ordinal", "label": "y", "classes": CLASSES},
                {"name": "cumulative_link"}, 3, ["accuracy", "mae_ordinal", "qwk", "f1_macro"]),
    "regression": ({"kind": "regression", "label": "y"}, {"name": "mse"}, 1, ["rmse", "mae"]),
    "anomaly": ({"kind": "anomaly", "label": "y", "positive": "1"}, {"name": "svdd"}, 4,
                ["roc_auc", "pr_auc", "anomaly_score_mean"]),
}


def _truth_of(name, n, g):
    if name == "multilabel":
        return (torch.rand(n, 3, generator=g) > 0.5).float()
    if name == "regression":
        return torch.randn(n, generator=g)
    k = {"binary": 2, "anomaly": 2}.get(name, 4)
    return torch.randint(0, k, (n,), generator=g)


@pytest.mark.parametrize("name", sorted(READINGS))
def test_the_accumulator_reads_what_the_score_decides_e9(pkg, name):
    task, loss, width, names = READINGS[name]
    ns = pkg.domain.NetSpec.from_dict(_spec_for(task, loss))
    readout = pkg.readout.TaskReadout.for_spec(ns)
    g = torch.Generator().manual_seed(3)
    model = _Centred(torch.randn(width, generator=g))
    acc = pkg.record.EpochAccumulator(readout, names)
    outs, truths = [], []
    for _ in range(3):                                  # three steps of one epoch
        out, truth = torch.randn(17, width, generator=g) * 2, _truth_of(name, 17, g)
        acc.update(readout.reduce(out, {}, model), truth)
        outs.append(out)
        truths.append(truth)
    got = acc.metrics()
    score = readout.score(torch.cat(outs), {}, model).numpy()
    truth = torch.cat(truths).numpy()
    label_only = [n for n in names if n in readout.label_only()]
    want, _ = pkg.metrics.compute(label_only, truth, score, None, readout.task)
    assert label_only and {n: got[n] for n in label_only} == want, name
    for n in set(names) - set(label_only):
        assert got["_unavailable"][n] == pkg.record.UNAVAILABLE, (name, n)


def test_the_unsupervised_accumulator_reads_the_reconstruction_error(pkg):
    feats = [fx.numeric(f"x{i}") for i in range(4)]
    ns = pkg.domain.NetSpec.from_dict(_spec_for({"kind": "unsupervised"},
                                                {"name": "reconstruction"}, features=feats))
    readout = pkg.readout.TaskReadout.for_spec(ns)
    acc = pkg.record.EpochAccumulator(readout, ["reconstruction_error"])
    g = torch.Generator().manual_seed(4)
    xs, outs = [], []
    for _ in range(2):
        x, out = torch.randn(9, 4, generator=g), torch.randn(9, 4, generator=g)
        acc.update(readout.reduce(out, {"x_num": x}, None), None)
        xs.append(x)
        outs.append(out)
    got = acc.metrics()["reconstruction_error"]
    want = pkg.metrics.metric("reconstruction_error", torch.cat(xs).numpy(),
                              readout.score(torch.cat(outs), {}, None).numpy())
    assert got == pytest.approx(want, rel=1e-6)


def test_the_accumulator_refuses_a_reduction_of_another_shape(pkg):
    """A step's reduction is checked against `readout.reduced_shape()` before it is kept: a
    forecast of H = 2 reduces each window to (2,), so a (B,) or a (B, 3) reduction is refused
    rather than scored on the wrong axis; tensor and array steps join into one epoch."""
    spec = rt.reference_spec("window")
    spec["task"]["horizon"] = spec["data_contract"]["sample"]["horizon"] = 2
    readout = pkg.readout.TaskReadout.for_spec(pkg.domain.NetSpec.from_dict(spec))
    assert readout.reduced_shape() == (2,)
    acc = pkg.record.EpochAccumulator(readout, ["mae"])
    for bad in (torch.zeros(5), torch.zeros(5, 3)):
        with pytest.raises(ValueError, match=r"reduces each sample to shape \(2,\)"):
            acc.update(bad, torch.zeros(5, 2))
    acc.update(torch.ones(3, 2), torch.zeros(3, 2))
    acc.update(np.full((2, 2), 3.0, dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
    assert acc.metrics()["mae"] == pytest.approx((3 * 2 * 1.0 + 2 * 2 * 3.0) / 10)


# ================================================================ 3. the run context and the record

class _Store:
    """The ModelStore port: records what it is asked to save."""

    def __init__(self):
        self.saved = []

    def save(self, model, path, spec=None, epoch=None):
        self.saved.append((path, epoch))
        return f"checkpoint@{epoch}"


def test_the_run_context_is_honoured(pkg, data, tmp_path):
    store = _Store()
    run = pkg.domain.RunContext(checkpoint_dir=str(tmp_path), grad_sample_every=3)
    spec = _with(data.row, training__max_epochs=3)
    records, result, model, collect = fit(pkg, spec, run=run, store=store)
    improved = [r["epoch"] for r in records if r["early_stopping"]["since_best"] == 0]
    assert [e for _, e in store.saved] == improved
    assert all(p == f"{tmp_path}/best.pt" for p, _ in store.saved)
    assert result.checkpoint == f"checkpoint@{result.best_epoch}"
    assert result.best_valid.as_dict()["pr_auc"] == result.best_value
    epochs, steps_per_epoch = 3, -(-120 // 16)
    assert [s["step"] for s in collect.steps] == list(range(3, epochs * steps_per_epoch + 1, 3))
    assert all(s["grad_norm"] > 0 and s["lr"] is not None for s in collect.steps)
    assert [s["epoch"] for s in collect.steps][:2] == [1, 1]
    assert len(collect.first) == 1 and collect.first[0]["block"] == model.ids[-1]
    budget = pkg.domain.RunContext(budget_min=1e-9)
    records, result, _, _ = fit(pkg, spec, run=budget)
    assert len(records) == 1 and result.stopped_by.value == "budget"


def test_a_monitor_naming_no_metric_is_refused(pkg, data):
    spec = _with(data.row, early_stopping__monitor="valid.nothing")
    with pytest.raises(ValueError, match="names 'nothing'"):
        fit(pkg, spec)


@pytest.mark.parametrize("name,hints", [("row", {}), ("window", {}),
                                        ("row", {"resident_tensors": True})])
def test_onecycle_steps_once_per_batch_of_the_source(pkg, data, name, hints):
    spec = _with(getattr(data, name), schedule__name="onecycle",
                 schedule__params={"max_lr": 0.01}, backend__hints=hints)
    records, _, _, _ = fit(pkg, spec)
    assert len(records) == 2 and records[1]["lr"] != records[0]["lr"]


def test_the_dispatch_table(pkg, data):
    open_source, cpu = pkg.engines.open_source, torch.device("cpu")
    reader = pkg.data.ShapedParts()

    def kind(name, hints):
        spec = pkg.domain.NetSpec.from_dict(getattr(data, name))
        return type(open_source(spec, reader, cpu, hints)).__name__

    assert kind("row", {}) == "StreamedRows"
    assert kind("row", {"resident_tensors": True}) == "HeldSamples"
    assert kind("row", {"resident_tensors": True, "cuda_graph": True}) == "HeldSamples"
    assert kind("window", {}) == "HeldSamples"
    assert kind("full", {}) == "FullGraph" and kind("sampled", {}) == "SampledGraph"
    assert kind("graphs", {"resident_tensors": True}) == "GraphMinibatches"
    held = open_source(pkg.domain.NetSpec.from_dict(data.window), reader, cpu, {})
    assert held.zero_to_none and not held.capturable and held.steps_per_epoch == 7


def test_a_graph_pass_out_of_memory_names_the_batching(pkg, data):
    """R-A11: a view that does not fit the device's memory is an error naming
    `sample.batching`, not a crash; any other runtime error passes through as it was."""
    spec = pkg.domain.NetSpec.from_dict(data.full)
    source = pkg.engines.open_source(spec, pkg.data.ShapedParts(), torch.device("cpu"))
    for message in ("CUDA out of memory. Tried to allocate 2.00 GiB",
                    "DefaultCPUAllocator: can't allocate memory: you tried to allocate 8 GB"):
        with pytest.raises(MemoryError, match=r"data_contract\.sample\.batching") as caught:
            with source.guard():
                raise RuntimeError(message)
        assert "54 train nodes" in str(caught.value)
    with pytest.raises(RuntimeError, match="a shape mismatch"):
        with source.guard():
            raise RuntimeError("a shape mismatch")


def test_evaluate_and_predict_read_a_held_part(pkg, data):
    ns = pkg.domain.NetSpec.from_dict(data.window)
    model = pkg.network.Network.build(ns)
    trainer = pkg.loop.Trainer()
    part = pkg.data.ShapedParts().load_part(ns, pkg.domain.Part.VALID)
    scored = trainer.predict(model, ns, part)
    score, truth = scored.score, scored.truth
    assert isinstance(scored, pkg.domain.ScoredPart)
    assert score.shape == (len(part), 2) and truth.shape == (len(part), 2)
    assert np.array_equal(scored.row_ids, np.asarray(part.row_ids))
    values = trainer.evaluate(model, ns, part, pkg.losses.build_loss(ns))
    assert values["mae"] == pytest.approx(float(np.mean(np.abs(score - truth))), rel=1e-6)
    assert values["mase"] is None and "mase" in values["_unavailable"]   # no train-side scale
    assert math.isfinite(values["loss"])
    streamed = trainer.evaluate(model, ns, part.slices(5), pkg.losses.build_loss(ns))
    assert streamed["mae"] == pytest.approx(values["mae"], rel=1e-6)


class _Counting:
    """A DataSource that counts how often each part is read whole or streamed."""

    def __init__(self, inner):
        self._inner, self.reads = inner, []

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name in ("iter_batches", "load_part"):
            def counted(spec, part, *a, **k):
                self.reads.append((name, str(getattr(part, "value", part))))
                return attr(spec, part, *a, **k)
            return counted
        return attr


def test_a_part_one_buffer_holds_is_read_once_and_permuted_as_streamed(pkg, data):
    """Final rehearsal round 2 (improve, 3 of 3 audits): the streamed source re-read and
    re-cut the whole part every epoch although one shuffle buffer held it. It is read once
    now, and every epoch's batches are the ones the stream draws — the same generator, the
    same permutation, the same tensors (bit for bit; the golden history holds it too)."""
    D, cpu = pkg.domain, torch.device("cpu")
    spec = D.NetSpec.from_dict(data.row)
    stream = pkg.data.ShapedParts()
    counting = _Counting(pkg.data.ShapedParts())
    source = pkg.engines.StreamedRows(spec, counting, cpu)
    for epoch in (1, 2, 3):
        held = list(source.train_batches(epoch))
        plan = D.BatchPlan(shuffle=True, seed=int(spec.seed + epoch),
                           shuffle_buffer=source._buffer)
        streamed = list(stream.iter_batches(spec, D.Part.TRAIN, batch_size=16, plan=plan))
        assert len(held) == len(streamed) > 1
        for a, b in zip(held, streamed):
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    valid = [list(source.valid_batches()) for _ in range(3)]
    want = list(stream.iter_batches(spec, D.Part.VALID, batch_size=16))
    for got in valid:
        assert all(torch.equal(a[k], b[k]) for a, b in zip(got, want) for k in a)
    assert counting.reads == [("load_part", "train"), ("iter_batches", "valid")]
    # a part larger than one buffer stays streamed, epoch after epoch
    small = D.NetSpec.from_dict(_with(data.row, training__shuffle_buffer=32))
    counting = _Counting(pkg.data.ShapedParts())
    source = pkg.engines.StreamedRows(small, counting, cpu)
    for epoch in (1, 2):
        list(source.train_batches(epoch))
    assert counting.reads == [("iter_batches", "train")] * 2


def test_a_small_network_fits_on_one_cpu_thread(pkg, data):
    """Final rehearsal round 2 (improve, 2 of 3 audits): a CPU fit of a small network ran on
    one thread per core, per-op dispatch costing more than the op. `cpu_threads` gives the fit
    one thread (or the spec's `backend.hints.cpu_threads`) and restores the process's count."""
    spec = pkg.domain.NetSpec.from_dict(data.row)
    model, before = pkg.network.Network.build(spec), torch.get_num_threads()
    with pkg.engines.cpu_threads(spec, model, "cpu") as threads:
        assert threads == torch.get_num_threads() == 1
    assert torch.get_num_threads() == before
    hinted = pkg.domain.NetSpec.from_dict(_with(data.row, backend__hints={"cpu_threads": 2}))
    with pkg.engines.cpu_threads(hinted, model, "cpu") as threads:
        assert threads == torch.get_num_threads() == 2
    assert torch.get_num_threads() == before
    with pkg.engines.cpu_threads(spec, model, "meta") as threads:
        assert threads == before == torch.get_num_threads()     # another device: untouched
    # the fit says which count it ran on (part of what the run is), and leaves none set
    result = pkg.loop.Trainer().fit(model, spec, pkg.data.ShapedParts())
    assert result.threads == 1 and torch.get_num_threads() == before


def test_the_shared_primitives_are_defined_once(pkg):
    """Final rehearsal round 2 (improve, 2-3 of 3 audits): the host-array conversion, the
    batch placement, a module's device, a file's digest and the shuffle-buffer default were
    each written in several modules, three sets already divergent. `domain` holds each once."""
    import re
    root = Path(pkg.root)
    copies = {"host": r"detach\(\)\.cpu\(\)\.numpy\(\)", "digest": r"\.read\(1 << 20\)",
              "buffer": r"BUFFER_BATCHES *= *\d",
              "device": r"next\(iter\(\w+\.parameters\(\)\)|\.parameters\(\), \w+\.buffers\(\)"}
    for what, pattern in copies.items():
        where = sorted(str(f.relative_to(root)) for f in root.rglob("*.py")
                       if re.search(pattern, f.read_text(encoding="utf-8")))
        assert where == ["domain.py"], (what, where)


def test_the_trainer_modules_support_functions(pkg, data):
    """The device, the learning rates and the loss are the Trainer module's support functions
    (optim, losses), the ones its loop calls; the port carries no static copy of them."""
    ns = pkg.domain.NetSpec.from_dict(data.row)
    assert pkg.optim.device_for(ns) == torch.device("cpu")
    assert pkg.optim.resolved_lrs(ns) == [pytest.approx(0.001)] * 2
    assert type(pkg.losses.build_loss(ns)).__name__ == "BCEWithLogits"
    assert not {"device_for", "lrs_for", "loss_for"} & set(vars(pkg.loop.Trainer))


# ================================================================ 4. the CUDA-graph path

def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available: the CUDA-graph path (CapturedSteps) needs a CUDA "
                    "device; the CPU cells above run the same step eagerly")
    try:
        torch.zeros(1, device="cuda").sum().item()
    except RuntimeError as exc:                  # a device present but unusable (held, broken)
        pytest.skip(f"CUDA is present but not usable here: {exc}")


def test_the_cuda_graph_path_on_a_window_cell(pkg, data):
    _cuda_or_skip()
    base = _with(data.window, training__device="cuda")
    captured, rc, _, cc = fit(pkg, _with(base, backend__hints={"resident_tensors": True,
                                                               "cuda_graph": True}))
    eager, re, _, _ = fit(pkg, _with(base, backend__hints={"resident_tensors": True}))
    assert [r["shuffle"]["graph"] for r in captured] == [True, True]
    assert [r["shuffle"]["graph"] for r in eager] == [False, False]
    assert captured[0]["shuffle"]["device"].startswith("cuda")
    for got, want in zip(captured, eager):
        for part, key in (("train", "loss"), ("valid", "loss"), ("train", "mae"),
                          ("valid", "mae")):
            assert got[part][key] == pytest.approx(want[part][key], rel=CUDA_TOLERANCE), \
                (got["epoch"], part, key)
    assert len(cc.first) == 1 and rc.epochs_run == re.epochs_run == 2
    onecycle = _with(base, schedule__name="onecycle", schedule__params={"max_lr": 0.01},
                     backend__hints={"resident_tensors": True, "cuda_graph": True})
    with pytest.raises(ValueError, match="per-step schedule cannot run under"):
        fit(pkg, onecycle)


def test_the_cuda_graph_path_on_a_row_cell_samples_its_replays(pkg, data):
    """Rows held resident with `cuda_graph` (the CIC R7 posture): the replayed steps are
    counted, sampled every k steps with their pre-clip norm and update ratio, and their reduced
    predictions reach the train metrics."""
    _cuda_or_skip()
    base = _with(data.row, training__device="cuda", training__max_epochs=2)
    run = pkg.domain.RunContext(grad_sample_every=2)
    captured, _, _, cc = fit(pkg, _with(base, backend__hints={"resident_tensors": True,
                                                               "cuda_graph": True}), run=run)
    eager, _, _, ce = fit(pkg, _with(base, backend__hints={"resident_tensors": True}), run=run)
    assert [r["shuffle"]["graph"] for r in captured] == [True, True]
    steps = 2 * -(-120 // 16)                     # 7 full batches of 16 and a tail of 8, twice
    assert [s["step"] for s in cc.steps] == [s["step"] for s in ce.steps] == \
        list(range(2, steps + 1, 2))
    assert all(math.isfinite(s["grad_norm"]) and s["grad_norm"] > 0 for s in cc.steps)
    assert captured[1]["update_ratio"] and all(v is None or v > 0
                                              for v in captured[1]["update_ratio"].values())
    for got, want in zip(captured, eager):
        for part, key in (("train", "loss"), ("valid", "loss"), ("valid", "roc_auc")):
            assert got[part][key] == pytest.approx(want[part][key], rel=CUDA_TOLERANCE), \
                (got["epoch"], part, key)
        assert 0.0 <= got["train"]["accuracy"] <= 1.0
        assert got["train"]["_unavailable"]["roc_auc"] == pkg.record.UNAVAILABLE


# ================================================================ 5. the size discipline

def _tree(pkg, name):
    return ast.parse((pkg.root / "training" / name).read_text(encoding="utf-8"))


def test_the_rendered_trainer_files_meet_c11_to_c13(pkg):
    import conformance
    for name in TRAINER_FILES:
        findings = conformance.size_findings(_tree(pkg, name), f"training/{name}")
        assert findings == [], [f.message() for f in findings]


def test_loop_py_is_the_trainer_class_alone(pkg):
    import conformance
    tree = _tree(pkg, "loop.py")
    adapter = conformance.adapter_class(tree, "Trainer", "Trainer")
    assert adapter is not None and adapter.name == "Trainer"
    findings = [f for node in tree.body
                for f in conformance._top_level_findings("training/loop.py", node, adapter,
                                                         "Trainer")]
    assert findings == [], [f.message() for f in findings]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
