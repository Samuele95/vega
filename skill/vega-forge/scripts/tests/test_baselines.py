#!/usr/bin/env python3
"""
test_baselines.py — the Evaluator adapter and the baselines of generation v2.3
(`templates/evaluate.py`, `baselines.py`, `baselines_extra.py`), proved on the rendered package
(build plan B13; backend design §3.4, §3.5, §11.1, §11.3; R-A3, R-A5, R-C2, R-C3).

    python3 -m pytest test_baselines.py [-q]
    python3 test_baselines.py [-v] [-k <substring>]

The package is rendered once with `render_templates.py` (the live v2.3 templates: domain, the
four ingestion files, network, readout, metrics, the two baseline files, evaluate, store; the
live blocks, builder and losses spliced), over shaped fixtures written the way the shaper
writes them (`data_fixtures.py`). What is proved:

1. **The score contract, per task** (§3.4; logits everywhere, R-C3): binary `(N,)` logits
   whose σ is the fitted probability, multiclass `(N, K)` log-probabilities over every class
   (one absent from train included), multilabel `(N, K)` per-label logits (a label constant in
   train by its prevalence), ordinal `(N, K−1)` cumulative logits `logit(P(y > j))` whose
   readout gives back the class probabilities, regression `(N,)` values (the quantile head's
   median column the same), a reconstruction `(N, n_numeric)` with R-C2's per-row score
   (numeric MSE + mean nominal cross-entropy), anomaly `(N,)` per-row scores (the mean-row and
   PCA errors, the isolation forest, boosted trees when labelled; a svdd spec's baselines
   still reconstruct) — `build` hands a reconstruction or anomaly task the reconstruction
   forms of the trio under the trio's names.
2. **Windows**: trivial / linear / tree `(N, H)`; `naive` is the anchor's target repeated and
   `seasonal-naive` the target one season back, both read from the autoregressive channel or
   — without it — from the part's labels appended by `sample_view`, with identical numbers; a
   sequence regression repeats the row before the anchor; a perturbed future (or the anchor's
   own label) never moves a naive prediction; a sequence classification reads the flattened
   window.
3. **Graphs**: node level — `linear` / `tree` read the target's own features (structure-blind),
   `graph-linear` reads `[x ‖ mean of the in-neighbours' x]` and beats the blind floor on a
   graph whose labels live in the neighbourhood; graph level — `[mean | max | sum]` pooled
   graphs, the trivial prior counted over graphs, not nodes.
4. **One design**: every baseline's batch design equals `ShapedParts.design_matrix` row for row
   (flat rows and windows, flat and neighbour-mean nodes, pooled graphs over several batches).
5. **`Evaluator.scored`, per shape**: `row_ids` are the raw positions (rows, window anchors,
   graph targets) or the graph ids (with `graph_ids` beside them), `row_id_kind` says which,
   `unscored` carries the window part's reasons; the network and a baseline give the same ids
   and the same truth; `row_score` is `TaskReadout.row_score` of the network's head
   (unsupervised) and the score itself (anomaly, svdd included), None otherwise; `heads` is
   the raw head (every head_out column, R-C2's stored `score_k`) in the same order, of which
   the score contract reads a part (a quantile's median, the numeric reconstruction, the
   distance to the svdd centre).
6. **`score_part` / `score_sealed_test` / `measure`**: the test part only through the seal and
   its token; a restored checkpoint scores as the network in memory; an unavailable metric is
   recorded under `_unavailable`; `mase` reads the train target's seasonal scale (at the
   spec's season unless the caller names one), or the caller's scale; refusals (a fitted baseline's `why`, an unknown baseline, a checkpoint without the
   store) name the offender.
7. **The size discipline**: the rendered `evaluate.py`, `baselines.py` and `baselines_extra.py`
   meet C11–C13, and `evaluate.py`'s top level is C14's (the `Evaluator` class, helpers,
   imports, constants); the Evaluator adapter imports along the blueprint's arrows (C4) and
   no adapter (C5).
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data_fixtures as fx

PKG = "b13_eval"
CATALOG = json.loads((fx.SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
ONLY = ("domain.py", "frames.py", "windows.py", "graphs.py", "data.py", "network.py",
        "readout.py", "metrics.py", "baselines.py", "baselines_extra.py", "evaluate.py",
        "store.py")
MODULES = {"domain": "domain", "data": "ingestion.data", "network": "model.network",
           "readout": "model.readout", "metrics": "model.metrics", "baselines": "model.baselines",
           "extra": "model.baselines_extra", "evaluate": "evaluation.evaluate",
           "store": "model.store"}


def _render():
    """The v2.3 templates B13 reads and writes, rendered as package `b13_eval` and imported;
    the spec that drives the splice uses every block type, so every block file is there."""
    work = tempfile.TemporaryDirectory(prefix="b13-render-")
    fx._KEEP.append(work)
    pipeline = fx.rt.reference_pipeline()
    pipeline["system"]["package"] = PKG
    wide = fx.rt.reference_spec("row")
    wide["blocks"] = [{"id": f"W{i}", "type": t, "params": {}}
                      for i, t in enumerate(CATALOG["blocks"])]
    rec = fx.rt.render(fx.TEMPLATES, Path(work.name), pipeline, wide, only=list(ONLY),
                       splice=True)
    sys.path.insert(0, work.name)
    importlib.invalidate_caches()
    ns = types.SimpleNamespace(root=Path(rec["root"]), pipeline=pipeline,
                               blueprint_map=rec["blueprint_map"])
    for key, dotted in MODULES.items():
        setattr(ns, key, importlib.import_module(f"{PKG}.{dotted}"))
    return ns


NS = _render()
D, B, X, E = NS.domain, NS.baselines, NS.extra, NS.evaluate
DV = D.DesignView


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def scratch(prefix: str) -> Path:
    """A temporary directory that lives as long as the run (data_fixtures keeps it)."""
    work = tempfile.TemporaryDirectory(prefix=prefix)
    fx._KEEP.append(work)
    return Path(work.name)


def made(spec, name, context=None):
    return X.build(spec, (name,), context)[0]


class Probe:
    """A baseline and what its fit gave (a `FittedBaseline` or a `RefusedBaseline`), read the
    way these tests compare numbers: on a design matrix (`score_matrix`, `row_score_matrix`),
    through the reading's probabilities (`predict_proba`), by the fitted state (`model`). The
    baseline itself scores batches only, through the Evaluator; everything else is forwarded
    to the outcome, `design_of` to the recipe."""

    def __init__(self, recipe, outcome):
        self.recipe, self.outcome = recipe, outcome

    def __getattr__(self, name):
        return getattr(self.outcome, name)

    def design_of(self, data, part):
        return self.recipe.design_of(data, part)

    def _read(self, x, reading):
        if not self.outcome.fitted:
            raise RuntimeError(f"the {self.outcome.name} baseline was not fitted: "
                               f"{self.outcome.why}")
        return self.outcome._read(np.asarray(x, dtype=np.float64), reading)

    @property
    def model(self):
        return self.outcome._fitted.state

    def score_matrix(self, x):
        return self._read(x, lambda r: r.score)

    def row_score_matrix(self, x):
        return self._read(x, lambda r: r.row_score)

    def predict_proba(self, batch):
        return self._read(B.design_rows(batch, self.outcome.view), lambda r: r.proba)


def fitted(spec, sp, name, context=None, part="train"):
    """`name` built for `spec` and fitted on its own design of the part."""
    baseline = made(spec, name, context)
    return Probe(baseline, baseline.fit(baseline.design_of(sp, sp.load_part(spec, part))))


def evaluator(sp, options=None):
    return E.Evaluator.over(sp, options)


# ================================================================ fixtures: rows

N_ROWS = 150
ROW_FEATURES = [fx.numeric("x0"), fx.numeric("x1"), fx.numeric("x2"), fx.nominal("zone", 5)]
ROW_TASKS = {
    "binary": ({"kind": "binary", "label": "flag", "classes": [0, 1], "positive": 1},
               "bce_with_logits", 1, "pr_auc"),
    "multiclass": ({"kind": "multiclass", "label": "colour", "classes": ["a", "b", "c", "d"]},
                   "cross_entropy", 4, "f1_macro"),
    "multilabel": ({"kind": "multilabel", "label": "tags", "classes": ["a", "b", "c"],
                    "label_format": {"kind": "delimited", "sep": "|"}},
                   "bce_with_logits", 3, "pr_auc_macro"),
    "ordinal": ({"kind": "ordinal", "label": "grade", "classes": ["lo", "mid", "hi"]},
                "cumulative_link", 2, "mae_ordinal"),
    "regression": ({"kind": "regression", "label": "amount"}, "mse", 1, "rmse"),
    "unsupervised": ({"kind": "unsupervised"}, "reconstruction", 8, "reconstruction_error"),
    "anomaly": ({"kind": "anomaly"}, "reconstruction", 8, "anomaly_score_mean"),
    "anomaly-labelled": ({"kind": "anomaly", "label": "odd", "positive": 1}, "reconstruction", 8,
                         "pr_auc"),
    "anomaly-svdd": ({"kind": "anomaly"}, "svdd", 4, "anomaly_score_mean"),
}


def row_frame(n: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(n, 3))
    signal = x[:, 0] + 0.5 * x[:, 1]
    return pd.DataFrame({
        "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "zone": rng.integers(0, 4, size=n),
        "flag": (signal + 0.3 * rng.normal(size=n) > 0).astype(int),
        "colour": np.array(["a", "b", "c"])[np.digitize(signal, [-0.5, 0.5])],
        "tags": ["|".join(t for t, on in zip("ab", (x[i, 0] > 0, x[i, 1] > 0)) if on)
                 for i in range(n)],
        "grade": np.array(["lo", "mid", "hi"])[np.digitize(signal, [-0.4, 0.6])],
        "amount": 2 * x[:, 0] - x[:, 2] + 0.1 * rng.normal(size=n),
        "odd": (np.abs(x[:, 2]) > 1.5).astype(int)})


def row_parts(n: int = N_ROWS) -> dict:
    perm = np.random.default_rng(22).permutation(n)
    return {"train": perm[:90], "valid": perm[90:120], "test": perm[120:]}


def row_fixture(key: str, loss=None, secondary=(), frame=None, out=None):
    """(ShapedParts, NetSpec, shaped dir, (frame, parts)) for one of ROW_TASKS; `out` overrides
    the head's width (a quantile head)."""
    task, default_loss, width, primary = ROW_TASKS[key]
    out = width if out is None else out
    frame = row_frame() if frame is None else frame
    parts = row_parts()
    root = scratch("b13-rows-")
    man = fx.write_shaped(root, {k: frame.iloc[v] for k, v in parts.items()}, parts)
    spec = fx.rt.reference_spec("row")
    spec["task"] = dict(task)
    spec["loss"] = loss or {"name": default_loss, "params": {}}
    spec["metrics"] = {"primary": primary, "secondary": list(secondary)}
    spec["data_contract"]["features"] = ROW_FEATURES
    spec["data_contract"]["input_dim"] = 4
    spec["blocks"] = [
        {"id": "B1", "type": "embedding", "params": {"columns": ["zone"], "dims": {"zone": 3}}},
        {"id": "B2", "type": "concat", "params": {}},
        {"id": "B3", "type": "dense", "params": {"units": 16, "activation": "relu"}},
        {"id": "B4", "type": "head", "params": {"out": out, "activation": "none"}}]
    spec = D.NetSpec.from_dict(fx.spec_on(spec, man))
    return NS.data.ShapedParts(), spec, root, (frame, parts)


# ================================================================ fixtures: windows

L, H, SEASON = 4, 3, 2
T_TRAIN, T_VALID, T_END, N_SERIES = 24, 32, 40, 3


def raw_of(series: int, t: int) -> int:
    """The raw position of series `series` at step `t` (rows are time-major)."""
    return int(t) * N_SERIES + int(series)


def window_frame() -> pd.DataFrame:
    rows = [{"series": f"s{s}", "ts": t, "x1": float(s), "zone": (t + s) % 4, "level": t % 3}
            for t in range(T_END) for s in range(N_SERIES)]
    frame = pd.DataFrame(rows)
    frame.insert(2, "x0", np.arange(len(frame), dtype=np.float64))
    t, s = frame.ts.to_numpy(), frame.x1.to_numpy()
    frame["target"] = np.sin(t / 3.0) * (1.0 + s) + 0.1 * t + s
    frame["up"] = (frame["target"] > frame["target"].median()).astype(int)
    return frame


def window_split(frame) -> dict:
    rng = np.random.default_rng(3)
    pos = np.arange(len(frame))
    parts = {"train": pos[frame.ts < T_TRAIN],
             "valid": pos[(frame.ts >= T_TRAIN) & (frame.ts < T_VALID)],
             "test": pos[frame.ts >= T_VALID]}
    return {k: rng.permutation(v) for k, v in parts.items()}


def window_fixture(frame=None, task=None, nominal=True, loss=None, **sample):
    """A forecast (H steps, autoregressive, season 2) over three series; `task` replaces it
    with a sequence task (no horizon, no channel); `sample` overrides."""
    frame = window_frame() if frame is None else frame
    parts = window_split(frame)
    root = scratch("b13-windows-")
    man = fx.write_shaped(root, {k: frame.iloc[v] for k, v in parts.items()}, parts)
    spec = fx.rt.reference_spec("window")
    dc = spec["data_contract"]
    dc["features"] = [fx.numeric("x0"), fx.numeric("x1")] + (
        [fx.nominal("zone", 5)] if nominal else [])
    dc["input_dim"] = len(dc["features"])
    dc["sample"].update({"length": L, "stride": 1, "horizon": H, "series_by": "series",
                         "order_by": "ts", "context": "previous-part", "autoregressive": True,
                         "season": SEASON})
    spec["metrics"] = {"primary": "mae", "secondary": ["mase"]}
    if loss:
        spec["loss"] = loss
    if task is not None:
        spec["task"] = task
        dc["sample"].pop("horizon", None)
        dc["sample"]["autoregressive"] = False
        spec["blocks"][-1]["params"]["out"] = 1
    dc["sample"].update(sample)
    return NS.data.ShapedParts(), D.NetSpec.from_dict(fx.spec_on(spec, man)), root, frame


def at(frame, anchors, offsets) -> np.ndarray:
    """The target of each anchor's series `offsets[h]` steps from the anchor (float32, as the
    shaped part holds it)."""
    target = frame.target.to_numpy(dtype=np.float32)
    out = [[target[raw_of(a % N_SERIES, a // N_SERIES + o)] for o in offsets] for a in anchors]
    return np.asarray(out, dtype=np.float64)


# ================================================================ fixtures: graphs

N_BEACONS, N_TARGETS = 40, 150
CONVS = ("gcn_conv", "sage_conv", "gat_conv", "gin_conv")


def graph_spec(shape: str) -> dict:
    """render_templates' reference graph spec, its convs' width under `out`, the name the
    builder reads (the reference specs still write `hidden`; B14 owns them)."""
    spec = fx.rt.reference_spec(shape)
    for block in spec["blocks"]:
        if block["type"] in CONVS and "out" not in block["params"]:
            block["params"]["out"] = block["params"].pop("hidden")
    return spec


def beacon_fixture():
    """A node-level graph whose labels live in the neighbourhood: a target's own feature `s`
    is noise, and it is linked to two beacons of its class (beacons: s = ±2). Beacons are all
    train; the targets split 90 / 30 / 30."""
    rng = np.random.default_rng(31)
    n = N_BEACONS + N_TARGETS
    klass = np.concatenate([np.arange(N_BEACONS) % 2, rng.integers(0, 2, size=N_TARGETS)])
    s = np.where(np.arange(n) < N_BEACONS, np.where(klass == 1, 2.0, -2.0), 0.0)
    s = s + rng.normal(scale=np.where(np.arange(n) < N_BEACONS, 0.1, 1.0))
    frame = pd.DataFrame({"x0": rng.normal(size=n), "s": s,
                          "community": np.where(klass == 1, "pos", "neg")})
    targets = rng.permutation(np.arange(N_BEACONS, n))
    parts = {"train": np.concatenate([np.arange(N_BEACONS), targets[:90]]),
             "valid": targets[90:120], "test": targets[120:]}
    part_of = {int(p): k for k, v in parts.items() for p in v}
    rows = []
    for t in range(N_BEACONS, n):
        mates = np.flatnonzero(klass[:N_BEACONS] == klass[t])
        for b in rng.choice(mates, size=2, replace=False):
            for a, c in ((int(b), t), (t, int(b))):
                rows.append({"src_pos": a, "dst_pos": c, "src_part": part_of[a],
                             "dst_part": part_of[c], "weight": None})
    edges = pd.DataFrame(rows)
    root = scratch("b13-nodes-")
    man = fx.write_shaped(root, {k: frame.iloc[v] for k, v in parts.items()}, parts, edges)
    spec = graph_spec("graph")
    spec["task"] = {"kind": "binary", "label": "community", "classes": ["neg", "pos"],
                    "positive": "pos"}
    spec["loss"] = {"name": "bce_with_logits", "params": {}}
    spec["metrics"] = {"primary": "roc_auc", "secondary": []}
    spec["blocks"][-1]["params"]["out"] = 1
    spec["data_contract"]["features"] = [fx.numeric("x0"), fx.numeric("s")]
    spec["data_contract"]["input_dim"] = 2
    spec = D.NetSpec.from_dict(fx.spec_on(spec, man))
    return NS.data.ShapedParts(), spec, root, (frame, parts, edges)


N_GRAPHS = 30


def molecule_fixture():
    """30 graphs of 3–5 nodes (chains), 20 / 5 / 5 by graph; a graph's label shifts its
    nodes' `x1`."""
    rng = np.random.default_rng(41)
    rows = []
    for g in range(N_GRAPHS):
        for _ in range(3 + g % 3):
            rows.append({"molecule": f"m{g:02d}", "x1": rng.normal() + 1.5 * (g % 2),
                         "y": g % 2})
    frame = pd.DataFrame(rows)
    frame.insert(1, "x0", np.arange(len(frame), dtype=np.float64))
    graph_of = frame.molecule.to_numpy()
    mol = sorted(set(graph_of))
    part_of = {m: ("train" if i < 20 else "valid" if i < 25 else "test")
               for i, m in enumerate(mol)}
    pos = np.arange(len(frame))
    parts = {p: pos[[part_of[m] == p for m in graph_of]] for p in ("train", "valid", "test")}
    rows = []
    for m in mol:
        members = np.flatnonzero(graph_of == m)
        for a, b in zip(members, members[1:]):
            for s, d in ((a, b), (b, a)):
                rows.append({"src_pos": int(s), "dst_pos": int(d), "src_part": part_of[m],
                             "dst_part": part_of[m], "weight": None})
    root = scratch("b13-graphs-")
    man = fx.write_shaped(root, {k: frame.iloc[v] for k, v in parts.items()}, parts,
                          pd.DataFrame(rows))
    spec = graph_spec("graph-level")
    spec["data_contract"]["features"] = [fx.numeric("x0"), fx.numeric("x1")]
    spec["data_contract"]["input_dim"] = 2
    spec["training"]["batch_size"] = 4
    spec = D.NetSpec.from_dict(fx.spec_on(spec, man))
    return NS.data.ShapedParts(), spec, root, (frame, parts)


# ================================================================ 1. the score contract, rows

def test_row_scores_follow_the_score_contract():
    widths = {"binary": (), "multiclass": (4,), "multilabel": (3,), "ordinal": (2,),
              "regression": ()}
    for key, tail in widths.items():
        sp, spec, _, _ = row_fixture(key)
        valid = sp.load_part(spec, "valid")
        for name in ("trivial", "linear", "tree"):
            baseline = fitted(spec, sp, name)
            assert baseline.fitted, (key, name, baseline.why)
            scored = evaluator(sp).scored(spec, valid, baseline)
            assert scored.score.shape == (30,) + tail, (key, name, scored.score.shape)
            assert np.isfinite(scored.score).all() and scored.row_score is None
    sp, spec, _, _ = row_fixture("binary")
    trivial = fitted(spec, sp, "trivial")
    score = trivial.score_matrix(trivial.design_of(sp, sp.load_part(spec, "valid")).x)
    prior = np.mean(sp.load_part(spec, "train").y.numpy())
    assert np.allclose(score, logit(prior)), "the trivial binary score is the prior's logit"
    assert not np.allclose(score, prior), "a binary baseline scores a logit, not a probability"


def test_binary_scores_are_logits_of_the_fitted_probabilities():
    sp, spec, _, _ = row_fixture("binary")
    valid = sp.load_part(spec, "valid")
    for name in ("linear", "tree"):
        baseline = fitted(spec, sp, name)
        x = baseline.design_of(sp, valid).x
        p = baseline.model.predict_proba(x.astype(np.float64))[:, 1]
        score = baseline.score_matrix(x)
        assert np.allclose(sigmoid(score), clip(p), atol=1e-9), name
        batch = next(iter(sp.make_loader(valid, spec, batch_size=len(valid))))
        assert np.allclose(baseline.predict_proba(batch), clip(p), atol=1e-9)
        assert np.allclose(baseline.score(batch), score)


def test_multiclass_log_probabilities_cover_every_class():
    sp, spec, _, _ = row_fixture("multiclass")
    valid = sp.load_part(spec, "valid")
    train_y = sp.load_part(spec, "train").y.numpy()
    assert 3 not in set(train_y.tolist()), "class d must be absent from train"
    for name in ("trivial", "linear", "tree"):
        baseline = fitted(spec, sp, name)
        x = baseline.design_of(sp, valid).x
        score = baseline.score_matrix(x)
        assert score.shape == (30, 4) and np.allclose(score[:, 3], np.log(1e-7)), name
        proba = np.exp(score)
        assert np.allclose(proba[:, :3].sum(axis=1) + proba[:, 3], 1.0, atol=1e-5), name
        if name != "trivial":
            assert np.array_equal(score.argmax(axis=1),
                                  baseline.model.predict(x.astype(np.float64))), name
    trivial = fitted(spec, sp, "trivial")
    want = np.bincount(train_y, minlength=4) / len(train_y)
    assert np.allclose(trivial.score_matrix(np.zeros((2, 4)))[0], np.log(clip(want)))


def test_multilabel_prevalence_and_a_constant_label():
    sp, spec, _, _ = row_fixture("multilabel")
    valid = sp.load_part(spec, "valid")
    train_y = sp.load_part(spec, "train").y.numpy()
    assert train_y[:, 2].max() == 0, "label c is constant (never on) in train"
    trivial = fitted(spec, sp, "trivial")
    score = trivial.score_matrix(trivial.design_of(sp, valid).x)
    assert score.shape == (30, 3)
    assert np.allclose(sigmoid(score[0]), clip(train_y.mean(axis=0)), atol=1e-9)
    for name in ("linear", "tree"):
        baseline = fitted(spec, sp, name)
        assert baseline.fitted and isinstance(baseline.model[2], float), name
        x = baseline.design_of(sp, valid).x
        score = baseline.score_matrix(x)
        assert score.shape == (30, 3) and np.allclose(score[:, 2], logit(0.0)), name
        on = baseline.model[0].predict_proba(x.astype(np.float64))[:, 1]
        assert np.allclose(sigmoid(score[:, 0]), clip(on), atol=1e-9), name


def test_ordinal_cumulative_logits_give_back_the_class_probabilities():
    sp, spec, _, _ = row_fixture("ordinal")
    valid = sp.load_part(spec, "valid")
    train_y = sp.load_part(spec, "train").y.numpy()
    prior = np.bincount(train_y, minlength=3) / len(train_y)
    trivial = fitted(spec, sp, "trivial")
    x = trivial.design_of(sp, valid).x
    score = trivial.score_matrix(x)
    assert score.shape == (30, 2)
    assert np.allclose(sigmoid(score[0]), [prior[1] + prior[2], prior[2]], atol=1e-9)
    batch = next(iter(sp.make_loader(valid, spec, batch_size=len(valid))))
    assert np.allclose(trivial.predict_proba(batch)[0], prior, atol=1e-6)
    for name in ("linear", "tree"):
        baseline = fitted(spec, sp, name)
        score = baseline.score_matrix(x)
        assert (np.diff(sigmoid(score), axis=1) <= 1e-12).all(), f"{name}: P(y > j) rises"
        want = baseline.model.predict_proba(x.astype(np.float64))
        assert np.allclose(baseline.predict_proba(batch), want, atol=1e-6), name


def test_regression_values_and_the_quantile_head():
    sp, spec, _, _ = row_fixture("regression")
    valid = sp.load_part(spec, "valid")
    train_y = sp.load_part(spec, "train").y.numpy()
    trivial = fitted(spec, sp, "trivial")
    x = trivial.design_of(sp, valid).x
    assert np.allclose(trivial.score_matrix(x), np.mean(train_y))
    linear = fitted(spec, sp, "linear")
    assert np.allclose(linear.score_matrix(x), linear.model.predict(x.astype(np.float64)))
    q = {"name": "quantile", "params": {"quantiles": [0.1, 0.5, 0.9]}}
    qsp, qspec, _, _ = row_fixture("regression", loss=q)
    for name in ("trivial", "linear"):
        a = fitted(spec, sp, name).score_matrix(x)
        b = fitted(qspec, qsp, name).score_matrix(x)
        assert b.shape == (30,) and np.allclose(a, b), name


def _row_error(x, mean, priors):
    """R-C2's per-row score by hand: numeric MSE + the mean nominal cross-entropy of the
    log-prior logits."""
    numeric, codes = x[:, :3].astype(np.float64), x[:, 3:].astype(np.int64)
    err = ((numeric - mean) ** 2).mean(axis=1)
    ce = []
    for i, lp in enumerate(priors):
        norm = lp - np.log(np.exp(lp).sum())
        ce.append(-norm[codes[:, i]])
    return err + np.mean(ce, axis=0)


def _log_priors(train_x):
    counts = np.bincount(train_x[:, 3].astype(np.int64), minlength=5) / len(train_x)
    return [np.log(clip(counts))]


def test_reconstruction_baselines_score_the_numeric_row_and_its_error():
    sp, spec, _, _ = row_fixture("unsupervised")
    train, valid = sp.load_part(spec, "train"), sp.load_part(spec, "valid")
    built = X.build(spec, ("trivial", "linear", "tree"))
    assert {type(b).__name__ for b in built} == {"Baseline"}, "one recipe, its form composed in"
    assert [b.name for b in built] == ["trivial", "linear", "tree"]
    trivial = fitted(spec, sp, "trivial")
    tx = trivial.design_of(sp, train).x.astype(np.float64)
    vx = trivial.design_of(sp, valid).x.astype(np.float64)
    mean = tx[:, :3].mean(axis=0)
    assert trivial.score_matrix(vx).shape == (30, 3)
    assert np.allclose(trivial.score_matrix(vx), mean)
    assert np.allclose(trivial.row_score_matrix(vx), _row_error(vx, mean, _log_priors(tx)))
    linear = fitted(spec, sp, "linear", D.BaselineContext(z=2))
    pca = linear.model["pca"]
    assert pca.n_components_ == 2
    recon = pca.inverse_transform(pca.transform(vx[:, :3]))
    assert np.allclose(linear.score_matrix(vx), recon)
    assert np.allclose(linear.row_score_matrix(vx), _row_error(vx, recon, _log_priors(tx)))
    tree = fitted(spec, sp, "tree")
    assert not tree.fitted and "reconstruction" in tree.why


def test_anomaly_baselines_are_per_row_scores():
    sp, spec, _, _ = row_fixture("anomaly")
    train, valid = sp.load_part(spec, "train"), sp.load_part(spec, "valid")
    trivial = fitted(spec, sp, "trivial")
    tx = trivial.design_of(sp, train).x.astype(np.float64)
    vx = trivial.design_of(sp, valid).x.astype(np.float64)
    want = _row_error(vx, tx[:, :3].mean(axis=0), _log_priors(tx))
    assert np.allclose(trivial.score_matrix(vx), want)
    assert np.allclose(trivial.row_score_matrix(vx), want)
    linear = fitted(spec, sp, "linear")
    pca = linear.model["pca"]
    assert np.allclose(linear.score_matrix(vx), _row_error(
        vx, pca.inverse_transform(pca.transform(vx[:, :3])), _log_priors(tx)))
    forest = fitted(spec, sp, "tree")
    assert type(forest.model).__name__ == "IsolationForest"
    assert np.allclose(forest.score_matrix(vx), -forest.model.score_samples(vx))
    lsp, lspec, _, _ = row_fixture("anomaly-labelled")
    boosted = fitted(lspec, lsp, "tree")
    assert type(boosted.model).__name__ == "HistGradientBoostingClassifier"
    p = boosted.model.predict_proba(vx)[:, 1]
    assert np.allclose(boosted.score_matrix(vx), logit(p))
    ssp, sspec, _, _ = row_fixture("anomaly-svdd")
    for name in ("trivial", "linear"):
        a = fitted(spec, sp, name).score_matrix(vx)
        b = fitted(sspec, ssp, name).score_matrix(vx)
        assert b.shape == (30,) and np.allclose(a, b), f"{name}: the svdd spec's baseline " \
                                                        "does not reconstruct"


def test_refusals_are_recorded_not_raised():
    sp, spec, _, (frame, parts) = row_fixture("binary")
    one = frame.copy()
    one["flag"] = 1
    osp, ospec, _, _ = row_fixture("binary", frame=one)
    linear = fitted(ospec, osp, "linear")
    assert not linear.fitted and "one class only" in linear.why
    fx.raises(lambda: linear.score_matrix(np.zeros((2, 4))), "RuntimeError", "linear",
              "one class only")
    naive = fitted(spec, sp, "naive")
    assert not naive.fitted and "window" in naive.why and "row" in naive.why
    glin = fitted(spec, sp, "graph-linear")
    assert not glin.fitted and "node-level" in glin.why and glin.view is DV.FLAT
    lens = fitted(spec, sp, "lens")
    assert not lens.fitted and "importance" in lens.why
    ctx = D.BaselineContext(protocol={"family": "gradient boosted trees"})
    lens = fitted(spec, sp, "lens", ctx)
    assert lens.fitted and lens.record()["delegate"] == "tree"
    tree = fitted(spec, sp, "tree")
    x = tree.design_of(sp, sp.load_part(spec, "valid")).x
    assert np.allclose(lens.score_matrix(x), tree.score_matrix(x))
    wsp, wspec, _, _ = window_fixture(season=None)
    seasonal = fitted(wspec, wsp, "seasonal-naive")
    assert not seasonal.fitted and "season" in seasonal.why
    far = fitted(wspec, wsp, "seasonal-naive", D.BaselineContext(season=L + 1))
    assert not far.fitted and f"season {L + 1}" in far.why
    msg = fx.raises(lambda: X.build(spec, ("trivial", "campaign-baseline")), "ValueError",
                    "campaign-baseline")
    assert "stored" in msg


# ================================================================ 2. windows

def test_window_forecast_baselines():
    sp, spec, _, frame = window_fixture()
    valid, train = sp.load_part(spec, "valid"), sp.load_part(spec, "train")
    ev = evaluator(sp)
    for name in ("trivial", "linear", "tree"):
        scored = ev.scored(spec, valid, fitted(spec, sp, name))
        assert scored.score.shape == (len(valid), H), name
    trivial = fitted(spec, sp, "trivial")
    tdesign = trivial.design_of(sp, train)
    assert tdesign.x.shape[1] == L * 3 + L * 1, "a window is flattened as L·F′ + L·C"
    assert np.allclose(ev.scored(spec, valid, trivial).score, tdesign.y.mean())
    naive = ev.scored(spec, valid, fitted(spec, sp, "naive"))
    assert np.allclose(naive.score, at(frame, valid.row_ids, (0, 0, 0)))
    offsets = [h - SEASON * math.ceil(h / SEASON) for h in range(1, H + 1)]
    assert offsets == [-1, 0, -1]
    seasonal = ev.scored(spec, valid, fitted(spec, sp, "seasonal-naive"))
    assert np.allclose(seasonal.score, at(frame, valid.row_ids, offsets))
    q = {"name": "quantile", "params": {"quantiles": [0.25, 0.5, 0.75]}}
    qsp, qspec, _, _ = window_fixture(loss=q)
    qnaive = fitted(qspec, qsp, "naive")
    batch = next(iter(qsp.make_loader(valid, qspec, batch_size=len(valid))))
    assert np.allclose(qnaive.score(batch), naive.score)


def test_naive_reads_the_labels_without_the_autoregressive_channel():
    sp, spec, _, frame = window_fixture()
    nsp, nspec, _, _ = window_fixture(autoregressive=False)
    ev, nev = evaluator(sp), evaluator(nsp)
    for name in ("naive", "seasonal-naive"):
        with_channel = ev.scored(spec, sp.load_part(spec, "valid"), fitted(spec, sp, name))
        baseline = fitted(nspec, nsp, name)
        part = nsp.load_part(nspec, "valid")
        assert part.x_num.shape[1] == 2 and baseline.sample_view(part).x_num.shape[1] == 3
        without = nev.scored(nspec, part, baseline)
        assert np.array_equal(with_channel.score, without.score), name
        assert np.array_equal(with_channel.row_ids, without.row_ids)
        bare = nsp.design_matrix(nspec, part)
        fx.raises(lambda: baseline.score_matrix(bare.x), "ValueError", "design_of",
                  "target")
        assert np.allclose(baseline.score_matrix(baseline.design_of(nsp, part).x),
                           without.score)


def test_naive_never_reads_the_target_it_predicts():
    seq = {"kind": "sequence-regression", "base_kind": "regression", "label": "target"}
    sp, spec, _, frame = window_fixture(task=seq)
    valid = sp.load_part(spec, "valid")
    naive = evaluator(sp).scored(spec, valid, fitted(spec, sp, "naive"))
    assert naive.score.shape == (len(valid),)
    assert np.allclose(naive.score, at(frame, valid.row_ids, (-1,))[:, 0])
    t0 = T_TRAIN + 3
    moved = frame.copy()
    moved.loc[moved.ts == t0, "target"] += 1000.0
    msp, mspec, _, _ = window_fixture(frame=moved, task=seq)
    mvalid = msp.load_part(mspec, "valid")
    again = evaluator(msp).scored(mspec, mvalid, fitted(mspec, msp, "naive"))
    ts = valid.row_ids // N_SERIES
    assert np.array_equal(naive.score[ts == t0], again.score[ts == t0]), \
        "the naive read the anchor's own label"
    assert not np.allclose(naive.score[ts == t0 + 1], again.score[ts == t0 + 1])
    fsp, fspec, _, fframe = window_fixture()
    fvalid = fsp.load_part(fspec, "valid")
    before = evaluator(fsp).scored(fspec, fvalid, fitted(fspec, fsp, "seasonal-naive"))
    later = fframe.copy()
    later.loc[later.ts > t0, "target"] -= 1000.0
    lsp, lspec, _, _ = window_fixture(frame=later)
    after = evaluator(lsp).scored(lspec, lsp.load_part(lspec, "valid"),
                                  fitted(lspec, lsp, "seasonal-naive"))
    early = (fvalid.row_ids // N_SERIES) <= t0
    assert early.any() and np.array_equal(before.score[early], after.score[early])
    ordinal = {"kind": "sequence-regression", "base_kind": "ordinal", "label": "level",
               "classes": [0, 1, 2]}
    osp, ospec, _, oframe = window_fixture(task=ordinal, loss={"name": "cumulative_link",
                                                               "params": {}})
    ovalid = osp.load_part(ospec, "valid")
    onaive = fitted(ospec, osp, "naive")
    batch = next(iter(osp.make_loader(onaive.sample_view(ovalid), ospec,
                                      batch_size=len(ovalid))))
    assert onaive.score(batch).shape == (len(ovalid), 2)
    previous = oframe.level.to_numpy()[[raw_of(a % N_SERIES, a // N_SERIES - 1)
                                        for a in ovalid.row_ids]]
    assert np.array_equal(onaive.predict_proba(batch).argmax(axis=1), previous)


def test_sequence_classification_reads_the_flattened_window():
    task = {"kind": "sequence-classification", "base_kind": "binary", "label": "up",
            "classes": [0, 1]}
    sp, spec, _, _ = window_fixture(task=task, loss={"name": "bce_with_logits", "params": {}})
    valid = sp.load_part(spec, "valid")
    for name in ("trivial", "linear", "tree"):
        baseline = fitted(spec, sp, name)
        assert baseline.fitted, (name, baseline.why)
        scored = evaluator(sp).scored(spec, valid, baseline)
        assert scored.score.shape == (len(valid),) and scored.truth.shape == (len(valid),)
    design = fitted(spec, sp, "linear").design_of(sp, valid)
    assert design.x.shape == (len(valid), L * 2 + L * 1)
    naive = fitted(spec, sp, "naive")
    assert not naive.fitted and "sequence-classification" in naive.why


# ================================================================ 3. graphs

def _accuracy(scored):
    return float(np.mean((sigmoid(scored.score) >= 0.5) == (scored.truth == 1)))


def test_graph_linear_reads_the_neighbours_and_beats_the_structure_blind_floor():
    sp, spec, _, (frame, parts, _) = beacon_fixture()
    valid = sp.load_part(spec, "valid")
    ev = evaluator(sp)
    blind, glin = fitted(spec, sp, "linear"), fitted(spec, sp, "graph-linear")
    assert blind.view is DV.FLAT and glin.view is DV.NEIGHBOUR_MEAN
    own = blind.design_of(sp, valid)
    assert own.x.shape == (30, 2) and np.array_equal(own.row_ids, valid.row_ids)
    x0 = frame.x0.to_numpy(dtype=np.float32)
    assert np.array_equal(own.x[:, 0], x0[valid.row_ids]), "linear reads another node's row"
    wide = glin.design_of(sp, valid)
    assert wide.x.shape == (30, 4) and np.array_equal(wide.x[:, :2], own.x)
    s = frame.s.to_numpy()
    targets = valid.x_num[:, 1].numpy()
    for i in range(0, 30, 7):
        node = int(valid.target_index[i])
        into = valid.edge_index[1].numpy() == node
        src = valid.edge_index[0].numpy()[into]
        assert abs(wide.x[i, 3] - valid.x_num[src, 1].numpy().mean()) < 1e-5
        assert abs(targets[node] - s[valid.row_ids[i]]) < 1e-5
    a_blind = _accuracy(ev.scored(spec, valid, blind))
    a_glin = _accuracy(ev.scored(spec, valid, glin))
    assert a_glin >= 0.9 and a_blind <= 0.8, (a_glin, a_blind)
    tree = ev.scored(spec, valid, fitted(spec, sp, "tree"))
    assert tree.score.shape == (30,)


def test_graph_level_baselines_read_pooled_graphs():
    sp, spec, _, (frame, parts) = molecule_fixture()
    train, valid = sp.load_part(spec, "train"), sp.load_part(spec, "valid")
    trivial = fitted(spec, sp, "trivial")
    assert trivial.view is DV.GRAPH_POOLED
    assert np.allclose(sigmoid(trivial.score_matrix(np.zeros((1, 6)))), 10 / 20), \
        "the prior counts graphs, not nodes"
    design = trivial.design_of(sp, valid)
    assert design.x.shape == (5, 6)
    rows = frame.iloc[parts["valid"]]
    agg = rows.groupby("molecule")["x1"].agg(["mean", "max", "sum"])
    assert np.allclose(design.x[:, [1, 3, 5]], agg.loc[list(valid.row_ids)].to_numpy(),
                       atol=1e-5)
    ev = evaluator(sp)
    for name in ("trivial", "linear", "tree"):
        scored = ev.scored(spec, valid, fitted(spec, sp, name))
        assert scored.score.shape == (5,) and list(scored.row_ids) == list(valid.row_ids)
    assert _accuracy(ev.scored(spec, valid, fitted(spec, sp, "linear"))) >= 0.8
    glin = fitted(spec, sp, "graph-linear")
    assert not glin.fitted and "level graph" in glin.why


# ================================================================ 4. one design

def _batch_design(sp, spec, part, view, batch_size=None):
    return np.concatenate([B.design_rows(b, view)
                           for b in sp.make_loader(part, spec, batch_size=batch_size)])


def test_batch_designs_equal_the_design_matrix():
    rsp, rspec, _, _ = row_fixture("binary")
    part = rsp.load_part(rspec, "valid")
    assert np.array_equal(_batch_design(rsp, rspec, part, DV.FLAT, 7),
                          rsp.design_matrix(rspec, part, DV.FLAT).x)
    wsp, wspec, _, _ = window_fixture()
    part = wsp.load_part(wspec, "valid")
    assert np.array_equal(_batch_design(wsp, wspec, part, DV.FLAT, 5),
                          wsp.design_matrix(wspec, part, DV.FLAT).x)
    nsp, nspec, _, _ = beacon_fixture()
    part = nsp.load_part(nspec, "valid")
    for view in (DV.FLAT, DV.NEIGHBOUR_MEAN):
        assert np.array_equal(_batch_design(nsp, nspec, part, view),
                              nsp.design_matrix(nspec, part, view).x), view
    gsp, gspec, _, _ = molecule_fixture()
    part = gsp.load_part(gspec, "train")
    assert len(list(gsp.make_loader(part, gspec))) == 5
    assert np.array_equal(_batch_design(gsp, gspec, part, DV.GRAPH_POOLED),
                          gsp.design_matrix(gspec, part, DV.GRAPH_POOLED).x)
    fx.raises(lambda: _batch_design(gsp, gspec, part, DV.FLAT), "ValueError", "graph-pooled")
    fx.raises(lambda: _batch_design(rsp, rspec, rsp.load_part(rspec, "valid"),
                                    DV.NEIGHBOUR_MEAN), "ValueError", "node-level")


# ================================================================ 5. Evaluator.scored

def _networks():
    """(fixture, network) per shape: a row MLP, a window GRU, a node GCN, a graph-level GIN."""
    out = {}
    for shape, fixture in (("row", lambda: row_fixture("binary")),
                           ("window", lambda: window_fixture(nominal=False)),
                           ("node", beacon_fixture), ("graph", molecule_fixture)):
        sp, spec, root, extra = fixture()
        out[shape] = (sp, spec, root, extra, NS.network.Network.build(spec))
    return out


def test_scored_row_ids_truth_and_kinds_per_shape():
    nets = _networks()
    for shape, (sp, spec, root, extra, net) in nets.items():
        valid = sp.load_part(spec, "valid")
        ev = evaluator(sp)
        by_net = ev.scored(spec, valid, net)
        by_base = ev.scored(spec, valid, fitted(spec, sp, "trivial"))
        assert np.array_equal(by_net.row_ids, by_base.row_ids), shape
        assert np.array_equal(by_net.truth, by_base.truth), shape
        assert by_net.score.shape == by_base.score.shape, shape
        assert np.array_equal(by_net.row_ids, np.asarray(valid.row_ids)), shape
        want = "graph_id" if shape == "graph" else "raw_position"
        assert by_net.row_id_kind.value == want, shape
        if shape == "graph":
            assert list(by_net.graph_ids) == list(by_net.row_ids)
            assert set(by_net.row_ids) == {f"m{g:02d}" for g in range(20, 25)}
        else:
            assert by_net.graph_ids is None
    sp, spec, _, (frame, parts), _ = nets["row"]
    scored = evaluator(sp).scored(spec, sp.load_part(spec, "valid"), nets["row"][4])
    assert list(scored.row_ids) == list(parts["valid"]) and scored.unscored == ()
    assert np.array_equal(scored.truth, frame.flag.to_numpy()[parts["valid"]])
    nsp, nspec, _, (nframe, nparts, _), nnet = nets["node"]
    scored = evaluator(nsp).scored(nspec, nsp.load_part(nspec, "valid"), nnet)
    assert set(scored.row_ids) == set(nparts["valid"]) and scored.score.shape == (30,)
    sp, spec, _, frame = window_fixture(context="none")
    valid = sp.load_part(spec, "valid")
    scored = evaluator(sp).scored(spec, valid, fitted(spec, sp, "naive"))
    reasons = [u.reason.value for u in scored.unscored]
    assert reasons.count("insufficient_context") == (L - 1) * N_SERIES
    assert reasons.count("horizon_beyond_part") == H * N_SERIES
    assert len(scored) == N_SERIES * (T_VALID - T_TRAIN - (L - 1) - H)
    assert scored.truth.shape == (len(scored), H)
    assert np.allclose(scored.truth, at(frame, scored.row_ids, (1, 2, 3)))


def test_row_score_of_the_reconstruction_and_anomaly_readings():
    sp, spec, _, _ = row_fixture("unsupervised")
    valid = sp.load_part(spec, "valid")
    net = NS.network.Network.build(spec)
    scored = evaluator(sp).scored(spec, valid, net)
    batch = next(iter(sp.make_loader(valid, spec, batch_size=len(valid))))
    net.eval()
    with torch.no_grad():
        want = NS.readout.TaskReadout.for_spec(spec).row_score(net(batch), batch, net).numpy()
    net.train()
    assert scored.score.shape == (30, 3) and np.allclose(scored.row_score, want, atol=1e-6)
    assert np.array_equal(scored.truth, valid.x_num.numpy()), "the truth is the numeric row"
    assert net.training, "scoring left a training network in eval mode"
    base = evaluator(sp).scored(spec, valid, fitted(spec, sp, "trivial"))
    assert base.row_score.shape == (30,) and base.truth.shape == (30, 3)
    for key in ("anomaly", "anomaly-svdd"):
        asp, aspec, _, _ = row_fixture(key)
        net = NS.network.Network.build(aspec)
        if key == "anomaly-svdd":
            net.svdd_center.copy_(torch.randn(4))
        scored = evaluator(asp).scored(aspec, asp.load_part(aspec, "valid"), net)
        assert scored.score.shape == (30,) and np.array_equal(scored.row_score, scored.score)
        assert scored.truth is None, "an unlabelled anomaly has no truth"
    bsp, bspec, _, _ = row_fixture("binary")
    scored = evaluator(bsp).scored(bspec, bsp.load_part(bspec, "valid"),
                                   NS.network.Network.build(bspec))
    assert scored.row_score is None


def test_heads_are_the_raw_head_in_scored_order():
    """`heads` is R-C2's raw head — every head_out column — where the score contract reads a
    part of it: a quantile's median, the numeric reconstruction, the distance to the centre."""
    q = {"name": "quantile", "params": {"quantiles": [0.1, 0.5, 0.9]}}
    cases = (("regression", q, 3, lambda h, c: h[:, 1]),
             ("unsupervised", None, None, lambda h, c: h[:, :3]),
             ("anomaly-svdd", None, None, lambda h, c: ((h - c) ** 2).sum(axis=1)),
             ("multiclass", None, None, lambda h, c: h))
    for key, loss, out, reading in cases:
        sp, spec, _, _ = row_fixture(key, loss=loss, out=out)
        valid = sp.load_part(spec, "valid")
        net = NS.network.Network.build(spec)
        centre = None
        if key == "anomaly-svdd":
            net.svdd_center.copy_(torch.randn(4))
            centre = net.svdd_center.numpy().astype(np.float64)
        ev = evaluator(sp)
        heads = ev.heads(spec, valid, net)
        batch = next(iter(sp.make_loader(valid, spec, batch_size=len(valid))))
        with torch.no_grad():
            net.eval()
            want = net(batch).numpy()
            net.train()
        width = ROW_TASKS[key][2] if out is None else out
        assert heads.shape == (30, width) and np.allclose(heads, want, atol=1e-6), key
        assert net.training, f"{key}: heads left a training network in eval mode"
        scored = ev.scored(spec, valid, net)
        assert np.allclose(scored.score, reading(heads.astype(np.float64), centre),
                           atol=1e-5), key
    nsp, nspec, _, _ = beacon_fixture()
    part = nsp.load_part(nspec, "valid")
    net = NS.network.Network.build(nspec)
    assert evaluator(nsp).heads(nspec, part, net).shape == (30, 1), "node heads: the targets'"
    wsp, wspec, _, frame = window_fixture(loss={"name": "quantile",
                                                "params": {"quantiles": [0.25, 0.5, 0.75]}})
    part = wsp.load_part(wspec, "valid")
    naive = fitted(wspec, wsp, "naive")
    heads = evaluator(wsp).heads(wspec, part, naive)
    last = at(frame, part.row_ids, (0,))
    assert heads.shape == (len(part), H * 3) and np.allclose(heads, np.repeat(last, H * 3, 1))
    asp, aspec, _, _ = row_fixture("anomaly")
    forest = fitted(aspec, asp, "tree")
    apart = asp.load_part(aspec, "valid")
    assert np.allclose(evaluator(asp).heads(aspec, apart, forest),
                       evaluator(asp).scored(aspec, apart, forest).score)
    unfitted = made(aspec, "linear")                 # a recipe is no scored model: by type
    assert not hasattr(unfitted, "head") and not hasattr(unfitted, "score")


# ================================================================ 6. the metrics

def test_score_part_and_the_sealed_test():
    sp, spec, root, (frame, parts) = row_fixture("binary", secondary=("roc_auc", "mase"))
    net = NS.network.Network.build(spec)
    ev = E.Evaluator.restoring(sp, NS.store.ModelRegistry(), NS.network.Network.build)
    fx.raises(lambda: ev.score_part(spec, "test", model=net), "SealedPartError", "test_seal")
    fx.raises(lambda: ev.score_sealed_test(spec, "tok", model=net,
                                           seal_path=str(root / "none.json")),
              "SealedPartError", "no seal file")
    seal = fx.seal(root, "tok")
    fx.raises(lambda: ev.score_sealed_test(spec, "nope", model=net, seal_path=str(seal)),
              "SealedPartError", "token")
    unstamped = ev.score_sealed_test(spec, "tok", model=net, seal_path=str(seal))
    assert unstamped.run is None and unstamped.exposure is None    # a seal without a look yet
    record = json.loads(seal.read_text(encoding="utf-8"))
    seal.write_text(json.dumps(dict(record, run="R2", exposures=2)), encoding="utf-8")
    test = ev.score_sealed_test(spec, "tok", model=net, seal_path=str(seal))
    # one record shape (metrics.record): every metric asked for, a failed one None
    assert test.part.value == "test" and set(test.as_dict()) == {"pr_auc", "roc_auc", "mase",
                                                                   "_unavailable"}
    assert test["mase"] is None and "mase" in test["_unavailable"]
    assert (test.run, test.exposure) == ("R2", 2), "the sealed record carries the seal's look"
    valid = ev.score_part(spec, "valid", model=net)
    path = scratch("b13-ck-") / "best.pt"
    checkpoint = NS.store.ModelRegistry().save(net, path, spec=spec)
    assert ev.score_part(spec, "valid", checkpoint=str(path)).as_dict() == valid.as_dict()
    assert ev.score_part(spec, "valid", checkpoint=checkpoint).as_dict() == valid.as_dict()
    scored = ev.scored(spec, sp.load_part(spec, "valid"), net)
    assert ev.measure(spec, "valid", scored).as_dict() == valid.as_dict()
    from sklearn.metrics import average_precision_score
    assert abs(valid["pr_auc"] - average_precision_score(scored.truth, scored.score)) < 1e-12
    bare = evaluator(sp)
    fx.raises(lambda: bare.score_part(spec, "valid", checkpoint=str(path)), "ValueError",
              "Evaluator.over", "Evaluator.restoring(data, store, build)")
    fx.raises(lambda: bare.score_part(spec, "valid"), "ValueError", "checkpoint")
    tree = fitted(spec, sp, "tree")
    by_tree = bare.score_part(spec, "valid", model=tree)
    assert 0.0 <= by_tree["pr_auc"] <= 1.0
    stamped = by_tree.with_provenance("R3", 1)
    assert stamped.run == "R3" and stamped.exposure == 1


def test_mase_reads_the_train_scale_or_the_callers():
    sp, spec, _, _ = window_fixture()
    naive = fitted(spec, sp, "naive")
    scale = sp.target_scale(spec, SEASON)
    assert scale and scale > 0
    m = evaluator(sp).score_part(spec, "valid", model=naive)
    assert abs(m["mase"] - m["mae"] / scale) < 1e-9, (m.as_dict(), scale)
    given = evaluator(sp).with_options(D.MetricOptions(mase_scale=2.0))
    m2 = given.score_part(spec, "valid", model=naive)
    assert abs(m2["mase"] - m2["mae"] / 2.0) < 1e-12 and m2["mae"] == m["mae"]
    assert sp.target_scale(spec, 1) != scale, "the fixture must tell the two seasons apart"
    threshold_only = evaluator(sp).with_options(D.MetricOptions(threshold=0.3))
    m3 = threshold_only.score_part(spec, "valid", model=naive)
    assert m3.as_dict() == m.as_dict(), "the spec's season fills the caller's unset one"
    chosen = evaluator(sp).with_options(D.MetricOptions(season=3))
    m4 = chosen.score_part(spec, "valid", model=naive)
    assert abs(m4["mase"] - m["mae"] / sp.target_scale(spec, 3)) < 1e-9, "a caller's season"
    assert evaluator(sp).metric_names(spec) == ["mae", "mase"]


def test_metric_options_are_resolved_against_the_spec_in_one_place():
    """`MetricOptions.stated(spec)` is the one resolution the Trainer and the Evaluator share:
    the spec fills what the caller left default (season, quantiles); a window's `scaled` adds
    the train scale `mase` alone asks the DataSource for; a caller's season or scale stands.
    Final rehearsal round 2 (improve, 3 of 3 audits): `stated` took `names` and `data` and, in
    every package but a window one, read neither — the window's scale is its own method now."""
    sp, spec, _, _ = window_fixture()
    bare = D.MetricOptions().stated(spec)
    assert bare.season == SEASON and bare.mase_scale is None
    assert bare.quantiles == tuple(spec.loss.quantiles)
    with_mase = bare.scaled(spec, ("mae", "mase"), sp)
    assert with_mase.mase_scale == sp.target_scale(spec, SEASON)
    assert bare.scaled(spec, ("mae",), sp).mase_scale is None
    assert bare.scaled(spec, ("mase",), None).mase_scale is None, "no DataSource, no scale"
    chosen = D.MetricOptions(season=3, mase_scale=2.0).stated(spec).scaled(spec, ("mase",), sp)
    assert (chosen.season, chosen.mase_scale) == (3, 2.0)
    assert list(inspect.signature(D.MetricOptions.stated).parameters) == ["self", "spec"]


def test_build_record_and_the_port():
    sp, spec, _, _ = window_fixture()
    names = ("trivial", "linear", "tree", "lens", "naive", "seasonal-naive", "graph-linear")
    built = X.build(spec, names, D.BaselineContext(protocol={"family": "ridge"}))
    assert [b.name for b in built] == list(names)
    assert {type(b).__name__ for b in built} == {"Baseline"}
    assert [b.name for b in X.build(spec)] == ["trivial", "linear", "tree", "lens"]
    valid, outcomes = sp.load_part(spec, "valid"), []
    for baseline in built:
        before = baseline.record()
        assert before["fitted"] is False and before["why"] is None
        outcome = baseline.fit(baseline.design_of(sp, sp.load_part(spec, "train")))
        assert baseline.record() == before, "fitting changes nothing on the recipe"
        record = outcome.record()
        assert record["name"] == baseline.name and record["view"] == "flat"
        assert record["fitted"] == (baseline.name != "graph-linear"), record
        assert outcome.fitted == record["fitted"]
        if not outcome.fitted:
            assert type(outcome).__name__ == "RefusedBaseline" and outcome.why
            continue
        assert not hasattr(outcome, "anatomy") and not hasattr(outcome, "block_shapes"), \
            "a baseline is a scored model, not an implementation of the Model port"
        batch = next(iter(sp.make_loader(outcome.sample_view(valid), spec, batch_size=4)))
        assert np.asarray(outcome.head(batch)).shape[0] == 4
        outcomes.append(outcome)
    assert [o.name for o in outcomes] == list(names[:-1])


# ================================================================ 7. the size discipline

def test_template_discipline():
    import conformance
    files = [NS.root / "evaluation" / "evaluate.py", NS.root / "model" / "baselines.py",
             NS.root / "model" / "baselines_extra.py"]
    _, findings = conformance.size_only([str(f) for f in files])
    assert not findings, [f.message() for f in findings]
    tree = ast.parse(files[0].read_text(encoding="utf-8"))
    adapter = conformance.adapter_class(tree, "Evaluator", "Evaluator")
    assert adapter is not None and adapter.name == "Evaluator"
    stray = [f for node in tree.body
             for f in conformance._top_level_findings("evaluation/evaluate.py", node, adapter,
                                                      "Evaluator")]
    assert not stray, [f.message() for f in stray]
    rows = {e["file"]: e for e in NS.blueprint_map["entries"]}
    assert rows["model/baselines.py"]["role"] == rows["model/baselines_extra.py"]["role"] \
        == "baseline"
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                assert node.args.kwarg is None and node.args.vararg is None, \
                    f"{path.name}:{node.lineno} {node.name} takes *args/**kw"


def test_the_evaluator_imports_along_the_arrows():
    """C4 and C5 on the Evaluator adapter of the rendered package: it reaches the model's
    module only for its support files (`metrics.py`, `readout.py`) and imports no adapter —
    the DataSource, the ModelStore and the builder arrive by constructor."""
    import conformance
    from vega_core import Level
    pkg, problem = conformance.Package.read(NS.root, NS.blueprint_map, NS.pipeline)
    assert problem is None, problem
    for check in (conformance.c4_imports_follow_the_arrows,
                  conformance.c5_adapters_depend_on_ports):
        bad = [m.message for m in check(pkg) if m.level is Level.ERROR
               and (m.where or "").startswith("evaluation/")]
        assert not bad, bad
    source = (NS.root / "evaluation" / "evaluate.py").read_text(encoding="utf-8")
    imported = {n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)}
    # the readout and the metrics through the model module's published API, never a file of it
    assert imported == {"__future__", "domain", "ports", "model"}, imported


def test_the_baseline_family_imports_one_way():
    """The T5 round 1 audit (finding 6): `baselines.build` imported `baselines_extra` inside the
    function, which imported four private helpers from `baselines` — a cycle deferred to call
    time. `baselines.py` now imports nothing of the family; `baselines_extra.py` imports its
    public classes and helpers and holds `build`."""
    src = {name: (NS.root / "model" / f"{name}.py").read_text(encoding="utf-8")
           for name in ("baselines", "baselines_extra")}
    base = ast.parse(src["baselines"])
    assert not [n for n in ast.walk(base) if isinstance(n, ast.ImportFrom) and n.level
                and "baselines_extra" in ((n.module or "") + "".join(a.name for a in n.names))]
    assert not [n for n in ast.walk(base) if isinstance(n, ast.FunctionDef) and n.name == "build"]
    extra = [n for n in ast.parse(src["baselines_extra"]).body
             if isinstance(n, ast.ImportFrom) and n.module == "baselines"]
    assert extra and all(not a.name.startswith("_") for n in extra for a in n.names)
    assert X.build is not None and not hasattr(B, "build")


if __name__ == "__main__":
    sys.exit(fx.main_for(globals(), sys.argv, "baseline"))


def test_the_lens_composes_its_configuring_concern():
    """The T5 round 3 audit (finding 5): the lens's recorded hyper-parameters are a collaborator
    composed into the delegate `build` gives the task (`configure`), never a subclass per
    (concern x baseline); the record is a value, read through the delegate's public surface."""
    sp, spec, _, _ = row_fixture("binary")
    ctx = D.BaselineContext(protocol={"family": "gradient-boosting",
                                      "hyper_parameters": {"max_depth": 2}})
    lens = made(spec, "lens", ctx)
    plain = made(spec, "tree", ctx)
    assert type(lens) is type(plain), (type(lens), type(plain))
    assert not [n for n in vars(X) if n.startswith("_Lens") or n == "_Configured"]
    assert lens.record()["hyper_parameters"]["not_carried"] == {"max_depth": "not fitted yet"}
    train = sp.load_part(spec, "train")
    fitted_lens = lens.fit(lens.design_of(sp, train))
    assert fitted_lens.record()["hyper_parameters"] == {"applied": {"max_depth": 2},
                                                         "not_carried": {}}
    assert fitted_lens.record()["delegate"] == "tree"
    shallow = fitted_lens._fitted.state
    assert shallow.get_params()["max_depth"] == 2
    tree = plain.fit(plain.design_of(sp, train))
    assert tree._fitted.state.get_params()["max_depth"] is None, "the plain tree is unconfigured"
