#!/usr/bin/env python3
"""
test_target_affine.py — a regression-family head ends on a fixed train-fitted target affine (A37).

    python3 -m pytest test_target_affine.py [-q]        (or: python3 test_target_affine.py)

The heads of the regression family — regression, sequence-regression, forecast and the quantile
heads — carry the buffers `target_shift` (the TRAIN target's mean) and `target_scale` (its
standard deviation, floored away from 0), one value per output channel: per output, per horizon
step, repeated over a quantile head's quantiles horizon-major (channel h·Q + q reads step h, the
layout `TaskReadout` reads). The head's last operation is `y = z · target_scale + target_shift`,
so the network learns in standardised units while its output — and the loss, the metrics, the
stored predictions and serving — stays in the target's own. The affine is fitted from the train
part alone, once (`target_fitted`); a checkpoint carries it; `init_notes` records it and the
architecture layer draws it. The templates are rendered with `render_templates.py` beside the
v2.3 DataSource, Model, ModelStore and Trainer (the spliced builder, blocks and losses). Proved:

1. **The buffers are the train statistics.** For a row regression, a row quantile regression, a
   window forecast (mean and spread per horizon step over the training windows), a window
   forecast under the quantile loss (per step, repeated per quantile, horizon-major), a sequence
   regression (the anchors' own labels), a node-level and a graph-level regression (one label
   per graph, not per row): after `Trainer.fit` the head's buffers equal the statistics
   recomputed here from the train parquet with numpy; the head's own bias keeps the spec's init.
2. **No valid or test data is read.** Every fixture's valid and test labels sit 1000 above the
   train ones, so any leak would move the statistics; a reader that records every file
   `ShapedParts.train_targets` opens sees the train part alone.
3. **Raw units out.** The network's output is `act(W·x + b) · scale + shift`; the Trainer's valid
   metrics are measured in the target's units (the valid labels' offset shows in the rmse).
4. **Fitted once, kept.** Only the three buffers differ from a plain build of the spec; a
   checkpoint written and read back through the ModelStore carries them, so a resumed run keeps
   its affine and reads nothing for it; an affine of another width is refused by name.
5. **Every other head carries none**: binary, multiclass, ordinal and reconstruction heads have no
   affine buffer, and the Trainer reads no target for them.
6. **Inspection reads it**: on the inspection zoo's regression cases a fitted affine keeps every
   probe's recompute check within the contract's tolerance, and the stage graph publishes the
   buffers as `<head>.target_shift` / `<head>.target_scale` for the page's output step.
7. **The forge records and shows it** (a headless browser; skipped only when none is
   installed): on B15's window spike, `emit_backend.py --smoke` writes
   `init_notes.target_affine` into `smoke.json` and `spec_snapshot.json`'s `_resolved`, equal to
   the package's own statistics; the architecture layer draws one `× scale + shift` per channel
   (`smoke_architecture.py`'s `target-affine-shown-iff-recorded`), its noscript says it, and a
   render that stops drawing it fails the smoke.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import re
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

PACKAGE = "a37_target_affine"
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
RENDERED = ("domain.py", "network.py", "readout.py", "metrics.py", "frames.py", "windows.py",
            "graphs.py", "data.py", "loop.py", "optim.py", "record.py", "engines.py",
            "graph_engines.py", "store.py")
LEAK = 1000.0            # the valid and test labels' offset: a leak would move every statistic
QUANTILES = [0.1, 0.5, 0.9]


# ================================================================ the rendered package

def _purge(prefix):
    for name in [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]:
        del sys.modules[name]


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    """Every template a fit reads, rendered with a spec that uses every block type, imported."""
    out = tmp_path_factory.mktemp("a37-package")
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = PACKAGE
    wide = rt.reference_spec("row")
    wide["blocks"] = [{"id": f"W{i}", "type": t, "params": {}}
                      for i, t in enumerate(CATALOG["blocks"])]
    rec = rt.render(TEMPLATES, out, pipeline, wide, only=list(RENDERED), splice=True)
    sys.path.insert(0, str(out))
    importlib.invalidate_caches()
    try:
        mods = {"domain": "domain", "network": "model.network", "readout": "model.readout",
                "data": "ingestion.data", "frames": "ingestion.frames", "loop": "training.loop",
                "store": "model.store"}
        yield types.SimpleNamespace(root=Path(rec["root"]),
                                    **{k: importlib.import_module(f"{PACKAGE}.{v}")
                                       for k, v in mods.items()})
    finally:
        sys.path.remove(str(out))
        _purge(PACKAGE)


def _still(spec: dict, epochs: int = 1) -> dict:
    """The spec at learning rate 0 on the CPU: the weights never move, so after a fit the head
    holds the bias it started from."""
    spec = copy.deepcopy(spec)
    spec["optimizer"]["params"] = {"lr": 0.0, "weight_decay": 0.0}
    spec["training"].update(max_epochs=epochs, device="cpu")
    return spec


def _affine(model) -> tuple:
    head = model.stages[-1]
    return (head.target_shift.detach().cpu().double().numpy(),
            head.target_scale.detach().cpu().double().numpy())


def _fit(pkg, spec: dict, model=None, data=None):
    ns = pkg.domain.NetSpec.from_dict(spec)
    model = model if model is not None else pkg.network.Network.build(ns)
    pkg.loop.Trainer().fit(model, ns, data if data is not None else pkg.data.ShapedParts())
    return model


class _NoTargets:
    """A DataSource that refuses `train_targets`: what the Trainer must not ask for."""

    def __init__(self, inner):
        self._inner = inner

    def train_targets(self, spec):
        raise AssertionError("train_targets was read for a head that fits no affine")

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ================================================================ the shaped datasets

def _stats(y, quantiles=None) -> tuple:
    """(shift, scale) per head channel, recomputed here: each target column's mean and
    standard deviation, repeated over the quantiles (horizon-major)."""
    y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    q = len(quantiles) if quantiles else 1
    return np.repeat(y.mean(axis=0), q), np.repeat(y.std(axis=0), q)


def _regression_label(rng, n, offset=40.0):
    return offset + rng.normal(0, 6.0, n)


def _leaky(frame: pd.DataFrame, parts: dict, label: str) -> dict:
    """The parts, with the valid and test labels moved LEAK above the train ones."""
    out = {}
    for name, pos in parts.items():
        f = frame.iloc[pos].copy()
        if name != "train":
            f[label] = f[label] + LEAK
        out[name] = f
    return out


def _mse(spec: dict, head_out: int, loss=None) -> dict:
    spec["loss"] = loss or {"name": "mse", "params": {}}
    spec["metrics"] = {"primary": "mae", "secondary": [], "source": "vega:default"}
    spec["early_stopping"] = {"monitor": "valid.mae", "mode": "min", "patience": 5}
    spec["blocks"][-1]["params"]["out"] = head_out
    return spec


def row_case(tmp, quantiles=None) -> tuple:
    rng = np.random.default_rng(21)
    n = 180
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    frame["zone"] = rng.integers(0, 5, size=n)
    frame["y"] = _regression_label(rng, n) + 3.0 * frame.x0
    frame["flag"] = (frame.x1 > 0).astype(np.int64)        # the other heads' binary label
    perm = rng.permutation(n)
    parts = {"train": perm[:110], "valid": perm[110:145], "test": perm[145:]}
    man = fx.write_shaped(tmp / f"rows-{bool(quantiles)}", _leaky(frame, parts, "y"), parts)
    spec = rt.reference_spec("row")
    spec["task"] = {"kind": "regression", "label": "y"}
    loss = {"name": "quantile", "params": {"quantiles": quantiles}} if quantiles else None
    spec = _mse(spec, len(quantiles) if quantiles else 1, loss)
    y = frame.y.to_numpy()[parts["train"]]
    return fx.spec_on(spec, man), _stats(y, quantiles)


def _series_frame(rng, steps=60, series=3):
    rows = []
    for t in range(steps):
        for s in range(series):
            rows.append({"series": f"s{s}", "ts": t, "x0": math.sin(t / 5 + s),
                         "x1": float(s) + rng.normal(0, 0.1), "zone": (t + s) % 4,
                         "target": 25.0 + 10 * math.sin(t / 5 + s) * (1 + s) + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def _window_targets(train: pd.DataFrame, length: int, horizon: int) -> np.ndarray:
    """Recomputed here, by the plan's rules (stride 1, no context in the train part): per
    series ordered by `ts`, an anchor is every row with L−1 rows before it and H after it; a
    forecast's targets are the H rows after the anchor, a sequence task's the anchor's own."""
    out = []
    for _, block in train.sort_values(["series", "ts"], kind="stable").groupby("series"):
        y = block.target.to_numpy()
        for a in range(length - 1, len(y) - horizon):
            out.append(y[a + 1: a + 1 + horizon] if horizon else y[a: a + 1])
    return np.asarray(out, dtype=np.float64)


def window_case(tmp, kind="forecast", quantiles=None) -> tuple:
    rng = np.random.default_rng(5)
    frame = _series_frame(rng)
    pos = np.arange(len(frame))
    parts = {"train": pos[frame.ts < 40], "valid": pos[(frame.ts >= 40) & (frame.ts < 50)],
             "test": pos[frame.ts >= 50]}
    parts = {k: np.random.default_rng(3).permutation(v) for k, v in parts.items()}
    man = fx.write_shaped(tmp / f"windows-{kind}-{bool(quantiles)}",
                          _leaky(frame, parts, "target"), parts)
    spec = rt.reference_spec("window")
    dc = spec["data_contract"]
    dc["features"] = [fx.numeric("x0"), fx.numeric("x1"), fx.nominal("zone", 4)]
    dc["input_dim"] = 3
    horizon = 2 if kind == "forecast" else 0
    dc["sample"].update({"length": 6, "stride": 1, "horizon": horizon, "season": 4,
                         "autoregressive": kind == "forecast"})
    if kind == "forecast":
        spec["task"]["horizon"] = horizon
    else:
        spec["task"] = {"kind": "sequence-regression", "base_kind": "regression",
                        "label": "target"}
    spec["blocks"] = [
        {"id": "E", "type": "embedding", "params": {"columns": ["zone"], "dims": {"zone": 2}}},
        {"id": "C", "type": "concat", "params": {}},
        {"id": "G", "type": "gru", "params": {"hidden": 8, "return": "last"}},
        {"id": "H", "type": "head", "params": {"out": 1, "activation": "none"}}]
    spec["training"]["batch_size"] = 16
    loss = {"name": "quantile", "params": {"quantiles": quantiles}} if quantiles else None
    spec = _mse(spec, max(1, horizon) * (len(quantiles) if quantiles else 1), loss)
    y = _window_targets(frame.iloc[parts["train"]], 6, horizon)
    return fx.spec_on(spec, man), _stats(y, quantiles)


def node_case(tmp) -> tuple:
    rng = np.random.default_rng(2)
    n = 90
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(3)})
    frame["y"] = _regression_label(rng, n, 20.0) + 4.0 * frame.x0
    perm = np.random.default_rng(4).permutation(n)
    parts = {"train": perm[:54], "valid": perm[54:72], "test": perm[72:]}
    part_of = {int(p): name for name, pos in parts.items() for p in pos}
    pairs, pick = set(), np.random.default_rng(6)
    while len(pairs) < 200:
        a, b = (int(v) for v in pick.integers(0, n, size=2))
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    edges = pd.DataFrame([{"src_pos": s, "dst_pos": d, "src_part": part_of[s],
                           "dst_part": part_of[d], "weight": None}
                          for a, b in sorted(pairs) for s, d in ((a, b), (b, a))])
    man = fx.write_shaped(tmp / "nodes", _leaky(frame, parts, "y"), parts, edges)
    spec = rt.reference_spec("graph")
    spec["task"] = {"kind": "regression", "label": "y"}
    spec["data_contract"]["features"] = [fx.numeric(f"x{i}") for i in range(3)]
    spec["data_contract"]["input_dim"] = 3
    spec["training"]["batch_size"] = "full"
    spec = _mse(spec, 1)
    return fx.spec_on(spec, man), _stats(frame.y.to_numpy()[parts["train"]])


def graphs_case(tmp) -> tuple:
    """Graphs of 3–12 nodes whose label grows with their size: the mean over graphs is not the
    mean over rows, so a statistic taken per row would be caught."""
    rng = np.random.default_rng(8)
    rows, edges, parts, labels = [], [], {"train": [], "valid": [], "test": []}, []
    for g in range(40):
        size = int(rng.integers(3, 13))
        label = 10.0 + 2.0 * size + rng.normal(0, 0.5)
        part = "train" if g < 24 else "valid" if g < 32 else "test"
        label += LEAK if part != "train" else 0.0
        labels += [label] if part == "train" else []
        first = len(rows)
        for k in range(size):
            rows.append({"molecule": f"m{g}", "x0": rng.normal(), "x1": float(k) / size,
                         "x2": rng.normal(), "y": label})
            parts[part].append(first + k)
        for a, b in [(first + k, first + k + 1) for k in range(size - 1)]:
            edges += [{"src_pos": s, "dst_pos": d, "src_part": part, "dst_part": part,
                       "weight": None} for s, d in ((a, b), (b, a))]
    frame = pd.DataFrame(rows)
    parts = {k: np.asarray(v) for k, v in parts.items()}
    man = fx.write_shaped(tmp / "graphs", {k: frame.iloc[v] for k, v in parts.items()}, parts,
                          pd.DataFrame(edges))
    spec = rt.reference_spec("graph-level")
    spec["task"] = {"kind": "regression", "label": "y"}
    spec["data_contract"]["features"] = [fx.numeric(f"x{i}") for i in range(3)]
    spec["data_contract"]["input_dim"] = 3
    spec["training"]["batch_size"] = 8
    spec = _mse(spec, 1)
    rows_mean = frame.y.to_numpy()[parts["train"]].mean()
    assert abs(rows_mean - np.mean(labels)) > 0.5, "the fixture must tell graphs from rows"
    return fx.spec_on(spec, man), _stats(labels)


CASES = {"row": lambda t: row_case(t), "row-quantile": lambda t: row_case(t, QUANTILES),
         "forecast": lambda t: window_case(t),
         "forecast-quantile": lambda t: window_case(t, quantiles=QUANTILES),
         "sequence-regression": lambda t: window_case(t, kind="sequence"),
         "node": node_case, "graph-level": graphs_case}


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a34-data")
    return {name: make(tmp) for name, make in CASES.items()}


# ================================================================ 1. the train statistics

@pytest.mark.parametrize("case", list(CASES))
def test_the_buffers_are_the_train_statistics(pkg, cases, case):
    spec, (shift, scale) = cases[case]
    model = _fit(pkg, _still(spec))
    got_shift, got_scale = _affine(model)
    assert got_shift.shape == shift.shape and got_scale.shape == scale.shape, (got_shift, shift)
    assert np.allclose(got_shift, shift, rtol=1e-6, atol=1e-5), f"{case}: {got_shift} != {shift}"
    assert np.allclose(got_scale, scale, rtol=1e-6, atol=1e-5), f"{case}: {got_scale} != {scale}"
    assert bool(model.stages[-1].target_fitted) and not model.target_affine_pending
    assert float(np.abs(got_shift).min()) > 1.0 and float(got_scale.min()) > 0.1
    bias = model.stages[-1].linear.bias.detach()
    assert torch.equal(bias, torch.zeros_like(bias)), "the head's own bias keeps the spec's init"


def test_the_quantile_channels_follow_the_readouts_layout(pkg, cases):
    """Horizon-major: channel h·Q + q carries step h's mean and spread, the step the readout's
    median column reads."""
    spec, (shift, scale) = cases["forecast-quantile"]
    ns = pkg.domain.NetSpec.from_dict(spec)
    affine = pkg.domain.TargetAffine.of(ns, pkg.data.ShapedParts().train_targets(ns), "H")
    assert affine.per == "horizon step and quantile" and affine.quantiles == tuple(QUANTILES)
    out = torch.tensor([affine.shift])
    median = pkg.readout.TaskReadout.for_spec(ns).score(out).numpy()[0]
    steps = shift.reshape(2, len(QUANTILES))
    assert np.allclose(median, steps[:, QUANTILES.index(0.5)], atol=1e-9)
    assert np.allclose(steps, steps[:, :1]) and np.allclose(affine.scale, scale)


def test_the_floor_keeps_a_constant_target_away_from_zero(pkg, cases):
    spec, _ = cases["row"]
    ns = pkg.domain.NetSpec.from_dict(spec)
    affine = pkg.domain.TargetAffine.of(ns, np.full((20, 1), 250.0), "B5")
    assert affine.shift == (250.0,) and 0 < affine.scale[0] <= 250.0 * 1e-6 + 1e-12


# ================================================================ 2. the train part alone

class _Recording:
    """A PartFrames that records the part of every file it is asked to open."""

    def __init__(self, inner):
        self._inner, self.parts = inner, []

    def _log(self, name):
        def call(spec, part, *args, **kwargs):
            self.parts.append(str(getattr(part, "value", part)))
            return getattr(self._inner, name)(spec, part, *args, **kwargs)
        return call

    def __getattr__(self, name):
        if name in ("open", "open_features_only", "chunks"):
            return self._log(name)
        if name == "unseal":
            raise AssertionError("train_targets asked for the sealed test part")
        return getattr(self._inner, name)


@pytest.mark.parametrize("case", list(CASES))
def test_only_the_train_part_is_read(pkg, cases, case):
    spec, _ = cases[case]
    reader = _Recording(pkg.frames.PartFrames())
    targets = pkg.data.ShapedParts(reader).train_targets(pkg.domain.NetSpec.from_dict(spec))
    assert reader.parts and set(reader.parts) == {"train"}, reader.parts
    assert float(np.nanmax(targets)) < LEAK / 2, "a valid or test label reached the targets"


# ================================================================ 3. raw units out

def test_the_output_and_the_metrics_are_in_the_targets_units(pkg, cases):
    spec, (shift, scale) = cases["row"]
    ns = pkg.domain.NetSpec.from_dict(spec)
    model = _fit(pkg, _still(spec)).eval()
    data = pkg.data.ShapedParts()
    valid = data.load_part(ns, "valid")
    batch = valid[torch.arange(len(valid))]
    with torch.no_grad():
        out = model(batch).double().numpy()[:, 0]
        head = model.stages[-1]
        z = head.act(head.linear(_through(model, batch))).double().numpy()[:, 0]
    assert np.allclose(out, z * scale[0] + shift[0], rtol=1e-5, atol=1e-4), "y = z · scale + shift"
    truth = batch["y"].double().numpy().reshape(-1)
    values = pkg.loop.Trainer().evaluate(model, ns, valid)
    assert values["mae"] == pytest.approx(float(np.mean(np.abs(out - truth))), rel=1e-4)
    assert values["mae"] > LEAK / 2, "the valid labels' offset shows: the metric reads raw units"


def _through(model, batch):
    """The running tensor the head reads (every stage but the last)."""
    flow = model._flows(batch)
    last = None
    for i, state in enumerate(flow):
        if i == len(model.stages) - 2:
            last = state.h
    return last


# ================================================================ 4. fitted once, kept

def test_only_the_affine_buffers_differ_from_a_plain_build(pkg, cases):
    spec, (shift, scale) = cases["forecast"]
    ns = pkg.domain.NetSpec.from_dict(spec)
    plain, fitted = pkg.network.Network.build(ns), pkg.network.Network.build(ns)
    affine = pkg.domain.TargetAffine.of(ns, pkg.data.ShapedParts().train_targets(ns), "H")
    assert plain.target_affine_pending and fitted.fit_target_affine(affine) is True
    a, b = plain.state_dict(), fitted.state_dict()
    last = f"stages.{len(plain.stages) - 1}"
    mine = {f"{last}.target_shift", f"{last}.target_scale", f"{last}.target_fitted"}
    assert list(a) == list(b) and mine <= set(a)
    for key in a:
        if key not in mine:
            assert torch.equal(a[key], b[key]), f"{key} moved: only the affine is fitted"
    assert torch.equal(a[f"{last}.target_scale"], torch.ones_like(a[f"{last}.target_scale"]))
    assert np.allclose(b[f"{last}.target_shift"].double().numpy(), shift)
    assert np.allclose(b[f"{last}.target_scale"].double().numpy(), scale)
    assert fitted.fit_target_affine(affine) is False, "the affine is fitted once"


def test_a_checkpoint_carries_the_affine_and_a_resumed_run_keeps_it(pkg, cases, tmp_path):
    spec, (shift, scale) = cases["row"]
    trained = copy.deepcopy(spec)
    trained["training"].update(max_epochs=2, device="cpu")
    first = _fit(pkg, trained)
    ns = pkg.domain.NetSpec.from_dict(spec)
    store = pkg.store.ModelRegistry()
    store.save(first, tmp_path / "last.pt", spec=ns, epoch=2)
    resumed = store.load(tmp_path / "last.pt", spec=ns, build=pkg.network.Network.build)
    assert not resumed.target_affine_pending, "the checkpoint carries target_fitted"
    assert np.allclose(_affine(resumed)[0], shift) and np.allclose(_affine(resumed)[1], scale)
    before = {k: v.clone() for k, v in resumed.state_dict().items() if "target_" in k}
    _fit(pkg, _still(spec), model=resumed, data=_NoTargets(pkg.data.ShapedParts()))
    after = resumed.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "a resumed run refitted"


def test_an_affine_of_another_width_is_refused(pkg, cases):
    spec, _ = cases["forecast"]
    ns = pkg.domain.NetSpec.from_dict(spec)
    model = pkg.network.Network.build(ns)
    wrong = pkg.domain.TargetAffine((1.0, 2.0, 3.0), (1.0, 1.0, 1.0), "horizon step", (), 10, "H")
    with pytest.raises(ValueError, match="the target affine of H"):
        model.fit_target_affine(wrong)
    assert model.target_affine_pending


def test_the_record_is_the_init_note(pkg, cases):
    spec, (shift, scale) = cases["row-quantile"]
    ns = pkg.domain.NetSpec.from_dict(spec)
    targets = pkg.data.ShapedParts().train_targets(ns)
    affine = pkg.domain.TargetAffine.of(ns, targets, "B5")
    notes = pkg.domain.TargetAffine.init_notes(affine, ns)
    rec = notes["target_affine"]
    assert rec["part"] == "train" and rec["block"] == "B5" and rec["source"] == "A37"
    assert rec["quantiles"] == QUANTILES and rec["samples"] == len(targets) == 110
    assert np.allclose(rec["shift"], shift) and np.allclose(rec["scale"], scale)
    assert notes["scheme"] == spec["init"]["scheme"] and len(notes["notes"]) == 1
    assert "y = z · scale + shift" in notes["notes"][0]
    assert pkg.domain.TargetAffine.init_notes(None, ns) == {
        "scheme": spec["init"]["scheme"], "target_affine": None, "notes": []}


# ================================================================ 5. every other head

def _other_heads(cases) -> dict:
    row, _ = cases["row"]
    binary = copy.deepcopy(row)
    binary["task"] = {"kind": "binary", "label": "flag", "classes": [0, 1], "positive": 1}
    binary["loss"] = {"name": "bce_with_logits", "params": {}}
    multiclass = copy.deepcopy(row)
    multiclass["task"] = {"kind": "multiclass", "label": "zone", "classes": [0, 1, 2, 3, 4]}
    multiclass["loss"] = {"name": "cross_entropy", "params": {}}
    multiclass["blocks"][-1]["params"]["out"] = 5
    ordinal = copy.deepcopy(multiclass)
    ordinal["task"]["kind"] = "ordinal"
    ordinal["loss"] = {"name": "cumulative_link", "params": {}}
    ordinal["blocks"][-1]["params"]["out"] = 4
    recon = copy.deepcopy(row)
    recon["task"] = {"kind": "unsupervised"}
    recon["loss"] = {"name": "reconstruction", "params": {}}
    recon["blocks"][-1]["params"]["out"] = 4 + 5           # recon_dim: 4 numerics, zone of 5
    for spec, primary in ((binary, "accuracy"), (multiclass, "accuracy"), (ordinal, "accuracy"),
                          (recon, "reconstruction_error")):
        spec["metrics"] = {"primary": primary, "secondary": [], "source": "vega:default"}
        spec["early_stopping"]["monitor"] = f"valid.{primary}"
    return {"binary": binary, "multiclass": multiclass, "ordinal": ordinal,
            "reconstruction": recon}


def test_other_heads_carry_no_affine_and_read_nothing(pkg, cases):
    for name, spec in _other_heads(cases).items():
        ns = pkg.domain.NetSpec.from_dict(spec)
        model = pkg.network.Network.build(ns)
        assert not [k for k in model.state_dict() if "target_" in k], name
        assert not model.target_affine_pending, name
        assert pkg.domain.TargetAffine.of(ns, np.ones((3, 1))) is None, name
        before = {k: v.clone() for k, v in model.state_dict().items()}
        _fit(pkg, _still(spec), model=model, data=_NoTargets(pkg.data.ShapedParts()))
        after = model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before), f"{name}: a weight moved"


# ================================================================ 6. inspection reads it

ZOO_REGRESSION = ("ft-transformer", "gru")


@pytest.fixture(scope="module")
def zoo():
    import _inspect_fixtures as zf
    return zf


@pytest.mark.parametrize("name", ZOO_REGRESSION)
def test_a_fitted_affine_keeps_every_probe_check_and_is_published(zoo, name):
    c = zoo.case(name)
    width = int(c.model.shapes[-1][-1])
    affine = c.domain.TargetAffine(tuple(40.0 + k for k in range(width)),
                                   tuple(7.0 + k for k in range(width)), "output", (), 10,
                                   c.model.ids[-1])
    assert c.model.fit_target_affine(affine) is True
    section, cap = zoo.explain(c, 0)
    head = c.graph.top_level()[-1]
    assert section["nodes"][head.id]["recompute"] is not None
    for node in c.graph.nodes:
        value = section["nodes"][node.id]["recompute"]
        if value is None:
            continue
        out = cap.take(cap.value(node.id), node)
        bound = 1e-4 * max(1.0, float(np.abs(out.detach().numpy()).max()))
        assert value <= bound, f"{node.id} ({node.type}): recompute {value} > {bound}"
    graph = c.graph_mod.StageGraph.build(c.model, c.domain.NetSpec.from_dict(c.spec),
                                         (c.batch, None))
    tensors = c.view.network_view(graph, "full", "sha256:" + "1" * 64)["tensors"]
    for key, want in (("target_shift", affine.shift), ("target_scale", affine.scale)):
        tid = f"{head.id}.{key}"
        assert tid in tensors and tensors[tid]["role"] == "buffer", sorted(tensors)
        assert np.allclose(c.codec.decode(tensors[tid]), want)


# ================================================================ 7. the forge records it

def _layered_window(work: Path):
    """B15's window spike forged into a layered file the way the forge stage builds one."""
    import test_architecture_sample as tas
    lay = tas.Layered(work)
    return tas, lay, lay("window")


@pytest.fixture(scope="module")
def forged(tmp_path_factory):
    tests = SCRIPTS.parent.parent / "vega" / "scripts" / "tests"
    for p in (str(HERE / "spike"), str(tests)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import smoke_architecture as sa
    if not sa.find_chrome():
        pytest.skip("no headless browser: smoke_architecture.py needs one ($CHROME or PATH)")
    work = tmp_path_factory.mktemp("a37-forge")
    tas, lay, html = _layered_window(work)
    return types.SimpleNamespace(tas=tas, sa=sa, html=html, run=Path(lay.fixture("window").root)
                                 / "run-window")


def test_the_forge_records_the_affine_in_the_snapshot(forged):
    smoke = json.loads((forged.run / "smoke.json").read_text(encoding="utf-8"))
    snap = json.loads((forged.run / "spec_snapshot.json").read_text(encoding="utf-8"))
    notes = smoke["init_notes"]
    assert snap["_resolved"]["init_notes"] == notes
    rec = notes["target_affine"]
    spec = json.loads((forged.run / "netspec.json").read_text(encoding="utf-8"))
    assert rec["part"] == "train" and rec["per"] == "horizon step"
    assert len(rec["shift"]) == len(rec["scale"]) == spec["data_contract"]["sample"]["horizon"] == 3
    package = spec["outputs"]["package"]
    sys.path.insert(0, str(forged.run))
    try:
        importlib.invalidate_caches()
        domain = importlib.import_module(f"{package}.domain")
        data = importlib.import_module(f"{package}.ingestion.data")
        ns = domain.NetSpec.from_dict(spec)
        targets = data.ShapedParts().train_targets(ns)
    finally:
        sys.path.remove(str(forged.run))
        _purge(package)
    assert np.allclose(rec["shift"], targets.mean(axis=0), rtol=1e-12)
    assert np.allclose(rec["scale"], targets.std(axis=0), rtol=1e-12)
    assert rec["samples"] == len(targets)


def test_the_architecture_layer_shows_it(forged):
    html = forged.html.read_text(encoding="utf-8")
    data, _ = forged.sa.layer_data(html)
    assert len(data["smoke"]["init_notes"]["target_affine"]["shift"]) == 3
    assert "Initialisation: the head" in html and "y = z · scale + shift" in html
    code, out = forged.tas.smoke(forged.html)
    assert code == 0, out[-3000:]
    for label in ("full", "no-measurements"):
        assert re.search(rf"^\s+OK\s+{label}:target-affine-shown-iff-recorded$", out, re.M), \
            out[-3000:]


def test_a_render_that_hides_the_affine_fails_the_smoke(forged, tmp_path):
    html = forged.html.read_text(encoding="utf-8")
    assert html.count("'a-tag a-affine'") == 1
    broken = tmp_path / "hidden.html"
    broken.write_text(html.replace("'a-tag a-affine'", "'a-tag'"), encoding="utf-8")
    code, out = forged.tas.smoke(broken)
    assert code != 0 and re.search(r"FAIL\s+full:target-affine-shown-iff-recorded", out), \
        out[-3000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
