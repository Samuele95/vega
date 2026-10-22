#!/usr/bin/env python3
"""
test_windows.py — the window half of the DataSource (`templates/windows.py` through
`ShapedParts`), proved on the rendered package (build plan B08; backend design §9.3, §17.1;
R-A5 window semantics).

    python3 test_windows.py [-v] [-k <substring>]
    python3 -m pytest test_windows.py

The fixture is a raw frame of three interleaved series whose `x0` is each row's raw-file
position and whose `target` is ten times it, split by time and stored in a scrambled order, so
every tensor a window holds names the raw rows it came from. What is proved:

  * no window crosses `series_by`, and within a series the rows are in `order_by` order;
  * context rows (the train tail before valid; the valid tail, then train, before the sealed
    test) are inputs only, never anchors; `context: none` leaves the first L−1 rows of each
    series unscored as `insufficient_context`;
  * forecast targets are the next H rows of the same part and series and never cross the
    part's end (`horizon_beyond_part`); a sequence task's target is the anchor's own label;
  * the stride thins the training anchors from the series end (the most recent kept); valid and
    test are dense, one window per scorable row;
  * `row_ids` are the anchors' raw-file positions; `x_cat` is `(B, L, C)`;
  * the autoregressive channel is the label at rows up to the anchor only: perturbing every
    later label leaves the batch identical;
  * the seal: load_part / iter_batches / load_resident refuse the test part before any file is
    opened; load_sealed_part refuses a missing seal and a wrong token; a window sample streams
    from its held part (E1); the design matrix is `(W, L·F' + L·C)`; mase's train-side scale;
  * the real shaper: a time-split multi-series csv shaped by `shape.py`, whose windows' targets
    are the raw file's next H values of the anchor's series.

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

NS = fx.ingestion("windows_pkg", "window", only=("frames.py", "windows.py", "data.py"))
D = NS.domain
L, H, T_TRAIN, T_VALID, T_END, N_SERIES = 4, 3, 24, 32, 40, 3


def raw_frame(n_steps: int = T_END, n_series: int = N_SERIES) -> pd.DataFrame:
    """Rows in time order, series interleaved: x0 = raw position, target = 10 × x0."""
    rows = [{"series": f"s{s}", "ts": t, "x1": float(s), "zone": (t + s) % 4,
             "state": ("low", "mid", "high")[(t + s) % 3]}
            for t in range(n_steps) for s in range(n_series)]
    frame = pd.DataFrame(rows)
    frame.insert(2, "x0", np.arange(len(frame), dtype=np.float64))
    frame["target"] = frame["x0"] * 10.0
    return frame


def split(frame, cuts=(T_TRAIN, T_VALID), scramble: bool = True) -> dict:
    """Raw positions of each part by time; each part stored in a scrambled order."""
    rng = np.random.default_rng(3)
    pos = np.arange(len(frame))
    parts = {"train": pos[frame.ts < cuts[0]], "valid": pos[(frame.ts >= cuts[0]) &
                                                          (frame.ts < cuts[1])],
             "test": pos[frame.ts >= cuts[1]]}
    return {k: (rng.permutation(v) if scramble else v) for k, v in parts.items()}


def fixture(frame=None, parts=None, task=None, **sample):
    """(ShapedParts, NetSpec, shaped dir) over a written fixture; `task` replaces the forecast
    (a sequence task: no horizon, no autoregressive channel); `sample` overrides."""
    frame = raw_frame() if frame is None else frame
    parts = split(frame) if parts is None else parts
    root = Path(tempfile.mkdtemp(prefix="windows-fx-"))
    man = fx.write_shaped(root, {k: frame.iloc[v] for k, v in parts.items()}, parts)
    spec = fx.rt.reference_spec("window")
    dc = spec["data_contract"]
    dc["features"] = [fx.numeric("x0"), fx.numeric("x1"), fx.nominal("zone", 5)]
    dc["input_dim"] = 3
    dc["sample"].update({"length": L, "stride": 1, "horizon": H, "series_by": "series",
                         "order_by": "ts", "context": "previous-part", "autoregressive": True})
    dc["sample"].update(sample)
    if task is not None:
        spec["task"] = task
        dc["sample"].pop("horizon", None)
        dc["sample"]["autoregressive"] = False
    spec = fx.spec_on(spec, man)
    return NS.data.ShapedParts(), D.NetSpec.from_dict(spec), root


def batch_of(sp, spec, part):
    return next(iter(sp.make_loader(part, spec, batch_size=len(part))))


def series_of(x0, frame):
    return frame.series.to_numpy()[x0.astype(np.int64)]


# --------------------------------------------------------------------------- the plan's rules

def test_no_window_crosses_a_series_and_rows_follow_order_by():
    frame = raw_frame()
    sp, spec, _ = fixture(frame)
    for part in ("train", "valid"):
        b = batch_of(sp, spec, sp.load_part(spec, part))
        x0 = b["x_num"][:, :, 0].numpy()
        names = series_of(x0, frame)
        assert (names == names[:, :1]).all(), f"{part}: a window holds two series"
        ts = frame.ts.to_numpy()[x0.astype(np.int64)]
        assert (np.diff(ts, axis=1) == 1).all(), f"{part}: window rows out of order_by order"


def test_context_rows_are_inputs_never_anchors():
    frame = raw_frame()
    parts = split(frame)
    sp, spec, _ = fixture(frame, parts)
    valid = sp.load_part(spec, "valid")
    b = batch_of(sp, spec, valid)
    x0 = b["x_num"][:, :, 0].numpy().astype(np.int64)
    assert set(valid.row_ids.tolist()) <= set(parts["valid"].tolist())
    assert np.array_equal(x0[:, -1], valid.row_ids), "the anchor is not the window's last row"
    first = np.isin(x0[:, 0], parts["train"])
    assert first.any(), "no valid window reads the train tail as its context"
    assert np.isin(x0[first], np.concatenate([parts["train"], parts["valid"]])).all()


def test_context_none_leaves_the_first_rows_unscored():
    sp, spec, _ = fixture(context="none")
    valid = sp.load_part(spec, "valid")
    reasons = [u.reason.value for u in valid.unscored]
    assert reasons.count("insufficient_context") == (L - 1) * N_SERIES, reasons
    assert reasons.count("horizon_beyond_part") == H * N_SERIES, reasons
    per_series = T_VALID - T_TRAIN
    assert len(valid) == N_SERIES * (per_series - (L - 1) - H)


def test_forecast_targets_stay_in_the_part_and_series():
    frame = raw_frame()
    parts = split(frame)
    sp, spec, _ = fixture(frame, parts)
    valid = sp.load_part(spec, "valid")
    b = batch_of(sp, spec, valid)
    targets = (b["y"].numpy() / 10.0).astype(np.int64)            # target = 10 × raw position
    assert b["y"].shape == (len(valid), H)
    assert np.isin(targets, parts["valid"]).all(), "a forecast target left the valid part"
    anchors = valid.row_ids
    assert (series_of(targets, frame) == series_of(anchors, frame)[:, None]).all()
    ts = frame.ts.to_numpy()
    assert (ts[targets] == ts[anchors][:, None] + np.arange(1, H + 1)).all()
    ends = {u.raw_pos for u in valid.unscored if u.reason.value == "horizon_beyond_part"}
    assert all(ts[p] >= T_VALID - H for p in ends) and len(ends) == H * N_SERIES


def test_stride_thins_training_anchors_and_keeps_the_most_recent():
    frame = raw_frame()
    sp, spec, _ = fixture(frame, stride=3)
    dense_sp, dense_spec, _ = fixture(frame, stride=1)
    thin, dense = sp.load_part(spec, "train"), dense_sp.load_part(dense_spec, "train")
    assert len(thin) < len(dense)
    ts = frame.ts.to_numpy()
    for s in range(N_SERIES):
        mine = sorted(ts[r] for r in thin.row_ids if frame.series[r] == f"s{s}")
        allr = sorted(ts[r] for r in dense.row_ids if frame.series[r] == f"s{s}")
        assert mine[-1] == allr[-1], "the most recent anchor was not kept"
        assert all(b - a == 3 for a, b in zip(mine, mine[1:])), mine
    valid = sp.load_part(spec, "valid")
    assert len(valid) == N_SERIES * (T_VALID - T_TRAIN - H), "valid is not scored densely"


def test_row_ids_are_anchor_raw_positions_and_x_cat_is_three_dimensional():
    frame = raw_frame()
    sp, spec, _ = fixture(frame)
    valid = sp.load_part(spec, "valid")
    b = batch_of(sp, spec, valid)
    assert b["x_cat"].shape == (len(valid), L, 1) and b["x_cat"].dtype == torch.long
    zone = frame.zone.to_numpy()[b["x_num"][:, :, 0].numpy().astype(np.int64)]
    assert np.array_equal(b["x_cat"][:, :, 0].numpy(), zone), "x_cat is not the window's rows"
    assert valid.row_id_kind.value == "raw_position"
    assert np.array_equal(valid.row_ids, b["x_num"][:, -1, 0].numpy().astype(np.int64))


def test_autoregressive_channel_reads_no_future_label():
    frame = raw_frame()
    parts = split(frame)
    sp, spec, _ = fixture(frame, parts)
    cut = 27
    later = frame.copy()
    later.loc[later.ts > cut, "target"] = -1.0e6                  # every later label perturbed
    sp2, spec2, _ = fixture(later, parts)
    a, b = batch_of(sp, spec, sp.load_part(spec, "valid")), batch_of(
        sp2, spec2, sp2.load_part(spec2, "valid"))
    anchors_ts = frame.ts.to_numpy()[sp.load_part(spec, "valid").row_ids]
    early = torch.from_numpy(anchors_ts <= cut)
    assert early.any() and (~early).any()
    assert torch.equal(a["x_num"][early], b["x_num"][early]), "a window read a future label"
    assert torch.equal(a["x_cat"], b["x_cat"])
    assert not torch.equal(a["x_num"][~early], b["x_num"][~early])
    channel = a["x_num"][:, :, -1].numpy()
    assert np.array_equal(channel, a["x_num"][:, :, 0].numpy() * 10.0), "channel is not the label"


def test_sequence_task_predicts_the_anchor_label():
    frame = raw_frame()
    task = {"kind": "sequence-classification", "base_kind": "multiclass", "label": "state",
            "classes": ["low", "mid", "high"]}
    sp, spec, _ = fixture(frame, task=task)
    valid = sp.load_part(spec, "valid")
    b = batch_of(sp, spec, valid)
    want = [("low", "mid", "high").index(frame.state[r]) for r in valid.row_ids]
    assert b["y"].dtype == torch.long and b["y"].tolist() == want
    assert b["x_num"].shape[-1] == 2, "a sequence task carries no autoregressive channel"
    assert len(valid) == N_SERIES * (T_VALID - T_TRAIN) and not valid.unscored


def test_sealed_test_reads_valid_then_train_as_context():
    frame = raw_frame()
    parts = split(frame, cuts=(T_TRAIN, T_TRAIN + 2))              # valid: 2 rows per series
    sp, spec, root = fixture(frame, parts)
    seal = fx.seal(root, "tok")
    test = sp.load_sealed_part(spec, "tok", str(seal))
    b = batch_of(sp, spec, test)
    x0 = b["x_num"][:, :, 0].numpy().astype(np.int64)
    assert np.isin(test.row_ids, parts["test"]).all()
    first = x0[np.argsort(frame.ts.to_numpy()[test.row_ids])[:N_SERIES]]
    assert np.isin(first[:, 0], parts["train"]).all() and np.isin(first[:, 1:3],
                                                                   parts["valid"]).all()
    assert len(test) == N_SERIES * (T_END - T_TRAIN - 2 - H)


# --------------------------------------------------------------------------- the seal and the API

def test_seal_refuses_the_test_part_before_opening_anything():
    sp, spec, _ = fixture()
    dead = spec.to_dict()
    dead["input"]["manifest"] = "/nonexistent/manifest.json"
    dead["data_contract"]["parts"] = {"train": "/nonexistent/a", "valid": "/nonexistent/b",
                                      "test": "sealed"}
    dead = D.NetSpec.from_dict(dead)
    for call in (lambda: sp.load_part(dead, "test"), lambda: sp.iter_batches(dead, "test"),
                 lambda: sp.load_resident(dead, "test", "cpu")):
        fx.raises(call, "SealedPartError", "test part")


def test_sealed_part_needs_the_seal_and_its_token():
    sp, spec, root = fixture()
    fx.raises(lambda: sp.load_sealed_part(spec, "tok", str(root / "no_seal.json")),
              "SealedPartError", "no seal file")
    seal = fx.seal(root, "right")
    fx.raises(lambda: sp.load_sealed_part(spec, "wrong", str(seal)), "SealedPartError", "token")
    assert len(sp.load_sealed_part(spec, "right", str(seal)))


def test_a_window_sample_streams_from_its_held_part():
    sp, spec, _ = fixture()
    plan = D.BatchPlan(shuffle=True, seed=11)
    first = [b["x_num"] for b in sp.iter_batches(spec, "train", 16, plan=plan)]
    again = [b["x_num"] for b in sp.iter_batches(spec, "train", 16, plan=plan)]
    assert len(first) == -(-len(sp.load_part(spec, "train")) // 16)
    assert sp.batch_count(spec, "train", 16, plan=plan) == len(first)      # T5 round 2
    assert all(torch.equal(a, b) for a, b in zip(first, again)), "the seeded order moved"
    assert all(x.shape[1:] == (L, 3) for x in first)
    fx.raises(lambda: sp.iter_batches(spec, "train", 16, columns=["x0"]), "ValueError",
              "columns")


def test_resident_part_and_design_matrix():
    sp, spec, _ = fixture()
    held = sp.load_part(spec, "valid")
    resident = sp.load_resident(spec, "valid", "cpu")
    assert torch.equal(resident.x_num, held.x_num) and torch.equal(resident.anchors, held.anchors)
    design = sp.design_matrix(spec, held)
    assert design.x.shape == (len(held), L * 3 + L * 1) and design.x.dtype == np.float32
    assert design.y.shape == (len(held), H)
    assert np.array_equal(design.row_ids, held.row_ids)
    b = batch_of(sp, spec, held)
    assert np.array_equal(design.x[:, :L * 3], b["x_num"].reshape(len(held), -1).numpy())
    fx.raises(lambda: sp.design_matrix(spec, held, "neighbour-mean"), "ValueError", "graph part")


def test_target_scale_is_the_train_seasonal_difference():
    frame = raw_frame()
    frame["target"] = np.sin(frame["x0"].to_numpy()) + frame["ts"].to_numpy() * 0.5
    sp, spec, _ = fixture(frame)
    train = frame[frame.ts < T_TRAIN].sort_values(["series", "ts"])
    for season in (1, 4):
        want = train.groupby("series")["target"].diff(season).abs().dropna().mean()
        assert abs(sp.target_scale(spec, season) - want) < 1e-12
    task = {"kind": "sequence-regression", "base_kind": "regression", "label": "x1"}
    sp2, spec2, _ = fixture(frame, task=task)
    assert sp2.target_scale(spec2, 2) is None
    assert not hasattr(sp, "exposure"), "a window package's DataSource carries no graph exposure"


def test_template_discipline():
    """C11–C13 on the rendered window files; C8 finds the seal on the adapter class."""
    import conformance
    files = [NS.root / "ingestion" / n for n in ("data.py", "frames.py", "windows.py")]
    _, findings = conformance.size_only([str(f) for f in files])
    assert not findings, [f.message() for f in findings]
    pkg, err = conformance.Package.read(NS.root, NS.blueprint_map, NS.pipeline)
    assert err is None, err
    marks = conformance.c8_datasource_seals_the_test(pkg)
    assert all(m.level.name == "OK" for m in marks), [m.line() for m in marks]
    assert "{{" not in (NS.root / "ingestion" / "windows.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the real shaper

def test_real_shaper_time_split_windows():
    """A multi-series csv shaped by shape.py (time split, nominal encoded, numerics scaled):
    every forecast window's targets are the raw file's next H values of its anchor's series."""
    work = Path(tempfile.mkdtemp(prefix="windows-shaper-"))
    frame = raw_frame(n_steps=60).drop(columns=["x0", "state"])
    rng = np.random.default_rng(5)
    frame["x0"] = rng.normal(size=len(frame))
    frame["target"] = rng.normal(size=len(frame)).cumsum()
    frame["ts"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(frame["ts"], unit="h")
    frame["zone"] = frame["zone"].map(lambda z: f"z{z}")
    csv = work / "series.csv"
    frame.to_csv(csv, index=False)
    steps = [fx.step("S1", "retype", ["zone"], {"to": "nominal"}, "geometry:typing/zone"),
             fx.step("S2", "select_partition", ["target"], {"label": "target",
                                                          "task": "regression"},
                     "geometry:partitions/target"),
             fx.step("S3", "split", [], {"kind": "time", "time_column": "ts", "test": 0.2,
                                          "valid": 0.2}, "analysis:F1"),
             fx.step("S4", "encode", ["zone"], {"strategy": "ordinal"}, "shaper:default"),
             fx.step("S5", "scale", [], {"kind": "standard", "columns": ["x0", "x1"]},
                     "shaper:default")]
    man = fx.run_shaper(work, csv, fx.recipe(csv, steps, "target"))
    shaped = pd.read_parquet(sorted(man.parent.glob("*.train.parquet"))[0])
    card = int(shaped["zone"].max()) + 2
    spec = fx.rt.reference_spec("window")
    dc = spec["data_contract"]
    dc["features"] = [fx.numeric("x0"), fx.numeric("x1"), fx.nominal("zone", card)]
    dc["input_dim"] = 3
    dc["sample"].update({"length": 6, "horizon": H, "series_by": "series", "order_by": "ts"})
    spec = D.NetSpec.from_dict(fx.spec_on(spec, man))
    sp = NS.data.ShapedParts()
    raw = pd.read_csv(csv)
    for part in ("train", "valid"):
        windows = sp.load_part(spec, part)
        b = batch_of(sp, spec, windows)
        for i in range(0, len(windows), 7):
            anchor = raw.iloc[int(windows.row_ids[i])]
            later = raw[(raw.series == anchor.series) & (raw.ts > anchor.ts)].sort_values("ts")
            assert np.allclose(b["y"][i].numpy(), later.target.to_numpy()[:H], atol=1e-5), part
            assert np.isclose(float(b["x_num"][i, -1, -1]), anchor.target, atol=1e-5)


def main(argv) -> int:
    return fx.main_for(globals(), argv, "window")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
