#!/usr/bin/env python3
"""
test_transform_lag.py — a step after `select_partition` that reads the label, served by the
transform-only module, and the row order a lag step leaves a part in.

    python3 test_transform_lag.py [<work-dir>] [-v]

Four recipes over one generated table (three interleaved series written one after the other,
so a time split lists each part's rows in time order and not in raw-file order), each through
the real scripts: `shape.py`, `verify_shape.py`, `shape.py --check`, `--transform-only` and the
twin check on every part — the two the shaper accepts; the other two it refuses at `shape.py`.

  label-lag    the lag of the label (phase 4) after `select_partition` (phase 2): the module
               keeps the label until the lag has read it, drops it after, and reads a lagged
               label from the history — refusing a history without it (shape-contract §2).
  group-label  an `impute` grouped by the label is refused (A46): grouped by its own label, a
               test row's filled value carries the label into its features, and a served query
               has no label to group by. The recipe checker refuses it before any data moves,
               whatever the source; `run_negatives_leakage.py` seeds the executor and the
               verifier with it too.
  lag-label    a `lag` grouped by the label is refused like the impute (A47(5)): its lagged
               value is read from the previous row sharing the row's label, so the label
               reaches the features, and a served query has no label to group its history by.
               The rule itself (`label_grouping_error`) is read directly: a lag grouped by the
               label, as one name or a list, is refused naming the step and the label; a lag
               grouped by the series key, or the lag of the label itself, is not. Then the
               recipe checker refuses it before any data moves, whatever the source;
               `run_negatives_leakage.py` (SL12) seeds the executor and the verifier with it.
  feature-lag  the lag of a feature: the lag writes back by row label, so every part stays in
               the order `__split_raw__` records; `verify_shape.py` reads that order on the
               columns carried through unchanged and fails a part put back in raw-file order.

Exit 0 when every check holds.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow.fs import LocalFileSystem      # A40: never a Python file object

HERE = Path(__file__).resolve().parent
SHAPE, VERIFY = HERE.parent / "shape.py", HERE.parent / "verify_shape.py"
VERBOSE = "-v" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and (VERBOSE or not cond) else ""))
    return bool(cond)


def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def tail(out, n=3):
    return " | ".join(out.strip().splitlines()[-n:])


def table(path):
    """Three series of 48 hours, written series after series; `demand` follows its own past."""
    rng = np.random.default_rng(3)
    frames = []
    for k, station in enumerate(("s0", "s1", "s2")):
        n = 48
        x = rng.normal(size=n)
        demand = np.round(10 + 3 * np.sin(np.arange(n) / 4 + k) + x, 4)
        frames.append(pd.DataFrame({
            "row_key": [f"{station}-{i}" for i in range(n)], "station": station,
            "ts": pd.date_range("2026-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
            "x": np.round(x, 6), "grade": np.where(demand > 10, "high", "low"),
            "demand": demand}))
    df = pd.concat(frames, ignore_index=True)
    df.loc[[5, 60, 101], "x"] = np.nan                         # for the grouped impute
    df.to_csv(path, index=False)
    return df


def step(sid, op, columns, params, source="user:test"):
    return {"id": sid, "op": op, "columns": columns, "params": params, "source": source,
            "rationale": f"{op} for the test", "consequences":
            {"rows": "unchanged", "columns": "unchanged", "downstream": "none"}}


def recipe(paths, label, split, middle, scaled):
    """A recipe from the raw csv to `out` (paths): phases 1-3 around the label, the split, the
    steps under test, then the scaling."""
    csv, out = paths
    head = [step("S01", "retype", ["station"], {"to": "nominal"}),
            step("S02", "drop_identity", ["row_key"], {}),
            step("S03", "parse_datetime", ["ts"], {"errors": "coerce"}),
            step("S04", "select_partition", [label], {"label": label, "task": "regression"},
                 source=f"geometry:partitions/{label}"),
            step("S05", "split", [], split)]
    tail_ = [step("S09", "scale", scaled, {"kind": "standard", "columns": scaled})]
    return {"schema": "dataset-shaper/recipe@1",
            "input": {"path": str(csv), "rows": 144, "columns": 6}, "seed": 1,
            "goal": "train-ready", "target": label,
            "outputs": {"dir": str(out), "format": "parquet", "files": []},
            "steps": head + middle + tail_}


TIME = {"kind": "time", "time_column": "ts", "test": 0.2, "valid": 0.2}
LAG = {"group_by": "station", "time_column": "ts"}


def load(path):
    spec = importlib.util.spec_from_file_location(f"pre_{abs(hash(str(path)))}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def through(name, rec, work):
    """The recipe through shape / verify / --check / --transform-only / the twin on every
    part; returns (out dir, fitted, module) or None."""
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    rp, out = d / "recipe.json", Path(rec["outputs"]["dir"])
    rp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    rc, log = _run(SHAPE, "--recipe", rp, "--out-dir", out)
    if not check(f"{name}: shape.py exits 0", rc == 0, tail(log)):
        return None
    rc, log = _run(VERIFY, "--recipe", rp, "--out-dir", out, "--json", d / "verification.json")
    check(f"{name}: verify_shape.py exits 0", rc == 0, tail(log))
    check(f"{name}: verify reads the row order on the carried columns",
          "keep the row order __split_raw__ records" in log, tail(log))
    rc, log = _run(SHAPE, "--check", "--out-dir", out)
    check(f"{name}: shape.py --check reproduces every part", rc == 0, tail(log))
    module = out / "preprocess.py"
    rc, log = _run(SHAPE, "--transform-only", "--recipe", rp, "--fitted", out / "fitted.json",
                  "--emit", module)
    if not check(f"{name}: --transform-only emits the module", rc == 0, tail(log)):
        return None
    for part in ("train", "valid", "test"):
        rc, log = _run(VERIFY, "--transform-only", "--recipe", rp, "--out-dir", out,
                      "--module", module, "--part", part)
        check(f"{name}: the twin check agrees on the {part} part", rc == 0, tail(log, 4))
    check(f"{name}: the module is at most 400 lines",
          len(module.read_text(encoding="utf-8").splitlines()) <= 400)
    fitted = json.loads((out / "fitted.json").read_text(encoding="utf-8"))
    return out, fitted, load(module)


def label_lag(raw, csv, work):
    lag = [step("S06", "lag", ["demand"], {"lags": [1, 2], **LAG}),
           step("S07", "impute", ["demand_lag1", "demand_lag2"], {"strategy": "median"})]
    got = through("label-lag", recipe((csv, work / "label-lag/shaped"), "demand", TIME, lag,
                                      ["x", "demand_lag1", "demand_lag2"]), work)
    if got is None:
        return
    out, fitted, mod = got
    needs = getattr(mod, "HISTORY_COLUMNS", None) or []
    check("label-lag: the history must carry the label, a query need not",
          "demand" in needs and "demand" not in mod.REQUIRED_COLUMNS,
          f"{needs} / {mod.REQUIRED_COLUMNS}")
    test = fitted["__split_raw__"]["test"]
    history = raw.drop(index=test)                       # every row before the test part
    query = raw.iloc[test]
    served = mod.transform(query.drop(columns=["demand"]), history=history)
    check("label-lag: a query without the label is served, and the label does not come back",
          "demand" not in served.columns and len(served) == len(query), list(served.columns))
    labelled = mod.transform(query, history=history)
    first = query.groupby("station", sort=False).head(1).index
    check("label-lag: a series' first query row reads its lags from the history",
          np.allclose(served.loc[first, ["demand_lag1", "demand_lag2"]].to_numpy(),
                      labelled.loc[first, ["demand_lag1", "demand_lag2"]].to_numpy()))
    try:
        mod.transform(query, history=history.drop(columns=["demand"]))
        check("label-lag: a history without the label is refused", False, "served")
    except KeyError as ex:
        check("label-lag: a history without the label is refused, naming it",
              "demand" in str(ex), str(ex))
    try:
        mod.transform(query)
        check("label-lag: no history is refused", False, "served")
    except ValueError as ex:
        check("label-lag: no history is refused, naming the flag", "--history" in str(ex))


def group_label(raw, csv, work):
    split = {"kind": "stratified", "by": "grade", "test": 0.2, "valid": 0.2}
    impute = [step("S06", "impute", ["x"], {"strategy": "group-median", "by": ["grade"]})]
    rec = recipe((csv, work / "group-label/shaped"), "grade", split, impute, ["x"])
    rec["steps"][3]["params"]["task"] = "binary classification"
    d = work / "group-label"
    d.mkdir(parents=True, exist_ok=True)
    rp = d / "recipe.json"
    rp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    rc, log = _run(SHAPE, "--recipe", rp, "--out-dir", rec["outputs"]["dir"])
    check("group-label: an impute grouped by the label is refused before any data moves",
          rc == 2 and "grouped by the label 'grade'" in log, tail(log))
    check("group-label: nothing was written", not Path(rec["outputs"]["dir"]).exists())


def lag_label(raw, csv, work):
    sh = load(SHAPE)
    by_label = step("S06", "lag", ["x"], {"lags": [1], "group_by": "grade", "time_column": "ts"})
    why = sh.label_grouping_error(by_label, "grade")
    check("lag-label: the rule refuses a lag grouped by the label, naming the step and the label",
          why is not None and "lag S06 grouped by the label 'grade'" in why, why)
    as_list = dict(by_label, params=dict(by_label["params"], group_by=["station", "grade"]))
    check("lag-label: group_by as a list holding the label is refused too",
          sh.label_grouping_error(as_list, "grade") is not None)
    check("lag-label: a lag grouped by the series key is not refused",
          sh.label_grouping_error(step("S06", "lag", ["x"], dict(LAG, lags=[1])), "grade") is None)
    check("lag-label: the lag of the label grouped by the series key is not refused",
          sh.label_grouping_error(step("S06", "lag", ["demand"], LAG), "demand") is None)
    check("lag-label: the checker lists it among the leakage errors",
          any("lag S06 grouped by the label 'grade'" in e for e in sh.leakage_errors(
              [step("S04", "select_partition", ["grade"], {"label": "grade"}), by_label], None)))
    lag = [by_label, step("S07", "impute", ["x", "x_lag1"], {"strategy": "median"})]
    rec = recipe((csv, work / "lag-label/shaped"), "grade", TIME, lag, ["x", "x_lag1"])
    rec["steps"][3]["params"]["task"] = "binary classification"
    d = work / "lag-label"
    d.mkdir(parents=True, exist_ok=True)
    rp = d / "recipe.json"
    rp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    rc, log = _run(SHAPE, "--recipe", rp, "--out-dir", rec["outputs"]["dir"])
    check("lag-label: a lag grouped by the label is refused before any data moves",
          rc == 2 and "lag S06 grouped by the label 'grade'" in log, tail(log))
    check("lag-label: nothing was written", not Path(rec["outputs"]["dir"]).exists())
    rc, log = _run(SHAPE, "--recipe", rp, "--check-only")
    check("lag-label: --check-only refuses it too", rc == 2 and "grouped by the label" in log,
          tail(log))


def feature_lag(raw, csv, work):
    lag = [step("S06", "lag", ["x"], {"lags": [1], **LAG}),
           step("S07", "impute", ["x", "x_lag1"], {"strategy": "median"})]
    got = through("feature-lag", recipe((csv, work / "feature-lag/shaped"), "demand", TIME, lag,
                                        ["x", "x_lag1"]), work)
    if got is None:
        return
    out, fitted, _ = got
    sets = fitted["__split_raw__"]
    check("feature-lag: the time split interleaves the series (a part is not in raw order)",
          any(v != sorted(v) for v in sets.values()))
    for part, pos in sets.items():
        shaped = pd.read_parquet(out / f"table.{part}.parquet", filesystem=LocalFileSystem())
        check(f"feature-lag: the {part} part is in __split_raw__ order",
              np.array_equal(shaped["demand"].to_numpy(), raw["demand"].to_numpy()[pos]))
    # The seeded defect: a part put back in raw-file order (what `sort_index` did) fails verify.
    seeded = out.parent / "seeded"
    shutil.copytree(out, seeded)
    train = pd.read_parquet(seeded / "table.train.parquet", filesystem=LocalFileSystem())
    order = np.argsort(np.asarray(sets["train"]), kind="stable")
    train.iloc[order].reset_index(drop=True).to_parquet(seeded / "table.train.parquet")
    rc, log = _run(VERIFY, "--recipe", out.parent / "recipe.json", "--out-dir", seeded)
    check("feature-lag: verify fails a part in raw-file order, naming __split_raw__",
          rc == 3 and "not in the order __split_raw__ records" in log, tail(log))


def main():
    work = Path(ARGS[0]) if ARGS else Path(tempfile.mkdtemp(prefix="shaper-lag-"))
    work.mkdir(parents=True, exist_ok=True)
    csv = work / "table.csv"
    raw = table(csv)
    for case in (label_lag, group_label, lag_label, feature_lag):
        name = case.__name__.replace("_", "-")
        print(name)
        try:
            case(raw, csv, work)
        except Exception as ex:                       # a case that crashes is a failed case
            check(f"{name}: runs to the end", False, f"{type(ex).__name__}: {ex}")
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed ({work})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
