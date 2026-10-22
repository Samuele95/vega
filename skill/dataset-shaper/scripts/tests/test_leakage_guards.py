#!/usr/bin/env python3
"""
test_leakage_guards.py — what the shaper's leakage guards let through, and what they record
(A46 track S; `shape-contract.md` §1-§3, `step-catalog.md` phases 4 and 6).

    python3 test_leakage_guards.py [<work-dir>] [-v]

`run_negatives_leakage.py` seeds the defects each guard refuses. This is the other half: the
recipes the guards must *not* refuse, through the real scripts (`shape.py`, `verify_shape.py`,
`shape.py --check`, `--transform-only` and the twin check on every part), and what they write.

  fit-on       box-cox and yeo-johnson record `fit_on: "train"` after a split (`"all"` without
               one), so does a `select_features kind=variance`; a closed-form `log1p` records
               none; the λ the manifest holds is the train part's, recomputed here.
  neighbours   `spatial_features` reads its neighbours from an index fitted on the train part:
               the index holds the train rows alone; a test row's count and nearest distance
               are the train rows', recomputed here without the executor's code; the values
               do not depend on the batch (one row served alone reads what the part records);
               the twin check holds on every part; `bearing_to` is the initial bearing to the
               point `to` names.
  row-local    a `custom` step before the split that filters and parses row by row passes the
               executor's row-locality probe, which the manifest records; its parsing half
               after the split runs per part and is not probed; both are served (a custom
               step's namespace names numpy, which the transform-only module now imports).

Exit 0 when every check holds.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow.fs import LocalFileSystem      # A40: never a Python file object

HERE = Path(__file__).resolve().parent
SHAPE, VERIFY = HERE.parent / "shape.py", HERE.parent / "verify_shape.py"
EARTH_R = 6371008.8
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
    """240 readings at 60 places around Turin (each place read four times, so a location
    repeats), a positive skewed `flow`, a code `code_txt` to parse, a binary label `busy`."""
    rng = np.random.default_rng(11)
    n_loc, reps = 60, 4
    lat = np.round(45.07 + rng.normal(scale=0.01, size=n_loc), 6)
    lon = np.round(7.68 + rng.normal(scale=0.014, size=n_loc), 6)
    rows = []
    for r in range(reps):
        for i in range(n_loc):
            flow = float(np.round(np.exp(rng.normal(1.5, 0.6)), 4))
            rows.append({"reading": f"r{r}-{i}", "lat": lat[i], "lon": lon[i], "flow": flow,
                         "noise": float(np.round(rng.normal(), 6)),
                         "code_txt": f"c{int(rng.integers(0, 5))}",
                         "busy": int(flow > 4.5)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def step(sid, op, columns, params, source="user:test"):
    return {"id": sid, "op": op, "columns": columns, "params": params, "source": source,
            "rationale": f"{op} for the test", "alternatives": [],
            "consequences": {"rows": "unchanged", "columns": "unchanged", "downstream": "none"}}


SPLIT = {"kind": "stratified", "by": "busy", "test": 0.25, "valid": 0.15}


def recipe(csv, out, middle, *, split=True, pre=(), tail_=()):
    """retype / drop_identity / select_partition (`busy`), the `pre` steps, the split (unless
    `split=False`), the steps under test, then `tail_`."""
    head = [step("S01", "drop_identity", ["reading"], {}),
            *pre,
            step("S02", "select_partition", ["busy"],
                 {"label": "busy", "task": "binary classification"},
                 source="geometry:partitions/busy")]
    if split:
        head.append(step("S03", "split", [], SPLIT))
    return {"schema": "dataset-shaper/recipe@1",
            "input": {"path": str(csv), "rows": 240, "columns": 7}, "seed": 3,
            "goal": "custom", "target": "busy",
            "outputs": {"dir": str(out), "format": "parquet", "files": []},
            "steps": head + list(middle) + list(tail_)}


def geometry():
    return step("S00", "parse_geometry", ["lat", "lon"],
                {"kind": "latlon", "lat": "lat", "lon": "lon", "crs": "EPSG:4326"},
                source="user:the readings are WGS84 degrees")


def load(path):
    spec = importlib.util.spec_from_file_location(f"pre_{abs(hash(str(path)))}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_part(out, part):
    return pd.read_parquet(out / f"table.{part}.parquet", filesystem=LocalFileSystem())


def through(name, rec, work, parts=("train", "valid", "test")):
    """The recipe through shape / verify / --check / --transform-only / the twin on every
    part; returns (out dir, manifest, fitted, module) or None."""
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    rp, out = d / "recipe.json", Path(rec["outputs"]["dir"])
    rp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    rc, log = _run(SHAPE, "--recipe", rp, "--out-dir", out)
    if not check(f"{name}: shape.py exits 0", rc == 0, tail(log)):
        return None
    rc, log = _run(VERIFY, "--recipe", rp, "--out-dir", out, "--json", d / "verification.json")
    check(f"{name}: verify_shape.py exits 0 and reads the leakage rules", rc == 0 and
          "OK: leakage:" in log, tail(log))
    rc, log = _run(SHAPE, "--check", "--out-dir", out)
    check(f"{name}: shape.py --check reproduces every part", rc == 0, tail(log))
    module = out / "preprocess.py"
    rc, log = _run(SHAPE, "--transform-only", "--recipe", rp, "--fitted", out / "fitted.json",
                   "--emit", module)
    if not check(f"{name}: --transform-only emits the module", rc == 0, tail(log)):
        return None
    for part in parts:
        rc, log = _run(VERIFY, "--transform-only", "--recipe", rp, "--out-dir", out,
                       "--module", module, "--part", part)
        check(f"{name}: the twin check agrees on the {part} part", rc == 0, tail(log, 4))
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    fitted = json.loads((out / "fitted.json").read_text(encoding="utf-8"))
    return out, man, fitted, load(module)


def fit_on_of(man, sid):
    return next(s for s in man["steps"] if s["id"] == sid)["fit_on"]


# --------------------------------------------------------------------------- cases
def fit_on(raw, csv, work):
    from scipy import stats
    middle = [step("S04", "transform", ["flow"], {"kind": "box-cox"}),
              step("S05", "transform", ["noise"], {"kind": "yeo-johnson"}),
              step("S06", "transform", ["lat"], {"kind": "log1p"}),
              step("S07", "select_features", [], {"kind": "variance", "threshold": 0.0})]
    got = through("fit-on", recipe(csv, work / "fit-on/shaped", middle), work)
    if got is None:
        return
    out, man, fitted, _ = got
    for sid, kind in (("S04", "box-cox"), ("S05", "yeo-johnson")):
        check(f"fit-on: {kind} records fit_on 'train'", fit_on_of(man, sid) == "train",
              fit_on_of(man, sid))
    check("fit-on: a variance selection records fit_on 'train'", fit_on_of(man, "S07") == "train")
    check("fit-on: a closed-form log1p records no fit_on", fit_on_of(man, "S06") is None)
    train = raw.iloc[fitted["__split_raw__"]["train"]]
    _, lam = stats.boxcox(train["flow"].to_numpy(dtype=float))
    _, lam_all = stats.boxcox(raw["flow"].to_numpy(dtype=float))
    rec = fitted["S04"]["fitted"]["flow"]["lambda"]
    check("fit-on: the box-cox λ is the train part's (not the whole file's)",
          rec == float(lam) and rec != float(lam_all), f"{rec} / train {lam} / all {lam_all}")
    # Without a split there is nothing to leak into, and the manifest says so.
    got = through("fit-on-all", recipe(csv, work / "fit-on-all/shaped", middle[:2], split=False),
                  work, parts=())
    if got is not None:
        check("fit-on: without a split box-cox and yeo-johnson record fit_on 'all'",
              fit_on_of(got[1], "S04") == "all" and fit_on_of(got[1], "S05") == "all")


def metric_xy(frame, proj):
    la = frame["lat"].to_numpy(dtype=float)
    lo = frame["lon"].to_numpy(dtype=float)
    x = np.radians(lo - proj["lon0"]) * EARTH_R * np.cos(np.radians(proj["lat0"]))
    return x, np.radians(la - proj["lat0"]) * EARTH_R


def neighbours(raw, csv, work):
    feats = {"features": ["n_neighbours_within", "nearest_distance", "centroid_x", "bearing_to"],
             "radius_m": 900, "to": {"lat": 45.0703, "lon": 7.6869}}
    rec = recipe(csv, work / "neighbours/shaped", [step("S04", "spatial_features", [], feats)],
                 pre=[geometry()])
    got = through("neighbours", rec, work)
    if got is None:
        return
    out, man, fitted, mod = got
    entry, sets = fitted["S04"], fitted["__split_raw__"]
    index, proj = entry["index"], entry["projection"]
    check("neighbours: the step records fit_on 'train', its index too",
          fit_on_of(man, "S04") == "train" and index["fit_on"] == "train")
    check("neighbours: the index holds the train rows and no other",
          index["n_fit"] == len(sets["train"]) == sum(index["counts"]),
          f"{index['n_fit']} / {len(sets['train'])}")
    # Recomputed without the executor: every train row against every test row, in metres.
    tx, ty = metric_xy(raw.iloc[sets["train"]], proj)
    qx, qy = metric_xy(raw.iloc[sets["test"]], proj)
    dist = np.hypot(qx[:, None] - tx[None, :], qy[:, None] - ty[None, :])
    away = np.where(dist > 0, dist, np.inf)
    test = read_part(out, "test")
    check("neighbours: a test row counts the train rows within the radius, its own place "
          "excluded", np.array_equal(test["n_within_900m"].to_numpy(),
                                     (away <= 900).sum(axis=1).astype(float)))
    check("neighbours: a test row's nearest distance is to the nearest other train place",
          np.allclose(test["nearest_neighbour_m"].to_numpy(), away.min(axis=1), rtol=0,
                      atol=1e-6))
    la1, lo1 = np.radians(raw["lat"].iloc[sets["test"]]), raw["lon"].iloc[sets["test"]]
    dl, la2 = np.radians(7.6869 - lo1), np.radians(45.0703)
    bearing = np.degrees(np.arctan2(np.sin(dl) * np.cos(la2), np.cos(la1) * np.sin(la2)
                                    - np.sin(la1) * np.cos(la2) * np.cos(dl))) % 360
    check("neighbours: bearing_to is the initial bearing to the point, in degrees from north",
          np.allclose(test["bearing_to_deg"].to_numpy(), bearing.to_numpy(), rtol=0, atol=1e-9))
    whole = mod.transform(raw.iloc[sets["test"]])
    one = mod.transform(raw.iloc[sets["test"][:1]])
    check("neighbours: one row served alone reads what it reads inside the batch",
          one[["n_within_900m", "nearest_neighbour_m"]].iloc[0].tolist()
          == whole[["n_within_900m", "nearest_neighbour_m"]].iloc[0].tolist())
    check("neighbours: the module carries no batch-dependence warning",
          "batch" not in Path(out / "preprocess.py").read_text(encoding="utf-8").lower())


ROW_LOCAL = ("def step(df, ctx):\n"
             "    df = df[df['noise'] > -2.5].copy()\n"
             "    df['code'] = df['code_txt'].str[1:].astype(int)\n"
             "    return df.drop(columns=['code_txt'])\n")


def row_local(raw, csv, work):
    pre = [step("S0c", "custom", [], {"phase": 1, "code": ROW_LOCAL})]
    got = through("row-local", recipe(csv, work / "row-local/shaped", [], pre=pre), work)
    if got is not None:
        probe = next(s for s in got[1]["steps"] if s["id"] == "S0c")["parameters_fitted"]
        probe = probe.get("row_local_probe") or {}
        check("row-local: the executor probed the pre-split custom step and recorded it",
              probe.get("passed") is True and probe.get("rows") == 240 and
              len(probe.get("slices") or []) == 4, probe)
        kept = sum(len(v) for v in got[2]["__split_raw__"].values())
        check("row-local: its filter reached the split", kept == int((raw["noise"] > -2.5).sum()),
              kept)
    # After the split a row filter would shape the corpus (a served batch would lose rows),
    # so the per-part case parses only.
    parse = ROW_LOCAL.replace("    df = df[df['noise'] > -2.5].copy()\n", "    df = df.copy()\n")
    after = [step("S0c", "custom", [], {"phase": 4, "code": parse})]
    got = through("row-local-after", recipe(csv, work / "row-local-after/shaped", after), work)
    if got is not None:
        rec = next(s for s in got[1]["steps"] if s["id"] == "S0c")["parameters_fitted"]
        check("row-local: after the split the step runs per part and is not probed",
              "row_local_probe" not in rec)


def main():
    work = Path(ARGS[0]) if ARGS else Path(tempfile.mkdtemp(prefix="shaper-leakage-"))
    work.mkdir(parents=True, exist_ok=True)
    csv = work / "table.csv"
    raw = table(csv)
    for case in (fit_on, neighbours, row_local):
        name = case.__name__.replace("_", "-")
        print(name)
        try:
            case(raw, csv, work)
        except Exception as ex:                       # a case that crashes is a failed case
            check(f"{name}: runs to the end", False, f"{type(ex).__name__}: {ex}")
    failed = [n for n, good in results if not good]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed ({work})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
