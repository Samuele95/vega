"""
test_analysis.py — the data-lens analysis paths the gate C rewrite added, each on a synthetic table
whose answer is known (gate C's review panel: the rewrite had no committed test, and the importance
interval excluded the baseline on noise).

    python3 -m pytest -q data-lens/scripts/tests/test_analysis.py

- importance: on a noise-only label the CV interval (corrected resampled t, k−1 df) holds the
  baseline; with a datetime the folds chain forward in time; the leakage probe refits with the
  label's rule columns the partition left among the features and reports their jump;
- the Monte-Carlo chi-square is seeded and bounded, and the robust chi-square switches to Fisher on
  a 2×2 table, to the Monte-Carlo test on a larger one;
- the label-rate drift reads a two-proportion z when every count is ≥ 5, else Fisher's exact test
  with the Newcombe interval;
- Mann–Kendall takes the Hamed–Rao correction when the residual is serially dependent;
- the seasonal period is checked on each half (passed, violated, n/a);
- Moran's I and Geary's C read a smooth spatial field as clustered and noise as not.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as A  # noqa: E402


def _ctx(features, label, *, derivations=(), time=None, spatial=None, types=None):
    types = types or {}
    cols = list(features) + [label]
    return {"typing": [{"column": c, "type": types.get(c, "numeric"), "role": "dimension"} for c in cols],
            "basis": list(features), "derivations": list(derivations),
            "partition": {"label": label, "task": "binary classification", "features": list(features),
                          "leakage": [], "provenance": "single-candidate"},
            "time": time, "spatial": spatial, "split_column": None}


# --------------------------------------------------------------------------- importance
def test_a_noise_label_is_not_learnable_and_its_interval_holds_the_baseline():
    rng = np.random.default_rng(11)
    n = 300
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n),
                       "y": rng.integers(0, 2, n)})
    ev = A.mod_importance(df, _ctx(["a", "b", "c"], "y", types={"y": "boolean"}), seed=7)
    assert ev["metric"] == "roc_auc" and ev["baseline"]["score"] == 0.5
    assert "corrected resampled t" in ev["ci_method"]
    for model in ev["models"]:
        lo, hi = model["ci"]
        assert lo <= 0.5 <= hi, (model["name"], model["ci"], model["folds"])
        k, s = len(model["folds"]), float(np.std(model["folds"], ddof=1))
        naive = 1.96 * float(np.std(model["folds"])) / math.sqrt(k)
        assert (hi - lo) / 2 > naive, "wider than the normal, population-sd interval it replaces"
        assert s > 0


def test_time_ordered_rows_are_scored_by_forward_chaining():
    rng = np.random.default_rng(3)
    n = 360
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    x = rng.normal(size=n)
    df = pd.DataFrame({"ts": t.astype(str), "x": x, "z": rng.normal(size=n),
                       "y": (x + 0.3 * rng.normal(size=n) > 0).astype(int)})
    ev = A.mod_importance(df, _ctx(["x", "z"], "y", time={"column": "ts"}, types={"y": "boolean"}), seed=7)
    assert ev["folds"].startswith("forward chaining in time (by ts)"), ev["folds"]
    assert ev["models"] and all(len(m["folds"]) == 5 for m in ev["models"])
    assert max(m["cv_score"] for m in ev["models"]) > 0.8, "x carries the label"


def test_the_leakage_probe_refits_with_the_rule_columns_left_among_the_features():
    rng = np.random.default_rng(5)
    n = 300
    d = rng.normal(size=n)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "d": d, "y": (d > 0).astype(int)})
    rule = [{"column": "y", "body": ["d"], "rule_id": "swrl-y", "formula": "y = d > 0"}]
    ev = A.mod_importance(df, _ctx(["a", "b", "d"], "y", derivations=rule, types={"y": "boolean"}), seed=7)
    probe = ev["leakage_probe"]
    assert probe["label_derivations_in_features"] == ["d"] and probe["suspect"] == ["d"]
    assert "d" not in ev["features"], "the rule column never reaches the fit"
    assert probe["delta"] > 0.3, probe


def test_a_refuted_or_set_aside_rule_on_the_label_excludes_nothing():
    """Only kept rules derive a column (dataset-forge report-contract §2). The T5 rehearsal's
    forge recorded the refuted hypothesis `cover = f(every band, the terrain)` beside a set-aside
    partial rule; importance took the body of the last rule on the label without asking whether
    it was kept, fitted the tile id alone and scored below the majority baseline."""
    rng = np.random.default_rng(5)
    n = 300
    d = rng.normal(size=n)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "d": d, "y": (d > 0).astype(int)})
    refuted = {"column": "y", "body": ["a", "b", "d"], "rule_id": "swrl-y-all", "formula": "y = f(a, b, d)",
               "provenance": {"semantic": {"status": "refuted"}}}
    aside = {"column": "y", "body": ["d"], "rule_id": "swrl-y", "formula": "y = d > 0", "kept": False,
             "provenance": {"semantic": {"status": "confirmed"}}}
    geo = {"typing": [{"column": c, "final_type": "numeric", "role": "dimension"} for c in "abd"]
           + [{"column": "y", "final_type": "boolean", "role": "dimension"}],
           "basis": {"members": ["a", "b", "d"]}, "derivations": [aside, refuted],
           "partitions": {"chosen": "y", "candidates": [{"label": "y", "task": "binary classification",
                                                         "features": ["a", "b", "d"]}]}}
    ctx = A.build_context(df, geo, None)
    assert [x["kept"] for x in ctx["derivations"]] == [False, False]
    assert A.label_rule_columns(ctx, "y", df) == []
    ev = A.mod_importance(df, ctx, seed=7)
    assert ev["features"] == ["a", "b", "d"], ev["features"]
    assert max(m["cv_score"] for m in ev["models"]) > 0.8, "d reaches the fit"
    kept = dict(aside, kept=True)
    ctx = A.build_context(df, dict(geo, derivations=[kept, refuted]), None)
    assert [x["kept"] for x in ctx["derivations"]] == [True, False]
    assert A.label_rule_columns(ctx, "y", df) == ["d"], "the kept rule still keeps its body out"


def test_the_interaction_screen_never_reads_the_labels_rule_columns_nor_runs_without_a_label():
    rng = np.random.default_rng(9)
    n = 300
    a, b, c, d = (rng.normal(size=n) for _ in range(4))
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "y": a * b + 0.1 * rng.normal(size=n)})
    rule = [{"column": "y", "body": ["a", "b"], "rule_id": "swrl-y", "formula": "y = a * b"}]
    ctx = _ctx(["a", "b", "c", "d"], "y", derivations=rule)
    pairs = {(r["a"], r["b"]) for r in A.mod_relations(df, ctx, seed=3)["interactions"]}
    assert pairs == {("c", "d")}, pairs
    assert {(r["a"], r["b"]) for r in A.mod_relations(df, _ctx(["a", "b", "c", "d"], "y"), seed=3)["interactions"]} \
        >= {("a", "b")}, "without the rule the product is screened"
    none = _ctx(["a", "b", "c", "d"], "y")
    none["partition"]["label"] = None
    assert A.mod_relations(df, none, seed=3)["interactions"] == [], "no label, no screen"


# --------------------------------------------------------------------------- chi-square
def test_the_monte_carlo_chi_square_is_seeded_and_bounded():
    table = [[30, 5, 2], [10, 25, 3]]
    p1, b1 = A.mc_chi2_p(table, seed=1)
    p2, _ = A.mc_chi2_p(table, seed=1)
    assert p1 == p2 and b1 == 999 and 1 / 1000 <= p1 < 0.01
    null, _ = A.mc_chi2_p([[10, 10, 10], [10, 10, 10]], seed=1)
    assert null > 0.5


def test_the_robust_chi_square_switches_by_the_tables_shape():
    two = A.robust_chi2([[3, 1], [1, 3]], seed=1)
    assert two["switched_to"] == "fisher-exact" and 0 < two["p"] <= 1
    big = A.robust_chi2([[3, 1, 0], [1, 3, 2]], seed=1)
    assert big["switched_to"] == "monte-carlo chi-square" and big["permutations"] == 999


# --------------------------------------------------------------------------- label-rate drift
def _drift_ctx():
    return {"typing": [{"column": "x", "type": "numeric", "role": "dimension"},
                       {"column": "y", "type": "boolean", "role": "dimension"}],
            "basis": ["x"], "derivations": [], "split_column": "part",
            "partition": {"label": "y", "task": "binary", "features": ["x"], "leakage": [], "provenance": "single-candidate"}}


def test_label_rate_drift_reads_a_z_test_when_every_count_holds():
    rng = np.random.default_rng(2)
    n = 400
    part = np.array(["train"] * (n // 2) + ["test"] * (n // 2))
    y = np.concatenate([rng.random(n // 2) < 0.3, rng.random(n // 2) < 0.5]).astype(int)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": y})
    rate = A.mod_drift(df, _drift_ctx(), 7, pd.Series(part))["label_rate"]
    assert rate["assumptions_checked"] == "passed" and rate["test"] == "two-proportion z"
    assert rate["ci"][0] <= rate["risk_difference"] <= rate["ci"][1]
    assert rate["p"] < 0.01


def test_label_rate_drift_switches_to_fisher_and_newcombe_on_small_counts():
    part = np.array(["train"] * 40 + ["test"] * 40)
    y = np.array([1] * 2 + [0] * 38 + [1] * 9 + [0] * 31)
    df = pd.DataFrame({"x": np.arange(80, dtype=float), "y": y})
    rate = A.mod_drift(df, _drift_ctx(), 7, pd.Series(part))["label_rate"]
    assert rate["assumptions_checked"] == "violated"
    assert rate["switched_to"] == "fisher-exact with the Newcombe interval"
    lo, hi = rate["ci"]
    assert lo <= rate["risk_difference"] <= hi and -1 <= lo and hi <= 1
    assert 0 < rate["p"] < 0.05


# --------------------------------------------------------------------------- time series
def test_mann_kendall_takes_the_hamed_rao_correction_on_dependent_residuals():
    rng = np.random.default_rng(4)
    n = 200
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.85 * e[i - 1] + rng.normal()
    y = 0.02 * np.arange(n) + e
    dep = A.mann_kendall(y)
    assert dep["assumptions_checked"] == "violated"
    assert dep["robust"]["variance_ratio"] > 1 and dep["p_reported"] == dep["robust"]["p"] >= dep["p"]
    flat = A.mann_kendall(rng.normal(size=n))
    assert flat["assumptions_checked"] == "passed" and flat["p_reported"] == flat["p"]


def test_the_seasonal_period_is_found_again_on_each_half():
    t = np.arange(240)
    stable = A._period_stability(np.sin(2 * np.pi * t / 12), 12)
    assert stable["assumptions_checked"] == "passed" and stable["halves"] == [12, 12]
    shifted = np.concatenate([np.sin(2 * np.pi * t[:120] / 12), np.sin(2 * np.pi * t[120:] / 24)])
    moved = A._period_stability(shifted, 12)
    assert moved["assumptions_checked"] == "violated" and moved["robust"]["period_by_half"][0] == 12
    assert A._period_stability(np.sin(2 * np.pi * t[:30] / 12), 12)["assumptions_checked"] == "n/a"


# --------------------------------------------------------------------------- spatial
def test_morans_i_and_gearys_c_read_a_smooth_field_as_clustered():
    rng = np.random.default_rng(9)
    la, lo = 45 + rng.random(200) * 0.5, 9 + rng.random(200) * 0.5
    df = pd.DataFrame({"lat": la, "lon": lo, "v": np.sin(la * 20) + np.cos(lo * 20)})
    ctx = {"typing": [{"column": c, "type": "numeric", "role": "dimension"} for c in ("lat", "lon", "v")],
           "basis": ["v"], "spatial": {"crs": None, "columns": [{"kind": "latlon", "column": "lat", "pair": "lon"}]}}
    rec = A.mod_spatial(df, ctx, seed=7)["morans_i"][0]
    assert rec["I"] > rec["expected_I"] and rec["pseudo_p"] < 0.01
    assert rec["gearys_c"]["C"] < 1 and rec["gearys_c"]["pseudo_p"] < 0.01
    noise = A.mod_spatial(df.assign(v=np.random.default_rng(2).normal(size=200)), ctx, seed=7)["morans_i"][0]
    assert noise["pseudo_p"] > 0.01 and noise["gearys_c"]["pseudo_p"] > 0.01


# --------------------------------------------------------------------------- the whole pass
def test_the_whole_pass_runs_every_module_on_a_dated_located_split_table(tmp_path):
    """analysis.py as the skill runs it, on the sensors fixture (a datetime, a latitude/longitude
    pair, a split column) read against its geometry layer (station chosen): time_series, spatial,
    drift and importance all run — the paths no matrix lane reaches (gate C) — and the run is
    kept beside the test's work directory as its evidence."""
    import json
    import subprocess
    here = Path(__file__).resolve().parents[2]
    layer = (here / "fixtures" / "sensors.geometry-layer.json").read_text(encoding="utf-8")
    model = tmp_path / "sensors.model.html"
    model.write_text('<html><script id="domain-model" type="text/turtle">ex:x ex:sourceKind "dataset" .</script>'
                     f'<script id="layer-geometry-data" type="application/json">{layer}</script></html>',
                     encoding="utf-8")
    out = tmp_path / "analysis.json"
    run = subprocess.run([sys.executable, str(here / "scripts" / "analysis.py"), str(here / "fixtures" / "sensors.csv"),
                          "--model", str(model), "--split", "split", "--out", str(out), "--seed", "7"],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    doc = json.loads(out.read_text(encoding="utf-8"))
    mods = doc["modules"]
    for name in ("time_series", "spatial", "drift", "importance"):
        assert mods[name].get("ran", True) is not False, (name, mods[name].get("skipped_because"))
    imp = mods["importance"].get("evidence") or mods["importance"]
    assert imp["folds"].startswith("forward chaining in time"), imp["folds"]
    assert imp["models"] and all(m["ci"][0] <= m["cv_score"] <= m["ci"][1] for m in imp["models"])
    spatial = mods["spatial"].get("evidence") or mods["spatial"]
    assert spatial["morans_i"] and "gearys_c" in spatial["morans_i"][0]
    drift = mods["drift"].get("evidence") or mods["drift"]
    assert drift["parts"] and drift["columns"]


# --------------------------------------------------------------------------- type coherence
def test_a_tile_code_is_not_a_date_stored_as_text_but_a_real_date_is():
    # release3 final3r1: `T09` parses as 9 January of year 1; the lens called the tile a date
    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame({"tile": [f"T{k:02d}" for k in rng.integers(1, 25, n)],
                       "when": pd.date_range("2021-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
                       "x": rng.normal(size=n), "y": rng.integers(0, 2, n)})
    ctx = _ctx(["tile", "when", "x"], "y", types={"tile": "nominal", "when": "nominal", "y": "boolean"})
    ev = A.mod_quality(df, ctx, seed=1)
    dated = {t["column"] for t in ev["type_coherence"] if t["issue"] == "date stored as text"}
    assert dated == {"when"}, ev["type_coherence"]


# --------------------------------------------------------------------------- segments
def test_the_segments_record_the_rule_that_places_a_row_in_one():
    """The profiles describe a segment; `evidence.assignment` places a row in one (the k-means
    prediction: the nearest centroid in the standardized basis), so a reader with the raw
    columns — /vega-eval cutting segment slices of its test rows — reproduces every row's
    segment without the lens's code."""
    rng = np.random.default_rng(5)
    centres = np.array([[0.0, 0.0, 10.0], [8.0, 1.0, 30.0], [2.0, 9.0, 50.0]])
    rows = np.concatenate([c + rng.normal(scale=0.6, size=(120, 3)) for c in centres])
    df = pd.DataFrame(rows, columns=["a", "b", "c"])
    df["y"] = rng.integers(0, 2, len(df))
    ev = A.mod_segments(df, _ctx(["a", "b", "c"], "y", types={"y": "boolean"}), seed=7)
    assert ev["chosen_k"] == 3
    rule = ev["assignment"]
    assert rule["features"] == ["a", "b", "c"] and rule["segments"] == [0, 1, 2]
    z = (df[rule["features"]].to_numpy() - np.array(rule["center"])) / np.array(rule["scale"])
    placed = ((z[:, None, :] - np.array(rule["centroids"])[None]) ** 2).sum(-1).argmin(1)
    sizes = np.bincount(placed, minlength=3)
    assert sorted(sizes.tolist()) == sorted(p["n"] for p in ev["profiles"]) == [120, 120, 120]
    for p in ev["profiles"]:
        assert sizes[p["segment"]] == p["n"]
