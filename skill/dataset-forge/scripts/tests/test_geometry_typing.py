"""
test_geometry_typing.py — the typing heuristic and the dependency screen of geometry.py on the
columns a real rehearsal tripped on (release3 final3r1, landcover seed 6173):

- a scene-tile code (`T01`…`T24`) is nominal, not a datetime: the lenient parser reads `T09` as
  9 January of year 1, and a time of day as today — neither is a date;
- real dates (ISO, slashed, month names) still type as datetime;
- the normalised mutual information never takes the square root of a negative entropy: a
  one-bin marginal can sum to 1 + ulp, and the screen crashed with `math domain error`;
- geometry.py runs end to end on the landcover fixture of that seed.

    python3 -m pytest -q dataset-forge/scripts/tests/test_geometry_typing.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import geometry as G  # noqa: E402


def _types(df):
    return {c: i["type"] for c, i in G.type_columns(df).items()}


def test_a_tile_code_is_nominal_not_a_datetime():
    rng = np.random.default_rng(0)
    tiles = [f"T{k:02d}" for k in rng.integers(1, 25, 400)]
    df = pd.DataFrame({"tile": tiles, "x": rng.normal(size=400)})
    assert _types(df)["tile"] == "nominal"
    assert not G.date_like(pd.Series(tiles))


def test_a_time_of_day_is_not_a_date():
    times = pd.Series([f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)])
    assert not G.date_like(times)


def test_real_dates_still_type_as_datetime():
    days = pd.date_range("2020-01-01", periods=300, freq="D")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%b %d %Y"):
        df = pd.DataFrame({"when": days.strftime(fmt), "x": np.arange(300)})
        assert _types(df)["when"] == "datetime", fmt


def test_normalized_mi_survives_a_one_bin_marginal():
    # 1800 rows in one bin: px sums to 1 + ulp in floating point on some row counts
    for n in (3, 7, 1800, 1801, 9999):
        a = pd.Series(["2020-01"] * n)
        b = pd.Series(np.arange(n) % 5).astype(str)
        assert G.normalized_mi(a, b) == 0.0
        assert G.normalized_mi(b, a) == 0.0


def test_geometry_runs_end_to_end_on_the_rehearsal_landcover_seed(tmp_path):
    tests = HERE.parents[2] / "vega" / "scripts" / "tests"
    sys.path.insert(0, str(tests))
    from lanekit import datasets  # noqa: E402
    csv = tmp_path / "landcover.csv"
    datasets.load("landcover").make(6173, 1800).to_csv(csv, index=False)
    out = tmp_path / "geometry.json"
    r = subprocess.run([sys.executable, str(HERE.parent / "geometry.py"), str(csv), "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    cols = json.loads(out.read_text())["columns"]
    assert cols["tile"]["type"] == "nominal"
    assert cols["parcel_id"]["type"] == "id"
