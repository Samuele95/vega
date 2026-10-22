# data-lens fixtures

`orders.csv` lives in `dataset-forge/fixtures/` and is the shared regression input for the
whole dataset lane (a geometry layer for it ships there too).

`sensors.csv` is this skill's own fixture: 1440 hourly rows from three stations, built to
exercise the modules a purely tabular fixture cannot reach.

| what it carries | which module it exercises |
|---|---|
| `ts` — hourly timestamps, one change point at row 900, daily seasonality, a slow trend | `time_series` |
| `lat` / `lon` — three spatial clusters around Turin | `spatial` |
| `split` — train / test by time, so the later part drifts | `drift` |
| `calibration` — missing at 62 % for the `hill` station and 6 % elsewhere | `quality` (missingness mechanism: MAR on `station`) |
| `alarm` — `reading_c > 24 AND humidity_pct < 62` | `importance` (learnable label with a real rule behind it) |
| `battery_pct` — a slow linear decay | `relations`, `time_series` |

Regenerate with the snippet in this skill's build notes; the generator is seeded (11) so the
file is stable. `analysis.py --seed 7` on it must produce the same JSON twice.
