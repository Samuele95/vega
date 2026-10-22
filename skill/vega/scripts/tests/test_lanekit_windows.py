"""
The window generators' tests (BUILD-PLAN T06): `bikeshare`, `vibration` and `tank` keep the
lanekit.datasets protocol, are deterministic in (seed, rows), agree with every lane that names
them (L10-L18), leave every part of the time split a full window per series, put every class
and label of their lanes into every part, and plant a signal a window can learn. The recipe
each lane gets from them selects the lane's label, drops the other labels, splits in time, and
lags and encodes as the lane asks.

    python3 -m pytest -q vega/scripts/tests/test_lanekit_windows.py

The lanes themselves (fixture -> forge -> train -> eval) are the lane program's:
`matrix.sh lanes --lanes L10,L11,L12,L13,L14,L15,L16,L17,L18 --to eval`.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import datasets, registry  # noqa: E402
from lanekit.recipes import ColumnPlan, standard_recipe, time_split_cuts  # noqa: E402

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

WINDOW_GENERATORS = ("bikeshare", "vibration", "tank")
PARTS = ("train", "valid", "test")


def _window_lanes():
    return [lane for lane in registry.load_lanes().values()
            if lane.driver == "lanekit" and lane.raw["generator"] in WINDOW_GENERATORS]


@pytest.fixture(scope="module")
def lanes():
    found = _window_lanes()
    assert sorted(lane.id for lane in found) == [f"L{i}" for i in range(10, 19)], \
        "the window lanes of BUILD-PLAN §4.8 are L10-L18"
    return found


@pytest.fixture(scope="module")
def frames(lanes):
    """{(generator, seed, rows): the raw frame} for every window lane's declaration."""
    out = {}
    for lane in lanes:
        key = (lane.raw["generator"], int(lane.raw["seed"]), int(lane.raw["rows"]))
        if key not in out:
            out[key] = datasets.load(key[0]).make(key[1], key[2])
    return out


def _frame(frames, lane):
    return frames[(lane.raw["generator"], int(lane.raw["seed"]), int(lane.raw["rows"]))]


def _parts(frame, order_by: str) -> dict:
    """The raw rows of each part, cut as shape.py's time split cuts them (stable time order)."""
    ordered = frame.iloc[pd.to_datetime(frame[order_by]).argsort(kind="stable")]
    cuts = time_split_cuts(len(ordered))
    return {name: ordered.iloc[a:b] for name, a, b in zip(PARTS, cuts, cuts[1:])}


# --------------------------------------------------------------------------- the protocol
@pytest.mark.parametrize("name", WINDOW_GENERATORS)
def test_each_window_generator_keeps_the_protocol(name):
    module = datasets.load(name)
    assert name in datasets.names() and datasets.PLANNED[name] == "T06"
    assert name not in datasets.pending()
    assert module.VOCAB and all(isinstance(w, str) and w for w in module.VOCAB)
    frame = module.make(3, 200)
    assert list(frame.columns) == [c.name for c in module.SCHEMA]
    times = [c.name for c in module.SCHEMA if c.role == "time"]
    assert len(times) == 1 and pd.to_datetime(frame[times[0]]).notna().all()


@pytest.mark.parametrize("name", WINDOW_GENERATORS)
def test_generators_are_deterministic_in_seed_and_rows(name):
    make = datasets.load(name).make
    first = make(5, 301)
    assert first.equals(make(5, 301)), "the same (seed, rows) gives the same rows"
    assert not first.equals(make(6, 301)), "another seed gives other rows"
    for rows in (97, 301, 640):
        assert len(make(5, rows)) == rows


def _shaper():
    """dataset-shaper's shape.py (the source of the time split), imported under a private name."""
    path = HERE.parents[2] / "dataset-shaper" / "scripts" / "shape.py"
    spec = importlib.util.spec_from_file_location("_lanekit_windows_shape", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_time_split_cuts_are_the_shapers(lanes, frames, tmp_path):
    """time_split_cuts against shape.py's own op_split, run with the split step a window lane's
    recipe carries, on rows given out of time order: a change on either side fails here, before
    vibration and tank plant their per-part events in the wrong part."""
    lane = next(lane for lane in lanes if lane.id == "L14")
    raw = tmp_path / "tank.csv"
    _frame(frames, lane).to_csv(raw, index=False)
    recipe = standard_recipe(datasets.load("tank").SCHEMA, raw, tmp_path / "shaped", lane)
    split = next(step for step in recipe["steps"] if step["op"] == "split")
    shape = _shaper()
    for rows in (2400, 2000, 1800, 101, 97, 30):
        stamps = pd.date_range("2026-03-02 06:00", periods=rows, freq="min")
        shuffled = np.random.default_rng(rows).permutation(rows)
        frame = pd.DataFrame({"ts": stamps[shuffled].strftime("%Y-%m-%d %H:%M:%S"), "x": shuffled})
        record = shape.op_split(shape.Ctx(frame, recipe, {}, 0), split)
        cuts = time_split_cuts(rows)
        for name, a, b in zip(PARTS, cuts, cuts[1:]):
            got = sorted(int(frame["x"].iloc[i]) for i in record["indices"][name])
            assert got == list(range(a, b)), f"{rows} rows, {name}: the shaper cuts elsewhere than {cuts}"
    assert time_split_cuts(1800) == (0, 1260, 1530, 1800)


# --------------------------------------------------------------------------- the lanes
def test_every_window_lane_agrees_with_its_generator(lanes, frames):
    for lane in lanes:
        module = datasets.load(lane.raw["generator"])
        plan = ColumnPlan.of(module.SCHEMA, lane)          # a ValueError names a disagreement
        target = lane.raw["target"]
        assert plan.label == target["column"] and plan.order_by == lane.raw["window"]["order_by"]
        labels = [c.name for c in module.SCHEMA if c.role == "label"]
        assert set(plan.leak) == set(labels) - {plan.label}, f"{lane.id}: the other labels leave"
        values = _frame(frames, lane)[plan.label]
        classes = target.get("classes")
        if not classes:
            assert values.notna().all() and np.isfinite(values.astype(float)).all()
        elif target.get("label_format") == "delimited":
            names = {n for cell in values.fillna("") for n in str(cell).split(target["sep"]) if n}
            assert names == set(classes), f"{lane.id}: {sorted(names)} != {classes}"
        else:
            assert set(values.astype(str)) == set(classes), f"{lane.id}: {sorted(set(values.astype(str)))}"


def test_every_part_holds_a_full_window_per_series(lanes, frames):
    for lane in lanes:
        window = lane.raw["window"]
        need = window["length"] + (window.get("horizon") or 0)
        for name, part in _parts(_frame(frames, lane), window["order_by"]).items():
            series = window.get("series_by")
            sizes = part.groupby(series).size() if series else pd.Series([len(part)])
            assert int(sizes.min()) >= need, f"{lane.id} {name}: {int(sizes.min())} < {need}"
            if series:
                assert len(sizes) == 3, f"{lane.id} {name}: every station in every part"


def test_every_class_and_label_reaches_every_part(lanes, frames):
    for lane in lanes:
        target = lane.raw["target"]
        classes = target.get("classes")
        if not classes:
            continue
        parts = _parts(_frame(frames, lane), lane.raw["window"]["order_by"])
        for name, part in parts.items():
            values = part[target["column"]]
            if target.get("label_format") == "delimited":
                cells = values.fillna("").astype(str)
                for label in classes:
                    has = cells.str.split(target["sep"]).apply(lambda names, lb=label: lb in names)
                    assert has.any() and not has.all(), f"{lane.id} {name}: {label} present and absent"
            else:
                seen = sorted(set(values.astype(str)))
                assert seen == sorted(classes), f"{lane.id} {name}: {seen}"


def test_the_alarm_rises_in_the_first_and_last_quarter_of_the_held_parts(lanes, frames):
    lane = next(lane for lane in lanes if lane.id == "L14")
    parts = _parts(_frame(frames, lane), "ts")
    for name in ("valid", "test"):
        alarm = parts[name]["alarm"].to_numpy()
        quarter = len(alarm) // 4
        for where, span in (("first", alarm[:quarter]), ("last", alarm[-quarter:])):
            assert 0 < span.mean() < 1, f"{name}: the {where} quarter holds both alarm and calm minutes"


# --------------------------------------------------------------------------- the signals
def _auc(score, positive) -> float:
    """The rank AUC of `score` for the positive rows (ties averaged)."""
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def test_the_tank_labels_read_the_level_and_its_trend():
    frame = datasets.load("tank").make(1, 1800)
    alarm = frame["alarm"].to_numpy() == 1
    assert 0.1 < alarm.mean() < 0.45
    assert _auc(-frame["level_pct"].to_numpy(), alarm) > 0.85
    early = alarm & (frame["level_pct"].to_numpy() >= 12)
    assert early.sum() >= 10, "the alarm fires before the level crosses: a falling trend is what tells"
    tte = frame["minutes_to_empty"]
    assert tte.between(0, 120).all() and 0.15 < (tte < 120).mean() < 0.7
    assert sorted(frame["grade"].unique()) == [0, 1, 2, 3]
    level = frame["level_pct"].rolling(15, min_periods=1).mean()
    assert np.corrcoef(level, frame["grade"])[0, 1] > 0.7


def test_the_vibration_faults_show_in_a_window_of_frames():
    module = datasets.load("vibration")
    frame = module.make(1, 2000)
    window = frame[["rms", "kurtosis", "crest", "peak_hz", "temp_c"]].rolling(32, min_periods=1)
    table = pd.concat([window.mean().add_suffix("_mean"), window.max().add_suffix("_max")], axis=1)
    feats = ((table - table.mean()) / table.std()).to_numpy()
    state = frame["state"].to_numpy()
    cut = time_split_cuts(len(frame))[1]
    names = list(module.STATES)
    centroids = np.stack([feats[:cut][state[:cut] == s].mean(0) for s in names])
    guess = np.array(names)[np.argmin(((feats[cut:, None, :] - centroids[None]) ** 2).sum(2), axis=1)]
    assert (guess == state[cut:]).mean() > 0.6, "a nearest centroid on window statistics reads the state"
    bearing = frame["faults"].fillna("").str.contains("bearing").to_numpy()[cut:]
    one_frame = _auc(frame["kurtosis"].to_numpy()[cut:], bearing)
    assert _auc(table["kurtosis_max"].to_numpy()[cut:], bearing) > one_frame + 0.05, \
        "a bearing's impulses are clearer over a window than in one frame"


def test_the_bikeshare_demand_follows_the_station_and_the_hour():
    frame = datasets.load("bikeshare").make(1, 2400)
    working = (frame["weekday"] < 5) & (frame["holiday"] == 0)
    profile = frame.groupby([frame["station"], frame["hour"], working])["demand"].transform("mean")
    residual = frame["demand"] - profile
    assert 1 - residual.var() / frame["demand"].var() > 0.6, "the daily profile explains most of the demand"
    lagged = residual.groupby(frame["station"]).shift(1)
    assert residual.corr(lagged) > 0.3, "the residual is autocorrelated: the target's past helps"
    assert frame.groupby("weather")["demand"].mean()["rainy"] < frame.groupby("weather")["demand"].mean()["sunny"]


# --------------------------------------------------------------------------- the recipes
def test_each_lane_recipe_takes_its_label_splits_in_time_and_lags_as_asked(lanes, frames, tmp_path):
    for lane in lanes:
        module = datasets.load(lane.raw["generator"])
        raw = tmp_path / f"{lane.id}.csv"
        _frame(frames, lane).to_csv(raw, index=False)
        recipe = standard_recipe(module.SCHEMA, raw, tmp_path / "shaped", lane)
        steps = {s["op"]: s for s in recipe["steps"]}
        plan = ColumnPlan.of(module.SCHEMA, lane)
        assert recipe["target"] == lane.raw["target"]["column"]
        assert steps["select_partition"]["params"]["label"] == plan.label
        assert set(steps.get("drop_leakage", {}).get("columns", [])) == set(plan.leak)
        assert steps["split"]["params"] == {"kind": "time", "time_column": "ts", "test": 0.15, "valid": 0.15}
        lag = (lane.raw.get("recipe_options") or {}).get("lag")
        if lag:
            assert steps["lag"]["columns"] == [lag["column"]]
            assert steps["lag"]["params"]["lags"] == list(range(1, lag["depth"] + 1))
            assert steps["lag"]["params"]["group_by"] == lane.raw["window"].get("series_by")
        else:
            assert "lag" not in steps
        if "weather" in plan.nominal:
            want = "ordinal" if lane.id == "L10" else "one-hot"
            assert [s["params"]["strategy"] for s in recipe["steps"] if s["op"] == "encode"] == [want]


# --------------------------------------------------------------------------- the vocabulary
D6_WORDS = frozenset({"demand", "station", "weather", "machine", "rms", "alarm", "grade"})

# The lint-allow.json entries (vega/lint-allow@1) that let D6 §4's withheld words (VOCAB_WITHHELD)
# join VOCAB: each covers one sense a shared template gives one of them, line by line through
# `match`, so the word itself stays a leak everywhere else. lint-allow.json is T04's: the entries
# go there (a need of T06, runs/T06/needs/T04.lint-allow.window-vocab.patch), and then the three
# generators fold VOCAB_WITHHELD into VOCAB.
SENSE_ALLOW = (
    {"file": "vega-forge/scripts/backends/pytorch/templates/serving/*", "token": "machine",
     "match": r"the build machine's|the machine copy|whatever the machine|on a machine with few cores",
     "reason": "a computer (the build machine, the machine-readable copy, the serving host), not the "
               "vibration lane's machine"},
    {"file": "vega-pack/scripts/*", "token": "machine",
     "match": r"on a machine with no\b|this machine's bytecode",
     "reason": "the computer a model card is read on or a bundle is built on, not the vibration lane's "
               "machine"},
    {"file": "vega-show/assets/page/*", "token": "machine", "match": r"\bmachine code\b",
     "reason": "a machine code: a problem's strings-table code (A23), not the vibration lane's machine"},
    {"file": "vega-show/assets/page/js/net/renderers/box.js", "token": "rms", "match": r"Math\.(sqrt|tanh)\(",
     "reason": "the root mean square of a tensor the box renderer glows by, not the vibration lane's rms "
               "column"},
    {"file": "vega-forge/scripts/backends/pytorch/templates/serving/*", "token": "alarm", "match": r'"alarm"',
     "reason": "the `alarm` answer kind of the output record (field_kinds: an anomaly flag), not the tank "
               "lane's label"},
    {"file": "vega-pack/scripts/*", "token": "alarm",
     "match": r'^alarm$|"alarm"|\balarm = output\.get\(|the alarm a score stratum|`alert_rate` the alarm',
     "reason": "the `alarm` field kind a monitor check or a demo preset reads (field_kinds), not the tank "
               "lane's label"},
    {"file": "vega-show/assets/page/*", "token": "alarm", "match": r'"alarm"',
     "reason": "the `alarm` expected kind of a preset and of the page's prediction fixtures, not the tank "
               "lane's label"},
    {"file": "vega-show/assets/page/tests/fixtures/predictions/*", "token": "grade", "match": r'"grade"',
     "reason": "the ordinal example column of the page's prediction fixtures (a test fixture no lane's "
               "page renders), not the tank lane's label"},
    {"file": "vega-show/assets/page/tests/fixtures/predictions/*", "token": "station",
     "match": r'"series_by": "station"',
     "reason": "the window example's series key in the page's prediction fixtures (a test fixture no "
               "lane's page renders), not the bikeshare lane's column"},
    {"file": "vega-show/assets/page/js/views/network.js", "token": "demand", "match": r"\bon demand\b",
     "reason": "'opens on demand' in a code comment, not the bikeshare lane's label"},
    {"file": "vega-show/assets/page/tests/roles.test.js", "token": "weather", "match": r"domain:weather",
     "reason": "a roles test's made-up domain basis, not the bikeshare lane's column"},
)


def _lint():
    import lint_templates
    return lint_templates


def _uncovered(words, extra_allow=(), whole_pack=False) -> list:
    """The hits of `words` in the shared templates (the literal gate's default scan: a pack source's
    string literals, every other template's text) that lint-allow.json plus extra_allow does not
    cover; with whole_pack, each pack source's whole text too, the way the monitor suite passes its
    pack files to the gate."""
    lt = _lint()
    tokens = lt.vocab_tokens({"window": frozenset(words)}, lt.read_json(lt.LITERALS, "the literals"))
    units = lt.default_units(lt.REPO)
    units += [(path, "text") for path, mode in units if mode == "pack"] if whole_pack else []
    hits = lt.scan(units, tokens, lt.REPO)
    rules = lt.load_allow(lt.ALLOW) + [lt.Allow(e["file"], e["token"], e["reason"], re.compile(e["match"]))
                                       for e in extra_allow]
    return [h.line_out(True) for h in hits if not any(rule.covers(h) for rule in rules)]


def test_the_window_vocabulary_is_no_hit_in_the_shared_templates():
    """What VOCAB holds today adds no hit to the literal gate (lint_templates.py)."""
    words = {w for g in WINDOW_GENERATORS for w in datasets.load(g).VOCAB}
    assert not _uncovered(words), _uncovered(words)[:5]


def test_the_withheld_d6_words_are_named_and_every_template_use_is_another_sense():
    """No silent narrowing: D6 §4's words VOCAB leaves out are each generator's VOCAB_WITHHELD, and
    SENSE_ALLOW covers every line of a shared template that uses one (pack sources read whole too),
    so they can join VOCAB the day lint-allow.json holds those entries."""
    kept = {w for g in WINDOW_GENERATORS for w in datasets.load(g).VOCAB}
    withheld = {w for g in WINDOW_GENERATORS for w in datasets.load(g).VOCAB_WITHHELD}
    assert D6_WORDS <= kept | withheld, sorted(D6_WORDS - kept - withheld)
    assert not kept & withheld
    left = _uncovered(D6_WORDS, SENSE_ALLOW, whole_pack=True)
    assert not left, left[:5]
    lt = _lint()
    for entry in SENSE_ALLOW:
        assert entry["token"] in D6_WORDS and entry["reason"].strip()
        rule = lt.Allow(entry["file"], entry["token"], entry["reason"], re.compile(entry["match"]))
        assert not rule.covers(lt.Hit("vega-show/assets/page/js/x.js", 1, entry["token"], "vocab:window",
                                      f"the {entry['token']} of this lane")), f"{entry}: covers the word itself"
