"""
The row generators' tests (BUILD-PLAN T05: "every generator deterministic in (seed, rows)"), and
what the lanes L02–L09 need of them before a fixture is built:

- the lanekit.datasets protocol; `make` deterministic in (seed, rows) (the same frame and CSV
  bytes twice, another frame on another seed), exactly its SCHEMA, no missing value;
- each lane agrees with its generator: the column plan holds, every declared class occurs, the
  embedded columns are nominals;
- the planted signals (L04: the trees beat a linear model; L07: three dimensions; the anomaly
  lanes: about 3 % faults a reconstruction finds);
- the unlabelled lanes (L07, L08): the geometry states the label candidate and abstains, which
  dataset-forge's validator accepts (A12), and the goal stance drops it;
- L05's coordinate pair (`expect.spatial`); the recipe extras (D6 §4: tickets' and housing's
  skewed column, a finding of their own measured on the rows, and the transform citing it);
- the literal lint: no VOCAB word is in a shared template, and every D6 §4 word is in VOCAB or
  in VOCAB_WITHHELD for a reason that holds.

    python3 -m pytest -q vega/scripts/tests/test_lanekit_rows.py
"""
from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import datasets, registry  # noqa: E402
from lanekit.columns import ColumnPlan, embedded_columns  # noqa: E402
from lanekit.goal import GOAL_SOURCE, goal_recipe  # noqa: E402
from lanekit.layers import RawTable, analysis_payload, candidate_task, geometry_payload, spatial_context  # noqa: E402
from lanekit.planted import planted_findings  # noqa: E402
from lanekit.recipes import standard_recipe  # noqa: E402

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = HERE.parents[2]
ROWS = ("landcover", "tickets", "wine", "housing", "meters", "turbine")
LANES = registry.load_lanes()
ROW_LANES = {lid: lane for lid, lane in LANES.items() if lane.raw.get("generator") in ROWS}
KINDS = {"float": "f", "int": "iu", "str": "OUT", "bool": "b", "datetime": "OUT"}


def _frame(lane):
    raw = lane.raw
    return datasets.load(raw["generator"]).make(int(raw["seed"]), int(raw["rows"]))


# --------------------------------------------------------------------------- the protocol
def test_every_row_lane_names_one_of_these_generators():
    assert sorted(ROW_LANES) == ["L02", "L03", "L04", "L05", "L06", "L07", "L08", "L09"]
    assert set(ROWS) <= set(datasets.names())
    assert all(datasets.PLANNED[name] == "T05" for name in ROWS)
    assert not set(ROWS) & set(datasets.pending())


@pytest.mark.parametrize("name", ROWS)
def test_make_is_deterministic_in_seed_and_rows(name, tmp_path):
    gen = datasets.load(name)
    first, again, other = gen.make(5, 240), gen.make(5, 240), gen.make(6, 240)
    assert first.equals(again)
    first.to_csv(tmp_path / "a.csv", index=False)
    again.to_csv(tmp_path / "b.csv", index=False)
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert not first.equals(other), "another seed draws another table"
    assert len(gen.make(5, 97)) == 97 and gen.make(5, 97).equals(gen.make(5, 97))


@pytest.mark.parametrize("name", ROWS)
def test_the_table_is_the_schema(name):
    gen = datasets.load(name)
    frame = gen.make(3, 300)
    assert list(frame.columns) == [c.name for c in gen.SCHEMA]
    assert len(frame) == 300
    assert not frame.isna().any().any(), f"{name}: a missing value in {frame.columns[frame.isna().any()].tolist()}"
    for col in gen.SCHEMA:
        assert frame[col.name].dtype.kind in KINDS[col.dtype], (name, col.name, frame[col.name].dtype)
        if col.dtype == "datetime":
            assert pd.to_datetime(frame[col.name], errors="coerce").notna().all(), (name, col.name)
    assert gen.VOCAB and all(isinstance(w, str) and w == w.strip() for w in gen.VOCAB)


# --------------------------------------------------------------------------- lane and generator agree
@pytest.mark.parametrize("lane_id", sorted(ROW_LANES))
def test_the_lane_and_its_generator_agree(lane_id):
    lane = ROW_LANES[lane_id]
    raw, gen = lane.raw, datasets.load(lane.raw["generator"])
    plan = ColumnPlan.of(gen.SCHEMA, lane)
    frame = _frame(lane)
    nominal = {c.name for c in gen.SCHEMA if c.role == "nominal"}
    assert set(embedded_columns(raw.get("blocks") or [])) <= nominal
    target = raw["target"]
    if target.get("column") is None:
        assert plan.label is None and plan.leak, "an unlabelled lane's generator still records a label candidate"
        return
    values = frame[target["column"]]
    classes = [str(c) for c in target.get("classes") or []]
    if target.get("label_format") == "delimited":
        seen = {t for cell in values for t in str(cell).split(target["sep"])}
        assert seen == set(classes) and all(str(v) for v in values), "every tag occurs; every row carries one"
    elif classes:
        assert set(values.astype(str)) == set(classes), (lane_id, sorted(set(values.astype(str))))
        if target.get("positive") is not None:
            assert classes[1] == str(target["positive"]), "the positive class is listed second (A16)"
    else:
        assert values.dtype.kind == "f"


def test_l04_is_built_to_be_beaten_by_its_tree_baseline():
    """The lane's declared verdict (expect.verdict beaten_by_baseline): the grade is mostly not
    linear in the measurements, so gradient-boosted trees read far more of it than a linear
    model does (mean absolute grade error, held-out rows)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    lane = ROW_LANES["L04"]
    assert lane.expect.get("verdict") == "beaten_by_baseline"
    frame = _frame(lane)
    x = pd.get_dummies(frame.drop(columns="quality")).astype(float).to_numpy()
    y = frame["quality"].to_numpy()
    cut = int(0.7 * len(y))                          # the rows are drawn i.i.d.: a plain cut holds out
    with threadpool_limits(2):                       # the unit tier runs suites side by side
        linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x[:cut], y[:cut])
        trees = HistGradientBoostingClassifier(max_depth=4, random_state=0).fit(x[:cut], y[:cut])
        mae_linear = np.abs(linear.predict(x[cut:]) - y[cut:]).mean()
        mae_trees = np.abs(trees.predict(x[cut:]) - y[cut:]).mean()
    assert mae_trees < 0.7 * mae_linear, (mae_trees, mae_linear)


def test_l04_types_its_grade_ordinal_and_the_recipe_retypes_it_in_the_lane_order(tmp_path):
    """The grade is spelled in digits but is neither a quantity nor a class index: the geometry
    types it ordinal, and the recipe retypes it to an ordered category in the lane's class order
    (so the shaped label holds the names task.classes lists)."""
    lane = ROW_LANES["L04"]
    wine = datasets.load("wine")
    table = _table(lane, tmp_path)
    typing = {t["column"]: t for t in geometry_payload(wine.SCHEMA, lane, table)["typing"]}
    assert typing["quality"]["final_type"] == "ordinal" and typing["alcohol"]["final_type"] == "numeric"
    context = {t["column"]: t["type"] for t in analysis_payload(wine.SCHEMA, lane, table)["context"]["typing"]}
    assert context["quality"] == "ordinal"
    steps = wine.RECIPE(table.path, tmp_path / "shaped", lane)["steps"]
    assert [s["op"] for s in steps][:3] == ["retype", "retype", "select_partition"]
    grade = steps[1]
    assert (grade["columns"], grade["params"], grade["source"]) == (
        ["quality"], {"to": "ordinal", "order": ["3", "4", "5", "6", "7", "8"]}, "geometry:typing/quality")
    assert [s["id"] for s in steps] == [f"S{i:02d}" for i in range(1, len(steps) + 1)]
    other = copy.deepcopy(dict(lane.raw))
    other["target"] = {**other["target"], "column": None}
    other["task"], other["goal"] = "unsupervised", "reconstruct"
    assert not any(s["op"] == "retype" and s["columns"] == ["quality"]
                   for s in wine.RECIPE(table.path, tmp_path / "shaped", other)["steps"])


def test_every_lane_optimizer_states_its_update_rule():
    from lanekit import spec_choices

    for lane in LANES.values():
        if lane.driver != "lanekit":
            continue
        node = spec_choices.optimizer(lane.raw)
        assert node["update_rule"].startswith(("m ←", "v ←", "b ←", "θ ←", "g ←")), (lane.id, node)
    assert spec_choices.update_rule("sgd", {"lr": 0.01, "momentum": 0.9}) == "b ← μb + g; θ ← θ − η·b"
    assert spec_choices.update_rule("adam", {"weight_decay": 0.01}).startswith("g ← g + λθ; m ←")
    assert "λθ)" in spec_choices.update_rule("adamw", {"weight_decay": 0.0001})


def test_l07_readings_live_near_three_dimensions():
    frame = _frame(ROW_LANES["L07"])
    x = frame[[c for c in frame.columns if c.startswith("kwh_")]].to_numpy()
    s = np.linalg.svd((x - x.mean(0)) / x.std(0), compute_uv=False)
    share = np.cumsum(s ** 2) / (s ** 2).sum()
    assert share[2] > 0.9 and share[1] < 0.9, share[:4]


@pytest.mark.parametrize("lane_id", ["L06", "L08", "L09"])
def test_the_anomaly_lanes_hold_rare_faults_a_reconstruction_finds(lane_id):
    from sklearn.metrics import roc_auc_score

    frame = _frame(ROW_LANES[lane_id])
    faulty = (frame["fault"] == "fault").to_numpy()
    assert 0.02 <= faulty.mean() <= 0.04
    gen = datasets.load("turbine")
    x = frame[[c.name for c in gen.SCHEMA if c.role == "numeric"]].to_numpy()
    z = (x - x.mean(0)) / x.std(0)
    basis = np.linalg.svd(z, full_matrices=False)[2][:4]
    error = ((z - z @ basis.T @ basis) ** 2).sum(1)
    assert roc_auc_score(faulty, error) > 0.85


# --------------------------------------------------------------------------- the unlabelled lanes (A12)
def _table(lane, tmp_path) -> RawTable:
    path = tmp_path / f"{lane.raw['generator']}.csv"
    _frame(lane).to_csv(path, index=False)
    return RawTable(path, pd.read_csv(path))


@pytest.mark.parametrize("lane_id,label,task", [("L07", "tariff", "multiclass classification"),
                                                ("L08", "fault", "binary classification")])
def test_an_unlabelled_lane_states_its_candidate_and_abstains(lane_id, label, task, tmp_path):
    lane = ROW_LANES[lane_id]
    gen = datasets.load(lane.raw["generator"])
    doc = geometry_payload(gen.SCHEMA, lane, _table(lane, tmp_path))
    partitions = doc["partitions"]
    assert (partitions["chosen"], partitions["provenance"]) == (None, "abstained")
    assert [(c["label"], c["task"]) for c in partitions["candidates"]] == [(label, task)]
    assert label not in partitions["candidates"][0]["features"]
    data = tmp_path / "geometry.json"
    data.write_text(json.dumps(doc), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(REPO / "dataset-forge/scripts/apply_geometry_layer.py"),
                           "--data", str(data), "--check-only"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "chosen None (abstained)" in proc.stdout
    context = analysis_payload(gen.SCHEMA, lane, _table(lane, tmp_path))["context"]["partition"]
    assert (context["label"], context["provenance"]) == (None, "abstained")


def test_a_labelled_lane_still_chooses_its_one_label(tmp_path):
    lane = ROW_LANES["L09"]
    doc = geometry_payload(datasets.load("turbine").SCHEMA, lane, _table(lane, tmp_path))
    assert (doc["partitions"]["chosen"], doc["partitions"]["provenance"]) == ("fault", "single-candidate")


def test_candidate_task_reads_the_values():
    assert candidate_task(pd.Series(["ok", "fault", "ok"])) == "binary classification"
    assert candidate_task(pd.Series(["a", "b", "c"])) == "multiclass classification"
    assert candidate_task(pd.Series(["a|b", "b"])) == "multilabel classification"
    assert candidate_task(pd.Series(np.linspace(0, 1, 50))) == "regression"
    assert candidate_task(pd.Series([3, 4, 5, 6])) == "multiclass classification"


@pytest.mark.parametrize("lane_id", ["L07", "L08"])
def test_the_goal_stance_drops_the_candidate(lane_id, tmp_path):
    lane = ROW_LANES[lane_id]
    gen = datasets.load(lane.raw["generator"])
    path = tmp_path / "raw.csv"
    _frame(lane).to_csv(path, index=False)
    recipe = gen.RECIPE(path, tmp_path / "shaped", lane)
    labels = [c.name for c in gen.SCHEMA if c.role == "label"]
    drops = [s for s in recipe["steps"] if s["op"] == "drop_leakage"]
    assert [(s["columns"], s["source"]) for s in drops] == [(labels, GOAL_SOURCE)]
    assert [s["id"] for s in recipe["steps"]] == [f"S{i:02d}" for i in range(1, len(recipe["steps"]) + 1)]
    assert not any(s["op"] == "select_partition" for s in recipe["steps"])


def test_the_goal_recipe_keeps_other_leakage_on_its_finding_and_leaves_a_labelled_lane_alone(tmp_path):
    housing = datasets.load("housing")
    lane = copy.deepcopy(dict(ROW_LANES["L05"].raw))
    path = tmp_path / "raw.csv"
    housing.make(1, 200).to_csv(path, index=False)
    assert goal_recipe(housing.SCHEMA, path, tmp_path, lane) == standard_recipe(housing.SCHEMA, path, tmp_path, lane)
    lane["target"] = {**lane["target"], "column": None}
    lane["goal"], lane["task"] = "reconstruct", "unsupervised"
    drops = [(s["columns"], s["source"]) for s in goal_recipe(housing.SCHEMA, path, tmp_path, lane)["steps"]
             if s["op"] == "drop_leakage"]
    assert drops == [(["notary_fee_keur"], "analysis:F6/T4"), (["price_keur"], GOAL_SOURCE)]


@pytest.mark.parametrize("lane_id", ["L06", "L07", "L09"])
def test_the_drift_finding_names_the_columns_the_evaluation_slices_by(lane_id, tmp_path):
    """vega-eval slices the test part by `modules.drift.columns` (and by the nominal inputs):
    a lane without nominal inputs still has slices to read, the drift finding's column."""
    lane = ROW_LANES[lane_id]
    gen = datasets.load(lane.raw["generator"])
    doc = analysis_payload(gen.SCHEMA, lane, _table(lane, tmp_path))
    f4 = next(f for f in doc["findings"] if f["id"] == "F4")
    assert doc["modules"]["drift"]["columns"] == f4["columns"] and doc["modules"]["drift"]["ran"] is True
    assert f4["columns"][0] in {c.name for c in gen.SCHEMA if c.role == "numeric"}


# --------------------------------------------------------------------------- L02's real sub-chain
def test_the_real_subchain_geometry_carries_the_scripts_twonn_and_cites_its_run(tmp_path, monkeypatch):
    """On L02's real sub-chain dataset-forge's apply script splices geometry.py's numbers in and
    refuses others: the payload states the script's TwoNN as basis.intrinsic_dim, and the
    validating writer hands it the run (`from_run`) so the numbers are checked, not trusted."""
    from lanekit import layers
    from lanekit.chain import Workspace

    lane = ROW_LANES["L02"]
    gen = datasets.load("landcover")
    table = _table(lane, tmp_path)
    evidence = {"space": {"intrinsic_dim_twonn": 7.25, "dims_95": 6}, "columns": {}}
    assert geometry_payload(gen.SCHEMA, lane, table, evidence)["basis"]["intrinsic_dim"] == 7.25
    composed = geometry_payload(gen.SCHEMA, lane, table)
    assert composed["basis"]["intrinsic_dim"] == composed["space"]["dims_95"]
    ws = Workspace(REPO, tmp_path / "work", "landcover").prepare()
    (ws.scratch / "geometry.json").write_text("{}", encoding="utf-8")
    seen = {}

    def fake_tool(argv, what, cwd=None, timeout=900):
        data = Path(argv[argv.index("--data") + 1])
        seen.update(json.loads(data.read_text(encoding="utf-8")))
        Path(argv[argv.index("--out") + 1]).write_text("<html></html>", encoding="utf-8")
        return ""

    monkeypatch.setattr(layers, "run_tool", fake_tool)
    layers.LayerWriter(ws, True).geometry("<html></html>", composed)
    assert seen["from_run"] == str(ws.scratch / "geometry.json")


# --------------------------------------------------------------------------- the spatial lane
def test_l05_carries_its_coordinate_pair_as_the_spatial_context(tmp_path):
    lane = ROW_LANES["L05"]
    gen = datasets.load("housing")
    assert lane.modality == "spatial-row"
    doc = analysis_payload(gen.SCHEMA, lane, _table(lane, tmp_path))
    assert doc["context"]["spatial"] == {"columns": [{"column": "lat", "pair": "lon", "kind": "latlon"}],
                                         "crs": "EPSG:4326"}
    assert doc["modules"]["spatial"]["ran"] is True
    for other in ("L02", "L09"):
        assert spatial_context(datasets.load(ROW_LANES[other].raw["generator"]).SCHEMA, ROW_LANES[other]) is None
    declared = copy.deepcopy(dict(ROW_LANES["L09"].raw))
    declared["expect"] = {**declared["expect"], "spatial": True}
    with pytest.raises(ValueError, match="no latitude/longitude pair"):
        spatial_context(datasets.load("turbine").SCHEMA, declared)


# --------------------------------------------------------------------------- the recipe extras
EXTRAS = {"L03": ("tickets", "length_chars", "log1p"), "L05": ("housing", "area_m2", "log")}


@pytest.mark.parametrize("lane_id", sorted(EXTRAS))
def test_the_skew_is_admitted_measured_and_answered_by_a_transform_in_phase_4(lane_id, tmp_path):
    name, column, kind = EXTRAS[lane_id]
    lane, gen = ROW_LANES[lane_id], datasets.load(name)
    table = _table(lane, tmp_path)
    doc = analysis_payload(gen.SCHEMA, lane, table)
    f7 = next(f for f in doc["findings"] if f["id"] == "F7")
    assert (f7["columns"], f7["module"]) == ([column], "distributions") and doc["modules"]["distributions"]["ran"]
    candidates = [(c["id"], c["op"], c["columns"]) for c in f7["transformation_candidates"]]
    assert candidates == [("T7", "transform", [column])]
    assert "T7" in doc["handoff"]["shaper_candidates"]
    skew, logged = f7["evidence"]["skew"][column], f7["evidence"]["skew_of_log"][column]
    assert skew > 1.0 and abs(logged) < 0.5 * skew, (skew, logged)
    steps = gen.RECIPE(table.path, tmp_path / "shaped", lane)["steps"]
    ops = [s["op"] for s in steps]
    at, step = ops.index("transform"), steps[ops.index("transform")]
    assert ops.count("transform") == 1
    assert (step["columns"], step["params"], step["source"]) == ([column], {"kind": kind}, "analysis:F7/T7")
    assert ops.index("split") < at < min(ops.index(op) for op in ("encode", "scale")), ops
    assert column in next(s for s in steps if s["op"] == "scale")["columns"], "the log is standardised after"
    assert [s["id"] for s in steps] == [f"S{i:02d}" for i in range(1, len(steps) + 1)]


def test_the_generator_findings_are_read_for_its_own_schema_only(monkeypatch, tmp_path):
    for lane_id in ("L02", "L04", "L09"):                    # no extras: the standard findings and recipe
        lane = ROW_LANES[lane_id]
        gen = datasets.load(lane.raw["generator"])
        assert set(planted_findings(gen.SCHEMA, lane)) <= {"F1", "F2", "F3", "F4", "F5", "F6"}
        assert "transform" not in [s["op"] for s in gen.RECIPE(_table(lane, tmp_path).path, tmp_path, lane)["steps"]]
    housing, lane = datasets.load("housing"), ROW_LANES["L05"]
    other = tuple(dataclasses.replace(c, comment="another table's") if c.name == "rooms" else c
                  for c in housing.SCHEMA)
    assert "F7" in planted_findings(housing.SCHEMA, lane) and "F7" not in planted_findings(other, lane)
    monkeypatch.setattr(housing, "FINDINGS", {"F4": housing.FINDINGS["F7"]})
    with pytest.raises(ValueError, match="reuses the standard id F4"):
        planted_findings(housing.SCHEMA, lane)


# --------------------------------------------------------------------------- the literal lint
def test_no_vocab_word_of_these_generators_is_in_a_shared_template():
    """The literal lint scans every shared template for every lane generator's VOCAB: a word
    that is already there (a template's own English) would turn the served-template checks red,
    so no VOCAB word of these generators may appear in one."""
    import lint_templates

    doc = lint_templates.read_json(lint_templates.LITERALS, "the literals")
    vocab = {name: frozenset(datasets.load(name).VOCAB) for name in ROWS}
    tokens = lint_templates.vocab_tokens(vocab, doc)
    assert {t.kind for t in tokens} == {f"vocab:{name}" for name in ROWS}
    hits = lint_templates.scan(lint_templates.default_units(REPO), tokens, REPO)
    assert hits == [], [f"{h.file}:{h.line}: {h.token}" for h in hits[:10]]


# withheld though no shared template uses them: a cross-lane leak (D6 §10.6), as the generator says
NO_TEMPLATE_HIT = {"water": "L22's (molecules: water-soluble)", "quality": "data-lens's quality module"}


def _d6_words(name: str) -> set[str]:
    """D6 §4's words of a generator: its column names, the classes of its labels and nominals
    (read off its rows), and the entity and row noun of every lane that names it."""
    gen = datasets.load(name)
    frame = gen.make(1, 600)
    words = {c.name for c in gen.SCHEMA}
    for col in gen.SCHEMA:
        if col.role in ("nominal", "label") and frame[col.name].dtype.kind not in "biufc":
            words |= {tag for value in frame[col.name].astype(str) for tag in value.split("|")}
    for lane in ROW_LANES.values():
        if lane.raw["generator"] == name:
            words |= {str(c) for c in lane.raw["target"].get("classes") or []} | {lane.raw["entity"]}
            row = (lane.raw.get("lexicon") or {}).get("row") or {}
            words |= {w for noun in (row.get("one"), row.get("many")) if noun for w in noun.split()}
    return words


def test_every_d6_word_is_in_vocab_or_withheld_for_a_reason_that_holds():
    """Every D6 §4 word the literal lint would read (lint_templates.vocab_tokens: not short, not
    generic, not a hard or soft token) is in VOCAB or in VOCAB_WITHHELD, never both; a withheld
    word is one a shared template uses in a sense of its own (so it hits one), or one named in
    NO_TEMPLATE_HIT — else it belongs in VOCAB."""
    import lint_templates

    doc = lint_templates.read_json(lint_templates.LITERALS, "the literals")
    withheld_all = set()
    for name in ROWS:
        gen = datasets.load(name)
        vocab, withheld = {w.lower() for w in gen.VOCAB}, {w.lower() for w in gen.VOCAB_WITHHELD}
        assert not vocab & withheld, (name, sorted(vocab & withheld))
        read = {t.token.lower() for t in lint_templates.vocab_tokens({name: frozenset(_d6_words(name))}, doc)}
        assert read <= vocab | withheld, (name, sorted(read - vocab - withheld))
        withheld_all |= set(gen.VOCAB_WITHHELD)
    tokens = lint_templates.vocab_tokens({"withheld": frozenset(withheld_all)}, doc)
    hit = {h.token for h in lint_templates.scan(lint_templates.default_units(REPO), tokens, REPO)}
    idle = sorted(withheld_all - hit - set(NO_TEMPLATE_HIT))
    assert not idle, f"withheld, yet no shared template uses them: {idle} belong in VOCAB"
