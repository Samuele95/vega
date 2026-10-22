"""
test_demo_pack.py — the demonstration sample and its presets (build plan S08; R-D8; D2 §8.4;
bundle-contract §7.4).

The closing tests of S08: strata per reading from the train part only; presets carry `kind` and
`facts` and no English; window items carry their history; graph items carry rows only;
`examples/demo.json` and `examples/presets.json` validate against `examples-2.schema.json`.

The bundles are real: predictor_fixtures' rendered v2.3 packages and the shaper's own outputs
(every row reading, a forecast and two sequence tasks over windows, two node-level graphs and a
graph-level set), answered by the rendered Predictor. The train rows keep the fixture's inputs;
their truth column is replaced where a reading needs classes or labels (the Predictor never
reads it). A stub Predictor pins the cases a seeded network cannot be made to show (a class the
model never gets right, an empty `flagged` stratum), and a stub Recipe the unseen-code search.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
SKILLS = PACK.parent.parent
FORGE_TESTS = SKILLS / "vega-forge" / "scripts" / "tests"
for _p in (str(PACK), str(FORGE_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import demo_pack  # noqa: E402
import demo_presets  # noqa: E402
import demo_strata  # noqa: E402
import predictor_fixtures as pf  # noqa: E402
from demo_samples import DemoInputs, RowCodec  # noqa: E402
from pack_context import Campaign, PackContext, PackExit  # noqa: E402
from vega_pack_core import CATALOG, BundlePaths  # noqa: E402

jsonschema = pytest.importorskip("jsonschema")

SCHEMA = json.loads((PACK.parent / CATALOG["examples"]["json_schema"]).read_text("utf-8"))
PRESET_KEYS = {"id", "kind", "group", "stratum", "expected", "facts", "row", "rows", "history"}
PER = 4
RNG_LABELS = 11


def _validator(name):
    return jsonschema.Draft202012Validator(
        {"$schema": SCHEMA["$schema"], "$defs": SCHEMA["$defs"], "$ref": f"#/$defs/{name}"})


def _errors(name, doc):
    return [f"{list(e.absolute_path)}: {e.message}" for e in _validator(name).iter_errors(doc)]


# --------------------------------------------------------------------------- the cases

def _classes(names):
    return lambda n, rng: rng.choice(names, n)


def _labels(n, rng):
    names = ("red", "green", "blue")
    return ["|".join(sorted(set(rng.choice(names, rng.integers(0, 3))))) for _ in range(n)]


def _numbers(n, rng):
    return rng.normal(size=n)


# reading -> the train truth the reading reads (None: the fixture's column as it is)
ROW_TRUTH = {"binary": _classes([0, 1]), "multiclass": _classes(["a", "b", "c"]),
             "multilabel": _labels, "ordinal": _classes(["low", "mid", "high", "top"]),
             "regression": _numbers, "quantile": _numbers, "reconstruction": None,
             "anomaly-reconstruction": _classes([0, 1]), "anomaly-svdd": None}
WINDOWS = ("forecast", "sequence-regression", "sequence-binary")
GRAPHS = ("graph-keypair", "graph-knn", "graph-level")


def _case(name):
    if name in ROW_TRUTH:
        return pf.row_case(name)
    if name in WINDOWS:
        return pf.window_case(name)
    return pf.graph_case(name)


def _train(case, name):
    """The case's train rows, its truth column rewritten where the reading needs it."""
    frame = case.shaped.rows("train").reset_index(drop=True)
    column = (case.manifest["output"].get("truth") or {}).get("column")
    rng = np.random.default_rng(RNG_LABELS)
    if ROW_TRUTH.get(name) is not None:
        frame[column] = ROW_TRUTH[name](len(frame), rng)
    elif name == "sequence-binary":
        frame[column] = (frame[column] > 0).astype(int)
    return frame


def _inputs(name, per=PER, output=None):
    case = _case(name)
    return DemoInputs(train=_train(case, name), output=output or case.manifest["output"],
                      sample=case.manifest["sample"], schema=case.schema, per_stratum=per,
                      recipe=case.shaped.module)


@pytest.fixture(scope="module")
def pkg():
    return pf.package()


BUILT = {}


def _built(name, pkg):
    """(inputs, documents, predictor) of one case, built once per test module."""
    if name not in BUILT:
        inputs, predictor = _inputs(name), _case(name).predictor(pkg)
        BUILT[name] = (inputs, demo_pack.build(inputs, predictor), predictor)
    return BUILT[name]


ALL = tuple(ROW_TRUTH) + WINDOWS + GRAPHS


# --------------------------------------------------------------------------- strata per reading

EXPECTED_KINDS = {"binary": {"class"}, "multiclass": {"class"}, "ordinal": {"class"},
                  "multilabel": {"label", "none"}, "regression": {"bin"}, "quantile": {"bin"},
                  "reconstruction": {"all", "flagged"}, "anomaly-reconstruction": {"class"},
                  "anomaly-svdd": {"typical", "flagged"}, "forecast": {"bin"},
                  "sequence-regression": {"bin"}, "sequence-binary": {"class"},
                  "graph-keypair": {"class"}, "graph-knn": {"bin"}, "graph-level": {"class"}}


@pytest.mark.parametrize("name", ALL)
def test_strata_follow_the_reading(name, pkg):
    inputs, docs, _ = _built(name, pkg)
    strata = docs.demo["strata"]
    assert {s["kind"] for s in strata} == EXPECTED_KINDS[name]
    counts = {s["id"]: sum(it["stratum"] == s["id"] for it in docs.demo["items"]) for s in strata}
    for s in strata:
        assert counts[s["id"]] == s["n"], (s, counts)
        assert s["n"] <= (5 * PER if s["kind"] == "all" else PER)
    output = inputs.output
    if EXPECTED_KINDS[name] == {"class"}:
        # one stratum per class present in train, named by the class, in the record's order
        column = output["truth"]["column"]
        present = {str(v) for v in inputs.train[column].dropna()}
        assert [s["name"] for s in strata] == [c for c in output["classes"] if c in present]
        assert all(it["truth"] == s["name"] for s in strata for it in docs.demo["items"]
                   if it["stratum"] == s["id"])


def test_multilabel_strata_are_the_labels_and_none(pkg):
    inputs, docs, _ = _built("multilabel", pkg)
    by_id = {s["id"]: s for s in docs.demo["strata"]}
    assert by_id["none"]["name"] is None and by_id["none"]["n"] > 0
    for it in docs.demo["items"]:
        stratum = by_id[it["stratum"]]
        if stratum["kind"] == "none":
            assert it["truth"] == []
        else:
            assert stratum["name"] in it["truth"]
            assert set(it["truth"]) <= set(inputs.output["classes"])


@pytest.mark.parametrize("name", ("regression", "quantile", "forecast", "graph-knn"))
def test_value_strata_are_five_contiguous_train_bins(name, pkg):
    inputs, docs, _ = _built(name, pkg)
    bins = docs.demo["strata"]
    assert 1 <= len(bins) <= 5
    assert all(b["range"][0] <= b["range"][1] for b in bins)
    assert all(a["range"][1] == b["range"][0] for a, b in zip(bins, bins[1:]))
    column = inputs.output["truth"]["column"]
    train = pd.to_numeric(inputs.train[column], errors="coerce")
    # the edges are the train truth's own quantiles: the first starts at its minimum (a
    # forecast's first step ranges over the targets of the candidate anchors, inside it)
    assert bins[0]["range"][0] >= train.min() and bins[-1]["range"][1] <= train.max()
    by_id = {b["id"]: b for b in bins}
    for it in docs.demo["items"]:
        lo, hi = by_id[it["stratum"]]["range"]
        first = it["truth"][0] if isinstance(it["truth"], list) else it["truth"]
        assert lo <= first <= hi
    for p in docs.presets["presets"]:
        if p["group"] == "typical":
            rng = by_id[p["stratum"]]["range"]
            assert p["facts"]["bin"] == rng and p["expected"]["range"] == rng
            assert p["expected"]["kind"] == "value"
            if name == "forecast":
                # a forecast preset carries its whole horizon's truth (gate C: the first step
                # alone made the page call one step's error the mean over all of them)
                value, horizon = p["expected"]["value"], inputs.output.get("horizon") or len(p["expected"]["value"])
                assert isinstance(value, list) and len(value) > 1 and all(isinstance(v, float) for v in value), p
                assert rng[0] <= value[0] <= rng[1]
            else:
                assert isinstance(p["expected"]["value"], float)


def test_reconstruction_flags_the_top_of_the_pool(pkg):
    inputs, docs, predictor = _built("reconstruction", pkg)
    rows = inputs.train[list(c["name"] for c in inputs.schema["input"])]
    errors = predictor.predict(rows)["reconstruction_error"].to_numpy()
    assert len(rows) <= demo_strata.POOL          # the pool is the whole train part here
    top = set(np.argsort(errors, kind="stable")[::-1][:PER].tolist())
    flagged = [it for it in docs.demo["items"] if it["stratum"] == "flagged"]
    chosen = {_position(inputs, it["rows"][0]) for it in flagged}
    assert chosen == top


def test_an_unlabelled_anomaly_reads_typical_low_and_flagged_at_the_threshold(pkg):
    inputs, docs, predictor = _built("anomaly-svdd", pkg)
    rows = inputs.train[list(c["name"] for c in inputs.schema["input"])]
    scores = predictor.predict(rows)["score"].to_numpy()
    low = set(np.argsort(scores, kind="stable")[:PER].tolist())
    items = docs.demo["items"]
    assert {_position(inputs, it["rows"][0]) for it in items if it["stratum"] == "typical"} == low
    threshold = inputs.output["threshold"]
    for it in items:
        if it["stratum"] == "flagged":
            assert scores[_position(inputs, it["rows"][0])] >= threshold
    expected = {p["stratum"]: p["expected"] for p in docs.presets["presets"]
                if p["group"] == "typical"}
    assert expected == {"typical": {"kind": "alarm", "value": False},
                        "flagged": {"kind": "alarm", "value": True}}


def _position(inputs, row):
    """The train position of a JSON row (the fixture's numeric inputs are distinct floats)."""
    num = [c["name"] for c in inputs.schema["input"] if c.get("dtype") == "float"]
    hit = np.ones(len(inputs.train), dtype=bool)
    for name in num:
        hit &= np.isclose(inputs.train[name].to_numpy(dtype=float), float(row[name]))
    found = np.flatnonzero(hit)
    assert len(found) == 1, row
    return int(found[0])


# --------------------------------------------------------------------------- train part only

def test_write_draws_from_the_train_part_only():
    """Through a real bundle, as pack runs the phase: the Predictor the bundle's own CLI
    `load`s, the Recipe the bundle's map names, the files written into its examples/."""
    case = pf.row_case("binary")
    bundle = pf.Bundle.build(case, "s08_demo_binary")
    predictor = pf.load_cli(bundle).load(bundle.root)
    s = case.shaped
    raw = s.raw.copy()
    raw["y"] = np.random.default_rng(RNG_LABELS).choice([0, 1], len(raw))
    ctx = _context(BundlePaths(root=bundle.root, package=bundle.package), case, raw)
    records = demo_pack.write(ctx, predictor)
    examples = ctx.paths.examples
    demo = json.loads((examples / "demo.json").read_text("utf-8"))
    presets = json.loads((examples / "presets.json").read_text("utf-8"))
    parts = {p: {_key(r) for r in raw.iloc[s.split[p]].to_dict("records")}
             for p in ("train", "valid", "test")}
    sent = [r for it in demo["items"] for r in it["rows"]] + \
        [p["row"] for p in presets["presets"] if p["group"] == "typical"]
    assert sent and all(_key(r) in parts["train"] for r in sent)
    assert not any(_key(r) in parts["valid"] | parts["test"] for r in sent)
    table = pd.read_csv(examples / "demo.csv")
    assert {_key(r) for r in table.to_dict("records")} <= parts["train"]
    assert list(table.columns) == [c["name"] for c in case.schema["input"]] + ["y"]
    record = records["examples.demo"]
    assert (record["source_part"], record["file"], record["items"]) == \
        ("train", "examples/demo.json", len(demo["items"]))
    assert record["presets"]["count"] == len(presets["presets"])
    assert record["digest"] == pf.digest(examples / "demo.json")
    assert records["serve.demo_sample"]["per_stratum"] == PER
    assert any(m.startswith("OK: demonstration sample:") for m in ctx.markers)
    assert not _errors("demo", demo) and not _errors("presets_file", presets)
    unseen = next(p for p in presets["presets"] if p["kind"] == "edge_unseen")
    got = predictor.predict(pd.DataFrame([unseen["row"]]))
    assert unseen["facts"]["column"] in got["warnings"].iloc[0]


def _key(row):
    return tuple(round(float(row[f"x{i}"]), 6) for i in range(4))


def _context(paths, case, raw, serve=True, demo_sample=PER):
    """A pack context over the case's shaped data and `raw`, building the bundle at `paths`."""
    s = case.shaped
    campaign = Campaign(dir=paths.root.parent, doc={}, ledger={}, seal={}, run_id="R0",
                        eval_dir=paths.root.parent, metrics={}, threshold=case.threshold,
                        shaped_dir=s.dir, shaped_manifest=s.manifest, recipe=s.recipe,
                        spec=case.spec, raw=raw, fitted=s.fitted)
    options = SimpleNamespace(serve=serve, demo_sample=demo_sample, pipeline=None)
    return PackContext(options=options, paths=paths, campaign=campaign, package=paths.package,
                       ident={}, schema=case.schema, markers=[])


def test_nothing_is_written_without_a_service_or_a_sample(tmp_path, pkg):
    case = pf.row_case("regression")
    for serve, n in ((False, PER), (True, 0)):
        paths = BundlePaths(root=tmp_path / f"{serve}-{n}", package="demo_pkg")
        ctx = _context(paths, case, case.shaped.raw, serve, n)
        assert demo_pack.write(ctx, case.predictor(pkg)) == {}
        assert not ctx.paths.examples.exists()
    ctx = _context(BundlePaths(root=tmp_path / "neg", package="demo_pkg"), case,
                   case.shaped.raw, True, -1)
    with pytest.raises(PackExit) as refused:
        demo_pack.write(ctx, case.predictor(pkg))
    assert refused.value.code == 2


# --------------------------------------------------------------------------- no English

@pytest.mark.parametrize("name", ALL)
def test_presets_carry_a_kind_and_facts_and_no_sentence(name, pkg):
    inputs, docs, _ = _built(name, pkg)
    presets = docs.presets["presets"]
    assert presets and presets[0]["group"] == "typical"
    kinds = CATALOG["examples"]["preset_kinds"]
    ids = {s["id"] for s in docs.demo["strata"]}
    names = {str(c) for c in inputs.output.get("classes") or ()}
    columns = {c["name"] for c in inputs.schema["input"]}
    for p in presets:
        assert set(p) == PRESET_KEYS, p.keys()          # no name, text or description
        assert p["group"] == kinds[p["kind"]]["group"] and p["stratum"] in ids
        assert set(kinds[p["kind"]]["facts"]) <= set(p["facts"])
        cells = {str(v) for r in [p["row"] or {}, *(p["rows"] or [])] for v in r.values()}
        allowed = {p["id"], p["kind"], p["group"], p["stratum"]} | names | columns | cells | \
            {"class", "labels", "value", "alarm"}
        for text in _strings({k: p[k] for k in ("expected", "facts")}):
            assert text in allowed, (p["id"], text)
    for s in docs.demo["strata"]:
        assert s["name"] is None or s["name"] in names
    assert docs.demo["source"] == "train"


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


@pytest.mark.parametrize("name", ("binary", "multiclass", "ordinal", "multilabel",
                                  "anomaly-reconstruction", "graph-keypair", "sequence-binary"))
def test_a_typical_preset_is_answered_as_its_facts_say(name, pkg):
    inputs, docs, predictor = _built(name, pkg)
    reader = demo_presets.AnswerReader(inputs.output)
    for p in docs.presets["presets"]:
        if p["group"] != "typical":
            continue
        rows = pd.DataFrame([p["row"]] if p["row"] else p["rows"])
        got = predictor.predict(rows, pd.DataFrame(p["history"]) if p["history"] else None)
        decided = reader.decided(got)[-1]
        right, of = p["facts"].get("right"), p["facts"]["of"]
        assert of == sum(it["stratum"] == p["stratum"] for it in docs.demo["items"])
        if p["expected"]["kind"] == "class":
            assert (decided == p["expected"]["value"]) == bool(right), (p, decided)
        else:
            assert (set(reader.label_sets(got)[-1]) == set(p["expected"]["value"])) == \
                bool(right), p


# --------------------------------------------------------------------------- windows, graphs

@pytest.mark.parametrize("name", WINDOWS)
def test_window_items_carry_their_history(name, pkg):
    inputs, docs, predictor = _built(name, pkg)
    sample = inputs.sample
    need, series, order = sample["history_rows"], sample["series_by"], sample["order_by"]
    train = inputs.train
    for it in docs.demo["items"]:
        assert len(it["rows"]) == 1 and len(it["history"]) == need
        anchor = it["rows"][0]
        same = train[train[series].astype(str) == str(anchor[series])].sort_values(order)
        at = int(np.flatnonzero(same[order].to_numpy() == anchor[order])[0])
        assert [h[order] for h in it["history"]] == same[order].iloc[at - need:at].tolist()
        if name == "forecast":
            ahead = same[inputs.output["truth"]["column"]].iloc[at + 1:at + 1 +
                                                                 sample["horizon"]].tolist()
            assert it["truth"] == pytest.approx(ahead)
        got = predictor.predict(pd.DataFrame(it["rows"]), pd.DataFrame(it["history"]))
        assert bool(got["answered"].iloc[0]) and not got["warnings"].iloc[0]
    for p in docs.presets["presets"]:
        assert p["row"] is None and len(p["rows"]) == 1 and len(p["history"]) == need
        if p["group"] == "typical":
            assert p["kind"] == "typical_sequence"


@pytest.mark.parametrize("name", GRAPHS)
def test_graph_items_carry_rows_only(name, pkg):
    inputs, docs, _ = _built(name, pkg)
    level = inputs.sample.get("level")
    for it in docs.demo["items"] + docs.presets["presets"]:
        assert it["history"] is None and it["rows"]
        assert it.get("row") is None
        if level == "node":
            assert len(it["rows"]) == 1
    if level == "graph":
        group = inputs.sample["graph_id"]
        sizes = inputs.train[group].astype(str).value_counts()
        for it in docs.demo["items"]:
            ids = {r[group] for r in it["rows"]}
            assert len(ids) == 1 and len(it["rows"]) == sizes[ids.pop()]
        assert all(p["group"] == "typical" for p in docs.presets["presets"])
    else:
        # a query row carries the raw keys the Recipe attaches it by
        query = inputs.recipe.GRAPH["query_columns"]
        assert all(set(query) <= set(it["rows"][0]) for it in docs.demo["items"])


def test_row_presets_carry_a_row(pkg):
    _, docs, _ = _built("binary", pkg)
    assert all(p["row"] and p["rows"] is None and p["history"] is None
               for p in docs.presets["presets"])
    assert len(docs.demo["rows"]) == len(docs.demo["labels"]) == len(docs.demo["items"])


# --------------------------------------------------------------------------- the schema

@pytest.mark.parametrize("name", ALL)
def test_demo_and_presets_validate_against_examples_2(name, pkg):
    _, docs, _ = _built(name, pkg)
    assert _errors("demo", docs.demo) == []
    assert _errors("presets_file", docs.presets) == []
    assert demo_pack.check_documents(docs) is True
    body = {"schema": "vega/examples@2", "test": {"source": "test", "rows": [], "history": None},
            "demo": docs.demo, "presets": docs.presets["presets"]}
    assert list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(body)) == []
    json.dumps(docs.demo, allow_nan=False)
    json.dumps(docs.presets, allow_nan=False)


def test_a_malformed_document_is_refused_before_it_ships(pkg):
    _, docs, _ = _built("binary", pkg)
    bad = dict(docs.presets, presets=[dict(docs.presets["presets"][0], text="nineteen of 20")])
    with pytest.raises(PackExit) as refused:
        demo_pack.check_documents(demo_pack.DemoDocuments(docs.demo, bad, None, ()))
    assert refused.value.code == 3 and "presets.json" in str(refused.value)


def test_the_draw_is_deterministic(pkg):
    _, docs, predictor = _built("multiclass", pkg)
    again = demo_pack.build(_inputs("multiclass"), predictor)
    assert again.demo == docs.demo and again.presets == docs.presets


# --------------------------------------------------------------------------- edge presets

def test_edge_presets_start_from_the_neutral_typical_item(pkg):
    inputs, docs, predictor = _built("multiclass", pkg)
    presets = docs.presets["presets"]
    edges = [p for p in presets if p["group"] == "edge"]
    neutral = {p["stratum"] for p in edges}
    assert len(neutral) == 1
    base = next(p for p in presets if p["group"] == "typical" and p["stratum"] in neutral)
    assert base is presets[0]
    counts = inputs.train["y"].value_counts()
    most = next(s for s in docs.demo["strata"] if s["id"] == base["stratum"])["name"]
    assert counts[most] == counts.max()
    by_kind = {p["kind"]: p for p in edges}
    ranged = next(c for c in inputs.schema["input"] if (c.get("range_seen") or [0, 0])[1] > 0)
    rng = by_kind["edge_range"]
    assert rng["facts"] == {"column": ranged["name"], "factor": 10}
    assert rng["row"][ranged["name"]] == pytest.approx(ranged["range_seen"][1] * 10)
    assert {k: v for k, v in rng["row"].items() if k != ranged["name"]} == \
        {k: v for k, v in base["row"].items() if k != ranged["name"]}
    unseen = by_kind["edge_unseen"]
    assert unseen["row"][unseen["facts"]["column"]] == unseen["facts"]["value"]
    for p in (rng, unseen):
        got = predictor.predict(pd.DataFrame([p["row"]]))
        assert got["warnings"].iloc[0], p["id"]            # answered, with a warning
    assert "edge_infinity" not in by_kind                  # training saw no non-finite value


def test_the_infinity_preset_names_the_columns_training_saw_non_finite(pkg):
    case = pf.row_case("regression")
    schema = json.loads(json.dumps(case.schema))
    schema["input"][1]["non_finite_seen"] = 3
    inputs = DemoInputs(train=_train(case, "regression"), output=case.manifest["output"],
                        sample=case.manifest["sample"], schema=schema, per_stratum=PER,
                        recipe=case.shaped.module)
    docs = demo_pack.build(inputs, case.predictor(pkg))
    inf = next(p for p in docs.presets["presets"] if p["kind"] == "edge_infinity")
    name = schema["input"][1]["name"]
    assert inf["facts"] == {"columns": [name]} and inf["row"][name] == "Infinity"
    assert _errors("presets_file", docs.presets) == []


class _Recipe:
    """A Recipe whose encoder was fitted on `seen` for each named column."""

    HISTORY_COLUMNS = ()

    def __init__(self, seen):
        self.seen = seen

    def transform(self, frame, history=None, with_unseen=False):
        unseen = [tuple(c for c, known in self.seen.items() if str(row[c]) not in known)
                  for row in frame.to_dict("records")]
        return frame, unseen


def _stub_inputs(train, schema, recipe):
    return DemoInputs(train=train, output={}, sample={"shape": "row"}, schema=schema,
                      per_stratum=PER, recipe=recipe)


def test_an_unseen_code_is_asked_of_the_recipe_never_guessed():
    train = pd.DataFrame({"code": [80, 443, 8080], "city": ["x", "y", "x"]})
    schema = {"input": [{"name": "code", "dtype": "string", "pattern": "^[0-9]+$"},
                        {"name": "city", "dtype": "string"}]}
    recipe = _Recipe({"code": {"80", "443", "8080", "8081"}, "city": {"x", "y"}})
    inputs = _stub_inputs(train, schema, recipe)
    code, city = schema["input"]
    assert demo_presets.unseen_value(inputs, 0, code) == "8082"     # 8081 was fitted
    assert demo_presets.unseen_value(inputs, 0, city) == "city-unseen-0"
    deaf = _stub_inputs(train, schema, _Recipe({}))
    assert demo_presets.unseen_value(deaf, 0, city) is None


def test_the_codec_types_cells_as_the_request_contract_does():
    schema = {"input": [{"name": "code", "dtype": "string"}, {"name": "n", "dtype": "int"},
                        {"name": "v", "dtype": "float"}]}
    frame = pd.DataFrame({"code": [443.0, None], "n": [3.0, 4.0], "v": [math.inf, np.nan],
                          "key": ["B1", None]})
    assert RowCodec(schema).rows(frame) == [
        {"code": "443", "n": 3, "v": "Infinity", "key": "B1"},
        {"code": None, "n": 4, "v": None, "key": None}]


# --------------------------------------------------------------------------- stub predictor

class _Stub:
    """A Predictor answering by a fixed rule, in the output record's fields."""

    def __init__(self, output, rule):
        self.output, self.rule = output, rule

    def info(self):
        return {"output": self.output, "sample": {"shape": "row"}}

    def predict(self, frame, history=None):
        return self.rule(frame.reset_index(drop=True))


def _always_a(frame):
    n = len(frame)
    got = {"label": ["a"] * n, "p_a": [0.8] * n, "p_b": [0.1] * n, "p_c": [0.1] * n}
    got.update({f"logit_{c}": [0.0] * n for c in "abc"})
    return pd.DataFrame(dict(got, warnings=[""] * n))


def test_a_class_the_model_never_gets_right_is_its_median_case():
    case = pf.row_case("multiclass")
    inputs = _inputs("multiclass")
    docs = demo_pack.build(inputs, _Stub(case.manifest["output"], _always_a))
    typical = {p["stratum"]: p for p in docs.presets["presets"] if p["group"] == "typical"}
    names = {s["id"]: s["name"] for s in docs.demo["strata"]}
    for sid, p in typical.items():
        if names[sid] == "a":
            assert p["kind"] == "typical_correct" and p["facts"]["right"] == p["facts"]["of"]
        else:
            assert p["kind"] == "typical_none_correct" and p["facts"]["right"] == 0
    assert _errors("presets_file", docs.presets) == []


def _scores(frame):
    n = len(frame)
    score = frame["x0"].to_numpy(dtype=float) ** 2
    return pd.DataFrame({"score": score, "prediction": np.zeros(n, dtype=int),
                         **{f"emb_{j}": np.zeros(n) for j in range(5)}, "warnings": [""] * n})


@pytest.mark.parametrize("threshold", (1e9, None))
def test_an_empty_flagged_stratum_is_listed_and_said(threshold):
    output = dict(pf.row_case("anomaly-svdd").manifest["output"], threshold=threshold)
    inputs = _inputs("anomaly-svdd", output=output)
    docs = demo_pack.build(inputs, _Stub(output, _scores))
    flagged = next(s for s in docs.demo["strata"] if s["id"] == "flagged")
    assert flagged["n"] == 0
    assert any("flagged is empty" in note for note in docs.notes)
    typical = next(p for p in docs.presets["presets"] if p["stratum"] == "typical")
    assert typical["expected"] == ({"kind": "alarm", "value": False} if threshold else None)
    assert _errors("demo", docs.demo) == [] and _errors("presets_file", docs.presets) == []


def test_an_answer_without_the_record_fields_is_refused():
    case = pf.row_case("multiclass")
    with pytest.raises(PackExit) as refused:
        demo_pack.build(_inputs("multiclass"),
                        _Stub(case.manifest["output"], lambda f: _always_a(f).drop(columns="p_b")))
    assert refused.value.code == 3 and "p_b" in str(refused.value)


class _Refusing:
    """A Recipe that refuses any frame it is asked about (a text step that parses the column)."""

    HISTORY_COLUMNS = ()

    def transform(self, frame, history=None, with_unseen=False):
        raise ValueError("zone: not a known spelling")


def test_a_recipe_refusing_the_candidates_is_said_not_raised():
    case = pf.row_case("multiclass")
    inputs = DemoInputs(train=_train(case, "multiclass"), output=case.manifest["output"],
                        sample=case.manifest["sample"], schema=case.schema, per_stratum=PER,
                        recipe=_Refusing())
    docs = demo_pack.build(inputs, _Stub(case.manifest["output"], _always_a))
    assert not any(p["kind"] == "edge_unseen" for p in docs.presets["presets"])
    assert any("refused the candidate codes (ValueError" in note for note in docs.notes)
    assert any(p["kind"] == "edge_range" for p in docs.presets["presets"])
    deaf = DemoInputs(train=inputs.train, output=inputs.output, sample=inputs.sample,
                      schema=inputs.schema, per_stratum=PER, recipe=_Recipe({}))
    docs = demo_pack.build(deaf, _Stub(case.manifest["output"], _always_a))
    assert not docs.notes and not any(p["kind"] == "edge_unseen"
                                      for p in docs.presets["presets"])
    assert any(r.startswith("no edge_unseen preset for zone") for r in docs.remarks)


def test_truths_are_read_as_the_service_reads_a_request_truth():
    """A class cell read as a float (`1.0`, a label column with a gap) is its class; a raw
    spelling decodes through schema.json; a missing or unknown cell has no truth and is
    counted; a multilabel empty cell is `none`; a forecast's truth is its horizon's targets."""
    from demo_samples import samples_for
    train = pd.DataFrame({"x": np.arange(6.0),
                          "y": [1.0, 0.0, np.nan, 7.0, "yes", 1.0]}).astype({"y": object})
    schema = {"input": [{"name": "x", "dtype": "float"}],
              "label": {"name": "y", "decode": {"yes": "1"}}}
    output = {"reading": "binary", "classes": ["0", "1"],
              "truth": {"column": "y", "type": "class", "sep": None}}
    inputs = DemoInputs(train=train, output=output, sample={"shape": "row"}, schema=schema,
                        per_stratum=PER, recipe=None)
    truths = demo_strata.Truths.of(inputs, samples_for(inputs))
    assert truths.values == {0: "1", 1: "0", 4: "1", 5: "1"} and truths.dropped == 2
    labels = dict(output, reading="multilabel", classes=["a", "b"],
                  truth={"column": "y", "type": "labels", "sep": "|"})
    train = pd.DataFrame({"x": np.arange(4.0), "y": ["b|a", np.nan, "a", "a|z"]})
    inputs = DemoInputs(train=train, output=labels, sample={"shape": "row"}, schema=schema,
                        per_stratum=PER, recipe=None)
    truths = demo_strata.Truths.of(inputs, samples_for(inputs))
    assert truths.values == {0: ["a", "b"], 1: [], 2: ["a"]} and truths.dropped == 1


def test_a_forecast_truth_is_its_horizon_and_a_non_finite_one_is_left_out():
    from demo_samples import samples_for
    train = pd.DataFrame({"s": ["A"] * 6, "t": range(6), "y": [0.0, 1.0, 2.0, np.inf, 4.0, 5.0]})
    output = {"reading": "forecast", "truth": {"column": "y", "type": "number", "sep": None}}
    sample = {"shape": "window", "length": 2, "history_rows": 1, "horizon": 2,
              "series_by": "s", "order_by": "t"}
    inputs = DemoInputs(train=train, output=output, sample=sample,
                        schema={"input": [{"name": "s", "dtype": "string"}]}, per_stratum=PER,
                        recipe=None)
    samples = samples_for(inputs)
    assert samples.keys.tolist() == [1, 2, 3]       # a history row before, two targets after
    truths = demo_strata.Truths.of(inputs, samples)
    assert truths.values == {3: [4.0, 5.0]} and truths.dropped == 2


@pytest.mark.parametrize("name", WINDOWS)
def test_window_runs_answer_as_each_item_is_sent(name, pkg):
    """The candidates are answered in runs of their series; each answer equals the one the
    item gets when sent alone with its history (a fully determined window, R-A6)."""
    from demo_samples import samples_for
    inputs, _, predictor = _built(name, pkg)
    samples = samples_for(inputs)
    keys = samples.keys[::3][:12]
    runs = samples.answers(predictor, keys)
    fields = [f for f, k in inputs.output["field_kinds"].items()
              if k["kind"] in ("value", "horizon", "raw", "probability")]
    assert fields and len(keys)
    for key in keys:
        item = samples.item(key)
        alone = predictor.predict(pd.DataFrame(item["rows"]), pd.DataFrame(item["history"]))
        assert bool(alone["answered"].iloc[0]) and bool(runs.loc[key, "answered"])
        for field in fields:
            assert float(alone[field].iloc[0]) == pytest.approx(float(runs.loc[key, field]),
                                                                rel=1e-5, abs=1e-6), field


def test_a_serving_adapter_older_than_the_record_writes_no_demo(tmp_path):
    """Until the serving cutover a served bundle may carry the v2.2 adapter, whose answer is not
    the output record's: the phase says so and writes nothing; `build` itself refuses (exit 3)."""
    case = pf.row_case("multiclass")
    raw = case.shaped.raw.copy()
    raw["y"] = np.random.default_rng(RNG_LABELS).choice(["a", "b", "c"], len(raw))
    old = _Stub(case.manifest["output"], lambda f: _always_a(f)[["label", "warnings"]])
    ctx = _context(BundlePaths(root=tmp_path / "old", package="demo_pkg"), case, raw)
    shutil.copytree(case.shaped.dir, ctx.paths.pkg / "features", dirs_exist_ok=True)
    assert demo_pack.write(ctx, old) == {}
    assert not (ctx.paths.examples / "demo.json").exists()
    assert any(m.startswith("WARN: no demonstration sample: the bundle's Predictor answers no")
               for m in ctx.markers)
