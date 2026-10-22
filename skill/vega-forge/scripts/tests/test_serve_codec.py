#!/usr/bin/env python3
"""
test_serve_codec.py — the serve contract split and the generic answer codec (build plan S02;
D2 §5, R-A8, R-C4; bundle-contract §7.3).

    python3 test_serve_codec.py [-v]        (or: pytest test_serve_codec.py)

The five templates `templates/serving/serve/{contract, rows, tables, answers, problems}.py` are
rendered the way `serving_pack` renders `serve/` (the placeholders filled, none left) into a
package of their own and imported. Output records are expanded from vega-pack's catalog
(`output_readings.readings`, the patterns K2 wrote) for every reading, so the fixtures follow the
contract rather than a copy of it. Then:

1. Answer keys follow `field_kinds`: per reading, the JSON keys are R-C4's (an oracle written
   here from the contract's table, not from the codec), every shown field's value is where its
   kind says, `logits`, `reconstruction` and `embedding` appear only on their `include`, and the
   pydantic `RowPrediction` accepts the answer and refuses a drifted key. A non-finite number
   in an answered row is null and the response model holds it; through FastAPI's default
   serialization a gated key without its include is absent, never null.
2. Shapes: an unanswered window row keeps its keys, null, `answered: false`; a graph-level
   answer carries `graph_id`, and the column naming a request row's graph is never null;
   `history` is taken by the row and window shapes, and a graph request that sends it is
   refused.
3. The truth is held to its type (`MANIFEST.output.truth`): a class (decoded, one of the
   classes), a finite number, or label names (a list in a request; one text split on `sep` in a
   row and in a CSV upload); one per row, `labels` accepted as its name; a bundle without truth
   takes none; the truth never reaches the recipe's frame.
4. The CSV answer's columns are `MANIFEST.output.fields` (every include given), the gated kinds
   only on include, `true_<column>` before `answered`/`warnings`; an upload is held to the same
   row model as a JSON body and gives the same raw frame.
5. The records: a v1 output record, an unknown kind, a field without a kind are refused; the
   contract is immutable (what it hands out included), versioned by the schema's digest, refuses
   a schema the manifest did not pin or pinned otherwise; `/v1/schema` carries the whole output
   record, the sample and the truth; every envelope names the reading.
6. The codec's kind table is vega-pack's `field_kinds` catalog; `conformance.py --size-only`
   reports no C11–C13 finding on the five files, and each file is within its D2 §5 line budget;
   the one literal gate (`lint_templates.py`, R-F10) finds nothing in them when it is present.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import atexit
import hashlib
import importlib
import io
import itertools
import json
import math
import operator
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
SERVE = SCRIPTS / "backends" / "pytorch" / "templates" / "serving" / "serve"
FILES = ("contract.py", "rows.py", "tables.py", "answers.py", "problems.py")
# D2 §5's line budget per file (the build plan lists S02 among the template budgets)
BUDGETS = {"problems.py": 160, "rows.py": 250, "tables.py": 250, "answers.py": 200,
           "contract.py": 200}
PACK_CATALOG = SKILLS / "vega-pack" / "scripts" / "catalog.json"
LINT = SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
PACKAGE = "serve_codec_under_test"
VERBOSE = "-v" in sys.argv

sys.path.insert(0, str(SCRIPTS))
import conformance  # noqa: E402

# the fills serving_pack._serve_fills gives the serve templates (an identity, not a dataset)
FILLS = {"{{ROW_MODEL}}": "ThingRow", "{{NOUNS}}": "things", "{{NOUN}}": "thing",
         "{{PACKAGE}}": "thing_pipeline", "{{ENV}}": "THING", "{{CMD}}": "thing"}


# --------------------------------------------------------------------------- the rendered package

def _render_package() -> Path:
    """The five templates rendered into `<tmp>/serve_codec_under_test/`, importable."""
    out = Path(tempfile.mkdtemp(prefix="serve-codec-"))
    atexit.register(shutil.rmtree, out, True)
    pkg = out / PACKAGE
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""serve (rendered for test_serve_codec)."""\n',
                                     encoding="utf-8")
    for name in FILES:
        text = (SERVE / name).read_text(encoding="utf-8")
        for key, value in FILLS.items():
            text = text.replace(key, value)
        left = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", text)))
        assert not left, f"serve/{name} keeps unfilled placeholder(s) {left}"
        (pkg / name).write_text(text, encoding="utf-8")
    return out


_ROOT = _render_package()
sys.path.insert(0, str(_ROOT))
contract_mod = importlib.import_module(f"{PACKAGE}.contract")
answers_mod = importlib.import_module(f"{PACKAGE}.answers")
rows_mod = importlib.import_module(f"{PACKAGE}.rows")
tables_mod = importlib.import_module(f"{PACKAGE}.tables")
problems_mod = importlib.import_module(f"{PACKAGE}.problems")
ApiContract = contract_mod.ApiContract
AnswerCodec = answers_mod.AnswerCodec
ContractError = problems_mod.ContractError
thawed = problems_mod.thawed
CATALOG = json.loads(PACK_CATALOG.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- records from the catalog

SCHEMA = {
    "input": [
        {"name": "amount", "dtype": "float", "required": True, "range_seen": [0.5, 90.0]},
        {"name": "code", "dtype": "string", "required": True, "pattern": "^[0-9]+$",
         "allowed": "any"},
        {"name": "region", "dtype": "string", "required": False, "allowed": ["north", "south"]}],
    "ignored": ["truth_col", "row_key"],
    "label": {"name": "truth_col", "decode": {"yes": "1", "no": "0"}},
}
FEATURES = ["amount", "region"]                         # the FeatureContract, numeric then nominal
SEP = "|"

# the prose conditions of output_readings.readings[*].conditional, as predicates
CONDITIONS = {
    "0.5 is one of the quantiles": lambda v: 0.5 in (v["quantiles"] or ()),
    "head_out equals the number of input features":
        lambda v: v["head_out"] == len(v["features"]),
    "head_out differs from the number of input features":
        lambda v: v["head_out"] != len(v["features"]),
    "task.classes is present (a labelled anomaly task)": lambda v: bool(v["classes"]),
    "the loss has no quantiles": lambda v: not v["quantiles"],
    "the loss has quantiles (horizon-major)": lambda v: bool(v["quantiles"]),
}


def _qtok(q: float) -> str:
    return format(q * 100, "g").replace(".", "p")


def _tokens(name: str, v: dict) -> list:
    """The values one catalog placeholder takes for this record."""
    median = min(v["quantiles"] or [0.5], key=lambda q: abs(q - 0.5))
    table = {"class": v["classes"], "label": v["classes"], "positive": [v["positive"]],
             "cum": v["classes"][:-1], "q": [_qtok(q) for q in v["quantiles"] or ()],
             "median": [_qtok(median)], "h": v["horizons"], "feature": v["features"],
             "j": list(range(v["head_out"])), "k": list(range(v["head_out"]))}
    return list(table[name])


def _expand(pattern: str, of, v: dict) -> list:
    """(field name, of) for every instance of a field pattern (horizon-major for two)."""
    names = re.findall(r"<(\w+)>", pattern)
    out = []
    for combo in itertools.product(*[_tokens(n, v) for n in names]):
        bound = dict(zip(names, combo))
        field = pattern
        for n, value in bound.items():
            field = field.replace(f"<{n}>", str(value))
        out.append((field, _of(of, bound, v)))
    return out


def _of(of, bound: dict, v: dict):
    if of is None:
        return None
    single = re.fullmatch(r"<(\w+)>", of)
    if single and single.group(1) in ("h", "j", "k"):
        return int(bound[single.group(1)])
    text = of
    for n in re.findall(r"<(\w+)>", of):
        text = text.replace(f"<{n}>", str(bound[n] if n in bound else _tokens(n, v)[0]))
    return text


def _record(reading: str, shape: str = "row", level: str | None = None, **values) -> dict:
    """A `vega/output@2` record for one reading, expanded from vega-pack's catalog."""
    v = {"classes": [], "positive": None, "quantiles": None, "horizons": [], "features": FEATURES,
         "head_out": 1, "truth": None}
    v.update(values)
    spec = CATALOG["output_readings"]["readings"][reading]
    fields, kinds = [], {}
    if shape == "graph" and level == "graph":
        fields.append("graph_id")
        kinds["graph_id"] = {"kind": "group", "of": None}
    for pattern in spec["fields"]:
        rule = spec.get("conditional", {}).get(pattern)
        if rule is not None and not CONDITIONS[rule](v):
            continue
        for name, of in _expand(pattern, spec["kinds"][pattern]["of"], v):
            fields.append(name)
            kinds[name] = {"kind": spec["kinds"][pattern]["kind"], "of": of}
    if shape == "window":
        fields.append("answered")
        kinds["answered"] = {"kind": "answered", "of": None}
    fields.append("warnings")
    kinds["warnings"] = {"kind": "warnings", "of": None}
    truth = v["truth"]
    if truth is None and spec["truth"] and not (spec["truth_optional"] and not v["classes"]):
        truth = {"column": "truth_col", "type": spec["truth"],
                 "sep": SEP if spec["truth"] == "labels" else None}
    return {"schema": "vega/output@2", "kind": values.get("kind", reading), "reading": reading,
            "base": None, "link": spec["link"], "decision": spec["decision"],
            "head_out": v["head_out"], "classes": v["classes"] or None,
            "positive": v["positive"], "threshold": None, "threshold_scale":
                spec["threshold_scale"], "quantiles": v["quantiles"],
            "horizons": v["horizons"] or None, "fields": fields, "field_kinds": kinds,
            "compared": [], "decided": [], "score_field": None, "truth": truth}


def cases() -> dict:
    """One record per reading (and per variant a reading has)."""
    two, three = ["0", "1"], ["alpha", "beta", "gamma"]
    return {
        "binary": _record("binary", classes=two, positive="1"),
        "multiclass": _record("multiclass", classes=three, head_out=3),
        "multilabel": _record("multilabel", classes=["red", "green", "blue"], head_out=3),
        "ordinal": _record("ordinal", classes=["low", "mid", "high"], head_out=2),
        "regression": _record("regression"),
        "quantile": _record("quantile", quantiles=[0.1, 0.5, 0.9], head_out=3),
        "forecast": _record("forecast", horizons=[1, 2, 3], head_out=3),
        "forecast-quantiles": _record("forecast", horizons=[1, 2], quantiles=[0.1, 0.9],
                                      head_out=4),
        "reconstruction": _record("reconstruction", head_out=2),
        "reconstruction-wide": _record("reconstruction", head_out=4),
        "anomaly-reconstruction": _record("anomaly-reconstruction", classes=["normal", "odd"],
                                          positive="odd", head_out=2),
        "anomaly-svdd": _record("anomaly-svdd", head_out=3),
        "anomaly-svdd-labelled": _record("anomaly-svdd", classes=["normal", "odd"],
                                         positive="odd", head_out=3),
        "window-multiclass": _record("multiclass", shape="window", classes=three, head_out=3,
                                     kind="sequence-classification"),
        "graph-level-regression": _record("regression", shape="graph", level="graph"),
    }


def frame_for(rec: dict, n: int = 6, unanswered=()) -> pd.DataFrame:
    """A Predictor-shaped frame for a record: one column per field, deterministic values."""
    rng = np.random.default_rng(11)
    classes = list(rec["classes"] or [])
    data = {}
    for name in rec["fields"]:
        kind = rec["field_kinds"][name]["kind"]
        if kind == "class":
            data[name] = [None if i in unanswered else classes[i % len(classes)]
                          for i in range(n)]
        elif kind in ("flag", "decision", "alarm"):
            data[name] = [(i + len(name)) % 2 for i in range(n)]
        elif kind == "group":
            data[name] = [f"g-{i // 2}" for i in range(n)]
        elif kind == "answered":
            data[name] = [i not in unanswered for i in range(n)]
        elif kind == "warnings":
            data[name] = ["" if i % 3 else "first note; second note" for i in range(n)]
        else:
            data[name] = [math.nan if i in unanswered else float(x)
                          for i, x in enumerate(rng.normal(size=n).round(6))]
    return pd.DataFrame(data)


def contract_for(rec: dict, shape: str = "row", max_rows: int = 50, schema=None,
                 sample=None) -> ApiContract:
    sample = sample or {"shape": shape}
    return ApiContract(schema or SCHEMA, {"output": rec, "sample": sample,
                                          "chain": {"shape": {"manifest_digest":
                                                              "sha256:" + "f" * 64}}},
                       max_rows=max_rows, version="sha256:" + "a" * 64)


# --------------------------------------------------------------------------- the oracle (R-C4)

ALWAYS = {"binary": {"label", "probabilities"}, "multiclass": {"label", "probabilities"},
          "multilabel": {"labels", "probabilities"},
          "ordinal": {"label", "probabilities", "cumulative"}, "regression": {"value"},
          "quantile": {"value", "quantiles"}, "forecast": {"values"},
          "reconstruction": {"reconstruction_error"},
          "anomaly-reconstruction": {"score", "anomalous"}, "anomaly-svdd": {"score", "anomalous"}}
ON_INCLUDE = {"binary": {"logits"}, "multiclass": {"logits"}, "multilabel": {"logits"},
              "ordinal": {"logits"}, "reconstruction": {"reconstruction"},
              "anomaly-reconstruction": {"reconstruction"}, "anomaly-svdd": {"embedding"}}
EVERY_INCLUDE = ("logits", "reconstruction", "embedding")


def expected_keys(rec: dict, include=()) -> set:
    keys = {"answered", "warnings"} | ALWAYS[rec["reading"]]
    keys |= {k for k in ON_INCLUDE.get(rec["reading"], set()) if k in include}
    if rec["reading"].startswith("anomaly") and rec["classes"]:
        keys.add("label")
    if "graph_id" in rec["fields"]:
        keys.add("graph_id")
    return keys


def found(item: dict, kind: str, of):
    """Where the contract's table (bundle-contract §7.3) says a field of `kind` lands."""
    if kind == "class":
        return item["label"]
    if kind == "flag":
        return int(str(of) in item["labels"])
    if kind == "alarm":
        return int(item["anomalous"])
    if kind == "horizon":
        step, _, q = str(of).partition("/")
        return item["values"][f"h{step}"][q] if q else item["values"][f"h{step}"]
    if kind == "embedding":
        return item["embedding"][int(of)]
    if kind == "group":
        return item["graph_id"]
    key = {"probability": "probabilities", "raw": "logits", "quantile": "quantiles",
           "cumulative": "cumulative", "recon": "reconstruction"}.get(kind)
    if key:
        return item[key][str(of)]
    return item[{"value": "value", "score": "score", "row_score": "reconstruction_error"}[kind]]


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-12
    return a == b


def _refusal(contract, model, body) -> list:
    """The grouped errors a request model's refusal of `body` gives (it must refuse it)."""
    try:
        model.model_validate(body)
    except ValidationError as exc:
        return contract.problems.group(exc.errors())
    raise AssertionError(f"{model.__name__} accepted {body}")


def _table_refusal(contract, text: str) -> list:
    """The errors a CSV upload's refusal gives, as plain JSON (it must be refused)."""
    try:
        contract.tables.accept(_read_csv(text))
    except ContractError as exc:
        return thawed(exc.errors)
    raise AssertionError(f"the upload {text!r} was accepted")


def _read_csv(text: str) -> pd.DataFrame:
    """A CSV read the way the bundle reads an upload: the schema's text columns as text."""
    text_columns = {c["name"] for c in SCHEMA["input"] if c.get("dtype") == "string"}
    return pd.read_csv(io.StringIO(text), dtype={c: str for c in text_columns | {"truth_col",
                                                                                "row_key"}},
                       keep_default_na=True)


# --------------------------------------------------------------------------- 1. answers per reading

def test_answer_keys_follow_the_field_kinds():
    for name, rec in cases().items():
        codec = contract_for(rec).answers
        frame = frame_for(rec)
        for include in ((), ("logits",), ("reconstruction",), ("embedding",), EVERY_INCLUDE):
            want = expected_keys(rec, include)
            for item in codec.rows(frame, include):
                assert set(item) == want, f"{name} include={include}: {sorted(item)} != " \
                                          f"{sorted(want)}"
            assert set(codec.keys(include)) == want, f"{name}: keys() {codec.keys(include)}"
        if VERBOSE:
            print(f"        {name}: {sorted(expected_keys(rec, EVERY_INCLUDE))}")


def test_every_shown_field_is_where_its_kind_says():
    for name, rec in cases().items():
        codec = contract_for(rec).answers
        frame = frame_for(rec)
        items = codec.rows(frame, EVERY_INCLUDE)
        for field in rec["fields"]:
            kind, of = rec["field_kinds"][field]["kind"], rec["field_kinds"][field]["of"]
            if kind in ("decision", "answered", "warnings"):
                continue
            for i, item in enumerate(items):
                got, want = found(item, kind, of), frame[field].iloc[i]
                want = want.item() if hasattr(want, "item") else want
                assert _same(got, want), f"{name}.{field} row {i}: {got!r} != {want!r}"
        for i, item in enumerate(items):
            text = frame["warnings"].iloc[i]
            assert item["warnings"] == [w for w in text.split("; ") if w], (name, item)
            assert item["answered"] is True


def test_the_binary_answer_is_the_v22_shape():
    rec = cases()["binary"]
    frame = pd.DataFrame({"label": ["1", "0"], "p_0": [0.31, 0.8], "p_1": [0.69, 0.2],
                          "score": [0.8, -1.39], "prediction": [1, 0], "warnings": ["", ""]})
    items = contract_for(rec).answers.rows(frame, ("logits",))
    assert items[0] == {"answered": True, "label": "1", "probabilities": {"0": 0.31, "1": 0.69},
                        "logits": {"1": 0.8}, "warnings": []}, items[0]
    assert "prediction" not in items[0] and items[1]["label"] == "0"


def test_the_forecast_answer_is_keyed_by_step():
    items = contract_for(cases()["forecast"]).answers.rows(pd.DataFrame(
        {"value_h1": [1.5], "value_h2": [2.5], "value_h3": [3.5], "warnings": [""]}))
    assert items[0] == {"answered": True, "values": {"h1": 1.5, "h2": 2.5, "h3": 3.5},
                        "warnings": []}, items[0]
    rec = cases()["forecast-quantiles"]
    frame = pd.DataFrame({f: [float(i)] for i, f in enumerate(rec["fields"][:-1])})
    frame["warnings"] = [""]
    values = contract_for(rec).answers.rows(frame)[0]["values"]
    assert values == {"h1": {"10": 0.0, "90": 1.0}, "h2": {"10": 2.0, "90": 3.0}}, values


def test_embedding_follows_coordinate_order():
    rec = cases()["anomaly-svdd"]
    shuffled = dict(rec, fields=["score", "prediction", "emb_2", "emb_0", "emb_1", "warnings"])
    frame = pd.DataFrame({"score": [2.0], "prediction": [1], "emb_0": [0.0], "emb_1": [1.0],
                          "emb_2": [2.0], "warnings": [""]})
    item = AnswerCodec(shuffled).rows(frame, ("embedding",))[0]
    assert item["embedding"] == [0.0, 1.0, 2.0] and item["anomalous"] is True, item


def test_gated_keys_only_on_include():
    for name, rec in cases().items():
        codec = contract_for(rec).answers
        frame = frame_for(rec)
        for gate in EVERY_INCLUDE:
            others = set(EVERY_INCLUDE) - {gate}
            for item in codec.rows(frame, (gate, "rows")):
                assert not others & set(item), f"{name}: {sorted(others & set(item))} without " \
                                               "their include"
        plain = codec.rows(frame)[0]
        assert not set(EVERY_INCLUDE) & set(plain), f"{name}: gated keys without include"


def test_the_response_model_holds_the_answer_and_refuses_a_drift():
    for name, rec in cases().items():
        contract = contract_for(rec)
        frame = frame_for(rec)
        for include in ((), EVERY_INCLUDE):
            body = dict(contract.envelope("sha256:" + "b" * 64, len(frame), "v1"),
                        predictions=contract.answers.rows(frame, include))
            model = contract.PredictResponse.model_validate(body)
            dumped = model.model_dump(mode="json")     # no exclude_unset: the model's own JSON
            assert dumped["output_kind"] == rec["reading"], name
            for item, sent in zip(dumped["predictions"], body["predictions"]):
                assert item == sent, f"{name}: the model reshaped {sorted(sent)} to {sorted(item)}"
        drifted = dict(body, predictions=[dict(body["predictions"][0], surprise=1.0)])
        try:
            contract.PredictResponse.model_validate(drifted)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"{name}: a key the record does not name was accepted")
    rec = cases()["multiclass"]
    bad = dict(contract_for(rec).envelope("m", 1, "v1"),
               predictions=[{"answered": True, "label": "delta", "probabilities": {},
                             "warnings": []}])
    try:
        contract_for(rec).PredictResponse.model_validate(bad)
    except ValidationError:
        pass
    else:
        raise AssertionError("a label outside the classes was accepted by the response model")


NUMBERS = ("probability", "raw", "quantile", "cumulative", "recon", "horizon", "embedding",
           "value", "score", "row_score")


def test_a_non_finite_answer_is_null_and_the_model_holds_it():
    """JSON has no literal for NaN or an infinity: in an answered row such a number is null —
    inside a mapped key, a horizon key and the embedding as in a scalar key — and the response
    model accepts the answer (a refusal there is FastAPI's 500)."""
    for name, rec in cases().items():
        contract = contract_for(rec)
        frame = frame_for(rec)
        numbers = [f for f in rec["fields"] if rec["field_kinds"][f]["kind"] in NUMBERS]
        assert numbers, name
        for i, bad in enumerate((math.inf, -math.inf, math.nan)):
            frame.loc[i, numbers] = bad
        items = contract.answers.rows(frame, EVERY_INCLUDE)
        for field in numbers:
            kind, of = rec["field_kinds"][field]["kind"], rec["field_kinds"][field]["of"]
            got = [found(item, kind, of) for item in items[:3]]
            assert got == [None, None, None], f"{name}.{field}: {got}"
            assert found(items[3], kind, of) is not None, f"{name}.{field}: row 3 lost"
        assert all(item["answered"] is True for item in items), name
        body = dict(contract.envelope("m", len(frame), "v1"), predictions=items)
        dumped = contract.PredictResponse.model_validate(body).model_dump(mode="json")
        assert dumped["predictions"] == items, name
        json.dumps(dumped, allow_nan=False)                   # strict JSON, no NaN literal


def test_fastapi_writes_the_answer_the_codec_wrote():
    """Through FastAPI's default serialization (no `response_model_exclude_unset`), a route
    returning the codec's dicts, or the validated model, answers what the codec wrote: a gated
    key without its include is absent, not null; an unanswered row keeps its nulls; a
    non-finite number in an answered row is null with a 200, never a 500."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from fastapi import FastAPI, Query
            from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"        not run: {exc}")
        return
    rec = cases()["window-multiclass"]
    contract = contract_for(rec, shape="window")
    frame = frame_for(rec, unanswered=(0,))
    frame.loc[2, "p_alpha"] = math.inf
    app = FastAPI()

    def body(include) -> dict:
        return dict(contract.envelope("m", len(frame), "v1"),
                    predictions=contract.answers.rows(frame, tuple(include)))

    @app.get("/dicts", response_model=contract.PredictResponse)
    def dicts(include: list[str] = Query(default=[])):
        return body(include)

    @app.get("/model", response_model=contract.PredictResponse)
    def model(include: list[str] = Query(default=[])):
        return contract.PredictResponse.model_validate(body(include))

    client = TestClient(app)
    for route in ("/dicts", "/model"):
        for include in ((), ("logits",)):
            reply = client.get(route, params={"include": list(include)})
            assert reply.status_code == 200, (route, reply.status_code, reply.text[:300])
            items = reply.json()["predictions"]
            assert items == body(include)["predictions"], (route, include, items[:2])
            assert all(("logits" in item) == bool(include) for item in items), (route, items)
        assert items[0]["answered"] is False and items[0]["label"] is None, items[0]
        assert items[0]["probabilities"] is None and items[0]["logits"] is None, items[0]
        assert items[2]["answered"] is True and items[2]["probabilities"]["alpha"] is None


# --------------------------------------------------------------------------- 2. shapes

def test_an_unanswered_window_row_keeps_its_keys_null():
    rec = cases()["window-multiclass"]
    contract = contract_for(rec, shape="window")
    frame = frame_for(rec, unanswered=(0, 1))
    frame.loc[0, "warnings"] = "window incomplete: needs 23 preceding rows, 0 available"
    items = contract.answers.rows(frame, ("logits",))
    assert items[0] == {"answered": False, "label": None, "probabilities": None, "logits": None,
                        "warnings": ["window incomplete: needs 23 preceding rows, 0 available"]}, \
        items[0]
    assert items[2]["answered"] is True and items[2]["label"] in rec["classes"]
    body = dict(contract.envelope("m", len(frame), "v1"), predictions=items)
    contract.PredictResponse.model_validate(body)
    table = contract.answers.table(frame)
    assert list(table.columns)[-2:] == ["answered", "warnings"], list(table.columns)
    assert table["answered"].tolist()[:3] == [False, False, True]


def test_a_graph_level_answer_names_its_graph():
    rec = cases()["graph-level-regression"]
    sample = {"shape": "graph", "level": "graph", "graph_id": "graph_key"}
    schema = dict(SCHEMA, input=SCHEMA["input"] + [{"name": "graph_key", "dtype": "string",
                                                    "required": True}])
    contract = contract_for(rec, schema=schema, sample=sample)
    items = contract.answers.rows(frame_for(rec))
    assert [i["graph_id"] for i in items] == ["g-0", "g-0", "g-1", "g-1", "g-2", "g-2"]
    rows = [{"amount": 1.0, "code": "7", "graph_key": "g-1"}]
    contract.models.PredictRequest.model_validate({"rows": rows})
    errors = _refusal(contract, contract.models.PredictRequest,
                      {"rows": rows + [dict(rows[0], graph_key=None)]})
    assert [(e["column"], e["rows"]) for e in errors] == [("graph_key", [1])], errors
    assert _table_refusal(contract, "amount,code,graph_key\n1,7,g-1\n2,8,\n")[0]["column"] == \
        "graph_key"
    # at node level (and in any other column) a null is a missing value, the recipe's to handle
    node = contract_for(rec, schema=schema, sample={"shape": "graph", "level": "node"})
    node.models.PredictRequest.model_validate({"rows": [dict(rows[0], graph_key=None,
                                                             region=None)]})


def test_history_is_taken_by_the_row_and_window_shapes_only():
    rec = cases()["regression"]
    rows = [{"amount": 1.0, "code": "7"}]
    for shape in ("row", "window"):
        models = contract_for(rec, shape=shape).models
        req = models.PredictRequest.model_validate({"rows": rows, "history": rows * 3})
        assert len(req.history) == 3, shape
        models.MonitorRequest.model_validate({"rows": rows, "history": rows})
        models.OneRowRequest.model_validate({"row": rows[0], "history": rows})
    contract = contract_for(rec, shape="graph")
    contract.models.PredictRequest.model_validate({"rows": rows})
    for model, body in ((contract.models.PredictRequest, {"rows": rows, "history": rows}),
                        (contract.models.MonitorRequest, {"rows": rows, "history": rows}),
                        (contract.models.OneRowRequest, {"row": rows[0], "history": rows})):
        errors = _refusal(contract, model, body)
        assert errors == [{"column": None,
                           "error": "history: not a field this bundle's request takes"}], errors
    # a bad history row is named as one; a bad one-row request names its column
    window = contract_for(rec, shape="window")
    errors = _refusal(window, window.models.PredictRequest,
                      {"rows": rows, "history": [{"amount": "lots", "code": "7"}]})
    assert errors == [{"column": "amount", "error": "expects a number, null, or one of the "
                       "tokens Infinity, -Infinity, NaN", "rows": [0], "count": 1,
                       "field": "history"}], errors
    errors = _refusal(window, window.models.OneRowRequest, {"row": {"amount": 1.0, "code": -7}})
    assert errors == [{"column": "code", "error": "expects a code: a non-negative integer, or its "
                       "decimal digits as text"}], errors


def test_the_row_cap_holds_for_rows_and_history():
    contract = contract_for(cases()["regression"], max_rows=3)
    row = {"amount": 1.0, "code": "7"}
    for body in ({"rows": [row] * 4}, {"rows": [row], "history": [row] * 4}, {"rows": []}):
        try:
            contract.models.PredictRequest.model_validate(body)
        except ValidationError:
            continue
        raise AssertionError(f"accepted {body} over a cap of 3")


# --------------------------------------------------------------------------- 3. the truth

def test_a_class_truth_is_one_of_the_classes():
    contract = contract_for(cases()["binary"])
    rows = [{"amount": 1.0, "code": "7"}] * 3
    req = contract.models.MonitorRequest.model_validate({"rows": rows, "truth": ["1", "yes", 0]})
    assert req.truth == ["1", "1", "0"], req.truth              # decoded; an integer is its digits
    req = contract.models.MonitorRequest.model_validate({"rows": rows, "labels": ["0", "0", "1"]})
    assert req.truth == ["0", "0", "1"]                          # the permanent alias
    for key in ("truth", "labels"):
        for truth in (["maybe", "1", "0"], [True, "1", "0"], [None, "1", "0"]):
            errors = _refusal(contract, contract.models.MonitorRequest, {"rows": rows,
                                                                         key: truth})
            assert errors == [{"column": "truth_col", "error": "not one of the 2 classes",
                               "rows": [0], "count": 1}], (key, errors)
    errors = _refusal(contract, contract.models.MonitorRequest, {"rows": rows, "truth": ["1"]})
    assert errors == [{"column": "truth_col",
                       "error": "1 truths for 3 rows; give one per row, or none"}], errors
    # inside a row, the truth column is held to the same rule and never reaches the recipe
    for cell in ("maybe", {"nested": [1, 2]}):
        errors = _refusal(contract, contract.models.PredictRequest,
                          {"rows": [dict(rows[0], truth_col=cell)]})
        assert errors == [{"column": "truth_col", "error": "not one of the 2 classes",
                           "rows": [0], "count": 1}], errors
    req = contract.models.PredictRequest.model_validate(
        {"rows": [dict(rows[0], truth_col="yes", row_key="r-1")]})
    assert list(contract.tables.frame(req.rows).columns) == ["amount", "code", "region"]


def test_a_number_truth_is_a_finite_number():
    contract = contract_for(cases()["regression"])
    rows = [{"amount": 1.0, "code": "7"}] * 3
    req = contract.models.MonitorRequest.model_validate({"rows": rows, "truth": [1, 2.5, None]})
    assert req.truth == [1, 2.5, None]
    for truth in (["3.5", 1, 2], [True, 1, 2], [math.inf, 1, 2], [math.nan, 1, 2]):
        errors = _refusal(contract, contract.models.MonitorRequest, {"rows": rows,
                                                                     "truth": truth})
        assert errors == [{"column": "truth_col", "error": "expects a number, or null",
                           "rows": [0], "count": 1}], errors
    errors = _refusal(contract, contract.models.PredictRequest,
                      {"rows": [dict(rows[0], truth_col="high")]})
    assert errors[0]["column"] == "truth_col" and errors[0]["error"] == \
        "expects a number, or null", errors
    assert _table_refusal(contract, "amount,code,truth_col\n1,7,inf\n") == [
        {"column": "truth_col", "error": "expects a number, or null", "rows": [0], "count": 1}]


def test_a_labels_truth_is_label_names_split_on_sep():
    rec = cases()["multilabel"]
    contract = contract_for(rec)
    rows = [{"amount": 1.0, "code": "7"}] * 2
    req = contract.models.MonitorRequest.model_validate(
        {"rows": rows, "truth": [["red", "blue"], []]})
    assert req.truth == [["red", "blue"], []]
    refusal = "expects label names out of the 3 labels: a list, or one text split on '|'"
    for truth in ([["purple"], []], ["red|blue", []]):          # a request's truth is a list
        errors = _refusal(contract, contract.models.MonitorRequest, {"rows": rows,
                                                                     "truth": truth})
        assert errors == [{"column": "truth_col", "error": refusal, "rows": [0],
                           "count": 1}], errors
    # inside a row it is the shaped column's form: one text split on sep
    contract.models.PredictRequest.model_validate({"rows": [dict(rows[0], truth_col="red|blue"),
                                                            dict(rows[0], truth_col="")]})
    errors = _refusal(contract, contract.models.PredictRequest,
                      {"rows": [dict(rows[0], truth_col="red|purple")]})
    assert errors == [{"column": "truth_col", "error": refusal, "rows": [0], "count": 1}], errors
    # and in a CSV upload, split on sep
    table = _read_csv("amount,code,truth_col\n1.5,7,red|blue\n2,8,\n3,9,green\n")
    frame, truth = contract.tables.accept(table)
    assert truth == [["red", "blue"], [], ["green"]], truth
    assert "truth_col" not in frame.columns
    answer = contract.answers.table(frame_for(rec, n=3), truth)
    assert answer["true_truth_col"].tolist() == ["red|blue", "", "green"]
    errors = _table_refusal(contract, "amount,code,truth_col\n1,7,red\n2,8,red|mauve\n")
    assert errors == [{"column": "truth_col", "error": refusal, "rows": [1], "count": 1}], errors


def test_a_csv_truth_is_held_to_its_type():
    contract = contract_for(cases()["binary"])
    frame, truth = contract.tables.accept(_read_csv("amount,code,truth_col\n1,7,yes\n2,8,0\n"))
    assert truth == ["1", "0"], truth
    errors = _table_refusal(contract, "amount,code,truth_col\n1,7,maybe\n2,8,1\n")
    assert errors == [{"column": "truth_col", "error": "not one of the 2 classes", "rows": [0],
                       "count": 1}], errors
    numbers = contract_for(cases()["regression"])
    _, truth = numbers.tables.accept(_read_csv("amount,code,truth_col\n1,7,3.5\n2,8,\n"))
    assert truth == [3.5, None], truth
    errors = _table_refusal(numbers, "amount,code,truth_col\n1,7,high\n")
    assert errors[0]["error"] == "expects a number, or null", errors


def test_the_truth_never_reaches_the_recipe():
    """A truth column the schema does not list among `ignored` is still the truth: typed in a
    row, taken from a table, and dropped from the frame the recipe reads."""
    rec = dict(cases()["regression"], truth={"column": "target", "type": "number", "sep": None})
    contract = contract_for(rec)
    rows = [{"amount": 1.0, "code": "7", "target": 2.5}]
    req = contract.models.PredictRequest.model_validate({"rows": rows})
    assert list(contract.tables.frame(req.rows).columns) == ["amount", "code", "region"]
    frame, truth = contract.tables.accept(_read_csv("amount,code,target\n1,7,2.5\n"))
    assert truth == [2.5] and "target" not in frame.columns, (truth, list(frame.columns))
    errors = _refusal(contract, contract.models.PredictRequest,
                      {"rows": [dict(rows[0], target="high")]})
    assert errors[0]["column"] == "target", errors
    # an autoregressive target is an input channel and the truth: it reaches the recipe
    schema = dict(SCHEMA, input=SCHEMA["input"] + [{"name": "target", "dtype": "float",
                                                    "required": True,
                                                    "role": "autoregressive"}])
    forecast = dict(cases()["forecast"], truth=rec["truth"])
    auto = contract_for(forecast, schema=schema, sample={"shape": "window"})
    req = auto.models.PredictRequest.model_validate({"rows": rows, "history": rows})
    assert "target" in auto.tables.frame(req.history).columns
    # a table carries it as an input: it is not the truth of answers that forecast later steps
    frame, truth = auto.tables.accept(_read_csv("amount,code,target\n1,7,2.5\n2,8,3.5\n"))
    assert truth is None and frame["target"].tolist() == [2.5, 3.5], (truth, frame)
    assert "true_target" not in auto.answers.table(frame_for(forecast, n=2), truth).columns
    errors = _refusal(auto, auto.models.PredictRequest, {"rows": [dict(rows[0])],
                                                         "history": [{"amount": 1.0,
                                                                      "code": "7"}]})
    assert errors == [{"column": "target", "error": "required column missing", "rows": [0],
                       "count": 1, "field": "history"}], errors


def test_a_bundle_without_truth_takes_none():
    rec = cases()["reconstruction"]
    assert rec["truth"] is None
    contract = contract_for(rec)
    rows = [{"amount": 1.0, "code": "7"}]
    errors = _refusal(contract, contract.models.MonitorRequest, {"rows": rows, "truth": [1.0]})
    assert errors == [{"column": None,
                       "error": "truth: not a field this bundle's request takes"}], errors
    contract.models.MonitorRequest.model_validate({"rows": rows})
    assert contract.answers.columns(truth=True)[-1] == "warnings"
    assert not any(c.startswith("true_") for c in contract.answers.columns(truth=True))


# --------------------------------------------------------------------------- 4. the CSV answer

def test_csv_answer_columns_are_the_output_fields():
    for name, rec in cases().items():
        answers = contract_for(rec).answers
        assert answers.columns(EVERY_INCLUDE) == rec["fields"], name
        gated = {f for f in rec["fields"] if rec["field_kinds"][f]["kind"] in
                 ("raw", "recon", "embedding")}
        assert answers.columns() == [f for f in rec["fields"] if f not in gated], name
        table = answers.table(frame_for(rec), include=EVERY_INCLUDE)
        assert list(table.columns) == rec["fields"], name
    multiclass = contract_for(cases()["multiclass"]).answers
    assert multiclass.columns()[0] == "label"                  # v2.2 readers stay valid
    binary = cases()["binary"]
    answers = contract_for(binary).answers
    assert answers.columns() == ["label", "p_0", "p_1", "prediction", "warnings"]
    assert answers.columns(("logits",)) == ["label", "p_0", "p_1", "score", "prediction",
                                            "warnings"]
    assert answers.columns(truth=True) == ["label", "p_0", "p_1", "prediction", "true_truth_col",
                                           "warnings"]
    window = contract_for(cases()["window-multiclass"], shape="window").answers
    assert window.columns(truth=True)[-3:] == ["true_truth_col", "answered", "warnings"]


def test_csv_answer_values_are_the_frames():
    rec = cases()["binary"]
    contract = contract_for(rec)
    frame = frame_for(rec)
    table = contract.answers.table(frame, ["1", "0", "1", "0", "1", "0"], ("logits",))
    for column in ("label", "p_0", "p_1", "score", "prediction"):
        assert table[column].tolist() == frame[column].tolist(), column
    assert table["true_truth_col"].tolist() == ["1", "0", "1", "0", "1", "0"]
    assert table["warnings"].tolist()[0] == "first note; second note"
    text = table.to_csv(index=False)
    assert text.startswith("label,p_0,p_1,score,prediction,true_truth_col,warnings\n"), text


def test_a_csv_upload_gives_the_json_bodys_frame():
    contract = contract_for(cases()["multiclass"])
    rows = [{"amount": 1.5, "code": 80, "region": "north"},
            {"amount": "Infinity", "code": "443", "region": None},
            {"amount": None, "code": "7", "region": "south"}]
    by_json = contract.tables.frame(contract.models.PredictRequest.model_validate(
        {"rows": rows}).rows)
    by_csv, truth = contract.tables.accept(_read_csv(
        "amount,code,region,row_key\n1.5,80,north,a\nInfinity,443,,b\n,7,south,c\n"))
    assert truth is None
    pd.testing.assert_frame_equal(by_csv.reset_index(drop=True), by_json, check_dtype=False)
    assert by_json["amount"].tolist()[1] == math.inf and by_json["code"].tolist() == \
        ["80", "443", "7"]
    back = contract.tables.json_rows(by_json)
    assert back[1] == {"amount": "Infinity", "code": "443", "region": None}, back
    again = contract.tables.frame(contract.models.PredictRequest.model_validate(
        {"rows": back}).rows)
    pd.testing.assert_frame_equal(again, by_json)


def test_a_csv_upload_is_held_to_the_row_model():
    contract = contract_for(cases()["multiclass"], max_rows=3)
    uploads = {
        "amount,code\nlots,7\n2,80.0\n3,-7\n": [
            {"column": "amount", "error": "expects a number, null, or one of the tokens "
             "Infinity, -Infinity, NaN", "rows": [0], "count": 1},
            {"column": "code", "error": "expects a code: a non-negative integer, or its decimal "
             "digits as text", "rows": [1, 2], "count": 2}],
        "amount,surprise\n1,2\n": [{"column": "code", "error": "required column missing"},
                                   {"column": "surprise", "error": "unknown column"}],
        "amount,code\n1,7\n1,7\n1,7\n1,7\n": [{"column": None, "error": "more than 3 rows; at "
                                                                   "most 3 per request"}],
    }
    for text, want in uploads.items():
        got = _table_refusal(contract, text)
        assert got == want, f"{text!r}: {got}"


def test_json_rows_refuse_what_the_row_model_refuses():
    contract = contract_for(cases()["multiclass"])
    body = {"rows": [{"amount": True, "code": "80.0"}, {"amount": 1, "code": -7, "extra": 1},
                     {"code": "7"}]}
    errors = _refusal(contract, contract.models.PredictRequest, body)
    by_column = {(e["column"], e["error"]): e.get("rows") for e in errors}
    assert by_column == {
        ("amount", "expects a number, null, or one of the tokens Infinity, -Infinity, NaN"): [0],
        ("code", "expects a code: a non-negative integer, or its decimal digits as text"): [0, 1],
        ("extra", "unknown column"): [1], ("amount", "required column missing"): [2]}, errors


# --------------------------------------------------------------------------- 5. the records

def test_records_the_codec_cannot_read_are_refused():
    v1 = {"kind": "multiclass", "classes": ["a", "b"], "fields": ["label", "warnings"]}
    unread = (
        (v1, "vega/output@1"),
        ({k: v for k, v in cases()["regression"].items() if k != "reading"}, "vega/output@1"),
        (dict(cases()["regression"], field_kinds={"value": {"kind": "odd"},
                                                  "warnings": {"kind": "warnings"}}),
         "'odd', not a kind the answer codec reads"),
        (dict(cases()["regression"], field_kinds={"value": {"kind": "value"}}),
         "field_kinds['warnings'] is None, not a kind the answer codec reads"))
    for output, words in unread:
        try:
            AnswerCodec(output)
        except ValueError as exc:
            assert words in str(exc), str(exc)
        else:
            raise AssertionError(f"accepted {output}")
    for truth, words in (({"column": "t", "type": "labels", "sep": None}, "without a sep"),
                         ({"column": "t", "type": "vector", "sep": None}, "is not one of")):
        try:
            contract_for(dict(cases()["multilabel"], truth=truth))
        except ValueError as exc:
            assert words in str(exc), str(exc)
        else:
            raise AssertionError(f"the truth {truth} was accepted")


def test_the_contract_is_the_bundles_and_immutable():
    rec = cases()["ordinal"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        schema_bytes = json.dumps(SCHEMA).encode("utf-8")
        (root / "schema.json").write_bytes(schema_bytes)
        digest = "sha256:" + hashlib.sha256(schema_bytes).hexdigest()
        manifest = {"input_schema": "schema.json", "files": {"schema.json": digest},
                    "output": rec, "sample": {"shape": "row"},
                    "chain": {"shape": {"manifest_digest": "sha256:" + "c" * 64}}}
        (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        contract = ApiContract.from_bundle(root, max_rows=20)
        assert contract.version == digest and contract.max_rows == 20
        assert contract.envelope("m", 2, "v1") == {
            "api_version": "v1", "model_version": "m", "contract_version": digest,
            "feature_contract_version": "sha256:" + "c" * 64, "n_rows": 2,
            "output_kind": "ordinal"}
        described = contract.describe()
        assert described["output"] == rec and described["sample"] == {"shape": "row"}
        assert described["truth"] == rec["truth"] and described["classes"] == rec["classes"]
        assert described["label"] == {"column": "truth_col", "decode": {"yes": "1", "no": "0"}}
        assert described["input"] == SCHEMA["input"]
        assert described["required"] == ["amount", "code"]
        assert json.loads(json.dumps(described)) == described
        _nothing_handed_out_changes_it(contract)
        refused = (({"schema.json": "sha256:" + "0" * 64},
                    "the input contract is not the one this bundle was packed with",
                    "a schema.json the manifest pinned otherwise"),
                   ({}, "MANIFEST.json does not pin schema.json",
                    "a schema.json the manifest did not pin"),
                   (None, "MANIFEST.json does not pin schema.json", "a manifest without files"))
        for files, words, what in refused:
            (root / "MANIFEST.json").write_text(json.dumps(dict(manifest, files=files)),
                                                encoding="utf-8")
            try:
                ApiContract.from_bundle(root)
            except ValueError as exc:
                assert words in str(exc), str(exc)
            else:
                raise AssertionError(f"{what} was accepted")


def _nothing_handed_out_changes_it(contract) -> None:
    described = contract.describe()
    described["input"][0]["required"] = "MUTATED"
    described["output"]["fields"].append("MUTATED")
    after = contract.describe()
    assert after["input"][0]["required"] is True and "MUTATED" not in after["output"]["fields"]
    for held, key in ((contract.models.columns.inputs[0], "required"),
                      (contract.schema["input"][0], "required"),
                      (contract.manifest["output"], "reading"),
                      (contract.models.truth.decode, "maybe"), (contract.answers.truth, "column")):
        try:
            operator.setitem(held, key, "changed")
        except TypeError:
            continue
        raise AssertionError(f"a record the contract handed out could be changed ({key})")
    for name, value in (("max_rows", 5), ("answers", None), ("models", None)):
        try:
            setattr(contract, name, value)
        except AttributeError:
            continue
        raise AssertionError(f"the contract accepted a new {name} after start-up")


def test_monitor_readings_as_json():
    class Origin:                                       # a ReadingProvenance's fields
        check, concern, title, window_rows, note = "psi", None, "amount drift", 40, None

    class Reading:                                      # a DriftReading's public surface
        def __init__(self, value):
            self.column, self.measure, self.value, self.verdict = "amount", "psi", value, "ok"
            self.window = Origin()

    out = ApiContract.readings([Reading(0.1), Reading(float("inf")), Reading(None)])
    assert out[0]["value"] == 0.1 and out[1]["value"] is None and out[2]["value"] is None
    assert out[0]["window_rows"] == 40 and out[0]["title"] == "amount drift"
    contract = contract_for(cases()["regression"])
    body = dict(contract.envelope("m", 3, "v1"), moved=0, readings=out)
    contract.MonitorResponse.model_validate(body)
    contract.Reading.model_validate(out[0])


def test_problem_bodies():
    problem = problems_mod.ProblemDetail(422, "Request does not honour the input schema", "x",
                                         ({"column": "amount", "error": "e", "rows": [1]},))
    assert problem.body() == {"type": "about:blank", "title": "Request does not honour the "
                              "input schema", "status": 422, "detail": "x",
                              "errors": [{"column": "amount", "error": "e", "rows": [1]}]}
    assert "errors" not in problems_mod.ProblemDetail(503, "Service not ready", "y").body()
    error = ContractError("refused", [{"column": "c", "error": "e", "rows": [0, 2]}])
    try:
        error.errors[0]["rows"] = []
    except TypeError:
        pass
    else:
        raise AssertionError("a ContractError's errors could be changed")
    assert thawed(error.errors) == [{"column": "c", "error": "e", "rows": [0, 2]}]


# --------------------------------------------------------------------------- 6. contract and gates

def test_the_codec_reads_the_pack_catalogs_field_kinds():
    kinds = CATALOG["field_kinds"]
    assert set(answers_mod.FIELD_KEYS) == set(kinds), \
        f"codec kinds {sorted(answers_mod.FIELD_KEYS)} != catalog {sorted(kinds)}"
    for kind, row in kinds.items():
        assert answers_mod.FIELD_KEYS[kind] == row["key"], (kind, row)
        gate = row["shown"].split("=", 1)[1] if row["shown"].startswith("include=") else None
        assert answers_mod.INCLUDES.get(kind) == gate, (kind, row["shown"])
    assert sorted(answers_mod.INCLUDES.values()) == sorted(CATALOG["include_values"])
    assert tuple(CATALOG["output_readings"]["truth_types"]) == rows_mod.TRUTH_TYPES
    readings = CATALOG["output_readings"]["readings"]
    assert set(ALWAYS) == set(readings), "the oracle names every reading of the catalog"


def test_the_five_files_meet_c11_to_c13_and_their_budgets():
    for name in FILES:
        path = SERVE / name
        text = path.read_text(encoding="utf-8")
        findings = conformance.size_findings(ast.parse(text), f"serve/{name}")
        assert not findings, "\n".join(f.message() for f in findings)
        lines = len(text.splitlines())
        assert lines <= BUDGETS[name], f"serve/{name} is {lines} lines > D2 §5's {BUDGETS[name]}"
    proc = subprocess.run([sys.executable, str(SCRIPTS / "conformance.py"), "--size-only",
                           *[str(SERVE / n) for n in FILES]], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not re.search(r"\bC1[123]\b", proc.stdout.replace("C11–C13", "")), proc.stdout
    assert f"C11–C13 hold on all {len(FILES)} file(s) measured" in proc.stdout, proc.stdout


def test_the_literal_gate_finds_nothing_in_the_five_files():
    """R-F10: `lint_templates.py` (T04, the static tier) is the one deny-list gate. It scans
    every shared template; here its hits are read for these five files, when it is present."""
    if not LINT.is_file():
        print(f"        not run: {LINT.relative_to(SKILLS)} (T04) is not on this tree")
        return
    proc = subprocess.run([sys.executable, str(LINT)], capture_output=True, text=True,
                          cwd=str(LINT.parent))
    assert "Traceback" not in proc.stderr, proc.stderr
    hits = [line for line in proc.stdout.splitlines()
            if any(f"serving/serve/{name}:" in line for name in FILES)]
    assert not hits, "\n".join(hits)


# --------------------------------------------------------------------------- the runner

def main() -> int:
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:                                  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        " + "\n        ".join(
                f"{type(exc).__name__}: {exc}".splitlines()[:14]))
    print(f"\n{len(tests) - len(failed)} of {len(tests)} serve codec test(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
