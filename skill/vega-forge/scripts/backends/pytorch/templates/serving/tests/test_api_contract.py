"""
tests/test_api_contract.py — the service's HTTP contract, whatever this bundle answers.

    python3 -m pytest tests/test_api_contract.py

What every served bundle promises, over the in-process test client, with every column and value
read from the bundle's own records (tests/_oracle.py), never named here: the service becomes
ready only after reproducing its stored examples; a request that breaks the input schema is
refused whole with 422 and a problem body naming every offending column as the caller wrote it,
never the branch of a type union it failed last; values training never showed are answered,
with warnings; the tokens JSON has no literal for (Infinity, NaN) are decoded; a code column
sent as a number or as its digits is one category; a CSV upload is held to the same row model
as a JSON body and answered exactly as its rows would be; every answer carries the model, the
input-contract and the feature-contract versions, named apart; /v1/schema describes the whole
output and sample records; MANIFEST.serve.endpoints lists exactly the routes the service
registers; the code /v1/code shows is the code that runs; the contract cannot be changed through
what it hands out; a defect, a wrong path or a wrong method answers a problem body, not a trace.
What the answers hold is test_api_answers.py's; how a shape reads a request, test_api_shapes.py's.
"""
import hashlib
import io

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient                  # noqa: E402

from _oracle import ORACLE, ROOT, applies, client          # noqa: E402,F401  (a fixture)
from serve.app import create_app                           # noqa: E402

MANIFEST = ORACLE.manifest
SCHEMA_FILE = MANIFEST.get("input_schema", "schema.json")
CONTRACT_VERSION = "sha256:" + hashlib.sha256((ROOT / SCHEMA_FILE).read_bytes()).hexdigest()
FEATURE_VERSION = MANIFEST["chain"]["shape"]["manifest_digest"]
TOL = ORACLE.tolerance
NUMERIC = ORACLE.numeric()
RANGED = ORACLE.ranged()
CODES = ORACLE.codes()
NON_FINITE = ORACLE.non_finite()
PROBLEM = "application/problem+json"
UNDOCUMENTED = ("/openapi.json",)                # FastAPI's own schema route, not the API's
SMALL = 3                                        # the row cap of the second service


@pytest.fixture(scope="module")
def small():
    """The same bundle served with a row cap of SMALL."""
    with TestClient(create_app(ROOT, max_rows=SMALL)) as c:
        yield c


def _table(name="rows.csv"):
    text = {c["name"]: "string" for c in ORACLE.inputs if c.get("dtype") == "string"}
    return pd.read_csv(ROOT / "examples" / name, dtype=text)


def _upload(client, frame_or_bytes, query="", headers=None):
    data = (frame_or_bytes if isinstance(frame_or_bytes, bytes)
            else frame_or_bytes.to_csv(index=False).encode("utf-8"))
    return client.post("/v1/predict/csv" + query, headers=headers or {},
                       files={"file": ("t.csv", data, "text/csv")})


def _one(**changes) -> dict:
    """examples/request.json with its first row changed: a request whose first answer is always
    given (a row; a window's last row with its history; the rows of one graph)."""
    body = ORACLE.one_request()
    body["rows"] = [dict(r) for r in body["rows"]]
    body["rows"][0].update(changes)
    return body


def _include(values) -> str:
    return "?" + "&".join(f"include={v}" for v in values) if values else ""


def test_ready_only_after_the_self_test(client):
    health = client.get("/health").json()
    assert health["ready"] is True, health
    test = health["self_test"]
    assert test["passed"] and test["max_abs_diff"] <= TOL and not test["unanswered"], test
    assert test["rows"] == len(ORACLE.compared_positions()), test


def test_missing_and_unknown_columns_are_refused_by_name(client):
    body = _one(**{"Not A Column": 1})
    body["rows"][0].pop(ORACLE.required())
    res = client.post("/v1/predict", json=body)
    assert res.status_code == 422 and res.headers["content-type"].startswith(PROBLEM)
    columns = {e["column"] for e in res.json()["errors"]}
    assert {ORACLE.required(), "Not A Column"} <= columns, res.json()


def test_empty_and_oversized_batches_are_refused(client, small):
    assert client.post("/v1/predict", json={"rows": []}).status_code == 422
    rows = ORACLE.one_request()["rows"] * (SMALL + 1)
    assert small.post("/v1/predict", json={"rows": rows[:SMALL + 1]}).status_code == 422


@applies(RANGED is not None)
def test_out_of_range_values_are_answered_with_warnings(client):
    low, high = (float(v) for v in RANGED["range_seen"])
    beyond = high + 10 * max(1.0, high - low, abs(high))
    res = client.post("/v1/predict", json=_one(**{RANGED["name"]: beyond}))
    assert res.status_code == 200, res.text
    assert any(RANGED["name"] in w for w in res.json()["predictions"][0]["warnings"]), res.json()


def test_infinity_tokens_are_decoded(client):
    if not NON_FINITE:
        pytest.skip("no column carried non-finite values in training")
    res = client.post("/v1/predict", json=_one(**dict(zip(NON_FINITE, ("Infinity", "NaN")))))
    assert res.status_code == 200, res.text


def test_code_columns_accept_number_or_text(client):
    if not CODES:
        pytest.skip("no digit-code column in this schema")
    code = str(ORACLE.one_request()["rows"][0][CODES[0]])
    by_number = client.post("/v1/predict", json=_one(**{CODES[0]: int(code)}))
    by_text = client.post("/v1/predict", json=_one(**{CODES[0]: code}))
    assert by_number.status_code == by_text.status_code == 200, by_number.text
    assert by_number.json()["predictions"] == by_text.json()["predictions"]


def test_csv_upload_and_csv_answer(client):
    table, data = _table(), (ROOT / "examples" / "rows.csv").read_bytes()
    res = client.post("/v1/predict/csv", files={"file": ("rows.csv", data, "text/csv")})
    assert res.status_code == 200, res.text
    assert res.json()["n_rows"] == MANIFEST["examples"]["rows"] == len(table)
    res = client.post("/v1/predict/csv", files={"file": ("rows.csv", data, "text/csv")},
                      headers={"accept": "text/csv"})
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/csv")
    truth = ORACLE.truth_column in table.columns and not ORACLE.truth_is_input()
    head = list(pd.read_csv(io.StringIO(res.text), nrows=0).columns)
    assert head == ORACLE.csv_columns(truth=truth), head
    assert len(pd.read_csv(io.StringIO(res.text))) == len(table)


def test_the_manifest_lists_every_route(client):
    registered = {f"{method} {route.path}" for route in client.app.routes
                  for method in (getattr(route, "methods", None) or ())
                  if method != "HEAD" and route.path not in UNDOCUMENTED}
    assert registered == set(MANIFEST["serve"]["endpoints"]), (
        sorted(registered ^ set(MANIFEST["serve"]["endpoints"])))


def test_schema_describes_output_and_sample(client):
    described = client.get("/v1/schema").json()
    assert described["version"] == CONTRACT_VERSION
    assert described["feature_contract_version"] == FEATURE_VERSION
    assert described["output"] == MANIFEST["output"]
    assert described["sample"] == (MANIFEST.get("sample") or {"shape": "row"})
    assert described["truth"] == MANIFEST["output"].get("truth")
    assert described["classes"] == list(ORACLE.classes)
    assert [c["name"] for c in described["input"]] == list(ORACLE.input_names())
    for key in ("threshold", "threshold_scale", "threshold_per_label", "fields", "field_kinds"):
        assert key in described["output"], key
    assert described["tokens"] and "Infinity" in described["tokens"]


def test_alias_metadata_and_page(client):
    assert client.post("/predict", json=ORACLE.one_request()).status_code == 200
    info = client.get("/v1/info").json()
    assert "files" not in info and "roles" not in info and info["api_version"] == "v1"
    assert "/" not in str(info.get("campaign")), info.get("campaign")  # an id, not a host path
    assert info["model_version"] == MANIFEST["model_version"]
    assert info["contract_version"] == CONTRACT_VERSION
    assert len(client.get("/v1/schema").json()["input"]) == len(ORACLE.inputs)
    for path in ("/", "/console"):
        page = client.get(path)
        assert page.status_code == 200 and "text/html" in page.headers["content-type"], path
        assert "no-cache" in page.headers.get("cache-control", ""), path  # revalidated per load


@applies(NUMERIC is not None)
def test_type_errors_name_the_column_not_the_union_branch(client):
    body = ORACLE.one_request()
    rows = [dict(r) for r in body["rows"]]
    rows = rows if len(rows) > 1 else [rows[0], dict(rows[0])]
    rows[1][NUMERIC] = "abc"
    res = client.post("/v1/predict", json=dict(body, rows=rows))
    assert res.status_code == 422
    errors = res.json()["errors"]
    assert [e["column"] for e in errors] == [NUMERIC] and errors[0]["rows"] == [1], errors


@applies(NUMERIC is not None)
def test_booleans_are_not_numbers(client):
    res = client.post("/v1/predict", json=_one(**{NUMERIC: True}))
    assert res.status_code == 422 and res.json()["errors"][0]["column"] == NUMERIC


def test_codes_are_refused_unless_they_look_like_codes(client):
    if not CODES:
        pytest.skip("no code column with a pattern in this schema")
    for bad in ("80.0", "abc", -7, 80.5, True):
        res = client.post("/v1/predict", json=_one(**{CODES[0]: bad}))
        assert res.status_code == 422, (bad, res.text)
        assert [e["column"] for e in res.json()["errors"]] == [CODES[0]], bad


def test_csv_is_held_to_the_same_row_model(client, small):
    table = _table()
    at = min(3, len(table) - 1)
    if NUMERIC is not None:
        bad = table.copy()
        bad[NUMERIC] = bad[NUMERIC].astype(object)
        bad.loc[at, NUMERIC] = "abc"
        res = _upload(client, bad)
        assert res.status_code == 422, res.text
        errors = res.json()["errors"]
        assert [e["column"] for e in errors] == [NUMERIC] and errors[0]["rows"] == [at], errors
    if CODES:
        bad = table.copy()
        bad[CODES[0]] = bad[CODES[0]].astype(object)
        bad.loc[0, CODES[0]] = "80.0"
        res = _upload(client, bad)
        assert res.status_code == 422 and res.json()["errors"][0]["column"] == CODES[0]
    for junk in (b"", b"\x00\x01\xff\xfe garbage", table.head(0).to_csv(index=False).encode()):
        res = _upload(client, junk)
        assert res.status_code == 422, (junk[:20], res.text)
        assert res.headers["content-type"].startswith(PROBLEM)
    assert _upload(small, table.head(SMALL + 1)).status_code == 422
    assert _upload(small, table.head(SMALL)).status_code == 200


def test_every_answer_carries_the_versions(client):
    assert MANIFEST["files"][SCHEMA_FILE] == CONTRACT_VERSION
    bodies = [client.post("/v1/predict", json=ORACLE.one_request()).json()]
    if MANIFEST.get("monitor"):
        bodies.append(client.post("/v1/monitor", json=ORACLE.one_request()).json())
    for body in bodies:
        assert body["contract_version"] == CONTRACT_VERSION, body
        assert body["feature_contract_version"] == FEATURE_VERSION
        assert body["model_version"] == MANIFEST["model_version"]
        assert body["output_kind"] == ORACLE.reading
    res = _upload(client, _table().head(5), headers={"accept": "text/csv"})
    assert res.status_code == 200
    assert res.headers["x-model-version"] == MANIFEST["model_version"]
    assert res.headers["x-contract-version"] == CONTRACT_VERSION
    assert res.headers["x-feature-contract-version"] == FEATURE_VERSION
    health = client.get("/health").json()
    assert health["contract_version"] == CONTRACT_VERSION
    assert health["feature_contract_version"] == FEATURE_VERSION


def test_the_two_contract_versions_are_named_apart(client):
    body = client.post("/v1/predict", json=ORACLE.one_request()).json()
    assert body["feature_contract_version"] == FEATURE_VERSION != body["contract_version"]


def test_a_defect_answers_a_problem_not_a_trace():
    app = create_app(ROOT)

    @app.get("/boom")
    def boom():
        raise RuntimeError("an internal detail")

    with TestClient(app, raise_server_exceptions=False) as c:
        res = c.get("/boom")
    assert res.status_code == 500 and res.headers["content-type"].startswith(PROBLEM)
    assert "internal detail" not in res.text


def test_the_contract_cannot_be_changed_through_what_it_hands_out(client):
    contract = client.app.state.contract
    described = contract.describe()
    described["input"][0]["required"] = "MUTATED"
    described["output"]["fields"].append("MUTATED")
    again = contract.describe()
    assert again["input"][0]["required"] != "MUTATED"
    assert "MUTATED" not in again["output"]["fields"]
    with pytest.raises(TypeError):
        contract.models.columns.inputs[0]["required"] = False
    with pytest.raises(TypeError):
        contract.manifest["output"]["reading"] = "MUTATED"
    with pytest.raises(AttributeError):
        contract.max_rows = 1


def test_wrong_paths_and_methods_answer_problem_bodies(client):
    for res in (client.get("/v1/nothing"), client.get("/v1/predict")):
        assert res.status_code in (404, 405)
        assert res.headers["content-type"].startswith(PROBLEM), res.text
        assert res.json()["status"] == res.status_code


def test_a_csv_is_answered_exactly_as_its_rows_in_json(client):
    name = "demo.csv" if (ROOT / "examples" / "demo.csv").is_file() else "rows.csv"
    by_csv = _upload(client, _table(name), _include((*ORACLE.includes(), "rows")))
    assert by_csv.status_code == 200, by_csv.text
    by_json = client.post("/v1/predict" + _include(ORACLE.includes()),
                          json={"rows": by_csv.json()["rows"]})
    assert by_json.status_code == 200, by_json.text
    assert by_csv.json()["predictions"] == by_json.json()["predictions"]


def test_the_code_shown_is_the_code_that_runs(client):
    body = client.get("/v1/code").json()
    assert body["schema"] == "vega/code-view@2" and body["verified"] is True and body["files"]
    for shown in body["files"]:
        digest = "sha256:" + hashlib.sha256(shown["text"].encode("utf-8")).hexdigest()
        assert shown["digest"] == MANIFEST["files"][shown["path"]] == digest, shown["path"]
        assert shown["kind"] in ("adapter", "support", "entrypoint"), shown["path"]
        assert shown["lines"] == len(shown["text"].splitlines()), shown["path"]
    assert body["files"][0]["role"] == "Model"
