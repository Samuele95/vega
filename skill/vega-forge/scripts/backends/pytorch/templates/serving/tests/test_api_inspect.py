"""
tests/test_api_inspect.py — the inspection routes (serve/routes_inspect.py), in process.

    python3 -m pytest tests/test_api_inspect.py

For any reading and sample shape: /v1/network describes every netspec.json block from bytes
built once (the digest pack pinned); tensors decode, a deferred one is fetched, an unknown id is
404; /v1/explain carries the row's /v1/predict answer, faithful within the round-trip tolerance,
its chain recomputing from the published weights, its attribution shaped by the sample; a request
it cannot explain is a 422 naming the field; switched off, the routes are 404 and the answers
unchanged; the weights switch only restricts; explaining is admitted and changes nothing; nothing
answers before the self-test; a --no-inspect bundle lists and serves none of the routes.
"""
import base64
import contextlib
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path
from unittest import mock
from urllib.parse import quote

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient      # noqa: E402

from serve.app import ROUTE_MODULES, create_app  # noqa: E402

MANIFEST = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
SPEC = json.loads((ROOT / "netspec.json").read_text(encoding="utf-8"))
ENV = MANIFEST["serve"]["identity"]["env"]
SWITCHES = (f"{ENV}_INSPECT", f"{ENV}_INSPECT_WEIGHTS")
SHAPE = (MANIFEST.get("sample") or {}).get("shape") or "row"
GRANT = (MANIFEST.get("serve") or {}).get("inspection") or {}
PINS = GRANT.get("network_digest")
FULL = (GRANT.get("weights") or "full") == "full"
SHIPPED = (ROOT / "serve" / "routes_inspect.py").is_file()
ENDPOINTS = (("GET", "/v1/network"), ("GET", "/v1/network/tensor/{tensor_id}"),
             ("POST", "/v1/explain"))
EXPLAINED = ("vega/network-view@2", "vega/explanation@2")
# a selector and the target kinds that admit it: one the task does not admit is a 422
SELECTORS = ((("step", 1), ("forecast_step",)), (("threshold", 0), ("threshold_logit",)),
             (("label", "no-such-label"), ("label_logit",)))
GATES = (("raw", "logits"), ("recon", "reconstruction"), ("embedding", "embedding"))
BROKEN = {"content": b"{ not json", "headers": {"content-type": "application/json"}}
shipped = pytest.mark.skipif(not SHIPPED, reason="packed with --no-inspect: no inspection routes")


@contextlib.contextmanager
def served(switches=None, lifespan=True):
    """This bundle's service, the inspection switches as given ({"INSPECT": "off"}), no other."""
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    env.update({f"{ENV}_{name}": value for name, value in (switches or {}).items()})
    with mock.patch.dict(os.environ, env, clear=True):
        app = create_app(ROOT)
        if not lifespan:              # the self-test never runs: the service stays unready
            yield TestClient(app)
            return
        with TestClient(app) as c:
            yield c


def _registered(app) -> list:
    return sorted({f"{m} {r.path}" for r in app.routes for m in getattr(r, "methods", ()) or ()
                   if m != "HEAD" and r.path != "/openapi.json"})


def _decode(payload) -> np.ndarray:
    """A tensor payload's values (json or base64-le), float32, in its shape."""
    shape = tuple(int(d) for d in payload["shape"])
    if payload["encoding"] == "json":
        return np.asarray(payload["data"], dtype=np.float32).reshape(shape)
    assert payload["encoding"] == "base64-le", payload["encoding"]
    return np.frombuffer(base64.b64decode(payload["data"]), dtype="<f4").reshape(shape)


def _tensor(client, net, tensor_id) -> np.ndarray:
    payload = net["tensors"][tensor_id]
    if payload["encoding"] == "deferred":
        payload = client.get(f"/v1/network/tensor/{quote(tensor_id)}").json()
    return _decode(payload)


def _problem(res, status: int) -> dict:
    assert res.status_code == status, res.text
    assert res.headers["content-type"].startswith("application/problem+json"), res.headers
    return res.json()


def _ok(res) -> dict:
    assert res.status_code == 200, res.text
    return res.json()


def _predicted(client, ask, query="") -> list:
    return _ok(client.post("/v1/predict" + query, json=ask["predict"]))["predictions"]


def _explain(client, body, query="") -> dict:
    return _ok(client.post("/v1/explain" + query, json=body))["explanation"]


@pytest.fixture(scope="module")
def client():
    with served() as c:
        yield c


@pytest.fixture(scope="module")
def net(client):
    return _ok(client.get("/v1/network"))


@pytest.fixture(scope="module")
def ask(client):
    """One answered example sample's explain request (a row alone; the last answered window
    anchor; a graph's first answered row) and the /v1/predict request of the same rows."""
    test = client.get("/v1/examples").json()["test"]
    rows, history = test["rows"], test.get("history")
    predict = {"rows": rows, **({"history": history} if history else {})}
    answers = client.post("/v1/predict", json=predict).json()["predictions"]
    answered = [i for i, a in enumerate(answers) if a["answered"]]
    assert answered, "no example row is answered"
    at = answered[-1] if SHAPE == "window" else answered[0]
    body, index = ({"row": rows[at]}, 0) if SHAPE == "row" else (dict(predict, at=at), at)
    if SHAPE == "row":              # the explained batch is the one row: so is the predicted one
        predict = {"rows": [rows[at]]}
    ask = {"body": body, "at": at, "rows": rows, "predict": predict, "index": index}
    return dict(ask, answer=_predicted(client, ask)[index])


@shipped
def test_the_module_declares_the_inspection_routes(client):
    ri = importlib.import_module("serve.routes_inspect")
    assert ri.ENDPOINTS == ENDPOINTS
    assert (ri.MODEL_ROUTES, ri.OPEN_WHILE_UNREADY, ri.CAPABILITY) == \
        ((("POST", "/v1/explain"),), (), "inspection")
    declared = {f"{method} {path}" for name in ROUTE_MODULES
                if (ROOT / "serve" / f"{name}.py").is_file()   # the modules this bundle ships
                for method, path in importlib.import_module(f"serve.{name}").ENDPOINTS}
    assert MANIFEST["serve"]["endpoints"] == sorted(declared) == _registered(client.app)


@shipped
def test_the_status_is_one_object_and_ready(client, net):
    status = client.get("/health").json()["inspection"]
    assert status == client.get("/v1/info").json()["inspection"]
    assert (status["enabled"], status["ready"], status["why"]) == (True, True, None)
    assert status["weights"] == net["weights"] == ("full" if FULL else "summary")
    assert status["pinned"] is bool(PINS)
    if FULL:
        assert status["network_digest"] == net["network_digest"]
    coverage, types = status["coverage"], sorted({n["type"] for n in net["nodes"]})
    assert (coverage["nodes"], coverage["types"]) == (len(net["nodes"]), types)
    assert {n["type"] for n in net["nodes"] if n["id"] in coverage["opaque"]} <= {"custom"}
    assert 1 <= status["fidelity"]["samples"] <= 16
    assert status["fidelity"]["max_abs_diff"] <= status["fidelity"]["tolerance"]
    assert isinstance(status["recompute"]["ok"], bool)


@shipped
def test_the_network_describes_every_spec_block(client, net):
    assert (net["schema"], net["available"]) == (EXPLAINED[0], True)
    assert net["model_version"] == MANIFEST["model_version"]
    blocks = [str(b.get("id") or b.get("type")) for b in SPEC["blocks"]]
    assert set(blocks) <= {n["block"] for n in net["nodes"]}
    top = [n for n in net["nodes"] if n["parent"] is None and n["call"] != "io"]
    assert len(top) == len(blocks)
    assert sum(n["params"]["count"] for n in top) == net["params"]["total"]
    features = [f["name"] for f in (SPEC.get("data_contract") or {}).get("features") or ()]
    named = net["inputs"]["numeric"] + [c["name"] for c in net["inputs"]["nominal"]]
    assert sorted(named) == sorted(features)


@shipped
def test_the_network_is_bytes_built_once_with_their_digest(client, net):
    first, again = client.get("/v1/network"), client.get("/v1/network")
    assert first.content == again.content and first.headers["content-type"].endswith("json")
    rest = {k: v for k, v in net.items() if k != "network_digest"}
    canonical = json.dumps(rest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert net["network_digest"] == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    if PINS and FULL:
        assert net["network_digest"] == (PINS["full"] if isinstance(PINS, dict) else PINS)


@shipped
def test_the_summary_view_withholds_every_tensor(client, net):
    body = _ok(client.get("/v1/network?weights=summary"))
    assert body["weights"] == "summary" and set(body["tensors"]) == set(net["tensors"])
    for tensor_id, payload in body["tensors"].items():
        assert (payload["encoding"], payload["data"]) == ("withheld", None), tensor_id
        assert payload["summary"]["n"] == math.prod(payload["shape"]), tensor_id
    assert not isinstance(PINS, dict) or PINS.get("summary") in (None, body["network_digest"])
    refused = _problem(client.get("/v1/network?weights=everything"), 422)
    assert refused["errors"][0]["field"] == "weights"


@shipped
def test_the_tensors_decode_and_a_deferred_one_is_fetched(client, net):
    first = next(iter(net["tensors"]), "any.weight")
    fetched = client.get(f"/v1/network/tensor/{quote(first)}")
    if not FULL:                     # --inspect-weights summary: every tensor withheld
        assert {p["encoding"] for p in net["tensors"].values()} <= {"withheld"}
        assert "raw weights" in _problem(fetched, 404)["detail"]
        return
    for tensor_id, payload in net["tensors"].items():
        values = _tensor(client, net, tensor_id)
        assert values.shape == tuple(payload["shape"]), tensor_id
        assert values.size == payload["summary"]["n"], tensor_id
    assert _ok(fetched)["encoding"] == "base64-le"
    assert np.array_equal(_decode(fetched.json()), _tensor(client, net, first))
    missing = _problem(client.get("/v1/network/tensor/no-such-node.weight"), 404)
    assert "no-such-node.weight" in missing["detail"]


@shipped
def test_explain_agrees_with_predict(client, ask):
    doc = _ok(client.post("/v1/explain", json=ask["body"]))
    x, envelope = doc["explanation"], (doc["model_version"], doc["n_rows"], doc["output_kind"])
    assert envelope == (MANIFEST["model_version"], 1, MANIFEST["output"]["reading"])
    assert (x["schema"], x["sample"]["shape"]) == (EXPLAINED[1], SHAPE)
    assert x["sample"]["row"] == ask["at"]
    assert x["prediction"] == ask["answer"]
    assert x["warnings"] == ask["answer"]["warnings"]
    f = x["fidelity"]
    assert f["ok"] is True and f["head_max_abs_diff"] <= f["tolerance"]
    assert (f["served_max_abs_diff"] or 0.0) <= f["tolerance"]      # null: no served field
    assert x["target"]["why"] in ("predicted", "default")
    assert x["outputs"]["link"] == MANIFEST["output"]["link"]


@shipped
def test_the_prediction_shows_what_include_asks(client, ask):
    kinds = {v["kind"] for v in MANIFEST["output"]["field_kinds"].values()}
    gated = [include for kind, include in GATES if kind in kinds]
    query = "?" + "&".join(f"include={i}" for _, i in GATES)
    shown = _explain(client, ask["body"], query)["prediction"]
    assert shown == _predicted(client, ask, query)[ask["index"]]
    plain = _explain(client, ask["body"])["prediction"]
    assert (set(shown) > set(plain)) if gated else shown == plain   # nothing gated: none shown


@shipped
def test_the_explained_chain_recomputes_from_the_published_weights(client, net, ask):
    chain = _explain(client, ask["body"])["chain"]
    assert (chain is None) == (net["chain"] is None)     # no chain in the network: none here
    if chain is None or not FULL:    # nothing to recompute: no chain, or the weights withheld
        return
    assert len(chain["pre"]) == len(chain["post"]) == len(net["chain"]["layers"])
    sources = [chain["inputs"]] + chain["post"][:-1]
    for layer, src, pre in zip(net["chain"]["layers"], sources, chain["pre"]):
        w = _tensor(client, net, layer["weight"]).astype(np.float64)
        b = _tensor(client, net, layer["bias"]).astype(np.float64) if layer["bias"] else 0.0
        assert np.allclose(w @ np.asarray(src, dtype=np.float64) + b, pre, rtol=1e-3,
                           atol=1e-3), layer["node"]


@shipped
def test_the_attribution_is_shaped_by_the_sample(client, net, ask):
    x = _explain(client, ask["body"])
    a, order = x["attribution"], net["inputs"]["order"]
    assert (a["shape"], a["method"], a["completeness"]) == (SHAPE, "grad_x_input", None)
    assert x["inputs"]["order"] == order
    if SHAPE == "row":
        f = a["features"]
        assert len(f["names"]) == len(f["attribution"]) == len(f["values"]) == len(order)
    elif SHAPE == "window":
        steps, width = net["sample"]["length"], len(a["features"]["names"])
        assert (a["steps"], len(a["per_step"]), len(a["per_feature"])) == (steps, steps, width)
        assert isinstance(a["matrix"], dict) or np.shape(a["matrix"]) == (steps, width)
    else:
        assert a["level"] == net["sample"]["level"] and isinstance(a["hops"], dict)
        node = a["level"] == "node"
        assert (len(a["self"]) == len(order)) if node else a["self"] is None


@shipped
def test_integrated_gradients_and_the_exact_detail(client, ask):
    ig = _explain(client, dict(ask["body"], method="integrated_gradients", steps=8))
    assert ig["attribution"]["method"] == "integrated_gradients"
    completeness = ig["attribution"]["completeness"]
    assert set(completeness) == {"sum", "target_minus_baseline", "gap"} and completeness["gap"] >= 0
    exact = _explain(client, dict(ask["body"], detail="exact"))
    assert exact["detail"] == "exact" and isinstance(exact["degraded"], bool)
    levels = {node["level"] for node in exact["nodes"].values()}
    assert "exact" in levels and levels <= {"exact", "summary", "omitted"}
    for node_id, node in exact["nodes"].items():
        tensors = (node.get("exact") or {}).values() if node["level"] == "exact" else ()
        for tensor in (t for t in tensors if t is not None and t["dtype"] == "float32"):
            assert _decode(tensor).size == math.prod(tensor["shape"]), node_id


def _foreign_target(net) -> dict:
    """A selector this task's targets do not admit."""
    admitted = set(net["task"]["targets"])
    return next(dict([sel]) for sel, kinds in SELECTORS if not admitted & set(kinds))


@shipped
def test_explain_refusals_are_problems_naming_the_field(client, net, ask):
    one = ask["rows"][ask["at"]]
    refusals = [({}, "row" if SHAPE == "row" else "rows"),
                (dict(ask["body"], target=_foreign_target(net)), "target."),
                (dict(ask["body"], method="guess"), "method"),
                (dict(ask["body"], detail="everything"), "detail"),
                (dict(ask["body"], method="integrated_gradients", steps=2), "steps")]
    if SHAPE == "row":
        refusals.append(({"rows": [one, one]}, "rows"))
    elif SHAPE == "window" and net["sample"]["length"] > 1:
        refusals.append(({"rows": [one]}, "at"))
    elif SHAPE == "graph":
        refusals.append((dict(ask["body"], history=[one]), "history"))
    for body, field in refusals:
        problem = _problem(client.post("/v1/explain", json=body), 422)
        assert problem["errors"][0]["field"].startswith(field), (body, problem)
    extra = _problem(client.post("/v1/explain", json=dict(ask["body"], extra=1)), 422)
    assert "extra" in json.dumps(extra["errors"])
    unknown = {**one, "no_such_column": 1}
    wrong = {"row": unknown} if SHAPE == "row" else dict(ask["body"], rows=[unknown])
    column = _problem(client.post("/v1/explain", json=wrong), 422)
    assert [e["column"] for e in column["errors"]] == ["no_such_column"]


@shipped
def test_switched_off_the_routes_are_404_and_the_answers_unchanged(client, ask):
    with served({"INSPECT": "off"}) as off:
        health = off.get("/health").json()
        assert health["ready"] is True
        assert (health["inspection"]["enabled"], health["inspection"]["ready"]) == (False, False)
        for res in (off.get("/v1/network"), off.get("/v1/network/tensor/any.weight"),
                    off.post("/v1/explain", json=ask["body"]),
                    off.post("/v1/explain", json={"not": "a request"})):
            assert "switched off" in _problem(res, 404)["detail"]
        answers = _predicted(off, ask)
    assert answers == _predicted(client, ask)


@shipped
def test_the_weights_switch_only_restricts(ask):
    with served({"INSPECT_WEIGHTS": "summary"}) as c:
        status = c.get("/health").json()["inspection"]
        assert (status["ready"], status["weights"]) == (True, "summary")
        body = _ok(c.get("/v1/network"))
        assert body["weights"] == "summary" and all(
            p["encoding"] == "withheld" for p in body["tensors"].values())
        assert "raw weights" in _problem(c.get("/v1/network?weights=full"), 404)["detail"]
        some = quote(next(iter(body["tensors"]), "any.weight"))
        _problem(c.get(f"/v1/network/tensor/{some}"), 404)
        assert _explain(c, ask["body"])["fidelity"]["ok"] is True


@shipped
def test_explain_takes_a_place_at_the_model_and_the_network_does_not(client):
    places, taken = client.app.state.inflight, 0
    while places.acquire(blocking=False):
        taken += 1
    try:
        busy = client.post("/v1/explain", **BROKEN)
        assert _problem(busy, 503)["title"] == "Busy"      # a 503, not a 422: never read
        assert client.get("/v1/network?weights=summary").status_code == 200
    finally:
        for _ in range(taken):
            places.release()
    assert client.post("/v1/explain", **BROKEN).status_code == 422


@shipped
def test_nothing_answers_before_the_self_test():
    with served(lifespan=False) as c:
        assert c.get("/health").status_code == 503
        for res in (c.get("/v1/network"), c.get("/v1/network/tensor/any.weight"),
                    c.post("/v1/explain", json={})):
            _problem(res, 503)


@shipped
def test_explaining_never_changes_the_model(client, ask):
    before = client.post("/v1/predict", json=ask["predict"]).content
    for extra in ({}, {"method": "integrated_gradients", "steps": 4}, {"detail": "exact"}):
        _explain(client, dict(ask["body"], **extra))
    model = client.app.state.service.predictor.model
    assert model.training is False
    assert all(p.grad is None for p in model.parameters())
    assert client.post("/v1/predict", json=ask["predict"]).content == before


def test_the_routes_are_listed_and_served_exactly_when_shipped(client):
    listed = {e.split(" ", 1)[1] for e in MANIFEST["serve"]["endpoints"]}
    paths = {path for _, path in ENDPOINTS}
    assert (paths <= listed) if SHIPPED else not paths & listed
    if not SHIPPED:                  # --no-inspect: the routes are not there at all
        for res in (client.get("/v1/network"), client.post("/v1/explain", json={})):
            _problem(res, 404)
