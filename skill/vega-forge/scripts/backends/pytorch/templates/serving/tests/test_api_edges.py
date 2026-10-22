"""
tests/test_api_edges.py — what the service decides before a route runs.

    python3 -m pytest tests/test_api_edges.py

Admission. A request to the model takes a place before its body is read: with every place
taken, a request to any route a module admits (its MODEL_ROUTES) is answered 503 with
Retry-After once `<ENV>_INFLIGHT_WAIT` runs out — a body that could never validate is a 503,
not a 422, which proves it was never parsed — while the routes nothing admits (the status, the
schema) still answer. Size. A body over the bound is 413 before it is read: declared by its
Content-Length, or counted as it arrives when none is declared and cut at the bound. The
environment prefix is the bundle's own (MANIFEST.serve.identity.env), never a name written here.
"""
import importlib

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient                  # noqa: E402

from _oracle import ORACLE, ROOT                           # noqa: E402
from serve.app import ROUTE_MODULES, create_app            # noqa: E402

ENV = ORACLE.env
JUNK = b"{ not json"
JSON = {"content-type": "application/json"}
PROBLEM = "application/problem+json"
BOUND = 2000                                     # the body bound the size tests set


def _admitted() -> list:
    """(method, path) of every route a route module of this bundle admits at the model."""
    routes = []
    for name in ROUTE_MODULES:
        if (ROOT / "serve" / f"{name}.py").is_file():
            module = importlib.import_module(f"serve.{name}")
            routes += [tuple(r) for r in module.MODEL_ROUTES]
    return routes


def test_requests_beyond_the_bound_are_refused_before_their_body_is_read(monkeypatch):
    monkeypatch.setenv(f"{ENV}_INFLIGHT_WAIT", "0.2")
    app = create_app(ROOT, max_inflight=1)
    admitted = _admitted()
    assert ("POST", "/v1/predict") in admitted, admitted
    with TestClient(app) as c:
        assert c.get("/health").json()["max_inflight"] == 1
        assert app.state.inflight.acquire(timeout=1)          # the one place, taken
        try:
            for method, path in admitted:
                res = c.request(method, path, content=JUNK, headers=JSON)
                assert res.status_code == 503, (path, res.text)
                assert res.headers.get("retry-after") == "1" and res.json()["title"] == "Busy"
            # what is never admitted answers while the model is busy
            assert c.get("/health").status_code == 200
            assert c.get("/v1/schema").status_code == 200
        finally:
            app.state.inflight.release()
        assert c.post("/v1/predict", json=ORACLE.one_request()).status_code == 200
        assert c.post("/v1/predict", content=JUNK, headers=JSON).status_code == 422


def _chunks(total: int, size: int = 512):
    """A body sent without a Content-Length: `total` bytes in chunks."""
    sent = 0
    while sent < total:
        yield b" " * min(size, total - sent)
        sent += size


def test_oversized_bodies_are_refused_before_they_are_read(monkeypatch):
    monkeypatch.setenv(f"{ENV}_MAX_BODY_BYTES", str(BOUND))
    table = pd.read_csv(ROOT / "examples" / "rows.csv", dtype=str, keep_default_na=False)
    wide = pd.concat([table] * (1 + 4 * BOUND // max(1, len(table.to_csv(index=False)))))
    with TestClient(create_app(ROOT)) as small:
        res = small.post("/v1/predict/csv", files={
            "file": ("t.csv", wide.to_csv(index=False).encode("utf-8"), "text/csv")})
        assert res.status_code == 413, res.text
        assert res.headers["content-type"].startswith(PROBLEM)
        many = ORACLE.one_request()["rows"][:1] * (1 + BOUND // 8)   # far over the bound
        res = small.post("/v1/predict", json={"rows": many})
        assert res.status_code == 413 and res.headers["content-type"].startswith(PROBLEM)
        res = small.post("/v1/predict", content=_chunks(2 * BOUND), headers=JSON)
        assert res.status_code == 413, res.text
        assert res.headers["content-type"].startswith(PROBLEM)
        # a body within the bound is read and answered
        one = {"rows": ORACLE.one_request()["rows"][:1]}
        assert small.post("/v1/predict", json=one).status_code == 200
