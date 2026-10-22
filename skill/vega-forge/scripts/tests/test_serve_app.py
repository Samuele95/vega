#!/usr/bin/env python3
"""
test_serve_app.py — the service split and the serving cutover (build plan S07; D2 §6, §8.5;
D3 §10–§12.6 status and code; D4 §16.6; R-D1–R-D7, R-D10, R-D12, R-D14).

    python3 -m pytest test_serve_app.py [-q]

The bundles are built the way pack builds them, through vega-pack's own phases, over a package
the v2.3 emitter wrote (`serve_app_fixtures.py`): `serving_pack.render_adapters` renders the one
Predictor and its support files and carries the Model side, `serving_pack.render` renders
`serve/` — `--serve` over the orders blueprint, whose serving mode is `batch` — and the bundle's
own `serve.app` is served in-process, or in a clean interpreter for a tampered copy.

What is proved:

  routes    MANIFEST.serve.endpoints equals the routes the app registers (the route modules'
            ENDPOINTS, read by pack without importing them); every route module declares the
            four literals; the paths open while unready are their union.
  edges     admission answers 503 Busy with Retry-After before the body is read (a body that
            could never validate); a body over the bound is 413 before it is read, declared or
            chunked; an unready service answers 503 before parsing, and only its open paths.
  closed    an unpinned route module stops the service (503 naming it, on every path), and so
  world     does an altered serve/ module, before any other module of serve/ is imported; an
            altered data file leaves it unready; a self-test that fails does too.
  pages     `/` and `/console` serve the console page when no showcase is installed, with
            no-cache, nosniff and the content security policy; a page override (the showcase
            slot's) answers `/` while `/console` stays the console.
  payloads  `/v1/code` is vega/code-view@2 with role ids, kinds and a boolean `verified`, the
            text and digest of the files that run; `/v1/examples` is vega/examples@2 (its JSON
            Schema); `/health` and `/v1/info` carry the `inspection` and `showcase` sections;
            answers, CSV answers, the truth and the monitor go through the contract.
  pack      `--serve` on the batch blueprint renders serve/ with `forced_by`; without it, or
            with `--no-serve`, nothing is served; a stream blueprint renders the consumer and
            the row contract; a route module of a capability the bundle does not enable is not
            rendered (`--no-inspect`); the showcase slot and templates records; the Model side
            the package maps and imports travels, and a reach into a dropped module is refused;
            the bundle passes the mechanical conformance gate; the package templates render
            with the bundle's sample shape (a row bundle's `samples.py` holds the row adapter
            and no window or graph code, and imports no `graphs` module).
  templates `create_app` and `main` within 25 statements; C11–C13 and D2's line budgets on
            serve/; no foreign literal in serve/, bundle/ and the fragments (lint_templates.py);
            the v2.2 serving templates are gone.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import serve_app_fixtures as sf  # noqa: E402
import conformance  # noqa: E402
import serving_closure  # noqa: E402
import serving_pack  # noqa: E402
import serving_templates  # noqa: E402
from pack_context import PackExit  # noqa: E402
from vega_pack_core import CATALOG  # noqa: E402

SERVING = sf.pf.SERVING_TEMPLATES
SERVE = SERVING / "serve"
LINT = sf.pf.SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
EXAMPLES_SCHEMA = sf.pf.SKILLS / "vega-pack" / "references" / "schemas" / "examples-2.schema.json"
# D2 §6's line budget per file of the split service
BUDGETS = {"app.py": 200, "edge.py": 250, "startup.py": 220, "service.py": 250,
           "routes_meta.py": 400, "routes_predict.py": 400, "routes_monitor.py": 400}
V22_TEMPLATES = ("app.py", "contract.py", "predict.py", "monitor.py", "static/index.html")
IGNORED_ROUTES = ("/openapi.json",)


@pytest.fixture(scope="module")
def bundle():
    return sf.build()


@pytest.fixture(scope="module")
def app(bundle):
    return sf.served(bundle).create_app(bundle.root)


@pytest.fixture(scope="module")
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


def _registered(app) -> list:
    return sorted({f"{method} {route.path}" for route in app.routes
                   for method in (getattr(route, "methods", None) or ())
                   if method != "HEAD" and route.path not in IGNORED_ROUTES})


def _env(bundle) -> str:
    return bundle.manifest["serve"]["identity"]["env"]


# --------------------------------------------------------------------------- routes

def test_the_manifest_lists_every_registered_route(bundle, app):
    assert bundle.manifest["serve"]["endpoints"] == _registered(app)
    assert serving_templates.serving_endpoints(bundle.root / "serve") == _registered(app)
    assert {"POST /v1/predict", "POST /v1/monitor", "GET /v1/code", "GET /console"} <= \
        set(bundle.manifest["serve"]["endpoints"])


def test_every_route_module_declares_its_literals(bundle):
    names = ast.literal_eval(next(
        n.value for n in ast.parse((SERVE / "app.py").read_text(encoding="utf-8")).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ROUTE_MODULES"))
    assert names == tuple(CATALOG["serve_record"]["route_modules"])
    opened = []
    for path in sorted(SERVE.glob("routes_*.py")):
        literals = serving_templates.route_literals(path.read_text(encoding="utf-8"), path.name)
        assert set(map(tuple, literals["MODEL_ROUTES"])) <= set(map(tuple,
                                                                   literals["ENDPOINTS"]))
        assert literals["CAPABILITY"] in CATALOG["serve_record"]["capabilities"]
        opened += list(literals["OPEN_WHILE_UNREADY"])
    assert {"/", "/console", "/health"} <= set(opened)
    with pytest.raises(PackExit) as refused:
        serving_templates.route_literals("ENDPOINTS = tuple(x for x in ())\n", "serve/r.py")
    assert refused.value.code == 2 and "not a literal" in refused.value.lines[0]


# --------------------------------------------------------------------------- edges

def test_ready_only_after_the_self_test(client, bundle):
    health = client.get("/health")
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["ready"] is True and body["self_test"]["passed"] is True
    assert body["self_test"]["rows"] == 12 and body["model_version"] == \
        bundle.manifest["model_version"]
    assert {"inspection", "showcase"} <= set(body)


def test_admission_refuses_before_the_body_is_read(bundle, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv(f"{_env(bundle)}_INFLIGHT_WAIT", "0.2")
    app = sf.served(bundle).create_app(bundle.root, max_inflight=1)
    with TestClient(app) as c:
        assert c.get("/health").json()["max_inflight"] == 1
        assert app.state.inflight.acquire(timeout=1)          # the one place, taken
        try:
            # a body that could never validate: a 503 (not a 422) proves it was never read
            res = c.post("/v1/predict", content=b"{ not json",
                         headers={"content-type": "application/json"})
            assert res.status_code == 503 and res.headers.get("retry-after") == "1", res.text
            assert res.json()["title"] == "Busy"
            assert c.post("/v1/monitor", content=b"{").status_code == 503
            # what is never admitted answers while the model is busy
            assert c.get("/health").status_code == 200
            assert c.get("/v1/schema").status_code == 200
        finally:
            app.state.inflight.release()
        assert c.post("/v1/predict", content=b"{ not json", headers={
            "content-type": "application/json"}).status_code == 422


def test_bodies_over_the_bound_are_refused_before_they_are_read(bundle, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv(f"{_env(bundle)}_MAX_BODY_BYTES", "500")
    table = (bundle.root / "examples" / "rows.csv").read_bytes()
    assert len(table) > 500
    with TestClient(sf.served(bundle).create_app(bundle.root)) as c:
        res = c.post("/v1/predict/csv", files={"file": ("t.csv", table, "text/csv")})
        assert res.status_code == 413, res.text
        assert res.headers["content-type"].startswith("application/problem+json")
        chunks = (table[i:i + 200] for i in range(0, len(table), 200))
        streamed = c.post("/v1/predict", content=chunks,
                          headers={"content-type": "application/json"})
        assert streamed.status_code == 413, streamed.text
        assert "without a Content-Length" in streamed.json()["detail"], streamed.text


# --------------------------------------------------------------------------- the closed world

def test_an_unpinned_route_module_stops_the_service(bundle, tmp_path):
    copy = sf.copy(bundle, tmp_path)
    (copy / "serve" / "routes_extra.py").write_text(
        "ENDPOINTS = ((\"GET\", \"/extra\"),)\nMODEL_ROUTES = ()\nOPEN_WHILE_UNREADY = ()\n"
        "CAPABILITY = \"core\"\n\ndef register(app, state):\n    pass\n", encoding="utf-8")
    out = sf.probe(copy, """
h = c.get("/health")
out["health"] = [h.status_code, h.json()]
out["page"] = c.get("/").status_code
out["predict"] = c.post("/v1/predict", json={"rows": []}).status_code
out["loaded"] = sorted(m for m in sys.modules if m.startswith("serve."))
""")
    status, problem = out["health"]
    assert status == 503 and "serve/routes_extra.py" in problem["detail"], problem
    assert out["page"] == out["predict"] == 503
    assert out["loaded"] == ["serve.app", "serve.startup"], out["loaded"]


def test_an_altered_serve_module_stops_the_service_before_it_runs(bundle, tmp_path):
    copy = sf.copy(bundle, tmp_path)
    with open(copy / "serve" / "rows.py", "a", encoding="utf-8") as fh:
        fh.write("\nraise SystemExit('this module must never run')\n")
    out = sf.probe(copy, """
h = c.get("/health")
out["health"] = [h.status_code, h.json()]
out["loaded"] = sorted(m for m in sys.modules if m.startswith("serve."))
""")
    status, problem = out["health"]
    assert status == 503 and "serve/rows.py" in problem["detail"], problem
    assert "serve.rows" not in out["loaded"]


def test_an_altered_data_file_leaves_the_service_unready(bundle, tmp_path):
    copy = sf.copy(bundle, tmp_path)
    with open(copy / "fitted.json", "a", encoding="utf-8") as fh:
        fh.write(" ")
    out = sf.probe(copy, """
h = c.get("/health")
out["health"] = [h.status_code, h.json()]
out["open"] = [c.get(p).status_code for p in ("/", "/console")]
out["closed"] = [c.get("/v1/info").status_code, c.post("/v1/predict", content=b"{").status_code]
""")
    status, body = out["health"]
    assert status == 503 and body["ready"] is False and "fitted.json" in body["error"], body
    assert out["open"] == [200, 200] and out["closed"] == [503, 503]


def test_a_failed_self_test_leaves_the_service_unready(bundle, tmp_path):
    copy = sf.copy(bundle, tmp_path)
    expected = pd.read_parquet(copy / "examples" / "expected.parquet")
    expected["score"] = expected["score"] + 1.0
    expected.to_parquet(copy / "examples" / "expected.parquet", index=False)
    manifest = json.loads((copy / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["files"]["examples/expected.parquet"] = \
        sf.pf.digest(copy / "examples" / "expected.parquet")
    (copy / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = sf.probe(copy, """
h = c.get("/health")
out["health"] = [h.status_code, h.json()]
""")
    status, body = out["health"]
    assert status == 503 and body["self_test"]["passed"] is False, body
    assert "not reproduced" in body["error"] and "score" in body["error"]


# --------------------------------------------------------------------------- pages

def test_the_console_page_is_served_without_a_showcase(client, bundle):
    page = (bundle.root / "serve" / "static" / "index.html").read_bytes()
    for path in ("/", "/console"):
        res = client.get(path)
        assert res.status_code == 200 and res.content == page
        assert res.headers["content-type"].startswith("text/html")
        assert res.headers["cache-control"] == "no-cache"
        assert res.headers["x-content-type-options"] == "nosniff"
        assert "default-src 'self'" in res.headers["content-security-policy"]
    assert b"{{" not in page


def test_a_page_override_answers_the_root_and_the_console_stays(client):
    from fastapi.responses import Response
    state = client.app.state.service
    state.page_override = lambda: Response(b"<p>showcase</p>", media_type="text/html")
    try:
        assert client.get("/").content == b"<p>showcase</p>"
        assert client.get("/console").content != b"<p>showcase</p>"
    finally:
        state.page_override = None


# --------------------------------------------------------------------------- payloads

def test_the_code_view_is_the_verified_code_with_role_ids(client, bundle):
    code = client.get("/v1/code").json()
    view = CATALOG["code_view"]
    assert code["schema"] == view["schema"] and code["verified"] is True
    assert code["language"] == "python"
    by_path = {f["path"]: f for f in code["files"]}
    assert by_path["predict.py"]["role"] == "composition_root"
    assert by_path["serve/app.py"]["role"] == "http_entry"
    assert by_path[f"{bundle.package}/serving/predict.py"]["role"] == "Predictor"
    assert by_path[f"{bundle.package}/model/network.py"]["kind"] == "adapter"
    assert by_path[f"{bundle.package}/model/readout.py"]["role"] == "support"
    assert by_path[f"{bundle.package}/serving/inspection/codec.py"]["role"] == "Inspection"
    for f in code["files"]:
        assert f["role"] in view["roles"] and f["kind"] in view["kinds"], f["path"]
        assert "about" not in f
        assert f["digest"] == bundle.manifest["files"][f["path"]]
        assert f["text"] == (bundle.root / f["path"]).read_text(encoding="utf-8")
    assert not any(p.startswith(f"{bundle.package}/training/") for p in by_path)
    assert code["files"][0]["role"] == "Model"


def test_the_examples_are_examples_2(client, bundle):
    jsonschema = pytest.importorskip("jsonschema")
    body = client.get("/v1/examples").json()
    jsonschema.validate(body, json.loads(EXAMPLES_SCHEMA.read_text(encoding="utf-8")))
    assert body["test"]["source"] == "test" and len(body["test"]["rows"]) == 12
    assert body["demo"]["schema"] == "vega/demo@2" and len(body["demo"]["items"]) == sf.DEMO_ROWS
    assert [p["kind"] for p in body["presets"]] == ["edge_range"]


def test_info_and_schema_describe_the_bundle(client, bundle):
    info = client.get("/v1/info").json()
    assert info["sample"] == bundle.manifest["sample"]
    assert info["serve"]["endpoints"] == bundle.manifest["serve"]["endpoints"]
    assert {"inspection", "showcase", "contract_version"} <= set(info)
    assert "files" not in info and "roles" not in info
    schema = client.get("/v1/schema").json()
    assert schema["output"]["reading"] == "binary" and schema["sample"]["shape"] == "row"


def test_answers_csv_and_monitor_go_through_the_contract(client, bundle):
    rows = client.get("/v1/examples").json()["test"]["rows"]
    by_json = client.post("/v1/predict?include=logits", json={"rows": rows}).json()
    assert by_json["output_kind"] == "binary" and len(by_json["predictions"]) == len(rows)
    assert {"label", "probabilities", "logits"} <= set(by_json["predictions"][0])
    assert "logits" not in client.post("/predict", json={"rows": rows[:1]}).json()[
        "predictions"][0]
    table = pd.read_csv(bundle.root / "examples" / "rows.csv", dtype=str)
    table["y"] = ["0", "1"] * (len(table) // 2)
    upload = client.post("/v1/predict/csv?include=logits&include=rows&monitor=true",
                         files={"file": ("t.csv", table.to_csv(index=False), "text/csv")})
    assert upload.status_code == 200, upload.text
    answer = upload.json()
    assert answer["predictions"] == by_json["predictions"]
    assert answer["truth"] == list(table["y"]) and len(answer["rows"]) == len(rows)
    assert answer["monitor"]["readings"][0]["check"] == "x0"
    csv = client.post("/v1/predict/csv", headers={"accept": "text/csv"},
                      files={"file": ("t.csv", table.to_csv(index=False), "text/csv")})
    head = csv.text.splitlines()[0].split(",")
    kinds = bundle.manifest["output"]["field_kinds"]
    shown = [f for f in bundle.manifest["output"]["fields"]
             if kinds[f]["kind"] not in ("raw", "recon", "embedding", "warnings")]
    assert head == shown + ["true_y", "warnings"], head
    assert csv.headers["x-model-version"] == bundle.manifest["model_version"]
    monitor = client.post("/v1/monitor?all=true", json={"rows": rows, "labels": ["1"] * 12})
    assert monitor.status_code == 200 and monitor.json()["readings"], monitor.text


def test_refusals_are_problem_bodies_naming_the_column(client):
    rows = client.get("/v1/examples").json()["test"]["rows"]
    missing = client.post("/v1/predict", json={"rows": [{k: v for k, v in rows[0].items()
                                                          if k != "x0"}]})
    assert missing.status_code == 422
    assert any(e["column"] == "x0" for e in missing.json()["errors"]), missing.text
    for res in (client.get("/v1/nothing"), client.get("/v1/predict")):
        assert res.status_code in (404, 405)
        assert res.headers["content-type"].startswith("application/problem+json")


# --------------------------------------------------------------------------- pack

def test_serve_on_the_batch_blueprint_renders_serve_with_forced_by(bundle):
    record = bundle.record
    assert bundle.ctx.mode == "batch"
    assert (record["present"], record["mode"], record["forced_by"]) == (True, "online", "--serve")
    assert "batch" in record["note"] and (bundle.root / "serve" / "app.py").is_file()
    slot = dict(CATALOG["serve_record"]["showcase_slot"], env=f"{_env(bundle)}_SHOWCASE")
    assert record["showcase_slot"] == slot
    digest = serving_templates.templates_digest(SERVING, "pytorch")
    assert record["templates"] == digest and "serve/app.py" in digest["files"]
    served = bundle.manifest["entrypoints"]
    assert served[:3] == ["predict.py", "serve/__init__.py", "serve/app.py"]
    assert served.index("serve/static/index.html") < served.index("requirements-serve.txt")
    # the API tests close the list, in name order (the oracle and the S11 suites, then the
    # route modules' own suites such as test_api_showcase.py, I07); the v2.2 test_api.py is gone
    tests = [rel for rel in served if rel.startswith("tests/")]
    assert {"tests/_oracle.py", "tests/test_api_contract.py", "tests/test_api_answers.py",
            "tests/test_api_shapes.py", "tests/test_api_edges.py",
            "tests/test_api_closed_world.py"} <= set(tests), tests
    assert "tests/test_api.py" not in tests and served[-len(tests):] == sorted(tests)


def test_without_serve_nothing_is_served():
    case = sf.pf.row_case("binary")
    for mode, flags in (("batch", ()), ("online", ("--no-serve",))):
        ctx = sf.context(case, "s07_plain", mode, flags)
        with contextlib.redirect_stdout(io.StringIO()):
            record = serving_pack.render(ctx)
        assert set(record) == {"serve"} and record["serve"]["present"] is False
        assert record["serve"]["mode"] == mode and "forced_by" not in record["serve"]
        assert not (ctx.paths.root / "serve").exists()


class _Templates:
    """A pack context whose serving templates are another directory's."""

    def __init__(self, ctx, templates):
        self._ctx, self.templates = ctx, templates

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _templates_with(tmp_path, extra) -> Path:
    root = tmp_path / "serving"
    shutil.copytree(SERVING, root, ignore=shutil.ignore_patterns("__pycache__"))
    for rel, text in extra.items():
        (root / rel).write_text(text, encoding="utf-8")
    return root


INSPECT_STUB = ('ENDPOINTS = (("GET", "/v1/network"),)\nMODEL_ROUTES = ()\n'
                'OPEN_WHILE_UNREADY = ()\nCAPABILITY = "inspection"\n\n'
                'def register(app, state):\n    pass\n')


def test_a_capability_the_bundle_does_not_enable_is_not_rendered(tmp_path):
    templates = _templates_with(tmp_path, {"serve/routes_inspect.py": INSPECT_STUB})
    case = sf.pf.row_case("binary")
    for flags, present in ((("--serve",), True), (("--serve", "--no-inspect"), False)):
        ctx = _Templates(sf.context(case, "s07_caps", "batch", flags), templates)
        with contextlib.redirect_stdout(io.StringIO()):
            serving_pack.render_adapters(ctx)
            record = serving_pack.render(ctx)["serve"]
        assert (ctx.paths.root / "serve" / "routes_inspect.py").is_file() is present
        assert ("GET /v1/network" in record["endpoints"]) is present
        assert (ctx.paths.pkg / "serving" / "inspection").is_dir() is present


def test_the_package_templates_carry_the_bundle_shape():
    ctx = sf.context(sf.pf.row_case("binary"), "s07_shape", "batch", ("--serve",))
    with contextlib.redirect_stdout(io.StringIO()):
        serving_pack.render_adapters(ctx)
    assert serving_templates.bundle_shape(ctx) == "row"
    samples = (ctx.paths.pkg / "serving" / "samples.py").read_text(encoding="utf-8")
    tree = ast.parse(samples)
    defined = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    assert {"RowAssembler", "_row", "assembler_for"} <= defined, defined
    assert not defined & {"WindowAssembler", "GraphAssembler", "KeyRefused", "_window", "_graph"}
    assert "from .graphs import" not in samples and "# >>> shape" not in samples
    rendered = [p for p in (ctx.paths.pkg / "serving").rglob("*.py")
                if "# >>> shape" in p.read_text(encoding="utf-8")]
    assert not rendered, rendered


def test_a_stream_blueprint_renders_the_consumer_and_the_row_contract(tmp_path):
    consumer = '"""serve.consumer of {{PACKAGE}} (a stand-in for the test)."""\n'
    templates = _templates_with(tmp_path, {"serve/consumer.py": consumer})
    ctx = _Templates(sf.context(sf.pf.row_case("binary"), "s07_stream", "stream", ()),
                     templates)
    with contextlib.redirect_stdout(io.StringIO()):
        serving_pack.render_adapters(ctx)
        out = serving_pack.render(ctx)
    record, root = out["serve"], ctx.paths.root
    assert (record["mode"], record["entry"], record["endpoints"]) == \
        ("stream", "serve/consumer.py", [])
    assert "showcase_slot" not in record and "forced_by" not in record
    assert sorted(p.relative_to(root).as_posix() for p in (root / "serve").rglob("*.py")) == \
        sorted(serving_templates.STREAM_SERVE)
    assert not (root / "serve" / "static").exists()
    assert "serve.consumer" in (root / "Dockerfile").read_text(encoding="utf-8")
    assert "## The stream consumer" in (root / "README.md").read_text(encoding="utf-8")
    assert not list((root / "tests").glob("test_api*.py")) and \
        not (root / "tests" / "_oracle.py").exists()        # the API tests need the online service
    assert out["entrypoints"][:3] == ["predict.py", "serve/__init__.py", "serve/consumer.py"]


def test_a_stream_blueprint_without_a_consumer_template_serves_nothing(tmp_path):
    templates = _templates_with(tmp_path, {})
    (templates / "serve" / "consumer.py").unlink(missing_ok=True)
    ctx = _Templates(sf.context(sf.pf.row_case("binary"), "s07_nostream", "stream", ()),
                     templates)
    with contextlib.redirect_stdout(io.StringIO()):
        absent = serving_pack.render(ctx)["serve"]
    assert absent["present"] is False and "consumer" in absent["note"]
    assert any("best effort" in m for m in ctx.markers)
    assert not (ctx.paths.root / "serve").exists()


def test_an_unfilled_placeholder_stops_the_pack(tmp_path):
    templates = _templates_with(tmp_path, {"serve/extra.py": "X = {{NOT_A_FILL}}\n"})
    ctx = _Templates(sf.context(sf.pf.row_case("binary"), "s07_fill", "batch", ("--serve",)),
                     templates)
    with pytest.raises(PackExit) as refused, contextlib.redirect_stdout(io.StringIO()):
        serving_pack.render(ctx)
    assert refused.value.code == 2 and "{{NOT_A_FILL}}" in refused.value.lines[0]


def test_the_model_side_travels_and_never_reaches_a_dropped_module(bundle):
    pkg = bundle.root / bundle.package
    for rel in ("model/readout.py", "model/metrics.py", "model/assemble.py", "model/builder.py",
                "model/blocks_core.py", "serving/samples.py"):
        assert (pkg / rel).is_file(), rel           # the readout reads its numpy half there
    # what no served role reads stays behind (the T5 round 3 bundle audit): the exporter of a
    # bundle that exports nothing — and the map says so rather than naming a file the bundle
    # lacks
    doc = json.loads((pkg / "blueprint-map.json").read_text(encoding="utf-8"))
    for rel in ("model/export.py",):
        assert not (pkg / rel).exists(), rel
        entry = next(e for e in doc["entries"] if e.get("file") == rel)
        assert entry["status"].startswith("out of scope"), entry
    # a row bundle carries no graph-only template: its domain holds no reference graph
    assert not (pkg / "serving" / "graphs.py").exists()
    for dropped in CATALOG["package_slice"]["drop_packages"]:
        assert not (pkg / dropped).exists()
    assert serving_closure.relative_imports(
        "from ..model.readout import TaskReadout\nfrom . import registry\n",
        "serving/heads.py") == {"model/readout", "model/readout/TaskReadout",
                               "serving/registry"}
    ctx = sf.context(sf.pf.row_case("binary"), "s07_reach", "batch", ("--serve",))
    (ctx.paths.pkg / "serving" / "reach.py").write_text("from ..training import losses\n",
                                                        encoding="utf-8")
    with pytest.raises(PackExit) as refused, contextlib.redirect_stdout(io.StringIO()):
        serving_closure.copy_model_side(ctx)
    assert refused.value.code == 3 and "training/" in refused.value.lines[0]


def test_the_rendered_package_is_mapped_and_passes_the_conformance_gate(bundle):
    doc = json.loads((bundle.root / bundle.package / "blueprint-map.json").read_text())
    support = {e["file"] for e in doc["entries"] if e.get("used_by") == ["Predictor"]}
    assert {"serving/records.py", "serving/samples.py", "serving/inspection/__init__.py",
            "serving/inspection/graph.py"} <= support
    assert "serving/monitor.py" not in support and "serving/predict.py" not in support
    points = {p["file"]: p for p in doc["entrypoints"]}
    assert points["serve/routes_predict.py"]["calls"] == ["Predictor"]
    assert points["serve/app.py"]["present"] is True
    proc = subprocess.run([sys.executable, str(sf.pf.SCRIPTS / "conformance.py"), "--package",
                           str(bundle.root / bundle.package), "--pipeline",
                           str(bundle.ctx.options.pipeline), "--map",
                           str(bundle.root / bundle.package / "blueprint-map.json"),
                           "--bundle"], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:]


def test_the_dockerignore_keeps_the_showcase_and_the_readme_lists_the_routes(bundle):
    ignored = (bundle.root / ".dockerignore").read_text(encoding="utf-8").split()
    assert {"showcase.prev/", "showcase.building/", "showcase/.smoke/",
            "showcase/.work/"} <= set(ignored)
    assert not any(p.rstrip("/") == "showcase" for p in ignored)
    docker = (bundle.root / "Dockerfile").read_text(encoding="utf-8")
    assert "{" not in docker and f"USER {bundle.manifest['serve']['identity']['cmd']}" in docker
    readme = (bundle.root / "README.md").read_text(encoding="utf-8")
    assert all(f"`{e}`" in readme for e in bundle.manifest["serve"]["endpoints"])


# --------------------------------------------------------------------------- the templates

def _statements(path: Path, name: str) -> int:
    measures = conformance.measure_module(ast.parse(path.read_text(encoding="utf-8")),
                                          path.name)
    return measures.function(name).statements


def test_create_app_and_main_stay_compositions():
    assert _statements(SERVE / "app.py", "create_app") <= 25
    assert _statements(SERVE / "app.py", "main") <= 25


def test_the_service_holds_the_size_discipline():
    files = sorted(SERVE.glob("*.py"))
    findings = [f for path in files
                for f in conformance.size_findings(ast.parse(path.read_text(encoding="utf-8")),
                                                   f"serve/{path.name}")]
    assert not findings, [f.message() for f in findings]
    for name, budget in BUDGETS.items():
        lines = len((SERVE / name).read_text(encoding="utf-8").splitlines())
        assert lines <= budget, f"serve/{name} is {lines} lines > {budget}"


def test_no_foreign_literal_in_the_served_templates():
    if not LINT.is_file():
        pytest.skip("lint_templates.py is not in this tree")
    proc = subprocess.run([sys.executable, str(LINT), str(SERVE), str(SERVING / "bundle"),
                           str(SERVING / "docker"), str(SERVING / "readme")],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_v22_serving_templates_are_gone():
    left = [rel for rel in V22_TEMPLATES if (SERVING / rel).exists()]
    assert not left, f"the v2.2 serving templates {left} are still shipped"


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", __file__]))
