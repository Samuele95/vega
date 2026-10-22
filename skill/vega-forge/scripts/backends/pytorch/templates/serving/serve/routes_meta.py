"""
serve.routes_meta — what {{PACKAGE}}'s service says about itself: its status, its records, its
examples, its code, and the page a browser opens.

    GET /health         the status: ready only after the self-test (200), else 503 and why
    GET /v1/info        an allow-list of MANIFEST.json, the contract versions, and the live
                        `inspection` and `showcase` sections
    GET /v1/schema      the input contract, the whole output record, the sample record, the truth
    GET /v1/examples    vega/examples@2: the shipped test rows, the demonstration sample and the
                        presets (examples/*.json), assembled once at start-up
    GET /v1/code        vega/code-view@2: the source this process runs, file by file, each with
                        its role id — the verified code, read where it runs
    GET /               the showcase entry when a route module set `page_override` (the
                        showcase slot), else the console page
    GET /console        always the console page: the service's own generic page, which renders
                        any answer as a key/value table

A route module (build plan R-D1): the module-level literals below are what pack reads, with
`ast.literal_eval`, to write MANIFEST.serve.endpoints. `/v1/code` is the `code` capability,
which every service carries with `core`: the code shown is the code that runs, so it is served
by the process that verified it, never by a copy. The payloads are built once, in `on_ready`,
from the files `startup` and `predict.verify` authenticated: a file altered after start-up is
never served while /health says ready. The pages carry `no-cache` (a re-packed bundle ships a
new page), `nosniff`, and a content security policy that lets them reach this origin only.
"""
import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import startup
from .problems import thawed
from .service import API_VERSION

__all__ = ["CAPABILITY", "ENDPOINTS", "MODEL_ROUTES", "OPEN_WHILE_UNREADY", "on_ready",
           "register"]

ENDPOINTS = (("GET", "/"), ("GET", "/console"), ("GET", "/health"), ("GET", "/v1/code"),
             ("GET", "/v1/examples"), ("GET", "/v1/info"), ("GET", "/v1/schema"))
MODEL_ROUTES = ()
OPEN_WHILE_UNREADY = ("/health", "/", "/console")
CAPABILITY = "core"

CONSOLE = Path(__file__).resolve().parent / "static" / "index.html"
# what /v1/info publishes of the manifest: an allow-list, so a key the packer adds later is not
# published by default (the file digests and the records stay in the bundle)
INFO_KEYS = ("schema", "slug", "run", "campaign", "package", "model_version", "created_at",
             "backend", "python", "include", "label", "output", "sample", "input_schema",
             "roundtrip", "examples", "exports", "serve", "usage", "stances", "monitor",
             "chain", "blueprint", "reemits")
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self' 'unsafe-inline'; connect-src 'self'; font-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")
PAGE_HEADERS = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": CSP}
EXAMPLE_FILES = {"demo": ("demo.json", "vega/demo@2"), "presets": ("presets.json",
                                                                   "vega/presets@2")}
CODE_SCHEMA = "vega/code-view@2"
ADAPTER_ROLES = ("Model", "FeatureTransformer", "Predictor", "ModelStore", "Monitor")
# the order a reader meets the code in: the network first, then what feeds and serves it
CODE_ORDER = (*ADAPTER_ROLES, "composition_root", "domain", "ports", "support", "Inspection",
              "http_entry")
log = logging.getLogger("{{CMD}}.serve")


def register(app: FastAPI, state) -> None:
    console = CONSOLE.read_bytes() if CONSOLE.is_file() else None

    @app.get("/health")
    def health():
        status = state.status
        return JSONResponse(status.to_dict(), status_code=200 if status.ready else 503)

    @app.get("/v1/info")
    def info():
        return _info(state)

    @app.get("/v1/schema")
    def schema():
        return state.contract.describe()

    @app.get("/v1/examples")
    def examples():
        state.ready()
        return Response(state.payloads["examples"], media_type="application/json")

    @app.get("/v1/code")
    def code():
        state.ready()
        return Response(state.payloads["code"], media_type="application/json")

    @app.get("/", include_in_schema=False)
    def page():
        override = state.page_override
        answer = override() if override is not None else None
        return answer if answer is not None else _console(console)

    @app.get("/console", include_in_schema=False)
    def console_page():
        return _console(console)


def on_ready(state) -> None:
    """The /v1/examples and /v1/code bodies, built once from the verified files."""
    state.payloads["examples"] = examples_payload(state)
    state.payloads["code"] = CodeView.of(state).payload()


def _console(body):
    if body is None:
        raise StarletteHTTPException(404, "this service ships no console page")
    return Response(body, media_type="text/html; charset=utf-8", headers=PAGE_HEADERS)


def _info(state) -> dict:
    manifest, status, contract = state.manifest, state.status, state.contract
    body = {k: thawed(manifest[k]) for k in INFO_KEYS if k in manifest}
    body.update(api_version=API_VERSION, contract_version=contract.version,
                feature_contract_version=contract.feature_version,
                inspection=thawed(status.inspection), showcase=thawed(status.showcase))
    return body


def examples_payload(state) -> bytes:
    """vega/examples@2: the shipped rows (examples/rows.csv, read with the schema's types and
    encoded by the contract), and the demonstration sample and presets pack wrote beside them
    (`examples/demo.json`, `examples/presets.json`) when it did."""
    root, tables = state.root, state.contract.tables
    rows = state.bundle.read_rows(root / "examples" / "rows.csv", root)
    rows = rows.drop(columns=[c for c in state.contract.models.columns.ignored
                              if c in rows.columns])
    kind = (state.manifest.get("examples") or {}).get("kind")
    presets = _example_doc(root, *EXAMPLE_FILES["presets"]) or {}
    body = {"schema": "vega/examples@2",
            "test": {"source": kind if kind in ("test", "synthetic") else "test",
                     "rows": tables.json_rows(rows), "history": None},
            "demo": _example_doc(root, *EXAMPLE_FILES["demo"]),
            "presets": list(presets.get("presets") or [])}
    return json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _example_doc(root: Path, name: str, schema: str):
    """One example file of the given schema, or None (absent, or another schema: a file this
    service cannot vouch for is left out, and the log says so)."""
    path = root / "examples" / name
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != schema:
        log.warning("examples/%s is not %s; /v1/examples leaves it out", name, schema)
        return None
    return doc


class CodeView:
    """/v1/code (vega/code-view@2): the source files this process runs, read where it runs them
    after `startup` checked every one against MANIFEST.json — so what the page shows is, byte
    for byte, what answers. Each file carries a role id from the package's blueprint map (the
    page renders `code.about.<role>`; no English is sent) and a kind: `adapter`, `support` or
    `entrypoint`. The modules the service never runs (a full bundle's data source, trainer and
    evaluator) are not shown."""

    def __init__(self, manifest: dict, entries: list):
        self.manifest, self.package = manifest, str(manifest.get("package") or "")
        self.by_file = {e.get("file"): e for e in entries if isinstance(e, dict)}
        roles = {e.get("role"): e.get("module") for e in entries if isinstance(e, dict)}
        self.served = {roles[r] for r in ADAPTER_ROLES if roles.get(r)} | {"domain"}
        self.inspection = f"{roles.get('Predictor') or 'serving'}/inspection/"

    @staticmethod
    def of(state) -> "CodeView":
        manifest = thawed(state.manifest)
        bmap = startup.CODE_HOME / str(manifest.get("package") or "") / "blueprint-map.json"
        doc = json.loads(bmap.read_text(encoding="utf-8")) if bmap.is_file() else {}
        return CodeView(manifest, list(doc.get("entries") or []))

    def payload(self) -> bytes:
        files = []
        for rel in (self.manifest.get("roles") or {}).get("code") or ():
            placed = self.role_of(rel)
            if placed is not None and rel.endswith(".py") and not rel.endswith("__init__.py"):
                files.append(self._file(rel, *placed))
        files.sort(key=lambda f: (CODE_ORDER.index(f["role"]), f["path"]))
        return json.dumps({"schema": CODE_SCHEMA, "verified": True, "language": "python",
                           "files": files}, ensure_ascii=False).encode("utf-8")

    def role_of(self, rel: str):
        """(role, kind) of one pinned code file, or None for a file the service never runs."""
        if rel == "predict.py":
            return "composition_root", "entrypoint"
        if rel.startswith("serve/"):
            return "http_entry", "entrypoint"
        if not rel.startswith(self.package + "/"):
            return None
        inner = rel[len(self.package) + 1:]
        role = (self.by_file.get(inner) or {}).get("role")
        if role:
            return (role, "adapter") if role in ADAPTER_ROLES else None
        if inner in ("ports.py", "domain.py"):
            return inner[:-3], "support"
        if "/" in inner and inner.split("/")[0] not in self.served:
            return None
        return ("Inspection" if inner.startswith(self.inspection) else "support"), "support"

    def _file(self, rel: str, role: str, kind: str) -> dict:
        text = (startup.CODE_HOME / rel).read_text(encoding="utf-8")
        return {"path": rel, "role": role, "kind": kind,
                "digest": (self.manifest.get("files") or {}).get(rel),
                "lines": len(text.splitlines()), "text": text}
