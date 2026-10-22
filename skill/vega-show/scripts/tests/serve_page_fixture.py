#!/usr/bin/env python3
"""
serve_page_fixture.py — one case of the page fixture matrix served under the real routes
(BUILD-PLAN W15; D4 §17.4), so that a browser can open the page template over it with no
training and no service code.

    python3 serve_page_fixture.py CASE [--fixtures DIR] [--port N] [--template DIR]

CASE is a case directory written by make_page_fixtures.py, or a case id (or number) under
--fixtures. The server answers what the bundle's service and its showcase slot answer:

    GET  /                          the template's index.html (no-cache, nosniff, the page's CSP)
    GET  /showcase/page/<path>      the template's files (an ETag; `If-None-Match` answers 304)
    GET  /showcase/story.json       the case's story; /showcase/page.json its page.json;
         /showcase/data/attribution.json its attribution (404 when the case has none);
         /showcase/briefs/<stage>-brief.md a stage brief the case ships (the code notes' links)
    GET  /health  /v1/info  /v1/schema  /v1/examples  /v1/network  /v1/story  /v1/showcase
                                    the case's documents
    GET  /v1/network/tensor/<id>    one tensor of network.json (404 when it carries no values)
    GET  /v1/code                   vega/code-view@2 of a bundle carrying the case's network:
                                    the live templates rendered from its netspec.json, once
                                    (make_page_fixtures.code_view)
    POST /v1/predict, /predict      the answers of answers.json: a row (or a graph node) by the
                                    row, a window by the row and the context history ++ rows
                                    give it, a graph by its rows (D2's one-row-out-per-row)
    POST /v1/predict/csv            the same, for an uploaded CSV in the schema's columns
    POST /v1/explain                the explanation of answers.json for that request
    POST /v1/monitor                a case with monitor.json: the contract's MonitorRequest
                                    (a request it refuses is a 422, recorded in `misses`), the
                                    rows answered as /v1/predict answers them, the body written by
                                    the route's own monitor_body with the templates' Monitor over
                                    the case's reference (make_page_fixtures.MonitorSide); `?all=true`
                                    every reading. Any other case: 404, it ships no reference

A request the case holds no answer for is answered 422 (a problem naming it), as the service
refuses a request it cannot answer, and is recorded in `misses` — the browser driver
(`smoke_page.py`) fails a case whose page asked something the fixture cannot answer, so the
answers a page shows are always the network's. Every request is recorded (`requests`, with its
status). Module use: `start(case_dir, template=None, port=0) -> FixtureServer` (`url`,
`requests`, `misses`, `stop()`).
"""
from __future__ import annotations

import argparse
import csv
import email.parser
import email.policy
import hashlib
import io
import json
import mimetypes
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE.parents[1] / "assets" / "page"
sys.path.insert(0, str(HERE))

import make_page_fixtures as mpf  # noqa: E402  (the request keys; code_view imports lazily)

MONITOR_FILE = mpf.MONITOR_FILE
BRIEF = re.compile(r"/showcase/briefs/([a-z-]+-brief\.md)")

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self' 'unsafe-inline'; connect-src 'self'; font-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")
TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".html": "text/html; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml",
         ".woff2": "font/woff2"}
DOCS = {"/health": "health.json", "/v1/info": "info.json", "/v1/schema": "schema.json",
        "/v1/examples": "examples.json", "/v1/network": "network.json", "/v1/story": "story.json",
        "/showcase/story.json": "story.json", "/showcase/page.json": "page.json",
        "/showcase/data/attribution.json": "attribution.json"}
NO_ANSWER = "the page fixture holds no answer for this request"


class Miss(Exception):
    """A request the case holds no answer for (422, recorded)."""


class FixtureCase:
    """The documents of one case and the lookups over its answers."""

    def __init__(self, case_dir: Path, template: Path):
        self.dir, self.template = Path(case_dir), Path(template)
        if not (self.dir / "case.json").is_file():
            raise FileNotFoundError(f"{self.dir} is not a page fixture case (no case.json)")
        self.answers = self.read("answers.json")
        self.schema = self.read("schema.json")
        self.inputs = {c["name"]: c for c in self.schema.get("input") or ()}
        self.keys = self.answers["keys"]
        self._code, self._lock = None, threading.Lock()
        self.monitor_doc, self._monitor = self.read(MONITOR_FILE), None

    def read(self, name: str):
        path = self.dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def raw(self, name: str) -> bytes | None:
        path = self.dir / name
        return path.read_bytes() if path.is_file() else None

    # -- /v1/code ------------------------------------------------------------------------
    def code(self) -> bytes:
        with self._lock:
            if self._code is None:
                with tempfile.TemporaryDirectory(prefix="fixture-code-") as tmp:
                    view = mpf.code_view(self.read("netspec.json"), Path(tmp))
                self._code = json.dumps(view, ensure_ascii=False).encode("utf-8")
            return self._code

    # -- the request contract ------------------------------------------------------------
    def check_rows(self, rows, field: str) -> None:
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise Miss(f"{field} is not a list of rows")
        for r in rows:
            extra = [c for c in r if c not in self.inputs]
            missing = [c for c, spec in self.inputs.items() if spec.get("required") and c not in r]
            if extra or missing:
                raise Miss(f"{field}: columns {extra} are not in the schema, {missing} are missing")

    # -- POST /v1/predict ----------------------------------------------------------------
    def predict(self, body: dict) -> dict:
        rows, history = body.get("rows"), body.get("history")
        self.check_rows(rows, "rows")
        if history is not None:
            self.check_rows(history, "history")
        shape, table = self.answers["shape"], self.answers
        if shape in ("row", "node"):
            said = [self._find(table["rows"], mpf.row_key(r), "row") for r in rows]
        elif shape == "window":
            pairs = mpf.window_context(rows, history, self.keys["series_by"], self.keys["need"])
            said = [self._find(table["windows"], mpf.rows_key([*ctx, r]), "window") for r, ctx in pairs]
        else:
            gid, groups = self.keys["graph_id"], {}
            for i, r in enumerate(rows):
                groups.setdefault(str(r.get(gid)), []).append(i)
            said = [None] * len(rows)
            for members in groups.values():
                got = self._find(table["graphs"], mpf.rows_key([rows[i] for i in members]), "graph")
                for i, answer in zip(members, got):
                    said[i] = answer
        return {**self.answers["envelope"], "n_rows": len(rows), "predictions": said}

    def predict_csv(self, data: bytes, content_type: str) -> dict:
        rows, truth = self._csv_rows(data, content_type)
        out = self.predict({"rows": rows})
        out["rows"] = rows
        if truth is not None:
            out["truth"] = truth
        return out

    def _csv_rows(self, data: bytes, content_type: str):
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode() + data)
        part = next((p for p in msg.iter_parts() if p.get_param("name", header="content-disposition") == "file"), None)
        if part is None:
            raise Miss("the upload has no `file` part")
        text = part.get_payload(decode=True).decode("utf-8-sig")
        truth_col = (self.schema.get("truth") or {}).get("column")
        rows, truth = [], []
        for rec in csv.DictReader(io.StringIO(text)):
            row = {}
            for name, value in rec.items():
                if name == truth_col and name not in self.inputs:
                    truth.append(value)
                    continue
                dtype = (self.inputs.get(name) or {}).get("dtype")
                row[name] = mpf._cell(value, dtype if dtype else "string")
            rows.append(row)
        return rows, (truth if truth else None)

    def _find(self, table: dict, key: str, what: str):
        if key not in table:
            raise Miss(f"{NO_ANSWER}: a {what} it was not built with")
        return table[key]

    # -- POST /v1/monitor ----------------------------------------------------------------
    def monitor(self, body: dict, report: bool) -> dict:
        """The monitor route's body (module docstring); Miss when the contract refuses the
        request or the case holds no answer for its rows."""
        with self._lock:
            if self._monitor is None:
                work = Path(tempfile.mkdtemp(prefix="fixture-watch-"))
                monitor_cls, route = mpf.watch_package(work)
                self._monitor = mpf.MonitorSide(self.monitor_doc, mpf.serve_package(work), monitor_cls, route)
        try:
            return self._monitor.answer(body, self.predict, report)
        except ValueError as exc:
            raise Miss(str(exc)) from None

    # -- POST /v1/explain ----------------------------------------------------------------
    def explain(self, body: dict) -> dict:
        for field in ("rows", "history"):
            if body.get(field) is not None:
                self.check_rows(body[field], field)
        if body.get("row") is not None:
            self.check_rows([body["row"]], "row")
        key = mpf.request_key(body)
        if key not in self.answers["explain"]:
            raise Miss(f"{NO_ANSWER}: an explain request other than its presets'")
        return self.answers["explain"][key]


def _problem(status: int, title: str, detail: str, field: str | None = None) -> bytes:
    return json.dumps({"type": "about:blank", "title": title, "status": status, "detail": detail,
                       "errors": [{"column": None, "field": field, "error": detail}]}).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "page-fixture/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):     # the requests are recorded, not printed
        pass

    @property
    def case(self) -> FixtureCase:
        return self.server.case

    def _send(self, status: int, body: bytes, ctype: str = "application/json", headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.server.requests.append((self.command, self.path, status))

    def _miss(self, why: str, field: str = "body"):
        self.server.misses.append({"method": self.command, "path": self.path, "why": why})
        self._send(422, _problem(422, "Request refused", why, field), "application/problem+json")

    def _not_found(self, why: str = "not found"):
        self._send(404, _problem(404, "Not Found", why), "application/problem+json")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return self._file(self.case.template / "index.html", page=True)
        if path.startswith("/showcase/page/"):
            return self._file(self.case.template / path[len("/showcase/page/"):])
        brief = BRIEF.fullmatch(path)
        if brief:
            body = self.case.raw(f"briefs/{brief.group(1)}")
            return self._send(200, body, "text/markdown; charset=utf-8") if body is not None else self._not_found()
        if path in DOCS:
            body = self.case.raw(DOCS[path])
            return self._send(200, body) if body is not None else self._not_found(f"the case has no {DOCS[path]}")
        if path == "/v1/showcase":
            return self._send(200, json.dumps({"verified": True, "reason": None, "source": "page fixture"}).encode())
        if path.startswith("/v1/network/tensor/"):
            return self._tensor(path[len("/v1/network/tensor/"):])
        if path == "/v1/code":
            try:
                return self._send(200, self.case.code())
            except Exception as exc:  # the render failed: the page shows the code room's error  # noqa: BLE001
                self.server.misses.append({"method": "GET", "path": path, "why": f"code view: {exc}"})
                return self._send(500, _problem(500, "Internal error", "the code view could not be rendered"),
                                  "application/problem+json")
        return self._not_found()

    def _tensor(self, tid: str):
        tensors = (self.case.read("network.json") or {}).get("tensors") or {}
        t = tensors.get(tid)
        if not t or t.get("encoding") not in ("json", "base64-le"):
            return self._not_found(f"no tensor {tid} with values")
        return self._send(200, json.dumps(t).encode())

    def _file(self, path: Path, page: bool = False):
        root = self.case.template.resolve()
        try:
            real = path.resolve()
            real.relative_to(root)
        except (OSError, ValueError):
            return self._not_found()
        if not real.is_file() or real.is_symlink():
            return self._not_found()
        body = real.read_bytes()
        etag = '"sha256:' + hashlib.sha256(body).hexdigest() + '"'
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if page:
            headers["Content-Security-Policy"] = CSP
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.server.requests.append((self.command, self.path, 304))
            return None
        ctype = TYPES.get(real.suffix) or mimetypes.guess_type(real.name)[0] or "application/octet-stream"
        return self._send(200, body, ctype, headers)

    def do_POST(self):
        url = urlsplit(self.path)
        path = url.path
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length) if length else b""
        if path == "/v1/monitor" and self.case.monitor_doc is None:
            return self._not_found("this bundle ships no monitor reference")
        try:
            if path == "/v1/monitor":
                report = "all=true" in (url.query or "").lower().split("&")
                return self._send(200, json.dumps(self.case.monitor(json.loads(data or b"{}"), report), allow_nan=False).encode())
            if path in ("/v1/predict", "/predict"):
                return self._send(200, json.dumps(self.case.predict(json.loads(data or b"{}"))).encode())
            if path == "/v1/predict/csv":
                return self._send(200, json.dumps(self.case.predict_csv(data, self.headers.get("Content-Type", ""))).encode())
            if path == "/v1/explain":
                return self._send(200, json.dumps(self.case.explain(json.loads(data or b"{}"))).encode())
        except Miss as exc:
            return self._miss(str(exc))
        except (ValueError, KeyError) as exc:
            return self._miss(f"the request could not be read: {type(exc).__name__}: {exc}")
        except Exception as exc:  # the service would answer 500: recorded, so the case fails  # noqa: BLE001
            self.server.misses.append({"method": "POST", "path": path, "why": f"{type(exc).__name__}: {exc}"})
            return self._send(500, _problem(500, "Internal error", "the fixture could not answer"), "application/problem+json")
        return self._not_found()


class FixtureServer:
    """A running fixture server: `url`, `requests` [(method, path, status)], `misses`."""

    def __init__(self, httpd: ThreadingHTTPServer, thread: threading.Thread):
        self.httpd, self.thread = httpd, thread
        self.url = f"http://127.0.0.1:{httpd.server_address[1]}"

    @property
    def case(self) -> FixtureCase:
        return self.httpd.case

    @property
    def requests(self) -> list:
        return list(self.httpd.requests)

    @property
    def misses(self) -> list:
        return list(self.httpd.misses)

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(10)


def start(case_dir: Path, template: Path | None = None, port: int = 0) -> FixtureServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    httpd.case = FixtureCase(Path(case_dir), Path(template or TEMPLATE_DIR))
    httpd.requests, httpd.misses = [], []
    thread = threading.Thread(target=httpd.serve_forever, name="page-fixture", daemon=True)
    thread.start()
    return FixtureServer(httpd, thread)


def case_dir(case: str, fixtures: Path) -> Path:
    """A case directory from a path, an id or a number under `fixtures`."""
    p = Path(case)
    if (p / "case.json").is_file():
        return p
    for d in sorted(fixtures.glob("*/case.json")):
        doc = json.loads(d.read_text(encoding="utf-8"))
        if case in (doc.get("id"), str(doc.get("n"))):
            return d.parent
    raise FileNotFoundError(f"no page fixture case {case!r} under {fixtures}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("case", help="a case directory, or a case id or number under --fixtures")
    ap.add_argument("--fixtures", default=".", help="where make_page_fixtures.py wrote the cases (default: here)")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--template", default=str(TEMPLATE_DIR))
    a = ap.parse_args(argv)
    try:
        srv = start(case_dir(a.case, Path(a.fixtures)), Path(a.template), a.port)
    except (OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"OK: serving {a.case} at {srv.url}/ (Ctrl-C stops)", flush=True)
    try:
        srv.thread.join()
    except KeyboardInterrupt:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
