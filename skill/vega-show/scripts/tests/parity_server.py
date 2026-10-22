#!/usr/bin/env python3
"""
parity_server.py — serve the page template in front of a running bundle service.

    python3 parity_server.py --template <dir> [--showcase <dir>] --upstream <url>
                             [--host 127.0.0.1] [--port 0] [--quiet]

The showcase slot's routes, played by a small stdlib server so the template can be exercised
against a real bundle before the slot exists (W01's parity probe on R7):

  GET /                    the template's index.html (with the slot's CSP, no-cache, nosniff)
  GET /showcase/page/<p>   a file of the template (never package.json or tests/)
  GET /showcase/<p>        any other showcase file, from --showcase (404 without it)
  *   /health, /v1/<p>     forwarded to the upstream service: method, query, body, content type
  GET /report.pdf          forwarded too (the target of the code notes' report links)
                           The page reads vega/network-view@2 only (W15 removed StageGraph.fromLegacy):
                           a v2.2 upstream's GET /v1/network answer (no schema, a dense chain of
                           layers) is passed through D3 §5.6's mapping (`view_of_legacy`, as the
                           node tests' tests/legacy-view.js), and the explain request's `detail`, a
                           v2.3 field, is not forwarded to a v2.2 upstream
  GET /favicon.ico         204 (browsers ask for it; the page has none)

Anything else is 404. Every answer is recorded (method, path, status) so a caller can check that
the page asked for nothing it should not have and received no error. Files carry an ETag equal to
their sha256; a matching If-None-Match answers 304.

As a module: `start(template, upstream, ServeOptions(showcase=None, host="127.0.0.1", port=0,
quiet=True)) -> ParityServer` with `.url`, `.requests` and `.stop()`.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

# The slot's media types (R-D5) and its CSP for HTML (D3 §12).
MEDIA = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml",
         ".woff2": "font/woff2", ".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8",
         ".png": "image/png"}
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self' 'unsafe-inline'; connect-src 'self'; font-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")
NOT_SERVED = ("package.json", "tests/", "node_modules/")
FORWARD_HEADERS = ("content-type", "accept", "if-none-match")
TIMEOUT = 300


class _NoRedirect(urllib.request.HTTPErrorProcessor):
    """Hands every upstream status back as it is (4xx and 5xx included)."""

    def http_response(self, request, response):
        return response

    https_response = http_response


_OPENER = urllib.request.build_opener(_NoRedirect)


def _inside(root: Path, rel: str) -> Path | None:
    """The file `rel` names under `root`, or None when it escapes it, is a link or is missing."""
    if not rel or rel.startswith("/") or "\\" in rel:
        return None
    path = root / rel
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (FileNotFoundError, ValueError, OSError):
        return None
    if path.is_symlink() or not resolved.is_file():
        return None
    return resolved


def _without_detail(data: bytes) -> bytes:
    try:
        doc = json.loads(data)
    except ValueError:
        return data
    if isinstance(doc, dict):
        doc.pop("detail", None)
    return json.dumps(doc).encode()


ACTIVATIONS = {"Identity": "none", "ReLU": "relu", "GELU": "gelu", "SiLU": "silu", "Tanh": "tanh",
               "Sigmoid": "sigmoid", "LeakyReLU": "leaky_relu"}


def _f32(w: dict) -> str:
    """A v2.2 weight record's values as base64 little-endian float32 (int8 records rescaled)."""
    if w.get("f32"):
        return w["f32"]
    raw = base64.b64decode(w["int8"])
    values = [((b + 128) % 256 - 128) * float(w["scale"]) for b in raw]
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def view_of_legacy(p: dict, task: str = "multiclass") -> dict:
    """The v2.2-merge network payload as vega/network-view@2 (D3 §5.6): a dense stage per layer
    (the last the head) with its activation, a dropout stage after a layer that drops units, the
    weights as base64-le tensors and the biases as JSON lists, the chain, a row sample."""
    nodes, tensors, chain, last = [], {}, [], len(p["layers"]) - 1
    for l, (layer, w) in enumerate(zip(p["layers"], p["weights"])):
        nid, act = layer.get("name") or f"layer{l}", ACTIVATIONS.get(layer.get("activation"), layer.get("activation") or "none")
        nodes.append({"id": nid, "type": "head" if l == last else "dense", "view": "units", "parent": None, "children": [],
                      "role": None, "path": nid, "in_shape": [layer["in"]], "out_shape": [layer["out"]], "per": "sample",
                      "config": {}, "formula": "", "code": None,
                      "params": {"count": layer.get("params") or 0, "weight": f"{nid}.weight", "bias": f"{nid}.bias", "activation": act}})
        tensors[f"{nid}.weight"] = {"id": f"{nid}.weight", "node": nid, "role": "weight", "shape": list(w["shape"]),
                                    "encoding": "base64-le", "data": _f32(w)}
        tensors[f"{nid}.bias"] = {"id": f"{nid}.bias", "node": nid, "role": "bias", "shape": [layer["out"]],
                                  "encoding": "json", "data": list(w.get("bias") or [])}
        if (layer.get("dropout") or 0) > 0:
            did = f"{nid}.dropout"
            nodes.append({"id": did, "type": "dropout", "view": "box", "parent": None, "children": [], "role": None, "path": did,
                          "in_shape": [layer["out"]], "out_shape": [layer["out"]], "per": "sample", "config": {"p": layer["dropout"]},
                          "formula": "", "code": None, "params": {"count": 0, "p": layer["dropout"]}})
        chain.append({"node": nid, "in": layer["in"], "out": layer["out"], "activation": None if act == "none" else act,
                      "dropout": layer.get("dropout") or 0, "weight": f"{nid}.weight", "bias": f"{nid}.bias",
                      "params": layer.get("params") or 0})
    link = {"multiclass": "softmax", "binary": "sigmoid", "multilabel": "sigmoid_each", "regression": "identity"}[task]
    outputs = {"multiclass": "classes", "binary": "score", "multilabel": "labels", "regression": "value"}[task]
    inputs = list(p["inputs"])
    return {"schema": "vega/network-view@2", "available": True, "family": {"name": None, "source": None},
            "task": {"kind": task, "base": None, "link": link, "outputs": {"kind": outputs, "names": list(p["outputs"])},
                     "positive": None, "threshold": None, "threshold_scale": None, "quantiles": None,
                     "decision_rule": "argmax" if link == "softmax" else "none", "targets": ["logit"], "default_target": "logit"},
            "sample": {"shape": "row", "length": None, "stride": None, "horizon": None, "autoregressive": None, "level": None,
                       "directed": None, "self_loops": None, "protocol": None, "batching": None},
            "inputs": {"order": inputs, "numeric": inputs, "nominal": [], "width": len(inputs), "past_target": None,
                       "edges": None, "axis_labels": {}},
            "params": {"total": p.get("params") or 0, "trainable": p.get("params") or 0},
            "nodes": nodes, "edges": [], "chain": {"inputs": inputs, "layers": chain}, "tensors": tensors}


class _Handler(BaseHTTPRequestHandler):
    server_version = "parity-server/1"
    protocol_version = "HTTP/1.1"

    # -- dispatch -------------------------------------------------------------------------------
    def do_GET(self):
        self._route()

    def do_HEAD(self):
        self._route()

    def do_POST(self):
        self._route()

    def _route(self):
        path = urlsplit(self.path).path
        if path == "/health" or path.startswith("/v1/") or path == "/report.pdf":
            return self._forward()
        if self.command not in ("GET", "HEAD"):
            return self._status(405)
        if path == "/":
            return self._file(self.server.template / "index.html", html=True)
        if path == "/favicon.ico":
            return self._status(204)
        if path.startswith("/showcase/page/"):
            found = _inside(self.server.template, unquote(path[len("/showcase/page/"):]))
            if found is not None and found.relative_to(self.server.template).as_posix().startswith(NOT_SERVED):
                found = None
            return self._file(found)
        if path.startswith("/showcase/") and self.server.showcase is not None:
            return self._file(_inside(self.server.showcase, unquote(path[len("/showcase/"):])))
        return self._status(404)

    # -- answers --------------------------------------------------------------------------------
    def _record(self, status: int):
        with self.server.lock:
            self.server.requests.append((self.command, self.path, status))

    def _status(self, status: int):
        self._record(status)
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _file(self, path: Path | None, html: bool = False):
        if path is None or path.suffix not in MEDIA:
            return self._status(404)
        body = path.read_bytes()
        etag = '"sha256:' + hashlib.sha256(body).hexdigest() + '"'
        if self.headers.get("If-None-Match") == etag:
            self._record(304)
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        self._record(200)
        self.send_response(200)
        self.send_header("Content-Type", MEDIA[path.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if html:
            self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length) if length else None
        path = urlsplit(self.path).path
        if path == "/v1/explain" and data and self.server.legacy:
            data = _without_detail(data)
        req = urllib.request.Request(self.server.upstream + self.path, data=data, method=self.command)
        for h in FORWARD_HEADERS:
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with _OPENER.open(req, timeout=TIMEOUT) as r:
                status, headers, body = r.status, r.headers, r.read()
        except (urllib.error.URLError, OSError) as exc:
            status, headers = 502, {}
            body = json.dumps({"title": "upstream unreachable", "detail": str(exc)}).encode()
        if path == "/v1/network" and status == 200:
            body = self._mapped(body)
        self._record(status)
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", headers.get("Cache-Control", "no-cache"))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _mapped(self, body: bytes) -> bytes:
        """A v2.2 upstream's network as network-view@2 (D3 §5.6); a v2.3 one as it is."""
        try:
            doc = json.loads(body)
        except ValueError:
            return body
        if not isinstance(doc, dict) or "schema" in doc or not isinstance(doc.get("layers"), list):
            return body
        self.server.legacy = True
        return json.dumps(view_of_legacy(doc)).encode()

    def log_message(self, fmt, *args):
        if not self.server.quiet:
            sys.stderr.write("parity-server: " + (fmt % args) + "\n")


class ParityServer:
    """A running parity server: its base URL, the requests it answered, and stop()."""

    def __init__(self, httpd: ThreadingHTTPServer, thread: threading.Thread):
        self._httpd, self._thread = httpd, thread
        host, port = httpd.server_address[:2]
        self.url = f"http://{host}:{port}"

    @property
    def requests(self) -> list[tuple[str, str, int]]:
        with self._httpd.lock:
            return list(self._httpd.requests)

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=10)


@dataclass(frozen=True)
class ServeOptions:
    """How the parity server serves, beyond the template and the upstream: a showcase directory for
    /showcase/* other than page/ (None: those answer 404), the address (port 0: a free one), and
    whether each request is logged on stderr."""

    showcase: str | Path | None = None
    host: str = "127.0.0.1"
    port: int = 0
    quiet: bool = True


def start(template, upstream: str, options: ServeOptions = ServeOptions()) -> ParityServer:
    template = Path(template).resolve()
    if not (template / "index.html").is_file():
        raise FileNotFoundError(f"{template} holds no index.html")
    httpd = ThreadingHTTPServer((options.host, options.port), _Handler)
    httpd.daemon_threads = True
    httpd.template = template
    httpd.showcase = Path(options.showcase).resolve() if options.showcase else None
    httpd.upstream = upstream.rstrip("/")
    httpd.requests, httpd.lock, httpd.quiet = [], threading.Lock(), options.quiet
    httpd.legacy = False                 # set once a v2.2 upstream answered /v1/network
    thread = threading.Thread(target=httpd.serve_forever, name="parity-server", daemon=True)
    thread.start()
    return ParityServer(httpd, thread)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, help="the page template (vega-show/assets/page)")
    ap.add_argument("--showcase", default=None, help="a showcase directory for /showcase/* other than page/")
    ap.add_argument("--upstream", required=True, help="the bundle service, e.g. http://127.0.0.1:8765")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0: a free port")
    ap.add_argument("--quiet", action="store_true", help="do not log each request")
    a = ap.parse_args(argv)
    try:
        srv = start(a.template, a.upstream, ServeOptions(a.showcase, a.host, a.port, a.quiet))
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"OK: serving {Path(a.template).resolve()} at {srv.url}/ in front of {a.upstream}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
