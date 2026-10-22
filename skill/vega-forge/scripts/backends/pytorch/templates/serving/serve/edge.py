"""
serve.edge — everything of {{PACKAGE}}'s service that decides before a route runs: the limits,
admission at the model, the body bound, the readiness gate, and every refusal as an RFC 9457
problem body.

    Limits.resolve(root, env, max_rows, torch_threads, max_inflight) -> Limits
    RouteTable.of(modules) -> RouteTable     the route modules' MODEL_ROUTES and
                                             OPEN_WHILE_UNREADY, united
    install(app, state, table)               the exception handlers, then the edge middlewares

Admission. A request to the model is admitted before its body is read: at most
{{ENV}}_MAX_INFLIGHT requests at the model at once (per process — `--workers N` multiplies it);
one that finds no place within {{ENV}}_INFLIGHT_WAIT seconds is answered 503 with Retry-After,
its body never parsed — the GIL-bound part of a large request — so the default bound is small
(at most 2). Only the routes a module lists in MODEL_ROUTES are admitted; nothing else queues.

Size. A declared body over the bound is 413 before it is read; one sent without Content-Length
is counted as it arrives and cut at the bound ({{ENV}}_MAX_BODY_BYTES, else `body_limit`).

Readiness. Until the self-test passed only the paths the route modules list in
OPEN_WHILE_UNREADY answer (exactly, or as a prefix for one ending in `/` other than `/`); every
other request is 503 before it is parsed, never a 422 from a contract it could not verify.
"""
import asyncio
import http
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .problems import ContractError, ProblemDetail
from .service import STARTING, NotReady

__all__ = ["Admission", "BodyBound", "Limits", "RouteTable", "install", "problem"]

FALLBACK_ROW_BYTES, BODY_SLACK = 4096, 65536     # no shipped examples: a generous row size
REFUSED_TITLE = "Request does not honour the input schema"
log = logging.getLogger("{{CMD}}.serve")


def _number(env: Mapping, name: str, default, kind=int):
    value = env.get(name)
    if value in (None, ""):
        return default
    try:
        return kind(value)
    except ValueError:
        raise ValueError(f"{name}={value!r} is not a number") from None


@dataclass(frozen=True)
class Limits:
    """The service's bounds, resolved once: arguments first, then {{ENV}}_* variables."""
    max_rows: int
    torch_threads: int
    max_inflight: int
    inflight_wait: float
    max_body: int

    @staticmethod
    def resolve(root: Path, env: Mapping, max_rows=None, torch_threads=None,
                max_inflight=None) -> "Limits":
        rows = int(max_rows or _number(env, "{{ENV}}_MAX_ROWS", 10000))
        threads = int(torch_threads or _number(env, "{{ENV}}_TORCH_THREADS", 2))
        # measured: the per-request parse is GIL-bound, so places beyond two add memory and
        # latency, not throughput; one on a machine with few cores
        places = int(max_inflight or _number(env, "{{ENV}}_MAX_INFLIGHT", 0)
                     or max(1, min(2, (os.cpu_count() or 2) // (2 * max(1, threads)))))
        wait = _number(env, "{{ENV}}_INFLIGHT_WAIT", 1.0, float)
        body = _number(env, "{{ENV}}_MAX_BODY_BYTES", 0) or body_limit(root, rows)
        return Limits(rows, threads, places, wait, int(body))


def body_limit(root: Path, max_rows: int) -> int:
    """The largest request body worth reading: the widest shipped example row as JSON (the wider
    of the two encodings: every key repeats), for the rows and as many history rows, plus slack."""
    try:
        rows = pd.read_csv(Path(root) / "examples" / "rows.csv", dtype=str, keep_default_na=False)
        widest = max(len(json.dumps(r)) for r in rows.to_dict(orient="records"))
    except Exception:
        widest = FALLBACK_ROW_BYTES
    return widest * (2 * max_rows + 1) * 2 + BODY_SLACK


@dataclass(frozen=True)
class RouteTable:
    """The admitted routes and the paths open while unready, over every route module."""
    model_routes: frozenset
    open_paths: tuple

    @staticmethod
    def of(modules) -> "RouteTable":
        admitted = {tuple(r) for m in modules for r in getattr(m, "MODEL_ROUTES", ())}
        paths = [p for m in modules for p in getattr(m, "OPEN_WHILE_UNREADY", ())]
        return RouteTable(frozenset(admitted), tuple(dict.fromkeys(paths)))

    def is_open(self, path: str) -> bool:
        return any(path == p or (p != "/" and p.endswith("/") and path.startswith(p))
                   for p in self.open_paths)


def problem(detail: ProblemDetail, headers=None) -> JSONResponse:
    return JSONResponse(detail.body(), detail.status, headers, "application/problem+json")


class Admission:
    """A place at the model, taken at the ASGI edge — before the body is read. Pure ASGI rather
    than a route dependency: FastAPI receives, parses and validates a body before a route runs,
    so a bound checked there bounds inference only. A request without a place is answered 503
    without awaiting `receive` at all. The places are a threading semaphore tried without
    blocking, polled until the wait runs out, so the event loop never blocks on it."""

    def __init__(self, app, places: threading.BoundedSemaphore, table: RouteTable, limits: Limits):
        self.app, self.places, self.table, self.limits = app, places, table, limits

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or \
                (scope.get("method"), scope.get("path")) not in self.table.model_routes:
            return await self.app(scope, receive, send)
        deadline = time.monotonic() + self.limits.inflight_wait
        while not self.places.acquire(blocking=False):
            if time.monotonic() >= deadline:
                busy = problem(ProblemDetail(
                    503, "Busy", f"{self.limits.max_inflight} request(s) already at the model; "
                    "retry shortly — this request's body was not read"),
                    headers={"Retry-After": "1"})
                return await busy(scope, receive, send)
            await asyncio.sleep(0.01)
        try:
            await self.app(scope, receive, send)
        finally:
            self.places.release()


class BodyBound:
    """The body bound for a request that declares no Content-Length (chunked): the bytes are
    counted as they arrive and the request is refused with 413 at the first chunk that crosses
    the bound — never read whole and then measured."""

    def __init__(self, app, max_body: int):
        self.app, self.max_body = app, max_body

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or any(k.lower() == b"content-length"
                                          for k, _ in scope.get("headers") or ()):
            return await self.app(scope, receive, send)
        seen = 0

        async def bounded():
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body") or b"")
                if seen > self.max_body:
                    raise StarletteHTTPException(413, f"more than {self.max_body} bytes sent "
                                                 "without a Content-Length; cut at the bound")
            return message

        await self.app(scope, bounded, send)


def install(app: FastAPI, state, table: RouteTable) -> None:
    """The refusals, then the edge: admission innermost, then the chunked-body bound, the readiness
    gate, the declared-size check and the access log outermost (a request refused outside never
    takes a place at the model)."""
    _handlers(app, state)
    places = threading.BoundedSemaphore(state.limits.max_inflight)
    app.state.inflight = places
    app.add_middleware(Admission, places=places, table=table, limits=state.limits)
    app.add_middleware(BodyBound, max_body=state.limits.max_body)
    _gate(app, state, table)
    _size_and_log(app, state.limits.max_body)


def _handlers(app: FastAPI, state) -> None:
    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        return problem(ProblemDetail(422, REFUSED_TITLE, "every offending column is listed in "
                                     "errors", tuple(state.contract.problems.group(exc.errors()))))

    @app.exception_handler(ContractError)
    async def refused(request: Request, exc: ContractError):
        return problem(ProblemDetail(422, REFUSED_TITLE, exc.detail, tuple(exc.errors)))

    @app.exception_handler(StarletteHTTPException)
    async def status(request: Request, exc: StarletteHTTPException):
        # a wrong path or method answers in the same RFC 9457 shape as every other refusal
        phrase = http.HTTPStatus(exc.status_code).phrase
        return problem(ProblemDetail(exc.status_code, phrase, str(exc.detail)),
                       headers=getattr(exc, "headers", None))

    @app.exception_handler(NotReady)
    async def not_ready(request: Request, exc: NotReady):
        return problem(ProblemDetail(503, "Service not ready", str(exc)))

    @app.exception_handler(ValueError)
    async def named_refusal(request: Request, exc: ValueError):
        # the package's refusal by column (samples.KeyRefused) is the request's to fix: 422; any
        # other ValueError is a defect, raised on to the handler below and the error log
        if not callable(getattr(exc, "problem_errors", None)):
            raise exc
        return problem(ProblemDetail(422, REFUSED_TITLE, str(exc), tuple(exc.problem_errors())))

    @app.exception_handler(Exception)
    async def defect(request: Request, exc: Exception):
        # not logged here: the server re-raises the exception to the ASGI server after this
        # response, and its error log records it (with the traceback) once
        return problem(ProblemDetail(500, "Internal error", "the request could not be "
                                     "answered; the server log records the cause"))


def _gate(app: FastAPI, state, table: RouteTable) -> None:
    @app.middleware("http")
    async def gate(request: Request, call_next):
        # before validation: an unready service answers 503, not a 422 from a contract it
        # could not verify
        if not state.status.ready and not table.is_open(request.url.path):
            return problem(ProblemDetail(503, "Service not ready",
                                         state.status.error or STARTING))
        return await call_next(request)


def _size_and_log(app: FastAPI, max_body: int) -> None:
    @app.middleware("http")
    async def size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body:
            return problem(ProblemDetail(413, "Request body too large",
                                         f"{int(length)} bytes; at most {max_body}"))
        return await call_next(request)

    @app.middleware("http")
    async def access(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        log.info(json.dumps({"method": request.method, "path": request.url.path,
                             "status": response.status_code,
                             "ms": round((time.perf_counter() - t0) * 1000, 1)}))
        return response
