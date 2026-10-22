"""
serve.routes_predict — {{PACKAGE}}'s answers over HTTP, through the {{PORT_PREDICTOR}} port.

    POST /v1/predict        JSON `{rows, history?}` -> the envelope and one `RowPrediction` per
                            row (`?include=logits|reconstruction|embedding` shows the gated keys)
    POST /predict           the same (an alias kept for v2.2 callers)
    POST /v1/predict/csv    a CSV in the raw schema (multipart `file`) -> the same; the truth
                            column, when the table carries it, comes back as `truth`;
                            `?monitor=true` adds the monitor's readings over the table,
                            `?include=rows` echoes the rows as JSON, and `Accept: text/csv`
                            answers a CSV of MANIFEST.output.fields with the versions as headers

A route module (build plan R-D1). Every route here reaches the model, so every one is admitted
(MODEL_ROUTES): it takes a place at the model before its body is read. The answer is the
Predictor's frame encoded by the contract's answer codec, generic over MANIFEST.output's field
kinds: there is no per-task branch in `serve/`. One row out per row in, in order; a window row
without its full context is answered `answered: false` with its warning (a 200, never a 422).
A CSV upload is read with the schema's types (`predict.read_rows`, the CLI's reader, stopping
one row past the cap) and held to the same row model as a JSON body.
"""
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import Response

from .problems import ContractError
from .routes_monitor import monitor_body
from .service import API_VERSION

__all__ = ["CAPABILITY", "ENDPOINTS", "MODEL_ROUTES", "OPEN_WHILE_UNREADY", "register"]

ENDPOINTS = (("POST", "/v1/predict"), ("POST", "/predict"), ("POST", "/v1/predict/csv"))
MODEL_ROUTES = (("POST", "/v1/predict"), ("POST", "/predict"), ("POST", "/v1/predict/csv"))
OPEN_WHILE_UNREADY = ()
CAPABILITY = "core"

NOT_CSV = "the upload is not a CSV table in the raw schema"


def register(app: FastAPI, state) -> None:
    # the request and response models are this bundle's, built from its records at start-up:
    # FastAPI reads them from the signatures below, so they are names bound here, not strings
    Request_ = state.contract.models.PredictRequest
    Response_ = state.contract.PredictResponse

    @app.post("/v1/predict")
    def predict_json(req: Request_, include: list[str] = Query(default=[])) -> Response_:
        return _answer(state, req, include)

    @app.post("/predict", include_in_schema=False)
    def predict_alias(req: Request_, include: list[str] = Query(default=[])) -> Response_:
        return _answer(state, req, include)

    @app.post("/v1/predict/csv")
    def predict_csv(request: Request, file: UploadFile = File(...),
                    include: list[str] = Query(default=[]), monitor: bool = False):
        return _csv(state, request, file, include, monitor)


def _answer(state, req, include) -> dict:
    predictor = state.ready()
    tables = state.contract.tables
    history = getattr(req, "history", None)
    out = predictor.predict(tables.frame(req.rows),
                            history=tables.frame(history) if history else None)
    body = state.envelope(len(out))
    body["predictions"] = state.contract.answers.rows(out, tuple(include))
    return body


def _csv(state, request: Request, file: UploadFile, include, monitor: bool):
    """A CSV upload answered as its rows would be in JSON; the truth it carries echoed."""
    predictor = state.ready()
    contract = state.contract
    try:
        # parsing stops one row past the cap: enough for the contract to refuse the table,
        # without reading the rest of an oversized upload into memory
        table = state.bundle.read_rows(file.file, state.root, nrows=contract.max_rows + 1)
    except (UnicodeDecodeError, ValueError) as exc:  # the parser's errors are ValueErrors
        raise ContractError(NOT_CSV, [{"column": None, "error": f"{type(exc).__name__}: "
                                       f"{str(exc)[:200]}"}]) from None
    frame, truth = contract.tables.accept(table)
    out = predictor.predict(frame)
    include = tuple(include)
    if "text/csv" in request.headers.get("accept", ""):
        return _csv_answer(state, contract.answers.table(out, truth, include), len(out))
    body = state.envelope(len(out))
    body["predictions"] = contract.answers.rows(out, include)
    if monitor and state.monitor is not None:
        body["monitor"] = monitor_body(state, frame, truth, out, True)
    if truth is not None:
        body["truth"] = truth
    if "rows" in include:
        body["rows"] = contract.tables.json_rows(frame)
    return body


def _csv_answer(state, table, n_rows: int) -> Response:
    """The CSV answer, with the versions every JSON answer carries as its headers."""
    stamp = state.envelope(n_rows)
    return Response(table.to_csv(index=False), media_type="text/csv", headers={
        "X-API-Version": API_VERSION, "X-Model-Version": str(stamp["model_version"]),
        "X-Contract-Version": str(stamp["contract_version"]),
        "X-Feature-Contract-Version": str(stamp["feature_contract_version"])})
