"""
serve.routes_monitor — {{PACKAGE}}'s drift readings over HTTP, through the {{PORT_MONITOR}} port.

    POST /v1/monitor    `{rows, history?, truth?}` (`labels` is the truth's permanent alias) ->
                        the envelope, `moved` (how many readings are watch or alert) and the
                        readings: the moved ones, or every one with `?all=true`

A route module (build plan R-D1); the route reaches the model (the Predictor answers the window
the output checks read), so it is admitted. The Monitor reads a window of raw rows with the
truth under the column MANIFEST.output.truth names — a class as its name, a number, or label
names, as the contract decoded them — and the Predictor's frame for those rows: one row per
row, a window row answered `false` left out by the monitor with a note. A bundle packed without
a monitor plan carries no reference: the route answers 404, a permanent state, never a 503 to
retry.
"""
import pandas as pd
from fastapi import FastAPI, Query
from starlette.exceptions import HTTPException as StarletteHTTPException

__all__ = ["CAPABILITY", "ENDPOINTS", "MODEL_ROUTES", "OPEN_WHILE_UNREADY", "monitor_body",
           "register"]

ENDPOINTS = (("POST", "/v1/monitor"),)
MODEL_ROUTES = (("POST", "/v1/monitor"),)
OPEN_WHILE_UNREADY = ()
CAPABILITY = "core"

MOVED = ("watch", "alert")
NO_REFERENCE = "this bundle ships no monitor reference"


def register(app: FastAPI, state) -> None:
    Request_ = state.contract.models.MonitorRequest
    Response_ = state.contract.MonitorResponse

    @app.post("/v1/monitor")
    def monitor(req: Request_, all: bool = Query(default=False)) -> Response_:
        predictor = state.ready()
        if state.monitor is None:
            raise StarletteHTTPException(404, NO_REFERENCE)
        tables = state.contract.tables
        frame = tables.frame(req.rows)
        history = getattr(req, "history", None)
        answers = predictor.predict(frame, history=tables.frame(history) if history else None)
        return monitor_body(state, frame, getattr(req, "truth", None), answers, all)


def monitor_body(state, frame, truth, answers, report: bool) -> dict:
    """The envelope and the Monitor's readings over one window: `frame` the raw rows the
    contract accepted, `truth` one per row or None, `answers` the Predictor's frame for them;
    `report` every reading, else the moved ones."""
    state.ready()
    if state.monitor is None:
        raise StarletteHTTPException(404, NO_REFERENCE)
    window = _with_truth(state, frame, truth)
    read = state.monitor.report if report else state.monitor.check
    readings = state.contract.readings(read(window, answers))
    body = state.envelope(len(frame))
    body["moved"] = sum(1 for r in readings if r["verdict"] in MOVED)
    body["readings"] = readings
    return body


def _with_truth(state, frame, truth):
    """The window as the Monitor reads it: the truth under its column (never over an input of
    the same name — an autoregressive target already is one)."""
    column = state.contract.models.truth.column
    if truth is None or not column or column in frame.columns:
        return frame
    values = pd.Series(list(truth), index=frame.index, dtype=object)
    return frame.assign(**{column: values})
