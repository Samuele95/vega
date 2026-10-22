"""
serve.routes_inspect — {{PACKAGE}}'s network read rather than called: what it is and what one
sample does as it passes through it, from the {{PORT_PREDICTOR}}'s own network.

    GET  /v1/network?weights=full|summary   vega/network-view@2: the stage graph, the weights
                                            (inline, deferred or withheld), the task's readout;
                                            the bytes built once at start-up, per weights mode
                                            (default: the mode this service publishes)
    GET  /v1/network/tensor/{tensor_id}     one tensor as base64-le, from the live weights
    POST /v1/explain                        `{row | rows, history?, at?, target?, method?, steps?,
                                            detail?}` -> the envelope and `explanation`
                                            (vega/explanation@2): the sample's pass through every
                                            stage, its attribution, the fidelity checks; its
                                            `prediction` is the /v1/predict answer of the row
                                            (`?include=` shows the gated keys, as there)

A route module (build plan R-D1), capability `inspection`: pack renders it unless the bundle is
packed with `--no-inspect`, and MANIFEST.serve.endpoints then lists its ENDPOINTS. Explaining runs
the model, so POST /v1/explain is admitted (MODEL_ROUTES) and takes a place at the model before its
body is read; the two network routes read memory and never queue. None of them answers before the
self-test passed (503).

Opening (`on_ready`, once the self-test passed). The surface is the Predictor's own inspector
(`Predictor.inspector()`, inspect-contract §6.3) opened on the bundle's example rows
(`examples/rows.csv`, every sample up to 16 explained at summary detail — the example
`predict.load_inspector` opens on): the stage graph, the targets, the hooks, the network bytes of
both weights modes (each digest equal to the one pack pinned in MANIFEST.serve.inspection), the
fidelity of every example explanation. The package's code is imported only then, after the
service verified it: nothing of it runs at import. The surface — the inspector, the explain rules
(`QueryRules.of(spec, max_rows)`), the network bytes this service publishes — is kept in
`state.payloads["inspection"]`, and its status (`enabled`, `ready`, `why`, `weights`,
`network_digest`, `pinned`, `coverage`, `fidelity`, `recompute`) is `/health.inspection`. An
inspector that does not open never costs the model its readiness: the routes then answer 404,
naming why.

Switches (inspect-contract §7), read at start-up; a variable only restricts what the bundle grants:
MANIFEST.serve.inspection.enabled false or `{{ENV}}_INSPECT=off` answers the three routes 404
("this bundle ships without inspection", "inspection is switched off on this service");
MANIFEST.serve.inspection.weights `summary` or `{{ENV}}_INSPECT_WEIGHTS=summary` publishes every
tensor `withheld`, and `weights=full` and the tensor route answer 404. A value that is not one of
the switch's words restricts (off, summary), and the log says so.

Explaining. The request is held to the row model /v1/predict holds its rows to (a column named on
every refusal), its rows and history become the raw frames /v1/predict builds, and
`QueryRules.parse` checks the rest for the bundle's sample shape. Refusals: 422 for the request
and the query (`errors` names the request field: `row`, `rows`, `at`, `target.<selector>`,
`method`, `steps`, `detail`); 404 when inspection is off or unavailable, and for an unknown or
withheld tensor; 500, with no internals, for an explanation that does not reproduce the served
answer or an attribution that cannot be given (the log records the measurement).
"""
# No postponed annotations here: FastAPI must read the route signatures as objects.
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Query
from fastapi.responses import Response
from pydantic import ConfigDict, create_model
from starlette.exceptions import HTTPException as StarletteHTTPException

from .edge import problem
from .problems import ProblemDetail, thawed

__all__ = ["CAPABILITY", "ENDPOINTS", "MODEL_ROUTES", "OPEN_WHILE_UNREADY", "InspectionSurface",
           "Switch", "explain_request", "on_ready", "register"]

ENDPOINTS = (("GET", "/v1/network"), ("GET", "/v1/network/tensor/{tensor_id}"),
             ("POST", "/v1/explain"))
MODEL_ROUTES = (("POST", "/v1/explain"),)
OPEN_WHILE_UNREADY = ()
CAPABILITY = "inspection"

SECTION = "inspection"
WEIGHTS = ("full", "summary")
SWITCH_WORDS = ("on", "off")
EXAMPLE_ROWS = "examples/rows.csv"
PASSED_AS_SENT = ("at", "target", "method", "steps", "detail")
WARNING_SEP = "; "                 # the Predictor joins a row's warnings with this in its frame
NOT_SHIPPED = "this bundle ships without inspection"
SWITCHED_OFF = "inspection is switched off on this service"
UNAVAILABLE = "inspection is unavailable on this service"
WITHHELD = "this service does not publish raw weights"
REFUSED = "Inspection request refused"
UNFAITHFUL = ("the explanation does not reproduce the served answer; the server log records "
              "the measured differences")
NO_ATTRIBUTION = "the attribution could not be given; the server log records the cause"
log = logging.getLogger("{{CMD}}.serve")


def register(app: FastAPI, state) -> None:
    # the request model is this bundle's, built from its row model at start-up: FastAPI reads it
    # from the signature below, so it is a name bound here, not a string
    Request_ = explain_request(state.contract.models)

    def opened():
        # a dependency: solved before the body is validated, so a surface that is off or
        # unavailable answers 404 whatever the request holds
        return _surface(state)

    @app.get("/v1/network")
    def network(weights: Optional[str] = Query(default=None), surface=Depends(opened)):
        return surface.network_answer(weights)

    @app.get("/v1/network/tensor/{tensor_id}")
    def tensor(tensor_id: str, surface=Depends(opened)):
        return surface.tensor_answer(tensor_id)

    @app.post("/v1/explain")
    def explain(req: Request_, include: list[str] = Query(default=[]), surface=Depends(opened)):
        return surface.explain_answer(state, req, tuple(include))


def on_ready(state) -> None:
    """Open the surface once the self-test passed and report its status (`/health.inspection`);
    a surface that does not open is reported, never raised: the model stays ready."""
    switch = Switch.of(os.environ, state.manifest)
    surface, status = None, switch.closed()
    if switch.on:
        try:
            surface, status = InspectionSurface.open(state, switch)
        except Exception as exc:
            log.exception("the inspection surface could not be opened")
            status = switch.closed(f"{UNAVAILABLE}: opening it failed ({type(exc).__name__}); "
                                   "the server log records the cause", enabled=True)
    if surface is not None:
        state.payloads[SECTION] = surface
    state.report(SECTION, status)


def explain_request(models) -> type:
    """The explain request model: the rows held to the bundle's row model (`models.Row`), every
    other field passed as sent to `QueryRules.parse`, which names the field it refuses."""
    row, rows, sent = models.Row, list[models.Row], (Any, None)
    return create_model("ExplainRequest", __config__=ConfigDict(extra="forbid"),
                        row=(Optional[row], None), rows=(Optional[rows], None),
                        history=(Optional[rows], None), at=sent, target=sent, method=sent,
                        steps=sent, detail=sent)


def _surface(state) -> "InspectionSurface":
    """The opened surface, or the 404 naming why there is none (503 first while unready)."""
    state.ready()
    surface = state.payloads.get(SECTION)
    if surface is not None:
        return surface
    status = state.status.inspection or {}
    why = status.get("why") or "it did not open"
    raise StarletteHTTPException(404, why if status.get("enabled") is False
                                 else f"{UNAVAILABLE}: {why}")


def _word(env: Mapping, name: str, words: tuple, restrictive: str) -> Optional[str]:
    """A switch variable's word (lower case), None when unset; a word the switch does not know
    is read as its restrictive one."""
    value = str(env.get(name) or "").strip().lower()
    if not value:
        return None
    if value not in words:
        log.warning("%s=%r is not one of %s; read as %r", name, value, "/".join(words),
                    restrictive)
        return restrictive
    return value


@dataclass(frozen=True)
class Switch:
    """What this service publishes of the inspection surface: what the bundle grants
    (MANIFEST.serve.inspection `enabled`, `weights`) narrowed by `{{ENV}}_INSPECT` and
    `{{ENV}}_INSPECT_WEIGHTS`."""
    on: bool
    weights: str
    why: Optional[str] = None

    @staticmethod
    def of(env: Mapping, manifest: Mapping) -> "Switch":
        record = thawed((manifest.get("serve") or {}).get("inspection")) or {}
        granted = record.get("weights") or "full"
        asked = _word(env, "{{ENV}}_INSPECT_WEIGHTS", WEIGHTS, "summary")
        weights = "summary" if "summary" in (granted, asked) or granted not in WEIGHTS \
            else "full"
        if record.get("enabled", True) is False:
            return Switch(False, weights, NOT_SHIPPED)
        if _word(env, "{{ENV}}_INSPECT", SWITCH_WORDS, "off") == "off":
            return Switch(False, weights, SWITCHED_OFF)
        return Switch(True, weights)

    @property
    def modes(self) -> tuple:
        """The weights modes this service publishes."""
        return WEIGHTS if self.weights == "full" else ("summary",)

    def closed(self, why: Optional[str] = None, enabled: bool = False) -> dict:
        """The status of a surface that does not answer (inspect-contract §6.2's keys)."""
        return {"enabled": enabled, "ready": False, "why": why or self.why,
                "weights": self.weights, "network_digest": None, "pinned": False,
                "coverage": None, "fidelity": None, "recompute": None}


@dataclass(frozen=True)
class _Kit:
    """What the routes read of the opened inspector's surface (`inspector.surface`, the
    package's `InspectionSurface`): the rules and the query, and the exceptions each answer
    maps. Published by the inspector itself — no module is reached by reflection."""
    rules: type
    query: type
    refused: tuple                 # 422
    absent: tuple                  # 404
    unfaithful: tuple              # 500, the fidelity check
    failed: tuple                  # 500, the attribution

    @staticmethod
    def of(inspector) -> "_Kit":
        s = inspector.surface
        return _Kit(s.rules, s.query, s.refused, s.absent, s.unfaithful, s.failed)


@dataclass(frozen=True, eq=False)
class InspectionSurface:
    """The opened surface: the inspector, the explain rules, the network bytes per published
    weights mode, and the weights mode this service publishes."""
    inspector: Any
    kit: _Kit
    rules: Any
    network: Mapping
    weights: str

    @staticmethod
    def open(state, switch: Switch) -> tuple:
        """(the surface, its status), or (None, the status saying why not)."""
        try:
            inspector = state.predictor.inspector()
        except LookupError as exc:          # a Predictor's module without inspection modules
            return None, switch.closed(str(exc))
        kit = _Kit.of(inspector)
        inspector.open(_example(state, kit))
        status = dict(inspector.status(), weights=switch.weights)
        if not status["ready"]:
            return None, status
        try:
            rules = kit.rules.of(state.predictor.records.spec, state.contract.max_rows)
            network = {mode: inspector.network_bytes(mode) for mode in switch.modes}
        except Exception:
            inspector.close()
            raise
        return InspectionSurface(inspector, kit, rules, network, switch.weights), status

    def network_answer(self, weights: Optional[str]) -> Response:
        """GET /v1/network: the bytes built at opening for the weights mode asked."""
        mode = self.weights if weights is None else weights
        if mode not in WEIGHTS:
            return _refusal("weights", f"`weights` is full or summary, not {weights!r}")
        body = self.network.get(mode)
        if body is None:
            raise StarletteHTTPException(404, WITHHELD)
        return Response(body, media_type="application/json")

    def tensor_answer(self, tensor_id: str) -> Response:
        """GET /v1/network/tensor/{id}: one tensor, read from the live weights."""
        if self.weights != "full":
            raise StarletteHTTPException(404, WITHHELD)
        try:
            payload = self.inspector.tensor(tensor_id)
        except self.kit.absent as exc:
            raise StarletteHTTPException(404, str(exc)) from None
        return _json(payload)

    def explain_answer(self, state, req, include: tuple) -> Response:
        """POST /v1/explain: the envelope and the explanation, its `prediction` the row's
        /v1/predict answer."""
        try:
            query = self.rules.parse(_fields(state.contract.tables, req, self.rules.shape))
            body = self.inspector.explain(query)
        except self.kit.refused as exc:
            return _refusal(getattr(exc, "field", None) or "body", str(exc))
        except self.kit.absent as exc:
            raise StarletteHTTPException(404, str(exc)) from None
        except self.kit.unfaithful:
            log.exception("explain: the explanation does not reproduce the served answer")
            return problem(ProblemDetail(500, "Internal error", UNFAITHFUL))
        except self.kit.failed:
            log.exception("explain: the attribution could not be given")
            return problem(ProblemDetail(500, "Internal error", NO_ATTRIBUTION))
        body["prediction"] = _prediction(state.contract, body["prediction"], include)
        out = state.envelope(1)
        out["explanation"] = body
        return _json(out)


def _example(state, kit: _Kit):
    """The example query the surface opens on: the bundle's example rows, the explained row the
    last (every sample the rows make, up to 16, is explained at summary detail)."""
    rows = state.bundle.read_rows(state.root / EXAMPLE_ROWS, state.root).reset_index(drop=True)
    return kit.query(rows, None, max(len(rows) - 1, 0), detail="summary")


def _fields(tables, req, shape: str) -> dict:
    """The request as `QueryRules.parse` reads it: rows and history as the raw frames
    /v1/predict builds from validated rows (tokens decoded, codes as text, the truth and the
    ignored columns dropped), the other fields as sent. A row sample's `row` is its one-row
    frame, unless `rows` came too (which the rules refuse)."""
    fields = {name: getattr(req, name) for name in PASSED_AS_SENT}
    for name in ("rows", "history"):
        value = getattr(req, name)
        fields[name] = None if value is None else tables.frame(value) if value else []
    if req.row is not None:
        if shape == "row" and req.rows is None:
            fields["rows"] = tables.frame([req.row])
        else:
            fields["row"] = req.row.model_dump(by_alias=True)
    return {name: value for name, value in fields.items() if value is not None}


def _prediction(contract, served: Mapping, include: tuple) -> dict:
    """The explained row's answer exactly as /v1/predict writes it: the served frame's row
    (its warnings joined back into the frame's one text) through the contract's answer codec,
    held to this bundle's `RowPrediction`."""
    row = dict(served)
    if isinstance(row.get("warnings"), list):
        row["warnings"] = WARNING_SEP.join(str(w) for w in row["warnings"])
    answer = contract.answers.rows(pd.DataFrame([row]), include)[0]
    return contract.answers.response_model().model_validate(answer).model_dump(mode="json")


def _refusal(field: str, detail: str):
    """A 422 problem naming the request field at fault."""
    return problem(ProblemDetail(422, REFUSED, detail,
                                 ({"column": None, "field": field, "error": detail},)))


def _json(doc) -> Response:
    """A JSON answer; a non-finite number is a defect (500), never a NaN on the wire."""
    text = json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return Response(text.encode("utf-8"), media_type="application/json")
