"""
serve.service — the state every route module of {{PACKAGE}}'s service reads, and the start-up
that fills it.

    NotReady                           raised by `ready()`; the service answers 503
    ServiceStatus                      what the service is at one moment (ready or not and why,
                                       the self-test, the `inspection` and `showcase` sections),
                                       replaced whole, never edited
    ServiceState.start(root, limits)   the bundle verified (code, evidence, data), its contract
    state.lifespan(modules)            the start-up: the ports, the self-test, each route
                                       module's `on_ready(state)`, then the status swap
    state.ready() -> Predictor         or NotReady
    state.envelope(n) -> dict          the version stamp every answer carries

A route module writes into the state only at start-up: `payloads` (the bytes it serves, built
once in its `on_ready`), `page_override` (a callable answering `GET /` with a Response, or None,
in place of the console page — the showcase slot sets it) and its status section (`report`).
After start-up nothing is written: the predictor and the monitor are loaded once and shared by
every request (inference in eval mode under no_grad; the recipe keeps its unseen-category state
per thread), and the status value is swapped whole, which every worker thread sees atomically.

The self-test is the bundle's own `check_examples` — the comparison `predict.py --check`, the
round-trip test and the notebook make — run on the loaded predictor: the service becomes ready
only once this install (its torch build included) reproduces the stored examples. A module of
capability `core` or `code` failing its `on_ready` leaves the service unready; any other (the
inspection surface, the showcase) is logged and reported in its own status section, and never
costs the model its readiness.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import startup
from .contract import ApiContract
from .problems import frozen, thawed

__all__ = ["API_VERSION", "NotReady", "ServiceState", "ServiceStatus"]

API_VERSION = "v1"
CORE_CAPABILITIES = ("core", "code")
SECTIONS = ("inspection", "showcase")
MONITOR_REFERENCE = "monitor-reference.json"
STARTING = "the service is starting"
# the contract the routes are registered with when schema.json cannot be read: the service is
# unready then and no request reaches a model, but every route still needs its request models
PLACEHOLDER_OUTPUT = {"reading": "unavailable", "fields": ["answered", "warnings"],
                      "field_kinds": {"answered": {"kind": "answered"},
                                      "warnings": {"kind": "warnings"}}}
log = logging.getLogger("{{CMD}}.serve")


class NotReady(RuntimeError):
    """The service cannot answer yet (or at all): its status says why."""


@dataclass(frozen=True)
class ServiceStatus:
    """What the service is, at one moment: ready or not, and why. Replaced whole (one reference
    swap, which every worker thread sees atomically), never edited in place."""
    ready: bool
    model_version: Optional[str]
    contract_version: Optional[str]
    feature_contract_version: Optional[str] = None
    self_test: Optional[Mapping] = None
    started_at: Optional[str] = None
    error: Optional[str] = None
    max_inflight: Optional[int] = None
    inspection: Optional[Mapping] = None
    showcase: Optional[Mapping] = None

    def __post_init__(self):
        for name in ("self_test", *SECTIONS):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, frozen(_finite(value)))

    def to_dict(self) -> dict:
        return {"ready": self.ready, "model_version": self.model_version,
                "contract_version": self.contract_version,
                "feature_contract_version": self.feature_contract_version,
                "self_test": thawed(self.self_test), "started_at": self.started_at,
                "error": self.error, "max_inflight": self.max_inflight,
                "inspection": thawed(self.inspection), "showcase": thawed(self.showcase)}


class ServiceState:
    """One bundle's service: its files verified at `start`, its ports loaded at `warm`."""

    def __init__(self, root: Path, manifest: Mapping, contract: ApiContract, limits: Any,
                 bundle: Any):
        self.root, self.manifest, self.contract = Path(root), frozen(dict(manifest)), contract
        self.limits, self.bundle = limits, bundle
        self.refusal: Optional[str] = None
        self.predictor = self.monitor = None
        self.payloads: dict = {}
        self.page_override: Optional[Callable] = None
        self.sections: dict = {}
        self.status = ServiceStatus(ready=False, model_version=manifest.get("model_version"),
                                    contract_version=contract.version,
                                    feature_contract_version=contract.feature_version,
                                    error=STARTING, max_inflight=limits.max_inflight)

    @staticmethod
    def start(root: Path, limits) -> "ServiceState":
        """The bundle's code, evidence and data verified, and its contract built. Every reason
        not to serve becomes the state's `refusal` (the service then starts unready)."""
        manifest = startup.read_manifest(root)
        bundle, why = startup.verified(root, manifest)
        contract, broken = _contract(root, manifest, limits.max_rows)
        state = ServiceState(root, manifest, contract, limits, bundle)
        state.refuse(why or broken)
        return state

    def refuse(self, why: Optional[str]) -> None:
        """Keep the first reason the service must not serve (None: nothing to keep)."""
        if why and not self.refusal:
            self.refusal = why
            self.status = replace(self.status, error=why)

    def report(self, section: str, value: Mapping) -> None:
        """A route module's status section (`inspection`, `showcase`), in /health from now on."""
        if section not in SECTIONS:
            raise ValueError(f"the service status has no section {section!r} ({SECTIONS})")
        self.sections[section] = value
        self.status = replace(self.status, **{section: value})

    def showcase_dir(self) -> Path:
        """Where the showcase slot is: {{ENV}}_SHOWCASE (`--showcase DIR` sets it), else
        `<bundle>/showcase`."""
        named = os.environ.get("{{ENV}}_SHOWCASE")
        return Path(named).resolve() if named else self.root / "showcase"

    def lifespan(self, modules: tuple):
        """The FastAPI lifespan: torch's intra-op pool capped, then `warm(modules)`."""
        @asynccontextmanager
        async def run(app):
            import torch
            torch.set_num_threads(self.limits.torch_threads)
            self.warm(modules)
            log.info(json.dumps({"event": "startup", **self.status.to_dict()}))
            yield
        return run

    def warm(self, modules: tuple) -> None:
        """Load the ports, run the self-test and the modules' `on_ready`, swap the status. Never
        raises: a failure is the unready status, named in /health (the log has the detail)."""
        self_test, error = None, None
        try:
            if self.refusal:
                raise NotReady(self.refusal)
            self_test = self._load()
            _passed(self_test)
            self._on_ready(modules)
        except NotReady as exc:
            error = str(exc)
            log.error(error)
        except Exception as exc:
            # /health is public: it names the kind of failure, the server log has the detail
            error = f"start-up failed ({type(exc).__name__}); the server log records the cause"
            log.exception("start-up failed")
        if error:
            self.predictor = self.monitor = None
        self.status = ServiceStatus(
            ready=error is None, model_version=self.manifest.get("model_version"),
            contract_version=self.contract.version,
            feature_contract_version=self.contract.feature_version, self_test=self_test,
            error=error, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            max_inflight=self.limits.max_inflight, **self.sections)

    def ready(self):
        """The predictor, or NotReady naming why there is none."""
        if not self.status.ready:
            raise NotReady(self.status.error or STARTING)
        return self.predictor

    def envelope(self, n_rows: int) -> dict:
        """The version stamp of an answer of `n_rows` rows: the API, the model, the input and
        feature contracts, and the reading (`output_kind`)."""
        return self.contract.envelope(self.status.model_version, n_rows, API_VERSION)

    def _load(self) -> dict:
        """The Predictor and the Monitor, built by the bundle's own `predict.py`, and the
        self-test over them (`check_examples`: the files verified again, the examples
        answered by this process's predictor)."""
        self.predictor = self.bundle.load(self.root)
        reference = self.root / MONITOR_REFERENCE
        self.monitor = self.bundle.load_monitor(self.root) if reference.is_file() else None
        return dict(self.bundle.check_examples(self.root, self.predictor))

    def _on_ready(self, modules: tuple) -> None:
        """Each module's `on_ready(state)`, in order, once the self-test passed."""
        for module in modules:
            hook = getattr(module, "on_ready", None)
            if hook is None:
                continue
            capability = getattr(module, "CAPABILITY", "core")
            try:
                hook(self)
            except Exception as exc:
                if capability in CORE_CAPABILITIES:
                    raise
                log.exception("%s: on_ready failed", module.__name__)
                self.sections.setdefault(capability, {
                    "ready": False, "why": f"start-up failed ({type(exc).__name__}); the "
                                           "server log records the cause"})


def _passed(self_test: Mapping) -> None:
    """NotReady, naming the first problems, when the self-test did not pass."""
    if not self_test.get("passed"):
        why = "; ".join(str(p) for p in self_test.get("problems") or ()) or "no detail"
        raise NotReady("the shipped examples were not reproduced within the manifest's "
                       f"tolerance; this install does not serve the evaluated model ({why})")


def _contract(root: Path, manifest: dict, max_rows: int):
    """(the input contract, None), or (a placeholder, why) when schema.json cannot be read."""
    try:
        return ApiContract.from_bundle(root, max_rows=max_rows), None
    except Exception as exc:
        log.exception("the input contract could not be built")
        placeholder = ApiContract({"input": []}, {"output": PLACEHOLDER_OUTPUT},
                                  max_rows=max_rows)
        return placeholder, f"schema.json could not be read ({type(exc).__name__})"


def _finite(value):
    """A status value as JSON can carry it: a NaN or an infinity (a failed comparison's
    distance) is null."""
    if isinstance(value, Mapping):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
