"""
The backend registry — one line per backend, per backend-contract.md §3.

A backend is a package under `scripts/backends/<name>/` exposing `emit(spec, pipeline,
out_dir, **kw)` (in its `emit.py`) and declaring its block, loss, optimizer, schedule, metric
and export coverage in `coverage.json`. Registering it here is the only change the rest of
the sub-chain needs: `/vega-train`, `/vega-eval` and `/vega-pack` talk to the emitted code
through the interface of §1, never to a framework.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

REGISTERED = ("pytorch",)


def available():
    """Backend names that are both registered and present on disk."""
    return tuple(n for n in REGISTERED if (HERE / n / "coverage.json").is_file())


def coverage(name):
    p = HERE / name / "coverage.json"
    if not p.is_file():
        raise ValueError(f"backend {name!r} declares no coverage.json at {p}; a backend that "
                         "does not say what it covers cannot be checked against a spec")
    return json.loads(p.read_text(encoding="utf-8"))


def load(name):
    """Import `backends/<name>/emit.py` and return its module (it exposes `emit`)."""
    if name not in REGISTERED:
        raise ValueError(f"backend {name!r} is not registered; known backends are "
                         f"{', '.join(REGISTERED)}. Add one under scripts/backends/<name>/ "
                         "with emit.py + coverage.json and name it in REGISTERED.")
    p = HERE / name / "emit.py"
    if not p.is_file():
        raise ValueError(f"backend {name!r} is registered but {p} is missing")
    spec = importlib.util.spec_from_file_location(f"vega_backend_{name}", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "emit"):
        raise ValueError(f"{p} exposes no emit(spec, pipeline, out_dir)")
    return mod
