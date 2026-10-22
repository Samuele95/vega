"""
{{PACKAGE}}.{{MODULE_MONITOR}}.monitor — the {{PORT_MONITOR}} adapter (role: Monitor).

{{RESPONSIBILITY_MONITOR}}
Source: {{SOURCE_MONITOR}}

Status: **pending: /vega-pack**.

What this monitor watches is not a designer's choice to make here: it is
`pipeline.json.monitoring[]` — what the analysis found fragile — and /vega-pack fills the
checks from it, with the thresholds /vega-eval's robustness slices measured. Inventing checks
now would be a monitor nobody gated, watching for what nobody found.

The blueprint declared {{N_MONITORS}} monitoring concern(s). They live in
`pipeline.json.monitoring[]`, and `blueprint-map.json` copies them into this file's entry as
`concerns`, so the pointer resolves without the bundle carrying the blueprint.
"""
from __future__ import annotations

__all__ = ["Monitor"]

_PENDING = (
    "pending: /vega-pack — the Monitor's checks come from pipeline.json.monitoring[] with "
    "/vega-eval's measured thresholds, and are written when the bundle is built."
)


class Monitor:
    """Pending: it declares the port's methods and the collaborators /vega-pack's adapter
    requires (the train reference, the recipe, the labels), so the stub is the port's shape and
    the one door to it, each method saying what fills it."""

    def __init__(self, reference, *, recipe, labels):
        raise NotImplementedError(_PENDING)

    def check(self, batch, predictions=None) -> tuple:
        raise NotImplementedError(_PENDING)

    def report(self, batch, predictions=None) -> tuple:
        raise NotImplementedError(_PENDING)
