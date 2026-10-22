"""demo_pipeline.serving.monitor — the Monitor adapter (role: Monitor). Status: pending: /vega-pack."""
from __future__ import annotations

__all__ = ["Monitor"]

_PENDING = "pending: /vega-pack — the Monitor is filled from pipeline.json.monitoring[]"


class Monitor:
    """Watch what the analysis found fragile."""

    def __init__(self, reference: dict):
        raise NotImplementedError(_PENDING)

    def check(self, batch, predictions=None) -> tuple:
        raise NotImplementedError(_PENDING)
