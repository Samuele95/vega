"""
demo_pipeline.serving.predict — the Predictor adapter (role: Predictor).

Status: pending: /vega-pack. The stub already has the shape C14 reads once the stage fills it:
the adapter class, a `_prefixed` constant, imports.
"""
from __future__ import annotations

__all__ = ["Predictor"]

_PENDING = "pending: /vega-pack — the Predictor is completed when the bundle is built"


class Predictor:
    """Raw rows -> predictions, through the Recipe and the Model."""

    def __init__(self, model, transform, records):
        raise NotImplementedError(_PENDING)

    def predict(self, rows, history=None):
        raise NotImplementedError(_PENDING)
