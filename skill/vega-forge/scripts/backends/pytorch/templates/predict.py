"""
{{PACKAGE}}.serving.predict — the {{PORT_PREDICTOR}} adapter (role: Predictor).

{{RESPONSIBILITY_PREDICTOR}}
Source: {{SOURCE_PREDICTOR}}

Status: **pending: /vega-pack**.

The predictor is the bundle's front door: raw rows -> the Recipe -> the Model -> predictions
with their warnings. It cannot be completed here because two of the three pieces do not exist
yet: the fitted transform (pending, see features/preprocess.py) and the selected checkpoint
(/vega-train chooses it, /vega-eval seals the verdict, /vega-pack pins both by digest).

Emitting the stub rather than a guess is the point: `blueprint-map.json` records this port as
`pending: /vega-pack`, so the map says what the code says.

Serving mode declared by the blueprint: {{SERVING_MODE}}
"""
from __future__ import annotations

__all__ = ["Predictor"]

_PENDING = (
    "pending: /vega-pack — the Predictor is completed when the bundle is built, from the "
    "fitted recipe, the selected checkpoint and the threshold /vega-eval chose. Run "
    "/vega-pack on the evaluated model."
)


class Predictor:
    """Raw rows -> predictions, through the Recipe and the Model. Pending: it declares the
    port's methods and the collaborators /vega-pack's adapter requires (the network, the fitted
    transform, the serving records; the context optional), so the stub is the port's shape and
    the one door to it, and every method says what fills it."""

    def __init__(self, model, transform, records, context=None):
        raise NotImplementedError(_PENDING)

    def predict(self, df_raw, history=None):
        raise NotImplementedError(_PENDING)

    def predictions(self, df_raw, history=None) -> tuple:
        raise NotImplementedError(_PENDING)

    def predict_one(self, row: dict, history=None) -> dict:
        raise NotImplementedError(_PENDING)
