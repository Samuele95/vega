"""
demo_pipeline.features.preprocess — the Recipe adapter (role: FeatureTransformer).

Status: pending: /vega-pack, until the stage copies the shaper's transform-only module here.
That module is functions — `transform`, `edges_for`, `edges_within` are the port (R-A3) — so
C14's class rule never reads this file; C10 reads its shape and C11–C13 its sizes.
"""
from __future__ import annotations

_PENDING = "pending: /vega-pack — the fitted transform is copied here when the bundle is built"


def transform(df_raw, history=None):
    raise NotImplementedError(_PENDING)
