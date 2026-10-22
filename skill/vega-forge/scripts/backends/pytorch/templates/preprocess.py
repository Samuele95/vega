"""
{{PACKAGE}}.{{MODULE_FEATURES}}.preprocess — the {{PORT_FEATURETRANSFORMER}} adapter
(role: FeatureTransformer).

{{RESPONSIBILITY_FEATURETRANSFORMER}}
Source: {{SOURCE_FEATURETRANSFORMER}}

Status: **pending: /vega-pack**.

The recipe that turns raw rows into shaped features already exists — it is /dataset-shaper's
fitted transform-only module, and /vega-pack copies it here when it builds the bundle. Until
then `{{MODULE_INGESTION}}/data.py` reads the shaped parts the shaper already wrote, so nothing
in training needs this port; and a re-implementation of the recipe here would be a second copy
of the fitted parameters, which is exactly the drift the shaper's manifest exists to prevent.

The contract /vega-pack fills is the port's, every method and parameter of it (`ports.py`):

{{TRANSFORMER_SIGNATURE}}

The shaper's transform-only module states it as module functions — its public functions ARE
the port, read structurally (code-contract §5, C10; the class rule C14 exempts this role), so
the stub below is those functions, generated from the port's signature, each raising the
pending message. `transform(..., with_unseen=True)` also returns, per row, the columns whose
category the fitted recipe never saw: the predictions boundary's unseen-column list (empty,
never null) is produced from it, and the Monitor's unseen-category signal reads it.
"""
from __future__ import annotations

__all__ = [{{TRANSFORMER_NAMES}}]

_PENDING = (
    "pending: /vega-pack — the FeatureTransformer adapter is filled by /vega-pack from "
    "dataset-shaper's `scripts/shape.py --transform-only` output (the fitted recipe), never "
    "written by hand here. Until the bundle is built, read the shaped parts through "
    "{{MODULE_INGESTION}}/data.py."
)


{{TRANSFORMER_STUBS}}
