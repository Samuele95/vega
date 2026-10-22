"""
inspection — the read-only view of the served network (inspect-contract §2, §6.3): support
modules inside the Predictor's module, map entries `kind: support`, `used_by: ["Predictor"]`.
They import `torch`, `numpy`, `pandas` and the package's `domain` only, never an adapter: the
model and the Predictor are handed in (`Predictor.inspector()`, the bundle's
`predict.load_inspector`).

  NetworkInspector   the facade (`inspector.py`): open, network, tensor, explain, attribute,
                     status; its `surface` (InspectionSurface) is what a service maps
  ExplainQuery       one explain request, parsed (`query.py`; `QueryRules.parse` builds it)

The rest are the facade's parts: the stage graph (`graph`, `wiring`, `view`, the readers and
their `registry`), the capture of one pass (`capture`, `focus`, `parts`, the probes), the
numbers (`codec`, `reduce`), the targets (`targets`, `target_kinds`), the attribution
(`attribution`, `attribution_parts`, `attribution_graph`, `interpolate`, `attribution_table`)
and the explanation body (`explanation`).
"""
from .explanation import FidelityError
from .inspector import (InspectionStatus, InspectionSurface, InspectionUnavailable,
                        NetworkInspector)
from .query import ExplainQuery, QueryError, QueryRules

__all__ = ["ExplainQuery", "FidelityError", "InspectionStatus", "InspectionSurface",
           "InspectionUnavailable", "NetworkInspector", "QueryError", "QueryRules"]
