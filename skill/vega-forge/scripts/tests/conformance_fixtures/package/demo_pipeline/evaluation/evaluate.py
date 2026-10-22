"""demo_pipeline.evaluation.evaluate — the Evaluator adapter (role: Evaluator)."""
from __future__ import annotations

from ..domain import Metrics, NetSpec, Part

__all__ = ["Evaluator"]


class Evaluator:
    """One sealed look, through the ports it is given (constructor injection)."""

    def __init__(self, data, store, build):
        self._data, self._store, self._build = data, store, build

    def score_part(self, spec: NetSpec, part: Part, checkpoint=None, model=None) -> Metrics:
        model = model or self._store.load(checkpoint, spec=spec, build=self._build)
        rows = self._data.load_part(spec, part)
        return Metrics(Part(part), tuple(model.score({"x_num": [r["x"] for r in rows]})))

    def score_sealed_test(self, spec: NetSpec, token: str, checkpoint=None, model=None,
                          seal_path: str = "test_seal.json") -> Metrics:
        model = model or self._store.load(checkpoint, spec=spec, build=self._build)
        rows = self._data.load_sealed_part(spec, token, seal_path)
        return Metrics(Part.TEST, tuple(model.score({"x_num": [r["x"] for r in rows]})))
