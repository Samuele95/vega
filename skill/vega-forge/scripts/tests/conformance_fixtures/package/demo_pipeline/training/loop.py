"""demo_pipeline.training.loop — the Trainer adapter (role: Trainer)."""
from __future__ import annotations

from ..domain import Metrics, NetSpec, Part, RunContext

__all__ = ["Trainer"]


class Trainer:
    """Fit a Model on a DataSource under a spec."""

    def fit(self, model, spec: NetSpec, data, callbacks=None, run: RunContext | None = None):
        run = run or RunContext()
        rows = data.load_part(spec, Part.TRAIN)
        for epoch in range(_epochs(run)):
            self._epoch(model, rows, epoch)
            if callbacks is not None:
                callbacks.on_epoch(epoch)
        return self.evaluate(model, spec, data.load_part(spec, Part.VALID))

    def evaluate(self, model, spec: NetSpec, loader) -> Metrics:
        return Metrics(Part.VALID, tuple(model.score({"x_num": [r["x"] for r in loader]})))

    def _epoch(self, model, rows, epoch: int) -> int:
        return len(rows) + epoch


def _epochs(run: RunContext) -> int:
    return 1 if run.budget_min is None else max(1, int(run.budget_min))
