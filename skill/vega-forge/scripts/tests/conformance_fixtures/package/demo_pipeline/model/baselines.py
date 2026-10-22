"""
demo_pipeline.model.baselines — the baselines as Model implementations (role: baseline).

Several classes and a public `build`, by design: a baseline module holds the extra
implementations of the Model port (C2), so C14's one-class rule is not its rule; C11–C13 are.
"""
from __future__ import annotations

from ..domain import NetSpec

__all__ = ["MeanBaseline", "TrivialBaseline", "build"]


class TrivialBaseline:
    """Scores every row 0."""

    def score(self, batch: dict) -> tuple:
        return tuple(0.0 for _ in batch["x_num"])


class MeanBaseline:
    """Scores every row with one constant."""

    def __init__(self, mean: float = 0.0):
        self._mean = mean

    def score(self, batch: dict) -> tuple:
        return tuple(self._mean for _ in batch["x_num"])


def build(spec: NetSpec, which=("trivial", "mean")) -> tuple:
    table = {"trivial": TrivialBaseline, "mean": MeanBaseline}
    return tuple(table[name]() for name in which)
