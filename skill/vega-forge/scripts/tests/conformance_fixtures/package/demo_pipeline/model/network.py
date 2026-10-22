"""demo_pipeline.model.network — the Model adapter (role: Model)."""
from __future__ import annotations

from ..domain import NetSpec

__all__ = ["Network", "build_model"]

_WIDTH = 4


class Network:
    """Features -> outputs; the parameters as one value."""

    def __init__(self, weights: tuple, spec_name: str):
        self._weights = weights
        self._spec_name = spec_name

    @classmethod
    def build(cls, spec: NetSpec, run_dir=None) -> "Network":
        """The one constructor of the network (C9)."""
        return cls(_initial_weights(spec.seed), spec.name)

    def score(self, batch: dict) -> tuple:
        return tuple(sum(w * x for w, x in zip(self._weights, row)) for row in batch["x_num"])

    def param_count(self) -> tuple:
        return len(self._weights), len(self._weights)


def _initial_weights(seed: int) -> tuple:
    return tuple(float((seed + i) % 3) for i in range(_WIDTH))


build_model = Network.build
