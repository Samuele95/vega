"""demo_pipeline.model.metrics — pure metric functions (the one FP module)."""
from __future__ import annotations

__all__ = ["accuracy", "mae"]


def accuracy(y_true, y_pred) -> float:
    pairs = list(zip(y_true, y_pred))
    return sum(1 for t, p in pairs if t == p) / max(len(pairs), 1)


def mae(y_true, y_score) -> float:
    pairs = list(zip(y_true, y_score))
    return sum(abs(t - s) for t, s in pairs) / max(len(pairs), 1)
