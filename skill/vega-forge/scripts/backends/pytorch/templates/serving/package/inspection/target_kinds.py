"""
inspection.target_kinds — the eleven target kinds of an explanation, one strategy each
(inspect-contract §4.2; build plan R-B7, R-C5, R-C6, R-F6), behind `targets.TaskTargets`.

A kind resolves a request's selector (`requested`: the class, label, threshold, quantile index
or level, feature or step a request names, or the `QueryError` that refuses it), picks the
task's default for one sample (`picked`: from the head's outputs, never from the truth), forms
the differentiable scalar `s` over a batch (`scalar`, through the network's `TaskReadout` where
the reading is the readout's: the value, the row error, the svdd distance, the forecast steps)
and says which `of` a served field answering it carries (`served_of`, matched by `of_matches`
against `MANIFEST.output.field_kinds`). `TargetFacts` is what they read: the effective kind and
loss, the classes (binary: `[negative, positive]`), the thresholds, quantiles, horizon, the
features and the reconstruction layout (numeric columns, then each nominal's `cardinality`
logits), the readout and the network. `feature_errors` is the reconstruction error term by term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from .query import QueryError

__all__ = ["ANY", "KIND_TABLE", "KINDS", "TargetFacts", "feature_errors", "of_matches"]

# target kind -> (its request selector, its unit): inspect-contract §4.2
KINDS = MappingProxyType({
    "logit": ("class", "logit"), "binary_logit": ("class", "logit"),
    "label_logit": ("label", "logit"), "value": (None, "target"),
    "quantile": ("quantile", "target"), "expected_rank": (None, "rank"),
    "threshold_logit": ("threshold", "logit"), "reconstruction_error": (None, "error"),
    "feature_error": ("feature", "error"), "svdd_distance": (None, "distance"),
    "forecast_step": ("step", "target")})
ANY = ("any",)


@dataclass(frozen=True, eq=False)
class TargetFacts:
    """What the targets read: the effective kind and loss, the classes (binary: [negative,
    positive]), the thresholds, the quantiles, the horizon, the features and the recon layout,
    the network's readout and its anatomy (the Model port's `anatomy()`: the svdd centre the
    readout measures a distance from)."""

    kind: str
    loss: str | None
    classes: tuple
    positive: str | None
    threshold: float | None
    per_label: Mapping
    quantiles: tuple
    horizon: int
    features: tuple
    cards: tuple
    readout: Any
    anatomy: Any

    @property
    def n_numeric(self) -> int:
        return len(self.features) - len(self.cards)

    @property
    def median(self) -> int:
        """The quantile column nearest 0.5 (`TaskReadout`'s point estimate)."""
        return min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))

    @property
    def negative(self) -> str | None:
        return next((c for c in self.classes if c != self.positive), None)

    def cut(self, label=None) -> float:
        """The decision threshold (probability scale) of the binary class or of one label."""
        if label is not None and str(label) in self.per_label:
            return float(self.per_label[str(label)])
        return 0.5 if self.threshold is None else float(self.threshold)


def _named(names, value, where: str) -> int:
    names = [str(n) for n in names]
    if str(value) not in names:
        raise QueryError(where, f"{where} {value!r} is not one of the task's outputs "
                                f"({', '.join(names)})")
    return names.index(str(value))


def feature_errors(f: TargetFacts, head, batch):
    """(B, F) the terms of the reconstruction error, feature by feature: `(x̂_f − x_f)²` per
    numeric column, then the cross-entropy of each nominal's logits slice (`recon_dim` order);
    the row error is their numeric mean plus their nominal mean."""
    n, cols = f.n_numeric, []
    if n:
        cols.append((head[..., :n] - batch["x_num"][..., :n].to(head.dtype)).pow(2))
    at = n
    for i, card in enumerate(f.cards):
        ce = F.cross_entropy(head[..., at: at + card], batch["x_cat"][..., i].long(),
                             reduction="none")
        cols.append(ce.unsqueeze(-1))
        at += int(card)
    return torch.cat(cols, dim=-1)


# ------------------------------------------------------------------------------ the kinds

class _Kind:
    """One target kind: a requested selector resolved, a default picked, the scalar, and the
    `of` a served field answering it carries."""

    NAME = ""

    def requested(self, f: TargetFacts, value) -> tuple:
        raise QueryError("target", f"a {self.NAME} target takes no selector")

    def picked(self, f: TargetFacts, z, batch, k: int) -> tuple:
        return None, None, 1, "default"

    def scalar(self, f: TargetFacts, target, head, batch):
        raise NotImplementedError

    def served_of(self, f: TargetFacts, target):
        return ANY


class _Logit(_Kind):
    NAME = "logit"

    def requested(self, f, value):
        i = _named(f.classes, value, "target.class")
        return i, f.classes[i], 1

    def picked(self, f, z, batch, k):
        i = int(z[k].argmax())
        return i, f.classes[i], 1, "predicted"

    def scalar(self, f, target, head, batch):
        return head[:, target.index]

    def served_of(self, f, target):
        return "text", str(target.name)


class _BinaryLogit(_Logit):
    NAME = "binary_logit"

    def requested(self, f, value):
        i = _named(f.classes, value, "target.class")
        return i, f.classes[i], 1 if f.classes[i] == f.positive else -1

    def picked(self, f, z, batch, k):
        name = f.positive if float(torch.sigmoid(z[k, 0])) >= f.cut() else f.negative
        return f.classes.index(name), name, 1 if name == f.positive else -1, "predicted"

    def scalar(self, f, target, head, batch):
        return target.sign * head[:, 0]

    def served_of(self, f, target):
        return "text", str(f.positive)


class _LabelLogit(_Logit):
    NAME = "label_logit"

    def requested(self, f, value):
        j = _named(f.classes, value, "target.label")
        return j, f.classes[j], 1


class _Value(_Kind):
    NAME = "value"

    def picked(self, f, z, batch, k):
        return (f.median if f.quantiles else 0), "value", 1, "default"

    def scalar(self, f, target, head, batch):
        return f.readout.score(head, batch, f.anatomy)


class _Quantile(_Kind):
    NAME = "quantile"

    def requested(self, f, value):
        levels = [f"{q:g}" for q in f.quantiles]
        if isinstance(value, float):
            hits = [i for i, q in enumerate(f.quantiles) if math.isclose(q, value, abs_tol=1e-9)]
        else:
            hits = [int(value)] if 0 <= int(value) < len(f.quantiles) else []
        if not hits:
            raise QueryError("target.quantile", f"target.quantile {value!r} is none of the "
                                                f"task's quantiles ({', '.join(levels)})")
        return hits[0], f"q{levels[hits[0]]}", 1

    def picked(self, f, z, batch, k):
        return f.median, f"q{f.quantiles[f.median]:g}", 1, "default"

    def scalar(self, f, target, head, batch):
        return head[:, target.index]

    def served_of(self, f, target):
        return "level", float(f.quantiles[target.index])


class _ExpectedRank(_Kind):
    NAME = "expected_rank"

    def scalar(self, f, target, head, batch):
        return torch.sigmoid(head).sum(dim=-1)

    def served_of(self, f, target):
        return None


class _ThresholdLogit(_Logit):
    NAME = "threshold_logit"

    def requested(self, f, value):
        j = int(value)
        if not 0 <= j < len(f.classes) - 1:
            raise QueryError("target.threshold", f"target.threshold {value!r} is outside the "
                                                 f"{len(f.classes) - 1} thresholds (0…"
                                                 f"{len(f.classes) - 2})")
        return j, f">{f.classes[j]}", 1

    def picked(self, f, z, batch, k):
        passed = int((torch.sigmoid(z[k]) >= 0.5).sum())
        j = min(passed, len(f.classes) - 2)
        return j, f">{f.classes[j]}", 1, "predicted"

    def served_of(self, f, target):
        return "text", str(f.classes[target.index])


class _ReconstructionError(_Kind):
    NAME = "reconstruction_error"

    def scalar(self, f, target, head, batch):
        return f.readout.row_score(head, batch, f.anatomy)


class _FeatureError(_Kind):
    NAME = "feature_error"

    def requested(self, f, value):
        i = _named(f.features, value, "target.feature")
        return i, f.features[i], 1

    def picked(self, f, z, batch, k):
        i = int(feature_errors(f, z, batch)[k].argmax())
        return i, f.features[i], 1, "predicted"

    def scalar(self, f, target, head, batch):
        return feature_errors(f, head, batch)[:, target.index]

    def served_of(self, f, target):
        return None


class _SvddDistance(_Kind):
    NAME = "svdd_distance"

    def scalar(self, f, target, head, batch):
        return f.readout.score(head, batch, f.anatomy)


class _ForecastStep(_Kind):
    NAME = "forecast_step"

    @staticmethod
    def _at(f, h: int) -> tuple:
        """(head column, output name) of step h: the median column under a quantile loss."""
        if not f.quantiles:
            return h - 1, f"t+{h}"
        return (h - 1) * len(f.quantiles) + f.median, f"t+{h}/q{f.quantiles[f.median]:g}"

    @staticmethod
    def step(f, index: int) -> int:
        return (index // len(f.quantiles) if f.quantiles else index) + 1

    def requested(self, f, value):
        if not 1 <= int(value) <= f.horizon:
            raise QueryError("target.step", f"target.step {value!r} is outside the horizon "
                                            f"1…{f.horizon}")
        return (*self._at(f, int(value)), 1)

    def picked(self, f, z, batch, k):
        return (*self._at(f, 1), 1, "default")

    def scalar(self, f, target, head, batch):
        steps = f.readout.score(head, batch, f.anatomy)
        return steps.reshape(steps.shape[0], -1)[:, self.step(f, target.index) - 1]

    def served_of(self, f, target):
        q = float(f.quantiles[f.median]) if f.quantiles else None
        return "step", self.step(f, target.index), q


KIND_TABLE = MappingProxyType({k.NAME: k for k in (
    _Logit(), _BinaryLogit(), _LabelLogit(), _Value(), _Quantile(), _ExpectedRank(),
    _ThresholdLogit(), _ReconstructionError(), _FeatureError(), _SvddDistance(),
    _ForecastStep())})


def of_matches(of, want) -> bool:
    """Whether a served field's `of` is the one a target wants (`want` from `served_of`)."""
    if want[0] == "any":
        return True
    if of is None:
        return False
    if want[0] == "text":
        return str(of) == want[1]
    try:
        if want[0] == "level":
            return math.isclose(float(of), want[1], abs_tol=1e-9)
        step, _, level = str(of).partition("/")
        if int(step) != want[1] or bool(level) != (want[2] is not None):
            return False
        return not level or math.isclose(float(level), want[2], abs_tol=1e-9)
    except (TypeError, ValueError):
        return False
