"""
inspection.targets — what an explanation explains: one differentiable scalar `s` of the
explained sample, per task (inspect-contract §4; build plan R-B6, R-B7, R-C5, R-C6, R-F6).

`TaskTargets.of(spec, output, model)` reads the task through the one target table
(`view.TARGETS`, with the loss rows `view.BY_LOSS`, which `network.task` publishes too) and the
bundle's `MANIFEST.output` (`vega/output@2`: the classes, the positive class, the thresholds and
`field_kinds`). The effective kind is `task.base_kind` for a `sequence-*` task; the anomaly
flavour follows `loss.name` (`svdd` explains the distance to the centre, `reconstruction` and
`vae` the reconstruction). A svdd network whose centre buffer was never set does not open
(`TargetError`): no distance is guessed.

  kind                   selector       `s` (per sample)                 picked when not requested
  logit                  class          z_c                              the argmax class
  binary_logit           class          sign·z (+1 positive, −1 other)   the class decided at the
                                                                         threshold
  label_logit            label          z_j                              the label of largest σ
  value                  —              ŷ (TaskReadout.score)            —
  quantile               quantile       ŷ_q                              the quantile nearest 0.5
  expected_rank          —              Σ_j σ(z_j)                       —
  threshold_logit        threshold      z_j                              the first threshold not
                                                                         passed (else the last)
  reconstruction_error   —              TaskReadout.row_score            —
  feature_error          feature        (x̂_f − x_f)², a nominal's CE     the largest error
  svdd_distance          —              ‖f(x) − c‖², c the centre buffer —
  forecast_step          step (1…H)     ŷ_{t+h}: the readout's step h    step 1

`value` under the quantile loss is the quantile nearest 0.5 (the readout's point estimate); a
`quantile` selector is the quantile's index or its level; `forecast_step` h reads column h−1 of
the readout's steps (their median under a quantile loss).

`index` and `name` locate the target: the class (binary: the class explained, `sign` its
direction), the label, the threshold `j` (named `>c_j`), the head column a value, quantile or
step reads (named as `network.task.outputs.names` names it), the feature; none for a scalar of
the whole head. `why` is `requested` (the request named it), `predicted` (picked from the
prediction) or `default`. A request's selector is checked against the task (`QueryError`, 422):
"target.class does not apply to a regression task"; an unknown class, label or feature name.
`kind` also takes the request vocabulary of the page and the serving readings (`ALIASES`).

`readout(head, batch, k)` is `explanation.outputs` (§4.5): the head's `kind` and `names` exactly
as `network.task.outputs` gives them, the `raw` outputs, `read` after the link (never a
distribution the head does not produce) and the task's `answer`.

`served(target, answer)` is the fidelity cross-check's served value (R-C5, §4.6): the field is
found in `MANIFEST.output.field_kinds` by `(kind, of)` through `TARGET_COLUMNS` — `raw` of the
class or label (binary: of the positive class, times `sign`), `value`, `quantile` of the level,
`horizon` of the step, `row_score` / `score` — never by a fixed column name; None where no field
serves the target. The kinds themselves (one strategy each) live in `target_kinds.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from .codec import num, nums
from .query import SELECTORS, QueryError
from .readers import GraphError
from .target_kinds import KIND_TABLE, KINDS, TargetFacts, feature_errors, of_matches
from .view import BY_LOSS, Inputs, task_section
from ...domain import Objective, SampleSpec, Task

__all__ = ["ALIASES", "KINDS", "SVDD_WHY", "TARGET_COLUMNS", "TARGET_KINDS", "ExplainTarget",
           "TargetError", "TaskTargets"]

TARGET_KINDS = tuple(KINDS)
# the request vocabulary of the page and the serving readings -> the kinds it may name (§4.2)
ALIASES = MappingProxyType({
    "class": ("logit", "binary_logit"), "true_class": ("logit", "binary_logit"),
    "positive_logit": ("binary_logit",), "label": ("label_logit",),
    "expected_ordinal": ("expected_rank",), "error": ("reconstruction_error",),
    "distance": ("svdd_distance",), "horizon_step": ("forecast_step",),
    "horizon": ("forecast_step",), "row_score": ("reconstruction_error", "svdd_distance")})
# target kind -> the `field_kinds` kinds of the served field that answers it, in preference
# order (R-C5); a target kind absent here has no served field
TARGET_COLUMNS = MappingProxyType({
    "logit": ("raw",), "binary_logit": ("raw",), "label_logit": ("raw",),
    "threshold_logit": ("raw",), "value": ("value",), "quantile": ("quantile",),
    "forecast_step": ("horizon",), "reconstruction_error": ("row_score", "score"),
    "svdd_distance": ("score",)})
SVDD_WHY = "the one-class centre was not persisted with the weights"
WHYS = ("predicted", "requested", "default")


class TargetError(GraphError):
    """The targets of this network cannot be built (a missing svdd centre, a manifest written
    for another network): the inspector does not open."""


@dataclass(frozen=True)
class ExplainTarget:
    """The explained scalar: its kind, where it sits (`index`, `name`), its sign and why."""

    kind: str
    index: int | None
    name: str | None
    sign: int
    why: str

    def __post_init__(self):
        if self.kind not in KINDS or self.why not in WHYS or self.sign not in (1, -1):
            raise ValueError(f"not a target: kind {self.kind!r}, why {self.why!r}, sign "
                             f"{self.sign!r}")

    @property
    def unit(self) -> str:
        return KINDS[self.kind][1]

    def to_dict(self, value) -> dict:
        """`explanation.target`, with the scalar's value for the explained sample."""
        return {"kind": self.kind, "index": self.index, "name": self.name, "sign": self.sign,
                "value": num(value), "why": self.why, "unit": self.unit}


def _data(spec) -> dict:
    return spec.to_dict() if callable(getattr(spec, "to_dict", None)) else dict(spec)


def _two_d(head):
    return head if head.dim() == 2 else head.reshape(head.shape[0], -1)


def _sample(z, k: int) -> None:
    if not 0 <= int(k) < z.shape[0]:
        raise IndexError(f"sample {k} is outside the {z.shape[0]} sample(s) of the head")


# ------------------------------------------------------------------------------ the readout

def _alarm(f: TargetFacts, score) -> dict:
    """An anomaly reading's decision at its threshold (score scale), when it has one."""
    if f.kind != "anomaly":
        return {}
    if f.threshold is None:
        return {"threshold": None, "anomalous": None}
    return {"threshold": num(f.threshold), "anomalous": bool(float(score) >= f.threshold)}


def _read_classes(f, z, batch, k):
    p = torch.softmax(z[k], dim=-1)
    c = int(p.argmax())
    return nums(p), {"class": f.classes[c], "p": num(p[c])}


def _read_score(f, z, batch, k):
    p = float(torch.sigmoid(z[k, 0]))
    name = f.positive if p >= f.cut() else f.negative
    return [num(p)], {"class": name, "p": num(p), "threshold": num(f.cut())}


def _read_labels(f, z, batch, k):
    p = torch.sigmoid(z[k])
    return nums(p), {"labels": [c for j, c in enumerate(f.classes) if float(p[j]) >= f.cut(c)]}


def _read_value(f, z, batch, k):
    return nums(z[k]), {"value": num(f.readout.score(z, batch, f.anatomy)[k])}


def _read_thresholds(f, z, batch, k):
    p = torch.sigmoid(z[k])
    rank = int((p >= 0.5).sum())
    return nums(p), {"rank": rank, "class": f.classes[rank], "expected_rank": num(p.sum())}


def _read_reconstruction(f, z, batch, k):
    """x̂ per feature: the numeric reconstruction, then each nominal's argmax category id."""
    rebuilt, at = [float(v) for v in z[k, :f.n_numeric]], f.n_numeric
    for card in f.cards:
        rebuilt.append(float(z[k, at: at + card].argmax()))
        at += int(card)
    error = f.readout.row_score(z, batch, f.anatomy)[k]
    terms = feature_errors(f, z, batch)[k]
    return nums(rebuilt), {"error": num(error), "per_feature": nums(terms), **_alarm(f, error)}


def _read_embedding(f, z, batch, k):
    distance = f.readout.score(z, batch, f.anatomy)[k]
    return None, {"distance": num(distance), **_alarm(f, distance)}


def _read_horizon(f, z, batch, k):
    return nums(z[k]), {"steps": nums(f.readout.score(z, batch, f.anatomy)[k].reshape(-1))}


# output kind (network.task.outputs.kind) -> (read, answer) of one sample
_READS = MappingProxyType({
    "classes": _read_classes, "score": _read_score, "labels": _read_labels,
    "value": _read_value, "quantiles": _read_value, "thresholds": _read_thresholds,
    "reconstruction": _read_reconstruction, "embedding": _read_embedding,
    "horizon": _read_horizon})


# ------------------------------------------------------------------------------ the facade

def _facts(data: dict, output: Mapping, section: Mapping, anatomy) -> TargetFacts:
    task, inputs = Task.from_spec(data), Inputs.of(data)
    classes = tuple(str(c) for c in (output.get("classes") or task.labels.classes or ()))
    kind = task.effective.value
    if kind == "binary" and section["positive"] != str(task.labels.at(1).name):
        raise TargetError(f"the binary head's logit reads task.classes[1] "
                          f"({task.labels.at(1).name!r}) and the positive class is "
                          f"{section['positive']!r}: list the positive class second")
    window = SampleSpec.from_spec(data).window
    per_label = {str(k): float(v) for k, v in (output.get("threshold_per_label") or {}).items()}
    return TargetFacts(kind=kind, loss=(data.get("loss") or {}).get("name"), classes=classes,
                       positive=section["positive"], threshold=section["threshold"],
                       per_label=MappingProxyType(per_label),
                       quantiles=tuple(float(q) for q in section["quantiles"] or ()),
                       horizon=int(window.horizon) if window is not None else 0,
                       features=tuple(inputs.numeric + inputs.nominal),
                       cards=tuple(inputs.cards), readout=anatomy.readout, anatomy=anatomy)


def _check(f: TargetFacts, output: Mapping, link: str) -> None:
    if f.readout.objective is Objective.SVDD:
        if f.anatomy.objective_center() is None or not f.anatomy.centre_set:
            raise TargetError(SVDD_WHY)
    if output.get("link") not in (None, link):
        raise TargetError(f"MANIFEST.output reads the head through the {output['link']!r} link "
                          f"and the network through {link!r}: the manifest is another model's")


class TaskTargets:
    """The targets of one served network, under its spec and the bundle's MANIFEST.output."""

    def __init__(self, facts: TargetFacts, section: Mapping, field_kinds: Mapping):
        self._facts, self._section, self._fields = facts, section, field_kinds

    @staticmethod
    def of(spec, output: Mapping | None, model) -> "TaskTargets":
        """`output` is MANIFEST.output (None: the spec's classes, σ = 0.5, no served fields);
        `model` the served network, read through its Model port's `anatomy()` (its readout,
        and its svdd centre)."""
        data, output = _data(spec), dict(output or {})
        section = task_section(data, model, output)
        facts = _facts(data, output, section, model.anatomy())
        _check(facts, output, section["link"])
        fields = {str(k): dict(v) for k, v in (output.get("field_kinds") or {}).items()}
        return TaskTargets(facts, MappingProxyType(section), MappingProxyType(fields))

    @property
    def link(self) -> str:
        return self._section["link"]

    @property
    def outputs(self) -> dict:
        """`{kind, names}` of the head, as `network.task.outputs` gives them."""
        return {"kind": self._section["outputs"]["kind"],
                "names": list(self._section["outputs"]["names"])}

    def kinds(self) -> tuple:
        """The target kinds this task explains."""
        return tuple(self._section["targets"])

    def default_kind(self) -> str:
        return self._section["default_target"]

    def validate(self, request) -> None:
        """Refuse (`QueryError`) a target request this task cannot explain, before any run."""
        strategy, value, _ = self._resolve(request)
        if value is not None:
            strategy.requested(self._facts, value)

    def choose(self, head, batch, request, k: int) -> ExplainTarget:
        """The target of sample `k`: the requested one, else the task's default picked from
        the head's outputs."""
        strategy, value, named = self._resolve(request)
        if value is not None:
            index, name, sign = strategy.requested(self._facts, value)
            return ExplainTarget(strategy.NAME, index, name, sign, "requested")
        with torch.no_grad():
            z = _two_d(head).detach()
            _sample(z, k)
            index, name, sign, why = strategy.picked(self._facts, z, batch, k)
        return ExplainTarget(strategy.NAME, index, name, sign,
                             "requested" if named and why == "default" else why)

    def scalar(self, target: ExplainTarget, head, batch):
        """(B,) the target's scalar for every sample of the batch, differentiable in `head`."""
        if target.kind not in self.kinds():
            raise ValueError(f"a {target.kind} target is not one this task explains "
                             f"({', '.join(self.kinds())})")
        return KIND_TABLE[target.kind].scalar(self._facts, target, _two_d(head), batch)

    def value(self, target: ExplainTarget, head, batch, k: int) -> float:
        """The target's value for sample `k`, at 9 significant digits."""
        with torch.no_grad():
            return num(self.scalar(target, head, batch)[int(k)])

    def readout(self, head, batch, k: int) -> dict:
        """`explanation.outputs` of sample `k` (inspect-contract §4.5)."""
        with torch.no_grad():
            z = _two_d(head).detach()
            _sample(z, k)
            read, answer = _READS[self._section["outputs"]["kind"]](self._facts, z, batch, k)
            return {**self.outputs, "link": self.link, "raw": nums(z[k]), "read": read,
                    "answer": answer}

    def served_field(self, target: ExplainTarget) -> tuple | None:
        """(the served field answering the target, the factor its value is read with), found
        by `(kind, of)` in MANIFEST.output.field_kinds; None when no field serves it."""
        want = KIND_TABLE[target.kind].served_of(self._facts, target)
        for kind in (TARGET_COLUMNS.get(target.kind, ()) if want is not None else ()):
            for field, entry in self._fields.items():
                if entry.get("kind") == kind and of_matches(entry.get("of"), want):
                    return field, (target.sign if target.kind == "binary_logit" else 1)
        return None

    def served(self, target: ExplainTarget, answer) -> float | None:
        """The served answer's value of the target (a Predictor frame row, or any mapping of
        field to value); None when no field serves it or the row was not answered."""
        found = self.served_field(target)
        value = None if found is None else answer.get(found[0])
        if value is None or not math.isfinite(float(value)):
            return None
        return found[1] * float(value)

    # -- the request ----------------------------------------------------------------------
    def _resolve(self, request) -> tuple:
        """(the kind's strategy, the selector's value or None, whether the request named it)."""
        given = {str(k): v for k, v in (request or {}).items() if v is not None}
        if not given:
            return KIND_TABLE[self.default_kind()], None, False
        if len(given) > 1:
            raise QueryError("target", "a target takes at most one selector; the request names "
                                       + ", ".join(sorted(given)))
        (key, value), = given.items()
        if key == "kind":
            return KIND_TABLE[self._kind_of(value)], None, True
        if key not in SELECTORS:
            raise QueryError(f"target.{key}", f"target.{key} is not a target selector (one of "
                                              f"{', '.join(SELECTORS)})")
        fits = [k for k in self.kinds() if KINDS[k][0] == key]
        if not fits:
            raise QueryError(f"target.{key}", f"target.{key} does not apply to {self._task()}")
        return KIND_TABLE[fits[0]], value, True

    def _kind_of(self, name) -> str:
        name = str(name)
        candidates = (name,) if name in KINDS else ALIASES.get(name)
        if candidates is None:
            raise QueryError("target.kind", f"target.kind {name!r} is not a target kind (one "
                                            f"of {', '.join(TARGET_KINDS)})")
        fits = [c for c in candidates if c in self.kinds()]
        if not fits:
            raise QueryError("target.kind", f"target.kind {name!r} does not apply to "
                                            f"{self._task()}; it explains "
                                            f"{', '.join(self.kinds())}")
        return fits[0]

    def _task(self) -> str:
        f = self._facts
        by_loss = any(kind == f.kind for kind, _ in BY_LOSS)
        article = "an" if f.kind[0] in "aeiou" else "a"
        return f"{article} {f.kind} task" + (f" trained with the {f.loss} loss" if by_loss else "")
