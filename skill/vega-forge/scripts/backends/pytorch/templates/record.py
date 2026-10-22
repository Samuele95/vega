"""
{{PACKAGE}}.{{MODULE_TRAINING}}.record — the numbers of the epoch record (a `kind: support` file
of the Trainer's module).

Emitted by /vega-forge from `backends/pytorch/templates/record.py`. A design defect here is
fixed in that template and the package re-emitted; never in this copy (code-contract §4).

The record is one JSON object per epoch (backend-contract §1) — the loss and metrics on train
and valid, the learning rate, the gradient and weight norms, the update ratio, the early
stopping state, the epoch source's description — and `first_step.json` once per run. The
pieces that compute it live here, one concern each:

  * `StepOutcome` — what one optimiser step leaves: its loss, its pre-clip gradient norm, the
    rows' reduced predictions and truth, the diagnostics of a diagnosed step;
  * `EpochTally` — the epoch's loss, summed on the device in float64 (the same bits as the
    v2.2 host sum, with no host synchronisation per step), the rows seen, the last gradient
    norm, the update ratio of the diagnosed steps, and the accumulator;
  * `EpochAccumulator` — the train-side metrics, from `TaskReadout.reduce` (one fixed-shape
    prediction per sample: `z > 0` for binary and multilabel, argmax, `Σ_j [z_j ≥ 0]` for
    ordinal, the values, the per-row errors) fed to `metrics.py` — the one implementation the
    evaluator and the notebooks call — as the metric input whose decision it is (the reading's
    `decided_input`, `TaskReadout.scores`). Names outside
    the reading's `label_only()` need the score and are reported unavailable, never
    approximated (E8, E9);
  * `GradientProbe` — per-block gradient and weight norms, parameter snapshots, the update
    ratio ‖Δθ‖ / ‖θ‖ and ‖Δθ‖ of one block, keyed by the spec's block ids (the i-th stage's
    parameters are the i-th id's);
  * `StepClock` — the run's optimiser-step counter, and which steps carry diagnostics: the
    run's first step (`first_step.json`) and every k-th (`callbacks.on_step`);
  * `EarlyStopping` and `WallClock` — the selection rule on the monitored valid metric, and the
    budget that stops the run at an epoch boundary.

Effective Java: EJ-17 (the outcome of a step is a value), EJ-49 (a monitor naming no metric the
spec declares is refused by name), EJ-15 (helpers are `_`-prefixed).
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping, NamedTuple

import torch

from ..domain import MetricOptions, NetSpec, SelectionDirection, host_array
from ..{{MODULE_MODEL}} import metrics as _metrics

__all__ = ["EarlyStopping", "EpochAccumulator", "EpochTally", "GradientProbe", "StepClock",
           "StepOutcome", "WallClock"]

#: why a train-side metric outside `label_only()` is not in the record
UNAVAILABLE = ("scored on a probability, which the epoch's accumulated predictions do not "
               "carry; the evaluator computes it on a scored part")


def _joined(chunks):
    """The steps' arrays as one host array. Tensors are joined where they are and copied to the
    host once: a resident epoch holds one chunk per step, and a copy per chunk would be a host
    synchronisation per step at the epoch's end."""
    if all(torch.is_tensor(c) for c in chunks):
        return host_array(torch.cat(list(chunks)))
    return _metrics.joined([host_array(c) for c in chunks])


class StepOutcome(NamedTuple):
    """One optimiser step: `loss` (0-d), `grad_norm` (0-d, the pre-clip global norm), the rows'
    `reduced` predictions and `truth` (None without a label), the sample count `rows`, and —
    for a diagnosed step — `diag` {per-block pre-clip norms, the parameter snapshot, the
    update ratio}."""

    loss: Any
    grad_norm: Any
    reduced: Any
    truth: Any
    rows: int
    diag: Any = None


class GradientProbe:
    """The record's norms, keyed by the spec's block ids."""

    @staticmethod
    def by_block(model):
        """(block id, parameter name, parameter) for each parameter, read from the structure
        that owns it — the i-th stage's parameters are the i-th block's — never parsed back out
        of a dotted name. A parameter outside every stage has no block and is refused by name."""
        seen = set()
        for i, (block, stage) in enumerate(model.named_stages()):
            for name, p in stage.named_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield block, f"stages.{i}.{name}", p
        stray = [n for n, p in model.named_parameters() if id(p) not in seen]
        if stray:
            raise ValueError(f"parameters {', '.join(stray)} belong to no stage of the network, "
                             "so no block of the spec owns them; the record keys every norm "
                             "by the block that holds the parameter")

    @staticmethod
    def norms(model) -> tuple:
        """(‖g‖, {block: ‖g‖}, ‖θ‖, {block: ‖θ‖}) over the parameters, in their order."""
        g_sq, w_sq, per_g, per_w = 0.0, 0.0, {}, {}
        for b, _name, p in GradientProbe.by_block(model):
            w = float(p.detach().norm().item() ** 2)
            w_sq += w
            per_w[b] = per_w.get(b, 0.0) + w
            if p.grad is not None:
                g = float(p.grad.detach().norm().item() ** 2)
                g_sq += g
                per_g[b] = per_g.get(b, 0.0) + g
        return (math.sqrt(g_sq), {k: math.sqrt(v) for k, v in per_g.items()},
                math.sqrt(w_sq), {k: math.sqrt(v) for k, v in per_w.items()})

    @staticmethod
    def snapshot(model) -> dict:
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    @staticmethod
    def update_ratio(before: dict, model) -> dict:
        """{block: ‖Δθ‖ / ‖θ‖} of the step since `before` (None for an all-zero block)."""
        num, den = {}, {}
        for b, name, p in GradientProbe.by_block(model):
            prev = before.get(name)
            if prev is None:
                continue
            num[b] = num.get(b, 0.0) + float((p.detach() - prev).norm().item() ** 2)
            den[b] = den.get(b, 0.0) + float(prev.norm().item() ** 2)
        return {b: (math.sqrt(num[b]) / math.sqrt(den[b]) if den.get(b) else None) for b in num}

    @staticmethod
    def delta_norm(before: dict, model, block) -> float:
        """‖Δθ‖ over one block's parameters (first_step.json reports it beside the ratio)."""
        total = 0.0
        for b, name, p in GradientProbe.by_block(model):
            prev = before.get(name)
            if prev is not None and b == block:
                total += float((p.detach() - prev).norm().item() ** 2)
        return math.sqrt(total)


class EpochAccumulator:
    """The epoch's train-side metrics, from the reduced predictions of every step."""

    def __init__(self, readout, names, options: MetricOptions | None = None):
        self._readout, self._names = readout, list(names)
        self._options = options or MetricOptions()
        self._shape = tuple(readout.reduced_shape())
        self._pred, self._true = [], []

    def update(self, reduced, truth) -> None:
        """One step's rows; tensors stay where they are until `metrics` (no host sync). A
        reduction whose per-sample shape is not the reading's `reduced_shape()` is refused (a
        shape is read without a synchronisation): its metrics would be computed on the wrong
        axis."""
        if tuple(reduced.shape[1:]) != self._shape:
            raise ValueError(f"the {self._readout.kind.value} reading reduces each sample to "
                             f"shape {self._shape}, and a step reduced {len(reduced)} sample(s) "
                             f"to {tuple(reduced.shape[1:])}")
        self._pred.append(reduced.detach() if torch.is_tensor(reduced) else reduced)
        self._true.append(None if truth is None else
                          (truth.detach() if torch.is_tensor(truth) else truth))

    def metrics(self) -> dict:
        if not self._pred:
            return {}
        pred = _joined(self._pred)
        truth = None if any(t is None for t in self._true) else _joined(self._true)
        truth, score = self._readout.scores.decided_input(pred, truth,
                                                          len(self._readout.task.labels))
        wanted = [n for n in self._names if n in self._readout.label_only()]
        out, unavailable = _metrics.compute(wanted, truth, score, self._options,
                                            self._readout.task)
        for name in self._names:
            if name not in wanted:
                unavailable[name] = UNAVAILABLE
        if unavailable:
            out["_unavailable"] = unavailable
        return out


def _sum_dtype(device):
    """float64 wherever the device has it (MPS has not): the sum's bits are the host's."""
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


class EpochTally:
    """One epoch's running numbers. `add` issues device operations only, so a step replayed
    from a CUDA graph is never followed by a host synchronisation."""

    def __init__(self, device, accumulator: EpochAccumulator):
        self._loss = torch.zeros((), dtype=_sum_dtype(device), device=device)
        self.seen, self.grad_norm, self._ratio = 0, None, {}
        self.accumulator, self.started = accumulator, time.time()

    @property
    def ratio(self) -> dict:
        """The update ratio of the epoch's last diagnosed step ({block: ‖Δθ‖ / ‖θ‖})."""
        return self._ratio

    def note_ratio(self, ratio: dict) -> None:
        """A diagnosed step's update ratio: the epoch's record carries the last one."""
        self._ratio = dict(ratio)

    def add(self, outcome: StepOutcome) -> None:
        self._loss += outcome.loss.detach().to(self._loss.dtype) * int(outcome.rows)
        self.seen += int(outcome.rows)
        self.grad_norm = outcome.grad_norm
        self.accumulator.update(outcome.reduced, outcome.truth)

    def mean_loss(self) -> float:
        return float(self._loss) / max(1, self.seen)

    @property
    def nan(self) -> bool:
        """A step's loss was not finite (the sum carries a NaN or an infinity)."""
        return not math.isfinite(float(self._loss))

    def clipped(self, clip) -> bool:
        """Whether the epoch's last step was clipped (its pre-clip norm above the bound)."""
        return bool(clip) and self.grad_norm is not None and float(self.grad_norm) > clip


class StepClock:
    """The run's optimiser steps: the count, and which steps carry diagnostics — the first of
    the run (`first_step.json`) and every `every`-th (`callbacks.on_step`; 0 = none)."""

    def __init__(self, every: int = 0):
        self.count, self.every, self._first_done = 0, int(every or 0), False

    @property
    def first_pending(self) -> bool:
        """Whether the run's first optimiser step (the one `first_step.json` records) is still
        to come."""
        return not self._first_done

    def first_taken(self) -> None:
        """The run's first step was taken and recorded."""
        self._first_done = True

    def tick(self) -> bool:
        """Counts one step; True when it is a sampled one."""
        self.count += 1
        return bool(self.every) and self.count % self.every == 0


class EarlyStopping:
    """Selection on the monitored valid metric: `update(value)` says whether it improved by
    more than `min_delta` in the monitor's direction; `stop` once `patience` epochs did not."""

    def __init__(self, key: str, mode: SelectionDirection, patience: int, min_delta: float):
        self.key, self.mode = key, SelectionDirection(mode)
        self.patience, self.min_delta = int(patience), float(min_delta)
        self.best, self.best_epoch, self.since = None, None, 0

    @staticmethod
    def for_spec(spec: NetSpec, known) -> "EarlyStopping":
        """`early_stopping`, falling back to `metrics.selection`. A monitor that names nothing
        the spec declares is refused: every epoch would score None, nothing would improve, no
        checkpoint would be written and the run would stop at patience as if it had converged."""
        rule = NetSpec.require(spec).selection
        known = set(known) | {"loss"}
        if rule.key not in known:
            raise ValueError(
                f"the early-stopping monitor {rule.monitor!r} names {rule.key!r}, which is not "
                f"one of the metrics this spec declares ({', '.join(sorted(known))}); selection "
                "would never improve and the run would stop at patience with no checkpoint")
        return EarlyStopping(rule.key, rule.mode, rule.patience, rule.min_delta)

    def _better(self, value) -> bool:
        if self.best is None:
            return True
        if self.mode is SelectionDirection.MAX:
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def measured(self, valid: Mapping, nan: bool = False):
        """The monitored metric of an epoch's valid record. One the epoch could not compute is
        not "did not improve": counting it so would expire patience with no checkpoint
        written, the run `for_spec` refuses — so it is refused here, naming the monitor and
        the reason `metrics.py` recorded. A NaN epoch is the exception: the loop stops on it."""
        value = valid.get(self.key)
        if value is None and not nan:
            why = (valid.get("_unavailable") or {}).get(self.key, "no value was recorded")
            raise ValueError(f"the early-stopping monitor valid.{self.key} could not be "
                             f"computed on this epoch ({why}); selection cannot read it, so the "
                             "run stops here rather than expire patience with no checkpoint")
        return value

    def update(self, value, epoch: int) -> bool:
        improved = value is not None and self._better(value)
        if improved:
            self.best, self.best_epoch, self.since = value, epoch, 0
        else:
            self.since += 1
        return improved

    @property
    def stop(self) -> bool:
        return self.since >= self.patience

    def as_record(self) -> dict:
        return {"best": self.best, "since_best": self.since, "stop": self.stop}


class WallClock:
    """The run's budget in minutes of wall clock (None: unbounded), read at epoch boundaries."""

    def __init__(self, budget_min=None):
        self.budget_min = float(budget_min) if budget_min else None
        self._t0 = time.time()

    @staticmethod
    def for_spec(spec: NetSpec, budget_min=None) -> "WallClock":
        """`budget_min` (the run's), else `training.budget.wall_clock_min`."""
        return WallClock(NetSpec.require(spec).training.budget_min if budget_min is None
                         else budget_min)

    def exceeded(self) -> bool:
        return bool(self.budget_min) and (time.time() - self._t0) / 60.0 >= self.budget_min
