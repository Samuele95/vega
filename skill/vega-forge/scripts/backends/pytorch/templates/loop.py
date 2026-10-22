"""
{{PACKAGE}}.{{MODULE_TRAINING}}.loop — the {{PORT_TRAINER}} adapter (role: Trainer).

{{RESPONSIBILITY_TRAINER}}
Source: {{SOURCE_TRAINER}}

Emitted by /vega-forge from `backends/pytorch/templates/loop.py`. Beside it in this module: the
loss (`losses.py`, spliced), the optimiser and schedule (`optim.py`), the record's numbers
(`record.py`) and the epoch sources (`engines.py`
# >>> shape: graph
and, for the graph views, `graph_engines.py`
# <<< shape
). Fix defects in the templates and re-emit; never in this copy (code-contract §4).

`Trainer` is the one class here (C14). Its loop is deliberately plain — for each epoch, for each
batch: forward, loss, backward, clip, optimiser step, schedule step; then the train and valid
metrics, the early-stopping check, the checkpoint — because it is also what the notebook
reproduces by hand. What it adds is the **record**: one `EpochRecord` per epoch
(backend-contract §1), handed to `callbacks.on_epoch(record)`, and `first_step.json`'s numbers,
handed to `callbacks.on_first_step(record)` after the run's first optimiser step.

Where an epoch's batches come from is the epoch source's (`engines.open_source`: streamed rows,
held rows or windows, a replayed CUDA graph, the graph views), picked from the sample's shape
and the spec's `backend.hints`; the step is the same arithmetic for all of them (`_step`), and
the train-side metrics read the head through `TaskReadout.reduce` — `z > 0` for a binary logit,
never an argmax over one column (E8), a k-hot for multilabel and the cumulative count for
ordinal (E9). A fit is reproducible to the bit on the CPU: every source of randomness is seeded
from `spec.seed` (and the epoch).

    Trainer(store=None).fit(model, spec, data, callbacks=None, run=RunContext(...)) -> FitResult
    Trainer().evaluate(model, spec, loader, loss_fn=None) -> {metric: value, "loss": ...}
    Trainer().predict(model, spec, part) -> ScoredPart

The port is those three methods and nothing more: the device, the per-epoch learning rates and
the loss a fit will use are the public functions of this module's own `optim.py`
(`device_for`, `resolved_lrs`) and `losses.py` (`build_loss`), which the notebooks and the smoke
pass read there (EJ-15, SOLID-I: one door into the training module, the port).

The valid numbers and the predictions are made by the Model module's one scoring pass
(`metrics.ScoringPass`), the pass the Evaluator scores with, so what selection compares and
what the evaluation reports cannot be read two ways.

Effective Java: EJ-64 (every signature is typed by the ports and the domain's value objects:
the network as `TrainedNetwork`, the data as `TrainingData` — the DataSource without its sealed
door —, the callbacks as `Callbacks`, the ModelStore given to the constructor; `RunContext`
carries the rest, C12), EJ-20 (they arrive as Protocols), EJ-15 (helpers are `_`-prefixed).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import torch

from ..domain import (Checkpoint, EpochRecord, FitResult, MetricOptions, Metrics, NetSpec,
                      Part, RunContext, SamplePart, ScoredPart, StopReason, device_of,
                      on_device,
                      # >>> task: regression forecast
                      TargetAffine,
                      # <<< task
                      # >>> shape: graph
                      GraphLevel,
                      # <<< shape
                      )
from ..ports import Callbacks, TrainedNetwork as _Trained, TrainingData
from ..ports import {{PORT_MODELSTORE}} as _Store
from ..{{MODULE_MODEL}} import TaskReadout, metrics as _metrics
from . import optim as _optim
from .engines import CallbackHooks, EpochSource, cpu_threads, open_source
from .losses import build_loss
from .record import (EarlyStopping, EpochAccumulator, EpochTally, GradientProbe, StepClock,
                     StepOutcome, WallClock)

__all__ = ["Trainer"]


def _lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _scoring_size(spec: NetSpec, part) -> int:
    """How many samples of a held part one scoring batch holds: a node-level view whole (one
    forward over the view), else `training.batch_size` ("full": the part)."""
    # >>> shape: graph
    if getattr(part, "level", None) is GraphLevel.NODE:
        return max(1, len(part))
    # <<< shape
    return spec.training.batch_size.for_part(len(part))


def _batches(spec: NetSpec, loader):
    """A held part (its slices, in stored order) or an iterable of batches, as batches."""
    return loader.slices(_scoring_size(spec, loader)) if hasattr(loader, "slices") else loader


# >>> task: regression forecast
def _fit_target(model: _Trained, spec: NetSpec, data) -> None:
    """A37: a fresh regression-family head's target affine is fitted on the TRAIN target, read
    through the DataSource from the train part alone; a resumed network is left as it is and
    nothing is read for it. A regression-family package's step only."""
    if model.target_affine_pending:
        model.fit_target_affine(TargetAffine.of(spec, data.train_targets(spec), model.ids[-1]))
# <<< task


def _first_step(model: _Trained, outcome: StepOutcome, lr: float) -> dict:
    """first_step.json: the head block's ‖g‖, ‖Δθ‖ and their ratio at the run's first step."""
    head = model.ids[-1] if model.ids else None
    diag = outcome.diag
    return {"block": head, "grad_norm_head": diag["per_block"].get(head),
            "delta_norm_head": GradientProbe.delta_norm(diag["snapshot"], model, head),
            "ratio_head": diag["ratio"].get(head), "loss": float(outcome.loss), "lr": lr}


@dataclass(frozen=True)
class _Fit:
    """One fit's collaborators, as `_prepare` resolved them (frozen: the epochs read them and
    replace none). The network, the optimiser, the schedule, the selection and the clocks are
    stateful objects the epochs drive; what the selection has kept so far is `_Best`, which
    each epoch returns rather than writes here."""

    model: _Trained
    spec: NetSpec
    device: Any
    readout: TaskReadout
    loss_fn: Any
    optimizer: Any
    schedule: _optim.Schedule
    source: EpochSource
    names: tuple
    options: MetricOptions
    stopping: EarlyStopping
    clock: WallClock
    steps: StepClock
    callbacks: CallbackHooks
    run: RunContext
    clip: float | None = None
    params: tuple = ()

    @property
    def max_epochs(self) -> int:
        return self.spec.training.max_epochs


@dataclass(frozen=True)
class _Best:
    """What the selection has kept so far: the best epoch's valid metrics and its checkpoint
    (None until an epoch improves, and the checkpoint None without a store to write it)."""

    valid: Mapping | None = None
    checkpoint: Checkpoint | None = None


class _Pass:
    """One epoch's steps, as the source drives them (the engines' `StepDriver`): each step's
    outcome into the tally; the diagnosed steps (the run's first, every k-th) to the callbacks."""

    def __init__(self, trainer, fit: _Fit, tally: EpochTally, epoch: int):
        self._trainer, self._fit, self._tally, self._epoch = trainer, fit, tally, epoch

    @property
    def first_pending(self) -> bool:
        return self._fit.steps.first_pending

    def eager(self, batch: dict) -> None:
        fit = self._fit
        sampled, first = fit.steps.tick(), fit.steps.first_pending
        lr = _lr(fit.optimizer) if (sampled or first) else None
        outcome = self._trainer._step(fit, batch, diag=sampled or first)
        self._tally.add(outcome)
        if outcome.diag is not None:
            self._tally.note_ratio(outcome.diag["ratio"])
        if sampled:
            self._sampled(outcome.diag["grad_norm"], lr, outcome.loss)
        if first:
            fit.callbacks.on_first_step(_first_step(fit.model, outcome, lr))
            fit.steps.first_taken()

    def pure(self, batch: dict) -> StepOutcome:
        """The step with no bookkeeping and no host synchronisation (what a graph captures)."""
        return self._trainer._step(self._fit, batch)

    def replayed(self, static: StepOutcome, replay) -> None:
        """A step replayed from a captured graph: `static` holds its outputs after `replay()`."""
        fit = self._fit
        sampled = fit.steps.tick()
        lr = _lr(fit.optimizer) if sampled else None
        before = GradientProbe.snapshot(fit.model) if sampled else None
        replay()
        truth = None if static.truth is None else static.truth.clone()
        self._tally.add(static._replace(reduced=static.reduced.clone(), truth=truth))
        if sampled:
            self._tally.note_ratio(GradientProbe.update_ratio(before, fit.model))
            self._sampled(float(static.grad_norm), lr, static.loss)

    def _sampled(self, grad_norm, lr, loss) -> None:
        self._fit.callbacks.on_step({"epoch": self._epoch, "step": self._fit.steps.count,
                                     "grad_norm": grad_norm, "lr": lr, "loss": float(loss)})


class Trainer:
    """The Trainer port's adapter: `fit`, `evaluate`, `predict`."""

    def __init__(self, store: _Store | None = None):
        """`store`: the ModelStore port the best checkpoint is written through; without one a
        fit writes no checkpoint (and its result names none)."""
        self._store = store

    def fit(self, model: _Trained, spec: NetSpec, data: TrainingData,
            callbacks: Callbacks | None = None, run: RunContext | None = None) -> FitResult:
        """The loop, with the epoch record, over the data the fit may read (the train and
        valid parts: `TrainingData` has no door to the test part). `run` carries the budget
        (minutes of wall clock; the loop stops at the epoch boundary after it and records
        `stopped_by: budget`), the checkpoint directory and the gradient sampling rate of
        `callbacks.on_step`."""
        spec = NetSpec.require(spec)
        device = _optim.device_for(spec)
        with cpu_threads(spec, model, device) as threads:
            result = self._fit(self._prepare(model, spec, data, run or RunContext(), callbacks))
        return replace(result, threads=threads if torch.device(device).type == "cpu" else None)

    def _fit(self, fit: _Fit) -> FitResult:
        """The epochs of a prepared fit, to its stop, and its result."""
        epoch, stopped_by, best = 0, StopReason.MAX_EPOCHS, _Best()
        for epoch in range(1, fit.max_epochs + 1):
            record, best = self._close_epoch(fit, epoch, self._epoch(fit, epoch), best)
            fit.callbacks.on_epoch(record)
            if record.nan:
                stopped_by = StopReason.NAN
            elif fit.stopping.stop:
                stopped_by = StopReason.EARLY_STOPPING
            elif fit.clock.exceeded():
                stopped_by = StopReason.BUDGET
            if stopped_by is not StopReason.MAX_EPOCHS:
                break
        best_valid = Metrics.of_values(Part.VALID, dict(best.valid)) if best.valid else None
        return FitResult(checkpoint=best.checkpoint, best_epoch=fit.stopping.best_epoch,
                         best_value=fit.stopping.best, epochs_run=epoch, stopped_by=stopped_by,
                         best_valid=best_valid)

    @torch.no_grad()
    def evaluate(self, model: _Trained, spec: NetSpec, loader, loss_fn=None) -> dict:
        """{metric: value, "loss": ..., "_unavailable": {name: why}} through the emitted
        metrics.py — the one implementation the evaluator and the notebooks also call. `loader`
        is a held part or an iterable of batches."""
        spec = NetSpec.require(spec)
        with cpu_threads(spec, model, device_of(model)):
            return self._values(model, spec, _batches(spec, loader), loss_fn,
                                MetricOptions().stated(spec))

    @torch.no_grad()
    def predict(self, model: _Trained, spec: NetSpec, part: SamplePart) -> ScoredPart:
        """The held `part` scored by `model` through the one scoring pass: the score as the
        task's score contract reads the head — logits, values, per-row scores, never a
        thresholded label —, the truth (None without one), the part's row ids."""
        spec = NetSpec.require(spec)
        device, scoring = device_of(model), _metrics.ScoringPass(TaskReadout.for_spec(spec))
        model.eval()
        with cpu_threads(spec, model, device):        # the fit's substrate posture, scored alike
            for batch in _batches(spec, part):
                b = on_device(batch, device)
                scoring.add_head(model(b), b, model)
        return scoring.scored(part)

    # ------------------------------------------------------------------ the parts of a fit

    def _prepare(self, model: _Trained, spec: NetSpec, data: TrainingData, run: RunContext,
                 callbacks: Callbacks | None) -> _Fit:
        """Device, seed, the head's target affine (A37), source, loss, optimiser, schedule,
        selection, budget."""
        device = _optim.device_for(spec)
        _optim.seed_everything(spec.seed, device)
        model = model.to(device)
        # >>> task: regression forecast
        _fit_target(model, spec, data)
        # <<< task
        source = open_source(spec, data, device)
        optimizer = _optim.build_optimizer(model, spec,
                                           capturable_on=device if source.capturable else None)
        steps = source.steps_per_epoch if _optim.per_step(spec) else 1
        schedule = _optim.Schedule.for_spec(optimizer, spec, steps)
        if source.capturable and schedule.per_step:
            raise ValueError("a per-step schedule cannot run under backend.hints.cuda_graph; use "
                             "a per-epoch schedule or set cuda_graph: false")
        names = tuple(_metrics.names_for(spec))
        options = MetricOptions().stated(spec)   # as the Evaluator resolves them
        # >>> shape: window
        options = options.scaled(spec, names, data)
        # <<< shape
        return _Fit(model=model, spec=spec, device=device, readout=TaskReadout.for_spec(spec),
                    loss_fn=build_loss(spec), optimizer=optimizer, schedule=schedule,
                    source=source, names=names, options=options,
                    stopping=EarlyStopping.for_spec(spec, names),
                    clock=WallClock.for_spec(spec, run.budget_min),
                    steps=StepClock(run.grad_sample_every),
                    callbacks=CallbackHooks(callbacks), run=run, clip=_optim.grad_clip(spec),
                    params=tuple(p for p in model.parameters() if p.requires_grad))

    def _epoch(self, fit: _Fit, epoch: int) -> EpochTally:
        """One pass of the source's train batches through `_step`."""
        fit.model.train()
        tally = EpochTally(fit.device, EpochAccumulator(fit.readout, fit.names, fit.options))
        with fit.source.guard():
            fit.source.run_epoch(epoch, _Pass(self, fit, tally, epoch))
        return tally

    def _step(self, fit: _Fit, batch: dict, diag: bool = False) -> StepOutcome:
        """Forward, loss, backward, clip (the pre-clip norm is returned), optimiser step,
        schedule step. With `diag`, the pre-clip per-block norms and the update ratio too."""
        model = fit.model
        fit.optimizer.zero_grad(set_to_none=fit.source.zero_to_none)
        out = model(batch)
        loss = fit.loss_fn(out, batch, model)
        loss.backward()
        seen = None
        if diag:
            g, per_g, _w, _per_w = GradientProbe.norms(model)
            seen = {"grad_norm": g, "per_block": per_g, "snapshot": GradientProbe.snapshot(model)}
        grad_norm = torch.nn.utils.clip_grad_norm_(fit.params, fit.clip if fit.clip else math.inf)
        fit.optimizer.step()
        fit.schedule.after_step()
        if seen is not None:
            seen["ratio"] = GradientProbe.update_ratio(seen["snapshot"], model)
        reduced = fit.readout.reduce(out.detach(), batch, model)
        return StepOutcome(loss.detach(), grad_norm, reduced, batch.get("y"), int(out.shape[0]),
                           seen)

    def _close_epoch(self, fit: _Fit, epoch: int, tally: EpochTally,
                     best: _Best) -> tuple[EpochRecord, _Best]:
        """The valid pass, the schedule's epoch step, selection and the checkpoint, the record;
        returns the record and what the selection keeps after this epoch."""
        gnorm, gper, wnorm, wper = GradientProbe.norms(fit.model)
        train = tally.accumulator.metrics()
        with fit.source.guard():
            valid = self._values(fit.model, fit.spec, fit.source.valid_batches(), fit.loss_fn,
                                 fit.options)
        train["loss"] = tally.mean_loss()
        lr = _lr(fit.optimizer)
        fit.schedule.after_epoch(valid.get(fit.stopping.key, valid.get("loss")))
        if fit.stopping.update(fit.stopping.measured(valid, tally.nan), epoch):
            best = _Best(MappingProxyType({k: v for k, v in valid.items()
                                           if not str(k).startswith("_")}),
                         self._checkpoint(fit, epoch) or best.checkpoint)
        return EpochRecord.of({
            "epoch": epoch, "time_s": time.time() - tally.started, "lr": lr, "train": train,
            "valid": valid, "update_ratio": tally.ratio, "nan": tally.nan,
            "grad_norm": {"global": gnorm, "clipped": tally.clipped(fit.clip), "per_block": gper},
            "weight_norm": {"global": wnorm, "per_block": wper},
            "early_stopping": fit.stopping.as_record(),
            "shuffle": fit.source.describe(epoch)}), best

    def _checkpoint(self, fit: _Fit, epoch: int) -> Checkpoint | None:
        """The improved epoch's weights through the ModelStore port, when the Trainer has one
        and the run names a checkpoint directory."""
        if not (fit.run.checkpoint_dir and self._store is not None):
            return None
        return self._store.save(fit.model, f"{fit.run.checkpoint_dir}/best.pt", spec=fit.spec,
                                epoch=epoch)

    @torch.no_grad()
    def _values(self, model: _Trained, spec: NetSpec, batches, loss_fn, options) -> dict:
        """The scored pass behind `evaluate` and the valid half of the record."""
        readout, device = TaskReadout.for_spec(spec), device_of(model)
        model.eval()
        total, n, scoring = 0.0, 0, _metrics.ScoringPass(readout)
        for batch in batches:
            b = on_device(batch, device)
            out = model(b)
            if loss_fn is not None:
                rows = int(out.shape[0])
                total += float(loss_fn(out, b, model)) * rows
                n += rows
            scoring.add_head(out, b, model)
        values = {"loss": (total / n) if n else float("nan")}
        if scoring.empty:
            return values
        computed, unavailable = _metrics.compute(_metrics.names_for(spec), scoring.truth,
                                                 scoring.score, options, readout.task)
        values.update(_metrics.record(computed, unavailable))
        return values
