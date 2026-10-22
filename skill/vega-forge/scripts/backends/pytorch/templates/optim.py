"""
{{PACKAGE}}.{{MODULE_TRAINING}}.optim — the optimiser, the schedule, the device and the seed of
a fit (a `kind: support` file of the Trainer's module).

Emitted by /vega-forge from `backends/pytorch/templates/optim.py`. A design defect here is fixed
in that template and the package re-emitted; never in this copy (code-contract §4).

What a run is set up with before its first step, read from the spec and nothing else:

  * `build_optimizer` — the spec's optimiser over the model's parameters; `capturable_on` builds
    it for a replayed CUDA graph (the step counter and the learning rate live on the device);
  * `Schedule.for_spec` — the spec's schedule, and *when* it steps: per epoch, per epoch on the
    monitored value, or per optimiser step (`onecycle`, which reads the source's
    `steps_per_epoch`, not a row count);
  * `resolved_lrs` — the per-epoch learning rates, drawn before any run (the architecture layer);
  * `device_for` / `seed_everything` / `grad_clip` — the device the spec asks for (`auto` picks
    CUDA, then MPS, then the CPU), the seeding that makes a CPU fit reproducible to the bit
    (backend-contract §1, determinism), and the spec's gradient-clipping bound.

The weight decay and the clipping bound are read from the regularization declaration
(`regularization[kind=weight_decay|grad_clip].value`, `domain.Regularization`; netspec-contract
§3, R2) and from their live paths (`optimizer.params.weight_decay`, `training.grad_clip`) only
when nothing declares them — so a ladder delta on the declaration moves the weights.

Effective Java: EJ-1 (named factories), EJ-17 (a schedule's policy is fixed at construction),
EJ-34 (the step policy is a closed set, not a string compared everywhere).
"""
from __future__ import annotations

import warnings
from enum import Enum
from types import MappingProxyType

import numpy as np
import torch
from torch import nn

from ..domain import NetSpec, Regularization

__all__ = ["Schedule", "StepPolicy", "build_optimizer", "device_for", "grad_clip", "per_step",
           "resolved_lrs", "seed_everything"]

_OPTIMIZERS = MappingProxyType({
    "adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD,
    "radam": torch.optim.RAdam, "rmsprop": torch.optim.RMSprop})



class StepPolicy(Enum):
    """When a schedule steps: after every epoch, after every epoch on the monitored value, or
    after every optimiser step (a closed set: EJ-34)."""

    EPOCH = "epoch"
    EPOCH_METRIC = "epoch-metric"
    STEP = "step"


def build_optimizer(model, spec: NetSpec, capturable_on=None):
    """The spec's optimiser. `capturable_on`, a device, builds it for a captured CUDA graph:
    `capturable=True` keeps the step counter on the device, and the learning rate becomes a
    0-d device tensor the schedulers update in place — which is what lets a per-epoch schedule
    reach a graph captured once (a float lr would be baked into the graph at capture)."""
    spec = NetSpec.require(spec)
    node = spec.optimizer
    cls = _OPTIMIZERS.get(node.name)
    if cls is None:
        raise ValueError(f"unknown optimizer {node.name!r}; this backend implements "
                         f"{sorted(_OPTIMIZERS)}")
    params = Regularization.of(spec).optimizer_params(node.params_dict())
    if "betas" in params:
        params["betas"] = tuple(params["betas"])
    if cls is torch.optim.SGD:
        params.pop("betas", None)
    if capturable_on is not None:
        params["capturable"] = True
        params["lr"] = torch.tensor(float(params.get("lr", 1e-3)), device=capturable_on)
    return cls(model.parameters(), **params)


def _cosine(optimizer, params, max_epochs, steps_per_epoch):
    warmup = int(params.get("warmup_epochs", 0))
    base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup))
    if not warmup:
        return base, StepPolicy.EPOCH
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                                             total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, base],
                                                 milestones=[warmup]), StepPolicy.EPOCH


def _plateau(optimizer, params, max_epochs, steps_per_epoch):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=params.get("mode", "min"), factor=float(params.get("factor", 0.5)),
        patience=int(params.get("patience", 5))), StepPolicy.EPOCH_METRIC


def _onecycle(optimizer, params, max_epochs, steps_per_epoch):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(params.get("max_lr", 0.01)), epochs=max_epochs,
        steps_per_epoch=max(1, int(steps_per_epoch))), StepPolicy.STEP


# schedule name -> (optimizer, params, max_epochs, steps_per_epoch) -> (scheduler, when)
_SCHEDULES = MappingProxyType({"cosine": _cosine, "plateau": _plateau, "onecycle": _onecycle})


def per_step(spec: NetSpec) -> bool:
    """Whether the spec's schedule steps per optimiser step (it then needs the step count)."""
    return _SCHEDULES.get(NetSpec.require(spec).schedule.name) is _onecycle


class Schedule:
    """The spec's learning-rate schedule and when it steps. `none` is a schedule that never
    steps, so the loop calls `after_step` and `after_epoch` unconditionally."""

    def __init__(self, scheduler, when):
        self._scheduler, self._when = scheduler, when

    @staticmethod
    def for_spec(optimizer, spec: NetSpec, steps_per_epoch: int = 1) -> "Schedule":
        spec = NetSpec.require(spec)
        node = spec.schedule
        name = node.name
        if name == "none":
            return Schedule(None, None)
        make = _SCHEDULES.get(name)
        if make is None:
            raise ValueError(f"unknown schedule {name!r}; this backend implements "
                             f"{', '.join(sorted(_SCHEDULES))}, none")
        return Schedule(*make(optimizer, node.params_dict(), spec.training.max_epochs,
                              steps_per_epoch))

    @property
    def per_step(self) -> bool:
        return self._when is StepPolicy.STEP

    def after_step(self) -> None:
        """Called after every optimiser step; steps a per-step schedule only."""
        if self._when is StepPolicy.STEP:
            self._scheduler.step()

    def after_epoch(self, monitored=None) -> None:
        """Called once per epoch, after the valid pass; `monitored` feeds `plateau`."""
        if self._when is StepPolicy.EPOCH:
            self._scheduler.step()
        elif self._when is StepPolicy.EPOCH_METRIC:
            self._scheduler.step(monitored)


def resolved_lrs(spec: NetSpec, steps_per_epoch: int = 1) -> list:
    """The per-epoch learning rates, so the layer can draw the schedule before any run.

    The scheduler is stepped against a throwaway parameter and no optimiser step ever follows,
    which is exactly the order torch warns about — the warning is about training, and this is a
    dry read of the schedule, so it is silenced here only."""
    spec = NetSpec.require(spec)
    opt = build_optimizer(nn.Linear(1, 1), spec)
    schedule = Schedule.for_spec(opt, spec, steps_per_epoch)
    lrs = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).*")
        for _ in range(spec.training.max_epochs):
            lrs.append(float(opt.param_groups[0]["lr"]))
            for _ in range(max(1, int(steps_per_epoch)) if schedule.per_step else 0):
                schedule.after_step()
            schedule.after_epoch(1.0)
    return lrs


def device_for(spec: NetSpec) -> torch.device:
    """`training.device`; `auto` is CUDA when present, then MPS, then the CPU."""
    want = NetSpec.require(spec).training.device
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, device) -> None:
    """The global generators a fit draws from (dropout, edge dropout, the GAT α-dropout), seeded
    once; on the CPU the deterministic kernels too, so two fits are equal to the bit. The flag
    is process-wide, so it is set for every device: a CUDA fit after a CPU fit in one process
    runs as it would alone (its claim is the stated tolerance, backend-contract §1)."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    try:
        torch.use_deterministic_algorithms(torch.device(device).type == "cpu")
    except Exception:  # noqa: BLE001  (a build without the switch)
        pass


def grad_clip(spec: NetSpec):
    """The spec's `regularization[kind=grad_clip].value` — `training.grad_clip` when nothing
    declares it — or None (no clipping)."""
    spec = NetSpec.require(spec)
    bound = Regularization.of(spec).value("grad_clip", spec.training.grad_clip)
    return None if bound is None else float(bound)
