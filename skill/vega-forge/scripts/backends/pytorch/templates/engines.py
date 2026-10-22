"""
{{PACKAGE}}.{{MODULE_TRAINING}}.engines — the epoch sources of a fit (a `kind: support` file of
the Trainer's module).

Emitted by /vega-forge from `backends/pytorch/templates/engines.py`. A design defect here is
fixed in that template and the package re-emitted; never in this copy (code-contract §4).

An epoch source is where a fit's batches come from and how they are ordered (backend design
§10.2). `open_source` picks one from the sample's shape and the spec's `backend.hints` — the
measured substrate posture the blueprint binds, supplied from outside the loop (EJ-5). The
package carries its spec's sample shape and no other (`domain.CARRIED_SHAPES`, selected when
/vega-forge emitted it), and so does the table:

| sample | hints | source | train batches | valid batches | `shuffle.kind` |
|---|---|---|---|---|---|
# >>> shape: row
| row | none | `StreamedRows` | the part streamed, buffered shuffle seeded `seed + epoch` | streamed | `buffered` |
| row | `resident_tensors` | `HeldSamples` on the device | a global permutation, gathered | `slices(max(bs, 65536))` | `global-randperm` |
# <<< shape
# >>> shape: row window
| row, window | `resident_tensors` + `cuda_graph` on CUDA | `CapturedSteps` | the step replayed from one captured CUDA graph | as held | `global-randperm`, `graph: true` |
# <<< shape
# >>> shape: window
| window | none | `HeldSamples` on the host | a global permutation of the anchors | `slices(bs)` | `global-randperm` |
| window | `resident_tensors` | `HeldSamples` on the device | as rows | as rows | `global-randperm` |
# <<< shape
# >>> shape: graph
| graph | any | `graph_engines.py` | `FullGraph`, `SampledGraph`, `GraphMinibatches` | the full valid view | `full-graph`, ... |
# <<< shape
# >>> shape: window

A window sample is never streamed: its batches are cut from a held part, so E1 is gone.
# <<< shape

Every permutation is a `torch.Generator` seeded `spec.seed + epoch` (determinism,
backend-contract §1). `steps_per_epoch` is the source's own count — anchors, targets or
graphs, not the row count of `data_contract` — so a per-step schedule sees the real number of
steps.

`CallbackHooks` is the caller's callbacks (`ports.Callbacks`, three optional hooks) as exactly
those three. A source drives the epoch's steps
through a `StepDriver` (the Trainer's pass): `eager(batch)`
runs one full step, `pure(batch)` is the same arithmetic with no bookkeeping (what a CUDA graph
captures), `replayed(outcome, replay)` accounts a replayed step. `CapturedSteps` is the v2.2
resident pass, moved: the run's first step eagerly with every diagnostic, three warm-up steps
on a side stream (capture requires them), the capture of one step over static index tensors,
then a replay per full batch; the rows a full batch does not reach take one eager step.
"""
from __future__ import annotations

import contextlib
from types import MappingProxyType
from typing import Iterator, Protocol

import torch

from ..domain import (NetSpec, SampleShape, SampleSpec, on_device,
                      # >>> shape: row
                      BatchPlan,
                      # <<< shape
                      # >>> shape: row window
                      Part,
                      # <<< shape
                      )
from ..ports import TrainingData

__all__ = [
    # >>> shape: row window
    "CapturedSteps", "HeldSamples",
    # <<< shape
    # >>> shape: row
    "StreamedRows",
    # <<< shape
    "CallbackHooks", "EpochSource", "StepDriver", "batch_size", "cpu_threads", "hints_of",
    "open_source"]

# >>> shape: row window
# the rows a resident valid pass scores at a time (v2.2's resident pass)
_RESIDENT_VALID_ROWS = 65536
# <<< shape


class CallbackHooks:
    """The caller's callbacks as the three hooks of `Callbacks`, each forwarded when the caller
    defines it and a no-op when it does not. Only these three exist: a hook name the loop
    misspells fails where it is called, instead of resolving to a silent no-op."""

    def __init__(self, given):
        self._given = given

    def _forward(self, name: str, record) -> None:
        hook = getattr(self._given, name, None)
        if callable(hook):
            hook(record)

    def on_epoch(self, record) -> None:
        self._forward("on_epoch", record)

    def on_first_step(self, record) -> None:
        self._forward("on_first_step", record)

    def on_step(self, record) -> None:
        self._forward("on_step", record)


class StepDriver(Protocol):
    """The Trainer's pass over one epoch, as a source drives it."""

    first_pending: bool

    def eager(self, batch: dict) -> None: ...

    def pure(self, batch: dict): ...

    def replayed(self, outcome, replay) -> None: ...


class EpochSource(Protocol):
    """Where an epoch's batches come from (a local protocol, not a blueprint port)."""

    steps_per_epoch: int
    zero_to_none: bool      # `zero_grad(set_to_none=...)`: False keeps the gradients' storage
    capturable: bool        # the optimiser must be built capturable (a CUDA graph replays it)

    def train_batches(self, epoch: int) -> Iterator[dict]: ...

    def valid_batches(self) -> Iterator[dict]: ...

    def describe(self, epoch: int) -> dict: ...

    def run_epoch(self, epoch: int, driver: StepDriver) -> None: ...

    def guard(self): ...


def hints_of(spec: NetSpec) -> dict:
    """The spec's `backend.hints` — read here and nowhere else."""
    return dict(NetSpec.require(spec).hints)


#: a network this small fits fastest on the CPU with one intra-op thread: per-op thread
#: dispatch costs more than the op (measured 3.3x on 2,758 parameters, 20 cores)
SMALL_NETWORK_PARAMS = 100_000


@contextlib.contextmanager
def cpu_threads(spec: NetSpec, model, device):
    """A CPU fit's intra-op thread count, for the fit and restored after it: the spec's
    `backend.hints.cpu_threads`, else one thread for a network of at most
    `SMALL_NETWORK_PARAMS` parameters, else torch's own. Another device is left alone."""
    hint, before = hints_of(spec).get("cpu_threads"), torch.get_num_threads()
    small = sum(p.numel() for p in model.parameters()) <= SMALL_NETWORK_PARAMS
    want = int(hint) if hint else (1 if small else before)
    if torch.device(device).type != "cpu" or want == before:
        yield before
        return
    torch.set_num_threads(want)
    try:
        yield want
    finally:
        torch.set_num_threads(before)


def batch_size(spec: NetSpec, samples: int) -> int:
    """`training.batch_size`; "full" is every sample of the part in one batch."""
    return NetSpec.require(spec).training.batch_size.for_part(samples)


# >>> shape: row window
class _Source:
    """What every source shares: an epoch is its train batches, each one eager step."""

    zero_to_none = True
    capturable = False

    def run_epoch(self, epoch: int, driver: StepDriver) -> None:
        for batch in self.train_batches(epoch):
            driver.eager(batch)

    def guard(self):
        return contextlib.nullcontext()
# <<< shape


# >>> shape: row
class StreamedRows(_Source):
    """A row part read in chunks — exactly the v2.2 path (the golden history is bit-identical
    through it): a buffered shuffle over `training.shuffle_buffer` rows (default 64 batches),
    seeded `seed + epoch`. A part one buffer holds whole (`steps · batch ≤ buffer`) has no
    streaming left in it — each epoch would re-read and re-cut the same frame — so it is read
    once and each epoch's batches are the same permutation of it the stream would draw (one
    generator, `BatchPlan.rng`); the valid part one buffer holds, its batches kept once."""

    def __init__(self, spec: NetSpec, data: TrainingData, device):
        self._spec, self._data, self._device = spec, data, device
        training = spec.training
        self._bs = training.batch_size.streamed()
        self._buffer = BatchPlan(shuffle_buffer=training.shuffle_buffer).buffer_rows(self._bs)
        self._steps = None
        self._held: dict = {}           # part -> the held train part / valid batches, or None

    def _plan(self, epoch: int) -> BatchPlan:
        return BatchPlan(shuffle=True, seed=int(self._spec.seed + epoch),
                         shuffle_buffer=self._buffer)

    @property
    def steps_per_epoch(self) -> int:
        """The stream's own batch count under the training plan, counted once and without a
        pass over the data (a per-step schedule must not step past its total)."""
        if self._steps is None:
            self._steps = max(1, self._data.batch_count(self._spec, Part.TRAIN,
                                                        batch_size=self._bs, plan=self._plan(0)))
        return self._steps

    def train_batches(self, epoch: int) -> Iterator[dict]:
        plan, held = self._plan(epoch), self._whole(Part.TRAIN)
        if held is None:
            batches = self._data.iter_batches(self._spec, Part.TRAIN, batch_size=self._bs,
                                              plan=plan)
        else:
            order = plan.rng(self._spec.seed).permutation(len(held))
            batches = (held.gather(torch.as_tensor(order[start:start + self._bs]))
                       for start in range(0, len(held), self._bs))
        for batch in batches:
            yield on_device(batch, self._device)

    def valid_batches(self) -> Iterator[dict]:
        held = self._whole(Part.VALID)
        batches = held if held is not None else self._data.iter_batches(
            self._spec, Part.VALID, batch_size=self._bs)
        for batch in batches:
            yield on_device(batch, self._device)

    def _whole(self, part: Part):
        """The part, read once, when one buffer holds it whole — the train part held (each
        epoch permutes it), the valid part's batches kept (in the stream's order) — else None:
        a part larger than a buffer stays streamed."""
        if part not in self._held:
            plan = self._plan(0) if part is Part.TRAIN else BatchPlan()
            count = self._data.batch_count(self._spec, part, batch_size=self._bs, plan=plan)
            whole = count * self._bs <= self._buffer
            self._held[part] = (None if not whole else
                                self._data.load_part(self._spec, part) if part is Part.TRAIN
                                else list(self._data.iter_batches(self._spec, part,
                                                                  batch_size=self._bs)))
        return self._held[part]

    def describe(self, epoch: int) -> dict:
        return {"kind": "buffered", "buffer_rows": self._buffer, "seed": int(self._spec.seed + epoch)}
# <<< shape


# >>> shape: row window
class HeldSamples(_Source):
    """A held part (rows or windows), permuted whole each epoch and gathered by index. On the
    device (`resident_tensors`) every minibatch is a gather there; on the host (a window
    sample without the hint) the gathered batch is moved per step."""

    def __init__(self, spec: NetSpec, data: TrainingData, device, resident: bool):
        self._spec, self._device = spec, torch.device(device)
        self._where = self._device if resident else torch.device("cpu")
        self.zero_to_none = not resident          # v2.2's resident pass kept the storage
        if resident:
            self.train = data.load_resident(spec, Part.TRAIN, self._where)
            self.valid = data.load_resident(spec, Part.VALID, self._where)
        else:
            self.train, self.valid = data.load_part(spec, Part.TRAIN), data.load_part(spec, Part.VALID)
        self.n = len(self.train)
        self.bs = batch_size(spec, self.n)
        self._valid_bs = max(self.bs, _RESIDENT_VALID_ROWS) if resident else self.bs

    @property
    def steps_per_epoch(self) -> int:
        return max(1, -(-self.n // self.bs))

    def order(self, epoch: int):
        """This epoch's permutation of the samples, drawn where they are held."""
        generator = torch.Generator(device=self._where)
        generator.manual_seed(int(self._spec.seed + epoch))
        return torch.randperm(self.n, generator=generator, device=self._where)

    def train_batches(self, epoch: int) -> Iterator[dict]:
        perm = self.order(epoch)
        for start in range(0, self.n, self.bs):
            yield on_device(self.train.gather(perm[start:start + self.bs]), self._device)

    def valid_batches(self) -> Iterator[dict]:
        for batch in self.valid.slices(self._valid_bs):
            yield on_device(batch, self._device)

    def describe(self, epoch: int) -> dict:
        out = {"kind": "global-randperm", "device": str(self._where),
               "seed": int(self._spec.seed + epoch), "graph": False, "rows": self.n}
        # >>> shape: window
        if SampleSpec.from_spec(self._spec).window is not None:
            out["sampling"] = {"windows": self.n, "unscored": len(self.train.unscored)}
        # <<< shape
        return out


class CapturedSteps:
    """A resident held part (a `HeldSamples` it holds, never extends) on a CUDA device, with
    every full batch's step replayed from one captured CUDA graph. The step is launch-bound,
    so replaying one graph instead of a launch per kernel is the measured gain; the gathers
    are static-shaped (rows and windows alike), so the capture holds them. A part with too
    few full batches to warm up runs eagerly. The held part's permutation, valid batches and
    description are its own; what this source adds is how a train epoch's steps are run."""

    #: eager steps before the capture: the diagnosed first step, then three on a side stream
    WARM_STEPS = 4
    capturable = True
    zero_to_none = False

    def __init__(self, spec: NetSpec, data: TrainingData, device):
        self._held = HeldSamples(spec, data, device, resident=True)
        self._device = torch.device(device)
        self._bs = self._held.bs
        self._full = self._held.n // self._bs
        self._use_graph = self._device.type == "cuda" and self._full > self.WARM_STEPS
        self._graph, self._static, self._warmed = None, None, 0
        self._idx = torch.zeros(self._bs, dtype=torch.long, device=self._device)

    @property
    def steps_per_epoch(self) -> int:
        return self._held.steps_per_epoch

    def train_batches(self, epoch: int) -> Iterator[dict]:
        return self._held.train_batches(epoch)

    def valid_batches(self) -> Iterator[dict]:
        return self._held.valid_batches()

    def guard(self):
        return contextlib.nullcontext()

    def run_epoch(self, epoch: int, driver: StepDriver) -> None:
        perm, train = self._held.order(epoch), self._held.train
        for i in range(self._full):
            idx = perm[i * self._bs:(i + 1) * self._bs]
            if driver.first_pending or not self._use_graph:
                driver.eager(train.gather(idx))
            elif self._graph is not None:
                self._idx.copy_(idx)
                driver.replayed(self._static, self._graph.replay)
            else:
                self._warm(driver, idx)
        tail = perm[self._full * self._bs:]
        if len(tail):
            driver.eager(train.gather(tail))

    def _warm(self, driver: StepDriver, idx) -> None:
        """A real training step on a side stream; after the last one, the capture."""
        train = self._held.train
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._idx.copy_(idx)
            driver.eager(train.gather(self._idx))
        torch.cuda.current_stream().wait_stream(side)
        self._warmed += 1
        if self._warmed >= self.WARM_STEPS - 1:
            torch.cuda.synchronize()
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._static = driver.pure(train.gather(self._idx))
            torch.cuda.synchronize()

    def describe(self, epoch: int) -> dict:
        return {**self._held.describe(epoch), "graph": self._graph is not None}
# <<< shape


# >>> shape: row
def _streamed(spec, data, device, hints):
    return StreamedRows(spec, data, device)
# <<< shape


# >>> shape: window
def _held_host(spec, data, device, hints):
    return HeldSamples(spec, data, device, resident=False)
# <<< shape


# >>> shape: row window
def _held_resident(spec, data, device, hints):
    return HeldSamples(spec, data, device, resident=True)


def _captured(spec, data, device, hints):
    return CapturedSteps(spec, data, device)


def _posture(hints: dict, device) -> str:
    if not hints.get("resident_tensors"):
        return "none"
    return "captured" if hints.get("cuda_graph") and torch.device(device).type == "cuda" else "resident"
# <<< shape


# >>> shape: graph
def _graph(spec, data, device, hints):
    # graph_engines imports this module's batch_size and on_device: imported on use
    from .graph_engines import graph_source
    return graph_source(spec, data, device, hints)


def _any_posture(hints: dict, device) -> str:
    return "any"                 # a graph source reads the hints itself
# <<< shape


# sample shape -> (the posture its source is chosen by, {posture: source}) (D1 §10.2): the
# shapes this package carries (domain.CARRIED_SHAPES), no other. A row or window posture is
# "none", "resident" or "captured" (resident_tensors + cuda_graph on a CUDA device).
_SOURCES = MappingProxyType({
    # >>> shape: row
    SampleShape.ROW: (_posture, MappingProxyType({
        "none": _streamed, "resident": _held_resident, "captured": _captured})),
    # <<< shape
    # >>> shape: window
    SampleShape.WINDOW: (_posture, MappingProxyType({
        "none": _held_host, "resident": _held_resident, "captured": _captured})),
    # <<< shape
    # >>> shape: graph
    SampleShape.GRAPH: (_any_posture, MappingProxyType({"any": _graph})),
    # <<< shape
})


def open_source(spec: NetSpec, data: TrainingData, device, hints=None) -> EpochSource:
    """The epoch source for this spec's sample, on `device`, under `hints`."""
    spec = NetSpec.require(spec)
    hints = dict(hints if hints is not None else hints_of(spec))
    posture, sources = _SOURCES[SampleSpec.from_spec(spec).shape]
    return sources[posture(hints, device)](spec, data, device, hints)
