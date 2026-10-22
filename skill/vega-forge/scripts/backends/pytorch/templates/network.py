"""
{{PACKAGE}}.{{MODULE_MODEL}}.network — the {{PORT_MODEL}} adapter (role: Model).

{{RESPONSIBILITY_MODEL}}
Source: {{SOURCE_MODEL}}

Emitted by /vega-forge from `backends/pytorch/templates/network.py`. The blocks live beside it
in the `blocks_*.py` files the spec's block types need (`blocks_core.py` always; the map's rows
name them), the builder in `builder.py`, the reading of the head in `readout.py`: the package imports
the framework and its own modules, nothing of the skills, because /vega-pack copies it into a
bundle that must run alone. A defect is fixed in the template and re-emitted, never here.

Spec digest: {{SPEC_DIGEST}}
Blocks: {{BLOCK_TYPES}}

`Network.build(spec)` — exported as `build_model`, the name the ports and C9 read — is the only
constructor of the network (code-contract §4): it seeds, asks `NetworkBuilder` for the stages,
constructs, and asks the builder to initialise. A change to the network is a change to the
spec, re-emitted.

The network (BUILD-PLAN R-A2, R-B1, R-B4–R-B7; D1 §7.2):

- `stages`, `ids`, `kinds`, `shapes` (and `types`) are read-only and aligned 1:1 with
  `spec.blocks`; `stages` is the `nn.ModuleList` whose module paths are `stages.<i>`;
- `forward(batch)` walks the stages through `_STEP`, a constant table from each `StageKind` to
  the call its blocks take (the batch keys passed positionally, R-B2), and returns outputs
  aligned with `batch["y"]`: at graph level node the head's rows are gathered at
  `target_index` **after** the head, once (the Predictor never gathers again);
- `score` and `predict_proba` read the head through the spec's `TaskReadout`, the one source
  of output semantics; nothing in a forward pass writes an attribute (R-B4), and a network
  already in eval mode is never switched, so a served model is never mutated by a request;
- under the svdd objective the hypersphere centre is the pair of registered buffers
  `svdd_center` / `svdd_center_set` (R-B6), written once by the loss through
  `objective_center()` and carried by `state_dict`;
- a regression-family head ends on the fixed train-fitted target affine (A37): the head's
  buffers `target_shift` / `target_scale`, written once by the Trainer through
  `fit_target_affine` from the train target and carried by `state_dict` with `target_fitted`,
  so a resumed run keeps them — the output is in the target's own units;
- there is no `describe` or `trace` (R-B5): `anatomy()` hands the serving and inspection
  code a read-only view of the network (`domain.NetworkAnatomy`: the stages, the readout, the
  identity, and — read when asked — the mode, the device, the centre and the forward), the
  Model port's surface they read beside `head`;
  the inspection machinery's `StageGraph` and `Capture` read the stages through it and the
  blocks' inspection surface.

Effective Java: EJ-1 (a named static factory), EJ-2 (the builder reads the spec's blocks in
order), EJ-17 (the structure is fixed at construction), EJ-18 (the network is composed of
blocks), EJ-34 (the stage kinds are the domain's enum), EJ-64 (`build` takes the NetSpec).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from ..domain import (BatchKey, NetSpec, NetworkAnatomy, Objective, SampleSpec, StageKind,
                      device_of, on_device)
# >>> task: regression forecast
from ..domain import TargetAffine
# <<< task
from .builder import NetworkBuilder
from .readout import TaskReadout


@dataclass(frozen=True)
class _Flow:
    """The running state of one forward pass: the batch's inputs, the pending embedding, the
    running tensor `h` and the graph the next graph block reads (edge_dropout replaces it)."""

    x_num: Any
    x_cat: Any
    h: Any
    emb: Any = None
    # >>> shape: graph
    edge_index: Any = None
    edge_weight: Any = None
    graph_index: Any = None
    num_graphs: Any = None
    # <<< shape

    @staticmethod
    def start(batch) -> "_Flow":
        x_num = batch.get(BatchKey.X_NUM)
        return _Flow(x_num=x_num, x_cat=batch.get(BatchKey.X_CAT), h=x_num,
                     # >>> shape: graph
                     edge_index=batch.get(BatchKey.EDGE_INDEX),
                     edge_weight=batch.get(BatchKey.EDGE_WEIGHT),
                     graph_index=batch.get(BatchKey.GRAPH_INDEX),
                     num_graphs=batch.get(BatchKey.NUM_GRAPHS),
                     # <<< shape
                     )


def _emb(module, state):
    return replace(state, emb=module(state.x_cat))


def _concat(module, state):
    return replace(state, h=module(state.x_num, state.emb))


def _tokens(module, state):
    return replace(state, h=module(state.x_num, state.x_cat))


def _tensor(module, state):
    return replace(state, h=module(state.h))


# >>> shape: graph
def _graph(module, state):
    return replace(state, h=module(state.h, state.edge_index, state.edge_weight))


def _graph_drop(module, state):
    edge_index, edge_weight = module(state.edge_index, state.edge_weight)
    return replace(state, edge_index=edge_index, edge_weight=edge_weight)


def _graph_pool(module, state):
    return replace(state, h=module(state.h, state.graph_index, state.num_graphs))
# <<< shape


# how a stage of each kind reads and updates the running state (R-B1, R-B2)
_STEP = MappingProxyType({
    StageKind.EMB: _emb, StageKind.CONCAT: _concat, StageKind.TOKENS: _tokens,
    StageKind.TENSOR: _tensor,
    # >>> shape: graph
    StageKind.GRAPH: _graph, StageKind.GRAPH_DROP: _graph_drop, StageKind.GRAPH_POOL: _graph_pool,
    # <<< shape
})


def _parameters(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


class Network(nn.Module):
    """The spec's blocks in order, called by stage kind; outputs aligned with `batch["y"]`."""

    def __init__(self, stages, readout: TaskReadout, sample: SampleSpec, spec_digest: str):
        super().__init__()
        if not stages:
            raise ValueError("a network needs at least one block; the spec lists none")
        readout.check_head(stages[-1].out_dims[-1], f"the last block ({stages[-1].id})")
        self.stages = nn.ModuleList([s.module for s in stages])
        self._ids = tuple(str(s.id) for s in stages)
        self._types = tuple(str(s.type) for s in stages)
        self._kinds = tuple(StageKind(s.kind) for s in stages)
        self._shapes = tuple(tuple(int(d) for d in s.out_dims) for s in stages)
        self._readout, self._sample, self._digest = readout, sample, spec_digest
        if readout.objective is Objective.SVDD:
            self.register_buffer("svdd_center", torch.zeros(self._shapes[-1][-1]))
            self.register_buffer("svdd_center_set", torch.zeros((), dtype=torch.bool))

    def __setattr__(self, name, value):
        if name == "stages" and "stages" in self.__dict__.get("_modules", {}):
            raise AttributeError("Network.stages is read-only: the stages are the spec's "
                                 "blocks, and a change to them is a change to the spec")
        super().__setattr__(name, value)

    @staticmethod
    def build(spec: NetSpec, run_dir=None) -> "Network":
        """The one constructor (C9). The spec's seed is applied before a parameter is drawn, so
        two builds of one spec are the same network."""
        spec = NetSpec.require(spec)
        torch.manual_seed(int(spec.seed))
        builder = NetworkBuilder(spec, run_dir)
        stages = builder.stages()
        net = Network(stages, TaskReadout.for_spec(spec), SampleSpec.from_spec(spec),
                      spec.digest)
        return builder.initialise(net)

    # ------------------------------------------------------------------ the structure

    @property
    def ids(self) -> tuple:
        return self._ids

    def named_stages(self) -> tuple:                  # ((id, stage), ...): the record's per block
        return tuple(zip(self._ids, self.stages))

    @property
    def kinds(self) -> tuple:                         # one StageKind per stage (forward's call)
        return self._kinds

    @property
    def shapes(self) -> tuple:                        # per stage: (width,) or (seq, width)
        return self._shapes

    @property
    def types(self) -> tuple:
        return self._types

    @property
    def readout(self) -> TaskReadout:
        return self._readout

    @property
    def task(self):                                   # the domain's Task
        return self._readout.task

    @property
    def sample(self) -> SampleSpec:
        return self._sample

    @property
    def spec_digest(self) -> str:
        return self._digest

    device = property(device_of)                      # where the tensors live (CPU for none)

    def anatomy(self) -> NetworkAnatomy:
        """The Model port's read-only view of this network, all serving reads beside `head`."""
        return NetworkAnatomy(self)

    # ------------------------------------------------------------------ the pass

    def forward(self, batch):
        """The head's raw outputs (the loss input), aligned with `batch["y"]`."""
        state = _Flow.start(batch)
        for kind, module in zip(self._kinds, self.stages):
            state = _STEP[kind](module, state)
        return self._aligned(state.h, batch)

    def _aligned(self, out, batch):                   # at level node: the targets' rows only
        target = batch.get(BatchKey.TARGET_INDEX) if self._sample.is_node_level else None
        return out if target is None else out.index_select(0, target)

    def _flows(self, batch):                          # the state after each stage (forward's)
        state = _Flow.start(batch)
        for kind, module in zip(self._kinds, self.stages):
            state = _STEP[kind](module, state)
            yield state

    @contextmanager
    def _scoring(self, batch):
        """The batch on the network's device, in eval mode (switched back after, when it was
        training)."""
        was = self.training
        if was:
            self.eval()
        try:
            yield on_device(batch, device_of(self))
        finally:
            if was:
                self.train(True)

    @torch.no_grad()
    def score(self, batch):
        """What the metrics read, per the task (`TaskReadout.score`); eval mode, no grad."""
        with self._scoring(batch) as b:
            return self._readout.score(self.forward(b), b, self)

    @torch.no_grad()
    def row_score(self, batch):
        """The per-row score of an unsupervised task (`TaskReadout.row_score`); eval, no grad."""
        with self._scoring(batch) as b:
            return self._readout.row_score(self.forward(b), b, self)

    def sample_view(self, dataset):
        """The part as the network reads it: the part itself (ScoredModel)."""
        return dataset

    @torch.no_grad()
    def head(self, batch):
        """The raw head of one batch, the stored `score_k` columns (ScoredModel); eval, no grad."""
        with self._scoring(batch) as b:
            return self.forward(b)

    @torch.no_grad()
    def predict_proba(self, batch):
        """The Estimator port's method: the task's distribution — σ of a binary logit, the
        softmax, σ per label, an ordinal's class probabilities — or its value where it has
        none (`TaskReadout.proba`); eval mode, no grad."""
        with self._scoring(batch) as b:
            return self._readout.proba(self.forward(b), b, self)

    # ------------------------------------------------------------------ the numbers

    def param_count(self) -> tuple:
        """(total, trainable) parameters; buffers are not parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return int(total), int(trainable)

    @torch.no_grad()
    def block_shapes(self, batch) -> list:
        """[{id, out_shape, params}] per stage — the numbers the architecture layer shows. The
        batch axis (rows, nodes or graphs) is dropped; an embedding stage reports its output,
        every other stage the running tensor it leaves."""
        with self._scoring(batch) as b:
            rows = []
            for i, state in enumerate(self._flows(b)):
                t = state.emb if self._kinds[i] is StageKind.EMB else state.h
                rows.append({"id": self._ids[i],
                             "out_shape": [] if t is None else [int(s) for s in t.shape[1:]],
                             "params": _parameters(self.stages[i])})
            return rows

    # >>> task: unsupervised anomaly
    def kl(self, batch=None):
        """The VAE's KL term. Without a batch: the one the last *training* forward computed
        (an eval forward writes nothing, R-B4). With a batch: KL(q(z|x) ‖ N(0, I)) for that
        batch, computed purely through the reparameterize stage's `divergence` — what an eval
        loss reads. None when the network has no reparameterize stage."""
        if batch is not None:
            return self._divergence(batch)
        return next((m.kl for m in self.stages if getattr(m, "kl", None) is not None), None)

    def _divergence(self, batch):
        state = _Flow.start(batch)
        for kind, module in zip(self._kinds, self.stages):
            if callable(getattr(module, "divergence", None)):
                return module.divergence(state.h)
            state = _STEP[kind](module, state)
        return None
    # <<< task

    # >>> task: regression forecast
    @property
    def target_affine_pending(self) -> bool:
        """A37: a regression-family head whose target affine was not yet fitted — a fresh
        network; a checkpoint carries `target_fitted`, so a resumed one is not pending."""
        flag = getattr(self.stages[-1], "target_fitted", None)
        return flag is not None and not bool(flag)

    def fit_target_affine(self, affine: TargetAffine | None) -> bool:
        """The head's target affine written once (a width other than the reading's is
        refused); False, and nothing changed, when none is pending or `affine` is None."""
        if affine is None or not self.target_affine_pending:
            return False
        self._readout.check_head(len(affine.shift), f"the target affine of {self._ids[-1]}")
        self.stages[-1].fit_target(affine.shift, affine.scale)
        return True
    # <<< task

    # >>> task: anomaly
    def objective_center(self):
        """The svdd centre buffer (z,) — the loss writes it once with `copy_` and sets
        `svdd_center_set` — or None under any other objective."""
        return getattr(self, "svdd_center", None)              # a buffer, through nn.Module
    # <<< task


build_model = Network.build

__all__ = ["Network", "build_model"]
