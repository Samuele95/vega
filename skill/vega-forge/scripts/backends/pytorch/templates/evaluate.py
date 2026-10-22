"""
{{PACKAGE}}.{{MODULE_EVALUATION}}.evaluate — the {{PORT_EVALUATOR}} adapter (role: Evaluator).

{{RESPONSIBILITY_EVALUATOR}}
Source: {{SOURCE_EVALUATOR}}

Emitted by /vega-forge from `backends/pytorch/templates/evaluate.py`. A defect is fixed in the
template and re-emitted, never in this copy (code-contract §4).

`Evaluator.restoring(data, store, build)` and `Evaluator.over(data)` are the two ways to make one,
each with a complete set of collaborators (EJ-1: a named factory per construction need, never a
constructor whose optional ports are required later): the first restores a checkpoint through the
ModelStore and the Model's builder, the second scores only a model handed in (a baseline fitted
in memory) and restores none. The ports arrive by factory (never sibling imports: EJ-20, EJ-64),
and every number comes from `{{MODULE_MODEL}}/metrics.py`, the code the trainer's valid numbers came from.
The protocol — the seal file, the baselines' fitting, the bootstrap, the slices, the robustness
readings — is /vega-eval's own script, which calls this adapter; a bundle can therefore carry
an evaluator without carrying the evaluation campaign.

`scored(spec, dataset, model)` is the **one scoring path** (backend design §11.1: the Model
module's `metrics.ScoringPass`, which the Trainer's valid numbers are made by too), for the
network and for every baseline alike — each is a Model, asked through the port's `score`:

  * the samples in stored order, through `data.make_loader` (a node-level graph in one
    full-batch pass over its view; graphs collated whole; windows densely, one per scorable
    row) — after the model's own `sample_view` of the part, when it names one (a baseline
    that reads the target's history);
  * `score` per the score contract (the raw head's reading: logits, values, per-row scores);
    `truth` keeps its dimension ((N,), (N, K), (N, H)), is the numeric input for a
    reconstruction and None for an unlabelled anomaly;
  * `row_score` (R-C2) for the reconstruction and anomaly readings: the anomaly score itself,
    or — unsupervised — `TaskReadout.row_score` of the network's head (numeric MSE plus the
    mean nominal cross-entropy), a baseline answering for itself;
  * `row_ids` from the part (the anchor's or the node's raw position, the graph id at graph
    level, with `graph_ids` beside it), and the part's `unscored` rows with their reasons.

`heads(spec, dataset, model)` is the same pass for the raw head — every `head_out` column the
network's forward returns (a baseline's own `head`), in `scored`'s order: what R-C2's stored
`score_k` columns hold, where the score contract reads only a part of the head (a quantile's
median, the numeric reconstruction, a distance to the centre).

`score_part` (train, valid) and `score_sealed_test` (the one door to the test part, with the
seal's token; the checkpoint is verified before the part is opened) end in `measure`: the
Trainer's one metric pass, `metrics.compute`, recorded the one way `metrics.record` writes it
(a metric the part cannot give as None, its reason under `_unavailable`); the metric
function's own failure modes are the only ones caught. The caller's `MetricOptions` (the threshold /vega-eval chose on
valid) arrive by factory or `with_options`, the spec filling the quantile levels the caller left
unset, and the run and the seal exposure are stamped from the seal on the sealed test's record
(`Metrics.with_provenance`).
# >>> shape: window
A forecast's `mase` reads the train target's seasonal scale (`data.target_scale`, at the spec's
`sample.season`, which also fills the season the caller left unset).
# <<< shape
"""
from __future__ import annotations

import numpy as np

from ..domain import (MetricOptions, Metrics, NetSpec, Part, SamplePart, ScoredPart,
                      SealedPartError, SealReading, Task, host_array)
from ..ports import ModelBuilder as _Builder, ScoredModel as _Scored
from ..ports import {{PORT_DATASOURCE}} as _DataSource, {{PORT_MODELSTORE}} as _Store
from ..{{MODULE_MODEL}} import TaskReadout, metrics as _metrics

__all__ = ["Evaluator"]



def _head(model, batch):
    """The raw head of one batch — the network's and a baseline's alike, through the declared
    `ScoredModel.head` (no type test: every model the Evaluator scores answers it)."""
    return host_array(model.head(batch))


class _Restorer:
    """The ModelStore and the Model's builder, together: what restoring a checkpoint takes."""

    def __init__(self, store: _Store, build: _Builder):
        self._store, self._build = store, build

    def load(self, checkpoint, spec: NetSpec):
        return self._store.load(checkpoint, spec, self._build)


class Evaluator:
    """The Evaluator port's adapter: one scoring path, metrics through `metrics.py`. Made by
    `Evaluator.restoring(data, store, build)` or `Evaluator.over(data)`."""

    def __init__(self, data: _DataSource, restorer: _Restorer | None,
                 options: MetricOptions | None):
        """Through the two factories: the DataSource, how a checkpoint is restored (None: this
        evaluator scores only the models handed to it) and the metric knobs."""
        self._data, self._restorer, self._options = data, restorer, options

    @classmethod
    def restoring(cls, data: _DataSource, store: _Store, build: _Builder,
                  options: MetricOptions | None = None) -> "Evaluator":
        """An evaluator that scores a model handed in or restores one from a checkpoint,
        through the ModelStore and the Model's builder."""
        return cls(data, _Restorer(store, build), options)

    @classmethod
    def over(cls, data: _DataSource, options: MetricOptions | None = None) -> "Evaluator":
        """An evaluator that scores the models handed to it (a baseline fitted in memory, a
        network already built) and restores none."""
        return cls(data, None, options)

    def with_options(self, options: MetricOptions) -> "Evaluator":
        """The same ports, reading every metric with `options` (the chosen threshold, ...)."""
        return Evaluator(self._data, self._restorer, options)

    def metric_names(self, spec: NetSpec) -> list:
        """The trainer's derivation of the metric names, primary first."""
        return list(_metrics.names_for(NetSpec.require(spec)))

    def scored(self, spec: NetSpec, dataset: SamplePart, model: _Scored) -> ScoredPart:
        """`dataset` (a part `load_part` or `load_sealed_part` returned) scored by `model`."""
        spec = NetSpec.require(spec)
        dataset = self._view(dataset, model)
        scoring = _metrics.ScoringPass(TaskReadout.for_spec(spec))
        for batch in self._data.make_loader(dataset, spec, False):
            scoring.add(model, batch)
        return scoring.scored(dataset)

    def heads(self, spec: NetSpec, dataset: SamplePart, model: _Scored):
        """The raw head of every sample of `dataset`, in `scored`'s order: (N, head_out), or
        (N,) for a head that is one per-row score (a baseline's isolation forest)."""
        spec = NetSpec.require(spec)
        dataset = self._view(dataset, model)
        outs = [_head(model, batch) for batch in self._data.make_loader(dataset, spec, False)]
        if len(outs) and sum(len(o) for o in outs) != len(dataset.row_ids):
            raise ValueError("the heads and the part's row ids disagree on what a sample is")
        return np.concatenate(outs, axis=0) if outs else np.zeros((0,))

    def score_part(self, spec: NetSpec, part: Part, checkpoint=None, model=None) -> Metrics:
        """Metrics on `train` or `valid`, of `model` or of the model `checkpoint` restores."""
        spec, part = NetSpec.require(spec), Part(part)
        if part is Part.TEST:
            raise SealedPartError("score_part does not open the test part; score_sealed_test "
                                  "does, with the token from the campaign's test_seal.json")
        model = self._model(spec, checkpoint, model)
        return self.measure(spec, part, self.scored(spec, self._data.load_part(spec, part), model))

    def score_sealed_test(self, spec: NetSpec, token: str, checkpoint=None, model=None,
                          seal_path: str = "test_seal.json") -> Metrics:
        """The one look at the test part, through the DataSource's one door; the model is
        restored (and its checkpoint verified) before the part is opened."""
        spec = NetSpec.require(spec)
        model = self._model(spec, checkpoint, model)
        dataset = self._data.load_sealed_part(spec, token, seal_path)
        seal = dataset.seal or SealReading()       # read once, by the door that checked it
        return self.measure(spec, Part.TEST, self.scored(spec, dataset, model)).with_provenance(
            seal.run, seal.exposures)

    def measure(self, spec: NetSpec, part: Part, scored: ScoredPart) -> Metrics:
        """The spec's metrics of a scored part through the Trainer's one metric pass
        (`metrics.compute`), the record `metrics.record` writes: the unavailable ones as None,
        with their reasons under `_unavailable`."""
        spec = NetSpec.require(spec)
        names = _metrics.names_for(spec)
        options = (self._options or MetricOptions()).stated(spec)
        # >>> shape: window
        options = options.scaled(spec, names, self._data)
        # <<< shape
        values, unavailable = _metrics.compute(names, scored.truth, scored.score, options,
                                               Task.from_spec(spec))
        return Metrics.of_values(Part(part), _metrics.record(values, unavailable))

    @staticmethod
    def _view(dataset, model: _Scored):
        """The part as `model` reads it (`sample_view`: the part itself, or a baseline's view
        with the target's history)."""
        return model.sample_view(dataset)

    def _model(self, spec: NetSpec, checkpoint, model):
        """The model to score: the one handed in (a baseline fitted in memory, a network
        already built), or the one `checkpoint` restores through the store."""
        if model is not None:
            return model
        if checkpoint is None:
            raise ValueError("scoring needs a `model` (a Model, network or baseline) or a "
                             "`checkpoint` to restore one from")
        if self._restorer is None:
            raise ValueError(f"this Evaluator was made by Evaluator.over(data): it scores the "
                             f"model handed to it and restores none, so {checkpoint} cannot be "
                             "restored; make it with Evaluator.restoring(data, store, build)")
        return self._restorer.load(checkpoint, spec)

