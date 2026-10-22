# Vega backend contract — what a backend must emit, and the PyTorch backend

The netspec is framework-agnostic; a **backend** is the emitter that turns it into runnable code. One backend ships — `pytorch` — and it is the reference implementation of this contract. A second backend (Keras, JAX/Flax, a non-Python target) is a new module under `vega-forge/scripts/backends/<name>/` that satisfies the same contract, registered in one place and with its block coverage declared (§3); the three downstream skills need no change, because `/vega-train`, `/vega-eval` and `/vega-pack` talk to the emitted code only through the interface below.

## 1. What every backend emits into the run directory

The files below are the **adapters** of the blueprint's ports, laid out in the package `code-contract.md` §2 describes (`<package>/model/network.py` is what this contract calls `model.py`, `<package>/ingestion/data.py` is `data.py`, `<package>/training/loop.py` is `train_loop.py`, `<package>/model/metrics.py` is `metrics.py`, `<package>/evaluation/evaluate.py` is `evaluate.py`, `<package>/model/store.py` is `store.py`; the short names are prose, the package paths are the files). A backend also emits `ports.py` (one Protocol per blueprint port), `domain.py` (the artifacts as frozen value objects and the enums), the adapters of the remaining ports (`serving/predict.py`, `serving/monitor.py`), the support files its adapters use (map entries `kind: support`), and `blueprint-map.json`. All of it comes from `backends/<name>/templates/`, where the *Effective Java* rules of `code-contract.md` §4 are kept — a design defect is fixed in the template and re-emitted, never in a run's copy.

**Signatures.** Each adapter module holds exactly one public class, the adapter, beside `_prefixed` helpers, constants and imports (`model.py` also holds the one alias `build_model = Network.build`). **No port method takes more than five parameters** (`self` and `cls` not counted): the rest travel in frozen value objects of `domain.py` — `RunContext` (the budget, the checkpoint directory, the gradient sampling — the ModelStore is the Trainer's constructor collaborator, not a field), `BatchPlan` (shuffle, seed, buffer; `buffer_rows(batch_size)` is the one default of the buffer and `rng(seed)` the one generator, which the DataSource shuffles with and the epoch source records and replays), `MetricOptions` (threshold, k, rate, bins, season, the `mase` scale, quantiles; `MetricOptions.stated(spec)` is the one resolution of what the spec states — the quantile loss's levels, a window's season — and a window package's `scaled(spec, names, data)` adds `mase`'s train scale from the DataSource; the Trainer and the Evaluator both call them), `BaselineContext` (z, protocol, season, horizon). `**kwargs` survive only in `_prefixed` helpers. The blueprint's `signature[]` for each port follows the same rule (the patched `/blueprint` declares it).

| file | must provide |
|---|---|
| `model.py` | `class Network` and `build_model = Network.build`: `Network.build(spec: NetSpec, run_dir=None) -> Network` — the network from the spec's `blocks`, `init` applied (`NetSpec` is the `domain.py` value object; `NetSpec.from_dict(d)` is where a mapping becomes a spec, at the entry points, and its sections are read through its typed views); `forward(batch) -> outputs` aligned with `batch["y"]` (§1.1); `score(batch)` and `predict_proba(batch)` through the one score contract (§1.2); `param_count() -> (total, trainable)`; `block_shapes(batch) -> [{id, out_shape, params}]`; `kl()`; `objective_center()` (the one-class centre, or `None`); read-only `stages`, `ids`, `kinds`, `shapes`, aligned one to one with the spec's blocks; and the Model port's serving seam (`code-contract.md` §1): `head(batch)` (the raw head of an assembled batch, eval mode, no gradient, on the network's device) and `anatomy() -> NetworkAnatomy` (a read-only view of the network: the stages, readout, sample, spec digest, and — read when asked — the mode, device, one-class centre and forward; the surface the Predictor and `inspect-contract.md` §3 read, and all they read) |
| `data.py` | `class ShapedParts` (the DataSource adapter): `load_part(spec, part) -> SamplePart` for `train` and `valid`; **`load_part(spec, "test")` raises `SealedPartError`** before any file is opened — the only code in the sub-chain that may read the test part whole (its labels included) is `/vega-eval`'s evaluator, and the one other way into the test file is the transductive door: `open_features_only(spec, "test")` on the part reader (`PartFrames`, the support file `ingestion/frames.py`) returns the test part's feature columns and never its label, and refuses unless the spec is a `graph` sample whose `protocol` is `transductive` with a `user:`-sourced `sample.exposure` (netspec-contract §1) — the graph view builds a transductive training graph through it, the forge's smoke pass included. The evaluator calls `load_sealed_part(spec, token, seal_path="test_seal.json")`: it reads the test path and digest from the manifest the spec names (`outputs.files.test`), checks the token against `test_seal.json` in the campaign directory (a file only `/vega-eval` writes, so the trainer and the forge cannot hold a valid token), and returns the part with its raw row ids; `iter_batches(spec, part, batch_size=None, columns=None, plan: BatchPlan = None)`; `make_loader(dataset, spec, shuffle=False, seed=None, batch_size=None) -> iterable of batches`; `load_resident(spec, part, device)`; `design_matrix(spec, dataset, view)` (the baselines' input); `exposure(spec)` (a graph's disclosed exposure, or `None`); `target_scale(spec, season=1)` (`mase`'s train-side scale); windowing for `sample.shape = window` and graph views for `graph` (`netspec-contract.md` §1); the shaper's nullable-integer nominals are cast to the embedding's index dtype, and a value equal to the encode step's `unseen_category_id` is a valid index (the reserved row) |
| `train_loop.py` | `class Trainer`, made with the ModelStore its checkpoints are written through (`Trainer(store=None)`): `fit(model: TrainedNetwork, spec: NetSpec, data: TrainingData, callbacks: Callbacks | None = None, run: RunContext = None) -> FitResult` (the loaders come from the DataSource, read as `TrainingData` — the port without its sealed door; the budget is `run.budget_min`): the plain loop — for each epoch: for each batch: forward, loss, backward, clip, optimizer step, schedule step; then metrics on train and valid, the early-stopping check, the checkpoint; **each epoch calls `callbacks.on_epoch(record)`** with the record below; `evaluate(model, spec, loader, loss_fn=None) -> {metric: value}` and `predict(model, spec, part) -> ScoredPart` (§1.2, §1.3; both made by `metrics.ScoringPass`, the one scoring pass the Evaluator's `scored` uses too); nothing else: the device, the loss and the per-epoch learning rates the notebooks and the smoke pass show are the Trainer module's own support functions, the ones the loop calls — `optim.device_for(spec)`, `losses.build_loss(spec)` (with `formula()`), `optim.resolved_lrs(spec, steps_per_epoch)` — reached by a stage through `PackageAdapters.device_for` / `loss_for` / `lrs_for`, so they never re-derive the loop's choices and the port stays the module's one door |
| `evaluate.py` | `class Evaluator`, made by `Evaluator.restoring(data, store, build)` (scores a model handed in or restores one) or `Evaluator.over(data)` (scores the models handed in, restores none): `scored(spec, dataset, model) -> ScoredPart` (score, truth, row ids, row-id kind, unscored rows; through `metrics.ScoringPass`), `score_part(spec, part, checkpoint=None, model=None)` for `train` and `valid`, and `score_sealed_test(spec, token, checkpoint=None, model=None, seal_path="test_seal.json")` — one scoring path for the network and the baselines |
| `metrics.py` | the backend's metric functions, one per metric name of the family catalog the package's task kind reads (`metric(name, y_true, y_score, options: MetricOptions = None) -> float`, `compute(names, y_true, y_score, options=None)`), each effective kind's reading of its score (`reading_for(kind)`, the numpy half of `TaskReadout.scores`) and the one scoring pass (`ScoringPass`), numpy/scikit-learn only; binary and multilabel scores are logits and `σ` is applied unconditionally, multiclass scores are logits and softmax is applied; copied into the run directory so the trainer, `/vega-eval`'s evaluator and the notebooks import the same file from where they run |
| `store.py` | `class ModelRegistry`: `save(model, path, spec, *, epoch=None) -> Checkpoint` (the whole state — parameters and registered buffers such as the one-class centre and a regression head's target affine — with the weights digest and the spec's digest written inside the file: a checkpoint without its spec is refused at save, and a file that records none is refused at load), `version(path) -> Checkpoint` (what a bundle stamps as `model_version` without building the network), `load(ref, spec, build, *, expected_digest=None, run_dir=None) -> TrainedNetwork` (`ref` a path or a `Checkpoint`; the digest is verified before loading; `build` is the Model port's builder, a parameter, so the store restores weights without knowing what a network is); exports go through the support file `model/export.py` (`ArtifactExporter`: `export(model, spec, fmt, path, batch=None)` and `parity(model, exported_path, batch, atol=1e-5)`) |
| `spec_snapshot.json` | the spec with the backend's resolved defaults filled in, the schedule's per-epoch learning rates listed, `determinism.tolerance`, and `read_paths` — the regularization paths the emitted code reads (netspec-contract §3, R2), with `_resolved.regularization` naming the live paths a declaration shadows and the value each trains with |
| `requirements.txt` | the backend's package(s) with minimum versions; nothing the code does not import |

One template generation ships, `v2.3` (`emit.py` `GENERATIONS`): it emits the classes above, and its `blueprint-map.json` records `generation: v2.3`. The v2.2 layout — module-level `build_model(spec)`, `load_part(spec, part)`, `fit(model, spec, data, callbacks, budget)`, `metric(name, y_true, y_score, **kw)` — was removed at the backend cutover: the emitter refuses `--generation v2.2` by name, and the stages that drive emitted code refuse a package whose map records another generation (`/vega-eval`'s `evaluate.py` before the seal is written, `nb_emit.py` before a cell is emitted) rather than call functions that are not there. The downstream skills reach the adapters only through `vega_core.PackageAdapters.load(package_dir, package)`, which imports the package from its own directory and resolves every adapter by its blueprint-map role (`data`, `trainer`, `evaluator`, `store`, `build`, `readout`, `metrics`, `baselines`), so no stage hard-codes a module path.

The epoch **record** (one JSON object per line of `history.jsonl`, written by `/vega-train`):

```json
{ "epoch": 12, "time_s": 0.84, "lr": 0.00071,
  "train": { "loss": 0.412, "pr_auc": 0.71, "roc_auc": 0.84 }, "valid": { "loss": 0.455, "pr_auc": 0.66, "roc_auc": 0.81 },
  "grad_norm": { "global": 1.31, "clipped": true, "per_block": { "B1": 0.12, "B3": 1.02, "B4": 0.41 } },
  "weight_norm": { "global": 41.2, "per_block": { "B3": 30.1 } }, "update_ratio": { "B3": 0.0021 },
  "nan": false, "early_stopping": { "best": 0.68, "since_best": 3, "stop": false } }
```

`update_ratio` is ‖Δθ‖ / ‖θ‖ per block for that epoch's last step — the number that says whether the learning rate is in range (∼1e-3 is healthy; > 1e-2 is thrashing, D5; < 1e-5 is not learning, D2). A window or graph run adds `sampling` (`{"windows", "unscored"}` or `{"targets", "nodes", "edges"}`) and its epoch source's `shuffle` description. The trainer also writes `first_step.json` once per run — ‖g‖ and ‖Δθ‖ of the head's weights at epoch 1, step 1, and their ratio — from a callback `fit` calls after the first optimizer step (`callbacks.on_first_step(record)`); it is the record the training layer's `mechanics.one_step` reads. These are the diagnostics `/vega-train`'s catalog reads; a backend that cannot compute a field writes `null`, and a diagnostic whose condition needs that field is recorded as `unavailable` rather than guessed.

**Determinism.** Given the spec's `seed`, the same backend version and the same device, two `fit` calls produce the same `history.jsonl` up to the tolerance the backend declares (`determinism.tolerance` in `spec_snapshot.json`: `0` on CPU for PyTorch with deterministic algorithms enabled, a stated relative tolerance on GPU). Every added source of randomness — the window anchor permutation, the neighbour sampler, edge dropout — is seeded from `seed` and the epoch. `/vega-train --check` replays the selected run's first epochs and compares.

**Checkpoints.** `save` / `load` in the backend's native format (`.pt` for PyTorch), plus `export` for the formats the backend supports (`onnx`, `torchscript`) with the parity check `parity(model, exported, batch, atol)`. The export's positional input order is fixed per sample shape: row `(x_num, x_cat?)`; window `(x_num, x_cat?)` with a step axis; graph node `(x_num, x_cat?, edge_index, edge_weight?, target_index)`; graph level `(x_num, x_cat?, edge_index, edge_weight?, graph_index, num_graphs)`.

### 1.1 The batch contract

What `data.py` yields and `Network.forward` reads is a mapping keyed by the `domain.py` enum `BatchKey` (`x_num`, `x_cat`, `y`, `edge_index`, `edge_weight`, `target_index`, `graph_index`, `num_graphs`; a `str` enum, so `batch["x_num"]` keeps working).

| shape | `x_num` | `x_cat` (optional) | graph keys | `y` (optional) |
|---|---|---|---|---|
| row | `(B, F)` float32 | `(B, C)` int64 | — | §1.2 |
| window | `(B, L, F′)` float32, `F′ = F + 1` when `autoregressive` | `(B, L, C)` int64 | — | §1.2 |
| graph, level node | `(N, F)` for every node of the batch graph | `(N, C)` | `edge_index (2, E)` int64 (row 0 the source, row 1 the destination), `edge_weight (E,)` float32 or absent, `target_index (B,)` int64 | `(B, …)`, aligned with `target_index` |
| graph, level graph | `(N, F)` | `(N, C)` | `edge_index`, `edge_weight?`, `graph_index (N,)` int64 in `[0, G)`, `num_graphs` (an int, `G`) | `(G, …)` |

`Network.forward(batch)` returns outputs **aligned with `y`**: at node level the network gathers `target_index` after the head, so losses, metrics and the trainer are shape-agnostic, and no reader gathers a second time.

### 1.2 The score contract

One object, `TaskReadout.for_spec(spec)`, turns the head's output into what every reader needs — the trainer's accumulator, the evaluator, the baselines' conversion and the serving Predictor. The effective kind is `task.base_kind` for the `sequence-*` kinds.

| effective kind | head out | `score` (what metrics read) | `proba` (`predict_proba`) | `y` |
|---|---|---|---|---|
| binary | 1 | `(B,)` logit, always | `σ(z)` | `(B,)` {0, 1} |
| multiclass | K | `(B, K)` logits | softmax | `(B,)` int |
| multilabel | K | `(B, K)` logits | `σ` per label | `(B, K)` {0, 1} |
| regression | 1, or Q under the `quantile` loss | `(B,)` value (the 0.5 column under the quantile loss) | = score | `(B,)` |
| ordinal | K−1 (rank-consistent) | `(B, K−1)` cumulative logits | class probabilities `(B, K)`: `p_0 = 1 − σ(z_0)`, `p_j = σ(z_{j−1}) − σ(z_j)`, `p_{K−1} = σ(z_{K−2})` | `(B,)` int in `[0, K)` |
| forecast | H, or H·Q (horizon-major) | `(B, H)` values (the median under the quantile loss) | = score | `(B, H)` |
| unsupervised | `recon_dim` | `(B, n_numeric)` numeric reconstruction; per-row score: numeric MSE + mean nominal cross-entropy | = score | none (the truth is `x_num`) |
| anomaly, reconstruction or vae loss | `recon_dim` | `(B,)` per-row anomaly score: numeric MSE + mean nominal cross-entropy | = score | `(B,)` {0, 1} or none |
| anomaly, svdd loss | z | `(B,)` `‖f(x) − c‖²`, with `c` the network's buffer | = score | as above |

`recon_dim = n_numeric + Σ cardinality` of the embedded nominals. The baselines emit the same shapes (logits for binary and multilabel, log-probabilities for multiclass, cumulative logits for ordinal). A decision threshold is in probability space for binary and multilabel and on the score scale for anomaly.

### 1.3 Row identity

| shape | `row_id` of a prediction | `vega.row_id_kind` |
|---|---|---|
| row | the shaped part's index (the raw-file row position) | `raw_position` |
| window | the raw position of the window's anchor (its last input row) | `raw_position` |
| graph, level node | the raw position of the target node's row | `raw_position` |
| graph, level graph | the `graph_id` value | `graph_id` |

A row that cannot be scored — a window without enough context under `context: none`, a forecast anchor whose horizon runs past its part — never gets a fabricated prediction: it is counted with its reason as an unscored row.

## 2. The PyTorch backend (`scripts/backends/pytorch/`)

- Each block type maps to one `nn.Module` construction in `blocks.py`, whose segments the emitter splices by `SEGMENT_FILE` into `<model>/blocks_core.py`, `blocks_seq.py`, `blocks_graph.py` and `blocks_gen.py` (each emitted only when the spec uses one of its blocks); composite blocks nest; `residual` inserts a projection `nn.Linear` when widths differ and `projection: auto`; `embedding` builds one `nn.Embedding` per nominal column with `cardinality` rows, where `cardinality = unseen_category_id + 1` from the shaper's encode step — the last row is the reserved unseen-category id that the shaper's executor and `/vega-pack`'s preprocessing both map unseen values to. A `custom` block's module is copied into `<model>/custom/<stem>.py`.
- The recurrent and graph blocks are pure PyTorch: `torch.nn.RNN` / `GRU` / `LSTM` behind one recurrent block with `return: last|all|mean|max`, and message passing over `index_add_` / `scatter_reduce` with no `torch_geometric` dependency. Every graph convolution takes the node features, `edge_index` and `edge_weight` positionally, and the edge weight multiplies its messages.
- `data.py` reads parquet/csv with pandas, casts per `data_contract.features[].dtype`, splits numeric and nominal tensors, and yields the batches of §1.1. `SealedPartError` is raised before any file is opened.
- `train_loop.py`: `torch.manual_seed(seed)`, `torch.use_deterministic_algorithms(True)` on CPU; `nn.utils.clip_grad_norm_` with the spec's value and the pre-clip norm recorded; per-block gradient norms from `named_parameters()` grouped by block id prefix; metrics through the emitted `metrics.py` (numpy/scikit-learn implementations — the same file `/vega-eval` and the notebooks import, so a valid metric in training and a test metric in evaluation are the same code; no second implementation, `torchmetrics` included); early stopping restores the best checkpoint; the budget stops the loop at the epoch boundary after `wall_clock_min` and records `stopped_by: budget`.
- Losses in `losses.py` (a support file of the training module): `bce_with_logits` (with `pos_weight`), `cross_entropy` (weights, label smoothing), `focal`, `mse`, `mae`, `huber`, `quantile`, `cumulative_link`, `reconstruction` (mse on numeric columns + cross-entropy on embedded nominals), `vae` (`reconstruction + β·kl`), `svdd` (the centre set once from the first batch into the network's buffers); each exposes `formula()` — the string the layer's mechanics panel shows.
- Optimizers and schedules are `torch.optim` / `torch.optim.lr_scheduler` with the spec's parameters; the resolved per-epoch learning rates are written to `spec_snapshot.json` so the layer can draw the schedule before any run.
- `export`: `torch.onnx.export` with dynamic batch, `torch.jit.trace`; `parity` compares outputs on a real batch at `atol 1e-5`.
- `coverage.json`: the block types the backend implements (`blocks`) and the ones it lists as not built yet (`planned`); every loss above, with the `loss.params` keys each takes (`loss_params`: the loss's own `PARAMS`, which `compile_spec.py` B2 holds a spec's `loss.params` to, so a key `build_loss` would refuse at the first step of a fit is refused at compile time — `test_compile_rules.py` holds the two equal); `torch.optim`'s `adamw`, `adam`, `sgd`, `radam`, `rmsprop`; the schedules `cosine`, `plateau`, `onecycle`, `none`; every metric of the catalog's task table and of `vega-eval/references/eval-contract.md` §3 (the evaluator's rows are the superset, and `metrics.py` implements all of them) plus `anomaly_score_mean`; the exports `onnx` and `torchscript`; `sample_shapes` (`row`, `window`, `graph`), `graph_batching` (`full`, `sampled`, `graphs`) and `recurrent_returns` (`last`, `all`, `mean`, `max`); and the `inspection` surface of §2.1.

### 2.1 The block inspection surface

The inspector (`inspect-contract.md`) reads each block through a fixed set of attributes and methods, and never through anything else. A backend that lists a block type under `blocks` lists its surface under `coverage.json` `inspection.surface`, and every block class of that type exposes it (`test_inspection_surface.py`). `inspection.blocks` names every catalog block type the backend's readers cover, which includes every type of `blocks`; `inspection.synthetic` names the five synthetic node types.

| type | attributes |
|---|---|
| `dense` | `linear`, `act` |
| `embedding` | `columns`, `embeddings`, `out_features` |
| `concat` | — |
| `batchnorm` | `bn` |
| `layernorm` | `ln` |
| `dropout` | `drop` |
| `activation` | `act` |
| `residual` | `repeats`, `projections` |
| `sequential` | `inner` |
| `parallel` | `branches`, `merge` |
| `conv1d` | `conv` |
| `pool1d` | `pool` |
| `rnn` | `rnn`, `ret` |
| `gru` | `rnn`, `ret` |
| `lstm` | `rnn`, `ret` |
| `attention` | `attn`, `pool` |
| `attention_pool` | `score`, `query`, `weights` |
| `transformer_encoder` | `enc` |
| `feature_tokenizer` | `weight`, `bias`, `cls` |
| `positional` | `kind`, `pe` |
| `flatten` | — |
| `reshape` | `shape` |
| `pool` | `kind` |
| `head` | `linear`, `act` |
| `decoder` | `net` |
| `reparameterize` | `z` |
| `gcn_conv` | `lin`, `bias`, `normalize`, `self_loops`, `coefficients` |
| `sage_conv` | `lin_root`, `lin_neigh`, `aggregator`, `normalize`, `coefficients` |
| `gat_conv` | `lin`, `att_src`, `att_dst`, `heads`, `concat`, `negative_slope`, `coefficients` |
| `gin_conv` | `mlp`, `eps`, `coefficients` |
| `graph_pool` | `readout` |
| `edge_dropout` | `p` |
| `custom` | — |

Two refinements, in `coverage.json` beside the table. **`surface_variants`**: on an ordinal task the head is the rank-consistent cumulative head, whose surface is `score` (the shared `nn.Linear(in, 1, bias=False)`), `first` and `gaps` (the thresholds are `first − cumsum([0, softplus(gaps)])`). **`surface_optional`**: a `feature_tokenizer` over a contract with nominal columns also exposes `embeddings`, one `nn.Embedding(cardinality, d)` per nominal in contract order.

The semantics behind the names: `rnn` is the `nn.RNN`/`GRU`/`LSTM` module and `ret` its readout (`last`, `all`, `mean`, `max`); `AttentionPool.weights(x) -> (B, L)` returns `α = softmax_t(q · tanh(W h_t + b))` computed from `score` (`W`, `b`) and `query` (`q`); the graph convolutions expose a pure `coefficients(...)` — `(edge_index, n, edge_weight=None) -> (src, dst, coef)` for GCN, SAGE and GIN, `(h, edge_index, edge_weight=None) -> (src, dst, alpha[E, heads])` for GAT — that reproduces the convolution's output at a target node, and `edge_weight` multiplies the messages in all four (for GAT it multiplies `α_ij` after the softmax, without renormalising). A regression-family head (regression, sequence-regression, forecast, quantile) also carries the buffers `target_shift` and `target_scale`, the fixed train-fitted affine it ends on (`y = act(z) · target_scale + target_shift`, per output, horizon step or quantile channel; measured on the train part only, saved in the checkpoint, recorded in `smoke.json` `init_notes.target_affine`), and the inspector reads them as the head's own tensors `<head id>.target_shift` and `<head id>.target_scale`. No block writes an attribute during an evaluation forward: there is no `last_*` state, and `Reparameterize.kl` is assigned only in training.

### 2.2 The explain catalog

A backend ships **`backends/<name>/explain-catalog.json`**, schema `vega/explain-catalog@1`: the backend-owned text the showcase page's stepper and code walk read for this backend's blocks. It covers every block type of the backend's `coverage.json` `blocks`, every activation of `coverage.json` `activations` and every link of `catalog.json` `inspection.links`, and carries the fixed walk entries:

```json
{ "schema": "vega/explain-catalog@1", "backend": "pytorch", "language": "python",
  "blocks": { "<type>": { "stage_kind": "…", "label": "…", "label_many": "…", "step": "…", "formula": "…", "prose": "…",
                          "unit_how": "…", "recompute": "…", "code": { "file": "<regex>", "at": ["<regex>"], "after": null, "to": null },
                          "walk": { "title": "…", "text": "…" } } },
  "activations": { "<name>": { "name": "…", "curve": "<name>", "silences_negative": true, "bend": "…" } },
  "links": { "<link>": { "formula": "…", "prose": "…", "answer": "…" } },
  "walk_fixed": [ { "id": "root | store | build | forward | link | shape | run | explain | route", "code": { }, "title": "…", "text": "…" } ] }
```

Its prose is English with no digit outside a placeholder (the placeholders are stage parameters — `{units}`, `{inputs}`, `{kernel}`, `{channels}`, `{hidden}`, `{heads}`, `{layers}`, `{z}`, `{n_sub}`, `{slope}`, `{eps}`, `{p}` — lexicon keys such as `{row.one}`, and the subscript helpers `{v}`, `{n_sub}`). Its `code` anchors are regular expressions that resolve against a rendered package: a block by its segment file (`blocks_core.py`, `blocks_seq.py`, `blocks_graph.py`, `blocks_gen.py`), `build` at `network.py` `^build_model = `, `store` at `store.py` `^\s+def load\(`, `route` at `serve/routes_predict.py` `^def register\(`. `tests/test_explain_catalog_sync.py` fails when an entry is missing, an anchor does not compile, or a prose value breaks the digit rule.

## 3. Adding a backend

Write `backends/<name>/emit.py` exposing `emit(spec, pipeline, out_dir)` that produces the package of `code-contract.md` §2 with the interface of §1 (the adapters plus ports, domain, the support files and the map), from templates that pass the conformance gate, register it in `backends/__init__.py`, and declare its coverage in `backends/<name>/coverage.json` — the block types, losses, optimizers, schedules, metrics, export formats, sample shapes, graph batching modes and recurrent readouts it implements, and the inspection surface of every block it lists (§2.1) (a backend may cover a subset; `compile_spec.py --backend <name>` refuses a spec that uses anything the backend lacks and names it, and a block the backend lists under `planned` is refused as planned; the shipped backend's `planned` is empty — `scripts/tests/test_no_planned.py` holds it so — and a spec naming what it does not implement is refused as not covered). Ship its `explain-catalog.json` (§2.2). A backend in another language keeps the same files as *thin Python shims* that call the foreign runtime and still yield the epoch record — the sub-chain never reads a framework's own log format.

**The coverage matrix is the evidence of coverage.** A backend covers what the coverage matrix proves it covers, never what its `coverage.json` claims: `scripts/tests/matrix/run_matrix.py --cells all` runs every cell of `catalog.json`'s admission table (`family-catalog.md` §7) — fixtures shaped by the real `shape.py`, then compile, emit with the smoke pass, the conformance gate C0–C14, a seeded fit through `PackageAdapters`, the sealed test score through `test_seal.json` and the stored predictions in `vega/predictions-layout@1` — and writes `matrix-report.json` (`vega/backend-matrix@1`). The backend ships only when that report is **green** (`summary.failed == []`: every cell passed every stage) and its cell list **equals** the admission table (`admission.table: "equal"`; `make_matrix_fixtures.py` refuses a recipe table that drifted from the catalog, naming the difference). A cell the table admits and the matrix does not run, or a cell the matrix runs that the table does not admit, is a coverage defect, not a skip; `--cells smoke` (one cell per family, 21) is the unit tier's quick subset and never stands in for `all`.
