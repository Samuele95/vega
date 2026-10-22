# Vega family catalog — tasks, families, blocks, defaults

The catalog is the vocabulary of `netspec.json`: which task kinds exist and what head, loss and metrics they take; which network families the skill knows, when each applies and what it is made of; every block type of the grammar; and the defaults a `vega:default` source may cite. A family is a **named composition of blocks**, not a closed enumeration — a spec may declare a family this file does not name as long as its blocks are known, and the layer then shows the family as *user-defined*. What the catalog closes is the *reason* behind a default: a `vega:default` must point at a row here through its `catalog` key (`families/<name>`, `blocks/<type>`, `defaults/<decision>`), and `scripts/catalog.json` is the machine copy of §1–§4 (the task table, the families, the blocks, the defaults) the compiler resolves those keys against, plus §5's fork points, §6's metric directions and §7's admission table — the build's tests fail when the two diverge. Which block types a backend compiles is not this file's business: each backend declares it in its own `coverage.json` (`backend-contract.md` §3).

## 1. Task kinds

The task comes from `geometry:partitions/<label>` (`task` on the candidate, as prose) and from what the shaper materialised; the mapping is fixed: `"binary classification"` → `binary`, `"multiclass classification"` → `multiclass`, `"multilabel classification"` → `multilabel`, `"regression"` → `regression`, `"ordinal regression"` → `ordinal`, no partition or `"none"` → `unsupervised` (or `anomaly` under `--goal anomaly`), and any of the first five with `analysis:context.time` plus a `time` split and declared windows → the `sequence-*` / `forecast` row. `data_contract.sample.shape` is `row` unless a sequence family windows the rows (`window`) or a graph family reads an `edges` part (`graph`). The head rule of each row below is what the compiler enforces (head `out` and activation against `task.kind`, the loss against both).

| `task.kind` | when | head (`out`, activation) | default loss | primary metric (selection) | secondary metrics |
|---|---|---|---|---|---|
| `binary` | one label with two classes | `1`, `none` | `bce_with_logits` (+ `pos_weight` when the minority share < 0.3 — `analysis:F<n>` on imbalance) | `pr_auc` when imbalanced, else `roc_auc` | `roc_auc`, `f1`, `balanced_accuracy`, `mcc`, `log_loss`, `brier`, `ece` |
| `multiclass` | one label, k > 2 classes | `k`, `none` | `cross_entropy` (+ class weights `loss.params.weight` or label smoothing 0.05 when imbalanced) | `f1_macro` | `accuracy`, `balanced_accuracy`, `log_loss`, `top_k`, `f1_per_class` |
| `multilabel` | several binary labels, carried as **one** shaped column of delimited class names (`"a\|c"`; the empty string is the empty set) | `k`, `none` | `bce_with_logits` per label | `f1_micro` | `f1_macro`, `pr_auc` per label (the breakdown) and `pr_auc_macro`, `hamming`, `subset_accuracy` |
| `regression` | a continuous label | `1`, `none` (`Q`, `none` under the `quantile` loss) | `huber` when the analysis found outliers in the label, else `mse`; `mae`, and `quantile` (Q quantiles) when the analysis or a `user:` turn asks for intervals | `rmse` | `mae`, `r2`, `mape` (over the rows whose label is not zero), `residual_quantiles` |
| `ordinal` | ordered classes | `k−1`, `none` — cumulative logits from a **rank-consistent** head: one shared score and `k−1` ordered thresholds, so `P(y > j) = σ(z_j)` falls with `j` and every class probability is non-negative | `cumulative_link` | `mae_ordinal` | `accuracy`, `qwk` |
| `unsupervised` | no partition chosen, or `--goal reconstruct` | reconstruction (`recon_dim`, `none`), where `recon_dim = n_numeric + Σ cardinality` of the embedded nominals (a one-hot nominal is numeric columns) | `reconstruction` (the backend's mixed loss: `mse` on numeric columns + `cross_entropy` on each embedded nominal's logits) | `reconstruction_error` on valid | per-feature error |
| `anomaly` | a rare label the user declares an anomaly, or no label with `--goal anomaly` | reconstruction (`recon_dim`), or one-class score (`z`) | `reconstruction` / `vae` / `svdd` | `pr_auc` on the score when a label exists, else `anomaly_score_mean` (the mean per-row anomaly score) | `precision_at_k`, `roc_auc` |
| `sequence-classification` / `sequence-regression` / `forecast` | `analysis:context.time` exists, the shaper's split is `time`, windows are declared | as the base task (`task.base_kind`); `forecast`: `horizon` (× the quantiles under the `quantile` loss) | as the base task; `forecast` uses `mse`/`mae`/`quantile` | as the base task; `forecast`: `mase` | `smape`, per-horizon error |

`metrics.selection.monitor` is always `valid.<primary>`; `early_stopping.monitor` defaults to the same (a loss-monitored early stop is an alternative to offer, not the default, because the loss and the metric can disagree under class weighting).

**Task fields** (`netspec-contract.md` §1, machine copy `task_fields`). `task.classes` is required for `binary`, `multiclass`, `multilabel` (the label names) and `ordinal` (in order). `task.base_kind` is required for the `sequence-*` kinds and names the effective kind every reader of the base uses — the head, the loss, the metrics, the readout and the serving reading; `sequence_map` says which bases each kind admits, and `forecast`'s base is `regression`. `task.label_format` (`index` | `names` | `delimited`, with `sep`) is required for `multilabel`, whose label is one column of delimited names; elsewhere it defaults to `names` when the label column holds strings, else `index`. `task.positive` names the class that means positive for `binary` and anomalous for an `anomaly` task with a label: for `binary` it is the second class, `classes[1]` (T1e refuses a spec that names the first), and for a labelled `anomaly` task it defaults to the second class. `task.anomaly_rate` (an `anomaly` task only, optional, with its `anomaly_rate_source`) is the expected share of anomalous rows — the rate the evaluation's threshold quantile and `precision_at_k` read when the task has no label (netspec-contract §1).

**Graph-level tasks.** A `graph` sample (`netspec-contract.md` §1) adds no task row: at `level: node` the rows above apply unchanged — every node is a row with the partition's label. At `level: graph`, `binary` | `multiclass` | `regression` keep their head, loss and metrics but require `sample_shape: graph` with a `graph_id` (rows grouped into graphs, one label per graph) and a `graph_pool` block before the head — the compiler's W2 (`netspec-contract.md` §3). `data_contract.sample.shape` is `row`, `window` (a sequence family) or `graph` (a graph family on the shaper's `edges` part); nothing else.

## 2. Families

| family | when the chain's state points to it | composition (blocks) | defaults it brings | alternatives to name |
|---|---|---|---|---|
| `mlp-plain` | tabular rows, `input_dim` ≤ ~200, `analysis:importance` learnable, no time context | `[embedding?] → concat → (dense → activation → dropout) × depth → head` | width and depth per `defaults/sizing`, `gelu`, dropout 0.1 | `mlp-residual` when depth ≥ 3; a linear head alone as the sanity ablation |
| `mlp-residual` | as above with depth ≥ 3, or `blueprint:posture` says *neural* on a non-linear importance | `[embedding?] → concat → dense(width) → residual(inner: dense → batchnorm\|layernorm → activation → dropout) × depth → head` | width and depth per `defaults/sizing`, `batchnorm` when batch ≥ 32 else `layernorm`, activation `gelu`, dropout 0.2 | `mlp-plain`; `ft-transformer` when nominals dominate |
| `ft-transformer` | wide tabular (`input_dim` > ~50) or many high-cardinality nominals; the user accepts the cost — a `user:` turn, recorded as the family's source (unattended, never the default) | `feature_tokenizer(d) → transformer_encoder(d, L) → pool(cls) → head` (the tokenizer prepends the `[cls]` token and reads the nominal columns itself, one token per nominal; an `embedding` before it is refused, G2) | `d`, `L` per `defaults/sizing`, heads 4, `ff` 2·`d`, dropout 0.1, `adamw` lr 1e-4 | `mlp-residual` (10× cheaper) |
| `cnn1d` | `sample.shape = window`, local temporal patterns (short windows, regular sampling per `analysis:context.time`) | `[embedding → concat]? → (conv1d → activation → pool1d) × depth → flatten → dense → head` | kernel 5, channels per `defaults/sizing`, `gelu` | `gru` for longer dependencies |
| `gru` / `lstm` | windows with long dependencies, irregular patterns, `forecast` | `[embedding → concat]? → gru\|lstm(hidden, num_layers, return: last) → head` (`return` per `defaults/recurrent_return`; its alternatives read every step — `return: all → pool(mean)` pools them, `return: all → attention(pool)` self-attends over them, then pools) | `hidden` per `defaults/sizing`, `num_layers` 1, dropout 0.1 between layers | `cnn1d`; `transformer-seq` |
| `transformer-seq` | long windows, many series, budget allows | `[embedding → concat]? → dense(d) → positional → transformer_encoder(d, L) → pool(mean) → head` | `d`, `L` per `defaults/sizing`, heads 4, `ff` 2·`d`, dropout 0.1 | `gru` |
| `autoencoder` | `unsupervised` / `anomaly`; reconstruction as the objective | `[embedding?] → concat → sequential(dense × depth) → dense(z) → decoder(widths reversed) → head(recon_dim, none)` — the `dense(z)` is the bottleneck; each embedded nominal is reconstructed as logits over its categories | z = `dims_95` from `geometry:space`, symmetric widths | `vae` (`reparameterize`) when a generative reading is wanted; `deep-svdd` for one-class |
| `vae` | as `autoencoder` with a latent prior | `[embedding?] → concat → sequential(dense × depth) → dense(2z) → reparameterize(z) → decoder → head(recon_dim, none)`; loss `vae` (`reconstruction + β·kl`) | β 1, z as above | `autoencoder` |
| `deep-svdd` | one-class anomaly with a hypersphere objective | `[embedding?] → concat → sequential(dense × depth) → head(z, none, bias: false)`; loss `svdd` (the head's output is the score's embedding; the centre is a model buffer, set once from the first batch and saved with the weights) | z 16, head `bias: false` (a bias lets the hypersphere collapse) | `autoencoder` |
| `linear` | the sanity floor every supervised campaign includes as an ablation, on every sample shape (`floor_shapes: [row, window, graph]`) | row: `[embedding?] → concat → head`; window: `[embedding?] → concat → flatten → head`; graph at `level: node`: `[embedding?] → concat → head` (structure-blind: each node's own features); graph at `level: graph`: `[embedding?] → concat → graph_pool(mean) → head` | — | — |
| `pca-floor` | the floor for `unsupervised` / `anomaly`: a linear autoencoder (one `dense(z)` encoder, one reconstruction head, no activation) — PCA by another name | `[embedding?] → concat → dense(z, none) → head(recon_dim, none)` | z as the family's | — |
| `rnn` | a sequence task (`sample.shape = window`); the ablation floor for the recurrent families — what `gru`, `lstm`, `bi-*` and `*-attention` must beat (the ladder's `R1b` rung) | `[embedding → concat]? → rnn(hidden, num_layers, return: last) → dropout → head` | `hidden` per `defaults/sizing`, 1 layer, `tanh`, `return: last` per `defaults/recurrent_return` | `gru`, `lstm` |
| `bi-lstm` / `bi-gru` | classification of *complete* windows where the whole window — its future included — is legitimately visible at inference. Never `forecast`, never `serving.mode: stream`: a bidirectional block on either is `ERROR:` (compiler rule **W3**, `netspec-contract.md` §3) | as `lstm` / `gru` with `bidirectional: true` — `[embedding → concat]? → lstm\|gru(hidden, num_layers, bidirectional: true, return: last) → dropout → head`; the readout is `2·hidden` wide | as the base family; `bidirectional: true` per `defaults/bidirectional` | the unidirectional family (`lstm` / `gru`) |
| `lstm-attention` / `gru-attention` | long windows where the informative step is not the last one (an event somewhere inside the window decides the label) | `[embedding → concat]? → lstm\|gru(hidden, num_layers, return: all) → attention_pool(hidden) → dropout → head` | as the base family; `return: all → attention_pool` per `defaults/recurrent_return` | the `return: last` family (`lstm` / `gru`) |
| `gcn` | an `edges` part exists (`manifest.outputs.files.edges`, from the shaper's `graph_edges` step) and the analysis found the rows related (a `relations` / `segments` / `spatial` finding) — the family `blueprint:posture` names, or the `vega:default` graph family when the posture says *graph* | `[edge_dropout]? → [embedding → concat]? → [gcn_conv → activation → dropout] × depth → head` at `level: node`; `… → graph_pool(mean) → head` at `level: graph` | depth 2 per `defaults/graph_depth`, `hidden` per `defaults/sizing` (the RNN rule, `width/2`), activation `relu`, dropout 0.5 (Kipf & Welling), `normalize: sym`, `self_loops: true`, protocol per `defaults/graph_protocol` | `graphsage` (inductive, neighbour sampling — when the graph is large); `gat` |
| `graphsage` | the `inductive` protocol on a large graph (neighbour sampling instead of the full adjacency), or heterophilous neighbourhoods where the root's own features must survive the aggregation | `[edge_dropout]? → [embedding → concat]? → [sage_conv → activation → dropout] × depth → head` (`+ graph_pool(mean)` before the head at `level: graph`) | depth 2, activation `relu`, `aggregator: mean`, `root_weight: true`, `neighbours: [25, 10]` when sampling | `gcn`; `gin` |
| `gat` | neighbour importance varies — some edges matter, most do not | `[edge_dropout]? → [embedding → concat]? → [gat_conv(heads: 8, concat: true) → activation → dropout] × (depth − 1) → gat_conv(heads: 1, concat: false) → head` (`+ graph_pool(mean)` before the head at `level: graph`) | depth 2, activation `leaky_relu` (as the GAT paper's), heads 8 then 1, `attn_dropout` 0.6, `negative_slope` 0.2 | `gcn` |
| `gin` | `level: graph` — the most discriminative aggregation for graph-level tasks (injective sum) | `[edge_dropout]? → [embedding → concat]? → [gin_conv → activation → dropout] × depth → graph_pool(sum) → head` | depth 3, activation `relu`, `eps` 0, `train_eps: false`, `mlp_layers` 2, `readout: sum` | `gcn + graph_pool` |

The optional parts of a composition read as follows. `[embedding?] → concat`: the `concat` is always there — it joins the numeric features with any embedding output — and the `embedding` appears when the data contract has a nominal column the shaper did not one-hot encode. `[embedding → concat]?`: both appear when the data contract has such a nominal column, and neither otherwise; on a window a nominal is `(B, L, C)` ids and embeds to `(B, L, Σdim)`, on a graph it is embedded per node. `[edge_dropout]?`: an optional first block of a graph family, whose kept edges replace the graph for every later block of the same forward pass (training only).

The recurrent variants are the base families with one parameter changed: `bi-*` is `bidirectional: true`, `*-attention` is `return: all` followed by an `attention_pool` readout. `return` names the readout of every recurrent block (`rnn`, `gru`, `lstm`): `last` (the last step's hidden state; for a bidirectional block the forward direction's last step joined with the backward direction's first), `all` (every step, `(N, L, H·dirs)` — for a pooling block to follow), `mean` or `max` (pooled over the steps inside the block); it replaces any implicit "last step" behaviour, and a spec that omits it means `last`. Plain `rnn` is the ablation floor and not a default: its gradients vanish over a long `L`, which is why `gru` / `lstm` are the default recurrent blocks.

The graph families (`gcn`, `graphsage`, `gat`, `gin`) carry `sample_shape: graph` and stand or fall on three things the compiler checks as **W2** (`netspec-contract.md` §3): an `edges` entry in the manifest's `outputs.files`, a declared `protocol` (`inductive` by default — `defaults/graph_protocol`; `transductive` only with a `user:` source, shown in the layer as a disclosed exposure of the test nodes' features), and a `graph_pool` block exactly when `level` is `graph`. A non-graph family on a `graph` shape is a `WARN:` — the edges are read by nobody, and the marker says so — except `linear`, whose `floor_shapes` name `graph`: the structure-blind floor is the honest thing a graph family must beat.

**Floors.** `linear` is the supervised floor on every shape (`floor_shapes`): on a window it inserts `flatten` before the head, at graph level `graph_pool(mean)`, and at node level it reads each node's own features only. `pca-floor` is the reconstruction floor on rows. `rnn` is the recurrent floor (`ablation_floor_for`: `gru`, `lstm`, `bi-lstm`, `bi-gru`, `lstm-attention`, `gru-attention`): the ladder's `R1b` rung swaps the recurrent block to `rnn` with the same `hidden` and `num_layers`, `return: last`, `bidirectional: false` and no `attention_pool`.

The blueprint's `serving.mode` is a force on this table: `online` with a stated latency makes `ft-transformer` and `transformer-seq` a `FORK:` that quotes the latency (source `blueprint:serving`): the blueprint settled the latency, not the family, so the heavy family is offered against its lighter alternative — unattended, the lighter one (`mlp-residual` for `ft-transformer`, `gru` for `transformer-seq`), marked `default applied (unattended)`, and `stream` requires a windowed sample, so only the sequence families apply. The blueprint's posture (`blueprint:posture`) names a family or a family *class* (linear / tree / neural / none yet). When it names *tree* or *none yet*, `/vega-forge` still runs — the user asked for a network — but the spec's `family.source` is `user:` (the user overruled the posture) or `vega:default` with the posture's forces quoted in the `rationale`, and the summary says the honest baselines in `/vega-eval` will be the test of that overrule.

## 3. Blocks

| type | params | shape rule | notes |
|---|---|---|---|
| `dense` | `units`, `activation` (`none\|relu\|gelu\|silu\|tanh\|sigmoid\|leaky_relu`), `bias` | `(…, in) → (…, units)` | activation may be a separate `activation` block instead |
| `embedding` | `columns[]`, `dim` (or `dims{}` per column) | nominal ids → `(…, Σdim)` | cardinality from `data_contract.features[].cardinality`; window ids `(B, L, C)` embed to `(B, L, Σdim)` |
| `concat` | — | joins the numeric features and every embedding output | required after `embedding` |
| `batchnorm` / `layernorm` | `eps`, `momentum` | shape-preserving | `batchnorm` needs batch ≥ 8; the compiler warns below 32 |
| `dropout` | `p` | shape-preserving | inactive at eval; the layer shows where it sits |
| `activation` | `name` | shape-preserving | |
| `residual` | `repeat`, `inner[]`, `projection` (`auto\|none`) | `x + f(x)`; `projection: auto` inserts a dense when widths differ | the compiler checks `inner` keeps the width or the projection is present |
| `sequential` / `parallel` | `inner[]` (`parallel`: + `merge: concat\|sum`) | `sequential`: the chain of its inner blocks' rules; `parallel`: every branch sees the input, `concat` sums the widths, `sum` requires equal widths | |
| `conv1d` / `pool1d` | `channels`, `kernel`, `stride`, `padding` / `kind: max\|avg`, `size` | `(B, C, L) → (B, C', L')` | windows only |
| `gru` / `lstm` | `hidden`, `num_layers` (default 1; `layers` is the same param under its old name, kept for existing specs), `bidirectional` (default `false`), `dropout` (between layers), `return: last\|all\|mean\|max` (default `last`) | `(B, L, F) → (B, hidden·dirs)` for `last` / `mean` / `max`, `(B, L, hidden·dirs)` for `all` | `return` replaces any implicit "last step" behaviour — a spec without it means `last`; `mean` and `max` are pooled over the steps inside the block; `bidirectional: true` is subject to W3 (never `forecast`, never `stream`) |
| `rnn` | `hidden`, `num_layers` (default 1), `nonlinearity: tanh\|relu`, `bidirectional` (default `false`), `dropout` (between layers), `return: last\|all\|mean\|max` (default `last`) | `(N, L, F) → (N, H·dirs)` for `last` / `mean` / `max`, `(N, L, H·dirs)` for `all` | plain Elman RNN; gradients vanish on long `L`, which is why `gru` / `lstm` are the default recurrent blocks and `rnn` is the ablation floor; `mean` and `max` are pooled inside the block |
| `attention` | `heads`, `dim`, `pool` (`true\|false`) | `(B, L, d) → (B, d)` when `pool: true`, else `(B, L, d)` | multi-head *self*-attention over the steps (each step attends to every other; `pool: true` mean-pools the result) — not a readout |
| `attention_pool` | `hidden` | `(N, L, H) → (N, H)` | one learned query attends over the `L` steps — a pooling *readout*, the one the `*-attention` families use after `return: all`: `α = softmax_t(q · tanh(W h_t + b))`, `y = Σ_t α_t h_t`; where `attention` mixes the steps among themselves, `attention_pool` only weighs them |
| `transformer_encoder` | `d`, `heads`, `layers`, `ff`, `dropout` | `(B, L, d)` | pre-norm layers |
| `feature_tokenizer` | `d` | rows → `(B, n_numeric + n_nominal + 1, d)` with a `[cls]` token | FT-Transformer entry: one token per numeric column (`x_j·w_j + b_j`) and one per nominal column from its own table, so the tokenizer reads the nominals itself and no `embedding` precedes it (G2) |
| `positional` | `kind: sinusoidal\|learned` | shape-preserving | |
| `flatten` | — | `(B, …) → (B, ∏…)` | |
| `reshape` | `shape[]` | `(B, n) → (B, *shape)` with ∏shape = n | |
| `pool` | `kind: mean\|max\|cls` | `(B, L, d) → (B, d)` | sequence outputs |
| `head` | `out`, `activation`, `bias` (default `true`) | `(…, in) → (…, out)` | exactly one per spec unless `multilabel` heads are declared; `bias: false` in `deep-svdd`; on an `ordinal` task the backend builds the rank-consistent cumulative head (one shared score, `out = k−1` ordered thresholds) |
| `decoder` | `widths[]` (default: the encoder's dense widths reversed), `out` (default `input_dim`), `activation` | `(B, z) → (B, out)` | autoencoders; the reconstruction head (`recon_dim`) follows it |
| `reparameterize` | `z` | `(B, 2z) → (B, z)` with the KL term | vae; the KL term is computed in training only |
| `custom` | `code_ref`, `out_shape` | the declared `out_shape`, checked by the smoke pass | `user:` source only; *unverified* in the layer; the emitter copies `code_ref` into `<model>/custom/<stem>.py`, where the builder resolves it |
| `concat` (structural) | — | forced after any `embedding`; `alternatives: []` with the rationale "structural" | catalog key `blocks/concat` |
| `gcn_conv` | `out`, `normalize: sym\|rw`, `self_loops` (default `true`), `bias` | `(N, F) + (E) → (N, out)` | `D^-½ (A+I) D^-½ X W` (`sym`) or `D^-1 (A+I) X W` (`rw`); an edge weight (the edges part's `weight` column) multiplies the messages and enters the degree; `graph` shape only |
| `sage_conv` | `out`, `aggregator: mean\|max\|sum`, `root_weight` (default `true`), `normalize` (default `false`) | `(N, F) + (E) → (N, out)` | `W₁ x_i + W₂ · agg_{j∈N(i)} x_j`, the root kept apart from its neighbours; an edge weight multiplies the messages; the block neighbour sampling applies to (`neighbours[k]` for the k-th conv) |
| `gat_conv` | `out`, `heads`, `concat` (default `true`), `attn_dropout`, `negative_slope` (default 0.2) | `(N, F) + (E) → (N, out·heads)` when `concat`, else `(N, out)` (heads averaged) | attention coefficients per edge (LeakyReLU, softmax over `N(i)`); an edge weight multiplies `α_ij` after the softmax, without renormalising; `attn_dropout` drops coefficients, not features |
| `gin_conv` | `out`, `eps` (default 0), `train_eps` (default `false`), `mlp_layers` (default 2) | `(N, F) + (E) → (N, out)` | `MLP((1+ε) x_i + Σ_{j∈N(i)} x_j)` — the injective sum aggregation; an edge weight multiplies the messages |
| `graph_pool` | `readout: mean\|max\|sum` | `(N, F) + graph_index → (G, F)` | groups nodes by `sample.graph_id` into `G` graphs; allowed only at `level: graph`, and required there (W2) |
| `edge_dropout` | `p` | `(E) → (E')` — node features untouched | drops edges, not features; training only (identity at eval); the kept edges replace the graph for every later block of the same forward pass |

The graph blocks take the node features plus the edge index — `(N, F) + (E)` reads "N nodes of F features and E edges from the `edges` part" — and only a `graph` sample supplies the edges. Which of these types a backend compiles is that backend's `coverage.json` (`backend-contract.md` §3): a type the backend lists under `planned` is refused by the compiler's B2 as *planned in that backend*, with the row named, and a type absent from both lists as *not covered*. The pytorch backend implements the graph and recurrent blocks in pure PyTorch — a normalised adjacency through `index_add_` / `scatter_reduce`, `torch.nn.RNN` for `rnn` — with no `torch_geometric` dependency; PyG is an optional accelerated path the coverage file may name.

## 4. Defaults a `vega:default` may cite

| decision | default | because | first alternative |
|---|---|---|---|
| sizing (`defaults/sizing`) | width `clip(2·input_dim, 32, 256)` rounded to a power of two, raised one step when the analysis found non-linear importance; depth 2 below 1 000 training rows, 3 below 100 000, 4 above; transformers `d = clip(width/2, 32, 128)`, `L = depth − 1`; RNN `hidden = width/2`; CNN channels `width/4 → width/2`; bottleneck `z = dims_95` from `geometry:space` | parameters scale with the rows the chain has and the input width, not with fashion | ×2 / ÷2 width and ±1 depth as `capacity` rungs |
| optimizer | `adamw` lr 1e-3, wd 1e-4, betas (0.9, 0.999) — tabular; lr 1e-4 for transformers; `sgd` momentum 0.9 lr 1e-2 only as an ablation | robust across widths without tuning; decoupled decay | `sgd` for generalisation ablations; any optimizer the backend's `coverage.json` lists by name |
| schedule | `cosine` with `warmup_epochs = 3` when `max_epochs` ≥ 30; `plateau` (factor 0.5, patience 5) otherwise | warmup steadies AdamW's first steps; cosine needs a known horizon | `onecycle` when a max lr was found by a range test; `none` as an ablation |
| batch size | `clip(rows_train / 8, 16, 512)` rounded to a power of two | ~8 optimizer steps per epoch at minimum, so an epoch means something | ×2 / ÷2 ablations; the smoke panel shows memory per batch |
| max epochs | 200 with early stopping; `budget.wall_clock_min` from the user or 20 | early stopping ends it; the budget bounds a run that never converges | a longer cap or budget when D9 cut a run before it converged — a `defaults_to_test` entry like any non-structural row (`training/max_epochs`); the patience is the usual rung (D13) |
| early stopping | monitor `valid.<primary>`, patience 10 (`max_epochs / 20`, ≥ 5), `min_delta` 1e-3, restore best | the primary metric is what selection uses | patience 20 on noisy valid metrics; monitor `valid.loss` |
| initialisation | `kaiming_uniform` for relu-family, `xavier_uniform` for tanh/sigmoid, zeros bias; embeddings N(0, 0.1) | keeps activations' variance at depth | orthogonal for RNNs |
| dropout | 0.1 (`mlp-plain`), 0.2 (`mlp-residual`), 0.1 (transformers), 0.5 (the graph families, Kipf & Welling), 0.1 between the layers of a stacked recurrent block, 0.1 for the `dropout` after a recurrent readout (`rnn`, `bi-*`, `*-attention`); 0 in the linear ablation. An `edge_dropout` block is optional and never a default: its `p` has no default row, so the block is composed only from an `analysis:` finding or a `user:` turn that also gives its `p` | the widths above vs the training rows the chain has | 0.3–0.5 when `train ≫ valid` (an `analysis:F<n>` on size, or a training diagnostic) |
| normalisation | `batchnorm` in MLP blocks when batch ≥ 32, else `layernorm`; `layernorm` in transformers and RNN outputs | batch statistics are unreliable on small batches | none as an ablation |
| weight decay | 1e-4 (AdamW) | mild shrinkage; the analysis's overfitting finding raises it | 1e-3, 0 |
| gradient clipping | global norm 1.0 | a cheap guard against the NaN epoch; the pre-clip norm is recorded in the epoch record's `grad_norm` | none |
| label smoothing | 0.05 for imbalanced multiclass | a softened target keeps a minority class's logits from being pushed to the majority's | 0 |
| precision | `fp32`; `bf16` autocast only with `--device cuda` and the user's word | numerics are inspectable in fp32 | `bf16` |
| seed | 7 (the chain's), and the data loader order seeded too | determinism across runs | — |
| embedding dim (`defaults/embedding_dim`) | the even number nearest to `min(50, 1.6 · cardinality^0.56)` — `2 · round(x / 2)`, rounding half to even, at least 2 (cardinality 42: 12.98 → 12) | the usual rule of thumb: sub-linear in the cardinality, capped so a rare nominal does not dominate the input width | ÷2 when train ≫ valid; a one-hot `encode` in the shaper when cardinality ≤ 8 |
| class imbalance | `pos_weight` / class weights (`loss.params.weight` of `cross_entropy`, `alpha` of `focal`) = inverse frequency clipped to 10 | keeps the gradient of the minority alive without exploding it | `focal` loss; resampling is the shaper's business, not the trainer's |
| device / workers (`defaults/device`) | `device: auto` (cuda → mps → cpu), `num_workers: 0`, `precision: fp32` | a run must be reproducible where it is read, and a worker pool changes batch order unless seeded | `--device`, `num_workers` > 0 for large parquet parts |
| graph protocol (`defaults/graph_protocol`) | `inductive`: the reference graph is the train nodes and the edges among them; a valid node is a query that receives messages from reference nodes only (never from another query — netspec-contract §1, *Protocol*), and the test nodes and every edge touching one enter through the seal at `/vega-eval`, as queries the same way | keeps the test seal's meaning — no test row is seen, as a feature or as a neighbour, before `/vega-eval` | `transductive` (`user:` source only, `exposure.kind: features` in the layer) when the deployment graph is fixed and the unlabeled nodes are legitimately present at training time |
| graph depth (`defaults/graph_depth`) | 2 conv blocks | over-smoothing beyond 2–3 hops on homophilous graphs — every node's representation converges to the same one | 3 with residual connections (`residual` around each conv) |
| recurrent return (`defaults/recurrent_return`) | `last` for classification and regression; `all → attention_pool` when the windows are long (the `*-attention` families) | the last state summarises a short window; over a long one the informative step is anywhere, and a learned readout finds it | `mean` (pooled inside the block) |
| bidirectional (`defaults/bidirectional`) | `false` | `forecast` and `stream` serving forbid the future — a bidirectional block reads it (W3) | `true` for classification of complete windows (the `bi-*` families) |
| window context (`defaults/window_context`) | `previous-part`: an anchor's `length − 1` context rows may come from the previous part of the same series (train before valid, valid before test), so every valid and test row is scored | every valid and test row gets a prediction; the context rows are inputs the deployment also has | `none`: drop the first `length − 1` rows of each series in each part and count them unscored |
| window series (`defaults/window_series`) | the `lag` step's `group_by`, else the split's `by`, else `null` | a window that crosses two series mixes two processes | `null` with a `user:` source |
| autoregressive (`defaults/autoregressive`) | `true` for `forecast` — the target's rows up to the anchor are one more input channel (`F′ = F + 1`) — and `false` otherwise; never `true` on a `sequence-*` task, where the channel would carry the anchor's own label | a forecast without the target's own past discards the strongest predictor | `false` (an exogenous-only forecast) |
| graph batching (`defaults/graph_batching`) | `graphs` at `level: graph`; else `full` when the training view has at most 200 000 nodes and 5 000 000 edges; else `sampled` with `neighbours: [25, 10]` (graphsage's) | full-batch gradients are exact and cheapest while the graph fits | `sampled` (memory), `full` (exactness) |

The rows' keys for `catalog:` are the identifiers enumerated here, not the column text (`defaults/optimizer`, `defaults/schedule`, `defaults/batch_size`, `defaults/max_epochs`, `defaults/early_stopping`, `defaults/init`, `defaults/dropout`, `defaults/normalisation`, `defaults/weight_decay`, `defaults/grad_clip`, `defaults/label_smoothing`, `defaults/precision`, `defaults/seed`, `defaults/imbalance`, `defaults/device`, `defaults/sizing`, `defaults/embedding_dim`, `defaults/graph_protocol`, `defaults/graph_depth`, `defaults/recurrent_return`, `defaults/bidirectional`, `defaults/window_context`, `defaults/window_series`, `defaults/autoregressive`, `defaults/graph_batching`).

## 5. Decision families and the fork format

Forks are batched per decision family, one `FORK:` per family, in this order: **architecture** (family, depth/width, embeddings, normalisation placement), **objective** (loss and its parameters, primary metric), **optimization** (optimizer, learning rate, schedule, batch size, initialisation), **regularization** (dropout, weight decay, early stopping, clipping). A fork names each choice with its alternatives and the one difference that separates them — the `when` clauses — and asks; unattended, the default is applied and marked `default applied (unattended)`. A choice memory already records as a Network stance is not offered. The architecture fork offers what the chain's state admits and nothing else: the graph families only when the manifest has an `edges` part, the recurrent variants (`rnn`, `bi-*`, `*-attention`) only when the sample shape is `window` — and among those, the `bi-*` variants never on a `forecast` task or a `stream` serving mode (W3).

The sample shapes add fork points to those families (machine copy: `fork_order[].points`). Each has an answer under `--unattended`:

| fork point | decision family | asked when | default | `--unattended` |
|---|---|---|---|---|
| `context` | architecture | a `window` sample | `defaults/window_context` | the default, applied and marked |
| `autoregressive` | architecture | a `forecast` task | `defaults/autoregressive` | the default, applied and marked |
| `series_by` | architecture | a `window` sample whose shaping record names no series key (no `lag` `group_by`, no split `by`) | `defaults/window_series` | `null`, with `WARN: windows may cross series — no series key in the shaping record` |
| `protocol` | architecture | a `graph` sample | `defaults/graph_protocol` | always `inductive`: never asked unattended, because `transductive` needs a `user:` turn (W2) |
| `batching` | optimization | a `graph` sample | `defaults/graph_batching` | the default, applied and marked |

Every such default lands in `defaults_to_test`, except `protocol`: it is not a default a rung tests, it is the seal's meaning.

## 6. Metric directions

The direction in which each metric is better — `max`, `min`, or `none` for a descriptive reading that is never a selection metric — for every metric a backend implements, plus the training `loss` (machine copy: `metric_direction`). `metrics.selection.mode` and `early_stopping.mode` follow it, and the later stages (`/vega-train`'s selection, `/vega-eval`'s differences, `/vega-show`'s results) read it from here rather than keep a copy. A per-unit metric (`f1_per_class`, `per_feature_error`, `per_horizon_error`) is read element-wise.

| metric | direction | what it reads |
|---|---|---|
| `roc_auc` | max | ranking of the positive class over the negatives |
| `pr_auc` | max | precision against recall over the thresholds; per label on `multilabel` |
| `pr_auc_macro` | max | `pr_auc` averaged over the labels |
| `accuracy` | max | the share of rows decided right |
| `balanced_accuracy` | max | recall averaged over the classes |
| `precision` | max | the share of positive decisions that are right |
| `recall` | max | the share of positives found |
| `f1` | max | the harmonic mean of precision and recall |
| `f1_macro` | max | `f1` averaged over the classes |
| `f1_micro` | max | `f1` over the pooled decisions |
| `f1_weighted` | max | `f1` averaged with the class shares as weights |
| `f1_per_class` | max | `f1` per class, element-wise |
| `mcc` | max | the correlation of decisions and truth |
| `specificity` | max | the share of negatives kept negative |
| `top_k` | max | the share of rows whose class is among the k best scores |
| `subset_accuracy` | max | the share of rows whose whole label set is right |
| `r2` | max | the share of the target's variance explained |
| `spearman` | max | the rank correlation of prediction and target |
| `qwk` | max | quadratic-weighted agreement of ordered classes |
| `precision_at_k` | max | the precision among the k most anomalous rows |
| `recall_at_k` | max | the recall among the k most anomalous rows |
| `loss` | min | the training objective |
| `log_loss` | min | the negative log-likelihood of the truth |
| `brier` | min | the squared error of the probabilities |
| `ece` | min | the calibration gap over probability bins |
| `hamming` | min | the share of label decisions that are wrong |
| `rmse` | min | the root mean squared error |
| `mae` | min | the mean absolute error |
| `median_ae` | min | the median absolute error |
| `mape` | min | the mean absolute error relative to the target |
| `mae_ordinal` | min | the mean distance in ranks |
| `reconstruction_error` | min | the mean reconstruction error of the numeric columns |
| `per_feature_error` | min | the reconstruction error per feature, element-wise |
| `smape` | min | the symmetric relative error of a forecast |
| `mase` | min | the forecast error over the train-side seasonal naive error |
| `per_horizon_error` | min | the forecast error per horizon step, element-wise |
| `anomaly_score_mean` | min | the mean per-row anomaly score, the unlabelled anomaly primary |
| `residual_quantiles` | none | the distribution of the residuals, described |

## 7. Admission — the closed definition of complete coverage

A **cell** is one family on one sample shape (and graph level) for one effective task; a `sequence-*` task counts once per base it admits, and `anomaly` is run with a label and without one. The table below is the contract the coverage matrix runs cell by cell (machine copy: `admission`); the variants are the extra cells a family must also pass. What the compiler enforces of it is the refusals named below the table (G1b, and W1–W3 where their own rule says so) for the **catalog's** families: the grammar stays open — a family outside the catalog, composed of known blocks, is no cell of this table and is accepted by G1b on its blocks, with the shape walk (G2), the custom-block forward pass and the smoke pass as its proof, never refused for having no row here.

| families | shape | level | admitted tasks | variants | cells |
|---|---|---|---|---|---|
| `mlp-plain`, `mlp-residual`, `ft-transformer`, `linear` | `row` | — | binary, multiclass, multilabel, regression, ordinal | — | 20 |
| `autoencoder`, `vae`, `pca-floor` | `row` | — | unsupervised; anomaly with a label; anomaly without one | — | 9 |
| `deep-svdd` | `row` | — | anomaly with a label; anomaly without one | — | 2 |
| `cnn1d`, `gru`, `lstm`, `transformer-seq`, `rnn`, `lstm-attention`, `gru-attention` | `window` | — | sequence-classification over binary, multiclass, multilabel; sequence-regression over regression, ordinal; forecast | `gru` at stride 2 | 43 |
| `bi-lstm`, `bi-gru` | `window` | — | as above, except forecast (W3) | — | 10 |
| `linear` | `window` | — | the six window kinds | — | 6 |
| `gcn`, `graphsage`, `gat` | `graph` | node | binary, multiclass, multilabel, regression, ordinal | `gcn` transductive; `graphsage` sampled | 17 |
| `gcn`, `graphsage`, `gat`, `gin` | `graph` | graph | binary, multiclass, regression | — | 12 |
| `linear` | `graph` | node | binary, multiclass, multilabel, regression, ordinal (structure-blind) | — | 5 |
| `linear` | `graph` | graph | binary, multiclass, regression | — | 3 |

That is **127 cells**. Not admitted, and the compiler says so by name: a reconstruction family (`autoencoder`, `vae`, `pca-floor`, `deep-svdd`) on `window` or `graph`; a graph family on `unsupervised` or `anomaly`; `gin` at `level: node`; `bi-lstm` or `bi-gru` on `forecast` or under `serving.mode: stream` (W3); any window family on a split that is not `time` (W1).
