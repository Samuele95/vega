# Vega inspect contract — the stage graph, `/v1/network`, `/v1/explain`, the showcase slot

A served network can be read, not only called. This contract fixes how: the **stage graph** that describes any network the backend builds (every block type, every sample shape, every task), the weights it publishes, what one sample does as it passes through, and the showcase slot that serves a presentation beside the answers on the same origin. The machinery lives in the bundle (`/vega-pack` renders it from the backend's serving templates); the presentation lives in `/vega-show` and reads only what this contract promises.

The contract is normative. Its machine copies are `scripts/catalog.json` `inspection` (§3's view table, §4's target and link tables, §7's limits), `scripts/backends/<name>/coverage.json` `inspection` (the block surface of `backend-contract.md` §2.1) and the two schema files of §5 (`references/schemas/network-view-2.schema.json`, `references/schemas/explanation-2.schema.json`); `scripts/tests/test_catalog_sync.py` fails when a copy and this text diverge.

## 1. Scope and the rules of reading a served model

The inspector is a **read-only view of the Predictor's own network**. It answers three questions: what the network is (`vega/network-view@2`, §5.3), what it did with one sample (`vega/explanation@2`, §5.4), and how a batch of samples attributes (`vega/attribution-table@1`, §5.5). It writes no prose: its only strings are identifiers, catalog vocabulary (view kinds, formulas, link and target names) and refusal reasons. Every number comes from the model or from the bundle's records.

**One tracing path, through the Model port.** The inspector reads the network through its Model port alone (`code-contract.md` §1): `anatomy()` — the domain's `NetworkAnatomy`: read-only `stages`, `ids`, `kinds`, `types` and `shapes`, aligned one to one with `spec.blocks`, the readout and the sample, the spec digest, the mode (`training`) and device when asked, the one-class centre, and `forward`, the network's own pass — `head(batch)` for a plain head without gradients, and `param_count()`. `kinds` are the stage kinds of §3 (`emb`, `concat`, `tokens`, `tensor`, `graph`, `gdrop`, `gpool`). It never calls the network object, reads a member off it, or hooks it: the hooks sit on the stage modules the anatomy hands over, so anything that implements the port can be inspected. The network has no `describe()` or `trace()` of its own: the inspector's `StageGraph` and `Capture` are the only inspection path, so two tracings can never disagree.

The rules every inspection call keeps:

- **R1 — hooks once, capture per thread.** `HookSet.attach(model, graph)` runs when the inspector opens and registers one forward hook per graph node module (plus `"<id>:pre"` on the `.linear` of a dense or head node and `"<id>:seq"` on the `.rnn` of a recurrent node) — every one a stage module of `anatomy().stages` or a module inside one, none on the network itself; `Capture.run` keeps the head the anatomy's `forward` returns under `output`. A hook reads a thread-local `Capture`; with none active on the calling thread it returns at once. A `/v1/predict` thread never captures, and a hook never alters an output.
- **R2 — no parameter gradients.** Gradients are taken with `torch.autograd.grad(s, inputs=[leaves…, captured outputs…], allow_unused=True)`. The inspector never calls `.backward()`, `.train()` or `.zero_grad()`, and refuses a network whose `anatomy().training` is true; after any number of explanations no parameter holds a `.grad`.
- **R3 — one `Capture` per call.** A `Capture(graph, mode)` (`mode` ∈ `shapes`, `values`) is entered for one explanation and left after it; tuple outputs keep `output[0]` under the key and the state under `key#state`; graph calls keep the positional graph arguments (`edge_index`, `edge_weight`).
- **R4 — the leaves.** An explanation runs under `torch.enable_grad()`; the leaves are `x_num.requires_grad_()` and, on a graph, `edge_weight = ones(E, requires_grad=True)`. `x_cat` holds ids: its attribution is taken at the embedding (or tokenizer) output.
- **R5 — probes that re-call a submodule** (the attention-weight recomputation, a transformer layer's `self_attn`) run inside `cap.suspended()`.
- **R6 — no shared state in a forward pass.** No block writes an attribute during an evaluation forward: there is no `last_weights`, `last_attention` or `last_edges`, and `Reparameterize.kl` is assigned only in training. The inspector recomputes what it shows (the attention weights, the KL term) and never reads such state.
- **R7 — the explained batch is the predicted batch.** The inspector takes its batch from `Predictor.prepare(df_raw, history=None)`, the only raw → batch path, which returns the `SampleIndex` naming the raw row each sample answers for. The network's forward returns outputs aligned with `y` (at node level it gathers `target_index` after the head), and the inspector resolves an explained row `at` to its position among the targets through `SampleIndex.rows`.

## 2. Placement

The inspector implements no blueprint port. It is a set of **support modules inside the Predictor's module**, map entries `kind: support`, `used_by: ["Predictor"]`, the pattern `code-contract.md` gives `model/metrics.py`. The templates live in `backends/pytorch/templates/serving/package/inspection/` and render into `<package>/<serving module>/inspection/` (vega-pack's `serving_templates` pattern `package/inspection/*.py`); `build_manifest` pins them with the rest of the package tree.

| file | holds |
|---|---|
| `inspection/__init__.py` | exports `NetworkInspector`, `ExplainQuery` |
| `inspection/inspector.py` | `class NetworkInspector`, the facade (§6.3) |
| `inspection/query.py` | `QueryRules` and the frozen `ExplainQuery` (§5.4): a request parsed and checked for the bundle's sample shape before the model runs |
| `inspection/graph.py` | `StageNode`, `StageEdge`, `StageGraph` (§3.2) and the dense chain (§5.3) |
| `inspection/wiring.py` | the placement steps 1–5 of §3.2 (`Build`, `Wiring`, `ShapeProbe`) |
| `inspection/view.py` | the `vega/network-view@2` body around the graph: `task`, `sample`, `inputs`, the header, the canonical bytes and `network_digest` (§5.3) |
| `inspection/readers.py`, `readers_seq.py`, `readers_graph.py`, `readers_group.py` | the readers, one per type of §3.1 (core; sequence; graph; composites, latent, opaque and synthetic) |
| `inspection/derived.py` | the numbers the readers derive from a module's parameters (norms, spectra, projections) and their names (§5.3) |
| `inspection/registry.py` | `READERS`, a constant mapping from each of the 33 block types and 5 synthetic types to its reader |
| `inspection/capture.py` | `Capture`, `HookSet` (§1) |
| `inspection/focus.py` | which slice of every captured tensor is the explained sample (§1 R7, §5.4) |
| `inspection/parts.py` | what each node of a captured pass contributes to an explanation (`nodes`, `recompute`, §5.4) |
| `inspection/probes.py`, `probes_seq.py`, `probes_graph.py`, `probes_group.py` | each node's `probe` and `recompute` (§3.1), per block family |
| `inspection/reduce.py` | `Reducer`, `Budget` (§5.4) |
| `inspection/codec.py` | `TensorCodec`, `num()`, `LIMITS` (§5.1, §5.2, §7) |
| `inspection/targets.py`, `target_kinds.py` | `ExplainTarget`, `TaskTargets`, `TARGET_COLUMNS` and the eleven target kinds (§4) |
| `inspection/attribution.py`, `attribution_parts.py`, `attribution_graph.py`, `interpolate.py` | `Attributor` (§5.4 *Attribution*): gradient × input, integrated gradients along the straight path, a graph sample's nodes by hop and edges |
| `inspection/explanation.py` | one `vega/explanation@2` body and its fidelity checks (§5.4) |
| `inspection/attribution_table.py` | `vega/attribution-table@1` (§5.5) |

The serve side is one route module, `serve/routes_inspect.py` (capability `inspection`, §6.1), rendered from `templates/serving/serve/`; `--no-inspect` renders neither it nor any file above.

| gate check | how it is met |
|---|---|
| C3 | pack adds one support entry per inspection file to `blueprint-map.json` |
| C4 | the files sit in the Predictor's module and import only `torch`, `numpy` and the package's `domain` |
| C5 | adapters may import support; the inspection modules import no adapter — the model and the predictor are handed in |
| C11–C14 | every module ≤ 400 lines, every class ≤ 200, every method ≤ 30 statements and ≤ 5 parameters; `serving/predict.py` keeps exactly one public class, `Predictor` |

## 3. Block views

### 3.1 The view table

Every node of the stage graph has a `type`: one of the catalog's 33 block types, or one of five synthetic types — `input` (one per input stream: `input.num`, `input.cat`, `input.edges`), `output` (the task readout), `residual_add` (one per residual repeat), `parallel_merge` (the merge of a parallel block), `encoder_layer` (one per layer of a transformer encoder). A block type without a reader is refused when the graph is built (`no inspection reader for block type '<t>'`), and pack turns that into `ERROR:`.

The table binds each type to its view (the renderer hint the page picks by), its call (the stage kind the network gives a top-level block of that type; §3.2), the parameter payload keys (§5.3), the probe keys of an explanation, the extra tensors a `detail: "exact"` explanation carries beyond `in`, `out` and `grad_out` (§5.4), the recompute check (`—` when none applies), and the formula the stepper shows.

| type | view | call | payload | probe | exact | recompute | formula |
|---|---|---|---|---|---|---|---|
| `dense` | units | tensor | `weight`, `bias`, `row_norms`, `col_norms`, `activation` | `pre`, `post`, `grad_pre` | `pre`, `post`, `grad_pre` | W·in + b = pre | `z = W·x + b ; a = act(z)` |
| `embedding` | table | emb | `tables` | `id`, `vector`, `grad`, `attribution` | — | table[id] = vector | `e_c = E_c[id_c]` |
| `concat` | join | concat | `parts` | — | — | [parts] = out | `h = [x_num ; e_1 ; … ; e_m]` |
| `batchnorm` | affine | tensor | `gamma`, `beta`, `running_mean`, `running_var`, `eps`, `scale`, `shift` | — | — | scale·in + shift = out | `y = γ·(x − μ)/√(σ² + ε) + β` |
| `layernorm` | affine | tensor | `gamma`, `beta`, `eps` | `mean`, `std` | — | the formula over the sample = out | `y = γ·(x − mean x)/√(var x + ε) + β` |
| `dropout` | pointwise | tensor | `p`, `active_at_inference` | — | — | in = out | `y = x (inactive at inference)` |
| `activation` | pointwise | tensor | `name` | — | — | act(in) = out | `a = act(z)` |
| `residual` | group | tensor | `repeats` | — | — | — | `x ← P·x + f(x)` |
| `sequential` | group | tensor | `children` | — | — | — | `f_n ∘ … ∘ f_1` |
| `parallel` | group | tensor | `branches`, `merge` | — | — | — | `[f_1(x) ; … ; f_n(x)] \| Σ_i f_i(x)` |
| `conv1d` | conv | tensor | `kernel`, `bias`, `channel_norms`, `tap_norms`, `pair_norms`, `stride`, `padding`, `dilation` | `maps` | — | conv(in) = out | `y[t,o] = b[o] + Σ_c Σ_k W[o,c,k]·x[t·s+k−p, c]` |
| `pool1d` | pool | tensor | `kind`, `size` | `winners` | — | the pooling formula over in = out | `y[t,c] = max\|mean_{m} x[t·m+m', c]` |
| `rnn` | recurrent | tensor | `cell`, `hidden`, `layers`, `directions`, `gates`, `return`, `per_layer` | `unroll` | `hidden` | the step-by-step unroll = the module's sequence output | `h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t−1} + b_hh)` |
| `gru` | recurrent | tensor | `cell`, `hidden`, `layers`, `directions`, `gates`, `return`, `per_layer` | `unroll` | `hidden` | the step-by-step unroll = the module's sequence output | `r,z = σ(…) ; n = tanh(W_in·x+b_in + r⊙(W_hn·h+b_hn)) ; h_t = (1−z)⊙n + z⊙h_{t−1}` |
| `lstm` | recurrent | tensor | `cell`, `hidden`, `layers`, `directions`, `gates`, `return`, `per_layer` | `unroll` | `hidden` | the step-by-step unroll = the module's sequence output | `i,f,o = σ(…) ; g = tanh(…) ; c_t = f⊙c_{t−1} + i⊙g ; h_t = o⊙tanh(c_t)` |
| `attention` | attention | tensor | `heads`, `dim`, `head_dim`, `in_proj`, `out_proj`, `per_head`, `pool` | `weights` | `alpha` | the recomputed attention over in = out | `A_h = softmax(Q_h K_hᵀ/√d_h) ; y = W_o·[A_1 V_1 ; … ; A_H V_H]` |
| `attention_pool` | attention | tensor | `score`, `query` | `weights` | `alpha` | Σ_t α_t h_t = out | `α = softmax_t(q·tanh(W h_t + b)) ; y = Σ_t α_t h_t` |
| `transformer_encoder` | group | tensor | `layers`, `d`, `heads`, `ff`, `norm_first` | — | — | — | `x ← x + MHA(LN₁ x) ; x ← x + FF(LN₂ x)` |
| `feature_tokenizer` | tokens | tokens | `weight`, `bias`, `cls`, `tables`, `token_norms`, `cls_norm` | `tokens` | — | the token formula over in = out | `T_j = x_j·W_j + b_j ; tokens = [cls ; T_1 … T_n]` |
| `positional` | tokens | tensor | `kind`, `pe`, `learned` | — | — | in + pe = out | `x_t ← x_t + P_t` |
| `flatten` | reshape | tensor | — | — | — | reshape(in) = out | `(L,d) → (L·d)` |
| `reshape` | reshape | tensor | `shape` | — | — | reshape(in) = out | `(n) → shape` |
| `pool` | pool | tensor | `kind` | `winners`, `cls` | — | the pooling formula over in = out | `mean_t x_t \| max_t x_t \| x_0` |
| `head` | units | tensor | `weight`, `bias`, `row_norms`, `col_norms`, `activation`, `thresholds` | `pre`, `post`, `grad_pre` | `pre`, `post`, `grad_pre` | W·in + b = pre | `z = W·x + b ; ordinal: z_j = w·x + θ_j with θ_0 ≥ θ_1 ≥ …` |
| `decoder` | group | tensor | `children` | — | — | — | `x̂ = W_k·act(… act(W_1 z + b_1) …) + b_k` |
| `reparameterize` | latent | tensor | `z` | `mu`, `sigma`, `kl` | — | mu = out | `z = μ at inference ; KL = −½ Σ (1 + log σ² − μ² − σ²)` |
| `gcn_conv` | graph | graph | `weight`, `bias`, `normalize`, `self_loops` | `messages` | — | Σ_j coef·W h_j + b at the target = out | `h'_i = Σ_{j∈N(i)∪{i}} Â_ij·W h_j + b` |
| `sage_conv` | graph | graph | `root`, `neigh`, `bias`, `aggregator`, `normalize` | `messages` | — | root + aggregate at the target = out | `h'_i = W_r h_i + W_n·agg_{j∈N(i)} h_j` |
| `gat_conv` | graph | graph | `weight`, `bias`, `att_src`, `att_dst`, `heads`, `concat`, `negative_slope` | `messages` | `alpha`, `edge_index` | the heads of Σ_j α·W h_j at the target = out | `α_ij = softmax_j(LeakyReLU(a_srcᵀ W h_j + a_dstᵀ W h_i)) ; h'_i = ‖_h Σ_j α^h_ij W^h h_j` |
| `gin_conv` | graph | graph | `mlp`, `eps`, `train_eps` | `messages`, `pre_mlp` | — | mlp(pre_mlp) at the target = out | `h'_i = MLP((1+ε)·h_i + Σ_{j∈N(i)} h_j)` |
| `graph_pool` | pool | gpool | `readout` | `node_share` | — | the readout over the graph's nodes = out | `g = mean\|max\|sum_{i∈G} h_i` |
| `edge_dropout` | edges | gdrop | `p`, `active_at_inference` | — | — | edges in = edges out | `E' = E (inactive at inference)` |
| `custom` | opaque | tensor | `class`, `parameters`, `declared_out_shape`, `verified` | — | — | — | `declared by user code (opaque)` |
| `input` | io | io | `columns`, `stream` | `values` | — | — | `x = the shaped inputs (numeric, nominal ids, edges)` |
| `output` | io | io | `link`, `outputs` | `readout` | — | — | `ŷ = link(z)` |
| `residual_add` | merge | inner | — | `skip_norm`, `branch_norm` | — | P·x + f(x) = out | `y = P·x + f(x)` |
| `parallel_merge` | merge | inner | `merge` | `branch_norms` | — | concat or sum of the branches = out | `[b_1 ; … ; b_n] \| Σ_i b_i` |
| `encoder_layer` | attention | inner | `self_attn`, `ff`, `norm1`, `norm2`, `norm_first`, `activation` | `weights`, `cls_row`, `stream_norms` | `alpha` | x + MHA(LN₁ x), then + FF(LN₂ ·) = out | `x ← x + MHA(LN₁ x) ; x ← x + FF(LN₂ x)` |

The view kinds are closed: `units`, `table`, `join`, `affine`, `pointwise`, `group`, `conv`, `pool`, `recurrent`, `attention`, `tokens`, `reshape`, `latent`, `graph`, `edges`, `opaque`, `io`, `merge`. A residual projection is a child of type `dense` with `role: "projection"`; each linear layer inside a `decoder` is a child of type `dense` with `role: "decoder_layer"`. The ordinal head is type `head` too: its `weight` is the shared score (`1 × in`), its `bias` is `null`, and `thresholds` are the `k−1` ordered offsets `θ_j` the network adds to the score.

On a weighted graph every convolution multiplies its messages by the edge weight `w_ij`; for `gat_conv` the weight multiplies `α_ij` **after** the softmax, without renormalising (the message is `α^h_ij · w_ij · W^h h_j`), and what the block's `coefficients(...)` returns — and the explanation publishes as the node's `alpha` — is that product `α^h_ij · w_ij`, the coefficient the message actually carries (`backend-contract.md` §2.1).

### 3.2 Building the graph

The graph is built once, when the inspector opens, from the Predictor's own network, the spec's `blocks` and one example batch.

1. **Alignment.** `len(spec.blocks) == len(anatomy.stages)` (the anatomy holds its `ids`, `kinds`, `types` and `shapes` aligned with its stages, or refuses to be built), else `GraphError("the network is not the spec's: <n> blocks, <m> stages")`. Every block id must match `^[A-Za-z0-9_.@:-]{1,128}$` (`netspec-contract.md` §2): node and tensor ids are built from it (`<node>.<name>`), and a graph with another id is refused.
2. **Inputs.** `input.num` carries the numeric stream (`[n_num]`, or `[L, F′]` on a window); `input.cat` is added when the contract has nominal columns; `input.edges` when `sample.shape == "graph"`.
3. **Top-level blocks.** Block `i` becomes node `<id>` (or `<id>@<i>` when two top-level ids collide) at path `stages.<i>`, expanded by its reader; its incoming data edges follow the stage kind:

   | call (stage kind) | the network calls | incoming edges |
   |---|---|---|
   | `emb` | `e = m(x_cat)`; the running tensor is untouched | from `input.cat`, kind `embed` |
   | `concat` | `h = m(x_num, e)` | from `input.num` and from the last `emb` exit |
   | `tokens` | `h = m(x_num, x_cat)` | from `input.num` and `input.cat` |
   | `tensor` | `h = m(h)` | from the running exit |
   | `graph` | `h = m(h, edge_index, edge_weight)` | from the running exit, plus an `edges` edge from `input.edges` or the last `gdrop` node |
   | `gdrop` | `edge_index, edge_weight = m(edge_index, edge_weight)` | from `input.edges` or the previous `gdrop` node, kind `edges` |
   | `gpool` | `h = m(h, graph_index, num_graphs)` | from the running exit |

   A node nested inside a composite has call `inner`; the `input` and `output` nodes have call `io`.
4. **Nested ids** are deterministic: `sequential` `<id>.<k>`; `residual` `<id>.r<rep>.<k>`, its projection `<id>.r<rep>.proj`, its add node `<id>.r<rep>.add` with a `skip` edge from the repeat's input; `parallel` branches `<id>.b<k>` (nesting further as `<id>.b<k>.<j>`) and merge `<id>.merge`; `decoder` `<id>.l<k>`; `transformer_encoder` `<id>.layer<k>`.
5. **The output node** `output` is fed from the last exit, kind `readout`.
6. **Shapes** come from one no-grad forward of the example (the anatomy's `forward`) inside `Capture(mode="shapes")`. The batch axis is dropped; on a graph the node or graph axis is dropped and recorded as `per` (`sample`, `node`, `graph`). A top-level `out_shape` must equal `anatomy.shapes[i]`. Axes name the kept dimensions: a vector `["unit"]`, a sequence `["step", "channel"]`, tokens `["token", "channel"]`, the head `["output"]`, the numeric input `["feature"]` (`["step", "feature"]` on a window).
7. **Parameters.** `params.count` sums `numel` over the node module's parameters (the composite total for a group node); `params.trainable` counts the trainable ones. The top-level counts sum to `model.param_count()[0]`, and only leaf nodes are summed when a reader totals the graph.
8. **Code.** `code = {"file", "class", "anchor"}` names the module file, the class and `class <Name>(`; the page binds nodes to `/v1/code` files by it.

## 4. Explained targets

An explanation explains one differentiable scalar `s` of the explained sample: its **target**. The effective kind is `task.base_kind` for the `sequence-*` kinds and `task.kind` otherwise; the anomaly flavour follows `spec.loss.name` (`svdd` gives the one-class target, `reconstruction` and `vae` the reconstruction targets).

### 4.1 Targets per task

| `task.kind` | link | decision rule | default target | selectable targets |
|---|---|---|---|---|
| `binary` | `sigmoid` | `threshold` | `binary_logit` — the class predicted at the threshold | `binary_logit` |
| `multiclass` | `softmax` | `argmax` | `logit` — the argmax class | `logit` |
| `multilabel` | `sigmoid_each` | `per_label_threshold` | `label_logit` — the label of the largest σ | `label_logit` |
| `regression` | `identity` | `none` | `value`; under the quantile loss the `quantile` nearest 0.5 | `value`; `quantile` under the quantile loss |
| `ordinal` | `cumulative` | `cumulative_count` | `expected_rank` | `expected_rank`, `threshold_logit` |
| `unsupervised` | `reconstruction` | `none` | `reconstruction_error` | `reconstruction_error`, `feature_error` |
| `anomaly` | `reconstruction`; `distance` under the `svdd` loss | `threshold` | `reconstruction_error`; `svdd_distance` under `svdd` | `reconstruction_error`, `feature_error`; `svdd_distance` under `svdd` |
| `sequence-classification` | as the base (`binary`, `multiclass`, `multilabel`) | as the base | as the base | as the base |
| `sequence-regression` | as the base (`regression`, `ordinal`) | as the base | as the base | as the base |
| `forecast` | `horizon` | `none` | `forecast_step` 1 | `forecast_step` |

### 4.2 Target kinds and request selectors

A request picks a target with at most one selector of `target`: `class`, `label`, `threshold`, `quantile`, `feature`, `step`, or `kind` (a target kind by name). A selector the task does not admit is a 422 naming it ("target.class does not apply to a regression task"); an unknown class, label or feature name is a 422 too.

| target kind | selector | scalar `s` | unit |
|---|---|---|---|
| `logit` | `class` | `z_c` | logit |
| `binary_logit` | `class` | `sign·z`, sign +1 for the positive class and −1 for the other | logit |
| `label_logit` | `label` | `z_j` | logit |
| `value` | — | `ŷ` | target |
| `quantile` | `quantile` | `ŷ_q` | target |
| `expected_rank` | — | `Σ_j σ(z_j)` | rank |
| `threshold_logit` | `threshold` | `z_j` | logit |
| `reconstruction_error` | — | `e(x) = mean_f (x̂_f − x_f)² + mean_c CE_c` | error |
| `feature_error` | `feature` | `e_f(x)` | error |
| `svdd_distance` | — | `‖f(x) − c‖²` | distance |
| `forecast_step` | `step` | `ŷ_{t+h}` | target |

The request vocabulary of the page and of the serving readings maps onto these kinds: `class` / `true_class` → `logit` (`binary_logit` on a binary task); `label` → `label_logit`; `value` → `value` or `quantile`; `expected_ordinal` → `expected_rank`; `error` → `reconstruction_error`; `distance` → `svdd_distance`; `horizon_step` / `horizon` → `forecast_step`; `row_score` → `reconstruction_error` or `svdd_distance`.

### 4.3 Links

One link vocabulary serves the whole chain — `MANIFEST.output.link`, `network-view@2 task.link`, the explanation's readout and the page. Each link names the readings it serves and the decision each reading takes; the decision rule maps onto `MANIFEST.output.decision` as `argmax` → `argmax`, `threshold` → `threshold`, `per_label_threshold` → `per-label-threshold`, `cumulative_count` → `cumulative`, `none` → `none`.

| link | readings (decision rule) | formula |
|---|---|---|
| `softmax` | `multiclass` (argmax) | `p_i = exp(z_i) / Σ_j exp(z_j)` |
| `sigmoid` | `binary` (threshold) | `p = σ(z)` |
| `sigmoid_each` | `multilabel` (per_label_threshold) | `p_k = σ(z_k), each label on its own` |
| `identity` | `regression` (none), `quantile` (none) | `ŷ = z` |
| `cumulative` | `ordinal` (cumulative_count) | `P(y > j) = σ(z_j) ; p_0 = 1 − σ(z_0), p_j = σ(z_{j−1}) − σ(z_j), p_{K−1} = σ(z_{K−2})` |
| `reconstruction` | `reconstruction` (none), `anomaly-reconstruction` (threshold) | `e(x) = mean_f (x̂_f − x_f)² + mean_c CE_c` |
| `distance` | `anomaly-svdd` (threshold) | `s = ‖f(x) − c‖²` |
| `horizon` | `forecast` (none) | `ŷ_{t+h} = z_h, h = 1…H (quantile loss: z_{h,q}, horizon-major)` |

### 4.4 Definitions

- **Binary classes** are `MANIFEST.output.classes`, else `("0", "1")`. The positive class is `task.positive`, else the second class. Scores are logits everywhere: the probability is `σ(z)`, applied unconditionally, and the predicted class is positive when `σ(z) ≥ MANIFEST.output.threshold` (probability scale).
- **Ordinal.** The head is rank-consistent, so `σ(z_j)` falls with `j`; the decision is `Σ_j [σ(z_j) ≥ 0.5]` (`cumulative_count`), and the class probabilities are those of the `cumulative` link.
- **Reconstruction error** is the loss's per-row form, the one `TaskReadout` scores: the mean squared error over the numeric columns plus, over each embedded nominal, the cross-entropy of its logits slice (`recon_dim` lays the numeric columns first, then each nominal's `cardinality` logits in contract order). `feature_error` for a feature is its squared error, or its cross-entropy for a nominal.
- **SVDD distance** reads the centre from the network's registered buffer `svdd_center` (set once by the loss, saved with the weights). When the buffer is absent or `svdd_center_set` is false the inspector refuses to open (`why = "the one-class centre was not persisted with the weights"`); no distance is guessed.
- **Quantiles** come from `spec.loss.params.quantiles`, named `q<q>` (`q0.1`, `q0.5`). A forecast head under the quantile loss is horizon-major, `(B, H, Q)`.

### 4.5 The readout

`explanation.outputs` and `network.task.outputs` describe the head the same way:

```json
{ "kind": "classes | score | labels | value | quantiles | thresholds | reconstruction | embedding | horizon",
  "names": ["…"], "link": "<§4.3>", "raw": [0.0], "read": [0.0], "answer": { } }
```

`names` are the classes (`multiclass`), the positive class (`binary`, kind `score`), the labels, `["value"]`, the quantile names, the thresholds `">c_0"` … (`ordinal`), the feature names (`reconstruction`), `"z0"` … (`embedding`, svdd) or `"t+1"` … `"t+H"` (`horizon`; `"t+<h>/q<q>"` under the quantile loss, horizon-major). `raw` is the head's output for the sample; `read` is the output after the link (softmax, σ, σ per label, identity, σ per threshold, x̂) and `null` for `distance`. A σ output is never renormalised: `read` claims no distribution the head does not produce. `answer` is task-specific: `{class, p}`, `{class, p, threshold}`, `{labels}`, `{value}`, `{rank, expected_rank}`, `{error, per_feature}`, `{distance}`, `{steps}`.

### 4.6 The served-field cross-check

The inspector compares its target value with the Predictor's answer for the same row where one exists. It finds the served field by `(kind, of)` in `MANIFEST.output.field_kinds` — `raw` for a logit (`of` the class or label; binary's `score` is the positive class's raw logit), `value`, `quantile` (`of` the quantile), `horizon` (`of` the step), `score` and `row_score` for the anomaly and reconstruction scores — never by a fixed column name. A target with no served field is recorded `served: null`.

## 5. Payloads

### 5.1 Numbers

Every number derived from a float32 tensor carries **9 significant digits** — `num(v) = float(f"{float(v):.9g}")` — which round-trips float32 exactly. Summaries are computed in float64 and then rounded the same way, so they are deterministic byte for byte. `NaN` and `±inf` are refused as a defect: the service emits no non-finite JSON (`allow_nan=False`).

### 5.2 Tensors

```json
{ "id": "B1.weight", "node": "B1", "role": "weight | bias | table | buffer", "shape": [256, 65], "axes": ["out", "in"],
  "dtype": "float32", "encoding": "json | base64-le | deferred | withheld",
  "data": [0.0] | "<base64 of little-endian float32, row-major>" | null,
  "summary": { "n": 16640, "mean": 0.0012, "std": 0.071, "min": -0.41, "max": 0.39, "l2": 9.21,
               "abs_q": [0.047, 0.118, 0.203, 0.41], "hist": { "lo": -0.41, "hi": 0.39, "counts": [0] } } }
```

- Ids follow `^[A-Za-z0-9_.@:-]{1,128}$` and read `<node id>.<name>`: `B3.r0.1.weight`, `B2.layer0.in_proj`, `B1.table.zip`, `B4.l0.w_ih.reverse`.
- `role` is `weight` (a learned multiplier or vector), `bias` (a learned additive vector), `table` (an embedding table) or `buffer` (registered state that is not learned). The one-class centre is the tensor `<head id>.centre` with role `buffer`; batchnorm running statistics are buffers too.
- Encoding: `json` (a flat row-major list) when `n ≤ json_list_max`; else `base64-le` when `n ≤ inline_tensor_max` and the running inline total stays `≤ inline_total_max` (tensors taken in node order); else `deferred` (fetched from `/v1/network/tensor/{id}`). Every tensor is `withheld` under weights mode `summary`. The summary is always present; `abs_q` are the |w| quantiles 50, 90, 99 and 100, and `hist` has 32 bins.

### 5.3 `GET /v1/network` — `vega/network-view@2`

Normative schema: `references/schemas/network-view-2.schema.json`.

```json
{ "schema": "vega/network-view@2", "available": true,
  "model_version": "sha256:…", "spec_digest": "sha256:…", "network_digest": "sha256:…", "weights": "full",
  "family": { "name": "mlp-plain", "source": "blueprint:posture" },
  "task": { "kind": "binary", "base": null, "link": "sigmoid",
            "outputs": { "kind": "score", "names": ["attack"] },
            "positive": "attack", "threshold": 0.42, "quantiles": null, "decision_rule": "threshold",
            "targets": ["binary_logit"], "default_target": "binary_logit" },
  "sample": { "shape": "row", "length": null, "stride": null, "horizon": null, "autoregressive": null,
              "level": null, "directed": null, "self_loops": null, "protocol": null, "batching": null },
  "inputs": { "numeric": ["…"], "nominal": [ { "name": "zip", "cardinality": 42, "dim": 12 } ], "past_target": null,
              "order": ["…"], "width": 65, "edges": null,
              "axis_labels": { "feature": ["…"], "step": null, "token": null } },
  "params": { "total": 86543, "trainable": 86543 },
  "nodes": [ { "id": "B1", "block": "B1", "type": "dense", "view": "units", "parent": null, "children": [],
               "role": null, "call": "tensor", "path": "stages.0", "in_shape": [65], "out_shape": [256],
               "axes": ["unit"], "per": "sample", "config": { "units": 256 },
               "params": { "count": 16896, "trainable": 16896, "weight": "B1.weight", "bias": "B1.bias" },
               "formula": "z = W·x + b ; a = act(z)",
               "code": { "file": "<pkg>/model/blocks_core.py", "class": "Dense", "anchor": "class Dense(" } } ],
  "edges": [ { "from": "input.num", "to": "B1", "kind": "data" } ],
  "chain": null, "tensors": { "B1.weight": { } },
  "limits": { }, "inspection": { "schema": "vega/inspection@1", "readers": "sha256:…" } }
```

- **`available`** is always `true`: a network that cannot be inspected is a 404 problem, never a body.
- **`network_digest`** is the sha256 of the canonical serialisation (`sort_keys`, separators `(",", ":")`, `allow_nan=False`) of the body without that key. Two GETs return identical bytes; when pack pinned the digest (`MANIFEST.serve.inspection.network_digest`), the served body must carry it.
- **`task`** carries the link and decision rule of §4, `positive` (binary and anomaly with classes), `quantiles` (the quantile levels, or `null`), `threshold` (probability scale for binary and multilabel, score scale for anomaly; `null` when none applies), the selectable target kinds and the default one.
- **`inputs`.** `numeric` and `nominal` are the network's input columns and **equal `netspec.data_contract.features[].name`**, numeric then nominal, in that order — the page's lexicon and input groups key off these names. `order` is numeric, then nominal, then — for an autoregressive forecast — the target column, whose past rows are the last channel of the numeric stream (`past_target` names it). `axis_labels.feature` labels the channels of `input.num`: the numeric names, then `{"name": "<target>", "past": true}` for the autoregressive channel. `axis_labels.step` is `t−(L−1) … t−1, t` on a window; `axis_labels.token` is `[cls]` then the feature names (ft-transformer). `edges` is `{"weighted": bool}` on a graph, else `null`.
- **`nodes`** are the stage graph (§3.2) in graph order, each with the payload of its type (`params` beside `count` and `trainable`; tensor references are ids into `tensors`, small vectors are inline lists). The payload keys per type are §3.1's `payload` column. For the recurrent types `per_layer` holds, per layer and direction, `w_ih`, `w_hh`, `b_ih`, `b_hh` (ids), `gate_norms`, `input_columns` (the column norms of each gate's input block, labelled by feature name on layer 0) and `rho_hh` (each gate's spectral radius); `gates` is `["h"]`, `["r","z","n"]` or `["i","f","g","o"]` in torch order. An embedding's `tables` entry is `{column, cardinality, dim, table, row_norms, pca2, unseen_row}`. `custom` nodes carry `verified: false`.
- **`chain`** keeps the exact dense drawing for MLP-like networks. It is non-null only when the sample is `row`, every top-level type is one of `embedding`, `concat`, `dense`, `activation`, `dropout`, `head`, the last top-level block is the `head`, the dense and head nodes chain width to width from the concat width (each one `nn.Linear` of shape `(out, in)` — so the ordinal task's rank-consistent cumulative head, one shared score and its thresholds, is never a chain layer, and its network's chain is null), dropouts fold into the preceding layer's `dropout` (the kept share multiplies), and each dense has at most one activation before the next: `{eligible, linear_reading, inputs, layers: [{node, in, out, activation, dropout, weight, bias, params}]}`.

`GET /v1/network/tensor/{tensor_id}` answers one tensor payload with `encoding: "base64-le"`, encoded on demand from the live parameters (an 8-entry LRU at most); an unknown id, weights mode `summary` or inspection off is a 404 problem. Neither route is admitted: they read memory and never run the model.

### 5.4 `POST /v1/explain` — `vega/explanation@2`

**Request** (extra fields forbidden):

| field | shape | meaning |
|---|---|---|
| `row` | one raw row | a `row` sample: the one row (exactly one of `row` or `rows` of length 1) |
| `rows` | 1 ≤ n ≤ `explain_rows_max` raw rows | a `window` sample: rows in time order (several series when `series_by` is set); a `graph` sample: the query nodes (node level) or the rows of the graphs (graph level) |
| `history` | ≤ the contract's `max_rows` raw rows | a window's preceding rows (lags and warm-up); refused on a graph |
| `at` | int | window: the index in `rows` of the explained window's anchor (default the last row); graph, node level: the explained query node (default the first) |
| `target` | `{class \| label \| threshold \| quantile \| feature \| step \| kind}` | at most one selector (§4.2) |
| `method` | `grad_x_input` \| `integrated_gradients` | default `grad_x_input` |
| `steps` | 4 ≤ n ≤ `ig_steps_max` | integrated gradients only, default 32; `steps × nodes ≤ ig_batch_max` |
| `detail` | `full` \| `summary` \| `exact` | default `full` |

A graph request carries no edges: they are derived as serving derives them — at node level each query node is attached to the bundle's reference graph through the transform module's `edges_for` (query → reference edges only, so an answer never depends on the other rows of the request); at graph level each graph's edges come from `edges_within`. A window needs `length` rows ending at `at` across `history + rows`, else a 422 ("a window needs L rows ending at `at`").

**Response.** The body is the serving envelope plus `"explanation"`, which is:

```json
{ "schema": "vega/explanation@2", "detail": "full", "degraded": false,
  "sample": { "shape": "row", "index": 0, "row": 0, "span": null, "graph": null },
  "prediction": { "label": "…", "probabilities": { }, "warnings": [] },
  "target": { "kind": "logit", "index": 3, "name": "…", "sign": 1, "value": 2.31, "why": "predicted", "unit": "logit" },
  "outputs": { "kind": "classes", "names": ["…"], "link": "softmax", "raw": [0.0], "read": [0.0], "answer": { } },
  "inputs": { "order": ["…"], "shaped": [0.0], "raw": { } },
  "attribution": { "method": "grad_x_input", "unit": "logit", "shape": "row",
                   "features": { "names": ["…"], "values": [0.0], "attribution": [0.0], "grad": [0.0] },
                   "total": 0.0, "completeness": null },
  "nodes": { "B1": { "level": "full", "why": null, "in": { }, "out": { }, "grad_out": { }, "relevance": { },
                     "extra": { }, "recompute": 0.0, "exact": null } },
  "chain": null,
  "fidelity": { "head_max_abs_diff": 0.0, "served_max_abs_diff": 0.0, "tolerance": 1e-05, "ok": true },
  "recompute": { "nodes": 7, "worst": 0.0, "worst_node": null },
  "budget": { "limit": 262144, "used": 1234, "summarised": [], "omitted": [] },
  "warnings": [] }
```

- `sample.graph` (a graph sample) gives the assembled graph's `level`, `nodes`, `edges` and `target_node`. A node id anywhere in the explanation is the node's index in the graph the service assembled (the reference nodes first, then the request's rows), never the data's name for it. At node level, when the graph names its nodes (a two-column `key_pair`: its first column is each row's own key), `sample.graph.names = {column, of}` gives that column and, by node id, its value for every reference node the explanation mentions (`GraphContext.reference_names`, from the `raw:<key>` column the reference carries); a graph that names no node (knn, spatial, a one-column `key_pair`) carries no `names`.
- `prediction` is exactly what `/v1/predict` answers for the sample's row (the serving `RowPrediction`, whose keys follow the reading); `warnings` carries every warning of that answer.
- **Reduced forms.** A captured per-sample tensor is reported as `scalar`, `vector` (≤ `vector_max` values, else the 64 largest by |v| plus a summary), `sequence` (`per_step` and `per_channel` statistics plus a heat map), `nodes` (graph: kept node ids, hops, norms and the target's values) or `attention` (one heat map per head). A heat map is a block mean down to at most `heat_max` per axis (summed for `relevance`). `relevance = out ⊙ grad_out`.
- **Levels and the budget.** Under `detail: "full"` nodes are processed in graph order against `explain_float_budget`: a node that would exceed what remains is rebuilt at level `summary` (per-axis norms and sums, ≤ 64 numbers each), and failing that `omitted` with a `why`. `detail: "summary"` puts every node at `summary`. Under **`detail: "exact"`** every node is at level `exact` and carries `exact`: full `in`, `out` and `grad_out` for the explained sample as base64-le float32 tensors, plus the extra tensors of §3.1's `exact` column — `pre`, `post`, `grad_pre` for dense and head nodes, the per-step hidden states (`hidden`, per layer and direction) of a recurrent node, the full attention `alpha` of an attention, encoder layer, attention pool or GAT node (GAT with its `edge_index`) — under `explain_exact_float_budget`. Past that budget the remaining nodes are summarised and the payload says **`degraded: true`**. The order is fixed, so the same request gives the same bytes.
- **`chain`**, when the network's chain is non-null, holds `inputs` (the concat output), `pre[l]`, `post[l]` and `grad_pre[l]` of each chain layer (dropout is the identity at inference; `post[L−1] = pre[L−1]` for a head without activation), so a reader recomputes `W·post[l−1] + b = pre[l]`.
- **Attribution.** `grad_x_input`: `a = x ⊙ ∂s/∂x` for the numeric inputs, and for an embedded nominal the dot product `Σ_d e_d·∂s/∂e_d` at the embedding output (its `grad` is the norm of that gradient). `integrated_gradients`: the baseline is 0 in shaped space and at each embedding output, the interpolants run as one batch at midpoint α, graph edges held at their own weight (1 on an unweighted graph; the weights are copied, never interpolated), and `completeness = {sum, target_minus_baseline, gap}` is always reported (it is `null` for `grad_x_input`, which makes no such claim). The payload by shape: a row gives one entry per input of `inputs.order`; a window a `steps × features` matrix (a heat map beyond 65 536 cells), `per_step`, `per_feature` and the nominal columns attributed per step; a graph the target node's own features (`self`), the neighbourhood (`neighbours` with hop and total, at most `graph_nodes_max` nodes, nearer hops first), the per-hop totals, what lies beyond the cap (`outside`) and the edge relevances.
- **Fidelity.** `head_max_abs_diff` compares the captured head with a plain no-grad forward of the same batch; `served_max_abs_diff` compares the target value with the served field (§4.6). The tolerance is `MANIFEST.roundtrip.tolerance.value`, else `1e-5`; beyond it the route answers 500 ("the explanation does not reproduce the served answer").
- **Recompute.** Each node's check of §3.1 reports its max |Δ|; `recompute.worst` above `1e-4·max(1, |out|)` is shown, not refused.

### 5.5 `vega/attribution-table@1` (library, no HTTP)

`NetworkInspector.attribute(rows, targets=None, method="grad_x_input", steps=32, history=None)` explains every sample the rows make (with `history` before them, for a window), in chunks of at most 256 samples (one forward and one gradient per chunk, or one run of the interpolants and one gradient for integrated gradients; no node payloads; at node level a chunk keeps the whole graph and a slice of the targets, at graph level whole graphs):

```json
{ "schema": "vega/attribution-table@1", "method": "grad_x_input", "unit": "logit", "shape": "row",
  "inputs": ["<inputs.order>"], "targets": [ { "kind": "logit", "name": "…" } ],
  "samples": [ { "row": 0, "prediction": { }, "target_value": 1.2,
                 "attribution": [0.0] } ] }
```

`attribution` is a list over `inputs.order` (row), `{"per_step", "per_feature"}` (window) or `{"self", "hops"}` (graph). `targets` takes one selector per request row (for instance `{"class": <the true class>}` — a name is a string, an index an integer, so `True`, `"0"` or `{"threshold": 0.5}` is refused), or `None` for every sample's default; each is checked as a request's would be, the `QueryError` naming the selector (`target.<selector>`) and its row (`targets[i]`); every target of one table has one unit, so the samples' totals compare. `samples` are in sample order: `row` the request row the sample answers, `prediction` the served frame's row for it. `/vega-show` builds `vega/attribution@2` from it.

## 6. Endpoints

### 6.1 The route module

The inspection routes live in `serve/routes_inspect.py`, one of the route modules the service composes (`routes_meta`, `routes_predict`, `routes_monitor`, `routes_inspect`, `routes_showcase`). Pack reads the module's literals with `ast.literal_eval`; there is no separate route declaration file.

```python
ENDPOINTS = (("GET", "/v1/network"), ("GET", "/v1/network/tensor/{tensor_id}"), ("POST", "/v1/explain"))
MODEL_ROUTES = (("POST", "/v1/explain"),)      # admitted: takes a place at the model
OPEN_WHILE_UNREADY = ()
CAPABILITY = "inspection"
def register(app, state): ...                    # the three handlers
def on_ready(state): ...                         # opens the inspector after the self-test passed
```

`MANIFEST.serve.endpoints` lists `"METHOD /path"` for every module whose capability the bundle enables, so a bundle packed with `--no-inspect` lists none of the three. Switches, read once at start-up and only ever restricting what the bundle grants: `MANIFEST.serve.inspection.enabled: false` or `<ENV>_INSPECT=off` answers all three routes 404 ("this bundle ships without inspection", "inspection is switched off on this service"); `MANIFEST.serve.inspection.weights: summary` or `<ENV>_INSPECT_WEIGHTS=summary` publishes every tensor `withheld`, and `weights=full` and the tensor route answer 404; a value that is not one of a switch's words restricts, and the log says so. The routes open after the self-test passed (`on_ready`), on the example rows (`examples/rows.csv`, up to 16 samples at `detail: "summary"`); a graph row whose keys match no reference node is explained with the served warning `no neighbour in the reference graph — answered from the row's own features`.

| route | answers | errors |
|---|---|---|
| `GET /v1/network?weights=full\|summary` | the bytes built once at start-up for that weights mode | 503 while unready; 404 problem when inspection is off or unavailable |
| `GET /v1/network/tensor/{tensor_id}` | one tensor (§5.3) | 404 unknown id, weights withheld, inspection off; 503 unready |
| `POST /v1/explain` | §5.4; its `prediction` is the `/v1/predict` answer of the same rows, and `?include=` shows the gated keys as there | 422 request or query validation — a row the row model refuses answers `/v1/predict`'s problem body, a query the explain rules refuse answers `{"title": "Inspection request refused", "errors": [{"column", "field", "error"}]}` with `field` one of `row`, `rows`, `at`, `target.<selector>`, `method`, `steps`, `detail` (a window short of history: `a window needs <length> rows ending at \`at\``); 413 body over the bound, before it is read; 503 unready, or busy (`Retry-After`); 404 inspection off or unavailable — answered before the body is validated; 500 fidelity failure or an attribution that cannot be given (the problem body carries no internals; the log records the measurement) |

### 6.2 Opening and status

`NetworkInspector.open(example)` runs, in order and stopping at the first failure (`ready: false` with `why`): build the stage graph; build the targets (a missing SVDD centre fails here); attach the hooks; build the network bytes for both weights modes and check the pinned digest (`"the network the service reads is not the one packed"`); explain every sample of the example (up to 16) at `detail: "summary"` and require `fidelity.ok`; measure the recompute (`recompute_ok: false` is a warning, not a failure). A failure never blocks readiness: it is logged and reported.

```json
{ "enabled": true, "ready": true, "why": null, "weights": "full", "network_digest": "sha256:…", "pinned": true,
  "coverage": { "nodes": 9, "types": ["dense", "head"], "opaque": [] },
  "fidelity": { "samples": 16, "max_abs_diff": 0.0, "tolerance": 1e-05 },
  "recompute": { "worst": 0.0, "ok": true } }
```

That object is `/health.inspection` and `ServiceStatus.inspection`.

### 6.3 The facade

```python
class NetworkInspector:
    def __init__(self, model, spec, answers, output, tolerance)   # answers: the Predictor (prepare, predict)
    def open(self, example) -> InspectionStatus
    def network(self, weights="full") -> dict
    def network_bytes(self, weights="full") -> bytes
    def tensor(self, tensor_id) -> dict
    def explain(self, query) -> dict
    def attribute(self, rows, targets=None, method="grad_x_input", steps=32, history=None) -> dict
    def status(self) -> dict
    def close(self) -> None                                        # removes the hooks
```

`answers` is the Predictor itself, held through a narrow protocol (`prepare`, `predict`): the inspector explains through the Predictor's one raw → batch path and reads the served answer from it, so an explanation's `prediction` is the answer `/v1/predict` gives the same rows. `tolerance` is the round trip's (`MANIFEST.roundtrip.tolerance.value`, positive). Every method but `open`, `status` and `close` needs an opened, ready inspector and otherwise raises `InspectionUnavailable` (404 at the route); `attribute` checks each target selector as a request's would (`query.QueryError`, naming it and its row; an integer class selector is refused — classes are named) and answers in chunks of at most 256 samples.

`Predictor.inspector()` builds it once (a bundle packed with `--no-inspect`, whose `inspection` package is absent, raises `LookupError` naming that; a partial render raises as it is), and the bundle's `predict.py` exposes `load_inspector(bundle_dir=None)`, which opens it on the bundle's own example rows — the library path `/vega-show` uses in every serving mode.

## 7. Limits and gating

| limit | value | bounds |
|---|---|---|
| `inline_tensor_max` | 262 144 | floats per tensor inlined in `/v1/network` |
| `inline_total_max` | 1 048 576 | floats inlined in `/v1/network` overall |
| `json_list_max` | 1 024 | a tensor this small travels as a JSON list |
| `vector_max` | 4 096 | a captured vector longer than this is summarised |
| `heat_max` | 64 | heat maps are block-reduced to at most 64 × 64 |
| `graph_nodes_max` | 256 | nodes reported around an explained node |
| `explain_float_budget` | 262 144 | numbers in one `full` explanation |
| `explain_exact_float_budget` | 4 194 304 | numbers in one `exact` explanation |
| `explain_rows_max` | 4 096 | raw rows (window) or nodes (graph) per request |
| `explain_edges_max` | 65 536 | edges of one explained graph |
| `ig_steps_default` | 32 | integrated-gradient steps by default |
| `ig_steps_max` | 128 | integrated-gradient steps at most |
| `ig_batch_max` | 65 536 | interpolants × nodes in one integrated-gradient batch |

A switch can only restrict what the bundle allows, never widen it:

| source | values | effect |
|---|---|---|
| `MANIFEST.serve.inspection.enabled` (pack `--no-inspect`) | true / false | false: the three routes answer 404 "this bundle ships without inspection" |
| `MANIFEST.serve.inspection.weights` (pack `--inspect-weights`) | `full` (default) / `summary` | summary: tensors `withheld`, the tensor route 404 |
| `<ENV>_INSPECT` | `on` / `off` | off: 404 "inspection is switched off on this service" |
| `<ENV>_INSPECT_WEIGHTS` | `full` / `summary` | summary as above; `full` cannot override a bundle's `summary` |

Publishing raw weights is a model-exfiltration surface, and the bundle contract says so. Pack records `MANIFEST.serve.inspection = {schema: "vega/inspection@1", enabled, weights, network_schema, explanation_schema, files, block_types, opaque, chain, network_digest: {full, summary}, self_test, source}`.

## 8. The showcase slot

The slot serves a verified, separately pinned `showcase/` directory on the service's origin. Its route module is `serve/routes_showcase.py` (`CAPABILITY = "showcase"`). The directory is `--showcase DIR` (which sets `<ENV>_SHOWCASE`), else `<ENV>_SHOWCASE`, else `<bundle root>/showcase`. Its manifest is `SHOWCASE.json`, schema `vega/showcase@1`, which `/vega-show` owns and writes (`vega-show/references/schemas/showcase-1.schema.json`); the slot reads `schema`, `bundle.manifest_digest`, `bundle.model_version`, `entry`, `story.file`, `report.pdf.file` and `digest`, `files` and `served`. The bundle's `MANIFEST.json` never lists the slot and is never written by it.

**Opening.** `routes_showcase.register` opens and verifies the slot synchronously, before the model is ready, and sets the page override; a refused slot never makes the model unready. In order: no directory or no `SHOWCASE.json` is an absent slot (the normal state); the schema must be `vega/showcase@1`; `bundle.manifest_digest` must equal the sha256 of the bundle's `MANIFEST.json` bytes and `bundle.model_version` its `model_version` ("the showcase was made for another bundle"); a manifest digest that equals some `MANIFEST.reemits[].previous_manifest_digest` is stale ("the bundle was re-emitted after this showcase; re-run /vega-show"); then every file of `files` must match the path grammar, resolve inside the directory without a symlink, have a `MEDIA` extension and its digest, within 64 MiB in total. Only `served ⊆ files` is served, from memory; a file altered after start-up is never served.

**Closed world.** The slot refuses a showcase holding a symlink, a path outside the directory, more than 64 MiB, or any file with a `MEDIA` extension that is neither in `files` nor under `.work/`, `.smoke/` or `docs/`. `MEDIA` covers `.html`, `.json`, `.js`, `.css`, `.svg`, `.png`, `.jpg`, `.webp`, `.woff2`, `.pdf`, `.txt` and `.md` (`text/markdown; charset=utf-8`).

| route | slot verified | otherwise |
|---|---|---|
| `GET /` | the entry | the console page (`serve/static/index.html`) |
| `GET /console` | the console page | the console page |
| `GET /showcase/{path}` | a served file | 404 problem |
| `GET /v1/story` | the story file | 404 problem |
| `GET /report.pdf` | the linked report | 404 problem |
| `GET /v1/showcase` | the slot status | the slot status |

`OPEN_WHILE_UNREADY = ("/", "/console", "/showcase/", "/v1/showcase")`: the page loads its own modules to show that the model is not ready. Every slot response carries `ETag` (the file's digest; 304 on `If-None-Match`), `Cache-Control: no-cache` and `X-Content-Type-Options: nosniff`; HTML also carries `Content-Security-Policy: default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'`.

**Status.** `GET /v1/showcase` and `/health.showcase` answer the same object:

```json
{ "present": true, "verified": true, "reason": null, "digest": "sha256:<SHOWCASE.json bytes>",
  "entry": "page/index.html", "template": { "version": "2.3.0", "manifest_digest": "sha256:…" },
  "report_status": "linked", "created_at": "…", "files": 212, "unlisted": [] }
```

A refused slot is `verified: false` with the `reason`. Pack records `MANIFEST.serve.showcase_slot = {dir: "showcase", schema: "vega/showcase@1", routes, env: "<ENV>_SHOWCASE", flag: "--showcase"}` whenever it renders `serve/`, and the bundle's `.dockerignore` keeps `showcase/` while excluding `showcase.prev/`, `showcase.building/`, `showcase/.smoke/` and `showcase/.work/`.

## 9. Tests

| test | proves |
|---|---|
| `scripts/tests/test_catalog_sync.py` | §3's and §4's tables equal `catalog.json` `inspection.views` and `inspection.targets` (whose keys equal the catalog's blocks plus the five synthetic types, and its task kinds); the eight links; `coverage.json` `inspection.blocks ⊇ blocks` with a surface for each; both schema files are valid JSON Schema draft 2020-12 |
| `test_inspection_surface.py` | every block class exposes `coverage.json` `inspection.surface[type]` (the cumulative head its variant) |
| `test_inspect_codec.py` | the base64 round trip; 9-digit JSON lists round-trip float32; the encoding thresholds; deterministic summaries; the budgets, including `degraded` |
| `test_inspect_graph.py` | on one rendered package per family: the network validates against the schema; every spec block id is a node; no `opaque` node but `custom`; `params.total == param_count()[0]`; `chain` exactly for eligible specs; top-level `out_shape == Network.shapes` |
| `test_inspect_probes.py`, `test_inspect_concurrency.py` | the recompute checks of §3.1; no `.grad` and no per-request hook after any explanation; concurrent explanations and predictions byte-identical to sequential ones |
| `test_inspect_targets.py`, `test_inspect_attribution.py` | §4 per effective kind; request validation; attribution per shape and integrated-gradient completeness |
| `test_inspect_zoo.py` | every family × admitted task × shape × level of `family-catalog.md` §7: network and explanation validate, fidelity (the zoo holds the captured head to `1e-6·max(1, |head|)`, scaled like the round trip, never an absolute 1e-6), recompute, targets, attribution; integrated-gradient completeness `gap ≤ 1e-2·max(1, |s(x) − s(x0)|)` at 64 steps on the cases whose path 64 midpoints resolve, and on the known exceptions (untrained `ft-transformer`, `mlp-residual` with an inner layernorm, `transformer-seq`, `sage_conv` with `normalize: true`) the gap reported in the explanation, never hidden |
| the bundle's `tests/test_api_inspect.py`, `tests/test_api_showcase.py` | the routes of §6 and §8, the switches of §7, the slot's closed world and staleness |
