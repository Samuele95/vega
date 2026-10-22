# dataset-shaper step catalog

The vocabulary of `recipe.steps[].op`. Each entry: what the step does, its `params`, its phase (see `shape-contract.md` §1), whether it is *fitted* (fit on train when a split exists), and the decision that usually sources it. A `/data-lens` transformation candidate must use one of these names; a step outside this catalog is a `custom` op with an inline `code` field, executed in a sandboxed cell and flagged in the layer as unverified.

## Phase 1 — typing and structure

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `retype` | `to: numeric \| nominal \| ordinal \| boolean \| datetime \| text`, `order: [...]` for ordinal | no | `geometry:typing/<col>` | the final typing the geometry layer recorded; never re-derived here |
| `drop_identity` | — | no | `geometry:typing/<col>` (role identity / key) | keys leave before any dependency-sensitive step |
| `drop_constant` | `tolerance: 0.0` (share of the dominant value above which a column is quasi-constant) | no | `geometry:typing/<col>` or `analysis:F<n>` | |
| `dedupe` | `on: "all" \| [columns]`, `keep: first \| last`, `near: {cols, tolerance}` optional | no | `analysis:F<n>` | reports removed rows in the manifest |
| `parse_datetime` | `format`, `tz`, `errors: coerce \| raise` | no | `analysis:context.time` | coerced failures become missing and are counted |
| `parse_geometry` | `kind: latlon \| wkt \| geohash \| h3`, `lat`, `lon`, `crs` | no | `analysis:context.spatial` | requires a declared `crs`; without one the step is refused (`FORK` or `WARN`+skip unattended) |

## Phase 2 — geometry

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `orient_cycle` | `cycle_id`, `orientation_id` | no | `geometry:cycles/<id>` | selects which member of a derivation cycle is derived; default = the layer's `default: true` orientation |
| `drop_derived` | `orientation: {cycle → orientation}` optional | no | `geometry:derivations` | drops every active rule head — the head of a **kept** derivation (dataset-forge report-contract §2: semantic `confirmed`, not `kept: false`) or of the chosen orientation's rules; a rejected derivation derives nothing, its head is a basis member and stays (F44); `keep: [...]` lists heads a `user:` step retains (documented as redundant-by-choice) |
| `select_partition` | `label`, `task` | no | `geometry:partitions/<label>` | names the label column; refused if the label is in the features later |
| `drop_leakage` | — | no | `geometry:partitions/<label>.dropped_for_leakage` | drops the leakage set of the selected partition |
| `keep_columns` | `columns: [...]` | no | `user:` | explicit projection to a named column set; must include the basis unless a `user:` rationale says otherwise |

## Phase 3 — split

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `split` | `kind: random \| stratified \| group \| time`, `by`, `test`, `valid`, `time_column`, `cutoff` | no | `shaper:default` (stratified on a class label — the executor refuses a stratified split when more than half its strata hold fewer rows than it takes to place one in the smaller held-out part (`⌈1 / min(test, valid)⌉`), as every stratum of a continuous column does, so a regression label splits `random` —, time when a datetime is the natural order) or `analysis:F<n>` (a time-series finding forces `time`) | at most one; everything fitted afterwards is fitted on train; a `group` split drops its key (`by`) from the frame — every valid and test group is unseen, so the key is never a feature — unless it is the level-graph `graph_id` (kept, `role: key`); a later step that reads the key is refused (`shape-contract.md` §1, F33) |

## Phase 4 — values

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `impute` | `strategy: median \| mean \| mode \| constant \| group-median \| group-mode \| knn \| iterative`, `by`, `value`, `indicator: bool` | yes | `analysis:F<n>/T<m>` | `indicator` adds `<col>_missing`; the label is never imputed (refused), and an impute grouped `by` the label is refused whatever its source (it carries the label into the row's features — `shape-contract.md` §1) |
| `clip` | `lower`, `upper` (absolute) | no | `analysis:F<n>` (unit / range anomaly) | |
| `winsorize` | `lower_q`, `upper_q` | yes | `analysis:F<n>` (outliers judged errors) | refused on a column a `stance` declares genuine |
| `transform` | `kind: log1p \| signed-log1p \| log \| sqrt \| box-cox \| yeo-johnson \| reciprocal` (`signed-log1p` = sign(x)·log1p(\|x\|), for heavy tails on columns that also hold negatives, and for zero-IQR columns a robust `scale` would pass through unscaled) | yes (box-cox / yeo-johnson λ, `fit_on: "train"`); the other kinds are closed-form and record `fit_on: null` | `analysis:F<n>` (skew, heteroscedasticity) | applied to every member of a kept derivation together or refused (a log on `qty` alone breaks `total = price × qty`) |
| `bin` | `kind: quantile \| width \| custom`, `k`, `edges` | yes for `quantile` and `width` (the edges are the fit part's quantiles or its min–max range cut in `k`, `fit_on: "train"`); `custom` edges are given | `analysis:F<n>` or `user:` | keeps the numeric column unless `replace: true` |
| `datetime_expand` | `parts: [year, month, dow, hour, is_weekend, ...]`, `cyclic: bool` | no | `analysis:F<n>` (time_series) | cyclic encodes month/hour as sin/cos pairs |
| `lag` | `columns`, `lags: [1, 7]`, `group_by`, `time_column` | no | `analysis:F<n>` (autocorrelation) | only after a `time` split or without split; refused after a random split (future leakage); a lag grouped by the label (`group_by` naming it) is refused whatever its source (it carries the label into the row's features — `shape-contract.md` §1) |

### Spatial (phase 4, require `parse_geometry` first)

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `spatial_reproject` | `to_crs` | no | `user:` or `shaper:default` (a metric CRS chosen from the data's UTM zone for distance steps) | |
| `spatial_distance` | `to: {point \| layer path}`, `name` | no | `analysis:F<n>` (spatial autocorrelation) | metres, computed in a metric CRS; a `layer path` only when the user supplies it, its digest recorded in the manifest as for `spatial_join` |
| `spatial_join` | `layer: <path>`, `how`, `predicate: within \| intersects \| nearest`, `columns` | no | `user:` | external layer digest recorded in the manifest |
| `spatial_grid` | `kind: h3 \| square`, `resolution \| size_m`, `aggregate: {col: fn}` | the metric projection's origin and, with `aggregate`, the per-cell aggregates: fitted on the train part and joined to every part by cell id (a test row never enters a cell's statistic) | `analysis:F<n>` or `user:` | adds the cell id; with `aggregate`, adds per-cell aggregates joined back |
| `spatial_features` | `features: [area, perimeter, centroid_x, centroid_y, bearing_to, n_neighbours_within, nearest_distance]`, `radius_m`, `to: {lat, lon}` (for `bearing_to`) | yes for `n_neighbours_within` and `nearest_distance` (the neighbour index, fitted on train, `fit_on: "train"` — `shape-contract.md` §2: every part and every served row is answered from the train rows, the row's own location excluded); no for the per-geometry features | `analysis:F<n>` | `area` and `perimeter` need polygon geometry: on the points `parse_geometry kind: latlon` declares, the executor refuses them; a feature it does not know is refused, never skipped; `bearing_to` is the initial great-circle bearing to `to`, degrees clockwise from north |

## Phase 5 — representation

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `encode` | `strategy: one-hot \| ordinal \| target \| frequency \| hashing`, `min_frequency`, `handle_unknown`, `order` | yes | `analysis:F<n>` (cardinality) or `shaper:default` | `target` only after `split`, fitted out-of-fold; one-hot on > 50 levels warns and suggests `frequency`/`target` |
| `scale` | `kind: standard \| robust \| minmax \| none`, `columns: "numeric" \| [...]` | yes | `shaper:default` (robust when outliers were judged genuine) | recorded statistics go to the manifest |
| `project` | `kind: pca`, `k \| variance: 0.95`, `columns`, `keep_named: bool` | yes | `geometry:space` (dims_95) or `user:` | loadings in the manifest; `keep_named: true` keeps the original basis columns beside the components — the honest default, because a rotated basis has no names |

## Phase 6 — selection

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `select_features` | `kind: importance \| variance \| explicit`, `threshold`, `columns` (explicit), `from: analysis:importance` (the analysis layer's permutation importance, `data-lens/references/analysis-contract.md` §1) | `variance` only: its variances are the train part's (`fit_on: "train"`; `"all"` without a split); `explicit` drops the named columns and fits nothing; `importance` reads the analysis layer's scores, measured **before the split** (the manifest records `source_of_scores`) | `variance`, `explicit`: `shaper:default`, `analysis:F<n>` or `user:` | `importance` (the default `kind`) is refused in a recipe that splits — by the recipe checker, the executor and the verifier (`shape-contract.md` §1): the analysis measured it before the split, over rows that are now the test part; after a split, select by `variance` or name the columns (`explicit`); in a recipe without a split there is no test part to inform it, and it takes the analysis's scores as they are; a column in the basis is never dropped by this step, nor the label |

## Phase 7 — graph

The last phase, and the only one that adds an output rather than changing the frame: the `edges` part (`shape-contract.md` §2). It sits after `split` because its fitted object — the reference index of train nodes — is a train-only estimate like every other fitted parameter, so "no fitted step before split" stays true by construction; and it sits after representation and selection because a kNN must measure distance in the columns the network will see (`scale` and `project` have run). A `graph_edges` step requires a split (the parts are what its edges record) and is refused without one, before it, or with a source that is not `analysis:` or `user:` (`shaper:default` and `geometry:` included). A recipe carries at most one: the edges part is one relation, recorded once. A step with a `graph_id` needs a `group` split by that column, or it is refused: a graph whose rows straddle parts would be trained on and tested on at once.

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `graph_edges` | `method: key_pair \| knn \| spatial` (per-method params below); common: `symmetric: bool` (default `true`), `self_loops: bool` (default `false`), `weight: <column> \| "distance" \| null`, `graph_id: <column> \| null`, `max_degree` (a key_pair cap needs `graph_id`) | `knn`, `spatial` at level node: yes (the reference index of train nodes); at level graph (`graph_id` set): no, computed within each graph; `key_pair`: no | `analysis:F<n>/T<m>` (a `relations`, `segments` or `spatial` finding that rows are related) or `user:`; never `shaper:default`, `geometry:` or any other | rows unchanged, columns unchanged; writes the `edges` part; `downstream`: "enables the graph network families in /vega-forge; the protocol (inductive/transductive) is the network stage's decision, this step only records which edges cross parts" |

Per method:

- `key_pair` — `columns: [src_col, dst_col]`: row *i* → row *j* when `i[dst_col] == j[src_col]` (a foreign key: the bus whose `feeder_to` is another bus's `bus_key`). With one column, rows sharing a value form a clique (a shared account or session id), capped by `max_clique` (≥ 2) at level graph: a group larger than that links each row to its `max_clique − 1` nearest group members by position, deterministically, and the manifest counts the capped groups. The columns are read from the input frame at the rows' raw positions, so a key `drop_identity` removed in phase 1 still relates rows and never re-enters the features; keys compare as text (a key read as `80.0` and one read as `80` are the same key); a missing key relates nothing. Not fitted: the edges are a fact of the keys, the same whichever part a row is in. **A cap needs a `graph_id`.** A key_pair cap (`max_clique`, `max_degree`) ranks a row's candidates by position; at level node that is the raw position, which a served row does not carry, so `edges_for` could not cap a query's attachments as the edges part did and the evaluation and the bundle would see different graphs: the recipe checker refuses a key_pair cap without a `graph_id`. At level graph the rank is the row's place within its graph, which a whole-graph query reproduces exactly.
- `knn` — `k`, `metric: euclidean \| cosine`, `features: [...] \| "basis"`. `"basis"` is the basis members as they stand at this step — imputed, scaled, projected, selected (their one-hot columns, or the projection's components when `project` replaced them): the fitted space the network sees; without a geometry layer it resolves to every numeric column at phase 7, the label and the graph key excluded, with a `WARN`. That is the ordering rule: `graph_edges` follows `scale` and `project` by phase, so a distance is measured on unscaled columns only when the recipe has no `scale`; explicit `features` are read as they stand too, and a missing value in them is refused (impute first). Every node's `k` nearest **train** nodes, a train node's own row excluded; fitted on train (`fit_on: "train"`), so a valid or test node reaches only into train and two non-train nodes are never joined. The fitted space is the features **as float32** — the dtype the network reads and `/vega-pack`'s reference nodes store (`vega/graph-reference@1`) — held in float64 for the sums, in every program, so a float32 reference reproduces the train matrix bit for bit. A tie goes to the train node that comes first in the train part; distances are accumulated one feature at a time, so a pair's distance depends on those two rows only and a served row gets the value the part records.
- `spatial` — `k \| radius_m` (exactly one). Reads the raw coordinates `parse_geometry` names (`lat`, `lon`) at the rows' raw positions and projects them into the metric CRS `spatial_reproject` recorded — the geometry as those two steps left it, not columns a later `scale` standardised — and is refused without both steps before it, as a radius in degrees is not a distance. Neighbours are train nodes, fitted and tie-broken as `knn`, the distance in metres.

**Level graph (a contract correction).** With `graph_id` set, `knn` and `spatial` are computed **within each graph and are not fitted** (`fitted: false`): each node's `k` nearest nodes of its own graph, whatever part the graph is in. Test graphs are disjoint from train graphs under the required group split, so a "k nearest train nodes" rule cannot hold inside a test graph. The graph key must reach the shaped parts (it is how a graph-level network groups its nodes: `role: key` in `output_schema`), so a recipe that drops it before phase 7 is refused.

Common params: `symmetric: true` writes every edge in both directions (two rows, the reverse carrying the edge's weight; `directed: false` in the manifest's `graph` block), `false` keeps the direction the method produced (`directed: true`); each directed pair is written once, the method's own edge winning over a reverse; `self_loops` defaults to `false` because the conv blocks add their own (`gcn_conv.self_loops`), and `true` writes one loop per node (weight 0 under `"distance"`, the row's own column, or null); `weight` is a column of the source row read from the input at its raw position, `"distance"` (the metric distance, metres for `spatial`; `knn` and `spatial` only) or `null`; `graph_id` joins only rows sharing the value and sets `level: graph` in the manifest (a graph-level task, one label per graph, `graph_pool` in the network), `null` is `level: node`; `max_degree` caps the out-degree of the method's own edges per node, before `symmetric` adds the reverses, keeping the smallest distances (`knn`, `spatial`) or the smallest position gaps (`key_pair`) — deterministic, no seed. `reversible: true` (delete the part; the frame is untouched).

## Custom

| op | params | fitted | usual source | notes |
|---|---|---|---|---|
| `custom` | `code` (a pandas function `def step(df, ctx) -> df`), `description`, `phase` (1–6, required) | no — it runs per part on that part alone, so it can fit nothing across parts | `user:` only — the executor refuses any other source (`shape-contract.md` §1) | placed by the executor in its declared phase; after the split it runs per split part on that part alone (no cross-part fitting possible); before the split it must be row-local — the executor probes it on slices of the frame and refuses one that computes over rows or renumbers them, and the recipe checker refuses one declaring `fitted: true` (`shape-contract.md` §1); the layer marks it *unverified*; the reproduction script inlines it |

## Presets (`--goal`)

`train-ready` = phases 1–6 with defaults where the layers are silent; `clean` = phases 1 and 4 only, no split, no encoding; `basis-only` = phases 1–2 (the basis and the label, nothing fitted); `spatial-features` = phases 1, 2 and the spatial steps; `custom` = only what the layers and the user name. A preset names the phases whose `shaper:default` steps it adds; the layer-sourced steps of phase 2 — `orient_cycle`, `drop_derived` for the active heads, `select_partition` and `drop_leakage` with a partition — are compiled under every preset, `clean` included, because they come from the geometry layer, not from the preset. A preset adds steps with `source: shaper:default`, each with a rationale and an alternative, and every one is shown to the user before execution. No preset includes `graph_edges` (phase 7): that rows are related is a claim only an `analysis:` finding or a `user:` turn can make, so `train-ready` never adds it on its own.
