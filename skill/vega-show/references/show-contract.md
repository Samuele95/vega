# vega-show contract — the facts, the story, the showcase slot, the report binding, the page, the `showcase` layer

`/vega-show` shows the shipped network to people who were not in the chain. It reads the model page `<stem>.pack.html`, the bundle it names and, when one exists, the compiled report; it writes **one directory**, `<bundle>/showcase/`, which the bundle's own service serves through its showcase slot, and **one layer**, `showcase`, on `<stem>.pack.show.html`. The rule it never bends: **the page says nothing the chain did not measure, and touches nothing the chain shipped.** Every number the page shows is a fact key a script extracted from the bundle's pinned records or from the report's own labels; the model writes the words under a digit rule; the bundle's `MANIFEST.json` is read, verified and pinned, never written.

`scripts/catalog.json` (`vega/show-catalog@1`) is the machine copy of this contract, and `scripts/tests/test_catalog_sync.py` (with its helpers `_sync_common.py`, `_sync_contract.py`, `_sync_catalogs.py` and `_sync_schemas.py`, each within R-G1's size limits) fails when the two diverge, when either disagrees with the vega-forge catalog, the vega-eval catalog or the backend's `METRICS`, and when a schema under `references/schemas/` does not validate. The schemas are normative for the shapes; the prose is normative for the rules. Closed sets are written in tables whose second column names the catalog key they are copied to.

## 1. The stage

Stage `show`, skill `vega-show`, layer `showcase`, stem tag `.show` (`….eval.pack.html` becomes `….eval.pack.show.html`). The run directory is `<bundle>/showcase/`; the stage brief is `<bundle>/showcase/docs/show-brief.md` and is **post-report** (§9). The stage runs after `pack`, and after `document` when the lane documents:

```
… ──► eval ──► pack ──► document ──► show          (objective carries +document)
… ──► eval ──► pack ──► show                         (no report: code notes link to the stage briefs, WARN)
document ──(report digests)──► show                  a new report digest makes the showcase stale
```

The report pins nothing of the showcase and the showcase pins the report, so the dependency runs one way: pack → document → show. `document` never reads `showcase/`, and the `show` brief is never a chapter.

**Inputs.** The model page with the `geometry`, `shape`, `architecture`, `training`, `evaluation` and `bundle` layers (`analysis` optional); the bundle directory next to it (`bundle/<slug>-<run>/`, or `--bundle`); the report (`--report`, default `docs/label-roles.json` beside the model or in the cwd, else `docs/*.pdf` with a same-stem `.aux`). **Outputs.** `<bundle>/showcase/` (§5), the layer (§8), the brief (§9). The bundle's `MANIFEST.json` and every file it pins are never written; the stage writes only under `showcase/`, `showcase.building/` and `showcase.prev/` in the bundle directory.

| closed set | catalog key | members |
|---|---|---|
| required layers | `stage.required_layers` | `geometry` `shape` `architecture` `training` `evaluation` `bundle` |
| markers | `markers` | `OK:` `WARN:` `ERROR:` `FORK:` |

**Exit codes.** Every script prints `OK:` / `WARN:` / `ERROR:` before prose, and `FORK:` when it stops to ask; `--json` gives a machine result. Exit 0 is OK, 2 is refused or invalid input, 3 is stale (check only) or the domain-forge layer writer missing (apply only).

### 1.1 Forks and their unattended answers

Every fork has its unattended answer (`check_unattended.py` enforces the pairing over `SKILL.md`). There is no `GATE:`: nothing in this stage is irreversible, because the previous showcase is kept as `showcase.prev/`.

| fork | name | when | attended | `--unattended` |
|---|---|---|---|---|
| `S1` | `report` | no compiled report was found (`document` did not run); a report without a label-role map takes proposed roles instead (§6.5) | `FORK:` continue with brief links, or stop and run `/document-project vega <lane>` first | brief links, with `WARN: no report — code notes link to the stage briefs; re-run /vega-show after /document-project` |
| `S2` | `roles` | the draft proposes an `alert` role, or `domain:` / `spec:` evidence raised one, and no `user:` turn or `show:` stance settles it | `FORK:` per-class roles, the proposal first | an unsupported `alert` becomes `plain`, with `WARN:`; a structural `neutral` is kept; the anomalous side under `task:anomaly` is never downgraded |
| `S3` | `motif` | the modality admits more than one motif and no `show: motif` stance exists | `FORK:` motif, the modality's base first, the admissible variants each with a one-line reason | the base |

A recorded stance (`show:` lines in `.claude/domain-forge-memory.md` § Network stances, §9) is applied and cited, never re-asked.

### 1.2 What is not a fork

- **A label-role map older than its `.aux`** is `ERROR:` exit 2 ("the role map is older than the report's .aux — recompile /document-project"), gate `PF6`.
- **A bundle without a service or without `MANIFEST.serve.showcase_slot`** is `ERROR:` naming `--serve` (gate `PF4`); the lane stops.
- **An unearned verdict** is shown as recorded, with `WARN:` (the resolver prints `WARN: verdict not earned …` — `— beaten by <baseline>` when a baseline did better — into the showcase's markers, which step 8 carries into the layer); the page carries the evaluation's verdict state and its mandatory limits, never a softer or stronger word: the results view says it in every state — earned, not earned, beaten by a baseline, not assessable — with the strongest baseline and the paired difference and its interval, and the start view puts its word beside the primary.
- **A verdict that is not assessable** (vega-eval eval-contract §5, A30: an unlabelled anomaly task, whose primary lives on each model's own score scale) is shown as recorded: `earned` null with the evaluation's `reason`. The page says *not assessable* in words, with that reason, and the mandatory limit `verdict_not_assessable`; it never draws it as a win or a loss (no baseline, no difference, no "lower is better" across the models' scales). A null `earned` without a reason is no verdict and is refused (`resolve`).

## 2. The fact set — `vega/show-facts@1` (`plan/facts.json`)

`extract_facts.py` (class `FactExtractor`) writes the fact set; `schemas/show-facts-1.schema.json` is its shape.

**Sources.** Facts are read only from these files of the bundle, each verified against `MANIFEST.files` before it is read (gate `PF2`), and from the report's labels through a role (§6):

| closed set | catalog key | members |
|---|---|---|
| fact sources | `fact_sources` | `MANIFEST.json` `netspec.json` `schema.json` `records/geometry.json` `records/analysis.json` `records/shape.json` `records/architecture.json` `records/training.json` `records/evaluation.json` `report:` |

`metrics.json` is read as the evaluation record's `metrics` object (`records/evaluation.json#metrics`), which the evaluation layer splices from `metrics.json` verbatim; `metrics.json.ci_level` is therefore `records/evaluation.json#metrics.ci_level`. The model HTML is read only for the `bundle` layer (to locate and compare the bundle), its Turtle digest and the prose sources.

**Ref grammar.** File-explicit, the grammar of v2.2 pack's `_story_lookup` moved here:

```
ref   := file "#" path   |   "report:" role "." ("number" | "page" | "title")
file  := "MANIFEST.json" | "netspec.json" | "schema.json" | "records/<layer>.json"
path  := seg ("." seg)*
seg   := name | name "[" key "=" value "]" | integer
e.g.  records/shape.json#manifest.steps[id=S4].rows_after.all
```

**Key grammar.** A fact key matches `^[a-z][a-z0-9_]{0,63}$` and is deterministic: ids are lower-cased with non-alphanumerics mapped to `_` (`H1-5` becomes `h1_5`). **Derivations.** One is allowed, `count_of`: the length of a list a ref points at (`classes_n`, `runs_n`). No other number is computed; differences drawn as bars are computed by the page for drawing and never cited in prose.

| closed set | catalog key | members |
|---|---|---|
| fact kinds | `fact_kinds` | `count` `size` `metric` `ci_metric` `diff` `param` `epoch` `list` `labels` `count_of` `report_number` `report_page` `text` |
| formats | `formats` | `count` `metric` `ci` `param` `epoch` `verbatim` `list` |
| derivations | `derivations` | `count_of` |
| fact groups | `fact_groups` | `sizes` `recipe` `split` `sample` `inputs` `class_balance` `network` `decisions` `runs` `selection` `hpo` `models` `metrics` `diffs` `breakdown` `verdict` `limits` `identity` `report` `briefs` `prose_sources` |

`direction` is present on metric facts only, and it is read from the **vega-forge catalog's `metric_direction`**; vega-show keeps no copy of that table.

### 2.1 Names

`names` is the whitelist the digit rule uses (§3.3): the union of the column names (raw and shaped), the class names, the tokens of the dataset file stem, the tokens of `MANIFEST.serve.identity.title`, the run ids and block ids, the metric names and family names, and the catalog's name tokens. A lane that needs another token adds it by a reviewed catalog edit, never per run.

| closed set | catalog key | members |
|---|---|---|
| name tokens | `name_tokens` | `F1` `L1` `L2` `ReLU6` `top5` `IPv4` `IPv6` `P2P` `3D` `2D` `1D` |

### 2.2 The reserved values

Two facts are bound by the extractor to the evaluation records and reach the story as reserved values (§4.1):

- **`primary`**: `eval_network_<primary>`, `{value, ci, n}`, read from `records/evaluation.json#metrics.models.network.<primary>` — the network's primary metric on the sealed test;
- **`ci_level`**: `eval_ci_level`, read from `records/evaluation.json#metrics.ci_level` — the bootstrap level the evaluation used. A bundle whose evaluation records no `ci_level` (pre-v2.3) gets no such fact and `WARN:`, and the page prints the interval without a level.

### 2.3 Groups

`groups` carries structured views. Every number in them is a fact key (a string); ids, names and ops are inline strings.

| group | fact keys created | applies to |
|---|---|---|
| `sizes` | `source_rows`, `source_columns`, `shaped_inputs`, `train_rows`, `valid_rows`, `test_rows`, `classes_n` | all |
| `recipe` | `step_<id>_rows_before[_<part>]`, `step_<id>_rows_after[_<part>]`, `step_<id>_columns_removed_n` | all |
| `split` | none (reuses `*_rows`) | all |
| `sample` | `window_length`, `window_stride`, `horizon`, `edges_n`, `graph_edges_k` | window, graph |
| `inputs` | none (`shaped` in `netspec.data_contract.features[].name` order, `origin`, `typing`, `created_by_recipe`, `encoded`, `relations`) | all |
| `class_balance` | `train_share_<class>` | class tasks |
| `network` | `params_total`, `madds_per_sample` | all |
| `decisions` | `block_<id>_<param>`, `loss_<param>`, `optimizer_<param>`, `schedule_<param>`, `training_<param>`, `early_stopping_<param>`, `regularization_<kind>_value` | all |
| `runs` | `run_<id>_primary`, `run_<id>_epochs`, `run_<id>_best_epoch`, `run_<id>_delta_<k>_from`, `run_<id>_delta_<k>_to`, `runs_n`, `runs_pruned_n` | all |
| `selection` | `selected_primary`, `selection_margin` | all |
| `hpo` | `hpo_<study>_trials`, `hpo_<study>_best_value` | when run |
| `models` | `eval_<model>_<metric>` | all |
| `metrics` | `eval_n_test`, `seal_exposures`, `threshold_value`, `eval_ci_level` | all |
| `diffs` | `eval_diff_<baseline>` | all |
| `breakdown` | `eval_<model>_<metric>_per_<unit>`, `eval_network_<metric>_per_<unit>_weak` | the tasks of §4.5 with a breakdown |
| `verdict` | `eval_verdict_reason` (a verdict that is not assessable; otherwise none, reuses `eval_diff_*`) | all |
| `limits` | none | all |
| `identity` | none (`MANIFEST.serve.identity` verbatim) | all |
| `report` | `report_<role>_number`, `report_<role>_page` (the role's dots become `_`) | with a report |
| `briefs` | none | always |
| `prose_sources` | none (texts the model reads, never copies) | all |

`features_tested` in `decisions` is generic: a run's delta that introduces, in `to`, string values equal to shaped feature names absent from `from` names those features (the ablation that removed a feature).

### 2.4 Limit kinds

A mandatory limit must be covered by an authored limit (rule `V9`).

| limit | when | mandatory |
|---|---|---|
| `split` | always (the split step's kind) | always |
| `exposures` | always (`seal.exposures`) | always |
| `one_seed` | always | always |
| `robustness_not_run` | `evaluation.robustness` records no ran checks | when present |
| `verdict_not_earned` | `verdict.earned` is false | when present |
| `verdict_beaten` | `verdict.beaten_by_baseline` is true | when present |
| `verdict_not_assessable` | `verdict.earned` is null with its `reason` (A30) | when present |
| `ship_stance` | `MANIFEST.stances` is non-empty | when present |
| `roundtrip_boundary` | `MANIFEST.roundtrip.boundary` is present (lag restarts at a part start, R-A6) | when present |
| `examples_synthetic` | `MANIFEST.examples.kind` is `synthetic` | never |
| `calibration` | `ece` is in the task's metric set | never |
| `slices_empty` | `evaluation.slices` is empty | never |
| `drift_suspected` | the analysis drift module lists drifted columns | never |
| `serving_forced` | `MANIFEST.serve.forced_by` is `--serve` (a batch or stream blueprint served for the demonstration) | never |
| `small_support` | a class of the task has fewer sealed test rows than the catalog's `small_support.below` (`metrics.class_support`): its per-class reading is an anecdote | when present |
| `warn` | each `WARN:` marker of the evaluation and bundle layers, as `warn_<stage>_<n>` | never |

A limit whose record holds words of its own carries them as `recorded` — a `warn_*` limit its `WARN:` line, `ship_stance` the `MANIFEST.stances` text, `small_support` the classes and their counts; verbatim, except a path of the machine's file system, cut to `…/<file name>` (`facts_results.local_paths_cut`: the directories the chain ran in are no part of a warning, and a route such as `/v1/network/tensor/{tensor_id}` stays) — and the story's limit keeps them (`story@2` `limits[].recorded`), quoted by the page beside the authored title and text: the authored prose says what the limit means, the record says what happened, verbatim.

## 3. The story plan — `vega/story-plan@2` (`plan/story-plan.json`)

One file, two halves (`schemas/story-plan-2.schema.json`). `draft_plan.py` writes `proposals` and every `authored` slot as `null`, with `status: draft`; the model fills `authored` and sets `status: authored`. The plan pins the fact set it was drafted from (`facts.digest`) and the page template (`template.manifest_digest`, the digest of `assets/page/TEMPLATE.json`), both checked by `V1`.

### 3.1 `proposals` (script)

`modality` and `view_key` (§4.3); `lexicon` (each entry with its source); `motif` (`modality`, `base`, `admissible`, `stance`, `fork: S3|null`); `class_roles` (`neutral {class, rule, share}`, `alert_candidates`, `basis_found`, `memory`, `evidence`, `fork: S2|null`, or null for a task without classes); `input_groups` (`[{suggested_id, members, evidence}]`, a partition of the shaped inputs); `choices` (`[{decision, family, type, params, rationale_ref, tested_by, mandatory}]`); `runs` (`[{id, label, selected, reference, primary}]`); `results_facts` (the task's §4.5 metrics the evaluation computed); `limits`; `code_steps` (`[{step: <walk id>, role, status: bound|brief|no-role}]`); `report_roles` (`{status: emitted|proposed|absent, bindings, fork: S1|null}`); `context_steps`; `prose_sources`.

### 3.2 `authored` (the model)

| field | constraint |
|---|---|
| `locale` | a member of `locales` |
| `lexicon` | the keys of §3.6: every required key of the lane's shape, no unknown key; values follow the lexicon value pattern |
| `motif` | one of `proposals.motif.admissible` |
| `class_roles` | `{roles: {class: role}, source: {class: basis}}`: every class (or anomaly side) exactly once, §3.5 |
| `report_roles` | `{role: label}` or null; only when `proposals.report_roles.status` is `proposed`: each label exists in the `.aux` with the role's kind (`V14`) |
| `headline`, `lead` | the digit rule; lengths per §3.7 |
| `context` | `{steps: {<step id>: text}, facts: [{label, fact}]}`: the step ids are exactly `proposals.context_steps`; 2–6 facts, each a fact key |
| `network` | `{lead, choices: [{decision, title, text}]}`: known decisions, every mandatory one present, within the choice bounds |
| `experiments` | `{lead, caption, labels: {run id: label}}`: keys equal the proposed runs; run labels are authored text under the digit rule |
| `results` | `{headline, notes: {fact key: note}, breakdown_caption, limits: [{id, title, text}]}`: notes keys ⊆ `proposals.results_facts`; every mandatory limit covered |
| `input_groups` | `[{id, label, text, members}]`: the members partition the shaped inputs exactly; bounds of §3.7 |
| `code_notes` | `{walk id: note}`: keys ⊆ the proposed code steps |
| `monitor_note` | a string or null; null when `MANIFEST.monitor` is absent |

Everything else the page needs comes from the facts, `page.json` and the page's strings table, not from the model: section titles, context step titles (`context.step.<id>`), result model names (`model.<name>`), metric names (`metric.<m>`), report-reference labels (`report.kind.<kind>`); run order, selection and reference come from facts.

### 3.3 The digit rule and the placeholder grammar

The digit rule is one module (`digit_rule.py`, H01) used by the resolver, the page lint and the layer. For every authored string, delete every placeholder, then tokenise the remainder with the Unicode word pattern `[^\W_]+(?:[-'][^\W_]+)*`. For each token that contains a decimal digit:

- **refuse** a token with no letter (a bare number);
- **refuse** a token not in `facts.names`, compared case-insensitively (a name the records do not carry, such as `top7`; `α2` is refused unless `facts.names` carries it);
- **accept** otherwise (`IDS2017`, `F1`, `R7`, `B1`).

Any other Unicode `Nd` character outside a token (a digit glued to punctuation) is refused. Number words (the catalog's list) are `WARN:`, never refused: `WARN: number word "<w>" in <where> — prefer a count fact (<key>)`.

**Placeholders** (`PLACEHOLDER_RE`, one grammar for the story, the resolver and the page):

```
\{([a-z][a-z0-9_]*)(?:\.([a-z][a-z0-9_]*))?(?::(int|num|pct|signed|count))?\}
```

A bare name is a fact key in story text (a runtime parameter in the page's strings); a dotted name is a lexicon path `{<key>.<path>}` (`{row.one}`, `{row.a}` — the entry's article and `one` — `{source.the}`) or a strings-derived noun (`{target.one}`). `V3` accepts lexicon paths and the format suffixes.

| closed set | catalog key | members |
|---|---|---|
| placeholder suffixes | `placeholder.suffixes` | `int` `num` `pct` `signed` `count` |
| lexicon paths | `lexicon.paths` | `one` `many` `a` `the` |
| number words | `number_words` | `one` `two` `three` `four` `five` `six` `seven` `eight` `nine` `ten` `eleven` `twelve` `thirteen` `fourteen` `fifteen` `sixteen` `seventeen` `eighteen` `nineteen` `twenty` `thirty` `forty` `fifty` `sixty` `seventy` `eighty` `ninety` `hundred` `thousand` `million` `billion` `half` `twice` `double` `triple` `percent` `dozen` |
| locales | `locales` | `en` |
| build locales | `build_locales` | `en` `qps-ploc` |

### 3.4 Motifs — one closed set, keyed by modality

The modality is `spatial-row` when the sample shape is `row` and `records/analysis.json#context.spatial` is non-null; otherwise it is the sample shape. The draft proposes the modality's base; a `show: motif` stance is applied without asking; otherwise fork `S3` is asked, and `--unattended` applies the base. A motif outside the admissible list is refused (`V5` in the plan, `BD2` in `page.json`). The motif lives only in `page.json.motif`; the template's `motif/field.js` sets `data-motif` on `#field`. Every motif draws on the same field, from template constants: nothing on the background is a fact, and the alert share of marks is the template constant below.

| closed set | catalog key | members |
|---|---|---|
| motifs | `motifs.set` | `grid` `points` `traces` `lanes` `links` `contours` |

| modality | base | admissible |
|---|---|---|
| `row` | `points` | `points` `lanes` `grid` |
| `spatial-row` | `contours` | `contours` `points` `lanes` `grid` |
| `window` | `traces` | `traces` `lanes` `grid` |
| `graph` | `links` | `links` `grid` |

### 3.5 Class roles

Every class the page names has one role; the page colours from roles, never from class names. The roles cover the task's classes (`classes`, `binary`, `ordinal` view keys), the label names (`labels`), or the two sides of the threshold for `anomaly` (named by the task's class names when a label exists, `normal` and `anomalous` otherwise); `value`, `reconstruction` and `horizon` carry no roles (`class_roles` is null). At most one `neutral` for the `classes`, `binary` and `anomaly` view keys.

**`alert` requires** a `user:` turn, a `show:` stance in `.claude/domain-forge-memory.md` § Network stances, or `task:anomaly` (the anomalous side is `alert` by task, and is never downgraded). `domain:` and `spec:` evidence only raises the proposal: fork `S2` is still asked, and `--unattended` turns an unsupported `alert` into `plain` with `WARN:`. lanekit writes `YYYY-MM-DD | <dataset> | show: roles … (user)` for every lane that declares `alert`; the orders lane (L01) expects `0` neutral (`structural:negative`) and `1` alert (the stance).

| closed set | catalog key | members |
|---|---|---|
| class roles | `roles.set` | `neutral` `alert` `plain` |
| alert requires | `roles.alert_requires` | `user:` `show:` `registry:` `task:anomaly` |
| evidence only | `roles.evidence_only` | `domain:` `spec:` |
| anomaly sides | `roles.anomaly_sides` | `normal` `anomalous` |

| basis | grants | rule |
|---|---|---|
| `structural:majority` | `neutral` | the train-majority class, when its train share is at least the neutral majority share (multiclass; sequence-classification over multiclass) |
| `structural:negative` | `neutral` | the negative class of a binary task (`schema.label.decode` value 0, or the class `data_contract` marks negative) |
| `task:anomaly` | `neutral` `alert` | the normal side is `neutral` and the anomalous side `alert`, by task |
| `user:<turn>` | `neutral` `alert` `plain` | the user said so in this conversation |
| `registry:<lane>` | `neutral` `alert` `plain` | the verification harness's lane registry declares it: `VEGA_LANE_REGISTRY`, which the matrix sets (`run_matrix.py`) and a production run never does, names the registry (`lanes.json`), and that registry holds the lane and gives the class that role (`lanes[].roles`); without it V6 refuses the basis. Never presented as a user's turn |
| `show:<stance date>` | `neutral` `alert` `plain` | a `show:` roles line of that date in § Network stances |
| `vega:default` | `neutral` `plain` | a proposal applied without an answer; never `alert` |

The bases a run authors for `alert` are three: `user:`, `show:` and `task:anomaly`. `registry:<lane>` is the fourth member of `alert_requires` for the verification harness alone: the harness's lanekit writes it, never the model, and V6 admits it only when `VEGA_LANE_REGISTRY` names a `vega/test-lanes@1` registry that declares that lane and gives the class that role; a production run has no such registry, so V6 refuses the basis there whatever the plan says.

### 3.6 The lexicon — `vega/lexicon@1`

The lexicon is the page's only source of domain words. Each key is `{one, many, article}`; `one` and `many` match the lexicon value pattern (lower case, no digit); `article` is `a` or `an`, and `the` is allowed on `source` only. `target` is optional: its default comes from the page's strings (`target.one` / `target.many` in the lane's view-key variant). The title and description are not lexicon entries: they come from `MANIFEST.serve.identity` into `page.json.identity`. The draft proposes the row noun from `MANIFEST.serve.identity.noun` / `.nouns` and the other keys from the defaults; the model rewrites entries with domain words (`feature: statistic`) inside the rules.

| closed set | catalog key | members |
|---|---|---|
| lexicon keys | `lexicon.keys` | `row` `feature` `source` `step` `window` `node` `edge` `graph` `target` |
| lexicon optional keys | `lexicon.optional` | `target` |
| lexicon entry fields | `lexicon.entry` | `one` `many` `article` |

| shape | required keys |
|---|---|
| `row` | `row` `feature` `source` |
| `window` | `row` `feature` `source` `step` `window` |
| `graph:node` | `row` `feature` `source` `node` `edge` |
| `graph:graph` | `row` `feature` `source` `node` `edge` `graph` |

D6's lane registry (`lanes.json` `lexicon`) and its `VS09`/`VS10` negatives use these keys.

### 3.7 Proposal rules and bounds

- **Input groups** (`FamilyProposer`, in order): shaped features the recipe created (one candidate per creating step with 2 or more members); encoded nominals; the remaining numerics joined by union-find over the analysis relations with |r| at or above the correlation threshold; singletons grouped by a shared name token; then candidates merged or split to the count bounds (3–9 when there are 9 or more shaped inputs, else 1 to the input count). Every member appears exactly once, in `netspec.data_contract.features[].name` order, and the network view's input names equal that list.
- **Choices.** One per decision except the block types in `choice_skip_types`. A choice is mandatory for the first decision of each vega-forge decision family, and for every decision a selected run's delta touched.
- **Runs.** Every `baseline` and `ablation` run with status `done`, the best HPO trial of each study and the selected run; ordered by the metric's direction; `reference` is the baseline run; labels are mechanical (`delta_label`), for the model to rewrite.
- **Code steps.** The walk ids of §6.2 for this network, each bound to the first role of `walk_roles` that resolves.

| closed set | catalog key | members |
|---|---|---|
| choice skip types | `choice_skip_types` | `concat` `flatten` `reshape` `positional` |

| bound | catalog key | value |
|---|---|---|
| input groups, fewest | `input_groups.min` | 3 |
| input groups, most | `input_groups.max` | 9 |
| inputs from which the fewest applies | `input_groups.min_applies_from_inputs` | 9 |
| correlation threshold | `input_groups.corr_threshold` | 0.8 |
| choices, fewest | `choices.min` | 3 |
| choices, most | `choices.max` | 8 |
| neutral majority share | `roles.neutral_majority_share` | 0.5 |
| motif alert share | `motifs.alert_share` | 0.07 |
| headline length | `authored_lengths.headline` | 160 |
| lead length | `authored_lengths.lead` | 280 |
| caption length | `authored_lengths.caption` | 200 |
| context step length | `authored_lengths.context_step` | 700 |
| fact label length | `authored_lengths.fact_label` | 40 |
| choice title length | `authored_lengths.choice_title` | 60 |
| choice text length | `authored_lengths.choice_text` | 600 |
| run label length | `authored_lengths.run_label` | 60 |
| results note length | `authored_lengths.note` | 160 |
| code note length | `authored_lengths.code_note` | 160 |
| limit title length | `authored_lengths.limit_title` | 60 |
| limit text length | `authored_lengths.limit_text` | 400 |
| group label length | `authored_lengths.group_label` | 40 |
| group text length | `authored_lengths.group_text` | 160 |
| monitor note length | `authored_lengths.monitor_note` | 240 |
| layer abstract, shortest | `authored_lengths.abstract_min` | 40 |

### 3.8 Validation — `plan_rules.py`, each refusal `ERROR: V<n> <where>: <why>`, exit 2

One owner per rule: the draft (`draft_plan.py` and its proposers) proposes the lexicon, the class roles and the motif; `plan_rules.py` holds the block checks (lexicon, class roles, input groups, focus); the digit rule and `PLACEHOLDER_RE` live in `digit_rule.py`. There is no separate text-check module.

| rule | checks |
|---|---|
| `V1` | `facts.digest` equals the sha256 of `plan/facts.json`; `template.manifest_digest` equals `page_template.digests()`'s — the installed `assets/page/TEMPLATE.json`'s digest, or, before one is written, the digest a fresh `--write` would give |
| `V2` | the digit rule over every authored string |
| `V3` | every placeholder is a fact key, or a lexicon path with a known key and path; the format suffix is a known one |
| `V4` | no record path in authored text: `records/`, `MANIFEST.json`, `netspec.json`, `#`-refs, `sha256:`, `report:` |
| `V5` | closed sets: `locale`; `motif` admissible for the modality; lexicon keys (required present, none unknown) and values |
| `V6` | class roles: every class once; the neutral bound; `alert` only on `user:`, `show:`, `task:anomaly` or `registry:<lane>`; `registry:<lane>` only when `VEGA_LANE_REGISTRY` names a lane registry that declares that lane and that role for that class; no role on a task without classes; the basis grammar |
| `V7` | input groups: a partition of the shaped inputs; the count bounds; unique ids of the id pattern; members exist; the carry's `_unplaced` group (the inputs a carried plan could not place) refused until its inputs are placed |
| `V8` | choices: known decision ids; every mandatory one present; the count bounds (fewer only when fewer decisions exist); focus kinds |
| `V9` | limits: every mandatory limit id covered; every id known |
| `V10` | experiment labels are keyed by exactly the proposed runs |
| `V11` | code-note keys are proposed walk ids |
| `V12` | lengths per `authored_lengths` |
| `V13` | `status` is `authored` and every required slot is non-null |
| `V14` | an authored report role is a catalog role whose label exists in the `.aux` with the role's kind |
| `V15` | what the words claim holds against the facts (`plan_claims.py`): a verdict limit never softens a recorded `beaten_by_baseline` ("could not separate", "does as well", "as well as", "indistinguishable"); a breakdown caption never calls its bars "the primary metric" when the breakdown's metric is another; no two network choices share a title, and a decision that is no block of the network is never titled or described as one; no two input groups share a label |

## 4. The story — `vega/story@2` (`showcase/story.json`) — and the page config — `vega/page@1`

`resolve_story.py` (class `StoryResolver`) writes `story.json` from an authored plan that passed `V1`–`V15`; `--check` re-resolves and compares byte for byte (exit 3 when it differs). The file is written once with `allow_nan=False` and `ensure_ascii=False`, pinned, and never edited. The page reads nothing else about the project from the story (`schemas/story-2.schema.json`).

### 4.1 The blocks

`story@2` carries D4 §15's blocks, and the resolver emits exactly these:

- **`lexicon@1`** (§3.6).
- **`class_roles@1`** = `{schema: "vega/class-roles@1", roles: {class: neutral|alert|plain}, source: {class: <basis>}, proposed: {neutral, rule, alert, evidence}}`, each basis one of §3.5; null for a task without classes.
- **`input_groups@2`**: `[{id, label, text, members}]`, members being network input names in `netspec.data_contract.features[].name` order (D5's `input_families`, renamed).
- **`focus@1`** on every network choice: `{kind: block, ref}`, `{kind: family, ref}`, `{kind: feature, ref}` or `{kind: all}` (§4.2).
- **`results.breakdown@1`**: `{kind, metric, names, values, weak, direction, caption}`; `weak` is a values key the resolver computes (§4.5), never the page.
- **`values.primary`** = `{value, ci, n}` and **`values.ci_level`** (§2.2). A bundle without `ci_level` omits it, with `WARN:`.
- **Context steps** carry ids (§4.4); the page takes the titles from `context.step.<id>`.
- **Result models** carry `name` (§4.6); the page renders `model.<name>`.
- **`results.verdict`** = `{earned, beaten_by_baseline, strongest_baseline, diff}` as the evaluation recorded it (`diff` a values key or null); a verdict that is not assessable (§1.2) is `{earned: null, beaten_by_baseline: null, strongest_baseline: null, diff: null, reason}`, `reason` the values key of `eval_verdict_reason` (format `verbatim`), which the page says after `results.verdict.not_assessable`; any other verdict has a boolean `earned` and `beaten_by_baseline` and no `reason`.
- **Experiment run labels** are authored text under the digit rule.
- **`code_notes[<walk id>]`** = `{note, link: {href, kind: "report"|"file"} | null, ref: {kind, number, page, title} | null}`; `number` and `page` are values keys (report facts), and the page composes the label from `report.kind.<kind>`, so no English label is composed in Python. In brief-link mode `href` points to `showcase/briefs/<stage>-brief.md` with `kind: file` and `ref` is null.
- **The motif** lives only in `page.json.motif`.

`values` holds only the facts some field references (placeholders plus the structured `value`, `names`, `values`, `weak`, `diff`, `reason`, `number` and `page` keys); `formats` and `refs` hold the same keys. The sections are, in order, `context`, `network`, `experiments`, `results`, and optionally `many` (`{monitor_note}`).

| closed set | catalog key | members |
|---|---|---|
| story sections | `sections.ids` | `context` `network` `experiments` `results` `many` |
| required sections | `sections.required` | `context` `network` `experiments` `results` |
| focus kinds | `focus.kinds` | `block` `family` `feature` `all` |
| focus families | `focus.families` | `architecture` `objective` `optimization` `regularization` |
| breakdown kinds | `breakdown.kinds` | `per_class` `per_label` `per_bin` `per_horizon` `per_feature` `at_k` |
| context step ids | `context_steps.ids` | `problem` `data` `windows` `graph` `cleaning` `split` |
| polarity values | `polarity.values` | `neutral-alert` `plain` `diverging` `normal-anomalous` |
| result model names | `result_models.names` | `network` `trivial` `linear` `tree` `naive` `seasonal-naive` `graph-linear` `lens` `campaign` |

### 4.2 Focus

A decision that is a block id becomes `{kind: block, ref: <id>}`. A decision whose `features_tested` is non-empty and whose authored text names a tested feature becomes `{kind: feature, ref: <input name>}`. The other decisions map to a decision family (`regularization[*]` included):

| decision | focus |
|---|---|
| `family` | `architecture` |
| `loss` | `objective` |
| `metrics` | `objective` |
| `optimizer` | `optimization` |
| `schedule` | `optimization` |
| `training` | `optimization` |
| `init` | `optimization` |
| `regularization` | `regularization` |
| `early_stopping` | `regularization` |

The decision families are the vega-forge catalog's `decision_families`, copied to `decision_families` and checked equal.

### 4.3 View keys

The view key selects the page's task module; a `sequence-*` task takes its base's key.

| task | view key |
|---|---|
| `binary` | `binary` |
| `multiclass` | `classes` |
| `multilabel` | `labels` |
| `regression` | `value` |
| `ordinal` | `ordinal` |
| `unsupervised` | `reconstruction` |
| `anomaly` | `anomaly` |
| `sequence-classification` | as the base: `binary` → `binary`, `multiclass` → `classes`, `multilabel` → `labels` |
| `sequence-regression` | as the base: `regression` → `value`, `ordinal` → `ordinal` |
| `forecast` | `horizon` |

### 4.4 Context steps and polarity

| sample shape | steps |
|---|---|
| `row` | `problem` `data` `cleaning` `split` |
| `window` | `problem` `data` `windows` `cleaning` `split` |
| `graph` | `problem` `data` `graph` `cleaning` `split` |

Polarity is derived by the resolver, never authored:

| task | with an alert role | otherwise |
|---|---|---|
| `binary` | `neutral-alert` | `plain` |
| `multiclass` | `neutral-alert` | `plain` |
| `multilabel` | `neutral-alert` | `plain` |
| `regression` | `diverging` | `diverging` |
| `ordinal` | `diverging` | `diverging` |
| `unsupervised` | `plain` | `plain` |
| `anomaly` | `normal-anomalous` | `plain` |
| `sequence-classification` | `neutral-alert` | `plain` |
| `sequence-regression` | `diverging` | `diverging` |
| `forecast` | `diverging` | `diverging` |

### 4.5 Results: the facts shown, and the breakdown

The primary is the task's: the vega-forge catalog's `tasks[].primary`, or its `primary_alt.metric` when that row's condition holds — on an anomaly lane with no label, `anomaly_score_mean` (mode `min`, R-C7) — as the evaluation recorded it (`values.primary`, §2.2). vega-show reads its direction as it reads every other (`story@2` `task.direction`, `page.json` `metric.direction`), so every task primary, like every results fact and every `direction_of`, is `max` or `min` in the vega-forge catalog's `metric_direction`, and has its `metric.<m>` strings key (§7.2).

`results_facts` names the metrics the results view states beside the primary, per task; every one is in the vega-eval catalog's metric set for the task and has a direction in the vega-forge catalog's `metric_direction`. A results fact is emitted only when `records/evaluation.json#metrics.models.network.<m>` exists (`results_emit`). The evaluation computes some metrics only when a label exists — `roc_auc`, `precision_at_k` and `recall_at_k` on an anomaly lane — so an unlabelled anomaly lane has no such facts and its results view shows only its primary. A missing results fact is neither a failure nor a `WARN:`: `proposals.results_facts` lists only the emitted facts, and the authored notes' keys are a subset of it (§3.2), so no note can name a missing one. The breakdown names the per-unit metric and the metric whose direction it reads (`direction_of`); `weak` is true for a unit whose value is worse than the trivial baseline's per-unit value in that direction, or whose interval's far end is worse than the evaluation's recorded bar. The multilabel breakdown is per-label `pr_auc`; no `f1_per_label` exists.

| task | metrics |
|---|---|
| `binary` | `roc_auc` `pr_auc` `balanced_accuracy` `ece` |
| `multiclass` | `balanced_accuracy` `accuracy` `ece` `mcc` |
| `multilabel` | `f1_macro` `hamming` `subset_accuracy` |
| `regression` | `mae` `r2` `median_ae` |
| `ordinal` | `accuracy` `qwk` |
| `unsupervised` | `r2` |
| `anomaly` | `roc_auc` `precision_at_k` `recall_at_k` |
| `sequence-classification` | `roc_auc` `pr_auc` `accuracy` |
| `sequence-regression` | `mae` `r2` |
| `forecast` | `smape` `rmse` `mae` |

| task | breakdown |
|---|---|
| `binary` | none |
| `multiclass` | `per_class` of `f1_per_class`, direction of `f1` |
| `multilabel` | `per_label` of `pr_auc`, direction of `pr_auc` |
| `regression` | none |
| `ordinal` | none |
| `unsupervised` | `per_feature` of `per_feature_error`, direction of `reconstruction_error` |
| `anomaly` | none |
| `sequence-classification` | as the base |
| `sequence-regression` | as the base |
| `forecast` | `per_horizon` of `per_horizon_error`, direction of `mae` |

### 4.6 Result models

The results view names each model by `name`; the evaluation's baselines map onto names as below (`network` is the evaluated network itself). `naive` and `seasonal-naive` apply to forecasts and sequence regression, `graph-linear` to graphs at node level.

| model | from the vega-eval baseline |
|---|---|
| `trivial` | `trivial` |
| `linear` | `linear` |
| `tree` | `tree` |
| `naive` | `naive` |
| `seasonal-naive` | `seasonal-naive` |
| `graph-linear` | `graph-linear` |
| `lens` | `lens` |
| `campaign` | `campaign-baseline` |

### 4.7 `page.json` — `vega/page@1`

The page template is copied byte-identical and nothing is substituted in it; everything lane-specific the page needs beyond the story is in `page.json`, built by `page_config.build_page_config` (W06) inside `build_showcase.py`, **after** `resolve_story`, and pinned (`schemas/page-1.schema.json`). It holds the template's version and `TEMPLATE.json` digest, the bundle `MANIFEST.json` digest, `identity` (from `MANIFEST.serve.identity`), `lane` (`task, base, view_key, shape, level, link, family, goal, score_source`), `metric` (`primary, direction, bounds, secondary`, the direction from the vega-forge catalog's `metric_direction`), `motif` (`id, base, modality, source, reason`), `views` (`order, off`), `recipe` (the ops that reach the network's inputs), `reference` (the reconstruction or distance quantiles on the demo sample, or null), `explain` (the explain-catalog subset for this lane's blocks, activations and link, verbatim), `walk` (§6.2) and `code`. `page_config` only assembles; it proposes nothing (the draft proposes, §3). **`views`**: a tour view is on when what it renders exists, else it is `off` with a strings key the page renders — `context`, `network`, `results` need the story section of that id (`views.off.no_section`); `try` needs a single-answer route, `POST /v1/explain` or `POST /v1/predict` (`views.off.no_route`), and a starting sample — a preset, a shipped test row or a demo item (`views.off.no_examples`); `many` needs `POST /v1/predict/csv` (`views.off.no_route`). A bundle without a service, or whose `MANIFEST.serve.endpoints` is missing, is refused. **`motif.source`**: a `show: motif` stance in the plan gives `show:<its date>`; the modality's base without a stance is the unattended default (`vega:default`); any other admissible motif without a stance was fork S3's answer (`user:S3`). `check_page_config(page, network)` refuses a link outside the one link vocabulary (the vega-forge catalog's `inspection.links`: `softmax`, `sigmoid`, `sigmoid_each`, `identity`, `cumulative`, `reconstruction`, `distance`, `horizon`) and a motif outside the modality's admissible list (gate `BD2`).

| closed set | catalog key | members |
|---|---|---|
| recipe ops | `page.recipe_ops` | `impute` `indicator` `encode` `clip` `log1p` `standard_scale` `robust_scale` `minmax_scale` `bin` `window` `graph_edges` `project` `other` |
| reference kinds | `page.reference_kinds` | `reconstruction_error` `distance` |

### 4.8 Attribution — `vega/attribution@2` (`showcase/data/attribution.json`)

Attribution belongs to vega-show. `build_showcase.py` (through `attribution_build.py`) runs `predict.load_inspector(bundle).attribute(...)` over the demo strata of `examples/demo.json` — the training part, never the test part — and writes `data/attribution.json`, pinned with role `data` (`schemas/attribution-2.schema.json`). One column per stratum of the lane's weights axis; each cell holds the per-sample attribution rows over the network inputs in network order, and how many the network got right. The payload carries ids and numbers, no English. pack's v2.2 `_class_attribution` and D2's `write_attribution` are not used. `attribution_build.py` runs in a clean child process through the bundle's own `load_inspector` (after checking `status()["ready"]`), asks with string class selectors, and sends the demo rows as the service decodes a JSON body, so the attribution is the one the page would get for the same row. Without a demonstration sample, without inspection, or with an inspector that is not ready, the showcase pins no `data/attribution.json`, with a `WARN:`.

The axis follows the bundle's reading (`MANIFEST.output.reading`, the page's weights axis); a column's id is the demo stratum's id (`class-0`, `bin-2`, `h1`, `all`) and its name the data's own (a class or label name, a bin's range, the step):

| reading | axis | a column | the target of its rows |
|---|---|---|---|
| binary | `class` | negative, positive | the positive class's logit |
| multiclass | `class` | each class | the column's class |
| ordinal | `class` | each class | the task's default, `expected_rank` |
| multilabel | `label` | each label | the column's label |
| regression, quantile | `bin` | each quintile bin of the target | the value (the quantile nearest 0.5) |
| forecast | `horizon` | each step ahead | that step, over every demo item |
| reconstruction | `group` | one column, `all` | the reconstruction error |
| anomaly (labelled) | `group` | each class | the error or the distance |
| anomaly (unlabelled) | `group` | one column, `all` | the error or the distance |

`correct` counts the rows whose answer's class is the truth (class columns) or whose label is flagged (label columns), null otherwise. A row is the per-sample attribution over the network inputs in network order: a row's own list, a window's per-feature sums over its steps (`sum_over_steps`), a graph node's own part (`self_only`).

| closed set | catalog key | members |
|---|---|---|
| attribution targets | `attribution.targets` | `logit` `binary_logit` `label_logit` `value` `quantile` `expected_rank` `threshold_logit` `reconstruction_error` `feature_error` `svdd_distance` `forecast_step` |
| attribution methods | `attribution.methods` | `grad_x_input` `integrated_gradients` |
| attribution reductions | `attribution.reduce` | `sum_over_steps` `self_only` |
| attribution axis kinds | `attribution.axis_kinds` | `class` `label` `bin` `group` `horizon` |

## 5. The showcase directory, `SHOWCASE.json` and the slot

### 5.1 Layout

The manifest is named `SHOWCASE.json`, not `MANIFEST.json`, because `bundle_root` treats any directory holding a `MANIFEST.json` as a bundle. It is never in its own `files`.

```
<bundle>/showcase/
  SHOWCASE.json                  vega/showcase@1 (not in files)
  page/**                        byte copy of vega-show/assets/page (the TEMPLATE.json files)
  page.json                      vega/page@1
  story.json                     vega/story@2
  data/attribution.json          vega/attribution@2 (optional)
  report/report.pdf              linked mode
  briefs/<stage>-brief.md        brief-link mode
  plan/facts.json                vega/show-facts@1
  plan/story-plan.json           vega/story-plan@2
  docs/show-brief.md             the stage brief (post-report)
  .work/  .smoke/                draft plan, checks, markers, layer data, qps builds, screenshots
<bundle>/showcase.prev/          the previous showcase, one level kept (never served)
<bundle>/showcase.building/      transient; swapped into showcase/ on success
```

| path | schema | role | served | pinned | when |
|---|---|---|---|---|---|
| `page/**` | `vega/page-template@1` | `page` | yes | yes | always |
| `page.json` | `vega/page@1` | `page` | yes | yes | always |
| `story.json` | `vega/story@2` | `story` | yes | yes | always |
| `data/attribution.json` | `vega/attribution@2` | `data` | yes | yes | optional |
| `report/report.pdf` | — | `report` | yes | yes | linked |
| `briefs/<stage>-brief.md` | — | `report` | yes | yes | brief-links |
| `plan/facts.json` | `vega/show-facts@1` | `record` | no | yes | always |
| `plan/story-plan.json` | `vega/story-plan@2` | `record` | no | yes | always |
| `docs/show-brief.md` | — | — | no | no | after publish |
| `.work/**` | — | — | no | no | scratch |
| `.smoke/**` | — | — | no | no | scratch |

| closed set | catalog key | members |
|---|---|---|
| showcase roles | `showcase.roles` | `page` `story` `report` `data` `record` |
| served roles | `showcase.served_roles` | `page` `story` `report` `data` |
| unpinned directories | `showcase.unpinned_dirs` | `.work/` `.smoke/` `docs/` |
| report status | `showcase.report_status` | `linked` `brief-links` |
| role-map origins | `showcase.role_origins` | `document-project` `proposed` |

### 5.2 Extensions

`.py` is refused everywhere (and `.pyc`, `.so`): the showcase holds no code the service could import, and no generated JavaScript — the page's modules are the template's, byte for byte.

| extension | where |
|---|---|
| `.html` `.json` `.pdf` `.md` | anywhere pinned |
| `.js` `.css` `.svg` `.woff2` | only under `page/` (and under `.work/qps/page/`, the pseudo-locale build), where the copy must equal `TEMPLATE.json` |
| `.png` | only under `.smoke/` |
| `.py` `.pyc` `.so` | refused everywhere |

The `.work/qps/page/` row widens R-E1's "only under `page/`" (a deviation of WP K3, recorded): R-F2 sends the `qps-ploc` build only to `showcase/.work/qps/` (gate `BD7`), and that build carries its own copy of the template, so the extension checks `BD3` and `CK3` accept the page extensions there as under `page/`, and nowhere else. `.work/` is neither pinned nor served (§5.1) and the slot's media check skips it (`SL8`), so the widening admits nothing the slot answers.

| closed set | catalog key | members |
|---|---|---|
| pinned-anywhere extensions | `showcase_extensions.anywhere` | `.html` `.json` `.pdf` `.md` |
| refused extensions | `showcase_extensions.refused` | `.py` `.pyc` `.so` |

### 5.3 `SHOWCASE.json` — `vega/showcase@1`

`schemas/showcase-1.schema.json` is normative. The fields the slot reads are marked ●:

```json
{ "schema": "vega/showcase@1",                                                          ●
  "created_at": "…", "produced_by": "/vega-show",
  "bundle": { "manifest": "MANIFEST.json", "manifest_digest": "sha256:…",                ●
              "slug": "…", "run": "R7", "model_version": "sha256:…" },                   ● model_version
  "model": { "path": "….eval.pack.html", "turtle_digest": "sha256:…" },
  "entry": "page/index.html",                                                           ●
  "template": { "name": "vega-show/page", "version": "2.3.0", "manifest_digest": "sha256:<TEMPLATE.json>" },
  "page": { "file": "page.json" },
  "story": { "file": "story.json", "schema": "vega/story@2", "values_n": 57, "placeholders_n": 63,   ● file
             "sections": ["context", "network", "experiments", "results"] },
  "attribution": { "file": "data/attribution.json" } | null,
  "plan": { "file": "plan/story-plan.json", "facts": "plan/facts.json" },
  "report": { "status": "linked" | "brief-links",
              "pdf": { "source": "docs/main.pdf", "file": "report/report.pdf", "digest": "sha256:…" } | null,   ● file, digest
              "aux": { "source": "docs/main.aux", "digest": "sha256:…" } | null,
              "roles": { "source": "docs/label-roles.json", "origin": "document-project" | "proposed",
                         "digest": "sha256:…", "bound": [ … ], "missing": [ … ] } | null,
              "briefs": { "<stage>": { "source": "…", "file": "briefs/<stage>-brief.md", "digest": "sha256:…" } } | null },
  "files": { "<relpath>": "sha256:…" },                                                 ●
  "roles": { "page": [ … ], "story": [ … ], "report": [ … ], "data": [ … ], "record": [ … ] },
  "served": [ … ],                                                                      ●
  "requires": { "endpoints": [ … ], "network_view": "vega/network-view@2",
                "explanation": "vega/explanation@2", "examples": "vega/examples@2" },
  "supersedes": { "digest": "sha256:…", "created_at": "…" } | null,
  "markers": [ "OK: …", "WARN: …" ] }
```

In brief-link mode `report.pdf`, `.aux` and `.roles` are null and `briefs` lists the copies, which are in `files`, in `roles.report` and in `served`. Invariants the builder and the checker enforce: every `served` path is in `files`; every `files` path is relative, normalised, without a `..` segment, not a symlink, and has an extension §5.2 allows where it is; `SHOWCASE.json` is not in `files`; the bundle's `MANIFEST.files` pins no `showcase/` path (one owner per file).

### 5.4 The slot

The slot is serving machinery (`serve/routes_showcase.py`, capability `showcase`, WP I07); this contract fixes what it accepts. It follows the route-module protocol: `ENDPOINTS`, `MODEL_ROUTES`, `OPEN_WHILE_UNREADY`, `CAPABILITY`, `register(app, state)`.

- **Location.** `<ENV>_SHOWCASE` (`main --showcase DIR` sets it), else `<bundle root>/showcase`.
- **The slot opens at start.** `register` opens and verifies the slot synchronously and sets `state.page_override`. Its `OPEN_WHILE_UNREADY` is below, so the page can load its own modules to show "not ready".
- **Verification.** `schema`; `bundle.manifest_digest` against the sha256 of the running bundle's `MANIFEST.json` bytes and `bundle.model_version` against `MANIFEST.model_version`; every `files` digest; the path invariants; `served ⊆ files`; the entry, `story.file` and `report.pdf.file` served. The served files are read into memory once; a file altered after start-up is never served.
- **The closed world.** The slot serves only `served ⊆ files`, from memory, with `ETag` = the file's digest (304 on `If-None-Match`), `Cache-Control: no-cache` and `X-Content-Type-Options: nosniff`; HTML responses also carry the content-security policy. These refuse the showcase: a symlink; a path outside the directory; a total above the byte cap; any file with a media extension that is not in `files` and not under `.work/`, `.smoke/` or `docs/`.
- **A refused showcase never makes the model unready.** `GET /` falls back to the console page; `/v1/showcase` says why.
- **Status.** `GET /v1/showcase` and `/health.showcase` return the same object (`verified: false` is D6's "refused"):

```json
{ "present": true, "verified": true, "reason": null, "digest": "sha256:<SHOWCASE.json bytes>",
  "entry": "page/index.html", "template": {"version": "2.3.0", "manifest_digest": "sha256:…"},
  "report_status": "linked", "created_at": "…", "files": 212, "unlisted": [] }
```

- **`MANIFEST.serve.showcase_slot`** is written by pack (S07) whenever `serve/` is rendered, and its absence is gate `PF4`:

```json
{ "dir": "showcase", "schema": "vega/showcase@1",
  "routes": ["GET /", "GET /console", "GET /showcase/{path}", "GET /v1/story", "GET /report.pdf", "GET /v1/showcase"],
  "env": "<ENV>_SHOWCASE", "flag": "--showcase" }
```

| route | serves |
|---|---|
| `GET /` | the verified entry (`page/index.html`), else the console page |
| `GET /console` | always the console page (the v2.2 generic page, generalised; literal-linted) |
| `GET /showcase/{path}` | a served file, else 404 as a problem body |
| `GET /v1/story` | `story.json` |
| `GET /report.pdf` | `report/report.pdf` (linked mode) |
| `GET /v1/showcase` | the status object |

| closed set | catalog key | members |
|---|---|---|
| slot routes | `slot.routes` | `GET /` `GET /console` `GET /showcase/{path}` `GET /v1/story` `GET /report.pdf` `GET /v1/showcase` |
| open while unready | `slot.open_while_unready` | `/` `/console` `/showcase/` `/v1/showcase` |
| slot reads | `slot.reads` | `schema` `bundle.manifest_digest` `bundle.model_version` `entry` `story.file` `report.pdf.file` `report.pdf.digest` `files` `served` |
| slot status fields | `slot.status_fields` | `present` `verified` `reason` `digest` `entry` `template` `report_status` `created_at` `files` `unlisted` |

| bound | catalog key | value |
|---|---|---|
| slot byte cap | `slot.max_bytes` | 67108864 |

The media table (`slot.media`) is the service's, plus `.md → text/markdown; charset=utf-8`.

### 5.5 Check, staleness and re-emit

`build_showcase.py --check <bundle>/showcase` (class `ShowcaseChecker`) exits 0 on pass, 2 on broken and 3 on stale, in the order of gates `CK1`–`CK10` (§10). It writes `showcase/.work/checks.json` (`{status, problems: [{rule, where, message}], warnings, checked_at}`) when the showcase directory exists — never at the showcase root, which `SHOWCASE.json` pins; `--json` prints it. The stage's other working files live there too: `showcase/.work/markers.json` (every step's marker lines, appended, never overwritten) and `showcase/.work/showcase-layer.json` (the layer's authoring file, §8); the build creates no symlink inside `showcase/`.

- **Broken** (exit 2): `SHOWCASE.json` unreadable or of another schema; a `files` digest that differs; a refused extension, a path-invariant breach or an unlisted media file outside the unpinned directories; `served ⊄ files`; `bundle.manifest_digest` different from the bundle's `MANIFEST.json` (the bundle was altered in place — a re-pack makes a new directory).
- **Stale** (exit 3): **the bundle was re-emitted after this showcase** — `bundle.manifest_digest` equals some `MANIFEST.reemits[].previous_manifest_digest` (a legitimate `--re-emit-serving` must not loop back to pack); a report, `.aux` or role-map source digest that changed; in brief-link mode, a brief source digest that changed; a report that now exists where the showcase records brief links. The slot refuses a showcase of a re-emitted bundle with the reason `the bundle was re-emitted after this showcase; re-run /vega-show`, and `/vega` proposes `show`.
- **Warn only**: the installed template's digest differs from the recorded one (`WARN: template updated since build`, not stale).

### 5.6 Docker

`.dockerignore` keeps `showcase/` and excludes `showcase.prev/`, `showcase.building/`, `showcase/.smoke/` and `showcase/.work/`: the image carries the page, never the scratch.

## 6. Report binding

Links from the page into the report go through **label roles**, never through typed numbers: a role names a kind of object the report carries (the network's forward equation, the serving slice), `document-project` records which LaTeX label is that object, and the `.aux` gives the label's number, page and title. The `.aux` is one adapter (`AuxLabelSource`); a future non-LaTeX report adds one adapter class.

### 6.1 Report roles

| role | kind | stage | what `document-project` labels |
|---|---|---|---|
| `intro.lane` | figure | forge-data | the lane as a chain diagram (introduction) |
| `chapter.forge-data` | section | forge-data | the data chapter |
| `chapter.lens` | section | lens | the analysis chapter |
| `chapter.shape` | section | shape | the shaping chapter |
| `chapter.blueprint` | section | blueprint | the architecture chapter |
| `chapter.net` | section | net | the network chapter |
| `chapter.train` | section | train | the training chapter |
| `chapter.eval` | section | eval | the evaluation chapter |
| `chapter.pack` | section | pack | the serving chapter |
| `shape.lineage` | figure | shape | the recipe as a lineage diagram |
| `blueprint.modules` | figure | blueprint | the module map and ports |
| `net.blocks` | figure | net | the block diagram from the spec |
| `net.decisions` | table | net | the decisions table |
| `net.forward` | equation | net | the family's forward equation |
| `net.loss` | equation | net | the loss |
| `net.update` | equation | net | the update rule |
| `train.ladder` | figure | train | the ladder |
| `train.curves` | figure | train | the selected run's curves |
| `eval.metrics` | table | eval | the metrics table with intervals |
| `eval.calibration` | figure | eval | the calibration curve |
| `pack.layout` | figure | pack | the bundle layout |
| `pack.slice` | figure | pack | the serving slice of the module map |
| `conclusion` | section | pack | the conclusion |

### 6.2 Walk ids and walk roles

The walk-id grammar is D4's: `root`, `store`, `build`, then one id per distinct block type in forward order (the type name), then `forward`, `link`, `shape`, `run`, `explain`, `route`. It is computed once, by `page_config.build_walk` (W06), so the page and the code notes cannot disagree about ids. A code step is bound to the first role of its row that resolves. The code anchors follow the v2.3 layout: the block files `blocks_core.py`, `blocks_seq.py`, `blocks_graph.py`, `blocks_gen.py` by `SEGMENT_FILE`; `network.py` `^build_model = `; `store.py` `^\s+def load\(`; `serve/routes_predict.py` `^def register\(`; they resolve against rendered packages (`test_walk_anchors.py`, W15).

| closed set | catalog key | members |
|---|---|---|
| walk head | `walk.head` | `root` `store` `build` |
| walk tail | `walk.tail` | `forward` `link` `shape` `run` `explain` `route` |

| walk id | roles, in order |
|---|---|
| `root` | `blueprint.modules` `chapter.blueprint` |
| `store` | `pack.slice` `chapter.pack` |
| `build` | `net.decisions` `net.blocks` `chapter.net` |
| `forward` | `net.forward` |
| `link` | `net.forward` `net.loss` |
| `shape` | `shape.lineage` `chapter.shape` |
| `run` | `pack.slice` |
| `explain` | `net.forward` |
| `route` | `pack.slice` `chapter.pack` |
| block types `dense` `embedding` `concat` `batchnorm` `layernorm` `activation` `residual` `sequential` `parallel` `conv1d` `pool1d` `rnn` `gru` `lstm` `attention` `attention_pool` `transformer_encoder` `feature_tokenizer` `positional` `flatten` `reshape` `pool` `head` `decoder` `reparameterize` `gcn_conv` `sage_conv` `gat_conv` `gin_conv` `graph_pool` | `net.forward` `net.blocks` |
| block types `dropout` `edge_dropout` | `net.blocks` `net.decisions` (the figure that draws the silenced units first, the decisions table after) |
| block type `custom` | `net.blocks` `chapter.net` |

### 6.3 `vega/label-roles@1` (`docs/label-roles.json`)

Written by `document-project` in Vega mode at every compile (the H02 patch) and by lanekit's `report_stub`, read by vega-show, never written by it (`schemas/label-roles-1.schema.json`):

```json
{ "schema": "vega/label-roles@1", "lane": "dataset:cicids2017", "generated_at": "2026-09-29T11:00:00Z",
  "adapter": "latex-aux",
  "report": { "tex": "docs/main.tex", "pdf": "docs/main.pdf", "aux": "docs/main.aux", "aux_digest": "sha256:…" },
  "roles": { "net.forward": { "label": "eq:mlp", "kind": "equation", "stage": "net", "number": "3", "page": 14, "title": "…" },
             "pack.slice":  { "label": "fig:serving", "kind": "figure", "stage": "pack" } } }
```

A role whose object the report does not carry is absent, never guessed. `number`, `page` and `title` are optional per role; when present, the resolver verifies each against the `.aux`. A map whose `report.aux_digest` differs from the current `.aux` is `ERROR:` exit 2 ("recompile /document-project", gate `PF6`), not a fork. lanekit's `report_stub` writes `docs/label-roles.json`, a `main.aux` whose anchors carry the `equation.`, `figure.`, `table.` and `section.` prefixes, and a `main.pdf`; `contracts.REPORT_ROLES()` returns the catalog's `report_roles`.

### 6.4 The `.aux` adapter

`AuxLabelSource(aux)` parses `\newlabel{name}{{number}{page}{title}{anchor}{…}}`, skipping `@cref` entries. The kind comes from the anchor prefix, and falls back to the label prefix:

| anchor prefix | kind |
|---|---|
| `chapter.` | section |
| `section.` | section |
| `subsection.` | section |
| `subsubsection.` | section |
| `figure.` | figure |
| `table.` | table |
| `equation.` | equation |
| `AMS.` | equation |

| label prefix | kind |
|---|---|
| `sec:` | section |
| `fig:` | figure |
| `tab:` | table |
| `eq:` | equation |

| closed set | catalog key | members |
|---|---|---|
| label kinds | `label_kinds.kinds` | `section` `figure` `table` `equation` |

### 6.5 Proposed roles

When a report exists but no role map was emitted (a hand-authored report), `LabelRoleProposer` ranks the `.aux` labels per role — the kind must match, the score counts the role's keywords in the label name and title, chapter roles prefer section-kind labels whose title matches the stage's words — and the draft shows the top candidate with its evidence. The model confirms or edits it in `authored.report_roles` (`V14`); `SHOWCASE.json.report.roles.origin` is `proposed`, with `WARN: label roles proposed from the .aux titles — /document-project in Vega mode emits them`.

### 6.6 Brief-link mode

Without a report (fork `S1`), each code note links to the stage brief that documents it: the brief is copied to `showcase/briefs/<stage>-brief.md`, pinned and served, and the note's link is `{href: "showcase/briefs/<stage>-brief.md", kind: "file"}`. Brief links are fork `S1`'s answer: `build_showcase.py --brief-links` is the user's, `--unattended` applies the fork's unattended answer, and with neither the build stops with `FORK:` (exit 2) before writing anything. The report gate `PF6` holds in this mode too: a brief the story links must still be the one the facts recorded (at its digest), and a showcase with no report and no stage brief at all to link is refused (`PF6 report.briefs`) — a lane that writes no stage brief cannot show its code notes in brief-link mode.

| stage | brief read from |
|---|---|
| `shape` | `records/shape.json#manifest.outputs.dir` + `/docs/shape-brief.md` |
| `blueprint` | the run directory of `records/architecture.json#netspec.input.pipeline` + `/docs/blueprint-brief.md` |
| `net` | `records/architecture.json#netspec.outputs.run_dir` + `/docs/net-brief.md` |
| `train` | `records/training.json#campaign.run_dir` + `/docs/train-brief.md` |
| `eval` | `records/training.json#campaign.run_dir` + `/docs/eval-brief.md` |
| `pack` | `<bundle>/docs/pack-brief.md` |

## 7. The page template

The template is D4's and lives in `vega-show/assets/page/`. It is **copied byte-identical, and nothing is substituted**: the lane is only in data (`story.json`, `page.json`, the service's payloads). There is no chrome table and no page manifest beside it (D5's `chrome.en.json` and `page-manifest.json` do not exist): the words are the page's keyed strings table, and the template's identity and required endpoints are `TEMPLATE.json`.

- **`TEMPLATE.json`** (`vega/page-template@1`, `schemas/page-template-1.schema.json`) is `{schema, version, files: {relpath: sha256}, requires_endpoints: [...], strings_keys_digest}`. The endpoint lists live in `page_template.py`: `REQUIRED_ENDPOINTS` (`GET /health`, `GET /v1/schema`, `GET /v1/examples`, `GET /v1/network`, `POST /v1/explain`, `GET /v1/code`, `POST /v1/predict`, `POST /v1/predict/csv`) is `requires_endpoints`; `OPTIONAL_ENDPOINTS` (`GET /v1/info`, `POST /v1/monitor`, `GET /v1/story`, `GET /v1/showcase`) are read when the service has them. A package that makes the page call a new route adds it to one of the two lists, and `write_manifest` refuses a template whose routes are on neither. The page also fetches deferred tensors from `GET /v1/network/tensor/{tensor_id}` (the inspection routes, bundle-contract §7.7); it is on neither list today, so a service without it leaves those tensors unfetched rather than failing PF5. `files` lists every file under `assets/page/` except `TEMPLATE.json` itself, `package.json` and `tests/`. `requires_endpoints` must be a subset of `MANIFEST.serve.endpoints` (gate `PF5`).
- **One module**, `vega-show/scripts/page_template.py` (W07), provides `write_manifest() -> dict`, `verify_template_copy(page_dir: Path) -> list[str]` and `digests() -> dict` (the last feeds `SHOWCASE.json.template`). `REQUIRED_SCRIPTS` for vega-show name `assets/page/index.html`, `assets/page/TEMPLATE.json` and `assets/page/strings.keys.json`.
- **Strings.** `js/core/strings.en.js` exports the single `STRINGS` table, merged from the area fragments `js/core/strings/<area>.en.js`; a duplicate key is a load error and a lint failure. `assets/page/strings.keys.json` is the generated key list this contract's sync test reads: once `TEMPLATE.json` exists, it must carry the key of every id vega-show emits (§7.2).
- **Test hooks.** `js/core/probe.js` exposes `window.__showcase = { version, ready, lane(), story(), page(), graph(), registry(), views(), current(), state(), math, missingKeys(), fetched(), probe(opts) }`. With `?probe=<views|all>`, the page writes `vega/page-probe@1` into `<pre id="showcase-probe">`. The `qps-ploc` pseudo-locale wraps every string with `⟦…⟧`; `build_showcase --locale en|qps-ploc` writes `qps-ploc` builds only to `showcase/.work/qps/`, which is never published.
- **One literal gate and one style lock.** D6's `lint_templates.py` with `foreign-literals.json` is the only deny-list gate; the page's computed-style golden is part of D6's `style-lock.json`. The template, the strings table and the renderers contain no class name, column name, dataset stem or domain word.

| closed set | catalog key | members |
|---|---|---|
| template excluded | `template.excluded` | `TEMPLATE.json` `package.json` `tests/` |
| probe hook members | `probe.hook_members` | `version` `ready` `lane` `story` `page` `graph` `registry` `views` `current` `state` `math` `missingKeys` `fetched` `probe` |

### 7.1 Views

The view ids are D4's. `tried` and `weights` are disclosures inside `network`, reported by the probe as sub-views. `#window` is an alias of `#many`.

| closed set | catalog key | members |
|---|---|---|
| views | `views.ids` | `start` `context` `network` `try` `many` `results` `code` |
| tour views | `views.tour` | `context` `network` `try` `many` `results` |
| network sub-views | `views.sub_views.network` | `tried` `weights` |

### 7.2 The strings keys vega-show's ids need

Every id vega-show writes into the story or `page.json` that the page renders through a strings key must have that key:

| key | for every member of |
|---|---|
| `context.step.{id}` | the context step ids (§4.1) |
| `model.{name}` | the result model names (§4.6) |
| `report.kind.{kind}` | the label kinds (§6.4) |
| `metric.{metric}` | the metrics of §4.5: the task primaries (`anomaly_score_mean` included), the results facts and the breakdowns |

The code room's words are the page's own too: `/v1/code` sends no English (`vega/code-view@2`, bundle-contract §7.2), and the page renders `code.role.<role>`, `code.about.<role>` (each `role` in snake case: `feature_transformer`, `model_store`, …) and `code.kind.<kind>` from its strings table; vega-show writes no such key, and the code notes it authors are keyed by walk id (§6.2).

## 8. The `showcase` layer

`apply_show_layer.py <model>.pack.html --data <bundle>/showcase/.work/showcase-layer.json --out <out>` splices one additive layer, `showcase` (`vega/showcase-layer@1`), through domain-forge's `apply_layer.py`, and the render asset `assets/showcase-render.js` mounts one tab, **Showcase**, `data-layer="showcase"`. The data holds `SHOWCASE.json` verbatim, the story summary (headline and lead rendered with values, section ids and counts, input groups with member counts, class roles, motif, polarity), the bindings (every placeholder key with its value, ref and where it is used), the report binding, the checks (`.work/checks.json` and the page smoke's `.smoke/smoke.json` — read from the showcase, never from the model's markers), the markers, the usage (the serve command from `MANIFEST.serve.identity`, the URL, the Docker line) and the readings. The render reads only `#layer-showcase-data`, is idempotent, shows an empty state when the data is missing, uses no `<img>` or `<canvas>`, and is at most 40 KB.

| closed set | catalog key | members |
|---|---|---|
| layer required keys | `layer.required` | `schema` `showcase` `story` `bindings` `report` `checks` `usage` `readings.abstract` |
| layer surfaces | `layer.surfaces` | `story` `bindings` `report` `open it` `checks` `markers and readings` |

| bound | catalog key | value |
|---|---|---|
| render size, KB | `layer.max_render_kb` | 40 |
| layer data size, KB | `layer.max_data_kb` | 400 |

The apply script refuses (exit 2) on gates `LY1`–`LY7` (§10), and exits 3 when the platform writer is missing.

## 9. Brief, memory, naming, re-run, hand-off

- **The brief.** `<bundle>/showcase/docs/show-brief.md`, brief-contract frontmatter plus `post_report: true` and the report digests (`report: {status, pdf_digest, aux_digest, roles_digest}`); the seven sections of the brief contract; written after the layer is published. It is **never a chapter**: `/document-project` skips a brief with `post_report: true`, and the slot's closed world excludes `docs/`.
- **Memory.** Lines in `.claude/domain-forge-memory.md` § Network stances, prefixed `show:`: roles (`YYYY-MM-DD | <dataset> | show: roles <class> <role> (<basis>), … (user)`), motif, lexicon, and one line per showcase built (`YYYY-MM-DD | <dataset> | show: showcase of <slug>-<run>, report <linked|brief-links>, motif <id>, roles <summary>`). A stance line is applied and cited, never re-asked.
- **Naming.** `….eval.pack.html` becomes `….eval.pack.show.html`.
- **Re-run.** Invoke again: the rebuild is in place, and the previous showcase is kept as `showcase.prev/`. `--plan <edited story-plan.json>` changes the wording without a new draft.
- **Hand-off.** The serve command and URL; `/document-project vega <lane>` when in brief-link mode; `/vega-show … --plan` for wording.

| what changed | `--check` | `/vega` proposes |
|---|---|---|
| the report PDF, `.aux` or role map | stale (3) | `show` |
| a report now exists; the showcase has brief links | stale (3) | `show` |
| a stage brief (brief-link mode) | stale (3) | `show` |
| the bundle was re-emitted (`--re-emit-serving`) | stale (3) | `show` |
| the bundle `MANIFEST.json` edited in place | broken (2) | `pack` |
| a showcase file edited | broken (2); the slot refuses and the service falls back | `show` |
| a re-pack (a new bundle directory) | the old showcase is untouched | `show` on the new `….pack.html` |
| the page template upgraded | pass with `WARN: template updated since build` | named in the summary, not proposed |

## 10. Gates

Every refusal of the stage has a gate id, and every gate id is the id of at least one passing seeded case in `scripts/tests/run_negatives.py` (D5's seeded defects, D6's `VS01`–`VS27` and the slot's `SS1`–`SS9`, D4's `PN1`–`PN12`, gate C's `GC-1`–`GC-4`, de-duplicated by gate id). "Seeded by" names the design cases each gate catches; a gate with none needs a case of its own. The slot's gates are verified by the serving machinery; a slot refusal is `verified: false` with the reason, and the model stays ready.

| closed set | catalog key | members |
|---|---|---|
| gate groups | `gate_groups` | `validate` `preflight` `build` `check` `slot` `layer` `page` `served` `catalog` |

| gate | group | where | holds that | exit | seeded by |
|---|---|---|---|---|---|
| `V1` | validate | `plan_rules.py` | the plan's facts and template digests are current | 2 | `D5-21` |
| `V2` | validate | `plan_rules.py` | the digit rule over every authored string | 2 | `D5-1` `D5-2` `D5-3` `VS01` |
| `V3` | validate | `plan_rules.py` | every placeholder is a fact key or a lexicon path, with a known suffix | 2 | `D5-4` `VS02` |
| `V4` | validate | `plan_rules.py` | no record path in authored text | 2 | `D5-5` `VS03` |
| `V5` | validate | `plan_rules.py` | closed sets: locale, motif, lexicon | 2 | `D5-6` `VS07` `VS08` `VS09` `VS10` `PN6` |
| `V6` | validate | `plan_rules.py` | class roles | 2 | `D5-7` `D5-8` `D5-9` `VS04` `VS05` `VS06` `PN5` |
| `V7` | validate | `plan_rules.py` | input groups | 2 | `D5-10` `D5-11` `D5-12` `VS11` `VS12` `VS13` `PN12` |
| `V8` | validate | `plan_rules.py` | choices | 2 | `D5-13` `D5-14` |
| `V9` | validate | `plan_rules.py` | limits | 2 | `D5-15` |
| `V10` | validate | `plan_rules.py` | experiment labels | 2 | `D5-16` |
| `V11` | validate | `plan_rules.py` | code-note keys | 2 | `D5-17` |
| `V12` | validate | `plan_rules.py` | lengths | 2 | `D5-18` `VS23` |
| `V13` | validate | `plan_rules.py` | required slots non-null | 2 | `D5-19` |
| `V14` | validate | `plan_rules.py` | authored report roles | 2 | `D5-20` `VS14` |
| `V15` | validate | `plan_rules.py` (`plan_claims.py`) | claims the facts contradict | 2 | `GC-1` `GC-2` `GC-3` `GC-4` |
| `PF1` | preflight | `extract_facts.py` | the model page reads (its layers through domain-forge's `apply_layer.py`), is the `/vega-pack` page with every required layer, and its `bundle` layer's manifest equals `MANIFEST.json` (canonical JSON; a NaN or an infinity in it is named and refused) | 2 | `D5-23` `VS18` |
| `PF2` | preflight | `extract_facts.py` | `MANIFEST.json` and every pinned record read, each matching its `MANIFEST.files` digest; the bundle pins the spec and a record of every required layer; the evaluation records a network estimate of the primary; `MANIFEST.json` holds no NaN or infinity; the fact set validates `show-facts-1.schema.json` | 2 | `D5-22` |
| `PF3` | preflight | `extract_facts.py` | `predict.py --check` passes from a clean shell | 2 | — |
| `PF4` | preflight | `extract_facts.py` | the bundle has a service and `MANIFEST.serve.showcase_slot` | 2 | `D5-24` `VS24` |
| `PF5` | preflight | `build_showcase.py` | `TEMPLATE.json` `requires_endpoints` ⊆ `MANIFEST.serve.endpoints` | 2 | — |
| `PF6` | preflight | `extract_facts.py` | the report's labels read and are current: the label-role map exists and was written from the report's current `.aux` (`ERROR: … recompile /document-project`, never a fork), `--report` names a compiled report (a PDF with a same-stem `.aux`), the `.aux` parses | 2 | `D5-29` `VS15` |
| `PF7` | preflight | `extract_facts.py` | the task kind, family, every block type (composites' `inner` blocks included) and the sample shape are known to the vega-forge catalog; the primary has a `max` or `min` direction there; a `sequence-*` task has a base kind its `sequence_map` knows; the vega-show catalog has a view key for the task; the spec names at least one network input | 2 | `D5-36` `D5-37` |
| `PF8` | preflight | `extract_facts.py` | `MANIFEST.roundtrip.status` is not `fail` | 2 | `VS25` |
| `PF9` | preflight | `build_showcase.py` | `MANIFEST.files` pins no `showcase/` path | 2 | `VS27` |
| `BD1` | build | `build_showcase.py` | `page/` is a byte copy of the `TEMPLATE.json` files | 2 | `PN4` |
| `BD2` | build | `build_showcase.py` | `page.json` passes `check_page_config` | 2 | `PN7` |
| `BD3` | build | `build_showcase.py` | extensions per §5.2 | 2 | `D5-25` |
| `BD4` | build | `build_showcase.py` | the path invariants of §5.3 | 2 | — |
| `BD5` | build | `build_showcase.py` | `served ⊆ files`, and the entry is a served page | 2 | `D5-26` |
| `BD6` | build | `build_showcase.py` | `SHOWCASE.json` validates `vega/showcase@1` | 2 | — |
| `BD7` | build | `build_showcase.py` | a `qps-ploc` build writes only under `.work/qps/` | 2 | — |
| `BD8` | build | `build_showcase.py` | nothing written outside `showcase*/`; `MANIFEST.json`'s bytes unchanged | 2 | — |
| `CK1` | check | `showcase_check.py` | `SHOWCASE.json` reads and is `vega/showcase@1` | 2 | — |
| `CK2` | check | `showcase_check.py` | every `files` digest matches | 2 | `D5-32` `PN4` |
| `CK3` | check | `showcase_check.py` | extensions, path invariants, no unlisted media file | 2 | `D5-25` |
| `CK4` | check | `showcase_check.py` | `served ⊆ files` | 2 | `D5-26` |
| `CK5` | check | `showcase_check.py` | the bundle manifest digest matches, unless `CK6` applies | 2 | `D5-27` |
| `CK6` | check | `showcase_check.py` | the bundle was not re-emitted after the showcase | 3 | — |
| `CK7` | check | `showcase_check.py` | the report sources are unchanged | 3 | `D5-28` |
| `CK8` | check | `showcase_check.py` | the brief sources are unchanged (brief-link mode) | 3 | — |
| `CK9` | check | `showcase_check.py` | no report appeared where the showcase links briefs | 3 | — |
| `CK10` | check | `showcase_check.py` | the template is the one built with (`WARN:` only) | 0 | — |
| `SL1` | slot | `serve/routes_showcase.py` | the schema is `vega/showcase@1` | — | `SS9` |
| `SL2` | slot | `serve/routes_showcase.py` | the manifest digest and model version match the running bundle | — | `SS2` `D5-27` |
| `SL3` | slot | `serve/routes_showcase.py` | the bundle was not re-emitted after the showcase | — | — |
| `SL4` | slot | `serve/routes_showcase.py` | no absolute path, no `..`, nothing outside the directory | — | `SS3` |
| `SL5` | slot | `serve/routes_showcase.py` | no symlink | — | `SS6` |
| `SL6` | slot | `serve/routes_showcase.py` | every `files` digest matches | — | `SS1` `D5-32` |
| `SL7` | slot | `serve/routes_showcase.py` | the total is within the byte cap | — | — |
| `SL8` | slot | `serve/routes_showcase.py` | no unlisted media file outside the unpinned directories | — | `SS4` |
| `SL9` | slot | `serve/routes_showcase.py` | `served ⊆ files`, and the entry, story and report are served | — | `D5-26` |
| `SL10` | slot | `serve/routes_showcase.py` | `MANIFEST.files` pins no `showcase/` path | — | `SS7` |
| `SL11` | slot | `serve/routes_showcase.py` | a showcase file never shadows a service route | — | `SS5` |
| `LY1` | layer | `apply_show_layer.py` | `checks.status` is `pass` | 2 | `D5-34` |
| `LY2` | layer | `apply_show_layer.py` | the served smoke passed, or `WARN: served page not checked` | 2 | `VS21` |
| `LY3` | layer | `apply_show_layer.py` | the abstract is present, long enough and digit-free | 2 | `D5-33` |
| `LY4` | layer | `apply_show_layer.py` | brief links carry `WARN: no report` | 2 | `D5-35` |
| `LY5` | layer | `apply_show_layer.py` | `SHOWCASE.json` pins the current bundle manifest | 2 | — |
| `LY6` | layer | `apply_show_layer.py` | data and render within their caps | 2 | — |
| `LY7` | layer | `apply_show_layer.py` | `--out` differs from the input | 2 | `VS22` |
| `PG1` | page | `tests/test_page_template.py` | `TEMPLATE.json` equals `write_manifest()` | 1 | — |
| `PG2` | page | `page_lint.literals` | no prose literal outside the strings table | 1 | `PN1` |
| `PG3` | page | `lint_templates.py` | no foreign literal or harvested lane token | 2 | `PN1` `D5-31` `VS20` |
| `PG4` | page | `tests/test_page_template.py` | no role literal in code or CSS | 1 | `PN3` |
| `PG5` | page | `tests/test_page_template.py` | strings and explain prose pass the digit rule | 1 | `PN2` |
| `PG6` | page | `tests/test_page_template.py` | key discipline | 1 | `VS19` |
| `PG7` | page | `tests/test_page_template.py` | no `{{…}}` token | 1 | `D5-30` |
| `PG8` | page | `style_lock.py` | the style lock holds | 1 | `PN10` |
| `PG9` | page | `tests/test_page_size.py` | PS1–PS5 | 1 | `PN11` |
| `PG10` | page | `js/core/lane.js (probe)` | `page.json.lane` agrees with `/v1/network` | 1 | `PN8` |
| `PG11` | page | `tests/coverage.test.js` | every block type has a stage kind and a renderer | 1 | `PN9` |
| `PG12` | page | `page smoke (qps-ploc)` | no visible text outside the keyed strings and the data | 1 | `VS26` |
| `SV1` | served | `serve_probe.py` | the service is ready and the slot verified | 2 | — |
| `SV2` | served | `serve_probe.py` | every view renders without a page error | 2 | — |
| `SV3` | served | `serve_probe.py` | no unresolved placeholder | 2 | `D5-32` |
| `SV4` | served | `serve_probe.py` | every report or brief link answers with its pinned digest | 2 | — |
| `SV5` | served | `serve_probe.py` | alert colour iff an alert role; `data-motif` equals `page.json.motif.id` | 2 | — |
| `CS1` | catalog | `tests/test_catalog_sync.py` | the contract, the catalog, the other catalogs and the schemas agree | 1 | — |
| `CS2` | catalog | `tests/test_catalog_sync.py` | every schema validates, and its examples validate | 1 | — |
| `CS3` | catalog | `tests/test_catalog_sync.py` | every report role is named in the document-project patch's label-role map section | 1 | `D5-39` |

Two design cases are caught outside vega-show: `D5-38` (a metric in `METRICS` without a direction) by the vega-forge catalog sync, and `D5-40` (a function of 31 statements in a vega-show script) by `conformance.py --size-only vega-show/scripts` (C11).

## 11. Schemas

| schema | file | written by | read by |
|---|---|---|---|
| `vega/showcase@1` | `references/schemas/showcase-1.schema.json` | `build_showcase.py` | the slot, `showcase_check.py`, `apply_show_layer.py`, `/vega` |
| `vega/story@2` | `references/schemas/story-2.schema.json` | `resolve_story.py` | the page, `page_config.py`, `apply_show_layer.py` |
| `vega/story-plan@2` | `references/schemas/story-plan-2.schema.json` | `draft_plan.py` and the model | `plan_rules.py`, `resolve_story.py` |
| `vega/show-facts@1` | `references/schemas/show-facts-1.schema.json` | `extract_facts.py` | `draft_plan.py`, `plan_rules.py`, `resolve_story.py`, the brief |
| `vega/label-roles@1` | `references/schemas/label-roles-1.schema.json` | `document-project` (patched), lanekit `report_stub` | `role_map.py`, `facts_report.py` |
| `vega/page@1` | `references/schemas/page-1.schema.json` | `page_config.build_page_config` in `build_showcase.py` | the page, `smoke_page.py` |
| `vega/attribution@2` | `references/schemas/attribution-2.schema.json` | `attribution_build.py` | the page (the weights read) |
| `vega/page-probe@1` | `references/schemas/page-probe-1.schema.json` | the page's `probe.js` | `smoke_page.py`, `serve_probe.py`, the lanes' page smoke |
| `vega/page-template@1` | `references/schemas/page-template-1.schema.json` | `page_template.write_manifest` | `page_template.verify_template_copy`, `build_showcase.py`, `showcase_check.py` |

Every schema is JSON Schema draft 2020-12, with `$id` `urn:vega:schema:<name>:<version>`, a `schema` constant equal to its id, and at least one complete example under `examples` that validates. The story's blocks are `$defs` of `story-2.schema.json`: `lexicon` (`vega/lexicon@1`), `class_roles` (`vega/class-roles@1`), `input_groups` (`input_groups@2`), `focus` (`focus@1`), `breakdown` (`breakdown@1`) and `code_note`.
