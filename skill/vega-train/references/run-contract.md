# vega-train contract — the campaign, the runs, the ladder, the `training` layer

`/vega-train` turns a spec into a **campaign**: a baseline run of the spec as written, an ablation ladder of one-change runs each stating the hypothesis it tests, and an optional hyper-parameter search under a stated budget — every run a *delta on the spec* with a source, every epoch a record, every selection a rule. The campaign directory is the truth: the trainer writes exactly what the records say, the notebook replays it, the layer shows it, and `/vega-eval` reads the selected run from it. Nothing about training exists outside the campaign.

## 1. The campaign directory

```
vega-runs/<UTC-timestamp>/                 ← the run directory /vega-forge created; the campaign lives in it
  netspec.json  <package>/  spec_snapshot.json  requirements.txt  smoke.json  engineer.ipynb   ← the forge's emitted package (code-contract.md §2): model/network.py, ingestion/data.py, training/loop.py, model/metrics.py, ports.py, domain.py, blueprint-map.json …
  campaign.json                            ← §2: input digests, budget, device, backend version, seed, status
  ladder.json                              ← §3: the planned runs with hypotheses, sources, deltas
  experiments.json                         ← §4: the ledger — every run's outcome, ranked; the selection
  runs/<run-id>/                           ← one per run: R0 (baseline), R1 (the floor), R1b (the recurrent floor), R2… (the other rungs), H<study>-<n> (HPO trials), <id>b, <id>c… (a run that supersedes <id>)
    run.json                               ← the delta applied, the resolved spec digest, the code digests, the checkpoint digest, status, timings, stopped_by
    netspec.resolved.json                  ← the spec with the delta applied, verbatim (what this run actually trained; /vega-pack copies it)
    <package>/                             ← the full re-emitted package when the delta touched the architecture; otherwise absent and run.json's code digests point at the campaign root's package
    history.jsonl                          ← one epoch record per line (backend-contract.md §1)
    grad_stats.jsonl                       ← per-step pre-clip gradient norms, sampled (every k steps)
    first_step.json                        ← ‖g‖, ‖Δθ‖ and their ratio for the head's weights at epoch 1, step 1 (backend-contract.md §1)
    checkpoints/best.pt  last.pt           ← the backend's native format, written through the ModelStore port (`ModelRegistry.save(model, path, spec=, epoch=)` at each improvement); best.pt's digest is in run.json and in experiments.json
    predictions.valid.parquet              ← the valid part scored from best.pt, in the layout /vega-eval stores the test part in (`vega/predictions-layout@1`, eval-contract.md §2, with `vega.part: valid` and `vega.score_source` in its metadata; `training-catalog.md` §1)
    figures/                               ← the curves as SVG (loss, metric, lr, grad norm, update ratio)
    log.txt
  hpo/<study>/space.json  trials.jsonl  importance.json
  train.ipynb                              ← executed (notebook-contract.md §2–3); train.remote.ipynb with --emit-only
  training-layer.json                      ← the judgement the skill authors; spliced into the layer
  test_seal.json  eval/<run-id>/  evaluate.ipynb   ← written later by /vega-eval (eval-contract.md); bundle/ is written by /vega-pack next to the model, not here
```

A campaign is **append-only**: a run is never deleted or overwritten; a campaign is planned once, on the forge's fresh run directory — `plan_ladder.py` refuses a directory that already holds `campaign.json` (a second plan would rewrite `campaign.json` and `ladder.json`, the pre-registered selection fork included) and writes the two together, only once the ladder planned, so a refused plan leaves the directory as the forge left it: continuing a campaign is `trainer.py --resume`, changing its ladder before any run is an edit of `ladder.json`'s rungs, and once a run exists the ladder only grows — a rung a fired diagnostic proposes is appended with its own hypothesis, source, delta and `expect_check`, and no planned rung is rewritten or removed, and a new campaign is a new forge run directory or `--after-eval`; a re-run is a new run id with `supersedes` set. `--resume` has two cases: a run whose `status` is `running` (interrupted before it finished — nothing about it was read or published) continues under its own id from `last.pt`; a run whose status is `done` or `stopped_by_budget` is never reopened — `--resume <id>` then creates a new run id with `supersedes: <id>`, warm-started from that run's `last.pt`, and its own records. It is the **same rung**, not a new one: the trainer reads the superseded id's rung in `ladder.json`, so the new run carries that rung's `hypothesis`, `source`, `delta` and `expect` into its own ledger row, and `supersedes` names the run it continues; a run id with no rung in `ladder.json` is refused (§3, *Every run is a rung*). The new id is `<id>b`, `<id>c`, … — the first letter suffix that no ladder rung, ledger row or run directory holds and no ladder rule owns (`catalog.json` `run_ids`: R1 is always the floor and R1b the recurrent floor, so a resumed R1 becomes R1c, never R1b); the trainer refuses a target run directory that already exists. **How a run trains**: `trainer.py` resolves the package's adapters by blueprint-map role (`vega_core.PackageAdapters.load`), builds the network through `Network.build(spec)`, and calls the Trainer port's `fit` with the run's `RunContext` (`training-catalog.md` §1 gives the call; `backend-contract.md` §1 the value object); it writes `netspec.resolved.json` (the spec as this run resolved it, the one a replay and `/vega-pack` read), `run.json.device` from the Trainer module's `optim.device_for(spec)` (`PackageAdapters.device_for`), and every `history.jsonl` line from the epoch record's `as_dict()`. A delta on a block parameter the spec leaves unstated is read from the value the block is built with (`stated_defaults`), never from a guess. `--emit-only` writes everything above except the runs, plus `run_campaign.sh` and `train.remote.ipynb`, for execution elsewhere; `--import-run <dir>` reads a campaign executed elsewhere and continues at the procedure's reading step (SKILL.md step 3, the readings; then selection).

**The test part stays sealed.** No run, diagnostic, selection or notebook cell of this stage reads the test part: the data adapter's `load_part(spec, "test")` raises before any file is opened, and the campaign carries the test part only as `"sealed"`. For a `graph` sample the training view is built by one rule: under the `inductive` protocol (the default) the training graph holds the train and valid nodes and the edges among them only, so no test node and no edge to one appears in any loader; under `transductive` — allowed only with a `user:` source and a declared `sample.exposure` (`netspec-contract.md` §1) — the test nodes' **feature columns** and their edges enter the training graph through the adapter's one named door, `open_features_only`, and their labels never do: no loss, metric, diagnostic or selection reads a test label, and the exposure is on the record — `campaign.json.exposure` (`{protocol: "transductive", declared: <the spec's sample.exposure>, what}`, written by `plan_ladder.py`; `null` under every other protocol), spliced into the layer with the campaign, and `apply_training_layer.py` refuses a layer whose campaign drops it or claims one the spec does not declare.

**A campaign after the test was seen.** `--after-eval` does not reopen this campaign: it creates a new run directory `vega-runs/<new-ts>/`, copies `netspec.json` and the emitted code from the old one (digests recorded), writes `campaign.json.after_eval: { "previous_campaign": "<dir>", "exposures": n }`, and takes the `.vega.html` (the architecture file) as its input — the old campaign, its evaluation and its bundle stay on disk untouched; the new campaign's output defaults to `<input-stem>.train-<n>.html` with `n` the exposure count (`orders….vega.train-1.html` after one look), a sibling file, never a replacement of the first `.train.html`.

## 2. `campaign.json`

```json
{ "schema": "vega/campaign@1", "run_dir": "vega-runs/2026-09-05T11-03Z/",
  "input": { "model": "orders.domain.analysis.shaped.vega.html", "model_digest": "sha256:…", "netspec_digest": "sha256:…",
             "package": "orders_pipeline",
             "code_digests": { "orders_pipeline/model/network.py": "sha256:…", "orders_pipeline/ingestion/data.py": "sha256:…", "orders_pipeline/training/loop.py": "sha256:…", "orders_pipeline/model/metrics.py": "sha256:…" },
             "parts": { "train": { "path": "shaped/orders.train.parquet", "digest": "sha256:…", "rows": 420 },
                        "valid": { "path": "shaped/orders.valid.parquet", "digest": "sha256:…", "rows": 60 }, "test": "sealed" } },
  "seed": 7, "device": "cpu", "backend": { "name": "pytorch", "version": "2.4.1" }, "determinism": { "tolerance": 0 },
  "budget": { "wall_clock_min": 20, "spent_min": 14.2, "trials": null }, "mode": "local | emit-only | imported",
  "status": "planned | running | done | stopped_by_budget", "test_seal": null, "after_eval": null,
  "exposure": null,                                                  ← a transductive graph's disclosed test-feature exposure (§1), else null
  "protocol": { "blueprint": "stratified hold-out 70/10/20 (analysis importance protocol)", "applied": "the shaper's split", "source": "blueprint:training | train:protocol-override", "reason": "…" } }
```

`test_seal` is `null` until `/vega-eval` writes `test_seal.json` beside this file and sets it to that path. The trainer refuses to start in a campaign directory whose `test_seal.json` exists; `--after-eval` starts a **new** campaign (§1) with `after_eval` set — training after the test was seen never reopens the campaign that saw it, and the flag records that the user knows.

## 3. `ladder.json` — the runs and their hypotheses

```json
{ "schema": "vega/ladder@1",
  "runs": [
    { "id": "R0", "kind": "baseline", "delta": [], "hypothesis": "the spec as engineered; establishes the curves every ablation is read against",
      "source": "netspec", "expect": "valid pr_auc ≥ 0.60 by epoch 30, train/valid loss gap < 0.1", "expect_check": { "relation": "learns" } },
    { "id": "R1", "kind": "ablation", "family": "architecture", "delta": [ { "path": "family/name", "from": "mlp-residual", "to": "linear" }, { "path": "blocks", "to": "<the catalog's linear composition, with the shape's insert>" } ],
      "hypothesis": "the non-linear structure F5 found is real: the linear floor must be clearly below R0", "source": "vega:ladder/floor",
      "expect": "valid pr_auc(R1) < pr_auc(R0) − 0.05", "expect_check": { "relation": "worse", "metric": "valid.primary", "than": "R0", "margin": 0.05 } },
    { "id": "R2", "kind": "ablation", "family": "regularization", "delta": [ { "path": "regularization[kind=dropout]/value", "from": 0.2, "to": 0.0 } ],
      "hypothesis": "F4: 420 rows overfit without dropout — removing it should widen the train/valid gap", "source": "analysis:F4",
      "expect": "gap(R2) > gap(R0)", "expect_check": { "relation": "worse", "metric": "gap.loss", "than": "R0" } },
    { "id": "R3", "kind": "ablation", "family": "optimization", "delta": [ { "path": "optimizer", "to": { "name": "sgd", "params": { "lr": 0.01, "momentum": 0.9 } } } ],
      "hypothesis": "AdamW vs SGD on this data (the spec's first alternative)", "source": "netspec:optimizer/alternatives[0]", "expect": "…" },
    { "id": "H1", "kind": "hpo", "study": "H1", "space": "hpo/H1/space.json", "budget": { "trials": 20, "wall_clock_min": 8 },
      "hypothesis": "lr, width and dropout interact; a TPE search around R0 finds the region", "source": "vega:ladder/hpo", "expect": "best trial ≥ R0 on valid pr_auc",
      "expect_check": { "relation": "not_worse", "metric": "valid.primary", "than": "R0" } }
  ],
  "forks": [ { "asked": "ladder R1–R3 + H1 under 20 min; selection: best valid pr_auc (rule ladder/selection), alternative: the user's pick with a reason", "answer": "approved (user)", "or": "default ladder applied (unattended)" } ] }
```

A run's `delta` is a list of edits on `netspec.json` in the **delta path syntax**: a slash path of keys (`optimizer/params/lr`), a block by id (`blocks/B3/params/units`, `blocks/B3/params/inner/0/params/units`), a list entry by a key's value (`regularization[kind=dropout]/value`), or a whole section (`optimizer`, `blocks`). **One change** means one *decision*: every path of an ablation's delta must fall under one decision family of the spec (`netspec.decision_families`), and the rung declares that family — a family swap legitimately touches `family` and `blocks` (both `architecture`); a rung that touches `optimizer` and `regularization` at once is two rungs, and `plan_ladder.py` and `trainer.py` refuse it. A delta on `backend` is refused outright: the backend is the forge's choice, not an experiment. **Every run is a rung**: `trainer.py` refuses a run id that is neither a rung of `ladder.json` nor a trial `H<study>-<n>` of a planned `hpo` rung — `--delta` included, which carries a trial's sampled point or the planned rung's own delta and nothing else (a different change is a different rung, planned with its own hypothesis); a study's own id is not a run, and `hpo.py` refuses a study the ladder does not plan. **HPO trials are runs like the others**: `hpo.py` samples a point of the study's space into a delta and calls `trainer.py --run H<study>-<n> --delta <json>` for it, so every trial recompiles, records digests and diagnostics, and carries the study rung's `hypothesis` and `expect` (evaluated once for the study, as `expect_held` on the study's row; the trials' rows say `expect: inherited from H<study>`); a trial's delta may span decision families, because it samples a declared space and its source says so. The trainer applies the delta, recompiles through `compile_spec.py --check-only` (an ablation that produces an invalid spec is refused before it runs), re-emits the package into `runs/<id>/<package>/` when the delta touches the `architecture` family (and passes `conformance.py` on it — `code-contract.md` §5, mechanical half), writes `netspec.resolved.json`, and records the resolved spec, code and checkpoint digests. **The floors** (`floors.py`) are built as `training-catalog.md` §2's `floor` and `recurrent-floor` rows say — R1 and R1b, each a one-change delta of the `architecture` family, or dropped with a `WARN:` saying why. A rung's `source` is one of: `netspec` (the spec as engineered — R0 only), `netspec:<path>` (an alternative the spec itself listed, by the path of the choice), `vega:ladder/<rule>` (a ladder rule of `training-catalog.md` §2), `train:diagnostic/<code>` (a diagnostic of `training-catalog.md` §3 fired on an earlier run — the run id is named), `train:hpo/<study>` (a trial), `analysis:F<n>` (a finding that predicts the rung's effect), `user:<turn|stance>`; nothing else. **A delta must reach the weights.** `spec_snapshot.json`'s `read_paths` (netspec-contract §3, rule R2) lists the regularization paths the emitted code reads, and `_resolved.regularization.shadowed_paths` the live paths a declaration covers; a delta on a `regularization[]` entry no emitted code reads, or on a shadowed live path, is a **dead delta** — it would train its baseline's numbers bit for bit. `plan_ladder.py` refuses a rung on one — the plan stops with `ERROR: dead delta — no emitted code reads <path>` and writes nothing, since the forge proposed a default that moves no weight — and drops a search dimension on one with a `WARN:`; `trainer.py` refuses it before anything is written (`ERROR: <id>: dead delta — no emitted code reads <path>`, exit 2). A snapshot written before R2 records no `read_paths`: its templates read only `grad_clip` of the declaration, and the scripts judge by that; a directory with no snapshot is judged by the forge's own derivation on its `netspec.json` (`vega_regularization.py`); either way with a `WARN:` saying so (`scripts/live_paths.py`). **The expectation is checkable.** `expect` is the prediction the hypothesis makes, in words; `expect_check` is the same prediction in the closed grammar of `scripts/expectations.py` — `{ "relation": "learns" }` (no D1 or D2, and an improvement on the first epoch); `{ "relation": "better" | "worse" | "not_better" | "not_worse", "metric": "valid.primary" | "valid.loss" | "train.primary" | "train.loss" | "gap.loss", "than": "<run id>", "margin": <fraction of the reference's value, default 0> }`, compared in the metric's direction (the primary in `metrics.selection.mode`, a loss and the train/valid loss gap downwards — a wider gap is worse); or `{ "any": [<check>, …] }`. `plan_ladder.py` writes every rung's from its rule's row (`catalog.json` `ladder_rules.<rule>.expect_check`, each rule's check given in words in `training-catalog.md` §2), with its `text`; the skill edits the check with the words, before the run. `trainer.py` refuses a rung whose expectation it cannot check — a `user:` or `analysis:` rung states its own `expect_check`, every other source names a rule that has one — and the ledger records whether it held (§4).

## 4. `experiments.json` — the ledger and the selection

```json
{ "schema": "vega/experiments@1",
  "runs": [
    { "id": "R0", "status": "done", "epochs": 47, "stopped_by": "early_stopping", "best_epoch": 37, "time_s": 41.0,
      "valid": { "pr_auc": 0.71, "roc_auc": 0.84, "loss": 0.41 }, "train": { "pr_auc": 0.79, "loss": 0.33 }, "gap": { "loss": 0.08 },
      "params": 21313, "diagnostics": ["D3"], "unavailable": [], "expect": "valid pr_auc ≥ 0.60 by epoch 30, train/valid loss gap < 0.1",
      "expect_check": { "relation": "learns", "text": "the run learns: no D1/D2 and an improvement on its first epoch" }, "expect_held": true,
      "expect_evidence": "R0 learned (best epoch 37, no D1/D2)",
      "spec_digest": "sha256:…", "checkpoint_digest": "sha256:…", "code_digests": { "orders_pipeline/model/network.py": "sha256:…" }, "reading": "<one paragraph>" }
  ],
  "ranking": { "by": "valid.pr_auc", "mode": "max", "order": ["H1-14", "R0", "R3", "R2", "R1"] },
  "selection": { "run": "H1-14", "rule": "best valid primary in the direction of metrics.selection.mode; ties → fewer params; ineligible: a run whose diagnostics include D1 (divergence) or D2 (not learning), a run stopped by budget before its first improvement (D9), a pruned trial",
                 "source": "vega:default", "catalog": "ladder/selection", "override": null, "alternatives": [ { "run": "R0", "when": "if the user prefers the un-searched spec for interpretability" } ],
                 "margin_over_baseline": 0.03, "reading": "<why this run, and what it did not settle>" },
  "hpo": [ { "study": "H1", "trials": 20, "completed": 18, "pruned": 2, "best": "H1-14", "best_value": 0.74, "importance": { "lr": 0.61, "width": 0.24, "dropout": 0.15 },
             "expect": "best trial ≥ R0 on valid pr_auc", "expect_check": { "relation": "not_worse", "metric": "valid.primary", "than": "R0" }, "expect_held": true,
             "expect_evidence": "valid.pr_auc H1 0.74 against R0 0.71: held" } ],
  "budget": { "wall_clock_min": 20, "spent_min": 14.2 } }
```

**Who writes the ledger.** Scripts only. `trainer.py` appends a run's row when the run ends — its numbers, digests and diagnostics, its rung's `expect` and `expect_check`, and `expect_held`: the check judged against the ledger the row joins, with `expect_evidence` naming the numbers compared. A check whose reference run has not run yet is `null` with that reason, and every pending row and study is judged again whenever another row lands — never by reading the curves and deciding. A trial's row says its expect is inherited; the study's is judged once, on its record in `hpo[]`, by `hpo.py`. A diagnostic a run could not be checked for when it ended stays `unavailable` with its reason in that row (D4 on a run that finished before the floor: the run's reading compares it with the floor's ledger value in words once the floor exists, citing both keys). `scripts/select.py` writes `ranking` and `selection` — the rule's pick, its eligibility reasons, `margin_over_baseline`, and an override only from `--override <run> --reason … --source user:<turn>`. `apply_training_layer.py` judges every record again and refuses a ledger whose `expect_held` is not its check's verdict on that ledger, or a rung or study with no check: the verdict is the check's, never written or edited by hand. The run's reading in `training-layer.json` (`readings.per_run`) argues the words beside the verdict.

The **selection rule** is a decision like any other, written into the ladder's fork record (`ladder.json` `forks[].asked` names it and its alternative) before any run: `vega:default` citing the catalog's `ladder/selection` row, with its alternative ("the user's pick with a reason") declared in the ladder fork before any run, and shown again as the last `FORK:` of the campaign (attended) or applied and marked (unattended). No script applies a rule other than the catalog's: `select.py` computes that one rule, and any other choice — one the user states at the planning fork included — is the declared alternative, applied through `select.py --override` with its `user:` source and reason, never by hand. Any pick that is not the rule's — an eligible run the user prefers as much as an ineligible one — is recorded in `selection.override` with its `user:` source and reason. *Before its first improvement* is mechanical too: the run's `best_epoch` is missing or 1 — it never bettered its first epoch on the valid primary. A run cut by the budget **after** an improvement stays *eligible for selection* — its best checkpoint is what was measured — but it is never read as a converged result: D9's reading travels with it, and when the rule selects it the selection's reading, the layer and the hand-off say it may not have converged and name a longer budget as the open question. A campaign whose baseline R0 itself fires D9 — cut before early stopping — cannot afford the baseline to converge: it stops after R0, plans no further rung, and its summary says the budget, not the spec, decided; no ablation is ever read against an unconverged baseline. Ineligibility is mechanical: a run that diverged, never learned, was cut by the budget before its first improvement, or is a pruned trial cannot be selected *by the rule*; the user may still select it, and then `selection.override` is `{ "source": "user:<turn>", "reason": "…" }`, the layer shows the override beside the rule, and the apply script accepts the ineligible run only when that override is present.

## 5. The `training` layer

```html
<!-- @LAYER:start training v1
     produced-by: /vega-train
     produced-at: <UTC ISO timestamp>
     input-digest: sha256:<digest of the canonical Turtle script>
     reverts-by: open the predecessor
 -->
<script id="layer-training-data" type="application/json"> { ...below... } </script>
<script id="layer-training-render" type="text/javascript"> (function(){ ... })(); </script>
<style id="layer-training-style"> .layer-training ... </style>
<!-- @LAYER:end training -->
```

`layer-training-data`:

```json
{
  "schema": "vega/training@1",
  "campaign": { "...": "campaign.json verbatim" },
  "ladder": { "...": "ladder.json verbatim" },
  "experiments": { "...": "experiments.json verbatim" },
  "histories": { "R0": { "path": "vega-runs/…/runs/R0/history.jsonl", "downsampled_from": 47, "epochs": [1, 2, "…"], "train.loss": ["…"], "valid.loss": ["…"], "valid.pr_auc": ["…"], "lr": ["…"], "grad_norm.global": ["…"], "update_ratio.B3": ["…"] } },
  "diagnostics": [ { "run": "R0", "code": "D3", "name": "overfitting after best epoch", "evidence": "valid loss +0.06 over 10 epochs after epoch 37 while train loss −0.05", "reading": "…", "led_to": "R2" } ],
  "mechanics": { "loss": "L = −[4.9·y·log σ(z) + (1−y)·log(1−σ(z))]", "update_rule": "AdamW: … with η=1e-3, β=(0.9, 0.999), λ=1e-4, ε=1e-8",
                 "schedule": { "name": "cosine", "warmup_epochs": 3, "lr_by_epoch": ["…"] }, "clip": 1.0, "one_step": "<what epoch 1 step 1 did to the head's weights: ‖g‖, ‖Δθ‖, the ratio>" },
  "notebook": { "path": "vega-runs/…/train.ipynb", "executed": true, "digest": "sha256:…", "replay_epochs": 3, "replay_ok": true },
  "forks": [ "…ladder fork, selection fork…" ],
  "markers": ["OK: …", "WARN: …"],
  "readings": { "abstract": "<paragraph: what was run, what the ladder established, which run was selected and why, what remains open>",
                "per_run": { "R0": "<…>" }, "hpo": "<…>", "selection": "<…>" }
}
```

Histories are **downsampled** — the points per series scale with the run count so that the layer stays ≤ 400 KB of data whatever the campaign's size (a ten-rung ladder with an eight-trial study is the case that broke a fixed 200), drawn values rounded to 6 significant digits; the full `history.jsonl` stays on disk, each history's `path` names it, and the notebook asserts against the file, not the layer. Required: `schema`, `campaign`, `ladder`, `experiments` (with `selection.run`), `histories` for every run in `experiments.runs` (each with `path`), `mechanics`, `notebook` (with `executed`; `false` accompanied by `WARN: notebook not executed` in `markers`), `readings.abstract`, `readings.selection`. The authoring file (`training-layer.json`) carries `from_campaign`, consumed by the apply script and not stored, naming the campaign directory whose records it splices in. `scripts/apply_training_layer.py` validates this, refuses a layer whose `selection.run` is ineligible without a recorded `override` or whose notebook replay failed, writes the `notebook` record itself from `--notebook <path>` (notebook-contract.md §3), injects the shipped `assets/training-render.js` + `assets/training-layer.css` through `domain-forge/scripts/apply_layer.py`, and adds a `<noscript>` fallback (the ledger table, the selection, the readings).

The render mounts one tab, `data-layer="training"`, labelled **Training**: the ledger table (run, kind, hypothesis, delta, valid primary, gap, epochs, stopped by, expect held — the ledger's field, null unless a script computed it, beside the run's reading that says whether it held — diagnostics), a run selector with the curves (train/valid loss and primary metric, learning rate, global and per-block gradient norms, update ratios — SVG, drawn from `histories`), the diagnostics panel linking each fired diagnostic to the run it led to, the HPO panel (trials on the primary metric, parameter importance), the mechanics panel, the selection with its rule and alternatives, the forks. It reads only `layer-training-data`, idempotent, empty state when missing, ≤ 60 KB of JS, SVG only, keyboard-operable, `window.__training = { runs(), history(id), selection(), version }` for tests.

## 6. Remote execution

`--emit-only` writes the campaign directory with `ladder.json`, `run_campaign.sh` (runs `trainer.py` for every ladder run then the HPO study, in order, resumable) and `train.remote.ipynb` whose cells train every run in full; the user executes either on the remote machine and brings the directory back. `--import-run <dir>` verifies the campaign's `netspec_digest`, code digests and part digests against the local ones, checks every `history.jsonl` for the record schema and monotone epochs, marks `campaign.mode: imported` with the remote device and backend version, and continues at the procedure's reading step (step 3's readings, then selection); the ordinary `train.ipynb` is then emitted from the imported records. Every check refuses on failure with `ERROR:`, nothing imported: a digest mismatch (the spec, the root package's code, a run's re-emitted package, a part), a `history.jsonl` line that is not an epoch record or epochs that are not monotone, a `run.json` whose checkpoint digest is not its `checkpoints/best.<ext>` file's — a campaign run on a different spec or different data, or recorded by something other than the trainer, is not this campaign.

## 7. Naming, memory, hand-off

Output: `<input-stem>.train.html` next to the input (`orders.domain.analysis.shaped.vega.html` → `orders.domain.analysis.shaped.vega.train.html`); the campaign in the run directory `/vega-forge` created. Memory: `.claude/domain-forge-memory.md` § **Network stances** — one line per fork resolved (the ladder approved or edited, the selection) and per campaign (`YYYY-MM-DD | <dataset> | campaign <ts>: selected H1-14, valid pr_auc 0.74`). Hand-off: the summary names `/vega-eval <out>` with the selected run, states that the test part has not been opened, and lists what the campaign left open (hypotheses whose `expect` did not hold, budget exhausted before convergence, an HPO importance that suggests a dimension the ladder never varied).

## 8. What proves the trainer covers every cell

The trainer drives every admitted cell of `vega-forge/scripts/catalog.json`'s admission table (`vega-forge/references/family-catalog.md` §7) through the same adapters this contract names: the backend coverage matrix (`vega-forge/scripts/tests/matrix/run_matrix.py --cells all`, report `vega/backend-matrix@1`) fits each cell through `PackageAdapters` → `Network.build` → `Trainer(store).fit(…, run=RunContext(…))` with its best checkpoint written through the ModelStore port the Trainer is given, and requires every epoch finite and a finite best valid primary. A change to this stage's scripts (`trainer.py`, `plan_ladder.py`, `floors.py`, `hpo.py`, `diagnostics.py`, `select.py`) or to the backend's training templates ships only when that report is green (`summary.failed == []`) and its cell list equals the admission table (`admission.table: "equal"`); a family, shape or task this stage claims to train and the matrix does not run is a coverage defect, not a supported case.
