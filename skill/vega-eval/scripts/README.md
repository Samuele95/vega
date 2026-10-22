# `/vega-eval` scripts

The one look at the test part, and everything that reads what it wrote.

| file | what it is |
|---|---|
| `catalog.json` | `references/eval-contract.md` §3–§4 in machine form: the metric row for each of the ten task kinds, which kinds decide at a threshold, which metrics are ranking metrics, and what each baseline is (including the task kinds where it is something else). Every script reads it; none restates it. `tests/test_catalog_sync.py` fails when it and the prose diverge. |
| `vega_eval_core.py` | The seal, the shared bootstrap, and the number-checking the readings are held to. Pure: frozen value objects and functions, no argv, no printing, no exits. `Estimate` has no constructor that produces a point without an interval; `Resampler` draws its indices once so every model is scored on the same resamples; `Seal` is a record with a history, not a convention. |
| `evaluate.py` | The protocol. Opens the emitted package through `vega_core.PackageAdapters` and refuses before sealing, seals before reading, opens the test part exactly once through the DataSource's `load_sealed_part`, scores the network and every baseline through the package Evaluator's `scored` (the raw head from `heads`), fits the baselines on train with valid for their own selection, and writes `eval/<run-id>/` — predictions, unscored, metrics, threshold, baselines, slices, robustness, curves, markers. |
| `predictions_layout.py` | `vega/predictions-layout@1` (R-C2): `PredictionsLayout.of(scored, heads, task_effective, prediction, baselines).write(path)` and `read(path)` → (frame, `vega.*` metadata) for `predictions.test.parquet`; `row_id` per shape (a `graph_id` column at graph level), `y`/`y_k`, the raw head as `score`/`score_k`, `row_score`, `prediction` only where a threshold decides, `score_<baseline>`; `unscored.json` lists the samples that got no prediction and why. |
| `eval_package.py` | The package as this stage drives it: `open_package` (refuses a package of another generation or with no Evaluator port), the default baseline set (`default_when`), the metric options (season, mase scale, quantile levels), the sample frame, the anomaly rate, the bottleneck width, and `lens_protocol` — the family and recorded hyper-parameters the `lens` baseline re-fits. |
| `lens_segments.py` | The analysis's segments as slices: `segment_plan` reads `modules.segments.evidence` (the k-means `profiles[]` and the `assignment` rule) and returns the rule or the plain reason no segment slice can be cut; `raw_test_rows` reads the test part's raw rows after the seal at the shaper's `__split_raw__.test` positions, the columns the rule reads. A slice not cut lands in `slices.json.not_cut` with its reason. |
| `user_slices.py` | The slices the user names, declared before the seal: each literal coerced to the column's dtype (the shaper's `output_schema`, else the train part's schema), and a slice with no value, on a column that is not a feature, or with a literal the dtype cannot hold refused (rule T3) — no test row is read for any of it. |
| `earlier_looks.py` | An after-eval campaign's `campaign.json.after_eval`, reconciled with the previous campaign's own seal (rule S3): its first look is a further exposure, and the seal counts the earlier looks. |
| `metric_gates.py` | The metrics whose meaning depends on the data (the catalog's `metric_gates`: `mape` when \|y\| > 0.1 on 99 % of the rows), decided on the train part's targets and recorded in `metrics.json.gates`; a withheld metric is recorded with its reason, never a number. |
| `campaign_baseline.py` | `campaign-baseline`: R0's `best.pt` predicted through the seal in a child process (R0's own emitted code), aligned to the network's test rows, so it carries a test CI and a paired difference. |
| `threshold_choice.py` | The decision threshold, chosen on valid and on the reading's own scale (the catalog's `threshold_scale`): a probability for binary and multilabel (σ applied unconditionally, a per-label chooser keyed by class name), the score's scale for an anomaly; `check_threshold_arg` refuses a `--threshold` before the seal. |
| `apply_evaluation_layer.py` | Validates the `layer-evaluation-data` document and writes the `evaluation` layer through `domain-forge/scripts/apply_layer.py`. The authoring file carries only judgement; everything measured is spliced from `from_eval`. |
| `smoke_evaluation.py` | Headless-Chromium check that the layer renders: the surfaces mount, every rendered estimate shows its interval, the curves are inline SVG, the slices table sorts by mouse and by keyboard, and `window.__evaluation` returns the record the page was given. |
| `tests/acceptance.sh` | The whole stage on a real trained campaign, then every gate against a seeded defect. |
| `tests/run_negatives.py` | The seeded defects, one at a time, each asserted to be caught by the gate that should catch it. |
| `tests/test_closure_e_eval.py` | The A46 promises without a campaign: slice coercion and its pre-seal refusals (T3), after-eval looks (S3), campaign-baseline's checks (B5), `mape`'s gate (M1), `lens`'s hyper-parameters on the rendered template (B6), and the apply script's threshold-CI, upstream and gate refusals (A1–A3). |
| `tests/test_predictions_layout.py` | The layout on B15's window, node-level (with a transductive variant) and graph-level fixtures, each trained with the package Trainer and evaluated once by the real `evaluate.py`; the pre-seal refusals; the writer's own guards. |
| `tests/test_scoring_threshold.py` | Every headline, slice and robustness number read at the run's threshold, on its scale, one per label; the rendered backend `metrics.py` read the same way. |
| `tests/test_lens_segments.py` | A real /data-lens run's segments cut the test rows (each in the segment the analysis's rule places it), a real published layer without the rule cuts nothing and says why (`fixtures/lens-segments-final3r2.json`), and every other reason a segment slice is not cut is said. |
| `tests/test_catalog_sync.py` | `catalog.json` against the contract's §3–§4, plus the check that every metric the catalog names is one the backend implements. |
| `tests/evaluation-layer.authoring.json` | The acceptance run's judgement: the five readings, with every number the records support. |

The notebook is not here: `nb_emit.py --stage eval` lives in `/vega-forge`, because one script writes every stage's notebook (`notebook-contract.md` §1).

## The order, and why it is the order

```
refuse  →  seal  →  read  →  measure  →  author  →  notebook  →  layer
```

Each arrow is load-bearing.

**Refuse before sealing.** A baseline set below the floor, a user slice that cannot be declared, and — once a seal exists, or the campaign is an after-eval one — a threshold value, user slices, or a baseline set narrower or wider than the last look's, are refused with no seal written. Anything chosen after a look is chosen knowing the answer, so it has to be caught before the look, not after it.

**Seal, then read.** `test_seal.json` is written before the first byte of the test part is read, and the digest is compared with the shaper's manifest immediately afterwards. Deleting the seal does not reset the count: an `eval/` directory with no seal beside it is refused, because the records prove a look that the missing seal would otherwise unremember.

**One look.** The token in the seal is the only key to `load_sealed_part`, and this file holds it. Everything downstream — the notebook, the layer, `/vega-pack` — reads `predictions.test.parquet`. That is why re-executing the notebook costs nothing, and why its last cell checks that the data adapter never entered `sys.modules` rather than merely claiming it did not.

**Every number with an interval.** `Estimate` makes it a type rather than a discipline, `Resampler` makes the *differences* have intervals too, and `apply_evaluation_layer.py` refuses to publish a metric without one. A bare 0.71 on 120 rows tells a reader nothing about whether the network was worth building.

**The readings summarise the records.** Every number in every reading must appear in the spliced records at the rounding the reading used, and the verdict's *earned* sentence must agree with the machine verdict. A prose verdict whose digits have drifted reads as evidence while being a memory of evidence, so it is refused rather than warned about.

## Running it

```bash
python3 evaluate.py --campaign vega-runs/<ts> [--run <id>] [--again "<reason>"]
python3 ../../vega-forge/scripts/nb_emit.py --stage eval --from vega-runs/<ts> \
        --out vega-runs/<ts>/evaluate.ipynb
python3 apply_evaluation_layer.py <model>.train.html \
        --data vega-runs/<ts>/evaluation-layer.json \
        --notebook vega-runs/<ts>/evaluate.ipynb --out <model>.train.eval.html
python3 smoke_evaluation.py <model>.train.eval.html --strict
```

`tests/acceptance.sh` does all of it on the fixture, and will run `/vega-forge`'s and
`/vega-train`'s acceptance first if no trained campaign is there yet.
