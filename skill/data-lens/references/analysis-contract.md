# data-lens report contract — the `analysis` layer

The output file is the input model (a `/dataset-forge` `.domain.html`, with or without later layers, or the minimal base model `/data-lens --standalone` bootstraps) plus **one additive layer** named `analysis`, appended before `</body>` per domain-forge's `references/future-skills.md` layer contract. The block is written by `scripts/apply_analysis_layer.py`, which validates the data document against §1, then calls the platform writer (`domain-forge/scripts/apply_layer.py`) with this skill's shipped `assets/analysis-render.js` and `assets/analysis-layer.css`, and adds a `<noscript>` fallback (the findings list, the evidence tables, the transcript). The render mounts one tab, `data-layer="analysis"`, labelled **Analysis**, as a `section.tab-pane` next to the host's tabs (the convention `/dataset-forge` and `/inferred-questions` use); it reads only `layer-analysis-data`, is idempotent, and renders an explanatory empty state when the data script is missing. It never touches the base runtime or earlier layers. The analyst authors the JSON; nobody authors the JavaScript per run.

```html
<!-- @LAYER:start analysis v1
     produced-by: /data-lens
     produced-at: <UTC ISO timestamp>
     input-digest: sha256:<digest of the canonical Turtle script>
     reverts-by: open the predecessor
 -->
<script id="layer-analysis-data" type="application/json"> { ...schema below... } </script>
<script id="layer-analysis-render" type="text/javascript"> (function(){ ... })(); </script>
<style id="layer-analysis-style"> .layer-analysis ... </style>
<!-- @LAYER:end analysis -->
```

A `--continue` run replaces the `analysis` layer (the platform writer strips the same-named layer and reports it): the base model and every *other* layer are preserved byte-for-byte, the previous file stays on disk as the record of the previous session, and the new layer carries the old turns plus the new ones.

## 1. `layer-analysis-data` schema

The analyst authors the judgement and never retypes a number. A layer document may carry

```json
{ "from_analysis": "<run-dir>/analysis.json", "context": { "reading": "…" }, "findings": [ … ] }
```

and `scripts/apply_analysis_layer.py` splices in, from that file: `source` (path, rows, columns,
seed, geometry), `context` (typing, basis, derivations, partition, time, spatial — the author's
`reading` is kept), every module's `ran` / `skipped_because` / `evidence` (the author's `reading`
is kept), and the SVG of every figure the engine drew, read from its `path` on disk. A figure may
also be given inline as `svg`, or as a `path` to an `.svg` file. What follows is the shape of the
spliced result — the document the apply script validates and writes.

```json
{
  "schema": "data-lens/analysis@1",
  "source": { "path": "orders.csv", "rows": 600, "columns": 15,
              "analysis_json": ".claude/data-lens-runs/<UTC-timestamp>/analysis.json", "seed": 7,
              "geometry": "present | absent",
              "model_digest": "sha256:<the input-digest above>" },
  "markers": ["OK: ...", "WARN: ...", "FORK: ..."],
  "context": {
    "typing": [ { "column": "zip", "type": "nominal", "role": "<one of the roles report-contract.md §1 enumerates>" } ],
    "basis": ["unit_price", "qty", "..."],
    "derivations": [ { "column": "total", "body": ["unit_price", "qty"], "rule_id": "swrl-total" } ],
    "partition": { "label": "late | null", "task": "binary classification | ...", "features": ["..."], "leakage": ["delivered_days"], "provenance": "<copied from the geometry: dataset-forge report-contract §1's vocabulary, not restated here>" },
    "spatial": { "columns": [ { "column": "lat", "pair": "lon", "kind": "latlon | wkt | geohash | h3" } ], "crs": "EPSG:4326 | unknown | null" },
    "time": { "column": "order_date | null", "resolution": "day", "span": ["2024-01-01", "2024-12-31"], "regular": true },
    "reading": "<one paragraph: what the geometry layer already settled and what this analysis therefore does not redo>"
  },
  "modules": {
    "quality":       { "ran": true,  "reading": "<paragraph>", "evidence": { "...": "verbatim from analysis.json" } },
    "distributions": { "ran": true,  "reading": "<paragraph>", "evidence": { } },
    "relations":     { "ran": true,  "reading": "<paragraph>", "evidence": { } },
    "inference":     { "ran": true,  "reading": "<paragraph>", "evidence": { } },
    "segments":      { "ran": true,  "reading": "<paragraph>", "evidence": { } },
    "importance":    { "ran": true,  "reading": "<paragraph>",
                       "evidence": { "task": "binary classification", "features": ["..."], "excluded": { "leakage": ["delivered_days"], "label_derivations": [] },
                                     "baseline": { "kind": "majority", "score": 0.71 }, "metric": "roc_auc",
                                     "models": [ { "name": "logistic-l2", "cv_score": 0.84, "ci": [0.80, 0.88] }, { "name": "gradient-boosting", "cv_score": 0.87, "ci": [0.83, 0.90] } ],
                                     "permutation_importance": [ { "feature": "qty", "mean": 0.12, "ci": [0.09, 0.15] } ],
                                     "leakage_probe": { "suspect": [], "note": "no feature collapses when the label's derivation columns are removed" },
                                     "learning_curve": [ { "fraction": 0.1, "score": 0.74 }, { "fraction": 1.0, "score": 0.87 } ] } },
    "time_series":   { "ran": false, "skipped_because": "no datetime column", "reading": "", "evidence": { } },
    "spatial":       { "ran": false, "skipped_because": "no coordinate pair", "reading": "", "evidence": { } },
    "drift":         { "ran": false, "skipped_because": "no split column and no --split", "reading": "",
                       "evidence": { "parts": ["train", "test"], "columns": [ { "column": "qty", "measure": "psi | ks | cramers_v", "value": 0.03, "drifted": false } ],
                                     "label_rate": { "train": 0.21, "test": 0.19, "risk_difference": 0.02, "ci": [-0.03, 0.07] } } }
  },
  "findings": [
    {
      "id": "F1", "module": "quality", "severity": "high | medium | low",
      "title": "delivered_days is missing in 12% of rows, and missingness depends on zip",
      "columns": ["delivered_days", "zip"],
      "evidence": { "missing_rate": 0.12, "test": "chi-square missingness ~ zip", "statistic": 41.3, "p_adj": 0.0004, "effect": "Cramér's V 0.27", "n": 600 },
      "method": { "name": "Little-style MCAR check by group", "assumptions": ["expected counts ≥ 5"], "assumptions_checked": "passed", "correction": "BH across 14 quality tests" },
      "reading": "<prose: motive, evidence, what it means for this dataset>",
      "so_what": { "preprocessing": "impute within zip group or add a missingness indicator; a global median hides the mechanism", "modeling": "...", "collection": "...", "interpretation": "..." },
      "transformation_candidates": [
        { "id": "T1", "op": "impute", "columns": ["delivered_days"], "params": { "strategy": "group-median", "by": ["zip"], "indicator": true },
          "alternatives": [ { "op": "impute", "params": { "strategy": "median" }, "when": "if the zip dependence is judged spurious" } ],
          "rationale": "F1 — missing not at random across zip" }
      ],
      "figures": ["fig-F1-missing-by-zip"]
    }
  ],
  "figures": [ { "id": "fig-F1-missing-by-zip", "kind": "bar", "title": "...", "svg": "<svg ...>...</svg>", "alt": "<one sentence>" } ],
  "transcript": [
    {
      "turn": 1, "question": "Are the outliers in unit_price real or entry errors?",
      "method": { "name": "MAD outlier scan + lookup of the affected rows", "assumptions": [], "assumptions_checked": "n/a" },
      "code": "<the exact Python executed by scripts/cell.py>",
      "result": { "...": "JSON returned by the cell, verbatim" },
      "figure": "fig-T1-unit-price | null",
      "answer": "<prose composed from the result>",
      "grounded": true,
      "caveats": ["7 rows; the scan cannot distinguish a genuine premium SKU from a decimal-point error — the product dictionary would"],
      "finding_ref": "F4 | null",
      "transformation_candidates": []
    }
  ],
  "stances": [ { "id": "ST1", "kind": "genuine-outliers | missingness-mechanism | out-of-scope | designed-experiment | fact",
                 "columns": ["unit_price"], "assertion": "outliers above 900 are genuine premium SKUs", "source": "user, turn 3" } ],
  "handoff": { "shaper_candidates": ["T1", "T3", "T7"], "note": "<one sentence: what /dataset-shaper would materialize and in what order>" }
}
```

Required keys (the apply script refuses without them): `schema`, `source` (with `path`, `geometry`, `seed`), `context` (with `typing`, `basis`, `partition`), `modules` (every module key present, each with `ran` and, when `ran` is false, `skipped_because`), `findings`, `transcript`, `handoff` (with `shaper_candidates`, an array that may be empty, in application order). `context.typing[].type` is the geometry layer's `final_type` and `role` its `role`, copied; the enumeration is the one `dataset-forge/references/report-contract.md` §1 defines. Datetime and coordinate columns are not roles: they are `context.time` and `context.spatial`, detected here. Transformation-candidate ids (`T<n>`) are unique across the whole layer (findings and turns together), so `handoff.shaper_candidates` resolves without a finding id; a candidate born in a turn is cited by `/dataset-shaper` as `analysis:turn<N>/T<m>`, one born in a finding as `analysis:F<n>/T<m>`. Each finding needs `id`, `module`, `severity`, `title`, `columns`, `evidence`, `method` (with `assumptions_checked` in `{passed, violated, n/a}`), `reading`, `so_what`; a finding whose `method` reports a p-value must also carry an `effect` entry in `evidence` and a `correction` entry in `method` (or `correction: "none — single pre-registered test"`). Each transcript turn needs `question`, `code`, `result`, `answer`, `grounded`; a turn with `grounded: false` has `code: null`, `result: null` and an `answer` that names the gap. Every `transformation_candidates[].op` must be a step name from `dataset-shaper/references/step-catalog.md` when that skill is installed, otherwise a free-form op with `"catalog": "none"`. Optional: `figures` (svg strings ≤ 120 KB each, render ≤ 1.5 MB total), `stances` (structured as above — `kind`, `columns`, `assertion`, `source` — so `/dataset-shaper` can check a step against them mechanically), `context.spatial` and `context.time` (absent or empty mean the same: not detected).

## 2. The admission rule for a finding

An observation becomes a finding only if it passes the **decision test**: it changes something downstream — a preprocessing step, a modeling choice, a data-collection question, or the reading of a result. `describe()` and `corr()` output is *evidence*, filed under `modules.*.evidence`; it is never a finding by itself. "Mean unit_price is 41.2" is evidence; "unit_price is bimodal with modes at 12 and 88 that coincide with two product families, so a single scaling decision misrepresents both" is a finding, and its `so_what` says what to do.

Two rules bind every finding that rests on a statistical test: the **assumption check comes before the p-value** (normality, homoscedasticity, independence, expected counts — checked by the script, reported as `passed` or `violated`, and a violation routes to the robust or non-parametric alternative rather than being noted and ignored), and **a p-value never travels alone** — it carries an effect size with its confidence interval and the multiple-comparison correction applied across the family it belongs to. Causal language ("X causes Y", "the effect of X") is not admitted on observational data; the finding says *associated*, and the `caveats` say why.

## 3. Modules — what each computes and what the analyst reads

The script computes; the analyst reads and admits. Each module's `evidence` is copied verbatim from `analysis.json`; each `reading` is prose per §5.

| module | computes (script) | the analyst decides |
|---|---|---|
| `quality` | missing rate per column and its dependence on other columns, duplicate rows and near-duplicates on the basis, constant / quasi-constant columns, cardinality, type coherence (mixed types, numerics stored as text, dates that do not parse), unit and range anomalies (negative counts, percentages above 100), inconsistent categories (case, whitespace) | which missingness mechanism is plausible (MCAR / MAR / MNAR) and what imputation follows; which anomalies are errors and which are the phenomenon |
| `distributions` | moments, quantiles, skew and kurtosis, normality (Shapiro on ≤ 5000 rows, otherwise D'Agostino), modality hint (KDE peaks), outliers by IQR, MAD and isolation forest with agreement counts, zero-inflation, nominal entropy and long-tail share | whether a transformation (log / Box-Cox / Yeo-Johnson) is warranted, whether outliers are genuine, which columns need a per-group reading |
| `relations` | robust correlations (Spearman, Kendall) among basis numerics, partial correlations controlling for the other basis members, mutual information, interaction screen on the label (pairwise product terms over at most eight basis numerics minus the leakage set and the label's own rule columns; only when a partition chose a label — with none, `interactions` is empty and the reading says so), VIF | which associations are new relative to the geometry layer's orthogonality report (that report is the baseline; this module reads *residual* structure only) |
| `inference` | group comparisons for every nominal × numeric pair with ≤ 12 groups: Welch ANOVA or Kruskal per the assumption check, ω² / ε² effect sizes, pairwise contrasts with Holm; nominal × nominal chi-square with Cramér's V; bootstrap CIs for the differences that matter; BH correction across the family; a nominal column with more than 12 groups is not tested, and the module's reading says which columns the cap excluded and why, so a silent omission never reads as a null result | which contrasts are practically relevant (effect size), not only significant; which tests were pre-declared by the user's question and which were exploratory |
| `segments` | k-means and agglomerative scans on the standardized basis (k = 2..8) with silhouette and stability under resampling; per-segment profiles; and the rule that places a row in a segment — `evidence: { scans[], chosen_k, features, profiles: [ { segment, n, share, medians, standardized_difference, dominant, label_rate } ], assignment: { rule, features, center, scale, centroids, segments } }`, `assignment` being the chosen k-means's own prediction (a row's segment is the nearest of `centroids` to `(x − center) / scale` over `features`, in `segments` order; a row missing one of them has none), which `/vega-eval` reads to cut segment slices of its test rows — a profile describes a segment, only the assignment places a row in one | whether segments are real or an artefact of scaling; what each segment *is* in the dataset's own terms |
| `importance` | only with a chosen partition (`context.partition.label` not null; otherwise `ran: false, skipped_because: "no partition chosen"`): a baseline model (regularized linear + gradient boosting, each at fixed hyper-parameters: no search over them, so nothing the record holds was chosen on any rows — `/vega-eval`'s `lens` baseline relies on this), stratified CV score with its corrected resampled t interval (`ci_method` names it), permutation importance, learning curve, a leakage probe (the fit never sees the columns of the geometry's rule for the label — `excluded.label_derivations`; those the partition still listed as features once the leakage set was removed, a set that should be empty, are added back in one refit of the best model, and the score's jump is what the label's definition would have carried; a near-perfect score is flagged on its own); folds chain forward in time when a datetime orders the rows (each trains on the rows before it, never on its future) | whether the task is learnable at all from the basis, which features carry it, whether an importance is leakage in disguise |
| `time_series` | only with a datetime column: regularity, gaps, trend (Theil–Sen, with Mann–Kendall checked for serial dependence of its residual), seasonality (STL by the detected period, checked for stability across the two halves), ACF/PACF, stationarity (ADF and KPSS together), change points | what the temporal structure implies for splitting (time-based, never random) and for features (lags, calendar) |
| `spatial` | only with a coordinate pair or a geometry column (an address column is text, not a location — it becomes a location only if the user supplies coordinates): CRS sanity (range check, axis order), duplicate locations, spatial autocorrelation (Moran's I on the numeric basis, Geary's C beside it), nearest-neighbour distances, clustering of points (DBSCAN in metres after projection) | whether location carries signal, which spatial features are worth deriving, whether the CRS is known enough to reproject |
| `drift` | only with a split column or `--split`: per-column distribution distance between splits (PSI, KS, Cramér's V), label-rate difference | whether the split is representative, which columns drift and why |

A module that cannot run says so (`ran: false`, `skipped_because`) — an empty module is a statement, not an omission. `importance`, `time_series`, `spatial` and `drift` have the preconditions the table names, and without one the script records `ran: false` with that reason (`no partition chosen`, `no datetime column`, `no coordinate pair`, `no split column and no --split`). `quality`, `distributions`, `relations`, `inference` and `segments` have none: they always run, and when their inputs are absent (fewer than two basis numerics for `relations` and `segments`, no chosen label for `relations`' interaction screen, no nominal × numeric or nominal × nominal pair for `inference`) their evidence is empty and the reading says which input was missing. A module left out by `--modules` is `ran: false, skipped_because: "not requested (--modules)"`; one that raised is `ran: false, skipped_because: "failed: <error>"`.

## 4. The transformation candidate

A finding may propose what to *do*; it never does it. The candidate is a step in `dataset-shaper`'s recipe vocabulary (`op`, `columns`, `params`) with its `rationale` pointing back at the finding, and `alternatives` the analyst actually weighed with the condition under which each would be preferred. `/dataset-shaper` reads `handoff.shaper_candidates` in order, treats each as a proposed step whose `source` is `analysis:F<n>/T<m>` (or `analysis:turn<N>/T<m>`), and asks the user (or, unattended, applies the default) — so the candidate must be concrete enough to execute without re-analysis: named columns, named strategy, named parameters.

## 5. Narrative register

The same five rules as `dataset-forge/references/report-contract.md` §4, restated so this file stands alone: **motive before mechanism** (a test is introduced by the question it answers), **connected paragraphs**, **no argument in bullets** (lists are for enumerable material), **length follows substance** (a module with nothing to say says so in one sentence), **terms defined in flow** (effect size, MAR, PSI, silhouette — defined the first time each appears, inside the sentence). The `context.reading` opens the tab by recalling what the geometry layer settled; each module's reading closes by handing off to the question the next module answers; the findings are the tab's headline, ordered by severity.

## 6. The dialogue turn

A turn is the unit of the interactive session. It is **reproducible** (the `code` field is the exact Python `scripts/cell.py` executed, and the rendered turn carries a *re-run* button that shows the stored code and result — the page cannot execute Python, so the button re-displays, and a cell that uses randomness seeds its own generator with the seed recorded in `source`, so the stored code reproduces the stored result), **grounded** (the `answer` is composed from `result`; a turn the analyst could not execute is `grounded: false` with the gap named), and **methodical** (`method.name` and `assumptions_checked` are filled even for a one-line lookup — "n/a" is an honest value). A turn may promote its result to a finding (`finding_ref`) or attach transformation candidates (ids unique across the layer); a turn may record a user stance (`stances`, structured) that later runs and `/dataset-shaper` honour.

## 7. The renderer (inside the Analysis tab)

`assets/analysis-render.js` is a fixed asset of this skill, checked once by `scripts/smoke_analysis.py`, injected by `apply_analysis_layer.py`; a run never regenerates it. It draws from `layer-analysis-data` alone, no libraries, no network:

- **Findings board** — severity-ordered cards; each opens to evidence, method with the assumption verdict, `so_what`, the linked figures, and the transformation candidates with a "hand to /dataset-shaper" note (the layer cannot write files).
- **Module panels** — one collapsible panel per module: the reading first, then the evidence tables (sortable), then the figures; a skipped module shows its reason.
- **Transcript** — the turns in order with question, method badge, code (folded), result table, figure, answer, caveats, and the *re-run* (re-display) button; an ungrounded turn is visibly marked.
- **Context strip** — what came from the geometry layer (basis, partition, leakage) as read-only chips, so the reader sees the baseline the analysis assumes.

Constraints: render script ≤ 60 KB; figures are inline SVG strings from the layer (the script never draws data charts itself — the Python side draws them under the `dataviz` skill's conventions when that skill is installed, otherwise matplotlib defaults with a restrained palette); keyboard-operable; `<noscript>` fallback carries findings, evidence tables and transcript; the render never mutates `layer-analysis-data` or earlier layers; for tests it exposes `window.__analysis = { findings(), turns(), version }` and nothing else.
