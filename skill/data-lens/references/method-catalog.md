# data-lens method catalog

The routing table for the automatic pass and for every dialogue turn: the **question** an analyst asks, the **method** that answers it, the **assumptions** the method makes, the **check** that tests them (what `analysis.py` / a cell runs), the **effect size** that must accompany the result, and the **robust alternative** the script switches to when the check fails. A method not in this table may be used in a turn, with the same six columns filled in the turn's `method` block.

A `—` in the **check** column means the method makes no assumption the data can fail and a script can test (a count, a share, a census, the pairing of two columns of one row, a sampling design such as i.i.d. that only the collection can vouch for — the reading then says so in its caveats); its `assumptions_checked` is `n/a`. A method whose assumption the data can fail always has a check — the script's, or, for a method only a dialogue turn runs, the turn's own cell — the script records `passed` or `violated`, and a `violated` row reports the robust alternative's result in its place. A `—` in the **robust alternative** column is used only where the check cannot fail into another method (the method has no check, or its reading already is the robust one).

Conventions: α = 0.05 on the *adjusted* p-value; the family for correction is the set of tests run to answer one question (all group comparisons of one module, all pairwise contrasts of one omnibus test); Benjamini–Hochberg across exploratory families, Holm within a pre-declared one; confidence intervals at 95%, bootstrap (BCa, 2000 resamples, seeded) when no closed form is trusted; n < 30 per group makes every parametric result low-confidence regardless of p — the finding's `caveats` say so (a confidence note, not the finding's `severity`).

## Quality

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Is missingness in X related to other columns? | chi-square of the missing indicator against each nominal; Mann–Whitney of each numeric by missingness | expected counts ≥ 5 | expected-count table | Cramér's V; rank-biserial r | Fisher exact (2×2), Monte-Carlo chi-square on a larger table |
| Are these rows duplicates? | exact match on all columns; near-match on the basis with per-column tolerances | — | — | share of rows | — |
| Is this column what its type says? | parse rate under the declared type; pattern census | — | — | share unparsable | — |
| Are values within plausible ranges? | range and unit checks from meaning (negative counts, percentages > 100, dates in the future) | — | — | share out of range | — |

## Distributions

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Is X normal? | Shapiro–Wilk (n ≤ 5000), D'Agostino K² otherwise | i.i.d. sample | — | skew, excess kurtosis | QQ reading; never a p alone |
| Does X have outliers? | IQR fence, MAD (3.5), isolation forest (contamination auto) — agreement count | continuous X | modality check first (bimodal X fools the fences) | share flagged, max |z| | per-group fences when a nominal explains the modes |
| Is X multimodal? | KDE peak count with bandwidth scan | continuous X | — | number of modes, separation | — |
| Should X be transformed? | skew and heteroscedasticity vs the label; Box-Cox λ estimate | X > 0 for Box-Cox | sign check | skew before/after | Yeo–Johnson for non-positive X |

## Relations

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Are X and Y monotonically related? | Spearman ρ, Kendall τ | paired observations | — | ρ, τ with CI | — |
| Is X related to Y controlling for Z? | partial Spearman | linearity of the rank relation | residual plot reading | partial ρ | conditional mutual information |
| Is there redundancy among features? | VIF on the basis numerics | linear model | — | VIF | condition number from the geometry layer (already there — cite it) |
| Do X and Z interact on the label? | screen: label ~ X + Z + X·Z, likelihood-ratio test | model family fits | residual check | ΔR² or Δdeviance | tree-based interaction strength (H-statistic) |

## Inference — group comparisons

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Does numeric Y differ across two groups? | Welch t | approximate normality within groups, or n ≥ 30 per group | Shapiro per group; QQ | Cohen's d (Hedges g for small n) with CI | Mann–Whitney, rank-biserial r; permutation test |
| … across k > 2 groups? | Welch ANOVA | as above | Shapiro per group; Levene | ω² | Kruskal–Wallis, ε²; then pairwise Dunn with Holm |
| Which groups differ? | Games–Howell pairwise (after Welch ANOVA) | as above | the omnibus test's check (the contrasts follow its route) | pairwise d | Dunn with Holm |
| Are nominal X and Y associated? | chi-square | expected counts ≥ 5 | expected-count table | Cramér's V (bias-corrected) | Fisher exact (2×2), Monte-Carlo chi-square (999 permutations, 199 above 50 000 rows) |
| Do proportions differ? | two-proportion z, or chi-square | n·p ≥ 5 and n·(1−p) ≥ 5 per group | the four counts, in the turn's cell | risk difference, odds ratio with CI | Fisher exact, with the Newcombe interval |
| Is the label imbalanced enough to matter? | class shares; expected baseline accuracy | — | — | minority share | stratified split candidate for the shaper |
| Paired before/after? | paired t | normality of differences | Shapiro on differences | d_z | Wilcoxon signed-rank |

## Segments

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Are there natural groups in the basis? | k-means k = 2..8 on standardized basis numerics; agglomerative (Ward) as cross-check | numeric basis, scale-comparable | silhouette across k; stability under 20 bootstrap resamples (ARI) | silhouette, ARI | Gower distance + PAM when nominals are in the basis |
| What is a segment, in the data's own terms? | per-segment profile: medians, dominant categories, label rate | — | — | standardized differences from the whole | — |

## Importance (on the chosen partition only)

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Is the task learnable from the basis? | stratified 5-fold CV of a regularized linear model and a gradient-boosting model | features = partition's `features` minus the leakage set and the label's rule columns | leakage probe: the label's rule columns the partition still listed as features are added back in one refit, and the score's jump is reported; a near-perfect CV score is itself a warning | CV score with its corrected resampled t interval (Nadeau–Bengio: the fold scores' sample sd, the variance widened by the folds' shared training rows, t with k−1 degrees of freedom) vs the baseline (majority / mean) | forward-chaining time folds (each fold trains on the rows before it) when a datetime orders the rows |
| Which features carry it? | permutation importance (10 repeats) on one held-out quarter of the rows (the last quarter in time when a datetime orders them) | as above | — | mean decrease with a t interval over the repeats (it leaves out the split's own variance) | SHAP when installed, else grouped permutation for correlated features |
| More data or better features? | learning curve at 10–100 % of train | as above | — | slope at the last point | — |

## Time series (only with a datetime column)

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Is the series regular? | gap census on the sorted index | — | — | share of gaps, largest gap | resample to the modal interval, count imputed points |
| Is there a trend? | Theil–Sen slope with CI; Mann–Kendall | monotone trend; serially independent residuals | lag-1 autocorrelation of the residual after the Theil–Sen line inside ±1.96/√n | slope per unit time with its CI | Hamed–Rao variance-corrected Mann–Kendall (its p is `p_reported`) |
| Is there seasonality? | STL at the detected period (ACF peak) | regular series (resampled to the modal interval first); a period stable across the series | period stability across halves (each half's ACF peak within 10 % of the period, at least one lag; `n/a` when a half holds fewer than two periods) | seasonal strength (Hyndman) | the period read per half, with no single seasonal strength claimed |
| Is it stationary? | ADF and KPSS together | no structural break (a break pushes both tests toward a unit root) | the change points of the next row, computed on the same series and read beside the four-way reading: with one, the reading holds within segments only | none: the reading is categorical, and neither test's p is reported as a finding, so the p-value rule (effect size, correction) has no p to attach to | the four-way reading (both / neither / one of the two) is the answer, not a p |
| Where does it change? | PELT change points (penalty BIC) | — | — | number of segments, mean shift | — |
| What does this mean for splitting? | — | — | — | — | time-based split candidate for the shaper; lag features from the ACF |

## Spatial (only with a coordinate pair or geometry)

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Are the coordinates sane? | range check in the declared CRS; axis-order check (lat/lon swapped) | a CRS | — | share out of bounds | `crs: unknown` → only the range check |
| Does location carry signal? | Moran's I of each numeric basis member on a k-NN weight matrix (k = 8) | projected coordinates (metres) | the latitude/longitude pair projected to local metres before the weights (another geometry kind is not measured) | I against its expectation −1/(n−1), with pseudo-p (999 permutations) | Geary's C on the same weights and permutations, reported beside I (it weighs local differences; a disagreement between the two is a reading) |
| Are there spatial clusters? | DBSCAN in metres (eps from the k-distance knee) | metric CRS | the same projection to local metres as Moran's I | number of clusters, share of noise | HDBSCAN when installed |
| Are points duplicated in space? | exact and within-r duplicate locations | — | — | share | — |

## Drift (only with a split)

| question | method | assumptions | check | effect size | robust alternative |
|---|---|---|---|---|---|
| Does numeric X drift between parts? | KS two-sample; PSI (10 quantile bins) | — | — | KS D; PSI (0.1 / 0.25 thresholds) | — (the Wasserstein distance is reported beside, as the size of the shift) |
| Does nominal X drift? | chi-square on the part × category table | expected counts ≥ 5 | the expected-count table (`min_expected`) | Cramér's V; PSI | Fisher exact (2×2) or Monte-Carlo chi-square (999 permutations, 199 above 50 000 rows) |
| Does the label rate drift? | two-proportion z | n·p ≥ 5 and n·(1−p) ≥ 5 in each part | the four counts | risk difference with CI | Fisher exact, with the Newcombe interval |

## What is not answered here

Causal effects on observational data (only associations, with the caveat written), forecasts (a trend reading is not a forecast), and anything about the ontology's rules and classes — that is `/model-chat`, which runs the engines.
