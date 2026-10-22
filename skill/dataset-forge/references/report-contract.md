# dataset-forge report contract — the `geometry` layer

The output file is a domain-forge model (canonical Turtle, JSON-LD mirror, Markdown abstract, optional `model-swrl` / `model-horn`) plus **one additive layer** named `geometry`. The layer's envelope (the `@LAYER` comments and their stamp) is domain-forge's, defined in its `references/future-skills.md` and written only by its platform writer `scripts/apply_layer.py`, which `scripts/apply_geometry_layer.py` calls; this contract does not restate it. Inside the envelope the layer carries the data block `<script id="layer-geometry-data" type="application/json">` (the document of §1), then the render script and the style — the shipped `assets/geometry-render.js` and `assets/geometry-layer.css`, under the ids the platform writer gives them.

The render script mounts one tab, `data-layer="geometry"`, labelled **Geometry**, into a new container appended to `<main>`; it reads only `layer-geometry-data`, is idempotent, and renders an explanatory empty state if the data script is missing. It never touches the base runtime or earlier layers.

## 1. `layer-geometry-data` schema

```json
{
  "schema": "dataset-forge/geometry@1",
  "source": { "path": "orders.csv", "rows": 600, "columns": 15, "geometry_json": "<run-dir>/geometry.json" },
  "markers": ["OK: ...", "WARN: ...", "FORK: ..."],
  "typing": [
    { "column": "zip", "script_type": "ordinal", "final_type": "nominal",
      "reason": "postal code parsed as integer; categorical by meaning", "role": "dimension",
      "consequences": { "...": "see §3 — required on every retyping (final_type ≠ script_type)" } }
  ],
  "space": { "ambient_dim": 10, "exact_rank": 10, "near_rank": 10, "dims_95": 6, "dims_99": 8,
             "participation_ratio": 5.0, "intrinsic_dim_twonn": 7.1,
             "reading": "<one paragraph: what these numbers mean for this dataset>" },
  "basis": {
    "members": ["unit_price", "qty", "discount_pct", "zip", "order_date", "delivered_days", "weight_kg", "note"],
    "size": 8, "intrinsic_dim": 7.1,
    "reading": "<paragraph: basis size vs intrinsic dimension, what may still be hiding>"
  },
  "derivations": [
    {
      "column": "total", "rule_id": "swrl-total", "layer": "swrl",
      "formula": "total = unit_price × qty", "body": ["unit_price", "qty"],
      "provenance": {
        "semantic":  { "status": "confirmed", "note": "price times quantity, by meaning" },
        "symbolic":  { "status": "confirmed", "verified_rows": 200, "of_rows": 200 },
        "empirical": { "status": "confirmed", "evidence": "degree-2 R²=1.000000 on 600 rows" }
      },
      "cycle": ["unit_price", "qty", "total"],
      "consequences": { "...": "see §3" }
    },
    {
      "column": "weight_kg", "rule_id": "swrl-weight", "layer": "swrl", "kept": false,
      "formula": "weight_kg = 0.8 × qty", "body": ["qty"],
      "provenance": { "semantic": { "status": "refuted", "note": "unit weight varies by product" },
                      "symbolic": { "status": "untested" },
                      "empirical": { "status": "refuted", "evidence": "residuals non-zero" } },
      "consequences": { "...": "see §3 — why it was rejected" }
    }
  ],
  "cycles": [
    { "id": "price-qty-total", "members": ["unit_price", "qty", "total"], "reading": "<paragraph>",
      "orientations": [
        { "id": "total-derived", "derived": ["total"], "consequences": { "...": "see §3" } },
        { "id": "unit-price-derived", "derived": ["unit_price"], "consequences": { "...": "see §3" },
          "derivations": [ { "column": "unit_price", "formula": "unit_price = total / qty", "body": ["total", "qty"] } ] },
        { "id": "qty-derived", "derived": ["qty"], "consequences": { "...": "see §3" },
          "derivations": [ { "column": "qty", "formula": "qty = total / unit_price", "body": ["total", "unit_price"] } ] }
      ] }
  ],
  "disagreements": [
    { "column": "weight_kg", "kind": "empirical-without-semantic",
      "evidence": "pearson 0.966 with qty", "reading": "<why: weight is qty times a per-unit weight the dataset does not carry; not derivable, but nearly collinear — a hidden dimension (unit weight) is the honest explanation>" }
  ],
  "orthogonality": {
    "pairs": [ { "a": "qty", "b": "weight_kg", "nmi": 0.63, "pearson": 0.97, "independence": "definitional, not statistical" } ],
    "reading": "<paragraph>"
  },
  "partitions": {
    "chosen": "late",
    "provenance": "user-chosen | single-candidate | abstained",
    "candidates": [
      { "label": "late", "task": "binary classification", "rule_id": "swrl-late",
        "features": ["unit_price", "qty", "discount_pct", "zip", "order_date", "weight_kg", "note"],
        "dropped_for_leakage": ["delivered_days"],
        "input_dim": 7,
        "consequences": { "...": "see §3" } }
      // a candidate step 4 does not admit (no kept rule whose body is in the basis derives it, no
      // exact functional dependency determines it) is refused unless it carries the stance that
      // names it: "stance": "user: <the user's words>" (a verification lane: "lane: …"); an
      // unattended run where step 4 admits none states the semantic reading as a hypothesis,
      // "stance": "unattended: <the reading>", accepted (WARN) only while provenance is
      // "abstained" and the label is not chosen
    ]
  },
  "stats": { "<column>": { "...": "verbatim from geometry.json stats" } },
  "handoff": { "next": "/data-lens <out> | /dataset-shaper <out> --target <label>", "note": "<one sentence>" }
}
```

Every `reading` and every `consequences` field is **prose** obeying §4. Numbers come verbatim from `geometry.json`; the layer never recomputes.

A **rejected** derivation (one the semantic channel refutes, or one set aside with `kept: false`) stays in `derivations` with its triple and the consequence block that says why; only kept rules derive a column. Every derivation that names a `cycle` (its member list) has a `cycles` entry with those members: the report's own orientation (its `derived` set is the kept rule heads among the members) and every alternative the cycle admits — a three-member cycle in which each member follows from the other two has three orientations, and all three are listed, as in the example above (at least one; the apply refuses fewer, and a listed orientation is the only one the explorer can offer), **each with its own consequence block** — the block View B shows when the analyst re-orients the cycle — and, optionally, the rules that orientation would use. `apply_geometry_layer.py` completes each orientation with `default`, `basis` and `rules` (what `/dataset-shaper` cites as `geometry:cycles/<id>`).

## 2. The provenance triple

Each derivation carries three independent verdicts. They are not a hierarchy; they are three instruments pointed at the same claim.

| channel | who | confirmed means | refuted means | untested means |
|---|---|---|---|---|
| `semantic` | the model | derivable by meaning, rule stated | not derivable by meaning | — (the model always has a view) |
| `symbolic` | the domain-forge engines (SWRL reasoner or Prolog runner via `model-chat/scripts/run_query.py`) on the A-box sample | the rule fires and its output equals the observed value on every sampled row (`verified_rows == of_rows`, taken from the runner's output) | fires but disagrees on ≥1 row (report the rows) | no engine could run the rule (runner error, unsupported operation) |
| `empirical` | `geometry.py` on the full data | derivability established (R² ≥ threshold, exact FD) | tested and not derivable | not testable (text, no numeric form) |

A derivation with `semantic: confirmed` and both others `untested` is still a derivation — but the report says so in those words. A `symbolic: refuted` with `empirical: confirmed` almost always means the formula is right and the rule encoding is wrong (units, rounding, a null); say which you think it is.

## 3. The consequence block

Attached to every decision (a retyping, a basis choice inside a cycle, a derivation kept or rejected, a partition candidate). Fields:

```json
{
  "decision": "unit_price and qty enter the basis; total is derived",
  "alternatives": ["total and qty in the basis; unit_price derived", "all three kept as features"],
  "downstream": {
    "preprocessing": "total needs no scaling decision of its own; it inherits from its parents",
    "model_family": "linear models: keeping all three adds an exactly collinear column — coefficients unidentifiable; trees: harmless but split budget wasted",
    "leakage": "if total is ever the label, unit_price and qty must leave the features or the model learns the multiplication",
    "dimensionality": "/vega-forge sees 2 inputs for this cycle, not 3",
    "interpretability": "unit_price is a price the business sets; total is an outcome — the chosen direction keeps the controllable quantities as inputs"
  }
}
```

Rules: every `downstream` entry names a concrete effect on *this* dataset or says "none". Alternatives are the ones the analyst actually weighed, not an exhaustive list. If the decision was forced (only one candidate), `alternatives` is empty and `decision` says why it was forced.

## 4. Narrative register

The Abstract (`model-markdown`) and every `reading` field are checked by `/audit-chapter`'s rubric, so they obey it:

- **Motive before mechanism.** Introduce a measure by the question it answers, never cold. Not "TwoNN = 7.1" but "the rank test says nothing is linearly redundant, so the next question is whether the data still lives on fewer dimensions than it has columns — the two-nearest-neighbour estimate answers that: about seven."
- **Connected paragraphs.** Each section opens by recalling what the previous one established and closes by handing off to the next question.
- **No argument in bullets.** Lists are for enumerable material (the basis members, the stats table); reasoning is prose.
- **Length follows substance.** A consequence with nothing downstream is one sentence.
- **Terms defined in flow.** Define *basis*, *derivation*, *intrinsic dimension*, *functional dependency* the first time each appears, inside the sentence.

The Abstract alone must let a reader understand the space the dataset occupies, its basis, the derivations, the chosen partition and why — the standard domain-forge sets for its Markdown block, applied to a dataset.

## 5. Hand-off (when a partition is chosen)

The chosen partition is a *decision*, recorded in `partitions.chosen` with its `provenance`; the data it implies is written by `/dataset-shaper` (`select_partition` + `drop_leakage` + the split, each a traced recipe step), and the network's `input_dim` is read by `/vega-forge` from the shaper's manifest, reconciled against `partitions.candidates[].input_dim` here. `handoff.next` names the invocation the summary proposes; nothing is materialized by this skill beyond the model HTML.

## 6. The interactive explorer (inside the Geometry tab)

The Geometry tab is not a printed report with a table at the bottom; it is the surface the analyst *moves through*. The render script draws four linked views from `layer-geometry-data` alone — plain DOM + hand-written inline SVG, no libraries, no network — and every view answers a question the report raised in prose. The data it needs ships in the layer: `explore` (a columnar sample of ≤ 2000 rows, text columns omitted, plus precomputed PCA scores and loadings — the script's `explore` object, verbatim) and the `derivations`, `basis`, `orthogonality`, `partitions` objects above. The JS never recomputes rank, PCA or dependencies; it filters, projects, joins and redraws.

**View A — Space.** The scree of singular values (bar) beside a 2-D scatter of the sample on any pair of *named* basis members (two selects), with a third select that colours points by a nominal or a label candidate. A toggle switches the axes to the two leading PCA components: the rotated view, labelled as such, with a loadings panel that says which named columns each component is made of. Hovering a point shows the row: its basis values, and its derived values each tagged with the rule that produced them and the provenance triple's status. Brushing a rectangle filters every other view to the selected rows.

**View B — Derivation graph.** Columns as nodes, derivations as directed edges (body → head), functional dependencies as dashed edges, cycles drawn as such. Basis members are filled, derived columns hollow, identities greyed. Clicking a node opens its card: type and retyping reason, provenance triple, the consequence block, the `WARN`/disagreement entries that mention it. Inside a cycle, the card offers **"make this the basis member"**: choosing it re-orients the cycle, recomputes the basis in JS (the set of columns that are the head of no active rule), updates View D's candidates, and shows the consequence block the analyst pre-wrote for that orientation (`cycles[].orientations[].consequences`) — so "what does a different basis imply" is one click, not a re-run.

**View C — Orthogonality.** A heatmap of the residual-dependence measures among the current basis members (the measure chosen by type pair, NMI as the common scale), colour-scaled with the thresholds from `geometry.json`. Clicking a cell opens that pair as the axes of View A. The `independence` reading of the pair ("definitional, not statistical") appears beside the value.

**View D — Partition.** The label candidates as selectable cards. Selecting one highlights on the graph the features that remain, strikes the leakage set, shows the input dimension `/vega-forge` would receive and the candidate's consequence block, and colours View A by the label. A "materialize" note explains that the chosen partition becomes shaped data through `/dataset-shaper <file> --target <label>`; the layer itself cannot write files.

**Cross-view state** is one object (`selectedRows`, `axes`, `colourBy`, `activeOrientation`, `activeCandidate`) that every view reads; a change anywhere redraws the others. The state is not persisted — reload restores the report's own choices, which are the analyst's pre-written decisions.

**Constraints.** Render script ≤ 80 KB; SVG only (no canvas), so the figures can be copied out; keyboard-operable selects and buttons; a `<noscript>` fallback that shows the prose and the stats table; if `explore` is missing or has fewer than 10 rows, Views A and C render their empty state and B and D still work (they need only the derivations). The render never mutates `layer-geometry-data` or earlier layers.
