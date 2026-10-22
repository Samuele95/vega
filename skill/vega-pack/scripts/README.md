# `/vega-pack` scripts

The network leaves the chain. What ships is what was evaluated, and the round trip is what
makes that a fact rather than an intention.

| file | what it is |
|---|---|
| `catalog.json` | `references/bundle-contract.md` §1–§7 in machine form: the files a bundle must hold, which modules of the emitted package are its serving slice and why each dropped port is out of scope, the round-trip tolerance formula, what each serving mode implies, the output fields per task kind, and — from v2.3 — the reading vocabulary (`vega/output@2`, each reading's link, decision and fields), the field kinds the answer codec reads, the sample shapes, the monitor kinds, the serving template patterns and what a re-emit may change. Every script reads it; none restates it. `tests/test_catalog_sync.py` fails when it and the prose diverge. |
| `vega_pack_core.py` | The value objects: `Tolerance` (a formula, not a knob), `FileDigest`, `Provenance`, `RoundTrip`, `BundlePaths`. Pure — frozen dataclasses and functions, no argv, no printing, no exits. `read_parquet` is the stage's one way to read a parquet file, through Arrow's `LocalFileSystem` (A40: pandas' default hands Arrow a Python file object, and the process can abort at exit). |
| `pack.py` | Builds the bundle, in the shape `--include` names. Refuses on provenance before writing anything, takes the transformer from the shaper's `--transform-only`, runs the shaper's twin check, derives `schema.json` from the raw columns, completes the serving adapter from the backend's template, and round-trips every held-out row from raw input against the evaluation's stored predictions. |
| `output_record.py` | The `vega/output@2` record and `MANIFEST.sample`: `resolve_reading` applies R1–R12, `output_record` writes the reading, fields, `field_kinds`, compared and decided pairs; `schema_output` writes the output half of `schema.json`. `field_patterns.py` fills the catalog's field patterns (a placeholder's token in a field name, its value in `of`); `input_schema.py` is the input half of `schema.json`. `tests/test_output_record.py` covers every reading. |
| `graph_pack.py` | The graph reference of a graph-shaped bundle (vega/graph-reference@1): `write(ctx)` writes `graph/reference-nodes.parquet` (train then valid reference nodes in `__split_raw__` order, the features as the network reads them plus `raw:<column>` as the input held them, no raw position), `graph/reference-edges.parquet` and `graph/graph.json` (the catalog's meta keys, `edges_for_columns`), and returns `MANIFEST.sample.reference` {nodes, edges, files}. It refuses a transductive protocol (exit 2, 'inductive graphs only' — `pack.py` calls `refuse_unserved` before anything is written under `--bundle-dir`), a transform that does not serve the graph (exit 3; the `edges_for` columns are compared as a set, and graph.json lists them in the transform's order) and a part or spec that is not the manifest's (exit 2). A row bundle writes nothing. The serving side is the backend's `serving/package/{samples,graphs}.py`; `vega-forge/scripts/tests/test_samples.py` covers both. |
| `monitor_pack.py` | `--monitor-plan`: `monitor_plan.validate_plan` checks the plan against the reading, the schema and the output record (exit 2 on a kind not admitted or an unknown column, field, class or label) and adds the automatic output check (`output_psi` on the score field, else `prediction_mix`, else `label_rate`; id `auto-output` reserved); `monitor_reference.py` measures the references (`vega/monitor-reference@2`, strict JSON): inputs on the train part, answers on the valid part from the stored predictions through the bundle's reader or else the Predictor, never the test part; the pack-time smoke goes through the bundle's own `load_monitor`. `vega-forge/scripts/tests/test_monitor_kinds.py` covers it. |
| `serving_pack.py` | The serving slice and the served adapter. `serving_templates.py` renders the backend's serving templates by the catalog's patterns (R-D2): `package/*.py` and `package/inspection/*.py` into the Predictor's module, `serve/*.py` for the online service (only the route modules of the enabled capabilities) or, for a stream blueprint, `serve/consumer.py` with the row contract it reuses; every `{{…}}` is filled from the bundle (package, role modules, ports, served identity) or the pack stops. `serving_closure.py` copies the Model side the Predictor imports (readout, assemble, builder, blocks, export) and its relative-import closure. The serve record carries the endpoints the route modules declare, the templates digest, `showcase_slot` (online) and `forced_by` when `--serve` overrides a batch or stream blueprint. `vega-forge/scripts/tests/test_serve_app.py` covers the service. |
| `demo_pack.py` | The demonstration sample (R-D8): `examples/demo.json` (vega/demo@2) and `examples/presets.json` (vega/presets@2) — plus `examples/demo.csv` for row and graph shapes — drawn from the train part only (seed 7, `--demo-sample` items per stratum), checked against `examples-2.schema.json` before they ship. `demo_samples.py` (candidates per sample shape, the Predictor calls), `demo_strata.py` (the strata per reading) and `demo_presets.py` (typical and edge presets, kinds and facts, no English). `tests/test_demo_pack.py` covers every reading and shape. |
| `roundtrip_plan.py` | The pack-time round trip per sample shape (bundle-contract §5, R-A5, R-A6): `run(ctx, predictor)` answers the raw rows behind every stored test prediction through the bundle's own Predictor and compares them with what `/vega-eval` stored, paired by `row_id` (by `graph_id` at graph level), never by position. Row: a lag recipe gets the shaper's empty history. Window: each series gets the last `history_rows` rows of train then valid as history, the stored count must be test rows minus `unscored.json` (N-S2), and the first `(length − 1) + lag_depth` test rows per series are the lag boundary, excluded and recorded in `MANIFEST.roundtrip.boundary`. Node level: the whole test part in 4096-row chunks. Graph level: each stored graph. `MANIFEST.roundtrip.rows` is the stored count, boundary rows included. Exit 3 on a failure, on a row-score reading without `row_score` (N-S1), on a compared column the stored layout lacks, on a layout other than `vega/predictions-layout@1` or without `row_id`, on a window evaluation without `unscored.json`, and on a graph Recipe without `edges_for`/`edges_within`. `roundtrip_inputs.py` reads the stored predictions through vega-eval's `predictions_layout.read` and the raw rows with what the Recipe reads (series and order off the SHAPED keys); `roundtrip_compare.py` compares the record's `compared` pairs at the tolerance and its `decided` fields exactly (N-S8). `tests/test_roundtrip_plan.py` covers every shape and N-S1/N-S2/N-S8/N-S9; `tests/roundtrip_fixtures.py` packs B15's spike fixtures with the real `evaluate.py` and `pack.py`. |
| `examples_pack.py` | `examples/rows.csv`, `examples/expected.parquet` and `examples/request.json` per sample shape (D2 §8.3): `expected.parquet` always carries `__row__` and every compared and decided field. Test examples carry the evaluation's own answer and are answered at pack time as `check_examples` answers them (exit 3 when they do not reproduce); window examples start at the test part's start with `length − 1` context rows per series; graph-level examples ship whole graphs with `graph_id`. `--examples synthetic` resamples train rows per shape and proves the bundle reproduces itself. |
| `inspection_pack.py` | The pack-time inspector check (D3 §13): every rendered `<serving>/inspection/*.py` is a support map entry; the Predictor's `NetworkInspector` opens on the bundle's example rows at the round-trip tolerance (exit 3 when not ready); `MANIFEST.serve.inspection` records `enabled`, `weights` and the pinned `network_digest {full, summary}`, re-verified at service start-up and by `load_inspector`. `--no-inspect` ships no `inspection/`; `--no-inspect` and `--inspect-weights summary` are recorded as stances. `api_declaration.py` is the blueprint declaration fork: served endpoints against `pipeline.json.serving.api.endpoints`, recorded as `blueprint.conformance.api_declared` / `declaration` (attended: a FORK line; unattended: the `/blueprint --extends` call recorded, or answer (c) when the installed blueprint lacks `--extends`). `tests/test_inspection_pack.py` covers it (NI1–NI5). |
| `reemit.py` | `--re-emit-serving` (bundle-contract §7.8): the bundle in place must still verify (exit 2); the pack runs again into `<bundle-dir>/.reemit/<slug>-<run>/`; `reemit_proof.py` compares the new manifest with the old one against the allow-list (serving-template outputs, `exports/*`, and `MANIFEST.entrypoints` / `blueprint-map.json` entries naming those outputs); anything else is refused with exit 3 and named, the bundle untouched. On success the bundle is swapped (the old one kept as `.prev` until the swap holds), the other stages' files are carried over and `MANIFEST.reemits[]` records `{at, reason, previous_manifest_digest, templates_digest, changed}`. `tests/test_reemit.py` covers it on the graph-level spike fixture. |
| `model_card.py` | `model-card.html`: one self-contained file, every number spliced from a layer or a bundle record, and the one hand-written passage refused if it contains a digit. |
| `apply_bundle_layer.py` | Validates the `layer-bundle-data` document and writes the `bundle` layer through `domain-forge/scripts/apply_layer.py`. |
| `smoke_bundle.py` | Headless-Chromium check that the layer renders and says what the record says. |
| `tests/acceptance.sh` | The whole stage on a real evaluated campaign, ending with the bundle run in an empty environment with the skills unreachable. |
| `tests/run_negatives.py` | 69 seeded defects, one at a time, each asserted to be caught by the gate that should catch it: the 35 grandfathered cases (`vega/scripts/tests/negatives/grandfathered.json`), N-S1…N-S9 and NI1–NI5. Seven cases need another lane's work dir from a lane matrix (`<matrix>/lanes/<id>/work`); without one they are reported as not run, never as passed. `--json` writes `vega/negatives-result@1` (`vega/scripts/tests/negatives/negkit.py`). |
| `tests/run_negatives_showcase.py` | The showcase slot and inspection route negatives SS1–SS9 and IN1–IN7 on seeded copies of a lane bundle (L02; L10 and L19 for the window and graph cases), each copy's own service started in a child process. `--json` as above. |
| `tests/test_catalog_sync.py` | `catalog.json` against the contract, plus the check that every task kind `/vega-eval` can evaluate has an output-field row here, that each reading maps to the link vega-forge's `inspection.links` names, and that the examples@2 schema accepts and refuses its seeded payloads. |
| `tests/test_pack_defects.py` | The W7 pack defects held fixed (A42 C4): `edges_for` columns compared as a set (L20), the transductive refusal before any write (N-S3), a rung's or a trial's own smoke in `records/architecture.json`, the layer stating recorded statistics as the record writes them (L14, L17), and no parquet read outside Arrow's filesystem (A40). L13's missing-label decoding is in `test_output_record.py`. |
| `tests/bundle-layer.authoring.json` | The acceptance run's judgement: the two readings, and no digit in the intended-use prose. |

The notebook is not here: `nb_emit.py --stage pack` lives in `/vega-forge`, because one script
writes every stage's notebook (`notebook-contract.md` §1).

## The order, and why it is the order

```
provenance  →  transform  →  twin check  →  round trip  →  card  →  notebook  →  layer
```

**Provenance first.** The chain recorded a digest for the resolved spec and the checkpoint; the
bundle carries whatever is on disk. A disagreement stops the pack before a directory is created,
because a bundle built from a different file would wear numbers it did not earn. `fitted.json`
has no upstream digest, and the twin check is its test instead — which is why the check exists
rather than a fourth digest.

**The transformer comes from the shaper.** `features/preprocess.py` is emitted by
`dataset-shaper/scripts/shape.py --transform-only`, which re-emits the very lines the
reproduction script carries. A preprocessing written here would be a second implementation of
the recipe, and it would drift from the one the network was trained on — not immediately, which
is the problem.

**The twin check runs before the round trip.** Both would catch a broken transform, but only the
twin check catches it *without the network in the way*, so its failure names the transform rather
than leaving you wondering whether the weights are wrong.

**The round trip covers every row.** Not a sample: a preprocessing bug is usually conditional —
a category that appears in a hundred rows, a value at the edge of a fitted clip — and a sample
that missed those rows would license a bundle that is wrong for exactly the rows it is wrong for.

**The records travel.** `records/<layer>.json` is the six layers' data copied out of the model
page verbatim, and it is what the card and the notebook read to explain themselves. They are
data, not code: nothing in the bundle imports them, and a bundle whose records were deleted
still predicts — what it loses is the ability to say why. That split is deliberate in the
notebook too: its explanatory half needs only `json`, `pathlib` and `pandas`, so the card's
numbers can be audited without torch installed, and its verifying half imports `predict.py`,
so a broken bundle fails there and only there.

**Nothing is typed.** Every number on the card and in the layer is spliced from a record that
measured it. The two hand-written passages are prose, and `readings.intended_use` is refused if
it contains a digit — the one figure that traced to nobody would be the one a reader would most
want to check.

## Running it

```bash
python3 pack.py --campaign vega-runs/<ts> --shaped shaped/ --bundle-dir bundle/ \
        --pipeline blueprint-runs/<ts>/pipeline.json --model <model>.eval.html \
        [--include serving|full] [--ship-anyway "<reason>"] [--examples 50|synthetic]
python3 model_card.py --bundle bundle/<slug>-<run> --model <model>.eval.html \
        --intended-use bundle/<slug>-<run>/bundle-layer.json \
        --out bundle/<slug>-<run>/model-card.html
python3 ../../vega-forge/scripts/nb_emit.py --stage pack --from bundle/<slug>-<run> \
        --out bundle/<slug>-<run>/predict.ipynb
python3 apply_bundle_layer.py <model>.eval.html --data bundle/<slug>-<run>/bundle-layer.json \
        --notebook bundle/<slug>-<run>/predict.ipynb --out <model>.eval.pack.html
python3 smoke_bundle.py <model>.eval.pack.html --strict
```

`tests/acceptance.sh` does all of it on the fixture, running the three earlier stages first if
the evaluation is not there, and builds the other shape as its own step.

## The two shapes

`--include serving` (the default) copies the serving slice and writes `predict.py`: the thing
that answers. `--include full` copies the whole emitted package and writes `train.py` and
`evaluate.py` beside it: the complete server-side system as code, for someone who will re-fit on
their own data.

Three things are worth saying about `full`, and the bundle says all three — in the README, on
the card, in the notebook and in the layer — because a `train.py` sitting in a directory called
`bundle` will otherwise be read as something it is not.

* It is not more proven. The round trip, the twin check and the digests test the serving path,
  in both shapes, because that is what they test.
* `train.py` is a composition root over `training/loop.py`, not a second `/vega-train`: one run
  of one spec, no ladder, no ablations, no diagnostics, no HPO, no ledger. Its checkpoint is a
  network no model card describes. `evaluate.py` is the same shape over `evaluation/evaluate.py`:
  point metrics, no intervals, no baselines, no verdict.
* The seal is not weakened. `ingestion/data.py`'s `load_part` refuses the test part in the
  bundle exactly as it does in the chain, and `load_sealed_part` still wants a `test_seal.json`
  and its token — the door travels with its lock, which is why shipping the door is defensible
  at all. What `serving` avoids is subtler: an evaluator with no seal to answer to, sitting
  beside a model, invites a second look at a part someone already spent their one look on.

`fitted.json` names the split in both shapes, because the round trip cannot run without it. The
serving slice is therefore not a containment boundary and this file does not claim it is.

## Not yet built

`--export onnx,torchscript` and the `--serve` scaffolds of §7. The manifest records `exports: []`
and `serve.present: false` with a note saying the scaffold was not written, rather than implying
a batch-only bundle was a decision when it was an omission.
