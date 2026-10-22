# `vega-forge/scripts` and `vega-forge/assets`

The tooling `SKILL.md` and the five contracts specify. Nothing here is a new decision: every
file implements a clause, and the clause is named in the file's docstring and, for the two
gate scripts, on every marker they print (`--explain`).

```
scripts/
  catalog.json                     the machine copy of family-catalog.md §1-§7 (task rows,
                                   families, blocks, forks, metric directions, the admission
                                   table) and of inspect-contract.md's closed sets
  vega_core.py                     the scripts' own domain: Marker/Level, Catalog,
                                   ChainInputs, Choice, SourceResolver — pure, no I/O;
                                   re-exports ShapeWalk and PackageAdapters
  vega_chain.py, vega_shapes.py,   split out of vega_core: the chain inputs, ShapeWalk (the
  vega_adapters.py                 block shape walk the rules share) and
                                   PackageAdapters.load(package_dir, package) (the v2.3 role
                                   classes; it still reads a map without a generation as
                                   v2.2, and every stage refuses such a package)
  compile_spec.py                  28 named rules over one immutable Context; --explain
                                   prints the rule and contract clause behind each marker;
                                   --out writes the spec resolved (the regularization live
                                   paths derived from the declaration)
  rules_chain.py, rules_task.py,   the rule bodies compile_spec registers: chain and
  rules_blocks.py,                 provenance (P0, P1, B1, S1, D1a-D1d, K1, M2), task (T1a-T1e,
  rules_sample.py, rules_graph.py, D2a/D2b), blocks (G1a-G1d, G2, B2), sample (W1, W3, R1),
  rules_regularization.py          graph (W2) and regularization (R2)
  vega_regularization.py           R2's derivation, pure: the one regularization declaration,
                                   the live paths it covers, read_paths / shadowed_paths (what
                                   spec_snapshot.json records), resolve; its twin in the
                                   emitted code is templates/domain.py's Regularization
  vega_roster.py                   the artifact roster, pure (code-contract §3): every
                                   pipeline.json.artifacts[] entry held by domain.py or
                                   declared external with its owner; the emitter writes it
                                   into blueprint-map.json and C6 reads it back;
                                   held_vocabulary: the names an artifact can be held under
  held_artifacts.py                lists held_vocabulary on the backend's domain.py template:
                                   what /blueprint --mode pipeline names §2's held artifacts by
  emit_backend.py                  drives the backend's emit() and, with --smoke, builds the
                                   model and runs one forward pass on a real train batch
  conformance.py                   the mechanical half of the gate (code-contract §5.1), C0-C14;
                                   the size discipline C11-C14 reports at warn until G2 flips
                                   --size-severity to error; --entrypoints, --size-only PATH...,
                                   --json
  nb_emit.py                       engineer.ipynb per notebook-contract §2
  apply_architecture_layer.py      validates and writes the layer through domain-forge's
                                   apply_layer.py; refuses a failed smoke, a spec that no
                                   longer compiles, a failed mechanical gate, a pending,
                                   invalid or malformed judgement, an attended confirmed
                                   finding without the user's recorded answer, a missing
                                   notebook (--unattended: a confirmed finding is a WARN and
                                   an open item)
  audit_quorum.py                  the judgement half's rule (code-contract §5, Reproduced,
                                   or recorded, and Advisory): one clean first audit, or the
                                   first and two fresh ones with distinct run ids; a
                                   structural/divergence finding two of three audits report
                                   (same file, same anchor or category) is confirmed and
                                   advisory — unattended a WARN and an open item, attended
                                   the user's FORK answer; every audit and the matching as
                                   one record; the forge's and the pack's apply scripts
                                   import it
  smoke_architecture.py            headless-Chromium check that the layer renders
  architecture_sample.py           the sample record apply and the smoke share: the sample
                                   panel's counts per shape (D1 §16), the unscored caption
                                   the declared context and horizon allow
  backends/
    __init__.py                    the registry (backend-contract §3)
    pytorch/
      coverage.json                what this backend covers; the compiler checks specs against it
      explain-catalog.json         vega/explain-catalog@1 (schema in
                                   references/schemas/explain-catalog-1.schema.json): per block,
                                   activation and link, the formula, code anchors and reads the
                                   page's explainers use
      blocks.py                    one nn.Module per catalog block type, in spliceable segments
      builder.py                   spec -> network, with shape tracking and block_shapes
      losses.py                    the eleven losses, each with formula()
      emit.py                      emit(spec, pipeline, out_dir) -> the blueprint's package;
                                   one template generation, v2.3 (the split layout; gate A
                                   removed v2.2 and --generation); one sample shape per
                                   package: the templates' `# >>> shape:` sections are kept
                                   for the spec's shape only (select_shapes), and
                                   domain.CARRIED_SHAPES refuses any other by name
      templates/                   where the Effective Java rules of code-contract §4 live;
                                   generation v2.3 adds network.py + readout.py (Network,
                                   TaskReadout), data.py + frames/windows/graphs.py
                                   (ShapedParts), metrics.py (logits, MetricOptions),
                                   assemble.py (the serving SampleAssembler),
                                   serving/serve/*.py (the answer codec) and
                                   serving/package/inspection/{codec,reduce}.py; W2 added
                                   loop.py (the Trainer port's class: fit/evaluate/predict)
                                   with optim.py (device_for, resolved_lrs), record.py
                                   (the epoch record, EpochAccumulator, early stopping),
                                   engines.py and graph_engines.py (the epoch sources);
                                   evaluate.py (the Evaluator: scored, heads, score_part,
                                   score_sealed_test, measure) with baselines.py and
                                   baselines_extra.py (the baselines per task and shape);
                                   serving/package/{records,heads,values,guards}.py (the
                                   readers; reader_for is in values.py) and
                                   serving/package/{samples,graphs}.py (assembler_for,
                                   Prepared, HeadOutput, GraphContext); and
                                   serving/package/inspection/{graph,wiring,view,registry,
                                   readers,readers_seq,readers_graph,readers_group}.py (the
                                   StageGraph and vega/network-view@2); W3 added
                                   serving/package/predict.py (the one Predictor: predict,
                                   predictions, predict_one with history, prepare, reader)
                                   and the bundle CLI's load/check_examples/verify/read_rows
                                   in serving/bundle/predict.py; serving/package/monitor.py
                                   (the Monitor per reading); and
                                   serving/package/inspection/{capture,focus,parts,probes,
                                   probes_seq,probes_graph,probes_group,derived}.py (HookSet,
                                   Capture, NodeParts, the per-type probes and recompute
                                   checks) with {targets,target_kinds,query}.py (the
                                   explained targets and the explain request); W4 split the
                                   service into serving/serve/{app,startup,service,edge,
                                   routes_meta,routes_predict,routes_monitor}.py with
                                   serve/static/index.html (the v2.2 serving app, contract,
                                   predict and monitor templates are gone), added
                                   serve/consumer.py (the stream consumer) and
                                   serving/package/inspection/{attribution,attribution_parts,
                                   attribution_graph,interpolate}.py (gradient x input,
                                   integrated gradients, the attribution table), and moved
                                   the full bundle's train.py/evaluate.py to the v2.3 classes;
                                   W5 added serving/package/inspection/{inspector,explanation,
                                   attribution_table}.py (NetworkInspector: open, status,
                                   network, tensor, explain, attribute; Predictor.inspector()
                                   and the bundle CLI's load_inspector),
                                   serving/serve/routes_showcase.py (the showcase slot: gates
                                   SL1-SL11, /showcase/{path}, /v1/showcase, the verified
                                   entry page as / ) and its bundle test
                                   serving/tests/test_api_showcase.py;
                                   W6 added serving/serve/routes_inspect.py (GET /v1/network,
                                   /v1/network/tensor/{id}, POST /v1/explain over the
                                   Predictor's own inspector, /health.inspection) and its
                                   bundle suite serving/tests/test_api_inspect.py
  tests/
    acceptance.sh                  the whole procedure on a real fixture, then every gate
                                   against a seeded defect, then the catalog check
    test_catalog_sync.py           fails when catalog.json and the prose diverge
    make_fixture.py                builds the `orders` fixture
    run_negatives.py               69 seeded defects, each expected at a named gate
    run_negatives_shapes.py        the shape negatives W2a-W2i, W3a-W3d and B2a on seeded copies
                                   of the lanes' own fixtures (built on demand when the
                                   matrix lacks a lane); --json as every run_negatives
    test_conformance_sizes.py      C11-C14 boundaries and seeds (conformance_fixtures/)
    capture_golden.py, golden/     the pre-refactor captures (orders fit history, smoke
                                   records, CIC R7 smoke) taken from a worktree of c849f8b
    test_domain_plans.py           the window and graph index plans of templates/domain.py
    test_blocks.py, block_oracles.py
                                   every block type against its float64 oracle
    test_inspection_surface.py     every block class exposes coverage.json's inspection surface
    test_emit_layout.py, render_templates.py
                                   the emit table and its partial renders (--shapes), the shape
                                   sections, a package's one shape and its refusals, C4 on a
                                   lazy import of a module the package does not hold
    test_readout.py                Network/NetworkBuilder/TaskReadout on the rendered package
                                   (orders golden init, R7 record with CIDS_ROOT)
    test_losses.py                 losses.py spliced as training/losses.py, per-task shapes
    test_windows.py, test_graphs.py, test_data_rows.py, data_fixtures.py
                                   ShapedParts: window plans, graph views and seals, the row
                                   stream batch-identical to the v2.2 data.py (read from git at c849f8b)
    test_metrics.py                templates/metrics.py (logit semantics, mase scale,
                                   anomaly_score_mean, per-label pr_auc, the catalog keys)
    test_assemble.py               the serving SampleAssembler against the domain plans
    test_compile_rules.py          the compiler rules W1-W3, G2, T1b, T1e, D2a/D2b, R1, B2
    test_regularization.py         R2: the derivation, the rule, the emitted twin, the
                                   snapshot's read_paths, and on the orders fixture a rung on
                                   regularization[kind=weight_decay|dropout]/value training
                                   different numbers from R0
    test_serve_codec.py            serving/serve/*.py: readings, answers, problems, tables
    test_inspect_codec.py          the inspection codec and reducers against the schemas
    test_explain_catalog_sync.py   explain-catalog.json against blocks.py, coverage.json and
                                   vega-show (support: _explain_sync_{common,anchors,facts}.py)
    test_no_planned.py             coverage.json `planned` is empty (green since gate A)
    test_golden_orders.py          Trainer().fit on orders, bit-identical to the B01 golden
                                   (the fit pinned to the capture's 20 threads)
    test_determinism.py            two CPU fits identical per window and graph cell, E8/E9,
                                   the run context, the CUDA-graph path (skipped without CUDA)
    test_baselines.py              the Evaluator and every baseline per task and shape
    test_readers.py                the serving readers per reading (17 cases), from_stored
    test_samples.py, samples_fixtures.py
                                   the serving sample assemblers per shape, GraphContext and
                                   vega-pack's graph_pack reference files
    test_inspect_graph.py          the StageGraph and network-view@2 per family (R7 chain
                                   with CIDS_ROOT)
    test_predictor.py, predictor_fixtures.py
                                   the Predictor per reading, window and graph against the
                                   evaluation's layout; the bundle CLI (check_examples,
                                   verify, --check/--input/--one --history)
    test_monitor_kinds.py          the Monitor's kinds, validate_plan refusals, references
                                   from train/valid only, the automatic output check
    test_inspect_probes.py, _inspect_fixtures.py
                                   capture, probes and recompute checks on every block type
    test_inspect_concurrency.py    explains racing plain forwards stay byte-identical
    test_inspect_targets.py        the explained target per effective task kind, and the
                                   explain request's validation
    test_inspect_attribution.py    gradient x input and integrated gradients per shape against
                                   independent references, completeness at 64 steps, the
                                   attribution table
    test_inspect_inspector.py, _inspector_fixtures.py
                                   the inspector facade per shape: explain against
                                   explanation-2 for every allowed target, the open()
                                   self-test statuses, attribute() tables against explain,
                                   the selector and warning rules, load_inspector
    test_inspect_zoo.py            the inspection zoo (D3 §16.1 (a)-(g), A31): the 127
                                   admission cells plus 6 extras through network, explain and
                                   attribution; --dump DIR writes vega/inspect-zoo@1 payloads
    test_serve_app.py, serve_app_fixtures.py
                                   the split service: endpoints equal the registered routes,
                                   admission and body bounds, the closed-world start-up,
                                   code-view@2 and examples@2, stream mode, the Model closure
    test_consumer.py               the stream consumer on a rendered window bundle: history
                                   across batches, refusals, monitor modes, start gates
    test_architecture_sample.py    the architecture layer's sample panel per shape, the smoke
                                   against seeded render defects, the render budget
    matrix/run_matrix.py, matrix/make_matrix_fixtures.py, matrix/recipes.json
                                   the backend coverage matrix (B19): the 127 admitted cells
                                   through compile, emit, smoke, conformance, fit, the sealed
                                   score and the predictions layout; --cells all|smoke|<glob>,
                                   --sanity; exit 0 green, 1 a cell failed, 2 unusable input
    spike/test_vertical_slice.py, spike/make_spike_fixtures.py
                                   B15's vertical slice (M1): evaluation and serving assembly
                                   agree per row_id (window with lag, node level, graph level);
                                   every fixture is the full emit (the fallback is retired)
assets/
  architecture-render.js           the shipped renderer (42.4 KB of the 60 KB the contract allows)
  architecture-layer.css           scoped styles, light and dark
```

## Running the acceptance

```
scripts/tests/acceptance.sh [work-dir]
```

Needs torch, pandas, pyarrow and scikit-learn; jsonschema>=4 (the catalog sync validates the
schema files); nbconvert and a kernel to execute the notebook; a headless Chromium for the
layer smoke test (`CHROME=`, or found on PATH).

Last full run (the v2.3 W4 integration, lane L01): the chain green end to end,
`validate_model.py` 15 passed / 0 failed, the architecture layer smoke test 97 checks, and 66
of 69 seeded defects caught by the gate that should catch them.

## What is not here

- **The collapsed-binary metrics** (`mcc_binary`, `roc_auc_binary`, `pr_auc_binary`) are neither
  built nor planned: a spec naming one is refused by B2.

`nb_emit.py` emits all four stages (`--stage forge|train|eval|pack`), and refuses what its cells
could not run (notebook-contract §1).

## Two changes `/vega-train` required of the templates

Both were made where the contract says design defects are fixed — in the templates, re-emitted —
and both are in this package:

1. **`fit` gained an optional `grad_sample_every` and a `callbacks.on_step`.** `/vega-train`'s
   `run-contract` §1 requires `grad_stats.jsonl`, sampled per-step *pre-clip* gradient norms,
   and the epoch record cannot carry them: by the epoch boundary the spikes have been averaged
   away, and D7 (exploding gradients) is defined on exactly those samples. A backend that does
   not want them never calls the hook.

2. **`build_model` applies the spec's seed before drawing a single parameter.** This was a real
   determinism defect, and `/vega-train`'s replay is what found it: seeding inside `fit` left
   the *initial weights* to whatever RNG state the caller happened to be in, so two runs of the
   same spec differed from epoch 1. After the fix the replay matches bit-exactly on CPU, which
   is what `spec_snapshot.json`'s declared tolerance of 0 claims.
