# `vega-train/scripts` and `vega-train/assets`

The tooling `SKILL.md`, `run-contract.md` and `training-catalog.md` specify. Nothing here is a
new decision: every file implements a clause, and the clause is named in the file's docstring.

```
scripts/
  catalog.json                 the machine copy of training-catalog.md §2-§4 (the ladder rules,
                               the thirteen diagnostics, the per-family search spaces, the
                               reserved and superseding run ids)
  vega_train_core.py           the delta path syntax and the campaign's value objects —
                               Edit, Delta, Rung, the one-decision-family rule, the frozen
                               sections, and the one seam to /vega-forge's scripts. Pure.
  diagnostics.py               D1-D13 as mechanical conditions over history.jsonl and
                               grad_stats.jsonl. Produces evidence, never a reading.
  expectations.py              a rung's expect as a check (the closed expect_check grammar)
                               and its judging against the ledger: expect_held. Pure.
  live_paths.py                the regularization paths the emitted code reads
                               (spec_snapshot.json read_paths, rule R2) and the dead deltas
                               that read nothing
  plan_ladder.py               drafts campaign.json and ladder.json, once — a directory that
                               already holds campaign.json is refused, as is a dead delta;
                               every rung is checked for one decision family, recompiled
                               before it reaches the file, and carries its rule's expect_check
  floors.py                    the floors a campaign reads its network against: R1 (linear,
                               or pca-floor with its dense(z) bottleneck, per sample shape)
                               and R1b (the gated recurrent block as rnn)
  trainer.py                   runs one rung (a run with no ladder rung, --delta included, a
                               dead delta and an expect it cannot check are refused): apply
                               the delta, recompile, re-emit + conformance
                               when the architecture moved, train through the package's
                               adapters (PackageAdapters: Trainer().fit(..., run=RunContext),
                               ModelRegistry), record, diagnose, append the ledger row with
                               expect_held judged (and every pending verdict again); a
                               --resume of a finished run opens the next free id (R1 -> R1c)
                               warm-started from its last.pt. Also --replay, which is what
                               tests determinism.
  stored_predictions.py        predictions.valid.parquet in vega/predictions-layout@1 (R-C2),
                               scored by the Evaluator (Trainer.predict without one)
  hpo.py                       one study, whose trials are runs like any other; TPE + median
                               pruning under optuna, random search with the same interface
                               without it; fANOVA or a marginal spread, and the record says which
  select.py                    the ladder/selection rule: ranking, mechanical eligibility, the
                               margin over the baseline, and --override for a human's pick
  apply_training_layer.py      validates and writes the layer through domain-forge's
                               apply_layer.py; refuses an ineligible selection with no override,
                               a failed replay, a missing notebook, a run with no history, an
                               expect_held that is not its expect_check's verdict
  smoke_training.py            headless-Chromium check that the layer renders and redraws
  tests/
    acceptance.sh              a real campaign end to end, then every gate against a seeded
                               defect, then the catalog check
    test_catalog_sync.py       fails when catalog.json and the prose diverge
    test_plan_ladder.py        the floor per sample shape (E17), R1b for the recurrent
                               families, the default-test rungs and the search spaces
    test_trainer_runs.py       runs through the v2.3 adapter classes on every sample shape,
                               the valid predictions layout, resume ids, HPO trials on
                               block params the spec leaves unstated; the runs refused before
                               a file is written; expect_held on the ledger rows
    test_campaign_integrity.py the expect_check grammar and its verdicts, dead deltas from
                               read_paths, the plan made once, the apply's verdict gate
    run_negatives.py           the seeded defects, each expected at a named gate
assets/
  training-render.js           the shipped renderer (44 KB of the 60 KB the contract allows)
  training-layer.css           scoped styles, light and dark
```

The notebook is `/vega-forge`'s `nb_emit.py --stage train` — one script writes every stage's
notebook, as notebook-contract §1 requires.

## Running the acceptance

```
scripts/tests/acceptance.sh [work-dir]
```

With no argument it uses `/vega-forge`'s acceptance directory, running the forge's acceptance
first if the forged model is not there.

Last full run: the ladder planned 14 rungs and a study (3 dropped — two duplicates and one
blocked by a recorded loss stance); R0, the linear floor and a weight-decay ablation ran; a
5-trial TPE study with 1 pruned; the selection by the rule; the train notebook executed with
its replay of the selected run matching **bit-exactly** at tolerance 0; `validate_model.py`
14 passed / 0 failed; `smoke_training.py --strict` 113 / 113; and 23 of 23 seeded defects
caught by the gate that should catch them.

## Two changes this pass made to /vega-forge

Both are template changes, made where the contract says design defects are fixed, and both are
needed for `/vega-train` to satisfy its own contract:

1. **`fit` gained an optional `grad_sample_every` and a `callbacks.on_step`.** `run-contract` §1
   requires `grad_stats.jsonl` — sampled per-step *pre-clip* gradient norms — and the epoch
   record cannot carry them, because by the epoch boundary the spikes have been averaged away.
   D7 (exploding gradients) is defined on exactly those samples. A backend that does not want
   them simply never calls the hook.

2. **`build_model` now applies the spec's seed before drawing a single parameter.** This was a
   real determinism defect and `/vega-train`'s replay is what found it: seeding inside `fit`
   left the *initial weights* to whatever RNG state the caller happened to be in, so two runs
   of the same spec differed from epoch 1. After the fix the replay matches bit-exactly on CPU.
