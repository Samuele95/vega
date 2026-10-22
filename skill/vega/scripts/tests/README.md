# vega/scripts/tests — the verification program

This directory holds the package's verification program (design D6): the lane registry, the
scheduler that runs the lanes and the tiers, the checks the lanes call, and the gates a release
must pass. Every skill keeps its own unit tests and acceptance script under
`<skill>/scripts/tests/`; this program finds them and runs them in its tiers.

## Running it

```bash
export PYTHON=<interpreter with torch, pandas, pyarrow, scikit-learn, nbconvert, jsonschema>
export CHROME=<headless Chrome or Chromium>          # else the acceptance scripts' discovery
bash vega/scripts/tests/matrix.sh release -j 6 --matrix <dir outside the repository>
```

| command | what runs | cap (s) |
|---|---|---|
| `matrix.sh static` | T0: every `test_catalog_sync.py`, the registry, declared coverage, the template lint, the style lock, `test_no_planned.py`, `check_unattended.py`, `verifier_gate.py` | 60 |
| `matrix.sh unit [-j N]` | T1: every `test_*.py` of every skill, the backend smoke cells, this program's own tests | 570 |
| `matrix.sh lanes [--lanes L01,L02 \| --tier T] [--from S] [--to S] [--strict]` | T2: the lanes of `lanes.json`, each from its fixture through `page` (L23 stops at `eval`; G00 starts at `golden`) | 1230 (release) |
| `matrix.sh negatives` | T3: every `run_negatives*.py` (the existing suites on L01's work dir, the new ones on the matrix dir) | 360 |
| `matrix.sh report` | T4: `coverage_report.py` (coverage.json / coverage.md) and `adaptation_report.py` (AD1–AD7) | 10 |
| `matrix.sh pr \| release \| nightly` | the composite gates: static, unit, the tier's lanes, negatives, report, all strict | 1440 / 1990 / 2760 |

A composite run is strict: a component or a stage that could not run fails it, because a gate
that did not run proves nothing. A tier that runs over its cap by more than 50 % also fails it
(`run_matrix.py` `TIER_CAPS_S`). The matrix directory receives `skills/` (a copy of the tracked
tree plus domain-forge), `lanes/<id>/` per lane, `summary.json`, `timing.json`, `timing.md`,
`coverage.md` and `adaptation.md`.

Tiers: `pr` is L01, L02, L05, L10, L17, L19, L20, L22; `release` is L01–L23; `nightly` is
`release` plus G00, the golden CIC lane, which needs `CIDS_ROOT` (the recorded R7 campaign and its
shaped parts) and runs about 8 minutes on CPU.

## The release gate (gate D, D6 §16.4)

1. `matrix.sh release` exits 0: every lane passes, the coverage verdict is PASS, every new gate is
   hit, the adaptation report passes, the timing caps hold.
2. `matrix.sh nightly` exits 0 on the release commit, G00 included.
3. `verifier_gate.py --release` passes: every skill's static verifier recorded SHIP on the
   prompt digests that ship (`verifier-runs/`), and `release/rehearsal.json` records a passing
   T5 rehearsal with its session id. The run's own check (`T5-run`) may instead say
   `pass-with-fixed-defects` for a rehearsal of record that met package defects (the user's
   decision of 2026-10-12): the check says `completed: true` and names its `defects`, every other
   (product) check says `pass`, and the record's `defects_fixed` lists exactly those defects as
   `{id, summary, fix, tests}`; the gate proves each fix commit is an ancestor of the released
   head and itself added or changed one of the named tests, which exist in the tree. Anything
   less is a finding; an accepted record prints a `NOTE:` line per fixed defect.
4. `coverage.md`, `adaptation.md` and `timing.md` from the release matrix are attached to the
   release notes.

## The T5 rehearsal (D6 §11.4)

The lanes prove the scripts and the page with a scripted author (`lanekit/authoring.py`). The
rehearsal proves that a model following the prompts carries a new dataset through the chain
unattended and writes a story that passes the contract. Procedure:

1. Install the package into a fresh project's `.claude/skills/`: the nine package skills
   (`git archive`), the required external skills (`domain-forge`, `spec-analysis`,
   `inferred-questions`, `model-chat`, `blueprint`, `document-project`, `architect`, `improve`,
   `effective-java`) with `patches/` applied. `env_check.py --env` must list every required
   skill as installed.
2. Generate a fresh synthetic dataset (`lanekit.datasets.load("landcover").make(seed, rows)`,
   with a seed no lane uses) into the project's `data/`.
3. Run, from the project directory,
   `claude -p "/vega data/landcover.csv --goal show --target cover --ship-anyway \"<reason>\" --unattended" --setting-sources project
   --strict-mcp-config --permission-mode bypassPermissions --output-format stream-json --verbose`.
   `--setting-sources project` keeps any other installed copy of the skills out of the session.
   `--target cover` is the user's answer to the forge's partition fork, given up front: the
   landcover label is an observed survey class no other column computes, so the geometry admits
   no candidate of its own, and an unattended run that names no target abstains and stops
   before `net` (dataset-forge step 4, the chain map's `forge-data` and `shape` rows).
   `--ship-anyway "<reason>"` is the user's answer to `pack`'s fork, given up front: the
   rehearsal tests the chain through the showcase, and a network that does not beat its
   baselines on a given seed (the verdict *not earned*) would otherwise stop it at `pack` —
   the lanes pass the same stance. The reason the round used goes in `release/rehearsal.json`.
   `bypassPermissions` is needed because the ledger and the forge memory live under `.claude/`,
   which the CLI refuses to write in any other non-interactive mode. Deny the publishing tools:
   nobody can approve a publish in an unattended run.
4. Check the products: `check_brief.py` on every `docs/*-brief.md`; `validate_model.py` and
   the stage's `smoke_*.py --strict` on every layered HTML; the served showcase with
   `smoke_show.py --serve --strict` and `inspect_check.py`; the vega-show gates (`build_showcase.py
   --check`) on the model-authored story.
5. Record the session id, the commit and each check's verdict in `release/rehearsal.json`
   (`vega/rehearsal@1`). A defect the run met and worked around is recorded in `defects_fixed`
   once it is fixed with a test (see the release gate, step 3); it is never folded into a plain
   `pass`.

## Files

- `matrix.sh`, `run_matrix.py`, `lanekit/`: the scheduler, the lane runner, the fixture
  generators (`lanekit/datasets/`), the scripted author and the stage drivers.
- `lanes.json`, `lanes.schema.json`: the lane registry (L01–L23, G00).
- `golden/`: the CIC golden lane (re-pack of R7, authoring, layout and content parity).
- `negatives/`: the negatives kit and the grandfathered list.
- `coverage_report.py`, `coverage_gates.py`, `coverage_md.py`, `adaptation_report.py`: the T4
  reports.
- `serving_check.py`, `inspect_check.py`, `inspect_*.py`, `oracles.py`, `page_smoke.py`,
  `page_smoke_checks.py`, `style_lock.py`, `lint_templates.py`: the live serving, inspection and
  page checks the lanes call.
- `check_brief.py`, `check_unattended.py`, `check_kit.py`, `verifier_gate.py`,
  `verifier-runs/`: the static checks and the recorded verifier runs.
- `known_flake.py`, `lanekit/flakes.json`: the one-retry wrapper and its known signatures.
