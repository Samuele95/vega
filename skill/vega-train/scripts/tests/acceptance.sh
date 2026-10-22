#!/usr/bin/env bash
# The /vega-train acceptance run: a real campaign on a real forged model, then every gate
# against a seeded defect, then the prose-vs-machine catalog check.
#
#   ./acceptance.sh [work-dir]
#
# The work dir must already contain a /vega-forge run — the campaign lives in the run directory
# the forge created. With no argument it uses the forge's own acceptance directory, running the
# forge's acceptance first if it is not there.
#
# Needs: torch, pandas, pyarrow, scikit-learn (optuna optional — without it the study is a
# random search and says so); nbconvert and a kernel; a headless Chromium for the layer smoke.
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(dirname "$TESTS")"
SKILL="$(dirname "$S")"
SKILLS="$(dirname "$SKILL")"
FORGE="$SKILLS/vega-forge/scripts"
DF="$SKILLS/domain-forge/scripts"
# validate_model.py runs through the known-flake wrapper: one retry, only on a signature of
# vega/scripts/tests/lanekit/flakes.json (the domain-forge profile race, patches/domain-forge/)
KF="$SKILLS/vega/scripts/tests/known_flake.py"
WORK="${1:-$FORGE/tests/.vega-acceptance}"

CAMP="vega-runs/2026-09-05T11-03Z"
MODEL="orders.domain.analysis.shaped.vega.html"
OUT="orders.domain.analysis.shaped.vega.train.html"
BUDGET="${BUDGET:-6}"
TRIALS="${TRIALS:-5}"
RUNGS="${RUNGS:-R0 R1 R6}"
export CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || ls -d /opt/pw-browsers/chromium*/chrome-linux/chrome 2>/dev/null | head -1 || true)}"

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

step "0. the machine catalog agrees with the prose"
python3 "$TESTS/test_catalog_sync.py"

if [ ! -f "$WORK/$MODEL" ]; then
  step "0b. no forged model in $WORK — running /vega-forge's acceptance first"
  "$FORGE/tests/acceptance.sh" "$WORK"
fi
cd "$WORK"

step "0c. reset the campaign (a campaign is append-only; the acceptance starts a fresh one)"
rm -rf "$CAMP/runs" "$CAMP/hpo" "$CAMP/experiments.json" "$CAMP/ladder.json" \
       "$CAMP/campaign.json" "$CAMP/train.ipynb" "$CAMP/training-layer.json" "$OUT"

step "1. plan the ladder (plan_ladder.py)"
python3 "$S/plan_ladder.py" --campaign "$CAMP" --model "$MODEL" --budget "$BUDGET" \
    --hpo "$TRIALS"

step "2. run the rungs (trainer.py) — $RUNGS"
for R in $RUNGS; do
  python3 "$S/trainer.py" --campaign "$CAMP" --run "$R" --budget-min 0.5 \
    | grep -E '^(OK|WARN|ERROR):'
done

step "3. the study (hpo.py)"
python3 "$S/hpo.py" --campaign "$CAMP" --study H1 --budget "trials=$TRIALS,minutes=3" \
  | grep -E '^(OK: H1|WARN|ERROR):'

step "4. the selection (select.py)"
python3 "$S/select.py" --campaign "$CAMP"

step "5. the notebook (nb_emit.py --stage train), executed in place"
cp "$TESTS/training-layer.authoring.json" "$CAMP/training-layer.json"
python3 "$FORGE/nb_emit.py" --stage train --from "$CAMP" --out "$CAMP/train.ipynb"
( cd "$CAMP" && python3 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 train.ipynb 2>&1 | grep -Ev 'WARNING|Kernel is running' )

step "6. write the layer (apply_training_layer.py)"
python3 "$S/apply_training_layer.py" "$MODEL" --data "$CAMP/training-layer.json" \
    --notebook "$CAMP/train.ipynb" --out "$OUT"

step "7. verify the layered file"
python3 "$KF" run --stage train -- python3 "$DF/validate_model.py" "$OUT" | tail -3
python3 "$S/smoke_training.py" "$OUT" --strict | tail -1

step "8. every gate refuses what it says it refuses"
python3 "$TESTS/run_negatives.py" "$WORK" | tail -3

step "acceptance passed"
printf 'work dir  %s\ncampaign  %s\nlayer     %s\nnotebook  %s\n' \
    "$WORK" "$WORK/$CAMP" "$WORK/$OUT" "$WORK/$CAMP/train.ipynb"
