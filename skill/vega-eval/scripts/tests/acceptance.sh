#!/usr/bin/env bash
# The /vega-eval acceptance run: one sealed look at a real test part on a real trained
# campaign, then the notebook that proves the numbers without reopening it, then the layer,
# then every gate against a seeded defect, then the prose-vs-machine catalog check.
#
#   ./acceptance.sh [work-dir]
#
# The work dir must already contain a /vega-train campaign with a selection. With no argument
# it uses the forge's own acceptance directory, running the forge's and the trainer's
# acceptance first if the campaign is not there.
#
# Needs: torch, pandas, pyarrow, scikit-learn, matplotlib; nbconvert and a kernel; a headless
# Chromium for the layer smoke.
#
# The evaluation is deliberately re-run from scratch here: a campaign whose test part has
# already been opened would make the acceptance's own numbers exposure-2 numbers, and the
# acceptance would then be testing the thing it exists to prevent.
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(dirname "$TESTS")"
SKILL="$(dirname "$S")"
SKILLS="$(dirname "$SKILL")"
FORGE="$SKILLS/vega-forge/scripts"
TRAIN="$SKILLS/vega-train/scripts"
DF="$SKILLS/domain-forge/scripts"
# validate_model.py runs through the known-flake wrapper: one retry, only on a signature of
# vega/scripts/tests/lanekit/flakes.json (the domain-forge profile race, patches/domain-forge/)
KF="$SKILLS/vega/scripts/tests/known_flake.py"
WORK="${1:-$FORGE/tests/.vega-acceptance}"

CAMP="vega-runs/2026-09-05T11-03Z"
MODEL="orders.domain.analysis.shaped.vega.train.html"
OUT="orders.domain.analysis.shaped.vega.train.eval.html"
BOOT="${BOOT:-1000}"
export CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || ls -d /opt/pw-browsers/chromium*/chrome-linux/chrome 2>/dev/null | head -1 || true)}"

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

step "0. the machine catalog agrees with the prose"
python3 "$TESTS/test_catalog_sync.py"

step "0a. every reported number is read at the threshold the run chose"
python3 "$TESTS/test_scoring_threshold.py"

step "0a'. the analysis's segments are cut from the test rows, or the layer says why (a real /data-lens run)"
python3 -m pytest -q -p no:cacheprovider "$TESTS/test_lens_segments.py" | tail -n 1

if [ ! -f "$WORK/$MODEL" ]; then
  step "0b. no trained campaign in $WORK — running /vega-train's acceptance first"
  "$TRAIN/tests/acceptance.sh" "$WORK"
fi
cd "$WORK"

step "0c. reset the seal (the acceptance must measure a first exposure, not its own last run)"
rm -rf "$CAMP/eval" "$CAMP/test_seal.json" "$CAMP/evaluate.ipynb" \
       "$CAMP/evaluation-layer.json" "$OUT"
python3 - "$CAMP/campaign.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text()); d.pop("test_seal", None)
p.write_text(json.dumps(d, indent=2) + "\n")
PY

step "1. the one look (evaluate.py) — seal, predict, baselines, bootstrap, verdict"
python3 "$S/evaluate.py" --campaign "$CAMP" --n-boot "$BOOT" | grep -E '^(OK|WARN|ERROR):'

step "2. the seal recorded exactly one exposure"
python3 - "$CAMP/test_seal.json" <<'PY'
import json, sys, pathlib
seal = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert seal["exposures"] == 1, f"exposures {seal['exposures']}, expected 1"
assert seal["test"]["digest_matches_manifest"] is True, "the test digest does not match"
assert seal.get("token"), "the seal carries no token"
print(f"OK: exposure 1 at {seal['opened_at']}, digest {seal['test']['digest'][:19]}…, "
      f"reason {seal['history'][0]['reason']!r}")
PY

step "3. the notebook (nb_emit.py --stage eval), executed in place"
# The authoring file is the acceptance's judgement, composed from THIS run's records
# (write_authoring.py): the numbers are read at the record's rounding and the sentences follow the
# facts, so a template change that moves the numbers cannot leave the fixture asserting another
# run's verdict. `from_eval` points at the run the seal records.
RUN_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"])' "$CAMP/test_seal.json")"
python3 "$TESTS/write_authoring.py" "$CAMP" "$RUN_ID" "$CAMP/evaluation-layer.json"
cp "$CAMP/evaluation-layer.json" "$CAMP/evaluation-layer.base.json"   # the negatives rewrite one reading at a time from this base
python3 "$FORGE/nb_emit.py" --stage eval --from "$CAMP" --out "$CAMP/evaluate.ipynb"
( cd "$CAMP" && python3 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 evaluate.ipynb 2>&1 | grep -Ev 'WARNING|Kernel is running' )

step "4. the notebook did not reopen the test part"
python3 - "$CAMP/evaluate.ipynb" "$CAMP/test_seal.json" <<'PY'
import json, sys, pathlib
nb = json.loads(pathlib.Path(sys.argv[1]).read_text())
text = "\n".join("".join(o.get("text") or [])
                 for c in nb["cells"] for o in (c.get("outputs") or []))
assert "VEGA-EVAL: sealed" in text, "the notebook does not report a sealed run"
seal = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert seal["exposures"] == 1, "executing the notebook changed the exposure count"
print("OK: VEGA-EVAL: sealed — the notebook recomputed every number from the stored "
      "predictions, and the seal still records one exposure")
PY

step "5. write the layer (apply_evaluation_layer.py)"
python3 "$S/apply_evaluation_layer.py" "$MODEL" --data "$CAMP/evaluation-layer.json" \
    --notebook "$CAMP/evaluate.ipynb" --out "$OUT"

step "6. verify the layered file"
python3 "$KF" run --stage eval -- python3 "$DF/validate_model.py" "$OUT" | tail -3
python3 "$S/smoke_evaluation.py" "$OUT" --strict | tail -1

step "7. every gate refuses what it says it refuses"
python3 "$TESTS/run_negatives.py" "$WORK" | tail -3

step "8. the seal survived the negatives at one exposure"
python3 - "$CAMP/test_seal.json" <<'PY'
import json, sys, pathlib
seal = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert seal["exposures"] == 1, (
    f"the negatives left the seal at {seal['exposures']} exposures; a refused run must not "
    "cost a look at the test part")
print("OK: exposures still 1 — every refusal happened before the seal was written")
PY

step "acceptance passed"
printf 'work dir  %s\ncampaign  %s\nevaluation %s\nlayer     %s\nnotebook  %s\n' \
    "$WORK" "$WORK/$CAMP" "$WORK/$CAMP/eval" "$WORK/$OUT" "$WORK/$CAMP/evaluate.ipynb"
