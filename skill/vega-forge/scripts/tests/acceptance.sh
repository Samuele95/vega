#!/usr/bin/env bash
# The /vega-forge acceptance run: the whole procedure on a real fixture, then every gate
# against a seeded defect, then the prose-vs-machine catalog check.
#
#   ./acceptance.sh [work-dir]        (default: .vega-acceptance/ beside this script)
#
# The happy path is the easy half. A gate that never fires is indistinguishable from no gate,
# so `run_negatives.py` seeds one defect at a time and asserts that the right script refuses
# it with a message that names the fix. Both halves have to pass.
#
# Steps 8-10 are the skill's suites, which read no work dir: the backend coverage matrix on one
# cell per family (B19, `run_matrix.py --cells smoke`, 21 cells), the inspection zoo (I10, every
# admitted cell explained) and the template, catalog and layout tests. The lane program runs
# this script as lane L01's forge stage with VEGA_ACCEPTANCE_SUITES=lane: the lane proves the
# chain, and its unit tier runs these same suites once for the whole matrix
# (vega/scripts/tests/lanekit/tiers.py: `backend-smoke-cells` and every test_*.py), so the
# lane says so and does not run them twice.
#
# Needs: torch, pandas, pyarrow, scikit-learn, jsonschema>=4, pytest; nbconvert + a kernel to
# execute the notebook; a headless Chromium for the layer smoke test (set CHROME, or it is found
# on PATH).
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(dirname "$TESTS")"
SKILL="$(dirname "$S")"
DF="$(dirname "$SKILL")/domain-forge/scripts"
# validate_model.py runs through the known-flake wrapper: one retry, only on a signature of
# vega/scripts/tests/lanekit/flakes.json (the domain-forge profile race, patches/domain-forge/)
KF="$(dirname "$SKILL")/vega/scripts/tests/known_flake.py"
WORK="${1:-$TESTS/.vega-acceptance}"
SUITES="${VEGA_ACCEPTANCE_SUITES:-run}"

RUN="vega-runs/2026-09-05T11-03Z"
MODEL="orders.domain.analysis.shaped.html"
OUT="orders.domain.analysis.shaped.vega.html"
PIPE="blueprint-runs/2026-09-05T10-12Z/pipeline.json"
ARCH=".claude/architect-runs/2026-09-05T11-20Z"
export CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || ls -d /opt/pw-browsers/chromium*/chrome-linux/chrome 2>/dev/null | head -1 || true)}"

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }
case "$SUITES" in run|lane) ;; *)
  echo "ERROR: VEGA_ACCEPTANCE_SUITES is '$SUITES'; it is run (the default) or lane" >&2; exit 2 ;;
esac

step "0. the machine catalog agrees with the prose"
python3 "$TESTS/test_catalog_sync.py"

step "0b. build the fixture in $WORK"
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$TESTS/fixtures/dot-claude" "$WORK/.claude"
cp "$TESTS/architecture-layer.authoring.json" "$WORK/"
python3 "$TESTS/make_fixture.py" "$WORK"
cd "$WORK"

step "1. compile the spec (compile_spec.py --check-only)"
python3 "$S/compile_spec.py" --spec netspec.json --model "$MODEL" --pipeline "$PIPE" \
    --memory .claude/domain-forge-memory.md --check-only

step "2. emit the package and smoke the forward pass (emit_backend.py --smoke)"
python3 "$S/emit_backend.py" --spec netspec.json --pipeline "$PIPE" --out "$RUN" --smoke

step "3. the mechanical conformance gate (conformance.py)"
python3 "$S/conformance.py" --package "$RUN/orders_pipeline" --pipeline "$PIPE"
echo "   (the judgement half is /architect --report-only --baseline, invoked through the"
echo "    Skill tool; this run uses the recorded findings at $ARCH)"

step "4. the notebook (nb_emit.py), executed in place"
cp netspec.json "$RUN/"
cp architecture-layer.authoring.json "$RUN/architecture-layer.json"
python3 "$S/nb_emit.py" --stage forge --from "$RUN" --out "$RUN/engineer.ipynb"
( cd "$RUN" && python3 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 engineer.ipynb 2>&1 | grep -Ev 'WARNING|Kernel is running' )

step "5. write the layer (apply_architecture_layer.py)"
python3 "$S/apply_architecture_layer.py" "$MODEL" --data "$RUN/architecture-layer.json" \
    --notebook "$RUN/engineer.ipynb" --architect-run "$ARCH" --out "$OUT"

step "6. verify the layered file"
python3 "$KF" run --stage forge -- python3 "$DF/validate_model.py" "$OUT" | tail -3
python3 "$S/smoke_architecture.py" "$OUT" --strict | tail -1

step "7. every gate refuses what it says it refuses"
python3 "$TESTS/run_negatives.py" "$WORK" | tail -3

if [ "$SUITES" = lane ]; then
  step "8-10. the skill's suites"
  echo "NOTE: not run inside the lane (VEGA_ACCEPTANCE_SUITES=lane): the coverage matrix, the"
  echo "      inspection zoo and the template tests read no work dir; the matrix's unit tier runs"
  echo "      them (backend-smoke-cells, unit:vega-forge/scripts/tests/test_*.py)"
else
  step "8. the coverage matrix, one cell per family (run_matrix.py --cells smoke)"
  python3 "$TESTS/matrix/run_matrix.py" "$WORK/backend-matrix" --cells smoke --keep \
    | grep -E '^(OK: backend matrix|FAIL|ERROR)'

  step "9. the inspection zoo (test_inspect_zoo.py): every admitted cell explained"
  python3 -m pytest -q -p no:cacheprovider "$TESTS/test_inspect_zoo.py" | tail -n 1

  step "10. the templates, the explain catalog and the package layout"
  python3 -m pytest -q -p no:cacheprovider "$TESTS/test_explain_catalog_sync.py" \
      "$TESTS/test_serving_templates.py" "$TESTS/test_domain_plans.py" \
      "$TESTS/test_emit_layout.py" | tail -n 1
  python3 "$TESTS/test_template_discipline.py" | tail -n 1
fi

step "acceptance passed"
printf 'work dir  %s\nlayer     %s\nnotebook  %s\n' "$WORK" "$WORK/$OUT" "$WORK/$RUN/engineer.ipynb"
