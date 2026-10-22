#!/usr/bin/env bash
# The /vega-show acceptance run (D5 §12.1, D4 §17.8, D6 §12.5; BUILD-PLAN H09): the page template
# held to its gates, the stage run on two lanes that differ in task, family, lexicon and report
# mode, every admitted cell of the catalog through the stage, every page fixture walked, every
# gate of the stage against a seeded defect, and the static verifier's SHIP still current.
#
#   ./acceptance.sh [matrix-dir]
#
# The matrix dir (default $VEGA_MATRIX_DIR, else ~/.cache/vega-matrix) is the lane program's: it
# must hold lanes L01, L02, L03 and L05 through their report stage (`matrix.sh lanes --lanes
# L01,L02,L03,L05 --to report`); a lane that is not there is run up to `report` first. Nothing is
# written into the lanes: every step works on a copy in a scratch directory ($TMPDIR).
#
# Needs: the interpreter that runs the chain (torch, pandas, pyarrow, scikit-learn, jsonschema,
# fastapi, uvicorn, pytest); a headless Chromium ($CHROME); node for the page's own tests;
# domain-forge beside the skills.
#
#  0. the template and the catalog: test_catalog_sync, the digit rule, the .aux adapter, the page
#     template's identity and literal gates (the literal lint over its served files among them,
#     PG3), its size, page.json, the style lock, the page's node tests, and conformance.py
#     --size-only over vega-show/scripts (C11–C13).
#  1. the lanes are in the matrix.
#  2. lane A (L01, orders — binary, mlp-residual, batch blueprint packed --serve): without a report
#     (brief links), then with the report fixture (linked), then the staleness transitions; the
#     served walk, the layer, invariants 13–16, the tab, the bundle untouched (_acceptance_lanes.py).
#  3. lane B (L02, land cover — multiclass, mlp-plain, the stub report).
#  4. the page fixtures: make_page_fixtures.py, smoke_page.py --fixture all --strict.
#  5. the coverage matrix: every admitted (family, task, shape, goal) cell through extract → draft
#     → autoauthor → resolve → build → check, and smoke_show --mock over every fixture case.
#  6. run_negatives.py: every gate id of catalog.json the id of a passing seeded case.
#  7. the static verifier's recorded SHIP is current (verifier_gate.py --skill vega-show; the
#     verifier itself runs in a fresh context, recorded in vega/scripts/tests/verifier-runs/).
# The CIC golden lane (G00: the CIDS bundle, its carried plan and the parity probe) is the lane
# program's, with CIDS_ROOT; it is said, not run, here.
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(dirname "$TESTS")"
SKILL="$(dirname "$S")"
SKILLS="$(dirname "$SKILL")"
VTESTS="$SKILLS/vega/scripts/tests"
MATRIX="$(cd "${1:-${VEGA_MATRIX_DIR:-$HOME/.cache/vega-matrix}}" 2>/dev/null && pwd || echo "${1:-${VEGA_MATRIX_DIR:-$HOME/.cache/vega-matrix}}")"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/vega-show-acceptance-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
export CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || ls -d /opt/pw-browsers/chromium*/chrome-linux/chrome 2>/dev/null | head -1 || true)}"
export PYTHONDONTWRITEBYTECODE=1
PYTEST=(python3 -m pytest -q -p no:cacheprovider)

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

if [ -z "$CHROME" ]; then
  echo "ERROR: no headless Chromium (\$CHROME): the served walks, the tab smoke and the page fixtures need one"
  exit 2
fi

step "0. the template and the catalog"
python3 "$TESTS/test_catalog_sync.py"
"${PYTEST[@]}" "$TESTS/test_digit_rule.py" "$TESTS/test_aux_adapter.py" "$TESTS/test_page_template.py" \
    "$TESTS/test_page_size.py" "$TESTS/test_page_config.py"
python3 "$VTESTS/style_lock.py" check --strict
if command -v node >/dev/null; then
  node --test "$SKILL"/assets/page/tests/*.test.js 2>&1 | tail -12
else
  echo "WARN: check not run: the page's node tests (assets/page/tests) — node is not on PATH"
fi
python3 "$SKILLS/vega-forge/scripts/conformance.py" --size-only "$S"

step "1. the lanes are in the matrix ($MATRIX)"
MISSING=()
for lane in L01 L02 L03 L05; do
  [ -f "$MATRIX/lanes/$lane/stamps/report.json" ] || MISSING+=("$lane")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "WARN: running ${MISSING[*]} up to report first (they are not in $MATRIX)"
  (IFS=,; "$VTESTS/matrix.sh" lanes --matrix "$MATRIX" --lanes "${MISSING[*]}" --to report -j 2)
fi
echo "OK: L01 L02 L03 L05 went through report in $MATRIX"

step "2. lane A — L01 without a report, with the report fixture, then stale"
python3 "$TESTS/_acceptance_lanes.py" "$MATRIX" --lane L01 --tmp "$SCRATCH/lanes"

step "3. lane B — L02 with its stub report"
python3 "$TESTS/_acceptance_lanes.py" "$MATRIX" --lane L02 --tmp "$SCRATCH/lanes"

step "4. the page fixtures, built from the live backend and driven"
python3 "$TESTS/make_page_fixtures.py" --out "$SCRATCH/fixtures" > "$SCRATCH/fixtures.log" \
  || { tail -20 "$SCRATCH/fixtures.log"; exit 1; }
echo "OK: $(ls -d "$SCRATCH"/fixtures/*/ | wc -l) page fixture cases built"
python3 "$S/smoke_page.py" --fixture all --fixtures "$SCRATCH/fixtures" --work "$SCRATCH/smoke-page" --strict | tail -5

step "5. the coverage matrix — every admitted cell through the stage, every fixture case walked"
python3 "$TESTS/test_coverage_matrix.py" --fixtures "$SCRATCH/fixtures"

step "6. the seeded negatives — every gate of catalog.json caught by a passing case"
mkdir -p "$MATRIX/negatives"
python3 "$TESTS/run_negatives.py" "$MATRIX" --json "$MATRIX/negatives/vega-show-run_negatives.json" | tail -4

step "7. the static verifier's SHIP is current"
python3 "$VTESTS/verifier_gate.py" --skill vega-show
echo "WARN: check not run: the CIC golden lane (G00, CIDS_ROOT) — the lane program runs it"

step "acceptance passed"
