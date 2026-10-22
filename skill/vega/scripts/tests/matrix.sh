#!/usr/bin/env bash
# The verification program's entry point (D6 §2.1): one command per tier.
#
#   ./matrix.sh static | unit | negatives | report         the tiers T0, T1, T3, T4 alone
#   ./matrix.sh lanes [--lanes L01,L02 | --tier pr|release|nightly] [--changed RANGE]
#                     [--from STAGE] [--to STAGE] [-j N] [--strict]      T2
#   ./matrix.sh pr | release | nightly                     the composite gates (strict)
#   ./matrix.sh validate | list | observed FILE...         the registry, the lanes, a result
#
# Every run works in a matrix dir outside the repository (--matrix DIR, or $VEGA_MATRIX_DIR,
# else ~/.cache/vega-matrix) and exits 0 green, 1 red, 2 on a usage error or a refused --from.
#
# Needs: the interpreter that holds torch, pandas, pyarrow, scikit-learn, nbconvert and
# jsonschema ($PYTHON, default the first python3 on PATH); a headless Chromium ($CHROME,
# default the acceptance scripts' discovery); domain-forge beside the skills or at
# $DOMAIN_FORGE_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
export PYTHON
export CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || ls -d /opt/pw-browsers/chromium*/chrome-linux/chrome 2>/dev/null | head -1 || true)}"

if [ "$#" -eq 0 ]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
exec "$PYTHON" "$HERE/run_matrix.py" "$@"
