#!/usr/bin/env bash
# The /dataset-shaper acceptance run: every fixture recipe through the real scripts as the chain
# runs them, then the graph step's properties, then every gate against a seeded defect.
#
#   ./acceptance.sh [work-dir]        (default: .vega-acceptance/ beside this script)
#
# Each fixture goes the whole way: the recipe checker (`shape.py --check-only`), the executor,
# the verifier (`verify_shape.py`, the graph checks included), the determinism check
# (`shape.py --check` re-runs the reproduction script), the inference twin
# (`shape.py --transform-only`) and its twin check on the test and the valid part. The fixtures
# are the three graph recipes (graph-keypair: key_pair at level node, the lane L19's shape;
# graph-knn: knn at level node, L21's; graph-level: key_pair within a graph id, L22's) and the
# orders recipe (rows, the chain's own first link). The happy path is the easy half:
# `run_negatives_graph.py` (SG1-SG7), `run_negatives_leakage.py` (SL1-SL13) and
# `run_negatives_layer.py` (SM1-SM2b: the shape layer's manifest_digest) seed one defect at a
# time and assert that the right gate refuses it. Both halves have to pass.
#
# Needs: pandas, numpy, pyarrow, scikit-learn, pytest; scipy for the spatial and box-cox steps.
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(dirname "$TESTS")"
SKILL="$(dirname "$S")"
SKILLS="$(dirname "$SKILL")"
FIX="$SKILL/fixtures"
WORK="${1:-$TESTS/.vega-acceptance}"
FIXTURES="${FIXTURES:-graph-keypair graph-knn graph-level orders}"

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

step "0. the transform steps, the lag, the leakage guards and the group split's key (unit)"
python3 -m pytest -q -p no:cacheprovider "$TESTS/test_transform_steps.py" | tail -n 1
python3 "$TESTS/test_transform_lag.py" | tail -n 1
python3 "$TESTS/test_leakage_guards.py" | tail -n 1
python3 "$TESTS/test_split_key.py" | tail -n 1
python3 -m pytest -q -p no:cacheprovider "$TESTS/test_rejected_derivation.py" | tail -n 1
python3 -m pytest -q -p no:cacheprovider "$TESTS/test_determinism_record.py" | tail -n 1
python3 -m pytest -q -p no:cacheprovider "$TESTS/test_shape_layer_digest.py" | tail -n 1

step "0b. a fresh work dir: $WORK"
rm -rf "$WORK"; mkdir -p "$WORK"
# the recipes name their input relative to the skills root (dataset-shaper/fixtures/…)
cd "$SKILLS"

N=1
for F in $FIXTURES; do
  R="$FIX/$F.recipe.json"
  OUT="$WORK/$F"
  step "$N. $F — check, shape, verify, reproduce, twin"
  python3 "$S/shape.py" --recipe "$R" --check-only | tail -n 1
  python3 "$S/shape.py" --recipe "$R" --out-dir "$OUT" --format parquet --seed 7 \
    | grep -E '^(OK: wrote .*edges|OK: manifest.json|WARN|ERROR)'
  python3 "$S/verify_shape.py" --recipe "$R" --out-dir "$OUT" --json "$WORK/$F.verification.json" \
    | tail -n 1
  python3 "$S/shape.py" --check --out-dir "$OUT" | grep -E '^OK: determinism'
  python3 "$S/shape.py" --transform-only --recipe "$R" --fitted "$OUT/fitted.json" \
      --emit "$OUT/preprocess.py" | tail -n 1
  for PART in test valid; do
    python3 "$S/verify_shape.py" --transform-only --recipe "$R" --out-dir "$OUT" \
        --module "$OUT/preprocess.py" --part "$PART" | tail -n 1
  done
  python3 - "$R" "$WORK/$F.verification.json" "$OUT" <<'PY'
import json, sys, pathlib
recipe = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
report = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
manifest = json.loads((pathlib.Path(sys.argv[3]) / "manifest.json").read_text(encoding="utf-8"))
graph = any(s.get("op") == "graph_edges" for s in recipe["steps"])
files = manifest["outputs"]["files"]
assert {"train", "valid", "test"} <= set(files), f"the parts written: {sorted(files)}"
if graph:
    assert report.get("graph") == "pass", f"verification.graph is {report.get('graph')!r}"
    assert "edges" in files and manifest.get("graph"), "no edges part, or no manifest graph block"
    g = manifest["graph"]
    print(f"OK: graph {g.get('method')} at level {g.get('level')} — "
          f"{files['edges'].get('rows')} edge row(s); verification.graph pass")
else:
    assert report.get("graph") in (None, "n/a"), f"verification.graph is {report.get('graph')!r}"
    print(f"OK: {len(files)} part(s) written, verified, reproduced and twinned; no graph step")
PY
  N=$((N + 1))
done

step "$N. the graph step's properties (test_graph_edges.py)"
python3 "$TESTS/test_graph_edges.py" "$WORK/graph-properties" | tail -n 1
N=$((N + 1))

step "$N. every graph gate refuses what it says it refuses (SG1-SG7)"
python3 "$TESTS/run_negatives_graph.py" "$WORK/neg-graph" | tail -n 3
N=$((N + 1))

step "$N. every leakage guard refuses what it says it refuses (SL1-SL13)"
python3 "$TESTS/run_negatives_leakage.py" "$WORK/neg-leakage" | tail -n 3
N=$((N + 1))

step "$N. the shape layer carries the manifest file's digest, or is refused (SM1-SM2b)"
python3 "$TESTS/run_negatives_layer.py" "$WORK" | tail -n 1

step "acceptance passed"
printf 'work dir  %s\nfixtures  %s\n' "$WORK" "$FIXTURES"
