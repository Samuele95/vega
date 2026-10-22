#!/usr/bin/env bash
# The /vega-pack acceptance run: the bundle built from a real evaluated campaign, proven by a
# round trip over every held-out row, then the card, the notebook and the layer, then every
# gate against a seeded defect, then the prose-vs-machine catalog check.
#
#   ./acceptance.sh [work-dir]
#
# The work dir must already contain a /vega-eval run. With no argument it uses the forge's own
# acceptance directory, running the earlier stages' acceptance first if the evaluation is not
# there.
#
# Needs: torch, pandas, pyarrow, scikit-learn, jsonschema>=4, pytest, fastapi and httpx (the
# bundle's own tests); nbconvert and a kernel; a headless Chromium for the layer smoke and the
# page fixtures; node >= 22 for the page's own tests.
#
# The bundle's central claim — that it runs without any skill installed — is tested the only
# way it can be: step 8 copies the bundle somewhere else and runs it with an empty environment
# and the skills unreachable.
#
# PACK_EXTRA (default empty) is appended to both pack.py calls (steps 1 and 9). The lane program
# runs this script as lane L01 with PACK_EXTRA=--serve, so the orders fixture's batch blueprint
# is packed with its online service (vega/scripts/tests/lanes.json, D6 §2.3).
#
# The v2.3 gates are seeded on the lanes whose shape they are about (run_negatives.py's N-S
# cases on L05, L06, L10, L19 and L23; run_negatives_showcase.py's SS and IN cases on L02, L10
# and L19). Step 10 runs both suites on the lane matrix NEG_MATRIX names; without one it first
# builds those six lanes to their pack stage in <work-dir>/neg-matrix (run_matrix.py lanes,
# about 4 minutes at -j 3; ~/.cache/vega-matrix/pack-acceptance-negatives when the work dir is
# inside the repository, as the default one is), so that no case is left not run.
#
# Steps 1b and 11 are the skill's suites, which read no work dir: the serving matrix, one family
# per cell (`serving_matrix.py --quick`, 17 runs), and the page the bundle's showcase slot
# serves (the page fixture matrix under `smoke_page.py --fixture all --strict`, every walk
# anchor resolved in a rendered package, the page's own node tests). The lane program runs this
# script as lane L01's pack stage with VEGA_ACCEPTANCE_SUITES=lane: the lane proves the chain on
# its own bundle, so the suites, and the showcase negatives that need every sibling lane packed,
# are left to the standalone run and to the matrix's negatives tier, and the lane says so.
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(dirname "$TESTS")"
SKILL="$(dirname "$S")"
SKILLS="$(dirname "$SKILL")"
FORGE="$SKILLS/vega-forge/scripts"
EVAL="$SKILLS/vega-eval/scripts"
DF="$SKILLS/domain-forge/scripts"
WORK="${1:-$FORGE/tests/.vega-acceptance}"
case "$WORK" in /*) ;; *) WORK="$PWD/$WORK" ;; esac     # later steps name it after `cd "$WORK"`

CAMP="vega-runs/2026-09-05T11-03Z"
MODEL="orders.domain.analysis.shaped.vega.train.eval.html"
OUT="orders.domain.analysis.shaped.vega.train.eval.pack.html"
STANCE="${STANCE:-the calibrated score is what the downstream dispatcher needs}"
PACK_EXTRA="${PACK_EXTRA:-}"
KNOWN_FLAKE="$SKILLS/vega/scripts/tests/known_flake.py"
LANEKIT="$SKILLS/vega/scripts/tests"
SHOW="$SKILLS/vega-show/scripts"
PAGE="$SKILLS/vega-show/assets/page"
SUITES="${VEGA_ACCEPTANCE_SUITES:-run}"
NEG_LANES="L02,L05,L06,L10,L19,L23"
NEG_MATRIX="${NEG_MATRIX:-}"
export CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || ls -d /opt/pw-browsers/chromium*/chrome-linux/chrome 2>/dev/null | head -1 || true)}"

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

case "$SUITES" in run|lane) ;; *)
  echo "ERROR: VEGA_ACCEPTANCE_SUITES is '$SUITES'; it is run (the default) or lane" >&2; exit 2 ;;
esac

# the bundle's own tests in an emptied environment, judged by the lane program's skip policy
# (lanes.json allowed_skips, never a forbidden_skip_patterns one; D6 §6.6): <bundle> <home>
bundle_tests() {
  local log="$2/pytest.log" rc=0
  ( cd "$1" && env -i PATH=/usr/bin:/bin HOME="$2" PYTHONDONTWRITEBYTECODE=1 \
      CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -rs -p no:cacheprovider tests/ ) > "$log" 2>&1 \
    || rc=$?
  if [ "$rc" -ne 0 ]; then
    tail -n 30 "$log"; echo "FAIL: the bundle's own tests exited $rc in $1"; exit 1
  fi
  python3 - "$1" "$log" "$LANEKIT" <<'PYSKIP'
import json, pathlib, sys
bundle, log, lanekit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, lanekit)
from lanekit.stages import SKIP_LINE, packed_without_inspection, skips_outside_policy
text = log.read_text(encoding="utf-8")
defaults = json.loads((pathlib.Path(lanekit) / "lanes.json").read_text(encoding="utf-8"))["defaults"]
reasons = SKIP_LINE.findall(text)
bad = skips_outside_policy(reasons, defaults, (), packed_without_inspection(bundle))
assert not bad, f"{len(bad)} skip(s) outside the policy: {bad[:3]}"
print(f"OK: tests/ — {text.strip().splitlines()[-1].strip('= ')}; "
      f"{len(reasons)} skip(s), every one allowed")
PYSKIP
}

step "0. the machine catalog agrees with the prose"
python3 "$TESTS/test_catalog_sync.py"

if [ ! -f "$WORK/$MODEL" ]; then
  step "0b. no evaluated model in $WORK — running /vega-eval's acceptance first"
  "$EVAL/tests/acceptance.sh" "$WORK"
fi
cd "$WORK"

step "0c. reset the bundle (a pack run rebuilds it from scratch)"
rm -rf bundle "$OUT"
RUN="$(python3 -c "import json;print(json.load(open('$CAMP/test_seal.json'))['run'])")"

step "1. pack the bundle (pack.py) — provenance, transform, round trip, conformance"
# the blueprint's Monitor port says filled_by /vega-pack, so pack refuses to ship it unfilled:
# without a plan nothing is written (the refusal is the first thing proven), and with it the
# Monitor is rendered from the plan — the blueprint's one monitoring concern (dist_km drifts
# between the parts, [analysis:F3]) and the binary reading's own check on the answer; pack
# measures the references on train and valid. Both bundle shapes (steps 1 and 9) pack with it.
UNFILLED="$(python3 "$S/pack.py" --campaign "$CAMP" --shaped shaped --bundle-dir bundle \
    --pipeline blueprint-runs/2026-09-05T10-12Z/pipeline.json --model "$MODEL" \
    --ship-anyway "$STANCE" $PACK_EXTRA 2>&1 || true)"
case "$UNFILLED" in
  *"filled_by /vega-pack, and no --monitor-plan was given"*)
    [ ! -d bundle ] || [ -z "$(ls -A bundle)" ] \
      || { echo "FAIL: the unfilled-Monitor refusal wrote into bundle/"; exit 1; }
    echo "OK: a pack that would leave the blueprint's Monitor unfilled is refused, nothing written" ;;
  *) echo "FAIL: pack did not refuse an unfilled Monitor — $UNFILLED"; exit 1 ;;
esac
cat > monitor-plan.json <<'PLAN'
{"note": "the pack acceptance's plan: the blueprint's concern and the reading's answer check",
 "checks": [
  {"id": "drift-dist-km", "kind": "psi", "column": "dist_km", "concern": "[analysis:F3]",
   "title": "dist_km distribution against train"},
  {"id": "answer-recall", "kind": "class_recall", "classes": ["0", "1"], "label_column": "late",
   "concern": "[analysis:F2]", "title": "recall per class"}]}
PLAN
python3 "$S/pack.py" --campaign "$CAMP" --shaped shaped --bundle-dir bundle \
    --pipeline blueprint-runs/2026-09-05T10-12Z/pipeline.json --model "$MODEL" \
    --monitor-plan monitor-plan.json \
    --ship-anyway "$STANCE" $PACK_EXTRA | grep -E '^(OK|WARN|ERROR):'
BUNDLE="$(ls -d bundle/*/ | head -1)"
BUNDLE="${BUNDLE%/}"

if [ "$SUITES" = lane ]; then
  step "1b. the serving matrix"
  echo "NOTE: not run inside the lane (VEGA_ACCEPTANCE_SUITES=lane): serving_matrix.py --quick"
  echo "      reads no work dir; the standalone pack acceptance runs it"
else
  step "1b. the serving matrix, one family per cell (serving_matrix.py --quick)"
  python3 "$TESTS/serving_matrix.py" "$WORK/serving-matrix" --quick --keep \
    | grep -E '^(OK: serving matrix: .* green|FAIL|ERROR)'
fi

step "2. the round trip covered every held-out row"
python3 - "$BUNDLE/MANIFEST.json" "$CAMP/eval/$RUN/predictions.test.parquet" <<'PY'
import json, sys, pathlib
import pandas as pd
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
stored = len(pd.read_parquet(sys.argv[2]))
trip = manifest["roundtrip"]
assert trip["status"] == "pass", f"round trip {trip['status']}"
assert trip["rows"] == stored, (
    f"the round trip covered {trip['rows']} row(s) of {stored} stored predictions; the "
    "contract says every test row, whatever --examples says")
assert trip["shaper_transform_check"] == "pass", "the shaper's twin check did not pass"
print(f"OK: {trip['rows']} of {stored} stored prediction(s) reproduced from raw rows, worst "
      f"|delta| {trip['max_abs_diff']} against {trip['tolerance']['value']:.3g} "
      f"({trip['tolerance']['rule']})")
PY

step "3. the bundle carries the serving slice and nothing that reaches the test part"
python3 - "$BUNDLE" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
package = root / manifest["package"]
for forbidden in ("ingestion", "training", "evaluation"):
    assert not (package / forbidden).exists(), (
        f"{forbidden}/ shipped with the bundle; ingestion/data.py is the one door to the "
        "sealed test part and it does not travel")
assert not list(root.rglob("__pycache__")), "the bundle ships this machine's bytecode"
print(f"OK: {manifest['package']}/ holds the serving slice; ingestion, training and "
      "evaluation are absent, and no bytecode shipped")
PY

step "3b. the inspection self-test passed and pinned the network (MANIFEST.serve.inspection)"
python3 - "$BUNDLE" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
record = (manifest.get("serve") or {}).get("inspection")
assert isinstance(record, dict), "MANIFEST.serve.inspection is missing: pack records it in every mode"
assert record.get("enabled") is True, f"inspection enabled {record.get('enabled')!r}"
test = record.get("self_test") or {}
assert test.get("ready") is True, f"the inspection self-test is not ready: {test.get('why')!r}"
pins = record.get("network_digest") or {}
assert {"full", "summary"} <= set(pins) and all(str(v).startswith("sha256:") for v in pins.values()), \
    f"the network digest is not pinned in both weights modes: {pins}"
files = record.get("files") or []
missing = [f for f in files if not (root / manifest["package"] / f).is_file()]
assert files and not missing, f"inspection files missing from the bundle: {missing or 'none listed'}"
print(f"OK: inspection enabled, self-test ready (fidelity {test['fidelity']['max_abs_diff']:.3g} "
      f"within {test['fidelity']['tolerance']:.3g}), {len(files)} module(s), network pinned in "
      f"{', '.join(sorted(pins))}")
PY

step "4. the model card (model_card.py)"
cp "$TESTS/bundle-layer.authoring.json" "$BUNDLE/bundle-layer.json"
python3 - "$BUNDLE/bundle-layer.json" "$BUNDLE" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); doc = json.loads(p.read_text())
doc["from_bundle"] = sys.argv[2]
p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
PY
python3 "$S/model_card.py" --bundle "$BUNDLE" --model "$MODEL" \
    --intended-use "$BUNDLE/bundle-layer.json" --out "$BUNDLE/model-card.html"

step "5. the notebook (nb_emit.py --stage pack), executed inside the bundle"
python3 "$FORGE/nb_emit.py" --stage pack --from "$BUNDLE" --out "$BUNDLE/predict.ipynb" \
  | grep -E '^OK: wrote'
( cd "$BUNDLE" && python3 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 predict.ipynb 2>&1 | grep -Ev 'WARNING|Kernel is running' )
python3 - "$BUNDLE/predict.ipynb" <<'PY'
import json, pathlib, sys
nb = json.loads(pathlib.Path(sys.argv[1]).read_text())
text = "\n".join("".join(o.get("text") or [])
                 for c in nb["cells"] for o in (c.get("outputs") or []))
assert "VEGA-PACK: reproduced" in text, "the notebook did not reproduce the bundle's examples"
print("OK: VEGA-PACK: reproduced — the bundle's own notebook recomputed its examples and "
      "checked every digest, importing nothing from the chain")
PY

step "6. write the layer (apply_bundle_layer.py)"
python3 "$S/apply_bundle_layer.py" "$MODEL" --data "$BUNDLE/bundle-layer.json" \
    --notebook "$BUNDLE/predict.ipynb" --out "$OUT"

step "7. verify the layered file"
python3 "$KNOWN_FLAKE" run --stage pack -- python3 "$DF/validate_model.py" "$OUT" | tail -3
python3 "$S/smoke_bundle.py" "$OUT" --strict | tail -1

step "8. the bundle runs with the skills unreachable"
TMP="$(mktemp -d)"
cp -r "$BUNDLE" "$TMP/loose"
# the interpreter this suite runs with (the one that holds the bundle's requirements.txt), named
# before the environment is emptied — the emptied PATH would otherwise pick whatever python3
# /usr/bin has, which is a claim about this machine, not the bundle; run_negatives.py makes the
# same check the same way (sys.executable, an empty env)
PY="$(command -v python3)"
( cd "$TMP/loose" && env -i PATH=/usr/bin:/bin HOME="$TMP" "$PY" predict.py --check )
( cd "$TMP/loose" && env -i PATH=/usr/bin:/bin HOME="$TMP" "$PY" predict.py \
    --input examples/rows.csv --output "$TMP/preds.csv" )
# the library path /vega-show takes without a service: the inspector opens on the example rows,
# ready, and re-checks the network against the digest pack pinned
( cd "$TMP/loose" && env -i PATH=/usr/bin:/bin HOME="$TMP" PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
import predict
status = predict.load_inspector(".").status()
assert status["ready"] is True, f"load_inspector is not ready: {status.get('why')!r}"
assert status["pinned"] is True, "load_inspector did not check the network digest pack pinned"
opaque = status["coverage"]["opaque"]
print(f"OK: load_inspector ready and pinned ({status['network_digest'][:19]}…), "
      f"{status['coverage']['nodes']} node(s), opaque: {', '.join(opaque) or 'none'}")
PY
)
bundle_tests "$TMP/loose" "$TMP"
rm -rf "$TMP"

step "9. the other shape: --include full builds, fits and refuses the seal; its monitor plan"
rm -rf bundle-full
# the plan written in step 1
python3 "$S/pack.py" --campaign "$CAMP" --shaped shaped --bundle-dir bundle-full \
    --include full --pipeline blueprint-runs/2026-09-05T10-12Z/pipeline.json --model "$MODEL" \
    --monitor-plan monitor-plan.json \
    --ship-anyway "$STANCE" $PACK_EXTRA | grep -E '^(OK: the whole|OK: predict.py, train.py|OK: bundle)'
FULL="$(ls -d bundle-full/*/ | head -1)"; FULL="${FULL%/}"
python3 - "$FULL" "$BUNDLE" "$PACK_EXTRA" <<'PYFULL'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
assert manifest["include"] == "full", manifest["include"]
cli = {"predict.py", "train.py", "evaluate.py"}
got = set(manifest["entrypoints"])
if "--serve" in sys.argv[3].split():
    # the serving scaffold rides along in both bundles: --include full adds exactly the two
    # CLIs to the entry points of the serving bundle packed in step 1 with the same flags
    served = json.loads((pathlib.Path(sys.argv[2]) / "MANIFEST.json").read_text())
    assert got == set(served["entrypoints"]) | cli and "serve/app.py" in got, \
        (sorted(got), sorted(served["entrypoints"]))
else:
    assert got == cli, manifest["entrypoints"]
pkg = root / manifest["package"]
for present in ("ingestion/data.py", "training/loop.py", "evaluation/evaluate.py"):
    assert (pkg / present).is_file(), f"{present} did not ship with the full bundle"
bmap = json.loads((pkg / "blueprint-map.json").read_text())
out = [e for e in bmap["entries"] if str(e.get("status", "")).startswith("out of scope")]
assert not out, f"the full bundle still marks {[e.get('role') for e in out]} out of scope"
for name in ("train.py", "evaluate.py"):
    assert name in manifest["files"], f"{name} is not under digest"
print(f"OK: {manifest['include']} — the whole package, every port implemented, "
      f"{len(manifest['files'])} digest(s), entry points "
      f"{', '.join(manifest['entrypoints'])}")
PYFULL
( cd "$FULL" && python3 train.py --shaped ../../shaped --out "$(mktemp -d)" --epochs 2 \
    | grep -E '^(OK|NOTE):' )
( cd "$FULL" && python3 evaluate.py --part valid --shaped ../../shaped | head -2 )
# the refusal is the point, so the non-zero exit is expected and `pipefail` must not read it
# as the step failing
SEALED="$( cd "$FULL" && python3 evaluate.py --part test --shaped ../../shaped 2>&1 || true )"
case "$SEALED" in
  *"not a way past the seal"*)
    echo "OK: --part test refuses without the campaign's seal and its token" ;;
  *) echo "FAIL: the sealed part was not refused — $SEALED"; exit 1 ;;
esac
( cd "$FULL" && python3 predict.py --check | tail -1 )
python3 - "$FULL" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
assert "monitor-reference.json" in manifest["files"], "the monitor's reference is not under digest"
ref = json.loads((root / "monitor-reference.json").read_text())
ids = [c.get("id") for c in ref.get("checks") or []]
assert {"drift-dist-km", "answer-recall"} <= set(ids), f"the plan's checks are not all measured: {ids}"
bmap = json.loads((root / manifest["package"] / "blueprint-map.json").read_text())
monitor = [e for e in bmap["entries"] if e.get("role") == "Monitor"]
assert monitor and all(e.get("status") == "implemented" for e in monitor), \
    f"the Monitor adapter is not implemented: {[e.get('status') for e in monitor]}"
print(f"OK: monitor plan — {len(ids)} check(s) measured ({', '.join(ids)}); the Monitor adapter "
      "is implemented and its reference is pinned")
PY
# the monitor read back from the bundle, on raw rows that carry the truth its recall reads: the
# whole raw file, since a PSI is measured on 500 rows or more (its min_rows) and unmeasured below
( cd "$FULL" && PYTHONDONTWRITEBYTECODE=1 python3 - "$WORK/orders.csv" <<'PY'
import sys, predict
window = predict.read_rows(sys.argv[1], ".")
readings = predict.load_monitor(".").report(window, predict.load(".").predict(window))
checks = [r.window.check for r in readings]          # the reading's ReadingProvenance
assert {"drift-dist-km", "answer-recall"} <= set(checks), f"readings for {checks}"
unmeasured = sorted({c for c, r in zip(checks, readings) if r.verdict == "unmeasured"})
assert not unmeasured, f"{len(window)} row(s) left {unmeasured} unmeasured"
print(f"OK: load_monitor read {len(readings)} reading(s) over {len(window)} row(s): "
      + ", ".join(f"{c} {r.measure} {r.verdict}" for c, r in zip(checks, readings)))
PY
)

step "9b. neither bundle shape ships the sealed test part's membership"
VEGA_PACK_BUNDLES="$WORK/$BUNDLE:$WORK/$FULL" python3 -m pytest -q -p no:cacheprovider \
    "$TESTS/test_bundle_membership.py" | tail -n 1

if [ "$SUITES" = lane ]; then
  step "10. every gate refuses what it says it refuses"
  python3 "$TESTS/run_negatives.py" "$WORK" | tail -3
  echo "NOTE: run_negatives_showcase.py is not run inside the lane (VEGA_ACCEPTANCE_SUITES=lane):"
  echo "      it seeds L02's, L10's and L19's bundles, which may still be packing; the matrix's"
  echo "      negatives tier runs it once every lane has packed"
else
  if [ -z "$NEG_MATRIX" ]; then
    step "10a. the lanes the v2.3 gates are seeded on ($NEG_LANES), to their pack stage"
    # beside the work dir, unless that sits in the repository (the default work dir does): a
    # matrix dir must not, its skills tree is a copy of the repository's
    NEG_MATRIX="$WORK/neg-matrix"
    if git -C "$SKILLS" rev-parse --show-toplevel > /dev/null 2>&1; then
      case "$NEG_MATRIX/" in "$(git -C "$SKILLS" rev-parse --show-toplevel)"/*)
        NEG_MATRIX="${HOME}/.cache/vega-matrix/pack-acceptance-negatives" ;; esac
    fi
    rm -rf "$NEG_MATRIX"
    python3 "$LANEKIT/run_matrix.py" lanes --matrix "$NEG_MATRIX" --lanes "$NEG_LANES" \
        --to pack -j 3 | grep -E '^(OK|ERROR|FAIL): L[0-9]+ '
  fi
  step "10. every gate refuses what it says it refuses (lanes from $NEG_MATRIX)"
  python3 "$TESTS/run_negatives.py" "$WORK" --matrix "$NEG_MATRIX" \
      --json "$WORK/negatives-pack.json" | tail -3
  python3 "$TESTS/run_negatives_showcase.py" "$NEG_MATRIX" \
      --json "$WORK/negatives-showcase.json" | tail -1
  python3 - "$WORK/negatives-pack.json" "$WORK/negatives-showcase.json" <<'PY'
import json, sys, pathlib
for path in sys.argv[1:]:
    doc = json.loads(pathlib.Path(path).read_text())
    left = [c.get("id") or c.get("name") for c in doc.get("not_run") or []]
    assert not left, f"{pathlib.Path(path).name}: {len(left)} case(s) not run: {left}"
    assert not doc["failed"], f"{pathlib.Path(path).name}: {doc['failed']} case(s) failed"
    print(f"OK: {pathlib.Path(path).name} — {doc['passed']} case(s) passed, none failed, none left not run")
PY
fi

if [ "$SUITES" = lane ]; then
  step "11. the page the showcase slot serves"
  echo "NOTE: not run inside the lane (VEGA_ACCEPTANCE_SUITES=lane): the page fixture matrix, the"
  echo "      walk anchors and the page's node tests read no work dir; the lane's own page stage"
  echo "      walks its served page, and the standalone pack acceptance runs these"
else
  step "11. the page the showcase slot serves: the fixture matrix, the walk anchors, the page's tests"
  python3 "$SHOW/smoke_page.py" --fixture all --strict --work "$WORK/page-fixtures" \
      --json "$WORK/page-fixtures.json" | tail -n 1
  python3 -m pytest -q -p no:cacheprovider "$SKILLS/vega-show/scripts/tests/test_walk_anchors.py" \
    | tail -n 1
  test -f "$PAGE/tests/coverage.test.js" && test -f "$PAGE/tests/probe.test.js"
  ( cd "$PAGE" && node --test tests/*.test.js 2>&1 | grep -E '(tests|pass|fail) [0-9]+$' )
  echo "OK: node --test — every page test passes, coverage.test.js and probe.test.js among them"
fi

step "acceptance passed"
printf 'work dir  %s\nbundle    %s\ncard      %s\nlayer     %s\nnotebook  %s\nfull      %s\n' \
    "$WORK" "$WORK/$BUNDLE" "$WORK/$BUNDLE/model-card.html" "$WORK/$OUT" \
    "$WORK/$BUNDLE/predict.ipynb" "$WORK/$FULL"
