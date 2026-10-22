#!/usr/bin/env python3
"""
run_negatives.py — every /vega-pack gate must refuse what it says it refuses.

    python3 run_negatives.py [<work-dir>] [-v] [--json <path>] [--matrix <dir>]

`acceptance.sh` proves the bundle builds and reproduces the evaluation. That is the easy half.
This stage's whole claim is *what ships is what was evaluated*, and a claim is only worth what
its refusals are worth, so this script seeds one defect at a time against a bundle that already
packed and asserts that the right script refuses it with a message that names the fix.

Three of these matter more than the rest:

  * **a bundle built from a file the chain did not record does not pack** — a different
    checkpoint, a different spec, an edited recipe;
  * **a round trip that did not pass does not get a layer** — publishing the claim without the
    proof is the one thing this stage cannot do;
  * **a digit in the hand-written prose is refused** — every other number on the card and in
    the layer traces to a record, and the one that does not is the one a reader would most
    want to check.

`<work-dir>` is the orders acceptance's (L01's). The v2.3 gates (D2 N-S1…N-S9, D3 NI1…NI5;
BUILD-PLAN T10, R-H6) are seeded on the lane whose shape they are about, read from the matrix
dir (`--matrix`, or the one `<work-dir>` sits in as `<matrix>/lanes/L01/work`):

| id    | lane | seed                                                        | expect                      |
|-------|------|-------------------------------------------------------------|-----------------------------|
| N-S1  | L06  | the stored layout without `row_score` (anomaly-reconstruction) | exit 3; per-row score     |
| N-S2  | L10  | one stored window dropped: the count is off by one          | exit 3; windows             |
| N-S3  | L23  | a transductive campaign packed                              | exit 2; inductive graphs only; nothing under --bundle-dir |
| N-S4a | L05  | a monitor plan with `class_recall` on a regression          | exit 2; not admitted        |
| N-S4b | L01  | a monitor plan with `residual` on a binary                  | exit 2; not admitted        |
| N-S5a | L01  | `--re-emit-serving` after the campaign's weights changed     | exit 3; weights.pt; bundle unchanged |
| N-S5b…d | L01 | `--reason` alone; no bundle to re-emit; other examples     | exit 2 / 2 / 3              |
| N-S6  | L01  | a serving template with an unfilled `{{X}}`                  | exit 2; unfilled placeholder |
| N-S7a…c | L01 | `--story-plan`, `--report-aux`, `--report-pdf`             | exit 2; moved to /vega-show |
| N-S8  | L01  | a stored binary `prediction` flipped                        | exit 3; a decided field disagrees with the evaluation |
| N-S9  | L19  | `graph/reference-nodes.parquet` altered after packing       | `predict.py --check` exit 1 naming it |
| NI1   | L01  | a reader removed from the bundle's `inspection/registry.py` | exit 3; no inspection reader for block type |
| NI2   | L01  | a probe that writes into the captured output (`mul_`)       | exit 3; does not reproduce the served answer |
| NI3   | L01  | an inspection file without its map entry                    | conformance C3; has neither an adapter, a support nor an entrypoint |
| NI4   | L01  | a `NetworkView`-like second public class in the Predictor's module | conformance C14; a public class that is not the Predictor adapter class |
| NI5   | L01  | `--serve --unattended`, the blueprint skill absent          | api_declared: false; WARN … not declared |

L01's cases run on `<work-dir>` alone, so the package's own acceptance runs them. A lane the
matrix does not hold (or no matrix at all) leaves its cases **not run**: said as a WARN and in
`--json`'s `not_run`, never counted as passed; the coverage report requires every gate id to be
the id of a passing case somewhere in the release run (D6 §12.8). The cases of gates older than
v2.3 carry no id: `vega/scripts/tests/negatives/grandfathered.json` lists them by gate.

Nothing here mutates the real bundle or campaign: every seeded case works on a copy.

Exit 0 when every seeded defect was caught by the gate that should catch it; `--json` writes
`vega/negatives-result@1`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

S = Path(__file__).resolve().parent.parent
SKILL = S.parent
SKILLS = SKILL.parent
FORGE = SKILLS / "vega-forge" / "scripts"
TEMPLATES = FORGE / "backends" / "pytorch" / "templates" / "serving"
CONFORMANCE = FORGE / "conformance.py"
sys.path.insert(0, str(SKILLS / "vega" / "scripts" / "tests" / "negatives"))

import negkit  # noqa: E402

_ap = argparse.ArgumentParser(description="seed one defect at a time; every pack gate must "
                                          "refuse it")
_ap.add_argument("work", nargs="?", default=None, help="the orders acceptance work dir (L01's)")
_ap.add_argument("-v", dest="verbose", action="store_true", help="print every case's output")
_ap.add_argument("--json", default=None, metavar="PATH",
                 help="also write the cases as vega/negatives-result@1 to PATH")
_ap.add_argument("--matrix", default=None, metavar="DIR",
                 help="the matrix dir holding the lanes the v2.3 gates are seeded on "
                      "(default: the one <work-dir> sits in)")
ARGS = _ap.parse_args()

WORK = Path(ARGS.work).resolve() if ARGS.work else Path.cwd()
MATRIX = Path(ARGS.matrix).resolve() if ARGS.matrix else negkit.matrix_of(WORK)
VERBOSE = ARGS.verbose
CAMP = WORK / "vega-runs" / "2026-09-05T11-03Z"
SHAPED = WORK / "shaped"
PIPELINE = WORK / "blueprint-runs" / "2026-09-05T10-12Z" / "pipeline.json"
PLAN = WORK / "monitor-plan.json"            # the acceptance's plan (acceptance.sh step 1)
MODEL = WORK / "orders.domain.analysis.shaped.vega.train.eval.html"
AUTHORING = Path(__file__).resolve().parent / "bundle-layer.authoring.json"
STANCE = "the calibrated score is what the downstream dispatcher needs"

CASES = negkit.Cases(__file__, WORK, VERBOSE)


def run(cmd, cwd=None):
    proc = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True,
                          cwd=str(cwd or WORK), timeout=1200)
    return proc.returncode, (proc.stdout + proc.stderr)


def bundled():
    """The bundle the acceptance built, by whatever slug and run it carries."""
    root = WORK / "bundle"
    return next((d for d in sorted(root.glob("*")) if (d / "MANIFEST.json").is_file()), None)


def pack_cmd(campaign, shaped, bundle_dir, *, pipeline=None, model=None, **extra):
    """pack.py as the acceptance packs: with the acceptance's monitor plan unless a case names
    its own (`monitor_plan=None` packs without one) — the blueprint's Monitor port says
    filled_by /vega-pack, and pack refuses to leave it unfilled."""
    if "monitor_plan" not in extra:
        extra["monitor_plan"] = PLAN
    cmd = [S / "pack.py", "--campaign", campaign, "--shaped", shaped,
           "--bundle-dir", bundle_dir, "--pipeline", pipeline or PIPELINE, "--model",
           model or MODEL]
    for flag, value in extra.items():
        if value is None:
            continue
        cmd += [f"--{flag.replace('_', '-')}"] + ([] if value is True else [str(value)])
    return cmd


def campaign_copy(td, *, mutate_experiments=None, drop_seal=False, break_checkpoint=False,
                  break_spec=False, source=None):
    source = Path(source or CAMP)
    dst = Path(td) / "vega-runs" / source.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dst)
    if drop_seal:
        (dst / "test_seal.json").unlink(missing_ok=True)
    run_id = json.loads((dst / "test_seal.json").read_text(encoding="utf-8"))["run"] \
        if (dst / "test_seal.json").is_file() else "R0"
    if break_checkpoint:
        ck = dst / "runs" / run_id / "checkpoints" / "best.pt"
        ck.write_bytes(ck.read_bytes() + b"\x00")
    if break_spec:
        spec = dst / "runs" / run_id / "netspec.resolved.json"
        doc = json.loads(spec.read_text(encoding="utf-8"))
        doc["seed"] = int(doc.get("seed", 7)) + 1
        spec.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    if mutate_experiments:
        path = dst / "experiments.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        mutate_experiments(doc)
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return dst


def bundle_copy(td, *, mutate_manifest=None, drop=(), source=None):
    src = Path(source or bundled())
    dst = Path(td) / src.name
    shutil.copytree(src, dst)
    for name in drop:
        target = dst / name
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    if mutate_manifest:
        path = dst / "MANIFEST.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        mutate_manifest(doc)
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return dst


def authoring_copy(bundle, **readings):
    doc = json.loads(AUTHORING.read_text(encoding="utf-8"))
    doc["from_bundle"] = str(bundle)
    for key, text in readings.items():
        if text is None:
            doc["readings"].pop(key, None)
        else:
            doc["readings"][key] = text
    path = Path(bundle) / "bundle-layer.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def no_bundle_written(name, where) -> None:
    CASES.check(name, not Path(where).exists(), f"{where} was written")


def sealed_run(campaign) -> str:
    return json.loads((Path(campaign) / "test_seal.json").read_text(encoding="utf-8"))["run"]


def digest(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root) -> dict:
    """Every file under `root` by its digest (a symlink by its target)."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): (str(p.readlink()) if p.is_symlink() else digest(p))
            for p in sorted(root.rglob("*")) if p.is_symlink() or p.is_file()}


def served(bundle) -> bool:
    """Whether the bundle was packed with its service (`serve/` shipped)."""
    return (Path(bundle) / "serve" / "app.py").is_file()


# --------------------------------------------------------------------------- pack.py

def pack_negatives():
    CASES.section("pack.py — what ships is what was evaluated", gate="vega-pack/pack.py")

    with tempfile.TemporaryDirectory() as td:
        camp = campaign_copy(td)
        rc, out = run(pack_cmd(camp, SHAPED, Path(td) / "b"))
        CASES.expect("an unearned verdict without --ship-anyway is refused", rc, out,
               contains=("not earned", "--ship-anyway"))
        CASES.expect("...and no bundle was written for it",
               0 if not (Path(td) / "b").exists() else 1, "", code=0)

    with tempfile.TemporaryDirectory() as td:
        camp = campaign_copy(td, break_checkpoint=True)
        rc, out = run(pack_cmd(camp, SHAPED, Path(td) / "b", ship_anyway=STANCE))
        CASES.expect("a checkpoint that is not the one the chain recorded is refused", rc, out,
               contains=("the checkpoint", "numbers it did not earn"))

    with tempfile.TemporaryDirectory() as td:
        camp = campaign_copy(td, break_spec=True)
        rc, out = run(pack_cmd(camp, SHAPED, Path(td) / "b", ship_anyway=STANCE))
        CASES.expect("a resolved spec that is not the one the chain recorded is refused", rc, out,
               contains=("netspec", "not the file the evaluation measured"))

    with tempfile.TemporaryDirectory() as td:
        camp = campaign_copy(td, drop_seal=True)
        rc, out = run(pack_cmd(camp, SHAPED, Path(td) / "b", ship_anyway=STANCE))
        CASES.expect("packing a campaign whose test part was never opened is refused", rc, out,
               contains=("test_seal.json", "nothing here has been measured"))

    with tempfile.TemporaryDirectory() as td:
        camp = campaign_copy(td)
        shutil.rmtree(camp / "eval")
        rc, out = run(pack_cmd(camp, SHAPED, Path(td) / "b", ship_anyway=STANCE))
        CASES.expect("packing without the evaluation's records is refused", rc, out,
               contains=("metrics.json",))

    with tempfile.TemporaryDirectory() as td:
        camp = campaign_copy(td)
        rc, out = run(pack_cmd(camp, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                               dataset_shaper_dir=Path(td) / "no-shaper"))
        CASES.expect("packing without the shaper's transform-only emitter is refused", rc, out,
               contains=("dataset-shaper", "never re-implemented here"))

    with tempfile.TemporaryDirectory() as td:
        # a fitted parameter changed after the shaper wrote it: the twin check is what
        # notices, and it must notice before the round trip so the message names the transform
        camp = campaign_copy(td)
        shaped = Path(td) / "shaped"
        shutil.copytree(SHAPED, shaped)
        fitted = shaped / "fitted.json"
        doc = json.loads(fitted.read_text(encoding="utf-8"))
        # Nudge a parameter the transform actually applies. `n_categories` is also a number
        # in this file and moving it changes nothing, which would make the case pass for the
        # wrong reason — a seeded defect that is not a defect proves the gate never fired.
        applied = ("center", "scale", "mean", "std", "lower", "upper", "fill", "lambda",
                   "median")
        moved = None
        for sid, step in doc.items():
            if sid.startswith("__") or not isinstance(step, dict):
                continue
            for name, column in (step.get("fitted") or {}).items():
                if not isinstance(column, dict):
                    continue
                for key in applied:
                    value = column.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        column[key] = float(value) + 1.0
                        moved = f"{sid}.{name}.{key}"
                        break
                if moved:
                    break
            if moved:
                break
        assert moved, ("no fitted parameter the transform applies was found to perturb; this "
                       "case would otherwise pass without testing anything")
        fitted.write_text(json.dumps(doc), encoding="utf-8")
        rc, out = run(pack_cmd(camp, shaped, Path(td) / "b", ship_anyway=STANCE))
        CASES.expect("a fitted parameter that no longer matches what was shaped is refused",
               rc, out, code=3,
               contains=("twin check", "would mean anything"))


def options_negatives():
    CASES.section("pack.py options — what moved away says where, and a re-emit proves what it "
                  "kept", gate="vega-pack/pack.py")

    # N-S7: the story and the written report are /vega-show's, built beside the bundle; a pack
    # asked for them refuses before anything is written and names where they went
    for letter, flag in zip("abc", ("--story-plan", "--report-aux", "--report-pdf")):
        with tempfile.TemporaryDirectory() as td:
            cmd = pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE)
            rc, out = run(cmd + [flag, Path(td) / "plan.json"])
            CASES.expect(f"N-S7{letter}  {flag} is refused: it moved to /vega-show", rc, out,
                         contains=(flag, "moved to /vega-show", "carries no story"))
            no_bundle_written(f"N-S7{letter}  ...and no bundle was written for {flag}",
                              Path(td) / "b")

    with tempfile.TemporaryDirectory() as td:
        rc, out = run(pack_cmd(CAMP, SHAPED, Path(td) / "b", unattended=True))
        CASES.expect("an unearned verdict under --unattended stops with WARN and no bundle", rc,
                     out, contains=("WARN: --unattended", "not earned", "--ship-anyway"),
                     absent=("ERROR:",))
        CASES.expect("...and no bundle was written under --unattended",
                     0 if not (Path(td) / "b").exists() else 1, "", code=0)

    with tempfile.TemporaryDirectory() as td:
        # the blueprint's Monitor port says filled_by /vega-pack: the plan is this stage's to
        # write, and a pack without one would ship the forge's stub in the port it was given
        rc, out = run(pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                               monitor_plan=None))
        CASES.expect("N-M1  a Monitor the blueprint says this stage fills is not left unfilled",
                     rc, out, contains=("filled_by /vega-pack", "no --monitor-plan was given",
                                        "refused before the bundle was written"))
        no_bundle_written("N-M1  ...and no bundle was written without the monitor plan",
                          Path(td) / "b")

    with tempfile.TemporaryDirectory() as td:
        rc, out = run(pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                               reason="nothing to re-emit"))
        CASES.expect("N-S5b  --reason without --re-emit-serving is refused", rc, out,
                     contains=("--reason", "--re-emit-serving"))

    with tempfile.TemporaryDirectory() as td:
        rc, out = run(pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                               re_emit_serving=True))
        CASES.expect("N-S5c  --re-emit-serving with no bundle in place is refused", rc, out,
                     contains=("no bundle to re-emit", "no MANIFEST.json", "pack it first"))

    with tempfile.TemporaryDirectory() as td:
        # a re-emit may change only what the serving templates produce: asked to ship other
        # example rows, it must refuse and leave the bundle in place byte for byte
        src = bundled()
        target = Path(td) / "b" / src.name
        shutil.copytree(src, target)
        before = snapshot(target)
        rc, out = run(pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                               re_emit_serving=True, reason="a seeded input change",
                               examples=7, serve=served(src) or None))
        CASES.expect("N-S5d  a re-emit that would change the shipped examples is refused", rc,
                     out, code=3, contains=("re-emit would change", "examples/rows.csv",
                                            "re-pack deliberately", "left as it was"))
        untouched = snapshot(target) == before and not (Path(td) / "b" / ".reemit").exists()
        CASES.check("N-S5d  ...and the bundle in place is untouched", untouched,
                    "the bundle in place changed, or the staging directory was left")


# --------------------------------------------------------------------------- v2.3 on L01

OVERRIDE = r"""
import sys
from pathlib import Path
sys.path.insert(0, {scripts!r})
seed, value = sys.argv[1], Path(sys.argv[2])
if seed == "templates":            # the backend's serving templates, from a seeded copy
    import pack_context
    pack_context.PackContext.templates = property(lambda self: value)
elif seed == "blueprint":          # the installed blueprint skill, as this path (absent)
    import api_declaration
    api_declaration.BLUEPRINT_SKILL = value
import pack
sys.exit(pack.main(sys.argv[3:]))
"""


def run_overridden(seed, value, cmd, cwd=None):
    """pack.py (`cmd` = pack_cmd(...)) with one of its inputs from the environment replaced:
    `templates` (a seeded copy of the serving templates) or `blueprint` (the skill's SKILL.md
    path, absent). The pack is the real one; only where it looks is moved."""
    proc = subprocess.run([sys.executable, "-c", OVERRIDE.format(scripts=str(S)), seed,
                           str(value), *map(str, cmd[1:])], capture_output=True, text=True,
                          cwd=str(cwd or WORK), timeout=1200)
    return proc.returncode, proc.stdout + proc.stderr


def rewrite_layout(path, change) -> None:
    """The stored predictions (vega/predictions-layout@1) with `change(frame)` applied, their
    layout metadata kept: the file stays one the evaluation's reader opens."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    kept = {k: v for k, v in (table.schema.metadata or {}).items() if k.startswith(b"vega.")}
    frame = table.to_pandas()
    frame = change(frame)
    out = pa.Table.from_pandas(frame, preserve_index=False)
    meta = dict(out.schema.metadata or {})
    meta.update(kept)
    pq.write_table(out.replace_schema_metadata(meta), path)


def resave_checkpoint(camp) -> None:
    """The campaign's selected checkpoint saved again — the same tensors, other bytes — and the
    ledger pointing at it, as a re-trained run under the same id would leave it."""
    import torch
    run_id = sealed_run(camp)
    run_dir = Path(camp) / "runs" / run_id
    best = run_dir / "checkpoints" / "best.pt"
    blob = torch.load(best, map_location="cpu", weights_only=True)
    blob["saved_again_by"] = "run_negatives N-S5a"
    torch.save(blob, best)
    new = digest(best)
    ledger = json.loads((Path(camp) / "experiments.json").read_text(encoding="utf-8"))
    for row in ledger["runs"]:
        if row.get("id") == run_id:
            row["checkpoint_digest"] = new
    (Path(camp) / "experiments.json").write_text(json.dumps(ledger, indent=2) + "\n",
                                                 encoding="utf-8")
    record = run_dir / "run.json"
    if record.is_file():
        doc = json.loads(record.read_text(encoding="utf-8"))
        doc["checkpoint_digest"] = new
        record.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def v23_orders_negatives():
    CASES.section("pack.py v2.3 — the re-emit, the templates, the stored decisions, the monitor "
                  "(L01)")
    bundle = bundled()
    serve = served(bundle) or None

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "b" / bundle.name
        shutil.copytree(bundle, target)
        camp = campaign_copy(td)
        resave_checkpoint(camp)
        before = snapshot(target)
        rc, out = run(pack_cmd(camp, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                               re_emit_serving=True, reason="the weights changed", serve=serve))
        CASES.expect("N-S5a  a re-emit after the campaign's weights changed is refused", rc, out,
                     code=3, contains=("re-emit would change", "weights.pt",
                                       "re-pack deliberately", "left as it was"))
        untouched = snapshot(target) == before and not (Path(td) / "b" / ".reemit").exists()
        CASES.check("N-S5a  ...and the bundle in place is byte for byte as it was", untouched,
                    "the bundle in place changed, or the staging directory was left")

    with tempfile.TemporaryDirectory() as td:
        seeded = Path(td) / "serving"
        shutil.copytree(TEMPLATES, seeded, ignore=shutil.ignore_patterns("__pycache__"))
        predictor = seeded / "package" / "predict.py"
        predictor.write_text(predictor.read_text(encoding="utf-8")
                             + "\n# {{SEEDED_NEVER_FILLED}}\n", encoding="utf-8")
        rc, out = run_overridden("templates", seeded,
                                 pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                                          serve=serve))
        CASES.expect("N-S6  a serving template with an unfilled placeholder is refused", rc, out,
                     contains=("unfilled placeholder", "SEEDED_NEVER_FILLED"))

    with tempfile.TemporaryDirectory() as td:
        camp = campaign_copy(td)
        stored = camp / "eval" / sealed_run(camp) / "predictions.test.parquet"

        def flip_first(frame):
            assert "prediction" in frame.columns, "the orders layout stores no prediction"
            frame.loc[0, "prediction"] = 1 - int(frame.loc[0, "prediction"])
            return frame
        rewrite_layout(stored, flip_first)
        rc, out = run(pack_cmd(camp, SHAPED, Path(td) / "b", ship_anyway=STANCE))
        CASES.expect("N-S8  a stored binary prediction the threshold does not make stops the "
                     "pack", rc, out, code=3,
                     contains=("a decided field disagrees with the evaluation", "does not ship"))

    with tempfile.TemporaryDirectory() as td:
        plan = Path(td) / "monitor-plan.json"
        plan.write_text(json.dumps({"checks": [{"id": "seeded", "kind": "residual"}]}),
                        encoding="utf-8")
        rc, out = run(pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                               monitor_plan=plan))
        CASES.expect("N-S4b  a residual check on a binary reading is refused", rc, out,
                     contains=("residual", "not admitted", "binary"))


# --------------------------------------------------------------------------- the card

def card_negatives():
    CASES.section("model_card.py — every number traces to a record, and the prose has none",
                  gate="vega-pack/model_card.py")
    bundle = bundled()
    with tempfile.TemporaryDirectory() as td:
        data = authoring_copy(bundle_copy(td),
                              intended_use="Use it to rank orders. It was fitted on about "
                                           "80% of the rows and must not be used elsewhere, "
                                           "and this sentence is long enough to pass the "
                                           "length check that guards short readings.")
        rc, out = run([S / "model_card.py", "--bundle", bundle, "--model", MODEL,
                       "--intended-use", data, "--out", Path(td) / "card.html"])
        CASES.expect("a digit in the hand-written section is refused", rc, out,
               contains=("80", "traces"))
        CASES.expect("...and no card was written",
               0 if not (Path(td) / "card.html").exists() else 1, "", code=0)

    with tempfile.TemporaryDirectory() as td:
        stripped = Path(td) / "no-eval.html"
        text = MODEL.read_text(encoding="utf-8")
        import re as _re
        stripped.write_text(_re.sub(
            r'(<script id="layer-evaluation-data"[^>]*>)[\s\S]*?(</script>)', r"\1{}\2", text),
            encoding="utf-8")
        rc, out = run([S / "model_card.py", "--bundle", bundle, "--model", stripped,
                       "--intended-use", bundle / "bundle-layer.json",
                       "--out", Path(td) / "card.html"])
        CASES.expect("a card with an empty required section is refused", rc, out,
               contains=("evaluation",))


# --------------------------------------------------------------------------- the notebook

def notebook_negatives():
    CASES.section("nb_emit.py --stage pack — the notebook documents a bundle that packed",
                  gate="vega-forge/nb_emit.py")
    with tempfile.TemporaryDirectory() as td:
        def failed(doc):
            doc["roundtrip"]["status"] = "fail"
        broken = bundle_copy(td, mutate_manifest=failed)
        rc, out = run([FORGE / "nb_emit.py", "--stage", "pack", "--from", broken,
                       "--out", Path(td) / "n.ipynb"])
        CASES.expect("a notebook for a bundle whose round trip failed is refused", rc, out,
               contains=("round trip did not pass", "record the failure twice"))

    with tempfile.TemporaryDirectory() as td:
        bare = bundle_copy(td, drop=("examples",))
        rc, out = run([FORGE / "nb_emit.py", "--stage", "pack", "--from", bare,
                       "--out", Path(td) / "n.ipynb"])
        CASES.expect("a notebook for a bundle with no examples is refused", rc, out,
               contains=("examples/rows.csv", "nothing to check"))


# --------------------------------------------------------------------------- the layer

FIXTURE_AUDIT = FORGE / "tests" / "fixtures" / "dot-claude" / "architect-runs" / "2026-09-05T11-20Z"


def audits(td, tool, *mutations):
    """One audit run dir per mutation of the forge's recorded fixture audit, the first audit
    first; `None` is an audit that found nothing."""
    root, runs = Path(td) / f"{tool}-runs", []
    for mutate in mutations:
        run_dir = root / f"audit-{len(runs) + 1}"
        run_dir.mkdir(parents=True)
        found = json.loads((FIXTURE_AUDIT / "findings.json").read_text(encoding="utf-8"))
        found["run"] = run_dir.name
        if mutate is None:
            found["findings"] = []
        else:
            mutate(found["findings"])
        (run_dir / "findings.json").write_text(json.dumps(found), encoding="utf-8")
        runs.append(run_dir)
    return runs


def _structural(found):
    found[0]["depth"] = "structural"


def _elsewhere(found):
    found[0].update(file="serving/monitor.py", anchors=["EJ-62"], axis="Model")


def _bundle_layer(path) -> dict:
    text = Path(path).read_text(encoding="utf-8") if Path(path).is_file() else ""
    start = text.find('id="layer-bundle-data"')
    if start < 0:
        return {}
    body = text[text.index(">", start) + 1:]
    return json.loads(body[:body.index("</script>")])


def judgement_negatives(bundle, nb):
    """The judgement half of the bundle's gate (code-contract §5): a structural or divergence
    finding is confirmed when two of three independent fresh audits report it, and a confirmed
    finding is advisory — unattended a WARN and an open item (the layer is written), attended
    the user's recorded FORK answer; a first audit's finding is never published alone; an
    absent tool is declared; the mechanical half and a duplicated run id refuse."""
    def apply(td, architect=(), improve=(), mutate=None, markers=None, forks=None,
              unattended=False):
        copy = bundle_copy(td, mutate_manifest=mutate)
        data = authoring_copy(copy)
        if markers is not None or forks is not None:
            doc = json.loads(data.read_text(encoding="utf-8"))
            if markers is not None:
                doc["markers"] = markers
            if forks is not None:
                doc["forks"] = forks
            data.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        cmd = [S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
               "--out", Path(td) / "o.html", *(["--unattended"] if unattended else [])]
        for flag, runs in (("--architect-run", architect), ("--improve-run", improve)):
            for r in runs:
                cmd += [flag, r]
        return run(cmd)

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", _structural))
        CASES.expect("N-J1  a single audit's structural finding on the bundle is not published "
                     "until two fresh audits judge it", rc, out,
                     contains=("first audit reported 1 structural", "fresh context"))

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", _structural, lambda f: None, None),
                        improve=audits(td, "improve", None))
        CASES.expect("N-J2  attended, a finding two of three audits confirm is not published "
                     "without the user's recorded FORK answer", rc, out,
                     contains=("confirmed by audits 1, 2 of 3", "no answer is recorded", "FORK:",
                               "fix now", "accept and record"))

    def advisory_written(td, rc, out, tool, mode):
        layer = _bundle_layer(Path(td) / "o.html") if rc == 0 else {}
        rec = (layer.get("conformance") or {}).get(tool) or {}
        advisory = rec.get("advisory") or {}
        items = advisory.get("open_items") or []
        rows = [f for f in layer.get("code_findings") or [] if f.get("open_item")]
        warns = [m for m in layer.get("markers") or [] if str(m).startswith(f"WARN: {tool} 1:")]
        ok = (rc == 0 and rec.get("structural") == 1 and advisory.get("mode") == mode
              and len(items) == 1 and "--re-emit-serving" in items[0]["text"]
              and len(rows) == 1 and len(warns) == 1 and f"WARN: {tool} 1:" in out)
        return ok, f"exit {rc}; conformance.{tool} {json.dumps(rec)[:300]}"

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", None),
                        improve=audits(td, "improve", _structural,
                                       lambda f: f.__setitem__(slice(None), f[:1]), None),
                        unattended=True)
        CASES.check("N-J3  unattended, a structural improve finding two of three audits confirm "
                    "is a WARN and an open item, and the layer is written",
                    *advisory_written(td, rc, out, "improve", "unattended"))

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", _structural, lambda f: None, None),
                        improve=audits(td, "improve", None), unattended=True)
        CASES.check("N-J8  unattended, a confirmed architect finding never stops the chain: WARN, "
                    "open item, layer written", *advisory_written(td, rc, out, "architect",
                                                                  "unattended"))

    def gate_fork(answer):
        return [{"gate": "architect", "findings": ["1:A-1"], "asked": "FORK: architect 1:A-1",
                 "answer": answer}]

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", _structural, lambda f: None, None),
                        improve=audits(td, "improve", None),
                        forks=gate_fork("accept and record — the user ships it as it is"))
        CASES.check("N-J9  attended, the user's recorded `accept and record` publishes the "
                    "confirmed finding as a WARN and an open item",
                    *advisory_written(td, rc, out, "architect", "attended"))

    for answer, said in (("fix now", "answered fix now"), ("stop", "answered stop")):
        with tempfile.TemporaryDirectory() as td:
            rc, out = apply(td, architect=audits(td, "architect", _structural, lambda f: None,
                                                 None),
                            improve=audits(td, "improve", None), forks=gate_fork(answer))
            CASES.expect(f"N-J10 attended, the answer `{answer}` publishes nothing", rc, out,
                         contains=(said,))

    with tempfile.TemporaryDirectory() as td:
        first = audits(td, "architect", _structural)[0]
        twin = Path(td) / "copy" / first.name
        twin.mkdir(parents=True)
        found = json.loads((first / "findings.json").read_text(encoding="utf-8"))
        found["findings"][0]["title"] = "another audit, started in the same second"
        (twin / "findings.json").write_text(json.dumps(found), encoding="utf-8")
        third = Path(td) / "third" / "audit-3"
        third.mkdir(parents=True)
        found["findings"] = []
        (third / "findings.json").write_text(json.dumps(found), encoding="utf-8")
        rc, out = apply(td, architect=[first, twin, third], improve=audits(td, "improve", None),
                        unattended=True)
        CASES.expect("N-J11 two audits that share a run id are refused, unattended too (F31)",
                     rc, out, contains=("share the run id", "one after another"))

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", _structural, _elsewhere, None),
                        improve=audits(td, "improve", None))
        layer = _bundle_layer(Path(td) / "o.html") if rc == 0 else {}
        arch = (layer.get("conformance") or {}).get("architect") or {}
        recorded = (arch.get("structural") == 0 and arch.get("unconfirmed", {}).get("structural") == 1
                    and len(arch.get("audits") or []) == 3
                    and [r.get("reported_by") for r in arch.get("matching") or []] == [[1]]
                    and any((f.get("quorum") or {}).get("confirmed") is False
                            for f in layer.get("code_findings") or []))
        CASES.check("N-J4  an unconfirmed structural finding is recorded with its three audits and "
                    "the layer is written", rc == 0 and recorded,
                    f"exit {rc}; conformance.architect {json.dumps(arch)[:300]}")

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", _structural, lambda f: None, None),
                        improve=audits(td, "improve", None), unattended=True,
                        mutate=lambda d: d["blueprint"]["conformance"].update(mechanical="fail"))
        CASES.expect("N-J5  the mechanical half refuses whatever the judgement half found, "
                     "unattended too", rc, out,
                     contains=("mechanical conformance is 'fail'", "beside the blueprint"))

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", None),
                        markers=["WARN: the judgement half ran for one tool only"])
        CASES.expect("N-J6  an unrecorded improve absence is refused", rc, out,
                     contains=("no improve run is recorded", "never silently skipped"))

    with tempfile.TemporaryDirectory() as td:
        rc, out = apply(td, architect=audits(td, "architect", _structural, _elsewhere))
        CASES.expect("N-J7  two audits are not a judgement", rc, out,
                     contains=("never another number",))


def layer_negatives():
    CASES.section("apply_bundle_layer.py — the layer publishes the proof, not only the claim",
                  gate="vega-pack/apply_bundle_layer.py")
    bundle = bundled()
    nb = bundle / "predict.ipynb"

    with tempfile.TemporaryDirectory() as td:
        def failed(doc):
            doc["roundtrip"]["status"] = "fail"
            doc["roundtrip"]["first_disagreement"] = {"row": 3, "abs_diff": 0.4}
        copy = bundle_copy(td, mutate_manifest=failed)
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("a layer for a bundle whose round trip failed is refused", rc, out,
               contains=("not 'pass'", "not a knob"))

    with tempfile.TemporaryDirectory() as td:
        def twin(doc):
            doc["roundtrip"]["shaper_transform_check"] = "fail"
        copy = bundle_copy(td, mutate_manifest=twin)
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("a layer whose twin check failed is refused", rc, out,
               contains=("twin check", "means nothing"))

    with tempfile.TemporaryDirectory() as td:
        def ungated(doc):
            doc["blueprint"]["conformance"]["mechanical"] = "not run"
        copy = bundle_copy(td, mutate_manifest=ungated)
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("a layer whose conformance gate did not run is refused", rc, out,
               contains=("mechanical conformance", "beside the blueprint"))

    with tempfile.TemporaryDirectory() as td:
        def structural(doc):
            doc["blueprint"]["conformance"]["architect"] = {"structural": 2, "divergence": 0,
                                                            "recorded": 1}
        copy = bundle_copy(td, mutate_manifest=structural)
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        # a count typed into the manifest is a malformed record (pack.py never writes one; the
        # judgement is spliced from the audits), refused under its grandfathered name
        CASES.expect("a layer with a structural architect finding is refused", rc, out,
               contains=("structural finding", "never typed into the manifest",
                         "a malformed record"))

    judgement_negatives(bundle, nb)

    with tempfile.TemporaryDirectory() as td:
        # a manifest that claims the full shape while the bundle holds only predict.py sends
        # a reader looking for a training loop that is not there
        def claims_full(doc):
            doc["include"] = "full"
        copy = bundle_copy(td, mutate_manifest=claims_full)
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("a layer claiming the full shape without its entry points is refused", rc, out,
               contains=("include 'full'", "the same bundle"))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td)
        doc = json.loads(AUTHORING.read_text(encoding="utf-8"))
        doc["from_bundle"] = str(copy)
        doc["markers"] = []                      # the architect's absence no longer declared
        data = Path(copy) / "bundle-layer.json"
        data.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("an unrecorded architect absence is refused", rc, out,
               contains=("no architect run is recorded", "never silently skipped"))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td, drop=("model-card.html",))
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("a layer with no model card is refused", rc, out,
               contains=("model card", "nowhere to check them"))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td)
        data = authoring_copy(copy, intended_use=(
            "Use this model to score an order for the risk it arrives late. It was measured "
            "on 119 held-out rows and should not be trusted beyond the ranges the schema "
            "lists, which is a long enough sentence to reach the length the gate wants."))
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("a digit in readings.intended_use is refused", rc, out,
               contains=("states a number", "tracing to nobody"))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td)
        data = authoring_copy(copy, abstract=None)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("a missing reading is refused", rc, out, contains=("readings.abstract",))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td)
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data,
                       "--out", Path(td) / "o.html"])
        CASES.expect("a layer with no notebook is refused", rc, out, contains=("notebook",))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td)
        doc = json.loads(AUTHORING.read_text(encoding="utf-8"))
        doc.pop("from_bundle")
        data = Path(copy) / "bundle-layer.json"
        data.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html"])
        CASES.expect("an authoring file with no from_bundle is refused", rc, out,
               contains=("from_bundle", "never retyped"))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td)
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", MODEL])
        CASES.expect("writing the layer over its own input is refused", rc, out,
               contains=("never modified",))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td)
        data = authoring_copy(copy)
        rc, out = run([S / "apply_bundle_layer.py", MODEL, "--data", data, "--notebook", nb,
                       "--out", Path(td) / "o.html",
                       "--domain-forge-dir", Path(td) / "nowhere"])
        CASES.expect("a missing platform writer is refused, and never worked around", rc, out,
               code=3, contains=("apply_layer.py", "never by this script"))


# --------------------------------------------------------------------------- inspection

CHECK = r"""
import json, sys
from pathlib import Path
scripts, bundle = sys.argv[1], Path(sys.argv[2]).resolve()
sys.path.insert(0, scripts)
sys.path.insert(0, str(bundle))
import predict
import inspection_pack
from pack_context import PackExit
tolerance = json.loads((bundle / "MANIFEST.json").read_text())["roundtrip"]["tolerance"]["value"]
try:
    opened = inspection_pack.inspection_check(bundle, predict.load(bundle), tolerance)
except PackExit as exc:
    print(json.dumps({"code": exc.code, "lines": list(exc.lines)})); sys.exit(0)
print(json.dumps({"code": 0, "digests": opened.digests}))
"""


def inspection_checked(bundle):
    """The pack's inspection self-test over `bundle` (as pack runs it before it ships):
    (exit code, what it said)."""
    proc = subprocess.run([sys.executable, "-c", CHECK, str(S), str(bundle)],
                          capture_output=True, text=True, timeout=900, cwd=str(bundle))
    try:
        doc = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return proc.returncode or 1, proc.stdout + proc.stderr
    return doc["code"], "\n".join(doc.get("lines") or [json.dumps(doc)])


def inspection_dir(bundle) -> Path:
    manifest = json.loads((Path(bundle) / "MANIFEST.json").read_text(encoding="utf-8"))
    return Path(bundle) / manifest["package"] / \
        Path(manifest["serve"]["inspection"]["files"][0]).parent


def conformance(bundle, *extra):
    manifest = json.loads((Path(bundle) / "MANIFEST.json").read_text(encoding="utf-8"))
    pkg = Path(bundle) / manifest["package"]
    proc = subprocess.run([sys.executable, str(CONFORMANCE), "--package", str(pkg),
                           "--pipeline", str(PIPELINE), "--map", str(pkg / "blueprint-map.json"),
                           "--bundle", *extra], capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def inspection_negatives():
    CASES.section("inspection — the network read is the network that answers (D3 NI1–NI5, L01)")
    bundle = bundled()
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    if not ((manifest.get("serve") or {}).get("inspection") or {}).get("files"):
        for cid in ("NI1", "NI2", "NI3", "NI4"):
            CASES.not_run(f"{cid}  the inspection gates on the orders bundle",
                          "the bundle ships no inspection/ (packed with --no-inspect)")
    else:
        _inspection_cases(bundle)
    _declaration_case()


def _inspection_cases(bundle):
    rc, out = inspection_checked(bundle)
    CASES.expect("NI1  control: the self-test passes on the bundle as shipped", rc, out, code=0)

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td, source=bundle)
        registry = inspection_dir(copy) / "registry.py"
        text = registry.read_text(encoding="utf-8")
        seeded = text.replace("for r in (*CORE, *GROUP, *SEQ, *GRAPH)})",
                              "for r in (*CORE, *GROUP, *SEQ, *GRAPH) if r.TYPE != 'head'})")
        CASES.check("NI1  the seed finds the registry's READERS line", seeded != text,
                    "registry.py's READERS line moved; update the seed")
        registry.write_text(seeded, encoding="utf-8")
        rc, out = inspection_checked(copy)
        CASES.expect("NI1  a reader removed from the registry stops the self-test", rc, out,
                     code=3, contains=("no inspection reader for block type",))

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td, source=bundle)
        probes = inspection_dir(copy) / "probes.py"
        text = probes.read_text(encoding="utf-8")
        anchor = ("    def recompute(self, cap, node):\n"
                  "        x, z, y, _ = self._parts(cap, node)\n")
        CASES.check("NI2  the seed finds the dense/head probe", text.count(anchor) == 1,
                    "the dense/head probe moved; update the seed")
        probes.write_text(text.replace(anchor, anchor + "        y.data.mul_(2.0)\n"),
                          encoding="utf-8")
        rc, out = inspection_checked(copy)
        CASES.expect("NI2  a probe writing into the captured output stops the self-test", rc,
                     out, code=3, contains=("does not reproduce the served answer",))

    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td, source=bundle)
        rc, out = conformance(copy)
        CASES.expect("NI3  control: the bundle as shipped passes the conformance gate", rc, out,
                     code=0)
        path = copy / manifest["package"] / "blueprint-map.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        gone = manifest["serve"]["inspection"]["files"][-1]
        doc["entries"] = [e for e in doc["entries"] if e.get("file") != gone]
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        rc, out = conformance(copy)
        CASES.expect("NI3  an inspection file without its map entry is refused", rc, out,
                     code=2, contains=(gone, "has neither an adapter, a support nor an entrypoint"))
        rc, out = conformance(copy, "--json")
        try:
            markers = json.JSONDecoder().raw_decode(out[out.index("{"):])[0]["markers"]
        except (ValueError, KeyError):
            markers = None
        rules = sorted({str(m.get("rule")) for m in markers or ()
                        if str(m.get("level")).upper() == "ERROR"})
        CASES.check("NI3  ...by rule C3, and by no other rule", rules == ["C3"],
                    f"the ERROR markers' rules are {rules}" if markers is not None else
                    f"conformance --json printed no markers: {out[-300:]}")

    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td, source=bundle)
        predictor = inspection_dir(copy).parent / "predict.py"
        with predictor.open("a", encoding="utf-8") as fh:
            fh.write("\n\nclass NetworkView:\n    \"\"\"v2.2's second public class.\"\"\"\n")
        rc, out = conformance(copy, "--size-severity", "error")
        CASES.expect("NI4  a NetworkView-like second public class beside the Predictor fails "
                     "C14", rc, out, code=2,
                     contains=("C14", "NetworkView",
                               "a public class that is not the Predictor adapter class"))


def _declaration_case():
    with tempfile.TemporaryDirectory() as td:
        rc, out = run_overridden("blueprint", Path(td) / "no-blueprint" / "SKILL.md",
                                 pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE,
                                          serve=True, unattended=True))
        CASES.expect("NI5  unattended, the blueprint skill absent: the divergence is recorded "
                     "and said", rc, out, code=0,
                     contains=("WARN: blueprint declaration", "not declared",
                               "blueprint skill with --extends is absent",
                               "api_declared: false"))
        found = sorted((Path(td) / "b").glob("*/MANIFEST.json"))
        conf = ((json.loads(found[0].read_text(encoding="utf-8")).get("blueprint") or {})
                .get("conformance") or {}) if found else {}
        record = conf.get("declaration") or {}
        CASES.check("NI5  ...and MANIFEST records api_declared false with the answer divergence",
                    conf.get("api_declared") is False and record.get("answer") == "divergence"
                    and record.get("source") == "vega:default",
                    f"blueprint.conformance is {json.dumps(conf)[:300]}")


# --------------------------------------------------------------------------- the lanes

def lane_pack_cmd(lane, campaign, bundle_dir, **extra):
    """pack.py as the lane program packs the lane (lanekit.stages: --serve --unattended, with the
    lane's monitor plan)."""
    extra.setdefault("monitor_plan", lane.work / "monitor-plan.json")
    return pack_cmd(campaign, lane.shaped, bundle_dir, pipeline=lane.pipeline,
                    model=lane.layered(".vega.train.eval"), ship_anyway=f"seeded negative on "
                    f"{lane.id}", serve=True, unattended=True, **extra)


def need(lane_id, names, *, evaluated=True, bundle=False):
    """The lane from the matrix, or None after saying which cases cannot run and why."""
    lane = negkit.lane(MATRIX, lane_id)
    why = None
    if MATRIX is None:
        why = f"no matrix dir: lane {lane_id} is read from --matrix (or <matrix>/lanes/L01/work)"
    elif lane is None:
        why = f"lane {lane_id} has no fixture in {MATRIX}"
    elif evaluated and not lane.evaluated():
        why = f"lane {lane_id} was not evaluated in {MATRIX}"
    elif bundle and lane.bundle() is None:
        why = f"lane {lane_id} packed no bundle in {MATRIX}"
    if why:
        for name in names:
            CASES.not_run(name, why)
        return None
    return lane


def lane_negatives():
    CASES.section("pack.py v2.3 — each shape's gate on the lane of that shape (matrix: "
                  f"{MATRIX or 'none'})")

    name = "N-S1  L06: a row-score reading whose stored layout lacks row_score is refused"
    lane = need("L06", [name])
    if lane:
        with tempfile.TemporaryDirectory() as td:
            camp = campaign_copy(td, source=lane.campaign)
            stored = camp / "eval" / sealed_run(camp) / "predictions.test.parquet"
            rewrite_layout(stored, lambda f: f.drop(columns=["row_score"]))
            rc, out = run(lane_pack_cmd(lane, camp, Path(td) / "b"), cwd=lane.work)
            CASES.expect(name, rc, out, code=3, contains=("per-row score",
                                                           "anomaly-reconstruction"))

    name = "N-S2  L10: a stored window count off by one is refused"
    lane = need("L10", [name])
    if lane:
        with tempfile.TemporaryDirectory() as td:
            camp = campaign_copy(td, source=lane.campaign)
            stored = camp / "eval" / sealed_run(camp) / "predictions.test.parquet"
            rewrite_layout(stored, lambda f: f.iloc[1:].reset_index(drop=True))
            rc, out = run(lane_pack_cmd(lane, camp, Path(td) / "b"), cwd=lane.work)
            CASES.expect(name, rc, out, code=3, contains=("windows", "unscored.json"))

    names = ["N-S3  L23: a transductive graph campaign is refused",
             "N-S3  ...before anything was written under --bundle-dir: no partial bundle is "
             "left",
             "N-S3  ...on a campaign whose sample is transductive"]
    lane = need("L23", names)
    if lane:
        with tempfile.TemporaryDirectory() as td:
            spec = json.loads((lane.campaign / "netspec.json").read_text(encoding="utf-8"))
            protocol = ((spec.get("data_contract") or {}).get("sample") or {}).get("protocol")
            rc, out = run(lane_pack_cmd(lane, lane.campaign, Path(td) / "b"), cwd=lane.work)
            CASES.expect(names[0], rc, out, contains=("inductive graphs only", "transductive"))
            written = sorted(str(p.relative_to(td)) for p in (Path(td) / "b").rglob("*"))
            CASES.check(names[1], not written, f"{written[:5]} written")
            CASES.check(names[2], protocol == "transductive", f"L23's protocol is {protocol!r}")

    name = "N-S4a  L05: a class_recall check on a regression reading is refused"
    lane = need("L05", [name])
    if lane:
        with tempfile.TemporaryDirectory() as td:
            plan = Path(td) / "monitor-plan.json"
            plan.write_text(json.dumps({"checks": [{"id": "seeded", "kind": "class_recall",
                                                    "classes": ["high"]}]}), encoding="utf-8")
            rc, out = run(lane_pack_cmd(lane, lane.campaign, Path(td) / "b", monitor_plan=plan),
                          cwd=lane.work)
            CASES.expect(name, rc, out, contains=("class_recall", "not admitted", "regression"))

    names = ["N-S9  L19: an altered reference graph fails the bundle's own check",
             "N-S9  control: the unaltered copy passes its check"]
    lane = need("L19", names, bundle=True)
    if lane:
        _reference_case(lane, names)


def _reference_case(lane, names):
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        copy = bundle_copy(td, source=lane.bundle())
        env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": td,
               "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        control = subprocess.run([sys.executable, "predict.py", "--check"], cwd=str(copy),
                                 capture_output=True, text=True, timeout=900, env=env)
        CASES.expect(names[1], control.returncode, control.stdout + control.stderr, code=0,
                     contains=("checks passed",))
        nodes = copy / "graph" / "reference-nodes.parquet"
        from pyarrow import fs      # Arrow's own reader: no Python in its threads at exit (A40)
        frame = pd.read_parquet(nodes, filesystem=fs.LocalFileSystem())
        column = next(c for c in frame.columns if c != "ref"
                      and pd.api.types.is_numeric_dtype(frame[c]))
        frame.loc[0, column] = frame.loc[0, column] + 1.0
        frame.to_parquet(nodes, index=False)
        proc = subprocess.run([sys.executable, "predict.py", "--check"], cwd=str(copy),
                              capture_output=True, text=True, timeout=900, env=env)
        CASES.expect(names[0], proc.returncode, proc.stdout + proc.stderr, code=1,
                     contains=("graph/reference-nodes.parquet",
                               "is not the file MANIFEST.json pinned"))


# --------------------------------------------------------------------------- what shipped

def bundle_negatives():
    CASES.section("the bundle itself — it carries the serving slice, and runs without the "
                  "chain", gate="vega-pack/pack.py")
    bundle = bundled()
    package = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8")).get("package",
                                                                                    "pkg")
    for forbidden in ("ingestion", "training", "evaluation"):
        present = (bundle / package / forbidden).exists()
        CASES.check(f"the bundle carries no {forbidden}/ package", not present,
                    f"{forbidden}/ shipped with the bundle")

    # what pack.py *writes* must be clean; running the notebook or --check inside a bundle
    # afterwards leaves caches, and that is the user's machine doing what python does rather
    # than something the packer shipped
    with tempfile.TemporaryDirectory() as td:
        rc, _ = run(pack_cmd(CAMP, SHAPED, Path(td) / "b", ship_anyway=STANCE))
        fresh = next((d for d in (Path(td) / "b").glob("*") if d.is_dir()), None)
        caches = list(fresh.rglob("__pycache__")) if fresh else ["(the pack failed)"]
    CASES.check("a freshly packed bundle ships source, not bytecode", not caches,
                f"{len(caches)} __pycache__ director(y/ies) shipped")

    schema = json.loads((bundle / "schema.json").read_text(encoding="utf-8"))
    rows = (bundle / "examples" / "rows.csv").read_text(encoding="utf-8").splitlines()[0]
    leaked = [c for c in (schema.get("ignored") or []) if c in rows.split(",")]
    CASES.check("the example rows carry no label and no leakage column", not leaked,
                f"{leaked} shipped in examples/rows.csv")

    # the whole claim: it runs with the skills unreachable
    with tempfile.TemporaryDirectory() as td:
        loose = Path(td) / bundle.name
        shutil.copytree(bundle, loose)
        proc = subprocess.run([sys.executable, "predict.py", "--check"], cwd=str(loose),
                              capture_output=True, text=True, timeout=900,
                              env={"PATH": "/usr/bin:/bin", "HOME": td})
        ok_ = proc.returncode == 0 and "2 checks passed" in proc.stdout
        CASES.check("the bundle reproduces itself from a clean shell", ok_,
                    (proc.stdout + proc.stderr)[-300:], out=proc.stdout + proc.stderr)

    layered = WORK / "orders.domain.analysis.shaped.vega.train.eval.pack.html"
    if layered.is_file():
        html = layered.read_text(encoding="utf-8")
        for what, needle in (("the seal's token", json.loads(
                (CAMP / "test_seal.json").read_text(encoding="utf-8")).get("token", "")),):
            clean = bool(needle) and needle not in html
            CASES.check(f"the published layer never carries {what}", clean,
                        f"{what} is in {layered.name}")


def main():
    if bundled() is None:
        print(f"ERROR: no bundle under {WORK / 'bundle'}; run acceptance.sh first")
        return 1
    pack_negatives()
    options_negatives()
    v23_orders_negatives()
    card_negatives()
    notebook_negatives()
    layer_negatives()
    inspection_negatives()
    lane_negatives()
    bundle_negatives()
    return CASES.finish(ARGS.json)


if __name__ == "__main__":
    sys.exit(main())
