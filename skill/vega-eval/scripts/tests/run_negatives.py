#!/usr/bin/env python3
"""
run_negatives.py — every /vega-eval gate must refuse what it says it refuses.

    python3 run_negatives.py <campaign-work-dir> [-v] [--json <path>]

`acceptance.sh` proves the evaluation runs. That is the easy half, and for this stage it is
the less important one: the whole point of /vega-eval is a set of refusals, and a gate that
never fires is indistinguishable from no gate. So this script seeds one defect at a time
against a campaign that has already been evaluated, and asserts that the right script refuses
it with a message that names the fix.

Two of these cases matter more than the rest, and they are the reason the file exists:

  * **the test part cannot be opened twice without the exposure being recorded** — not by
    running again, not by deleting the seal, not by pointing a fresh campaign at the same
    part;
  * **a number in a reading that no record supports is refused** — the readings are where the
    verdict is stated in prose, and a verdict whose digits drifted from its evidence reads as
    evidence while being a memory of it.

Nothing here mutates the real campaign: every seeded case works on a copy.

Exit 0 when every seeded defect was caught by the gate that should catch it. With `--json`,
the cases are also written to <path> as `vega/negatives-result@1` (id, name, ok, detail and,
for a case older than v2.3, the gate it is grandfathered under; passed and failed
counts), whatever the outcome.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from pathlib import Path as pathlib_Path

S = Path(__file__).resolve().parent.parent
SKILL = S.parent
FORGE = SKILL.parent / "vega-forge" / "scripts"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vega" / "scripts" / "tests"
                       / "negatives"))

import negkit  # noqa: E402

sys.path.insert(0, str(S))
from threshold_choice import requested_mode  # noqa: E402

_ap = argparse.ArgumentParser(description="seed one defect at a time; every gate must refuse it")
_ap.add_argument("work", nargs="?", default=None, help="the acceptance work dir (default: .)")
_ap.add_argument("-v", dest="verbose", action="store_true", help="print every case's output")
_ap.add_argument("--json", default=None, metavar="PATH",
                 help="also write the cases as vega/negatives-result@1 to PATH")
ARGS = _ap.parse_args()
WORK = Path(ARGS.work).resolve() if ARGS.work else Path.cwd()
VERBOSE = ARGS.verbose
CAMP = WORK / "vega-runs" / "2026-09-05T11-03Z"
MODEL = WORK / "orders.domain.analysis.shaped.vega.train.html"


def evaluated_run():
    """Which run the fixture's evaluation looked at, read from the seal it wrote.

    Hard-coding a run id here couples the negatives to whichever rung the selection rule
    happened to pick, and the day a better trial wins every case in this file starts failing
    for a reason that has nothing to do with the gate it is testing.
    """
    seal = CAMP / "test_seal.json"
    if seal.is_file():
        return json.loads(seal.read_text(encoding="utf-8"))["run"]
    ledger = CAMP / "experiments.json"
    if ledger.is_file():
        return ((json.loads(ledger.read_text(encoding="utf-8")).get("selection") or {})
                .get("run") or "R0")
    return "R0"


RUN = evaluated_run()
AUTHORING = Path(__file__).resolve().parent / "evaluation-layer.authoring.json"


def _authoring_base(camp) -> Path:
    """The judgement the acceptance composed from this campaign's records (write_authoring.py,
    kept as evaluation-layer.base.json) when it exists — its numbers are the run's — else the
    static file, kept as the format example."""
    gen = Path(camp) / "evaluation-layer.base.json"
    return gen if gen.is_file() else AUTHORING
BOOT = ["--n-boot", "40"]

CASES = negkit.Cases(__file__, WORK, VERBOSE, head=True)


def run(cmd, cwd=None):
    proc = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True,
                          cwd=str(cwd or WORK), timeout=900)
    return proc.returncode, (proc.stdout + proc.stderr)


def expect(name, rc, out, *, code=2, contains=(), absent=()):
    """A negative passes when the script refused AND said why in the words that name the fix."""
    CASES.expect(name, rc, out, code=code, contains=contains, absent=absent)


def section(title, script):
    """A heading; the cases under it (no id: older than v2.3) are grandfathered under the
    script that refuses them (vega/scripts/tests/negatives/grandfathered.json)."""
    CASES.section(title, gate=script)


def campaign_copy(td, *, drop_seal=False, drop_eval=False, mutate_seal=None,
                  mutate_metrics=None, mutate_threshold=None, mutate_slices=None,
                  mutate_baselines=None, mutate_experiments=None):
    """A copy of the campaign with one thing changed. The original is never touched."""
    dst = Path(td) / "vega-runs" / "2026-09-05T11-03Z"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CAMP, dst)
    if drop_seal:
        (dst / "test_seal.json").unlink(missing_ok=True)
        doc = json.loads((dst / "campaign.json").read_text(encoding="utf-8"))
        doc.pop("test_seal", None)
        (dst / "campaign.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    if drop_eval:
        shutil.rmtree(dst / "eval", ignore_errors=True)
    for name, mutate in (("test_seal.json", mutate_seal),
                         (f"eval/{RUN}/metrics.json", mutate_metrics),
                         (f"eval/{RUN}/threshold.json", mutate_threshold),
                         (f"eval/{RUN}/slices.json", mutate_slices),
                         (f"eval/{RUN}/baselines.json", mutate_baselines),
                         ("experiments.json", mutate_experiments)):
        if mutate is None:
            continue
        path = dst / name
        doc = json.loads(path.read_text(encoding="utf-8"))
        mutate(doc)
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return dst


def authoring_copy(camp, **readings):
    """The acceptance authoring file with one reading rewritten, written beside the campaign."""
    doc = json.loads(_authoring_base(camp).read_text(encoding="utf-8"))
    for key, text in readings.items():
        if text is None:
            doc["readings"].pop(key, None)
        else:
            doc["readings"][key] = text
    path = Path(camp) / "evaluation-layer.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def apply_cmd(camp, data, out, notebook=None):
    cmd = [S / "apply_evaluation_layer.py", MODEL, "--data", data, "--out", out]
    if notebook:
        cmd += ["--notebook", notebook]
    return cmd


# --------------------------------------------------------------------------- the seal

def seal_negatives():
    section("evaluate.py — the test part is opened once, under a seal, and every further look "
          "is on the record", "vega-eval/evaluate.py")

    rc, out = run([S / "evaluate.py", "--campaign", CAMP, *BOOT])
    expect("a second look with no --again is refused", rc, out,
           contains=("already been opened", "--again"))

    rc, out = run([S / "evaluate.py", "--campaign", CAMP, *BOOT, "--again", ""])
    expect("--again with an empty reason is refused", rc, out,
           contains=("--again",))

    rc, out = run([S / "evaluate.py", "--campaign", CAMP, *BOOT,
                   "--again", "a reviewer asked", "--threshold", "0.42"])
    expect("a threshold value chosen after a look is refused", rc, out,
           contains=("knowing the answer", "valid-optimum"))

    first = json.loads((CAMP / "eval" / json.loads((CAMP / "test_seal.json").read_text(
        encoding="utf-8"))["run"] / "threshold.json").read_text(encoding="utf-8"))
    switched = "valid-optimum" if requested_mode(first) == "0.5" else "0.5"
    rc, out = run([S / "evaluate.py", "--campaign", CAMP, *BOOT,
                   "--again", "a reviewer asked", "--threshold", switched])
    expect("T1b: a further look that switches the threshold mode is refused", rc, out,
           contains=("knowing the answer", "keeps the mode"))

    rc, out = run([S / "evaluate.py", "--campaign", CAMP, *BOOT,
                   "--again", "a reviewer asked", "--slices", "region=4"])
    expect("a slice named after a look is refused", rc, out,
           contains=("fitted to the test part",))

    rc, out = run([S / "evaluate.py", "--campaign", CAMP, *BOOT,
                   "--again", "a reviewer asked", "--baselines", "trivial,linear"])
    expect("a baseline set narrower than the last exposure's is refused", rc, out,
           contains=("narrower", "after seeing it"))

    rc, out = run([S / "evaluate.py", "--campaign", CAMP, *BOOT, "--baselines", "linear,tree"])
    expect("a baseline set without the trivial floor is refused", rc, out,
           contains=("trivial", "floor"))

    rc, out = run([S / "evaluate.py", "--campaign", CAMP, *BOOT, "--baselines", "trivial,lens"])
    expect("a baseline set with neither linear nor tree is refused", rc, out,
           contains=("neither", "non-linearity"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_seal=True)
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("deleting the seal does not reset the exposure count", rc, out,
               contains=("already holds an evaluation", "restore"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_seal=True, drop_eval=True)
        rc, out = run([S / "evaluate.py", "--campaign", dst, "--run", "R9999", *BOOT])
        expect("a run that is not in the ledger is refused", rc, out,
               contains=("not in the ledger",))

    with tempfile.TemporaryDirectory() as td:
        def wrong_digest(doc):
            for row in doc["runs"]:
                if row["id"] == RUN:
                    row["checkpoint_digest"] = "sha256:" + "0" * 64
        dst = campaign_copy(td, drop_seal=True, drop_eval=True,
                            mutate_experiments=wrong_digest)
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("a checkpoint whose digest is not the ledger's is refused before the seal",
               rc, out, contains=("digest", "wrong weights"))
        expect("...and no seal was written for it",
               0 if not (dst / "test_seal.json").is_file() else 1, "", code=0)

    # ---- the refusals that guard the test part itself. Each of these was reachable only by a
    # code path nothing exercised; a gate that never fires is indistinguishable from no gate,
    # and these are the gates the whole stage is for.

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_seal=True, drop_eval=True)
        (dst / "runs" / RUN / "checkpoints" / "best.pt").unlink()
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("a selected run with no checkpoint is refused before the seal", rc, out,
               contains=("best.pt", "no checkpoint"))
        expect("...and no seal was written for it",
               0 if not (dst / "test_seal.json").is_file() else 1, "", code=0)

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_seal=True, drop_eval=True)
        spec = json.loads((dst / "netspec.json").read_text(encoding="utf-8"))
        spec.setdefault("input", {})["manifest"] = str(dst / "no-such-manifest.json")
        (dst / "netspec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("an unreadable shaped manifest is refused before the seal", rc, out,
               contains=("manifest", "never the spec"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_seal=True, drop_eval=True)
        spec = json.loads((dst / "netspec.json").read_text(encoding="utf-8"))
        man_path = pathlib_Path(spec["input"]["manifest"])
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["outputs"]["files"]["test"]["path"] = str(dst / "gone.parquet")
        moved = dst / "manifest.moved.json"
        moved.write_text(json.dumps(man, indent=2), encoding="utf-8")
        spec["input"]["manifest"] = str(moved)
        (dst / "netspec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("a test part the manifest names but that does not exist is refused", rc, out,
               contains=("does not exist",))

    with tempfile.TemporaryDirectory() as td:
        # THE seal-critical one: the file the manifest sealed is not the file on disk. It is
        # refused after the seal is written, deliberately — the look was authorised, the
        # exposure is spent, and the record says so.
        dst = campaign_copy(td, drop_seal=True, drop_eval=True)
        spec = json.loads((dst / "netspec.json").read_text(encoding="utf-8"))
        man_path = pathlib_Path(spec["input"]["manifest"])
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["outputs"]["files"]["test"]["digest"] = "sha256:" + "1" * 64
        moved = dst / "manifest.retampered.json"
        moved.write_text(json.dumps(man, indent=2), encoding="utf-8")
        spec["input"]["manifest"] = str(moved)
        (dst / "netspec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("a test part that does not hash to the manifest's digest is refused", rc, out,
               code=3, contains=("a new chain, not a new number",))
        sealed = (dst / "test_seal.json").is_file()
        expect("...and the seal stands, with the exposure spent",
               0 if sealed else 1, "", code=0)
        if sealed:
            seal = json.loads((dst / "test_seal.json").read_text(encoding="utf-8"))
            expect("...recording the mismatch rather than a passed check",
                   0 if seal["test"]["digest_matches_manifest"] is False else 1, "", code=0)

    with tempfile.TemporaryDirectory() as td:
        # a manifest with no digest at all: the comparison cannot be made, and the seal must
        # say so rather than record `true` for a check nobody ran
        dst = campaign_copy(td, drop_seal=True, drop_eval=True)
        spec = json.loads((dst / "netspec.json").read_text(encoding="utf-8"))
        man_path = pathlib_Path(spec["input"]["manifest"])
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["outputs"]["files"]["test"].pop("digest", None)
        moved = dst / "manifest.nodigest.json"
        moved.write_text(json.dumps(man, indent=2), encoding="utf-8")
        spec["input"]["manifest"] = str(moved)
        (dst / "netspec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("a manifest with no digest is evaluated, with the gap on the record", rc, out,
               code=0, contains=("records no digest", "could not be checked"))
        seal = json.loads((dst / "test_seal.json").read_text(encoding="utf-8"))
        expect("...and the seal records `null`, not a passed check",
               0 if seal["test"]["digest_matches_manifest"] is None else 1, "", code=0)

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        (dst / "test_seal.json").write_text("{ not json", encoding="utf-8")
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT, "--again", "a reason"])
        expect("a malformed seal is refused rather than replaced", rc, out,
               contains=("test_seal.json",))


# --------------------------------------------------------------------------- A46 (track E)

def _after_eval_copy(td, exposures=None):
    """A fresh campaign (no seal, no eval) whose campaign.json says /vega-train --after-eval
    started it after the acceptance campaign's looks — the same test part, already seen."""
    seen = json.loads((CAMP / "test_seal.json").read_text(encoding="utf-8"))["exposures"]
    dst = campaign_copy(td, drop_seal=True, drop_eval=True)
    doc = json.loads((dst / "campaign.json").read_text(encoding="utf-8"))
    doc["after_eval"] = {"previous_campaign": str(CAMP),
                         "exposures": seen if exposures is None else exposures}
    (dst / "campaign.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return dst, seen


def _no_seal(name, dst):
    expect(name, 0 if not (dst / "test_seal.json").is_file() else 1, "", code=0)


def _analysis_with(dst, importance):
    """The chain's model with its analysis layer's importance protocol replaced, written where
    evaluate.py looks for it first (beside the campaign directory)."""
    import re
    spec = json.loads((dst / "netspec.json").read_text(encoding="utf-8"))
    name = Path((spec.get("input") or {}).get("model") or "").name
    html = (WORK / name).read_text(encoding="utf-8")
    pattern = r'(<script\s+id="layer-analysis-data"[^>]*>)([\s\S]*?)(</script>)'
    match = re.search(pattern, html)
    layer = json.loads(match.group(2))
    layer.setdefault("modules", {})["importance"] = importance
    text = json.dumps(layer).replace("</", "<\\/")
    (dst.parent / name).write_text(html[:match.start(2)] + text + html[match.end(2):],
                                   encoding="utf-8")


def closure_negatives():
    section("evaluate.py — the promises A46 put in code: slices coerced, earlier looks counted, "
            "R0 predicted through the seal, lens's hyper-parameters, mape's gate",
            "vega-eval/evaluate.py")

    # ---- T3: a user slice is declared before the seal, its literal coerced to the dtype
    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_seal=True, drop_eval=True)
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT, "--slices", "region=abc"])
        expect("T3  a slice literal the column's dtype cannot hold is refused before the seal",
               rc, out, contains=("region=abc", "dtype", "before the look"))
        _no_seal("T3  ...and no seal was written for it", dst)
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT, "--slices", "nosuch=1"])
        expect("T3a  a slice on a column that is not a feature is refused before the seal",
               rc, out, contains=("nosuch", "not one of the features"))
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT, "--slices", "region"])
        expect("T3b  a slice with no value is refused before the seal", rc, out,
               contains=("gives no value",))
        _no_seal("T3b  ...and no seal was written for any of them", dst)

    # ---- S3: an after-eval campaign's first look is a further exposure
    with tempfile.TemporaryDirectory() as td:
        dst, seen = _after_eval_copy(td)
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT, "--threshold", "0.42"])
        expect("S3  an after-eval campaign's first look refuses a threshold value", rc, out,
               contains=("after-eval campaign", "knowing the answer"))
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT, "--slices", "region=4"])
        expect("S3a  an after-eval campaign's first look refuses user slices", rc, out,
               contains=("after-eval campaign", "fitted to the test part"))
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT,
                       "--baselines", "trivial,linear"])
        expect("S3b  an after-eval campaign's first look refuses a narrower baseline set", rc,
               out, contains=("narrower", "previous campaign"))
        _no_seal("S3b  ...and no seal was written for any of them", dst)
    with tempfile.TemporaryDirectory() as td:
        dst, seen = _after_eval_copy(td, exposures=7)
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("S3c  an after-eval record the previous seal does not reconcile is refused", rc,
               out, contains=("records 7", "reconcile"))
        _no_seal("S3c  ...and no seal was written for it", dst)
    with tempfile.TemporaryDirectory() as td:
        dst, seen = _after_eval_copy(td)
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT])
        expect("S3d  an after-eval campaign's first look runs as a further exposure", rc, out,
               code=0, contains=(f"seen {seen + 1} times", "previous campaign"))
        seal = json.loads((dst / "test_seal.json").read_text(encoding="utf-8")) \
            if (dst / "test_seal.json").is_file() else {}
        CASES.check("S3d  ...and its seal counts the earlier looks",
                    seal.get("exposures") == seen + 1
                    and (seal.get("after_eval") or {}).get("exposures") == seen,
                    f"the seal records {seal.get('exposures')} exposure(s) and after_eval "
                    f"{seal.get('after_eval')}, not {seen + 1} and {seen}")

    # ---- B5, B6, T3c: one first look on a copy whose R0 checkpoint no longer hashes to the
    # ledger, whose analysis records an invalid hyper-parameter, with a slice the old text
    # comparison lost (a code the design matrix holds as 4.0)
    with tempfile.TemporaryDirectory() as td:
        def r0_changed(doc):
            for row in doc["runs"]:
                if row["id"] == "R0":
                    row["checkpoint_digest"] = "sha256:" + "2" * 64
        dst = campaign_copy(td, drop_seal=True, drop_eval=True, mutate_experiments=r0_changed)
        _analysis_with(dst, {"family": "HistGradientBoosting",
                             "hyper_parameters": {"max_depth": -3, "n_estimators": 5}})
        rc, out = run([S / "evaluate.py", "--campaign", dst, *BOOT, "--slices", "region=4"])
        expect("B5  a campaign-baseline whose R0 weights changed is not predicted, and says so",
               rc, out, code=0, contains=("campaign-baseline was not predicted",
                                          "weights changed"))
        ev = dst / "eval" / (json.loads((dst / "test_seal.json").read_text(encoding="utf-8"))
                             ["run"] if (dst / "test_seal.json").is_file() else RUN)
        rows = {}
        if (ev / "baselines.json").is_file():
            rows = {b["name"]: b for b in json.loads((ev / "baselines.json")
                                                      .read_text(encoding="utf-8"))["baselines"]}
        cb = rows.get("campaign-baseline") or {}
        CASES.check("B5  ...its row keeps the ledger's valid number and no test number",
                    cb.get("predicted") is False and cb.get("test") is None
                    and "weights changed" in str(cb.get("why")), f"the row is {cb}")
        expect("B6  lens is fitted with the analysis's recorded hyper-parameters — an invalid "
               "one fails the fit, by name", rc, out, code=0,
               contains=("'lens' was not fitted", "max_depth"))
        lens = rows.get("lens") or {}
        CASES.check("B6  ...and one the estimator does not take is named as not carried",
                    "n_estimators" in ((lens.get("hyper_parameters") or {})
                                       .get("not_carried") or {})
                    and "n_estimators" in out,
                    f"the lens row is {lens.get('hyper_parameters')}")
        sliced = {}
        if (ev / "slices.json").is_file():
            sliced = {r["name"]: r for r in json.loads((ev / "slices.json")
                                                        .read_text(encoding="utf-8"))["slices"]}
        user = sliced.get("region=4") or {}
        CASES.check("T3c  a user slice is coerced to the column's dtype and measured",
                    user.get("source") == "user:turn" and int(user.get("n") or 0) >= 5,
                    f"region=4 was {'measured' if user else 'lost'}: {user.get('n')} row(s)")

    # ---- B5: the acceptance's own look predicted R0 through the seal, with a test CI
    rows = {b["name"]: b for b in json.loads((CAMP / "eval" / RUN / "baselines.json")
                                             .read_text(encoding="utf-8"))["baselines"]}
    metrics = json.loads((CAMP / "eval" / RUN / "metrics.json").read_text(encoding="utf-8"))
    cb = rows.get("campaign-baseline") or {}
    CASES.check("B5a  campaign-baseline is predicted through the seal with a test CI and a "
                "paired difference",
                cb.get("predicted") is True and len((cb.get("test") or {}).get("ci") or []) == 2
                and "campaign-baseline" in (metrics.get("differences") or {})
                and metrics.get("verdict", {}).get("strongest_baseline") != "campaign-baseline",
                f"the row is {cb}, the differences {sorted(metrics.get('differences') or {})}")


def closure_layer_negatives():
    section("apply_evaluation_layer.py — what A46 added to the refusals",
            "vega-eval/apply_evaluation_layer.py")

    def apply_on(dst):
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        return run(apply_cmd(dst, data, Path(dst).parent / "out.html", dst / "evaluate.ipynb"))

    with tempfile.TemporaryDirectory() as td:
        def bare(doc):
            metric = sorted(doc.get("metrics_at_threshold") or {})[0]
            doc["metrics_at_threshold"][metric].pop("ci", None)
        dst = campaign_copy(td, mutate_threshold=bare)
        rc, out = apply_on(dst)
        expect("A1  a threshold metric published without its confidence interval is refused",
               rc, out, contains=("metrics_at_threshold", "confidence interval"))

    with tempfile.TemporaryDirectory() as td:
        def upstream_test(doc):
            for b in doc["baselines"]:
                if b["name"] == "campaign-baseline":
                    b["test"] = {"value": (b.get("valid") or {}).get("value"), "_upstream": True}
        dst = campaign_copy(td, mutate_baselines=upstream_test)
        rc, out = apply_on(dst)
        expect("A2  a test number marked _upstream is refused", rc, out,
               contains=("marked _upstream", "no earlier stage saw the test part"))

    with tempfile.TemporaryDirectory() as td:
        def gated(doc):
            doc["gates"] = {"mape": {"applies": False, "share": 0.8, "part": "train",
                                     "rule": "|y| > 0.1 for 99 % of the train part's rows",
                                     "why": "|y| > 0.1 on only 80.00% of the train rows"}}
            doc["models"]["network"]["mape"] = {"value": 0.31, "ci": [0.2, 0.4], "n": 1}
        dst = campaign_copy(td, mutate_metrics=gated)
        rc, out = apply_on(dst)
        expect("A3  a metric its gate withheld, published with a value, is refused", rc, out,
               contains=("mape", "withheld"))


# --------------------------------------------------------------------------- the notebook

def notebook_negatives():
    section("nb_emit.py --stage eval — the notebook documents a look that happened",
            "vega-forge/nb_emit.py")

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_seal=True)
        rc, out = run([FORGE / "nb_emit.py", "--stage", "eval", "--from", dst,
                       "--out", Path(td) / "e.ipynb"])
        expect("an eval notebook with no seal is refused", rc, out,
               contains=("test_seal.json", "evaluate.py"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        (dst / "eval" / RUN / "predictions.test.parquet").unlink()
        rc, out = run([FORGE / "nb_emit.py", "--stage", "eval", "--from", dst,
                       "--out", Path(td) / "e.ipynb"])
        expect("an eval notebook with no stored predictions is refused", rc, out,
               contains=("predictions.test.parquet", "opening the test part again"))


# --------------------------------------------------------------------------- the layer

def layer_negatives():
    section("apply_evaluation_layer.py — the layer publishes what the evaluation measured, "
          "and only what the readings can support", "vega-eval/apply_evaluation_layer.py")

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        nb = dst / "evaluate.ipynb"
        shutil.copy(CAMP / "evaluate.ipynb", nb)
        out_html = Path(td) / "out.html"

        data = authoring_copy(dst, verdict=(
            "R0 scores pr_auc 0.4596 [0.1723, 0.7703] on the 120 held-out rows. Against "
            "linear the paired difference is -0.1266 [-0.3846, 0.0908], so the network has "
            "earned its cost."))
        rc, out = run(apply_cmd(dst, data, out_html, nb))
        expect("a reading that says earned when the record says not earned is refused",
               rc, out, contains=("reads as earned", "they disagree"))

        data = authoring_copy(dst, verdict=(
            "R0 scores pr_auc 0.46 on the held-out rows and the paired difference against "
            "linear is -0.13, so the network has not earned its cost against a logistic "
            "regression."))
        rc, out = run(apply_cmd(dst, data, out_html, nb))
        expect("a verdict reading with no interval bounds is refused", rc, out,
               contains=("lower bound", "rounding"))

        data = authoring_copy(dst, limits=(
            "One test part, one seed, 350 rows. The interval on the primary metric runs from "
            "0.1723 to 0.7703, which is the honest width of what the part can say."))
        rc, out = run(apply_cmd(dst, data, out_html, nb))
        expect("a retyped digit no record supports is refused", rc, out,
               contains=("350", "no spliced record holds"))

        # The worst slice is whatever this campaign's evaluation found; naming one here would
        # test the gate against a finding the fixture may no longer hold, and the case would
        # then be caught by the retyped-digit rule instead — a pass for the wrong reason.
        worst = json.loads((CAMP / "eval" / RUN / "slices.json")
                           .read_text(encoding="utf-8"))["worst"]
        data = authoring_copy(dst, slices=(
            "The worst slice sits on one side of a median split, and the network's difficulty "
            "is concentrated there rather than spread across the part."))
        rc, out = run(apply_cmd(dst, data, out_html, nb))
        expect("a slices reading that does not name the worst slice is refused", rc, out,
               contains=("worst slice", str(worst["name"])))

        data = authoring_copy(dst, slices=(
            f"The worst slice on pr_auc is {worst['name']} at {worst['value']}, and the "
            "network's difficulty is concentrated there rather than spread across the part, "
            "so the whole-part figure should not be quoted without it."))
        rc, out = run(apply_cmd(dst, data, out_html, nb))
        expect("a slices reading that names the worst slice without its n is refused",
               rc, out, contains=("without its n",))

        data = authoring_copy(dst, calibration=None)
        rc, out = run(apply_cmd(dst, data, out_html, nb))
        expect("a missing reading is refused", rc, out,
               contains=("readings.calibration",))

        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, out_html))
        expect("a layer with no notebook is refused", rc, out, contains=("notebook",))

    with tempfile.TemporaryDirectory() as td:
        def strip_ci(doc):
            doc["models"]["network"]["pr_auc"].pop("ci", None)
        dst = campaign_copy(td, mutate_metrics=strip_ci)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, Path(td) / "out.html", dst / "evaluate.ipynb"))
        expect("a metric published without its confidence interval is refused", rc, out,
               contains=("confidence interval", "without a scale"))

    with tempfile.TemporaryDirectory() as td:
        def on_test(doc):
            doc["source"] = "test-optimum"
        dst = campaign_copy(td, mutate_threshold=on_test)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, Path(td) / "out.html", dst / "evaluate.ipynb"))
        expect("a threshold chosen on the test part is refused", rc, out,
               contains=("names the test part", "after the look"))

    with tempfile.TemporaryDirectory() as td:
        def user_threshold(doc):
            doc["source"] = "user:turn"
        def twice(doc):
            doc["exposures"] = 2
        dst = campaign_copy(td, mutate_threshold=user_threshold, mutate_seal=twice)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, Path(td) / "out.html", dst / "evaluate.ipynb"))
        expect("a user threshold on a second exposure is refused", rc, out,
               contains=("exposure 2", "knowing the answer"))

    with tempfile.TemporaryDirectory() as td:
        def drop_trivial(doc):
            doc["baselines"] = [b for b in doc["baselines"] if b["name"] != "trivial"]
        dst = campaign_copy(td, mutate_baselines=drop_trivial)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, Path(td) / "out.html", dst / "evaluate.ipynb"))
        expect("a published baseline set without the trivial floor is refused", rc, out,
               contains=("trivial", "floor"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        shutil.rmtree(dst / "eval" / RUN)
        (dst / "eval" / RUN).mkdir()
        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, Path(td) / "out.html", dst / "evaluate.ipynb"))
        expect("a layer whose from_eval holds no records is refused", rc, out,
               contains=("metrics.json", "run evaluate.py"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, MODEL, dst / "evaluate.ipynb"))
        expect("writing the layer over its own input is refused", rc, out,
               contains=("never modified",))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_seal=True)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, Path(td) / "out.html", dst / "evaluate.ipynb"))
        expect("publishing test numbers with no seal beside them is refused", rc, out,
               contains=("test_seal.json", "impossible"))

    with tempfile.TemporaryDirectory() as td:
        def mismatched(doc):
            doc["test"]["digest_matches_manifest"] = False
        dst = campaign_copy(td, mutate_seal=mismatched)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        rc, out = run(apply_cmd(dst, data, Path(td) / "out.html", dst / "evaluate.ipynb"))
        expect("a layer whose seal records a digest mismatch is refused", rc, out,
               contains=("does not hash", "the chain sealed"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        doc = json.loads(AUTHORING.read_text(encoding="utf-8"))
        doc.pop("from_eval")
        data = Path(dst) / "evaluation-layer.json"
        data.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        rc, out = run(apply_cmd(dst, data, Path(td) / "out.html", dst / "evaluate.ipynb"))
        expect("an authoring file with no from_eval is refused", rc, out,
               contains=("from_eval", "never retyped"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        big = Path(td) / "too-big.js"
        big.write_text("/*" + ("x" * (61 * 1024)) + "*/", encoding="utf-8")
        rc, out = run([S / "apply_evaluation_layer.py", MODEL, "--data", data,
                       "--out", Path(td) / "o.html", "--notebook", dst / "evaluate.ipynb",
                       "--render", big])
        expect("a render script over the contract's cap is refused", rc, out,
               contains=("caps it at",))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        shutil.copy(CAMP / "evaluate.ipynb", dst / "evaluate.ipynb")
        data = authoring_copy(dst)
        rc, out = run([S / "apply_evaluation_layer.py", MODEL, "--data", data,
                       "--out", Path(td) / "o.html", "--notebook", dst / "evaluate.ipynb",
                       "--domain-forge-dir", Path(td) / "nowhere"])
        expect("a missing platform writer is refused, and never worked around", rc, out,
               code=3, contains=("apply_layer.py", "never by this script"))


# --------------------------------------------------------------------------- the token

def token_negatives():
    section("the published layer never carries the seal's token",
            "vega-eval/apply_evaluation_layer.py")

    layered = WORK / "orders.domain.analysis.shaped.vega.train.eval.html"
    if not layered.is_file():
        for name in ("the token is not in the published model",
                     '"token" is not a key in the layer data'):
            CASES.not_run(name, f"{layered.name} not built yet")
        return
    seal = json.loads((CAMP / "test_seal.json").read_text(encoding="utf-8"))
    token = seal.get("token") or ""
    html = layered.read_text(encoding="utf-8")
    CASES.check("the token is not in the published model", bool(token) and token not in html,
                "the layer carries the token that opens the test part")
    CASES.check('"token" is not a key in the layer data', '"token"' not in html,
                '"token" is a key of the layer data')


def main():
    if not CAMP.is_dir():
        print(f"ERROR: no campaign at {CAMP}; run acceptance.sh first")
        return 1
    if not (CAMP / "test_seal.json").is_file():
        print(f"ERROR: no test_seal.json in {CAMP}; run evaluate.py first")
        return 1
    seal_negatives()
    closure_negatives()
    notebook_negatives()
    layer_negatives()
    closure_layer_negatives()
    token_negatives()
    return CASES.finish(ARGS.json)

if __name__ == "__main__":
    sys.exit(main())
