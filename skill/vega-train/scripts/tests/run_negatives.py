#!/usr/bin/env python3
"""
run_negatives.py — every /vega-train gate must refuse what it says it refuses.

    python3 run_negatives.py <campaign-work-dir> [-v] [--json <path>]

`acceptance.sh` proves the campaign runs. That is the easy half. A gate that never fires is
indistinguishable from no gate, so this script seeds one defect at a time against a campaign
that already ran, and asserts that the right script refuses it with a message that names the
fix. Nothing here mutates the real campaign: every seeded case works on a copy.

Exit 0 when every seeded defect was caught by the gate that should catch it. With `--json`,
the cases are also written to <path> as `vega/negatives-result@1` (id, name, ok, detail and,
for a case older than v2.3, the gate it is grandfathered under; passed and failed
counts), whatever the outcome.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

S = Path(__file__).resolve().parent.parent
FORGE = S.parent.parent / "vega-forge" / "scripts"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vega" / "scripts" / "tests"
                       / "negatives"))

import negkit  # noqa: E402

_ap = argparse.ArgumentParser(description="seed one defect at a time; every gate must refuse it")
_ap.add_argument("work", nargs="?", default=None, help="the acceptance work dir (default: .)")
_ap.add_argument("-v", dest="verbose", action="store_true", help="print every case's output")
_ap.add_argument("--json", default=None, metavar="PATH",
                 help="also write the cases as vega/negatives-result@1 to PATH")
ARGS = _ap.parse_args()
WORK = Path(ARGS.work).resolve() if ARGS.work else Path.cwd()
VERBOSE = ARGS.verbose
CAMP = WORK / "vega-runs" / "2026-09-05T11-03Z"
MODEL = WORK / "orders.domain.analysis.shaped.vega.html"

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


def campaign_copy(td, mutate_campaign=None, mutate_ladder=None, mutate_experiments=None,
                  seal=False, drop_runs=False, mutate_snapshot=None):
    """A copy of the campaign with one thing changed. The original is never touched."""
    dst = Path(td) / "vega-runs" / "2026-09-05T11-03Z"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("runs") if drop_runs else None
    shutil.copytree(CAMP, dst, ignore=ignore)
    # The fixture campaign may have been evaluated since — /vega-eval leaves a seal beside
    # campaign.json — and every gate in this file would then refuse for that reason instead of
    # the one being seeded. The copy starts unsealed; the one case that is *about* the seal
    # asks for it with seal=True.
    (dst / "test_seal.json").unlink(missing_ok=True)
    for name, mutate in (("campaign.json", mutate_campaign), ("ladder.json", mutate_ladder),
                         ("experiments.json", mutate_experiments),
                         ("spec_snapshot.json", mutate_snapshot)):
        if mutate is None:
            continue
        path = dst / name
        doc = json.loads(path.read_text(encoding="utf-8"))
        mutate(doc)
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    if seal:
        (dst / "test_seal.json").write_text(
            json.dumps({"token": "t", "opened_at": "2026-09-05T12:00:00Z", "exposures": 1}),
            encoding="utf-8")
    return dst


# --------------------------------------------------------------------------- plan_ladder

def _digests(root, names):
    """{name: sha256 of the file, or None when it is absent}."""
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            if (root / n).is_file() else None for n in names}


#: a read_paths record in which the emitted code reads only the clipping declaration
ONLY_GRAD_CLIP = ["regularization[kind=grad_clip]/value"]


def only_grad_clip(snapshot):
    snapshot["read_paths"] = list(ONLY_GRAD_CLIP)
    snapshot.setdefault("_resolved", {}).pop("regularization", None)


def plan_negatives():
    section("plan_ladder.py — the ladder is planned once, and never after the test was seen",
            "vega-train/plan_ladder.py")

    records = ("campaign.json", "ladder.json")
    # on an unsealed copy: the fixture campaign may have been evaluated since (a lane run to
    # bundle_api seals it), and the seal's refusal would then fire first, for another reason
    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        before = _digests(dst, records)
        rc, out = run([S / "plan_ladder.py", "--campaign", dst, "--model", MODEL])
        expect("L1a  planning over a directory that holds campaign.json is refused", rc, out,
               contains=("already holds campaign.json", "planned once", "--resume"))
        CASES.check("L1b  the refused plan leaves campaign.json and ladder.json as they were",
                    _digests(dst, records) == before,
                    "a refused plan rewrote the campaign's records")

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        (dst / "campaign.json").unlink()
        rc, out = run([S / "plan_ladder.py", "--campaign", dst, "--model", MODEL])
        expect("a replan over an existing ladder is refused", rc, out,
               contains=("already exists", "append-only"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, seal=True)
        rc, out = run([S / "plan_ladder.py", "--campaign", dst, "--model", MODEL])
        expect("planning in a campaign whose test was opened is refused", rc, out,
               contains=("test part has been opened", "--after-eval"))

    # the forge's fresh run directory, whose snapshot says the code reads only grad_clip of
    # the declaration: the weight-decay default test lands on a dead path and the plan stops
    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_runs=True, mutate_snapshot=only_grad_clip)
        for name in ("campaign.json", "ladder.json", "experiments.json"):
            (dst / name).unlink(missing_ok=True)
        rc, out = run([S / "plan_ladder.py", "--campaign", dst, "--model", MODEL, "--no-hpo"])
        expect("H1h  a default-test rung on a dead path refuses the plan", rc, out,
               contains=("ERROR: dead delta — no emitted code reads "
                         "regularization[kind=weight_decay]/value", "baseline's numbers"))
        CASES.check("H1h  ...and the refused plan wrote neither record",
                    not (dst / "campaign.json").exists() and not (dst / "ladder.json").exists(),
                    "a plan refused for a dead delta wrote a campaign record")


# --------------------------------------------------------------------------- trainer

def trainer_negatives():
    section("trainer.py — one change means one decision, and the seal holds",
            "vega-train/trainer.py")

    def with_rung(td, rung, run_id="RX"):
        def mutate(ladder):
            ladder["runs"] = [r for r in ladder["runs"] if r["id"] != run_id]
            ladder["runs"].append(dict(rung, id=run_id))
        return campaign_copy(td, mutate_ladder=mutate)

    base = {"kind": "ablation", "source": "vega:ladder/default-test",
            "hypothesis": "a seeded defect", "expect": "refused"}

    with tempfile.TemporaryDirectory() as td:
        dst = with_rung(td, dict(base, family="optimization",
                                 delta=[{"path": "backend/name", "to": "keras"}]))
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX"])
        expect("a delta on `backend` is refused outright", rc, out,
               contains=("may not be changed by a run", "the forge's choice, not an experiment"))

    with tempfile.TemporaryDirectory() as td:
        dst = with_rung(td, dict(base, family="optimization", delta=[
            {"path": "optimizer/params/lr", "to": 0.01},
            {"path": "regularization[kind=dropout]/value", "to": 0.4}]))
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX"])
        expect("an ablation spanning two decision families is refused", rc, out,
               contains=("decision families", "one change means one decision"))

    with tempfile.TemporaryDirectory() as td:
        dst = with_rung(td, dict(base, family="optimization",
                                 delta=[{"path": "optimizer/params/nonesuch", "to": 1}]))
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX"])
        expect("a delta path that addresses nothing is refused", rc, out,
               contains=("no key", "nonesuch"))

    with tempfile.TemporaryDirectory() as td:
        dst = with_rung(td, dict(base, family="optimization", delta=[
            {"path": "optimizer/params/lr", "from": 0.5, "to": 0.01}]))
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX"])
        expect("a delta whose `from` disagrees with the spec is refused", rc, out,
               contains=("planned against a different spec",))

    with tempfile.TemporaryDirectory() as td:
        dst = with_rung(td, dict(base, family="optimization", source="vega:whatever",
                                 delta=[{"path": "optimizer/params/lr", "to": 0.01}]))
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX"])
        expect("a rung source outside the closed set is refused", rc, out,
               contains=("is not one the contract allows", "vega:ladder/"))

    with tempfile.TemporaryDirectory() as td:
        dst = with_rung(td, dict(base, family="objective",
                                 delta=[{"path": "loss/name", "to": "mse"}]))
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX"])
        expect("a rung that produces an uncompilable spec is refused before it runs", rc, out,
               contains=("does not compile", "before it runs"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, seal=True)
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "R0"])
        expect("training in a campaign whose test was opened is refused", rc, out,
               contains=("test_seal.json exists", "--after-eval"))

    lr = json.dumps([{"path": "optimizer/params/lr", "to": 0.01}])
    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, drop_runs=True)
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX", "--delta", lr])
        expect("H1a  a --delta run with no ladder rung is refused", rc, out,
               contains=("has no rung in ladder.json", "no run exists without a hypothesis"))
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "R0", "--delta", lr])
        expect("H1b  a --delta that is not the rung's planned delta is refused", rc, out,
               contains=("is not the rung's planned delta",))
        CASES.check("H1a  ...and no run directory was written", not (dst / "runs").exists(),
                    "a refused run left a run directory")

    with tempfile.TemporaryDirectory() as td:
        dst = with_rung(td, dict(base, family="regularization", delta=[
            {"path": "regularization[kind=weight_decay]/value", "to": 0.0}]))
        path = dst / "spec_snapshot.json"
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        only_grad_clip(snapshot)
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX"])
        expect("H1c  a dead delta is refused before it runs", rc, out,
               contains=("dead delta", "no emitted code reads "
                         "regularization[kind=weight_decay]/value", "baseline's numbers"))

    with tempfile.TemporaryDirectory() as td:
        dst = with_rung(td, dict(base, family="optimization", source="user:turn4",
                                 delta=[{"path": "optimizer/params/lr", "to": 0.01}]))
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "RX"])
        expect("H1d  a rung whose expect cannot be checked is refused", rc, out,
               contains=("not checkable", "expect_check"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        rc, out = run([S / "trainer.py", "--campaign", dst, "--run", "R0"])
        expect("re-running a finished run without --resume is refused", rc, out,
               contains=("append-only", "--resume"))


# --------------------------------------------------------------------------- hpo

def hpo_negatives():
    section("hpo.py — a study id is used once",
            "vega-train/hpo.py")
    # on an unsealed copy: once the fixture has been evaluated, the real campaign refuses for
    # the seal's reason, which is a different gate from the one seeded here
    with tempfile.TemporaryDirectory() as td:
        camp = campaign_copy(td)
        rc, out = run([S / "hpo.py", "--campaign", camp, "--study", "H1",
                       "--budget", "trials=1,minutes=1"])
    expect("re-running a study under the same id is refused", rc, out,
           contains=("already has", "append-only", "new study id"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        (dst / "hpo" / "H9").mkdir(parents=True, exist_ok=True)
        (dst / "hpo" / "H9" / "space.json").write_text(json.dumps(
            {"schema": "vega/space@1", "dimensions": [
                {"path": "optimizer/params/lr", "kind": "float", "range": [1e-4, 1e-2]}]}),
            encoding="utf-8")
        rc, out = run([S / "hpo.py", "--campaign", dst, "--study", "H9",
                       "--space", "hpo/H9/space.json", "--budget", "trials=1,minutes=1"])
        expect("H1g  a study the ladder does not plan is refused", rc, out,
               contains=("is not an hpo rung of ladder.json", "Plan it in ladder.json"))

    def plan_h9(ladder):
        ladder["runs"].append({"id": "H9", "kind": "hpo", "study": "H9", "delta": [],
                               "space": "hpo/H9/space.json", "source": "vega:ladder/hpo",
                               "hypothesis": "a seeded defect", "expect": "refused"})

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, mutate_ladder=plan_h9)
        (dst / "hpo" / "H9").mkdir(parents=True, exist_ok=True)
        (dst / "hpo" / "H9" / "space.json").write_text(
            json.dumps({"schema": "vega/space@1", "dimensions": []}), encoding="utf-8")
        rc, out = run([S / "hpo.py", "--campaign", dst, "--study", "H9",
                       "--space", "hpo/H9/space.json"])
        expect("a study with an empty space is refused", rc, out,
               contains=("declares no dimensions", "nothing to search"))


# --------------------------------------------------------------------------- select

def select_negatives():
    section("select.py — an override is never silent",
            "vega-train/select.py")

    rc, out = run([S / "select.py", "--campaign", CAMP, "--override", "R1",
                   "--reason", "because"])
    expect("an override without a user: source is refused", rc, out,
           contains=("reason and a user: source", "never silent"))

    rc, out = run([S / "select.py", "--campaign", CAMP, "--override", "R1",
                   "--reason", "because", "--source", "vega:default"])
    expect("an override whose source is not a user: source is refused", rc, out,
           contains=("must be a user: source",))

    rc, out = run([S / "select.py", "--campaign", CAMP, "--override", "R99",
                   "--reason", "x", "--source", "user:turn3"])
    expect("an override naming a run that is not in the ledger is refused", rc, out,
           contains=("is not a run in this ledger",))

    with tempfile.TemporaryDirectory() as td:
        def mutate(exp):
            for row in exp["runs"]:
                row["diagnostics"] = ["D1"]
        dst = campaign_copy(td, mutate_experiments=mutate)
        rc, out = run([S / "select.py", "--campaign", dst])
        expect("a campaign whose every run diverged has no selection by the rule", rc, out,
               contains=("no run is eligible", "--override"))


# --------------------------------------------------------------------------- apply

def apply_negatives():
    section("apply_training_layer.py — what may not be published",
            "vega-train/apply_training_layer.py")

    layer = json.loads((CAMP / "training-layer.json").read_text(encoding="utf-8"))
    notebook = CAMP / "train.ipynb"

    def apply_with(mutate_doc=None, mutate_experiments=None, notebook_path=notebook,
                   name="", contains=(), code=2):
        with tempfile.TemporaryDirectory() as td:
            dst = campaign_copy(td, mutate_experiments=mutate_experiments)
            doc = copy.deepcopy(layer)
            doc["from_campaign"] = str(dst)
            if mutate_doc:
                mutate_doc(doc)
            data = Path(td) / "training-layer.json"
            data.write_text(json.dumps(doc), encoding="utf-8")
            args = [S / "apply_training_layer.py", MODEL, "--data", data,
                    "--out", Path(td) / "out.html"]
            if notebook_path:
                args += ["--notebook", notebook_path]
            rc, out = run(args)
        expect(name, rc, out, code=code, contains=contains)

    def make_ineligible(exp):
        exp["selection"]["run"] = "H1-4"
        exp["selection"]["override"] = None
        for row in exp["runs"]:
            if row["id"] == "H1-4":
                row["eligible"] = False
                row["ineligible_because"] = ["a pruned trial"]

    apply_with(mutate_experiments=make_ineligible,
               name="an ineligible selection with no override is refused",
               contains=("ineligible by the rule", "select.py --override", "in the open"))

    def flip_verdict(exp):
        row = next(r for r in exp["runs"] if r["id"] == "R0")
        row["expect_held"] = not row["expect_held"] if row["expect_held"] is not None else True

    apply_with(mutate_experiments=flip_verdict,
               name="H1e  a ledger whose expect_held was edited is refused",
               contains=("'R0' says expect_held", "its expect_check judges", "never edited"))

    def drop_check(exp):
        next(r for r in exp["runs"] if r["id"] == "R1").pop("expect_check", None)

    apply_with(mutate_experiments=drop_check,
               name="H1f  a ledger row with no expect_check is refused",
               contains=("'R1' carries no expect_check",))

    apply_with(mutate_doc=lambda d: d.__setitem__("notebook", None), notebook_path=None,
               name="a layer with no notebook is refused",
               contains=("notebook.path is required", "judgement → notebook → apply"))

    apply_with(mutate_doc=lambda d: d["readings"].pop("selection"),
               name="a layer with no selection reading is refused",
               contains=("readings.selection", "a verdict without its reasoning"))

    apply_with(mutate_doc=lambda d: d["forks"][0].__setitem__("answer", "both"),
               name="a fork carrying both an answer and an `or` is refused",
               contains=("both `answer` and `or`",))

    def drop_history(doc):
        doc["histories"] = {k: v for k, v in (doc.get("histories") or {}).items()
                            if k != "R0"}
    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        doc = copy.deepcopy(layer)
        doc["from_campaign"] = str(dst)
        # splice first, then remove one history, so the check sees a ledger row without curves
        doc["histories"] = {"R1": {"path": "x", "epochs": [1]}}
        data = Path(td) / "training-layer.json"
        data.write_text(json.dumps(doc), encoding="utf-8")
        rc, out = run([S / "apply_training_layer.py", MODEL, "--data", data,
                       "--notebook", notebook, "--out", Path(td) / "out.html"])
        expect("a ledger run with no history is refused", rc, out,
               contains=("histories has no entry", "every run in the ledger carries its curves"))

    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td)
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                                    "text": ["VEGA-REPLAY: mismatch R0 problems=4\n"]}]
        broken = Path(td) / "train.ipynb"
        broken.write_text(json.dumps(nb), encoding="utf-8")
        doc = copy.deepcopy(layer)
        doc["from_campaign"] = str(dst)
        data = Path(td) / "training-layer.json"
        data.write_text(json.dumps(doc), encoding="utf-8")
        rc, out = run([S / "apply_training_layer.py", MODEL, "--data", data,
                       "--notebook", broken, "--out", Path(td) / "out.html"])
        expect("a layer whose notebook replay failed is refused", rc, out,
               contains=("replay did not match", "determinism is a claim"))


# --------------------------------------------------------------------------- nb_emit

def notebook_negatives():
    section("nb_emit.py --stage train — the notebook replays the selected run",
            "vega-forge/nb_emit.py")
    with tempfile.TemporaryDirectory() as td:
        dst = campaign_copy(td, mutate_experiments=lambda e: e.__setitem__("selection", {}))
        rc, out = run([FORGE / "nb_emit.py", "--stage", "train", "--from", dst,
                       "--out", Path(td) / "train.ipynb"])
        expect("a campaign with no selection cannot have a train notebook", rc, out,
               contains=("records no selection", "select.py"))


def main():
    if not (CAMP / "experiments.json").is_file():
        print(f"ERROR: no campaign at {CAMP}; run acceptance.sh first")
        return 2
    plan_negatives()
    trainer_negatives()
    hpo_negatives()
    select_negatives()
    apply_negatives()
    notebook_negatives()
    return CASES.finish(ARGS.json)

if __name__ == "__main__":
    sys.exit(main())
