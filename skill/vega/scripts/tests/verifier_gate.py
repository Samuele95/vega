#!/usr/bin/env python3
"""
verifier_gate.py — every skill's static verifier said SHIP on the prompt as it stands (D6 §15.7).

    python3 verifier_gate.py [--skill S ...] [--root DIR] [--runs DIR] [--release] [--json OUT]
    python3 verifier_gate.py record --skill S --session ID (--response FILE | --verdict V
                             --blocking N --non-blocking N [--inconclusive N]) [--audited PATH ...]

For each skill of the package (vega-forge, vega-train, vega-eval, vega-pack, vega-show, vega,
dataset-forge, data-lens, dataset-shaper; --skill narrows):
  - <skill>/verifiers/static-verifier.md exists and holds the package checks: for a stage,
    "Published before summarised", "Stage brief" and "Orchestrator named" (U1-U3; vega-eval
    numbers them U4-U6); for the orchestrator, "Published and briefed";
  - verifier-runs/<skill>.json (`vega/verifier-run@1`) records the last LLM run of that
    verifier: verdict SHIP, no blocking finding, and an `audited` map {path: sha256:...} that
    covers SKILL.md and every file the verifier's Input line names, each digest equal to the
    file's today (a digest that differs is a prompt changed after the verifier read it).
--release also requires release/rehearsal.json (the T5 rehearsal, `vega/rehearsal@1`): a
session and every check passed. One form of the run's own check is accepted besides `pass`, and
only with the evidence spelled out (the user's decision of 2026-10-12 on the rehearsal of record):
the `T5-run` check may say `pass-with-fixed-defects` when
  - it says `completed: true` (every stage through the goal ran unattended) and names its
    defects (`defects: [id, ...]`);
  - every OTHER check (the product checks) says `pass`;
  - the record's top-level `defects_fixed` lists exactly those ids, each as
    {id, summary, fix: <commit>, tests: [<path relative to the root>, ...]}, and for each one
    the fix commit is an ancestor of the released head (git HEAD of --root), every test file
    exists in the tree, and the fix commit itself added or changed at least one of them.
Anything short of that is a finding (a root that is not a git checkout cannot prove it and
fails); an accepted record prints a NOTE per fixed defect, never a silent pass.

`record` writes verifier-runs/<skill>.json for a run just made: the verdict from the
verifier's first JSON line (--response) or the flags, the digests of SKILL.md, the verifier
itself and the Input files (or of --audited, relative to the skill), and the time.

Exit 0: every skill passes (record: written). 1: a skill does not. 2: a usage error or an
unreadable record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = HERE / "verifier-runs"
REHEARSAL = "release/rehearsal.json"
RUN_CHECK = "T5-run"                          # the run's own check; the others are product checks
FIXED = "pass-with-fixed-defects"             # its one verdict besides pass (see the docstring)
SKILLS = ("vega-forge", "vega-train", "vega-eval", "vega-pack", "vega-show", "vega", "dataset-forge", "data-lens",
          "dataset-shaper")
STAGE_CHECKS = ("Published before summarised", "Stage brief", "Orchestrator named")
ORCHESTRATOR_CHECKS = ("Published and briefed",)
VERIFIER = "verifiers/static-verifier.md"
VERDICT_LINE = re.compile(r"\{[^{}]*\"verdict\"[^{}]*\}")


class GateError(Exception):
    """A usage error or an unreadable record (exit 2)."""


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def input_files(verifier_text: str, skill: str) -> list[str]:
    """The files the verifier's Input line names, relative to the skill (`vega-eval/SKILL.md` ->
    `SKILL.md`); files of another skill keep their repository path, prefixed `../`."""
    m = re.search(r"\bInput:(.*?)(?:\bOutput:|$)", verifier_text, re.M)
    out = []
    for path in re.findall(r"`([\w./-]+\.(?:md|json|py|js|css))`", m.group(1) if m else ""):
        out.append(path[len(skill) + 1:] if path.startswith(skill + "/") else f"../{path}")
    return sorted(set(out))


# ---- the checks of one skill -----------------------------------------------------------------------------
def verifier_findings(skill_dir: Path, skill: str) -> list[str]:
    path = skill_dir / VERIFIER
    if not path.is_file():
        return [f"{skill}: no {VERIFIER}"]
    text = path.read_text(encoding="utf-8")
    wanted = ORCHESTRATOR_CHECKS if skill == "vega" else STAGE_CHECKS
    return [f"{skill}: {VERIFIER} has no \"{title}\" check"
            for title in wanted if not re.search(rf"^\*\*[A-Z]\d+\s+{re.escape(title)}", text, re.M | re.I)]


def load_record(runs: Path, skill: str) -> dict | None:
    path = runs / f"{skill}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc


def record_findings(record: dict | None, skill_dir: Path, skill: str) -> list[str]:
    if record is None:
        return [f"{skill}: no verifier run recorded (verifier-runs/{skill}.json)"]
    out = []
    if record.get("schema") != "vega/verifier-run@1" or record.get("skill") != skill:
        out.append(f"{skill}: the record is not a vega/verifier-run@1 run of {skill}")
    if record.get("verdict") != "SHIP" or record.get("blocking", 1) != 0:
        out.append(f"{skill}: the last verifier run says {record.get('verdict')} with "
                   f"{record.get('blocking')} blocking finding(s), not SHIP")
    audited = record.get("audited") or {}
    verifier = skill_dir / VERIFIER
    need = {"SKILL.md", *(input_files(verifier.read_text(encoding="utf-8"), skill) if verifier.is_file() else [])}
    out += [f"{skill}: the verifier run did not audit {p}" for p in sorted(need - set(audited))]
    for rel, want in sorted(audited.items()):
        path = skill_dir / rel
        if not path.is_file():
            out.append(f"{skill}: {rel}, audited by the verifier run, no longer exists")
        elif digest(path) != want:
            out.append(f"{skill}: {rel} changed after the verifier read it (recorded {want[:19]}…, "
                       f"now {digest(path)[:19]}…): re-run the verifier")
    return out


def load_rehearsal(root: Path) -> dict | None:
    path = root / REHEARSAL
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def fixed_defect_findings(root: Path, doc: dict, check: dict) -> list[str]:
    """Why the run check's `pass-with-fixed-defects` does not hold ([] when it does)."""
    out = []
    if check.get("completed") is not True:
        out.append(f"release: rehearsal check {RUN_CHECK} says {FIXED} but not completed: true")
    named = check.get("defects") or []
    if not named:
        out.append(f"release: rehearsal check {RUN_CHECK} says {FIXED} and names no defect")
    others = [c for c in doc.get("checks") or [] if c is not check]
    if not others:
        out.append(f"release: rehearsal check {RUN_CHECK} says {FIXED} and the record holds no product check")
    defects = doc.get("defects_fixed")
    if not isinstance(defects, list) or not defects:
        return out + [f"release: rehearsal check {RUN_CHECK} says {FIXED} and the record has no defects_fixed"]
    by_id = {d.get("id"): d for d in defects if isinstance(d, dict)}
    out += [f"release: defect {i}, named by {RUN_CHECK}, is not in defects_fixed" for i in named if i not in by_id]
    out += [f"release: defects_fixed lists {i}, which {RUN_CHECK} does not name" for i in by_id if i not in named]
    if len(by_id) != len(defects):
        out.append("release: defects_fixed holds an entry that is not an object with an id, or a repeated id")
    top = _git(root, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
        return out + [f"release: {root} is not a git checkout, so no fix in defects_fixed can be proved on the "
                      f"released head"]
    for i, d in by_id.items():
        fix, tests = d.get("fix"), d.get("tests") or []
        if not d.get("summary"):
            out.append(f"release: defect {i} has no summary")
        if not tests:
            out.append(f"release: defect {i} names no test (a fixed defect is fixed with a test)")
        out += [f"release: defect {i}'s test {t} is not in the tree" for t in tests if not (root / t).is_file()]
        sha = _git(root, "rev-parse", "--verify", f"{fix}^{{commit}}") if fix else None
        if sha is None or sha.returncode != 0:
            out.append(f"release: defect {i}'s fix {fix!r} is not a commit of this repository")
            continue
        if _git(root, "merge-base", "--is-ancestor", sha.stdout.strip(), "HEAD").returncode != 0:
            out.append(f"release: defect {i}'s fix {fix} is not on the released head")
        changed = set(_git(root, "show", "--format=", "--name-only", "--no-renames", sha.stdout.strip())
                      .stdout.split())
        if tests and not changed & set(tests):
            out.append(f"release: defect {i}'s fix {fix} adds or changes none of its tests ({', '.join(tests)})")
    return out


def rehearsal_findings(root: Path) -> list[str]:
    doc = load_rehearsal(root)
    if doc is None:
        return [f"release: no {REHEARSAL} (the T5 rehearsal is a release gate)"]
    checks = doc.get("checks") or []
    out = [] if doc.get("session") else [f"release: {REHEARSAL} names no session"]
    out += [] if checks else [f"release: {REHEARSAL} records no check"]
    for c in checks:
        if c.get("verdict") == "pass":
            continue
        if c.get("verdict") == FIXED and c.get("id") == RUN_CHECK:
            out += fixed_defect_findings(root, doc, c)
        elif c.get("verdict") == FIXED:
            out.append(f"release: rehearsal check {c.get('id')} is {FIXED}: only the run's own check "
                       f"({RUN_CHECK}) may be, every product check must pass")
        else:
            out.append(f"release: rehearsal check {c.get('id')} is {c.get('verdict')}")
    if doc.get("defects_fixed") and not any(c.get("id") == RUN_CHECK and c.get("verdict") == FIXED for c in checks):
        out.append(f"release: defects_fixed is recorded but no {RUN_CHECK} check says {FIXED}")
    return out


def rehearsal_notes(root: Path) -> list[str]:
    """The NOTE lines of an accepted record: each defect the rehearsal of record met and its fix."""
    doc = load_rehearsal(root) or {}
    return [f"release: {RUN_CHECK} passed with a fixed defect: {d.get('id')} fixed in {d.get('fix')}, tested by "
            f"{', '.join(d.get('tests') or [])}" for d in doc.get("defects_fixed") or [] if isinstance(d, dict)]


def gate(skills: list[str], root: Path, runs: Path, release: bool) -> dict[str, list[str]]:
    """{skill: findings} for every skill (and `release` with --release)."""
    out = {}
    for skill in skills:
        skill_dir = root / skill
        if not (skill_dir / "SKILL.md").is_file():
            out[skill] = [f"{skill}: no SKILL.md in {root}"]
            continue
        out[skill] = verifier_findings(skill_dir, skill) + record_findings(load_record(runs, skill), skill_dir, skill)
    if release:
        out["release"] = rehearsal_findings(root)
    return out


# ---- record -----------------------------------------------------------------------------------------------
def verdict_of(a: argparse.Namespace) -> dict:
    if a.response:
        text = Path(a.response).read_text(encoding="utf-8")
        m = VERDICT_LINE.search(text)
        if not m:
            raise GateError(f"{a.response}: no verdict JSON line (the verifier begins with one)")
        head = json.loads(m.group(0))
    else:
        head = {"verdict": a.verdict, "blocking": a.blocking, "non_blocking": a.non_blocking,
                "inconclusive": a.inconclusive}
    if head.get("verdict") not in ("SHIP", "REVISE", "BLOCK") or head.get("blocking") is None:
        raise GateError(f"a verdict needs SHIP|REVISE|BLOCK and a blocking count, got {head}")
    return {k: head.get(k, 0) for k in ("verdict", "blocking", "non_blocking", "inconclusive")}


def record(a: argparse.Namespace) -> Path:
    skill_dir = a.root / a.skill
    verifier = skill_dir / VERIFIER
    if not verifier.is_file():
        raise GateError(f"{a.skill}: no {VERIFIER} to record a run of")
    paths = a.audited or ["SKILL.md", VERIFIER, *input_files(verifier.read_text(encoding="utf-8"), a.skill)]
    missing = [p for p in paths if not (skill_dir / p).is_file()]
    if missing:
        raise GateError(f"{a.skill}: cannot audit {', '.join(missing)} (no such file)")
    doc = {"schema": "vega/verifier-run@1", "skill": a.skill, **verdict_of(a),
           "audited": {p: digest(skill_dir / p) for p in sorted(set(paths))},
           "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "session": a.session}
    out = a.runs / f"{a.skill}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return out


# ---- the command ------------------------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("mode", nargs="?", choices=("check", "record"), default="check")
    ap.add_argument("--skill", action="append", default=None, help="a skill of the package (repeatable)")
    ap.add_argument("--root", type=Path, default=REPO, help="the skills tree (default: this repository)")
    ap.add_argument("--runs", type=Path, default=RUNS, help="the verifier-runs directory")
    ap.add_argument("--release", action="store_true", help="also require release/rehearsal.json")
    ap.add_argument("--json", default=None, help="write {skill: findings} here ('-': stdout)")
    ap.add_argument("--session", default=None, help="record: the session that ran the verifier")
    ap.add_argument("--response", default=None, help="record: the verifier's response (its first JSON line)")
    ap.add_argument("--verdict", default=None, choices=("SHIP", "REVISE", "BLOCK"))
    ap.add_argument("--blocking", type=int, default=None)
    ap.add_argument("--non-blocking", type=int, default=0)
    ap.add_argument("--inconclusive", type=int, default=0)
    ap.add_argument("--audited", nargs="*", default=None, help="record: files audited, relative to the skill")
    return ap


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    try:
        if a.mode == "record":
            if not (a.skill and len(a.skill) == 1 and a.session):
                raise GateError("record needs exactly one --skill and a --session")
            a.skill = a.skill[0]
            print(f"OK: the {a.skill} verifier run recorded in {record(a)}")
            return 0
        unknown = sorted(set(a.skill or []) - set(SKILLS))
        if unknown:
            raise GateError(f"not a skill of the package: {', '.join(unknown)} (one of {', '.join(SKILLS)})")
        findings = gate(a.skill or list(SKILLS), a.root, a.runs, a.release)
        notes = rehearsal_notes(a.root) if a.release else []
    except GateError as exc:
        print(f"ERROR: {exc}")
        return 2
    return _report(findings, a.json, notes)


def _report(findings: dict[str, list[str]], json_out: str | None, notes: list[str] = ()) -> int:
    bad = [s for s, p in findings.items() if p]
    if json_out:
        doc = {"schema": "vega/verifier-gate@1", "findings": findings, **({"notes": list(notes)} if notes else {})}
        text = json.dumps(doc, indent=1) + "\n"
        if json_out == "-":
            sys.stdout.write(text)
            return 1 if bad else 0
        Path(json_out).write_text(text, encoding="utf-8")
    for problems in findings.values():
        for p in problems:
            print(f"FAIL: {p}")
    for note in notes:
        print(f"NOTE: {note}")
    if bad:
        print(f"FAIL: {len(bad)} of {len(findings)} skill(s) without a current SHIP: {', '.join(bad)}")
        return 1
    print(f"OK: {len(findings)} skill(s), each verifier's last run SHIP on the prompt as it stands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
