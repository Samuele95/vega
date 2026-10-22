#!/usr/bin/env python3
"""
run_matrix.py — the lane scheduler and tier driver of the verification program (D6 §2).

    run_matrix.py lanes [--lanes L01,L02 | --tier pr|release|nightly] [--changed RANGE]
                        [--from STAGE] [--to STAGE] [-j N] [--strict] [--matrix DIR]
    run_matrix.py static | unit | negatives | report  [--matrix DIR] [--strict]  (unit: [-j N])
    run_matrix.py pr | release | nightly               [--matrix DIR] [--changed RANGE]
    run_matrix.py validate [--skills DIR]              the registry and flakes.json (T0)
    run_matrix.py observed FILE...                     validate observed.json documents
    run_matrix.py list [--tier T]

`matrix.sh <command> …` is the same with the environment settled (interpreter, browser).

Every run works in a matrix directory outside the repository (default `$VEGA_MATRIX_DIR`,
else `~/.cache/vega-matrix`): `skills/` is rebuilt as a copy of the repository plus
domain-forge (lanekit.skilltree), each lane works in `lanes/<id>/` (work dir, logs, stamps,
results, observed.json), and the run writes `summary.json`, `timing.json` and `timing.md`.

A lane or tier component that needs a file some later work package provides is `absent`,
named with that package. A non-strict run reports it and exits 0 when nothing failed; a strict
run — and the composite tiers pr, release and nightly are strict — fails on it, because a gate
that did not run proves nothing.

Exit codes: 0 green; 1 a lane or component failed, a strict run met an absent part, or a tier
ran over its cap by more than 50 %; 2 usage, or `--from` refused a stale stamp.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lanekit import contracts, drift, observe, registry, skilltree  # noqa: E402
from lanekit.context import ABSENT, MatrixEnv, RunOptions  # noqa: E402
from lanekit.lane import LaneRun  # noqa: E402
from lanekit.runner import Flakes, lane_env  # noqa: E402
from lanekit.timing import flake_warnings, write_timing  # noqa: E402
from lanekit.stages import stage_plan  # noqa: E402
from lanekit.tiers import (ComponentOutcome, TierRun, negative_components,  # noqa: E402
                           report_components, run_components, static_components, unit_components)

# D6 §2.1's tier budgets, the ones a run measured over set to 1.5 x the measured time rounded up to
# 30 s (T12, risk 11; runs/final/T12): unit 372 s and lanes:release 816 s (-j 6) under a machine load
# of about 44 on 20 cores, lanes:pr from L01's 456 s (the longest pr lane), the composites their parts'
# sum; a tier still over its cap by more than 50 % fails the run
TIER_CAPS_S = {"static": 60, "unit": 570, "lanes:pr": 690, "lanes:release": 1230,
               "lanes:nightly": 1500, "negatives": 360, "report": 10, "pr": 1440,
               "release": 1990, "nightly": 2760}
COMPOSITES = {"pr": ("static", "unit", "lanes", "negatives", "report"),
              "release": ("static", "unit", "lanes", "negatives", "report"),
              "nightly": ("static", "unit", "lanes", "negatives", "report")}


def say(text: str) -> None:
    print(text, flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- the matrix dir
@dataclass
class Matrix:
    """One run's working directory, its skills tree and its shared environment."""
    root: Path
    repo: Path
    python: str
    device: str
    tier: str | None
    menv: MatrixEnv | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def open(cls, args: argparse.Namespace, tier: str | None) -> "Matrix":
        repo = skilltree.repo_root(HERE)
        root = Path(args.matrix or os.environ.get("VEGA_MATRIX_DIR")
                    or Path.home() / ".cache" / "vega-matrix").expanduser().resolve()
        if root == repo or repo in root.parents:
            raise SystemExit(f"ERROR: the matrix dir {root} is inside the repository; the skills "
                             "tree must stay pristine — pass --matrix outside it")
        root.mkdir(parents=True, exist_ok=True)
        return cls(root, repo, sys.executable, getattr(args, "device", "cpu"), tier)

    def lock(self) -> Any:
        handle = open(self.root / ".lock", "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"ERROR: another matrix run holds {self.root}/.lock")
        return handle

    def prepare(self) -> MatrixEnv:
        """Rebuild the skills tree, pin domain-forge, and settle the lane environment."""
        if self.menv is not None:
            return self.menv
        tree = skilltree.build(self.root, self.repo)
        self.warnings += skilltree.check_pins(self.root, tree)
        for warning in self.warnings:
            say(warning)
        lanes_file = tree.root / contracts.TESTS_DIR / "lanes.json"
        defaults = registry.load_document(lanes_file)["defaults"]
        # the tree is a copy, not a checkout: a test that reads git history (test_emit_layout's
        # c849f8b reference) finds the repository it was copied from here; VEGA_LANE_REGISTRY is
        # how vega-show's V6 knows a `registry:<lane>` role basis is the harness's (a production
        # run never sets it, and V6 refuses the basis there)
        env = lane_env(os.environ, self.python, int(defaults["threads"]), self.device,
                       {"VEGA_SOURCE_REPO": str(self.repo), "VEGA_LANE_REGISTRY": str(lanes_file)})
        self.menv = MatrixEnv(self.root, tree.root, self.repo, self.python, lanes_file, defaults,
                              Flakes.load(tree.root / contracts.LANEKIT_DIR / "flakes.json"), env,
                              skilltree.git_state(self.repo), self.tier, tree.pins)
        say(f"OK: skills tree {tree.root} — {tree.files} file(s), domain-forge from {tree.domain_forge}")
        return self.menv


# --------------------------------------------------------------------------- lanes
@dataclass
class LanesOutcome:
    docs: list[dict]
    refused: list[str]
    seconds: float

    def failed(self, strict: bool) -> list[str]:
        bad = [d["lane"] for d in self.docs if d["verdict"] == "fail"]
        if strict:
            bad += [d["lane"] for d in self.docs if d["verdict"] == "partial"]
        return bad + self.refused


def select_lanes(args: argparse.Namespace, repo: Path, tier: str | None) -> list[registry.Lane]:
    lanes = registry.load_lanes(repo / contracts.TESTS_DIR / "lanes.json")
    ids = [i for i in (args.lanes or "").split(",") if i.strip()]
    chosen = registry.lanes_for(lanes, ids or None, None if ids else (args.tier or tier or "release"))
    if getattr(args, "changed", None):
        changed = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", args.changed],
                                 capture_output=True, text=True, check=True).stdout.split()
        routed = registry.affected_lanes(lanes, changed, repo, args.changed)
        for lane_id, paths in routed.items():
            if lanes[lane_id] not in chosen:
                say(f"OK: {lane_id} routed by --changed ({', '.join(paths[:3])})")
                chosen.append(lanes[lane_id])
    return chosen


def run_lanes(matrix: Matrix, lanes: Sequence[registry.Lane], options: RunOptions,
              jobs: int) -> LanesOutcome:
    menv = matrix.prepare()
    started, docs, refused = time.monotonic(), [], []
    say(f"OK: {len(lanes)} lane(s): {', '.join(l.id for l in lanes)} (-j {jobs})")
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        # the LaneRun is built in the worker: its budget clock starts when the lane runs
        futures = {pool.submit(_run_lane, lane, menv, options): lane for lane in lanes}
        for future in as_completed(futures):
            lane = futures[future]
            try:
                doc = future.result()
            except KeyError as exc:
                say(f"ERROR: {lane.id}: {exc}")
                refused.append(lane.id)
                continue
            docs.append(doc)
            report_lane(doc)
            if doc["verdict"] == "refused":
                refused.append(lane.id)
    docs.sort(key=lambda d: d["lane"])
    return LanesOutcome(docs, refused, time.monotonic() - started)


def _run_lane(lane: registry.Lane, menv: MatrixEnv, options: RunOptions) -> dict:
    return LaneRun(lane, menv, options).run()


def report_lane(doc: Mapping[str, Any]) -> None:
    lane, verdict = doc["lane"], doc["verdict"]
    passed = [s["id"] for s in doc["stages"] if s["status"] in ("pass", "stamped")]
    seconds = sum(s["seconds"] for s in doc["stages"])
    if verdict == "pass":
        say(f"OK: {lane} pass — {len(passed)} stage(s) in {seconds:.0f} s")
    elif verdict == "partial":
        absent = next(s for s in doc["stages"] if s["status"] == ABSENT)
        say(f"WARN: {lane} partial — {', '.join(passed)} passed ({seconds:.0f} s); stopped before "
            f"{absent['id']}: {absent['detail']}")
    elif verdict in ("skip", "refused"):
        say(f"{'WARN' if verdict == 'skip' else 'ERROR'}: {lane} {verdict} — {doc.get('reason')}")
    else:
        bad = next(s for s in doc["stages"] if s["status"] in ("fail", "timeout"))
        say(f"ERROR: {lane} fail at {bad['id']} ({bad['status']}) — {bad['detail'][:400]}"
            f"  [log: lanes/{lane}/{bad['log']}]")
    for flake in doc.get("flakes") or []:
        say(f"WARN: {lane} flake {flake['id']} retried in {flake['stage']} "
            f"(second attempt {flake['second_attempt']}, via {flake.get('via', 'runner')})")
    for what in not_run(doc):
        say(f"WARN: {lane} check not run: {what}")


def not_run(doc: Mapping[str, Any]) -> list[str]:
    """The checks a lane's stages said they did not run (observed.json `stages[].not_run`)."""
    return [f"{s['id']}: {what}" for s in doc["stages"] for what in s.get("not_run") or []]


# --------------------------------------------------------------------------- commands
def cmd_validate(args: argparse.Namespace) -> int:
    skills = Path(args.skills).resolve() if args.skills else skilltree.repo_root(HERE)
    lanes_file = Path(args.lanes_file) if args.lanes_file else HERE / "lanes.json"
    flakes = json.loads((HERE / "lanekit" / "flakes.json").read_text(encoding="utf-8"))
    report = registry.validate(registry.load_document(lanes_file), registry.Catalogs.load(skills), flakes)
    report.errors.extend(f"contracts.py drifted from a catalog: {d}" for d in drift.drift(skills))
    for line in report.warnings:
        say(f"WARN: {line}")
    for line in report.errors:
        say(f"ERROR: {line}")
    n = len(registry.load_lanes(lanes_file))
    if report.ok:
        say(f"OK: {lanes_file.name} validates — {n} lane(s) against the catalogs of {skills}; "
            "flakes.json holds; contracts.py agrees with the catalogs it copies")
    return 0 if report.ok else 1


def cmd_observed(args: argparse.Namespace) -> int:
    bad = 0
    for name in args.files:
        problems = observe.validate(json.loads(Path(name).read_text(encoding="utf-8")))
        bad += bool(problems)
        say(f"OK: {name} validates as {contracts.LANE_OBSERVED}" if not problems
            else f"ERROR: {name}: " + "; ".join(problems[:5]))
    return 1 if bad else 0


def cmd_list(args: argparse.Namespace) -> int:
    lanes = registry.load_lanes(HERE / "lanes.json")
    for lane in registry.lanes_for(lanes, None, args.tier):
        say(f"{lane.id}  {lane.driver:<10} {lane.family:<16} {lane.task:<24} {','.join(lane.tiers):<20} "
            f"{' > '.join(stage_plan(lane))}")
    return 0


def cmd_lanes(args: argparse.Namespace, matrix: Matrix | None = None, tier: str | None = None) -> tuple[int, LanesOutcome | None]:
    matrix = matrix or Matrix.open(args, args.tier)
    try:
        lanes = select_lanes(args, matrix.repo, tier)
    except KeyError as exc:
        say(f"ERROR: {exc}")
        return 2, None
    options = RunOptions(args.from_stage, args.to_stage, args.strict)
    outcome = run_lanes(matrix, lanes, options, args.jobs or int(matrix.prepare().defaults["jobs"]))
    all_lanes = registry.load_lanes(matrix.repo / contracts.TESTS_DIR / "lanes.json")
    warn_factor = float(matrix.menv.defaults["budget_warn_factor"])
    warns = (write_timing(matrix.root, matrix.tier, outcome.docs, all_lanes, warn_factor)
             + flake_warnings(outcome.docs, matrix.menv.flakes))
    for warn in warns:
        say(warn)
    failed = outcome.failed(args.strict)
    code = 2 if outcome.refused else (1 if failed else 0)
    write_summary(matrix, "lanes", {"lanes": {d["lane"]: {"verdict": d["verdict"], "reason": d.get("reason"),
                                                          "not_run": not_run(d)}
                                              for d in outcome.docs},
                                    "flakes": [f for d in outcome.docs for f in d.get("flakes") or []],
                                    "warnings": matrix.warnings + warns, "failed": failed}, code)
    if code == 0 and not any(d["verdict"] == "fail" for d in outcome.docs):
        skilltree.record_green(matrix.root)          # the pins the next run compares with
    return code, outcome


def write_summary(matrix: Matrix, command: str, body: Mapping[str, Any], code: int) -> None:
    doc = {"schema": contracts.MATRIX_SUMMARY, "command": command, "tier": matrix.tier,
           "finished_at": _now(), "git": skilltree.git_state(matrix.repo), **body, "exit": code}
    (matrix.root / "summary.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def cmd_tier(args: argparse.Namespace, name: str) -> int:
    """static | unit | negatives | report, alone."""
    matrix = Matrix.open(args, getattr(args, "tier", None))
    with matrix.lock():
        outcome = _tier_components(matrix, name, getattr(args, "jobs", None))
        return _tier_exit(name, outcome, args.strict, matrix)


def _tier_components(matrix: Matrix, name: str, jobs: int | None = None) -> ComponentOutcome:
    menv = matrix.prepare()
    if name == "static":
        comps = static_components(menv.skills)
    elif name == "unit":
        comps = unit_components(menv.skills)
    elif name == "negatives":
        comps = negative_components(menv.skills, matrix.root)
        if not (matrix.root / "lanes" / "L01" / "work" / "bundle").is_dir():
            say("ERROR: negatives: the existing suites run on L01's work dir — run `matrix.sh lanes --lanes L01` first")
            return ComponentOutcome(failed=["negatives: no L01 work dir"])
    else:
        comps = report_components(matrix.tier, menv.defaults["adaptation_pairs"])
    fields = {"python": menv.python, "skills": str(menv.skills), "tests": str(menv.skills / contracts.TESTS_DIR),
              "matrix": str(matrix.root), "work": str(matrix.root / "lanes" / "L01" / "work"),
              "tier": matrix.tier or "release"}
    (matrix.root / "negatives").mkdir(exist_ok=True)
    run = TierRun(menv.skills, menv.env, fields, matrix.root / "tiers" / name)
    if name == "unit":
        return run_side_by_side(run, name, comps, jobs or unit_jobs(menv.defaults))
    return run_components(run, name, comps, say)


def unit_jobs(defaults: Mapping[str, Any]) -> int:
    """How many unit suites run at once: the cores over the intra-op threads each suite process
    is started with (lanes.json `defaults.threads`, OMP/MKL_NUM_THREADS), so the tier fills the
    machine without oversubscribing it."""
    return max(1, (os.cpu_count() or 1) // max(1, int(defaults["threads"])))


def run_side_by_side(run: TierRun, name: str, comps: Sequence[Any], jobs: int) -> ComponentOutcome:
    """`tiers.run_components` with up to `jobs` components running at once (PLAN-AMENDMENTS A25):
    the same absent rule, one log per component under `run.log_dir`, a component passes when it
    exits 0; the outcome lists keep the components' order, each line is said as its component
    ends, and `seconds` is the tier's wall clock — the time its cap applies to."""
    out, started = ComponentOutcome(), time.monotonic()
    run.log_dir.mkdir(parents=True, exist_ok=True)
    ready = []
    for comp in comps:
        missing = [n for n in comp.needs if not (run.skills / n).exists()]
        if missing:
            out.absent.append(f"{comp.id} (needs {missing[0]}, {comp.provider})")
            say(f"WARN: {name}: {comp.id} absent — needs {missing[0]} ({comp.provider})")
        else:
            ready.append(comp)
    say(f"OK: {name}: {len(ready)} component(s), {min(jobs, len(ready)) or 1} at a time")
    lock = threading.Lock()

    def one(comp: Any) -> int:
        log = run.log_dir / (re.sub(r"[^\w.-]+", "_", comp.id) + ".log")
        began = time.monotonic()
        with open(log, "w", encoding="utf-8") as fh:
            code = subprocess.run(comp.resolve(run.fields), cwd=run.skills, env=dict(run.env),
                                  stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT).returncode
        seconds = time.monotonic() - began
        with lock:
            say(f"{'OK' if code == 0 else 'ERROR'}: {name}: {comp.id} ({seconds:.0f} s)"
                + ("" if code == 0 else f" exited {code} [{log}]"))
        return code

    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(ready)))) as pool:
        codes = list(pool.map(one, ready))
    for comp, code in zip(ready, codes):
        (out.passed if code == 0 else out.failed).append(comp.id)
    out.seconds = time.monotonic() - started
    return out


def _tier_exit(name: str, outcome: ComponentOutcome, strict: bool, matrix: Matrix) -> int:
    over = _over_cap(name, outcome.seconds)
    code = 1 if outcome.failed or (strict and outcome.absent) or over else 0
    write_summary(matrix, name, {"passed": outcome.passed, "failed": outcome.failed,
                                 "absent": outcome.absent, "seconds": round(outcome.seconds, 1)}, code)
    say(f"{'OK' if code == 0 else 'ERROR'}: {name}: {len(outcome.passed)} passed, {len(outcome.failed)} failed, "
        f"{len(outcome.absent)} absent in {outcome.seconds:.0f} s")
    return code


def _over_cap(name: str, seconds: float) -> bool:
    cap = TIER_CAPS_S.get(name)
    if cap and seconds > 1.5 * cap:
        say(f"ERROR: tier {name} took {seconds:.0f} s, over its cap {cap} s by more than 50 %")
        return True
    return False


def cmd_composite(args: argparse.Namespace, name: str) -> int:
    """pr | release | nightly: static, unit, the tier's lanes, negatives, report — strict."""
    args.strict, args.tier = True, name
    matrix = Matrix.open(args, name)
    started, codes = time.monotonic(), []
    with matrix.lock():
        for part in COMPOSITES[name]:
            if part == "lanes":
                code, _ = cmd_lanes(args, matrix, name)
                codes.append(code)
                continue
            codes.append(_tier_exit(part, _tier_components(matrix, part), True, matrix))
    over = _over_cap(name, time.monotonic() - started)
    code = 2 if 2 in codes else (1 if any(codes) or over else 0)
    say(f"{'OK' if code == 0 else 'ERROR'}: {name} {'green' if code == 0 else 'red'} "
        f"({time.monotonic() - started:.0f} s)")
    return code


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("lanes", "static", "unit", "negatives", "report", "pr", "release", "nightly"):
        p = sub.add_parser(name)
        p.add_argument("--matrix", help="the matrix dir (default $VEGA_MATRIX_DIR or ~/.cache/vega-matrix)")
        p.add_argument("--strict", action="store_true", help="an absent stage or component fails the run")
        p.add_argument("--tier", choices=registry.TIERS, default=None)
        if name == "unit":
            p.add_argument("-j", "--jobs", type=int, default=None,
                           help="suites at once (default: the cores over defaults.threads)")
        if name in ("lanes", "pr", "release", "nightly"):
            p.add_argument("--lanes", default="", help="comma-separated lane ids")
            p.add_argument("--changed", help="a git range; the lanes its changed paths affect are added")
            p.add_argument("--from", dest="from_stage", choices=contracts.STAGES + (contracts.GOLDEN_STAGE,))
            p.add_argument("--to", dest="to_stage", choices=contracts.STAGES + (contracts.GOLDEN_STAGE,))
            p.add_argument("-j", "--jobs", type=int, default=None)
            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p = sub.add_parser("validate")
    p.add_argument("--skills", help="the skills root whose catalogs the registry is checked against")
    p.add_argument("--lanes-file")
    p = sub.add_parser("observed")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("list")
    p.add_argument("--tier", choices=registry.TIERS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "observed":
        return cmd_observed(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "lanes":
        matrix = Matrix.open(args, args.tier)
        with matrix.lock():
            return cmd_lanes(args, matrix, args.tier)[0]
    if args.command in COMPOSITES:
        return cmd_composite(args, args.command)
    return cmd_tier(args, args.command)


if __name__ == "__main__":
    sys.exit(main())
