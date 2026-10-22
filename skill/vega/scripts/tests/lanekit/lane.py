"""
The lane driver: `LaneRun` runs one lane's stages in order and writes its observed.json.

    doc = LaneRun(lane, menv, RunOptions(from_stage=None, to_stage=None)).run()

- A lane whose `requires` names an unset variable is `skip` (G00 without `CIDS_ROOT`).
- A fresh run clears the lane's work dir, stamps, results and logs; `--from <stage>` keeps them,
  re-checks every earlier stamp (lanekit.stamps) and refuses a stale one before anything runs.
- A stage whose driving files are not in the tree yet is `absent` and names the work package
  that provides them; every later stage is then `blocked`, as after a failure.
- Each stage has a budget: `stage_timeout_factor x budget_s x share` (lanes.json defaults), and
  the lane as a whole `lane_kill_factor x budget_s`, counted from the moment `run()` starts (a
  lane queued behind others loses nothing); a command still running then is killed.
- The service the `serve` stage starts lives until the last service stage, then stops; the
  bundle must still be what pack pinned (serving writes nothing, D6 §6.9).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from . import contracts, observe, service as svc, skilltree
from .context import (ABSENT, BLOCKED, FAIL, NOT_REQUESTED, PASS, SERVICE_STAGES, SKIP, STAMPED,
                      TIMEOUT, LaneState, MatrixEnv, RunOptions, StageContext, StageFailure,
                      StageResult, pin_now)
from .registry import Lane
from .runner import CommandRunner, Deadline, FlakeRecorder
from .stamps import StampInputs, clear_stamps, resume_problems, write_stamp
from .stages import function_for, missing_provider, shares_for, stage_plan


class ResumeRefused(Exception):
    """`--from` cannot resume: the listed stamps are stale."""


class LaneRun:
    """Runs one lane's stages in order, stamps what passed, writes observed.json."""

    def __init__(self, lane: Lane, menv: MatrixEnv, options: RunOptions) -> None:
        self.lane, self.menv, self.options = lane, menv, options
        self.lane_dir = menv.matrix / "lanes" / lane.id
        self.work = self.lane_dir / "work"
        self.plan = stage_plan(lane)
        self.state = LaneState()
        self.recorder = FlakeRecorder()
        self.results: list[StageResult] = []
        self.carried_not_run: dict[str, tuple[str, ...]] = {}
        self.deadline: float | None = None
        self.started = _now()

    def run(self) -> dict:
        # the lane's clock starts when the lane starts running, not when it was queued: a lane
        # waiting for a free worker (-j) keeps its whole budget and its started_at is its own
        budget = float(self.lane.budget_s) * float(self.menv.defaults["lane_kill_factor"])
        self.deadline = time.monotonic() + budget
        self.started = _now()
        missing = [v for v in self.lane.get("requires") or [] if not os.environ.get(v)]
        if missing:
            return self._finish("skip", f"requires {', '.join(missing)} (unset)", skip=True)
        first = self._first_index()
        try:
            self._prepare(first)
        except ResumeRefused as exc:
            return self._finish("refused", str(exc), skip=True)
        halted: str | None = None
        for idx, stage in enumerate(self.plan):
            result = self._one(idx, stage, first, halted)
            self.results.append(result)
            if result.status not in (PASS, STAMPED, NOT_REQUESTED) and halted is None:
                halted = f"{stage} {result.status}"
            if stage in SERVICE_STAGES and (halted or stage == self._last_requested()):
                self._stop_service()
        self._stop_service()
        return self._finish(self._verdict(), halted)

    # ---- plan bounds
    def _first_index(self) -> int:
        start = self.options.from_stage
        if start is None:
            return 0
        if start not in self.plan:
            raise KeyError(f"{self.lane.id} has no stage {start!r} (its stages: {', '.join(self.plan)})")
        return self.plan.index(start)

    def _last_requested(self) -> str:
        to = self.options.to_stage
        return to if to in self.plan else self.plan[-1]

    def _requested(self, idx: int) -> bool:
        return idx <= self.plan.index(self._last_requested())

    # ---- preparation and resumption
    def _prepare(self, first: int) -> None:
        if first == 0:
            for sub in ("work", "stamps", "results", "log", "docs", "shots", "show"):
                shutil.rmtree(self.lane_dir / sub, ignore_errors=True)
            for name in ("fixture.json", "observed.json"):
                (self.lane_dir / name).unlink(missing_ok=True)
        else:
            problems = resume_problems(self.lane_dir, self.plan[:first], self.inputs,
                                       lambda outs: self._outputs_now(outs))
            if problems:
                raise ResumeRefused(f"--from {self.plan[first]} refused, stale stamps: " + "; ".join(problems))
            clear_stamps(self.lane_dir, self.plan[first:])
            self._reload_state()
        self.work.mkdir(parents=True, exist_ok=True)

    def _reload_state(self) -> None:
        fx = self.lane_dir / "fixture.json"
        if fx.is_file():
            self.state.fixture = json.loads(fx.read_text(encoding="utf-8"))
        pin = self.lane_dir / "results" / "bundle_pin.json"
        if pin.is_file():
            self.state.bundle_pin = json.loads(pin.read_text(encoding="utf-8"))
        seen = self.lane_dir / "observed.json"          # a stamped stage keeps the checks it
        if seen.is_file():                               # said it did not run when it passed
            stages = json.loads(seen.read_text(encoding="utf-8")).get("stages") or []
            self.carried_not_run = {st["id"]: tuple(st.get("not_run") or ()) for st in stages}

    def inputs(self, stage: str) -> StampInputs:
        lane_doc = json.dumps({"lane": self.lane.raw, "defaults": self.menv.defaults}, sort_keys=True)
        code = skilltree.digest_paths(self.menv.skills, contracts.STAGE_CODE.get(stage, ()))
        return StampInputs(_sha(lane_doc), code, dict(self.menv.pins))

    def output_patterns(self, stage: str) -> tuple[str, ...]:
        pats = contracts.STAGE_OUTPUTS.get(stage, ())
        if self.lane.driver == "acceptance" and stage == "forge":
            pats = contracts.STAGE_OUTPUTS["fixture"] + pats
        return pats

    def outputs(self, stage: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for pattern in self.output_patterns(stage):
            base = self.work if not pattern.startswith("../") else self.lane_dir
            pat = pattern[3:] if pattern.startswith("../") else pattern
            for path in sorted(base.glob(pat)):
                if path.is_file():
                    found[str(path.relative_to(self.lane_dir))] = contracts.sha256_file(path)
        return found

    def _outputs_now(self, recorded: Mapping[str, str]) -> dict[str, str | None]:
        return {rel: (contracts.sha256_file(self.lane_dir / rel) if (self.lane_dir / rel).is_file() else None)
                for rel in recorded}

    # ---- one stage
    def _one(self, idx: int, stage: str, first: int, halted: str | None) -> StageResult:
        if idx < first:
            return StageResult(stage, STAMPED, detail="passed earlier; stamp re-checked by --from",
                               not_run=self.carried_not_run.get(stage, ()))
        if not self._requested(idx):
            return StageResult(stage, NOT_REQUESTED, detail=f"after --to {self._last_requested()}")
        if halted:
            return StageResult(stage, BLOCKED, detail=f"after {halted}")
        missing = missing_provider(stage, self.menv.skills, self.lane.driver)
        if missing:
            return StageResult(stage, ABSENT, provider=missing[1],
                               detail=f"needs {missing[0]} ({missing[1]}), not in the tree yet")
        result = self._execute(stage)
        if result.status == PASS:
            upstream = _stamp_digest(self.lane_dir, self.plan[idx - 1]) if idx else None
            write_stamp(self.lane_dir, stage, self.inputs(stage), self.outputs(stage), upstream)
        return result

    def _execute(self, stage: str) -> StageResult:
        share = float(shares_for(self.lane, self.menv.defaults).get(stage, 0.05))
        timeout = float(self.menv.defaults["stage_timeout_factor"]) * float(self.lane.budget_s) * share
        log = self.lane_dir / "log" / f"{stage}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"# lane {self.lane.id} stage {stage} — {_now()}\n", encoding="utf-8")
        self.recorder.stage = stage
        runner = CommandRunner(self.menv.env, log, Deadline.earliest(time.monotonic() + timeout, self.deadline),
                               self.menv.flakes, self.recorder)
        ctx = StageContext(self.lane, self.menv, self.lane_dir, self.work, stage, runner,
                           self.state, self.recorder)
        started = time.monotonic()
        status, detail = PASS, ""
        try:
            detail = function_for(self.lane.driver, stage)(self.lane, self.state.fixture, ctx) or ""
        except StageFailure as exc:
            status, detail = (TIMEOUT if str(exc).startswith("timed out") else FAIL), str(exc)
        except Exception as exc:                        # a defect in a stage is a failed stage
            runner.note("".join(traceback.format_exception(exc)))
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        if status == PASS and runner.markers.error:
            status, detail = FAIL, "an ERROR: marker in the stage log"
        runner.note(f"{'OK' if status == PASS else 'ERROR'}: stage {stage} {status}"
                    + (f" — {detail}" if detail else ""))
        return StageResult(stage, status, time.monotonic() - started, runner.cpu_s, runner.markers,
                           detail[:2000], str(log.relative_to(self.lane_dir)), round(timeout, 1),
                           not_run=tuple(runner.not_run))

    # ---- service lifecycle and the bundle pin
    def _stop_service(self) -> None:
        service = self.state.service
        if service is None:
            return
        svc.stop(service)
        self.state.service = None
        pin = self.state.bundle_pin
        if pin and pin_now(Path(pin["bundle"])) != {k: pin[k] for k in ("manifest", "tree")}:
            serve = next((r for r in self.results if r.id == "serve"), None)
            if serve:
                serve.status, serve.detail = FAIL, "the bundle changed while it was served (bundle pin)"

    # ---- verdict and observed.json
    def _verdict(self) -> str:
        statuses = {r.status for r in self.results}
        if statuses & {FAIL, TIMEOUT}:
            return "fail"
        return "partial" if ABSENT in statuses else "pass"

    def _finish(self, verdict: str, reason: str | None, skip: bool = False) -> dict:
        if skip:
            status = SKIP if verdict == "skip" else NOT_REQUESTED
            self.results = [StageResult(s, status, detail=reason or "") for s in self.plan]
        run = {"git": dict(self.menv.git), "started_at": self.started, "finished_at": _now(),
               "verdict": verdict, "reason": reason, "tier": self.menv.tier,
               "stages": [r.to_json() for r in self.results],
               "flakes": self.recorder.retries, "flake_calls": dict(self.recorder.calls)}
        self.lane_dir.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        doc = observe.build(self.lane.id, self.lane_dir, self.work, run)
        problems = observe.validate(doc)
        if problems:
            raise ValueError(f"{self.lane.id}: observed.json does not validate: {problems[:3]}")
        if verdict != "refused":                 # a refused --from ran nothing: the last
            observe.write(self.lane_dir, doc)    # observation stays what the lane proved
        return doc


def _stamp_digest(lane_dir: Path, stage: str) -> str | None:
    path = lane_dir / "stamps" / f"{stage}.json"
    return contracts.sha256_file(path) if path.is_file() else None


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
