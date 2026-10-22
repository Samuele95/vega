"""
What a lane's stages share: the run-wide `MatrixEnv`, the per-lane `LaneState`, the
`StageContext` a stage function receives, and the `StageResult` it becomes.

A stage function is `run_<stage>(lane, fx, ctx) -> str` (lanekit.stages): it drives commands
through `ctx.run` (logged, budgeted, retried once on a known flake), calls lanekit helpers in
their own process through `ctx.helper`, and raises `StageFailure` with the reason when an
assertion fails. What it returns is the stage's one-line detail.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import contracts, service as svc, skilltree
from .registry import Lane, flatten_types
from .runner import CommandResult, CommandRunner, FlakeRecorder, Flakes, Markers, error_lines, warn_lines

PASS, FAIL, TIMEOUT, BLOCKED, ABSENT = "pass", "fail", "timeout", "blocked", "absent"
SKIP, NOT_REQUESTED, STAMPED = "skip", "not-requested", "stamped"
SERVICE_STAGES = ("serve", "serve_check", "inspect", "page")


class StageFailure(Exception):
    """A stage's assertion failed; the message says which and why."""


# --------------------------------------------------------------------------- run-wide context
@dataclass(frozen=True)
class MatrixEnv:
    """What every lane of one matrix run shares."""
    matrix: Path
    skills: Path
    repo: Path
    python: str
    lanes_file: Path
    defaults: Mapping[str, Any]
    flakes: Flakes
    env: Mapping[str, str]
    git: Mapping[str, Any]
    tier: str | None = None
    pins: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOptions:
    from_stage: str | None = None
    to_stage: str | None = None
    strict: bool = False


@dataclass
class LaneState:
    """What stages hand each other within one lane run."""
    fixture: dict | None = None
    service: svc.Service | None = None
    bundle_pin: dict | None = None


@dataclass
class StageContext:
    lane: Lane
    menv: MatrixEnv
    lane_dir: Path
    work: Path
    stage: str
    runner: CommandRunner
    state: LaneState
    recorder: FlakeRecorder

    def path(self, rel: str) -> Path:
        return self.menv.skills / rel

    def py(self, rel: str, args: Sequence[Any] = ()) -> list[str]:
        """The suite's interpreter on a script of the skills tree, with its arguments."""
        return [self.menv.python, str(self.path(rel)), *map(str, args)]

    def helper(self, name: str, args: Sequence[Any] = ()) -> CommandResult:
        """A lanekit.checks helper, run in its own process from the skills tree's copy of
        lanekit (python -m lanekit.checks <name> …)."""
        argv = [self.menv.python, "-m", "lanekit.checks", name, "--lanes-file",
                str(self.menv.lanes_file), "--lane", self.lane.id, "--lane-dir",
                str(self.lane_dir), "--skills", str(self.menv.skills), *map(str, args)]
        return self.run(argv, env_extra={"PYTHONPATH": str(self.path(contracts.TESTS_DIR))})

    def run(self, argv: Sequence[Any], cwd: Path | None = None, retry: bool = True,
            env_extra: Mapping[str, str] | None = None, check: bool = True) -> CommandResult:
        result = self.runner.run(argv, cwd=cwd or self.work, env_extra=env_extra, retry=retry)
        if check and result.timed_out:
            raise StageFailure(f"timed out: {Path(str(argv[1] if len(argv) > 1 else argv[0])).name}")
        if check and not result.ok:
            raise StageFailure(_why(result))
        if check and result.markers.error:
            raise StageFailure("; ".join(error_lines(result.output)) or "an ERROR: marker")
        return result

    def validate_layer(self, html: Path) -> None:
        """domain-forge's validator on a layered file (retried once on the known flake)."""
        self.run(self.py("domain-forge/scripts/validate_model.py", [html]))

    def note(self, text: str) -> None:
        self.runner.note(text)


def _why(result: CommandResult) -> str:
    name = next((Path(a).name for a in result.argv[1:3] if a.endswith((".py", ".sh"))), result.argv[0])
    errors = error_lines(result.output)
    return f"{name} exited {result.returncode}: " + ("; ".join(errors) if errors else result.tail(4))


# --------------------------------------------------------------------------- results
@dataclass
class StageResult:
    id: str
    status: str
    seconds: float = 0.0
    cpu_s: float = 0.0
    markers: Markers = field(default_factory=Markers)
    detail: str = ""
    log: str | None = None
    timeout_s: float | None = None
    provider: str | None = None
    not_run: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {"id": self.id, "status": self.status, "seconds": round(self.seconds, 2),
                "cpu_s": round(self.cpu_s, 2), "timeout_s": self.timeout_s, "log": self.log,
                "markers": self.markers.to_json(), "detail": self.detail, "provider": self.provider,
                "not_run": list(self.not_run)}


# --------------------------------------------------------------------------- what several stages assert
def pin_now(bundle: Path) -> dict:
    """sha256(MANIFEST.json) and the digest of the bundle tree without showcase/ and its siblings
    (D6 §6.5)."""
    return {"manifest": contracts.sha256_file(bundle / "MANIFEST.json"),
            "tree": skilltree.tree_digest(bundle, exclude_top=(contracts.SHOWCASE_DIR, *contracts.SHOWCASE_SIBLINGS))}


def record_pin(ctx: StageContext) -> dict:
    bundle = find_bundle(ctx.work)
    pin = {"bundle": str(bundle), **pin_now(bundle)}
    ctx.state.bundle_pin = pin
    out = ctx.lane_dir / "results" / "bundle_pin.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    return pin


def find_bundle(work: Path) -> Path:
    manifests = sorted(work.glob("bundle/*/MANIFEST.json"))
    if not manifests:
        raise StageFailure("no bundle/*/MANIFEST.json in the work dir")
    return manifests[0].parent


def assert_packed_blocks(lane: Lane, work: Path) -> str:
    """The packed netspec's block-type multiset equals the lane's declared blocks (D6 §6.2):
    the observed coverage source is the network that shipped."""
    spec = json.loads((find_bundle(work) / "netspec.json").read_text(encoding="utf-8"))
    packed = sorted(flatten_types(spec.get("blocks") or []))
    declared = sorted(lane.flat_block_types())
    if packed != declared:
        raise StageFailure(f"the packed network's blocks {packed} are not the lane's {declared}")
    return f"packed blocks = declared ({len(packed)} types)"


def flake_record_path(ctx: StageContext) -> Path:
    """Where the flake wrapper inside a script this stage runs appends its retry records
    ($VEGA_FLAKE_RECORD); emptied before the stage runs the script."""
    path = ctx.lane_dir / "results" / f"flakes.{ctx.stage}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return path


def record_inner_flakes(ctx: StageContext, record: Path) -> int:
    """The retries X1's known_flake.py made inside a script this stage ran, read from its
    record file (one observed.json.flakes[] object per line), recorded for observed.json and
    summary.json (D6 §6.12: always reported). Returns how many it recorded."""
    if not record.is_file():
        return 0
    count = 0
    for line in record.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            ctx.note(f"WARN: an unreadable flake record in {record.name}: {line[:120]}")
            continue
        if not isinstance(entry, Mapping) or not all(k in entry for k in contracts.FLAKE_RECORD_KEYS):
            ctx.note(f"WARN: a flake record without {', '.join(contracts.FLAKE_RECORD_KEYS)}: {line[:120]}")
            continue
        ctx.recorder.retries.append({"id": str(entry["id"]), "stage": str(entry.get("stage") or ctx.stage),
                                     "command": str(entry["command"]), "retried": bool(entry["retried"]),
                                     "second_attempt": "pass" if entry["second_attempt"] == "pass" else "fail",
                                     "via": "known_flake.py"})
        count += 1
    return count


def record_flakes_seen(ctx: StageContext, result: CommandResult) -> None:
    """A passing script whose output shows a known signature recovered, through a wrapper that
    left no record file; the matrix records that retry all the same (D6 §6.12)."""
    for fid in ctx.menv.flakes.fired_in(result.output):
        ctx.recorder.retried(fid, f"{result.command} (inside)", result.ok, via="known_flake.py")


def count_inner_calls(ctx: StageContext, script: Path) -> None:
    """The calls a signature could fire on inside a script the runner does not see command by
    command: the script's own invocations of the matching command (the 5 % rule's denominator)."""
    text = script.read_text(encoding="utf-8")
    for entry in ctx.menv.flakes.entries:
        calls = len(re.findall(r"^[^#\n]*" + entry["command_regex"], text, re.M))
        ctx.recorder.call([entry["id"]] * calls)


def markers_budget(ctx: StageContext, result: CommandResult) -> None:
    """The lane's WARN budget for this stage (expect.markers.<stage>), default 0 on the compile."""
    spec = (ctx.lane.expect.get("markers") or {}).get(ctx.stage) or {}
    limit = spec.get("warn_max", 0)
    warns = warn_lines(result.output)
    if limit is not None and len(warns) > limit:
        raise StageFailure(f"{len(warns)} WARN: marker(s), the lane allows {limit}: {warns[:3]}")
    for needle in spec.get("warn_require") or []:
        if not any(re.search(needle, w) for w in warns):
            raise StageFailure(f"no WARN: matching {needle!r}")
