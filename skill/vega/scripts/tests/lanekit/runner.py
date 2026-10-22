"""
How a lane's commands run: logged, timed, killed on budget, and retried once on a known flake.

    runner = CommandRunner(env, log, deadline, flakes, recorder)
    result = runner.run([python, "validate_model.py", out])       # CommandResult

Every command's stdout and stderr go to the stage log, each attempt under a `$ <command>`
header. Each command runs in its own session (its process group is what a timeout kills) with
`stdin` closed, so a fork that waits for an answer under `--unattended` hits EOF, not a hang
(D6 §2.4, §15.6). CPU time is the child's own rusage (`os.wait4`), grandchildren included.

The flake policy (D6 §6.12) lives in `Flakes`: a failed command is retried once, and only when
its command line matches an entry's `command_regex` and its output matches the entry's
`output_regex` in `flakes.json`. Commands matching `never_retried` (the negatives, the page
smoke, the inspection and serving checks, the round trip) are never retried. The retry is
announced in the log with the notice `flakes.json` prescribes and recorded for observed.json.
Stamps, which `--from` trusts, are lanekit.stamps.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
# a marker line, with the stage prefix a script's output may carry ("geometry layer| WARN: …":
# any text without a pipe, spaces included — a prefix with a space was dropped before gate C)
PREFIX = r"^\s*(?:[^|\n]+\|\s*)?"
MARKER = re.compile(PREFIX + r"(OK|WARN|ERROR|FORK|NOTE):", re.M)
KILL_GRACE_S = 5.0


# --------------------------------------------------------------------------- markers
@dataclass
class Markers:
    ok: int = 0
    warn: int = 0
    error: int = 0
    fork: int = 0

    @classmethod
    def of(cls, text: str) -> "Markers":
        m = cls()
        for kind in MARKER.findall(text):
            if kind != "NOTE":
                setattr(m, kind.lower(), getattr(m, kind.lower()) + 1)
        return m

    def add(self, other: "Markers") -> None:
        for name in ("ok", "warn", "error", "fork"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_json(self) -> dict:
        return {"ok": self.ok, "warn": self.warn, "error": self.error, "fork": self.fork}


def error_lines(text: str, limit: int = 3) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if re.match(PREFIX + r"ERROR:", line)][:limit]


def warn_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if re.match(PREFIX + r"WARN:", line)]


NOT_RUN = re.compile(PREFIX + r"WARN: check not run: (.+?)\s*$", re.M)


def not_run_lines(text: str) -> list[str]:
    """The checks a command says it did not run (`WARN: check not run: <what> — <why>`): a
    gate whose provider has not landed, or a wrapper that is not installed. They are counted
    as WARN like any other and listed per stage in observed.json (`stages[].not_run`)."""
    return NOT_RUN.findall(text)


# --------------------------------------------------------------------------- flakes
class Flakes:
    """The retryable signatures (`vega/known-flakes@1`)."""

    def __init__(self, doc: Mapping[str, Any]) -> None:
        self.doc = doc
        self.entries = list(doc.get("flakes") or [])
        self.never = [re.compile(p) for p in doc.get("never_retried") or []]
        self.notice_format = doc.get("retry_notice") or "WARN: flake {id} fired; retrying once: {command}"
        self.warn_share = float(doc.get("warn_share", 0.05))

    @classmethod
    def load(cls, path: Path = HERE / "flakes.json") -> "Flakes":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def match(self, command: str, output: str) -> Mapping[str, Any] | None:
        """The entry a failed command's output matches, or None (never for a never-retried
        command)."""
        if any(p.search(command) for p in self.never):
            return None
        for entry in self.entries:
            if re.search(entry["command_regex"], command) and re.search(entry["output_regex"], output):
                return entry
        return None

    def fired_in(self, output: str) -> list[str]:
        """The ids of the signatures a (passing) command's output shows — a wrapper inside
        it retried and recovered: the signature's own output, this runner's notice, or the
        notice of X1's known_flake.py ("WARN: known flake <id> (owner …): … retrying once")."""
        ids = [e["id"] for e in self.entries if re.search(e["output_regex"], output)]
        notice = re.compile(r"^\s*WARN: (?:known )?flake ([\w.-]+)\b(?: fired| \(owner)", re.M)
        return sorted(set(ids) | set(notice.findall(output)))

    def notice(self, entry: Mapping[str, Any], command: str) -> str:
        return self.notice_format.format(id=entry["id"], command=command)

    def calls_matching(self, command: str) -> list[str]:
        return [e["id"] for e in self.entries if re.search(e["command_regex"], command)]


@dataclass
class FlakeRecorder:
    """Every retry, and every call a signature could have fired on (the 5 % rule)."""
    stage: str = ""
    retries: list[dict] = field(default_factory=list)
    calls: dict[str, int] = field(default_factory=dict)

    def call(self, ids: Sequence[str]) -> None:
        for fid in ids:
            self.calls[fid] = self.calls.get(fid, 0) + 1

    def retried(self, entry: Mapping[str, Any], command: str, second_ok: bool, via: str = "runner") -> None:
        self.retries.append({"id": entry["id"] if isinstance(entry, Mapping) else entry,
                             "stage": self.stage, "command": command, "retried": True,
                             "second_attempt": "pass" if second_ok else "fail", "via": via})


# --------------------------------------------------------------------------- commands
@dataclass
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    seconds: float
    cpu_s: float
    output: str
    markers: Markers
    timed_out: bool = False
    flake: dict | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)

    def tail(self, lines: int = 12) -> str:
        return "\n".join(self.output.rstrip().splitlines()[-lines:])


class Deadline:
    """The instant (time.monotonic) a command must be done by; None is no deadline."""

    def __init__(self, at: float | None = None) -> None:
        self.at = at

    @classmethod
    def earliest(cls, stage_at: float | None, lane_at: float | None) -> "Deadline":
        """The earlier of the stage's and the lane's budgets (either may be None)."""
        return cls(min((t for t in (stage_at, lane_at) if t is not None), default=None))

    def remaining(self) -> float | None:
        return None if self.at is None else max(self.at - time.monotonic(), 0.0)


class CommandRunner:
    """Runs one stage's commands into its log."""

    def __init__(self, env: Mapping[str, str], log: Path, deadline: Deadline,
                 flakes: Flakes, recorder: FlakeRecorder) -> None:
        self.env, self.log, self.deadline = dict(env), log, deadline
        self.flakes, self.recorder = flakes, recorder
        self.markers = Markers()
        self.not_run: list[str] = []
        self.cpu_s = 0.0
        log.parent.mkdir(parents=True, exist_ok=True)

    def run(self, argv: Sequence[str], cwd: Path | None = None,
            env_extra: Mapping[str, str] | None = None, retry: bool = True) -> CommandResult:
        argv = tuple(str(a) for a in argv)
        command = " ".join(shlex.quote(a) for a in argv)
        self.recorder.call(self.flakes.calls_matching(command))
        result = self._attempt(argv, cwd, env_extra)
        entry = self.flakes.match(command, result.output) if retry and not result.ok and not result.timed_out else None
        if entry is not None:
            self._write(f"\n{self.flakes.notice(entry, command)}\n")
            second = self._attempt(argv, cwd, env_extra)
            self.recorder.retried(entry, command, second.ok)
            second.flake = {"id": entry["id"], "second_attempt": "pass" if second.ok else "fail"}
            result = second
        self.markers.add(result.markers)
        self.not_run += [w for w in dict.fromkeys(not_run_lines(result.output)) if w not in self.not_run]
        return result

    def note(self, text: str) -> None:
        """A line the runner itself adds to the stage log (assertion outcomes)."""
        self._write(text.rstrip("\n") + "\n")
        self.markers.add(Markers.of(text))

    def _write(self, text: str) -> None:
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(text)

    def _attempt(self, argv: tuple[str, ...], cwd: Path | None,
                 env_extra: Mapping[str, str] | None) -> CommandResult:
        env = dict(self.env, **(env_extra or {}))
        self._write(f"\n$ {' '.join(shlex.quote(a) for a in argv)}" + (f"   (cwd {cwd})" if cwd else "") + "\n")
        start_offset = self.log.stat().st_size
        started = time.monotonic()
        with open(self.log, "ab") as out:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            killer = _Killer(proc.pid, self.deadline.remaining())
            _, status, usage = os.wait4(proc.pid, 0)
            killer.cancel()
        proc.returncode = os.waitstatus_to_exitcode(status)
        seconds = time.monotonic() - started
        cpu = usage.ru_utime + usage.ru_stime
        self.cpu_s += cpu
        output = _read_from(self.log, start_offset)
        if killer.fired:
            self._write(f"\nERROR: killed after {seconds:.0f} s — the stage or lane budget ran out\n")
        return CommandResult(argv, proc.returncode, seconds, cpu, output, Markers.of(output), killer.fired)


class _Killer:
    """Kills a process group when its time is up: SIGTERM, then SIGKILL after a grace."""

    def __init__(self, pid: int, timeout: float | None) -> None:
        self.pid, self.fired = pid, False
        self.timer = threading.Timer(timeout, self._kill) if timeout is not None else None
        if self.timer:
            self.timer.daemon = True
            self.timer.start()

    def _kill(self) -> None:
        self.fired = True
        for sig, pause in ((signal.SIGTERM, KILL_GRACE_S), (signal.SIGKILL, 0)):
            try:
                os.killpg(self.pid, sig)
            except ProcessLookupError:
                return
            time.sleep(pause)

    def cancel(self) -> None:
        if self.timer:
            self.timer.cancel()


def _read_from(path: Path, offset: int) -> str:
    with open(path, "rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", errors="replace")


def kill_group(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        time.sleep(0.5)


# --------------------------------------------------------------------------- environment
def lane_env(base: Mapping[str, str], python: str, threads: int, device: str,
             extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment every lane process runs with (D6 §2.4)."""
    env = dict(base)
    env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TZ": "UTC",
                "LC_ALL": "C.UTF-8", "PYTHON": python,
                "PATH": os.pathsep.join([str(Path(python).parent), base.get("PATH", "/usr/bin:/bin")])})
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(extra or {})
    return env
