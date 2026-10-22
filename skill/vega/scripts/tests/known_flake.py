#!/usr/bin/env python3
"""Run a command, and retry it ONCE only when it failed with a known flake signature.

    python3 known_flake.py run [--flakes FILE] [--stage NAME] [--record FILE] -- <cmd> [args...]
    python3 known_flake.py check [--flakes FILE]
    python3 known_flake.py match --command "<command line>" --output FILE [--flakes FILE]

The policy (design D6 §6.12, build plan R-H7) is one signature, one retry, always reported:

- A command is retried at most once, and only when (a) it exited non-zero, (b) its command line
  matches a signature's `command_regex`, and (c) its output, stdout and stderr merged in the
  order they arrived, matches the same signature's `output_regex` (`re.search`, no implicit
  flags; a signature that spans lines says `(?s)` itself).
- Any other failure is never retried. A second failure is final. There is never a third attempt.
- The round trip, the negative suites, the page smoke and the inspection checks are never
  retried, whatever a signature says (`NEVER_RETRIED`).
- Every retry is reported: `WARN:` on stderr before the second attempt, `OK:` or `ERROR:` after
  it, and one JSON line in the record file (`--record`, or `$VEGA_FLAKE_RECORD`), in the shape
  of `observed.json.flakes[]`: `{id, stage, command, retried, second_attempt}`.

The signatures are read from `lanekit/flakes.json` beside this file (or `--flakes`, or
`$VEGA_KNOWN_FLAKES`). The format is `vega/known-flakes@1`:

    {"schema": "vega/known-flakes@1",
     "flakes": [{"id": "df-profile-race", "command_regex": "validate_model\\\\.py",
                 "output_regex": "(?s)tempfile.*OSError: \\\\[Errno 39\\\\] ...",
                 "owner": "domain-forge", "upstream": "patches/domain-forge/README.md"}]}

The five keys are required non-empty strings, and `id` is unique. A missing file is a `WARN:`,
and the command then runs once, because no signature is known and so nothing is retryable. A
malformed file is an `ERROR:` with exit status 2, and the command does not run: a broken
allow-list must not pass for an empty one.

`run` streams the command's stdout to stdout and its stderr to stderr as they arrive, so a
caller's pipe (`... | tail -3`) sees what it would see without the wrapper. The wrapper's own
markers go to stderr. The exit status is the last attempt's (128 + n for a signal n; 127 when the
command cannot be started).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

FLAKES_SCHEMA = "vega/known-flakes@1"
REQUIRED_KEYS = ("id", "command_regex", "output_regex", "owner", "upstream")
DEFAULT_FLAKES = Path(__file__).resolve().parent / "lanekit" / "flakes.json"
ENV_FLAKES = "VEGA_KNOWN_FLAKES"
ENV_RECORD = "VEGA_FLAKE_RECORD"
EXIT_USAGE = 2
EXIT_NOT_STARTED = 127

# D6 §6.12: retries never apply to these, whatever a signature says. A flaky negative, round
# trip, page smoke or inspection check would hide exactly the defect it exists to catch.
NEVER_RETRIED = (
    re.compile(r"run_negatives[\w-]*\.py"),
    re.compile(r"(?:page_smoke|smoke_page)\.py"),
    re.compile(r"(?:inspect_check|serving_check)\.py"),
    re.compile(r"test_roundtrip\.py"),
    re.compile(r"predict\.py\S*\s+(?:.*\s)?--check\b"),
)


class FlakesError(ValueError):
    """The known-flakes file exists but is not a valid `vega/known-flakes@1` document."""


@dataclass(frozen=True)
class Signature:
    """One retryable failure: which command, which output, whose bug, where it is tracked."""

    id: str
    command_regex: re.Pattern
    output_regex: re.Pattern
    owner: str
    upstream: str

    def matches(self, command_line: str, output: str) -> bool:
        return bool(self.command_regex.search(command_line)
                    and self.output_regex.search(output))


@dataclass(frozen=True)
class Attempt:
    """What one run of the command left: its exit status and its merged output."""

    returncode: int
    output: str


def _signature_of(entry, where: str) -> Signature:
    if not isinstance(entry, dict):
        raise FlakesError(f"{where} is not an object")
    missing = [k for k in REQUIRED_KEYS
               if not isinstance(entry.get(k), str) or not entry.get(k).strip()]
    if missing:
        raise FlakesError(f"{where} lacks a non-empty {', '.join(missing)} "
                          f"(every entry needs {', '.join(REQUIRED_KEYS)}; an entry with no "
                          "upstream reference has nobody to fix it)")
    compiled = {}
    for key in ("command_regex", "output_regex"):
        try:
            compiled[key] = re.compile(entry[key])
        except re.error as exc:
            raise FlakesError(f"{where}.{key} does not compile: {exc}") from exc
    return Signature(entry["id"], compiled["command_regex"], compiled["output_regex"],
                     entry["owner"], entry["upstream"])


def load_signatures(path: Path) -> tuple[Signature, ...] | None:
    """The signatures of a known-flakes file, None when the file does not exist.

    Raises FlakesError when the file exists but is not a valid `vega/known-flakes@1` document.
    """
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FlakesError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != FLAKES_SCHEMA:
        raise FlakesError(f"{path} is not a {FLAKES_SCHEMA} document "
                          f"(schema {doc.get('schema') if isinstance(doc, dict) else None!r})")
    entries = doc.get("flakes")
    if not isinstance(entries, list):
        raise FlakesError(f"{path}: `flakes` must be a list of signatures")
    signatures = tuple(_signature_of(e, f"flakes[{i}]") for i, e in enumerate(entries))
    ids = [s.id for s in signatures]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise FlakesError(f"{path}: duplicated id {', '.join(duplicated)}")
    return signatures


def flakes_path(explicit: str | None) -> Path:
    """`--flakes`, else `$VEGA_KNOWN_FLAKES`, else `lanekit/flakes.json` beside this file."""
    return Path(explicit or os.environ.get(ENV_FLAKES) or DEFAULT_FLAKES)


def never_retried(command_line: str) -> bool:
    return any(p.search(command_line) for p in NEVER_RETRIED)


def retryable(signatures, command_line: str, attempt: Attempt) -> Signature | None:
    """The signature that makes this failed attempt retryable, or None.

    None when the attempt passed, when the command is one that is never retried, or when no
    signature matches both the command line and the output.
    """
    if attempt.returncode == 0 or never_retried(command_line):
        return None
    return next((s for s in signatures or () if s.matches(command_line, attempt.output)), None)


def _pump(stream, sink, transcript: list, lock: threading.Lock) -> None:
    for chunk in iter(stream.readline, b""):
        sink.write(chunk)
        sink.flush()
        with lock:
            transcript.append(chunk)
    stream.close()


def run_attempt(argv: list[str]) -> Attempt:
    """Run the command once, streaming stdout and stderr through as they arrive."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    transcript: list[bytes] = []
    lock = threading.Lock()
    pumps = [threading.Thread(target=_pump, args=(proc.stdout, sys.stdout.buffer, transcript, lock)),
             threading.Thread(target=_pump, args=(proc.stderr, sys.stderr.buffer, transcript, lock))]
    for t in pumps:
        t.start()
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise
    finally:
        for t in pumps:
            t.join()
    if returncode < 0:
        returncode = 128 - returncode
    return Attempt(returncode, b"".join(transcript).decode("utf-8", errors="replace"))


def short_command(argv: list[str]) -> str:
    """The script a command runs, as `observed.json` names it: `validate_model.py …`."""
    script = next((a for a in argv if a.endswith(".py")), argv[0] if argv else "")
    rest = len(argv) - (argv.index(script) + 1 if script in argv else 0)
    return os.path.basename(script) + (" …" if rest > 0 else "")


def append_record(path: str | None, record: dict) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _say(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _load_or_exit(explicit: str | None):
    path = flakes_path(explicit)
    try:
        signatures = load_signatures(path)
    except FlakesError as exc:
        _say(f"ERROR: known flakes: {exc}")
        raise SystemExit(EXIT_USAGE)
    return path, signatures


def cmd_run(args) -> int:
    argv = list(args.command)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        _say("ERROR: known_flake.py run: no command after `--`")
        return EXIT_USAGE
    path, signatures = _load_or_exit(args.flakes)
    if signatures is None:
        _say(f"WARN: no known-flakes file at {path}; no failure is retryable, running once")
    command_line = shlex.join(argv)
    try:
        first = run_attempt(argv)
    except OSError as exc:
        _say(f"ERROR: known_flake.py: cannot start {argv[0]!r}: {exc}")
        return EXIT_NOT_STARTED
    sig = retryable(signatures, command_line, first)
    if sig is None:
        return first.returncode
    _say(f"WARN: known flake {sig.id} (owner {sig.owner}; upstream {sig.upstream}): "
         f"{short_command(argv)} failed with its signature (exit {first.returncode}); "
         "retrying once")
    second = run_attempt(argv)
    passed = second.returncode == 0
    append_record(args.record or os.environ.get(ENV_RECORD) or None,
                  {"id": sig.id, "stage": args.stage, "command": short_command(argv),
                   "retried": True, "second_attempt": "pass" if passed else "fail"})
    if passed:
        _say(f"OK: known flake {sig.id}: the second attempt passed")
    else:
        _say(f"ERROR: known flake {sig.id}: the second attempt failed too (exit "
             f"{second.returncode}); a flake is retried once, so this failure stands")
    return second.returncode


def cmd_check(args) -> int:
    path, signatures = _load_or_exit(args.flakes)
    if signatures is None:
        _say(f"ERROR: no known-flakes file at {path}")
        return EXIT_USAGE
    print(f"OK: {len(signatures)} known-flake signature(s) in {path}: "
          + (", ".join(f"{s.id} (owner {s.owner})" for s in signatures) or "none"))
    return 0


def cmd_match(args) -> int:
    _path, signatures = _load_or_exit(args.flakes)
    output = Path(args.output).read_text(encoding="utf-8", errors="replace")
    sig = retryable(signatures, args.command, Attempt(1, output))
    print(sig.id if sig else "none")
    return 0 if sig else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="known_flake.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="run a command, retrying once on a known flake signature")
    run.add_argument("--flakes", default=None, help=f"signatures file (default ${ENV_FLAKES} "
                     f"or {DEFAULT_FLAKES.relative_to(DEFAULT_FLAKES.parents[1])})")
    run.add_argument("--stage", default=None, help="the stage named in the retry record")
    run.add_argument("--record", default=None,
                     help=f"append one JSON line per retry here (default ${ENV_RECORD})")
    run.add_argument("command", nargs=argparse.REMAINDER, help="-- <cmd> [args...]")
    run.set_defaults(func=cmd_run)
    check = sub.add_parser("check", help="validate the signatures file")
    check.add_argument("--flakes", default=None)
    check.set_defaults(func=cmd_check)
    match = sub.add_parser("match", help="name the signature a failed command's output matches")
    match.add_argument("--flakes", default=None)
    match.add_argument("--command", required=True, help="the command line, as one string")
    match.add_argument("--output", required=True, help="a file holding the command's output")
    match.set_defaults(func=cmd_match)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
