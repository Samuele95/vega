"""
Unit tests of the lane program itself (T01), part 2: lanes run by LaneRun and the scheduler —
stamps and `--from`, flake retries recorded in observed.json (by the runner and through the
acceptance scripts' wrapper), absent and failing stages, `--to`, `requires`, the lane budget
under `-j`, and the checks a stage says it did not run.

Nothing here trains a network: the lanes these tests drive are synthetic stage functions over
temporary directories, so the suites run in seconds. The real lanes run under matrix.sh.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lanekit_testkit import HERE, PY, REPO, SYNTH_PLAN, fake_validator, synth, write_stage_output  # noqa: E402,F401 (fixtures)
from lanekit import contracts, observe, runner, stages, timing  # noqa: E402
from lanekit.context import MatrixEnv, RunOptions  # noqa: E402
from lanekit.lane import LaneRun  # noqa: E402
from lanekit.registry import Lane  # noqa: E402


def test_a_lane_runs_stamps_and_writes_observed(synth):
    lane, menv = synth
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] == "pass" and [s["status"] for s in doc["stages"]] == ["pass"] * 3
    lane_dir = menv.matrix / "lanes" / "L99"
    assert observe.validate(json.loads((lane_dir / "observed.json").read_text())) == []
    stamp = json.loads((lane_dir / "stamps" / "train.json").read_text())
    assert stamp["schema"] == contracts.LANE_STAMP and stamp["outputs"] == {"work/train.out": contracts.sha256_file(lane_dir / "work/train.out")}
    assert stamp["upstream"] == contracts.sha256_file(lane_dir / "stamps" / "forge.json")


def test_from_resumes_after_valid_stamps(synth):
    lane, menv = synth
    LaneRun(lane, menv, RunOptions()).run()
    doc = LaneRun(lane, menv, RunOptions(from_stage="train")).run()
    assert [s["status"] for s in doc["stages"]] == ["stamped", "stamped", "pass"]
    assert doc["verdict"] == "pass"


@pytest.mark.parametrize("tamper, needle", [
    (lambda lane, menv: (menv.skills / "code" / "forge.py").write_text("# forge v2\n"),
     "forge: code code/forge.py changed since it passed"),
    (lambda lane, menv: (menv.matrix / "lanes/L99/work/fixture.out").write_text("edited"),
     "fixture: its output work/fixture.out changed since it passed"),
    (lambda lane, menv: (menv.matrix / "lanes/L99/stamps/fixture.json").unlink(),
     "fixture: no stamp"),
    (lambda lane, menv: lane.raw.update(budget_s=61),
     "fixture: the lane's registry entry changed since it passed"),
])
def test_from_refuses_a_stale_stamp(synth, tamper, needle):
    lane, menv = synth
    LaneRun(lane, menv, RunOptions()).run()
    before = (menv.matrix / "lanes/L99/work/train.out").stat().st_mtime_ns
    observed = (menv.matrix / "lanes/L99/observed.json").read_bytes()
    tamper(lane, menv)
    doc = LaneRun(lane, menv, RunOptions(from_stage="train")).run()
    assert doc["verdict"] == "refused" and needle in doc["reason"], doc["reason"]
    assert (menv.matrix / "lanes/L99/work/train.out").stat().st_mtime_ns == before       # nothing ran
    assert (menv.matrix / "lanes/L99/observed.json").read_bytes() == observed           # nor recorded


def test_from_refuses_a_broken_chain(synth):
    """A stamp re-written after the stamps that follow it (an earlier stage re-ran on its own)
    breaks the chain: the later stamps were made on another run of it."""
    lane, menv = synth
    LaneRun(lane, menv, RunOptions()).run()
    path = menv.matrix / "lanes/L99/stamps/fixture.json"
    stamp = json.loads(path.read_text())
    stamp["at"] = "2099-01-01T00:00:00Z"
    path.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    doc = LaneRun(lane, menv, RunOptions(from_stage="train")).run()
    assert doc["verdict"] == "refused" and "forge: its stamp does not follow the previous stage's" in doc["reason"]


def test_from_rewrites_the_stamps_it_reruns(synth):
    lane, menv = synth
    LaneRun(lane, menv, RunOptions()).run()
    LaneRun(lane, menv, RunOptions(from_stage="forge")).run()
    doc = LaneRun(lane, menv, RunOptions(from_stage="train")).run()
    assert doc["verdict"] == "pass"


def test_a_lane_records_a_flake_retry_in_observed(synth, monkeypatch):
    """The closing case of T01: a seeded Errno 39 in a stage's validate_model.py call is retried
    once, the stage passes, and observed.json and the run's summary line record it."""
    lane, menv = synth
    (menv.skills / "domain-forge" / "scripts").mkdir(parents=True)
    validator = fake_validator(menv.skills / "domain-forge" / "scripts", fail_times=1)
    counter = menv.matrix / "count"

    def validates(lane, fx, ctx):
        ctx.run([PY, validator, counter])
        return "validated"

    monkeypatch.setitem(stages.FUNCTIONS, "forge", validates)
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] == "pass" and counter.read_text() == "2"
    assert doc["flakes"] == [{"id": "df-profile-race", "stage": "forge", "command": doc["flakes"][0]["command"],
                              "retried": True, "second_attempt": "pass", "via": "runner"}]
    assert "validate_model.py" in doc["flakes"][0]["command"] and doc["flake_calls"] == {"df-profile-race": 1}
    saved = json.loads((menv.matrix / "lanes/L99/observed.json").read_text())
    assert saved["flakes"] == doc["flakes"] and observe.validate(saved) == []
    assert timing.flake_warnings([doc], runner.Flakes.load()) == ["WARN: flake df-profile-race fired 1 of 1 calls"]


STAND_IN_WRAPPER = textwrap.dedent("""\
    # A stand-in for X1's known_flake.py: the same CLI, retry and record (run --stage S -- cmd).
    import json, os, re, subprocess, sys
    args = sys.argv[1:]
    stage = args[args.index("--stage") + 1]
    cmd = args[args.index("--") + 1:]
    first = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(first.stdout)
    if first.returncode == 0 or not re.search(r"(?s)tempfile.*Errno 39", first.stdout):
        sys.exit(first.returncode)
    sys.stderr.write("WARN: known flake df-profile-race (owner domain-forge; upstream x): "
                     "validate_model.py … failed with its signature (exit 1); retrying once\\n")
    second = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(second.stdout)
    with open(os.environ["VEGA_FLAKE_RECORD"], "a") as fh:
        fh.write(json.dumps({"id": "df-profile-race", "stage": stage, "command": "validate_model.py …",
                             "retried": True, "second_attempt": "pass" if second.returncode == 0 else "fail"}) + "\\n")
    sys.exit(second.returncode)
    """)


def _acceptance_lane(synth, monkeypatch, wrapper: Path, fail_times: int = 1,
                     pack_function: bool = False) -> tuple[Lane, MatrixEnv, Path]:
    """The synthetic lane driven like L01: its forge stage is an acceptance script whose
    validate_model.py call goes through the flake wrapper, and whose validator fails
    `fail_times` times with the seeded Errno 39. With `pack_function` the call is the pack
    acceptance's own validate_model.py line (read from the script), with the wrapper at `wrapper`."""
    lane, menv = synth
    lane.raw["driver"] = "acceptance"
    monkeypatch.setitem(stages.PLANS, "acceptance", ("forge",))
    scripts = menv.skills / "accept"
    scripts.mkdir(parents=True)
    validator = fake_validator(scripts, fail_times=fail_times)
    counter = menv.matrix / "count"
    call = (f"KNOWN_FLAKE={wrapper}\n{pack_flake_call(f'{PY} {validator} {counter}')} | tail -3"
            if pack_function else f"python3 {wrapper} run --stage forge -- {PY} {validator} {counter} | tail -3")
    (scripts / "acceptance.sh").write_text("set -euo pipefail\nWORK=\"$1\"\n" + call + textwrap.dedent("""
        echo done > "$WORK/forge.out"
        printf '\\n=== acceptance passed\\n'
        """))
    monkeypatch.setitem(contracts.ACCEPTANCE, "forge", "accept/acceptance.sh")
    return lane, menv, counter


PACK_VALIDATE = 'python3 "$KNOWN_FLAKE" run --stage pack -- python3 "$DF/validate_model.py" "$OUT"'


def pack_flake_call(command: str) -> str:
    """vega-pack's acceptance.sh validate_model.py call, X1's form, with `command` for the
    validator: the one line of the script that runs it, always through the wrapper."""
    text = (REPO / contracts.ACCEPTANCE["pack"]).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if "validate_model.py" in ln and not ln.lstrip().startswith("#")]
    assert len(lines) == 1 and lines[0].startswith(PACK_VALIDATE), lines
    return PACK_VALIDATE.replace('python3 "$DF/validate_model.py" "$OUT"', command)


def test_an_acceptance_stage_records_the_wrappers_retry(synth, monkeypatch, tmp_path):
    """L01's path: the wrapper inside the acceptance script retries the seeded Errno 39 once and
    appends its record to $VEGA_FLAKE_RECORD; observed.json carries it, via known_flake.py."""
    wrapper = tmp_path / "known_flake_stand_in.py"
    wrapper.write_text(STAND_IN_WRAPPER)
    lane, menv, counter = _acceptance_lane(synth, monkeypatch, wrapper)
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] == "pass", doc["stages"]
    assert counter.read_text() == "2"
    assert doc["flakes"] == [{"id": "df-profile-race", "stage": "forge", "command": "validate_model.py …",
                              "retried": True, "second_attempt": "pass", "via": "known_flake.py"}]
    assert doc["flake_calls"] == {"df-profile-race": 1}
    assert observe.validate(doc) == []


@pytest.mark.parametrize("pack_function, stage", [(False, "forge"), (True, "pack")])
def test_an_acceptance_stage_with_x1s_wrapper(synth, monkeypatch, pack_function, stage):
    """The same through X1's real known_flake.py and this package's flakes.json, once both are
    in the tree (the W0 integration): X1's call form, and the pack acceptance's own function."""
    wrapper = HERE / "known_flake.py"
    if not wrapper.is_file():
        pytest.skip("known_flake.py (work package X1) is not in this tree yet; the stand-in covers it")
    lane, menv, counter = _acceptance_lane(synth, monkeypatch, wrapper, pack_function=pack_function)
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] == "pass" and counter.read_text() == "2", doc["stages"]
    assert [(f["id"], f["stage"], f["second_attempt"], f["via"]) for f in doc["flakes"]] == \
        [("df-profile-race", stage, "pass", "known_flake.py")]
    assert doc["stages"][0]["not_run"] == []


def test_the_pack_acceptance_calls_the_wrapper_unconditionally(synth, monkeypatch, tmp_path):
    """The pack acceptance's validate_model.py call is X1's form, with no bare fallback (T11): on
    a tree without the wrapper the stage fails, it does not run the validator unretried."""
    text = (REPO / contracts.ACCEPTANCE["pack"]).read_text(encoding="utf-8")
    assert "retry_known_flake" not in text and "is not installed (X1)" not in text
    lane, menv, counter = _acceptance_lane(synth, monkeypatch, tmp_path / "no_known_flake.py",
                                           fail_times=0, pack_function=True)
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] != "pass" and not counter.exists(), doc["stages"]


def test_the_pack_acceptance_with_the_wrapper_retries(synth, monkeypatch, tmp_path):
    """The same call with a wrapper installed: one retry, recorded under the pack stage's name,
    and nothing listed as not run."""
    wrapper = tmp_path / "known_flake_stand_in.py"
    wrapper.write_text(STAND_IN_WRAPPER)
    lane, menv, counter = _acceptance_lane(synth, monkeypatch, wrapper, pack_function=True)
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] == "pass" and counter.read_text() == "2", doc["stages"]
    assert [(f["id"], f["stage"], f["second_attempt"]) for f in doc["flakes"]] == [("df-profile-race", "pack", "pass")]
    assert doc["stages"][0]["not_run"] == []


def test_a_stage_lists_the_checks_it_did_not_run(synth, monkeypatch):
    """A `WARN: check not run: …` line (lanekit.assertions' Checker.not_run) is a WARN, not a
    failure, and observed.json lists it under the stage; the next stages list nothing. A
    `--from` run keeps the list on the stage it did not re-run."""
    lane, menv = synth

    def says(lane, fx, ctx):
        line = "WARN: check not run: the per-block shape oracle (D6 §6.2) — oracles.py arrives with T08"
        ctx.run([PY, "-c", f"print({line!r}); print({line!r})"])
        return write_stage_output(lane, fx, ctx)
    monkeypatch.setitem(stages.FUNCTIONS, "forge", says)
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] == "pass" and observe.validate(doc) == []
    assert [s["not_run"] for s in doc["stages"]] == [
        [], ["the per-block shape oracle (D6 §6.2) — oracles.py arrives with T08"], []]
    assert doc["stages"][1]["markers"]["warn"] == 2
    resumed = LaneRun(lane, menv, RunOptions(from_stage="train")).run()
    assert [(st["status"], st["not_run"]) for st in resumed["stages"][:2]] == [
        ("stamped", []), ("stamped", ["the per-block shape oracle (D6 §6.2) — oracles.py arrives with T08"])]


def test_absent_stage_stops_the_lane_and_names_its_provider(synth, monkeypatch):
    lane, menv = synth
    monkeypatch.setitem(contracts.PROVIDERS, "train", (("vega/scripts/tests/lanekit/authoring.py", "T03"),))
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] == "partial"
    assert doc["stages"][2]["status"] == "absent" and doc["stages"][2]["provider"] == "T03"


def test_a_failing_stage_blocks_the_rest(synth, monkeypatch):
    lane, menv = synth

    def fails(lane, fx, ctx):
        ctx.run([PY, "-c", "print('ERROR: the gate refused it'); raise SystemExit(2)"])

    monkeypatch.setitem(stages.FUNCTIONS, "forge", fails)
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert [s["status"] for s in doc["stages"]] == ["pass", "fail", "blocked"]
    assert "the gate refused it" in doc["stages"][1]["detail"] and doc["verdict"] == "fail"


def test_to_stops_early(synth):
    lane, menv = synth
    doc = LaneRun(lane, menv, RunOptions(to_stage="forge")).run()
    assert [s["status"] for s in doc["stages"]] == ["pass", "pass", "not-requested"]
    assert doc["verdict"] == "pass"


def test_a_queued_lane_keeps_its_whole_budget(synth, monkeypatch):
    """-j 1 and two lanes of a 3 s budget whose forge takes 1.5 s: the second lane waits for the
    first, then still gets its whole budget, because its clock starts when it runs."""
    import run_matrix
    _, menv = synth
    defaults = dict(menv.defaults, lane_kill_factor=1, stage_timeout_factor=1,
                    stage_shares={stage: 1.0 for stage in SYNTH_PLAN})
    menv = dataclasses.replace(menv, defaults=defaults)

    def slow(lane, fx, ctx):
        ctx.run([PY, "-c", "import time; time.sleep(1.5)"])
        return write_stage_output(lane, fx, ctx)
    monkeypatch.setitem(stages.FUNCTIONS, "forge", slow)
    lanes = [Lane(i, {"id": i, "driver": "lanekit", "budget_s": 3, "expect": {}, "profile": "core"})
             for i in ("L97", "L98")]
    matrix = run_matrix.Matrix(menv.matrix, REPO, PY, "cpu", None, menv=menv)
    outcome = run_matrix.run_lanes(matrix, lanes, RunOptions(), jobs=1)
    got = {d["lane"]: [(s["id"], s["status"]) for s in d["stages"]] for d in outcome.docs}
    assert [d["verdict"] for d in outcome.docs] == ["pass", "pass"], got
    first, second = (json.loads((menv.matrix / "lanes" / i / "observed.json").read_text()) for i in ("L97", "L98"))
    assert second["started_at"] >= first["finished_at"]


def test_a_lane_whose_env_is_missing_is_skipped(synth, monkeypatch):
    lane, menv = synth
    lane.raw["requires"] = ["VEGA_TEST_UNSET_VAR"]
    monkeypatch.delenv("VEGA_TEST_UNSET_VAR", raising=False)
    doc = LaneRun(lane, menv, RunOptions()).run()
    assert doc["verdict"] == "skip" and {s["status"] for s in doc["stages"]} == {"skip"}

