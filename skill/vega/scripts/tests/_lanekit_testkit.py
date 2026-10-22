"""
What the lane program's unit tests share (test_lanekit.py, test_lanekit_runs.py,
test_lanekit_parts.py): the import path, the registry fixtures, the seeded Errno 39 signature,
a validator that fails on cue, and the synthetic three-stage lane.

Not a test module (no `test_` prefix): the unit tier and pytest collect only the three above.
Fixtures are imported by name into the modules that use them.
"""
from __future__ import annotations

import copy
import os
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import contracts, registry, runner, stages  # noqa: E402
from lanekit.context import MatrixEnv  # noqa: E402
from lanekit.registry import Lane  # noqa: E402

REPO = HERE.parents[2]
PY = sys.executable
SIGNATURE = textwrap.dedent("""\
    Traceback (most recent call last):
      File "validate_model.py", line 760, in check_projector_round_trip
      File "/usr/lib/python3.12/tempfile.py", line 946, in cleanup
        _rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
      File "/usr/lib/python3.12/shutil.py", line 759, in rmtree
    OSError: [Errno 39] Directory not empty: 'Default'
    """)


# --------------------------------------------------------------------------- the registry
@pytest.fixture(scope="module")
def catalogs() -> registry.Catalogs:
    return registry.Catalogs.load(REPO)


@pytest.fixture()
def doc() -> dict:
    return copy.deepcopy(registry.load_document())


def lane_of(doc: dict, lane_id: str) -> dict:
    return next(e for e in doc["lanes"] if e["id"] == lane_id)


def errors_for(doc: dict, catalogs: registry.Catalogs) -> list[str]:
    return registry.validate(doc, catalogs).errors


# --------------------------------------------------------------------------- the flake policy
def fake_validator(tmp: Path, fail_times: int, output: str = SIGNATURE) -> Path:
    """A script named validate_model.py that fails `fail_times` times with `output`, then passes."""
    script = tmp / "validate_model.py"
    script.write_text(textwrap.dedent(f"""\
        import pathlib, sys
        count = pathlib.Path(sys.argv[1])
        n = int(count.read_text()) if count.exists() else 0
        count.write_text(str(n + 1))
        if n < {fail_times}:
            sys.stdout.write({output!r})
            sys.exit(1)
        print("OK: the model validates")
        """), encoding="utf-8")
    return script


# --------------------------------------------------------------------------- a synthetic lane
SYNTH_PLAN = ("fixture", "forge", "train")


@pytest.fixture()
def synth(tmp_path, monkeypatch):
    """A synthetic three-stage lane over a temporary skills tree: each stage runs one command
    that writes its output file, and the code each stage runs is one file in the tree."""
    skills = tmp_path / "skills"
    for stage in SYNTH_PLAN:
        (skills / "code").mkdir(parents=True, exist_ok=True)
        (skills / "code" / f"{stage}.py").write_text(f"# {stage} v1\n")
    monkeypatch.setitem(stages.PLANS, "lanekit", SYNTH_PLAN)
    for stage in SYNTH_PLAN:
        monkeypatch.setitem(contracts.PROVIDERS, stage, ())
        monkeypatch.setitem(contracts.STAGE_CODE, stage, (f"code/{stage}.py",))
        monkeypatch.setitem(contracts.STAGE_OUTPUTS, stage, (f"{stage}.out",))
        monkeypatch.setitem(stages.FUNCTIONS, stage, write_stage_output)
    raw = {"id": "L99", "driver": "lanekit", "budget_s": 60, "expect": {}, "profile": "core"}
    defaults = registry.load_document()["defaults"]
    menv = MatrixEnv(tmp_path / "matrix", skills, REPO, PY, registry.LANES_FILE, defaults,
                     runner.Flakes.load(), dict(os.environ), {"commit": "abc1234", "dirty": False})
    return Lane("L99", raw), menv


def write_stage_output(lane, fx, ctx):
    """The synthetic stage: one command that writes `<stage>.out` in the work dir."""
    out = ctx.work / f"{ctx.stage}.out"
    ctx.run([PY, "-c", f"import pathlib; pathlib.Path({str(out)!r}).write_text('{ctx.stage} done')"])
    return f"{ctx.stage} wrote {out.name}"

