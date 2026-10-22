"""
test_exit_abort.py — the A40 closure's own proof (PLAN-AMENDMENTS A40, A42 track C5).

- exit_abort_probe.py --sites finds a pandas read_parquet that does not pass `filesystem=` —
  in code and in code a script emits — and passes one that does; it reads the package's own
  sources (a shipped template's tests too), never a linked external skill, a pytest module or
  its fixtures; and on this tree it finds no such call left (outside the directories another
  closure track still owns, PENDING, each a strict xfail that fails once its track lands).
- The `--no-inspect` skip reason is in lanes.json's allowed_skips, and the lane check allows it
  only on a bundle whose MANIFEST says it was packed with --no-inspect.
- The bundle_api stage's share of a lane budget is at least 0.25 (L10/L19 took 23–63 s under
  load against a 30 s cap).
- The show stage's draft step names the lane's dataset (`--dataset <generator>`).

    python3 -m pytest -q test_exit_abort.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exit_abort_probe as P  # noqa: E402
from lanekit import contracts as C  # noqa: E402
from lanekit import stages as S  # noqa: E402
from lanekit.registry import load_lanes  # noqa: E402

LANES = json.loads((HERE / "lanes.json").read_text(encoding="utf-8"))
DEFAULTS = LANES["defaults"]


# --------------------------------------------------------------------------- the call sites
def test_a_default_read_is_found_and_a_native_one_passes():
    text = ("import pandas as pd\n"
            "a = pd.read_parquet(path)\n"
            "b = pd.read_parquet(path, columns=['x'],\n"
            "                    filesystem=fs.LocalFileSystem())\n"
            "c = __import__('pandas').read_parquet(f(path, g(1)))\n"
            "d = pandas.read_parquet(\n    path,\n    columns=cols)\n"
            "e = check_kit.read_parquet(path)\n")
    assert P.call_sites(text) == [(2, False), (3, True), (5, False), (6, False)]


def test_emitted_code_is_read_too():
    emitter = 'code("""PRED = EVAL / "p.parquet"\npred = pd.read_parquet(PRED)\n""")\n'
    assert P.call_sites(emitter) == [(2, False)]
    fixed = emitter.replace("(PRED)", "(PRED, filesystem=pafs.LocalFileSystem())")
    assert P.call_sites(fixed) == [(2, True)]


def test_production_files_leave_out_tests_fixtures_and_linked_skills(tmp_path):
    root = tmp_path / "repo"
    (root / "skill" / "scripts" / "tests").mkdir(parents=True)
    tpl = root / "skill" / "templates" / "serving" / "tests"
    tpl.mkdir(parents=True)
    bad = "import pandas as pd\npd.read_parquet('x.parquet')\n"
    for rel in ("skill/scripts/run.py", "skill/scripts/tests/test_x.py",
                "skill/scripts/tests/conftest.py", "skill/scripts/tests/zoo_fixtures.py",
                "skill/scripts/tests/run_negatives.py", "skill/templates/serving/tests/test_api.py"):
        (root / rel).write_text(bad, encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    (external / "reader.py").write_text(bad, encoding="utf-8")
    (root / "linked").symlink_to(external, target_is_directory=True)
    (root / "domain-forge").mkdir()
    (root / "domain-forge" / "reader.py").write_text(bad, encoding="utf-8")
    assert P.default_reads(root) == ["skill/scripts/run.py:2",
                                     "skill/scripts/tests/run_negatives.py:2",
                                     "skill/templates/serving/tests/test_api.py:2"]
    assert P.main(["--sites", "--root", str(root)]) == 1
    for rel in P.default_reads(root):
        path = root / rel.split(":")[0]
        path.write_text(bad.replace("')", "', filesystem=fs.LocalFileSystem())"), encoding="utf-8")
    assert P.default_reads(root) == []
    assert P.main(["--sites", "--root", str(root), "--json", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text())["default_reads"] == []


# The skill directories whose call sites another W7 closure track owns (A42): C2 moves the
# dataset-shaper reads (C4's vega-pack reads have landed). Each entry is a strict xfail: once
# the directory reads clean, the xfail passes unexpectedly, the test goes red, and the entry is
# to be removed here (never the check). An empty mapping holds the whole tree to A40.
# A42 C2 'gate' moved the dataset-shaper reads (shape.py, verify_shape.py, the reproduction
# script it emits, its test harnesses): no directory is pending any more.
PENDING: dict[str, str] = {}


def _default_reads_in(skill: str | None) -> list[str]:
    found = P.default_reads(P.REPO)
    if skill is None:
        return [s for s in found if s.split("/")[0] not in PENDING]
    return [s for s in found if s.split("/")[0] == skill]


def test_the_tree_has_no_default_read_left():
    """Every production pandas read_parquet of the package passes Arrow's own filesystem."""
    assert _default_reads_in(None) == []


@pytest.mark.parametrize("skill", [pytest.param(s, marks=pytest.mark.xfail(strict=True, reason=r))
                                   for s, r in PENDING.items()])
def test_a_pending_directory_is_still_pending(skill):
    assert _default_reads_in(skill) == []


def test_the_probe_still_needs_a_parquet_without_sites():
    with pytest.raises(SystemExit) as exc:
        P.main([])
    assert exc.value.code == 2


# --------------------------------------------------------------------------- lanes.json
def test_the_no_inspect_skip_is_allowed_only_on_a_bundle_packed_without_inspection():
    assert C.NO_INSPECT_SKIP in DEFAULTS["allowed_skips"]
    reason = C.NO_INSPECT_SKIP + ": no inspection routes"
    assert S.skips_outside_policy([reason], DEFAULTS, (), no_inspect=True) == []
    assert S.skips_outside_policy([reason], DEFAULTS, (), no_inspect=False) == [reason]
    # a lane may still name it among its own skips; the other reasons are unchanged
    assert S.skips_outside_policy([reason], DEFAULTS, (C.NO_INSPECT_SKIP,), no_inspect=False) == []
    assert S.skips_outside_policy(["monitor reference absent"], DEFAULTS, (), False) == []
    assert S.skips_outside_policy(["the view cannot draw"], DEFAULTS, (), True) == ["the view cannot draw"]
    assert S.skips_outside_policy(["some other reason"], DEFAULTS, (), True) == ["some other reason"]


@pytest.mark.parametrize("serve, off", [({"inspection": {"enabled": False}}, True),
                                        ({"inspection": {"enabled": True}}, False),
                                        ({"inspection": {}}, False), ({}, False), (None, False)])
def test_packed_without_inspection_reads_the_manifest(tmp_path, serve, off):
    manifest = {"schema": "x"} if serve is None else {"serve": serve}
    (tmp_path / C.BUNDLE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert S.packed_without_inspection(tmp_path) is off


def test_the_bundle_api_share_covers_a_loaded_run():
    assert DEFAULTS["stage_shares"]["bundle_api"] >= 0.25


# --------------------------------------------------------------------------- the show draft
def test_the_draft_step_names_the_lanes_dataset(tmp_path):
    draft = C.SHOW_CLI["draft"]
    assert draft[draft.index("--dataset") + 1] == "{dataset}"
    lane = load_lanes(HERE / "lanes.json")["L02"]
    work = tmp_path / "work"
    work.mkdir()
    (work / "landcover.pack.html").write_text("<html></html>", encoding="utf-8")
    ctx = SimpleNamespace(work=work, lane_dir=tmp_path / "lane", lane=lane,
                          path=lambda rel: tmp_path / "skills" / rel,
                          menv=SimpleNamespace(python=sys.executable))
    fields = S._show_fields(ctx, tmp_path / "bundle", tmp_path / "plan")
    assert fields["dataset"] == lane.generator == "landcover"
    argv = S._cli(ctx, draft, fields)
    assert argv[argv.index("--dataset") + 1] == "landcover"
