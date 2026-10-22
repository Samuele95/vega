"""
test_negatives_result.py — the seeded-negative suites' shared record (BUILD-PLAN T10, D6 §12).

    python3 -m pytest vega/scripts/tests/negatives/test_negatives_result.py -q

`negkit` writes `vega/negatives-result@1` for every suite that takes `--json`, and checks what
it writes; `grandfathered.json` lists the cases of the gates older than v2.3 by gate. This file
holds both to their form, and holds each v2.3 suite to the gate ids the build plan gives it
(declared, by reading the suites: a gate id no case names is a gate no run can cover).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import negkit  # noqa: E402
from lanekit import contracts  # noqa: E402

PACK_SUITE = SKILLS / "vega-pack" / "scripts" / "tests" / "run_negatives.py"
SHOWCASE_SUITE = SKILLS / "vega-pack" / "scripts" / "tests" / "run_negatives_showcase.py"
SHAPES_SUITE = SKILLS / "vega-forge" / "scripts" / "tests" / "run_negatives_shapes.py"
SUITES = (SKILLS / "vega-forge" / "scripts" / "tests" / "run_negatives.py", PACK_SUITE,
          SHOWCASE_SUITE, SHAPES_SUITE,
          SKILLS / "vega-train" / "scripts" / "tests" / "run_negatives.py",
          SKILLS / "vega-eval" / "scripts" / "tests" / "run_negatives.py")


# --------------------------------------------------------------------------- ids

@pytest.mark.parametrize("name, wanted", [
    ("N-S1  a stored layout without row_score is refused", "N-S1"),
    ("N-S7a  --story-plan is refused", "N-S7a"),
    ("SS3  a path outside is refused", "SS3"),
    ("W2a  L19: a graph family on rows", "W2a"),
    ("IN1b  a manifest turning inspection off", "IN1b"),
    ("C11a  a 31-statement function is refused", "C11a"),
    ("an unearned verdict without --ship-anyway is refused", None),
    ("N-S7 one space is not the convention", None),
    ("...and no bundle was written", None),
])
def test_a_case_id_is_the_gate_id_and_two_spaces(name, wanted):
    assert negkit.case_id(name) == wanted


# --------------------------------------------------------------------------- the record

def _cases(tmp_path, *verdicts):
    cases = negkit.Cases(tmp_path / "run_negatives_x.py", tmp_path)
    cases.section("x.py — a gate", gate="vega-x/x.py")
    for name, ok in verdicts:
        cases.check(name, ok, "" if ok else "it was not refused")
    return cases


def test_a_finished_run_writes_a_result_that_validates(tmp_path, capsys):
    cases = _cases(tmp_path, ("SS1  a refused showcase", True), ("an older case", True))
    cases.not_run("IN4  L10: the window case", "lane L10 is not in this matrix")
    out = tmp_path / "r.json"
    assert cases.finish(out) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert negkit.problems(doc) == []
    assert doc["schema"] == "vega/negatives-result@1"
    assert (doc["passed"], doc["failed"]) == (2, 0)
    assert [c["id"] for c in doc["cases"]] == ["SS1", None]
    assert doc["cases"][1]["gate"] == "vega-x/x.py" and "gate" not in doc["cases"][0]
    assert doc["not_run"] == [{"id": "IN4", "name": "IN4  L10: the window case",
                               "why": "lane L10 is not in this matrix"}]
    assert "WARN  not run: IN4" in capsys.readouterr().out


def test_a_failed_case_fails_the_run_and_says_why(tmp_path):
    cases = _cases(tmp_path, ("SS1  a refused showcase", False))
    out = tmp_path / "r.json"
    assert cases.finish(out) == 1
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert negkit.problems(doc) == [] and doc["failed"] == 1
    assert doc["cases"][0]["detail"] == "it was not refused"


def test_expect_reads_the_exit_code_and_every_needle(tmp_path):
    cases = negkit.Cases(tmp_path / "s.py", tmp_path)
    assert cases.expect("N-S2  windows", 3, "ERROR: the evaluation stored 9 windows", code=3,
                        contains=("windows",))
    assert not cases.expect("N-S2  windows", 2, "ERROR: windows", code=3, contains=("windows",))
    assert not cases.expect("N-S2  windows", 3, "ERROR: rows", code=3, contains=("windows",))
    assert not cases.expect("N-S2  windows", 3, "ERROR: windows", code=3, absent=("error",))
    details = [r["detail"] for r in cases.results]
    assert details[1] == "exit 2, expected 3"
    assert details[2] == "message does not mention 'windows'"
    assert details[3] == "message should not mention 'error'"


def _doc(**changes):
    doc = negkit.result_doc("vega-pack/scripts/tests/run_negatives.py", "/w",
                            [{"id": "N-S1", "name": "N-S1  x", "ok": True, "detail": ""},
                             {"id": None, "name": "older", "ok": False, "detail": "why",
                              "gate": "vega-pack/pack.py"}])
    doc.update(changes)
    return doc


@pytest.mark.parametrize("changes, needle", [
    ({"schema": "vega/negatives-result@2"}, "schema"),
    ({"passed": 2}, "passed"),
    ({"failed": 0}, "failed"),
    ({"extra": 1}, "unknown key 'extra'"),
    ({"cases": "none"}, "cases is not a list"),
    ({"not_run": [{"name": "x"}]}, "not_run[0]"),
    ({"work": ""}, "work"),
])
def test_what_is_not_a_result_is_named(changes, needle):
    wrong = negkit.problems(_doc(**changes))
    assert any(needle in w for w in wrong), wrong


def test_a_case_without_its_keys_or_with_a_silent_failure_is_named():
    doc = _doc()
    doc["cases"][0] = {"id": "N-S1", "name": "N-S1  x", "ok": "yes", "detail": "", "x": 1}
    doc["cases"][1]["detail"] = ""
    wrong = " | ".join(negkit.problems(doc))
    assert "ok is not a boolean" in wrong and "unknown key 'x'" in wrong
    assert "a failed case says nothing about why" in wrong


def test_the_forge_suites_own_writer_writes_the_same_form():
    """G1's run_negatives.write_json (id = the name's first word when it has no gate id)."""
    doc = {"schema": "vega/negatives-result@1",
           "script": "vega-forge/scripts/tests/run_negatives.py", "work": "/w",
           "cases": [{"id": "C11a", "name": "C11a  x", "ok": True, "detail": ""},
                     {"id": "a", "name": "a case", "ok": True, "detail": ""}],
           "passed": 2, "failed": 0}
    assert negkit.problems(doc) == []


# --------------------------------------------------------------------------- the lanes

def test_a_lane_is_read_off_its_fixture_json(tmp_path):
    root = tmp_path / "m" / "lanes" / "L19"
    (root / "work").mkdir(parents=True)
    spec = root / "work" / "netspec.json"
    spec.write_text("{}", encoding="utf-8")
    fx = {"work": str(root / "work"), "model_html": str(root / "work" / "grid.shaped.html"),
          "netspec": str(spec), "pipeline": str(root / "work" / "p.json"),
          "shaped_dir": str(root / "work" / "shaped"), "run_dir": str(root / "work" / "c"),
          "manifest_digest": "sha256:x", "package": "grid_pipeline"}
    (root / "fixture.json").write_text(json.dumps(fx), encoding="utf-8")
    lane = negkit.lane(tmp_path / "m", "L19")
    assert lane is not None and lane.work == root / "work" and lane.bundle() is None
    assert lane.layered(".vega.train.eval").name == "grid.shaped.vega.train.eval.html"
    assert not lane.evaluated()
    assert negkit.lane(tmp_path / "m", "L20") is None and negkit.lane(None, "L19") is None
    assert negkit.matrix_of(root / "work") == (tmp_path / "m").resolve()
    assert negkit.matrix_of(tmp_path) is None


# --------------------------------------------------------------------------- grandfathered

def test_grandfathered_is_gate_to_case_names_of_scripts_that_exist():
    doc = negkit.grandfathered()
    assert isinstance(doc, dict) and doc
    for gate, names in doc.items():
        script, _, rule = gate.partition("#")
        skill, _, name = script.partition("/")
        assert (SKILLS / skill / "scripts" / name).is_file(), gate
        assert isinstance(names, list) and names, gate
        for case in names:
            assert isinstance(case, str) and case
            # a grandfathered case carries no id, or (forge) the rule id its gate names
            assert negkit.case_id(case) == (rule or None), (gate, case)


def test_the_four_existing_suites_keep_their_grandfathered_cases():
    """Every grandfathered case name is still a case of its suite (read in the source; an
    f-string case is matched on its literal parts)."""
    sources = {s: s.read_text(encoding="utf-8") for s in SUITES}
    for gate, names in negkit.grandfathered().items():
        for case in names:
            text = case.split("  ", 1)[-1] if "#" in gate else case
            parts = [p for p in re.split(r"(?:ingestion|training|evaluation|--story-plan)", text)
                     if p]
            assert any(all(p in src for p in parts) for src in sources.values()), (gate, case)


# --------------------------------------------------------------------------- declared ids

def _ids(path: Path) -> set:
    """The gate ids the suite's case names start with (a literal, or an f-string whose letter
    is filled in: `f"N-S7{letter}  …"`)."""
    found = set()
    pattern = r"(?<![\w-])((?:N-)?[A-Z]{1,3}\d+)[a-z]?(?:\{letter\})?  \S"
    for m in re.finditer(pattern, path.read_text(encoding="utf-8")):
        found.add(m.group(1))
    return found


def test_every_slot_gate_is_a_case_of_the_showcase_suite():
    named = _ids(SHOWCASE_SUITE)
    missing = [g for g in contracts.SLOT_GATES if g not in named]
    assert not missing, missing
    assert set(contracts.SLOT_GATE_ALIASES.values()) <= named


def test_every_pack_gate_of_the_plan_is_a_case_of_the_pack_suite():
    named = _ids(PACK_SUITE)
    wanted = [f"N-S{i}" for i in range(1, 10)] + [f"NI{i}" for i in range(1, 6)]
    wanted += list(contracts.PACK_GRAPH_GATES)
    assert not [g for g in wanted if g not in named], sorted(named)


def test_the_shape_suite_names_w2_w3_and_b2():
    text = SHAPES_SUITE.read_text(encoding="utf-8")
    table = set(re.findall(r'\("((?:W2|W3|B2)[a-z])", "L\d\d"', text))
    wanted = {f"W2{c}" for c in "abcdefghi"} | {f"W3{c}" for c in "abcd"} | {"B2a"}
    assert table == wanted, sorted(wanted ^ table)


def test_every_suite_takes_json_so_the_negatives_tier_passes_it():
    for suite in SUITES:
        assert '"--json"' in suite.read_text(encoding="utf-8"), suite
