#!/usr/bin/env python3
"""
test_audit_quorum.py — the judgement half of the conformance gate, reproduced or recorded, and
advisory (code-contract.md §5; the user's decisions of 2026-10-11).

    python3 -m pytest -q vega-forge/scripts/tests/test_audit_quorum.py

The rule `audit_quorum.py` states once for the forge's and the pack's apply scripts: a
structural or divergence finding of `/architect --baseline` or `/improve --report-only` blocks
only when at least two of three independent fresh audits report it — the same file and the
same principle anchor or category; one clean first audit passes alone; a first audit that
reports one is judged only with two fresh audits beside it. The cases: confirmed (by anchor, by
category, at another depth, the same defect counted once), unconfirmed (another file, another
principle, a one-run-only finding of a later audit), the one-run cases (clean, pending), the
procedure's refusals (two runs, a run given twice, a copied run, another code), the location
parser (scope prefixes, line numbers, the baseline's files, prose), and the command's exit
codes; the advisory outcome (unattended a WARN and an open item, attended the user's recorded
FORK answer — accept and record publishes, fix now, stop and no answer refuse, a default `or`
is not an answer) and the advisory record's own check; run ids (F31: two audits that share one
are refused, at one path or two).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))

import audit_quorum as q  # noqa: E402

SCOPE = "vega-runs/2026-10-11T00-00Z/landcover_cover"


def finding(fid, location, anchors=("EJ-64",), depth="surface", tags=(), axis="Abstraction"):
    return {"id": fid, "depth": depth, "tags": list(tags), "anchors": list(anchors),
            "axis": axis, "location": location, "title": f"finding {fid}",
            "anchor_justification": {a: f"{a}: why" for a in anchors}}


def audit(tmp: Path, name: str, *findings, scope=SCOPE) -> Path:
    run = tmp / ".claude" / "architect-runs" / name
    run.mkdir(parents=True)
    doc = {"run_timestamp": name, "scope": {"paths": [scope], "report_only": True,
                                            "baseline": "blueprint-runs/2026-10-11T00-00Z/blueprint.md"},
           "findings": list(findings)}
    (run / "findings.json").write_text(json.dumps(doc), encoding="utf-8")
    return run


STRUCT = dict(depth="structural")
DIVERGE = dict(depth="surface", tags=("divergence",))


# --------------------------------------------------------------------------- one audit

def test_one_clean_audit_passes_alone(tmp_path):
    first = audit(tmp_path, "a1", finding(1, f"{SCOPE}/domain.py:10-20"),
                  finding(2, "model/store.py:4", depth="architectural"))
    j = q.judge_paths([first], "architect")
    assert j.passes and j.need == 0 and not j.problems
    rec = j.record
    assert rec["audits_required"] == 1 and len(rec["audits"]) == 1
    assert rec["structural"] == rec["divergence"] == 0 and rec["recorded"] == 2
    assert rec["audits"][0]["digest"].startswith("sha256:") and rec["audits"][0]["scope"] == [SCOPE]


def test_a_first_audit_with_a_structural_finding_needs_two_fresh_audits(tmp_path):
    first = audit(tmp_path, "a1", finding(1, f"{SCOPE}/serving/predict.py:12", **STRUCT))
    j = q.judge_paths([first], "architect")
    assert not j.passes and j.need == 2
    assert "2 more fresh audit(s)" in j.record["pending"]
    assert j.record["audits_required"] == 3 and j.record["matching"] == []


def test_a_divergence_tag_is_blocking_class_too(tmp_path):
    first = audit(tmp_path, "a1", finding(1, "ports.py:3", **DIVERGE))
    assert q.judge_paths([first], "architect").need == 2


# --------------------------------------------------------------------------- confirmed

def test_a_finding_two_audits_report_is_confirmed_and_blocks(tmp_path):
    runs = [audit(tmp_path, "a1", finding(1, f"{SCOPE}/serving/predict.py:12-40",
                                          anchors=("HEX", "SOLID-D"), **STRUCT)),
            audit(tmp_path, "a2", finding(4, "serving/predict.py:88", anchors=("SOLID-D",),
                                          axis="Composition", **STRUCT)),
            audit(tmp_path, "a3", finding(1, "domain.py:3", anchors=("EJ-15",)))]
    j = q.judge_paths(runs, "architect")
    assert not j.passes and j.need == 0
    rec = j.record
    assert rec["structural"] == 1, "the same defect reported twice is counted once"
    rows = {r["finding"]: r for r in rec["matching"]}
    assert rows["1:1"]["reported_by"] == [1, 2] and rows["1:1"]["confirmed"] and rows["1:1"]["counted"]
    assert rows["2:4"]["confirmed"] and not rows["2:4"]["counted"]
    assert rows["1:1"]["matched"] == ["2:4"]
    assert len(j.problems) == 1 and "confirmed by audits 1, 2 of 3" in j.problems[0]
    assert j.judged and not j.invalid and j.confirmed == j.problems, "confirmed is advisory, not invalid"


def test_the_same_category_confirms_without_a_shared_anchor(tmp_path):
    runs = [audit(tmp_path, "a1", finding(1, "model/store.py:122", anchors=("EJ-51",),
                                          axis="Abstraction", **STRUCT)),
            audit(tmp_path, "a2"),
            audit(tmp_path, "a3", finding(9, f"{SCOPE}/model/store.py:140-166",
                                          anchors=("EJ-1",), axis="abstraction"))]
    rec = q.judge_paths(runs, "architect").record
    assert rec["structural"] == 1 and rec["matching"][0]["reported_by"] == [1, 3]


def test_a_report_at_another_depth_reproduces_the_finding(tmp_path):
    runs = [audit(tmp_path, "a1"),
            audit(tmp_path, "a2", finding(2, "serving/monitor.py:28", anchors=("HEX",), **DIVERGE)),
            audit(tmp_path, "a3", finding(5, "serving/monitor.py:30-40", anchors=("HEX",),
                                          depth="surface"))]
    j = q.judge_paths(runs, "architect")
    assert j.record["divergence"] == 1 and j.record["structural"] == 0 and not j.passes


# --------------------------------------------------------------------------- unconfirmed

def test_a_finding_no_second_audit_reports_is_recorded_and_does_not_block(tmp_path):
    runs = [audit(tmp_path, "a1", finding(1, "serving/predict.py:12", anchors=("HEX",),
                                          axis="Composition", **STRUCT)),
            audit(tmp_path, "a2", finding(3, "serving/predict.py:12", anchors=("EJ-62",),
                                          axis="Model", depth="surface")),
            audit(tmp_path, "a3", finding(2, "model/predict.py:12", anchors=("HEX",),
                                          axis="Composition"))]
    j = q.judge_paths(runs, "architect")
    assert j.passes, j.problems
    rec = j.record
    assert rec["structural"] == 0 and rec["unconfirmed"] == {"structural": 1, "divergence": 0}
    row = rec["matching"][0]
    assert row["finding"] == "1:1" and row["reported_by"] == [1] and not row["confirmed"]
    assert row["files"] == ["serving/predict.py"] and row["anchors"] == ["HEX"]


def test_a_one_run_only_finding_of_a_fresh_audit_does_not_block(tmp_path):
    runs = [audit(tmp_path, "a1", finding(1, "domain.py:9", anchors=("DDD-ValueObject",), **STRUCT)),
            audit(tmp_path, "a2", finding(1, "training/loop.py:5", anchors=("EJ-49",), **STRUCT)),
            audit(tmp_path, "a3")]
    j = q.judge_paths(runs, "architect")
    assert j.passes
    assert [(r["finding"], r["reported_by"]) for r in j.record["matching"]] == [("1:1", [1]), ("2:1", [2])]
    assert j.record["unconfirmed"]["structural"] == 2


def test_two_unconfirmed_findings_of_the_same_audit_do_not_confirm_each_other(tmp_path):
    runs = [audit(tmp_path, "a1", finding(1, "ports.py:3", anchors=("HEX",), **STRUCT),
                  finding(2, "ports.py:9", anchors=("HEX",), **STRUCT)),
            audit(tmp_path, "a2"), audit(tmp_path, "a3")]
    assert q.judge_paths(runs, "architect").passes


# --------------------------------------------------------------------------- the procedure

def test_two_audits_are_not_a_judgement(tmp_path):
    runs = [audit(tmp_path, "a1", finding(1, "ports.py:3", **STRUCT)),
            audit(tmp_path, "a2", finding(1, "ports.py:3", **STRUCT))]
    j = q.judge_paths(runs, "architect")
    assert not j.passes and "never another number" in j.record["invalid"][0]


def test_a_run_given_twice_or_copied_is_not_an_independent_audit(tmp_path):
    first = audit(tmp_path, "a1", finding(1, "ports.py:3", **STRUCT))
    j = q.judge_paths([first, first, audit(tmp_path, "a3")], "architect")
    assert any("the same run" in p for p in j.record["invalid"])
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "findings.json").write_bytes((first / "findings.json").read_bytes())
    j = q.judge_paths([first, copy, audit(tmp_path, "a4")], "architect")
    assert any("copied audit" in p for p in j.record["invalid"]) and not j.passes


def test_audits_of_other_code_are_refused(tmp_path):
    runs = [audit(tmp_path, "a1", finding(1, "ports.py:3", **STRUCT)),
            audit(tmp_path, "a2", scope="vega-runs/2026-10-11T00-42Z/landcover_cover"),
            audit(tmp_path, "a3")]
    j = q.judge_paths(runs, "architect")
    assert any("did not audit the same code" in p for p in j.record["invalid"])


def test_one_directory_spelled_absolute_and_relative_is_the_same_code(tmp_path):
    """F42 (release4 rehearsal round 1): improve audit 2 wrote its scope as the absolute path
    of the bundle, audits 1 and 3 project-relative; the quorum compared the strings, refused
    the three as audits of different code, and the session edited the audit's record to pass.
    A path inside the project is read relative to it, in the scope and in a location; another
    directory is still another code."""
    absolute = str((tmp_path / SCOPE).resolve())
    runs = [audit(tmp_path, "a1", finding(1, f"{SCOPE}/ports.py:3", **STRUCT)),
            audit(tmp_path, "a2", finding(1, f"{absolute}/ports.py:9", **STRUCT), scope=absolute),
            audit(tmp_path, "a3")]
    j = q.judge_paths(runs, "architect")
    assert not j.record.get("invalid"), j.record.get("invalid")
    assert j.record["structural"] == 1                 # the two reports are matched, file and anchor
    other = [audit(tmp_path, "b1", finding(1, "ports.py:3", **STRUCT)),
             audit(tmp_path, "b2", scope=str(tmp_path / "vega-runs" / "elsewhere")),
             audit(tmp_path, "b3")]
    assert any("did not audit the same code" in p for p in q.judge_paths(other, "architect").record["invalid"])


def test_a_scope_written_as_a_glob_is_the_code_under_its_root(tmp_path):
    """F43 (release4 rehearsal round 2): architect audit 3 wrote its scope as
    `<package>/**/*.py`, audits 1 and 2 as `<package>/` and `<package>`; the quorum compared the
    strings, refused the three as audits of different code, and the session re-emitted the
    package and ran three more audits. A glob names the code under its literal root; a glob
    under another directory is still another code."""
    runs = [audit(tmp_path, "a1", finding(1, f"{SCOPE}/ports.py:3", **STRUCT), scope=SCOPE + "/"),
            audit(tmp_path, "a2", finding(1, "ports.py:9", **STRUCT), scope=f"{SCOPE}/**/*.py"),
            audit(tmp_path, "a3")]
    j = q.judge_paths(runs, "architect")
    assert not j.record.get("invalid"), j.record.get("invalid")
    assert j.record["structural"] == 1                 # the two reports are matched, file and anchor
    other = [audit(tmp_path, "b1", finding(1, "ports.py:3", **STRUCT)),
             audit(tmp_path, "b2", scope=f"{SCOPE}/model/*.py"),
             audit(tmp_path, "b3")]
    assert any("did not audit the same code" in p for p in q.judge_paths(other, "architect").record["invalid"])
    assert q._literal_root("**/*.py") == "**/*.py" and q._literal_root("a/b") == "a/b"


def test_an_unreadable_run_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(q.QuorumError, match="no findings.json"):
        q.judge_paths([tmp_path / "empty"], "improve")


# --------------------------------------------------------------------------- locations

def test_locations_are_files_relative_to_the_audited_root():
    keys = q.file_keys(f"{SCOPE}/ingestion/data.py:136-148; training/engines.py:136-146; "
                       "ports.py:28 (vs blueprint-runs/2026-10-11T00-00Z/pipeline.json §3)", (SCOPE,))
    assert keys == {"ingestion/data.py", "training/engines.py", "ports.py"}
    assert q.file_keys("domain.py:1078-1100 (LabelSet.from_spec reads netspec.task.classes)") == {"domain.py"}
    assert q.file_keys("the Monitor port as a whole") == {"text:the monitor port as a whole"}
    assert q.file_keys(["serving/a.py:1", {"also": "serving/b.py"}]) == {"serving/a.py", "serving/b.py"}


def test_the_same_file_is_compared_from_the_file_name_up():
    assert q.same_file("domain.py", "landcover_cover/domain.py")
    assert q.same_file("serving/predict.py", "landcover_cover/serving/predict.py")
    assert not q.same_file("model/store.py", "serving/store.py")
    assert not q.same_file("text:the port", "ports.py") and q.same_file("text:the port", "text:the port")


def test_improve_findings_with_justification_maps_and_file_keys_are_read(tmp_path):
    raw = {"id": 3, "axis": "Design", "depth": "structural", "anchors": ["EJ-65"],
           "anchor_justification": {"EJ-65": "reflection"}, "file": "training/loop.py:120-130"}
    one = audit(tmp_path, "i1", raw)
    run = q.load_run(one, 1)
    f = run.findings[0]
    assert f.blocking and f.files == {"training/loop.py"} and f.anchors == {"EJ-65"} and f.category == "design"


# --------------------------------------------------------------------------- the command

def _cli(*args):
    done = subprocess.run([sys.executable, str(SCRIPTS / "audit_quorum.py"), *map(str, args)],
                          capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def test_the_command_says_what_the_stage_does_next(tmp_path):
    clean = audit(tmp_path, "c1", finding(1, "domain.py:1"))
    assert _cli("--tool", "architect", clean)[0] == 0
    first = audit(tmp_path, "a1", finding(1, "ports.py:3", anchors=("HEX",), **STRUCT))
    rc, out = _cli("--tool", "architect", first)
    assert rc == 1 and "NEED:" in out and "fresh context" in out
    second = audit(tmp_path, "a2", finding(7, "ports.py:30", anchors=("HEX",), **STRUCT))
    rc, out = _cli("--tool", "architect", first, second, audit(tmp_path, "a3"))
    assert rc == 2 and "WARN: 1:1" in out and "FORK:" in out and "ERROR" not in out
    assert "fix now" in out and "accept and record" in out and "stop" in out
    rc, out = _cli("--tool", "architect", first, second, audit(tmp_path, "a3b"), "--mode", "unattended")
    assert rc == 2 and "FORK:" not in out and "advisory (unattended)" in out and "the chain continues" in out
    rc, out = _cli("--tool", "architect", first, audit(tmp_path, "b2"), audit(tmp_path, "b3"))
    assert rc == 0 and "unconfirmed" in out
    rc, out = _cli("--tool", "architect", first, second)
    assert rc == 3 and "never another number" in out
    rc, out = _cli("--tool", "architect", first, second, audit(tmp_path, "a4"), "--json", "-")
    assert rc == 2 and json.loads(out)["structural"] == 1



# --------------------------------------------------------------------------- run ids (F31)

def test_two_audits_that_share_a_run_id_are_refused(tmp_path):
    """Two concurrent audits started in the same second write one run id: at one path the
    second overwrites the first; in two project copies they are two runs named alike. Either
    way the judgement is refused — the fresh audits run one after another."""
    first = audit(tmp_path, "20261011T191222Z", finding(1, "ports.py:3", **STRUCT))
    copy_root = tmp_path / "copy"
    twin = audit(copy_root, "20261011T191222Z", finding(4, "domain.py:9", anchors=("EJ-17",), **STRUCT))
    j = q.judge_paths([first, twin, audit(tmp_path, "20261011T192500Z")], "improve")
    assert not j.judged and any("share the run id '20261011T191222Z'" in p for p in j.record["invalid"])
    assert any("one after another" in p for p in j.invalid)
    assert [a["run_id"] for a in j.record["audits"]][0] == "20261011T191222Z"
    rc, out = _cli("--tool", "improve", first, twin, audit(tmp_path, "20261011T193000Z"))
    assert rc == 3 and "share the run id" in out
    # the run id of a findings.json given as a file is its directory's name
    assert q.load_run(first / "findings.json", 1).run_id == "20261011T191222Z"


def test_distinct_run_ids_in_one_minute_are_distinct_audits(tmp_path):
    runs = [audit(tmp_path, "2026-10-11T17-24Z", finding(1, "ports.py:3", **STRUCT)),
            audit(tmp_path, "2026-10-11T17-24Z-b3041059"), audit(tmp_path, "2026-10-11T17-41Z")]
    assert q.judge_paths(runs, "improve").judged


# --------------------------------------------------------------------------- advisory

def _confirmed(tmp_path, tool="architect"):
    runs = [audit(tmp_path, "a1", finding(1, "serving/predict.py:12", anchors=("HEX",), **STRUCT)),
            audit(tmp_path, "a2", finding(4, "serving/predict.py:88", anchors=("HEX",), **STRUCT)),
            audit(tmp_path, "a3")]
    return q.judge_paths(runs, tool).record


def test_unattended_a_confirmed_finding_is_a_warn_and_an_open_item(tmp_path):
    rec = _confirmed(tmp_path)
    advice = q.advise(rec, "unattended", remedy="fix the template and re-emit")
    assert advice.errors == () and len(advice.warns) == 1 and advice.warns[0].startswith("WARN: architect 1:1")
    assert "advisory (unattended)" in advice.warns[0] and "the chain continues" in advice.warns[0]
    item = advice.open_items[0]
    assert item["finding"] == "1:1" and item["for"] == ["/architect", "/improve"]
    assert item["text"].startswith("architect 1:1 (structural, serving/predict.py")
    assert "open item for /architect and /improve: fix the template and re-emit" in item["text"]
    assert advice.record["outcome"] == "warn" and advice.record["confirmed"] == ["1:1"]
    rec["advisory"] = advice.record
    assert q.advisory_problems(rec) == []


def test_attended_a_confirmed_finding_needs_the_users_recorded_answer(tmp_path):
    rec = _confirmed(tmp_path)
    advice = q.advise(rec, "attended")
    assert advice.errors and "no answer is recorded" in advice.errors[0] and "FORK:" in advice.errors[0]
    assert advice.record["outcome"] == "refused" and advice.open_items == () and advice.warns == ()
    default = [{"gate": "architect", "findings": ["1:1"], "asked": "…", "or": "default applied (unattended)"}]
    assert "carries `or`" in q.advise(rec, "attended", default).errors[0]
    partial = [{"gate": "architect", "findings": ["9:9"], "asked": "…", "answer": "accept and record"}]
    assert "names every confirmed finding" in q.advise(rec, "attended", partial).errors[0]
    for answer, said in (("fix now — the template first", "fix now"), ("Stop.", "answered stop")):
        fork = [{"gate": "architect", "findings": ["1:1"], "asked": "…", "answer": answer}]
        assert said in q.advise(rec, "attended", fork).errors[0]
    other_tool = [{"gate": "improve", "findings": ["1:1"], "asked": "…", "answer": "accept and record"}]
    assert q.advise(rec, "attended", other_tool).errors
    accepted = [{"gate": "architect", "findings": ["1:1"], "asked": "…",
                 "answer": "accept and record — the user keeps the predictor as it is"}]
    advice = q.advise(rec, "attended", accepted)
    assert advice.errors == () and advice.record["outcome"] == "accepted"
    assert advice.record["answer"].startswith("accept and record") and len(advice.open_items) == 1
    assert "accepted and recorded (the user)" in advice.warns[0]


def test_nothing_confirmed_is_clear_in_either_mode(tmp_path):
    rec = q.judge_paths([audit(tmp_path, "c1", finding(1, "domain.py:1"))], "improve").record
    for mode in q.MODES:
        advice = q.advise(rec, mode)
        assert advice.errors == advice.warns == advice.open_items == () and advice.record["outcome"] == "clear"
    assert q.advisory_problems(rec) == []
    with pytest.raises(q.QuorumError, match="mode"):
        q.advise(rec, "sometimes")


def test_a_confirmed_record_without_its_advisory_is_malformed(tmp_path):
    rec = _confirmed(tmp_path)
    assert "carries no advisory outcome" in q.advisory_problems(rec)[0]
    rec["advisory"] = dict(q.advise(rec, "unattended").record, confirmed=[], open_items=[])
    problems = q.advisory_problems(rec)
    assert any("not the confirmed findings" in p for p in problems)
    assert any("every confirmed finding is an open item" in p for p in problems)
    rec["advisory"] = dict(q.advise(rec, "unattended").record, outcome="refused")
    assert "not warn (unattended) or accepted (attended)" in q.advisory_problems(rec)[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
