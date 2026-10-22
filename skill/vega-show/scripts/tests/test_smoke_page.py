#!/usr/bin/env python3
"""
test_smoke_page.py — the page driver catches what it is there to catch (BUILD-PLAN W15; D4 §17.5,
§17.7): cases of the page fixture matrix (02, a binary row network; 16, a forecast that ships a
monitor reference; 01, whose code notes link a stage brief; 04, a multilabel network with
per-label thresholds; 25, a gcn over a node-level graph) driven by `smoke_page.py --fixture`
as built, then with defects seeded, each of which must fail the named check and nothing must
pass it silently.

    python3 -m pytest -q test_smoke_page.py          # needs Chrome ($CHROME) and the backend

| seeded defect                                                   | caught by          |
|-----------------------------------------------------------------|--------------------|
| none                                                            | every check passes |
| PN8: page.json's lane link `softmax` against a binary network   | P-PROBE (the page refuses to boot: the lane cross-check) |
| a strings key the page names removed from the template copy     | P-TXT (missing key) |
| a preset's explanation missing from the fixture                 | P-ERR (the fixture's 422), P-STEP (the preset unanswered) |
| the results view throwing when it shows                         | P-ERR (a page error) |
| a stepper explainer whose recompute is off by far more than 1e-3 | P-STEP (the gap rule) |
| the "every weight" switch withheld (NetCanvas.hasDetail false), and the binary weights map's negative side dropped | P-NET (a stage kind declares the level), P-AXIS (the binary sides) |
| a forecast's truth sent to the drift monitor (case 16)           | P-ERR (the contract's 422, a fixture miss), P-MANY |
| the stage brief a code note links, gone (case 01)                | P-LINKS (404), P-ERR |
| a context fact's label rendered raw, not through the story text | P-TXT (a `{row.many}` token shown) |
| the results view withholding the verdict the story records      | P-TXT (the verdict not said) |
| the output step deciding on the single threshold (case 04, per-label thresholds) | P-STEP (`decision`) |
| the try view not naming the attribution's method                 | P-STEP (`method_unnamed`) |
| the phone rule for a bar's value gone, at 390 × 844             | P-TXT (a value cut off) |
| a graph's reference node named by its index in the assembled graph (case 25, node level) | P-STEP (`node_index_named`) |
| jsonschema not importable                                        | P-PROBE under --strict; a `WARN:` line otherwise |
"""
from __future__ import annotations

import builtins
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
TEMPLATE = SCRIPTS.parent / "assets" / "page"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

import cdp  # noqa: E402
import smoke_page  # noqa: E402

CASE = "02-mlp-plain-binary"
FORECAST, LINKED, LABELS = "16-cnn1d-forecast", "01-mlp-plain-multiclass", "04-mlp-residual-multilabel"
NODES = "25-gcn-node-multiclass"
pytestmark = pytest.mark.skipif(cdp.find_chrome() is None, reason="no Chrome ($CHROME)")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import make_page_fixtures
    out = tmp_path_factory.mktemp("fixtures")
    assert make_page_fixtures.main(["--cases", "1,2,4,16,25", "--out", str(out)]) == 0
    return out


def drive(tmp: Path, fixtures: Path, template: Path = TEMPLATE, case: str = "2", extra: tuple = ()) -> tuple[int, dict]:
    out = tmp / "smoke.json"
    code = smoke_page.main(["--fixture", case, "--fixtures", str(fixtures), "--template", str(template),
                            "--work", str(tmp / "work"), "--json", str(out), "--strict", "--jobs", "1", *extra])
    return code, json.loads(out.read_text(encoding="utf-8"))["results"][0]


def failed(result: dict) -> dict:
    return {c["id"]: c["problems"] for c in result["checks"] if not c["ok"]}


def copy_case(built: Path, tmp: Path, case: str = CASE) -> Path:
    root = tmp / "fixtures"
    shutil.copytree(built / case, root / case)
    return root


def copy_template(tmp: Path) -> Path:
    dst = tmp / "template"
    shutil.copytree(TEMPLATE, dst, ignore=shutil.ignore_patterns("tests", "node_modules"))
    return dst


def edit_json(path: Path, change) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def edit_text(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{path.name}: the seed's anchor {old!r} is gone"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_the_case_as_built_passes_every_check(built, tmp_path):
    code, result = drive(tmp_path, built)
    assert code == 0 and result["ok"], failed(result)
    assert len(result["checks"]) == len(smoke_page.judge.__globals__["CHECKS"])


def test_pn8_a_lane_link_against_the_network_stops_the_page(built, tmp_path):
    root = copy_case(built, tmp_path)
    edit_json(root / CASE / "page.json", lambda d: d["lane"].update(link="softmax"))
    code, result = drive(tmp_path, root)
    assert code == 1
    assert "P-PROBE" in failed(result), failed(result)
    assert result["probe"]["lane"]["ok"] is False


def test_a_missing_strings_key_is_caught(built, tmp_path):
    template = copy_template(tmp_path)
    fragment = template / "js" / "core" / "strings" / "core.en.js"
    text = fragment.read_text(encoding="utf-8")
    key = re.search(r'^\s*"(try\.run)": ".*",?$', text, re.M)
    assert key, "the seed's key try.run is gone from core.en.js"
    fragment.write_text(text.replace(key.group(0) + "\n", "", 1), encoding="utf-8")
    code, result = drive(tmp_path, built, template)
    assert code == 1
    assert any("try.run" in p for p in failed(result).get("P-TXT", [])), failed(result)


def test_an_explanation_the_fixture_cannot_give_is_caught(built, tmp_path):
    root = copy_case(built, tmp_path)
    edit_json(root / CASE / "answers.json", lambda d: d["explain"].pop(next(iter(d["explain"]))))
    code, result = drive(tmp_path, root)
    bad = failed(result)
    assert code == 1
    assert any("no answer" in p for p in bad.get("P-ERR", [])), bad
    assert bad.get("P-STEP"), bad


def test_a_view_that_throws_is_caught(built, tmp_path):
    template = copy_template(tmp_path)
    results = template / "js" / "views" / "results.js"
    text = results.read_text(encoding="utf-8")
    m = re.search(r"\nfunction render\(([^)]*)\) \{\n", text)
    assert m, "the seed's anchor (the results view's render) is gone"
    results.write_text(text.replace(m.group(0), m.group(0) + '  throw new Error("seeded");\n', 1), encoding="utf-8")
    code, result = drive(tmp_path, built, template)
    assert code == 1
    assert any("seeded" in p for p in failed(result).get("P-ERR", [])), failed(result)


def test_a_recompute_off_the_network_is_caught(built, tmp_path):
    template = copy_template(tmp_path)
    sum_js = template / "js" / "how" / "sum.js"
    edit_text(sum_js, "export function gapOf(page, network) {", "export function gapOf(page0, network) {\n  const page = page0 + 1;")
    code, result = drive(tmp_path, built, template)
    assert code == 1
    assert any(p.endswith("|gap") for p in failed(result).get("P-STEP", [])), failed(result)


def test_a_withheld_detail_switch_and_a_wrong_weights_axis_are_caught(built, tmp_path):
    template = copy_template(tmp_path)
    canvas = template / "js" / "net" / "canvas.js"
    edit_text(canvas, "get hasDetail() {", "get hasDetail() { return false; }\n  get hasDetailAsBuilt() {")
    binary = template / "js" / "views" / "tasks" / "binary.js"
    edit_text(binary, "return namedAxis([negative, positive]", "return namedAxis([positive]")
    code, result = drive(tmp_path, built, template)
    bad = failed(result)
    assert code == 1
    assert any("offers no switch" in p for p in bad.get("P-NET", [])), bad
    assert any("expected ['0', '1']" in p for p in bad.get("P-AXIS", [])), bad


def test_a_forecast_truth_sent_to_the_monitor_is_caught(built, tmp_path):
    case = drive(tmp_path / "as-built", built, case="16")[1]
    assert case["ok"], failed(case)
    assert (case["probe"]["sub_views"]["many"] or {}).get("monitor"), "the drift readings are shown"
    template = copy_template(tmp_path)
    window = template / "js" / "views" / "shapes" / "window.js"
    edit_text(window, "truth: one ? truth : null", "truth")
    code, result = drive(tmp_path, built, template, case="16")
    bad = failed(result)
    assert code == 1
    assert any("monitor request breaks its contract" in p for p in bad.get("P-ERR", [])), bad
    assert bad.get("P-MANY"), bad


def test_a_report_link_that_does_not_answer_is_caught(built, tmp_path):
    root = copy_case(built, tmp_path, LINKED)
    (root / LINKED / "briefs" / "net-brief.md").unlink()
    code, result = drive(tmp_path, root, case="1")
    bad = failed(result)
    assert code == 1
    assert any("answered 404" in p for p in bad.get("P-LINKS", [])), bad


def test_a_story_string_rendered_raw_is_caught(built, tmp_path):
    template = copy_template(tmp_path)
    edit_text(template / "js" / "views" / "context.js", 'storyText(f.label, el("span", { class: "l caps" }))', 'el("span", { class: "l caps" }, f.label)')
    code, result = drive(tmp_path, built, template)
    bad = failed(result)
    assert code == 1
    assert any("view context shows" in p and "{row.many}" in p for p in bad.get("P-TXT", [])), bad


def test_a_verdict_the_view_withholds_is_caught(built, tmp_path):
    template = copy_template(tmp_path)
    edit_text(template / "js" / "views" / "results.js", "const host = $(\"res-verdict\"), w = verdictWords(sec, vals);",
              "const host = $(\"res-verdict\"), w = notAssessable(sec) ? verdictWords(sec, vals) : null;")
    code, result = drive(tmp_path, built, template)
    bad = failed(result)
    assert code == 1
    assert any("does not say the verdict" in p for p in bad.get("P-TXT", [])), bad


def test_a_step_deciding_on_the_single_threshold_is_caught(built, tmp_path):
    case = drive(tmp_path / "as-built", built, case="4")[1]
    assert case["ok"], failed(case)
    template = copy_template(tmp_path)
    edit_text(template / "js" / "how" / "explainers" / "link.js", "if (o.perLabel && Number.isFinite(o.perLabel[name]))",
              "if (false && o.perLabel && Number.isFinite(o.perLabel[name]))")
    code, result = drive(tmp_path, built, template, case="4")
    bad = failed(result)
    assert code == 1
    assert any(p.endswith("/link|decision") for p in bad.get("P-STEP", [])), bad


def test_an_unnamed_attribution_method_is_caught(built, tmp_path):
    template = copy_template(tmp_path)
    edit_text(template / "js" / "views" / "shapes" / "row.js", "methodNote(ex && ex.attribution), bars", "bars")
    code, result = drive(tmp_path, built, template)
    bad = failed(result)
    assert code == 1
    assert any("method_unnamed" in p for p in bad.get("P-STEP", [])), bad


def test_a_value_cut_at_phone_width_is_caught(built, tmp_path):
    """At 390 × 844 a results value with a four-digit interval ("0.6651 [0.6052, 0.7253]", as a
    real evaluation writes it) stays on screen; with the phone rule for a bar's value removed it
    runs past the viewport's edge and P-TXT says so."""
    phone = ("--viewport", "390x844", "--mobile")
    root = copy_case(built, tmp_path)

    def long(doc):
        for key, v in doc["values"].items():
            if isinstance(v, dict) and "ci" in v and key.startswith(("eval_", "primary")):
                v.update(value=round(v["value"] + 0.00012, 4), ci=[round(v["ci"][0] + 0.00023, 4), round(v["ci"][1] + 0.00031, 4)])
    edit_json(root / CASE / "story.json", long)
    case = drive(tmp_path / "as-built", root, extra=phone)[1]
    assert case["ok"], failed(case)
    template = copy_template(tmp_path)
    sheet = template / "css" / "views.css"
    text = sheet.read_text(encoding="utf-8")
    start = text.index("@media (max-width:600px){")
    sheet.write_text(text[:start] + text[text.index("}\n}", start) + 3:], encoding="utf-8")   # the phone rule, gone
    code, result = drive(tmp_path, root, template, extra=phone)
    bad = failed(result)
    assert code == 1
    assert any("cuts off a value" in p for p in bad.get("P-TXT", [])), bad


def test_a_graph_node_named_by_its_index_is_caught(built, tmp_path):
    """A node-level graph answer's node ids are indices in the graph the service assembled: as
    built, the try view and every stepper step name a reference node by the key the service sent or
    by its rank in the answer; with the namer giving the index instead ("node 37"), P-STEP says so."""
    root = copy_case(built, tmp_path, NODES)
    case = drive(tmp_path / "as-built", root, case="25")[1]
    assert case["ok"], failed(case)
    template = copy_template(tmp_path)
    edit_text(template / "js" / "core" / "node-names.js",
              'return rank.has(id) ? tText("graph.node.ref", { k: rank.get(id) }) : tText("graph.node.other");',
              'return tText("graph.node.nth", { k: id });')
    code, result = drive(tmp_path, root, template, case="25")
    bad = failed(result)
    assert code == 1
    assert any("node_index_named" in p for p in bad.get("P-STEP", [])), bad


def test_an_unvalidated_probe_output_fails_only_under_strict(monkeypatch, capsys):
    import smoke_page_checks as checks
    real = builtins.__import__

    def no_jsonschema(name, *args, **kwargs):
        if name == "jsonschema" or name.startswith("jsonschema."):
            raise ImportError(name)
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_jsonschema)
    monkeypatch.setattr(checks, "_warned", [])
    assert checks.schema_problems({}, strict=True) == [checks.NO_VALIDATOR]
    assert checks.schema_problems({}, strict=False) == []
    assert "WARN: " + checks.NO_VALIDATOR in capsys.readouterr().err
