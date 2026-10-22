#!/usr/bin/env python3
"""
test_show_layer.py — the `showcase` layer, its render asset and the served walk (H07:
`apply_show_layer.py`, `show_layer_data.py`, `show_layer_story.py`, `smoke_show.py`,
`smoke_show_layer.py`, `serve_probe.py`, `assets/showcase-render.js`, `assets/showcase-layer.css`;
contract §8, §10 LY1–LY7 and SV1–SV5; D5 §7.8, §7.9, §9).

    python3 -m pytest -q vega-show/scripts/tests/test_show_layer.py
    VEGA_SHOW_LANES=<matrix dir> python3 -m pytest -q …/test_show_layer.py -k lane

- **The layer data**: spliced from the showcase, never from the authoring file — SHOWCASE.json
  verbatim, the headline and lead rendered with their values, the outline, the bindings with
  their records and uses, the report binding, the build check and the served walk, the markers;
  every required key of `layer.required`.
- **The gates** (each a seeded defect, exit 2 with its gate id): LY1 no check recorded, a check
  that is not pass (D5-34), a check older than the build, a pinned file edited after the check;
  LY2 no walk and no marker (VS21), a failed walk whatever the markers say, a walk of another
  build, a walk not run with and without the marker; LY3 a digit in the abstract (D5-33), a short
  abstract, a placeholder in it, a digit inside a name facts.names carries (accepted); LY4 brief
  links without `WARN: no report` (D5-35); LY5 SHOWCASE.json not pinning the bundle's current
  MANIFEST.json; LY6 a render over 40 KB, data over the cap; LY7 --out the input; exit 3 without
  the platform writer.
- **The written page**: validate_model.py invariants 13–16, and `smoke_show.py OUT --strict`'s
  nine checks pass; a seeded render that mounts its pane without its surfaces is caught.
- **The served walk without a browser**: `status: not-run` recorded, WARN and exit 0, exit 2
  under --strict.
- **The mock walk** (`--mock`): page fixture case 01 with its own story passes; a story holding
  a placeholder no fact resolves fails P-TXT.
- **The lanes** (with `VEGA_SHOW_LANES`, a matrix whose `lanes/L01` and `lanes/L02` went through
  `report`): extract → draft → author → resolve → build → check on a copy, `smoke_show.py --serve
  --strict` (every view, no page error, no unresolved placeholder, `/v1/showcase` verified, links
  at their pinned digests), the layer, validate_model 13–16, `smoke_show.py OUT --strict`, the
  bundle's MANIFEST.json unchanged; and the walk refusing a `story.json` edited after the build
  (the slot unverified: SV1).
"""
from __future__ import annotations

import contextlib
import functools
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SCRIPTS))

import apply_show_layer  # noqa: E402
import cdp  # noqa: E402
import show_layer_story as sls  # noqa: E402
import smoke_show  # noqa: E402
import test_build_showcase as tbs  # noqa: E402
from show_layer_data import ShowLayer  # noqa: E402
from vega_show_core import CATALOG, digest_bytes, now, write_json  # noqa: E402

SKILLS = TESTS.parents[2]
DOMAIN_FORGE = Path(os.environ.get("DOMAIN_FORGE_DIR") or SKILLS / "domain-forge")
VALIDATE = DOMAIN_FORGE / "scripts" / "validate_model.py"
CHROME = cdp.find_chrome()
ABSTRACT = ("The page shows a reader outside the chain how the network reads each row and how "
            "it did on the sealed test part.")
HOST = ('<body>\n<nav class="tabs" role="tablist"><button data-tab="ontology" class="active" role="tab">'
        'Ontology</button></nav>\n<section class="tab-pane active" id="tab-ontology" data-tab="ontology">'
        '<p>model</p></section>\n')


def run(main, argv) -> tuple[int, str]:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = main([str(a) for a in argv])
    return code, printed.getvalue()


@functools.cache
def _built(linked: bool) -> Path:
    p = tbs.project(Path(tempfile.mkdtemp(prefix="show-layer-")) / "p", linked)
    code, printed = p.build(*([] if linked else ["--unattended"]))
    assert code == 0, printed
    return p.root


@pytest.fixture
def proj(tmp_path):
    """A copy of the built brief-link project, its check recorded, its model a host page."""
    return _project(tmp_path, linked=False)


@pytest.fixture
def linked(tmp_path):
    return _project(tmp_path, linked=True)


def _project(tmp_path: Path, linked: bool) -> SimpleNamespace:
    root = tmp_path / "p"
    shutil.copytree(_built(linked), root, symlinks=True)
    model = root / tbs.MODEL
    model.write_text(model.read_text(encoding="utf-8").replace("<body>\n", HOST, 1), encoding="utf-8")
    p = SimpleNamespace(root=root, model=model, bundle=root / "bundle" / "demo-R0",
                        report=root / "docs" if linked else None)
    p.showcase = p.bundle / "showcase"
    p.data = p.showcase / ".work" / "showcase-layer.json"
    p.out = root / (tbs.MODEL[:-len(".html")] + ".show.html")
    check(p)
    author(p, markers=["WARN: no report — code notes link to the stage briefs; re-run /vega-show after "
                       "/document-project"] if not linked else [])
    smoke(p, "pass")
    return p


def check(p) -> int:
    with contextlib.chdir(p.root):
        code, _ = run(tbs.build_showcase.main, ["--check", p.showcase, *(["--report", p.report] if p.report else [])])
    return code


def author(p, abstract: str = ABSTRACT, markers=(), **extra) -> None:
    doc = {"from_showcase": str(p.showcase), "readings": {"abstract": abstract}, "markers": list(markers), **extra}
    p.data.write_text(json.dumps(doc), encoding="utf-8")


def smoke(p, status: str, digest: str | None = None, why: str | None = None) -> None:
    """A recorded served walk (the walk itself is proven on the lanes)."""
    raw = (p.showcase / "SHOWCASE.json").read_bytes()
    checks = [{"id": "SV1", "ok": status == "pass", "problems": [] if status == "pass" else ["seeded"]}]
    write_json(p.showcase / ".smoke" / "smoke.json", {
        "schema": "vega/show-smoke@1", "status": status, "why_not_run": why, "checked_at": now(),
        "showcase_digest": digest or digest_bytes(raw), "verified": status == "pass",
        "views": ["start", "context"], "page_errors": 0, "placeholders_unresolved": [], "links": [],
        "checks": checks if status != "not-run" else [], "screenshots": []})


def apply(p, *extra) -> tuple[int, str]:
    return run(apply_show_layer.main, [p.model, "--data", p.data, "--out", p.out, *extra])


def refused(p, gate: str, *extra) -> str:
    code, printed = apply(p, *extra)
    assert code == 2, printed
    assert f"ERROR: {gate} " in printed, printed
    assert not p.out.exists()
    return printed


def layer_doc(p) -> dict:
    layer = ShowLayer.load(p.data)
    assert layer.problems() == []
    return layer.document()


# =============================================================================== the story summary

STORY = {"values": {"primary": {"value": 0.8178, "ci": [0.7, 0.9], "n": 9}, "rows": 2830743, "lr": 0.0023,
                    "share": 0.256, "diff": -0.05, "units": [1, 2, 3]},
         "formats": {"primary": "ci"}, "refs": {"primary": "records/evaluation.json#metrics.x", "rows": "r#a"},
         "lexicon": {"row": {"one": "flow", "many": "flows", "article": "a"},
                     "source": {"one": "capture file", "many": "capture files", "article": "the"}},
         "headline": "Each {row.one} of {rows:int}: {primary}",
         "lead": "{row.a}, {source.the}, {share:pct}, {diff:signed}, {units:count}, {lr} and {target.one}",
         "sections": [{"id": "context", "steps": [{"text": "{rows} {row.many}"}], "facts": [1, 2]},
                      {"id": "results", "headline": "{primary}", "models": [1], "limits": [1, 2, 3]}],
         "input_groups": [{"id": "g", "label": "timing", "members": ["a", "b"]}],
         "class_roles": {"roles": {"A": "neutral"}, "source": {"A": "structural:majority"}}, "polarity": "neutral"}


def test_the_headline_and_lead_are_rendered_as_the_page_writes_them():
    head, missed = sls.render_text(STORY["headline"], STORY)
    assert head == "Each flow of 2,830,743: 0.8178 [0.7, 0.9]" and missed == []
    lead, missed = sls.render_text(STORY["lead"], STORY)
    assert lead == ("a flow, the capture file, 25.6 %, −0.050, 3, 0.0023 and {target.one}")
    assert missed == ["{target.one}"]


def test_every_value_is_a_binding_with_its_record_and_its_uses():
    rows = {b["key"]: b for b in sls.bindings(STORY)}
    assert list(rows) == sorted(STORY["values"])
    assert rows["primary"]["ref"] == "records/evaluation.json#metrics.x"
    assert rows["primary"]["used_in"] == ["headline", "sections.results.headline"]
    assert rows["rows"]["used_in"] == ["headline", "sections.context.steps[0].text"]
    assert rows["lr"]["used_in"] == ["lead"] and rows["lr"]["ref"] is None


def test_the_summary_counts_the_sections_and_the_groups():
    s = sls.story_summary(STORY, {"motif": {"id": "points", "source": "vega:default"}})
    assert s["sections"] == [{"id": "context", "items": {"steps": 1, "facts": 2}},
                             {"id": "results", "items": {"models": 1, "limits": 3}}]
    assert s["input_families"] == [{"id": "g", "label": "timing", "members_n": 2}]
    assert s["motif"] == "points" and s["unresolved"] == ["{target.one}"]


# =============================================================================== the layer data

def test_the_layer_is_spliced_from_the_showcase(proj):
    doc = layer_doc(proj)
    sc = json.loads((proj.showcase / "SHOWCASE.json").read_text(encoding="utf-8"))
    story = json.loads((proj.showcase / "story.json").read_text(encoding="utf-8"))
    assert doc["schema"] == CATALOG["layer"]["schema"] and doc["showcase"] == sc
    assert [b["key"] for b in doc["bindings"]] == sorted(story["values"])
    assert doc["story"]["headline"] == sls.render_text(story["headline"], story)[0]
    assert doc["report"]["status"] == "brief-links" and doc["report"]["briefs"]
    assert doc["checks"]["build"]["status"] == "pass" and doc["checks"]["served"]["status"] == "pass"
    assert doc["usage"]["url"].startswith("http://127.0.0.1:") and doc["usage"]["serve"]
    assert any(m.startswith("build_showcase.py: ") for m in doc["markers"])
    assert ShowLayer.load(proj.data).missing_keys(doc) == []


def test_a_linked_showcase_carries_its_roles(linked):
    doc = layer_doc(linked)
    assert doc["report"]["status"] == "linked" and doc["report"]["bound"] and doc["report"]["pdf_digest"]


def test_the_layer_is_written_through_the_platform_writer(proj):
    code, printed = apply(proj)
    assert code == 0, printed
    html = proj.out.read_text(encoding="utf-8")
    assert "@LAYER:start showcase" in html and "produced-by: /vega-show" in html
    assert 'id="layer-showcase-render"' in html and "layer-showcase-noscript" in html
    assert html.startswith(proj.model.read_text(encoding="utf-8").split("</body>")[0][:200])


# =============================================================================== the gates

def test_ly1_no_check_recorded(proj):
    (proj.showcase / ".work" / "checks.json").unlink()
    refused(proj, "LY1")


def test_ly1_d5_34_a_stale_check_is_refused(proj):
    path = proj.showcase / ".work" / "checks.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(dict(doc, status="stale", problems=[{"rule": "CK7", "where": "report.pdf",
                                                                     "message": "seeded"}])), encoding="utf-8")
    refused(proj, "LY1")


def test_ly1_a_check_older_than_the_build_is_refused(proj):
    path = proj.showcase / ".work" / "checks.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text(encoding="utf-8")), checked_at="2000-01-01T00:00:00Z")),
                    encoding="utf-8")
    refused(proj, "LY1")


def test_ly1_a_pinned_file_edited_after_the_check_is_refused(proj):
    page = proj.showcase / "page.json"
    page.write_bytes(page.read_bytes() + b" ")
    refused(proj, "LY1")


def test_ly2_vs21_no_walk_and_no_marker(proj):
    (proj.showcase / ".smoke" / "smoke.json").unlink()
    refused(proj, "LY2")


def test_ly2_a_walk_not_run_needs_the_marker(proj):
    smoke(proj, "not-run", why="no headless browser")
    refused(proj, "LY2")
    author(proj, markers=["WARN: no report — briefs", "WARN: served page not checked — no headless browser"])
    code, printed = apply(proj)
    assert code == 0, printed
    served = layer_doc(proj)["checks"]["served"]
    assert served["status"] == "not-run" and served["why_not_run"] == "no headless browser"


def test_ly2_a_failed_walk_is_refused_whatever_the_markers_say(proj):
    smoke(proj, "fail")
    author(proj, markers=["WARN: no report — briefs", "WARN: served page not checked"])
    refused(proj, "LY2")


def test_ly2_a_walk_of_another_build_is_refused(proj):
    smoke(proj, "pass", digest="sha256:" + "0" * 64)
    refused(proj, "LY2")


@pytest.mark.parametrize("abstract", ["The page shows the network to its 3 readers and how it did on the test.",
                                      "Too short to say anything.",
                                      "The page shows {primary} and how it did on the sealed test part for readers."])
def test_ly3_d5_33_the_abstract_is_prose_without_a_digit(proj, abstract):
    author(proj, abstract=abstract, markers=["WARN: no report"])
    refused(proj, "LY3")


def test_ly3_a_name_the_facts_carry_may_hold_a_digit(proj):
    facts = json.loads((proj.showcase / "plan" / "facts.json").read_text(encoding="utf-8"))
    names = [n for n in facts.get("names") or [] if any(c.isdigit() for c in n) and any(c.isalpha() for c in n)]
    if not names:
        pytest.skip("the fixture's facts carry no name with a digit")
    author(proj, abstract=f"The page shows how run {names[0]} reads each row and how it did on the test part.",
           markers=["WARN: no report"])
    code, printed = apply(proj)
    assert code == 0, printed


def test_ly4_d5_35_brief_links_without_the_marker(proj):
    author(proj, markers=[])
    refused(proj, "LY4")


def test_ly5_a_showcase_that_does_not_pin_the_current_manifest(proj):
    manifest = proj.bundle / "MANIFEST.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    refused(proj, "LY5")


def test_ly6_a_render_over_its_cap(proj, tmp_path):
    big = tmp_path / "big.js"
    big.write_text("/*" + "x" * (CATALOG["layer"]["max_render_kb"] * 1024) + "*/", encoding="utf-8")
    code, printed = apply(proj, "--render", big)
    assert code == 2 and "LY6" in printed and not proj.out.exists()


def test_ly6_data_over_its_cap(proj, monkeypatch):
    import show_layer_data
    monkeypatch.setattr(show_layer_data, "MAX_DATA", 1024)
    refused(proj, "LY6")


def test_ly7_out_is_the_input(proj):
    code, printed = run(apply_show_layer.main, [proj.model, "--data", proj.data, "--out", proj.model])
    assert code == 2 and "LY7" in printed


def test_exit_3_without_the_platform_writer(proj, tmp_path):
    code, printed = apply(proj, "--domain-forge-dir", tmp_path / "nowhere")
    assert code == 3 and "platform writer" in printed and not proj.out.exists()


# =============================================================================== the written page

needs_browser = pytest.mark.skipif(CHROME is None, reason="no headless browser")


@needs_browser
def test_the_written_page_validates_and_its_tab_passes_the_nine_checks(proj):
    code, printed = apply(proj)
    assert code == 0, printed
    # the fixture's other layers are stand-ins that mount nothing: validate the showcase alone
    sys.path.insert(0, str(DOMAIN_FORGE / "scripts"))
    import apply_layer
    html = proj.out.read_text(encoding="utf-8")
    for layer in apply_layer.list_layers(html):
        if layer["name"] != "showcase":
            html = apply_layer.strip_layer(html, layer["name"])
    alone = proj.out.with_name("alone.html")
    alone.write_text(html, encoding="utf-8")
    v = subprocess.run([sys.executable, str(VALIDATE), str(alone)], capture_output=True, text=True, timeout=600)
    for n in (13, 14, 15, 16):
        assert f"PASS  invariant {n}:" in v.stdout, v.stdout[-3000:]
    code, printed = run(smoke_show.main, [proj.out, "--strict"])
    assert code == 0, printed
    assert "9 check(s) passed" in printed


@needs_browser
def test_a_render_without_its_surfaces_fails_the_smoke(proj, tmp_path):
    stub = tmp_path / "stub.js"
    stub.write_text("(function(){var p=document.createElement('section');p.className='tab-pane';"
                    "p.setAttribute('data-layer','showcase');p.setAttribute('data-tab','showcase');"
                    "document.body.appendChild(p);})();", encoding="utf-8")
    code, printed = apply(proj, "--render", stub)
    assert code == 0, printed
    code, printed = run(smoke_show.main, [proj.out, "--strict"])
    assert code == 1
    for check in ("1", "3", "4", "7"):
        assert f": {check}: " in printed, printed


def test_the_walk_without_a_browser_is_recorded_not_run(proj):
    a = SimpleNamespace(serve=str(proj.bundle), python=sys.executable, port=None, strict=False, json=False)
    code, printed = run(lambda argv: smoke_show.run_serve(a, None), [])
    assert code == 0 and "WARN: served page not checked" in printed
    doc = json.loads((proj.showcase / ".smoke" / "smoke.json").read_text(encoding="utf-8"))
    assert doc["status"] == "not-run" and doc["why_not_run"]
    code, printed = run(lambda argv: smoke_show.run_serve(SimpleNamespace(**dict(vars(a), strict=True)), None), [])
    assert code == 2 and "ERROR: served page not checked" in printed


def test_a_re_walk_that_finds_the_same_leaves_the_pinned_record_alone(proj):
    """release3 final3r1: the show brief pins `.smoke/smoke.json`; a re-walk that found the same
    thing (only its time, probe digest and log path differ) keeps the record byte for byte."""
    a = SimpleNamespace(serve=str(proj.bundle), python=sys.executable, port=None, strict=False, json=False)
    record = proj.showcase / ".smoke" / "smoke.json"
    record.unlink(missing_ok=True)
    run(lambda argv: smoke_show.run_serve(a, None), [])
    first = record.read_bytes()
    time.sleep(1.1)
    run(lambda argv: smoke_show.run_serve(a, None), [])
    assert record.read_bytes() == first
    old = json.loads(first)
    paths = SimpleNamespace(smoke=record.parent)
    moved = dict(old, checked_at="2999-01-01T00:00:00Z", probe_digest="sha256:0", service_log="/x")
    assert smoke_show._recorded_walk(paths, moved) == old
    assert smoke_show._recorded_walk(paths, dict(old, status="fail")) is None


# =============================================================================== the lanes

LANES = os.environ.get("VEGA_SHOW_LANES")
needs_lanes = pytest.mark.skipif(not LANES or CHROME is None, reason="VEGA_SHOW_LANES and a browser are needed")


def _lane(tmp_path: Path, lane: str) -> SimpleNamespace:
    src = Path(LANES) / "lanes" / lane
    if not (src / "work").is_dir():
        pytest.skip(f"{src} has no work dir")
    dst = tmp_path / lane
    shutil.copytree(src, dst, symlinks=True)
    work = dst / "work"
    bundle = next(d for d in sorted((work / "bundle").iterdir()) if (d / "MANIFEST.json").is_file())
    model = next(iter(sorted(work.glob("*.pack.html"))))
    plan, docs = dst / "show" / "plan", dst / "docs"
    rep = ["--report", docs] if (docs / "label-roles.json").is_file() else []
    steps = [(tbs.extract_facts.main, ["--model", model, "--bundle", bundle, *rep, "--out", plan / "facts.json"]),
             (tbs.draft_plan.main, ["--facts", plan / "facts.json", "--out", plan / "draft.json", "--unattended"]),
             (tbs.autoauthor.main, ["--plan", plan / "draft.json", "--facts", plan / "facts.json",
                                    "--out", plan / "story-plan.json"]),
             (tbs.resolve_story.main, ["--plan", plan / "story-plan.json", "--facts", plan / "facts.json", *rep,
                                       "--out", plan / "story.json"]),
             (tbs.build_showcase.main, ["--bundle", bundle, "--model", model, "--plan", plan / "story-plan.json",
                                        "--story", plan / "story.json", "--facts", plan / "facts.json", *rep,
                                        "--out", bundle / "showcase", "--unattended"]),
             (tbs.build_showcase.main, ["--check", bundle / "showcase", *rep])]
    for main, argv in steps:
        with contextlib.chdir(work):
            code, printed = run(main, argv)
        assert code == 0, printed
    return SimpleNamespace(bundle=bundle, model=model, showcase=bundle / "showcase",
                           data=bundle / "showcase" / ".work" / "showcase-layer.json",
                           out=model.with_name(model.name[:-len(".html")] + ".show.html"), linked=bool(rep))


@needs_lanes
@pytest.mark.parametrize("lane", ["L01", "L02"])
def test_lane_the_served_walk_the_layer_and_the_tab(tmp_path, lane):
    p = _lane(tmp_path, lane)
    before = digest_bytes((p.bundle / "MANIFEST.json").read_bytes())
    code, printed = run(smoke_show.main, ["--serve", p.bundle, "--strict"])
    assert code == 0, printed
    walk = json.loads((p.showcase / ".smoke" / "smoke.json").read_text(encoding="utf-8"))
    assert walk["status"] == "pass" and walk["verified"] and walk["page_errors"] == 0
    assert walk["placeholders_unresolved"] == [] and walk["links"] and all(l["ok"] for l in walk["links"])
    assert set(CATALOG["views"]["tour"]) <= set(walk["views"])
    assert all((p.bundle / s).is_file() for s in walk["screenshots"])
    author(p, markers=[] if p.linked else ["WARN: no report"])
    code, printed = apply(p)
    assert code == 0, printed
    v = subprocess.run([sys.executable, str(VALIDATE), str(p.out)], capture_output=True, text=True, timeout=600)
    for n in (13, 14, 15, 16):
        assert f"PASS  invariant {n}:" in v.stdout, v.stdout[-3000:]
    code, printed = run(smoke_show.main, [p.out, "--strict"])
    assert code == 0, printed
    assert digest_bytes((p.bundle / "MANIFEST.json").read_bytes()) == before


@needs_lanes
def test_lane_a_story_edited_after_the_build_fails_the_walk(tmp_path):
    p = _lane(tmp_path, "L02")
    story = p.showcase / "story.json"
    story.write_bytes(story.read_bytes().replace(b'"headline": "', b'"headline": "{undefined_fact} ', 1))
    code, printed = run(smoke_show.main, ["--serve", p.bundle, "--strict"])
    assert code == 1, printed
    walk = json.loads((p.showcase / ".smoke" / "smoke.json").read_text(encoding="utf-8"))
    sv1 = next(c for c in walk["checks"] if c["id"] == "SV1")
    assert walk["status"] == "fail" and not sv1["ok"] and not walk["verified"]
    author(p, markers=["WARN: served page not checked"])
    refused(p, "LY1")                                   # the edit broke the pinned digest too


# =============================================================================== the mock walk

@functools.cache
def _case() -> Path:
    out = Path(tempfile.mkdtemp(prefix="show-mock-"))
    import make_page_fixtures
    assert make_page_fixtures.main(["--out", str(out), "--cases", "01-*"]) == 0
    return next(d for d in out.iterdir() if (d / "case.json").is_file())


def _mock_showcase(case: Path, root: Path) -> Path:
    showcase = root / "showcase"
    (showcase / "data").mkdir(parents=True)
    for name in ("story.json", "page.json"):
        shutil.copyfile(case / name, showcase / name)
    shutil.copyfile(case / "attribution.json", showcase / "data" / "attribution.json")
    return showcase


@needs_browser
def test_the_mock_walk_passes_on_a_case_and_its_own_story(tmp_path):
    case = _case()
    code, printed = run(smoke_show.main, ["--mock", case, "--showcase", _mock_showcase(case, tmp_path), "--strict"])
    assert code == 0, printed


@needs_browser
def test_the_mock_walk_catches_an_unresolved_placeholder(tmp_path):
    case = _case()
    showcase = _mock_showcase(case, tmp_path)
    story = json.loads((showcase / "story.json").read_text(encoding="utf-8"))
    story["headline"] = "{no_such_fact} " + story["headline"]
    (showcase / "story.json").write_text(json.dumps(story), encoding="utf-8")
    code, printed = run(smoke_show.main, ["--mock", case, "--showcase", showcase, "--strict"])
    assert code == 1 and "P-TXT: unresolved placeholder {no_such_fact}" in printed, printed
