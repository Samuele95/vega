#!/usr/bin/env python3
"""
test_page_smoke.py — the lanes' page stage catches what it is there to catch (BUILD-PLAN T09;
D6 §10, VS26): its own checks on seeded readings, and the pseudo-locale check in a browser on a
case of the page fixture matrix (02, a binary row network) served with its story wrapped as
build_showcase wraps a qps-ploc build.

    python3 -m pytest -q test_page_smoke.py          # the browser cases need Chrome ($CHROME)

| seeded defect                                                       | caught by  |
|---------------------------------------------------------------------|------------|
| none (the fixture case as built, under qps-ploc)                    | nothing    |
| a template string outside t(): a literal in the results view (VS26) | P-QPS, naming p#res-ci-note |
| a hard token, a soft token the lane does not own, another lane's word | P-LEAK, naming the node |
| a page wider than its viewport; a mobile run not 390 px wide         | P-FIT |
| text on its background under 4.5:1                                  | P-CONTRAST |
| a view of the lane's set off the tour; another motif                | P-EXPECT |
| a day run left in night                                             | P-THEMED |
| ready after 5 s                                                     | P-PERF |
| one locked token changed on the page                                | P-LOCK (style_lock.compare) |
| a pseudo-locale that did not apply; a missing strings key           | P-QPS |
| another thread closing the fds 3 and 4 while a browser is spawned (page_smoke's runs at once: L08's EBADF) | cdp.Browser spawns all the same |
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SHOW = HERE.parents[2] / "vega-show" / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SHOW))
sys.path.insert(0, str(SHOW / "tests"))

import cdp  # noqa: E402
import page_smoke as ps  # noqa: E402
import page_smoke_checks as pc  # noqa: E402
import style_lock  # noqa: E402
from showcase_manifest import wrap_prose  # noqa: E402

CASE = "02-mlp-plain-binary"
TEMPLATE = SHOW.parent / "assets" / "page"
R_A, R_B, R_Q = (next(r for r in ps.RUNS if r.id == i) for i in ("R-a", "R-b", "R-q"))
L02 = pc.LaneFacts("L02", "points", ("start", "context", "network", "tried", "weights", "try", "many", "results", "code"),
                   frozenset({"cropland", "parcel", "water", "flow"}), frozenset({"bikeshare", "cropland", "harbour"}))


# ---- P-LEAK -------------------------------------------------------------------------------------
def test_a_clean_page_leaks_nothing():
    assert pc.leak_problems([["title", "Land cover classifier"], ["div#lead", "cropland parcels near water"]], L02) == []


def test_a_hard_token_is_named_with_its_node():
    got = pc.leak_problems([["title", "Land cover"], ["p#res-ci-note", "the attack classes"]], L02)
    assert got and all("p#res-ci-note" in g for g in got) and any("'attack'" in g for g in got), got


def test_a_soft_token_leaks_unless_the_lane_owns_it():
    assert pc.leak_problems([["span.x", "one flow per row"]], L02) == []
    lane = pc.LaneFacts(**{**L02.__dict__, "own": frozenset({"cropland"})})
    assert any("'flow'" in g and "span.x" in g for g in pc.leak_problems([["span.x", "one flow per row"]], lane))


def test_another_lanes_word_leaks_but_not_its_own_or_a_generic_word():
    got = pc.leak_problems([["li.a", "rentals at the harbour"], ["li.b", "the cropland class"]], L02)
    assert [g for g in got if "li.a" in g] and not [g for g in got if "li.b" in g], got


def test_the_page_text_is_the_gate_and_the_nodes_only_name_what_it_found():
    # the code view highlights Python's zip( in two spans: on the page it is the call, not the
    # orders fixture's column, so only the real leak is reported
    got = pc.leak_problems([["div#src > span", "zip"], ["div#src > span", "(rows)"], ["p.x", "an attack"]], L02)
    assert got == ["'attack' in p.x: 'an attack'"], got


def test_a_column_of_the_lane_is_its_own_in_the_words_the_page_draws_it_with():
    roads = pc.LaneFacts(**{**L02.__dict__, "own": frozenset(pc.own_parts({"flow_vph", "speed_limit"}) | {"flow_vph"})})
    assert pc.leak_problems([["button.rh", "flow vph"]], roads) == []
    assert pc.leak_problems([["button.rh", "flow vph"]], pc.LaneFacts(**{**L02.__dict__, "own": frozenset()}))


def test_the_golden_lane_may_show_its_hard_tokens():
    g00 = pc.LaneFacts(**{**L02.__dict__, "id": "G00"})
    assert pc.leak_problems([["td", "BENIGN"]], g00) == []
    assert pc.leak_problems([["td", "BENIGN"]], L02)


def test_the_golden_lane_owns_the_soft_tokens_of_its_own_dataset(tmp_path):
    """G00's words are golden/cic.py's VOCAB_WITHHELD (the lane's generator is external): its
    page may say 'traffic' and 'capture day', another lane's may not."""
    lanes = HERE / "lanes.json"
    g00 = pc.LaneFacts.read(lanes, "G00", tmp_path)
    assert {"traffic", "capture", "port", "extractor"} <= g00.own
    page = [["p", "the same capture days and the network traffic"]]
    assert pc.leak_problems(page, g00) == []
    assert pc.leak_problems(page, pc.LaneFacts.read(lanes, "L02", tmp_path))


def test_a_page_with_no_text_fails():
    assert pc.leak_problems([["title", " "]], L02) == ["no text was read off the page"]


# ---- P-FIT, P-CONTRAST, P-THEMED, P-PERF, P-EXPECT ----------------------------------------------
def test_fit():
    assert pc.fit_problems([{"id": "start", "view": "start", "sw": 1440, "iw": 1440}], R_A) == []
    assert pc.fit_problems([{"id": "try", "view": "try", "sw": 1500, "iw": 1440}], R_A)
    assert pc.fit_problems([{"id": "try", "view": "try", "sw": 400, "iw": 400}], R_B)
    assert pc.fit_problems([{"id": "try", "view": "start", "sw": 390, "iw": 390}], R_B)
    assert pc.fit_problems([], R_A)


def test_contrast():
    assert pc.contrast_problems({"fg": "rgb(230, 230, 230)", "bg": "rgb(10, 12, 16)"}, "night") == []
    assert pc.contrast_problems({"fg": "rgb(120, 120, 120)", "bg": "rgb(100, 100, 100)"}, "day")
    assert pc.contrast_problems({"fg": "", "bg": "rgb(0, 0, 0)"}, "day")
    assert abs(pc.contrast((0, 0, 0), (255, 255, 255)) - 21) < 1e-9


def test_themed():
    assert pc.themed_problems({"theme": "day", "clicked": "day"}, R_B) == []
    assert pc.themed_problems({"theme": "night", "clicked": "day"}, R_B)
    assert pc.themed_problems({"theme": "day", "clicked": "night"}, R_B)
    assert pc.themed_problems({"theme": "night", "clicked": None}, R_A) == []


def test_perf():
    assert pc.perf_problems({"ready_ms": 900}) == []
    assert pc.perf_problems({"ready_ms": 5200}) and pc.perf_problems({})


def test_expect():
    views = {v: {"rendered": True} for v in ("start", "context", "network", "try", "many", "results", "code")}
    probe = {"views": views, "sub_views": {"tried": {"rendered": True}, "weights": {"rendered": True}}}
    page = {"motif": {"id": "points"}}
    assert pc.expect_problems(probe, L02, page) == []
    assert pc.expect_problems(probe, L02, {"motif": {"id": "traces"}})
    off = {**probe, "views": {**views, "many": {"off": "no many", "rendered": False}}}
    assert any("many" in p for p in pc.expect_problems(off, L02, page))
    no_tried = {**probe, "sub_views": {"weights": {"rendered": True}}}
    assert any("tried" in p for p in pc.expect_problems(no_tried, L02, page))


# ---- P-LOCK -------------------------------------------------------------------------------------
def test_a_changed_locked_token_fails_the_lock():
    lock = style_lock.load_lock()
    theme = next(iter(lock["tokens"]["night"]))
    reading = {"viewport": lock["computed"]["captured"]["viewport"], "reduced_motion": False,
               "themes": {t: {"styles": lock["computed"]["themes"][t],
                              "tokens": {**lock["tokens"]["root"], **lock["tokens"][t]}} for t in ("night", "day")}}
    assert style_lock.compare(lock, reading) == []
    reading["themes"]["night"]["tokens"][theme] = "#123456"
    assert any(theme in p for p in style_lock.compare(lock, reading))


# ---- P-QPS on a DOM -----------------------------------------------------------------------------
QPS_PROBE = {"locale": "qps-ploc", "extensions": {"missingKeys": [], "unresolvedPlaceholders": []}}
QPS_DOM = ("<html><body><h1 id='h'>⟦Results⟧ ⟪the network ‹water› wins⟫</h1><p class='soft'>5.33e+4</p>"
           "<span class='k'>c</span><pre id='src'>import torch</pre>{}</body></html>")


def test_a_wrapped_dom_passes_the_pseudo_locale():
    assert pc.qps_problems(QPS_PROBE, QPS_DOM.format("")) == []


def test_an_unwrapped_word_names_its_node():
    got = pc.qps_problems(QPS_PROBE, QPS_DOM.format("<div id='res'><p class='hint'>seeded words</p></div>"))
    assert got == ["unwrapped text in div#res > p.hint: ['seeded', 'words'] (in 'seeded words')"], got


def test_a_pseudo_locale_that_did_not_apply_fails():
    got = pc.qps_problems({**QPS_PROBE, "locale": "en"}, "<html><body><p>⟦a⟧</p></body></html>")
    assert any("not qps-ploc" in g for g in got) and any("‹" in g for g in got) and any("⟪" in g for g in got)
    missing = {**QPS_PROBE, "extensions": {"missingKeys": ["try.run"], "unresolvedPlaceholders": []}}
    assert any("try.run" in g for g in pc.qps_problems(missing, QPS_DOM.format("")))


# ---- VS26 in a browser: a template string outside t() in a qps-ploc page ----------------------------
needs_chrome = pytest.mark.skipif(cdp.find_chrome() is None, reason="no Chrome ($CHROME)")


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    path, before = list(sys.path), set(sys.modules)
    try:
        import make_page_fixtures                 # puts the forge's tests (its own run_matrix) on the path
        out = tmp_path_factory.mktemp("fixtures")
        assert make_page_fixtures.main(["--cases", "2", "--out", str(out)]) == 0
    finally:                                      # this directory's modules stay this directory's
        sys.path[:] = path
        for name in set(sys.modules) - before:
            if (HERE / f"{name}.py").is_file() and Path(getattr(sys.modules[name], "__file__", "") or "").parent != HERE:
                del sys.modules[name]
    story = out / CASE / "story.json"            # wrapped as build_showcase wraps a qps-ploc build
    story.write_text(json.dumps(wrap_prose(json.loads(story.read_text(encoding="utf-8"))),
                                ensure_ascii=False), encoding="utf-8")
    return out / CASE


def qps_run(case_dir: Path, template: Path) -> list[str]:
    import serve_page_fixture
    srv = serve_page_fixture.start(case_dir, template)
    try:
        got = ps.drive(cdp.find_chrome(), srv.url, R_Q, None, None)
    finally:
        srv.stop()
    assert got["probe"].get("ready"), got["probe"]
    return pc.qps_problems(got["probe"], got["dom"])


@needs_chrome
def test_vs26_the_case_as_built_passes_the_pseudo_locale(case):
    assert qps_run(case, TEMPLATE) == []


@needs_chrome
def test_vs26_a_template_string_outside_t_fails_naming_its_node(case, tmp_path):
    template = tmp_path / "template"
    shutil.copytree(TEMPLATE, template, ignore=shutil.ignore_patterns("tests", "node_modules"))
    view = template / "js" / "views" / "results.js"
    anchor = '$("res-ci-note").replaceChildren('
    text = view.read_text(encoding="utf-8")
    assert anchor in text, "the seed's anchor is gone from results.js"
    view.write_text(text.replace(anchor, anchor + '"Seeded words outside the table ", ', 1), encoding="utf-8")
    got = qps_run(case, template)
    assert got and all("p#res-ci-note" in g for g in got), got
    assert any("'Seeded'" in g for g in got), got


# ---- the browser spawn under concurrent runs ------------------------------------------------------
SPAWN_RACE = """
import os, subprocess, sys
sys.path.insert(0, sys.argv[1])
import cdp
held = [os.open(os.devnull, os.O_RDONLY) for _ in range(2)]
assert held == [3, 4], held
real = subprocess._fork_exec
def fork_exec(*a, **k):                       # another run's browser closes its fds 3 and 4 now
    for f in held:
        os.close(f)
    held.clear()
    return real(*a, **k)
subprocess._fork_exec = fork_exec
b = cdp.Browser(cdp.find_chrome())
print("spawned", b.version)
b.close()
"""


@needs_chrome
def test_a_browser_spawns_while_another_run_closes_fds_3_and_4():
    got = subprocess.run([sys.executable, "-c", SPAWN_RACE, str(SHOW)], capture_output=True, text=True, timeout=120)
    assert got.returncode == 0 and "spawned" in got.stdout, got.stderr[-800:]

