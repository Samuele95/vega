#!/usr/bin/env python3
"""
test_smoke_training_selector.py — smoke_training's run-selector checks take one of two branches,
and neither is skipped or loosened (A33).

    python3 -m pytest test_smoke_training_selector.py [-q]

A campaign of two or more runs keeps every switch check: the probe presses another run's
button, and the panel must follow it (selector-switches-run, selector-button-aria-pressed),
redraw different curves (curves-redraw-on-switch) and replace its charts
(redraw-replaces-not-appends). A campaign of one run — a lane whose campaign is R0 only — has no
second run to switch to, and the smoke asserts that rendering instead: one button and no switch
target, that button pressed, the one run shown as the selected run with the rule that picked it,
and, pressed again, a panel that keeps the run and redraws the same record in place.

1. **The branches on probe records.** A correct one-run record passes the one-run branch and
   every one of its checks runs; each seeded one-run defect (a second button, the button not
   pressed, a selection that names another run, no rule, a re-press that moves the panel,
   appended charts, different polylines) fails the check that names it. A correct two-run
   record passes the switch branch; a two-run record that did not switch fails it, and the
   branch is chosen by the ledger's run count, never by what the probe happened to find.
2. **The real render, in a headless browser** (skipped only when no browser is installed):
   the probe of smoke_training.py driven over training-render.js, on a one-run and a two-run
   layer, with and without histories, passes its branch with no failed check.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import smoke_training  # noqa: E402

RULE = ("best valid primary in the direction of metrics.selection.mode; ties -> fewer params; "
        "ineligible: a run whose diagnostics include D1 (divergence) or D2 (not learning)")
ONE_RUN_CHECKS = {"one-selector-button-per-run", "curves-open-on-selected-run",
                  "five-curve-panels", "curves-drawn", "single-run-no-selector-switch",
                  "single-run-button-pressed", "single-run-shown-as-selected",
                  "single-run-selected-with-its-rule", "single-run-reselect-keeps-the-run",
                  "single-run-redraw-draws-the-same-record", "redraw-replaces-not-appends"}
SWITCH_CHECKS = {"one-selector-button-per-run", "curves-open-on-selected-run",
                 "five-curve-panels", "curves-drawn", "selector-switches-run",
                 "selector-button-aria-pressed", "curves-redraw-on-switch",
                 "redraw-replaces-not-appends"}
CHARTS = ["loss", "primary", "lr", "grad-norm", "update-ratio"]


def outcomes(record, rows, selection, degraded=False):
    """{check name: passed} for smoke_training.selector_checks on one probe record."""
    got = {}

    def check(cond, name, detail=""):
        key = name.split(":", 1)[1]
        assert key not in got, f"check {key} ran twice"
        got[key] = bool(cond)

    label = "no-histories" if degraded else "full"
    smoke_training.selector_checks(check, record, rows, selection, label, degraded)
    return got


def one_run_record(**change):
    rec = {"run_buttons": 1, "curves_run": "R0", "selected_run": "R0", "chart_ids": CHARTS,
           "curve_lines": 10, "charts": 5, "first_path": "M0,0L1,1", "switch_to": None,
           "pressed_open": "true", "pressed_count_open": 1, "reselect": "R0",
           "curves_run_after": "R0", "pressed_after": "true", "pressed_count_after": 1,
           "first_path_after": "M0,0L1,1", "charts_after": 5,
           "selection_text": f"R0 the rule {RULE}"}
    rec.update(change)
    return rec


def two_run_record(**change):
    rec = {"run_buttons": 2, "curves_run": "R0", "selected_run": "R0", "chart_ids": CHARTS,
           "curve_lines": 10, "charts": 5, "first_path": "M0,0L1,1", "switch_to": "R1",
           "pressed_open": "true", "pressed_count_open": 1, "reselect": None,
           "curves_run_after": "R1", "pressed_after": "true", "pressed_count_after": 1,
           "first_path_after": "M0,0L1,2", "charts_after": 5,
           "selection_text": f"R0 the rule {RULE}"}
    rec.update(change)
    return rec


ONE = [{"id": "R0"}]
TWO = [{"id": "R0"}, {"id": "R1"}]
SELECTION = {"run": "R0", "rule": RULE}


# ------------------------------------------------------------------ 1. the branches on records
def test_one_run_asserts_its_rendering():
    got = outcomes(one_run_record(), ONE, SELECTION)
    assert set(got) == ONE_RUN_CHECKS
    assert all(got.values()), {k: v for k, v in got.items() if not v}


def test_one_run_degraded_asserts_its_rendering():
    rec = one_run_record(first_path=None, first_path_after=None, curve_lines=0, chart_ids=[])
    got = outcomes(rec, ONE, SELECTION, degraded=True)
    assert set(got) == ONE_RUN_CHECKS - {"five-curve-panels", "curves-drawn",
                                         "single-run-redraw-draws-the-same-record",
                                         "redraw-replaces-not-appends"}
    assert all(got.values()), {k: v for k, v in got.items() if not v}


@pytest.mark.parametrize("change, selection, failing", [
    ({"run_buttons": 2, "switch_to": "R1"}, SELECTION, "single-run-no-selector-switch"),
    ({"pressed_open": "false"}, SELECTION, "single-run-button-pressed"),
    ({"pressed_count_open": 0}, SELECTION, "single-run-button-pressed"),
    ({}, {"run": None, "rule": RULE}, "single-run-shown-as-selected"),
    ({"selected_run": None}, SELECTION, "single-run-shown-as-selected"),
    ({}, {"run": "R0", "rule": ""}, "single-run-selected-with-its-rule"),
    ({"selection_text": "R0"}, SELECTION, "single-run-selected-with-its-rule"),
    ({"reselect": None, "curves_run_after": None, "pressed_after": None},
     SELECTION, "single-run-reselect-keeps-the-run"),
    ({"curves_run_after": "R9"}, SELECTION, "single-run-reselect-keeps-the-run"),
    ({"pressed_after": "false"}, SELECTION, "single-run-reselect-keeps-the-run"),
    ({"first_path_after": "M9,9"}, SELECTION, "single-run-redraw-draws-the-same-record"),
    ({"charts_after": 10}, SELECTION, "redraw-replaces-not-appends"),
])
def test_one_run_defects_fail_their_check(change, selection, failing):
    got = outcomes(one_run_record(**change), ONE, selection)
    assert got[failing] is False, f"{failing} passed on {change} / {selection}"


def test_two_runs_keep_every_switch_check():
    got = outcomes(two_run_record(), TWO, SELECTION)
    assert set(got) == SWITCH_CHECKS
    assert all(got.values()), {k: v for k, v in got.items() if not v}
    degraded = outcomes(two_run_record(curve_lines=0, chart_ids=[]), TWO, SELECTION, True)
    assert set(degraded) == {"one-selector-button-per-run", "curves-open-on-selected-run",
                             "selector-switches-run", "selector-button-aria-pressed"}
    assert all(degraded.values())


@pytest.mark.parametrize("change, failing", [
    ({"curves_run_after": "R0"}, "selector-switches-run"),
    ({"pressed_after": "false"}, "selector-button-aria-pressed"),
    ({"first_path_after": "M0,0L1,1"}, "curves-redraw-on-switch"),
    ({"charts_after": 10}, "redraw-replaces-not-appends"),
])
def test_two_run_defects_fail_their_check(change, failing):
    assert outcomes(two_run_record(**change), TWO, SELECTION)[failing] is False


def test_the_branch_is_the_ledgers_run_count_not_the_probes():
    # a two-run ledger whose probe found no second button (a selector that lost a run) is not
    # read as a one-run campaign: it fails the switch checks
    got = outcomes(one_run_record(), TWO, SELECTION)
    assert set(got) == SWITCH_CHECKS
    assert not got["one-selector-button-per-run"]
    assert not got["selector-switches-run"]
    assert not got["curves-redraw-on-switch"]


# ------------------------------------------------------------------ 2. the real render
def _history(k):
    """Run k's record: its curves bend differently from another run's (a chart scales to its
    own range, so a constant multiple would draw the same polyline)."""
    bend = (1.0, 0.7 - 0.2 * k, 0.5, 0.4 + 0.05 * k)
    return {"epochs": [1, 2, 3, 4],
            "train.loss": list(bend), "valid.loss": [v + 0.1 for v in bend],
            "train.mse": list(bend), "valid.mse": [v + 0.1 for v in bend],
            "lr": [1e-3, 1e-3, 5e-4, 2.5e-4],
            "grad_norm.global": [2.0, 1.5 - 0.3 * k, 1.2, 1.0],
            "grad_norm.B1": [1.0, 0.8, 0.7 - 0.2 * k, 0.6],
            "update_ratio.B1": [1e-2, 8e-3, 6e-3 + 1e-3 * k, 5e-3]}


def _layer(run_ids):
    rows = [{"id": rid, "kind": "baseline" if i == 0 else "ablation", "status": "done",
             "primary": 0.5 + 0.1 * i, "primary_name": "mse", "best_epoch": 4, "epochs": 4,
             "diagnostics": []} for i, rid in enumerate(run_ids)]
    return {"schema": "vega/training@1",
            "experiments": {"runs": rows, "hpo": [],
                            "selection": {"run": run_ids[0], "rule": RULE, "override": None,
                                          "alternatives": []}},
            "histories": {rid: _history(i) for i, rid in enumerate(run_ids)},
            "diagnostics": [], "readings": {"abstract": "a campaign", "per_run": {}}}


def _page(doc):
    render = (SCRIPTS.parent / "assets" / "training-render.js").read_text(encoding="utf-8")
    data = json.dumps(doc).replace("</", "<\\/")
    return ("<!doctype html><html><head><meta charset='utf-8'></head><body>"
            "<nav class='tabs'><button data-tab='model'>Model</button></nav>"
            "<section class='tab-pane active' data-tab='model'></section>"
            f"<script id='layer-training-data' type='application/json'>{data}</script>"
            f"<script>{render}</script></body></html>")


@pytest.mark.parametrize("run_ids", [["R0"], ["R0", "R1"]], ids=["one-run", "two-runs"])
@pytest.mark.parametrize("degraded", [False, True], ids=["full", "no-histories"])
def test_the_real_render_passes_its_branch(run_ids, degraded):
    chrome = smoke_training.find_chrome()
    if not chrome:
        pytest.skip("no headless browser installed (smoke_training.CHROME_CANDIDATES)")
    doc = _layer(run_ids)
    with tempfile.TemporaryDirectory() as td:
        page = smoke_training.probe_copy(_page(doc), td, "layer.html", no_histories=degraded)
        dom, _ = smoke_training.dump_dom(chrome, page)
    probe = smoke_training.parse_results(dom)
    assert probe is not None, "the probe wrote no record"
    assert not probe["errors"], probe["errors"]
    got = outcomes(probe["checks"], doc["experiments"]["runs"], doc["experiments"]["selection"],
                   degraded)
    branch = "single-run-no-selector-switch" if len(run_ids) == 1 else "selector-switches-run"
    assert branch in got
    assert all(got.values()), {k: v for k, v in got.items() if not v}
