#!/usr/bin/env python3
"""
smoke_training.py — headless-Chromium check that the /vega-train `training` layer renders.

    python3 smoke_training.py OUT.html [--strict]

A copy of OUT.html is opened in headless Chromium with a probe appended that exercises the page —
including a click on the curve panel's run selector — and writes its findings into a
<pre id="training-smoke">; the DOM dump is then parsed.

  1. the tab button (data-tab="training") sits in the host's nav.tabs / nav[role=tablist] and is
     labelled Training;
  2. the pane section.tab-pane[data-layer="training"] is mounted and inactive by default;
  3. the seven surfaces exist (data-view = ledger | curves | diagnostics | hpo | mechanics |
     selection | forks);
  4. the ledger holds one row per run in experiments.runs, the selected run is marked, and every
     run the selection rule may not pick is marked ineligible — a cut or pruned run read as a
     result is the one mistake this layer exists to prevent;
  5. the curves render for the selected run — loss, primary metric, learning rate, gradient norms,
     update ratios — and redraw when the run selector changes, driven here from the probe; a
     campaign of one run asserts its own rendering instead (one pressed button, no switch, the
     run shown as selected with its rule, a re-press that redraws the same run in place);
  6. the diagnostics panel shows one entry per fired diagnostic, with its evidence;
  7. the HPO panel plots every trial of each study, marks the pruned ones apart, and shows the
     importance bars together with the string that says how the importance was computed;
  8. the mechanics panel carries the loss, the update rule, the schedule curve and the first step;
  9. the selection panel states the rule and distinguishes an override from the rule's own pick;
 10. window.__training exposes runs(), history(id), selection() and diagnostics(), and runs() and
     history(id) return the ledger and the series the page was given, not rebuilt ones;
 11. no uncaught JS error;
 12. on a copy whose `histories` is emptied, the pane still mounts every surface, the ledger still
     holds its rows, and the curve panel says the history is missing rather than drawing nothing.

Exit 0 on success, 1 on a failed check (or, with --strict, when no browser is available).
Without --strict and without a browser the run prints a WARN: and exits 0, unverified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_CANDIDATES = ("chromium", "google-chrome", "chromium-browser", "chrome")
DATA_RE = re.compile(r'(<script\s+id="layer-training-data"[^>]*>)([\s\S]*?)(</script>)', re.I)
VIEWS = ("ledger", "curves", "diagnostics", "hpo", "mechanics", "selection", "forks")

PROBE = r"""
<script id="training-smoke-probe">
(function(){
  var R = { errors: [], checks: {} };
  window.addEventListener('error', function(e){ R.errors.push('onerror: ' + (e.message || e)); });
  var origErr = console.error;
  console.error = function(){ R.errors.push('console.error: ' + Array.prototype.slice.call(arguments).map(String).join(' ')); if (origErr) origErr.apply(console, arguments); };
  function q(s){ return document.querySelector(s); }
  function text(pane, view){
    var v = pane ? pane.querySelector('[data-view="' + view + '"]') : null;
    return v ? (v.textContent || '') : '';
  }
  function count(pane, sel){ return pane ? pane.querySelectorAll(sel).length : 0; }
  function firstPath(pane){
    var p = pane ? pane.querySelector('[data-view="curves"] path.t-line') : null;
    return p ? p.getAttribute('d') : null;
  }
  function run(){
    try {
      var nav = q('nav.tabs') || q('nav[role="tablist"]');
      var btn = nav ? nav.querySelector('button[data-tab="training"]') : null;
      R.checks.tab_button_in_nav = !!btn;
      R.checks.tab_label = btn ? (btn.textContent || '').trim() : '';
      var pane = q('section.tab-pane[data-layer="training"]');
      R.checks.pane_mounted = !!pane;
      R.checks.pane_inactive_by_default = !!pane && !pane.classList.contains('active');
      // idempotence: exactly one pane, whatever the guard saw (the tab button carries the same
      // data-layer attribute, so a bare [data-layer] count would always read two)
      R.checks.panes_with_layer = document.querySelectorAll('section.tab-pane[data-layer="training"]').length;
      R.checks.views = {};
      ['ledger','curves','diagnostics','hpo','mechanics','selection','forks'].forEach(function(v){
        R.checks.views[v] = !!(pane && pane.querySelector('[data-view="' + v + '"]'));
      });

      R.checks.ledger_rows = count(pane, '[data-view="ledger"] tbody tr.t-row');
      R.checks.ledger_ids = [];
      if (pane) Array.prototype.forEach.call(
        pane.querySelectorAll('[data-view="ledger"] tbody tr.t-row'), function(r){
          R.checks.ledger_ids.push(r.getAttribute('data-run'));
        });
      var sel = pane ? pane.querySelectorAll('[data-view="ledger"] tr.t-selected') : [];
      R.checks.selected_marked = sel.length;
      R.checks.selected_run = sel.length ? sel[0].getAttribute('data-run') : null;
      R.checks.selected_says_so = sel.length ? (sel[0].textContent || '').indexOf('selected') >= 0 : false;
      R.checks.ineligible_rows = [];
      if (pane) Array.prototype.forEach.call(
        pane.querySelectorAll('[data-view="ledger"] tr.t-ineligible'), function(r){
          R.checks.ineligible_rows.push(r.getAttribute('data-run'));
        });
      R.checks.abstract_len = pane ? (pane.querySelector('.t-abstract') || {textContent:''}).textContent.length : 0;

      var curves = pane ? pane.querySelector('[data-view="curves"] .t-curves') : null;
      R.checks.run_buttons = count(pane, '[data-view="curves"] button.t-runbtn');
      R.checks.curves_run = curves ? curves.getAttribute('data-run') : null;
      R.checks.charts = count(pane, '[data-view="curves"] figure.t-fig');
      R.checks.chart_ids = [];
      if (pane) Array.prototype.forEach.call(
        pane.querySelectorAll('[data-view="curves"] figure.t-fig'), function(f){
          R.checks.chart_ids.push(f.getAttribute('data-chart'));
        });
      R.checks.curve_lines = count(pane, '[data-view="curves"] path.t-line');
      R.checks.curves_empty = count(pane, '[data-view="curves"] .t-curves .t-empty');
      R.checks.curves_text = text(pane, 'curves').slice(0, 4000);
      R.checks.first_path = firstPath(pane);
      // 5: drive the selector — a second run must redraw the panel in place. A campaign of one
      // run has no second run to switch to: its one button is pressed again instead, and the
      // redraw must keep that run and replace the panel in place (A33)
      var btns = pane ? pane.querySelectorAll('[data-view="curves"] button.t-runbtn') : [];
      var pressed = '[data-view="curves"] button.t-runbtn[aria-pressed="true"]';
      var other = null, own = null, i;
      for (i = 0; i < btns.length; i++) {
        if (btns[i].getAttribute('data-run') === R.checks.curves_run) { if (!own) own = btns[i]; }
        else if (!other) other = btns[i];
      }
      R.checks.pressed_open = own ? own.getAttribute('aria-pressed') : null;
      R.checks.pressed_count_open = count(pane, pressed);
      var target = other || own;
      if (target) {
        R.checks.switch_to = other ? other.getAttribute('data-run') : null;
        R.checks.reselect = other ? null : own.getAttribute('data-run');
        target.click();
        R.checks.curves_run_after = curves ? curves.getAttribute('data-run') : null;
        R.checks.first_path_after = firstPath(pane);
        R.checks.charts_after = count(pane, '[data-view="curves"] figure.t-fig');
        R.checks.pressed_after = target.getAttribute('aria-pressed');
        R.checks.pressed_count_after = count(pane, pressed);
        R.checks.curves_text_after = text(pane, 'curves').slice(0, 4000);
      }

      R.checks.diag_rows = count(pane, '[data-view="diagnostics"] tr.t-diag');
      R.checks.diag_codes = [];
      if (pane) Array.prototype.forEach.call(
        pane.querySelectorAll('[data-view="diagnostics"] tr.t-diag'), function(r){
          R.checks.diag_codes.push(r.getAttribute('data-code') + '@' + r.getAttribute('data-run'));
        });
      R.checks.diag_text = text(pane, 'diagnostics').slice(0, 6000);
      R.checks.diag_links = count(pane, '[data-view="diagnostics"] a.t-runlink');

      R.checks.studies = count(pane, '[data-view="hpo"] .t-study');
      R.checks.trials = count(pane, '[data-view="hpo"] .t-trial');
      R.checks.trials_pruned = count(pane, '[data-view="hpo"] .t-trial-pruned');
      R.checks.bars = count(pane, '[data-view="hpo"] .t-barrow');
      R.checks.hpo_text = text(pane, 'hpo').slice(0, 6000);

      R.checks.formulas = count(pane, '[data-view="mechanics"] .t-formula');
      R.checks.schedule_lines = count(pane, '[data-view="mechanics"] [data-chart="schedule"] path.t-line');
      R.checks.mech_stats = count(pane, '[data-view="mechanics"] .t-stat');
      R.checks.mech_text = text(pane, 'mechanics').slice(0, 4000);

      R.checks.selection_text = text(pane, 'selection').slice(0, 6000);
      var ov = pane ? pane.querySelector('[data-view="selection"] .t-override') : null;
      R.checks.override_state = ov ? ov.getAttribute('data-override') : null;
      R.checks.override_marked = count(pane, '[data-view="selection"] .t-override-on');
      R.checks.alt_rows = count(pane, '[data-view="selection"] tr.t-alt');

      R.checks.fork_rows = count(pane, '[data-view="forks"] tr.t-fork');
      R.checks.fork_unattended = count(pane, '[data-view="forks"] tr.t-unattended');
      R.checks.forks_text = text(pane, 'forks').slice(0, 3000);

      R.checks.hook = !!(window.__training
                         && typeof window.__training.runs === 'function'
                         && typeof window.__training.history === 'function'
                         && typeof window.__training.selection === 'function'
                         && typeof window.__training.diagnostics === 'function');
      if (R.checks.hook) {
        R.checks.hook_version = window.__training.version;
        var rs = window.__training.runs() || [];
        R.checks.hook_runs = rs.length;
        R.checks.hook_run_ids = rs.map(function(r){ return r.id; });
        R.checks.hook_selection = (window.__training.selection() || {}).run;
        R.checks.hook_diagnostics = (window.__training.diagnostics() || []).length;
        var h = rs.length ? window.__training.history(rs[0].id) : null;
        R.checks.hook_history_keys = h ? Object.keys(h).length : 0;
        R.checks.hook_history_epochs = (h && h.epochs) ? h.epochs.length : 0;
        R.checks.hook_history_missing = window.__training.history('no-such-run') === null;
      }
      if (btn) { btn.click(); R.checks.tab_activates = !!(pane && pane.classList.contains('active')); }
    } catch (e) { R.errors.push('probe: ' + (e && e.stack || e)); }
    var pre = document.createElement('pre'); pre.id = 'training-smoke';
    pre.textContent = JSON.stringify(R); document.body.appendChild(pre);
  }
  function go(){ setTimeout(run, 300); }
  if (document.readyState === 'complete') go(); else window.addEventListener('load', go);
})();
</script>
"""


def find_chrome():
    env = os.environ.get("CHROME")
    if env and (Path(env).is_file() or shutil.which(env)):
        return env if Path(env).is_file() else shutil.which(env)
    for c in CHROME_CANDIDATES:
        if shutil.which(c):
            return c
    return None


def dump_dom(chrome, path, budget_ms=8000):
    proc = subprocess.run(
        [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
         f"--virtual-time-budget={budget_ms}", "--enable-logging=stderr", "--v=0",
         "--dump-dom", f"file://{path.resolve()}"], capture_output=True, text=True, timeout=120)
    return proc.stdout, proc.stderr


def layer_data(html):
    m = DATA_RE.search(html)
    if not m:
        return None, None
    try:
        return json.loads(m.group(2)), m
    except json.JSONDecodeError:
        return None, m


def ineligible(row):
    """The selection rule's mechanical exclusions (run-contract.md §4), counted the way the
    render marks them: a pruned trial, a run that diverged or never learned, a run the budget
    cut before it improved once."""
    codes = row.get("diagnostics") or []
    if row.get("status") in ("pruned", "failed", "error"):
        return True
    if "D1" in codes or "D2" in codes:
        return True
    return "D9" in codes and not row.get("best_epoch")


def probe_copy(html, tmpdir, name, no_histories=False):
    if no_histories:
        doc, m = layer_data(html)
        if doc is not None:
            doc["histories"] = {}
            html = (html[:m.start(2)] + "\n" +
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n" +
                    html[m.end(2):])
    idx = html.rfind("</body>")
    html = html[:idx] + PROBE + html[idx:] if idx >= 0 else html + PROBE
    p = Path(tmpdir) / name
    p.write_text(html, encoding="utf-8")
    return p


def parse_results(dom):
    m = re.search(r'<pre id="training-smoke">([\s\S]*?)</pre>', dom)
    if not m:
        return None
    txt = (m.group(1).replace("&quot;", '"').replace("&lt;", "<")
           .replace("&gt;", ">").replace("&amp;", "&"))
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def selector_checks(check, c, rows, selection, label, degraded):
    """5: the curve panel's run selector, read from the probe's record `c`.

    A campaign of two or more runs must switch: the probe presses another run's button, and the
    panel must follow it, redraw different curves and replace its charts rather than append.
    A campaign of one run (A33) has nothing to switch to, and the smoke asserts that rendering
    instead of skipping it: one button and no second run to switch to, that button pressed, the
    one run shown as the selected run with the rule that picked it, and — pressed again — the
    panel keeps the run and redraws the same record in place."""
    ids = [r.get("id") for r in rows]
    check(c.get("run_buttons") == len(rows), f"{label}:one-selector-button-per-run",
          f"{c.get('run_buttons')} != {len(rows)}")
    check(c.get("curves_run") == (selection.get("run") or (ids[0] if ids else None)),
          f"{label}:curves-open-on-selected-run", str(c.get("curves_run")))
    if not degraded:
        want = {"loss", "primary", "lr", "grad-norm", "update-ratio"}
        check(want.issubset(set(c.get("chart_ids") or [])), f"{label}:five-curve-panels",
              str(c.get("chart_ids")))
        check((c.get("curve_lines") or 0) >= 8, f"{label}:curves-drawn",
              str(c.get("curve_lines")))
    if len(rows) == 1:
        only = ids[0]
        check(c.get("run_buttons") == 1 and c.get("switch_to") is None,
              f"{label}:single-run-no-selector-switch",
              f"{c.get('run_buttons')} button(s), a switch to {c.get('switch_to')!r}")
        check(c.get("pressed_open") == "true" and c.get("pressed_count_open") == 1,
              f"{label}:single-run-button-pressed",
              f"aria-pressed={c.get('pressed_open')!r}, {c.get('pressed_count_open')} pressed")
        check(selection.get("run") == only and c.get("curves_run") == only
              and c.get("selected_run") == only,
              f"{label}:single-run-shown-as-selected",
              f"selection {selection.get('run')!r}, curves {c.get('curves_run')!r}, "
              f"ledger {c.get('selected_run')!r} — the one run is {only!r}")
        rule = selection.get("rule") or ""
        check(bool(rule) and rule[:40] in (c.get("selection_text") or ""),
              f"{label}:single-run-selected-with-its-rule",
              "no rule recorded" if not rule else "the rule is not in the selection panel")
        check(c.get("reselect") == only and c.get("curves_run_after") == only
              and c.get("pressed_after") == "true" and c.get("pressed_count_after") == 1,
              f"{label}:single-run-reselect-keeps-the-run",
              f"pressed {c.get('reselect')!r}: the panel shows {c.get('curves_run_after')!r}, "
              f"aria-pressed={c.get('pressed_after')!r}, {c.get('pressed_count_after')} pressed")
        if not degraded:
            check(c.get("first_path") and c.get("first_path_after") == c.get("first_path"),
                  f"{label}:single-run-redraw-draws-the-same-record",
                  "the polylines changed although the run did not")
            check(c.get("charts_after") == c.get("charts"),
                  f"{label}:redraw-replaces-not-appends",
                  f"{c.get('charts')} -> {c.get('charts_after')}")
        return
    check(c.get("curves_run_after") == c.get("switch_to"),
          f"{label}:selector-switches-run",
          f"{c.get('curves_run_after')} != {c.get('switch_to')}")
    check(c.get("pressed_after") == "true", f"{label}:selector-button-aria-pressed")
    if not degraded:
        check(c.get("first_path") and c.get("first_path_after")
              and c.get("first_path") != c.get("first_path_after"),
              f"{label}:curves-redraw-on-switch",
              "the polylines did not change when the run changed")
        check(c.get("charts_after") == c.get("charts"),
              f"{label}:redraw-replaces-not-appends",
              f"{c.get('charts')} -> {c.get('charts_after')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("html")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    path = Path(a.html)
    if not path.is_file():
        print(f"ERROR: {path} not found")
        return 1
    html = path.read_text(encoding="utf-8")
    doc, _ = layer_data(html)
    if doc is None:
        print("ERROR: no parseable layer-training-data script in the file")
        return 1
    chrome = find_chrome()
    if not chrome:
        msg = f"no headless browser available (checked {CHROME_CANDIDATES})"
        if a.strict:
            print(f"ERROR: {msg} — failing under --strict")
            return 1
        print(f"WARN: {msg}; the render is unverified")
        return 0

    exp = doc.get("experiments") or {}
    rows = exp.get("runs") or []
    selection = exp.get("selection") or {}
    histories = doc.get("histories") or {}
    diags = doc.get("diagnostics") or []
    studies = exp.get("hpo") or []
    forks = doc.get("forks") or (doc.get("ladder") or {}).get("forks") or []
    ids = [r.get("id") for r in rows]
    n_ineligible = len([r for r in rows if ineligible(r)])
    n_unattended = len([f for f in forks if not isinstance(f, dict) or f.get("answer") is None])
    n_trials = len([r for r in rows if r.get("kind") in ("hpo", "trial")])
    n_pruned = len([r for r in rows if r.get("status") == "pruned"])
    n_bars = sum(len((s.get("importance") or {})) for s in studies)
    hows = [str(s.get("importance_how") or s.get("how") or "") for s in studies]
    failures, checks = [], 0

    def check(cond, name, detail=""):
        nonlocal checks
        checks += 1
        if cond:
            print(f"  OK   {name}")
        else:
            failures.append(name)
            print(f"  FAIL {name}{': ' + detail if detail else ''}")

    with tempfile.TemporaryDirectory() as td:
        for label, degraded in (("full", False), ("no-histories", True)):
            p = probe_copy(html, td, f"smoke-{label}.html", no_histories=degraded)
            try:
                dom, err = dump_dom(chrome, p)
            except subprocess.TimeoutExpired:
                check(False, f"{label}:render", "chromium timed out")
                continue
            R = parse_results(dom)
            if R is None:
                check(False, f"{label}:probe", "no probe output captured")
                continue
            c = R["checks"]

            # 1–3: the mount
            check(c.get("tab_button_in_nav"), f"{label}:tab-button")
            check(c.get("tab_label") == "Training", f"{label}:tab-labelled-Training",
                  str(c.get("tab_label")))
            check(c.get("pane_mounted"), f"{label}:pane")
            check(c.get("pane_inactive_by_default"), f"{label}:pane-inactive-by-default")
            check(c.get("panes_with_layer") == 1, f"{label}:mounted-once",
                  str(c.get("panes_with_layer")))
            for v in VIEWS:
                check((c.get("views") or {}).get(v), f"{label}:view-{v}")

            # 4: the ledger
            check(c.get("ledger_rows") == len(rows), f"{label}:one-row-per-run",
                  f"{c.get('ledger_rows')} != {len(rows)}")
            check(c.get("ledger_ids") == ids, f"{label}:ledger-run-ids",
                  f"{c.get('ledger_ids')} != {ids}")
            if selection.get("run"):
                check(c.get("selected_marked") == 1 and c.get("selected_run") == selection["run"],
                      f"{label}:selected-run-marked",
                      f"{c.get('selected_marked')} row(s), {c.get('selected_run')}")
                check(c.get("selected_says_so"), f"{label}:selected-row-says-so")
            check(len(c.get("ineligible_rows") or []) == n_ineligible,
                  f"{label}:ineligible-runs-marked",
                  f"{c.get('ineligible_rows')} != {n_ineligible} expected")
            check((c.get("abstract_len") or 0) > 200, f"{label}:abstract-rendered",
                  str(c.get("abstract_len")))

            # 5: the curves and their redraw
            selector_checks(check, c, rows, selection, label, degraded)

            # 6: the diagnostics
            check(c.get("diag_rows") == len(diags), f"{label}:one-entry-per-fired-diagnostic",
                  f"{c.get('diag_rows')} != {len(diags)}")
            want_codes = [f"{d.get('code')}@{d.get('run')}" for d in diags]
            check(c.get("diag_codes") == want_codes, f"{label}:diagnostic-code-and-run",
                  f"{c.get('diag_codes')} != {want_codes}")
            dtext = c.get("diag_text") or ""
            check(all((d.get("evidence") or "")[:40] in dtext for d in diags),
                  f"{label}:diagnostic-evidence-shown")
            linked = len([d for d in diags if d.get("led_to") in ids])
            check((c.get("diag_links") or 0) >= linked + len([d for d in diags
                                                              if d.get("run") in ids]),
                  f"{label}:diagnostic-rows-link-to-their-runs", str(c.get("diag_links")))

            # 7: the study
            check(c.get("studies") == len(studies), f"{label}:one-panel-per-study",
                  f"{c.get('studies')} != {len(studies)}")
            check(c.get("trials") == n_trials, f"{label}:every-trial-plotted",
                  f"{c.get('trials')} != {n_trials}")
            check(c.get("trials_pruned") == n_pruned, f"{label}:pruned-trials-marked-apart",
                  f"{c.get('trials_pruned')} != {n_pruned}")
            check(c.get("bars") == n_bars, f"{label}:importance-bars",
                  f"{c.get('bars')} != {n_bars}")
            htext = c.get("hpo_text") or ""
            check(all(h and h in htext for h in hows), f"{label}:importance-how-shown",
                  f"{hows} not in the panel")

            # 8: the mechanics
            check((c.get("formulas") or 0) >= 2, f"{label}:loss-and-update-rule",
                  str(c.get("formulas")))
            check((c.get("schedule_lines") or 0) >= 1, f"{label}:schedule-curve")
            check((c.get("mech_stats") or 0) >= 3, f"{label}:first-step-shown",
                  str(c.get("mech_stats")))
            # the clip is a number, and 1.0 rendered as "1" is the same number — compare
            # numerically against what the panel printed rather than by string identity
            clip = (doc.get("mechanics") or {}).get("clip")
            text = (c.get("mech_text") or "").replace("·", ".")
            shown = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]
            check(clip is None
                  or any(abs(v - float(clip)) < 1e-9 for v in shown)
                  or "grad clip off" in text,
                  f"{label}:clip-shown", str(clip))

            # 9: the selection
            stext = c.get("selection_text") or ""
            check(selection.get("rule", "")[:40] in stext, f"{label}:selection-rule-shown")
            over = selection.get("override")
            check(c.get("override_state") == ("yes" if over else "no"),
                  f"{label}:override-state", str(c.get("override_state")))
            check(c.get("override_marked") == (1 if over else 0),
                  f"{label}:override-kept-apart-from-the-rule")
            check(c.get("alt_rows") == len(selection.get("alternatives") or []),
                  f"{label}:alternatives-listed", str(c.get("alt_rows")))

            # the forks
            check(c.get("fork_rows") == len(forks), f"{label}:fork-rows",
                  f"{c.get('fork_rows')} != {len(forks)}")
            check(c.get("fork_unattended") == n_unattended, f"{label}:unattended-forks-marked",
                  f"{c.get('fork_unattended')} != {n_unattended}")
            if n_unattended:
                check("default applied (unattended)" in (c.get("forks_text") or ""),
                      f"{label}:unattended-fork-says-so")

            # 10: the hook
            check(c.get("hook"), f"{label}:window.__training")
            check(c.get("hook_version") == "vega-train/training-render@1",
                  f"{label}:hook-version", str(c.get("hook_version")))
            check(c.get("hook_runs") == len(rows) and c.get("hook_run_ids") == ids,
                  f"{label}:hook-runs-are-the-ledger", str(c.get("hook_run_ids")))
            check(c.get("hook_selection") == selection.get("run"), f"{label}:hook-selection",
                  str(c.get("hook_selection")))
            check(c.get("hook_diagnostics") == len(diags), f"{label}:hook-diagnostics",
                  str(c.get("hook_diagnostics")))
            check(c.get("hook_history_missing"), f"{label}:hook-history-of-unknown-run-is-null")
            if not degraded and ids:
                first = histories.get(ids[0]) or {}
                check(c.get("hook_history_keys") == len(first),
                      f"{label}:hook-history-is-the-record",
                      f"{c.get('hook_history_keys')} != {len(first)} keys")
                check(c.get("hook_history_epochs") == len(first.get("epochs") or []),
                      f"{label}:hook-history-epochs",
                      f"{c.get('hook_history_epochs')} != {len(first.get('epochs') or [])}")

            # 11 / 12: errors, and the degraded pass
            check(c.get("tab_activates"), f"{label}:tab-button-activates-pane")
            check(not R["errors"], f"{label}:no-js-errors", "; ".join(R["errors"])[:300])
            if degraded:
                check((c.get("curve_lines") or 0) == 0, f"{label}:no-curve-invented",
                      str(c.get("curve_lines")))
                check((c.get("curves_empty") or 0) >= 1, f"{label}:curve-empty-state")
                check("carries no history for" in (c.get("curves_text") or ""),
                      f"{label}:degraded-pane-says-so")
                after = "reselect" if len(rows) == 1 else "switch"
                check("carries no history for" in (c.get("curves_text_after") or ""),
                      f"{label}:degraded-pane-still-says-so-after-{after}")
                check(c.get("ledger_rows") == len(rows), f"{label}:ledger-survives-degradation",
                      f"{c.get('ledger_rows')} != {len(rows)}")
            if err and "ERROR:" in err:
                bad = [ln for ln in err.splitlines() if "ERROR:" in ln and "GPU" not in ln
                       and "gpu" not in ln and "dbus" not in ln.lower()
                       and "sandbox" not in ln.lower() and "google_apis/gcm/" not in ln
                       and "page_load_metrics" not in ln]    # paint timing under load: the browser's, not the page's
                check(not bad, f"{label}:chromium-stderr", "; ".join(bad)[:200])

    if failures:
        print(f"ERROR: training layer smoke test failed for {path} "
              f"({len(failures)} of {checks} checks)")
        return 1
    print(f"OK: training layer smoke test passed for {path} ({checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
