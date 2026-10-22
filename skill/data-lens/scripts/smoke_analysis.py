#!/usr/bin/env python3
"""
smoke_analysis.py — headless-Chromium check that the /data-lens `analysis` layer renders.

    python3 smoke_analysis.py OUT.html [--strict]

The render script only runs in a browser, so a copy of OUT.html is opened in headless
Chromium with a probe appended that exercises the page and writes its findings into a
<pre id="analysis-smoke">; the DOM dump is then parsed. Checks:

  1. the tab button (data-tab="analysis") sits in the host's nav.tabs / nav[role=tablist];
  2. the pane section.tab-pane[data-layer="analysis"] is mounted and hidden by default;
  3. the four surfaces exist (data-view = context | findings | modules | transcript);
  4. every finding in the data has a card, and severity filtering narrows them;
  5. every module in the data has a panel, and a skipped module states its reason;
  6. every turn has a re-run control that reveals its stored code, and an ungrounded turn
     is visibly marked;
  7. the shipped figures render as SVG inside the pane;
  8. no uncaught JS error (window.onerror, console.error, Chromium stderr);
  9. on a copy with `findings` and `transcript` emptied, the pane still renders and says so
     — the empty states are part of the contract, not an accident.

Exit 0 on success, 1 on a failed check (or, with --strict, when no browser is available);
without --strict a missing browser prints WARN and exits 0. Markers: OK: / WARN: / ERROR:.
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
DATA_RE = re.compile(r'(<script\s+id="layer-analysis-data"[^>]*>)([\s\S]*?)(</script>)', re.I)

PROBE = r"""
<script id="analysis-smoke-probe">
(function(){
  var R = { errors: [], checks: {} };
  window.addEventListener('error', function(e){ R.errors.push('onerror: ' + (e.message || e)); });
  var origErr = console.error;
  console.error = function(){ R.errors.push('console.error: ' + Array.prototype.slice.call(arguments).map(String).join(' ')); if (origErr) origErr.apply(console, arguments); };
  function q(s){ return document.querySelector(s); }
  function run(){
    try {
      var nav = q('nav.tabs') || q('nav[role="tablist"]');
      R.checks.tab_button_in_nav = !!(nav && nav.querySelector('button[data-tab="analysis"]'));
      var pane = q('section.tab-pane[data-layer="analysis"]');
      R.checks.pane_mounted = !!pane;
      R.checks.pane_hidden_by_default = !!pane && !pane.classList.contains('active');
      R.checks.views = {};
      ['context','findings','modules','transcript'].forEach(function(v){
        R.checks.views[v] = !!(pane && pane.querySelector('[data-view="' + v + '"]'));
      });
      R.checks.hook = !!(window.__analysis && typeof window.__analysis.findings === 'function'
                         && typeof window.__analysis.turns === 'function');
      R.checks.cards = pane ? pane.querySelectorAll('[data-view="findings"] .a-card').length : 0;
      R.checks.module_panels = pane ? pane.querySelectorAll('[data-view="modules"] .a-mod-panel').length : 0;
      R.checks.skipped_panels = 0;
      if (pane) Array.prototype.forEach.call(pane.querySelectorAll('[data-view="modules"] .a-mod-panel'), function(d){
        if (/skipped/.test(d.querySelector('summary').textContent)) R.checks.skipped_panels++;
      });
      R.checks.turns = pane ? pane.querySelectorAll('[data-view="transcript"] .a-turn').length : 0;
      R.checks.ungrounded = pane ? pane.querySelectorAll('[data-view="transcript"] .a-ungrounded').length : 0;
      R.checks.svgs = pane ? pane.querySelectorAll('svg.a-svg').length : 0;
      R.checks.empty_states = pane ? pane.querySelectorAll('.a-empty').length : 0;
      R.checks.context_chips = pane ? pane.querySelectorAll('[data-view="context"] .a-chip').length : 0;
      /* a pass with no dialogue turn (--no-dialogue, --unattended) says so in the transcript */
      R.checks.transcript_empty_state = !!(pane && pane.querySelector('[data-view="transcript"] .a-empty'));
      R.checks.rerun_controls = pane ? pane.querySelectorAll('[data-view="transcript"] .a-rerun').length : 0;
      /* the re-run control reveals the stored code */
      var btn = pane ? pane.querySelector('[data-view="transcript"] .a-rerun') : null;
      if (btn) {
        var box = btn.nextSibling;
        var before = box && box.style.display;
        btn.click();
        R.checks.rerun_reveals = !!(box && box.style.display === 'block'
                                    && box.querySelector('.a-code'));
        R.checks.rerun_code_nonempty = !!(box && box.querySelector('.a-code')
                                          && box.querySelector('.a-code').textContent.length > 10);
        btn.click();
        R.checks.rerun_toggles_back = !!(box && box.style.display === 'none' && before === 'none');
      }
      /* severity filter narrows the board */
      var sel = pane ? pane.querySelector('[data-view="findings"] select') : null;
      if (sel && window.__analysis) {
        var all = pane.querySelectorAll('[data-view="findings"] .a-card').length;
        sel.value = 'high';
        sel.onchange();
        var high = pane.querySelectorAll('[data-view="findings"] .a-card').length;
        sel.value = 'all';
        sel.onchange();
        var back = pane.querySelectorAll('[data-view="findings"] .a-card').length;
        R.checks.filter = { all: all, high: high, back: back,
                            narrows: high <= all, restores: back === all };
      }
      /* sorting a module table must not throw */
      var th = pane ? pane.querySelector('[data-view="modules"] .a-sortable') : null;
      if (th) { th.click(); R.checks.sort_ok = true; }
    } catch (e) { R.errors.push('probe: ' + (e && e.stack || e)); }
    var pre = document.createElement('pre'); pre.id = 'analysis-smoke';
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


def dump_dom(chrome, path, budget_ms=6000):
    proc = subprocess.run(
        [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
         f"--virtual-time-budget={budget_ms}", "--enable-logging=stderr", "--v=0",
         "--dump-dom", f"file://{path.resolve()}"], capture_output=True, text=True, timeout=90)
    return proc.stdout, proc.stderr


def layer_data(html):
    m = DATA_RE.search(html)
    if not m:
        return None, None
    try:
        return json.loads(m.group(2)), m
    except json.JSONDecodeError:
        return None, m


def probe_copy(html, tmpdir, name, empty=False):
    if empty:
        doc, m = layer_data(html)
        if doc is not None:
            doc["findings"] = []
            doc["transcript"] = []
            html = (html[:m.start(2)] + "\n" +
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n" +
                    html[m.end(2):])
    idx = html.rfind("</body>")
    html = html[:idx] + PROBE + html[idx:] if idx >= 0 else html + PROBE
    p = Path(tmpdir) / name
    p.write_text(html, encoding="utf-8")
    return p


def parse_results(dom):
    m = re.search(r'<pre id="analysis-smoke">([\s\S]*?)</pre>', dom)
    if not m:
        return None
    txt = (m.group(1).replace("&quot;", '"').replace("&lt;", "<")
           .replace("&gt;", ">").replace("&amp;", "&"))
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("html", help="layered HTML output to check")
    ap.add_argument("--strict", action="store_true", help="exit 1 when no browser is available")
    a = ap.parse_args(argv)
    path = Path(a.html)
    if not path.is_file():
        print(f"ERROR: {path} not found")
        return 1
    html = path.read_text(encoding="utf-8")
    doc, _ = layer_data(html)
    if doc is None:
        print("ERROR: no parseable layer-analysis-data script in the file")
        return 1
    chrome = find_chrome()
    if not chrome:
        msg = f"no headless browser available (checked {CHROME_CANDIDATES})"
        if a.strict:
            print(f"ERROR: {msg} — failing under --strict")
            return 1
        print(f"WARN: {msg}; the render is unverified")
        return 0

    failures, checks = [], 0

    def check(cond, name, detail=""):
        nonlocal checks
        checks += 1
        if cond:
            print(f"  OK   {name}")
        else:
            failures.append(f"{name}{': ' + detail if detail else ''}")
            print(f"  FAIL {name}{': ' + detail if detail else ''}")

    n_find = len(doc.get("findings") or [])
    n_turn = len(doc.get("transcript") or [])
    n_mod = len(doc.get("modules") or {})
    n_skip = len([m for m in (doc.get("modules") or {}).values() if not m.get("ran")])
    n_ung = len([t for t in (doc.get("transcript") or []) if t.get("grounded") is False])
    n_fig = len(doc.get("figures") or [])

    with tempfile.TemporaryDirectory() as td:
        for label, empty in (("full", False), ("empty", True)):
            p = probe_copy(html, td, f"smoke-{label}.html", empty=empty)
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
            check(c.get("tab_button_in_nav"), f"{label}:tab-button")
            check(c.get("pane_mounted"), f"{label}:pane")
            check(c.get("pane_hidden_by_default"), f"{label}:pane-inactive-by-default")
            for v in ("context", "findings", "modules", "transcript"):
                check((c.get("views") or {}).get(v), f"{label}:view-{v}")
            check(c.get("hook"), f"{label}:window.__analysis")
            check(c.get("context_chips", 0) > 0, f"{label}:context-chips",
                  str(c.get("context_chips")))
            check(c.get("module_panels") == n_mod, f"{label}:module-panels",
                  f"{c.get('module_panels')} != {n_mod}")
            check(c.get("skipped_panels") == n_skip, f"{label}:skipped-panels-state",
                  f"{c.get('skipped_panels')} != {n_skip}")
            check(not R["errors"], f"{label}:no-js-errors", "; ".join(R["errors"])[:300])
            if not empty:
                check(c.get("cards") == n_find, f"{label}:finding-cards",
                      f"{c.get('cards')} != {n_find}")
                check(c.get("turns") == n_turn, f"{label}:turns", f"{c.get('turns')} != {n_turn}")
                check(c.get("ungrounded") == n_ung, f"{label}:ungrounded-marked",
                      f"{c.get('ungrounded')} != {n_ung}")
                if n_turn:
                    check(c.get("rerun_reveals"), f"{label}:rerun-reveals-code")
                    check(c.get("rerun_code_nonempty"), f"{label}:rerun-code-nonempty")
                    check(c.get("rerun_toggles_back"), f"{label}:rerun-toggles-back")
                else:
                    # an automatic pass only (--no-dialogue, or --unattended without
                    # --questions) has no turn to re-run: the transcript says so instead
                    check(c.get("transcript_empty_state"), f"{label}:transcript-empty-state")
                    check(c.get("rerun_controls") == 0, f"{label}:no-rerun-controls",
                          str(c.get("rerun_controls")))
                f = c.get("filter") or {}
                check(f.get("narrows") and f.get("restores"), f"{label}:severity-filter",
                      json.dumps(f))
                if n_fig:
                    check(c.get("svgs", 0) > 0, f"{label}:figures-render",
                          f"{c.get('svgs')} svg nodes for {n_fig} shipped figures")
            else:
                check(c.get("cards") == 0, f"{label}:no-cards")
                check(c.get("empty_states", 0) >= 2, f"{label}:empty-states",
                      str(c.get("empty_states")))
            if err and "ERROR:" in err:
                bad = [ln for ln in err.splitlines() if "ERROR:" in ln
                       and "GPU" not in ln and "gpu" not in ln and "dbus" not in ln.lower()
                       and "sandbox" not in ln.lower()
                       and "page_load_metrics" not in ln]    # paint timing under load: the browser's, not the page's
                check(not bad, f"{label}:chromium-stderr", "; ".join(bad)[:200])

    if failures:
        print(f"ERROR: analysis layer smoke test failed for {path} "
              f"({len(failures)} of {checks} checks)")
        return 1
    print(f"OK: analysis layer smoke test passed for {path} ({checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
