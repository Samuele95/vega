#!/usr/bin/env python3
"""
smoke_shape.py — headless-Chromium check that the /dataset-shaper `shape` layer renders.

    python3 smoke_shape.py OUT.html [--strict]

A copy of OUT.html is opened in headless Chromium with a probe appended that exercises the
page and writes its findings into a <pre id="shape-smoke">; the DOM dump is then parsed.

  1. the tab button (data-tab="shape") sits in the host's nav.tabs / nav[role=tablist];
  2. the pane section.tab-pane[data-layer="shape"] is mounted and hidden by default;
  3. the four surfaces exist (data-view = before-after | recipe | lineage | verification);
  4. every recipe step has a row, and every row shows a source chip — the provenance is the
     one thing this layer may never render blank;
  5. the lineage graph draws one line per column and marks the dropped ones;
  6. the verification panel shows the structural and determinism verdicts;
  7. window.__shape exposes steps(), lineage() and verification();
  8. no uncaught JS error;
  9. on a copy whose lineage is emptied, the pane still renders and says so.

Exit 0 on success, 1 on a failed check (or, with --strict, when no browser is available).
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
DATA_RE = re.compile(r'(<script\s+id="layer-shape-data"[^>]*>)([\s\S]*?)(</script>)', re.I)

PROBE = r"""
<script id="shape-smoke-probe">
(function(){
  var R = { errors: [], checks: {} };
  window.addEventListener('error', function(e){ R.errors.push('onerror: ' + (e.message || e)); });
  var origErr = console.error;
  console.error = function(){ R.errors.push('console.error: ' + Array.prototype.slice.call(arguments).map(String).join(' ')); if (origErr) origErr.apply(console, arguments); };
  function q(s){ return document.querySelector(s); }
  function run(){
    try {
      var nav = q('nav.tabs') || q('nav[role="tablist"]');
      R.checks.tab_button_in_nav = !!(nav && nav.querySelector('button[data-tab="shape"]'));
      var pane = q('section.tab-pane[data-layer="shape"]');
      R.checks.pane_mounted = !!pane;
      R.checks.pane_hidden_by_default = !!pane && !pane.classList.contains('active');
      R.checks.views = {};
      ['before-after','recipe','lineage','verification'].forEach(function(v){
        R.checks.views[v] = !!(pane && pane.querySelector('[data-view="' + v + '"]'));
      });
      R.checks.steps = pane ? pane.querySelectorAll('[data-view="recipe"] .s-step').length : 0;
      R.checks.source_chips = 0;
      if (pane) Array.prototype.forEach.call(pane.querySelectorAll('[data-view="recipe"] .s-step'), function(s){
        var c = s.querySelector('.s-src-geometry, .s-src-analysis, .s-src-user, .s-src-default');
        if (c && c.textContent && c.textContent !== '—') R.checks.source_chips++;
      });
      R.checks.phases = pane ? pane.querySelectorAll('[data-view="recipe"] .s-phase').length : 0;
      R.checks.lineage_lines = pane ? pane.querySelectorAll('[data-view="lineage"] .s-lline').length : 0;
      R.checks.lineage_dead = pane ? pane.querySelectorAll('[data-view="lineage"] .s-lline.s-dead').length : 0;
      R.checks.lineage_empty = !!(pane && pane.querySelector('[data-view="lineage"] .s-empty'));
      R.checks.verdict_chips = pane ? pane.querySelectorAll('[data-view="verification"] .s-chip').length : 0;
      R.checks.verdict_text = pane ? (pane.querySelector('[data-view="verification"]').textContent || '') : '';
      R.checks.stats = pane ? pane.querySelectorAll('[data-view="before-after"] .s-stat').length : 0;
      R.checks.hook = !!(window.__shape && typeof window.__shape.steps === 'function'
                         && typeof window.__shape.lineage === 'function'
                         && typeof window.__shape.verification === 'function');
      if (R.checks.hook) R.checks.hook_steps = window.__shape.steps().length;
      var det = pane ? pane.querySelector('[data-view="recipe"] details') : null;
      if (det) { det.open = true; R.checks.details_open = det.open; }
    } catch (e) { R.errors.push('probe: ' + (e && e.stack || e)); }
    var pre = document.createElement('pre'); pre.id = 'shape-smoke';
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
            doc["lineage"] = {}
            html = (html[:m.start(2)] + "\n" +
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n" +
                    html[m.end(2):])
    idx = html.rfind("</body>")
    html = html[:idx] + PROBE + html[idx:] if idx >= 0 else html + PROBE
    p = Path(tmpdir) / name
    p.write_text(html, encoding="utf-8")
    return p


def parse_results(dom):
    m = re.search(r'<pre id="shape-smoke">([\s\S]*?)</pre>', dom)
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
        print("ERROR: no parseable layer-shape-data script in the file")
        return 1
    chrome = find_chrome()
    if not chrome:
        msg = f"no headless browser available (checked {CHROME_CANDIDATES})"
        if a.strict:
            print(f"ERROR: {msg} — failing under --strict")
            return 1
        print(f"WARN: {msg}; the render is unverified")
        return 0

    n_steps = len(((doc.get("recipe") or {}).get("steps")) or [])
    lin = (doc.get("lineage") or {}).get("columns") or {}
    n_cols = len(lin)
    n_dead = len([c for c, v in lin.items() if not v.get("present_in_output")])
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
        for label, empty in (("full", False), ("no-lineage", True)):
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
            for v in ("before-after", "recipe", "lineage", "verification"):
                check((c.get("views") or {}).get(v), f"{label}:view-{v}")
            check(c.get("hook"), f"{label}:window.__shape")
            check(c.get("hook_steps") == n_steps, f"{label}:hook-steps",
                  f"{c.get('hook_steps')} != {n_steps}")
            check(c.get("steps") == n_steps, f"{label}:step-rows",
                  f"{c.get('steps')} != {n_steps}")
            check(c.get("source_chips") == n_steps, f"{label}:every-step-shows-its-source",
                  f"{c.get('source_chips')} of {n_steps}")
            check(c.get("phases", 0) >= 3, f"{label}:phases-grouped", str(c.get("phases")))
            check(c.get("stats", 0) >= 2, f"{label}:before-after-stats", str(c.get("stats")))
            vt = c.get("verdict_text") or ""
            check("structural" in vt and "determinism" in vt, f"{label}:verdicts-shown")
            check(not R["errors"], f"{label}:no-js-errors", "; ".join(R["errors"])[:300])
            if not empty:
                check(c.get("lineage_lines") == n_cols, f"{label}:lineage-lines",
                      f"{c.get('lineage_lines')} != {n_cols}")
                check(c.get("lineage_dead") == n_dead, f"{label}:lineage-dropped-marked",
                      f"{c.get('lineage_dead')} != {n_dead}")
                check(c.get("details_open"), f"{label}:alternatives-expand")
            else:
                check(c.get("lineage_empty"), f"{label}:lineage-empty-state")
            if err and "ERROR:" in err:
                bad = [ln for ln in err.splitlines() if "ERROR:" in ln and "GPU" not in ln
                       and "gpu" not in ln and "dbus" not in ln.lower()
                       and "sandbox" not in ln.lower()
                       and "page_load_metrics" not in ln]    # paint timing under load: the browser's, not the page's
                check(not bad, f"{label}:chromium-stderr", "; ".join(bad)[:200])

    if failures:
        print(f"ERROR: shape layer smoke test failed for {path} "
              f"({len(failures)} of {checks} checks)")
        return 1
    print(f"OK: shape layer smoke test passed for {path} ({checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
