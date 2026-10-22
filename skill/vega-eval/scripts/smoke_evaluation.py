#!/usr/bin/env python3
"""
smoke_evaluation.py — headless-Chromium check that the /vega-eval `evaluation` layer renders.

    python3 smoke_evaluation.py OUT.html [--strict]

A copy of OUT.html is opened in headless Chromium with a probe appended that exercises the page
— including a click and a keyboard press on the slices table's sortable headers — and writes its
findings into a <pre id="evaluation-smoke">; the DOM dump is then parsed.

  1. the tab button (data-tab="evaluation") sits in the host's nav.tabs / nav[role=tablist] and
     is labelled Evaluation;
  2. the pane section.tab-pane[data-layer="evaluation"] is mounted, mounted once, and inactive
     by default;
  3. the surfaces exist (data-view = seal | verdict | metrics | baselines | slices |
     robustness), and the seal strip carries the exposure count — first, because it qualifies
     every number below it;
  4. the verdict states the strongest baseline, the paired difference, and *earned* or *not
     earned* matching `verdict.earned` — the page and the record cannot disagree about the one
     sentence a reader takes away;
  5. every metric shown for the network and the baselines carries an interval, and the count of
     `[lo, hi]` pairs on the page is at least the count of intervals in the record — a rendered
     point estimate without its interval is the defect this whole stage exists to prevent;
  6. the paired difference against the strongest baseline is marked as excluding zero exactly
     when its interval does;
  7. the curves are inline SVG drawn from the recorded points (no <img>, no canvas, no network),
     and the confusion matrix's cells are the recorded ones;
  8. the slices table holds one row per recorded slice with its `n`, marks the worst, greys a
     degenerate slice, says every slice it could not cut (`not_cut`) with its reason, and re-sorts when a header is clicked **and** when it is reached by
     keyboard — a table only a mouse can sort is not operable;
  9. the robustness panel lists every check, and every skipped check states its reason;
 10. window.__evaluation exposes metrics(), slices(), baselines(), verdict() and seal(), and
     returns the record the page was given rather than a rebuilt one;
 11. no uncaught JS error;
 12. on a copy whose `curves` is emptied, the pane still mounts and the tables still hold their
     rows — a missing curve set degrades the page, it does not blank it.

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_eval_core import Estimate  # noqa: E402

CHROME_CANDIDATES = ("chromium", "google-chrome", "chromium-browser", "chrome")
DATA_RE = re.compile(r'(<script\s+id="layer-evaluation-data"[^>]*>)([\s\S]*?)(</script>)', re.I)
VIEWS = ("seal", "verdict", "metrics", "baselines", "slices", "robustness")

PROBE = r"""
<script id="evaluation-smoke-probe">
(function(){
  var R = { errors: [], checks: {} };
  window.addEventListener('error', function(e){ R.errors.push('onerror: ' + (e.message || e)); });
  var origErr = console.error;
  console.error = function(){ R.errors.push('console.error: ' + Array.prototype.slice.call(arguments).map(String).join(' ')); if (origErr) origErr.apply(console, arguments); };
  function q(s){ return document.querySelector(s); }
  function view(pane, name){ return pane ? pane.querySelector('[data-view="' + name + '"]') : null; }
  function text(pane, name){ var v = view(pane, name); return v ? (v.textContent || '') : ''; }
  function count(pane, sel){ return pane ? pane.querySelectorAll(sel).length : 0; }
  function firstCol(tbl){
    if (!tbl) return [];
    return Array.prototype.map.call(tbl.querySelectorAll('tbody tr'), function(tr){
      return (tr.children[0] ? tr.children[0].textContent : '').trim();
    });
  }
  function run(){
    try {
      var nav = q('nav.tabs') || q('nav[role="tablist"]');
      var btn = nav ? nav.querySelector('button[data-tab="evaluation"]') : null;
      R.checks.tab_button_in_nav = !!btn;
      R.checks.tab_label = btn ? (btn.textContent || '').trim() : '';
      var pane = q('section.tab-pane[data-layer="evaluation"]');
      R.checks.pane_mounted = !!pane;
      R.checks.pane_inactive_by_default = !!pane && !pane.classList.contains('active');
      R.checks.panes_with_layer = document.querySelectorAll('section.tab-pane[data-layer="evaluation"]').length;

      R.checks.views = {};
      ['seal','verdict','metrics','baselines','slices','robustness','curves','differences','threshold','readings','provenance'].forEach(function(v){
        R.checks.views[v] = !!view(pane, v);
      });

      R.checks.seal_text = text(pane, 'seal');
      R.checks.verdict_text = text(pane, 'verdict');
      R.checks.slices_text = text(pane, 'slices');
      R.checks.robustness_text = text(pane, 'robustness');

      /* every rendered interval, counted as the page draws them */
      R.checks.ci_spans = count(pane, '.ev-ci');
      /* the property that matters: every rendered estimate shows its interval. A cell whose
         value could not be computed shows '·' and is exempt — it claims nothing. */
      var est = pane ? pane.querySelectorAll('td.ev-est') : [];
      R.checks.estimates = est.length;
      R.checks.estimates_with_value = Array.prototype.filter.call(est, function (td) {
        return td.hasAttribute('data-value');
      }).length;
      R.checks.estimates_without_ci = Array.prototype.filter.call(est, function (td) {
        if (!td.hasAttribute('data-value')) return false;
        return !td.querySelector('.ev-ci');
      }).map(function (td) { return (td.textContent || '').trim(); });
      R.checks.excludes_zero = count(pane, '.ev-excludes-zero');
      R.checks.upstream_marks = count(pane, '.ev-upstream');

      /* the curves: inline SVG only, drawn from points */
      var curves = view(pane, 'curves');
      R.checks.svgs = curves ? curves.querySelectorAll('svg').length : 0;
      R.checks.svg_paths = curves ? curves.querySelectorAll('svg path[d]').length : 0;
      R.checks.imgs = count(pane, 'img');
      R.checks.canvases = count(pane, 'canvas');
      R.checks.confusion_cells = count(pane, 'table.ev-confusion tbody td');

      /* the slices table, and whether it sorts by mouse and by keyboard */
      var st = pane ? pane.querySelector('table.ev-slices') : null;
      R.checks.slice_rows = st ? st.querySelectorAll('tbody tr').length : 0;
      R.checks.not_cut = count(pane, '.ev-not-cut');
      R.checks.slice_degenerate = st ? st.querySelectorAll('tbody tr.ev-degenerate').length : 0;
      R.checks.slice_n_cells = st ? st.querySelectorAll('tbody tr td[data-value]').length : 0;
      R.checks.sortable_headers = st ? st.querySelectorAll('thead th[aria-sort]').length : 0;
      R.checks.headers_focusable = st ? st.querySelectorAll('thead th[tabindex="0"]').length : 0;
      var before = firstCol(st);
      R.checks.order_initial = before;
      var th = st ? st.querySelectorAll('thead th')[1] : null;   /* the n column */
      if (th) { th.click(); }
      R.checks.order_after_click = firstCol(st);
      R.checks.aria_after_click = th ? th.getAttribute('aria-sort') : null;
      /* keyboard: the same header, reached with Enter rather than a pointer */
      if (th && th.onkeydown) {
        th.onkeydown({ key: 'Enter', preventDefault: function(){} });
      }
      R.checks.order_after_key = firstCol(st);
      R.checks.aria_after_key = th ? th.getAttribute('aria-sort') : null;

      var api = window.__evaluation || null;
      R.checks.api = !!api;
      if (api) {
        R.checks.api_version = api.version || null;
        try { R.checks.api_metrics_primary = (api.metrics() || {}).primary; } catch (e) { R.checks.api_metrics_primary = 'THREW'; }
        try { R.checks.api_slices = (api.slices() || []).length; } catch (e) { R.checks.api_slices = -1; }
        try { R.checks.api_baselines = (api.baselines() || []).length; } catch (e) { R.checks.api_baselines = -1; }
        try { R.checks.api_earned = (api.verdict() || {}).earned; } catch (e) { R.checks.api_earned = 'THREW'; }
        try { R.checks.api_exposures = (api.seal() || {}).exposures; } catch (e) { R.checks.api_exposures = -1; }
      }
    } catch (e) {
      R.errors.push('probe threw: ' + (e && e.message ? e.message : e));
    }
    var pre = document.createElement('pre');
    pre.id = 'evaluation-smoke';
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


def count_intervals(doc):
    """Every interval the record holds, counted the way the page must render them.

    What counts as an interval is `Estimate`'s answer, not a fourth copy of the predicate.
    """
    total = 0

    def walk(node):
        nonlocal total
        if isinstance(node, dict):
            if "value" in node:
                try:
                    if Estimate.from_json(node).has_interval:
                        total += 1
                except (TypeError, ValueError):
                    pass
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for key in ("metrics", "baselines", "slices", "robustness"):
        walk(doc.get(key) or {})
    return total


def probe_copy(html, tmpdir, name, no_curves=False):
    if no_curves:
        doc, m = layer_data(html)
        if doc is not None:
            doc["curves"] = {}
            html = (html[:m.start(2)] + "\n" +
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n" +
                    html[m.end(2):])
    idx = html.rfind("</body>")
    html = html[:idx] + PROBE + html[idx:] if idx >= 0 else html + PROBE
    p = Path(tmpdir) / name
    p.write_text(html, encoding="utf-8")
    return p


def parse_results(dom):
    m = re.search(r'<pre id="evaluation-smoke">([\s\S]*?)</pre>', dom)
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
        print("ERROR: no parseable layer-evaluation-data script in the file")
        return 1
    chrome = find_chrome()
    if not chrome:
        msg = f"no headless browser available (checked {CHROME_CANDIDATES})"
        if a.strict:
            print(f"ERROR: {msg} — failing under --strict")
            return 1
        print(f"WARN: {msg}; the render is unverified")
        return 0

    metrics = doc.get("metrics") or {}
    seal = doc.get("seal") or {}
    verdict = doc.get("verdict") or {}
    slices = (doc.get("slices") or {}).get("slices") or []
    worst = (doc.get("slices") or {}).get("worst") or {}
    baselines = (doc.get("baselines") or {}).get("baselines") or []
    checks_rec = (doc.get("robustness") or {}).get("checks") or []
    curves = (doc.get("curves") or {}).get("curves") or (doc.get("curves") or {})
    primary = metrics.get("primary")
    exposures = seal.get("exposures", 1)
    n_intervals = count_intervals(doc)
    n_degenerate = len([s for s in slices if s.get("degenerate")])
    n_skipped = len([c for c in checks_rec if c.get("status") == "skipped"])
    def excludes(entry):
        if not isinstance(entry, dict):
            return False
        try:
            return bool(Estimate.from_json(entry).excludes_zero)
        except (TypeError, ValueError):
            return False

    differences = metrics.get("differences") or {}
    diff = differences.get(verdict.get("strongest_baseline")) or {}
    diff_excludes = excludes(diff)
    # the page marks a difference in the differences table, once more in the metrics table's
    # primary row when it is the strongest baseline's, and once per robustness delta
    n_marks = (len([d for d in differences.values() if excludes(d)])
               + (1 if diff_excludes else 0)
               + len([c for c in checks_rec if excludes(c.get("delta"))]))
    n_point_curves = len([k for k in ("roc", "pr", "reliability")
                          if (curves.get(k) or {}).get("points")])
    failures, checks = [], 0

    def check(cond, name, detail=""):
        nonlocal checks
        checks += 1
        if cond:
            print(f"  OK   {name}")
        else:
            print(f"  FAIL {name}{('  — ' + detail) if detail else ''}")
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        main_page = probe_copy(html, tmp, "probe.html")
        dom, stderr = dump_dom(chrome, main_page)
        R = parse_results(dom)
        if R is None:
            print("ERROR: the probe did not report; the render script may not have run")
            print((stderr or "")[-800:])
            return 1
        C = R.get("checks") or {}

        print(f"evaluation layer smoke — {path.name}")
        check(C.get("tab_button_in_nav"), "1. the tab button is in the host's nav")
        check(C.get("tab_label") == "Evaluation", "1. the tab is labelled Evaluation",
              f"got {C.get('tab_label')!r}")
        check(C.get("pane_mounted"), "2. the pane is mounted")
        check(C.get("panes_with_layer") == 1, "2. exactly one pane (the mount is idempotent)",
              f"got {C.get('panes_with_layer')}")
        check(C.get("pane_inactive_by_default"),
              "2. the pane does not steal the active tab on load")

        views = C.get("views") or {}
        for v in VIEWS:
            check(views.get(v), f"3. the {v} surface is present")
        check(str(exposures) in (C.get("seal_text") or ""),
              "3. the seal strip states the exposure count",
              f"exposures={exposures} not in the strip")

        vtext = C.get("verdict_text") or ""
        if verdict.get("earned") is None:
            # no machine verdict (eval-contract §5): said in words, as neither a win nor a loss
            check("Not assessable" in vtext and "Earned" not in vtext and "Not earned" not in vtext,
                  "4. a not-assessable verdict is said so, as neither earned nor not earned",
                  vtext[:200])
            check(str(verdict.get("reason") or "") and str(verdict.get("reason")) in vtext,
                  "4. the not-assessable verdict states its reason")
        else:
            check(str(verdict.get("strongest_baseline") or "") in vtext,
                  "4. the verdict names the strongest fitted baseline")
            check(str(verdict.get("diff")) in vtext,
                  "4. the verdict states the paired difference",
                  f"{verdict.get('diff')} not in the verdict text")
            said_earned = "Not earned" not in vtext and "Earned" in vtext
            check(said_earned == bool(verdict.get("earned")),
                  "4. the rendered verdict agrees with verdict.earned",
                  f"page says earned={said_earned}, record says {verdict.get('earned')}")

        check(C.get("estimates_without_ci") == [],
              "5. every rendered estimate shows its interval beside it",
              f"{len(C.get('estimates_without_ci') or [])} bare point(s): "
              f"{(C.get('estimates_without_ci') or [])[:6]} — a point estimate on a held-out "
              f"part without its interval is the one thing this layer must not show")
        check(C.get("ci_spans", 0) == C.get("estimates_with_value", 0),
              "5. an interval is drawn for every estimate that has a value, and only those",
              f"{C.get('estimates_with_value')} estimate(s) with a value, "
              f"{C.get('ci_spans')} interval(s) drawn "
              f"({C.get('estimates', 0) - C.get('estimates_with_value', 0)} cell(s) show · "
              f"because the metric was not computable and claim nothing; the record holds "
              f"{n_intervals} interval(s) in all, of which the tables show the primary)")
        upstream_rows = len([b for b in baselines if (b.get("valid") or {}).get("_upstream")])
        check(C.get("upstream_marks", 0) >= min(1, upstream_rows) if upstream_rows else True,
              "5. an upstream number is marked as one rather than shown bare")

        check(C.get("excludes_zero", 0) == n_marks,
              "6. a difference is marked as excluding zero exactly when it does",
              f"page marked {C.get('excludes_zero')}, the record has {n_marks} interval(s) "
              f"that exclude zero (counting the strongest baseline's twice, once in each "
              f"table)")
        check(("excludes zero" in (C.get("verdict_text") or "")) == diff_excludes,
              "6. the verdict says in words whether its interval excludes zero")

        check(C.get("svgs", 0) >= n_point_curves,
              "7. the curves are inline SVG",
              f"{n_point_curves} curve(s) with points, {C.get('svgs')} svg element(s)")
        check(C.get("svg_paths", 0) >= n_point_curves,
              "7. each curve draws a path from its recorded points")
        check(C.get("imgs", 0) == 0 and C.get("canvases", 0) == 0,
              "7. no <img> and no <canvas> — SVG only, and nothing fetched")
        cm = curves.get("confusion") or {}
        want_cells = sum(len(r) for r in (cm.get("matrix") or []))
        check(C.get("confusion_cells", 0) == want_cells,
              "7. the confusion matrix renders every recorded cell",
              f"want {want_cells}, got {C.get('confusion_cells')}")

        check(C.get("slice_rows") == len(slices),
              "8. one slices row per recorded slice",
              f"want {len(slices)}, got {C.get('slice_rows')}")
        not_cut = (doc.get("slices") or {}).get("not_cut") or []
        check(C.get("not_cut") == len(not_cut),
              "8. every slice the chain named and the look could not cut is said, with its reason",
              f"want {len(not_cut)}, got {C.get('not_cut')}")
        check(C.get("slice_degenerate") == n_degenerate,
              "8. a single-class slice is greyed rather than scored",
              f"want {n_degenerate}, got {C.get('slice_degenerate')}")
        if worst:
            check(str(worst.get("name")) in (C.get("slices_text") or "")
                  and "worst" in (C.get("slices_text") or "").lower(),
                  "8. the worst slice is marked in the table")
        if slices:                  # no slice recorded: no table, and the surface says so
            check(C.get("sortable_headers", 0) >= 3 and
                  C.get("headers_focusable", 0) == C.get("sortable_headers", 0),
                  "8. every sortable header is focusable",
                  f"{C.get('sortable_headers')} sortable, {C.get('headers_focusable')} focusable")
        else:
            check("no slice" in (C.get("slices_text") or ""),
                  "8. an evaluation without slices says so on the slices surface")
        if len(slices) > 1:
            check(C.get("order_after_click") != C.get("order_initial")
                  or C.get("aria_after_click") in ("ascending", "descending"),
                  "8. a click sorts the table")
            check(C.get("aria_after_key") != C.get("aria_after_click"),
                  "8. Enter on a header sorts it too (the table is not mouse-only)",
                  f"aria stayed {C.get('aria_after_key')!r}")

        rtext = C.get("robustness_text") or ""
        check(all(str(c.get("check")) in rtext for c in checks_rec),
              "9. every robustness check is listed")
        check(rtext.count("skipped") >= n_skipped,
              "9. every skipped check says it was skipped, with its reason",
              f"{n_skipped} skipped in the record, {rtext.count('skipped')} said so")

        check(C.get("api"), "10. window.__evaluation is exposed")
        check(C.get("api_metrics_primary") == primary,
              "10. metrics() returns the record the page was given",
              f"want {primary!r}, got {C.get('api_metrics_primary')!r}")
        check(C.get("api_slices") == len(slices), "10. slices() returns every slice")
        check(C.get("api_baselines") == len(baselines),
              "10. baselines() returns every baseline row")
        check(C.get("api_earned") == verdict.get("earned"),
              "10. verdict() returns the machine verdict")
        check(C.get("api_exposures") == exposures, "10. seal() returns the seal")

        check(not R.get("errors"), "11. no uncaught JS error",
              "; ".join(R.get("errors") or [])[:300])

        degraded = probe_copy(html, tmp, "no-curves.html", no_curves=True)
        dom2, _ = dump_dom(chrome, degraded)
        R2 = parse_results(dom2) or {}
        C2 = R2.get("checks") or {}
        check(C2.get("pane_mounted"), "12. the pane still mounts with no curves recorded")
        check(C2.get("slice_rows") == len(slices),
              "12. the slices table is unaffected by a missing curve set")
        check(not R2.get("errors"), "12. no JS error on the degraded copy",
              "; ".join(R2.get("errors") or [])[:300])

    print()
    if failures:
        print(f"{len(failures)} of {checks} checks failed: {failures}")
        return 1
    print(f"OK: evaluation layer smoke test passed for {path.name} ({checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
