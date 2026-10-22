#!/usr/bin/env python3
"""
smoke_bundle.py — headless-Chromium check that the /vega-pack `bundle` layer renders.

    python3 smoke_bundle.py OUT.html [--strict]

A copy of OUT.html is opened in headless Chromium with a probe appended that exercises the page
— including a click and a keyboard press on the input-schema table's sortable headers — and
writes its findings into a <pre id="bundle-smoke">; the DOM dump is then parsed.

  1. the tab button (data-tab="bundle") sits in the host's nav.tabs / nav[role=tablist] and is
     labelled Bundle;
  2. the pane section.tab-pane[data-layer="bundle"] is mounted, mounted once, and inactive by
     default;
  3. the surfaces exist (data-view = verdict | summary | roundtrip | examples | schema | usage
     | readings), and the verdict is the *first* of them — this is the page a reader opens to
     decide whether to use the model, and the sentence that governs that decision cannot sit
     below the fold;
  4. the verdict agrees with `manifest.chain.evaluation.verdict`, and distinguishes its three
     states: earned, not earned, and beaten by a baseline. `beaten_by_baseline: true` is a
     stronger and different statement than `earned: false` — a page that renders them alike
     tells a reader that a comparison was undecided when the record says a logistic regression
     won. Every recorded stance is rendered, inside the verdict box rather than folded away:
     the stance is the condition under which an unearned bundle ships;
  5. every input column the record holds appears in the table, and every column the record
     gives a domain for renders that domain — the property that matters here, as an interval
     beside every estimate is the property that matters in /vega-eval. A column shown without
     the values or the range seen in training invites a caller to extrapolate past bounds
     nothing was ever measured at. The output is read by its READING (`MANIFEST.output`,
     vega/output@2): the output table describes exactly the field patterns the reading's
     fields expand from (`p_<class>`, `value_h<h>`, … — `output_record.schema_output` over the
     record), one row per pattern, and the summary names every concrete field;
  6. the round trip — the claim this whole stage rests on — shows the recorded status, the
     recorded row count, and the tolerance *formula* rather than only its number; a pass
     without the bound it was measured against is not a result. `shaper_transform_check` is
     rendered as its own result, because it tests a different file;
  7. `examples.kind` is visible, and `test` and `synthetic` are not rendered alike: one says
     the bundle reproduces the evaluation, the other that it reproduces only itself;
  8. the exports table holds every recorded export and marks a failing parity; the serving
     record states its mode;
  9. every code finding is listed with its severity, and each audited tool's judgement (its
     audits, the confirmed and the unconfirmed findings) is stated, each confirmed — advisory —
     finding as its open item for /architect and /improve; the usage snippets are the
     manifest's verbatim; the model card is *linked* as a sibling file and not inlined; the notebook is
     named with its executed state;
 10. every marker is listed, and a WARN marker is marked as one;
 11. window.__bundle exposes manifest() and version, and returns the record the page was given
     rather than a rebuilt one;
 12. no <img>, no <canvas>, no uncaught JS error, and the shipped render is within the 40 KB
     the contract allows;
 13. a degraded copy — empty `code_findings`, no `exports` key, `examples.kind: synthetic` —
     still mounts, still holds its schema rows, and says plainly that its shipped test proves
     only that the bundle reproduces itself.

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

from output_record import schema_output  # noqa: E402
from vega_pack_core import CATALOG  # noqa: E402

CHROME_CANDIDATES = ("chromium", "google-chrome", "chromium-browser", "chrome")
DATA_RE = re.compile(r'(<script\s+id="layer-bundle-data"[^>]*>)([\s\S]*?)(</script>)', re.I)
VIEWS = ("verdict", "summary", "roundtrip", "examples", "schema", "usage", "readings")
RENDER_JS = HERE.parent / "assets" / "bundle-render.js"
JS_LIMIT = 40 * 1024

PROBE = r"""
<script id="bundle-smoke-probe">
(function(){
  var R = { errors: [], checks: {} };
  window.addEventListener('error', function(e){ R.errors.push('onerror: ' + (e.message || e)); });
  var origErr = console.error;
  console.error = function(){ R.errors.push('console.error: ' + Array.prototype.slice.call(arguments).map(String).join(' ')); if (origErr) origErr.apply(console, arguments); };
  function q(s){ return document.querySelector(s); }
  function view(pane, name){ return pane ? pane.querySelector('[data-view="' + name + '"]') : null; }
  function text(pane, name){ var v = view(pane, name); return v ? (v.textContent || '') : ''; }
  function count(pane, sel){ return pane ? pane.querySelectorAll(sel).length : 0; }
  function attrs(pane, sel, name){
    if (!pane) return [];
    return Array.prototype.map.call(pane.querySelectorAll(sel), function(n){
      return n.getAttribute(name);
    });
  }
  function firstCol(tbl){
    if (!tbl) return [];
    return Array.prototype.map.call(tbl.querySelectorAll('tbody tr'), function(tr){
      return (tr.children[0] ? tr.children[0].textContent : '').trim();
    });
  }
  function statusIn(pane, check){
    var host = pane ? pane.querySelector('[data-check="' + check + '"]') : null;
    var s = host ? host.querySelector('.bn-status') : null;
    return { text: host ? (host.textContent || '') : '', status: s ? s.getAttribute('data-status') : null };
  }
  function run(){
    try {
      var nav = q('nav.tabs') || q('nav[role="tablist"]');
      var btn = nav ? nav.querySelector('button[data-tab="bundle"]') : null;
      R.checks.tab_button_in_nav = !!btn;
      R.checks.tab_label = btn ? (btn.textContent || '').trim() : '';
      var pane = q('section.tab-pane[data-layer="bundle"]');
      R.checks.pane_mounted = !!pane;
      R.checks.pane_inactive_by_default = !!pane && !pane.classList.contains('active');
      R.checks.panes_with_layer = document.querySelectorAll('section.tab-pane[data-layer="bundle"]').length;

      R.checks.views = {};
      ['verdict','summary','digests','roundtrip','examples','schema','exports','serve','findings','usage','readings','provenance','stances'].forEach(function(v){
        R.checks.views[v] = !!view(pane, v);
      });
      var firstView = pane ? pane.querySelector('[data-view]') : null;
      R.checks.first_view = firstView ? firstView.getAttribute('data-view') : null;

      /* ---- the verdict, and the stance that ships with it */
      var vbox = view(pane, 'verdict');
      R.checks.verdict_state = vbox ? vbox.getAttribute('data-verdict') : null;
      R.checks.verdict_text = vbox ? (vbox.textContent || '') : '';
      var flag = vbox ? vbox.querySelector('.bn-flag') : null;
      R.checks.flag_text = flag ? (flag.textContent || '').trim() : '';
      R.checks.stance_in_verdict = !!(vbox && vbox.querySelector('[data-view="stances"]'));
      R.checks.stances = pane ? Array.prototype.map.call(
        pane.querySelectorAll('.bn-stance-list li'), function(li){ return (li.textContent||'').trim(); }) : [];
      R.checks.exposures_text = pane ? ((pane.querySelector('.bn-exposures')||{}).textContent || '') : '';
      R.checks.ci_bars = count(pane, 'svg.bn-ci-bar');

      /* ---- the manifest summary */
      R.checks.summary_text = text(pane, 'summary');
      R.checks.digest_cells = count(pane, '.bn-digest');

      /* ---- the round trip */
      R.checks.rt = statusIn(pane, 'roundtrip');
      R.checks.tol = statusIn(pane, 'tolerance');
      R.checks.shaper = statusIn(pane, 'shaper');
      var rule = pane ? pane.querySelector('.bn-rule') : null;
      R.checks.rule_text = rule ? (rule.textContent || '').trim() : null;

      /* ---- examples */
      var ex = view(pane, 'examples');
      R.checks.examples_kind = ex ? ex.getAttribute('data-kind') : null;
      R.checks.examples_proves = ex ? ex.getAttribute('data-proves') : null;
      R.checks.examples_text = ex ? (ex.textContent || '') : '';

      /* ---- the input schema, and the property: every column shows the domain it was seen in */
      var st = pane ? pane.querySelector('table.bn-schema') : null;
      R.checks.schema_columns = attrs(pane, 'table.bn-schema tbody tr', 'data-column');
      R.checks.schema_domains = st ? st.querySelectorAll('td.bn-dom[data-domain]').length : 0;
      R.checks.columns_without_domain = st ? Array.prototype.filter.call(
        st.querySelectorAll('tbody tr'), function(tr){
          var d = tr.querySelector('td.bn-dom');
          return !d || !d.hasAttribute('data-domain');
        }).map(function(tr){ return tr.getAttribute('data-column'); }) : [];
      R.checks.output_fields = attrs(pane, 'table.bn-output tbody tr', 'data-field');
      R.checks.ignored_text = pane ? ((pane.querySelector('.bn-ignored')||{}).textContent || '') : '';
      R.checks.sortable_headers = st ? st.querySelectorAll('thead th[aria-sort]').length : 0;
      R.checks.headers_focusable = st ? st.querySelectorAll('thead th[tabindex="0"]').length : 0;
      R.checks.order_initial = firstCol(st);
      var th = st ? st.querySelectorAll('thead th')[0] : null;
      if (th) th.click();
      R.checks.order_after_click = firstCol(st);
      R.checks.aria_after_click = th ? th.getAttribute('aria-sort') : null;
      if (th && th.onkeydown) th.onkeydown({ key: 'Enter', preventDefault: function(){} });
      R.checks.order_after_key = firstCol(st);
      R.checks.aria_after_key = th ? th.getAttribute('aria-sort') : null;

      /* ---- exports, serving, findings */
      R.checks.export_formats = attrs(pane, 'table.bn-exports tbody tr', 'data-format');
      R.checks.export_marked = count(pane, 'table.bn-exports tbody tr.bn-warn-row');
      R.checks.export_statuses = attrs(pane, 'table.bn-exports tbody tr .bn-status', 'data-status');
      R.checks.exports_text = text(pane, 'exports');
      var srv = view(pane, 'serve');
      R.checks.serve_mode = srv ? srv.getAttribute('data-mode') : null;
      R.checks.finding_ids = attrs(pane, 'table.bn-findings tbody tr', 'data-finding');
      R.checks.finding_severities = attrs(pane, 'table.bn-findings tbody tr', 'data-severity');
      R.checks.findings_text = text(pane, 'findings');
      R.checks.judgements = attrs(pane, '[data-judgement]', 'data-judgement');
      R.checks.open_items = attrs(pane, '[data-open-item]', 'data-open-item');

      /* ---- usage, the card, the notebook */
      function usage(kind){
        var n = pane ? pane.querySelector('pre[data-usage="' + kind + '"]') : null;
        return n ? (n.textContent || '') : null;
      }
      R.checks.usage_cli = usage('cli');
      R.checks.usage_module = usage('module');
      var card = pane ? pane.querySelector('a.bn-card-link') : null;
      R.checks.card_href = card ? card.getAttribute('href') : null;
      R.checks.card_is_link = !!card && card.tagName.toLowerCase() === 'a';
      var nb = pane ? pane.querySelector('[data-notebook]') : null;
      R.checks.notebook_path = nb ? nb.getAttribute('data-notebook') : null;
      R.checks.notebook_text = nb ? (nb.textContent || '') : '';

      R.checks.readings_text = text(pane, 'readings');

      /* ---- markers */
      R.checks.markers = pane ? Array.prototype.map.call(pane.querySelectorAll('.bn-markers li'),
        function(li){ return (li.textContent || '').trim(); }) : [];
      R.checks.markers_warned = count(pane, '.bn-markers li.bn-m-warn');

      R.checks.imgs = count(pane, 'img');
      R.checks.canvases = count(pane, 'canvas');

      var api = window.__bundle || null;
      R.checks.api = !!api;
      if (api) {
        R.checks.api_version = api.version || null;
        try { R.checks.api_run = (api.manifest() || {}).run; } catch (e) { R.checks.api_run = 'THREW'; }
        try { R.checks.api_slug = (api.manifest() || {}).slug; } catch (e) { R.checks.api_slug = 'THREW'; }
        try { R.checks.api_inputs = ((api.schema() || {}).input || []).length; } catch (e) { R.checks.api_inputs = -1; }
        try { R.checks.api_rt = (api.roundtrip() || {}).status; } catch (e) { R.checks.api_rt = 'THREW'; }
        try { R.checks.api_findings = (api.findings() || []).length; } catch (e) { R.checks.api_findings = -1; }
        try { R.checks.api_markers = (api.markers() || []).length; } catch (e) { R.checks.api_markers = -1; }
      }
    } catch (e) {
      R.errors.push('probe threw: ' + (e && e.message ? e.message : e));
    }
    var pre = document.createElement('pre');
    pre.id = 'bundle-smoke';
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


def probe_copy(html, tmpdir, name, degrade=False):
    """A copy with the probe appended; `degrade` writes the copy check 13 describes.

    Degrading is done to the record, not to the render: emptying `code_findings`, dropping
    `exports` entirely, and turning the shipped examples synthetic are the three ways a real
    bundle differs from the fixture, and each one changes what the page may claim.
    """
    if degrade:
        doc, m = layer_data(html)
        if doc is not None:
            doc["code_findings"] = []
            doc.pop("exports", None)
            (doc.get("manifest") or {}).pop("exports", None)
            (doc.get("manifest") or {}).setdefault("examples", {})["kind"] = "synthetic"
            (doc["manifest"]["examples"])["source"] = "predictor"
            html = (html[:m.start(2)] + "\n" +
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n" +
                    html[m.end(2):])
    idx = html.rfind("</body>")
    html = html[:idx] + PROBE + html[idx:] if idx >= 0 else html + PROBE
    p = Path(tmpdir) / name
    p.write_text(html, encoding="utf-8")
    return p


def parse_results(dom):
    m = re.search(r'<pre id="bundle-smoke">([\s\S]*?)</pre>', dom)
    if not m:
        return None
    txt = (m.group(1).replace("&quot;", '"').replace("&lt;", "<")
           .replace("&gt;", ">").replace("&amp;", "&"))
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def reading_patterns(manifest):
    """(the field patterns the manifest's reading describes its fields by, the record's
    concrete fields): `schema_output` over `MANIFEST.output` and `MANIFEST.sample` — what the
    output table must hold, derived from the reading and never from the field names one by
    one. (None, fields) for a record that is not vega/output@2."""
    record = manifest.get("output") or {}
    fields = [str(f) for f in record.get("fields") or []]
    if record.get("schema") != CATALOG["output_readings"]["schema"]:
        return None, fields
    try:
        described = schema_output(record, manifest.get("sample"))["output"]
    except (KeyError, TypeError, ValueError):
        return None, fields
    return sorted(described), fields


def reading_checks(check, C, man) -> None:
    """Check 5's reading half: the output table holds one row per field pattern of the
    reading, and the summary names every concrete field."""
    patterns, fields = reading_patterns(man)
    shown = sorted(C.get("output_fields") or [])
    check(patterns is not None and shown == patterns,
          "5. the output table describes the reading's field patterns, one row per pattern",
          f"reading {(man.get('output') or {}).get('reading')!r} describes {patterns}, the "
          f"page {shown}")
    summary = C.get("summary_text") or ""
    check(bool(fields) and all(f in summary for f in fields),
          "5. the summary names every output field of the reading",
          f"{[f for f in fields if f not in summary]} not named")


def expected_state(verdict):
    """The three states the render must tell apart, read off the record.

    `beaten_by_baseline` is not a louder `earned: false`: it says the held-out part put a
    fitted baseline *above* the network, which is the one reading a user must not be spared.
    """
    if verdict.get("earned") is True:
        return "earned"
    return "beaten" if verdict.get("beaten_by_baseline") is True else "not-earned"


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
        print("ERROR: no parseable layer-bundle-data script in the file")
        return 1
    chrome = find_chrome()
    if not chrome:
        msg = f"no headless browser available (checked {CHROME_CANDIDATES})"
        if a.strict:
            print(f"ERROR: {msg} — failing under --strict")
            return 1
        print(f"WARN: {msg}; the render is unverified")
        return 0

    man = doc.get("manifest") or {}
    chain = man.get("chain") or {}
    ev = chain.get("evaluation") or {}
    verdict = ev.get("verdict") or {}
    primary = ev.get("primary") or {}
    stances = man.get("stances") or []
    schema = doc.get("input_schema") or {}
    inputs = schema.get("input") or []
    rt = doc.get("roundtrip") or man.get("roundtrip") or {}
    tol = rt.get("tolerance") or {}
    examples = man.get("examples") or {}
    exports = doc.get("exports") or man.get("exports") or []
    serve = doc.get("serve") or man.get("serve") or {}
    findings = doc.get("code_findings") or []
    card = doc.get("model_card") or {}
    notebook = doc.get("notebook") or {}
    usage = doc.get("usage") or man.get("usage") or {}
    markers = doc.get("markers") or []
    want_state = expected_state(verdict)
    # a column claims a domain when the record gives it one; those are the columns the page
    # must not render bare
    with_domain = [c for c in inputs
                   if c.get("allowed") is not None or c.get("range_seen") is not None]
    warn_markers = [m for m in markers if str(m).startswith("WARN")]
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

        print(f"bundle layer smoke — {path.name}")
        check(C.get("tab_button_in_nav"), "1. the tab button is in the host's nav")
        check(C.get("tab_label") == "Bundle", "1. the tab is labelled Bundle",
              f"got {C.get('tab_label')!r}")
        check(C.get("pane_mounted"), "2. the pane is mounted")
        check(C.get("panes_with_layer") == 1, "2. exactly one pane (the mount is idempotent)",
              f"got {C.get('panes_with_layer')}")
        check(C.get("pane_inactive_by_default"),
              "2. the pane does not steal the active tab on load")

        views = C.get("views") or {}
        for v in VIEWS:
            check(views.get(v), f"3. the {v} surface is present")
        check(C.get("first_view") == "verdict",
              "3. the verdict is the first surface on the page, not a badge below the fold",
              f"the first surface is {C.get('first_view')!r}")

        vtext = C.get("verdict_text") or ""
        check(C.get("verdict_state") == want_state,
              "4. the rendered verdict agrees with the recorded one",
              f"page says {C.get('verdict_state')!r}, the record says {want_state!r} "
              f"(earned={verdict.get('earned')}, "
              f"beaten_by_baseline={verdict.get('beaten_by_baseline')})")
        flag = (C.get("flag_text") or "").lower()
        check(("beaten" in flag) == (want_state == "beaten"),
              "4. the beaten-by-a-baseline case is said in those words, not as a plain "
              "'not earned'",
              f"the flag reads {C.get('flag_text')!r} for state {want_state!r}")
        if want_state == "beaten":
            check("is better than the network" in vtext or "is better" in vtext,
                  "4. the verdict states which model the held-out part put on top",
                  "the page does not say the baseline is better")
            check("undecided comparison" in vtext or "not an undecided" in vtext,
                  "4. the verdict separates 'a baseline won' from 'the comparison was "
                  "undecided'")
        check(str(verdict.get("strongest_baseline") or "") in vtext,
              "4. the verdict names the strongest fitted baseline")
        check(str(verdict.get("diff")) in vtext,
              "4. the verdict states the paired difference",
              f"{verdict.get('diff')} not in the verdict text")
        check(str(primary.get("value")) in vtext,
              "4. the verdict states the primary metric it is a verdict on",
              f"{primary.get('value')} not in the verdict text")
        check(C.get("stance_in_verdict") and
              len(C.get("stances") or []) == len(stances),
              "4. every recorded stance is rendered inside the verdict box",
              f"{len(stances)} recorded, {len(C.get('stances') or [])} rendered, "
              f"in-box={C.get('stance_in_verdict')}")
        check(all(any(str(s) in r for r in (C.get("stances") or [])) for s in stances),
              "4. the stances are the recorded ones, verbatim")
        check(str(ev.get("seal_exposures")) in (C.get("exposures_text") or ""),
              "4. the exposure count qualifying these numbers is stated",
              f"exposures={ev.get('seal_exposures')} not in "
              f"{(C.get('exposures_text') or '')[:80]!r}")

        rendered_cols = [c for c in (C.get("schema_columns") or []) if c]
        check(rendered_cols == [str(c.get("name")) for c in inputs],
              "5. every input column in the record appears in the table, in order",
              f"record {[c.get('name') for c in inputs]}, page {rendered_cols}")
        check(C.get("columns_without_domain") == [],
              "5. every column the record gives a domain for renders that domain",
              f"{C.get('columns_without_domain')} rendered bare — a column without the values "
              f"or the range seen in training invites a caller to extrapolate past bounds "
              f"nothing was measured at")
        check(C.get("schema_domains", 0) == len(with_domain),
              "5. a domain is drawn for every column that has one, and only those",
              f"{len(with_domain)} column(s) with a recorded domain, "
              f"{C.get('schema_domains')} drawn")
        check(sorted(C.get("output_fields") or []) == sorted((schema.get("output") or {}).keys()),
              "5. every output field is described")
        reading_checks(check, C, man)
        check(all(str(i) in (C.get("ignored_text") or "") for i in schema.get("ignored") or []),
              "5. the ignored columns are named, so a caller knows what not to send")
        check(C.get("sortable_headers", 0) >= 4 and
              C.get("headers_focusable", 0) == C.get("sortable_headers", 0),
              "5. every sortable header is focusable",
              f"{C.get('sortable_headers')} sortable, {C.get('headers_focusable')} focusable")
        if len(inputs) > 1:
            check(C.get("order_after_click") != C.get("order_initial")
                  or C.get("aria_after_click") in ("ascending", "descending"),
                  "5. a click sorts the schema table")
            check(C.get("aria_after_key") != C.get("aria_after_click"),
                  "5. Enter on a header sorts it too (the table is not mouse-only)",
                  f"aria stayed {C.get('aria_after_key')!r}")

        check((C.get("rt") or {}).get("status") == rt.get("status"),
              "6. the round-trip status shown is the recorded one",
              f"page {(C.get('rt') or {}).get('status')!r}, record {rt.get('status')!r}")
        check(str(rt.get("rows")) in ((C.get("rt") or {}).get("text") or ""),
              "6. the round trip states how many rows it reproduced",
              f"{rt.get('rows')} not in the line")
        check(C.get("rule_text") == str(tol.get("rule")),
              "6. the tolerance is rendered as its formula, not only as a number",
              f"page {C.get('rule_text')!r}, record {tol.get('rule')!r}")
        check(tol.get("rule") == CATALOG["roundtrip"]["rule"],
              "6. and that formula is the catalog's, not a second copy of it",
              f"record {tol.get('rule')!r}, catalog {CATALOG['roundtrip']['rule']!r}")
        check(str(rt.get("max_abs_diff")) in ((C.get("tol") or {}).get("text") or "")
              or str(int(rt.get("max_abs_diff", 0)))
              in ((C.get("tol") or {}).get("text") or ""),
              "6. the worst difference actually seen is shown beside the bound")
        check((C.get("shaper") or {}).get("status") == rt.get("shaper_transform_check"),
              "6. the shaper's transform check is its own rendered result",
              f"page {(C.get('shaper') or {}).get('status')!r}, "
              f"record {rt.get('shaper_transform_check')!r}")

        etext = C.get("examples_text") or ""
        check(C.get("examples_kind") == str(examples.get("kind")),
              "7. examples.kind is visible on the page",
              f"page {C.get('examples_kind')!r}, record {examples.get('kind')!r}")
        want_proves = {"test": "the evaluation", "synthetic": "itself"}.get(
            str(examples.get("kind")), "unrecorded")
        check(C.get("examples_proves") == want_proves
              and ("proves the bundle reproduces the evaluation" in etext)
              == (examples.get("kind") == "test"),
              "7. a test-row bundle says it reproduces the evaluation and a synthetic one does "
              "not — they are different guarantees and are not rendered alike",
              f"kind={examples.get('kind')!r} but the page says it proves "
              f"{C.get('examples_proves')!r}")

        check([f for f in (C.get("export_formats") or []) if f]
              == [str(e.get("format")) for e in exports],
              "8. every recorded export is listed",
              f"record {[e.get('format') for e in exports]}, "
              f"page {C.get('export_formats')}")
        bad = len([e for e in exports if e.get("status") != "pass"])
        check(C.get("export_marked", 0) == bad,
              "8. a failing parity ships marked rather than quietly",
              f"{bad} export(s) did not pass, {C.get('export_marked')} marked")
        check(C.get("serve_mode") == str(serve.get("mode")),
              "8. the serving record states its mode",
              f"page {C.get('serve_mode')!r}, record {serve.get('mode')!r}")

        ftext = C.get("findings_text") or ""
        check([f for f in (C.get("finding_ids") or []) if f]
              == [str(f.get("id")) for f in findings],
              "9. every code finding is listed")
        check([s for s in (C.get("finding_severities") or []) if s]
              == [str(f.get("severity") or f.get("depth")) for f in findings],
              "9. each finding carries its severity")
        check(all(str(f.get("title")) in ftext for f in findings),
              "9. each finding says what it found, not only that it exists")
        judged = sorted(t for t, r in (doc.get("conformance") or {}).items() if isinstance(r, dict))
        check(sorted(j for j in (C.get("judgements") or []) if j) == judged,
              "9. each tool's judgement — its audits, confirmed and unconfirmed — is stated",
              f"page {C.get('judgements')}, record {judged}")
        items = [str(i.get("finding")) for r in (doc.get("conformance") or {}).values()
                 if isinstance(r, dict) for i in (r.get("advisory") or {}).get("open_items") or []]
        check([i for i in (C.get("open_items") or []) if i] == items,
              "9. each confirmed (advisory) finding is shown as its open item",
              f"page {C.get('open_items')}, record {items}")
        check(C.get("usage_cli") == usage.get("cli"),
              "9. the CLI snippet is the manifest's, verbatim",
              f"page {C.get('usage_cli')!r}")
        check(C.get("usage_module") == usage.get("module"),
              "9. the module snippet is the manifest's, verbatim")
        check(C.get("card_is_link") and C.get("card_href") == card.get("path"),
              "9. the model card is linked as the sibling file it is, not inlined",
              f"href {C.get('card_href')!r}, record {card.get('path')!r}")
        check(C.get("notebook_path") == notebook.get("path")
              and ("executed" in (C.get("notebook_text") or "")),
              "9. the notebook is named with whether it was executed")
        intended = str((doc.get("readings") or {}).get("intended_use") or "")
        check(bool(intended) and intended[:60] in (C.get("readings_text") or ""),
              "9. the intended-use prose is rendered as written",
              "the recorded prose is not on the page")

        check([m for m in (C.get("markers") or [])] == [str(m) for m in markers],
              "10. every marker pack.py wrote is listed",
              f"{len(markers)} recorded, {len(C.get('markers') or [])} rendered")
        check(C.get("markers_warned", 0) == len(warn_markers),
              "10. a WARN marker is marked as one",
              f"{len(warn_markers)} WARN in the record, {C.get('markers_warned')} marked")

        check(C.get("api"), "11. window.__bundle is exposed")
        check(C.get("api_run") == man.get("run") and C.get("api_slug") == man.get("slug"),
              "11. manifest() returns the record the page was given",
              f"got run={C.get('api_run')!r} slug={C.get('api_slug')!r}")
        check(bool(C.get("api_version")), "11. the render states its version",
              f"got {C.get('api_version')!r}")

        check(C.get("imgs", 0) == 0 and C.get("canvases", 0) == 0,
              "12. no <img> and no <canvas> — inline SVG only, and nothing fetched")
        check(C.get("ci_bars", 0) >= (1 if verdict.get("ci") else 0),
              "12. the paired difference is drawn against zero as inline SVG")
        js = RENDER_JS.read_bytes() if RENDER_JS.is_file() else b""
        check(0 < len(js) <= JS_LIMIT,
              f"12. the shipped render is within the {JS_LIMIT // 1024} KB the contract allows",
              f"{RENDER_JS} is {len(js)} bytes")
        check(not R.get("errors"), "12. no uncaught JS error",
              "; ".join(R.get("errors") or [])[:300])

        degraded = probe_copy(html, tmp, "degraded.html", degrade=True)
        dom2, _ = dump_dom(chrome, degraded)
        R2 = parse_results(dom2) or {}
        C2 = R2.get("checks") or {}
        check(C2.get("pane_mounted"),
              "13. the pane still mounts with no findings and no exports recorded")
        check([c for c in (C2.get("schema_columns") or []) if c] == rendered_cols,
              "13. the schema table is unaffected by the missing records")
        d2 = C2.get("examples_text") or ""
        check(C2.get("examples_kind") == "synthetic"
              and C2.get("examples_proves") == "itself"
              and "proves only that the bundle reproduces itself" in d2
              and "proves the bundle reproduces the evaluation" not in d2,
              "13. a synthetic bundle says its shipped test proves only that it reproduces "
              "itself",
              f"kind={C2.get('examples_kind')!r}")
        check(not R2.get("errors"), "13. no JS error on the degraded copy",
              "; ".join(R2.get("errors") or [])[:300])

    print()
    if failures:
        print(f"{len(failures)} of {checks} checks failed: {failures}")
        return 1
    print(f"OK: bundle layer smoke test passed for {path.name} ({checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
