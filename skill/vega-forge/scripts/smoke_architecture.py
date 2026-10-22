#!/usr/bin/env python3
"""
smoke_architecture.py — headless-Chromium check that the /vega-forge `architecture` layer renders.

    python3 smoke_architecture.py OUT.html [--strict]

A copy of OUT.html is opened in headless Chromium with a probe appended that exercises the
page and writes its findings into a <pre id="architecture-smoke">; the DOM dump is then parsed.

  1. the tab button (data-tab="architecture") sits in the host's nav.tabs / nav[role=tablist]
     and is labelled Network;
  2. the pane section.tab-pane[data-layer="architecture"] is mounted and inactive by default;
  3. the nine surfaces exist (data-view = blueprint-map | conformance | blocks | sample |
     decisions | mechanics | smoke | forks | defaults);
  4. the decisions table holds one row per traced choice in the spec, and every row shows a
     source chip — the provenance is the one thing this layer may never render blank;
  5. the block diagram draws one node per top-level block, and one group per composite block
     that repeats inner blocks;
  6. the sample panel (E20) sits beside the block diagram with the spec's sample shape, and
     shows exactly the smoke pass's `sample` record: a window's length, stride, horizon (a
     forecast only), series, the windows of train and valid, the valid rows left unscored
     (captioned with the reasons the declared context and horizon allow, never one it cannot
     know), the context and the autoregressive flag; a graph's level, protocol and batching,
     the training and valid views' nodes, edges and targets and the test edges withheld, with
     the exposure chip iff the protocol is transductive; a row's training rows. The smoke panel
     names the unit the mult-adds are counted per (`madds_unit`). What the panel must show is
     `architecture_sample.py`'s, the rule `apply_architecture_layer.py` also holds;
  7. the conformance panel states the mechanical verdict and, per judgement tool, its audits
     and what they confirmed and left unconfirmed (or the tool's absence, as a warning), and
     each confirmed — advisory — finding as its open item for /architect and /improve;
  8. the mechanics panel writes the loss formula iff the smoke record or the spec carries one,
     and the update rule iff the spec's optimizer carries one (else it says which is missing);
     the smoke panel draws one `× scale + shift` per output channel of a regression-family
     head's target affine iff the smoke record's `init_notes.target_affine` carries it (A37);
  9. window.__architecture exposes spec(), blocks(), conformance(), smoke() and sample(), and
     they return what the page was given, not a rebuilt copy;
 10. no uncaught JS error; the render script in the file is within the contract's 60 KB;
 11. on a copy whose smoke record lost its measurements (per_block emptied, `sample` removed),
     the pane still renders the diagram and the sample panel from the spec, both say what is
     missing, and the sample panel shows no count.

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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import architecture_sample as samples  # noqa: E402

CHROME_CANDIDATES = ("chromium", "google-chrome", "chromium-browser", "chrome")
DATA_RE = re.compile(r'(<script\s+id="layer-architecture-data"[^>]*>)([\s\S]*?)(</script>)', re.I)
RENDER_RE = re.compile(r'<script\s+id="layer-architecture-render"[^>]*>([\s\S]*?)</script>', re.I)
MAX_RENDER = 60 * 1024
VIEWS = ("blueprint-map", "conformance", "blocks", "sample", "decisions", "mechanics", "smoke",
         "forks", "defaults")
COMPOSITE = ("residual", "sequential", "parallel")
# the browser's own complaints, not the page's: no GPU, no D-Bus, no sandbox, paint-timing under
# load ("Invalid first_paint ..."), google-chrome's GCM push registration ("google_apis/gcm/")
CHROME_NOISE = ("gpu", "dbus", "sandbox", "page_load_metrics", "google_apis/gcm/")
# the four families in the order the contract fixes them; an unknown family is appended
FAM_ORDER = ("architecture", "objective", "optimization", "regularization")

PROBE = r"""
<script id="architecture-smoke-probe">
(function(){
  var R = { errors: [], checks: {} };
  window.addEventListener('error', function(e){ R.errors.push('onerror: ' + (e.message || e)); });
  var origErr = console.error;
  console.error = function(){ R.errors.push('console.error: ' + Array.prototype.slice.call(arguments).map(String).join(' ')); if (origErr) origErr.apply(console, arguments); };
  function q(s){ return document.querySelector(s); }
  function run(){
    try {
      var nav = q('nav.tabs') || q('nav[role="tablist"]');
      var btn = nav ? nav.querySelector('button[data-tab="architecture"]') : null;
      R.checks.tab_button_in_nav = !!btn;
      R.checks.tab_label = btn ? (btn.textContent || '').trim() : '';
      var pane = q('section.tab-pane[data-layer="architecture"]');
      R.checks.pane_mounted = !!pane;
      R.checks.pane_inactive_by_default = !!pane && !pane.classList.contains('active');
      R.checks.views = {};
      ['blueprint-map','conformance','blocks','sample','decisions','mechanics','smoke','forks','defaults']
        .forEach(function(v){ R.checks.views[v] = !!(pane && pane.querySelector('[data-view="' + v + '"]')); });
      R.checks.decision_rows = pane ? pane.querySelectorAll('[data-view="decisions"] tbody tr.a-drow').length : 0;
      R.checks.decision_sources = 0;
      if (pane) Array.prototype.forEach.call(
        pane.querySelectorAll('[data-view="decisions"] tbody tr.a-drow'), function(r){
          var c = r.querySelector('.a-src-geometry, .a-src-analysis, .a-src-shape, '
                                  + '.a-src-blueprint, .a-src-user, .a-src-default');
          if (c && c.textContent && c.textContent.trim() && c.textContent.indexOf('unsourced') < 0)
            R.checks.decision_sources++;
        });
      R.checks.families = pane ? pane.querySelectorAll('[data-view="decisions"] .a-fam').length : 0;
      R.checks.block_nodes = pane ? pane.querySelectorAll('[data-view="blocks"] .a-node').length : 0;
      R.checks.block_groups = pane ? pane.querySelectorAll('[data-view="blocks"] .a-diagram .a-ncomp').length : 0;
      R.checks.blocks_empty = !!(pane && pane.querySelector('[data-view="blocks"] > .a-empty, '
                                                        + '[data-view="blocks"] .a-diagram .a-empty'));
      R.checks.blocks_text = pane && pane.querySelector('[data-view="blocks"]')
        ? (pane.querySelector('[data-view="blocks"]').textContent || '') : '';
      R.checks.conformance_text = pane && pane.querySelector('[data-view="conformance"]')
        ? (pane.querySelector('[data-view="conformance"]').textContent || '') : '';
      R.checks.open_items = pane ? pane.querySelectorAll('[data-view="conformance"] [data-open-item]').length : 0;
      R.checks.map_rows = pane ? pane.querySelectorAll('[data-view="blueprint-map"] tr.a-brow').length : 0;
      R.checks.map_pending = pane ? pane.querySelectorAll('[data-view="blueprint-map"] tr.a-pending').length : 0;
      R.checks.chart_lines = pane ? pane.querySelectorAll('[data-view="mechanics"] polyline.a-line').length : 0;
      R.checks.formulas = pane ? pane.querySelectorAll('[data-view="mechanics"] .a-formula').length : 0;
      R.checks.smoke_stats = pane ? pane.querySelectorAll('[data-view="smoke"] .a-stat').length : 0;
      R.checks.target_affine = pane ? pane.querySelectorAll('[data-view="smoke"] [data-init="target-affine"] .a-affine').length : 0;
      R.checks.fork_rows = pane ? pane.querySelectorAll('[data-view="forks"] tr.a-fork').length : 0;
      R.checks.fork_unattended = pane ? pane.querySelectorAll('[data-view="forks"] tr.a-unattended').length : 0;
      R.checks.forks_text = pane && pane.querySelector('[data-view="forks"]')
        ? (pane.querySelector('[data-view="forks"]').textContent || '') : '';
      R.checks.default_rows = pane ? pane.querySelectorAll('[data-view="defaults"] tr.a-default').length : 0;
      var sp = pane ? pane.querySelector('[data-view="blocks"] [data-view="sample"]') : null;
      var dg = pane ? pane.querySelector('[data-view="blocks"] .a-diagram') : null;
      R.checks.sample_mounted = !!sp;
      R.checks.sample_shape = sp ? sp.getAttribute('data-shape') : null;
      R.checks.sample_beside = !!(sp && dg && sp.parentNode === dg.parentNode
                                  && (dg.querySelector('svg.a-blocks') || dg.querySelector('.a-empty')));
      R.checks.sample_values = {}; R.checks.sample_notes = {};
      if (sp) Array.prototype.forEach.call(sp.querySelectorAll('[data-k]'), function(n){
        var v = n.querySelector('.a-statv'), t = n.querySelector('.a-statn');
        R.checks.sample_values[n.getAttribute('data-k')] = ((v || n).textContent || '').trim();
        if (t) R.checks.sample_notes[n.getAttribute('data-k')] = (t.textContent || '').trim();
      });
      R.checks.sample_exposure = sp ? sp.querySelectorAll('.a-exposure').length : 0;
      R.checks.sample_missing = !!(sp && sp.querySelector('.a-sample-missing'));
      R.checks.sample_strip = sp ? sp.querySelectorAll('svg.a-wstrip').length : 0;
      R.checks.sample_text = sp ? (sp.textContent || '') : '';
      R.checks.mechanics_text = pane && pane.querySelector('[data-view="mechanics"]')
        ? (pane.querySelector('[data-view="mechanics"]').textContent || '') : '';
      R.checks.smoke_text = pane && pane.querySelector('[data-view="smoke"]')
        ? (pane.querySelector('[data-view="smoke"]').textContent || '') : '';
      R.checks.hook = !!(window.__architecture
                         && typeof window.__architecture.spec === 'function'
                         && typeof window.__architecture.blocks === 'function'
                         && typeof window.__architecture.conformance === 'function'
                         && typeof window.__architecture.smoke === 'function');
      if (R.checks.hook) {
        var hs = window.__architecture.spec() || {};
        R.checks.hook_version = window.__architecture.version;
        R.checks.hook_spec_blocks = (hs.blocks || []).length;
        R.checks.hook_blocks = window.__architecture.blocks().length;
        R.checks.hook_mechanical = (window.__architecture.conformance() || {}).mechanical;
        R.checks.hook_params = (window.__architecture.smoke() || {}).params_total;
        R.checks.hook_sample = typeof window.__architecture.sample === 'function'
          ? (window.__architecture.sample() || null) : 'absent';
      }
      var det = pane ? pane.querySelector('[data-view="decisions"] details') : null;
      if (det) { det.open = true; R.checks.details_open = det.open; }
      if (btn) { btn.click(); R.checks.tab_activates = !!(pane && pane.classList.contains('active')); }
    } catch (e) { R.errors.push('probe: ' + (e && e.stack || e)); }
    var pre = document.createElement('pre'); pre.id = 'architecture-smoke';
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


def traced_choices(spec):
    """The choices the decisions table must show, counted the way the render enumerates them:
    each decision family's keys in order, a list-valued key contributing one choice per member."""
    fams = spec.get("decision_families") or {}
    order = [f for f in FAM_ORDER if f in fams] + [f for f in fams if f not in FAM_ORDER]
    out = []
    for fam in order:
        for key in fams.get(fam) or []:
            v = spec.get(key)
            if v is None:
                continue
            items = v if isinstance(v, list) else [v]
            out.extend((fam, key, o) for o in items if isinstance(o, dict))
    return out


def render_bytes(html):
    """The size of the render script the layered file carries (None: no render script)."""
    m = RENDER_RE.search(html)
    return len(m.group(1).strip().encode("utf-8")) if m else None


def composite_groups(spec):
    """The top-level blocks the diagram draws as a group: a composite that repeats inner blocks."""
    return sum(1 for b in spec.get("blocks") or [] if isinstance(b, dict)
               and b.get("type") in COMPOSITE and ((b.get("params") or {}).get("inner") or []))


def probe_copy(html, tmpdir, name, no_per_block=False):
    """A copy of the layered file with the probe appended; `no_per_block` also takes the smoke
    record's measurements away (per_block emptied, the `sample` block removed)."""
    if no_per_block:
        doc, m = layer_data(html)
        if doc is not None:
            doc.setdefault("smoke", {})["per_block"] = []
            doc["smoke"].pop("sample", None)
            html = (html[:m.start(2)] + "\n" +
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n" +
                    html[m.end(2):])
    idx = html.rfind("</body>")
    html = html[:idx] + PROBE + html[idx:] if idx >= 0 else html + PROBE
    p = Path(tmpdir) / name
    p.write_text(html, encoding="utf-8")
    return p


def parse_results(dom):
    m = re.search(r'<pre id="architecture-smoke">([\s\S]*?)</pre>', dom)
    if not m:
        return None
    txt = (m.group(1).replace("&quot;", '"').replace("&lt;", "<")
           .replace("&gt;", ">").replace("&amp;", "&"))
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


class Checker:
    """The check callable: prints each check, counts them, remembers the failed ones by name."""

    def __init__(self):
        self.count, self.failures = 0, []

    def __call__(self, cond, name, detail=""):
        self.count += 1
        if cond:
            print(f"  OK   {name}")
        else:
            self.failures.append(name)
            print(f"  FAIL {name}{': ' + detail if detail else ''}")


@dataclass(frozen=True)
class Expected:
    """What the layer's data says the pane must show, surface by surface."""
    spec: dict
    smoke: dict
    conf: dict
    choices: int
    blocks: int
    map_rows: int
    pending: int
    forks: int
    unattended: int
    defaults: int
    groups: int
    has_loss: bool      # the smoke record's loss formula, else the spec's
    has_rule: bool      # the spec's optimizer carries an update rule

    @classmethod
    def of(cls, doc):
        spec, smoke = doc.get("netspec") or {}, doc.get("smoke") or {}
        entries = (doc.get("blueprint_map") or {}).get("entries") or []
        forks = doc.get("forks") or []
        return cls(spec=spec, smoke=smoke, conf=doc.get("conformance") or {},
                   choices=len(traced_choices(spec)), blocks=len(spec.get("blocks") or []),
                   map_rows=len(entries),
                   pending=len([e for e in entries
                                if str(e.get("status") or "").startswith("pending")]),
                   forks=len(forks), unattended=len([f for f in forks if f.get("answer") is None]),
                   defaults=len(doc.get("defaults_to_test") or []), groups=composite_groups(spec),
                   has_loss=bool(smoke.get("loss_formula")
                                 or (spec.get("loss") or {}).get("formula")),
                   has_rule=bool((spec.get("optimizer") or {}).get("update_rule")))


def surface_checks(check, label, c, want):
    """The tab, the pane, its nine surfaces, the decisions, the blocks, the blueprint map and
    the conformance verdict."""
    check(c.get("tab_button_in_nav"), f"{label}:tab-button")
    check(c.get("tab_label") == "Network", f"{label}:tab-labelled-Network", str(c.get("tab_label")))
    check(c.get("pane_mounted"), f"{label}:pane")
    check(c.get("pane_inactive_by_default"), f"{label}:pane-inactive-by-default")
    for v in VIEWS:
        check((c.get("views") or {}).get(v), f"{label}:view-{v}")
    check(c.get("decision_rows") == want.choices, f"{label}:decision-rows",
          f"{c.get('decision_rows')} != {want.choices}")
    check(c.get("decision_sources") == want.choices, f"{label}:every-choice-shows-its-source",
          f"{c.get('decision_sources')} of {want.choices}")
    check(c.get("families", 0) >= 4, f"{label}:families-grouped", str(c.get("families")))
    check(c.get("block_nodes") == want.blocks, f"{label}:one-node-per-block",
          f"{c.get('block_nodes')} != {want.blocks}")
    check(c.get("map_rows") == want.map_rows, f"{label}:blueprint-map-rows",
          f"{c.get('map_rows')} != {want.map_rows}")
    check(c.get("map_pending") == want.pending, f"{label}:pending-ports-marked",
          f"{c.get('map_pending')} != {want.pending}")
    ct = c.get("conformance_text") or ""
    check("mechanical" in ct and str(want.conf.get("mechanical", "")) in ct,
          f"{label}:mechanical-verdict-shown")
    arch = want.conf.get("architect")
    if isinstance(arch, dict):
        check(all(str(arch.get(k)) in ct for k in ("structural", "divergence", "recorded")),
              f"{label}:architect-counts-shown")
        check(_judgement_shown("Architect", arch, ct), f"{label}:architect-audits-shown")
    else:
        # a gate that did not run must read as a warning, not as a pass
        check("WARNING" in ct or "not installed" in ct, f"{label}:architect-absence-warned")
    imp = want.conf.get("improve")
    if isinstance(imp, dict):
        check(_judgement_shown("Improve", imp, ct), f"{label}:improve-counts-shown")
    else:
        check("the improve skill is" in ct, f"{label}:improve-absence-warned")
    # a confirmed finding is advisory (code-contract §5): each is shown as its open item
    items = [i for r in (arch, imp) if isinstance(r, dict)
             for i in (r.get("advisory") or {}).get("open_items") or []]
    check(c.get("open_items") == len(items), f"{label}:open-items-shown",
          f"{c.get('open_items')} != {len(items)}")


def mechanics_checks(check, label, c, want):
    """The loss formula and the update rule, each written iff carried, else said missing; the
    schedule's curve; the smoke panel and the head's target affine iff recorded (A37)."""
    formulas = int(want.has_loss) + int(want.has_rule)
    check(c.get("formulas") == formulas, f"{label}:loss-formula-and-update-rule",
          f"{c.get('formulas')} != {formulas}")
    text = c.get("mechanics_text") or ""
    if not want.has_loss:
        check("carries the loss formula" in text, f"{label}:missing-loss-formula-said")
    if not want.has_rule:
        check("carries no update rule" in text, f"{label}:missing-update-rule-said")
    check(c.get("chart_lines", 0) >= 1, f"{label}:schedule-curve")
    check(c.get("smoke_stats", 0) >= 6, f"{label}:smoke-panel", str(c.get("smoke_stats")))
    affine = len(((want.smoke.get("init_notes") or {}).get("target_affine") or {}).get("shift") or [])
    check(c.get("target_affine") == affine, f"{label}:target-affine-shown-iff-recorded",  # A37
          f"{c.get('target_affine')} channel(s) drawn, {affine} recorded")


def fork_checks(check, label, c, want):
    check(c.get("fork_rows") == want.forks, f"{label}:fork-rows",
          f"{c.get('fork_rows')} != {want.forks}")
    check(c.get("fork_unattended") == want.unattended, f"{label}:unattended-forks-marked",
          f"{c.get('fork_unattended')} != {want.unattended}")
    if want.unattended:
        check("default applied (unattended)" in (c.get("forks_text") or ""),
              f"{label}:unattended-fork-says-so")
    check(c.get("default_rows") == want.defaults, f"{label}:defaults-to-test-rows",
          f"{c.get('default_rows')} != {want.defaults}")


def hook_checks(check, label, c, want):
    check(c.get("hook"), f"{label}:window.__architecture")
    check(c.get("hook_version") == "vega-forge/architecture-render@1", f"{label}:hook-version",
          str(c.get("hook_version")))
    check(c.get("hook_spec_blocks") == want.blocks, f"{label}:hook-spec-blocks",
          f"{c.get('hook_spec_blocks')} != {want.blocks}")
    check(c.get("hook_blocks") == want.blocks, f"{label}:hook-blocks",
          f"{c.get('hook_blocks')} != {want.blocks}")
    check(c.get("hook_mechanical") == want.conf.get("mechanical"), f"{label}:hook-conformance",
          str(c.get("hook_mechanical")))
    check(c.get("hook_params") == want.smoke.get("params_total"), f"{label}:hook-smoke",
          str(c.get("hook_params")))


def diagram_checks(check, c, want, case):
    """The full pass groups the composite blocks and shows the per-block numbers; the degraded
    one says they are missing."""
    label = case.label
    if not case.degraded:
        check(c.get("block_groups") == want.groups, f"{label}:composite-blocks-grouped",
              f"{c.get('block_groups')} != {want.groups}")
        bt = c.get("blocks_text") or ""
        check("params" in bt and "out [" in bt and not c.get("blocks_empty"),
              f"{label}:per-block-numbers-shown")
    else:
        check(c.get("blocks_empty"), f"{label}:per-block-empty-state")
        check("per-block numbers" in (c.get("blocks_text") or ""), f"{label}:degraded-pane-says-so")


def browser_checks(check, label, R, err):
    c = R["checks"]
    check(c.get("details_open"), f"{label}:alternatives-expand")
    check(c.get("tab_activates"), f"{label}:tab-button-activates-pane")
    check(not R["errors"], f"{label}:no-js-errors", "; ".join(R["errors"])[:300])
    if err and "ERROR:" in err:
        bad = [ln for ln in err.splitlines() if "ERROR:" in ln
               and not any(noise in ln.lower() for noise in CHROME_NOISE)]
        check(not bad, f"{label}:chromium-stderr", "; ".join(bad)[:200])


def run_pass(check, chrome, page, want, case):
    """One headless render of `page` (a probed copy) and every check on what it drew."""
    try:
        dom, err = dump_dom(chrome, page)
    except subprocess.TimeoutExpired:
        check(False, f"{case.label}:render", "chromium timed out")
        return
    R = parse_results(dom)
    if R is None:
        check(False, f"{case.label}:probe", "no probe output captured")
        return
    c = R["checks"]
    surface_checks(check, case.label, c, want)
    mechanics_checks(check, case.label, c, want)
    fork_checks(check, case.label, c, want)
    hook_checks(check, case.label, c, want)
    samples.panel_checks(check, c, case)
    diagram_checks(check, c, want, case)
    browser_checks(check, case.label, R, err)


def run_passes(check, chrome, html, want):
    """The full pass, then the pass on a copy whose smoke record lost its measurements."""
    with tempfile.TemporaryDirectory() as td:
        for label, degraded in (("full", False), ("no-measurements", True)):
            page = probe_copy(html, td, f"smoke-{label}.html", no_per_block=degraded)
            run_pass(check, chrome, page, want,
                     samples.Case(label, want.spec, want.smoke, degraded))



def _judgement_shown(name, record, text) -> bool:
    """One tool's judgement as the panel states it: its audits from the first run, what two of
    three confirmed and what one audit alone reported (code-contract §5)."""
    audits = len(record.get("audits") or [])
    left = record.get("unconfirmed") or {}
    return (f"{name}: {audits} audit" in text and f"from {record.get('run') or '·'}" in text
            and f"unconfirmed (recorded, not blocking) {left.get('structural')} structural" in text)

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
        print("ERROR: no parseable layer-architecture-data script in the file")
        return 1
    chrome = find_chrome()
    if not chrome:
        msg = f"no headless browser available (checked {CHROME_CANDIDATES})"
        if a.strict:
            print(f"ERROR: {msg} — failing under --strict")
            return 1
        print(f"WARN: {msg}; the render is unverified")
        return 0

    check = Checker()
    size = render_bytes(html)
    check(size is not None and size <= MAX_RENDER, "render-within-60-KB",
          f"{size} bytes" if size is not None else "no layer-architecture-render script")
    run_passes(check, chrome, html, Expected.of(doc))
    if check.failures:
        print(f"ERROR: architecture layer smoke test failed for {path} "
              f"({len(check.failures)} of {check.count} checks)")
        return 1
    print(f"OK: architecture layer smoke test passed for {path} ({check.count} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
