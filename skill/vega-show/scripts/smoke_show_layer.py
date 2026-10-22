"""
smoke_show_layer — the `showcase` tab of a model page, opened in a headless Chrome (`cdp.py`) and
held to D5 §9.2's nine checks (`smoke_show.py OUT.html`):

  1. the tab button (data-tab="showcase") sits in the host's nav.tabs / nav[role=tablist],
     labelled Showcase;
  2. the pane section.tab-pane[data-layer="showcase"] is mounted once and inactive by default;
  3. the six surfaces are there, in the catalog's order (layer.surfaces), and nothing else is;
  4. every binding of the data is a row of the bindings table, and the filter narrows it;
  5. the report status shown is SHOWCASE.json's, and brief-links is rendered as a warning;
  6. the served walk's status is rendered, and a walk that did not run shows its reason;
  7. no uncaught error, no console error, no <img> or <canvas> in the pane, and
     window.__showcase exposes story(), bindings() and the render's version;
  8. the shipped render is at most layer.max_render_kb;
  9. a degraded copy (brief links, the served walk not run, no bindings) still mounts and says so.

    check_layer(chrome, out_html) -> [{id, ok, problems}]
"""
from __future__ import annotations

import html as _html
import json
import re
import tempfile
import time
from pathlib import Path

import cdp
from vega_show_core import CATALOG

__all__ = ["check_layer", "layer_parts", "degraded_copy"]

LAYER = CATALOG["layer"]
SURFACES = LAYER["surfaces"]
VERSION = "vega-show/showcase-render@1"
DATA_RE = re.compile(r'(<script\s+id="layer-showcase-data"[^>]*>)([\s\S]*?)(</script>)', re.I)
RENDER_RE = re.compile(r'<script\s+id="layer-showcase-render"[^>]*>([\s\S]*?)</script>', re.I)
NOT_RUN_WHY = "degraded copy: the served walk did not run"

PROBE = r"""(() => {
  const R = {};
  const pane = document.querySelectorAll('section.tab-pane[data-layer="showcase"]');
  const nav = document.querySelector('nav.tabs') || document.querySelector('nav[role="tablist"]');
  const btn = nav ? nav.querySelector('button[data-tab="showcase"]') : null;
  const p = pane[0] || null;
  R.button = !!btn; R.label = btn ? (btn.textContent || '').trim() : null;
  R.panes = pane.length; R.active = p ? p.classList.contains('active') : null;
  R.surfaces = p ? Array.from(p.querySelectorAll('[data-surface]')).map(n => n.getAttribute('data-surface')) : [];
  R.rows = p ? Array.from(p.querySelectorAll('table.sc-bindings tbody tr')).map(tr => tr.getAttribute('data-key')) : [];
  const rep = p ? p.querySelector('[data-surface="report"]') : null;
  R.report_status = rep ? (rep.querySelector('[data-report-status]') || {getAttribute: () => null}).getAttribute('data-report-status') : null;
  R.report_warn = rep ? !!rep.querySelector('.sc-warn') : false;
  const sv = p ? p.querySelector('[data-check="served"]') : null;
  const st = sv ? sv.querySelector('.sc-status') : null;
  R.served_status = st ? st.getAttribute('data-status') : null;
  R.served_text = sv ? (sv.textContent || '') : '';
  R.media = p ? p.querySelectorAll('img, canvas').length : null;
  const h = window.__showcase;
  R.hook = h ? {version: h.version, story: typeof h.story === 'function', bindings: typeof h.bindings === 'function',
                n: typeof h.bindings === 'function' ? (h.bindings() || []).length : null} : null;
  const f = p ? p.querySelector('input.sc-filter') : null;
  if (f && R.rows.length) {
    const shown = () => Array.from(p.querySelectorAll('table.sc-bindings tbody tr')).filter(tr => !tr.hidden);
    f.value = R.rows[0]; f.dispatchEvent(new Event('input'));
    R.filtered_first = shown().some(tr => tr.getAttribute('data-key') === R.rows[0]);
    f.value = 'no binding reads this'; f.dispatchEvent(new Event('input'));
    R.filtered_none = shown().length;
    f.value = ''; f.dispatchEvent(new Event('input'));
    R.filtered_all = shown().length;
  }
  R.empty = p ? !!p.querySelector('.sc-empty') : null;
  return R;
})()"""


def layer_parts(html: str) -> tuple[dict | None, str | None]:
    """The layer's data (parsed) and its render script, as the page carries them."""
    m, r = DATA_RE.search(html), RENDER_RE.search(html)
    data = None
    if m:
        try:
            data = json.loads(_html.unescape(m.group(2)) if m.group(2).lstrip().startswith("&") else m.group(2))
        except ValueError:
            data = None
    return data, (r.group(1) if r else None)


def degraded_copy(html: str, data: dict) -> str:
    """The page with its data degraded: no report, the walk not run, no bindings."""
    doc = json.loads(json.dumps(data))
    doc["report"] = {"status": "brief-links", "origin": None, "bound": [], "missing": [], "briefs": None}
    doc["checks"] = {"build": (doc.get("checks") or {}).get("build"),
                     "served": {"status": "not-run", "why_not_run": NOT_RUN_WHY}}
    doc["bindings"] = []
    text = json.dumps(doc, ensure_ascii=False).replace("</", "<\\/")
    return DATA_RE.sub(lambda m: m.group(1) + text + m.group(3), html, count=1)


def _probe(chrome: str, html: str, work: Path, name: str) -> tuple[dict, list[str]]:
    path = work / f"{name}.html"
    path.write_text(html, encoding="utf-8")
    with cdp.Browser(chrome) as browser:
        page = browser.page()
        page.goto(path.as_uri())
        page.wait("document.readyState === 'complete' && !!document.querySelector('[data-layer=\"showcase\"]')", 60)
        time.sleep(0.5)
        got = page.eval(PROBE) or {}
        return got, [e["text"] for e in page.problem_records()]


def _c(cid: str, problems: list[str]) -> dict:
    return {"id": cid, "ok": not problems, "problems": problems}


def _checks(got: dict, data: dict, errors: list[str]) -> list[dict]:
    keys = [b.get("key") for b in data.get("bindings") or []]
    sc_status = ((data.get("showcase") or {}).get("report") or {}).get("status")
    served = (data.get("checks") or {}).get("served") or {}
    hook = got.get("hook") or {}
    return [
        _c("1", [] if got.get("button") and got.get("label") == LAYER["tab"] else
           [f"no tab button labelled {LAYER['tab']} in the host's nav (found {got.get('label')!r})"]),
        _c("2", [] if got.get("panes") == 1 and got.get("active") is False else
           [f"{got.get('panes')} pane(s), active {got.get('active')}: mounted once and inactive by default"]),
        _c("3", [] if got.get("surfaces") == SURFACES else [f"surfaces {got.get('surfaces')}, expected {SURFACES}"]),
        _c("4", ([] if got.get("rows") == keys else [f"{len(got.get('rows') or [])} binding row(s), the data holds {len(keys)}"])
           + ([] if not keys or (got.get("filtered_first") and got.get("filtered_none") == 0
                                 and got.get("filtered_all") == len(keys)) else
              [f"the filter does not narrow the rows (first {got.get('filtered_first')}, none "
               f"{got.get('filtered_none')}, all {got.get('filtered_all')})"])),
        _c("5", ([] if got.get("report_status") == sc_status else
                 [f"report status shown {got.get('report_status')!r}, SHOWCASE.json says {sc_status!r}"])
           + ([] if sc_status != "brief-links" or got.get("report_warn") else ["brief links are not rendered as a warning"])),
        _c("6", ([] if got.get("served_status") == served.get("status") else
                 [f"served walk shown {got.get('served_status')!r}, recorded {served.get('status')!r}"])
           + ([] if served.get("status") != "not-run" or str(served.get("why_not_run")) in (got.get("served_text") or "")
              else ["a walk that did not run does not show its reason"])),
        _c("7", errors + ([] if got.get("media") == 0 else [f"{got.get('media')} <img>/<canvas> in the pane"])
           + ([] if hook.get("version") == VERSION and hook.get("story") and hook.get("bindings") else
              [f"window.__showcase is {hook!r}"])),
    ]


def check_layer(chrome: str, out_html: Path) -> list[dict]:
    html = Path(out_html).read_text(encoding="utf-8")
    data, render = layer_parts(html)
    if data is None or render is None:
        return [_c("0", [f"{out_html} carries no showcase layer (data {data is not None}, render {render is not None})"])]
    with tempfile.TemporaryDirectory(prefix="smoke-show-") as tmp:
        got, errors = _probe(chrome, html, Path(tmp), "layer")
        checks = _checks(got, data, errors)
        size = len(render.encode("utf-8"))
        checks.append(_c("8", [] if size <= LAYER["max_render_kb"] * 1024 else
                         [f"the render is {size} bytes, over {LAYER['max_render_kb']} KB"]))
        low, low_errors = _probe(chrome, degraded_copy(html, data), Path(tmp), "degraded")
        problems = [f"degraded copy: {e}" for e in low_errors]
        if low.get("panes") != 1 or low.get("surfaces") != SURFACES:
            problems.append(f"degraded copy: {low.get('panes')} pane(s), surfaces {low.get('surfaces')}")
        if low.get("report_status") != "brief-links" or not low.get("report_warn"):
            problems.append("degraded copy: brief links not shown as a warning")
        if low.get("served_status") != "not-run" or NOT_RUN_WHY not in (low.get("served_text") or ""):
            problems.append("degraded copy: the walk not run is not shown with its reason")
        checks.append(_c("9", problems))
    return checks
