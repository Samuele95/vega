#!/usr/bin/env python3
"""
page_smoke.py — the lanes' page stage (D6 §10; BUILD-PLAN R-H3, T09): a lane's served showcase
driven in headless Chrome four times (two widths, two themes, two motions), with the leak sets,
the style lock and the pseudo-locale, on top of the page driver's own checks.

    python3 page_smoke.py --base-url URL --bundle B --lane ID [--lanes-file F] --out OUT
                          [--shots DIR] [--runs R-a,R-b,R-c,R-d,R-q] [--chrome EXE] [--jobs N]

A thin wrapper over W15's driver (vega-show/scripts: `cdp.py`, `smoke_page.py`,
`smoke_page_checks.py`): each run opens the live service's page at `/?probe=<mode>`, lets the
page's own probe walk it, and judges the probe's output with the driver's 16 checks (P-PROBE …
P-LINKS). The theme of a day run is set by clicking the page's own control (`#theme`) on a first
load, never through the hook. Then every view of the tour is reached by its control and read again
(and photographed on R-a and R-b, `<shots>/<width>-<theme>-<view>.png`).

| run | viewport | theme | motion | probe | this stage's checks besides the driver's |
|---|---|---|---|---|---|
| R-a | 1440×900 | night | reduced | all | P-EXPECT P-FIT P-LEAK P-THEMED P-CONTRAST P-LOCK P-PERF |
| R-b | 390×844 mobile | day | reduced | all | P-FIT P-LEAK P-THEMED P-CONTRAST |
| R-c | 1440×900 | day | full | views | P-FIT P-THEMED P-CONTRAST P-LOCK |
| R-d | 390×844 mobile | night | full | views | P-FIT P-THEMED P-CONTRAST |
| R-q | 1440×900 | night | reduced | all, `?locale=qps-ploc` | P-QPS |

The stage's own checks (P-EXPECT, P-FIT, P-LEAK, P-THEMED, P-CONTRAST, P-LOCK, P-PERF, and P-QPS on
the lane's qps-ploc build, VS26) are page_smoke_checks.py's: its docstring says what each holds.

The result (`--out`) is `vega/page-smoke@1`: {schema, lane, base_url, chrome, verdict, runs: [{id,
width, height, mobile, theme, motion, mode, locale, ok, ready_ms, checks: [{id, ok, problems}],
views: [{id, visible}], console: [{level: error | allowed-404, text}], readings}]}, which
lanekit.observe reads (it counts the `error` ones). `readings` keeps what the adaptation report
(adaptation_report.py, D6 §11.3) compares across lanes: the style-lock reading (R-a, R-c), the
probe's palette, motif, missing and used strings keys, the layout boxes of `header`, `nav.index`,
`main`, `h1.big` and the primary canvas per view (R-a), and the story's ⟪…⟫ and the data's ‹…›
runs of the pseudo-locale page (R-q).

Exit 0: every run passed. 1: a check failed (each printed `FAIL:`). 2: the stage could not run
(no browser, the service or the qps build missing, a run that never got a probe output).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import html
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[2]
SHOW = SKILLS / "vega-show" / "scripts"
for _p in (HERE, SHOW, SHOW / "tests"):
    sys.path.insert(0, str(_p))

import cdp  # noqa: E402
import lint_templates as lint  # noqa: E402
import parity_server  # noqa: E402
import smoke_page  # noqa: E402
import style_lock  # noqa: E402
from lanekit import contracts  # noqa: E402
from page_smoke_checks import LaneFacts, Unrunnable, read_json, stage_checks  # noqa: E402
from smoke_page_checks import _allowed, judge  # noqa: E402

SCHEMA = contracts.PAGE_SMOKE
LANES_FILE = HERE / "lanes.json"
QPS_DIR = contracts.SHOW_PSEUDO_DIR
VIEW_S = 5.0


@dataclass(frozen=True)
class Run:
    id: str
    width: int
    height: int
    mobile: bool
    theme: str
    motion: str
    mode: str
    checks: tuple
    locale: str = "en"
    shots: bool = False


RUNS = (Run("R-a", 1440, 900, False, "night", "reduced", "all",
            ("P-EXPECT", "P-FIT", "P-LEAK", "P-THEMED", "P-CONTRAST", "P-LOCK", "P-PERF"), shots=True),
        Run("R-b", 390, 844, True, "day", "reduced", "all", ("P-FIT", "P-LEAK", "P-THEMED", "P-CONTRAST"), shots=True),
        Run("R-c", 1440, 900, False, "day", "full", "views", ("P-FIT", "P-THEMED", "P-CONTRAST", "P-LOCK")),
        Run("R-d", 390, 844, True, "night", "full", "views", ("P-FIT", "P-THEMED", "P-CONTRAST")),
        Run("R-q", 1440, 900, False, "night", "reduced", "all", ("P-QPS",), locale=contracts.PSEUDO_LOCALE))
QPS_DRIVER_CHECKS = ("P-PROBE", "P-ERR", "P-VIS", "P-TXT")


# ============================================================================ the page's readings

READY_JS = ("(async () => { const t0 = Date.now(); while (!window.__showcase && Date.now() - t0 < 60000) "
            "await new Promise(r => setTimeout(r, 50)); const h = window.__showcase; if (!h) return false; "
            "return await Promise.race([h.ready.then(() => true), new Promise(r => setTimeout(() => r(false), 60000))]); })()")
THEME_JS = "document.documentElement.dataset.theme || 'night'"
CLICK_THEME_JS = "(() => { const b = document.getElementById('theme'); if (!b) return null; b.click(); return document.documentElement.dataset.theme; })()"
GO_JS = """(async (id) => { const h = window.__showcase, key = k => document.dispatchEvent(new KeyboardEvent('keydown', {key: k, bubbles: true}));
  if (h.current() !== id) { if (id === 'start') key('0'); else if (id === 'code') key('c');
    else { const b = document.querySelector(`nav.index button[data-view="${id}"]`); if (b) b.click(); } }
  for (let t = 0; t < %d && h.current() !== id; t += 50) await new Promise(r => setTimeout(r, 50));
  await new Promise(r => requestAnimationFrame(() => setTimeout(r, 120)));
  return {view: h.current(), sw: document.documentElement.scrollWidth, iw: innerWidth}; })(%s)"""
TEXTS_JS = r"""(() => { const out = [], skip = 'script, style, template, #showcase-probe';
  const path = el => { const p = []; for (let e = el; e && e.nodeType === 1 && e !== document.documentElement; e = e.parentElement) {
      if (e.id) { p.unshift(e.tagName.toLowerCase() + '#' + e.id); break; }
      p.unshift(e.tagName.toLowerCase() + (e.classList.length ? '.' + [...e.classList].join('.') : '')); } return p.join(' > '); };
  out.push(['title', document.title]);
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = w.nextNode(); n; n = w.nextNode()) { const el = n.parentElement;
    if (!el || el.closest(skip) || !n.textContent.trim()) continue; out.push([path(el), n.textContent]); }
  for (const el of document.querySelectorAll('[title], [alt], [aria-label]'))
    for (const a of ['title', 'alt', 'aria-label']) if (el.hasAttribute(a)) out.push([path(el) + '@' + a, el.getAttribute(a)]);
  return out; })()"""
COLOURS_JS = """(() => { const cs = getComputedStyle(document.body); let bg = cs.backgroundColor;
  for (const el of [document.body, document.documentElement]) { const c = getComputedStyle(el).backgroundColor;
    if (!/rgba\\([^)]*,\\s*0\\)|transparent/.test(c)) { bg = c; break; } }
  return {fg: cs.color, bg}; })()"""
DOM_JS = "document.documentElement.outerHTML"
LAYOUT_RUN = "R-a"                      # 1440×900 night: AD7's layout parity reads its boxes
BOXES_JS = """(() => { const seen = el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const first = sel => [...document.querySelectorAll(sel)].find(seen) || null;
  const box = el => { if (!el) return null; const r = el.getBoundingClientRect(); return [r.x, r.y, r.width, r.height].map(Math.round); };
  return {header: box(first('header')), nav: box(first('nav.index')), main: box(first('main')), h1: box(first('h1.big')),
          canvas: box(first('main canvas') || document.getElementById('field'))}; })()"""


def drive(chrome: str, url: str, run: Run, lock: dict | None, shots: Path | None) -> dict:
    """One run: the theme click (day), the probe's walk, the views read again; what the browser heard."""
    query = f"?probe={run.mode}" + (f"&locale={run.locale}" if run.locale != "en" else "")
    out = {"clicked": None}
    with cdp.Browser(chrome, viewport=(run.width, run.height)) as browser:
        page = browser.page(viewport=(run.width, run.height), mobile=run.mobile, reduced_motion=run.motion == "reduced")
        if run.theme == "day":                      # the real control, on a first load; the page stores it
            page.goto(url + "/" + (f"?locale={run.locale}" if run.locale != "en" else ""))
            if not page.eval(READY_JS, timeout=90):
                raise Unrunnable(f"{run.id}: the page was not ready to switch its theme; it said {page.problems()[:3]}")
            out["clicked"] = page.eval(CLICK_THEME_JS)
        page.goto(url + "/" + query)
        raw = page.wait(smoke_page.PROBE_OUT, smoke_page.PROBE_S, every=0.5)
        if not raw:
            raise Unrunnable(f"{run.id}: the probe wrote nothing in {smoke_page.PROBE_S} s; the page said {page.problems()[:5]}")
        time.sleep(smoke_page.LATE_S)
        out["probe"] = json.loads(page.eval(smoke_page.PROBE_OUT) or raw)
        out["theme"] = page.eval(THEME_JS)
        out["texts"] = page.eval(TEXTS_JS) if "P-LEAK" in run.checks else []
        out["dom"] = page.eval(DOM_JS) if "P-QPS" in run.checks else ""
        out["colours"] = page.eval(COLOURS_JS)
        out["lock"] = page.eval(style_lock.probe_js(lock), timeout=120) if lock and "P-LOCK" in run.checks else None
        out["views"], out["boxes"] = [], {}
        for vid in (page.eval("window.__showcase.views()") or []) if "P-QPS" not in run.checks else []:
            got = page.eval(GO_JS % (int(VIEW_S * 1000), json.dumps(vid)), timeout=VIEW_S + 30)
            out["views"].append({"id": vid, **(got or {})})
            if run.id == LAYOUT_RUN and got and got.get("view") == vid:
                out["boxes"][vid] = page.eval(BOXES_JS)
            if shots is not None and run.shots and got and got.get("view") == vid:
                page.screenshot(shots / f"{run.width}-{run.theme}-{vid}.png")
        out.update(console=page.problem_records(), responses=page.responses(), misses=[])
    return out


# ============================================================================ the runs

@contextlib.contextmanager
def served(url: str, bundle: Path, run: Run):
    """The page a run opens: the live service, or (R-q) the qps build in front of it."""
    if run.locale == "en":
        yield url
        return
    build = bundle / contracts.SHOWCASE_DIR / QPS_DIR
    if not (build / "page" / "index.html").is_file() or not (build / "story.json").is_file():
        raise Unrunnable(f"no {contracts.PSEUDO_LOCALE} build in {build} (the show stage writes it)")
    srv = parity_server.start(build / "page", url, parity_server.ServeOptions(showcase=build))
    try:
        yield srv.url
    finally:
        srv.stop()


def target(url: str, bundle: Path) -> tuple[smoke_page.Served, dict]:
    """What the driver's checks need from the live service, and its page.json."""
    try:
        slot = smoke_page._get(url + "/v1/showcase")
        page_doc = smoke_page._get(url + "/showcase/page.json")
        docs = {"schema": smoke_page._get(url + "/v1/schema"),
                "attribution": smoke_page._get_or_none(url + "/showcase/data/attribution.json")}
    except (OSError, ValueError) as exc:
        raise Unrunnable(f"the service at {url} did not answer: {exc}") from exc
    if not slot.get("verified"):
        raise Unrunnable(f"the showcase slot is not verified: {slot.get('reason')}")
    manifest = read_json(bundle / "MANIFEST.json") or {}
    allowed = ("/favicon.ico",) + (() if manifest.get("monitor") else ("/v1/monitor",))
    return smoke_page.Served(bundle.name, url, page_doc.get("lane") or {}, lambda: [], allowed, lambda: docs, True), page_doc


def one(chrome: str, a, run: Run, lane: LaneFacts, base: smoke_page.Served, page_doc: dict, lock: dict | None) -> dict:
    t0 = time.time()
    with served(a.base_url, Path(a.bundle), run) as url:
        got = drive(chrome, url, run, lock, Path(a.shots) if a.shots else None)
    driver = judge(got, replace(base, url=url))
    if "P-QPS" in run.checks:
        driver = [c for c in driver if c["id"] in QPS_DRIVER_CHECKS]
    checks = driver + stage_checks(run, got, lane, page_doc, lock)
    seen = {**{k: v for k, v in (got["probe"].get("sub_views") or {}).items() if k in contracts.SUB_VIEWS["network"]},
            **(got["probe"].get("views") or {})}
    views = [{"id": k, "visible": bool(v.get("rendered"))} for k, v in seen.items()]
    return {"id": run.id, "width": run.width, "height": run.height, "mobile": run.mobile, "theme": run.theme,
            "motion": run.motion, "mode": run.mode, "locale": run.locale, "ok": all(c["ok"] for c in checks),
            "ready_ms": got["probe"].get("ready_ms"), "seconds": round(time.time() - t0, 1), "checks": checks,
            "views": views, "console": [{"level": "allowed-404" if _allowed(c.get("url"), c["text"], base.allowed) else "error",
                                         "text": c["text"]} for c in got["console"]], "readings": readings(got)}


def readings(got: dict) -> dict:
    """What the adaptation report compares across two lanes' pages (the module docstring)."""
    probe, ext = got["probe"], got["probe"].get("extensions") or {}
    dom = re.sub(r"<(script|style)\b.*?</\1>|<[^>]+>", " ", got.get("dom") or "", flags=re.S)
    runs = lambda a, b: sorted({m.strip() for m in re.findall(f"{a}([^{a}{b}]*){b}", html.unescape(dom)) if m.strip()})  # noqa: E731
    return {"lock": got.get("lock"), "palette": probe.get("palette"), "motif": probe.get("motif"),
            "missing_keys": probe.get("missing_keys") or [], "used_keys": ext.get("usedKeys"),
            "boxes": got.get("boxes") or {}, "content": {"story": runs("⟪", "⟫"), "data": runs("‹", "›")} if got.get("dom") else None}


def run_all(chrome: str, a) -> dict:
    bundle = Path(a.bundle).resolve()
    lane = LaneFacts.read(Path(a.lanes_file), a.lane, bundle)
    base, page_doc = target(a.base_url.rstrip("/"), bundle)
    lock = style_lock.load_lock()
    chosen = [r for r in RUNS if r.id in a.runs.split(",")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = [pool.submit(one, chrome, a, r, lane, base, page_doc, lock) for r in chosen]
        runs = [f.result() for f in futures]
    return {"schema": SCHEMA, "lane": a.lane, "base_url": a.base_url, "bundle": str(bundle), "chrome": chrome,
            "verdict": "pass" if all(r["ok"] for r in runs) else "fail", "runs": runs}


def report(doc: dict) -> int:
    for r in doc["runs"]:
        for c in r["checks"]:
            for p in c["problems"]:
                print(f"FAIL: {doc['lane']} {r['id']}: {c['id']}: {p}")
        if r["ok"]:
            print(f"OK: {doc['lane']} {r['id']} ({r['width']}x{r['height']} {r['theme']} {r['motion']} {r['locale']}): "
                  f"{len(r['checks'])} checks in {r['seconds']} s; ready in {r['ready_ms']} ms")
    return 0 if doc["verdict"] == "pass" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--base-url", required=True, help="the lane's live service")
    ap.add_argument("--bundle", required=True, help="the bundle the service serves (its showcase, its qps build)")
    ap.add_argument("--lane", required=True)
    ap.add_argument("--lanes-file", default=str(LANES_FILE))
    ap.add_argument("--out", required=True, help="the result, vega/page-smoke@1")
    ap.add_argument("--shots", help="the screenshots of R-a and R-b, one per view")
    ap.add_argument("--runs", default=",".join(r.id for r in RUNS))
    ap.add_argument("--chrome")
    ap.add_argument("--jobs", type=int, default=2, help="runs driven at once (default 2)")
    a = ap.parse_args(argv)
    chrome = cdp.find_chrome(a.chrome)
    if not chrome:
        print(f"ERROR: no headless browser (checked --chrome, $CHROME, {', '.join(cdp.CHROME_CANDIDATES)}): "
              "the page stage cannot run")
        return 2
    if a.shots:
        Path(a.shots).mkdir(parents=True, exist_ok=True)
    try:
        doc = run_all(chrome, a)
    except (Unrunnable, cdp.CdpError, style_lock.LockError, lint.LintError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return report(doc)


if __name__ == "__main__":
    sys.exit(main())
