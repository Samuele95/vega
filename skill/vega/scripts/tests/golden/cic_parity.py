"""
cic_parity — what the golden CIC lane asserts on served pages (D6 §11.3 AD7, §11.5).

    serve(bundle, python, log, extra=()) -> Served          a bundle's own service on a free port
    layout(base, chrome, aliases=None, shots=None, scheme="dark", viewport=VIEWPORT, mobile=False)
        -> {view or view:state: {box: rect | None}}
    layout_problems(hand, built, tolerance=8) -> [str]      AD7: the same views, the same boxes
    content(base, chrome, show_scripts, aliases=None) -> {kind: [words]}
    content_problems(hand, built, allow) -> ([problem], [allowed])
                                                            AD7: what the hand-built page says is
                                                            still said, or an allowed change
    alert_presets(base, chrome, neutral) -> (record, [problem])
                                                            P-ROLE inverted: every preset whose
                                                            expected class is an attack is drawn
                                                            in --alert, the neutral one never

**AD7, layout parity** ("keeps EXACTLY the current visual style"): the hand-built page the v2.2
bundle serves and the page this release's showcase serves are opened in three passes — 1440×900
in the night theme (the page's default: `prefers-color-scheme: dark`), 1440×900 in the day theme
(`light`), and 390×844 on a phone in the night theme — reduced motion, each view by its hash
(`#window` is the hand-built page's name for `#many`), and the boxes of `header`, `nav.index`,
`main`, the view's first `h1.big` and its first canvas are read with getBoundingClientRect,
clipped to the viewport: what a reader sees at that size. Each view is read at rest and after
the reader acts (STATES: a preset answered, the first many-at-once source run, each decision
clicked, each code step opened), with the boxes the act changes (`EXTRA`, compared on their left
and right edges: their height and their place follow their words and the scroll the act made).
Night at 1440 compares every box whole, as it always did; the day theme and the phone compare
the frame (header, nav, main) whole and the other boxes on their left and right edges, since at
390 px a box's height and place follow the words above it. Every compared edge must match within
8 px; a box one page draws and the other does not is a difference too, and so is a view one page
shows and the other does not (a state only one page reaches, such as a decision the hand-built
page never had, is content, not layout).

**AD7, content parity**: what the hand-built page says — its limits' titles, its results facts'
labels, its choices' titles, each code step's report reference — is read from both served pages
(their own rendering) and must be said by the showcase too, or be an entry of the lane's allow
list with its reason (`cic.CONTENT_ALLOW`): an intended change is named, a lost one fails.

**P-ROLE inverted**: on the CIC lane the attack classes are alert by the user's stance, so the
try view's presets — one dot per demonstration row, in the colour of the class it should get —
must draw every attack class's dot in the alert colour (the `::before` background resolves to
`--alert`) and the neutral class's never.

Standard library and `vega-show/scripts/cdp.py` only.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = ["BOXES", "VIEWS", "Served", "alert_presets", "layout", "layout_problems", "serve"]

VIEWS = ("start", "context", "network", "try", "many", "results", "code")
HAND_ALIASES = {"many": "window"}            # the hand-built page's name for a view
BOXES = ("header", "nav", "main", "h1", "canvas")
VIEWPORT = (1440, 900)
# the passes: (name, prefers-color-scheme, viewport, mobile)
PASSES = (("night-1440", "dark", (1440, 900), False), ("day-1440", "light", (1440, 900), False),
          ("night-390", "dark", (390, 844), True))
# the boxes an act changes, compared on their left and right edges (their height and their place
# follow the words above them and the scroll the act made)
EXTRA = {"try": ("#presets", "#try-verdict"), "many": (".sources",), "network": ("#choices",),
         "code": ("#walk", "#code-step")}
STATE_MAX = 8                                 # decisions and code steps read per view
SETTLE_S = 1.5
READY_S = 120

_MEASURE = """(() => {
  const [W, H] = [innerWidth, innerHeight];
  const view = document.querySelector('.view[data-view="%s"]');
  const shown = (e) => e && e.getBoundingClientRect().width > 0 && e.getBoundingClientRect().height > 0;
  const box = (e) => {
    if (!shown(e)) return null;
    const r = e.getBoundingClientRect();
    const x0 = Math.max(0, r.left), y0 = Math.max(0, r.top), x1 = Math.min(W, r.right), y1 = Math.min(H, r.bottom);
    return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1].map((v) => Math.round(v * 10) / 10) : null;
  };
  if (!view || view.hidden) return {active: false};
  return {active: true,
          header: box(document.querySelector('header')), nav: box(document.querySelector('nav.index')),
          main: box(document.querySelector('main')),
          h1: box([...view.querySelectorAll('h1.big')].find(shown)),
          canvas: box([...view.querySelectorAll('canvas')].find(shown))};
})()"""

_PRESETS = """(() => {
  const probe = document.createElement('span');
  probe.style.color = 'var(--alert)';
  document.body.append(probe);
  const alert = getComputedStyle(probe).color;
  probe.remove();
  return {alert, done: !!document.querySelector('#presets[data-done]'),
          buttons: [...document.querySelectorAll('#presets button')].map((b) => ({
            id: b.dataset.id, cls: b.className, dot: getComputedStyle(b, '::before').backgroundColor}))};
})()"""


# --------------------------------------------------------------------------- the services
@dataclass
class Served:
    url: str
    process: subprocess.Popen
    log: Path

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def serve(bundle: Path, python: str, log: Path, extra=()) -> Served:
    """`python -m serve.app` in `bundle` on a free port (CPU, no bytecode written into the bundle),
    once /health says ready."""
    port = _free_port()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    fh = open(log, "wb")
    proc = subprocess.Popen([python, "-m", "serve.app", "--port", str(port), *extra], cwd=bundle,
                            env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    served, deadline = Served(f"http://127.0.0.1:{port}", proc, log), time.time() + READY_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the service of {bundle} exited {proc.returncode} (log {log})")
        try:
            if _get_json(served.url + "/health").get("ready"):
                return served
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    served.stop()
    raise RuntimeError(f"the service of {bundle} was not ready in {READY_S} s (log {log})")


def _cdp(show_scripts: Path):
    if str(show_scripts) not in sys.path:
        sys.path.insert(0, str(show_scripts))
    import cdp
    return cdp


# --------------------------------------------------------------------------- AD7
# after an act the page may have scrolled (a step opened, a decision clicked): its boxes are read in
# the document's coordinates, unclipped, so where the act left the scroll is not a difference
_MEASURE_DOC = (_MEASURE.replace("const box = (e) => {", "const box = (e) => { const [sx, sy] = e && getComputedStyle(e).position === 'fixed' ? [0, 0] : [scrollX, scrollY];")
                .replace("const x0 = Math.max(0, r.left), y0 = Math.max(0, r.top), x1 = Math.min(W, r.right), y1 = Math.min(H, r.bottom);",
                         "const x0 = r.left + sx, y0 = r.top + sy, x1 = r.right + sx, y1 = r.bottom + sy;"))
_EXTRA = """(() => {
  const [W, H] = [innerWidth, innerHeight], out = {};
  for (const sel of %s) {
    const e = document.querySelector(sel), r = e && e.getBoundingClientRect();
    out[sel] = r && r.width > 0 && r.height > 0 && r.left < W && r.top < H
      ? [r.left, r.top, Math.min(W, r.right)].map((v) => Math.round(v * 10) / 10) : null;
  }
  return out;
})()"""
_COUNT = "document.querySelectorAll(%s).length"
_CLICK = ("(() => { const e = document.querySelectorAll(%s)[%d]; if (e) { e.scrollIntoView({block: 'center'}); "
          "(e.querySelector('button') || e).click(); } return !!e; })()")


def _acts(view: str, page) -> list[tuple[str, str]]:
    """(state name, the act's JS) of a view: what a reader does there (module doc)."""
    if view == "try":
        return [("preset", _CLICK % (json.dumps("#presets button"), 0))]
    if view == "many":
        return [("source", _CLICK % (json.dumps(".sources button"), 0))]
    if view in ("network", "code"):
        sel = "#choices > button" if view == "network" else "#walk li"
        n = int(page.eval(_COUNT % json.dumps(sel)) or 0)
        return [(f"{'decision' if view == 'network' else 'step'}-{k + 1}", _CLICK % (json.dumps(sel), k))
                for k in range(min(n, STATE_MAX))]
    return []


def layout(base: str, chrome: str, show_scripts: Path, aliases=None, shots: Path | None = None,
           scheme: str = "dark", viewport: tuple = VIEWPORT, mobile: bool = False) -> dict:
    """{view: boxes, "view:state": boxes} of the page at `base`, each view opened by its hash, then
    each act of the view (`_acts`) done in turn and the view read again with its EXTRA boxes."""
    cdp, out = _cdp(show_scripts), {}
    with cdp.Browser(chrome, viewport=viewport) as browser:
        for view in VIEWS:
            name = (aliases or {}).get(view, view)
            page = browser.page(viewport=viewport, mobile=mobile, scheme=scheme, reduced_motion=True)
            page.goto(f"{base}/#{name}")
            ok = page.wait(f"(() => {{ const v = document.querySelector('.view[data-view=\"{name}\"]');"
                           " return !!v && !v.hidden && document.fonts.status === 'loaded'; })()", 30)
            time.sleep(SETTLE_S)
            got = page.eval(_MEASURE % name) if ok else {"active": False}
            if shots is not None:
                shots.mkdir(parents=True, exist_ok=True)
                page.screenshot(shots / f"{view}.png")
            out[view] = got
            if not got.get("active"):
                continue
            for state, act in _acts(view, page):
                page.eval(act)
                time.sleep(SETTLE_S)
                after = page.eval(_MEASURE_DOC % name)
                after["extra"] = page.eval(_EXTRA % json.dumps(list(EXTRA.get(view, ()))))
                out[f"{view}:{state}"] = after
                if shots is not None:
                    page.screenshot(shots / f"{view}-{state}.png")
    return out


def layout_problems(hand: dict, built: dict, tolerance: float = 8, frame_only: bool = False) -> list[str]:
    """AD7 over two `layout` readings: the hand-built page's and this release's — every view at
    rest, and every state both pages reach (a state one page alone has, such as a decision the
    hand-built page never had, is content: `content_problems`). With `frame_only` (the passes
    beyond night at 1440: the day theme, the phone) the header, the nav and main are compared
    whole and every other box on its left and right edges only, as wide and as far in, since at
    390 px its height and its place follow the words above it."""
    problems = []
    states = sorted(k for k in hand if ":" in k and k in built)
    for view in (*VIEWS, *states):
        a, b = hand.get(view) or {}, built.get(view) or {}
        if not (a.get("active") and b.get("active")):
            problems.append(f"{view}: the view is {'shown' if a.get('active') else 'absent'} on the "
                            f"hand-built page and {'shown' if b.get('active') else 'absent'} on the showcase")
            continue
        pairs = [(box, a.get(box), b.get(box)) for box in BOXES]
        pairs += [(sel, (a.get("extra") or {}).get(sel), (b.get("extra") or {}).get(sel))
                  for sel in EXTRA.get(view.split(":")[0], ()) if ":" in view]
        for box, ra, rb in pairs:
            if (ra is None) != (rb is None):
                problems.append(f"{view} {box}: drawn on {'the hand-built page' if ra else 'the showcase'} only")
            elif ra is not None:
                edges = range(len(ra))
                if (frame_only and box not in ("header", "nav", "main")) or (":" in view and box not in ("header", "nav")):
                    edges = (0, 2)                     # left and right: as wide and as far in
                worst = max(abs(ra[i] - rb[i]) for i in edges)
                if worst > tolerance:
                    problems.append(f"{view} {box}: {rb} against the hand-built {ra} (off by {worst:g} px "
                                    f"> {tolerance:g})")
    return problems


# --------------------------------------------------------------------------- AD7: content
_CONTENT = """(() => {
  const txt = (n) => (n ? n.textContent : '').replace(/\\s+/g, ' ').trim();
  const all = (sel) => [...document.querySelectorAll(sel)].map(txt).filter(Boolean);
  return {limits: all('#res-limits b'), facts: all('#res-facts .fact .l'),
          choices: [...document.querySelectorAll('#choices > button')].map((b) => txt(b).replace(/^\\d+\\s*/, ''))};
})()"""
_NOTE = "(() => { const r = document.querySelector('#code-step .code-ref'); return r ? r.textContent.replace(/\\s+/g, ' ').trim() : ''; })()"


def content(base: str, chrome: str, show_scripts: Path, aliases=None) -> dict:
    """What the page at `base` says, its own rendering read back: the results' limit titles and
    fact labels, the network's choice titles, and each code step's report reference ({step
    title: the reference line}), in lower case. Each view on a page of its own, read once drawn."""
    cdp, out = _cdp(show_scripts), {}
    with cdp.Browser(chrome, viewport=VIEWPORT) as browser:
        def opened(view, ready):
            page = browser.page(viewport=VIEWPORT, scheme="dark", reduced_motion=True)
            page.goto(f"{base}/#{(aliases or {}).get(view, view)}")
            page.wait(ready, 60)
            time.sleep(SETTLE_S)
            return page
        got = opened("results", "document.querySelectorAll('#res-facts .fact').length > 0").eval(_CONTENT)
        out["limits"], out["facts"] = ([s.lower() for s in got.get(k) or []] for k in ("limits", "facts"))
        got = opened("network", "document.querySelectorAll('#choices > button').length > 0").eval(_CONTENT)
        out["choices"] = [s.lower() for s in got.get("choices") or []]
        page, notes = opened("code", "document.querySelectorAll('#walk li').length > 0"), {}
        for k in range(int(page.eval(_COUNT % json.dumps("#walk li")) or 0)):
            page.eval(_CLICK % (json.dumps("#walk li"), k))
            time.sleep(0.3)
            title = page.eval(f"(() => {{ const e = document.querySelectorAll('#walk li')[{k}]; "
                              "return e ? e.textContent.replace(/\\s+/g, ' ').replace(/^\\d+\\s*/, '').trim() : ''; })()")
            notes[str(title).lower()] = str(page.eval(_NOTE)).lower()
        out["code_notes"] = notes
    return out


REPORT_OBJECT = re.compile(r"(?:equation \(\d+\)|figure \d+|table \d+)")


def content_problems(hand: dict, built: dict, allow: Mapping) -> tuple[list[str], list[str]]:
    """(the hand-built page's words the showcase no longer says and no allow entry names, the
    allowed changes with their reasons)."""
    problems, allowed = [], []
    for kind in ("limits", "facts", "choices"):
        have = set(built.get(kind) or [])
        for item in hand.get(kind) or []:
            if item in have:
                continue
            why = (allow.get(kind) or {}).get(item)
            (allowed if why else problems).append(f"{kind}: {item!r} " + (f"— {why}" if why else
                                                                            "is no longer said"))
    notes = built.get("code_notes") or {}
    for step, ref in (hand.get("code_notes") or {}).items():
        if not ref or step not in notes:
            continue
        object_ = ref.rsplit("·", 1)[-1].strip().split(" — ")[0]
        # the report objects the hand label names (it may name two, "equation (2) and figure 19",
        # or qualify one, "equation (2) as a whole"); a showcase ref is one report label, so it
        # names one of them. A label naming no numbered object is compared as its words.
        named = REPORT_OBJECT.findall(object_)
        if object_ and not (any(o in notes[step] for o in named) if named else object_ in notes[step]):
            why = (allow.get("code_notes") or {}).get(step)
            (allowed if why else problems).append(f"code note {step!r}: {object_!r} " + (
                f"— {why}" if why else f"is not what the showcase names ({notes[step][:80]!r})"))
    return problems, allowed


# --------------------------------------------------------------------------- P-ROLE inverted
def alert_presets(base: str, chrome: str, show_scripts: Path, neutral: str) -> tuple[dict, list[str]]:
    """(what the try view drew, the problems): every attack class's preset dot in --alert."""
    cdp = _cdp(show_scripts)
    presets = {p["id"]: p for p in _get_json(base + "/v1/examples").get("presets") or []}
    with cdp.Browser(chrome, viewport=VIEWPORT) as browser:
        page = browser.page(viewport=VIEWPORT, scheme="dark", reduced_motion=True)
        page.goto(f"{base}/#try")
        page.wait("!!document.querySelector('#presets[data-done]')", 60)
        time.sleep(SETTLE_S)
        got = page.eval(_PRESETS)
    problems, alert_n = [], 0
    if not got.get("done") or not got.get("buttons"):
        return got, ["the try view drew no presets"]
    for button in got["buttons"]:
        want = ((presets.get(button.get("id")) or {}).get("expected") or {})
        if want.get("kind") != "class":
            continue
        is_alert = "role-alert" in str(button.get("cls")).split()
        if want.get("value") == neutral:
            if is_alert or button.get("dot") == got["alert"]:
                problems.append(f"preset {button['id']} ({neutral}) is drawn in the alert colour")
        elif not is_alert or button.get("dot") != got["alert"]:
            problems.append(f"preset {button['id']} ({want.get('value')}) is not drawn in --alert "
                            f"(class {button.get('cls')!r}, dot {button.get('dot')}, alert {got['alert']})")
        else:
            alert_n += 1
    if not alert_n:
        problems.append("no attack class's preset is drawn in --alert")
    got["alert_presets"] = alert_n
    return got, problems
