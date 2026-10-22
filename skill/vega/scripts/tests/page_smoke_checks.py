"""
page_smoke_checks — the lanes' page stage's own checks (page_smoke.py; BUILD-PLAN T09, D6 §10.4-10.6),
each a function of what one run read off the page, returning its problems; and `LaneFacts`, what
the page of one lane is held to. The driver's 16 checks are W15's (vega-show/scripts/
smoke_page_checks.py); these come on top of them, on the runs page_smoke.py's table names.

    P-EXPECT   every view of the lane's set (lanekit.contracts.expected_views: the network's tried
               and weights disclosures when the campaign ran two runs or more) rendered, none off
               the tour; the motif page.json names is the lane's (`expect.motif`)
    P-FIT      on every view: no horizontal overflow (scrollWidth ≤ innerWidth + 1); a mobile run
               is 390 px wide
    P-LEAK     no hard token of foreign-literals.json (G00 excepted), no soft token the lane does
               not own, no word of another lane's VOCAB the lane does not own and no generic word
               is, in the page's text (textContent, every view the walk drew), its title or a
               title/alt/aria-label (D6 §10.6). The lane owns its generator's VOCAB (and the words
               VOCAB withholds from the template lint), its input columns, its target's column and
               classes and its lexicon, each name also in the words it is drawn with (`flow_vph`
               shown `flow vph`). Each leak names the node that holds it
    P-THEMED   the run's page is in the run's theme (a day run: the click flipped it, the probe
               walked it in day)
    P-CONTRAST body text on its background ≥ 4.5:1 in the run's theme
    P-LOCK     the style lock (style-lock.json: style_lock.probe_js on the live page, both themes)
               holds: the computed styles of D4's selectors and the resolved tokens
    P-PERF     the page was ready within 5 s of its navigation (CPU)
    P-QPS      the lane's qps-ploc build (showcase/.work/qps, never served by the bundle) served in
               front of the live service: once the ⟦…⟧ (strings table), ‹…› (data) and ⟪…⟫ (the
               show's prose, wrapped by build_showcase) runs are taken out, no word is left in the
               walked DOM (exempt: the code view's source, canvases, a one-character keycap);
               each word left names its node's CSS path (D6 §10.5, VS26). Also: every kind of run
               present, no missing strings key, no unresolved placeholder, the driver's P-PROBE,
               P-ERR, P-VIS and P-TXT

    stage_checks(run, got, lane, page_doc, lock) -> [{id, ok, problems}]   the run's checks, in order
"""
from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOW_TESTS = HERE.parents[2] / "vega-show" / "scripts" / "tests"
for _p in (HERE, SHOW_TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import check_pseudo_locale as qps  # noqa: E402
import lint_templates as lint  # noqa: E402
import style_lock  # noqa: E402
from lanekit import contracts  # noqa: E402

HARD_ALLOWED = frozenset({"G00"})              # the golden CIC lane shows its own dataset
CONTRAST_MIN, PERF_S, FIT_SLACK, MOBILE_W = 4.5, 5.0, 1, 390


class Unrunnable(Exception):
    """The stage could not run (page_smoke.py exits 2)."""


# ============================================================================ the lane

@dataclass(frozen=True)
class LaneFacts:
    """What the page of one lane is held to: its id, the motif it expects, its view set, the words
    it owns and the words of the other lanes."""
    id: str
    motif: str | None
    views: tuple
    own: frozenset
    others: frozenset

    @classmethod
    def read(cls, lanes_file: Path, lane_id: str, bundle: Path) -> "LaneFacts":
        doc = json.loads(lanes_file.read_text(encoding="utf-8"))
        lanes = {l["id"]: l for l in doc["lanes"]}
        if lane_id not in lanes:
            raise Unrunnable(f"lane {lane_id} is not in {lanes_file}")
        lane = lanes[lane_id]
        vocab, withheld = _vocab(lanes_file.parent)
        own = set(vocab.get(lane["generator"], ())) | set(withheld.get(lane["generator"], ()))
        own |= _bundle_words(bundle) | _lexicon_words(lane)
        own |= own_parts(own)
        others = set().union(*(vocab.get(l["generator"], frozenset()) for l in lanes.values()
                               if l["generator"] != lane["generator"]))
        facts = read_json(bundle / contracts.SHOWCASE_DIR / "plan" / "facts.json") or {}
        runs = len(((facts.get("groups") or {}).get("runs")) or [])
        return cls(lane_id, (lane.get("expect") or {}).get("motif"), contracts.expected_views(runs),
                   frozenset(own), frozenset(others))


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _vocab(tests_dir: Path) -> tuple[dict, dict]:
    """{generator: VOCAB} and {generator: VOCAB_WITHHELD} of the lane generators (T04's reader)."""
    vocab, _ = lint.generator_vocab(tests_dir)
    sys.path.insert(0, str(tests_dir))
    try:
        from lanekit.datasets import GENERATORS
        withheld = {k: frozenset(getattr(m, "VOCAB_WITHHELD", ()) or ()) for k, m in GENERATORS.items()}
    finally:
        sys.path.remove(str(tests_dir))
    for name, rel in contracts.EXTERNAL_GENERATORS.items():     # cic (G00): golden/cic.py's own words
        withheld.setdefault(name, _module_words(tests_dir.parents[2] / rel, "VOCAB_WITHHELD"))
    return vocab, withheld


def _module_words(path: Path, name: str) -> frozenset:
    """A module's literal tuple of words `name`, read without importing it (empty when absent)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return frozenset()
    found = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == name for t in n.targets)]
    return frozenset(ast.literal_eval(found[0])) if found else frozenset()


def _bundle_words(bundle: Path) -> set:
    """The lane's own data words: its input columns, its target's column and classes."""
    schema = read_json(bundle / "schema.json") or {}
    netspec = read_json(bundle / "netspec.json") or {}
    task = netspec.get("task") or {}
    names = [c.get("name") for c in schema.get("input") or [] if isinstance(c, dict)]
    names += [task.get("label"), *(task.get("classes") or [])]
    return {str(n) for n in names if n is not None}


def own_parts(words) -> set:
    """The words of the lane's own names as the page spells them: a column `flow_vph` is drawn
    `flow vph`, and its `flow` is the lane's own, not the CIC row noun."""
    return {part for w in words for part in re.split(r"[_\-\s]+", str(w)) if part}


def _lexicon_words(lane: dict) -> set:
    return {w for entry in (lane.get("lexicon") or {}).values() if isinstance(entry, dict)
            for k in ("one", "many") for w in str(entry.get(k) or "").split()}


# ============================================================================ the checks

def stage_checks(run, got: dict, lane: LaneFacts, page_doc: dict, lock: dict | None) -> list[dict]:
    """This stage's checks of one run (the module docstring's table)."""
    found = {"P-EXPECT": lambda: expect_problems(got["probe"], lane, page_doc),
             "P-FIT": lambda: fit_problems(got["views"], run),
             "P-LEAK": lambda: leak_problems(got["texts"], lane),
             "P-THEMED": lambda: themed_problems(got, run),
             "P-CONTRAST": lambda: contrast_problems(got["colours"], run.theme),
             "P-LOCK": lambda: style_lock.compare(lock, got["lock"]) if lock and got.get("lock") else
             ["no style-lock reading of the page"],
             "P-PERF": lambda: perf_problems(got["probe"]),
             "P-QPS": lambda: qps_problems(got["probe"], got["dom"])}
    ready = bool(got["probe"].get("ready"))
    return [{"id": c, "ok": not (p := (found[c]() if ready else [f"{c} not judged: the page never became ready"])),
             "problems": p} for c in run.checks]


def expect_problems(probe: dict, lane: LaneFacts, page_doc: dict) -> list[str]:
    views, subs = probe.get("views") or {}, probe.get("sub_views") or {}
    out = []
    for vid in lane.views:
        sub = next((k for k, v in contracts.SUB_VIEWS.items() if vid in v), None)
        v = (subs if sub else views).get(vid) or {}
        if v.get("off"):
            out.append(f"view {vid} is off the tour: {v['off']}")
        elif not v.get("rendered"):
            out.append(f"view {vid} of the lane's set was not rendered")
    motif = (page_doc.get("motif") or {}).get("id")
    if lane.motif and motif != lane.motif:
        out.append(f"page.json's motif is {motif!r}, the lane expects {lane.motif!r}")
    return out


def fit_problems(views: list, run) -> list[str]:
    if not views:
        return ["no view of the tour was reached to measure"]
    out = [f"view {v['id']}: the page is {v.get('sw')} px wide in a {v.get('iw')} px viewport"
           for v in views if v.get("sw") is None or v["sw"] > v["iw"] + FIT_SLACK]
    out += [f"view {v['id']}: the mobile viewport is {v.get('iw')} px, not {MOBILE_W}"
            for v in views if run.mobile and v.get("iw") != MOBILE_W]
    return out + [f"view {v['id']} not reached by its control (the page shows {v.get('view')!r})"
                  for v in views if v.get("view") != v["id"]]


def leak_problems(texts: list, lane: LaneFacts, literals: Path = lint.LITERALS) -> list[str]:
    """Each leaked token with the node that holds it (D6 §10.6; lint_templates.page_leaks reads the sets)."""
    if not any(text.strip() for _, text in texts):
        return ["no text was read off the page"]
    hard = {e["token"] for e in lint.read_json(literals, "the literals").get("hard") or []}
    leaks = lambda text: [t for t in lint.page_leaks(text, lane.own, lane.others, literals)  # noqa: E731
                          if not (t in hard and lane.id in HARD_ALLOWED)]
    found = set(leaks("\n".join(t for _, t in texts)))   # the page's text is the gate; the nodes name it
    if not found:
        return []
    out = []
    for path, text in texts:
        out += [f"{tok!r} in {path}: {text.strip()[:100]!r}" for tok in leaks(text) if tok in found]
    return out or [f"{sorted(found)} leak across two text nodes"]


def themed_problems(got: dict, run) -> list[str]:
    out = [] if got.get("theme") == run.theme else [f"the page is in {got.get('theme')!r}, the run is {run.theme!r}"]
    if run.theme == "day" and got.get("clicked") != "day":
        out.append(f"the theme control did not switch the page to day (it shows {got.get('clicked')!r})")
    return out


def _rgb(css: str) -> tuple | None:
    m = re.match(r"rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)", css or "")
    return tuple(float(x) for x in m.groups()) if m else None


def contrast(fg: tuple, bg: tuple) -> float:
    def lum(c):
        ch = [x / 255 for x in c]
        ch = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def contrast_problems(colours: dict, theme: str) -> list[str]:
    fg, bg = _rgb((colours or {}).get("fg")), _rgb((colours or {}).get("bg"))
    if fg is None or bg is None:
        return [f"the body's colours could not be read: {colours}"]
    ratio = contrast(fg, bg)
    return [] if ratio >= CONTRAST_MIN else [f"{theme}: body text on its background is {ratio:.2f}:1, under {CONTRAST_MIN}:1"]


def perf_problems(probe: dict) -> list[str]:
    ms = probe.get("ready_ms")
    return [] if isinstance(ms, (int, float)) and ms <= PERF_S * 1000 else [f"ready after {ms} ms, over {PERF_S:g} s"]


def qps_problems(probe: dict, dom: str) -> list[str]:
    found, counts = qps.wrapped_runs(dom)
    ext, out = probe.get("extensions") or {}, []
    if probe.get("locale") != contracts.PSEUDO_LOCALE:
        out.append(f"the page ran in {probe.get('locale')!r}, not {contracts.PSEUDO_LOCALE}")
    for name in ("missingKeys", "unresolvedPlaceholders"):
        if name not in ext:
            out.append(f"the probe does not report {name}")
        elif ext[name]:
            out.append(f"{name}: {ext[name]}")
    out += [f"no {k}…{qps.RUNS[k]} run in the page: the pseudo-locale did not apply" for k, n in counts.items() if not n]
    return out + [f"unwrapped text in {f['path']}: {f['words']} (in {f['text']!r})" for f in found]
