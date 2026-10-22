#!/usr/bin/env python3
"""
adaptation_report.py — the adaptation proof (D6 §11; BUILD-PLAN T12): two lanes' served pages keep
one style and one table of words, and carry each its own content, colours and motif.

    adaptation_report.py --matrix DIR --pairs L02:L01,L10:L02[,L02:G00] [--json OUT] [--md OUT]

For each pair A:B it reads what the lanes' page stage recorded (`lanes/<id>/results/page.json`,
`vega/page-smoke@1`: each run's `readings`, page_smoke.py), the lanes' showcases (the role map in
`story.json`, the strings table the page was built with) and the registry (modality, motif):

| id | assertion per pair (A:B) |
|---|---|
| AD1 | the style-lock readings (style_lock.probe_js: computed styles of D4's selectors and the resolved tokens) are identical in A's and B's pages, night (R-a) and day (R-c) |
| AD2 | every chrome key A's page rendered (the page's `usedKeys`, every run) is a key of A's strings table and none is missing; every key both pages rendered has the same English in both tables; a page with no classes renders no classification-only key (a class view key's variant `@binary`/`@classes`/`@labels`, or a key naming classes, labels or a confusion matrix: D6 §11.2) |
| AD3 | the content — the story's ⟪…⟫ and the data's ‹…› runs of the pseudo-locale page (R-q) — overlaps little: the Jaccard index of the two content word sets (below) is under 0.3 |
| AD4 | A's page has zero hard or cross-lane leaks (P-LEAK held on every run that checks it) |
| AD5 | A's class colours come from A's role map: the page counted as many alert roles as the map holds; a map with no alert draws no alert (no `.role-alert` element, no alert mark on the field: L02, L10); a map with one draws it (G00's attack classes) |
| AD6 | A's motif is its modality's (lanekit.contracts.MOTIFS_BY_MODALITY, and the lane's own when it declares one) and the field drew it; when A's and B's modalities differ, so do their motifs (L10 vs L02) |
| AD7 | when one side is a golden lane (G00), layout parity at 1440×900 night ("keeps EXACTLY the current visual style"): the golden lane's own measurement (`results/golden.json` `layout`, golden/cic_parity.py) found every box of every view — `header`, `nav.index`, `main`, `h1.big`, the first canvas — of this release's page within 8 px of the hand-built page's, on the same content; and A's and B's pages have the same view set and the same frame (`header`, `nav.index`, `main`) within 8 px in every view |

AD3's content words are the words a lane brings to its page: the runs' words without the
chain's own — English function words, the words of the strings table (the template's chrome),
the words of lanekit's story author (lanekit/authoring.py's literals: the stand-in for the model
the scripted lanes use, whose sentences about the chain every lane's story shares), and the
module names of the code view's file paths (the backend's code layout; a path keeps its package,
the lane's own name), and the chain's own marker lines a page quotes verbatim (a run that is an
`OK:`/`WARN:`/`ERROR:` line: the pack's or the evaluation's words about the chain, which every
lane whose records hold the same warning shares). What is left is the lane's own: its names, columns, classes, lexicon and
numbers' words. A page that shows another lane's data, or one fixed story, fails it; how much
two model-written stories share is T5's rehearsal to judge (D6 §11.4).

AD7 compares the content boxes (`h1.big`, the canvas) only on one content, where the golden lane
measures them: across two lanes a headline's height follows its own words (AD3 wants them to
differ), and what sits under it moves with it.

A pair whose golden side did not run (its lane `requires` an unset variable, or the tier holds no
golden lane) is `skipped` with a `WARN:` line; any other missing input fails the pair.

Writes `adaptation.json` (`vega/adaptation-report@1`) and `adaptation.md`, by default in the
matrix dir. Exit 0 every pair passed or was skipped, 1 an assertion failed, 2 a usage error.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lanekit import contracts, registry  # noqa: E402

SCHEMA = "vega/adaptation-report@1"
JACCARD_MAX = 0.3
LAYOUT_PX = 8
FRAME = ("header", "nav", "main")                       # the boxes no content moves
NIGHT_RUN, DAY_RUN, QPS_RUN = "R-a", "R-c", "R-q"
CLASS_VARIANTS = ("binary", "classes", "labels")        # the class view keys (vega-show catalog view_keys)
CLASS_SEGMENTS = ("classes", "labels", "confusion", "per_label")
STRING_ENTRY = re.compile(r'^\s*"([^"\\]+)"\s*:\s*("(?:[^"\\\n]|\\.)*")\s*,?\s*$', re.M)
WORD = re.compile(r"[a-z][a-z'-]+")
PATH = re.compile(r"/|\.py$")                            # a code view file path
FUNCTION_WORDS = frozenset("""a an the and or but nor of to in on at by for with from into onto over under than then
that this these those it its is are was were be been being has have had do does did not no as so if when
while which who whom whose what where why how each every any all some one per can may might must will would
shall should their there they them we our you your he she his her i me my via out up down off about also only
more most less least very much many such same other another both either neither""".split())


# --------------------------------------------------------------------------- one lane's record
class LaneRecord:
    """What one lane's page stage and showcase hold."""

    def __init__(self, matrix: Path, lane: registry.Lane) -> None:
        self.lane, self.dir = lane, matrix / "lanes" / lane.id
        self.page = _load(self.dir / "results" / "page.json")
        observed = _load(self.dir / "observed.json") or {}
        bundle = (observed.get("artifacts") or {}).get("bundle")
        self.showcase = self.dir / bundle / contracts.SHOWCASE_DIR if bundle else None
        self.observed = observed

    @property
    def present(self) -> bool:
        return isinstance(self.page, dict) and bool(self.page.get("runs"))

    def run(self, run_id: str) -> Mapping[str, Any]:
        return next((r for r in (self.page or {}).get("runs") or [] if r.get("id") == run_id), {})

    def readings(self, run_id: str) -> Mapping[str, Any]:
        return self.run(run_id).get("readings") or {}

    def used_keys(self) -> set[str] | None:
        found = [r.get("readings", {}).get("used_keys") for r in (self.page or {}).get("runs") or []]
        if not any(isinstance(f, list) for f in found):
            return None
        return {k for f in found if isinstance(f, list) for k in f}

    def missing_keys(self) -> set[str]:
        return {k for r in (self.page or {}).get("runs") or [] for k in (r.get("readings") or {}).get("missing_keys") or []}

    def table(self) -> dict[str, str]:
        """The strings table the page was built with: the showcase copy's area fragments."""
        out: dict[str, str] = {}
        folder = self.showcase / "page" / "js" / "core" / "strings" if self.showcase else None
        for path in sorted(folder.glob("*.en.js")) if folder and folder.is_dir() else []:
            for key, value in STRING_ENTRY.findall(path.read_text(encoding="utf-8")):
                out[key] = json.loads(value)
        return out

    def roles(self) -> dict[str, str] | None:
        story = _load(self.showcase / "story.json") if self.showcase else None
        roles = ((story or {}).get("class_roles") or {}).get("roles")
        return dict(roles) if isinstance(roles, dict) else None

    def has_classes(self) -> bool:
        return bool(self.lane.classes) or self.roles() not in (None, {})


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- the assertions
def ad1(a: LaneRecord, b: LaneRecord) -> list[str]:
    out = []
    for run_id, theme in ((NIGHT_RUN, "night"), (DAY_RUN, "day")):
        la, lb = a.readings(run_id).get("lock"), b.readings(run_id).get("lock")
        if not la or not lb:
            out.append(f"{theme}: no style-lock reading of {a.lane.id if not la else b.lane.id} ({run_id})")
        elif la != lb:
            out.append(f"{theme}: the readings differ at {', '.join(_diff(la, lb)[:5])}")
    return out


def _diff(x: Any, y: Any, path: str = "") -> list[str]:
    if isinstance(x, dict) and isinstance(y, dict):
        return [d for k in sorted(set(x) | set(y)) for d in _diff(x.get(k), y.get(k), f"{path}/{k}")]
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        return [d for i, (p, q) in enumerate(zip(x, y)) for d in _diff(p, q, f"{path}[{i}]")]
    return [] if x == y else [f"{path or '/'} ({x!r} vs {y!r})"[:200]]


def class_only(key: str) -> bool:
    """A classification-only chrome key: a class view key's variant, or a key naming classes,
    labels or a confusion matrix (D6 §11.2, contracts.CLASS_ONLY_KEYS's reading)."""
    variant = key.split("@", 1)[1].split("#")[0].split("+")[0] if "@" in key else None
    base = key.split("@")[0].split("#")[0]
    return variant in CLASS_VARIANTS or any(seg in CLASS_SEGMENTS for seg in base.split("."))


def ad2(a: LaneRecord, b: LaneRecord) -> list[str]:
    used_a, used_b = a.used_keys(), b.used_keys()
    if used_a is None or used_b is None:
        return [f"{a.lane.id if used_a is None else b.lane.id}'s page reports no used strings keys"]
    ta, tb = a.table(), b.table()
    if not ta or not tb:
        return [f"{a.lane.id if not ta else b.lane.id}'s showcase holds no strings table"]
    chrome = {k for k in used_a if not k.startswith("explain.")}
    out = [f"{len(chrome - set(ta))} rendered key(s) not in {a.lane.id}'s table: {sorted(chrome - set(ta))[:8]}"] \
        if chrome - set(ta) else []
    if a.missing_keys():
        out.append(f"{a.lane.id}'s page missed key(s) {sorted(a.missing_keys())[:8]}")
    shared = sorted(k for k in chrome & used_b if k in ta and k in tb and ta[k] != tb[k])
    if shared:
        out.append(f"{len(shared)} shared key(s) render different English: {shared[:8]}")
    if not a.has_classes():
        leaked = sorted(k for k in chrome if class_only(k))
        if leaked:
            out.append(f"{a.lane.id} has no classes but rendered classification key(s) {leaked[:8]}")
    return out


def words(texts) -> set[str]:
    return {w.strip("'-") for w in WORD.findall(" ".join(texts).lower())} - FUNCTION_WORDS - {""}


def chain_words(*tables: Mapping[str, str]) -> set[str]:
    """The words every lane's page says: the strings tables' and lanekit's story author's."""
    tree = ast.parse((HERE / "lanekit" / "authoring.py").read_text(encoding="utf-8"))
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return words(literals) | words(v for t in tables for v in t.values())


# a chain marker quoted verbatim, as a run reads it (the pseudo-locale's ⟪ ‹ marks before it)
MARKER_LINE = re.compile(r"^[\s⟪‹]*(?:OK|WARN|ERROR|FORK|GATE):")


def content_words(rec: LaneRecord, chain: set[str]) -> set[str] | None:
    content = rec.readings(QPS_RUN).get("content")
    if not isinstance(content, dict):
        return None
    data = [d.split("/")[0] if PATH.search(d) else d for d in content.get("data") or []
            if not MARKER_LINE.match(str(d))]
    story = [t for t in content.get("story") or [] if not MARKER_LINE.match(str(t))]
    return words([*story, *data]) - chain


def ad3(a: LaneRecord, b: LaneRecord) -> tuple[list[str], dict]:
    chain = chain_words(a.table(), b.table())
    wa, wb = content_words(a, chain), content_words(b, chain)
    if not wa or not wb:
        return [f"no pseudo-locale content read off {a.lane.id if not wa else b.lane.id}'s page ({QPS_RUN})"], {}
    jaccard = len(wa & wb) / len(wa | wb)
    detail = {"jaccard": round(jaccard, 3), "words": [len(wa), len(wb)], "shared": sorted(wa & wb)[:40]}
    return ([] if jaccard < JACCARD_MAX else
            [f"the content overlaps: Jaccard {jaccard:.3f} ≥ {JACCARD_MAX} ({len(wa & wb)} shared of {len(wa | wb)})"]), detail


def ad4(a: LaneRecord) -> list[str]:
    checks = [(r["id"], c) for r in a.page["runs"] for c in r.get("checks") or [] if c.get("id") == "P-LEAK"]
    if not checks:
        return [f"{a.lane.id}'s page stage ran no P-LEAK check"]
    return [f"{rid}: {p}" for rid, c in checks if not c.get("ok") for p in (c.get("problems") or ["leak"])][:10]


def ad5(a: LaneRecord) -> list[str]:
    roles = a.roles() or {}
    alert = sum(1 for r in roles.values() if r == "alert")
    pal, motif = a.readings(NIGHT_RUN).get("palette"), a.readings(NIGHT_RUN).get("motif") or {}
    if not isinstance(pal, dict):
        return [f"{a.lane.id}'s page recorded no palette ({NIGHT_RUN})"]
    out = []
    if pal.get("alert_roles") != alert:
        out.append(f"the page counted {pal.get('alert_roles')} alert role(s); {a.lane.id}'s role map holds {alert}")
    drew = bool(pal.get("alert_used")) or bool(pal.get("role_alert_elements")) or bool(motif.get("alert_marks"))
    if alert == 0 and drew:
        out.append(f"{a.lane.id}'s role map has no alert, yet the page drew alert ({pal}, field marks {motif.get('alert_marks')})")
    if alert > 0 and not pal.get("alert_used"):
        out.append(f"{a.lane.id}'s role map names {alert} alert class(es), yet the page drew none")
    return out


def ad6(a: LaneRecord, b: LaneRecord) -> list[str]:
    def motif(rec: LaneRecord) -> tuple[str | None, str | None]:
        m = rec.readings(NIGHT_RUN).get("motif") or {}
        return m.get("id"), m.get("data_motif")
    (ma, drawn), (mb, _) = motif(a), motif(b)
    modality = a.lane.modality
    allowed = contracts.MOTIFS_BY_MODALITY.get(modality, {}).get("admissible", ())
    out = []
    if ma not in allowed:
        out.append(f"{a.lane.id}'s motif {ma!r} is not one of its modality's ({modality}: {', '.join(allowed)})")
    if drawn != ma:
        out.append(f"the field drew {drawn!r}, page.json names {ma!r}")
    declared = a.lane.expect.get("motif")
    if declared and ma != declared:
        out.append(f"{a.lane.id} declares motif {declared!r}; the page shows {ma!r}")
    if modality != b.lane.modality and ma == mb:
        out.append(f"{a.lane.id} ({modality}) and {b.lane.id} ({b.lane.modality}) show the same motif {ma!r}")
    return out


GOLDEN_PASSES = ("night-1440", "day-1440", "night-390")   # golden/cic_parity.PASSES, by name


def golden_layout(rec: LaneRecord) -> list[str]:
    """The golden lane's own layout parity against the hand-built page (golden/cic_parity.py):
    `layout.passes` holds one measurement per pass — 1440×900 night (every box), 1440×900 day and
    390×844 night on a phone (the frame) — each with both pages' boxes."""
    layout = (_load(rec.dir / "results" / "golden.json") or {}).get("layout")
    passes = layout.get("passes") if isinstance(layout, dict) else None
    first = passes.get(GOLDEN_PASSES[0]) if isinstance(passes, dict) else None
    if not isinstance(first, dict) or not first.get("hand") or not first.get("showcase"):
        return [f"{rec.lane.id} recorded no layout parity (results/golden.json layout)"]
    out = [f"{rec.lane.id}'s layout parity has no {name} pass" for name in GOLDEN_PASSES[1:]
           if not isinstance(passes.get(name), dict) or not passes[name].get("hand")
           or not passes[name].get("showcase")]
    if not isinstance(layout.get("tolerance_px"), (int, float)) or layout["tolerance_px"] > LAYOUT_PX:
        out.append(f"{rec.lane.id}'s layout tolerance is {layout.get('tolerance_px')} px, not {LAYOUT_PX}")
    return out + [f"{rec.lane.id} against the hand-built page: {p}" for p in layout.get("problems") or []]


def ad7(a: LaneRecord, b: LaneRecord) -> list[str]:
    out = [p for rec in (a, b) if golden(rec.lane) for p in golden_layout(rec)]
    ba, bb = a.readings(NIGHT_RUN).get("boxes") or {}, b.readings(NIGHT_RUN).get("boxes") or {}
    if not ba or not bb:
        return out + [f"no layout boxes recorded on {a.lane.id if not ba else b.lane.id}'s page ({NIGHT_RUN})"]
    out += [] if set(ba) == set(bb) else [f"the view sets differ: {sorted(ba)} vs {sorted(bb)}"]
    for view in sorted(set(ba) & set(bb)):
        for name in FRAME:
            x, y = ba[view].get(name), bb[view].get(name)
            if x is None or y is None:
                if x != y:
                    out.append(f"{view}: {name} is on one page only")
                continue
            worst = max(abs(p - q) for p, q in zip(x, y))
            if worst > LAYOUT_PX:
                out.append(f"{view}: {name} moved {worst} px ({x} vs {y})")
    return out


# --------------------------------------------------------------------------- one pair
def golden(lane: registry.Lane) -> bool:
    return lane.id.startswith("G")


def judge(matrix: Path, lanes: Mapping[str, registry.Lane], pair: str) -> dict:
    a_id, b_id = pair.split(":")
    if a_id not in lanes or b_id not in lanes:
        return {"pair": pair, "status": "fail", "why": f"unknown lane in {pair}", "checks": []}
    a, b = LaneRecord(matrix, lanes[a_id]), LaneRecord(matrix, lanes[b_id])
    absent = [r for r in (a, b) if not r.present]
    if absent:
        rec = absent[0]
        why = f"{rec.lane.id} has no page result ({rec.dir / 'results' / 'page.json'})"
        if all(golden(r.lane) for r in absent):
            return {"pair": pair, "status": "skipped", "why": why + "; the golden lane runs nightly with "
                    + ", ".join(rec.lane.get("requires") or ["its inputs"]), "checks": []}
        return {"pair": pair, "status": "fail", "why": why, "checks": []}
    j_problems, j_detail = ad3(a, b)
    checks = [("AD1", ad1(a, b), None), ("AD2", ad2(a, b), None), ("AD3", j_problems, j_detail),
              ("AD4", ad4(a), None), ("AD5", ad5(a), None), ("AD6", ad6(a, b), None)]
    if golden(a.lane) or golden(b.lane):
        checks.append(("AD7", ad7(a, b), None))
    rows = [{"id": cid, "ok": not problems, "problems": problems, **({"detail": d} if d else {})}
            for cid, problems, d in checks]
    return {"pair": pair, "status": "pass" if all(r["ok"] for r in rows) else "fail", "checks": rows}


# --------------------------------------------------------------------------- the report
def render(doc: Mapping[str, Any]) -> str:
    out = [f"# Adaptation — {doc['verdict']}", "", f"Generated {doc['generated_at']} over {doc['matrix']}.", ""]
    for p in doc["pairs"]:
        out += [f"## {p['pair']} — {p['status']}", ""]
        if p.get("why"):
            out += [p["why"], ""]
        if p["checks"]:
            out += ["| id | ok | problems / detail |", "|---|---|---|"]
            for c in p["checks"]:
                said = "; ".join(c["problems"]) or (json.dumps(c["detail"], ensure_ascii=False)[:300] if c.get("detail") else "—")
                out.append(f"| {c['id']} | {'yes' if c['ok'] else '**no**'} | {said} |")
            out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--pairs", required=True, help="A:B[,A:B…], e.g. L02:L01,L10:L02,L02:G00")
    ap.add_argument("--lanes-file", default=str(HERE / "lanes.json"))
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--md", dest="md_out")
    a = ap.parse_args(argv)
    pairs = [p.strip() for p in a.pairs.split(",") if p.strip()]
    if not pairs or any(not re.fullmatch(r"[LG]\d{2}:[LG]\d{2}", p) for p in pairs):
        ap.error(f"--pairs must be A:B lane ids, got {a.pairs!r}")
    matrix = Path(a.matrix).resolve()
    lanes = registry.load_lanes(Path(a.lanes_file))
    results = [judge(matrix, lanes, p) for p in pairs]
    verdict = "FAIL" if any(r["status"] == "fail" for r in results) else "PASS"
    doc = {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "matrix": str(matrix), "pairs": results, "verdict": verdict}
    Path(a.json_out or matrix / "adaptation.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                                                               encoding="utf-8")
    Path(a.md_out or matrix / "adaptation.md").write_text(render(doc), encoding="utf-8")
    for r in results:
        if r["status"] == "skipped":
            print(f"WARN: {r['pair']} skipped — {r['why']}")
        elif r["status"] == "fail" and r.get("why"):
            print(f"ERROR: {r['pair']}: {r['why']}")
        for c in r["checks"]:
            for p in c["problems"]:
                print(f"ERROR: {r['pair']} {c['id']}: {p}")
        if r["status"] == "pass":
            print(f"OK: {r['pair']} — {', '.join(c['id'] for c in r['checks'])} hold")
    print(f"{'OK' if verdict == 'PASS' else 'ERROR'}: adaptation {verdict} ({len(results)} pair(s))")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
