#!/usr/bin/env python3
"""
run_negatives.py — every gate of /vega-show refuses what it says it refuses (D5 §12.2, D6 §12.5,
D4 §17.7; BUILD-PLAN R-H6, H09; show-contract §10).

    python3 run_negatives.py <matrix-dir> [-v] [--json <path>] [--only GATE,...] [--keep DIR]

**One table, de-duplicated by gate id.** D5's forty seeded defects, D6's VS01–VS27, the slot's
SS1–SS9, D4's PN1–PN12 and gate C's GC-1–GC-4 are merged into one case table keyed by the gate
that must catch each seed: a case's name starts with its gate id (the vega/negatives-result@1 convention), then the
design cases it realises. `catalog.json.gates` is the de-duplication: every gate id of the
catalog is the id of at least one case here, and every design case a gate lists is realised by
a case of that gate (a gate the designs leave without a case has one of its own). Fork S1's
cases (VS16, VS17) carry the fork id; the two design cases caught outside vega-show carry the
gate that catches them (D5-38 the vega-forge catalog sync, D5-40 conformance's C11). Before
anything runs, the table is held to the catalog and a gap is a failing case.

**The inputs** are the matrix's lanes, staged once each through the stage itself (extract →
draft → author with `autoauthor.py` → resolve → build → check, and the served walk on L02) and
put back at their path before every case: L02 (multiclass, a linked stub report), L03 (no
report: brief links) and L05 (regression). A lane the matrix does not hold leaves its cases not
run (said, never passed). The template, catalog and conformance cases seed copies of the
skill's own files and need no lane. A browser (`$CHROME`) is needed for the served cases (SV*,
PG10, PG12) and the style lock (PG8); `node` for PG11.

| group | gates | how a case seeds and what it runs |
|---|---|---|
| validate | V1–V15 | a copy of the authored plan or the fact set, then `resolve_story.py`: exit 2, `ERROR: V<n>` and no other rule |
| fork S1 | S1 | L03 built without `--unattended` (`FORK:`, nothing written) and with it (exit 0, `WARN:`, the code notes link to pinned files) |
| preflight | PF1–PF9 | the bundle, its MANIFEST (and the page's bundle layer, alike) or the report edited, then `extract_facts.py` or `build_showcase.py` |
| build | BD1–BD8 | a seeded template, a locale, a staged file, or the build's own code path seeded in a child (`seeded_build`) |
| check | CK1–CK10 | the built showcase, the bundle, the report or the briefs edited, then `build_showcase.py --check` (2 broken, 3 stale, CK10 a WARN) |
| slot | SL1–SL11 | the showcase seeded, then the bundle's own `serve/routes_showcase.py` opened on it (SL11: the bundle's app) |
| layer | LY1–LY7 | the layer's authoring data, the recorded check or walk seeded, then `apply_show_layer.py` |
| page | PG1–PG12 | a copy of the page template seeded, then the gate's own check; PG10 and PG12 in a browser |
| served | SV1–SV5 | the showcase seeded and re-pinned (so the slot verifies it), then `smoke_show.py --serve --strict` |
| catalog | CS1–CS3 | a seeded copy of the contract, a schema or the document-project patch, then the sync's checks |

Exit 0 when every case that ran was caught by its gate; 1 otherwise; `--json` writes
`vega/negatives-result@1`. `--only` narrows to gate ids (the table check still runs);
`--keep DIR` stages there and leaves the work behind.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent), str(SKILLS / "vega" / "scripts" / "tests" / "negatives")]

import negkit  # noqa: E402
from _negatives_kit import CATALOG, Check, Expect, StageError, Staged  # noqa: E402
import _negatives_bundle  # noqa: E402
import _negatives_plan  # noqa: E402
import _negatives_served  # noqa: E402
import _negatives_static  # noqa: E402

STAGED = {"L02": True, "L03": False, "L05": False}      # lane: walked once staged
MODULES = (_negatives_plan, _negatives_bundle, _negatives_served, _negatives_static)


def table() -> list:
    return [c for m in MODULES for c in m.CASES]


def table_gaps(cases) -> list[str]:
    """What the case table misses of the catalog: a gate without a case, a design case a gate
    lists that no case of that gate realises, an id that is no gate."""
    known = {g["id"] for g in CATALOG["gates"]} | set(CATALOG["forks"]) | {"C11"}
    by_gate: dict = {}
    for c in cases:
        by_gate.setdefault(c.gate, set()).update(c.ids)
    gaps = [f"{c.gate} is no gate of the catalog ({c.name})" for c in cases if c.gate and c.gate not in known]
    wanted = [(g["id"], g["cases"]) for g in CATALOG["gates"]]
    wanted += [(f, row.get("cases") or []) for f, row in CATALOG["forks"].items() if row.get("cases")]
    for gate, ids in wanted:
        if gate not in by_gate:
            gaps.append(f"the gate {gate} has no case")
        gaps += [f"{gate}: the design case {i} is realised by no case of the gate" for i in ids
                 if i not in by_gate.get(gate, ())]
    realised = {i for ids in by_gate.values() for i in ids}
    gaps += [f"the external case {i} is realised by no case" for i in CATALOG.get("external_cases", {})
             if i not in realised]
    return gaps


def stage_lanes(matrix: Path, tmp: Path, needed: set, say=print) -> dict:
    staged = {}
    for lane_id, walk in STAGED.items():
        if lane_id not in needed:
            continue
        lane = negkit.lane(matrix, lane_id)
        if lane is None:
            staged[lane_id] = f"lane {lane_id} is not in this matrix"
            continue
        try:
            staged[lane_id] = Staged.of(lane, tmp, walk=walk, say=say)
        except (StageError, StopIteration, OSError) as exc:
            staged[lane_id] = f"lane {lane_id} could not be staged: {exc}"
            say(f"WARN: {staged[lane_id]}")
    return staged


def run_case(cases: negkit.Cases, c, staged: dict, tmp: Path) -> None:
    if not c.gate:                         # caught outside vega-show: recorded under that suite
        cases.gate = CATALOG["external_cases"][c.ids[0]]
    where = staged.get(c.lane) if c.lane else None
    if c.lane and not isinstance(where, Staged):
        cases.not_run(c.name, where or f"lane {c.lane} was not staged")
        return
    try:
        if c.lane:
            with where.fresh() as w:
                verdict = c.fn(w)
        else:
            scratch = Path(tempfile.mkdtemp(prefix=f"{c.gate}-", dir=tmp))
            verdict = c.fn(scratch)
    except Exception as exc:  # noqa: BLE001 — a case that cannot seed is a failed case, said
        cases.check(c.name, False, f"the case could not run: {type(exc).__name__}: {exc}")
        return
    if isinstance(verdict, Expect):
        cases.expect(c.name, verdict.rc, verdict.out, code=verdict.code, contains=verdict.contains,
                     absent=verdict.absent)
    else:
        cases.check(c.name, verdict.ok, verdict.detail, verdict.out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("matrix", help="the matrix dir (lanes/<id>/fixture.json and the lanes' work)")
    ap.add_argument("-v", dest="verbose", action="store_true")
    ap.add_argument("--json", default=None, metavar="PATH")
    ap.add_argument("--only", default=None, help="comma-separated gate ids")
    ap.add_argument("--keep", default=None, metavar="DIR", help="stage in DIR and keep it")
    a = ap.parse_args(argv)
    matrix = Path(a.matrix).resolve()
    cases = negkit.Cases(__file__, matrix, a.verbose)
    every = table()
    cases.section("the case table — every catalog gate and design case has a case", gate="vega-show/catalog.json")
    gaps = table_gaps(every)
    cases.check("the case table covers catalog.json's gates and their design cases", not gaps, "; ".join(gaps))
    chosen = [c for c in every if not a.only or c.gate in a.only.split(",")]
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="vega-show-negatives-"))
    tmp.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        staged = stage_lanes(matrix, tmp, {c.lane for c in chosen if c.lane})
        group = None
        for c in chosen:
            if c.fn.__module__ != group:
                group = c.fn.__module__
                cases.section(f"{group.lstrip('_')} — {sys.modules[group].__doc__.strip().splitlines()[0]}")
            run_case(cases, c, staged, tmp)
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"({time.monotonic() - started:.0f} s)")
    return cases.finish(a.json)


if __name__ == "__main__":
    sys.exit(main())
