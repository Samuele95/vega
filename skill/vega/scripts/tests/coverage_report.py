#!/usr/bin/env python3
"""
coverage_report.py — the complete-coverage verdict of the verification program (D6 §0, §12.8, §13;
BUILD-PLAN T12): every catalog entry a lane must prove, and every new gate a seeded case must hit.

    coverage_report.py --matrix DIR [--declared-only] [--tier pr|release|nightly]
                       [--json OUT] [--md OUT] [--skills ROOT]

Inputs: the forge `catalog.json` and the backend `coverage.json`, the vega-show `catalog.json`,
`lanes.json`, every `<matrix>/lanes/*/observed.json` (`vega/lane-observed@1`) and every
`<matrix>/negatives/*.json` (`vega/negatives-result@1`). `--skills` is the tree whose catalogs and
registry are read (default: the one this script sits in; the matrix's own copy is the same tree).

An entry is covered iff at least one lane OBSERVED it (its packed or selected netspec, its
evaluation records, its showcase and page results: lanekit.observe) and that lane's stages up to
the dimension's `proved_at` stage (lanekit.contracts.PROVED_AT) all passed. A stage that passed
but said it did not run a check (`stages[].not_run`) proves nothing past itself: a gate that did
not run proves nothing.

| rule | |
|---|---|
| 1 | a must dimension with an uncovered entry → FAIL; each uncovered entry names the lanes that declare it and where each stopped |
| 2 | a report-only dimension (optimizers, schedules, init schemes, metrics) → `WARN:` lines, the verdict unaffected |
| 3 | an entry observed (or declared) that the catalog does not hold → FAIL: the registry and the catalog diverged |
| 4 | `--tier pr`: the lanes of the registry that did not run stand in by declaration; `--tier release` likewise for the entries only lanes outside the tier declare (G00's `lanes` motif, nightly): reported `deferred` with the tier that proves them, never covered |
| 5 | every new gate id of the rule tables is the id of at least one passing seeded case (coverage_gates) |

`--declared-only` (T0, static): `lanes.json` declarations stand in for observations — a fast check
that the registry still spans the catalog after someone adds a family; the gates are not read.

Writes `--json` (`vega/coverage-report@1`: D6 §13's document, plus each lane's verdict and where it
stopped, the warnings and the unreadable inputs) and `--md` (coverage_md: one table per dimension,
the family × task and task × shape grids, the gates). Exit 0 PASS, 1 FAIL, 2 the inputs
could not be read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import coverage_gates  # noqa: E402
import coverage_md  # noqa: E402
from lanekit import contracts, registry  # noqa: E402
from lanekit.stages import stage_plan  # noqa: E402

SCHEMA = "vega/coverage-report@1"
PASSED = ("pass", "stamped")
THEMES = ("night", "day")              # D4: the page's two themes, page_smoke.RUNS drives both
WIDTHS = ("1440", "390")               # D4: desktop and the 390 px mobile viewport
REPORT_PATHS = {"linked": "stub", "brief-links": "none"}   # show.json `report` -> the lane's report path
REPORT_ONLY = ("optimizers", "schedules", "init_schemes", "metrics")
OBSERVED_ONLY = ("metrics",)          # a lane declares no metric: the evaluation computes them
# the evaluation's verdict states, each proved at page (the results view says it, P-TXT); a lane
# declares the one it expects (`expect.verdict`) only where the registry pins it
VERDICTS = ("earned", "not_earned", "beaten_by_baseline", "not_assessable")
# D6 §0: themes and widths, like the other vega-show closed sets, are proved at page
PROVED_AT = {**contracts.PROVED_AT, "themes": "page", "widths": "page", "verdicts": "page"}


# --------------------------------------------------------------------------- the dimensions
def dimension_entries(skills: Path) -> dict[str, list[str]]:
    """{dimension: catalog entries}, in D6 §0's order."""
    read = lambda rel: json.loads((skills / rel).read_text(encoding="utf-8"))  # noqa: E731
    forge, cov, show = read(contracts.FORGE_CATALOG), read(contracts.BACKEND_COVERAGE), read(contracts.SHOW_CATALOG)
    planned = set((cov.get("planned") or {}).get("metrics") or [])
    views = list(show["views"]["ids"]) + [v for vs in show["views"]["sub_views"].values() for v in vs]
    return {"families": list(forge["families"]), "blocks": list(forge["blocks"]), "tasks": list(forge["tasks"]),
            "goals": list(forge["goal_map"]), "sample_shapes": list(forge["sample_shapes"]),
            "graph_levels": ["node", "graph"], "graph_protocols": list(forge["graph_protocols"]),
            "activations": list(forge["activations"]), "losses": list(cov["losses"]),
            "optimizers": list(cov["optimizers"]), "schedules": list(cov["schedules"]),
            "init_schemes": list(cov["init_schemes"]),
            "metrics": [m for m in cov["metrics"] if m not in planned],
            "motifs": list(show["motifs"]["set"]), "class_roles": list(show["roles"]["set"]),
            "views": views, "themes": list(THEMES), "widths": list(WIDTHS),
            "report_paths": sorted(set(REPORT_PATHS.values())), "verdicts": list(VERDICTS)}


def observed_values(o: Mapping[str, Any]) -> dict[str, set[str]]:
    """{dimension: the values one lane's observed.json holds}."""
    graph, show, page = o.get("graph") or {}, o.get("showcase") or {}, o.get("page") or {}
    one = lambda v: {str(v)} if v not in (None, "") else set()  # noqa: E731
    return {"families": one(o.get("family")), "blocks": set(o.get("blocks") or []), "tasks": one(o.get("task")),
            "goals": one(o.get("goal")), "sample_shapes": one(o.get("shape")), "graph_levels": one(graph.get("level")),
            "graph_protocols": one(o.get("protocol") or graph.get("protocol")),
            "activations": set(o.get("activations") or []), "losses": one(o.get("loss")),
            "optimizers": one(o.get("optimizer")), "schedules": one(o.get("schedule")), "init_schemes": one(o.get("init")),
            "metrics": set(o.get("metrics_finite") or []), "motifs": one(show.get("motif")),
            "class_roles": {str(role) for _, role in show.get("class_roles") or []},
            "views": set(page.get("views") or []), "themes": set(page.get("themes") or []),
            "widths": {str(w) for w in page.get("widths") or []},
            "report_paths": one(REPORT_PATHS.get(show.get("report"), show.get("report"))),
            "verdicts": one(o.get("verdict"))}


def declared_values(lane: registry.Lane) -> dict[str, set[str]]:
    """{dimension: what a lane declares}: the stand-in for an observation (--declared-only, rule 4)."""
    blocks = lane.raw.get("blocks") or []
    graph = lane.raw.get("graph") or {}
    acts = {(b.get("params") or {}).get("name") for b in _walk(blocks) if b["type"] == "activation"}
    acts |= {(b.get("params") or {}).get("activation") or "none" for b in _walk(blocks) if b["type"] in ("dense", "head")}
    runs = len((lane.raw.get("campaign") or {}).get("rungs") or []) + int((lane.raw.get("campaign") or {}).get("hpo_trials") or 0)
    page = "page" in stage_plan(lane)
    one = lambda v: {str(v["name"] if isinstance(v, Mapping) else v)} if v else set()  # noqa: E731
    roles = lane.raw.get("roles") or {}
    return {"families": one(lane.family), "blocks": set(lane.flat_block_types()), "tasks": one(lane.task),
            "goals": one(lane.goal), "sample_shapes": one(lane.shape), "graph_levels": one(graph.get("level")),
            "graph_protocols": one(graph.get("protocol")), "activations": {a for a in acts if a},
            "losses": one((lane.raw.get("loss") or {}).get("name")), "optimizers": one(lane.raw.get("optimizer")),
            "schedules": one(lane.raw.get("schedule")), "init_schemes": one(lane.raw.get("init")), "metrics": set(),
            "motifs": one(lane.expect.get("motif")), "class_roles": {str(r) for r in roles.values()},
            "views": set(contracts.expected_views(runs)) if page else set(),
            "themes": set(THEMES) if page else set(), "widths": set(WIDTHS) if page else set(),
            "report_paths": one({"stub": "stub", "real": "stub", "none": "none"}.get(lane.raw.get("report"))),
            "verdicts": one(lane.expect.get("verdict"))}


def _walk(blocks: Sequence[Mapping]) -> Iterable[Mapping]:
    for block in blocks:
        yield block
        yield from _walk((block.get("params") or {}).get("inner") or [])


# --------------------------------------------------------------------------- one lane's proof
def proved_through(doc: Mapping[str, Any]) -> tuple[set[str], str | None]:
    """The stages up to which the lane's run proves (every earlier stage passed and ran all its
    checks), and where it stopped (None: it ran its whole plan)."""
    good: set[str] = set()
    if doc.get("verdict") in ("skip", "refused"):
        return good, f"the lane was {doc['verdict']}: {doc.get('reason') or ''}".rstrip(": ")
    for stage in doc.get("stages") or []:
        if stage["status"] not in PASSED:
            return good, f"{stage['id']} {stage['status']}" + (f": {stage['detail'][:160]}" if stage.get("detail") else "")
        if stage.get("not_run"):
            return good, f"{stage['id']} passed without running: {'; '.join(stage['not_run'])[:200]}"
        good.add(stage["id"])
    return good, None


def stage_reached(good: set[str], proved_at: str) -> bool:
    return proved_at in good


# --------------------------------------------------------------------------- the report
class Report:
    """One coverage verdict over a matrix dir (or the registry alone)."""

    def __init__(self, skills: Path, matrix: Path | None, tier: str | None, declared_only: bool,
                 lanes_file: Path | None = None) -> None:
        self.skills, self.matrix, self.tier, self.declared_only = skills, matrix, tier, declared_only
        self.lanes_file = lanes_file or skills / contracts.TESTS_DIR / "lanes.json"
        self.lanes = registry.load_lanes(self.lanes_file)
        self.entries = dimension_entries(skills)
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self.observed: Mapping[str, dict] = {}

    # ---- the lanes that stand in, and the observations
    def in_tier(self) -> list[registry.Lane]:
        return registry.lanes_for(self.lanes, None, self.tier)

    def observations(self) -> dict[str, dict]:
        """{lane: observed.json} of the tier's lanes the matrix holds."""
        out = {}
        if self.declared_only or self.matrix is None:
            return out
        for lane in self.in_tier():
            path = self.matrix / "lanes" / lane.id / "observed.json"
            if not path.is_file():
                continue
            try:
                out[lane.id] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.problems.append(f"{path}: unreadable ({exc})")
        return out

    # ---- the dimensions
    def dimensions(self, observed: Mapping[str, dict]) -> tuple[dict, list[dict]]:
        self.observed = observed
        dims, uncovered = {}, []
        proofs = {lid: proved_through(doc) for lid, doc in observed.items()}
        values = {lid: observed_values(doc.get("observed") or {}) for lid, doc in observed.items()}
        standing = self._standing_in(observed)
        stood = {l.id: declared_values(l) for l in standing}
        for dim, entries in self.entries.items():
            if self.declared_only and dim in OBSERVED_ONLY:
                continue
            proved_at = PROVED_AT[dim]
            must = dim not in REPORT_ONLY
            table: dict[str, dict] = {}
            for entry in entries:
                table[entry] = self._entry(dim, entry, proved_at, proofs, values, stood)
            for lid, vals in [*values.items(), *stood.items()]:
                for alien in sorted(vals[dim] - set(entries)):
                    uncovered.append({"dimension": dim, "entry": alien,
                                      "why": f"{lid} uses {alien!r}, which the catalog does not hold"})
                    table.setdefault(alien, {"covered": False, "lanes": [lid], "unknown": True})
            dims[dim] = {"must": must, "proved_at": proved_at, "entries": table}
            for entry, cell in table.items():
                if cell["covered"] or cell.get("unknown") or cell.get("deferred"):
                    if cell.get("deferred"):
                        self.warnings.append(f"WARN: {dim}:{entry} is proved by {', '.join(cell['declared_by'])} "
                                             f"in tier {cell['deferred']}, not in {self.tier}")
                    continue
                why = self._why(dim, entry, cell)
                if must:
                    uncovered.append({"dimension": dim, "entry": entry, "why": why})
                else:
                    self.warnings.append(f"WARN: {dim}:{entry} not covered ({why})")
        return dims, uncovered

    def _standing_in(self, observed: Mapping[str, dict]) -> list[registry.Lane]:
        """The lanes whose declarations stand in (rule 4): all of them with --declared-only; on pr,
        the registry's lanes that did not run."""
        if self.declared_only:
            return list(self.lanes.values()) if self.tier in (None, "pr") else self.in_tier()
        if self.tier == "pr":
            return [l for l in self.lanes.values() if l.id not in observed and "pr" not in l.tiers]
        return []

    def _entry(self, dim: str, entry: str, proved_at: str, proofs: Mapping, values: Mapping, stood: Mapping) -> dict:
        lanes = sorted(lid for lid, vals in values.items() if entry in vals[dim])
        proving = [lid for lid in lanes if stage_reached(proofs[lid][0], proved_at)]
        standing = sorted(lid for lid, vals in stood.items() if entry in vals[dim]
                          and proved_at in stage_plan(self.lanes[lid]))
        cell: dict = {"covered": bool(proving or standing), "lanes": proving or lanes}
        if standing and not proving:
            cell["declared"] = standing
        stopped = {lid: proofs[lid][1] or f"its plan ends before {proved_at}"
                   for lid in lanes if lid not in proving}
        if stopped:
            cell["stopped"] = stopped
        if not cell["covered"]:
            outside = self._declared_outside_tier(dim, entry, proved_at)
            if outside:
                cell.update(deferred=outside[0], declared_by=outside[1])
        return cell

    def _declared_outside_tier(self, dim: str, entry: str, proved_at: str) -> tuple[str, list[str]] | None:
        """(the tier that proves it, its lanes) when only lanes outside this tier declare `entry`."""
        if self.tier is None:
            return None
        decl = [l for l in self.lanes.values() if entry in declared_values(l)[dim] and proved_at in stage_plan(l)]
        if not decl or any(self.tier in l.tiers for l in decl):
            return None
        later = next((t for t in registry.TIERS[registry.TIERS.index(self.tier) + 1:]
                      if any(t in l.tiers for l in decl)), None)
        return (later, sorted(l.id for l in decl)) if later else None

    def _why(self, dim: str, entry: str, cell: Mapping) -> str:
        if dim in OBSERVED_ONLY and not cell.get("stopped"):
            return "no lane's evaluation recorded a finite value of it"
        if cell.get("stopped"):
            return "; ".join(f"{lid} stopped at {where}" for lid, where in cell["stopped"].items())
        decl = sorted(l.id for l in self.in_tier() if entry in declared_values(l)[dim])
        ran = [lid for lid in decl if lid in self.observed]
        if ran:
            return (f"declared by {', '.join(decl)}; the observed.json of {', '.join(ran)} does not hold it "
                    f"(observed {dim}: {sorted(observed_values(self.observed[ran[0]].get('observed') or {})[dim]) or 'none'})")
        if decl:
            return f"declared by {', '.join(decl)}, which left no observed.json in this matrix"
        return "no lane of the registry declares it"

    # ---- the whole
    def build(self) -> dict:
        observed = self.observations()
        if not self.declared_only and self.matrix is not None and not observed:
            self.problems.append(f"no observed.json of the tier's lanes under {self.matrix}/lanes")
        dims, uncovered = self.dimensions(observed)
        gates: dict = {}
        if not self.declared_only:
            cases = coverage_gates.read_negatives(self.matrix / "negatives")
            self.problems += cases.problems
            gates, missing, warns = coverage_gates.cover(coverage_gates.gate_tables(self.skills), cases)
            uncovered += missing
            self.warnings += warns
        verdict = "PASS" if not uncovered and not self.problems else "FAIL"
        return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "git": git_state(Path(os.environ.get("VEGA_SOURCE_REPO") or self.skills)), "tier": self.tier, "declared_only": self.declared_only,
                "sources": self.sources(), "lanes": {lid: {"verdict": d.get("verdict"),
                                                         "stopped": proved_through(d)[1]} for lid, d in observed.items()},
                "dimensions": dims, "gates": gates, "cross": cross(observed, self.skills),
                "uncovered": uncovered, "warnings": self.warnings, "problems": self.problems, "verdict": verdict}

    def sources(self) -> dict:
        def src(rel: str | Path) -> dict:
            path = self.skills / rel if not Path(rel).is_absolute() else Path(rel)
            return {"path": str(rel), "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
        return {"catalog": src(contracts.FORGE_CATALOG), "coverage": src(contracts.BACKEND_COVERAGE),
                "show_catalog": src(contracts.SHOW_CATALOG), "lanes": src(self.lanes_file)}


def cross(observed: Mapping[str, dict], skills: Path) -> dict:
    """The family × task and task × shape grids: the combinations a lane proved at page, and the
    admission table's cells (catalog.json `admission`, built by the T1 matrix) — admitted, not shown."""
    fam_task: dict[str, set] = {}
    task_shape: dict[str, set] = {}
    for doc in observed.values():
        o = doc.get("observed") or {}
        if not stage_reached(proved_through(doc)[0], "page") or not o.get("family"):
            continue
        fam_task.setdefault(o["family"], set()).add(o.get("task"))
        task_shape.setdefault(o.get("task"), set()).add(o.get("shape"))
    forge = json.loads((skills / contracts.FORGE_CATALOG).read_text(encoding="utf-8"))
    admitted: dict[str, set] = {}
    for row in (forge.get("admission") or {}).get("rows") or []:
        for fam in row.get("families") or []:
            admitted.setdefault(fam, set()).update(t.split("/")[0] for t in row.get("tasks") or [])
    as_lists = lambda d: {k: sorted(v) for k, v in sorted(d.items())}  # noqa: E731
    return {"family_x_task": as_lists(fam_task), "task_x_shape": as_lists(task_shape),
            "admitted_family_x_task": as_lists(admitted)}


def git_state(skills: Path) -> dict:
    """The commit the report speaks for: the repository the matrix's skills tree was copied from
    ($VEGA_SOURCE_REPO, which run_matrix.py exports), else the tree itself."""
    try:
        commit = subprocess.run(["git", "-C", str(skills), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(skills), "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": False}


# --------------------------------------------------------------------------- the CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--matrix", help="the matrix dir (lanes/*/observed.json, negatives/*.json)")
    ap.add_argument("--declared-only", action="store_true", help="lanes.json declarations stand in (T0)")
    ap.add_argument("--tier", choices=registry.TIERS)
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--md", dest="md_out")
    ap.add_argument("--skills", help="the tree whose catalogs and lanes.json are read (default: this one)")
    ap.add_argument("--lanes-file", help="a lanes.json other than the tree's")
    a = ap.parse_args(argv)
    if not a.declared_only and not a.matrix:
        ap.error("--matrix is required unless --declared-only")
    skills = Path(a.skills).resolve() if a.skills else HERE.parents[2]
    matrix = Path(a.matrix).resolve() if a.matrix else None
    try:
        report = Report(skills, matrix, a.tier, a.declared_only, Path(a.lanes_file) if a.lanes_file else None)
        doc = report.build()
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: the coverage inputs could not be read: {type(exc).__name__}: {exc}")
        return 2
    for path, text in ((a.json_out, json.dumps(doc, indent=2, ensure_ascii=False) + "\n"),
                       (a.md_out, coverage_md.render(doc))):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
    return say(doc)


def say(doc: Mapping[str, Any]) -> int:
    for line in doc["warnings"]:
        print(line)
    for line in doc["problems"]:
        print(f"ERROR: {line}")
    for miss in doc["uncovered"]:
        print(f"ERROR: uncovered {miss['dimension']}:{miss['entry']} — {miss['why']}")
    n = sum(len(d["entries"]) for d in doc["dimensions"].values())
    covered = sum(1 for d in doc["dimensions"].values() for c in d["entries"].values() if c["covered"])
    gates = [g for t in doc["gates"].values() for g in t.values() if g["new"]]
    mode = "declared" if doc["declared_only"] else "observed"
    print(f"{'OK' if doc['verdict'] == 'PASS' else 'ERROR'}: coverage {doc['verdict']} ({mode}, tier "
          f"{doc['tier'] or 'all'}): {covered}/{n} entries covered"
          + (f", {sum(1 for g in gates if g['covered'])}/{len(gates)} new gates hit" if gates else ""))
    return 0 if doc["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
