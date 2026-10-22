"""
coverage_gates — the gate half of the coverage report (D6 §12.8, §13; BUILD-PLAN T12): the rule
tables every new gate id is read from, the seeded-negative results that cover them.

    tables = gate_tables(skills)                     # [GateTable], in report order
    cases = read_negatives(matrix / "negatives")     # Cases: the passing cases, the problems
    gates, uncovered = cover(tables, cases)          # {gate: {...}}, [{dimension, entry, why}]

A gate id is covered when at least one PASSING case of the suites its table names carries that id,
or the id with a trailing letter (case `C11a` covers gate `C11`; case `C11a` never covers `C1`).
A vega-show gate is also covered by any passing case its catalog row names (`gates[].cases`, from
whichever suite ran it: the slot's `SS*` run in /vega-pack's showcase suite).

New and grandfathered gates. Every vega-show, shaper-graph, slot and pack-graph gate is new in
v2.3. Of the compiler's `RULES` and the conformance `CHECKS`, the ids the v2.2 baseline (c849f8b)
already had are pre-v2.3: their cases predate the id convention and are listed in
`negatives/grandfathered.json` under `<skill>/<script>#<gate>` (a family key `T1` stands for
`T1a`…`T1d`, reported as such). A pre-v2.3 gate with no passing case is a `WARN:` line; a new gate
with none is uncovered, and the verdict FAILs (D6 §12.8: a gate that never fires is
indistinguishable from no gate).

The tables:

| table | ids from | suites whose cases count |
|---|---|---|
| compile_spec | `vega-forge/scripts/compile_spec.py` `Rule("<id>", …)` first arguments (AST) | forge `run_negatives.py`, `run_negatives_shapes.py` |
| conformance | the `c<n>_` check functions of `vega-forge/scripts/conformance*.py` and the module docstring's `C<n>` list | forge `run_negatives.py` |
| shaper_graph | `lanekit.contracts.SHAPER_GRAPH_GATES` | dataset-shaper `run_negatives_graph.py` |
| slot | `lanekit.contracts.SLOT_GATES` (D3's aliases `SLOT_GATE_ALIASES` count as their gate) | vega-pack `run_negatives_showcase.py` |
| pack_graph | `lanekit.contracts.PACK_GRAPH_GATES` | vega-pack `run_negatives.py` |
| vega_show | `vega-show/scripts/catalog.json` `gates[].id` | every vega-show suite (by id); any suite (by the row's `cases`) |
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
NEGATIVES_DIR = HERE / "negatives"

from lanekit import contracts  # noqa: E402

# The v2.2 baseline's rule ids (c849f8b: `git show c849f8b:vega-forge/scripts/compile_spec.py`,
# `…/conformance.py`): the gates whose cases grandfathered.json lists without an id convention.
COMPILE_SPEC_BASELINE = frozenset({"P0", "P1", "B1", "B2", "T1a", "T1b", "T1c", "T1d", "S1", "D2a", "D2b",
                                   "G1a", "G1b", "G1c", "G1d", "W1", "R1", "D1a", "D1b", "D1c", "D1d",
                                   "K1", "M2"})
CONFORMANCE_BASELINE = frozenset(f"C{i}" for i in range(1, 11))
# a case id at the start of a case name: `W2a`, `N-S5b`, `SG4c`, `D5-21`
CASE_ID = re.compile(r"^((?:N-)?[A-Z]{1,3}\d+(?:-\d+)?[a-z]?)(?=\s)")
FAMILY = re.compile(r"^(.*\d)[a-z]$")


@dataclass(frozen=True)
class GateTable:
    """One rule table: its gate ids, the suites whose cases count, and which ids predate v2.3."""
    name: str
    source: str
    ids: tuple[str, ...]
    suites: tuple[str, ...]                        # skills-relative scripts
    baseline: frozenset = frozenset()
    grandfathered_key: str | None = None           # `<skill>/<script>` in grandfathered.json
    by_cases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)   # gate -> case ids
    aliases: Mapping[str, str] = field(default_factory=dict)                # case id -> gate


@dataclass(frozen=True)
class Case:
    id: str | None
    name: str
    suite: str


@dataclass
class Cases:
    """The passing cases of every `vega/negatives-result@1` file, and what could not be read."""
    passing: list[Case] = field(default_factory=list)
    failing: list[Case] = field(default_factory=list)
    suites: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- the tables
def rule_ids(path: Path) -> tuple[str, ...]:
    """The `Rule("<id>", …)` first arguments of compile_spec.py, by AST, in order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Rule" and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            out.append(node.args[0].value)
    return tuple(dict.fromkeys(out))


def check_ids(scripts: Path) -> tuple[str, ...]:
    """C<n> from the `c<n>_` check functions of conformance*.py and conformance.py's docstring list."""
    found: set[int] = set()
    for path in sorted(scripts.glob("conformance*.py")):
        text = path.read_text(encoding="utf-8")
        found.update(int(n) for n in re.findall(r"^def c(\d+)_", text, re.M))
        if path.name == "conformance.py":
            doc = ast.get_docstring(ast.parse(text)) or ""
            found.update(int(n) for n in re.findall(r"^\s*C(\d+)\s{2}", doc, re.M))
    return tuple(f"C{n}" for n in sorted(found))


def gate_tables(skills: Path) -> list[GateTable]:
    forge = skills / "vega-forge" / "scripts"
    forge_neg = "vega-forge/scripts/tests/run_negatives.py"
    show = json.loads((skills / contracts.SHOW_CATALOG).read_text(encoding="utf-8"))
    show_rows = show.get("gates") or []
    show_suites = tuple(sorted(str(p.relative_to(skills)) for p in
                               (skills / "vega-show" / "scripts" / "tests").glob("run_negatives*.py")))
    return [
        GateTable("compile_spec", "vega-forge/scripts/compile_spec.py RULES", rule_ids(forge / "compile_spec.py"),
                  (forge_neg, "vega-forge/scripts/tests/run_negatives_shapes.py"), COMPILE_SPEC_BASELINE,
                  "vega-forge/compile_spec.py"),
        GateTable("conformance", "vega-forge/scripts/conformance*.py CHECKS", check_ids(forge), (forge_neg,),
                  CONFORMANCE_BASELINE, "vega-forge/conformance.py"),
        GateTable("shaper_graph", "lanekit.contracts.SHAPER_GRAPH_GATES", tuple(contracts.SHAPER_GRAPH_GATES),
                  ("dataset-shaper/scripts/tests/run_negatives_graph.py",)),
        GateTable("slot", "lanekit.contracts.SLOT_GATES", tuple(contracts.SLOT_GATES),
                  ("vega-pack/scripts/tests/run_negatives_showcase.py",),
                  aliases=dict(contracts.SLOT_GATE_ALIASES)),
        GateTable("pack_graph", "lanekit.contracts.PACK_GRAPH_GATES", tuple(contracts.PACK_GRAPH_GATES),
                  ("vega-pack/scripts/tests/run_negatives.py",)),
        GateTable("vega_show", contracts.SHOW_CATALOG + " gates[].id", tuple(r["id"] for r in show_rows),
                  show_suites or ("vega-show/scripts/tests/run_negatives.py",),
                  by_cases={r["id"]: tuple(r.get("cases") or ()) for r in show_rows}),
    ]


# --------------------------------------------------------------------------- the results
def case_id(case: Mapping) -> str | None:
    if case.get("id"):
        return str(case["id"])
    m = CASE_ID.match(str(case.get("name") or ""))
    return m.group(1) if m else None


def read_negatives(directory: Path) -> Cases:
    """Every `*.json` under `directory` that is a `vega/negatives-result@1` document."""
    import sys
    if str(NEGATIVES_DIR) not in sys.path:
        sys.path.insert(0, str(NEGATIVES_DIR))
    import negkit
    out = Cases()
    for path in sorted(Path(directory).glob("*.json")) if Path(directory).is_dir() else []:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            out.problems.append(f"{path.name}: unreadable ({exc})")
            continue
        wrong = negkit.problems(doc)
        if wrong:
            out.problems.append(f"{path.name}: not {negkit.SCHEMA}: {'; '.join(wrong[:3])}")
            continue
        suite = doc["script"]
        out.suites.add(suite)
        out.files.append(path.name)
        for case in doc["cases"]:
            (out.passing if case["ok"] else out.failing).append(Case(case_id(case), case["name"], suite))
    return out


def grandfathered(path: Path = NEGATIVES_DIR / "grandfathered.json") -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# --------------------------------------------------------------------------- matching
def _matches(case_id_: str | None, gate: str) -> bool:
    """`C11a` and `C11` cover `C11`; `C11a` does not cover `C1`."""
    if not case_id_:
        return False
    m = FAMILY.match(case_id_)
    return case_id_ == gate or (m is not None and m.group(1) == gate)


def _family(gate: str, key: str) -> bool:
    """A grandfathered family key `T1` stands for `T1a`…`T1d`."""
    m = FAMILY.match(gate)
    return key == gate or (m is not None and m.group(1) == key)


def table_cover(table: GateTable, cases: Cases, legacy: Mapping[str, Sequence[str]]) -> dict[str, dict]:
    """{gate: {covered, cases, new, grandfathered?}} for one table."""
    names_of = {key.split("#", 1)[1]: set(names) for key, names in legacy.items()
                if table.grandfathered_key and key.startswith(table.grandfathered_key + "#")}
    claimed = {n for key, names in legacy.items() if "#" in key for n in names}
    mine = [c for c in cases.passing if c.suite in table.suites]
    out = {}
    for gate in table.ids:
        new = gate not in table.baseline
        hits = [c.id or c.name for c in mine if c.name not in claimed and
                (_matches(c.id, gate) or table.aliases.get(c.id or "") == gate)]
        hits += [c.id or c.name for c in cases.passing if c.id and c.id in table.by_cases.get(gate, ())]
        entry: dict = {"covered": bool(hits), "cases": sorted(set(hits)), "new": new}
        if not new:
            family = sorted(k for k in names_of if _family(gate, k))
            passed = sorted({c.name for c in mine if any(c.name in names_of[k] for k in family)})
            entry["grandfathered"] = {"keys": family, "cases": passed}
            entry["covered"] = entry["covered"] or bool(passed)
        out[gate] = entry
    return out


def cover(tables: Iterable[GateTable], cases: Cases,
          legacy: Mapping[str, Sequence[str]] | None = None) -> tuple[dict, list[dict], list[str]]:
    """({table: {gate: entry}}, uncovered new gates, WARN lines for pre-v2.3 gates without a case)."""
    legacy = grandfathered() if legacy is None else legacy
    gates, uncovered, warns = {}, [], []
    for table in tables:
        entries = table_cover(table, cases, legacy)
        gates[table.name] = entries
        missing_suites = [s for s in table.suites if s not in cases.suites]
        for gate, entry in entries.items():
            if entry["covered"]:
                continue
            if not entry["new"]:
                warns.append(f"WARN: gate {table.name}:{gate} predates v2.3 and no passing case covers it")
                continue
            why = (f"no result from {', '.join(missing_suites)}" if missing_suites else
                   f"no passing case of {', '.join(table.suites)} carries the id {gate}"
                   + (f" or one of {', '.join(table.by_cases[gate])}" if table.by_cases.get(gate) else ""))
            uncovered.append({"dimension": f"gates:{table.name}", "entry": gate, "why": why})
    return gates, uncovered, warns
