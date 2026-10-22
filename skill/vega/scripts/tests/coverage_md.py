"""
coverage_md — `coverage.md`, the human reading of a `vega/coverage-report@1` document (D6 §13):
the verdict and what is uncovered first, then one table per dimension, the family × task and task ×
shape grids (proved at page; the admission table's cells marked "admitted", built by the T1
matrix, not shown), and the gate tables.

    text = render(doc)
"""
from __future__ import annotations

from typing import Any, Mapping


def _cell(cell: Mapping[str, Any]) -> tuple[str, str]:
    if cell.get("covered"):
        how = ", ".join(cell.get("lanes") or cell.get("declared") or [])
        return "yes", how + (" (declared)" if cell.get("declared") and not cell.get("stopped") and not cell.get("lanes") else "")
    if cell.get("deferred"):
        return f"deferred to {cell['deferred']}", ", ".join(cell.get("declared_by") or [])
    if cell.get("unknown"):
        return "NOT IN CATALOG", ", ".join(cell.get("lanes") or [])
    stopped = cell.get("stopped") or {}
    return "**no**", "; ".join(f"{lid}: {where}" for lid, where in stopped.items()) or "no lane"


def render(doc: Mapping[str, Any]) -> str:
    mode = "declared (lanes.json stands in)" if doc.get("declared_only") else "observed"
    out = [f"# Coverage — {doc['verdict']}", "",
           f"Tier {doc.get('tier') or 'all'}, {mode}; generated {doc['generated_at']} at "
           f"{(doc.get('git') or {}).get('commit')}{' (dirty)' if (doc.get('git') or {}).get('dirty') else ''}.", ""]
    if doc.get("problems"):
        out += ["## Inputs that could not be read", ""] + [f"- {p}" for p in doc["problems"]] + [""]
    if doc.get("uncovered"):
        out += ["## Uncovered", "", "| dimension | entry | why |", "|---|---|---|"]
        out += [f"| {u['dimension']} | {u['entry']} | {u['why']} |" for u in doc["uncovered"]] + [""]
    if doc.get("warnings"):
        out += ["## Warnings", ""] + [f"- {w}" for w in doc["warnings"]] + [""]
    if doc.get("lanes"):
        out += ["## Lanes", "", "| lane | verdict | stopped |", "|---|---|---|"]
        out += [f"| {lid} | {l.get('verdict')} | {l.get('stopped') or '—'} |" for lid, l in sorted(doc["lanes"].items())]
        out.append("")
    out += ["## Dimensions", ""]
    for dim, d in doc["dimensions"].items():
        n = len(d["entries"])
        got = sum(1 for c in d["entries"].values() if c["covered"])
        out += [f"### {dim} — {got}/{n} ({'must' if d['must'] else 'report only'}, proved at `{d['proved_at']}`)", "",
                "| entry | covered | lanes / where they stopped |", "|---|---|---|"]
        out += [f"| {entry} | {' | '.join(_cell(cell))} |" for entry, cell in d["entries"].items()] + [""]
    out += _grids(doc.get("cross") or {})
    out += _gates(doc.get("gates") or {})
    return "\n".join(out) + "\n"


def _grids(cross: Mapping[str, Any]) -> list[str]:
    out = []
    fam, admitted = cross.get("family_x_task") or {}, cross.get("admitted_family_x_task") or {}
    tasks = sorted({t for v in [*fam.values(), *admitted.values()] for t in v})
    if tasks:
        out += ["## Family × task (proved at page: ●; admitted, built by the T1 matrix, not shown: ○)", "",
                "| family | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
        for family in sorted(set(fam) | set(admitted)):
            marks = ["●" if t in fam.get(family, []) else ("○" if t in admitted.get(family, []) else "")
                     for t in tasks]
            out.append(f"| {family} | " + " | ".join(marks) + " |")
        out.append("")
    ts = cross.get("task_x_shape") or {}
    shapes = sorted({s for v in ts.values() for s in v if s})
    if shapes:
        out += ["## Task × shape (proved at page)", "", "| task | " + " | ".join(shapes) + " |",
                "|---|" + "---|" * len(shapes)]
        out += [f"| {task} | " + " | ".join("●" if s in v else "" for s in shapes) + " |" for task, v in ts.items()]
        out.append("")
    return out


def _gates(gates: Mapping[str, Any]) -> list[str]:
    if not gates:
        return []
    out = ["## Gates (D6 §12.8)", ""]
    for table, entries in gates.items():
        new = [g for g in entries.values() if g["new"]]
        out += [f"### {table} — {sum(1 for g in new if g['covered'])}/{len(new)} new gates hit, "
                f"{len(entries) - len(new)} pre-v2.3", "", "| gate | new | covered | cases |", "|---|---|---|---|"]
        for gate, g in entries.items():
            cases = list(g["cases"])
            legacy = (g.get("grandfathered") or {})
            if legacy.get("cases"):
                cases.append(f"grandfathered {', '.join(legacy['keys'])}: {len(legacy['cases'])} case(s)")
            out.append(f"| {gate} | {'yes' if g['new'] else 'pre-v2.3'} | {'yes' if g['covered'] else '**no**'} | "
                       f"{'; '.join(cases) or '—'} |")
        out.append("")
    return out
