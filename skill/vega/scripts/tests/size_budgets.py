#!/usr/bin/env python3
"""
size_budgets.py — the lanes' time budgets from measured runs (D6 §16.5; BUILD-PLAN T12, risk 11;
PLAN-AMENDMENTS A44(a), A47(6)): each lane's `budget_s` is 1.5 × its measured wall time, rounded up
to 30 s, and each stage share is wide enough that every lane's stage limit holds 1.5 × that stage's
measured time.

    size_budgets.py --timing T [--timing T …] [--lanes-file F] [--write] [--factor 1.5] [--round 30]

T is a `timing.json` (`vega/matrix-timing@1`, written by run_matrix.py) of a lanes run. Only a
lane that PASSED counts (a lane cut short measures nothing about its whole), and only a run that
ran every stage measures its wall time (a `--from` run's stamped stages took no time); over several
runs the largest wall time per lane and per stage is taken, so a run under load sizes for load.

The limits a lane runs under (lanekit.lane): a stage `stage_timeout_factor × budget_s × share`,
the lane `lane_kill_factor × budget_s`. So the rules are:

    budget_s(lane)   = ceil_30(factor × wall(lane))
    share(stage)     = max(current share, max over the lanes of its group of
                           factor × stage_s(lane) / (stage_timeout_factor × budget_s(lane)))

rounded up to 0.01, per share group (`stage_shares` for lanekit lanes, `acceptance_shares` for L01,
`golden_shares` for G00). A share never shrinks here: a stage this run measured fast keeps the
room an earlier one needed. The bundle_api stage (`predict.py --check` and the bundle's own tests)
is the one A44(a) found thin: its share comes out of the slowest suite measured, not a guess.

A lane no timing measured (G00 without `CIDS_ROOT`) keeps its budget and is listed. Without
`--write` the result is printed; with it lanes.json is rewritten (indent 2, the file's own form).
Exit 0, or 2 when a timing file cannot be read.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
GROUPS = {"lanekit": "stage_shares", "acceptance": "acceptance_shares", "golden": "golden_shares"}


def ceil_to(value: float, step: int) -> int:
    return int(math.ceil(value / step) * step)


def measured(timings: Sequence[Mapping]) -> dict[str, dict]:
    """{lane: {"wall": s, "stages": {stage: s}, "runs": n}} over the passing lanes of every run."""
    out: dict[str, dict] = {}
    for doc in timings:
        for row in doc.get("lanes") or []:
            if row.get("verdict") != "pass":
                continue
            cell = out.setdefault(row["lane"], {"wall": 0.0, "stages": {}, "runs": 0})
            if not any(st.get("status") == "stamped" for st in row.get("stages") or []):
                cell["wall"] = max(cell["wall"], float(row["wall_s"]))
                cell["runs"] += 1
            for st in row.get("stages") or []:
                if st.get("status") == "pass":
                    cell["stages"][st["id"]] = max(cell["stages"].get(st["id"], 0.0), float(st["wall_s"]))
    return out


def size(doc: Mapping, seen: Mapping[str, dict], factor: float = 1.5, step: int = 30) -> tuple[dict, dict, list[str]]:
    """({lane: budget_s}, {group key: {stage: share}}, notes)."""
    defaults = doc["defaults"]
    timeout = float(defaults["stage_timeout_factor"])
    budgets, notes = {}, []
    for lane in doc["lanes"]:
        if seen.get(lane["id"], {}).get("runs"):
            budgets[lane["id"]] = max(30, ceil_to(factor * seen[lane["id"]]["wall"], step))
        else:
            budgets[lane["id"]] = lane["budget_s"]
            notes.append(f"{lane['id']}: not measured (no passing run in the timing files); keeps {lane['budget_s']} s")
    shares = {key: dict(defaults[key]) for key in GROUPS.values()}
    for lane in doc["lanes"]:
        cell = seen.get(lane["id"])
        if not cell or not cell["runs"]:
            continue
        group = shares[GROUPS[lane.get("driver", "lanekit")]]
        for stage, secs in cell["stages"].items():
            need = math.ceil(100 * factor * secs / (timeout * budgets[lane["id"]])) / 100
            if need > group.get(stage, 0.0):
                notes.append(f"{GROUPS[lane.get('driver', 'lanekit')]}.{stage}: {group.get(stage)} -> {need} "
                             f"({lane['id']} measured {secs:.0f} s of a {budgets[lane['id']]} s budget)")
                group[stage] = min(1.0, need)
    return budgets, shares, notes


def apply(doc: dict, budgets: Mapping[str, int], shares: Mapping[str, dict]) -> dict:
    for lane in doc["lanes"]:
        lane["budget_s"] = budgets[lane["id"]]
    for key, table in shares.items():
        doc["defaults"][key] = table
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--timing", action="append", required=True, help="a lanes run's timing.json (repeatable)")
    ap.add_argument("--lanes-file", default=str(HERE / "lanes.json"))
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--round", dest="step", type=int, default=30)
    ap.add_argument("--write", action="store_true", help="rewrite --lanes-file with the sized budgets")
    a = ap.parse_args(argv)
    try:
        timings = [json.loads(Path(t).read_text(encoding="utf-8")) for t in a.timing]
    except (OSError, ValueError) as exc:
        print(f"ERROR: a timing file cannot be read: {exc}")
        return 2
    path = Path(a.lanes_file)
    doc = json.loads(path.read_text(encoding="utf-8"))
    seen = measured(timings)
    budgets, shares, notes = size(doc, seen, a.factor, a.step)
    for lane in doc["lanes"]:
        cell = seen.get(lane["id"])
        print(f"{lane['id']}  wall {cell['wall'] if cell and cell['runs'] else float('nan'):7.1f} s  "
              f"budget {lane['budget_s']:5d} -> {budgets[lane['id']]:5d} s")
    for note in notes:
        print(f"NOTE: {note}")
    if a.write:
        path.write_text(json.dumps(apply(doc, budgets, shares), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"OK: wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
