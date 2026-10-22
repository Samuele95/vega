"""
The run-level records of a lanes run (D6 §16.3, §6.12): `timing.json` / `timing.md` — per lane
and stage, wall and CPU seconds against the lane's budget — and the flake summary line.

    warns = write_timing(matrix_root, tier, observed_docs, lanes, warn_factor)
    warns += flake_warnings(observed_docs, flakes)

A lane over `budget_warn_factor x budget_s` is a `WARN:`; T12 sets the budgets to 1.5 x the
measured time. A flake that fired on more than its policy's share of the calls it could have
fired on is a `WARN:` naming the counts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import contracts
from .registry import Lane
from .runner import Flakes


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_timing(root: Path, tier: str | None, docs: Sequence[Mapping], lanes: Mapping[str, Lane],
                 warn_factor: float = 1.25) -> list[str]:
    """timing.json / timing.md under `root`; the `WARN:` lines of the lanes over budget."""
    rows, warns = [], []
    for doc in docs:
        stages = [{"id": s["id"], "status": s["status"], "wall_s": s["seconds"], "cpu_s": s.get("cpu_s", 0)}
                  for s in doc["stages"]]
        wall = round(sum(s["wall_s"] for s in stages), 1)
        budget = lanes[doc["lane"]].budget_s
        rows.append({"lane": doc["lane"], "verdict": doc["verdict"], "wall_s": wall,
                     "cpu_s": round(sum(s["cpu_s"] for s in stages), 1), "budget_s": budget, "stages": stages})
        if wall > warn_factor * budget:
            warns.append(f"WARN: {doc['lane']} exceeded its budget: {wall:.0f} s > {warn_factor} x {budget} s")
    record = {"schema": contracts.MATRIX_TIMING, "generated_at": _now(), "tier": tier,
              "wall_s": _span(docs), "lanes": rows}
    (root / "timing.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (root / "timing.md").write_text(timing_md(record), encoding="utf-8")
    return warns


def _span(docs: Sequence[Mapping]) -> float:
    """The run's wall time: first lane start to last lane finish."""
    if not docs:
        return 0.0
    parse = lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    start = min(parse(d["started_at"]) for d in docs)
    return round((max(parse(d["finished_at"]) for d in docs) - start).total_seconds(), 1)


def timing_md(doc: Mapping[str, Any]) -> str:
    stages = [s for s in contracts.STAGES] + [contracts.GOLDEN_STAGE]
    head = "| lane | verdict | wall s | cpu s | budget s | " + " | ".join(stages) + " |"
    lines = [f"# Matrix timing — {doc['generated_at']} (tier {doc['tier']}, wall {doc['wall_s']} s)",
             "", head, "|" + "---|" * (5 + len(stages))]
    for row in doc["lanes"]:
        by_id = {s["id"]: s for s in row["stages"]}
        cells = [f"{by_id[s]['wall_s']:.0f}" if s in by_id and by_id[s]["status"] in ("pass", "fail", "timeout")
                 else ("·" if s not in by_id else by_id[s]["status"]) for s in stages]
        lines.append(f"| {row['lane']} | {row['verdict']} | {row['wall_s']} | {row['cpu_s']} | "
                     f"{row['budget_s']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def flake_warnings(docs: Sequence[Mapping], flakes: Flakes) -> list[str]:
    """`WARN: flake <id> fired <n> of <m> calls` above the policy's share (D6 §6.12): the
    retries every lane recorded against the calls its logs and scripts made."""
    fired: dict[str, int] = {}
    calls: dict[str, int] = {}
    for doc in docs:
        for flake in doc.get("flakes") or []:
            fired[flake["id"]] = fired.get(flake["id"], 0) + 1
        for fid, n in (doc.get("flake_calls") or {}).items():
            calls[fid] = calls.get(fid, 0) + n
    return [f"WARN: flake {fid} fired {n} of {max(calls.get(fid, n), n)} calls" for fid, n in fired.items()
            if n > flakes.warn_share * max(calls.get(fid, n), n)]
