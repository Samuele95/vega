"""
The recipe of a generator whose table serves a lane that takes no label (lanekit.datasets.meters
and .turbine use it; it lives beside the other lanekit modules, so a change to it routes every
lane, lanekit.routing).

    recipe = goal_recipe(SCHEMA, raw_path, shaped_dir, lane)     # dataset-shaper/recipe@1

A reconstruction (L07) or an unlabelled anomaly search (L08) reads rows whose generator still
records a label a supervised question would take (a household's tariff, a confirmed fault).
The geometry states it as a candidate and abstains (lanekit.layers, amendment A12), so no
partition is chosen and its leakage set is empty; the column is still a member of the basis, and
the shaper's verifier refuses a basis member that leaves without a `user:` step saying so.

Leaving is the lane's goal decision: the goal stance the memory file records ("goal: … without a
label (user)", lanekit.fixture) is what drops it. So the recipe is the standard one
(lanekit.recipes.standard_recipe), except that on a lane without a target the label candidates
leave in their own `drop_leakage` step citing `user:stance/goal`; any other column the standard
step drops keeps its analysis citation. The steps are renumbered in order. A labelled lane's
recipe is the standard one, unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .recipes import standard_recipe

__all__ = ["goal_recipe", "GOAL_SOURCE"]

GOAL_SOURCE = "user:stance/goal"
GOALS = {"reconstruct": "a reconstruction", "anomaly": "an unlabelled anomaly search"}


def _raw(lane: Any) -> Mapping[str, Any]:
    return lane.raw if hasattr(lane, "raw") else lane


def goal_recipe(schema: Sequence[Any], raw_path: Path, shaped_dir: Path, lane: Any) -> dict:
    """The standard recipe, with the label candidates of a lane that takes no label dropped by
    the lane's goal stance (module docstring)."""
    recipe = standard_recipe(schema, raw_path, shaped_dir, lane)
    raw = _raw(lane)
    if (raw.get("target") or {}).get("column"):
        return recipe
    labels = {c.name for c in schema if c.role == "label"}
    steps: list[dict] = []
    for step in recipe["steps"]:
        held = [c for c in step.get("columns") or [] if c in labels] if step.get("op") == "drop_leakage" else []
        if not held:
            steps.append(step)
            continue
        rest = [c for c in step["columns"] if c not in labels]
        if rest:
            steps.append({**step, "columns": rest,
                          "consequences": {**step["consequences"], "columns": f"-{len(rest)}"}})
        steps.append(_goal_step(raw, held))
    for number, step in enumerate(steps, start=1):
        step["id"] = f"S{number:02d}"
    recipe["steps"] = steps
    return recipe


def _goal_step(raw: Mapping[str, Any], held: Sequence[str]) -> dict:
    goal = GOALS.get(str(raw.get("goal")), f"the {raw.get('goal')} goal")
    names = ", ".join(held)
    return {"id": "", "op": "drop_leakage", "columns": list(held), "params": {}, "source": GOAL_SOURCE,
            "rationale": f"the lane's question is {goal}, which takes no label: {names} "
                         f"{'is a label' if len(held) == 1 else 'are labels'} a supervised question would "
                         "take, and leave before the split so that no input carries one",
            "consequences": {"rows": "unchanged", "columns": f"-{len(held)}",
                             "downstream": "the network reads the rows without any label"}}
