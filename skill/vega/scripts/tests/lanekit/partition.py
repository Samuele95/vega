"""
The partition block of a lane's geometry layer (report-contract §1 `partitions`), which
lanekit.layers.geometry_payload writes:

    block = partitions(SCHEMA, lane, plan, features, table)   # chosen, provenance, candidates
    task = candidate_task(frame[column])                       # the task a label column would pose

On a labelled lane the block states the lane's one label, chosen (`single-candidate`). On a lane
that takes no label (L07's reconstruction, L08's unlabelled anomaly) it states every label
column of the generator as the candidate a supervised question would take, and abstains
(BUILD-PLAN amendment A12: dataset-forge's validator wants at least one candidate). Every
candidate carries its consequence block (report-contract §3: the decision, the alternatives,
what it changes downstream).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .recipes import ColumnPlan, task_prose

__all__ = ["partitions", "candidate_task"]


def _raw(lane: Any) -> Mapping[str, Any]:
    return lane.raw if hasattr(lane, "raw") else lane


def partitions(schema: Sequence[Any], lane: Any, plan: ColumnPlan, features: list[str],
               table: Any = None) -> dict:
    """The lane's partition block: its one label, chosen; or, on a lane that takes no label,
    every label column of the generator as a candidate, the choice abstained. `table` (a
    lanekit.layers.RawTable) lets a candidate's task be read off its values."""
    dim = len([c for c in schema if c.role == "numeric"]) + len(plan.nominal)
    lid = (_raw(lane) or {}).get("id")
    if plan.label:
        # the registry names the label: a candidate by that declaration (dataset-forge refuses a
        # candidate step 4 does not admit unless a stance names it), never by its column's name
        return {"chosen": plan.label, "provenance": "single-candidate",
                "candidates": [{"label": plan.label, "task": task_prose(lane), "features": features,
                                "stance": f"lane: the lane registry of {lid} names {plan.label} as the target",
                                "dropped_for_leakage": list(plan.leak), "input_dim": dim,
                                "consequences": _chosen_consequences(lane, plan, dim)}]}
    labels = [c.name for c in schema if c.role == "label"]
    drops = [c.name for c in schema if c.role == "drop"]
    candidates = []
    for label in labels:
        leak = [c for c in labels if c != label] + drops
        task = candidate_task(table.frame[label] if table is not None and label in table.frame else None)
        candidates.append({"label": label, "task": task, "features": features, "dropped_for_leakage": leak,
                           "stance": f"lane: the generator of {lid} holds {label} as a label column",
                           "input_dim": dim, "consequences": _abstained_consequences(lane, label, task, leak, dim)})
    return {"chosen": None, "provenance": "abstained", "candidates": candidates}


def candidate_task(values: Any) -> str:
    """The task a label column would pose, read off its values: a delimited set of tags is a
    multilabel classification, two values a binary one, a few (at most 20) a multiclass one, a
    quantity a regression."""
    if values is None:
        return "multiclass classification"
    kept = values.dropna()
    textual = kept.dtype.kind not in "biufc"          # object, or pandas' own string dtype
    if textual and kept.astype(str).str.contains("|", regex=False).any():
        return "multilabel classification"
    distinct = kept.nunique()
    if distinct == 2:
        return "binary classification"
    if textual or distinct <= 20:
        return "multiclass classification"
    return "regression"


def _chosen_consequences(lane: Any, plan: ColumnPlan, dim: int) -> dict:
    """The chosen candidate's consequence block: the decision — forced, the lane declares one
    target — no alternative, and what it changes downstream."""
    leak = list(plan.leak)
    held = f"{', '.join(leak)} leave{'s' if len(leak) == 1 else ''} with it" if leak else "nothing else leaves"
    return {"decision": f"{plan.label} is the label: the lane declares one target, so the choice is forced",
            "alternatives": [],
            "downstream": {
                "preprocessing": f"the shaper selects {plan.label} and {held}, before the split",
                "model_family": f"a {task_prose(lane)} network over {dim} inputs",
                "leakage": (f"{', '.join(leak)} would give {plan.label} away: kept, the network would "
                            "learn the leak instead of the phenomenon") if leak else
                           f"none: no column of the generator derives {plan.label}",
                "dimensionality": f"/vega-forge receives input_dim {dim}",
                "interpretability": f"the network answers {plan.label} from the columns the lane "
                                    "knows before the fact"}}


def _abstained_consequences(lane: Any, label: str, task: str, leak: Sequence[str], dim: int) -> dict:
    """The consequence block of a candidate the lane does not choose: the lane's goal takes no
    label, so it abstains; choosing the candidate is the alternative."""
    raw = _raw(lane)
    goal = {"reconstruct": "a reconstruction", "anomaly": "an unlabelled anomaly search"}.get(
        str(raw.get("goal")), f"the {raw.get('goal')} goal")
    return {"decision": f"no partition: the lane's question is {goal}, which takes no label, so {label} "
                        "stays a candidate and nothing is chosen",
            "alternatives": [f"choose {label}: a {task} over {dim} inputs"],
            "downstream": {
                "preprocessing": f"the shaper selects no label; {label} leaves before the split with the "
                                 "other label candidates, so no input carries it",
                "model_family": f"the {raw.get('family')} network reads the {dim} inputs alone; a {task} "
                                "network would need the label",
                "leakage": f"none for this lane: nothing is predicted; were {label} chosen, "
                           + (f"{', '.join(leak)} would leave with it" if leak else "no other column leaves"),
                "dimensionality": f"/vega-forge receives input_dim {dim}",
                "interpretability": f"{label} stays outside the network, available to read its outputs against"}}
