"""
plan_roles — the class-role check of `plan_rules.PlanValidator` (V6, contract §3.5, R-E5), which
`plan_rules` re-exports with the other block checks (`plan_rules.check_class_roles`).

Every class the page names has one role, and every role one basis that grants it:

- the roles cover the task's classes, or an anomaly task's two sides (`covered_classes`); a task
  whose view key carries no roles (`value`, `reconstruction`, `horizon`) takes none;
- at most one `neutral` for the `classes`, `binary` and `anomaly` view keys;
- the basis follows the catalog's grammar and grants the role (`roles.basis`): `alert` only on a
  `user:` turn, a `show:` stance, `task:anomaly` or `registry:<lane>`; `vega:default` never alert;
- `registry:<lane>` only inside the verification harness: `VEGA_LANE_REGISTRY` (which the matrix
  sets, a production run never) names the lane registry (`vega/test-lanes@1`), and it declares
  that lane and that role for that class (`lanes[].roles`); anywhere else the basis is refused;
- a structural basis only on the class its rule gives (the draft's proposed neutral), and
  `task:anomaly` only as the task gives it (the normal side neutral, the anomalous side alert);
- the anomalous side of an anomaly task is alert and is never downgraded.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping

from vega_show_core import CATALOG, Problem, TaskShape

__all__ = ["covered_classes", "check_class_roles", "LANE_REGISTRY_ENV"]

LANE_REGISTRY_ENV = "VEGA_LANE_REGISTRY"   # the verification harness's lane registry (lanes.json)
LANE_REGISTRY_SCHEMA = "vega/test-lanes@1"


def covered_classes(task: TaskShape, catalog: Mapping = CATALOG) -> list[str] | None:
    """The names class roles cover: the classes, or an anomaly task's two sides; None for a task
    whose view key carries no roles."""
    if catalog["roles"]["covers"].get(task.view_key) is None:
        return None
    names = task.outcomes if task.view_key == "anomaly" else task.classes
    return list(names or ()) or None


def check_class_roles(block, task: TaskShape, proposal,
                      catalog: Mapping = CATALOG) -> list[Problem]:
    """V6 (contract §3.5): the class-role block of an authored plan."""
    where, covered = "authored.class_roles", covered_classes(task, catalog)
    if covered is None:
        roles = block.get("roles") if isinstance(block, Mapping) else None
        alert = " (an alert included)" if isinstance(roles, Mapping) and "alert" in roles.values() \
            else ""
        return [] if block is None else [Problem("V6", where, f"a {task.view_key} task has no "
                                                 f"classes: no class role{alert} is given, and "
                                                 "class_roles is null")]
    if block is None:
        return []
    if not (isinstance(block, Mapping) and isinstance(block.get("roles"), Mapping)
            and isinstance(block.get("source"), Mapping)):
        return [Problem("V6", where, "class roles are {roles: {class: role}, "
                                     "source: {class: basis}}")]
    roles, source = block["roles"], block["source"]
    missing = [c for c in covered if c not in roles]
    out = [Problem("V6", f"{where}.roles", f"every class needs a role: {missing} have none "
                                           f"(classes: {', '.join(covered)})")] if missing else []
    out += [Problem("V6", f"{where}.roles.{c}", f"{c!r} is not a class of this task "
                                                f"({', '.join(covered)})") for c in roles
            if c not in covered]
    out += [Problem("V6", f"{where}.source", f"every class names its basis: {c!r} has none")
            for c in roles if c in covered and c not in source]
    neutrals = [c for c in covered if roles.get(c) == "neutral"]
    if len(neutrals) > 1 and task.view_key in catalog["roles"]["neutral_at_most_one"]:
        out.append(Problem("V6", f"{where}.roles", f"{len(neutrals)} classes are neutral "
                           f"({', '.join(neutrals)}); a {task.view_key} task has at most one"))
    rules = _RoleRules(task, proposal, covered, catalog)
    return out + [p for c in covered if c in roles and c in source
                  if (p := rules.problem(c, roles[c], source[c]))]


class _RoleRules:
    """What one class's role and basis must satisfy: the basis grammar, what the basis grants,
    a structural basis only where its rule points, the anomaly sides by task."""

    def __init__(self, task: TaskShape, proposal, covered: list, catalog: Mapping) -> None:
        self._task, self._covered, self._spec = task, covered, catalog["roles"]
        neutral = (proposal or {}).get("neutral") if isinstance(proposal, Mapping) else None
        self._neutral = neutral if isinstance(neutral, Mapping) else {}

    def problem(self, cls: str, role, basis) -> Problem | None:
        where, spec = f"authored.class_roles.roles.{cls}", self._spec
        if role not in spec["set"]:
            return Problem("V6", where, f"{role!r} is not a class role ({', '.join(spec['set'])})")
        if not isinstance(basis, str) or not re.fullmatch(spec["basis_pattern"], basis):
            return Problem("V6", f"authored.class_roles.source.{cls}",
                           f"the basis {basis!r} is not one of {spec['basis_pattern']}")
        anomalous = self._task.view_key == "anomaly" and cls == self._covered[-1]
        if anomalous and role != "alert":
            return Problem("V6", where, f"the anomalous side {cls!r} is alert by task and is never "
                                        "downgraded")
        grants = spec["basis"][_basis_key(basis)]["grants"]
        if role not in grants:
            why = ("alert requires a user: turn, a show: stance, task:anomaly or the harness's "
                   "registry:<lane>" if role == "alert" else f"it grants {', '.join(grants)}")
            return Problem("V6", where, f"{cls!r} is {role} on the basis {basis!r}; {why}")
        if basis.startswith("registry:"):
            return _registry_problem(where, cls, role, basis[len("registry:"):])
        return self._rule_problem(where, cls, role, basis)

    def _rule_problem(self, where: str, cls: str, role: str, basis: str) -> Problem | None:
        if basis.startswith("structural:") and (self._neutral.get("class"), self._neutral.get(
                "rule")) != (cls, basis):
            return Problem("V6", where, f"{cls!r} takes the basis {basis!r}, which the structural "
                                        f"rule gives only to {self._neutral.get('class')!r}")
        if basis == "task:anomaly":
            sides = {self._covered[0]: "neutral", self._covered[-1]: "alert"}
            if self._task.view_key != "anomaly" or sides.get(cls) != role:
                return Problem("V6", where, f"task:anomaly makes the normal side neutral and the "
                                            f"anomalous side alert; not {cls!r} {role}")
        return None


def _basis_key(basis: str) -> str:
    """The catalog's `roles.basis` entry a basis is: `user:<turn>`, `show:<stance date>`,
    `registry:<lane>`, or the basis itself."""
    for prefix, key in (("user:", "user:<turn>"), ("show:", "show:<stance date>"),
                        ("registry:", "registry:<lane>")):
        if basis.startswith(prefix):
            return key
    return basis


def _registry_problem(where: str, cls: str, role: str, lane: str) -> Problem | None:
    """`registry:<lane>` is the verification harness's basis: the lane registry `VEGA_LANE_REGISTRY`
    names must declare that lane, and that role for that class. A run without one (every
    production run) cannot use it."""
    path = os.environ.get(LANE_REGISTRY_ENV)
    if not path:
        return Problem("V6", where, f"registry:{lane} is the verification harness's basis and no "
                       f"lane registry is given ({LANE_REGISTRY_ENV}); a run outside the harness "
                       "gives a user: turn, a show: stance or task:anomaly")
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        lanes = doc.get("lanes") or [] if doc.get("schema") == LANE_REGISTRY_SCHEMA else None
    except (OSError, ValueError, AttributeError) as exc:
        return Problem("V6", where, f"the lane registry {path} is unreadable: {exc}")
    if lanes is None:
        return Problem("V6", where, f"{path} is no lane registry ({LANE_REGISTRY_SCHEMA}); "
                                    f"registry:{lane} needs the harness's own")
    declared = next((dict(l.get("roles") or {}) for l in lanes
                     if isinstance(l, Mapping) and l.get("id") == lane), None)
    if declared is None or declared.get(cls) != role:
        said = "has no lane " + repr(lane) if declared is None else \
            f"gives {cls!r} the role {declared.get(cls)!r}"
        return Problem("V6", where, f"{cls!r} is {role} on registry:{lane}, and the lane registry "
                                    f"{said}")
    return None
