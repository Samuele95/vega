"""
propose_roles — `RoleProposer`: the class roles a draft proposes (contract §3.5, R-E5; D4
§10.2's proposal, which R-E9 makes the draft's).

The page colours from roles, never from class names. The roles cover the task's classes (view
keys `classes`, `binary`, `labels`, `ordinal`) or the two sides of an anomaly task (the fact
set's `task.outcomes`: the class names when a label exists, else `normal` / `anomalous`); a task
whose view key is `value`, `reconstruction` or `horizon` has no roles (the proposal is null).

**The neutral class** comes from a structural rule (`catalog.roles.neutral_rules`, keyed by the
task kind, and by the base for a `sequence-*` task):

- `structural:negative` — the negative class of a binary task: the fact set's `outcomes[0]`
  (negative first), else `classes[0]` (the positive class is `classes[1]`, A16);
- `structural:majority` — the train-majority class, when its train share is at least
  `roles.neutral_majority_share` and no other class, including those whose share the fact set
  could not key, can match it;
- `task:anomaly` — the normal side; the anomalous side is `alert` by task, never downgraded.

Multilabel, ordinal and the tasks without classes get no neutral. **`alert` is never proposed
without a basis outside the model**: `task:anomaly`, or a `show: roles <class> <role> (<basis>),
… (user)` stance of this dataset (`memory_stances`; its classes take the basis `show:<date>`).
Every other class is `plain` (`vega:default`). The alert candidates — every class but the
neutral — are offered to fork `S2`, open when there are candidates and neither a stance nor the
task settles them.

The proposal is the story-plan's `proposals.class_roles`:

    {neutral: {class, rule, share} | null, alert_candidates, basis_found, memory, evidence,
     fork: S2 | null, roles: {class: role}, source: {class: basis}}

`roles` and `source` are the whole proposal (D4 §10.2: the proposed neutral, the settled alerts,
all others plain) — what fork S2 offers first. A stance entry that confirms what the structural
rule or the task already gives keeps that basis (the orders lane: `0` neutral
`structural:negative`, `1` alert `show:<date>`); every other entry takes `show:<date>`. On a
binary task the positive class is `spec:` evidence (the class the network is asked to find):
it is listed in `evidence`, never made `alert` by itself (R-E5: `spec:` only raises the
proposal), and `basis_found` names only a basis that settles (`show:`, `task:anomaly`).

`s2_open(proposal, authored)` says whether fork S2 is still to be asked: the proposal opens it,
and no authored role rests on a `user:` turn or a `show:` stance (roles a carry filled from an
earlier plan answer nothing). `unattended(proposal, authored)` is its unattended answer: the
authored roles when a carry filled them, else the proposal; an `alert` whose basis is not
`user:`, `show:` or `task:anomaly` becomes `plain`, and the catalog's S2 `WARN:` names the
candidates left plain.
"""
from __future__ import annotations

import re
from typing import Mapping

from memory_stances import USER_SUFFIX, MemoryStances, Stance
from vega_show_core import CATALOG, TaskShape

__all__ = ["RoleProposer", "s2_open"]

_DEFAULT = "vega:default"
_ANOMALY = "task:anomaly"
_SPEC_POSITIVE = "spec:task.positive"
_ALERT_BASES = tuple(CATALOG["roles"]["alert_requires"])
_ENTRY = re.compile(r"\s*(.+?)\s+(neutral|alert|plain)(?:\s+\(([^()]*)\))?\s*(?:,|$)")


class RoleProposer:
    """The class-role proposal of one task, from its train shares and the recorded stances."""

    def __init__(self, task: TaskShape, balance: Mapping[str, tuple], stances: MemoryStances,
                 catalog: Mapping = CATALOG) -> None:
        self._task, self._balance, self._stances = task, dict(balance or {}), stances
        self._spec, self._forks = catalog["roles"], catalog["forks"]
        self._markers: list[str] = []

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def propose(self) -> dict | None:
        self._markers = []
        covered = self._covered()
        if not covered:
            return None
        rule = self._rule()
        neutral = self._neutral(rule, covered)
        roles = {c: "plain" for c in covered}
        source = {c: _DEFAULT for c in covered}
        if neutral:
            roles[neutral["class"]], source[neutral["class"]] = "neutral", rule
        if rule == _ANOMALY:
            roles[covered[1]], source[covered[1]] = "alert", _ANOMALY
        stance = self._stance(covered, roles, source)
        candidates = [c for c in covered if neutral and c != neutral["class"]]
        settling = ([stance.basis] if stance else []) + ([_ANOMALY] if rule == _ANOMALY else [])
        return {"neutral": neutral, "alert_candidates": candidates,
                "basis_found": settling[0] if settling else None,
                "memory": stance.line if stance else None,
                "evidence": settling + self._spec_evidence(covered),
                "fork": None if settling or not candidates else self._spec["fork"],
                "roles": roles, "source": source}

    def unattended(self, proposal: Mapping | None,
                   authored: Mapping | None = None) -> tuple[dict | None, list[str]]:
        """Fork S2's unattended answer (module docstring) and its `WARN:` markers."""
        if not proposal:
            return None, []
        base = authored or proposal
        roles, source = dict(base["roles"]), dict(base["source"])
        unsupported = [c for c, r in roles.items()
                       if r == "alert" and not str(source.get(c, "")).startswith(_ALERT_BASES)]
        for name in unsupported:
            roles[name], source[name] = "plain", _DEFAULT
        left = unsupported
        if s2_open(proposal, authored):
            left = [c for c in proposal["alert_candidates"] if roles.get(c) != "alert"]
        marker = self._forks["S2"]["marker"].replace("<classes>", ", ".join(left))
        return {"roles": roles, "source": source}, [marker] if left else []

    def _spec_evidence(self, covered: list[str]) -> list[str]:
        """`spec:` evidence for an alert: the positive side of a binary view (negative first)."""
        sides = list(self._task.outcomes or covered)
        if self._task.view_key != "binary" or len(sides) != 2:
            return []
        return [f"{_SPEC_POSITIVE} {sides[1]}"]

    def _covered(self) -> list[str] | None:
        covers = self._spec["covers"].get(self._task.view_key)
        if covers is None:
            return None
        names = self._task.outcomes if self._task.view_key == "anomaly" else self._task.classes
        return list(names or ()) or None

    def _rule(self) -> str | None:
        rule = self._spec["neutral_rules"].get(self._task.kind)
        return rule.get(self._task.base) if isinstance(rule, Mapping) else rule

    def _neutral(self, rule: str | None, covered: list[str]) -> dict | None:
        if rule == "structural:negative":
            name = (self._task.outcomes or covered)[0]
        elif rule == _ANOMALY:
            name = covered[0]
        elif rule == "structural:majority":
            name = self._majority(covered)
        else:
            return None
        if name is None:
            return None
        return {"class": name, "rule": rule, "share": (self._balance.get(name) or (None,))[0]}

    def _majority(self, covered: list[str]) -> str | None:
        known = {c: v for c in covered if isinstance(v := (self._balance.get(c) or (None, None))[1],
                                                     (int, float)) and not isinstance(v, bool)}
        if not known:
            self._markers.append("WARN: the fact set holds no train share of any class; no "
                                 "structural:majority neutral is proposed")
            return None
        top = max(known, key=lambda c: (known[c], -covered.index(c)))
        others = [v for c, v in known.items() if c != top]
        unknown = max(0.0, 1.0 - sum(known.values())) if len(known) < len(covered) else 0.0
        share = known[top]
        if share >= self._spec["neutral_majority_share"] and share > unknown and \
                all(share > v for v in others):
            return top
        return None

    def _stance(self, covered: list[str], roles: dict, source: dict) -> Stance | None:
        """Apply the latest `show: roles` stance to `roles`/`source` in place; the stance, or
        None when there is none or it cannot be applied (with a `WARN:`)."""
        stance = self._stances.latest("roles")
        if stance is None:
            return None
        body = USER_SUFFIX.sub("", stance.body)
        entries = [(m.group(1).strip(), m.group(2)) for m in _ENTRY.finditer(body)]
        unknown = [c for c, _ in entries if c not in covered]
        if unknown or not entries:
            self._markers.append(f"WARN: the stance {stance.line!r} names no role of this task's "
                                 f"classes {covered}{f' (unknown: {unknown})' if unknown else ''}; "
                                 "not applied — fork S2 stays open")
            return None
        trial_roles, trial_source = dict(roles), dict(source)
        for name, role in entries:
            if source[name] == _ANOMALY and roles[name] == "alert" and role != "alert":
                self._markers.append(f"WARN: the stance {stance.line!r} makes the anomalous side "
                                     f"{name!r} {role}; the anomalous side is alert by task and is "
                                     "never downgraded")
                continue
            if role == roles[name] and source[name] != _DEFAULT:
                continue  # the stance confirms the structural rule or the task: that basis stays
            trial_roles[name], trial_source[name] = role, stance.basis
        neutrals = [c for c, r in trial_roles.items() if r == "neutral"]
        if len(neutrals) > 1 and self._task.view_key in self._spec["neutral_at_most_one"]:
            self._markers.append(f"WARN: the stance {stance.line!r} gives {neutrals} the neutral "
                                 "role; a task of this view key has at most one; not applied — "
                                 "fork S2 stays open")
            return None
        roles.update(trial_roles)
        source.update(trial_source)
        return stance


def s2_open(proposal: Mapping | None, authored: Mapping | None) -> bool:
    """Whether fork S2 is still to be asked: the proposal opens it and no authored role rests
    on a `user:` turn or a `show:` stance."""
    if not proposal or not proposal.get("fork"):
        return False
    bases = ((authored or {}).get("source") or {}).values()
    return not any(str(b).startswith(("user:", "show:")) for b in bases)
