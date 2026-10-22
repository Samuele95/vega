"""
plan_claims — rule V15 of `PlanValidator` (contract §3.8): what an authored story plan's words
claim must hold against what the facts record. The other rules check that a slot is the right
shape and cites what exists; this one reads what the words say about it (gate C: pages whose
prose contradicted their own records passed V1–V14).

    claim_problems(authored, groups, task) -> [Problem]          each ERROR: V15 <where>: <why>

V15 refuses:

- a verdict limit (`verdict_not_earned`, `verdict_beaten`) that softens a beaten verdict — "could
  not separate", "does as well", "as well as", "indistinguishable" — when the evaluation recorded
  `beaten_by_baseline` (the record says a baseline did better; the page never says less);
- a breakdown caption that calls its bars "the primary metric" when the breakdown's metric is
  another (a per-horizon error under a MASE primary, a per-label PR-AUC under a micro F-score);
- two network choices with one title, and a decision that is no block of the network titled or
  described as one ("The block", "a block of the network");
- two input groups with one label (the page would draw two families no reader can tell apart);
- a class display name (`class_names`) for a class the task does not have, two classes shown by
  one name, or a name that carries a digit (the words a class is shown by state no number).
"""
from __future__ import annotations

import re
from typing import Mapping

from vega_show_core import Problem

__all__ = ["claim_problems", "SOFTER"]

SOFTER = ("could not separate", "does as well", "as well as", "indistinguishable")
_VERDICT_LIMITS = ("verdict_not_earned", "verdict_beaten")
_PRIMARY = re.compile(r"\bprimary metric\b", re.IGNORECASE)
_BLOCK_WORDS = re.compile(r"^the block$|\ba block of the network\b", re.IGNORECASE)


def claim_problems(authored: Mapping, groups: Mapping, task) -> list[Problem]:
    authored = authored if isinstance(authored, Mapping) else {}
    results = authored.get("results") if isinstance(authored.get("results"), Mapping) else {}
    return [*_verdict(results, groups.get("verdict") or {}),
            *_breakdown(results, groups.get("breakdown") or {}, getattr(task, "primary", None)),
            *_choices(authored.get("network"), groups.get("decisions") or ()),
            *_groups(authored.get("input_groups")),
            *_class_names(authored.get("class_names"), getattr(task, "classes", None))]


def _verdict(results: Mapping, verdict: Mapping) -> list[Problem]:
    if not verdict.get("beaten_by_baseline"):
        return []
    out = []
    for i, lim in enumerate(results.get("limits") or ()):
        if not isinstance(lim, Mapping) or lim.get("id") not in _VERDICT_LIMITS:
            continue
        said = f"{lim.get('title') or ''} {lim.get('text') or ''}".lower()
        soft = [w for w in SOFTER if w in said]
        if soft:
            out.append(Problem("V15", f"authored.results.limits[{i}]", f"the evaluation recorded the "
                               f"network beaten by a baseline; {soft[0]!r} says less than that"))
    return out


def _breakdown(results: Mapping, breakdown: Mapping, primary) -> list[Problem]:
    caption, metric = results.get("breakdown_caption"), breakdown.get("metric")
    if not isinstance(caption, str) or not metric or metric == primary or not _PRIMARY.search(caption):
        return []
    return [Problem("V15", "authored.results.breakdown_caption", f"the caption calls the bars the "
                    f"primary metric; the breakdown reads {metric}, the primary is {primary}")]


def _choices(network, decisions) -> list[Problem]:
    choices = network.get("choices") if isinstance(network, Mapping) else None
    if not isinstance(choices, list):
        return []
    blocks = {str(d.get("id")) for d in decisions if isinstance(d, Mapping)
              and str(d.get("path", "")).startswith("blocks[")}
    out, seen = [], {}
    for i, choice in enumerate(c for c in choices if isinstance(c, Mapping)):
        title, did = str(choice.get("title") or ""), str(choice.get("decision"))
        if title in seen:
            out.append(Problem("V15", f"authored.network.choices[{i}].title", f"{title!r} is also "
                               f"choice {seen[title]}'s title; each choice is told apart"))
        seen.setdefault(title, i)
        said = [t for t in (title, str(choice.get("text") or "")) if _BLOCK_WORDS.search(t)]
        if said and did not in blocks:
            out.append(Problem("V15", f"authored.network.choices[{i}]", f"the decision {did!r} is no "
                               "block of the network, and the choice calls it one"))
    return out


def _groups(groups) -> list[Problem]:
    if not isinstance(groups, list):
        return []
    out, seen = [], {}
    for i, group in enumerate(g for g in groups if isinstance(g, Mapping)):
        label = str(group.get("label") or "")
        if label in seen:
            out.append(Problem("V15", f"authored.input_groups[{i}].label", f"{label!r} is also group "
                               f"{seen[label]}'s label; a reader cannot tell the two families apart"))
        seen.setdefault(label, i)
    return out


def _class_names(names, classes) -> list[Problem]:
    if not names:
        return []
    known = {str(c) for c in classes or ()}
    where, out, seen = "authored.class_names", [], {}
    for cls, words in names.items():
        if str(cls) not in known:
            out.append(Problem("V15", f"{where}.{cls}", f"{cls!r} is not a class of the task "
                               f"({', '.join(sorted(known)) or 'none'})"))
        if any(ch.isdigit() for ch in str(words)):
            out.append(Problem("V15", f"{where}.{cls}", f"{words!r} carries a digit; a class is shown "
                               "by words"))
        if str(words) in seen:
            out.append(Problem("V15", f"{where}.{cls}", f"{words!r} also names {seen[str(words)]!r}"))
        seen.setdefault(str(words), str(cls))
    return out
