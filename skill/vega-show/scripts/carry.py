"""
carry — `PlanCarrier`: the authored text of an earlier story plan, carried into a new draft
(`draft_plan.py --carry`, D5 §7.6). The draft's proposals are the new fact set's; the carrier
fills the draft's `authored` slots from the earlier plan wherever what they reference still
exists, and every slot or item it cannot carry is a `WARN:` (a `Problem` of rule `carry`),
never a silent loss. The model then completes the plan under the contract (`V1`–`V14`).

- **From a `vega/story-plan@2`**: every authored slot whose referenced keys and ids still exist
  is kept — a placeholder must still be a fact key (or a lexicon path), a choice's decision, a
  run label's run, a results note's fact, a limit's id and a code note's walk id must still be
  proposed, the motif still admissible, the class roles must still cover exactly the proposed
  classes, and authored report roles are kept only while the report's roles are still
  `proposed`. A slot object whose own text (a lead, a caption, a headline) cannot be carried is
  dropped whole; a list or map inside it loses only the items that cannot. A slot that does not
  read as its schema says is dropped with a `WARN:`, never a traceback.
- **From a `vega/story-plan@1`** (the hand-authored v2.2 plan): `carry_v1.V1Carry` maps each
  `values` entry to a fact key by ref and renames the placeholders (see that module).

**Input groups** come out as a partition of the shaped inputs either way: a member that is no
longer an input is dropped, a member placed twice stays in its first group, and the inputs no
carried group holds are gathered in a group `_unplaced`, which `V7` refuses (its id breaks the
id pattern) until the model places them.

`carry(draft)` takes the new draft (`vega/story-plan@2`, its authored slots null) and returns
`(authored, problems)`: every slot of the draft's `authored`, carried or null.
"""
from __future__ import annotations

from typing import Callable, Mapping

from digit_rule import PLACEHOLDER_RE
from vega_show_core import CATALOG, FactSet, Problem, thaw

__all__ = ["PlanCarrier", "CarryContext", "carry_slots", "rename_placeholders", "UNPLACED",
           "RULE"]

RULE = "carry"
UNPLACED = "_unplaced"
_V1, _V2 = "vega/story-plan@1", "vega/story-plan@2"
_LEXICON_KEYS = frozenset(CATALOG["lexicon"]["keys"])
_GROUPS = "authored.input_groups"


def rename_placeholders(text: str, mapping: Mapping[str, str]) -> tuple[str, list[str]]:
    """`text` with each `{name}` placeholder renamed through `mapping` (its format suffix kept)
    and each lexicon path kept; the placeholders `mapping` does not name (or a lexicon path of
    an unknown key) are listed as missing."""
    missing: list[str] = []

    def rename(m):
        name, path, suffix = m.group(1), m.group(2), m.group(3)
        if path is not None:
            if name not in _LEXICON_KEYS:
                missing.append(m.group(0))
            return m.group(0)
        if name not in mapping:
            missing.append(m.group(0))
            return m.group(0)
        return "{" + mapping[name] + (f":{suffix}" if suffix else "") + "}"
    return PLACEHOLDER_RE.sub(rename, text), missing


class CarryContext:
    """What a carrier reads and shares: the fact set, its groups, the draft's proposals and the
    `WARN:` problems of the carry."""

    def __init__(self, facts: FactSet, draft: Mapping) -> None:
        self.facts, self.groups = facts, thaw(facts.groups)
        self.proposals = dict(draft.get("proposals") or {})
        self.keys = {k: k for k in facts.keys()}
        self.problems: list[Problem] = []

    def warn(self, where: str, message: str) -> None:
        self.problems.append(Problem(RULE, where, message, level="WARN"))

    def text(self, where: str, text, mapping: Mapping[str, str] | None = None) -> str | None:
        """`text` with its placeholders renamed (`mapping`, default the fact keys themselves);
        None, with a `WARN:`, when it is no text or a placeholder names no fact."""
        if not isinstance(text, str) or not text.strip():
            if text is not None:
                self.warn(where, "not carried: not a text")
            return None
        out, missing = rename_placeholders(text, self.keys if mapping is None else mapping)
        if missing:
            self.warn(where, f"not carried: {', '.join(missing)} names no fact of this fact set "
                             "— the model writes it again")
            return None
        return out

    def texts(self, where: str, obj: Mapping, fields: tuple[str, ...],
              mapping: Mapping[str, str] | None = None) -> tuple | None:
        """The carried texts of `fields` of `obj`, or None when any of them is not carried."""
        out = tuple(self.text(f"{where}.{f}", obj.get(f), mapping) for f in fields)
        return None if any(t is None for t in out) else out

    def keyed(self, where: str, items: Mapping, allowed, why: str) -> dict:
        """The items of a `{key: text}` map whose key is `allowed` and whose text carries; each
        other key is a `WARN:` saying `why`."""
        out = {}
        for key, text in items.items():
            if key not in allowed:
                self.warn(f"{where}.{key}", f"{why}; dropped")
            elif (carried := self.text(f"{where}.{key}", text)) is not None:
                out[key] = carried
        return out

    def ids(self, proposal: str, field: str) -> list[str]:
        """The `field` of every item of a list proposal (`runs` → ids, `code_steps` → steps)."""
        items = self.proposals.get(proposal) or ()
        return [str(p.get(field)) for p in items if isinstance(p, Mapping)]

    @property
    def shaped(self) -> list[str]:
        return [str(n) for n in (self.groups.get("inputs") or {}).get("shaped") or ()]

    @property
    def limit_ids(self) -> list[str]:
        limits = self.groups.get("limits") or ()
        return [str(lim.get("id")) for lim in limits if isinstance(lim, Mapping)]

    def partition(self, groups: list[dict]) -> list[dict]:
        """The carried input groups as a partition of the shaped inputs (module docstring)."""
        order = {n: i for i, n in enumerate(self.shaped)}
        placed, out = set(), []
        for group in groups:
            where, members = f"{_GROUPS}.{group['id']}", []
            for name in group.get("members") or ():
                if name not in order or name in placed:
                    why = "is no shaped input" if name not in order else "is in an earlier group"
                    self.warn(where, f"{name!r} {why}; dropped from this group")
                else:
                    members.append(name)
                    placed.add(name)
            if members:
                out.append({**group, "members": sorted(members, key=order.get)})
            else:
                self.warn(where, "no input left in the group; dropped")
        left = [n for n in self.shaped if n not in placed]
        if left:
            self.warn(_GROUPS, f"the inputs no carried group holds are in the group {UNPLACED!r}, "
                               "which V7 refuses until the model places them")
            out.append({"id": UNPLACED, "label": "unplaced", "text": "", "members": left})
        return out


def carry_slots(ctx: CarryContext, slots: Mapping[str, Callable], old: Mapping) -> dict:
    """`{slot: carried value}` for every slot of `old` that is set, each through its carrier;
    a slot that does not read as its schema says (a list where an object belongs) is dropped
    with a `WARN:`, never a traceback."""
    out = {}
    for slot, carry in slots.items():
        if old.get(slot) is None:
            continue
        try:
            out[slot] = carry(old[slot])
        except (AttributeError, KeyError, TypeError) as exc:
            ctx.warn(f"authored.{slot}", f"not carried: the slot does not read as its schema "
                                         f"says ({type(exc).__name__}: {exc})")
    return out


class PlanCarrier:
    """The authored slots of an earlier plan (`vega/story-plan@1` or `@2`) carried into a draft."""

    def __init__(self, facts: FactSet, previous: Mapping) -> None:
        if previous.get("schema") not in (_V1, _V2):
            raise ValueError(f"not a story plan to carry: schema {previous.get('schema')!r} "
                             f"(known: {_V1}, {_V2})")
        self._facts, self._previous = facts, previous

    def carry(self, draft: Mapping) -> tuple[dict, list[Problem]]:
        ctx = CarryContext(self._facts, draft)
        slots = {slot: None for slot in draft.get("authored") or {}}
        if self._previous["schema"] == _V1:
            from carry_v1 import V1Carry  # the one-off migration (it builds on this module)
            filled = V1Carry(ctx, self._previous).authored()
        else:
            filled = _V2Carry(ctx).authored(self._previous.get("authored") or {})
        for slot, value in filled.items():
            if slot in slots and value is not None:
                slots[slot] = value
        return slots, ctx.problems


class _V2Carry:
    """The slots of a `vega/story-plan@2` whose references still exist."""

    def __init__(self, ctx: CarryContext) -> None:
        self._ctx = ctx

    def authored(self, old: Mapping) -> dict:
        slots = {"locale": self._locale, "lexicon": self._lexicon, "motif": self._motif,
                 "class_roles": self._class_roles, "report_roles": self._report_roles,
                 "headline": self._plain("headline"), "lead": self._plain("lead"),
                 "context": self._context, "network": self._network,
                 "experiments": self._experiments, "results": self._results,
                 "input_groups": self._input_groups, "code_notes": self._code_notes,
                 "monitor_note": self._plain("monitor_note"), "class_names": self._class_names}
        return carry_slots(self._ctx, slots, old)

    def _class_names(self, value):
        """The class display names whose class the task still has; each other one a `WARN:`."""
        classes = {str(c) for c in self._ctx.facts.task.classes or ()}
        kept = {str(c): str(w) for c, w in value.items() if str(c) in classes}
        for c in value:
            if str(c) not in kept:
                self._ctx.warn(f"authored.class_names.{c}", "not a class of the task now; dropped")
        return kept or None

    def _plain(self, slot: str) -> Callable:
        return lambda value: self._ctx.text(f"authored.{slot}", value)

    def _locale(self, value):
        if value in CATALOG["locales"]:
            return value
        return self._ctx.warn("authored.locale", f"{value!r} is no locale of the catalog; dropped")

    def _lexicon(self, value):
        known = {k: v for k, v in value.items() if k in _LEXICON_KEYS}
        for key in sorted(set(value) - set(known)):
            self._ctx.warn(f"authored.lexicon.{key}", "not a lexicon key; dropped")
        return known or None

    def _motif(self, value):
        admissible = (self._ctx.proposals.get("motif") or {}).get("admissible") or ()
        if value in admissible:
            return value
        return self._ctx.warn("authored.motif", f"{value!r} is not admissible for this lane "
                                                f"({', '.join(admissible)}); dropped")

    def _class_roles(self, value):
        covered = set(((self._ctx.proposals.get("class_roles") or {}).get("roles") or {}))
        roles, source = value.get("roles") or {}, value.get("source") or {}
        if covered and set(roles) == covered and set(source) == covered:
            return {"roles": dict(roles), "source": dict(source)}
        return self._ctx.warn("authored.class_roles", "the roles do not cover exactly this "
                                                      "task's classes; dropped — fork S2 decides")

    def _report_roles(self, value):
        if (self._ctx.proposals.get("report_roles") or {}).get("status") != "proposed":
            return self._ctx.warn("authored.report_roles", "the report's roles are not proposed "
                                                           "now (emitted, or no report); dropped")
        kept = {r: str(label) for r, label in value.items() if r in CATALOG["report_roles"]}
        return kept or None

    def _context(self, value):
        ctx, old = self._ctx, value.get("steps") or {}
        proposed = list(ctx.proposals.get("context_steps") or ())
        ordered = {**{s: old[s] for s in proposed if s in old}, **old}  # the proposed order first
        steps = ctx.keyed("authored.context.steps", ordered, proposed,
                          "not a proposed context step")
        facts = [dict(f) for f in value.get("facts") or ()
                 if isinstance(f, Mapping) and f.get("fact") in ctx.keys]
        if len(facts) < len(value.get("facts") or ()):
            ctx.warn("authored.context.facts", "a fact no longer in the fact set; dropped")
        return {"steps": steps, "facts": facts} if steps or facts else None

    def _network(self, value):
        ctx, lead = self._ctx, self._ctx.text("authored.network.lead", value.get("lead"))
        if lead is None:
            return None
        decisions, choices = set(ctx.ids("choices", "decision")), []
        for i, choice in enumerate(value.get("choices") or ()):
            where = f"authored.network.choices[{i}]"
            if choice.get("decision") not in decisions:
                ctx.warn(where, f"the decision {choice.get('decision')!r} is not proposed now; "
                                "dropped")
            elif (texts := ctx.texts(where, choice, ("title", "text"))) is not None:
                choices.append({"decision": choice["decision"], "title": texts[0],
                                "text": texts[1]})
        return {"lead": lead, "choices": choices}

    def _experiments(self, value):
        texts = self._ctx.texts("authored.experiments", value, ("lead", "caption"))
        if texts is None:
            return None
        labels = self._ctx.keyed("authored.experiments.labels", value.get("labels") or {},
                                 set(self._ctx.ids("runs", "id")), "the run is not proposed now")
        return {"lead": texts[0], "caption": texts[1], "labels": labels}

    def _results(self, value):
        ctx = self._ctx
        headline = ctx.text("authored.results.headline", value.get("headline"))
        if headline is None:
            return None
        notes = ctx.keyed("authored.results.notes", value.get("notes") or {},
                          set(ctx.proposals.get("results_facts") or ()),
                          "not a proposed results fact now")
        caption = value.get("breakdown_caption")
        if caption is not None:
            caption = ctx.text("authored.results.breakdown_caption", caption)
        return {"headline": headline, "notes": notes, "breakdown_caption": caption,
                "limits": self._limits(value.get("limits") or ())}

    def _limits(self, limits) -> list[dict]:
        ctx, known, out = self._ctx, set(self._ctx.limit_ids), []
        for lim in limits:
            where = f"authored.results.limits.{lim.get('id')}"
            if lim.get("id") not in known:
                ctx.warn(where, "not a limit of this fact set now; dropped")
            elif (texts := ctx.texts(where, lim, ("title", "text"))) is not None:
                out.append({"id": lim["id"], "title": texts[0], "text": texts[1]})
        return out

    def _input_groups(self, value):
        kept = []
        for group in value:
            gid = str(group.get("id"))
            if gid == UNPLACED:  # its members are unplaced again, by the partition
                continue
            texts = self._ctx.texts(f"{_GROUPS}.{gid}", group, ("label", "text"))
            if texts is not None:
                kept.append({"id": gid, "label": texts[0], "text": texts[1],
                             "members": list(group.get("members") or ())})
        return self._ctx.partition(kept)

    def _code_notes(self, value):
        steps = set(self._ctx.ids("code_steps", "step"))
        return self._ctx.keyed("authored.code_notes", value, steps,
                               "not a proposed walk step now") or None
