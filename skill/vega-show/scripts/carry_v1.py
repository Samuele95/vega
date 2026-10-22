"""
carry_v1 — `V1Carry`: the hand-authored v2.2 story plan (`vega/story-plan@1`) carried into a
`vega/story-plan@2` draft (D5 §7.6), through `carry.PlanCarrier`.

**Values become fact keys.** Each `values` entry `{name: "source:path"}` is translated to the
contract's ref grammar — `netspec:` → `netspec.json#`, `schema:` → `schema.json#`, `<layer>:` →
`records/<layer>.json#`, `report:<label>[.number|.page|.title]` → `report:<role>.<field>` for the
role whose label it is (the report's emitted or proposed bindings) — and mapped to the fact
whose ref equals it. When several facts share a ref (`netspec.json#task.classes` is both the
class names and their count), the scalar one wins: a v1 placeholder stood in prose. Every
placeholder of a carried text is renamed to its fact key; a text holding a placeholder with no
fact is not carried (`WARN:`). The v1 names with no fact are listed in one `WARN:`.

**Text slots**, as v1 held them:

- `headline`, `lead`; the context steps' texts **by position** onto the proposed context steps,
  and the context facts (`{label, value}` → `{label, fact}`);
- the network's lead and choices, each choice's v1 `focus` mapped to a decision: `hidden` → the
  first dense block (the first dropout block for a choice that speaks of dropout), `inputs` → the first block, `activation` → the first activation block,
  `outputs` → `loss`, `all` and `edges` → `optimizer`, `feature:<name>` → the first decision
  whose `features_tested` holds the name, a proposed decision id → itself;
- the experiments' lead, caption and run labels (a run's `value` → its primary fact → the run
  id), kept for the proposed runs;
- the results' headline, the notes of the proposed results facts, the per-class caption, and
  the limits **by order**, each mapped to the unused limit id of this fact set whose words it
  carries most (`_LIMIT_WORDS`; one with none is not carried);
- `input_groups`: each group's `match` regexes applied once to the shaped inputs, in input
  order, the first group matching an input taking it; the rest go to `_unplaced`;
- `code_notes`: each note under its walk id (`act` → `activation`, `softmax`/`output` →
  `link`, `block:<type>` → `<type>`), kept for the proposed code steps;
- `neutral_class` → `class_roles` only when it is the class the structural rule proposes (the
  proposal's roles, its basis the rule's): a v1 plan is no `user:` turn and no `show:` stance,
  so fork S2 is still asked; any other neutral is not carried (`WARN:`).

Nothing else is carried: the lexicon, the motif, the locale, the report roles and the monitor
note did not exist in v1.
"""
from __future__ import annotations

import re
from typing import Mapping

from carry import UNPLACED, CarryContext, carry_slots
from vega_show_core import CATALOG

__all__ = ["V1Carry", "v2_ref"]

_FILES = {"netspec": "netspec.json", "schema": "schema.json",
          **{layer: f"records/{layer}.json" for layer in
             ("geometry", "analysis", "shape", "architecture", "training", "evaluation")}}
_REPORT_FIELDS = ("number", "page", "title")
_KINDS = ("count", "count_of", "size", "metric", "ci_metric", "diff", "param", "epoch",
          "report_number", "report_page", "text", "list", "labels")
_SKELETON = (("values", Mapping), ("sections", list), ("input_groups", list),
             ("code_notes", Mapping))
_FOCUS = {"outputs": "loss", "all": "optimizer", "edges": "optimizer"}
_STEPS = {"act": "activation", "softmax": "link", "output": "link"}
_RUN = re.compile(r"#experiments\.runs\[id=([^\]]+)\]\.")
# the words a v1 limit's title and text carry for each limit kind (lower case, as substrings)
_LIMIT_WORDS = {"split": ("split", "distribution", "partition"),
                "exposures": ("sealed", "exposure", "opened"), "one_seed": ("seed", "single run"),
                "robustness_not_run": ("robust",), "verdict_not_earned": ("earned", "verdict"),
                "verdict_beaten": ("beaten",),
                "verdict_not_assessable": ("assessable", "verdict"),
                "ship_stance": ("ship", "stance"),
                "roundtrip_boundary": ("round trip", "boundary"),
                "examples_synthetic": ("synthetic",), "calibration": ("calibrat", "probabilit"),
                "slices_empty": ("slice",), "drift_suspected": ("drift",),
                "serving_forced": ("forced", "demonstration"),
                "small_support": ("anecdot", "rare class", "handful", "few rows")}


def v2_ref(ref: str, roles_by_label: Mapping[str, str]) -> str | None:
    """A v1 value ref in the contract's grammar; None for a source or label it cannot place."""
    source, _, path = str(ref).partition(":")
    if source == "report":
        label, field = path, "number"
        head, dot, tail = path.rpartition(".")
        if dot and tail in _REPORT_FIELDS:
            label, field = head, tail
        role = roles_by_label.get(label)
        return f"report:{role}.{field}" if role else None
    return f"{_FILES[source]}#{path}" if source in _FILES and path else None


class V1Carry:
    """The authored slots of one `vega/story-plan@1`, in the draft's terms."""

    def __init__(self, ctx: CarryContext, plan: Mapping) -> None:
        for key, kind in _SKELETON:
            if plan.get(key) is not None and not isinstance(plan[key], kind):
                raise ValueError(f"not a vega/story-plan@1: {key} is a {type(plan[key]).__name__}")
        self._ctx, self._plan = ctx, plan
        self._sections = {s.get("id"): s for s in plan.get("sections") or ()
                          if isinstance(s, Mapping)}
        self._names = self._value_map()

    def authored(self) -> dict:
        """Every slot v1 can fill (a section that does not read as v1 wrote it is a `WARN:`)."""
        top = self._plan
        slots = {"headline": lambda _: self._text("authored.headline", top.get("headline")),
                 "lead": lambda _: self._text("authored.lead", top.get("lead")),
                 "context": lambda _: self._context(), "network": lambda _: self._network(),
                 "experiments": lambda _: self._experiments(),
                 "results": lambda _: self._results(),
                 "input_groups": lambda _: self._input_groups(),
                 "code_notes": lambda _: self._code_notes(),
                 "class_roles": lambda _: self._class_roles()}
        return carry_slots(self._ctx, slots, {slot: True for slot in slots})

    # ---------------------------------------------------------------- values → fact keys
    def _value_map(self) -> dict:
        by_ref: dict[str, list] = {}
        for fact in self._ctx.facts.facts:
            by_ref.setdefault(str(fact.ref), []).append(fact)
        roles = self._roles_by_label()
        names, lost = {}, []
        for name, ref in (self._plan.get("values") or {}).items():
            found = by_ref.get(v2_ref(ref, roles) or "")
            if found:
                names[str(name)] = min(found, key=lambda f: _KINDS.index(str(f.kind))).key
            else:
                lost.append(f"{name} ({ref})")
        if lost:
            self._ctx.warn("values", f"no fact for the v1 values {', '.join(lost)}; a text using "
                                     "one is not carried")
        return names

    def _roles_by_label(self) -> dict:
        emitted = (self._ctx.groups.get("report") or {}).get("roles") or {}
        proposed = (self._ctx.proposals.get("report_roles") or {}).get("bindings") or {}
        out = {}
        for bindings in (proposed, emitted):  # an emitted binding wins
            out.update({str(b.get("label")): role for role, b in bindings.items()
                        if isinstance(b, Mapping)})
        return out

    def _text(self, where: str, text) -> str | None:
        return self._ctx.text(where, text, self._names)

    # ---------------------------------------------------------------- sections
    def _context(self) -> dict | None:
        section = self._sections.get("context") or {}
        ids, steps = list(self._ctx.proposals.get("context_steps") or ()), {}
        for i, step in enumerate(section.get("steps") or ()):
            if i >= len(ids):
                self._ctx.warn("authored.context.steps", "the v1 steps after the proposed ones "
                                                         "are not carried")
                break
            text = self._text(f"authored.context.steps.{ids[i]}", (step or {}).get("text"))
            if text is not None:
                steps[ids[i]] = text
        facts = []
        for fact in section.get("facts") or ():
            key = self._names.get(fact.get("value"))
            label = self._text("authored.context.facts", fact.get("label"))
            if key is not None and label is not None:
                facts.append({"label": label, "fact": key})
            else:
                self._ctx.warn("authored.context.facts", f"the v1 fact {fact.get('value')!r} is "
                                                         "not carried")
        return {"steps": steps, "facts": facts} if steps or facts else None

    def _network(self) -> dict | None:
        section = self._sections.get("network") or {}
        lead = self._text("authored.network.lead", section.get("lead"))
        if lead is None:
            return None
        choices = []
        for i, choice in enumerate(section.get("choices") or ()):
            where, focus = f"authored.network.choices[{i}]", str(choice.get("focus"))
            said = f"{choice.get('title') or ''} {choice.get('text') or ''}".lower()
            decision = _decision_of(self._ctx, focus, said)
            if decision is None:
                self._ctx.warn(where, f"the v1 focus {focus!r} names no proposed decision; dropped")
                continue
            if decision in {c["decision"] for c in choices}:
                self._ctx.warn(where, f"a second choice on the decision {decision!r} (v1 focus "
                                      f"{focus!r}); the model may move it")
            texts = self._ctx.texts(where, choice, ("title", "text"), self._names)
            if texts is not None:
                choices.append({"decision": decision, "title": texts[0], "text": texts[1]})
        return {"lead": lead, "choices": choices}

    def _experiments(self) -> dict | None:
        section = self._sections.get("experiments") or {}
        texts = self._ctx.texts("authored.experiments", section, ("lead", "caption"),
                                self._names)
        if texts is None:
            return None
        runs, labels = set(self._ctx.ids("runs", "id")), {}
        for run in section.get("runs") or ():
            key = self._names.get(run.get("value"))
            found = _RUN.search(str(self._ctx.facts.get(key).ref)) if key else None
            if found is None or found.group(1) not in runs:
                self._ctx.warn("authored.experiments.labels", f"the v1 run {run.get('value')!r} "
                                                              "is no proposed run; not carried")
                continue
            text = self._text(f"authored.experiments.labels.{found.group(1)}", run.get("label"))
            if text is not None:
                labels[found.group(1)] = text
        return {"lead": texts[0], "caption": texts[1], "labels": labels}

    def _results(self) -> dict | None:
        section = self._sections.get("results") or {}
        headline = self._text("authored.results.headline", section.get("headline"))
        if headline is None:
            return None
        shown, notes = set(self._ctx.proposals.get("results_facts") or ()), {}
        for fact in (f for f in section.get("facts") or () if f.get("note") is not None):
            key = self._names.get(fact.get("value"))
            if key not in shown:
                self._ctx.warn("authored.results.notes", f"the v1 note on {fact.get('value')!r} "
                                                         "is on no proposed results fact; dropped")
            elif (note := self._text(f"authored.results.notes.{key}", fact["note"])) is not None:
                notes[key] = note
        unit = section.get("per_class") or section.get("per_unit") or {}
        caption = None
        if unit.get("caption") and self._ctx.groups.get("breakdown"):
            caption = self._text("authored.results.breakdown_caption", unit["caption"])
        return {"headline": headline, "notes": notes, "breakdown_caption": caption,
                "limits": self._limits(section.get("limits") or ())}

    def _limits(self, limits) -> list[dict]:
        free, out = [i for i in self._ctx.limit_ids if i in _LIMIT_WORDS], []
        for i, lim in enumerate(limits):
            words = f"{lim.get('title') or ''} {lim.get('text') or ''}".lower()
            scored = [(sum(w in words for w in _LIMIT_WORDS[lid]), -free.index(lid), lid)
                      for lid in free]
            best = max(scored, default=(0, 0, None))
            if best[0] == 0:
                self._ctx.warn(f"authored.results.limits[{i}]",
                               f"the v1 limit {lim.get('title')!r} names no limit kind of this "
                               "fact set; dropped")
            elif texts := self._ctx.texts(f"authored.results.limits[{i}]", lim,
                                          ("title", "text"), self._names):
                free.remove(best[2])
                out.append({"id": best[2], "title": texts[0], "text": texts[1]})
        return out

    def _input_groups(self) -> list[dict] | None:
        groups, placed, taken = [], set(), set()
        for i, group in enumerate(self._plan.get("input_groups") or ()):
            where = f"authored.input_groups[{i}]"
            patterns = _patterns(self._ctx, where, group.get("match") or ())
            members = [n for n in self._ctx.shaped
                       if n not in placed and any(p.search(n) for p in patterns)]
            texts = self._ctx.texts(where, group, ("label", "text"), self._names)
            if texts is not None:
                groups.append({"id": _group_id(texts[0], taken), "label": texts[0],
                               "text": texts[1], "members": members})
                placed.update(members)
        return self._ctx.partition(groups) if groups else None

    def _code_notes(self) -> dict | None:
        steps, out = set(self._ctx.ids("code_steps", "step")), {}
        for step, note in (self._plan.get("code_notes") or {}).items():
            target = step[len("block:"):] if step.startswith("block:") else _STEPS.get(step, step)
            where = f"authored.code_notes.{target}"
            if target not in steps or target in out:
                self._ctx.warn(where, f"the v1 note {step!r} is on no proposed walk step (or on "
                                      "one already noted); not carried")
                continue
            text = self._text(where, note.get("note") if isinstance(note, Mapping) else note)
            if text is not None:
                out[target] = text
        return out or None

    def _class_roles(self) -> dict | None:
        neutral, proposal = self._plan.get("neutral_class"), self._ctx.proposals.get("class_roles")
        if neutral is None:
            return None
        proposed = ((proposal or {}).get("neutral") or {}).get("class")
        if proposed is None or str(neutral) != proposed:
            self._ctx.warn("authored.class_roles", f"the v1 neutral class {neutral!r} is not the "
                                                   f"class the structural rule proposes "
                                                   f"({proposed!r}); not carried — fork S2 decides")
            return None
        return {"roles": dict(proposal["roles"]), "source": dict(proposal["source"])}


def _decision_of(ctx: CarryContext, focus: str, said: str = "") -> str | None:
    """The proposed decision a v1 choice's focus names (module docstring), or None. A `hidden`
    choice that speaks of dropout is the first dropout block's (v1 lit the hidden layers to show
    where dropout acts; the first dense block is not what it is about)."""
    decisions = [d for d in ctx.groups.get("decisions") or () if isinstance(d, Mapping)]
    proposed = set(ctx.ids("choices", "decision"))
    blocks = [d for d in decisions if str(d.get("path", "")).startswith("blocks[")]
    if focus.startswith("feature:"):
        name = focus[len("feature:"):]
        found = [d for d in decisions if name in (d.get("features_tested") or ())]
    elif focus in ("hidden", "activation"):
        kind = "dense" if focus == "hidden" else "activation"
        if focus == "hidden" and "dropout" in said and any(d.get("type") == "dropout" for d in blocks):
            kind = "dropout"
        found = [d for d in blocks if d.get("type") == kind]
    elif focus == "inputs":
        found = blocks
    else:
        found = [d for d in decisions if d.get("id") == _FOCUS.get(focus, focus)]
    return next((str(d["id"]) for d in found if str(d.get("id")) in proposed), None)


def _patterns(ctx: CarryContext, where: str, patterns) -> list[re.Pattern]:
    """The v1 group's regexes that compile; each other is a `WARN:`."""
    out = []
    for pattern in patterns:
        try:
            out.append(re.compile(str(pattern)))
        except re.error as exc:
            ctx.warn(where, f"the v1 regex {pattern!r} does not compile ({exc}); skipped")
    return out


def _group_id(label: str, taken: set) -> str:
    """A group id from its v1 label, matching the id pattern and unique."""
    pattern = re.compile(CATALOG["input_groups"]["id_pattern"])
    base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:28]
    base = base if base[:1].isalpha() else f"g_{base}".rstrip("_")
    gid, n = base, 2
    while gid in taken or gid == UNPLACED or not pattern.fullmatch(gid):
        gid, n = f"{base}_{n}", n + 1
    taken.add(gid)
    return gid
