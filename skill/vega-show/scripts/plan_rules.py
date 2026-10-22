"""
plan_rules — `PlanValidator`, the rules V1–V15 an authored story plan (`vega/story-plan@2`)
passes before `resolve_story.py` turns it into `vega/story@2` (contract §3.8), and the block
checks the story's blocks are held to — lexicon, class roles (`plan_roles`, re-exported here),
input groups, choices and their focus (R-E9: one owner; the digit rule and the placeholder
grammar are `digit_rule.py`'s, the text walk and the text rules `plan_texts.py`'s).

    PlanValidator(facts, PlanPins(facts_digest, template), labels=None).validate(plan)
        -> [Problem]          each ERROR: V<n> <where>: <why>, and WARN: for number words

`facts_digest` is the sha256 of the `facts.json` read, `template` the installed template's
`page_template.digests()`, `labels` the report's `.aux` (`LabelSource`) when its roles are
proposed. The closed sets come from the fact set and the catalog, never from the plan's own
proposals where the facts settle them (modality, classes, inputs, limits), so a hand-edited
proposal widens nothing. A rule refuses:

- V1 a plan drafted from another fact set, or against another page template;
- V2–V4, V12 the text rules (`plan_texts`): a stray digit, a placeholder or named fact that is
  none, a record path, a length out of bounds (and a context of other than 2–6 facts);
- V5 a locale, motif or lexicon outside its closed set (required keys, values, articles);
- V6 class roles (`plan_roles`): coverage, the neutral bound, a basis that does not grant its
  role, a structural basis off its class, the anomalous side downgraded, roles without classes;
- V7 input groups that do not partition the shaped inputs, out of the count bounds, with an id
  that repeats or breaks its pattern — the carry's `_unplaced` group included;
- V8 a choice on no proposed decision or on one that maps to no focus, a mandatory choice
  missing, a count out of bounds (fewer only when fewer decisions exist);
- V9 a limit of no known id or said twice, a mandatory limit not said; V10 run labels not keyed
  by exactly the proposed runs; V11 a code note on a step the draft did not propose;
- V13 a status other than `authored`, a required slot null (class roles on a task with classes,
  report roles while proposed), a text slot without text, a field the schema requires missing or
  a container of another type (`plan_texts.shape_problems`), context steps other than the lane's;
- V14 report roles authored while not proposed, outside the catalog, or on a label the `.aux`
  lacks or holds with another kind than the role's; V15 a claim the facts contradict (`plan_claims`).

A slot so malformed that a rule cannot read it is that rule's refusal, never a traceback."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping

from carry import UNPLACED
from digit_rule import strip_placeholders
from plan_claims import claim_problems
from plan_texts import STORY_SCHEMA, TextRules, authored_texts, shape_problems
from plan_roles import check_class_roles, covered_classes
from propose_lexicon import shape_key
from show_ports import LabelSource
from vega_show_core import CATALOG, HERE, FactSet, Problem, thaw

__all__ = ["PlanPins", "PlanValidator", "check_motif", "check_lexicon", "check_class_roles",
           "check_input_groups", "check_choices", "focus_of", "covered_classes", "group_bounds"]

_PLAN_SCHEMA = json.loads((HERE.parent / CATALOG["schemas"]["vega/story-plan@2"]["file"]).read_text("utf-8"))
REQUIRED_SLOTS = tuple(_PLAN_SCHEMA["allOf"][0]["then"]["properties"]["authored"]["properties"])
_BLOCK_REF = re.compile(STORY_SCHEMA["$defs"]["focus"]["oneOf"][0]["properties"]["ref"]["pattern"])
_BLOCK_PATH = "blocks["


@dataclass(frozen=True)
class PlanPins:
    """What V1 compares the plan's pins with: the digest of the fact set read, and the
    installed page template's `{name, version, manifest_digest}` (`page_template.digests`)."""

    facts_digest: str
    template: Mapping


@dataclass(frozen=True)
class _View:
    plan: Mapping
    authored: Mapping
    proposals: Mapping
    texts: tuple

    @classmethod
    def of(cls, plan) -> "_View":
        plan = plan if isinstance(plan, Mapping) else {}
        authored, proposals = plan.get("authored"), plan.get("proposals")
        authored = authored if isinstance(authored, Mapping) else {}
        return cls(plan, authored, proposals if isinstance(proposals, Mapping) else {},
                   tuple(authored_texts(authored)))


class PlanValidator:
    """V1–V15 over one authored plan, against one fact set, template and report."""

    RULES = tuple(f"V{n}" for n in range(1, 16))

    def __init__(self, facts: FactSet, pins: PlanPins, labels: LabelSource | None = None,
                 catalog: Mapping = CATALOG) -> None:
        self._facts, self._pins, self._labels, self._cat = facts, pins, labels, catalog
        self._groups = thaw(facts.groups)
        self._texts = TextRules(facts, catalog)

    def validate(self, plan) -> list[Problem]:
        view, out = _View.of(plan), []
        for rule in self.RULES:
            try:
                out += getattr(self, f"_{rule.lower()}")(view)
            except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
                out.append(Problem(rule, "plan", "the plan does not read as vega/story-plan@2 "
                                                 f"says ({type(exc).__name__}: {exc})"))
        return out

    def _v1(self, v: _View) -> list[Problem]:
        out, redraft = [], "draft the plan again (draft_plan.py --carry keeps the authored text)"
        recorded = (v.plan.get("facts") or {}).get("digest")
        if recorded != self._pins.facts_digest:
            out.append(Problem("V1", "facts.digest", f"the plan was drafted from another fact "
                               f"set ({recorded}; read: {self._pins.facts_digest}) — {redraft}"))
        pinned, now = (v.plan.get("template") or {}).get("manifest_digest"), \
            self._pins.template.get("manifest_digest")
        if pinned != now:
            out.append(Problem("V1", "template.manifest_digest", f"the plan pins another page "
                               f"template ({pinned}; the installed one is {now}) — {redraft}"))
        return out

    def _v2(self, v: _View) -> list[Problem]:
        return self._texts.digits(v.texts)

    def _v3(self, v: _View) -> list[Problem]:
        return self._texts.placeholders(v.texts, v.authored,
                                        [str(k) for k in v.proposals.get("results_facts") or ()])

    def _v4(self, v: _View) -> list[Problem]:
        return self._texts.record_paths(v.texts)

    def _v5(self, v: _View) -> list[Problem]:
        a, task, out = v.authored, self._facts.task, []
        if a.get("locale") is not None and a["locale"] not in self._cat["locales"]:
            out.append(Problem("V5", "authored.locale", f"{a['locale']!r} is not a locale "
                               f"of the catalog ({', '.join(self._cat['locales'])})"))
        out += check_motif(a.get("motif"), task.modality, self._cat)
        if a.get("lexicon") is not None:
            out += check_lexicon(a["lexicon"], shape_key(task), self._cat)
        return out

    _v15 = lambda self, v: claim_problems(v.authored, self._groups, self._facts.task)  # noqa: E731

    def _v6(self, v: _View) -> list[Problem]:
        return check_class_roles(v.authored.get("class_roles"), self._facts.task,
                                 v.proposals.get("class_roles"), self._cat)

    def _v7(self, v: _View) -> list[Problem]:
        shaped = [str(n) for n in (self._groups.get("inputs") or {}).get("shaped") or ()]
        return check_input_groups(v.authored.get("input_groups"), shaped, self._cat)

    def _v8(self, v: _View) -> list[Problem]:
        return check_choices(v.authored.get("network"), v.proposals.get("choices") or [],
                             self._groups.get("decisions") or [], self._cat)

    def _v9(self, v: _View) -> list[Problem]:
        res, where = v.authored.get("results"), "authored.results.limits"
        if not isinstance(res, Mapping):
            return []
        limits = res.get("limits")
        out, said = [] if isinstance(limits, list) else [
            Problem("V9", where, "the limits are a list of {id, title, text}")], []
        known = {str(lim["id"]): lim for lim in self._groups.get("limits") or ()}
        for i, lim in enumerate(limits if isinstance(limits, list) else ()):
            lid = lim.get("id") if isinstance(lim, Mapping) else None
            if lid not in known:
                out.append(Problem("V9", f"{where}[{i}].id", f"{lid!r} is not a limit of this lane "
                                                             f"({', '.join(known)})"))
            elif lid in said:
                out.append(Problem("V9", f"{where}[{i}].id", f"the limit {lid!r} is said twice"))
            said.append(lid)
        return out + [Problem("V9", where, f"the mandatory limit {lid!r} is not said")
                      for lid, lim in known.items() if lim.get("mandatory") and lid not in said]

    def _v10(self, v: _View) -> list[Problem]:
        exp, where = v.authored.get("experiments"), "authored.experiments.labels"
        if not isinstance(exp, Mapping):
            return []
        labels = exp.get("labels")
        if not isinstance(labels, Mapping):
            return [Problem("V10", where, "the run labels are an object {run id: label}")]
        runs = [str(r.get("id")) for r in v.proposals.get("runs") or () if isinstance(r, Mapping)]
        return [Problem("V10", where, f"the run {r!r} has no label; every proposed run has one")
                for r in runs if r not in labels] + \
            [Problem("V10", f"{where}.{r}", f"{r!r} is not a proposed run ({', '.join(runs)})")
             for r in labels if r not in runs]

    def _v11(self, v: _View) -> list[Problem]:
        notes = v.authored.get("code_notes")
        steps = [str(s.get("step")) for s in v.proposals.get("code_steps") or ()
                 if isinstance(s, Mapping)]
        return [Problem("V11", f"authored.code_notes.{k}", f"{k!r} is not a proposed walk step "
                                                          f"({', '.join(steps)})")
                for k in (notes if isinstance(notes, Mapping) else {}) if k not in steps]

    def _v12(self, v: _View) -> list[Problem]:
        return self._texts.lengths(v.texts, v.authored)

    def _v13(self, v: _View) -> list[Problem]:
        a, out, status = v.authored, [], v.plan.get("status")
        if status != "authored":
            out.append(Problem("V13", "status", f"the plan's status is {status!r}; the model sets "
                                                "it to 'authored' once every slot is filled"))
        out += [Problem("V13", f"authored.{s}", "a required slot is null") for s in REQUIRED_SLOTS
                if a.get(s) is None]
        if covered_classes(self._facts.task, self._cat) and a.get("class_roles") is None:
            out.append(Problem("V13", "authored.class_roles", "a task with classes needs its "
                                                              "class roles (fork S2)"))
        if (v.proposals.get("report_roles") or {}).get("status") == "proposed" and \
                a.get("report_roles") is None:
            out.append(Problem("V13", "authored.report_roles", "the report's roles are proposed: "
                               "the model confirms or edits them here ({} confirms none)"))
        out += [Problem("V13", t.where, "a text slot holds no text") for t in v.texts
                if not isinstance(t.text, str)] + shape_problems(a)
        return out + self._context_steps(a.get("context"))

    def _context_steps(self, ctx) -> list[Problem]:
        if not isinstance(ctx, Mapping):
            return []
        shape = self._facts.task.sample_shape
        want = self._cat["context_steps"]["by_shape"][shape]
        steps = ctx.get("steps") if isinstance(ctx.get("steps"), Mapping) else {}
        return [Problem("V13", f"authored.context.steps.{s}", "a context step of this lane has "
                                                              "no text") for s in want
                if s not in steps] + \
            [Problem("V13", f"authored.context.steps.{s}", f"{s!r} is not a context step of a "
                                                           f"{shape} lane ({', '.join(want)})")
             for s in steps if s not in want]

    def _v14(self, v: _View) -> list[Problem]:
        roles, where = v.authored.get("report_roles"), "authored.report_roles"
        if roles is None:
            return []
        status = (v.proposals.get("report_roles") or {}).get("status")
        if status != "proposed":
            return [Problem("V14", where, f"the report's roles are {status}, not proposed: report "
                                          "roles are authored only for a report without a role "
                                          "map — leave it null")]
        if not isinstance(roles, Mapping):
            return [Problem("V14", where, "report roles are {role: label}")]
        if self._labels is None:
            return [Problem("V14", where, "the report's .aux is not at hand to check the labels "
                                          "against (resolve_story.py --report)")]
        return [p for role, label in roles.items() if (p := self._role(role, label))]

    def _role(self, role: str, label) -> Problem | None:
        spec, where = self._cat["report_roles"].get(role), f"authored.report_roles.{role}"
        found = self._labels.labels().get(label) if isinstance(label, str) else None
        if spec is None:
            return Problem("V14", where, f"{role!r} is not a report role of the catalog")
        if found is None:
            return Problem("V14", where, f"the label {label!r} is not in the report's .aux")
        if str(found.kind) != spec["kind"]:
            return Problem("V14", where, f"the label {label!r} is a {found.kind} in the .aux; the "
                                         f"role {role} takes a {spec['kind']}")
        return None


# =========================================================================== the block checks

def check_motif(motif, modality: str, catalog: Mapping = CATALOG) -> list[Problem]:
    """V5: the motif is one of the closed set and admissible for the lane's modality."""
    if motif is None:
        return []
    spec, closed, where = catalog["motifs"]["by_modality"][modality], catalog["motifs"]["set"], \
        "authored.motif"
    if motif not in closed:
        return [Problem("V5", where, f"the motif {motif!r} is not one of the closed set "
                                     f"({', '.join(closed)})")]
    if motif not in spec["admissible"]:
        return [Problem("V5", where, f"the motif {motif!r} is not admissible for the {modality} "
                                     f"modality ({', '.join(spec['admissible'])}; base "
                                     f"{spec['base']})")]
    return []


def check_lexicon(lexicon, shape: str, catalog: Mapping = CATALOG) -> list[Problem]:
    """V5: `lexicon@1` — every key the shape requires, no unknown key, each entry exactly
    `{one, many, article}` with lexicon values and an article its key takes."""
    spec, where = catalog["lexicon"], "authored.lexicon"
    if not isinstance(lexicon, Mapping):
        return [Problem("V5", where, "the lexicon is an object of entries {one, many, article}")]
    required = spec["required"][shape]
    out = [Problem("V5", f"{where}.{k}", f"the lexicon has no {k!r} entry; a {shape} lane "
                                         f"requires {', '.join(required)}")
           for k in required if k not in lexicon]
    for key, entry in lexicon.items():
        if key not in spec["keys"]:
            out.append(Problem("V5", f"{where}.{key}", f"{key!r} is not a lexicon key "
                                                       f"({', '.join(spec['keys'])})"))
        else:
            out += _entry_problems(f"{where}.{key}", key, entry, spec)
    return out


def _entry_problems(where: str, key: str, entry, spec: Mapping) -> list[Problem]:
    if not isinstance(entry, Mapping) or set(entry) != set(spec["entry"]):
        return [Problem("V5", where, f"a lexicon entry is exactly {{{', '.join(spec['entry'])}}}")]
    pattern = re.compile(spec["value_pattern"])
    out = [Problem("V5", f"{where}.{f}", f"{entry[f]!r} is not a lexicon value "
                                         f"({spec['value_pattern']}: lower case, no digit)")
           for f in ("one", "many") if not (isinstance(entry[f], str)
                                             and pattern.fullmatch(entry[f]))]
    articles = spec["articles"].get(key, spec["articles"]["default"])
    if entry["article"] not in articles:
        out.append(Problem("V5", f"{where}.article", f"the article {entry['article']!r} is not "
                           f"one the {key} entry takes ({', '.join(articles)})"))
    return out


def group_bounds(n_inputs: int, catalog: Mapping = CATALOG) -> tuple[int, int]:
    """(fewest, most) input groups for `n_inputs` shaped inputs (contract §3.7)."""
    spec = catalog["input_groups"]
    if n_inputs >= spec["min_applies_from_inputs"]:
        return spec["min"], spec["max"]
    return 1, max(1, n_inputs)


def check_input_groups(groups, shaped: list[str], catalog: Mapping = CATALOG) -> list[Problem]:
    """V7: the groups partition the shaped inputs exactly, within the count bounds, with unique
    ids of the pattern; the carry's `_unplaced` group is refused until its inputs are placed."""
    where, id_re = "authored.input_groups", re.compile(catalog["input_groups"]["id_pattern"])
    if groups is None:
        return []
    if not isinstance(groups, list) or not all(isinstance(g, Mapping) for g in groups):
        return [Problem("V7", where, "input groups are a list of {id, label, text, members}")]
    known, out, ids, placed = set(shaped), [], set(), {}
    for i, group in enumerate(groups):
        gid, at, members = group.get("id"), f"{where}[{i}]", group.get("members")
        if gid == UNPLACED:
            out.append(Problem("V7", at, f"the group {UNPLACED!r} holds the inputs a carry could "
                                         f"not place ({members}); place each in a group"))
        elif not isinstance(gid, str) or not id_re.fullmatch(gid) or gid in ids:
            why = "is used twice" if gid in ids else f"does not match {id_re.pattern}"
            out.append(Problem("V7", f"{at}.id", f"the group id {gid!r} {why}"))
        ids.add(gid)
        members = members if isinstance(members, list) else []
        if not members:
            out.append(Problem("V7", f"{at}.members", "a group holds at least one input"))
        for name in members:
            if name not in known:
                out.append(Problem("V7", f"{at}.members", f"{name!r} is not a network input"))
            elif name in placed:
                out.append(Problem("V7", f"{at}.members", f"{name!r} is in {placed[name]!r} and "
                                   f"in {gid!r}; every input is in exactly one group"))
            else:
                placed[name] = gid
    missing = [n for n in shaped if n not in placed]
    if missing:
        out.append(Problem("V7", where, f"{missing} in no group; every input is in exactly one"))
    low, high = group_bounds(len(shaped), catalog)
    if not low <= len(groups) <= high:
        out.append(Problem("V7", where, f"{len(groups)} groups for {len(shaped)} inputs; the "
                                        f"bounds are {low} to {high} groups"))
    return out


def check_choices(network, proposed: list, decisions: list, catalog: Mapping = CATALOG
                  ) -> list[Problem]:
    """V8: every choice on a proposed decision that maps to a focus kind, every mandatory one
    present, the count within `choices.min`–`max` (fewer only when fewer are proposed)."""
    where = "authored.network.choices"
    choices = network.get("choices") if isinstance(network, Mapping) else None
    if not isinstance(choices, list):
        return [] if network is None else [Problem("V8", where, "the choices are a list of "
                                                                "{decision, title, text}")]
    offered = {str(c.get("decision")): c for c in proposed if isinstance(c, Mapping)}
    by_id = {str(d.get("id")): d for d in decisions if isinstance(d, Mapping)}
    out, named = [], set()
    for i, choice in enumerate(choices):
        did = choice.get("decision") if isinstance(choice, Mapping) else None
        named.add(did)
        if did not in offered:
            out.append(Problem("V8", f"{where}[{i}].decision", f"the decision {did!r} is not a "
                               f"proposed decision ({', '.join(offered)})"))
        elif focus_of(by_id.get(did, {"id": did}), _said(choice), catalog) is None:
            out.append(Problem("V8", f"{where}[{i}].decision", f"the decision {did!r} maps to no "
                               f"focus kind ({', '.join(catalog['focus']['kinds'])})"))
    out += [Problem("V8", where, f"the mandatory choice {d!r} is missing")
            for d, c in offered.items() if c.get("mandatory") and d not in named]
    low, high = min(catalog["choices"]["min"], len(offered)), catalog["choices"]["max"]
    if not low <= len(choices) <= high:
        out.append(Problem("V8", where, f"{len(choices)} choices; the network section holds "
                                        f"{low} to {high}"))
    return out


def _said(choice: Mapping) -> str:
    return " ".join(t for t in (choice.get("title"), choice.get("text")) if isinstance(t, str))


def focus_of(decision: Mapping, text: str, catalog: Mapping = CATALOG) -> dict | None:
    """`focus@1` of a choice (contract §4.2): an input the decision's runs tested that the text
    names; else the block, for a block decision; else its decision family; None when none fits."""
    did, bare = str(decision.get("id")), strip_placeholders(text or "")
    for name in decision.get("features_tested") or ():
        if re.search(rf"(?<!\w){re.escape(str(name))}(?!\w)", bare, re.IGNORECASE):
            return {"kind": "feature", "ref": str(name)}
    if str(decision.get("path", "")).startswith(_BLOCK_PATH):
        return {"kind": "block", "ref": did} if _BLOCK_REF.fullmatch(did) else None
    family = catalog["focus"]["by_decision"].get(did.split("[", 1)[0])
    return {"kind": "family", "ref": family} if family else None
