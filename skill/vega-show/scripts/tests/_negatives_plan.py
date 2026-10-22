"""
_negatives_plan — `run_negatives.py`'s cases on the authored plan (gates V1–V15) and on fork S1:
each seeds one defect into a copy of the staged lane's `plan/story-plan.json` (or its fact set)
and runs `resolve_story.py` as the stage runs it. L02 (multiclass, a linked report) carries the
seeds; VS06 needs a task without classes (L05, regression); fork S1 needs a lane without a report
(L03).
"""
from __future__ import annotations

from _negatives_kit import (Check, Expect, case, classes_of, edit_json, read_json, refused, seeded_plan,
                            shaped_inputs, write_json)

BASE = "L02"


def resolve_seeded(w, mutate, *words, gate: str):
    """Resolve a seeded copy of the plan: refused by `gate` alone, naming `words`, nothing written."""
    out = w.plan / "seeded-story.json"
    rc, printed = w.resolve(plan=seeded_plan(w, mutate), out=out)
    others = sorted({line.split()[1] for line in printed.splitlines()
                     if line.startswith("ERROR: ") and len(line.split()) > 1} - {gate, "resolve"})
    if rc == 2 and (others or out.exists()):
        return Check(False, f"refused by {others} besides {gate}" if others else "a story was written", printed)
    return refused((rc, printed), gate, *words)


def _authored(**slots):
    return lambda p: p["authored"].update(**slots)


def _choice(**fields):
    return lambda p: p["authored"]["network"]["choices"][0].update(**fields)


def _roles(plan, **pairs):
    for name, (role, basis) in pairs.items():
        plan["authored"]["class_roles"]["roles"][name] = role
        plan["authored"]["class_roles"]["source"][name] = basis


def _without_class(plan, name):
    for part in ("roles", "source"):
        plan["authored"]["class_roles"][part].pop(name)


def _drop(items: list, key: str, value) -> None:
    items[:] = [i for i in items if i.get(key) != value]


def _drop_input(plan) -> str:
    """One shaped input taken out of its group (the group dropped when it empties); its name."""
    groups = plan["authored"]["input_groups"]
    group = max(groups, key=lambda g: len(g["members"]))
    name = group["members"].pop()
    groups[:] = [g for g in groups if g["members"]]
    return name


def _groups_of(n: int, shaped: list) -> list:
    size, extra = divmod(len(shaped), n)
    out, at = [], 0
    for i in range(n):
        take = size + (1 if i < extra else 0)
        out.append({"id": f"group_{chr(97 + i)}", "label": f"group {chr(97 + i)}", "text": "",
                    "members": shaped[at:at + take]})
        at += take
    return out


# =============================================================================== V1–V15

@case("V1", "D5-21", BASE, "the fact set edited after the draft")
def v1(w):
    facts = read_json(w.plan / "facts.json")
    key = next(k for k, f in facts["facts"].items() if isinstance(f.get("value"), int) and not isinstance(f["value"], bool))
    facts["facts"][key]["value"] += 1
    write_json(w.plan / "facts.edited.json", facts)
    return refused(w.resolve(facts=w.plan / "facts.edited.json", out=w.plan / "seeded-story.json"), "V1", "facts")


@case("V2", "D5-1 VS01", BASE, "a digit in the headline ('the network reads 12 bands')")
def v2_headline(w):
    return resolve_seeded(w, _authored(headline="the network reads 12 bands"), "digit", "headline", gate="V2")


@case("V2", "D5-2", BASE, "'top7', which is no name the facts carry, in a choice")
def v2_top7(w):
    return resolve_seeded(w, _choice(text="It ranks the top7 first."), "top7", gate="V2")


@case("V2", "D5-3", BASE, "a digit after a Greek letter ('β2') that is not a name")
def v2_beta(w):
    return resolve_seeded(w, _choice(title="The β2 setting"), "β2", gate="V2")


@case("V3", "D5-4 VS02", BASE, "a {nonexistent} placeholder")
def v3(w):
    return resolve_seeded(w, _authored(lead="It reads {nonexistent} things."), "placeholder", "nonexistent", gate="V3")


@case("V4", "D5-5", BASE, "a record path (records/shape.json#…) typed in the text")
def v4_path(w):
    return resolve_seeded(w, lambda p: p["authored"]["context"]["steps"].update(
        cleaning="See records/shape.json#manifest.steps for how."), "record path", gate="V4")


@case("V4", "VS03", BASE, "an authored record path (evaluation:metrics.models…) instead of a fact key")
def v4_record(w):
    return resolve_seeded(w, lambda p: p["authored"]["context"]["steps"].update(
        data="The rows are evaluation:metrics.models.network, as recorded."), "fact", gate="V4")


@case("V5", "D5-6 VS08", BASE, "motif traces on a row lane")
def v5_traces(w):
    return resolve_seeded(w, _authored(motif="traces"), "motif", "modality", gate="V5")


@case("V5", "VS07", BASE, "motif 'packets', outside the closed set")
def v5_packets(w):
    return resolve_seeded(w, _authored(motif="packets"), "motif", "grid, points, traces, lanes, links, contours",
                          gate="V5")


@case("V5", "VS09", BASE, "a lexicon without the row noun")
def v5_no_row(w):
    return resolve_seeded(w, lambda p: p["authored"]["lexicon"].pop("row"), "row", gate="V5")


@case("V5", "VS10", BASE, "a lexicon with an unknown key")
def v5_unknown_key(w):
    return resolve_seeded(w, lambda p: p["authored"]["lexicon"].update(
        noun={"one": "flow", "many": "flows", "article": "a"}), "noun", gate="V5")


@case("V5", "PN6", BASE, "a lexicon value 'flow2'")
def v5_flow2(w):
    return resolve_seeded(w, lambda p: p["authored"]["lexicon"]["row"].update(one="flow2"), "flow2", gate="V5")


@case("V6", "D5-7 VS05", BASE, "a class missing from the roles")
def v6_missing(w):
    gone = classes_of(w)[-1]
    return resolve_seeded(w, lambda p: _without_class(p, gone), "every class", gate="V6")


@case("V6", "D5-8", BASE, "two neutral classes")
def v6_two_neutral(w):
    a, b = classes_of(w)[:2]
    return resolve_seeded(w, lambda p: _roles(p, **{a: ("neutral", "user:turn-1"), b: ("neutral", "user:turn-1")}),
                          "neutral", gate="V6")


@case("V6", "D5-9", BASE, "alert with the basis structural:majority")
def v6_alert_majority(w):
    last = classes_of(w)[-1]
    return resolve_seeded(w, lambda p: _roles(p, **{last: ("alert", "structural:majority")}), "alert", gate="V6")


@case("V6", "VS04", BASE, "a role for a class not in task.classes")
def v6_bogus(w):
    return resolve_seeded(w, lambda p: _roles(p, Bogus=("plain", "vega:default")), "Bogus", gate="V6")


@case("V6", "PN5", BASE, "class_roles with alert and source vega:default")
def v6_alert_default(w):
    last = classes_of(w)[-1]
    return resolve_seeded(w, lambda p: _roles(p, **{last: ("alert", "vega:default")}), "alert", gate="V6")


@case("V6", "VS06", "L05", "an alert role on a regression lane")
def v6_regression(w):
    def seed(p):
        p["authored"]["class_roles"] = {"roles": {"high": "alert"}, "source": {"high": "user:turn-1"}}
    return resolve_seeded(w, seed, "alert", "classes", gate="V6")


@case("V7", "D5-10 VS11", BASE, "an input family member duplicated")
def v7_duplicate(w):
    def seed(p):
        groups = p["authored"]["input_groups"]
        groups[0]["members"].append(groups[1]["members"][0])
    return resolve_seeded(w, seed, "exactly one", gate="V7")


@case("V7", "D5-11 VS12 PN12", BASE, "an input missing from the groups")
def v7_missing(w):
    plan = read_json(w.plan / "story-plan.json")
    name = _drop_input(plan)
    return resolve_seeded(w, lambda p: p.update(authored=plan["authored"]), repr(name), gate="V7")


@case("V7", "D5-12", BASE, "ten input families")
def v7_ten(w):
    shaped = shaped_inputs(w)
    return resolve_seeded(w, _authored(input_groups=_groups_of(10, shaped)), "groups", gate="V7")


@case("V7", "VS13", BASE, "two input groups, under the bound")
def v7_two(w):
    shaped = shaped_inputs(w)
    return resolve_seeded(w, _authored(input_groups=_groups_of(2, shaped)), "groups", gate="V7")


@case("V8", "D5-13", BASE, "a choice on an unknown decision")
def v8_unknown(w):
    return resolve_seeded(w, lambda p: p["authored"]["network"]["choices"].append(
        {"decision": "B99", "title": "Another block", "text": "It does something."}), "B99", gate="V8")


@case("V8", "D5-14", BASE, "the mandatory loss choice dropped")
def v8_loss(w):
    return resolve_seeded(w, lambda p: _drop(p["authored"]["network"]["choices"], "decision", "loss"), "loss",
                          gate="V8")


@case("V9", "D5-15", BASE, "the exposures limit dropped")
def v9(w):
    return resolve_seeded(w, lambda p: _drop(p["authored"]["results"]["limits"], "id", "exposures"), "exposures",
                          gate="V9")


@case("V10", "D5-16", BASE, "a run label for a run the draft did not propose")
def v10(w):
    return resolve_seeded(w, lambda p: p["authored"]["experiments"]["labels"].update(R99="a run"), "R99", gate="V10")


@case("V11", "D5-17", BASE, "a code note on an unknown walk step")
def v11(w):
    return resolve_seeded(w, lambda p: p["authored"]["code_notes"].update(nowhere="A note."), "nowhere", gate="V11")


@case("V12", "D5-18", BASE, "a headline of 400 characters")
def v12_headline(w):
    return resolve_seeded(w, _authored(headline="a" * 400), "headline", gate="V12")


@case("V12", "VS23", BASE, "a choice title above its length bound")
def v12_title(w):
    return resolve_seeded(w, _choice(title="a" * 61), "length", gate="V12")


@case("V13", "D5-19", BASE, "a plan left at status draft")
def v13(w):
    return resolve_seeded(w, lambda p: p.update(status="draft"), "status", gate="V13")


def _proposed_roles(w):
    """The report kept, its role map gone: the draft proposes the roles from the .aux titles
    (`WARN: label roles proposed …`) and the authored plan confirms them (V14's ground)."""
    roles = read_json(w.docs / "label-roles.json")["roles"]
    (w.docs / "label-roles.json").unlink()
    w.stage(steps=("facts", "draft", "author"))
    return roles


@case("V14", "D5-20", BASE, "an authored report role whose label is of another kind")
def v14_kind(w):
    roles = _proposed_roles(w)
    eq = next(r for r, e in roles.items() if e["kind"] == "equation")
    other = next(e["label"] for e in roles.values() if e["kind"] != "equation")
    return resolve_seeded(w, lambda p: p["authored"]["report_roles"].update({eq: other}), eq, gate="V14")


@case("V14", "VS14", BASE, "a report role the catalog does not hold")
def v14_role(w):
    label = next(iter(_proposed_roles(w).values()))["label"]
    return resolve_seeded(w, lambda p: p["authored"]["report_roles"].update({"net.bogus": label}), "role",
                          gate="V14")


# =============================================================================== fork S1

def _limit_text(lid: str, text: str):
    def mutate(p):
        for lim in p["authored"]["results"]["limits"]:
            if lim["id"] == lid:
                lim["text"] = text
    return mutate


@case("V15", "GC-1", BASE, "a beaten verdict softened: 'could not separate … does as well'")
def v15_verdict(w):
    return resolve_seeded(w, _limit_text("verdict_not_earned", "The evaluation could not separate the network "
                                         "from the strongest baseline: a cheaper model does as well."),
                          "beaten", gate="V15")


@case("V15", "GC-2", BASE, "a per-class breakdown captioned as the primary metric")
def v15_caption(w):
    return resolve_seeded(w, lambda p: p["authored"]["results"].update(
        breakdown_caption="The primary metric unit by unit on the same test part."), "primary", gate="V15")


@case("V15", "GC-3", BASE, "the loss titled and described as a block")
def v15_block(w):
    def mutate(p):
        for c in p["authored"]["network"]["choices"]:
            if c["decision"] == "loss":
                c.update(title="The block", text="It is a block of the network.")
    return resolve_seeded(w, mutate, "loss", gate="V15")


@case("V15", "GC-4", BASE, "two input groups with one label")
def v15_groups(w):
    def mutate(p):
        groups = p["authored"]["input_groups"]
        groups[1]["label"] = groups[0]["label"]
    return resolve_seeded(w, mutate, "label", gate="V15")


@case("S1", "VS16", "L03", "no report and no --unattended: the build asks and writes nothing")
def s1_attended(w):
    import shutil
    shutil.rmtree(w.showcase)
    rc, out = w.build(unattended=False)
    if w.showcase.exists():
        return Check(False, "a showcase was written before the fork was answered", out)
    return Expect(rc, out, 2, ("FORK:",))


@case("S1", "VS17", "L03", "no report with --unattended: built, WARN, the code notes link to package files")
def s1_unattended(w):
    import shutil
    shutil.rmtree(w.showcase)
    rc, out = w.build()
    if rc != 0:
        return Check(False, f"exit {rc}", out)
    story = read_json(w.showcase / "story.json")
    doc = read_json(w.showcase / "SHOWCASE.json")
    links = [(n or {}).get("link") or {} for n in (story.get("code_notes") or {}).values()]
    hrefs = [l.get("href") for l in links if l.get("href")]
    files = doc.get("files") or {}
    outside = [h for h in hrefs if not h.lstrip("/").startswith("showcase/")
               or h.lstrip("/")[len("showcase/"):].split("#")[0] not in files]
    problems = ([] if "WARN:" in out else ["no WARN: line"]) + \
        ([] if doc["report"]["status"] == "brief-links" else [f"report.status {doc['report']['status']}"]) + \
        ([] if hrefs else ["no code note links"]) + [f"link {h} is not a pinned showcase file" for h in outside]
    return Check(not problems, "; ".join(problems), out)


CASES = [v for v in list(globals().values()) if type(v).__name__ == "Case"]
