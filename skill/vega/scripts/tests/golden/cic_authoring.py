"""
cic_authoring — what the golden CIC lane (G00) authors, in the places a model authors it in the
chain (BUILD-PLAN R2, D6 §11.5). Nothing here is a number: every figure is the records', and the
prose is either carried from what a person wrote for the v2.2 bundle or the lane program's own.

    label_roles(cids, docs, roles_file, show_scripts, lane="G00") -> dict
        docs/{main.aux, main.pdf, main.tex} copied from $CIDS_ROOT/docs, and docs/label-roles.json
        (vega/label-roles@1) binding the vega-show report roles to the real report's labels
        (golden/cic-report-roles.json), each with its number, page and title read from the .aux
    pack_layer(bundle, old_layer, from_bundle) -> dict
        the bundle layer's authoring: the hand-written intended use of the v2.2 bundle carried
        verbatim (it is about the network, which is the one the evaluation measured), and an
        abstract that says what this release rebuilt and what proves it
    write_stances(memory, lane) -> [line]
        the user's two stances (`show: motif lanes`, `show: roles alert = every class but
        BENIGN`) as § Network stances lines, so vega-show's draft finds them (R-E5)
    story_plan(draft, facts, lane) -> (dict, [edit])
        `vega-show --carry`'s draft completed: every slot the carry filled from the hand-authored
        v1 plan is kept; the slots v1 had no notion of (locale, lexicon, motif, class roles from
        the stances, report roles, monitor note), the ones the carry could not fill and the
        mandatory choices and limits v1 never had are the lane program's
        (lanekit.authoring.show_authoring); inputs the carry left `_unplaced` join the group the
        draft proposed for them; the carried prose that breaks this release's story contract is
        edited as little as it takes (CARRY_EDITS), and every edit is returned
    digit_problems(doc, kind) -> [str]                (lanekit.authoring's rule)
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from lanekit import authoring as lane_authoring  # noqa: E402

__all__ = ["DATASET", "REPORT_ROLES_FILE", "digit_problems", "label_roles", "pack_layer",
           "story_plan", "write_stances"]

DATASET = "cicids2017"                      # the model's file stem, the stances' dataset
REPORT_ROLES_FILE = HERE / "cic-report-roles.json"
UNPLACED = "_unplaced"
# the slots the lane program decides whatever the carry found: the v1 plan had no lexicon,
# motif, locale, report roles or monitor note, and its neutral class was no stance (fork S2)
LANE_SLOTS = ("locale", "lexicon", "motif", "class_roles", "report_roles", "monitor_note")
# The v1 plan's prose, held to this release's story contract where it breaks it, each edit the
# smallest that does, made the way the model completing the carried draft would (D5 §7.6):
# V2 — the corpus's name carries a digit no record names; V12 — the results headline is over
# its bound of characters.
CARRY_EDITS = (
    ("lead", "Trained on CIC-IDS2017,", "Trained on the CIC intrusion-detection capture,"),
    ("context.steps.data", "CIC-IDS2017 records", "The CIC intrusion-detection capture records"),
    ("results.headline", "On the sealed test part the network reaches a macro F-score of",
     "Sealed test part: macro F-score"),
    ("results.headline", "ahead of gradient-boosted trees by", "ahead of boosted trees by"),
)
ARCHITECT_MARKER = ("WARN: no /architect run was made on this bundle; the mechanical half of the "
                    "conformance gate ran and passed, and the judgement half is absent rather than "
                    "clean")
IMPROVE_MARKER = ("WARN: no /improve run was made on this bundle; the method-level half of the "
                  "judgement is absent rather than clean")
ABSTRACT = (
    "The serving bundle of the run the campaign's rule selected, the unweighted cross-entropy "
    "network, rebuilt by this release from what the sealed evaluation left behind: the resolved "
    "spec and the weights it measured, byte for byte; the package re-emitted from that spec by "
    "this release's emitter, inside the blueprint's module map with the port-owning modules made "
    "objects; and the shaper's own transform as its preprocessing. The round trip beside this "
    "text is what makes it the same network: from raw rows without their label it reproduced the "
    "evaluation's stored scores on every test row. The test part was not opened again: its "
    "stored predictions were re-encoded in this release's layout, never recomputed, and the seal "
    "still records its one exposure. The verdict travels as the evaluation wrote it. The network "
    "must not be used as a detector for traffic outside the laboratory week it was trained on, "
    "and its scores must not be read as probabilities.")


def digit_problems(doc: Any, kind: str) -> list[str]:
    return lane_authoring.digit_problems(doc, kind)


# --------------------------------------------------------------------------- the report
def label_roles(cids: Path, docs: Path, roles_file: Path, show_scripts: Path,
                lane: str = "G00") -> dict:
    """docs/label-roles.json for the real report, and the report beside it (module doc)."""
    spec = json.loads(Path(roles_file).read_text(encoding="utf-8"))
    docs.mkdir(parents=True, exist_ok=True)
    for key in ("aux", "pdf", "tex"):
        shutil.copy2(cids / spec["report"][key], docs / Path(spec["report"][key]).name)
    sys.path.insert(0, str(show_scripts))
    try:
        from report_labels import AuxLabelSource
        from vega_show_core import CATALOG
    finally:
        sys.path.remove(str(show_scripts))
    source = AuxLabelSource(docs / Path(spec["report"]["aux"]).name)
    labels, catalog = source.labels(), CATALOG["report_roles"]
    roles = {}
    for role, label in spec["roles"].items():
        if role not in catalog:
            raise ValueError(f"{roles_file}: {role!r} is not a vega-show report role")
        found = labels.get(label)
        if found is None:
            raise ValueError(f"{roles_file}: the label {label!r} ({role}) is not in the report's .aux")
        roles[role] = {"label": label, "kind": catalog[role]["kind"], "stage": catalog[role]["stage"],
                       "number": str(found.number), "page": found.page, "title": found.title}
    doc = {"schema": "vega/label-roles@1", "lane": lane,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "adapter": "latex-aux",
           "report": {"tex": spec["report"]["tex"], "pdf": spec["report"]["pdf"],
                      "aux": spec["report"]["aux"], "aux_digest": source.digest()},
           "roles": roles}
    (docs / "label-roles.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return doc


# --------------------------------------------------------------------------- the bundle layer
def pack_layer(bundle: Path, old_layer: Path, from_bundle: str) -> dict:
    """The bundle layer's authoring for the re-packed R7 (module doc)."""
    old = json.loads(Path(old_layer).read_text(encoding="utf-8"))
    intended = ((old.get("readings") or {}).get("intended_use") or "").strip()
    if not intended:
        raise ValueError(f"{old_layer} carries no readings.intended_use to carry")
    doc = {"schema": "vega/bundle-layer@1", "from_bundle": from_bundle,
           "readings": {"abstract": ABSTRACT, "intended_use": intended},
           "markers": [ARCHITECT_MARKER, IMPROVE_MARKER], "code_findings": []}
    problems = digit_problems(doc, "pack")
    if problems:
        raise ValueError("the bundle layer's prose states a number: " + "; ".join(problems))
    return doc


# --------------------------------------------------------------------------- the story plan
def write_stances(memory: Path, lane: Any) -> list[str]:
    return lane_authoring.write_show_stances(memory, lane, dataset=DATASET)


def story_plan(draft: Mapping[str, Any], facts: Mapping[str, Any], lane: Any,
               out=None) -> tuple[dict, list[str]]:
    """(the authored plan, the edits made to carried text): the carry's slots over the lane
    program's, the mandatory choices and limits the v1 plan never had added in the lane
    program's words, and the carried prose held to this release's contract (module doc)."""
    plan = lane_authoring.show_authoring(facts, lane, draft, out=out)
    carried = (draft or {}).get("authored") or {}
    authored = plan["authored"]
    for slot, value in carried.items():
        if slot not in LANE_SLOTS and value not in (None, {}, []):
            authored[slot] = _over(authored.get(slot), value)
    props = (draft or {}).get("proposals") or {}
    network, results = authored.get("network") or {}, authored.get("results") or {}
    have = {c.get("decision") for c in network.get("choices") or []}
    # the mandatory choices v1 never had, in the lane program's words — a decision by its own
    # family and type (gradient clipping is "Gradient clipping", never "The block"), told apart
    # from the carried titles
    added = lane_authoring._choices([c for c in props.get("choices") or []
                                     if c.get("mandatory") and c.get("decision") not in have])
    titles = {c.get("title") for c in network.get("choices") or []}
    network["choices"] = list(network.get("choices") or []) + [
        c if c["title"] not in titles else {**c, "title": f"{c['title']}, as the forge set it"} for c in added]
    have = {l.get("id") for l in results.get("limits") or []}
    results["limits"] = list(results.get("limits") or []) + [
        lane_authoring._limit(l, facts) for l in props.get("limits") or []
        if l.get("mandatory") and l.get("id") not in have]
    authored["input_groups"] = _placed(authored.get("input_groups") or [],
                                       props.get("input_groups") or [])
    return plan, _edit(authored)


def _over(base: Any, top: Any) -> Any:
    """`top` over `base`: mappings merged key by key (a null or empty value in `top` keeps
    `base`'s), anything else replaced."""
    if not (isinstance(base, Mapping) and isinstance(top, Mapping)):
        return copy.deepcopy(top)
    merged = dict(base)
    for key, value in top.items():
        if value not in (None, {}, []):
            merged[key] = _over(base.get(key), value)
    return merged


def _edit(authored: dict) -> list[str]:
    """CARRY_EDITS applied to the carried prose; each one made is listed (`path: old -> new`)."""
    made = []
    for path, old, new in CARRY_EDITS:
        *parents, leaf = path.split(".")
        node = authored
        for key in parents:
            node = (node or {}).get(key) if isinstance(node, Mapping) else None
        text = node.get(leaf) if isinstance(node, Mapping) else None
        if isinstance(text, str) and old in text:
            node[leaf] = text.replace(old, new)
            made.append(f"{path}: {old!r} -> {new!r}")
    return made


def _placed(groups: list, proposed: list) -> list:
    """The carried groups with `_unplaced` dissolved: each of its inputs joins the carried group
    the draft's proposal puts it with (by a member they share), else a group of the proposal's
    own, in the lane program's words."""
    unplaced = next((g for g in groups if g.get("id") == UNPLACED), None)
    if unplaced is None:
        return groups
    out = [dict(g, members=list(g.get("members") or [])) for g in groups if g.get("id") != UNPLACED]
    home = {m: g for g in out for m in g["members"]}
    for name in unplaced.get("members") or []:
        proposal = next((p for p in proposed if name in (p.get("members") or [])), None)
        mates = [m for m in (proposal or {}).get("members") or [] if m in home]
        if mates:
            home[mates[0]]["members"].append(name)
            home[name] = home[mates[0]]
            continue
        made = lane_authoring._group(proposal) if proposal else None
        target = next((g for g in out if made and g.get("id") == made["id"]), None)
        if target is None:
            target = dict(made or {"id": "other_statistics", "label": "other statistics",
                                   "text": "The statistics no other family holds."}, members=[])
            out.append(target)
        target["members"].append(name)
        home[name] = target
    return out
