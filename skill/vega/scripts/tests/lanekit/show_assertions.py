"""
The show stage's artifact assertions (D6 §6.8), run by `python -m lanekit.checks check-show`:
the showcase's closed world, the bundle pin, the class roles, the motif, the lexicon and the
one digit rule over the story's prose. The Checker and Ctx are lanekit.assertions'.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from . import contracts, observe
from .assertions import Checker, Ctx, _import, _json


# --------------------------------------------------------------------------- show (D6 §6.8)
def check_show(ctx: Ctx, check: Checker) -> None:
    bundle = sorted(ctx.work.glob("bundle/*/MANIFEST.json"))[0].parent
    showcase = bundle / contracts.SHOWCASE_DIR
    doc = _json(showcase / contracts.SHOWCASE_MANIFEST)
    _closed_world(check, showcase, doc)
    pin = _json(ctx.args.pin)
    manifest_digest = contracts.sha256_file(bundle / "MANIFEST.json")
    check.that(doc["bundle"]["manifest_digest"] == manifest_digest == pin["manifest"],
               "SHOWCASE.json names this bundle's MANIFEST, unchanged since pack",
               "the showcase's bundle digest, the MANIFEST and the pack-time pin disagree")
    story = _json(showcase / doc["story"]["file"])
    page = _json(showcase / doc["page"]["file"]) if doc.get("page") else {}
    _roles(ctx, check, story)
    motif = page.get("motif") or story.get("motif")
    motif = motif.get("id") if isinstance(motif, Mapping) else motif   # page.json's motif is {id, base, …}
    want = ctx.lane.expect.get("motif")
    check.that(want is None or motif == want, f"motif {motif}", f"motif {motif!r}, the lane expects {want!r}")
    lexicon = {k: v for k, v in (story.get("lexicon") or {}).items() if k in contracts.LEXICON_KEYS}
    declared = {k: v for k, v in (ctx.lane.raw.get("lexicon") or {}).items() if k != "target"}
    check.that({k: lexicon.get(k) for k in declared} == declared and set(lexicon) - {"target"} == set(declared),
               "the lexicon is the lane's", f"lexicon {sorted(lexicon)} != the lane's {sorted(declared)}")
    _digit_rule(ctx, check, story)
    _claims(check, story)
    out = observe.showcase_facts(showcase)
    (ctx.lane_dir / "results").mkdir(exist_ok=True)
    (ctx.lane_dir / "results" / "show.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def _closed_world(check: Checker, showcase: Path, doc: Mapping[str, Any]) -> None:
    pinned = set(doc.get("files") or {})
    present = {str(p.relative_to(showcase)) for p in showcase.rglob("*") if p.is_file()}
    present = {p for p in present if p != contracts.SHOWCASE_MANIFEST and not p.startswith(contracts.SHOWCASE_UNPINNED)}
    check.that(present <= pinned, "every showcase file is pinned", f"unpinned: {sorted(present - pinned)[:5]}")
    check.that(pinned <= present, "every pinned file exists", f"missing: {sorted(pinned - present)[:5]}")
    bad = [rel for rel, digest in (doc.get("files") or {}).items()
           if (showcase / rel).is_file() and contracts.sha256_file(showcase / rel) != digest]
    check.that(not bad, "every pinned digest holds", f"digests differ: {bad[:5]}")


def _roles(ctx: Ctx, check: Checker, story: Mapping[str, Any]) -> None:
    roles = ((story.get("class_roles") or {}).get("roles")) or {}
    declared = ctx.lane.raw.get("roles") or {}
    check.that({str(k): v for k, v in roles.items()} == declared, "the class roles are the lane's",
               f"class roles {roles} != the lane's {declared}")


def _claims(check: Checker, story: Mapping[str, Any]) -> None:
    """What the story's words claim holds against what they describe (gate C): every network
    choice titled apart from the others, a decision that is not a block of the network never
    called one, every input group labelled apart, and a beaten verdict never said in a softer
    word ("could not separate", "does as well")."""
    sections = {s.get("id"): s for s in story.get("sections") or () if isinstance(s, Mapping)}
    choices = (sections.get("network") or {}).get("choices") or []
    titles = [str(c.get("title")) for c in choices]
    check.that(len(titles) == len(set(titles)), "every network choice has a title of its own",
               f"choice titles repeat: {sorted({t for t in titles if titles.count(t) > 1})}")
    blockish = [str(c.get("title")) for c in choices
                if ((c.get("focus") or {}).get("kind") != "block")
                and ("block of the network" in str(c.get("text")) or str(c.get("title")) == "The block")]
    check.that(not blockish, "no decision but a block is called a block", f"called a block: {blockish}")
    labels = [str(g.get("label")) for g in story.get("input_groups") or []]
    check.that(len(labels) == len(set(labels)), "every input group has a label of its own",
               f"group labels repeat: {sorted({x for x in labels if labels.count(x) > 1})}")
    results = sections.get("results") or {}
    if (results.get("verdict") or {}).get("beaten_by_baseline"):
        soft = [lim.get("id") for lim in results.get("limits") or []
                if any(w in str(lim.get("text")).lower() for w in ("could not separate", "does as well"))]
        check.that(not soft, "a beaten verdict is said as beaten", f"softened in limits {soft}")


def _digit_rule(ctx: Ctx, check: Checker, story: Mapping[str, Any]) -> None:
    """Every prose string of the story (a string with a space: ids, refs, paths and digests
    have none) passes the one digit rule (R-E10, H01) with the facts' name whitelist."""
    rel, name = contracts.DIGIT_RULE_API
    rule = ctx.skills / rel
    if not rule.is_file():
        check.error(f"{rel} is missing; the story's digit rule cannot be checked")
        return
    sys.path.insert(0, str(rule.parent))
    check_prose = getattr(_import(rule.stem), name)
    facts = _json(ctx.lane_dir / "show" / "plan" / "facts.json")
    names = facts.get(contracts.FACTS_NAMES) or []
    text_keys = {k: v for k, v in story.items() if k not in STORY_RECORD_KEYS}
    found = [f"{where}: {p}" for where, text in _prose(text_keys, "story")
             for p in check_prose(text, names=names)]
    check.that(not found, "no digit outside a placeholder in the story's prose", f"digit rule: {found[:3]}")


# the story's record keys: the facts' values and their sources, the task, the lexicon words, the
# class roles with their evidence, the bundle and report pins — data, not the story's text, which
# is every other key (headline, lead, sections, input_groups, code_notes; D6 §6.8 assertion 3)
STORY_RECORD_KEYS = frozenset({"schema", "locale", "task", "lexicon", "class_roles", "polarity", "values",
                               "formats", "refs", "bundle", "report"})


def _prose(node: Any, where: str) -> list[tuple[str, str]]:
    if isinstance(node, str):
        return [(where, node)] if " " in node.strip() else []
    if isinstance(node, Mapping):          # a limit's `recorded` is the record's own words: data
        return [hit for k, v in node.items() if k != "recorded" for hit in _prose(v, f"{where}.{k}")]
    if isinstance(node, list):
        return [hit for i, v in enumerate(node) for hit in _prose(v, f"{where}[{i}]")]
    return []
