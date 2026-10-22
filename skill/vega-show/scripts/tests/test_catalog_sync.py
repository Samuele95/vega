#!/usr/bin/env python3
"""
test_catalog_sync.py — `scripts/catalog.json` is the machine copy of `show-contract.md`, and
this test fails when the two diverge, when either disagrees with the catalogs vega-show reads
from, or when a schema under `references/schemas/` does not validate.

    python3 test_catalog_sync.py [-v] [--patch <document-project SKILL.md>]

The vega-show scripts read every closed set, bound and table from the catalog and never restate
one in code, so a contract that gains a motif, a role, a gate or a lexicon key the JSON does not
carry would ship a page the checks cannot refuse — the drift this test exists to prevent. It
checks, as gates CS1, CS2 and CS3 of the contract's §10:

1. **The contract and the catalog** (CS1, `_sync_contract.py`): every closed-set and bound row
   against the key it names, and every structured table against its catalog key.
2. **The catalog itself** (CS1, here): its own consistency, and the design cases every gate
   carries — every design negative maps onto a gate, a fork or an external suite.
3. **The other catalogs** (CS1, `_sync_catalogs.py`): the vega-forge catalog (tasks, families,
   blocks, sample shapes, decision families, `metric_direction` with a max/min direction for every
   results fact, `direction_of` and task primary, and the inspection rows once present), the
   vega-eval catalog (baselines, the metric set and the primary per task) and the backend's
   `METRICS`.
4. **The schemas** (CS2, `_sync_schemas.py`): draft 2020-12, examples valid, seeded mutations
   refused, every enum or pattern that copies a closed set equal to the catalog's.
5. **The page's strings keys** (here), once `assets/page/TEMPLATE.json` exists (W15 generates
   it): every id vega-show emits and the page renders through a key has that key in
   `strings.keys.json` — the task primaries included.
6. **The document-project patch** (CS3, `_sync_patch.py`): `patches/document-project/SKILL.md` (or the file
   `--patch` names) has a "The label-role map" section whose role table is exactly
   `catalog.report_roles`, each with its kind and stage, whose `vega/label-roles@1` example
   validates, and the patch states the post-report exclusion of the `show` brief (D5 §13.8).
   Seeded copies of the patch — a role dropped (design negative D5-39), an unknown role added, a
   kind changed, the section renamed, the exclusion sentence removed — must each be refused.

Needs `jsonschema` (draft 2020-12 support); without it the schema gate CS2 fails rather than
passing unchecked. Exit 0 when everything agrees.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sync_catalogs import check_catalogs                                      # noqa: E402
from _sync_common import (BRIEF_STAGES, CATALOG_JSON, CONTRACT_MD, EVAL_CATALOG,  # noqa: E402
                          FIXED_WALK, FORGE_CATALOG, PAGE_DIR, fail, failures, load_json, ok,
                          same, show_metrics)
from _sync_contract import check_contract                                      # noqa: E402
from _sync_patch import check_patch, patch_path                                # noqa: E402
from _sync_schemas import check_schemas                                        # noqa: E402

# the design cases every vega-show negative suite must be able to seed, de-duplicated by gate:
# D5 §12.2 (1–40), D6 §12.5 (VS01–VS27) and §12.4 (SS1–SS9; SS8 is the positive case), D4 §17.7
CASE_UNIVERSE = ([f"D5-{i}" for i in range(1, 41)] + [f"VS{i:02d}" for i in range(1, 28)]
                 + [f"SS{i}" for i in range(1, 10) if i != 8] + [f"PN{i}" for i in range(1, 13)]
                 + [f"GC-{i}" for i in range(1, 5)])          # gate C's review panel (V15)


def check_catalog_itself(cat):
    same("catalog schema id", cat.get("schema"), "vega/show-catalog@1")
    same("stage identity", (cat["stage"]["name"], cat["stage"]["layer"], cat["stage"]["stem_tag"]),
         ("show", "showcase", ".show"))
    ids = [g["id"] for g in cat["gates"]]
    if len(ids) != len(set(ids)):
        fail(f"duplicate gate ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    same("validate gates are V1–V15", [i for i in ids if re.fullmatch(r"V\d+", i)],
         [f"V{n}" for n in range(1, 16)])
    for gate in cat["gates"]:
        if gate["group"] not in cat["gate_groups"]:
            fail(f"gate {gate['id']} is in the unknown group {gate['group']!r}")
    empty = [g for g in cat["gate_groups"] if not any(x["group"] == g for x in cat["gates"])]
    if empty:
        fail(f"gate groups with no gate: {empty}")
    _check_cases(cat)
    _check_motifs_roles(cat)
    _check_showcase(cat)
    _check_story_sets(cat)
    _check_results_emit(cat)


def _check_cases(cat):
    covered = {c for g in cat["gates"] for c in g["cases"]}
    covered |= {c for f in cat["forks"].values() for c in f.get("cases", [])}
    covered |= set(cat.get("external_cases", {}))
    bad = sorted(c for c in covered if not re.fullmatch(r"D5-\d+|VS\d{2}|PN\d+|SS\d|GC-\d+", c))
    if bad:
        fail(f"case ids outside the designs' grammar: {bad}")
    missing = [c for c in CASE_UNIVERSE if c not in covered]
    if missing:
        fail(f"design negatives no gate, fork or external suite catches: {missing}")
    else:
        ok(f"all {len(CASE_UNIVERSE)} design negatives map onto a gate, a fork or an external suite")


def _check_motifs_roles(cat):
    motifs = cat["motifs"]
    for modality, row in motifs["by_modality"].items():
        if row["base"] not in row["admissible"] or not set(row["admissible"]) <= set(motifs["set"]):
            fail(f"motifs.by_modality.{modality}: the base must be admissible and the admissible "
                 "motifs members of the set")
    unused = set(motifs["set"]) - {m for r in motifs["by_modality"].values() for m in r["admissible"]}
    if unused:
        fail(f"motifs no modality admits: {sorted(unused)}")
    roles = cat["roles"]
    pattern = re.compile(roles["basis_pattern"])
    for good in ("structural:majority", "structural:negative", "task:anomaly", "user:turn-3",
                 "show:2026-09-29", "vega:default"):
        if not pattern.fullmatch(good):
            fail(f"roles.basis_pattern refuses the basis {good!r}")
    for bad in ("domain:ontology", "spec:line-4", "structural:rare", "memory:x", "show:yesterday"):
        if pattern.fullmatch(bad):
            fail(f"roles.basis_pattern accepts {bad!r}; R-E5 grants no role to it")
    for basis, row in roles["basis"].items():
        if "alert" in row["grants"] and not basis.startswith(tuple(roles["alert_requires"])):
            fail(f"the basis {basis} grants alert, which R-E5 reserves for {roles['alert_requires']}")
        if not set(row["grants"]) <= set(roles["set"]):
            fail(f"the basis {basis} grants a role outside {roles['set']}")


def _check_showcase(cat):
    show = cat["showcase"]
    for path, row in show["files"].items():
        if row["role"] is not None and row["role"] not in show["roles"]:
            fail(f"showcase file {path} has the unknown role {row['role']!r}")
        if row["served"] and row["role"] not in show["served_roles"]:
            fail(f"showcase file {path} is served with a role that is never served")
        if row["served"] and not row["pinned"]:
            fail(f"showcase file {path} is served but not pinned; the slot serves only files")
    slot = cat["slot"]
    if slot["media"].get(".md") != "text/markdown; charset=utf-8":
        fail("slot.media lacks .md → text/markdown; charset=utf-8 (R-D5)")
    if cat["showcase_extensions"]["refused"][0] != ".py":
        fail("showcase_extensions.refused must name .py first; R-E1 refuses it everywhere")
    if not set(cat["showcase"]["unpinned_dirs"]) <= {".work/", ".smoke/", "docs/"}:
        fail("showcase.unpinned_dirs is wider than R-D5's .work/, .smoke/ and docs/")
    for route in ("GET /", "GET /showcase/{path}", "GET /v1/showcase"):
        if route not in slot["routes"]:
            fail(f"slot.routes lacks {route}")


def _check_results_emit(cat):
    """A results fact is read where the primary is (§4.5): the network's row of the evaluation."""
    emit = cat["results_emit"]["ref"].replace("<metric>", "<primary>")
    same("results_emit.ref is the primary's ref with the metric named",
         emit, cat["reserved_values"]["primary"]["ref"])


def _check_lexicon(cat):
    lex = cat["lexicon"]
    value = re.compile(lex["value_pattern"])
    for key, entry in lex["defaults"].items():
        articles = lex["articles"].get(key, lex["articles"]["default"])
        if not (value.fullmatch(entry["one"]) and value.fullmatch(entry["many"])
                and entry["article"] in articles):
            fail(f"lexicon.defaults.{key} breaks the lexicon's own value rules")
    for shape, keys in lex["required"].items():
        if not set(keys) <= set(lex["keys"]) - set(lex["optional"]):
            fail(f"lexicon.required.{shape} names an optional or unknown key")
    needed = {k for keys in lex["required"].values() for k in keys}
    if needed - set(lex["defaults"]):
        fail(f"required lexicon keys without a default: {sorted(needed - set(lex['defaults']))}")


def _check_story_sets(cat):
    _check_lexicon(cat)
    steps = cat["context_steps"]
    for shape, ids in steps["by_shape"].items():
        if not set(ids) <= set(steps["ids"]):
            fail(f"context_steps.by_shape.{shape} names an unknown step")
    for task, pair in cat["polarity"]["by_task"].items():
        if not set(pair) <= set(cat["polarity"]["values"]):
            fail(f"polarity.by_task.{task} names an unknown polarity")
    rm = cat["result_models"]
    if "network" not in rm["names"] or not set(rm["from_baseline"].values()) <= set(rm["names"]):
        fail("result_models: every baseline must map to a model name, and network be one")
    kinds = cat["breakdown"]["kinds"]
    for task, row in cat["breakdown"]["by_task"].items():
        if isinstance(row, dict) and row["kind"] not in kinds:
            fail(f"breakdown.by_task.{task} has the unknown kind {row['kind']!r}")
    focus = cat["focus"]
    if not set(focus["by_decision"].values()) <= set(focus["families"]):
        fail("focus.by_decision maps onto a family focus.families does not hold")
    for role, row in cat["report_roles"].items():
        if row["kind"] not in cat["label_kinds"]["kinds"] or row["stage"] not in BRIEF_STAGES:
            fail(f"report role {role}: unknown kind or stage")
    for wid, roles in cat["walk_roles"].items():
        if not set(roles) <= set(cat["report_roles"]):
            fail(f"walk_roles.{wid} names a role report_roles does not hold")
    same("walk head and tail", tuple(cat["walk"]["head"] + cat["walk"]["tail"]), FIXED_WALK)


def check_page_keys(cat, primaries):
    """Once the page template exists, every id vega-show emits has its strings key (§7.2)."""
    template = PAGE_DIR / "TEMPLATE.json"
    if not template.is_file():
        ok("assets/page/TEMPLATE.json does not exist yet (W15 generates it); the strings keys are "
           "checked from then on")
        return
    keys_file = PAGE_DIR / "strings.keys.json"
    doc = load_json(keys_file, "the page's strings key list")
    if doc is None:
        return
    found = set()

    def walk(node):
        if isinstance(node, str):
            found.add(re.split(r"[@#]", node)[0])
        elif isinstance(node, dict):
            for k, v in node.items():
                found.add(re.split(r"[@#]", k)[0])
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(doc)
    members = {"context.step.{id}": cat["context_steps"]["ids"],
               "model.{name}": cat["result_models"]["names"],
               "report.kind.{kind}": cat["label_kinds"]["kinds"],
               "metric.{metric}": sorted(show_metrics(cat) | primaries)}
    same("page_keys templates", sorted(cat["page_keys"]), sorted(members))
    for template_key, names in members.items():
        stem = template_key.split("{")[0]
        missing = [stem + n for n in names if stem + n not in found]
        if missing:
            fail(f"strings.keys.json lacks {missing}; vega-show emits those ids")



def main():
    print(f"catalog sync — {CATALOG_JSON.name} against {CONTRACT_MD.name}, the vega-forge and "
          "vega-eval catalogs, METRICS, the schemas and the document-project patch")
    cat = load_json(CATALOG_JSON, "catalog.json")
    forge = load_json(FORGE_CATALOG, "the vega-forge catalog")
    evaluation = load_json(EVAL_CATALOG, "the vega-eval catalog")
    try:
        md = CONTRACT_MD.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"show-contract.md does not read: {exc}")
        md = ""
    if cat is None or forge is None or evaluation is None or not md:
        print(f"\n{len(failures)} failure(s)")
        return 1
    check_contract(cat, md)
    check_catalog_itself(cat)
    primaries = check_catalogs(cat, forge, evaluation)
    check_schemas(cat, forge)
    check_page_keys(cat, primaries)
    check_patch(cat, patch_path(sys.argv))
    print()
    if failures:
        print(f"{len(failures)} disagreement(s) between the contract, the catalog, the other "
              "catalogs, the schemas and the document-project patch")
        return 1
    print(f"OK: catalog.json and {CONTRACT_MD.name} agree ({len(cat['gates'])} gates, "
          f"{len(cat['schemas'])} schemas, {len(cat['report_roles'])} report roles), and both "
          "agree with the vega-forge and vega-eval catalogs and the document-project patch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
