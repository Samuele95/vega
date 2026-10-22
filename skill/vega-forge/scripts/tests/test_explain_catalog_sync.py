#!/usr/bin/env python3
"""
test_explain_catalog_sync.py — the pytorch backend's explain catalog covers what the backend
covers, speaks the page's vocabulary, and anchors its code steps where the code is.

    python3 test_explain_catalog_sync.py [-v]      (each `test_*` function also runs under pytest)

`backends/pytorch/explain-catalog.json` (schema `vega/explain-catalog@1`,
`references/schemas/explain-catalog-1.schema.json`) is the backend-owned text the showcase
page's stepper and code walk read (backend-contract §2.2). vega-show copies the subset a lane
uses into `page.json`, so a block without an entry, a formula that names the wrong tensors or an
anchor that no longer finds its class breaks a page long after this backend changed. Each group
below is one `test_*` function, whose docstring says what it holds: the schema (draft 2020-12,
its closed sets, damaged copies refused); coverage of `coverage.json` `inspection.blocks` (the
33), the activations and `inspection.links` (R-C1), in the catalogs' order; the stage kinds
(D4 §7.2); the fixed walk (the page's order, R-E8's anchors, the package paths each step must
and must not find, a vega-show `walk_roles` row per walk id); the anchor syntax JavaScript and
Python both read; the block anchors, resolved as the page resolves them against the segment
files `blocks.py` is cut into (resolution against rendered packages is W15's
`test_walk_anchors.py`); the inspection surface each entry `reads`, optional surface included,
and R-B8's `attention_pool`; the `recompute` legend (D4 §8.1); the activations (D4 §8.2,
`_act()`); the placeholders (backend-contract §2.2, the page's grammar, the lexicon keys every
sample shape of a block carries); and the digit rule on every prose field, through vega-show's
`digit_rule.py` (WP H01) when present and always as the vega-show catalog states it.

The plumbing is `_explain_sync_common.py`, the code-anchor checks `_explain_sync_anchors.py`,
the facts the catalog is held to `_explain_sync_facts.py` (the underscore keeps them out of the
unit tier's `test_*.py` discovery). Exit 0 when everything agrees.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _explain_sync_anchors import (  # noqa: E402
    check_block_anchor, check_custom_anchor, check_walk_anchors, check_walk_paths, segment_files,
    segment_of, segment_texts)
from _explain_sync_common import (  # noqa: E402
    BACKEND_MD, CATALOG_JSON, COVERAGE_JSON, EXPLAIN_JSON, SCHEMA_JSON, Check, blocks_module,
    catalog, catalog_digit_rule, code_objects, contract_placeholders, coverage, explain,
    formula_fields, h01_digit_rule, lexicon_keys_at, placeholder_re, prose_fields, run, schema,
    schema_validator, show_catalog, variants_of)
from _explain_sync_facts import (  # noqa: E402
    ATTENTION_POOL_READS, EXPLAINERS, KIND_OF, NO_RECOMPUTE, RE8_AT, RENDERER_KINDS, SCHEMA_ID,
    SILENCING, WALK_ORDER, damages, js_unreadable)


def test_schema():
    """The schema file is draft 2020-12, its closed sets are this test's, the catalog
    validates and damaged copies are refused."""
    c = Check("schema")
    validator = schema_validator(c)
    if validator is not None:
        c.expect(schema()["properties"]["schema"].get("const") == SCHEMA_ID,
                 f"the schema const must be {SCHEMA_ID!r}")
        c.expect(explain().get("schema") == SCHEMA_ID,
                 f"explain-catalog.json schema != {SCHEMA_ID}")
        _closed_sets(c, schema())
        errors = sorted(validator.iter_errors(explain()), key=lambda e: list(e.path))
        c.expect(not errors, "explain-catalog.json is refused by its schema: "
                             + "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:5]))
        _damaged_copies(c, validator)
    c.close()


def _closed_sets(c, sch):
    """The schema's enumerations equal the sets this test and the catalogs hold."""
    defs = sch["$defs"]
    links = list(catalog()["inspection"]["links"])
    walk_items = sch["properties"]["walk_fixed"]["prefixItems"]
    closed = (
        (defs["stage_kind"]["enum"], sorted(set(KIND_OF.values())), "stage kinds", True),
        (defs["link"]["enum"], links, "links", False),
        (defs["activation_name"]["enum"], list(coverage()["activations"]), "activations", False),
        (defs["explainer"]["enum"], list(EXPLAINERS), "explainers", True),
        (defs["walk_step"]["properties"]["id"]["enum"], list(WALK_ORDER), "walk ids", False),
        ([p["properties"]["id"]["const"] for p in walk_items], list(WALK_ORDER), "walk order",
         False),
        (sch["properties"]["links"]["required"], links, "required links", False),
    )
    for got, want, what, as_set in closed:
        same = sorted(got) == sorted(want) if as_set else list(got) == list(want)
        c.expect(same, f"schema {what} {list(got)} != {list(want)}")


def _damaged_copies(c, validator):
    """Each seeded damage, applied to a copy of the catalog, is refused by the schema."""
    for what, damage in damages(explain()):
        broken = copy.deepcopy(explain())
        try:
            damage(broken)
        except (KeyError, IndexError, TypeError) as exc:
            c.fail(f"cannot seed {what} into this catalog ({exc!r}): its shape is not the "
                   "schema's")
            continue
        c.expect(not validator.is_valid(broken), f"the schema accepts a catalog with {what}")


def test_coverage():
    """One entry per block type, activation and link the backend and the catalog name."""
    c = Check("coverage")
    cat, cov, fc = explain(), coverage(), catalog()
    c.expect(cat["backend"] == cov["name"], f"backend {cat['backend']!r} != coverage.json name "
                                            f"{cov['name']!r}")
    c.expect(cat["language"] == cov["language"],
             f"language {cat['language']!r} != coverage.json language {cov['language']!r}")
    types = list(cat["blocks"])
    inspected = list(cov["inspection"]["blocks"])
    c.expect(types == inspected, f"blocks {types} != coverage.json inspection.blocks {inspected}")
    c.expect(types == list(fc["blocks"]),
             f"blocks differ from the catalog's block types on "
             f"{sorted(set(types) ^ set(fc['blocks']))} (or in order)")
    built = set(cov["blocks"]) | set((cov.get("planned") or {}).get("blocks") or ())
    c.expect(set(types) == built, f"blocks differ from coverage.json blocks + planned.blocks on "
                                  f"{sorted(set(types) ^ built)}")
    c.expect(len(types) == 33, f"{len(types)} block types, the v2.3 grammar has 33")
    acts = list(cat["activations"])
    c.expect(acts == list(cov["activations"]),
             f"activations {acts} != coverage.json activations {cov['activations']}")
    c.expect(acts == list(fc["activations"]),
             f"activations {acts} != the catalog's activations {fc['activations']}")
    links, want = list(cat["links"]), list(fc["inspection"]["links"])
    c.expect(links == want, f"links {links} != the catalog's inspection.links {want} (R-C1)")
    c.expect(len(links) == 8, f"{len(links)} links, R-C1 has 8")
    c.ok(f"{len(types)} blocks, {len(acts)} activations, {len(links)} links")
    c.close()


def test_stage_kinds():
    """Each block type's stage kind is the page's (D4 §7.2)."""
    c = Check("stage kinds")
    blocks = explain()["blocks"]
    c.expect(set(KIND_OF) == set(catalog()["blocks"]),
             f"this test's KIND_OF differs from the catalog's block types on "
             f"{sorted(set(KIND_OF) ^ set(catalog()['blocks']))}")
    for t, b in blocks.items():
        c.expect(b["stage_kind"] == KIND_OF.get(t),
                 f"blocks.{t}.stage_kind {b['stage_kind']!r} != D4 §7.2 {KIND_OF.get(t)!r}")
    kinds = {b["stage_kind"] for b in blocks.values()}
    c.expect(len(kinds) + 1 == RENDERER_KINDS,
             f"{len(kinds)} stage kinds plus input, the page's renderer registry has "
             f"{RENDERER_KINDS}")
    c.close()


def test_walk():
    """The fixed walk: the page's order, R-E8's anchors, the right files, a role per id."""
    c = Check("walk")
    steps = explain()["walk_fixed"]
    ids = [s["id"] for s in steps]
    c.expect(tuple(ids) == WALK_ORDER, f"walk_fixed ids {ids} != {list(WALK_ORDER)}")
    walk = show_catalog()["walk"]
    c.expect(ids == list(walk["head"]) + list(walk["tail"]),
             f"walk_fixed ids {ids} != vega-show catalog walk.head + walk.tail "
             f"{list(walk['head']) + list(walk['tail'])}")
    by_id = {s["id"]: s for s in steps}
    for sid, at in RE8_AT.items():
        if sid in by_id:
            c.expect(by_id[sid]["code"]["at"] == [at],
                     f"walk_fixed.{sid}.code.at {by_id[sid]['code']['at']} != R-E8 [{at!r}]")
    check_walk_anchors(c, by_id, walk.get("anchors") or {})
    check_walk_paths(c, by_id)
    roles = show_catalog().get("walk_roles") or {}
    for wid in ids + list(explain()["blocks"]):
        c.expect(wid in roles, f"walk id {wid!r} has no walk_roles row in the vega-show catalog")
    c.close()


def test_anchor_syntax():
    """Every anchor pattern compiles and keeps to what JavaScript's RegExp also reads."""
    c = Check("anchor syntax")
    for where, code in code_objects(explain()):
        patterns = [("file", code["file"])] + [(f"at[{i}]", p) for i, p in enumerate(code["at"])]
        patterns += [(k, code[k]) for k in ("after", "to") if code.get(k)]
        for key, pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                c.fail(f"{where}.code.{key} {pattern!r} does not compile: {exc}")
                continue
            bad = js_unreadable(pattern)
            c.expect(not bad, f"{where}.code.{key} {pattern!r} uses {bad}, which JavaScript's "
                              "RegExp does not read")
    c.close()


def test_block_anchors():
    """Each block anchor lands on its class in the segment file the emitter writes it to."""
    c = Check("block anchors")
    B = blocks_module()
    segments = segment_texts()
    c.expect(set(segments) == set(B.SEGMENT_FILE),
             f"blocks.py segments differ from SEGMENT_FILE on "
             f"{sorted(set(segments) ^ set(B.SEGMENT_FILE))}")
    files, paths = segment_files(segments, B.SEGMENT_FILE)
    for t, b in explain()["blocks"].items():
        if t == "custom":
            check_custom_anchor(c, b["code"], paths)
            continue
        cls_name = B.CLASS_FOR.get(t)
        if not c.expect(cls_name is not None, f"block type {t!r} has no CLASS_FOR entry"):
            continue
        check_block_anchor(c, f"blocks.{t}", b["code"], segment_of(segments, cls_name),
                           (cls_name, files, paths))
        for v, var in variants_of(b):
            vcls = (B.CLASS_VARIANTS.get(t) or {}).get(v)
            if c.expect(vcls is not None, f"blocks.{t}.variants.{v}: CLASS_VARIANTS has no "
                                          f"{t}/{v}"):
                check_block_anchor(c, f"blocks.{t}.variants.{v}", var["code"],
                                   segment_of(segments, vcls), (vcls, files, paths))
    c.close()


def test_surface():
    """What an entry reads is its type's inspection surface, optional surface included;
    variants follow coverage.json."""
    c = Check("surface")
    ins = coverage()["inspection"]
    surface, variants = ins["surface"], ins.get("surface_variants") or {}
    B = blocks_module()
    for t, b in explain()["blocks"].items():
        optional = set(((ins.get("surface_optional") or {}).get(t) or {}).get("attributes") or ())
        allowed = set(surface.get(t) or ()) | optional
        extra = [r for r in b["reads"] if r not in allowed]
        c.expect(not extra, f"blocks.{t}.reads {extra} are not inspection surface of {t} "
                            f"(coverage.json: {sorted(allowed)})")
        c.expect(optional <= set(b["reads"]),
                 f"blocks.{t}.reads leaves out its optional surface "
                 f"{sorted(optional - set(b['reads']))} (coverage.json surface_optional)")
        want, have = set(variants.get(t) or {}), {v for v, _ in variants_of(b)}
        c.expect(have == want, f"blocks.{t}.variants {sorted(have)} != coverage.json "
                               f"surface_variants {sorted(want)}")
        c.expect(have == set(B.CLASS_VARIANTS.get(t) or {}),
                 f"blocks.{t}.variants {sorted(have)} != blocks.CLASS_VARIANTS "
                 f"{sorted(B.CLASS_VARIANTS.get(t) or {})}")
        for v, var in variants_of(b):
            vallowed = set(surface.get(t) or ()) | set((variants.get(t) or {}).get(v, {})
                                                       .get("attributes") or ())
            extra = [r for r in var["reads"] if r not in vallowed]
            c.expect(not extra, f"blocks.{t}.variants.{v}.reads {extra} are not its surface")
    pool = explain()["blocks"].get("attention_pool") or {}
    c.expect(pool.get("reads") == ATTENTION_POOL_READS,
             f"R-B8: blocks.attention_pool.reads {pool.get('reads')} != {ATTENTION_POOL_READS}")
    formula = re.sub(r"\s+", "", pool.get("formula", ""))
    c.expect("softmax" in formula and "q·tanh(W" in formula and "Σ" in formula,
             "R-B8: blocks.attention_pool.formula must read α = softmax_t(q · tanh(W h_t + b)), "
             f"y = Σ_t α_t h_t; it reads {pool.get('formula')!r}")
    c.close()


def test_recompute():
    """Every method is in the legend and used; non-computing kinds recompute nothing."""
    c = Check("recompute")
    cat = explain()
    legend = cat["recompute"]
    used = {}
    for t, b in cat["blocks"].items():
        used.setdefault(b["recompute"], []).append(t)
        for v, var in variants_of(b):
            used.setdefault(var["recompute"], []).append(f"{t}/{v}")
    c.expect(set(used) <= set(legend),
             f"recompute methods {sorted(set(used) - set(legend))} are not in the legend")
    c.expect(set(legend) <= set(used),
             f"legend methods {sorted(set(legend) - set(used))} are named by no block")
    for method, entry in legend.items():
        c.expect(entry["explainer"] in EXPLAINERS,
                 f"recompute.{method}.explainer {entry['explainer']!r} is not a block explainer "
                 f"{list(EXPLAINERS)}")
    c.expect(legend.get("none", {}).get("explainer") == "generic",
             "recompute.none is the generic explainer's (formula and shapes only)")
    for t, b in cat["blocks"].items():
        if b["stage_kind"] in NO_RECOMPUTE:
            c.expect(b["recompute"] == "none",
                     f"blocks.{t} ({b['stage_kind']}) is not a step at answer time and "
                     f"recomputes nothing, not {b['recompute']!r} (D4 §8.1)")
        else:
            c.expect(b["recompute"] != "none",
                     f"blocks.{t} ({b['stage_kind']}) computes and needs a recompute method")
    c.close()


def test_activations():
    """The backend's activation table, the page's curves, the silencing set, the parameters."""
    c = Check("activations")
    B = blocks_module()
    acts = explain()["activations"]
    c.expect(list(acts) == list(B._ACT), f"activations {list(acts)} != blocks._ACT "
                                         f"{list(B._ACT)}")
    for name, entry in acts.items():
        c.expect(entry["curve"] == name, f"activations.{name}.curve {entry['curve']!r} != "
                                         f"{name!r} (the page's act.js is keyed by catalog names)")
        c.expect(entry["silences_negative"] == (name in SILENCING),
                 f"activations.{name}.silences_negative {entry['silences_negative']} "
                 f"(D4 §8.2: true for {sorted(SILENCING)})")
    leaky = B._act("leaky_relu")
    c.expect((acts.get("leaky_relu", {}).get("params") or {}).get("slope")
             == leaky.negative_slope,
             f"activations.leaky_relu.params.slope != the slope _act() builds "
             f"({leaky.negative_slope})")
    gelu = B._act("gelu")
    c.expect((acts.get("gelu", {}).get("params") or {}).get("approximate") == gelu.approximate,
             f"activations.gelu.params.approximate != the form _act() builds "
             f"({gelu.approximate!r})")
    c.close()


def test_placeholders():
    """Placeholders: the contract's set, the page's grammar, lexicon keys the shapes carry."""
    c = Check("placeholders")
    cat = explain()
    declared = set(cat["placeholders"])
    contract = contract_placeholders()
    c.expect(contract, "backend-contract §2.2 lists no placeholder")
    c.expect(declared == contract,
             f"placeholders {sorted(declared)} != backend-contract §2.2 {sorted(contract)}")
    used = set()
    for where, text in list(prose_fields(cat)) + list(formula_fields(cat)):
        for m in re.finditer(r"\{[^{}]*\}", text):
            used |= _placeholder(c, where, m.group(0), declared)
        rest = placeholder_re().sub("", text)
        c.expect("{" not in rest and "}" not in rest,
                 f"{where}: a brace outside a placeholder: {text!r}")
    c.expect(used >= declared, f"placeholders {sorted(declared - used)} are used nowhere")
    c.close()


def _placeholder(c, where, token, declared):
    """One `{…}` of the field at `where`: in the page's grammar, and either a declared bare
    name or a lexicon path the field's sample shapes all carry. Returns the bare names used."""
    pm = placeholder_re().fullmatch(token)
    if not c.expect(pm, f"{where}: {token} is not a placeholder of the page's grammar"):
        return set()
    name, path = pm.group(1), pm.group(2)
    if path is None:
        c.expect(name in declared, f"{where}: {token} names no declared placeholder")
        return {name}
    allowed = lexicon_keys_at(where)
    lex_paths = set(show_catalog()["lexicon"]["paths"])
    c.expect(name in allowed and path in lex_paths,
             f"{where}: {token} is not a lexicon path every shape of it carries "
             f"(keys {sorted(allowed)}, paths {sorted(lex_paths)})")
    return set()


def test_digit_rule():
    """Every prose field passes the digit rule: H01's module when present, and always the rule
    the vega-show catalog states, with no lane's names."""
    c = Check("digit rule")
    rules = [("vega-show catalog digit_rule", catalog_digit_rule())]
    h01 = h01_digit_rule()
    if h01 is not None:
        rules.append(("vega-show/scripts/digit_rule.py", h01))
    else:
        print("  note  vega-show/scripts/digit_rule.py (WP H01) is not in this tree: the prose "
              "is checked against the rule the vega-show catalog states")
    fields = list(prose_fields(explain()))
    for label, refusals in rules:
        for where, text in fields:
            bad = refusals(text, where)
            c.expect(not bad, f"{where} breaks the digit rule ({label}): {bad}: {text!r}")
        c.ok(f"{len(fields)} prose fields pass ({label})")
    c.close()


TESTS = (test_schema, test_coverage, test_stage_kinds, test_walk, test_anchor_syntax,
         test_block_anchors, test_surface, test_recompute, test_activations, test_placeholders,
         test_digit_rule)


def _summary():
    cat = explain()
    return (f"\nexplain-catalog.json agrees: {len(cat['blocks'])} block types, "
            f"{len(cat['activations'])} activations, {len(cat['links'])} links, "
            f"{len(cat['walk_fixed'])} fixed walk steps, {len(list(prose_fields(cat)))} prose "
            f"fields under the digit rule")


if __name__ == "__main__":
    print(f"comparing {EXPLAIN_JSON.name} with {SCHEMA_JSON.name}, {COVERAGE_JSON.name}, "
          f"{CATALOG_JSON.name}, blocks.py, {BACKEND_MD.name} and the vega-show catalog")
    sys.exit(run(TESTS, _summary))
