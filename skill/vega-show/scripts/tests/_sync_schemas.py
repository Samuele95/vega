"""
_sync_schemas.py — gate CS2: each schema under `references/schemas/` is JSON Schema draft 2020-12
with the `$id` and `schema` constant its id implies, its examples validate, seeded mutations of
an example do not, every enum or pattern that copies a closed set equals the catalog's, and every
bounded text field carries the catalog's length. A helper of `test_catalog_sync.py`.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from _sync_common import BRIEF_STAGES, DRAFT_2020_12, LINKS, SCHEMA_DIR, fail, load_json, ok, same

def pointer(doc, path):
    node = doc
    for part in path.strip("/").split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def schema_sets(cat, forge):
    """(schema file, JSON pointer, expected members) for every enum or pattern that copies a
    closed set; lists are compared as sets, strings exactly."""
    views = cat["views"]["ids"] + cat["views"]["sub_views"]["network"]
    view_keys = sorted({v for r in cat["view_keys"].values()
                        for v in (r.values() if isinstance(r, dict) else [r])})
    roles, lex = sorted(cat["report_roles"]), cat["lexicon"]
    models = cat["result_models"]["names"]
    tasks, shapes = sorted(forge["tasks"]), forge["sample_shapes"]
    modalities = sorted(cat["motifs"]["by_modality"])
    return [
        ("showcase-1", "/properties/report/properties/status/enum", cat["showcase"]["report_status"]),
        ("showcase-1", "/properties/roles/required", cat["showcase"]["roles"]),
        ("showcase-1", "/$defs/report_role/enum", roles),
        ("showcase-1", "/properties/report/properties/roles/oneOf/1/properties/origin/enum",
         cat["showcase"]["role_origins"]),
        ("showcase-1", "/properties/entry/const", cat["showcase"]["entry"]),
        ("showcase-1", "/properties/story/properties/sections/items/enum", cat["sections"]["ids"]),
        ("showcase-1", "/$defs/relpath/pattern", cat["showcase"]["path_pattern"]),
        ("showcase-1", "/properties/report/properties/briefs/oneOf/1/propertyNames/enum",
         list(cat["briefs"]["stages"])),
        ("label-roles-1", "/properties/roles/propertyNames/enum", roles),
        ("label-roles-1", "/$defs/binding/properties/kind/enum", cat["label_kinds"]["kinds"]),
        ("label-roles-1", "/$defs/binding/properties/stage/enum", BRIEF_STAGES),
        ("page-probe-1", "/$defs/view_id/enum", views),
        ("page-probe-1", "/$defs/motif/enum", cat["motifs"]["set"]),
        ("page-probe-1", "/properties/lane/properties/view_key/enum", view_keys),
        ("page-probe-1", "/properties/lane/properties/link/enum", LINKS),
        ("attribution-2", "/properties/target/enum", cat["attribution"]["targets"]),
        ("attribution-2", "/properties/axis/properties/kind/enum", cat["attribution"]["axis_kinds"]),
        ("attribution-2", "/properties/reduce/enum", [None] + cat["attribution"]["reduce"]),
        ("attribution-2", "/properties/method/properties/name/enum", cat["attribution"]["methods"]),
        ("show-facts-1", "/$defs/fact/properties/kind/enum", cat["fact_kinds"]),
        ("show-facts-1", "/$defs/fact/properties/format/enum", cat["formats"]),
        ("show-facts-1", "/properties/groups/propertyNames/enum", cat["fact_groups"]),
        ("show-facts-1", "/properties/facts/propertyNames/pattern", cat["fact_key_pattern"]),
        ("show-facts-1", "/$defs/fact/properties/ref/pattern", cat["ref_pattern"]),
        ("show-facts-1", "/properties/task/properties/kind/enum", tasks),
        ("show-facts-1", "/properties/task/properties/modality/enum", modalities),
        ("show-facts-1", "/properties/task/properties/view_key/enum", view_keys),
        ("show-facts-1", "/properties/task/properties/sample_shape/enum", shapes),
        ("show-facts-1", "/properties/groups/properties/breakdown/oneOf/1/properties/kind/enum",
         cat["breakdown"]["kinds"]),
        ("story-2", "/$defs/lexicon/required", ["schema"] + lex["required"]["row"]),
        ("story-2", "/$defs/lexicon/properties/source_of/propertyNames/enum", lex["keys"]),
        ("story-2", "/$defs/entry/properties/one/pattern", lex["value_pattern"]),
        ("story-2", "/$defs/entry/properties/many/pattern", lex["value_pattern"]),
        ("story-2", "/$defs/entry/properties/article/enum", lex["articles"]["default"]),
        ("story-2", "/$defs/source_entry/properties/article/enum", lex["articles"]["source"]),
        ("story-2", "/$defs/role/enum", cat["roles"]["set"]),
        ("story-2", "/$defs/basis/pattern", cat["roles"]["basis_pattern"]),
        ("story-2", "/$defs/focus/oneOf/1/properties/ref/enum", cat["focus"]["families"]),
        ("story-2", "/$defs/breakdown/properties/kind/enum", cat["breakdown"]["kinds"]),
        ("story-2", "/properties/polarity/enum", cat["polarity"]["values"]),
        ("story-2", "/$defs/view_key/enum", view_keys),
        ("story-2", "/$defs/section_results/properties/models/items/properties/name/enum", models),
        ("story-2", "/$defs/section_results/properties/verdict/properties/strongest_baseline/enum",
         [m for m in models if m != "network"] + [None]),
        ("story-2", "/$defs/section_context/properties/steps/items/properties/id/enum",
         cat["context_steps"]["ids"]),
        ("story-2", "/properties/formats/additionalProperties/enum", cat["formats"]),
        ("story-2", "/$defs/key/pattern", cat["fact_key_pattern"]),
        ("story-2", "/$defs/ref/pattern", cat["ref_pattern"]),
        ("story-2", "/properties/locale/enum", cat["locales"]),
        ("story-2", "/$defs/code_note/properties/ref/oneOf/1/properties/kind/enum",
         cat["label_kinds"]["kinds"]),
        ("story-2", "/$defs/input_groups/items/properties/id/pattern",
         cat["input_groups"]["id_pattern"]),
        ("story-2", "/$defs/class_roles/properties/proposed/properties/rule/enum",
         [b for b in cat["roles"]["basis"] if b.startswith(("structural:", "task:"))] + ["none"]),
        ("story-2", "/properties/task/properties/kind/enum", tasks),
        ("story-plan-2", "/$defs/motif/enum", cat["motifs"]["set"]),
        ("story-plan-2", "/$defs/lexicon_key/enum", lex["keys"]),
        ("story-plan-2", "/$defs/context_step/enum", cat["context_steps"]["ids"]),
        ("story-plan-2", "/$defs/report_role/enum", roles),
        ("story-plan-2", "/properties/proposals/properties/modality/enum", modalities),
        ("story-plan-2", "/$defs/noun/pattern", lex["value_pattern"]),
        ("page-1", "/$defs/motif/enum", cat["motifs"]["set"]),
        ("page-1", "/properties/motif/properties/modality/enum", modalities),
        ("page-1", "/properties/motif/properties/source/pattern", cat["motifs"]["source_pattern"]),
        ("page-1", "/$defs/tour_view/enum", cat["views"]["tour"]),
        ("page-1", "/properties/recipe/properties/ops/items/properties/op/enum",
         cat["page"]["recipe_ops"]),
        ("page-1", "/properties/reference/oneOf/1/properties/kind/enum", cat["page"]["reference_kinds"]),
        ("page-1", "/properties/lane/properties/link/enum", LINKS),
        ("page-1", "/properties/lane/properties/view_key/enum", view_keys),
        ("page-1", "/properties/lane/properties/task/enum", tasks),
        ("page-1", "/properties/lane/properties/shape/enum", shapes),
        ("page-1", "/properties/lane/properties/level/enum", cat["page"]["levels"] + [None]),
    ]


def schema_lengths(cat):
    """(schema pointer in story-2, authored_lengths key) for every bounded text field."""
    lengths = cat["authored_lengths"]
    rows = [("/properties/headline", "headline"), ("/properties/lead", "lead"),
            ("/$defs/section_context/properties/steps/items/properties/text", "context_step"),
            ("/$defs/section_context/properties/facts/items/properties/label", "fact_label"),
            ("/$defs/section_network/properties/lead", "lead"),
            ("/$defs/section_network/properties/choices/items/properties/title", "choice_title"),
            ("/$defs/section_network/properties/choices/items/properties/text", "choice_text"),
            ("/$defs/section_experiments/properties/lead", "lead"),
            ("/$defs/section_experiments/properties/caption", "caption"),
            ("/$defs/section_experiments/properties/runs/items/properties/label", "run_label"),
            ("/$defs/section_results/properties/headline", "headline"),
            ("/$defs/section_results/properties/facts/items/properties/note", "note"),
            ("/$defs/section_results/properties/limits/items/properties/title", "limit_title"),
            ("/$defs/section_results/properties/limits/items/properties/text", "limit_text"),
            ("/$defs/breakdown/properties/caption", "caption"),
            ("/$defs/code_note/properties/note", "code_note"),
            ("/$defs/input_groups/items/properties/label", "group_label"),
            ("/$defs/input_groups/items/properties/text", "group_text"),
            ("/$defs/section_many/properties/monitor_note", "monitor_note")]
    return [(p, lengths[k]) for p, k in rows]


DELETE = object()


def mutations():
    """Seeded mutations of each schema's first example that the schema must refuse."""
    zero = "sha256:" + "0" * 64
    flow = {"one": "flow", "many": "flows", "article": "a"}
    fact = {"value": 1, "ref": "netspec.json#a", "kind": "count", "format": "count"}
    return {
        "showcase-1": [
            (["served", 0], "../MANIFEST.json"), (["files", "SHOWCASE.json"], zero),
            (["report", "status"], "linked"), (["entry"], "index.html"), (["motif"], "points"),
            (["bundle", "manifest_digest"], "sha256:xyz"), (["story", "schema"], "vega/story@1")],
        "story-2": [
            (["motif"], "points"), (["class_roles", "roles", "1"], "threat"),
            (["sections", 1, "choices", 0, "focus"], {"kind": "region", "ref": "hidden"}),
            (["lexicon", "row", "article"], "the"), (["lexicon", "flow"], flow),
            (["headline"], "x" * 161), (["sections", 3], DELETE),
            (["values", "primary", "ci"], DELETE),
            (["class_roles", "source", "1"], "domain:ontology"), (["lexicon", "row", "one"], "flow2")],
        "story-plan-2": [
            (["authored", "motif"], "packets"), (["status"], "final"),
            (["proposals", "motif", "base"], "graph")],
        "show-facts-1": [
            (["facts", "source_rows", "kind"], "guess"), (["groups", "chrome"], {}),
            (["facts", "Bad-Key"], fact), (["facts", "source_rows", "ref"], "evaluation:metrics.models")],
        "label-roles-1": [
            (["roles", "net.head"], {"label": "eq:x", "kind": "equation", "stage": "net"}),
            (["roles", "net.forward", "kind"], "listing"), (["adapter"], "json")],
        "page-1": [
            (["motif", "id"], "traces"), (["lane", "link"], "tanh"), (["views", "order", 0], "window"),
            (["motif", "id"], "packets"), (["recipe", "ops", 0, "op"], "magic")],
        "attribution-2": [(["target"], "true_class"), (["source", "part"], "test")],
        "page-probe-1": [
            (["views", "tried2"], {"rendered": True, "errors": []}),
            (["fetched", 0, "url"], "https://elsewhere.example/x"), (["motif", "id"], "graph")],
        "page-template-1": [
            (["files", "package.json"], zero), (["files", "tests/a.test.js"], zero),
            (["requires_endpoints", 0], "fetch /x"), (["files", "index.html"], DELETE)],
    }


def mutate(doc, path, value):
    out = copy.deepcopy(doc)
    node = out
    for part in path[:-1]:
        node = node[part]
    if value is DELETE:
        del node[path[-1]]
    elif isinstance(node, list) and path[-1] >= len(node):
        node.append(value)
    else:
        node[path[-1]] = value
    return out


def check_schemas(cat, forge):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        fail("jsonschema is not importable; every schema must validate as draft 2020-12 (gate CS2)")
        return
    listed = {Path(v["file"]).name: k for k, v in cat["schemas"].items()}
    on_disk = {p.name for p in SCHEMA_DIR.glob("*.json")}
    same("schema files on disk are the catalog's", on_disk, set(listed))
    loaded = {}
    for name, sid in sorted(listed.items()):
        schema = load_json(SCHEMA_DIR / name, f"schema {sid}")
        if schema is None:
            continue
        stem = name[: -len(".schema.json")]
        loaded[stem] = schema
        _check_one_schema(Draft202012Validator, sid, stem, schema)
    for stem, path, want in schema_sets(cat, forge):
        try:
            got = pointer(loaded[stem], path)
        except (KeyError, IndexError, TypeError):
            fail(f"{stem}{path} is missing")
            continue
        if isinstance(want, list):
            same(f"{stem}{path}", sorted(got, key=str), sorted(want, key=str))
        else:
            same(f"{stem}{path}", got, want)
    _check_schema_bounds(cat, loaded)


def _check_one_schema(validator_cls, sid, stem, schema):
    name, version = re.fullmatch(r"vega/([a-z-]+)@(\d+)", sid).groups()
    same(f"{stem} $schema", schema.get("$schema"), DRAFT_2020_12)
    same(f"{stem} $id", schema.get("$id"), f"urn:vega:schema:{name}:{version}")
    same(f"{stem} schema constant", (schema.get("properties", {}).get("schema") or {}).get("const"), sid)
    try:
        validator_cls.check_schema(schema)
    except Exception as exc:  # the meta-schema's own error type, whichever jsonschema raises
        fail(f"{stem} is not a valid draft 2020-12 schema: {str(exc).splitlines()[0]}")
        return
    validator = validator_cls(schema)
    examples = schema.get("examples") or []
    if not examples:
        fail(f"{stem} carries no example; the contract requires at least one that validates")
    for i, example in enumerate(examples):
        for err in list(validator.iter_errors(example))[:3]:
            fail(f"{stem} example {i} does not validate at {list(err.absolute_path)}: "
                 f"{err.message[:160]}")
    for path, value in mutations().get(stem, []) if examples else []:
        if validator.is_valid(mutate(examples[0], path, value)):
            shown = "deleted" if value is DELETE else repr(value)[:40]
            fail(f"{stem} accepts example 0 with {path} {shown}; the schema must refuse it")
    ok(f"{stem}: meta-valid, {len(examples)} example(s) valid, "
       f"{len(mutations().get(stem, []))} mutation(s) refused")


def _check_schema_bounds(cat, loaded):
    story = loaded.get("story-2")
    if not story:
        return
    for path, length in schema_lengths(cat):
        same(f"story-2{path} maxLength", pointer(story, path).get("maxLength"), length)
    same("story-2 input groups maxItems", pointer(story, "/$defs/input_groups").get("maxItems"),
         cat["input_groups"]["max"])
    same("story-2 network choices maxItems",
         pointer(story, "/$defs/section_network/properties/choices").get("maxItems"),
         cat["choices"]["max"])
    lex_props = set(pointer(story, "/$defs/lexicon/properties")) - {"schema", "source_of"}
    same("story-2 lexicon keys", sorted(lex_props), sorted(cat["lexicon"]["keys"]))
    kinds = [b["properties"]["kind"]["const"] for b in pointer(story, "/$defs/focus/oneOf")]
    same("story-2 focus kinds", sorted(kinds), sorted(cat["focus"]["kinds"]))
    page = loaded.get("page-1")
    if page:
        for branch in pointer(page, "/properties/motif/allOf"):
            modality = branch["if"]["properties"]["modality"]["const"]
            row = cat["motifs"]["by_modality"][modality]
            then = branch["then"]["properties"]
            same(f"page-1 motif rule for {modality}",
                 (then["base"]["const"], sorted(then["id"]["enum"])),
                 (row["base"], sorted(row["admissible"])))
    facts = loaded.get("show-facts-1")
    if facts:
        limit_id = "/properties/groups/properties/limits/items/properties/id/pattern"
        pattern = re.compile(pointer(facts, limit_id))
        for kind in cat["limit_kinds"]:
            probe = "warn_eval_1" if kind == "warn" else kind
            if not pattern.fullmatch(probe):
                fail(f"show-facts-1 refuses the limit id {probe!r}")
