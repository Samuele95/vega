"""
_sync_contract.py — gate CS1's first half: `show-contract.md` against `catalog.json`. Every
`| closed set | catalog key | members |` row and every `| bound | catalog key | value |` row is
compared with the key it names, and every structured table (forks, limits, groups, motifs, bases,
lexicon, validation rules, focus, view keys, context steps, polarity, results facts, breakdowns,
result models, layout, extensions, slot routes, report roles, walk roles, label kinds, briefs,
page keys, gates, schemas) with its catalog key. A helper of `test_catalog_sync.py`.
"""
from __future__ import annotations

import re

from _sync_common import dig, fail, same

# every closed-set and bound row the contract must carry; a row dropped from the prose is a
# rule nobody reads any more, even when the JSON still holds it
REQUIRED_SETS = {
    "stage.required_layers", "markers", "fact_sources", "fact_kinds", "formats", "derivations",
    "fact_groups", "name_tokens", "placeholder.suffixes", "lexicon.paths", "number_words",
    "locales", "build_locales", "motifs.set", "roles.set", "roles.alert_requires",
    "roles.evidence_only", "roles.anomaly_sides", "lexicon.keys", "lexicon.optional",
    "lexicon.entry", "choice_skip_types", "sections.ids", "sections.required", "focus.kinds",
    "focus.families", "breakdown.kinds", "context_steps.ids", "polarity.values",
    "result_models.names", "page.recipe_ops", "page.reference_kinds", "attribution.targets",
    "attribution.methods", "attribution.reduce", "attribution.axis_kinds", "showcase.roles",
    "showcase.served_roles", "showcase.unpinned_dirs", "showcase.report_status",
    "showcase.role_origins", "showcase_extensions.anywhere", "showcase_extensions.refused",
    "slot.routes", "slot.open_while_unready", "slot.reads", "slot.status_fields", "walk.head",
    "walk.tail", "label_kinds.kinds", "template.excluded", "probe.hook_members", "views.ids",
    "views.tour", "views.sub_views.network", "layer.required", "layer.surfaces", "gate_groups",
}
REQUIRED_BOUNDS = {
    "input_groups.min", "input_groups.max", "input_groups.min_applies_from_inputs",
    "input_groups.corr_threshold", "choices.min", "choices.max", "roles.neutral_majority_share",
    "motifs.alert_share", "slot.max_bytes", "layer.max_render_kb", "layer.max_data_kb",
}


def split_row(line):
    """The cells of one markdown table row; a pipe inside backticks is content."""
    cells, cur, tick = [], [], False
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    for ch in body:
        if ch == "`":
            tick = not tick
        if ch == "|" and not tick:
            cells.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def header_key(cells):
    return tuple(re.sub(r"[`*]", "", c).strip().lower() for c in cells)


def tables(md):
    """Every table of the contract as (header, rows), header cells lower-cased and unticked."""
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and re.fullmatch(r"\s*\|[\s|:-]+\|\s*", lines[i + 1]):
            head, rows, i = header_key(split_row(lines[i])), [], i + 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            out.append((head, rows))
            continue
        i += 1
    return out


def _rows_of(all_tables, header):
    """The rows of every table whose header is exactly `header` (a tuple of lower-cased,
    unticked cells)."""
    rows = [r for head, body in all_tables if head == header for r in body]
    if not rows:
        fail(f"the contract has no table headed {' | '.join(header)}")
    return rows


def ticks(cell):
    return re.findall(r"`([^`]+)`", cell)


def tick1(cell):
    found = ticks(cell)
    return found[0] if found else cell.strip()


def check_closed_sets(cat, all_tables):
    seen = set()
    for row in _rows_of(all_tables, ("closed set", "catalog key", "members")):
        key = tick1(row[1])
        seen.add(key)
        try:
            want = dig(cat, key)
        except KeyError:
            fail(f"the contract's closed set {row[0]!r} names {key}, which catalog.json lacks")
            continue
        same(f"closed set {key}", ticks(row[2]), want)
    for key in sorted(REQUIRED_SETS - seen):
        fail(f"the contract no longer states the closed set {key}")


def check_bounds(cat, all_tables):
    seen = set()
    for row in _rows_of(all_tables, ("bound", "catalog key", "value")):
        key = tick1(row[1])
        seen.add(key)
        try:
            want = dig(cat, key)
        except KeyError:
            fail(f"the contract's bound {row[0]!r} names {key}, which catalog.json lacks")
            continue
        same(f"bound {key}", float(row[2]), float(want))
    required = REQUIRED_BOUNDS | {f"authored_lengths.{k}" for k in cat["authored_lengths"]}
    for key in sorted(required - seen):
        fail(f"the contract no longer states the bound {key}")


def check_forks(cat, all_tables):
    rows = _rows_of(all_tables, ("fork", "name", "when", "attended", "--unattended"))
    prose = {tick1(r[0]): r for r in rows}
    same("fork ids (R-G3: S1–S3)", sorted(prose), sorted(cat["forks"]))
    expect = {"S1": "brief links", "S2": "plain", "S3": "base"}
    for fid, fork in cat["forks"].items():
        row = prose.get(fid)
        if not row:
            continue
        same(f"fork {fid} name", tick1(row[1]), fork["name"])
        if expect.get(fid, "") not in fork["unattended"] or expect.get(fid, "") not in row[4]:
            fail(f"fork {fid}: the unattended answer must say {expect.get(fid)!r} (R-G3) in both "
                 "the catalog and the contract")
        marker = fork["marker"]
        if fid == "S1" and marker.split(" — ")[0] not in row[4]:
            fail(f"fork S1: the contract's unattended cell does not carry the marker {marker!r}")


def check_limits_and_groups(cat, all_tables):
    wording = {"always": "always", "when present": "when_present", "never": "never"}
    rows = _rows_of(all_tables, ("limit", "when", "mandatory"))
    prose = {tick1(r[0]): wording.get(r[2].strip(), r[2].strip()) for r in rows}
    same("limit kinds", prose, {k: v["mandatory"] for k, v in cat["limit_kinds"].items()})
    rows = _rows_of(all_tables, ("group", "fact keys created", "applies to"))
    groups = [tick1(r[0]) for r in rows]
    same("fact groups (§2.3 rows)", groups, cat["fact_groups"])


def check_motifs_roles_lexicon(cat, all_tables):
    rows = _rows_of(all_tables, ("modality", "base", "admissible"))
    prose = {tick1(r[0]): {"base": tick1(r[1]), "admissible": ticks(r[2])} for r in rows}
    same("motifs by modality (R-E4)", prose, cat["motifs"]["by_modality"])
    rows = _rows_of(all_tables, ("basis", "grants", "rule"))
    same("role bases (R-E5)", {tick1(r[0]): ticks(r[1]) for r in rows},
         {k: v["grants"] for k, v in cat["roles"]["basis"].items()})
    rows = _rows_of(all_tables, ("shape", "required keys"))
    same("lexicon required keys (R-E6)", {tick1(r[0]): ticks(r[1]) for r in rows},
         cat["lexicon"]["required"])
    rows = _rows_of(all_tables, ("rule", "checks"))
    validate = [g["id"] for g in cat["gates"] if g["group"] == "validate"]
    same("validation rules (§3.8)", [tick1(r[0]) for r in rows], validate)


def check_story_tables(cat, all_tables):
    rows = _rows_of(all_tables, ("decision", "focus"))
    same("focus by decision", {tick1(r[0]): tick1(r[1]) for r in rows}, cat["focus"]["by_decision"])
    rows = _rows_of(all_tables, ("task", "view key"))
    prose = {}
    for row in rows:
        pairs = re.findall(r"`([a-z-]+)` → `([a-z-]+)`", row[1])
        prose[tick1(row[0])] = dict(pairs) if pairs else tick1(row[1])
    same("view keys", prose, cat["view_keys"])
    rows = _rows_of(all_tables, ("sample shape", "steps"))
    same("context steps", {tick1(r[0]): ticks(r[1]) for r in rows}, cat["context_steps"]["by_shape"])
    rows = _rows_of(all_tables, ("task", "with an alert role", "otherwise"))
    same("polarity", {tick1(r[0]): [tick1(r[1]), tick1(r[2])] for r in rows},
         cat["polarity"]["by_task"])
    rows = _rows_of(all_tables, ("task", "metrics"))
    same("results facts", {tick1(r[0]): ticks(r[1]) for r in rows}, cat["results_facts"])
    rows = _rows_of(all_tables, ("task", "breakdown"))
    same("breakdown by task", {tick1(r[0]): _breakdown_cell(r[1]) for r in rows},
         cat["breakdown"]["by_task"])
    rows = _rows_of(all_tables, ("model", "from the vega-eval baseline"))
    same("result models from baselines", {tick1(r[1]): tick1(r[0]) for r in rows},
         cat["result_models"]["from_baseline"])


def _breakdown_cell(cell):
    if cell.strip() == "none":
        return None
    if cell.strip() == "as the base":
        return "base"
    kind, metric, direction_of = (ticks(cell) + [None, None, None])[:3]
    return {"kind": kind, "metric": metric, "direction_of": direction_of}


def check_showcase_tables(cat, all_tables):
    rows = _rows_of(all_tables, ("path", "schema", "role", "served", "pinned", "when"))
    yes = {"yes": True, "no": False}
    prose = {tick1(r[0]): {"schema": None if r[1] == "—" else tick1(r[1]),
                           "role": None if r[2] == "—" else tick1(r[2]),
                           "served": yes.get(r[3]), "pinned": yes.get(r[4]), "when": r[5]}
             for r in rows}
    same("showcase layout (R-E1)", prose, cat["showcase"]["files"])
    anywhere, under, refused = [], {}, []
    for row in _rows_of(all_tables, ("extension", "where")):
        exts = [t for t in ticks(row[0]) if t.startswith(".")]
        dirs = [t for t in ticks(row[1]) if t.endswith("/")]
        if "refused" in row[1]:
            refused = exts
        elif not dirs:
            anywhere = exts
        for d in dirs:
            under[d] = exts
    ext = cat["showcase_extensions"]
    same("extensions anywhere", anywhere, ext["anywhere"])
    same("extensions by directory", under, ext["under"])
    same("refused extensions", refused, ext["refused"])
    routes = [tick1(r[0]) for r in _rows_of(all_tables, ("route", "serves"))]
    same("slot routes table (R-D6)", routes, cat["slot"]["routes"])


def check_report_tables(cat, all_tables):
    rows = _rows_of(all_tables, ("role", "kind", "stage", "what document-project labels"))
    same("report roles", {tick1(r[0]): {"kind": r[1].strip(), "stage": r[2].strip()} for r in rows},
         {k: {"kind": v["kind"], "stage": v["stage"]} for k, v in cat["report_roles"].items()})
    walk = {}
    for row in _rows_of(all_tables, ("walk id", "roles, in order")):
        for wid in ticks(row[0]):
            walk[wid] = ticks(row[1])
    same("walk roles (R-E8)", walk, cat["walk_roles"])
    rows = _rows_of(all_tables, ("anchor prefix", "kind"))
    same("anchor kinds", {tick1(r[0]): r[1].strip() for r in rows},
         cat["label_kinds"]["anchor_prefixes"])
    rows = _rows_of(all_tables, ("label prefix", "kind"))
    same("label-prefix kinds", {tick1(r[0]): r[1].strip() for r in rows},
         cat["label_kinds"]["label_prefixes"])
    rows = _rows_of(all_tables, ("stage", "brief read from"))
    same("brief stages", [tick1(r[0]) for r in rows], list(cat["briefs"]["stages"]))
    rows = _rows_of(all_tables, ("key", "for every member of"))
    same("page keys", [tick1(r[0]) for r in rows], list(cat["page_keys"]))


def check_gates_and_schemas(cat, all_tables):
    rows = _rows_of(all_tables, ("gate", "group", "where", "holds that", "exit", "seeded by"))
    prose = [{"id": tick1(r[0]), "group": r[1].strip(), "where": tick1(r[2]),
              "exit": None if r[4].strip() == "—" else int(r[4]), "cases": ticks(r[5])} for r in rows]
    want = [{k: g[k] for k in ("id", "group", "where", "exit", "cases")} for g in cat["gates"]]
    if len(prose) != len(want):
        fail(f"§10 lists {len(prose)} gates and catalog.json {len(want)}")
    for got, gate in zip(prose, want):
        same(f"gate {gate['id']}", got, gate)
    rows = _rows_of(all_tables, ("schema", "file", "written by", "read by"))
    same("schema index (§11)", {tick1(r[0]): tick1(r[1]) for r in rows},
         {k: v["file"] for k, v in cat["schemas"].items()})


CONTRACT_CHECKS = (check_closed_sets, check_bounds, check_forks, check_limits_and_groups,
                   check_motifs_roles_lexicon, check_story_tables, check_showcase_tables,
                   check_report_tables, check_gates_and_schemas)


def check_contract(cat, md):
    """Every contract table against the catalog."""
    all_tables = tables(md)
    for check in CONTRACT_CHECKS:
        check(cat, all_tables)
