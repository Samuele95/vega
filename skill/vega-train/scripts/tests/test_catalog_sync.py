#!/usr/bin/env python3
"""
test_catalog_sync.py — `scripts/catalog.json` is the machine copy of `training-catalog.md`
§2-§4, and this test fails when the two diverge.

    python3 test_catalog_sync.py [-v]

`plan_ladder.py` resolves every `vega:ladder/<rule>` against `ladder_rules`, `trainer.py`
evaluates every `D<n>` against `diagnostics`, and `hpo.py` builds its space from `spaces`. If
the prose gains a rule or a diagnostic and the JSON does not, a rung cites a rule no script
knows — which is the drift the machine copy exists to prevent.

Row *sets* are compared exactly; row *contents* are prose in one file and structure in the
other, so what is checked there is that each JSON row carries the fields the prose's columns
promise.

What this catalog says about `/vega-forge`'s catalog is checked against it too: every search
space covers a forge family and its `<… id>` placeholders resolve to block types the forge
defines; the recurrent and graph families have their own rows (D1 §12.4, E21); the
`recurrent-floor` row names every family `families/rnn.ablation_floor_for` lists; the floor's
inserts are blocks of the `linear` composition on shapes its `floor_shapes` admit (R-A12); the
PCA floor's insert is the `dense(z, none)` of the `pca-floor` composition (D1 §6); and
`run_ids` gives the floor and the recurrent floor the ids `plan_ladder.py` numbers them with.

Exit 0 when the two agree.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILL = SCRIPTS.parent
CATALOG_JSON = SCRIPTS / "catalog.json"
CATALOG_MD = SKILL / "references" / "training-catalog.md"
FORGE_CATALOG = SKILL.parent / "vega-forge" / "scripts" / "catalog.json"
VERBOSE = "-v" in sys.argv
sys.path.insert(0, str(SCRIPTS))
from expectations import ExpectCheck, ExpectError  # noqa: E402

failures = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    if VERBOSE:
        print(f"  ok    {msg}")


def section(text, heading):
    match = re.search(rf"^##\s*{re.escape(heading)}", text, re.M)
    if not match:
        return ""
    tail = text[match.end():]
    nxt = re.search(r"^##\s", tail, re.M)
    return tail[:nxt.start()] if nxt else tail


def table_rows(body):
    """(first cell, whole line) for every table row that is not the header or the rule."""
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or not cells[0] or cells[0].lower() in ("rule", "code", "family", "item"):
            continue
        out.append((cells[0], cells))
    return out


def _families_of(key, forge_families):
    """The forge families a space row's key covers: itself, or every family a `prefix-*`
    wildcard matches."""
    if key.endswith("*"):
        return sorted(f for f in forge_families if f.startswith(key[:-1]))
    return [key] if key in forge_families else []


def forge_checks(md, data):
    """What this catalog says about the forge's catalog must agree with it: every search
    space names a real family and real block types, the families the forge left without an
    HPO row (E21) have one, and the floors this ladder builds are the compositions the forge
    admits (R-A12)."""
    forge = json.loads(FORGE_CATALOG.read_text(encoding="utf-8"))
    families, blocks = forge.get("families") or {}, forge.get("blocks") or {}
    sys.path.insert(0, str(SCRIPTS))
    import plan_ladder as PL

    # every space row covers a forge family; every placeholder resolves to real block types
    covered, used = set(), {}
    for key, dims in (data.get("spaces") or {}).items():
        hit = _families_of(key, families)
        if not hit:
            fail(f"spaces/{key} covers no family of the forge catalog")
        covered.update(hit)
        for dim in dims:
            for holder in re.findall(r"<[^>]+>", dim.get("path", "")):
                used.setdefault(holder, f"spaces/{key}: {dim['path']}")
    for holder, where in sorted(used.items()):
        rule = (data.get("space_placeholders") or {}).get(holder)
        if rule is None:
            fail(f"{where} uses {holder}, which space_placeholders does not define")
            continue
        for kind in PL.placeholder_types(rule):
            if kind not in blocks:
                fail(f"space_placeholders/{holder} names block type {kind!r}, which the forge "
                     "catalog does not define")
    # E21: the recurrent and graph families have their own rows, not the generic space
    rnn = families.get("rnn") or {}
    graph = [f for f, row in families.items() if (row or {}).get("sample_shape") == "graph"]
    for family in ["rnn", *(rnn.get("ablation_floor_for") or []), *graph]:
        if family not in covered:
            fail(f"the {family} family has no search space; §4 gives it one (D1 §12.4, E21)")
    if not graph:
        fail("the forge catalog names no graph family (sample_shape graph)")

    # R1b: the prose names exactly the families the forge's rnn row is the floor of
    floor_for = set(rnn.get("ablation_floor_for") or [])
    rows = dict(table_rows(section(md, "2. Ladder rules")))
    cells = next((v for k, v in rows.items() if "recurrent-floor" in k), None)
    named = set(re.findall(r"`([a-z*][a-z0-9*-]*)`", cells[1])) if cells else set()
    if not floor_for:
        fail("the forge catalog's rnn row names no ablation_floor_for families")
    elif not floor_for <= named:
        fail(f"§2's recurrent-floor row does not name {sorted(floor_for - named)}, which the "
             "forge catalog's families/rnn.ablation_floor_for lists")
    # the floor's inserts are blocks the linear composition holds, on the shapes it admits
    linear = families.get("linear") or {}
    inserts = ((data.get("ladder_rules") or {}).get("floor") or {}).get("inserts") or {}
    shapes = set(linear.get("floor_shapes") or [])
    for key, listed in inserts.items():
        if key.split("/")[0] not in shapes:
            fail(f"ladder_rules/floor/inserts/{key}: {key.split('/')[0]} is not one of the "
                 f"linear floor_shapes {sorted(shapes)}")
        for block in listed:
            if block.get("type") not in (linear.get("block_types") or []):
                fail(f"ladder_rules/floor/inserts/{key}: {block.get('type')!r} is not a block "
                     "type of the linear family")
    for key in ("window", "graph/graph"):
        if not inserts.get(key):
            fail(f"ladder_rules/floor/inserts/{key} is empty; R-A12 inserts a block there")
    # the PCA floor's bottleneck is the dense(z, none) of the forge's pca-floor composition
    pca = families.get("pca-floor") or {}
    pca_inserts = ((data.get("ladder_rules") or {}).get("floor") or {}).get("pca_inserts") or []
    if not pca_inserts:
        fail("ladder_rules/floor/pca_inserts is empty; the pca-floor composition holds "
             "dense(z, none) before the head (D1 §6)")
    for block in pca_inserts:
        kind, params = block.get("type"), block.get("params") or {}
        if kind not in (pca.get("block_types") or []):
            fail(f"ladder_rules/floor/pca_inserts: {kind!r} is not a block type of the "
                 "pca-floor family")
        wanted = f"{kind}({params.get('units')}, {params.get('activation')})"
        if wanted.replace(" ", "") not in str(pca.get("composition", "")).replace(" ", ""):
            fail(f"ladder_rules/floor/pca_inserts: {wanted} is not in the pca-floor "
                 f"composition {pca.get('composition')!r}")
    # the ids the rules own are the ids plan_ladder gives
    run_ids = data.get("run_ids") or {}
    for source, rung_id in PL.FIXED_IDS.items():
        rule = source.split("/", 1)[1]
        if run_ids.get(rule) != rung_id:
            fail(f"run_ids/{rule} is {run_ids.get(rule)!r}; plan_ladder numbers the {rule} "
                 f"rung {rung_id}")
    ok("the forge catalog agrees: spaces, placeholders, E21 rows, R1b families, floor inserts")


def main():
    md = CATALOG_MD.read_text(encoding="utf-8")
    data = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    print(f"comparing {CATALOG_JSON.name} with {CATALOG_MD.name}")

    # --- §2 ladder rules
    body = section(md, "2. Ladder rules")
    prose_rules = set()
    for first, _ in table_rows(body):
        for name in re.findall(r"`([^`]+)`", first) or [first]:
            prose_rules.add(name.strip())
    json_rules = set(data["ladder_rules"])
    for missing in sorted(prose_rules - json_rules):
        fail(f"ladder rule {missing!r} is in the prose's §2 but not in catalog.json")
    for extra in sorted(json_rules - prose_rules):
        fail(f"ladder rule {extra!r} is in catalog.json but not in the prose's §2")
    if prose_rules == json_rules:
        ok(f"§2: {len(json_rules)} ladder rules agree")
    for name, row in data["ladder_rules"].items():
        if name == "selection":
            for field in ("rule", "alternative", "ineligible_codes"):
                if field not in row:
                    fail(f"ladder_rules/selection carries no {field!r}")
            continue
        for field in ("rung", "hypothesis", "expect"):
            if field not in row:
                fail(f"ladder_rules/{name} carries no {field!r}; §2's columns promise the rung, "
                     "the hypothesis it tests and the expectation")
        # the expectation's checkable form, which the trainer judges expect_held by
        # (run-contract §3); a rule without one plans rungs whose outcome no script can record
        try:
            ExpectCheck.parse(row.get("expect_check"))
        except ExpectError as exc:
            fail(f"ladder_rules/{name}/expect_check: {exc}")
    for rule in data.get("unattended_ladder") or []:
        if rule not in json_rules:
            fail(f"unattended_ladder names {rule!r}, which is not a ladder rule")

    # --- §3 diagnostics
    body = section(md, "3. Diagnostics")
    prose_codes = set()
    for first, _ in table_rows(body):
        match = re.match(r"(D\d+)", first)
        if match:
            prose_codes.add(match.group(1))
    json_codes = set(data["diagnostics"])
    for missing in sorted(prose_codes - json_codes):
        fail(f"diagnostic {missing!r} is in the prose's §3 but not in catalog.json")
    for extra in sorted(json_codes - prose_codes):
        fail(f"diagnostic {extra!r} is in catalog.json but not in the prose's §3")
    if prose_codes == json_codes:
        ok(f"§3: {len(json_codes)} diagnostics agree")
    for code, row in data["diagnostics"].items():
        for field in ("name", "symptom", "reading"):
            if field not in row:
                fail(f"diagnostics/{code} carries no {field!r}; §3's columns promise the "
                     "symptom, the reading and the rung it proposes")
        if "remedies" not in row:
            fail(f"diagnostics/{code} carries no `remedies` list (empty is legitimate — D10 "
                 "and D11 propose no rung — but the key says so explicitly)")

    # every diagnostic the code evaluates must be a catalog row
    sys.path.insert(0, str(SCRIPTS))
    import diagnostics as DX
    for code in DX.CODES:
        if code not in json_codes:
            fail(f"diagnostics.py evaluates {code!r}, which catalog.json does not define")
    for code in json_codes:
        if code not in DX.CODES:
            fail(f"catalog.json defines {code!r}, which diagnostics.py never evaluates")

    # --- §4 spaces
    body = section(md, "4. Search spaces")
    prose_families = set()
    for first, _ in table_rows(body):
        for name in re.findall(r"`([^`]+)`", first):
            prose_families.add(name.strip())
    json_families = set(data["spaces"])
    for missing in sorted(prose_families - json_families):
        fail(f"search space for {missing!r} is in the prose's §4 but not in catalog.json")
    for extra in sorted(json_families - prose_families):
        fail(f"search space for {extra!r} is in catalog.json but not in the prose's §4")
    if prose_families == json_families:
        ok(f"§4: {len(json_families)} search spaces agree")
    for family, dims in data["spaces"].items():
        if not any(d["path"] == "optimizer/params/lr" for d in dims):
            fail(f"spaces/{family} does not search lr, and §4 says lr is always searched "
                 "because every other dimension's effect depends on it")
        for dim in dims:
            if "path" not in dim or "kind" not in dim:
                fail(f"spaces/{family} has a dimension with no path or kind")
            if dim.get("kind") == "categorical" and "choices" not in dim:
                fail(f"spaces/{family}: {dim.get('path')} is categorical with no choices")
            elif dim.get("kind") in ("float", "int") and "range" not in dim:
                fail(f"spaces/{family}: {dim.get('path')} has no range")
    for key, rule in (data.get("space_placeholders") or {}).items():
        if not rule.get("block_type"):
            fail(f"space_placeholders/{key} names no block_type to resolve it from")

    forge_checks(md, data)

    if failures:
        print(f"\n{len(failures)} divergence(s) between the prose and the machine copy")
        return 1
    print(f"\ncatalog.json and training-catalog.md agree: {len(data['ladder_rules'])} ladder "
          f"rules, {len(data['diagnostics'])} diagnostics, {len(data['spaces'])} search spaces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
