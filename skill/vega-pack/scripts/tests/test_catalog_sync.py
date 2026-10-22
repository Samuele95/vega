#!/usr/bin/env python3
"""
test_catalog_sync.py — `scripts/catalog.json` is the machine copy of `bundle-contract.md`
§1–§7, and this test fails when the two diverge.

    python3 test_catalog_sync.py [-v]

`pack.py` builds the bundle from the catalog's file list, its package slice and its
round-trip formula; `apply_bundle_layer.py` reads its caps. If the prose gains a required file
and the JSON does not, a bundle ships without it and every gate still passes — which is the
drift the machine copy exists to prevent, and the reason the list is not simply written out in
code.

From v2.3 the catalog also holds what serving reads: the ten readings of `vega/output@2` and the
rules that resolve a task to one (the rules are executed here on representative specs, not only
compared as text), the field kinds the answer codec encodes, the sample shapes, the monitor
kinds per reading, the serving template patterns and what a re-emit may change. Each is checked
against the contract section that states it; the example manifest of §2 is parsed and held to
the records; `references/schemas/examples-2.schema.json` is checked against the catalog's
preset and stratum vocabularies and run on good payloads (one per demo family, a multilabel
`none` stratum among them) and on bad ones; and the link vocabulary and the reading → link
table are compared with the ones vega-forge's catalog owns (`inspection.links`, R-C1).

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
CATALOG_MD = SKILL / "references" / "bundle-contract.md"
VERBOSE = "-v" in sys.argv

failures = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    if VERBOSE:
        print(f"  ok    {msg}")


def section(md, heading):
    """The text of one numbered section, up to the next one."""
    i = md.find(heading)
    if i < 0:
        return ""
    j = md.find("\n## ", i + len(heading))
    return md[i:j if j > 0 else len(md)]


def tree_files(text):
    """The file names the §1 directory listing names, ignoring its commentary.

    The listing is a code block of `name  ← what it is` lines with the tree's own indentation;
    what matters is the left-hand token, and only where it looks like a file rather than a
    directory or a prose fragment.
    """
    block = re.search(r"```\n([\s\S]*?)```", text)
    if not block:
        return set()
    names = set()
    for line in block.group(1).splitlines():
        # everything left of the arrow is the tree; the commentary to its right names files
        # too (`predictions.test.parquet`) and those are not part of the bundle
        left = line.split("←")[0]
        if not left.strip() or left.strip().startswith(("#", "bundle/")):
            continue
        # a line may carry two names (`recipe.json  fitted.json`), so take every token that
        # looks like a file rather than only the first
        for token in left.split():
            if token.endswith("/"):
                continue
            if re.fullmatch(r"[A-Za-z0-9_./<>-]+\.[A-Za-z0-9<>]+", token):
                names.add(token)
    return names


# --------------------------------------------------------------------------- v2.3 helpers

def subsection(md, heading):
    """The text of one `### n.m` subsection, up to the next subsection or section."""
    i = md.find(heading)
    if i < 0:
        return ""
    ends = [k for k in (md.find("\n### ", i + len(heading)), md.find("\n## ", i + len(heading)))
            if k > 0]
    return md[i:min(ends) if ends else len(md)]


def table_rows(text):
    """The body rows of every markdown table in `text`, as lists of stripped cells."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.fullmatch(r"\|[\s|:-]+\|", line):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def ticks(text):
    """Every backticked span in `text`."""
    return set(re.findall(r"`([^`\n]+)`", text))


def tree_tokens(md):
    """Every whitespace-separated token left of an arrow in §1's directory listing."""
    block = re.search(r"```\n([\s\S]*?)```", section(md, "## 1. The bundle directory"))
    out = []
    for line in (block.group(1).splitlines() if block else []):
        out += [t for t in line.split("←")[0].split() if t != "|"]
    return out


def _head_out(spec):
    heads = [b for b in spec.get("blocks") or [] if b.get("type") == "head"]
    try:
        return int((heads[-1].get("params") or {}).get("out")) if heads else None
    except (TypeError, ValueError):
        return None


def _dotted(spec, path):
    node = spec
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _holds(when, spec):
    """Whether a resolve rule's `when` holds for a spec; an unknown condition raises."""
    loss = (spec.get("loss") or {}).get("name")
    head_out = _head_out(spec)
    for key, want in (when or {}).items():
        if key == "loss":
            if loss not in want:
                return False
        elif key == "head_out_gt":
            if head_out is None or not head_out > want:
                return False
        elif key == "head_out_eq":
            if head_out != want:
                return False
        elif key == "has":
            if _dotted(spec, want) is None:
                return False
        else:
            raise KeyError(key)
    return True


def resolve_reading(catalog, spec, kind=None):
    """The catalog's resolution rules, executed: first match wins, `@base` recurses."""
    task = spec.get("task") or {}
    kind = kind or task.get("kind")
    for rule in catalog["output_readings"]["resolve"]:
        if rule["kind"] != kind or not _holds(rule.get("when"), spec):
            continue
        if rule["reading"] != "@base":
            return rule["reading"]
        base = rule.get("base") or {}
        chosen = task.get("base_kind") if base.get("from") == "task.base_kind" else None
        if chosen is None:
            for fallback in base.get("v22_fallback") or []:
                if _holds(fallback.get("when"), spec):
                    chosen = fallback["base"]
                    break
        if chosen not in (base.get("allowed") or []):
            return None
        return resolve_reading(catalog, spec, kind=chosen)
    return None


def _pattern_re(pattern):
    """A field pattern (`p_<class>`) as a regex over concrete field names."""
    parts = re.split(r"(<[a-z_]+>)", pattern)
    return re.compile("".join(".+" if p.startswith("<") else re.escape(p) for p in parts) + "$")


def _normal(pattern):
    """A score_field pattern mapped onto the field pattern it instantiates."""
    return (pattern.replace("<positive>", "<class>").replace("<median>", "<q>")
            .replace("_h1", "_h<h>"))


def _sibling_catalog(name):
    path = SKILL.parent / name / "scripts" / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# --------------------------------------------------------------------------- v2.3 checks

def check_readings(catalog, md):
    """§2.1: every reading is named, with the catalog's link, and its record is coherent."""
    table = catalog["output_readings"]
    readings = table["readings"]
    kinds = catalog["field_kinds"]
    sub = subsection(md, "### 2.1 Readings")
    if not sub:
        fail("bundle-contract §2.1 (Readings) is missing; catalog.json holds "
             f"{len(readings)} reading(s) and nothing states them")
        return
    rows = {r[0].strip("`"): r for r in table_rows(sub)
            if len(r) >= 6 and r[0].strip("`") in readings}
    for name, row in readings.items():
        where = f"reading {name!r}"
        if name not in rows:
            fail(f"catalog.json defines the {where} and §2.1's table has no row for it")
            continue
        prose = rows[name]
        if f"`{row['link']}`" not in prose[1]:
            fail(f"{where}: catalog link {row['link']!r}, §2.1 says {prose[1]!r}")
        for pattern in row["fields"]:
            if f"`{pattern}`" not in prose[3]:
                fail(f"{where}: the field pattern {pattern!r} is not in §2.1's fields cell")
        if row["link"] not in table["links"]:
            fail(f"{where}: link {row['link']!r} is not in the link vocabulary {table['links']}")
        if row["decision"] not in table["decisions"]:
            fail(f"{where}: decision {row['decision']!r} is not one of {table['decisions']}")
        thresholded = row["decision"] in ("threshold", "per-label-threshold")
        if thresholded != (row["threshold_scale"] is not None):
            fail(f"{where}: a {row['decision']} decision with threshold_scale "
                 f"{row['threshold_scale']!r}; a threshold needs its scale and nothing else "
                 "has one")
        if list(row["kinds"]) != row["fields"] or set(row["describe"]) != set(row["fields"]):
            fail(f"{where}: `kinds` and `describe` must name exactly the fields, in order")
        for pattern, fk in row["kinds"].items():
            if fk.get("kind") not in kinds:
                fail(f"{where}: field {pattern!r} has kind {fk.get('kind')!r}, not a field kind")
        for pattern in re.findall(r"<[a-z_]+>", " ".join(row["fields"])):
            if pattern not in table["placeholders"]:
                fail(f"{where}: placeholder {pattern} is not defined in `placeholders`")
        fields = set(row["fields"])
        for bundle_field, stored in row["compared"]:
            if bundle_field not in fields:
                fail(f"{where}: compares {bundle_field!r}, which is not one of its fields")
            if stored not in ("score", "row_score", "score_<k>"):
                fail(f"{where}: compares against {stored!r}, which the stored layout lacks")
        for bundle_field, source in row["decided"]:
            if bundle_field not in fields:
                fail(f"{where}: decides {bundle_field!r}, which is not one of its fields")
            if source != "prediction" and source not in table["decided_sources"]:
                fail(f"{where}: decided from {source!r}, neither a stored column nor one of "
                     f"{table['decided_sources']}")
        for pattern in row["score_field"]:
            if _normal(pattern) not in fields:
                fail(f"{where}: score_field {pattern!r} names no field of the reading")
        if not set(row["conditional"]) <= fields:
            fail(f"{where}: `conditional` names {sorted(set(row['conditional']) - fields)}, "
                 "which are not its fields")
        if row["truth"] not in table["truth_types"] + [None]:
            fail(f"{where}: truth {row['truth']!r} is not one of {table['truth_types']}")
        if row["truth"] is None and not row["truth_optional"]:
            fail(f"{where}: no truth type and yet truth is not optional")
        if row["sums_to_one"] and "p_<class>" not in fields:
            fail(f"{where}: sums_to_one without per-class probabilities")
    extra = sorted(set(rows) - set(readings))
    if extra:
        fail(f"§2.1 lists reading(s) {extra} that catalog.json does not define")
    for key in table["record_keys"]:
        if f"`{key}`" not in sub:
            fail(f"vega/output@2 key {key!r} is not named in §2.1")
    for key, row in table["shape_fields"].items():
        if f"`{key}`" not in sub or f"`{row['kind']}`" not in sub:
            fail(f"§2.1 does not name the shape field {key!r} (kind {row['kind']!r})")
    ok(f"{len(readings)} reading(s): link, decision, fields, compared, decided and kinds "
       "agree with §2.1")


def check_resolution(catalog, md):
    """§2.1: the rules R1–R12, executed on representative specs."""
    table = catalog["output_readings"]
    rules = table["resolve"]
    ids = [r["rule"] for r in rules]
    if ids != [f"R{i}" for i in range(1, len(rules) + 1)]:
        fail(f"the resolve rules are not R1…R{len(rules)} in order: {ids}")
    for rule in rules:
        whens = [rule.get("when")] + [f.get("when") for f in
                                      (rule.get("base") or {}).get("v22_fallback") or []]
        for when in whens:
            unknown = set(when or {}) - set(table["conditions"])
            if unknown:
                fail(f"{rule['rule']}: unknown condition(s) {sorted(unknown)}")
        if rule["reading"] not in table["readings"] and rule["reading"] != "@base":
            fail(f"{rule['rule']}: reading {rule['reading']!r} is not defined")
    for kind in {r["kind"] for r in rules}:
        if not any(r["kind"] == kind and not r.get("when") for r in rules):
            fail(f"task.kind {kind!r} has no unconditional rule, so some spec of it resolves "
                 "to nothing")
    forge = _sibling_catalog("vega-forge")
    if forge:
        by_kind = {}
        for base, kind in (forge.get("sequence_map") or {}).items():
            by_kind.setdefault(kind, set()).add(base)
        for rule in rules:
            base = rule.get("base") or {}
            if rule["reading"] == "@base" and set(base.get("allowed") or []) != \
                    by_kind.get(rule["kind"], set()):
                fail(f"{rule['rule']}: allowed bases {base.get('allowed')} differ from "
                     f"vega-forge's sequence_map {sorted(by_kind.get(rule['kind'], set()))}")
    head = lambda n: [{"type": "dense"}, {"type": "head", "params": {"out": n}}]  # noqa: E731
    cases = [
        ({"task": {"kind": "binary"}}, "binary"),
        ({"task": {"kind": "multiclass"}}, "multiclass"),
        ({"task": {"kind": "multilabel"}}, "multilabel"),
        ({"task": {"kind": "ordinal"}}, "ordinal"),
        ({"task": {"kind": "regression"}, "loss": {"name": "quantile"}, "blocks": head(3)},
         "quantile"),
        ({"task": {"kind": "regression"}, "loss": {"name": "quantile"}, "blocks": head(1)},
         "regression"),
        ({"task": {"kind": "regression"}, "loss": {"name": "mse"}, "blocks": head(1)},
         "regression"),
        ({"task": {"kind": "unsupervised"}}, "reconstruction"),
        ({"task": {"kind": "anomaly"}, "loss": {"name": "svdd"}}, "anomaly-svdd"),
        ({"task": {"kind": "anomaly"}, "loss": {"name": "vae"}}, "anomaly-reconstruction"),
        ({"task": {"kind": "sequence-classification", "base_kind": "multilabel"}},
         "multilabel"),
        ({"task": {"kind": "sequence-classification", "base_kind": "binary"},
          "blocks": head(1)}, "binary"),
        ({"task": {"kind": "sequence-regression", "base_kind": "regression"},
          "loss": {"name": "quantile"}, "blocks": head(3)}, "quantile"),
        ({"task": {"kind": "sequence-regression", "base_kind": "ordinal"}}, "ordinal"),
        ({"task": {"kind": "sequence-regression", "base_kind": "multiclass"}}, None),
        ({"task": {"kind": "sequence-classification"}, "blocks": head(1)}, "binary"),
        ({"task": {"kind": "sequence-classification"}, "loss": {"name": "bce_with_logits"},
          "blocks": head(4)}, "multilabel"),
        ({"task": {"kind": "sequence-classification"}, "loss": {"name": "cross_entropy"},
          "blocks": head(4)}, "multiclass"),
        ({"task": {"kind": "sequence-regression"}, "loss": {"name": "cumulative_link"}},
         "ordinal"),
        ({"task": {"kind": "forecast"}, "blocks": head(3)}, "forecast"),
        ({"task": {"kind": "clustering"}}, None),
    ]
    for spec, want in cases:
        got = resolve_reading(catalog, spec)
        if got != want:
            fail(f"the resolve rules give {got!r} for {spec['task']} "
                 f"(loss {(spec.get('loss') or {}).get('name')!r}); §2.1 says {want!r}")
    reached = {w for _, w in cases if w}
    unreached = sorted(set(table["readings"]) - reached)
    if unreached:
        fail(f"reading(s) {unreached} are reached by no resolve case")
    sub = subsection(md, "### 2.1 Readings")
    prose = {r[0]: r for r in table_rows(sub) if re.fullmatch(r"R\d+", r[0])}
    for rule in rules:
        row = prose.get(rule["rule"])
        if not row:
            fail(f"§2.1's resolution table has no row {rule['rule']}")
            continue
        if f"`{rule['kind']}`" not in row[1]:
            fail(f"{rule['rule']}: catalog kind {rule['kind']!r}, §2.1 says {row[1]!r}")
        if rule["reading"] == "@base":
            if "base" not in row[3]:
                fail(f"{rule['rule']}: the catalog reads the base, §2.1 says {row[3]!r}")
        elif f"`{rule['reading']}`" not in row[3]:
            fail(f"{rule['rule']}: catalog reading {rule['reading']!r}, §2.1 says {row[3]!r}")
    if "has no serving reading" not in sub:
        fail("§2.1 no longer states the refusal of a kind no rule resolves")
    ok(f"{len(rules)} resolve rule(s) agree with §2.1 and resolve {len(cases)} spec(s) as it "
       "says")


def check_field_kinds(catalog, md):
    """§7.3: every field kind, its JSON key, and the answer keys per reading."""
    kinds = catalog["field_kinds"]
    readings = catalog["output_readings"]["readings"]
    used = {fk["kind"] for row in readings.values() for fk in row["kinds"].values()}
    used |= {row["kind"] for row in catalog["output_readings"]["shape_fields"].values()}
    if used - set(kinds):
        fail(f"field kind(s) {sorted(used - set(kinds))} are used and not defined")
    if set(kinds) - used:
        fail(f"field kind(s) {sorted(set(kinds) - used)} are defined and used by no field")
    sub = subsection(md, "### 7.3 Answers per reading")
    if not sub:
        fail("bundle-contract §7.3 (Answers per reading) is missing")
        return
    # the field-kind table has four columns; the answer-keys table below it has two, and a
    # reading can share a kind's name (`quantile`), so the width tells the tables apart
    rows = {r[0].strip("`"): r for r in table_rows(sub)
            if len(r) == 4 and r[0].strip("`") in kinds}
    for name, row in kinds.items():
        prose = rows.get(name)
        if not prose:
            fail(f"field kind {name!r} is not named in §7.3's table")
            continue
        key = row["key"]
        if (key is None and "—" not in prose[1]) or (key and f"`{key}`" not in prose[1]):
            fail(f"field kind {name!r}: catalog key {key!r}, §7.3 says {prose[1]!r}")
        shown = row["shown"]
        if shown.startswith("include="):
            value = shown.split("=", 1)[1]
            if value not in catalog["include_values"]:
                fail(f"field kind {name!r} is shown on include={value}, not an include value")
            if f"?include={value}" not in prose[3]:
                fail(f"field kind {name!r}: §7.3 does not say it is shown on ?include={value}")
    for value in catalog["include_values"]:
        if f"?include={value}" not in sub:
            fail(f"the include value {value!r} is not named in §7.3")
    answers = [r for r in table_rows(sub) if len(r) == 2]
    silent = {"decision", "warnings", "answered", "group"}
    for name, row in readings.items():
        prose = [r for r in answers if f"`{name}`" in r[0]]
        if not prose:
            fail(f"§7.3 gives no answer keys for the reading {name!r}")
            continue
        for fk in row["kinds"].values():
            key = kinds[fk["kind"]]["key"]
            if fk["kind"] not in silent and key and f"`{key}`" not in prose[0][1]:
                fail(f"reading {name!r}: its {fk['kind']!r} field answers under {key!r}, which "
                     "§7.3's keys row does not name")
    ok(f"{len(kinds)} field kind(s) and the answer keys of {len(readings)} reading(s) agree "
       "with §7.3")


def check_monitor_kinds(catalog, md):
    """§7.5: every monitor kind with the readings that admit it."""
    monitor = catalog["monitor_kinds"]
    readings = catalog["output_readings"]["readings"]
    sub = subsection(md, "### 7.5 The monitor per reading")
    rows = {r[0].strip("`"): r for r in table_rows(sub) if r[0].strip("`") in monitor}
    for name, row in monitor.items():
        admitted = row.get("readings")
        if admitted is not None and set(admitted) - set(readings):
            fail(f"monitor kind {name!r} admits unknown reading(s) "
                 f"{sorted(set(admitted) - set(readings))}")
        prose = rows.get(name)
        if not prose:
            fail(f"monitor kind {name!r} is not named in §7.5's table")
            continue
        if admitted is None:
            if "every reading" not in prose[2]:
                fail(f"monitor kind {name!r} applies to every reading and §7.5 says "
                     f"{prose[2]!r}")
        elif set(re.findall(r"`([^`]+)`", prose[2])) != set(admitted):
            fail(f"monitor kind {name!r}: catalog readings {sorted(admitted)}, §7.5 says "
                 f"{prose[2]!r}")
    with_score = sorted(n for n, r in readings.items() if r["score_field"])
    if sorted(monitor["output_psi"]["readings"]) != with_score:
        fail(f"output_psi reads the score field and is admitted for "
             f"{sorted(monitor['output_psi']['readings'])}; the readings with a score field "
             f"are {with_score}")
    for kind, truth in (("class_recall", "class"), ("residual", "number")):
        wrong = [r for r in monitor[kind]["readings"] if readings[r]["truth"] != truth]
        if wrong:
            fail(f"{kind} needs a {truth} truth and is admitted for {wrong}")
    if "not admitted" not in sub:
        fail("§7.5 no longer states that a kind the reading does not admit is refused")
    ok(f"{len(monitor)} monitor kind(s) agree with §7.5")


def check_sample_shapes(catalog, md):
    """§2.2–§2.3: the sample records, the graph reference files."""
    shapes = catalog["sample_shapes"]
    forge = _sibling_catalog("vega-forge")
    if forge and "sample_shapes" in forge and set(forge["sample_shapes"]) != set(shapes):
        fail(f"catalog.json knows the sample shapes {sorted(shapes)}; vega-forge compiles "
             f"{sorted(forge['sample_shapes'])}")
    sub = subsection(md, "### 2.2 `MANIFEST.sample`")
    block = re.search(r"```json\n([\s\S]*?)```", sub)
    found = {}
    for chunk in (block.group(1).split("\n\n") if block else []):
        try:
            record = json.loads("{" + chunk.strip() + "}")["sample"]
        except (ValueError, KeyError) as exc:
            fail(f"§2.2 has a sample example that is not JSON: {exc}")
            continue
        found[record.get("shape")] = record
    if set(found) != set(shapes):
        fail(f"§2.2 shows sample records for {sorted(found)}; catalog.json defines "
             f"{sorted(shapes)}")
    for name, record in found.items():
        want = shapes.get(name, {}).get("record") or []
        if list(record) != want:
            fail(f"§2.2's {name} sample record has keys {list(record)}; catalog.json says "
                 f"{want}")
        rule = shapes.get(name, {}).get("rule")
        if rule and record.get("rule") != rule:
            fail(f"§2.2's {name} rule {record.get('rule')!r} is not the catalog's {rule!r}")
    window = found.get("window") or {}
    if window and window.get("history_rows") != \
            window.get("length", 0) - 1 + window.get("lag_depth", 0):
        fail("§2.2's window example breaks history_rows = (length - 1) + lag_depth")
    graph = shapes["graph"]
    for protocol in graph["protocols_served"]:
        if f"protocol: {protocol}" not in sub:
            fail(f"§2.2 does not say the graph protocol {protocol!r} is served")
    for protocol, message in graph["protocols_refused"].items():
        if message.split(" (exit")[0] not in sub:
            fail(f"§2.2 does not state the refusal of the {protocol} protocol verbatim")
    three = subsection(md, "### 2.3 Graph reference files")
    tree = tree_tokens(md)
    for path in graph["files"]:
        if f"`{path}`" not in three or path not in tree:
            fail(f"the graph reference file {path} is not named in §2.3 and §1's tree")
    ref = catalog["graph_reference"]
    if ref["schema"] != graph["reference_schema"] or ref["schema"] not in three:
        fail(f"the graph reference schema {ref['schema']!r} is not the one §2.3 names")
    for key in ref["meta_keys"]:
        if f'"{key}"' not in three:
            fail(f"graph/graph.json's key {key!r} is not in §2.3")
    for column in ref["edges_columns"]:
        if f"`{column}`" not in three:
            fail(f"reference-edges column {column!r} is not in §2.3")
    if "No raw position ships" not in three:
        fail("§2.3 no longer says that no raw position ships in the graph reference")
    ok(f"{len(shapes)} sample shape(s) and the graph reference agree with §2.2–§2.3")


def check_serving_templates(catalog, md):
    """§7.1: the template patterns are well formed, named, and matched as stated."""
    templates = catalog["serving_templates"]
    rule = catalog["serving_template_rule"]
    if set(templates) != set(rule["destinations"]):
        fail(f"serving_templates destinations {sorted(templates)} differ from the rule's "
             f"{sorted(rule['destinations'])}")
    sub = subsection(md, "### 7.1 Serving templates")
    rows = {r[0].strip("`"): r for r in table_rows(sub) if r[0].strip("`") in templates}
    seen = set()
    for dest, patterns in templates.items():
        if not patterns:
            fail(f"serving_templates.{dest} names no pattern")
        for pattern in patterns:
            segments = pattern.split("/")
            problems = []
            if pattern in seen:
                problems.append("is listed twice")
            if pattern.startswith("/") or "\\" in pattern or ".." in segments:
                problems.append("is not a relative path")
            if "**" in pattern or any(not s for s in segments):
                problems.append("has an empty or recursive segment")
            if segments[0] != dest or len(segments) < 2:
                problems.append(f"does not start with its destination {dest}/")
            if any(re.search(r"[*?\[]", s) for s in segments[:-1]):
                problems.append("globs a directory segment")
            for why in problems:
                fail(f"serving_templates pattern {pattern!r} {why}")
            seen.add(pattern)
            if dest not in rows or f"`{pattern}`" not in rows[dest][1]:
                fail(f"serving_templates pattern {pattern!r} is not named in §7.1's "
                     f"{dest} row")
    sys.path.insert(0, str(SCRIPTS))
    from vega_pack_core import template_matches  # noqa: E402
    for pattern, rel, want in (("package/*.py", "package/inspection/codec.py", False),
                               ("package/inspection/*.py", "package/inspection/codec.py", True),
                               ("bundle/*", "bundle/.dockerignore", True),
                               ("serve/*.py", "serve/static/index.html", False),
                               ("serve/static/index.html", "serve/static/index.html", True)):
        if template_matches(pattern, rel) != want:
            fail(f"template_matches({pattern!r}, {rel!r}) is not {want}, as §7.1 states")
        if f"`{pattern}`" not in sub:
            fail(f"§7.1 does not show the pattern {pattern!r} it states a match for")
    if "fnmatch.fnmatchcase" not in sub:
        fail("§7.1 no longer states the matching rule (fnmatch.fnmatchcase per segment)")
    ok(f"{len(seen)} serving template pattern(s) over {len(templates)} destination(s), well "
       "formed and named in §7.1")


def check_reemit(catalog, md):
    """§7.8: what a re-emit may change and what it never changes."""
    reemit = catalog["reemit"]
    flat = [p for patterns in catalog["serving_templates"].values() for p in patterns]
    if sorted(reemit["may_change"]["templates"]) != sorted(flat):
        fail(f"reemit.may_change names {sorted(reemit['may_change']['templates'])}; the "
             f"serving_templates patterns are {sorted(flat)} (R-D2: the same patterns)")
    sub = subsection(md, "### 7.8 `--re-emit-serving`")
    if not sub:
        fail("bundle-contract §7.8 (--re-emit-serving) is missing")
        return
    for entry in reemit["never"]:
        if f"`{entry.split(' files outside')[0]}`" not in sub and f"`{entry}`" not in sub:
            fail(f"reemit.never entry {entry!r} is not stated in §7.8")
    for pattern in reemit["may_change"]["templates"]:
        if f"`{pattern}`" not in sub:
            fail(f"reemit.may_change pattern {pattern!r} is not named in §7.8")
    for key in reemit["may_change"]["manifest"]:
        if f"`{key.split(' ')[0]}`" not in sub:
            fail(f"reemit.may_change manifest key {key!r} is not named in §7.8")
    if reemit["refusal"].split(" (exit")[0] not in sub:
        fail("§7.8 does not state the re-emit refusal as the catalog words it")
    ok(f"re-emit: {len(flat)} template pattern(s) may change, {len(reemit['never'])} "
       "never-changing entr(ies) stated in §7.8")


def check_tree_tokens(catalog, md):
    """§1: the showcase is a directory token only; the removed files are gone."""
    tokens = tree_tokens(md)
    if "showcase/" not in tokens:
        fail("§1's tree does not name the showcase/ slot")
    under = [t for t in tokens if t.startswith("showcase/") and t != "showcase/"]
    if under:
        fail(f"§1's tree names {under} under showcase/; the slot is /vega-show's and only its "
             "directory token belongs in the bundle's tree (a file there would read as one "
             "the bundle must hold)")
    for gone in ("story.json", "graph/edges_for.json", "examples/attribution.json"):
        if gone in tokens:
            fail(f"§1's tree still names {gone}, which v2.3 removed from the bundle")
    if any(t.startswith("report/") for t in tokens):
        fail("§1's tree still names a report/ file; the bundle carries no report")
    for key, row in catalog["examples"]["files"].items():
        if row["path"] not in tokens:
            fail(f"§1's tree does not name {row['path']} ({key})")
    ok("§1's tree: showcase/ as a directory token only, no story, no report, the v2.3 "
       "example and graph files named")


def check_manifest_example(catalog, md):
    """§2's example manifest, parsed and held to the catalog's records."""
    two = section(md, "## 2. `MANIFEST.json`")
    block = re.search(r"```json\n([\s\S]*?)```", two)
    try:
        example = json.loads(block.group(1)) if block else None
    except ValueError as exc:
        fail(f"§2's example manifest is not JSON: {exc}")
        return
    if not example:
        fail("§2 has no example manifest")
        return
    table = catalog["output_readings"]
    output = example.get("output") or {}
    if list(output) != table["record_keys"]:
        fail(f"§2's output record has keys {list(output)}; vega/output@2 is "
             f"{table['record_keys']}")
    if output.get("schema") != table["schema"]:
        fail(f"§2's output schema is {output.get('schema')!r}, not {table['schema']!r}")
    reading = table["readings"].get(output.get("reading"))
    if not reading:
        fail(f"§2's output reading {output.get('reading')!r} is not a reading")
        return
    if output.get("link") != reading["link"] or output.get("decision") != reading["decision"]:
        fail("§2's output link or decision is not its reading's")
    patterns = dict(reading["kinds"])
    patterns.update({k: {"kind": v["kind"]} for k, v in table["shape_fields"].items()})
    fields = output.get("fields") or []
    for name in fields:
        match = [p for p in patterns if _pattern_re(p).match(name)]
        if not match:
            fail(f"§2's output field {name!r} matches no pattern of the {output['reading']} "
                 "reading")
        elif (output.get("field_kinds") or {}).get(name, {}).get("kind") != \
                patterns[match[0]]["kind"]:
            fail(f"§2's field {name!r} has kind "
                 f"{(output.get('field_kinds') or {}).get(name)}; its pattern says "
                 f"{patterns[match[0]]['kind']!r}")
    if set(output.get("field_kinds") or {}) != set(fields):
        fail("§2's field_kinds does not name exactly its fields")
    for pair in (output.get("compared") or []) + (output.get("decided") or []):
        if pair[0] not in fields:
            fail(f"§2's output compares or decides {pair[0]!r}, which is not in its fields")
    if (output.get("truth") or {}).get("type") != reading["truth"]:
        fail("§2's truth type is not its reading's")
    if (example.get("sample") or {}).get("shape") not in catalog["sample_shapes"]:
        fail(f"§2's sample shape {(example.get('sample') or {}).get('shape')!r} is unknown")
    serve = example.get("serve") or {}
    env = (serve.get("identity") or {}).get("env")
    slot = json.loads(json.dumps(catalog["serve_record"]["showcase_slot"])
                      .replace("<ENV>", str(env)))
    if serve.get("showcase_slot") != slot:
        fail(f"§2's serve.showcase_slot {serve.get('showcase_slot')} is not the catalog's "
             f"{slot}")
    for removed in catalog["serve_record"]["removed"]:
        if removed in serve:
            fail(f"§2's serve record still carries {removed!r}")
    for added in catalog["serve_record"]["added"]:
        if added not in serve:
            fail(f"§2's serve record does not show {added!r}")
    trip = example.get("roundtrip") or {}
    if not {"columns_compared", "decisions_compared"} <= set(trip):
        fail("§2's roundtrip record does not show columns_compared and decisions_compared")
    ok("§2's example manifest holds a vega/output@2 record, a sample, and the v2.3 serve "
       "record")


def check_serve_record(catalog, md):
    """§2.4, §7.2, §7.6: the serve record, the route modules, the code view, the slot."""
    rec = catalog["serve_record"]
    four = subsection(md, "### 2.4 The serve record")
    for key in rec["added"] + rec["removed"]:
        if f"`{key}`" not in four:
            fail(f"serve.{key} is not named in §2.4")
    slot_json = [t for t in ticks(four) if t.startswith('{"dir"')]
    try:
        slot = json.loads(slot_json[0]) if slot_json else None
    except ValueError:
        slot = None
    if slot != rec["showcase_slot"]:
        fail(f"§2.4 does not state serve.showcase_slot as the catalog holds it: "
             f"{rec['showcase_slot']}")
    for key in rec["inspection"]["keys"]:
        if f'"{key}"' not in four:
            fail(f"serve.inspection key {key!r} is not in §2.4")
    for value in (rec["inspection"]["schema"], rec["inspection"]["network_schema"],
                  rec["inspection"]["explanation_schema"]):
        if value not in four:
            fail(f"§2.4 does not name {value}")
    if f"`{rec['forced_by'][0]}`" not in four and f'"{rec["forced_by"][0]}"' not in four:
        fail(f"§2.4 does not say forced_by is {rec['forced_by'][0]!r}")
    two = subsection(md, "### 7.2 Route modules and endpoints")
    for name in rec["route_modules"] + rec["route_module_literals"]:
        if name not in two:
            fail(f"{name} is not named in §7.2")
    for capability in rec["capabilities"]:
        if f'"{capability}"' not in two and f"`{capability}`" not in two:
            fail(f"the capability {capability!r} is not named in §7.2")
    code = catalog["code_view"]
    if code["schema"] not in two:
        fail(f"§7.2 does not name {code['schema']}")
    for name in code["roles"] + code["kinds"]:
        if f"`{name}`" not in two:
            fail(f"code-view role or kind {name!r} is not named in §7.2")
    six = subsection(md, "### 7.6 The showcase slot")
    opens = "(" + ", ".join(f'"{p}"' for p in rec["showcase_open_while_unready"]) + ")"
    if opens not in six:
        fail(f"§7.6 does not state the slot's OPEN_WHILE_UNREADY {opens}")
    for route in rec["showcase_slot"]["routes"]:
        if f"`{route}`" not in six:
            fail(f"the slot route {route!r} is not named in §7.6")
    if "The slot rule" not in six or "never in MANIFEST.files" not in md:
        fail("§7.6 no longer states the slot rule (the showcase is never in MANIFEST.files)")
    ok(f"serve record, {len(rec['route_modules'])} route module(s), the code view and the "
       "showcase slot agree with §2.4, §7.2 and §7.6")


def _schema_refs(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                yield value
            else:
                yield from _schema_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _schema_refs(value)


def _preset_facts(schema):
    """kind -> the facts its `if/then` requires, read back out of the schema."""
    out = {}
    for branch in schema["$defs"]["preset"].get("allOf") or []:
        kind = ((branch.get("if") or {}).get("properties") or {}).get("kind", {}).get("const")
        facts = (((branch.get("then") or {}).get("properties") or {}).get("facts") or {})
        if kind:
            out[kind] = facts.get("required") or []
    return out


def _schema_vocabulary(ex, schema, name):
    """The schema's header, its $refs, and its enums and preset facts against the catalog."""
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail(f"{name} does not declare JSON Schema draft 2020-12")
    if schema.get("title") != ex["schema"]:
        fail(f"{name} is titled {schema.get('title')!r}, not {ex['schema']!r}")
    defs = schema.get("$defs") or {}
    for ref in _schema_refs(schema):
        if not ref.startswith("#/$defs/") or ref.split("/")[-1] not in defs:
            fail(f"{name}: $ref {ref!r} does not resolve")
    for key, row in ex["files"].items():
        ref = row.get("json_schema_def")
        if ref and ref.split("/")[-1] not in defs:
            fail(f"examples.files.{key} names {ref}, which {name} does not define")

    def enum(def_name, prop):
        return ((defs.get(def_name) or {}).get("properties") or {}).get(prop, {}).get("enum")

    enums = {"preset kind": (enum("preset", "kind"), list(ex["preset_kinds"])),
             "preset group": (enum("preset", "group"), ex["preset_groups"]),
             "stratum kind": (enum("stratum", "kind"), ex["stratum_kinds"]),
             "expected kind": (enum("expected", "kind"), ex["expected_kinds"]),
             "test source": (enum("test", "source"), ex["kinds"])}
    for what, (have, want) in enums.items():
        if sorted(have or []) != sorted(want):
            fail(f"{name}'s {what} enum {have} is not the catalog's {want}")
    facts = _preset_facts(schema)
    for kind, row in ex["preset_kinds"].items():
        if sorted(facts.get(kind) or []) != sorted(row["facts"]):
            fail(f"preset kind {kind!r}: the schema requires facts {facts.get(kind)}, the "
                 f"catalog says {row['facts']}")
        if not kind.startswith(row["group"]):
            fail(f"preset kind {kind!r} is in group {row['group']!r}; the schema ties the "
                 "group to the kind's prefix")


def _examples_prose(ex, four):
    """§7.4 names the schema file, its $defs, and every preset, stratum and group."""
    for needle in [ex["json_schema"], ex["schema"]] + \
            [row["json_schema_def"] for row in ex["files"].values() if row["json_schema_def"]]:
        if needle not in four:
            fail(f"§7.4 does not name {needle}")
    for name in list(ex["preset_kinds"]) + ex["stratum_kinds"] + ex["preset_groups"]:
        if f"`{name}`" not in four:
            fail(f"§7.4 does not name {name!r}")
    if "No English ships in the payload" not in four:
        fail("§7.4 no longer says that no English ships in the presets")


def _preset(pid, kind, stratum, expected, facts):
    """A preset with no input yet: the caller sets `row` (row shape) or `rows` and `history`."""
    group = "typical" if kind.startswith("typical_") else "edge"
    return {"id": pid, "kind": kind, "group": group, "stratum": stratum, "expected": expected,
            "facts": facts, "row": None, "rows": None, "history": None}


def _stratum(sid, kind, name, n, bounds=None):
    return {"id": sid, "kind": kind, "name": name, "n": n, "range": bounds}


def _demo(strata, items):
    return {"source": "train", "schema": "vega/demo@2", "strata": strata,
            "items": [{"stratum": s, "rows": r, "history": h, "truth": t}
                      for s, r, h, t in items]}


def _good_examples():
    """Well-formed vega/examples@2 bodies, one per demo family §7.4 describes.

    A row class reading; a multilabel reading, whose `none` stratum's items carry the truth []
    (R-A8 types multilabel truth as a list of label names, and no label is positive); a
    window value reading, whose items carry a history and one number per horizon; an unlabelled
    anomaly task (no truth, an empty `flagged` stratum, an alarm expectation); and a bundle
    packed with --demo-sample 0 (`demo` null).
    """
    row = {"x": 1.5, "zip": "60614"}
    typical = dict(_preset("typical-class-0", "typical_correct", "class-0",
                           {"kind": "class", "value": "a"}, {"right": 19, "of": 20}), row=row)
    edge = dict(_preset("edge-range", "edge_range", "class-0", None,
                        {"column": "x", "factor": 10}), row={"x": 15.0, "zip": "60614"})
    test = {"source": "test", "rows": [row], "history": None}
    history = [{"t": 1, "y": 1.0}, {"t": 2, "y": 1.5}]
    window = [{"t": 3, "y": 1.1}]
    return {
        "class": {"schema": "vega/examples@2", "test": test,
                  "demo": _demo([_stratum("class-0", "class", "a", 20)],
                                [("class-0", [row], None, "a")]),
                  "presets": [typical, edge]},
        "multilabel": {"schema": "vega/examples@2", "test": test,
                       "demo": _demo([_stratum("label-a", "label", "a", 20),
                                      _stratum("none", "none", None, 20)],
                                     [("label-a", [row], None, ["a"]),
                                      ("none", [row], None, [])]),
                       "presets": [
                           dict(_preset("typical-label-a", "typical_correct", "label-a",
                                        {"kind": "labels", "value": ["a"]},
                                        {"right": 18, "of": 20}), row=row),
                           dict(_preset("typical-none", "typical_none_correct", "none",
                                        {"kind": "labels", "value": []},
                                        {"right": 0, "of": 20}), row=row),
                           dict(edge, stratum="none")]},
        "window": {"schema": "vega/examples@2",
                   "test": {"source": "synthetic", "rows": window, "history": history},
                   "demo": _demo([_stratum("bin-0", "bin", None, 5, [0.0, 1.2])],
                                 [("bin-0", window, history, [1.1, 1.3])]),
                   "presets": [
                       dict(_preset("typical-bin-0", "typical_sequence", "bin-0",
                                    {"kind": "value", "value": 1.1, "range": [0.0, 1.2]},
                                    {"of": 5}), rows=window, history=history),
                       dict(_preset("edge-infinity", "edge_infinity", "bin-0", None,
                                    {"columns": ["y"]}), rows=window, history=history)]},
        "unlabelled anomaly": {"schema": "vega/examples@2", "test": test,
                               "demo": _demo([_stratum("typical", "typical", None, 20),
                                              _stratum("flagged", "flagged", None, 0)],
                                             [("typical", [row], None, None)]),
                               "presets": [
                                   dict(_preset("typical-score", "typical_score", "typical",
                                                {"kind": "alarm", "value": False}, {"of": 20}),
                                        row=row),
                                   dict(edge, stratum="typical")]},
        "no demo": {"schema": "vega/examples@2", "test": test, "demo": None,
                    "presets": [typical, edge]}}


def _bad_examples(good):
    """Malformed bodies, each one defect away from a good one."""
    typical, edge = good["class"]["presets"]
    demo = good["multilabel"]["demo"]
    bin_stratum = good["window"]["demo"]["strata"][0]

    def with_demo(**changes):
        return dict(good["multilabel"], demo=dict(demo, **changes))

    def with_presets(*presets):
        return dict(good["class"], presets=list(presets))

    return {
        "a preset carrying an English sentence":
            with_presets(dict(typical, text="nineteen of twenty")),
        "an edge preset with an expectation":
            with_presets(dict(edge, expected={"kind": "class", "value": "a"})),
        "a typical preset without its count": with_presets(dict(typical, facts={"right": 19})),
        "a preset whose group is not its kind's": with_presets(dict(edge, group="typical")),
        "a demo drawn from the test part": with_demo(source="test"),
        "a truth mixing label names and numbers":
            with_demo(items=[dict(demo["items"][0], truth=["a", 1])]),
        "a bin stratum without its range":
            dict(good["window"], demo=dict(good["window"]["demo"],
                                           strata=[dict(bin_stratum, range=None)]))}


def _def_validator(jsonschema, schema, name):
    """A validator for one `$defs` entry, as the packer runs it on an examples/*.json file."""
    return jsonschema.Draft202012Validator(
        {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"})


def _schema_documents(jsonschema, schema, name):
    """The schema accepts every good body (whole, and its demo.json and presets.json against
    their $defs) and refuses every bad one."""
    whole = jsonschema.Draft202012Validator(schema)
    demo = _def_validator(jsonschema, schema, "demo")
    presets = _def_validator(jsonschema, schema, "presets_file")
    good = _good_examples()
    for family, body in good.items():
        files = [(whole, body, "body")]
        if body["demo"] is not None:
            files.append((demo, body["demo"], "demo.json"))
        files.append((presets, {"schema": "vega/presets@2", "presets": body["presets"]},
                      "presets.json"))
        for validator, doc, what in files:
            errors = sorted(f"{list(e.absolute_path)}: {e.message}"
                            for e in validator.iter_errors(doc))
            if errors:
                fail(f"{name} refuses the well-formed {family} example's {what}: {errors[:2]}")
    bad = _bad_examples(good)
    for why, body in bad.items():
        if whole.is_valid(body):
            fail(f"{name} accepts {why}")
    ok(f"{name}: valid draft 2020-12, enums equal to the catalog's, accepts {len(good)} "
       f"well-formed payload(s) (class, multilabel with a `none` stratum, window, unlabelled "
       f"anomaly, no demo) and refuses {len(bad)} malformed one(s)")


def check_examples_schema(catalog, md):
    """§7.4 and references/schemas/examples-2.schema.json against the catalog."""
    ex = catalog["examples"]
    path = SKILL / ex["json_schema"]
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"{ex['json_schema']} cannot be read as JSON: {exc}")
        return
    _schema_vocabulary(ex, schema, path.name)
    _examples_prose(ex, subsection(md, "### 7.4 Examples"))
    try:
        import jsonschema  # noqa: PLC0415
    except ImportError:
        ok(f"{path.name}: structure checked; jsonschema is not installed, so the draft "
           "2020-12 metaschema check and the sample documents did not run")
        return
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        fail(f"{path.name} is not a valid draft 2020-12 schema: {exc.message}")
        return
    _schema_documents(jsonschema, schema, path.name)


# Keys build plan K1 adds to vega-forge's catalog together with `inspection` (R-C7). A forge
# catalog that has one of them and no `inspection` section is a regression, not the pre-K1 tree.
K1_MARKERS = ("metric_direction",)


def _forge_inspection(forge):
    """vega-forge's `inspection` section; None when there is none (a NOTE on the pre-K1 tree,
    a failure once the catalog carries K1's other keys)."""
    if "inspection" in forge:
        return forge["inspection"] or {}
    k1 = [key for key in K1_MARKERS if key in forge]
    if k1:
        fail(f"vega-forge's catalog carries {k1} (build plan K1) but no `inspection` section; "
             "R-C1 puts the one link vocabulary and the reading → link table there, and "
             "without it nothing holds vega-pack's links to them")
    else:
        print("  NOTE  vega-forge's catalog has no `inspection` section yet (build plan K1 "
              "adds inspection.links); the link vocabulary was checked against the readings "
              "only")
    return None


def _forge_reading_links(links):
    """{reading: [link, …]} from vega-forge's inspection.links when it is keyed by link and its
    rows carry `readings` (K1's form); None for a plain vocabulary (a list, or rows without
    `readings`), where only the link names can be compared. A row without `readings` beside
    rows that have them maps no reading, so a partial table shows as a difference."""
    if not isinstance(links, dict) or not any(
            isinstance(row, dict) and "readings" in row for row in links.values()):
        return None
    table = {}
    for link, row in links.items():
        for reading in ((row if isinstance(row, dict) else {}).get("readings") or {}):
            table.setdefault(reading, []).append(link)
    return table


def _compare_reading_links(theirs, ours):
    """R-C1's reading → link table: vega-forge's inspection.links against vega-pack's readings."""
    table = _forge_reading_links(theirs)
    if table is None:
        print("  NOTE  vega-forge's inspection.links is a plain vocabulary; the link names "
              "were compared, the reading → link table was not")
        return
    twice = sorted(reading for reading, links in table.items() if len(links) > 1)
    if twice:
        fail(f"vega-forge's inspection.links maps {twice} under more than one link")
    flat = {reading: links[0] for reading, links in table.items()}
    differ = sorted(r for r in set(flat) | set(ours) if flat.get(r) != ours.get(r))
    if differ:
        fail("vega-forge's inspection.links and vega-pack's readings map readings to different "
             "links (R-C1: one table): " + ", ".join(
                 f"{r}: forge {flat.get(r)}, pack {ours.get(r)}" for r in differ))
    else:
        ok(f"vega-forge's inspection.links maps the {len(ours)} reading(s) to the links "
           "vega-pack's readings name")


def check_links_against_forge(catalog):
    """R-C1: one link vocabulary and one reading → link table, held by vega-forge's catalog;
    the pack's readings use them."""
    table = catalog["output_readings"]
    links = table["links"]
    ours = {name: row["link"] for name, row in table["readings"].items()}
    if set(links) != set(ours.values()):
        fail(f"the link vocabulary {sorted(links)} and the links the readings use "
             f"{sorted(set(ours.values()))} differ")
    forge = _sibling_catalog("vega-forge")
    if forge is None:
        fail("vega-forge/scripts/catalog.json is not beside this skill; the link vocabulary "
             "cannot be compared with the one it owns")
        return
    inspection = _forge_inspection(forge)
    if inspection is None:
        return
    theirs = inspection.get("links")
    if theirs is None:
        fail("vega-forge's catalog has an inspection section without `links`; R-C1 puts the "
             "one link vocabulary there")
        return
    names = set(theirs) if isinstance(theirs, (list, dict)) else set()
    if names != set(links):
        fail(f"vega-forge's inspection.links {sorted(names)} and vega-pack's reading links "
             f"{sorted(links)} differ (R-C1: one vocabulary)")
    else:
        ok(f"the {len(links)} reading link(s) are vega-forge's inspection.links")
    readings = inspection.get("readings")
    if readings is not None and set(readings) != set(ours):
        fail(f"vega-forge's inspection.readings {sorted(readings)} and vega-pack's readings "
             f"{sorted(ours)} differ")
    _compare_reading_links(theirs, ours)


def check_task_kinds(catalog):
    """Every task kind the chain can produce has a serving reading (the v2.2 check that every
    evaluated kind had an `output_fields` row, made executable: v2.3 resolves a kind to a
    reading through rules, so the rules are run rather than a key looked up)."""
    kinds = set()
    for sibling in ("vega-eval", "vega-forge"):
        other = _sibling_catalog(sibling)
        if other is not None:
            kinds |= set(other["tasks"])
    readings = catalog["output_readings"]["readings"]
    unresolved = sorted(k for k in kinds
                        if resolve_reading(catalog, {"task": {"kind": k}}) not in readings)
    if unresolved:
        fail(f"the chain produces the task kind(s) {unresolved}, and no output_readings rule "
             "resolves them to a reading, so a bundle could not say what it returns")
    else:
        ok(f"every one of {len(kinds)} task kind(s) the chain produces resolves to a reading")


def check_v23(catalog, md):
    """What serving reads from v2.3 on (the module docstring's second paragraph)."""
    check_task_kinds(catalog)
    for check in (check_readings, check_resolution, check_field_kinds, check_monitor_kinds,
                  check_sample_shapes, check_serving_templates, check_reemit, check_tree_tokens,
                  check_manifest_example, check_serve_record, check_examples_schema):
        check(catalog, md)
    check_links_against_forge(catalog)


def main():
    print(f"catalog sync — {CATALOG_JSON.name} against {CATALOG_MD.name} §1–§7")
    catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    md = CATALOG_MD.read_text(encoding="utf-8")

    # ---- §1, the required files
    listed = tree_files(section(md, "## 1. The bundle directory"))
    required = set(catalog["required_files"])
    # the tree also names what only a flag writes, and what the packer generates per package
    optional = {"model-card.html", "predict.ipynb", "app.py", "consumer.py", "Dockerfile",
                "model.onnx", "model.pt", "parity.json", "weights.<ext>",
                # written only with --serve, or with a monitor plan (§1, §7)
                "pyproject.toml", "requirements-serve.txt", "monitor-reference.json"}
    expected = {n for n in listed if n not in optional and "/" not in n or
                n in ("examples/rows.csv", "examples/expected.parquet",
                      "tests/test_roundtrip.py")}
    expected -= optional
    missing = sorted(n for n in expected if n not in required
                     and not n.startswith(("serve/", "exports/")))
    if missing:
        fail(f"§1's tree names {missing}, which catalog.json's required_files omits")
    else:
        ok(f"{len(required)} required file(s), all named by §1")
    unknown = sorted(n for n in required if n not in listed)
    if unknown:
        fail(f"catalog.json requires {unknown}, which §1's tree does not name")

    # ---- §1, the package slice
    slice_text = section(md, "## 1. The bundle directory")
    for module in catalog["package_slice"]["keep"]:
        if module.split("/")[-1] not in slice_text:
            fail(f"catalog.json keeps {module} in the serving slice and §1 does not name it")
    for dropped in catalog["package_slice"]["drop_packages"]:
        if f"{dropped}/" not in slice_text:
            fail(f"catalog.json drops {dropped}/ and §1 does not say so")
    if "are not copied and their ports are `out of scope`" not in slice_text:
        fail("§1 no longer says the dropped packages' ports go `out of scope`; "
             "catalog.json still records a reason for each")
    for role, reason in catalog["package_slice"]["out_of_scope"].items():
        if not reason.strip():
            fail(f"the out-of-scope entry for {role} has no reason; a port marked out of "
                 "scope without one is a port nobody decided about")
    ok(f"{len(catalog['package_slice']['keep'])} module(s) kept, "
       f"{len(catalog['package_slice']['drop_packages'])} package(s) dropped, each with a "
       "reason")

    # ---- §1, the records the bundle carries
    for layer in catalog["records"]["layers"]:
        if f"records/{layer}.json" not in slice_text:
            fail(f"catalog.json copies records/{layer}.json into the bundle and §1's tree "
                 "does not name it")
    if "nothing in the bundle imports them" not in slice_text:
        fail("§1 no longer says the records are data rather than code; catalog.json says so "
             "and pack.py relies on it (a bundle whose records were deleted still predicts)")
    ok(f"{len(catalog['records']['layers'])} layer record(s), all named by §1")

    # ---- §1 and §4, the two shapes
    include = catalog["include"]
    if f"`{include['default']}` (the default)" not in slice_text:
        fail(f"catalog.json defaults --include to {include['default']!r} and §1 does not say "
             "which shape is the default")
    four = section(md, "## 4. `predict.py`")
    for mode, row in include["modes"].items():
        if f"`{mode}`" not in slice_text:
            fail(f"catalog.json defines the --include shape {mode!r} and §1 does not name it")
            continue
        for entry in row["entrypoints"]:
            where = slice_text + four
            if entry not in where:
                fail(f"shape {mode!r} writes {entry} and neither §1 nor §4 names that file")
    if include["modes"]["serving"]["out_of_scope"] == \
            include["modes"]["full"]["out_of_scope"]:
        fail("catalog.json has both shapes treating the dropped packages' ports the same "
             "way; §1 says serving marks them out of scope and full implements them")
    if "seal" not in include["modes"]["full"]:
        fail("catalog.json's full shape records no seal note; §1 says the seal travels with "
             "the door, and a shape that ships ingestion without saying so is the one thing "
             "a reader would want said")
    ok(f"{len(include['modes'])} --include shape(s) agree, default {include['default']}")

    # ---- §5, the round-trip tolerance
    five = section(md, "## 5. The round trip")
    rule = catalog["roundtrip"]["rule"]
    prose_rule = re.search(r"`\|Δ\| ≤ ([^`]+)`", five)
    if not prose_rule:
        fail("§5 no longer states the round-trip formula in backticks; catalog.json holds "
             f"{rule!r} and nothing checks it")
    else:
        # the prose writes it as mathematics, the JSON as an expression; compare the parts
        if "1e-6" not in prose_rule.group(1) or "max|score|" not in prose_rule.group(1):
            fail(f"§5's formula {prose_rule.group(1)!r} and catalog.json's {rule!r} are not "
                 "the same rule")
        else:
            ok(f"the round-trip formula agrees: {rule}")
    if str(catalog["roundtrip"]["precision"]) not in five:
        fail(f"§5 does not mention {catalog['roundtrip']['precision']}, which catalog.json "
             "names as the precision the default tolerance assumes")

    # ---- §7, the serving modes and the exports
    seven = section(md, "## 7. Serving and exports")
    for mode, row in catalog["serving_modes"].items():
        if f"`{mode}`" not in seven:
            fail(f"catalog.json defines the serving mode {mode!r} and §7 does not name it")
            continue
        if row["serve"] and row["entry"] and row["entry"] not in seven:
            fail(f"mode {mode!r} writes {row['entry']} and §7 does not name that file")
    if catalog["serving_modes"]["batch"]["serve"]:
        fail("catalog.json has batch mode writing a serving scaffold; §7 says batch writes "
             "nothing beyond predict.py")
    ok(f"{len(catalog['serving_modes'])} serving mode(s) agree")
    for fmt in catalog["export_formats"]:
        if fmt not in seven:
            fail(f"catalog.json lists the export format {fmt!r} and §7 does not name it")
    # the parity tolerance is stated in §5, beside the round trip it is the export's analogue
    # of, and again in §2's example — look in all three rather than deciding which owns it
    atol = catalog["export_parity_atol"]
    where = five + seven + section(md, "## 2. `MANIFEST.json`")
    if f"{atol:g}" not in where and str(atol) not in where and "1e-5" not in where:
        fail(f"neither §2, §5 nor §7 states the parity tolerance {atol}, which catalog.json "
             "records")
    ok(f"{len(catalog['export_formats'])} export format(s) and their parity tolerance agree")

    # ---- §2, the manifest's required keys
    two = section(md, "## 2. `MANIFEST.json`")
    for key in catalog["manifest_required_keys"]:
        if f'"{key}"' not in two:
            fail(f"catalog.json requires the manifest key {key!r} and §2's example has no "
                 "such field")
    ok(f"{len(catalog['manifest_required_keys'])} manifest key(s), all in §2's example")

    check_v23(catalog, md)

    print()
    if failures:
        print(f"{len(failures)} disagreement(s) between the prose and the machine catalog")
        return 1
    print(f"OK: catalog.json and {CATALOG_MD.name} §1–§7 agree "
          f"({len(required)} required file(s), {len(catalog['serving_modes'])} serving mode(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
