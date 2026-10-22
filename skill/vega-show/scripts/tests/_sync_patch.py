"""
_sync_patch.py — gate CS3: the document-project patch (`patches/document-project/SKILL.md`, D5
§13.8) is what writes `vega/label-roles@1` beside a real report (lanekit's `report_stub` writes it
for the test lanes), so it must name every report role vega-show binds, and no other, each with
the kind and stage `catalog.report_roles` gives it; its `vega/label-roles@1` example must
validate; and it must state that the post-report `show` brief is never a chapter. Seeded copies of the patch — a role dropped (design negative D5-39), an
unknown role added, a kind changed, the section renamed, the exclusion sentence removed — must
each be refused, so the gate is shown to bite. A helper of `test_catalog_sync.py`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from _sync_common import SCHEMA_DIR, SKILLS, fail, ok
from _sync_contract import tables, tick1

# CS3 (D5 §13.8): what the document-project patch states, compared up to line wrapping
DP_PATCH = SKILLS / "patches" / "document-project" / "SKILL.md"
ROLE_MAP_SECTION = "The label-role map"
POST_REPORT_TEXTS = (
    "The `show` stage's brief is **post-report**: it is never a chapter, and this skill never "
    "reads the showcase.",
    "(a `show` brief, when present, is skipped: it was written after the report and cites it)",
    "`post_report: true`")


def flat(text):
    return " ".join(text.split())


def role_map_section(text):
    """The body of the patch's one "The label-role map" section, up to the next heading of its
    level or above (fenced code is never a heading); None when there is not exactly one."""
    lines, heads, fence = text.splitlines(), [], False
    for i, line in enumerate(lines):
        fence = fence != line.lstrip().startswith("```")
        if not fence and re.fullmatch(r"#{2,6}\s+" + re.escape(ROLE_MAP_SECTION) + r"\s*", line):
            heads.append(i)
    if len(heads) != 1:
        return None
    level, body, fence = len(lines[heads[0]].split()[0]), [], False
    for line in lines[heads[0] + 1:]:
        fence = fence != line.lstrip().startswith("```")
        head = re.match(r"(#{1,6})\s", line)
        if not fence and head and len(head.group(1)) <= level:
            break
        body.append(line)
    return "\n".join(body)


def role_map_validator(cat):
    """A draft 2020-12 validator of vega/label-roles@1, or None when CS2 already fails for want
    of jsonschema or of a readable schema."""
    try:
        from jsonschema import Draft202012Validator
        path = SCHEMA_DIR / Path(cat["schemas"]["vega/label-roles@1"]["file"]).name
        return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))
    except (ImportError, KeyError, OSError, ValueError):
        return None


def _example_problems(cat, section, validator):
    blocks = re.findall(r"```json\n(.*?)```", section, re.S)
    if not blocks:
        return ["the label-role map section carries no ```json example of vega/label-roles@1"]
    try:
        doc = json.loads(blocks[0])
    except ValueError as exc:
        return [f"the label-role map example is not JSON: {exc}"]
    errors = list(validator.iter_errors(doc))[:3] if validator else []
    out = [f"the label-role map example does not validate vega/label-roles@1 at "
           f"{list(e.absolute_path)}: {e.message[:120]}" for e in errors]
    roles = doc.get("roles") if isinstance(doc, dict) and isinstance(doc.get("roles"), dict) else {}
    for role, row in roles.items():
        spec = cat["report_roles"].get(role)
        pair = (row.get("kind"), row.get("stage")) if isinstance(row, dict) else None
        if spec and pair and pair != (spec["kind"], spec["stage"]):
            out.append(f"the label-role map example binds {role} as {pair[0]}/{pair[1]}, the "
                       f"catalog as {spec['kind']}/{spec['stage']}")
    return out


def _table_problems(cat, section):
    """The section's role table against catalog.report_roles: exactly its roles, each with its
    kind and stage."""
    rows = [r for head, body in tables(section) if head[:3] == ("role", "kind", "stage")
            for r in body]
    got = {tick1(r[0]): (tick1(r[1]), tick1(r[2])) for r in rows if len(r) >= 3}
    want = {role: (row["kind"], row["stage"]) for role, row in cat["report_roles"].items()}
    out = [] if len(got) == len(rows) else [
        "the patch's label-role map lists a role twice, or a row without kind and stage"]
    out += [f"report role {role} is missing from the document-project patch's label-role map "
            "(D5-39)" for role in want if role not in got]
    out += [f"the patch's label-role map names {role}, which catalog.report_roles does not hold"
            for role in got if role not in want]
    out += [f"the patch's label-role map gives {role} as {'/'.join(got[role])}, the catalog "
            f"as {'/'.join(want[role])}" for role in got if got[role] != want.get(role, got[role])]
    return out


def patch_problems(cat, text, validator=None):
    """What the patch text lacks for gate CS3, as messages; empty when it holds."""
    problems = [f"the document-project patch does not state {want!r} (D5 §13.8)"
                for want in POST_REPORT_TEXTS if flat(want) not in flat(text)]
    section = role_map_section(text)
    if section is None:
        return problems + [f"the document-project patch has no single {ROLE_MAP_SECTION!r} "
                           "section"]
    problems += _table_problems(cat, section)
    if "docs/label-roles.json" not in section:
        problems.append("the patch's label-role map section does not name docs/label-roles.json")
    return problems + _example_problems(cat, section, validator)


def seeded_patches(cat, text):
    """(what, mutated patch text) pairs gate CS3 must refuse."""
    first, spec = next(iter(cat["report_roles"].items()))
    row = re.compile(r"^\|\s*`" + re.escape(first) + r"`\s*\|.*\n", re.M)
    other = next(k for k in cat["label_kinds"]["kinds"] if k != spec["kind"])
    swap = f"| {spec['kind']} |", f"| {other} |"
    extra = "| `net.head` | equation | net | an equation no role names |\n"
    sentence = re.compile(r"\s+".join(map(re.escape, POST_REPORT_TEXTS[0].split())))
    return [(f"the role {first} dropped (D5-39)", row.sub("", text, count=1)),
            ("the unknown role net.head added",
             row.sub(lambda m: m.group(0) + extra, text, count=1)),
            (f"the kind of {first} changed to {other}",
             row.sub(lambda m: m.group(0).replace(*swap, 1), text, count=1)),
            ("the section renamed", text.replace("# " + ROLE_MAP_SECTION, "# The label map")),
            ("the post-report exclusion sentence removed", sentence.sub("", text))]


def check_patch(cat, path):
    """Gate CS3 on the patch, then on its seeded copies (each must be refused)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"the document-project patch does not read ({path}): {exc}; gate CS3 reads it")
        return
    validator = role_map_validator(cat)
    problems = patch_problems(cat, text, validator)
    for problem in problems:
        fail(problem)
    if problems:
        return
    ok(f"the document-project patch names the {len(cat['report_roles'])} report roles and the "
       "post-report exclusion")
    for what, mutated in seeded_patches(cat, text):
        if mutated == text:
            fail(f"gate CS3 cannot seed {what}: the patch lacks the text the seed changes")
        elif not patch_problems(cat, mutated, validator):
            fail(f"gate CS3 accepts the document-project patch with {what}")
        else:
            ok(f"gate CS3 refuses the patch with {what}")


def patch_path(argv):
    if "--patch" not in argv:
        return DP_PATCH
    at = argv.index("--patch") + 1
    if at >= len(argv):
        raise SystemExit("--patch needs the path of a document-project SKILL.md")
    return Path(argv[at])
