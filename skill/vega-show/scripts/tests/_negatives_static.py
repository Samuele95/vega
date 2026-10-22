"""
_negatives_static — `run_negatives.py`'s cases on the skill's own files, no lane needed: the page
template's gates (PG1–PG9, PG11, each on a seeded copy of `assets/page/`, through the gate's own
tool or check), the catalog sync's (CS1–CS3, on a seeded copy of the contract, a schema or the
document-project patch), and the two design cases caught outside vega-show (D5-38 by the vega-forge
catalog sync, D5-40 by `conformance.py --size-only`, C11).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from _negatives_kit import (CATALOG, HERE, PY, SCRIPTS, SKILLS, Check, Expect, case, edit_text, py, run, show,
                            template_copy)

LINT = SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
STYLE_LOCK = SKILLS / "vega" / "scripts" / "tests" / "style_lock.py"
CONFORMANCE = SKILLS / "vega-forge" / "scripts" / "conformance.py"
FORGE_TESTS = SKILLS / "vega-forge" / "scripts" / "tests"
CONTRACT = SCRIPTS.parent / "references" / "show-contract.md"
SCHEMAS = SCRIPTS.parent / "references" / "schemas"
PATCH = SKILLS / "patches" / "document-project" / "SKILL.md"
STRINGS_ANCHOR = '"app.status.ready": "ready"'


def page_check(name: str, page: Path) -> tuple[int, str]:
    """One of `page_checks`' template checks (test_page_template.py's checks 4–7) on `page`:
    exit 1 with its problems, 0 with none."""
    return py(f"import page_checks\nfrom pathlib import Path\nfound = page_checks.{name}(Path(sys.argv[1]))\n"
              "print('\\n'.join(found))\nsys.exit(1 if found else 0)\n", page)


def module(page: Path, rel: str, text: str) -> None:
    path = page / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# =============================================================================== page template

@case("PG1", "", None, "a template file changed and TEMPLATE.json not rewritten")
def pg1(tmp):
    page = template_copy(tmp, lambda p: edit_text(p / "page.css", "--accent:", "--accent: "), pin=False)
    rc, out = show("page_template.py", "--template", page, "--check")
    return Expect(rc, out, 1, ("page.css",))


@case("PG2", "PN1", None, "a string literal 'benign traffic' in views/try.js")
def pg2(tmp):
    page = template_copy(tmp, lambda p: module(p, "js/views/try.js", 'export const VERDICT = "benign traffic";\n'),
                         pin=False)
    rc, out = show("page_lint.py", "--template", page)
    return Expect(rc, out, 1, ("PG2 js/views/try.js:1", "benign traffic"))


@case("PG3", "PN1 D5-31 VS20", None, "a class name (BENIGN) seeded into the page template")
def pg3(tmp):
    page = template_copy(tmp, pin=False)
    start = page / "js" / "views" / "start.js"
    start.write_text(start.read_text(encoding="utf-8") + '\nexport const NEUTRAL = "BENIGN";\n', encoding="utf-8")
    line = len(start.read_text(encoding="utf-8").splitlines())
    rc, out = run([PY, LINT, start])
    return Expect(rc, out, 1, ("BENIGN", f"start.js:{line}"))


@case("PG4", "PN3", None, "a .threat class in page.css")
def pg4(tmp):
    page = template_copy(tmp, pin=False)
    with (page / "page.css").open("a", encoding="utf-8") as sheet:
        sheet.write(".threat{color:var(--alert)}\n")
    return Expect(*page_check("role_literals", page), 1, ("PG4 page.css", "threat"))


@case("PG5", "PN2", None, "'95 %' typed into a strings value")
def pg5(tmp):
    page = template_copy(tmp, lambda p: edit_text(p / "js/core/strings/core.en.js", STRINGS_ANCHOR,
                                                  '"app.status.ready": "95 %"'), pin=False)
    return Expect(*page_check("digit_problems", page), 1, ("app.status.ready", "digit rule"))


@case("PG6", "VS19", None, "a chrome key the shell uses deleted from a copy of the strings table")
def pg6(tmp):
    page = template_copy(tmp, lambda p: edit_text(p / "js/core/strings/core.en.js", '  "start.begin": "begin",\n', ""),
                         pin=False)
    return Expect(*page_check("key_problems", page), 1, ("PG6 index.html", "start.begin"))


@case("PG7", "D5-30", "L02", "a template with a leftover {{TITLE_HTML}} token, built")
def pg7(w):
    page = template_copy(w.root / "tpl", lambda p: edit_text(p / "index.html", "<title>", "<title>{{TITLE_HTML}}"),
                         pin=False)
    rc, out = w.build("--template", page)
    return Expect(rc, out, 2, ("ERROR: PG7 ", "TITLE_HTML"))


@case("PG8", "PN10", None, "--accent changed in page.css")
def pg8(tmp):
    page = template_copy(tmp, lambda p: edit_text(p / "page.css", "--accent:#3fd0c2", "--accent:#3fd0c3"), pin=False)
    rc, out = run([PY, STYLE_LOCK, "check", page, "--static-only"])
    return Expect(rc, out, 1, ("--accent", "#3fd0c3"))


@case("PG9", "PN11", None, "a 450-line module")
def pg9(tmp):
    page = template_copy(tmp, lambda p: module(p, "js/views/long.js", "".join(
        f"export const K{i} = {i};\n\n" for i in range(450))), pin=False)
    rc, out = show("page_lint.py", "--template", page)
    return Expect(rc, out, 1, ("PS1 js/views/long.js:1 module", "450"))


@case("PG11", "PN9", None, "a catalog block type removed from KIND_OF")
def pg11(tmp):
    if not shutil.which("node"):
        return Check(False, "node is not on PATH: coverage.test.js cannot run")
    page = template_copy(tmp, lambda p: edit_text(p / "js/net/kinds.js", 'gin_conv: "graph_conv", ', ""), pin=False)
    rc, out = run([shutil.which("node"), "--test", page / "tests" / "coverage.test.js"],
                  env={"VEGA_FORGE": str(SKILLS / "vega-forge" / "scripts")})
    return Expect(rc, out, 1, ("block gin_conv has no stage kind",))


# =============================================================================== catalog sync

SYNC = ("import json\nimport _sync_common as c\nfrom pathlib import Path\n"
        "cat = json.loads(c.CATALOG_JSON.read_text(encoding='utf-8'))\n")


def failures_of(body: str, *args) -> tuple[int, str]:
    return py(SYNC + body + "\nprint('\\n'.join('FAIL: ' + f for f in c.failures))\n"
              "sys.exit(1 if c.failures else 0)\n", *args)


@case("CS1", "", None, "the contract's gate table loses the LY7 row the catalog holds")
def cs1(tmp):
    seeded = tmp / "show-contract.md"
    text = CONTRACT.read_text(encoding="utf-8")
    row = re.compile(r"^\| `LY7` \|.*\n", re.M)
    if not row.search(text):
        return Check(False, "the contract has no LY7 row to drop: re-anchor the seed")
    seeded.write_text(row.sub("", text, count=1), encoding="utf-8")
    rc, out = failures_of("from _sync_contract import check_contract\n"
                          "check_contract(cat, Path(sys.argv[1]).read_text(encoding='utf-8'))", seeded)
    return Expect(rc, out, 1, ("LY7",))


@case("CS2", "", None, "a schema whose example no longer validates")
def cs2(tmp):
    schema = json.loads((SCHEMAS / "showcase-1.schema.json").read_text(encoding="utf-8"))
    schema["examples"][0].pop("entry")
    (tmp / "showcase-1.schema.json").write_text(json.dumps(schema), encoding="utf-8")
    rc, out = failures_of("from jsonschema import Draft202012Validator\nfrom _sync_schemas import _check_one_schema\n"
                          "_check_one_schema(Draft202012Validator, 'vega/showcase@1', 'showcase-1', "
                          "json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))", tmp / "showcase-1.schema.json")
    return Expect(rc, out, 1, ("example 0 does not validate", "entry"))


@case("CS3", "D5-39", None, "a report role dropped from the document-project patch's label-role map")
def cs3(tmp):
    role = next(iter(CATALOG["report_roles"]))
    text = PATCH.read_text(encoding="utf-8")
    row = re.compile(r"^\|\s*`" + re.escape(role) + r"`\s*\|.*\n", re.M)
    if not row.search(text):
        return Check(False, f"the patch has no row for {role}: re-anchor the seed")
    seeded = tmp / "SKILL.md"
    seeded.write_text(row.sub("", text, count=1), encoding="utf-8")
    rc, out = run([PY, HERE / "test_catalog_sync.py", "--patch", seeded])
    return Expect(rc, out, 1, (role,))


# =============================================================================== caught outside vega-show

@case("C11", "D5-40", None, "a function of 31 statements added to a vega-show script")
def c11(tmp):
    scripts = tmp / "scripts"
    shutil.copytree(SCRIPTS, scripts, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    body = "".join(f"    x{i} = {i}\n" for i in range(30))
    with (scripts / "digit_rule.py").open("a", encoding="utf-8") as fh:
        fh.write(f"\n\ndef seeded_long_function():\n{body}    return x0\n")
    rc, out = run([PY, CONFORMANCE, "--size-only", scripts])
    return Expect(rc, out, 2, ("ERROR: C11 ", "seeded_long_function"))


@case("", "D5-38", None, "a metric in METRICS without a direction (the vega-forge catalog sync)")
def d5_38(tmp):
    return Expect(*py(f"sys.path.insert(0, {str(FORGE_TESTS)!r})\nimport test_catalog_sync as t\n"
                      "real = t.metrics_py_names\nt.metrics_py_names = lambda: [*real(), 'seeded_metric']\n"
                      "try:\n    t.test_metric_direction()\nexcept AssertionError as exc:\n"
                      "    print('FAIL:', exc)\n    sys.exit(1)\nsys.exit(0)\n"), 1, ("seeded_metric",))


CASES = [v for v in list(globals().values()) if type(v).__name__ == "Case"]
