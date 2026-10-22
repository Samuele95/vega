#!/usr/bin/env python3
"""
test_page_template.py — the page template holds no dataset literal and ships as itself (D4
§17.1 checks 1–7; BUILD-PLAN R-E2, R-F10; gates PG1–PG8, BD1).

    python3 -m pytest -q vega-show/scripts/tests/test_page_template.py

1. identity: `write_manifest()` pins every served file (the catalog's exclusions out) under
   `vega/page-template@1`; `TEMPLATE.json` equals it — checked only when the file exists (W15
   writes it); `verify_template_copy` is empty only for an exact copy (PN4: a byte changed in
   `fmt.js`); `digests()` is `SHOWCASE.json.template`; `requires_endpoints` is the declared list
   (D5 §2.2 and `GET /health`) and every route a module calls is declared required or optional;
2. prose only in the strings table (`page_lint`, every module, `js/legacy/**` included): PN1's
   `"benign traffic"` in `views/try.js` and typed text in the shell are caught;
3. the deny list is `lint_templates.py` (R-F10) over every template file: PN1 again;
4. no role literal (PN3: a `.threat` class in `page.css`); 5. digits (PN2: `"95 %"` in a
   strings value); 6. keys (VS19: a key the shell names deleted from the table); 7. no `{{…}}`
   (D5-30); PN10 (`--accent` changed in `page.css`) is caught by T04's style lock.

Every seeded negative runs on a copy of the template and asserts the findings its seed adds
(`_seeded`), so it holds whatever other packages' files hold today; a seed into a strings
fragment goes right after `export const STRINGS = Object.freeze({`, so it holds whether the
fragment is empty or filled. The installed template must pass each check as it is: those are
the `test_the_installed_template_*` tests, one per check.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(1, str(REPO / "vega" / "scripts" / "tests"))

import lint_templates  # noqa: E402
import page_checks  # noqa: E402
import page_lint  # noqa: E402
import page_template as pt  # noqa: E402
import style_lock  # noqa: E402
from page_words import TableError, Vocabulary, strings_table  # noqa: E402
from vega_show_core import digest_file  # noqa: E402

SCHEMAS = SCRIPTS.parent / "references" / "schemas"


@pytest.fixture
def copy(tmp_path) -> Path:
    """A copy of the installed template to seed."""
    return Path(shutil.copytree(pt.TEMPLATE_DIR, tmp_path / "page"))


def _validate(doc: dict, schema: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(schema).validate(doc)


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{path.name} no longer holds {old!r}: re-anchor the seed"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _module(copy: Path, rel: str, text: str) -> Path:
    path = copy / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


STRINGS_ANCHOR = "export const STRINGS = Object.freeze({"
FILLED_FRAGMENT = ("// a filled area fragment, as its owner writes it\n" + STRINGS_ANCHOR + "\n"
                   '  "net.filled.a": "a",\n  "net.filled.b": "b",\n});\nexport const DYNAMIC = Object.freeze({});\n')


def _seed_strings(copy: Path, area: str, entries: str) -> None:
    """Insert `entries` (`"key": value,` …) at the head of an area fragment's STRINGS table."""
    _edit(copy / "js" / "core" / "strings" / f"{area}.en.js", STRINGS_ANCHOR, f"{STRINGS_ANCHOR} {entries}")


def _seeded(check, copy: Path, seed) -> list[str]:
    """The findings `seed()` adds to what `check(copy)` finds on the copy as installed."""
    before = set(check(copy))
    seed()
    return [p for p in check(copy) if p not in before]


@pytest.fixture(params=["as installed", "filled"])
def fragment(request, copy) -> str:
    """The network fragment as installed, or filled as its owner (W05) fills it."""
    if request.param == "filled":
        (copy / "js" / "core" / "strings" / "network.en.js").write_text(FILLED_FRAGMENT, encoding="utf-8")
    return request.param


def _showcase_copy(template: Path, dest: Path) -> Path:
    """What build_showcase copies: the pinned files, byte for byte."""
    for rel in pt.write_manifest(template)["files"]:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template / rel, dest / rel)
    return dest


# ---- 1. identity ------------------------------------------------------------------------------------
def test_write_manifest_pins_every_served_file_under_the_schema():
    doc = pt.write_manifest()
    on_disk = {p.relative_to(pt.TEMPLATE_DIR).as_posix() for p in pt.TEMPLATE_DIR.rglob("*") if p.is_file()}
    served = {rel for rel in on_disk if not rel.startswith("tests/") and rel not in ("package.json", "TEMPLATE.json")}
    assert set(doc["files"]) == served and {"index.html", "strings.keys.json", "js/core/fmt.js"} <= served
    assert doc["strings_keys_digest"] == doc["files"]["strings.keys.json"] == digest_file(pt.TEMPLATE_DIR / "strings.keys.json")
    assert doc["version"] == pt.VERSION and doc["schema"] == "vega/page-template@1"
    _validate(doc, _schema("page-template-1.schema.json"))


def test_requires_endpoints_is_the_declared_list_D5(copy):
    live = pt.requires_endpoints()
    example = _schema("page-template-1.schema.json")["examples"][0]["requires_endpoints"]
    assert live == list(pt.REQUIRED_ENDPOINTS) and set(live) == set(example), "D5 §2.2's list and GET /health"
    assert "POST /v1/monitor" in pt.OPTIONAL_ENDPOINTS, "the monitor ships only with one (D5 §2.2)"
    assert not set(pt.REQUIRED_ENDPOINTS) & set(pt.OPTIONAL_ENDPOINTS)
    calls = pt.endpoint_calls()
    assert pt.endpoint_problems() == [] and {"GET /health", "GET /v1/schema", "POST /v1/predict"} <= set(calls)
    assert not [e for e in calls if "showcase/" in e or "?" in e], calls
    _module(copy, "js/views/extra.js", 'import { api, post, URLS } from "../core/api.js";\n'
            'export const soft = () => post(URLS.monitor, {});\n'
            'export const code = (f) => api(`${URLS.code}?file=${f}`);\n'
            'export const dot = (el) => el("span", { class: "health" });\n')
    assert pt.requires_endpoints(copy) == live and "js/views/extra.js:3" in pt.endpoint_calls(copy)["GET /v1/code"]
    _module(copy, "js/views/extra.js", 'import { api, URLS } from "../core/api.js";\n'
            'export const put = () => api(URLS.info, { method: "put" });\n')
    assert pt.endpoint_problems(copy) == ["js/views/extra.js:2: PUT /v1/info is called but is neither a required "
                                          "nor an optional endpoint (REQUIRED_ENDPOINTS / OPTIONAL_ENDPOINTS of "
                                          "page_template.py)"]
    with pytest.raises(pt.TemplateError, match="PUT /v1/info"):
        pt.write_manifest(copy)


@pytest.mark.parametrize("typed", ['api("v1/other")', 'api("/v1/other")', 'fetch("./health")', "api(`v1/x/${id}`)",
                                   "api(`/health`)", 'api(BASE + "/v1/predict")', 'const u = "/v1/predict"'])
def test_a_service_path_typed_outside_the_URLS_table_is_refused(copy, typed):
    _module(copy, "js/views/extra.js", f'import {{ api }} from "../core/api.js";\nexport const f = (id) => {typed};\n')
    with pytest.raises(pt.TemplateError, match="js/views/extra.js:2: .* types a service URL outside the URLS table"):
        pt.requires_endpoints(copy)


def test_a_call_on_an_unknown_URLS_name_is_refused(copy):
    _module(copy, "js/views/extra.js", 'import { api, URLS } from "../core/api.js";\nexport const f = () => api(URLS.nope);\n')
    with pytest.raises(pt.TemplateError, match="URLS.nope"):
        pt.requires_endpoints(copy)


def test_the_manifest_refuses_what_the_slot_would_not_serve(copy):
    (copy / "notes.md").write_text("x\n", encoding="utf-8")
    with pytest.raises(pt.TemplateError, match="notes.md"):
        pt.write_manifest(copy)
    (copy / "notes.md").unlink()
    (copy / "js" / "alias.js").symlink_to(copy / "js" / "main.js")
    with pytest.raises(pt.TemplateError, match="symlink"):
        pt.write_manifest(copy)


def test_identity_is_checked_only_when_TEMPLATE_json_exists(copy, capsys):
    assert pt.identity_problems() == [], "the installed TEMPLATE.json is stale: page_template.py --write"
    if not (copy / "TEMPLATE.json").exists():
        assert pt.main(["--template", str(copy), "--check"]) == 0 and "does not exist yet" in capsys.readouterr().out
    assert pt.main(["--template", str(copy), "--write"]) == 0
    pinned = json.loads((copy / "TEMPLATE.json").read_text(encoding="utf-8"))
    assert "TEMPLATE.json" not in pinned["files"] and pt.identity_problems(copy) == []
    fmt = copy / "js" / "core" / "fmt.js"
    fmt.write_bytes(fmt.read_bytes() + b"\n")
    (copy / "css" / "extra.css").write_text(".x{}\n", encoding="utf-8")
    problems = pt.identity_problems(copy)
    assert any(p.startswith("js/core/fmt.js: the template has sha256:") for p in problems), problems
    assert any(p.startswith("css/extra.css: in the template but not pinned") for p in problems), problems
    assert pt.main(["--template", str(copy), "--check"]) == 1
    _validate(pinned, _schema("page-template-1.schema.json"))


def test_verify_template_copy_is_empty_only_for_an_exact_copy_PN4(copy, tmp_path):
    page = _showcase_copy(pt.TEMPLATE_DIR, tmp_path / "showcase" / "page")
    assert pt.verify_template_copy(page) == []
    fmt = page / "js" / "core" / "fmt.js"
    fmt.write_bytes(fmt.read_bytes().replace(b"f", b"g", 1))                       # PN4
    (page / "js" / "core" / "api.js").unlink()
    (page / "extra.js").write_text("export {};\n", encoding="utf-8")
    (page / "link.css").symlink_to(page / "page.css")
    problems = pt.verify_template_copy(page)
    for rel, why in (("js/core/fmt.js", "the copy has sha256:"), ("js/core/api.js", "pinned but missing"),
                     ("extra.js", "in the copy but not pinned"), ("link.css", "a symlink")):
        assert any(p.startswith(f"{rel}: {why}") for p in problems), (rel, problems)
    assert pt.main(["--verify", str(page)]) == 1
    assert pt.verify_template_copy(tmp_path / "absent") == [f"{tmp_path / 'absent'}: not a directory"]


def test_a_copy_is_verified_against_TEMPLATE_json_once_it_exists(copy, tmp_path):
    assert pt.main(["--template", str(copy), "--write"]) == 0
    page = _showcase_copy(copy, tmp_path / "showcase" / "page")
    _edit(copy / "page.css", "--accent:#3fd0c2", "--accent:#3fd0c3")      # the template moves on, unpinned
    assert pt.verify_template_copy(page, copy) == [], "the copy is checked against what TEMPLATE.json pins"
    assert pt.identity_problems(copy) and pt.verify_template_copy(_showcase_copy(copy, tmp_path / "showcase" / "p2"), copy)


def test_digests_are_SHOWCASE_json_template(copy):
    before = pt.digests(copy)
    assert pt.main(["--template", str(copy), "--write"]) == 0
    after = pt.digests(copy)
    assert before == after == {"name": "vega-show/page", "version": pt.VERSION,
                               "manifest_digest": digest_file(copy / "TEMPLATE.json")}
    showcase = _schema("showcase-1.schema.json")
    _validate(pt.digests(), {**showcase["properties"]["template"], "$defs": showcase.get("$defs", {})})


# ---- 2. prose only in the strings table -------------------------------------------------------------------
def test_the_installed_template_holds_no_prose_literal():
    problems = [p for p in page_lint.lint_template().problems if p.startswith("PG2")]
    assert problems == [], "\n".join(problems)


def test_PN1_benign_traffic_in_views_try_is_caught_by_checks_2_and_3(copy):
    problems = _seeded(lambda d: page_lint.lint_template(d).problems, copy,
                       lambda: _module(copy, "js/views/try.js", 'export const VERDICT = "benign traffic";\n'))
    assert [p.split(" outside")[0] for p in problems] == ["PG2 js/views/try.js:1: prose literal 'benign traffic'"], problems
    report = lint_templates.lint([copy / "js" / "views" / "try.js"], root=REPO)
    assert {"BENIGN", "traffic"} <= {h.token for h in report.hits} and all(h.line == 1 for h in report.hits)


def test_the_classifier_reads_the_page_vocabulary(tmp_path):
    vocab = Vocabulary.of(pt.TEMPLATE_DIR)
    fine = ["caps muted", "nav.index button[data-view]", "#exp-bars .bar .name", "fill:var(--muted);font:11px var(--mono)",
            "(prefers-reduced-motion: reduce)", "(min-width: 600px) and (hover: hover)", "./strings/core.en.js",
            "app.status.ready", "metric.", '<span class="tk-s">x</span>', "button:not(.on)", "hidden"]
    assert [t for t in fine if page_lint.classify(t, vocab)[0] == "prose"] == []
    prose = ["benign traffic", "not ready", "the service is not ready", "is hidden", '<label for="x">']
    assert [t for t in prose if page_lint.classify(t, vocab)[0] != "prose"] == []
    mod = tmp_path / "m.js"
    mod.write_text('const r = /the rows are here/g;\nconst p = new RegExp("the rows are " + x);\n'
                   'const s = `${n} rows are ready`;\n// the rows are here\n', encoding="utf-8")
    found = page_lint.literals(mod, vocab, "js/views/m.js")
    assert [(x.line, x.category) for x in found] == [(1, "regex"), (2, "regex"), (3, "prose")]
    assert {x.category for x in page_lint.literals(mod, vocab, "js/core/strings/views.en.js")} == {"table"}


def test_the_shell_holds_no_prose(copy):
    shell = copy / "index.html"

    def seed():
        _edit(shell, '<span data-t="start.begin"></span>', '<span data-t="start.begin">begin</span>')
        _edit(shell, '<main id="main">', '<main id="main"><p>see the rows</p>')
        _edit(shell, 'data-t-aria="app.theme.aria"', 'aria-label="Switch theme"')
    problems = _seeded(lambda d: page_lint.html_problems(d / "index.html")[0], copy, seed)
    for why in ("typed inside a data-t slot", "text node 'see the rows'", "aria-label=…> types text"):
        assert any(why in p for p in problems), (why, problems)
    assert not any("→" in p or "≡" in p for p in problems), "symbols and entities are not prose"


# ---- 3–7 -------------------------------------------------------------------------------------------------
def _served(template: Path) -> list[Path]:
    """Every template file check 3 reads (D4 §17.1: any template file, js/legacy/** included)."""
    return [template / r for r in pt.template_files(template)] + \
        ([template / pt.MANIFEST] if (template / pt.MANIFEST).exists() else [])


def test_check_3_the_deny_list_holds_on_the_served_files():
    report = lint_templates.lint(_served(pt.TEMPLATE_DIR), root=REPO)
    assert report.hits == [], [h.line_out(True) for h in report.hits]


INSTALLED = {
    "check 4 (PG4) no role literal": page_checks.role_literals,
    "check 5 (PG5) the digit rule": page_checks.digit_problems,
    "check 6 (PG6) every key named and every key used": page_checks.key_problems,
    "check 7 (PG7) nothing substituted": page_checks.brace_tokens,
    "PN10's lock (style-lock.json)": lambda d: style_lock.check(str(d), style_lock.load_lock(), None),
}


@pytest.mark.parametrize("check", sorted(INSTALLED))
def test_the_installed_template_passes(check):
    problems = INSTALLED[check](pt.TEMPLATE_DIR)
    assert problems == [], "\n".join(problems)


def test_check_4_no_role_literal_PN3(copy):
    def seed():
        with (copy / "page.css").open("a", encoding="utf-8") as sheet:
            sheet.write(".threat{color:var(--alert)}\n")                                  # PN3
        _module(copy, "js/views/roles2.js", 'export const isCalm = (c) => c === "attack";\n')
    problems = _seeded(page_checks.role_literals, copy, seed)
    assert any(p.startswith("PG4 page.css:") and "'threat'" in p for p in problems), problems
    assert {p.split(":")[0] for p in problems if "roles2.js" in p} == {"PG4 js/views/roles2.js"}
    assert len([p for p in problems if "roles2.js" in p]) == 2 and len(problems) == 3, problems


def test_check_5_digits_PN2(copy, tmp_path):
    problems = _seeded(page_checks.digit_problems, copy, lambda: _edit(
        copy / "js/core/strings/core.en.js", '"app.status.ready": "ready"', '"app.status.ready": "95 %"'))   # PN2
    assert len(problems) == 1 and "strings[app.status.ready] (core)" in problems[0] and "digit rule" in problems[0]
    catalog = json.loads(page_checks.EXPLAIN.read_text(encoding="utf-8"))
    catalog["blocks"]["dense"]["prose"] = "adds 3 numbers"
    (tmp_path / "explain.json").write_text(json.dumps(catalog), encoding="utf-8")
    before = set(page_checks.digit_problems(pt.TEMPLATE_DIR))
    problems = [p for p in page_checks.digit_problems(pt.TEMPLATE_DIR, tmp_path / "explain.json") if p not in before]
    assert len(problems) == 1 and "explain-catalog blocks.dense.prose" in problems[0], problems


def test_check_6_keys_VS19(copy, fragment):
    def seed():
        _edit(copy / "js/core/strings/core.en.js", '  "start.begin": "begin",\n', "")                   # VS19
        _seed_strings(copy, "network", '"net.unused": "never named", "app.fresh.a": "a",')
        _module(copy, "js/views/keys.js", 'import { t } from "../core/text.js";\n'
                'export const f = (x) => t("app.fresh." + x);\nexport const g = () => t("app.nowhere");\n')
    problems = _seeded(page_checks.key_problems, copy, seed)
    shell = (copy / "index.html").read_text(encoding="utf-8").splitlines()
    begin = next(i for i, line in enumerate(shell, start=1) if 'data-t="start.begin"' in line)
    for why in (f"PG6 index.html:{begin}: key 'start.begin' is not in the strings table",
                "PG6 js/views/keys.js:3: key 'app.nowhere' is not in the strings table",
                "PG6 js/views/keys.js:2: dynamic prefix 'app.fresh.' is not declared",
                "PG6 strings[net.unused] (network): a dead key"):
        assert any(p.startswith(why) for p in problems), (why, fragment, problems)


def test_check_7_nothing_is_substituted_D5_30(copy):
    problems = _seeded(page_checks.brace_tokens, copy,
                       lambda: _edit(copy / "index.html", "<title></title>", "<title>{{TITLE_HTML}}</title>"))
    shell = (copy / "index.html").read_text(encoding="utf-8").splitlines()
    line = next(i for i, text in enumerate(shell, start=1) if "{{TITLE_HTML}}" in text)
    assert problems == [f"PG7 index.html:{line}: '{{{{TITLE_HTML}}}}' — nothing in the template is substituted"]


def test_PN10_the_style_lock_catches_an_accent_change(copy):
    lock = style_lock.load_lock()
    problems = _seeded(lambda d: style_lock.check(str(d), lock, None), copy,
                       lambda: _edit(copy / "page.css", "--accent:#3fd0c2", "--accent:#3fd0c3"))     # PN10
    assert any(p.startswith("token --accent [night]") and "#3fd0c3" in p for p in problems), problems


# ---- the strings table as data ------------------------------------------------------------------------------
def test_the_strings_table_is_read_as_the_page_merges_it(copy, fragment):
    table = strings_table(copy)
    assert table.strings["app.status.ready"] == "ready" and table.areas["app.status.ready"] == "core"
    if fragment == "filled":
        assert table.strings["net.filled.a"] == "a" and table.areas["net.filled.a"] == "network"
    node = shutil.which("node")
    if node:
        script = ('const s = await import("./js/core/strings.en.js"), k = await import("./js/core/strings.keys.js");'
                  "console.log(JSON.stringify({strings: s.STRINGS, dynamic: k.DYNAMIC_KEYS}));")
        out = subprocess.run([node, "--input-type=module", "-e", script], cwd=copy,
                             capture_output=True, text=True, timeout=60, check=True).stdout
        assert json.loads(out) == {"strings": table.strings, "dynamic": table.dynamic}
    network = copy / "js" / "core" / "strings" / "network.en.js"
    shipped = network.read_text(encoding="utf-8")
    _seed_strings(copy, "network", '"app.back": "again",')
    with pytest.raises(TableError, match=re.escape("'app.back' is defined by both core and network")):
        strings_table(copy)
    network.write_text(shipped, encoding="utf-8")
    _seed_strings(copy, "network", '"net.x": label(),')
    with pytest.raises(TableError, match="not plain data"):
        strings_table(copy)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
