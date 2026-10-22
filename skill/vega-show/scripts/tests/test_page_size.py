#!/usr/bin/env python3
"""
test_page_size.py — the page's size discipline, PS1–PS5 through `page_lint.metrics` (D4 §3.4,
§17.2; gate PG9; seeded negative PN11).

    python3 -m pytest -q vega-show/scripts/tests/test_page_size.py

- the installed template's served modules hold PS1–PS5 (`js/legacy/**` is left out of the size
  rules while it exists — each area's package replaces its legacy file — and still read by PG2);
- PN11: a 450-line module is caught (PS1); the strings table is exempt from PS1 and nothing else;
- PS2 function length (`{` to `}`), PS3 class length (`class` to `}`), PS4 parameters (a
  destructured object is one), PS5 module-level state (`let`, `var`, a mutable literal or
  collection; `js/core/state.js` and `js/core/decode.js`'s WeakMap exempt) — each at its limit
  and one past it;
- an expression-bodied arrow runs from `=>` to its expression's last token;
- braces inside strings, templates, regexes and comments are not structure, and a `/` after the
  head of `if` / `while` / `for` / `with` opens a regex.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import page_lint  # noqa: E402
import page_template as pt  # noqa: E402


@pytest.fixture
def copy(tmp_path) -> Path:
    return Path(shutil.copytree(pt.TEMPLATE_DIR, tmp_path / "page"))


def _problems(tmp_path: Path, rel: str, text: str) -> list[str]:
    path = tmp_path / "m.js"
    path.write_text(text, encoding="utf-8")
    return page_lint.metrics(path, rel).problems(rel)


def _function(name: str, lines: int, head: str = "function {name}() {{") -> str:
    """A function whose body runs `lines` lines, `{` and `}` counted."""
    return "\n".join([head.format(name=name), *(["  x += 1;"] * (lines - 2)), "}"]) + "\n"


def test_the_installed_template_holds_ps1_to_ps5():
    problems = [p for p in page_lint.lint_template().problems if p.startswith("PS")]
    assert problems == [], "\n".join(problems)


def test_PN11_a_450_line_module_is_caught(copy, capsys):
    long = copy / "js" / "views" / "long.js"
    long.parent.mkdir(parents=True, exist_ok=True)
    long.write_text("".join(f"export const K{i} = {i};\n\n" for i in range(450)), encoding="utf-8")   # PN11
    problems = page_lint.lint_template(copy).problems
    assert "PS1 js/views/long.js:1 module — 450 non-blank lines > 400" in problems, problems
    assert page_lint.main(["--template", str(copy)]) == 1
    assert "PS1 js/views/long.js:1 module" in capsys.readouterr().out


def test_the_strings_table_is_exempt_from_PS1_only(tmp_path):
    table = "".join(f'  "k.a{i}": "a",\n' for i in range(420))
    text = f"export const STRINGS = Object.freeze({{\n{table}}});\nlet leak = 1;\n"
    for rel in ("js/core/strings.en.js", "js/core/strings/views.en.js"):
        assert _problems(tmp_path, rel, text) == [f"PS5 {rel}:{423} leak — module-level let: mutable state "
                                                   "lives in js/core/state.js"], rel
    assert _problems(tmp_path, "js/core/text.js", text)[0] == "PS1 js/core/text.js:1 module — 423 non-blank lines > 400"
    # the installed fragments are the table's data whatever their length (the 420-key table above
    # proves the exemption past 400 lines; after the legacy.how.* keys left, no fragment passes it)
    for frag in sorted((pt.TEMPLATE_DIR / "js/core/strings").glob("*.en.js")):
        rel = f"js/core/strings/{frag.name}"
        assert page_lint.metrics(frag, rel).problems() == [], f"{rel} is the table's data"


def test_PS2_a_function_runs_at_most_fifty_lines(tmp_path):
    rel = "js/views/f.js"
    assert _problems(tmp_path, rel, _function("ok", 50)) == []
    assert _problems(tmp_path, rel, _function("long", 51)) == [f"PS2 {rel}:1 long — 51 lines > 50"]
    arrow = _function("draw", 51, "export const {name} = (a) => {{")
    assert _problems(tmp_path, rel, arrow) == [f"PS2 {rel}:1 draw — 51 lines > 50"]
    method = "class A {\n" + _function("paint", 51, "  {name}(g) {{") + "}\n"
    assert _problems(tmp_path, rel, method) == [f"PS2 {rel}:2 paint — 51 lines > 50"]


def test_PS2_an_expression_bodied_arrow_runs_from_the_arrow_to_its_last_token(tmp_path):
    rel = "js/views/f.js"
    arrow = lambda n: ("export const draw = (a) =>\n  g(a,\n" + "    a,\n" * (n - 3)  # noqa: E731
                       + "  );\nexport const h = (x) => x + 1, k = (y) => [y,\n  y];\n")
    assert _problems(tmp_path, rel, arrow(50)) == []
    assert _problems(tmp_path, rel, arrow(51)) == [f"PS2 {rel}:1 draw — 51 lines > 50"]
    m = page_lint.metrics(tmp_path / "m.js", rel)
    assert [(f.name, f.lines) for f in m.functions] == [("draw", 51), ("h", 1), ("k", 2)]
    no_semicolons = "export const f = (x) =>\n  x + 1\nexport const g = 2\n"
    (tmp_path / "n.js").write_text(no_semicolons, encoding="utf-8")
    assert [(f.name, f.lines) for f in page_lint.metrics(tmp_path / "n.js", rel).functions] == [("f", 2)]


def test_PS3_a_class_runs_at_most_two_hundred_lines(tmp_path):
    rel = "js/net/renderers/dense.js"
    body = lambda n: "export class Diagram {\n" + "".join(f"  m{i}() {{ return {i}; }}\n" for i in range(n)) + "}\n"  # noqa: E731
    assert _problems(tmp_path, rel, body(198)) == []
    assert _problems(tmp_path, rel, body(199)) == [f"PS3 {rel}:1 class Diagram — 201 lines > 200"]


def test_PS4_at_most_five_parameters_a_destructured_object_is_one(tmp_path):
    rel = "js/core/bars.js"
    fine = ("export function barRow({ name, v, max, label, selected, i, ci, cls, fillCls }) { return name; }\n"
            "export const f = (a, b, c, d, e = g(1, 2)) => a;\nexport function h([x, y], ...rest) {}\n")
    assert _problems(tmp_path, rel, fine) == []
    wide = ("export function barRow(name, v, max, label, selected, i) {}\n"
            "export const sum = (a, b, c, d, e, f) => a;\nclass K { m(a, b, c, d, e, { f }) {} }\n")
    assert _problems(tmp_path, rel, wide) == [f"PS4 {rel}:1 barRow — 6 parameters > 5",
                                              f"PS4 {rel}:2 sum — 6 parameters > 5",
                                              f"PS4 {rel}:3 m — 6 parameters > 5"]


def test_PS5_no_module_level_state_outside_state_js(tmp_path):
    text = ("let at = 0;\nvar old;\nconst CACHE = new Map(), SEEN = [];\nexport const S = { view: null };\n"
            "const WEAK = new WeakMap();\nconst FIXED = Object.freeze({ a: 1 });\nconst N = 3, F = (x) => x;\n"
            "export function f() { let i = 0; const o = {}; for (let j = 0; j < 2; j++) i += j; return o; }\n")
    rel = "js/views/tour.js"
    found = [p.split(" — ")[0] for p in _problems(tmp_path, rel, text)]
    assert found == [f"PS5 {rel}:1 at", f"PS5 {rel}:2 old", f"PS5 {rel}:3 CACHE", f"PS5 {rel}:3 SEEN",
                     f"PS5 {rel}:4 S", f"PS5 {rel}:5 WEAK"]
    assert _problems(tmp_path, "js/core/state.js", text) == []
    decode = [p.split(" — ")[0] for p in _problems(tmp_path, "js/core/decode.js", text)]
    assert "PS5 js/core/decode.js:5 WEAK" not in decode and "PS5 js/core/decode.js:1 at" in decode


def test_braces_in_strings_templates_regexes_and_comments_are_not_structure(tmp_path):
    text = ('export function f(s) {\n  const a = "{ ( [";\n  const r = /\\{\\}[{]/g;\n'
            "  // } ) ]\n  const t = `${ { k: '}' }.k } {`;\n  /* } */\n  return a + r + t;\n}\n")
    (tmp_path / "x.js").write_text(text, encoding="utf-8")
    m = page_lint.metrics(tmp_path / "x.js")
    assert [(f.name, f.line, f.lines, f.params) for f in m.functions] == [("f", 1, 8, 1)]
    assert m.classes == () and m.state == ()


def test_legacy_modules_are_measured_only_with_the_flag_and_always_read_by_PG2(copy):
    served = page_lint.modules_of(pt.TEMPLATE_DIR)
    assert served and not [r for r in served if r.startswith(page_lint.LEGACY)]
    everything = page_lint.modules_of(pt.TEMPLATE_DIR, legacy=True)
    legacy = sorted(set(everything) - set(served))
    assert legacy == sorted(r for r in pt.template_files(pt.TEMPLATE_DIR) if r.startswith("js/legacy/"))
    old = copy / "js" / "legacy" / "old.js"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text('let at = 0;\nexport const s = "the rows are here";\n', encoding="utf-8")
    plain = page_lint.lint_template(copy).problems
    assert any(p.startswith("PG2 js/legacy/old.js:2: prose literal") for p in plain), plain
    assert not [p for p in plain if p.startswith("PS") and "js/legacy/" in p], plain
    flagged = page_lint.lint_template(copy, legacy=True).problems
    assert any(p.startswith("PS5 js/legacy/old.js:1 at") for p in flagged), flagged


def test_a_slash_after_a_statement_head_opens_a_regex(tmp_path):
    rel = "js/views/r.js"
    path = tmp_path / "r.js"
    path.write_text('export function f(s, ok) {\n  if (ok) /"x/.test(s);\n  while (g(s)) /y/g.exec(s);\n'
                    '  return Symbol.for("k") / 2 / h(s) / 3;\n}\n', encoding="utf-8")
    found = page_lint.literals(path, None, rel)
    assert [(x.line, x.kind) for x in found] == [(2, "regex"), (3, "regex"), (4, "string")]
    assert [(f.name, f.lines) for f in page_lint.metrics(path, rel).functions] == [("f", 5)]


def test_the_command_line_exits_two_on_an_unreadable_module(tmp_path, capsys):
    bad = tmp_path / "bad.js"
    bad.write_text('export const s = "open;\n', encoding="utf-8")
    assert page_lint.main([str(bad)]) == 2 and "unterminated string" in capsys.readouterr().out
    block = tmp_path / "block.js"
    block.write_text('if (ok) {} /"x/.test(s);\n', encoding="utf-8")
    assert page_lint.main([str(block)]) == 2
    assert "was read as division" in capsys.readouterr().out, "the message names the tokenizer's limit"
    good = tmp_path / "good.js"
    good.write_text("export const f = (a) => a;\n", encoding="utf-8")
    assert page_lint.main([str(good)]) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
