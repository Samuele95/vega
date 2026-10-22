"""
page_checks — D4 §17.1 checks 4–7 over a page template directory (gates PG4–PG7), as
functions a test or a seeded negative runs on the installed template or on a copy of it.
Each returns one line per finding, `PGn <file>:<line>: <why>`; `[]` means the check holds.

- `role_literals` (PG4): no identifier, CSS class, custom property or string of any module or
  sheet says `calm`, `threat`, `benign` or `attack` (a plural or a camel-case part counts:
  `isThreat`, `.threat-row`, `--calm`). A role is a word of the role map (neutral, alert,
  plain), never of a lane.
- `digit_problems` (PG5): every value of the strings table and every prose field of the
  backend's explain catalog passes `digit_rule.check_prose` (no digit outside a placeholder,
  Unicode rule; placeholders of the grammar).
- `key_problems` (PG6): every key the modules and the shell name exists (a literal given to
  `t`, `tn`, `tText`, `tnText` or `hasKey`, a key-shaped literal under a root of the table,
  a `data-t`, `data-t-title`, `data-t-aria` slot); every dynamic prefix a module builds a key
  from (`"view." + id`, `` `metric.${m}` ``) is declared in a fragment's `DYNAMIC`, and each
  of its members has its key; no key of the table is named nowhere.
- `brace_tokens` (PG7): no `{{…}}` in any template file (and in `TEMPLATE.json`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from digit_rule import check_prose
from js_tokens import flat, tokenize_file
from page_lint import html_problems, is_strings_table
from page_template import MANIFEST, template_files
from page_words import EXPLAIN, strings_table

__all__ = ["ROLE_WORDS", "role_literals", "digit_problems", "explain_prose", "key_problems",
           "brace_tokens"]

ROLE_WORDS = frozenset({"calm", "threat", "threats", "benign", "attack", "attacks"})
KEY_RE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_][a-z0-9_-]*)+(?:@[a-z_]+(?:\+[a-z_]+)?)?(?:#(?:zero|one|other))?")
PREFIX_RE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_][a-z0-9_-]*)*\.")
FILE_RE = re.compile(r"\.(?:pdf|json|js|css|html|csv|parquet)$")
KEY_FUNCTIONS = frozenset({"t", "tn", "tText", "tnText", "hasKey"})
PROSE_KEYS = frozenset({"label", "label_many", "step", "prose", "unit_how", "title", "text", "name",
                        "bend", "answer"})
BRACES_RE = re.compile(r"\{\{[^{}]*\}\}")


def _parts(text: str) -> list[str]:
    """The lower-case word parts of a text: camel case, snake case and hyphens split."""
    return [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", text)]


def _modules(template: Path) -> list[str]:
    return [rel for rel in template_files(template) if rel.endswith(".js")]


# ---- PG4 --------------------------------------------------------------------------------------------------
def role_literals(template: Path) -> list[str]:
    template, out = Path(template), []
    for rel in _modules(template):
        for t in flat(tokenize_file(template / rel)):
            text = t.value if t.kind in ("string", "template") else t.text if t.kind == "ident" else ""
            hit = sorted(set(_parts(text)) & ROLE_WORDS)
            if hit:
                out.append(f"PG4 {rel}:{t.line}: role literal {hit[0]!r} in the {t.kind} {t.text[:60]!r}")
    for rel in template_files(template):
        if rel.endswith((".css", ".html")):
            text = re.sub(r"/\*.*?\*/|<!--.*?-->", lambda m: "\n" * m.group().count("\n"),
                          (template / rel).read_text(encoding="utf-8"), flags=re.S)
            for line_no, line in enumerate(text.splitlines(), start=1):
                hit = sorted(set(_parts(line)) & ROLE_WORDS)
                if hit:
                    out.append(f"PG4 {rel}:{line_no}: role literal {hit[0]!r}: {line.strip()[:80]!r}")
    return out


# ---- PG5 --------------------------------------------------------------------------------------------------
def explain_prose(catalog: dict):
    """(where, text) of every prose field of the explain catalog (its formulas and code
    anchors are notation, not prose)."""
    def walk(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key in PROSE_KEYS:
                    yield f"{where}.{key}", value
                elif isinstance(value, (dict, list)) and key != "code":
                    yield from walk(value, f"{where}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield from walk(value, f"{where}[{i}]")
    for section in ("blocks", "activations", "links", "walk_fixed"):
        yield from walk(catalog.get(section) or {}, section)


def digit_problems(template: Path, explain: Path = EXPLAIN) -> list[str]:
    table = strings_table(Path(template))
    out = [f"PG5 {line}" for key, text in table.strings.items()
           for line in check_prose(text, where=f"strings[{key}] ({table.areas[key]})")]
    catalog = json.loads(Path(explain).read_text(encoding="utf-8"))
    return out + [f"PG5 {line}" for where, text in explain_prose(catalog)
                  for line in check_prose(text, where=f"explain-catalog {where}")]


# ---- PG6 --------------------------------------------------------------------------------------------------
def _base(key: str) -> str:
    return re.split(r"[@#]", key)[0]


def _named(template: Path, roots: set) -> tuple[list, list]:
    """([(rel, line, key)] named by the modules and the shell, [(rel, line, prefix)] built on)."""
    named, prefixes = [], []
    for rel in _modules(template):
        if is_strings_table(rel):
            continue
        toks = flat(tokenize_file(template / rel))
        for i, t in enumerate(toks):
            called = i >= 2 and toks[i - 1].is_("punct", "(") and toks[i - 2].text in KEY_FUNCTIONS
            if t.kind == "string" and (called or (KEY_RE.fullmatch(t.value) and t.value.split(".")[0] in roots
                                                  and not FILE_RE.search(t.value))):
                named.append((rel, t.line, t.value))
            head = t.parts[0] if t.kind == "template" and len(t.parts) > 1 else \
                t.value if t.kind == "string" and i + 1 < len(toks) and toks[i + 1].is_("punct", "+") else None
            if head and PREFIX_RE.fullmatch(head) and head.split(".")[0] in roots:
                prefixes.append((rel, t.line, head))
    named += [("index.html", line, key) for line, key in html_problems(template / "index.html")[1]]
    return named, prefixes


def key_problems(template: Path) -> list[str]:
    template = Path(template)
    table = strings_table(template)
    keys, bases = set(table.strings), {_base(k) for k in table.strings}
    named, prefixes = _named(template, {k.split(".")[0] for k in keys})
    out = [f"PG6 {rel}:{line}: key {key!r} is not in the strings table"
           for rel, line, key in named if key not in keys and key not in bases]
    out += [f"PG6 {rel}:{line}: dynamic prefix {p!r} is not declared in a fragment's DYNAMIC"
            for rel, line, p in prefixes if p not in table.dynamic]
    dynamic = {p + m for p, members in table.dynamic.items() for m in members}
    out += [f"PG6 DYNAMIC {k!r}: the member has no key" for k in sorted(dynamic - bases)]
    used = {key for _, _, key in named} | dynamic
    out += [f"PG6 strings[{k}] ({table.areas[k]}): a dead key, named nowhere"
            for k in sorted(keys) if k not in used and _base(k) not in used]
    return out


# ---- PG7 --------------------------------------------------------------------------------------------------
def brace_tokens(template: Path) -> list[str]:
    template, out = Path(template), []
    files = template_files(template) + ([MANIFEST] if (template / MANIFEST).is_file() else [])
    for rel in files:
        text = (template / rel).read_text(encoding="utf-8", errors="replace")
        out += [f"PG7 {rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group()!r} — nothing in the "
                "template is substituted" for m in BRACES_RE.finditer(text)]
    return out
