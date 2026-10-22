#!/usr/bin/env python3
"""
page_lint — the page template's prose gate and size discipline (D4 §3.4, §17.1 check 2,
§17.2; gates PG2 and PG9).

    python3 page_lint.py [--template DIR] [--legacy] [--json OUT] [FILE ...]

**`literals(path) -> list[Literal]`** classifies every string, template and regex literal of
one module (`js_tokens`: a `${…}` substitution's literals too). A literal is **prose** — a PG2
failure naming file:line — when it holds a run of two words of two letters or more (adjacent
words with white space between them) and not every word of it is in the page's vocabulary
(`page_words.Vocabulary`: a CSS class or token the sheets define or use, a DOM / SVG / ARIA /
event / canvas / media name, the shell's ids and classes). A module path, a key of the strings
or explain tables, a regex literal and the pattern given to `RegExp(…)` are never prose; the
strings table itself (`js/core/strings.en.js` and its fragments `js/core/strings/*.en.js`) is
where prose lives. `index.html` holds no text node with a letter or a digit (whitespace,
entities and symbols only), no visible attribute text (`title`, `alt`, `placeholder`,
`aria-label`, … — the slot's `data-t-title` / `data-t-aria` name its key instead), and no typed
text inside a `data-t` slot.

**`metrics(path) -> ModuleMetrics`** measures PS1–PS5 (`js_shape`), and
`ModuleMetrics.problems(rel)` reads them against the limits:

| PS1 | module | ≤ 400 non-blank lines | the strings table exempt (a data table, as `domain.py` is from C13) |
| PS2 | function / method | ≤ 50 lines, `{` to `}` both counted | an expression-bodied arrow: `=>` to its last token |
| PS3 | class | ≤ 200 lines, `class` to `}` | |
| PS4 | parameters | ≤ 5 (a destructured object counts as 1) | |
| PS5 | module-level state | no `let` / `var` / mutable literal at module level | `js/core/state.js`; `js/core/decode.js`'s `new WeakMap()` |

`lint_template(template)` runs PG2 over every module the template serves and over `index.html`
(D4 §17.1: every `.js` module), and PS1–PS5 over every served module but `js/legacy/**`, which
the size rules leave out while it exists (BUILD-PLAN W07: the transitional modules each area's
package replaces); `--legacy` measures it too.

Exit 0: no finding. 1: a finding. 2: the template cannot be read.
"""
from __future__ import annotations

import argparse
import html.parser
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from js_shape import shape_of
from js_tokens import TokenError, flat, tokenize_file
from page_template import STRINGS_TABLE, TEMPLATE_DIR, TemplateError, template_files
from page_words import MEDIA_WORDS, PSEUDO, TableError, Vocabulary

__all__ = ["Literal", "ModuleMetrics", "Report", "LIMITS", "LEGACY", "classify", "literals", "metrics",
           "html_problems", "is_strings_table", "modules_of", "lint_template", "main"]

LIMITS = {"PS1": 400, "PS2": 50, "PS3": 200, "PS4": 5}
LEGACY = "js/legacy/"
PS5_ALLOWED = {"js/core/state.js": None, "js/core/decode.js": frozenset({"new WeakMap"})}
WORD_RE = re.compile(r"(?<![\w-])[^\W\d_]\w*(?:-\w+)*")
MEDIA_RE = re.compile(r"\s*(?:@media\s+)?(?:(?:only|not)\s+)?(?:all|screen|print|\()")
MODULE_RE = re.compile(r"\.{0,2}/?(?:[\w.-]+/)*[\w.-]+\.(?:js|mjs|json|css)")
VISIBLE_ATTRS = frozenset({"title", "alt", "placeholder", "label", "aria-label", "aria-description",
                           "aria-roledescription", "aria-valuetext", "aria-placeholder"})


def is_strings_table(rel: str) -> bool:
    return bool(STRINGS_TABLE.match(rel))


@dataclass(frozen=True)
class Literal:
    """One literal of a module and what it is: table | regex | module | key | blank | word |
    vocabulary | prose (only prose is a failure)."""

    file: str
    line: int
    kind: str
    text: str
    category: str
    unknown: tuple = ()

    @property
    def prose(self) -> bool:
        return self.category == "prose"

    def __str__(self) -> str:
        shown = self.text if len(self.text) <= 80 else self.text[:77] + "..."
        return (f"PG2 {self.file}:{self.line}: prose literal {shown!r} outside the strings table "
                f"(not page vocabulary: {', '.join(self.unknown)})")


def _letters(word: str) -> int:
    return sum(ch.isalpha() for ch in word)


def _has_run(words: list) -> bool:
    """Two adjacent words of two letters or more with white space between them."""
    return any(_letters(a.group()) >= 2 and _letters(b.group()) >= 2
               and any(ch.isspace() for ch in a.string[a.end():b.start()])
               for a, b in zip(words, words[1:]))


def classify(text: str, vocab: Vocabulary) -> tuple[str, tuple]:
    """(category, unknown words) of a literal's text outside the strings table."""
    if MODULE_RE.fullmatch(text):
        return "module", ()
    if text in vocab.keys:
        return "key", ()
    words = list(WORD_RE.finditer(text))
    if not any(_letters(w.group()) >= 2 for w in words):
        return "blank", ()
    if not _has_run(words):
        return "word", ()
    media = bool(MEDIA_RE.match(text))
    unknown = tuple(dict.fromkeys(w.group() for w in words if not _allowed(w, vocab, media)))
    return ("prose", unknown) if unknown else ("vocabulary", ())


def _allowed(word: re.Match, vocab: Vocabulary, media: bool) -> bool:
    """A word of the vocabulary; a pseudo-class right after `:`; a connective of a media query."""
    w = word.group()
    if vocab.allows(w):
        return True
    if word.string[max(0, word.start() - 1):word.start()] == ":" and w.lower() in PSEUDO:
        return True
    return media and w.lower() in MEDIA_WORDS


def _is_regexp_argument(toks: list, i: int) -> bool:
    return i >= 2 and toks[i - 1].is_("punct", "(") and toks[i - 2].is_("ident", "RegExp")


def literals(path, vocab: Vocabulary | None = None, rel: str | None = None) -> list[Literal]:
    """Every literal of one module, classified (the template's vocabulary unless given)."""
    path = Path(path)
    rel = rel or _rel(path)
    vocab = vocab or Vocabulary.of(_template_of(path))
    toks = flat(tokenize_file(path))
    out = []
    for i, t in enumerate(toks):
        if t.kind not in ("string", "template", "regex"):
            continue
        if is_strings_table(rel):
            category, unknown = "table", ()
        elif t.kind == "regex" or _is_regexp_argument(toks, i):
            category, unknown = "regex", ()
        else:
            category, unknown = classify(t.value, vocab)
        out.append(Literal(rel, t.line, t.kind, t.value, category, unknown))
    return out


def _template_of(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / "index.html").is_file() and (parent / "page.css").is_file():
            return parent
    return TEMPLATE_DIR


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_template_of(path)).as_posix()
    except ValueError:
        return path.name


# ---- index.html --------------------------------------------------------------------------------------
def _textual(text: str) -> bool:
    return any(unicodedata.category(ch)[0] in "LN" for ch in text)


class _ShellCheck(html.parser.HTMLParser):
    RAW = frozenset({"script", "style"})
    VOID = frozenset("area base br col embed hr img input link meta source track wbr".split())

    def __init__(self, rel: str) -> None:
        super().__init__(convert_charrefs=True)
        self.rel, self.problems, self.keys = rel, [], []
        self.slots, self.raw = [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        line = self.getpos()[0]
        for name in sorted(VISIBLE_ATTRS & set(a)):
            if _textual(a[name] or ""):
                self.problems.append(f"PG2 {self.rel}:{line}: <{tag} {name}=…> types text; "
                                     f"name its key with data-t-{'aria' if name.startswith('aria') else 'title'}")
        self.keys += [(line, a[k]) for k in ("data-t", "data-t-title", "data-t-aria") if a.get(k)]
        if "data-t" in a and tag not in self.VOID:
            self.slots.append(tag)
        if tag in self.RAW:
            self.raw = tag

    def handle_endtag(self, tag):
        if self.slots and self.slots[-1] == tag:
            self.slots.pop()
        if tag == self.raw:
            self.raw = None

    def handle_data(self, data):
        if self.raw or not data.strip():
            return
        line = self.getpos()[0]
        if self.slots:
            self.problems.append(f"PG2 {self.rel}:{line}: text {data.strip()[:40]!r} typed inside a data-t slot")
        elif _textual(data):
            self.problems.append(f"PG2 {self.rel}:{line}: text node {data.strip()[:60]!r} — the shell holds "
                                 "no prose: a data-t slot names its key")


def html_problems(path, rel: str = "index.html") -> tuple[list[str], list[tuple[int, str]]]:
    """(PG2 problems, [(line, key)] of the data-t / data-t-title / data-t-aria slots) of the shell."""
    check = _ShellCheck(rel)
    check.feed(Path(path).read_text(encoding="utf-8"))
    check.close()
    return check.problems, check.keys


# ---- PS1–PS5 ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModuleMetrics:
    file: str
    lines: int                     # non-blank
    functions: tuple = ()
    classes: tuple = ()
    state: tuple = ()

    def problems(self, rel: str | None = None) -> list[str]:
        rel = rel or self.file
        out = []
        if self.lines > LIMITS["PS1"] and not is_strings_table(rel):
            out.append(f"PS1 {rel}:1 module — {self.lines} non-blank lines > {LIMITS['PS1']}")
        out += [f"PS2 {rel}:{f.line} {f.name} — {f.lines} lines > {LIMITS['PS2']}"
                for f in self.functions if f.lines > LIMITS["PS2"]]
        out += [f"PS3 {rel}:{c.line} class {c.name} — {c.lines} lines > {LIMITS['PS3']}"
                for c in self.classes if c.lines > LIMITS["PS3"]]
        out += [f"PS4 {rel}:{f.line} {f.name} — {f.params} parameters > {LIMITS['PS4']}"
                for f in self.functions if f.params > LIMITS["PS4"]]
        allowed = PS5_ALLOWED.get(rel, frozenset())
        out += [f"PS5 {rel}:{s.line} {s.name} — module-level {s.kind}: mutable state lives in js/core/state.js"
                for s in self.state if allowed is not None and s.kind not in allowed]
        return out


def metrics(path, rel: str | None = None) -> ModuleMetrics:
    """PS1–PS5's measures of one module."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    shape = shape_of(tokenize_file(path))
    return ModuleMetrics(rel or _rel(path), sum(1 for line in text.splitlines() if line.strip()),
                         shape.functions, shape.classes, shape.state)


# ---- the template ------------------------------------------------------------------------------------------
@dataclass
class Report:
    problems: list = field(default_factory=list)
    modules: int = 0
    literals: int = 0

    def as_json(self) -> dict:
        return {"schema": "vega/page-lint@1", "modules": self.modules, "literals": self.literals,
                "problems": self.problems}


def modules_of(template: Path, legacy: bool = False) -> list[str]:
    """The served modules PS1–PS5 measure (js/legacy/** only with legacy=True); PG2 reads
    modules_of(template, legacy=True), every one."""
    return [rel for rel in template_files(template)
            if rel.endswith(".js") and (legacy or not rel.startswith(LEGACY))]


def lint_template(template: Path = TEMPLATE_DIR, legacy: bool = False) -> Report:
    """PG2 over every served module and index.html; PG9 (PS1–PS5) over every served module
    outside js/legacy/** (inside it too with legacy=True)."""
    template = Path(template)
    vocab, report = Vocabulary.of(template), Report()
    for rel in modules_of(template, legacy=True):
        found = literals(template / rel, vocab, rel)
        report.literals += len(found)
        report.problems += [str(x) for x in found if x.prose]
        if legacy or not rel.startswith(LEGACY):
            report.problems += metrics(template / rel, rel).problems(rel)
        report.modules += 1
    report.problems += html_problems(template / "index.html")[0]
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("files", nargs="*", type=Path, help="modules to read (default: the template's)")
    ap.add_argument("--template", type=Path, default=TEMPLATE_DIR)
    ap.add_argument("--legacy", action="store_true", help="measure js/legacy/** too (PS1–PS5)")
    ap.add_argument("--json", default=None, help="write the vega/page-lint@1 report here ('-': stdout)")
    a = ap.parse_args(argv)
    try:
        report = _files(a.files, a.template) if a.files else lint_template(a.template, a.legacy)
    except (OSError, TokenError, TableError, TemplateError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if a.json:
        text = json.dumps(report.as_json(), indent=1) + "\n"
        if a.json == "-":
            sys.stdout.write(text)
            return 1 if report.problems else 0
        Path(a.json).write_text(text, encoding="utf-8")
    for line in report.problems:
        print(f"ERROR: {line}")
    print(f"{'FAIL' if report.problems else 'OK'}: {len(report.problems)} finding(s) in {report.modules} "
          f"module(s), {report.literals} literal(s) read")
    return 1 if report.problems else 0


def _files(files: list, template: Path) -> Report:
    vocab, report = Vocabulary.of(template), Report()
    for path in files:
        rel = _rel(path)
        found = literals(path, vocab, rel)
        report.literals += len(found)
        report.problems += [str(x) for x in found if x.prose] + metrics(path, rel).problems(rel)
        report.modules += 1
    return report


if __name__ == "__main__":
    sys.exit(main())
