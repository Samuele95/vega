"""
js_shape — the structure `page_lint.metrics` measures in one ES module (D4 §3.4, PS1–PS5): its
functions (declarations, expressions, methods, arrows — each with its brace span and parameter
count), its classes (their brace span) and its module-level state (every `let` / `var` at
module level, and every `const` whose initializer is a mutable literal or collection).

It reads `js_tokens`' stream, never a regex over the text: a brace inside a string, a template
or a regex literal is not structure, and a `${…}` substitution's functions are measured too.

- A `{` opens a **function body** after `=>`, after the parameter list of `function name(…)`,
  or after a method's parameter list (`name(…) {`, `get name() {`, `[key](…) {`); it opens a
  **class body** when the statement holds `class` before it; anything else is a block or an
  object literal. A control statement's `(…) {` (`if`, `for`, `while`, `switch`, `catch`,
  `with`) is a block.
- A function's lines are its body's, from the line of `{` to the line of `}`, both counted; a
  class's run from the `class` keyword to its closing brace. An arrow with an expression body
  runs from the line of `=>` to the line of the expression's last token: the expression ends
  at a `,` or `;` at its own bracket depth, at the bracket that closes the group holding it,
  or before a statement keyword (`const`, `export`, `if`, …) that starts a later line.
- Parameters are the commas at the top of the list plus one: a destructured object or array,
  a default value and a rest parameter each count as one.
- **Module-level state** is at bracket depth 0 of the module: `let` and `var` always; `const`
  when its initializer starts with `{`, `[` or `new Map|Set|WeakMap|WeakSet|Array|Object`
  (`Object.freeze(…)` is a constant).
"""
from __future__ import annotations

from dataclasses import dataclass

from js_tokens import significant

__all__ = ["Function", "Klass", "State", "Shape", "shape_of", "MUTABLE_CONSTRUCTORS"]

MUTABLE_CONSTRUCTORS = frozenset({"Map", "Set", "WeakMap", "WeakSet", "Array", "Object"})
_CONTROL = frozenset({"if", "for", "while", "switch", "catch", "with", "await"})   # `for await (…) {`
_BLOCK_AFTER = frozenset({"else", "try", "finally", "do"})
_STATEMENT_EDGE = frozenset({";", "{", "}"})
_STATEMENT_START = frozenset({"export", "import", "const", "let", "var", "function", "class", "if", "for",
                              "while", "do", "return", "throw", "switch", "try"})
_OPEN, _CLOSE = "([{", ")]}"


@dataclass(frozen=True)
class Function:
    name: str
    line: int
    lines: int
    params: int


@dataclass(frozen=True)
class Klass:
    name: str
    line: int
    lines: int


@dataclass(frozen=True)
class State:
    name: str
    line: int
    kind: str           # let | var | object | array | new <Constructor>


@dataclass(frozen=True)
class Shape:
    functions: tuple
    classes: tuple
    state: tuple


def _match_back(toks: list, close: int) -> int:
    """The index of the bracket that opens the one closing at `close`."""
    depth = 0
    for i in range(close, -1, -1):
        t = toks[i]
        if t.kind == "punct" and t.text in _CLOSE:
            depth += 1
        elif t.kind == "punct" and t.text in _OPEN:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _params(toks: list, open_: int, close: int) -> int:
    """Parameters between the brackets at `open_` and `close`: top-level commas plus one."""
    inside = toks[open_ + 1:close]
    if not inside:
        return 0
    depth, commas = 0, 0
    for t in inside:
        if t.kind == "punct" and t.text in _OPEN:
            depth += 1
        elif t.kind == "punct" and t.text in _CLOSE:
            depth -= 1
        elif t.is_("punct", ",") and depth == 0:
            commas += 1
    return commas + (0 if inside[-1].is_("punct", ",") else 1)


def _expression_end(toks: list, first: int) -> int:
    """The index of the last token of the expression body that starts at `first`."""
    depth, last = 0, first
    for j in range(first, len(toks)):
        t = toks[j]
        if t.kind == "punct" and t.text in _CLOSE:
            if depth == 0:
                return last
            depth -= 1
        elif t.kind == "punct" and t.text in _OPEN:
            depth += 1
        elif depth == 0 and j > first and (
                (t.kind == "punct" and t.text in (",", ";"))
                or (t.kind == "keyword" and t.text in _STATEMENT_START and t.line > toks[last].end_line)):
            return last
        last = j
    return last


def _name_before(toks: list, start: int) -> str | None:
    """The name a function expression or arrow starting at `start` is bound to: `name = …`,
    `name: …` or `this.name = …`; None when it is anonymous."""
    i = start - 1
    if i >= 0 and toks[i].is_("keyword", "async"):
        i -= 1
    if i >= 1 and toks[i].kind == "punct" and toks[i].text in ("=", ":", "||=", "??="):
        name = toks[i - 1]
        if name.kind in ("ident", "keyword", "string"):
            return name.value if name.kind == "string" else name.text
    return None


def _class_before(toks: list, i: int) -> int:
    """The index of the `class` keyword of the statement a `{` at `i` ends, or -1. A
    parenthesised or bracketed group (`extends mix(A, B)`) is skipped; a brace, a `;` or the
    start of the list ends the statement; `class` as a property (`{class: …}`, `x.class`) is no
    class."""
    j = i - 1
    while j >= 0:
        t = toks[j]
        if t.kind == "punct" and t.text in ")]":
            j = _match_back(toks, j) - 1
            continue
        if t.kind == "punct" and t.text in _STATEMENT_EDGE:
            return -1
        if t.is_("keyword", "class") and _is_class_keyword(toks, j):
            return j
        j -= 1
    return -1


def _is_class_keyword(toks: list, j: int) -> bool:
    after = toks[j + 1] if j + 1 < len(toks) else None
    before = toks[j - 1] if j >= 1 else None
    if before is not None and before.kind == "punct" and before.text in (".", "?."):
        return False
    return after is not None and (after.kind == "ident" or after.is_("keyword", "extends")
                                  or after.is_("punct", "{"))


class _Walker:
    """One pass over one token list; nested template substitutions get their own walker."""

    def __init__(self, toks: list, module_level: bool) -> None:
        self.toks, self.module_level = toks, module_level
        self.functions, self.classes, self.state = [], [], []
        self.opened = {}      # index of `{` → ("function", name, line, params) | ("class", name, line)

    def run(self) -> "_Walker":
        stack, decl = [], None
        for i, t in enumerate(self.toks):
            for sub in t.inner:
                self._absorb(_Walker(significant(sub), module_level=False).run())
            if t.is_("punct", "=>"):
                self._arrow(i)
            elif t.is_("punct", "{"):
                self._open(i)
            elif t.is_("punct", "}") and stack:
                self._close(stack[-1], i)
            if not stack and self.module_level:
                decl = self._state(i, decl)
            if t.kind == "punct" and t.text in _OPEN:
                stack.append(i)
            elif t.kind == "punct" and t.text in _CLOSE and stack:
                stack.pop()
        return self

    def _absorb(self, other: "_Walker") -> None:
        self.functions += other.functions
        self.classes += other.classes

    def _arrow(self, i: int) -> None:
        toks = self.toks
        if i >= 1 and toks[i - 1].is_("punct", ")"):
            start = _match_back(toks, i - 1)
            params = _params(toks, start, i - 1)
        else:
            start, params = i - 1, 1
        name = _name_before(toks, start) or "<arrow>"
        if i + 1 < len(toks) and toks[i + 1].is_("punct", "{"):
            self.opened[i + 1] = ("function", name, toks[i + 1].line, params)
        elif i + 1 < len(toks):
            end = toks[_expression_end(toks, i + 1)].end_line
            self.functions.append(Function(name, toks[i].line, end - toks[i].line + 1, params))

    def _open(self, i: int) -> None:
        toks = self.toks
        if i in self.opened:
            return
        at = _class_before(toks, i)
        if at >= 0:
            name = toks[at + 1].text if toks[at + 1].kind == "ident" else "<class>"
            self.opened[i] = ("class", name, toks[at].line)
            return
        prev = toks[i - 1] if i else None
        if prev is None or not prev.is_("punct", ")"):
            return
        start = _match_back(toks, i - 1)
        head = toks[start - 1] if start >= 1 else None
        if head is None or (head.kind == "keyword" and head.text in _CONTROL | _BLOCK_AFTER):
            return
        self.opened[i] = ("function", self._function_name(start), toks[i].line,
                          _params(toks, start, i - 1))

    def _function_name(self, paren: int) -> str:
        toks = self.toks
        head = toks[paren - 1]
        before = toks[paren - 2] if paren >= 2 else None
        if head.is_("keyword", "function") or head.is_("punct", "*"):
            return _name_before(toks, paren - (1 if head.is_("keyword", "function") else 2)) or "<function>"
        if head.is_("punct", "]"):
            return "[computed]"
        if before is not None and (before.is_("keyword", "function") or before.is_("punct", "*")):
            return head.text
        return head.value if head.kind == "string" else head.text

    def _close(self, open_: int, i: int) -> None:
        what = self.opened.pop(open_, None)
        if what is None:
            return
        end = self.toks[i].line
        if what[0] == "function":
            self.functions.append(Function(what[1], what[2], end - what[2] + 1, what[3]))
        else:
            self.classes.append(Klass(what[1], what[2], end - what[2] + 1))

    def _state(self, i: int, decl: str | None) -> str | None:
        """Module-level state at the depth-0 token `i`; returns the declaration kind in force."""
        toks, t = self.toks, self.toks[i]
        if t.kind == "keyword" and t.text in ("let", "var", "const"):
            decl = t.text
            self._declarator(i + 1, decl)
        elif t.is_("punct", ",") and decl is not None:
            self._declarator(i + 1, decl)
        elif t.is_("punct", ";") or (t.kind == "keyword" and t.text in ("function", "class", "import")):
            decl = None
        return decl

    def _declarator(self, j: int, decl: str) -> None:
        toks = self.toks
        if j >= len(toks):
            return
        name = toks[j].text if toks[j].kind in ("ident", "keyword") else "<pattern>"
        if decl in ("let", "var"):
            self.state.append(State(name, toks[j].line, decl))
            return
        if j + 2 >= len(toks) or not toks[j + 1].is_("punct", "="):
            return
        init = toks[j + 2]
        if init.is_("punct", "{"):
            self.state.append(State(name, toks[j].line, "object"))
        elif init.is_("punct", "["):
            self.state.append(State(name, toks[j].line, "array"))
        elif (init.is_("keyword", "new") and j + 3 < len(toks)
              and toks[j + 3].text in MUTABLE_CONSTRUCTORS):
            self.state.append(State(name, toks[j].line, f"new {toks[j + 3].text}"))


def shape_of(tokens) -> Shape:
    """The functions, classes and module-level state of one module's token stream."""
    w = _Walker(significant(tokens), module_level=True).run()
    key = lambda x: (x.line, x.name)  # noqa: E731
    return Shape(tuple(sorted(w.functions, key=key)), tuple(sorted(w.classes, key=key)),
                 tuple(sorted(w.state, key=key)))
