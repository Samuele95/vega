"""
js_tokens — a small tokenizer for the page template's ES modules (D4 §3.4, §17.1): the token
stream `page_lint.py` classifies (literals) and measures (PS1–PS5), and `page_template.py`
reads for the endpoints the page calls.

It knows what a regex tool cannot: strings (both quotes, escapes decoded), template literals
(their text parts cooked, each `${…}` substitution tokenized on its own, nested templates
included), comments (line and block), and regex literals, recognised where JavaScript starts
one — after an operator or a keyword, and after the `)` that closes the head of `if`, `while`,
`for` or `with` (`if (ok) /x/.test(s)`); never after a value (an identifier, a number, a
string, `]`, any other `)`). After `}` a `/` divides: the tokenizer cannot tell a block's `}`
from an object literal's, so a regex that opens a statement after a block is written
`(/x/).test(s)`; when an unterminated string, template or regex follows such a division on its
line, the TokenError says so. It does not parse: `page_lint` reads structure (functions,
classes, module level) from the brace, bracket and parenthesis tokens this stream keeps in
source order; `flat` gives them with every substitution's tokens spliced in after its template.

A token is `Token(kind, text, line, end_line, value, parts, inner)`:
  - kind: ident | keyword | punct | number | string | template | regex | comment;
  - value: a string's decoded value; a template's text parts joined by `SUB`;
  - parts / inner: a template's text parts and, per substitution, its tokens.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SUB", "Token", "TokenError", "tokenize", "tokenize_file", "significant", "flat"]

SUB = "\u241a"  # stands for a `${…}` substitution in a template's joined value: never a letter
KEYWORDS = frozenset(
    "await break case catch class const continue debugger default delete do else export extends "
    "finally for function if import in instanceof let new of return static super switch this "
    "throw try typeof var void while with yield async get set".split())
# after these a `/` starts a regex literal; after any other identifier it divides
_REGEX_AFTER_WORDS = frozenset(
    "return typeof instanceof in of new delete void throw case do else yield await".split())
_VALUE_END = frozenset(")]}")
# a `(` after these opens a statement head: a `/` after its `)` starts a regex
_CONTROL_HEADS = frozenset({"if", "while", "for", "with"})
_PUNCT = sorted(
    (">>>= ... === !== **= <<= >>= >>> &&= ||= ??= => == != <= >= && || ?? ?. ++ -- += -= *= "
     "/= %= &= |= ^= ** << >> { } ( ) [ ] ; , < > + - * / % & | ^ ! ~ ? : = . @").split(),
    key=len, reverse=True)
_NUMBER_RE = re.compile(r"0[xXoObB][0-9a-fA-F_]+n?|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?n?")
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
_HEX_RE = re.compile(r"x([0-9a-fA-F]{2})|u\{([0-9a-fA-F]+)\}|u([0-9a-fA-F]{4})")


class TokenError(ValueError):
    """The source cannot be tokenized: an unterminated string, template, comment or regex."""


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    end_line: int
    value: str | None = None
    parts: tuple = ()
    inner: tuple = ()

    def is_(self, kind: str, text: str | None = None) -> bool:
        return self.kind == kind and (text is None or self.text == text)


def _decode(raw: str) -> str:
    """The value of a string or template text part: escapes decoded, line continuations gone."""
    out, i = [], 0
    while i < len(raw):
        ch = raw[i]
        if ch != "\\" or i + 1 >= len(raw):
            out.append(ch)
            i += 1
            continue
        nxt = raw[i + 1]
        m = _HEX_RE.match(raw, i + 1)
        if m:
            out.append(chr(int(next(g for g in m.groups() if g), 16)))
            i = m.end()
        elif nxt in "\r\n\u2028\u2029":
            i += 3 if raw[i + 1:i + 3] == "\r\n" else 2
        else:
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
    return "".join(out)


def _opens_head(prev: Token | None, before: Token | None) -> bool:
    """Does a `(` after `prev` (and `before` it) open the head of if / while / for / with?
    `for await (` does; a property call (`Symbol.for(…)`) does not."""
    if prev is not None and prev.is_("keyword", "await"):
        return before is not None and before.is_("keyword", "for")
    if before is not None and before.kind == "punct" and before.text in (".", "?."):
        return False
    return prev is not None and prev.kind == "keyword" and prev.text in _CONTROL_HEADS


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch in "$_#"


def _is_ident_part(ch: str) -> bool:
    return ch.isalnum() or ch in "$_\u200c\u200d"


class _Lexer:
    """One pass over a source; `run(pos, closing)` tokenizes up to an unmatched `}` (a
    template substitution's end) or the end of the source."""

    def __init__(self, src: str, where: str) -> None:
        self.src, self.where = src, where
        self.starts = [0] + [m.end() for m in re.finditer("\n", src)]
        self.division_at = None       # the line of the last `/` read as division after `)` or `}`

    def line_at(self, pos: int) -> int:
        return bisect.bisect_right(self.starts, pos)

    def fail(self, what: str, pos: int):
        hint = ""
        if self.division_at == self.line_at(pos):
            hint = (" (a `/` after `)` or `}` on this line was read as division: js_tokens starts a "
                    "regex there only after the head of if/while/for/with; write the regex as (/…/))")
        raise TokenError(f"{self.where}:{self.line_at(pos)}: unterminated {what}{hint}")

    def run(self, pos: int, closing: bool) -> tuple[list, int]:
        out, depth, prev, before = [], 0, None, None
        heads, control = [], False    # per open `(`: a statement head?; did the last token close one
        while pos < len(self.src):
            ch = self.src[pos]
            if ch.isspace():
                pos += 1
                continue
            if closing and ch == "}" and depth == 0:
                return out, pos + 1
            tok, pos = self.one(pos, prev, control)
            depth += {"{": 1, "}": -1}.get(tok.text, 0) if tok.kind == "punct" else 0
            out.append(tok)
            if tok.kind == "comment":
                continue
            if tok.is_("punct", "("):
                heads.append(_opens_head(prev, before))
            control = tok.is_("punct", ")") and bool(heads) and heads.pop()
            prev, before = tok, prev
        if closing:
            self.fail("template substitution", pos)
        return out, pos

    def one(self, pos: int, prev: Token | None, control: bool = False) -> tuple[Token, int]:
        src, ch = self.src, self.src[pos]
        if src.startswith("//", pos) or src.startswith("/*", pos):
            return self.comment(pos)
        if ch in "'\"":
            return self.string(pos)
        if ch == "`":
            return self.template(pos)
        if ch == "/" and (control or self.regex_allowed(prev)):
            return self.regex(pos)
        if ch == "/" and prev is not None and prev.kind == "punct" and prev.text in ")}":
            self.division_at = self.line_at(pos)
        if ch.isdigit() or (ch == "." and src[pos + 1:pos + 2].isdigit()):
            m = _NUMBER_RE.match(src, pos)
            return self.make("number", pos, m.end()), m.end()
        if _is_ident_start(ch):
            end = pos + 1
            while end < len(src) and _is_ident_part(src[end]):
                end += 1
            kind = "keyword" if src[pos:end] in KEYWORDS else "ident"
            return self.make(kind, pos, end), end
        op = next((p for p in _PUNCT if src.startswith(p, pos)), ch)
        return self.make("punct", pos, pos + len(op)), pos + len(op)

    def make(self, kind: str, start: int, end: int, **extra) -> Token:
        return Token(kind, self.src[start:end], self.line_at(start), self.line_at(end - 1), **extra)

    @staticmethod
    def regex_allowed(prev: Token | None) -> bool:
        if prev is None:
            return True
        if prev.kind == "punct":
            return prev.text not in _VALUE_END
        return prev.kind == "keyword" and prev.text in _REGEX_AFTER_WORDS

    def comment(self, pos: int) -> tuple[Token, int]:
        if self.src.startswith("//", pos):
            end = self.src.find("\n", pos)
            end = len(self.src) if end < 0 else end
        else:
            end = self.src.find("*/", pos + 2)
            if end < 0:
                self.fail("comment", pos)
            end += 2
        return self.make("comment", pos, end), end

    def string(self, pos: int) -> tuple[Token, int]:
        quote, i = self.src[pos], pos + 1
        while i < len(self.src) and self.src[i] != quote:
            if self.src[i] == "\n":
                self.fail("string", pos)
            i += 2 if self.src[i] == "\\" else 1
        if i >= len(self.src):
            self.fail("string", pos)
        return self.make("string", pos, i + 1, value=_decode(self.src[pos + 1:i])), i + 1

    def template(self, pos: int) -> tuple[Token, int]:
        parts, inner, i, start = [], [], pos + 1, pos + 1
        while i < len(self.src):
            ch = self.src[i]
            if ch == "\\":
                i += 2
            elif ch == "`":
                parts.append(_decode(self.src[start:i]))
                return self.make("template", pos, i + 1, value=SUB.join(parts), parts=tuple(parts),
                                 inner=tuple(inner)), i + 1
            elif self.src.startswith("${", i):
                parts.append(_decode(self.src[start:i]))
                toks, i = self.run(i + 2, closing=True)
                inner.append(tuple(toks))
                start = i
            else:
                i += 1
        self.fail("template literal", pos)

    def regex(self, pos: int) -> tuple[Token, int]:
        i, in_class = pos + 1, False
        while i < len(self.src):
            ch = self.src[i]
            if ch == "\n":
                break
            if ch == "\\":
                i += 2
                continue
            if ch == "[" or (ch == "]" and in_class):
                in_class = ch == "["
            elif ch == "/" and not in_class:
                end = i + 1
                while end < len(self.src) and _is_ident_part(self.src[end]):
                    end += 1
                return self.make("regex", pos, end, value=self.src[pos + 1:i]), end
            i += 1
        self.fail("regex literal", pos)


def tokenize(src: str, where: str = "<source>") -> list[Token]:
    """Every token of an ES module, comments included, in source order."""
    if src.startswith("#!"):
        src = "//" + src[2:]
    return _Lexer(src, where).run(0, closing=False)[0]


def tokenize_file(path) -> list[Token]:
    path = Path(path)
    return tokenize(path.read_text(encoding="utf-8"), str(path))


def significant(tokens) -> list[Token]:
    """The tokens without the comments."""
    return [t for t in tokens if t.kind != "comment"]


def flat(tokens) -> list[Token]:
    """The significant tokens, each template's substitutions' tokens spliced in after it."""
    out = []
    for t in significant(tokens):
        out.append(t)
        for sub in t.inner:
            out.extend(flat(sub))
    return out
