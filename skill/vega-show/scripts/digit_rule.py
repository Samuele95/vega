"""
digit_rule — the digit rule and the placeholder grammar, one module for every text check of
`/vega-show`: the resolver (V2, V3), the page lint (PG5), the explain catalog's prose and the
layer's readings (LY3).

**The digit rule** (contract §3.3; build plan R-E10). A number the page shows comes from a
fact, as a `{key}` placeholder, never from typed text. For every authored string: delete every
placeholder, then tokenise the rest with the Unicode word pattern `[^\\W_]+(?:[-'][^\\W_]+)*`.
A token that contains a decimal digit (`str.isdecimal` on any character) is

- **refused** when it has no letter: a bare number (`256`, `0.82`, `٣`, `３`);
- **refused** when it is not a name the records carry, compared case-insensitively with the
  tokens of `facts.names` (`top7`; `α2` unless the names carry it);
- **accepted** otherwise (`IDS2017`, `F1`, `R7`, `B1`).

A hyphen- or apostrophe-joined token (`IDS2017-style`) is also accepted when every part that
holds a digit is itself such a name and has a letter; a part that is a bare number refuses it
(`top-5`). Any Unicode `Nd` character outside a token is refused. Every refusal is `V2` and
its message starts `digit rule:`, so the resolver's line names the rule whatever the token
(`ERROR: V2 authored.headline: digit rule: a bare number "12" — …`). The rule is Unicode-aware on
purpose: v2.2's `(?<![A-Za-z\\d])\\d` read a digit after `β`, `é` or `Ж` as a number, and a
full-width or Arabic-Indic digit as nothing at all. Subscript and superscript digits (`w₁`,
`σ²`) are not decimal digits and pass: they are notation, not numbers.

**Number words** (`one` … `billion`, `half`, `twice`, `percent`, the catalog's list) are
`WARN:`, never refused: `number word "<w>" in <where> — prefer a count fact (<key>)`, naming
the facts whose integer value spells the word (none named when no fact does).

**Placeholders** (`PLACEHOLDER_RE`, D4's grammar, the catalog's `placeholder.pattern`):
`{name}` or `{lex.path}`, optionally with `:int|num|pct|signed|count`. A bare name is a fact
key in story text (a runtime parameter in the page's strings); a dotted name is a lexicon path
(`{row.one}`, `{source.the}`, `{target.many}`). `check_placeholders` refuses (V3) an unknown
fact key, an unknown lexicon key or path, a format suffix on a lexicon path, and every other
brace group without a space or a nested brace that the grammar does not accept (`{Units}`,
`{x:float}`, `{x.y.z}`, `{a:int:pct}`, `{éx}`), so no literal brace reaches the page. A brace
group with a space (`{rows, at}`) or an empty one (`{}`) is text, not a placeholder.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping

from vega_show_core import CATALOG, FactSet, Problem

__all__ = ["PLACEHOLDER_RE", "TOKEN_RE", "Placeholder", "DigitRule", "placeholders",
           "strip_placeholders", "check_placeholders", "check_prose"]

PLACEHOLDER_RE = re.compile(CATALOG["placeholder"]["pattern"])
TOKEN_RE = re.compile(CATALOG["digit_rule"]["token_pattern"])
# a brace group shaped like a placeholder: no space, no nested brace, whatever the grammar says
_LOOSE_RE = re.compile(r"\{[^{}\s]+\}")
_RULE_DIGITS, _RULE_PLACEHOLDERS = "V2", "V3"
_DIGIT = "digit rule: "  # every V2 refusal names its rule, so a resolver's output says "digit"
_NUMBER_WORD = "(?:{})".format("|".join(map(re.escape, sorted(CATALOG["number_words"], key=len,
                                                                reverse=True))))
# a number word, or a run of them joined by a hyphen or a space ("twenty-six", "one hundred")
_WORD_RE = re.compile(rf"(?<![^\W_])({_NUMBER_WORD}(?:[- ]{_NUMBER_WORD})*)(?![^\W_])", re.IGNORECASE)
_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_BIG = {100: "hundred", 1000: "thousand", 10 ** 6: "million", 10 ** 9: "billion"}


@dataclass(frozen=True)
class Placeholder:
    """One `{name}`, `{name:fmt}` or `{key.path}` occurrence in a text."""

    text: str
    name: str
    path: str | None = None
    suffix: str | None = None

    @property
    def is_lexicon_path(self) -> bool:
        return self.path is not None


def placeholders(text: str) -> tuple[Placeholder, ...]:
    return tuple(Placeholder(m.group(0), m.group(1), m.group(2), m.group(3))
                 for m in PLACEHOLDER_RE.finditer(text or ""))


def strip_placeholders(text: str) -> str:
    """The text with every placeholder deleted (a space kept, so words do not fuse)."""
    return PLACEHOLDER_RE.sub(" ", text or "")


class DigitRule:
    """The digit rule over one whitelist of names, with the number-word lint beside it."""

    def __init__(self, names: Iterable[str] = (),
                 spelled: Mapping[str, Iterable[str]] | None = None) -> None:
        self._names = frozenset(_name_tokens(names))
        self._spelled = {w: tuple(keys) for w, keys in (spelled or {}).items()}

    @classmethod
    def from_catalog(cls) -> "DigitRule":
        """The rule for text that has no fact set (the page's strings, the explain catalog):
        only the catalog's name tokens are names."""
        return cls(CATALOG["name_tokens"])

    @classmethod
    def for_facts(cls, facts: FactSet) -> "DigitRule":
        """The rule for a lane's authored text: `facts.names` is the whitelist, and the
        integer facts spell the number words the lint suggests replacing."""
        spelled: dict[str, list[str]] = {}
        for fact in sorted(facts.facts, key=lambda f: f.key):
            word = _spell(fact.value)
            if word:
                spelled.setdefault(word, []).append(fact.key)
        return cls(facts.names, spelled)

    def is_name(self, token: str) -> bool:
        return token.casefold() in self._names

    def refusals(self, text: str, where: str) -> list[Problem]:
        """V2 refusals of one text: every digit that is not inside a whitelisted name."""
        bare, out, covered = strip_placeholders(text), [], set()
        for m in TOKEN_RE.finditer(bare):
            covered.update(range(m.start(), m.end()))
            why = self._token_problem(m.group(0))
            if why:
                out.append(Problem(_RULE_DIGITS, where, why))
        stray = "".join(ch for i, ch in enumerate(bare)
                        if i not in covered and unicodedata.category(ch) == "Nd")
        if stray:
            out.append(Problem(_RULE_DIGITS, where, f"{_DIGIT}a digit outside any word ({stray!r}) "
                                                    "— numbers come from the facts as {key} placeholders"))
        return out

    def warnings(self, text: str, where: str) -> list[Problem]:
        """The number-word lint: WARN, never a refusal."""
        out = []
        for m in _WORD_RE.finditer(strip_placeholders(text)):
            keys = self._spelled.get(m.group(1).lower())
            hint = f" ({', '.join(keys)})" if keys else ""
            out.append(Problem(_RULE_DIGITS, where, f'number word "{m.group(1)}" in {where} — '
                                                    f"prefer a count fact{hint}", level="WARN"))
        return out

    def check(self, text: str, where: str) -> list[Problem]:
        return self.refusals(text, where) + self.warnings(text, where)

    def _token_problem(self, token: str) -> str | None:
        if not any(ch.isdecimal() for ch in token):
            return None
        if not any(ch.isalpha() for ch in token):
            return (f'{_DIGIT}a bare number "{token}" — numbers come from the facts as {{key}} '
                    "placeholders")
        if self.is_name(token):
            return None
        parts = [p for p in re.split(r"[-']", token) if any(ch.isdecimal() for ch in p)]
        if all(any(ch.isalpha() for ch in p) and self.is_name(p) for p in parts):
            return None
        return (f'{_DIGIT}"{token}" holds a digit and is not a name the records carry — a number '
                "comes from a {key} placeholder, a name from facts.names")


def check_placeholders(text: str, where: str, known: Iterable[str] | None = None,
                       lexicon_keys: Iterable[str] | None = None) -> list[Problem]:
    """V3 over one text. `known` is the set of bare names allowed (fact keys, or a string's
    runtime parameters); None accepts any bare name. `lexicon_keys` defaults to the catalog's."""
    known = None if known is None else frozenset(known)
    lex = frozenset(CATALOG["lexicon"]["keys"] if lexicon_keys is None else lexicon_keys)
    out = []
    for m in _LOOSE_RE.finditer(text or ""):
        why = _placeholder_problem(m.group(0), known, lex)
        if why:
            out.append(Problem(_RULE_PLACEHOLDERS, where, why))
    return out


def check_prose(text: str, known: Iterable[str] | None = None,
                lexicon_keys: Iterable[str] | None = None, names: Iterable[str] | None = None,
                where: str = "text") -> list[str]:
    """D4's one-call check for keyed prose (the page's strings, the explain catalog): the digit
    rule (names default to the catalog's name tokens) and the placeholder grammar, as ERROR
    lines. Number words are not reported here; `DigitRule.warnings` gives them."""
    rule = DigitRule.from_catalog() if names is None else DigitRule(names)
    problems = rule.refusals(text, where) + check_placeholders(text, where, known, lexicon_keys)
    return [str(p) for p in problems]


def _placeholder_problem(whole: str, known: frozenset | None, lex: frozenset) -> str | None:
    """Why one brace group is not an acceptable placeholder; None when it is."""
    m = PLACEHOLDER_RE.fullmatch(whole)
    if m is None:
        return (f"{whole} is not a placeholder of the grammar {{name}}, {{key.path}} with an "
                f"optional :{'|:'.join(CATALOG['placeholder']['suffixes'])}")
    name, path, suffix = m.group(1), m.group(2), m.group(3)
    if path is None:
        if known is not None and name not in known:
            return f"{whole}: {name!r} is not a known fact key"
        return None
    if name not in lex:
        return f"{whole}: {name!r} is not a lexicon key ({', '.join(sorted(lex))})"
    if path not in CATALOG["lexicon"]["paths"]:
        return f"{whole}: {path!r} is not a lexicon path ({', '.join(CATALOG['lexicon']['paths'])})"
    if suffix is not None:
        return f"{whole}: a lexicon path is a word and takes no format suffix"
    return None


def _name_tokens(names: Iterable[str]) -> set[str]:
    """Every token of every name, case-folded, whole and split at its joiners."""
    out = set()
    for name in names:
        for m in TOKEN_RE.finditer(str(name)):
            token = m.group(0).casefold()
            out.add(token)
            out.update(p for p in re.split(r"[-']", token) if p)
    return out


def _spell(value) -> str | None:
    """The English words for a whole number the lint can suggest: below a hundred, or fewer
    than twenty round hundreds, thousands, millions or billions; None for anything else."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    if value < 20:
        return _UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        return _TENS[tens] + (f"-{_UNITS[unit]}" if unit else "")
    for size, word in sorted(_BIG.items(), reverse=True):
        count, rest = divmod(value, size)
        if count and not rest and count < 20:
            return f"{_UNITS[count]} {word}"
    return None
