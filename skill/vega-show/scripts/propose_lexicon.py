"""
propose_lexicon — `LexiconProposer`: the lexicon a draft proposes (`vega/lexicon@1`, contract
§3.6; D4 §11.2's defaults, which R-E9 makes the draft's).

The lexicon is the page's only source of domain words: each key is `{one, many, article}`. The
draft proposes every key the lane's shape requires (`catalog.lexicon.required`: `row`,
`feature`, `source`; a window adds `step`, `window`; a graph adds `node`, `edge`, and `graph` at
graph level), each with the source it came from:

- **`row`** from `MANIFEST.serve.identity.noun` / `.nouns` (pack's row noun; source
  `bundle:MANIFEST.serve.identity.noun`) when both are there and are lexicon values (lower
  case, no digit); otherwise the default, with a `WARN:` naming what the identity lacks;
- **every other key** from `catalog.lexicon.defaults` (source `vega:default`: `input/inputs`,
  `source data`, `step`, `window`, `node`, `edge`, `graph`);
- **a `show: lexicon <key> <one>/<many> (user)` stance** of this dataset replaces the entry of
  its key (source `show:<date>`), the latest stance of a key winning; a stance whose key the
  lexicon does not have or whose words are not lexicon values is a `WARN:` and not applied;
  `target`, the one optional key, is proposed only from a stance (its default is the page's
  strings, `target.<view key>`).

The article follows the first letter (`a` before a consonant, `an` before a vowel); `source`
takes `the`, the one key whose article may be `the`. The model rewrites entries with domain
words under the same rules (`V5`); the title and description are not lexicon entries (they
reach `page.json.identity` from `MANIFEST.serve.identity`).
"""
from __future__ import annotations

import re
from typing import Mapping

from memory_stances import USER_SUFFIX, MemoryStances
from vega_show_core import CATALOG, TaskShape

__all__ = ["LexiconProposer", "shape_key", "article_of"]

_IDENTITY_SOURCE = "bundle:MANIFEST.serve.identity.noun"
_DEFAULT_SOURCE = "vega:default"
_VOWELS = "aeiou"


def shape_key(task: TaskShape) -> str:
    """The key of `catalog.lexicon.required` for a task: `row`, `window`, `graph:<level>`."""
    if task.sample_shape == "graph":
        return f"graph:{task.graph_level or 'node'}"
    return task.sample_shape


def article_of(key: str, one: str) -> str:
    """The article a proposed entry takes: `the` for `source`, else the first-letter rule."""
    if key == "source":
        return "the"
    return "an" if one[:1] in _VOWELS else "a"


class LexiconProposer:
    """The lexicon of one lane: required keys from the identity, the defaults and the stances."""

    def __init__(self, task: TaskShape, identity: Mapping | None, stances: MemoryStances,
                 catalog: Mapping = CATALOG) -> None:
        self._task, self._identity = task, dict(identity or {})
        self._stances, self._spec = stances, catalog["lexicon"]
        self._pattern = re.compile(self._spec["value_pattern"])
        self._markers: list[str] = []

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def propose(self) -> dict:
        """`{key: {one, many, article, source}}` for every required key (and a stanced
        `target`), in the catalog's key order."""
        self._markers = []
        entries = {key: self._default(key) for key in self._spec["required"][shape_key(self._task)]}
        entries["row"] = self._row() or entries["row"]
        for stance in self._stances.of_kind("lexicon"):
            self._apply(entries, stance)
        return {k: entries[k] for k in self._spec["keys"] if k in entries}

    def _default(self, key: str) -> dict:
        entry = self._spec["defaults"][key]
        return {"one": entry["one"], "many": entry["many"], "article": entry["article"],
                "source": _DEFAULT_SOURCE}

    def _row(self) -> dict | None:
        noun, nouns = (_word(self._identity.get(k)) for k in ("noun", "nouns"))
        if noun is None and nouns is None:
            self._markers.append("WARN: MANIFEST.serve.identity names no row noun; the row entry "
                                 "is the default (row/rows) — the model writes the domain's noun")
            return None
        if noun is None or nouns is None:
            missing = "plural" if nouns is None else "singular"
            self._markers.append(f"WARN: MANIFEST.serve.identity names the row noun "
                                 f"{noun or nouns!r} without its {missing}; the row entry is the "
                                 "default (row/rows) — the model writes both")
            return None
        if not (self._pattern.fullmatch(noun) and self._pattern.fullmatch(nouns)):
            self._markers.append(f"WARN: MANIFEST.serve.identity's row noun {noun!r} / "
                                 f"{nouns!r} is not a pair of lexicon values "
                                 f"({self._spec['value_pattern']}); the row entry is the default")
            return None
        return {"one": noun, "many": nouns, "article": article_of("row", noun),
                "source": _IDENTITY_SOURCE}

    def _apply(self, entries: dict, stance) -> None:
        key, _, words = USER_SUFFIX.sub("", stance.body).partition(" ")
        one, sep, many = words.partition("/")
        one, many = one.strip(), many.strip()
        allowed = key in entries or key in self._spec["optional"]
        if not (allowed and sep and self._pattern.fullmatch(one) and self._pattern.fullmatch(many)):
            self._markers.append(f"WARN: the stance {stance.line!r} is not a lexicon entry of this "
                                 f"lane (<key> <one>/<many>, a key of {sorted(entries)} or "
                                 f"{self._spec['optional']}, lexicon values); not applied")
            return
        entries[key] = {"one": one, "many": many, "article": article_of(key, one),
                        "source": stance.basis}


def _word(value) -> str | None:
    return value.strip().lower() if isinstance(value, str) and value.strip() else None
