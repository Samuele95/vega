"""
plan_texts — the authored strings of a story plan (`vega/story-plan@2`, contract §3.2), and the
four rules of `plan_rules.PlanValidator` that read them as text (contract §3.3, §3.8):

- **V2** the digit rule (`digit_rule.DigitRule.for_facts`: `facts.names` is the whitelist) over
  every authored string; the number-word lint beside it is `WARN:`, never a refusal;
- **V3** every placeholder is a fact key, or a lexicon path of a key the plan's lexicon holds
  (`target` always: its default is the page's strings; a key the lane's shape requires always:
  its absence is V5's), with a known suffix; and every fact the
  plan names outside its text is one: a context fact, a results note's key (a proposed results
  fact, the only ones the evaluation computed);
- **V4** no record path in authored text: a `records/` path, a `*.json` file name, a `#`-ref, a
  `sha256:` digest, a `report:` ref, or a v1-style `<source>:<path>` ref
  (`evaluation:metrics.models…`) — a number the text cites is a `{fact key}` placeholder;
- **V12** lengths: each text within its `authored_lengths` bound, and non-empty wherever the
  story's schema requires a word (every text but a caption and a group's text); and the
  context's fact count within the story schema's bounds (2–6).

`shape_problems(authored)` is V13's other half, read from `story-plan-2.schema.json` itself: a
field the schema requires missing (a choice's `title`, a limit's `text`), or a container that is
not the object or list the schema says (`results.notes` a list) — what the resolver indexes.

`authored_texts(authored)` is the walk: one `AuthoredText(where, text, bound)` per authored
string, `where` the plan path the refusal names (`authored.network.choices[2].title`) and
`bound` its `authored_lengths` key. A slot that is not the container its schema says is skipped
here (V13 names it); a nullable text left null (`results.breakdown_caption`, `monitor_note`) is
not a text to check. Every refusal is a `vega_show_core.Problem`, `ERROR: V<n> <where>: <why>`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

from digit_rule import DigitRule, check_placeholders
from propose_lexicon import shape_key
from vega_show_core import CATALOG, HERE, FactSet, Problem

__all__ = ["AuthoredText", "authored_texts", "shape_problems", "TextRules", "RECORD_PATHS",
           "STORY_SCHEMA"]

STORY_SCHEMA = json.loads((HERE.parent / CATALOG["schemas"]["vega/story@2"]["file"])
                          .read_text(encoding="utf-8"))
_CONTEXT_FACTS = STORY_SCHEMA["$defs"]["section_context"]["properties"]["facts"]
_PLAN_AUTHORED = json.loads((HERE.parent / CATALOG["schemas"]["vega/story-plan@2"]["file"])
                            .read_text(encoding="utf-8"))["properties"]["authored"]["properties"]
_CONTAINERS = {"object": Mapping, "array": list}
# the texts the story's schema lets be empty (a caption, a group's text); every other says a word
_MAY_BE_EMPTY = frozenset({"caption", "group_text"})
_SOURCES = "netspec|schema|manifest|geometry|analysis|shape|architecture|training|evaluation|bundle"
RECORD_PATHS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"records/", r"[\w-]+\.json\b", r"sha256:", r"\breport:[A-Za-z]",
    rf"\b(?:{_SOURCES}):[A-Za-z_\[]", r"#[A-Za-z_][\w-]*(?:\[[^\]\s]*\])?\."))
_PATH_WORDS = ("a record path", "a number the text cites is a {fact key} placeholder, never a ref "
               "into the records")


@dataclass(frozen=True)
class AuthoredText:
    """One authored string: where the plan holds it, the value there, its length bound's key."""

    where: str
    text: object
    bound: str


def authored_texts(authored: Mapping) -> list[AuthoredText]:
    """Every authored string of the plan, in slot order (module docstring)."""
    authored = authored if isinstance(authored, Mapping) else {}
    out = [AuthoredText(f"authored.{k}", authored[k], k) for k in ("headline", "lead")
           if k in authored and authored[k] is not None]
    for slot, walk in (("context", _context), ("network", _network),
                       ("experiments", _experiments), ("results", _results),
                       ("input_groups", _groups), ("code_notes", _keyed_bound("code_note"))):
        value = authored.get(slot)
        if isinstance(value, (Mapping, list)):
            out += list(walk(f"authored.{slot}", value))
    if isinstance(authored.get("monitor_note"), str):
        out.append(AuthoredText("authored.monitor_note", authored["monitor_note"], "monitor_note"))
    return out


def _fields(where: str, obj, pairs) -> Iterator[AuthoredText]:
    if isinstance(obj, Mapping):
        for field, bound in pairs:
            if field in obj:
                yield AuthoredText(f"{where}.{field}", obj[field], bound)


def _items(where: str, items, pairs) -> Iterator[AuthoredText]:
    for i, item in enumerate(items if isinstance(items, list) else ()):
        yield from _fields(f"{where}[{i}]", item, pairs)


def _keyed_bound(bound: str):
    def walk(where: str, items) -> Iterator[AuthoredText]:
        for key, text in (items.items() if isinstance(items, Mapping) else ()):
            yield AuthoredText(f"{where}.{key}", text, bound)
    return walk


def _context(where: str, ctx) -> Iterator[AuthoredText]:
    if isinstance(ctx, Mapping):
        yield from _keyed_bound("context_step")(f"{where}.steps", ctx.get("steps"))
        yield from _items(f"{where}.facts", ctx.get("facts"), (("label", "fact_label"),))


def _network(where: str, net) -> Iterator[AuthoredText]:
    yield from _fields(where, net, (("lead", "lead"),))
    if isinstance(net, Mapping):
        yield from _items(f"{where}.choices", net.get("choices"),
                          (("title", "choice_title"), ("text", "choice_text")))


def _experiments(where: str, exp) -> Iterator[AuthoredText]:
    yield from _fields(where, exp, (("lead", "lead"), ("caption", "caption")))
    if isinstance(exp, Mapping):
        yield from _keyed_bound("run_label")(f"{where}.labels", exp.get("labels"))


def _results(where: str, res) -> Iterator[AuthoredText]:
    yield from _fields(where, res, (("headline", "headline"),))
    if not isinstance(res, Mapping):
        return
    yield from _keyed_bound("note")(f"{where}.notes", res.get("notes"))
    if res.get("breakdown_caption") is not None:
        yield AuthoredText(f"{where}.breakdown_caption", res["breakdown_caption"], "caption")
    yield from _items(f"{where}.limits", res.get("limits"),
                      (("title", "limit_title"), ("text", "limit_text")))


def _groups(where: str, groups) -> Iterator[AuthoredText]:
    yield from _items(where, groups, (("label", "group_label"), ("text", "group_text")))


def shape_problems(authored: Mapping) -> list[Problem]:
    """V13: every field the plan's schema requires inside a set slot is there, and every
    container is the object or list the schema says."""
    out = []
    for slot, node in _PLAN_AUTHORED.items():
        if isinstance(authored, Mapping) and authored.get(slot) is not None:
            out += _shape(node, authored[slot], f"authored.{slot}")
    return out


def _shape(node: Mapping, value, where: str) -> list[Problem]:
    alternatives = [n for n in node.get("oneOf", ()) if n.get("type") != "null"]
    node = alternatives[0] if alternatives else node
    kind = node.get("type") if isinstance(node.get("type"), str) else None
    if kind in _CONTAINERS and not isinstance(value, _CONTAINERS[kind]):
        return [Problem("V13", where, f"is not the {kind} the plan's schema says")]
    if kind == "array":
        return [p for i, item in enumerate(value) if "items" in node
                for p in _shape(node["items"], item, f"{where}[{i}]")]
    if kind != "object":
        return []
    out = [Problem("V13", f"{where}.{field}", "a required field is missing")
           for field in node.get("required", ()) if field not in value]
    for field, sub in (node.get("properties") or {}).items():
        if value.get(field) is not None:
            out += _shape(sub, value[field], f"{where}.{field}")
    return out


class TextRules:
    """V2, V3, V4 and V12 over the authored strings of one plan, against one fact set."""

    def __init__(self, facts: FactSet, catalog: Mapping = CATALOG) -> None:
        self._facts, self._cat = facts, catalog
        self._rule = DigitRule.for_facts(facts)
        self._keys = facts.keys()

    def digits(self, texts: Iterable[AuthoredText]) -> list[Problem]:
        """V2: the digit rule's refusals, then the number-word lint's `WARN:` lines."""
        out = []
        for item in _strings(texts):
            out += self._rule.refusals(item.text, item.where)
            out += self._rule.warnings(item.text, item.where)
        return out

    def placeholders(self, texts: Iterable[AuthoredText], authored: Mapping,
                     results_facts: Iterable[str]) -> list[Problem]:
        """V3: placeholders, context facts and results-note keys name facts."""
        lexicon, spec = authored.get("lexicon"), self._cat["lexicon"]
        keys = set(lexicon) if isinstance(lexicon, Mapping) else set(spec["keys"])
        keys |= {*spec["optional"], *spec["required"][shape_key(self._facts.task)]}
        out = [Problem("V3", p.where, f"placeholder {p.message}")
               for item in _strings(texts)
               for p in check_placeholders(item.text, item.where, self._keys, keys)]
        ctx = authored.get("context") if isinstance(authored.get("context"), Mapping) else {}
        for i, fact in enumerate(_list(ctx.get("facts"))):
            key = fact.get("fact") if isinstance(fact, Mapping) else None
            if key not in self._keys:
                out.append(Problem("V3", f"authored.context.facts[{i}].fact",
                                   f"{key!r} is not a fact key of this fact set"))
        res = authored.get("results") if isinstance(authored.get("results"), Mapping) else {}
        proposed = set(results_facts)
        notes = res.get("notes") if isinstance(res.get("notes"), Mapping) else {}
        out += [Problem("V3", f"authored.results.notes.{key}", f"a note on {key!r}, which is not "
                        f"a proposed results fact ({', '.join(sorted(proposed)) or 'none'})")
                for key in notes if key not in proposed]
        return out

    def record_paths(self, texts: Iterable[AuthoredText]) -> list[Problem]:
        """V4: no record path, digest or ref in authored text."""
        out = []
        for item in _strings(texts):
            found = next((m.group(0) for p in RECORD_PATHS if (m := p.search(item.text))), None)
            if found is not None:
                out.append(Problem("V4", item.where, f"{_PATH_WORDS[0]} ({found!r}) in authored "
                                                     f"text — {_PATH_WORDS[1]}"))
        return out

    def lengths(self, texts: Iterable[AuthoredText], authored: Mapping) -> list[Problem]:
        """V12: every text within its bound (and non-empty where the story needs a word), and
        the context's fact count within the story schema's bounds."""
        bounds, out = self._cat["authored_lengths"], []
        for item in _strings(texts):
            size, bound = len(item.text), bounds[item.bound]
            if size > bound:
                out.append(Problem("V12", item.where, f"length: {size} characters, above the "
                                                      f"{item.bound} bound of {bound}"))
            elif not item.text.strip() and item.bound not in _MAY_BE_EMPTY:
                out.append(Problem("V12", item.where, "length: an empty text where the story "
                                                      "says a word"))
        ctx = authored.get("context")
        facts = _list(ctx.get("facts")) if isinstance(ctx, Mapping) else None
        low, high = _CONTEXT_FACTS["minItems"], _CONTEXT_FACTS["maxItems"]
        if facts is not None and not low <= len(facts) <= high:
            out.append(Problem("V12", "authored.context.facts", f"length: the context states "
                                                                f"{len(facts)} facts; it states "
                                                                f"{low} to {high}"))
        return out


def _strings(texts: Iterable[AuthoredText]) -> Iterator[AuthoredText]:
    return (t for t in texts if isinstance(t.text, str))


def _list(value) -> list:
    return value if isinstance(value, list) else []
