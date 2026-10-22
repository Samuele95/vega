"""
show_layer_story — what the `showcase` layer says about the story (contract §8; D5 §9.1): the
headline and the lead rendered with their values, the section outline, the motif, polarity,
class roles and input groups, and the bindings — every fact key the story holds with its value,
the record it was read from and where the story's text uses it.

    story_summary(story, page) -> dict       the layer's `story`
    bindings(story)            -> [dict]     the layer's `bindings`
    render_text(text, story)   -> (str, [unresolved placeholders])

Nothing is computed here but text: a value is written as the page writes it (js/core/fmt.js and
text.js FORMATS), a lexicon path as the lexicon's word. A placeholder neither the values nor the
lexicon resolve (the target noun of a lexicon without `target`, which the page takes from its
strings table) is left as written and named in `story.unresolved`.
"""
from __future__ import annotations

import re
from typing import Iterator, Mapping

from digit_rule import PLACEHOLDER_RE

__all__ = ["story_summary", "bindings", "render_text", "fmt_num", "fmt_value"]

# the story keys that hold no authored text (the record of values, its sources, the bundle pin)
NOT_TEXT = frozenset({"schema", "locale", "task", "lexicon", "class_roles", "polarity", "values",
                      "formats", "refs", "bundle", "report"})
# the list a section shows, per section id; every other list of the section is counted too
SECTION_LISTS = ("steps", "facts", "choices", "runs", "models", "limits")


def _group(n: int) -> str:
    return f"{n:,}"


def fmt_num(v) -> str:
    """js/core/fmt.js fmtNum: integers grouped, then by magnitude."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if float(v).is_integer():
        return _group(int(v))
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return _trim(f"{v:.4f}")
    return _trim(f"{float(f'{v:.2g}')!r}")


def _trim(s: str) -> str:
    return s.rstrip("0").rstrip(".") if "." in s and "e" not in s else s


def _scalar(v):
    return v.get("value") if isinstance(v, Mapping) and "value" in v else v


def fmt_value(v, short: bool = False) -> str:
    """fmt.js fmtVal: a list counted, {value, ci} as the value and its interval."""
    if isinstance(v, list):
        return _group(len(v))
    if isinstance(v, Mapping) and "value" in v:
        ci = v.get("ci")
        tail = f" [{fmt_num(ci[0])}, {fmt_num(ci[1])}]" if isinstance(ci, list) and len(ci) == 2 and not short else ""
        return fmt_num(v["value"]) + tail
    return fmt_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _signed(v: float) -> str:
    return ("+" if v >= 0 else "−") + f"{abs(v):.3f}"


FORMATS = {
    "int": lambda v: _group(round(float(_scalar(v)))),
    "num": lambda v: fmt_num(float(_scalar(v))),
    "pct": lambda v: f"{_trim(f'{100 * float(_scalar(v)):.1f}')} %",
    "signed": lambda v: _signed(float(_scalar(v))),
    "count": lambda v: _group(len(v) if isinstance(v, list) else int(_scalar(v))),
}


def _lexicon_word(lexicon: Mapping, key: str, path: str) -> str | None:
    entry = lexicon.get(key) if isinstance(lexicon, Mapping) else None
    if not isinstance(entry, Mapping):
        return None
    if path in ("one", "many"):
        return entry.get(path)
    if path == "a":
        return f"{entry.get('article')} {entry.get('one')}" if entry.get("one") else None
    if path == "the":
        return f"the {entry.get('one')}" if entry.get("one") else None
    return None


def render_text(text: str, story: Mapping) -> tuple[str, list[str]]:
    """`text` with its placeholders filled from the story's values and lexicon."""
    values, lexicon, missed = story.get("values") or {}, story.get("lexicon") or {}, []

    def fill(m: re.Match) -> str:
        name, path, suffix = m.group(1), m.group(2), m.group(3)
        if path is None and name in values:
            v = values[name]
            return FORMATS[suffix](v) if suffix else fmt_value(v)
        word = _lexicon_word(lexicon, name, path) if path else None
        if word is None:
            missed.append(m.group(0))
            return m.group(0)
        return word

    return PLACEHOLDER_RE.sub(fill, str(text or "")), missed


def _texts(node, path: str) -> Iterator[tuple[str, str]]:
    """(path, text) of every string under `node`; a section is named by its id."""
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, Mapping):
        for k, v in node.items():
            yield from _texts(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _texts(v, f"{path}[{i}]")


def text_paths(story: Mapping) -> Iterator[tuple[str, str]]:
    for key, node in story.items():
        if key in NOT_TEXT:
            continue
        if key == "sections" and isinstance(node, list):
            for sec in node:
                body = {k: v for k, v in (sec or {}).items() if k != "id"}
                yield from _texts(body, f"sections.{(sec or {}).get('id')}")
        else:
            yield from _texts(node, key)


def bindings(story: Mapping) -> list[dict]:
    """One row per fact key of the story's values: its value, its record and its uses."""
    uses: dict[str, list[str]] = {}
    for where, text in text_paths(story):
        for m in PLACEHOLDER_RE.finditer(text):
            if m.group(2) is None:
                uses.setdefault(m.group(1), [])
                if where not in uses[m.group(1)]:
                    uses[m.group(1)].append(where)
    refs, formats = story.get("refs") or {}, story.get("formats") or {}
    return [{"key": key, "value": value, "ref": refs.get(key), "format": formats.get(key),
             "used_in": uses.get(key, [])}
            for key, value in sorted((story.get("values") or {}).items())]


def _sections(story: Mapping) -> list[dict]:
    out = []
    for sec in story.get("sections") or []:
        items = {k: len(v) for k, v in sec.items() if isinstance(v, list)}
        ordered = {k: items[k] for k in SECTION_LISTS if k in items}
        ordered.update({k: n for k, n in items.items() if k not in ordered})
        out.append({"id": sec.get("id"), "items": ordered})
    return out


def _families(story: Mapping) -> list[dict]:
    return [{"id": g.get("id"), "label": g.get("label"), "members_n": len(g.get("members") or [])}
            for g in story.get("input_groups") or []]


def story_summary(story: Mapping, page: Mapping) -> dict:
    """The layer's `story`: the headline and lead rendered, the outline, the roles and groups."""
    headline, miss_h = render_text(story.get("headline"), story)
    lead, miss_l = render_text(story.get("lead"), story)
    roles = story.get("class_roles") or {}
    motif = (page.get("motif") or {}) if isinstance(page, Mapping) else {}
    return {"headline": headline, "lead": lead,
            "headline_template": story.get("headline"), "lead_template": story.get("lead"),
            "locale": story.get("locale"), "sections": _sections(story),
            "motif": motif.get("id"), "motif_source": motif.get("source"),
            "polarity": story.get("polarity"),
            "class_roles": {"roles": roles.get("roles") or {}, "source": roles.get("source") or {}},
            "input_families": _families(story),
            "values_n": len(story.get("values") or {}),
            "unresolved": sorted(set(miss_h + miss_l))}
