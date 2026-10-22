"""
field_patterns — the placeholder language of `catalog.json` `output_readings`: a reading's field
patterns (`p_<class>`, `value_h<h>_q<q>`, `recon_<feature>`) filled with what one bundle has.

    field_patterns.binding(facts) -> {placeholder: [(token, value)]}
    field_patterns.expand(pattern, binding) -> [(name, {placeholder: value})]
    field_patterns.of(template, values, binding) -> the kind's `of`, filled
    field_patterns.gated(row, pattern, facts) -> bool          a `conditional` pattern's gate
    field_patterns.named(row, facts) -> {name: (pattern, {"kind", "of"})}

`facts` is `{"classes", "positive", "quantiles", "horizons", "head_out", "features"}` and `row`
is one reading of `output_readings.readings`. The placeholders are the catalog's
(`output_readings.placeholders`): a placeholder is written as its *token* in a field name
(`q_2p5`, `value_h3`, `emb_0`) and as its *value* in `of` (0.025, 3, 0), and a pattern with two
of them fills in the pattern's order, so `value_h<h>_q<q>` is horizon-major like the head.
"""
from __future__ import annotations

import itertools
import re

__all__ = ["CONDITIONS", "HOLE", "binding", "expand", "gated", "named", "of"]

HOLE = re.compile(r"<([a-z]+)>")
# the catalog states each gate in prose (`output_readings.readings.<r>.conditional`); these are
# those sentences, keyed by the field pattern they gate
CONDITIONS = {
    "value": lambda f: any(abs(q - 0.5) < 1e-9 for q in f["quantiles"] or []),
    "value_h<h>": lambda f: not f["quantiles"],
    "value_h<h>_q<q>": lambda f: bool(f["quantiles"]),
    "recon_<feature>": lambda f: f["head_out"] == len(f["features"] or []),
    "out_<k>": lambda f: f["head_out"] != len(f["features"] or []),
    "label": lambda f: bool(f["classes"]),
}


def binding(facts) -> dict:
    """Placeholder -> [(token, value)] for these facts; an absent fact binds nothing, so a
    pattern naming it fills to no field."""
    classes, quantiles = facts.get("classes") or [], facts.get("quantiles") or []
    width = list(range(facts.get("head_out") or 0))
    positive = facts.get("positive")
    median = [min(quantiles, key=lambda q: abs(q - 0.5))] if quantiles else []
    return {"class": _pairs(classes), "label": _pairs(classes), "cum": _pairs(classes[:-1]),
            "positive": _pairs([positive] if positive is not None else []),
            "q": [(_q_token(q), q) for q in quantiles],
            "median": [(_q_token(q), q) for q in median],
            "h": _pairs(facts.get("horizons") or []),
            "feature": _pairs(facts.get("features") or []),
            "j": _pairs(width), "k": _pairs(width)}


def _pairs(values):
    return [(str(v), v) for v in values]


def _q_token(q) -> str:
    """0.1 -> "10", 0.025 -> "2p5": `format(q*100, "g")` with the point spelled `p`."""
    return format(q * 100, "g").replace(".", "p")


def expand(pattern, bound):
    """[(name, {placeholder: value})]: the product of the pattern's placeholders' values."""
    holes = HOLE.findall(pattern)
    out = []
    for combo in itertools.product(*(bound.get(hole) or [] for hole in holes)):
        tokens = {hole: token for hole, (token, _) in zip(holes, combo)}
        out.append((HOLE.sub(lambda m: tokens[m.group(1)], pattern),
                    {hole: value for hole, (_, value) in zip(holes, combo)}))
    return out


def of(template, values, bound):
    """A kind's `of` filled: a lone placeholder gives its value, a composite one its text
    (`"<h>/<q>"` -> `"1/0.1"`); a placeholder the field's name does not carry (`<positive>` of
    binary's `score`) takes its one bound value."""
    if template is None:
        return None
    filled = {hole: values[hole] if hole in values else (bound.get(hole) or [(None, None)])[0][1]
              for hole in HOLE.findall(template)}
    whole = HOLE.fullmatch(template)
    return filled[whole.group(1)] if whole else \
        HOLE.sub(lambda m: format(filled[m.group(1)], "g"), template)


def gated(row, pattern, facts) -> bool:
    """Whether a pattern yields fields: always, unless the reading lists it `conditional`."""
    return pattern not in row["conditional"] or CONDITIONS[pattern](facts)


def named(row, facts) -> dict:
    """name -> (pattern, {"kind", "of"}) for every name the reading's field patterns give these
    facts, gated or not; the first pattern to give a name keeps it."""
    bound, out = binding(facts), {}
    for pattern in row["fields"]:
        kind = row["kinds"][pattern]
        for name, values in expand(pattern, bound):
            out.setdefault(name, (pattern, {"kind": kind["kind"],
                                            "of": of(kind["of"], values, bound)}))
    return out
