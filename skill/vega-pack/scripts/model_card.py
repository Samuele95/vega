#!/usr/bin/env python3
"""
model_card.py — the bundle's `model-card.html`: the reproducibility document a reader opens
without the chain.

    python3 model_card.py --bundle <bundle dir> --model <model.html>
                          --intended-use <bundle dir>/bundle-layer.json
                          --out <bundle dir>/model-card.html

One self-contained HTML file — no external reference of any kind, inline SVG only — holding the
seven sections of bundle-contract.md §6: Identity, Task and data, Architecture, Training,
Evaluation, Intended use and limits, How to use.

Three things are worth stating plainly, because they are the substance rather than the plumbing:

**No number here was typed by anyone.** Every figure on the card is spliced out of a layer's
`<script id="layer-*-data">` JSON or out of the bundle's own records (`MANIFEST.json`,
`schema.json`, `markers.json`, `blueprint-map.json`). This module formats and places numbers; it
never computes one, and it never carries one of its own. That is the whole reason a reader who
has never seen the chain can trust the page: each figure can be traced back to the record that
measured it, at the rounding it is shown in. That makes it a testable property rather than a
promise — extract the numbers from a rendered card and check each against the spliced records
with `unsupported_numbers` (vega-eval's core), which implements exactly this comparison.

**One passage is hand-written, and it may hold no digit.** *Intended use and limits* is the
packer's prose. `check_intended_use` refuses a digit in it, names the offending number, and says
why: a number typed there would be the single figure on the card that traces to nobody, and it is
the one a reader would most want to check.

**The verdict is not summarised.** `readings.verdict` from the evaluation goes onto the card
verbatim, at the top, where it cannot be scrolled past, together with the stances the bundle
ships under and the worst slice the evaluation found. A card that rendered an unearned model as a
neutral metrics table would be a more comfortable document and a false one.

Exit codes: 0 the card is written; 2 a refusal — a required section could not be filled, or the
hand-written prose contains a digit (no file is written).
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_pack_core import read_json  # noqa: E402

# the layers the card reads, in chain order. `analysis` is optional: a dataset whose analysis
# stage recorded nothing still has a partition, a recipe, a spec and a verdict.
LAYERS = ("geometry", "analysis", "shape", "architecture", "training", "evaluation")
REQUIRED_LAYERS = ("geometry", "shape", "architecture", "training", "evaluation")

_LAYER_RE = ('<script id="layer-{name}-data" type="application/json">(.*?)</script>')

# the four ink colours of the chain's other charts, so a reader who has seen the layer render
# recognises what a line means here
COLOURS = ("#1b3a73", "#9a3a2c", "#2e6b3f", "#7a3c77")

# any run of digits in the hand-written passage — not "any number", because `H1-3` and `fp32`
# are digits a reader would also have to take on trust
_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")


# --------------------------------------------------------------------------- value objects

@dataclass(frozen=True)
class Problem:
    """One reason the card cannot be rendered, with the record it should have come from.

    Carried rather than printed, so the refusal can name every gap at once: a card built one
    missing key at a time would send the packer round the chain five times.
    """

    section: str
    what: str
    where: str

    def line(self) -> str:
        return f"{self.section}: {self.what} — expected in {self.where}"


class Checker:
    """Collects what is missing while the card is being assembled, and returns it.

    Nothing is printed from inside the walk. The caller decides what to say and what to return,
    which is what lets `build_card` be used by a test that wants the problems as data.
    """

    def __init__(self) -> None:
        self._problems: list = []

    def need(self, present, section: str, what: str, where: str) -> bool:
        """Record a gap unless `present` is truthy. Returns whether it was present."""
        if not present:
            self._problems.append(Problem(section, what, where))
            return False
        return True

    def add(self, section: str, what: str, where: str) -> None:
        self._problems.append(Problem(section, what, where))

    @property
    def problems(self) -> tuple:
        """The gaps, one per missing record.

        Two sections may want the same record — the verdict reading is quoted at the top and
        again beside the numbers it judges — and reporting one absence twice makes a refusal
        look longer than the work needed to answer it.
        """
        seen, out = set(), []
        for problem in self._problems:
            if problem.where in seen:
                continue
            seen.add(problem.where)
            out.append(problem)
        return tuple(out)


@dataclass(frozen=True)
class Sources:
    """Everything the card is spliced from, gathered once and never re-read.

    The card is a document about provenance, so the set of records it may draw on is a value:
    a renderer that could reach past this object could put a number on the page that came from
    somewhere the card does not name.
    """

    layers: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] = field(default_factory=dict)
    schema: Mapping[str, Any] = field(default_factory=dict)
    markers: Sequence[str] = field(default_factory=tuple)
    blueprint_map: Mapping[str, Any] = field(default_factory=dict)
    intended_use: str = ""
    package: str = ""
    post_eval: Mapping[str, Any] = field(default_factory=dict)

    def layer(self, name: str) -> Mapping[str, Any]:
        value = self.layers.get(name)
        return value if isinstance(value, Mapping) else {}

    def as_records(self) -> dict:
        """The spliced records, as one tree — the corpus every number on the card must trace to.

        The hand-written prose is deliberately not in it: it is the one passage that is not
        evidence, and including it would let a number in it stand as its own support.
        """
        return {"layers": dict(self.layers), "manifest": dict(self.manifest),
                "schema": dict(self.schema), "markers": list(self.markers),
                "blueprint_map": dict(self.blueprint_map)}


# --------------------------------------------------------------------------- reading inputs

def layer_data(model_html: str, name: str):
    """One layer's JSON, by the `<script id="layer-<name>-data">` block the chain writes."""
    match = re.search(_LAYER_RE.format(name=name), model_html, re.S)
    if not match:
        return None, f"the model page carries no layer-{name}-data script"
    try:
        return json.loads(match.group(1)), None
    except json.JSONDecodeError as exc:
        return None, f"layer-{name}-data is not readable JSON: {exc}"


def package_of(bundle: Path, manifest: Mapping) -> str:
    """The emitted package's directory name inside the bundle.

    Taken from the manifest's own file list first — the bundle names its files by the paths it
    knows them by — and only then by looking for the map on disk, so a bundle whose manifest and
    contents disagree is not quietly papered over by a glob.
    """
    for name in (manifest.get("files") or {}):
        if str(name).endswith("/blueprint-map.json"):
            return str(name).rsplit("/", 1)[0]
    found = sorted(bundle.glob("*/blueprint-map.json"))
    return found[0].parent.name if found else ""


def intended_use_text(path: Path):
    """The packer's prose, from `bundle-layer.json`'s `readings.intended_use`.

    A plain text file is accepted too — the prose is the input, and refusing a `.txt` would make
    the digit rule harder to test than to satisfy.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"--intended-use {path} could not be read: {exc}"
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return text.strip(), None
    if not isinstance(doc, Mapping):
        return text.strip(), None
    value = ((doc.get("readings") or {}).get("intended_use")
             if isinstance(doc.get("readings"), Mapping) else None)
    if value is None:
        value = doc.get("intended_use")
    if isinstance(value, (list, tuple)):
        value = "\n\n".join(str(v) for v in value)
    if not isinstance(value, str) or not value.strip():
        return None, (f"{path} carries no readings.intended_use; §6 says the packer writes the "
                      "prose of *Intended use and limits* and the card splices everything else")
    return value.strip(), None


def check_intended_use(text: str) -> list:
    """The one rule about the hand-written passage: it may contain no digit.

    Prevents the failure the whole card is built to avoid — a number on the page that traces to
    no record. Every other figure here was spliced from a layer's data script or the bundle's own
    files, and a reader can follow any of them back to what measured it. A digit typed into this
    passage would be the single exception, and it would be the number a reader most wants to
    check. The message names it, because "no digits allowed" without the offending number sends
    the author hunting through their own paragraph.
    """
    problems = []
    for match in _DIGITS.finditer(text or ""):
        start = max(0, match.start() - 42)
        end = min(len(text), match.end() + 42)
        context = " ".join(text[start:end].split())
        problems.append(
            f"the hand-written *Intended use and limits* states {match.group()} "
            f"— in “…{context}…”")
    return problems


# --------------------------------------------------------------------------- formatting

def esc(value) -> str:
    """One record value, escaped. There is no unescaped path: every cell on this card holds a
    string out of a record, and a renderer with a "this one is already markup" door is a
    renderer where a record's text can eventually reach the page as markup."""
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return html.escape(num(value))
    if isinstance(value, (list, tuple)):
        # a record's list read as a sentence, not as its Python repr: `['a pruned trial']` in a
        # table cell tells the reader what language the card was written in, not what happened
        return html.escape(", ".join(str(v) for v in value)) if value else "—"
    return html.escape(str(value))


def plural(count, singular: str, many: str | None = None) -> str:
    """`1 row` / `119 rows`, from a record's own count.

    Written out because `row(s)` is the mark of a document assembled by a program, and this one
    is meant to be read by a person and sent to a colleague.
    """
    try:
        one = float(count) == 1
    except (TypeError, ValueError):
        one = False
    return f"{num(count)} {singular if one else (many or singular + 's')}"


def num(value, places: int = 4) -> str:
    """A record's number, in plain decimal, at a rounding that can be traced back to it.

    Never an exponent form: `3.02e-05` does not survive the rounding comparison a reader (or
    `unsupported_numbers`) uses to check a shown figure against the record it came from, and a
    number on this card that cannot be traced is the one defect the card cannot carry.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return str(f)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    exponent = math.floor(math.log10(abs(f)))
    decimals = min(12, max(places, places - 1 - exponent))
    text = f"{f:.{decimals}f}".rstrip("0").rstrip(".")
    return text or "0"


def ci(entry) -> str:
    """`[lo, hi]` from a record that carries an interval, and a plain dash from one that does
    not — a metric shown without its interval must look different from one shown with it."""
    bounds = (entry or {}).get("ci") if isinstance(entry, Mapping) else None
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2
            and bounds[0] is not None and bounds[1] is not None):
        return "—"
    return f"[{num(bounds[0])}, {num(bounds[1])}]"


def value_of(entry) -> str:
    if not isinstance(entry, Mapping):
        return num(entry)
    return num(entry.get("value"))


def joined(values, sep: str = ", ") -> str:
    if not values:
        return "—"
    return sep.join(str(v) for v in values)


# --------------------------------------------------------------------------- html pieces

def para(text, klass: str | None = None) -> str:
    attr = f' class="{html.escape(klass, quote=True)}"' if klass else ""
    return f"<p{attr}>{esc(text)}</p>"


def note(text) -> str:
    return para(text, "note")


def quote(text, attribution: str) -> str:
    """A reading, verbatim, with the record it was copied from named beside it.

    The attribution is the point: a quotation whose source is not on the page is a paraphrase a
    reader has to take on trust.
    """
    return ("<figure class=\"quote\"><blockquote>"
            + "".join(f"<p>{esc(chunk)}</p>" for chunk in str(text).split("\n\n") if chunk.strip())
            + f"</blockquote><figcaption>{esc(attribution)}</figcaption></figure>")


# a digest, a dotted path, a long identifier: one token with nowhere to break politely. Only
# these may be broken mid-word — everything else breaks at spaces, or the table turns
# `ingestion` into `ingestio n` and `architecture` into `architec ture`.
_UNBROKEN = re.compile(r"^(sha256:|[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+$)|^[A-Za-z0-9_]{24,}$")


def _cell_class(value) -> str:
    text = str(value)
    if _UNBROKEN.search(text) or (len(text) > 28 and " " not in text):
        return " class=\"tok\""
    return ""


def table(headers, rows, caption: str | None = None, klass: str = "") -> str:
    """A table, or the statement that the record it would have drawn is empty.

    A cell holding one unbreakable token is marked `tok` so the stylesheet may break it
    anywhere; every other cell wraps at spaces only. The distinction matters more than it
    sounds: a digest that runs off the page is unreadable, and a word split down the middle to
    save four pixels is unreadable too — but only one of them looks like a mistake.
    """
    if not rows:
        return note(f"{caption or 'This table'}: the record holds no rows.")
    head = "".join(f"<th scope=\"col\">{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = [f"<th scope=\"row\"{_cell_class(row[0])}>{esc(row[0])}</th>"]
        cells += [f"<td{_cell_class(cell)}>{esc(cell)}</td>" for cell in row[1:]]
        body.append("<tr>" + "".join(cells) + "</tr>")
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return (f"<div class=\"scroll\"><table class=\"{klass}\">{cap}"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def facts(pairs) -> str:
    """The identity strip: label / value, one per line, printable."""
    items = "".join(f"<div><dt>{esc(k)}</dt><dd{_cell_class(v)}>{esc(v)}</dd></div>"
                    for k, v in pairs if v not in (None, "", []))
    return f"<dl class=\"facts\">{items}</dl>"


def code_block(text) -> str:
    return f"<pre class=\"code\"><code>{html.escape(str(text))}</code></pre>"


def absent(what: str, why: str) -> str:
    """A section the bundle legitimately does not have.

    Stated, never omitted: a heading with nothing under it reads as an author who did not check,
    and "there is no serving scaffold, because the mode is batch" is a fact about the bundle.
    """
    return f"<p class=\"absent\"><strong>{esc(what)}</strong> {esc(why)}</p>"


def section(anchor: str, number: str, title: str, body: str) -> str:
    return (f"<section id=\"{anchor}\"><h2><span class=\"secnum\">{esc(number)}</span> "
            f"{esc(title)}</h2>{body}</section>")


# --------------------------------------------------------------------------- inline svg
#
# No `xmlns` on any of these: inline SVG in an HTML document is already in the SVG namespace,
# and the attribute would put the one absolute URI in the file — which a reader (or a checker)
# grepping this card for an external reference would have to stop and reason about. A card that
# reaches nowhere should also *look* like a card that reaches nowhere.

def _svg_text(x, y, text, *, size=9, anchor="middle", fill="#555", extra="") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" font-family="Georgia, \'Times New Roman\', serif"{extra}>'
            f'{html.escape(str(text))}</text>')


def _spaced(ticks, lo, hi, apart=0.09):
    """The ticks that will not print on top of each other.

    A curve's own extreme is one of the labels — the card labels axes with values out of the
    record, never with round numbers it invented — and an extreme that lands beside the end of
    the axis would print two numbers in the same place. The later tick is dropped, not moved:
    moving it would put a label somewhere the value is not.
    """
    span = (float(hi) - float(lo)) or 1.0
    kept = []
    for value, label in ticks:
        if any(abs(float(value) - float(seen)) / abs(span) < apart for seen, _ in kept):
            continue
        kept.append((value, label))
    return kept


def line_chart(series, *, xdomain, ydomain, xticks, yticks, xlabel, ylabel, title,
               diagonal=False, width=560, height=300) -> str:
    """A chart whose every tick label is a value out of the record it plots.

    Tick labels are not "nice" round numbers chosen by an axis algorithm: they are the first and
    last epoch, the smallest and largest value the series actually reached. An invented tick
    would be a number on the card that no record holds — the one thing this card may not carry.
    """
    pad_l, pad_r, pad_t, pad_b = 58, 14, 26, 40
    x0, x1 = pad_l, width - pad_r
    y0, y1 = height - pad_b, pad_t
    xlo, xhi = xdomain
    ylo, yhi = ydomain
    xspan = (xhi - xlo) or 1.0
    yspan = (yhi - ylo) or 1.0

    def px(v):
        return x0 + (x1 - x0) * ((float(v) - xlo) / xspan)

    def py(v):
        return y0 + (y1 - y0) * ((float(v) - ylo) / yspan)

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{html.escape(title)}">',
           f'<title>{html.escape(title)}</title>',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>']
    for value, label in _spaced(yticks, ylo, yhi):
        y = py(value)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e6e2da"/>')
        out.append(_svg_text(x0 - 6, y + 3, label, anchor="end"))
    for value, label in _spaced(xticks, xlo, xhi):
        x = px(value)
        out.append(_svg_text(x, y0 + 15, label))
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8a8378"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8a8378"/>')
    if diagonal:
        out.append(f'<line x1="{px(xlo)}" y1="{py(ylo)}" x2="{px(xhi)}" y2="{py(yhi)}" '
                   'stroke="#bdb6aa" stroke-dasharray="4 3"/>')
    for i, (label, points) in enumerate(series):
        if not points:
            continue
        colour = COLOURS[i % len(COLOURS)]
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        out.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.8" '
                   f'points="{d}"/>')
        out.append(_svg_text(x1, y1 + 11 * (i + 1) - 4, label, anchor="end", fill=colour,
                             size=10))
    out.append(_svg_text((x0 + x1) / 2, height - 6, xlabel))
    out.append(_svg_text(0, 0, ylabel, anchor="middle",
                         extra=f' transform="translate(13,{(y0 + y1) / 2:.0f}) rotate(-90)"'))
    out.append("</svg>")
    return "".join(out)


def interval_bar(point, bounds, *, label, width=560, height=104) -> str:
    """A paired difference drawn against zero.

    The verdict of the whole chain is "does this interval clear zero, and on which side": drawn,
    that is one glance; written, it is three numbers a reader has to hold in their head.
    """
    if point is None or not (isinstance(bounds, (list, tuple)) and len(bounds) == 2
                             and bounds[0] is not None and bounds[1] is not None):
        return ""
    lo, hi = float(bounds[0]), float(bounds[1])
    span = max(abs(lo), abs(hi), abs(float(point))) or 1.0
    dlo, dhi = -span * 1.25, span * 1.25
    x0, x1, y = 20, width - 20, 46

    def px(v):
        return x0 + (x1 - x0) * ((v - dlo) / (dhi - dlo))

    zero = px(0.0)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{html.escape(label)}">',
           f'<title>{html.escape(label)}</title>',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
           f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#e6e2da"/>',
           f'<line x1="{zero:.1f}" y1="{y - 12}" x2="{zero:.1f}" y2="{y + 12}" '
           'stroke="#8a8378" stroke-dasharray="3 3"/>',
           _svg_text(zero, y + 24, "0"),
           f'<line x1="{px(lo):.1f}" y1="{y}" x2="{px(hi):.1f}" y2="{y}" '
           f'stroke="{COLOURS[1] if hi < 0 else COLOURS[0]}" stroke-width="6" '
           'stroke-linecap="round"/>',
           f'<circle cx="{px(float(point)):.1f}" cy="{y}" r="5" fill="#2b2a26"/>',
           # the bounds sit above the bar and the zero marker below it: an interval whose upper
           # bound is a hair from zero — which is exactly the case this chart is for — would
           # otherwise print its bound on top of the zero line
           _svg_text(px(lo), y - 18, num(lo)),
           _svg_text(px(hi), y - 18, num(hi)),
           _svg_text(px(float(point)), y + 24, num(point), fill="#2b2a26", size=10),
           _svg_text(width / 2, height - 6, label),
           "</svg>"]
    return "".join(out)


def bar_chart(rows, *, label, width=560, bar_h=30) -> str:
    """One bar per slice, with its interval drawn on it.

    The slices are where a whole-part average stops being true, and a table's last row is where
    that finding goes to be missed.
    """
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return ""
    height = 34 + bar_h * len(rows)
    # the right gutter holds the value and the slice's n; a bar that ran to the frame would
    # push that label off the edge of the figure
    x0, x1 = 148, width - 104
    top = 22
    values = [float(r[1]) for r in rows]
    bounds = [b for r in rows for b in (r[2] or ()) if b is not None]
    hi = max(values + [float(b) for b in bounds] + [0.0]) or 1.0

    def px(v):
        return x0 + (x1 - x0) * (float(v) / hi)

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{html.escape(label)}">',
           f'<title>{html.escape(label)}</title>',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>']
    for i, (name, value, interval, tail) in enumerate(rows):
        y = top + bar_h * i
        out.append(_svg_text(x0 - 8, y + 13, name, anchor="end", fill="#2b2a26", size=10))
        out.append(f'<rect x="{x0}" y="{y + 4}" width="{max(1.0, px(value) - x0):.1f}" '
                   f'height="14" fill="{COLOURS[0] if i else COLOURS[1]}" opacity="0.85"/>')
        if interval and interval[0] is not None and interval[1] is not None:
            out.append(f'<line x1="{px(interval[0]):.1f}" y1="{y + 11}" '
                       f'x2="{px(interval[1]):.1f}" y2="{y + 11}" stroke="#2b2a26" '
                       'stroke-width="1.4"/>')
        out.append(_svg_text(x1 + 6, y + 15, f"{num(value)} {tail}", anchor="start",
                             fill="#2b2a26", size=10))
    out.append(_svg_text(width / 2, height - 5, label))
    out.append("</svg>")
    return "".join(out)


def block_diagram(blocks, per_block, input_dim, width=560) -> str:
    """The network as the spec declares it, one box per block, in order.

    Shapes and parameter counts are the smoke record's, not this module's arithmetic: the card
    states what the forward pass reported, so a reader comparing it with `network.py` is
    comparing two things that were both measured.
    """
    if not blocks:
        return ""
    shapes = {str(b.get("id")): b for b in (per_block or []) if isinstance(b, Mapping)}
    row_h, top = 44, 40
    height = top + row_h * len(blocks) + 46
    bx, bw = 24, width - 210
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           'aria-label="the network block by block">',
           '<title>the network block by block</title>',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
           _svg_text(bx + bw / 2, 20, f"input: {num(input_dim)} features", size=10,
                     fill="#2b2a26")]
    for i, block in enumerate(blocks):
        y = top + row_h * i
        bid = str(block.get("id"))
        measured = shapes.get(bid) or {}
        out.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="30" rx="4" fill="#f6f3ec" '
                   'stroke="#c9c2b5"/>')
        out.append(_svg_text(bx + 10, y + 19, f"{bid}  {block.get('type')}", anchor="start",
                             fill="#2b2a26", size=11))
        tail = []
        if measured.get("out_shape") is not None:
            tail.append("out " + " x ".join(num(v) for v in (measured.get("out_shape") or [])))
        if measured.get("params") is not None:
            tail.append(f"{num(measured.get('params'))} params")
        out.append(_svg_text(bx + bw + 10, y + 19, ", ".join(tail), anchor="start", size=10))
        if i + 1 < len(blocks):
            out.append(f'<line x1="{bx + bw / 2:.0f}" y1="{y + 30}" x2="{bx + bw / 2:.0f}" '
                       f'y2="{y + row_h}" stroke="#8a8378"/>')
    out.append(_svg_text(bx + bw / 2, height - 14, "output", size=10, fill="#2b2a26"))
    out.append("</svg>")
    return "".join(out)


def thin(points, limit=400):
    """Every nth point of a long curve. A card is read, and a 5,000-point polyline is weight
    without information — but the thinning keeps the endpoints, which are the values the axis
    labels are taken from."""
    points = list(points or ())
    if len(points) <= limit:
        return points
    step = max(1, len(points) // limit)
    kept = points[::step]
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


def figure(svg: str, caption: str) -> str:
    if not svg:
        return ""
    return f"<figure class=\"chart-figure\">{svg}<figcaption>{esc(caption)}</figcaption></figure>"


# --------------------------------------------------------------------------- the sections

def verdict_banner(src: Sources, check: Checker) -> str:
    """The verdict, first and whole, with the stance it ships under.

    §6 puts the verdict in *Evaluation*, verbatim. It is here as well, above everything, because
    a reader deciding whether to use this model decides it in the first screen: an unearned
    network whose verdict lives on page four of a metrics document has been given a badge, not a
    warning. The evaluation's own words, both times — this module never rewords a verdict.
    """
    ev = src.layer("evaluation")
    verdict = ev.get("verdict") or {}
    metrics = ev.get("metrics") or {}
    network = ((metrics.get("models") or {}).get("network") or {})
    primary = metrics.get("primary") or (src.manifest.get("chain") or {}).get(
        "evaluation", {}).get("primary", {}).get("name")
    reading = (ev.get("readings") or {}).get("verdict")
    check.need(verdict, "Evaluation", "no verdict record",
               "layer-evaluation-data verdict")
    check.need(reading, "Evaluation", "no verdict reading to quote",
               "layer-evaluation-data readings.verdict")

    earned = verdict.get("earned") is True
    beaten = verdict.get("beaten_by_baseline") is True
    state = "earned" if earned else ("beaten" if beaten else "not-earned")
    flag = ("Earned" if earned else
            "Not earned: beaten by a baseline" if beaten else "Not earned")
    base = verdict.get("strongest_baseline")

    head = [f"<p class=\"flagline\"><span class=\"flag flag-{state}\">{esc(flag)}</span> "
            f"{esc(src.manifest.get('slug'))} {esc(src.manifest.get('run'))}"
            f"{', a ' + esc(ev.get('metrics', {}).get('task')) + ' model' if metrics.get('task') else ''}"
            f" for {esc(src.manifest.get('label'))}.</p>"]
    if beaten:
        head.append(para(
            "On the held-out part a fitted baseline scored better than the network, and the "
            "interval on that difference does not cover zero. This is not an undecided "
            "comparison. The bundle ships the network anyway, under the stance below."))
    elif not earned:
        head.append(para(
            "The held-out part did not separate the network from the strongest fitted "
            "baseline. The bundle ships under the stance below."))
    head.append(table(
        ["", "value", "interval"],
        [[f"the network on {primary}", value_of(network.get(primary)), ci(network.get(primary))],
         [f"network − {base} on {primary}", num(verdict.get("diff")),
          ci({"ci": verdict.get("ci")})]],
        caption=None, klass="tight"))
    head.append(figure(
        interval_bar(verdict.get("diff"), verdict.get("ci"),
                     label=f"network − {base} on {primary}, against zero"),
        f"the paired difference against {base}, with zero marked"))
    if reading:
        head.append(quote(reading, "the evaluation's verdict, verbatim"))

    slices = ev.get("slices") or {}
    worst = slices.get("worst") or {}
    if worst.get("name"):
        others = [s for s in (slices.get("slices") or [])
                  if s.get("name") != worst.get("name")]
        best = max(others, key=lambda s: ((s.get("metrics") or {}).get(
            slices.get("primary") or "", {}) or {}).get("value") or 0, default=None)
        tail = ""
        if best:
            entry = (best.get("metrics") or {}).get(slices.get("primary") or "") or {}
            tail = (f" On the other side of the same split, {best.get('name')} scores "
                    f"{value_of(entry)} on n = {num(best.get('n'))}.")
        head.append(para(
            f"The evaluation's worst slice on {slices.get('primary')} is "
            f"{worst.get('name')} at {num(worst.get('value'))}, on n = {num(worst.get('n'))}."
            + tail + " A whole-part average over two regimes this far apart is not a "
            "description of either.", "worst"))

    stances = src.manifest.get("stances") or []
    if stances:
        head.append("<p class=\"stance-lead\">It ships under a recorded stance:</p><ul "
                    "class=\"stances\">"
                    + "".join(f"<li>{esc(s)}</li>" for s in stances) + "</ul>")
    elif not earned:
        head.append(para("No stance is recorded. An unearned network with no recorded reason "
                         "for shipping is a bundle nobody has taken responsibility for.",
                         "stance-lead warn"))
    else:
        head.append(para("No stance was needed: the verdict was earned.", "stance-lead"))

    seal = ev.get("seal") or {}
    if seal.get("exposures") is not None:
        head.append(note(
            f"The held-out part was opened {plural(seal.get('exposures'), 'time')}"
            + (", for the evaluation these numbers come from."
               if float(seal.get("exposures") or 0) <= 1 else
               " — every number above was read after the test part had already been seen.")))
    return f"<section id=\"verdict\" class=\"verdict verdict-{state}\">{''.join(head)}</section>"


def _include_line(man: Mapping) -> str:
    """Which of the two bundles this is, said in Identity where a reader looks first.

    The difference is not cosmetic: a `full` bundle carries the training loop and the
    evaluator, so a reader who finds `train.py` beside `predict.py` needs to know from the
    card that a checkpoint from it is not the network this card describes.
    """
    mode = man.get("include") or "serving"
    entry = joined(man.get("entrypoints") or ["predict.py"])
    if mode == "full":
        return (f"full — the whole emitted package, entry points {entry}. train.py fits one "
                "spec (no ladder, no diagnostics, no ledger) and evaluate.py prints point "
                "metrics (no intervals, no baselines, no verdict); neither produces what this "
                "card reports, and the test part stays sealed in both.")
    return f"serving — the serving slice, entry point {entry}"


def identity_section(src: Sources, check: Checker) -> str:
    man = src.manifest
    chain = man.get("chain") or {}
    ev = chain.get("evaluation") or {}
    output = man.get("output") or {}
    check.need(man.get("slug") and man.get("run"), "Identity",
               "the bundle has no slug or run", "MANIFEST.json slug / run")
    check.need(man.get("created_at"), "Identity", "the bundle records no build time",
               "MANIFEST.json created_at")
    check.need(man.get("files"), "Identity", "no file digests",
               "MANIFEST.json files")

    body = [facts([
        ("slug", man.get("slug")), ("run", man.get("run")),
        ("campaign", man.get("campaign")), ("built", man.get("created_at")),
        ("task", ev.get("task") or output.get("kind")), ("label", man.get("label")),
        ("backend", f"{(man.get('backend') or {}).get('name')} "
                    f"{(man.get('backend') or {}).get('version')}"),
        ("python", man.get("python")),
        ("output", joined(output.get("fields"))),
        ("shape", _include_line(man)),
        ("threshold", f"{num(output.get('threshold'))} ({output.get('threshold_source')})"
                      if output.get("threshold") is not None else None),
    ])]
    body.append(note(
        "Every decision this bundle makes moves with that threshold; it was chosen upstream on "
        "the valid part and is recorded here so it can be changed knowingly."))
    body.append(table(["file", "sha256"],
                      [[k, v] for k, v in (man.get("files") or {}).items()],
                      caption="what shipped, by digest"))
    recorded = [("the evaluated model page", chain.get("model_digest")),
                ("netspec", (chain.get("architecture") or {}).get("netspec_digest")),
                ("shape manifest", (chain.get("shape") or {}).get("manifest_digest")),
                ("recipe", (chain.get("shape") or {}).get("recipe_digest"))]
    body.append(table(["what the chain recorded", "sha256"],
                      [[k, v] for k, v in recorded if v],
                      caption="what the chain said should ship"))
    body.append(note(
        "MANIFEST.files can be checked from inside the bundle: tests/test_roundtrip.py "
        "re-hashes each of these without needing the chain that built them."))
    if src.markers:
        body.append("<details class=\"fold\"><summary>what the packer checked</summary><ul "
                    "class=\"markers\">"
                    + "".join(
                        f"<li class=\"{'warn' if str(m).startswith('WARN') else 'ok'}\">"
                        f"{esc(m)}</li>" for m in src.markers)
                    + "</ul></details>")
    roundtrip = man.get("roundtrip") or {}
    if roundtrip:
        tol = roundtrip.get("tolerance") or {}
        body.append(para(
            f"The round trip that proves this is the evaluated model: {roundtrip.get('status')} "
            f"on {plural(roundtrip.get('rows'), 'row')} from {roundtrip.get('source')}, worst "
            f"absolute difference {num(roundtrip.get('max_abs_diff'))} against a bound of "
            f"{num(tol.get('value'))} — the tolerance is the formula "
            f"{tol.get('rule')} at {tol.get('precision')}, not a knob. The shaper's own "
            f"transform check: {roundtrip.get('shaper_transform_check')}."))
    return section("identity", "1", "Identity", "".join(body))


def task_section(src: Sources, check: Checker) -> str:
    geometry = src.layer("geometry")
    shape = src.layer("shape")
    manifest = shape.get("manifest") or {}
    netspec = (src.layer("architecture").get("netspec") or {})
    partitions = geometry.get("partitions") or {}
    label = src.manifest.get("label") or manifest.get("label")
    # The geometry layer writes {chosen, provenance, candidates: [{label, task, input_dim, …}]};
    # an older shape keyed the partitions by label. Resolve either to the chosen candidate.
    candidates = partitions.get("candidates") if isinstance(partitions, dict) else None
    if isinstance(candidates, list):
        chosen = partitions.get("chosen") or label
        partition = next((c for c in candidates if c.get("label") == chosen),
                         candidates[0] if candidates else {})
        partitions = {c.get("label"): c for c in candidates}
    else:
        partition = partitions.get(label) or (list(partitions.values())[0] if partitions else {})
    check.need(partitions, "Task and data", "no partition",
               "layer-geometry-data partitions")
    check.need(manifest.get("input"), "Task and data", "no source dataset record",
               "layer-shape-data manifest.input")
    check.need(manifest.get("steps"), "Task and data", "no recipe steps",
               "layer-shape-data manifest.steps")
    check.need(src.schema.get("input"), "Task and data", "no input schema",
               "schema.json input")

    source = manifest.get("input") or {}
    partition_name = next((k for k, v in partitions.items() if v is partition), None)
    body = [para(
        f"The partition is {partition_name}: the label is {manifest.get('label')} and the task "
        f"is {partition.get('task')}, on {plural(partition.get('input_dim'), 'feature column')} "
        "the geometry named.")]
    body.append(facts([
        ("source dataset", source.get("path")),
        ("rows", num(source.get("rows"))),
        ("columns", num(source.get("columns"))),
        ("digest", source.get("digest")),
        ("seed", num(manifest.get("seed"))),
        ("libraries", ", ".join(f"{k} {v}" for k, v in
                                (manifest.get("library_versions") or {}).items()) or None),
    ]))

    typing = geometry.get("typing") or {}
    if isinstance(typing, list):          # the geometry layer writes [{column, role, …}]
        typing = {t.get("column"): t for t in typing if isinstance(t, dict)}
    contract = {f.get("name"): f for f in (netspec.get("data_contract") or {}).get(
        "features", []) if isinstance(f, Mapping)}
    rows = []
    for name, spec in (manifest.get("output_schema") or {}).items():
        feature = contract.get(name) or {}
        rows.append([name, spec.get("dtype"), spec.get("role"), spec.get("origin"),
                     (typing.get(name) or {}).get("role"),
                     num(feature.get("cardinality")) if feature.get("cardinality") is not None
                     else "—",
                     num(feature.get("unseen_category_id"))
                     if feature.get("unseen_category_id") is not None else "—"])
    body.append(table(
        ["column", "dtype", "role", "from step", "geometry type", "categories", "unseen id"],
        rows, caption="the shaped feature contract — what the network reads"))

    split = next((s for s in (manifest.get("steps") or []) if s.get("op") == "split"), {})
    fitted = split.get("parameters_fitted") or {}
    sizes = fitted.get("sizes") or split.get("rows_after") or {}
    files = (manifest.get("outputs") or {}).get("files") or {}
    if sizes:
        body.append(table(
            ["part", "rows", "columns", "digest"],
            [[part, num(count), num((files.get(part) or {}).get("columns")),
              (files.get(part) or {}).get("digest")] for part, count in sizes.items()],
            caption=f"the split — kind: {fitted.get('kind')}, "
                    f"from step {split.get('id')}"))
    else:
        body.append(absent("The split", "is not recorded in the shape manifest's steps."))

    body.append(table(
        ["step", "op", "source", "fitted on", "rows after", "columns removed"],
        [[s.get("id"), s.get("op"), s.get("source"), s.get("fit_on") or "—",
          ", ".join(f"{k} {num(v)}" for k, v in (s.get("rows_after") or {}).items()),
          joined(s.get("columns_removed")) if s.get("columns_removed") else "—"]
         for s in (manifest.get("steps") or [])],
        caption="the recipe, one line per phase, copied from the shape layer"))
    findings = (src.layer("analysis").get("findings") or [])
    if findings:
        body.append(table(["finding", "title", "detail"],
                          [[f.get("id"), f.get("title"), f.get("detail")] for f in findings],
                          caption="what the analysis found in this data"))
    else:
        body.append(absent("The analysis layer", "recorded no findings for this dataset."))
    return section("task", "2", "Task and data", "".join(body))


def architecture_section(src: Sources, check: Checker) -> str:
    arch = src.layer("architecture")
    netspec = arch.get("netspec") or {}
    smoke = arch.get("smoke") or {}
    mechanics = (src.layer("training").get("mechanics") or {})
    bmap = src.blueprint_map
    entries = bmap.get("entries") or []
    check.need(entries, "Architecture", "the bundle carries no blueprint map",
               f"{src.package}/blueprint-map.json entries")
    check.need(netspec.get("blocks"), "Architecture", "the spec declares no blocks",
               "layer-architecture-data netspec.blocks")
    check.need(smoke.get("params_total") is not None, "Architecture",
               "no parameter count", "layer-architecture-data smoke.params_total")
    check.need(mechanics.get("loss") and mechanics.get("update_rule"), "Architecture",
               "no loss or update rule", "layer-training-data mechanics")

    body = [table(
        ["file", "module", "port / role", "boundary", "status"],
        [[e.get("file"), e.get("module"),
          e.get("port") or e.get("role") or e.get("kind"),
          joined(e.get("boundary")) if e.get("boundary") else "—",
          e.get("status") or "—"] for e in entries],
        caption=f"the blueprint map as the bundle carries it — package {bmap.get('package')}")]
    entrypoints = bmap.get("entrypoints") or []
    if entrypoints:
        body.append(table(["entrypoint", "calls", "stage", "present"],
                          [[e.get("file"), e.get("calls"), e.get("stage"), e.get("present")]
                           for e in entrypoints],
                          caption="the entrypoints the map lists"))
    if bmap.get("package_slice"):
        body.append(note(bmap.get("package_slice")))

    conformance = ((src.manifest.get("blueprint") or {}).get("conformance") or {})
    upstream = arch.get("conformance") or {}
    rows = [["the bundle (mechanical)", conformance.get("mechanical") or "not recorded"],
            ["the bundle (architect)",
             _architect_line(conformance.get("architect"))],
            ["the emitted package upstream (mechanical)",
             upstream.get("mechanical") or "not recorded"],
            ["the emitted package upstream (architect)",
             _architect_line(upstream.get("architect"))]]
    body.append(table(["conformance", "result"], rows, caption="the conformance gate"))
    if (conformance.get("mechanical") or "") != "pass":
        body.append(note(
            "The bundle's own conformance result is not a pass on this record; the map above is "
            "what the bundle carries, and it has not been gated against the blueprint here."))

    decisions = []
    for family, keys in (netspec.get("decision_families") or {}).items():
        for key in keys:
            entry = netspec.get(key)
            for chosen, spec in _decision_rows(key, entry):
                decisions.append([family, key, chosen, spec.get("source"),
                                  spec.get("catalog") if isinstance(spec.get("catalog"), str)
                                  else joined(spec.get("catalog")),
                                  spec.get("rationale")])
    body.append(table(["family", "what", "chosen", "source", "catalog", "why"], decisions,
                      caption="the decisions, with the source of each"))
    forks = arch.get("forks") or []
    if forks:
        body.append(table(["family", "asked", "answered"],
                          [[f.get("family"), f.get("asked"),
                            f.get("answer") or f.get("or")] for f in forks],
                          caption="which of those a person answered"))

    run = src.manifest.get("run")
    if "forged_netspec" in arch:
        row = next((r for r in (src.layer("training").get("experiments") or {}).get("runs") or []
                    if r.get("id") == run), {})
        body.append(note(
            f"This bundle ships run {run} ({row.get('kind') or 'a run of the campaign'}), not the "
            "forged spec: the decisions above, the diagram and the counts below are its resolved "
            "spec (netspec.json)" + (" and its own smoke record (records/architecture.json)"
                                     if arch.get("smoke_of") else "")
            + (f"; its change from the forged spec: {row.get('delta_text')}"
               if row.get("delta_text") else "") + "."))
    body.append(figure(block_diagram(netspec.get("blocks") or [], smoke.get("per_block"),
                                     (netspec.get("data_contract") or {}).get("input_dim")),
                       f"the network as run {run}'s resolved spec declares it"))
    body.append(facts([
        ("parameters", num(smoke.get("params_total"))),
        ("trainable", num(smoke.get("params_trainable"))),
        ("multiply-adds per sample", num(smoke.get("madds_per_sample"))),
        ("forward pass", "ok" if smoke.get("forward_ok") else "not recorded"),
        ("output shape", joined([num(v) for v in (smoke.get("out_shape") or [])])),
        ("backend", f"{(netspec.get('backend') or {}).get('name')} "
                    f"{smoke.get('backend_version') or ''}".strip()),
    ]))
    body.append("<h3>The loss and the update rule</h3>")
    body.append(code_block(mechanics.get("loss")))
    body.append(code_block(mechanics.get("update_rule")))
    reading = (arch.get("readings") or {}).get("mechanics")
    if reading:
        body.append(quote(reading, "the architecture layer's mechanics reading, verbatim"))
    return section("architecture", "3", "Architecture", "".join(body))


def _architect_line(architect) -> str:
    """The architect's result, or the statement that it did not run.

    Absent is not the same as clean, and a blank cell in a conformance table reads as clean.
    """
    if not isinstance(architect, Mapping):
        return "not run — no architect record on this bundle"
    advisory = architect.get("advisory") if isinstance(architect.get("advisory"), Mapping) else {}
    items = [str(i.get("finding")) for i in advisory.get("open_items") or [] if isinstance(i, Mapping)]
    said = (f"; advisory ({advisory.get('mode')}): open items {', '.join(items)} for /architect "
            "and /improve" if items else "")
    return (f"structural {num(architect.get('structural'))}, "
            f"divergence {num(architect.get('divergence'))}, "
            f"recorded {num(architect.get('recorded'))} ({architect.get('run')}){said}")


def _decision_rows(key, entry):
    """One row per decision, whatever shape the netspec key holds.

    `regularization` is a list, `blocks` is a list of blocks, the rest are single records: the
    table would otherwise need a branch at every call site.
    """
    if isinstance(entry, Mapping):
        chosen = entry.get("name") or entry.get("primary") or entry.get("scheme")
        if chosen is None and key == "training":
            chosen = (f"batch {num(entry.get('batch_size'))}, "
                      f"max epochs {num(entry.get('max_epochs'))}, "
                      f"{entry.get('precision')} on {entry.get('device')}")
        if chosen is None and key == "early_stopping":
            chosen = (f"monitor {entry.get('monitor')} ({entry.get('mode')}), patience "
                      f"{num(entry.get('patience'))}")
        params = entry.get("params") or {}
        if params:
            chosen = f"{chosen} ({', '.join(f'{k} {num(v)}' for k, v in params.items())})"
        return [(chosen, entry)]
    if isinstance(entry, list):
        rows = []
        for item in entry:
            if not isinstance(item, Mapping):
                continue
            if key == "blocks":
                params = item.get("params") or {}
                shown = ", ".join(f"{k} {num(v)}" for k, v in params.items()
                                  if not isinstance(v, (list, dict)))
                rows.append((f"{item.get('id')} {item.get('type')}"
                             + (f" ({shown})" if shown else ""), item))
            else:
                rows.append((f"{item.get('kind')} {num(item.get('value'))}", item))
        return rows
    return []


def training_section(src: Sources, check: Checker) -> str:
    training = src.layer("training")
    experiments = training.get("experiments") or {}
    runs = experiments.get("runs") or []
    selection = experiments.get("selection") or {}
    campaign = training.get("campaign") or {}
    histories = training.get("histories") or {}
    check.need(runs, "Training", "the ledger holds no runs",
               "layer-training-data experiments.runs")
    check.need(selection.get("rule"), "Training", "no selection rule",
               "layer-training-data experiments.selection.rule")

    selected = selection.get("run") or src.manifest.get("run")
    primary_name = selection.get("primary") or (runs[0].get("primary_name") if runs else None)
    rows = []
    for run in runs:
        rows.append([
            ("▸ " if run.get("id") == selected else "") + str(run.get("id")),
            run.get("kind"), run.get("delta_text"),
            num(run.get("epochs")), num(run.get("best_epoch")), run.get("stopped_by"),
            num(run.get("primary")), num(run.get("params")),
            joined(run.get("diagnostics")) if run.get("diagnostics") else "—",
            run.get("ineligible_because") or ("eligible" if run.get("eligible") else "—"),
        ])
    body = [table(["run", "kind", "delta", "epochs", "best", "stopped by",
                   f"valid {primary_name}", "params", "diagnostics", "selection"],
                  rows, caption="the ledger — every run the campaign ran")]

    history = histories.get(selected) or {}
    epochs = history.get("epochs") or []
    if epochs:
        body.append(_curves(history, epochs, primary_name, selected))
    else:
        body.append(absent("The curves of the selected run",
                           "are not in the training layer's histories for this run."))

    body.append("<h3>How this run was chosen</h3>")
    body.append(quote(selection.get("rule"), "the selection rule, verbatim from the ladder"))
    body.append(facts([
        ("selected", selection.get("run")),
        (f"its valid {selection.get('primary')}", num(selection.get("value"))),
        ("mode", selection.get("mode")),
        ("margin over the campaign baseline", num(selection.get("margin_over_baseline"))),
        ("override", selection.get("override") or "none"),
        ("budget (minutes)", f"{num((experiments.get('budget') or {}).get('wall_clock_min'))} "
                             f"allowed, "
                             f"{num((experiments.get('budget') or {}).get('spent_min'))} spent"),
        ("device", campaign.get("device")),
        ("seed", num(campaign.get("seed"))),
        ("determinism tolerance",
         num((campaign.get("determinism") or {}).get("tolerance"))),
        ("protocol", (campaign.get("protocol") or {}).get("applied")),
        ("run directory", campaign.get("run_dir")),
    ]))
    hpo = experiments.get("hpo") or []
    if hpo:
        body.append(table(
            ["study", "trials", "completed", "pruned", "best", f"best {primary_name}",
             "sampler"],
            [[h.get("study"), num(h.get("trials")), num(h.get("completed")),
              num(h.get("pruned")), h.get("best"), num(h.get("best_value")),
              h.get("sampler")] for h in hpo],
            caption="the search"))
    markers = training.get("markers") or []
    if markers:
        body.append("<details class=\"fold\"><summary>what the campaign recorded</summary>"
                    "<ul class=\"markers\">"
                    + "".join(f"<li class=\"{'warn' if str(m).startswith('WARN') else 'ok'}\">"
                              f"{esc(m)}</li>" for m in markers) + "</ul></details>")
    return section("training", "4", "Training", "".join(body))


def _curves(history, epochs, primary_name, selected) -> str:
    """The selected run's own curves, from the per-epoch history the campaign recorded."""
    out = []
    for keys, what in ((("train.loss", "valid.loss"), "loss"),
                       ((f"train.{primary_name}", f"valid.{primary_name}"), primary_name)):
        series, values = [], []
        for key in keys:
            points = history.get(key)
            if not points:
                continue
            # a downsampled history may hold null where the backend wrote none (a train
            # metric not computed at that epoch): a point with no value is not drawn, never
            # forced to zero — the card plots the record, it does not fill it in
            pairs = [(e, v) for e, v in zip(epochs, points) if isinstance(v, (int, float))]
            if not pairs:
                continue
            series.append((key, thin(pairs)))
            values.extend(v for _, v in pairs)
        if not series:
            continue
        lo, hi = min(values), max(values)
        out.append(figure(line_chart(
            series, xdomain=(epochs[0], epochs[-1]), ydomain=(lo, hi),
            xticks=[(epochs[0], num(epochs[0])), (epochs[-1], num(epochs[-1]))],
            yticks=[(lo, num(lo)), (hi, num(hi))],
            xlabel="epoch", ylabel=what, title=f"{selected}: {what} per epoch"),
            f"{selected} — {what} per epoch, from the recorded history"))
    return "".join(out)


def evaluation_section(src: Sources, check: Checker) -> str:
    ev = src.layer("evaluation")
    metrics = ev.get("metrics") or {}
    models = metrics.get("models") or {}
    network = models.get("network") or {}
    primary = metrics.get("primary")
    seal = ev.get("seal") or {}
    readings = ev.get("readings") or {}
    check.need(network, "Evaluation", "no metrics for the network",
               "layer-evaluation-data metrics.models.network")
    check.need(seal, "Evaluation", "no seal record", "layer-evaluation-data seal")
    check.need(readings.get("verdict"), "Evaluation", "no verdict reading",
               "layer-evaluation-data readings.verdict")

    test = seal.get("test") or {}
    body = [facts([
        ("run", (ev.get("run") or {}).get("id")),
        ("test part", test.get("path")),
        ("rows", num(test.get("rows"))),
        ("digest", test.get("digest")),
        ("digest matches the shaper's manifest", test.get("digest_matches_manifest")),
        ("opened", seal.get("opened_at")),
        ("exposures", num(seal.get("exposures"))),
        ("bootstrap", f"{num(metrics.get('n_boot'))} resamples at seed "
                      f"{num(metrics.get('boot_seed'))}, shared by every model"),
    ])]
    if seal.get("history"):
        body.append(table(["opened", "run", "reason", "gate"],
                          [[h.get("at"), h.get("run"), h.get("reason"), h.get("gate")]
                           for h in seal.get("history")],
                          caption="every look at the held-out part"))

    names = [n for n in network if isinstance(network.get(n), Mapping)]
    body.append(table(["metric", "value", "interval", "n"],
                      [[name, value_of(network.get(name)), ci(network.get(name)),
                        num((network.get(name) or {}).get("n"))] for name in names],
                      caption=f"the network on the held-out part — primary: {primary}"))

    differences = metrics.get("differences") or {}
    baselines = (ev.get("baselines") or {}).get("baselines") or []
    rows = []
    for baseline in baselines:
        name = baseline.get("name")
        diff = differences.get(name) or {}
        rows.append([name, baseline.get("definition"),
                     value_of(baseline.get("test")) if baseline.get("test") else "—",
                     ci(baseline.get("test")),
                     num(diff.get("value")) if diff else "—",
                     ci(diff) if diff else "—",
                     baseline.get("note") or ("carried from the ledger, no new interval"
                                              if baseline.get("_upstream") else "")])
    body.append(table(["baseline", "what it is", f"test {primary}", "interval",
                       "network − it", "interval", "note"], rows,
                      caption="the baselines, and the paired difference against each"))
    body.append(note(
        "The differences have intervals because every model was scored on the same bootstrap "
        "resamples; two independently drawn intervals would say nothing about which model is "
        "better."))

    curves = (ev.get("curves") or {}).get("curves") or {}
    reliability = (curves.get("reliability") or {}).get("points") or []
    if reliability:
        xs = [p[0] for p in reliability]
        ys = [p[1] for p in reliability]
        body.append(figure(line_chart(
            [("reliability", [(x, y) for x, y in reliability])],
            xdomain=(0.0, 1.0), ydomain=(0.0, 1.0),
            xticks=[(0.0, "0"), (1.0, "1"), (max(xs), num(max(xs)))],
            yticks=[(0.0, "0"), (1.0, "1"), (max(ys), num(max(ys)))],
            xlabel=(curves.get("reliability") or {}).get("x") or "mean predicted",
            ylabel=(curves.get("reliability") or {}).get("y") or "observed rate",
            title="calibration", diagonal=True),
            "the calibration curve; the dashed line is perfect calibration"))
    else:
        body.append(absent("The calibration curve", "is not in the evaluation's curve record."))
    for key, title, diagonal in (("roc", "ROC", True), ("pr", "precision – recall", False)):
        points = (curves.get(key) or {}).get("points") or []
        if not points:
            continue
        body.append(figure(line_chart(
            [(key, thin([(p[0], p[1]) for p in points]))],
            xdomain=(0.0, 1.0), ydomain=(0.0, 1.0),
            xticks=[(0.0, "0"), (1.0, "1")], yticks=[(0.0, "0"), (1.0, "1")],
            xlabel=(curves.get(key) or {}).get("x") or "",
            ylabel=(curves.get(key) or {}).get("y") or "",
            title=title, diagonal=diagonal),
            f"{title} on the held-out part"))
    if readings.get("calibration"):
        body.append(quote(readings.get("calibration"),
                          "the evaluation's calibration reading, verbatim"))

    slices = ev.get("slices") or {}
    slice_rows = slices.get("slices") or []
    if slice_rows:
        sprimary = slices.get("primary") or primary
        body.append(figure(bar_chart(
            [(s.get("name"), ((s.get("metrics") or {}).get(sprimary) or {}).get("value"),
              ((s.get("metrics") or {}).get(sprimary) or {}).get("ci"),
              f"n = {num(s.get('n'))}")
             for s in sorted(slice_rows,
                             key=lambda s: ((s.get("metrics") or {}).get(sprimary) or {}
                                            ).get("value") or 0)],
            label=f"{sprimary} by slice, with intervals"),
            f"{sprimary} on each slice the evaluation cut"))
        body.append(table(
            ["slice", "source", "n", sprimary, "interval", "note"],
            [[s.get("name"), s.get("source"), num(s.get("n")),
              value_of((s.get("metrics") or {}).get(sprimary)),
              ci((s.get("metrics") or {}).get(sprimary)), s.get("note") or ""]
             for s in slice_rows],
            caption="the slices"))
    else:
        body.append(absent("The slices", "were not cut by the evaluation."))
    if readings.get("slices"):
        body.append(quote(readings.get("slices"), "the evaluation's slice reading, verbatim"))

    checks = (ev.get("robustness") or {}).get("checks") or []
    if checks:
        body.append(table(
            ["check", "status", "n", "value", "interval", "why not"],
            [[c.get("check"), c.get("status") or "run", num(c.get("n")),
              value_of(c.get("primary")) if c.get("primary") else "—",
              ci(c.get("primary")) if c.get("primary") else "—", c.get("why") or ""]
             for c in checks],
            caption="the robustness checks, including the ones that could not run"))
    if readings.get("robustness"):
        body.append(quote(readings.get("robustness"),
                          "the evaluation's robustness reading, verbatim"))

    body.append("<h3>The verdict</h3>")
    body.append(quote(readings.get("verdict"), "the evaluation's verdict, verbatim"))
    forks = ev.get("forks") or []
    if forks:
        body.append(table(["the question the evaluation left open", "the options it named"],
                          [[f.get("asked"), joined(f.get("or"), "; ") if isinstance(
                              f.get("or"), list) else (f.get("answer") or f.get("or"))]
                           for f in forks],
                          caption="what was decided by a person"))
    return section("evaluation", "5", "Evaluation", "".join(body))


def intended_use_section(src: Sources, check: Checker) -> str:
    """The one hand-written passage, beside the ranges it is a judgement about."""
    limits = (src.layer("evaluation").get("readings") or {}).get("limits")
    inputs = src.schema.get("input") or []
    check.need(src.intended_use, "Intended use and limits", "no hand-written prose",
               "bundle-layer.json readings.intended_use")
    check.need(limits, "Intended use and limits", "no limits reading",
               "layer-evaluation-data readings.limits")
    check.need(inputs, "Intended use and limits", "no input schema to draw the ranges from",
               "schema.json input")

    body = ["<div class=\"handwritten\">"
            + "".join(f"<p>{esc(chunk)}</p>"
                      for chunk in str(src.intended_use).split("\n\n") if chunk.strip())
            + "</div>",
            note("The paragraphs above are the only prose on this card written by hand rather "
                 "than copied from a record, which is why they contain no figures: every "
                 "number here is spliced, and one typed into this passage would be the single "
                 "figure a reader could not trace.")]
    body.append(quote(limits, "the evaluation's limits reading, verbatim"))
    body.append(post_eval_block(src.post_eval))
    rows = []
    for column in inputs:
        allowed = column.get("allowed")
        seen = column.get("range_seen")
        domain = ("—" if allowed is None and seen is None else
                  (f"[{num(seen[0])}, {num(seen[1])}]" if seen else joined(allowed)))
        rows.append([column.get("name"), column.get("dtype"), column.get("required"),
                     domain, num(column.get("distinct_seen"))
                     if column.get("distinct_seen") is not None else "—",
                     column.get("note") or ""])
    body.append(table(["column", "dtype", "required", "seen in training", "distinct", "note"],
                      rows, caption="the ranges seen in training, from schema.json"))
    body.append(note(
        "A value outside those ranges is answered and flagged rather than refused: the "
        "prediction is still returned, with a warning on the row, and it is extrapolation "
        "nobody measured."))
    if src.schema.get("ignored"):
        body.append(note("Columns the predictor ignores when they are present: "
                         + joined(src.schema.get("ignored")) + "."))
    stances = src.manifest.get("stances") or []
    if stances:
        body.append("<p class=\"stance-lead\">The stances under which it ships:</p>"
                    "<ul class=\"stances\">"
                    + "".join(f"<li>{esc(s)}</li>" for s in stances) + "</ul>")
    else:
        body.append(absent("Stances", "none were recorded for this bundle."))
    return section("intended-use", "6", "Intended use and limits", "".join(body))


def post_eval_block(doc) -> str:
    """Figures computed after the evaluation from its stored predictions, when supplied.

    They are not the evaluator's — its multiclass catalog lacks them — so they are shown apart,
    with their own intervals and the digest of the prediction file they were recomputed from:
    no checkpoint was loaded and the test part was not opened again, so the exposure count is
    the evaluation's. The record's binary figures collapse the classes to one reference class
    against the rest; the card names them so, never by a dataset's own class names.
    """
    if not doc:
        return ""
    src_rec = doc.get("source") or {}
    boot = doc.get("bootstrap") or {}
    names = {"mcc": "MCC, multiclass", "mcc_binary": "MCC, the reference class vs the rest",
             "attack_recall": "recall of the rest (any other row flagged as any other class)",
             "fpr": "false-positive rate on the reference class's rows"}
    rows = [[names.get(k, k), num(v.get("value"), 6),
             f"[{num((v.get('ci') or [None, None])[0], 6)}, {num((v.get('ci') or [None, None])[1], 6)}]"]
            for k, v in (doc.get("metrics") or {}).items()]
    b = doc.get("binary") or {}
    out = [f"<p class=\"stance-lead\">Measured after the evaluation, from its stored predictions "
           f"({esc(src_rec.get('predictions'))}, {esc(str(src_rec.get('digest'))[:23])}…, "
           f"exposure {esc(src_rec.get('exposure'))}; bootstrap {esc(boot.get('n_boot'))} resamples, "
           f"seed {esc(boot.get('seed'))}):</p>",
           table(["metric", "value", "95% interval"], rows,
                 caption="post-evaluation metrics, recomputed from the stored test predictions"),
           table(["reference rows", "false positives", "other rows", "missed"],
                 [[b.get("benign_rows"), b.get("false_positives"), b.get("attack_rows"),
                   b.get("missed_attacks")]],
                 caption="the reference class vs the rest, counts on the test part")]
    conf = [[c.get("true"), c.get("predicted"), c.get("count")]
            for c in (doc.get("top_confusions") or [])]
    if conf:
        out.append(table(["true class", "predicted as", "test rows"], conf,
                         caption="the largest confusions on the test part"))
    pc = [[c, v.get("support"), "—" if v.get("recall") is None else num(v.get("recall"), 4)]
          for c, v in sorted((doc.get("per_class") or {}).items(),
                             key=lambda kv: kv[1].get("support") or 0)]
    if pc:
        out.append(table(["class", "test rows", "recall"], pc,
                         caption="per-class recall with its support; a class with a handful of "
                                 "rows is an anecdote, not a measurement"))
    return "".join(out)


def usage_section(src: Sources, check: Checker) -> str:
    man = src.manifest
    usage = man.get("usage") or {}
    output = man.get("output") or {}
    check.need(usage.get("cli") or usage.get("module"), "How to use",
               "no usage lines", "MANIFEST.json usage")

    body = [para("Install the pinned requirements, then call it. The bundle imports nothing "
                 "from the chain that built it."),
            code_block("pip install -r requirements.txt")]
    if usage.get("cli"):
        # the interpreter word is taken from the manifest's own line rather than assumed, so the
        # two snippets on the card cannot disagree about how this bundle is invoked
        interpreter = str(usage.get("cli")).split(" ", 1)[0] or "python"
        body.append("<h3>Command line</h3>")
        body.append(code_block(usage.get("cli")))
        body.append(code_block(f"{interpreter} predict.py --check"))
        body.append(note("The second line re-runs the shipped round trip: the bundle's own "
                         "predictions against the ones it was packed with."))
    if usage.get("module"):
        body.append("<h3>As a module</h3>")
        body.append(code_block(usage.get("module")))
    if usage.get("train") or usage.get("evaluate"):
        body.append("<h3>Fitting and scoring (this bundle carries the whole package)</h3>")
        body.append(code_block("\n".join(
            line for line in (usage.get("train"), usage.get("evaluate")) if line)))
        body.append(note(
            "Neither is the stage it echoes. train.py runs one spec once — no ladder, no "
            "ablations, no diagnostics, no ledger — so its checkpoint is a network this card "
            "does not describe; evaluate.py prints point metrics with no intervals, no "
            "baselines and no verdict, so its numbers do not belong beside section 5's. "
            "Both need shaped parts, which do not travel with the bundle; recipe.json is what "
            "makes them. The test part stays sealed: load_part refuses it, and the one door "
            "wants the campaign's test_seal.json and its token."))
    body.append(table(["field", "what it is"],
                      [[k, v] for k, v in (src.schema.get("output") or {}).items()],
                      caption="what predict returns"))
    if output.get("threshold") is not None:
        body.append(note(f"The decision threshold is {num(output.get('threshold'))}, chosen "
                         f"upstream by {output.get('threshold_source')}."))

    serve = man.get("serve") or {}
    body.append("<h3>Serving</h3>")
    if serve.get("present"):
        body.append(facts([("mode", serve.get("mode")), ("framework", serve.get("framework")),
                           ("endpoint", serve.get("endpoint")),
                           ("source", serve.get("source"))]))
    else:
        body.append(absent(
            "No serving scaffold ships with this bundle",
            f"— the blueprint's serving mode is {serve.get('mode') or 'not recorded'}, "
            f"and its interface is: {serve.get('endpoint') or 'predict.py --input'}."))
        if serve.get("note"):
            body.append(note(serve.get("note")))

    exports = man.get("exports") or []
    body.append("<h3>Exports</h3>")
    if exports:
        body.append(table(["format", "file", "parity", "status"],
                          [[e.get("format"), e.get("file"), num(e.get("parity_atol")),
                            e.get("status")] for e in exports],
                          caption="exported models, checked against predict.py"))
        failed = [e for e in exports if e.get("status") != "pass"]
        if failed:
            body.append(note("An export whose parity did not pass ships marked, never "
                             "silently: " + joined([e.get("format") for e in failed]) + "."))
    else:
        body.append(absent("No exports were built for this bundle",
                           "— the network is served through predict.py only."))

    examples = man.get("examples") or {}
    body.append("<h3>What the shipped test proves</h3>")
    if examples.get("kind") == "synthetic":
        body.append(para(
            f"examples/rows.csv holds {plural(examples.get('rows'), 'synthetic row')} and "
            "examples/expected.parquet the predictor's own outputs at pack time, so "
            "tests/test_roundtrip.py proves only that the bundle still reproduces itself. The "
            "proof that it reproduces the evaluation was run at pack time, on the whole test "
            "part, and is recorded above."))
    elif examples:
        body.append(para(
            f"examples/rows.csv holds {plural(examples.get('rows'), 'row')} of the test part and "
            f"examples/expected.parquet the scores the evaluation stored for them "
            f"({examples.get('source')}), so tests/test_roundtrip.py proves the bundle "
            "reproduces the evaluation."))
    else:
        body.append(absent("No examples", "are recorded in the manifest."))
    return section("usage", "7", "How to use", "".join(body))


# --------------------------------------------------------------------------- the document

STYLE = """
:root { --ink:#2b2a26; --soft:#5c584f; --rule:#d8d2c6; --paper:#fdfcf9; --band:#f6f3ec;
        --bad:#9a3a2c; --good:#2e6b3f; --deep:#1b3a73; }
* { box-sizing: border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
       font-family: Georgia, 'Iowan Old Style', 'Times New Roman', serif;
       font-size:17px; line-height:1.62; }
main { max-width: 46rem; margin:0 auto; padding: 3rem 1.5rem 5rem; }
h1 { font-size:2.1rem; line-height:1.2; margin:0 0 .3rem; letter-spacing:-.01em; }
h2 { font-size:1.45rem; margin:3.2rem 0 .8rem; padding-top:.9rem;
     border-top:1px solid var(--rule); }
h3 { font-size:1.1rem; margin:2rem 0 .5rem; }
h4 { font-size:1rem; margin:1.4rem 0 .3rem; }
.secnum { color:var(--soft); font-size:.95rem; margin-right:.5rem; }
p { margin:.75rem 0; }
.lede { font-size:1.1rem; color:var(--soft); margin:0 0 1.6rem; }
.note { color:var(--soft); font-size:.92rem; }
.absent { background:var(--band); border-left:3px solid var(--rule); padding:.6rem .9rem;
          font-size:.95rem; }
/* Navigation. On a wide screen the contents become a fixed rail beside the page and mark
   where the reader is; below that width they collapse back into a block at the top, which is
   the only sensible place for them on a phone. Printed, they disappear — paper has page
   numbers. */
.contents { background:var(--band); padding:.9rem 1.2rem; margin:2rem 0; font-size:.95rem; }
.contents ol { margin:.3rem 0; padding-left:1.4rem; }
.contents a { color:var(--deep); }
.contents .rail-title { display:none; }

@media screen and (min-width: 78rem) {
  .contents { position:fixed; top:0; left:0; bottom:0; width:16rem; margin:0;
              padding:2.4rem 1.2rem 2rem 1.6rem; overflow-y:auto; background:var(--band);
              border-right:1px solid var(--rule); font-size:.9rem; }
  .contents .rail-title { display:block; font-size:.72rem; letter-spacing:.09em;
                          text-transform:uppercase; color:var(--soft);
                          margin:0 0 .9rem; font-weight:600; }
  .contents ol { list-style:none; padding-left:0; margin:0; counter-reset:sec; }
  .contents li { counter-increment:sec; margin:0; }
  .contents li a { display:block; padding:.34rem .5rem .34rem 1.7rem; position:relative;
                   text-decoration:none; color:var(--soft); border-radius:3px;
                   line-height:1.35; }
  .contents li a::before { content:counter(sec); position:absolute; left:.45rem;
                           color:var(--rule); font-variant-numeric:tabular-nums; }
  .contents li a:hover { background:#fff; color:var(--ink); }
  /* where the reader is, without a line of script: the section reached by an anchor lights
     its own entry. Written out per section by `rail_target_rules()` from the same tuple the
     rail itself is built from. */
__TARGET_RULES__
  .contents li a.here { background:#fff; color:var(--ink); font-weight:600;
                        box-shadow:inset 3px 0 0 var(--bad); }
  .contents .rail-foot { margin-top:1.4rem; padding-top:.9rem;
                         border-top:1px solid var(--rule); font-size:.82rem; }
  .contents .rail-foot a { color:var(--deep); text-decoration:none; display:block;
                           padding:.2rem 0; }
  main { margin-left:16rem; max-width:48rem; }
  /* a heading landed on by an anchor should not sit flush against the top of the window */
  section[id] { scroll-margin-top:1.5rem; }
}

/* every section ends with the way back, for the reader who followed a link into the middle */
.totop { text-align:right; margin:.6rem 0 0; font-size:.82rem; }
.totop a { color:var(--soft); text-decoration:none; }
.totop a:hover { color:var(--deep); }
.verdict { border:2px solid var(--ink); padding:1.2rem 1.4rem; margin:1.6rem 0 0; }
.verdict-beaten, .verdict-not-earned { border-color:var(--bad); background:#fbf5f3; }
.verdict-earned { border-color:var(--good); background:#f4f8f4; }
.flagline { font-size:1.05rem; margin-top:0; }
.flag { display:inline-block; padding:.15rem .5rem; margin-right:.4rem; color:#fff;
        background:var(--bad); font-size:.9rem; letter-spacing:.02em; }
.flag-earned { background:var(--good); }
.worst { border-left:3px solid var(--bad); padding-left:.9rem; }
.stance-lead { margin-bottom:.2rem; }
ul.stances { margin:.2rem 0 .6rem; padding-left:1.2rem; }
ul.stances li { margin:.25rem 0; }
.warn { color:var(--bad); }
figure { margin:1.4rem 0; }
figure.quote { border-left:3px solid var(--rule); margin:1.4rem 0; padding:0 0 0 1.1rem; }
blockquote { margin:0; font-size:1rem; }
figcaption { color:var(--soft); font-size:.85rem; margin-top:.4rem; }
.chart-figure { text-align:center; }
svg.chart { max-width:100%; height:auto; border:1px solid var(--rule); background:#fff; }
.scroll { overflow-x:auto; margin:1.2rem 0; }
table { border-collapse:collapse; width:100%; font-size:.88rem;
        font-family:-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
caption { text-align:left; color:var(--soft); font-size:.85rem; padding-bottom:.35rem; }
th, td { border-bottom:1px solid var(--rule); padding:.35rem .5rem; text-align:left;
         vertical-align:top; overflow-wrap:break-word; hyphens:none; }
/* the exception, and only it: one token with nowhere to break politely */
.tok { overflow-wrap:anywhere; font-family:'SF Mono', Menlo, Consolas, monospace;
       font-size:.92em; }
dl.facts dd.tok { font-family:'SF Mono', Menlo, Consolas, monospace; font-size:.92em; }
thead th { border-bottom:1.5px solid var(--ink); font-weight:600; white-space:nowrap; }
tbody th { font-weight:600; }
td { color:var(--soft); }
table.tight { width:auto; }
dl.facts { display:grid; grid-template-columns:1fr; gap:0; margin:1.1rem 0;
           font-size:.92rem; }
dl.facts > div { display:grid; grid-template-columns:13rem 1fr; gap:.6rem;
                 border-bottom:1px solid var(--rule); padding:.3rem 0; }
dl.facts dt { color:var(--soft); }
dl.facts dd { margin:0; overflow-wrap:break-word; }
pre.code { background:var(--band); border:1px solid var(--rule); padding:.7rem .9rem;
           overflow-x:auto; font-size:.85rem; line-height:1.45;
           font-family:'SF Mono', Menlo, Consolas, monospace; }
.handwritten { border:1px solid var(--rule); background:#fff; padding:.4rem 1.1rem;
               margin:1.2rem 0; }
details.fold { margin:1.2rem 0; font-size:.9rem; }
details.fold summary { cursor:pointer; color:var(--deep); }
ul.markers { list-style:none; padding-left:0; }
ul.markers li { border-bottom:1px solid var(--rule); padding:.25rem 0; font-size:.85rem;
                font-family:-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
ul.markers li.warn { color:var(--bad); }
footer { margin-top:3.5rem; padding-top:1rem; border-top:1px solid var(--rule);
         color:var(--soft); font-size:.88rem; }
@media print {
  body { font-size:10.5pt; background:#fff; }
  main { max-width:none; padding:0; }
  a { color:inherit; text-decoration:none; }
  h2 { page-break-after:avoid; }
  /* a long table is allowed to break across pages and repeats its head when it does; a table
     forbidden to break pushes itself whole onto the next sheet and leaves half a page blank */
  thead { display:table-header-group; }
  /* on paper there is no sideways scroll: a digest that ran off the sheet on screen would be
     printed truncated, and a truncated digest is worse than none — it looks checkable */
  .scroll { overflow:visible; }
  td, th { overflow-wrap:break-word; }
  .tok { overflow-wrap:anywhere; }
  tr { page-break-inside:avoid; }
  figure, .verdict, .handwritten { page-break-inside:avoid; }
  details.fold { display:none; }
  .contents, .totop { display:none; }
  main { margin-left:0; max-width:none; }
}
"""

def stylesheet() -> str:
    """The whole stylesheet, with the per-section target rules written in."""
    return STYLE.replace("__TARGET_RULES__", rail_target_rules())


def rail_target_rules() -> str:
    """One rule per section: when it is the anchor target, its rail entry is `here`.

    Generated rather than written out so that adding a section to CONTENTS cannot leave a rail
    entry that never lights — the drift would be invisible, because the page would still work.
    """
    return "\n".join(
        f"  main:has(#{a}:target) ~ .contents a[href=\"#{a}\"] {{ background:#fff; "
        f"color:var(--ink); font-weight:600; box-shadow:inset 3px 0 0 var(--bad); }}"
        for a, _ in CONTENTS)


CONTENTS = (("verdict", "The verdict"), ("identity", "Identity"),
            ("task", "Task and data"), ("architecture", "Architecture"),
            ("training", "Training"), ("evaluation", "Evaluation"),
            ("intended-use", "Intended use and limits"), ("usage", "How to use"))


def build_card(src: Sources):
    """The whole document, and the problems that stopped it.

    Returns `(html_or_None, problems)`: the caller prints and exits, this function decides
    nothing about the process. A card is returned only when nothing is missing, because a card
    with a hole in it is worse than no card — it reads as a document whose author checked.
    """
    check = Checker()
    banner = verdict_banner(src, check)
    sections = [identity_section(src, check), task_section(src, check),
                architecture_section(src, check), training_section(src, check),
                evaluation_section(src, check), intended_use_section(src, check),
                usage_section(src, check)]
    problems = check.problems
    if problems:
        return None, problems

    man = src.manifest
    # The rail sits after `main` in the document so that a section reached by an anchor can
    # light its own entry — `main:has(#x:target) ~ nav a[href="#x"]` — without a line of
    # script. It is fixed-positioned, so document order costs nothing visually, and where
    # `:has()` is not supported the reader simply gets a rail with no highlight rather than a
    # page that does not work.
    entries = "".join(
        f"<li><a href=\"#{a}\">{esc(t)}</a></li>" for a, t in CONTENTS)
    contents = (
        "<nav class=\"contents\" aria-label=\"Contents\">"
        "<p class=\"rail-title\">Model card</p>"
        f"<ol>{entries}</ol>"
        "<div class=\"rail-foot\">"
        f"<a href=\"#top\">{esc(man.get('slug'))} {esc(man.get('run'))}</a>"
        "</div></nav>")
    head = (f"<h1>Model card</h1><p class=\"lede\">{esc(man.get('slug'))} "
            f"{esc(man.get('run'))} — the network this bundle serves, what it was "
            f"measured at, and what it must not be used for.</p>")
    foot = ("<footer><p>Every figure on this card was spliced by "
            "<code>scripts/model_card.py</code> from the model page's layer data and the "
            "bundle's own records — MANIFEST.json, schema.json, markers.json and "
            "blueprint-map.json — at the rounding each record holds. Nothing here was "
            "retyped. The only hand-written passage is <em>Intended use and limits</em>, and "
            "the card refuses to render if it contains a digit.</p>"
            "<p>This file is self-contained: no script, no stylesheet, no font and no image is "
            "loaded from anywhere. It can be read, printed and archived on a machine with no "
            "network and none of the skills that built it.</p></footer>")
    # every section ends with the way back, for a reader who followed a link into the middle
    sections = [sec + "<p class=\"totop\"><a href=\"#top\">↑ contents</a></p>"
                for sec in sections]
    body = ("<main id=\"top\">" + "".join([head, banner] + sections + [foot]) + "</main>"
            + contents)
    return (f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>Model card — {html.escape(str(man.get('slug')))} "
            f"{html.escape(str(man.get('run')))}</title>\n<style>{stylesheet()}</style>\n</head>\n"
            f"<body>\n{body}\n</body>\n</html>\n"), ()


def gather(bundle: Path, model: Path, intended: Path, post_eval: Path | None = None):
    """Read every record the card is spliced from. Returns `(Sources, errors)`."""
    errors = []
    try:
        model_html = model.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"--model {model} could not be read: {exc}"]
    layers = {}
    for name in LAYERS:
        doc, err = layer_data(model_html, name)
        if err:
            if name in REQUIRED_LAYERS:
                errors.append(err)
            continue
        layers[name] = doc
    manifest, err = read_json(bundle / "MANIFEST.json", "MANIFEST.json")
    if err:
        errors.append(err)
    schema, err = read_json(bundle / "schema.json", "schema.json")
    if err:
        errors.append(err)
    markers_doc, _ = read_json(bundle / "markers.json", "markers.json")
    prose, err = intended_use_text(intended)
    if err:
        errors.append(err)
    if errors:
        return None, errors
    layers["architecture"] = shipped_architecture(bundle, layers["architecture"], manifest)
    package = package_of(bundle, manifest)
    bmap, err = read_json(bundle / package / "blueprint-map.json", "blueprint-map.json")
    if err:
        errors.append(err)
        return None, errors
    markers = (markers_doc or {}).get("markers") or []
    post = {}
    if post_eval is not None:
        post, err = read_json(post_eval, str(post_eval))
        if err:
            return None, [err]
    return Sources(layers=layers, manifest=manifest, schema=schema, markers=tuple(markers),
                   blueprint_map=bmap, intended_use=prose, package=package,
                   post_eval=post or {}), []


def shipped_architecture(bundle: Path, arch: Mapping[str, Any], manifest: Mapping) -> dict:
    """The architecture record of the network the bundle ships (A44(b)). The layer describes the
    forged spec; a run the campaign selected off its ladder or its search is a delta on it, so the
    card reads the bundle's `netspec.json` (the selected run's resolved spec) and, when pack wrote
    that run's own smoke into `records/architecture.json` (`smoke_of`), its counts. The forged
    spec stays beside them as `forged_netspec` when the two differ."""
    out = dict(arch)
    spec, _ = read_json(bundle / "netspec.json", "netspec.json")
    if isinstance(spec, Mapping) and spec.get("blocks") and spec != arch.get("netspec"):
        out["forged_netspec"], out["netspec"] = arch.get("netspec"), spec
    record, _ = read_json(bundle / "records" / "architecture.json", "records/architecture.json")
    shipped = (record or {}).get("smoke_of") or {}
    if shipped.get("run") == manifest.get("run") and (record or {}).get("smoke"):
        out["smoke"], out["smoke_of"] = record["smoke"], shipped
    return out


REFUSAL_TAIL = (
    "A section the bundle legitimately does not have — no exports, no serving scaffold "
    "— is written on the card as absent, which is a fact about the bundle. An empty "
    "required section is not: this file is the only document a reader who never saw the chain "
    "will have, and a hole in it reads as an author who did not check.")

DIGIT_TAIL = (
    "Every other number on this card is spliced from a layer's data script or the bundle's own "
    "records, which is the only reason a reader without the chain can trust the page: each "
    "figure traces back to the record that measured it, at the rounding it is shown in. A digit "
    "typed into the hand-written passage would be the one number on the card that traces to "
    "nobody — and it is the number a reader would most want to check. Say it in words, or "
    "point at a table the card already splices: the ranges seen in training are printed "
    "directly beside that prose.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    ap.add_argument("--bundle", required=True, type=Path,
                    help="the bundle directory (MANIFEST.json, schema.json, markers.json, "
                         "<package>/blueprint-map.json)")
    ap.add_argument("--model", required=True, type=Path,
                    help="the evaluated model HTML, for its layer data scripts")
    ap.add_argument("--intended-use", required=True, type=Path,
                    help="bundle-layer.json, for readings.intended_use — the only prose on the "
                         "card the packer writes")
    ap.add_argument("--out", required=True, type=Path, help="where to write model-card.html")
    ap.add_argument("--post-eval", type=Path, default=None,
                    help="optional record of metrics recomputed from the stored test "
                         "predictions after the evaluation (spliced, with its source digest)")
    a = ap.parse_args()

    src, errors = gather(a.bundle, a.model, a.intended_use, a.post_eval)
    if errors:
        print("ERROR: model-card.html was not written — the card could not be spliced:")
        for line in errors:
            print(f"  - {line}")
        return 2

    digits = check_intended_use(src.intended_use)
    if digits:
        print("ERROR: model-card.html was not written — the hand-written section states a "
              "number:")
        for line in digits:
            print(f"  - {line}")
        print(f"  {DIGIT_TAIL}")
        return 2

    document, problems = build_card(src)
    if problems:
        print(f"ERROR: model-card.html was not written — {len(problems)} required "
              "section(s) could not be filled:")
        for problem in problems:
            print(f"  - {problem.line()}")
        print(f"  {REFUSAL_TAIL}")
        return 2

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(document, encoding="utf-8")
    size = a.out.stat().st_size
    print(f"OK: {a.out} — {size} bytes, self-contained, "
          f"{len(CONTENTS) - 1} section(s) plus the verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
