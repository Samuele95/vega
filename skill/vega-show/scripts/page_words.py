"""
page_words — what a literal of the page template may say without being prose (D4 §17.1 check
2), and the page's strings table as data.

`Vocabulary.of(template)` is the set of words `page_lint.literals` accepts outside the strings
table:

- the CSS classes and custom properties the template's sheets define (`page.css`, `css/*.css`),
  and every other word their declarations use (properties, keywords, functions, fonts): a word
  quoted in a sheet (`content: "…"`) is not harvested;
- the shell's DOM: `index.html`'s tag and attribute names, its ids, classes and `data-*` values;
- the DOM, SVG, ARIA, event, canvas and media-query names below (`aria-*` and `data-*` by
  pattern);
- the keys of the strings table (a whole literal: `app.status.ready`, a dynamic prefix
  `metric.`) and of the backend's explain catalog (block types, activations, links).

`strings_table(template)` reads the table the way `js/core/strings.en.js` merges it: the area
fragments `js/core/strings/<area>.en.js` it imports, in import order, each exporting `STRINGS`
(key → text) and `DYNAMIC` (prefix → members) as plain data — strings, arrays, objects,
`Object.freeze(…)`, `+` between strings. Anything else in a table (a call, a variable) is a
`TableError`, and so is a key two fragments define (the page's load error).
"""
from __future__ import annotations

import html.parser
import json
import re
from dataclasses import dataclass
from pathlib import Path

from js_tokens import significant, tokenize_file

__all__ = ["Vocabulary", "StringsTable", "TableError", "strings_table", "css_words", "EXPLAIN",
           "PSEUDO", "MEDIA_WORDS"]

SKILLS = Path(__file__).resolve().parents[2]
EXPLAIN = SKILLS / "vega-forge" / "scripts" / "backends" / "pytorch" / "explain-catalog.json"
TABLE = "js/core/strings.en.js"
FRAGMENT_RE = re.compile(r"^\./strings/([a-z-]+)\.en\.js$")

_DOM = """
a abbr address area article aside audio b bdi bdo blockquote body br button canvas caption cite code col
colgroup data datalist dd del details dfn dialog div dl dt em embed fieldset figcaption figure footer form
h1 h2 h3 h4 h5 h6 head header hgroup hr html i iframe img input ins kbd label legend li link main map mark
menu meta meter nav noscript object ol optgroup option output p picture pre progress q rp rt ruby s samp
script search section select slot small source span strong style sub summary sup table tbody td template
textarea tfoot th thead time title tr track u ul var video wbr
svg g path circle ellipse line polyline polygon rect text tspan defs use symbol marker clipPath mask pattern
linearGradient radialGradient stop filter title desc foreignObject image
id class style lang dir hidden tabindex role href src rel target type name value placeholder alt width
height min max step form disabled checked selected readonly required multiple accept download media
charset content viewBox fill stroke stroke-width stroke-linecap stroke-linejoin stroke-dasharray
stroke-opacity fill-opacity opacity d cx cy r rx ry x y x1 y1 x2 y2 dx dy points transform text-anchor
dominant-baseline font-size font-family font-weight preserveAspectRatio xmlns open colspan rowspan scope
spellcheck autocomplete inputmode draggable contenteditable noopener noreferrer _blank _self
beforebegin afterbegin beforeend afterend
click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave pointerdown pointerup
pointermove pointerenter pointerleave pointercancel wheel keydown keyup keypress input change submit focus
blur focusin focusout resize scroll load error hashchange popstate visibilitychange animationend
transitionend dragover drop dragenter dragleave touchstart touchmove touchend contextmenu message toggle
alert alertdialog application banner cell checkbox columnheader complementary dialog document feed grid
gridcell group heading img list listbox listitem log marquee math menu menubar menuitem navigation none
note presentation progressbar radio radiogroup region row rowgroup rowheader separator slider spinbutton
status switch tab tablist tabpanel term textbox timer toolbar tooltip tree treegrid treeitem polite
assertive
left right center start end top bottom middle alphabetic hanging ideographic round butt square miter bevel
source-over lighter multiply screen destination-out destination-in copy nonzero evenodd
prefers-reduced-motion reduce no-preference prefers-color-scheme dark light max-width min-width
max-height min-height orientation landscape portrait hover pointer fine coarse screen print
px em rem ch ex vh vw vmin vmax ms s deg rad turn fr var calc min max clamp rgb rgba hsl hsla url attr
linear-gradient radial-gradient conic-gradient translate translateX translateY scale scaleX scaleY rotate
matrix inherit initial unset revert auto normal bold bolder lighter italic oblique solid dashed dotted
double transparent currentColor monospace sans-serif serif system-ui ui-monospace ui-sans-serif ui-serif
cursive fantasy tabular-nums nowrap pre-wrap ellipsis block inline inline-block flex inline-flex grid
contents absolute relative fixed sticky visible pointer-events user-select
""".split()

# a selector's pseudo-classes and pseudo-elements: words only right after `:` or `::`
PSEUDO = frozenset("""
first-child last-child nth-child nth-last-child first-of-type last-of-type nth-of-type nth-last-of-type
only-child only-of-type not is where has hover focus focus-visible focus-within active visited link checked
disabled enabled empty root target before after placeholder selection marker backdrop first-line
first-letter defined open indeterminate invalid valid required optional read-only read-write
""".split())
# a media query's connectives: words only inside a literal that is a media query
MEDIA_WORDS = frozenset({"and", "not", "only", "or", "all", "screen", "print"})


class TableError(ValueError):
    """The strings table is not plain data, or two fragments define one key."""


@dataclass(frozen=True)
class StringsTable:
    strings: dict          # key → text, merged
    dynamic: dict          # prefix → [members], merged
    areas: dict            # key → the area that defines it


def _value(toks: list, i: int, where: str):
    """(value, next index) of the plain-data expression at toks[i]."""
    t = toks[i]
    if t.kind == "string" or (t.kind == "template" and not t.inner):
        value, i = t.value, i + 1
        while i + 1 < len(toks) and toks[i].is_("punct", "+"):
            nxt, i = _value(toks, i + 1, where)
            if not isinstance(nxt, str):
                raise TableError(f"{where}:{t.line}: '+' joins a string to something else")
            value += nxt
        return value, i
    if t.kind == "number":
        return float(t.text.replace("_", "")), i + 1
    if t.kind in ("ident", "keyword") and t.text in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[t.text], i + 1
    if t.is_("ident", "Object") and [x.text for x in toks[i + 1:i + 4]] == [".", "freeze", "("]:
        value, i = _value(toks, i + 4, where)
        return value, _expect(toks, i, ")", where) + 1
    if t.is_("punct", "["):
        return _array(toks, i + 1, where)
    if t.is_("punct", "{"):
        return _object(toks, i + 1, where)
    raise TableError(f"{where}:{t.line}: {t.text!r} is not plain data (a table holds strings, arrays, objects)")


def _expect(toks: list, i: int, text: str, where: str) -> int:
    if i >= len(toks) or not toks[i].is_("punct", text):
        got = toks[i].text if i < len(toks) else "the end"
        raise TableError(f"{where}: expected {text!r}, found {got!r}")
    return i


def _array(toks: list, i: int, where: str):
    out = []
    while not toks[i].is_("punct", "]"):
        value, i = _value(toks, i, where)
        out.append(value)
        i += 1 if toks[i].is_("punct", ",") else 0
    return out, i + 1


def _object(toks: list, i: int, where: str):
    out = {}
    while not toks[i].is_("punct", "}"):
        key = toks[i]
        if key.kind not in ("string", "ident", "keyword", "number"):
            raise TableError(f"{where}:{key.line}: {key.text!r} is not a key")
        i = _expect(toks, i + 1, ":", where)
        out[key.value if key.kind == "string" else key.text], i = _value(toks, i + 1, where)
        i += 1 if toks[i].is_("punct", ",") else 0
    return out, i + 1


def exported(path: Path, name: str):
    """The plain-data value of `export const <name> = …` in one module (None when absent)."""
    toks = significant(tokenize_file(path))
    for i, t in enumerate(toks[:-3]):
        if t.is_("keyword", "const") and toks[i + 1].is_("ident", name) and toks[i + 2].is_("punct", "="):
            return _value(toks, i + 3, str(path))[0]
    return None


def fragments(template: Path) -> list[tuple[str, Path]]:
    """(area, path) of every fragment the table imports, in import order."""
    out = []
    for t in significant(tokenize_file(template / TABLE)):
        m = FRAGMENT_RE.match(t.value or "") if t.kind == "string" else None
        if m:
            out.append((m.group(1), template / "js" / "core" / "strings" / f"{m.group(1)}.en.js"))
    if not out:
        raise TableError(f"{template / TABLE} imports no fragment ./strings/<area>.en.js")
    return out


def strings_table(template: Path) -> StringsTable:
    strings, dynamic, areas, prefixes = {}, {}, {}, {}
    for area, path in fragments(Path(template)):
        for key, text in (exported(path, "STRINGS") or {}).items():
            if key in areas:
                raise TableError(f"strings: STRINGS key {key!r} is defined by both {areas[key]} and {area}")
            strings[key], areas[key] = text, area
        for prefix, members in (exported(path, "DYNAMIC") or {}).items():
            if prefix in prefixes:
                raise TableError(f"strings: DYNAMIC key {prefix!r} is defined by both {prefixes[prefix]} and {area}")
            dynamic[prefix], prefixes[prefix] = list(members), area
    return StringsTable(strings, dynamic, areas)


# ---- the sheets and the shell -----------------------------------------------------------------------
QUOTED_RE = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")
FONT_PROPERTY_RE = re.compile(r"font(?:-family)?|--(?:sans|mono|serif|font[\w-]*)")


def css_words(css: str) -> tuple[set, set, set]:
    """(classes, custom properties, declaration words) of a stylesheet: the classes its
    selectors define, the custom properties it defines or reads, and the property names,
    keywords, functions and fonts of its declarations (a quoted family name of a font
    declaration included). Selectors' other words, at-rule preludes, keyframe stops (`from`,
    `to`), any other quoted text and comments are not harvested."""
    css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    bare = QUOTED_RE.sub(" ", css)
    classes = {c for m in re.finditer(r"([^{};]*)\{", bare)
               for c in re.findall(r"\.(-?[A-Za-z_][\w-]*)", m.group(1))}
    props, words = set(re.findall(r"--([\w-]+)", bare)), set()
    for block in re.findall(r"\{([^{}]*)\}", css):
        for decl in block.split(";"):
            name, colon, value = decl.partition(":")
            if colon:
                name = name.strip()
                words.add(name)
                text = value if FONT_PROPERTY_RE.fullmatch(name) else QUOTED_RE.sub(" ", value)
                words.update(re.findall(r"(?<![\w#-])([A-Za-z][\w-]*)", text))
    return classes, props, {w for w in words if w}


class _Shell(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.words = set()

    def handle_starttag(self, tag, attrs):
        self.words.add(tag)
        for name, value in attrs:
            self.words.add(name)
            if name in ("id", "class", "role", "type") or name.startswith("data-"):
                self.words.update((value or "").split())


def _explain_keys(path: Path) -> set:
    try:
        cat = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {k for part in ("blocks", "activations", "links") for k in (cat.get(part) or {})}


@dataclass(frozen=True)
class Vocabulary:
    words: frozenset       # lower-cased
    classes: frozenset
    keys: frozenset        # table keys, their bases, dynamic prefixes, explain keys

    @classmethod
    def of(cls, template: Path, explain: Path = EXPLAIN) -> "Vocabulary":
        template = Path(template)
        classes, words = set(), {w.lower() for w in _DOM}
        for sheet in [template / "page.css", *sorted((template / "css").glob("*.css"))]:
            if sheet.is_file():
                c, p, w = css_words(sheet.read_text(encoding="utf-8"))
                classes |= c
                words |= {x.lower() for x in c | p | w}
        shell = _Shell()
        if (template / "index.html").is_file():
            shell.feed((template / "index.html").read_text(encoding="utf-8"))
        words |= {w.lower() for w in shell.words}
        table = strings_table(template)
        bases = {re.split(r"[@#]", k)[0] for k in table.strings}
        explain_keys = _explain_keys(explain)
        keys = set(table.strings) | bases | set(table.dynamic) | explain_keys | {f"explain.{k}" for k in explain_keys}
        return cls(frozenset(words | {k.lower() for k in explain_keys}), frozenset(classes), frozenset(keys))

    def allows(self, word: str) -> bool:
        low = word.lower()
        return low in self.words or bool(re.fullmatch(r"(?:aria|data)-[a-z][a-z0-9-]*", low))
