"""
report_labels — `AuxLabelSource`, the `LabelSource` adapter over a LaTeX `.aux` file.

The report's own numbering is read where LaTeX wrote it: every

    \\newlabel{name}{{number}{page}{title}{anchor}{…}}

of the `.aux` becomes a `LabelRef`. A section renumbered by an edit of the report moves here on
the next read, instead of going stale inside a sentence someone typed. This is v2.2 pack's
`report_labels` parser, moved to vega-show (contract §6.4) and made kind-aware:

- `@cref` entries (cleveref's shadow labels) are skipped;
- groups nest (`{The \\num {2830743} rows}` is one title) and an escaped brace (`\\{`, `\\}`)
  is text, not structure;
- a label without hyperref's groups (`{{number}{page}}`) has an empty title and anchor;
- the page is an int when it is a decimal number, else the text LaTeX wrote (`iv`);
- the kind comes from the anchor prefix (`equation.3` → equation, `figure.caption.37` →
  figure, `subsection.2.1` → section), then from the label prefix (`eq:`, `fig:`, `tab:`,
  `sec:`), both tables the catalog's `label_kinds`; a label neither names is `other`, which
  no role can take;
- a label defined twice keeps its last definition, as LaTeX does.

The file is read once: `digest()` is the sha256 of exactly the bytes that were parsed.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from show_ports import LabelRef, LabelSource
from vega_show_core import CATALOG, LabelKind, ShowError, digest_bytes

__all__ = ["AuxLabelSource", "ReportUnreadable"]

_NEWLABEL = "\\newlabel{"
_SKIP_SUFFIX = "@cref"
_ANCHOR_KINDS = tuple(CATALOG["label_kinds"]["anchor_prefixes"].items())
_LABEL_KINDS = tuple(CATALOG["label_kinds"]["label_prefixes"].items())


class ReportUnreadable(ShowError):
    """The report's `.aux` cannot be read, or does not parse (unbalanced braces)."""


class AuxLabelSource(LabelSource):
    """The labels of one compiled LaTeX report, read from its `.aux`: the `LabelSource` port's
    adapter, declared as one."""

    def __init__(self, aux: Path) -> None:
        self._path = Path(aux)
        try:
            raw = self._path.read_bytes()
        except OSError as exc:
            raise ReportUnreadable(f"the report's .aux could not be read ({self._path}): {exc}") from exc
        self._digest = digest_bytes(raw)
        text = raw.decode("utf-8", errors="replace")
        try:
            self._labels = MappingProxyType(_parse_labels(text))
        except ValueError as exc:
            raise ReportUnreadable(f"{self._path}: {exc}") from exc

    @property
    def path(self) -> Path:
        return self._path

    def labels(self) -> Mapping[str, LabelRef]:
        return self._labels

    def digest(self) -> str:
        return self._digest


def _parse_labels(text: str) -> dict[str, LabelRef]:
    out, i = {}, 0
    while (i := text.find(_NEWLABEL, i)) >= 0:
        name, j = _group(text, i + len(_NEWLABEL) - 1)
        i = j
        if name.endswith(_SKIP_SUFFIX) or j >= len(text) or text[j] != "{":
            continue
        body, i = _group(text, j)
        parts = _groups(body)
        if len(parts) >= 2:
            out[name] = _label(name, parts)
    return out


def _label(name: str, parts: list[str]) -> LabelRef:
    number, page = parts[0].strip(), parts[1].strip()
    title = parts[2] if len(parts) > 2 else ""
    anchor = parts[3].strip() if len(parts) > 3 else ""
    return LabelRef(label=name, number=number,
                    page=int(page) if page.isascii() and page.isdecimal() else page,
                    title=title, kind=_kind(name, anchor), anchor=anchor)


def _kind(name: str, anchor: str) -> LabelKind:
    for prefix, kind in _ANCHOR_KINDS:
        if anchor.startswith(prefix):
            return LabelKind(kind)
    for prefix, kind in _LABEL_KINDS:
        if name.startswith(prefix):
            return LabelKind(kind)
    return LabelKind("other")


def _groups(body: str) -> list[str]:
    """The top-level `{…}` groups of a label body, in order (text between them is skipped)."""
    parts, k = [], 0
    while k < len(body):
        if body[k] == "\\":
            k += 2
        elif body[k] == "{":
            part, k = _group(body, k)
            parts.append(part)
        else:
            k += 1
    return parts


def _group(text: str, k: int) -> tuple[str, int]:
    """The content of the `{…}` group opening at `text[k]`, and the index after it."""
    depth, i = 0, k
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return text[k + 1:i], i + 1
        i += 1
    raise ValueError(f"unbalanced braces in the .aux from offset {k}")
