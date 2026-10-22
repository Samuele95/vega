"""
memory_stances — the `show:` stances a draft applies: the lines of
`.claude/domain-forge-memory.md` § Network stances that settle a presentation choice for this
dataset (contract §9, R-E4, R-E5). A recorded stance is applied and cited, never re-asked.

A stance line is the catalog's `stances.lines` form, one decision per line:

    YYYY-MM-DD | <dataset> | show: roles <class> <role> (<basis>), … (user)
    YYYY-MM-DD | <dataset> | show: motif <id> (user)
    YYYY-MM-DD | <dataset> | show: lexicon <key> <one>/<many> (user)
    YYYY-MM-DD | <dataset> | show: showcase of <slug>-<run>, …

`MemoryStances.read(path, datasets)` keeps the lines of the § Network stances section (a
markdown heading named by `catalog.stances.section`; HTML comments, where the memory template
keeps its examples, are skipped; a leading `- ` or `* ` bullet is allowed) whose dataset is one
of `datasets` (compared case-insensitively) and whose body starts with the `show:` prefix. Each
becomes a `Stance` value; the kind is the body's first word. `latest(kind)` is the stance a draft
applies: the latest date, and the later line of one date (the memory is appended to). The body
of each kind is parsed by the proposer that owns the choice (`propose_roles`, `propose_motif`,
`propose_lexicon`), which cites the stance as `show:<date>` (`Stance.basis`).

A memory file that is absent is no stance (a first run); one that cannot be read, or a `show:`
line of this dataset whose kind the catalog does not know, is a `WARN:` in `markers`, never a
traceback: a stance the draft cannot apply leaves its fork open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from vega_show_core import CATALOG

__all__ = ["Stance", "MemoryStances", "USER_SUFFIX", "datasets_of"]

_SPEC = CATALOG["stances"]
_PREFIX = _SPEC["prefix"]
_KINDS = tuple(_SPEC["lines"])
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING = re.compile(r"^#+\s+(.*?)\s*#*\s*$")
_LINE = re.compile(r"^\s*(?:[-*]\s+)?(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$")
# the closing "(user)" of a line the user settled; the proposers strip it before parsing
USER_SUFFIX = re.compile(r"\s*\(user\)\s*$")


@dataclass(frozen=True)
class Stance:
    """One `show:` line of this dataset: its date, dataset, kind, the body after the kind, and
    the line as the memory holds it (quoted in the plan as the stance)."""

    date: str
    dataset: str
    kind: str
    body: str
    line: str

    @property
    def basis(self) -> str:
        """The basis a choice this stance settles cites (`show:<stance date>`, §3.5)."""
        return f"show:{self.date}"


class MemoryStances:
    """The `show:` stances of one dataset, in memory order, and the warnings reading them gave."""

    def __init__(self, stances: Sequence[Stance] = (), markers: Sequence[str] = ()) -> None:
        self._stances = tuple(stances)
        self._markers = tuple(markers)

    @property
    def stances(self) -> tuple[Stance, ...]:
        return self._stances

    @property
    def markers(self) -> tuple[str, ...]:
        return self._markers

    @classmethod
    def read(cls, path, datasets: Iterable[str]) -> "MemoryStances":
        """The stances of `datasets` in the memory file at `path` (None: no memory)."""
        if path is None or not Path(path).is_file():
            return cls()
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return cls((), [f"WARN: the memory {path} could not be read ({exc}); no show: stance "
                            "is applied and every fork stays open"])
        return cls.from_text(text, datasets)

    @classmethod
    def from_text(cls, text: str, datasets: Iterable[str]) -> "MemoryStances":
        wanted = {str(d).strip().lower() for d in datasets if str(d).strip()}
        stances, markers = [], []
        for line in _section_lines(text):
            found = _LINE.match(line)
            if not found or found.group(2).lower() not in wanted:
                continue
            body = found.group(3)
            if not body.startswith(_PREFIX):
                continue
            kind, _, rest = body[len(_PREFIX):].strip().partition(" ")
            if kind not in _KINDS:
                markers.append(f"WARN: the memory line {line.strip()!r} is a show: stance of an "
                               f"unknown kind {kind!r} (known: {', '.join(_KINDS)}); not applied")
                continue
            stances.append(Stance(found.group(1), found.group(2), kind, rest.strip(),
                                  _clean(line)))
        return cls(stances, markers)

    def of_kind(self, kind: str) -> list[Stance]:
        """Every stance of `kind`, oldest first (a stable sort by date keeps the file order)."""
        return sorted((s for s in self._stances if s.kind == kind), key=lambda s: s.date)

    def latest(self, kind: str) -> Stance | None:
        """The stance of `kind` a draft applies: the latest by date, the last of one date."""
        found = self.of_kind(kind)
        return found[-1] if found else None


def datasets_of(header: Mapping) -> list[str]:
    """The names a stance line may give this dataset, from the fact set's header: the model
    page's file stem up to its first dot (`turbine.domain….pack.html` → `turbine`), and the
    bundle's slug."""
    names = []
    model = str(((header or {}).get("model") or {}).get("path") or "")
    if model:
        names.append(Path(model).name.split(".")[0])
    slug = ((header or {}).get("bundle") or {}).get("slug")
    if isinstance(slug, str) and slug:
        names.append(slug)
    return [n for n in dict.fromkeys(names) if n]


def _section_lines(text: str) -> list[str]:
    """The lines of the § Network stances section, comments removed."""
    out, inside = [], False
    for line in _COMMENT.sub("", text).splitlines():
        heading = _HEADING.match(line)
        if heading:
            inside = heading.group(1).strip().lower() == _SPEC["section"].lower()
            continue
        if inside and line.strip():
            out.append(line)
    return out


def _clean(line: str) -> str:
    return re.sub(r"^\s*[-*]\s+", "", line).strip()

