"""
fact_sink — `FactSink`, where the fact collectors of `/vega-show` put what they read, and the
two small values they share: `Catalogs` (the three catalogs a fact is checked against) and
`Refusal` (a preflight gate that stops the extraction).

A fact is a value read at a ref (contract §2): the sink never computes one. `take()` reads the
first of a fact's candidate refs that resolves in the bundle's `RecordStore` and adds it under
its key, checked against its kind:

- `count`, `epoch` and `count_of` are whole numbers; `count_of` is the one derivation, the length
  of the list the ref points at (`classes_n`, `runs_n`);
- `metric`, `size` and `param` are finite numbers (`param` also takes a word: `gelu`, `cosine`);
- `ci_metric` and `diff` read an estimate `{value, ci, n}` (kept to those three fields) or a bare
  number; an estimate with no point (a per-element vector) is not one;
- `list` is a list of finite numbers (a per-unit vector: `f1_per_class.values`); `labels` a list
  of names, read as it stands or as the `name` of each object of a list (`data_contract.features`);
- `text`, `report_number` and `report_page` take the value verbatim.

A ref that resolves to a value of the wrong type is skipped with a `WARN:` naming it, so a record
whose shape moved is heard, not silently dropped. A ref names a mapping key or selector value only
when the ref grammar can spell it (`spellable`): a class called `DoS Hulk` has no ref into a
mapping keyed by class names, and its fact is skipped with a `WARN:`. The format of a fact follows
from its kind (`FORMATS`), except that an estimate `{value, ci, n}` is `ci` whatever its kind.

A key is spelled from record ids (`step_s1_rows_after`, `run_h1_5_primary`), so two ids can
spell one key (`S1` and `s1`, `H1-5` and `H1_5`) or an id a key too long for the contract's
pattern: `take()` keeps the first fact, and skips the second, or the unspellable one, with a
`WARN:` naming both refs — a record never stops the extraction with a traceback. `add()`, which
the collectors call with keys they spell themselves, still refuses a key taken twice with a
`ValueError` (D5 §7.3): that is a defect of the extractor, not of the records.
`build()` returns the `FactSet` with the groups in the catalog's `fact_groups` order.

**A non-finite number is never a value of the fact set** (JSON cannot spell `NaN` or
`Infinity`, and `facts.json` is written without them). A fact read at a non-finite value is
skipped with a `WARN:` naming its ref (above); a group that copies a record value inline (a
run's `eligible`, a study's `best`, an input's `origin`) keeps null in its place, with a `WARN:`
naming the group path (`group()`). `nonfinite()` lists where a JSON value holds one, for the
gates that refuse instead (`extract_facts`: the bundle's manifest, the fact set as written).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from show_ports import RecordStore
from vega_show_core import (CATALOG, HERE, Fact, FactMissing, FactRef, FactSet, ShowError,
                            TaskShape, fact_key)

__all__ = ["FactSink", "Catalogs", "Refusal", "FORMATS", "spellable", "peek", "keys_in", "key_of",
           "nonfinite"]

FORMATS = MappingProxyType({"count": "count", "size": "count", "metric": "metric",
                            "ci_metric": "metric", "diff": "metric", "param": "param",
                            "epoch": "epoch", "list": "list", "labels": "list",
                            "count_of": "count", "report_number": "verbatim",
                            "report_page": "verbatim", "text": "verbatim"})
_WHOLE = ("count", "epoch")
_ESTIMATE = ("ci_metric", "diff")
_VERBATIM = ("text", "report_number", "report_page")
_ESTIMATE_FIELDS = ("value", "ci", "n")
# what a ref can spell as a mapping key (a name segment) and as a selector's value
_NAME_OK = re.compile(r"[^\s\[\].=#]+")
_VALUE_OK = re.compile(r"[^\s\[\]#]+")
_SKILLS = HERE.parent.parent


class Refusal(ShowError):
    """A preflight gate of the extraction refused its input (`PF1`–`PF8`): printed as
    `ERROR: <gate> <where>: <why>`, exit 2, never worked around."""

    def __init__(self, gate: str, where: str, message: str) -> None:
        self.gate, self.where, self.message = gate, str(where), message
        super().__init__(f"{gate} {self.where}: {message}")


@dataclass(frozen=True)
class Catalogs:
    """The vega-show catalog, and the vega-forge and vega-eval catalogs a fact is checked
    against (tasks, families, blocks, sample shapes, metric directions, metric sets)."""

    show: Mapping
    forge: Mapping
    eval: Mapping

    @classmethod
    def load(cls, skills: Path | None = None) -> "Catalogs":
        root = Path(skills) if skills is not None else _SKILLS
        forge = json.loads((root / "vega-forge/scripts/catalog.json").read_text(encoding="utf-8"))
        path = root / "vega-eval/scripts/catalog.json"
        evals = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return cls(CATALOG, MappingProxyType(forge), MappingProxyType(evals))

    def direction(self, metric: str | None) -> str | None:
        """`max` or `min` from the vega-forge `metric_direction`; None for any other entry."""
        found = (self.forge.get("metric_direction") or {}).get(str(metric))
        return found if found in ("max", "min") else None


def spellable(text, *, selector: bool = False) -> bool:
    """Whether a ref can spell `text` as a mapping key (or, `selector=True`, a selector value)."""
    return bool((_VALUE_OK if selector else _NAME_OK).fullmatch(str(text)))


def peek(store: RecordStore, ref: str, default=None):
    """The value at `ref`, or `default` when it does not resolve: for reading structure (ids,
    kinds, names) that becomes no fact."""
    try:
        return store.read(FactRef.parse(ref))
    except FactMissing:
        return default


class FactSink:
    """The facts, groups, names and markers one extraction collects."""

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._facts: dict[str, Fact] = {}
        self._groups: dict[str, object] = {}
        self._names: set[str] = set()
        self._markers: list[str] = []
        self.name(names)

    def __contains__(self, key: str) -> bool:
        return key in self._facts

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def value(self, key: str):
        """The JSON value of a fact already taken."""
        return self._facts[key].json_value

    def add(self, key: str, ref, value, kind: str, direction: str | None = None) -> str:
        """Add one fact; a key taken twice is a `ValueError`."""
        if key in self._facts:
            raise ValueError(f"fact key {key!r} is taken twice ({self._facts[key].ref} and {ref})")
        fmt = "ci" if isinstance(value, Mapping) else FORMATS[str(kind)]
        self._facts[key] = Fact(key, value, ref if isinstance(ref, FactRef) else FactRef.parse(ref),
                                kind, fmt, direction)
        return key

    def take(self, store: RecordStore, key, refs, kind: str,
             direction: str | None = None) -> str | None:
        """Read the first candidate ref that resolves to a value of `kind` and add it under
        `key` (a key, or the parts `fact_key` joins into one: `("run", rid, "primary")`); None
        when none does (the fact is absent, as the contract allows), and None with a `WARN:`
        when the parts spell no key or another ref already holds it."""
        for ref in ([refs] if isinstance(refs, str) else refs):
            if ref is None:
                continue
            try:
                raw = store.read(FactRef.parse(ref))
            except FactMissing:
                continue
            value, why = _as_kind(raw, kind)
            if why:
                self.warn(f"{ref} {why}; no fact {_spelled(key)} from it")
                continue
            if value is not None:
                spelled = self._vacant(key, ref)
                return spelled if spelled is None or spelled in self._facts else \
                    self.add(spelled, ref, value, kind, direction)
        return None

    def vacant(self, key: str, ref: str) -> bool:
        """Whether a fact read at `ref` may take `key`: a key another ref already holds is a
        `WARN:` naming both, and no second fact."""
        spelled = self._vacant(key, ref)
        return spelled is not None and spelled not in self._facts

    def _vacant(self, key, ref: str) -> str | None:
        """The key to add under (the held one when `ref` is the ref that holds it); None, with a
        `WARN:`, when the parts spell no key or another ref holds it."""
        spelled = key if isinstance(key, str) else key_of(key)
        if spelled is None:
            self.warn(f"{ref}: its ids spell no fact key the contract admits "
                      f"({CATALOG['fact_key_pattern']}); no fact from it")
            return None
        held = self._facts.get(spelled)
        if held is not None and str(held.ref) != str(ref):
            self.warn(f"fact key {spelled!r} is spelled by both {held.ref} and {ref} (two ids "
                      f"that differ only in case or punctuation); the second is no fact")
            return None
        return spelled

    def group(self, name: str, value) -> None:
        """Set a group; a non-finite number inside it becomes null, with a `WARN:` naming
        its path (`groups.hpo.0.best`)."""
        if name not in CATALOG["fact_groups"]:
            raise ValueError(f"{name!r} is not a fact group of the catalog "
                             f"({CATALOG['fact_groups']})")
        bad = nonfinite(value, f"groups.{name}")
        for path in bad:
            self.warn(f"{path}: a non-finite number read from the records is no value of the "
                      "fact set; it is null")
        self._groups[name] = _finite_only(value) if bad else value

    def name(self, tokens: Iterable) -> None:
        """Whitelist names for the digit rule (empty and non-string values are skipped)."""
        self._names.update(str(t) for t in tokens if isinstance(t, (str, int)) and str(t).strip())

    def warn(self, message: str) -> None:
        line = message if message.startswith("WARN:") else f"WARN: {message}"
        if line not in self._markers:
            self._markers.append(line)

    def build(self, task: TaskShape, header: Mapping | None = None) -> FactSet:
        order = CATALOG["fact_groups"]
        groups = {g: self._groups[g] for g in order if g in self._groups}
        return FactSet(tuple(self._facts.values()), groups, task, frozenset(self._names),
                       dict(header or {}))


def _as_kind(raw, kind: str):
    """(value, None) when `raw` reads as `kind`; (None, why) when it cannot; (None, None) for
    a null, which is no fact and no warning."""
    if raw is None:
        return None, None
    if kind == "count_of":
        return (len(raw), None) if isinstance(raw, list) else (None, "is not a list to count")
    if kind in _ESTIMATE:
        return _estimate(raw)
    if kind == "labels":
        return _labels(raw)
    if kind == "list":
        ok = isinstance(raw, list) and all(_finite(v) for v in raw)
        return (raw, None) if ok else (None, "is not a list of finite numbers")
    if kind in _VERBATIM:
        ok = isinstance(raw, str) or _finite(raw)
        return (raw, None) if ok else (None, "is not a string or a finite number")
    return _scalar(raw, kind)


def _scalar(raw, kind: str):
    if kind in _WHOLE:
        ok = isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0
        return (raw, None) if ok else (None, f"is not a whole number for a {kind} fact")
    if kind == "param" and isinstance(raw, (str, bool)):
        return (raw, None) if isinstance(raw, bool) or raw.strip() else (None, "is an empty word")
    if _finite(raw):
        return raw, None
    return None, f"is not a finite number for a {kind} fact"


def _estimate(raw):
    if _finite(raw):
        return raw, None
    if not isinstance(raw, Mapping):
        return None, "is not an estimate {value, ci, n}"
    if raw.get("value") is None:
        return None, None
    doc = {k: raw[k] for k in _ESTIMATE_FIELDS if k in raw}
    ci, n = doc.get("ci"), doc.get("n")
    if not _finite(doc["value"]) or (ci is not None and not (isinstance(ci, list)
                                                            and all(_finite(v) for v in ci))) \
            or (n is not None and not _finite(n)):
        return None, "is an estimate with a non-finite point, interval or n"
    return doc, None


def _labels(raw):
    if not isinstance(raw, list):
        return None, "is not a list of names"
    names = [v.get("name") if isinstance(v, Mapping) else v for v in raw]
    ok = all(isinstance(n, (str, int)) and not isinstance(n, bool) for n in names)
    return (names, None) if ok else (None, "is not a list of names")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def key_of(parts) -> str | None:
    """`fact_key(parts)`, or None when the parts spell no key the contract admits (an id long
    enough to push the key past its pattern's 64 characters)."""
    try:
        return fact_key(parts)
    except ValueError:
        return None


def _spelled(key) -> str:
    """A key, or its parts as `fact_key` would join them, for a message."""
    return key if isinstance(key, str) else key_of(key) or "_".join(map(str, key))


def keys_in(sink: FactSink, keys: Sequence[str | None]) -> list[str]:
    """The keys of `keys` the sink holds, in order (a limit's or a group's fact list)."""
    return [k for k in keys if k is not None and k in sink]


def nonfinite(value, path: str = "") -> list[str]:
    """The paths (`a.b.0.c`, below `path`) where a JSON value holds a NaN or an infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return [path or "(root)"]
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        return []
    return [p for key, v in items for p in nonfinite(v, f"{path}.{key}" if path else str(key))]


def _finite_only(value):
    """`value` with every non-finite number replaced by None."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {k: _finite_only(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_only(v) for v in value]
    return value
