"""
vega_show_core — the value objects of `/vega-show`, its digests and its JSON edges: frozen
values and pure functions, no argv, no printing, no exits; every script of the stage imports
it, and it imports none of them.

- **The closed sets are the catalog's.** `FactKind`, `ClassRole`, `Motif`, `LabelKind` and
  `CheckStatus` are built at import from `scripts/catalog.json`, the machine copy of
  `references/show-contract.md`, so a member the contract gains or drops needs no edit here.
- **A fact is a value with the ref it was read from.** `FactRef` is the contract's §2 grammar
  (`records/shape.json#manifest.steps[id=S4].rows_after.all`, `report:net.forward.number`),
  `Fact` a §2 key with its value, ref, kind and format, `FactSet` `vega/show-facts@1` as one
  value whose `to_json()` and `from_json()` are exact inverses.
- **A value is checked where it is made.** `Fact` and `TaskShape` refuse what
  `show-facts-1.schema.json` refuses (its enums, types and patterns, read from the schema that
  CS2 keeps equal to the catalogs), so a `FactSet` always writes a valid `facts.json`.
- **JSON inside a value is frozen.** `freeze()` makes it hashable (`FrozenJson` for an object,
  a tuple for a list) and `thaw()` is its inverse (EJ-17); they live in `frozen_json.py`.
- **What the stage writes, others read.** `write_json` gives a file the mode `open()` gives
  (0666 less the umask), never `mkstemp`'s 0600: the serving image runs as a non-root user.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from frozen_json import FrozenJson, freeze, thaw

__all__ = ["HERE", "CATALOG", "FactKind", "ClassRole", "Motif", "LabelKind", "CheckStatus",
           "ShowError", "FactMissing", "FrozenJson", "freeze", "thaw", "FactRef", "Fact",
           "TaskShape", "FactSet", "Problem", "ShowPaths", "digest_file", "digest_bytes",
           "canonical_json", "read_json", "write_json", "fact_key", "now"]

HERE = Path(__file__).resolve().parent
CATALOG: Mapping = MappingProxyType(json.loads((HERE / "catalog.json").read_text(encoding="utf-8")))


def _closed_set(name: str, members) -> type:
    """A StrEnum whose members are a catalog list, spelled as the catalog spells them."""
    return StrEnum(name, [(str(m).upper().replace("-", "_"), str(m)) for m in members])


FactKind = _closed_set("FactKind", CATALOG["fact_kinds"])
ClassRole = _closed_set("ClassRole", CATALOG["roles"]["set"])
Motif = _closed_set("Motif", CATALOG["motifs"]["set"])
# `other` is no report kind: a label whose anchor and name carry no kind prefix; no role takes it
LabelKind = _closed_set("LabelKind", [*CATALOG["label_kinds"]["kinds"], "other"])
CheckStatus = _closed_set("CheckStatus", CATALOG["check_status"])

_KEY_RE = re.compile(CATALOG["fact_key_pattern"])
_REF_RE = re.compile(CATALOG["ref_pattern"])
_SEG = r"(?P<name>[^\[\].=]+)(?:\[(?P<key>[^\[\]=]+)=(?P<value>[^\[\]]*)\])?"
_SEG_RE = re.compile(_SEG + r"(?=\.|$)")
_PATH_RE = re.compile("{0}(?:\\.{0})*".format(re.sub(r"\?P<\w+>", "", _SEG)))
_REPORT = "report:"
_FACTS_SCHEMA = HERE.parent / CATALOG["schemas"]["vega/show-facts@1"]["file"]


@functools.cache
def _facts_schema() -> Mapping:
    """`show-facts-1.schema.json`, read once, on the first value that needs it."""
    return MappingProxyType(json.loads(_FACTS_SCHEMA.read_text(encoding="utf-8")))


# the JSON Schema type of a Python value, first match (a tuple is an array, as a frozen list is)
_JSON_TYPES = (("null", type(None)), ("boolean", bool), ("integer", int), ("number", float),
               ("string", str), ("array", (list, tuple)), ("object", Mapping))


def _schema_problem(value, spec: Mapping) -> str | None:
    """Why `value` breaks one flat property of the schema (an `enum`, else a `type` with an
    optional `minLength` and `pattern`); None when it does not."""
    if "enum" in spec:
        return None if value in spec["enum"] else f"{value!r} is not one of {spec['enum']}"
    types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
    kind = next((k for k, t in _JSON_TYPES if isinstance(value, t)), type(value).__name__)
    if kind not in types and not (kind == "integer" and "number" in types):
        return f"{value!r} is not {' or '.join(types)}"
    if kind == "string" and (len(value) < spec.get("minLength", 0)
                             or ("pattern" in spec and not re.search(spec["pattern"], value))):
        return f"{value!r} does not match {spec.get('pattern') or 'a non-empty string'}"
    return None


class ShowError(RuntimeError):
    """A refusal of the stage, never worked around: the input is fixed upstream. `BundleAltered`
    (records.py) and `ReportUnreadable` (report_labels.py) are its adapter-side kinds."""


class FactMissing(ShowError, LookupError):
    """A ref that does not resolve: a missing key, index or match, or an unpinned file."""

    def __init__(self, ref, reason: str) -> None:
        self.ref, self.reason = str(ref), reason
        super().__init__(f"{self.ref}: {reason}")


# --------------------------------------------------------------------------- facts

@dataclass(frozen=True)
class FactRef:
    """Where a fact was read: `file#path`, or `report:<role>.<number|page|title>`."""

    file: str
    path: str

    def __post_init__(self):
        if not _REF_RE.fullmatch(str(self)):
            raise ValueError(f"not a fact ref of the contract's grammar: {str(self)!r}")
        if not self.is_report:
            self.segments()

    @classmethod
    def parse(cls, text: str) -> "FactRef":
        text = str(text)
        if text.startswith(_REPORT):
            return cls(_REPORT, text[len(_REPORT):])
        file, sep, path = text.partition("#")
        if not sep:
            raise ValueError(f"a fact ref names its file and path as file#path: {text!r}")
        return cls(file, path)

    def __str__(self) -> str:
        return f"{_REPORT}{self.path}" if self.file == _REPORT else f"{self.file}#{self.path}"

    @property
    def is_report(self) -> bool:
        return self.file == _REPORT

    def report_field(self) -> tuple[str, str]:
        """(role, field) of a `report:` ref: `report:net.forward.page` is ("net.forward", "page")."""
        if not self.is_report:
            raise ValueError(f"{self} is not a report ref")
        role, _, which = self.path.rpartition(".")
        return role, which

    def segments(self) -> tuple[tuple[str, str | None, str | None], ...]:
        """The path as `(name, key, value)` triples (`key`, `value` from `name[key=value]`)."""
        if self.is_report or not _PATH_RE.fullmatch(self.path):
            raise ValueError(f"{self} has no path of the grammar seg(.seg)*")
        return tuple((m["name"], m["key"], m["value"]) for m in _SEG_RE.finditer(self.path))


@dataclass(frozen=True)
class Fact:
    """One number, name list or label the page may show, with the ref it came from."""

    key: str
    value: object
    ref: FactRef
    kind: FactKind
    fmt: str
    direction: str | None = None

    def __post_init__(self):
        if not _KEY_RE.fullmatch(self.key):
            raise ValueError(f"fact key {self.key!r} does not match {CATALOG['fact_key_pattern']}")
        if self.fmt not in CATALOG["formats"]:
            raise ValueError(f"fact {self.key}: format {self.fmt!r} is not one of {CATALOG['formats']}")
        spec = _facts_schema()["$defs"]["fact"]["properties"]["direction"]
        if self.direction is not None and (why := _schema_problem(self.direction, spec)):
            raise ValueError(f"fact {self.key}: direction {why}")
        object.__setattr__(self, "kind", FactKind(self.kind))
        object.__setattr__(self, "value", freeze(self.value))
        if isinstance(self.ref, str):
            object.__setattr__(self, "ref", FactRef.parse(self.ref))

    @property
    def json_value(self):
        return thaw(self.value)

    def to_json(self) -> dict:
        doc = {"value": self.json_value, "ref": str(self.ref), "kind": str(self.kind),
               "format": self.fmt}
        if self.direction is not None:
            doc["direction"] = self.direction
        return doc

    @classmethod
    def from_json(cls, key: str, doc: Mapping) -> "Fact":
        return cls(key, doc["value"], FactRef.parse(doc["ref"]), FactKind(doc["kind"]),
                   doc["format"], doc.get("direction"))


@dataclass(frozen=True, kw_only=True)
class TaskShape:
    """The `task` object of `vega/show-facts@1`, its fields in the schema's order and checked
    against the schema's enums, types and patterns: `kind`, `goal`, `sample_shape`, `modality`,
    `family`, `view_key`, `primary` and `direction` are never None. `modality` defaults to
    `sample_shape`; `classes` and `outcomes` are None or a list, kept as a tuple of strings;
    `from_json` takes every field of the schema and no other."""

    kind: str
    base: str | None = None
    goal: str
    sample_shape: str
    graph_level: str | None = None
    modality: str | None = None
    family: str
    view_key: str
    label: str | None = None
    classes: tuple[str, ...] | None = None
    outcomes: tuple[str, ...] | None = None
    primary: str
    direction: str

    def __post_init__(self):
        for name in ("classes", "outcomes"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (list, tuple)):
                raise ValueError(f"task.{name} is a list of strings, not {value!r}")
            object.__setattr__(self, name, None if value is None else tuple(map(str, value)))
        if self.modality is None:
            object.__setattr__(self, "modality", self.sample_shape)
        spec = _facts_schema()["properties"]["task"]["properties"]
        problems = [f"{f.name} {why}" for f in fields(self)
                    if (why := _schema_problem(getattr(self, f.name), spec[f.name]))]
        if problems:
            raise ValueError("not a task of vega/show-facts@1: " + "; ".join(problems))

    def to_json(self) -> dict:
        doc = {f.name: getattr(self, f.name) for f in fields(self)}
        doc.update({n: list(doc[n]) for n in ("classes", "outcomes") if doc[n] is not None})
        return doc

    @classmethod
    def from_json(cls, doc: Mapping) -> "TaskShape":
        names = [f.name for f in fields(cls)]
        missing, extra = [n for n in names if n not in doc], sorted(set(doc) - set(names))
        if missing or extra:
            raise ValueError(f"not a task of vega/show-facts@1: missing {missing}, unknown {extra}")
        return cls(**{n: doc[n] for n in names})


@dataclass(frozen=True)
class FactSet:
    """`vega/show-facts@1` as one value: the facts, the name whitelist, the groups, the task."""

    facts: tuple[Fact, ...]
    groups: FrozenJson
    task: TaskShape
    names: frozenset[str]
    header: FrozenJson = field(default_factory=FrozenJson)  # extracted_at, bundle, model

    def __post_init__(self):
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "names", frozenset(map(str, self.names)))
        object.__setattr__(self, "groups", freeze(dict(self.groups)))
        object.__setattr__(self, "header", freeze(dict(self.header)))
        keys = [f.key for f in self.facts]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate fact keys: {sorted({k for k in keys if keys.count(k) > 1})}")

    def get(self, key: str) -> Fact:
        found = next((f for f in self.facts if f.key == key), None)
        if found is None:
            raise FactMissing(key, "no such fact key in the fact set")
        return found

    def keys(self) -> frozenset[str]:
        return frozenset(f.key for f in self.facts)

    def to_json(self) -> dict:
        return {"schema": "vega/show-facts@1", **thaw(self.header), "task": self.task.to_json(),
                "facts": {f.key: f.to_json() for f in self.facts}, "names": sorted(self.names),
                "groups": thaw(self.groups)}

    @classmethod
    def from_json(cls, doc: Mapping) -> "FactSet":
        if doc.get("schema") != "vega/show-facts@1":
            raise ValueError(f"not vega/show-facts@1: schema {doc.get('schema')!r}")
        header = {k: doc[k] for k in ("extracted_at", "bundle", "model") if k in doc}
        facts = tuple(Fact.from_json(k, v) for k, v in (doc.get("facts") or {}).items())
        return cls(facts, doc.get("groups") or {}, TaskShape.from_json(doc["task"]),
                   frozenset(doc.get("names") or ()), header)


@dataclass(frozen=True)
class Problem:
    """One finding of a gate: `ERROR: V2 authored.headline: <why>` when printed."""

    rule: str
    where: str
    message: str
    level: str = "ERROR"

    def __post_init__(self):
        if f"{self.level}:" not in CATALOG["markers"]:
            raise ValueError(f"problem level {self.level!r} is not a marker of {CATALOG['markers']}")

    def __str__(self) -> str:
        return f"{self.level}: {self.rule} {self.where}: {self.message}"

    def to_json(self) -> dict:
        return {"rule": self.rule, "where": self.where, "message": self.message, "level": self.level}


def _showcase_file(schema: str) -> str:
    """The showcase-relative path the catalog files under a schema id."""
    return next(p for p, spec in CATALOG["showcase"]["files"].items() if spec["schema"] == schema)


@dataclass(frozen=True)
class ShowPaths:
    """The directories one run of the stage reads and writes, beside one bundle."""

    bundle: Path
    showcase: Path
    building: Path
    previous: Path

    @classmethod
    def of(cls, bundle) -> "ShowPaths":
        root, spec = Path(bundle), CATALOG["showcase"]
        return cls(root, root / spec["dir"], root / spec["siblings"]["building"],
                   root / spec["siblings"]["previous"])

    bundle_manifest = property(lambda self: self.bundle / "MANIFEST.json")
    manifest = property(lambda self: self.showcase / CATALOG["showcase"]["manifest"])
    facts = property(lambda self: self.showcase / _showcase_file("vega/show-facts@1"))
    plan = property(lambda self: self.showcase / _showcase_file("vega/story-plan@2"))
    story = property(lambda self: self.showcase / _showcase_file("vega/story@2"))
    page_config = property(lambda self: self.showcase / _showcase_file("vega/page@1"))
    page = property(lambda self: self.showcase / "page")
    work = property(lambda self: self.showcase / ".work")
    smoke = property(lambda self: self.showcase / ".smoke")
    docs = property(lambda self: self.showcase / "docs")
    brief = property(lambda self: self.docs / "show-brief.md")
    draft = property(lambda self: self.work / "story-plan.draft.json")
    markers = property(lambda self: self.work / "markers.json")
    checks = property(lambda self: self.work / "checks.json")
    layer_data = property(lambda self: self.work / "showcase-layer.json")


# --------------------------------------------------------------------------- digests, JSON IO

def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_file(path) -> str:
    with Path(path).open("rb") as fh:
        return "sha256:" + hashlib.file_digest(fh, "sha256").hexdigest()


def canonical_json(value) -> bytes:
    """One byte string per JSON value: sorted keys, no spaces, UTF-8, no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def read_json(path, what: str) -> tuple[dict | None, str | None]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"{what} not found: {path}"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{what} could not be read ({path}): {exc}"


def write_json(path, doc) -> str:
    """Write `doc` atomically (a temporary sibling, then a rename); return its digest. The file
    gets the mode a plain `open()` gives a new file (0666 less the umask), so a bundle packed
    by one user stays readable by the one that serves it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return digest_bytes(data)


def fact_key(parts) -> str:
    """The contract's §2 key of an id or a sequence of parts, joined by `_`: lower case, every
    run of characters outside `[a-z0-9]` one `_` (`H1-5` → `h1_5`, `("step", "S4")` → `step_s4`)."""
    parts = (parts,) if isinstance(parts, str) else tuple(parts)
    key = re.sub(r"[^a-z0-9]+", "_", "_".join(map(str, parts)).lower()).strip("_")
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"{parts!r} gives the key {key!r}, which does not match "
                         f"{CATALOG['fact_key_pattern']}")
    return key


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
