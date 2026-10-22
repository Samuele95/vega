"""
vega_core — the scripts' own domain and ports.

`compile_spec.py`, `emit_backend.py`, `conformance.py`, `nb_emit.py` and
`apply_architecture_layer.py` are the skill's tooling, and they are held to the same
discipline the code they emit is (`/architect`'s Model / Abstraction / Composition axes,
`code-contract.md` §4's anchors):

  * **Functional core, imperative shell.** Everything here is pure: it reads no argv, prints
    nothing, and exits nothing. A rule returns `Marker`s; the CLI shell prints and chooses the
    exit code. That is what makes the rules testable and the output one script's business.
  * **EJ-17 / DDD-ValueObject** — `Marker`, `Choice` and `Catalog` are frozen and compare by
    value; nothing mutates a spec, a catalog or a finding after construction.
  * **EJ-34** — `Level` is an enum; the string "ERROR" never crosses a function boundary.
  * **EJ-20 / HEX** — the chain's inputs (the model's layers, the shaped manifest, the
    blueprint's pipeline) arrive through one `ChainInputs` port-shaped value, so a rule never
    knows whether they came from a file, a fixture or a test.
  * **EJ-49** — every constructor validates and names the offender.
  * **EJ-15** — module-private helpers are `_`-prefixed; `__all__` is the surface.

No module-level mutable state lives here, by the same rule `conformance.py` enforces on the
emitted package.

Three siblings are part of this surface and re-exported from here, so every caller keeps one
import: `ChainInputs`, `SourceResolver` and the layer readers (`vega_chain.py`, the chain around
a spec read as values), `ShapeWalk` (`vega_shapes.py`, the torch-free axis walk behind compiler
rule G2) and `PackageAdapters` (`vega_adapters.py`, the one resolver of an emitted package's
adapters by blueprint-map role, D1 §12.1).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from vega_adapters import AdapterLoadError, PackageAdapters  # noqa: F401  (re-exported)
from vega_chain import (ChainInputs, SourceResolver, canonical_digest,  # noqa: F401
                        dig, find_in_list, model_layers)  # (re-exported)
from vega_shapes import ShapeFinding, ShapeWalk  # noqa: F401  (re-exported)

__all__ = ["Level", "Marker", "Markers", "Catalog", "ChainInputs", "Choice", "SourceResolver",
           "walk_blocks", "read_json", "model_layers", "dig", "find_in_list", "nearest",
           "canonical_digest",
           "ShapeWalk", "ShapeFinding", "PackageAdapters", "AdapterLoadError"]


class Level(str, Enum):
    """The marker vocabulary every Vega script prints. `ERROR` is what refuses."""

    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"
    FORK = "FORK"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Marker:
    """One line a script will print, with the rule and contract clause behind it.

    `rule` and `clause` are not decoration: a reader who disagrees with a refusal can go to
    the clause and argue with the contract rather than with the script.
    """

    level: Level
    message: str
    rule: str | None = None
    clause: str | None = None
    where: str | None = None

    def __post_init__(self):
        if not self.message:
            raise ValueError("a marker with no message says nothing; give it one")
        if not isinstance(self.level, Level):
            raise ValueError(f"level must be a Level, got {self.level!r}")

    def line(self, verbose: bool = False) -> str:
        head = f"{self.level}: "
        body = f"{self.where}: {self.message}" if self.where else self.message
        tail = ""
        if verbose and (self.rule or self.clause):
            tail = f"  [{self.rule or '-'}" + (f" · {self.clause}" if self.clause else "") + "]"
        return head + body + tail


@dataclass(frozen=True)
class Markers:
    """An immutable collection of markers. Rules return one; a shell prints it.

    Combining is a new value, never a mutation (EJ-17), so a rule cannot reach back and edit
    what an earlier rule found.
    """

    items: tuple = field(default_factory=tuple)

    @staticmethod
    def of(*markers: Marker) -> "Markers":
        return Markers(tuple(m for m in markers if m is not None))

    @staticmethod
    def gather(chunks: Iterable) -> "Markers":
        out: list = []
        for c in chunks:
            if c is None:
                continue
            out.extend(c.items if isinstance(c, Markers) else
                       (c if isinstance(c, (list, tuple)) else [c]))
        return Markers(tuple(out))

    def __add__(self, other) -> "Markers":
        return Markers.gather([self, other])

    def __iter__(self) -> Iterator[Marker]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def at(self, level: Level) -> tuple:
        return tuple(m for m in self.items if m.level is level)

    @property
    def errors(self) -> tuple:
        return self.at(Level.ERROR)

    @property
    def warns(self) -> tuple:
        return self.at(Level.WARN)

    @property
    def ok(self) -> bool:
        return not self.errors

    def lines(self, verbose: bool = False) -> list:
        order = {Level.ERROR: 0, Level.FORK: 1, Level.WARN: 2, Level.OK: 3}
        return [m.line(verbose) for m in sorted(self.items, key=lambda m: order[m.level])]

    def as_json(self) -> list:
        return [{"level": str(m.level), "message": m.message, "rule": m.rule,
                 "clause": m.clause, "where": m.where} for m in self.items]


def error(message, rule=None, clause=None, where=None) -> Marker:
    return Marker(Level.ERROR, message, rule, clause, where)


def warn(message, rule=None, clause=None, where=None) -> Marker:
    return Marker(Level.WARN, message, rule, clause, where)


def ok(message, rule=None, clause=None, where=None) -> Marker:
    return Marker(Level.OK, message, rule, clause, where)


# --------------------------------------------------------------------------- reading

def read_json(path, what: str):
    """(value, Marker|None) — never raises for a missing or malformed input, because the
    caller's job is to report it, not to crash on it."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, error(f"{what} not found: {path}", rule="IO")
    except (OSError, json.JSONDecodeError) as exc:
        return None, error(f"{what} could not be read ({path}): {exc}", rule="IO")


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nearest(word, known, within=3):
    """The nearest known name, so a typo is answered with the type the author meant."""
    best, dist = None, within + 1
    for k in known:
        d = _lev(str(word or ""), k)
        if d < dist:
            best, dist = k, d
    return best if dist <= within else None


# --------------------------------------------------------------------------- the catalog

@dataclass(frozen=True)
class Catalog:
    """`scripts/catalog.json` — the machine copy of `family-catalog.md` §1-§4.

    The catalog is data, not code: a new family, block or default is a row here and in the
    prose, and nothing in the scripts changes. `Catalog.load()` is the static factory
    (EJ-1); every accessor below answers a question a rule asks, so a rule never indexes
    raw JSON and a renamed key breaks in one place.
    """

    data: Mapping[str, Any]

    @staticmethod
    def load(path) -> tuple:
        raw, err = read_json(path, "catalog.json")
        if err is not None:
            return None, err
        for key in ("tasks", "families", "blocks", "defaults", "sources"):
            if key not in raw:
                return None, error(f"catalog.json has no {key!r} section; it is the machine "
                                   "copy of family-catalog.md §1-§4 and must carry all four",
                                   rule="C1")
        return Catalog(raw), None

    # -- rows

    def task(self, kind):
        return (self.data.get("tasks") or {}).get(kind)

    def family(self, name):
        return (self.data.get("families") or {}).get(name)

    def block(self, kind):
        return (self.data.get("blocks") or {}).get(kind)

    def default(self, name):
        return (self.data.get("defaults") or {}).get(name)

    @property
    def task_kinds(self) -> tuple:
        return tuple(sorted(self.data.get("tasks") or {}))

    @property
    def block_types(self) -> tuple:
        return tuple(sorted(self.data.get("blocks") or {}))

    @property
    def activations(self) -> tuple:
        return tuple(self.data.get("activations") or ())

    @property
    def structural_rows(self) -> tuple:
        return tuple(self.data.get("structural_catalog_rows") or ())

    @property
    def runtime_rows(self) -> tuple:
        return tuple(self.data.get("runtime_default_rows") or ())

    @property
    def decision_families(self) -> Mapping[str, Any]:
        return self.data.get("decision_families") or {}

    @property
    def source_patterns(self) -> Mapping[str, str]:
        return (self.data.get("sources") or {}).get("patterns") or {}

    def row(self, key):
        """Resolve a `catalog:` key of the form `<section>/<row>`; None when it is not there."""
        section, _, name = str(key).partition("/")
        table = {"defaults": "defaults", "families": "families", "blocks": "blocks",
                 "tasks": "tasks"}.get(section)
        if table is None:
            return None
        return (self.data.get(table) or {}).get(name.split("/")[0])

    def row_is_structural(self, key) -> bool:
        if str(key) in self.structural_rows:
            return True
        entry = self.row(key)
        return bool(isinstance(entry, Mapping) and entry.get("structural"))

    def row_is_runtime(self, key) -> bool:
        return str(key) in self.runtime_rows

    # -- the v2.3 sections (task fields, sequence bases, admission)

    @property
    def task_fields(self) -> Mapping[str, Any]:
        return self.data.get("task_fields") or {}

    @property
    def sequence_map(self) -> Mapping[str, str]:
        return self.data.get("sequence_map") or {}

    @property
    def admission(self) -> Mapping[str, Any]:
        return self.data.get("admission") or {}

    def families_on(self, shape: str) -> tuple:
        """The catalog families whose own sample shape is `shape`."""
        return tuple(sorted(n for n, f in (self.data.get("families") or {}).items()
                            if isinstance(f, Mapping) and f.get("sample_shape") == shape))

    def is_floor_on(self, family, shape) -> bool:
        """A floor family (`floor_shapes`) runs on shapes beyond its own (R-A13)."""
        row = self.family(family) or {}
        return shape in (row.get("floor_shapes") or ())




# --------------------------------------------------------------------------- choices

@dataclass(frozen=True)
class Choice:
    """One traced decision in the spec: where it sits, and the node that carries its
    `source` / `rationale` / `alternatives`."""

    where: str
    node: Mapping[str, Any]
    structural: bool = False
    inherits: bool = False

    @property
    def source(self):
        return self.node.get("source") if isinstance(self.node, Mapping) else None

    @property
    def catalog_rows(self) -> tuple:
        rows = self.node.get("catalog") if isinstance(self.node, Mapping) else None
        if rows is None:
            return ()
        return (rows,) if isinstance(rows, str) else tuple(rows)

    @property
    def alternatives(self):
        return self.node.get("alternatives") if isinstance(self.node, Mapping) else None

    @property
    def rationale(self):
        return self.node.get("rationale") if isinstance(self.node, Mapping) else None


def walk_blocks(blocks, prefix="") -> Iterator[tuple]:
    """(path, block, inherited) for every block, composites' `inner` included.

    `inherited` marks a block nested in a composite: netspec-contract §1 says it inherits its
    parent's provenance, so a rule must not demand a source of it.
    """
    for i, b in enumerate(blocks or []):
        if not isinstance(b, Mapping):
            yield f"{prefix}blocks[{i}]", b, bool(prefix)
            continue
        path = f"{prefix}blocks[{b.get('id', i)}]"
        yield path, b, bool(prefix)
        inner = (b.get("params") or {}).get("inner")
        if isinstance(inner, (list, tuple)):
            yield from walk_blocks(inner, prefix=f"{path}.inner.")
