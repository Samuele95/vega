"""
vega_train_core — the delta path syntax, the campaign's value objects, and the sibling lookup.

`plan_ladder.py`, `trainer.py`, `hpo.py` and `apply_training_layer.py` all speak in *deltas on
the spec*, so the syntax that expresses one, the rule that decides which decision family it
touches, and the code that applies it live here once. Same discipline as the code the chain
emits (`code-contract.md` §4) and as `/vega-forge`'s own tooling:

  * **Functional core, imperative shell.** Nothing here reads argv, prints or exits. `Delta`
    and `Edit` are frozen; applying a delta returns a new spec rather than mutating one, so a
    rung can be planned, checked and discarded without touching the campaign's `netspec.json`.
  * **EJ-17 / DDD-ValueObject** — `Edit`, `Delta`, `Rung` and `RunRecord` compare by value.
  * **EJ-1** — every one of them has a named static factory that validates: `Edit.parse`,
    `Delta.of`, `Rung.from_json`.
  * **EJ-49** — a malformed path is refused at construction, naming the segment that failed.
  * **EJ-20 / HEX** — the sibling skills are reached through `forge_scripts()`, one seam, so a
    relocated `/vega-forge` is a flag rather than an edit.

Contracts: vega-train/references/run-contract.md §3 (the delta path syntax and the one-family
rule), §4 (the ledger), training-catalog.md §2-§4.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

HERE = Path(__file__).resolve().parent
SKILL_DIR = HERE.parent
CATALOG = HERE / "catalog.json"

__all__ = ["Edit", "Delta", "Rung", "PathError", "forge_scripts", "load_catalog",
           "decision_family_of", "read_jsonl", "append_jsonl", "downsample", "digest_file",
           "digest_text"]

# a segment is either a plain key, or a list selector `key[field=value]`
_SELECTOR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([A-Za-z_][A-Za-z0-9_]*)=([^\]]+)\]$")

# the sections a delta may never touch: the backend is the forge's choice, not an experiment
FROZEN_SECTIONS = ("backend", "input", "schema", "outputs", "decision_families")


class PathError(ValueError):
    """A delta path that does not address anything, with the segment that failed named."""


# --------------------------------------------------------------------------- the siblings

def forge_scripts(override=None) -> Path:
    """`/vega-forge`'s scripts directory — the one seam to the sibling skill.

    `compile_spec.py`, `emit_backend.py`, `conformance.py` and `nb_emit.py` are the forge's,
    and this skill calls them rather than carrying a second copy of the compiler or the
    emitter (which is exactly the drift the contracts exist to prevent).
    """
    if override:
        path = Path(override)
        return path if path.name == "scripts" else path / "scripts"
    return SKILL_DIR.parent / "vega-forge" / "scripts"


def import_forge(name, override=None):
    """Import a module from `/vega-forge`'s scripts (`vega_core`, `compile_spec`, …)."""
    scripts = forge_scripts(override)
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return __import__(name)


def load_catalog(path=CATALOG):
    """The machine copy of `training-catalog.md` §2-§4."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"catalog.json not found at {path}"
    except json.JSONDecodeError as exc:
        return None, f"catalog.json is not valid JSON: {exc}"


# --------------------------------------------------------------------------- small helpers

def read_jsonl(path) -> list:
    """Every parseable line of a JSONL file; a truncated last line (an interrupted run) is
    dropped rather than raising, because a half-written epoch is not a defect in the reader."""
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_jsonl(path, record) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def downsample(values, limit=200) -> list:
    """At most `limit` points, endpoints kept — the layer caps each series and the full
    `history.jsonl` stays on disk (run-contract §5)."""
    values = list(values)
    if len(values) <= limit:
        return values
    step = (len(values) - 1) / (limit - 1)
    return [values[min(len(values) - 1, round(i * step))] for i in range(limit)]


def digest_text(text) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def digest_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------- the path

def _coerce(text):
    """A selector's value comes out of a path string; `kind=dropout` is a string but
    `id=3` is an int, and the spec's own value decides which comparison succeeds."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    if text in ("true", "false"):
        return text == "true"
    return text


def _segments(path: str) -> list:
    if not path or not isinstance(path, str):
        raise PathError(f"a delta path must be a non-empty string, got {path!r}")
    out = []
    for raw in path.split("/"):
        if raw == "":
            raise PathError(f"empty segment in the delta path {path!r}")
        match = _SELECTOR.match(raw)
        if match:
            out.append(("select", match.group(1), match.group(2), _coerce(match.group(3))))
        elif raw.isdigit():
            out.append(("index", int(raw)))
        else:
            out.append(("key", raw))
    return out


def _resolve(node, segments, path, *, create=False):
    """Walk to the parent of the last segment, returning (parent, accessor)."""
    for i, seg in enumerate(segments[:-1]):
        node = _step(node, seg, path, create=create)
        if node is None:
            raise PathError(f"the delta path {path!r} does not resolve at segment "
                            f"{i + 1} ({_render(seg)})")
    return node, segments[-1]


def _step(node, seg, path, *, create=False):
    kind = seg[0]
    if kind == "key":
        if not isinstance(node, Mapping):
            raise PathError(f"{path!r}: {seg[1]!r} addresses a key, but the value there is "
                            f"{type(node).__name__}")
        if seg[1] not in node and create:
            node[seg[1]] = {}
        return node.get(seg[1])
    if kind == "index":
        if not isinstance(node, list) or seg[1] >= len(node):
            raise PathError(f"{path!r}: index {seg[1]} is out of range")
        return node[seg[1]]
    _, key, field_name, value = seg
    if not isinstance(node, Mapping) or not isinstance(node.get(key), list):
        raise PathError(f"{path!r}: {key!r} is not a list, so {key}[{field_name}=…] does not "
                        "address anything")
    hit = next((x for x in node[key] if isinstance(x, Mapping) and x.get(field_name) == value),
               None)
    if hit is None:
        # `blocks/B3` is the common shorthand: a block addressed by its id
        raise PathError(f"{path!r}: no entry of {key!r} has {field_name} == {value!r}")
    return hit


def _render(seg):
    if seg[0] == "key":
        return seg[1]
    if seg[0] == "index":
        return str(seg[1])
    return f"{seg[1]}[{seg[2]}={seg[3]}]"


def _final(node, seg, path, *, create=False):
    """(container, key) for the last segment, so a caller can read or write it."""
    kind = seg[0]
    if kind == "key":
        if not isinstance(node, Mapping):
            raise PathError(f"{path!r}: the parent of {seg[1]!r} is {type(node).__name__}")
        if seg[1] not in node and not create:
            raise PathError(f"{path!r}: the spec has no key {seg[1]!r} there")
        return node, seg[1]
    if kind == "index":
        if not isinstance(node, list) or seg[1] >= len(node):
            raise PathError(f"{path!r}: index {seg[1]} is out of range")
        return node, seg[1]
    _, key, field_name, value = seg
    if not isinstance(node, Mapping) or not isinstance(node.get(key), list):
        raise PathError(f"{path!r}: {key!r} is not a list")
    for i, entry in enumerate(node[key]):
        if isinstance(entry, Mapping) and entry.get(field_name) == value:
            return node[key], i
    raise PathError(f"{path!r}: no entry of {key!r} has {field_name} == {value!r}")


def read_path(spec, path):
    """The value a delta path addresses. Raises PathError when it addresses nothing."""
    segments = _segments(path)
    if len(segments) == 1:
        container, key = _final(spec, segments[0], path)
    else:
        parent, last = _resolve(spec, segments, path)
        container, key = _final(parent, last, path)
    return container[key]


# a block addressed as `blocks/B3/...` is the id shorthand for `blocks[id=B3]/...`
_BLOCK_SHORTHAND = re.compile(r"^blocks/([A-Za-z_][A-Za-z0-9_]*)(/|$)")
_BLOCK_BRACKET = re.compile(r"^blocks\[([A-Za-z_][A-Za-z0-9_]*)\]")


def normalise_path(path: str) -> str:
    """`blocks/B3/params/units` -> `blocks[id=B3]/params/units`.

    The contract writes both forms; the shorthand is what a reader types and what the ladder
    prints, and the selector form is what actually addresses a list. Normalising here means
    every other function sees one syntax.
    """
    bracket = _BLOCK_BRACKET.match(path)
    if bracket:
        # `blocks[B1]/params/dims/zip` — the bare-id bracket form a human writes in a
        # defaults_to_test entry; the selector form is what actually addresses the list
        return f"blocks[id={bracket.group(1)}]" + path[bracket.end():]
    match = _BLOCK_SHORTHAND.match(path)
    if not match or match.group(1) in ("0", ""):
        return path
    rest = path[match.end(1):]
    return f"blocks[id={match.group(1)}]{rest}"


# --------------------------------------------------------------------------- decision families

DEFAULT_FAMILIES = {
    "architecture": ["family", "blocks"],
    "objective": ["loss", "metrics"],
    "optimization": ["optimizer", "schedule", "training", "init"],
    "regularization": ["regularization", "early_stopping"],
}


def decision_family_of(path: str, spec) -> str | None:
    """Which decision family of the spec a delta path falls under (run-contract §3).

    "One change" means one *decision*, not one key: a family swap legitimately touches both
    `family` and `blocks` because the spec's `decision_families` puts them in the same bucket.
    """
    head = normalise_path(path).split("/")[0].split("[")[0]
    families = (spec or {}).get("decision_families") or DEFAULT_FAMILIES
    for name, keys in families.items():
        if head in keys:
            return name
    return None


# --------------------------------------------------------------------------- the delta

@dataclass(frozen=True)
class Edit:
    """One edit of a delta: a path, the value it had, the value it takes."""

    path: str
    to: Any
    frm: Any = None
    _has_from: bool = field(default=False, repr=False)

    @staticmethod
    def parse(raw: Mapping[str, Any]) -> "Edit":
        if not isinstance(raw, Mapping) or "path" not in raw:
            raise PathError(f"an edit is an object with a `path` and a `to`, got {raw!r}")
        path = str(raw["path"])
        _segments(normalise_path(path))          # validates the syntax at construction
        if "to" not in raw:
            raise PathError(f"the edit at {path!r} carries no `to`; an edit that changes "
                            "nothing is not a rung")
        return Edit(path=path, to=raw["to"], frm=raw.get("from"), _has_from="from" in raw)

    def as_json(self) -> dict:
        out = {"path": self.path}
        if self._has_from:
            out["from"] = self.frm
        out["to"] = self.to
        return out

    @property
    def section(self) -> str:
        return normalise_path(self.path).split("/")[0].split("[")[0]


@dataclass(frozen=True)
class Delta:
    """A run's whole change to the spec: the edits, and the family they are allowed to touch.

    Applying one returns a *new* spec. The campaign's `netspec.json` is never edited — that is
    what makes a rung reversible and a ladder plannable before anything runs.
    """

    edits: tuple = field(default_factory=tuple)

    @staticmethod
    def of(raw: Iterable) -> "Delta":
        return Delta(tuple(Edit.parse(e) for e in (raw or [])))

    def as_json(self) -> list:
        return [e.as_json() for e in self.edits]

    def __len__(self):
        return len(self.edits)

    def __iter__(self):
        return iter(self.edits)

    @property
    def sections(self) -> tuple:
        return tuple(dict.fromkeys(e.section for e in self.edits))

    def families(self, spec) -> tuple:
        return tuple(dict.fromkeys(decision_family_of(e.path, spec) for e in self.edits))

    def touches_architecture(self, spec) -> bool:
        """Whether the emitter has to run again: an architecture delta changes the network."""
        return "architecture" in self.families(spec)

    # -- validation

    def problems(self, spec, declared_family=None, *, multi_family_ok=False) -> list:
        """Everything wrong with this delta against this spec, as sentences.

        The one-family rule and the frozen sections are checked here rather than in three
        scripts, because "one change means one decision" is the ladder's whole claim.

        `multi_family_ok` is the contract's one exception, and it is narrow: an HPO **trial**
        samples a space that was declared and gated before the study ran, and its source says
        so (`train:hpo/<study>`). An ablation never gets it — a rung that moves the optimizer
        and the dropout at once tells you nothing about either.
        """
        out = []
        for edit in self.edits:
            if edit.section in FROZEN_SECTIONS:
                out.append(f"{edit.path}: `{edit.section}` may not be changed by a run — "
                           + ("the backend is the forge's choice, not an experiment"
                              if edit.section == "backend" else
                              "it identifies the spec rather than deciding anything"))
        if not self.edits:
            return out
        families = [f for f in self.families(spec)]
        unknown = [e.path for e in self.edits if decision_family_of(e.path, spec) is None]
        for path in unknown:
            out.append(f"{path}: falls under no decision family of the spec "
                       f"({', '.join((spec or {}).get('decision_families') or DEFAULT_FAMILIES)})")
        known = [f for f in families if f is not None]
        if len(set(known)) > 1 and not multi_family_ok:
            out.append(f"the delta touches {len(set(known))} decision families "
                       f"({', '.join(sorted(set(known)))}) — one change means one decision, "
                       "so this is two rungs, not one")
        if declared_family and known and declared_family not in known:
            out.append(f"the rung declares family {declared_family!r} but its delta touches "
                       f"{', '.join(sorted(set(known)))}")
        for edit in self.edits:
            try:
                current = read_path(spec, normalise_path(edit.path))
            except PathError as exc:
                out.append(str(exc))
                continue
            if edit._has_from and current != edit.frm:
                out.append(f"{edit.path}: the delta says it changes {edit.frm!r}, but the spec "
                           f"has {current!r} there — the rung was planned against a different "
                           "spec")
        return out

    # -- application

    def apply(self, spec) -> dict:
        """A new spec with the edits applied, in order. The input is never modified."""
        out = copy.deepcopy(spec)
        for edit in self.edits:
            path = normalise_path(edit.path)
            segments = _segments(path)
            if len(segments) == 1:
                container, key = _final(out, segments[0], path, create=True)
            else:
                parent, last = _resolve(out, segments, path, create=True)
                container, key = _final(parent, last, path, create=True)
            container[key] = copy.deepcopy(edit.to)
        return out

    def resolved(self, spec) -> "Delta":
        """The same delta with every `from` filled in from the spec, so the record says what
        actually changed rather than what the planner assumed."""
        filled = []
        for edit in self.edits:
            try:
                current = read_path(spec, normalise_path(edit.path))
            except PathError:
                current = None
            filled.append(Edit(edit.path, edit.to, current, _has_from=True))
        return Delta(tuple(filled))

    def describe(self) -> str:
        """One line for the ladder table and the ledger."""
        if not self.edits:
            return "(none — the spec as engineered)"
        return "; ".join(f"{e.path}: {_short(e.frm)} → {_short(e.to)}" if e._has_from
                         else f"{e.path} → {_short(e.to)}" for e in self.edits)


def _short(value, width=28):
    if value is None:
        return "·"
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return text if len(text) <= width else text[:width - 1] + "…"


# --------------------------------------------------------------------------- the ladder

@dataclass(frozen=True)
class Rung:
    """One planned run: what it changes, what it claims, and what would show the claim held."""

    id: str
    kind: str
    delta: Delta
    hypothesis: str = ""
    source: str = ""
    expect: str = ""
    family: str | None = None
    study: str | None = None
    space: str | None = None
    budget: Mapping[str, Any] | None = None
    note: str | None = None
    expect_check: Mapping[str, Any] | None = None    # expectations.py's grammar, as written

    @staticmethod
    def from_json(raw: Mapping[str, Any]) -> "Rung":
        if not isinstance(raw, Mapping) or not raw.get("id"):
            raise PathError(f"a ladder run needs an id, got {raw!r}")
        return Rung(id=str(raw["id"]), kind=raw.get("kind", "ablation"),
                    delta=Delta.of(raw.get("delta")), hypothesis=raw.get("hypothesis", ""),
                    source=raw.get("source", ""), expect=raw.get("expect", ""),
                    family=raw.get("family"), study=raw.get("study"), space=raw.get("space"),
                    budget=raw.get("budget"), note=raw.get("note"),
                    expect_check=raw.get("expect_check"))

    def as_json(self) -> dict:
        out = {"id": self.id, "kind": self.kind}
        if self.family:
            out["family"] = self.family
        if self.study:
            out["study"] = self.study
        if self.space:
            out["space"] = self.space
        out["delta"] = self.delta.as_json()
        out["hypothesis"] = self.hypothesis
        out["source"] = self.source
        out["expect"] = self.expect
        if self.expect_check is not None:
            out["expect_check"] = self.expect_check
        if self.budget:
            out["budget"] = self.budget
        if self.note:
            out["note"] = self.note
        return out

    @property
    def is_trial(self) -> bool:
        return self.kind == "hpo" and "-" in self.id


SOURCE_RE = re.compile(
    r"^(netspec(:[A-Za-z0-9_\[\]=/\-]+)?|vega:ladder/[a-z\-]+|train:diagnostic/D\d+"
    r"|train:hpo/[A-Za-z0-9_\-]+|analysis:F\d+|user:.+)$")


def rung_source_ok(source: str) -> bool:
    """The closed set of run-contract §3. A rung whose source is outside it is a run nobody
    can trace back to a reason."""
    return bool(SOURCE_RE.match(str(source or "")))
