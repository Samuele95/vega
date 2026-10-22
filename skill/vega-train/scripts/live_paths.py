"""
live_paths — which delta paths the emitted code reads, and the dead deltas that read nothing.

A rung exists to move the weights. A delta that lands on a value no emitted code reads trains
its baseline's numbers bit for bit and reads as "this default does not matter" — the R6-equals-
R0 symptom. The forge's rule R2 (netspec-contract §3) makes `regularization[]` the one
declaration of dropout, weight decay, label smoothing and gradient clipping, derives the live
paths the code reads from it, and records them in `spec_snapshot.json` under `read_paths`;
this module reads that record and names the edits of a delta that no emitted code reads:

  * an edit on a `regularization[]` entry — `regularization[kind=<k>]/…`,
    `regularization/<i>/…`, or a whole `regularization` list whose `<k>` entry it changes —
    whose declaration is not among the read paths (a kind no code reads, a dropout that covers
    no block reading a rate, label smoothing under a loss that takes none);
  * an edit on a live path a declaration covers (`_resolved.regularization.shadowed_paths`,
    or a live path recorded as derived from a declaration): the code reads the declaration,
    so the change belongs there.

Any other path is outside R2's domain and is not judged here (the compiler and the one-family
rule judge it). A run directory whose snapshot records no `read_paths` was emitted before R2:
its templates read, of `regularization[]`, only `kind: grad_clip` (netspec-contract §3 as
built before R2), and that is what is used. A directory with no snapshot at all has no record
of its emission: the paths are derived from its `netspec.json` by the forge's own derivation
(`vega_regularization.py`, through the one seam to the sibling skill). Either way the source
is said, never silent (`ReadPaths.source`).

Pure apart from `of_campaign`, which reads the files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from vega_train_core import Delta, PathError, import_forge, normalise_path, read_path

__all__ = ["ReadPaths", "PRE_R2", "of_snapshot", "of_campaign", "dead_edits", "dead_path"]

REG = "regularization"
_KIND = re.compile(r"^regularization\[kind=([A-Za-z_][A-Za-z0-9_]*)\]")

#: what the templates read of the declaration before rule R2 (netspec-contract §3, as built)
PRE_R2 = ("regularization[kind=grad_clip]/value",)


class ReadPaths(NamedTuple):
    """The paths the emitted code reads (`read`), the live paths a declaration covers
    (`derived`: live path → the declaration, or "" when the record does not say which), and
    where the record came from."""

    read: frozenset
    derived: Mapping[str, str]
    source: str
    recorded: bool


def _norm(path: str) -> str:
    """A recorded path in the delta path syntax: the contract's dotted spelling
    (`optimizer.params.weight_decay`) becomes slashes outside a `[…]` selector."""
    text = str(path)
    if "/" not in text:
        text = re.sub(r"\.(?![^\[]*\])", "/", text)
    return normalise_path(text)


def _strings(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def _pair(read: set, derived: dict, declaration: str | None, lives) -> None:
    if declaration:
        read.add(_norm(declaration))
    for live in lives:
        live = _norm(live)
        read.add(live)
        if declaration and live != _norm(declaration):
            derived[live] = _norm(declaration)


def of_snapshot(snapshot: Mapping | None) -> ReadPaths:
    """The read paths a `spec_snapshot.json` records — a list of paths, a map of declaration
    to its live path(s), or a list of such entries — or the pre-R2 set when it records none."""
    snap = snapshot or {}
    raw = snap.get("read_paths")
    if raw is None:
        raw = (snap.get("_resolved") or {}).get("read_paths")
    if raw is None:
        return ReadPaths(frozenset(_norm(p) for p in PRE_R2), {},
                         "the pre-R2 templates (spec_snapshot.json records no read_paths): of "
                         "regularization[] only grad_clip is read", False)
    read, derived = set(), {}
    shadowed = (((snap.get("_resolved") or {}).get("regularization")) or {}).get(
        "shadowed_paths") or []
    for path in _strings(shadowed):
        derived[_norm(path)] = ""
    entries = raw.items() if isinstance(raw, Mapping) else enumerate(raw or [])
    for key, value in entries:
        if isinstance(raw, Mapping):
            names = [str(key), *_strings(value)]
        elif isinstance(value, Mapping):
            names = [s for v in value.values() for s in _strings(v)]
        else:
            names = _strings(value)
        declaration = next((n for n in names if n.startswith(REG)), None)
        _pair(read, derived, declaration, [n for n in names if n != declaration])
    return ReadPaths(frozenset(read), derived, "spec_snapshot.json read_paths", True)


def of_campaign(root, forge=None) -> ReadPaths:
    """The read paths of the campaign's emitted code (`<root>/spec_snapshot.json`); with no
    snapshot, the forge's derivation on `<root>/netspec.json`."""
    path = Path(root) / "spec_snapshot.json"
    if path.is_file():
        try:
            return of_snapshot(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return of_snapshot(None)
    try:
        spec = json.loads((Path(root) / "netspec.json").read_text(encoding="utf-8"))
        derive = import_forge("vega_regularization", forge)
        doc = {"read_paths": derive.read_paths(spec),
               "_resolved": {"regularization": derive.snapshot(spec)}}
    except (OSError, json.JSONDecodeError, ImportError, AttributeError):
        return of_snapshot(None)
    return of_snapshot(doc)._replace(
        source="the forge's R2 derivation on netspec.json (the directory holds no "
               "spec_snapshot.json)", recorded=False)


def _entry_read(entry: str, paths: ReadPaths) -> bool:
    return any(p == entry or p.startswith(entry + "/") for p in paths.read)


def _declared_read(kind: str, paths: ReadPaths, spec=None) -> bool:
    """Whether the code reads the `kind` declaration, by its selector or by its index."""
    if _entry_read(f"{REG}[kind={kind}]", paths):
        return True
    rows = (spec or {}).get(REG) or []
    return any(_entry_read(f"{REG}/{i}", paths) for i, row in enumerate(rows)
               if isinstance(row, Mapping) and str(row.get("kind")) == kind)


def _covered(norm: str, paths: ReadPaths) -> str | None:
    hit = next((p for p in paths.derived if norm == p or norm.startswith(p + "/")), None)
    return hit


def dead_path(path: str, paths: ReadPaths, *, changed_kinds=None, spec=None) -> str | None:
    """Why `path` is dead under `paths`, or None when emitted code reads it (or R2 does not
    govern it). `changed_kinds` are the kinds a whole-list `regularization` edit changes;
    `spec` resolves a `regularization/<i>` index to its kind."""
    norm = normalise_path(str(path))
    covered = _covered(norm, paths)
    if covered is not None:
        by = paths.derived[covered]
        return (f"no emitted code reads {path}: "
                + (f"the compiler derives it from {by}" if by else
                   "a regularization[] declaration covers it and the code reads the declaration")
                + ", so the change belongs on the declaration")
    if norm == REG:
        dead = [k for k in (changed_kinds or ()) if not _declared_read(k, paths, spec)]
        return (f"no emitted code reads regularization[kind={dead[0]}] "
                f"({', '.join(dead)} changed by the edit on {path})") if dead else None
    match = _KIND.match(norm)
    if match and not _declared_read(match.group(1), paths, spec):
        return f"no emitted code reads {path}"
    index = re.match(r"^regularization/([0-9]+)(?:/|$)", norm)
    if index and not _entry_read(f"{REG}/{index.group(1)}", paths):
        rows = (spec or {}).get(REG) or []
        i = int(index.group(1))
        kind = rows[i].get("kind") if i < len(rows) and isinstance(rows[i], Mapping) else None
        if not (kind and _declared_read(str(kind), paths, spec)):
            return f"no emitted code reads {path}"
    return None


def _changed_kinds(before, after) -> list:
    def by_kind(entries):
        return {str(e.get("kind")): e for e in (entries or []) if isinstance(e, Mapping)}
    old, new = by_kind(before), by_kind(after)
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def dead_edits(delta: Delta, spec: Mapping[str, Any], paths: ReadPaths) -> list:
    """`[(edit path, why)]` for every edit of `delta` that lands on a value no emitted code
    reads; empty when every edit is live."""
    out = []
    for edit in delta:
        kinds = None
        if normalise_path(edit.path) == REG:
            try:
                before = read_path(spec, REG)
            except PathError:
                before = []
            kinds = _changed_kinds(before, edit.to)
        why = dead_path(edit.path, paths, changed_kinds=kinds, spec=spec)
        if why:
            out.append((edit.path, why))
    return out
