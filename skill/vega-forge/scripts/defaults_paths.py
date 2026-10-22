#!/usr/bin/env python3
"""
defaults_paths.py — whether a `defaults_to_test` entry's path is one a `/vega-train` rung can
move (netspec-contract §4: "in the path syntax `/vega-train` deltas use — on a path the code
reads").

A `defaults_to_test` entry becomes a `default-test` rung: `plan_ladder.py` reads the entry's
path with the delta-path grammar of run-contract §3 and drops, with a `WARN:`, an entry whose
path addresses nothing. Unattended, the ladder keeps the first four entries ranked dropout,
weight decay, schedule, normalisation, so an entry written `regularization[dropout]` (no
selector field, no `/value`) took the place of the dropout rung and then disappeared from the
ladder (F40, release4 rehearsal round 1): the hypothesis the forge handed on was never tested,
and nothing refused it where it was written.

    problem(spec, path) -> str | None

Returns the reason the path cannot be a rung's delta, or None. The grammar is the trainer's
(`vega-train/scripts/vega_train_core.py`, `read_path` after `normalise_path`):

  * `/`-separated segments; a segment is a key, a list index, or a selector `key[field=value]`
    (`regularization[kind=dropout]`, `blocks[id=B3]`);
  * `blocks/<id>/…` and `blocks[<id>]/…` are the block-id shorthands for `blocks[id=<id>]/…`;
  * a sample fork point the forge defaulted (`data_contract/sample/<point>`, a point of the
    catalog's `fork_order` with `defaults_to_test`) stands while the spec has a sample, even
    when the default left the point unstated (`series_by` null; plan_ladder.py then says why
    it drafts no rung);
  * a path under `regularization` must be one the emitted code reads
    (`vega_regularization.read_paths`), and a live path a declaration shadows
    (`vega_regularization.shadowed_paths`) is refused: a rung there trains the baseline again
    (R2), and `plan_ladder.py` stops the whole plan on it (`ERROR: dead delta`).

`vega-train/scripts/tests/test_defaults_paths_twin.py` holds this module and the trainer's
resolver to one verdict on every path form.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vega_regularization  # noqa: E402

__all__ = ["problem", "normalise", "resolves", "SAMPLE_POINTS"]

# the sample fork points a `defaults_to_test` entry may name (catalog.json `fork_order`, the
# points with `defaults_to_test: true`): the forge defaults them, and an unattended default may
# leave the point unstated (`series_by` null), so the entry stands while the sample itself does
_SAMPLE_POINT = re.compile(r"^(?:data_contract/)?sample/([a-z_]+)$")


def _sample_points() -> frozenset:
    import json
    doc = json.loads((Path(__file__).resolve().parent / "catalog.json").read_text(encoding="utf-8"))
    return frozenset(pt["id"] for fam in doc.get("fork_order") or []
                     for pt in fam.get("points") or [] if pt.get("defaults_to_test"))


SAMPLE_POINTS = _sample_points()

_SELECTOR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([A-Za-z_][A-Za-z0-9_]*)=([^\]]+)\]$")
_BLOCK_SHORTHAND = re.compile(r"^blocks/([A-Za-z_][A-Za-z0-9_]*)(/|$)")
_BLOCK_BRACKET = re.compile(r"^blocks\[([A-Za-z_][A-Za-z0-9_]*)\]")
_BARE_BRACKET = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([A-Za-z_][A-Za-z0-9_]*)\]")


def normalise(path: str) -> str:
    """The block-id shorthands in the selector form, as the trainer reads them."""
    bracket = _BLOCK_BRACKET.match(path)
    if bracket:
        return f"blocks[id={bracket.group(1)}]" + path[bracket.end():]
    match = _BLOCK_SHORTHAND.match(path)
    if not match or match.group(1) in ("0", ""):
        return path
    return f"blocks[id={match.group(1)}]{path[match.end(1):]}"


def _coerce(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    if text in ("true", "false"):
        return text == "true"
    return text


def _step(node: Any, raw: str) -> tuple[bool, Any, str]:
    """(found, the value there, why not)."""
    sel = _SELECTOR.match(raw)
    if sel:
        key, field, value = sel.group(1), sel.group(2), _coerce(sel.group(3))
        if not isinstance(node, Mapping) or not isinstance(node.get(key), list):
            return False, None, f"{key!r} is not a list there, so {raw} addresses nothing"
        hit = next((x for x in node[key] if isinstance(x, Mapping) and x.get(field) == value), None)
        return (hit is not None, hit, f"no entry of {key!r} has {field} == {value!r}")
    if raw.isdigit():
        if not isinstance(node, list) or int(raw) >= len(node):
            return False, None, f"index {raw} is out of range"
        return True, node[int(raw)], ""
    if not isinstance(node, Mapping) or raw not in node:
        return False, None, f"the spec has no key {raw!r} there"
    return True, node[raw], ""


def resolves(spec: Mapping, path: str) -> tuple[bool, str]:
    """(True, "") when the path addresses a value of the spec, else (False, why)."""
    if not isinstance(path, str) or not path:
        return False, "the path is empty"
    node: Any = spec
    for raw in normalise(path).split("/"):
        if raw == "":
            return False, "an empty segment"
        ok, node, why = _step(node, raw)
        if not ok:
            return False, why
    return True, ""


def _hint(path: str) -> str:
    bare = _BARE_BRACKET.match(path)
    if bare and bare.group(1) == "regularization":
        return f" — a declaration is addressed `regularization[kind={bare.group(2)}]/value`"
    if bare:
        return (f" — a list entry is addressed by a selector, `{bare.group(1)}[<field>="
                f"{bare.group(2)}]`")
    return ""


def problem(spec: Mapping, path: str) -> str | None:
    """Why `path` cannot be a `default-test` rung's delta on `spec`, or None."""
    point = _SAMPLE_POINT.match(path) if isinstance(path, str) else None
    if point and point.group(1) in SAMPLE_POINTS and resolves(spec, "data_contract/sample")[0]:
        return None
    ok, why = resolves(spec, path)
    if not ok:
        return (f"defaults_to_test path {path!r} does not resolve on the spec ({why}){_hint(path)}; "
                "/vega-train drops such an entry, and the default it names goes untested")
    norm = normalise(path)
    try:
        read = set(vega_regularization.read_paths(spec))
        shadowed = set(vega_regularization.shadowed_paths(spec))
    except Exception:  # noqa: BLE001 - a spec the regularization reader cannot read is the compiler's refusal
        return None
    if norm in shadowed:
        return (f"defaults_to_test path {path!r} is a live path a regularization declaration "
                f"shadows: no emitted code reads it (read paths: {', '.join(sorted(read)) or 'none'})")
    if norm.split("/")[0].split("[")[0] == "regularization" and norm not in read:
        return (f"defaults_to_test path {path!r} is not a regularization path the emitted code "
                f"reads (read paths: {', '.join(sorted(read)) or 'none'})")
    return None
