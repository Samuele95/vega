#!/usr/bin/env python3
"""
plan_ladder.py — draft the campaign's ladder, and the campaign record it belongs to.

    python3 plan_ladder.py --campaign <dir> --model <model.vega.html>
                           [--budget 20] [--hpo 20 | --no-hpo] [--unattended]
                           [--rules baseline,floor,recurrent-floor,default-test,hpo]

The ladder is the campaign's argument: a baseline, a floor, one rung per default the forge
said it wanted tested, the spec's own first alternatives, and a search. **The deltas are this
script's; the hypotheses and the expectations are yours** — the script writes a draft of each
from the source that motivated the rung, and the procedure's step 1 says to read it and edit
the judgement before anything runs.

Every rung is checked before it reaches `ladder.json`: its delta must address something the
spec actually has, must fall under exactly one decision family (one change means one
*decision*, not one key), and must recompile through the forge's `compile_spec.py`. A rung
that fails any of those is dropped with a `WARN:` naming the rule that proposed it — a ladder
that cannot run is worse than a shorter one. A **dead delta** is refused outright: a rung
whose delta lands on a value no emitted code reads (a `regularization[]` declaration the code
does not read, or a live path a declaration shadows — `live_paths.py`, from
`spec_snapshot.json`'s `read_paths`, netspec-contract §3 R2) would train its baseline's
numbers bit for bit, so the plan stops with `ERROR: dead delta — no emitted code reads
<path>` and writes nothing (the defect is upstream: the forge proposed a default that moves
no weight). A search dimension on a dead path is dropped from the space with a `WARN:`.
Every rung carries its `expect_check` (`expectations.py`: the rule's default from
catalog.json), the form of its `expect` the trainer judges `expect_held` by.

A campaign is planned **once**, on the forge's fresh run directory: a directory that already
holds `campaign.json` is refused (continuing a campaign is `trainer.py --resume`; changing
its ladder before any run is an edit of `ladder.json`'s rungs; a new campaign is a new forge
run, or `--after-eval`). `campaign.json` and `ladder.json` are written together, only once the
ladder planned, so a refused plan leaves the directory as the forge left it.

Exit codes: 0 wrote campaign.json and ladder.json; 2 an input error, a directory that already
holds a campaign, a dead delta, or every rung was dropped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_train_core import (Delta, PathError, digest_file, digest_text,  # noqa: E402
                             forge_scripts, load_catalog, normalise_path, read_path)
from floors import (Latent, positive_width, rung_floor,  # noqa: E402,F401
                    rung_recurrent_floor)
import live_paths as LP  # noqa: E402
from expectations import check_for  # noqa: E402


# --------------------------------------------------------------------------- inputs

def read_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"{what} not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"{what} is not valid JSON ({path}): {exc}"


def _layer(model_path, name):
    """The `name` layer's data island in the model HTML, or None."""
    if not model_path or not Path(model_path).is_file():
        return None
    html = Path(model_path).read_text(encoding="utf-8")
    match = re.search(rf'<script\s+id="layer-{name}-data"[^>]*>([\s\S]*?)</script>', html)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def architecture_layer(model_path):
    """The forge's layer, for its `defaults_to_test` — the entries this ladder exists to test."""
    return _layer(model_path, "architecture")


def geometry_dims(model_path):
    """The geometry layer's `space.dims_95` (the directions that hold 95 % of the standardised
    variance): the forge catalog's default for the PCA floor's z and the `dims_95` of the
    autoencoder spaces. None when the model carries no geometry layer."""
    value = (((_layer(model_path, "geometry") or {}).get("space")) or {}).get("dims_95")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 \
        else None


def defaults_to_test(spec, layer):
    """The forge's list when the layer carries one; otherwise derived from the spec's own
    `vega:default` choices, so a campaign can still be planned from a bare run directory."""
    if layer and layer.get("defaults_to_test"):
        return list(layer["defaults_to_test"]), "the forge's architecture layer"
    derived, runtime = [], ("defaults/device", "defaults/precision", "defaults/seed")

    def note(path, node):
        if not isinstance(node, Mapping) or node.get("source") != "vega:default":
            return
        rows = node.get("catalog")
        rows = [rows] if isinstance(rows, str) else (rows or [])
        if rows and all(r in runtime for r in rows):
            return
        derived.append({"path": path, "observation": "(no observation recorded by the forge)"})

    for key in ("optimizer", "schedule", "early_stopping", "init", "training"):
        note(key, spec.get(key))
    for reg in spec.get("regularization") or []:
        note(f"regularization[kind={(reg or {}).get('kind')}]/value", reg)
    for block in spec.get("blocks") or []:
        if isinstance(block, Mapping):
            note(f"blocks/{block.get('id')}", block)
    return derived, "the spec's vega:default choices (the layer carried no defaults_to_test)"


# --------------------------------------------------------------------------- the campaign record

def make_campaign(camp_dir: Path, spec, model_path, budget_min, device, unattended):
    """`campaign.json` per run-contract §2, from the records the forge left behind."""
    package = (spec.get("outputs") or {}).get("package") or "pipeline"
    smoke, _ = read_json(camp_dir / "smoke.json", "smoke.json")
    smoke = smoke or {}
    snapshot, _ = read_json(camp_dir / "spec_snapshot.json", "spec_snapshot.json")
    determinism = (((snapshot or {}).get("_resolved") or {}).get("determinism") or {})
    dev = smoke.get("device", device or "cpu")
    parts = {}
    for name in ("train", "valid"):
        path = ((spec.get("data_contract") or {}).get("parts") or {}).get(name)
        if path and Path(path).is_file():
            parts[name] = {"path": path, "digest": digest_file(path)}
    parts["test"] = "sealed"
    package_dir = camp_dir / package
    code_digests = {f"{package}/{p.relative_to(package_dir).as_posix()}": digest_file(p)
                    for p in sorted(package_dir.rglob("*.py"))} if package_dir.is_dir() else {}
    protocol = {}
    pipeline_path = (spec.get("input") or {}).get("pipeline")
    if pipeline_path and Path(pipeline_path).is_file():
        pipeline, _ = read_json(pipeline_path, "pipeline.json")
        blueprint_protocol = ((pipeline or {}).get("training") or {}).get("protocol")
        protocol = {"blueprint": blueprint_protocol,
                    "applied": "the shaper's fixed split",
                    "source": "blueprint:training",
                    "reason": "the shaper already materialised the parts; the network trains "
                              "on what it wrote"}
        if blueprint_protocol and "fold" in str(blueprint_protocol).lower():
            protocol.update(source="train:protocol-override",
                            reason="the blueprint asks for a k-fold protocol but the shaper "
                                   "materialised a fixed split; the split is what exists on "
                                   "disk, and the override is recorded rather than silent")
    return {
        "schema": "vega/campaign@1", "run_dir": str(camp_dir),
        "input": {"model": str(model_path) if model_path else None,
                  "model_digest": digest_file(model_path) if model_path
                  and Path(model_path).is_file() else None,
                  "netspec_digest": digest_file(camp_dir / "netspec.json"),
                  "package": package, "code_digests": code_digests, "parts": parts},
        "seed": spec.get("seed", 7), "device": dev,
        "backend": {"name": (spec.get("backend") or {}).get("name"),
                    "version": smoke.get("backend_version")},
        "determinism": {"tolerance": (determinism.get(str(dev).split(":")[0]) or {})
                        .get("tolerance", 0.0)},
        "budget": {"wall_clock_min": budget_min, "spent_min": 0.0, "trials": None},
        "mode": "local", "status": "planned", "test_seal": None, "after_eval": None,
        "unattended": bool(unattended), "protocol": protocol,
        "exposure": test_feature_exposure(spec),
        "planned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def test_feature_exposure(spec) -> dict | None:
    """The disclosed exception to the seal (run-contract §1): under a graph sample's
    `transductive` protocol the test nodes' feature columns — never their labels — are part of
    the training graph, and the campaign says so with the spec's declared `exposure`; None
    under every other protocol, where the test part is sealed whole."""
    sample = ((spec or {}).get("data_contract") or {}).get("sample") or {}
    if sample.get("shape") != "graph" or sample.get("protocol") != "transductive":
        return None
    return {"protocol": "transductive", "declared": sample.get("exposure"),
            "what": "the test nodes' feature columns and their edges, never their labels"}


# --------------------------------------------------------------------------- the rungs

def _first_alternative(node):
    alts = (node or {}).get("alternatives") or []
    return alts[0] if alts and isinstance(alts[0], Mapping) else None


def _blocks_by_type(spec, kind):
    return [b for b in spec.get("blocks") or []
            if isinstance(b, Mapping) and b.get("type") == kind]


def rung_baseline(spec, layer):
    per_family = ((layer or {}).get("readings") or {}).get("per_family") or {}
    expect = ((layer or {}).get("readings") or {}).get("abstract") or ""
    if per_family:
        expect = " ".join(v for v in per_family.values() if v)[:600]
    return {"id": "R0", "kind": "baseline", "delta": [],
            "hypothesis": "the spec as engineered; establishes the curves every ablation is "
                          "read against",
            "source": "netspec",
            "expect": expect or "what the forge's per-family readings said the first run must "
                                "show"}


# R1, the floor of the sample's shape, and R1b, the recurrent floor, are built in floors.py
# (`rung_floor`, `rung_recurrent_floor`); their ids are fixed below (FIXED_IDS)

_REMOVAL = {"dropout": 0.0, "weight_decay": 0.0, "grad_clip": None, "label_smoothing": 0.0}


_SAMPLE_POINT = re.compile(r"^(?:data_contract/)?sample/([a-z_]+)$")


def _point_row(point, forge_cat):
    """The forge catalog's defaults row a sample fork point cites (fork_order points[]), as
    (row id, row) — ("", {}) when the catalog names none."""
    for family in (forge_cat or {}).get("fork_order") or []:
        for spot in (family or {}).get("points") or []:
            if spot.get("id") == point and spot.get("default"):
                key = str(spot["default"])
                return key, ((forge_cat.get("defaults") or {}).get(key.split("/", 1)[-1]) or {})
    return "", {}


def point_alternative(point, current, spec, forge_cat):
    """The catalog's alternative for a sample fork point the forge defaulted (T03's
    `data_contract/sample/<point>` rows): a list of (field, value) edits and the reason, or
    (None, why not). The alternative is the catalog row's, never invented here."""
    row_id, row = _point_row(point, forge_cat)
    if not row_id:
        return None, f"the forge catalog names no default row for the {point} fork point"
    sample = ((spec.get("data_contract") or {}).get("sample")) or {}
    cite = f"the catalog's alternative ({row_id}: {row.get('alternative') or 'unstated'})"
    if point == "context":
        others = [v for v in row.get("values") or () if v != current]
        return ([("context", others[0])], cite) if others else \
            (None, f"{row_id} lists no value other than {current!r}")
    if point == "autoregressive":
        return [("autoregressive", not bool(current))], cite
    if point == "series_by":
        if current is None:
            return None, ("the default left series_by null (no series key in the shaping "
                          "record); the catalog's alternative is that same null")
        return [("series_by", None)], cite
    if point == "batching":
        if current == "full" and "neighbours" in sample:
            fan = (((row.get("rule") or {}).get("otherwise")) or {}).get("neighbours")
            return [("batching", "sampled"), ("neighbours", list(fan or [25, 10]))], cite
        if current == "sampled":
            edits = [("batching", "full")]
            return (edits + ([("neighbours", None)] if "neighbours" in sample else [])), cite
        return None, (f"batching {current!r} has no catalog alternative this ladder can run "
                      "(graphs is the only batching at level graph; a sampled rung needs the "
                      "sample's neighbours key)")
    return None, f"the {point} fork point is not one the ladder tests"


class _SamplePoint(NamedTuple):
    """A `data_contract/sample/<point>` entry: its path, the fork point and its value."""

    path: str
    point: str
    current: Any


def _default_rung(entry, index, delta, why) -> dict:
    """A `default-test` rung; the caller fills in its decision family from the delta."""
    return {"id": f"D{index}", "kind": "ablation", "family": None, "delta": delta,
            "hypothesis": f"the forge marked this a default worth testing — {why}",
            "source": "vega:ladder/default-test",
            "expect": entry.get("observation") or "the forge recorded no observation for this "
                                                  "default"}


def _rung_sample_point(entry, spec, index, where: _SamplePoint, forge_cat):
    """A `data_contract/sample/<point>` entry: the catalog's alternative for the fork point."""
    edits, why = point_alternative(where.point, where.current, spec, forge_cat)
    if edits is None:
        return None, f"{where.path}: {why}"
    base = where.path[: -len(where.point)]
    delta = []
    for field, value in edits:
        try:
            was = read_path(spec, normalise_path(base + field))
        except PathError as exc:
            return None, f"{base + field}: {exc}"
        delta.append({"path": base + field, "from": was, "to": value})
    rung = _default_rung(entry, index, delta, why)
    rung["hypothesis"] = f"the forge marked the {where.point} fork point a default worth " \
                         f"testing — {why}"
    return rung, None


def _owner_choice(spec, path) -> tuple:
    """The choice node the path sits in, for its own `alternatives`: the path's first segment,
    or — for the `blocks/<id>` shorthand, whose first segment is the whole block list — the
    block the id names. (owner path, owner or None, its first alternative or None)."""
    first = normalise_path(path).split("/")[0]
    owner_path = path.split("/")[0]
    if normalise_path(owner_path) != first:
        owner_path = first
    try:
        owner = read_path(spec, normalise_path(owner_path))
    except PathError:
        owner = None
    owner = owner if isinstance(owner, Mapping) else None
    return owner_path, owner, _first_alternative(owner)


def _default_move(path, current, spec) -> tuple:
    """((path, from, to, why), None) for a default-test entry — the default removed, the
    value its choice's alternative states, that alternative merged into its choice, or the
    default doubled — or (None, why not)."""
    owner_path, owner, alternative = _owner_choice(spec, path)
    match = re.search(r"\[kind=([A-Za-z_]+)\]", path)
    kind = match.group(1) if match else None
    listed = "the alternative the spec itself listed"
    if kind in _REMOVAL and path.endswith("/value"):
        return (path, current, _REMOVAL[kind], f"removing the {kind} the forge defaulted to"), \
            None
    if alternative and not isinstance(current, Mapping) and "value" in alternative:
        return (path, current, alternative["value"], listed), None
    if alternative and owner is not None:
        # the alternative lives on the choice node, not on the leaf the entry pointed at, so
        # the rung's path and its `from` both move up to the owner — a delta whose `from` is
        # the leaf's value and whose path is the owner's addresses two different things
        merged = _merge_alternative(owner, alternative, f"{listed}, run as a rung of this ladder")
        return (owner_path, owner, merged, listed), None
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return (path, current, current * 2 if current else 1,
                "doubling the default, since no alternative was recorded and removal is not "
                "defined here"), None
    return None, f"{path}: no removal and no first alternative are defined for it"


def rung_default_test(entry, spec, index, forge_cat=None):
    """One rung per `defaults_to_test` entry: the default removed, or its first alternative
    (a sample fork point's alternative is the forge catalog's row for it)."""
    path = str(entry.get("path") or "")
    if not path:
        return None, "an entry with no path"
    try:
        current = read_path(spec, normalise_path(path))
    except PathError as exc:
        return None, f"{path}: {exc}"
    point = _SAMPLE_POINT.match(path)
    if point:
        return _rung_sample_point(entry, spec, index,
                                  _SamplePoint(path, point.group(1), current), forge_cat)
    move, why = _default_move(path, current, spec)
    if move is None:
        return None, why
    path, current, to, why = move
    return _default_rung(entry, index, [{"path": path, "from": current, "to": to}], why), None


def _merge_alternative(node, alternative, why):
    """The choice with its first alternative's values, and its provenance untouched.

    A rung changes what a choice *is*, never where it came from: the spec's `source` and
    `catalog` stay as the forge wrote them (netspec-contract §1's closed set has no
    `vega:ladder/…` or `netspec:…` in it), the rationale gains a clause saying this is the
    alternative being tested, and the old value becomes the alternative — so the resolved spec
    still names what it was weighed against.
    """
    payload = {k: v for k, v in alternative.items() if k != "when"}
    params = payload.pop("params", None)
    merged = dict(node)
    changed = set(payload)
    if isinstance(node.get("params"), Mapping):
        if _renames(node, payload):
            merged["params"] = dict(params or {})
        elif params is not None:
            merged["params"] = {**node["params"], **params}
        changed.add("params")
    elif isinstance(params, Mapping):
        # a choice whose values sit at its top level (early_stopping: patience, min_delta;
        # training: batch_size) takes the alternative's `params` as those values — a `params`
        # key the choice does not read would make the rung train its baseline again
        payload.update({k: v for k, v in params.items() if k in node})
        extra = {k: v for k, v in params.items() if k not in node}
        if extra:
            payload["params"] = extra
        changed |= set(params)
    merged.update(payload)
    was = {k: v for k, v in node.items()
           if k in ("name", "scheme", "primary", "value", "params") or k in changed}
    was.pop("alternatives", None)
    merged["rationale"] = f"{node.get('rationale', '')} ({why}: {alternative.get('when')})"\
        .strip()
    merged["alternatives"] = [dict(was, when="the spec's own choice, which this rung is "
                                             "read against")]
    return merged


def _renames(node, payload) -> bool:
    """Whether the alternative is a different choice (another loss, optimizer, schedule or
    init scheme) rather than new values for the same one. A different choice's params
    **replace** the node's — `focal`'s `gamma` does not inherit `bce_with_logits`'s
    `pos_weight`, and `sgd` does not inherit `adamw`'s betas — while new values for the same
    choice are merged over the ones the alternative leaves unstated."""
    return any(key in payload and payload[key] != node.get(key) for key in ("name", "scheme"))


def rung_spec_alternative(section, spec, index):
    node = spec.get(section)
    alternative = _first_alternative(node)
    if not alternative:
        return None, f"{section} lists no alternative"
    if not {k for k in alternative if k != "when"}:
        return None, f"{section}'s first alternative carries no value"
    merged = _merge_alternative(node, alternative,
                               "the alternative the spec itself weighed")
    return {"id": f"A{index}", "kind": "ablation", "family": None,
            "delta": [{"path": section, "to": merged}],
            "hypothesis": f"the spec's own first alternative for {section}: "
                          f"{alternative.get('when') or 'no `when` recorded'}",
            "source": f"netspec:{section}/alternatives[0]",
            "expect": alternative.get("when") or "stated by the alternative's `when`"}, None


def rung_capacity(spec, index, factor):
    dense = _blocks_by_type(spec, "dense")
    if not dense:
        return None, "the spec declares no dense block to resize"
    block = dense[0]
    units = (block.get("params") or {}).get("units")
    if not isinstance(units, int):
        return None, f"block {block.get('id')} has no integer width"
    new = max(8, int(units * factor))
    edits = [{"path": f"blocks/{block.get('id')}/params/units", "from": units, "to": new}]
    # the residual's inner widths follow the first dense, or x + f(x) stops adding up
    for residual in _blocks_by_type(spec, "residual"):
        for i, inner in enumerate((residual.get("params") or {}).get("inner") or []):
            if isinstance(inner, Mapping) and inner.get("type") == "dense" \
                    and (inner.get("params") or {}).get("units") == units:
                edits.append({"path": f"blocks/{residual.get('id')}/params/inner/{i}/params/units",
                              "from": units, "to": new})
    return {"id": f"C{index}", "kind": "ablation", "family": "architecture", "delta": edits,
            "hypothesis": f"the catalog's sizing rule is near the right region: width "
                          f"{'doubled' if factor > 1 else 'halved'} to {new}",
            "source": "vega:ladder/capacity",
            "expect": "the curve of valid metric against capacity peaks near the baseline"}, None


# --------------------------------------------------------------------------- the search space

# a block param addressed by path: `blocks[id=<id>]/params/<p>` once normalised
_BLOCK_PARAM = re.compile(r"^blocks\[id=([^\]]+)\]/params/([A-Za-z_][A-Za-z0-9_]*)$")
# the older names a block still reads a param under (blocks.py: `num_layers`, else `layers`)
_PARAM_ALIASES = {"num_layers": ("layers",)}


def unstated_param(spec, path, forge_cat):
    """The value a block is built with for a param its spec leaves unstated, as (where, value).

    For `blocks/<id>/params/<p>` of a block that does not state `p`: the older name the block
    still reads (`layers` for `num_layers`), else the forge catalog's `blocks[<type>].defaults`
    — a forge-composed `gru` or `lstm` usually leaves `num_layers` at its default, and the
    space still searches it (D1 §12.4). (None, None) when the path is no block param, the
    spec states it, or neither gives a value."""
    match = _BLOCK_PARAM.match(normalise_path(str(path)))
    if not match:
        return None, None
    block = next((b for b in spec.get("blocks") or []
                  if isinstance(b, Mapping) and str(b.get("id")) == match.group(1)), None)
    param = match.group(2)
    params = (block or {}).get("params") or {}
    if block is None or param in params:
        return None, None
    for alias in _PARAM_ALIASES.get(param, ()):
        if alias in params:
            return f"blocks/{match.group(1)}/params/{alias} (the older name)", params[alias]
    defaults = ((((forge_cat or {}).get("blocks") or {}).get(block.get("type"))) or {}) \
        .get("defaults") or {}
    if param not in defaults:
        return None, None
    return f"vega-forge catalog blocks/{block.get('type')}.defaults", defaults[param]


def _space_row(spec, catalog) -> tuple:
    """(the family's §4 row, generic?): its own row, a `prefix*` row, or — for a family the
    catalog has no row for — `lr` plus every regularization value the spec carries."""
    family = (spec.get("family") or {}).get("name") or ""
    spaces = catalog.get("spaces") or {}
    row = spaces.get(family)
    if row is None:
        row = next((v for k, v in spaces.items()
                    if k.endswith("*") and family.startswith(k[:-1])), None)
    if row is not None:
        return row, False
    row = [{"path": "optimizer/params/lr", "kind": "float", "range": [1e-4, 1e-2], "log": True}]
    for reg in spec.get("regularization") or []:
        kind = (reg or {}).get("kind")
        if isinstance((reg or {}).get("value"), (int, float)) and kind:
            row.append({"path": f"regularization[kind={kind}]/value", "kind": "float",
                        "range": [0.0, 0.5]})
    return row, True


def _space_dimension(dim, spec, placeholders, forge_cat, dims_95, read=None) -> tuple:
    """(entry, None) for one catalog dimension resolved on the spec, or (None, (bucket, note))
    with bucket `absent` (an optional block the spec does not carry) or `dropped` (nothing
    there, or a dead path: no emitted code reads it, so every trial would train the same
    numbers along it)."""
    path = dim["path"]
    match = re.search(r"<([^>]+)>", path)
    if match:
        spec_key = f"<{match.group(1)}>"
        block = _resolve_placeholder(spec, placeholders.get(spec_key))
        if block is None:
            # an optional dimension names a block the family may or may not carry (an
            # edge_dropout): its absence is the spec's choice, not a gap in the space
            return None, ("absent" if dim.get("optional") else "dropped",
                          f"{path} (no {spec_key} block in this spec)")
        path = path.replace(spec_key, str(block.get("id")))
    entry = {k: v for k, v in dim.items() if k != "optional"}
    try:
        current = read_path(spec, normalise_path(path))
    except PathError:
        where, current = unstated_param(spec, path, forge_cat)
        if where is None:
            return None, ("dropped", f"{path} (the spec has nothing there)")
        entry["current_from"] = where
    dead = LP.dead_path(path, read if read is not None else LP.of_snapshot(None), spec=spec)
    if dead:
        return None, ("dropped", f"{path} (dead delta — {dead})")
    entry.update(path=path, current=current)
    if "range" in entry:
        entry["range"] = _resolve_range(entry.get("range"), dims_95)
    return entry, None


def build_space(spec, catalog, forge_cat=None, dims_95=None, read=None):
    """The family's §4 space with its `<… id>` placeholders resolved from the spec's blocks.
    A block param the spec leaves at its default is searched from that default
    (`unstated_param`); `dims_95` resolves the autoencoder ranges written in its terms; a
    dimension on a path no emitted code reads (`read`, the campaign's `live_paths.ReadPaths`;
    the pre-R2 set when none is given) is dropped as a dead delta."""
    row, generic = _space_row(spec, catalog)
    forge_cat = forge_cat if forge_cat is not None else forge_catalog()
    placeholders = catalog.get("space_placeholders") or {}
    out, notes = [], {"dropped": [], "absent": []}
    for dim in row:
        entry, why = _space_dimension(dim, spec, placeholders, forge_cat, dims_95, read)
        if entry is None:
            notes[why[0]].append(why[1])
        else:
            out.append(entry)
    if not any(d["path"] == "optimizer/params/lr" for d in out):
        out.insert(0, {"path": "optimizer/params/lr", "kind": "float",
                       "range": [1e-4, 1e-2], "log": True,
                       "current": ((spec.get("optimizer") or {}).get("params") or {}).get("lr")})
    return {"schema": "vega/space@1", "family": (spec.get("family") or {}).get("name") or "",
            "generic": generic, "dimensions": out, "dropped": notes["dropped"],
            "not_applicable": notes["absent"],
            "note": catalog.get("generic_space_note") if generic else None}, notes["dropped"]


def placeholder_types(rule) -> list:
    """The block types a placeholder rule tries, in order: `block_type`, then `fallback_types`
    (or the older single `fallback_type`)."""
    rule = rule or {}
    kinds = [rule.get("block_type"), *(rule.get("fallback_types") or ()),
             rule.get("fallback_type")]
    return [k for k in kinds if k]


def _resolve_placeholder(spec, rule):
    if not rule:
        return None
    for kind in placeholder_types(rule):
        if not kind:
            continue
        found = _blocks_by_type(spec, kind)
        if not found:
            continue
        if rule.get("pick") == "narrowest":
            with_units = [b for b in found if isinstance((b.get("params") or {}).get("units"),
                                                         int)]
            if with_units:
                return min(with_units, key=lambda b: b["params"]["units"])
        return found[0]
    return None


def _resolve_range(rng, dims_95=None):
    """`dims_95/2 … 2*dims_95` is written in the catalog's prose form: resolved on the
    geometry layer's dims_95, or on 8 when the model carries none."""
    if not isinstance(rng, list) or not any(isinstance(v, str) for v in rng):
        return rng
    dims = positive_width(dims_95) or 8
    return [max(2, int(dims / 2)), max(4, int(2 * dims))]


# --------------------------------------------------------------------------- planning

def plan(camp_dir: Path, spec, layer, catalog, *, budget_min, unattended, rules, hpo_trials,
         forge=None, out=print, dims_95=None):
    entries, source_of_defaults = defaults_to_test(spec, layer)
    out(f"OK: {len(entries)} default(s) to test, from {source_of_defaults}")
    forge_cat = forge_catalog(forge)
    read = LP.of_campaign(camp_dir, forge)
    if not read.recorded:
        out(f"WARN: dead deltas are judged against {read.source}")

    drafted, dropped = [], []

    def add(rung, why=None):
        if rung is None:
            if why:
                dropped.append(why)
            return
        drafted.append(rung)

    if "baseline" in rules:
        add(rung_baseline(spec, layer))
    if "floor" in rules:
        rung, why = rung_floor(spec, catalog, dims_95)
        add(rung, f"vega:ladder/floor — {why}" if why else None)
    if "recurrent-floor" in rules:
        rung, why = rung_recurrent_floor(spec, forge_cat)
        add(rung, f"vega:ladder/recurrent-floor — {why}" if why else None)

    if "default-test" in rules:
        chosen = entries
        if unattended:
            order = (catalog["ladder_rules"]["default-test"].get("unattended_order") or [])
            limit = catalog["ladder_rules"]["default-test"].get("unattended_max", 4)

            def rank(entry):
                path = str(entry.get("path", ""))
                for i, key in enumerate(order):
                    if key in path:
                        return i
                return len(order)

            chosen = sorted(entries, key=rank)[:limit]
            if not chosen:
                out("OK: vega:ladder/default-test — no applicable entry; the rule is skipped")
        for i, entry in enumerate(chosen, start=1):
            rung, why = rung_default_test(entry, spec, i, forge_cat)
            add(rung, f"vega:ladder/default-test — {why}" if why else None)

    if "spec-alternative" in rules:
        for i, section in enumerate(catalog["ladder_rules"]["spec-alternative"]["sections"],
                                    start=1):
            rung, why = rung_spec_alternative(section, spec, i)
            add(rung, f"vega:ladder/spec-alternative — {why}" if why else None)

    if "capacity" in rules:
        for i, factor in enumerate((2.0, 0.5), start=1):
            rung, why = rung_capacity(spec, i, factor)
            add(rung, f"vega:ladder/capacity — {why}" if why else None)

    # --- validate every drafted rung against the spec, then renumber
    #
    # Two rules propose the same experiment often enough to matter: `default-test` on the
    # optimizer and `spec-alternative` on the optimizer are the same run, and a campaign that
    # trains it twice spends the budget to learn nothing. The first rule to propose it keeps
    # it, and the duplicate is dropped with both rules named.
    accepted, seen, dead_rungs = [], {}, []
    for rung in drafted:
        delta = Delta.of(rung["delta"])
        families = [f for f in delta.families(spec) if f]
        if rung.get("family") is None and families:
            rung["family"] = families[0]
        problems = delta.problems(spec, rung.get("family"))
        if problems:
            dropped.append(f"{rung['source']} ({rung['id']}) — " + "; ".join(problems))
            continue
        dead = LP.dead_edits(delta, spec, read)
        if dead:
            dead_rungs += [f"dead delta — {why} ({rung['source']} ({rung['id']}): the rung "
                           "would train its baseline's numbers)" for _, why in dead]
            continue
        ok, why = _compiles(delta.apply(spec), camp_dir, spec, forge)
        if not ok:
            dropped.append(f"{rung['source']} ({rung['id']}) — the delta produces a spec the "
                           f"compiler refuses: {why}")
            continue
        signature = _signature(delta)
        if signature in seen:
            dropped.append(f"{rung['source']} ({rung['id']}) — the same experiment as "
                           f"{seen[signature]}, which {rung['source']} and that rule both "
                           "propose; the budget buys nothing by running it twice")
            continue
        seen[signature] = f"{rung['id']} ({rung['source']})"
        accepted.append(rung)

    if dead_rungs:
        for note in dropped:
            out(f"WARN: rung dropped — {note}")
        for line in dead_rungs[:-1]:
            out(f"ERROR: {line}")
        return None, dead_rungs[-1]

    number_rungs(accepted)
    primary = (spec.get("metrics") or {}).get("primary")

    # --- the study
    study, space = None, None
    if "hpo" in rules and hpo_trials:
        space, space_dropped = build_space(spec, catalog, forge_cat, dims_95, read)
        if space["dimensions"]:
            study = {"id": "H1", "kind": "hpo", "study": "H1", "space": "hpo/H1/space.json",
                     "delta": [],
                     "budget": {"trials": hpo_trials,
                                "wall_clock_min": round((budget_min or 20) * 0.4, 2)},
                     "hypothesis": "the dimensions interact — "
                                   + ", ".join(d["path"] for d in space["dimensions"])
                                   + "; a search around the baseline finds the region",
                     "source": "vega:ladder/hpo",
                     "expect": "the best trial is at least the baseline on valid primary"}
            accepted.append(study)
            if space["generic"]:
                out("WARN: the family has no §4 search space; the space is generic (lr plus "
                    "every regularization value the spec carries)")
            for note in space_dropped:
                out(f"WARN: search dimension dropped — {note}")

    for note in dropped:
        out(f"WARN: rung dropped — {note}")

    if not accepted:
        return None, "every rung was dropped; there is no ladder to run"

    # --- every rung's expect as the check the trainer judges expect_held by
    for rung in accepted:
        check, why = check_for(rung.get("expect_check"), rung.get("source"), catalog,
                               rung.get("id"))
        if check is None:
            return None, f"{rung['id']}: {why}"
        rung["expect_check"] = check.as_json(primary)

    if study is not None:
        (camp_dir / "hpo" / "H1").mkdir(parents=True, exist_ok=True)
        (camp_dir / "hpo" / "H1" / "space.json").write_text(
            json.dumps(space, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- budget shares
    unshared = [r for r in accepted if not r.get("budget")]
    if budget_min and unshared:
        spent = sum((r.get("budget") or {}).get("wall_clock_min", 0) for r in accepted)
        share = max(0.1, round((budget_min - spent) / len(unshared), 2))
        for rung in unshared:
            rung["budget"] = {"wall_clock_min": share}

    fork = {"asked": f"ladder {', '.join(r['id'] for r in accepted)} under "
                     f"{budget_min} min, selection by the catalog's ladder/selection rule"}
    if unattended:
        fork["or"] = "default ladder applied (unattended)"
    else:
        fork["answer"] = None      # the skill records what the user said

    return {"schema": "vega/ladder@1", "runs": accepted, "forks": [fork],
            "rules_applied": sorted(rules),
            "unattended": bool(unattended)}, None


#: the rung ids a ladder rule owns whether or not its rung was planned (catalog `run_ids`):
#: the floor is always R1 and the recurrent floor R1b, so a dropped floor never lends its id
#: to another rung and a lane's `R1` always means the floor
FIXED_IDS = {"vega:ladder/floor": "R1", "vega:ladder/recurrent-floor": "R1b"}


def number_rungs(accepted) -> None:
    """R0 the baseline, R1 the floor, R1b the recurrent floor, R2… every other rung in ladder
    order; a study keeps its own id (H<n>)."""
    n = 2
    for rung in accepted:
        if rung["kind"] in ("baseline", "hpo"):
            continue
        fixed = FIXED_IDS.get(rung.get("source"))
        if fixed:
            rung["id"] = fixed
            continue
        rung["id"] = f"R{n}"
        n += 1


def forge_catalog(forge=None):
    """`/vega-forge`'s catalog.json (the families' compositions and the rnn row's
    `ablation_floor_for`), read through the one seam to the sibling skill."""
    path = forge_scripts(forge) / "catalog.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# the keys that carry prose rather than behaviour: two rungs whose values agree and whose
# rationales differ are the same experiment, and the budget should buy it once
_PROSE = ("rationale", "alternatives", "source", "catalog", "formula", "update_rule", "when")


def _strip_prose(value):
    if isinstance(value, Mapping):
        return {k: _strip_prose(v) for k, v in sorted(value.items()) if k not in _PROSE}
    if isinstance(value, list):
        return [_strip_prose(v) for v in value]
    return value


def _signature(delta):
    """What a rung actually *does*, with the prose removed."""
    return json.dumps(sorted((normalise_path(e.path), _strip_prose(e.to)) for e in delta),
                      sort_keys=True, separators=(",", ":"), default=str)


def _compiles(resolved, camp_dir, spec, forge):
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "netspec.json"
        path.write_text(json.dumps(resolved), encoding="utf-8")
        args = [sys.executable, str(forge_scripts(forge) / "compile_spec.py"),
                "--spec", str(path), "--check-only"]
        model = (spec.get("input") or {}).get("model")
        for base in (camp_dir, camp_dir.parent, Path.cwd()):
            if model and (base / model).is_file():
                args += ["--model", str(base / model)]
                break
        else:
            args += ["--model", str(path)]      # no model: the compiler says so and refuses
        pipeline = (spec.get("input") or {}).get("pipeline")
        if pipeline:
            args += ["--pipeline", str(pipeline)]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            return True, ""
        errors = [l[len("ERROR: "):] for l in (proc.stdout + proc.stderr).splitlines()
                  if l.startswith("ERROR: ") and "does not compile" not in l]
        return False, (errors[0] if errors else "no reason given")


# --------------------------------------------------------------------------- the shell

def refusal(camp_dir: Path) -> str | None:
    """Why this directory cannot be planned, or None. A campaign is planned once (run-contract
    §1): the seal says the test was seen; `campaign.json` says a campaign already lives here,
    and a second plan would rewrite it and the ladder — the pre-registered selection fork
    included; a `ladder.json` without one is a ladder nobody planned here."""
    if (camp_dir / "test_seal.json").is_file():
        return ("this campaign's test part has been opened; plan a new campaign with "
                "--after-eval rather than replanning this one")
    if (camp_dir / "campaign.json").is_file():
        return (f"{camp_dir} already holds campaign.json — a campaign is planned once and is "
                "append-only, so it is never replanned: continue it with trainer.py --resume, "
                "change its ladder before any run by editing ladder.json's rungs, or start a "
                "new campaign (a new /vega-forge run directory, or --after-eval)")
    if (camp_dir / "ladder.json").is_file():
        return (f"{camp_dir / 'ladder.json'} already exists without a campaign.json; a "
                "campaign is append-only and its ladder is planned with its campaign record — "
                "plan in the run directory /vega-forge created")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True, help="the run directory /vega-forge created")
    ap.add_argument("--model", required=True,
                    help="the .vega.html the forge wrote; its architecture layer carries the "
                         "defaults_to_test this ladder tests, and campaign.json records it as "
                         "the model every rung recompiles against (F37: a campaign planned "
                         "without it could never compile a rung)")
    ap.add_argument("--budget", type=float, default=20.0, metavar="MINUTES")
    ap.add_argument("--hpo", type=int, default=None, metavar="TRIALS")
    ap.add_argument("--no-hpo", action="store_true")
    ap.add_argument("--unattended", action="store_true")
    ap.add_argument("--rules", default=None,
                    help="comma-separated ladder rules (default: the catalog's unattended set "
                         "under --unattended, all of them otherwise)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vega-forge-dir", default=None)
    a = ap.parse_args(argv)

    camp_dir = Path(a.campaign)
    spec, err = read_json(camp_dir / "netspec.json", "netspec.json")
    if err:
        print(f"ERROR: {err} — the campaign lives in the run directory /vega-forge created")
        return 2
    catalog, err = load_catalog()
    if err:
        print(f"ERROR: {err}")
        return 2
    refused = refusal(camp_dir)
    if refused:
        print(f"ERROR: {refused}")
        return 2
    if not Path(a.model).is_file():
        print(f"ERROR: --model {a.model} not found — it is the .vega.html /vega-forge wrote, "
              "and campaign.json records it as the model every rung recompiles against")
        return 2

    layer = architecture_layer(a.model)
    if a.model and layer is None:
        print(f"WARN: {a.model} carries no architecture layer; the defaults to test are "
              "derived from the spec instead")

    rules = ([r.strip() for r in a.rules.split(",")] if a.rules else
             (catalog["unattended_ladder"] if a.unattended
              else ["baseline", "floor", "recurrent-floor", "default-test", "spec-alternative",
                    "capacity", "hpo"]))
    trials = 0 if a.no_hpo else (a.hpo or (catalog["hpo_defaults"]["trials_unattended"]
                                           if a.unattended
                                           else catalog["hpo_defaults"]["trials_attended"]))

    campaign = make_campaign(camp_dir, spec, a.model, a.budget, a.device, a.unattended)
    ladder, err = plan(camp_dir, spec, layer, catalog, budget_min=a.budget,
                       unattended=a.unattended, rules=set(rules), hpo_trials=trials,
                       forge=a.vega_forge_dir, dims_95=geometry_dims(a.model))
    if err:
        print(f"ERROR: {err}")
        return 2
    # the two records of a campaign land together, once the ladder planned: a refused plan
    # leaves the directory as the forge left it, and plannable again
    (camp_dir / "campaign.json").write_text(
        json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (camp_dir / "ladder.json").write_text(
        json.dumps(ladder, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OK: campaign.json — package {campaign['input']['package']}, device "
          f"{campaign['device']}, backend {campaign['backend']['name']} "
          f"{campaign['backend']['version']}, determinism tolerance "
          f"{campaign['determinism']['tolerance']}, budget {a.budget} min")
    if campaign["protocol"].get("source") == "train:protocol-override":
        print(f"WARN: protocol override — {campaign['protocol']['reason']}")

    print(f"OK: ladder.json — {len(ladder['runs'])} run(s)")
    width = max(len(r["id"]) for r in ladder["runs"])
    for rung in ladder["runs"]:
        share = (rung.get("budget") or {}).get("wall_clock_min")
        print(f"    {rung['id']:<{width}}  {rung['kind']:<9} "
              f"{(rung.get('family') or rung.get('study') or '-'):<14} "
              f"{Delta.of(rung['delta']).describe()[:64]:<66} {share} min")
    print("OK: the deltas are the script's; the hypotheses and the expectations are yours — "
          "read ladder.json and edit the judgement before running anything")
    if not a.unattended:
        print("FORK: approve the ladder, or add / remove / reorder rungs and change the budget "
              "or the selection rule")
    return 0


if __name__ == "__main__":
    sys.exit(main())
