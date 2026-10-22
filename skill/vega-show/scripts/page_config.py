"""
page_config — `showcase/page.json` (`vega/page@1`, D4 §14): the page's mechanical configuration,
assembled from what the chain recorded. `build_showcase.py` (H06) calls `build_page_config`
after `resolve_story`, pins the result, and refuses it unless `check_page_config` is empty
(gate BD2). Nothing here proposes (R-E9): the lexicon, the class roles and the motif are the
draft's (H04), and this module copies the motif the authored plan settled.

    build_page_config(paths, netspec, records, story, catalogs) -> dict
    build_walk(netspec, explain_catalog) -> list[dict]          walk ids of R-E8, computed once
    recipe_summary(shape_record, netspec) -> dict               (page_derive)
    metric_record(netspec, catalogs) -> dict                    direction from metric_direction
    views_available(examples, story, serve) -> dict             order + off
    reference_record(lane, scores) -> dict | None               reconstruction / anomaly lanes
    page_records(store, examples=None, reference_scores=None)   `records` from a RecordStore
    check_page_config(page, network=None, catalogs=None)        (page_check) [] when valid

**`records`** is a mapping: `manifest` (the bundle's `MANIFEST.json`), `manifest_digest`,
`shape` (`records/shape.json`, at least its `manifest` and `recipe`), `analysis`
(`records/analysis.json` or None), `examples` (the `vega/examples@2` body or None) and
`reference_scores` (`{scores: [row score, …], source}`: the bundle run on the demo sample — the
training part — at show time; required on the `reconstruction` and `anomaly` view keys).
**`paths.plan`** is the authored `vega/story-plan@2` of the showcase being built (H06 passes the
paths of `showcase.building/`). **`serve`** is `MANIFEST.serve`.

**Views** (`views_available`): a tour view is on when what it renders exists —

| view | needs | off reason key |
|---|---|---|
| `context`, `network`, `results` | the story section of that id | `views.off.no_section` |
| `try` | a starting sample (a preset, a shipped test row or a demo item) | `views.off.no_examples` |
| `try` | a single-answer route (`POST /v1/explain` or `POST /v1/predict`) | `views.off.no_route` |
| `many` | the many-rows route `POST /v1/predict/csv` | `views.off.no_route` |

**The motif source** (R-E4, S3): a `show: motif` stance of the plan gives `show:<its date>`; the
modality's base without a stance is the unattended default (`vega:default`); any other admissible
motif without a stance was the answer to fork S3 (`user:S3`).
"""
from __future__ import annotations

import copy
import math
import re
from typing import Mapping

from page_check import check_page_config
from page_derive import (METRIC_BOUNDS, Catalogs, activation_names, block_types, lane_record,
                         modality_of, recipe_summary)
from vega_show_core import CATALOG, FactMissing, ShowError, read_json

__all__ = ["Catalogs", "METRIC_BOUNDS", "REFERENCE_QUANTILES", "build_page_config", "build_walk",
           "recipe_summary", "metric_record", "views_available", "reference_record",
           "page_records", "check_page_config"]

REFERENCE_QUANTILES = ("0.5", "0.9", "0.99")
METRIC_SOURCE = "netspec:metrics; vega-forge catalog metric_direction"
IDENTITY_SOURCE = "bundle:MANIFEST.serve.identity"
_SINGLE_ROUTES = ("POST /v1/explain", "POST /v1/predict")
_MANY_ROUTE = "POST /v1/predict/csv"
_REFERENCE_KIND = {"reconstruction": "reconstruction_error", "distance": "distance"}
_STANCE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_STANCE_MOTIF = re.compile(r"show:\s*motif\s+([a-z]+)")


def build_page_config(paths, netspec: Mapping, records: Mapping, story: Mapping,
                      catalogs: Catalogs) -> dict:
    """page.json: lane, metric, motif (from the settled plan), views, recipe, reference,
    explain, walk, code — every field from the chain's records, nothing proposed."""
    manifest, plan = _required(records, "manifest"), _authored_plan(paths)
    lane, metric = lane_record(netspec, catalogs), metric_record(netspec, catalogs)
    walk = build_walk(netspec, catalogs.explain)
    _check_story(story, lane, metric, walk)
    return {
        "schema": "vega/page@1",
        "template": _template(catalogs, plan),
        "bundle": {"manifest_digest": _required(records, "manifest_digest")},
        "identity": _identity(manifest),
        "lane": lane,
        "metric": metric,
        "motif": _motif(plan, modality_of(lane["shape"], records.get("analysis"))),
        "views": views_available(records.get("examples"), story, manifest.get("serve")),
        "recipe": recipe_summary(_required(records, "shape"), netspec),
        "reference": reference_record(lane, records.get("reference_scores")),
        "explain": _explain(netspec, lane, catalogs),
        "walk": walk,
        "code": {"language": catalogs.coverage["language"], "source": "coverage.json:language"},
    }


def build_walk(netspec: Mapping, explain_catalog: Mapping) -> list[dict]:
    """R-E8: `root, store, build`, one step per distinct block type in forward order (the id
    is the type), then `forward, link, shape, run, explain, route`; the fixed steps and each
    block's `walk` and `code` anchor verbatim from the backend's explain catalog."""
    head, tail = CATALOG["walk"]["head"], CATALOG["walk"]["tail"]
    fixed = {s.get("id"): s for s in explain_catalog.get("walk_fixed") or ()}
    missing = [i for i in (*head, *tail) if i not in fixed]
    if missing:
        raise ShowError(f"the explain catalog's walk_fixed has no step {missing}")
    steps = [_walk_step(fixed[i]) for i in head]
    for kind in block_types(netspec.get("blocks")):
        entry = (explain_catalog.get("blocks") or {}).get(kind)
        if entry is None or kind in head or kind in tail:
            raise ShowError(f"block type {kind!r} has no walk step of its own in the explain catalog")
        steps.append(_walk_step({"id": kind, **entry["walk"], "code": entry["code"]}))
    return steps + [_walk_step(fixed[i]) for i in tail]


def _walk_step(src: Mapping) -> dict:
    code = src["code"]
    return {"id": src["id"], "title": src.get("title", ""), "text": src.get("text", ""),
            "code": {"file": code["file"], "at": list(code["at"]), "after": code.get("after"),
                     "to": code.get("to")}}


def metric_record(netspec: Mapping, catalogs: Catalogs) -> dict:
    """primary, its direction from the vega-forge catalog's `metric_direction` (the spec's
    `selection.mode` must agree), its drawing bounds, and the secondary metrics."""
    metrics = netspec.get("metrics") or {}
    primary, direction = metrics.get("primary"), catalogs.direction(metrics.get("primary"))
    if direction not in ("max", "min"):
        raise ShowError(f"primary metric {primary!r} has no max/min direction in the vega-forge "
                        "catalog's metric_direction (R-C7)")
    mode = (metrics.get("selection") or {}).get("mode")
    if mode is not None and mode != direction:
        raise ShowError(f"netspec metrics.selection.mode {mode!r} contradicts the catalog's "
                        f"direction {direction!r} for {primary}")
    bounds = METRIC_BOUNDS.get(primary)
    if bounds is None:
        raise ShowError(f"primary metric {primary!r} has no bounds in METRIC_BOUNDS")
    secondary = list(dict.fromkeys(m for m in metrics.get("secondary") or () if m != primary))
    unknown = [m for m in secondary if catalogs.direction(m) is None]
    if unknown:
        raise ShowError(f"secondary metrics {unknown} are not in the catalog's metric_direction")
    return {"primary": primary, "direction": direction, "bounds": list(bounds),
            "secondary": secondary, "source": METRIC_SOURCE}


def views_available(examples: Mapping | None, story: Mapping, serve: Mapping | None) -> dict:
    """The tour views this lane can show, in tour order, and `{view: reason key}` for the rest
    (the rules of the module's table)."""
    if not isinstance(serve, Mapping) or serve.get("present") is False:
        raise ShowError("the bundle has no service (MANIFEST.serve.present) — no showcase without "
                        "one; re-pack with --serve")
    routes = serve.get("endpoints")
    if not isinstance(routes, list):
        raise ShowError("MANIFEST.serve.endpoints is missing (R-D1): the page's routes are unknown")
    sections = {s.get("id") for s in story.get("sections") or () if isinstance(s, Mapping)}
    off = {v: "views.off.no_section" for v in ("context", "network", "results") if v not in sections}
    if not any(r in routes for r in _SINGLE_ROUTES):
        off["try"] = "views.off.no_route"
    elif not _has_sample(examples):
        off["try"] = "views.off.no_examples"
    if _MANY_ROUTE not in routes:
        off["many"] = "views.off.no_route"
    tour = CATALOG["views"]["tour"]
    return {"order": [v for v in tour if v not in off], "off": {v: off[v] for v in tour if v in off}}


def _has_sample(examples: Mapping | None) -> bool:
    if not isinstance(examples, Mapping):
        return False
    demo, test = examples.get("demo") or {}, examples.get("test") or {}
    return bool(examples.get("presets") or test.get("rows") or demo.get("items"))


def reference_record(lane: Mapping, scores: Mapping | None) -> dict | None:
    """The row-score quantiles of the demo sample (training part), on the lanes whose answer is
    read against them (`reconstruction`, `anomaly`); None on every other lane."""
    if lane["view_key"] not in ("reconstruction", "anomaly"):
        return None
    values = (scores or {}).get("scores")
    if not isinstance(values, (list, tuple)) or not values:
        raise ShowError(f"a {lane['view_key']} lane needs the demo sample's row scores "
                        "(records.reference_scores) for page.json.reference")
    floats = [float(v) for v in values]
    if not all(math.isfinite(v) for v in floats):
        raise ShowError(f"{sum(not math.isfinite(v) for v in floats)} demo row score(s) are not "
                        "finite — the bundle scored the demo sample badly")
    ordered = sorted(floats)
    return {"kind": _REFERENCE_KIND[lane["link"]],
            "quantiles": {q: _nine(_quantile(ordered, float(q))) for q in REFERENCE_QUANTILES},
            "n": len(ordered), "source": str(scores.get("source") or "examples/demo.json (train)")}


def _quantile(ordered: list[float], q: float) -> float:
    """The linear-interpolation quantile (numpy's default) of an ascending list."""
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _nine(value: float) -> float:
    """Nine significant digits (R-F8)."""
    return float(f"{value:.9g}")


def page_records(store, examples: Mapping | None = None,
                 reference_scores: Mapping | None = None) -> dict:
    """`records` for `build_page_config` from the bundle's `RecordStore` (every file read is
    verified against `MANIFEST.files`) plus what the build read or ran itself."""
    shape = {part: store.read(f"records/shape.json#{part}") for part in ("manifest", "recipe")}
    try:
        analysis = {"context": store.read("records/analysis.json#context")}
    except FactMissing:
        analysis = None
    return {"manifest": store.manifest(), "manifest_digest": store.manifest_digest(),
            "shape": shape, "analysis": analysis, "examples": examples,
            "reference_scores": reference_scores}


# --------------------------------------------------------------------------- assembly

def _required(records: Mapping, key: str):
    value = records.get(key)
    if value is None:
        raise ShowError(f"page_config needs records[{key!r}]")
    return value


def _authored_plan(paths) -> Mapping:
    plan, why = read_json(paths.plan, "the authored story plan")
    if plan is None:
        raise ShowError(why)
    if plan.get("schema") != "vega/story-plan@2" or plan.get("status") != "authored":
        raise ShowError(f"{paths.plan} is not an authored vega/story-plan@2 "
                        f"(schema {plan.get('schema')!r}, status {plan.get('status')!r})")
    return plan


def _template(catalogs: Catalogs, plan: Mapping) -> dict:
    """The installed template's identity; the plan pinned the template it was drafted against
    (V1), and the two must agree. Before W15 writes `TEMPLATE.json`, the plan's pin is used."""
    pinned, installed = plan.get("template"), catalogs.template
    if installed is not None and pinned is not None and dict(installed) != dict(pinned):
        raise ShowError(f"the plan was drafted against template {dict(pinned)}, the installed one "
                        f"is {dict(installed)} — re-draft the plan (V1)")
    chosen = installed if installed is not None else pinned
    if not isinstance(chosen, Mapping):
        raise ShowError("no template identity: neither TEMPLATE.json nor the plan's template pin")
    return {"version": chosen.get("version"), "manifest_digest": chosen.get("manifest_digest")}


def _identity(manifest: Mapping) -> dict:
    identity = ((manifest.get("serve") or {}).get("identity")) or {}
    if not isinstance(identity.get("title"), str) or not identity["title"]:
        raise ShowError("MANIFEST.serve.identity.title is missing (R-E6)")
    return {"title": identity["title"], "description": str(identity.get("description") or ""),
            "source": IDENTITY_SOURCE}


def _motif(plan: Mapping, modality: str) -> dict:
    """The motif the authored plan settled, with the source of its settling (module docstring)."""
    proposal, chosen = (plan.get("proposals") or {}).get("motif") or {}, (plan.get("authored") or {}).get("motif")
    if proposal.get("modality") != modality:
        raise ShowError(f"the plan proposed a motif for modality {proposal.get('modality')!r}; "
                        f"the records give {modality!r} — re-draft the plan")
    if not isinstance(chosen, str):
        raise ShowError("the authored plan settles no motif (authored.motif)")
    base = CATALOG["motifs"]["by_modality"][modality]["base"]
    stance = proposal.get("stance")
    if stance:
        return {"id": chosen, "base": base, "modality": modality,
                "source": _stance_source(stance, chosen), "reason": f"recorded stance: show: motif {chosen}"}
    if chosen == base:
        return {"id": chosen, "base": base, "modality": modality, "source": "vega:default",
                "reason": "default applied (unattended)"}
    return {"id": chosen, "base": base, "modality": modality, "source": "user:S3",
            "reason": "the answer to fork S3"}


def _stance_source(stance: str, chosen: str) -> str:
    date, named = _STANCE_DATE.search(stance), _STANCE_MOTIF.search(stance)
    if date is None:
        raise ShowError(f"the plan's motif stance {stance!r} carries no date")
    if named is not None and named.group(1) != chosen:
        raise ShowError(f"the stance says motif {named.group(1)!r}, the plan settled {chosen!r}")
    return f"show:{date.group(0)}"


def _explain(netspec: Mapping, lane: Mapping, catalogs: Catalogs) -> dict:
    """The explain catalog's entries for this lane's block types, activations and link,
    verbatim, with the catalog's path and digest."""
    catalog, blocks = catalogs.explain, netspec.get("blocks")
    subset = {"blocks": block_types(blocks), "activations": activation_names(blocks, catalogs.forge),
              "links": [lane["link"]]}
    out = {"catalog": catalogs.explain_ref, "digest": catalogs.explain_digest}
    for part, names in subset.items():
        table = catalog.get(part) or {}
        missing = [n for n in names if n not in table]
        if missing:
            raise ShowError(f"the explain catalog has no {part} entry for {missing}")
        out[part] = {n: copy.deepcopy(table[n]) for n in names}
    return out


def _check_story(story: Mapping, lane: Mapping, metric: Mapping, walk: list) -> None:
    """The story was resolved for this network: its task agrees with the lane and the metric,
    and every code note is keyed by a walk id."""
    if story.get("schema") != "vega/story@2":
        raise ShowError(f"the story is not vega/story@2 (schema {story.get('schema')!r})")
    task = story.get("task") or {}
    want = {"kind": lane["task"], "base": lane["base"], "view_key": lane["view_key"],
            "sample_shape": lane["shape"], "graph_level": lane["level"], "family": lane["family"],
            "goal": lane["goal"], "primary": metric["primary"], "direction": metric["direction"]}
    differ = {k: (task.get(k), v) for k, v in want.items() if task.get(k) != v}
    if differ:
        raise ShowError(f"the story was resolved for another network: {differ} (story, netspec)")
    ids = {s["id"] for s in walk}
    stray = sorted(set(story.get("code_notes") or {}) - ids)
    if stray:
        raise ShowError(f"story code notes {stray} are not walk ids of this network")
