#!/usr/bin/env python3
"""
page_check — `check_page_config(page, network=None, catalogs=None)`: gate BD2 over one
`page.json` (`vega/page@1`). `[]` when valid; otherwise one `ERROR: BD2 <where>: <why>` line per
problem. `build_showcase.py` refuses a page config with any (exit 2).

    python3 page_check.py <page.json> [--network <network-view.json>]      # exit 0 / 2

It checks the schema (`references/schemas/page-1.schema.json`, draft 2020-12), then the rules the
schema cannot state, each against the catalogs rather than a copy:

- **lane** — `link` is one of the one link vocabulary (the vega-forge catalog's
  `inspection.links`, R-C1), and `view_key`, `link` and `goal` are the ones the task, its base
  and its score source give; a base only on a `sequence-*` task, a score source only on
  `anomaly`, a level only on a graph sample;
- **motif** — the id is in the modality's admissible list and the base is the modality's (R-E4;
  negative PN7), and the modality is the lane's shape (or `spatial-row` on a row);
- **metric** — the direction is the catalog's `metric_direction` and the bounds are
  `METRIC_BOUNDS`; every secondary metric is a catalog metric;
- **views**, **recipe**, **reference**, **walk**, **explain**, **code** — the tour partition,
  the closed op set, the reference kind of the lane's link, the R-E8 walk ids and the explain
  subset they name, the backend's language;
- **the network** (when given, `vega/network-view@2`) — its task kind, base and link, its sample
  shape and level and its family equal the lane's (the boot cross-check of PG10, done early).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Mapping

from page_derive import METRIC_BOUNDS, Catalogs, key_pattern, lane_fields
from vega_show_core import CATALOG, HERE, Problem, ShowError, read_json

__all__ = ["check_page_config"]

RULE = "BD2"
SCHEMA = HERE.parent / CATALOG["schemas"]["vega/page@1"]["file"]
_SHAPE_MODALITIES = {"row": ("row", "spatial-row"), "window": ("window",), "graph": ("graph",)}
_REFERENCE_KIND = {"reconstruction": "reconstruction_error", "distance": "distance"}


def check_page_config(page: Mapping, network: Mapping | None = None,
                      catalogs: Catalogs | None = None) -> list[str]:
    """Every problem of `page` as `ERROR: BD2 <where>: <why>`; [] when it is valid."""
    problems = _schema_problems(page)
    if isinstance(page, Mapping):
        cats = catalogs if catalogs is not None else Catalogs.load()
        for rule in (_lane, _motif, _metric, _views, _recipe, _reference, _walk, _explain, _code):
            problems += rule(page, cats)
        if network is not None:
            problems += _network(page, network) if isinstance(network, Mapping) else [
                _p("network", f"the network view is a {type(network).__name__}, not an object")]
    return [str(p) for p in problems]


def _p(where: str, message: str) -> Problem:
    return Problem(RULE, where, message)


def _schema_problems(page) -> list[Problem]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return [_p("(schema)", "jsonschema (draft 2020-12) is needed to validate vega/page@1; "
                               "install jsonschema>=4")]
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(page), key=lambda e: list(e.path))
    return [_p("/".join(map(str, e.path)) or "(root)", f"not vega/page@1: {e.message}")
            for e in errors]


def _part(page: Mapping, key: str) -> Mapping:
    value = page.get(key)
    return value if isinstance(value, Mapping) else {}


def _lane(page: Mapping, cats: Catalogs) -> list[Problem]:
    lane, forge = _part(page, "lane"), cats.forge
    kind, base, source = lane.get("task"), lane.get("base"), lane.get("score_source")
    links = list(forge["inspection"]["links"])
    out = [] if lane.get("link") in links else [
        _p("lane.link", f"{lane.get('link')!r} is not in the one link vocabulary {links} (R-C1)")]
    if kind not in forge["tasks"]:
        return out + [_p("lane.task", f"{kind!r} is not a vega-forge catalog task")]
    if str(kind).startswith("sequence-") != (base is not None):
        out.append(_p("lane.base", f"base {base!r} on task {kind} (a base iff sequence-*)"))
    if (kind == "anomaly") != (source is not None):
        out.append(_p("lane.score_source", f"{source!r} on task {kind} (a score source iff anomaly)"))
    if (lane.get("shape") == "graph") != (lane.get("level") is not None):
        out.append(_p("lane.level", f"level {lane.get('level')!r} on shape {lane.get('shape')!r}"))
    try:
        want = lane_fields(kind, base, source, cats)
    except ShowError as exc:
        return out + [_p("lane", str(exc))]
    return out + [_p(f"lane.{k}", f"{lane.get(k)!r}, but task {kind} (base {base}, score source "
                                  f"{source}) gives {v!r}") for k, v in want.items() if lane.get(k) != v]


def _motif(page: Mapping, cats: Catalogs) -> list[Problem]:
    motif, shape = _part(page, "motif"), _part(page, "lane").get("shape")
    table = cats.show["motifs"]["by_modality"]
    modality, chosen = motif.get("modality"), motif.get("id")
    if modality not in table:
        return [_p("motif.modality", f"{modality!r} is not a modality ({', '.join(table)})")]
    spec, out = table[modality], []
    if chosen not in spec["admissible"]:
        out.append(_p("motif.id", f"{chosen!r} is not admissible for the modality {modality} "
                                  f"({', '.join(spec['admissible'])}) — R-E4"))
    if motif.get("base") != spec["base"]:
        out.append(_p("motif.base", f"{motif.get('base')!r}, the base of {modality} is {spec['base']!r}"))
    if modality not in _SHAPE_MODALITIES.get(shape, ()):
        out.append(_p("motif.modality", f"{modality} is not a modality of the {shape!r} sample shape"))
    return out


def _metric(page: Mapping, cats: Catalogs) -> list[Problem]:
    metric = _part(page, "metric")
    primary, out = metric.get("primary"), []
    if metric.get("direction") != cats.direction(primary):
        out.append(_p("metric.direction", f"{metric.get('direction')!r}, the catalog's "
                                          f"metric_direction of {primary!r} is {cats.direction(primary)!r}"))
    bounds = METRIC_BOUNDS.get(primary)
    if bounds is None or list(bounds) != metric.get("bounds"):
        out.append(_p("metric.bounds", f"{metric.get('bounds')!r}, METRIC_BOUNDS of {primary!r} is "
                                       f"{None if bounds is None else list(bounds)!r}"))
    unknown = [m for m in metric.get("secondary") or () if cats.direction(m) is None]
    return out + ([_p("metric.secondary", f"{unknown} are not catalog metrics")] if unknown else [])


def _views(page: Mapping, cats: Catalogs) -> list[Problem]:
    views, tour = _part(page, "views"), list(cats.show["views"]["tour"])
    order, off = list(views.get("order") or ()), dict(views.get("off") or {})
    out = []
    if [v for v in tour if v in order] != order or set(order) & set(off):
        out.append(_p("views.order", f"{order} is not a tour-ordered subset disjoint from off"))
    if sorted(set(order) | set(off)) != sorted(tour):
        out.append(_p("views", f"order ∪ off is {sorted(set(order) | set(off))}, the tour is {tour}"))
    bad = [k for k in off.values() if not isinstance(k, str) or not key_pattern(k)]
    return out + ([_p("views.off", f"reasons {bad} are not strings keys")] if bad else [])


def _recipe(page: Mapping, cats: Catalogs) -> list[Problem]:
    known = cats.show["page"]["recipe_ops"]
    ops = [o for o in _part(page, "recipe").get("ops") or () if isinstance(o, Mapping)]
    return [_p(f"recipe.ops[{i}].op", f"{o.get('op')!r} is not a recipe op ({', '.join(known)})")
            for i, o in enumerate(ops) if o.get("op") not in known]


def _reference(page: Mapping, cats: Catalogs) -> list[Problem]:
    lane, reference = _part(page, "lane"), page.get("reference")
    wanted = lane.get("view_key") in ("reconstruction", "anomaly")
    if not wanted:
        return [] if reference is None else [_p("reference", f"a {lane.get('view_key')} lane has no reference")]
    if not isinstance(reference, Mapping):
        return [_p("reference", f"a {lane.get('view_key')} lane needs the demo sample's score quantiles")]
    out, kind = [], _REFERENCE_KIND.get(lane.get("link"))
    if reference.get("kind") != kind:
        out.append(_p("reference.kind", f"{reference.get('kind')!r}, the link {lane.get('link')} reads {kind!r}"))
    quantiles = reference.get("quantiles")
    try:
        values = [float(quantiles[q]) for q in sorted(quantiles, key=float)]
    except (AttributeError, TypeError, ValueError):
        return out + [_p("reference.quantiles", f"{quantiles!r} is not a map of numbers by quantile")]
    if values != sorted(values):
        out.append(_p("reference.quantiles", f"{quantiles} do not grow with the quantile"))
    return out


def _walk(page: Mapping, cats: Catalogs) -> list[Problem]:
    ids = [s.get("id") for s in page.get("walk") or () if isinstance(s, Mapping)]
    head, tail = list(cats.show["walk"]["head"]), list(cats.show["walk"]["tail"])
    out = [] if len(ids) == len(set(ids)) else [_p("walk", f"walk ids repeat: {ids}")]
    if ids[:len(head)] != head or ids[len(ids) - len(tail):] != tail:
        return out + [_p("walk", f"{ids} does not open with {head} and close with {tail} (R-E8)")]
    middle = ids[len(head):len(ids) - len(tail)]
    unknown = [t for t in middle if t not in cats.forge["blocks"]]
    if unknown:
        out.append(_p("walk", f"{unknown} are not block types of the vega-forge catalog"))
    named = list((_part(page, "explain").get("blocks") or {}))
    if sorted(middle) != sorted(named):
        out.append(_p("walk", f"block steps {middle} are not the explain subset's blocks {named}"))
    return out


def _explain(page: Mapping, cats: Catalogs) -> list[Problem]:
    explain, link = _part(page, "explain"), _part(page, "lane").get("link")
    out = []
    if list(explain.get("links") or {}) != [link]:
        out.append(_p("explain.links", f"{list(explain.get('links') or {})}, the lane's link is {link!r}"))
    for part in ("blocks", "activations"):
        table = cats.explain.get(part) or {}
        stray = [n for n in explain.get(part) or {} if n not in table]
        if stray:
            out.append(_p(f"explain.{part}", f"{stray} are not in the backend's explain catalog"))
    return out


def _code(page: Mapping, cats: Catalogs) -> list[Problem]:
    language = _part(page, "code").get("language")
    want = cats.coverage.get("language")
    return [] if language == want else [_p("code.language", f"{language!r}, the backend's is {want!r}")]


def _network(page: Mapping, network: Mapping) -> list[Problem]:
    """The lane against `/v1/network` (`vega/network-view@2`)."""
    if network.get("schema") != "vega/network-view@2":
        return [_p("network", f"not vega/network-view@2 (schema {network.get('schema')!r}; R-F7)")]
    lane, task, sample = _part(page, "lane"), network.get("task") or {}, network.get("sample") or {}
    family = network.get("family")
    theirs = {"task": task.get("kind"), "base": task.get("base"), "link": task.get("link"),
              "shape": sample.get("shape"), "level": sample.get("level"),
              "family": family.get("name") if isinstance(family, Mapping) else family}
    return [_p(f"lane.{k}", f"{lane.get(k)!r}, the network says {v!r} — the showcase was built "
                            "for another network") for k, v in theirs.items() if lane.get(k) != v]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="gate BD2: check one page.json (vega/page@1)")
    ap.add_argument("page")
    ap.add_argument("--network", help="a vega/network-view@2 document to cross-check the lane")
    args = ap.parse_args(argv)
    docs = [read_json(args.page, "page.json")] + ([read_json(args.network, "the network view")]
                                                   if args.network else [])
    unread = [why for doc, why in docs if doc is None]
    if unread:
        print("\n".join(f"ERROR: {RULE} {why}" for why in unread))
        return 2
    problems = check_page_config(docs[0][0], docs[1][0] if args.network else None)
    print("\n".join(problems) if problems else f"OK: {RULE} {args.page} is a valid vega/page@1")
    return 2 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
