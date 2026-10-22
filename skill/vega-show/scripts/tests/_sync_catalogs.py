"""
_sync_catalogs.py — gate CS1's second half: `catalog.json` against the catalogs vega-show reads
from. The vega-forge catalog (tasks, sequence bases, graph-level tasks, families, blocks, sample
shapes, decision families, `metric_direction` — which must be present, because vega-show keeps no
copy — and, once they exist, `inspection.links`, `inspection.target_kinds` and
`inspection.targets`), the vega-eval catalog (baselines, the metric set and the primary per task)
and the backend's `METRICS`. A helper of `test_catalog_sync.py`.
"""
from __future__ import annotations

import ast
import re

from _sync_common import (FIXED_WALK, FORGE_COVERAGE, FORGE_METRICS, LINKS,
                          UNLABELLED_ANOMALY_PRIMARY, dig, fail, load_json, ok, same, show_metrics)

def metric_directions(forge):
    """The forge catalog's metric_direction as {metric: direction}, whichever way K1 keys it
    ({metric: dir} or {dir: [metrics]})."""
    table = forge.get("metric_direction")
    if not isinstance(table, dict):
        return None
    if set(table) <= {"max", "min", "none"} and all(isinstance(v, list) for v in table.values()):
        return {m: d for d, ms in table.items() for m in ms}
    out = {}
    for metric, row in table.items():
        out[metric] = row.get("direction") or row.get("mode") if isinstance(row, dict) else row
    return out


def check_forge(cat, forge):
    tasks = set(forge["tasks"])
    for key in ("results_facts", "breakdown.by_task", "view_keys", "polarity.by_task"):
        same(f"{key} covers the forge task kinds", set(dig(cat, key)), tasks)
    bases = {}
    for base, seq in forge["sequence_map"].items():
        bases.setdefault(seq, set()).add(base)
    for seq, want in bases.items():
        keys = cat["view_keys"].get(seq)
        same(f"view_keys.{seq} bases", set(keys) if isinstance(keys, dict) else keys, want)
        if cat["breakdown"]["by_task"].get(seq) != "base":
            fail(f"breakdown.by_task.{seq} must be 'base' (a sequence task reads its base's row)")
    rules = cat["roles"]["neutral_rules"]
    if not set(rules) <= tasks:
        fail(f"roles.neutral_rules names unknown tasks {sorted(set(rules) - tasks)}")
    if not set(forge["graph_level"]["tasks"]) <= tasks:
        fail("the forge graph_level tasks are not all task kinds")
    shapes = forge["sample_shapes"]
    same("context steps cover the sample shapes", set(cat["context_steps"]["by_shape"]), set(shapes))
    same("motif modalities are the shapes plus spatial-row", set(cat["motifs"]["by_modality"]),
         set(shapes) | {"spatial-row"})
    lex_shapes = {s for s in shapes if s != "graph"} | {"graph:node", "graph:graph"}
    same("lexicon.required covers every shape and graph level", set(cat["lexicon"]["required"]),
         lex_shapes)
    same("decision families (vega-forge catalog)", cat["decision_families"],
         forge["decision_families"])
    decisions = {d for ds in forge["decision_families"].values() for d in ds} - {"blocks"}
    same("focus.by_decision covers every non-block decision", set(cat["focus"]["by_decision"]),
         decisions)
    same("focus families", set(cat["focus"]["families"]), set(forge["decision_families"]))
    _check_forge_blocks(cat, forge, shapes)
    _check_forge_inspection(cat, forge)


def _check_forge_blocks(cat, forge, shapes):
    blocks = set(forge["blocks"])
    walk_blocks = set(cat["walk_roles"]) - set(FIXED_WALK)
    same("walk_roles has one row per forge block type", walk_blocks, blocks)
    clash = blocks & set(FIXED_WALK)
    if clash:
        fail(f"block types collide with fixed walk ids: {sorted(clash)}")
    if not set(cat["choice_skip_types"]) <= blocks:
        fail("choice_skip_types names unknown block types "
             f"{sorted(set(cat['choice_skip_types']) - blocks)}")
    for name, family in forge["families"].items():
        if family.get("sample_shape") not in shapes:
            fail(f"family {name}: sample shape {family.get('sample_shape')!r} is unknown")
        if not set(family.get("block_types", [])) <= set(cat["walk_roles"]):
            fail(f"family {name}: block types without walk roles")
        if not set(family.get("floor_shapes", [])) <= set(shapes):
            fail(f"family {name}: floor shapes outside the sample shapes")
    ok(f"{len(forge['families'])} families and {len(blocks)} block types have walk roles and "
       "known shapes")


def primary_metrics(forge, evaluation, vocabulary):
    """The task primaries, which vega-show states as `values.primary` and whose direction it
    reads (`page.json` `metric.direction`, `story@2` `task.direction`): each vega-forge
    `tasks[].primary` and `primary_alt.metric`, each metric a vega-eval `tasks[].primary` names,
    and R-C7's unlabelled-anomaly primary. `vocabulary` tells a metric from prose."""
    names = {UNLABELLED_ANOMALY_PRIMARY}
    for row in forge["tasks"].values():
        alt = row.get("primary_alt") if isinstance(row.get("primary_alt"), dict) else {}
        names |= {m for m in (row.get("primary"), alt.get("metric")) if m in vocabulary}
    for row in evaluation["tasks"].values():
        words = re.findall(r"[a-z_0-9]+", str(row.get("primary", "")))
        names |= {w for w in words if w in vocabulary}
    return names


def _check_forge_metrics(cat, forge, directions, primaries):
    if directions is None:
        fail("the vega-forge catalog has no metric_direction table (R-C7, WP K1); vega-show reads "
             "every metric's direction there and keeps no copy")
        return
    directed = show_metrics(cat, directed_only=True)
    for metric in sorted(directed | primaries):
        what = "a task primary" if metric not in directed else "a results fact or direction_of"
        if directions.get(metric) not in ("max", "min"):
            fail(f"metric {metric} ({what}, whose direction vega-show reads) has no max/min "
                 f"direction in the vega-forge catalog's metric_direction "
                 f"(got {directions.get(metric)!r})")
    for metric in sorted(show_metrics(cat) - directed):
        if metric not in directions:
            fail(f"metric {metric} (a breakdown vega-show names) is not a row of metric_direction")
    same(f"R-C7: {UNLABELLED_ANOMALY_PRIMARY} is read as min",
         directions.get(UNLABELLED_ANOMALY_PRIMARY), "min")
    alt = forge["tasks"].get("anomaly", {}).get("primary_alt")
    if isinstance(alt, dict):
        same("R-C7: the vega-forge anomaly primary_alt", alt.get("metric"),
             UNLABELLED_ANOMALY_PRIMARY)
    ok(f"{len(show_metrics(cat) | primaries)} metric(s) vega-show names are rows of "
       "metric_direction")


def _check_forge_inspection(cat, forge):
    inspection = forge.get("inspection") if isinstance(forge.get("inspection"), dict) else {}
    links = inspection.get("links")
    if links is None:
        ok("the forge catalog has no inspection.links yet; page.json's link enum is checked "
           "against R-C1's eight values")
    else:
        names = list(links) if isinstance(links, dict) else [
            (x.get("name") or x.get("id")) if isinstance(x, dict) else x for x in links]
        same("R-C1 links (vega-forge catalog inspection.links)", set(names), set(LINKS))
    kinds = inspection.get("target_kinds")
    if isinstance(kinds, (dict, list)):
        same("attribution targets are the forge catalog's inspection.target_kinds",
             sorted(kinds), sorted(cat["attribution"]["targets"]))
    targets = inspection.get("targets")
    if isinstance(targets, dict):
        used = set()
        for row in targets.values():
            if isinstance(row, dict):
                used |= {row["default"]} if row.get("default") else set()
                used |= set(row.get("selectable") or {})
        extra = sorted(used - set(cat["attribution"]["targets"]))
        if extra:
            fail(f"inspection targets the attribution schema cannot name: {extra}")


def backend_metrics():
    """The keys of the METRICS dict literal in the backend's metrics.py, read with ast."""
    try:
        tree = ast.parse(FORGE_METRICS.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        fail(f"the backend's metrics.py does not parse: {exc}")
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "METRICS"
                                                for t in node.targets):
            if isinstance(node.value, ast.Dict):
                return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    fail("the backend's metrics.py has no METRICS dict literal")
    return set()


def check_eval_and_metrics(cat, evaluation, known):
    """The results facts and breakdowns against the backend's METRICS, its coverage.json and the
    vega-eval catalog; the primaries are the forge sync's to check against METRICS (R-C7)."""
    coverage = load_json(FORGE_COVERAGE, "the backend's coverage.json") or {}
    declared = set(coverage.get("metrics") or [])
    unknown = sorted(show_metrics(cat) - known)
    undeclared = sorted(show_metrics(cat) - declared)
    if unknown or undeclared:
        fail(f"metrics vega-show names that the backend does not implement ({unknown}) or its "
             f"coverage.json does not declare ({undeclared})")
    else:
        ok(f"every metric vega-show names is a backend METRICS key and a coverage.json metric")
    rows = evaluation["baselines"]["rows"]
    mapping = cat["result_models"]["from_baseline"]
    for name in list(rows) + evaluation["baselines"]["default_set"]:
        if name not in mapping:
            fail(f"the vega-eval baseline {name!r} has no result model name "
                 "(result_models.from_baseline)")
    tasks = evaluation["tasks"]
    for task, metrics in cat["results_facts"].items():
        offered = _eval_metric_set(tasks, task, known)
        if not set(metrics) <= offered:
            fail(f"results_facts.{task} names {sorted(set(metrics) - offered)}, which the vega-eval "
                 "catalog does not compute for that task")
        row = cat["breakdown"]["by_task"].get(task)
        if isinstance(row, dict) and row["metric"] not in offered:
            fail(f"breakdown.by_task.{task}: {row['metric']} is not in the task's vega-eval metric set")
    ok(f"{len(cat['results_facts'])} task rows agree with the vega-eval catalog")


def _eval_metric_set(tasks, task, known):
    row = tasks.get(task) or {}
    names = set(row.get("secondary") or [])
    names |= {w for w in re.findall(r"[a-z_0-9]+", str(row.get("primary", ""))) if w in known}
    return names


def check_catalogs(cat, forge, evaluation):
    """Every check of this module; returns the task primaries (the page's `metric.{metric}`
    keys cover them too)."""
    known = backend_metrics()
    directions = metric_directions(forge)
    primaries = primary_metrics(forge, evaluation, known | set(directions or {}))
    check_forge(cat, forge)
    _check_forge_metrics(cat, forge, directions, primaries)
    check_eval_and_metrics(cat, evaluation, known)
    return primaries
