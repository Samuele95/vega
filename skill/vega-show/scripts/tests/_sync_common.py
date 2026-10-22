"""
_sync_common.py — what the catalog sync's parts share: the paths of the files it reads, the few
values the build plan fixes and the sync compares against, the failure list and the reporting
helpers. A helper of `test_catalog_sync.py`; it runs nothing on import.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILL = SCRIPTS.parent
SKILLS = SKILL.parent
CATALOG_JSON = SCRIPTS / "catalog.json"
CONTRACT_MD = SKILL / "references" / "show-contract.md"
SCHEMA_DIR = SKILL / "references" / "schemas"
PAGE_DIR = SKILL / "assets" / "page"
FORGE = SKILLS / "vega-forge" / "scripts"
FORGE_CATALOG = FORGE / "catalog.json"
FORGE_COVERAGE = FORGE / "backends" / "pytorch" / "coverage.json"
FORGE_METRICS = FORGE / "backends" / "pytorch" / "templates" / "metrics.py"
EVAL_CATALOG = SKILLS / "vega-eval" / "scripts" / "catalog.json"
VERBOSE = "-v" in sys.argv

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
# R-C1: the one link vocabulary (build plan §1.3); the forge catalog's inspection.links, once
# written, must hold the same eight values, and page.json's lane.link enum copies them
LINKS = ["softmax", "sigmoid", "sigmoid_each", "identity", "cumulative", "reconstruction",
         "distance", "horizon"]
# R-C7: the primary of an anomaly lane with no label; the page reads its direction and renders
# it through metric.anomaly_score_mean like any other primary
UNLABELLED_ANOMALY_PRIMARY = "anomaly_score_mean"
FIXED_WALK = ("root", "store", "build", "forward", "link", "shape", "run", "explain", "route")
BRIEF_STAGES = ["forge-data", "lens", "shape", "blueprint", "net", "train", "eval", "pack"]

failures = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    if VERBOSE:
        print(f"  ok    {msg}")


def same(what, got, want):
    """One comparison, reported either way."""
    if got == want:
        ok(what)
        return True
    fail(f"{what}: {got!r} != {want!r}")
    return False


def load_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"{what} ({path}) does not read as JSON: {exc}")
        return None


def dig(doc, dotted):
    """The value at a dotted catalog key, or a KeyError naming it."""
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def show_metrics(cat, directed_only=False):
    """Every metric the catalog names; with directed_only, the ones whose direction vega-show
    reads (the results facts and each breakdown's direction_of — a breakdown's own per-unit
    metric, such as f1_per_class, is a vector with no direction of its own). The task primaries
    come from the other catalogs (`_sync_catalogs.primary_metrics`)."""
    names = {m for ms in cat["results_facts"].values() for m in ms}
    for row in cat["breakdown"]["by_task"].values():
        if isinstance(row, dict):
            names |= {row["direction_of"]} if directed_only else {row["metric"], row["direction_of"]}
    return names
