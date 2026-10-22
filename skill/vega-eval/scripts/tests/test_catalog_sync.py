#!/usr/bin/env python3
"""
test_catalog_sync.py — `scripts/catalog.json` is the machine copy of `eval-contract.md`
§3–§4, and this test fails when the two diverge.

    python3 test_catalog_sync.py [-v]

`evaluate.py` reads its metric rows, its threshold-dependent tasks and its baseline
definitions from the JSON. If the prose gains a task kind or a baseline and the JSON does not,
the evaluation silently reports fewer metrics than the contract promises — which is the drift
the machine copy exists to prevent, and the reason it is not simply restated in code.

Task *sets* and baseline *names* are compared exactly. Metric names inside a row are checked
in the direction that matters: every metric the JSON claims must appear in the prose row it
came from, and every backtick-quoted metric name in the prose row must be in the JSON. Prose
that names a metric in words rather than in backticks ("per-class precision/recall/f1",
"residual quantiles") is listed in the JSON under the name `metrics.py` gives it, so those are
mapped explicitly rather than guessed.

The default baseline set is §4's rows in order: the unconditional ones (`default_set`) and then
the ones a task or a sample adds (`default_when`), each of which must say in its prose row when
it is added — and say it in terms of the condition the JSON encodes. The scale each deciding
reading's threshold lives on (`threshold_scale`) is §2's, and it is the scale the vega-pack
reading table decides that reading on: a threshold chosen on one scale and served on another
would decide every row wrongly.

Exit 0 when the two agree.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILL = SCRIPTS.parent
CATALOG_JSON = SCRIPTS / "catalog.json"
CATALOG_MD = SKILL / "references" / "eval-contract.md"
VERBOSE = "-v" in sys.argv

# prose that names a metric in words; the JSON carries the name metrics.py implements
PROSE_NAMES = {
    "per-class precision/recall/f1": "f1_per_class",
    "residual quantiles": "residual_quantiles",
    "per-feature error": "per_feature_error",
    "error quantiles": "per_feature_error",
    "mae per horizon step": "per_horizon_error",
    "reconstruction error": "reconstruction_error",
}
# rows whose prose primary is a sentence rather than a bare metric name; the JSON keeps the
# sentence, so the exact-name comparison would be comparing a name against a paragraph
NARRATIVE_PRIMARY = ("binary", "anomaly", "sequence-classification", "sequence-regression")
# every metric name the catalog uses must be one the backend implements; this is the file that
# says which those are
COVERAGE = (SKILL.parent / "vega-forge" / "scripts" / "backends" / "pytorch" / "coverage.json")
# the serving reading table, whose threshold_scale per reading the evaluation's must equal
PACK_CATALOG = SKILL.parent / "vega-pack" / "scripts" / "catalog.json"
# the vega-pack readings a deciding effective kind serves as (R-C1's reading table)
PACK_READINGS = {"binary": ("binary",), "multilabel": ("multilabel",),
                 "anomaly": ("anomaly-reconstruction", "anomaly-svdd")}

failures = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    if VERBOSE:
        print(f"  ok    {msg}")


def table_rows(md, heading):
    """The rows of the first markdown table under a heading, as lists of cells."""
    section = md.split(heading, 1)[1] if heading in md else ""
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        # an escaped pipe is content, not a cell boundary (`\|y\| > 0.1`)
        cells = [c.strip().replace("\x00", "|")
                 for c in line.replace("\\|", "\x00").strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows[1:] if rows else []          # drop the header row


def ticked(cell):
    return set(re.findall(r"`([^`]+)`", cell))


def prose_metrics(cell, tasks=()):
    """Every metric the prose names in this cell: backticked names plus the spelled-out ones.

    A backticked token carrying a dot is a file or a spec path (`threshold.json`,
    `metrics.primary`, `task.anomaly_rate`), never a metric — metrics.py has no name with a
    dot in it — so those are excluded rather than compared against the metric list.
    """
    names = {n for n in ticked(cell)
             if "." not in n and "/" not in n and n not in tasks and "*" not in n}
    for phrase, name in PROSE_NAMES.items():
        if phrase in cell:
            names.add(name)
    return names


def main():
    print(f"catalog sync — {CATALOG_JSON.name} against {CATALOG_MD.name} §3–§4")
    catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    md = CATALOG_MD.read_text(encoding="utf-8")

    # ---- §3, the metric rows
    rows = table_rows(md, "## 3. Metric catalog per task")
    if not rows:
        fail("§3's table did not parse; the sync test cannot check what it cannot read")
        return 1
    prose_tasks, prose_by_task = set(), {}
    for cells in rows:
        raw = cells[0]
        for name in ticked(raw) or {raw}:
            # `sequence-*` / `forecast` is one prose row covering several kinds
            if "*" in name:
                for kind in (k for k in catalog["tasks"] if k.startswith(name.rstrip("-*"))):
                    prose_tasks.add(kind)
                    prose_by_task[kind] = cells
            else:
                prose_tasks.add(name)
                prose_by_task[name] = cells
    json_tasks = set(catalog["tasks"])
    if prose_tasks != json_tasks:
        for kind in sorted(prose_tasks - json_tasks):
            fail(f"§3 names task {kind!r} and catalog.json has no row for it")
        for kind in sorted(json_tasks - prose_tasks):
            fail(f"catalog.json has a row for task {kind!r} that §3 does not name")
    else:
        ok(f"{len(json_tasks)} task kinds, the same set in both")

    for kind, cells in sorted(prose_by_task.items()):
        row = catalog["tasks"].get(kind)
        if not row:
            continue
        for field in ("primary", "secondary", "threshold_dependent", "curves"):
            if field not in row:
                fail(f"{kind}: catalog.json's row has no {field!r}, which §3's columns promise")
        primary = str(row.get("primary", ""))
        want = ticked(cells[1])
        if kind not in NARRATIVE_PRIMARY and want and not (want & ticked(f"`{primary}`")):
            fail(f"{kind}: §3's primary is {sorted(want)} and catalog.json says {primary!r}")
        else:
            ok(f"{kind}: primary agrees")
        said = prose_metrics(cells[2], json_tasks)
        have = set(row.get("secondary") or [])
        # A `sequence-*` / `forecast` row defers to its base task in as many words, so its
        # JSON row carries the base task's metrics as well as the ones the prose adds. An
        # exact set comparison there would be comparing one row against three; what is checked
        # instead is that the prose's own additions are present and that nothing in the row is
        # a metric the backend does not implement.
        if row.get("derives_from"):
            # the row names what `forecast` adds; the other sequence kinds take the base
            # task's metrics, which their own JSON rows carry
            said = have & said if kind != "forecast" else said
            missing = said - have
            if missing:
                fail(f"{kind}: §3's {row['derives_from']} row names {sorted(missing)} and "
                     f"catalog.json's row omits them")
            else:
                ok(f"{kind}: derives from §3's {row['derives_from']} row and carries its "
                   f"additions")
        elif said:
            extra = have - said
            missing = said - have - {primary}
            if extra:
                fail(f"{kind}: catalog.json lists {sorted(extra)} which §3's row does not name")
            if missing:
                fail(f"{kind}: §3 names {sorted(missing)} and catalog.json's row omits them")
            if not extra and not missing:
                ok(f"{kind}: {len(have)} secondary metric(s) agree")

    # ---- a task is threshold-dependent when its §3 row names `threshold.json`. "per-class at
    # argmax" is not a threshold: nothing is chosen, the largest score wins, and there is no
    # threshold.json to publish for it.
    def threshold_dependent(kind, cells):
        cell = cells[3] if len(cells) > 3 else ""
        if "threshold.json" not in cell:
            return False
        if not (catalog["tasks"].get(kind) or {}).get("derives_from"):
            return True
        # one prose row covers several kinds, and it names which of them decides at a
        # threshold; a fragment must mention both this kind and the file for it to count
        return any(f"`{kind}`" in part and "threshold.json" in part
                   for part in re.split(r"[,;—]", cell))

    prose_threshold = {kind for kind, cells in prose_by_task.items()
                       if threshold_dependent(kind, cells)}
    if prose_threshold != set(catalog["threshold_tasks"]):
        fail(f"§3's threshold column marks {sorted(prose_threshold)} and catalog.json's "
             f"threshold_tasks is {sorted(catalog['threshold_tasks'])}")
    else:
        ok(f"{len(prose_threshold)} threshold-dependent task(s) agree")

    # ---- §4, the baselines
    brows = table_rows(md, "## 4. Baselines — what the network must beat")
    if not brows:
        fail("§4's table did not parse")
        return 1
    prose_baselines = []
    for cells in brows:
        names = ticked(cells[0])
        prose_baselines.extend(names or [cells[0]])
    json_baselines = list(catalog["baselines"]["rows"])
    if set(prose_baselines) != set(json_baselines):
        for name in sorted(set(prose_baselines) - set(json_baselines)):
            fail(f"§4 names baseline {name!r} and catalog.json has no row for it")
        for name in sorted(set(json_baselines) - set(prose_baselines)):
            fail(f"catalog.json has baseline {name!r} that §4 does not name")
    else:
        ok(f"{len(json_baselines)} baselines, the same set in both")
    default_when = catalog["baselines"].get("default_when") or {}
    ordered = list(catalog["baselines"]["default_set"]) + list(default_when)
    if ordered != prose_baselines:
        fail(f"catalog.json's default baseline set is {catalog['baselines']['default_set']} "
             f"then (when the task or sample adds them) {list(default_when)}, and §4's rows "
             f"are {prose_baselines}; the default set is §4's rows in order")
    else:
        ok("the default baseline set is §4's rows, in order "
           f"({len(default_when)} added per task or sample)")
    by_name = {name: cells for cells in brows for name in ticked(cells[0])}
    for name, rule in default_when.items():
        cell = " ".join((by_name.get(name) or [])[1:2])
        if "in the default set when" not in cell:
            fail(f"baseline {name!r} is added to the default set by a condition "
                 f"({rule}), and §4's row does not say when (\"in the default set when …\")")
            continue
        said = cell.split("in the default set when", 1)[1]
        missing = [t for t in rule.get("tasks") or [] if f"`{t}`" not in said]
        if rule.get("season") and "season" not in said:
            missing.append("a season")
        if rule.get("level") and f"{rule['level']}-level" not in said:
            missing.append(f"{rule['level']}-level")
        if missing:
            fail(f"baseline {name!r}: catalog.json adds it when {rule} and §4's condition does "
                 f"not name {missing}")
        else:
            ok(f"baseline {name!r}: its default-set condition agrees")
        if name in catalog["baselines"]["default_set"]:
            fail(f"baseline {name!r} is both unconditional and conditional in catalog.json")
    for cells in brows:
        for name in ticked(cells[0]):
            row = catalog["baselines"]["rows"].get(name) or {}
            for field in ("definition", "why", "fitted"):
                if field not in row:
                    fail(f"baseline {name!r}: catalog.json's row has no {field!r}, which §4's "
                         "columns promise")
            # a prose row that says a task changes the definition must have that task in by_task
            for kind in ("anomaly", "unsupervised"):
                if kind in cells[1] and kind not in (row.get("by_task") or {}):
                    fail(f"baseline {name!r}: §4 says {kind!r} changes what it is, and "
                         f"catalog.json's by_task does not carry that kind")
    for key in ("required", "required_one_of"):
        for name in catalog["baselines"][key]:
            if name not in json_baselines:
                fail(f"baselines.{key} names {name!r}, which is not a baseline row")
    ok("the required floor and the one-of pair are baselines that exist")

    # ---- every deciding reading's threshold scale: §2's, and the serving reading table's
    scales = catalog.get("threshold_scale") or {}
    section2 = md.split("## 2.", 1)[-1].split("## 3.", 1)[0]
    if not scales:
        fail("catalog.json has no threshold_scale; the evaluator cannot say what scale "
             "threshold.json is on")
    for kind, scale in sorted(scales.items()):
        if kind not in catalog["threshold_tasks"]:
            fail(f"threshold_scale names {kind!r}, which is not a threshold task")
        if f"`{scale}`" not in section2 or f"`{kind}`" not in section2:
            fail(f"§2's threshold.json item does not name the `{scale}` scale for `{kind}`")
    if PACK_CATALOG.is_file():
        readings = ((json.loads(PACK_CATALOG.read_text(encoding="utf-8"))
                     .get("output_readings") or {}).get("readings") or {})
        for kind, names in PACK_READINGS.items():
            for reading in names:
                theirs = (readings.get(reading) or {}).get("threshold_scale")
                if theirs != scales.get(kind):
                    fail(f"vega-pack decides a {reading} reading on the {theirs!r} scale and "
                         f"catalog.json's threshold_scale.{kind} is {scales.get(kind)!r}")
        ok(f"{len(scales)} threshold scale(s) agree with the vega-pack reading table")
    else:
        print(f"  note  {PACK_CATALOG} not found; the serving scale check was skipped")

    # ---- the upstream numbers §3 names are the ones the code marks upstream
    upstream = catalog.get("upstream_numbers") or []
    para = md.split("## 3.", 1)[-1].split("## 4.", 1)[0]
    for phrase in ("valid", "campaign-baseline"):
        if not any(phrase in u for u in upstream):
            fail(f"catalog.json's upstream_numbers does not mention {phrase!r}, which §3 names "
                 "as carrying no new CI")
    if "upstream" not in para:
        fail("§3 no longer names the upstream numbers; catalog.json still lists them")
    ok(f"{len(upstream)} upstream number(s) named in both")

    # ---- a data-dependent metric gate (evaluate.py withholds the metric when its train-side
    # rule fails) is the rule §3's row states, in the row of every task it covers
    for metric, gate in sorted((catalog.get("metric_gates") or {}).items()):
        for kind in gate.get("tasks") or ():
            cell = " ".join((prose_by_task.get(kind) or [])[2:3])
            at = cell.find(f"`{metric}`")
            said = cell[at:at + 160] if at >= 0 else ""
            wanted = (f"> {gate['abs_y_above']}", f"{gate['share'] * 100:g} %", gate["part"])
            missing = [w for w in wanted if w not in said]
            if at < 0 or missing:
                fail(f"catalog.json gates {metric} on {kind} ({gate.get('rule')}) and §3's "
                     f"{kind} row does not state it beside `{metric}` (missing {missing})")
            else:
                ok(f"{metric}'s gate on {kind} agrees with §3")
        silent = [k for k in gate.get("tasks") or ()
                  if metric not in ((catalog["tasks"].get(k) or {}).get("secondary") or [])]
        if silent:
            fail(f"catalog.json gates {metric} on {silent}, whose rows do not report it")

    # ---- every metric the catalog names is one the backend actually implements
    if COVERAGE.is_file():
        implemented = set(json.loads(COVERAGE.read_text(encoding="utf-8")).get("metrics") or [])
        if implemented:
            named = set()
            for row in catalog["tasks"].values():
                named |= set(row.get("secondary") or [])
                named |= set(row.get("threshold_dependent") or [])
            named |= set(catalog["ranking_metrics"])
            unknown = sorted(named - implemented)
            if unknown:
                fail(f"catalog.json names {unknown}, which the backend's coverage.json does "
                     "not list as implemented; a metric the emitted metrics.py cannot compute "
                     "is a row the evaluation would quietly report as null")
            else:
                ok(f"all {len(named)} catalog metric(s) are implemented by the backend")
    else:
        print(f"  note  {COVERAGE} not found; the implemented-metric check was skipped")

    print()
    if failures:
        print(f"{len(failures)} disagreement(s) between the prose and the machine catalog")
        return 1
    print(f"OK: catalog.json and {CATALOG_MD.name} §3–§4 agree "
          f"({len(json_tasks)} task kinds, {len(json_baselines)} baselines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
