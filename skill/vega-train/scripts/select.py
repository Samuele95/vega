#!/usr/bin/env python3
"""
select.py — apply the `ladder/selection` rule to the ledger, and record what it picked.

    python3 select.py --campaign <dir> [--override <run-id> --reason "…" --source user:<turn>]
                      [--json]

Ineligibility is **mechanical**, and the contract says so: a run that diverged (D1), never
learned (D2), was cut by the budget before its first improvement (D9), or is a pruned trial
cannot be selected *by the rule*. This script computes exactly that, plus the ranking and the
margin over the baseline, and writes them into `experiments.json`.

What it does not do is write the reading. Why this run and what it did not settle is the
skill's paragraph, and a script that generated it would be inventing judgement from numbers it
only sorted.

The user may still select an ineligible run — that is what `--override` is for. An override is
never silent: it carries a `user:` source and a reason, the layer shows it beside the rule's
own pick, and `apply_training_layer.py` refuses an ineligible selection that has no override
recorded.

Exit codes: 0 a selection was recorded; 2 an input error, or no eligible run exists and no
override was given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_train_core import load_catalog  # noqa: E402
import trainer as T  # noqa: E402


def eligibility(row, catalog):
    """(eligible, reasons) for one ledger row, by the rule alone."""
    rule = (catalog.get("ladder_rules") or {}).get("selection") or {}
    blocking = set(rule.get("ineligible_codes") or ("D1", "D2"))
    fired = set(row.get("diagnostics") or ())
    reasons = []
    for code in sorted(blocking & fired):
        name = ((catalog.get("diagnostics") or {}).get(code) or {}).get("name", code)
        reasons.append(f"{code} ({name})")
    if row.get("status") == "pruned":
        reasons.append("a pruned trial")
    if "D9" in fired and _cut_before_first_improvement(row):
        reasons.append("D9 (budget cut) before its first improvement")
    if row.get("status") == "failed":
        reasons.append("the run failed")
    if row.get("primary") is None:
        reasons.append("no valid primary was recorded")
    return (not reasons), reasons


def _cut_before_first_improvement(row):
    """A budget-cut run that never improved on its first epoch is not a result at all."""
    best = row.get("best_epoch")
    return best is None or int(best) <= 1


def rank(rows, mode):
    scored = [r for r in rows if r.get("primary") is not None]
    reverse = (mode == "max")
    return [r["id"] for r in sorted(
        scored, key=lambda r: (float(r["primary"]), -(r.get("params") or 0)),
        reverse=reverse)] + [r["id"] for r in rows if r.get("primary") is None]


def choose(rows, mode, catalog):
    """The rule: best valid primary in the mode's direction among eligible runs; ties → fewer
    parameters."""
    eligible = []
    for row in rows:
        ok, reasons = eligibility(row, catalog)
        row["eligible"] = ok
        row["ineligible_because"] = reasons
        if ok:
            eligible.append(row)
    if not eligible:
        return None, eligible
    best = (max if mode == "max" else min)(eligible, key=lambda r: float(r["primary"]))
    tied = [r for r in eligible
            if abs(float(r["primary"]) - float(best["primary"])) < 1e-12]
    if len(tied) > 1:
        with_params = [r for r in tied if r.get("params") is not None]
        best = min(with_params or tied, key=lambda r: (r.get("params") or 0, r["id"]))
    return best, eligible


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--override", default=None, metavar="RUN_ID",
                    help="select a run the rule did not pick; needs --reason and --source")
    ap.add_argument("--reason", default=None)
    ap.add_argument("--source", default=None, metavar="user:<turn>")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    camp, err = T.Campaign.read(a.campaign)
    if err:
        print(f"ERROR: {err}")
        return 2
    catalog, cat_err = load_catalog()
    if cat_err:
        print(f"ERROR: {cat_err}")
        return 2

    ledger = camp.experiments()
    rows = ledger.get("runs") or []
    if not rows:
        print("ERROR: the ledger has no runs; there is nothing to select from")
        return 2

    # A ledger row whose run directory is gone is not a result: something removed the records
    # the row summarises, and selecting it would publish a number nothing on disk supports.
    orphans = [r for r in rows if not (camp.run_dir(r["id"]) / "run.json").is_file()]
    if orphans:
        for row in orphans:
            print(f"WARN: {row['id']} has a ledger row but no records at "
                  f"{camp.run_dir(row['id'])}; it is excluded from the ranking and the "
                  "selection. A campaign is append-only — if these records were removed, the "
                  "honest repair is to re-run those rungs under new ids, not to select from a "
                  "row nothing backs")
        rows = [r for r in rows if r not in orphans]
        ledger["orphaned_rows"] = [r["id"] for r in orphans]
        if not rows:
            print("ERROR: every ledger row is orphaned; there is nothing to select from")
            return 2

    metrics = camp.spec.get("metrics") or {}
    mode = (metrics.get("selection") or {}).get("mode", "max")
    primary = metrics.get("primary")
    rule_pick, eligible = choose(rows, mode, catalog)
    order = rank(rows, mode)
    by_id = {r["id"]: r for r in rows}
    baseline = by_id.get("R0")

    if a.override:
        if a.override not in by_id:
            print(f"ERROR: --override {a.override!r} is not a run in this ledger")
            return 2
        if not (a.reason and a.source):
            print("ERROR: an override carries a reason and a user: source — a pick that "
                  "overrules a mechanical rule is never silent")
            return 2
        if not str(a.source).startswith("user:"):
            print(f"ERROR: --source {a.source!r} must be a user: source")
            return 2
        chosen = by_id[a.override]
    elif rule_pick is None:
        print("ERROR: no run is eligible by the rule:")
        for row in rows:
            print(f"    {row['id']}: {', '.join(row.get('ineligible_because') or ['—'])}")
        print("ERROR: select one deliberately with --override <id> --reason … --source "
              "user:<turn>, or add budget and re-run the rungs the budget cut")
        return 2
    else:
        chosen = rule_pick

    margin = None
    if baseline and baseline.get("primary") is not None and chosen.get("primary") is not None:
        delta = float(chosen["primary"]) - float(baseline["primary"])
        margin = round(delta if mode == "max" else -delta, 6)

    alternatives = []
    if rule_pick and chosen["id"] != rule_pick["id"]:
        alternatives.append({"run": rule_pick["id"],
                             "when": "the rule's own pick, which this override set aside"})
    elif baseline and chosen["id"] != "R0":
        alternatives.append({"run": "R0",
                             "when": "if the un-searched spec is preferred for "
                                     "interpretability"})

    selection = {
        "run": chosen["id"],
        "rule": (catalog["ladder_rules"]["selection"]["rule"]),
        "source": "vega:default", "catalog": "ladder/selection",
        "mode": mode, "primary": primary, "value": chosen.get("primary"),
        "eligible": bool(chosen.get("eligible")),
        "margin_over_baseline": margin,
        "alternatives": alternatives,
        "override": None if not a.override else {
            "source": a.source, "reason": a.reason,
            "rule_would_have_picked": rule_pick["id"] if rule_pick else None,
            "ineligible_because": chosen.get("ineligible_because") or []},
        "reading": ledger.get("selection", {}).get("reading", ""),
    }
    ledger["runs"] = rows
    ledger["ranking"] = {"by": f"valid.{primary}", "mode": mode, "order": order}
    ledger["selection"] = selection
    camp.write_experiments(ledger)

    if a.json:
        print(json.dumps(selection, indent=2, ensure_ascii=False))
        return 0

    width = max(len(r["id"]) for r in rows)
    print(f"OK: ranking by valid.{primary} ({mode}):")
    for i, run_id in enumerate(order, start=1):
        row = by_id[run_id]
        mark = "→" if run_id == chosen["id"] else " "
        state = "" if row.get("eligible") else \
            "   ineligible: " + ", ".join(row.get("ineligible_because") or [])
        value = "·" if row.get("primary") is None else f"{float(row['primary']):.5g}"
        params = row.get("params") or 0
        print(f"  {mark} {i:>2}. {run_id:<{width}}  {value:>9}  {params:>7,} params  "
              f"{row.get('stopped_by') or '·'}{state}")
    if a.override:
        print(f"WARN: {chosen['id']} was selected by override, not by the rule "
              f"({selection['override']['source']}): {a.reason}")
        if rule_pick:
            print(f"WARN: the rule would have picked {rule_pick['id']}")
        if not chosen.get("eligible"):
            print(f"WARN: {chosen['id']} is ineligible by the rule "
                  f"({', '.join(chosen.get('ineligible_because') or [])}); the layer shows the "
                  "override beside the rule and the apply script accepts it only because the "
                  "override is recorded")
    else:
        print(f"OK: selected {chosen['id']} by the ladder/selection rule — valid {primary} "
              f"{chosen['primary']:.5g}, margin over the baseline "
              f"{'·' if margin is None else f'{margin:+.5g}'}")
        if margin is not None and margin <= 0:
            print("WARN: the selected run does not beat the baseline; the campaign bought "
                  "nothing on this metric, and the summary should say so")
    print("OK: experiments.json updated with the ranking, the eligibility and the selection — "
          "the reading is yours to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
