#!/usr/bin/env python3
"""write_authoring.py <campaign dir> <run id> <out json> — the acceptance's judgement, composed
from the records of one evaluation.

The acceptance used to ship a hand-written `evaluation-layer.authoring.json` whose numbers were
one particular run's. Every change to the templates (F-1: shuffling, resident tensors) moves
those numbers, and the apply script — rightly — refuses a reading whose digits the records do not
hold. So the acceptance's judgement is now *composed from the records*, in the fixed form of
eval-contract §5, with every number read at the record's own rounding: the five readings say what
this run's `metrics.json`, `slices.json`, `robustness.json` and `threshold.json` say, and the
sentences bend to the facts (earned or not, an interval that covers zero or not) instead of
asserting a fact that was true of another run. The prose register is the contract's; it is a
fixture's judgement, not a template for the skill's — the skill writes its own, from the same
records, with the same constraint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _num(x) -> str:
    """A number as the record holds it (no re-rounding: the apply script matches digits)."""
    return json.dumps(x)


def _ci(rec) -> str:
    lo, hi = rec["ci"]
    return f"{_num(rec['value'])} [{_num(lo)}, {_num(hi)}]"


def _covers_zero(ci) -> bool:
    lo, hi = ci
    return lo <= 0.0 <= hi


def _cell(rec) -> str:
    """A metric cell: its value with its interval, or its value alone when it has none."""
    return _ci(rec) if rec.get("ci") else _num(rec.get("value"))


def verdict(m, primary, n) -> str:
    net = m["models"]["network"][primary]
    v = m["verdict"]
    strongest = v.get("strongest_baseline")
    if v.get("earned") is None or strongest not in (m.get("differences") or {}):
        return (f"The network scores {primary} {_cell(net)} on the {n} held-out rows. No paired "
                "verdict is read on this part: the record says the comparison is not assessable, "
                "so every model's metrics are reported with their intervals and none is ranked "
                "against another, and the network is neither credited nor faulted for its cost.")
    diff = m["differences"][strongest]
    triv = m["differences"].get("trivial")
    s = (f"The network scores {primary} {_ci(net)} on the {n} held-out rows"
         + (f"; against the trivial floor the paired difference is {_ci(triv)}. " if triv else ". "))
    s += (f"The strongest fitted baseline is {strongest}, at {_num(m['models'][strongest][primary]['value'])}, "
          f"and the paired difference network minus {strongest} is {_ci(diff)} — ")
    if v["earned"]:
        s += ("an interval that excludes zero on the right side, so this part separates the network "
              "from its strongest contender: the network has earned its cost. ")
    elif _covers_zero(diff["ci"]):
        s += ("an interval that covers zero, so this part cannot separate the network from a "
              f"{strongest} model on the same shaped features: the network has NOT earned its cost, "
              f"and what the user gets instead is the {strongest} baseline, which is cheaper and "
              "here indistinguishable. ")
    else:
        s += ("an interval that excludes zero on the wrong side: this part does not merely fail to "
              f"separate them, it says {strongest} is better; the network has NOT earned its cost, and "
              f"what the user gets instead is a recommendation — ship {strongest}. ")
    if strongest != "linear" and "linear" in m["differences"]:
        lin = m["differences"]["linear"]
        s += (f"Against linear the difference is {_ci(lin)}, and the interval "
              + ("covers zero, so that comparison is undecided." if _covers_zero(lin["ci"])
                 else "does not cover zero."))
    elif "tree" in m["differences"]:
        tr = m["differences"]["tree"]
        s += (f"Against tree the difference is {_ci(tr)}, and the interval "
              + ("covers zero, so that comparison is undecided." if _covers_zero(tr["ci"])
                 else "does not cover zero."))
    return s


def calibration(m, thr) -> str:
    net = m["models"]["network"]
    parts = []
    if "ece" in net:
        parts.append(f"ECE is {_ci(net['ece'])}")
    if "brier" in net:
        parts.append(f"the Brier score is {_ci(net['brier'])}")
    head = " and ".join(parts) if parts else "No calibration metric was recorded"
    s = f"{head}."
    if "brier" in net:
        s += (" On a part where the positive class is a minority, a low Brier score is largely "
              "the cost of predicting near zero everywhere, so the reliability curve rather than the "
              "summary number carries the information.")
    if thr.get("applies"):
        s += (f" The decision threshold {_num(thr['value'])} was chosen on the valid part "
              f"({thr['source']}) and every threshold-dependent metric is read at it.")
    else:
        s += " No decision threshold applies here, so no metric of this evaluation is read at one."
    return s


def slices(sl, primary) -> str:
    rows = [r for r in sl["slices"] if not r.get("degenerate")]
    named = (sl.get("worst") or {}).get("name")
    worst = next((r for r in rows if r["name"] == named), None)
    if worst is None:
        if not sl["slices"]:
            return ("The chain named no slice this evaluation could read — no nominal input, no "
                    "segment, no drifting column and no time order — so no part of the test rows "
                    "is singled out and no worst slice is reported.")
        return ("Every slice the chain named holds one class only, so a ranking metric has nothing "
                "to rank on any of them and no worst slice is reported.")
    w = worst["metrics"][primary]
    s = (f"The worst slice on {primary} is {worst['name']} at {_num(w['value'])}, "
         f"with n = {worst['n']}")
    twin = next((r for r in rows if r["name"].split("<=")[0].split(">")[0] ==
                 worst["name"].split("<=")[0].split(">")[0] and r is not worst), None)
    if twin:
        t = twin["metrics"][primary]
        s += (f" — against {_num(t['value'])} on the other side of the same split, "
              f"{twin['name']} at n = {twin['n']}. A gap of that size is the network working on one "
              "half of the data and not the other, and it is the finding of this evaluation that "
              "most deserves a follow-up upstream.")
    else:
        s += "."
    return s


def robustness(rb) -> str:
    checks = rb["checks"]
    ran = [c for c in checks if c.get("status") != "skipped"]
    skipped = [c for c in checks if c.get("status") == "skipped"]
    s = ""
    # a perturbation reads the whole test part shifted (its `delta` against the unshifted
    # primary); a slice side reads n rows of it — two kinds of entry, said apart (F39: the
    # perturbations ran for the first time once the analysis was read at its real path)
    perturbed = [c for c in ran if "delta" in c]
    ran = [c for c in ran if "delta" not in c]
    for c in perturbed:
        moved = not _covers_zero(c["delta"]["ci"])
        s += (f"Shifting {c['check'].split(':', 1)[-1]} by {c.get('shift', 'one train sd')} scores "
              f"{_ci(c['primary'])}, a change of {_ci(c['delta'])}: "
              + ("the network leans on that column measurably. " if moved else
                 "the interval covers zero, so the network does not measurably depend on it. "))
    if ran:
        s += "; ".join(f"{c['check']} scores {_ci(c['primary'])} (n = {c['n']})" for c in ran) + ". "
        if len(ran) >= 2:
            lo = min(ran, key=lambda c: c["primary"]["value"])
            hi = max(ran, key=lambda c: c["primary"]["value"])
            overlap = lo["primary"]["ci"][1] >= hi["primary"]["ci"][0]
            s += ("The intervals overlap, so the difference between these slices is not a robustness "
                  "finding on its own. " if overlap else
                  "The intervals do not overlap: performance is conditional on the slicing column "
                  "rather than merely varying with it. ")
    for c in skipped:
        s += f"The {c['check']} check was skipped: {c['why']}. "
    return s.strip()


def limits(m, primary, n, seed) -> str:
    net = m["models"]["network"][primary]
    lo, hi = net["ci"]
    v = m["verdict"]
    s = (f"One test part, one seed ({seed}), {n} rows. The interval on the primary metric runs "
         f"from {_num(lo)} to {_num(hi)}, and ")
    if v.get("strongest_baseline") and v.get("diff") is not None and v.get("ci"):
        s += (f"every comparison above inherits that width: the difference against "
              f"{v['strongest_baseline']}, {_num(v['diff'])}, is read with an interval of "
              f"[{_num(v['ci'][0])}, {_num(v['ci'][1])}], and ")
    return s + (f"{n} rows bound what can be claimed. The lens baseline, when it re-fits the same "
                "algorithm as tree, is a check that two stages agree rather than a fifth opinion.")


def main(argv):
    camp, run, out = Path(argv[1]), argv[2], Path(argv[3])
    ev = camp / "eval" / run
    m = json.loads((ev / "metrics.json").read_text())
    thr = json.loads((ev / "threshold.json").read_text())
    sl = json.loads((ev / "slices.json").read_text())
    rb = json.loads((ev / "robustness.json").read_text())
    primary, n, seed = m["primary"], m["n_test"], m.get("boot_seed", 7)
    doc = {
        "from_eval": f"eval/{run}",
        "readings": {
            "verdict": verdict(m, primary, n),
            "calibration": calibration(m, thr),
            "slices": slices(sl, primary),
            "robustness": robustness(rb),
            "limits": limits(m, primary, n, seed),
        },
        "forks": [],
    }
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OK: judgement composed from {ev} — verdict {'earned' if m['verdict']['earned'] else 'not earned'} "
          f"vs {m['verdict']['strongest_baseline']}, pointed at {doc['from_eval']}")


if __name__ == "__main__":
    main(sys.argv)
