#!/usr/bin/env python3
"""
run_negatives_leakage.py — every leakage guard of /dataset-shaper must refuse what it says it
refuses (A46 track S: the rules `shape-contract.md` §1-§3 state and the phase order cannot).

    python3 run_negatives_leakage.py [<work-dir>] [-v]

`test_leakage_guards.py` proves what the guards let through. This script seeds one defect at a
time and asserts that the right gate refuses it with a message naming what is wrong. Each rule
is held in three places, and each place gets its own seed: the recipe checker (`shape.py`,
exit 2, before any data moves), the executor (`shape.py`, exit 3, where the step runs — reached
here by calling `execute()` with the checker bypassed, or by a defect only the data shows), and
the verifier (`verify_shape.py`, exit 3, on a recipe or manifest that was published with the
defect).

| id    | seed                                                              | expect                               |
|-------|-------------------------------------------------------------------|--------------------------------------|
| SL1   | impute group-median grouped by the label (a user: source)         | shape.py exit 2: grouped by the label |
| SL1b  | the same, `by` given as one column name                           | shape.py exit 2: grouped by the label |
| SL2   | select_features kind=importance in a recipe that splits           | shape.py exit 2: importance, test    |
| SL2b  | select_features with no kind (importance by default), after split | shape.py exit 2: importance          |
| SL3   | a custom step before the split declaring fitted: true             | shape.py exit 2: fitted: true        |
| SL4   | a pre-split custom step centring a column on the frame's mean     | shape.py exit 3: row-local           |
| SL4b  | a pre-split custom step ranking a column                          | shape.py exit 3: row-local           |
| SL4c  | a pre-split custom step dropping rows that repeat a code          | shape.py exit 3: row-local           |
| SL4d  | a pre-split custom step renumbering the rows (positions + 1000)   | shape.py exit 3: index               |
| SL4e  | a pre-split custom step filtering, then counting rows from 0      | shape.py exit 3: row-local           |
| SL5   | spatial_features asking for an area of point geometry             | shape.py exit 3: polygon             |
| SL5b  | spatial_features asking for a feature it does not know            | shape.py exit 3: not a feature       |
| SL5c  | spatial_features bearing_to without the point to take it to       | shape.py exit 3: params.to           |
| SL6   | execute() with the checker bypassed: impute grouped by the label  | ValueError: grouped by the label     |
| SL6b  | execute() with the checker bypassed: importance after the split   | ValueError: importance               |
| SL7   | a published recipe whose impute is grouped by the label           | verify exit 3: leakage, label        |
| SL7b  | a published recipe carrying select_features importance            | verify exit 3: leakage, importance   |
| SL8   | a box-cox step's manifest fit_on removed                          | verify exit 3: fit_on                |
| SL8b  | a yeo-johnson step's manifest fit_on set to "all"                 | verify exit 3: fit_on                |
| SL8c  | an impute's manifest fit_on set to "all"                          | verify exit 3: fit_on                |
| SL9   | a pre-split custom step's row-locality probe record removed       | verify exit 3: probe                 |
| SL10  | a neighbour index fitted on every row (n_fit = the input's rows)  | verify exit 3: neighbour index       |
| SL10b | a neighbour index recording fit_on "all"                          | verify exit 3: neighbour index       |
| SL11  | a stratified split by a continuous column (sparse strata)         | shape.py exit 3: strata              |
| SL12  | a lag grouped by the label (a user: source; A47(5))               | shape.py exit 2: step, label         |
| SL12b | the same, `group_by` given as a list                              | shape.py exit 2: grouped by the label |
| SL12c | execute() with the checker bypassed: a lag grouped by the label   | ValueError: step, label              |
| SL12d | a published recipe whose lag is grouped by the label              | verify exit 3: leakage, step, label  |
| SL13  | an encode of the group split's key after the split (F33)          | shape.py exit 2: step, split's key   |
| SL13b | a lag grouped by the group split's key                            | shape.py exit 2: step, split's key   |
| SL13c | an impute grouped by the group split's key                        | shape.py exit 2: step, split's key   |
| SL13d | execute() with the checker bypassed: an encode of the split's key | ValueError: split's key              |
| SL13e | a published output whose schema carries the split's key a feature | verify exit 3: split, split's key    |
| SL13f | a published recipe that encodes the split's key after the split   | verify exit 3: split, step, key      |

Exit 0 when every seeded defect was caught by the gate that should catch it.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SHAPE, VERIFY = SCRIPTS / "shape.py", SCRIPTS / "verify_shape.py"
VERBOSE = "-v" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
results = []


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load(HERE / "test_leakage_guards.py", "leakage_guards")     # the table and the recipes
step, recipe, geometry = G.step, G.recipe, G.geometry


def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          timeout=900)
    return proc.returncode, proc.stdout + proc.stderr


def expect(name, rc, out, *, code, contains=()):
    problems = [f"exit {rc}, expected {code}"] if rc != code else []
    problems += [f"message does not mention {n!r}" for n in contains
                 if n.lower() not in out.lower()]
    results.append((name, not problems))
    print(f"  {'ok  ' if not problems else 'FAIL'}  {name}"
          + (f"  — {'; '.join(problems)}" if problems else ""))
    if VERBOSE or problems:
        lines = [ln for ln in out.strip().splitlines() if ln.startswith(("ERROR", "WARN"))]
        print("        " + "\n        ".join((lines or out.strip().splitlines())[-6:]))


def write(work, case, r):
    d = work / case
    d.mkdir(parents=True, exist_ok=True)
    p = d / "recipe.json"
    r = copy.deepcopy(r)
    r["outputs"]["dir"] = str(d / "shaped")
    p.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return p, d / "shaped"


def shaped(work, case, r, *, code, contains):
    p, out = write(work, case, r)
    rc, log = _run(SHAPE, "--recipe", p, "--out-dir", out)
    expect(case, rc, log, code=code, contains=contains)


def custom(code):
    return step("S0c", "custom", [], {"phase": 1, "code": code})


IMPUTE_BY_LABEL = {"strategy": "group-median", "by": ["busy"]}
LAG_BY_LABEL = {"lags": [1], "group_by": "busy"}
CENTRE = "def step(df, ctx):\n    df = df.copy()\n    df['noise'] = df['noise'] - df['noise'].mean()\n    return df\n"
RANK = "def step(df, ctx):\n    df = df.copy()\n    df['flow_rank'] = df['flow'].rank()\n    return df\n"
# F33: a group split by `code_txt` (five groups), then the steps under test.
GROUP_SPLIT = {"kind": "group", "by": "code_txt", "test": 0.25, "valid": 0.15}
KEY = "group split's key"


def grouped(csv, out, after):
    return recipe(csv, out, [], split=False, tail_=[step("S03", "split", [], GROUP_SPLIT),
                                                    *after])


DEDUPE = "def step(df, ctx):\n    return df.drop_duplicates(subset=['code_txt'])\n"
RENUMBER = "def step(df, ctx):\n    return df.set_axis(df.index + 1000)\n"
RECOUNT = "def step(df, ctx):\n    return df[df['noise'] > 0].reset_index(drop=True)\n"
ROW_LOCAL = "def step(df, ctx):\n    df = df.copy()\n    df['code'] = df['code_txt'].str[1:].astype(int)\n    return df.drop(columns=['code_txt'])\n"


# --------------------------------------------------------------------------- recipe checker
def recipe_gates(work, csv):
    base = lambda middle, **kw: recipe(csv, work / "x", middle, **kw)      # noqa: E731
    shaped(work, "SL1 impute grouped by the label (user: source)",
           base([step("S04", "impute", ["noise"], IMPUTE_BY_LABEL)]),
           code=2, contains=("grouped by the label", "busy", "whatever the source"))
    shaped(work, "SL1b impute grouped by the label, by as one name",
           base([step("S04", "impute", ["noise"], dict(IMPUTE_BY_LABEL, by="busy"))]),
           code=2, contains=("grouped by the label",))
    shaped(work, "SL2 select_features kind=importance after the split",
           base([step("S04", "select_features", [], {"kind": "importance", "threshold": 0.0})]),
           code=2, contains=("importance", "test"))
    shaped(work, "SL2b select_features with the default kind after the split",
           base([step("S04", "select_features", [], {"threshold": 0.0})]),
           code=2, contains=("importance",))
    shaped(work, "SL12 a lag grouped by the label (user: source)",
           base([step("S04", "lag", ["noise"], LAG_BY_LABEL)]),
           code=2, contains=("lag S04 grouped by the label 'busy'", "whatever the source"))
    shaped(work, "SL12b a lag grouped by the label, group_by as a list",
           base([step("S04", "lag", ["noise"], dict(LAG_BY_LABEL, group_by=["busy"]))]),
           code=2, contains=("lag S04 grouped by the label",))
    shaped(work, "SL13 an encode of the group split's key after the split",
           grouped(csv, work / "x", [step("S04", "encode", ["code_txt"], {"strategy": "ordinal"})]),
           code=2, contains=("S04", KEY, "code_txt"))
    shaped(work, "SL13b a lag grouped by the group split's key",
           grouped(csv, work / "x", [step("S04", "lag", ["noise"], {"lags": [1],
                                                                     "group_by": "code_txt"})]),
           code=2, contains=("S04", KEY, "code_txt"))
    shaped(work, "SL13c an impute grouped by the group split's key",
           grouped(csv, work / "x", [step("S04", "impute", ["noise"],
                                          {"strategy": "group-median", "by": ["code_txt"]})]),
           code=2, contains=("S04", KEY, "code_txt"))
    shaped(work, "SL3 a pre-split custom step declaring fitted: true",
           base([], pre=[step("S0c", "custom", [], {"phase": 1, "code": ROW_LOCAL,
                                                     "fitted": True})]),
           code=2, contains=("fitted: true", "test part"))


# --------------------------------------------------------------------------- executor
def executor_gates(work, csv):
    base = lambda **kw: recipe(csv, work / "x", [], **kw)      # noqa: E731
    shaped(work, "SL4 a pre-split custom step centring on the frame's mean",
           base(pre=[custom(CENTRE)]), code=3, contains=("S0c", "row-local", "noise"))
    shaped(work, "SL4b a pre-split custom step ranking a column",
           base(pre=[custom(RANK)]), code=3, contains=("S0c", "row-local", "flow_rank"))
    shaped(work, "SL4c a pre-split custom step dropping rows that repeat a code",
           base(pre=[custom(DEDUPE)]), code=3, contains=("S0c", "row-local"))
    shaped(work, "SL4d a pre-split custom step renumbering the rows",
           base(pre=[custom(RENUMBER)]), code=3, contains=("S0c", "keep each row's index"))
    shaped(work, "SL4e a pre-split custom step filtering, then counting rows from 0",
           base(pre=[custom(RECOUNT)]), code=3, contains=("S0c", "row-local"))
    for case, feats, needle in (
            ("SL5 spatial_features asking for an area of point geometry", ["area"],
             ("area", "polygon")),
            ("SL5b spatial_features asking for a feature it does not know", ["elevation"],
             ("elevation", "not a feature")),
            ("SL5c spatial_features bearing_to without the point", ["bearing_to"],
             ("bearing_to", "params.to"))):
        shaped(work, case, recipe(csv, work / "x", [step(
            "S04", "spatial_features", [], {"features": ["n_neighbours_within", *feats]})],
            pre=[geometry()]), code=3, contains=needle)
    shaped(work, "SL11 a stratified split by a continuous column",
           recipe(csv, work / "x", [], split=False,
                  tail_=[step("S03", "split", [], dict(G.SPLIT, by="flow"))]),
           code=3, contains=("stratified", "flow", "strata hold fewer than"))
    sh = _load(SHAPE, "shape_under_test")
    import pandas as pd
    df = pd.read_csv(csv)
    for name, middle, needle in (
            ("SL6 execute(): impute grouped by the label",
             [step("S04", "impute", ["noise"], IMPUTE_BY_LABEL)], "grouped by the label"),
            ("SL6b execute(): importance after the split",
             [step("S04", "select_features", [], {"kind": "importance"})], "importance"),
            ("SL12c execute(): a lag grouped by the label",
             [step("S04", "lag", ["noise"], LAG_BY_LABEL)], "lag S04 grouped by the label 'busy'")):
        try:
            sh.execute(recipe(csv, work / "x", middle), {}, df.copy(), 3)
            expect(name, 0, "executed", code="ValueError", contains=(needle,))
        except ValueError as ex:
            expect(name, "ValueError", str(ex), code="ValueError", contains=(needle,))
    name = "SL13d execute(): an encode of the group split's key"
    try:
        sh.execute(grouped(csv, work / "x", [step("S04", "encode", ["code_txt"],
                                                  {"strategy": "ordinal"})]), {}, df.copy(), 3)
        expect(name, 0, "executed", code="ValueError", contains=(KEY,))
    except ValueError as ex:
        expect(name, "ValueError", str(ex), code="ValueError", contains=(KEY, "code_txt"))


# --------------------------------------------------------------------------- verifier
def published(work, csv):
    """A clean run every verifier seed starts from: a row-local pre-split custom step, the
    split, an impute, box-cox, yeo-johnson and the neighbour features."""
    middle = [step("S04", "impute", ["noise"], {"strategy": "median"}),
              step("S05", "transform", ["flow"], {"kind": "box-cox"}),
              step("S06", "transform", ["noise"], {"kind": "yeo-johnson"}),
              step("S07", "spatial_features", [], {"features": ["n_neighbours_within"],
                                                   "radius_m": 900})]
    p, out = write(work, "base", recipe(csv, work / "x", middle,
                                        pre=[geometry(), custom(ROW_LOCAL)]))
    rc, log = _run(SHAPE, "--recipe", p, "--out-dir", out)
    expect("base: the clean recipe shapes", rc, log, code=0)
    rc, log = _run(VERIFY, "--recipe", p, "--out-dir", out)
    expect("base: the clean output verifies", rc, log, code=0, contains=("OK: leakage",))
    return p, out


def tampered(work, case, base, *, recipe_edit=None, manifest_edit=None, contains=()):
    p0, out0 = base
    d = work / case
    if d.exists():
        shutil.rmtree(d)
    out = d / "shaped"
    shutil.copytree(out0, out)
    r = json.loads(p0.read_text(encoding="utf-8"))
    if recipe_edit:
        recipe_edit(r)
    p = d / "recipe.json"
    p.write_text(json.dumps(r, indent=1), encoding="utf-8")
    if manifest_edit:
        mp = out / "manifest.json"
        man = json.loads(mp.read_text(encoding="utf-8"))
        manifest_edit({s["id"]: s for s in man["steps"]}, man)
        mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    rc, log = _run(VERIFY, "--recipe", p, "--out-dir", out)
    expect(case, rc, log, code=3, contains=contains)


def _step(r, sid):
    return next(s for s in r["steps"] if s["id"] == sid)


def verifier_gates(work, csv, n_rows):
    base = published(work, csv)
    tampered(work, "SL7 a published impute grouped by the label", base,
             recipe_edit=lambda r: _step(r, "S04")["params"].update(IMPUTE_BY_LABEL),
             contains=("leakage", "grouped by the label"))
    tampered(work, "SL7b a published select_features importance", base,
             recipe_edit=lambda r: r["steps"].append(
                 step("S08", "select_features", [], {"kind": "importance"})),
             contains=("leakage", "importance"))
    tampered(work, "SL12d a published lag grouped by the label", base,
             recipe_edit=lambda r: r["steps"].append(step("S08", "lag", ["noise"], LAG_BY_LABEL)),
             contains=("leakage", "S08", "grouped by the label 'busy'"))
    tampered(work, "SL8 a box-cox step's fit_on removed", base,
             manifest_edit=lambda s, m: s["S05"].update(fit_on=None),
             contains=("leakage", "S05", "fit_on"))
    tampered(work, "SL8b a yeo-johnson step's fit_on set to all", base,
             manifest_edit=lambda s, m: s["S06"].update(fit_on="all"),
             contains=("leakage", "S06", "fit_on"))
    tampered(work, "SL8c an impute's fit_on set to all", base,
             manifest_edit=lambda s, m: s["S04"].update(fit_on="all"),
             contains=("leakage", "S04", "fit_on"))
    tampered(work, "SL9 a pre-split custom step's probe record removed", base,
             manifest_edit=lambda s, m: s["S0c"]["parameters_fitted"].pop("row_local_probe"),
             contains=("leakage", "S0c", "probe"))

    def all_rows(s, m):
        index = s["S07"]["parameters_fitted"]["index"]
        index["counts"][0] += n_rows - index["n_fit"]
        index["n_fit"] = n_rows
    tampered(work, "SL10 a neighbour index fitted on every row", base, manifest_edit=all_rows,
             contains=("leakage", "S07", "neighbour index"))
    tampered(work, "SL10b a neighbour index recording fit_on all", base,
             manifest_edit=lambda s, m: s["S07"]["parameters_fitted"]["index"].update(
                 fit_on="all"), contains=("leakage", "S07", "neighbour index"))


def split_key_gates(work, csv):
    """F33 on what was published: a clean group split (the key gone), then the key put back
    into the schema as a feature, and a recipe that encodes it after the split."""
    p, out = write(work, "base-group", grouped(csv, work / "x", [
        step("S04", "impute", ["noise"], {"strategy": "median"})]))
    rc, log = _run(SHAPE, "--recipe", p, "--out-dir", out)
    expect("base-group: the clean group split shapes", rc, log, code=0,
           contains=("leaves the features",))
    rc, log = _run(VERIFY, "--recipe", p, "--out-dir", out)
    expect("base-group: the clean output verifies", rc, log, code=0,
           contains=("is no feature (gone from every part)",))
    base = (p, out)
    tampered(work, "SL13e a published schema carrying the split's key as a feature", base,
             manifest_edit=lambda s, m: m["output_schema"].update(
                 code_txt={"dtype": "Int64", "role": "feature", "origin": "S04"}),
             contains=("split", KEY, "code_txt"))
    tampered(work, "SL13f a published recipe encoding the split's key after the split", base,
             recipe_edit=lambda r: r["steps"].append(
                 step("S05", "encode", ["code_txt"], {"strategy": "ordinal"})),
             contains=("split", "S05", KEY))


def main():
    work = Path(ARGS[0]).resolve() if ARGS else Path(tempfile.mkdtemp(prefix="leak-neg-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"run_negatives_leakage: work in {work}")
    csv = work / "table.csv"
    raw = G.table(csv)
    print("\n[recipe checker] shape.py, exit 2")
    recipe_gates(work, csv)
    print("\n[executor] shape.py exit 3, execute()")
    executor_gates(work, csv)
    print("\n[published output] verify_shape.py, exit 3")
    verifier_gates(work, csv, len(raw))
    split_key_gates(work, csv)
    failed = [n for n, good in results if not good]
    print(f"\n{len(results) - len(failed)} of {len(results)} seeded defects caught")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
