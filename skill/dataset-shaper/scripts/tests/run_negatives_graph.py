#!/usr/bin/env python3
"""
run_negatives_graph.py — every graph gate of /dataset-shaper must refuse what it says it refuses.

    python3 run_negatives_graph.py [<work-dir>] [-v] [--json <path>]

`test_graph_edges.py` proves the graph step builds, verifies, reproduces and serves. That is
the easy half. This script seeds one defect at a time and asserts that the right script refuses
it with a message naming what is wrong — the shaper graph gates SG1–SG7 of the test design
(D6 §12.3), plus the ones the implementation added. SG8 (a tampered reference-nodes file) is
a pack gate since the build plan's R-A4 (negative N-S9 of /vega-pack): the shaper ships no
reference file.

The design seeds the lanes; this package runs before the lanes exist, so each case seeds the
shaper fixture that stands for the lane: L19 (key_pair, level node) is `graph-keypair`, L21
(knn) is `graph-knn`, L22 (key_pair with a graph id, level graph) is `graph-level`.

| id   | seed                                                           | expect                                  |
|------|----------------------------------------------------------------|-----------------------------------------|
| SG1  | graph_edges with `source: "shaper:default"`                    | shape.py exit 2: graph_edges, analysis:, user: |
| SG1b | graph_edges with a `geometry:` source                          | exit 2: an analysis: or user: source    |
| SG2  | the step moved before `split`                                  | exit 2: after, split                    |
| SG3  | a recipe with no `split` and a graph_edges step                | exit 2: requires a split                |
| SG3b | a graph_id recipe split by stratification, not by group        | exit 2: group split                     |
| SG4  | key_pair naming a missing column                               | exit 3, naming the column               |
| SG4b | a post-split custom step renumbers the parts' rows             | exit 3: raw positions the split recorded |
| SG4c | a key_pair cap (max_clique) at level node                      | exit 2: needs a graph_id                |
| SG5  | knn edges part with one test -> test edge added                | verify exit 3: two non-train nodes, cross_part_edges |
| SG5b | an edges part edited without its manifest digest               | verify exit 3: digest                   |
| SG6  | `symmetric: true` with one reverse edge deleted                | verify exit 3: reverse                  |
| SG6b | a test -> reference edge removed from the part                 | twin exit 3: edges_for diverges         |
| SG7  | graph-level edges with one edge across two molecule_id values  | verify exit 3: graph_id                 |
| SG7b | a knn edge re-pointed at a farther train node                  | verify exit 3: nearest                  |

In SG5–SG7, SG6b and SG7b the manifest's file digest and row count are rewritten after the
edit, so the file digest is not what catches the defect: the gate under test is. The recipe
checker's other static rules (the parameters per method, a second graph step, a spatial step
without its geometry) are then put to `check_recipe` directly, one mutation each.

Exit 0 when every seeded defect was caught by the gate that should catch it. `--json` writes the
cases as `vega/negatives-result@1` (D6 §12): each case's id is the gate id its name starts with
(SG1…SG7b), null for the static-rule cases; the coverage report reads it (D6 §12.8).
"""
from __future__ import annotations

import copy
import hashlib
import re
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from pyarrow.fs import LocalFileSystem      # A40: never a Python file object

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILL = SCRIPTS.parent
ROOT = SKILL.parent
FIX = SKILL / "fixtures"
SHAPE = SCRIPTS / "shape.py"
VERIFY = SCRIPTS / "verify_shape.py"
VERBOSE = "-v" in sys.argv
JSON_OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv[1:-1] else None
ARGS = [a for a in sys.argv[1:] if not a.startswith("-") and a != JSON_OUT]

results = []


def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          cwd=str(ROOT), timeout=900)
    return proc.returncode, proc.stdout + proc.stderr


def expect(name, rc, out, *, code, contains=()):
    problems = [f"exit {rc}, expected {code}"] if rc != code else []
    problems += [f"message does not mention {n!r}" for n in contains
                 if n.lower() not in out.lower()]
    results.append((name, not problems, "; ".join(problems)))
    print(f"  {'ok  ' if not problems else 'FAIL'}  {name}"
          + (f"  — {'; '.join(problems)}" if problems else ""))
    if VERBOSE or problems:
        lines = [ln for ln in out.strip().splitlines() if ln.startswith(("ERROR", "WARN"))]
        print("        " + "\n        ".join((lines or out.strip().splitlines())[-6:]))


def recipe(name):
    return json.loads((FIX / f"{name}.recipe.json").read_text(encoding="utf-8"))


def graph_step(r):
    return next(s for s in r["steps"] if s["op"] == "graph_edges")


def split_step(r):
    return next(s for s in r["steps"] if s["op"] == "split")


def write(work, case, r):
    d = work / case
    d.mkdir(parents=True, exist_ok=True)
    p = d / "recipe.json"
    p.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return p, d / "shaped"


# --------------------------------------------------------------------------- recipe gates
def move_before_split(r):
    g = graph_step(r)
    r["steps"].remove(g)
    r["steps"].insert(r["steps"].index(split_step(r)), g)


def renumber_after_split(r):
    r["steps"].insert(r["steps"].index(graph_step(r)), {
        "id": "S6b", "op": "custom", "columns": [],
        "params": {"phase": 6, "code": "def step(df, ctx):\n    return df.reset_index(drop=True)\n"},
        "source": "user:renumber the rows", "rationale": "a seeded defect",
        "alternatives": [], "consequences": {"rows": "none", "columns": "none",
                                             "downstream": "none"}, "reversible": False})


# (id, what, fixture, mutation, exit code, the words the refusal must carry)
RECIPE_GATES = [
    ("SG1", "graph_edges with a shaper:default source", "graph-keypair",
     lambda r: graph_step(r).update(source="shaper:default"), 2, ("graph_edges", "analysis:",
                                                                   "user:")),
    ("SG1b", "graph_edges with a geometry: source", "graph-keypair",
     lambda r: graph_step(r).update(source="geometry:typing/zone"), 2,
     ("an analysis: or user: source",)),
    ("SG2", "graph_edges moved before the split", "graph-keypair", move_before_split, 2,
     ("after", "split")),
    ("SG3", "graph_edges in a recipe without a split", "graph-keypair",
     lambda r: r["steps"].remove(split_step(r)), 2, ("requires a split",)),
    ("SG3b", "a graph_id recipe without a group split on it", "graph-level",
     lambda r: split_step(r).update(params={"kind": "stratified", "by": "soluble",
                                            "test": 0.2, "valid": 0.1}), 2,
     ("group split by 'molecule_id'",)),
    ("SG4", "key_pair naming a missing column", "graph-keypair",
     lambda r: graph_step(r)["params"].update(columns=["bus_key", "feeder_id"]), 3,
     ("feeder_id", "not a column of the input")),
    ("SG4b", "a step after the split renumbers the parts' rows", "graph-keypair",
     renumber_after_split, 3, ("graph_edges", "raw positions the split recorded")),
    ("SG4c", "a key_pair cap at level node (served rows carry no raw position)",
     "graph-keypair", lambda r: graph_step(r).update(params={
         "method": "key_pair", "columns": ["zone"], "max_clique": 4}), 2,
     ("needs a graph_id", "raw position")),
]


def recipe_gates(work):
    for case, what, fixture, mutate, code, contains in RECIPE_GATES:
        r = recipe(fixture)
        mutate(r)
        p, out = write(work, case, r)
        rc, log = _run(SHAPE, "--recipe", p, "--out-dir", out)
        expect(f"{case} {what}", rc, log, code=code, contains=contains)


# (what, fixture, mutation of the graph step's params or the recipe, the words)
STATIC_RULES = [
    ("an unknown method", "graph-keypair", {"method": "bfs"}, "method must be one of"),
    ("weight distance on key_pair", "graph-keypair", {"weight": "distance"}, "needs a metric"),
    ("max_clique on a two-column key_pair", "graph-level", {"max_clique": 4},
     "max_clique caps a one-column clique"),
    ("max_degree on a key_pair at level node", "graph-keypair", {"max_degree": 2},
     "needs a graph_id"),
    ("a graph_id without a group split on it", "graph-keypair", {"graph_id": "zone"},
     "group split by 'zone'"),
    ("spatial without parse_geometry / spatial_reproject", "graph-keypair",
     {"method": "spatial", "k": 3}, "spatial_reproject"),
    ("spatial with both k and radius_m", "graph-keypair",
     {"method": "spatial", "k": 3, "radius_m": 100}, "exactly one of k"),
    ("knn k 0", "graph-keypair", {"method": "knn", "k": 0}, "positive integer"),
    ("a second graph_edges step", "graph-keypair",
     lambda r: r["steps"].append(copy.deepcopy(dict(graph_step(r), id="S8"))),
     "at most one graph_edges"),
]


def static_rules():
    spec = importlib.util.spec_from_file_location("shaper_shape", str(SHAPE))
    shape = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shape)
    errs = " | ".join(shape.check_recipe(recipe("graph-keypair"), {}, lambda m: None))
    expect("rules: the fixture recipe passes the checker", 0 if not errs else 2, errs, code=0)
    for what, fixture, change, needle in STATIC_RULES:
        r = recipe(fixture)
        change(r) if callable(change) else graph_step(r)["params"].update(change)
        errs = " | ".join(shape.check_recipe(r, {}, lambda m: None))
        expect(f"rules: {what} is refused", 2 if errs else 0, errs, code=2, contains=(needle,))


# --------------------------------------------------------------------------- edges-part gates
def shape_base(work, name):
    p, out = write(work / "base", name, recipe(name))
    rc, log = _run(SHAPE, "--recipe", p, "--out-dir", out)
    if rc != 0:
        raise SystemExit(f"ERROR: the {name} fixture does not shape (exit {rc}); the negatives "
                         f"need a clean base:\n{log[-1500:]}")
    return p, out


def tampered(work, case, base, edit=None):
    """A copy of a shaped directory whose edges part `edit` changed; the manifest's file digest
    and row count follow the edit, so only the gate under test can catch it."""
    d = work / case / "shaped"
    if d.parent.exists():
        shutil.rmtree(d.parent)
    shutil.copytree(base, d)
    if edit is None:
        return d
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    meta = man["outputs"]["files"]["edges"]
    path = d / Path(meta["path"]).name
    e = edit(pd.read_parquet(path, filesystem=LocalFileSystem()), d)
    e.to_parquet(path, index=False)
    meta["digest"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    meta["rows"] = int(len(e))
    (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return d


def split_of(d):
    return json.loads((d / "fitted.json").read_text(encoding="utf-8"))["__split_raw__"]


def add_test_test(e, d):
    a, b = split_of(d)["test"][:2]
    extra = pd.DataFrame({"src_pos": [a], "dst_pos": [b], "src_part": ["test"],
                          "dst_part": ["test"], "weight": [0.5]})
    return pd.concat([e, extra.astype(e.dtypes.to_dict())], ignore_index=True)


def drop_one_reverse(e, d):
    return e.drop(index=e.index[len(e) // 2]).reset_index(drop=True)


def across_graphs(e, d):
    raw = pd.read_csv(ROOT / recipe("graph-level")["input"]["path"], low_memory=False)
    train = split_of(d)["train"]
    a = train[0]
    b = next(i for i in train if raw["molecule_id"][i] != raw["molecule_id"][a])
    extra = pd.DataFrame({"src_pos": [a, b], "dst_pos": [b, a], "src_part": ["train"] * 2,
                          "dst_part": ["train"] * 2, "weight": [float("nan")] * 2})
    return pd.concat([e, extra.astype(e.dtypes.to_dict())], ignore_index=True)


def repoint(e, d):
    e = e.copy()
    train = set(split_of(d)["train"])
    row = e.index[(e["src_part"] == "test")][0]
    have = set(e.loc[e["src_pos"] == e.loc[row, "src_pos"], "dst_pos"])
    e.loc[row, "dst_pos"] = next(i for i in sorted(train, reverse=True) if i not in have)
    return e.sort_values(["src_pos", "dst_pos"], kind="stable").reset_index(drop=True)


def double_first_weight(d):
    """An edit behind the manifest's back: the digest is left as the executor wrote it."""
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    path = d / Path(man["outputs"]["files"]["edges"]["path"]).name
    e = pd.read_parquet(path, filesystem=LocalFileSystem())
    e.loc[0, "weight"] = e.loc[0, "weight"] * 2
    e.to_parquet(path, index=False)


def twin_gate(work, kn_recipe, kn):
    def drop_attachment(e, d):
        hit = e.index[(e["src_part"] == "test") & (e["dst_part"] == "train")][0]
        return e.drop(index=[hit]).reset_index(drop=True)
    d = tampered(work, "SG6b", kn, drop_attachment)
    module = d / "preprocess.py"
    rc, log = _run(SHAPE, "--transform-only", "--recipe", kn_recipe, "--fitted",
                   d / "fitted.json", "--emit", module)
    if rc == 0:
        rc, log = _run(VERIFY, "--transform-only", "--recipe", kn_recipe, "--out-dir", d,
                       "--module", module, "--part", "test")
        expect("SG6b a test -> reference edge missing from the part (twin)", rc, log, code=3,
               contains=("edges_for diverges", "1 extra"))
    else:
        expect("SG6b a test -> reference edge missing from the part (twin)", rc, log, code=0)


def edge_gates(work):
    bases = {n: shape_base(work, n) for n in ("graph-keypair", "graph-knn", "graph-level")}
    cases = [("SG5", "a test -> test edge in a knn edges part", "graph-knn", add_test_test,
              ("two non-train nodes", "cross_part_edges")),
             ("SG6", "symmetric edges with one reverse deleted", "graph-keypair",
              drop_one_reverse, ("reverse",)),
             ("SG7", "an edge across two molecule_id values", "graph-level", across_graphs,
              ("graph_id", "molecule_id")),
             ("SG7b", "a knn edge re-pointed at a train node that is not among the nearest",
              "graph-knn", repoint, ("not among their source's", "nearest"))]
    for case, what, name, edit, contains in cases:
        d = tampered(work, case, bases[name][1], edit)
        rc, log = _run(VERIFY, "--recipe", bases[name][0], "--out-dir", d)
        expect(f"{case} {what}", rc, log, code=3, contains=contains)
    d = tampered(work, "SG5b", bases["graph-knn"][1])
    double_first_weight(d)
    rc, log = _run(VERIFY, "--recipe", bases["graph-knn"][0], "--out-dir", d)
    expect("SG5b an edges part edited behind its manifest digest", rc, log, code=3,
           contains=("digest", "changed after the executor wrote it"))
    twin_gate(work, *bases["graph-knn"])


def main():
    work = Path(ARGS[0]).resolve() if ARGS else Path(tempfile.mkdtemp(prefix="graph-neg-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"run_negatives_graph: work in {work}")
    print("\n[recipe gates] shape.py")
    recipe_gates(work)
    print("\n[edges-part gates] verify_shape.py")
    edge_gates(work)
    print("\n[static rules] check_recipe")
    static_rules()
    failed = [n for n, good, _ in results if not good]
    print(f"\n{len(results) - len(failed)} of {len(results)} seeded defects caught")
    if JSON_OUT:
        write_json(Path(JSON_OUT), work)
    return 1 if failed else 0


def write_json(path, work):
    """`vega/negatives-result@1`: the cases in the order they ran, each with its gate id."""
    cases = [{"id": (m.group(1) if (m := re.match(r"(SG\d+[a-z]?)\s", name)) else None), "name": name,
              "ok": good, "detail": detail} for name, good, detail in results]
    passed = sum(1 for c in cases if c["ok"])
    doc = {"schema": "vega/negatives-result@1", "script": str(Path(__file__).resolve().relative_to(ROOT)),
           "work": str(work), "cases": cases, "passed": passed, "failed": len(cases) - passed}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} (vega/negatives-result@1)")


if __name__ == "__main__":
    sys.exit(main())
