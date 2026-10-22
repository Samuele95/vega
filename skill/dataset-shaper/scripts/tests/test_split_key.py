#!/usr/bin/env python3
"""
test_split_key.py — a group split's key never reaches the shaped parts as a feature (F33;
`shape-contract.md` §1 and §3, `step-catalog.md` phase 3).

    python3 test_split_key.py [<work-dir>] [-v]

A group split puts each group whole into one part, so every valid and test row carries a group
the fit never saw: encoded, the key is information in train alone and the encoder's reserved
unseen id everywhere else. The executor drops it at the split (unless it is the graph key of a
level-graph `graph_edges` step, which keeps `role: key`), the reproduction script and the
transform-only module drop it the same way, and the verifier holds the published output to it.
`run_negatives_leakage.py` (SL13-SL13e) seeds the defects; this is the other half, through the
real scripts.

  dropped     a numeric `tile` (12 groups) split by group: no part, no output_schema entry, no
              scale fitted on it (a `scale` over the numeric columns does not see it), the
              split's manifest record and lineage name it, verify / --check / the twin on
              every part pass, the module requires no `tile` and a served row loses it.
  graph key   the fixture graph-level recipe: `molecule_id` stays as the graph key, role key.

Exit 0 when every check holds.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[2]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load(HERE / "test_leakage_guards.py", "leakage_guards_for_split_key")
check, step, recipe, through, read_part = G.check, G.step, G.recipe, G.through, G.read_part
ARGS = G.ARGS
GROUP = {"kind": "group", "by": "tile", "test": 0.25, "valid": 0.15}


def tiled(work):
    """The leakage table with a numeric `tile`: the 60 places in 12 tiles of five."""
    csv = work / "table.csv"
    raw = G.table(csv)
    raw["tile"] = [int(i % 60) // 5 for i in range(len(raw))]
    raw.to_csv(csv, index=False)
    return raw, csv


def dropped(raw, csv, work):
    rec = recipe(csv, work / "dropped/shaped",
                 [step("S04", "encode", ["code_txt"], {"strategy": "ordinal"}),
                  step("S05", "scale", [], {"kind": "standard"})], split=False,
                 tail_=[])
    rec["steps"].insert(2, step("S03", "split", [], GROUP))
    got = through("dropped", rec, work)
    if got is None:
        return
    out, man, fitted, mod = got
    parts = {p: read_part(out, p) for p in ("train", "valid", "test")}
    check("dropped: 'tile' is in no shaped part",
          all("tile" not in d.columns for d in parts.values()),
          {p: list(d.columns) for p, d in parts.items()})
    check("dropped: output_schema has no 'tile'", "tile" not in man["output_schema"],
          sorted(man["output_schema"]))
    split = next(s for s in man["steps"] if s["op"] == "split")
    key = split["parameters_fitted"].get("key") or {}
    check("dropped: the split's manifest record names the key it dropped",
          key.get("column") == "tile" and key.get("fate") == "dropped"
          and "tile" in split["columns_removed"], key)
    scaled = next(s for s in man["steps"] if s["id"] == "S05")["parameters_fitted"]["columns"]
    check("dropped: a scale over the numeric columns never saw 'tile'", "tile" not in scaled,
          scaled)
    lineage = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    check("dropped: lineage records 'tile' removed by the split",
          (lineage["removed"].get("tile") or {}).get("op") == "split",
          lineage["removed"].get("tile"))
    groups = {p: set(raw["tile"].iloc[fitted["__split_raw__"][p]]) for p in parts}
    check("dropped: the groups stay whole, and valid and test hold tiles train never saw",
          not (groups["train"] & groups["test"]) and not (groups["train"] & groups["valid"])
          and groups["test"] and groups["valid"], groups)
    check("dropped: the module requires no 'tile'", "tile" not in mod.REQUIRED_COLUMNS,
          mod.REQUIRED_COLUMNS)
    served = mod.transform(raw.iloc[fitted["__split_raw__"]["test"][:3]])
    check("dropped: a served row loses 'tile' as the shaped parts did",
          "tile" not in served.columns and
          list(served.columns) == [c for c in parts["test"].columns if c != "busy"],
          list(served.columns))
    text = (out / "preprocess.py").read_text(encoding="utf-8")
    check("dropped: the module says why the key leaves", "group key leaves here" in text)


def graph_key(work):
    r = json.loads((SKILLS / "dataset-shaper/fixtures/graph-level.recipe.json")
                   .read_text(encoding="utf-8"))
    r["input"]["path"] = str(SKILLS / r["input"]["path"])
    r["outputs"]["dir"] = str(work / "graph-key/shaped")
    got = through("graph-key", r, work)
    if got is None:
        return
    out, man, fitted, _ = got
    split = next(s for s in man["steps"] if s["op"] == "split")
    check("graph key: molecule_id stays, role key",
          (man["output_schema"].get("molecule_id") or {}).get("role") == "key",
          man["output_schema"].get("molecule_id"))
    check("graph key: the split records it as the graph key",
          (split["parameters_fitted"].get("key") or {}).get("fate") == "graph key",
          split["parameters_fitted"].get("key"))


def main():
    work = Path(ARGS[0]) if ARGS else Path(tempfile.mkdtemp(prefix="shaper-split-key-"))
    work.mkdir(parents=True, exist_ok=True)
    raw, csv = tiled(work)
    for name, case in (("dropped", lambda: dropped(raw, csv, work)),
                       ("graph-key", lambda: graph_key(work))):
        print(name)
        try:
            case()
        except Exception as ex:                       # a case that crashes is a failed case
            check(f"{name}: runs to the end", False, f"{type(ex).__name__}: {ex}")
    failed = [n for n, good in G.results if not good]
    print(f"\n{len(G.results) - len(failed)} passed, {len(failed)} failed ({work})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
