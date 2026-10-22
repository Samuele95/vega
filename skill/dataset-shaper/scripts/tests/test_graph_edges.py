#!/usr/bin/env python3
"""
test_graph_edges.py — the shaper's graph step (phase 7), its edges part, and the two entry
points the transform-only module serves it through.

    python3 test_graph_edges.py [<work-dir>] [-v]

Every fixture and variant goes through the real scripts as the chain runs them: `shape.py`,
`verify_shape.py` (the graph checks), `shape.py --check` (the edges part reproduces),
`--transform-only`, and the twin check on the test and the valid part, whose reference is
built as /vega-pack ships `vega/graph-reference@1` (every feature as shaped, in float32, and
the input columns `edges_for` reads as `raw:<column>`). On top of those exit codes it asserts
the build plan's properties (B05, R-A3): knn at level node joins no valid–test, test–test or
valid–valid pair; `cross_part_edges` and the other counts equal a recount; `edges_for` serves
exactly the part's query <-> reference pairs and weights, alone as in a batch and in any row
order, from a float32 or a float64 reference, and refuses one that is short, reordered or
scaled otherwise; `edges_within` never crosses `graph_id`; knn and spatial at level graph are
computed within each graph and not fitted; every emitted module is at most 400 lines. The
static recipe rules are gates, seeded by `run_negatives_graph.py`. The fixtures are
`dataset-shaper/fixtures/graph-{keypair,knn,level}.*`. Exit 0 when every check holds.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow.fs import LocalFileSystem      # A40: never a Python file object

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILL = SCRIPTS.parent
ROOT = SKILL.parent                       # the skills root: recipe input paths resolve here
FIX = SKILL / "fixtures"
SHAPE = SCRIPTS / "shape.py"
VERIFY = SCRIPTS / "verify_shape.py"
VERBOSE = "-v" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and
                                                        (VERBOSE or not cond) else ""))
    return bool(cond)


def refuses(name, call, error, needle):
    """`call()` raises `error`, and its message names `needle`."""
    try:
        call()
    except error as ex:
        return check(name, needle in str(ex), str(ex))
    return check(name, False, "served without an error")


def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          cwd=str(ROOT), timeout=900)
    return proc.returncode, proc.stdout + proc.stderr


def tail(out, n=4):
    return " | ".join(out.strip().splitlines()[-n:])


def read_out(path):
    """An executor output, bit for bit (csv floats parsed as the round-trip text they are)."""
    if Path(path).suffix == ".parquet":
        return pd.read_parquet(path, filesystem=LocalFileSystem())
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"pre_{abs(hash(str(path)))}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Shaped:
    """One recipe taken through the scripts; the outputs, read back."""

    def __init__(self, name, recipe, work, fmt="parquet"):
        self.name, self.recipe, self.dir = name, recipe, work / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.recipe_path = self.dir / "recipe.in.json"
        self.recipe_path.write_text(json.dumps(recipe, indent=1), encoding="utf-8")
        self.out = self.dir / "shaped"
        rc, self.shape_log = _run(SHAPE, "--recipe", self.recipe_path, "--out-dir", self.out,
                                  "--format", fmt)
        self.ok = check(f"{name}: shape.py exits 0", rc == 0, tail(self.shape_log))
        if self.ok:
            self._verify()
            self._emit_and_twin()
            self._load()

    def _verify(self):
        report = self.dir / "verification.json"
        rc, out = _run(VERIFY, "--recipe", self.recipe_path, "--out-dir", self.out,
                       "--json", report)
        check(f"{self.name}: verify_shape.py exits 0 (graph checks included)", rc == 0,
              tail(out, 6))
        graph = json.loads(report.read_text(encoding="utf-8")).get("graph")
        check(f"{self.name}: verification.graph is pass", graph == "pass", str(graph))
        rc, out = _run(SHAPE, "--check", "--out-dir", self.out)
        check(f"{self.name}: shape.py --check reproduces every file, the edges part included",
              rc == 0 and "edges" in out, tail(out))

    def _emit_and_twin(self):
        self.module_path = self.out / "preprocess.py"     # beside fitted.json, as it loads it
        rc, out = _run(SHAPE, "--transform-only", "--recipe", self.recipe_path, "--fitted",
                       self.out / "fitted.json", "--emit", self.module_path)
        check(f"{self.name}: --transform-only emits the module", rc == 0, tail(out))
        n = len(self.module_path.read_text(encoding="utf-8").splitlines()) if rc == 0 else 0
        check(f"{self.name}: the emitted module is at most 400 lines (C13)", 0 < n <= 400, str(n))
        for part in ("test", "valid"):
            rc, out = _run(VERIFY, "--transform-only", "--recipe", self.recipe_path, "--out-dir",
                           self.out, "--module", self.module_path, "--part", part)
            check(f"{self.name}: twin check on the {part} part exits 0", rc == 0, tail(out, 6))

    def _load(self):
        self.manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.fitted = json.loads((self.out / "fitted.json").read_text(encoding="utf-8"))
        files = self.manifest["outputs"]["files"]
        self.edges = read_out(self.out / Path(files["edges"]["path"]).name)
        self.parts = {k: read_out(self.out / Path(files[k]["path"]).name)
                      for k in ("train", "valid", "test")}
        self.raw = pd.read_csv(ROOT / self.recipe["input"]["path"], low_memory=False)
        self.sets = self.fitted["__split_raw__"]
        self.mod = load_module(self.module_path)

    def reference(self, parts=("train", "valid"), dtype="float32"):
        """The reference as vega/graph-reference@1 ships it (D2 §3.4): dense ids, train then
        valid in __split_raw__ order, every feature as shaped (floats as `dtype`) beside each
        `raw:<column>` read from the input, no raw position; then nodes[["pos", *columns]]."""
        frames = []
        for k in parts:
            f = self.parts[k].drop(columns=[self.manifest["label"]])
            f = f.astype({c: dtype for c in f.columns if f[c].dtype.kind == "f"})
            for name, col in self.mod.GRAPH["reference"]["raw"].items():
                f[name] = self.raw[col].to_numpy()[np.asarray(self.sets[k], dtype="int64")]
            frames.append(f)
        nodes = pd.concat(frames, ignore_index=True)
        nodes.insert(0, "pos", np.arange(len(nodes), dtype="int64"))
        raw_of = np.concatenate([np.asarray(self.sets[k], dtype="int64") for k in parts])
        return nodes[["pos", *self.mod.EDGES_FOR_COLUMNS]], raw_of

    def rows(self, part):
        return self.raw.iloc[np.asarray(self.sets[part], dtype="int64")]


def recipe(name):
    return json.loads((FIX / f"{name}.recipe.json").read_text(encoding="utf-8"))


def step_of(r, op):
    return next(s for s in r["steps"] if s["op"] == op)


# --------------------------------------------------------------------------- properties
def recount(s):
    e, block = s.edges, s.manifest["graph"]
    sp, dp = e["src_part"].astype(str), e["dst_part"].astype(str)
    cross = {f"{a}-{b}": int((((sp == a) & (dp == b)) | ((sp == b) & (dp == a))).sum())
             for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    within = {k: int(((sp == k) & (dp == k)).sum()) for k in ("train", "valid", "test")}
    check(f"{s.name}: cross_part_edges equals a recount of the edges part",
          cross == block["cross_part_edges"], f"{cross} vs {block['cross_part_edges']}")
    check(f"{s.name}: within_part_edges, n_edges and n_nodes equal the recount",
          within == block["within_part_edges"] and block["n_edges"] == len(e) and
          block["n_nodes"] == {k: len(v) for k, v in s.sets.items()} and
          s.manifest["outputs"]["files"]["edges"]["rows"] == len(e))
    check(f"{s.name}: the edges part has the contract's schema",
          list(e.columns) == ["src_pos", "dst_pos", "src_part", "dst_part", "weight"] and
          str(e["src_pos"].dtype) == "int64" and str(e["weight"].dtype) == "float64")


def knn_properties(s):
    e = s.edges
    sp, dp = e["src_part"].astype(str), e["dst_part"].astype(str)
    between = e["src_pos"] != e["dst_pos"]            # a self loop joins a node to itself
    off = int(((sp != "train") & (dp != "train") & between).sum())
    check(f"{s.name}: knn joins no two non-train nodes (no valid–test, test–test, valid–valid)",
          off == 0 and s.manifest["graph"]["cross_part_edges"]["valid-test"] == 0, f"{off}")
    if s.manifest["graph"]["directed"]:
        check(f"{s.name}: every directed knn edge points into train",
              bool((dp[between] == "train").all()))
    fit = s.fitted[s.manifest["graph"]["step"]]
    check(f"{s.name}: the fitted entry names the train matrix it was built on",
          fit.get("fit_on") == "train" and fit.get("reference") == "__split_raw__.train" and
          str(fit.get("matrix_digest", "")).startswith("sha256:") and
          fit.get("n_reference") == len(s.sets["train"]), str(fit.get("matrix")))


def attachments(s, query, reference):
    """{(query raw position, reference raw position): weight} for every pair the edges part
    joins in either direction, a query-sourced row winning: how the evaluation's view attaches
    a query node (R-A3)."""
    e = s.edges
    sp, dp = e["src_part"].astype(str), e["dst_part"].astype(str)
    into = e[sp.isin(reference) & (dp == query)]
    own = e[(sp == query) & dp.isin(reference)]
    out = {(int(q), int(r)): float(w) for r, q, w in zip(into["src_pos"], into["dst_pos"],
                                                         into["weight"])}
    out.update({(int(q), int(r)): float(w) for q, r, w in zip(own["src_pos"], own["dst_pos"],
                                                              own["weight"])})
    return out


def same_edges(a, b):
    """Two {pair: weight} maps hold the same pairs and weights (NaN equal to NaN)."""
    keys = sorted(a)
    return sorted(b) == keys and np.array_equal(np.array([a[k] for k in keys]),
                                                np.array([b[k] for k in keys]), equal_nan=True)


def edges_for_properties(s):
    ref, ref_raw = s.reference()
    rows = s.rows("test")
    q_raw = np.asarray(s.sets["test"], dtype="int64")
    got = s.mod.edges_for(rows, ref)
    check(f"{s.name}: edges_for returns DataFrame[query_pos, ref_pos, weight] of query -> "
          "reference pairs", list(got.columns) == ["query_pos", "ref_pos", "weight"] and
          bool(got["query_pos"].between(0, len(rows) - 1).all()) and
          bool(got["ref_pos"].isin(ref["pos"]).all()) and len(got) > 0, f"{len(got)} rows")
    mine = {(int(q_raw[q]), int(ref_raw[r])): float(w)
            for q, r, w in zip(got["query_pos"], got["ref_pos"], got["weight"])}
    want = attachments(s, "test", ("train", "valid"))
    check(f"{s.name}: edges_for serves exactly the part's test <-> reference pairs and weights "
          "(either direction, each once) from a vega/graph-reference@1 frame",
          same_edges(mine, want) and len(got) == len(mine), f"{len(mine)} vs {len(want)}")
    batch_independence(s, rows, ref, got)
    entry_refusals(s, rows, ref)


def batch_independence(s, rows, ref, got):
    key = ["query_pos", "ref_pos"]
    alone = pd.concat([s.mod.edges_for(rows.iloc[[i]], ref).assign(query_pos=i)
                       for i in range(len(rows))], ignore_index=True)
    alone = alone.sort_values(key, kind="stable").reset_index(drop=True)
    batch = got.sort_values(key, kind="stable").reset_index(drop=True)
    check(f"{s.name}: a row sent alone gets exactly the edges it gets in the batch",
          alone[key].equals(batch[key]) and
          np.array_equal(alone["weight"].to_numpy(), batch["weight"].to_numpy(), equal_nan=True))
    perm = np.random.default_rng(3).permutation(len(rows))
    shuf = s.mod.edges_for(rows.iloc[perm], ref)
    check(f"{s.name}: edges_for does not depend on the order of the query rows",
          set(zip(perm[shuf["query_pos"]].tolist(), shuf["ref_pos"].tolist())) ==
          set(zip(got["query_pos"].tolist(), got["ref_pos"].tolist())))


def entry_refusals(s, rows, ref):
    refuses(f"{s.name}: edges_within refuses a level-node recipe, naming edges_for",
            lambda: s.mod.edges_within(rows), ValueError, "edges_for")
    refuses(f"{s.name}: a reference without pos is refused",
            lambda: s.mod.edges_for(rows, ref.drop(columns=["pos"])), KeyError, "pos")
    need = s.mod.GRAPH["query_columns"]
    if need:
        refuses(f"{s.name}: a query without {need[0]!r} is refused naming it",
                lambda: s.mod.edges_for(rows.drop(columns=[need[0]]), ref), KeyError, need[0])
    col = s.mod.EDGES_FOR_COLUMNS[0]
    refuses(f"{s.name}: a reference without {col!r} is refused naming it",
            lambda: s.mod.edges_for(rows, ref.drop(columns=[col])), KeyError, col)


def knn_reference_rules(kn):
    """What a knn reference must be: the fitted node set, as the fitted space holds it."""
    rows, n = kn.rows("test"), kn.mod.GRAPH["n_fit"]
    ref, _ = kn.reference()
    base = kn.mod.edges_for(rows, ref)
    check("graph-knn: a float64 reference (the shaped part's values) serves exactly what the "
          "float32 one does", kn.mod.edges_for(rows, kn.reference(dtype="float64")[0]).equals(base))
    train, _ = kn.reference(parts=("train",))
    check("graph-knn: the train nodes alone are the fitted node set",
          len(base) == 6 * len(rows) and kn.mod.edges_for(rows, train).equals(base))
    refuses("graph-knn: valid nodes listed first are refused by the fitted matrix digest",
            lambda: kn.mod.edges_for(rows, kn.reference(parts=("valid", "train"))[0]),
            ValueError, "not the matrix")
    backwards = pd.concat([ref.iloc[:n].iloc[::-1], ref.iloc[n:]], ignore_index=True)
    refuses("graph-knn: the train nodes out of the train part's order are refused (a tie goes "
            "to the node that comes first)", lambda: kn.mod.edges_for(rows, backwards),
            ValueError, "order")
    feats = kn.mod.GRAPH["features"]
    scaled = ref.assign(**{c: ref[c] * np.float32(1.001) for c in feats})
    refuses("graph-knn: a reference scaled otherwise is refused by the fitted matrix digest",
            lambda: kn.mod.edges_for(rows, scaled), ValueError, "not the matrix")
    refuses("graph-knn: a reference short of the fitted node set is refused",
            lambda: kn.mod.edges_for(rows, train.iloc[:-1]), ValueError, "fitted on")
    # a query row with a hole in a coordinate: refused naming the column, and the refusal
    # carries the columns a served bundle answers 422 with (never a 500)
    space = list(kn.mod.EDGES_FOR_COLUMNS)
    holed = rows.copy()
    holed.loc[holed.index[1], space[0]] = np.nan
    try:
        kn.mod.edges_for(holed, ref)
        check("graph-knn: a query row with a missing coordinate is refused", False, "served")
    except ValueError as ex:
        named = ex.problem_errors() if callable(getattr(ex, "problem_errors", None)) else []
        check("graph-knn: a query row with a missing coordinate is refused naming the column, "
              "with the rows a service answers 422 with",
              space[0] in str(ex) and [e["column"] for e in named] == [space[0]] and named[0]["rows"] == [1],
              f"{ex} / {named}")


def edges_within_properties(s):
    gid = s.mod.GRAPH["graph_id"]
    rows = s.rows("test")
    got = s.mod.edges_within(rows)
    keys = rows[gid].astype(str).to_numpy()
    check(f"{s.name}: edges_within never crosses {gid} (test part)",
          len(got) > 0 and bool((keys[got["src_pos"]] == keys[got["dst_pos"]]).all()))
    mixed_graphs(s, gid)
    refuses(f"{s.name}: edges_for refuses a level-graph recipe, naming edges_within",
            lambda: s.mod.edges_for(rows, pd.DataFrame({"pos": [0]})), ValueError,
            "edges_within")
    refuses(f"{s.name}: a query without {gid!r} is refused naming it",
            lambda: s.mod.edges_within(rows.drop(columns=[gid])), KeyError, gid)


def mixed_graphs(s, gid):
    """Whole train graphs in shuffled graph order, each graph's rows in input order."""
    tr = s.rows("train").reset_index()
    graphs = list(dict.fromkeys(tr[gid].astype(str)))
    pick = [graphs[i] for i in np.random.default_rng(5).permutation(len(graphs))[:9]]
    mixed = pd.concat([tr[tr[gid].astype(str) == k] for k in pick], ignore_index=True)
    got = s.mod.edges_within(mixed.drop(columns=["index"]))
    keys, at = mixed[gid].astype(str).to_numpy(), mixed["index"].to_numpy()
    mine = set(zip(at[got["src_pos"]].tolist(), at[got["dst_pos"]].tolist()))
    sel = set(mixed["index"].tolist())
    want = {(a, b) for a, b in zip(s.edges["src_pos"].tolist(), s.edges["dst_pos"].tolist())
            if a in sel and b in sel}
    check(f"{s.name}: edges_within on a mixed set of graphs never crosses {gid} and equals "
          "the part's edges among them",
          bool((keys[got["src_pos"]] == keys[got["dst_pos"]]).all()) and mine == want,
          f"{len(mine)} vs {len(want)}")


def level_unfitted(s):
    """Level graph: computed within each graph, not fitted, the graph key a key column."""
    g = s.manifest["graph"]
    fit = s.fitted.get(g["step"]) or {}
    check(f"{s.name}: level graph — fitted false, no fitted entry, no cross-part edge, and "
          f"{g['graph_id']} is a key column", g["level"] == "graph" and g["fitted"] is False
          and "matrix_digest" not in fit and fit.get("fit_on") is None and
          sum(g["cross_part_edges"].values()) == 0 and
          s.manifest["output_schema"][g["graph_id"]]["role"] == "key")
    gid = s.raw[g["graph_id"]].astype(str).to_numpy()
    check(f"{s.name}: no edge of the part crosses two {g['graph_id']} values", len(s.edges) > 0
          and bool((gid[s.edges["src_pos"]] == gid[s.edges["dst_pos"]]).all()))


# --------------------------------------------------------------------------- variants
def csv_parts(s):
    check(f"{s.name}: the parts and the edges part are csv files",
          all(str(v["path"]).endswith(".csv") for v in s.manifest["outputs"]["files"].values()))


def raw_namespace(s):
    """The raw coordinates (or weight) edges_for reads sit beside a scaled column of the same
    input name: the reference names them raw:<column>."""
    cols = s.mod.EDGES_FOR_COLUMNS
    check(f"{s.name}: EDGES_FOR_COLUMNS {cols} name the raw input columns raw:<column>, one "
          "beside a scaled column of the same input name", all(c.startswith("raw:") for c in cols)
          and any(c[4:] in s.parts["train"].columns for c in cols))


def basis_cosine(s):
    check(f"{s.name}: basis resolves to the numeric features at phase 7 (no geometry layer: a "
          "WARN says so)", "resolved to every numeric column" in s.shape_log and
          s.mod.GRAPH["features"] == ["lat", "lon", "lanes", "speed_limit", "flow_vph",
                                      "occupancy"], str(s.mod.GRAPH.get("features")))
    loops = int((s.edges["src_pos"] == s.edges["dst_pos"]).sum())
    check(f"{s.name}: self_loops writes one loop per node",
          loops == sum(len(v) for v in s.sets.values()), str(loops))


def weight_from_input(s):
    e = s.edges
    src, dst = e["src_pos"].to_numpy(), e["dst_pos"].to_numpy()
    raw_w = s.raw["line_cap_mw"].to_numpy()
    own = s.raw["feeder_to"].astype(str).to_numpy()[src] == \
        s.raw["bus_key"].astype(str).to_numpy()[dst]      # the method's direction
    rev = e.set_index(["src_pos", "dst_pos"])["weight"]
    mirrored = rev.loc[list(zip(dst[own].tolist(), src[own].tolist()))].to_numpy()
    check(f"{s.name}: a method edge weighs its source row's input column, and its reverse the "
          "same", bool(own.any()) and np.array_equal(e["weight"].to_numpy()[own], raw_w[src[own]])
          and np.array_equal(mirrored, raw_w[src[own]]) and int(own.sum()) * 2 == len(e))
    raw_namespace(s)


def directed_attachments(s):
    """A directed key_pair: a test row attaches by its own edges and by the reference rows'
    edges into it, each weighed by its source row."""
    e = s.edges
    sp, dp = e["src_part"].astype(str), e["dst_part"].astype(str)
    inward = int((sp.isin(["train", "valid"]) & (dp == "test")).sum())
    outward = int(((sp == "test") & dp.isin(["train", "valid"])).sum())
    check(f"{s.name}: the part is directed and holds reference -> test and test -> reference "
          "rows", s.manifest["graph"]["directed"] and inward > 0 and outward > 0,
          f"{inward} in, {outward} out")


def cap_binds(s):
    grp = s.raw.groupby([s.raw["molecule_id"].astype(str), s.raw["element"].astype(str)]).size()
    uncapped = int((grp * (grp - 1)).sum())
    check(f"{s.name}: the cap binds (fewer edges than the full cliques)",
          len(s.edges) < uncapped, f"{len(s.edges)} edge rows, {uncapped} uncapped")


def spatial_recipe(by_radius, base="graph-knn"):
    r = recipe(base)
    geom = {"id": "S0", "op": "parse_geometry", "columns": ["lat", "lon"],
            "params": {"kind": "latlon", "lat": "lat", "lon": "lon", "crs": "EPSG:4326"},
            "source": "user:the sensors report WGS84 degrees", "rationale": "declare the CRS",
            "alternatives": [], "consequences": {"rows": "none", "columns": "none",
                                                 "downstream": "none"}, "reversible": True}
    repro = dict(geom, id="S3b", op="spatial_reproject", params={"to_crs": "EPSG:32632"},
                 source="user:metres for the radius", rationale="a radius needs a metric CRS")
    r["steps"] = [geom] + r["steps"]
    r["steps"].insert(r["steps"].index(step_of(r, "split")) + 1, repro)
    step_of(r, "graph_edges")["params"] = {
        "method": "spatial", "symmetric": True, "weight": "distance",
        **({"radius_m": 1200, "max_degree": 8} if by_radius else {"k": 5})}
    return r


def district_recipe(work):
    """spatial within each district (level graph): the sensors, grouped 12 to a district."""
    raw = pd.read_csv(FIX / "graph-knn.csv", low_memory=False)
    raw.insert(1, "district", [f"D{i // 12:02d}" for i in range(len(raw))])
    path = work / "graph-knn-districts.csv"
    raw.to_csv(path, index=False)
    r = spatial_recipe(False)
    r["input"].update(path=str(path), columns=int(raw.shape[1]),
                      digest="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
    step_of(r, "split")["params"] = {"kind": "group", "by": "district", "test": 0.2,
                                     "valid": 0.1}
    step_of(r, "graph_edges")["params"].update(k=3, graph_id="district")
    return r


def variant_recipes(work):
    """(name, recipe, format, properties) per variant."""
    weighted, directed = recipe("graph-keypair"), recipe("graph-keypair")
    step_of(weighted, "graph_edges")["params"]["weight"] = "line_cap_mw"
    step_of(directed, "graph_edges")["params"].update(symmetric=False, weight="load_mw")
    cosine, capped, level_knn = recipe("graph-knn"), recipe("graph-level"), recipe("graph-level")
    step_of(cosine, "graph_edges")["params"] = {"method": "knn", "k": 4, "features": "basis",
                                                "metric": "cosine", "symmetric": True,
                                                "self_loops": True, "weight": "distance"}
    step_of(capped, "graph_edges")["params"] = {"method": "key_pair", "columns": ["element"],
                                                "graph_id": "molecule_id", "max_clique": 3,
                                                "symmetric": True}
    step_of(level_knn, "graph_edges")["params"] = {
        "method": "knn", "k": 3, "features": ["charge", "degree"], "graph_id": "molecule_id",
        "symmetric": False, "weight": "distance"}
    node = (knn_properties, recount, edges_for_properties)
    within = (level_unfitted, recount, edges_within_properties)
    return [("variant-spatial-k", spatial_recipe(False), "parquet", node + (raw_namespace,)),
            ("variant-knn-csv", recipe("graph-knn"), "csv", (csv_parts,) + node),
            ("variant-spatial-radius", spatial_recipe(True), "parquet", node[:2]),
            ("variant-knn-basis-cosine", cosine, "parquet", (basis_cosine,) + node),
            ("variant-keypair-weight", weighted, "parquet", (weight_from_input, recount,
                                                             edges_for_properties)),
            ("variant-keypair-directed", directed, "parquet", (directed_attachments,
                                                               edges_for_properties)),
            ("variant-level-clique-capped", capped, "parquet", (cap_binds,) + within),
            ("variant-level-knn", level_knn, "parquet", within),
            ("variant-level-spatial", district_recipe(work), "parquet", within)]


# --------------------------------------------------------------------------- main
def fixtures(work):
    kp, kn, lv = (Shaped(n, recipe(n), work) for n in ("graph-keypair", "graph-knn",
                                                        "graph-level"))
    for s in (s for s in (kp, kn, lv) if s.ok):
        recount(s)
    if kn.ok:
        knn_properties(kn)
        edges_for_properties(kn)
        knn_reference_rules(kn)
    if kp.ok:
        edges_for_properties(kp)
        qq = int(((kp.edges["src_part"] == "test") & (kp.edges["dst_part"] == "test")).sum())
        check("graph-keypair: the part holds test–test rows (the key relation is a fact of the "
              "keys), which edges_for leaves out by construction", qq > 0, f"{qq} rows")
    if lv.ok:
        level_unfitted(lv)
        edges_within_properties(lv)


def main():
    work = Path(ARGS[0]).resolve() if ARGS else Path(tempfile.mkdtemp(prefix="graph-edges-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"test_graph_edges: work in {work}")
    print("\n[fixtures] graph-keypair, graph-knn, graph-level")
    fixtures(work)
    print("\n[variants]")
    for name, r, fmt, properties in variant_recipes(work):
        s = Shaped(name, r, work, fmt=fmt)
        for prop in properties if s.ok else ():
            prop(s)
    failed = [n for n, good in results if not good]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
