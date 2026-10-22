#!/usr/bin/env python3
"""
make_page_fixtures.py — the page fixture matrix (BUILD-PLAN W15; D4 §17.4, R-F9, R-H5): the 35
cases of `page-matrix.json`, each a tiny real network built by the backend's own package and
explained in-process through D3's inspector, written as what the bundle's service and the
showcase slot answer for it, so that `serve_page_fixture.py` can serve the page template over it
under the real routes and `smoke_page.py --fixture` can drive the page with no training and no
service code.

    python3 make_page_fixtures.py [--cases GLOB[,GLOB…]] [--out DIR] [--check EARLIER] [--json]

**How a case is made.** The inspection zoo (`vega-forge/scripts/tests/test_inspect_zoo.py`, I10)
renders the package a bundle carries once, with every block type; a case starts from the zoo's
admission cell its `zoo` names (the catalog family's composition at the matrix's small sizes,
the task's head, loss and metrics), applies its `edits` (`EDITS`), and builds the network with
the package's own `build_model` in eval mode (a regression-family head's target affine fitted,
an svdd centre persisted, as the zoo does). Its data are seeded and synthetic, in the zoo's
inputs (six numeric features and one nominal of cardinality 5): a train part, a test part and a
reference graph where the shape has one. The answers are the package's: the zoo's harness (the
Model side's `SampleAssembler` and the reading's reader, a Predictor without a Recipe — the rows
are already shaped) answers every request, and the package's `NetworkInspector`, opened on the
test rows, explains every preset.

Everything a service would derive is derived by the code that derives it in a bundle:

- `MANIFEST.output` / `.sample` by vega-pack's `output_record`, `schema.json`'s output half by
  its `schema_output`; `/v1/schema` by the serving template's `ApiContract.describe()` over that
  schema (the request contract, `rows.py`/`tables.py`, rendered from the live templates);
- the demonstration sample and the presets by vega-pack's `demo_pack.build` over the train part
  (strata, typical and edge presets in the request form of the shape), answered by the harness;
  the truth of the train rows is the network's own answer for three rows in four (so that every
  class the network answers has a typical preset it gets right) and a seeded draw for the rest;
- the answers by the serving `AnswerCodec` (the `RowPrediction` keys of `MANIFEST.output`'s field
  kinds), each explanation's `prediction` as `routes_inspect` writes it;
- `showcase/data/attribution.json` by vega-show's `AttributionBuilder` over the demo sample,
  through the inspector's `attribute`; `page.json` by `page_config.build_page_config` (gate BD2:
  `check_page_config` must be empty) over the fixed authoring of `page_fixture_story.py`.

**What is written**, per case, under `--out` (default: a fresh temporary directory, printed). The
fixtures are built, not committed: every case is about half a megabyte of exact explanations and
the 35 build in about 15 s on the CPU, always from the live backend, so they can never drift from
the templates the page is tested against (`smoke_page.py --fixture` builds them itself).

    case.json         the matrix entry, the lane and the counts below
    netspec.json      the network's spec (serve_page_fixture renders /v1/code's package from it)
    health.json       GET /health            info.json       GET /v1/info
    schema.json       GET /v1/schema         examples.json   GET /v1/examples
    network.json      GET /v1/network (full weights; every tensor inline)
    answers.json      vega/page-fixture-answers@1: POST /v1/predict's answers per request row
                      (rows and graph nodes by the row, windows by the row and its context,
                      graphs by the graph's rows) and POST /v1/explain's per request, keyed
                      by `request_key` (the request as the page sends it, `detail` aside)
    story.json        showcase/story.json    page.json       showcase/page.json
    attribution.json  showcase/data/attribution.json
    monitor.json      a case whose matrix entry has `monitor` (a forecast): vega/page-fixture-
                      monitor@1, the monitor reference vega-pack's MonitorRender measures (one psi
                      check on an input, pack's own output check; on the train part: a fixture has
                      no valid part) with the request schema and MANIFEST halves the contract is
                      built from — POST /v1/monitor is then answered by the serving templates' own
                      Monitor and route body over the case's answers (`MonitorSide`)
    briefs/<stage>-brief.md  a case whose matrix entry names `briefs`: a stub stage brief, which
                      the story's code notes link (story.report.status `brief-links`)

and `index.json` (`vega/page-fixtures@1`) lists the cases written. The explain requests are the
page's own for each preset (the shape module's editor: a row's `{row}` over the schema's input
columns; a window's `{rows: [its last row], history: the rows before}`; a node's `{rows: [it],
at: 0}`; a graph's `{rows, at: 0}`), with `detail: "exact"` (R-F5), as `views/try.js` sends them.

`--check EARLIER` builds every selected case in a scratch directory and compares it with an earlier
build, document by document, after the volatile digests (`model_version`, the template pin) are
set aside; the numbers are held to 1e-6 relative (a float32 network on another BLAS may differ in
its last bits): the build is deterministic. Exit 0: written (or identical). 1: `--check` found a
difference. 2: a case could not be built.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fnmatch
import hashlib
import importlib
import json
import math
import re
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = HERE.parents[2]
FORGE = REPO / "vega-forge" / "scripts"
FORGE_TESTS = FORGE / "tests"
TEMPLATES = FORGE / "backends" / "pytorch" / "templates"
SERVE = TEMPLATES / "serving" / "serve"
PACK = REPO / "vega-pack" / "scripts"
MATRIX = HERE / "page-matrix.json"
INDEX_SCHEMA, ANSWERS_SCHEMA, MATRIX_SCHEMA = ("vega/page-fixtures@1", "vega/page-fixture-answers@1",
                                               "vega/page-matrix@1")
MONITOR_FILE, MONITOR_SCHEMA = "monitor.json", "vega/page-fixture-monitor@1"
WATCH_PKG = "fixture_watch"
BRIEF_TEXT = "# {stage} brief\n\nA stub stage brief of the page fixture matrix: the code notes link it.\n"
for _p in (str(SCRIPTS), str(HERE), str(FORGE_TESTS), str(FORGE_TESTS / "matrix"), str(PACK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 11
N_ROWS, N_TEST = 48, 8
SERIES, N_STEPS, N_TEST_STEPS = ("A", "B", "C"), 28, 6
N_TRAIN_QUERIES, N_TEST_QUERIES = 32, 8
N_TRAIN_GRAPHS, N_TEST_GRAPHS = 8, 2
AGREE = 0.75                       # the share of train rows whose truth is the network's answer
PER_STRATUM = 2                    # demo items per stratum (pack's --demo-sample)
API_VERSION = "v1"
SERVE_FILL = {"NOUN": "row", "NOUNS": "rows", "ROW_MODEL": "Row", "PACKAGE": "fixture_pkg",
              "CMD": "fixture", "ENV": "FIXTURE"}
SERVE_MODULES = ("problems", "rows", "tables", "answers", "contract")
CUSTOM_MODULE = '''"""A user module of the page fixture matrix (case 34): a tanh layer the catalog does not know."""
import torch
from torch import nn


class Scorer(nn.Module):
    def __init__(self, width, out):
        super().__init__()
        self.linear = nn.Linear(width, out)

    def forward(self, x):
        return torch.tanh(self.linear(x))


def build(in_shape, params):
    return Scorer(int(in_shape[-1]), int(params["out_shape"][-1]))
'''


CUSTOM_SOURCES = {"custom/page_scorer.py": CUSTOM_MODULE}


class FixtureError(Exception):
    """A case could not be built (exit 2)."""


# ============================================================================ the matrix

def load_matrix(path: Path = MATRIX) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != MATRIX_SCHEMA:
        raise FixtureError(f"{path} is not {MATRIX_SCHEMA}")
    cases = doc["cases"]
    numbers = [c["n"] for c in cases]
    if numbers != list(range(1, len(cases) + 1)):
        raise FixtureError(f"{path}: the cases are not numbered 1…{len(cases)} in order")
    return cases


def selected(cases: list[dict], globs: str | None) -> list[dict]:
    if not globs:
        return cases
    out = []
    for pattern in [g.strip() for g in globs.split(",") if g.strip()]:
        hits = [c for c in cases if fnmatch.fnmatchcase(c["id"], pattern) or str(c["n"]) == pattern]
        if not hits:
            raise FixtureError(f"--cases {pattern!r} matches no case of the matrix")
        out += [c for c in hits if c not in out]
    return out


# ============================================================================ the edits

def _numeric_only(spec: dict, zoo) -> None:
    """The golden path's shape: numeric inputs only, so the network is a pure dense chain."""
    contract = spec["data_contract"]
    contract["features"] = [f for f in contract["features"] if f.get("role") != "nominal"]
    contract["input_dim"] = len(contract["features"])
    spec["blocks"] = [b for b in spec["blocks"] if b["type"] not in ("embedding", "concat")]


def _user(name: str, spec: dict) -> None:
    spec["family"] = {"name": name, "source": "user:page-matrix"}


def _block(bid: str, kind: str, params: dict) -> dict:
    return {"id": bid, "type": kind, "params": params, "source": "user:page-matrix",
            "rationale": "the page fixture matrix's user-defined composition", "alternatives": []}


def _user_parallel(spec: dict, zoo) -> None:
    """A user-defined family: two dense branches side by side, reshaped and flattened back."""
    _user("two-branch", spec)
    embed, concat, head = spec["blocks"][0], spec["blocks"][1], spec["blocks"][-1]
    spec["blocks"] = [embed, concat,
                      _block("B3", "parallel", {"merge": "concat", "inner": [
                          {"type": "dense", "params": {"units": 8, "activation": "relu"}},
                          {"type": "dense", "params": {"units": 8, "activation": "tanh"}}]}),
                      _block("B4", "reshape", {"shape": [4, 4]}),
                      _block("B5", "flatten", {}), {**head, "id": "B6"}]


def _user_custom(spec: dict, zoo) -> None:
    """A user-defined family with a `custom` block (its module ships with the package)."""
    _user("custom-scorer", spec)
    embed, concat, head = spec["blocks"][0], spec["blocks"][1], spec["blocks"][-1]
    spec["blocks"] = [embed, concat, _block("B3", "dense", {"units": 16, "activation": "relu"}),
                      _block("B4", "custom", {"code_ref": "custom/page_scorer.py", "out_shape": [8]}),
                      {**head, "id": "B5"}]


EDITS = {"numeric_only": _numeric_only, "user_parallel": _user_parallel, "user_custom": _user_custom,
         "quantile": lambda spec, zoo: zoo._quantile(spec),
         "attention_pool": lambda spec, zoo: zoo._attention_reducer(spec),
         "layernorm": lambda spec, zoo: zoo._layernorm_inner(spec)}


def case_spec(case: dict, zoo) -> dict:
    """The case's netspec: the zoo cell's, then the case's edits in order."""
    if case["zoo"] not in zoo.CASES:
        raise FixtureError(f"case {case['id']}: the zoo has no cell {case['zoo']!r}")
    spec = zoo.spec_of(zoo.CASES[case["zoo"]])
    for name in case.get("edits") or ():
        if name not in EDITS:
            raise FixtureError(f"case {case['id']}: no edit {name!r} ({', '.join(EDITS)})")
        EDITS[name](spec, zoo)
    return spec


# ============================================================================ the data

def _features(rng, n: int, nominal: bool = True) -> dict:
    cols = {f"x{i}": [round(float(v), 4) for v in rng.standard_normal(n)] for i in range(6)}
    if nominal:
        cols["zone"] = [int(v) for v in rng.integers(0, 5, n)]
    return cols


def _edges(rng, n: int, m: int):
    import numpy as np
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    pick = sorted(rng.choice(len(pairs), size=m, replace=False).tolist())
    return np.asarray([pairs[p] for p in pick], dtype=np.int64).T


class World:
    """The raw parts of one case (already shaped: the harness has no Recipe), its request
    columns and the reference graph the harness attaches a query to."""

    shape = "row"
    meta = None
    keys: tuple = ()

    def __init__(self, spec: dict, rng):
        import pandas as pd
        self.spec, self.pd = spec, pd
        frame = pd.DataFrame(_features(rng, N_ROWS + N_TEST))
        self.train, self.test = frame.iloc[:N_ROWS].reset_index(drop=True), frame.iloc[N_ROWS:].reset_index(drop=True)

    def reference(self, pkg, spec):
        return None

    def edges_for(self, shaped):
        return None

    def example(self):
        """(rows, history) the inspector opens on: the test part."""
        return self.test.copy(), None

    def key_inputs(self) -> list[dict]:
        return []


class WindowWorld(World):
    shape = "window"

    def __init__(self, spec: dict, rng):
        import numpy as np
        import pandas as pd
        self.spec, self.pd = spec, pd
        sample = spec["data_contract"]["sample"]
        self.series_by, self.order_by = sample["series_by"], sample["order_by"]
        n = N_STEPS + N_TEST_STEPS
        frame = pd.DataFrame({self.series_by: np.tile(SERIES, n), self.order_by: np.repeat(np.arange(n), len(SERIES)),
                              **_features(rng, n * len(SERIES))})
        label = spec["task"].get("label")
        if sample.get("autoregressive") and label:
            frame[label] = np.round(3.0 + 2.0 * rng.standard_normal(len(frame)), 4)
        cut = N_STEPS * len(SERIES)
        self.train, self.test = frame.iloc[:cut].reset_index(drop=True), frame.iloc[cut:].reset_index(drop=True)
        self.keys = (self.series_by, self.order_by)

    def history_for_test(self, need: int):
        """The last `need` train rows of every series: the context of the test part's first rows."""
        parts = [self.train[self.train[self.series_by] == s].tail(need) for s in SERIES]
        return self.pd.concat(parts).sort_values([self.order_by, self.series_by], kind="stable").reset_index(drop=True)

    def example(self):
        need = int(self.spec["data_contract"]["sample"]["length"]) - 1
        a = self.test[self.test[self.series_by] == SERIES[0]]
        h = self.train[self.train[self.series_by] == SERIES[0]].tail(need)
        return a.reset_index(drop=True), h.reset_index(drop=True)

    def key_inputs(self) -> list[dict]:
        return [{"name": self.series_by, "required": True, "dtype": "string", "allowed": list(SERIES),
                 "distinct_seen": len(SERIES)},
                {"name": self.order_by, "required": True, "dtype": "int",
                 "range_seen": [0.0, float(N_STEPS - 1)]}]


class NodeWorld(World):
    """The reference graph (40 nodes, 120 weighted edges, a key per node), and query rows each
    linked to the reference nodes of its key (key-pair style, as the zoo's)."""

    shape = "node"
    N_REF, N_EDGES, KEYS = 40, 120, 8

    def __init__(self, spec: dict, rng):
        import numpy as np
        import pandas as pd
        self.spec, self.pd = spec, pd
        self.meta = {"n_nodes": self.N_REF, "n_edges": self.N_EDGES, "method": "key_pair", "directed": True,
                     "self_loops": False}
        self.nodes = pd.DataFrame({"ref": np.arange(self.N_REF), "key": np.arange(self.N_REF) % self.KEYS,
                                   **_features(rng, self.N_REF)})
        ends = _edges(rng, self.N_REF, self.N_EDGES)
        self.edges = pd.DataFrame({"src_ref": ends[0], "dst_ref": ends[1],
                                   "weight": np.round(rng.uniform(0.5, 1.5, self.N_EDGES), 4)})
        n = N_TRAIN_QUERIES + N_TEST_QUERIES
        frame = pd.DataFrame({"key": [int(v) for v in rng.integers(0, self.KEYS, n)], **_features(rng, n)})
        self.train = frame.iloc[:N_TRAIN_QUERIES].reset_index(drop=True)
        self.test = frame.iloc[N_TRAIN_QUERIES:].reset_index(drop=True)
        self.keys = ("key",)

    def reference(self, pkg, spec):
        return pkg.domain.GraphReference.from_frames(self.nodes, self.edges, spec.feature_contract)

    def edges_for(self, shaped):
        import numpy as np
        keys = self.nodes["key"].to_numpy()
        links = [(i, int(ref), 0.75 + 0.25 * (int(ref) % 3))
                 for i, key in enumerate(shaped["key"].to_numpy()) for ref in np.flatnonzero(keys == int(key))]
        return self.pd.DataFrame(links, columns=["query_pos", "ref_pos", "weight"])

    def key_inputs(self) -> list[dict]:
        return [{"name": "key", "required": True, "dtype": "int", "allowed": "any", "role": "graph",
                 "note": "read by the graph step, not by the network: a row is attached to the "
                         "reference graph through this column"}]


class GraphWorld(World):
    """Graphs of 10 nodes and 30 weighted edges each; a request sends whole graphs."""

    shape = "graph"
    NODES, EDGES = 10, 30

    def __init__(self, spec: dict, rng):
        import numpy as np
        import pandas as pd
        self.spec, self.pd = spec, pd
        self.graph_id = spec["data_contract"]["sample"]["graph_id"]
        n_graphs = N_TRAIN_GRAPHS + N_TEST_GRAPHS
        self.meta = {"n_nodes": n_graphs * self.NODES, "n_edges": n_graphs * self.EDGES, "method": "key_pair",
                     "directed": True, "self_loops": False}
        ids = [f"g{i}" for i in range(n_graphs)]
        frame = pd.DataFrame({self.graph_id: np.repeat(ids, self.NODES), "node": np.tile(np.arange(self.NODES), n_graphs),
                              **_features(rng, n_graphs * self.NODES)})
        self.within = {g: (_edges(rng, self.NODES, self.EDGES), np.round(rng.uniform(0.5, 1.5, self.EDGES), 4)) for g in ids}
        cut = N_TRAIN_GRAPHS * self.NODES
        self.train, self.test = frame.iloc[:cut].reset_index(drop=True), frame.iloc[cut:].reset_index(drop=True)
        self.keys = (self.graph_id, "node")

    def edges_for(self, shaped):
        where = {(str(g), int(n)): i for i, (g, n) in enumerate(zip(shaped[self.graph_id], shaped["node"]))}
        edges = []
        for g in dict.fromkeys(str(v) for v in shaped[self.graph_id]):
            ends, weights = self.within[g]
            edges += [(where[(g, int(a))], where[(g, int(b))], float(w)) for (a, b), w in zip(ends.T, weights)
                      if (g, int(a)) in where and (g, int(b)) in where]
        return self.pd.DataFrame(edges, columns=["src_pos", "dst_pos", "weight"])

    def example(self):
        return self.test.copy(), None

    def key_inputs(self) -> list[dict]:
        note = ("read by the graph step, not by the network: a row is attached to the rows of its "
                "own graph through this column")
        return [{"name": self.graph_id, "required": True, "dtype": "string", "allowed": "any", "role": "graph",
                 "note": "the graph each row belongs to: the rows sharing a value are answered as one graph"},
                {"name": "node", "required": True, "dtype": "int", "allowed": "any", "role": "graph", "note": note}]


WORLDS = {"row": World, "window": WindowWorld, "graph-node": NodeWorld, "graph-graph": GraphWorld}


# ============================================================================ the request key

def _norm(value):
    """A cell as the page and the server both read it: a number as a float (an integral JSON
    number and its float are one cell), a bool, a string, None."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return float(f"{f:.9g}")
    if hasattr(value, "item"):
        return _norm(value.item())
    return str(value)


def norm_row(row: dict) -> dict:
    return {str(k): _norm(v) for k, v in sorted(row.items())}


def row_key(row: dict) -> str:
    return json.dumps(norm_row(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rows_key(rows) -> str:
    return json.dumps([norm_row(r) for r in rows], sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(body: dict) -> str:
    """The key of an explain request: the request as sent, `detail` aside (the server answers
    the exact explanation whatever detail it is asked), rows normalised."""
    out = {}
    for k, v in sorted(body.items()):
        if k == "detail" or v is None:
            continue
        if k == "row":
            out[k] = norm_row(v)
        elif k in ("rows", "history"):
            out[k] = [norm_row(r) for r in v]
        else:
            out[k] = v
    return json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def window_context(rows: list, history: list | None, series_by: str, need: int) -> list:
    """[(row, its context)] of a window request: per row, the rows before it in history ++ rows
    of its series, the last `need` of them (what the Predictor's assembly reads)."""
    joined = list(history or []) + list(rows)
    out, start = [], len(history or [])
    for i, row in enumerate(rows):
        before = [r for r in joined[:start + i] if str(r.get(series_by)) == str(row.get(series_by))]
        out.append((row, before[-need:] if need > 0 else []))
    return out


# ============================================================================ the serving codecs

def serve_package(work: Path):
    """The request contract and the answer codec of the serving templates (`serve/{problems,
    rows, tables, answers, contract}.py`), rendered into a package of their own."""
    pkg = work / "fixture_serve"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    fill = lambda m: SERVE_FILL.get(m.group(1), m.group(0))  # noqa: E731
    for name in SERVE_MODULES:
        text = (SERVE / f"{name}.py").read_text(encoding="utf-8")
        (pkg / f"{name}.py").write_text(re.sub(r"\{\{([A-Z0-9_]+)\}\}", fill, text), encoding="utf-8")
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    return importlib.import_module("fixture_serve.contract")


def endpoints() -> list[str]:
    """MANIFEST.serve.endpoints as pack writes them: every route module's ENDPOINTS literal."""
    out = set()
    for path in sorted(SERVE.glob("routes_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ENDPOINTS" for t in node.targets):
                out.update(f"{m} {p}" for m, p in ast.literal_eval(node.value))
    return sorted(out)


def watch_package(work: Path):
    """(the Monitor class, the monitor route module): the serving templates' monitor side
    rendered into a package of its own — `domain.py` (the DriftReading a reading is),
    `watch/monitor.py` (templates/serving/package/monitor.py) and `routes_monitor.py`
    (serve/routes_monitor.py, whose `monitor_body` answers the route). Rendered once a process."""
    if f"{WATCH_PKG}.routes_monitor" not in sys.modules:
        pkg = work / WATCH_PKG
        (pkg / "watch").mkdir(parents=True, exist_ok=True)
        for init in (pkg / "__init__.py", pkg / "watch" / "__init__.py"):
            init.write_text("", encoding="utf-8")
        fill = lambda m: SERVE_FILL.get(m.group(1), m.group(0))  # noqa: E731
        for src, dst in ((TEMPLATES / "domain.py", pkg / "domain.py"),
                         (TEMPLATES / "serving" / "package" / "monitor.py", pkg / "watch" / "monitor.py"),
                         (SERVE / "routes_monitor.py", pkg / "routes_monitor.py")):
            dst.write_text(re.sub(r"\{\{([A-Z0-9_]+)\}\}", fill, src.read_text(encoding="utf-8")), encoding="utf-8")
        if str(work) not in sys.path:
            sys.path.insert(0, str(work))
    monitor = importlib.import_module(f"{WATCH_PKG}.watch.monitor")
    return monitor.Monitor, importlib.import_module(f"{WATCH_PKG}.routes_monitor")


class MonitorSide:
    """POST /v1/monitor of a case that ships a monitor reference, answered as the service answers
    it: the request validated by the contract's MonitorRequest (a refusal is the service's 422),
    the rows answered by `predict` (the case's answers, as /v1/predict), and the body written by
    the route's own `monitor_body` with the templates' Monitor over the reference."""

    def __init__(self, doc: dict, contract_mod, monitor_cls, route):
        self.doc, self.route = doc, route
        self.contract = contract_mod.ApiContract(doc["request_schema"], doc["manifest"], max_rows=10000,
                                                 version=_digest(doc["request_schema"]))
        monitor = monitor_cls(doc["reference"], recipe=None, labels=None)
        version = doc["manifest"]["model_version"]
        self.state = types.SimpleNamespace(ready=lambda: None, monitor=monitor, contract=self.contract,
                                           envelope=lambda n: self.contract.envelope(version, n, API_VERSION))

    def answer(self, body: dict, predict, report: bool) -> dict:
        """The route's body; ValueError (with the contract's words) when it refuses the request."""
        import pandas as pd
        from pydantic import ValidationError
        try:
            req = self.contract.models.MonitorRequest.model_validate(body)
        except ValidationError as exc:
            raise ValueError("the monitor request breaks its contract: "
                             + "; ".join(f"{'/'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])) from None
        frame = self.contract.tables.frame(req.rows)
        said = predict({k: body[k] for k in ("rows", "history") if body.get(k) is not None})["predictions"]
        answers = pd.DataFrame(said).reset_index(drop=True)
        return self.route.monitor_body(self.state, frame, getattr(req, "truth", None), answers, report)


def monitor_document(fx, case: dict) -> dict | None:
    """vega/page-fixture-monitor@1 of a case whose matrix entry has `monitor` (module docstring):
    the reference measured by vega-pack's MonitorRender from a plan validated by its own
    validate_plan, plus pack's output check."""
    if not case.get("monitor"):
        return None
    from monitor_plan import output_check, validate_plan
    from monitor_reference import MonitorInputs, MonitorRender
    output, label = fx.manifest["output"], fx.spec["task"].get("label")
    column = next(f["name"] for f in fx.spec["data_contract"]["features"] if f.get("role") != "nominal")
    plan = {"checks": [{"id": "inputs-psi", "kind": "psi", "column": column, "concern": "[page-matrix:inputs]",
                        "title": f"{column} distribution against the train part"}],
            "note": "the page fixture matrix's monitor: one input and the answer"}
    problems = validate_plan(plan, output, fx.schema)
    if problems:
        raise FixtureError(f"case {case['id']}: the monitor plan: {problems}")
    auto = output_check(output)
    plan["checks"] += [auto] if auto else []
    train = fx.world.train.copy()
    inputs = MonitorInputs(train=train, valid=train, stored_valid=None, predictor=_Answering(fx),
                           recipe=lambda: None, classes=tuple(str(c) for c in output.get("classes") or ()),
                           label=label, decode={}, ignored=tuple(fx.schema.get("ignored") or ()),
                           sample=dict(fx.manifest["sample"]))
    reference = MonitorRender(output, fx.schema, inputs).render(plan)
    manifest = {k: fx.manifest[k] for k in ("output", "sample", "model_version")}
    return {"schema": MONITOR_SCHEMA, "reference": reference, "manifest": manifest, "request_schema": fx.schema}


# ============================================================================ one case

class FixtureCase:
    """One case built end to end (module docstring)."""

    def __init__(self, case: dict, zoo, contract_mod, catalogs):
        import numpy as np
        import torch
        self.case, self.zoo, self.np = case, zoo, np
        self.pkg = pkg = zoo.package()
        self.spec = case_spec(case, zoo)
        cell = zoo.CASES[case["zoo"]].cell
        self.world = WORLDS[cell.shape_tag](self.spec, np.random.default_rng(SEED + case["n"]))
        if any(b["type"] == "custom" for b in self.spec["blocks"]):
            target = pkg.root / "model" / "custom"
            target.mkdir(parents=True, exist_ok=True)
            (target / "page_scorer.py").write_text(CUSTOM_SOURCES["custom/page_scorer.py"], encoding="utf-8")
        self.net_spec = pkg.domain.NetSpec.from_dict(self.spec)
        torch.manual_seed(SEED)
        self.model = pkg.build(self.spec)
        zoo._fit_affine(pkg, self.model, self.spec)
        assembly = pkg.domain.NetSpec.from_dict(zoo._inductive(self.spec))
        self.assembler = pkg.assemble.SampleAssembler(assembly, self.world.reference(pkg, assembly))
        rows, history = self.world.example()
        example = self.assembler.assemble(rows, history, self.world.edges_for(rows)).batch
        self.reading = zoo.orec.resolve_reading(self.spec)
        if self.reading == "anomaly-svdd":
            zoo._persist_centre(pkg, self.model, self.net_spec, example)
        self.manifest = self._manifest(example)
        self.schema = self._schema()
        self.contract = contract_mod.ApiContract(self.schema, self.manifest, max_rows=10000,
                                                 version=_digest(self.schema))
        self.harness = zoo._Harness(pkg, self.model, self.net_spec, self.manifest, self.assembler, self.world)
        self.inspector = pkg.inspection.NetworkInspector(self.model, self.net_spec, self.harness,
                                                         self.manifest["output"], self.tolerance)
        self.inspector.open(pkg.query.ExplainQuery(rows.reset_index(drop=True), history, len(rows) - 1,
                                                   detail="summary"))
        status = self.inspector.status()
        if not status.get("ready"):
            raise FixtureError(f"case {case['id']}: the inspector did not open: {status.get('why')}")
        self.status = status
        self.rules = pkg.query.QueryRules.of(self.net_spec)
        self.catalogs = catalogs

    # -- the records ---------------------------------------------------------------------
    def _manifest(self, example) -> dict:
        import torch
        task = self.spec["task"]
        with torch.no_grad():
            out = self.model(example)
            rows = self.model.readout.row_score(out, example, self.model)
        scores = out.double().numpy().ravel().tolist()
        if rows is not None:
            scores += rows.double().numpy().ravel().tolist()
        threshold = self.zoo._threshold(self.reading, bool(task.get("label")), task.get("classes") or (),
                                        None if rows is None else rows.numpy())
        tolerance = self.zoo.Tolerance.for_scores(scores).as_json()
        self.tolerance = float(tolerance["value"])
        recipe = {"steps": []}
        return {"output": self.zoo.orec.output_record(self.spec, threshold, task.get("label")),
                "sample": self.zoo.orec.sample_record(self.spec, recipe, self.world.meta),
                "model_version": "sha256:" + hashlib.sha256(self.case["id"].encode()).hexdigest(),
                "roundtrip": {"tolerance": tolerance}, "chain": {"shape": {"manifest_digest": None}}}

    def _schema(self) -> dict:
        """schema.json: the network's inputs (with the train part's ranges), the shape's key
        columns, a forecast's autoregressive target; the output half from the output record."""
        frame = self.world.train
        inputs = []
        for f in self.spec["data_contract"]["features"]:
            name = f["name"]
            values = frame[name].astype(float)
            col = {"name": name, "required": True, "dtype": "int" if f.get("role") == "nominal" else "float",
                   "range_seen": [round(float(values.min()), 4), round(float(values.max()), 4)]}
            if f.get("role") == "nominal":
                col["note"] = "the recipe types this column nominal; an unseen value is answered and flagged"
            inputs.append(col)
        inputs = self.world.key_inputs() + inputs
        label = self.spec["task"].get("label")
        sample = self.manifest["sample"]
        if sample.get("shape") == "window" and sample.get("autoregressive") and label:
            values = frame[label].astype(float)
            inputs.append({"name": label, "required": True, "dtype": "float",
                           "range_seen": [round(float(values.min()), 4), round(float(values.max()), 4)],
                           "role": "autoregressive",
                           "note": "the target's own past: the forecast reads it as an input channel"})
        out = self.zoo.orec.schema_output(self.manifest["output"], sample)
        truth = self.manifest["output"].get("truth")
        ignored = [label] if label and truth and not any(c["name"] == label for c in inputs) else []
        doc = {"input": inputs, **out, "ignored": ignored}
        if truth:
            doc["label"] = {"name": label, "decode": {}, "source": "the page fixture matrix's synthetic truth"}
        return doc

    # -- the answers ---------------------------------------------------------------------
    def frame(self, rows):
        """JSON rows as the service reads a request: through the contract's table codec."""
        return self.contract.tables.frame(rows)

    def json_rows(self, frame) -> list:
        return self.contract.tables.json_rows(frame.reset_index(drop=True))

    def predict(self, frame, history=None):
        return self.harness.predict(frame.reset_index(drop=True), history)

    def answers(self, out) -> list:
        return self.contract.answers.rows(out.reset_index(drop=True), ())

    def label_train(self) -> None:
        """The truth column of the train and test parts: the network's own decision for three
        rows in four, a seeded draw for the rest (module docstring)."""
        truth = self.manifest["output"].get("truth")
        label = self.spec["task"].get("label")
        if not truth or not label or self.manifest["sample"].get("autoregressive"):
            return
        rng = self.np.random.default_rng(SEED + 100 + self.case["n"])
        for part in ("train", "test"):
            frame = getattr(self.world, part)
            got = self.predict(frame, self._part_history(part))
            decided = self._decided(got, truth, rng)
            keep = rng.random(len(frame)) < AGREE
            drawn = self._decided(None, truth, rng, n=len(frame))
            values = [d if k else r for d, r, k in zip(decided, drawn, keep)]
            if self.world.shape == "graph":           # one truth per graph: its first row's
                first = {}
                for g, v in zip(frame[self.world.graph_id], values):
                    first.setdefault(str(g), v)
                values = [first[str(g)] for g in frame[self.world.graph_id]]
            frame[label] = values

    def _part_history(self, part: str):
        if part == "test" and self.world.shape == "window":
            return self.world.history_for_test(int(self.manifest["sample"]["history_rows"]))
        return None

    def _decided(self, got, truth, rng, n: int | None = None) -> list:
        """The truth cells of the network's answers (or, with n, n seeded cells), in the raw
        label's form: a class (the task's own value), label names joined, a number."""
        out = self.manifest["output"]
        kinds = out["field_kinds"]
        kind = truth["type"]
        classes = list(self.spec["task"].get("classes") or [])
        if kind == "class":
            names = [str(c) for c in out.get("classes") or classes]
            raw = {str(c): c for c in classes} if classes else {n_: n_ for n_ in names}
            if got is None:
                return [raw.get(names[int(i)], names[int(i)]) for i in rng.integers(0, len(names), n)]
            field = next((f for f, k in kinds.items() if k["kind"] in ("class", "alarm")), None)
            cells = got[field].tolist()
            if kinds[field]["kind"] == "alarm":
                return [raw.get(names[1 if (c is True or str(c) in ("1", "True", "true")) else 0]) for c in cells]
            return [raw.get(str(c), random_class(names, raw, rng)) for c in cells]
        if kind == "labels":
            names = [str(c) for c in out.get("classes") or classes]
            sep = truth.get("sep") or "|"
            if got is None:
                return [sep.join(nm for nm in names if rng.random() < 0.4) for _ in range(n)]
            flags = [f for f, k in kinds.items() if k["kind"] == "flag"]
            return [sep.join(str(kinds[f]["of"]) for f in flags if _flag(row[f])) for _, row in got.iterrows()]
        if got is None:
            return [round(float(v), 4) for v in 3.0 + 2.0 * rng.standard_normal(n)]
        field = next((f for f, k in kinds.items() if k["kind"] == "value"), None)
        if field is None:
            quantiles = [(f, k["of"]) for f, k in kinds.items() if k["kind"] == "quantile"]
            field = min(quantiles, key=lambda q: abs(float(q[1]) - 0.5))[0]
        return [round(float(v) + float(e), 4) if v is not None and math.isfinite(float(v)) else 0.0
                for v, e in zip(got[field].tolist(), 0.3 * rng.standard_normal(len(got)))]

    # -- the documents -------------------------------------------------------------------
    def examples(self) -> tuple[dict, dict]:
        """(vega/examples@2, the demo documents) — demo and presets by vega-pack's demo_pack."""
        import demo_pack
        recipe = types.SimpleNamespace(transform=None, GRAPH={}, HISTORY_COLUMNS=())
        inputs = demo_pack.DemoInputs(train=self.world.train.copy(), output=self.manifest["output"],
                                      sample=self.manifest["sample"], schema=self.schema,
                                      per_stratum=PER_STRATUM, recipe=recipe)
        docs = demo_pack.build(inputs, _Answering(self))
        demo_pack.check_documents(docs)
        history = self._part_history("test")
        test = {"source": "synthetic", "rows": self.json_rows(self._request_frame(self.world.test)),
                "history": self.json_rows(self._request_frame(history)) if history is not None else None}
        body = {"schema": "vega/examples@2", "test": test, "demo": docs.demo,
                "presets": list(docs.presets["presets"])}
        return body, {"notes": list(docs.notes), "remarks": list(docs.remarks)}

    def _request_frame(self, frame):
        names = [c["name"] for c in self.schema["input"]]
        return frame[[c for c in names if c in frame.columns]]

    def page_requests(self, examples: dict) -> list[dict]:
        """The explain request of every preset, as the shape module's editor sends it."""
        names = [c["name"] for c in self.schema["input"]]
        dtype = {c["name"]: c.get("dtype") for c in self.schema["input"]}
        cell = lambda name, v: _cell(v, dtype.get(name))  # noqa: E731
        out = []
        for p in examples["presets"]:
            if self.world.shape == "row":
                row = p.get("row") or (p.get("rows") or [None])[0] or {}
                out.append({"row": {n: cell(n, row.get(n)) for n in names}, "detail": "exact"})
            elif self.world.shape == "window":
                rows = [*(p.get("history") or []), *(p.get("rows") or ([p["row"]] if p.get("row") else []))]
                rows = [{n: cell(n, r.get(n)) for n in names} for r in rows]
                out.append({"rows": rows[-1:], "history": rows[:-1] or None, "detail": "exact"})
            elif self.world.shape == "node":
                rows = p.get("rows") or [p.get("row")]
                last = {**rows[-1], **{n: cell(n, rows[-1].get(n)) for n in names}}
                out.append({"rows": [last], "at": 0, "detail": "exact"})
            else:
                rows = p.get("rows") or [p.get("row")]
                out.append({"rows": [{**r, **{n: cell(n, r.get(n)) for n in names}} for r in rows], "at": 0,
                            "detail": "exact"})
        return out

    def explain(self, body: dict) -> dict:
        """POST /v1/explain's answer: the envelope and the explanation, its prediction the row's
        /v1/predict answer (routes_inspect's _fields and _prediction)."""
        fields = {k: v for k, v in body.items() if k not in ("row", "rows", "history")}
        for name in ("rows", "history"):
            if body.get(name) is not None:
                fields[name] = self.frame(body[name]) if body[name] else []
        if body.get("row") is not None:
            fields["rows"] = self.frame([body["row"]])
        query = self.rules.parse(fields)
        explanation = self.inspector.explain(query)
        row = dict(explanation["prediction"])
        if isinstance(row.get("warnings"), list):
            row["warnings"] = "; ".join(str(w) for w in row["warnings"])
        import pandas as pd
        answer = self.contract.answers.rows(pd.DataFrame([row]), ())[0]
        explanation["prediction"] = self.contract.answers.response_model().model_validate(answer).model_dump(mode="json")
        return {**self.envelope(1), "explanation": explanation}

    def envelope(self, n: int) -> dict:
        return self.contract.envelope(self.manifest["model_version"], n, API_VERSION)

    def answer_table(self, examples: dict, requests: list[dict]) -> dict:
        """vega/page-fixture-answers@1 (module docstring)."""
        table = {"schema": ANSWERS_SCHEMA, "shape": self.world.shape, "envelope": self.envelope(0),
                 "keys": self._keys(), "rows": {}, "windows": {}, "graphs": {}, "explain": {}}
        for part, history in (("train", None), ("test", self._part_history("test"))):
            self._answer_frame(table, self._request_frame(getattr(self.world, part)), history)
        for p in examples["presets"]:
            rows = p.get("rows") or ([p["row"]] if p.get("row") else [])
            self._answer_frame(table, self.frame(rows), self.frame(p["history"]) if p.get("history") else None)
        for body in requests:
            got = self.explain(body)
            table["explain"][request_key(body)] = got
            self._answer_frame(table, self.frame(body.get("rows") or [body["row"]]),
                               self.frame(body["history"]) if body.get("history") else None)
        return table

    def _keys(self) -> dict:
        sample = self.manifest["sample"]
        return {"series_by": sample.get("series_by"), "order_by": sample.get("order_by"),
                "graph_id": sample.get("graph_id"), "need": int(sample.get("history_rows") or 0),
                "inputs": [c["name"] for c in self.schema["input"]]}

    def _answer_frame(self, table: dict, frame, history) -> None:
        """Every row of `frame` answered (with `history` before it, for a window) into the
        table, under the key the server looks it up by."""
        if frame is None or not len(frame):
            return
        out = self.predict(frame, history)
        answers = self.answers(out)
        rows = self.json_rows(frame)
        if self.world.shape in ("row", "node"):
            for row, answer in zip(rows, answers):
                table["rows"][row_key(row)] = answer
        elif self.world.shape == "window":
            keys = self._keys()
            hist = self.json_rows(history) if history is not None else None
            for (row, context), answer in zip(window_context(rows, hist, keys["series_by"], keys["need"]), answers):
                table["windows"][rows_key([*context, row])] = answer
        else:
            gid = self.world.graph_id
            groups = {}
            for row, answer in zip(rows, answers):
                groups.setdefault(str(row[gid]), ([], []))
                groups[str(row[gid])][0].append(row)
                groups[str(row[gid])][1].append(answer)
            for members, said in groups.values():
                table["graphs"][rows_key(members)] = said

    def attribution(self, examples: dict) -> dict | None:
        import attribution_build as ab
        demo = examples.get("demo")
        if not demo:
            return None
        plan = ab.AttributionPlan.of(demo, self.manifest["output"], self.manifest["sample"])
        return ab.AttributionBuilder(plan, self.inspector.attribute, self.frame, self.manifest["output"]).build()

    def reference_scores(self, examples: dict) -> dict | None:
        import attribution_build as ab
        demo = examples.get("demo")
        if not demo:
            return None
        return ab.reference_scores(demo, self.manifest["output"], self.manifest["sample"],
                                   lambda rows, history: self.predict(rows, history), self.frame)


def _flag(value) -> bool:
    try:
        return value is not None and float(value) != 0.0
    except (TypeError, ValueError):
        return str(value).lower() == "true"


def random_class(names, raw, rng):
    return raw.get(names[int(rng.integers(0, len(names)))])


TOKENS = ("Infinity", "-Infinity", "+Infinity", "NaN")


def _cell(value, dtype):
    """A value as the page's editor reads it back into a request (views/shapes/row.js cellOf of
    the field's text): empty is null, a non-finite token stays a token, a text column's value is
    text, a number stays a number (the request key reads an integral number and its float as
    one cell), a mistyped number stays text."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str) and value.strip() in TOKENS:
        return value.strip()
    if dtype == "string":
        return str(value).strip() if not isinstance(value, float) or not value.is_integer() else str(int(value))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).strip())
    except ValueError:
        return str(value).strip()


class _Answering:
    """The Predictor demo_pack asks: the harness, over the frame as the request carries it."""

    def __init__(self, fx: FixtureCase):
        self.fx = fx

    def predict(self, frame, history=None):
        return self.fx.predict(frame, history)


def _digest(doc) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ============================================================================ the showcase side

def health(fx: FixtureCase, network_digest: str) -> dict:
    return {"ready": True, "model_version": fx.manifest["model_version"], "contract_version": fx.contract.version,
            "feature_contract_version": None, "self_test": {"passed": True, "source": "page fixture"},
            "started_at": "2026-10-01T00:00:00Z", "error": None, "max_inflight": 4,
            "inspection": {"enabled": True, "ready": True, "why": None, "weights": "full",
                           "network_digest": network_digest, "pinned": True},
            "showcase": {"verified": True, "reason": None}}


def info(fx: FixtureCase, serve: dict, network_digest: str) -> dict:
    return {"schema": "vega/bundle@2", "slug": fx.case["id"], "run": "R1", "package": "fixture_pkg",
            "model_version": fx.manifest["model_version"], "output": fx.manifest["output"],
            "sample": fx.manifest["sample"], "input_schema": "schema.json",
            "roundtrip": fx.manifest["roundtrip"], "serve": serve, "api_version": API_VERSION,
            "contract_version": fx.contract.version, "feature_contract_version": None,
            "inspection": {"enabled": True, "ready": True, "why": None, "weights": "full",
                           "network_digest": network_digest, "pinned": True},
            "showcase": {"verified": True, "reason": None}}


def page_documents(fx: FixtureCase, examples: dict, serve: dict, manifest_digest: str, sizes: dict,
                   reference_scores):
    """(story.json, page.json): the fixed authoring's story, then page.json by the builder."""
    import page_fixture_story as pfs
    from page_config import build_page_config, build_walk, check_page_config
    from page_derive import lane_record, modality_of
    import page_template
    from vega_show_core import ShowPaths, write_json
    catalogs = fx.catalogs
    lane = lane_record(fx.spec, catalogs)
    from page_config import metric_record
    metric = metric_record(fx.spec, catalogs)
    walk = build_walk(fx.spec, catalogs.explain)
    story = pfs.story_of(fx.case, fx.spec, lane, metric, walk, fx.manifest["output"], sizes)
    story["bundle"]["manifest_digest"] = manifest_digest
    analysis = {"context": {"spatial": {"crs": "EPSG:4326"} if pfs.spatial(fx.case) else None}}
    template = catalogs.template or page_template.digests()
    plan = pfs.plan_of(fx.case, modality_of(lane["shape"], analysis), template)
    with tempfile.TemporaryDirectory(prefix="page-fixture-") as tmp:
        paths = ShowPaths.of(Path(tmp))
        write_json(paths.plan, plan)
        records = {"manifest": {"schema": "vega/bundle@2", "files": {}, "serve": serve},
                   "manifest_digest": manifest_digest, "shape": pfs.shape_record(fx.spec),
                   "analysis": analysis, "examples": examples, "reference_scores": reference_scores}
        page = build_page_config(paths, fx.spec, records, story, catalogs)
    return story, page, check_page_config


def build_case(case: dict, out: Path, zoo, contract_mod, catalogs) -> dict:
    """Builds one case into out/<id>/ and returns its index entry."""
    fx = FixtureCase(case, zoo, contract_mod, catalogs)
    fx.label_train()
    examples, demo_notes = fx.examples()
    requests = fx.page_requests(examples)
    answers = fx.answer_table(examples, requests)
    network_bytes = fx.inspector.network_bytes("full")
    network = json.loads(network_bytes)
    network_digest = "sha256:" + hashlib.sha256(network_bytes).hexdigest()
    serve = {"present": True, "mode": "online", "endpoints": endpoints(),
             "identity": {"title": f"page fixture {case['n']}", "description": "A network of the page fixture matrix."}}
    manifest_digest = _digest({"case": case["id"], "model": fx.manifest["model_version"]})
    sizes = {"source": len(fx.world.train) + len(fx.world.test), "train": len(fx.world.train), "valid": 0,
             "test": len(fx.world.test)}
    attribution = fx.attribution(examples)
    reference = fx.reference_scores(examples)
    story, page, check = page_documents(fx, examples, serve, manifest_digest, sizes, reference)
    problems = check(page, network, catalogs)
    if problems:
        raise FixtureError(f"case {case['id']}: page.json fails BD2: {problems}")
    lane = page["lane"]
    want = case.get("expect") or {}
    differ = {k: (lane.get(k), v) for k, v in want.items() if lane.get(k) != v}
    if differ:
        raise FixtureError(f"case {case['id']}: the lane is not the matrix's: {differ} (lane, expected)")
    docs = {"netspec.json": fx.spec, "health.json": health(fx, network_digest),
            "info.json": info(fx, serve, network_digest), "schema.json": fx.contract.describe(),
            "examples.json": examples, "network.json": network, "answers.json": answers,
            "story.json": story, "page.json": page}
    if attribution is not None:
        docs["attribution.json"] = attribution
    monitor = monitor_document(fx, case)
    if monitor is not None:
        docs[MONITOR_FILE] = monitor
    where = out / case["id"]
    where.mkdir(parents=True, exist_ok=True)
    for stale in [*where.glob("*.json"), *where.glob("briefs/*")]:
        if stale.name not in docs and stale.name != "case.json":
            stale.unlink()
    for stage in case.get("briefs") or ():
        (where / "briefs").mkdir(exist_ok=True)
        (where / "briefs" / f"{stage}-brief.md").write_text(BRIEF_TEXT.format(stage=stage), encoding="utf-8")
    entry = {"n": case["n"], "id": case["id"], "family": lane["family"], "task": lane["task"],
             "base": lane["base"], "view_key": lane["view_key"], "link": lane["link"], "shape": lane["shape"],
             "level": lane["level"], "motif": page["motif"]["id"], "reading": fx.reading,
             "covers": case.get("covers") or [],
             "counts": {"presets": len(examples["presets"]), "demo_items": len((examples.get("demo") or {}).get("items") or []),
                        "test_rows": len(examples["test"]["rows"]), "explain": len(answers["explain"]),
                        "answers": len(answers["rows"]) + len(answers["windows"]) + len(answers["graphs"]),
                        "nodes": len(network.get("nodes") or [])},
             "attribution": attribution is not None, "monitor": monitor is not None,
             "briefs": list(case.get("briefs") or ()), "demo_notes": demo_notes["notes"]}
    docs["case.json"] = {"schema": "vega/page-fixture-case@1", **entry}
    for name, doc in docs.items():
        (where / name).write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                             separators=(",", ":")) + "\n", encoding="utf-8")
    return entry


def write_index(out: Path) -> dict:
    entries = sorted((json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*/case.json")),
                     key=lambda e: e["n"])
    doc = {"schema": INDEX_SCHEMA, "source": "vega-show/scripts/tests/make_page_fixtures.py",
           "matrix": "vega-show/scripts/tests/page-matrix.json",
           "cases": [{k: v for k, v in e.items() if k != "schema"} for e in entries]}
    (out / "index.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


# ============================================================================ the code view

def code_view(spec: dict, work: Path) -> dict:
    """GET /v1/code of a bundle carrying this network (vega/code-view@2): the forge's package
    rendered in full from the live templates, the serving package in its Predictor module, the
    composition root and the route modules, each file with its role and kind (the rule of
    `serve.routes_meta.CodeView`), as test_code_room renders a bundle's code."""
    import render_templates as rt
    pipeline = rt.reference_pipeline()
    emit = rt.load_emit()
    plan = emit.plan(spec, pipeline)
    mapping = emit.common_mapping(spec, pipeline, plan)
    for block in spec.get("blocks") or ():           # a custom block's module, under the campaign root
        ref = (block.get("params") or {}).get("code_ref") if block.get("type") == "custom" else None
        if ref in CUSTOM_SOURCES:
            (work / ref).parent.mkdir(parents=True, exist_ok=True)
            (work / ref).write_text(CUSTOM_SOURCES[ref], encoding="utf-8")
    fill = lambda text: re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: str(mapping.get(m.group(1), m.group(1).lower())), text)  # noqa: E731
    serving = TEMPLATES / "serving" / "package"
    used = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", "".join(p.read_text(encoding="utf-8") for p in serving.rglob("*.py"))))
    rt.render(TEMPLATES, work, pipeline, spec, full=True, campaign_root=work)
    rt.render_partial(serving, work, pipeline, spec, rt.PartialOptions(
        into="Predictor", recursive=True, defines={k: k.lower() for k in used if k not in mapping}))
    package = work / plan["package"]
    entries = {e["file"]: e for e in json.loads((package / "blueprint-map.json").read_text(encoding="utf-8"))["entries"]
               if e.get("file")}
    texts = {f"{plan['package']}/{p.relative_to(package).as_posix()}": p.read_text(encoding="utf-8")
             for p in sorted(package.rglob("*.py")) if p.name != "__init__.py" and "__pycache__" not in p.parts}
    texts["predict.py"] = fill((TEMPLATES / "serving" / "bundle" / "predict.py").read_text(encoding="utf-8"))
    for p in sorted(SERVE.glob("*.py")):
        if p.name != "__init__.py":
            texts[f"serve/{p.name}"] = fill(p.read_text(encoding="utf-8"))
    roles = json.loads((PACK / "catalog.json").read_text(encoding="utf-8"))["code_view"]["roles"]
    files = []
    for rel, text in texts.items():
        role, kind = _code_role(rel, entries, roles)
        files.append({"path": rel, "role": role, "kind": kind,
                      "digest": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                      "lines": len(text.splitlines()), "text": text})
    order = ("Model", "FeatureTransformer", "Predictor", "ModelStore", "Monitor", "composition_root", "domain",
             "ports", "support", "Inspection", "http_entry")
    files.sort(key=lambda f: (order.index(f["role"]) if f["role"] in order else len(order), f["path"]))
    return {"schema": "vega/code-view@2", "verified": True, "language": "python", "files": files}


def _code_role(rel: str, entries: dict, roles: list) -> tuple[str, str]:
    if rel == "predict.py":
        return "composition_root", "entrypoint"
    if rel.startswith("serve/"):
        return "http_entry", "entrypoint"
    name = rel.split("/", 1)[1]
    if name in ("ports.py", "domain.py"):
        return name[:-3], "support"
    role = (entries.get(name) or {}).get("role")
    if role in roles and role in ("Model", "FeatureTransformer", "Predictor", "ModelStore", "Monitor"):
        return role, "adapter"
    if "/inspection/" in rel:
        return "Inspection", "support"
    return "support", "support"


# ============================================================================ the check

VOLATILE = ("model_version", "manifest_digest", "network_digest", "contract_version", "digest")


def _same(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'the earlier' if k in a else 'the fresh'} build")
            else:
                out += _same(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items, {len(b)} fresh"]
        return [p for x, y in zip(a, b) for p in _same(x, y, path + "[]")][:20]
    if isinstance(a, float) or isinstance(b, float):
        try:
            if abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)), abs(float(b))):
                return []
        except (TypeError, ValueError):
            pass
    return [] if a == b else [f"{path}: {str(a)[:60]} != {str(b)[:60]}"]


def check(cases: list[dict], earlier: Path, built: Path) -> list[str]:
    out = []
    for case in cases:
        a, b = earlier / case["id"], built / case["id"]
        names = sorted({p.name for p in a.glob("*.json")} | {p.name for p in b.glob("*.json")})
        for name in names:
            if not (a / name).is_file() or not (b / name).is_file():
                out.append(f"{case['id']}/{name}: missing on one side")
                continue
            diffs = _same(json.loads((a / name).read_text(encoding="utf-8")),
                          json.loads((b / name).read_text(encoding="utf-8")), "")
            out += [f"{case['id']}/{name}{d}" for d in diffs[:5]]
    return out


# ============================================================================ main

def load_zoo():
    """The inspection zoo's module (its package renders on first use)."""
    import os
    os.environ.setdefault("VEGA_ZOO_CASES", "")
    return importlib.import_module("test_inspect_zoo")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="comma list of case ids, numbers or globs (default: every case)")
    ap.add_argument("--out", help="where the cases go (default: a fresh temporary directory)")
    ap.add_argument("--check", metavar="EARLIER", help="rebuild in a scratch directory and compare with an earlier build")
    ap.add_argument("--json", action="store_true", help="print the index as JSON")
    a = ap.parse_args(argv)
    try:
        cases = selected(load_matrix(), a.cases)
    except FixtureError as exc:
        print(f"ERROR: {exc}")
        return 2
    import torch
    torch.set_num_threads(2)
    from page_derive import Catalogs
    zoo = load_zoo()
    catalogs = Catalogs.load()
    work = Path(tempfile.mkdtemp(prefix="page-fixtures-"))
    contract_mod = serve_package(work)
    out = Path(a.check) if a.check else Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="page-fixtures-"))
    target = work / "built" if a.check else out
    print(f"OK: building {len(cases)} case(s) into {target}")
    target.mkdir(parents=True, exist_ok=True)
    code = 0
    for case in cases:
        try:
            entry = build_case(case, target, zoo, contract_mod, catalogs)
        except Exception as exc:  # a case that cannot be built is named, the others go on  # noqa: BLE001
            print(f"ERROR: case {case['id']}: {type(exc).__name__}: {exc}")
            code = 2
            continue
        c = entry["counts"]
        print(f"OK: {case['id']}: {entry['view_key']}/{entry['shape']}"
              f"{'/' + entry['level'] if entry['level'] else ''}, motif {entry['motif']}; {c['nodes']} nodes, "
              f"{c['presets']} presets, {c['demo_items']} demo items, {c['explain']} explanations, "
              f"{c['answers']} answers")
    if a.check:
        diffs = check(cases, out, target)
        for d in diffs:
            print(f"FAIL: {d}")
        if diffs:
            return 1 if code == 0 else code
        print(f"OK: {len(cases)} case(s) as built in {out}")
        return code
    index = write_index(out)
    if a.json:
        print(json.dumps(index, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
