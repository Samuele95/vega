#!/usr/bin/env python3
"""
test_catalog_sync.py — the machine copies of vega-forge's contracts agree with their prose, and
fail loudly the day they drift.

    python3 test_catalog_sync.py [-v]          (each `test_*` function also runs under pytest)

`scripts/catalog.json` is the machine copy of `family-catalog.md` — the contract says so in as
many words: "the machine copy of the catalog's §2-§4 that the build keeps in sync with the prose
(a test fails when they differ)". Without this test the prose and the compiler drift apart
silently, and a `vega:default` ends up citing a row a reader cannot find — which is the whole
failure mode the `catalog` key exists to prevent. v2.3 widened what is mirrored, and every
mirror is checked here:

- `family-catalog.md` §1-§4 (task kinds, families, blocks, defaults) against the catalog's rows,
  §5's fork points against `fork_order`, §6's metric directions against `metric_direction` (and
  that table against the backend's `METRICS`), §7's admission table against `admission`;
- `inspect-contract.md` §3 (the view table), §4 (targets, target kinds, links) and §7 (limits)
  against the catalog's `inspection` section;
- `backend-contract.md` §2.1 (the block inspection surface) and the catalog against the pytorch
  backend's `coverage.json` (`sample_shapes`, `graph_batching`, `recurrent_returns`, `metrics`,
  `inspection`);
- the two schema files `references/schemas/{network-view-2,explanation-2}.schema.json`: valid
  JSON Schema draft 2020-12, their closed sets equal to the catalog's, and a minimal payload of
  each shape accepted (and a wrong one refused) so the schemas are known to be satisfiable.

What is compared, and what is deliberately not: **row sets** are compared exactly (every task
kind, family, block type, default, view type, target kind, link, metric, fork point and
admission row in the prose is in the JSON and the other way round), because a missing row is
what breaks a citation. Identifier columns (a view kind, a stage kind, a link, a decision rule,
a default key, a direction, a limit) are compared cell by cell, and so are the machine strings a
page shows verbatim (the formulas). Prose columns (`when`, `because`, notes) are not diffed word
for word; what is checked is that each JSON row carries the fields the prose's columns promise.

Exit 0 when everything agrees.
"""
from __future__ import annotations

import ast
import functools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILL = SCRIPTS.parent
REFS = SKILL / "references"
CATALOG_JSON = SCRIPTS / "catalog.json"
CATALOG_MD = REFS / "family-catalog.md"
INSPECT_MD = REFS / "inspect-contract.md"
BACKEND_MD = REFS / "backend-contract.md"
NETSPEC_MD = REFS / "netspec-contract.md"
SCHEMAS = REFS / "schemas"
COVERAGE_JSON = SCRIPTS / "backends" / "pytorch" / "coverage.json"
METRICS_PY = SCRIPTS / "backends" / "pytorch" / "templates" / "metrics.py"
VERBOSE = "-v" in sys.argv

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
#: R-C1: the one link vocabulary of the chain
LINKS = ("softmax", "sigmoid", "sigmoid_each", "identity", "cumulative", "reconstruction",
         "distance", "horizon")
#: the four defaults rows v2.3 adds for the window and graph samples
NEW_DEFAULTS = ("window_context", "window_series", "autoregressive", "graph_batching")
#: the forge fork points the sample shapes raise, each with its unattended answer (R-G3)
FORK_POINTS = ("context", "autoregressive", "series_by", "protocol", "batching")
SYNTHETIC = ("input", "output", "residual_add", "parallel_merge", "encoder_layer")
#: the stage kind a top-level block of these types gets; every other block type is `tensor`
STAGE_KIND = {"embedding": "emb", "concat": "concat", "feature_tokenizer": "tokens",
              "gcn_conv": "graph", "sage_conv": "graph", "gat_conv": "graph",
              "gin_conv": "graph", "edge_dropout": "gdrop", "graph_pool": "gpool"}
#: the serving reading of each (effective task, loss flavour), to check the link table
READING_OF = {"binary": "binary", "multiclass": "multiclass", "multilabel": "multilabel",
              "regression": "regression", "regression/quantile": "quantile", "ordinal": "ordinal",
              "unsupervised": "reconstruction",
              "anomaly": "anomaly-reconstruction", "anomaly/svdd": "anomaly-svdd",
              "forecast": "forecast"}
DASH = ("—", "-", "")


# --------------------------------------------------------------------------- plumbing

class Check:
    """One named group of checks: failures are printed as they happen and raised at the end,
    so the CLI reports every divergence and pytest sees one failing test per group."""

    def __init__(self, name):
        self.name = name
        self.errors = []

    def fail(self, msg):
        self.errors.append(msg)
        print(f"  FAIL  {msg}")

    def ok(self, msg):
        if VERBOSE:
            print(f"  ok    {msg}")

    def expect(self, cond, msg):
        if not cond:
            self.fail(msg)
        return bool(cond)

    def close(self):
        if self.errors:
            raise AssertionError(f"{self.name}: {len(self.errors)} divergence(s)\n  "
                                 + "\n  ".join(self.errors))


@functools.lru_cache(maxsize=None)
def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _text(path):
    return Path(path).read_text(encoding="utf-8")


def catalog():
    return _json(str(CATALOG_JSON))


def coverage():
    return _json(str(COVERAGE_JSON))


def section(text, heading, level=2):
    """The body of one `## N. Heading` section (or `### N.M Heading` at level 3)."""
    hashes = "#" * level
    match = re.search(rf"^{hashes}\s*{re.escape(heading)}\s*$", text, re.M)
    if not match:
        return ""
    tail = text[match.end():]
    nxt = re.search(rf"^#{{1,{level}}}\s", tail, re.M)
    return tail[:nxt.start()] if nxt else tail


# the header cell of each §1-§4 table, which is not a row
HEADERS = {"task.kind", "family", "type", "decision", "item", "when"}


def first_column(body):
    """The first cell of every table row in a markdown section, backticks stripped and the
    header row dropped."""
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cell = line.strip("|").split("|")[0].strip()
        if not cell:
            continue
        for name in re.findall(r"`([^`]+)`", cell) or [cell]:
            name = name.strip()
            if name and name.lower() not in HEADERS:
                out.append(name)
    return out


def split_alternatives(names):
    """`gru` / `lstm` and `batchnorm` / `layernorm` are one prose row for two types."""
    out = []
    for name in names:
        out.extend(part.strip() for part in re.split(r"\s*/\s*", name) if part.strip())
    return out


_SEPARATOR = re.compile(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def table_rows(body):
    """The body rows of every markdown table in `body`: the header row (the one right above a
    separator line) and the separator are dropped, cells are split on unescaped pipes and an
    escaped `\\|` inside a cell reads back as `|`."""
    lines = [line.strip() for line in body.splitlines()]
    rows = []
    for i, line in enumerate(lines):
        if not line.startswith("|") or _SEPARATOR.fullmatch(line):
            continue
        if i + 1 < len(lines) and _SEPARATOR.fullmatch(lines[i + 1]):
            continue
        cells = _CELL_SPLIT.split(line.strip().strip("|"))
        rows.append([cell.strip().replace("\\|", "|") for cell in cells])
    return rows


def ticks(cell):
    """The backticked tokens of a cell, in order."""
    return re.findall(r"`([^`]+)`", cell)


def keyed_rows(body, check, what):
    """`{first backticked token of the first cell: cells}` for every table row of `body`."""
    out = {}
    for cells in table_rows(body):
        names = ticks(cells[0])
        if not names:
            check.fail(f"{what}: a table row has no backticked name in its first cell: "
                       f"{cells[0]!r}")
            continue
        out[names[0]] = cells
    return out


def metrics_py_names():
    """The keys of the backend's `METRICS` dict literal, read with `ast` (no torch import)."""
    tree = ast.parse(METRICS_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "METRICS" for t in targets) and \
                isinstance(node.value, ast.Dict):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    return []


# --------------------------------------------------------------------------- family-catalog §1-§4

def test_tasks():
    """§1: the task kinds, and the columns the prose promises on every row."""
    c = Check("§1 task kinds")
    md, data = _text(str(CATALOG_MD)), catalog()
    prose_tasks = set(split_alternatives(first_column(section(md, "1. Task kinds"))))
    prose_tasks = {t for t in prose_tasks if not t.startswith("sequence-*")}
    json_tasks = set(data["tasks"])
    # the prose writes the three sequence rows as one cell
    prose_tasks |= {"sequence-classification", "sequence-regression", "forecast"}
    prose_tasks -= {"sequence-classification / sequence-regression / forecast"}
    for missing in sorted(prose_tasks - json_tasks):
        c.fail(f"task kind {missing!r} is in the prose's §1 but not in catalog.json")
    for extra in sorted(json_tasks - prose_tasks):
        c.fail(f"task kind {extra!r} is in catalog.json but not in the prose's §1")
    if prose_tasks == json_tasks:
        c.ok(f"§1: {len(json_tasks)} task kinds agree")
    for kind, row in data["tasks"].items():
        for field in ("head", "default_loss", "primary"):
            if field not in row:
                c.fail(f"tasks/{kind} carries no {field!r}; §1's columns promise head, default "
                       "loss and primary metric")
        if "activation" not in (row.get("head") or {}):
            c.fail(f"tasks/{kind}.head carries no activation; the compiler checks it")
    c.close()


def test_task_rows_v23():
    """§1 as v2.3 states it: recon_dim, the rank-consistent ordinal head, the unlabelled anomaly
    primary, the delimited multilabel column, and the task fields every reader agrees on."""
    c = Check("§1 v2.3 task rows")
    data = catalog()
    tasks = data["tasks"]
    c.expect(tasks["unsupervised"]["head"]["out"] == "recon_dim",
             "tasks/unsupervised.head.out must be recon_dim (numeric columns + the embedded "
             "nominals' cardinalities), not input_dim")
    c.expect("recon_dim" in tasks["unsupervised"],
             "tasks/unsupervised must define recon_dim")
    c.expect(tasks["anomaly"]["head"]["out"] == "recon_dim|z",
             "tasks/anomaly.head.out must be recon_dim|z")
    c.expect((tasks["anomaly"].get("primary_alt") or {}).get("metric") == "anomaly_score_mean",
             "tasks/anomaly.primary_alt must be anomaly_score_mean (the unlabelled primary)")
    c.expect((tasks["ordinal"]["head"] or {}).get("form") == "rank-consistent",
             "tasks/ordinal.head.form must say rank-consistent")
    fmt = ((tasks["multilabel"].get("label") or {}).get("label_format") or {})
    c.expect(fmt.get("kind") == "delimited" and fmt.get("sep"),
             "tasks/multilabel.label must be one delimited column (label_format delimited)")
    fields = data.get("task_fields") or {}
    for name in ("base_kind", "classes", "label_format", "positive", "anomaly_rate"):
        c.expect(name in fields, f"task_fields has no {name!r} (netspec-contract §1)")
    c.expect(set((fields.get("anomaly_rate") or {}).get("applies_to") or []) == {"anomaly"},
             "task_fields.anomaly_rate must apply to the anomaly task only (PLAN-AMENDMENTS A10)")
    c.expect("`anomaly_rate`" in _text(str(NETSPEC_MD)),
             "netspec-contract §1 must name task.anomaly_rate (PLAN-AMENDMENTS A10)")
    for kind in (fields.get("base_kind") or {}).get("required_when") or []:
        c.expect(kind in tasks and kind.startswith("sequence-"),
                 f"task_fields.base_kind.required_when names {kind!r}, not a sequence kind")
    for kind in (fields.get("classes") or {}).get("required_when") or []:
        c.expect(kind in tasks, f"task_fields.classes.required_when names {kind!r}")
    for kind in ("sequence-classification", "sequence-regression"):
        c.expect(tasks[kind].get("base_field") == "task.base_kind",
                 f"tasks/{kind}.base_field must be task.base_kind")
    c.close()


def test_families():
    """§2: the families, the columns the prose promises, and block types §3 defines."""
    c = Check("§2 families")
    md, data = _text(str(CATALOG_MD)), catalog()
    prose_families = set(split_alternatives(first_column(section(md, "2. Families"))))
    json_families = set(data["families"])
    for missing in sorted(prose_families - json_families):
        c.fail(f"family {missing!r} is in the prose's §2 but not in catalog.json")
    for extra in sorted(json_families - prose_families):
        c.fail(f"family {extra!r} is in catalog.json but not in the prose's §2")
    if prose_families == json_families:
        c.ok(f"§2: {len(json_families)} families agree")
    for name, row in data["families"].items():
        for field in ("when", "composition", "block_types"):
            if field not in row:
                c.fail(f"families/{name} carries no {field!r}; §2's columns promise when it "
                       "applies and what it is made of")
        unknown = [b for b in row.get("block_types") or [] if b not in data["blocks"]]
        if unknown:
            c.fail(f"families/{name} names block type(s) {unknown} that §3 does not define")
    c.close()


def test_compositions():
    """R-A13 and the floors: the window families can embed a nominal, the graph families can
    embed and drop edges, linear is the floor on every shape, rnn floors every recurrent
    family, and the reconstruction families reconstruct recon_dim."""
    c = Check("§2 compositions")
    data = catalog()
    fam, shapes = data["families"], data["sample_shapes"]
    for name, row in fam.items():
        c.expect(row.get("sample_shape") in shapes,
                 f"families/{name}.sample_shape {row.get('sample_shape')!r} is not a sample "
                 "shape")
        for s in row.get("floor_shapes") or []:
            c.expect(s in shapes, f"families/{name}.floor_shapes names {s!r}")
        bt = row["block_types"]
        if row.get("sample_shape") == "window":
            ok = "embedding" in bt and "concat" in bt and \
                bt.index("embedding") < bt.index("concat")
            c.expect(ok, f"window family {name!r} must list embedding then concat (a window "
                         "nominal is (B, L, C) ids that embed and join)")
        if row.get("sample_shape") == "graph":
            missing = {"embedding", "concat", "edge_dropout"} - set(bt)
            c.expect(not missing, f"graph family {name!r} lacks {sorted(missing)}")
    lin = fam["linear"]
    c.expect({"flatten", "graph_pool"} <= set(lin["block_types"]),
             "families/linear must list flatten and graph_pool (the floor on windows and "
             "at graph level)")
    c.expect(lin.get("floor_shapes") == shapes,
             f"families/linear.floor_shapes must be {shapes}")
    floor_for = fam["rnn"].get("ablation_floor_for") or []
    for name in ("gru", "lstm", "bi-lstm", "bi-gru", "lstm-attention", "gru-attention"):
        c.expect(name in floor_for, f"families/rnn.ablation_floor_for must name {name!r}")
    for name in floor_for:
        row = fam.get(name) or {}
        c.expect(row.get("sample_shape") == "window" and
                 {"gru", "lstm"} & set(row.get("block_types") or []),
                 f"rnn floors {name!r}, which is not a recurrent window family")
    for name in ("autoencoder", "vae", "pca-floor"):
        c.expect("recon_dim" in fam[name]["composition"],
                 f"families/{name} must reconstruct recon_dim")
        c.expect({"embedding", "concat"} <= set(fam[name]["block_types"]),
                 f"families/{name} must embed its nominals")
    c.expect(fam["deep-svdd"]["defaults"].get("head_bias") is False,
             "families/deep-svdd must default the head's bias off")
    c.expect("embedding" not in fam["ft-transformer"]["block_types"],
             "ft-transformer's tokenizer reads the nominals itself; no embedding block")
    for name in ("gru", "lstm"):
        dfl = fam[name]["defaults"]
        c.expect("num_layers" in dfl and "layers" not in dfl,
                 f"families/{name}.defaults must use num_layers (the catalog's param name)")
    c.close()


def test_blocks():
    """§3: the block types, and a params and shape rule on every row."""
    c = Check("§3 blocks")
    md, data = _text(str(CATALOG_MD)), catalog()
    prose_blocks = set(split_alternatives(first_column(section(md, "3. Blocks"))))
    prose_blocks = {b for b in prose_blocks if "(" not in b}
    json_blocks = set(data["blocks"])
    for missing in sorted(prose_blocks - json_blocks):
        c.fail(f"block type {missing!r} is in the prose's §3 but not in catalog.json")
    for extra in sorted(json_blocks - prose_blocks):
        c.fail(f"block type {extra!r} is in catalog.json but not in the prose's §3")
    if prose_blocks == json_blocks:
        c.ok(f"§3: {len(json_blocks)} block types agree")
    for name, row in data["blocks"].items():
        for field in ("params", "shape_rule"):
            if field not in row:
                c.fail(f"blocks/{name} carries no {field!r}; §3's columns promise params and a "
                       "shape rule for every type")
        c.expect("backend_status" not in row,
                 f"blocks/{name} carries backend_status; what a backend builds is its own "
                 "coverage.json, not the catalog")
    c.expect("bias" in data["blocks"]["head"]["params"], "blocks/head must take bias")
    c.expect("*Planned*" not in section(md, "3. Blocks"),
             "§3 still marks rows *Planned*; the backend's coverage.json says what is planned")
    c.close()


def test_defaults():
    """§4: the citable default rows, and default / because / alternative on each."""
    c = Check("§4 defaults")
    md, data = _text(str(CATALOG_MD)), catalog()
    tail = section(md, "4. Defaults a `vega:default` may cite")
    prose_defaults = set(re.findall(r"`defaults/([a-z_]+)`", tail))
    json_defaults = set(data["defaults"])
    for missing in sorted(prose_defaults - json_defaults):
        c.fail(f"default row {missing!r} is named in the prose's §4 but not in catalog.json")
    for extra in sorted(json_defaults - prose_defaults):
        c.fail(f"default row {extra!r} is in catalog.json but §4 does not name it as a "
               "citable key")
    if prose_defaults == json_defaults:
        c.ok(f"§4: {len(json_defaults)} default rows agree")
    for name, row in data["defaults"].items():
        for field in ("default", "because", "alternative"):
            if field not in row:
                c.fail(f"defaults/{name} carries no {field!r}; §4's columns promise the "
                       "default, why it is the default, and the first alternative")
    c.close()


def test_defaults_rows():
    """The four v2.3 rows (window context, window series, autoregressive, graph batching):
    present in both copies, with a real default, reason and alternative, and values the
    compiler and the backend can act on."""
    c = Check("§4 v2.3 defaults rows")
    md, data = _text(str(CATALOG_MD)), catalog()
    D = data["defaults"]
    tail = section(md, "4. Defaults a `vega:default` may cite")
    table_keys = {k for cells in table_rows(tail)
                  for k in re.findall(r"`defaults/([a-z_]+)`", cells[0])}
    for key in NEW_DEFAULTS:
        row = D.get(key)
        if not c.expect(isinstance(row, dict), f"defaults/{key} is missing from catalog.json"):
            continue
        for field in ("default", "because", "alternative"):
            c.expect(isinstance(row.get(field), str) and row[field].strip(),
                     f"defaults/{key}.{field} must be a non-empty sentence")
        c.expect(row.get("structural") is False,
                 f"defaults/{key} is a decision, so structural must be false")
        c.expect(key in table_keys, f"defaults/{key} has no row in §4's table")
    wc = D.get("window_context") or {}
    c.expect(wc.get("value") == "previous-part" and wc.get("values") == ["previous-part", "none"],
             "defaults/window_context: value previous-part, values [previous-part, none]")
    ws = D.get("window_series") or {}
    c.expect(isinstance(ws.get("rule"), list) and len(ws["rule"]) == 3 and ws["rule"][-1] is None,
             "defaults/window_series.rule: lag group_by, else split by, else null")
    c.expect(bool(ws.get("unattended_warn")),
             "defaults/window_series must carry the unattended WARN text")
    ar = D.get("autoregressive") or {}
    c.expect(ar.get("value") == {"forecast": True, "otherwise": False},
             "defaults/autoregressive: true for forecast, false otherwise")
    c.expect(set((ar.get("forbidden_on") or {}).get("tasks") or []) >=
             {"sequence-classification", "sequence-regression"},
             "defaults/autoregressive must be forbidden on the sequence-* tasks (W1)")
    gb = D.get("graph_batching") or {}
    rule = gb.get("rule") or {}
    c.expect(gb.get("values") == coverage().get("graph_batching"),
             "defaults/graph_batching.values must equal coverage.json graph_batching")
    c.expect(rule.get("level_graph") == "graphs", "graph_batching: graphs at level graph")
    full = rule.get("full_when") or {}
    c.expect(isinstance(full.get("max_nodes"), int) and isinstance(full.get("max_edges"), int),
             "graph_batching.full_when needs integer max_nodes and max_edges")
    otherwise = rule.get("otherwise") or {}
    c.expect(otherwise.get("batching") == "sampled" and otherwise.get("neighbours") ==
             data["families"]["graphsage"]["defaults"].get("neighbours"),
             "graph_batching.otherwise must sample with graphsage's neighbours")
    c.close()


def test_compiler_rows():
    """The rows the compiler treats specially exist, and every source pattern compiles."""
    c = Check("compiler rows")
    data = catalog()
    for key in data.get("structural_catalog_rows", []):
        section_name, _, row = key.partition("/")
        if row not in (data.get(section_name) or {}):
            c.fail(f"structural_catalog_rows names {key!r}, which is not a row")
    for key in data.get("runtime_default_rows", []):
        if key.split("/", 1)[1] not in data["defaults"]:
            c.fail(f"runtime_default_rows names {key!r}, which is not a defaults row")
    for name, pattern in (data.get("sources") or {}).get("patterns", {}).items():
        try:
            re.compile(pattern)
        except re.error as exc:
            c.fail(f"the {name!r} source pattern does not compile: {exc}")
    c.close()


# --------------------------------------------------------------------------- family-catalog §5-§7

def test_forks():
    """§5: `fork_order` follows the decision families, and every fork point a sample raises
    names its default row and its unattended answer (R-G3)."""
    c = Check("§5 forks")
    md, data = _text(str(CATALOG_MD)), catalog()
    fams = data["decision_families"]
    order = data.get("fork_order") or []
    c.expect([e.get("family") for e in order] == list(fams),
             f"fork_order must list the decision families in order {list(fams)}")
    points = {}
    for entry in order:
        fam = entry.get("family")
        c.expect(entry.get("choices") == fams.get(fam),
                 f"fork_order[{fam}].choices must equal decision_families[{fam}]")
        c.expect(bool(entry.get("unattended")),
                 f"fork_order[{fam}] carries no unattended answer")
        for p in entry.get("points") or []:
            if p.get("id") in points:
                c.fail(f"fork point {p.get('id')!r} is listed twice")
            points[p.get("id")] = (fam, p)
    c.expect(set(points) == set(FORK_POINTS),
             f"fork points {sorted(points)} != {sorted(FORK_POINTS)}")
    for pid, (fam, p) in points.items():
        row = str(p.get("default", "")).partition("/")[2]
        c.expect(str(p.get("default", "")).startswith("defaults/") and row in data["defaults"],
                 f"fork point {pid!r} cites {p.get('default')!r}, which is not a defaults row")
        for field in ("asks", "when", "unattended"):
            c.expect(bool(p.get(field)), f"fork point {pid!r} carries no {field!r}")
        c.expect(p.get("defaults_to_test") is (pid != "protocol"),
                 f"fork point {pid!r}: every such default is tested by a rung except protocol")
    body = section(md, "5. Decision families and the fork format")
    prose = keyed_rows(body, c, "§5")
    c.expect(set(prose) == set(points),
             f"§5's fork table names {sorted(prose)}, catalog.json {sorted(points)}")
    for pid, cells in prose.items():
        if pid not in points or len(cells) < 5:
            continue
        fam, p = points[pid]
        c.expect(cells[1] == fam, f"§5 {pid}: family {cells[1]!r} != {fam!r}")
        c.expect(p.get("default") in ticks(cells[3]),
                 f"§5 {pid}: the default column does not cite {p.get('default')!r}")
        c.expect(cells[4].strip() not in DASH, f"§5 {pid}: no --unattended answer")
    c.close()


def test_metric_direction():
    """§6 and R-C7: one direction per metric the backend implements (METRICS, read from the
    template), plus anomaly_score_mean and the loss; every task's primary has one."""
    c = Check("§6 metric directions")
    md, data = _text(str(CATALOG_MD)), catalog()
    table = data.get("metric_direction") or {}
    c.expect(set(table.values()) <= {"max", "min", "none"},
             f"metric_direction values must be max, min or none: {sorted(set(table.values()))}")
    implemented = metrics_py_names()
    c.expect(bool(implemented), f"no METRICS dict literal found in {METRICS_PY}")
    required = set(implemented) | {"anomaly_score_mean"}
    missing = sorted(required - set(table))
    c.expect(not missing, f"metric_direction has no direction for {missing}")
    missing_cov = sorted(set(coverage().get("metrics") or []) - set(table))
    c.expect(not missing_cov, f"coverage.json metrics {missing_cov} have no direction")
    c.expect(table.get("anomaly_score_mean") == "min", "anomaly_score_mean is mode min")
    c.expect(table.get("loss") == "min", "the training loss is mode min")
    for kind, row in data["tasks"].items():
        primaries = [row.get("primary"), (row.get("primary_alt") or {}).get("metric")]
        for name in [p for p in primaries if p and not str(p).startswith("as the base")]:
            c.expect(table.get(name) in ("max", "min"),
                     f"tasks/{kind}'s primary {name!r} needs a max or min direction")
        for name in row.get("secondary") or []:
            c.expect(name in table, f"tasks/{kind}'s secondary {name!r} has no direction")
    prose = keyed_rows(section(md, "6. Metric directions"), c, "§6")
    c.expect(set(prose) == set(table),
             f"§6 names {sorted(set(prose) ^ set(table))} on one side only")
    for name, cells in prose.items():
        if name in table and len(cells) > 1:
            c.expect(cells[1] == table[name],
                     f"§6 {name}: {cells[1]!r} != catalog.json {table[name]!r}")
    c.close()


def test_admission():
    """§7: the admission table — 127 cells, every family admitted somewhere, each row on a
    shape the family admits, the W2/W3 exclusions kept — equal in both copies."""
    c = Check("§7 admission")
    md, data = _text(str(CATALOG_MD)), catalog()
    adm = data.get("admission") or {}
    rows = adm.get("rows") or []
    fam, tasks, seq = data["families"], data["tasks"], data["sequence_map"]
    level_tasks = set(data["graph_level"]["tasks"])
    counts = [len(r["families"]) * len(r["tasks"]) + len(r["variants"]) for r in rows]
    c.expect(sum(counts) == adm.get("cells") == 127,
             f"admission cells {sum(counts)} (stated {adm.get('cells')}) != 127")
    seen = set()
    for r, n in zip(rows, counts):
        seen |= set(r["families"])
        shape, level = r["shape"], r["level"]
        c.expect((level is None) == (shape != "graph"),
                 f"admission row {r['families']}: a level exactly on graph rows")
        for f in r["families"]:
            row = fam.get(f) or {}
            c.expect(shape == row.get("sample_shape") or shape in (row.get("floor_shapes") or ()),
                     f"admission: {f!r} admitted on {shape!r}, which it does not take")
            forbidden = set((row.get("forbidden_on") or {}).get("tasks") or ())
            c.expect(not forbidden & {t.split("/")[0] for t in r["tasks"]},
                     f"admission: {f!r} admitted on {sorted(forbidden)}, which it forbids (W3)")
            if level == "node":
                c.expect(row.get("level") != "graph", f"admission: {f!r} is graph-level only")
        for t in r["tasks"]:
            kind, _, base = t.partition("/")
            c.expect(kind in tasks, f"admission task {t!r}: {kind!r} is not a task kind")
            if kind.startswith("sequence-"):
                c.expect(seq.get(base) == kind, f"admission task {t!r}: bad base")
            elif kind == "anomaly":
                c.expect(base in ("labelled", "unlabelled"), f"admission task {t!r}")
            if level == "graph":
                c.expect(kind in level_tasks, f"admission: {kind!r} at level graph")
        for v in r["variants"]:
            c.expect(v.get("family") in r["families"] and v.get("task") in r["tasks"],
                     f"admission variant {v} is not a cell of its row")
    c.expect(seen == set(fam), f"families never admitted: {sorted(set(fam) - seen)}")
    body = section(md, "7. Admission — the closed definition of complete coverage")
    prose = table_rows(body)
    c.expect(len(prose) == len(rows),
             f"§7 has {len(prose)} rows, catalog.json admission {len(rows)}")
    for cells, r, n in zip(prose, rows, counts):
        if len(cells) < 6:
            c.fail(f"§7 row {cells[0]!r} has fewer than six cells")
            continue
        c.expect(set(ticks(cells[0])) == set(r["families"]),
                 f"§7 row {cells[0]!r} != {r['families']}")
        c.expect(ticks(cells[1])[:1] == [r["shape"]], f"§7 row {cells[0]!r}: shape")
        c.expect((cells[2] in DASH) if r["level"] is None else cells[2] == r["level"],
                 f"§7 row {cells[0]!r}: level {cells[2]!r} != {r['level']!r}")
        c.expect(cells[5].strip().isdigit() and int(cells[5]) == n,
                 f"§7 row {cells[0]!r}: {cells[5]!r} cells, catalog.json gives {n}")
    c.expect("**127 cells**" in body, "§7 does not state the 127 cells")
    c.close()


# --------------------------------------------------------------------------- inspect-contract

def test_links():
    """R-C1: exactly the eight links, each reading served by exactly one link, decision rules
    from the closed set, and the prose's link table equal to the catalog's."""
    c = Check("inspection links")
    ins = catalog()["inspection"]
    links = ins.get("links") or {}
    c.expect(tuple(links) == LINKS, f"inspection.links {list(links)} != {list(LINKS)}")
    c.expect(len(links) == 8, f"{len(links)} links; R-C1 names 8")
    served = [r for row in links.values() for r in (row.get("readings") or {})]
    c.expect(sorted(served) == sorted(ins.get("readings") or []) and
             len(served) == len(set(served)),
             f"every reading must be served by exactly one link: {sorted(served)}")
    rules = set(ins.get("decision_rules") or [])
    for name, row in links.items():
        for reading, rule in (row.get("readings") or {}).items():
            c.expect(rule in rules, f"link {name}: reading {reading!r} decides by {rule!r}")
        c.expect(bool(row.get("formula")), f"link {name} has no formula")
    c.expect(set(ins.get("manifest_decision") or {}) == rules,
             "inspection.manifest_decision must map every decision rule")
    link_of = {r: name for name, row in links.items() for r in row.get("readings") or {}}
    rule_of = {r: rule for row in links.values() for r, rule in row.get("readings").items()}
    for kind, row in (ins.get("targets") or {}).items():
        flavours = [(kind, row)] + [(f"{kind}/{loss}", sub)
                                    for loss, sub in (row.get("by_loss") or {}).items()]
        for key, sub in flavours:
            if "as_base" in sub:
                continue
            reading = READING_OF.get(key)
            c.expect(sub.get("link") == link_of.get(reading),
                     f"targets/{key}: link {sub.get('link')!r}, the {reading!r} reading's is "
                     f"{link_of.get(reading)!r}")
            c.expect(sub.get("decision_rule") == rule_of.get(reading),
                     f"targets/{key}: decision {sub.get('decision_rule')!r} != "
                     f"{rule_of.get(reading)!r}")
    body = section(_text(str(INSPECT_MD)), "4.3 Links", level=3)
    prose = keyed_rows(body, c, "inspect-contract §4.3")
    c.expect(tuple(prose) == LINKS, f"§4.3 lists {list(prose)}")
    for name, cells in prose.items():
        if name not in links or len(cells) < 3:
            continue
        pairs = dict(re.findall(r"`([a-z-]+)` \(([a-z_]+)\)", cells[1]))
        c.expect(pairs == links[name]["readings"],
                 f"§4.3 {name}: readings {pairs} != {links[name]['readings']}")
        c.expect(ticks(cells[2])[:1] == [links[name]["formula"]],
                 f"§4.3 {name}: the formula differs from catalog.json")
    c.close()


def _check_view_row(c, t, row, ins):
    c.expect(row.get("view") in ins["view_kinds"], f"views/{t}: view {row.get('view')!r}")
    c.expect(row.get("call") in ins["calls"], f"views/{t}: call {row.get('call')!r}")
    want = ("io" if t in ("input", "output") else "inner") if t in SYNTHETIC \
        else STAGE_KIND.get(t, "tensor")
    c.expect(row.get("call") == want, f"views/{t}: call {row.get('call')!r}, want {want!r}")
    c.expect(bool(row.get("formula")), f"views/{t} has no formula")
    for field in ("payload", "probe", "exact"):
        c.expect(isinstance(row.get(field), list), f"views/{t}.{field} must be a list")
    c.expect(row.get("recompute") is None or isinstance(row.get("recompute"), str),
             f"views/{t}.recompute must be a check or null")


def _check_target_row(c, kind, row, ins, data):
    kinds = ins["target_kinds"]
    if "as_base" in row:
        bases = set(row["as_base"])
        want = {b for b, k in data["sequence_map"].items() if k == kind}
        c.expect(bases == want, f"targets/{kind}.as_base {sorted(bases)} != {sorted(want)}")
        for b in bases:
            c.expect("as_base" not in (ins["targets"].get(b) or {"as_base": 1}),
                     f"targets/{kind}: base {b!r} must have a row of its own")
        return
    for key, sub in [(kind, row)] + [(f"{kind}/{loss}", s)
                                     for loss, s in (row.get("by_loss") or {}).items()]:
        c.expect(sub.get("link") in ins["links"], f"targets/{key}: link {sub.get('link')!r}")
        c.expect(sub.get("decision_rule") in ins["decision_rules"],
                 f"targets/{key}: decision rule {sub.get('decision_rule')!r}")
        sel = sub.get("selectable") or []
        c.expect(sel and set(sel) <= set(kinds), f"targets/{key}: selectable {sel}")
        c.expect(sub.get("default") in sel,
                 f"targets/{key}: default {sub.get('default')!r} is not selectable")
    for loss in row.get("by_loss") or {}:
        c.expect(loss in data["tasks"][kind]["losses_allowed"],
                 f"targets/{kind}.by_loss names {loss!r}, not a loss of the task")


def _row_targets(row):
    flavours = [row] + list((row.get("by_loss") or {}).values())
    return ({f["link"] for f in flavours}, {f["decision_rule"] for f in flavours},
            {f["default"] for f in flavours}, {k for f in flavours for k in f["selectable"]})


def test_inspection_rows():
    """D3 §14.4: the view table (inspect-contract §3.1) equals `inspection.views`, whose keys
    are the catalog's blocks plus the five synthetic types; the target tables (§4.1, §4.2)
    equal `inspection.targets` (keyed by the task kinds) and `inspection.target_kinds`; the
    backend's readers cover every block it builds; the limits (§7) agree."""
    c = Check("inspection rows")
    data, cov = catalog(), coverage()
    ins = data["inspection"]
    md = _text(str(INSPECT_MD))
    views, targets = ins.get("views") or {}, ins.get("targets") or {}
    c.expect(tuple(ins.get("synthetic") or ()) == SYNTHETIC,
             f"inspection.synthetic must be {list(SYNTHETIC)}")
    c.expect(set(views) == set(data["blocks"]) | set(SYNTHETIC),
             f"inspection.views differs from blocks + synthetic on "
             f"{sorted(set(views) ^ (set(data['blocks']) | set(SYNTHETIC)))}")
    c.expect(set(targets) == set(data["tasks"]),
             f"inspection.targets differs from the task kinds on "
             f"{sorted(set(targets) ^ set(data['tasks']))}")
    for t, row in views.items():
        _check_view_row(c, t, row, ins)
    for kind, row in targets.items():
        _check_target_row(c, kind, row, ins, data)
    for name, row in (ins.get("target_kinds") or {}).items():
        c.expect(row.get("selector") in set(ins["selectors"]) | {None},
                 f"target_kinds/{name}: selector {row.get('selector')!r}")
        c.expect(bool(row.get("unit")) and bool(row.get("scalar")),
                 f"target_kinds/{name} needs a unit and a scalar")
    c.expect(len(ins.get("target_kinds") or {}) == 11, "D3 names 11 target kinds")

    # §3.1, cell by cell
    prose = keyed_rows(section(md, "3.1 The view table", level=3), c, "inspect-contract §3.1")
    c.expect(set(prose) == set(views),
             f"§3.1's view table differs from inspection.views on "
             f"{sorted(set(prose) ^ set(views))}")
    for t, cells in prose.items():
        row = views.get(t)
        if row is None or len(cells) < 8:
            continue
        c.expect(cells[1] == row["view"], f"§3.1 {t}: view {cells[1]!r} != {row['view']!r}")
        c.expect(cells[2] == row["call"], f"§3.1 {t}: call {cells[2]!r} != {row['call']!r}")
        for col, field in ((3, "payload"), (4, "probe"), (5, "exact")):
            c.expect(set(ticks(cells[col])) == set(row[field]),
                     f"§3.1 {t}: {field} {ticks(cells[col])} != {row[field]}")
        c.expect((cells[6] in DASH) == (row["recompute"] is None),
                 f"§3.1 {t}: recompute presence differs")
        c.expect(ticks(cells[7])[:1] == [row["formula"]],
                 f"§3.1 {t}: formula {ticks(cells[7])[:1]} != {row['formula']!r}")

    # §4.1 and §4.2
    prose = keyed_rows(section(md, "4.1 Targets per task", level=3), c, "inspect-contract §4.1")
    c.expect(set(prose) == set(targets),
             f"§4.1 differs from inspection.targets on {sorted(set(prose) ^ set(targets))}")
    for kind, cells in prose.items():
        row = targets.get(kind)
        if row is None or len(cells) < 5:
            continue
        if "as_base" in row:
            c.expect(set(ticks(cells[1])) & set(data["tasks"]) == set(row["as_base"]),
                     f"§4.1 {kind}: the bases differ")
            continue
        links, rules, defaults, selectable = _row_targets(row)
        c.expect(set(ticks(cells[1])) & set(ins["links"]) == links, f"§4.1 {kind}: link")
        c.expect(set(ticks(cells[2])) & set(ins["decision_rules"]) == rules,
                 f"§4.1 {kind}: decision rule")
        c.expect(ticks(cells[3])[:1] == [row["default"]] and defaults <= set(ticks(cells[3])),
                 f"§4.1 {kind}: default target")
        c.expect(set(ticks(cells[4])) & set(ins["target_kinds"]) == selectable,
                 f"§4.1 {kind}: selectable targets")
    body42 = section(md, "4.2 Target kinds and request selectors", level=3)
    prose = keyed_rows(body42, c, "inspect-contract §4.2")
    c.expect(set(prose) == set(ins["target_kinds"]),
             f"§4.2 differs from target_kinds on {sorted(set(prose) ^ set(ins['target_kinds']))}")
    for name, cells in prose.items():
        row = ins["target_kinds"].get(name)
        if row is None or len(cells) < 4:
            continue
        c.expect((ticks(cells[1])[:1] or [None])[0] == row["selector"],
                 f"§4.2 {name}: selector {cells[1]!r} != {row['selector']!r}")
        c.expect(cells[3] == row["unit"], f"§4.2 {name}: unit {cells[3]!r} != {row['unit']!r}")
    for sel in ins["selectors"]:
        c.expect(f"`{sel}`" in body42, f"§4.2 does not name the selector {sel!r}")

    # the readers cover what the backend builds
    ci = cov.get("inspection") or {}
    c.expect(set(ci.get("blocks") or ()) >= set(cov.get("blocks") or ()),
             "coverage.json inspection.blocks must include every block the backend builds: "
             f"{sorted(set(cov.get('blocks') or ()) - set(ci.get('blocks') or ()))}")
    c.expect(set(ci.get("blocks") or ()) == set(data["blocks"]),
             "coverage.json inspection.blocks must be the catalog's block types")
    c.expect(tuple(ci.get("synthetic") or ()) == SYNTHETIC,
             "coverage.json inspection.synthetic must be the five synthetic types")

    # §7 limits
    prose = keyed_rows(section(md, "7. Limits and gating"), c, "inspect-contract §7")
    limits = ins.get("limits") or {}
    prose = {k: v for k, v in prose.items() if k in limits or k.startswith(("inline", "json",
                                                                            "vector", "heat",
                                                                            "graph_nodes",
                                                                            "explain", "ig_"))}
    c.expect(set(prose) == set(limits),
             f"§7's limits differ from inspection.limits on {sorted(set(prose) ^ set(limits))}")
    for name, cells in prose.items():
        if name in limits:
            value = re.sub(r"[\s  ,]", "", cells[1])
            c.expect(value == str(limits[name]),
                     f"§7 {name}: {cells[1]!r} != catalog.json {limits[name]}")
    c.expect(limits.get("explain_exact_float_budget") == 4_194_304,
             "R-F5: explain_exact_float_budget is 4 194 304")
    c.expect((ins.get("numbers") or {}).get("significant_digits") == 9,
             "R-F8: JSON numbers carry 9 significant digits")
    c.close()


# --------------------------------------------------------------------------- coverage.json

def test_coverage_keys():
    """The pytorch coverage keys K1 owns: the sample shapes, graph batching modes, recurrent
    readouts and anomaly_score_mean, and the inspection surface equal to backend-contract
    §2.1."""
    c = Check("coverage.json")
    data, cov = catalog(), coverage()
    c.expect(cov.get("sample_shapes") == data["sample_shapes"],
             "coverage.json sample_shapes must equal the catalog's")
    c.expect(cov.get("graph_batching") == ["full", "sampled", "graphs"],
             "coverage.json graph_batching must be full, sampled, graphs")
    returns = set()
    for name in ("rnn", "gru", "lstm"):
        returns |= set(str(data["blocks"][name]["params"].get("return", "")).split("|"))
    c.expect(set(cov.get("recurrent_returns") or ()) == returns,
             f"coverage.json recurrent_returns must be the recurrent blocks' {sorted(returns)}")
    c.expect("anomaly_score_mean" in (cov.get("metrics") or ()),
             "coverage.json metrics must include anomaly_score_mean")
    built, planned = set(cov.get("blocks") or ()), set((cov.get("planned") or {}).get("blocks")
                                                       or ())
    c.expect(not built & planned, f"blocks both built and planned: {sorted(built & planned)}")
    c.expect((built | planned) <= set(data["blocks"]),
             f"coverage.json names blocks the catalog does not: "
             f"{sorted((built | planned) - set(data['blocks']))}")
    ci = cov.get("inspection") or {}
    surface = ci.get("surface") or {}
    c.expect(set(surface) == set(ci.get("blocks") or ()),
             "coverage.json inspection.surface must have one row per inspection block")
    ident = re.compile(r"^[a-z_][a-z0-9_]*$")
    for t, attrs in surface.items():
        c.expect(isinstance(attrs, list) and all(ident.match(a) for a in attrs),
                 f"surface[{t}] must be a list of attribute names")
    for key in ("surface_variants", "surface_optional"):
        for t in ci.get(key) or {}:
            c.expect(t in surface, f"coverage.json inspection.{key} names {t!r}")
    variant = (((ci.get("surface_variants") or {}).get("head") or {}).get("cumulative") or {})
    c.expect(variant.get("attributes") == ["score", "first", "gaps"],
             "the ordinal head's surface variant must be score, first, gaps")
    body = section(_text(str(BACKEND_MD)), "2.1 The block inspection surface", level=3)
    prose = keyed_rows(body, c, "backend-contract §2.1")
    c.expect(set(prose) == set(surface),
             f"backend-contract §2.1 differs from the surface on {sorted(set(prose) ^ set(surface))}")
    for t, cells in prose.items():
        if t in surface and len(cells) > 1:
            c.expect(ticks(cells[1]) == surface[t],
                     f"backend-contract §2.1 {t}: {ticks(cells[1])} != {surface[t]}")
    for name in ("surface_variants", "surface_optional", "`score`", "`first`", "`gaps`",
                 "`embeddings`", "explain-catalog.json", "vega/explain-catalog@1"):
        c.expect(name in _text(str(BACKEND_MD)), f"backend-contract.md does not state {name}")
    c.close()


# --------------------------------------------------------------------------- schemas

def _network_example():
    d = "sha256:" + "0" * 64
    summary = {"n": 4, "mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "l2": 2.0,
               "abs_q": [1.0, 1.0, 1.0, 1.0], "hist": {"lo": -1.0, "hi": 1.0, "counts": [0] * 32}}

    def node(nid, typ, view, call, block, in_shape, out_shape, axes, count, code=True):
        return {"id": nid, "block": block, "type": typ, "view": view, "parent": None,
                "children": [], "role": None, "call": call, "path": f"stages.{nid}",
                "in_shape": in_shape, "out_shape": out_shape, "axes": axes, "per": "sample",
                "config": {}, "params": {"count": count, "trainable": count},
                "formula": catalog()["inspection"]["views"][typ]["formula"],
                "code": {"file": "pkg/model/blocks_core.py", "class": "Dense",
                         "anchor": "class Dense("} if code else None}
    return {
        "schema": "vega/network-view@2", "available": True, "model_version": d,
        "spec_digest": d, "network_digest": d, "weights": "full",
        "family": {"name": "mlp-plain", "source": "vega:default"},
        "task": {"kind": "binary", "base": None, "link": "sigmoid",
                 "outputs": {"kind": "score", "names": ["attack"]}, "positive": "attack",
                 "threshold": 0.5, "threshold_scale": "probability", "quantiles": None,
                 "decision_rule": "threshold", "targets": ["binary_logit"],
                 "default_target": "binary_logit"},
        "sample": {"shape": "row", "length": None, "stride": None, "horizon": None,
                   "autoregressive": None, "level": None, "directed": None, "self_loops": None,
                   "protocol": None, "batching": None},
        "inputs": {"numeric": ["a", "b"], "nominal": [], "past_target": None, "order": ["a", "b"],
                   "width": 2, "edges": None,
                   "axis_labels": {"feature": ["a", {"name": "y", "past": True}], "step": None,
                                   "token": None}},
        "params": {"total": 3, "trainable": 3},
        "nodes": [node("input.num", "input", "io", "io", None, None, [2], ["feature"], 0, False),
                  node("B1", "head", "units", "tensor", "B1", [2], [1], ["output"], 3),
                  node("output", "output", "io", "io", None, [1], [1], ["output"], 0, False)],
        "edges": [{"from": "input.num", "to": "B1", "kind": "data"},
                  {"from": "B1", "to": "output", "kind": "readout"}],
        "chain": {"eligible": True, "linear_reading": True, "inputs": ["a", "b"],
                  "layers": [{"node": "B1", "in": 2, "out": 1, "activation": None, "dropout": 0,
                              "weight": "B1.weight", "bias": "B1.bias", "params": 3}]},
        "tensors": {"B1.weight": {"id": "B1.weight", "node": "B1", "role": "weight",
                                  "shape": [1, 2], "axes": ["out", "in"], "dtype": "float32",
                                  "encoding": "json", "data": [0.5, -0.25], "summary": summary},
                    "B1.bias": {"id": "B1.bias", "node": "B1", "role": "bias", "shape": [1],
                                "axes": ["out"], "dtype": "float32", "encoding": "base64-le",
                                "data": "AAAAAA==", "summary": summary}},
        "limits": dict(catalog()["inspection"]["limits"]),
        "inspection": {"schema": "vega/inspection@1", "readers": d},
    }


def _explanation_example():
    exact = {"shape": [2], "axes": ["feature"], "dtype": "float32", "encoding": "base64-le",
             "data": "AAAAAAAAAAA="}
    return {
        "schema": "vega/explanation@2", "detail": "exact", "degraded": True,
        "sample": {"shape": "row", "index": 0, "row": 12, "span": None, "graph": None},
        "prediction": {"label": "attack", "probabilities": {"attack": 0.7}, "warnings": []},
        "target": {"kind": "binary_logit", "index": 0, "name": "attack", "sign": 1,
                   "value": 0.85, "why": "predicted", "unit": "logit"},
        "outputs": {"kind": "score", "names": ["attack"], "link": "sigmoid", "raw": [0.85],
                    "read": [0.7], "answer": {"class": "attack", "p": 0.7, "threshold": 0.5}},
        "inputs": {"order": ["a", "b"], "shaped": [1.0, 2.0], "raw": {"a": 1, "b": "x"}},
        "attribution": {"method": "grad_x_input", "unit": "logit", "shape": "row",
                        "features": {"names": ["a", "b"], "values": [1.0, 2.0],
                                     "attribution": [0.5, -0.5], "grad": [0.5, -0.25]},
                        "total": 0.0, "completeness": None},
        "nodes": {
            "B1": {"level": "exact", "why": None,
                   "in": {"form": "vector", "n": 2, "values": [1.0, 2.0]},
                   "out": {"form": "vector", "n": 1, "values": [0.85]},
                   "grad_out": None, "relevance": None, "extra": {}, "recompute": 0.0,
                   "exact": {"in": exact, "out": exact, "grad_out": exact, "pre": exact,
                             "post": exact, "grad_pre": exact}},
            "output": {"level": "summary", "why": "past the exact budget",
                       "in": {"form": "summary", "norms": {"unit": [0.85]}, "sums": {}},
                       "out": None, "grad_out": None, "relevance": None, "extra": {},
                       "recompute": None, "exact": None}},
        "chain": {"inputs": [1.0, 2.0], "pre": [[0.85]], "post": [[0.85]], "grad_pre": [[1.0]]},
        "fidelity": {"head_max_abs_diff": 0.0, "served_max_abs_diff": 0.0, "tolerance": 1e-05,
                     "ok": True},
        "recompute": {"nodes": 1, "worst": 0.0, "worst_node": "B1"},
        "budget": {"limit": 4194304, "used": 12, "summarised": ["output"], "omitted": []},
        "warnings": [],
    }


def test_schemas():
    """Both inspection schemas are JSON Schema draft 2020-12, name their schema id, carry the
    catalog's closed sets, accept a minimal payload and refuse a wrong one."""
    c = Check("inspection schemas")
    try:
        import jsonschema
    except ImportError:
        c.fail("jsonschema >= 4 is this test's dependency (pip install jsonschema); the schema "
               "files cannot be checked without it")
        c.close()
        return
    ins = catalog()["inspection"]
    files = ins.get("schema_files") or {}
    docs = {}
    for key in ("network", "explanation"):
        path = SKILL / files.get(key, "")
        if not c.expect(path.is_file(), f"inspection.schema_files.{key} {path} does not exist"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        docs[key] = doc
        c.expect(doc.get("$schema") == DRAFT_2020_12,
                 f"{path.name}: $schema must be {DRAFT_2020_12}")
        try:
            jsonschema.Draft202012Validator.check_schema(doc)
            c.ok(f"{path.name} is a valid draft 2020-12 schema")
        except jsonschema.exceptions.SchemaError as exc:
            c.fail(f"{path.name} is not a valid draft 2020-12 schema: {exc.message}")
        c.expect(doc["properties"]["schema"].get("const") == ins["schemas"][key],
                 f"{path.name}: schema const must be {ins['schemas'][key]!r}")
    if len(docs) < 2:
        c.close()
        return
    nv, ex = docs["network"], docs["explanation"]
    closed = (
        (nv["$defs"]["link"]["enum"], list(ins["links"]), "network link"),
        (ex["$defs"]["link"]["enum"], list(ins["links"]), "explanation link"),
        (nv["$defs"]["target_kind"]["enum"], list(ins["target_kinds"]), "network target kinds"),
        (ex["$defs"]["target_kind"]["enum"], list(ins["target_kinds"]),
         "explanation target kinds"),
        (nv["$defs"]["task"]["properties"]["decision_rule"]["enum"], ins["decision_rules"],
         "decision rules"),
        (nv["$defs"]["node"]["properties"]["type"]["enum"],
         list(catalog()["blocks"]) + list(SYNTHETIC), "node types"),
        (nv["$defs"]["node"]["properties"]["view"]["enum"], ins["view_kinds"], "view kinds"),
        (nv["$defs"]["node"]["properties"]["call"]["enum"], ins["calls"], "calls"),
        (nv["$defs"]["edge"]["properties"]["kind"]["enum"], ins["edge_kinds"], "edge kinds"),
        (nv["$defs"]["tensor"]["properties"]["role"]["enum"], ins["tensor_roles"],
         "tensor roles"),
        (nv["$defs"]["output_kind"]["enum"], ins["output_kinds"], "output kinds"),
        (nv["$defs"]["limits"]["required"], list(ins["limits"]), "limits"),
        (ex["properties"]["detail"]["enum"], ins["detail"], "explanation detail"),
    )
    for got, want, what in closed:
        c.expect(list(got) == list(want), f"schema {what} {list(got)} != catalog {list(want)}")
    c.expect("buffer" in ins["tensor_roles"], "R-B6: tensors carry the role buffer")
    for field in ("positive", "quantiles", "decision_rule"):
        c.expect(field in nv["$defs"]["task"]["required"], f"R-F7: task.{field} is required")
    cases = ((nv, _network_example(), "network-view"), (ex, _explanation_example(), "explanation"))
    for schema, example, what in cases:
        errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(example),
                        key=lambda e: list(e.path))
        c.expect(not errors, f"the minimal {what} payload is refused: "
                             + "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:3]))
    wrong = _network_example()
    wrong["task"]["link"] = "sigmoid-per-label"
    c.expect(not jsonschema.Draft202012Validator(nv).is_valid(wrong),
             "the network schema accepts a link outside R-C1's vocabulary")
    wrong = _explanation_example()
    wrong["nodes"]["B1"]["exact"] = None
    c.expect(not jsonschema.Draft202012Validator(ex).is_valid(wrong),
             "the explanation schema accepts an exact node without its exact tensors")
    c.close()


# --------------------------------------------------------------------------- runner

TESTS = (test_tasks, test_task_rows_v23, test_families, test_compositions, test_blocks,
         test_defaults, test_defaults_rows, test_compiler_rows, test_forks,
         test_metric_direction, test_admission, test_links, test_inspection_rows,
         test_coverage_keys, test_schemas)


def main():
    data = catalog()
    print(f"comparing {CATALOG_JSON.name} and {COVERAGE_JSON.name} with {CATALOG_MD.name}, "
          f"{INSPECT_MD.name}, {BACKEND_MD.name} and the schema files")
    failed = []
    for test in TESTS:
        try:
            test()
            if VERBOSE:
                print(f"  pass  {test.__name__}")
        except AssertionError:
            failed.append(test.__name__)
    if failed:
        print(f"\n{len(failed)} group(s) diverged: {', '.join(failed)}")
        return 1
    ins = data["inspection"]
    print(f"\ncatalog.json, coverage.json and the prose agree: {len(data['tasks'])} task kinds, "
          f"{len(data['families'])} families, {len(data['blocks'])} block types, "
          f"{len(data['defaults'])} default rows, {len(data['metric_direction'])} metric "
          f"directions, {data['admission']['cells']} admitted cells, {len(ins['views'])} view "
          f"rows, {len(ins['targets'])} target rows, {len(ins['links'])} links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
