"""
The registry's validation (D6 §3.1, T0): the JSON Schema (`lanes.schema.json`) for the grammar,
and the cross-checks a schema cannot hold —

- a family, task, goal, loss, optimizer, schedule and init scheme exist in the forge catalog
  and the backend coverage;
- a lane's shape is its family's `sample_shape`, or one of its `floor_shapes` (R-A13);
- a window lane has its window (a horizon iff forecast), a graph lane its graph, with the
  level, batching, neighbour and protocol rules (R-A10, R-A15);
- every block is a catalog block with its required params (BlockGrammar);
- a lane's roles cover exactly its classes and an `alert` has a basis (R-E5);
- its lexicon holds exactly the keys its shape requires (R-E6), and a motif off the modality's
  base has a stance (R-E4);
- the flake list names an upstream fix for every signature (D6 §6.12).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import contracts
from .registry import (LANES_SCHEMA, STAGE_ORDER, Catalogs, Lane, choice_name,
                       lexicon_articles, motifs_by_modality, required_lexicon)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def err(self, lane: str, message: str) -> None:
        self.errors.append(f"{lane}: {message}")

    def warn(self, lane: str, message: str) -> None:
        self.warnings.append(f"{lane}: {message}")

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(doc: Mapping[str, Any], catalogs: Catalogs, flakes: Mapping | None = None) -> Report:
    """The registry against its schema, the catalogs, and itself."""
    report = Report()
    _schema_errors(doc, report)
    ids = [entry.get("id") for entry in doc.get("lanes", [])]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        report.err("registry", f"duplicate lane id(s): {', '.join(dupes)}")
    for entry in doc.get("lanes", []):
        LaneChecks(Lane(entry.get("id", "?"), entry), catalogs, report).run()
    _check_pairs(doc, set(ids), report)
    if catalogs.generators is None:
        report.warnings.append("registry: generators not checked — lanekit/datasets does not "
                               "exist yet (T02)")
    if flakes is not None:
        report.errors.extend(flake_problems(flakes))
    return report


def _schema_errors(doc: Mapping[str, Any], report: Report) -> None:
    try:
        import jsonschema
    except ImportError:                                          # pragma: no cover
        report.warnings.append("registry: jsonschema not installed — schema not checked")
        return
    schema = json.loads(LANES_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in error.absolute_path) or "(root)"
        report.errors.append(f"schema: {where}: {error.message}")


def _check_pairs(doc: Mapping[str, Any], ids: set, report: Report) -> None:
    for pair in (doc.get("defaults") or {}).get("adaptation_pairs", []):
        for side in pair.split(":"):
            if side not in ids:
                report.err("registry", f"adaptation pair {pair} names unknown lane {side}")


def flake_problems(flakes: Mapping[str, Any]) -> list[str]:
    """T0's check of `flakes.json`: the schema id, compilable regexes, and an upstream
    reference on every entry (the list may only shrink, D6 §6.12)."""
    problems = []
    if flakes.get("schema") != contracts.KNOWN_FLAKES:
        problems.append(f"flakes: schema is {flakes.get('schema')!r}, not {contracts.KNOWN_FLAKES}")
    for entry in flakes.get("flakes", []):
        fid = entry.get("id", "?")
        if not str(entry.get("upstream") or "").strip():
            problems.append(f"flakes: {fid} names no upstream fix — an entry without one "
                            "is a failure the matrix hides")
        for key in ("command_regex", "output_regex"):
            try:
                re.compile(entry.get(key) or "")
            except re.error as exc:
                problems.append(f"flakes: {fid}.{key} does not compile: {exc}")
    return problems


class LaneChecks:
    """The cross-checks of one lane. One method per rule; `run` calls them in order."""

    def __init__(self, lane: Lane, catalogs: Catalogs, report: Report) -> None:
        self.lane, self.cat, self.report = lane, catalogs, report
        self.forge = catalogs.forge

    def run(self) -> None:
        if not isinstance(self.lane.raw.get("family"), str):
            return                                              # the schema already said why
        for check in (self.family_task_goal, self.shape, self.window, self.graph, self.blocks,
                      self.loss_and_training, self.campaign, self.roles, self.lexicon,
                      self.motif, self.generator, self.stages):
            check()

    def err(self, message: str) -> None:
        self.report.err(self.lane.id, message)

    def warn(self, message: str) -> None:
        self.report.warn(self.lane.id, message)

    def family_task_goal(self) -> None:
        raw = self.lane.raw
        for key, table in (("family", "families"), ("task", "tasks"), ("goal", "goal_map")):
            if raw[key] not in self.forge[table]:
                self.err(f"{key} {raw[key]!r} is not in catalog.json {table}")
        implied = self.forge["goal_map"].get(raw["goal"])
        if implied and raw["task"] != implied:
            self.err(f"goal {raw['goal']!r} implies task {implied!r}, not {raw['task']!r}")
        if not implied and raw["task"] in ("unsupervised", "anomaly"):
            self.err(f"goal {raw['goal']!r} cannot carry task {raw['task']!r}")
        base = self.lane.target.get("base")
        if raw["task"].startswith("sequence-") and not base:
            self.err("a sequence task names its base kind (target.base)")

    def shape(self) -> None:
        raw = self.lane.raw
        if raw["shape"] not in self.forge["sample_shapes"]:
            self.err(f"shape {raw['shape']!r} is not in catalog.json sample_shapes")
        family = self.forge["families"].get(raw["family"]) or {}
        allowed = [family.get("sample_shape")] + list(family.get("floor_shapes") or [])
        if raw["shape"] not in allowed:
            self.err(f"shape {raw['shape']!r} is not family {raw['family']}'s sample_shape "
                     f"{family.get('sample_shape')!r} nor one of its floor_shapes "
                     f"{family.get('floor_shapes') or []}")
        level = family.get("level")
        if level and self.lane.graph_level and level != self.lane.graph_level:
            self.err(f"family {raw['family']} is level {level}, the lane says {self.lane.graph_level}")

    def window(self) -> None:
        raw, window = self.lane.raw, self.lane.raw.get("window")
        if (raw["shape"] == "window") != bool(window):
            self.err("window is required iff the shape is window")
            return
        if not window:
            return
        forecast = raw["task"] == "forecast"
        if forecast != (window.get("horizon") is not None):
            self.err("window.horizon is required iff the task is forecast")
        if not forecast and window.get("autoregressive") is not None:
            self.err("window.autoregressive applies to forecast only (R-A7)")
        source = window.get("autoregressive_source")
        if forecast and window.get("autoregressive") is False and not str(source or "").startswith("user:"):
            self.err("autoregressive false departs from the default: it needs a user: source")

    def graph(self) -> None:
        raw, graph = self.lane.raw, self.lane.raw.get("graph")
        if (raw["shape"] == "graph") != bool(graph):
            self.err("graph is required iff the shape is graph")
            return
        if not graph:
            return
        rules = (
            (graph["level"] == "graph" and not graph.get("graph_id"), "level graph needs graph_id"),
            (graph["level"] == "graph" and "graph_pool" not in self.lane.flat_block_types(),
             "level graph needs a graph_pool block (catalog graph_level.requires)"),
            ((graph["batching"] == "graphs") != (graph["level"] == "graph"),
             "batching graphs iff level graph"),
            ((graph["batching"] == "sampled") != bool(graph.get("neighbours")),
             "neighbours iff batching sampled (R-A15)"),
            (graph["method"] == "knn" and not graph.get("k"), "knn needs k"),
            (graph["protocol"] == "transductive"
             and not str(graph.get("protocol_source") or "").startswith("user:"),
             "transductive needs a user: source (catalog graph_protocols)"),
            (graph["protocol"] == "transductive" and not self.lane.expect.get("pack_refusal"),
             "a transductive lane stops at eval and asserts pack's refusal (R-A10)"),
        )
        for broken, message in rules:
            if broken:
                self.err(message)

    def blocks(self) -> None:
        blocks = self.lane.raw.get("blocks")
        if blocks is None:
            if not self.lane.get("recorded"):
                self.err("blocks may be null only for a recorded lane")
            return
        ids = [b.get("id") for b in blocks]
        for dupe in sorted({i for i in ids if ids.count(i) > 1}):
            self.err(f"block id {dupe} is not unique")
        BlockGrammar(self.forge["blocks"], self.err, self.warn).check_list(blocks, "blocks")
        if blocks and blocks[-1]["type"] != "head":
            self.err("the last block is the head")

    def loss_and_training(self) -> None:
        raw, cov = self.lane.raw, self.cat.coverage
        loss = raw["loss"]["name"]
        allowed = self.forge["tasks"].get(raw["task"], {}).get("losses_allowed") or []
        if loss not in allowed:
            self.err(f"loss {loss!r} is not in tasks.{raw['task']}.losses_allowed {allowed}")
        if loss not in cov["losses"]:
            self.err(f"loss {loss!r} is not in coverage.json losses")
        for key, table in (("optimizer", "optimizers"), ("schedule", "schedules"),
                           ("init", "init_schemes")):
            name = choice_name(raw.get(key))
            if name is None and not raw.get("recorded"):
                self.err(f"{key} may be null only for a recorded lane")
            elif name is not None and name not in cov[table]:
                self.err(f"{key} {name!r} is not in coverage.json {table}")

    def campaign(self) -> None:
        campaign = self.lane.raw["campaign"]
        if campaign["rungs"][:1] != ["R0"]:
            self.err("campaign.rungs starts with R0")
        floor = campaign.get("expect_floor")
        if floor is not None and "R1" not in campaign["rungs"]:
            self.err("expect_floor names the R1 rung's outcome; R1 is not in the rungs")
        if "R1b" in campaign["rungs"] and self.lane.raw["family"] not in RECURRENT_R1B:
            self.err(f"R1b swaps a recurrent block to rnn; {self.lane.raw['family']} has none (R-A12)")

    def roles(self) -> None:
        roles, classes = self.lane.raw.get("roles"), self.lane.classes
        reaches_show = STAGE_ORDER.index(self.lane.last_stage) >= STAGE_ORDER.index("show")
        if roles is None:
            if classes and reaches_show:
                self.err("a lane with classes that reaches show declares their roles")
            return
        if not classes:
            self.err("roles on a task without classes")
            return
        if set(roles) != set(classes):
            self.err(f"roles cover {sorted(roles)}, the classes are {sorted(classes)}")
        if "alert" in roles.values() and not self._alert_basis():
            self.err("an alert role needs a user:/show: stance or task:anomaly (R-E5)")

    def _alert_basis(self) -> bool:
        if self.lane.raw["task"] == "anomaly":
            return True
        return any(re.search(r"\bshow: roles\b.*\(user\)", s) for s in self.lane.get("stances") or [])

    def lexicon(self) -> None:
        lexicon = self.lane.raw.get("lexicon") or {}
        modality = contracts.lexicon_modality(self.lane.raw["shape"], self.lane.graph_level)
        required = set(required_lexicon(self.cat.show, modality))
        given = set(lexicon) - {"target"}
        if given != required:
            self.err(f"lexicon keys {sorted(given)} are not the {modality} set {sorted(required)} (R-E6)")
        pattern = ((self.cat.show or {}).get("lexicon") or {}).get("value_pattern") or contracts.LEXICON_WORD
        word = re.compile(pattern)
        for key, entry in lexicon.items():
            missing = [part for part in contracts.LEXICON_ENTRY if part not in entry]
            if missing:
                self.err(f"lexicon.{key} lacks {', '.join(missing)} (every entry is one, many, article)")
            if any(not isinstance(entry.get(w), str) or not word.match(entry[w]) for w in ("one", "many")):
                self.err(f"lexicon.{key} words must match {pattern}")
            allowed = lexicon_articles(self.cat.show, key)
            if entry.get("article") not in allowed:
                self.err(f"lexicon.{key}: article {entry.get('article')!r} is not one of {list(allowed)}")

    def motif(self) -> None:
        motif = self.lane.expect.get("motif")
        if motif is None:
            return
        table = motifs_by_modality(self.cat.show)
        admissible = (table.get(self.lane.modality) or {}).get("admissible", ())
        if motif not in admissible:
            self.err(f"motif {motif!r} is not admissible for {self.lane.modality} {list(admissible)}")
        base = (table.get(self.lane.modality) or {}).get("base")
        stanced = any(f"show: motif {motif}" in s for s in self.lane.get("stances") or [])
        if motif != base and not stanced:
            self.err(f"motif {motif!r} departs from the base {base!r}: it needs a show: stance (R-E4)")

    def generator(self) -> None:
        gens, name = self.cat.generators, self.lane.raw["generator"]
        if gens is None or name in gens:
            return
        from .datasets import PLANNED
        if name in PLANNED:
            self.warn(f"generator {name!r} is not written yet ({PLANNED[name]} provides it)")
        else:
            self.err(f"generator {name!r} is not in lanekit.datasets.GENERATORS")

    def stages(self) -> None:
        last = self.lane.expect.get("last_stage")
        if last is not None and last not in STAGE_ORDER:
            self.err(f"expect.last_stage {last!r} is not a stage")
        if self.lane.expect.get("pack_refusal") and last != "eval":
            self.err("a lane that asserts pack's refusal stops at eval")


RECURRENT_R1B = {"gru", "lstm", "bi-gru", "bi-lstm", "lstm-attention", "gru-attention"}


class BlockGrammar:
    """A block list against catalog.json `blocks`: known types, required params present, the
    declared param types. An undeclared param is a warning — the compiler is the gate for the
    grammar's details; this catches the registry drifting from the catalog early."""

    SPECIAL = {("head", "out"): {"recon_dim"}}

    def __init__(self, table: Mapping[str, Any], err, warn) -> None:
        self.table, self.err, self.warn = table, err, warn

    def check_list(self, blocks: Sequence[Mapping], where: str) -> None:
        for i, block in enumerate(blocks):
            self.check(block, f"{where}[{i}]")

    def check(self, block: Mapping, where: str) -> None:
        kind = block.get("type")
        spec = self.table.get(kind)
        if spec is None:
            self.err(f"{where}: block type {kind!r} is not in catalog.json blocks")
            return
        params = block.get("params") or {}
        for name in spec.get("required") or []:
            if name not in params:
                self.err(f"{where}: {kind} needs param {name!r}")
        declared = spec.get("params") or {}
        for name, value in params.items():
            if name not in declared:
                self.warn(f"{where}: {kind}.{name} is not a catalog param (yet)")
            elif not self._fits(kind, name, declared[name], value):
                self.err(f"{where}: {kind}.{name}={value!r} does not fit {declared[name]!r}")
        if "inner" in params:
            self.check_list(params["inner"], f"{where}.inner")

    def _fits(self, kind: str, name: str, grammar: str, value: Any) -> bool:
        if value in self.SPECIAL.get((kind, name), ()):
            return True
        head = grammar.split(" ", 1)[0].strip()
        if "|" in head and "[" not in head:
            options = head.split("|")
            if "int" in options and isinstance(value, int):
                return True
            if "str" in options and isinstance(value, str):
                return True
            return value in options
        checks = {"int": lambda v: isinstance(v, int) and not isinstance(v, bool),
                  "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
                  "bool": lambda v: isinstance(v, bool), "str": lambda v: isinstance(v, str),
                  "list[str]": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
                  "list[int]": lambda v: isinstance(v, list) and all(isinstance(x, int) for x in v),
                  "list[block]": lambda v: isinstance(v, list),
                  "dict[str,int]": lambda v: isinstance(v, dict)}
        check = checks.get(head)
        return True if check is None else check(value)


