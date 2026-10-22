"""
The judgement a lane's stages publish, written without a number (D6 §2.2, §6; BUILD-PLAN T03).

Every layer the chain publishes has two halves. The mechanical half is the records the stage
scripts wrote, which the apply scripts splice in verbatim (`from_run`, `from_campaign`,
`from_bundle`). The judgement is prose, forks and markers — in the chain, the skill's model
writes it. A lane has no model in the loop, so these writers compose the judgement from the
records instead, in the form each apply script validates:

    forge_authoring(run_dir, lane) -> dict        vega/architecture@1 authoring, from_run "."
    train_authoring(campaign, lane) -> dict       vega/training@1 authoring, from_campaign "."
    pack_authoring(bundle, lane) -> dict          bundle layer authoring, from_bundle "bundle/<slug>"
    show_authoring(facts, lane, draft) -> dict    vega/story-plan@2 with `authored` filled

`from_bundle` is relative to the lane's work dir, where the pack stage runs the apply script
(cwd = the work dir) and the model page lives, so the layer's model-card link resolves from the
page; a bundle outside a `bundle/` directory gets ".".

The one rule every writer keeps: **no digit outside a `{placeholder}`** in any authored string.
A number typed into the judgement would be the one figure tracing to no record — the apply
scripts and the story's digit rule (R-E10) refuse exactly that — so the figures stay the
records', spliced beside the prose, and the story cites them only as `{fact_key}` or lexicon
placeholders. `prose(doc, kind)` lists the strings a writer authored and `digit_problems(doc,
kind)` checks them; identifiers (ids, paths, class names, labels, fact keys) are not prose.

The class roles a lane declares need a basis the story accepts (R-E5): lanekit writes a
`YYYY-MM-DD | <dataset> | show: roles … (user)` line into `.claude/domain-forge-memory.md`
§ Network stances for every lane that declares `alert`, and a `show: motif … (user)` line for
a lane whose stances name its motif, before the show stage drafts. Each class's basis in the
line is one the vega-show catalog's `roles.basis_pattern` accepts (`structural:negative`,
`task:anomaly`, `user:<turn>`); `<dataset>` is the generator, the model's file stem. When the
draft found no such stance, `show_authoring` falls back to the registry as the user for the
alert roles and says so with a `WARN:` on stdout — the stage log, where the runner counts it:

    show_stances(lane, dataset=None, date=None) -> [line]
    write_show_stances(memory, lane, dataset=None, date=None) -> [line written]
    python -m lanekit.authoring stances --lanes-file F --lane ID --memory M [--dataset D]
    python -m lanekit.authoring check --kind forge|train|pack|show FILE
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

KINDS = ("forge", "train", "pack", "show")
UNATTENDED = "default applied (unattended)"
STANCE_SECTION = "Network stances"
PLACEHOLDER = r"\{([a-z][a-z0-9_]*)(?:\.([a-z][a-z0-9_]*))?(?::(int|num|pct|signed|count))?\}"   # R-E10
REGISTRY_BASIS = "registry:{lane}"  # the lane registry declared it: never written as a user's turn (gate C)


def registry_basis(lane: Any) -> str:
    """The basis of a role the lane registry declares (vega-show catalog `registry:<lane>`): the
    plan's fallback when no stance was found, and a roles line's per-class basis."""
    raw = _raw(lane) if not isinstance(lane, str) else {"id": lane}
    return REGISTRY_BASIS.format(lane=re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw.get("id") or "lane")))


# --------------------------------------------------------------------------- small readers
def _raw(lane: Any) -> Mapping[str, Any]:
    """A registry Lane or its raw mapping."""
    return lane.raw if hasattr(lane, "raw") else lane


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_or_none(path: Path) -> Any:
    try:
        return _json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _claude_dir(start: Path) -> Path | None:
    """The nearest `.claude/` at or above `start` (the lane's work dir holds it)."""
    for folder in (Path(start).resolve(), *Path(start).resolve().parents):
        if (folder / ".claude").is_dir():
            return folder / ".claude"
    return None


def _stance_lines(claude: Path | None) -> list[str]:
    """The § Network stances lines of the memory file beside `claude`."""
    memory = claude / "domain-forge-memory.md" if claude else None
    if memory is None or not memory.is_file():
        return []
    return _section_lines(memory.read_text(encoding="utf-8"))


def _section_lines(text: str) -> list[str]:
    heading = re.search(rf"^#+\s*{STANCE_SECTION}\s*$", text, re.I | re.M)
    if not heading:
        return []
    tail = text[heading.end():]
    nxt = re.search(r"^#+\s", tail, re.M)
    body = tail[:nxt.start()] if nxt else tail
    return [line.strip("-* \t") for line in body.splitlines() if "|" in line and line.strip()]


# --------------------------------------------------------------------------- names as words
NAME_WORDS = {"cnn1d": "one-dimensional convolutional", "conv1d": "one-dimensional convolution",
              "pool1d": "one-dimensional pooling", "f1": "F-score", "l1": "absolute-error",
              "l2": "squared-error", "top5": "top-five"}
DIGIT_NAMES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def words(name: Any) -> str:
    """An identifier as prose with no digit: underscores become spaces, a known digit-bearing
    name its words (`conv1d` → one-dimensional convolution), any other digit its name."""
    text = str(name if name is not None else "").replace("_", " ").strip()

    def token(match: re.Match) -> str:
        word = match.group(0)
        if not any(ch.isdigit() for ch in word):
            return word
        known = NAME_WORDS.get(word.lower())
        if known:
            return known
        spelled = re.sub(r"\d", lambda d: f"-{DIGIT_NAMES[unicodedata.digit(d.group(0))]}-", word)
        return spelled.strip("-").replace("--", "-")

    return re.sub(r"[^\s,;()]+", token, text)


def _name(node: Any) -> str | None:
    if isinstance(node, Mapping):
        return node.get("name") or node.get("scheme")
    return node if isinstance(node, str) else None


def _source(node: Any) -> str:
    return str((node or {}).get("source") or "") if isinstance(node, Mapping) else ""


def _source_phrase(source: str) -> str:
    """Where a choice came from, in words."""
    prefix = source.split(":", 1)[0]
    return {"user": "a user turn chose it", "blueprint": "the blueprint's posture bet on it",
            "vega": "the catalog's default for this task", "analysis": "an analysis finding asked for it",
            "geometry": "the data's geometry set it", "shape": "the shaping record set it",
            "memory": "a recorded stance set it", "stance": "a recorded stance set it",
            "netspec": "the spec set it"}.get(prefix, "its source is recorded beside it")


BLOCK_WORDS = {
    "dense": "dense layer", "embedding": "embedding", "concat": "concatenation",
    "batchnorm": "batch normalisation", "layernorm": "layer normalisation", "dropout": "dropout",
    "activation": "activation", "residual": "residual block", "sequential": "sequential stack",
    "parallel": "parallel branches", "conv1d": "convolution along the sequence",
    "pool1d": "pooling along the sequence", "rnn": "recurrent layer", "gru": "gated recurrent layer",
    "lstm": "long short-term memory layer", "attention": "attention layer",
    "attention_pool": "attention pooling", "transformer_encoder": "transformer encoder",
    "feature_tokenizer": "feature tokenizer", "positional": "positional encoding",
    "flatten": "flattening", "reshape": "reshape", "pool": "pooling over the sequence",
    "head": "head", "decoder": "decoder", "reparameterize": "reparameterisation",
    "gcn_conv": "graph convolution", "sage_conv": "neighbourhood aggregation",
    "gat_conv": "graph attention layer", "gin_conv": "graph isomorphism layer",
    "graph_pool": "graph pooling", "edge_dropout": "edge dropout", "custom": "custom block"}


def block_words(block_type: Any) -> str:
    return BLOCK_WORDS.get(str(block_type), words(block_type))


def _flat_types(blocks: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for block in blocks or []:
        if isinstance(block, Mapping):
            out.append(str(block.get("type")))
            out.extend(_flat_types((block.get("params") or {}).get("inner") or []))
    return out


def _listing(items: Iterable[str]) -> str:
    items = list(dict.fromkeys(i for i in items if i))
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# =========================================================================== forge
FORK_ASKS = {
    "architecture": "the family and its blocks: {family} as the lane composes it, or another row of the catalog",
    "objective": "the loss and the metrics: {loss} with {primary} as the primary, or the task row's alternatives",
    "optimization": "the optimizer, the schedule, the initialisation and the batching: {optimizer} with a {schedule} "
                    "schedule, or the alternatives the spec lists",
    "regularization": "the regularization and early stopping: {regularization}, or the alternatives the spec lists"}
FORK_POINTS = {    # family-catalog §5 / catalog.json fork_order points, in the catalog's order
    "context": ("architecture", "the window context: the previous part of each series, or none"),
    "autoregressive": ("architecture", "whether the target is fed back to the network as an input channel"),
    "series_by": ("architecture", "the series key the windows are cut within"),
    "protocol": ("architecture", "the graph protocol: inductive or transductive"),
    "batching": ("optimization", "the graph batching: the full graph at once, or sampled neighbourhoods")}
# netspec-contract §1: a point's provenance is `<point>_source`, except the series key's
POINT_SOURCE_KEYS = {"series_by": "series_source"}
UNTESTED_POINTS = ("protocol",)    # netspec-contract §3: every such default lands in defaults_to_test but protocol
POINT_ANSWERS = {"user": "the user's recorded stance, applied without asking",
                 "shape": "the shaping record's choice, applied without asking"}
SERIES_UNATTENDED = "null, the catalog's default when the shaping record names no series key — " + UNATTENDED
PROTOCOL_UNATTENDED = ("inductive, always: never asked unattended, because transductive needs a user: "
                       "turn — " + UNATTENDED)
POINT_UNATTENDED = {"series_by": SERIES_UNATTENDED, "protocol": PROTOCOL_UNATTENDED}   # catalog fork_order
SERIES_WARN = "WARN: windows may cross series — no series key in the shaping record"   # netspec-contract §3
POINT_OBSERVATIONS = {
    "context": "windows at the start of each valid series, which read their context from the train part's "
               "tail, scoring clearly worse than the windows after them would mean that context misleads more "
               "than it helps",
    "autoregressive": "a forecast error that does not move when the target's own past leaves the inputs would "
                      "mean the channel adds nothing the other inputs do not already carry",
    "series_by": "windows that straddle two series scoring clearly worse than the windows inside a single "
                 "series would mean the windows need a series key the shaping record does not name",
    "series_by_key": "the windows of every series scoring alike, no series standing apart, would mean the key "
                     "separates no distinct process and a window could cross it",
    "batching": "a train loss still falling steadily when the run ends would mean a single full-graph step per "
                "epoch gives too few updates, and sampled neighbourhoods would give more of them",
    "batching_graphs": "none: at graph level each graph is a sample and a batch is whole graphs, so no run's "
                       "curve can overturn it"}
DEFAULT_OBSERVATIONS = {
    "family": "a floor rung that reaches the baseline's valid score would mean the family's extra capacity "
              "is not being used",
    "loss": "a valid loss that keeps falling while the primary metric stalls would mean the loss is not "
            "measuring what selection reads",
    "metrics": "a primary metric that orders the runs differently from the valid loss would send the metric "
               "row back for a second look",
    "optimizer": "an update ratio far above the catalog's band in the first epochs (thrashing) would overturn "
                 "the optimizer's learning rate",
    "schedule": "early stopping firing long before the schedule reaches its horizon would mean the schedule "
                "never reached its tail",
    "early_stopping": "a best epoch close to the last epoch would mean the patience stopped a run that was "
                      "still improving",
    "init": "per-block gradient norms collapsing from the head to the first block in the first epoch would "
            "mean the scheme is not holding the variance",
    "training": "the budget ending a run before early stopping does would mean the epoch cap was never "
                "reachable at this batch size",
    "weight_decay": "a train/valid gap that does not move when the decay is removed would mean the decay is "
                    "inert",
    "dropout": "a train/valid gap still wide at the best epoch would mean the dropout rate is too low",
    "grad_clip": "a global gradient norm that never approaches the clip would mean the clip is inert and can "
                 "be dropped",
    "structural": "none: the block is structural, and no run's curve can overturn it — the network would not "
                  "build without it"}
RUNTIME_ROWS = ("defaults/device", "defaults/precision", "defaults/seed")
STRUCTURAL_TYPES = ("concat", "head", "flatten", "reshape")


def forge_authoring(run_dir: Path, lane: Any) -> dict:
    """The /vega-forge judgement for the network in `run_dir` (netspec.json there): the
    abstract, the per-family readings, the mechanics, a fork per decision family (answered by
    a recorded stance, else the unattended default) and per fork point the sample raises
    (netspec-contract §3), a defaults_to_test row for every `vega:default` the ladder can
    test — the sample's fork-point defaults included, protocol excepted — and the markers.
    The architect run the lane's work dir holds (`.claude/architect-runs/*/`) is spliced by
    the apply script; without one the conformance says `not installed`, out loud."""
    run = Path(run_dir)
    spec = _json(run / "netspec.json")
    claude = _claude_dir(run)
    stances = " ".join(_stance_lines(claude)).lower()
    findings = _architect_findings(claude)
    markers = ["OK: the spec compiled with no error, and the emitter's smoke forward pass ran on a real batch",
               "OK: the mechanical conformance gate passed on the emitted package"]
    conformance: dict[str, Any] = {"mechanical": "pass", "markers": []}
    if findings is None:
        conformance["architect"] = "not installed"
        markers.append("WARN: architect not installed — conformance judged mechanically only; the lane's work "
                       "dir holds no recorded architect run")
    if claude is None or not sorted((claude / "improve-runs").glob("*/findings.json")):
        conformance["improve"] = "not installed"
        markers.append("WARN: improve not installed — conformance judged mechanically only for the "
                       "method-level half; the lane's work dir holds no recorded improve run")
    if "series_by" in _sample_points(spec) and _point_state(_sample(spec), "series_by") == "default":
        markers.append(SERIES_WARN)
    return {"schema": "vega/architecture@1", "from_run": ".", "conformance": conformance,
            "code_findings": [{"id": fid, "next": "none from the lane: the finding belongs to the recorded "
                                                  "architect run the fixture copies, and the lane splices it "
                                                  "to prove the path, not to act on it"}
                              for fid in findings or []],
            "defaults_to_test": _defaults_to_test(spec),
            "forks": _forge_forks(spec, stances),
            "markers": markers,
            "readings": _forge_readings(spec, _raw(lane))}


def _architect_findings(claude: Path | None) -> list[str] | None:
    """The finding ids of the last recorded architect run (the one the stage passes), or None."""
    runs = sorted((claude / "architect-runs").glob("*/")) if claude else []
    if not runs:
        return None
    raw = _json_or_none(runs[-1] / "findings.json")
    found = raw.get("findings") if isinstance(raw, Mapping) else raw
    return [str(f["id"]) for f in found or [] if isinstance(f, Mapping) and f.get("id")]


def _defaults_to_test(spec: Mapping[str, Any]) -> list[dict]:
    """One row per `vega:default` choice, labelled the way apply_architecture_layer.py looks
    them up; runtime rows (device, precision, seed) get none."""
    rows = []

    def note(label: str, node: Any, key: str) -> None:
        if not isinstance(node, Mapping) or node.get("source") != "vega:default":
            return
        catalog = node.get("catalog")
        catalog = [catalog] if isinstance(catalog, str) else list(catalog or [])
        if catalog and all(r in RUNTIME_ROWS for r in catalog):
            return
        observation = DEFAULT_OBSERVATIONS.get(key) or (
            f"an ablation without the {words(key)} that matches the baseline would mean it is doing nothing")
        rows.append({"path": label, "observation": observation})

    for key in ("family", "loss", "metrics", "optimizer", "schedule", "early_stopping", "init", "training"):
        note(key, spec.get(key), key)
    for reg in spec.get("regularization") or []:
        kind = (reg or {}).get("kind")
        note(f"regularization[kind={kind}]/value", reg, str(kind))
    for block in spec.get("blocks") or []:
        if not isinstance(block, Mapping):
            continue
        btype = block.get("type")
        if btype in STRUCTURAL_TYPES:
            note(f"blocks[{block.get('id')}]", block, "structural")
            continue
        if block.get("source") == "vega:default":
            rows.append({"path": f"blocks[{block.get('id')}]",
                         "observation": f"a train/valid gap that opens early, or a train loss that plateaus above "
                                        f"the floor's, would say the {block_words(btype)} is sized wrong"})
    return rows + _point_defaults(spec)


def _point_defaults(spec: Mapping[str, Any]) -> list[dict]:
    """A row per fork point of the sample whose default was applied (its source `vega:default`,
    or nothing settled it), protocol excepted; the path is the one plan_ladder reads
    (`data_contract/sample/<point>`)."""
    sample, where = _sample(spec), _sample_path(spec)
    rows = []
    for point in _sample_points(spec):
        if point in UNTESTED_POINTS:
            continue
        if _point_state(sample, point) == "default" or _point_source(sample, point) == "vega:default":
            rows.append({"path": f"{where}/{point}", "observation": _point_observation(sample, point)})
    return rows


def _point_observation(sample: Mapping[str, Any], point: str) -> str:
    if point == "series_by" and sample.get("series_by") is not None:
        return POINT_OBSERVATIONS["series_by_key"]
    if point == "batching" and sample.get("batching") == "graphs":
        return POINT_OBSERVATIONS["batching_graphs"]
    return POINT_OBSERVATIONS[point]


def _sample(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    return spec.get("sample") or (spec.get("data_contract") or {}).get("sample") or {}


def _sample_path(spec: Mapping[str, Any]) -> str:
    return "sample" if isinstance(spec.get("sample"), Mapping) else "data_contract/sample"


def _sample_points(spec: Mapping[str, Any]) -> list[str]:
    """The fork points the sample raises (family-catalog §5), in the catalog's order: a window's
    context, its autoregressive channel on a forecast, its series key; a graph's protocol and
    batching."""
    shape, task = _sample(spec).get("shape"), (spec.get("task") or {}).get("kind")
    if shape == "window":
        return ["context"] + (["autoregressive"] if task == "forecast" else []) + ["series_by"]
    return ["protocol", "batching"] if shape == "graph" else []


def _point_source(sample: Mapping[str, Any], point: str) -> str:
    return str(sample.get(POINT_SOURCE_KEYS.get(point, f"{point}_source")) or "")


def _point_state(sample: Mapping[str, Any], point: str) -> str | None:
    """How a fork point stands: None when it is not asked — the series key the shaping record
    names (a `shape:` source, or a key the catalog's rule read from the shaping record); `answer`
    when a source other than the catalog's default settled it (a `user:` turn, the shaping
    record); `default` when the catalog's default was applied — for the series key, null."""
    source = _point_source(sample, point)
    if point == "series_by" and not source.startswith("user:") and (
            source.startswith("shape:") or sample.get("series_by") is not None):
        return None
    return "default" if source in ("", "vega:default") else "answer"


def _answered(nodes: Iterable[Any], names: Iterable[str], stances: str) -> bool:
    if any(_source(n).startswith("user:") for n in nodes):
        return True
    return any(name and re.search(rf"\b{re.escape(name.lower())}\b", stances) for name in names)


def _forge_forks(spec: Mapping[str, Any], stances: str) -> list[dict]:
    fam, loss = _name(spec.get("family")), _name(spec.get("loss"))
    regs = [r for r in spec.get("regularization") or [] if isinstance(r, Mapping)]
    fill = {"family": words(fam), "loss": words(loss),
            "primary": words((spec.get("metrics") or {}).get("primary")),
            "optimizer": words(_name(spec.get("optimizer"))), "schedule": words(_name(spec.get("schedule"))),
            "regularization": _listing(words(r.get("kind")) for r in regs) or "none declared"}
    settled = {
        "architecture": _answered([spec.get("family")], [f"family: {fam}"], stances),
        "objective": _answered([spec.get("loss"), spec.get("metrics")], [f"loss: {loss}"], stances),
        "optimization": _answered([spec.get(k) for k in ("optimizer", "schedule", "training", "init")],
                                  [f"optimizer: {_name(spec.get('optimizer'))}"], stances),
        "regularization": _answered(regs + [spec.get("early_stopping")],
                                    [str(r.get("kind")) for r in regs if r.get("source") != "vega:default"],
                                    stances)}
    forks = []
    for family, template in FORK_ASKS.items():
        fork = {"family": family, "asked": template.format(**fill)}
        if settled[family]:
            fork["answer"] = "the recorded choice, applied without asking (a user turn or a Network stance)"
        else:
            fork["or"] = UNATTENDED
        forks.append(fork)
    return forks + _point_forks(spec)


def _point_forks(spec: Mapping[str, Any]) -> list[dict]:
    """A fork per point the sample raises and asks: settled by its source, else the default
    applied unattended — null for a series key nothing names (the marker says so), inductive
    for the graph protocol, which is never asked unattended (catalog fork_order `unattended`)."""
    sample, forks = _sample(spec), []
    for point in _sample_points(spec):
        state = _point_state(sample, point)
        if state is None:
            continue
        family, asked = FORK_POINTS[point]
        fork = {"family": family, "asked": asked}
        if state == "answer":
            prefix = _point_source(sample, point).split(":", 1)[0]
            fork["answer"] = POINT_ANSWERS.get(prefix, "the recorded source, applied without asking")
        else:
            fork["or"] = POINT_UNATTENDED.get(point, UNATTENDED)
        forks.append(fork)
    return forks


def _forge_readings(spec: Mapping[str, Any], lane: Mapping[str, Any]) -> dict:
    fam_node, loss_node = spec.get("family") or {}, spec.get("loss") or {}
    family, task = words(_name(fam_node)), words((spec.get("task") or {}).get("kind") or lane.get("task"))
    metrics = spec.get("metrics") or {}
    types = [block_words(t) for t in _flat_types(spec.get("blocks") or [])]
    regs = [r for r in spec.get("regularization") or [] if isinstance(r, Mapping)]
    secondary = [words(m) for m in metrics.get("secondary") or []]
    sample = _sample(spec)
    shape_phrase = {"window": "as windows cut along each series", "graph": "as nodes of a graph with their edges"
                    }.get(sample.get("shape"), "as one row per sample")
    abstract = (f"This network belongs to the {family} family and answers the {task} task. The lane program "
                "forged it from the composition the lane declares rather than from a conversation. Every choice "
                "names its source — a recorded stance, an analysis finding, or the catalog's default — and every "
                "default the ladder can test carries the observation that would overturn it. What the first run "
                "must show is that this composition learns on the lane's data at all; the ladder's floor is what "
                "it is then read against.")
    primary = (f"The primary metric is {words(metrics['primary'])}, the one selection reads on the valid part"
               if metrics.get("primary") else "The metrics row names no primary metric")
    per_family = {
        "architecture": (f"The family is {family}: {_source_phrase(_source(fam_node))}. Its blocks run in forward "
                         f"order through {_listing(types) or 'no declared block'}, and the inputs arrive "
                         f"{shape_phrase}, as the shaping record's output schema types them."),
        "objective": (f"The loss is {words(_name(loss_node))}: {_source_phrase(_source(loss_node))}. {primary}"
                      + (f", with {_listing(secondary)} beside it." if secondary else ".")),
        "optimization": (f"The optimizer is {words(_name(spec.get('optimizer')))} with a "
                         f"{words(_name(spec.get('schedule')))} schedule and {words(_name(spec.get('init')))} "
                         "initialisation; the batch size and the epoch cap are the training row's, and the wall "
                         "clock budget ends a run the patience does not."),
        "regularization": ((f"The regularization is {_listing(words(r.get('kind')) for r in regs)}, "
                            if regs else "No regularization is declared, ")
                           + "with early stopping on the monitor the metrics row names; each carries its source "
                             "and, when it is a default, the observation that would overturn it.")}
    mechanics = ("The loss the spec writes out and the optimizer's update rule are spliced beside this text from "
                 "the spec itself, and the schedule's curve from the emitter's smoke record; the notebook "
                 "recomputes the first update by hand from them and compares it with the backend's own step. "
                 "Nothing here restates a figure: each one is the record's.")
    return {"abstract": abstract, "per_family": per_family, "mechanics": mechanics}


# =========================================================================== train
STOPPED_BY = {"early_stopping": "early stopping ended it at its patience",
              "budget": "the wall clock budget ended it", "max_epochs": "it ran to the epoch cap",
              "pruned": "the search pruned it", "diverged": "it diverged and was stopped"}
RUN_KIND = {"baseline": "the baseline", "ablation": "a one-change rung of the ladder",
            "hpo": "a trial of the search", "trial": "a trial of the search", "floor": "the floor rung"}


def train_authoring(campaign: Path, lane: Any) -> dict:
    """The /vega-train judgement for the campaign in `campaign`: the abstract, one reading
    per run in the ledger, the study, the selection, the two forks (the ladder applied
    unattended; the selection by the rule unless an override is recorded) and the markers.
    The fired diagnostics are spliced by the apply script from the ledger."""
    camp = Path(campaign)
    exp = _json(camp / "experiments.json")
    spec = _json_or_none(camp / "netspec.json") or {}
    runs = [r for r in exp.get("runs") or [] if isinstance(r, Mapping)]
    selection = exp.get("selection") or {}
    selected = next((r for r in runs if r.get("id") == selection.get("run")), None)
    studies = exp.get("hpo") or []
    readings: dict[str, Any] = {
        "abstract": _train_abstract(spec, runs, selected, bool(studies), _raw(lane)),
        "per_run": {str(r.get("id")): _run_reading(r, selection.get("run")) for r in runs},
        "selection": _selection_reading(selection, selected)}
    if studies:
        readings["hpo"] = ("A search ran around the baseline under the lane's budget. Its trials sit in the "
                           "ledger like any rung, each with its own digests and diagnostics; pruned trials are "
                           "marked apart and kept out of the verdict. At this budget the importances are a hint "
                           "about where to look next, not a measurement.")
    failed = [r for r in runs if r.get("status") not in ("done", "pruned")]
    markers = ["OK: every run in the ledger was trained and recorded by the trainer; its history is on disk"
               if not failed else
               "WARN: a run in the ledger did not finish; the ledger keeps its error rather than a curve",
               "OK: the selection names a run of the ledger" if selected else
               "WARN: the selection names no run of the ledger"]
    override = bool(selection.get("override"))
    forks = [{"asked": "the ladder as planned: the baseline, the rungs the lane declares and the search, under the "
                       "lane's budget", "or": "default ladder applied (unattended)"},
             {"asked": "the selection: the ladder's rule, or a run picked by hand",
              "answer": "an override, recorded with its reason" if override else "the rule"}]
    return {"schema": "vega/training@1", "from_campaign": ".", "forks": forks, "markers": markers,
            "readings": readings}


def _train_abstract(spec: Mapping[str, Any], runs: Sequence[Mapping], selected: Mapping | None,
                    searched: bool, lane: Mapping[str, Any]) -> str:
    family = words(_name(spec.get("family")) or lane.get("family"))
    kinds = {r.get("kind") for r in runs}
    parts = ["the baseline"] + (["the ladder's one-change rungs"] if "ablation" in kinds else []) \
        + (["a search around the baseline"] if searched else [])
    pick = RUN_KIND.get(str((selected or {}).get("kind")), "a run of the ledger") if selected else None
    text = (f"The campaign trained the {family} network the forge engineered: {_listing(parts)}, each under the "
            "same seed, data and budget, so a difference between two runs is the change between them. ")
    text += (f"The selection rule picked {pick}. " if pick else "No run was selected. ")
    text += ("Nothing in this reading restates a figure: the ledger, the curves and the selection record beside it "
             "carry them, and which runs the rule excluded, and why, is in the ledger.")
    return text


def _run_reading(run: Mapping[str, Any], selected: Any) -> str:
    text = _sentence(RUN_KIND.get(str(run.get("kind")), "a run of the ledger"))
    if run.get("kind") == "ablation" and run.get("family"):
        text += f", changing a single {words(run.get('family'))} choice against the baseline"
    text += ". "
    status = run.get("status")
    if status == "done":
        text += _sentence(STOPPED_BY.get(str(run.get("stopped_by")), "it finished")) + ". "
    elif status == "pruned":
        text += "The search pruned it, so its history is partial and the rule keeps it out of the verdict. "
    else:
        text += "It did not finish; the ledger keeps its error rather than a curve. "
    fired = [words(d.get("name")) for d in run.get("diagnostics_detail") or []
             if isinstance(d, Mapping) and d.get("fired") and d.get("name")]
    text += (f"Diagnostics that fired: {_listing(fired)}; the evidence is in the ledger. " if fired
             else "No diagnostic fired. ")
    if run.get("eligible") is False:
        text += "The selection rule excludes it, and the ledger says why. "
    if run.get("id") == selected:
        text += "The selection rule picked it."
    return text.strip()


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _selection_reading(selection: Mapping[str, Any], selected: Mapping | None) -> str:
    if not selected:
        return "No run was selected, so the campaign has not reached a verdict."
    kind = RUN_KIND.get(str(selected.get("kind")), "a run of the ledger")
    text = (f"The rule picks the run with the best valid primary metric in the metric's direction, fewer "
            f"parameters breaking a tie, among the runs it allows; it picked {kind}. Its margin over the "
            "baseline is in the selection record, not restated here.")
    if selection.get("override"):
        text += " An override is recorded with its reason, in the open, and it is what shipped."
    return text


# =========================================================================== pack
def pack_authoring(bundle: Path, lane: Any) -> dict:
    """The /vega-pack judgement for the bundle in `bundle`: the abstract and the intended use
    (no number, which the card and the layer refuse), and the marker that says no architect
    run judged this bundle. `from_bundle` is `bundle/<slug>`, relative to the lane's work dir,
    where the pack stage runs the apply script and the model page lives."""
    root = Path(bundle)
    manifest = _json_or_none(root / "MANIFEST.json") or {}
    identity = (manifest.get("serve") or {}).get("identity") or {}
    raw = _raw(lane)
    row = (raw.get("lexicon") or {}).get("row") or {}
    noun = _noun(identity.get("noun") or row.get("one"), "row")
    nouns = _noun(identity.get("nouns") or row.get("many"), "rows" if noun == "row" else noun + "s")
    stance = bool(manifest.get("stances"))
    abstract = (f"This bundle ships the network the campaign selected together with the shaper's own "
                f"transform, so a raw {noun} goes in and the answer the evaluation stored comes out; the round "
                "trip beside this text is the proof, over the stored test predictions it records. It was packed "
                "by the lane program, which ships every lane whatever its verdict to prove the chain end to end"
                + ("; the stance it ships under is recorded in the manifest." if stance else "."))
    intended = [
        f"Use this bundle to answer for each {noun} given in the input schema this page lists. It answers "
        f"in the order the {nouns} are given, and every answer carries its warnings, so a caller can join "
        "the answers back and see which ones to distrust.",
        "Do not use it outside the ranges the training data covered. The schema table carries the range seen "
        "for every numeric column and the vocabulary seen for every categorical one; an answer beyond them is "
        "extrapolation that nothing measured, and an unseen category lands on the reserved code the shaper set "
        "aside for it.",
        ("It ships by a recorded stance rather than on its verdict: the evaluation layer says whether the "
         "network earned its cost against the baselines, and a caller who needs only what a baseline gives "
         "should take the baseline." if stance else
         "Read the evaluation layer's verdict before relying on it: it says whether the network earned its "
         "cost against the baselines trained on the same shaped inputs."),
        "This bundle is a test artifact of the lane program, packed to prove the chain end to end; it is not "
        "a model anyone chose to deploy."]
    # relative to the lane's work dir (where the stage runs and the model page lives), so the
    # layer's card link resolves from the page; "." for a bundle outside a bundle/ directory
    where = f"{root.parent.name}/{root.name}" if root.parent.name == "bundle" else "."
    return {"schema": "vega/bundle-layer@1", "from_bundle": where,
            "readings": {"abstract": abstract, "intended_use": "\n\n".join(intended)},
            "markers": ["WARN: no /architect run was made on this bundle; the mechanical half of the conformance "
                        "gate ran and passed, and the judgement half is absent rather than clean",
                        "WARN: no /improve run was made on this bundle; the method-level half of the judgement "
                        "is absent rather than clean"],
            "code_findings": []}


def _noun(word: Any, default: str) -> str:
    """A noun for layer prose; a missing one, or one carrying a digit, is the default."""
    text = str(word or "").strip()
    return text if text and not any(ch.isdigit() for ch in text) else default


# =========================================================================== show (the story plan)
HEADLINES = {"binary": "What a network answers for each {row.one}, and how sure it is",
             "classes": "Which class a network gives each {row.one}, and how sure it is",
             "labels": "Which labels a network puts on each {row.one}",
             "ordinal": "Where a network ranks each {row.one} on the scale, and how sure it is",
             "value": "The value a network puts on each {row.one}",
             "horizon": "What comes next, as a network reads the {row.many} before it",
             "reconstruction": "How a network rebuilds each {row.one}, and what it cannot rebuild",
             "anomaly": "Which {row.many} a network finds out of the ordinary"}
LEAD = ("Every {row.one} {source.the} records passes through the recipe and then the network. This page follows "
        "it from the raw table to the answer the service gives, and every figure on it comes from the records "
        "of the run that shipped.")
CONTEXT_STEPS = {
    "problem": "The task: for each {row.one}, the network gives the answer the page is about. The records come "
               "from {source.the}, and each {row.one} is read on its own terms, by the {feature.many} it carries.",
    "data": "The raw table holds a {row.one} per line. The shaper kept the {feature.many} the network reads, "
            "typed and encoded as its recipe records, and set the rest aside with the reason for each.",
    "windows": "The network reads windows: a run of consecutive {step.many} of the same series ends at each "
               "{row.one} it answers for, and the {window.one} never crosses from a series into another.",
    "graph": "The {row.many} are {node.many} of a graph whose {edge.many} the recipe built; each answer reads a "
             "{node.one} together with its neighbours, and a new {node.one} is answered from the reference "
             "graph alone.",
    "cleaning": "The recipe cleaned the table before the split. Every step it took is recorded with the "
                "{row.many} it touched, and the same steps run again on every request the service answers.",
    "split": "The {row.many} were split before any model saw them, into a train part, a valid part and a sealed "
             "test part that the evaluation opened a single time."}
CONTEXT_FACTS = (("source_rows", "{row.many} in the source"), ("train_rows", "{row.many} to train on"),
                 ("valid_rows", "{row.many} to select on"), ("test_rows", "{row.many} sealed for the test"),
                 ("shaped_inputs", "{feature.many} the network reads"))
CHOICE_TITLES = {"family": "The family", "blocks": "The blocks", "loss": "The loss", "metrics": "The metrics",
                 "optimizer": "The optimizer", "schedule": "The schedule", "training": "The training run",
                 "init": "The initialisation", "regularization": "The regularization",
                 "early_stopping": "Early stopping"}
FAMILY_TITLES = {"architecture": "The architecture", "objective": "The objective",
                 "optimization": "The optimization", "regularization": "The regularization"}
FAMILY_TEXT = {"architecture": "A choice about the shape of the network.",
               "objective": "A choice about what the network is trained to get right.",
               "optimization": "A choice about how the weights move while it trains.",
               "regularization": "A choice that keeps the network from memorising the train part."}
TYPE_TEXT = {
    "dense": "It mixes every input it receives into each of its units.",
    "embedding": "It turns each categorical code into a short learned vector.",
    "concat": "It joins the numeric inputs and the embeddings into a single vector.",
    "batchnorm": "It rescales each unit by the batch's statistics while training.",
    "layernorm": "It rescales each sample across its units.",
    "dropout": "It silences a random share of units while training, never when answering.",
    "activation": "It bends the signal, so the stack can model more than a straight line.",
    "residual": "It adds its input back to its output, so a deeper stack still trains.",
    "sequential": "It runs its inner blocks in order as a single unit.",
    "parallel": "It runs its inner branches side by side and joins their outputs.",
    "conv1d": "It slides a learned filter along the sequence.",
    "pool1d": "It summarises neighbouring steps of the sequence.",
    "rnn": "It reads the sequence step by step, carrying a state.",
    "gru": "It reads the sequence step by step, gating what its state keeps.",
    "lstm": "It reads the sequence step by step with a memory cell and its gates.",
    "attention": "It lets each step weigh every other step.",
    "attention_pool": "It weighs the steps and sums them into a single vector.",
    "transformer_encoder": "It stacks attention and feed-forward layers over the sequence.",
    "feature_tokenizer": "It turns each input into a token the attention layers read.",
    "positional": "It tells the network where each step sits in the sequence.",
    "flatten": "It lays the sequence out as a single long vector.",
    "reshape": "It changes the tensor's shape without changing its values.",
    "pool": "It summarises the whole sequence into a single vector.",
    "head": "It turns the last representation into the answer the task asks for.",
    "decoder": "It rebuilds the inputs from the compressed representation.",
    "reparameterize": "It samples the latent code, so the network can learn its spread.",
    "gcn_conv": "It averages each node's neighbourhood through a learned map.",
    "sage_conv": "It combines each node with an aggregate of its neighbours.",
    "gat_conv": "It weighs each neighbour by learned attention.",
    "gin_conv": "It sums each neighbourhood and passes it through a small network.",
    "graph_pool": "It summarises each graph's nodes into a single vector.",
    "edge_dropout": "It drops a random share of edges while training.",
    "custom": "It is the user's own block, loaded from the file the spec names."}
LIMITS = {
    "split": ("How the parts were cut", "The {row.many} were split into train, valid and test parts before any "
              "model saw them; every figure on this page is read on those parts, and another cut would move each "
              "of them."),
    "exposures": ("The test part was opened once", "The sealed test part was opened a single time, by the "
                  "evaluation. Nothing on this page was tuned on it, and a second look would spend what that "
                  "single look is worth."),
    "one_seed": ("A single seed", "Every run was trained from a single seed. The intervals come from resampling "
                 "the test part, not from retraining, so they leave out the spread another seed would bring."),
    "robustness_not_run": ("Robustness was not tested", "The evaluation ran no robustness check on this network, "
                           "so nothing here says how its answers move when the inputs shift."),
    # verdict_not_earned and verdict_beaten: _verdict_limit, by the record
    "ship_stance": ("Shipped by a stance", "The bundle ships because a recorded stance says so, not because the "
                    "verdict earned it; the stance, quoted below as the bundle's manifest records it, is the "
                    "reason."),
    "small_support": ("Rare classes are anecdotes", "Some classes have only a handful of {row.many} in the sealed "
                      "test part, listed below with their counts; their per-class scores are anecdotes, not "
                      "measurements."),
    "roundtrip_boundary": ("Answers at the start of a part", "At the start of each part the evaluation's lags "
                           "restart while served answers read the history, so the first {row.many} of a part are "
                           "answered from more context than the evaluation had."),
    "examples_synthetic": ("Synthetic examples", "The example {row.many} the page offers are synthetic, drawn to "
                           "cover the input ranges, not {row.many} of the data."),
    "calibration": ("Calibration", "A score is only as good as its calibration: the evaluation's reliability "
                    "curve says how far the scores sit from the frequencies they claim."),
    "slices_empty": ("No slices", "The evaluation found no slice worth reading apart, so the whole-part figures "
                     "are the only view of how the network behaves."),
    "drift_suspected": ("Drift suspected", "The analysis flagged {feature.many} whose distribution moves across "
                        "the parts; answers on such {row.many} rest on less evidence than the averages suggest."),
    "serving_forced": ("Served for the demonstration", "The blueprint did not ask for an online service; this "
                       "service runs so the page can show the network answering.")}
WARN_LIMIT = ("A warning the chain recorded", "A stage recorded the warning quoted below; the page repeats it "
              "rather than soften it.")
# the verdict limits, written by what the evaluation recorded (never a softer word than the record's): a
# baseline that did better is said to have done better, and only an interval that holds zero is "could not
# separate" (gate C: every beaten lane said a cheaper model "does as well")
VERDICT_LIMITS = {
    ("verdict_not_earned", "beaten"): ("The network did not earn its cost", "A baseline trained on the same "
                                       "inputs did better than the network on the sealed test part, so the "
                                       "network does not earn its cost; the network minus that baseline is "
                                       "{diff}."),
    ("verdict_not_earned", "level"): ("The network did not earn its cost", "Against the strongest baseline "
                                      "trained on the same inputs, the evaluation could not separate the network "
                                      "from it: the difference {diff} has an interval that holds zero."),
    ("verdict_beaten", "beaten"): ("A baseline did better", "A baseline trained on the same shaped inputs scored "
                                   "better than the network on the sealed test part: the network minus that "
                                   "baseline is {diff}.")}
WALK_NOTES = {
    "root": "Where the package starts: the module map the blueprint drew, a file per port.",
    "store": "How the bundle loads its weights and checks their digests before it answers.",
    "build": "Where the network is assembled from the spec's blocks, in forward order.",
    "forward": "The forward pass: the blocks applied in order, from the inputs to the head.",
    "link": "How the head's raw output becomes the answer the service returns.",
    "shape": "The shaper's recipe, replayed on raw {row.many} before the network sees them.",
    "run": "The call that turns a request's {row.many} into answers.",
    "explain": "How the service traces a {row.one} through every block for the explain view.",
    "route": "The route the service answers on, and the checks it runs first."}
GROUP_TEXT = {"created": ("made by the recipe", "{feature.many} the recipe created from the raw table, not read "
                                                "from it"),
              "encoded": ("categorical codes", "categorical {feature.many}, encoded to integers the network "
                                               "embeds or reads"),
              "correlation": ("moving together", "{feature.many} that move together in the analysis, read as a "
                                                 "group"),
              "name-token": ("sharing a name", "{feature.many} whose names share a word"),
              None: ("the remaining inputs", "{feature.many} the network reads as the recipe left them")}
MONITOR_NOTE = ("The service keeps a reference of the training data and flags the {row.many} whose "
                "{feature.many} drift away from it.")


def show_authoring(facts: Mapping[str, Any], lane: Any, draft: Mapping[str, Any], *,
                   out: Callable[[str], None] | None = None) -> dict:
    """The story plan the model would write from `draft` (vega/story-plan@2, status draft):
    a copy with `authored` filled under the contract (show-contract §3) and status `authored`.
    The lexicon is the lane's, the motif its expected one when admissible (else the base), the
    class roles its declared roles with a basis the rules accept; every prose slot is written
    from the draft's proposals with lexicon and fact placeholders only — no digit.

    An alert role the draft found no `show:` stance for falls back to the lane registry as the
    registry (`registry:<lane>`, never a user's turn) — the plan has no slot for a marker, so the fallback is said to `out`
    (default: stdout, which is the stage log when the lane helper calls this) as a `WARN:`."""
    raw = _raw(lane)
    plan = copy.deepcopy(dict(draft))
    props = plan.get("proposals") or {}
    for line in role_warnings(props, raw):
        (out or _say)(line)
    lexicon = {k: {p: v[p] for p in ("one", "many", "article")} for k, v in (raw.get("lexicon") or {}).items()}
    plan["authored"] = {
        "locale": "en",
        "lexicon": lexicon,
        "motif": _motif(props, raw),
        "class_roles": _class_roles(props, raw),
        # the words the registry gives each class ("1" is "late"), else the classes as named
        "class_names": dict((raw.get("target") or {}).get("class_names") or {}) or None,
        "report_roles": _report_roles(props),
        "headline": HEADLINES.get(str(props.get("view_key")), HEADLINES["value"]),
        "lead": LEAD,
        "context": {"steps": {s: CONTEXT_STEPS[s] for s in props.get("context_steps") or [] if s in CONTEXT_STEPS},
                    "facts": _context_facts(facts)},
        "network": {"lead": "The network the forge built for this task, block by block. Each choice below says "
                            "what it does and whether a rung of the ladder tested it.",
                    "choices": _choices(_chosen(props.get("choices") or []))},
        "experiments": {"lead": "Each rung of the ladder changed a single thing against the baseline under the "
                                "same budget, and the selection rule, not a hand, picked the run that shipped.",
                        "caption": "Each run's score on the valid part, in the metric's direction; the baseline "
                                   "is the reference the others are read against.",
                        "labels": {str(r.get("id")): _run_label(r) for r in props.get("runs") or []}},
        "results": {"headline": "How the shipped network did on the sealed test part",
                    "notes": _result_notes(props, facts),
                    "breakdown_caption": ("Each unit on the same sealed test part, read by the metric named "
                                          "above, which need not be the primary; a weak unit falls below the "
                                          "trivial baseline."
                                          if (facts.get("groups") or {}).get("breakdown") else None),
                    "limits": [_limit(l, facts) for l in props.get("limits") or []]},
        "input_groups": _groups(props.get("input_groups") or []),
        "code_notes": {str(s.get("step")): _code_note(str(s.get("step"))) for s in props.get("code_steps") or []},
        "monitor_note": MONITOR_NOTE if _has_monitor(facts) else None}
    plan["status"] = "authored"
    return plan


def _motif(props: Mapping[str, Any], lane: Mapping[str, Any]) -> str | None:
    motif = props.get("motif") or {}
    want = (lane.get("expect") or {}).get("motif")
    return want if want in (motif.get("admissible") or []) else motif.get("base")


def _class_roles(props: Mapping[str, Any], lane: Mapping[str, Any]) -> dict | None:
    """The lane's declared roles, each with a basis V6 accepts: the draft's structural neutral
    when it names that class, task:anomaly on an anomaly task, the recorded show: stance the
    draft found, else the lane registry (a user: basis); plain is vega:default."""
    roles = lane.get("roles")
    if not roles:
        return None
    proposal = props.get("class_roles") or {}
    neutral = proposal.get("neutral") or {}
    found = str(proposal.get("basis_found") or "")
    stance = found if re.fullmatch(r"show:\d{4}-\d{2}-\d{2}", found) else None
    anomaly = lane.get("task") == "anomaly"
    source = {}
    for cls, role in roles.items():
        if role == "neutral":
            basis = (neutral.get("rule") if str(neutral.get("class")) == str(cls) and neutral.get("rule")
                     else "task:anomaly" if anomaly else stance or "vega:default")
        elif role == "alert":
            basis = "task:anomaly" if anomaly else stance or registry_basis(lane)
        else:
            basis = "vega:default"
        source[str(cls)] = basis
    return {"roles": {str(k): v for k, v in roles.items()}, "source": source}


def role_warnings(props: Mapping[str, Any], lane: Any) -> list[str]:
    """The `WARN:` for the alert roles whose basis is the registry fallback: no `show: roles`
    stance was found in the memory the draft read, so R-E5's 'the alert basis is the stance'
    does not hold for this lane."""
    raw = _raw(lane)
    roles = _class_roles(props, raw) or {}
    fallback = [cls for cls, basis in (roles.get("source") or {}).items() if basis == registry_basis(raw)]
    if not fallback:
        return []
    return [f"WARN: no show: roles stance found for lane {raw.get('id')}: the alert role of "
            f"{_listing(fallback)} rests on the lane registry ({registry_basis(raw)}), not on a recorded stance "
            f"— write it with `python -m lanekit.authoring stances` before the show draft (R-E5)"]


def _say(line: str) -> None:
    print(line, flush=True)


def _report_roles(props: Mapping[str, Any]) -> dict | None:
    """Only a proposed role map is confirmed by the author (V14); an emitted one is the report's."""
    report = props.get("report_roles") or {}
    if report.get("status") != "proposed":
        return None
    return {role: str(b.get("label")) for role, b in (report.get("bindings") or {}).items() if b.get("label")}


def _context_facts(facts: Mapping[str, Any]) -> list[dict]:
    """Two to six size facts: the ones the extractor keyed, in the page's reading order."""
    known = facts.get("facts") or {}
    sizes = (facts.get("groups") or {}).get("sizes") or {}
    keyed = {str(v) for v in sizes.values()} | set(sizes)
    out = [{"label": label, "fact": key} for key, label in CONTEXT_FACTS if key in known and key in keyed]
    if len(out) < 2:
        extra = [k for k, f in known.items() if (f or {}).get("kind") in ("count", "size", "count_of")
                 and k not in {o["fact"] for o in out}]
        out += [{"label": "{row.many} counted", "fact": k} for k in extra[: 2 - len(out)]]
    return out[:6]


def _chosen(choices: Sequence[Mapping[str, Any]], low: int = 3, high: int = 8) -> list[Mapping[str, Any]]:
    """Every mandatory choice, then the others in the draft's order up to the bounds."""
    mandatory = [c for c in choices if c.get("mandatory")]
    rest = [c for c in choices if not c.get("mandatory")]
    picked = list(mandatory)
    for choice in rest:
        if len(picked) >= max(low, min(high, len(mandatory))):
            break
        picked.append(choice)
    return [c for c in choices if any(c is p for p in picked)]


ORDINALS = ("", "second ", "third ", "fourth ", "fifth ", "sixth ", "seventh ", "eighth ", "ninth ", "tenth ")
# a non-block decision's own words, by its type where the family's title would be too broad
DECISION_TYPE_TITLES = {"weight_decay": "Weight decay", "grad_clip": "Gradient clipping",
                        "label_smoothing": "Label smoothing", "dropout": "Dropout",
                        "edge_dropout": "Edge dropout", "early_stopping": "Early stopping"}
DECISION_TYPE_TEXT = {
    "weight_decay": "It pulls every weight a little towards zero at each step.",
    "grad_clip": "It caps the size of each update, so one bad batch cannot throw the weights far.",
    "label_smoothing": "It keeps the targets a little short of certainty.",
    "early_stopping": "It stops the training when the valid score stops improving."}


def _choices(choices: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Each choice's words, a repeated title told apart by its order ("The second dense layer")."""
    out, seen = [], {}
    for choice in choices:
        made = _choice(choice)
        title = made["title"]
        n = seen.get(title, 0)
        seen[title] = n + 1
        if n and n < len(ORDINALS):
            made["title"] = "The " + ORDINALS[n] + title[len("The "):] if title.startswith("The ") \
                else f"{title}, the {ORDINALS[n].strip()}"
        out.append(made)
    return out


def _choice(choice: Mapping[str, Any]) -> dict:
    """A decision's words: a block of the architecture by its type (it is a block of the network),
    any other decision by its family and type — never called a block (gate C: the loss, the
    optimizer and the weight decay were all "The block … It is a block of the network")."""
    decision, family, btype = str(choice.get("decision")), choice.get("family"), choice.get("type")
    base = decision.split("[")[0].split(".")[0]
    block = family == "architecture" and re.fullmatch(r"B\d+", decision) is not None
    if block:
        title = "The " + BLOCK_WORDS.get(str(btype), "block")
    else:
        title = (DECISION_TYPE_TITLES.get(str(btype)) or CHOICE_TITLES.get(base)
                 or FAMILY_TITLES.get(str(family), "The choice"))
    text = FAMILY_TEXT.get(str(family), "A choice the forge made.")
    if block:
        text += " " + TYPE_TEXT.get(str(btype), "It is a block of the network.")
    elif str(btype) in DECISION_TYPE_TEXT:
        text += " " + DECISION_TYPE_TEXT[str(btype)]
    settings = [(p, k) for p, k in (choice.get("params") or {}).items()
                if isinstance(k, str) and not any(ch.isdigit() for ch in str(p))][:3]
    if settings:
        text += " Its settings: " + ", ".join(f"{words(p)} {{{k}}}" for p, k in settings) + "."
    text += (" A rung of the ladder tested it." if choice.get("tested_by")
             else " No rung tested it; it stands as the forge set it.")
    return {"decision": decision, "title": title, "text": text}


def _run_label(run: Mapping[str, Any]) -> str:
    """The draft's mechanical label when it carries no digit, else the run's role in words."""
    label = str(run.get("label") or "").strip()
    if label and not any(ch.isdigit() for ch in label) and len(label) <= 60:
        return label
    if run.get("selected"):
        return "the run that shipped"
    if run.get("reference"):
        return "the baseline"
    return "a trial of the search" if re.match(r"^H\w*-", str(run.get("id"))) else "a rung of the ladder"


def _result_notes(props: Mapping[str, Any], facts: Mapping[str, Any]) -> dict:
    primary = f"eval_network_{(facts.get('task') or {}).get('primary')}"
    notes = {}
    for key in props.get("results_facts") or []:
        notes[str(key)] = ("The primary metric, which selection read on the valid part, measured on the sealed "
                           "test part." if key == primary else
                           "Every figure of this view was read on these {row.many}." if key == "test_rows" else
                           "A secondary metric of the same task, measured on the same sealed test part.")
    return notes


def _limit(limit: Mapping[str, Any], facts: Mapping[str, Any] | None = None) -> dict:
    lid = str(limit.get("id"))
    title, text = _verdict_limit(lid, facts or {}) or LIMITS.get(lid) or WARN_LIMIT
    return {"id": lid, "title": title, "text": text}


def _verdict_limit(lid: str, facts: Mapping[str, Any]) -> tuple | None:
    """A verdict limit's words by the record: beaten (the interval excludes zero the other way),
    else level (it holds zero); the difference cited as its fact, or in words without one."""
    if lid not in ("verdict_not_earned", "verdict_beaten"):
        return None
    verdict = (facts.get("groups") or {}).get("verdict") or {}
    state = "beaten" if verdict.get("beaten_by_baseline") else "level"
    title, text = VERDICT_LIMITS.get((lid, state)) or VERDICT_LIMITS[(lid, "beaten")]
    diff = verdict.get("diff")
    cited = f"{{{diff}}}" if diff and diff in (facts.get("facts") or {}) else "not recorded"
    return title, text.replace("{diff}", cited)


def _groups(groups: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Every group's words (`_group`), a label two groups share told apart by its order ("the
    remaining inputs", "a second set: the remaining inputs") — gate C: four groups of one lane
    were all "the remaining inputs"."""
    out, seen = [], {}
    for group in groups:
        made = _group(group)
        n = seen.get(made["label"], 0)
        seen[made["label"]] = n + 1
        if n:
            made["label"] = f"a {ORDINALS[min(n, len(ORDINALS) - 1)].strip()} set: {made['label']}"[:40]
        out.append(made)
    return out


def _group(group: Mapping[str, Any]) -> dict:
    """A group's words: its evidence's text, labelled by its members' names when they carry no
    digit (the lanes' authored strings hold none), else by its evidence (a name token the group
    shares, or the evidence kind's words)."""
    kinds = [e.get("kind") for e in group.get("evidence") or [] if isinstance(e, Mapping)]
    kind = next((k for k in kinds if k in GROUP_TEXT), None)
    label, text = GROUP_TEXT[kind]
    gid, members = str(group.get("suggested_id") or ""), [str(m) for m in group.get("members") or []]
    token = next((str(e.get("token")) for e in group.get("evidence") or []
                  if isinstance(e, Mapping) and e.get("kind") == "name-token" and e.get("token")), "")
    spelled = [m.replace("_", " ") for m in members]
    if members and len(members) <= 3 and not any(ch.isdigit() for m in members for ch in m):
        label = spelled[0] if len(spelled) == 1 else ", ".join(spelled[:-1]) + " and " + spelled[-1]
    elif token and not any(ch.isdigit() for ch in token):
        label = f"sharing the word {token}"
    return {"id": gid, "label": label[:40], "text": text, "members": members}


def _code_note(step: str) -> str:
    if step in WALK_NOTES:
        return WALK_NOTES[step]
    if step == "custom":
        return "The user's own block, loaded from the file the spec names, and what it computes."
    return f"Where the {BLOCK_WORDS.get(step, 'block')} is defined and what it computes."


def _has_monitor(facts: Mapping[str, Any]) -> bool:
    """MANIFEST.monitor present: read from the bundle the facts name, else from the facts' refs."""
    bundle = Path(str((facts.get("bundle") or {}).get("dir") or ""))
    manifest = _json_or_none(bundle / "MANIFEST.json") if str(bundle) not in ("", ".") else None
    if isinstance(manifest, Mapping):
        return bool(manifest.get("monitor"))
    return any(str((f or {}).get("ref", "")).startswith("MANIFEST.json#monitor")
               for f in (facts.get("facts") or {}).values())


# =========================================================================== stances (R-E5)
def show_stances(lane: Any, *, dataset: str | None = None, date: str | None = None) -> list[str]:
    """The `show:` lines a lane's declarations put in § Network stances: a roles line for a
    lane that declares `alert` (or states a `show: roles` stance), a motif line per `show:
    motif` stance. `dataset` defaults to the lane's generator (the model's file stem)."""
    raw = _raw(lane)
    day = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    name = dataset or raw.get("generator") or raw.get("id")
    stances = [str(s) for s in raw.get("stances") or []]
    roles = raw.get("roles") or {}
    lines = []
    if roles and ("alert" in roles.values() or any(s.startswith("show: roles") for s in stances)):
        parts = ", ".join(f"{cls} {role} ({_stance_basis(str(cls), role, raw)})" for cls, role in roles.items())
        lines.append(f"{day} | {name} | show: roles {parts} (user)")
    for stance in stances:
        motif = re.fullmatch(r"show: motif (\S+) \(user\)", stance.strip())
        if motif:
            lines.append(f"{day} | {name} | show: motif {motif.group(1)} (user)")
    return lines


def _stance_basis(cls: str, role: str, lane: Mapping[str, Any]) -> str:
    """The per-class basis a roles line records: the structural rule where the task makes it
    one (the negative class of a binary task, either side of an anomaly task), else the user
    (`registry:<lane>`, the vega-show catalog's basis for a matrix lane's declaration)."""
    task = lane.get("task")
    target = lane.get("target") or {}
    if task == "anomaly":
        return "task:anomaly"
    binary = task == "binary" or (task == "sequence-classification" and target.get("base") == "binary")
    if role == "neutral" and binary and target.get("positive") is not None and cls != str(target["positive"]):
        return "structural:negative"
    return registry_basis(lane)


def write_show_stances(memory: Path, lane: Any, *, dataset: str | None = None, date: str | None = None) -> list[str]:
    """Append the lane's `show:` lines to § Network stances of `memory` (created, with the
    section, when absent). A line whose body (everything after the date) is already there is
    not written again. Returns the lines written."""
    path = Path(memory)
    text = path.read_text(encoding="utf-8") if path.is_file() else "# domain-forge memory\n"
    present = {_body(line) for line in _section_lines(text)}
    new = [line for line in show_stances(lane, dataset=dataset, date=date) if _body(line) not in present]
    if not new:
        return []
    heading = re.search(rf"^#+\s*{STANCE_SECTION}\s*$", text, re.I | re.M)
    block = "".join(f"- {line}\n" for line in new)
    if heading is None:
        text = text.rstrip("\n") + f"\n\n## {STANCE_SECTION}\n\n" + block
    else:
        tail = text[heading.end():]
        nxt = re.search(r"^#+\s", tail, re.M)
        cut = heading.end() + (nxt.start() if nxt else len(tail))
        head, rest = text[:cut].rstrip("\n"), text[cut:]
        text = head + "\n" + block + ("\n" + rest if rest else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return new


def _body(line: str) -> str:
    return line.split("|", 1)[1].strip() if "|" in line else line.strip()


# =========================================================================== the digit discipline
IDENTIFIER_FIELDS = frozenset({"schema", "from_run", "from_campaign", "from_bundle", "path", "id", "family",
                               "fact", "decision", "members", "locale", "motif"})
IDENTIFIER_SUBTREES = frozenset({"class_roles", "report_roles", "lexicon"})


def prose(doc: Any, kind: str) -> list[tuple[str, str]]:
    """(where, text) for every authored string of an authoring document: for the story plan,
    the `authored` half; for a layer, the whole document. Dictionary keys, identifier fields
    (ids, paths, fact keys, decisions, members, the closed-set values) and the class-role,
    report-role and lexicon maps (closed sets and labels) are not prose."""
    if kind not in KINDS:
        raise ValueError(f"unknown authoring kind {kind!r}; one of {KINDS}")
    root, where = (doc.get("authored") or {}, "authored") if kind == "show" else (doc, kind)
    out: list[tuple[str, str]] = []

    def walk(node: Any, at: str, field: str | None) -> None:
        if isinstance(node, str):
            if field not in IDENTIFIER_FIELDS:
                out.append((at, node))
        elif isinstance(node, Mapping):
            for key, value in node.items():
                if key not in IDENTIFIER_SUBTREES:
                    walk(value, f"{at}.{key}", str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{at}[{i}]", field)

    walk(root, where, None)
    return out


def digit_problems(doc: Any, kind: str) -> list[str]:
    """Every decimal digit (Unicode Nd) in an authored string, outside a `{placeholder}`."""
    problems = []
    for where, text in prose(doc, kind):
        bare = re.sub(PLACEHOLDER, "", text) if kind == "show" else text
        digits = [ch for ch in bare if unicodedata.category(ch) == "Nd"]
        if digits:
            problems.append(f"{where}: digit {digits[0]!r} outside a placeholder in {text[:80]!r}")
    return problems


def number_words(doc: Any, kind: str, vocabulary: Iterable[str]) -> list[str]:
    """The number words (the vega-show catalog's list) in authored prose: the digit rule's
    `WARN:` cases, which a story should not carry either."""
    vocab = {w.lower() for w in vocabulary}
    hits = []
    for where, text in prose(doc, kind):
        bare = re.sub(PLACEHOLDER, " ", text)
        for token in re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", bare):
            if token.lower() in vocab:
                hits.append(f"{where}: number word {token!r}")
    return hits


# =========================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lanekit.authoring", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    st = sub.add_parser("stances", help="write a lane's show: stance lines into a memory file")
    st.add_argument("--lanes-file", type=Path, default=None)
    st.add_argument("--lane", required=True)
    st.add_argument("--memory", required=True, type=Path)
    st.add_argument("--dataset")
    st.add_argument("--date")
    ck = sub.add_parser("check", help="refuse an authoring file with a digit outside a placeholder")
    ck.add_argument("--kind", required=True, choices=KINDS)
    ck.add_argument("file", type=Path)
    args = ap.parse_args(argv)
    if args.command == "stances":
        from .registry import LANES_FILE, load_lanes
        lane = load_lanes(args.lanes_file or LANES_FILE)[args.lane]
        written = write_show_stances(args.memory, lane, dataset=args.dataset, date=args.date)
        for line in written:
            print(f"OK: stance written: {line}")
        if not written:
            print(f"OK: no show: stance to write for {args.lane} (none declared, or already recorded)")
        return 0
    doc = _json(args.file)
    problems = digit_problems(doc, args.kind)
    for problem in problems:
        print(f"ERROR: {problem}")
    if not problems:
        print(f"OK: {args.file.name}: no digit outside a placeholder ({len(prose(doc, args.kind))} strings)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
