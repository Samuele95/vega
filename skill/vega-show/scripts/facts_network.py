"""
facts_network — `NetworkFacts`: what the shipped network is (contract §2.3, the groups `network`
and `decisions`), and `task_shape()`, the `task` object of the fact set.

The network is the one the bundle ships: its `netspec.json` (the selected run's spec), not the
forged one the ladder started from. Decisions are one per top-level block id, plus `family`,
`loss`, `metrics`, `optimizer`, `schedule`, `training`, `init`, `early_stopping` and one
`regularization[<kind>]` per regularizer — each with its decision family (the vega-forge
catalog's `decision_families`), its scalar settings as facts (`block_b1_units`,
`loss_label_smoothing`, `optimizer_lr`, `regularization_dropout_value`; a word such as `gelu` is a
text fact), the ref of its rationale, how many alternatives the forge weighed
(`<decision>_alternatives_n`), the runs whose delta touched it (`tested_by`) and the inputs the
changes of those runs that belong to it introduced (`features_tested`, `facts_campaign`'s rule:
R6's masking block is a `family` change that names `Destination Port`).

**Gate PF7.** The task kind, the family, every block type (composites' `inner` blocks included)
and the sample shape must be known to the vega-forge catalog; an unknown one is refused, never
skipped (D5's seeded defects 36 and 37). The task's primary is the one the evaluation recorded,
and its direction the vega-forge `metric_direction`'s (`max` or `min`, else refused).

**The task.** `goal` is the inverse of the catalog's `goal_map` (`unsupervised` → reconstruct,
`anomaly` → anomaly, else predict); `modality` is `spatial-row` when the analysis records a
spatial context on a row sample; `view_key` is the catalog's (a `sequence-*` task takes its
base's). `outcomes` names the two sides of a two-sided task, negative (or normal) first:
`[negative, positive]` for binary (the positive is `task.positive`, else the second class) and
the anomaly sides — the class names when a label exists, else the catalog's `normal`,
`anomalous`; it is null for every other task.
"""
from __future__ import annotations

from typing import Mapping

from facts_campaign import RUNS, input_names, run_changes
from fact_sink import Catalogs, FactSink, Refusal, peek, spellable
from show_ports import RecordStore
from vega_show_core import TaskShape

__all__ = ["NetworkFacts", "task_shape", "block_types"]

NETSPEC = "netspec.json"
_SETTINGS = {"loss": "params", "optimizer": "params", "schedule": "params", "training": None,
             "early_stopping": None}
_ORDER = ("family", "loss", "metrics", "optimizer", "schedule", "training", "init",
          "early_stopping")
_TYPE_FIELD = {"family": "name", "loss": "name", "metrics": "primary", "optimizer": "name",
               "schedule": "name", "init": "scheme"}
_CLASS_TASKS = ("binary", "multiclass", "multilabel", "ordinal", "sequence-classification",
                "sequence-regression", "anomaly")


def block_types(blocks) -> list[str]:
    """Every block type of a block list, composites' `inner` blocks included, in order."""
    out = []
    for block in blocks if isinstance(blocks, list) else []:
        if isinstance(block, list):
            out += block_types(block)
        elif isinstance(block, Mapping):
            out.append(str(block.get("type")))
            out += block_types((block.get("params") or {}).get("inner"))
    return out


def task_shape(store: RecordStore, catalog: Catalogs) -> TaskShape:
    """The fact set's `task`, after gate PF7."""
    task = peek(store, f"{NETSPEC}#task", {}) or {}
    kind, forge = task.get("kind"), catalog.forge
    sample = peek(store, f"{NETSPEC}#data_contract.sample", {}) or {}
    family = (peek(store, f"{NETSPEC}#family", {}) or {}).get("name")
    _pf7(kind, family, sample.get("shape", "row"), catalog)
    _pf7_blocks(peek(store, f"{NETSPEC}#blocks", []), catalog)
    base = task.get("base_kind") or \
        (forge["tasks"][kind].get("base") if kind == "forecast" else None)
    if str(kind).startswith("sequence-") and base not in forge["sequence_map"]:
        raise Refusal("PF7", f"{NETSPEC}#task.base_kind", f"a {kind} task needs a base_kind the "
                      f"vega-forge sequence_map knows, not {base!r}")
    primary = peek(store, "records/evaluation.json#metrics.primary") or \
        peek(store, f"{NETSPEC}#metrics.primary")
    direction = catalog.direction(primary)
    if direction is None:
        raise Refusal("PF7", "records/evaluation.json#metrics.primary", f"the primary {primary!r} "
                      "has no max/min direction in the vega-forge metric_direction")
    shape = sample.get("shape", "row")
    try:
        return TaskShape(kind=kind, base=base if base in forge["sequence_map"] else None,
                         goal=_goal(kind, forge), sample_shape=shape,
                         graph_level=sample.get("level", "node") if shape == "graph" else None,
                         modality=_modality(store, shape), family=family,
                         view_key=_view_key(kind, base, catalog), label=task.get("label"),
                         classes=_classes(task, kind), outcomes=_outcomes(task, kind, catalog),
                         primary=primary, direction=direction)
    except ValueError as exc:  # a record value the fact-set schema does not admit
        raise Refusal("PF2", f"{NETSPEC}#task", str(exc)) from exc


class NetworkFacts:
    """The shipped network's structure and its decisions."""

    def __init__(self, records: RecordStore, catalog: Catalogs) -> None:
        self._store, self._catalog = records, catalog

    def collect(self, sink: FactSink) -> None:
        s = self._store
        blocks = [b for b in peek(s, f"{NETSPEC}#blocks", []) or [] if isinstance(b, Mapping)]
        family = (peek(s, f"{NETSPEC}#family", {}) or {}).get("name")
        sink.name([family, *(b.get("id") for b in blocks)])
        smoke = "records/architecture.json#smoke"
        sink.group("network", {
            "family": family,
            "blocks": [{"id": b.get("id"), "type": b.get("type")} for b in blocks],
            "params_total": sink.take(s, "params_total", f"{smoke}.params_total", "size"),
            "madds": sink.take(s, "madds_per_sample", f"{smoke}.madds_per_sample", "size")})
        tested = self._tested({b.get("id") for b in blocks})
        decisions = [self._decision(sink, d, tested) for d in self._specs(blocks)]
        sink.group("decisions", [d for d in decisions if d is not None])

    def _specs(self, blocks: list) -> list[tuple]:
        """(id, decision family, ref, settings) for every decision the spec makes; `settings`
        is where its scalar settings sit: a key of the node, None for the node itself, `value`
        for a regularizer, False for none."""
        families = {name: fam for fam, names in self._catalog.forge["decision_families"].items()
                    for name in names}
        out = [("family", families.get("family"), f"{NETSPEC}#family", None)]
        for b in blocks:
            if spellable(b.get("id") or "", selector=True):
                out.append((str(b["id"]), families.get("blocks"),
                            f"{NETSPEC}#blocks[id={b['id']}]", "params"))
        for name in _ORDER[1:]:
            out.append((name, families.get(name), f"{NETSPEC}#{name}", _SETTINGS.get(name, False)))
        for reg in peek(self._store, f"{NETSPEC}#regularization", []) or []:
            if isinstance(reg, Mapping) and spellable(reg.get("kind") or "", selector=True):
                out.append((f"regularization[{reg['kind']}]", families.get("regularization"),
                            f"{NETSPEC}#regularization[kind={reg['kind']}]", "value"))
        return out

    def _decision(self, sink: FactSink, spec: tuple, tested: Mapping) -> dict | None:
        did, family, ref, settings = spec
        node = peek(self._store, ref)
        if not isinstance(node, Mapping):
            return None
        prefix = _prefix(did)
        runs, features = tested.get(did, ([], []))
        rationale = f"{ref}.rationale"
        return {"id": did, "family": family, "path": ref.split("#", 1)[1],
                "type": node.get("type") or node.get(_TYPE_FIELD.get(did, "kind")),
                "params": self._settings(sink, (prefix, ref, settings), node),
                "source": node.get("source"),
                "rationale_ref": rationale if isinstance(node.get("rationale"), str) else None,
                "alternatives_n": sink.take(self._store, (*prefix, "alternatives_n"),
                                            f"{ref}.alternatives", "count_of"),
                "tested_by": runs, "features_tested": features}

    def _settings(self, sink: FactSink, where: tuple, node: Mapping) -> dict:
        prefix, ref, path = where
        if path is False:
            return {}
        if path == "value":
            key = sink.take(self._store, (*prefix, "value"), f"{ref}.value", "param")
            return {"value": key} if key else {}
        values = node.get(path) if path else node
        base = f"{ref}.{path}" if path else ref
        out = {}
        for name, value in (values or {}).items() if isinstance(values, Mapping) else ():
            words_ok = path is not None or not isinstance(value, str)
            if name in ("source", "rationale", "catalog", "alternatives") or not spellable(name) \
                    or not words_ok or not _is_setting(value):
                continue
            key = sink.take(self._store, (*prefix, name), f"{base}.{name}",
                            "text" if isinstance(value, str) else "param")
            if key:
                out[name] = key
        return out

    def _tested(self, block_ids: set) -> dict:
        """{decision: (run ids, inputs its changes introduced)} from every run's delta."""
        runs = peek(self._store, RUNS, []) or []
        inputs = input_names(self._store)
        out: dict[str, tuple[list, list]] = {}
        for run in runs:
            if not isinstance(run, Mapping) or not spellable(run.get("id") or "", selector=True):
                continue
            ref = f"{RUNS}[id={run['id']}]"
            for change in run_changes(run, ref, frozenset(block_ids), inputs):
                if change.decision is None:
                    continue
                ids, feats = out.setdefault(change.decision, ([], []))
                if run["id"] not in ids:
                    ids.append(run["id"])
                feats.extend(n for n in change.features if n not in feats)
        return out


def _is_setting(value) -> bool:
    return isinstance(value, (int, float)) or (isinstance(value, str) and bool(value.strip()))


def _prefix(did: str) -> tuple[str, ...]:
    """The fact-key parts a decision's keys start with: `block_b1`, `loss`,
    `regularization_dropout`."""
    if did.startswith("regularization["):
        return "regularization", did[len("regularization["):-1]
    return (did,) if did in _ORDER else ("block", did)


def _pf7(kind, family, shape, catalog: Catalogs) -> None:
    forge = catalog.forge
    for what, value, known in (("task kind", kind, forge["tasks"]),
                               ("family", family, forge["families"]),
                               ("sample shape", shape, forge["sample_shapes"])):
        if value not in known:
            raise Refusal("PF7", NETSPEC, f"the {what} {value!r} is not in the vega-forge catalog "
                                          f"({', '.join(sorted(known))})")


def _pf7_blocks(blocks, catalog: Catalogs) -> None:
    unknown = sorted({t for t in block_types(blocks) if t not in catalog.forge["blocks"]})
    if unknown:
        raise Refusal("PF7", f"{NETSPEC}#blocks", f"block type(s) {unknown} are not in the "
                                                  "vega-forge catalog")


def _goal(kind, forge: Mapping) -> str:
    by_task = {task: goal for goal, task in forge["goal_map"].items() if task}
    return by_task.get(kind, "predict")


def _modality(store: RecordStore, shape: str) -> str:
    spatial = peek(store, "records/analysis.json#context.spatial")
    return "spatial-row" if shape == "row" and spatial is not None else shape


def _view_key(kind, base, catalog: Catalogs) -> str:
    key = catalog.show["view_keys"].get(kind)
    key = key.get(base) if isinstance(key, Mapping) else key
    if key is None:
        raise Refusal("PF7", f"{NETSPEC}#task", f"the vega-show catalog has no view key for the "
                                                f"task {kind!r} (base {base!r})")
    return key


def _classes(task: Mapping, kind) -> tuple | None:
    classes = task.get("classes")
    return tuple(map(str, classes)) if kind in _CLASS_TASKS and isinstance(classes, list) else None


def _outcomes(task: Mapping, kind, catalog: Catalogs) -> list | None:
    classes = [str(c) for c in task.get("classes") or []]
    if kind == "anomaly" and len(classes) != 2:
        return list(catalog.show["roles"]["anomaly_sides"])
    if kind not in ("binary", "anomaly") or len(classes) != 2:
        return None
    positive = str(task["positive"]) if task.get("positive") is not None else classes[1]
    negative = next((c for c in classes if c != positive), classes[0])
    return [negative, positive]
