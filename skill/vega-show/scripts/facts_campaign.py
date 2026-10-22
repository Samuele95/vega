"""
facts_campaign — `CampaignFacts`: the ladder the network climbed (contract §2.3, the groups
`runs`, `selection` and `hpo`), and `run_changes()`, what each run changed in the spec.

A run's `delta` is a list of `{path, from, to}` in the training record, at whatever depth the
ladder wrote it: a scalar (`optimizer/params/lr`), an object (`schedule`, `family`) or the whole
block list (`blocks`, an ablation that inserts a masking block). `run_changes()` walks each entry
down to its leaves — objects by key, block lists by `id`, regularization lists by `kind`,
annotations (`rationale`, `alternatives`, `source`, …) skipped — so every change is one path, one
decision and one before/after pair:

    blocks[id=B1]/params/units   B1        256 → 512           (R9)
    blocks[id=B0]                family    added custom B0     (R6: a block the spec never had)
    loss/params/weight           loss      [...] → null        (R7)

A block the spec has belongs to its own decision; a block added by a run belongs to the
architecture's `family` decision, and so does every block change of a run that changes the
family (the floor rung's blocks are the linear family's, not six separate ablations). A scalar
before or after becomes a fact (`run_<id>_delta_<k>_from`, `…_to`) whose ref points into the
delta itself, so the run's label ("B1, B4 units {run_r9_delta_0_from} → {run_r9_delta_0_to}")
cites the record.

`features_tested` of a run are the network inputs a change of its delta introduces (contract
§2.3), each on the change that names it: a string value of `to` equal to the input's name that
no value of `from` equals, annotations skipped (a mask listing `Destination Port`); or, inside an
item the run adds, the name as a whole word anywhere in it (R6's masking block B0, which names
`Destination Port` only in its rationale) when `from` never names it. A value naming the choice
itself (a `name`, `type`, `kind`, `scheme` or `id`) is a catalog word or an id, never an input:
the floor rung's `family/name: linear` tests no input called `linear`. An input merely mentioned
in the prose of an object a run replaced is not one: the floor rung's linear family calls itself
"the floor every supervised campaign includes", which tests no input called `floor`.

**The `runs` group** (D5 §6.2): one entry per run, `{id, kind, family, primary, epochs,
best_epoch, delta, features, label, selected, reference, eligible, status}`. `delta` holds one
`{path, from, to, decision, change}` per leaf change (`from`/`to` the fact keys of scalar values,
else null; `decision` and `change` say which decision it belongs to and whether it `changed`,
`added` or `removed` a value), with `features` when the change introduces inputs; `features` is
the run's inputs over all its changes, in `data_contract.features` order (R6: `Destination
Port`). `label` is mechanical (`run_label`): the catalog's `delta_label` templates keyed by the
change path's first segment, every scalar value a `{fact key}` placeholder — `without
Destination Port`, `B1, B4 units {run_r9_delta_0_from} → {run_r9_delta_0_to}` — for the model
to rewrite; null for a run that changed nothing (the baseline).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping

from fact_sink import Catalogs, FactSink, peek, spellable
from show_ports import RecordStore

__all__ = ["CampaignFacts", "Change", "run_changes", "decision_of", "features_named",
           "input_names", "run_label"]

TRAINING = "records/training.json"
RUNS = f"{TRAINING}#experiments.runs"
ANNOTATIONS = frozenset({"rationale", "alternatives", "source", "catalog", "formula", "note",
                         "update_rule", "when", "role_source", "consequences"})
_CHOICES = frozenset({"name", "type", "kind", "scheme", "id"})
_NOT_INPUTS = ANNOTATIONS | _CHOICES
_LIST_KEYS = ("id", "kind")
_ROOTS = frozenset({"family", "loss", "metrics", "optimizer", "schedule", "training", "init",
                    "early_stopping"})
_FIRST = re.compile(r"^([A-Za-z_]+)(?:\[(\w+)=([^\]]+)\])?(?:/([^/\[]+))?")


@dataclass(frozen=True)
class Change:
    """One leaf a run changed: its path (the ladder's slash grammar), the decision it belongs
    to, the values before and after, `changed`/`added`/`removed`, the refs of the two values
    inside the training record (None where the grammar cannot spell them), and the inputs the
    change introduces (`features_tested`)."""

    path: str
    decision: str | None
    before: object
    after: object
    change: str
    refs: tuple[str | None, str | None]
    features: tuple[str, ...] = ()


def decision_of(path: str, block_ids=frozenset()) -> str | None:
    """The decision a delta path belongs to: `family`, a block id, `loss`, `optimizer`, …,
    `regularization[<kind>]`; None for a path outside the spec's decisions (data_contract)."""
    m = _FIRST.match(str(path))
    if not m:
        return None
    root, key, value, nxt = m.groups()
    if root == "blocks":
        block = value if key == "id" else nxt
        return None if block is None else (block if block in block_ids else "family")
    if root == "regularization":
        return f"regularization[{value}]" if key == "kind" else "regularization"
    return root if root in _ROOTS else None


def run_changes(run: Mapping, run_ref: str, block_ids=frozenset(), inputs=()) -> list[Change]:
    """Every leaf change of one run's delta, in the delta's order, each with the `inputs` it
    introduces."""
    out: list[Change] = []
    for k, entry in enumerate(run.get("delta") or []):
        if not isinstance(entry, Mapping) or "path" not in entry:
            continue
        base, leaves = f"{run_ref}.delta.{k}", []
        _walk(entry.get("from"), entry.get("to"), str(entry["path"]),
              (f"{base}.from", f"{base}.to"), leaves)
        was = _Before(entry.get("from"))
        out += [replace(c, features=was.introduced(c, inputs)) for c in leaves]
    refamily = any(c.path.split("/")[0] == "family" for c in out)
    return [replace(c, decision="family" if refamily and c.path.startswith("blocks") else
                    decision_of(c.path, block_ids)) for c in out]


def features_named(run: Mapping, inputs) -> list[str]:
    """The inputs a run's delta introduces, over all its changes, in `inputs` order."""
    return _in_order(run_changes(run, "", inputs=inputs), inputs)


def input_names(store: RecordStore) -> list[str]:
    """The network's inputs, in `netspec.data_contract.features` order."""
    features = peek(store, "netspec.json#data_contract.features", []) or []
    return [f["name"] for f in features
            if isinstance(f, Mapping) and isinstance(f.get("name"), str)]


def _in_order(changes, inputs) -> list[str]:
    named = {n for c in changes for n in c.features}
    return [n for n in inputs if n in named]


class CampaignFacts:
    """The campaign's runs, the selection and the searches."""

    def __init__(self, records: RecordStore, catalog: Catalogs) -> None:
        self._store, self._catalog = records, catalog

    def collect(self, sink: FactSink) -> None:
        runs = [r for r in peek(self._store, RUNS, []) or []
                if isinstance(r, Mapping) and spellable(r.get("id") or "", selector=True)]
        selection = peek(self._store, f"{TRAINING}#experiments.selection", {}) or {}
        blocks = frozenset(b.get("id") for b in peek(self._store, "netspec.json#blocks", []) or []
                           if isinstance(b, Mapping))
        spec = (blocks, input_names(self._store))
        sink.take(self._store, "runs_n", RUNS, "count_of")
        sink.group("runs", [self._run(sink, run, selection, spec) for run in runs])
        sink.name(str(r["id"]) for r in runs)
        self._selection(sink, selection)
        self._hpo(sink)

    def _run(self, sink: FactSink, run: Mapping, selection: Mapping, spec: tuple) -> dict:
        """One entry of the `runs` group; `spec` is (the spec's block ids, its input names)."""
        rid, s, (blocks, inputs) = str(run["id"]), self._store, spec
        ref = f"{RUNS}[id={rid}]"
        direction = self._catalog.direction(run.get("primary_name") or selection.get("primary"))
        changes = run_changes(run, ref, blocks, inputs)
        delta = [self._change(sink, rid, k, c) for k, c in enumerate(changes)]
        features = _in_order(changes, inputs)
        sink.name(_item_ids(changes))
        return {"id": rid, "kind": run.get("kind"), "family": run.get("family"),
                "primary": sink.take(s, ("run", rid, "primary"), f"{ref}.primary", "metric",
                                     direction),
                "epochs": sink.take(s, ("run", rid, "epochs"), f"{ref}.epochs", "epoch"),
                "best_epoch": sink.take(s, ("run", rid, "best_epoch"), f"{ref}.best_epoch",
                                        "epoch"),
                "delta": delta, "features": features,
                "label": run_label(changes, [(d["from"], d["to"]) for d in delta],
                                   self._catalog.show["delta_label"], features),
                "selected": rid == selection.get("run"),
                "reference": run.get("kind") == "baseline",
                "eligible": run.get("eligible"), "status": run.get("status")}

    def _change(self, sink: FactSink, rid: str, k: int, change: Change) -> dict:
        out = {"path": change.path}
        for side, value, ref in (("from", change.before, change.refs[0]),
                                 ("to", change.after, change.refs[1])):
            out[side] = None
            if _scalar(value) and ref:
                kind = "param" if not isinstance(value, str) else "text"
                out[side] = sink.take(self._store, ("run", rid, "delta", str(k), side), ref,
                                      kind)
            if side == "to" and change.path.split("/")[0] == "family" and isinstance(value, str):
                sink.name([value])
        out.update(decision=change.decision, change=change.change)
        if change.features:
            out["features"] = list(change.features)
        return out

    def _selection(self, sink: FactSink, selection: Mapping) -> None:
        s, primary = self._store, selection.get("primary")
        direction = self._catalog.direction(primary)
        base = f"{TRAINING}#experiments.selection"
        value = sink.take(s, "selected_primary", f"{base}.value", "metric", direction)
        sink.take(s, "selection_margin", f"{base}.margin_over_baseline", "diff", direction)
        rule = f"{base}.rule"
        sink.group("selection", {"run": selection.get("run"),
                                 "rule_ref": rule if peek(s, rule) is not None else None,
                                 "primary_name": primary, "direction": direction, "value": value})

    def _hpo(self, sink: FactSink) -> None:
        studies = [h for h in peek(self._store, f"{TRAINING}#experiments.hpo", []) or []
                   if isinstance(h, Mapping) and spellable(h.get("study") or "", selector=True)]
        out = []
        for study in studies:
            sid, s = str(study["study"]), self._store
            ref = f"{TRAINING}#experiments.hpo[study={sid}]"
            primary = peek(s, f"{TRAINING}#experiments.selection.primary")
            direction = self._catalog.direction(primary)
            out.append({"study": sid, "best": study.get("best"),
                        **{f: sink.take(s, ("hpo", sid, f), f"{ref}.{f}", "count")
                           for f in ("trials", "completed", "pruned")},
                        "best_value": sink.take(s, ("hpo", sid, "best_value"),
                                                f"{ref}.best_value", "metric", direction)})
        if len(studies) == 1:
            sink.take(self._store, "runs_pruned_n",
                      f"{TRAINING}#experiments.hpo[study={studies[0]['study']}].pruned", "count")
        if out:
            sink.group("hpo", out)


# --------------------------------------------------------------------------- the delta walk

def _walk(before, after, path: str, refs, out: list) -> None:
    if before == after:
        return
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in [*before, *(k for k in after if k not in before)]:
            if key not in ANNOTATIONS:
                _walk(before.get(key), after.get(key), f"{path}/{key}",
                      tuple(_seg(r, key) for r in refs), out)
        return
    key = _list_key(before, after)
    if key is None:
        out.append(Change(path, None, before, after, "changed", refs))
        return
    _walk_list(before, after, (path, key), refs, out)


def _walk_list(before: list, after: list, where: tuple, refs, out: list) -> None:
    path, key = where
    old = {item[key]: item for item in before}
    new = {item[key]: item for item in after}
    for name in [*old, *(n for n in new if n not in old)]:
        sub = f"{path}[{key}={name}]"
        spelled = tuple(f"{r}[{key}={name}]" if r and spellable(name, selector=True) else None
                        for r in refs)
        if name not in new:
            out.append(Change(sub, None, old[name], None, "removed", (spelled[0], None)))
        elif name not in old:
            out.append(Change(sub, None, None, new[name], "added", (None, spelled[1])))
        else:
            _walk(old[name], new[name], sub, spelled, out)


def _list_key(before, after) -> str | None:
    """`id` or `kind` when both sides are lists of objects that every one carries, uniquely."""
    if not (isinstance(before, list) and isinstance(after, list)):
        return None
    items = [*before, *after]
    for key in _LIST_KEYS:
        if items and all(isinstance(i, Mapping) and isinstance(i.get(key), str) for i in items) \
                and all(len({i[key] for i in side}) == len(side) for side in (before, after)):
            return key
    return None


def _seg(ref: str | None, key) -> str | None:
    return f"{ref}.{key}" if ref and spellable(key) else None


def _scalar(value) -> bool:
    return isinstance(value, (str, int, float)) and value is not None


class _Before:
    """What one delta entry's `from` names: its string values (annotations skipped) and all of
    its text (annotations included)."""

    def __init__(self, before) -> None:
        self._equal = frozenset(_strings(before, ANNOTATIONS))
        self._text = "\n".join(_strings(before))

    def introduced(self, change: Change, inputs) -> tuple[str, ...]:
        """The inputs `change` introduces: a string value of its `to` equal to the name that no
        value of `from` equals; or, in an item it adds, the name as a whole word that `from`
        never names. A value that names the choice itself (a `name`, `type`, `kind`, `scheme` or
        `id`: `linear`, `custom`, `B0`) is the catalog's word or an id, never an input."""
        leaf = re.sub(r"\[[^\]]*\]$", "", change.path.rsplit("/", 1)[-1])
        values = () if leaf in _CHOICES else frozenset(_strings(change.after, _NOT_INPUTS))
        added = "\n".join(_strings(change.after, _CHOICES)) if change.change == "added" else ""
        return tuple(n for n in inputs
                     if (n in values and n not in self._equal)
                     or (added and _names(n, added) and not _names(n, self._text)))


def _strings(value, skip=frozenset()):
    """The string values inside `value`, those under a key of `skip` left out."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, v in value.items():
            if key not in skip:
                yield from _strings(v, skip)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v, skip)


def _names(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text) is not None


def _item_ids(changes) -> list[str]:
    """The ids of the items (blocks) the changes add or remove: names for the digit rule."""
    return [str(side["id"]) for c in changes if c.change != "changed"
            for side in (c.before, c.after) if isinstance(side, Mapping) and side.get("id")]


# --------------------------------------------------------------------------- the run label

_NONE, _ELIDED = "none", "…"
_OWN_VALUE = frozenset({"name", "type", "kind", "scheme", "value"})
_HEAD = re.compile(r"^([A-Za-z_]+)(?:\[\w+=([^\]]+)\])?$")


def run_label(changes, sides, templates: Mapping, features=None) -> str | None:
    """The run's mechanical label: one part per change, from the `delta_label` template of the
    change path's first segment (`family`, `blocks`, `loss`, …; a segment without a template
    reads like `loss`'s: `<segment> {param} {from} → {to}`), its values the `{key}` placeholders
    of `sides` (the (from, to) fact keys of each change), else `none` for an absent value and
    `…` for a list, an object without a type or a value no fact holds (a number is never
    typed). The changes that introduce inputs are one part, `without {features}`; a run that
    changes the family says so once, not once per block the new family brings; block changes
    of one parameter with equal values are one part (`B1, B4 units …`). Parts are joined by
    `; `, in the delta's order; None for a run that changed nothing."""
    parts: list[dict] = []
    refamily = any(c.path.split("/")[0] == "family" for c in changes)
    featured = list(features or dict.fromkeys(n for c in changes for n in c.features))
    for change, keys in zip(changes, sides):
        if change.features:
            if not any(p["root"] == "features" for p in parts):
                parts.append({"root": "features", "features": featured})
            continue
        part = _part(change, keys)
        if part is None or (refamily and part["root"] == "blocks"):
            continue
        same = next((p for p in parts if p.get("sig") == part["sig"] and part["sig"]), None)
        if same is not None:
            same["block"].append(part["block"][0])
        else:
            parts.append(part)
    text = [_render(p, templates) for p in parts]
    return "; ".join(dict.fromkeys(text)) or None


def _part(change: Change, keys) -> dict | None:
    """The words of one change's label part: root, block ids, parameter, kind, from, to."""
    segs = change.path.split("/")
    head = _HEAD.match(segs[0])
    if head is None:
        return None
    root, selected = head.groups()
    rest = segs[1:]
    block = selected
    if root == "blocks" and selected is None and rest:
        block, rest = rest[0], rest[1:]
    names = [re.sub(r"\[.*\]$", "", s) for s in rest if not s.isdigit()]
    param = names[-1] if names else ("type" if root == "blocks" else "")
    words = [f"{{{key}}}" if key else _word(value, whole=not rest)
             for key, value in zip(keys, (change.before, change.after))]
    sig = (param, repr(change.before), repr(change.after)) if root == "blocks" else None
    return {"root": root, "block": [block or ""], "param": param, "kind": selected or root,
            "from": words[0], "to": words[1], "sig": sig}


def _word(value, whole: bool) -> str:
    """How a value no fact holds is written: `none`, an item's type (a whole block added or
    removed), else `…`."""
    if value is None:
        return _NONE
    if whole and isinstance(value, Mapping):
        word = value.get("type") or value.get("name")
        return word if isinstance(word, str) and word.strip() else _ELIDED
    return _ELIDED


def _render(part: dict, templates: Mapping) -> str:
    if part["root"] == "features":
        return templates["features"].format(features=", ".join(part["features"]))
    template = templates.get(part["root"]) or f"{part['root']} {{param}} {{from}} → {{to}}"
    if "{param}" not in template and part["param"] not in _OWN_VALUE and part["param"]:
        template = template.replace("{from}", "{param} {from}")
    return template.format(**{"block": ", ".join(part["block"]), "param": part["param"],
                              "kind": part["kind"], "from": part["from"], "to": part["to"]})
