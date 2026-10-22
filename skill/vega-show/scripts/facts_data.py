"""
facts_data — `DataFacts`: what the data was and what the shaping did to it (contract §2.3, the
groups `sizes`, `recipe`, `split`, `sample`, `inputs` and `class_balance`).

Every number is read where a record wrote it, with the first candidate ref that resolves: the
source's size from the geometry record (`source.rows`), else from the shaper's input record;
the parts' sizes from the shaper's output files, else from the split step's rows. The `sizes`
group is D5 §6.2's `{source_rows, source_columns, shaped_inputs, parts: {train, valid, test}}`
(each a fact key, null when no record holds it), with `classes_n` on a task with classes. The
recipe is the shaper's manifest, step by step: a step that ran before the split counts `all`
rows (`step_s4_rows_after`), one after it counts each part (`step_s9_rows_after_train`).

The `inputs` group is structure, not numbers: the network's inputs in
`netspec.data_contract.features` order (the order the input groups are validated against), then
an autoregressive window's target channel (origin `autoregressive`: the network reads the
target's past after the features, and the network view lists it last), the
step each came out of, the raw typing, the inputs the recipe created (absent from the raw
columns), the encoded ones, and the analysis's pairwise relations among the inputs (`{a, b, r}`,
`r` as the analysis measured it: Spearman first). The names it touches — raw and shaped
columns, class names, the tokens of the dataset's file stem — go to the digit rule's whitelist.
"""
from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Mapping

from digit_rule import TOKEN_RE
from fact_sink import Catalogs, FactSink, Refusal, key_of, peek, spellable
from show_ports import RecordStore

__all__ = ["DataFacts"]

NETSPEC, GEOMETRY, SHAPE, ANALYSIS = ("netspec.json", "records/geometry.json",
                                      "records/shape.json", "records/analysis.json")
PARTS = ("train", "valid", "test")
_ENCODED_TYPES = ("nominal", "ordinal", "boolean")
_CORRELATIONS = ("spearman", "r", "pearson", "kendall")
AUTOREGRESSIVE = "autoregressive"           # inputs.origin of a window's autoregressive target channel


def typing_of(store: RecordStore) -> tuple[dict, list]:
    """{raw column: final type} from the geometry record (a list of rows or a mapping), and the
    raw column names (the typing's, then the input schema's)."""
    typing = peek(store, f"{GEOMETRY}#typing", None)
    rows = typing if isinstance(typing, list) else \
        [{"column": k, **v} for k, v in typing.items() if isinstance(v, Mapping)] \
        if isinstance(typing, Mapping) else []
    out = {str(r["column"]): r.get("final_type") or r.get("role")
           for r in rows if isinstance(r, Mapping) and r.get("column") is not None}
    schema = peek(store, "schema.json#input", []) or []
    raw = list(out) + [str(c["name"]) for c in schema if isinstance(c, Mapping)
                       and c.get("name") and str(c["name"]) not in out]
    return out, raw


def steps_with_op(store: RecordStore, op: str) -> set[str]:
    """The ids of the recipe steps of one op."""
    steps = peek(store, f"{SHAPE}#manifest.steps", []) or []
    return {str(s.get("id")) for s in steps if isinstance(s, Mapping) and s.get("op") == op}


class DataFacts:
    """The data and shaping facts of one bundle."""

    def __init__(self, records: RecordStore, catalog: Catalogs) -> None:
        self._store, self._catalog = records, catalog

    def collect(self, sink: FactSink) -> None:
        split = self._recipe(sink)
        self._sizes(sink, split)
        self._split(sink, split)
        self._sample(sink)
        self._inputs(sink)
        self._class_balance(sink)

    # ------------------------------------------------------------------ sizes and recipe
    def _sizes(self, sink: FactSink, split: str | None) -> None:
        s = self._store
        sizes = {"source_rows": sink.take(s, "source_rows", (
                     f"{GEOMETRY}#source.rows", f"{SHAPE}#manifest.input.rows",
                     f"{SHAPE}#recipe.input.rows"), "count"),
                 "source_columns": sink.take(s, "source_columns", (
                     f"{GEOMETRY}#source.columns", f"{SHAPE}#manifest.input.columns"), "count"),
                 "shaped_inputs": sink.take(s, "shaped_inputs", (
                     f"{NETSPEC}#data_contract.input_dim",
                     "records/architecture.json#netspec.data_contract.input_dim"), "count")}
        parts = {}
        for part in PARTS:
            after = f"{SHAPE}#manifest.steps[id={split}].rows_after.{part}" if split else None
            parts[part] = sink.take(s, f"{part}_rows", (
                f"{SHAPE}#manifest.outputs.files.{part}.rows", after), "count")
        sizes["parts"] = parts
        classes_n = sink.take(s, "classes_n", f"{NETSPEC}#task.classes", "count_of")
        sink.group("sizes", {**sizes, "classes_n": classes_n} if classes_n else sizes)

    def _recipe(self, sink: FactSink) -> str | None:
        """The recipe group, one entry per manifest step; returns the split step's id."""
        steps, entries, split = peek(self._store, f"{SHAPE}#manifest.steps", []), [], None
        for step in steps if isinstance(steps, list) else []:
            sid = step.get("id") if isinstance(step, Mapping) else None
            if not spellable(sid or "", selector=True):
                sink.warn(f"{SHAPE}: a recipe step without a usable id ({sid!r}) has no facts")
                continue
            entries.append(self._step(sink, step))
            split = sid if step.get("op") == "split" and split is None else split
        sink.group("recipe", entries)
        return split

    def _step(self, sink: FactSink, step: Mapping) -> dict:
        sid, s = str(step["id"]), self._store
        base = f"{SHAPE}#manifest.steps[id={sid}]"
        rows = {which: self._rows(sink, step, which) for which in ("before", "after")}
        source = step.get("source")
        finding = re.match(r"analysis:(F\w+)", source) if isinstance(source, str) else None
        rationale = f"{SHAPE}#recipe.steps[id={sid}].rationale"
        return {"step": sid, "op": step.get("op"), "source": source,
                "rows_before": rows["before"], "rows_after": rows["after"],
                "columns_added": [str(c) for c in step.get("columns_added") or []],
                "columns_removed_n": sink.take(s, ("step", sid, "columns_removed_n"),
                                               f"{base}.columns_removed", "count_of"),
                "fit_on": step.get("fit_on"),
                "rationale_ref": rationale if peek(s, rationale) is not None else None,
                "finding": finding.group(1) if finding else None}

    def _rows(self, sink: FactSink, step: Mapping, which: str) -> dict:
        rows, sid = step.get(f"rows_{which}"), str(step["id"])
        if not isinstance(rows, Mapping):
            return {}
        out = {}
        for part in rows:
            if not spellable(part):
                continue
            parts = ("step", sid, "rows", which) if list(rows) == ["all"] else \
                ("step", sid, "rows", which, part)
            ref = f"{SHAPE}#manifest.steps[id={sid}].rows_{which}.{part}"
            key = sink.take(self._store, parts, ref, "count")
            if key:
                out[str(part)] = key
        return out

    def _split(self, sink: FactSink, split: str | None) -> None:
        if split is None:
            sink.warn(f"{SHAPE}: the recipe records no split step; the split group is empty")
            return
        kind = (peek(self._store, f"{SHAPE}#recipe.steps[id={split}].params.kind")
                or peek(self._store, f"{SHAPE}#manifest.steps[id={split}].parameters_fitted.kind"))
        parts = {p: f"{p}_rows" for p in PARTS if f"{p}_rows" in sink}
        sink.group("split", {"step": split, "kind": kind, "parts": parts})

    # ------------------------------------------------------------------ sample
    def _sample(self, sink: FactSink) -> None:
        sample = peek(self._store, f"{NETSPEC}#data_contract.sample", {}) or {}
        shape = sample.get("shape", "row")
        out = {"shape": shape}
        if shape == "window":
            out.update(self._window(sink, sample))
        elif shape == "graph":
            out["graph"] = self._graph(sink, sample)
        sink.group("sample", out)

    def _window(self, sink: FactSink, sample: Mapping) -> dict:
        s, ref = self._store, f"{NETSPEC}#data_contract.sample"
        window = {"length": sink.take(s, "window_length", f"{ref}.length", "param"),
                  "stride": sink.take(s, "window_stride", f"{ref}.stride", "param")}
        extra = {k: sample.get(k) for k in ("series_by", "order_by", "context", "autoregressive")}
        return {"window": window, "horizon": sink.take(s, "horizon", f"{ref}.horizon", "param"),
                **extra}

    def _graph(self, sink: FactSink, sample: Mapping) -> dict:
        s, graph = self._store, peek(self._store, f"{SHAPE}#manifest.graph", {}) or {}
        return {"level": sample.get("level", "node"), "protocol": sample.get("protocol"),
                "batching": sample.get("batching"), "edges_method": graph.get("method"),
                "edges_n": sink.take(s, "edges_n", f"{SHAPE}#manifest.graph.n_edges", "count"),
                "k": sink.take(s, "graph_edges_k", f"{SHAPE}#manifest.graph.params.k", "param")}

    # ------------------------------------------------------------------ inputs
    def _inputs(self, sink: FactSink) -> None:
        features = peek(self._store, f"{NETSPEC}#data_contract.features", []) or []
        shaped = [str(f["name"]) for f in features if isinstance(f, Mapping) and f.get("name")]
        if not shaped:
            raise Refusal("PF7", f"{NETSPEC}#data_contract.features",
                          "the shipped spec names no network input")
        typing, raw = typing_of(self._store)
        origin = {str(f["name"]): f.get("origin") for f in features
                  if isinstance(f, Mapping) and f.get("name")}
        channel = self._autoregressive_channel()
        if channel and channel not in shaped:      # the network reads it after the features
            shaped.append(channel)
            origin[channel] = AUTOREGRESSIVE
        encoded_steps = steps_with_op(self._store, "encode")
        encoded = [str(f["name"]) for f in features if isinstance(f, Mapping) and f.get("name")
                   and (f.get("role") == "nominal" or f.get("origin") in encoded_steps
                        or typing.get(str(f["name"])) in _ENCODED_TYPES)]
        sink.name([*shaped, *raw, *self._stem_tokens()])
        sink.group("inputs", {"shaped": shaped, "origin": origin, "typing": typing,
                              "created_by_recipe": [n for n in shaped if raw and n not in raw
                                                    and origin.get(n) != AUTOREGRESSIVE],
                              "encoded": encoded, "relations": self._relations(sink, set(shaped))})

    def _autoregressive_channel(self) -> str | None:
        """An autoregressive window's target column: an input of the network beyond the spec's
        features (the network view appends it to the numeric channels), so an input the story's
        groups must place too."""
        sample = peek(self._store, f"{NETSPEC}#data_contract.sample", {}) or {}
        if sample.get("shape") != "window" or not sample.get("autoregressive"):
            return None
        label = peek(self._store, f"{NETSPEC}#task.label", None)
        return str(label) if label else None

    def _relations(self, sink: FactSink, shaped: set[str]) -> list[dict]:
        """`{a, b, r}` per analysis pair of two inputs, `r` the first measure it records; a pair
        whose measure is not a finite number is dropped with a `WARN:` naming its ref."""
        ref = f"{ANALYSIS}#modules.relations.evidence.pairs"
        pairs, out = peek(self._store, ref, []) or [], []
        for i, pair in enumerate(pairs if isinstance(pairs, list) else []):
            if not isinstance(pair, Mapping) or not {pair.get("a"), pair.get("b")} <= shaped:
                continue
            k = next((k for k in _CORRELATIONS if isinstance(pair.get(k), (int, float))
                      and not isinstance(pair.get(k), bool)), None)
            if k is not None and not math.isfinite(pair[k]):
                sink.warn(f"{ref}.{i}.{k} is not a finite number; the relation of {pair['a']} "
                          f"and {pair['b']} is dropped")
            elif k is not None:
                out.append({"a": pair["a"], "b": pair["b"], "r": pair[k]})
        return out

    def _stem_tokens(self) -> list[str]:
        path = (peek(self._store, f"{GEOMETRY}#source.path")
                or peek(self._store, f"{SHAPE}#manifest.input.path") or "")
        stem = PurePosixPath(str(path)).name.split(".")[0]
        return [m.group(0) for m in TOKEN_RE.finditer(stem)]

    # ------------------------------------------------------------------ class balance
    def _class_balance(self, sink: FactSink) -> None:
        shares = peek(self._store, f"{NETSPEC}#data_contract.class_balance.train", None)
        classes = peek(self._store, f"{NETSPEC}#task.classes", None)
        if not isinstance(shares, Mapping) or not isinstance(classes, list):
            return
        sink.name(classes)
        out, unspellable = {}, []
        for name in [str(c) for c in classes if str(c) in shares]:
            key = key_of(("train_share", name))
            if not spellable(name) or key is None or key in sink:
                unspellable.append(name)
                continue
            if sink.take(self._store, key, f"{NETSPEC}#data_contract.class_balance.train.{name}",
                         "param"):
                out[name] = key
        if unspellable:
            sink.warn(f"no train-share fact for {unspellable}: the ref grammar cannot spell a "
                      "class_balance key holding a space or a dot, two names share a key, or a "
                      "name is too long for one")
        sink.group("class_balance", out)
