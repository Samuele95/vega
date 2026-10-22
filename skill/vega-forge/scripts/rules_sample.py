"""
rules_sample — the compiler's rules on the sample shape (netspec-contract §1 and §3, D1 §14.1).

Each is a pure `Context -> markers` function that `compile_spec.py` registers in `RULES`:

  W1  windows: the shape, length ≥ 2, stride ≥ 1, horizon ≥ 1 iff forecast, the context with a
      source, series_by / order_by as key columns, the series fork, autoregressive only on
      forecast, season ≥ 2, the time split
  W3  the future: no bidirectional recurrent block on a forecast or under a stream serving mode
  R1  the runtime: batch_size "full" iff batching full, an integer batch at most the training
      targets (rows, windows, seed nodes or graphs), the budget, batchnorm on small batches

W2, the graph sample, is `rules_graph.py`'s; it shares the helpers here — a rule's marker
makers, a sample field's own source, the key columns.

**The series fork** (family-catalog §5, D1 §14.4). `series_by` defaults to the `lag` step's
`group_by`, else the time split's `by` (`defaults/window_series`). A null `series_by` when the
shaping record names such a key is refused unless a `user:` source says so (the windows would
cross the series the shaper grouped its lags by); a null `series_by` when the record names none
is the unattended answer, and the compiler repeats its warning: *windows may cross series — no
series key in the shaping record*.

**Sub-sources.** A sample field that answers a fork carries its own source beside it —
`context_source`, `autoregressive_source`, `series_source`, `protocol_source`,
`batching_source` — and a `vega:default` one names its catalog row in `<field>_catalog`. The
context must carry one (W1 says "with a source"); the others are checked whenever present, and
`protocol: transductive` requires a `user:` one (W2).
"""
from __future__ import annotations

import math
from typing import Mapping

from vega_core import Level, error, ok, warn

__all__ = ["windows", "future", "batch_and_budget", "rule_markers", "sub_source",
           "sample_source", "key_columns", "int_at_least"]

_CONTEXTS = ("previous-part", "none")
_RECURRENT = ("rnn", "gru", "lstm")
_SERIES_WARN = "windows may cross series — no series key in the shaping record"


def rule_markers(rule, clause="netspec-contract §3"):
    """The marker makers of one rule, so each call names its rule once."""
    def err(message, where="data_contract.sample"):
        return error(message, rule=rule, clause=clause, where=where)

    def wrn(message, where="data_contract.sample"):
        return warn(message, rule=rule, clause=clause, where=where)
    return err, wrn


def sub_source(ctx, field, required=False):
    """The source a sample field carries beside it (`<field>_source`, `<field>_catalog`)."""
    src = ctx.sample.get(f"{field}_source")
    where = f"data_contract.sample.{field}_source"
    if not src:
        if not required:
            return []
        return [error(f"data_contract.sample.{field} carries no source: a fork's answer is a "
                      f"choice like any other — {field}_source (vega:default with "
                      f"{field}_catalog, or the user turn that settled it)", rule="P1",
                      clause="netspec-contract §1", where=where)]
    if not ctx.resolver.well_formed(src):
        return [error(f"source {src!r} is not one of blueprint: / shape: / geometry: / "
                      "analysis: / user: / vega:default", rule="P1",
                      clause="netspec-contract §1 (the closed set)", where=where)]
    resolved, detail = ctx.resolver.resolve(src)
    if not resolved:
        return [error(f"source {src!r} does not resolve ({detail})", rule="P1",
                      clause="netspec-contract §1", where=where)]
    row = ctx.sample.get(f"{field}_catalog")
    if src == "vega:default" and (not row or ctx.catalog.row(row) is None):
        return [error(f"a vega:default must name the catalog row it cites ({field}_catalog: "
                      f"\"defaults/…\"); got {row!r}", rule="P1",
                      clause="family-catalog.md §intro", where=where)]
    return []


def sample_source(ctx, what):
    src = ctx.sample.get("source")
    if not src:
        return [error(f"data_contract.sample carries no source (the {what} is a choice like "
                      "any other — analysis:context.time, shape:manifest/graph or a user turn)",
                      rule="P1", clause="netspec-contract §1")]
    resolved, detail = ctx.resolver.resolve(src)
    if not resolved:
        return [error(f"data_contract.sample.source {src!r} does not resolve ({detail})",
                      rule="P1", clause="netspec-contract §1")]
    return []


def key_columns(ctx, fields, rule):
    """Key columns are in the shaped output schema and are never features."""
    err, _ = rule_markers(rule, "netspec-contract §1 (keys)")
    columns = ctx.inputs.output_columns()
    names = {f.get("name") for f in ctx.features if isinstance(f, Mapping)}
    out = []
    for field in fields:
        column = ctx.sample.get(field)
        if column is None:
            continue
        if columns is not None and column not in columns:
            out.append(err(f"sample.{field} {column!r} is not a column of the shaped output "
                           "schema (shape:manifest/output_schema)"))
        if column in names:
            out.append(err(f"sample.{field} {column!r} is a key column and is listed among "
                           "data_contract.features: a key orders or groups the rows and is "
                           "never an input"))
    return out


# -- W1 windows --------------------------------------------------------------

def int_at_least(value, least) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


def _window_numbers(ctx):
    err, _ = rule_markers("W1")
    s, out = ctx.sample, []
    for key, least in (("length", 2), ("stride", 1)):
        if s.get(key) is None:
            out.append(err(f"data_contract.sample.{key} is required on a windowed sample"))
        elif not int_at_least(s.get(key), least):
            out.append(err(f"data_contract.sample.{key} is {s.get(key)!r}; it must be an "
                           f"integer >= {least}"))
    if ctx.kind == "forecast":
        if s.get("horizon") is None:
            out.append(err("data_contract.sample.horizon is required for a forecast task"))
        elif not int_at_least(s.get("horizon"), 1):
            out.append(err(f"data_contract.sample.horizon is {s.get('horizon')!r}; it must be "
                           "an integer >= 1"))
    elif s.get("horizon") is not None:
        out.append(err(f"data_contract.sample.horizon is set on a {ctx.kind} task: a horizon "
                       "is forecast-only (a sequence task predicts the anchor's own label)"))
    if s.get("season") is not None and not int_at_least(s.get("season"), 2):
        out.append(err(f"data_contract.sample.season is {s.get('season')!r}; when set it is an "
                       "integer >= 2 (mase's seasonal scale)"))
    return out


def _context(ctx):
    err, _ = rule_markers("W1")
    value = ctx.sample.get("context")
    if value not in _CONTEXTS:
        return [err(f"data_contract.sample.context is {value!r}; declare previous-part (the "
                    "default, defaults/window_context: every valid and test row is scored) or "
                    "none (the first length-1 rows of each series are unscored)")]
    return sub_source(ctx, "context", required=True)


def _series(ctx):
    """The series fork: its answer's own source first (P1 — well formed, resolving, a
    `vega:default` naming its row), then what that answer may be given the shaping record."""
    err, wrn = rule_markers("W1", "family-catalog.md §5 (series_by)")
    out = sub_source(ctx, "series")
    series = ctx.sample.get("series_by")
    key, where = ctx.inputs.series_key()
    source = str(ctx.sample.get("series_source") or "")
    if source.startswith("user:"):
        return out
    if series is None and key:
        return out + [err(f"series_by is null, but the shaping record names the series key "
                          f"{key!r} ({where}; defaults/window_series): windows would cross the "
                          f"series the shaper grouped by — set series_by: {key!r}, or null "
                          "with a user: source")]
    if series is None:
        text = (ctx.catalog.default("window_series") or {}).get("unattended_warn") or _SERIES_WARN
        return out + [wrn(text)]
    if source == "vega:default" and key and series != key:
        return out + [err(f"series_by {series!r} cites vega:default, but "
                          f"defaults/window_series resolves to {key!r} ({where})")]
    return out


def _autoregressive(ctx):
    err, _ = rule_markers("W1")
    value = ctx.sample.get("autoregressive")
    if value is True and ctx.kind != "forecast":
        return [err(f"data_contract.sample.autoregressive is true on a {ctx.kind} task: the "
                    "channel is the target's own past, forecast only — on any other task it "
                    "would carry the anchor's own label, the target")]
    if ctx.kind == "forecast" and value is not None:
        if not isinstance(value, bool):
            return [err(f"data_contract.sample.autoregressive is {value!r}; it is true or false")]
        return sub_source(ctx, "autoregressive")
    return []


def _time_split(ctx):
    err, wrn = rule_markers("W1", "family-catalog.md §1")
    split = ctx.inputs.split_kind()
    if split and split != "time":
        return [err(f"a sequence family needs a time split; the shaper's split is {split!r}",
                    where=None)]
    if not split:
        return [wrn("the manifest records no split kind; the time split could not be checked",
                    where=None)]
    return []


def windows(ctx):
    if not ctx.needs_window():
        out = []
        if ctx.sample_shape == "window" and not ctx.catalog.is_floor_on(ctx.family_name,
                                                                          "window"):
            out.append(warn(f"data_contract.sample.shape is 'window' on a non-sequence "
                            f"family {ctx.family_name!r}", rule="W1",
                            clause="family-catalog.md §2"))
        return out + _autoregressive(ctx)
    if ctx.sample_shape != "window":
        return [error(f"family {ctx.family_name!r} / task {ctx.kind!r} needs windows but "
                      f"data_contract.sample.shape is {ctx.sample.get('shape')!r}",
                      rule="W1", clause="netspec-contract §3")] + _time_split(ctx)
    out = (_window_numbers(ctx) + sample_source(ctx, "window") + _context(ctx)
           + key_columns(ctx, ("series_by", "order_by"), "W1") + _series(ctx)
           + _autoregressive(ctx) + _time_split(ctx))
    if not any(m.level is Level.ERROR for m in out):
        out.append(ok(f"the window sample (length {ctx.sample.get('length')}, stride "
                      f"{ctx.sample.get('stride')}, context {ctx.sample.get('context')}) is "
                      "declared and sourced", rule="W1", clause="netspec-contract §3"))
    return out


# -- W3 the future -----------------------------------------------------------

def future(ctx):
    mode = ((ctx.inputs.pipeline or {}).get("serving") or {}).get("mode")
    settings = [s for s, on in (("task.kind forecast", ctx.kind == "forecast"),
                                ("the blueprint's serving.mode stream", mode == "stream")) if on]
    if not settings:
        return []
    why = " and ".join(settings)
    out = [error(f"block {b.get('id', path)} ({b.get('type')}) is bidirectional: it reads the "
                 f"window's future, which {why} does not have at inference — set "
                 "bidirectional: false (the bi-* families classify complete windows)",
                 rule="W3", clause="netspec-contract §3", where=path)
           for path, b, _ in ctx.blocks if isinstance(b, Mapping)
           and b.get("type") in _RECURRENT and (b.get("params") or {}).get("bidirectional")]
    forbidden = (ctx.family_row or {}).get("forbidden_on") or {}
    if not out and (ctx.kind in (forbidden.get("tasks") or ())
                    or mode in (forbidden.get("serving_modes") or ())):
        out.append(error(f"family {ctx.family_name!r} reads the window's future and is "
                         f"forbidden under {why} (catalog families/{ctx.family_name}."
                         "forbidden_on)", rule="W3", clause="netspec-contract §3"))
    return out


# -- R1 the runtime ----------------------------------------------------------

def _training_targets(ctx) -> tuple:
    """(count, unit): what one epoch iterates over, from the manifest's counts."""
    rows = ctx.inputs.rows_of("train")
    shape, s = ctx.sample_shape, ctx.sample
    if shape == "window" and rows and int_at_least(s.get("stride"), 1):
        return math.ceil(rows / s["stride"]), "training windows at most (train rows / stride)"
    if shape != "graph":
        return rows, "training rows"
    nodes = ((ctx.inputs.graph or {}).get("n_nodes") or {}).get("train") or rows
    if s.get("level") == "graph":
        graphs_ = ctx.inputs.graphs_of("train")
        return ((graphs_, "training graphs") if graphs_ else
                (nodes, "training graphs at most (bounded by the train nodes: the manifest "
                        "records no graph count)"))
    return nodes, "training seed nodes" if s.get("batching") == "sampled" else "training nodes"


def _batch_size(ctx):
    batch = ctx.training.get("batch_size")
    batching = ctx.sample.get("batching") if ctx.sample_shape == "graph" else None
    if batch == "full" or batching == "full":
        if batch != "full" or batching != "full":
            return [error(f"training.batch_size is {batch!r} and data_contract.sample.batching "
                          f"is {batching!r}: batch_size \"full\" (one batch is the whole "
                          "view) goes with batching full, and only with it", rule="R1",
                          clause="netspec-contract §3")]
        return []
    if batch is None:
        return []
    if not int_at_least(batch, 1):
        return [error(f"training.batch_size is {batch!r}; it is a positive integer, or \"full\" "
                      "under batching full", rule="R1", clause="netspec-contract §3")]
    count, unit = _training_targets(ctx)
    if count and batch > count:
        return [error(f"training.batch_size {batch} exceeds the {count} {unit}", rule="R1",
                      clause="netspec-contract §3")]
    return []


def batch_and_budget(ctx):
    out = _batch_size(ctx)
    batch = ctx.training.get("batch_size")
    if not (ctx.training.get("budget") or {}).get("wall_clock_min"):
        out.append(error("training.budget.wall_clock_min is required — a run that never "
                         "converges must still end", rule="R1", clause="netspec-contract §3"))
    if int_at_least(batch, 1) and batch < 32 and any(
            b.get("type") == "batchnorm" for _, b, _ in ctx.blocks if isinstance(b, Mapping)):
        out.append(warn(f"batchnorm with batch_size {batch} < 32: batch statistics are "
                        "unreliable; the catalog's default below 32 is layernorm", rule="R1",
                        clause="family-catalog.md §4 (normalisation)"))
    return out
