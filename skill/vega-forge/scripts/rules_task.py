"""
rules_task — the compiler's rules on the task and on the inputs the network reads.

Each is a pure `Context -> markers` function that `compile_spec.py` registers in `RULES`:

  T1a the task kind is a catalog row                           family-catalog.md §1
  T1b the head matches the task                                family-catalog.md §1
  T1c the loss matches the task (and the base task's row)      family-catalog.md §1
  T1d the monitors name declared valid metrics                 family-catalog.md §1
  T1e the task fields its readers need                         netspec-contract §1
  D2a input_dim = output schema − label − keys                 netspec-contract §3
  D2b every nominal is embedded, one-hot or tokenized          netspec-contract §3

**T1b** (D1 §14.1, v2.3): the head's `out` is `recon_dim` under the reconstruction and vae
losses (unsupervised, and anomaly under those losses) — `n_numeric + Σ cardinality` of the
embedded nominals, since each embedded nominal is reconstructed as logits over its categories
and a one-hot nominal is numeric columns already (D1 §3.4); any positive `z` under `svdd`;
`k − 1` for ordinal; `H` (× `Q` under the quantile loss) for forecast; `Q` for regression under
the quantile loss; the base kind's for `sequence-*`, read from `task.base_kind` and never guessed
from the head.

**T1e** reads `catalog.json` `task_fields` for `base_kind`, `classes` and `label_format`, and
follows the contract's prose for `positive` (netspec-contract §1): required for an `anomaly`
task with a label; on `binary` (and `sequence-classification` over a binary base) it defaults to
the second class, so it is checked but never required there — a member of `classes`, and the
second one (PLAN-AMENDMENTS A16): the logit, the threshold, the serving readers and the page all
read `classes[1]` as the positive class, so a spec that names `classes[0]` is refused with
"list the positive class second" rather than read the other way round.
"""
from __future__ import annotations

from typing import Mapping

from vega_core import error, ok, warn

__all__ = ["task_known", "head_matches_task", "loss_matches_task", "monitors", "task_fields",
           "input_dim", "nominals", "head_out", "recon_dim"]

_SVDD_Z = "z"                       # the head rule under svdd: any positive width
_LABEL_FORMATS = ("index", "names", "delimited")


# -- T1a ---------------------------------------------------------------------

def task_known(ctx):
    if ctx.task_row is None:
        return [error(f"task.kind {ctx.kind!r} is not a catalog task kind "
                      f"({', '.join(ctx.catalog.task_kinds)})", rule="T1a",
                      clause="family-catalog.md §1")]
    if not ctx.task.get("source"):
        return [error("task carries no source (the geometry partition or the user turn that "
                      "set it)", rule="P1", clause="netspec-contract §1", where="task")]
    return []


# -- T1b ---------------------------------------------------------------------

def recon_dim(ctx):
    """n_numeric + Σ cardinality of the embedded nominals (D1 §3.4); None when a cardinality
    is missing. Without a features list the contract's input_dim is all there is."""
    features = [f for f in ctx.features if isinstance(f, Mapping)]
    if not features:
        return ctx.data_contract.get("input_dim")
    embedded = ctx.embedded_columns
    numeric = sum(1 for f in features if f.get("role") != "nominal")
    cards = [f.get("cardinality") for f in features
             if f.get("role") == "nominal" and f.get("name") in embedded]
    if any(not isinstance(c, int) for c in cards):
        return None
    return numeric + sum(cards)


def _quantiles(ctx) -> int:
    qs = ((ctx.spec.get("loss") or {}).get("params") or {}).get("quantiles")
    return len(qs) if isinstance(qs, (list, tuple)) and qs else 1


def head_out(ctx, kind=None):
    """The head width the task row requires: a number, `_SVDD_Z`, or None (not checkable)."""
    kind = kind or ctx.kind
    k, loss = ctx.n_classes, ctx.loss_name
    if str(kind).startswith("sequence-"):
        base = ctx.task.get("base_kind")
        return head_out(ctx, base) if base in ctx.catalog.sequence_map else None
    if kind == "binary":
        return 1
    if kind in ("multiclass", "multilabel"):
        return k
    if kind == "regression":
        return _quantiles(ctx) if loss == "quantile" else 1
    if kind == "ordinal":
        return None if k is None else k - 1
    if kind == "forecast":
        horizon = ctx.sample.get("horizon")
        return horizon * _quantiles(ctx) if isinstance(horizon, int) else None
    if kind in ("unsupervised", "anomaly"):
        return _SVDD_Z if loss == "svdd" else recon_dim(ctx)
    return None


def _head_width_marker(ctx, head, want):
    got = (head.get("params") or {}).get("out")
    where = f"head {head.get('id')}"
    if want == _SVDD_Z:
        if isinstance(got, int) and got > 0:
            return None
        return error(f"out {got!r}: under the svdd loss the head is the embedding z the "
                     "hypersphere lives in, a positive integer", rule="T1b",
                     clause="family-catalog.md §1", where=where)
    if got == want:
        return None
    why = ctx.task_row["head"]["out"]
    if ctx.kind in ("unsupervised", "anomaly"):
        why = ("recon_dim = the numeric columns + the sum of the cardinalities of the embedded "
               "nominals, each reconstructed as logits over its categories")
    elif str(ctx.kind).startswith("sequence-"):
        why = f"the base task {ctx.task.get('base_kind')!r}'s head"
    return error(f"out {got!r} but the {ctx.kind} row requires {want} ({why})", rule="T1b",
                 clause="family-catalog.md §1", where=where)


def head_matches_task(ctx):
    if ctx.task_row is None:
        return []
    out = []
    if len(ctx.heads) != 1 and ctx.effective_kind != "multilabel":
        out.append(error(f"exactly one head block is required; the spec declares "
                         f"{len(ctx.heads)}", rule="T1b", clause="family-catalog.md §3 (head)"))
    want_act = ctx.task_row["head"]["activation"]
    want_out = head_out(ctx)
    for head in ctx.heads:
        params = head.get("params") or {}
        if params.get("activation", "none") != want_act:
            out.append(error(f"activation {params.get('activation')!r} but the {ctx.kind} row "
                             f"requires {want_act!r} — the loss applies the link function, "
                             "never the head (a double sigmoid is the classic silent bug)",
                             rule="T1b", clause="family-catalog.md §1",
                             where=f"head {head.get('id')}"))
        if want_out is not None:
            out.append(_head_width_marker(ctx, head, want_out))
    return [m for m in out if m is not None]


# -- T1c, T1d ----------------------------------------------------------------

def loss_matches_task(ctx):
    if ctx.task_row is None:
        return []
    name = (ctx.spec.get("loss") or {}).get("name")
    rows = [(ctx.kind, ctx.task_row)]
    if ctx.effective_kind != ctx.kind and ctx.catalog.task(ctx.effective_kind):
        rows.append((f"base {ctx.effective_kind}", ctx.catalog.task(ctx.effective_kind)))
    for label, row in rows:
        allowed = row.get("losses_allowed") or []
        if name and allowed and name not in allowed:
            return [error(f"loss {name!r} is not one the {label} row allows "
                          f"({', '.join(allowed)}); its default is "
                          f"{row.get('default_loss')!r}", rule="T1c",
                          clause="family-catalog.md §1")]
    return []


def monitors(ctx):
    metrics = ctx.spec.get("metrics") or {}
    declared = ctx.declared_metrics
    pairs = (("metrics.selection.monitor", (metrics.get("selection") or {}).get("monitor")),
             ("early_stopping.monitor", (ctx.spec.get("early_stopping") or {}).get("monitor")))
    out = []
    for label, monitor in pairs:
        if not monitor:
            out.append(error("is required", rule="T1d", clause="family-catalog.md §1",
                             where=label))
            continue
        part, _, name = str(monitor).partition(".")
        if part != "valid":
            out.append(error(f"is {monitor!r}: a monitor names a metric on `valid` — `train` "
                             "overfits the selection and `test` is sealed", rule="T1d",
                             clause="family-catalog.md §1", where=label))
        elif name not in declared:
            out.append(error(f"names {name!r}, which is not among the declared metrics "
                             f"({', '.join(sorted(declared))})", rule="T1d",
                             clause="family-catalog.md §1", where=label))
    return out


# -- T1e ---------------------------------------------------------------------

def _required_for(ctx, name) -> tuple:
    return tuple((ctx.catalog.task_fields.get(name) or {}).get("required_when") or ())


def _t1e(message, where="task"):
    return error(message, rule="T1e", clause="netspec-contract §1 (task fields)", where=where)


def _base_kind(ctx):
    base, kind, seq = ctx.task.get("base_kind"), ctx.kind, ctx.catalog.sequence_map
    if kind in _required_for(ctx, "base_kind"):
        if base is None:
            return [_t1e(f"a {kind} task needs task.base_kind — the effective kind every "
                         "reader of the base uses (head, loss, metrics, readout); it is never "
                         "guessed from the head")]
        if seq.get(base) != kind:
            bases = sorted(b for b, k in seq.items() if k == kind)
            return [_t1e(f"task.base_kind {base!r} is not a base of {kind} "
                         f"(catalog sequence_map: {', '.join(bases)})")]
        return []
    if base is not None and not (kind == "forecast" and base == "regression"):
        return [_t1e(f"task.base_kind {base!r} is set on a {kind} task; it is required iff "
                     "the kind starts with sequence- (forecast's base is regression)")]
    return []


def _classes(ctx):
    eff, classes = ctx.effective_kind, ctx.task.get("classes")
    if eff not in _required_for(ctx, "classes"):
        return []
    if not isinstance(classes, (list, tuple)) or len(classes) < 2:
        return [_t1e(f"a {eff} task needs task.classes (at least two, in order for ordinal) — "
                     "the head width and every reader's class names come from them")]
    if eff == "binary" and len(classes) != 2:
        return [_t1e(f"a binary task has exactly two classes, not {len(classes)}")]
    return []


def _label_format(ctx):
    eff, fmt = ctx.effective_kind, ctx.task.get("label_format")
    if eff in _required_for(ctx, "label_format"):
        if not isinstance(fmt, Mapping) or fmt.get("kind") != "delimited" or not fmt.get("sep"):
            return [_t1e(f"a {eff} task needs task.label_format {{kind: delimited, sep}}: its "
                         "label is one shaped column holding a delimited set of class names "
                         f"(R-A8), got {fmt!r}")]
        return []
    if fmt is None:
        return []
    if not isinstance(fmt, Mapping) or fmt.get("kind") not in _LABEL_FORMATS:
        return [_t1e(f"task.label_format {fmt!r} is not one of {', '.join(_LABEL_FORMATS)}")]
    if fmt.get("kind") == "delimited" and not fmt.get("sep"):
        return [_t1e("task.label_format delimited names no sep")]
    return []


def _positive(ctx):
    task, eff = ctx.task, ctx.effective_kind
    positive, classes = task.get("positive"), task.get("classes")
    if eff == "anomaly" and task.get("label") and positive is None:
        return [_t1e("an anomaly task with a label needs task.positive — the class that means "
                     "anomalous; the threshold, the readings and the page all read it")]
    if positive is not None and isinstance(classes, (list, tuple)) and classes \
            and str(positive) not in {str(c) for c in classes}:
        return [_t1e(f"task.positive {positive!r} is not one of task.classes {list(classes)}")]
    if eff == "binary" and positive is not None and isinstance(classes, (list, tuple)) \
            and len(classes) == 2 and str(positive) != str(classes[1]):
        return [_t1e(f"task.positive {positive!r} is task.classes[0]; the positive class of a "
                     "binary task must be the second class, which the logit, the threshold and "
                     "every reader take as positive — list the positive class second "
                     f"(task.classes: [{classes[1]!r}, {classes[0]!r}])")]
    return []


def task_fields(ctx):
    if ctx.task_row is None:
        return []
    out = _base_kind(ctx) + _classes(ctx) + _label_format(ctx) + _positive(ctx)
    if not out:
        out.append(ok(f"the {ctx.kind} task carries the fields its readers need", rule="T1e",
                      clause="netspec-contract §1 (task fields)"))
    return out


# -- D2a, D2b ----------------------------------------------------------------

def input_dim(ctx):
    columns = ctx.inputs.output_columns()
    if columns is None:
        return [warn("the shaped manifest carries no output_schema; input_dim was not "
                     "cross-checked", rule="D2a", clause="netspec-contract §3")]
    label, keys = ctx.task.get("label"), ctx.keys
    out = []
    declared = ctx.data_contract.get("keys")
    if declared is not None and sorted(map(str, declared)) != sorted(keys):
        out.append(error(f"data_contract.keys is {list(declared)} but the sample's key columns "
                         f"(series_by, order_by, graph_id) are {list(keys)} — keys are "
                         "derived, never chosen", rule="D2a", clause="netspec-contract §1"))
    # the sample's keys, and any column the shaper itself marks `role: key`: neither is a feature
    roles = ctx.inputs.output_roles() or {}
    skip = list(keys) + [c for c, r in roles.items() if r == "key" and c not in keys]
    want = len([c for c in columns if c != label and c not in skip])
    got = ctx.data_contract.get("input_dim")
    tail = f" and the key column(s) {', '.join(skip)}" if skip else ""
    if got != want:
        out.append(error(f"data_contract.input_dim is {got} but the shaped manifest's output "
                         f"schema has {want} feature column(s) besides the label {label!r}"
                         f"{tail}", rule="D2a", clause="netspec-contract §3"))
    elif not out:
        out.append(ok(f"input_dim {want} agrees with the shaped manifest's output schema",
                      rule="D2a", clause="netspec-contract §3"))
    return out


def _cardinality(ctx, feature, how):
    name = feature.get("name")
    unseen = ctx.inputs.unseen_category_id(name)
    if unseen is None:
        return warn(f"feature {name!r} is {how} but the manifest records no "
                    "unseen_category_id for its encode step; the reserved row "
                    "cannot be checked", rule="D2b", clause="netspec-contract §3")
    if feature.get("cardinality") != unseen + 1:
        return error(f"feature {name!r}: cardinality {feature.get('cardinality')} "
                     f"but the encode step's unseen_category_id is {unseen}, so "
                     f"the table needs {unseen + 1} rows (the last one is "
                     "what an unseen category maps to)", rule="D2b",
                     clause="netspec-contract §3")
    return None


def nominals(ctx):
    embedded = ctx.embedded_columns
    tokenized = any(isinstance(b, Mapping) and b.get("type") == "feature_tokenizer"
                    for _, b, _ in ctx.blocks)
    out = []
    for feature in ctx.features:
        if feature.get("role") != "nominal":
            continue
        name = feature.get("name")
        if name in embedded or tokenized:
            out.append(_cardinality(ctx, feature,
                                    "embedded" if name in embedded else "tokenized"))
            continue
        step = ctx.inputs.encode_step_for(name)
        scheme = ((step or {}).get("parameters") or (step or {}).get("params") or {}).get("scheme")
        if scheme in ("one_hot", "onehot", "one-hot"):
            continue
        out.append(error(f"feature {name!r} is nominal (geometry:typing) and still one column "
                         "in the shaped output, but no embedding block covers it, no "
                         "feature_tokenizer reads it and the shaper did not one-hot it — an "
                         "ordinal-encoded id fed to a dense layer invents an order that does "
                         "not exist", rule="D2b", clause="netspec-contract §3"))
    return [m for m in out if m is not None]
