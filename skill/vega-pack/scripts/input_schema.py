"""
input_schema — what a caller sends: the input half of `schema.json` (bundle-contract §3).

    input_schema.derive(campaign, sample, markers) -> {"input": […], "ignored": […][, "label"]}
    input_schema.label_decoding(raw_train, shaped_train_path, label, markers) -> dict | None

The columns are read off the raw rows the recipe consumed on the train part — never off the
shaped columns, which are what the network sees and not what a user has — and "seen in
training" means the train part: the ranges and the category lists are read on its raw rows,
never on the valid or test rows the network was measured on. `output_record.write_schema`
composes this half with the output half it derives from the output record.

The sample shape adds inputs the recipe's feature steps alone do not name. An autoregressive
forecast reads its target's past as an input channel, so the target column is an input with
`"role": "autoregressive"`, required in a request's `rows` and `history` (R-A7), not an ignored
label. A graph-level bundle groups a request's rows into graphs by the raw `graph_id` column,
so that column is a required string input whatever the raw file's type. And a graph bundle
attaches each request row to the graph through the columns its `graph_edges` step reads at the
rows' raw positions — the key columns of `key_pair`, a weight column, the coordinates of
`spatial` — which the Recipe's `edges_for` / `edges_within` read from the request (the shaped
manifest's `graph` block names them): each is a required input with `"role": "graph"`, typed as
the raw file types it, its domain `"allowed": "any"` — never the values or the range the train
part held (a query's keys are new by construction, and the train part's are the reference
graph's, not the caller's). The graph-level group column's domain is `"any"` for the same reason.
"""
from __future__ import annotations

import re
from pathlib import Path

from pack_context import out_ok, out_warn
from vega_pack_core import read_parquet

__all__ = ["derive", "label_decoding"]

DROP_OPS = ("drop_identity", "drop_leakage", "drop_constant", "drop_derived")
AUTOREGRESSIVE_NOTE = ("the target's own past: the forecast reads it as an input channel, so "
                       "it is required in rows and in history (and ignored as a label)")
GROUP_NOTE = ("the graph each row belongs to: the rows sharing a value are answered as one "
              "graph, with the edges between them")
GRAPH_NOTE = ("read by the graph step, not by the network: a row is attached to the reference "
              "graph (and, at graph level, to the rows of its own graph) through this column")
GRAPH_ENTRY_DROPS = ("allowed", "distinct_seen", "pattern", "range_seen", "non_finite_seen",
                     "note")


def derive(campaign, sample, markers) -> dict:
    """`{"input", "ignored"}` of `schema.json`, and `label` when the recipe decodes the label."""
    train_positions = campaign.split.get("train")
    raw = campaign.part_rows("train") if train_positions else campaign.raw
    if not train_positions:
        out_warn("fitted.json records no train positions; schema ranges read on the whole "
                 "raw file", markers)
    consumed, ignored = _consumed_columns(campaign.recipe, campaign.shaped_manifest, raw)
    nominal = _nominal_columns(campaign.recipe)
    columns = [_input_column(name, raw[name], name in nominal) for name in consumed]
    target = _label_column(campaign.recipe, campaign.shaped_manifest)
    columns, ignored = _sample_inputs(columns, ignored, raw, sample or {}, target)
    columns, ignored = _graph_inputs(columns, ignored, raw, sample or {},
                                     (campaign.shaped_manifest or {}).get("graph") or {})
    out = {"input": columns, "ignored": ignored}
    train_path = _shaped_train_path(campaign)
    decoding = (label_decoding(raw, train_path, campaign.label, markers)
                if train_positions and train_path else None)
    if decoding:
        out["label"] = decoding
    return out


def _sample_inputs(columns, ignored, raw, sample, target):
    """The inputs the sample shape adds: the autoregressive target, the graph-level group."""
    if sample.get("autoregressive") and target in raw.columns:
        entry = dict(_input_column(target, raw[target], False), role="autoregressive",
                     note=AUTOREGRESSIVE_NOTE)
        columns = [c for c in columns if c["name"] != target] + [entry]
        ignored = [c for c in ignored if c != target]
    graph_level = sample.get("shape") == "graph" and sample.get("level") == "graph"
    group = sample.get("graph_id") if graph_level else None
    if group and group in raw.columns:
        entry = {"name": group, "required": True, "dtype": "string", "allowed": "any",
                 "note": GROUP_NOTE}
        listed = any(c["name"] == group for c in columns)
        columns = [entry if c["name"] == group else c for c in columns] + \
            ([] if listed else [entry])
        ignored = [c for c in ignored if c != group]
    return columns, ignored


def _graph_inputs(columns, ignored, raw, sample, graph):
    """The columns a graph bundle's `edges_for` / `edges_within` read from a request (the key
    columns, the coordinates, a weight column of the shaped manifest's `graph` block): required
    inputs with `"role": "graph"`, never `ignored` — the service passes an ignored column to no
    one, and the Recipe then cannot attach a request row to the graph."""
    if sample.get("shape") != "graph":
        return columns, ignored
    read = [c for c in graph_columns(graph) if c in raw.columns]
    listed = {c["name"] for c in columns}          # an input already: the network reads it too
    added = [dict({k: v for k, v in _input_column(name, raw[name], False).items()
                   if k not in GRAPH_ENTRY_DROPS}, allowed="any", role="graph", note=GRAPH_NOTE)
             for name in read if name not in listed]
    return columns + added, [c for c in ignored if c not in read]


def graph_columns(graph) -> list:
    """The raw columns the `graph_edges` step reads at the rows' raw positions, from the shaped
    manifest's `graph` block: `key_pair`'s key columns, a weight column, and every column
    `edges_for` reads of a reference node (`edges_for.raw`: keys, a weight, `spatial`'s
    coordinates)."""
    params = graph.get("params") or {}
    method = params.get("method") or graph.get("method")
    named = list(params.get("columns") or []) if method == "key_pair" else []
    if params.get("weight") not in (None, "distance"):
        named.append(params["weight"])
    named += list(((graph.get("edges_for") or {}).get("raw") or {}).values())
    return [c for c in dict.fromkeys(named) if c]


def _nominal_columns(recipe) -> set:
    # A `retype ... to nominal` step is the recipe saying this column is a category, and it
    # says so precisely because the raw file does not: a postal code parses as an integer and
    # would otherwise be published with a range, inviting a caller to send 60041 as if the
    # space between codes meant something.
    nominal = set()
    for step in recipe.get("steps") or []:
        if step.get("op") == "retype" and (step.get("params") or {}).get("to") == "nominal":
            nominal.update(step.get("columns") or [])
    return nominal


def _input_column(name, series, nominal) -> dict:
    entry = {"name": name, "required": True}
    if nominal:
        entry["dtype"] = "string"
        seen = sorted({str(v) for v in series.dropna().unique().tolist()})
        entry["allowed"] = seen if len(seen) <= 64 else "any"
        entry["distinct_seen"] = len(seen)
        if seen and all(re.fullmatch(r"[0-9]+", v) for v in seen):
            # every code training saw is a run of digits; the service holds callers to it
            entry["pattern"] = "^[0-9]+$"
        entry["note"] = ("the recipe types this column nominal; an unseen value is "
                         "answered and flagged, not refused")
        return entry
    if _is_numeric(series):
        entry.update(_numeric_entry(series))
        return entry
    entry["dtype"] = "string"
    seen = sorted({str(v) for v in series.dropna().unique().tolist()})
    entry["allowed"] = seen if len(seen) <= 64 else "any"
    if len(seen) > 64:
        entry["distinct_seen"] = len(seen)
    return entry


def _numeric_entry(series) -> dict:
    import numpy as np
    import pandas as pd
    entry = {"dtype": "float" if _has_fraction(series) else "int"}
    # finite values only: an infinity is not a bound (and is not JSON), and the recipe treats
    # it as missing — how many there were is said beside the range
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype="float64"))]
    entry["range_seen"] = ([_round(finite.min()), _round(finite.max())] if len(finite)
                           else None)
    non_finite = int(numeric.notna().sum() - len(finite))
    if non_finite:
        entry["non_finite_seen"] = non_finite
    return entry


def _label_column(recipe, shaped_manifest):
    return (recipe or {}).get("target") or (shaped_manifest or {}).get("label")


def _consumed_columns(recipe, shaped_manifest, raw):
    """The raw columns a caller must supply, and the ones the recipe throws away.

    A drop step names its columns only when the author knew them: `drop_constant` and
    `drop_derived` discover theirs while running, so the recipe alone does not say what they
    took. The executed manifest does — each step records `columns_removed` — and reading it
    is the difference between an input schema that asks a caller for a column the recipe
    immediately discards and one that does not.

    A dropped column stays listed as `ignored` rather than vanishing: a caller sending whole
    rows should not be told a column is unknown when the recipe simply has no use for it.
    """
    dropped, label = set(), _label_column(recipe, shaped_manifest)
    for step in recipe.get("steps") or []:
        if step.get("op") in DROP_OPS:
            dropped.update(step.get("columns") or [])
    for step in shaped_manifest.get("steps") or []:
        if step.get("op") in DROP_OPS:
            dropped.update(step.get("columns_removed") or [])
    ignored = sorted({c for c in raw.columns if c in dropped} | ({label} if label else set()))
    consumed = [c for c in raw.columns if c not in ignored]
    return consumed, ignored


def _is_numeric(series):
    import pandas as pd
    return pd.api.types.is_numeric_dtype(series)


def _has_fraction(series):
    import numpy as np
    values = series.dropna().to_numpy()
    return bool(len(values)) and not np.all(np.equal(np.mod(values, 1), 0))


def _round(value, places=6):
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- the label

def _shaped_train_path(campaign):
    """The shaped train part's file, as the shaped manifest names it."""
    train_file = (((campaign.shaped_manifest.get("outputs") or {}).get("files") or {})
                  .get("train") or {})
    train_path = Path(train_file.get("path", "")) if isinstance(train_file, dict) else None
    if train_path is not None and not train_path.is_absolute():
        shaped_dir = campaign.shaped_dir
        train_path = next((c for c in (shaped_dir.parent / train_path,
                                       shaped_dir / train_path.name, Path.cwd() / train_path)
                           if c.is_file()), None)
    return train_path


def label_decoding(raw_train, shaped_train_path, label, markers) -> dict | None:
    """How the recipe spells each raw label value as a class, read off its own output.

    The train part's raw rows and its shaped rows are the same rows in the same order
    (`__split_raw__`); pairing their labels gives the recipe's decoding as data — a lossy-source
    repair such as a mis-decoded byte included — without re-implementing the step that did it.
    Only the pairs that change something are kept. A raw value mapping to two classes would mean
    the pairing is wrong, and the schema then carries no decoding rather than a guessed one.

    A missing raw label is not a spelling: it is a row without truth (a multilabel row with no
    label at all, an unlabelled row), and the service reads a missing truth as missing, never
    through the decoding. Those rows are left out of the pairing, so the decoding's keys are
    texts the caller can send — never a missing value, which JSON could not even hold as a key.
    A raw spelling the recipe turns into a missing label is not a class either; it is reported
    and not recorded.
    """
    import pandas as pd
    if not label or raw_train is None or label not in raw_train.columns:
        return None
    shaped = read_parquet(shaped_train_path, columns=[label])
    if len(shaped) != len(raw_train):
        out_warn("the shaped train part and its raw rows differ in length; no label decoding "
                 "recorded", markers)
        return None
    raw_values = raw_train[label].reset_index(drop=True)
    cls_values = shaped[label].reset_index(drop=True)
    unlabelled = raw_values.isna().to_numpy()
    blanked = (~unlabelled) & cls_values.isna().to_numpy()
    kept = ~(unlabelled | blanked)
    pairs = pd.DataFrame({"raw": raw_values[kept].astype(str).to_numpy(dtype=object),
                          "cls": cls_values[kept].astype(str).to_numpy(dtype=object)}
                         ).drop_duplicates()
    if pairs["raw"].duplicated().any():
        out_warn("a raw label maps to two classes on the train part; no label decoding recorded",
                 markers)
        return None
    decode = {str(r): str(c) for r, c in zip(pairs["raw"], pairs["cls"]) if r != c}
    if blanked.any():
        out_warn(f"{int(blanked.sum())} train row(s) carry a raw label the recipe turns into no "
                 "label; those spellings are not classes and are not recorded in the decoding",
                 markers)
    out_ok(f"label decoding: {len(pairs)} raw spelling(s) on the train part, {len(decode)} "
           "repaired by the recipe — recorded in schema.json, applied by the service"
           + (f"; {int(unlabelled.sum())} row(s) without a raw label are rows without truth, "
              "not a spelling" if unlabelled.any() else ""), markers)
    return {"name": label, "decode": decode,
            "source": "the recipe's output on the train part (raw -> shaped label pairs)"}
