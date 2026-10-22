"""
output_record — what the bundle answers and how its rows are assembled: `MANIFEST.output`
(`vega/output@2`), `MANIFEST.sample` and `schema.json`.

    output_record.records(ctx) -> {"output": …, "sample": …}             (the first phase)
    output_record.write_schema(campaign, root, markers) -> schema         (before the context)
    output_record.resolve_reading(spec) -> str                            PackExit(2) for none
    output_record.output_record(spec, threshold, truth_column) -> dict    vega/output@2
    output_record.sample_record(spec, recipe, graph_meta=None) -> dict    MANIFEST.sample
    output_record.lag_depth(recipe) -> int
    output_record.field_kinds(record) -> {field: {"kind", "of"}}
    output_record.schema_output(record, sample=None) -> {"output", "truth"[, "sample"]}

**The reading, not the task, drives serving** (bundle-contract §2.1). A spec resolves to one of
ten readings through the rules of `catalog.json` `output_readings.resolve`, and every fact of a
reading — link, decision, field patterns and their kinds, the compared and decided pairs, the
descriptions — is that table's; this module resolves the spec and gathers what fills the
patterns (`field_patterns` fills them: a placeholder is its *token* in a field name, `q_2p5`,
and its *value* in `of`, 0.025, which is what the answer codec keys its JSON objects with).

`records` is what the Predictor reads at construction — `pack.py` writes it into the manifest
stub before the round trip and into the manifest after it — and `write_schema` derives the
schema's output half from the same record, so the two can never describe different frames.
A graph sample's `reference` counts are read off the shaped manifest's `graph` block (train
and valid nodes, the edges among them); `sample_record` takes `graph/graph.json` as well, so the
phase that writes the reference files can record their own counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import field_patterns as fp
import input_schema
from pack_context import PackExit, out_ok
from vega_pack_core import CATALOG

__all__ = ["field_kinds", "lag_depth", "output_record", "records", "resolve_reading",
           "sample_record", "schema_output", "write_schema"]

TABLE = CATALOG["output_readings"]
SHAPES = CATALOG["sample_shapes"]
THRESHOLD_DECISIONS = ("threshold", "per-label-threshold")
SHAPE_DESCRIBE = {
    "graph_id": "the graph this row belongs to, as the request's graph_id column named it",
    "answered": ("true when the row's window was complete; false, with a warning naming the "
                 "rows it lacked, when fewer than length-1 rows of its series preceded it"),
    "warnings": "a string per row: unseen categories and values outside the ranges above",
}


# --------------------------------------------------------------------------- the records

def records(ctx) -> dict:
    """`MANIFEST.output` and `MANIFEST.sample` for this bundle."""
    campaign = ctx.campaign
    output = output_record(ctx.spec, campaign.threshold, _truth_column(campaign))
    sample = sample_record(ctx.spec, campaign.recipe or {}, _graph_meta(campaign))
    out_ok(f"output: reading {output['reading']} (link {output['link']}, decision "
           f"{output['decision']}), {len(output['fields'])} field(s), "
           f"{len(output['compared'])} compared and {len(output['decided'])} decided pair(s); "
           f"sample {sample['shape']}", ctx.markers)
    return {"output": output, "sample": sample}


def _truth_column(campaign):
    return campaign.label or ((campaign.spec or {}).get("task") or {}).get("label")


def _graph_meta(campaign):
    return (campaign.shaped_manifest or {}).get("graph")


# --------------------------------------------------------------------------- the reading

def resolve_reading(spec) -> str:
    """The spec's reading (§2.1), or `PackExit(2)`: `task.kind <k> has no serving reading`."""
    return _resolution(spec)[0]


def _resolution(spec, kind=None):
    """(reading, base): the first rule whose kind matches and whose `when` holds; a sequence
    kind reads as its base, resolved through the same rules with the same spec."""
    kind = kind or (spec.get("task") or {}).get("kind")
    for rule in TABLE["resolve"]:
        if rule["kind"] != kind or not _holds(rule.get("when"), spec):
            continue
        base = rule.get("base") or {}
        if rule["reading"] != "@base":
            return rule["reading"], base.get("fixed")
        chosen = _base_kind(spec, base)
        if chosen not in (base.get("allowed") or []):
            raise PackExit(2, f"task.kind {kind} has no serving reading: its base "
                              f"{chosen!r} is not one of {', '.join(base.get('allowed') or [])}")
        return _resolution(spec, chosen)[0], chosen
    raise PackExit(2, f"task.kind {kind} has no serving reading")


def _base_kind(spec, base):
    """`task.base_kind`, else the v2.2 fallback read off the head and the loss."""
    stated = (spec.get("task") or {}).get("base_kind") if base.get("from") else None
    return stated or next((f["base"] for f in base.get("v22_fallback") or []
                           if _holds(f.get("when"), spec)), None)


def _holds(when, spec) -> bool:
    loss, head_out = (spec.get("loss") or {}).get("name"), _head_out(spec)
    tests = {"loss": lambda want: loss in want,
             "head_out_gt": lambda want: head_out is not None and head_out > want,
             "head_out_eq": lambda want: head_out == want,
             "has": lambda want: _dotted(spec, want) is not None}
    return all(tests[key](want) for key, want in (when or {}).items())


def _head_out(spec):
    """`params.out` of the spec's last `head` block."""
    heads = [b for b in spec.get("blocks") or [] if b.get("type") == "head"]
    try:
        return int((heads[-1].get("params") or {}).get("out")) if heads else None
    except (TypeError, ValueError):
        return None


def _dotted(spec, path):
    node = spec
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


# --------------------------------------------------------------------------- the output record

def output_record(spec, threshold, truth_column) -> dict:
    """`MANIFEST.output` (`vega/output@2`): the reading, its fields by name with their kinds,
    what the round trip compares and decides, the threshold on its scale, and the truth.

    `threshold` is `threshold.json` (or None); `truth_column` the label column a caller may
    send. Every compared and decided field is one of `fields`, or `PackExit(3)`."""
    reading, base = _resolution(spec)
    row = TABLE["readings"][reading]
    facts = _facts(spec, reading)
    fields = _fields(spec, row, facts)
    record = {"schema": TABLE["schema"], "kind": (spec.get("task") or {}).get("kind"),
              "reading": reading, "base": base, "link": row["link"], "decision": row["decision"],
              "head_out": facts["head_out"], "classes": facts["classes"],
              "positive": facts["positive"], "quantiles": facts["quantiles"],
              "horizons": facts["horizons"], "fields": fields}
    record.update(_threshold_record(threshold, row, reading))
    record["field_kinds"] = field_kinds(record)
    record["compared"] = _compared(row, facts)
    bound = fp.binding(facts)
    record["decided"] = [[name, source] for bundle, source in row["decided"]
                         if fp.gated(row, bundle, facts) for name, _ in fp.expand(bundle, bound)]
    record["score_field"] = next((name for pattern in row["score_field"]
                                  for name, _ in fp.expand(pattern, bound) if name in fields),
                                 None)
    record["truth"] = _truth(row, facts, truth_column, spec)
    _check_closed(record)
    return {key: record[key] for key in TABLE["record_keys"]}


def _facts(spec, reading) -> dict:
    """What fills the patterns: classes, positive, quantiles, horizons, head width, features."""
    task, loss = spec.get("task") or {}, spec.get("loss") or {}
    head_out = _head_out(spec)
    wants_q = reading in ("quantile", "forecast") and loss.get("name") == "quantile"
    quantiles = [float(q) for q in (loss.get("params") or {}).get("quantiles") or []] \
        if wants_q else []
    classes, positive = _classes(task, reading, head_out)
    contract = spec.get("data_contract") or {}
    features = [f.get("name") for role in ("numeric", "nominal")
                for f in contract.get("features") or [] if f.get("role") == role]
    horizon = (contract.get("sample") or {}).get("horizon") or \
        (head_out or 0) // max(1, len(quantiles))
    return {"classes": classes, "positive": positive, "quantiles": quantiles or None,
            "horizons": list(range(1, int(horizon) + 1)) if reading == "forecast" else None,
            "head_out": head_out, "features": features}


def _classes(task, reading, head_out):
    """(class names in the reading's order, the positive class). Binary, and an anomaly task
    with classes, read `[negative, positive]`: `positive` is `task.positive`, else the last
    class, else "1"."""
    names = [str(c) for c in task.get("classes") or []]
    if reading == "binary" or (reading.startswith("anomaly-") and names):
        positive = str(task["positive"]) if task.get("positive") is not None else \
            (names[-1] if names else "1")
        negative = next((c for c in names if c != positive), "0" if positive != "0" else "1")
        return [negative, positive], positive
    if reading in ("multiclass", "multilabel", "ordinal"):
        width = (head_out or 0) + (1 if reading == "ordinal" else 0)
        return names or [str(k) for k in range(width)], None
    return None, None


def _fields(spec, row, facts) -> list:
    """The concrete field names in order: `graph_id` first at graph level, the reading's
    patterns filled, `answered` for a window, `warnings` last. Duplicates are refused."""
    sample = (spec.get("data_contract") or {}).get("sample") or {}
    bound = fp.binding(facts)
    named = [name for pattern in row["fields"] if fp.gated(row, pattern, facts)
             for name, _ in fp.expand(pattern, bound)]
    graph_level = sample.get("shape") == "graph" and sample.get("level") == "graph"
    fields = (["graph_id"] if graph_level else []) + named + \
        (["answered"] if sample.get("shape") == "window" else []) + ["warnings"]
    twice = sorted({f for f in fields if fields.count(f) > 1})
    if twice:
        raise PackExit(2, f"the output fields {', '.join(twice)} are not unique: two class, "
                          "label or feature names read as the same field name")
    return fields


def field_kinds(record) -> dict:
    """Every field of the record -> `{"kind", "of"}` (bundle-contract §7.3), re-derived from the
    reading's patterns and the record's own classes, quantiles, horizons and width."""
    named, shape = _named(record), TABLE["shape_fields"]
    unknown = [f for f in record["fields"] if f not in named and f not in shape]
    if unknown:
        raise PackExit(3, f"field(s) {', '.join(unknown)} match no pattern of the "
                          f"{record['reading']} reading")
    return {f: named[f][1] if f in named else {"kind": shape[f]["kind"], "of": None}
            for f in record["fields"]}


def _named(record) -> dict:
    """name -> (pattern, {"kind", "of"}) for this record: its own classes, quantiles, horizons
    and width, and the features its `recon_` fields name."""
    facts = {key: record.get(key) for key in ("classes", "positive", "quantiles", "horizons",
                                              "head_out")}
    facts["features"] = [f[len("recon_"):] for f in record["fields"] if f.startswith("recon_")]
    return fp.named(TABLE["readings"][record["reading"]], facts)


def _compared(row, facts) -> list:
    """[bundle field, stored column] pairs: `score_<k>` numbers the pattern's names in order,
    and is the one column `score` when the head has a single output."""
    bound, pairs = fp.binding(facts), []
    for bundle, stored in row["compared"]:
        if not fp.gated(row, bundle, facts):
            continue
        for k, (name, _) in enumerate(fp.expand(bundle, bound)):
            column = "score" if stored == "score_<k>" and facts["head_out"] == 1 else \
                stored.replace("<k>", str(k))
            pairs.append([name, column])
    return pairs


def _threshold_record(threshold, row, reading) -> dict:
    """The decision threshold on the reading's scale: null where no threshold applies (argmax,
    cumulative, none) and where `threshold.json.applies` is false."""
    doc = threshold or {}
    scale = row["threshold_scale"]
    if doc.get("scale") and scale and doc["scale"] != scale:
        raise PackExit(2, f"threshold.json is on the {doc['scale']} scale, and a {reading} "
                          f"reading decides on the {scale} scale; re-run /vega-eval")
    applies = row["decision"] in THRESHOLD_DECISIONS and doc.get("applies") is not False
    value = float(doc["value"]) if applies and doc.get("value") is not None else None
    per_label = dict(doc["per_label"]) if applies and reading == "multilabel" and \
        doc.get("per_label") else None
    source = doc.get("source") if value is not None or per_label else None
    return {"threshold": value, "threshold_source": source, "threshold_scale": scale,
            "threshold_per_label": per_label}


def _truth(row, facts, column, spec):
    """`{"column", "type", "sep"}` (R-A8), or null: no label column, or a reading whose truth
    is optional and has no classes (an unlabelled anomaly task, a reconstruction)."""
    kind = row["truth"] if (facts["classes"] or not row["truth_optional"]) else None
    if kind is None or not column:
        return None
    label_format = (spec.get("task") or {}).get("label_format") or {}
    return {"column": column, "type": kind,
            "sep": (label_format.get("sep") or "|") if kind == "labels" else None}


def _check_closed(record) -> None:
    """Every compared and decided bundle field is one of `fields` (bundle-contract §2.1)."""
    outside = [pair[0] for pair in record["compared"] + record["decided"]
               if pair[0] not in record["fields"]]
    if outside:
        raise PackExit(3, f"MANIFEST.output compares or decides {', '.join(outside)}, which "
                          "its fields do not name")


# --------------------------------------------------------------------------- the sample record

def sample_record(spec, recipe, graph_meta=None) -> dict:
    """`MANIFEST.sample` (bundle-contract §2.2) from `data_contract.sample`. `graph_meta` is
    `graph/graph.json`, or the shaped manifest's `graph` block before that file exists."""
    sample = (spec.get("data_contract") or {}).get("sample") or {}
    shape = sample.get("shape") or "row"
    if shape == "window":
        return _window_record(spec, sample, recipe)
    if shape == "graph":
        return _graph_record(sample, graph_meta or {})
    if shape != "row":
        raise PackExit(2, f"data_contract.sample.shape {shape!r} is not one of "
                          f"{', '.join(SHAPES)}")
    return {"shape": "row"}


def _window_record(spec, sample, recipe) -> dict:
    """R-A5: `horizon` and `autoregressive` for a forecast only; `history_rows` is what a caller
    sends so that every row is answered with its lag context fully determined."""
    length = sample.get("length")
    if not isinstance(length, int) or isinstance(length, bool) or length < 2:
        raise PackExit(2, f"data_contract.sample.length is {length!r}; a window sample needs "
                          "an integer length >= 2")
    depth = lag_depth(recipe)
    table = SHAPES["window"]
    values = {"shape": "window", "length": length, "stride": int(sample.get("stride") or 1),
              "horizon": sample.get("horizon"), "series_by": sample.get("series_by"),
              "order_by": sample.get("order_by"),
              "context": sample.get("context") or table["contexts"][0],
              "autoregressive": sample.get("autoregressive") is not False,
              "lag_depth": depth, "history_rows": (length - 1) + depth, "rule": table["rule"]}
    forecast = (spec.get("task") or {}).get("kind") == "forecast"
    return {key: values[key] for key in table["record"]
            if forecast or key not in table["forecast_only"]}


def _graph_record(sample, meta) -> dict:
    """The inference graph (R-A3): the reference is train ∪ valid, its files `graph_pack`'s."""
    table = SHAPES["graph"]
    within, cross = meta.get("within_part_edges"), meta.get("cross_part_edges") or {}
    edges = (sum(int(within.get(p) or 0) for p in table["reference_parts"])
             + int(cross.get("-".join(table["reference_parts"])) or 0)) \
        if isinstance(within, dict) else meta.get("n_edges")
    nodes = meta.get("n_nodes")
    reference = {"parts": list(table["reference_parts"]),
                 "nodes": sum(int(nodes.get(p) or 0) for p in table["reference_parts"])
                 if isinstance(nodes, dict) else nodes,
                 "edges": edges, "files": dict(zip(("nodes", "edges", "meta"), table["files"]))}
    values = {"shape": "graph", "level": sample.get("level") or "node",
              "protocol": sample.get("protocol") or "inductive", "method": meta.get("method"),
              "directed": bool(sample.get("directed", meta.get("directed", False))),
              "self_loops": bool(sample.get("self_loops", meta.get("self_loops", False))),
              "graph_id": sample.get("graph_id"), "reference": reference, "rule": table["rule"]}
    return {key: values[key] for key in table["record"]}


def lag_depth(recipe) -> int:
    """The largest lag of the recipe's `lag` steps, 0 without one. A lag step with no `lags`
    lags by 1, as the shaper reads it (`shape.py`: `p.get("lags") or [1]`)."""
    depth = 0
    for step in (recipe or {}).get("steps") or []:
        if step.get("op") == "lag":
            lags = (step.get("params") or {}).get("lags") or [1]
            depth = max([depth] + [int(v) for v in (lags if isinstance(lags, list) else [lags])])
    return depth


# --------------------------------------------------------------------------- the schema

def schema_output(record, sample=None) -> dict:
    """The output half of `schema.json`: one description per field pattern present in the
    record (the catalog's `describe`, `<positive>` and `<median>` filled), the shape's own
    fields, the truth the monitor takes, and for a window the history a caller sends."""
    row = TABLE["readings"][record["reading"]]
    named, present = _named(record), set(record["fields"])
    patterns = {named[name][0] for name in present if name in named}
    bound = fp.binding({"positive": record.get("positive"),
                        "quantiles": record.get("quantiles")})
    single = {hole: bound[hole][0][0] for hole in ("positive", "median") if bound[hole]}
    output = {name: SHAPE_DESCRIBE[name] for name in ("graph_id",) if name in present}
    output.update({pattern: fp.HOLE.sub(lambda m: single.get(m.group(1), m.group(0)),
                                        row["describe"][pattern])
                   for pattern in row["fields"] if pattern in patterns})
    output.update({name: SHAPE_DESCRIBE[name] for name in ("answered", "warnings")
                   if name in present})
    out = {"output": output, "truth": record.get("truth")}
    if (sample or {}).get("shape") == "window":
        out["sample"] = {"history_rows": sample["history_rows"]}
    return out


def write_schema(campaign, root, markers) -> dict:
    """`schema.json` at the bundle root: what a caller sends (`input_schema`), what comes back
    and the truth it may send (`schema_output`), derived on the train part's raw rows."""
    sample = sample_record(campaign.spec, campaign.recipe or {}, _graph_meta(campaign))
    record = output_record(campaign.spec, campaign.threshold, _truth_column(campaign))
    inputs = input_schema.derive(campaign, sample, markers)
    schema = {"input": inputs["input"], **schema_output(record, sample),
              "ignored": inputs["ignored"]}
    if inputs.get("label"):
        schema["label"] = inputs["label"]
    (Path(root) / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    out_ok(f"schema.json — {len(schema['input'])} input column(s), {len(schema['ignored'])} "
           f"ignored, output {record['reading']}: {len(record['fields'])} field(s) described by "
           f"{len(schema['output'])} pattern(s)", markers)
    return schema
