"""
monitor_plan — the gate a monitor plan passes before a single reference is measured
(bundle-contract §7.5, D2 §7.3).

    monitor_plan.validate_plan(plan, output, schema) -> [problem, …]      (empty: renderable)
    monitor_plan.output_check(output) -> the automatic output check, or None

A plan may use only the kinds the bundle's reading admits (catalog `monitor_kinds`), and every
column, field, class and label a check names must be the bundle's: anything else is a problem,
`not admitted` or `unknown`, which pack reports as `ERROR:` with exit 2. A check that reads the
answer and names no field reads its kind's default (`monitor_reference.resolve_check`);
`output_psi` reads the record's `score_field` and no other field (catalog `reads`). Every
parameter the monitor reads as a number, a limit or a direction must be readable as it reads
it, and the plan must be strict JSON, so a malformed plan is a problem, never a traceback at
measurement or in the smoke. The automatic output check is pack's own: `output_psi` on the
score field where the reading has one, else the decided classes' mix, else the labels' rates.
"""
from __future__ import annotations

import json
import math

from monitor_reference import NAMED, OPS, field_of, resolve_check
from vega_pack_core import CATALOG, monitor_kinds_for

__all__ = ["OUTPUT_CHECK_ID", "OUTPUT_CONCERN", "output_check", "predicate_leaves",
           "validate_plan"]

OUTPUT_CHECK_ID = "auto-output"
OUTPUT_CONCERN = "[vega-pack:output-drift]"
# what a kind cannot be written without (a field has its default: resolve_check)
REQUIRED = {"psi": ("column",), "rate": ("predicate",), "unseen_rate": ("column",),
            "class_recall": ("classes",), "label_rate": ("labels",)}
# the field kinds an answer check reads; a kind whose catalog `reads` is `score_field` reads
# the output record's score_field and no other field
FIELD_READS = {"prediction_mix": ("class",), "class_recall": ("class",),
               "alert_rate": ("alarm",), "residual": ("value", "horizon")}
SCORE_READERS = tuple(kind for kind, row in CATALOG["monitor_kinds"].items()
                      if row.get("reads") == "score_field")
TRUTH = {"class_recall": "class", "residual": "number"}
DECIDED_BY_THRESHOLD = ("alert_rate", "class_recall")
# the parameters the monitor reads as numbers, and what each must be
COUNTS = ("bins_n", "top_k", "min_rows")                   # whole numbers of at least 1
POSITIVE = ("epsilon",)
NUMBERS = ("watch_z", "alert_z", "watch_ratio", "alert_ratio")
LIMITS = ("watch", "alert")                                # the keys of a PSI `thresholds`
DIRECTIONS = ("up", "down", "both")


def validate_plan(plan, output, schema) -> list:
    """Every reason the plan cannot be rendered for this bundle (empty when it can): an unknown
    kind, a kind the reading does not admit, a missing parameter, a column outside the schema's
    input (or a predicate the monitor cannot read), a field outside the output's fields (or of
    a kind the check cannot read), a class or label outside the output's classes, a truth or a
    threshold the bundle does not record, a parameter the monitor cannot read, an id that is
    not a string, is used twice or is reserved, a NaN or an Infinity anywhere."""
    checks = plan.get("checks") if isinstance(plan, dict) else None
    if not isinstance(checks, list) or not checks:
        return ["the plan carries no checks"]
    try:
        json.dumps(plan, allow_nan=False)
    except (TypeError, ValueError):
        return ["the plan is not strict JSON (a NaN, an Infinity): the reference it becomes "
                "could not carry it"]
    ids = {n: c.get("id") for n, c in enumerate(checks)
           if isinstance(c, dict) and c.get("id") is not None}
    named = [i for i in ids.values() if isinstance(i, str)]
    problems = [f"checks[{n}]: id {i!r} is not a string" for n, i in ids.items()
                if not isinstance(i, str)]
    problems += [f"check id {i!r} is used twice"
                 for i in sorted({i for i in named if named.count(i) > 1})]
    if OUTPUT_CHECK_ID in named:
        problems.append(f"check id {OUTPUT_CHECK_ID!r} is the output check pack adds itself")
    inputs = {c.get("name") for c in schema.get("input") or []}
    for n, check in enumerate(checks):
        problems += _check_problems(check, n, output, inputs)
    return problems


def _check_problems(check, n, output, inputs) -> list:
    if not isinstance(check, dict):
        return [f"checks[{n}] is not an object"]
    where, kind = f"check {check.get('id') or f'#{n}'}", check.get("kind")
    if not isinstance(kind, str) or kind not in CATALOG["monitor_kinds"]:
        return [f"{where}: unknown kind {kind!r} (the kinds are "
                f"{', '.join(CATALOG['monitor_kinds'])})"]
    admitted = monitor_kinds_for(output["reading"])
    if kind not in admitted:
        return [f"{where}: kind {kind} is not admitted for the {output['reading']} reading "
                f"(admitted: {', '.join(admitted)})"]
    check = resolve_check(check, output)
    reads_field = kind in FIELD_READS or kind in SCORE_READERS
    missing = [key for key in (*REQUIRED.get(kind, ()), *(("field",) if reads_field else ()))
               if not check.get(key)]
    problems = [f"{kind} needs {', '.join(missing)}"] if missing else \
        [*_column_problems(check, inputs), *_field_problems(check, output),
         *_class_problems(check, output)]
    return [f"{where}: {p}" for p in (*problems, *_truth_problems(check, output),
                                      *_value_problems(check))]


def _column_problems(check, inputs) -> list:
    kind = check["kind"]
    if kind in ("psi", "unseen_rate"):
        return [] if _named_in(check["column"], inputs) else \
            [f"column {check['column']!r} is not an input column of schema.json"]
    if kind != "rate":
        return []
    out = []
    for term in predicate_leaves(check["predicate"]):
        if not isinstance(term, dict):
            out.append("a predicate term is not an object {column, op, value}")
            continue
        if not _named_in(term.get("column"), inputs):
            out.append(f"column {term.get('column')!r} is not an input column of schema.json")
        if term.get("op") not in OPS:
            out.append(f"predicate op {term.get('op')!r} is not one of {', '.join(OPS)}")
        if not _is_number(term.get("value")):
            out.append(f"predicate value {term.get('value')!r} is not a number")
    return out


def _field_problems(check, output) -> list:
    kind, field = check["kind"], check.get("field")
    if kind not in FIELD_READS and kind not in SCORE_READERS:
        return []
    got = (output.get("field_kinds") or {}).get(field) or {} if isinstance(field, str) else {}
    if not got:
        return [f"field {field!r} is not one of MANIFEST.output.fields "
                f"({', '.join(output.get('fields') or [])})"]
    if kind in SCORE_READERS:
        score = output.get("score_field")
        return [] if field == score else \
            [f"{kind} reads the record's score_field {score!r}, not {field!r}"]
    if got.get("kind") not in FIELD_READS[kind]:
        return [f"field {field!r} is a {got.get('kind')} field; {kind} reads a "
                f"{' or '.join(FIELD_READS[kind])} field"]
    step, _, q = str(got.get("of")).partition("/")
    median = min(output.get("quantiles") or [0.5], key=lambda v: abs(float(v) - 0.5))
    if kind == "residual" and q and float(q) != float(median):
        return [f"field {field!r} is the {q} quantile of step {step}; residual reads the point "
                f"forecast, the {median} quantile's field"]
    return []


def _class_problems(check, output) -> list:
    key = NAMED.get(check["kind"])
    named = check.get(key) if key else None
    if named is not None and not isinstance(named, list):
        return [f"{key} is a list of names, not {named!r}"]
    classes = [str(c) for c in output.get("classes") or []]
    return [f"class or label {name!r} is not one of the output's classes "
            f"({', '.join(classes) or 'none'})" for name in named or [] if name not in classes]


def _truth_problems(check, output) -> list:
    kind, truth = check["kind"], output.get("truth") or {}
    want, out = TRUTH.get(kind), []
    if want and truth.get("type") != want:
        out.append(f"{kind} reads a {want} truth, and this bundle takes "
                   f"{truth.get('type') or 'no truth'}")
    column = check.get("label_column") or check.get("target_column")
    if want and column and column != truth.get("column"):
        out.append(f"the truth column is {truth.get('column')!r}, not {column!r}")
    undecided = output["reading"].startswith("anomaly-") and output.get("threshold") is None
    if kind in DECIDED_BY_THRESHOLD and undecided:
        out.append(f"{kind} reads the decision the threshold makes, and this bundle records no "
                   "threshold (threshold.json applies false)")
    return out


def _value_problems(check) -> list:
    """The parameters the monitor reads as numbers, limits or a direction, each readable as the
    monitor reads it: a count a whole number of at least 1, `epsilon` above 0, a limit a finite
    number, `thresholds` an object of `watch` and `alert`, `direction` up, down or both."""
    out = [f"{key} {check[key]!r} is not a whole number of at least 1" for key in COUNTS
           if key in check and not (_is_number(check[key]) and float(check[key]).is_integer()
                                    and check[key] >= 1)]
    out += [f"{key} {check[key]!r} is not a number above 0" for key in POSITIVE
            if key in check and not (_is_number(check[key]) and check[key] > 0)]
    out += [f"{key} {check[key]!r} is not a number" for key in NUMBERS
            if key in check and not _is_number(check[key])]
    limits = check.get("thresholds") or {}            # none: the kind's own limits
    if not isinstance(limits, dict):
        out.append(f"thresholds {limits!r} is not an object {{watch, alert}}")
        limits = {}
    out += [f"thresholds names {key!r}; its limits are watch and alert" for key in limits
            if key not in LIMITS]
    out += [f"thresholds.{key} {value!r} is not a number" for key, value in limits.items()
            if key in LIMITS and not _is_number(value)]
    if "direction" in check and check["direction"] not in DIRECTIONS:
        out.append(f"direction {check['direction']!r} is not one of {', '.join(DIRECTIONS)}")
    return out


def _named_in(name, names) -> bool:
    return isinstance(name, str) and name in names


def _is_number(value) -> bool:
    """A finite JSON number, what the monitor reads with `float()` (a bool is not one)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(value)


def predicate_leaves(spec) -> list:
    """The `{"column", "op", "value"}` terms of a `rate` predicate, through `any` / `all`
    (a malformed node is itself a leaf, for the gate to refuse)."""
    if isinstance(spec, dict) and ("any" in spec or "all" in spec):
        parts = spec.get("any") if "any" in spec else spec.get("all")
        return [t for part in parts for t in predicate_leaves(part)] \
            if isinstance(parts, list) and parts else [None]
    return [spec]


def output_check(output):
    """Pack's own check on the answer: `output_psi` on the score field, else the class mix,
    else the labels' rates — the first the reading admits; None when it admits none."""
    admitted, field = monitor_kinds_for(output["reading"]), output.get("score_field")
    if field and "output_psi" in admitted:
        check, what = {"kind": "output_psi", "field": field}, field
    elif "prediction_mix" in admitted and field_of(output, "class"):
        what = field_of(output, "class")
        check = {"kind": "prediction_mix", "field": what}
    elif "label_rate" in admitted and output.get("classes"):
        check = {"kind": "label_rate", "labels": [str(c) for c in output["classes"]]}
        what = "each label"
    else:
        return None
    return {"id": OUTPUT_CHECK_ID, **check, "concern": OUTPUT_CONCERN,
            "title": f"{what} distribution against the valid part"}
