"""
geometry_layer_shape.py — the structural pass over a `layer-geometry-data` document: every
container the checks walk has its type, every column list holds column names, every identifier
is a string (`references/report-contract.md` §1 types them so).

It runs before the splice and before every other check, and a document it refuses is refused
there: the semantic checks read `body` as the columns of a rule, `features` as the columns a
model sees, `members` as a cycle's columns — a string where a list belongs would iterate as
characters and let a leak through, and a dict where a list belongs would crash the checks.
Requiredness is not its business (an absent or null value passes here and the semantic check
that needs it names it); the type of a value that is present is.

Imported by `apply_geometry_layer.py`; not a command.
"""
from __future__ import annotations

import json

# What each kind accepts, in the words the refusal uses.
KINDS = {
    "object": ("an object", lambda v: isinstance(v, dict)),
    "list": ("a list", lambda v: isinstance(v, list)),
    "string": ("a non-empty string", lambda v: isinstance(v, str) and bool(v.strip())),
    "names": ("a list of column names",
              lambda v: isinstance(v, list) and all(_is_name(x) for x in v)),
    "names+": ("a non-empty list of column names",
               lambda v: isinstance(v, list) and bool(v) and all(_is_name(x) for x in v)),
}

# (path, kind). A path is dot-separated keys; `[]` stands for each item of a list and `{}` for
# each value of an object. A parent of the wrong type is reported by its own, earlier, entry.
SHAPE = (
    ("from_run", "string"), ("source", "object"), ("source.column_names", "names"),
    ("markers", "list"),
    ("space", "object"), ("basis", "object"), ("basis.members", "names"),
    ("typing", "list"), ("typing.[]", "object"), ("typing.[].column", "string"),
    ("typing.[].role", "string"), ("typing.[].final_type", "string"),
    ("derivations", "list"), ("derivations.[]", "object"), ("derivations.[].column", "string"),
    ("derivations.[].rule_id", "string"), ("derivations.[].layer", "string"),
    ("derivations.[].formula", "string"), ("derivations.[].body", "names+"),
    ("derivations.[].cycle", "names"),
    ("cycles", "list"), ("cycles.[]", "object"), ("cycles.[].id", "string"),
    ("cycles.[].members", "names"), ("cycles.[].orientations", "list"),
    ("cycles.[].orientations.[]", "object"), ("cycles.[].orientations.[].id", "string"),
    ("cycles.[].orientations.[].derived", "names"), ("cycles.[].orientations.[].basis", "names"),
    ("cycles.[].orientations.[].derivations", "list"),
    ("cycles.[].orientations.[].derivations.[]", "object"),
    ("cycles.[].orientations.[].derivations.[].column", "string"),
    ("cycles.[].orientations.[].derivations.[].body", "names+"),
    ("cycles.[].orientations.[].rules", "list"), ("cycles.[].orientations.[].rules.[]", "object"),
    ("cycles.[].orientations.[].rules.[].column", "string"),
    ("cycles.[].orientations.[].rules.[].body", "names+"),
    ("disagreements", "list"), ("disagreements.[]", "object"),
    ("disagreements.[].column", "string"), ("disagreements.[].kind", "string"),
    ("orthogonality", "object"), ("orthogonality.pairs", "list"), ("orthogonality.pairs.[]", "object"),
    ("orthogonality.pairs.[].a", "string"), ("orthogonality.pairs.[].b", "string"),
    ("partitions", "object"), ("partitions.chosen", "string"), ("partitions.candidates", "list"),
    ("partitions.candidates.[]", "object"), ("partitions.candidates.[].label", "string"),
    ("partitions.candidates.[].rule_id", "string"), ("partitions.candidates.[].task", "string"),
    ("partitions.candidates.[].stance", "string"),
    ("partitions.candidates.[].features", "names"),
    ("partitions.candidates.[].dropped_for_leakage", "names"),
    ("handoff", "object"), ("stats", "object"), ("stats.{}", "object"),
    ("explore", "object"), ("explore.columns", "object"), ("explore.columns.{}", "list"),
    ("explore.pca", "object"), ("explore.index", "list"),
    ("columns", "object"), ("columns.{}", "object"), ("dependencies", "object"),
    ("dependencies.pairs", "list"), ("dependencies.pairs.[]", "object"),
    ("functional_dependencies", "list"), ("functional_dependencies.[]", "object"),
    ("functional_dependencies.[].lhs", "names"), ("functional_dependencies.[].rhs", "string"),
    ("linear_derivability", "list"), ("linear_derivability.[]", "object"),
    ("linear_derivability.[].column", "string"),
)


def _is_name(x):
    return isinstance(x, str) and bool(x.strip())


def _walk(node, steps, where):
    """(where, value) for every present, non-null value the path reaches."""
    if node is None:
        return
    if not steps:
        yield where, node
        return
    step, rest = steps[0], steps[1:]
    if step == "[]":
        if isinstance(node, list):
            for i, v in enumerate(node):
                yield from _walk(v, rest, f"{where}[{i}]")
    elif step == "{}":
        if isinstance(node, dict):
            for k, v in node.items():
                yield from _walk(v, rest, f"{where}.{k}")
    elif isinstance(node, dict) and step in node:
        yield from _walk(node[step], rest, f"{where}.{step}" if where else step)


def _shown(v):
    text = json.dumps(v, ensure_ascii=False, default=str)
    return text if len(text) <= 60 else text[:57] + "..."


def check_shape(doc):
    """The refusals of the structural pass, one line each (empty when the shape holds)."""
    if not isinstance(doc, dict):
        return ["the document is not a JSON object"]
    errs = []
    for path, kind in SHAPE:
        desc, ok = KINDS[kind]
        for where, v in _walk(doc, path.split("."), ""):
            if not ok(v):
                errs.append(f"{where} must be {desc}, not {_shown(v)}")
    return errs
