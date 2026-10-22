"""
geometry_layer_cases.py — the documents `test_geometry_layer.py` feeds the validator: mutations
of the orders fixture it must refuse (each with the words its refusal must carry) or accept, the
retyped variant whose basis the explorer must agree on, and a generated file with every column
kind. Data, not tests: the unit tier runs only `test_*.py`.
"""
from __future__ import annotations

import copy


def mutate(base_doc, fn):
    d = copy.deepcopy(base_doc)
    fn(d)
    return d


def deriv(d, col):
    return next(x for x in d["derivations"] if x["column"] == col)


def cand(d):
    return d["partitions"]["candidates"][0]


def cycle(d, cid):
    return next(c for c in d["cycles"] if c["id"] == cid)


def typed(d, col):
    return next(t for t in d["typing"] if t["column"] == col)


def _cq(decision):
    return {"decision": decision, "alternatives": [], "downstream": "no downstream consequence"}


def _body_as_text(d):
    """The label's rule body written as a string, and the column it names kept as a feature:
    read as characters it would hide the leak."""
    deriv(d, "late")["body"] = "delivered_days"
    cand(d)["features"].append("delivered_days")
    cand(d)["dropped_for_leakage"] = []


def _fd_candidate(d):
    """A second candidate, zip, keeping city: geometry.json holds the exact FD city → zip."""
    c = copy.deepcopy(cand(d))
    c.update(label="zip", task="multiclass classification", rule_id=None,
             features=["city", "unit_price", "qty"], dropped_for_leakage=[], input_dim=3,
             consequences=_cq("zip as the label"))
    d["partitions"]["candidates"].append(c)
    d["partitions"]["provenance"] = "user-chosen"


def _named_candidate(d, stance=None):
    """A second candidate, weight_kg: a basis member no rule derives and no functional dependency
    determines — a name, which step 4 does not admit unless a stance names it."""
    c = copy.deepcopy(cand(d))
    c.update(label="weight_kg", task="regression", rule_id=None,
             features=["unit_price", "qty", "discount_pct"], dropped_for_leakage=[], input_dim=3,
             consequences=_cq("weight_kg as the label"))
    if stance:
        c["stance"] = stance
    d["partitions"]["candidates"].append(c)
    d["partitions"]["provenance"] = "user-chosen"


UNATTENDED_STANCE = "unattended: the semantic channel reads weight_kg as the measured outcome"


def unattended_abstained(d):
    """No user, and step 4 admits no candidate the run states as an outcome: the semantic
    reading stands weight_kg as a hypothesis by an `unattended:` stance, and nobody chooses."""
    _named_candidate(d, UNATTENDED_STANCE)
    d["partitions"].update(chosen=None, provenance="abstained")


def _unattended_chosen(d):
    """The same hypothesis made the choice: an unattended run picked the label it guessed."""
    _named_candidate(d, UNATTENDED_STANCE)
    d["partitions"].update(chosen="weight_kg", provenance="user-chosen")


def retyped(d):
    """The analyst overrules the script both ways, stating no role: order_id (script: id) is
    retyped numeric and kept in the basis; customer_id (script: numeric) is retyped id."""
    t = typed(d, "order_id")
    t.pop("role", None)
    t.update(final_type="numeric", reason="a sequence number the business assigns in time order",
             consequences=_cq("order_id is read as a quantity"))
    d["basis"]["members"].append("order_id")
    t = typed(d, "customer_id")
    t.pop("role", None)
    t["final_type"] = "id"


NEGATIVES = [
    ("a derivation without its provenance triple", "no provenance triple",
     lambda d: deriv(d, "total").pop("provenance")),
    ("a provenance missing its symbolic channel", "provenance.symbolic",
     lambda d: deriv(d, "total")["provenance"].pop("symbolic")),
    ("a candidate without its leakage set", "no leakage set",
     lambda d: cand(d).pop("dropped_for_leakage")),
    ("a candidate without its consequence block", "missing its consequence block",
     lambda d: cand(d).pop("consequences")),
    ("symbolic confirmed without verified_rows", "without a verified_rows count",
     lambda d: deriv(d, "total")["provenance"].__setitem__("symbolic", {"status": "confirmed"})),
    ("symbolic confirmed with fewer verified rows than it ran", "verified_rows 190 != of_rows 200",
     lambda d: deriv(d, "total")["provenance"].__setitem__(
         "symbolic", {"status": "confirmed", "verified_rows": 190, "of_rows": 200})),
    ("symbolic counts beyond the A-box sample", "exceeds the 200 A-box rows",
     lambda d: deriv(d, "total")["provenance"].__setitem__(
         "symbolic", {"status": "confirmed", "verified_rows": 600, "of_rows": 600})),
    ("untested with a verified count", "carries verified_rows",
     lambda d: deriv(d, "total")["provenance"].__setitem__(
         "symbolic", {"status": "untested", "verified_rows": 5, "of_rows": 200})),
    ("semantic untested", "semantic channel always has a view",
     lambda d: deriv(d, "total")["provenance"]["semantic"].__setitem__("status", "untested")),
    ("a disagreement resolved silently", "never resolve it silently",
     lambda d: deriv(d, "total")["provenance"].__setitem__(
         "symbolic", {"status": "refuted", "verified_rows": 190, "of_rows": 200, "rows": [4, 9]})),
    ("a number that is not the script's", "numbers come verbatim",
     lambda d: d["space"].__setitem__("dims_95", 5)),
    ("a script_type the script did not write", "script_type 'nominal'",
     lambda d: typed(d, "zip").__setitem__("script_type", "nominal")),
    ("a basis member that is a rule head", "head of rule swrl-total",
     lambda d: d["basis"]["members"].append("total")),
    ("a column neither in the basis nor derived", "'note' is neither a basis member",
     lambda d: d["basis"]["members"].remove("note")),
    ("an identity in the basis", "'order_id' is typed identity",
     lambda d: d["basis"]["members"].append("order_id")),
    ("the label's rule body among its features", "derive the label",
     lambda d: (cand(d)["features"].append("delivered_days"), cand(d).__setitem__("dropped_for_leakage", []))),
    ("a leakage column kept among the features", "still among the features",
     lambda d: cand(d)["features"].append("delivered_days")),
    ("a rule body that is not a list, with the label's defining column kept",
     "derivations[4].body must be a non-empty list of column names", _body_as_text),
    ("the determinant of an exact functional dependency on the label kept as a feature",
     "['city'] determines the label exactly", _fd_candidate),
    ("a feature that is not a column name", "features must be a list of column names",
     lambda d: cand(d)["features"].append(["qty"])),
    ("a leakage set written as a string", "dropped_for_leakage must be a list of column names",
     lambda d: cand(d).__setitem__("dropped_for_leakage", "delivered_days")),
    ("a cycle's members written as a string", "members must be a list of column names",
     lambda d: cycle(d, "zip-city-region").__setitem__("members", "zip city region")),
    ("an orientation's derived set written as a string", "derived must be a list of column names",
     lambda d: cycle(d, "price-qty-total")["orientations"][1].__setitem__("derived", "unit_price")),
    ("a cycles slot that is an object", "cycles must be a list, not",
     lambda d: d.__setitem__("cycles", {"a": 1})),
    ("a typing entry that is not an object", "typing[0] must be an object",
     lambda d: d["typing"].__setitem__(0, "order_id")),
    ("a cycle without the report's orientation", "report's own orientation",
     lambda d: cycle(d, "price-qty-total")["orientations"].pop(0)),
    ("a derivation naming a cycle no entry carries", "View B cannot re-orient it",
     lambda d: d.__setitem__("cycles", [c for c in d["cycles"] if c["id"] != "zip-city-region"])),
    ("an orientation without its consequence block", "orientations[1]: missing its consequence block",
     lambda d: cycle(d, "price-qty-total")["orientations"][1].pop("consequences")),
    ("a retyping without its consequence block", "typing[2] (zip): missing its consequence block",
     lambda d: typed(d, "zip").pop("consequences")),
    ("a downstream block missing an entry", "lacks leakage",
     lambda d: deriv(d, "late")["consequences"]["downstream"].pop("leakage")),
    ("abstained with a chosen partition", "'abstained' but a partition is chosen",
     lambda d: d["partitions"].__setitem__("provenance", "abstained")),
    ("single-candidate with nothing chosen", "nothing is chosen",
     lambda d: d["partitions"].__setitem__("chosen", None)),
    ("a missing disagreements slot", "missing required top-level key 'disagreements'",
     lambda d: d.pop("disagreements")),
    ("a candidate step 4 does not admit, named by no stance", "a name is a hypothesis, not a candidate",
     _named_candidate),
    ("an 'unattended:' stance with no reading after it", "a name is a hypothesis, not a candidate",
     lambda d: (_named_candidate(d, "unattended: "),
                d["partitions"].update(chosen=None, provenance="abstained"))),
    ("an 'unattended:' hypothesis made the chosen partition", "stands only while partitions.provenance is 'abstained'",
     _unattended_chosen),
    ("an 'unattended:' hypothesis beside a partition someone chose", "stands only while partitions.provenance is 'abstained'",
     lambda d: _named_candidate(d, UNATTENDED_STANCE)),
]

POSITIVES = [
    ("a rounding of the script's number is accepted",
     lambda d: d["basis"].__setitem__("intrinsic_dim", 7.1)),
    ("a refuted channel with its disagreement entry is accepted",
     lambda d: (deriv(d, "total")["provenance"].__setitem__(
         "symbolic", {"status": "refuted", "verified_rows": 190, "of_rows": 200, "rows": [4, 9]}),
         d["disagreements"].append({"column": "total", "kind": "symbolic-refuted-empirical-confirmed",
                                    "evidence": "10 of 200 rows differ in the last cent",
                                    "reading": "rounding: the file stores totals to the cent"}))),
    ("symbolic confirmed with its counts is accepted",
     lambda d: deriv(d, "late")["provenance"].__setitem__(
         "symbolic", {"status": "confirmed", "verified_rows": 200, "of_rows": 200})),
    ("the one-sentence 'no downstream consequence' is accepted",
     lambda d: deriv(d, "region")["consequences"].__setitem__("downstream", "no downstream consequence")),
    ("the analyst's final_type decides a role left unstated (id retyped numeric stays in the "
     "basis, numeric retyped id stays out)", retyped),
    ("a candidate step 4 does not admit stands on a user's stance that names it",
     lambda d: _named_candidate(d, "user: the analyst asks for weight_kg as the target")),
    ("an unattended run states a candidate step 4 does not admit as an abstained hypothesis",
     unattended_abstained),
]


def messy_frame():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 300
    a = rng.normal(10, 3, n).round(4)
    return pd.DataFrame({
        "rid": np.arange(n) + 5000, "a": a, "b": 2 * a + 1,
        "c": rng.choice([f"k{i}" for i in range(12)], n),
        "d": [" ".join(rng.choice(["alpha", "beta", "gamma", "delta", "eps"], 8)) + f" #{i}"
              for i in range(n)],
        "e": np.where(rng.random(n) < 0.2, np.nan, rng.gamma(2, 2, n)).round(3),
        "f": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"), "g": 7})


def messy_layer():
    return {
        "schema": "dataset-forge/geometry@1", "from_run": "geometry.json",
        "source": {"path": "messy.csv"}, "markers": ["OK: a generated file with every column kind"],
        "typing": [{"column": "rid", "final_type": "id", "role": "identity", "reason": "the key"},
                   {"column": "g", "final_type": "constant", "role": "degenerate", "reason": "one value"},
                   {"column": "d", "final_type": "text", "role": "text", "reason": "free text"}],
        "space": {"reading": "a and b are one line; the rest is noise."},
        "basis": {"members": ["a", "c", "e", "f"], "reading": "four primitives"},
        "derivations": [{"column": "b", "rule_id": "swrl-b", "layer": "swrl", "formula": "b = 2a + 1",
                         "body": ["a"], "cycle": ["a", "b"],
                         "provenance": {"semantic": {"status": "confirmed"},
                                        "symbolic": {"status": "confirmed", "verified_rows": 200,
                                                     "of_rows": 200},
                                        "empirical": {"status": "confirmed", "evidence": "R² 1"}},
                         "consequences": _cq("b is derived")}],
        "cycles": [{"id": "C1", "members": ["a", "b"], "orientations": [
            {"id": "C1a", "derived": ["b"], "consequences": _cq("a in the basis")},
            {"id": "C1b", "derived": ["a"], "consequences": _cq("b in the basis")}]}],
        "disagreements": [],
        "orthogonality": {"pairs": [{"a": "a", "b": "e", "independence": "both"}], "reading": "flat"},
        "partitions": {"chosen": None, "provenance": "abstained", "candidates": [
            {"label": "b", "task": "regression", "rule_id": "swrl-b", "features": ["c", "e", "f"],
             "dropped_for_leakage": ["a"], "input_dim": 3, "consequences": _cq("b as the label")},
            {"label": "c", "task": "multiclass classification", "rule_id": None,
             "stance": "user: the analyst asks for c, which no rule derives",
             "features": ["a", "e", "f"], "dropped_for_leakage": [], "input_dim": 3,
             "consequences": _cq("c as the label")}]},
        "handoff": {"next": "/data-lens messy.domain.html", "note": "the fork was abstained"}}
