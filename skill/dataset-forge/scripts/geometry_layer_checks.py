"""
geometry_layer_checks.py — the validation of a spliced `layer-geometry-data` document against
`references/report-contract.md` §1-§3 and §5: the structural pass first, then the top level,
the typing, the derivations with their provenance triple, the basis and the roles, the cycles,
the partitions and the views. `validate(doc, out)` returns the ERROR lines; WARN lines go to
`out`. The rules, and why each exists, are listed in `apply_geometry_layer.py --help`.

Imported by `apply_geometry_layer.py`; not a command.
"""
from __future__ import annotations

from geometry_layer_cycles import check_cycles
from geometry_layer_model import (CHANNELS, OUTSIDE_BASIS, ROLES, RULE_LAYERS, SCHEMA, STATUSES,
                                  TYPES, MARKER_RE, Checker, effective_roles, is_num, kept_heads,
                                  status, typing_of, working_type)
from geometry_layer_partitions import check_partitions, check_views
from geometry_layer_shape import check_shape


def _is_count(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _said(v):
    return bool(str(v or "").strip())


def check_top(C):
    doc = C.doc
    for k in ("schema", "source", "typing", "space", "basis", "derivations", "disagreements",
              "partitions"):
        if k not in doc:
            C.err(f"missing required top-level key {k!r}")
    if C.errs:
        return False
    if doc.get("schema") != SCHEMA:
        C.err(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    src = doc["source"]
    if not isinstance(src, dict) or not src.get("path"):
        C.err("source.path is required")
    for k in ("orthogonality", "handoff"):
        if k not in doc:
            C.warn(f"no {k!r} section")
    for m in doc.get("markers") or []:
        if not isinstance(m, str) or not MARKER_RE.match(m):
            C.warn(f"marker {m!r} does not start with OK:/WARN:/ERROR:/FORK:")
    return True


# --------------------------------------------------------------------------- typing
def check_typing(C):
    seen = set()
    for i, t in enumerate(C.doc.get("typing") or []):
        if not t.get("column"):
            C.err(f"typing[{i}] must be an object with 'column'")
            continue
        _check_typing_entry(C, f"typing[{i}] ({t['column']})", t, seen)


def _check_typing_entry(C, w, t, seen):
    c = t["column"]
    if c in seen:
        C.err(f"{w}: the column is typed twice")
    seen.add(c)
    C.known(c, w)
    ft, st, role = t.get("final_type"), t.get("script_type"), t.get("role")
    if not ft:
        C.err(f"{w}: final_type is required")
    elif ft not in TYPES:
        C.warn(f"{w}: final_type {ft!r} is not one of {', '.join(TYPES)}")
    if role is not None and role not in ROLES:
        C.err(f"{w}: role {role!r} is not one of {', '.join(ROLES)}")
    retyped = bool(st and ft and st != ft)
    if retyped and not _said(t.get("reason")):
        C.err(f"{w}: retyped {st} → {ft} without a reason")
    elif not retyped and role in OUTSIDE_BASIS and not _said(t.get("reason")):
        C.warn(f"{w}: role {role} carries no reason")
    if retyped or t.get("consequences") is not None:
        C.consequences(t.get("consequences"), w)


# --------------------------------------------------------------------------- derivations
def check_provenance(C, d, w):
    """Contract §2: three independent verdicts, each confirmed / refuted / untested."""
    P = d.get("provenance")
    if not isinstance(P, dict):
        C.err(f"{w}: no provenance triple — every derivation carries semantic, symbolic and "
              "empirical verdicts (contract §2)")
        return
    for ch in CHANNELS:
        v = P.get(ch)
        if not isinstance(v, dict) or v.get("status") not in STATUSES:
            C.err(f"{w}: provenance.{ch} must be an object whose status is one of "
                  f"{', '.join(STATUSES)} (contract §2: the provenance triple is complete or "
                  "the derivation is refused)")
    if any(not isinstance(P.get(ch), dict) for ch in CHANNELS):
        return
    if P["semantic"].get("status") == "untested":
        C.err(f"{w}: provenance.semantic is 'untested' — the semantic channel always has a "
              "view: confirmed (derivable by meaning) or refuted")
    _check_symbolic(C, P["symbolic"], w)
    emp = P["empirical"]
    if emp.get("status") in ("confirmed", "refuted") and not _said(emp.get("evidence")):
        C.warn(f"{w}: empirical {emp.get('status')} without its evidence")


def _check_symbolic(C, sym, w):
    """The engine's counts: confirmed holds on every sampled row, refuted on fewer, untested
    ran nowhere; and no count exceeds the A-box sample the script drew."""
    vr, of, s = sym.get("verified_rows"), sym.get("of_rows"), sym.get("status")
    if s == "confirmed":
        _check_confirmed(C, vr, of, w)
    elif s == "refuted":
        if not _is_count(of) or not _is_count(vr):
            C.err(f"{w}: symbolic 'refuted' needs verified_rows and of_rows from the engine run "
                  "that refuted it")
        elif vr >= of:
            C.err(f"{w}: symbolic 'refuted' but verified_rows {vr} >= of_rows {of}")
        if not sym.get("rows"):
            C.warn(f"{w}: symbolic 'refuted' without the disagreeing rows listed")
    elif s == "untested" and _is_count(vr) and vr > 0:
        C.err(f"{w}: symbolic 'untested' carries verified_rows {vr} — a count means an engine "
              "ran; record what it said")
    abox = (C.doc.get("source") or {}).get("abox_rows")
    if _is_count(of) and is_num(abox) and of > abox:
        C.err(f"{w}: symbolic of_rows {of} exceeds the {abox} A-box rows the script sampled; "
              "the engine verifies rules on the A-box sample only")


def _check_confirmed(C, vr, of, w):
    if not _is_count(vr):
        C.err(f"{w}: symbolic 'confirmed' without a verified_rows count — the count comes "
              "from the run_query.py output, and without one the rule is 'untested'")
    elif not _is_count(of) or of <= 0:
        C.err(f"{w}: symbolic 'confirmed' needs of_rows (the sampled rows the engine ran on)")
    elif vr != of:
        C.err(f"{w}: symbolic 'confirmed' means the rule held on every sampled row, but "
              f"verified_rows {vr} != of_rows {of} (that is 'refuted', with the rows listed)")


def _check_rule_fields(C, d, w, rule_ids):
    for k in ("column", "rule_id", "layer", "formula", "body"):
        if not d.get(k):
            C.err(f"{w}: {k!r} is required")
    if d.get("rule_id") in rule_ids:
        C.err(f"{w}: rule_id {d.get('rule_id')!r} is used twice")
    rule_ids.add(d.get("rule_id"))
    if d.get("layer") and d["layer"] not in RULE_LAYERS:
        C.err(f"{w}: layer {d['layer']!r} is not one of {', '.join(RULE_LAYERS)} "
              "(SWRL for arithmetic and thresholds, Horn `rules` for lookups)")
    body = d.get("body") or []
    if d.get("column"):
        C.known(d["column"], w)
    for b in body:
        C.known(b, f"{w}.body")
    if d.get("column") in body:
        C.err(f"{w}: the column is in its own body")


def check_derivations(C):
    derivs = C.doc.get("derivations")
    if not isinstance(derivs, list):
        C.err("derivations must be a list")
        return
    rule_ids = set()
    disagree = {x.get("column") for x in C.doc.get("disagreements") or []}
    for i, d in enumerate(derivs):
        w = f"derivations[{i}] ({d.get('column')})"
        _check_rule_fields(C, d, w, rule_ids)
        check_provenance(C, d, w)
        C.consequences(d.get("consequences"), w)
        verdicts = {status(d, ch) for ch in CHANNELS}
        if {"confirmed", "refuted"} <= verdicts and d.get("column") not in disagree:
            said = ", ".join(f"{ch} {status(d, ch)}" for ch in CHANNELS)
            C.err(f"{w}: its channels disagree ({said}) and no disagreements entry names the "
                  "column — never resolve it silently")


# --------------------------------------------------------------------------- basis and roles
def check_basis_and_roles(C, heads):
    B = C.doc.get("basis")
    members = B.get("members") if isinstance(B, dict) else None
    if not isinstance(members, list) or not members:
        C.err("basis.members must be a non-empty list")
        return
    if len(set(members)) != len(members):
        C.err("basis.members lists a column twice")
    if B.get("size") != len(members):
        C.err(f"basis.size {B.get('size')!r} != {len(members)} members")
    for c in members:
        C.known(c, "basis.members")
    if not _said(B.get("reading")):
        C.warn("basis.reading is missing")
    for c, rules in heads.items():
        if len(rules) > 1:
            C.err(f"{c!r} is the head of {len(rules)} kept rules ({', '.join(map(str, rules))}); "
                  "a derived column is the head of exactly one")
    for c, role in effective_roles(C.doc, heads, set(members)).items():
        _check_role(C, c, role, c in members, heads)
    tw = (C.doc.get("space") or {}).get("intrinsic_dim_twonn")
    numeric = numeric_members(C.doc, members)
    if is_num(tw) and tw > 0 and 2 * len(numeric) >= len(members) and len(numeric) > 2 * tw + 2:
        C.warn(f"{len(numeric)} numeric basis member(s) against an intrinsic dimension of {tw}: "
               "derivations may still be hiding — the reading should say why")


# the working types TwoNN measures (geometry.numeric_frame): the numeric span; nominal, date and
# text members are counted apart and never compared with the intrinsic dimension (SKILL step 2)
SPAN_TYPES = ("numeric", "ordinal", "boolean")


def numeric_members(doc, members):
    """The basis members whose working type TwoNN measures."""
    typed, scripted = typing_of(doc), doc.get("columns") or {}
    return [c for c in members if working_type(c, typed, scripted) in SPAN_TYPES]


def _check_role(C, c, role, in_basis, heads):
    """Every non-identity column is a basis member or the head of a kept rule, never both."""
    is_head = c in heads
    if role in OUTSIDE_BASIS:
        if in_basis:
            C.err(f"{c!r} is typed {role} but sits in the basis — an identifier or a "
                  "constant is not a dimension")
        return
    if role == "text" and not in_basis:
        return
    if in_basis and is_head:
        C.err(f"{c!r} is a basis member and the head of rule {heads[c][0]} — the basis is "
              "the set of columns that are the head of no rule")
    elif not in_basis and not is_head:
        C.err(f"{c!r} is neither a basis member nor the head of a kept rule (every "
              "non-identity column is one or the other)")
    if role == "derived" and not is_head:
        C.err(f"{c!r} is typed role 'derived' but no kept rule derives it")
    if role == "dimension" and is_head:
        C.err(f"{c!r} is typed role 'dimension' but rule {heads[c][0]} derives it")


# --------------------------------------------------------------------------- the whole document
def validate(doc, out):
    """The ERROR lines of a spliced document (empty when it validates)."""
    errs = check_shape(doc)
    if errs:
        return errs
    C = Checker(doc, out)
    if not check_top(C):
        return C.errs
    check_typing(C)
    check_derivations(C)
    heads = kept_heads(doc)
    check_basis_and_roles(C, heads)
    check_cycles(C, heads)
    check_partitions(C, heads)
    check_views(C)
    return C.errs


def summary(doc):
    """One line for the OK marker: the provenance counts, the basis, the cycles, the partition."""
    derivs = doc.get("derivations") or []
    counts = {ch: {s: sum(1 for d in derivs if status(d, ch) == s) for s in STATUSES}
              for ch in CHANNELS}
    P, B, cycles = doc.get("partitions") or {}, doc.get("basis") or {}, doc.get("cycles") or []
    prov = "; ".join(f"{ch} " + ", ".join(f"{n} {s}" for s, n in counts[ch].items() if n)
                     for ch in CHANNELS)
    members = [c for c in B.get("members") or [] if isinstance(c, str)]
    numeric = numeric_members(doc, members)
    against = (f"{len(numeric)} numeric against intrinsic dimension {B.get('intrinsic_dim')}"
               if 2 * len(numeric) >= len(members) else
               f"{len(numeric)} numeric, mostly non-numeric: the intrinsic dimension does not apply")
    return (f"{len(derivs)} derivation(s) ({prov}); basis {B.get('size')} ({against}); {len(cycles)} cycle(s) with "
            f"{sum(len(c.get('orientations') or []) for c in cycles)} orientation(s); "
            f"{len(P.get('candidates') or [])} candidate(s), chosen {P.get('chosen')!r} "
            f"({P.get('provenance')}); {len(doc.get('disagreements') or [])} disagreement(s)")
