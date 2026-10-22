"""
geometry_layer_partitions.py — the partition candidates and the views of a
`layer-geometry-data` document (`references/report-contract.md` §1, §5, §6).

A candidate states its task, its features, its leakage set and its consequence block, and never
keeps among its features the label, a column of its own leakage set, a column of the label's
kept rule, or the determinant of an exact functional dependency on the label (SKILL.md step 4:
a column that derives the label is a leakage, and the model would learn the rule instead of the
phenomenon). `partitions.chosen` agrees with `partitions.provenance`.

The views: every disagreement carries its reading, orthogonality pairs name two columns, the
explore payload is rectangular, and what the script proposed as derivable is either derived or
recorded as a disagreement.

Imported by `geometry_layer_checks.py`; not a command.
"""
from __future__ import annotations

import json

from geometry_layer_model import PROVENANCES, exact_fds, kept


def _rule_body(doc, label):
    """The columns of every kept rule deriving the label (empty when none does)."""
    body = []
    for d in doc.get("derivations") or []:
        if kept(d) and d.get("column") == label:
            body += [b for b in d.get("body") or [] if b not in body]
    return body


def _check_leakage(C, lab, feats, leak, w):
    """The label's own columns stay out of its features: itself, its leakage set, its rule's
    body, and every exact functional dependency's determinant."""
    if lab in feats:
        C.err(f"{w}: the label is among its own features")
    kept_leak = [f for f in leak if f in feats]
    if kept_leak:
        C.err(f"{w}: leakage column(s) {kept_leak} are still among the features")
    own = [b for b in _rule_body(C.doc, lab) if b in feats]
    if own:
        C.err(f"{w}: {own} derive the label (rule body) yet stay among the features — the "
              "model would learn the rule instead of the phenomenon")
    reported = set(kept_leak) | set(own)
    for lhs, rhs in exact_fds(C.doc):
        if rhs == lab and lhs and set(lhs) <= set(feats) and not set(lhs) & reported:
            one = len(lhs) == 1
            C.err(f"{w}: {lhs} determine{'s' if one else ''} the label exactly (the functional "
                  f"dependency {' + '.join(lhs)} → {lab} in geometry.json) yet stay{'s' if one else ''} "
                  "among the features — the model would learn the lookup instead of the "
                  f"phenomenon; drop {'it' if one else 'one of them'} for leakage")
            reported |= set(lhs)


# who may stand a candidate step 4 does not admit: the user (a turn, `user: …`) or a lane registry
# that declares the target (`lane: …`, the verification harness's)
STANCES = ("user:", "lane:")
# an unattended run where step 4 admits no candidate (the label the data records is an observed
# outcome, not a derived column) states the semantic channel's reading as a hypothesis nobody
# chose (`unattended: …`, SKILL step 4): accepted only while the partition abstains, never chosen
UNATTENDED = "unattended:"


def _admitted(doc, lab, heads):
    """Step 4's candidates: the head of a kept rule whose body lies entirely in the basis, or a
    column an exact functional dependency determines."""
    if lab in {rhs for _, rhs in exact_fds(doc)}:
        return True
    if lab not in heads:
        return False
    basis = set((doc.get("basis") or {}).get("members") or [])
    return set(_rule_body(doc, lab)) <= basis


def _check_candidate(C, c, w, heads):
    lab = c["label"]
    C.known(lab, w)
    if not str(c.get("task") or "").strip():
        C.err(f"{w}: 'task' is required")
    if not isinstance(c.get("dropped_for_leakage"), list):
        C.err(f"{w}: no leakage set — 'dropped_for_leakage' lists the columns that derive "
              "the label (an empty list when none does)")
    C.consequences(c.get("consequences"), w)
    feats = c.get("features")
    if not isinstance(feats, list):
        C.err(f"{w}: 'features' must list the columns that remain")
        return
    leak = c.get("dropped_for_leakage") or []
    for f in feats + leak:
        C.known(f, w)
    _check_leakage(C, lab, feats, leak, w)
    rules = {d.get("rule_id") for d in C.doc.get("derivations") or []}
    if c.get("rule_id") and c["rule_id"] not in rules:
        C.err(f"{w}: rule_id {c['rule_id']!r} names no derivation")
    if not _admitted(C.doc, lab, heads):
        stance = str(c.get("stance") or "")
        if stance.startswith(STANCES):
            C.warn(f"{w}: the label is neither the head of a kept rule whose body lies in the basis "
                   "nor determined by a functional dependency; it stands as a candidate by the "
                   f"stance {stance!r} only")
        elif stance.startswith(UNATTENDED) and stance[len(UNATTENDED):].strip():
            _check_unattended(C, lab, stance, w)
        else:
            C.err(f"{w}: the label is neither the head of a kept rule whose body lies in the basis "
                  "nor determined by a functional dependency — a name is a hypothesis, not a "
                  "candidate (SKILL step 4); a user's stance that names it states it in 'stance'")
    if not isinstance(c.get("input_dim"), int):
        C.warn(f"{w}: no input_dim — /vega-forge reconciles it against the shaper's manifest")
    elif c["input_dim"] != len(feats):
        C.warn(f"{w}: input_dim {c['input_dim']} differs from its {len(feats)} features")


def _check_unattended(C, lab, stance, w):
    """An `unattended:` stance stands a hypothesis only while nobody chooses: the partition
    abstains and this candidate is not `chosen` (SKILL step 4, report-contract §1)."""
    P = C.doc.get("partitions") or {}
    if P.get("provenance") != "abstained" or P.get("chosen") == lab:
        C.err(f"{w}: an 'unattended:' stance states a hypothesis nobody chose — it stands only while "
              "partitions.provenance is 'abstained' and the label is not chosen; a chosen label "
              "the geometry does not admit needs the user's stance ('user: …', e.g. --partition)")
        return
    C.warn(f"{w}: the label is neither the head of a kept rule whose body lies in the basis nor "
           "determined by a functional dependency; an unattended run states it as a hypothesis "
           f"by the stance {stance!r}, and nobody chose it")


def _check_choice(C, P, labels):
    """`chosen` agrees with `provenance` (contract §5)."""
    prov, chosen, n = P.get("provenance"), P.get("chosen"), len(P.get("candidates") or [])
    if prov not in PROVENANCES:
        C.err(f"partitions.provenance must be one of {', '.join(PROVENANCES)}, got {prov!r}")
    if chosen is not None and chosen not in labels:
        C.err(f"partitions.chosen {chosen!r} is not one of the candidates")
    if prov == "abstained" and chosen is not None:
        C.err("partitions.provenance is 'abstained' but a partition is chosen")
    if prov == "single-candidate" and n != 1:
        C.err(f"partitions.provenance is 'single-candidate' with {n} candidates")
    if prov in ("user-chosen", "single-candidate") and chosen is None:
        C.err(f"partitions.provenance is {prov!r} but nothing is chosen")


def check_partitions(C, heads):
    P = C.doc.get("partitions")
    if not isinstance(P, dict):
        C.err("partitions must be an object")
        return
    cands = P.get("candidates")
    if not isinstance(cands, list) or not cands:
        C.err("partitions.candidates must state at least one label candidate")
        return
    labels = []
    for i, c in enumerate(cands):
        if not c.get("label"):
            C.err(f"partitions.candidates[{i}] must be an object with a 'label'")
            continue
        w = f"partitions.candidates[{i}] ({c['label']})"
        if c["label"] in labels:
            C.err(f"{w}: the label is a candidate twice")
        labels.append(c["label"])
        _check_candidate(C, c, w, heads)
    _check_choice(C, P, labels)


# --------------------------------------------------------------------------- the views
def _check_disagreements(C):
    if not isinstance(C.doc.get("disagreements"), list):
        C.err("disagreements must be a list (empty when the three channels agree everywhere)")
        return
    for i, x in enumerate(C.doc["disagreements"]):
        w = f"disagreements[{i}]"
        if not x.get("column") or not x.get("kind"):
            C.err(f"{w} must name a 'column' and a 'kind'")
            continue
        C.known(x["column"], w)
        if not str(x.get("reading") or "").strip():
            C.err(f"{w} ({x['column']}): a disagreement carries the reading that explains it")


def _check_orthogonality(C):
    O = C.doc.get("orthogonality")
    if not isinstance(O, dict):
        return
    basis = set((C.doc.get("basis") or {}).get("members") or [])
    for i, p in enumerate(O.get("pairs") or []):
        w = f"orthogonality.pairs[{i}]"
        if not p.get("a") or not p.get("b"):
            C.err(f"{w} must name 'a' and 'b'")
            continue
        C.known(p["a"], w)
        C.known(p["b"], w)
        if not {p["a"], p["b"]} <= basis:
            C.warn(f"{w} ({p['a']}, {p['b']}) is not a pair of basis members")
        if not str(p.get("independence") or "").strip():
            C.warn(f"{w} carries no 'independence' reading")


def _check_handoff(C):
    H = C.doc.get("handoff")
    if not isinstance(H, dict):
        return
    if not H.get("next"):
        C.warn("handoff.next is missing (the summary proposes /data-lens or /dataset-shaper)")
    if any(str(k).startswith("nn_") for k in H) or "nn-" in json.dumps(H):
        C.warn("handoff names the retired nn-* family; input_dim is the key /vega-forge reads")


def _check_explore(C):
    E = C.doc.get("explore")
    if E is None:
        C.warn("no explore payload: Views A and C render their empty state")
        return
    if isinstance(E.get("columns"), dict):
        n = E.get("n")
        bad = [c for c, v in E["columns"].items() if len(v) != n]
        if bad:
            C.err(f"explore.columns {bad[:5]} do not hold explore.n = {n} values each")
    pca = E.get("pca") or {}
    if pca and len(pca.get("scores") or []) != len(pca.get("index") or []):
        C.err("explore.pca.scores and explore.pca.index differ in length")


def _check_unaddressed(C):
    """What the script proposed as derivable is derived, in a cycle, or a disagreement."""
    doc = C.doc
    addressed = {d.get("column") for d in doc.get("derivations") or []}
    for cy in doc.get("cycles") or []:
        addressed |= set(cy.get("members") or [])
    addressed |= {x.get("column") for x in doc.get("disagreements") or []}
    unaddressed = [p.get("column") for p in doc.get("linear_derivability") or []
                   if p.get("column") not in addressed]
    unaddressed += [rhs for _, rhs in exact_fds(doc) if rhs not in addressed]
    if unaddressed:
        C.warn(f"the script proposed {sorted(set(map(str, unaddressed)))} as derivable and the "
               "layer neither derives them nor records the disagreement")


def check_views(C):
    _check_disagreements(C)
    _check_orthogonality(C)
    _check_handoff(C)
    _check_explore(C)
    sp = C.doc.get("space")
    if isinstance(sp, dict) and not str(sp.get("reading") or "").strip():
        C.warn("space.reading is missing")
    _check_unaddressed(C)
