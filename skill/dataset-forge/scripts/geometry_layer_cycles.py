"""
geometry_layer_cycles.py — the `cycles` slot of a `layer-geometry-data` document: the
orientations of a symmetric derivation, checked (`check_cycles`) and completed with the keys
/dataset-shaper reads (`complete_cycles`).

    {"id": "price-qty-total", "members": ["unit_price", "qty", "total"], "reading": "...",
     "orientations": [
       {"id": "total-derived", "derived": ["total"], "consequences": {...}},
       {"id": "unit-price-derived", "derived": ["unit_price"], "consequences": {...},
        "derivations": [{"column": "unit_price", "formula": "unit_price = total / qty",
                         "body": ["total", "qty"]}]}]}

The report's orientation is the one whose `derived` set equals the kept rule heads among the
members. The author writes `id`, `derived`, the consequence block and optional rules; the
completion adds `default` (true on the report's orientation alone), `basis` (the members that
stay) and `rules` (one per derived member), and the check refuses an authored `default`,
`basis` or `rules` that contradicts `derived` — the explorer reads `derived`, the shaper reads
the completed keys, and the two may never describe different orientations.

Imported by `geometry_layer_checks.py` and `apply_geometry_layer.py`; not a command.
"""
from __future__ import annotations

from geometry_layer_model import CYCLE_ID_RE, kept

SLOT_NOTE = ("the `cycles` slot is described in `apply_geometry_layer.py --help` until "
             "report-contract §1 lists it")


def check_shaper_keys(C, o, wo, stay, is_report):
    """An authored `default`, `basis` or `rules` must say what `derived` says."""
    if o.get("default") is not None:
        if not isinstance(o["default"], bool):
            C.err(f"{wo}: 'default' must be true or false")
        elif o["default"] != is_report:
            C.err(f"{wo}: 'default' is {str(o['default']).lower()} but this is "
                  f"{'' if is_report else 'not '}the report's orientation — the shaper applies the "
                  "default orientation, and it must be the one the report's rules describe")
    if o.get("basis") is not None and frozenset(o["basis"]) != stay:
        C.err(f"{wo}: 'basis' {o['basis']!r} contradicts 'derived' — the members that stay "
              f"are {sorted(stay)}")
    if o.get("rules") is not None:
        heads = {r.get("column") for r in o["rules"]}
        if heads != set(o["derived"]):
            C.err(f"{wo}: 'rules' derive {sorted(h for h in heads if h)} but 'derived' is "
                  f"{sorted(o['derived'])}")


class _Orientations:
    """The orientations of one cycle, checked against each other and the report's rules."""

    def __init__(self, C, w, members, report):
        self.C, self.w, self.members, self.report = C, w, frozenset(members), report
        self.seen, self.ids, self.matches = set(), set(), 0

    def check(self, j, o):
        wo = f"{self.w}.orientations[{j}]"
        if not o.get("derived"):
            self.C.err(f"{wo}: 'derived' must list the members this orientation derives")
            return
        self._check_id(wo, o.get("id"))
        der = frozenset(o["derived"])
        if not der < self.members:
            self.C.err(f"{wo}: derived {sorted(der)} must be a proper subset of the members")
        if der in self.seen:
            self.C.err(f"{wo}: the orientation deriving {sorted(der)} is listed twice")
        self.seen.add(der)
        self.matches += der == self.report
        self.C.consequences(o.get("consequences"), wo)
        for key in ("derivations", "rules"):
            for k, od in enumerate(o.get(key) or []):
                if od.get("column") not in der or not od.get("body"):
                    self.C.err(f"{wo}.{key}[{k}] must name a derived member and its body")
        check_shaper_keys(self.C, o, wo, self.members - der, der == self.report)

    def _check_id(self, wo, oid):
        if not oid:
            self.C.err(f"{wo}: an orientation needs an id — the shaper's orient_cycle step names "
                       "the orientation it applies by it")
        elif oid in self.ids:
            self.C.err(f"{wo}: the orientation id {oid!r} is used twice in the cycle")
        self.ids.add(oid)


def _check_cycle(C, i, cy, heads, ids):
    """The members of a well-formed cycle (None when they are not)."""
    cid, members = cy.get("id"), cy.get("members")
    w = f"cycles[{i}] ({cid})"
    if not cid or cid in ids:
        C.err(f"{w}: a cycle needs a unique id (/dataset-shaper cites geometry:cycles/<id>)")
    elif not CYCLE_ID_RE.match(cid):
        C.err(f"{w}: the id may hold only letters, digits, '_', '.' and '-' — the shaper cites "
              "it as the path segment geometry:cycles/<id>")
    ids.add(cid)
    if not isinstance(members, list) or len(set(members)) < 2:
        C.err(f"{w}: members must list at least two columns")
        return None
    for c in members:
        C.known(c, f"{w}.members")
    orients = cy.get("orientations")
    if not isinstance(orients, list) or len(orients) < 2:
        C.err(f"{w}: a cycle carries at least two orientations (the report's and an "
              "alternative), each with its consequence block — else there is nothing to "
              "re-orient")
        return frozenset(members)
    report = frozenset(c for c in members if c in heads)
    seen = _Orientations(C, w, members, report)
    for j, o in enumerate(orients):
        seen.check(j, o)
    if seen.matches != 1:
        C.err(f"{w}: the report's own orientation (derived: {sorted(report) or 'none'}) is "
              "not among the cycle's orientations")
    return frozenset(members)


def check_cycles(C, heads):
    """Every cycle is well formed, and every derivation that names a cycle has its entry."""
    known, ids = set(), set()
    for i, cy in enumerate(C.doc.get("cycles") or []):
        members = _check_cycle(C, i, cy, heads, ids)
        if members:
            known.add(members)
    for i, d in enumerate(C.doc.get("derivations") or []):
        cyc = d.get("cycle")
        if cyc and frozenset(cyc) not in known:
            C.err(f"derivations[{i}] ({d.get('column')}) names the cycle {cyc} but no cycles "
                  "entry has those members — View B cannot re-orient it (add it with an "
                  f"orientation and a consequence block per basis choice; {SLOT_NOTE})")


# --------------------------------------------------------------------------- completion
def _rule(m, src, stay):
    """The rule deriving member `m`: from `src` (an authored rule), else from the members that
    stay."""
    if not src:
        return {"column": m, "body": list(stay), "formula": f"{m} from {', '.join(stay)}"}
    body = list(src.get("body") or [])
    rule = {"column": m, "body": body,
            "formula": src.get("formula") or f"{m} from {', '.join(body)}"}
    if src.get("rule_id"):
        rule["rule_id"] = src["rule_id"]
    return rule


def _complete_orientation(o, members, heads, report):
    der = list(o.get("derived") or [])
    stay = [m for m in members if m not in der]
    o["default"] = set(der) == report
    o["basis"] = stay
    if o.get("rules"):
        return
    own = {}
    for r in o.get("derivations") or []:
        own.setdefault(r.get("column"), r)
    o["rules"] = [_rule(m, own.get(m) or (heads.get(m) if o["default"] else None), stay)
                  for m in der]


def complete_cycles(doc):
    """Complete every orientation with `default`, `basis` and `rules` (the keys the shaper's
    `orient_cycle` / `drop_derived` read). Runs after validation, so `derived` is sound."""
    heads = {d.get("column"): d for d in doc.get("derivations") or [] if kept(d)}
    for cy in doc.get("cycles") or []:
        members = list(cy.get("members") or [])
        report = {m for m in members if m in heads}
        for o in cy.get("orientations") or []:
            _complete_orientation(o, members, heads, report)
    return doc
