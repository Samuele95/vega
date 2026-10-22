"""
geometry_layer_model.py — the vocabulary of a `layer-geometry-data` document and the readings
every check shares (`references/report-contract.md` §1-§3): which rules are kept, which columns
the document names, what role each column plays, and the `Checker` that collects the verdicts.

Imported by `apply_geometry_layer.py` and its sibling modules; not a command.
"""
from __future__ import annotations

import re

LAYER = "geometry"
SCHEMA = "dataset-forge/geometry@1"
STATUSES = ("confirmed", "refuted", "untested")
CHANNELS = ("semantic", "symbolic", "empirical")
ROLES = ("dimension", "derived", "identity", "degenerate", "foreign-key", "text")
OUTSIDE_BASIS = ("identity", "degenerate", "foreign-key")
TYPES = ("numeric", "ordinal", "nominal", "boolean", "datetime", "text", "id", "constant")
RULE_LAYERS = ("swrl", "rules", "horn")
PROVENANCES = ("user-chosen", "single-candidate", "abstained")
DOWNSTREAM = ("preprocessing", "model_family", "leakage", "dimensionality", "interpretability")
MARKER_RE = re.compile(r"^(OK|WARN|ERROR|FORK):\s")
CYCLE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")    # a path segment of geometry:cycles/<id>
# The role a type implies when the analyst states none (SKILL.md step 0: identifiers and
# constants leave the analysis as identity and degenerate).
ROLE_OF_TYPE = {"id": "identity", "constant": "degenerate"}


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def status(d, channel):
    return ((d.get("provenance") or {}).get(channel) or {}).get("status")


def kept(d):
    """A derivation the report keeps: the semantic channel confirms it and it is not set aside."""
    return isinstance(d, dict) and status(d, "semantic") == "confirmed" and \
        d.get("kept", True) is not False


def kept_heads(doc):
    """{column: [rule ids]} of the kept derivations."""
    heads = {}
    for d in doc.get("derivations") or []:
        if kept(d):
            heads.setdefault(d.get("column"), []).append(d.get("rule_id"))
    return heads


def exact_fds(doc):
    """The exact functional dependencies geometry.py found, as (lhs list, rhs)."""
    return [(list(f.get("lhs") or []), f.get("rhs")) for f in doc.get("functional_dependencies") or []
            if isinstance(f, dict) and f.get("exact")]


def _named_columns(doc):
    """Every column the document names, in first-mention order (no authoritative list)."""
    seen = []

    def add(c):
        if isinstance(c, str) and c not in seen:
            seen.append(c)
    for t in doc.get("typing") or []:
        add(t.get("column"))
    for c in (doc.get("basis") or {}).get("members") or []:
        add(c)
    for d in doc.get("derivations") or []:
        add(d.get("column"))
        for b in d.get("body") or []:
            add(b)
    for c in (doc.get("explore") or {}).get("columns") or {}:
        add(c)
    for c in doc.get("stats") or {}:
        add(c)
    return seen


def column_universe(doc):
    """(columns in file order, whether the list is authoritative). Call it on a document the
    structural pass accepted."""
    src = doc.get("source") or {}
    for key in ("column_names", "columns"):
        if isinstance(src.get(key), list):
            return [str(c) for c in src[key]], True
    return _named_columns(doc), False


def typing_of(doc):
    """{column: typing entry}."""
    return {t.get("column"): t for t in doc.get("typing") or [] if isinstance(t, dict)}


def working_type(c, typed, scripted):
    """The type a column is analysed as: the analyst's final_type, else the script's type. The
    script's types are heuristics and the analyst owns the final typing (SKILL.md step 0)."""
    t = typed.get(c) or {}
    return t.get("final_type") or (scripted.get(c) or {}).get("type") or t.get("script_type")


def effective_roles(doc, heads, basis):
    """{column: role}: the stated role, else the one the working type and the rules imply."""
    typed, scripted = typing_of(doc), doc.get("columns") or {}
    roles = {}
    for c in column_universe(doc)[0]:
        stated = (typed.get(c) or {}).get("role")
        if stated:
            roles[c] = stated
            continue
        t = working_type(c, typed, scripted)
        if t in ROLE_OF_TYPE:
            roles[c] = ROLE_OF_TYPE[t]
        elif c in heads:
            roles[c] = "derived"
        elif t == "text" and c not in basis:
            roles[c] = "text"
        else:
            roles[c] = "dimension"
    return roles


class Checker:
    """Collects ERROR lines; WARN / OK lines go straight to `out`."""

    def __init__(self, doc, out):
        self.doc, self.out, self.errs = doc, out, []
        self.cols, self.authoritative = column_universe(doc)

    def err(self, msg):
        self.errs.append(msg)

    def warn(self, msg):
        self.out(f"WARN: {msg}")

    def known(self, c, where):
        if self.authoritative and c not in self.cols:
            self.err(f"{where}: {c!r} is not a column of the dataset")
            return False
        return True

    def consequences(self, block, where):
        """Contract §3: the decision, the alternatives weighed, the effect downstream."""
        if not isinstance(block, dict):
            self.err(f"{where}: missing its consequence block (contract §3: every decision "
                     "carries the alternatives weighed and what each changes downstream)")
            return
        if not str(block.get("decision") or "").strip():
            self.err(f"{where}.consequences.decision is required")
        if not isinstance(block.get("alternatives"), list):
            self.err(f"{where}.consequences.alternatives must be a list (empty when the "
                     "decision was forced, and then the decision says why)")
        self._downstream(block.get("downstream"), where)

    def _downstream(self, ds, where):
        if isinstance(ds, str) and ds.strip():
            return                               # the one-sentence "no downstream consequence"
        if not isinstance(ds, dict):
            self.err(f"{where}.consequences.downstream must name the effect on "
                     f"{', '.join(DOWNSTREAM)} (or be one sentence saying there is none)")
            return
        missing = [k for k in DOWNSTREAM if not str(ds.get(k) or "").strip()]
        if missing:
            self.err(f"{where}.consequences.downstream lacks {', '.join(missing)} — each entry "
                     "names a concrete effect on this dataset or says 'none'")
