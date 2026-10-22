"""
smoke_geometry_checks.py — what `smoke_geometry.py` holds the probe's observations against: the
counts and targets the layer's own data implies (`expectations`), and the checks of one probed
copy (`check_copy`), each named `<copy>:<what>` in the report (report-contract §6).

Imported by `smoke_geometry.py`; not a command.
"""
from __future__ import annotations


def kept(d):
    return isinstance(d, dict) and ((d.get("provenance") or {}).get("semantic") or {}).get(
        "status") == "confirmed" and d.get("kept", True) is not False


def columns_of(doc):
    src = doc.get("source") or {}
    cols = list(src.get("column_names") or (src["columns"] if isinstance(src.get("columns"), list) else []))
    extra = [t.get("column") for t in doc.get("typing") or [] if isinstance(t, dict)]
    extra += list((doc.get("basis") or {}).get("members") or [])
    for d in doc.get("derivations") or []:
        if isinstance(d, dict):
            extra += [d.get("column")] + list(d.get("body") or [])
    extra += list(((doc.get("explore") or {}).get("columns") or {}).keys())
    for c in extra:
        if isinstance(c, str) and c not in cols:
            cols.append(c)
    return cols


def expectations(doc):
    """The counts and the targets the probe's observations are held against."""
    heads = {d["column"]: d for d in doc.get("derivations") or [] if kept(d)}
    basis = list((doc.get("basis") or {}).get("members") or [])
    fd_pairs = set()
    for f in doc.get("functional_dependencies") or []:
        if isinstance(f, dict) and f.get("exact"):
            for a in f.get("lhs") or []:
                fd_pairs.add((a, f.get("rhs")))
    edges = {(b, h) for h, d in heads.items() for b in d.get("body") or []}
    cycles = [c for c in doc.get("cycles") or [] if isinstance(c, dict)
              and len(c.get("members") or []) > 1 and len(c.get("orientations") or []) > 1]
    derived_member = None
    for cy in cycles:
        derived_member = next((m for m in cy["members"] if m in heads), None)
        if derived_member:
            break
    target = derived_member or next(iter(heads), None)
    measured = set((doc.get("dependencies") or {}).get("columns") or [])
    k = len([c for c in basis if c in measured])
    E = doc.get("explore") or {}
    P = doc.get("partitions") or {}
    cands = [c for c in P.get("candidates") or [] if isinstance(c, dict)]
    return {"columns": columns_of(doc), "basis": basis, "heads": heads, "edges": len(edges),
            "fd_edges": len(fd_pairs - edges), "cycles": cycles, "target": target,
            "cells": k * (k - 1), "sv": len((doc.get("space") or {}).get("singular_values") or []),
            "n": E.get("n") if isinstance(E, dict) else None, "pca": bool((E or {}).get("pca")),
            "candidates": cands,
            "candidate": P.get("chosen") or (cands[0].get("label") if cands else None),
            "explore_cols": set(((E or {}).get("columns") or {}).keys()),
            "explore": isinstance(E, dict) and (E.get("n") or 0) >= 10}


class Checks:
    def __init__(self):
        self.failures, self.count = [], 0

    def __call__(self, cond, name, detail=""):
        self.count += 1
        if cond:
            print(f"  OK   {name}")
        else:
            self.failures.append(f"{name}{': ' + str(detail) if detail != '' else ''}")
            print(f"  FAIL {name}{': ' + str(detail) if detail != '' else ''}")
        return bool(cond)


def check_full(check, c, X):
    check(c.get("scree_bars") == X["sv"], "full:scree-bars", f"{c.get('scree_bars')} != {X['sv']}")
    check(c.get("points", 0) >= min(10, X["n"] or 0) and c.get("points", 0) <= (X["n"] or 0),
          "full:scatter-points", f"{c.get('points')} of {X['n']}")
    if X["pca"]:
        check(str(c.get("pca_label") or "").startswith("PC1") and "rotated" in str(c.get("pca_label")),
              "full:pca-axes-labelled-rotated", c.get("pca_label"))
        check(c.get("pca_loadings"), "full:pca-loadings-panel")
        check(c.get("pca_points", 0) > 0, "full:pca-points", c.get("pca_points"))
        check(c.get("named_back"), "full:pca-toggles-back")
    check(c.get("row_values", 0) > 0, "full:hover-row-values", c.get("row_values"))
    if any(h in X["explore_cols"] for h in X["heads"]):
        check(c.get("row_rules", 0) > 0, "full:hover-derived-tagged-with-rule", c.get("row_rules"))
    check(0 < c.get("brushed", 0) < (X["n"] or 0), "full:brush-selects-some-rows",
          f"{c.get('brushed')} of {X['n']}")
    check(c.get("banner"), "full:selection-banner")
    check(c.get("dimmed", 0) > 0, "full:unselected-points-dimmed", c.get("dimmed"))
    check(c.get("cleared"), "full:selection-clears")
    check(c.get("cells") == X["cells"], "full:heatmap-cells", f"{c.get('cells')} != {X['cells']}")
    if c.get("cell_pair"):
        check(c.get("axes_after_cell") == c["cell_pair"], "full:cell-opens-pair-as-axes",
              f"{c.get('axes_after_cell')} != {c.get('cell_pair')}")
        check(c.get("pair_panel"), "full:pair-panel")


def check_graph(check, label, c, X):
    check(c.get("nodes") == len(X["columns"]), f"{label}:graph-nodes",
          f"{c.get('nodes')} != {len(X['columns'])}")
    check(c.get("basis_nodes") == len(X["basis"]), f"{label}:basis-nodes",
          f"{c.get('basis_nodes')} != {len(X['basis'])}")
    check(c.get("edges") == X["edges"], f"{label}:rule-edges", f"{c.get('edges')} != {X['edges']}")
    check(c.get("fd_edges") == X["fd_edges"], f"{label}:fd-edges",
          f"{c.get('fd_edges')} != {X['fd_edges']}")
    check(c.get("cycles") == len(X["cycles"]), f"{label}:cycles-drawn",
          f"{c.get('cycles')} != {len(X['cycles'])}")
    check(sorted(c.get("basis_before") or []) == sorted(X["basis"]), f"{label}:report-basis",
          c.get("basis_before"))
    if X["target"]:
        check(c.get("card"), f"{label}:node-card")
        check(c.get("card_prov") == 3, f"{label}:card-provenance-triple", c.get("card_prov"))
    cy = next((k for k in X["cycles"] if X["target"] in k["members"]), None)
    if not cy:
        return
    check(c.get("make_basis"), f"{label}:make-basis-offered")
    oi = c.get("orientation")
    if not isinstance(oi, int) or oi >= len(cy["orientations"]):
        check(False, f"{label}:orientation-chosen", oi)
        return
    o = cy["orientations"][oi]
    der = set(o.get("derived") or [])
    expected = {b for b in X["basis"] if b not in der} | {m for m in cy["members"] if m not in der}
    after = set(c.get("basis_after") or [])
    check(X["target"] in after, f"{label}:reorient-target-joins-basis", sorted(after))
    check(after == expected, f"{label}:reorient-basis", f"{sorted(after)} != {sorted(expected)}")
    check(set((c.get("basis_line_after") or "").split(",")) == expected, f"{label}:basis-line-updates",
          c.get("basis_line_after"))
    check(c.get("changed_chip"), f"{label}:reorient-marked")
    check(c.get("decision_after") == ((o.get("consequences") or {}).get("decision")),
          f"{label}:orientation-consequences-shown", c.get("decision_after"))
    check(c.get("others_after") != c.get("others_before"), f"{label}:candidates-update",
          f"{c.get('others_before')} -> {c.get('others_after')}")
    check(sorted(c.get("basis_restored") or []) == sorted(X["basis"]), f"{label}:restore-report-basis",
          c.get("basis_restored"))
    check(c.get("others_restored") == c.get("others_before"), f"{label}:restore-candidates")


def check_partition(check, label, c, X, explore):
    check(c.get("cards") == len(X["candidates"]), f"{label}:candidate-cards",
          f"{c.get('cards')} != {len(X['candidates'])}")
    cand = next((k for k in X["candidates"] if k.get("label") == X["candidate"]), None)
    if not cand:
        return
    check(c.get("cand_on"), f"{label}:candidate-selectable")
    check(c.get("leak_nodes") == sorted(cand.get("dropped_for_leakage") or []),
          f"{label}:leakage-struck", c.get("leak_nodes"))
    check(c.get("feature_nodes") == sorted(cand.get("features") or []), f"{label}:features-marked",
          c.get("feature_nodes"))
    check(c.get("label_nodes") == [cand["label"]], f"{label}:label-marked", c.get("label_nodes"))
    want = cand.get("input_dim") if isinstance(cand.get("input_dim"), int) else len(cand.get("features") or [])
    check(c.get("input_dim") == want, f"{label}:input-dim-shown", f"{c.get('input_dim')} != {want}")
    if explore and cand["label"] in X["explore_cols"]:
        check(c.get("colour_after") == cand["label"], f"{label}:space-coloured-by-label",
              c.get("colour_after"))



def check_copy(check, label, R, err, X):
    """Every check of one probed copy: `full` (the layer as written) or `no-explore`."""
    c = R["c"]
    check(c.get("tab_button"), f"{label}:tab-button")
    check(c.get("pane"), f"{label}:pane")
    check(c.get("hidden_by_default"), f"{label}:pane-inactive-by-default")
    check(c.get("active_after_click"), f"{label}:tab-activates-pane")
    for v in ("space", "graph", "orthogonality", "partition"):
        check((c.get("views") or {}).get(v), f"{label}:view-{v}")
    check(c.get("hook"), f"{label}:window.__geometry")
    full = label == "full" and X["explore"]
    if full:
        check(c.get("empty_space") == 0 and c.get("empty_ortho") == 0, "full:no-empty-states")
        check_full(check, c, X)
    else:
        check(c.get("empty_space", 0) >= 1, f"{label}:space-empty-state")
        check(c.get("empty_ortho", 0) >= 1, f"{label}:orthogonality-empty-state")
    check_graph(check, label, c, X)
    check_partition(check, label, c, X, full)
    check(not R["errors"], f"{label}:no-js-errors", "; ".join(R["errors"])[:400])
    bad = [ln for ln in (err or "").splitlines() if "ERROR:" in ln and "CONSOLE" in ln]
    check(not bad, f"{label}:chromium-console", "; ".join(bad)[:300])
