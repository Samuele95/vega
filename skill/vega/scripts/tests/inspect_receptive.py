"""
inspect_receptive.py — EI11 of D6 §8.2: what an explanation says it read is what the request
sent, and nothing else.

    ei11_window(explanation, request, sample_record, length)
        the explained window ends at the explained row and spans `length` rows of its own
        series across `history + rows`, laid out as the assembler lays a series out (history
        first, each part in `order_by` order); its raw row is the request's row
    ei11_node(explanation, request, reference)
        node level: every node of the receptive field is a reference node or the explained
        query; an edge between reference nodes is a reference edge; an edge touching the query
        runs reference -> query (no query sends a message) and, on a key_pair graph, pairs the
        query's keys with the reference node's raw keys (edges_for)
    ei11_graph(explanation, request, sample_record, reference)
        graph level: the receptive field is the explained graph's own rows, and every edge
        joins two of them (edges_within), on a key_pair graph by their keys

Each returns `(ok, detail)` observations. Positions: a window's `span` counts `history` rows
negatively (−len(history) … −1) and `rows` from 0; a graph's node ids are batch positions — at
node level the reference nodes first (their `ref` ids), then the query rows; at graph level
the request rows.
"""
from __future__ import annotations

from typing import Mapping

from inspect_context import keys_pair
from check_kit import close

__all__ = ["ei11_graph", "ei11_node", "ei11_window", "window_rows"]


def _same_value(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            not isinstance(a, bool) and not isinstance(b, bool):
        return close(a, b, 1e-9, 1e-12)
    return str(a) == str(b)


def _order_key(value) -> tuple:
    """A row's place in its series: numbers by value, anything else (an ISO time) as text."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0, float(value), ""
    return 1, 0.0, str(value)


def window_rows(request: Mapping, anchor: int, sample: Mapping, length: int) -> list:
    """The request positions of the window ending at `anchor`, oldest first."""
    rows = list(request.get("rows") or ())
    history = list(request.get("history") or ())
    series, order = sample.get("series_by"), sample.get("order_by")
    every = [(i - len(history), r) for i, r in enumerate(history)] + list(enumerate(rows))
    key = (rows[anchor] if anchor >= 0 else history[anchor]).get(series) if series else None
    mine = [(pos, r) for pos, r in every if series is None or r.get(series) == key]
    parts = [[p for p in mine if p[0] < 0], [p for p in mine if p[0] >= 0]]
    if order:
        parts = [sorted(part, key=lambda p: _order_key(p[1].get(order))) for part in parts]
    laid = [pos for part in parts for pos, _ in part]
    if anchor not in laid:
        return []
    end = laid.index(anchor)
    return laid[max(0, end - length + 1): end + 1]


def ei11_window(explanation: Mapping, request: Mapping, sample: Mapping, length: int) -> list:
    told = explanation.get("sample") or {}
    anchor, span = int(told.get("row")), told.get("span") or []
    window = window_rows(request, anchor, sample, length)
    out = [(len(window) == length and list(span) == [window[0], window[-1]],
            f"span {span} = the {length} rows of its series ending at row {anchor} "
            f"({window[:1]}…{window[-1:]})")]
    rows, history = list(request.get("rows") or ()), list(request.get("history") or ())
    sent = rows[anchor] if anchor >= 0 else history[anchor]
    raw = (explanation.get("inputs") or {}).get("raw") or {}
    differ = [c for c, v in raw.items() if c in sent and not _same_value(v, sent[c])]
    shaped = (explanation.get("inputs") or {}).get("shaped") or []
    out.append((not differ and len(shaped) == length,
                f"the explained raw row is request row {anchor} (differ: {differ[:3]}); "
                f"{len(shaped)} shaped step(s)"))
    return out


def _field(explanation: Mapping) -> tuple:
    """(node ids, edges (src, dst)) the explanation names: the attribution's neighbours and
    edges, every graph node's kept ids and every conv message."""
    att = explanation.get("attribution") or {}
    nodes = {int(n["node"]) for n in att.get("neighbours") or ()}
    edges = {(int(e["src"]), int(e["dst"])) for e in att.get("edges") or ()}
    for payload in (explanation.get("nodes") or {}).values():
        for side in ("in", "out"):
            reduced = payload.get(side) or {}
            if reduced.get("form") == "nodes":
                nodes |= {int(i) for i in reduced.get("ids") or ()}
        for m in (payload.get("extra") or {}).get("messages") or ():
            if m.get("src") != m.get("dst") and not m.get("root"):
                edges.add((int(m["src"]), int(m["dst"])))
    return nodes, edges


def ei11_node(explanation: Mapping, request: Mapping, reference) -> list:
    graph = (explanation.get("sample") or {}).get("graph") or {}
    target, size = int(graph.get("target_node")), reference.size
    row = int((explanation.get("sample") or {}).get("row") or 0)
    query = list(request.get("rows") or ())[row]
    nodes, edges = _field(explanation)
    strangers = sorted(n for n in nodes if n >= size and n != target)
    out = [(not strangers and target == size + row,
            f"{len(nodes)} receptive-field node(s): reference nodes and the query {target} "
            f"(= {size} reference nodes + row {row}); others: {strangers[:4]}")]
    inner = [(s, d) for s, d in edges if s < size and d < size]
    missing = [e for e in inner if not reference.has_edge(*e)]
    out.append((not missing, f"{len(inner)} edge(s) among reference nodes are reference edges "
                             f"(missing: {missing[:3]})"))
    touching = [(s, d) for s, d in edges if s >= size or d >= size]
    wrong = [e for e in touching if e[1] != target or e[0] >= size]
    columns = reference.key_columns()
    unpaired = [] if columns is None else [
        e for e in touching if e not in wrong and not keys_pair(
            query, {c: reference.raw(e[0], c) for c in columns}, columns)]
    out.append((not wrong and not unpaired,
                f"{len(touching)} query edge(s) run reference -> query (wrong: {wrong[:3]})" +
                ("" if columns is None else f", pairing keys {list(columns)} (unpaired: "
                                            f"{unpaired[:3]})")))
    return out


def ei11_graph(explanation: Mapping, request: Mapping, sample: Mapping, reference) -> list:
    rows = list(request.get("rows") or ())
    told = explanation.get("sample") or {}
    group = sample.get("graph_id")
    anchor = rows[int(told.get("row") or 0)]
    members = {i for i, r in enumerate(rows) if group is None or r.get(group) == anchor.get(group)}
    nodes, edges = _field(explanation)
    outside = sorted(n for n in nodes if n not in members)
    out = [(not outside and len(members) == int((told.get("graph") or {}).get("nodes") or 0),
            f"the receptive field is the explained graph's {len(members)} row(s) "
            f"(outside: {outside[:4]})")]
    columns = reference.key_columns()
    crossing = [e for e in edges if e[0] not in members or e[1] not in members]
    unpaired = [] if columns is None else [
        e for e in edges if e not in crossing and not keys_pair(rows[e[0]], rows[e[1]], columns)]
    out.append((not crossing and not unpaired,
                f"{len(edges)} edge(s) inside the graph (crossing: {crossing[:3]})" +
                ("" if columns is None else f", pairing keys {list(columns)} (unpaired: "
                                            f"{unpaired[:3]})")))
    return out

