// The words a node of a graph answer goes by (D4 §6.5), for every place that names one: the try and
// many views' drawings and lists (js/views/shapes/graph.js), the stepper's convolution steps
// (js/how/explainers/graph.js) and the network canvas's message captions (js/net/renderers/graph.js).
//
// A node id in an explanation is the node's index in the graph the service assembled (the reference
// nodes first, then the request's rows): it is no name of the node, and the page never writes it as
// one. A node is named, in this order:
//   the explained node of a node-level answer   "this {node.one}"
//   a row of the request (`rows`, in the order sent)   its key column's value, else its place there
//   a reference node the service named          "{node.one} <key>": the data's own key, which the
//                                               explanation carries beside the answer when the graph
//                                               keys its nodes (`sample.graph.names`, inspect-contract
//                                               §5.4: a two-column key_pair)
//   any other node of a node-level answer       "neighbour k": its rank among the nodes the answer
//                                               lists (the attribution's neighbours first, then the
//                                               node payloads' kept ids and messages) — an ordinal
//                                               of this answer, said as one
//   any other node of a graph-level answer      "{node.one} k": its place among its graph's nodes
//                                               (`order`, the graph's node ids in row order, else
//                                               the ids the answer lists, ascending)
//   a node the answer does not list             "another {node.one}"
//
//   nodeNamer(ex, {rows, key, level, refNodes, order}) -> (id) => string
//   rowName(row, k, key) -> string               a request row's name (the editor's chips, the lists)
//   indexNamed(ex, texts) -> [string]            the probe's rule (core/probe-walk.js, P-STEP
//                                                `node_index_named`): where `texts` name a node of a
//                                                node-level answer by its index ("node 1001",
//                                                "node #1001"), at most eight of them
import { tText, data } from "./text.js";

const finite = (v) => typeof v === "number" && Number.isFinite(v);
const has = (o, k) => !!o && Object.prototype.hasOwnProperty.call(o, k);

// A request row's name: its key column's value, else its place (1-based) among the rows sent.
export function rowName(row, k, key) {
  return key && row && row[key] !== null && row[key] !== undefined && row[key] !== "" ? data(row[key]) : tText("graph.node.nth", { k: k + 1 });
}

// The graph attribution of an explanation, or null.
function graphAttribution(ex) {
  const a = ex && ex.attribution;
  return a && a.shape === "graph" ? a : null;
}

// The node ids an answer lists, in its own order: the attribution's neighbours (nearer hops first),
// then each node payload's kept ids and the ends of its messages.
const FORMS = Object.freeze(["in", "out", "grad_out", "relevance"]);
function listedIds(ex, a) {
  const out = new Set(), push = (id) => { if (Number.isInteger(id)) out.add(id); };
  ((a && a.neighbours) || []).forEach((n) => push(n && n.node));
  for (const p of Object.values((ex && ex.nodes) || {})) {
    if (!p || typeof p !== "object") continue;
    FORMS.forEach((f) => { const v = p[f]; if (v && v.form === "nodes" && Array.isArray(v.ids)) v.ids.forEach(push); });
    const msgs = p.extra && Array.isArray(p.extra.messages) ? p.extra.messages : [];
    msgs.forEach((m) => { push(m && m.src); push(m && m.dst); });
  }
  return [...out];
}

export function nodeNamer(ex, { rows = [], key = null, level = null, refNodes = null, order = null } = {}) {
  const a = graphAttribution(ex), g = (ex && ex.sample && ex.sample.graph) || null;
  const lv = level || (g && g.level) || (a && a.level) || "node";
  const target = a && a.target_node !== undefined ? a.target_node : g ? g.target_node : null;
  const named = (g && g.names && g.names.of) || {};
  const sent = Array.isArray(rows) ? rows : [];
  // where the request's rows start among the answer's node ids: after the reference nodes (node
  // level), at 0 when the answer's graph is exactly the rows sent (graph level)
  const offset = lv === "node" ? (g && finite(g.nodes) ? g.nodes - sent.length : refNodes)
    : g && g.nodes === sent.length ? 0 : null;
  const listed = listedIds(ex, a);
  const rank = new Map(listed.filter((id) => id !== target).map((id, i) => [id, i + 1]));
  const place = new Map((Array.isArray(order) ? order : listed.slice().sort((p, q) => p - q)).map((id, i) => [id, i + 1]));
  return (id) => {
    if (lv === "node" && target !== null && id === target) return tText("graph.node.this");
    const k = offset === null || !finite(offset) ? -1 : id - offset;
    if (k >= 0 && k < sent.length) return rowName(sent[k], k, key);
    if (has(named, String(id))) return tText("graph.node.key", { key: data(named[String(id)]) });
    if (lv === "graph") return place.has(id) ? tText("graph.node.nth", { k: place.get(id) }) : tText("graph.node.other");
    return rank.has(id) ? tText("graph.node.ref", { k: rank.get(id) }) : tText("graph.node.other");
  };
}

// The node ids of a node-level answer written as names in `texts`: "#<digits>", or the lexicon's node
// noun followed by one of the answer's node ids. An id that is also a key the service sent is not
// looked for (the key is the data's own name for a node).
export function indexNamed(ex, texts) {
  const g = ex && ex.sample && ex.sample.graph;
  if (!g || g.level !== "node") return [];
  const keys = new Set(Object.values((g.names && g.names.of) || {}).map(String));
  const ids = [...new Set([g.target_node, ...listedIds(ex, graphAttribution(ex))])].filter((i) => Number.isInteger(i) && !keys.has(String(i)));
  const noun = tText("graph.node.nth", { k: 0 }).replace(/\s*0\s*$/, "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`#\\d+${noun && ids.length ? `|\\b${noun} (?:${ids.join("|")})\\b` : ""}`, "g");
  const found = new Set();
  texts.forEach((t) => (String(t || "").match(re) || []).forEach((m) => found.add(m)));
  return [...found].slice(0, 8);
}
