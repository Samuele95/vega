// The graph shape (D4 §6.5, R-A3): a sample is a node of a graph (level node) or a whole small graph
// (level graph). The service derives every edge itself from the linking columns (the reference graph
// of train ∪ valid at node level, the rows of the same graph at graph level), so a request carries
// rows only: {rows, at}.
//   editor        node level: the node's own form, one field per raw column of /v1/schema, the
//                 linking columns marked (they decide its neighbours); graph level: the graph's nodes
//                 as chips, the chosen node's form (node selection edits it), and the try view's
//                 edit toggle names the graph it edits, not one row
//   sources       many at once: the demonstration sample (one node, or one graph, per item), the
//                 shipped examples, the reader's CSV (nodes; at graph level, rows naming their graph);
//                 the rows of the last batch are kept, so a table row opens its whole graph
//   attribution   what tipped the answer: the neighbourhood drawn (the node in the centre, its
//                 neighbours by hop; at graph level the whole graph), each node sized and coloured by
//                 what it tipped, the edges by their relevance; the node's own families of features;
//                 the share of each hop; the neighbours that weighed most (click one: its features);
//                 what lies beyond the nodes drawn and any gap the method reports, each a row of its
//                 own (A31)
//   inputStep     the stepper's first step: the node's own shaped features, and how many neighbours
//                 each hop brought, the edges' origin (the linking columns); at graph level the
//                 graph's largest shaped cells, read as one whole graph whose edges join the rows of
//                 that graph (every one of them can change the answer, a row of another graph none)
//   drawInput     the input segment (net/renderers/input.js: families, with its ego-graph glyph)
import { el } from "../../core/dom.js";
import { fmtNum, sg, plain } from "../../core/fmt.js";
import { termRow } from "../../core/bars.js";
import { t, tText, tn, tnText, data, cls } from "../../core/text.js";
import { familiesOf } from "../../net/families.js";
import { joinList } from "../../net/guide.js";
import inputRenderer from "../../net/renderers/input.js";
import { S } from "../../core/state.js";
import { viewState, methodNote } from "../shared.js";
import { explainedName } from "../payloads.js";
import { cellOf } from "./row.js";
import { nodeNamer, rowName as rowNameOf } from "../../core/node-names.js";

const SVG = "http://www.w3.org/2000/svg";
const K_TOP = 6, K_NEAR = 8, DRAWN_MAX = 40, EGO = 150, PAD = 14;
const has = (o, k) => !!o && Object.prototype.hasOwnProperty.call(o, k);
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const columnsOf = (schema) => (schema && Array.isArray(schema.input) ? schema.input : []);

// ---- the lane's graph --------------------------------------------------------------------------------
// {level, graphId, links, key, refNodes}: node | graph; the column naming a row's graph (graph level);
// the linking columns (role graph); the column that names a node; how many reference nodes precede
// the request's rows in an answer's node ids. Only a key_pair over two columns keys its nodes (the
// first is a row's own key, the second the key of the node it links to, inspect-contract §5.4): the
// columns of knn or spatial are coordinates, and a one-column key_pair's value is shared by the rows
// it links, so neither names a node.
export function graphOf(schema, env = {}) {
  const sample = (schema && schema.sample) || {}, lane = env.lane, net = env.graph ? env.graph.sample : {};
  const level = (lane && lane.level) || sample.level || (net && net.level) || "node";
  const graphId = sample.graph_id || null, links = columnsOf(schema).filter((c) => c.role === "graph").map((c) => c.name);
  const ref = sample.reference && Number.isInteger(sample.reference.nodes) ? sample.reference.nodes : null;
  const keys = links.filter((c) => c !== graphId), keyed = (!sample.method || sample.method === "key_pair") && keys.length >= 2;
  return { level, graphId, links, key: keyed ? keys[0] : null, refNodes: level === "node" ? ref : 0 };
}

// The rows of the graph a row belongs to: first in `batch` (the rows many at once last sent, when the
// row is one of its), then in the examples the page holds (the demonstration items, the shipped rows).
// A row of many at once opens its whole graph. Ids compare as text: a CSV's "7" is the service's 7.
export function graphRows(examples, graphId, row, batch = null) {
  if (!graphId || !row || !has(row, graphId)) return [row];
  const id = String(row[graphId]), same = (r) => r && r[graphId] !== null && r[graphId] !== undefined && String(r[graphId]) === id;
  const demo = examples && examples.demo && Array.isArray(examples.demo.items) ? examples.demo.items.map((it) => it.rows || []) : [];
  const test = examples && examples.test && Array.isArray(examples.test.rows) ? [examples.test.rows] : [];
  const pools = [...(Array.isArray(batch) ? [batch] : []), ...demo, ...test];
  const found = pools.map((rows) => rows.filter(same)).find((rows) => rows.length > 1);
  return found ? found.slice() : [row];
}

// The rows many at once last sent (its sources' run() keep them): a table row clicked looks for its
// graph there first, the reader's CSV included, which the examples never hold.
const batchState = () => viewState("graph-batch", { rows: null, read: 0 });
function keep(rows) { const b = batchState(); b.read++; b.rows = Array.isArray(rows) ? rows : null; }

// The rows a preset (or a row of many at once: a preset without an id) puts in the editor.
export function presetRows(preset, env = {}) {
  if (!preset) return [];
  const rows = Array.isArray(preset.rows) && preset.rows.length ? preset.rows : preset.row ? [preset.row] : [];
  const G = graphOf(env.schema, env), many = preset.id === null || preset.id === undefined;
  if (G.level !== "graph") return rows.length ? [rows[rows.length - 1]] : [];
  return rows.length === 1 ? graphRows(env.examples, G.graphId, rows[0], many ? batchState().rows : null) : rows.slice();
}

// ---- the reader's CSV, read on the page too ------------------------------------------------------------
// Its rows by the header, each cell typed as the editor types it (cellOf, by /v1/schema's dtype), the
// schema's input columns only: the rows a table row's graph is found in (RFC 4180: quoted fields,
// doubled quotes, CRLF).
export function csvRows(text, schema) {
  const recs = [], cols = columnsOf(schema);
  let rec = [], cell = "", q = false;
  const s = String(text || "").replace(/^\uFEFF/, "");
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === '"' && s[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; continue; }
    if (c === '"') q = true;
    else if (c === ",") { rec.push(cell); cell = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && s[i + 1] === "\n") i++; rec.push(cell); recs.push(rec); rec = []; cell = ""; }
    else cell += c;
  }
  if (cell !== "" || rec.length) { rec.push(cell); recs.push(rec); }
  const body = recs.filter((r) => r.some((v) => v !== "")), head = body.shift() || [];
  const dtype = new Map(cols.map((c) => [c.name, c.dtype || ""])), keepCol = (h) => !cols.length || dtype.has(h);
  return body.map((r) => Object.fromEntries(head.map((h, k) => [h, r[k] ?? ""]).filter(([h]) => keepCol(h)).map(([h, v]) => [h, cellOf(v, dtype.get(h))])));
}

// The CSV source's run: many at once uploads the file; the page reads it too, for the rows' graphs.
function readCsv(file, schema) {
  keep(null);
  const b = batchState(), mine = b.read;
  if (file && typeof file.text === "function") file.text().then((txt) => { if (b.read === mine) b.rows = csvRows(txt, schema); }).catch(() => {});
  return { file };
}

// ---- the editor --------------------------------------------------------------------------------------
const edState = () => viewState("graph-editor", { host: null, schema: null, rows: [], at: 0, G: null });

function field(col, G) {
  const link = G.links.includes(col.name);
  return el("label", { class: link ? "graph-link" : "" }, data(col.name), el("input", { "data-col": col.name, "data-type": col.dtype || "", value: "" }),
    link ? el("small", { class: "graph-link-note" }, t(col.name === G.graphId ? "graph.edit.graph_id" : "graph.edit.link")) : null);
}

// The chosen row's values into the fields, or the fields back into the chosen row.
function fill(st) {
  const row = st.rows[st.at] || {};
  for (const input of st.host.querySelectorAll("input")) {
    const v = row[input.dataset.col];
    input.value = v === null || v === undefined ? "" : String(v);
  }
}
function commit(st) {
  if (!st.host || !st.rows.length) return;
  const row = { ...st.rows[st.at] };
  for (const input of st.host.querySelectorAll("input")) row[input.dataset.col] = cellOf(input.value, input.getAttribute("data-type"));
  st.rows[st.at] = row;
}

// A node's name in the editor and the drawings: its key column's value, else its place.
const rowName = (row, k, G) => rowNameOf(row, k, G.key);

function renderChips(st) {
  const box = st.host.querySelector(".graph-nodes");
  if (!box) return;
  box.hidden = st.G.level !== "graph";
  box.replaceChildren(el("span", { class: "caps muted" }, tn("graph.edit.nodes", st.rows.length)), ...st.rows.map((row, k) => {
    const b = el("button", { type: "button", class: "graph-chip" + (k === st.at ? " on" : "") }, rowName(row, k, st.G));
    b.onclick = () => { commit(st); st.at = k; renderChips(st); fill(st); };
    return b;
  }));
}

// The try view's edit toggle (its text slot, data-t) says what the editor edits: at graph level the
// whole graph (graph.edit.toggle_graph), else the shell's own words (try.edit). The slot keeps its
// key, so a later pass over the shell's text slots says the same.
export const TOGGLE_KEYS = Object.freeze({ graph: "graph.edit.toggle_graph", node: "try.edit" });
function toggleWords(G) {
  const toggle = [...document.querySelectorAll(".disclose")].find((b) => b.getAttribute("data-for") === "edit");
  const slot = toggle ? [...toggle.querySelectorAll("span")].find((n) => n.getAttribute("data-t") !== null) : null;
  const key = G.level === "graph" ? TOGGLE_KEYS.graph : TOGGLE_KEYS.node;
  if (!slot || slot.getAttribute("data-t") === key) return;
  slot.setAttribute("data-t", key);
  slot.replaceChildren(G.level === "graph" ? t("graph.edit.toggle_graph") : t("try.edit"));
}

export const editor = Object.freeze({
  mount(host, schema, env) {
    const st = edState();
    Object.assign(st, { host, schema, G: graphOf(schema, env || {}), rows: [], at: 0 });
    host.replaceChildren(el("div", { class: "graph-nodes" }), ...columnsOf(schema).map((col) => field(col, st.G)));
    renderChips(st);
    toggleWords(st.G);
  },
  load(preset, env) {
    const st = edState();
    if (!st.host) return;
    st.rows = presetRows(preset, { ...(env || {}), schema: st.schema }).map((r) => ({ ...r }));
    st.at = 0;
    renderChips(st); fill(st);
  },
  read() {
    const st = edState();
    if (!st.rows.length) st.rows = [{}];
    commit(st);
    if (!st.G || st.G.level !== "graph") return { rows: [st.rows[st.at]], at: 0 };
    return { rows: st.rows.slice(), at: st.at };
  },
  describe(preset) {
    if (!preset) return { title: "", text: "" };
    const typical = preset.expected !== null && preset.expected !== undefined && (preset.group === "typical" || preset.kind === null) && !Array.isArray(preset.expected);
    return { title: typical ? tText("try.typical", { name: cls(preset.expected) }) : preset.name, text: preset.text };
  },
});

// ---- many at once ------------------------------------------------------------------------------------
// {rows, truth, history}: node level, one node per demonstration item and its truth; graph level,
// every item's rows, each with its graph's truth (the service answers a graph-level request row by
// row, each row with its graph's answer and graph_id, so the truth goes with the rows).
export function demoBatch(examples, level) {
  const d = examples && examples.demo, items = d && Array.isArray(d.items) ? d.items.filter((it) => Array.isArray(it.rows) && it.rows.length) : [];
  if (!items.length) return null;
  const truth = items.map((it) => it.truth ?? null);
  if (level !== "graph") return { rows: items.map((it) => it.rows[it.rows.length - 1]), truth, history: null };
  return { rows: items.flatMap((it) => it.rows), truth: items.flatMap((it, k) => it.rows.map(() => truth[k])), history: null };
}

export function sources(examples, env = {}) {
  const G = graphOf(env.schema, env), demo = demoBatch(examples, G.level), test = examples && examples.test && Array.isArray(examples.test.rows) && examples.test.rows.length ? examples.test.rows : null;
  return [
    demo ? { id: "demo", title: t("many.src.demo"), sub: t("many.src.demo_note"), run: () => { keep(demo.rows); return demo; } } : null,
    test ? { id: "test", title: t("many.src.test"), sub: t("many.src.test_note"), run: () => { keep(test); return { rows: test, truth: null, history: null }; } } : null,
    { id: "csv", file: true, title: t("many.src.csv"), sub: t(G.level === "graph" ? "graph.src.csv_graph" : "graph.src.csv_node"), run: (file) => readCsv(file, env.schema || edState().schema) },
  ].filter(Boolean);
}

// ---- reading an answer -------------------------------------------------------------------------------
// The graph attribution of an explanation@2 (D3 §8.3), or null.
export function graphAttribution(ex) {
  const a = ex && ex.schema === "vega/explanation@2" ? ex.attribution : null;
  return a && a.shape === "graph" ? a : null;
}

// Node id → the name the page gives it (js/core/node-names.js): the explained node, a row of the
// request, a reference node by the key the service sent beside the answer, else by its rank in the
// answer — never by its index in the graph the service assembled, which is no name of it.
export function namer(ex, rows, G) {
  return nodeNamer(ex, { rows, key: G.key, level: G.level, refNodes: G.refNodes });
}

// The namer of an answer as the try view's editor holds it, for the stepper (its env.nodeName).
export function nodeNames(ex, env = {}) {
  const st = edState();
  return namer(ex, st.rows || [], st.G || graphOf(env.schema || st.schema, env));
}

// The nodes a drawing of the answer shows (at most DRAWN_MAX, nearer hops first as the answer ranks
// them) and the edges between them.
export function drawnOf(a) {
  const nodes = (a.neighbours || []).slice(0, DRAWN_MAX), ids = new Set(nodes.map((n) => n.node));
  const edges = (a.edges || []).filter((e) => ids.has(e.src) && ids.has(e.dst));
  return { nodes, edges, hidden: (a.neighbours || []).length - nodes.length };
}

// Positions: node level, the target in the centre and each hop on a ring further out; graph level,
// every node on one ring.
export function positions(nodes, level, size = EGO) {
  const c = size / 2, R = c - PAD, rings = new Map(), out = new Map();
  nodes.forEach((n) => { const h = level === "graph" ? 1 : n.hop; if (!rings.has(h)) rings.set(h, []); rings.get(h).push(n.node); });
  const far = Math.max(1, ...rings.keys());
  for (const [h, ids] of rings) ids.forEach((id, k) => {
    if (h === 0 && level !== "graph") { out.set(id, { x: c, y: c }); return; }
    const r = R * h / far, a = -Math.PI / 2 + 2 * Math.PI * (k + .5 * (h % 2)) / ids.length;
    out.set(id, { x: c + r * Math.cos(a), y: c + r * Math.sin(a) });
  });
  return out;
}

function svgEl(tag, attrs) {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  return n;
}

// The neighbourhood as an SVG: the edges by their relevance, the nodes by what they tipped; a node
// clicked calls pick(node).
export function neighbourhoodSvg(a, level, { name, pick }) {
  const D = drawnOf(a), at = positions(D.nodes, level), svg = svgEl("svg", { viewBox: `0 0 ${EGO} ${EGO}`, class: "graph-ego", role: "img" });
  const emx = Math.max(1e-12, ...D.edges.map((e) => Math.abs(e.relevance || 0))), nmx = Math.max(1e-12, ...D.nodes.map((n) => Math.abs(n.total || 0)));
  D.edges.forEach((e) => {
    const p = at.get(e.src), q = at.get(e.dst), v = e.relevance || 0;
    svg.append(svgEl("line", { x1: p.x, y1: p.y, x2: q.x, y2: q.y, class: "graph-ego-edge " + (v >= 0 ? "pos" : "neg"),
      "stroke-width": (.5 + 2.5 * Math.abs(v) / emx).toFixed(2), opacity: (.25 + .75 * Math.abs(v) / emx).toFixed(2) }));
  });
  D.nodes.forEach((n) => {
    const p = at.get(n.node), v = n.total || 0, dot = svgEl("circle", { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: (2.5 + 5 * Math.abs(v) / nmx).toFixed(2),
      class: "graph-ego-node " + (v >= 0 ? "pos" : "neg") + (n.node === a.target_node ? " self" : ""), "data-node": n.node });
    const tip = svgEl("title", {});
    tip.append(document.createTextNode(tText("graph.why.node_tip", { name: name(n.node), hop: n.hop, v: sg(v, 3) })));
    dot.append(tip);
    dot.onclick = () => pick(n);
    svg.append(dot);
  });
  return svg;
}

// One node's features that weighed most (the answer's `top`: [feature index, a]).
export function nodeCard(n, a, name) {
  const names = (a.features && a.features.names) || [], top = n.top || [], mx = Math.max(1e-9, ...top.map((x) => Math.abs(x[1])));
  const bars = el("div", { class: "bars diverge eq" });
  top.forEach(([f, v], i) => bars.append(termRow({ name: data(names[f] ?? f), v, max: mx, i, val: [sg(v, 3)] })));
  return el("div", { class: "graph-pick" }, el("div", { class: "caps muted" }, t("graph.why.node", { name: name(n.node), hop: n.hop, v: sg(n.total || 0, 3) })), bars);
}

// ---- what tipped the answer --------------------------------------------------------------------------
function bar({ name, v, mx, i, onclick = null }) {
  const row = termRow({ name, v, max: mx, i, val: [tText(v >= 0 ? "try.why.raised" : "try.why.lowered", { v: Math.abs(v).toPrecision(2) })] });
  if (onclick) { row.classList.add("graph-go"); row.onclick = onclick; }
  return row;
}

// The node's own features, added up per family of the story's input groups (node level).
function selfBars(a, env) {
  if (!Array.isArray(a.self)) return null;
  const names = (a.features && a.features.names) || [], by = new Map(names.map((n, i) => [n, a.self[i] || 0]));
  const fam = familiesOf({ groups: env.story && env.story.input_groups, names, units: names.map((_, i) => [i]) }).families;
  const rows = fam.map((f) => ({ f, v: f.members.reduce((s, m) => s + (by.get(m) || 0), 0) })).sort((x, y) => Math.abs(y.v) - Math.abs(x.v));
  const mx = Math.max(1e-9, ...rows.map((r) => Math.abs(r.v))), bars = el("div", { class: "bars diverge" });
  rows.forEach((r, i) => bars.append(bar({ name: r.f.label, v: r.v, mx, i })));
  return el("div", {}, el("div", { class: "caps muted views-sub" }, t("graph.why.self")), bars);
}

// What each hop brought: the node itself, then one row per hop (node level; a whole graph is one distance).
function hopBars(a) {
  const hops = Object.entries(a.hops || {}).map(([h, v]) => [Number(h), v]).filter(([, v]) => finite(v)).sort((x, y) => x[0] - y[0]);
  if (hops.length < 2) return null;                       // one distance only (a whole graph): nothing to split
  const mx = Math.max(1e-9, ...hops.map(([, v]) => Math.abs(v))), bars = el("div", { class: "bars diverge" });
  hops.forEach(([h, v], i) => bars.append(bar({ name: h === 0 ? tText("graph.why.hop0") : tText("graph.why.hop", { n: h }), v, mx, i })));
  return el("div", {}, el("div", { class: "caps muted views-sub" }, t("graph.why.hops")), bars);
}

// The neighbours that weighed most (at graph level, the graph's nodes); a row clicked opens its card.
function nearBars(a, name, pick) {
  const near = (a.neighbours || []).filter((n) => n.node !== a.target_node && finite(n.total))
    .sort((x, y) => Math.abs(y.total) - Math.abs(x.total)).slice(0, K_NEAR);
  if (!near.length) return null;
  const mx = Math.max(1e-9, ...near.map((n) => Math.abs(n.total))), bars = el("div", { class: "bars diverge" });
  near.forEach((n, i) => bars.append(bar({ name: name(n.node), v: n.total, mx, i, onclick: () => pick(n) })));
  return el("div", {}, el("div", { class: "caps muted views-sub" }, t(a.level === "graph" ? "graph.why.near_graph" : "graph.why.near", { k: near.length })), bars);
}

// What the drawing leaves out, each on a row of its own: the nodes past the drawn ones, the share of
// the nodes the answer did not list (`outside`), and the gap the method reports (A31: never folded
// into the attributions).
function restRows(a, hidden) {
  const out = [], tiny = 1e-9 * Math.max(1, Math.abs(a.total || 0)), c = a.completeness;
  if (hidden > 0) out.push(el("p", { class: "hint" }, tn("graph.why.hidden", hidden)));
  if (finite(a.outside) && Math.abs(a.outside) > tiny) out.push(el("p", { class: "hint graph-rest" }, t("graph.why.outside", { v: sg(a.outside, 3) })));
  if (c && finite(c.gap)) {
    out.push(el("p", { class: "hint graph-rest graph-gap" }, t("graph.why.gap", { gap: plain(c.gap, 4), sum: sg(c.sum ?? 0, 3), change: sg(c.target_minus_baseline ?? 0, 3) })));
  }
  return out.length ? el("div", {}, ...out) : null;
}

export function attribution(ex, env = {}) {
  const a = graphAttribution(ex);
  if (!a) return el("p", { class: "hint" }, t("graph.why.none"));
  const st = edState(), G = st.G || graphOf(env.schema, env), name = namer(ex, st.rows || [], G), host = el("div", { class: "graph-pick-host" });
  const pick = (n) => host.replaceChildren(nodeCard(n, a, name));
  const D = drawnOf(a), lead = t(a.level === "graph" ? "graph.why.lead_graph" : "graph.why.lead", { name: explainedName(ex) });
  return el("div", {}, el("p", { class: "hint views-why-lead" }, lead), methodNote(a, { gap: false }),
    el("div", { class: "graph-why" }, neighbourhoodSvg(a, a.level, { name, pick }), host),
    selfBars(a, env), hopBars(a), nearBars(a, name, pick), restRows(a, D.hidden));
}

// ---- the stepper's input step ------------------------------------------------------------------------
function raw(rawRow, col) {
  if (!rawRow || !has(rawRow, col)) return null;
  const v = rawRow[col];
  return v === null || v === "" ? tText("shape.row.input.empty") : typeof v === "number" ? fmtNum(v) : data(v);
}

// The largest shaped cells: [{name, x, raw}] as signed bars.
function topCells(cells) {
  const top = cells.slice().sort((p, q) => Math.abs(q.x) - Math.abs(p.x)).slice(0, K_TOP), mx = Math.max(1e-9, ...top.map((c) => Math.abs(c.x)));
  const bars = el("div", { class: "bars diverge eq" });
  top.forEach((c, k) => bars.append(termRow({ name: c.name, v: c.x, max: mx, i: k, val: [c.raw !== null ? el("span", { class: "wx" }, `${c.raw}  → `) : "", tText("shape.row.input.x", { v: sg(c.x) })] })));
  return { bars, top };
}

// How many neighbours each hop brought, and where the edges come from. At graph level the answer
// reads one whole graph: no node is read alone, and the edges join the rows of that graph, so every
// one of its rows can change the answer and a row of another graph none.
// The stepper's env carries no schema (D4 §6.2 leaves it to the view): the editor's, else the page's.
function neighbourLine(ex, env) {
  const e = env || {}, schema = e.schema || edState().schema || (S.env && S.env.schema) || null;
  const a = graphAttribution(ex), hops = new Map(), G = graphOf(schema, e);
  if (G.level === "graph") {
    return el("div", {}, el("p", { class: "hint" }, t("graph.input.whole_graph")),
      G.links.length ? el("p", { class: "hint" }, t("graph.input.edges_graph", { columns: data(G.links.join(", ")) })) : null);
  }
  ((a && a.neighbours) || []).filter((n) => n.hop > 0).forEach((n) => hops.set(n.hop, (hops.get(n.hop) || 0) + 1));
  const per = [...hops].sort((p, q) => p[0] - q[0]).map(([h, n]) => tnText("graph.input.hop", n, { h }));
  return el("div", {}, per.length ? el("p", { class: "hint" }, t("graph.input.hops", { hops: joinList(per) })) : el("p", { class: "hint" }, t("graph.input.alone")),
    G.links.length ? el("p", { class: "hint" }, t("graph.input.edges", { columns: data(G.links.join(", ")) })) : null);
}

export function inputStep(ex, row, env = {}) {
  const i = (ex && ex.inputs) || {}, names = i.order || [], x = i.shaped, rawRow = i.raw || row || null;
  if (Array.isArray(x) && Array.isArray(x[0])) {
    const cells = x.flatMap((r, k) => r.map((v, f) => ({ name: tText("graph.input.cell", { feature: data(names[f] ?? f), k: k + 1 }), x: v, raw: null })));
    const { bars } = topCells(cells);
    return el("div", {}, el("p", {}, t("graph.input.lead_graph", { nodes: x.length, f: names.length })),
      el("div", { class: "caps muted views-sub" }, t("shape.row.input.top", { k: Math.min(K_TOP, cells.length), zero: 0 })), bars, neighbourLine(ex, env));
  }
  const v = Array.isArray(x) ? x : [], cells = v.map((val, k) => ({ name: data(names[k] ?? k), x: val, raw: raw(rawRow, names[k]) }));
  const { bars, top } = topCells(cells), rest = cells.filter((c) => !top.includes(c)).map((c) => c.x);
  return el("div", {}, el("p", {}, t("graph.input.lead", { n: v.length })),
    el("div", { class: "caps muted views-sub" }, t("shape.row.input.top", { k: top.length, zero: 0 })), bars,
    rest.length ? el("p", { class: "hint" }, t("shape.row.input.rest", { n: rest.length, lo: plain(Math.min(...rest)), hi: plain(Math.max(...rest)) })) : null,
    neighbourLine(ex, env));
}

export const drawInput = Object.freeze({
  measure: inputRenderer.measure, layout: inputRenderer.layout, draw: inputRenderer.draw,
  hit: inputRenderer.hit, caption: inputRenderer.caption, acts: inputRenderer.acts,
});
