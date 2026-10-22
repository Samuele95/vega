// "The weights, read" (D4 §18: the hand-built page's index.html:1471–1642). Single weights mean
// little on their own (units can be permuted, scales traded between layers); two readings do mean
// something, and a third shows the raw material:
//   the map, on real rows   each column takes the real rows of one output from the training demo
//                           sample (vega/attribution@2, or the v2.2 examples' attribution) and asks
//                           what pushed that output's own score up or down, read by the method the
//                           record names (gradient × input: local and first-order, not additive)
//   the map, linear         straight through the product of the weight matrices (a pure dense
//                           chain only): what the weights say, not what a row does
//   layer by layer          every weight of every layer as a heat map, ordered so its structure
//                           shows (a pure chain); on any other graph every dense stage's matrix
//                           in its own order
// The columns: on real rows the lane's weights axis when the caller gives it (ctx.columns, the task
// module's weightsAxis, W08: [{id, name}], each read from the attribution's cell of that id), else the
// head's outputs read by name; the linear reading and the layers always the head's outputs.
import { $, el, css } from "../core/dom.js";
import { fmtInt } from "../core/fmt.js";
import { tip, shade } from "../core/bars.js";
import { t, tText, tnText, data } from "../core/text.js";
import { effective } from "./card.js";

const verb = (v) => tText(v >= 0 ? "legacy.net.verb.raises" : "legacy.net.verb.lowers");
const norm2 = (arr) => Math.sqrt(arr.reduce((a, v) => a + v * v, 0));
const MIXED = .7;
// The attribution methods whose reading the lead names (the demo sample's vega/attribution@2
// method.name): only a method whose terms add up to the score is ever called exact.
const METHOD_LEAD = Object.freeze(["grad_x_input", "integrated_gradients"]);

// The demo sample's attribution, in one form: {inputs, columns: [{id, name}], cells: {id: {n, rows}}}.
export function attributionOf(doc) {
  if (!doc) return null;
  if (doc.schema === "vega/attribution@2") {
    return { inputs: doc.inputs, columns: doc.axis.columns.map((c) => ({ id: c.id, name: c.name })), cells: doc.cells,
             method: doc.method && typeof doc.method.name === "string" ? doc.method.name : typeof doc.method === "string" ? doc.method : null };
  }
  if (!doc.by_class) return null;
  const columns = (doc.classes || Object.keys(doc.by_class)).map((c) => ({ id: c, name: c }));
  return { inputs: doc.inputs, columns, cells: doc.by_class, method: null };
}

// The map's columns, [{id, name}]: see the header.
function columnsOf(mode, ctx) {
  if (mode === "real" && Array.isArray(ctx.columns) && ctx.columns.length) return ctx.columns;
  return ctx.graph.headLabels().map((name) => ({ id: null, name }));
}

// ctx: {graph, model (a pure chain's run model, or null), families, attribution (attributionOf),
// columns (the lane's weights axis, or null), palette}
function reading(mode, ctx, cols) {
  const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;
  const agreeOf = (xs, m) => xs.filter((v) => Math.sign(v) === Math.sign(m) && v !== 0).length;
  if (mode === "real") {
    const A = ctx.attribution, col = new Map(ctx.graph.inputs.names.map((n) => [n, A.inputs.indexOf(n)]));
    const byName = A.columns.reduce((m, c) => m.set(c.name, c.id), new Map());
    const cell = (c) => A.cells[c.id ?? byName.get(c.name)] || null;
    const rows = cols.map((c) => (cell(c) || { rows: [] }).rows);
    const stat = (xs) => (xs.length ? { m: mean(xs), agree: agreeOf(xs, mean(xs)), n: xs.length } : null);
    const at = (r, n) => (col.get(n) >= 0 ? r[col.get(n)] : 0);
    return { counts: cols.map((c) => (cell(c) || { n: 0 }).n), lead: `${tText("legacy.net.map.real_lead")} ${tText(METHOD_LEAD.includes(A.method) ? `net.map.method.${A.method}` : "net.map.method.unknown")}`,
      cellOf: (f, k) => stat(rows[k].map((r) => f.members.reduce((s, n) => s + at(r, n), 0))),
      subOf: (n, k) => stat(rows[k].map((r) => at(r, n))) };
  }
  const E = effective(ctx.model), M = E.toIn[E.L - 1], val = (c, i) => M.w[c * M.in + i], unit = (n) => ctx.model.inputs.indexOf(n);
  return { counts: null, lead: tText("legacy.net.map.linear_lead"),
    cellOf: (f, k) => ({ m: f.idx.reduce((s, i) => s + val(k, i), 0) }), subOf: (n, k) => ({ m: val(k, unit(n)) }) };
}

function cellNode({ st, row, name, sub, mx }, mode, palette) {
  if (!st) return el("div", { class: "c none " + (sub ? "sub" : ""), title: tText("legacy.net.map.none") });
  const mixed = mode === "real" && st.n && st.agree / st.n < MIXED;
  const d = el("div", { class: "c " + (sub ? "sub " : "") + (mixed ? "mixed" : ""), style: `background-color:${shade(st.m, mx, palette)}` });
  const say = mode === "real"
    ? [el("div", {}, `${row} → ${data(name)}`), el("div", { class: "v " + (st.m >= 0 ? "acc" : "") }, t("legacy.net.map.tip_real", { n: st.n, name: data(name), verb: verb(st.m), v: Math.abs(st.m).toPrecision(3) })),
      el("div", { class: "v muted" }, t("legacy.net.map.tip_agree", { agree: st.agree, n: st.n }), mixed ? " " : "", mixed ? t("legacy.net.map.tip_mixed") : "")]
    : [el("div", {}, `${row} → ${data(name)}`), el("div", { class: "v " + (st.m >= 0 ? "acc" : "") }, t("legacy.net.map.tip_linear", { verb: verb(st.m), v: (st.m >= 0 ? "+" : "−") + Math.abs(st.m).toPrecision(3) }))];
  d.addEventListener("mousemove", (e) => tip(e, say));
  d.addEventListener("mouseleave", (e) => tip(e, null));
  return d;
}

function familyRows(grid, { fam, names, famCells, subCells, famMx, subMx }, mode, palette) {
  fam.forEach((f, gi) => {
    const strength = (n) => Math.max(0, ...subCells.get(n).filter(Boolean).map((c) => Math.abs(c.m)));
    const members = f.members.slice().sort((a, b) => strength(b) - strength(a)), subs = [];
    const head = el("button", { class: "rh", title: f.text || f.label }, `${f.label} ${members.length > 1 ? "›" : ""}`);
    head.onclick = () => { if (!subs.length) return; const open = subs[0].hidden; subs.forEach((n) => { n.hidden = !open; }); head.textContent = `${f.label} ${open ? "⌄" : "›"}`; };
    grid.append(head, ...names.map((c, k) => cellNode({ st: famCells[gi][k], row: f.label, name: c, sub: false, mx: famMx }, mode, palette)));
    if (members.length > 1) members.forEach((n) => {
      const nodes = [el("div", { class: "rh sub", title: data(n) }, data(n)), ...names.map((c, k) => cellNode({ st: subCells.get(n)[k], row: data(n), name: c, sub: true, mx: subMx }, mode, palette))];
      nodes.forEach((x) => { x.hidden = true; subs.push(x); grid.append(x); });
    });
  });
}

function renderMap(host, mode, ctx) {
  const fam = ctx.families, cols = columnsOf(mode, ctx), names = cols.map((c) => c.name), K = names.length, R = reading(mode, ctx, cols);
  const famCells = fam.map((f) => names.map((_, k) => R.cellOf(f, k)));
  const famMx = Math.max(1e-12, ...famCells.flat().filter(Boolean).map((c) => Math.abs(c.m)));
  const subCells = new Map(); fam.forEach((f) => f.members.forEach((n) => subCells.set(n, names.map((_, k) => R.subOf(n, k)))));
  const subMx = Math.max(1e-12, ...[...subCells.values()].flat().filter(Boolean).map((c) => Math.abs(c.m)));
  host.append(el("p", { class: "hint net-map-lead" }, R.lead));
  const grid = el("div", { class: "wmap", style: `grid-template-columns:minmax(150px,230px) repeat(${K},minmax(20px,1fr))` });
  const neutral = (c) => !!ctx.roles && ctx.roles.isNeutral(c);
  grid.append(el("div"), ...names.map((c, k) => el("div", { class: "ch " + (neutral(c) ? "role-neutral" : ""), title: R.counts ? tnText("legacy.net.map.col_title", R.counts[k], { name: data(c) }) : data(c) }, data(c))));
  familyRows(grid, { fam, names, famCells, subCells, famMx, subMx }, mode, ctx.palette);
  const P = ctx.palette;
  host.append(grid, el("div", { class: "wscale" }, verb(-1), el("i", { style: `background:linear-gradient(90deg,${shade(-1, 1, P)},${shade(-.05, 1, P)},${shade(.05, 1, P)},${shade(1, 1, P)})` }), verb(1),
    mode === "real" ? el("span", { class: "net-map-scale" }, t("legacy.net.map.scale_real", { counts: [...new Set(R.counts)].sort((a, b) => b - a).join(", ") })) : null));
}

// The map with its reading switch: on real rows when the sample's attribution is there, the linear
// reading when the network is a pure dense chain.
export function drawWeightMap(host, ctx) {
  const modes = el("div", { class: "wmodes caps" }), body = el("div");
  const real = !!ctx.attribution, linear = !!ctx.model;
  if (!real && !linear) { host.append(el("p", { class: "hint" }, t("weights.map.unavailable"))); return; }
  const pick = (m) => { [...modes.children].forEach((b) => b.classList.toggle("on", b.dataset.m === m)); body.replaceChildren(); renderMap(body, m, ctx); };
  if (real) modes.append(Object.assign(el("button", { "data-m": "real" }, t("legacy.net.map.real")), { onclick: () => pick("real") }));
  if (linear) modes.append(Object.assign(el("button", { "data-m": "linear" }, t("legacy.net.map.linear")), { onclick: () => pick("linear") }));
  host.append(modes, body);
  pick(real ? "real" : "linear");
}

// ---- layer by layer
function matrixFigure(m, { rowSpec, colSpec, caption, palette }) {
  const R = rowSpec.order.length, C = colSpec.order.length;
  const cw = Math.max(1, Math.min(6, Math.floor(560 / C))), rh = Math.max(1, Math.min(14, Math.floor(420 / R)));
  const top = 20, right = 200, W = C * cw, H = R * rh, cv = document.createElement("canvas"), d = devicePixelRatio || 1;
  cv.width = (W + right) * d; cv.height = (H + top) * d; cv.style.width = (W + right) + "px"; cv.style.height = (H + top) + "px";
  const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0);
  let mx = 0; for (let k = 0; k < m.w.length; k++) mx = Math.max(mx, Math.abs(m.w[k]));
  rowSpec.order.forEach((r, y) => colSpec.order.forEach((c, x) => { g.fillStyle = shade(m.w[r * m.in + c], mx, palette); g.fillRect(x * cw, top + y * rh, cw, rh); }));
  g.font = `11px ${css("--sans")}`; g.textBaseline = "middle"; g.strokeStyle = css("--line-hi"); g.lineWidth = 1;
  colSpec.bands.forEach((b) => {                         // bands: family or class groups, labelled where there is room
    const x0 = b.start * cw, x1 = b.end * cw;
    g.beginPath(); g.moveTo(x0 + .5, top - 5); g.lineTo(x0 + .5, top + H); g.stroke();
    g.fillStyle = css("--fg2"); if (g.measureText(b.label).width < x1 - x0 - 4) { g.textAlign = "center"; g.fillText(b.label, (x0 + x1) / 2, top - 9); }
  });
  rowSpec.bands.forEach((b) => {
    const y0 = top + b.start * rh, y1 = top + b.end * rh;
    g.beginPath(); g.moveTo(0, y0 + .5); g.lineTo(W + 6, y0 + .5); g.stroke();
    if (y1 - y0 >= 11) { g.fillStyle = css("--fg2"); g.textAlign = "left"; g.fillText(b.label, W + 10, (y0 + y1) / 2); }
  });
  cv.addEventListener("mousemove", (e) => {
    const rc = cv.getBoundingClientRect(), x = Math.floor((e.clientX - rc.left) / cw), y = Math.floor((e.clientY - rc.top - top) / rh);
    if (x < 0 || x >= C || y < 0 || y >= R) { tip(e, null); return; }
    const r = rowSpec.order[y], c = colSpec.order[x], v = m.w[r * m.in + c];
    tip(e, [el("div", {}, `${rowSpec.name(r)} ← ${colSpec.name(c)}`), el("div", { class: "v " + (v >= 0 ? "acc" : "") }, t("legacy.net.layers.tip", { w: (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4), verb: verb(v) }))]);
  });
  cv.addEventListener("mouseleave", (e) => tip(e, null));
  return el("figure", { class: "net-matrix" }, el("div", { class: "caps muted net-matrix-cap" }, caption), cv);
}

const bandsOf = (order, key, label) => {
  const out = [];
  order.forEach((j, pos) => { const k = key(j); if (!out.length || out[out.length - 1].k !== k) out.push({ k, label: label(k), start: pos, end: pos + 1 }); else out[out.length - 1].end = pos + 1; });
  return out;
};
const orderBy = (n, key, strength) => Array.from({ length: n }, (_, j) => j).sort((a, b) => key(a) - key(b) || strength(b) - strength(a));

// the class each hidden unit pushes most, read through the rest of the network
function topClass(E, h) {
  const m = E.toOut[h];
  return Array.from({ length: m.in }, (_, j) => { let b = 0, bv = -Infinity; for (let c = 0; c < m.out; c++) { const v = m.w[c * m.in + j]; if (v > bv) { bv = v; b = c; } } return b; });
}

// hidden units ordered: the first layer by the family each listens to most, later ones by the output each pushes most
function unitOrders(ctx) {
  const E = effective(ctx.model), mats = ctx.model.layers.map((l) => l.W), fam = ctx.families, classes = ctx.graph.headLabels();
  const rows = [], cols = [];
  for (let h = 1; h < E.L; h++) {
    const tc = topClass(E, h), st = (j) => Math.abs(E.toOut[h].w[tc[j] * E.toOut[h].in + j]);
    if (h === 1) {
      const m = mats[0], dom = Array.from({ length: m.out }, (_, j) => { let b = 0, bv = -1; fam.forEach((f, gi) => { const v = Math.sqrt(f.idx.reduce((s, i) => s + m.w[j * m.in + i] ** 2, 0) / f.idx.length); if (v > bv) { bv = v; b = gi; } }); return b; });
      const o = orderBy(m.out, (j) => dom[j], (j) => norm2(Array.from(m.w.subarray(j * m.in, (j + 1) * m.in))));
      rows[h] = { rows: o, bands: bandsOf(o, (j) => dom[j], (k) => tText("legacy.net.layers.listen", { name: fam[k].label })) };
    }
    const o = orderBy(E.toOut[h].in, (j) => tc[j], st);
    cols[h] = { cols: o, bands: bandsOf(o, (j) => tc[j], (k) => data(classes[k])) };
  }
  return { rows, cols };
}

function layerSpecs(ctx, l, U) {
  const L = ctx.model.layers.length, classes = ctx.graph.headLabels(), fam = ctx.families;
  const push = (b) => ({ ...b, label: tText("legacy.net.layers.push", { name: b.label }) });
  const unit = (r, layer) => tText("legacy.net.layers.unit", { r: r + 1, l: layer });
  const inOrder = fam.flatMap((f) => f.idx), inBands = [];
  let at = 0; fam.forEach((f) => { inBands.push({ label: f.label, start: at, end: at + f.idx.length }); at += f.idx.length; });
  const rowSpec = l === L - 1 ? { order: classes.map((_, c) => c), bands: classes.map((c, k) => ({ label: data(c), start: k, end: k + 1 })), name: (r) => data(classes[r]) }
    : l === 0 ? { order: U.rows[1].rows, bands: U.rows[1].bands, name: (r) => unit(r, 1) }
      : { order: U.cols[l + 1].cols, bands: U.cols[l + 1].bands.map(push), name: (r) => unit(r, l + 1) };
  const colSpec = l === 0 ? { order: inOrder, bands: inBands, name: (c) => data(ctx.model.inputs[c]) }
    : { order: l === 1 ? U.rows[1].rows : U.cols[l].cols, bands: l === 1 ? U.rows[1].bands : U.cols[l].bands.map(push), name: (c) => unit(c, l) };
  return { rowSpec, colSpec };
}

// A pure chain's layers, ordered so their structure shows.
function chainLayers(host, ctx) {
  const U = unitOrders(ctx), L = ctx.model.layers.length;
  host.append(el("p", { class: "hint" }, t("legacy.net.layers.lead")));
  ctx.model.layers.forEach((layer, l) => {
    const what = l === 0 ? tText("legacy.net.layers.first") : l === L - 1 ? tText("legacy.net.layers.last") : tText("legacy.net.layers.middle", { a: l, b: l + 1 });
    const caption = t("legacy.net.layers.caption", { l: l + 1, what, n: fmtInt(layer.params) });
    host.append(matrixFigure(layer.W, { ...layerSpecs(ctx, l, U), caption, palette: ctx.palette }));
  });
}

// Any other graph: every dense-like stage's matrix in its own order.
function stageLayers(host, ctx) {
  host.append(el("p", { class: "hint" }, t("weights.stages.lead")));
  const graph = ctx.graph, ids = [...graph.denseRuns().flatMap((r) => r.stages), ...(graph.head() ? [graph.head().id] : [])];
  ids.forEach((id) => {
    const s = graph.byId(id), W = graph.matrix(s.weight);
    if (!W) return;
    const nat = (n) => ({ order: Array.from({ length: n }, (_, i) => i), bands: [] });
    const rowSpec = { ...nat(W.out), name: (r) => tText("weights.stages.row", { r: r + 1 }) }, colSpec = { ...nat(W.in), name: (c) => tText("weights.stages.col", { c: c + 1 }) };
    const caption = t("weights.stages.caption", { name: data(id), from: W.in, to: W.out, n: fmtInt(s.params) });
    host.append(matrixFigure(W, { rowSpec, colSpec, caption, palette: ctx.palette }));
  });
}

export function drawLayerMatrices(host, ctx) {
  if (ctx.model) chainLayers(host, ctx); else stageLayers(host, ctx);
}

// The disclosure's two panes (#w-map, #w-layers) and their switch (#w-switch). ctx as reading();
// ctx.columns is optional (the head's outputs without it).
export function drawWeights(ctx) {
  const palette = { pos: css("--accent"), neg: css("--neg") }, c = { ...ctx, palette };
  if (ctx.graph.weightsWithheld()) {
    if (!$("w-map").childElementCount) $("w-map").append(el("p", { class: "hint" }, t("weights.withheld")));
  } else if (!$("w-map").childElementCount) drawWeightMap($("w-map"), c);
  document.querySelectorAll("#w-switch button").forEach((b) => { b.onclick = () => {
    document.querySelectorAll("#w-switch button").forEach((x) => x.classList.toggle("on", x === b));
    $("w-map").hidden = b.dataset.w !== "map"; $("w-layers").hidden = b.dataset.w !== "layers";
    if (b.dataset.w === "layers" && !$("w-layers").childElementCount) drawLayerMatrices($("w-layers"), c);
  }; });
}
