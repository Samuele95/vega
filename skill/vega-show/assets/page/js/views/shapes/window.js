// The window shape (D4 §6.5): a sample is the last `length` rows of one series, read with the rows
// before them as context (R-A5: every row is a window end, answered when length − 1 rows of its
// series precede it in history ++ rows).
//   editor        the series as a table, one row per step and one field per raw input column of
//                 /v1/schema, the last `length` rows marked as the window, the rows before them as
//                 context; above it one sparkline per numeric column. Editing a cell asks again. The
//                 request is {rows: [the last row], history: [the rows before it]}.
//   sources       many at once: the demonstration sample (each item a window and its context; one
//                 window per series a run, since a request holds one window end per series beside
//                 its history — runs go round the sample), the shipped test rows (consecutive: every
//                 step with a full window answered), the reader's CSV (consecutive rows; the service
//                 builds the windows). A row of their table opens in try one with its context.
//   attribution   what tipped the answer: a step × feature heat of gradient × input (nominal inputs
//                 by their embedding), the totals per family of inputs, the totals per step
//   inputStep     the stepper's first step: the shaped window as a heat, and the cells furthest from
//                 zero named <feature> @ <step>
//   drawInput     the input segment of the network drawing (net/renderers/input.js draws a window's
//                 families with the steps they span)
import { el } from "../../core/dom.js";
import { S } from "../../core/state.js";
import { URLS, api } from "../../core/api.js";
import { plain, sg } from "../../core/fmt.js";
import { termRow } from "../../core/bars.js";
import { t, tText, tn, data } from "../../core/text.js";
import { familiesOf } from "../../net/families.js";
import { colours } from "../../net/canvas.js";
import inputRenderer from "../../net/renderers/input.js";
import { heatMatrix, matrix, at, transpose, paintCells, paintBars, finite, downsample } from "../../net/seq.js";
import { viewState, methodNote } from "../shared.js";
import { testOf, explainedName } from "../payloads.js";
import { cellOf } from "./row.js";

const NUMERIC = Object.freeze(["float", "int", "integer", "number"]);
const K_TOP = 6, ROW_H = 14, HEAT_STEP_W = 14, SPARK_W = 132, SPARK_H = 26;
const columnsOf = (schema) => (schema && Array.isArray(schema.input) ? schema.input : []);
const editorState = () => viewState("window-editor", { host: null, schema: null, env: null, rows: [], table: null, sparks: null });
const sourceState = () => viewState("window-sources", { round: 0, batch: null, contexts: null });
const sampleState = () => viewState("window-sample", { sample: null, asked: false });

// The bundle's sample record (MANIFEST.sample through /v1/info: length, series_by, history_rows),
// asked once; null until it arrives (or when the service has no /v1/info).
export function sampleRecord() {
  const st = sampleState();
  if (!st.asked) {
    st.asked = true;
    Promise.resolve().then(() => api(URLS.info)).then((info) => { st.sample = info && info.sample ? info.sample : null; }, () => { st.sample = null; });
  }
  return st.sample;
}

// The window's length: the network's sample, else the sample record's.
export function windowLength(env) {
  const g = env && env.graph, s = sampleState().sample;
  return (g && g.sample && g.sample.length) || (s && s.length) || 0;
}

// ---- the editor
// The rows of a preset: its history then its rows; a bare row (one row of many at once) with the
// context it was sent with, when it came from this page's last batch: its own window's history, or
// the rows before it in history ++ rows — those of its series (when the sample record names the
// series column), the last `history_rows` of them (when it states how many a window end needs).
export function rowsOf(preset) {
  if (!preset) return [];
  if (Array.isArray(preset.rows) && preset.rows.length) return [...(preset.history || []), ...preset.rows];
  if (!preset.row) return [];
  const st = sourceState(), b = st.batch, i = b ? b.rows.indexOf(preset.row) : -1;
  if (i < 0) return [preset.row];
  if (st.contexts && st.contexts[i]) return [...st.contexts[i], preset.row];
  const record = sampleState().sample, by = record && record.series_by, need = record && Number(record.history_rows);
  const before = [...(b.history || []), ...b.rows.slice(0, i)].filter((r) => !by || String(r[by]) === String(preset.row[by]));
  return [...(need > 0 ? before.slice(-need) : before), preset.row];
}

// A step's name in the window: the network's step labels (t−k … t) for the last `length` rows, else
// its position in the window; a row before the window is context.
function stepName(env, k, n, L) {
  const steps = env && env.graph ? env.graph.inputs.steps : null, back = n - 1 - k;
  if (back >= L) return tText("shape.window.context");
  return steps && steps.length === L ? data(steps[L - 1 - back]) : String(L - back);
}

function inputCell(col, value) {
  const input = el("input", { "data-col": col.name, "data-type": col.dtype || "" });
  input.value = value === null || value === undefined ? "" : String(value);
  return el("td", {}, input);
}

function renderTable(st) {
  const cols = columnsOf(st.schema), rows = st.rows, L = windowLength(st.env) || rows.length;
  const head = el("tr", {}, el("th", {}), ...cols.map((c) => el("th", {}, data(c.name))));
  const body = rows.map((row, k) => el("tr", { class: rows.length - k <= L ? "seq-in" : "seq-ctx" },
    el("th", {}, stepName(st.env, k, rows.length, L)), ...cols.map((c) => inputCell(c, row[c.name]))));
  st.table = el("table", { class: "seq-table" }, el("thead", {}, head), el("tbody", {}, ...body));
  st.sparks = el("div", { class: "seq-sparks" });
  const lead = rows.length ? t("shape.window.editor.lead", { n: Math.min(L, rows.length) }) : t("shape.window.editor.empty");
  st.box.replaceChildren(el("p", { class: "hint seq-lead" }, lead), st.sparks, el("div", { class: "seq-scroll" }, st.table));
  drawSparks(st);
}

// The table's rows read back, each cell by its column's type (row.js cellOf).
function readTable(st) {
  if (!st.table) return [];
  const body = st.table.querySelector("tbody");
  return [...(body ? body.children : [])].map((tr) => {
    const row = {};
    for (const input of tr.querySelectorAll("input")) row[input.getAttribute("data-col")] = cellOf(input.value, input.getAttribute("data-type"));
    return row;
  });
}

export const editor = Object.freeze({
  mount(host, schema, env) {
    const st = editorState();
    Object.assign(st, { host, schema, env, rows: [], box: el("div", { class: "seq-editor" }) });
    host.replaceChildren(st.box);
    if (typeof host.addEventListener === "function") host.addEventListener("input", () => drawSparks(st));
    sampleRecord();
    renderTable(st);
  },
  load(preset, env) {
    const st = editorState();
    if (!st.host) return;
    if (env) st.env = env;
    st.rows = rowsOf(preset);
    renderTable(st);
  },
  read() {
    const rows = readTable(editorState());
    return { rows: rows.slice(-1), history: rows.length > 1 ? rows.slice(0, -1) : null };
  },
  describe(preset) {
    if (!preset) return { title: "", text: "" };
    return { title: preset.name, text: preset.text };
  },
});

// ---- canvases (a page canvas; nothing is drawn where the document has none, as under node)
function canvasOf(w, h, cls) {
  const c = el("canvas", { class: cls, width: String(w), height: String(h), style: `width:${w}px;height:${h}px` });
  const g = typeof c.getContext === "function" ? c.getContext("2d") : null;
  if (!g) return { c, g: null };
  const d = (typeof devicePixelRatio === "number" && devicePixelRatio) || 1;
  c.width = w * d; c.height = h * d; g.setTransform(d, 0, 0, d, 0, 0);
  return { c, g };
}

// One sparkline per numeric column over the table's rows, the window's span shaded.
function drawSparks(st) {
  if (!st.sparks) return;
  const rows = readTable(st), cols = columnsOf(st.schema).filter((c) => NUMERIC.includes(c.dtype)), L = windowLength(st.env) || rows.length;
  st.sparks.replaceChildren(...cols.map((col) => {
    const values = rows.map((r) => (typeof r[col.name] === "number" ? r[col.name] : null)), { c, g } = canvasOf(SPARK_W, SPARK_H, "seq-spark");
    if (g) sparkline(g, values, Math.min(L, values.length));
    return el("div", { class: "seq-spark-box" }, el("span", { class: "muted" }, data(col.name)), c);
  }));
}

function sparkline(g, values, inWindow) {
  const C = colours(), ok = values.filter(finite), n = values.length;
  if (!ok.length || n < 2) return;
  const lo = Math.min(...ok), hi = Math.max(...ok), span = hi - lo || 1, x = (k) => 2 + k * (SPARK_W - 4) / (n - 1), y = (v) => SPARK_H - 3 - (v - lo) / span * (SPARK_H - 6);
  g.fillStyle = C.line; g.globalAlpha = .5; g.fillRect(x(n - inWindow), 0, SPARK_W - x(n - inWindow), SPARK_H); g.globalAlpha = 1;
  g.strokeStyle = C.acc; g.lineWidth = 1.2; g.beginPath();
  values.forEach((v, k) => { if (!finite(v)) return; if (k === 0 || !finite(values[k - 1])) g.moveTo(x(k), y(v)); else g.lineTo(x(k), y(v)); });
  g.stroke();
}

// ---- many at once
// The demonstration sample's windows: [{row, history, truth, series}] (series: the series_by
// column's value, null when the bundle has one series or the record is not known yet).
export function demoWindows(examples, seriesBy) {
  const items = examples && examples.demo && Array.isArray(examples.demo.items) ? examples.demo.items : [];
  return items.filter((it) => Array.isArray(it.rows) && it.rows.length).map((it) => {
    const row = it.rows[it.rows.length - 1];
    return { row, history: [...(it.history || []), ...it.rows.slice(0, -1)], truth: it.truth ?? null,
             series: seriesBy ? String(row[seriesBy]) : null };
  });
}

// The windows of one run: the round-th window of each series (one window end per series and request:
// the rows of a request are the windows' ends, their series' history the context). Without the
// sample record every window is a series of its own (a run of one).
export function roundOf(windows, round, seriesBy) {
  const groups = new Map();
  windows.forEach((w, k) => { const key = seriesBy ? w.series : `#${k}`; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(w); });
  const lists = [...groups.values()], depth = Math.max(1, ...lists.map((l) => l.length)), r = round % depth;
  return seriesBy ? lists.map((l) => l[r % l.length]) : [windows[round % Math.max(1, windows.length)]].filter(Boolean);
}

// The batch of a run: {rows, history, truth}, remembered with each row's context for try one. The
// truth travels only when every window's is one value (a class, a label set, a number): many at once
// hands it to the drift monitor as well, which reads one truth per row in the truth column's form, so
// a forecast's truth — the values of the steps ahead — stays out (its windows are answered without it).
export function batchOf(chosen) {
  const truth = chosen.map((w) => w.truth), one = truth.every((v) => v !== null && v !== undefined && (!Array.isArray(v) || v.every((x) => typeof x === "string")));
  const st = sourceState(), batch = { rows: chosen.map((w) => w.row), history: chosen.flatMap((w) => w.history), truth: one ? truth : null };
  Object.assign(st, { batch, contexts: chosen.map((w) => w.history) });
  return batch;
}

function remember(batch) {
  Object.assign(sourceState(), { batch, contexts: null });
  return batch;
}

export function sources(examples) {
  const record = sampleRecord(), seriesBy = record ? record.series_by || null : null, test = testOf(examples);
  const windows = demoWindows(examples, seriesBy);
  const nextRun = () => {
    const now = sampleRecord(), by = now ? now.series_by || null : null, st = sourceState();
    const chosen = roundOf(demoWindows(examples, by), st.round, by);
    st.round += 1;
    return batchOf(chosen);
  };
  return [
    windows.length ? { id: "demo", title: t("many.src.demo"), sub: t("shape.window.src.demo_note"), run: nextRun } : null,
    test ? { id: "test", title: t("shape.window.src.test"), sub: t("shape.window.src.test_note"), run: () => remember(test) } : null,
    { id: "csv", file: true, title: t("many.src.csv"), sub: t("shape.window.src.csv_note"), run: (file) => ({ file }) },
  ].filter(Boolean);
}

// ---- what tipped the answer
// The window's attribution as {steps, names, M (steps × features, nominal inputs last), perStep,
// perFeature}, or null when the explanation carries no window attribution.
export function windowAttribution(ex) {
  const a = ex && ex.attribution;
  if (!a || a.shape !== "window" || !a.features) return null;
  const num = Array.isArray(a.matrix) ? heatMatrix({ values: a.matrix }) : heatMatrix(a.matrix);
  const nom = a.nominal && Array.isArray(a.nominal.attribution) ? a.nominal : null, L = num ? num.rows : (a.per_step || []).length;
  const names = [...a.features.names, ...(nom ? nom.names : [])], F = names.length, v = new Float32Array(L * F);
  for (let r = 0; r < L; r++) {
    a.features.names.forEach((_, c) => { v[r * F + c] = num ? at(num, r, c) : 0; });
    if (nom) nom.names.forEach((_, j) => { const row = [].concat(nom.attribution[j]); v[r * F + a.features.names.length + j] = Number(row[r]) || 0; });
  }
  const M = matrix(L, F, v), perFeature = names.map((_, c) => Array.from({ length: L }, (_, r) => at(M, r, c)).reduce((s, x) => s + x, 0));
  return { steps: a.step_labels || [], names, M, perStep: Array.isArray(a.per_step) ? a.per_step : null, perFeature };
}

// A heat of a (steps × features) Matrix, the features as rows (their names beside), the steps as
// columns (the first and the last named under).
function heatBlock(M, names, steps) {
  const T = transpose(M), shown = downsample(T, names.length, 64), w = Math.max(120, Math.min(560, shown.cols * HEAT_STEP_W)), h = shown.rows * ROW_H;
  const { c, g } = canvasOf(w, h, "seq-heat-canvas");
  if (g) paintCells(g, shown, { x: 0, y: 0, w, h }, { colors: colours() });
  const ends = steps.length ? el("div", { class: "seq-heat-steps muted" }, el("span", {}, data(steps[0])), el("span", {}, data(steps[steps.length - 1]))) : null;
  return el("div", { class: "seq-heat" }, el("div", { class: "seq-heat-names" }, ...names.map((n) => el("div", {}, data(n)))),
    el("div", {}, c, ends));
}

// The families of the window's inputs: the story's groups, or — when they do not cover the inputs the
// window reads (the drawing then says why) — each input a family of its own.
function familiesFor(names, env) {
  const units = names.map((_, i) => [i]), got = familiesOf({ groups: env && env.story && env.story.input_groups, names, units });
  return got.problems.length ? names.map((n) => ({ label: data(n), members: [n] })) : got.families;
}

function familyBars(wa, env) {
  const fam = familiesFor(wa.names, env);
  const by = new Map(wa.names.map((n, i) => [n, wa.perFeature[i]]));
  const rows = fam.map((f) => ({ f, v: f.members.reduce((s, m) => s + (by.get(m) || 0), 0) })).filter((r) => r.v !== 0)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const mx = Math.max(1e-12, ...rows.map((r) => Math.abs(r.v))), bars = el("div", { class: "bars diverge" });
  rows.forEach((r, i) => bars.append(termRow({ name: r.f.label, v: r.v, max: mx, i,
    val: [tText(r.v >= 0 ? "shape.window.why.raised" : "shape.window.why.lowered", { v: Math.abs(r.v).toPrecision(2) })] })));
  return bars;
}

function stepBars(perStep) {
  const w = Math.max(160, Math.min(560, perStep.length * HEAT_STEP_W)), { c, g } = canvasOf(w, 36, "seq-steps-canvas");
  if (g) paintBars(g, perStep, { x: 0, y: 0, w, h: 36 }, { colors: colours() });
  return c;
}

export function attribution(ex, env) {
  const wa = windowAttribution(ex);
  if (!wa) return el("p", { class: "hint" }, t("shape.window.why.lead", { name: explainedName(ex) }));
  return el("div", {}, el("p", { class: "hint views-why-lead" }, t("shape.window.why.lead", { name: explainedName(ex) })), methodNote(ex && ex.attribution),
    heatBlock(wa.M, wa.names, wa.steps),
    el("div", { class: "caps muted views-sub" }, t("shape.window.why.families")), familyBars(wa, env),
    wa.perStep ? el("div", { class: "caps muted views-sub" }, t("shape.window.why.steps")) : null, wa.perStep ? stepBars(wa.perStep) : null);
}

// ---- the stepper's input step
// The shaped window of an explanation as {names, M (steps × inputs)}, or null.
export function shapedWindow(ex) {
  const i = ex && ex.inputs, s = i ? i.shaped : null;
  if (!s) return null;
  const M = Array.isArray(s) ? (Array.isArray(s[0]) ? heatMatrix({ values: s }) : null) : s.form === "sequence" ? heatMatrix(s.heat) : heatMatrix(s);
  return M ? { names: (i.order || []).slice(0, M.cols), M } : null;
}

// The cells furthest from zero among the numeric inputs, [{feature, step, v}] at most k.
export function topCells(x, steps, nominal, k = K_TOP) {
  const cells = [];
  x.names.forEach((name, c) => {
    if (nominal.includes(name)) return;
    for (let r = 0; r < x.M.rows; r++) cells.push({ feature: name, step: steps[r] ?? String(r + 1), v: at(x.M, r, c) });
  });
  return cells.filter((q) => finite(q.v)).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, k);
}

export function inputStep(ex, row, env) {
  const x = shapedWindow(ex), graph = (env && env.graph) || S.graph;
  if (!x) return el("p", { class: "hint" }, t("shape.window.input.title", { n: 0 }));
  const nominal = graph ? graph.inputs.nominal.map((c) => c.name) : [], steps = (graph && graph.inputs.steps) || [];
  const keep = x.names.map((n, c) => [n, c]).filter(([n]) => !nominal.includes(n)), shown = keep.map(([n]) => n);
  const v = new Float32Array(x.M.rows * keep.length);
  for (let r = 0; r < x.M.rows; r++) keep.forEach(([, c], j) => { v[r * keep.length + j] = at(x.M, r, c); });
  const top = topCells(x, steps.length === x.M.rows ? steps : [], nominal), mx = Math.max(1e-9, ...top.map((q) => Math.abs(q.v)));
  const bars = el("div", { class: "bars diverge eq" });
  top.forEach((q, i) => bars.append(termRow({ name: t("shape.window.input.cell", { feature: data(q.feature), step: data(q.step) }), v: q.v, max: mx, i,
    val: [tText("shape.window.input.x", { v: sg(q.v) })] })));
  const noms = x.names.filter((n) => nominal.includes(n));
  return el("div", {}, el("div", { class: "caps muted views-sub" }, t("shape.window.input.title", { n: x.M.rows })),
    heatBlock(matrix(x.M.rows, keep.length, v), shown, steps.length === x.M.rows ? steps : []),
    el("div", { class: "caps muted views-sub" }, t("shape.window.input.top", { k: top.length, zero: plain(0, 0) })), bars,
    noms.length ? el("p", { class: "hint" }, tn("shape.window.input.nominal", noms.length, { names: data(noms.join(", ")) })) : null);
}

export const drawInput = Object.freeze({
  measure: inputRenderer.measure, layout: inputRenderer.layout, draw: inputRenderer.draw,
  hit: inputRenderer.hit, caption: inputRenderer.caption, acts: inputRenderer.acts,
});
