// What the sequence renderers (renderers/conv.js, recurrent.js, attention.js) and the window shape
// (views/shapes/window.js) share: the explanation's parts read as matrices, and the few ways they
// are painted on a canvas.
//
//   Matrix      {rows, cols, v} — row-major values (a Float32Array or an Array), the one form every
//               reader below answers in: a Heat of explanation@2 (D3 §6.3: {rows, cols, values}),
//               a reduced `sequence` form (its heat), an exact tensor of a `detail: "exact"` answer
//               (its shape and base64 values), a weight tensor of the network view
//   seqOut      one stage's output for the explained sample as (steps × channels): the exact tensor,
//               else the conv probe's `maps.heat`, else the reduced output's heat
//   attention   one attention stage's maps, one Matrix per head (query × key): the exact `alpha`,
//               else the probe's `weights` (an attention form of D3 §6.3)
//   paintCells  a Matrix as coloured cells in a rectangle: the sign picks the colour (the page's
//               accent raises, its negative lowers), the size the opacity, on a square-root scale
// A reader that finds nothing answers null: the renderer then draws its resting state and says so.
import { css, el } from "../core/dom.js";
import { t, tText, tn } from "../core/text.js";
import { tensorValues } from "./graph.js";
import { stageLabel } from "./card.js";

export const finite = (v) => typeof v === "number" && Number.isFinite(v);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// The payload of one stage in an explanation@2, or null (absent, or omitted by the budget).
export function exNode(ex, id) {
  const n = ex && ex.nodes && id ? ex.nodes[id] : null;
  return n && n.level !== "omitted" ? n : null;
}

// ---- matrices
export const matrix = (rows, cols, v) => ({ rows, cols, v });
export const at = (M, r, c) => M.v[r * M.cols + c];

// A Heat ({rows, cols, values: [[…]]}) as a Matrix.
export function heatMatrix(h) {
  if (!h || !Array.isArray(h.values) || !h.values.length) return null;
  const rows = h.values.length, cols = h.values[0].length, v = new Float32Array(rows * cols);
  h.values.forEach((row, r) => row.forEach((x, c) => { v[r * cols + c] = Number(x) || 0; }));
  return matrix(rows, cols, v);
}

// A reduced part (D3 §6.3) as a Matrix: a sequence form's heat, a vector as one row, a scalar as one cell.
export function reducedMatrix(red) {
  if (!red) return null;
  if (red.form === "sequence") return heatMatrix(red.heat);
  if (red.form === "vector" && Array.isArray(red.values)) return matrix(1, red.values.length, Float32Array.from(red.values));
  if (red.form === "scalar" && finite(red.value)) return matrix(1, 1, Float32Array.of(red.value));
  return null;
}

// A tensor payload (network view or exact explanation part) as a Matrix of its first axis by the
// product of the others; `lead` more leading axes are folded into the rows.
export function tensorMatrix(t, lead = 0) {
  const v = t ? tensorValues(t) : null;
  if (!v || !Array.isArray(t.shape) || !t.shape.length) return null;
  const k = Math.min(t.shape.length - 1, 1 + lead), rows = t.shape.slice(0, k).reduce((a, b) => a * b, 1);
  return matrix(rows, v.length / Math.max(1, rows), v);
}

// Rows [from, to) of a Matrix (a gate's block of a recurrent weight, a projection's slice).
export function rowBlock(M, from, to) {
  if (!M) return null;
  return matrix(to - from, M.cols, M.v.subarray ? M.v.subarray(from * M.cols, to * M.cols) : M.v.slice(from * M.cols, to * M.cols));
}

// The Matrix transposed (a steps × channels map drawn as channels × steps).
export function transpose(M) {
  const v = new Float32Array(M.rows * M.cols);
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) v[c * M.rows + r] = M.v[r * M.cols + c];
  return matrix(M.cols, M.rows, v);
}

// Block means down to at most maxR × maxC (a thumbnail's cells).
export function downsample(M, maxR, maxC) {
  if (!M || (M.rows <= maxR && M.cols <= maxC)) return M;
  const R = Math.min(M.rows, maxR), C = Math.min(M.cols, maxC), v = new Float32Array(R * C), n = new Float32Array(R * C);
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    const k = Math.floor(r * R / M.rows) * C + Math.floor(c * C / M.cols);
    v[k] += M.v[r * M.cols + c]; n[k] += 1;
  }
  for (let k = 0; k < v.length; k++) v[k] = n[k] ? v[k] / n[k] : 0;
  return matrix(R, C, v);
}

export function maxAbs(M) {
  let m = 0;
  if (M) for (let k = 0; k < M.v.length; k++) { const a = Math.abs(M.v[k]); if (a > m) m = a; }
  return m;
}

// The values of a list, each divided by the largest magnitude (null entries kept as null).
export function normalised(values) {
  const mx = Math.max(1e-12, ...values.filter(finite).map(Math.abs));
  return values.map((v) => (finite(v) ? v / mx : null));
}

// The l2 norm of every row of a Matrix.
export function rowNorms(M) {
  return Array.from({ length: M.rows }, (_, r) => { let s = 0; for (let c = 0; c < M.cols; c++) s += at(M, r, c) ** 2; return Math.sqrt(s); });
}

// ---- the explained sample, read as matrices
// One stage's output as (steps × channels) (null when the explanation carries none of it, or the
// output is not a sequence).
export function seqOut(ex, id) {
  const n = exNode(ex, id);
  if (!n) return null;
  if (n.exact && n.exact.out && Array.isArray(n.exact.out.shape) && n.exact.out.shape.length === 2) return tensorMatrix(n.exact.out);
  const maps = n.extra && n.extra.maps && heatMatrix(n.extra.maps.heat);
  if (maps) return maps;
  return n.out && n.out.form === "sequence" ? heatMatrix(n.out.heat) : null;
}

// One attention stage's maps, a Matrix (query × key) per head, or null.
export function attentionMaps(ex, id) {
  const n = exNode(ex, id);
  if (!n) return null;
  const alpha = n.exact && n.exact.alpha;
  if (alpha && Array.isArray(alpha.shape)) {
    const v = tensorValues(alpha), s = alpha.shape;
    if (v && s.length === 3) return Array.from({ length: s[0] }, (_, h) => matrix(s[1], s[2], v.subarray(h * s[1] * s[2], (h + 1) * s[1] * s[2])));
    if (v && s.length === 1) return [matrix(1, s[0], v)];
  }
  const w = n.extra && n.extra.weights;
  if (!w || w.form !== "attention" || !Array.isArray(w.maps)) return null;
  const maps = w.maps.map(heatMatrix);
  return maps.every(Boolean) ? maps : null;
}

// ---- painting (ctx: {colors} of the canvas; opts.mx the scale, else the Matrix's largest magnitude)
// Each cell of M in rect, the colour by the sign, the opacity by the size (floor: a faint cell stays
// visible); opts.floor draws cells that are zero at that opacity.
export function paintCells(g, M, rect, { colors, mx = null, floor = .05 } = {}) {
  if (!M || !M.rows || !M.cols) return;
  const scale = mx || maxAbs(M) || 1, cw = rect.w / M.cols, ch = rect.h / M.rows, alpha = g.globalAlpha;
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    const x = at(M, r, c);
    g.globalAlpha = alpha * (floor + (1 - floor) * Math.sqrt(clamp(Math.abs(x) / scale, 0, 1)));
    g.fillStyle = x >= 0 ? colors.acc : colors.neg;
    g.fillRect(rect.x + c * cw, rect.y + r * ch, Math.max(1, cw - (cw > 4 ? 1 : 0)), Math.max(1, ch - (ch > 4 ? 1 : 0)));
  }
  g.globalAlpha = alpha;
}

// Signed bars along a baseline at rect's middle (up raises, down lowers), one per value.
export function paintBars(g, values, rect, { colors, mx = null } = {}) {
  const n = values.length;
  if (!n) return;
  const scale = mx || Math.max(1e-12, ...values.filter(finite).map(Math.abs)), bw = rect.w / n, mid = rect.y + rect.h / 2;
  values.forEach((v, k) => {
    if (!finite(v) || v === 0) return;
    const h = Math.max(1, Math.abs(v) / scale * rect.h / 2);
    g.fillStyle = v >= 0 ? colors.acc : colors.neg;
    g.fillRect(rect.x + k * bw + bw * .15, v >= 0 ? mid - h : mid, Math.max(1, bw * .7), h);
  });
}

// A thin frame around rect (the resting state of a map that fills in once answered).
export function frame(g, rect, colour) {
  g.strokeStyle = colour; g.lineWidth = 1;
  g.strokeRect(rect.x + .5, rect.y + .5, Math.max(1, rect.w - 1), Math.max(1, rect.h - 1));
}

// A small arrow from (x0, y) to (x1, y), its head at x1.
export function arrow(g, x0, x1, y) {
  const d = Math.sign(x1 - x0) || 1;
  g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.moveTo(x1 - 3 * d, y - 2.5); g.lineTo(x1, y); g.lineTo(x1 - 3 * d, y + 2.5); g.stroke();
}

// The share of the drawing's phase a segment has reached (it grows while it is reached).
export const reach = (st) => clamp(st.phase * (st.n || 1) - (st.i || 0), 0, 1);

// A shape as the boxes write it: 24×16.
export const shapeText = (s) => (Array.isArray(s) && s.length ? s.join("×") : "·");

// Whether (x, y) falls in rect widened by pad.
export const inside = (rect, x, y, pad = 4) => x >= rect.x - pad && x <= rect.x + rect.w + pad && y >= rect.y - pad && y <= rect.y + rect.h + pad;

// A segment's stage and whether the focus lights it.
export const litStage = (st, id) => st.focus.node(id);

// ---- a stage's words
// The caption of a sequence stage: its name and shape change, the lines given, its formula and its
// parameter count (null lines are left out).
export function stageCaption(stage, ...lines) {
  return el("div", {}, el("div", {}, el("span", { class: "who" }, stageLabel(stage)), el("span", { class: "muted" }, " ",
    t("box.shape", { from: shapeText(stage.in_shape), to: shapeText(stage.out_shape) }))),
  ...lines.filter(Boolean), stage.formula ? el("div", { class: "formula" }, stage.formula) : null,
  stage.params ? el("div", { class: "muted" }, tn("card.stage_params", stage.params)) : null);
}

// The stage's name above its drawing (at y `top`) and its shape change under it (at `bottom`),
// centred on the segment's box.
export function drawStageText(g, L, st, { top, bottom, name = null }) {
  const C = st.ctx.colors, cx = L.box.x + L.box.w / 2;
  g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = C.fg2; g.font = `11px ${css("--sans")}`;
  g.fillText(name || stageLabel(L.stage), cx, top, L.box.w + 16);
  g.fillStyle = C.muted; g.font = `10px ${css("--mono")}`;
  g.fillText(tText("box.shape", { from: shapeText(L.stage.in_shape), to: shapeText(L.stage.out_shape) }), cx, bottom, L.box.w + 24);
}
