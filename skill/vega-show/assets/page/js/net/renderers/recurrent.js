// A recurrent layer — rnn, gru or lstm (D4 §7.2): the cell unrolled over the steps it reads, one
// cell per step (at most SHOW_MAX: the first EDGE, a gap, the last EDGE when the window is longer),
// arrows carrying the state from one step to the next; a second row running backwards when it is
// bidirectional. What it hands on is lit: the last cell (`return: last` — a backward row's last is
// its first step), every cell (`all`), or every cell under a bracket that averages or takes the
// largest (`mean`, `max`). Every weight: each gate's two matrices as thumbnails, what the step's
// input feeds it (W_ih) and what the previous state feeds it (W_hh), from the first layer's published
// weights (their norms when the service withholds them). One answer: each cell as bright as the
// hidden state's norm at that step (the probe's unroll, top layer), and under the cells what each
// step tipped (the window's attribution summed over its features).
import { css, el } from "../../core/dom.js";
import { t, tText, tn, data } from "../../core/text.js";
import { fmtNum } from "../../core/fmt.js";
import { stageLabel } from "../card.js";
import { exNode, rowBlock, tensorMatrix, rowNorms, normalised, downsample, paintCells, paintBars, frame, arrow, reach,
  shapeText, inside, litStage, finite } from "../seq.js";

const SHOW_MAX = 16, EDGE = 6, PAD = 10, GAP = 3, BARS_H = 16, THUMB = 20;
// The readouts a recurrent block may hand on (catalog: blocks.gru.params.return), each named in the
// strings table (recurrent.return.*).
export const RETURNS = Object.freeze(["last", "all", "mean", "max"]);

// The steps drawn, in order: every one, or the first EDGE, null for the gap, the last EDGE.
export function shownSteps(L) {
  const n = Math.max(0, L | 0), range = (a, b) => Array.from({ length: b - a }, (_, k) => a + k);
  return n <= SHOW_MAX ? range(0, n) : [...range(0, EDGE), null, ...range(n - EDGE, n)];
}

// {L, F, H, dirs, layers, ret, cell, gates} of a recurrent stage.
export function recurrentShape(stage) {
  const p = stage.payload || {}, c = stage.config || {}, [L, F] = stage.in_shape || [0, 0];
  const ret = RETURNS.includes(p.return || c.return) ? p.return || c.return : "last";
  return { L, F, H: Number(p.hidden || c.hidden || 0), dirs: Number(p.directions || (c.bidirectional ? 2 : 1)),
           layers: Number(p.layers || c.num_layers || 1), ret, cell: String(p.cell || stage.type), gates: Array.isArray(p.gates) ? p.gates : [] };
}

// The steps each row hands on: the last of the forward row and the first of the backward row, or all.
export function handedOn(shape) {
  if (shape.ret !== "last") return Array.from({ length: shape.dirs }, () => null);
  return Array.from({ length: shape.dirs }, (_, d) => (d === 0 ? shape.L - 1 : 0));
}

// Each gate's two matrices of the first layer's forward direction: [{gate, ih, hh, norms}] (ih and hh
// Matrix or null when the weights are not in the payload; norms the published gate norms).
export function gateBlocks(graph, stage) {
  const s = recurrentShape(stage), p = stage.payload || {}, layer = (p.per_layer || []).find((l) => l.layer === 0 && l.direction !== "reverse");
  if (!layer || !s.gates.length) return [];
  const read = (ref) => (typeof ref === "string" && graph.tensor(ref) ? tensorMatrix(graph.tensor(ref)) : null);
  const ih = read(layer.w_ih), hh = read(layer.w_hh), H = s.H;
  return s.gates.map((gate, k) => ({ gate, ih: ih ? rowBlock(ih, k * H, (k + 1) * H) : null, hh: hh ? rowBlock(hh, k * H, (k + 1) * H) : null,
    norms: (layer.gate_norms || {})[gate] || null, rho: (layer.rho_hh || {})[gate] ?? null }));
}

function cellsOf(box, shape, cs) {
  const shown = shownSteps(shape.L), width = shown.length * (cs + GAP) - GAP, x0 = box.x + (box.w - width) / 2, mid = box.y + box.h / 2;
  const ys = shape.dirs === 2 ? [mid - cs * .9, mid + cs * .9] : [mid];
  return { x0, width, ys, cells: ys.flatMap((y, d) => shown.map((step, k) => ({ x: x0 + k * (cs + GAP), y: y - cs / 2, d, k, step }))) };
}

// ---- drawing
function drawCells(g, L, st) {
  const C = st.ctx.colors, acts = st.acts && st.acts[st.i], lit = handedOn(L.shape), hov = st.hover && st.hover.i === st.i ? st.hover.hit : null;
  const alpha = g.globalAlpha;
  L.cells.forEach((c) => {
    if (c.step === null) { g.fillStyle = C.muted; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText("…", c.x + L.cs / 2, c.y + L.cs / 2); return; }
    const norm = acts && acts.norms && acts.norms[c.d] ? acts.norms[c.d][c.step] : null, on = lit[c.d] === null || lit[c.d] === c.step;
    g.globalAlpha = alpha * (finite(norm) ? .25 + .75 * Math.abs(norm) : .55);
    g.fillStyle = finite(norm) ? C.acc : C.mut; g.fillRect(c.x, c.y, L.cs, L.cs);
    g.globalAlpha = alpha; g.lineWidth = on ? 1.4 : 1;
    g.strokeStyle = hov && hov.d === c.d && hov.step === c.step ? C.fg : on ? C.acc : C.line;
    g.strokeRect(c.x + .5, c.y + .5, L.cs - 1, L.cs - 1);
  });
}

function drawArrows(g, L, st) {
  const C = st.ctx.colors;
  g.strokeStyle = C.muted; g.lineWidth = 1;
  L.cells.forEach((c, i) => {
    const next = L.cells[i + 1];
    if (!next || next.d !== c.d || c.step === null || next.step === null) return;
    const y = c.y + L.cs / 2, a = c.x + L.cs + .5, b = next.x - .5;
    if (c.d === 0) arrow(g, a, b, y); else arrow(g, b, a, y);
  });
}

// mean / max: a bracket under every cell, the readout's word on it
function drawBracket(g, L, st) {
  if (L.shape.ret !== "mean" && L.shape.ret !== "max") return;
  const C = st.ctx.colors, y = L.ys[L.ys.length - 1] + L.cs / 2 + 4, x0 = L.x0, x1 = L.x0 + L.width;
  g.strokeStyle = C.fg2; g.lineWidth = 1; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0, y + 4); g.lineTo(x1, y + 4); g.lineTo(x1, y); g.stroke();
  g.fillStyle = C.fg2; g.font = `10px ${css("--mono")}`; g.textAlign = "center"; g.textBaseline = "top";
  g.fillText(tText(`recurrent.return.${L.shape.ret}`), (x0 + x1) / 2, y + 6);
}

// what each shown step tipped, under the cells
function drawSteps(g, L, st) {
  const acts = st.acts && st.acts[st.i];
  if (!acts || !acts.perStep) return;
  const shown = L.cells.filter((c) => c.d === 0), values = shown.map((c) => (c.step === null ? null : acts.perStep[c.step]));
  paintBars(g, values, L.bars, { colors: st.ctx.colors, mx: 1 });
}

// every weight: per gate, W_ih and W_hh side by side above the cells
function drawGates(g, L, st) {
  if (st.ctx.mode !== "detail" || !L.gates.length) return;
  const C = st.ctx.colors, n = L.gates.length, w = Math.min(THUMB, (L.width - (n - 1) * 6) / (2 * n)), y = L.ys[0] - L.cs / 2 - w - 16;
  g.font = `9px ${css("--mono")}`; g.textAlign = "center"; g.textBaseline = "bottom";
  L.gates.forEach((gb, k) => {
    const x = L.x0 + k * (2 * w + 6);
    [gb.ih, gb.hh].forEach((M, j) => {
      const rect = { x: x + j * w, y, w: w - 1, h: w };
      if (M) paintCells(g, downsample(M, 12, 12), rect, { colors: C }); else frame(g, rect, C.line);
    });
    g.fillStyle = C.muted; g.fillText(data(gb.gate), x + w, y - 1);
  });
}

function drawLabel(g, L, st) {
  const C = st.ctx.colors, cx = L.box.x + L.box.w / 2, top = L.ys[0] - L.cs / 2 - (st.ctx.mode === "detail" && L.gates.length ? 44 : 12);
  g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = C.fg2; g.font = `11px ${css("--sans")}`;
  g.fillText(L.label + (L.shape.layers > 1 ? " " + tText("recurrent.layers", { n: L.shape.layers }) : ""), cx, top, L.box.w + 16);
  g.fillStyle = C.muted; g.font = `10px ${css("--mono")}`;
  g.fillText(tText("box.shape", { from: shapeText(L.stage.in_shape), to: shapeText(L.stage.out_shape) }), cx, L.bars.y + L.bars.h + 10, L.box.w + 24);
}

export default {
  kinds: ["recurrent"],
  detail: true,

  measure(seg, ctx) {
    const slots = shownSteps(recurrentShape(ctx.graph.byId(seg.stage)).L).length, per = ctx.compact ? 9 : 14;
    return { w: Math.max(ctx.boxW * (ctx.compact ? 1.3 : 1.6), Math.min(ctx.compact ? 150 : 270, slots * per + 2 * PAD)), minH: 100 };
  },

  layout(seg, box, ctx) {
    const stage = ctx.graph.byId(seg.stage), shape = recurrentShape(stage), slots = Math.max(1, shownSteps(shape.L).length);
    const cs = Math.max(4, Math.min(12, (box.w - 2 * PAD) / slots - GAP)), c = cellsOf(box, shape, cs);
    const bottom = c.ys[c.ys.length - 1] + cs / 2, lit = handedOn(shape), acts = ctx.acts || null;
    const outCells = c.cells.filter((x) => x.step !== null && (lit[x.d] === null ? x.k === slots - 1 && x.d === 0 : lit[x.d] === x.step));
    return {
      box, stage, shape, cs, ...c, acts, label: stageLabel(stage), gates: gateBlocks(ctx.graph, stage),
      bars: { x: c.x0, y: bottom + (shape.ret === "mean" || shape.ret === "max" ? 22 : 6), w: c.width, h: BARS_H },
      in: [{ x: c.x0 - 4, y: box.y + box.h / 2 }],
      out: outCells.map((x) => ({ x: x.x + cs + 2, y: x.y + cs / 2, v: acts && acts.norms && acts.norms[x.d] ? acts.norms[x.d][x.step] : null })),
      captions: [{ x: box.x + box.w / 2, text: tText("recurrent.caption", { cell: data(shape.cell.toUpperCase()), h: shape.H * shape.dirs }) }],
    };
  },

  draw(g, seg, L, st) {
    g.globalAlpha = (litStage(st, L.stage.id) ? 1 : .35) * Math.max(.15, reach(st));
    g.font = `10px ${css("--mono")}`;
    drawGates(g, L, st);
    drawArrows(g, L, st);
    drawCells(g, L, st);
    drawBracket(g, L, st);
    drawSteps(g, L, st);
    drawLabel(g, L, st);
    g.globalAlpha = 1;
  },

  hit(seg, L, x, y) {
    const c = L.cells.find((q) => q.step !== null && inside({ x: q.x, y: q.y, w: L.cs, h: L.cs }, x, y, 2));
    const raw = L.acts && L.acts.raw && L.acts.raw[c ? c.d : 0];
    if (c) return { stage: seg.stage, d: c.d, step: c.step, norm: raw && finite(raw[c.step]) ? raw[c.step] : null };
    return inside({ x: L.x0, y: L.ys[0] - L.cs * 2, w: L.width, h: L.bars.y + L.bars.h - L.ys[0] + L.cs * 2 }, x, y, 8) ? { stage: seg.stage } : null;
  },

  caption(seg, hit, ctx) {
    const s = ctx.graph.byId(seg.stage), shape = recurrentShape(s), steps = ctx.graph.inputs.steps;
    const rho = gateBlocks(ctx.graph, s).filter((g) => finite(g.rho)).map((g) => `${data(g.gate)} ${fmtNum(g.rho)}`).join(" · ");
    const at = hit && Number.isInteger(hit.step) ? hit.step : null;
    return el("div", {}, el("div", {}, el("span", { class: "who" }, stageLabel(s)), el("span", { class: "muted" }, " ",
      t("box.shape", { from: shapeText(s.in_shape), to: shapeText(s.out_shape) }))),
      el("div", {}, tn("recurrent.cap.units", shape.H, { cell: data(shape.cell) }), " ", t(shape.dirs === 2 ? "recurrent.cap.both" : "recurrent.cap.forward"),
        " ", t(`recurrent.cap.return.${shape.ret}`)),
      rho ? el("div", { class: "muted" }, t("recurrent.cap.memory", { list: data(rho) })) : null,
      at !== null ? el("div", { class: "muted" }, t(hit.d === 1 ? "recurrent.cap.step_back" : "recurrent.cap.step",
        { step: steps && steps.length === shape.L ? data(steps[at]) : at + 1 })) : null,
      at !== null && finite(hit.norm) ? el("div", { class: "muted" }, t("recurrent.cap.norm", { v: fmtNum(hit.norm) })) : null,
      s.formula ? el("div", { class: "formula" }, s.formula) : null,
      s.params ? el("div", { class: "muted" }, tn("card.stage_params", s.params)) : null);
  },

  // One answer: the hidden state's norm at every step of each direction (the top layer's unroll, or
  // the exact hidden states), each direction scaled to its largest; and what each step tipped (the
  // window's attribution per step, when it reads the network's own window).
  acts(seg, ex, graph) {
    const stage = graph.byId(seg.stage), shape = recurrentShape(stage), raw = hiddenNorms(ex, seg.stage, shape);
    const a = ex && ex.attribution, perStep = a && a.shape === "window" && Array.isArray(a.per_step) && a.per_step.length === shape.L ? normalised(a.per_step) : null;
    if (!raw && !perStep) return null;
    return { raw, norms: raw ? raw.map((r) => (r ? normalised(r) : null)) : null, perStep };
  },
};

// The hidden state's norm per step, one list per direction (forward first), from the probe's unroll
// (its top layer) or the exact hidden states (layers·directions × steps × units); null when neither.
export function hiddenNorms(ex, id, shape) {
  const n = exNode(ex, id);
  if (!n) return null;
  const per = n.extra && n.extra.unroll && Array.isArray(n.extra.unroll.per_layer) ? n.extra.unroll.per_layer : null;
  if (per && per.length) {
    const top = Math.max(...per.map((p) => p.layer || 0)), rows = per.filter((p) => (p.layer || 0) === top);
    const byDir = (dir) => (rows.find((p) => p.direction === dir) || {}).hidden_norm || null;
    return [byDir("forward"), ...(shape.dirs === 2 ? [byDir("reverse")] : [])];
  }
  const h = n.exact && n.exact.hidden;
  if (!h || !Array.isArray(h.shape) || h.shape.length !== 3) return null;
  const M = tensorMatrix(h, 1), last = h.shape[0] - shape.dirs;
  return Array.from({ length: shape.dirs }, (_, d) => rowNorms(rowBlock(M, (last + d) * h.shape[1], (last + d + 1) * h.shape[1])));
}
