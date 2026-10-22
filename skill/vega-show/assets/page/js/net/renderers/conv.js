// A convolution along the steps and the pooling that narrows it (D4 §7.2).
//   conv1d   families level: a strip of the output, one row per channel (C′, at most ROWS_MAX
//            drawn, the rest folded into block means) by one column per position (L′), each row
//            shaded by its channel's weight norm; at its left the kernel, k taps wide, bracketed over
//            the positions one output reads. Every weight: the kernel itself as a thumbnail, C′ rows
//            by C·k columns (input channel by input channel, tap by tap), from the published kernel
//            (its channel-pair norms when the service withholds the weights). One answer: the
//            strip becomes this sample's activation map (the probe's heat), coloured by sign.
//   pool1d   a strip that narrows from the input's positions to the output's, the pooling's kind
//            written on it; one answer fills the narrow end with the pooled map and marks, under
//            the input's positions, how many cells each one won (max pooling).
// The strip's rows and columns are hit one by one: pointing at a cell names its channel and its
// position.
import { css, el } from "../../core/dom.js";
import { t, tText, tn, data } from "../../core/text.js";
import { stageLabel } from "../card.js";
import { seqOut, exNode, transpose, downsample, maxAbs, matrix, tensorMatrix, paintCells, paintBars, frame, reach,
  shapeText, inside, litStage, normalised } from "../seq.js";

const ROWS_MAX = 16, COLS_MAX = 48, KERNEL_W = 26, PAD = 8, LABEL_GAP = 13;
// The pooling kinds the strings table names (conv.pool1d.kind.*); another is written as max.
export const POOL_KINDS = Object.freeze(["max", "avg"]);

// {L, C, Lo, Co, k, stride, padding, dilation} of a conv1d or pool1d stage.
export function convShape(stage) {
  const [L, C] = stage.in_shape || [0, 0], [Lo, Co] = stage.out_shape || [0, 0], p = stage.payload || {}, c = stage.config || {};
  return { L, C, Lo, Co, k: Number(c.kernel || p.size || c.size || 1), stride: Number(p.stride ?? c.stride ?? 1),
           padding: Number(p.padding ?? c.padding ?? 0), dilation: Number(p.dilation ?? c.dilation ?? 1) };
}

// The kernel as a Matrix of C′ rows by C·k columns (the weight tensor's own order), or, when the
// service withholds it, the channel-pair norms (C′ × C); null when neither is in the payload.
export function kernelMatrix(graph, stage) {
  const p = stage.payload || {}, t = typeof p.kernel === "string" ? graph.tensor(p.kernel) : null;
  const M = t ? tensorMatrix(t) : null;
  if (M) return M;
  const pairs = p.pair_norms;
  if (!Array.isArray(pairs) || !pairs.length || !Array.isArray(pairs[0])) return null;
  return matrix(pairs.length, pairs[0].length, Float32Array.from(pairs.flat()));
}

// The channel norms of a conv stage, normalised to [0, 1] (null for a pooling).
function channelShares(stage) {
  const n = stage.payload && stage.payload.channel_norms;
  return Array.isArray(n) && n.length ? normalised(n) : null;
}

const kindOf = (L) => L.stage.kind;
const glyphKind = (stage) => String((stage.payload && stage.payload.kind) || (stage.config && stage.config.kind) || "max");

function stripRect(box, isConv, rows) {
  const x = box.x + (isConv ? KERNEL_W : PAD), w = Math.max(20, box.x + box.w - PAD - x);
  const h = Math.min(box.h * .55, Math.max(28, rows * 7)), y = box.y + (box.h - h) / 2;
  return { x, y, w, h };
}

// ---- drawing
function drawKernel(g, L, st) {                         // the taps, bracketed over the positions they read
  const C = st.ctx.colors, s = L.shape, taps = Math.min(9, s.k), r = L.rect, cw = r.w / Math.min(COLS_MAX, Math.max(1, s.Lo));
  const x0 = L.box.x + 2, y = r.y + r.h / 2, tw = Math.min(4, (KERNEL_W - 8) / taps);
  g.fillStyle = C.fg2;
  for (let j = 0; j < taps; j++) g.fillRect(x0 + j * (tw + 1), y - 6, tw, 12);
  g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath();
  g.moveTo(x0 + taps * (tw + 1), y - 8); g.lineTo(r.x, r.y - 3); g.lineTo(r.x + cw * Math.min(s.k, s.Lo), r.y - 3); g.stroke();
}

// the resting strip: each channel's row at the opacity of its weight norm
function drawRest(g, L, st) {
  const C = st.ctx.colors, r = L.rect, rows = L.rows, cols = L.cols, share = L.shares;
  const alpha = g.globalAlpha;
  for (let i = 0; i < rows; i++) {
    const v = share ? Math.abs(share[Math.floor(i * share.length / rows)] || 0) : .5;
    g.globalAlpha = alpha * (.22 + .6 * v); g.fillStyle = C.mut;
    for (let j = 0; j < cols; j++) g.fillRect(r.x + j * r.w / cols, r.y + i * r.h / rows, Math.max(1, r.w / cols - 1), Math.max(1, r.h / rows - 1));
  }
  g.globalAlpha = alpha;
}

// pool1d: the input's positions narrowing into the output's
function drawNarrowing(g, L, st) {
  const C = st.ctx.colors, r = L.rect, w0 = L.box.x + PAD, out = r;
  g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath();
  g.moveTo(w0, r.y - 6); g.lineTo(out.x, out.y); g.moveTo(w0, r.y + r.h + 6); g.lineTo(out.x, out.y + out.h); g.stroke();
  g.fillStyle = C.fg2; g.font = `10px ${css("--mono")}`; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(tText(`conv.pool1d.kind.${POOL_KINDS.includes(glyphKind(L.stage)) ? glyphKind(L.stage) : "max"}`), (w0 + out.x) / 2, r.y + r.h / 2);
}

function drawStrip(g, L, st) {
  const C = st.ctx.colors, acts = st.acts && st.acts[st.i], detail = st.ctx.mode === "detail" && L.kernel;
  if (detail) paintCells(g, downsample(L.kernel, ROWS_MAX * 2, COLS_MAX * 2), L.rect, { colors: C });
  else if (acts && acts.map) paintCells(g, downsample(transpose(acts.map), L.rows, L.cols), L.rect, { colors: C, mx: acts.mx });
  else drawRest(g, L, st);
  frame(g, L.rect, st.hover && st.hover.i === st.i ? C.fg : C.line);
  if (acts && acts.winners && L.inW) {                   // the input positions each max won
    const w = acts.winners, band = { x: L.box.x + PAD, y: L.rect.y + L.rect.h + 8, w: L.rect.x - L.box.x - PAD, h: 10 };
    paintBars(g, w.map((v) => Math.abs(v)), { ...band, y: band.y - band.h / 2, h: band.h * 2 }, { colors: C });
  }
}

function drawLabel(g, L, st) {
  const C = st.ctx.colors, r = L.rect, cx = L.box.x + L.box.w / 2;
  g.globalAlpha = Math.min(1, g.globalAlpha + .3); g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = C.fg2; g.font = `11px ${css("--sans")}`;
  g.fillText(L.label, cx, r.y - LABEL_GAP, L.box.w + 16);
  g.fillStyle = C.muted; g.font = `10px ${css("--mono")}`;
  g.fillText(tText("box.shape", { from: shapeText(L.stage.in_shape), to: shapeText(L.stage.out_shape) }), cx, r.y + r.h + LABEL_GAP + (L.inW ? 8 : 0), L.box.w + 24);
}

// the cell of the strip under (x, y): its channel (row) and position (column), in the stage's own counts
function cellAt(L, x, y) {
  const r = L.rect, s = L.shape, col = Math.floor((x - r.x) / r.w * Math.max(1, s.Lo)), row = Math.floor((y - r.y) / r.h * Math.max(1, s.Co));
  return { step: Math.max(0, Math.min(s.Lo - 1, col)), channel: Math.max(0, Math.min(s.Co - 1, row)) };
}

export default {
  kinds: ["conv", "pool1d"],
  detail: true,

  measure(seg, ctx) {
    const s = convShape(ctx.graph.byId(seg.stage)), base = ctx.compact ? ctx.boxW * 1.3 : ctx.boxW * 1.6;
    return { w: Math.max(base, Math.min(ctx.compact ? 130 : 220, s.Lo * 4 + KERNEL_W + 2 * PAD)), minH: 90 };
  },

  layout(seg, box, ctx) {
    const stage = ctx.graph.byId(seg.stage), shape = convShape(stage), isConv = stage.kind === "conv";
    const rows = Math.max(1, Math.min(ROWS_MAX, shape.Co)), cols = Math.max(1, Math.min(COLS_MAX, shape.Lo));
    let rect = stripRect(box, isConv, rows);
    if (!isConv) { const w = Math.max(16, rect.w * Math.max(.3, shape.L ? shape.Lo / shape.L : .5)); rect = { ...rect, x: rect.x + rect.w - w, w }; }
    const mid = rect.y + rect.h / 2, acts = ctx.acts || null;
    return {
      box, rect, stage, shape, rows, cols, label: stageLabel(stage), shares: isConv ? channelShares(stage) : null, inW: !isConv,
      kernel: isConv ? kernelMatrix(ctx.graph, stage) : null,
      in: [{ x: box.x + 2, y: mid }], out: [{ x: rect.x + rect.w + 2, y: mid, v: acts && acts.last !== undefined ? acts.last : null }],
      captions: [{ x: box.x + box.w / 2, text: tText(isConv ? "conv.caption" : "conv.caption.pool1d", { l: shape.Lo, c: shape.Co }) }],
    };
  },

  draw(g, seg, L, st) {
    const on = litStage(st, L.stage.id);
    g.globalAlpha = (on ? 1 : .35) * Math.max(.15, reach(st));
    if (kindOf(L) === "conv") drawKernel(g, L, st); else drawNarrowing(g, L, st);
    drawStrip(g, L, st);
    drawLabel(g, L, st);
    g.globalAlpha = 1;
  },

  hit(seg, L, x, y) {
    if (inside(L.rect, x, y, 0)) return { stage: seg.stage, ...cellAt(L, x, y) };
    return inside({ x: L.box.x, y: L.rect.y - 20, w: L.box.w, h: L.rect.h + 40 }, x, y) ? { stage: seg.stage } : null;
  },

  caption(seg, hit, ctx) {
    const s = ctx.graph.byId(seg.stage), shape = convShape(s), isConv = s.kind === "conv", steps = ctx.graph.inputs.steps;
    return el("div", {}, el("div", {}, el("span", { class: "who" }, stageLabel(s)), el("span", { class: "muted" }, " ",
      t("box.shape", { from: shapeText(s.in_shape), to: shapeText(s.out_shape) }))),
      isConv ? el("div", {}, t("conv.cap.kernel", { k: shape.k, stride: shape.stride, padding: shape.padding, c: shape.C, co: shape.Co }))
        : el("div", {}, t("conv.cap.pool", { kind: tText(`conv.pool1d.kind.${POOL_KINDS.includes(glyphKind(s)) ? glyphKind(s) : "max"}`), size: shape.k })),
      hit && Number.isInteger(hit.step) ? el("div", { class: "muted" }, t("conv.cap.cell", { channel: hit.channel + 1, at: hit.step + 1,
        step: steps && steps.length === shape.Lo ? data(steps[hit.step]) : hit.step + 1 })) : null,
      s.formula ? el("div", { class: "formula" }, s.formula) : null,
      s.params ? el("div", { class: "muted" }, tn("card.stage_params", s.params)) : null);
  },

  // One answer: the stage's output map for the explained sample (steps × channels), its scale, the
  // mean of its last position (the port's sign), and, for a max pooling, the positions each won.
  acts(seg, ex) {
    const map = seqOut(ex, seg.stage), n = exNode(ex, seg.stage);
    if (!map) return null;
    const lastRow = Array.from({ length: map.cols }, (_, c) => map.v[(map.rows - 1) * map.cols + c]);
    const winners = n && n.extra && Array.isArray(n.extra.winners) ? n.extra.winners : null;
    return { map, mx: maxAbs(map) || 1, last: lastRow.reduce((a, v) => a + v, 0) / Math.max(1, lastRow.length), winners };
  },
};
