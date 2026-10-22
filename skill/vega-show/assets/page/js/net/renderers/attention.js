// The attention family (D4 §7.2): one module for four stage kinds, each drawn by its own part.
//   attention       one step × step map per head (at most HEADS_MAX drawn, "+n" for the rest): a
//                   frame at rest, filled once answered with how much each step (row) read each other
//                   step (column); a pooling layer (`pool: true`) adds the bracket of the mean over
//                   the steps. Every weight: the query, key, value and output projections as
//                   thumbnails. When an answer carries no map for it, it says so (attention.not_exposed).
//   attention_pool  one bar per step: how much of the pooled vector each step makes up — equal at rest,
//                   the answer's weights once answered. Every weight: the query vector and the score
//                   matrix it reads the steps through.
//   transformer     and tokenizer: net/seq-tokens.js (the encoder's stacked layers, the feature tokens).
import { el } from "../../core/dom.js";
import { t, tText, tn, data } from "../../core/text.js";
import { attentionMaps, exNode, tensorMatrix, rowBlock, downsample, paintCells, paintBars, frame, reach, inside, litStage,
  finite, stageCaption, drawStageText, matrix } from "../seq.js";
import { TRANSFORMER, TOKENIZER } from "../seq-tokens.js";

const HEADS_MAX = 4, PAD = 10, GAP = 6, BARS_MAX = 48;

// {L, d, heads, pool} of an attention stage.
export function attentionShape(stage) {
  const p = stage.payload || {}, c = stage.config || {}, [L, d] = stage.in_shape || [0, 0];
  return { L, d: Number(p.dim || c.dim || d || 0), heads: Math.max(1, Number(p.heads || c.heads || 1)), pool: !!(p.pool ?? c.pool) };
}

// The four projections of an attention stage as Matrix thumbnails [{name, M}] (M null when the
// weights are not in the payload): q, k and v are the thirds of in_proj, o is out_proj.
export function projections(graph, p) {
  const read = (ref) => (typeof ref === "string" && graph.tensor(ref) ? tensorMatrix(graph.tensor(ref)) : null);
  const inProj = read(p.in_proj), d = inProj ? inProj.rows / 3 : 0;
  return [["q", 0], ["k", 1], ["v", 2]].map(([name, k]) => ({ name, M: inProj ? rowBlock(inProj, k * d, (k + 1) * d) : null }))
    .concat([{ name: "o", M: read(p.out_proj) }]);
}

// Squares in a row, centred in box: n of them, each at most `max` wide.
export function squares(box, n, max) {
  const s = Math.max(10, Math.min(max, (box.w - 2 * PAD - (n - 1) * GAP) / Math.max(1, n), box.h * .42));
  const w = n * s + (n - 1) * GAP, x0 = box.x + (box.w - w) / 2, y = box.y + (box.h - s) / 2;
  return Array.from({ length: n }, (_, k) => ({ x: x0 + k * (s + GAP), y, w: s, h: s }));
}

// ---- attention
const ATTENTION = Object.freeze({
  measure(stage, ctx) {
    const n = Math.min(HEADS_MAX, attentionShape(stage).heads);
    return { w: Math.max(ctx.boxW * 1.3, Math.min(ctx.compact ? 130 : 200, n * 40 + 2 * PAD)), minH: 90 };
  },
  layout(stage, box, ctx) {
    const s = attentionShape(stage), shown = Math.min(HEADS_MAX, s.heads), rects = squares(box, shown, 44);
    const detail = ctx.mode === "detail" ? projections(ctx.graph, stage.payload || {}) : null;
    return { rects, detailRects: detail ? squares(box, 4, 34) : null, detail, shape: s, more: s.heads - shown,
             top: rects[0].y - 14, bottom: rects[0].y + rects[0].h + (s.pool ? 26 : 14) };
  },
  draw(g, L, st) {
    const C = st.ctx.colors, acts = st.acts && st.acts[st.i];
    if (L.detail) L.detail.forEach((p, k) => { const r = L.detailRects[k]; if (p.M) paintCells(g, downsample(p.M, 16, 16), r, { colors: C }); frame(g, r, C.line); });
    else L.rects.forEach((r, h) => {
      const M = acts && acts.maps ? acts.maps[h] : null;
      if (M) paintCells(g, downsample(M, 24, 24), r, { colors: C, floor: .08 }); else restMap(g, r, L.shape, C);
      frame(g, r, st.hover && st.hover.i === st.i && st.hover.hit.head === h ? C.fg : C.line);
    });
    g.fillStyle = C.muted; g.textAlign = "left"; g.textBaseline = "middle";
    if (L.more > 0 && !L.detail) { const r = L.rects[L.rects.length - 1]; g.fillText(tText("attention.more", { n: L.more }), r.x + r.w + 3, r.y + r.h / 2); }
    if (L.shape.pool) poolBracket(g, L.rects, C);
    if (acts && !acts.maps) { g.textAlign = "center"; g.fillText(tText("attention.not_exposed"), L.box.x + L.box.w / 2, L.bottom + 12, L.box.w + 20); }
  },
  hit(L, x, y) {
    const h = L.rects.findIndex((r) => inside(r, x, y, 2));
    return h >= 0 ? { head: h } : null;
  },
  caption(stage, hit) {
    const s = attentionShape(stage);
    return stageCaption(stage, el("div", {}, tn("attention.cap.heads", s.heads, { d: s.d })),
      s.pool ? el("div", { class: "muted" }, t("attention.cap.pool")) : null,
      hit && Number.isInteger(hit.head) ? el("div", { class: "muted" }, t("attention.cap.head", { h: hit.head + 1 })) : null);
  },
  acts(stage, ex) {
    if (!exNode(ex, stage.id)) return null;
    return { maps: attentionMaps(ex, stage.id) };
  },
});

// a map at rest: the frame, and its diagonal (every step reads itself) faint
function restMap(g, r, shape, C) {
  const alpha = g.globalAlpha, n = Math.max(1, Math.min(24, shape.L || 1));
  g.globalAlpha = alpha * .18; g.fillStyle = C.mut; g.fillRect(r.x, r.y, r.w, r.h);
  g.globalAlpha = alpha * .5; g.fillStyle = C.muted;
  for (let k = 0; k < n; k++) g.fillRect(r.x + k * r.w / n, r.y + k * r.h / n, Math.max(1, r.w / n), Math.max(1, r.h / n));
  g.globalAlpha = alpha;
}

function poolBracket(g, rects, C) {
  const a = rects[0], b = rects[rects.length - 1], y = a.y + a.h + 5;
  g.strokeStyle = C.fg2; g.lineWidth = 1; g.beginPath(); g.moveTo(a.x, y); g.lineTo(a.x, y + 4); g.lineTo(b.x + b.w, y + 4); g.lineTo(b.x + b.w, y); g.stroke();
  g.fillStyle = C.fg2; g.textAlign = "center"; g.textBaseline = "top"; g.fillText(tText("attention.pool.mean"), (a.x + b.x + b.w) / 2, y + 6);
}

// ---- attention pooling
// The weights of one answer over the steps (the exact alpha, or the probe's one-row map), or null.
export function poolWeights(ex, id) {
  const maps = attentionMaps(ex, id);
  if (!maps || !maps[0]) return null;
  const M = maps[0];
  return Array.from({ length: M.cols }, (_, c) => M.v[(M.rows - 1) * M.cols + c]);
}

const POOL = Object.freeze({
  measure(stage, ctx) {
    const L = (stage.in_shape || [0])[0];
    return { w: Math.max(ctx.boxW * 1.3, Math.min(ctx.compact ? 130 : 200, Math.min(BARS_MAX, L) * 4 + 2 * PAD)), minH: 90 };
  },
  layout(stage, box, ctx) {
    const L = (stage.in_shape || [0])[0], h = Math.min(56, box.h * .4), rect = { x: box.x + PAD, y: box.y + (box.h - h) / 2, w: box.w - 2 * PAD, h };
    const p = stage.payload || {}, read = (ref) => (typeof ref === "string" && ctx.graph.tensor(ref) ? tensorMatrix(ctx.graph.tensor(ref)) : null);
    const detail = ctx.mode === "detail" ? { query: read(p.query) || (Array.isArray(p.query) ? matrix(1, p.query.length, Float32Array.from(p.query)) : null), score: read(p.score) } : null;
    return { rect, steps: L, detail, top: rect.y - 14, bottom: rect.y + rect.h + 14 };
  },
  draw(g, L, st) {
    const C = st.ctx.colors, acts = st.acts && st.acts[st.i], n = Math.max(1, Math.min(BARS_MAX, L.steps || 1));
    if (L.detail) { detailPool(g, L, C); return; }
    const w = acts && acts.weights ? downsampleList(acts.weights, n) : null, bars = w || Array.from({ length: n }, () => 1 / n);
    const alpha = g.globalAlpha;
    g.globalAlpha = alpha * (w ? 1 : .35);
    paintBars(g, bars.map((v) => 2 * v), { ...L.rect, h: L.rect.h * 2 }, { colors: C, mx: 2 * Math.max(1e-12, ...bars) });
    g.globalAlpha = alpha;
    g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath(); g.moveTo(L.rect.x, L.rect.y + L.rect.h); g.lineTo(L.rect.x + L.rect.w, L.rect.y + L.rect.h); g.stroke();
  },
  hit(L, x, y) {
    if (!inside(L.rect, x, y, 2)) return null;
    return { step: Math.max(0, Math.min(L.steps - 1, Math.floor((x - L.rect.x) / L.rect.w * L.steps))) };
  },
  caption(stage, hit, ctx) {
    const steps = ctx.graph.inputs.steps, L = (stage.in_shape || [0])[0];
    return stageCaption(stage, el("div", {}, t("attention.cap.pool_steps", { n: L })),
      hit && Number.isInteger(hit.step) ? el("div", { class: "muted" }, t("attention.cap.step", { step: steps && steps.length === L ? data(steps[hit.step]) : hit.step + 1 })) : null);
  },
  acts(stage, ex) {
    const weights = poolWeights(ex, stage.id);
    return weights ? { weights } : exNode(ex, stage.id) ? { weights: null } : null;
  },
});

// A list's block means down to n entries.
export function downsampleList(values, n) {
  if (values.length <= n) return values.slice();
  return Array.from({ length: n }, (_, k) => {
    const a = Math.floor(k * values.length / n), b = Math.max(a + 1, Math.floor((k + 1) * values.length / n));
    return values.slice(a, b).reduce((s, v) => s + (finite(v) ? v : 0), 0) / (b - a);
  });
}

function detailPool(g, L, C) {
  const r = L.rect, q = { x: r.x, y: r.y, w: r.w, h: Math.max(6, r.h * .2) }, s = { x: r.x, y: r.y + q.h + 6, w: r.w, h: r.h - q.h - 6 };
  if (L.detail.query) paintCells(g, downsample(L.detail.query, 1, BARS_MAX), q, { colors: C });
  if (L.detail.score) paintCells(g, downsample(L.detail.score, 12, BARS_MAX), s, { colors: C });
  frame(g, q, C.line); frame(g, s, C.line);
}

// ---- the module: each kind's part, one interface
const PARTS = Object.freeze({ attention: ATTENTION, attention_pool: POOL, transformer: TRANSFORMER, tokenizer: TOKENIZER });
const partOf = (stage) => PARTS[stage.kind] || ATTENTION;

export default {
  kinds: ["attention", "attention_pool", "transformer", "tokenizer"],
  detail: true,

  measure(seg, ctx) { const s = ctx.graph.byId(seg.stage); return partOf(s).measure(s, ctx); },

  layout(seg, box, ctx) {
    const stage = ctx.graph.byId(seg.stage), part = partOf(stage), own = part.layout(stage, box, ctx), mid = box.y + box.h / 2;
    return { box, stage, part, ...own, in: own.in || [{ x: box.x + 2, y: mid }], out: own.out || [{ x: box.x + box.w - 2, y: mid }],
             captions: [{ x: box.x + box.w / 2, text: tText(`attention.caption.${stage.kind}`, captionParams(stage)) }] };
  },

  draw(g, seg, L, st) {
    g.globalAlpha = (litStage(st, L.stage.id) ? 1 : .35) * Math.max(.15, reach(st));
    L.part.draw(g, L, st);
    g.globalAlpha = Math.min(1, g.globalAlpha + .3);
    drawStageText(g, L, st, { top: L.top, bottom: L.bottom + (L.stage.kind === "attention" && L.shape && L.shape.pool ? 14 : 0) });
    g.globalAlpha = 1;
  },

  hit(seg, L, x, y) {
    const own = L.part.hit(L, x, y);
    if (own) return { stage: seg.stage, ...own };
    return inside({ x: L.box.x, y: L.top - 8, w: L.box.w, h: L.bottom - L.top + 16 }, x, y) ? { stage: seg.stage } : null;
  },

  caption(seg, hit, ctx) { const s = ctx.graph.byId(seg.stage); return partOf(s).caption(s, hit, ctx); },

  acts(seg, ex, graph) { const s = graph.byId(seg.stage); return partOf(s).acts(s, ex, graph); },
};

// The numbers a kind's caption under the drawing names: heads, steps, layers, tokens.
function captionParams(stage) {
  const c = stage.config || {}, p = stage.payload || {}, [L] = stage.in_shape || [0];
  return { h: Number(p.heads || c.heads || 1), n: L, layers: Number(p.layers || c.layers || (stage.children || []).length || 1),
           tokens: (stage.out_shape || [0])[0] };
}
