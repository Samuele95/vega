// Two parts of the attention renderer (renderers/attention.js) that read tokens (D4 §7.2):
//   TRANSFORMER  a transformer encoder: one box per layer (at most LAYERS_MAX, else the first
//                titled "× n"), each a column of its tokens and a thumbnail per head. One answer:
//                each token's tick as long as its norm out of the layer, each head's map, and over
//                the column the attention the tokens receive — the [cls] query's row when the tokens
//                come from a feature tokenizer, else the mean over every query. Every weight: each
//                layer's projections (in_proj) and its feed-forward's first matrix.
//   TOKENIZER    feature tokens: each input feature (a numeric column, a nominal column) becomes one
//                token, beside the [cls] token the encoder reads the answer from. At rest each
//                token is as bright as its weights' norm; one answer, as its norm for the sample.
//                Every weight: the per-feature weights and biases (n × d each).
// A part answers the attention renderer's calls: measure(stage, ctx), layout(stage, box, ctx),
// draw(g, L, st), hit(L, x, y), caption(stage, hit, ctx), acts(stage, ex, graph).
import { el } from "../core/dom.js";
import { t, tText, tn, data } from "../core/text.js";
import { fmtNum } from "../core/fmt.js";
import { attentionMaps, exNode, tensorMatrix, downsample, paintCells, frame, inside, finite, normalised,
  stageCaption } from "./seq.js";

const LAYERS_MAX = 3, HEADS_MAX = 4, TOKENS_MAX = 16, PAD = 10, GAP = 8;

// The encoder's layers: its encoder_layer children (network-view@2), in order.
export function encoderLayers(graph, stage) {
  return (stage.children || []).map((id) => graph.byId(id)).filter((n) => n && n.type === "encoder_layer");
}

// The tokens shown of T: every one, or TOKENS_MAX spread evenly (the first and the last kept).
export function shownTokens(T) {
  if (T <= TOKENS_MAX) return Array.from({ length: T }, (_, k) => k);
  return Array.from({ length: TOKENS_MAX }, (_, k) => Math.round(k * (T - 1) / (TOKENS_MAX - 1)));
}

// What the tokens receive in one layer: the [cls] query's row averaged over the heads, else every
// query's row averaged (the mean attention a token gets); null without maps.
export function received(ex, layerId, maps) {
  const n = exNode(ex, layerId), cls = n && n.extra && Array.isArray(n.extra.cls_row) ? n.extra.cls_row : null;
  if (cls && cls.length && Array.isArray(cls[0])) return cls[0].map((_, k) => cls.reduce((s, row) => s + (row[k] || 0), 0) / cls.length);
  if (!maps || !maps.length) return null;
  const M = maps[0], out = new Array(M.cols).fill(0);
  maps.forEach((m) => { for (let r = 0; r < m.rows; r++) for (let c = 0; c < m.cols; c++) out[c] += m.v[r * m.cols + c] / (m.rows * maps.length); });
  return out;
}

// ---- the encoder
function layerBoxes(box, shown) {
  const w = (box.w - 2 * PAD - (shown - 1) * GAP) / shown, h = Math.min(box.h * .62, 96), y = box.y + (box.h - h) / 2;
  return Array.from({ length: shown }, (_, k) => ({ x: box.x + PAD + k * (w + GAP), y, w, h }));
}

function layerParts(r, heads) {
  const col = { x: r.x + 6, y: r.y + 12, w: Math.max(10, r.w * .38), h: r.h - 18 }, n = Math.min(HEADS_MAX, heads);
  const s = Math.max(6, Math.min(20, (r.h - 18 - (n - 1) * 4) / n, r.w * .4));
  const thumbs = Array.from({ length: n }, (_, k) => ({ x: r.x + r.w - s - 6, y: r.y + 12 + k * (s + 4), w: s, h: s }));
  return { col, thumbs, query: { x: col.x, y: r.y + 3, w: col.w, h: 6 } };
}

function drawLayer(g, L, k, st) {
  const C = st.ctx.colors, r = L.boxes[k], p = layerParts(r, L.heads), acts = st.acts && st.acts[st.i], a = acts ? acts.layers[k] : null;
  frame(g, r, st.hover && st.hover.i === st.i && st.hover.hit.layer === k ? C.fg : C.line);
  const toks = shownTokens(L.T), step = p.col.h / Math.max(1, toks.length), alpha = g.globalAlpha;
  toks.forEach((tk, j) => {
    const v = a && a.tokens ? a.tokens[tk] : null;
    g.globalAlpha = alpha * (finite(v) ? .35 + .65 * Math.abs(v) : .45); g.fillStyle = finite(v) ? C.acc : C.mut;
    g.fillRect(p.col.x, p.col.y + j * step, Math.max(2, p.col.w * (finite(v) ? Math.abs(v) : .5)), Math.max(1, step - 1));
  });
  g.globalAlpha = alpha;
  if (a && a.query) paintCells(g, { rows: 1, cols: toks.length, v: toks.map((tk) => a.query[tk] || 0) }, p.query, { colors: C });
  if (st.ctx.mode === "detail" && L.detail) {
    const d = L.detail[k], thumbs = [d.inProj, d.ff];
    thumbs.forEach((M, j) => { const tr = p.thumbs[j] || p.thumbs[p.thumbs.length - 1]; if (M) paintCells(g, downsample(M, 12, 12), tr, { colors: C }); frame(g, tr, C.line); });
    return;
  }
  p.thumbs.forEach((tr, h) => { const M = a && a.maps ? a.maps[h] : null; if (M) paintCells(g, downsample(M, 12, 12), tr, { colors: C, floor: .08 }); frame(g, tr, C.line); });
}

export const TRANSFORMER = Object.freeze({
  measure(stage, ctx) {
    const n = Math.max(1, encoderLayers(ctx.graph, stage).length || Number((stage.config || {}).layers) || 1), shown = n > LAYERS_MAX ? 1 : n;
    return { w: Math.max(ctx.boxW * 1.3, Math.min(ctx.compact ? 150 : 260, shown * 70 + 2 * PAD)), minH: 100 };
  },
  layout(stage, box, ctx) {
    const layers = encoderLayers(ctx.graph, stage), n = Math.max(1, layers.length || Number((stage.config || {}).layers) || 1);
    const shown = n > LAYERS_MAX ? 1 : n, boxes = layerBoxes(box, shown), c = stage.config || {};
    const read = (ref) => (typeof ref === "string" && ctx.graph.tensor(ref) ? tensorMatrix(ctx.graph.tensor(ref)) : null);
    const detail = ctx.mode === "detail" ? layers.map((l) => ({ inProj: read(((l.payload || {}).self_attn || {}).in_proj), ff: read(((l.payload || {}).ff || {}).linear1) })) : null;
    return { boxes, layers, n, collapsed: shown < n, T: (stage.in_shape || [0])[0], heads: Math.max(1, Number(c.heads || 1)), detail,
             top: boxes[0].y - 14, bottom: boxes[0].y + boxes[0].h + 14 };
  },
  draw(g, L, st) {
    L.boxes.forEach((_, k) => drawLayer(g, L, k, st));
    if (L.collapsed) {
      const r = L.boxes[0], C = st.ctx.colors;
      g.fillStyle = C.fg2; g.textAlign = "right"; g.textBaseline = "bottom"; g.fillText(tText("attention.layers", { n: L.n }), r.x + r.w, r.y - 2);
    }
  },
  hit(L, x, y) {
    const k = L.boxes.findIndex((r) => inside(r, x, y, 2));
    return k >= 0 ? { layer: k } : null;
  },
  caption(stage, hit, ctx) {
    const c = stage.config || {}, n = Math.max(1, encoderLayers(ctx.graph, stage).length || Number(c.layers) || 1);
    return stageCaption(stage, el("div", {}, tn("attention.cap.encoder", n, { h: Number(c.heads || 1), d: Number(c.d || 0), ff: Number(c.ff || 0) })),
      hit && Number.isInteger(hit.layer) ? el("div", { class: "muted" }, t("attention.cap.layer", { k: hit.layer + 1 })) : null);
  },
  // One answer, per layer: its maps, its tokens' norms out of it (scaled to the largest), what the
  // tokens receive.
  acts(stage, ex, graph) {
    const layers = encoderLayers(graph, stage).map((l) => {
      const n = exNode(ex, l.id), maps = attentionMaps(ex, l.id), norms = n && n.extra && n.extra.stream_norms ? n.extra.stream_norms.out : null;
      return n ? { maps, tokens: Array.isArray(norms) ? normalised(norms) : null, query: received(ex, l.id, maps) } : null;
    });
    return layers.some(Boolean) ? { layers } : null;
  },
});

// ---- the feature tokens
// The tokens' names: the network view's token labels ([cls] first), else [cls] and the features.
export function tokenNames(graph, stage) {
  const T = (stage.out_shape || [0])[0], own = graph.inputs.tokens;
  if (Array.isArray(own) && own.length === T) return own.slice();
  return [tText("attention.cls"), ...graph.inputs.names].slice(0, T);
}

// Each token's weight norm, [cls] first, scaled to the largest (a nominal token 0.5: its table has one row per value).
function restNorms(stage, T) {
  const p = stage.payload || {}, w = p.token_norms && Array.isArray(p.token_norms.weight) ? p.token_norms.weight : [];
  const raw = [finite(p.cls_norm) ? p.cls_norm : null, ...Array.from({ length: T - 1 }, (_, k) => (finite(w[k]) ? w[k] : null))];
  const n = normalised(raw);
  return n.map((v) => (finite(v) ? v : .5));
}

export const TOKENIZER = Object.freeze({
  measure(stage, ctx) { return { w: Math.max(ctx.boxW * 1.3, ctx.compact ? 90 : 130), minH: 100 }; },
  layout(stage, box, ctx) {
    const T = (stage.out_shape || [0])[0], toks = shownTokens(T), h = Math.min(box.h * .7, toks.length * 12), y0 = box.y + (box.h - h) / 2;
    const ys = toks.map((_, j) => y0 + (toks.length === 1 ? h / 2 : j * h / (toks.length - 1)));
    const xf = box.x + PAD + 4, xt = box.x + box.w - PAD - 8;
    return { T, toks, ys, xf, xt, rest: restNorms(stage, T), acts: ctx.acts || null, detail: ctx.mode === "detail" ? tokenWeights(ctx.graph, stage) : null, top: y0 - 16, bottom: y0 + h + 14, in: [{ x: xf, y: box.y + box.h / 2 }], out: [{ x: xt + 8, y: box.y + box.h / 2 }] };
  },
  draw(g, L, st) {
    const C = st.ctx.colors, acts = st.acts && st.acts[st.i], alpha = g.globalAlpha, hov = st.hover && st.hover.i === st.i ? st.hover.hit : null;
    if (st.ctx.mode === "detail" && L.detail) { detailTokens(g, L, st); return; }
    L.toks.forEach((tk, j) => {
      const y = L.ys[j], v = acts && acts.tokens ? acts.tokens[tk] : L.rest[tk];
      if (tk > 0) { g.globalAlpha = alpha * .35; g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(L.xf, y); g.lineTo(L.xt - 1, y); g.stroke(); g.fillStyle = C.mut; g.beginPath(); g.arc(L.xf, y, 2.5, 0, 7); g.fill(); }
      g.globalAlpha = alpha * (.3 + .7 * Math.abs(finite(v) ? v : .5)); g.fillStyle = acts && acts.tokens ? C.acc : tk === 0 ? C.fg2 : C.mut;
      g.fillRect(L.xt, y - 4, 8, 8);
      g.globalAlpha = alpha; g.strokeStyle = hov && hov.token === tk ? C.fg : C.line; g.strokeRect(L.xt + .5, y - 3.5, 7, 7);
    });
  },
  hit(L, x, y) {
    if (x < L.xf - 8 || x > L.xt + 14) return null;
    let best = null, bd = 7;
    L.ys.forEach((cy, j) => { const d = Math.abs(cy - y); if (d < bd) { bd = d; best = { token: L.toks[j] }; } });
    if (best && L.acts && L.acts.raw) best.norm = L.acts.raw[best.token];
    return best;
  },
  caption(stage, hit, ctx) {
    const T = (stage.out_shape || [0])[0], d = (stage.out_shape || [0, 0])[1], names = tokenNames(ctx.graph, stage);
    return stageCaption(stage, el("div", {}, tn("attention.cap.tokens", T, { d })),
      hit && Number.isInteger(hit.token) ? el("div", { class: "muted" }, t(hit.token === 0 ? "attention.cap.cls" : "attention.cap.token", { name: data(names[hit.token] ?? "") })) : null,
      hit && finite(hit.norm) ? el("div", { class: "muted" }, t("attention.cap.norm", { v: fmtNum(hit.norm) })) : null);
  },
  // One answer: each token's norm for the sample ([cls] first), scaled to the largest.
  acts(stage, ex) {
    const n = exNode(ex, stage.id), tokens = n && n.extra && Array.isArray(n.extra.tokens) ? n.extra.tokens : null;
    return tokens ? { tokens: normalised(tokens), raw: tokens } : null;
  },
});

// every weight: the per-feature weights and biases, n × d each
function detailTokens(g, L, st) {
  const C = st.ctx.colors, w = (L.xt - L.xf) / 2 - 4, y = L.ys[0], h = Math.max(12, L.ys[L.ys.length - 1] - y);
  [L.detail.weight, L.detail.bias].forEach((M, k) => {
    const r = { x: L.xf + k * (w + 8), y, w, h };
    if (M) paintCells(g, downsample(M, TOKENS_MAX, 16), r, { colors: C });
    frame(g, r, C.line);
  });
}

// The tokenizer's weight and bias as Matrix thumbnails (null when not in the payload).
export function tokenWeights(graph, stage) {
  const p = stage.payload || {}, read = (ref) => (typeof ref === "string" && graph.tensor(ref) ? tensorMatrix(graph.tensor(ref)) : null);
  return { weight: read(p.weight), bias: read(p.bias) };
}
