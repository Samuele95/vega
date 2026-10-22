// The graph stages (D4 §7.2): a graph convolution (gcn_conv, sage_conv, gat_conv, gin_conv) and a
// graph readout (graph_pool).
//
//   graph_conv   an ego-graph glyph: the node in the centre, a ring of neighbours whose messages it
//                gathers. gcn draws its normalised self-loop, sage its root weight apart from the
//                neighbours (a second path from the node itself), gin its (1 + ε) self-loop, gat an α
//                on its edges and its head count. One answer puts the explained node's real
//                neighbourhood on the ring: the eight messages that carried the most (|relevance|,
//                the gradient × input of the message), coloured by sign, each edge as thick as its
//                coefficient (gat: the attention α, averaged over the heads). At graph level every node
//                gathers; the ring holds the eight strongest messages of the whole graph.
//   graph_pool   many nodes drawn into one, with the readout (sum, mean, max); one answer sizes each
//                node by its share of the pooled vector.
//
// The "every weight" level draws each convolution's matrices as thumbnails: W (gcn), W root and W
// neighbours (sage), W with a_src and a_dst (gat), the inner MLP's layers (gin).
//
// The readers below (messagesOf, convState, poolState) are the ones views/shapes/graph.js reads the
// same answer with.
import { css, el } from "../../core/dom.js";
import { t, tText, tn, tnText } from "../../core/text.js";
import { fmtNum, sg } from "../../core/fmt.js";
import { stageLabel } from "../card.js";
import { nodeNamer } from "../../core/node-names.js";

const EXPLANATION = "vega/explanation@2";
export const RING_STATIC = 6;
export const RING_TRY = 8;
const POOL_DOTS = 8, THUMB_CELLS = 24, EXTRA_W = 32;
const shape = (s) => (Array.isArray(s) && s.length ? s.join("×") : "·");
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const num = (v) => (finite(v) ? v : 0);

// ---- reading one answer ----------------------------------------------------------------------------
// The payload of one node of an explanation@2, or null.
export function nodePayload(ex, id) {
  return ex && ex.schema === EXPLANATION && ex.nodes && ex.nodes[id] && ex.nodes[id].level !== "omitted" ? ex.nodes[id] : null;
}

// The level the answer explains (node | graph), from its sample.
export const levelOf = (ex) => (ex && ex.sample && ex.sample.graph && ex.sample.graph.level) || null;

// A message's coefficient: a number, or gat's α per head (their mean).
export const coefOf = (c) => (Array.isArray(c) ? c.reduce((s, v) => s + num(v), 0) / Math.max(1, c.length) : num(c));

// The messages a convolution passed for the explained sample (D3 §6.4 `messages`), each
// {src, dst, hop, root, self, rel, coef, heads, norm}; null when the answer carries none.
export function messagesOf(ex, id) {
  const n = nodePayload(ex, id), m = n && n.extra && Array.isArray(n.extra.messages) ? n.extra.messages : null;
  if (!m) return null;
  return m.map((x) => ({ src: x.src, dst: x.dst, hop: x.hop, root: !!x.root, self: x.src === x.dst, rel: num(x.relevance),
                         coef: coefOf(x.coef), heads: Array.isArray(x.coef) ? x.coef.slice() : null, norm: num(x.norm) }));
}

// What one answer draws on a convolution: its ring (the k messages from other nodes that carried the
// most, largest first), the node's own term (its self-loop or root message; at graph level their
// sum), how many messages there were, the largest |relevance| and the largest coefficient on the ring.
export function convState(ex, id, k = RING_TRY) {
  const msgs = messagesOf(ex, id);
  if (!msgs || !msgs.length) return null;
  const mine = msgs.filter((m) => m.self || m.root), others = msgs.filter((m) => !m.self && !m.root);
  const ring = others.slice().sort((a, b) => Math.abs(b.rel) - Math.abs(a.rel)).slice(0, k);
  const own = mine.length ? { rel: mine.reduce((s, m) => s + m.rel, 0), root: mine.some((m) => m.root), coef: mine[0].coef } : null;
  return { level: levelOf(ex), ring, own, n: msgs.length, others: others.length,
           mx: Math.max(1e-12, ...msgs.map((m) => Math.abs(m.rel))), cmax: Math.max(1e-12, ...ring.map((m) => Math.abs(m.coef))) };
}

// What one answer draws on a readout: each node's share of the pooled vector (D3 §6.4 `node_share`),
// the k largest first.
export function poolState(ex, id, k = POOL_DOTS) {
  const n = nodePayload(ex, id), s = n && n.extra && n.extra.node_share;
  if (!s || !Array.isArray(s.nodes) || !Array.isArray(s.share)) return null;
  const all = s.nodes.map((node, i) => ({ node, share: num(s.share[i]) }));
  const top = all.slice().sort((a, b) => Math.abs(b.share) - Math.abs(a.share)).slice(0, k);
  return { nodes: all.length, top, mx: Math.max(1e-12, ...top.map((x) => Math.abs(x.share))) };
}

// How strongly a stage answered: tanh of the root mean square of the explained node's output (its
// row of a node-level payload), in [0, 1).
export function glowOf(ex, id) {
  const n = nodePayload(ex, id), out = n && n.out;
  const v = !out ? null : out.form === "nodes" && out.target ? out.target.values : out.form === "vector" ? out.values : null;
  if (!Array.isArray(v) || !v.length) return null;
  return Math.tanh(Math.sqrt(v.reduce((s, x) => s + num(x) * num(x), 0) / v.length));
}

// ---- the weights -------------------------------------------------------------------------------------
// The matrices a convolution's detail level draws: [{key (a strings key), ref}] in the stage's order.
export function matricesOf(stage) {
  const p = stage.payload || {};
  const one = (key, ref) => (typeof ref === "string" ? [{ key, ref }] : []);
  if (stage.type === "sage_conv") return [...one("graph.w.root", p.root), ...one("graph.w.neigh", p.neigh)];
  if (stage.type === "gat_conv") return [...one("graph.w.weight", p.weight), ...one("graph.w.att_src", p.att_src), ...one("graph.w.att_dst", p.att_dst)];
  if (stage.type === "gin_conv") return (Array.isArray(p.mlp) ? p.mlp : []).flatMap((m) => one("graph.w.mlp", m && m.weight));
  return one("graph.w.weight", p.weight);
}

// A matrix read down to at most `cells` × `cells` blocks, each the block's mean: {rows, cols, v, mx}.
export function thumbOf(M, cells = THUMB_CELLS) {
  if (!M) return null;
  const R = Math.min(cells, M.out), C = Math.min(cells, M.in), v = new Float32Array(R * C);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const r0 = Math.floor(r * M.out / R), r1 = Math.max(r0 + 1, Math.floor((r + 1) * M.out / R));
    const c0 = Math.floor(c * M.in / C), c1 = Math.max(c0 + 1, Math.floor((c + 1) * M.in / C));
    let s = 0;
    for (let i = r0; i < r1; i++) for (let j = c0; j < c1; j++) s += M.w[i * M.in + j];
    v[r * C + c] = s / ((r1 - r0) * (c1 - c0));
  }
  return { rows: R, cols: C, v, mx: Math.max(1e-12, ...Array.from(v, Math.abs)) };
}

// ---- drawing -----------------------------------------------------------------------------------------
// Points on a ring of radius r around c, the first at the top, clockwise.
export function ringPoints(n, c, r) {
  return Array.from({ length: n }, (_, k) => { const a = -Math.PI / 2 + 2 * Math.PI * k / Math.max(1, n); return { x: c.x + r * Math.cos(a), y: c.y + r * Math.sin(a) }; });
}

// What the stage's glyph says beside the ring: a self-loop (gcn with self loops, gin), a root path
// (sage), an α (gat).
function markOf(stage) {
  const p = stage.payload || {}, cfg = stage.config || {};
  if (stage.type === "gin_conv") return { loop: true, text: tText("graph.gin.self", { one: 1, eps: fmtNum(num(p.eps ?? cfg.eps)) }) };
  if (stage.type === "sage_conv") return { root: true, text: tText("graph.sage.root") };
  if (stage.type === "gat_conv") return { alpha: true, text: tnText("graph.gat.heads", Number(p.heads ?? cfg.heads ?? 1)) };
  return { loop: !!(p.self_loops ?? cfg.self_loops), text: "" };
}

function drawRing(g, L, a, C) {
  const pts = ringPoints(a ? a.ring.length : RING_STATIC, L.c, L.r);
  if (a && !pts.length) {                                // no message from another node reached it
    g.globalAlpha = .45; g.strokeStyle = C.muted; g.lineWidth = 1; g.setLineDash([2, 3]);
    g.beginPath(); g.arc(L.c.x, L.c.y, L.r, 0, 7); g.stroke(); g.setLineDash([]);
  }
  pts.forEach((p, k) => {
    const m = a ? a.ring[k] : null, w = m ? .8 + 2.6 * Math.abs(m.coef) / a.cmax : 1;
    g.globalAlpha = m ? .25 + .75 * Math.abs(m.rel) / a.mx : .6;
    g.strokeStyle = m ? (m.rel >= 0 ? C.acc : C.neg) : C.muted; g.lineWidth = w;
    g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(L.c.x, L.c.y); g.stroke();
    g.fillStyle = m ? (m.rel >= 0 ? C.acc : C.neg) : C.mut;
    g.beginPath(); g.arc(p.x, p.y, m ? 2.5 + 3.5 * Math.abs(m.rel) / a.mx : 3.5, 0, 7); g.fill();
  });
  return pts;
}

function drawMark(g, L, a, C) {
  const mk = L.mark, c = L.c;
  g.globalAlpha = .85; g.strokeStyle = C.fg2; g.fillStyle = C.fg2; g.lineWidth = 1;
  if (mk.loop) { g.beginPath(); g.arc(c.x + 6, c.y - 6, 6, 0, 7); g.stroke(); }
  if (mk.root) {
    const y = c.y + L.r + 10;
    g.setLineDash([2, 2]); g.beginPath(); g.moveTo(c.x, c.y); g.lineTo(c.x, y); g.stroke(); g.setLineDash([]);
    g.beginPath(); g.arc(c.x, y, 3, 0, 7); g.fill();
  }
  const own = a && a.own;
  g.globalAlpha = 1; g.fillStyle = own ? (own.rel >= 0 ? C.acc : C.neg) : C.fg2;
  g.beginPath(); g.arc(c.x, c.y, 6, 0, 7); g.fill();
  if (mk.text) {
    g.fillStyle = C.muted; g.font = `9.5px ${css("--mono")}`; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(mk.text, c.x, L.rect.y + 8, L.rect.w - 6);
  }
}

function drawPool(g, L, a, C) {
  const pts = poolPoints(L, a ? a.top.length : POOL_DOTS), out = { x: L.rect.x + L.rect.w * .78, y: L.c.y };
  pts.forEach((p, k) => {
    const x = a ? a.top[k] : null;
    g.globalAlpha = x ? .3 + .7 * Math.abs(x.share) / a.mx : .55; g.strokeStyle = C.muted; g.lineWidth = .8;
    g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(out.x, out.y); g.stroke();
    g.fillStyle = x ? C.acc : C.mut; g.beginPath(); g.arc(p.x, p.y, x ? 2 + 3.5 * Math.abs(x.share) / a.mx : 3, 0, 7); g.fill();
  });
  g.globalAlpha = 1; g.fillStyle = C.fg2; g.beginPath(); g.arc(out.x, out.y, 7, 0, 7); g.fill();
  g.fillStyle = C.muted; g.font = `10px ${css("--mono")}`; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(tText(`graph.readout.${L.readout}`), out.x, out.y - 16, L.rect.w / 2);
}

// The nodes a readout gathers, in two loose columns on the left of its box.
function poolPoints(L, n) {
  const x0 = L.rect.x + L.rect.w * .2, h = L.rect.h * .6, y0 = L.c.y - h / 2;
  return Array.from({ length: n }, (_, k) => ({ x: x0 + (k % 2) * 12, y: y0 + h * (n === 1 ? .5 : k / (n - 1)) }));
}

// The detail level: each matrix as a heat thumbnail, side by side (a missing one dashed).
function drawThumbs(g, L, graph, C) {
  const mats = L.thumbs, n = Math.max(1, mats.length), gap = 6, w = (L.rect.w - 8 - gap * (n - 1)) / n, h = Math.min(w, L.rect.h * .55);
  mats.forEach((m, k) => {
    const x = L.rect.x + 4 + k * (w + gap), y = L.c.y - h / 2, T = thumbOf(graph.matrix(m.ref));
    m.box = { x, y, w, h };
    if (!T) { g.globalAlpha = .6; g.strokeStyle = C.muted; g.setLineDash([2, 3]); g.strokeRect(x + .5, y + .5, w - 1, h - 1); g.setLineDash([]); return; }
    const cw = w / T.cols, ch = h / T.rows;
    for (let r = 0; r < T.rows; r++) for (let c = 0; c < T.cols; c++) {
      const v = T.v[r * T.cols + c];
      g.globalAlpha = .08 + .9 * Math.sqrt(Math.abs(v) / T.mx); g.fillStyle = v >= 0 ? C.acc : C.neg;
      g.fillRect(x + c * cw, y + r * ch, Math.ceil(cw), Math.ceil(ch));
    }
    g.globalAlpha = .9; g.fillStyle = C.muted; g.font = `9.5px ${css("--mono")}`; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(tText(m.key), x + w / 2, y - 8, w + gap);
  });
}

// A thumbnail pointed at: its matrix's name and size, or that the payload does not carry it.
function thumbLine(m, graph) {
  const M = graph && typeof graph.matrix === "function" ? graph.matrix(m.ref) : null;
  return M ? t("graph.cap.thumb", { name: tText(m.key), rows: M.out, cols: M.in }) : t("graph.cap.thumb_missing", { name: tText(m.key) });
}

const isDetail = (st) => st.mode === "detail" || (st.ctx && st.ctx.mode === "detail");

// The name of a node in a caption: the answer's namer (core/node-names.js: "this node", a reference
// node's key, else its rank in the answer), never its index in the assembled graph.
const unnamed = () => tText("graph.node.other");

export default {
  kinds: ["graph_conv", "graph_pool"],
  detail: true,

  measure(seg, ctx) {
    const pool = ctx.graph.byId(seg.stage).kind === "graph_pool";
    return { w: ctx.boxW + (pool || ctx.compact ? 0 : EXTRA_W), minH: 80 };
  },

  layout(seg, box, ctx) {
    const stage = ctx.graph.byId(seg.stage), pool = stage.kind === "graph_pool";
    const h = Math.min(150, Math.max(64, box.w * 1.1), Math.max(64, box.h * .5)), rect = { x: box.x, y: box.y + (box.h - h) / 2, w: box.w, h };
    const c = { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 }, r = Math.max(10, Math.min(rect.w * .34, rect.h * .34));
    const readout = String((stage.payload && stage.payload.readout) || (stage.config && stage.config.readout) || "sum");
    return { box, rect, stage, pool, c, r, readout, mark: pool ? null : markOf(stage), acts: ctx.acts || null,
             thumbs: pool ? [] : matricesOf(stage).map((m) => ({ ...m })), label: stageLabel(stage),
             in: [{ x: rect.x, y: c.y }], out: [{ x: rect.x + rect.w, y: c.y }] };
  },

  draw(g, seg, L, st) {
    const C = st.ctx.colors, on = st.focus.node(L.stage.id), a = st.acts && st.acts[st.i], hot = st.hover && st.hover.i === st.i, r = L.rect;
    g.globalAlpha = on ? 1 : .35;
    if (a && finite(a.glow)) { g.globalAlpha = .08 + .4 * a.glow; g.fillStyle = C.acc; g.fillRect(r.x, r.y, r.w, r.h); g.globalAlpha = on ? 1 : .35; }
    g.strokeStyle = hot ? C.fg : st.focus.active && !st.focus.all && on ? C.acc : C.line; g.lineWidth = hot ? 1.6 : 1;
    g.strokeRect(r.x + .5, r.y + .5, r.w - 1, r.h - 1);
    const reveal = Math.max(0, Math.min(1, st.phase * st.n - st.i));
    g.save(); g.globalAlpha *= st.phase >= 1 ? 1 : reveal;
    L.detail = !L.pool && isDetail(st) && L.thumbs.length > 0;  // hit() reads what was drawn
    if (L.pool) drawPool(g, L, a && a.pool, C);
    else if (L.detail) drawThumbs(g, L, st.ctx.graph, C);
    else { L.pts = drawRing(g, L, a && a.conv, C); drawMark(g, L, a && a.conv, C); }
    g.restore();
    g.globalAlpha = on ? .92 : .35; g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = C.fg2; g.font = `11px ${css("--sans")}`;
    g.fillText(L.label, r.x + r.w / 2, r.y + r.h + 12, r.w + 24);
    g.fillStyle = C.muted; g.font = `10px ${css("--mono")}`;
    g.fillText(tText("box.shape", { from: shape(L.stage.in_shape), to: shape(L.stage.out_shape) }), r.x + r.w / 2, r.y + r.h + 26, r.w + 24);
    g.globalAlpha = 1;
  },

  hit(seg, L, x, y) {
    const r = L.rect;
    if (x < r.x - 4 || x > r.x + r.w + 4 || y < r.y - 14 || y > r.y + r.h + 30) return null;
    if (L.detail) {                                     // the thumbnails are on screen, not the ring
      const m = L.thumbs.find((b) => b.box && x >= b.box.x && x <= b.box.x + b.box.w && y >= b.box.y && y <= b.box.y + b.box.h);
      return { stage: seg.stage, k: -1, alone: false, message: null, thumb: m ? { key: m.key, ref: m.ref } : null };
    }
    const conv = L.acts && L.acts.conv, pts = conv ? ringPoints(conv.ring.length, L.c, L.r) : [];
    const k = pts.findIndex((p) => Math.hypot(p.x - x, p.y - y) <= 9);
    const m = k >= 0 ? conv.ring[k] : null;
    const name = (L.acts && L.acts.name) || unnamed;
    return { stage: seg.stage, k, alone: !!conv && !conv.ring.length,
             message: m ? { src: m.src, dst: m.dst, from: name(m.src), to: name(m.dst), hop: m.hop, rel: m.rel, coef: m.coef } : null };
  },

  caption(seg, hit, ctx) {
    const s = ctx.graph.byId(seg.stage), m = hit && hit.message, heads = s.payload && s.payload.heads;
    return el("div", {}, el("div", {}, el("span", { class: "who" }, stageLabel(s)), el("span", { class: "muted" }, " ",
      t("box.shape", { from: shape(s.in_shape), to: shape(s.out_shape) }))),
      s.formula ? el("div", { class: "formula" }, s.formula) : null,
      el("div", { class: "muted" }, t(`graph.about.${s.type}`, { heads: Number(heads || 1), readout: tText(`graph.readout.${(s.payload && s.payload.readout) || "sum"}`) })),
      hit && hit.thumb ? el("div", { class: "graph-cap-message" }, thumbLine(hit.thumb, ctx.graph)) : null,
      hit && hit.alone ? el("div", { class: "graph-cap-message" }, t("graph.cap.alone")) : null,
      m ? el("div", { class: "graph-cap-message" }, t("graph.cap.message", { from: m.from, to: m.to, hop: m.hop, rel: sg(m.rel, 3), coef: fmtNum(+m.coef.toPrecision(3)) })) : null,
      s.params ? el("div", { class: "muted" }, tn("card.stage_params", s.params)) : null);
  },

  // One answer: the messages (a convolution) or the node shares (a readout), and the glow.
  acts(seg, ex, graph) {
    const stage = graph.byId(seg.stage), glow = glowOf(ex, seg.stage);
    const conv = stage.kind === "graph_conv" ? convState(ex, seg.stage) : null, pool = stage.kind === "graph_pool" ? poolState(ex, seg.stage) : null;
    return conv || pool || glow !== null ? { conv, pool, glow, name: nodeNamer(ex) } : null;
  },
};
