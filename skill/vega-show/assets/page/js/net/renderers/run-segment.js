// A dense run inside a larger graph (D4 §7.3): one column of units per dense stage, the K units of
// each that both listen and speak the most (every unit at the "every weight" level, drawn as ticks
// past 80), the strongest weights between neighbouring columns, and a caption under each column.
// Its first column is fed by the previous segment through the first stage's weights (the canvas's
// connector reads `inMatrix`), its last column feeds the next segment. A fused bend, rate or
// normalisation is a mark on its column; a decoder's columns say so; the narrowest column of an
// autoencoder is its waist. One answer colours every unit by its value for the explained sample, and
// its ports carry those values (`v`), so a fan into or out of a box takes the sign of what it carries.
// Under D4 §7.3's overflow rule (ctx.compact) its columns are narrow.
import { el } from "../../core/dom.js";
import { t, tText, tn, data } from "../../core/text.js";
import { signedNorm, stageValues } from "../relevance.js";
import { tensorValues } from "../graph.js";

const COL = 64, COL_NARROW = 46, TICKS = 80;
const norm2 = (w, from, to) => { let s = 0; for (let k = from; k < to; k++) s += w[k] * w[k]; return Math.sqrt(s); };

// The run's layers with their matrices (graph.runModel; the head is a segment of its own here).
function modelOf(seg, ctx) {
  if (!seg.model) seg.model = ctx.graph.runModel(seg, { head: false });
  return seg.model;
}

// The units kept in each column: by ‖row‖ · ‖next layer's column‖ (‖row‖ alone for the last), K at most.
function keptUnits(layers, K, acts) {
  return layers.map((l, k) => {
    const W = l.W, next = layers[k + 1] && layers[k + 1].W, n = l.out;
    const score = Array.from({ length: n }, (_, j) => {
      if (acts && acts[k]) return [Math.abs(acts[k][j] || 0), j];
      const listen = W ? norm2(W.w, j * W.in, (j + 1) * W.in) : 1;
      let speak = 1;
      if (next) { speak = 0; for (let o = 0; o < next.out; o++) speak += next.w[o * next.in + j] ** 2; speak = Math.sqrt(speak); }
      return [listen * speak, j];
    });
    return score.sort((a, b) => b[0] - a[0]).slice(0, Math.min(K, n)).map((s) => s[1]).sort((a, b) => a - b);
  });
}

// Between column k and k+1 (weights of layer k+1): each kept target's 3 strongest kept sources and
// each source's strongest target; per pair [ja, jb, w, n].
function innerEdges(layers, kept) {
  return kept.slice(0, -1).map((srcs, k) => {
    const W = layers[k + 1].W, dst = kept[k + 1];
    if (!W) return [];
    const w = (a, b) => W.w[dst[b] * W.in + srcs[a]], seen = new Set(), out = [];
    const add = (a, b) => { const key = a + ":" + b; if (!seen.has(key)) { seen.add(key); out.push([a, b, w(a, b)]); } };
    dst.forEach((_, b) => srcs.map((_, a) => [a, Math.abs(w(a, b))]).sort((p, q) => q[1] - p[1]).slice(0, 3).forEach(([a]) => add(a, b)));
    srcs.forEach((_, a) => { let best = 0, bw = -1; dst.forEach((_, b) => { const v = Math.abs(w(a, b)); if (v > bw) { bw = v; best = b; } }); add(a, best); });
    const mx = Math.max(1e-9, ...out.map((e) => Math.abs(e[2])));
    return out.map(([a, b, v]) => [a, b, v, Math.abs(v) / mx]);
  });
}

// A column's caption: the layer's number in the drawing, the units drawn of how many; a decoder's
// layers and an autoencoder's waist say so.
function captionOf(ctx, layers, kept, k) {
  const l = layers[k], key = l.id === ctx.waist ? "net.caption.bottleneck" : l.group ? "net.caption.decoder" : "net.caption.hidden";
  return tText(key, { h: ctx.ordinal && ctx.ordinal.has(l.id) ? ctx.ordinal.get(l.id) : k + 1, k: kept[k].length, n: l.out });
}

function lit(st, id) { return st.focus.node(id); }

function drawColumn(g, L, k, st) {
  const C = st.ctx.colors, x = L.xs[k], ys = L.ys[k], ticks = ys.length > TICKS, acts = st.acts && st.acts[st.i];
  const on = lit(st, L.layers[k].id);
  ys.forEach((y, n) => {
    const j = L.kept[k][n], v = acts ? acts[k][j] || 0 : null, hot = st.hover && st.hover.i === st.i && st.hover.hit.k === k && st.hover.hit.n === n;
    g.globalAlpha = hot ? 1 : v !== null ? .2 + .8 * Math.min(1, Math.abs(v)) : on ? .95 : .25;
    g.fillStyle = hot ? C.fg : v !== null ? (v >= 0 ? C.acc : C.neg) : st.focus.active && !st.focus.all && on ? C.acc : C.mut;
    const r = 4.5 + (v !== null ? 2.5 * Math.min(1, Math.abs(v)) : 0);
    if (ticks) g.fillRect(x - 4, y - .6, 8, 1.3); else { g.beginPath(); g.arc(x, y, r, 0, 7); g.fill(); }
  });
  const l = L.layers[k];
  if (l.norm || l.dropout > 0) {                      // fused marks: a normalisation tick, a dimmed rate bar
    g.globalAlpha = .6; g.strokeStyle = C.muted; g.lineWidth = 1;
    if (l.norm) { g.beginPath(); g.moveTo(x - 9, L.box.y + 4); g.lineTo(x + 9, L.box.y + 4); g.stroke(); }
    if (l.dropout > 0) { g.setLineDash([2, 3]); g.beginPath(); g.moveTo(x + 11, L.box.y + 8); g.lineTo(x + 11, L.box.y + L.box.h - 8); g.stroke(); g.setLineDash([]); }
  }
}

export default {
  measure(seg, ctx) {
    const n = seg.layers.length, w = ctx.narrow || ctx.compact ? COL_NARROW : COL;
    return { w: Math.max(w, n * w), minH: 120, grow: true };
  },

  layout(seg, box, ctx) {
    const layers = modelOf(seg, ctx).layers, K = Number.isFinite(ctx.K) ? ctx.K : Infinity;
    const acts = ctx.acts || null, kept = keptUnits(layers, K, acts), n = layers.length;
    const xs = layers.map((_, k) => box.x + box.w * (n === 1 ? .5 : (k + .5) / n));
    const ys = kept.map((ks) => ks.map((_, i) => box.y + box.h * (ks.length === 1 ? .5 : i / (ks.length - 1))));
    const first = layers[0], value = (k, j) => (acts && acts[k] ? acts[k][j] : undefined);
    return {
      box, layers, kept, xs, ys, edges: innerEdges(layers, kept),
      in: kept[0].map((j, i) => ({ x: xs[0] - 6, y: ys[0][i], unit: j, v: value(0, j) })), inMatrix: first && first.W,
      out: kept[n - 1].map((j, i) => ({ x: xs[n - 1] + 6, y: ys[n - 1][i], units: [j], v: value(n - 1, j) })),
      captions: layers.map((_, k) => ({ x: xs[k], text: captionOf(ctx, layers, kept, k) })),
    };
  },

  draw(g, seg, L, st) {
    const C = st.ctx.colors, reach = Math.max(0, Math.min(1, st.phase * st.n - st.i));
    L.edges.forEach((list, k) => list.forEach(([a, b, w, nw]) => {
      const x0 = L.xs[k], y0 = L.ys[k][a], x1 = L.xs[k + 1], y1 = L.ys[k + 1][b], on = st.focus.edge(L.layers[k].id, L.layers[k + 1].id);
      g.globalAlpha = (on ? Math.min(1, nw * 2) * .55 : .04) * Math.min(1, reach * 2); g.strokeStyle = w >= 0 ? C.acc : C.neg; g.lineWidth = .9;
      const mid = (x0 + x1) / 2; g.beginPath(); g.moveTo(x0, y0); g.bezierCurveTo(mid, y0, mid, y1, x1, y1); g.stroke();
    }));
    if (reach <= 0) return;
    L.layers.forEach((_, k) => drawColumn(g, L, k, st));
  },

  hit(seg, L, x, y) {
    let best = null, bd = 12;
    L.xs.forEach((cx, k) => { if (Math.abs(cx - x) > 12) return; L.ys[k].forEach((cy, n) => { const d = Math.abs(cy - y); if (d < bd) { bd = d; best = { k, n, j: L.kept[k][n] }; } }); });
    return best;
  },

  caption(seg, hit, ctx) {
    const layers = modelOf(seg, ctx).layers, l = layers[hit.k], W = l.W, j = hit.j;
    const row = W ? Array.from(W.w.subarray(j * W.in, (j + 1) * W.in)) : [];
    const top = row.map((v, i) => [i, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3);
    const names = hit.k === 0 ? modelOf(seg, ctx).inputs : null;
    const list = top.map(([i, v]) => tText("legacy.net.cap.pair", { name: names ? data(names[i]) : tText("legacy.net.unit", { j: i + 1 }), verb: tText(v >= 0 ? "legacy.net.verb.raises" : "legacy.net.verb.lowers") })).join(", ");
    return el("div", {}, el("div", {}, el("span", { class: "who" }, t("legacy.net.cap.unit", { j: j + 1 })), el("span", { class: "muted" }, " ", t("legacy.net.cap.unit_of", { n: l.out, l: ctx.ordinal && ctx.ordinal.has(l.id) ? ctx.ordinal.get(l.id) : hit.k + 1 }))),
      list ? el("div", {}, t("legacy.net.cap.listens", { list })) : null, tn("card.stage_params", l.params));
  },

  // One answer: every column's values for the explained sample, normalised per column (null when
  // the explanation carries none of the run's stages).
  acts(seg, ex) {
    const values = seg.layers.map((l) => stageValues(ex, l.dense, tensorValues));
    if (values.every((v) => v === null)) return null;
    return values.map((v) => (v ? signedNorm(v) : null));
  },
};
