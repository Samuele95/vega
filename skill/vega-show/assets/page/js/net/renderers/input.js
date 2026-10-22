// The inputs' segment (D4 §7.2): one circle per input family, its label on the left. On windows each
// family carries a strip of the last steps (a family × step glyph); on graphs a small ego-graph glyph
// says a node reads its neighbours too. Its circles feed the next segment through the first dense
// stage's weights (each port carries the family's unit indices) or a fan. One answer sizes each
// family by what it tipped (gradient × input, added up per family).
import { css, el } from "../../core/dom.js";
import { t, tText, tn, data } from "../../core/text.js";
import { signedNorm } from "../relevance.js";

const LABEL = 150, LABEL_NARROW = 110, STEP_MAX = 12;

// what tipped the answer, per family, from an explanation@2's attribution (row, window or graph)
export function tippedByFamily(ex, families) {
  const a = ex && ex.attribution, byName = new Map();
  if (!a || !a.features) return null;
  if (a.shape === "row") a.features.names.forEach((n, i) => byName.set(n, a.features.attribution[i]));
  else if (a.shape === "window") {
    a.features.names.forEach((n, i) => byName.set(n, a.per_feature[i]));
    if (a.nominal) a.nominal.names.forEach((n, i) => byName.set(n, [].concat(a.nominal.attribution[i]).reduce((s, v) => s + v, 0)));
  } else if (a.shape === "graph" && Array.isArray(a.self)) a.features.names.forEach((n, i) => byName.set(n, a.self[i]));
  else return null;
  return signedNorm(families.map((f) => f.members.reduce((s, m) => s + (byName.get(m) || 0), 0)));
}

// Per family: whether it holds a nominal input the network reads through an embedding.
export function embeddedFamilies(graph, families) {
  const nominal = new Set(graph.inputs.nominal.map((c) => c.name)), embeds = graph.stages().some((n) => n.kind === "embedding");
  return families.map((f) => embeds && f.members.some((m) => nominal.has(m)));
}

export default {
  kinds: ["input"], detail: false,

  measure(seg, ctx) { return { w: (ctx.narrow ? LABEL_NARROW : LABEL) + 24, minH: 60 }; },

  layout(seg, box, ctx) {
    const fam = ctx.env.families, x = box.x + box.w - 10, graph = ctx.graph;
    const ys = fam.map((_, j) => box.y + box.h * (fam.length === 1 ? .5 : j / (fam.length - 1)));
    const steps = graph.sample.shape === "window" ? Math.min(STEP_MAX, graph.sample.length || 0) : 0;
    return { box, x, ys, steps, fam, graph: graph.sample.shape === "graph", embedded: embeddedFamilies(graph, fam),
             out: fam.map((f, j) => ({ x: x + 6, y: ys[j], units: f.idx })), captions: [{ x, text: tText("net.caption.inputs") }] };
  },

  draw(g, seg, L, st) {
    const C = st.ctx.colors, acts = st.acts && st.acts[st.i], feature = st.focus.feature, on = st.focus.node("input");
    g.textBaseline = "middle"; g.font = `12.5px ${css("--sans")}`; g.textAlign = "right";
    L.fam.forEach((f, j) => {
      const v = acts ? acts[j] : null, hit = feature !== null && f.members.includes(feature);
      const hot = st.hover && st.hover.i === st.i && st.hover.hit.j === j;
      g.globalAlpha = hot || hit ? 1 : v !== null ? .2 + .8 * Math.min(1, Math.abs(v)) : on ? .95 : .25;
      g.fillStyle = hot ? C.fg : hit ? C.acc : v !== null ? (v >= 0 ? C.acc : C.neg) : C.mut;
      g.beginPath(); g.arc(L.x, L.ys[j], 7 + (v !== null ? 2.5 * Math.min(1, Math.abs(v)) : 0), 0, 7); g.fill();
      for (let s = 0; s < L.steps; s++) g.fillRect(L.x - 16 - s * 4, L.ys[j] - 2, 3, 4);   // the steps it spans
      if (L.embedded[j]) for (let r = 0; r < 3; r++) g.fillRect(L.x - 4, L.ys[j] - 17 + r * 3, 8, 1.5);   // read through an embedding
      g.globalAlpha = on || hit ? .92 : .35; g.fillStyle = hit ? C.acc : C.fg2;
      g.fillText(f.label, L.x - 14 - L.steps * 4, L.ys[j]);
    });
    if (L.graph) {                                       // a node and its neighbours
      const cx = L.x, cy = L.box.y + L.box.h + 4;
      g.globalAlpha = .6; g.strokeStyle = C.muted; g.lineWidth = 1;
      for (let k = 0; k < 5; k++) { const a = k * 1.2566; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + 9 * Math.cos(a), cy + 9 * Math.sin(a)); g.stroke(); }
    }
  },

  hit(seg, L, x, y) {
    let best = null, bd = 14;
    L.ys.forEach((cy, j) => { const d = Math.abs(cy - y); if (d < bd && x > L.box.x - 10) { bd = d; best = { j }; } });
    return best;
  },

  caption(seg, hit, ctx) {
    const f = ctx.env.families[hit.j];
    if (!f) return el("span");
    const names = f.members.map((n) => data(n)), shape = ctx.graph.sample.shape;
    return el("div", {}, el("div", {}, el("span", { class: "who" }, f.label), f.text ? ` — ${f.text}.` : ""),
      el("div", { class: "members" }, tn("legacy.net.cap.members", names.length, { names: `${names.slice(0, 8).join(", ")}${names.length > 8 ? ", …" : ""}` })),
      shape === "window" ? el("div", { class: "muted" }, t("input.steps", { n: ctx.graph.sample.length })) : null,
      shape === "graph" ? el("div", { class: "muted" }, t("input.neighbours")) : null,
      embeddedFamilies(ctx.graph, [f])[0] ? el("div", { class: "muted" }, t("input.embedded")) : null);
  },

  acts(seg, ex, graph, ctx) { return tippedByFamily(ex, ctx.env.families); },
};
