// A stage drawn as one box (D4 §7.2): a normalisation, positions, a pooling, a reshape, a join, an
// embedding, a bend or a dropout standing alone, an edge dropout, a user's custom block. The box
// carries a small glyph of what the stage does, its name (the explain catalog's label when the page
// has it, else the strings table's word for its kind) and its shape change; a stage that does
// nothing at answer time is dimmed and says so; a custom block wears the unverified chip. One answer
// lights the box by the size of what it passed on.
import { css, el } from "../../core/dom.js";
import { t, tText, tn } from "../../core/text.js";
import { stageValues } from "../relevance.js";
import { tensorValues } from "../graph.js";
import { stageLabel as boxLabel } from "../card.js";

const TRAIN_ONLY = Object.freeze(["dropout", "edge_dropout"]);
const shape = (s) => (Array.isArray(s) && s.length ? s.join("×") : "·");

// the glyph of a kind, drawn in a w × h cell at (x, y)
const GLYPHS = Object.freeze({
  activation: (g, x, y, w, h) => { g.beginPath(); g.moveTo(x, y + h); g.lineTo(x + w / 2, y + h); g.lineTo(x + w, y); g.stroke(); },
  positional: (g, x, y, w, h) => { g.beginPath(); for (let k = 0; k <= 24; k++) g.lineTo(x + w * k / 24, y + h / 2 - h / 2 * Math.sin(k / 24 * 6.283)); g.stroke(); },
  norm: (g, x, y, w, h) => { g.beginPath(); for (let k = 0; k <= 24; k++) { const u = (k / 12 - 1) * 2.4; g.lineTo(x + w * k / 24, y + h - h * Math.exp(-u * u / 2)); } g.stroke(); },
  pool: (g, x, y, w, h) => { for (let k = 0; k < 4; k++) g.fillRect(x + k * w / 4, y + h * (1 - (k + 1) / 5), w / 5, h * (k + 1) / 5); },
  reshape: (g, x, y, w, h) => { for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) g.fillRect(x + c * w / 3, y + r * h / 3, w / 4, h / 4); },
  concat: (g, x, y, w, h) => { g.beginPath(); g.moveTo(x, y); g.lineTo(x + w / 2, y + h / 2); g.lineTo(x, y + h); g.moveTo(x + w / 2, y + h / 2); g.lineTo(x + w, y + h / 2); g.stroke(); },
  embedding: (g, x, y, w, h) => { for (let r = 0; r < 4; r++) g.fillRect(x, y + r * h / 4, w, h / 6); },
  dropout: (g, x, y, w, h) => { g.setLineDash([2, 3]); g.strokeRect(x, y, w, h); g.setLineDash([]); },
  custom: (g, x, y, w, h) => { g.strokeRect(x, y, w, h); g.beginPath(); g.moveTo(x, y); g.lineTo(x + w, y + h); g.stroke(); },
});

function lit(st, stage) { return st.focus.node(stage.id); }

export default {
  kinds: ["embedding", "concat", "activation", "dropout", "norm", "positional", "reshape", "pool", "edge_dropout", "custom"],
  detail: false,

  measure(seg, ctx) { return { w: ctx.boxW, minH: 60 }; },

  layout(seg, box, ctx) {
    const h = Math.min(90, Math.max(44, box.h * .35)), y = box.y + (box.h - h) / 2, stage = ctx.graph.byId(seg.stage);
    const rect = { x: box.x, y, w: box.w, h };
    return { box, rect, stage, label: boxLabel(stage), in: [{ x: box.x, y: y + h / 2 }], out: [{ x: box.x + box.w, y: y + h / 2 }] };
  },

  draw(g, seg, L, st) {
    const C = st.ctx.colors, s = L.stage, r = L.rect, quiet = TRAIN_ONLY.includes(s.kind), on = lit(st, s);
    const act = st.acts && st.acts[st.i], hot = st.hover && st.hover.i === st.i;
    g.globalAlpha = (on ? 1 : .35) * (quiet ? .55 : 1);
    g.fillStyle = act !== null && act !== undefined ? C.acc : C.bg; g.strokeStyle = hot ? C.fg : st.focus.active && !st.focus.all && on ? C.acc : C.line;
    if (act) { g.globalAlpha = .12 + .5 * act.glow; g.fillRect(r.x, r.y, r.w, r.h); g.globalAlpha = on ? 1 : .35; }
    g.lineWidth = hot ? 1.6 : 1; if (quiet) g.setLineDash([3, 3]);
    g.strokeRect(r.x + .5, r.y + .5, r.w - 1, r.h - 1); g.setLineDash([]);
    const glyph = GLYPHS[s.kind === "edge_dropout" ? "dropout" : s.kind];
    g.strokeStyle = C.fg2; g.fillStyle = C.fg2; g.lineWidth = 1.2;
    if (glyph) glyph(g, r.x + r.w * .3, r.y + 8, r.w * .4, r.h * .38);
    g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = C.fg2; g.font = `11px ${css("--sans")}`;
    g.fillText(L.label, r.x + r.w / 2, r.y + r.h * .72, r.w - 6);
    g.fillStyle = C.muted; g.font = `10px ${css("--mono")}`;
    g.fillText(tText("box.shape", { from: shape(s.in_shape), to: shape(s.out_shape) }), r.x + r.w / 2, r.y + r.h + 12, r.w + 20);
    if (quiet) g.fillText(tText("box.note.train_only"), r.x + r.w / 2, r.y - 10, r.w + 30);
    if (s.kind === "custom") { g.fillStyle = C.alert; g.fillText(tText("box.unverified"), r.x + r.w / 2, r.y - 10, r.w + 30); }
    g.globalAlpha = 1;
  },

  hit(seg, L, x, y) {
    const r = L.rect;
    return x >= r.x - 4 && x <= r.x + r.w + 4 && y >= r.y - 14 && y <= r.y + r.h + 16 ? { stage: seg.stage } : null;
  },

  caption(seg, hit, ctx) {
    const s = ctx.graph.byId(seg.stage);
    return el("div", {}, el("div", {}, el("span", { class: "who" }, boxLabel(s)), el("span", { class: "muted" }, " ",
      t("box.shape", { from: shape(s.in_shape), to: shape(s.out_shape) }))),
      s.formula ? el("div", { class: "formula" }, s.formula) : null,
      TRAIN_ONLY.includes(s.kind) ? el("div", { class: "muted" }, t("box.note.train_only")) : null,
      s.params ? el("div", { class: "muted" }, tn("card.stage_params", s.params)) : null);
  },

  // One answer: the root mean square of what the stage passed on, as a glow in [0, 1).
  acts(seg, ex) {
    const v = stageValues(ex, seg.stage, tensorValues);
    if (!v || !v.length) return null;
    const rms = Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length);
    return { glow: Math.tanh(rms), rms };
  },
};
