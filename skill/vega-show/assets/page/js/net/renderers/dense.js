// The families drawing of a dense run (D4 §7.2–7.3; the hand-built page's Diagram, 0d06aea
// index.html:819–1091): input families → the hidden units that matter → the outputs. For a pure
// dense chain the whole canvas is one DenseRun, drawn exactly as today's page drew it; its words
// come from the strings table, its output colours from the class roles. Inside a larger graph a
// run is a segment of the canvas (run-segment.js). The "every weight" level is dense-detail.js.
//
// The choices of what to draw are pure functions, so the unit tests hold them to today's page:
//   keepByWeights   the K units of each hidden layer that both listen and speak the most
//   edgesByWeight   every drawn node's 3 strongest inputs and its strongest output
//   answerDrawing   one answer: the K units that carried it most, the connections that carried it
import { css, el } from "../../core/dom.js";
import { t, tText, tn, data } from "../../core/text.js";
import { Focus } from "../graph.js";
import { readAnswer, relevance, signedNorm } from "../relevance.js";
import { animateDrawing, colours } from "../canvas.js";
import { DenseDetail } from "./dense-detail.js";
import runSegment from "./run-segment.js";

const norm2 = (arr) => Math.sqrt(arr.reduce((a, v) => a + v * v, 0));

// Per hidden layer h (between mats[h-1] and mats[h]): the K units with the largest ‖row‖·‖column‖.
export function keepByWeights(mats, K) {
  const keep = [];
  for (let h = 1; h < mats.length; h++) {
    const inM = mats[h - 1], outM = mats[h], score = [];
    for (let j = 0; j < inM.out; j++) {
      const a = norm2(Array.from(inM.w.subarray(j * inM.in, (j + 1) * inM.in)));
      let b = 0; for (let o = 0; o < outM.out; o++) b += outM.w[o * outM.in + j] ** 2;
      score.push([a * Math.sqrt(b), j]);
    }
    score.sort((p, q) => q[0] - p[0]);
    keep.push(score.slice(0, Math.min(K, score.length)).map((s) => s[1]).sort((a, b) => a - b));
  }
  return keep;
}

// The node columns: families, the kept units of each hidden layer, the outputs.
export const columnsOf = (fam, keep, outputs) => [fam.map((g) => ({ kind: "family", g })),
  ...keep.map((ks, h) => ks.map((j) => ({ kind: "unit", layer: h + 1, j }))), outputs.map((c, j) => ({ kind: "class", j, c }))];

// The weight between two drawn nodes; a family's is the combined pull of its members.
export function weightOf(mats, cols, [l, a, b]) {
  const m = mats[l], src = cols[l][a], dst = cols[l + 1][b];
  if (src.kind === "family") {
    let s = 0, q = 0;
    for (const i of src.g.idx) { const w = m.w[dst.j * m.in + i]; s += w; q += w * w; }
    return Math.sign(s || 1) * Math.sqrt(q);
  }
  return m.w[dst.j * m.in + src.j];
}

// For every drawn node: its strongest incoming connections and its strongest outgoing one, so no
// node looks like a dead end. Per layer: [{a, b, w, n}] (n: |w| over the layer's largest).
export function edgesByWeight(mats, cols) {
  const E = [];
  for (let l = 0; l < mats.length; l++) {
    const list = [], seen = new Set(), wt = (a, b) => weightOf(mats, cols, [l, a, b]);
    const add = (a, b) => { const k = a + ":" + b; if (!seen.has(k)) { seen.add(k); list.push([a, b, wt(a, b)]); } };
    cols[l + 1].forEach((_, b) => {
      const cand = cols[l].map((_, a) => [a, Math.abs(wt(a, b))]);
      cand.sort((p, q) => q[1] - p[1]);
      cand.slice(0, 3).forEach(([a]) => add(a, b));
    });
    cols[l].forEach((_, a) => {
      let best = 0, bw = -1;
      cols[l + 1].forEach((_, b) => { const w = Math.abs(wt(a, b)); if (w > bw) { bw = w; best = b; } });
      add(a, best);
    });
    const mx = Math.max(1e-9, ...list.map((e) => Math.abs(e[2])));
    E.push(list.map(([a, b, w]) => ({ a, b, w, n: Math.abs(w) / mx })));
  }
  return E;
}

// One answer (R: relevance()): the K units per hidden layer that carried most of it, the
// connections that carried it (into the answer only, on the last layer), and every node's share.
export function answerDrawing(R, fam, outputs, K) {
  const keep = R.hidden.map((r) => r.map((v, j) => [Math.abs(v), j]).sort((a, b) => b[0] - a[0])
    .slice(0, Math.min(K, r.length)).map((q) => q[1]).sort((a, b) => a - b));
  const cols = columnsOf(fam, keep, outputs), L = cols.length - 1;
  const carried = (l, a, b) => {
    const src = cols[l][a], dst = cols[l + 1][b];
    return src.kind === "family" ? src.g.idx.reduce((s, i) => s + R.edge(0, i, dst.j), 0) : R.edge(l, src.j, dst.j);
  };
  const edges = cols.slice(0, -1).map((col, l) => {
    const list = [], last = l === L - 1;
    cols[l + 1].forEach((_, b) => {
      if (last && b !== R.win) return;
      const cand = col.map((_, a) => [a, b, carried(l, a, b)]).sort((p, q) => Math.abs(q[2]) - Math.abs(p[2]));
      list.push(...cand.slice(0, last ? 6 : 3));
    });
    const mx = Math.max(1e-12, ...list.map((e) => Math.abs(e[2])));
    return list.map(([a, b, w]) => ({ a, b, w, n: Math.abs(w) / mx }));
  });
  const famRel = signedNorm(fam.map((g) => g.idx.reduce((s, i) => s + (R.inputs ? R.inputs[i] : 0), 0)));
  const acts = [famRel, ...keep.map((ks, h) => { const all = signedNorm(R.hidden[h]); return ks.map((j) => all[j]); }), R.probs];
  return { keep, cols, edges, acts, winner: R.win };
}

const verb = (v) => tText(v >= 0 ? "legacy.net.verb.raises" : "legacy.net.verb.lowers");

// ---- drawing helpers (d: a DenseRun)
function drawEdges(d, g, C) {
  for (let l = 0; l < d.L; l++) {
    const r = Math.max(0, Math.min(1, d.phase * d.L - l)); if (r <= 0) continue;
    for (const e of d.edges[l]) {
      const [x0, y0] = [d.xs[l], d.ys[l][e.a]], [x1, y1] = [d.xs[l + 1], d.ys[l + 1][e.b]];
      let alpha, width = .9;                     // thin lines, strength carried by opacity alone
      if (d.carried) { const f = d.carried[l].get(e.a + ":" + e.b) || 0; alpha = .1 + .8 * f; width = .5 + 1.8 * f; }
      else alpha = d.edgeLit(l, e) ? Math.min(1, e.n * 2) * .55 : .04;
      if (d.hover) { const hot = d.touches(l, e); alpha = hot ? .9 : alpha * .22; if (hot) width = 1.2; }
      const mid = (x0 + x1) / 2;
      g.globalAlpha = alpha; g.strokeStyle = e.w >= 0 ? C.acc : C.neg; g.lineWidth = width;
      g.beginPath(); g.moveTo(x0, y0);
      if (r < 1) g.lineTo(x0 + (x1 - x0) * r, y0 + (y1 - y0) * r); else g.bezierCurveTo(mid, y0, mid, y1, x1, y1);
      g.stroke();
    }
  }
}

// pulses travelling along the busiest connections, one layer at a time
function drawPulses(d, g, C) {
  if (!d.carried || d.phase >= 1) return;
  const lf = d.phase * d.L, l = Math.floor(lf), s = lf - l;
  if (l >= d.L) return;
  for (const e of d.edges[l]) {
    const f = d.carried[l].get(e.a + ":" + e.b) || 0; if (f < .35) continue;
    const [x0, y0] = [d.xs[l], d.ys[l][e.a]], [x1, y1] = [d.xs[l + 1], d.ys[l + 1][e.b]], mid = (x0 + x1) / 2, u = 1 - s;
    const px = u * u * u * x0 + 3 * u * u * s * mid + 3 * u * s * s * mid + s * s * s * x1;
    const py = u * u * u * y0 + 3 * u * u * s * y0 + 3 * u * s * s * y1 + s * s * s * y1;
    g.globalAlpha = .9; g.fillStyle = e.w >= 0 ? C.acc : C.neg;
    g.beginPath(); g.arc(px, py, 2 + 2 * f, 0, 7); g.fill();
  }
}

function nodeStyle(d, [l, j], C) {
  const node = d.cols[l][j], lit = d.nodeLit(l, j), feature = !!d.focus.feature;
  let r = node.kind === "family" ? 7 : node.kind === "class" ? 5.5 : 4.5, fill = C.mut, a = lit ? .95 : .25, ring = null;
  if (node.kind === "family" && d.famHit(j)) { fill = C.acc; a = 1; }
  if (d.focus.active && !d.focus.all && lit && !feature) fill = C.acc;
  if (d.acts) {
    const v = d.acts[l][j];
    a = .2 + .8 * Math.min(1, Math.abs(v)); fill = v >= 0 ? C.acc : C.neg; r += 2.5 * Math.min(1, Math.abs(v));
    if (node.kind === "class") { fill = d.alert(node.c) ? C.alert : C.acc; if (d.winner === j && d.phase >= 1) ring = fill; }
  }
  if (d.hover && d.hover[0] === l && d.hover[1] === j) { fill = C.fg; a = 1; r += 1.5; }
  return { r, fill, a, ring };
}

function drawNodes(d, g, C) {
  for (let l = 0; l <= d.L; l++) {
    const shown = Math.max(0, Math.min(1, d.phase * d.L - l + 1)); if (shown <= 0) continue;
    d.cols[l].forEach((_, j) => {
      const x = d.xs[l], y = d.ys[l][j], s = nodeStyle(d, [l, j], C);
      g.globalAlpha = s.a * shown; g.fillStyle = s.fill;
      g.beginPath(); g.arc(x, y, s.r, 0, 7); g.fill();
      if (s.ring) { g.globalAlpha = .9; g.strokeStyle = s.ring; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, s.r + 5, 0, 7); g.stroke(); }
    });
  }
}

function drawLabels(d, g, C) {
  g.textBaseline = "middle"; g.font = `12.5px ${css("--sans")}`;
  g.textAlign = "right";                                 // families, left
  d.cols[0].forEach((node, j) => {
    let a = d.nodeLit(0, j) ? .92 : .35, col = C.fg2;
    if (d.famHit(j)) { a = 1; col = C.acc; }
    if (d.acts) a = .35 + .65 * Math.min(1, Math.abs(d.acts[0][j]));
    g.globalAlpha = a; g.fillStyle = col; g.fillText(node.g.label, d.xs[0] - 14, d.ys[0][j]);
  });
  g.textAlign = "left";                                  // outputs, right: where they fit, or those that score
  const fits = (d.H - 54) / Math.max(1, d.cols[d.L].length - 1) >= 13, lit = d.nodeLit(d.L, 0) && !d.focus.feature;
  d.cols[d.L].forEach((node, j) => {
    let a = lit || !d.focus.active ? .85 : .35, col = C.fg2;
    if (d.acts) { const p = d.acts[d.L][j]; if (!fits && p < .02) return; a = .3 + .7 * p; if (d.winner === j) col = d.alert(node.c) ? C.alert : C.acc; }
    else if (!fits) return;
    g.globalAlpha = a; g.fillStyle = col; g.fillText(data(node.c), d.xs[d.L] + 14, d.ys[d.L][j]);
  });
  g.globalAlpha = .9; g.fillStyle = C.muted; g.textAlign = "center"; g.font = `10.5px ${css("--mono")}`;
  d.captions().forEach((c, l) => g.fillText(c, d.xs[l], d.H - 10));
  g.globalAlpha = 1;
}

// The families drawing of a pure dense chain. model: graph.runModel(seg) (the head included);
// opts: {families, k, roles, caption (host node), waist (hidden layer index or null)}.
export class DenseRun {
  constructor(model, { families, k = 14, roles = null, caption = null, waist = null }) {
    Object.assign(this, { model, fam: families, K: k, roles, captionHost: caption, waist });
    this.mats = model.layers.map((l) => l.W); this.L = this.mats.length;
    this.keep = keepByWeights(this.mats, this.K);
    this.cols = columnsOf(this.fam, this.keep, model.outputs);
    this.edges = edgesByWeight(this.mats, this.cols);
    Object.assign(this, { focus: new Focus(null), hover: null, acts: null, carried: null, phase: 1, winner: -1, active: true, W: 320, H: 180 });
  }

  // The column ids a focus names: "input", the hidden layers' stage ids, "head".
  colId(l) { return l === 0 ? "input" : l === this.L ? "head" : this.model.layers[l - 1].id; }
  nodeLit(l) { return this.focus.node(this.colId(l)); }
  edgeLit(l, e) {
    if (this.focus.feature) return l === 0 && this.famHit(e.a);
    return this.focus.edge(this.colId(l), this.colId(l + 1));
  }
  famHit(a) { const f = this.focus.feature; return f !== null && this.cols[0][a].g.members.includes(f); }
  alert(name) { return !!this.roles && this.roles.isAlert(name); }
  touches(l, e) { const h = this.hover; return (h[0] === l && h[1] === e.a) || (h[0] === l + 1 && h[1] === e.b); }

  layout(g, W, H) {
    Object.assign(this, { g, W: Math.max(320, W), H: Math.max(180, H) });
    const padL = this.W < 700 ? 120 : 170, padR = this.W < 700 ? 150 : 200, top = 18, bottom = 36, n = this.cols.length;
    this.xs = this.cols.map((_, i) => padL + (this.W - padL - padR) * i / (n - 1));
    this.ys = this.cols.map((col) => col.map((_, j) => top + (this.H - top - bottom) * (col.length === 1 ? .5 : j / (col.length - 1))));
  }

  captions() {
    const hidden = this.keep.map((ks, h) => {
      const layer = this.model.layers[h], key = h === this.waist ? "net.caption.bottleneck" : layer.group ? "net.caption.decoder" : "net.caption.hidden";
      return tText(key, { h: h + 1, k: ks.length, n: layer.out });
    });
    return [tText("net.caption.inputs"), ...hidden, tText("head.caption")];
  }

  draw() {
    const g = this.g, C = colours();
    g.clearRect(0, 0, this.W, this.H);
    drawEdges(this, g, C); drawPulses(this, g, C); drawNodes(this, g, C);
    g.globalAlpha = 1;
    drawLabels(this, g, C);
  }

  hitAt(x, y) {
    let best = null, bd = 14;
    this.xs.forEach((cx, l) => {
      const labelZone = (l === 0 && x < cx && x > cx - 160) || (l === this.L && x > cx && x < cx + 190);
      if (Math.abs(cx - x) > 12 && !labelZone) return;
      this.ys[l].forEach((cy, j) => { const d = Math.abs(cy - y); if (d < bd) { bd = d; best = [l, j]; } });
    });
    return best;
  }

  // pointer at (x, y): the node under it lit, its caption written; true when the hover changed
  point(x, y) {
    const best = this.hitAt(x, y), same = best && this.hover && best[0] === this.hover[0] && best[1] === this.hover[1];
    if (same || (!best && !this.hover)) return false;
    this.hover = best; this.draw(); this.caption(best);
    return true;
  }
  leave() { this.hover = null; this.draw(); this.caption(null); }
  picked() { return this.hover ? this.cols[this.hover[0]][this.hover[1]] : null; }

  name(l, a) {
    const n = this.cols[l][a];
    return n.kind === "family" ? n.g.label : n.kind === "class" ? data(n.c) : tText("legacy.net.unit", { j: n.j + 1 });
  }
  strongestOf(l, a, dir) {            // among the drawn nodes, by the same weights the lines use
    const out = [], w = (args) => weightOf(this.mats, this.cols, args);
    if (dir === "in" && l > 0) this.cols[l - 1].forEach((_, s) => out.push([s, w([l - 1, s, a])]));
    if (dir === "out" && l < this.L) this.cols[l + 1].forEach((_, d) => out.push([d, w([l, a, d])]));
    return out.sort((p, q) => Math.abs(q[1]) - Math.abs(p[1])).slice(0, 2);
  }
  caption(h) { if (this.captionHost) this.captionHost.replaceChildren(...captionNodes(this, h)); }

  animate(ms, done) { animateDrawing(this, ms, done); }

  // The drawing of one answer: the hidden units that carried it and the connections along which it
  // travelled towards the answer (accent) or against it (the negative colour).
  showActivations(ex) {
    const answer = readAnswer(ex, this.model.outputs);
    if (!answer) return false;
    const R = relevance(answer, this.mats), A = answerDrawing(R, this.fam, this.model.outputs, this.K);
    Object.assign(this, { keep: A.keep, cols: A.cols, edges: A.edges, acts: A.acts, winner: A.winner });
    this.carried = this.edges.map((list) => new Map(list.map((e) => [e.a + ":" + e.b, e.n])));
    this.layout(this.g, this.W, this.H);
    return true;
  }
}

function captionNodes(d, h) {
  if (!h) return [el("span", { class: "muted" }, t("legacy.net.cap.hint"))];
  const [l, a] = h, node = d.cols[l][a];
  const say = (pairs, dl) => pairs.map(([i, w]) => tText("legacy.net.cap.pair", { name: d.name(dl, i), verb: verb(w) })).join(", ");
  if (node.kind === "family") {
    const names = node.g.members.map((n) => data(n));
    return [el("div", {}, el("span", { class: "who" }, node.g.label), node.g.text ? ` — ${node.g.text}.` : ""),
      el("div", {}, t("legacy.net.cap.feeds", { list: say(d.strongestOf(0, a, "out"), 1) })),
      el("div", { class: "members" }, tn("legacy.net.cap.members", names.length, { names: `${names.slice(0, 8).join(", ")}${names.length > 8 ? ", …" : ""}` }))];
  }
  if (node.kind === "unit") {
    return [el("div", {}, el("span", { class: "who" }, t("legacy.net.cap.unit", { j: node.j + 1 })), el("span", { class: "muted" }, " ", t("legacy.net.cap.unit_of", { n: d.model.layers[node.layer - 1].out, l: node.layer }))),
      el("div", {}, t("legacy.net.cap.listens", { list: say(d.strongestOf(l, a, "in"), l - 1) })),
      el("div", {}, t("legacy.net.cap.pushes", { list: say(d.strongestOf(l, a, "out"), l + 1) }))];
  }
  return [el("div", {}, el("span", { class: "who" }, data(node.c)), el("span", { class: "muted" }, " ", t("legacy.net.cap.class"))),
    el("div", {}, t("legacy.net.cap.support", { list: say(d.strongestOf(l, a, "in"), l - 1) }))];
}

// The renderer (D4 §7.3): a dense run inside a larger graph, and the pure chain's two drawings.
export default {
  kinds: ["dense", "decoder"], detail: true,
  Run: DenseRun, Detail: DenseDetail,
  ...runSegment,
};
