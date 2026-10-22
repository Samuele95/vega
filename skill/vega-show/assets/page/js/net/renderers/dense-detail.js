// The every-weight drawing of a pure dense chain (the hand-built page's Detail, 0d06aea
// index.html:1094–1300): every input, every unit, every output; for every unit its three strongest
// inputs (six into an output) and its strongest output. For one answer, every unit coloured by what
// it contributed and the connections that carried most of it, ending in the answer. Its words come
// from the strings table, its output colours from the class roles.
import { css, el } from "../../core/dom.js";
import { fmtInt } from "../../core/fmt.js";
import { t, tText, data } from "../../core/text.js";
import { readAnswer, relevance, signedNorm, strongest, topInputs } from "../relevance.js";
import { animateDrawing } from "../canvas.js";

// Every unit's k strongest inputs (six into the outputs) and every input's strongest output.
export function topEdges(mats) {
  return mats.map((m, li) => {
    const k = li === mats.length - 1 ? 6 : 3, edges = [], seen = new Set();
    const add = (i, j) => { const key = i * 65536 + j; if (!seen.has(key)) { seen.add(key); edges.push([i, j, m.w[j * m.in + i]]); } };
    for (let j = 0; j < m.out; j++) {
      const row = []; for (let i = 0; i < m.in; i++) row.push([Math.abs(m.w[j * m.in + i]), i]);
      row.sort((a, b) => b[0] - a[0]);
      for (let s = 0; s < Math.min(k, row.length); s++) add(row[s][1], j);
    }
    for (let i = 0; i < m.in; i++) {
      let best = 0, bw = -1;
      for (let j = 0; j < m.out; j++) { const w = Math.abs(m.w[j * m.in + i]); if (w > bw) { bw = w; best = j; } }
      add(i, best);
    }
    return edges;
  });
}

// labels at their units' heights, pushed apart to `gap` as centred blocks within [lo, hi]
export function spread(items, gap, lo, hi) {
  items.sort((a, b) => a.y - b.y);
  const place = (b) => {
    const h = (b.items.length - 1) * gap, mid = b.items.reduce((a, it) => a + it.y, 0) / b.items.length;
    b.s = Math.max(lo, Math.min(hi - h, mid - h / 2));
  };
  const blocks = items.map((it) => ({ items: [it] })); blocks.forEach(place);
  for (let k = 1; k < blocks.length;) {
    const a = blocks[k - 1], b = blocks[k];
    if (b.s < a.s + a.items.length * gap - .01) { a.items.push(...b.items); place(a); blocks.splice(k, 1); k = 1; } else k++;
  }
  blocks.forEach((b) => b.items.forEach((it, s) => { it.ly = b.s + s * gap; }));
  return items;
}

function fit(g, text, w) {
  if (g.measureText(text).width <= w) return text;
  let lo = 0, hi = text.length;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (g.measureText(text.slice(0, m) + "…").width <= w) lo = m; else hi = m - 1; }
  return lo ? text.slice(0, lo).trimEnd() + "…" : "";
}

// side -1: right-aligned left of the inputs; +1: left of the outputs. A thin leader joins a name to
// its unit when it had to move; a name too long for the margin is cut.
function drawSide(d, items, side) {
  const g = d.g, x0 = d.xs[side < 0 ? 0 : d.last], room = (side < 0 ? d.padL : d.padR) - 17;
  g.font = d.fLab; g.textBaseline = "middle"; g.textAlign = side < 0 ? "right" : "left";
  items.forEach((it) => {
    if (Math.abs(it.ly - it.y) > 1) {
      g.globalAlpha = .6; g.strokeStyle = it.color; g.lineWidth = .8;
      g.beginPath(); g.moveTo(x0 + side * 4, it.y); g.lineTo(x0 + side * 11, it.ly); g.stroke();
    }
    g.globalAlpha = .88; g.fillStyle = it.color; g.fillText(fit(g, it.text, room), x0 + side * 13, it.ly);
  });
}

function drawLines(d, g, C) {
  const L = d.mats.length;
  (d.path || d.top).forEach((edges, l) => {           // each layer's lines grow in turn
    const reach = Math.max(0, Math.min(1, d.phase * L - l)); if (reach <= 0) return;
    edges.forEach(([i, j, w]) => {
      const hot = d.hover && ((d.hover[0] === l && d.hover[1] === i) || (d.hover[0] === l + 1 && d.hover[1] === j));
      g.globalAlpha = d.path ? .1 + .8 * Math.abs(w) : hot ? .9 : Math.min(1, Math.abs(w) / d.maxW * 2) * (d.hover ? .12 : .55);
      g.strokeStyle = w >= 0 ? C.acc : C.neg; g.lineWidth = d.path ? .5 + 1.8 * Math.abs(w) : hot ? 1.2 : .6;
      const x0 = d.xs[l], y0 = d.ys[l][i], x1 = d.xs[l + 1], y1 = d.ys[l + 1][j];
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x0 + (x1 - x0) * reach, y0 + (y1 - y0) * reach); g.stroke();
    });
  });
  if (!d.path || d.phase >= 1) return;                // pulses on the busiest lines of the layer in flight
  const lf = d.phase * L, l = Math.floor(lf), s = lf - l;
  if (l < L) d.path[l].forEach(([i, j, w]) => {
    if (Math.abs(w) < .3) return;
    const x0 = d.xs[l], y0 = d.ys[l][i], x1 = d.xs[l + 1], y1 = d.ys[l + 1][j];
    g.globalAlpha = .9; g.fillStyle = w >= 0 ? C.acc : C.neg;
    g.beginPath(); g.arc(x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, 1.4 + 1.8 * Math.abs(w), 0, 7); g.fill();
  });
}

function drawUnits(d, g, C) {
  const L = d.mats.length;
  d.sizes.forEach((n, l) => {                          // and each column appears as they reach it
    const shown = Math.max(0, Math.min(1, d.phase * L - l + 1)); if (shown <= 0) return;
    d.ys[l].forEach((y, j) => {
      let a = .9, fill = d.hover && d.hover[0] === l && d.hover[1] === j ? C.fg : C.mut, size = 0;
      if (d.acts) {
        const v = d.acts[l][j]; a = .12 + .88 * Math.min(1, Math.abs(v)); size = Math.min(1, Math.abs(v));
        fill = l === d.last ? (d.alert(d.model.outputs[j]) ? C.alert : C.acc) : v >= 0 ? C.acc : C.neg;
      }
      g.globalAlpha = a * shown; g.fillStyle = fill;
      if (n <= 80) { g.beginPath(); g.arc(d.xs[l], y, 2.3 + 2.2 * size, 0, 7); g.fill(); }
      else g.fillRect(d.xs[l] - 3 - 3 * size, y - .6, 6 + 6 * size, 1.3);
      if (d.acts && l === d.last && j === d.winner && d.phase >= 1) {
        g.globalAlpha = .9; g.strokeStyle = fill; g.lineWidth = 1.4; g.beginPath(); g.arc(d.xs[l], y, 9, 0, 7); g.stroke();
      }
    });
  });
}

// column captions: full, then short, then only those that stay clear of their neighbours
function drawCaptions(d, g) {
  g.font = d.fCap; g.fillStyle = css("--muted"); g.globalAlpha = .85; g.textAlign = "center"; g.textBaseline = "middle";
  const hid = d.model.layers.slice(0, -1);
  const put = (ts) => ts.map((s, l) => { const w = g.measureText(s).width; return { s, w, x: Math.max(4 + w / 2, Math.min(d.W - 4 - w / 2, d.xs[l])) }; });
  const clear = (a, b) => a.x + a.w / 2 + 10 <= b.x - b.w / 2, ok = (cs) => cs.every((c, k) => !k || clear(cs[k - 1], c));
  const full = [tText("legacy.net.detail.inputs"), ...hid.map((_, i) => tText("legacy.net.detail.hidden", { i: i + 1 })), tText("legacy.net.detail.outputs")];
  const short = [tText("legacy.net.detail.in"), ...hid.map((_, i) => tText("legacy.net.detail.h", { i: i + 1 })), tText("legacy.net.detail.out")];
  let caps = put(full.map((c, l) => `${c} · ${d.sizes[l]}`));
  if (!ok(caps)) caps = put(short.map((c, l) => `${c} ${d.sizes[l]}`));
  if (!ok(caps)) {
    const keep = [caps[0]];
    caps.slice(1).forEach((c, k) => {
      if (clear(keep[keep.length - 1], c)) keep.push(c);
      else if (k === caps.length - 2 && keep.length > 1) { keep.pop(); if (clear(keep[keep.length - 1], c)) keep.push(c); }
    });
    caps = keep;
  }
  caps.forEach((c) => g.fillText(c.s, c.x, d.H - 10));
  g.globalAlpha = 1;
}

// the names: the outputs (every one when the rows allow it; otherwise the ones that matter), and
// the inputs pointed at or that carried the most signal
function drawNames(d, g, C) {
  const gap = d.gap, lo = 7, hi = d.H - d.bottom + 5, fits = Math.floor((hi - lo) / gap) + 1;
  const hov = (l) => (d.hover && d.hover[0] === l ? d.hover[1] : -1);
  const outs = d.model.outputs, oy = d.ys[d.last], p = d.acts && d.acts[d.last];
  let pick = outs.map((_, j) => j);
  if (outs.length > 1 && oy[1] - oy[0] < gap) {
    const stride = Math.ceil(outs.length / fits);
    pick = pick.map((j) => [j === hov(d.last) ? 4 : p ? (j === d.winner ? 3 : p[j] >= .02 ? 1 + p[j] : 0) : j % stride === 0 ? 1 : 0, j])
      .filter((r) => r[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, fits).map((r) => r[1]);
  }
  drawSide(d, spread(pick.map((j) => ({ y: oy[j], text: data(outs[j]), color: (p && j === d.winner) || j === hov(d.last) ? C.fg : C.fg2 })), gap, lo, hi), 1);
  const named = [...new Set([hov(0), ...(d.acts && d.pathInputs ? d.pathInputs : [])])].filter((i) => i >= 0).slice(0, fits);
  drawSide(d, spread(named.map((i) => ({ y: d.ys[0][i], text: data(d.model.inputs[i]), color: C.acc })), gap, lo, hi), -1);
  drawCaptions(d, g);
}

// model: graph.runModel(seg) of a pure chain (the head included); opts: {roles, caption, isStatic}
export class DenseDetail {
  constructor(model, { roles = null, caption = null, isStatic = false } = {}) {
    Object.assign(this, { model, roles, captionHost: caption, isStatic });
    this.mats = model.layers.map((l) => l.W);
    this.sizes = [model.inputs.length, ...model.layers.map((l) => l.out)];
    this.last = this.sizes.length - 1;
    this.top = topEdges(this.mats);
    this.maxW = Math.max(...this.mats.map((m) => m.w.reduce((a, v) => Math.max(a, Math.abs(v)), 0)));
    Object.assign(this, { phase: 1, hover: null, acts: null, path: null, pathInputs: null, winner: -1, active: false, W: 320, H: 180, mw: null });
  }

  alert(name) { return !!this.roles && this.roles.isAlert(name); }

  // The margins follow the canvas and the longest name drawn there: on the left every input when any
  // can be pointed at, only the named ones in a static answer; on the right every output. Capped, so
  // the network keeps at least half the width.
  layout(g, W, H) {
    Object.assign(this, { g, W: Math.max(320, W), H: Math.max(180, H) });
    const n = this.sizes.length, narrow = this.W < 560, top = 12;
    this.fLab = `${narrow ? 10.5 : 11.5}px ${css("--sans")}`; this.fCap = `${narrow ? 10 : 10.5}px ${css("--mono")}`;
    this.gap = narrow ? 12 : 13; this.bottom = 34;
    if (!this.mw || this.mw.font !== this.fLab) {
      g.font = this.fLab;
      this.mw = { font: this.fLab, ins: this.model.inputs.map((s) => g.measureText(data(s)).width), outs: this.model.outputs.map((s) => g.measureText(data(s)).width) };
    }
    const left = this.isStatic ? (this.acts && this.pathInputs ? [...this.pathInputs] : []) : this.model.inputs.map((_, i) => i);
    const wl = left.reduce((a, i) => Math.max(a, this.mw.ins[i]), 0), wr = Math.max(0, ...this.mw.outs);
    const needL = wl + 22, needR = wr + 22, k = Math.min(1, this.W * .46 / (needL + needR));   // shared out when both cannot fit
    this.padL = Math.round(Math.max(28, needL * k)); this.padR = Math.round(Math.max(28, needR * k));
    this.xs = this.sizes.map((_, i) => this.padL + (this.W - this.padL - this.padR) * i / (n - 1));
    this.ys = this.sizes.map((s) => Array.from({ length: s }, (_, j) => top + (this.H - top - this.bottom) * (s === 1 ? .5 : j / (s - 1))));
  }

  draw() {
    const g = this.g, C = { acc: css("--accent"), neg: css("--neg"), mut: css("--muted2"), fg: css("--fg"), fg2: css("--fg2"), alert: css("--alert") };
    this.layout(g, this.W, this.H);                     // the named inputs may have changed
    g.clearRect(0, 0, this.W, this.H);
    drawLines(this, g, C); drawUnits(this, g, C);
    if (this.phase >= 1) drawNames(this, g, C);
    g.globalAlpha = 1;
  }

  // Every unit coloured by what it contributed to this answer; the lines are the connections that
  // carried most of it, into the answer at the end.
  showActivations(ex) {
    const answer = readAnswer(ex, this.model.outputs);
    if (!answer) return false;
    const R = relevance(answer, this.mats);
    this.acts = [signedNorm(R.inputs || answer.inputs.map(() => 0)), ...R.hidden.map(signedNorm), R.probs];
    this.winner = R.win;
    this.path = strongest(R, this.mats).map((top) => { const mx = top.length ? Math.abs(top[0][2]) : 1; return top.map(([i, j, c]) => [i, j, c / mx]); });
    this.pathInputs = new Set(topInputs(R, 6));
    return true;
  }

  point(x, y) {
    if (this.phase < 1) return false;
    let best = null, bd = 12;
    this.xs.forEach((cx, l) => { if (Math.abs(cx - x) > 12) return; this.ys[l].forEach((cy, j) => { const dd = Math.abs(cy - y); if (dd < bd) { bd = dd; best = [l, j]; } }); });
    this.hover = best; this.draw();
    this.caption(best);
    return true;
  }
  leave() { return false; }
  picked() { return null; }

  caption(best) {
    const cap = this.captionHost; if (!cap) return;
    cap.replaceChildren();
    if (!best) { cap.append(el("span", { class: "muted" }, t("legacy.net.detail.cap", { n: fmtInt(this.model.params) }))); return; }
    const [l, j] = best;
    const nm = l === 0 ? data(this.model.inputs[j]) : l === this.last ? data(this.model.outputs[j]) : tText("legacy.net.detail.unit", { j: j + 1, l });
    cap.append(el("span", { class: "who" }, nm));
  }

  animate(ms, done) { animateDrawing(this, ms, done); }
}
