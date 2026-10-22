// A recurrent step (D4 §8.1, recurrent; explain-catalog.json#recompute.{rnn,gru,lstm}_step): the
// page unrolls the layer itself, from a zero state, step by step and for every layer and direction,
// with the weights it was sent (torch's gate order: GRU r, z, n; LSTM i, f, g, o), and shows one
// step of one unit — the one that carried most of the explained target, or the one picked:
//
//   RNN   h′ = tanh(W_ih·x + b_ih + W_hh·h + b_hh)            (ReLU when the layer says so)
//   GRU   r = σ(·), z = σ(·), n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn)), h′ = (1 − z) ⊙ n + z ⊙ h
//   LSTM  i, f, o = σ(·), g = tanh(·), c′ = f ⊙ c + i ⊙ g, h′ = o ⊙ tanh(c′)
//
// beside the network's own state at that step (the answer's `hidden`, per layer and direction,
// detail "exact"), and what the layer passes on (`return`: its last state — a bidirectional layer's
// backward direction ends at the first step —, every state, their mean or their maximum) beside the
// stage's output. The largest difference over the whole unroll is said, never hidden.
import { el } from "../../core/dom.js";
import { t, tText, data } from "../../core/text.js";
import { plain } from "../../core/fmt.js";
import { sigmoid } from "../act.js";
import { Answer } from "../answer.js";
import { rankUnits, gapOf, checkRows, stageParams, stageProse, stageFormula } from "../sum.js";
import { sizesOf, tensorsOf, fieldsOf, stopped, checkOf, worstOf, positionName, cellName, cellPicks, affine, rowOf, sizeOf } from "../cells.js";

export const GATES = Object.freeze({ rnn: Object.freeze(["h"]), gru: Object.freeze(["r", "z", "n"]), lstm: Object.freeze(["i", "f", "g", "o"]) });
const RETURNS = Object.freeze(["last", "all", "mean", "max"]);

// The layer's configuration, read off its node.
export function configOf(n) {
  const p = n.payload || {}, c = n.config || {};
  const cell = GATES[p.cell] ? p.cell : GATES[n.type] ? n.type : null;
  const ret = String(p.return ?? c.return ?? "last");
  return { cell, H: Number(p.hidden ?? c.hidden), layers: Number(p.layers ?? c.num_layers ?? 1), dirs: Number(p.directions ?? (c.bidirectional ? 2 : 1)),
           ret: RETURNS.includes(ret) ? ret : "last", relu: (c.nonlinearity ?? p.nonlinearity) === "relu", per: (p.per_layer || []).slice() };
}

// One cell step: the gates and the new state, from the input part a = W_ih·x + b_ih and the state
// part b = W_hh·h + b_hh (both G·H long), the previous state h and cell c.
export function cellStep(cfg, a, b, h, c) {
  const H = cfg.H, at = (k, u) => k * H + u, gates = {}, hNew = new Array(H), cNew = cfg.cell === "lstm" ? new Array(H) : null;
  GATES[cfg.cell].forEach((g) => { gates[g] = new Array(H); });
  for (let u = 0; u < H; u++) {
    if (cfg.cell === "rnn") {
      const s = a[u] + b[u];
      hNew[u] = gates.h[u] = cfg.relu ? Math.max(0, s) : Math.tanh(s);
    } else if (cfg.cell === "gru") {
      const r = sigmoid(a[at(0, u)] + b[at(0, u)]), z = sigmoid(a[at(1, u)] + b[at(1, u)]), nn = Math.tanh(a[at(2, u)] + r * b[at(2, u)]);
      gates.r[u] = r; gates.z[u] = z; gates.n[u] = nn;
      hNew[u] = (1 - z) * nn + z * h[u];
    } else {
      const i = sigmoid(a[at(0, u)] + b[at(0, u)]), f = sigmoid(a[at(1, u)] + b[at(1, u)]), g = Math.tanh(a[at(2, u)] + b[at(2, u)]), o = sigmoid(a[at(3, u)] + b[at(3, u)]);
      gates.i[u] = i; gates.f[u] = f; gates.g[u] = g; gates.o[u] = o;
      cNew[u] = f * c[u] + i * g;
      hNew[u] = o * Math.tanh(cNew[u]);
    }
  }
  return { gates, h: hNew, c: cNew };
}

// One direction of one layer over the inputs X ([L][F]): per step {gates, h, c, prev, prevC}.
function runDirection(cfg, w, X, reverse) {
  const L = X.length, G = GATES[cfg.cell].length * cfg.H, F = X.length ? X[0].length : 0, out = new Array(L);
  let h = new Array(cfg.H).fill(0), c = new Array(cfg.H).fill(0);
  for (let k = 0; k < L; k++) {
    const tt = reverse ? L - 1 - k : k, s = cellStep(cfg, affine(w.w_ih, G, F, X[tt], w.b_ih || null), affine(w.w_hh, G, cfg.H, h, w.b_hh || null), h, c);
    out[tt] = { ...s, prev: h, prevC: c };
    h = s.h; c = s.c || c;
  }
  return out;
}

// The whole unroll: states[layer][direction][step]; layer l > 0 reads both directions of layer
// l − 1, side by side (torch's layout).
export function unroll(cfg, weights, X) {
  const states = [];
  let input = X;
  for (let l = 0; l < cfg.layers; l++) {
    const dirs = Array.from({ length: cfg.dirs }, (_, d) => runDirection(cfg, weights[l * cfg.dirs + d], input, d === 1));
    states.push(dirs);
    input = input.map((_, tt) => dirs.flatMap((s) => s[tt].h));
  }
  return states;
}

// What the layer passes on, from the top layer's states (RecurrentBlock's `return`).
export function returned(cfg, top) {
  const L = top[0].length, H = cfg.H;
  if (cfg.ret === "all") return Array.from({ length: L }, (_, tt) => top.flatMap((s) => s[tt].h)).flat();
  if (cfg.ret === "last") return top.flatMap((s, d) => s[d === 0 ? L - 1 : 0].h);
  return top.flatMap((s) => Array.from({ length: H }, (_, u) => (cfg.ret === "mean" ? s.reduce((a, q) => a + q.h[u], 0) / L : Math.max(...s.map((q) => q.h[u])))));
}

// The (direction, step, unit) of the top layer an output unit j reads.
export function cellOfOutput(cfg, top, j) {
  const H = cfg.H, L = top[0].length;
  if (cfg.ret === "all") { const r = j % (H * cfg.dirs); return { d: Math.floor(r / H), t: Math.floor(j / (H * cfg.dirs)), u: r % H }; }
  const d = Math.floor(j / H), u = j % H;
  if (cfg.ret === "max") return { d, u, t: top[d].reduce((b, q, tt) => (q.h[u] > top[d][b].h[u] ? tt : b), 0) };
  return { d, u, t: d === 0 ? L - 1 : 0 };
}

// The per-layer weights the node names, in (layer, direction) order: {list, lost}.
function weightsOf(graph, cfg) {
  const order = cfg.per.slice().sort((a, b) => a.layer - b.layer || (a.direction === "reverse") - (b.direction === "reverse"));
  const lost = [], list = order.map((e) => {
    const got = tensorsOf(graph, { w_ih: e.w_ih, w_hh: e.w_hh, b_ih: e.b_ih, b_hh: e.b_hh });
    lost.push(...got.lost);
    return { ...got.w, shape: got.shape };
  });
  return { list, lost };
}

// The numbers of one recurrent step (see the header), or why they cannot be recomputed.
export function recurrentValues(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage), cfg = configOf(n);
  const { have, missing } = fieldsOf(A, graph, n.id, ["in", "out"]), hidden = A.values(n.id, "hidden");
  const base = { A, n, cfg, x: have.in, y: have.out, hidden, R: A.unitRelevance(n.id) };
  if (!cfg.cell || cfg.per.length !== cfg.layers * cfg.dirs) return { ...base, why: "fit", what: sizesOf(n.id, { per_layer: cfg.per.length }) };
  const { list, lost } = weightsOf(graph, cfg);
  if (lost.length) return { ...base, why: "weights", what: lost.join(", ") };
  if (missing) return { ...base, why: "missing", what: missing };
  const [L, F] = n.in_shape || [], G = GATES[cfg.cell].length * cfg.H;
  const fits = base.x.length === L * F && list[0].shape.w_ih[0] === G && list[0].shape.w_ih[1] === F && base.y.length === sizeOf(n.out_shape)
    && (!hidden || hidden.length === cfg.layers * cfg.dirs * L * cfg.H);
  if (!fits) return { ...base, why: "fit", what: sizesOf(n.id, { in: base.x, out: base.y, hidden }) };
  const states = unroll(cfg, list, Array.from({ length: L }, (_, tt) => rowOf(base.x, F, tt)));
  return { ...base, L, F, states, top: states[states.length - 1] };
}

// The network's state of (layer, direction, step, unit), or null without `hidden`.
const netState = (v, l, d, tt, u) => (v.hidden ? v.hidden[((l * v.cfg.dirs + d) * v.L + tt) * v.cfg.H + u] : null);

// Every state of the unroll, page and network, flat in the answer's `hidden` order.
function allStates(v) {
  if (!v.hidden) return null;
  const page = v.states.flatMap((dirs) => dirs.flatMap((s) => s.flatMap((q) => q.h)));
  return worstOf("states", page, v.hidden);
}

function shown(step, v) {
  const j = rankUnits(v.R, v.y, step.pick).j, at = cellOfOutput(v.cfg, v.top, j);
  return { j, ...at, q: v.top[at.d][at.t] };
}

function recompute(step, ex, graph) {
  const v = recurrentValues(step, ex, graph);
  if (v.why) return null;
  const s = shown(step, v), l = v.cfg.layers - 1, net = netState(v, l, s.d, s.t, s.u);
  const main = net === null ? gapOf(returned(v.cfg, v.top)[s.j], v.y[s.j]) : gapOf(s.q.h[s.u], net);
  return checkOf({ ...main, what: net === null ? "out" : "h", unit: s.j, step: s.t }, [allStates(v), worstOf("out", returned(v.cfg, v.top), v.y)]);
}

// The words of a unit of the output: a (position, unit) of a layer that passes every state on,
// else a unit, a backward one named so.
function outputName(graph, v, j) {
  const at = cellOfOutput(v.cfg, v.top, j), unit = tText(at.d ? "how.recurrent.unit_back" : "how.recurrent.unit", { u: at.u + 1 });
  return v.cfg.ret === "all" ? cellName(positionName(graph, v.L, at.t), unit) : unit;
}

// The rows of one step of one unit: each gate (the page's; the network publishes no gate per unit),
// the LSTM's cell, and the new state beside the network's.
function stepRows(v, s) {
  const g = s.q.gates, u = s.u, rows = (v.cfg.cell === "rnn" ? [] : GATES[v.cfg.cell]).map((k) => ({ name: tText(`how.recurrent.gate.${k}`), page: g[k][u], network: null }));
  if (v.cfg.cell === "lstm") rows.push({ name: tText("how.recurrent.gate.c"), page: s.q.c[u], network: null });
  rows.push({ name: tText("how.recurrent.gate.h"), page: s.q.h[u], network: netState(v, v.cfg.layers - 1, s.d, s.t, u) ?? (v.cfg.ret === "all" || v.cfg.ret === "last" ? v.y[s.j] : null) });
  return rows;
}

function render(step, ex, env) {
  const graph = env.graph, v = recurrentValues(step, ex, graph), n = v.n, params = stageParams(n, { one: 1 });
  const head = [stageProse(n, "how.recurrent.lead", params), stageFormula(n, `how.recurrent.formula.${v.cfg.cell || "rnn"}`, params)];
  if (v.why) return el("div", {}, ...head, stopped(v.why, { what: v.what, stage: env.nameOf(step), shape: n.in_shape }));
  const s = shown(step, v), { rank, R } = rankUnits(v.R, v.y, step.pick), cfg = v.cfg, states = allStates(v);
  const pos = positionName(graph, v.L, s.t), prevC = cfg.cell === "lstm" ? t("how.recurrent.prev_c", { c: plain(s.q.prevC[s.u], 3) }) : "";
  return el("div", {}, ...head,
    cellPicks({ rank, R, on: s.j, target: env.target, choose: env.choose, nameOf: (u) => outputName(graph, v, u) }),
    el("p", {}, t(s.d ? "how.recurrent.at_back" : "how.recurrent.at", { u: s.u + 1, position: pos, f: v.F, h: cfg.H, layer: cfg.layers })),
    el("p", { class: "hint" }, t("how.recurrent.prev", { u: s.u + 1, h: plain(s.q.prev[s.u], 3), cell: prevC })),
    checkRows(stepRows(v, s)),
    states ? el("p", { class: "hint" }, t("how.recurrent.unrolled", { n: v.L, layers: cfg.layers * cfg.dirs, gap: plain(Math.abs(states.gap), 7) }))
      : el("p", { class: "hint how-miss" }, t("how.recurrent.no_states", { what: data(`${n.id}.hidden`) })),
    el("p", { class: "hint how-next" }, t(cfg.ret === "last" && cfg.dirs > 1 ? "how.recurrent.return.last_both" : `how.recurrent.return.${cfg.ret}`, { n: v.y.length })));
}

export default {
  kinds: ["recurrent"],
  render,
  recompute,
  unit: (step, ex, graph) => { const v = recurrentValues(step, ex, graph); return v.why ? (v.y ? rankUnits(v.R, v.y, step.pick).j : null) : shown(step, v).j; },
};
