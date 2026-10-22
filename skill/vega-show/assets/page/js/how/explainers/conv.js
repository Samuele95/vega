// A convolution step (D4 §8.1, conv; explain-catalog.json#recompute.conv1d): one output number —
// one channel c′ at one position t, the one that carried most of the explained target or the one
// picked — recomputed from the kernel, the bias and the numbers under the window:
//
//   y[t, c′] = b[c′] + Σ_c Σ_(j < k) W[c′, c, j] · x[t·s + j·d − pad, c]
//
// the stage's (L, C) input laid out as blocks.py lays it (position t's channel c at t·C + c), a
// position before the start or past the end of the sequence reading zero. The terms are shown as a
// weighted sum (sum.js's block: the largest terms, the rest, the bias, the gap, the network's own
// value), each term named by its channel and its position; then the whole map is recomputed and the
// largest difference said, never hidden. pool1d is not here: the pool explainer's
// (explain-catalog.json#recompute.pool1d).
import { el } from "../../core/dom.js";
import { t, tText } from "../../core/text.js";
import { Answer } from "../answer.js";
import { sumParts, sumBlock, rankUnits, gapOf, stageParams, stageProse, stageFormula } from "../sum.js";
import { sizesOf, tensorsOf, fieldsOf, stopped, checkOf, worstOf, allLine, positionName, channelName, cellName, cellPicks, lastOf, sizeOf } from "../cells.js";

// The geometry of a conv stage: its kernel (C′ × C × k), stride, dilation and the zero padding
// before the first position (a "same" padding as torch pads it: (k − 1)·d split, the smaller half
// first), and its input and output lengths.
export function geometryOf(n, kernelShape) {
  const p = n.payload || {}, c = n.config || {};
  const [out, inC, k] = kernelShape, stride = Number(p.stride ?? c.stride ?? 1), dil = Number(p.dilation ?? c.dilation ?? 1);
  const raw = p.padding ?? c.padding ?? 0;
  const pad = raw === "same" ? Math.floor((dil * (k - 1)) / 2) : raw === "valid" ? 0 : Number(raw);
  const L = (n.in_shape || [])[0], Lout = (n.out_shape || [])[0];
  return { out, inC, k, stride, dil, pad, L, Lout };
}

// The numbers of one conv step: the stage, its kernel and bias, what it read and passed on, and why
// it cannot be recomputed when it cannot (missing | weights | fit).
export function convValues(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage), p = n.payload || {};
  const { w, shape, lost } = tensorsOf(graph, { kernel: p.kernel, bias: typeof p.bias === "string" ? p.bias : null });
  const { have, missing } = fieldsOf(A, graph, n.id, ["in", "out"]);
  const base = { A, n, x: have.in, y: have.out, W: w.kernel || null, b: w.bias || null, R: A.unitRelevance(n.id) };
  if (lost.length) return { ...base, why: "weights", what: lost.join(", ") };
  if (missing) return { ...base, why: "missing", what: missing };
  const g = geometryOf(n, shape.kernel);
  const fits = shape.kernel.length === 3 && base.x.length === g.L * g.inC && base.y.length === g.Lout * g.out && lastOf(n.out_shape) === g.out;
  if (!fits) return { ...base, g, why: "fit", what: [sizesOf(n.id, { in: base.x, out: base.y }), `${p.kernel}[${shape.kernel.join("×")}]`].join(", ") };
  return { ...base, g };
}

// The window of output (t, c′): for each input channel c and tap j, the position it reads (null
// past either end) — the terms of the sum, in (c, j) order.
export function windowOf(g, tt) {
  const cells = [];
  for (let c = 0; c < g.inC; c++) {
    for (let j = 0; j < g.k; j++) {
      const pos = tt * g.stride + j * g.dil - g.pad;
      cells.push({ c, j, pos: pos >= 0 && pos < g.L ? pos : null });
    }
  }
  return cells;
}

// The page's value of output unit u (flat: position ⌊u / C′⌋, channel u mod C′), with its terms.
export function convUnit(v, u) {
  const g = v.g, o = u % g.out, tt = Math.floor(u / g.out), cells = windowOf(g, tt);
  const src = cells.map((q) => (q.pos === null ? 0 : v.x[q.pos * g.inC + q.c]));
  const row = { out: 1, in: cells.length, w: Float64Array.from(cells, (q) => v.W[(o * g.inC + q.c) * g.k + q.j]) };
  const parts = sumParts(row, 0, src, { bias: v.b ? [v.b[o]] : null, z: v.y[u] });
  return { o, t: tt, cells, parts };
}

// Every output number, recomputed (the whole map, for the largest difference).
const allUnits = (v) => Array.from({ length: v.y.length }, (_, u) => convUnit(v, u).parts.total);

function unitOf(step, v) { return rankUnits(v.R, v.y, step.pick).j; }

const nameOfUnit = (graph, v, u) => cellName(positionName(graph, v.g.Lout, Math.floor(u / v.g.out)), tText("how.adv.channel", { c: (u % v.g.out) + 1 }));

function render(step, ex, env) {
  const graph = env.graph, v = convValues(step, ex, graph), n = v.n, params = stageParams(n);
  const head = [stageProse(n, "how.conv.lead", params), stageFormula(n, "how.conv.formula", params)];
  if (v.why) return el("div", {}, ...head, stopped(v.why, { what: v.what, stage: env.nameOf(step), shape: n.in_shape }));
  const { rank, j, R } = rankUnits(v.R, v.y, step.pick), unit = convUnit(v, j), g = v.g;
  const read = unit.cells.filter((q) => q.pos !== null).map((q) => q.pos), pads = unit.cells.some((q) => q.pos === null);
  const nameOf = (i) => {
    const q = unit.cells[i];
    return q.pos === null ? tText("how.conv.pad_term", { channel: channelName(graph, step, q.c) })
      : tText("how.input.cell", { feature: channelName(graph, step, q.c), step: positionName(graph, g.L, q.pos) });
  };
  const at = read.length ? t("how.conv.at", { what: nameOfUnit(graph, v, j), k: g.k, c: g.inC,
    first: positionName(graph, g.L, Math.min(...read)), last: positionName(graph, g.L, Math.max(...read)) })
    : t("how.conv.at_pad", { what: nameOfUnit(graph, v, j) });
  return el("div", {}, ...head,
    cellPicks({ rank, R, on: j, target: env.target, choose: env.choose, nameOf: (u) => nameOfUnit(graph, v, u) }),
    el("p", {}, at),
    pads ? el("p", { class: "hint" }, t("how.conv.pad", { pad: g.pad })) : null,
    sumBlock(unit.parts, { nameOf, biasName: tText("how.conv.bias", { c: unit.o + 1 }) }),
    allLine(v.y.length, worstOf("y", allUnits(v), v.y)),
    el("p", { class: "hint how-next" }, t("how.conv.next", { n: v.y.length, positions: g.Lout, channels: g.out })));
}

function recompute(step, ex, graph) {
  const v = convValues(step, ex, graph);
  if (v.why) return null;
  const j = unitOf(step, v), unit = convUnit(v, j);
  return checkOf({ ...gapOf(unit.parts.total, v.y[j]), what: "y", unit: j }, [worstOf("map", allUnits(v), v.y)]);
}

export default {
  kinds: ["conv"],
  render,
  recompute,
  unit: (step, ex, graph) => { const v = convValues(step, ex, graph); return v.y && sizeOf(v.n.out_shape) === v.y.length ? unitOf(step, v) : null; },
};
