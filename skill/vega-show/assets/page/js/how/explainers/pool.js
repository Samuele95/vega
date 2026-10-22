// A pooling step (D4 §8.1, pool): the positions of a sequence merged into one vector — each
// channel's mean, its maximum, or the first position alone (the [cls] token) — or, for pool1d,
// consecutive groups of `size` positions merged the same way (max or avg), channel by channel. The
// explain catalog's recompute table gives pool1d to this explainer (explain-catalog.json
// #recompute.pool1d.explainer "pool"; D4 §3.2 had listed it under conv.js). One output number
// at a time: the one that carried most of the explained target, or the one picked; the page reads
// the channel's numbers out of the stage's input and pools them itself, beside the network's value.
// A flatten or a reshape, when a network gives it a step, lays the same numbers out again: the page
// checks that nothing changed.
import { el } from "../../core/dom.js";
import { t, tText, data } from "../../core/text.js";
import { plain } from "../../core/fmt.js";
import { Answer } from "../answer.js";
import { picksRow, rankUnits, noRecompute, missingOf, checkRows, gapOf, stageParams, stageProse, stageFormula } from "../sum.js";

const SHOWN = 6;
const REDUCE = Object.freeze({
  mean: (xs) => xs.reduce((s, v) => s + v, 0) / xs.length,
  max: (xs) => Math.max(...xs),
  cls: (xs) => xs[0],
});

// The poolings each stage kind names (the forge catalog's blocks.pool.params.kind mean|max|cls and
// blocks.pool1d.params.kind max|avg; `first` an older name of the [cls] token, a pool1d's `mean`
// the explain catalog's formula's word for avg), each to the reduction the page applies, and the
// kind the network's block defaults to (blocks.py: Pool(kind="mean"), Pool1d(kind="max", size=2)).
const KINDS = Object.freeze({
  pool: Object.freeze({ names: Object.freeze({ mean: "mean", max: "max", cls: "cls", first: "cls" }), default: "mean" }),
  pool1d: Object.freeze({ names: Object.freeze({ max: "max", avg: "mean", mean: "mean" }), default: "max" }),
});

const last = (s) => (Array.isArray(s) && s.length ? s[s.length - 1] : 0);

// The pooling of a node: {name (as the network names it), kind (the page's reduction, null when the
// page does not know the name), size (pool1d: its groups of `size` positions; pool: every position)}.
export function poolOf(n) {
  const K = KINDS[n.kind] || KINDS.pool, name = String(n.payload.kind ?? n.config.kind ?? K.default);
  const kind = Object.prototype.hasOwnProperty.call(K.names, name) ? K.names[name] : null;
  return { name, kind, size: n.kind === "pool1d" ? Number(n.payload.size ?? n.config.size ?? 2) : null };
}

// The channels of a pooling's (L, C) input: its last axis (the output's last axis is the same).
const channelsOf = (n) => last(n.in_shape) || last(n.out_shape);

// The positions output unit j of a pooling reads: [{t, x}] of one channel. The input is (L, C)
// row-major, position t's channel c at t·C + c (blocks.py's (B, L, C) layout, one sample); a
// pool1d's output unit j is channel j mod C of group ⌊j / C⌋, the positions size·⌊j / C⌋ onwards (a
// last group shorter than `size` is dropped by the network, so it has no unit).
export function poolInputs(n, x, j) {
  const C = channelsOf(n), L = C ? Math.floor(x.length / C) : 0, { size } = poolOf(n), c = j % C;
  const t0 = size ? Math.floor(j / C) * size : 0, t1 = size ? t0 + size : L;
  return Array.from({ length: Math.max(0, Math.min(t1, L) - t0) }, (_, k) => ({ t: t0 + k, x: x[(t0 + k) * C + c] }));
}

// The page's pooled value of output unit j, or null (a pooling the page does not know, or no
// positions for the unit in what the stage read).
export function poolUnit(n, x, j) {
  const { kind } = poolOf(n), xs = kind ? poolInputs(n, x, j).map((p) => p.x) : [];
  return kind && xs.length ? REDUCE[kind](xs) : null;
}

function values(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage);
  return { A, n, x: A.values(n.id, "in"), y: A.values(n.id, "out"), R: A.unitRelevance(n.id) };
}

// The positions a unit read, the one a max kept marked.
function positionsLine(n, x, j, y) {
  const ps = poolInputs(n, x, j), won = poolOf(n).kind === "max" ? ps.findIndex((p) => p.x === y) : -1;
  const shown = ps.slice(0, SHOWN).map((p, k) => `${plain(p.x, 3)}${k === won ? "*" : ""}`).join("  ");
  return el("p", { class: "hint how-positions" }, t(ps.length > SHOWN ? "how.pool.positions_more" : "how.pool.positions",
    { n: ps.length, values: shown, more: ps.length - SHOWN }));
}

const miss = (key, params) => el("p", { class: "hint how-miss" }, t(key, params));

// A pooling step: the unit's channel, the positions it read and the page's pooled value beside the
// network's. What stops the recompute is said as it is: a pooling the page does not know (named as
// the network names it), the tensors the answer did not bring, or no positions of the unit in what
// the stage read (its numbers do not fit the stage's (L, C) shape).
function renderPool(step, ex, env) {
  const { n, x, y, R: rel } = values(step, ex, env.graph), P = poolOf(n);
  const head = [stageProse(n, "how.pool.lead", stageParams(n)), stageFormula(n)];
  if (!P.kind) return el("div", {}, ...head, miss("how.pool.unknown", { kind: data(P.name) }));
  if (!x || !y) return el("div", {}, ...head, noRecompute(data(missingOf(n.id, { in: x, out: y }))));
  const { rank, j, R } = rankUnits(rel, y, step.pick), picks = picksRow({ rank, R, on: j, target: env.target, choose: env.choose });
  const C = channelsOf(n), ps = poolInputs(n, x, j);
  if (!ps.length) return el("div", {}, ...head, picks, miss("how.pool.no_positions", { c: (j % (C || 1)) + 1, n: x.length, shape: data((n.in_shape || []).join(" × ")) }));
  return el("div", {}, ...head, picks, el("p", {}, t(`how.pool.kind.${P.kind}`, { c: (j % C) + 1, n: ps.length })), positionsLine(n, x, j, y[j]),
    checkRows([{ name: tText("how.unit", { j: j + 1, stage: env.nameOf(step) }), page: poolUnit(n, x, j), network: y[j] }]));
}

// A flatten or a reshape: the same numbers, laid out from one shape into another.
function renderReshape(step, ex, env) {
  const { n, x, y } = values(step, ex, env.graph), shape = (s) => data((s || []).join(" × "));
  const head = [stageProse(n, "how.reshape.lead", stageParams(n)), stageFormula(n),
    el("p", {}, t("how.reshape.shapes", { from: shape(n.in_shape), to: shape(n.out_shape) }))];
  if (!x || !y) return el("div", {}, ...head);
  const same = x.length === y.length && x.every((v, k) => v === y[k]);
  return el("div", {}, ...head, el("p", { class: "hint" }, t(same ? "how.reshape.same" : "how.reshape.moved", { n: y.length })));
}

function recompute(step, ex, graph) {
  const { n, x, y, R } = values(step, ex, graph);
  if (!x || !y) return null;
  if (step.kind === "reshape") {
    const worst = y.reduce((w, v, k) => (Math.abs(v - (x[k] ?? NaN)) > Math.abs(w.gap) ? { j: k, gap: v - x[k] } : w), { j: 0, gap: 0 });
    return x.length === y.length ? { ...gapOf(x[worst.j], y[worst.j]), what: "y", unit: worst.j } : null;
  }
  const j = rankUnits(R, y, step.pick).j, page = poolUnit(n, x, j);
  return page === null ? null : { ...gapOf(page, y[j]), what: "y", unit: j };
}

export default {
  kinds: ["pool", "pool1d", "reshape"],
  render: (step, ex, env) => (step.kind === "reshape" ? renderReshape(step, ex, env) : renderPool(step, ex, env)),
  recompute,
  unit: (step, ex, graph) => { const v = values(step, ex, graph); return v.y && step.kind !== "reshape" ? rankUnits(v.R, v.y, step.pick).j : null; },
};
