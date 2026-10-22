// A dense step (D4 §8.1; the hand-built page's hidden-layer step, 1845–1873): one unit of the layer
// — the one that carried most of the explained target, or the one the reader picked — as its
// weighted sum z = Σ w·x + b, term by term from the weights the page was sent, beside the network's
// own z; then what the layer does to z before passing it on, in order: its own bend, and each
// stage fused into it (a bend, a normalisation; a dropout does nothing while answering), the bend
// drawn as the curve with the unit on it. On a layer after a bend that silences negative sums, the
// note of how many units of the layer before stayed quiet.
//
// A standalone activation (kind "activation") is the bend alone: a = f(z) for one unit.
import { el } from "../../core/dom.js";
import { t, tText, hasKey, data } from "../../core/text.js";
import { plain } from "../../core/fmt.js";
import { unitMatrix, unitBias } from "../../net/segments.js";
import { Answer } from "../answer.js";
import { actOf, actPlot, silencesNegative } from "../act.js";
import { sumParts, sumBlock, picksRow, rankUnits, formula, noRecompute, missingOf, gapOf, stageParams, stageProse, stageFormula } from "../sum.js";
import { normUnit, normLine } from "./norm.js";
import { sizesOf, stopped } from "../cells.js";

// The explain catalog's entry of an activation (page.json#explain.activations), for its words and
// its parameters.
const actEntry = (env, name) => (env && env.page && env.page.explain && env.page.explain.activations ? env.page.explain.activations[name] || null : null);
export const actName = (name) => (hasKey(`explain.act.${name}.name`) ? tText(`explain.act.${name}.name`) : data(name));
// The activation an activation block applies (its catalog name).
const blockAct = (n) => n.payload.name || n.config.name || "none";

// What a dense stage's own bend passed on: its post-activation — but a decoder layer (role
// decoder_layer) is a bare linear whose output is z itself, the bend it carries applied after it, so
// what that bend passed on is what the decoder's next layer reads (inspect-contract §3.1).
export function bentOf(graph, n, A) {
  if (n.role !== "decoder_layer") return A.values(n.id, "post");
  const kids = n.parent && graph.byId(n.parent) ? graph.byId(n.parent).children : [], k = kids.indexOf(n.id);
  return k >= 0 && k + 1 < kids.length ? A.values(kids[k + 1], "in") : null;
}

// What a layer does to z after the sum, in order: its own bend (the dense stage's activation), then
// each stage fused into it but a dropout: [{kind: "act" | "norm", node, name, x, y}], x what the
// operation read and y what it passed on (null when the answer does not carry them).
export function layerOps(graph, step, A, z) {
  const n = graph.byId(step.stage), fused = (step.layer ? step.layer.fused : []).map((id) => graph.byId(id)).filter((f) => f.kind !== "dropout");
  const ops = [], chained = A.chainAt.has(n.id);
  let x = z;
  if (n.activation) {
    const y = bentOf(graph, n, A);
    ops.push({ kind: "act", node: n, name: n.activation, x, y });
    x = y;
  }
  fused.forEach((f, k) => {
    const last = k === fused.length - 1, y = A.values(f.id, "out") || (last && chained ? A.values(n.id, "post") : null);
    ops.push({ kind: f.kind === "norm" ? "norm" : "act", node: f, name: f.kind === "norm" ? null : blockAct(f), x: A.values(f.id, "in") || x, y });
    x = y;
  });
  return ops;
}

// The numbers of one dense step: the stage, its matrix and bias, what it read, its z, what the layer
// did to z, what it passed on, and each unit's share of the target.
export function denseValues(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage), fused = step.layer ? step.layer.fused : [];
  const last = fused.length ? fused[fused.length - 1] : n.id;
  const z = A.values(n.id, "pre") || (n.activation ? null : A.values(n.id, "out"));
  const ops = layerOps(graph, step, A, z), end = ops.length ? ops[ops.length - 1].y : null;
  const a = (last === n.id ? bentOf(graph, n, A) : A.values(last, "out")) || end || (ops.length ? null : z);
  const W = unitMatrix(graph, n), src = A.values(n.id, "in");
  // a stage's numbers for one sample are its widths: anything else (several nodes' rows, a dense
  // applied at every position of a sequence) is not read, and is said not to fit rather than missing
  const fits = (v, width) => (v && W && v.length === width ? v : null);
  const unfit = W && ((src && src.length !== W.in) || (z && z.length !== W.out)) ? sizesOf(n.id, { in: src, pre: z }) : null;
  return { A, n, W, bias: unitBias(graph, n), src: W ? fits(src, W.in) : src, z: W ? fits(z, W.out) : z, ops, a, R: A.unitRelevance(n.id, last), unfit };
}

// The unit a dense step shows: the reader's pick, else the one that carried most of the target.
function unit(step, ex, graph) {
  const v = denseValues(step, ex, graph);
  return rankUnits(v.R, v.a || v.z, step.pick).j;
}

// Whether a step reads the network's inputs (the input step's, or an embedding's joined to them).
const readsInputs = (step) => !step.from || step.from.kind === "input" || step.from.kind === "embedding";

// The names of the units a stage reads: the network's inputs when it reads them — the dense
// chain's names, or the numeric columns then each embedding dim, or on a flattened window each
// cell "<feature> @ <step>" — else "unit i · <the step before>".
export function sourceNames(graph, env, step, width) {
  const c = graph.chain, i = graph.inputs, n = graph.byId(step.stage);
  const names = !readsInputs(step) ? null : c && c.layers[0] && n && c.layers[0].node === n.id ? c.inputs
    : [...i.numeric, ...i.nominal.flatMap((col) => Array.from({ length: col.dim || 0 }, (_, d) => `${col.name}[${d}]`))];
  if (names && names.length === width) return (k) => data(names[k]);
  const cells = readsInputs(step) ? windowCells(graph, width) : null;
  if (cells) return cells;
  const stage = env.nameOf(step.from);
  return (k) => tText("how.unit", { j: k + 1, stage });
}

// The cells of a window laid end to end (a flatten before the first dense stage): unit k is
// feature k mod F at step ⌊k / F⌋, named by the network view's axis labels; null when the width is
// not the window's.
function windowCells(graph, width) {
  const feats = graph.inputs.features || graph.inputs.numeric, steps = graph.inputs.steps, L = graph.sample.length || (steps ? steps.length : 0);
  if (graph.sample.shape !== "window" || !feats.length || !L || L * feats.length !== width) return null;
  const F = feats.length, feat = (f) => (feats[f] && typeof feats[f] === "object" ? tText("how.input.past", { name: data(feats[f].name) }) : data(feats[f]));
  const step = (s) => (steps && steps[s] ? data(steps[s]) : s === L - 1 ? tText("how.input.now") : tText("how.input.back", { k: L - 1 - s }));
  return (k) => tText("how.input.cell", { feature: feat(k % F), step: step(Math.floor(k / F)) });
}

// How many units of the step before stayed quiet under a bend that silences negative sums.
function quietNote(step, ex, graph, env) {
  const from = step.from;
  if (!from || from.kind !== "dense") return null;
  const f = denseValues(from, ex, graph), op = f.ops.find((o) => o.kind === "act");
  if (!op || !op.x || !op.y || !silencesNegative(op.name, actEntry(env, op.name))) return null;
  const q = op.x.map((z, k) => (z < 0 ? op.y[k] : null)).filter((v) => v !== null), n = op.x.length;
  return el("p", { class: "hint" }, t("how.dense.quiet", { q: q.length, n, stage: env.nameOf(from), act: actName(op.name), lo: plain(Math.min(0, ...q)), zero: 0, live: n - q.length }));
}

// The bend of one unit: its curve, a = f(z), and what the bend does to it.
export function bendBlock(env, name, zin, a) {
  const f = actOf(name, actEntry(env, name)), act = actName(name);
  const state = tText(zin > 0 ? "how.dense.fires" : "how.dense.stays_quiet"), slope = (actEntry(env, name) || {}).params;
  const bend = hasKey(`explain.act.${name}.bend`) ? t("how.dense.bend", { act, bend: tText(`explain.act.${name}.bend`, { slope: slope ? slope.slope : 0.01 }), state })
    : t("how.dense.bend_other", { act });
  const words = { aria: tText("how.act.aria", { z: zin.toFixed(2), a: a.toFixed(2) }), axisZ: (r) => tText("how.act.axis_z", { r }), axisA: tText("how.act.axis_a") };
  return el("div", { class: "act" }, f ? actPlot(f, zin, a, words) : null,
    el("div", { class: "t" }, el("div", { class: "caps muted" }, t("how.dense.bends")),
      formula(t("how.formula.act", { act, z: plain(zin, 3), a: el("span", { class: "num" }, plain(a, 3)) })),
      el("p", { class: "how-bend" }, bend)));
}

// What the layer did to unit j after its sum, operation by operation.
function opsBlocks(env, v, j) {
  if (!v.ops.length) return [el("p", {}, t("how.dense.no_bend"))];
  return v.ops.map((o) => {
    if (!o.x || !o.y) return noRecompute(data(missingOf(o.node.id, o.node === v.n ? { pre: o.x, post: o.y } : { in: o.x, out: o.y })));
    if (o.kind === "norm") return el("div", { class: "how-norm" }, el("div", { class: "caps muted" }, t("how.dense.norms")), normLine(env.graph, o.node, { x: o.x, y: o.y, j }));
    return bendBlock(env, o.name, o.x[j], o.y[j]);
  });
}

// The catalog's placeholders for a dense stage: its widths, and the letter of what it reads.
const denseParams = (n, v, step) => stageParams(n, { ...(v.W ? { inputs: v.W.in } : {}), v: tText(readsInputs(step) ? "how.sym.input" : "how.sym.stage") });

function renderDense(step, ex, env) {
  const graph = env.graph, v = denseValues(step, ex, graph), n = v.n, params = denseParams(n, v, step);
  const from = readsInputs(step) ? tText("how.dense.from_inputs", { n: params.inputs }) : tText("how.dense.from_stage", { n: params.inputs, stage: env.nameOf(step.from) });
  const head = [stageProse(n, "how.dense.lead", { ...params, stage: env.nameOf(step), from }), stageFormula(n, "how.formula.sum", params)];
  if (v.unfit) return el("div", {}, ...head, stopped("fit", { what: v.unfit, stage: env.nameOf(step), shape: n.in_shape }));
  if (!v.W || !v.src || !v.z) return el("div", {}, ...head, noRecompute(data(missing(v, n))));
  const { rank, j, R } = rankUnits(v.R, v.a || v.z, step.pick);
  const parts = sumParts(v.W, j, v.src, { bias: v.bias, z: v.z[j] });
  const reader = env.next ? (env.next.kind === "link" ? tText("how.reader.link") : env.nameOf(env.next)) : "";
  return el("div", {}, ...head,
    picksRow({ rank, R, on: j, target: env.target, choose: env.choose }), quietNote(step, ex, graph, env),
    el("div", { class: "caps muted how-terms" }, t("how.dense.terms", { j: j + 1 })),
    sumBlock(parts, { nameOf: sourceNames(graph, env, step, v.W.in), mark: env.pickOf(step.from) }),
    ...opsBlocks(env, v, j),
    el("p", { class: "hint how-next" }, t("how.dense.next", { others: params.units - 1, stage: env.nameOf(step), reader })));
}

// What a dense step could not read (the tensor ids or fields the recompute needs).
function missing(v, n) {
  return [!v.W ? n.weight : null, !v.src ? `${n.id}.in` : null, !v.z ? `${n.id}.pre` : null].filter(Boolean).join(", ");
}

// The page's value of one operation at unit j: the bend of x, or the normalisation of x.
function opValue(graph, o, j) {
  if (o.kind === "norm") return normUnit(graph, o.node, o.x, j);
  const f = actOf(o.name);
  return f ? f(o.x[j]) : null;
}

// The recompute of a dense step: the sum (page Σ vs the network's z) and each operation after it
// (page vs the network's value), `within` only when every one is.
function recomputeDense(step, ex, graph) {
  const v = denseValues(step, ex, graph);
  if (!v.W || !v.src || !v.z) return null;
  const j = rankUnits(v.R, v.a || v.z, step.pick).j, sum = gapOf(sumParts(v.W, j, v.src, { bias: v.bias, z: v.z[j] }).total, v.z[j]);
  const ops = v.ops.filter((o) => o.x && o.y).map((o) => ({ what: o.kind, stage: o.node.id, ...gapOf(opValue(graph, o, j), o.y[j]) }))
    .filter((o) => Number.isFinite(o.page));
  return { ...sum, what: "z", unit: j, ops, within: sum.within && ops.every((o) => o.within) };
}

function renderActivation(step, ex, env) {
  const A = Answer.of(ex, env.graph), n = env.graph.byId(step.stage), z = A.values(n.id, "in"), a = A.values(n.id, "out");
  const head = [stageProse(n, "how.activation.lead", stageParams(n)), stageFormula(n)];
  if (!z || !a) return el("div", {}, ...head, noRecompute(data(missingOf(n.id, { in: z, out: a }))));
  const { rank, j, R } = rankUnits(A.unitRelevance(n.id), a, step.pick);
  return el("div", {}, ...head, picksRow({ rank, R, on: j, target: env.target, choose: env.choose }), bendBlock(env, blockAct(n), z[j], a[j]));
}

function recomputeActivation(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage), z = A.values(n.id, "in"), a = A.values(n.id, "out"), f = actOf(blockAct(n));
  if (!z || !a || !f) return null;
  const j = rankUnits(A.unitRelevance(n.id), a, step.pick).j;
  return { ...gapOf(f(z[j]), a[j]), what: "a", unit: j };
}

export default {
  kinds: ["dense", "activation", "decoder"],
  render: (step, ex, env) => (step.kind === "activation" ? renderActivation(step, ex, env) : renderDense(step, ex, env)),
  recompute: (step, ex, graph) => (step.kind === "activation" ? recomputeActivation(step, ex, graph) : recomputeDense(step, ex, graph)),
  unit: (step, ex, graph) => {
    if (step.kind !== "activation") return unit(step, ex, graph);
    const A = Answer.of(ex, graph), a = A.values(step.stage, "out");
    return a ? rankUnits(A.unitRelevance(step.stage), a, step.pick).j : null;
  },
};
