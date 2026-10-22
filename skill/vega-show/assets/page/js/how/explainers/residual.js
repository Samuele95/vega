// The steps a group adds after its inside (D4 §8.1): a residual repeat's sum y = x + f(x) — with a
// learned projection when the widths differ, y = P·x + f(x), P·x recomputed term by term from P's
// weights — and a parallel block's merge, its branches' outputs laid side by side (concat) or added
// (sum). One unit at a time: the one that carried most of the explained target, or the one picked;
// the page's own y beside the network's.
import { el } from "../../core/dom.js";
import { t, tText, data } from "../../core/text.js";
import { plain, sg, subs } from "../../core/fmt.js";
import { unitMatrix, unitBias } from "../../net/segments.js";
import { Answer } from "../answer.js";
import { sumParts, sumBlock, picksRow, rankUnits, noRecompute, checkRows, gapOf, stageParams, stageProse, stageFormula } from "../sum.js";
import { sourceNames } from "./dense.js";

const last = (xs) => xs[xs.length - 1];

// The output of the first of `ids` the answer carries.
function outOf(A, ids) {
  for (const id of ids) { const v = id ? A.values(id, "out") : null; if (v) return v; }
  return null;
}

// ---- residual: x, P·x (page and network), f(x), y (page and network)
export function residualValues(step, ex, graph) {
  const A = Answer.of(ex, graph), inner = step.inner || [], g = graph.byId(step.group);
  const x = (inner.length ? A.values(inner[0], "in") : null) || (step.projection ? A.values(step.projection, "in") : null);
  const f = inner.length ? A.values(last(inner), "out") : null;
  const repeats = g && Array.isArray(g.payload.repeats) ? g.payload.repeats.length : Math.max(1, Number(g && g.config.repeat) || 1);
  const y = outOf(A, [step.add, step.repeat === repeats - 1 ? step.group : null]);
  const P = step.projection ? graph.byId(step.projection) : null;
  return { A, x, f, y, P, W: P ? unitMatrix(graph, P) : null, bias: P ? unitBias(graph, P) : null, px: P ? A.values(P.id, "out") : null,
           R: A.unitRelevance(step.add || step.group) };
}

// The page's P·x at unit j (recomputed from P's weights), x_j without a projection; null when it
// cannot be.
function skipAt(v, j) {
  if (!v.P) return v.x ? v.x[j] : null;
  return v.W && v.x ? sumParts(v.W, j, v.x, { bias: v.bias }).total : null;
}

function renderResidual(step, ex, env) {
  const v = residualValues(step, ex, env.graph), n = env.graph.byId(step.group);
  const head = [stageProse(n, "how.residual.lead", stageParams(n)), stageFormula(n, "how.residual.formula")];
  if (!v.x || !v.f || !v.y) return el("div", {}, ...head, noRecompute(data([step.add || step.group, ...(step.inner || [])].join(", "))));
  const { rank, j, R } = rankUnits(v.R, v.y, step.pick), skip = skipAt(v, j);
  if (skip === null) return el("div", {}, ...head, noRecompute(data(v.P.weight)));
  const proj = v.P ? [el("div", { class: "caps muted how-terms" }, t("how.residual.proj", { j: j + 1 })),
    sumBlock(sumParts(v.W, j, v.x, { bias: v.bias, z: v.px ? v.px[j] : null }), { nameOf: sourceNames(env.graph, env, step, v.W.in) })] : [];
  const page = skip + v.f[j];
  return el("div", {}, ...head, picksRow({ rank, R, on: j, target: env.target, choose: env.choose }), ...proj,
    el("div", { class: "formula" }, t(v.P ? "how.residual.line_proj" : "how.residual.line", { j: subs(j + 1), skip: plain(skip, 3), inner: sg(v.f[j], 3), y: el("span", { class: "num" }, plain(v.y[j], 3)) })),
    checkRows([{ name: tText("how.unit", { j: j + 1, stage: env.nameOf(step) }), page, network: v.y[j] }]),
    el("p", { class: "hint" }, t("how.residual.share", { skip: plain(skip, 3), inner: plain(v.f[j], 3) })));
}

function recomputeResidual(step, ex, graph) {
  const v = residualValues(step, ex, graph);
  if (!v.x || !v.f || !v.y) return null;
  const j = rankUnits(v.R, v.y, step.pick).j, skip = skipAt(v, j);
  if (skip === null) return null;
  const sum = gapOf(skip + v.f[j], v.y[j]), proj = v.P && v.px ? { what: "projection", stage: v.P.id, ...gapOf(skip, v.px[j]) } : null;
  return { ...sum, what: "y", unit: j, ops: proj ? [proj] : [], within: sum.within && (!proj || proj.within) };
}

// ---- merge: the branches' outputs, and what the merge made of them
function mergeValues(step, ex, graph) {
  const A = Answer.of(ex, graph), branches = (step.branches || []).map((b) => (b.ids.length ? A.values(last(b.ids), "out") : null));
  return { A, branches, y: outOf(A, [step.node, step.group]), R: A.unitRelevance(step.node || step.group) };
}

// Unit j of a concat merge: {b (the branch, from 1), u (its unit)}.
function concatAt(branches, j) {
  let at = j;
  for (let k = 0; k < branches.length; k++) { if (at < branches[k].length) return { b: k + 1, u: at }; at -= branches[k].length; }
  return null;
}

// The page's y_j: the branch unit it is (concat) or the sum of the branches' units (sum).
function mergeAt(step, v, j) {
  if (step.merge === "sum") return v.branches.reduce((s, b) => s + b[j], 0);
  const at = concatAt(v.branches, j);
  return at ? v.branches[at.b - 1][at.u] : null;
}

function renderMerge(step, ex, env) {
  const v = mergeValues(step, ex, env.graph), n = env.graph.byId(step.group), k = v.branches.length;
  const head = [stageProse(n, "how.merge.lead", stageParams(n)), stageFormula(n)];
  if (!v.y || v.branches.some((b) => !b)) return el("div", {}, ...head, noRecompute(data(step.node || step.group)));
  const { rank, j, R } = rankUnits(v.R, v.y, step.pick), page = mergeAt(step, v, j);
  if (page === null) return el("div", {}, ...head, noRecompute(data(step.node || step.group)));
  const at = step.merge === "sum" ? null : concatAt(v.branches, j);
  const line = at ? t("how.merge.concat", { k, j: j + 1, u: at.u + 1, b: at.b }) : t("how.merge.sum", { k, j: j + 1 });
  const rows = step.merge === "sum" ? v.branches.map((b, i) => ({ name: tText("how.merge.branch", { b: i + 1 }), page: b[j], network: null })) : [];
  rows.push({ name: tText("how.unit", { j: j + 1, stage: env.nameOf(step) }), page, network: v.y[j] });
  return el("div", {}, ...head, picksRow({ rank, R, on: j, target: env.target, choose: env.choose }), el("p", {}, line), checkRows(rows));
}

function recomputeMerge(step, ex, graph) {
  const v = mergeValues(step, ex, graph);
  if (!v.y || v.branches.some((b) => !b)) return null;
  const j = rankUnits(v.R, v.y, step.pick).j, page = mergeAt(step, v, j);
  return page === null ? null : { ...gapOf(page, v.y[j]), what: "y", unit: j };
}

function unit(step, ex, graph) {
  const v = step.kind === "merge" ? mergeValues(step, ex, graph) : residualValues(step, ex, graph);
  return v.y ? rankUnits(v.R, v.y, step.pick).j : null;
}

export default {
  kinds: ["residual", "merge"],
  render: (step, ex, env) => (step.kind === "merge" ? renderMerge(step, ex, env) : renderResidual(step, ex, env)),
  recompute: (step, ex, graph) => (step.kind === "merge" ? recomputeMerge(step, ex, graph) : recomputeResidual(step, ex, graph)),
  unit,
};
