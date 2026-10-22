// A normalisation step (D4 §8.1, norm): one unit of a batch normalisation, answering with the
// running mean and variance fitted in training, y = γ·(x − μ)/√(σ² + ε) + β; or of a layer
// normalisation, centred and scaled by the mean and variance of its own vector. The page recomputes
// the unit from the parameters it was sent (γ, β, the running statistics, ε) and the stage's input,
// beside the network's own y. On a sequence the unit is one channel at one position: the running
// statistics are the channel's, the layer normalisation's vector the position's. normUnit and
// normLine also serve a normalisation fused into a dense layer (dense.js).
import { el } from "../../core/dom.js";
import { t, tText, data } from "../../core/text.js";
import { plain, sg, subs } from "../../core/fmt.js";
import { Answer } from "../answer.js";
import { picksRow, rankUnits, noRecompute, missingOf, checkRows, gapOf, stageParams, stageProse, stageFormula } from "../sum.js";

const BATCH = "batchnorm";
const EPS = 1e-5;

// A parameter vector of a normalisation node (a tensor id or an inline list), or null.
function param(graph, n, key) {
  const ref = n.payload[key];
  return ref === undefined || ref === null ? null : graph.vector(ref);
}

// The width of the normalised vector (the channels of a sequence).
const widthOf = (n) => n.in_shape[n.in_shape.length - 1];

// The constants of a normalisation at unit j of its (flat) input x: {gamma, beta, eps, mean, var,
// x}; batchnorm: the channel's running statistics; layernorm: those of the position's vector. null
// when a parameter or the input is missing.
export function normAt(graph, n, x, j) {
  if (!x || !x.length || j < 0 || j >= x.length) return null;
  const C = widthOf(n) || x.length, c = j % C, at = j - c;
  const g = param(graph, n, "gamma"), b = param(graph, n, "beta");
  const eps = Number(n.payload.eps ?? n.config.eps ?? EPS), gamma = g ? g[c] : 1, beta = b ? b[c] : 0;
  if (n.type === BATCH) {
    const mu = param(graph, n, "running_mean"), vr = param(graph, n, "running_var");
    return mu && vr ? { gamma, beta, eps, mean: mu[c], var: vr[c], x: x[j], n: C } : null;
  }
  const seg = x.slice(at, at + C), mean = seg.reduce((s, v) => s + v, 0) / seg.length;
  return { gamma, beta, eps, mean, var: seg.reduce((s, v) => s + (v - mean) ** 2, 0) / seg.length, x: x[j], n: seg.length };
}

// y_j recomputed from the stage's input x, or null.
export function normUnit(graph, n, x, j) {
  const c = normAt(graph, n, x, j);
  return c ? c.gamma * ((c.x - c.mean) / Math.sqrt(c.var + c.eps)) + c.beta : null;
}

// The line of one unit with its numbers: γ·(x − μ)/√(σ² + ε) + β = … = y.
export function normLine(graph, n, { x, y, j }) {
  const c = normAt(graph, n, x, j), page = normUnit(graph, n, x, j);
  if (!c || page === null) return noRecompute(data(`${n.id}.${n.type === BATCH ? "running_var" : "in"}`));
  const key = n.type === BATCH ? "how.norm.batch" : "how.norm.layer";
  return el("div", { class: "formula" }, t(key, { j: subs(j + 1), g: plain(c.gamma, 3), x: plain(c.x, 3), mu: plain(c.mean, 3), var: plain(c.var, 3), eps: c.eps, b: sg(c.beta, 3),
    y: el("span", { class: "num" }, plain(y ? y[j] : page, 3)) }));
}

function values(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage);
  return { A, n, x: A.values(n.id, "in"), y: A.values(n.id, "out"), R: A.unitRelevance(n.id) };
}

function render(step, ex, env) {
  const { n, x, y, R: rel } = values(step, ex, env.graph), params = stageParams(n);
  const head = [stageProse(n, "how.norm.lead", params), stageFormula(n)];
  if (!x || !y) return el("div", {}, ...head, noRecompute(data(missingOf(n.id, { in: x, out: y }))));
  const { rank, j, R } = rankUnits(rel, y, step.pick), c = normAt(env.graph, n, x, j), page = normUnit(env.graph, n, x, j);
  const stats = n.type !== BATCH && c ? el("p", { class: "hint" }, t("how.norm.stats", { n: c.n, mean: plain(c.mean, 3), var: plain(c.var, 3) })) : null;
  return el("div", {}, ...head, picksRow({ rank, R, on: j, target: env.target, choose: env.choose }), stats,
    normLine(env.graph, n, { x, y, j }),
    page === null ? null : checkRows([{ name: tText("how.unit", { j: j + 1, stage: env.nameOf(step) }), page, network: y[j] }]));
}

function recompute(step, ex, graph) {
  const { n, x, y, R } = values(step, ex, graph);
  if (!x || !y) return null;
  const j = rankUnits(R, y, step.pick).j, page = normUnit(graph, n, x, j);
  return page === null ? null : { ...gapOf(page, y[j]), what: "y", unit: j };
}

export default {
  kinds: ["norm"],
  render, recompute,
  unit: (step, ex, graph) => { const v = values(step, ex, graph); return v.y ? rankUnits(v.R, v.y, step.pick).j : null; },
};
