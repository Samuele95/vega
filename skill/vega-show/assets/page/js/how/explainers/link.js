// The last step (D4 §8.1, §5.2; the hand-built page's output step, 1874–1892): the head's weighted
// sum for the output explained — term by term from the weights the page was sent, beside the
// network's own raw number — then what the lane's link makes of the head's numbers, recomputed here
// (act.js LINK_READ) and held beside the network's reading:
//   softmax        each class's exponential over the sum of them all; the answer the largest
//   sigmoid        p = σ(z), the positive class when p reaches the threshold
//   sigmoid_each   each label's σ on its own, every label past its threshold
//   identity       the number itself (each quantile's), put back in the target's units by the recipe
//                  (a regression-family head first ends on its fixed train-fitted target affine,
//                  y = z · scale + shift, the head's exported buffers, A37)
//   cumulative     P(y > j) = σ(zⱼ) per boundary, the class the boundaries passed count, each
//                  class's probability from its neighbours
//   reconstruction the rebuilt inputs against the real ones: the squared errors, the nominal
//                  columns' cross-entropy, their means added
//   distance       the squared distance of the network's point from the centre fitted in training
//   horizon        one number per step ahead (a band of quantiles under a quantile loss)
// and, as today, the note that dropout (and a VAE's sampling) is off while the network answers.
import { el } from "../../core/dom.js";
import { t, tText, hasKey, data, cls } from "../../core/text.js";
import { plain, pct } from "../../core/fmt.js";
import { unitMatrix, unitBias } from "../../net/segments.js";
import { Answer } from "../answer.js";
import { LINK_READ, sigmoid, ordinalProbs, ordinalRank, reconstructionError, svddDistance } from "../act.js";
import { sumParts, sumBlock, noRecompute, checkRows, gapOf, stageParams, stageProse, stageFormula } from "../sum.js";
import { sourceNames } from "./dense.js";

const TOP = 3, TOP_TABLE = 8;
const argmax = (xs) => xs.reduce((b, v, i) => (v > xs[b] ? i : b), 0);
const byAbs = (xs) => xs.map((_, j) => j).sort((a, b) => Math.abs(xs[b]) - Math.abs(xs[a]));

// The head's numbers for one answer: the stage, its matrix and bias, what it read, its raw numbers
// z (the stage's own pre-activation when it holds one per unit, else the readout's raw), the
// readout, the link, and the head's target affine when it ends on one (`affine`, and `y` the
// numbers the link reads: z · scale + shift, else z).
export function headValues(step, ex, graph) {
  const A = Answer.of(ex, graph), n = step.stage ? graph.byId(step.stage) : null, out = A.output();
  const W = n ? unitMatrix(graph, n) : null, pre = n ? A.values(n.id, "pre") : null;
  const z = pre && (!W || pre.length === W.out) ? pre : out.raw, labels = graph.headLabels();
  const affine = pre && z === pre ? affineOf(n, graph, z.length) : null;
  return { A, n, out, W, bias: n ? unitBias(graph, n) : null, src: n ? A.values(n.id, "in") : null, z, link: step.link || graph.link,
           affine, y: affine ? Array.from(z, (x, k) => x * affine.scale[k] + affine.shift[k]) : z,
           labels: labels.length === z.length ? labels : out.names.length === z.length ? out.names : null };
}

// A regression-family head's fixed target affine (A37): its exported buffers, one per unit, or null.
function affineOf(n, graph, k) {
  const shift = n ? graph.vector(`${n.id}.target_shift`) : null, scale = n ? graph.vector(`${n.id}.target_scale`) : null;
  return shift && scale && shift.length === k && scale.length === k ? { shift, scale } : null;
}

// The affine's lines: each shown unit's z, times the train target's spread, plus its mean.
function affineLines(v, units) {
  if (!v.affine) return [];
  const a = v.affine;
  return [el("p", { class: "how-link" }, t("how.link.affine.lead")),
    ...units.map((k) => el("div", { class: "formula" }, t("how.link.affine.line", { name: unitName(v, k), z: plain(v.z[k], 3),
      s: plain(a.scale[k], 3), m: plain(a.shift[k], 3), v: el("span", { class: "num" }, plain(v.y[k], 3)) })))];
}

// The words of head unit k: its label (the class, the label, the threshold, the rebuilt input, the
// step ahead), else "output k".
const unitName = (v, k) => (v.labels && v.labels[k] !== undefined && v.labels[k] !== null ? cls(v.labels[k]) : tText("how.link.nth", { k: k + 1 }));

// What the link makes of the head, per unit, for the page to rank the units by and to name the one
// explained by default: {score (a number per unit), unit (the default unit)}.
export function linkUnits(v, graph) {
  const z = v.y, tg = v.A.target(), index = tg && Number.isInteger(tg.index) && tg.index >= 0 && tg.index < z.length ? tg.index : null;
  const L = v.link;
  if (L === "reconstruction") {
    const r = reconError(v, graph);
    const score = r ? r.unitOf.map((f) => r.per[f]) : z.map(Math.abs);
    return { score, unit: argmax(score.map((s) => (Number.isFinite(s) ? s : -Infinity))) };
  }
  if (L === "distance") {
    const c = centreOf(v, graph), score = c ? z.map((f, j) => (f - c[j]) ** 2) : z.map(Math.abs);
    return { score, unit: argmax(score) };
  }
  const read = LINK_READ[L] ? LINK_READ[L](z) : null, score = read || z;
  // an ordinal head: by default the first boundary the row does not pass (the last one when it
  // passes them all), the one its answer stops at
  if (L === "cumulative") return { score, unit: index ?? Math.min(ordinalRank(z), z.length - 1) };
  return { score, unit: index ?? (L === "softmax" || L === "sigmoid_each" ? argmax(score) : 0) };
}

// The one-class centre (the head's exported buffer), or null.
const centreOf = (v, graph) => (v.n ? graph.vector(`${v.n.id}.centre`) : null);

// The reconstruction of this row, from the head's numbers and the inputs as the network read them.
function reconError(v, graph) {
  const x = v.A.inputValues();
  return x ? reconstructionError(v.z, x, graph.inputs.nominal.map((c) => c.cardinality)) : null;
}

// The network's reading of unit j after the link (its read, else its answer), and the page's.
export function linkCheck(v, graph, j) {
  const L = v.link, z = v.y, net = v.out.read, ans = v.out.answer || {};
  if (L === "distance") {
    const c = centreOf(v, graph), page = c ? svddDistance(z, c) : null, network = Number.isFinite(ans.distance) ? ans.distance : (v.A.target() || {}).value;
    return page === null || !Number.isFinite(network) ? null : { what: "distance", ...gapOf(page, network) };
  }
  if (L === "reconstruction") {
    const r = reconError(v, graph), network = Number.isFinite(ans.error) ? ans.error : null;
    return r && network !== null ? { what: "error", ...gapOf(r.error, network) } : null;
  }
  const read = LINK_READ[L] ? LINK_READ[L](z) : null;
  if (!read || !net || net.length !== read.length) return null;
  const at = L === "sigmoid" ? 0 : j;
  return { what: "read", unit: at, ...gapOf(read[at], net[at]) };
}

// The head's sum for unit j: its block (or what is missing), and its recompute.
function headSum(v, j) {
  if (!v.W || !v.src) return { parts: null, missing: [v.n && !v.W ? v.n.weight : null, v.n && !v.src ? `${v.n.id}.in` : null].filter(Boolean).join(", ") };
  return { parts: sumParts(v.W, j, v.src, { bias: v.bias, z: v.z[j] }) };
}

// ---- the link's block, per link
function softTable(v, env, j) {
  const z = v.z, names = v.out.names, zmax = Math.max(...z), shift = zmax > 80 ? zmax : 0;
  const e = z.map((x) => Math.exp(x - shift)), tot = e.reduce((s, x) => s + x, 0);
  const fe = (x) => (x >= 1e5 || x < 1e-3 ? x.toExponential(2) : x.toPrecision(3));
  const role = (name) => (env.roles && env.roles.cls ? env.roles.cls(name) : "");
  const heads = ["how.link.soft.class", "how.sum.z", shift ? "how.link.soft.exp_shift" : "how.link.soft.exp", "how.link.soft.score"];
  const soft = el("div", { class: "soft" }, ...heads.map((h) => el("span", { class: "h" }, tText(h))));
  z.map((_, k) => k).sort((a, b) => z[b] - z[a]).slice(0, TOP).forEach((k) => soft.append(
    el("span", { class: "c" + (k === j ? ` win ${role(names[k])}` : ""), title: cls(names[k]) }, cls(names[k])),
    el("span", {}, plain(z[k], 3)), el("span", {}, fe(e[k])), el("span", {}, `${(100 * e[k] / tot).toFixed(1)} %`)));
  soft.append(el("span", { class: "c h" }, t("how.link.soft.all", { n: z.length })), el("span", {}),
    el("span", { class: "h" }, t("how.link.soft.sum", { v: fe(tot) })), el("span", { class: "h" }, pct(1, 0)));
  return [el("p", { class: "how-link" }, t("how.link.softmax.lead", { k: z.length })), linkFormula("softmax"), soft];
}

function sigmoidBlock(v, graph, env) {
  const z = v.z[0], p = sigmoid(z), th = thresholdOf(v, graph, 0, env), positive = graph.task.positive || v.out.names[0];
  const answer = v.out.answer && v.out.answer.class !== undefined ? v.out.answer.class : null;
  const verdict = p >= th ? t("how.link.sigmoid.positive", { name: cls(positive) })
    : answer !== null && answer !== positive ? t("how.link.sigmoid.negative", { name: cls(answer) }) : t("how.link.sigmoid.other");
  return [linkFormula("sigmoid"), el("div", { class: "formula" }, t("how.link.sigmoid.line", { z: plain(z, 3), p: el("span", { class: "num" }, plain(p, 3)) })),
    el("p", { class: "how-link" }, t("how.link.sigmoid.rule", { t: plain(th, 3), verdict }))];
}

// The threshold of unit j (probability scale), the one the service decides with: the output
// record's own for that label (/v1/schema output.threshold_per_label), else a per-unit one the
// network view or the answer states, else the record's single threshold, the network view's or
// the answer's, else ½.
export function thresholdOf(v, graph, j, env = null) {
  const o = (env && env.output) || {}, name = v.out.names[j];
  if (o.perLabel && Number.isFinite(o.perLabel[name])) return o.perLabel[name];
  const one = (th) => (Array.isArray(th) ? th[j] : th && typeof th === "object" ? th[name] : null);
  for (const th of [graph.task.threshold, (v.out.answer || {}).threshold]) if (Number.isFinite(one(th))) return one(th);
  for (const th of [o.threshold, graph.task.threshold, (v.out.answer || {}).threshold]) if (Number.isFinite(th)) return th;
  return 0.5;
}

// The units at or above their threshold (sigmoid_each), in head order.
const unitsOn = (v, graph, env) => v.z.map((x, k) => (sigmoid(x) >= thresholdOf(v, graph, k, env) ? k : -1)).filter((k) => k >= 0);

function eachBlock(v, graph, env) {
  const z = v.z, p = z.map(sigmoid);
  const rows = byAbs(p.map((x) => x - 0.5)).slice(0, TOP_TABLE).map((k) => ({ k, th: thresholdOf(v, graph, k, env) }));
  const grid = el("div", { class: "soft how-table" }, ...["how.link.each.label", "how.sum.z", "how.link.each.p", "how.link.each.threshold"].map((h) => el("span", { class: "h" }, tText(h))));
  rows.forEach(({ k, th }) => grid.append(el("span", { class: "c" + (p[k] >= th ? " win" : "") }, unitName(v, k)), el("span", {}, plain(z[k], 3)),
    el("span", {}, plain(p[k], 3)), el("span", {}, plain(th, 3))));
  const on = unitsOn(v, graph, env);
  return [linkFormula("sigmoid_each"), grid, el("p", { class: "how-link" }, t("how.link.each.answer", { n: on.length, k: z.length }))];
}

function identityBlock(v, graph, j) {
  const value = (v.out.answer || {}).value, many = v.y.length > 1;
  const lines = many ? Array.from(v.y, (x, k) => el("div", { class: "formula" }, t("how.link.identity.quantile", { name: unitName(v, k), v: plain(x, 3) })))
    : [el("div", { class: "formula" }, t("how.link.identity.line", { v: el("span", { class: "num" }, plain(v.y[0], 3)) }))];
  const back = Number.isFinite(value) && Math.abs(value - v.y[j]) > 1e-6 * Math.max(1, Math.abs(value)) ? el("p", { class: "hint" }, t("how.link.identity.inverse", { v: plain(value, 3) })) : null;
  // under the affine its lines end on ŷ itself, and "ŷ = z" would name the standardised number
  return v.affine ? [...affineLines(v, Array.from(v.y, (_, k) => k)), back] : [linkFormula("identity"), ...lines, back];
}

function cumulativeBlock(v, graph) {
  const z = v.z, s = z.map(sigmoid), probs = ordinalProbs(z), classes = graph.ordinalClasses(), rank = ordinalRank(z);
  const name = (c) => (classes[c] === null || classes[c] === undefined ? tText("how.link.cumulative.above", { name: data(classes[c - 1] ?? "") }) : data(classes[c]));
  const grid = el("div", { class: "soft how-table" }, ...["how.link.cumulative.boundary", "how.sum.z", "how.link.cumulative.p", "how.link.cumulative.passed"].map((h) => el("span", { class: "h" }, tText(h))));
  z.forEach((x, k) => grid.append(el("span", { class: "c" }, unitName(v, k)), el("span", {}, plain(x, 3)), el("span", {}, plain(s[k], 3)),
    el("span", {}, tText(s[k] >= 0.5 ? "how.link.cumulative.yes" : "how.link.cumulative.no"))));
  return [linkFormula("cumulative"), grid, el("p", { class: "how-link" }, t("how.link.cumulative.answer", { n: rank, name: name(rank), p: pct(probs[rank], 1) }))];
}

function reconstructionBlock(v, graph) {
  const r = reconError(v, graph);
  if (!r) return [linkFormula("reconstruction"), noRecompute(data(tText("how.link.reconstruction.inputs")))];
  const names = [...graph.inputs.numeric, ...graph.inputs.nominal.map((c) => c.name)];
  const grid = el("div", { class: "soft how-table how-two" }, ...["how.link.reconstruction.input", "how.link.reconstruction.term"].map((h) => el("span", { class: "h" }, tText(h))));
  byAbs(r.per.map((x) => (Number.isFinite(x) ? x : 0))).slice(0, TOP_TABLE).forEach((f) => grid.append(el("span", { class: "c" }, data(names[f] ?? "")), el("span", {}, plain(r.per[f], 4))));
  return [linkFormula("reconstruction"), grid, el("div", { class: "formula" }, t("how.link.reconstruction.line", { e: el("span", { class: "num" }, plain(r.error, 4)) }))];
}

function distanceBlock(v, graph) {
  const c = centreOf(v, graph);
  if (!c) return [linkFormula("distance"), noRecompute(data(v.n ? `${v.n.id}.centre` : ""))];
  const th = (v.out.answer || {}).threshold ?? graph.task.threshold, s = svddDistance(v.z, c);
  return [linkFormula("distance"), el("div", { class: "formula" }, t("how.link.distance.line", { n: v.z.length, s: el("span", { class: "num" }, plain(s, 4)) })),
    Number.isFinite(th) ? el("p", { class: "how-link" }, t(s >= th ? "how.link.distance.flagged" : "how.link.distance.typical", { t: plain(th, 4) })) : null];
}

function horizonBlock(v) {
  const grid = el("div", { class: "soft how-table how-two" }, ...["how.link.horizon.step", "how.link.horizon.value"].map((h) => el("span", { class: "h" }, tText(h))));
  Array.from(v.y).forEach((x, k) => grid.append(el("span", { class: "c" }, unitName(v, k)), el("span", {}, plain(x, 3))));
  return [...affineLines(v, Array.from(v.y, (_, k) => k)), linkFormula("horizon"), grid];
}

// The link's formula: the catalog's (page.json#explain.links) when the page carries it.
const linkFormula = (link) => el("div", { class: "formula" }, t(hasKey(`explain.link.${link}.formula`) ? `explain.link.${link}.formula` : `how.link.${link}.formula`));

const BLOCKS = Object.freeze({
  softmax: (v, g, env, j) => softTable(v, env, j), sigmoid: (v, g, env) => sigmoidBlock(v, g, env), sigmoid_each: (v, g, env) => eachBlock(v, g, env),
  identity: (v, g, env, j) => identityBlock(v, g, j), cumulative: (v, g) => cumulativeBlock(v, g),
  reconstruction: (v, g) => reconstructionBlock(v, g), distance: (v, g) => distanceBlock(v, g), horizon: (v) => horizonBlock(v),
});

// ---- the step
// The notes of the last step (D4 §8.1; today 1883, 1891): dropout off while the network answers,
// with the largest rate it trained with, and a VAE's sampling off.
function offNotes(graph) {
  const stages = graph.stages(), p = Math.max(0, ...stages.map((n) => n.dropout || 0));
  return [p > 0 ? el("p", { class: "hint" }, t("how.dropout_off", { p: +(100 * p).toFixed(1) })) : null,
    stages.some((n) => n.type === "reparameterize") ? el("p", { class: "hint" }, t("how.latent_off")) : null];
}

// The unit shown: the reader's pick, else the link's default.
const pickOr = (pick, unit, n) => (Number.isInteger(pick) && pick >= 0 && pick < n ? pick : unit);

// The picks of the last step: the outputs the link ranks first, by name, with their reading.
function outputPicks(v, score, j, choose) {
  const rank = byAbs(score).slice(0, TOP), name = (k) => unitName(v, k);
  return el("div", { class: "picks" }, el("span", {}, t("how.link.picks")),
    ...rank.map((k) => Object.assign(el("button", { class: k === j ? "on" : "", title: tText("how.link.pick_title", { name: name(k), v: plain(score[k], 3) }) },
      t("how.link.pick", { name: name(k), v: plain(score[k], 3) })), { onclick: () => choose(k) })));
}

// The recompute's row under the link's block: the page's reading beside the network's.
function checkLine(v, graph, j) {
  const c = linkCheck(v, graph, j);
  return c ? checkRows([{ name: tText(`how.link.check.${c.what}`), page: c.page, network: c.network }], 4) : null;
}

function render(step, ex, env) {
  const graph = env.graph, v = headValues(step, ex, graph), { score, unit } = linkUnits(v, graph), j = pickOr(step.pick, unit, v.z.length);
  const from = step.from && step.from.kind !== "input" ? tText("how.dense.from_stage", { n: v.W ? v.W.in : 0, stage: env.nameOf(step.from) })
    : tText("how.dense.from_inputs", { n: v.W ? v.W.in : 0 });
  const params = v.n ? stageParams(v.n, { ...(v.W ? { inputs: v.W.in } : {}), v: tText(step.from && step.from.kind !== "input" ? "how.sym.stage" : "how.sym.input") }) : {};
  const name = unitName(v, j), s = headSum(v, j);
  return el("div", {},
    v.n ? stageProse(v.n, "how.link.lead", { ...params, k: v.z.length, from, name }) : el("p", {}, t("how.link.lead", { k: v.z.length, from, name })),
    v.n ? stageFormula(v.n, "how.formula.sum", params) : null,
    v.z.length > 1 ? outputPicks(v, score, j, env.choose) : null,
    el("div", { class: "caps muted how-terms" }, t("how.link.terms", { name })),
    s.parts ? sumBlock(s.parts, { nameOf: sourceNames(graph, env, step, v.W.in), mark: env.pickOf(step.from) }) : noRecompute(data(s.missing)),
    ...(BLOCKS[v.link] ? BLOCKS[v.link](v, graph, env, j) : []), checkLine(v, graph, j), ...offNotes(graph));
}

// The decision the last step states against the network's own: a multilabel head's labels at or
// above their thresholds (sigmoid_each) beside the answer's labels, a binary head's class beside
// the answer's class; null when the answer states none.
export function decisionCheck(v, graph, env) {
  const ans = v.out.answer || {};
  if (v.link === "sigmoid_each" && Array.isArray(ans.labels)) {
    return { what: "decision", page: unitsOn(v, graph, env).map((k) => String(v.out.names[k])), network: ans.labels.map(String) };
  }
  if (v.link === "sigmoid" && ans.class !== undefined && ans.class !== null && graph.task.positive !== undefined && graph.task.positive !== null) {
    const on = sigmoid(v.z[0]) >= thresholdOf(v, graph, 0, env);
    return { what: "decision", page: [on ? "positive" : "negative"], network: [String(ans.class) === String(graph.task.positive) ? "positive" : "negative"] };
  }
  return null;
}

// The recompute of the last step: the head's sum for the unit shown (page Σ vs the network's raw
// number) and the link's reading (page vs the network's read, error or distance), `within` only
// when both are; with the decision the step states (decisionCheck) as one more op.
function recompute(step, ex, graph, env = null) {
  const v = headValues(step, ex, graph), j = pickOr(step.pick, linkUnits(v, graph).unit, v.z.length), s = headSum(v, j);
  const sum = s.parts ? { what: "z", unit: j, ...gapOf(s.parts.total, v.z[j]) } : null, link = linkCheck(v, graph, j);
  const decision = decisionCheck(v, graph, env);
  if (!sum && !link) return null;
  return { ...(sum || link), ops: [...(sum && link ? [link] : []), ...(decision ? [decision] : [])], within: [sum, link].filter(Boolean).every((c) => c.within) };
}

export default {
  kinds: ["link", "head"],
  render, recompute,
  unit: (step, ex, graph) => { const v = headValues(step, ex, graph); return pickOr(step.pick, linkUnits(v, graph).unit, v.z.length); },
};
