// The graph steps (D4 §8.1, graph; explain-catalog.json#recompute.{gcn,sage,gat,gin}_message,
// graph_readout). A graph convolution sums, at each node, the messages its in-edges bring (the
// answer's `messages`: source, destination, hop, coefficient, size and share of the explained
// target, inspect-contract §6.4):
//
//   gcn   m = c·W·hⱼ                       h′ᵢ = Σ m + b
//   sage  m = W_n·(c·hⱼ), the root W_r·hᵢ    h′ᵢ = Σ m (max: the channel-wise maximum of c·hⱼ, then W_n)
//                                             + W_r·hᵢ + b, divided by its length when `normalize`
//   gat   m = α·w·W^h·hⱼ per head            h′ᵢ = the heads side by side (or their mean) + b
//   gin   m = c·hⱼ, the self term (1 + ε)·hᵢ   h′ᵢ = MLP(Σ m)
//
// At graph level the answer brings every node's numbers (detail "exact": `in` and `out` over the
// graph's nodes, in node order) and every message: the page recomputes every node's output from the
// weights it was sent, shows one unit of one node — the one that carried most of the explained
// target, or the one picked — and says the largest difference. At node level the answer brings the
// explained node's own numbers only, not its neighbours': the page recomputes the message the node
// sends itself (its size, from the node's numbers and the weights) and checks that the messages'
// shares of the target add up, with the bias's, to the output's own share ⟨∂s/∂h′, h′⟩ (a GIN runs
// its small network on the sum the answer brings, `pre_mlp`). A readout is the sum, mean or maximum
// over the graph's nodes, recomputed channel by channel.
import { el } from "../../core/dom.js";
import { termRow } from "../../core/bars.js";
import { t, tText, data } from "../../core/text.js";
import { nodeNamer } from "../../core/node-names.js";
import { plain, sg } from "../../core/fmt.js";
import { Answer, reducedValues } from "../answer.js";
import { rankUnits, gapOf, checkRows, stageParams, stageProse, stageFormula } from "../sum.js";
import { sizesOf, tensorsOf, fieldsOf, extraOf, stopped, checkOf, worstOf, allLine, cellName, cellPicks, affine, rowOf } from "../cells.js";

const SHOWN = 6;
// Messages the answer lists at most around one explanation (inspect-contract LIMITS.graph_nodes_max):
// a list this long may have been cut, and a sum over it is then not the node's.
export const MESSAGES_MAX = 256;
const norm = (v) => Math.sqrt(v.reduce((s, q) => s + q * q, 0));
const dot = (a, b) => a.reduce((s, q, i) => s + q * b[i], 0);

// ---- the conv's weights, by block type
function weightRefs(n) {
  const p = n.payload || {}, bias = typeof p.bias === "string" ? p.bias : null;
  if (n.type === "sage_conv") return { neigh: p.neigh, root: p.root, bias };
  if (n.type === "gin_conv") return Object.fromEntries((p.mlp || []).flatMap((l, k) => [[`m${k}`, l.weight], [`mb${k}`, l.bias]]));
  if (n.type === "gat_conv") return { W: p.weight, bias };
  return { W: p.weight, bias };
}

// The small network of a GIN: its linear layers with ReLU between.
export function ginMlp(w, shape, x) {
  let h = x;
  for (let k = 0; shape[`m${k}`]; k++) {
    const [o, i] = shape[`m${k}`];
    h = affine(w[`m${k}`], o, i, h, w[`mb${k}`] || null);
    if (shape[`m${k + 1}`]) h = h.map((v) => Math.max(0, v));
  }
  return h;
}

// One message's vector (the conv's space) from the source's numbers x and its coefficient.
export function messageOf(n, P, x, coef) {
  const p = n.payload || {};
  if (n.type === "gin_conv") return x.map((v) => coef * v);
  if (n.type === "sage_conv") return affine(P.w.neigh, P.shape.neigh[0], P.shape.neigh[1], x.map((v) => coef * v));
  const Wx = affine(P.w.W, P.shape.W[0], P.shape.W[1], x);
  if (n.type !== "gat_conv") return Wx.map((v) => coef * v);
  const heads = Number(p.heads || 1), out = Wx.length / heads, cs = Array.isArray(coef) ? coef : [coef];
  const per = Array.from({ length: heads }, (_, h) => Wx.slice(h * out, (h + 1) * out).map((v) => cs[h] * v));
  return p.concat === false ? Array.from({ length: out }, (_, c) => per.reduce((s, r) => s + r[c], 0) / heads) : per.flat();
}

// The output of node i from the messages into it, over the graph's rows X ([N][F]) at `pos` (node
// id → row): the sum (the sage max aggregate), the root, the bias, then the normalisation or the MLP.
export function nodeOutput(n, P, X, pos, msgs) {
  const p = n.payload || {}, into = msgs.filter((m) => !m.root);
  if (n.type === "sage_conv" && p.aggregator === "max") {
    const F = X[0].length, agg = into.length ? Array.from({ length: F }, (_, c) => Math.max(...into.map((m) => m.coef * X[pos.get(m.src)][c]))) : new Array(F).fill(0);
    return sageFinish(p, P, affine(P.w.neigh, P.shape.neigh[0], F, agg), rootRow(X, pos, msgs));
  }
  const W = n.type === "gin_conv" ? X[0].length : n.type === "sage_conv" ? P.shape.neigh[0] : null;
  const sum = into.reduce((s, m) => { const v = messageOf(n, P, X[pos.get(m.src)], m.coef); return s ? s.map((q, c) => q + v[c]) : v; }, null) || new Array(W || 0).fill(0);
  if (n.type === "gin_conv") return ginMlp(P.w, P.shape, sum);
  if (n.type === "sage_conv") return sageFinish(p, P, sum, rootRow(X, pos, msgs));
  return P.w.bias ? sum.map((v, c) => v + P.w.bias[c]) : sum;
}

// The numbers of the node a sage conv's root term reads (its root message's), or null.
function rootRow(X, pos, msgs) {
  const root = msgs.find((m) => m.root);
  return root ? X[pos.get(root.dst)] : null;
}

function sageFinish(p, P, agg, x) {
  const rootV = x && P.w.root ? affine(P.w.root, P.shape.root[0], P.shape.root[1], x) : null;
  const pre = agg.map((v, c) => v + (rootV ? rootV[c] : 0) + (P.w.bias ? P.w.bias[c] : 0));
  const len = norm(pre);
  return p.normalize ? pre.map((v) => v / Math.max(len, 1e-12)) : pre;
}

// ---- reading a conv step
const levelOf = (A, graph) => ((A.ex && A.ex.sample && A.ex.sample.graph && A.ex.sample.graph.level) || graph.sample.level || "node");

// The graph's node ids in row order: the readout's node list when one reports it, else the sorted
// destinations of the messages — when they are as many as the rows.
function nodeOrder(A, graph, msgs, N) {
  const share = graph.stages().filter((s) => s.kind === "graph_pool").map((s) => extraOf(A, graph, s.id, "node_share")).find(Boolean);
  const ids = share && Array.isArray(share.nodes) && share.nodes.length === N ? share.nodes.slice().sort((a, b) => a - b)
    : [...new Set(msgs.map((m) => m.dst))].sort((a, b) => a - b);
  return ids.length === N ? new Map(ids.map((id, r) => [id, r])) : null;
}

export function convValues(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage), level = levelOf(A, graph);
  const got = tensorsOf(graph, weightRefs(n)), P = { w: got.w, shape: got.shape }, { have, missing } = fieldsOf(A, graph, n.id, ["in", "out"]);
  const msgs = extraOf(A, graph, n.id, "messages"), base = { A, n, P, level, x: have.in, y: have.out, g: A.values(n.id, "grad_out"), R: A.unitRelevance(n.id), msgs: Array.isArray(msgs) ? msgs : null };
  if (got.lost.length) return { ...base, why: "weights", what: got.lost.join(", ") };
  if (missing || !base.msgs) return { ...base, why: "missing", what: missing || `${n.id}.messages` };
  const F = (n.in_shape || [])[0], W = (n.out_shape || [])[0];
  if (level === "graph") {
    const N = base.x.length / F, pos = Number.isInteger(N) && base.y.length === N * W ? nodeOrder(A, graph, base.msgs, N) : null;
    if (!pos || base.msgs.length >= MESSAGES_MAX) return { ...base, why: "fit", what: sizesOf(n.id, { messages: base.msgs, in: base.x, out: base.y }) };
    const X = Array.from({ length: N }, (_, r) => rowOf(base.x, F, r)), ids = [...pos.keys()];
    const page = ids.flatMap((id) => nodeOutput(n, P, X, pos, base.msgs.filter((m) => m.dst === id)));
    return { ...base, F, W, N, X, pos, ids, page };
  }
  if (base.x.length !== F || base.y.length !== W) return { ...base, why: "fit", what: sizesOf(n.id, { in: base.x, out: base.y }) };
  return { ...base, F, W, target: A.ex.sample && A.ex.sample.graph ? A.ex.sample.graph.target_node : null };
}

// The node-level checks: the self message's size, the shares against the output's own share, and
// (a GIN) its small network on the sum the answer brought.
function nodeChecks(v) {
  const self = v.msgs.find((m) => m.src === m.dst && (v.n.type === "sage_conv" ? m.root : !m.root));
  const selfV = !self ? null : v.n.type === "sage_conv" ? affine(v.P.w.root, v.P.shape.root[0], v.F, v.x) : messageOf(v.n, v.P, v.x, self.coef);
  const out = { self: self && selfV ? { what: "self", ...gapOf(norm(selfV), self.norm) } : null };
  if (v.n.type === "gin_conv") {
    const pre = reducedValues(extraOf(v.A, v.A.graph, v.n.id, "pre_mlp"));
    out.mlp = pre && pre.length === v.F ? ginMlp(v.P.w, v.P.shape, pre) : null;
  } else if (v.g && !(v.n.type === "sage_conv" && v.n.payload.normalize) && v.msgs.every((m) => Number.isFinite(m.relevance)) && v.msgs.length < MESSAGES_MAX) {
    const b = v.P.w.bias ? dot(v.g, v.P.w.bias) : 0, sum = v.msgs.reduce((s, m) => s + m.relevance, 0);
    out.share = { what: "share", sum, bias: b, ...gapOf(sum + b, dot(v.g, v.y)) };
  }
  return out;
}

function shown(step, v) { return rankUnits(v.R, v.y, step.pick).j; }

function recomputeConv(step, ex, graph) {
  const v = convValues(step, ex, graph);
  if (v.why) return null;
  const j = shown(step, v);
  if (v.level === "graph") return checkOf({ ...gapOf(v.page[j], v.y[j]), what: "y", unit: j }, [worstOf("out", v.page, v.y)]);
  const c = nodeChecks(v);
  if (c.mlp) return checkOf({ ...gapOf(c.mlp[j], v.y[j]), what: "y", unit: j }, [c.self, worstOf("out", c.mlp, v.y)]);
  const main = c.share || c.self;
  return main ? checkOf({ ...main, unit: j }, [c.share ? c.self : null]) : null;
}

// ---- the page's words
// A node by the name the try view's drawings give it (env.nodeName), else by core/node-names.js over
// this answer: the explained node of a node-level answer is "this node", a reference node its key
// when the service sent one, else its rank in the answer — never its index in the assembled graph.
const nodeName = (id, v, env) => (env && typeof env.nodeName === "function" ? env.nodeName : nodeNamer(v.A.ex, { level: v.level, order: v.ids || null }))(id);
function unitName(v, j, env) {
  if (v.level !== "graph") return tText("how.adv.unit", { u: j + 1 });
  return cellName(nodeName(v.ids[Math.floor(j / v.W)], v, env), tText("how.adv.unit", { u: (j % v.W) + 1 }));
}

// The messages into a node as bars of their shares of the target, the largest first.
function messageBars(v, dst, env) {
  const into = v.msgs.filter((m) => m.dst === dst).sort((a, b) => Math.abs(b.relevance || 0) - Math.abs(a.relevance || 0)), top = into.slice(0, SHOWN);
  const mx = Math.max(1e-9, ...top.map((m) => Math.abs(m.relevance || 0)));
  const who = (m) => (m.root ? tText("how.graph.root") : m.src === m.dst ? tText("how.graph.self") : tText("how.graph.from", { node: nodeName(m.src, v, env), hop: m.hop }));
  const coef = (c) => (Array.isArray(c) ? c.map((q) => plain(q, 3)).join(" / ") : plain(c, 3));
  const bars = el("div", { class: "bars diverge eq" }, ...top.map((m, i) => termRow({ name: who(m), v: m.relevance || 0, max: mx, i,
    val: [tText("how.graph.msg", { coef: data(coef(m.coef)), size: plain(m.norm, 3) }), " ", sg(m.relevance || 0)] })));
  const lead = v.level === "graph" ? t("how.graph.messages_into", { n: into.length, shown: top.length, node: nodeName(dst, v, env) })
    : t("how.graph.messages", { n: into.length, shown: top.length });
  return el("div", {}, el("p", {}, lead), bars);
}

function convBody(step, v, graph, env) {
  const j = shown(step, v);
  if (v.level === "graph") {
    const id = v.ids[Math.floor(j / v.W)];
    return [messageBars(v, id, env), el("p", {}, t("how.graph.sum", { what: unitName(v, j, env) })), checkRows([{ name: unitName(v, j, env), page: v.page[j], network: v.y[j] }]),
      v.n.type === "sage_conv" && v.n.payload.normalize ? el("p", { class: "hint" }, t("how.graph.normalised")) : null];
  }
  const c = nodeChecks(v), rows = [];
  if (c.self) rows.push({ name: tText("how.graph.self_size"), page: c.self.page, network: c.self.network });
  if (c.share) rows.push({ name: tText("how.graph.share"), page: c.share.page, network: c.share.network });
  if (c.mlp) rows.push({ name: unitName(v, j, env), page: c.mlp[j], network: v.y[j] });
  return [messageBars(v, v.target, env), el("p", { class: "hint" }, t("how.graph.node_only", { node: nodeName(v.target, v, env) })),
    c.share ? el("p", {}, t("how.graph.shares", { sum: sg(c.share.sum, 3), b: sg(c.share.bias, 3), page: sg(c.share.page, 3), network: sg(c.share.network, 3) })) : null,
    c.mlp ? el("p", {}, t("how.graph.gin", { n: v.F })) : null,
    rows.length ? checkRows(rows) : stopped("missing", { what: `${v.n.id}.messages` })];
}

function renderConv(step, ex, env) {
  const graph = env.graph, v = convValues(step, ex, graph), n = v.n, params = stageParams(n);
  const head = [stageProse(n, "how.graph.lead", params), stageFormula(n, "how.graph.formula", params)];
  if (v.why) return el("div", {}, ...head, stopped(v.why, { what: v.what, stage: env.nameOf(step), shape: n.in_shape }));
  const { rank, j, R } = rankUnits(v.R, v.y, step.pick);
  return el("div", {}, ...head, cellPicks({ rank, R, on: j, target: env.target, choose: env.choose, nameOf: (u) => unitName(v, u, env) }),
    ...convBody(step, v, graph, env), v.level === "graph" ? allLine(v.y.length, worstOf("out", v.page, v.y)) : null);
}

// ---- the readout
const READOUT = Object.freeze({
  sum: (xs) => xs.reduce((s, q) => s + q, 0),
  mean: (xs) => (xs.length ? xs.reduce((s, q) => s + q, 0) / xs.length : 0),
  max: (xs) => (xs.length ? Math.max(...xs) : 0),
});

export function poolValues(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage), { have, missing } = fieldsOf(A, graph, n.id, ["in", "out"]);
  const readout = String(n.payload.readout ?? n.config.readout ?? "mean"), base = { A, n, readout, x: have.in, y: have.out, R: A.unitRelevance(n.id) };
  if (missing) return { ...base, why: "missing", what: missing };
  const W = (n.out_shape || [])[0], N = base.x.length / W;
  if (!READOUT[readout] || !Number.isInteger(N) || base.y.length !== W) return { ...base, why: "fit", what: sizesOf(n.id, { in: base.x, out: base.y }) };
  const col = (c) => Array.from({ length: N }, (_, r) => base.x[r * W + c]);
  return { ...base, N, W, col, page: Array.from({ length: W }, (_, c) => READOUT[readout](col(c))) };
}

function renderPool(step, ex, env) {
  const v = poolValues(step, ex, env.graph), n = v.n, params = stageParams(n);
  const head = [stageProse(n, "how.graph.pool.lead", params), stageFormula(n, "how.graph.pool.formula", params)];
  if (v.why) return el("div", {}, ...head, stopped(v.why, { what: v.what, stage: env.nameOf(step), shape: n.in_shape }));
  const { rank, j, R } = rankUnits(v.R, v.y, step.pick), xs = v.col(j), won = xs.indexOf(Math.max(...xs));
  return el("div", {}, ...head, cellPicks({ rank, R, on: j, target: env.target, choose: env.choose, nameOf: (u) => tText("how.adv.channel", { c: u + 1 }) }),
    el("p", {}, t(`how.graph.pool.${v.readout}`, { c: j + 1, n: v.N, row: won + 1 })),
    checkRows([{ name: tText("how.adv.channel", { c: j + 1 }), page: v.page[j], network: v.y[j] }]), allLine(v.y.length, worstOf("out", v.page, v.y)));
}

function recomputePool(step, ex, graph) {
  const v = poolValues(step, ex, graph);
  if (v.why) return null;
  const j = rankUnits(v.R, v.y, step.pick).j;
  return checkOf({ ...gapOf(v.page[j], v.y[j]), what: "y", unit: j }, [worstOf("out", v.page, v.y)]);
}

export default {
  kinds: ["graph_conv", "graph_pool"],
  render: (step, ex, env) => (step.kind === "graph_pool" ? renderPool(step, ex, env) : renderConv(step, ex, env)),
  recompute: (step, ex, graph) => (step.kind === "graph_pool" ? recomputePool(step, ex, graph) : recomputeConv(step, ex, graph)),
  unit: (step, ex, graph) => {
    const v = step.kind === "graph_pool" ? poolValues(step, ex, graph) : convValues(step, ex, graph);
    return v.y ? rankUnits(v.R, v.y, step.pick).j : null;
  },
};
