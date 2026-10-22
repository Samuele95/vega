// The attention steps (D4 §8.1; explain-catalog.json#recompute.attention_row, attention_pool,
// transformer_layer, token_affine), each recomputed by the page from the weights it was sent:
//
//   attention       Q, K, V = x·W_q + b_q, x·W_k + b_k, x·W_v + b_v; per head A = softmax(Q·Kᵀ/√d_h);
//                   y = [A₁V₁ ‖ … ‖ A_hV_h]·W_o + b_o, the mean over the positions when pooled
//   attention_pool  α = softmaxₜ(q · tanh(W·hₜ + b)); y = Σₜ αₜ·hₜ
//   transformer     one encoder layer (a step each): normalisation first, x₁ = x + MHA(LN₁ x),
//                   y = x₁ + W₂·f(W₁·LN₂ x₁ + b₁) + b₂ (without it, x₁ = LN₁(x + MHA x) and
//                   y = LN₂(x₁ + FF x₁)); the layer norms with ε = 1e-5, torch's default
//   tokenizer       t₀ = [cls]; tⱼ = xⱼ·wⱼ + bⱼ for a numeric input; tₖ = Eₖ[idₖ] for a nominal one
//
// One number is shown — the one that carried most of the explained target, or the one picked —
// with the attention weights it was built from beside the network's own (`alpha`, detail
// "exact"), and the whole stage is recomputed: the largest difference is said, never hidden.
import { el } from "../../core/dom.js";
import { t, tText, data } from "../../core/text.js";
import { plain } from "../../core/fmt.js";
import { softmax, actOf } from "../act.js";
import { Answer } from "../answer.js";
import { rankUnits, gapOf, checkRows, stageParams, stageProse, stageFormula } from "../sum.js";
import { sizesOf, tensorsOf, biasRef, fieldsOf, stopped, checkOf, worstOf, allLine, shapeText, positionName, cellName, cellPicks, affine, rowOf } from "../cells.js";

export const LN_EPS = 1e-5;
const SHOWN = 4;

// ---- the arithmetic (pure; rows are plain arrays)
export function layerNorm(x, g, b, eps = LN_EPS) {
  const n = x.length, mean = x.reduce((s, v) => s + v, 0) / n, v = x.reduce((s, q) => s + (q - mean) ** 2, 0) / n;
  return x.map((q, i) => (g ? g[i] : 1) * (q - mean) / Math.sqrt(v + eps) + (b ? b[i] : 0));
}

// Multi-head self-attention over X ([L][d]): {A: [head][query][key], Y: [L][d]} (P: win, bin,
// wout, bout — torch's packed in_proj, rows q then k then v).
export function mha(X, P, heads) {
  const L = X.length, d = X[0].length, dh = d / heads, qkv = X.map((x) => affine(P.win, 3 * d, d, x, P.bin || null));
  const A = [], mix = X.map(() => new Array(d).fill(0));
  for (let h = 0; h < heads; h++) {
    const Ah = [];
    for (let q = 0; q < L; q++) {
      const s = qkv.map((r) => { let z = 0; for (let c = 0; c < dh; c++) z += qkv[q][h * dh + c] * r[d + h * dh + c]; return z / Math.sqrt(dh); });
      const a = softmax(s);
      Ah.push(a);
      for (let c = 0; c < dh; c++) mix[q][h * dh + c] = a.reduce((z, w, k) => z + w * qkv[k][2 * d + h * dh + c], 0);
    }
    A.push(Ah);
  }
  return { A, Y: mix.map((m) => affine(P.wout, d, d, m, P.bout || null)) };
}

// Attention pooling over X ([L][H]): {alpha: [L], y: [H]}.
export function attentionPool(X, P) {
  const k = P.shape.score[0], H = X[0].length;
  const e = X.map((x) => affine(P.score, k, H, x, P.score_b || null).reduce((s, v, i) => s + P.query[i] * Math.tanh(v), 0));
  const alpha = softmax(e);
  return { alpha, y: Array.from({ length: H }, (_, c) => X.reduce((s, x, tt) => s + alpha[tt] * x[c], 0)) };
}

// One encoder layer over X ([L][d]): {A, attn: [L][d], ff: [L][d], y: [L][d]}.
export function encoderLayer(X, P, cfg) {
  const d = X[0].length, ffw = P.shape.l1[0], f = actOf(cfg.activation) || actOf("relu");
  const ln = (x, k) => layerNorm(x, P[`n${k}g`], P[`n${k}b`]), ff = (x) => affine(P.l2, d, ffw, affine(P.l1, ffw, d, x, P.l1b || null).map(f), P.l2b || null);
  const add = (a, b) => a.map((v, i) => v + b[i]);
  if (cfg.normFirst) {
    const at = mha(X.map((x) => ln(x, 1)), P, cfg.heads), x1 = X.map((x, q) => add(x, at.Y[q])), fo = x1.map((x) => ff(ln(x, 2)));
    return { A: at.A, attn: at.Y, ff: fo, y: x1.map((x, q) => add(x, fo[q])) };
  }
  const at = mha(X, P, cfg.heads), x1 = X.map((x, q) => ln(add(x, at.Y[q]), 1)), fo = x1.map(ff);
  return { A: at.A, attn: at.Y, ff: fo, y: x1.map((x, q) => ln(add(x, fo[q]), 2)) };
}

// ---- reading a stage: {n, id, x ([L][d] or flat), y (flat), alpha (the network's, flat), P, why, what}
const rowsOf = (x, shape) => Array.from({ length: shape[0] }, (_, q) => rowOf(x, shape[1], q));
const stageId = (step) => (step.kind === "transformer" && step.node ? step.node : step.stage);

function read(step, ex, graph, refs) {
  const A = Answer.of(ex, graph), id = stageId(step), n = graph.byId(id), fields = fieldsOf(A, graph, id, ["in", "out"]);
  const got = tensorsOf(graph, refs), base = { A, n, id, x: fields.have.in, y: fields.have.out, alpha: A.values(id, "alpha"), P: { ...got.w, shape: got.shape }, R: A.unitRelevance(id) };
  if (got.lost.length) return { ...base, why: "weights", what: got.lost.join(", ") };
  return fields.missing ? { ...base, why: "missing", what: fields.missing } : base;
}

const unfit = (v) => ({ ...v, why: "fit", what: sizesOf(v.id, { in: v.x, out: v.y }) });

function attentionValues(step, ex, graph) {
  const n = graph.byId(step.stage), p = n.payload || {}, heads = Number(p.heads ?? n.config.heads ?? 1);
  const v = read(step, ex, graph, { win: p.in_proj, bin: biasRef(graph, p.in_proj), wout: p.out_proj, bout: biasRef(graph, p.out_proj) });
  if (v.why) return v;
  const [L, d] = n.in_shape || [], pool = !!(p.pool ?? n.config.pool);
  if (v.x.length !== L * d || v.y.length !== (pool ? d : L * d) || v.P.shape.win[0] !== 3 * d || d % heads) return unfit(v);
  const r = mha(rowsOf(v.x, [L, d]), v.P, heads), Y = pool ? Array.from({ length: d }, (_, c) => r.Y.reduce((s, y) => s + y[c], 0) / L) : r.Y.flat();
  return { ...v, L, d, heads, pool, A: r.A, rows: r.Y, page: Y };
}

function poolValues(step, ex, graph) {
  const n = graph.byId(step.stage), p = n.payload || {};
  const v = read(step, ex, graph, { score: p.score, score_b: biasRef(graph, p.score), query: p.query });
  if (v.why) return v;
  const [L, H] = n.in_shape || [];
  if (v.x.length !== L * H || v.y.length !== H || v.P.shape.score[1] !== H || v.P.query.length !== v.P.shape.score[0]) return unfit(v);
  const r = attentionPool(rowsOf(v.x, [L, H]), v.P);
  return { ...v, L, d: H, heads: 1, A: [[r.alpha]], page: r.y, rows: rowsOf(v.x, [L, H]) };
}

function encoderValues(step, ex, graph) {
  const n = graph.byId(stageId(step)), p = n.payload || {}, sa = p.self_attn || {}, ff = p.ff || {}, n1 = p.norm1 || {}, n2 = p.norm2 || {};
  const v = read(step, ex, graph, { win: sa.in_proj, bin: biasRef(graph, sa.in_proj), wout: sa.out_proj, bout: biasRef(graph, sa.out_proj),
    l1: ff.linear1, l1b: biasRef(graph, ff.linear1), l2: ff.linear2, l2b: biasRef(graph, ff.linear2), n1g: n1.gamma, n1b: n1.beta, n2g: n2.gamma, n2b: n2.beta });
  if (v.why) return v;
  const [L, d] = n.in_shape || [], cfg = { heads: Number(sa.heads ?? n.config.heads ?? 1), activation: p.activation ?? n.config.activation ?? "relu", normFirst: (p.norm_first ?? n.config.norm_first) !== false };
  if (v.x.length !== L * d || v.y.length !== L * d || v.P.shape.win[0] !== 3 * d || v.P.shape.l1[1] !== d) return unfit(v);
  const X = rowsOf(v.x, [L, d]), r = encoderLayer(X, v.P, cfg);
  return { ...v, L, d, heads: cfg.heads, A: r.A, X, parts: r, page: r.y.flat(), normFirst: cfg.normFirst };
}

// The nominal ids a tokenizer reads: the answer's nominal input stream (the input node whose stream
// is nominal), else the row's input values.
const nominalStream = (graph) => graph.nodes().find((n) => n.type === "input" && n.config.stream === "nominal") || null;
function nominalIds(A, graph) {
  const stream = nominalStream(graph), ids = stream ? A.values(stream.id, "out") : null;
  if (ids) return ids;
  const vals = A.inputValues(), names = A.inputs().names;
  return vals ? graph.inputs.nominal.map((c) => vals[names.indexOf(c.name)]) : null;
}

function tokenizerValues(step, ex, graph) {
  const n = graph.byId(step.stage), p = n.payload || {}, tables = p.tables || [];
  const refs = { w: p.weight, b: p.bias, cls: p.cls, ...Object.fromEntries(tables.map((tb, k) => [`e${k}`, tb.table])) };
  const v = read(step, ex, graph, refs);
  if (v.why) return v;
  const [T, d] = n.out_shape || [], nNum = v.x.length, ids = tables.length ? nominalIds(v.A, graph) : [];
  if (!ids) return { ...v, why: "missing", what: `${(nominalStream(graph) || { id: n.id }).id}.out` };
  if (v.y.length !== T * d || T !== 1 + nNum + tables.length || v.P.shape.w[0] !== nNum || v.P.shape.w[1] !== d) return unfit(v);
  const tok = (k, c) => (k === 0 ? v.P.cls[c] : k <= nNum ? v.x[k - 1] * v.P.w[(k - 1) * d + c] + v.P.b[(k - 1) * d + c]
    : v.P[`e${k - 1 - nNum}`][Math.round(ids[k - 1 - nNum]) * d + c]);
  return { ...v, L: T, d, nNum, ids, tables, page: Array.from({ length: T * d }, (_, j) => tok(Math.floor(j / d), j % d)) };
}

// The words of a stage the explain catalog does not describe (its prose, its formula), by step kind.
const FALLBACK = Object.freeze({
  attention: Object.freeze(["how.attention.lead", "how.attention.formula"]),
  attention_pool: Object.freeze(["how.attention_pool.lead", "how.attention_pool.formula"]),
  transformer: Object.freeze(["how.transformer.lead", "how.transformer.formula"]),
  tokenizer: Object.freeze(["how.tokenizer.lead", "how.tokenizer.formula"]),
});

const VALUES = Object.freeze({ attention: attentionValues, attention_pool: poolValues, transformer: encoderValues, tokenizer: tokenizerValues });

// The network's attention weights against the page's A ([head][query][key]), flat. When the answer
// brought α in another shape the comparison cannot be made: that is reported as a check that does
// not fit (never within), and said on the page, not silently dropped.
const alphaShape = (v) => [v.A.length, v.A[0].length, v.A[0][0].length];
const alphaFits = (v) => !v.alpha || v.A.flat(2).length === v.alpha.length;
function alphaCheck(v) {
  if (!v.A || !v.alpha) return null;
  if (!alphaFits(v)) return { what: "alpha", fit: false, page: null, network: null, gap: null, within: false,
                               sizes: sizesOf(v.id, { alpha: v.alpha }), shape: alphaShape(v) };
  return worstOf("alpha", v.A.flat(2), v.alpha);
}

// The shown unit j of a stage's output and where it sits: (position, channel), or a channel of a
// pooled output with the position whose share of it is largest.
function shownOf(step, v) {
  const j = rankUnits(v.R, v.y, step.pick).j, c = j % v.d;
  if (v.page.length === v.L * v.d) return { j, q: Math.floor(j / v.d), c };
  const rows = step.kind === "attention_pool" ? v.rows.map((x, q) => v.A[0][0][q] * x[c]) : v.rows.map((y) => y[c]);
  return { j, c, q: rows.reduce((b, s, q) => (Math.abs(s) > Math.abs(rows[b]) ? q : b), 0) };
}

function recompute(step, ex, graph) {
  const v = VALUES[step.kind](step, ex, graph);
  if (v.why) return null;
  const s = shownOf(step, v);
  return checkOf({ ...gapOf(v.page[s.j], v.y[s.j]), what: "y", unit: s.j }, [alphaCheck(v), worstOf("out", v.page, v.y)]);
}

// ---- the page's words
const unitName = (graph, v, j) => (v.page.length === v.L * v.d ? cellName(positionName(graph, v.L, Math.floor(j / v.d)), tText("how.adv.channel", { c: (j % v.d) + 1 }))
  : tText("how.adv.channel", { c: j + 1 }));

// The largest weights of one query row per head, the page's beside the network's.
function weightRows(graph, v, q) {
  const L = v.A[0][0].length, net = v.alpha, rows = [];
  v.A.forEach((Ah, h) => {
    const a = Ah[step0(v) ? 0 : q], order = a.map((_, k) => k).sort((x, y) => a[y] - a[x]).slice(0, SHOWN);
    for (const k of order) {
      const at = step0(v) ? k : (h * v.L + q) * L + k;
      rows.push({ name: v.heads > 1 ? tText("how.attention.head_key", { h: h + 1, key: positionName(graph, L, k) }) : positionName(graph, L, k),
                  page: a[k], network: net && net.length === v.A.flat(2).length ? net[at] : null });
    }
  });
  return rows;
}
// An attention pool has one query: its A is [[α]].
const step0 = (v) => v.A.length === 1 && v.A[0].length === 1;

function body(step, v, s, graph) {
  const pos = positionName(graph, v.L, s.q), out = [{ name: unitName(graph, v, s.j), page: v.page[s.j], network: v.y[s.j] }];
  if (step.kind === "tokenizer") {
    const k = s.q, kind = k === 0 ? "cls" : k <= v.nNum ? "num" : "nom";
    const words = { k: pos, c: s.c + 1, x: plain(k && k <= v.nNum ? v.x[k - 1] : 0, 3), id: k > v.nNum ? Math.round(v.ids[k - 1 - v.nNum]) : 0 };
    return [el("p", {}, t(`how.tokenizer.${kind}`, words)), checkRows(out)];
  }
  const lead = step.kind === "attention_pool" ? t("how.attention_pool.weights", { n: v.L }) : t("how.attention.query", { position: pos, n: v.L, heads: v.heads });
  const parts = step.kind === "transformer" ? el("p", {}, t("how.transformer.parts", { position: pos, c: s.c + 1, x: plain(v.X[s.q][s.c], 3),
    a: plain(v.parts.attn[s.q][s.c], 3), f: plain(v.parts.ff[s.q][s.c], 3), y: plain(v.page[s.j], 3) })) : null;
  const pooled = v.pool ? el("p", { class: "hint" }, t("how.attention.pooled", { n: v.L })) : null;
  const sum = step.kind === "attention_pool" ? el("p", {}, t("how.attention_pool.sum", { c: s.c + 1, n: v.L })) : null;
  const misfit = alphaFits(v) ? null : el("p", { class: "hint how-miss" },
    t("how.attention.alpha_fit", { what: data(sizesOf(v.id, { alpha: v.alpha })), shape: shapeText(alphaShape(v)) }));
  return [el("p", {}, lead), checkRows(weightRows(graph, v, s.q)), misfit, parts, sum, pooled, checkRows(out),
    step.kind === "transformer" && !v.normFirst ? el("p", { class: "hint" }, t("how.transformer.post_norm")) : null];
}

function render(step, ex, env) {
  const graph = env.graph, v = VALUES[step.kind](step, ex, graph), n = step.kind === "transformer" ? graph.byId(step.stage) : v.n;
  const params = stageParams(n), [lead, formula] = FALLBACK[step.kind], head = [stageProse(n, lead, params), stageFormula(n, formula, params)];
  if (v.why) return el("div", {}, ...head, stopped(v.why, { what: v.what, stage: env.nameOf(step), shape: v.n.in_shape }));
  const s = shownOf(step, v), { rank, R } = rankUnits(v.R, v.y, step.pick);
  return el("div", {}, ...head,
    cellPicks({ rank, R, on: s.j, target: env.target, choose: env.choose, nameOf: (u) => unitName(graph, v, u) }),
    ...body(step, v, s, graph), allLine(v.y.length, worstOf("out", v.page, v.y)));
}

export default {
  kinds: ["attention", "attention_pool", "transformer", "tokenizer"],
  render,
  recompute,
  unit: (step, ex, graph) => { const v = VALUES[step.kind](step, ex, graph); return v.why ? null : shownOf(step, v).j; },
};

