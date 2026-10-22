// node --test: the advanced explainers of "how the network computed it" (D4 §8.1, BUILD-PLAN W14):
// js/how/explainers/{conv, recurrent, attention, graph, latent}.js and their shared parts
// (js/how/cells.js), on small stages whose every number is computed here by hand from the formula
// the network implements (blocks.py): a convolution with padding, the three recurrent cells (a
// bidirectional layer's last state read as blocks.py reads it), self-attention, attention pooling,
// a tokenizer, a latent code with its clipped log-variance and KL term, a graph convolution at node
// level (its own message and the shares) and at graph level (a GIN over every node), and the three
// readouts. Each recomputes the network's value within the probe's tolerance; each says what stops
// it — the answer's missing tensors, the weights it did not receive, numbers that do not fit — in
// words the reader can tell apart, renders with no key missing and no placeholder left, and never
// throws. The registry loads every advanced module with no fallback. The real networks (every
// family of the catalog, against torch) are vega-show/scripts/tests/test_page_math_advanced.py's.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { install } from "./dom-shim.js";

install();
const { load, gapsOf, EXPLAINER_OF, MODULES, CORE_MODULES } = await import("../js/how/explainers/index.js");
const { configureText, missingKeys, unresolved } = await import("../js/core/text.js");
const { Lexicon } = await import("../js/core/lexicon.js");
const conv = (await import("../js/how/explainers/conv.js")).default;
const recurrent = (await import("../js/how/explainers/recurrent.js")).default;
const attention = (await import("../js/how/explainers/attention.js")).default;
const graphX = (await import("../js/how/explainers/graph.js")).default;
const latent = (await import("../js/how/explainers/latent.js")).default;
const { split, LOGVAR_CLIP } = await import("../js/how/explainers/latent.js");
const { sizesOf, positionName, channelName } = await import("../js/how/cells.js");

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const sig = (z) => 1 / (1 + Math.exp(-z));
const f32 = (xs, shape = [xs.length]) => ({ shape, axes: shape.map((_, i) => `a${i}`), dtype: "float32", encoding: "base64-le", data: Buffer.from(new Float32Array(xs).buffer).toString("base64") });
const i64 = (xs) => ({ shape: [xs.length], axes: ["a0"], dtype: "int64", encoding: "base64-le", data: Buffer.from(new BigInt64Array(xs.map(BigInt)).buffer).toString("base64") });
const close = (a, b, what) => assert.ok(Math.abs(a - b) <= 1e-5 * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);

// A stage graph as the explainers read it: nodes by id, the tensors by id (`withheld` names those
// sent without values), the inputs' axis labels and the sample.
class Graph {
  constructor({ nodes, tensors = {}, withheld = [], inputs = {}, sample = { shape: "window" } }) {
    this.map = new Map(nodes.map((n) => [n.id, { config: {}, payload: {}, children: [], parent: null, ...n }]));
    Object.assign(this, { tensors, withheld, sample, chain: null });
    this.inputs = { names: [], numeric: [], nominal: [], features: null, steps: null, tokens: null, ...inputs };
  }
  byId(id) { return this.map.get(id) || null; }
  nodes() { return [...this.map.values()]; }
  stages() { return this.nodes().filter((n) => n.kind); }
  tensor(id) { return this.tensors[id] ? { id, shape: this.tensors[id].shape, encoding: this.withheld.includes(id) ? "withheld" : "json" } : null; }
  vector(id) { return this.tensors[id] && !this.withheld.includes(id) ? Float32Array.from(this.tensors[id].values) : null; }
}
const W = (values, shape) => ({ values, shape });
// An explanation@2 at detail exact: {node id: {field: tensor}} (and `extra` per node when given).
const explanation = (nodes, sample = { shape: "window" }) => ({
  schema: "vega/explanation@2", detail: "exact", degraded: false, sample,
  nodes: Object.fromEntries(Object.entries(nodes).map(([id, f]) => [id, { level: "exact", exact: f.exact || {}, extra: f.extra || {} }])),
});
const words = () => configureText({ lex: new Lexicon(null, null), lane: { viewKey: "value", shape: "window" } });
// A step's body, as the stepper renders it.
function body(x, step, ex, graph) {
  words();
  const env = { graph, target: "the value", nameOf: () => "this step", pickOf: () => -1, choose: () => {}, next: null, page: null };
  return x.render(step, ex, env).textContent;
}
const noLeftovers = () => { assert.deepEqual(missingKeys(), []); assert.deepEqual(unresolved(), []); };

// ---- the registry
test("every advanced step kind loads its own module, with the whole interface, and none falls back", async () => {
  const advanced = ["conv", "recurrent", "attention", "attention_pool", "transformer", "tokenizer", "latent", "graph_conv", "graph_pool"];
  const map = await load(advanced);
  assert.deepEqual(gapsOf(map), []);
  for (const k of advanced) {
    const x = map.get(k);
    assert.ok(x.kinds.includes(k) && typeof x.render === "function" && typeof x.recompute === "function" && typeof x.unit === "function", k);
  }
  for (const m of MODULES.filter((f) => !CORE_MODULES.includes(f))) assert.ok(existsSync(here(`../js/how/explainers/${m}`)), m);
  assert.deepEqual([...new Set(advanced.map((k) => EXPLAINER_OF[k]))].sort(), ["attention.js", "conv.js", "graph.js", "latent.js", "recurrent.js"]);
});

// ---- conv: (L, C) = (4, 2), one output channel, kernel 3, padding 1, stride 1
const CONV_X = [1, 2, 3, 4, 5, 6, 7, 8];                 // position t's channel c at t·C + c
const CONV_Y = [0.25, 2.25, 5.25, 12.25];                // by hand: b + Σ_c Σ_j W[0, c, j]·x[t + j − 1, c]
function convCase({ y = CONV_Y, withheld = [], exact = true } = {}) {
  const graph = new Graph({ nodes: [{ id: "C", type: "conv1d", kind: "conv", in_shape: [4, 2], out_shape: [4, 1], payload: { kernel: "C.kernel", bias: "C.bias", stride: 1, padding: 1, dilation: 1 } }],
    tensors: { "C.kernel": W([1, 0, -1, 0.5, 0.5, 0.5], [1, 2, 3]), "C.bias": W([0.25], [1]) }, withheld,
    inputs: { features: ["a", "b"], steps: ["t−3", "t−2", "t−1", "t"] } });
  const ex = explanation(exact ? { C: { exact: { in: f32(CONV_X, [4, 2]), out: f32(y, [y.length, 1]), grad_out: f32([0, 0, 1, 0].slice(0, y.length)) } } } : {});
  return { graph, ex, step: { kind: "conv", stage: "C", from: { kind: "input" }, pick: null } };
}

test("conv.js: one output at one position, the padding read as zero, as blocks.py computes it", () => {
  const { graph, ex, step } = convCase();
  const chk = conv.recompute(step, ex, graph);
  assert.equal(chk.unit, 2, "the unit that carried most of the target");
  close(chk.page, 5.25, "y[t−1]");
  assert.ok(chk.within && chk.ops[0].what === "map" && chk.ops[0].within);
  for (const [u, want] of CONV_Y.entries()) close(conv.recompute({ ...step, pick: u }, ex, graph).page, want, `y[${u}]`);
  const text = body(conv, { ...step, pick: 0 }, ex, graph);
  assert.match(text, /t−3 · channel 1 reads 3 positions of each of the 2 input channels, from t−3 to t−2/);
  assert.match(text, /read as zero \(a padding of 1 on each side\)/);
  assert.match(text, /a @ t−2/, "a term named by its feature and its position");
  noLeftovers();
});

test("conv.js says what stops it: the answer's missing tensors, withheld weights, numbers that do not fit", () => {
  const missing = convCase({ exact: false }), withheld = convCase({ withheld: ["C.kernel"] }), unfit = convCase({ y: [1, 2, 3] });
  for (const c of [missing, withheld, unfit]) assert.equal(conv.recompute(c.step, c.ex, c.graph), null);
  assert.match(body(conv, missing.step, missing.ex, missing.graph), /did not bring C\.in, C\.out/);
  assert.match(body(conv, withheld.step, withheld.ex, withheld.graph), /did not receive the weights this step needs \(C\.kernel\)/);
  assert.match(body(conv, unfit.step, unfit.ex, unfit.graph), /do not fit its shape 4 × 2 \(C\.in\[8\], C\.out\[3\], C\.kernel\[1×2×3\]\)/);
  noLeftovers();
});

// ---- recurrent cells, by hand
function recCase(cell, { dirs = 1, ret = "last", L = 2, hiddenFrom = null } = {}) {
  const G = { rnn: 1, gru: 3, lstm: 4 }[cell], x = [0.5, -1, 0.25].slice(0, L);
  const w = { ih: Array.from({ length: G }, (_, k) => 0.3 + 0.2 * k), hh: Array.from({ length: G }, (_, k) => 0.9 - 0.3 * k), bi: Array(G).fill(0.1), bh: Array(G).fill(-0.05) };
  const per = Array.from({ length: dirs }, (_, d) => ({ layer: 0, direction: d ? "reverse" : "forward", w_ih: `R.wi${d}`, w_hh: `R.wh${d}`, b_ih: `R.bi${d}`, b_hh: `R.bh${d}` }));
  const tensors = Object.fromEntries(per.flatMap((_, d) => [[`R.wi${d}`, W(w.ih, [G, 1])], [`R.wh${d}`, W(w.hh, [G, 1])], [`R.bi${d}`, W(w.bi, [G])], [`R.bh${d}`, W(w.bh, [G])]]));
  const hidden = hiddenFrom ? hiddenFrom(cell, w, x) : null;
  const out = dirs === 2 ? [hidden[L - 1], hidden[L]] : ret === "all" ? hidden.slice(0, L) : [hidden[L - 1]];
  const graph = new Graph({ nodes: [{ id: "R", type: cell, kind: "recurrent", in_shape: [L, 1], out_shape: ret === "all" ? [L, dirs] : [dirs],
    payload: { cell, hidden: 1, layers: 1, directions: dirs, return: ret, per_layer: per } }], tensors });
  const ex = explanation({ R: { exact: { in: f32(x, [L, 1]), out: f32(out), grad_out: f32(out.map(() => 1)), hidden: f32(hidden, [dirs, L, 1]) } } });
  return { graph, ex, step: { kind: "recurrent", stage: "R", from: { kind: "input" }, pick: null }, x, w };
}

// The states of one direction by the cell equations (torch's documentation), H = F = 1.
function byHand(cell, w, x) {
  let h = 0, c = 0;
  return x.map((xt) => {
    const a = (k) => w.ih[k] * xt + w.bi[k], b = (k) => w.hh[k] * h + w.bh[k];
    if (cell === "rnn") h = Math.tanh(a(0) + b(0));
    else if (cell === "gru") { const r = sig(a(0) + b(0)), z = sig(a(1) + b(1)), n = Math.tanh(a(2) + r * b(2)); h = (1 - z) * n + z * h; }
    else { const i = sig(a(0) + b(0)), f = sig(a(1) + b(1)), g = Math.tanh(a(2) + b(2)), o = sig(a(3) + b(3)); c = f * c + i * g; h = o * Math.tanh(c); }
    return h;
  });
}
const oneWay = (cell, w, x) => byHand(cell, w, x);
const bothWays = (cell, w, x) => [...byHand(cell, w, x), ...byHand(cell, w, x.slice().reverse()).reverse()];

for (const cell of ["rnn", "gru", "lstm"]) {
  test(`recurrent.js: the ${cell} cell unrolled from a zero state, state by state, as torch computes it`, () => {
    const c = recCase(cell, { L: 3, ret: "all", hiddenFrom: oneWay }), chk = recurrent.recompute(c.step, c.ex, c.graph);
    assert.ok(chk.within, JSON.stringify(chk));
    assert.deepEqual(chk.ops.map((o) => o.what), ["states", "out"]);
    for (const o of chk.ops) assert.ok(Math.abs(o.gap) < 1e-6, `${o.what}: ${o.gap}`);
    const text = body(recurrent, c.step, c.ex, c.graph);
    assert.match(text, /Unrolled from a zero state over all 3 positions/);
    if (cell === "lstm") assert.match(text, /new cell c′/);
    if (cell === "gru") assert.match(text, /reset gate r.*update gate z.*candidate n/);
    noLeftovers();
  });
}

test("recurrent.js: a bidirectional layer's last state is the forward one at the end and the backward one at the start", () => {
  const c = recCase("gru", { dirs: 2, L: 3, hiddenFrom: bothWays }), chk = recurrent.recompute(c.step, c.ex, c.graph);
  assert.ok(chk.within);
  const back = recurrent.recompute({ ...c.step, pick: 1 }, c.ex, c.graph);
  assert.equal(back.step, 0, "the backward direction's final state is at the first position");
  close(back.page, bothWays("gru", c.w, c.x)[3], "the backward state at the first position");
  assert.match(body(recurrent, { ...c.step, pick: 1 }, c.ex, c.graph), /reading the sequence backwards/);
  noLeftovers();
});

test("recurrent.js: a state the network did not compute is shown as a gap, never hidden", () => {
  const c = recCase("rnn", { L: 2, hiddenFrom: (cell, w, x) => oneWay(cell, w, x).map((h, t) => (t === 0 ? h + 0.5 : h)) });
  const chk = recurrent.recompute(c.step, c.ex, c.graph);
  assert.equal(chk.within, false);
  assert.equal(chk.ops.find((o) => o.what === "states").at, 0);
});

// ---- attention: self-attention over two positions, one head, identity projections
test("attention.js: self-attention's weights per query and its output, against the formula", () => {
  const I2 = [1, 0, 0, 1], X = [1, 0, 0, 1], s = 1 / Math.sqrt(2);
  const a0 = [Math.exp(s), 1].map((v, _, all) => v / (Math.exp(s) + 1)), a1 = [1, Math.exp(s)].map((v) => v / (Math.exp(s) + 1));
  const Y = [a0[0], a0[1], a1[0], a1[1]];
  const graph = new Graph({ nodes: [{ id: "A", type: "attention", kind: "attention", in_shape: [2, 2], out_shape: [2, 2], payload: { heads: 1, in_proj: "A.in", out_proj: "A.out", pool: false } }],
    tensors: { "A.in": W([...I2, ...I2, ...I2], [6, 2]), "A.in_bias": W([0, 0, 0, 0, 0, 0], [6]), "A.out": W(I2, [2, 2]), "A.out_bias": W([0, 0], [2]) } });
  const ex = explanation({ A: { exact: { in: f32(X, [2, 2]), out: f32(Y, [2, 2]), grad_out: f32([0, 1, 0, 0]), alpha: f32([...a0, ...a1], [1, 2, 2]) } } });
  const step = { kind: "attention", stage: "A", from: { kind: "input" }, pick: null }, chk = attention.recompute(step, ex, graph);
  assert.equal(chk.unit, 1);
  close(chk.page, a0[1], "position 1, channel 2");
  assert.deepEqual(chk.ops.map((o) => o.what), ["alpha", "out"]);
  assert.ok(chk.within && chk.ops.every((o) => Math.abs(o.gap) < 1e-6));
  assert.match(body(attention, step, ex, graph), /The query at position 1: its weights over the 2 positions/);
  noLeftovers();
});

test("attention.js: network α in a shape the page cannot place is reported, not dropped", () => {
  const I2 = [1, 0, 0, 1], X = [1, 0, 0, 1], s = 1 / Math.sqrt(2);
  const a0 = [Math.exp(s), 1].map((v) => v / (Math.exp(s) + 1)), a1 = [1, Math.exp(s)].map((v) => v / (Math.exp(s) + 1));
  const Y = [a0[0], a0[1], a1[0], a1[1]];
  const graph = new Graph({ nodes: [{ id: "A", type: "attention", kind: "attention", in_shape: [2, 2], out_shape: [2, 2], payload: { heads: 1, in_proj: "A.in", out_proj: "A.out", pool: false } }],
    tensors: { "A.in": W([...I2, ...I2, ...I2], [6, 2]), "A.in_bias": W([0, 0, 0, 0, 0, 0], [6]), "A.out": W(I2, [2, 2]), "A.out_bias": W([0, 0], [2]) } });
  // α averaged over the queries ([key]) instead of per head and query ([head, query, key])
  const ex = explanation({ A: { exact: { in: f32(X, [2, 2]), out: f32(Y, [2, 2]), grad_out: f32([0, 1, 0, 0]), alpha: f32([0.5, 0.5], [2]) } } });
  const step = { kind: "attention", stage: "A", from: { kind: "input" }, pick: null }, chk = attention.recompute(step, ex, graph);
  close(chk.page, a0[1], "the output still recomputes");
  const op = chk.ops.find((o) => o.what === "alpha");
  assert.ok(op && op.fit === false && op.within === false && op.sizes === "A.alpha[2]", JSON.stringify(op));
  assert.deepEqual(op.shape, [1, 2, 2]);
  assert.equal(chk.within, false, "a check the page could not make is never within");
  assert.match(body(attention, step, ex, graph), /attention weights \(A\.alpha\[2\]\) do not fit the page's heads × queries × keys 1 × 2 × 2/);
  noLeftovers();
});

test("attention.js: attention pooling's step weights and weighted sum", () => {
  const X = [0, 1], e = [0, 2 * Math.tanh(1)], alpha = e.map((v) => Math.exp(v) / (1 + Math.exp(e[1]))), y = [alpha[1]];
  const graph = new Graph({ nodes: [{ id: "P", type: "attention_pool", kind: "attention_pool", in_shape: [2, 1], out_shape: [1], payload: { score: "P.score", query: "P.query" } }],
    tensors: { "P.score": W([1], [1, 1]), "P.score_bias": W([0], [1]), "P.query": W([2], [1, 1]) } });
  const ex = explanation({ P: { exact: { in: f32(X, [2, 1]), out: f32(y), grad_out: f32([1]), alpha: f32(alpha) } } });
  const step = { kind: "attention_pool", stage: "P", from: { kind: "recurrent" }, pick: null }, chk = attention.recompute(step, ex, graph);
  close(chk.page, alpha[1], "Σ α·h");
  assert.ok(chk.within && chk.ops[0].what === "alpha");
  assert.match(body(attention, step, ex, graph), /Channel 1 of the output is Σₜ αₜ·hₜ over the 2 positions/);
  noLeftovers();
});

test("attention.js: a tokenizer's [cls], numeric and nominal tokens", () => {
  const out = [0.1, 0.2, /* x·w + b */ 2 * 0.5 + 0.1, 2 * -1 + 0.2, /* row 1 of the table */ 7, 8];
  const graph = new Graph({ nodes: [{ id: "T", type: "feature_tokenizer", kind: "tokenizer", in_shape: [1], out_shape: [3, 2],
    payload: { weight: "T.w", bias: "T.b", cls: "T.cls", tables: [{ column: "zone", table: "T.zone", cardinality: 2 }] } },
  { id: "input.cat", type: "input", config: { stream: "nominal" } }],
  tensors: { "T.w": W([0.5, -1], [1, 2]), "T.b": W([0.1, 0.2], [1, 2]), "T.cls": W([0.1, 0.2], [1, 1, 2]), "T.zone": W([5, 6, 7, 8], [2, 2]) },
  inputs: { tokens: ["[cls]", "x", "zone"] }, sample: { shape: "row" } });
  const ex = explanation({ T: { exact: { in: f32([2]), out: f32(out, [3, 2]), grad_out: f32([0, 0, 0, 0, 1, 0]) } }, "input.cat": { exact: { out: i64([1]) } } });
  const step = { kind: "tokenizer", stage: "T", from: { kind: "input" }, pick: null };
  for (let u = 0; u < 6; u++) close(attention.recompute({ ...step, pick: u }, ex, graph).page, out[u], `token value ${u}`);
  assert.match(body(attention, step, ex, graph), /The token of zone is row 1 of its table/);
  assert.match(body(attention, { ...step, pick: 2 }, ex, graph), /The token of x is its value 2\.000 times a learned vector/);
  noLeftovers();
});

// ---- latent
test("latent.js: the code is μ, the log-variance clipped as blocks.py clips it, the KL term recomputed", () => {
  const x = [0.5, -1, 0, 30], s = split(x, 2), kl = -0.5 * ((1 + 0 - 0.25 - 1) + (1 + LOGVAR_CLIP - 1 - Math.exp(LOGVAR_CLIP)));
  assert.deepEqual(s.mu, [0.5, -1]);
  assert.deepEqual(s.logvar, [0, LOGVAR_CLIP]);
  close(s.sigma[1], Math.exp(LOGVAR_CLIP / 2), "σ");
  close(s.kl, kl, "KL");
  const graph = new Graph({ nodes: [{ id: "Z", type: "reparameterize", kind: "latent", in_shape: [4], out_shape: [2], payload: { z: 2 } }], sample: { shape: "row" } });
  const ex = explanation({ Z: { exact: { in: f32(x), out: f32([0.5, -1]), grad_out: f32([0, 1]) }, extra: { kl: { form: "scalar", value: kl }, sigma: { form: "vector", n: 2, values: s.sigma } } } });
  const chk = latent.recompute({ kind: "latent", stage: "Z", pick: null }, ex, graph);
  assert.equal(chk.unit, 1);
  assert.deepEqual(chk.ops.map((o) => o.what), ["code", "sigma", "kl"]);
  assert.ok(chk.within);
  const text = body(latent, { kind: "latent", stage: "Z", pick: null }, ex, graph);
  assert.match(text, /clipped to ±10/);
  assert.match(text, /While the network answers, z = μ/);
  noLeftovers();
});

// ---- graphs
test("graph.js at node level: the node's own message recomputed, the messages' shares adding up to the output's", () => {
  // gcn, F = 2 → 1: W = [1, 2], b = 0.5; the node reads x = [1, 1] (W·x = 3), its own message at
  // c = 0.5 (size 1.5) and one neighbour's at c = 0.25 (W·h = 4: size 1); out = 1.5 + 1 + 0.5 = 3;
  // ∂s/∂out = 2, so the shares are 3 and 2 and the bias's 1: 6 = 2 × 3.
  const graph = new Graph({ nodes: [{ id: "G", type: "gcn_conv", kind: "graph_conv", in_shape: [2], out_shape: [1], payload: { weight: "G.w", bias: "G.b" } }],
    tensors: { "G.w": W([1, 2], [1, 2]), "G.b": W([0.5], [1]) }, sample: { shape: "graph", level: "node" } });
  const msgs = [{ src: 7, dst: 7, hop: 0, root: false, coef: 0.5, norm: 1.5, relevance: 3 }, { src: 3, dst: 7, hop: 1, root: false, coef: 0.25, norm: 1, relevance: 2 }];
  const ex = { ...explanation({ G: { exact: { in: f32([1, 1]), out: f32([3]), grad_out: f32([2]) }, extra: { messages: msgs } } }), sample: { shape: "graph", graph: { level: "node", target_node: 7 } } };
  const step = { kind: "graph_conv", stage: "G", pick: null }, chk = graphX.recompute(step, ex, graph);
  assert.equal(chk.what, "share");
  close(chk.page, 6, "Σ shares + the bias's");
  close(chk.network, 6, "the output's share");
  assert.equal(chk.ops[0].what, "self");
  close(chk.ops[0].page, 1.5, "the size of its own message");
  assert.ok(chk.within);
  const text = body(graphX, step, ex, graph);
  assert.match(text, /The 2 messages into this node/);
  // node 3 is a reference node the answer names by no key: an ordinal of this answer, never its id
  assert.match(text, /neighbour 1 · hop 1/);
  assert.doesNotMatch(text, /#\d|node 3\b/);
  noLeftovers();
});

test("graph.js at graph level: every node's output recomputed from every message (a GIN)", () => {
  // nodes 10, 11 (rows 0, 1), x = [1, 3]; messages 10→10, 11→11 (self terms, ε = 0) and 10→11;
  // the MLP one linear layer, 2·h: out = [2·1, 2·(3 + 1)] = [2, 8]
  const graph = new Graph({ nodes: [{ id: "G", type: "gin_conv", kind: "graph_conv", in_shape: [1], out_shape: [1], payload: { mlp: [{ weight: "G.m0", bias: "G.b0" }], eps: 0 } }],
    tensors: { "G.m0": W([2], [1, 1]), "G.b0": W([0], [1]) }, sample: { shape: "graph", level: "graph" } });
  const msgs = [{ src: 10, dst: 10, hop: 0, root: false, coef: 1, norm: 1, relevance: 0 }, { src: 11, dst: 11, hop: 0, root: false, coef: 1, norm: 3, relevance: 6 },
    { src: 10, dst: 11, hop: 1, root: false, coef: 1, norm: 1, relevance: 2 }];
  const ex = { ...explanation({ G: { exact: { in: f32([1, 3], [2, 1]), out: f32([2, 8], [2, 1]), grad_out: f32([0, 1]) }, extra: { messages: msgs } } }), sample: { shape: "graph", graph: { level: "graph" } } };
  const step = { kind: "graph_conv", stage: "G", pick: null }, chk = graphX.recompute(step, ex, graph);
  assert.equal(chk.unit, 1);
  close(chk.page, 8, "node 11");
  assert.ok(chk.within && chk.ops[0].what === "out");
  // node 11 is the graph's second row: named by its place there, never by its index in the batch
  assert.match(body(graphX, step, ex, graph), /node 2 · unit 1: the sum of its messages plus the bias/);
  assert.doesNotMatch(body(graphX, step, ex, graph), /#\d|node 11\b/);
  noLeftovers();
});

for (const [readout, want] of Object.entries({ sum: [6, 0], mean: [2, 0], max: [3, 1] })) {
  test(`graph.js: the ${readout} readout over a graph's nodes`, () => {
    const graph = new Graph({ nodes: [{ id: "R", type: "graph_pool", kind: "graph_pool", in_shape: [2], out_shape: [2], payload: { readout } }], sample: { shape: "graph", level: "graph" } });
    const ex = explanation({ R: { exact: { in: f32([1, -1, 2, 1, 3, 0], [3, 2]), out: f32(want), grad_out: f32([1, 0]) } } });
    const chk = graphX.recompute({ kind: "graph_pool", stage: "R", pick: null }, ex, graph);
    close(chk.page, want[0], readout);
    assert.ok(chk.within && chk.ops[0].within);
    assert.match(body(graphX, { kind: "graph_pool", stage: "R", pick: null }, ex, graph), new RegExp(`Channel 1 is the ${readout === "max" ? "largest" : readout} over the 3 nodes`));
    noLeftovers();
  });
}

test("graph.js and latent.js say what stops them, and never throw", () => {
  const graph = new Graph({ nodes: [{ id: "G", type: "gcn_conv", kind: "graph_conv", in_shape: [2], out_shape: [1], payload: { weight: "G.w", bias: "G.b" } },
    { id: "Z", type: "reparameterize", kind: "latent", in_shape: [4], out_shape: [2], payload: { z: 2 } }],
  tensors: { "G.w": W([1, 2], [1, 2]), "G.b": W([0.5], [1]) }, withheld: ["G.w"] });
  const ex = explanation({ Z: { exact: { in: f32([1, 2, 3]), out: f32([1, 2]), grad_out: f32([1, 1]) } } });
  assert.equal(graphX.recompute({ kind: "graph_conv", stage: "G" }, ex, graph), null);
  assert.equal(latent.recompute({ kind: "latent", stage: "Z" }, ex, graph), null);
  assert.match(body(graphX, { kind: "graph_conv", stage: "G" }, ex, graph), /did not receive the weights this step needs \(G\.w\)/);
  assert.match(body(latent, { kind: "latent", stage: "Z" }, ex, graph), /do not fit its shape 4 \(Z\.in\[3\], Z\.out\[2\]\)/);
  noLeftovers();
});

// ---- the shared parts
test("cells.js: sizes, positions and channels named from the network view", () => {
  assert.equal(sizesOf("B3", { in: [1, 2, 3], out: null, per_layer: 2 }), "B3.in[3], B3.per_layer[2]");
  words();
  const g = new Graph({ nodes: [], inputs: { features: ["a", { name: "y", past: true }], nominal: [{ name: "zone", dim: 2 }], steps: ["t−1", "t"] } });
  assert.equal(String(positionName(g, 2, 1)), "t");
  assert.equal(positionName(g, 3, 1), "position 2");
  assert.equal(String(channelName(g, { from: { kind: "input" } }, 1)), "y");
  assert.equal(channelName(g, { from: { kind: "input" } }, 3), "zone · dim 2");
  assert.equal(channelName(g, { from: { kind: "conv" } }, 0), "channel 1");
});
