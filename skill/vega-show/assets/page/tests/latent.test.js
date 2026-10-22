// node --test: net/renderers/latent.js — the latent renderers (D4 P7) on real payloads: the
// network-view@2 and explanation@2 pairs of the inspection zoo (vega-forge/scripts/tests/
// test_inspect_zoo.py --dump DIR; fixtures/latent/<family>.{network,explain}.json are the row cases
// vae anomaly-unlabelled, pca-floor unsupervised, deep-svdd anomaly-labelled and autoencoder
// unsupervised, with a nominal input). What is checked: the renderer loads with the whole interface;
// the waist rule of the latent families (bottleneckOf, waistLayer); one answer's μ, σ and KL read
// from the probe's own record and, without it, split from what the stage read; the sphere's centre
// and the distance recomputed from it; the latent stage laid out, drawn (at rest and answered),
// pointed at and captioned; the rows capped on a long code.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { StageGraph, Focus } from "../js/net/graph.js";
import { load, MEMBERS } from "../js/net/renderers/index.js";
import { rendererFor } from "../js/net/canvas.js";
import { configureText } from "../js/core/text.js";
import { install } from "./dom-shim.js";
import latent, { LATENT_FAMILIES, bottleneckOf, waistLayer, latentOf, sphereOf, sphereAnswer } from "../js/net/renderers/latent.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const read = (p) => JSON.parse(readFileSync(here(p), "utf8"));
const pair = (name) => ({ net: read(`./fixtures/latent/${name}.network.json`), ex: read(`./fixtures/latent/${name}.explain.json`) });
const graphOf = (name) => StageGraph.from(pair(name).net);
const close = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const COLORS = Object.freeze({ acc: "a", neg: "n", mut: "m", fg: "f", fg2: "f2", muted: "mu", alert: "al", bg: "b", line: "l" });

// a 2-D context that records the calls the drawing makes
function recorder() {
  const calls = [];
  const g = new Proxy({ measureText: (s) => ({ width: 6 * String(s).length }) }, {
    get: (o, k) => (k in o ? o[k] : (...args) => { calls.push([k, args]); }),
    set: (o, k, v) => { o[k] = v; return true; },
  });
  return { g, calls, count: (name) => calls.filter((c) => c[0] === name).length, texts: () => calls.filter((c) => c[0] === "fillText").map((c) => c[1][0]) };
}

function withPage(fn) {
  const saved = { gcs: globalThis.getComputedStyle, doc: globalThis.document, win: globalThis.window };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
  install();
  try { configureText({}); return fn(); } finally {
    globalThis.getComputedStyle = saved.gcs; globalThis.document = saved.doc; globalThis.window = saved.win;
  }
}

const ctxOf = (g, extra = {}) => ({ W: 1100, H: 400, narrow: false, mode: "simple", graph: g, env: {}, K: 14, colors: COLORS, boxW: 88,
  collapse: false, compact: false, top: 18, bottom: 36, ordinal: new Map(), waist: null, ...extra });
const stOf = (ctx, acts, extra = {}) => ({ phase: 1, focus: new Focus(null), hover: null, acts: [acts], mode: "simple", n: 1, i: 0, ctx, ...extra });
const latentSeg = (g) => g.segments().find((s) => s.kind === "box" && g.byId(s.stage).kind === "latent");

test("the latent renderer loads through the registry with the whole interface", async () => {
  const map = await load(["latent"]);
  const r = map.get("latent");
  for (const m of MEMBERS) assert.ok(m in r, m);
  assert.deepEqual(r.kinds, ["latent"]);
  assert.equal(r, latent);
  assert.equal(r.detail, false, "a reparameterize stage has no weights: no 'every weight' level");
  assert.ok(rendererFor(map, graphOf("vae"), latentSeg(graphOf("vae"))) === latent);
});

test("the waist of each latent family: the VAE's code, the narrowest encoder layer of an autoencoder and a PCA floor", () => {
  assert.deepEqual([...LATENT_FAMILIES], ["autoencoder", "pca-floor", "vae"]);
  assert.deepEqual(bottleneckOf(graphOf("vae")), { kind: "latent", stage: "B5", units: 4 });
  assert.deepEqual(bottleneckOf(graphOf("pca-floor")), { kind: "dense", stage: "B3", units: 3 }, "a PCA floor's one hidden layer is its waist");
  assert.deepEqual(bottleneckOf(graphOf("autoencoder")), { kind: "dense", stage: "B4", units: 4 }, "the decoder's layers are not the encoder's");
  assert.equal(bottleneckOf(graphOf("deep-svdd")), null);
  assert.equal(bottleneckOf(StageGraph.from(read("./fixtures/network-view/mlp-plain.json"))), null);
  assert.equal(bottleneckOf(null), null);
  const ae = StageGraph.from(read("./fixtures/network-view/autoencoder.json")), b = bottleneckOf(ae);
  assert.equal(b.kind, "dense");
  const widths = ae.denseRuns().flatMap((r) => r.layers.filter((l) => l.group === null).map((l) => ae.byId(l.dense).out_shape.slice(-1)[0]));
  assert.equal(b.units, Math.min(...widths));
});

test("waistLayer: the waist's column in a dense drawing's layers, never the head", () => {
  const pca = graphOf("pca-floor"), run = pca.segments().find((s) => s.kind === "run");
  const model = pca.runModel(run, { head: true });
  assert.deepEqual(model.layers.map((l) => l.id), ["B3", "B4"]);
  assert.equal(waistLayer(pca, model), 0);
  assert.equal(waistLayer(pca, pca.runModel(run, { head: false })), 0);
  assert.equal(waistLayer(graphOf("vae"), model), null, "a VAE's waist is its latent stage, drawn by this renderer");
  assert.equal(waistLayer(graphOf("deep-svdd"), model), null);
  assert.equal(waistLayer(pca, null), null);
});

test("one answer's μ, σ and KL: the probe's own record, else split from what the stage read", () => {
  const { ex } = pair("vae"), node = ex.nodes.B5, own = latentOf(ex, "B5", 4);
  assert.deepEqual(own.mu, node.extra.mu.values);
  assert.deepEqual(own.sigma, node.extra.sigma.values);
  assert.equal(own.kl, node.extra.kl.value);
  assert.deepEqual(own.code, node.out.values, "z = μ at answer time");
  own.mu.forEach((m, j) => assert.ok(close(m, own.code[j]), `z${j} = μ${j}`));
  const bare = structuredClone(ex);
  delete bare.nodes.B5.extra.mu; delete bare.nodes.B5.extra.sigma; delete bare.nodes.B5.extra.kl;
  const split = latentOf(bare, "B5", 4);
  assert.deepEqual(split.mu, node.in.values.slice(0, 4));
  split.sigma.forEach((s, j) => assert.ok(close(s, node.extra.sigma.values[j], 1e-5), `σ${j} = exp(½ log σ²)`));
  assert.ok(close(split.kl, node.extra.kl.value, 1e-5), "KL = −½ Σ (1 + log σ² − μ² − σ²)");
});

test("a reduced reading is not filled in: a top-k σ is no σ; an omitted stage, none at all", () => {
  const { ex } = pair("vae"), cut = structuredClone(ex), n = cut.nodes.B5;
  n.extra.sigma = { form: "vector", n: 4, top: [[0, 0.5]] };
  n.in = { form: "vector", n: 8, summary: { l2: 3 } };
  const r = latentOf(cut, "B5", 4);
  assert.deepEqual(r.mu, ex.nodes.B5.extra.mu.values);
  assert.equal(r.sigma, null);
  assert.equal(r.kl, ex.nodes.B5.extra.kl.value);
  n.level = "omitted";
  assert.equal(latentOf(cut, "B5", 4), null);
  assert.equal(latentOf(ex, "no-such-node", 4), null);
  assert.equal(latentOf(null, "B5", 4), null);
});

test("an exact explanation: μ and σ split from the exact input tensor", () => {
  const { ex } = pair("vae"), exact = structuredClone(ex), n = exact.nodes.B5, values = Float32Array.from(n.in.values);
  n.extra = {}; n.in = { form: "summary" };
  n.exact = { in: { id: "B5.in", encoding: "base64-le", shape: [8], data: Buffer.from(values.buffer).toString("base64") } };
  const r = latentOf(exact, "B5", 4);
  r.mu.forEach((m, j) => assert.ok(close(m, values[j]), `μ${j}`));
  r.sigma.forEach((s, j) => assert.ok(close(s, ex.nodes.B5.extra.sigma.values[j], 1e-5), `σ${j}`));
});

test("the sphere: the head's exported centre, and the distance recomputed from it equals the service's", () => {
  const { net, ex } = pair("deep-svdd"), g = StageGraph.from(net), s = sphereOf(g);
  assert.equal(s.head, "B4");
  assert.equal(s.centre.length, 8);
  assert.equal(s.units, 8);
  assert.equal(s.threshold, net.task.threshold);
  const a = sphereAnswer(s, ex);
  assert.ok(a.gap <= 1e-4 * Math.max(1, Math.abs(a.network)), JSON.stringify(a));
  assert.equal(a.network, ex.outputs.answer.distance);
  assert.equal(sphereOf(graphOf("vae")), null, "no sphere on a reconstruction link");
  assert.equal(sphereAnswer(s, { outputs: { raw: [1, 2], answer: {} } }), null, "an output of another length");
  const bare = StageGraph.from({ ...net, tensors: Object.fromEntries(Object.entries(net.tensors).filter(([k]) => k !== "B4.centre")) });
  assert.equal(sphereOf(bare).centre, null);
  assert.equal(sphereAnswer(sphereOf(bare), ex), null);
});

test("the latent stage laid out: z rows, μ ports above log σ² ports, one code port per row, its caption", () => withPage(() => {
  const g = graphOf("vae"), seg = latentSeg(g), ctx = ctxOf(g);
  assert.deepEqual(latent.measure(seg, ctx), { w: 114, minH: 90 });
  assert.equal(latent.measure(seg, { ...ctx, boxW: 48 }).w, 62, "the overflow rule's narrow boxes");
  const L = latent.layout(seg, { x: 400, y: 18, w: 114, h: 346 }, ctx);
  assert.deepEqual(L.rows, [0, 1, 2, 3]);
  assert.equal(L.in.length, 8);
  assert.deepEqual(L.in.map((p) => p.unit), [0, 1, 2, 3, 4, 5, 6, 7], "the first z units are μ, the last z log σ²");
  assert.ok(Math.max(...L.in.slice(0, 4).map((p) => p.y)) < Math.min(...L.in.slice(4).map((p) => p.y)), "μ above log σ²");
  assert.equal(L.out.length, 4);
  assert.ok(L.out.every((p) => p.v === undefined), "no answer: the code's ports carry no value");
  assert.deepEqual(L.captions.map((c) => c.text), ["LATENT · 4 OF 4"]);
  assert.equal(L.span, 3, "at rest the axis spans the prior's ±3σ");
}));

test("drawn at rest and answered; pointed at; captioned", () => withPage(() => {
  const { ex } = pair("vae"), g = graphOf("vae"), seg = latentSeg(g), box = { x: 400, y: 18, w: 114, h: 346 };
  const rest = recorder(), ctx = ctxOf(g), L0 = latent.layout(seg, box, ctx);
  latent.draw(rest.g, seg, L0, stOf(ctx, null));
  assert.equal(rest.count("arc"), 4, "one dot per number of the code");
  assert.equal(rest.count("fillRect"), 0, "no μ bar before an answer");
  assert.ok(rest.texts().includes("z = μ") && rest.texts().includes("μ") && rest.texts().includes("log σ²"));
  const acts = latent.acts(seg, ex, g);
  assert.deepEqual(acts.mu, ex.nodes.B5.extra.mu.values);
  const on = recorder(), L1 = latent.layout(seg, box, { ...ctx, acts });
  assert.ok(L1.span >= Math.max(...acts.mu.map((m, j) => Math.abs(m) + acts.sigma[j])) - 1e-9, "the axis reaches every μ ± σ");
  assert.ok(L1.out.every((p, k) => close(p.v, acts.code[L1.rows[k]] / L1.span)), "the code's ports carry its values");
  latent.draw(on.g, seg, L1, stOf(ctx, acts));
  assert.equal(on.count("fillRect"), 4, "one μ bar per number");
  assert.ok(on.texts().some((s) => s.startsWith("KL ")), "the KL term under the column");
  const none = recorder();
  latent.draw(none.g, seg, L1, stOf(ctx, acts, { phase: 0 }));
  assert.equal(none.calls.length, 0, "nothing before its share of the growing phase");
  assert.deepEqual(latent.hit(seg, L1, L1.xc, L1.ys[2]), { j: 2 });
  assert.deepEqual(latent.hit(seg, L1, L1.xc, (L1.ys[0] + L1.ys[1]) / 2), { stage: "B5" });
  assert.equal(latent.hit(seg, L1, box.x - 50, L1.ys[0]), null);
  const cap = latent.caption(seg, { j: 2 }, ctx).textContent;
  assert.match(cap, /number 3 of the code's 4/);
  assert.match(cap, /first 4 are the means μ, the last 4 the log-variances/);
  assert.match(latent.caption(seg, { stage: "B5" }, ctx).textContent, /a code of 4 numbers/);
}));

test("a long code: at most 16 rows, the largest |μ| once answered, evenly spread before", () => withPage(() => {
  const g = graphOf("vae"), seg = latentSeg(g), n = g.byId("B5");
  const wide = new StageGraph({ ...g.model, nodes: g.model.nodes.map((x) => (x.id === "B5" ? Object.freeze({ ...n, in_shape: [80], out_shape: [40] }) : x)) });
  const box = { x: 400, y: 18, w: 114, h: 346 }, L = latent.layout(seg, box, ctxOf(wide));
  assert.equal(L.rows.length, 16);
  assert.deepEqual([L.rows[0], L.rows[15]], [0, 39]);
  const mu = Array.from({ length: 40 }, (_, j) => (j % 5 === 0 ? 10 + j : 0.1)), acts = { z: 40, mu, sigma: mu.map(() => 1), kl: 1, code: mu };
  const A = latent.layout(seg, box, { ...ctxOf(wide), acts });
  assert.equal(A.rows.length, 16);
  assert.deepEqual(A.rows, A.rows.slice().sort((a, b) => a - b), "in the code's order");
  assert.ok([0, 5, 10, 15, 20, 25, 30, 35].every((j) => A.rows.includes(j)), "the largest |μ| are kept");
  const short = latent.layout(seg, { ...box, h: 80 }, ctxOf(wide));
  // 80 px: the name (14), the frame's two mark lines (2 × 18) and its edge (2) leave 28 px of rows
  assert.equal(short.rows.length, Math.floor(28 / 9) + 1, "no closer than 9 px apart");
  assert.ok(short.ys.every((y, k) => k === 0 || y - short.ys[k - 1] >= 9));
}));
