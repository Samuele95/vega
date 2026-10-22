// node --test: net/relevance.js and the drawings' choices (renderers/dense.js, dense-detail.js) on the
// R7 bundle's own network and two of its answers (fixtures/network-view/r7-legacy.json, captured by
// capture_r7.mjs from the service with the values the hand-built page's code computed from them).
// Read through StageGraph.fromLegacy, the answers reproduce today's numbers exactly: the hidden
// units' relevance, every connection the "every weight" drawing draws and what it carried, the
// families drawing's kept units, connections and circle sizes; and, without an answer, the families
// drawing's units and connections and the every-weight drawing's connections. The same answers read
// as vega/explanation@2 (their dense chain, the target's index) give the same numbers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { StageGraph, tensorValues } from "../js/net/graph.js";
import { viewOfLegacy } from "./legacy-view.js";
import { readAnswer, relevance, strongest, topInputs, signedNorm, stageValues, EXPLANATION_SCHEMA } from "../js/net/relevance.js";
import { keepByWeights, columnsOf, edgesByWeight, answerDrawing } from "../js/net/renderers/dense.js";
import { topEdges } from "../js/net/renderers/dense-detail.js";
import { familiesOf } from "../js/net/families.js";

const R7 = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/network-view/r7-legacy.json", import.meta.url)), "utf8"));
const graph = StageGraph.from(viewOfLegacy(R7.network));
const model = graph.runModel(graph.segments()[1]);
const mats = model.layers.map((l) => l.W);
const fam = R7.families.map((f) => ({ ...f, members: f.idx.map((i) => R7.network.inputs[i]) }));
const triples = (edges) => edges.map((layer) => layer.map((e) => [e.a, e.b, e.w]));
// JSON writes −0 as 0: the numbers are compared by value (x + 0 turns −0 into 0 and changes nothing else)
const plain = (rows) => rows.map((r) => r.map((v) => v + 0));

// An answer of the v2.2 service in vega/explanation@2's names (D3 §8.2's legacy mapping).
function asV2(ex) {
  const outputs = R7.network.outputs, win = outputs.indexOf(ex.label), L = ex.pre.length;
  const grad = ex.grad_pre.map((g, l) => (l === L - 1 ? outputs.map((_, c) => (c === win ? 1 : 0)) : g));
  return {
    schema: EXPLANATION_SCHEMA, chain: { inputs: ex.inputs, pre: ex.pre, post: ex.activations, grad_pre: grad },
    target: { kind: "logit", index: win, name: ex.label, sign: 1, value: ex.pre[L - 1][win], why: "predicted", unit: "logit" },
    outputs: { kind: "classes", names: outputs, link: "softmax", raw: ex.pre[L - 1], read: outputs.map((c) => ex.probabilities[c]), answer: {} },
    attribution: { method: "grad_x_input", unit: "logit", shape: "row", features: { names: R7.network.inputs, values: ex.inputs, attribution: ex.attribution, grad: ex.attribution }, total: 0, completeness: null },
  };
}

test("the fixture is the R7 network: 65 inputs, two hidden layers of 256, 15 outputs, 86543 weights", () => {
  assert.deepEqual(mats.map((m) => [m.out, m.in]), [[256, 65], [256, 256], [15, 256]]);
  assert.equal(graph.params.total, 86543);
  assert.equal(R7.explanations.length, 2);
  assert.notEqual(R7.expected[0].win, R7.expected[1].win, "two answers of two different classes");
});

R7.explanations.forEach((ex, n) => {
  const want = R7.expected[n];

  test(`answer ${n}: the hidden units' relevance and the answer are today's, exactly`, () => {
    const R = relevance(readAnswer(ex, model.outputs), mats);
    assert.equal(R.win, want.win);
    assert.deepEqual(R.probs, want.probs);
    assert.deepEqual(plain(R.hidden), want.hidden);
    assert.deepEqual(R.inputs, want.inputs);
  });

  test(`answer ${n}: every connection the every-weight drawing draws, with what it carried`, () => {
    const R = relevance(readAnswer(ex, model.outputs), mats);
    assert.deepEqual(strongest(R, mats), want.detail);
    assert.deepEqual(topInputs(R, 6), want.detail_inputs);
    for (const [l, layer] of want.detail.entries()) for (const [i, j, c] of layer.slice(0, 5)) assert.equal(R.edge(l, i, j), c);
  });

  test(`answer ${n}: the families drawing keeps today's units, connections and circle sizes`, () => {
    const R = relevance(readAnswer(ex, model.outputs), mats), A = answerDrawing(R, fam, model.outputs, want.families.k);
    assert.deepEqual(A.keep, want.families.keep);
    assert.deepEqual(A.edges.map((layer) => layer.map((e) => [e.a, e.b, e.w])), want.families.edges);
    assert.deepEqual(A.acts, want.families.acts);
    assert.equal(A.winner, want.win);
  });

  test(`answer ${n}: read as vega/explanation@2, the same numbers`, () => {
    const v2 = asV2(ex), a = readAnswer(v2, model.outputs), R = relevance(a, mats);
    assert.equal(a.win, want.win);
    assert.deepEqual(plain(R.hidden), want.hidden);
    assert.deepEqual(strongest(R, mats), want.detail);
    assert.deepEqual(answerDrawing(R, fam, model.outputs, 12).edges.map((l) => l.map((e) => [e.a, e.b, e.w])), want.families.edges);
  });
});

test("without an answer: today's families drawing (K = 14) and every-weight connections", () => {
  const keep = keepByWeights(mats, R7.drawing.families.k);
  assert.deepEqual(keep, R7.drawing.families.keep);
  assert.deepEqual(triples(edgesByWeight(mats, columnsOf(fam, keep, model.outputs))), R7.drawing.families.edges);
  assert.deepEqual(topEdges(mats), R7.drawing.detail);
});

test("the families of the fixture are what familiesOf makes of their members", () => {
  const groups = fam.map((f, i) => ({ id: `g${i}`, label: f.label, text: "", members: f.members }));
  const got = familiesOf({ groups, names: R7.network.inputs }).families;
  assert.deepEqual(got.map((f) => f.idx), fam.map((f) => f.idx.slice().sort((a, b) => a - b)));
});

test("readAnswer: no chain, no answer; the target's gradient is the explanation's own", () => {
  assert.equal(readAnswer(null, model.outputs), null);
  assert.equal(readAnswer({ schema: EXPLANATION_SCHEMA, chain: null }, model.outputs), null);
  assert.equal(readAnswer({ label: "k0" }, model.outputs), null);
  const v2 = asV2(R7.explanations[0]);
  v2.chain.grad_pre = v2.chain.grad_pre.map((g, l, all) => (l === all.length - 1 ? g.map((_, c) => (c === 3 ? 1 : 0)) : g));
  v2.target.index = 3;
  const R = relevance(readAnswer(v2, model.outputs), mats);
  assert.equal(R.win, 3, "an explanation of another output is drawn towards that output");
  assert.notDeepEqual(R.hidden, R7.expected[0].hidden);
  const noGrad = relevance({ ...readAnswer(R7.explanations[0], model.outputs), grad_pre: null }, mats);
  assert.deepEqual(noGrad.gpre[2], model.outputs.map((_, c) => (c === R7.expected[0].win ? 1 : 0)));
  assert.ok(noGrad.gpre[0].every((g) => g === 1));
});

test("stageValues reads a stage's output from exact tensors or reduced vectors", () => {
  const f32 = new Float32Array([1.5, -2, 0.25]), b64 = Buffer.from(f32.buffer).toString("base64"), decode = tensorValues;
  const ex = { schema: EXPLANATION_SCHEMA, nodes: {
    A: { level: "exact", exact: { in: null, out: null, grad_out: null, post: { shape: [3], axes: ["unit"], dtype: "float32", encoding: "base64-le", data: b64 } } },
    B: { level: "full", out: { form: "vector", n: 3, values: [1, 2, 3] } },
    C: { level: "summary", out: { form: "vector", n: 4, top: [[2, 9]], summary: {} } },
    D: { level: "summary", out: { form: "scalar", value: 4 } },
    E: { level: "omitted", why: "budget" } } };
  assert.deepEqual(stageValues(ex, "A", decode), [1.5, -2, 0.25]);
  assert.deepEqual(stageValues(ex, "B", decode), [1, 2, 3]);
  assert.deepEqual(stageValues(ex, "C", decode), [0, 0, 9, 0]);
  assert.deepEqual(stageValues(ex, "D", decode), [4]);
  assert.equal(stageValues(ex, "E", decode), null);
  assert.equal(stageValues(ex, "Z", decode), null);
  assert.deepEqual(signedNorm([2, -4]), [.5, -1]);
});
