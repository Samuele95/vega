// node --test: the page's output step on a regression-family head that ends on its fixed
// train-fitted target affine (A37): the head's exported buffers `<head>.target_shift` and
// `<head>.target_scale` put the head's standardised z back in the target's units, y = z · scale +
// shift. The link reads y, not z, so the page's reading agrees with the network's; the head's
// weighted sum is still checked against z; the step draws the affine line; and a head without the
// buffers reads z itself, as before.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { install } from "./dom-shim.js";

install();
const { StageGraph } = await import("../js/net/graph.js");
const { Stepper } = await import("../js/how/stepper.js");
const { headValues, linkCheck } = await import("../js/how/explainers/link.js");
const { configureText, missingKeys, unresolved } = await import("../js/core/text.js");
const { Lexicon } = await import("../js/core/lexicon.js");

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const LINEAR = JSON.parse(readFileSync(here("./fixtures/network-view/linear.json"), "utf8"));
const SHIFT = 50, SCALE = 10;

const f32 = (xs) => ({ shape: [xs.length], axes: ["a0"], dtype: "float32", encoding: "base64-le",
  data: Buffer.from(new Float32Array(xs).buffer).toString("base64") });
const buffer = (id, value) => ({ id, node: "B1", role: "buffer", shape: [1], axes: ["unit"], dtype: "float32", encoding: "json",
  data: [value], summary: { ...LINEAR.tensors["B1.bias"].summary, n: 1, mean: value, min: value, max: value, l2: Math.abs(value) } });

// The linear fixture (row regression, the head alone), with or without the fitted affine.
function view(affine) {
  const v = structuredClone(LINEAR);
  if (affine) {
    v.tensors["B1.target_shift"] = buffer("B1.target_shift", SHIFT);
    v.tensors["B1.target_scale"] = buffer("B1.target_scale", SCALE);
  }
  return v;
}

// One explained row: the head read x, its weighted sum is z (the recorded pre), and the network
// answered z · scale + shift in the target's units.
function explanation(graph) {
  const W = graph.matrix("B1.weight"), b = graph.vector("B1.bias")[0];
  const x = Array.from({ length: W.in }, (_, k) => (k % 3) - 1 + k / 10);
  const z = x.reduce((s, xi, k) => s + W.w[k] * xi, b), y = z * SCALE + SHIFT;
  return {
    z, y, ex: {
      schema: "vega/explanation@2", detail: "exact", degraded: false,
      sample: { shape: "row", index: 0, row: 0, span: null },
      target: { kind: "value", index: 0, name: "y", sign: 1, value: y, why: "predicted", unit: "value" },
      outputs: { kind: "value", names: ["y"], link: "identity", raw: [y], read: [y], answer: { value: y } },
      inputs: { order: graph.inputs.names, shaped: x, raw: null },
      nodes: { B1: { level: "exact", exact: { in: f32(x), pre: f32([z]), out: f32([y]), grad_out: f32([1]) } } },
      chain: null, warnings: [],
    },
  };
}

test("the link reads z · scale + shift from the head's exported buffers, and agrees with the network", () => {
  const graph = StageGraph.from(view(true)), { z, y, ex } = explanation(graph);
  const v = headValues({ kind: "link", stage: "B1" }, ex, graph);
  assert.ok(v.affine, "the fitted affine is read from B1.target_shift / B1.target_scale");
  assert.ok(Math.abs(v.z[0] - z) < 1e-4, "z stays the head's own weighted sum");
  assert.ok(Math.abs(v.y[0] - y) < 1e-3, "y is z in the target's units");
  const check = linkCheck(v, graph, 0);
  assert.equal(check.what, "read");
  assert.ok(check.within, `the page's reading ${check.page} and the network's ${check.network}`);
});

test("without the buffers the link reads z itself — and a network answering in raw units would disagree", () => {
  const graph = StageGraph.from(view(false)), { ex } = explanation(graph);
  const v = headValues({ kind: "link", stage: "B1" }, ex, graph);
  assert.equal(v.affine, null);
  assert.equal(v.y, v.z);
  assert.equal(linkCheck(v, graph, 0).within, false, "z is not the raw answer: the affine is what reconciles them");
});

test("the output step draws the affine line and its checks pass", async () => {
  const doc = install();
  const graph = StageGraph.from(view(true)), { ex } = explanation(graph);
  configureText({ lex: new Lexicon(null, null), lane: { viewKey: graph.viewKey, shape: "row" } });
  const host = doc.createElement("div");
  doc.body.append(host);
  const st = await new Stepper({ graph, ex }).mount(host);
  const last = st.steps().length - 1;
  st.show(last);
  const body = doc.getElementById("how-body").textContent;
  assert.match(body, /standardised units/);
  assert.match(body, new RegExp(`× ${SCALE}(\\.0+)? \\+ ${SHIFT}(\\.0+)? = `));
  assert.doesNotMatch(body, /ŷ = z/, "z is the standardised number: the affine line ends on ŷ");
  assert.deepEqual(st.reading().failed, []);
  assert.deepEqual(missingKeys(), []);
  assert.deepEqual(unresolved(), []);
});
