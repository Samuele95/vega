// node --test: the output step of a multilabel head decides each label against the threshold the
// service decides it with — the output record's per-label one (/v1/schema
// output.threshold_per_label) before the single one the network view states — and its recompute
// carries that decision beside the answer's labels, so the probe sees a stepper that contradicts
// the answer shown above it (gate C: L13 showed 0.508 on every label and "0 labels" while the
// service answered the first two labels).
import { test } from "node:test";
import assert from "node:assert/strict";
import { install } from "./dom-shim.js";

install();
const { thresholdOf, decisionCheck } = await import("../js/how/explainers/link.js");

const logit = (p) => Math.log(p / (1 - p));
const NAMES = ["first", "second", "third"];
// p = 0.337, 0.4, 0.5: against the single 0.508 none passes; against 0.290 / 0.349 / 0.578 the
// first two do — the service's answer
const head = (labels) => ({ link: "sigmoid_each", z: [0.337, 0.4, 0.5].map(logit),
  out: { names: NAMES, answer: { labels } } });
const graph = { task: { threshold: 0.508 } };
const env = { output: { threshold: 0.508, perLabel: { first: 0.29, second: 0.349, third: 0.578 } } };

test("a label's threshold is the record's per-label one before the single threshold", () => {
  const v = head([]);
  assert.deepEqual(NAMES.map((_, j) => thresholdOf(v, graph, j, env)), [0.29, 0.349, 0.578]);
  assert.deepEqual(NAMES.map((_, j) => thresholdOf(v, graph, j, null)), [0.508, 0.508, 0.508], "without the record, the network view's");
  const partial = { output: { threshold: 0.4, perLabel: { second: 0.2 } } };
  assert.deepEqual(NAMES.map((_, j) => thresholdOf(v, { task: {} }, j, partial)), [0.4, 0.2, 0.4], "a label the map leaves out takes the single one");
});

test("the step's decision is the service's: the labels at or above their own thresholds", () => {
  const ok = decisionCheck(head(["first", "second"]), graph, env);
  assert.equal(ok.what, "decision");
  assert.deepEqual(ok.page.sort(), ["first", "second"]);
  assert.deepEqual(ok.network.sort(), ["first", "second"]);
});

test("a step deciding with the single threshold disagrees with the answer, and the check says so", () => {
  const wrong = decisionCheck(head(["first", "second"]), graph, null);
  assert.deepEqual(wrong.page, [], "0.508 on every label: no label passes");
  assert.notDeepEqual(wrong.page.sort(), wrong.network.sort());
});

test("an answer stating no labels gives no decision to check", () => {
  assert.equal(decisionCheck({ ...head([]), out: { names: NAMES, answer: {} } }, graph, env), null);
});
