// node --test: the drift monitor's request (views/payloads.js monitorBody, read by views/many.js):
// the truth travels only to a bundle whose requests take one (/v1/schema `truth`), and only when a
// row has one. An unsupervised bundle refuses `labels` with a 422 (T09: L07's page stage).
import { test } from "node:test";
import assert from "node:assert/strict";
import { monitorBody } from "../js/views/payloads.js";

const body = { rows: [{ a: 1 }, { a: 2 }] };

test("the truth goes to a bundle that takes one", () => {
  const schema = { truth: { column: "kind", type: "class", sep: null } };
  assert.deepEqual(monitorBody(body, ["a", "b"], schema), { ...body, labels: ["a", "b"] });
});

test("no truth to a bundle without one, nor a truth no row has", () => {
  assert.deepEqual(monitorBody(body, ["x", "y"], { truth: null }), body, "an unsupervised bundle's schema names no truth");
  assert.deepEqual(monitorBody(body, ["x", "y"], {}), body);
  assert.deepEqual(monitorBody(body, ["x", "y"], null), body);
  assert.deepEqual(monitorBody(body, [null, null], { truth: { type: "class" } }), body, "the demo's items carried no truth");
  assert.deepEqual(monitorBody(body, null, { truth: { type: "class" } }), body);
});
