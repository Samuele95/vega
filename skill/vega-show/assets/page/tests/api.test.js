// node --test: core/api.js — every URL is relative (proxy-safe, same origin), a refused answer
// keeps its problem, and a payload of another version is named.
import { test } from "node:test";
import assert from "node:assert/strict";
import { URLS, EXPECT, ProblemError, schemaMismatch } from "../js/core/api.js";

test("every URL is relative and under health, v1/, showcase/ or the report", () => {
  for (const [name, url] of Object.entries(URLS)) {
    assert.ok(!url.startsWith("/") && !/^[a-z]+:/.test(url), `${name}: ${url} is not relative`);
    assert.ok(url === "health" || url.startsWith("v1/") || url.startsWith("showcase/") || url === "report.pdf", `${name}: ${url}`);
  }
});

test("a problem keeps its title, detail and column errors", () => {
  const e = new ProblemError(422, { title: "Unprocessable", detail: "two columns", errors: [{ column: "a", error: "not a number", rows: [1] }] }, "Unprocessable Entity");
  assert.equal(e.status, 422);
  assert.equal(e.message, "Unprocessable — two columns");
  assert.deepEqual(e.errors, [{ column: "a", error: "not a number", rows: [1] }]);
  const bare = new ProblemError(503, {}, "Service Unavailable");
  assert.equal(bare.message, "Service Unavailable");
  assert.deepEqual(bare.errors, []);
});

test("schemaMismatch names the document, what it is and what the page reads", () => {
  assert.equal(schemaMismatch("story", { schema: EXPECT.story }), null);
  assert.deepEqual(schemaMismatch("story", { schema: "vega/story@1" }), { doc: "story", got: "vega/story@1", want: "vega/story@2" });
  assert.deepEqual(schemaMismatch("network", { available: true }), { doc: "network", got: null, want: "vega/network-view@2" });
  assert.equal(schemaMismatch("story", null), null);
});
