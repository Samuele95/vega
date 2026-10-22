// node --test: the page's test hook (js/core/probe.js, R-F1): its members are the vega-show
// catalog's probe.hook_members, every one of them answers once the hook is installed, the union's
// readings come from the modules that registered them (extendProbe), and an error is stated in the
// probe output's form, "kind: text" (vega/page-probe@1).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

globalThis.window = globalThis;
globalThis.addEventListener = globalThis.addEventListener || (() => {});
const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const SHOW = JSON.parse(readFileSync(here("../../../scripts/catalog.json"), "utf8"));
const probe = await import("../js/core/probe.js");

test("the hook's members are the catalog's probe.hook_members", () => {
  assert.deepEqual([...probe.HOOK_MEMBERS].sort(), [...SHOW.probe.hook_members].sort());
  assert.equal(probe.PROBE_SCHEMA, SHOW.probe.schema);
  assert.match(probe.PROBE_VERSION, /^\d+\.\d+\.\d+$/);
});

test("startProbe installs every member; the union reads the registered readings by name", async () => {
  probe.extendProbe("lane", () => ({ view_key: "binary" }), { report: true });
  probe.extendProbe("page", () => ({ schema: "vega/page@1" }));
  probe.extendProbe("graph", () => ({ family: "mlp-plain" }));
  probe.extendProbe("registry", () => ({ resolved: true }), { report: true });
  probe.extendProbe("math", { sigmoid: (z) => 1 / (1 + Math.exp(-z)) });
  probe.extendProbe("missingKeys", () => ["no.such.key"], { report: true });
  probe.extendProbe("field", () => ({ alert_marks: false }), { report: true });
  const hook = probe.startProbe({ ready: Promise.resolve(true), views: () => ["start", "context"], current: () => "start",
                                  story: () => ({ schema: "vega/story@2" }), state: () => ({ stepper: [] }) });
  for (const m of probe.HOOK_MEMBERS) assert.ok(m in hook, m);
  assert.equal(await hook.ready, true);
  assert.deepEqual(hook.views(), ["start", "context"]);
  assert.deepEqual(hook.lane(), { view_key: "binary" });
  assert.deepEqual(hook.page(), { schema: "vega/page@1" });
  assert.deepEqual(hook.graph(), { family: "mlp-plain" });
  assert.deepEqual(hook.registry(), { resolved: true });
  assert.equal(hook.math.sigmoid(0), 0.5);
  assert.deepEqual(hook.missingKeys(), ["no.such.key"]);
  assert.deepEqual(hook.state(), { stepper: [] });
  assert.deepEqual(hook.field(), { alert_marks: false }, "a reading no member names is answered too");
  assert.deepEqual(Object.keys(probe.extensions()).sort(), ["field", "lane", "missingKeys", "registry"]);
  assert.ok(Array.isArray(hook.fetched()));
});

test("a reading that throws says so in its place; an error is 'kind: text'", () => {
  probe.extendProbe("broken", () => { throw new Error("nope"); }, { report: true });
  assert.deepEqual(probe.reading("broken"), { error: "nope" });
  assert.deepEqual(probe.extensions().broken, { error: "nope" });
  assert.equal(probe.reading("never-registered"), null);
  assert.equal(probe.said({ kind: "console", text: "boom" }), "console: boom");
  assert.equal(probe.said("csp: script-src"), "csp: script-src");
});
