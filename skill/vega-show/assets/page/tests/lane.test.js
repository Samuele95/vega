// node --test: the lane (core/lane.js, D4 §5). page.json#lane gives the task, its base, the view key,
// the sample shape and graph level, the link and the family; the page derives the view key and the
// link from the task again, holds them to the catalogs (the vega-show catalog's view_keys, the
// vega-forge catalog's tasks, sequence_map, graph_level and the one link vocabulary, R-C1) and
// cross-checks the lane against the network the service answers with. A lane that contradicts
// itself or the network is a LaneMismatch (the page's error state: the showcase was built for
// another network, PN8).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  Lane, LaneMismatch, TASKS, SHAPES, LEVELS, BASES, GRAPH_LEVEL_TASKS, VIEW_KEY_OF, LINK_OF, VIEW_KEYS, derive, effectiveOf,
  isLegacyNetwork,
} from "../js/core/lane.js";
import { tText } from "../js/core/text.js";

const PAGE = fileURLToPath(new URL("..", import.meta.url));
const SKILLS = join(PAGE, "..", "..", "..");
const json = (...p) => JSON.parse(readFileSync(join(SKILLS, ...p), "utf8"));
const SHOW = json("vega-show", "scripts", "catalog.json");
const FORGE = json("vega-forge", "scripts", "catalog.json");

// D4 §5.1, row by row: (task, base, score source) → view key, link, goal.
const TABLE = [
  ["multiclass", null, null, "classes", "softmax", "predict"],
  ["binary", null, null, "binary", "sigmoid", "predict"],
  ["multilabel", null, null, "labels", "sigmoid_each", "predict"],
  ["regression", null, null, "value", "identity", "predict"],
  ["ordinal", null, null, "ordinal", "cumulative", "predict"],
  ["unsupervised", null, null, "reconstruction", "reconstruction", "reconstruct"],
  ["anomaly", null, "reconstruction", "anomaly", "reconstruction", "anomaly"],
  ["anomaly", null, "distance", "anomaly", "distance", "anomaly"],
  ["sequence-classification", "binary", null, "binary", "sigmoid", "predict"],
  ["sequence-classification", "multiclass", null, "classes", "softmax", "predict"],
  ["sequence-classification", "multilabel", null, "labels", "sigmoid_each", "predict"],
  ["sequence-regression", "regression", null, "value", "identity", "predict"],
  ["sequence-regression", "ordinal", null, "ordinal", "cumulative", "predict"],
  ["forecast", null, null, "horizon", "horizon", "predict"],
];
// The reading each (effective task, score source) answers in (R-C1's table), for the forge links.
const READING = { binary: "binary", multiclass: "multiclass", multilabel: "multilabel", regression: "regression", ordinal: "ordinal",
  unsupervised: "reconstruction", forecast: "forecast", "anomaly:reconstruction": "anomaly-reconstruction", "anomaly:distance": "anomaly-svdd" };

// page.json#lane of a table row on a shape and level.
function pageLane([task, base, source, viewKey, link, goal], shape = "row", level = null) {
  return { task, base, view_key: viewKey, shape, level, link, family: "mlp-plain", goal, score_source: source,
           source: "netspec:task, netspec:data_contract.sample, netspec:family, netspec:loss" };
}

// The network-view@2 task and sample of the same lane (only what the cross-check reads).
function network(lane, over = {}) {
  return { schema: "vega/network-view@2", available: true, why: null, family: lane.family,
           task: { kind: lane.task, base: lane.base, link: lane.link, ...(over.task || {}) },
           sample: { shape: lane.shape, level: lane.level, ...(over.sample || {}) } };
}

function mismatch(fn, field) {
  try { fn(); } catch (e) {
    assert.ok(e instanceof LaneMismatch, `a LaneMismatch, not ${e && e.name}`);
    assert.equal(e.field, field);
    return e;
  }
  assert.fail(`no LaneMismatch at ${field}`);
}

// ---- the closed sets are the catalogs'
test("the tasks, shapes, sequence bases and graph-level tasks are the vega-forge catalog's", () => {
  assert.deepEqual([...TASKS].sort(), Object.keys(FORGE.tasks).sort());
  assert.deepEqual([...SHAPES].sort(), [...FORGE.sample_shapes].sort());
  assert.deepEqual([...GRAPH_LEVEL_TASKS].sort(), [...FORGE.graph_level.tasks].sort());
  for (const [task, bases] of Object.entries(BASES)) {
    const want = Object.entries(FORGE.sequence_map).filter(([, kind]) => kind === task).map(([b]) => b).sort();
    assert.deepEqual([...bases].sort(), want, task);
  }
  assert.deepEqual(Object.keys(BASES).sort(), [...new Set(Object.values(FORGE.sequence_map))].sort());
  assert.deepEqual([...LEVELS], ["node", "graph"]);
});

test("the view key of every task and base is the vega-show catalog's view_keys", () => {
  for (const task of TASKS) {
    const want = SHOW.view_keys[task];
    for (const base of BASES[task] || [null]) {
      const source = task === "anomaly" ? "reconstruction" : null;
      assert.equal(derive({ task, base, scoreSource: source }).viewKey, typeof want === "string" ? want : want[base], `${task} ${base}`);
    }
  }
  assert.deepEqual([...VIEW_KEYS].sort(), [...new Set(Object.values(VIEW_KEY_OF))].sort());
  assert.equal(VIEW_KEYS.length, 8, "eight view keys cover the ten task kinds");
});

test("the link of every lane is the one link vocabulary's (vega-forge catalog inspection.links, R-C1)", () => {
  const linkOf = (reading) => Object.entries(FORGE.inspection.links).filter(([, s]) => reading in s.readings).map(([l]) => l);
  for (const row of TABLE) {
    const [task, base, source, , link] = row, eff = effectiveOf(task, base);
    const reading = READING[task === "anomaly" ? `anomaly:${source}` : eff];
    assert.deepEqual(linkOf(reading), [link], `${task} ${base} ${source}`);
    if (task !== "anomaly") assert.equal(LINK_OF[eff], link);
  }
  assert.deepEqual(Object.keys(FORGE.inspection.links).sort(), ["cumulative", "distance", "horizon", "identity", "reconstruction", "sigmoid", "sigmoid_each", "softmax"]);
});

// ---- D4 §5.1 on every shape and level
test("every task kind, sequence base and score source derives D4 §5.1's view key, link and goal", () => {
  assert.equal(new Set(TABLE.map((r) => r[0])).size, 10, "all ten task kinds");
  for (const row of TABLE) {
    const [task, base, scoreSource, viewKey, link, goal] = row;
    assert.deepEqual(derive({ task, base, scoreSource }), { viewKey, link, goal }, `${task} ${base} ${scoreSource}`);
  }
});

test("a lane reads on rows and windows, at node level on graphs, and at graph level only for binary, multiclass and regression", () => {
  for (const row of TABLE) {
    for (const [shape, level] of [["row", null], ["window", null], ["graph", "node"], ["graph", "graph"]]) {
      const p = pageLane(row, shape, level), eff = effectiveOf(row[0], row[1]);
      if (level === "graph" && !GRAPH_LEVEL_TASKS.includes(eff)) {
        mismatch(() => Lane.from(p, null), "level");
        continue;
      }
      const lane = Lane.from(p, network(p));
      assert.equal(lane.viewKey, row[3]);
      assert.equal(lane.link, row[4]);
      assert.equal(lane.shape, shape);
      assert.equal(lane.level, level);
      assert.equal(lane.effective, eff);
      assert.equal(lane.scoreSource, row[2]);
      assert.deepEqual(lane.variants(), [`${row[3]}+${shape}`, row[3], shape]);
      const back = lane.toJSON();
      for (const k of ["task", "base", "view_key", "shape", "level", "link", "family", "goal", "score_source"]) assert.equal(back[k], p[k], k);
      assert.ok(Object.isFrozen(lane));
    }
  }
});

test("the three graph-level tasks keep their view keys at graph level", () => {
  const got = TABLE.filter((r) => GRAPH_LEVEL_TASKS.includes(r[0]) && r[1] === null).map((r) => Lane.from(pageLane(r, "graph", "graph")).viewKey);
  assert.deepEqual(got.sort(), ["binary", "classes", "value"]);
});

// ---- mismatches
test("PN8: a link the page's task does not derive is a LaneMismatch (the lane does not hold together)", () => {
  const binary = pageLane(TABLE[1]);
  const e = mismatch(() => Lane.from({ ...binary, link: "softmax" }, network(binary)), "link");
  assert.equal(e.page, "softmax");
  assert.equal(e.network, null);
  assert.equal(e.message, tText("app.lane_mismatch", { what: tText("lane.mismatch.self", { field: "link", page: "\"softmax\"" }) }));
});

test("a lane that holds together but was built for another network is a LaneMismatch at the first field that differs", () => {
  const classes = pageLane(TABLE[0]), binary = pageLane(TABLE[1]);
  let e = mismatch(() => Lane.from(classes, network(binary)), "task");
  assert.deepEqual([e.page, e.network], ["multiclass", "binary"]);
  e = mismatch(() => Lane.from(classes, network(classes, { task: { link: "sigmoid" } })), "link");
  assert.deepEqual([e.page, e.network], ["softmax", "sigmoid"]);
  e = mismatch(() => Lane.from(classes, network(classes, { sample: { shape: "window" } })), "shape");
  assert.deepEqual([e.page, e.network], ["row", "window"]);
  const seq = pageLane(TABLE[9], "window");
  e = mismatch(() => Lane.from(seq, network(seq, { task: { base: "binary" } })), "base");
  assert.deepEqual([e.page, e.network], ["multiclass", "binary"]);
  const node = pageLane(TABLE[0], "graph", "node");
  e = mismatch(() => Lane.from(node, network(node, { sample: { level: "graph" } })), "level");
  assert.deepEqual([e.page, e.network], ["node", "graph"]);
});

test("a page lane that contradicts itself is refused before the network is read", () => {
  const classes = pageLane(TABLE[0]);
  mismatch(() => Lane.from({ ...classes, task: "clustering" }), "task");
  mismatch(() => Lane.from({ ...classes, view_key: "binary" }), "view_key");
  mismatch(() => Lane.from({ ...classes, goal: "anomaly" }), "goal");
  mismatch(() => Lane.from({ ...classes, shape: "image" }), "shape");
  mismatch(() => Lane.from({ ...classes, level: "node" }), "level");
  mismatch(() => Lane.from({ ...classes, family: "" }), "family");
  mismatch(() => Lane.from({ ...classes, base: "binary" }), "base");
  mismatch(() => Lane.from(pageLane(TABLE[9], "graph", null)), "level");
  mismatch(() => Lane.from({ ...pageLane(TABLE[6]), score_source: null }), "score_source");
  mismatch(() => Lane.from({ ...classes, score_source: "distance" }), "score_source");
  mismatch(() => Lane.from({ ...pageLane(TABLE[6]), score_source: "distance" }), "link");
  mismatch(() => Lane.from(null), "task");
});

test("the mismatch speaks the strings table's words, with the values as data", () => {
  const classes = pageLane(TABLE[0]), binary = pageLane(TABLE[1]);
  const e = mismatch(() => Lane.from(classes, network(binary)), "task");
  const what = tText("lane.mismatch.field", { field: "task", page: "\"multiclass\"", network: "\"binary\"" });
  assert.equal(e.message, tText("app.lane_mismatch", { what }));
  assert.ok(!/lane\.mismatch|app\.lane/.test(e.message), "every key resolved");
});

// ---- no network to hold the lane to
test("without a network (inspection off, not read) or with an unavailable one, the lane is not cross-checked", () => {
  const p = pageLane(TABLE[3]);
  assert.equal(Lane.from(p, null).viewKey, "value");
  assert.equal(Lane.from(p, { schema: "vega/network-view@2", available: false, why: "custom_only" }).link, "identity");
});

test("a v2.2 network (a dense chain on rows, no schema) holds only a row lane", () => {
  const legacy = { layers: [{ in: 3, out: 2 }], outputs: ["a", "b"], inputs: ["x", "y", "z"] };
  assert.ok(isLegacyNetwork(legacy));
  assert.ok(!isLegacyNetwork({ schema: "vega/network-view@2", layers: [] }));
  assert.equal(Lane.from(pageLane(TABLE[0]), legacy).viewKey, "classes");
  const e = mismatch(() => Lane.from(pageLane(TABLE[4], "window"), legacy), "shape");
  assert.deepEqual([e.page, e.network], ["window", "row"]);
});
