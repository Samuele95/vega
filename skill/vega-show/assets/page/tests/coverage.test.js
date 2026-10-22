// node --test: the page covers what the backend can build (D4 §17.3, items 1–7; BUILD-PLAN W15).
// It reads the forge's catalog.json, the backend's coverage.json and its explain-catalog.json
// (from $VEGA_FORGE, the vega-forge/scripts directory; by default the one beside this skill) and
// holds the page's own tables to them:
//
//   1  every catalog block type has a stage kind (net/kinds.js KIND_OF), and that kind a renderer
//      module (RENDERER_OF) that loads, has every renderer member and lists the kind;
//   2  every stage kind that computes (STEP_KINDS), each group's step (residual, merge), the input
//      and the link steps have an explainer module that loads and lists the kind (none falls back to
//      the generic one); every link of D4 §5.2 (the network view's link set) has its reading in the
//      link explainer's table (how/act.js LINK_READ), and nothing more;
//   3  every catalog task — each sequence task with each of its bases, an anomaly task with each score
//      source, each graph-level task at level graph — derives a view key and a link (core/lane.js),
//      and the view key's task module loads with every member (views/registry.js);
//   4  every sample shape has its shape module, with every member;
//   5  every family's block types are covered by (1);
//   6  every metric of coverage.json has its strings key `metric.<name>`, and every activation its
//      entry in how/act.js ACT;
//   7  the explain catalog has an entry for every block of coverage.json — all 33.
//
// Each item is a function returning its problems, run on the page's tables (no problem) and on a
// copy with one entry taken out, which it must catch (PN9 is item 1's: a catalog block type removed
// from KIND_OF).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { install } from "./dom-shim.js";

install();
const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const FORGE = process.env.VEGA_FORGE || here("../../../../vega-forge/scripts");
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
const CATALOG = read(join(FORGE, "catalog.json"));
const COVERAGE = read(join(FORGE, "backends", "pytorch", "coverage.json"));
const EXPLAIN = read(join(FORGE, "backends", "pytorch", "explain-catalog.json"));

const { KIND_OF, RENDERER_OF, STEP_KINDS, GROUP_STEP } = await import("../js/net/kinds.js");
const renderers = await import("../js/net/renderers/index.js");
const explainers = await import("../js/how/explainers/index.js");
const { LINKS } = await import("../js/net/graph.js");
const { LINK_READ, ACT } = await import("../js/how/act.js");
const { derive, TASKS, BASES, SCORE_SOURCES, GRAPH_LEVEL_TASKS } = await import("../js/core/lane.js");
const { moduleFor, TASK_MODULES, SHAPE_MODULES } = await import("../js/views/registry.js");
const { STRINGS } = await import("../js/core/strings.en.js");

const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const rendererImport = async (file) => (await import(`../js/net/renderers/${file}`)).default;
const explainerImport = async (file) => (await import(`../js/how/explainers/${file}`)).default;

// ---- 1 and 5
async function blockProblems(kindOf, rendererOf) {
  const out = [];
  for (const type of Object.keys(CATALOG.blocks)) {
    if (!own(kindOf, type)) { out.push(`block ${type} has no stage kind`); continue; }
    const kind = kindOf[type], file = rendererOf[kind];
    if (!file) { out.push(`kind ${kind} (${type}) has no renderer`); continue; }
    let mod;
    try { mod = await rendererImport(file); } catch (e) { out.push(`renderer ${file} does not load: ${e.message}`); continue; }
    const lacks = renderers.MEMBERS.filter((m) => !(m in (mod || {})));
    if (lacks.length) out.push(`renderer ${file} lacks ${lacks.join(", ")}`);
    else if (!mod.kinds.includes(kind)) out.push(`renderer ${file} does not list kind ${kind} (${type})`);
  }
  return out;
}

function familyProblems(kindOf) {
  return Object.entries(CATALOG.families).flatMap(([family, f]) =>
    (f.block_types || []).filter((t) => !own(kindOf, t)).map((t) => `family ${family}: block type ${t} has no stage kind`));
}

test("1: every catalog block type has a stage kind and a renderer module that draws it", async () => {
  assert.equal(Object.keys(CATALOG.blocks).length, 33);
  assert.deepEqual(await blockProblems(KIND_OF, RENDERER_OF), []);
  assert.deepEqual(await renderers.load(new Set(Object.values(KIND_OF)), { importer: rendererImport }).then((m) => m.size), new Set(Object.values(KIND_OF)).size);
});

test("1 catches a block type removed from KIND_OF (PN9), and a kind whose renderer does not list it", async () => {
  const { gcn_conv: _, ...less } = KIND_OF;
  assert.deepEqual(await blockProblems(less, RENDERER_OF), ["block gcn_conv has no stage kind"]);
  const moved = { ...RENDERER_OF, latent: "box.js" };
  assert.deepEqual(await blockProblems(KIND_OF, moved), ["renderer box.js does not list kind latent (reparameterize)"]);
});

test("5: every family's block types are covered", () => {
  assert.equal(Object.keys(CATALOG.families).length, 21);
  assert.deepEqual(familyProblems(KIND_OF), []);
  const { attention_pool: _, ...less } = KIND_OF;
  assert.deepEqual(familyProblems(less), ["family lstm-attention: block type attention_pool has no stage kind",
    "family gru-attention: block type attention_pool has no stage kind"]);
});

// ---- 2
async function stepProblems(stepKinds, explainerOf) {
  const kinds = [...new Set([...stepKinds, ...Object.values(GROUP_STEP), ...explainers.ALWAYS])], out = [];
  for (const kind of kinds) {
    const file = explainerOf[kind];
    if (!file) { out.push(`step kind ${kind} has no explainer module`); continue; }
    let mod;
    try { mod = await explainerImport(file); } catch (e) { out.push(`explainer ${file} does not load: ${e.message}`); continue; }
    const lacks = explainers.MEMBERS.filter((m) => !(m in (mod || {})));
    if (lacks.length) out.push(`explainer ${file} lacks ${lacks.join(", ")}`);
    else if (!mod.kinds.includes(kind)) out.push(`explainer ${file} does not list kind ${kind}`);
  }
  return out;
}

test("2: every stage kind that computes, each group's step, the input and the link have an explainer", async () => {
  assert.deepEqual(await stepProblems(STEP_KINDS, explainers.EXPLAINER_OF), []);
  const loaded = await explainers.load(new Set([...STEP_KINDS, ...Object.values(GROUP_STEP)]), null, { importer: explainerImport });
  assert.deepEqual(explainers.gapsOf(loaded), []);
  const { conv: _, ...less } = explainers.EXPLAINER_OF;
  assert.deepEqual(await stepProblems(STEP_KINDS, less), ["step kind conv has no explainer module"]);
});

test("2: every link of D4 §5.2 has its reading in the link explainer, and no other", () => {
  assert.equal(LINKS.length, 8);
  assert.deepEqual([...Object.keys(LINK_READ)].sort(), [...LINKS].sort());
  for (const link of LINKS) assert.equal(typeof LINK_READ[link], "function", link);
  const catalogLinks = Object.keys(CATALOG.inspection.links).sort();
  assert.deepEqual(catalogLinks, [...LINKS].sort(), "the forge catalog's inspection.links are the page's links");
});

// ---- 3
function taskCases() {
  const out = [];
  for (const task of Object.keys(CATALOG.tasks)) {
    const bases = task.startsWith("sequence-") ? Object.keys(CATALOG.sequence_map).filter((b) => CATALOG.sequence_map[b] === task) : [null];
    const sources = task === "anomaly" ? SCORE_SOURCES : [null];
    for (const base of bases) for (const scoreSource of sources) out.push({ task, base, scoreSource, level: null });
  }
  for (const task of CATALOG.graph_level.tasks) out.push({ task, base: null, scoreSource: null, level: "graph" });
  return out;
}

async function taskProblems(taskModules) {
  const out = [];
  for (const lane of taskCases()) {
    let d;
    try { d = derive(lane); } catch (e) { out.push(`${lane.task}/${lane.base}/${lane.scoreSource}: no view key (${e.message})`); continue; }
    if (lane.level === "graph" && !GRAPH_LEVEL_TASKS.includes(lane.task)) out.push(`${lane.task}: not read at graph level`);
    if (!taskModules[d.viewKey]) { out.push(`view key ${d.viewKey} has no task module`); continue; }
    if (!LINKS.includes(d.link)) out.push(`${lane.task}: link ${d.link} is not one of the page's`);
  }
  return out;
}

test("3: every catalog task (bases, score sources, graph level) derives a view key with its task module", async () => {
  assert.deepEqual([...TASKS].sort(), Object.keys(CATALOG.tasks).sort());
  assert.deepEqual(Object.keys(BASES).filter((t) => t.startsWith("sequence-")).sort(), ["sequence-classification", "sequence-regression"]);
  assert.deepEqual([...GRAPH_LEVEL_TASKS].sort(), [...CATALOG.graph_level.tasks].sort());
  assert.deepEqual(await taskProblems(TASK_MODULES), []);
  for (const viewKey of new Set(taskCases().map((l) => derive(l).viewKey))) {
    const mod = await moduleFor("task", viewKey, { importer: (p) => import(new URL(`../js/views/${p.slice(2)}`, import.meta.url)) });
    assert.equal(typeof mod.headLabel, "string", viewKey);
  }
  const { horizon: _, ...less } = TASK_MODULES;
  assert.deepEqual(await taskProblems(less), ["view key horizon has no task module"]);
});

// ---- 4
test("4: every sample shape has its shape module with every member", async () => {
  assert.deepEqual(Object.keys(SHAPE_MODULES).sort(), [...CATALOG.sample_shapes].sort());
  for (const shape of CATALOG.sample_shapes) {
    const mod = await moduleFor("shape", shape, { importer: (p) => import(new URL(`../js/views/${p.slice(2)}`, import.meta.url)) });
    for (const m of ["editor", "sources", "attribution", "inputStep", "drawInput"]) assert.ok(m in mod, `${shape}: ${m}`);
  }
  await assert.rejects(moduleFor("shape", "image", {}), (e) => e.name === "RegistryGap" && e.key === "image");
});

// ---- 6
function wordProblems(strings, act) {
  return [...COVERAGE.metrics.filter((m) => !own(strings, `metric.${m}`)).map((m) => `metric ${m} has no strings key metric.${m}`),
          ...COVERAGE.activations.filter((a) => !own(act, a)).map((a) => `activation ${a} has no ACT entry`)];
}

test("6: every metric has its strings key, every activation its ACT entry", () => {
  assert.equal(COVERAGE.metrics.length >= 30, true);
  assert.deepEqual(wordProblems(STRINGS, ACT), []);
  const { "metric.mase": _, ...less } = STRINGS;
  const { gelu: __, ...fewer } = ACT;
  assert.deepEqual(wordProblems(less, fewer), ["metric mase has no strings key metric.mase", "activation gelu has no ACT entry"]);
});

// ---- 7
function explainProblems(catalog) {
  return COVERAGE.blocks.filter((b) => !own(catalog.blocks || {}, b)).map((b) => `block ${b} has no explain catalog entry`);
}

test("7: the explain catalog has an entry for every block of coverage.json (all 33)", () => {
  assert.equal(COVERAGE.blocks.length, 33);
  assert.deepEqual([...COVERAGE.blocks].sort(), Object.keys(CATALOG.blocks).sort());
  assert.deepEqual(explainProblems(EXPLAIN), []);
  const { custom: _, ...less } = EXPLAIN.blocks;
  assert.deepEqual(explainProblems({ blocks: less }), ["block custom has no explain catalog entry"]);
});
