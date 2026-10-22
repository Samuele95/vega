// node --test: net/graph.js, net/segments.js, net/kinds.js, net/families.js, net/guide.js — the stage
// graph adapter over vega/network-view@2 (the fixtures under fixtures/network-view, made by
// make_fixtures.py and valid against the forge's schema) and over the v2.2-merge payload of the R7
// bundle (fixtures/network-view/r7-legacy.json): the kinds of all 33 block types, the segments the
// canvas lays out, the dense runs, the pure dense chain, the focus of a decision, the stepper's kinds,
// the decoded weights, the input families, the reading guide's words.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { StageGraph, NetworkViewError, Focus, VIEW_SCHEMA, LINKS } from "../js/net/graph.js";
import { KIND_OF, STAGE_KINDS, RENDERER_OF, SYNTHETIC, kindOf, activationName } from "../js/net/kinds.js";
import { familiesOf, familyOf } from "../js/net/families.js";
import { guideSteps, joinList } from "../js/net/guide.js";
import { load, MEMBERS, MODULES, RendererGap } from "../js/net/renderers/index.js";
import { configureText, hasKey } from "../js/core/text.js";
import { spreadRow, drawCaptions, rendererFor, connectorLines, complete, fitRow, FIT_STAGES, GROUP_PAD } from "../js/net/canvas.js";
import head, { winners, ordinalAnswer, unitErrors, distanceOf, outputsOf, supportOf } from "../js/net/renderers/head.js";
import group, { skipShare, branchShares, rowsOf, shownRepeats } from "../js/net/renderers/group.js";
import { groupChild, lastStageOf } from "../js/net/segments.js";
import { stageNorm } from "../js/net/relevance.js";
import { tensorValues } from "../js/net/graph.js";
import { viewOfLegacy } from "./legacy-view.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const FIX = (name) => JSON.parse(readFileSync(here(`./fixtures/network-view/${name}.json`), "utf8"));
const FORGE = JSON.parse(readFileSync(here("../../../../vega-forge/scripts/catalog.json"), "utf8"));
const graph = (name) => StageGraph.from(FIX(name));
const ALL = ["mlp-plain", "linear", "mlp-residual", "embedding", "autoencoder", "parallel", "lstm-window", "gcn-node", "ordinal", "svdd", "forecast", "recon-nominal", "residual-deep"];
const R7 = FIX("r7-legacy");
const shape = (segs) => segs.map((s) => s.kind === "run" ? { run: s.stages, fused: s.layers.map((l) => l.fused), input: s.input }
  : s.kind === "box" ? { box: s.stage } : s.kind === "head" ? { head: s.stage } : s.kind === "input" ? "input"
    : { group: s.group, mode: s.mode, inside: (s.repeats || s.branches).map((b) => shape(b.segments)) });

// ---- kinds
test("every catalog block type has a stage kind, every stage kind a renderer module (27 kinds)", () => {
  assert.deepEqual(Object.keys(KIND_OF).sort(), Object.keys(FORGE.blocks).sort());
  assert.equal(Object.keys(FORGE.blocks).length, 33);
  assert.equal(STAGE_KINDS.length, 27);
  assert.deepEqual(Object.keys(RENDERER_OF).sort(), [...STAGE_KINDS].sort());
  const d4 = ["input.js", "dense.js", "box.js", "group.js", "conv.js", "recurrent.js", "attention.js", "graph.js", "latent.js", "head.js"];
  for (const m of MODULES) assert.ok(d4.includes(m), `${m} is one of D4's renderer modules`);
  assert.deepEqual([...SYNTHETIC], FORGE.inspection.synthetic);
  assert.equal(kindOf("gat_conv"), "graph_conv");
  assert.equal(kindOf("output"), null);
  assert.equal(activationName("GELU"), "gelu");
  assert.equal(activationName("Identity"), "none");
  assert.deepEqual([...LINKS].sort(), Object.keys(FORGE.inspection.links).sort());
});

test("the renderers this package draws with load, each with the whole interface and its kinds", async () => {
  const own = STAGE_KINDS.filter((k) => ["input.js", "dense.js", "box.js", "group.js", "head.js"].includes(RENDERER_OF[k]));
  const map = await load(own);
  assert.deepEqual([...map.keys()], own);
  for (const [kind, r] of map) {
    for (const m of MEMBERS) assert.ok(m in r, `${kind}: ${m}`);
    assert.ok(r.kinds.includes(kind), kind);
  }
  assert.ok(map.get("dense").Run && map.get("dense").Detail, "the dense module carries the pure chain's two drawings");
  await assert.rejects(load(["no_such_kind"]), (e) => e instanceof RendererGap && e.why === "net.renderer.unmapped" && /no_such_kind/.test(e.message));
  const missing = STAGE_KINDS.filter((k) => !existsSync(here(`../js/net/renderers/${RENDERER_OF[k]}`)));
  for (const k of missing) await assert.rejects(load([k]), RendererGap, `${k}: a module not written yet is a gap, never a silent box`);
});

test("kinds of two modules that do not load: one RendererGap, and no rejection left unhandled", async () => {
  const unhandled = [], seen = (e) => unhandled.push(e);
  process.on("unhandledRejection", seen);
  try {
    const own = (file) => import(`../js/net/renderers/${file}`).then((m) => m.default);
    const failing = (file) => (["recurrent.js", "attention.js"].includes(file) ? Promise.reject(new Error(`no ${file}`)) : own(file));
    await assert.rejects(load(["input", "recurrent", "attention_pool", "head"], { importer: failing }),
      (e) => e instanceof RendererGap && e.kind === "recurrent" && e.module === "recurrent.js" && e.why === "net.renderer.unloaded" && /no recurrent\.js/.test(e.message));
    const throwing = (file) => { if (file === "head.js") throw new Error("head.js broke"); return own(file); };
    await assert.rejects(load(["input", "head"], { importer: throwing }), (e) => e instanceof RendererGap && e.kind === "head");
    await assert.rejects(load(["input", "recurrent", "no_such_kind"], { importer: failing }), (e) => e.why === "net.renderer.unmapped",
      "every kind is mapped before any module is imported");
    // the modules not written yet, two of them at once (lstm + attention pooling, conv + recurrent, …)
    const missing = [...new Map(STAGE_KINDS.filter((k) => !existsSync(here(`../js/net/renderers/${RENDERER_OF[k]}`))).map((k) => [RENDERER_OF[k], k])).values()];
    if (missing.length >= 2) await assert.rejects(load(["input", ...missing, "head"]), (e) => e instanceof RendererGap && e.kind === missing[0]);
    await new Promise((r) => setTimeout(r, 50));
    assert.deepEqual(unhandled.map((e) => `${e.name}: ${e.message}`), []);
  } finally { process.off("unhandledRejection", seen); }
});

// ---- reading the payloads
test("from() reads every fixture and refuses what is not a stage graph", () => {
  for (const name of ALL) {
    const g = graph(name), p = FIX(name);
    assert.equal(g.task.link, p.task.link);
    assert.equal(g.params.total, p.params.total);
    assert.equal(g.stages().length, p.nodes.filter((n) => !SYNTHETIC.includes(n.type)).length, name);
  }
  const p = FIX("mlp-plain");
  const bad = (edit, key, re) => {
    const q = JSON.parse(JSON.stringify(p)); edit(q);
    assert.throws(() => StageGraph.from(q), (e) => e instanceof NetworkViewError && e.key === key && re.test(e.message), key);
  };
  bad((q) => { q.schema = "vega/network-view@1"; }, "net.error.schema", /expected vega\/network-view@2, found "vega\/network-view@1"/);
  bad((q) => { q.nodes[1].type = "dense2"; }, "net.error.node_type", /unknown type/);
  bad((q) => { q.edges.push({ from: "B1", to: "nowhere", kind: "data" }); }, "net.error.edge", /unknown node/);
  bad((q) => { q.nodes[1].params.weight = "B9.weight"; }, "net.error.node_tensor", /not in tensors/);
  bad((q) => { q.task.link = "svdd"; }, "net.error.link", /unknown link/);
  bad((q) => { q.nodes[2].id = q.nodes[1].id; }, "net.error.node_id", /repeated/);
  bad((q) => { delete q.chain; }, "net.error.field", /no "chain"/);
  bad((q) => { q.available = false; }, "net.error.unavailable", /not available/);
  bad((q) => { q.nodes = []; }, "net.error.no_nodes", /no nodes/);
  for (const key of ["net.error.not_object", "net.error.task_field", "net.error.node_parent", "net.error.node_child", "net.error.tensor",
    ]) assert.ok(hasKey(key), key);
  assert.throws(() => StageGraph.from(null), (e) => e.key === "net.error.not_object");
  assert.equal(StageGraph.fromLegacy, undefined, "the page reads network-view@2 only (W15)");
  assert.throws(() => StageGraph.from(R7.network), (e) => e.key === "net.error.schema" && /found null/.test(e.message), "a v2.2 payload is refused by name");
  assert.throws(() => graph("linear").attachTensor({ id: "nope" }), (e) => e.key === "net.error.tensor");
  assert.equal(p.schema, VIEW_SCHEMA);
});

test("the top-level stages come in forward order, children under their parents", () => {
  assert.deepEqual(graph("gcn-node").topLevel(), ["B1", "B2", "B3", "B4", "B5", "B6"]);
  const g = graph("parallel");
  assert.deepEqual(g.topLevel(), ["B1", "B2", "B3", "B4"]);
  assert.deepEqual(g.childrenOf("B2").map((n) => n.id), ["B2.b0", "B2.b1", "B2.merge"]);
  assert.deepEqual(g.descendants("B2"), ["B2", "B2.b0", "B2.b1", "B2.b1.0", "B2.b1.1", "B2.merge"]);
  assert.equal(g.byId("B2.merge").kind, "parallel_merge");
  assert.equal(g.head().id, "B4");
});

// ---- segments and dense runs
test("segments: runs with their fused stages, boxes, groups, the head", () => {
  assert.deepEqual(shape(graph("mlp-plain").segments()), ["input", { run: ["B1", "B4"], fused: [["B2", "B3"], []], input: [] }, { head: "B5" }]);
  assert.deepEqual(shape(graph("linear").segments()), ["input", { head: "B1" }]);
  assert.deepEqual(shape(graph("mlp-residual").segments()), ["input", { run: ["B1"], fused: [[]], input: [] },
    { group: "B2", mode: "residual", inside: [[{ run: ["B2.r0.0"], fused: [["B2.r0.1", "B2.r0.2", "B2.r0.3"]], input: [] }]] }, { head: "B3" }]);
  assert.deepEqual(shape(graph("embedding").segments()), ["input", { run: ["B3"], fused: [[]], input: ["B1", "B2"] }, { head: "B4" }]);
  assert.deepEqual(shape(graph("autoencoder").segments()), ["input", { run: ["B1.0", "B1.1", "B2", "B3.l0", "B3.l1"], fused: [[], [], [], [], []], input: [] }, { head: "B4" }]);
  assert.deepEqual(shape(graph("parallel").segments()), ["input", { run: ["B1"], fused: [[]], input: [] },
    { group: "B2", mode: "parallel", inside: [[{ run: ["B2.b0"], fused: [[]], input: [] }], [{ run: ["B2.b1.0"], fused: [["B2.b1.1"]], input: [] }]] }, { box: "B3" }, { head: "B4" }]);
  assert.deepEqual(shape(graph("lstm-window").segments()), ["input", { box: "B1" }, { box: "B2" }, { head: "B3" }]);
  assert.deepEqual(shape(graph("gcn-node").segments()), ["input", { box: "B1" }, { box: "B2" }, { box: "B3" }, { box: "B4" }, { box: "B5" }, { head: "B6" }]);
  assert.deepEqual(shape(graph("ordinal").segments()), ["input", { run: ["B1"], fused: [[]], input: [] }, { head: "B2" }]);
  assert.deepEqual(shape(graph("svdd").segments()), ["input", { run: ["B1"], fused: [["B2", "B3"]], input: [] }, { head: "B4" }]);
  assert.deepEqual(shape(graph("forecast").segments()), ["input", { box: "B1" }, { run: ["B2"], fused: [[]], input: [] }, { head: "B3" }]);
  assert.deepEqual(shape(graph("recon-nominal").segments()), ["input", { run: ["B3"], fused: [[]], input: ["B1", "B2"] }, { head: "B4" }]);
  assert.deepEqual(shape(graph("residual-deep").segments()), ["input", { run: ["B1"], fused: [[]], input: [] },
    { group: "B2", mode: "residual", inside: [0, 1, 2].map((r) => [{ run: [`B2.r${r}.0`], fused: [[`B2.r${r}.1`]], input: [] }]) }, { box: "B3" }, { box: "B4" }, { head: "B5" }]);
  const deep = graph("residual-deep").segments()[2];
  assert.deepEqual(deep.repeats.map((r) => [r.projection, r.add]), [["B2.r0.proj", "B2.r0.add"], [null, "B2.r1.add"], [null, "B2.r2.add"]]);
  assert.deepEqual([0, 1, 2].map((r) => lastStageOf(deep.repeats[r].segments)), ["B2.r0.1", "B2.r1.1", "B2.r2.1"], "a repeat ends in the bend fused into its dense stage");
  assert.equal(groupChild(deep, [2, 0]), deep.repeats[2].segments[0]);
  const par = graph("parallel").segments()[2];
  assert.deepEqual([lastStageOf(par.branches[0].segments), lastStageOf(par.branches[1].segments), lastStageOf([par])], ["B2.b0", "B2.b1.1", "B2.merge"]);
});

test("a fused stage's bend, rate and normalisation ride on its dense layer; a decoder's layers say so", () => {
  const [, run] = graph("mlp-plain").segments();
  assert.deepEqual(run.layers.map((l) => [l.activation, l.dropout, l.norm]), [["relu", 0.1, null], ["gelu", 0, null]]);
  const res = graph("mlp-residual").segments()[2].repeats[0];
  assert.deepEqual([res.add, res.projection], ["B2.r0.add", null]);
  assert.deepEqual(res.segments[0].layers.map((l) => [l.activation, l.dropout, l.norm]), [["relu", 0.2, "B2.r0.1"]]);
  const ae = graph("autoencoder").segments()[1];
  assert.deepEqual(ae.layers.map((l) => l.group), [null, null, null, "B3", "B3"]);
});

test("dense runs, inside groups too; the pure dense chain", () => {
  const ids = (name) => graph(name).denseRuns().map((r) => r.stages);
  assert.deepEqual(ids("mlp-plain"), [["B1", "B4"]]);
  assert.deepEqual(ids("mlp-residual"), [["B1"], ["B2.r0.0"]]);
  assert.deepEqual(ids("parallel"), [["B1"], ["B2.b0"], ["B2.b1.0"]]);
  assert.deepEqual(ids("lstm-window"), []);
  const pure = Object.fromEntries(["mlp-plain", "linear", "mlp-residual", "embedding", "autoencoder", "parallel", "lstm-window", "gcn-node"].map((n) => [n, graph(n).isPureDenseChain()]));
  assert.deepEqual(pure, { "mlp-plain": true, linear: true, "mlp-residual": false, embedding: false, autoencoder: false, parallel: false, "lstm-window": false, "gcn-node": false });
  assert.equal(StageGraph.from(viewOfLegacy(R7.network)).isPureDenseChain(), true);
  const twoBends = FIX("mlp-plain"); twoBends.nodes.find((n) => n.id === "B1").params.activation = "tanh";
  assert.equal(StageGraph.from(twoBends).isPureDenseChain(), false, "a dense stage's own bend and a bend block after it: not today's chain");
});

test("the stepper's kinds: the input, what computes, a residual's sum, a parallel's merge", () => {
  const kinds = (name) => [...graph(name).stepKinds()].sort();
  assert.deepEqual(kinds("mlp-plain"), ["dense", "head", "input"]);
  assert.deepEqual(kinds("linear"), ["head", "input"]);
  assert.deepEqual(kinds("mlp-residual"), ["dense", "head", "input", "residual"]);
  assert.deepEqual(kinds("parallel"), ["custom", "dense", "head", "input", "merge"], "D4 §8.1: a parallel group's step is its merge");
  assert.deepEqual(kinds("residual-deep"), ["dense", "head", "input", "norm", "residual"], "a dropout standing alone is a note, not a step");
  assert.deepEqual(kinds("lstm-window"), ["head", "input", "recurrent"]);
  assert.deepEqual(kinds("gcn-node"), ["activation", "graph_conv", "head", "input"], "a bend standing alone is a step of its own");
});

// ---- focus
test("focusSet: blocks, families, features, everything; today's region words on a v2.2 graph", () => {
  const g = graph("mlp-plain"), set = (f) => [...g.focusSet(f)].sort();
  assert.deepEqual(set({ kind: "block", ref: "B2" }), ["B1#act", "B2"]);
  assert.deepEqual(set({ kind: "block", ref: "B5" }), ["B5", "head"]);
  assert.deepEqual(set({ kind: "family", ref: "objective" }), ["B5", "head"]);
  assert.deepEqual(set({ kind: "family", ref: "optimization" }), ["edges"]);
  assert.deepEqual(set({ kind: "family", ref: "regularization" }), ["B1#drop", "B3"]);
  assert.deepEqual(set({ kind: "family", ref: "architecture" }), ["B1", "B2", "B3", "B4", "B5", "head", "input"]);
  assert.deepEqual(set({ kind: "feature", ref: "x3" }), ["feature:x3", "input"]);
  assert.ok(g.focusSet({ kind: "all" }).has("*"));
  assert.deepEqual(set({ kind: "block", ref: "B9" }), []);
  assert.equal(g.focusSet(null), null);
  assert.equal(g.focusSet({ kind: "family", ref: "nothing" }), null);
  assert.deepEqual([...graph("mlp-residual").focusSet({ kind: "block", ref: "B2" })].sort(),
    ["B2", "B2.r0.0", "B2.r0.0#act", "B2.r0.0#drop", "B2.r0.0#norm", "B2.r0.1", "B2.r0.2", "B2.r0.3", "B2.r0.add"]);
  const legacy = StageGraph.from(viewOfLegacy(R7.network)), L = (f) => [...legacy.focusSet(f)].sort();
  assert.deepEqual(L("hidden"), ["stages.0.linear", "stages.3.linear"]);
  assert.deepEqual(L("activation"), ["stages.0.linear#act", "stages.3.linear#act"]);
  assert.deepEqual(L("outputs"), ["head", "stages.6.linear"]);
  assert.deepEqual(L("inputs"), ["input"]);
  assert.deepEqual(L("feature:i0"), ["feature:i0", "input"]);
  assert.equal(legacy.focusSet("no-such-word"), null);
});

test("Focus lights columns and connections as today's drawing did", () => {
  const g = StageGraph.from(viewOfLegacy(R7.network)), [h1, h2] = ["stages.0.linear", "stages.3.linear"];
  const hidden = new Focus(g.focusSet("hidden"));
  assert.deepEqual(["input", h1, h2, "head"].map((c) => hidden.node(c)), [false, true, true, false]);
  assert.deepEqual([hidden.edge("input", h1), hidden.edge(h1, h2), hidden.edge(h2, "head")], [false, true, false]);
  const outs = new Focus(g.focusSet("outputs"));
  assert.deepEqual([outs.edge("input", h1), outs.edge(h1, h2), outs.edge(h2, "head")], [false, false, true]);
  const act = new Focus(g.focusSet("activation"));
  assert.deepEqual(["input", h1, h2, "head"].map((c) => act.node(c)), [false, true, true, false]);
  assert.deepEqual([act.edge("input", h1), act.edge(h1, h2), act.edge(h2, "head")], [false, false, false]);
  const edges = new Focus(g.focusSet("edges"));
  assert.ok(edges.node(h1) && edges.edge("input", h1) && !edges.all);
  const all = new Focus(g.focusSet("all"));
  assert.ok(all.all && all.edge("input", h1) && all.node("head"));
  const none = new Focus(null);
  assert.ok(!none.active && none.node(h1) && none.edge(h1, h2));
  assert.equal(new Focus(g.focusSet("feature:i4")).feature, "i4");
});

// ---- weights
test("weights decode from json lists and from base64 little-endian float32 alike", () => {
  const p = FIX("mlp-plain"), g = StageGraph.from(p);
  for (const id of ["B1.weight", "B4.weight", "B5.weight", "B4.bias"]) {
    const t = p.tensors[id], v = g.vector(id), l2 = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    assert.equal(v.length, t.summary.n, id);
    assert.ok(Math.abs(l2 - t.summary.l2) <= 1e-6 * t.summary.l2, `${id}: l2 ${l2} vs ${t.summary.l2}`);
    assert.equal(Math.min(...v), Math.fround(t.summary.min), id);
  }
  assert.equal(p.tensors["B4.weight"].encoding, "base64-le");
  const W = g.matrix("B4.weight");
  assert.deepEqual([W.out, W.in, W.w.length], [30, 40, 1200]);
  assert.equal(g.matrix("nope"), null);
  assert.deepEqual(g.missingTensors(), []);
  assert.equal(g.weightsWithheld(), false);
  const q = JSON.parse(JSON.stringify(p)); q.tensors["B4.weight"] = { ...q.tensors["B4.weight"], encoding: "deferred", data: null };
  const d = StageGraph.from(q);
  assert.deepEqual(d.missingTensors(), ["B4.weight"]);
  assert.equal(d.matrix("B4.weight"), null);
  d.attachTensor(p.tensors["B4.weight"]);
  assert.deepEqual(Array.from(d.matrix("B4.weight").w), Array.from(W.w));
  const run = g.segments()[1];
  assert.deepEqual(g.runModel(run).layers.map((l) => l.id), ["B1", "B4", "B5"], "the pure chain drawn whole carries its head");
  assert.deepEqual(g.runModel(run, { head: false }).layers.map((l) => l.id), ["B1", "B4"], "a run drawn as a segment leaves the head to its own");
  assert.equal(complete(g), true);
  const w = JSON.parse(JSON.stringify(p));
  for (const t of Object.values(w.tensors)) Object.assign(t, { encoding: "withheld", data: null });
  const withheld = StageGraph.from(w);
  assert.equal(withheld.weightsWithheld(), true);
  assert.equal(complete(withheld), false, "without its weights a pure chain is drawn as segments");
});

test("the input units: a numeric input one column, a nominal input its embedding's dims", () => {
  const g = graph("embedding");
  assert.deepEqual(g.featureUnits(), [[0], [1], [2], [3], [4], [5, 6, 7], [8, 9]]);
  assert.deepEqual(g.runModel(g.segments()[1]).inputs, ["x0", "x1", "x2", "x3", "x4", "c0·e0", "c0·e1", "c0·e2", "c1·e0", "c1·e1"]);
  assert.deepEqual(graph("mlp-residual").headLabels(), ["yes"]);
  assert.deepEqual(graph("autoencoder").headLabels(), graph("autoencoder").inputs.names);
});

// ---- the v2.2 payload, through D3 §5.6's mapping (tests/legacy-view.js; the page reads @2 only)
test("R7's v2.2 dense chain, mapped to network-view@2: dense stages and a head, its weights the chain's tensors", () => {
  const g = StageGraph.from(viewOfLegacy(R7.network)), n = R7.network;
  assert.deepEqual(g.stages().map((s) => [s.id, s.kind, s.activation, s.dropout]),
    [["stages.0.linear", "dense", "gelu", 0], ["stages.0.linear.dropout", "dropout", null, 0.1], ["stages.3.linear", "dense", "gelu", 0],
     ["stages.3.linear.dropout", "dropout", null, 0.1], ["stages.6.linear", "head", null, 0]]);
  assert.deepEqual(g.denseRuns()[0].layers.map((l) => [l.dense, l.dropout]), [["stages.0.linear", 0.1], ["stages.3.linear", 0.1]]);
  assert.deepEqual([g.task.kind, g.task.link, g.task.outputs.names], ["multiclass", "softmax", n.outputs]);
  assert.equal(g.params.total, n.params);
  assert.deepEqual(g.inputs.names, n.inputs);
  const m = g.runModel(g.segments()[1]);
  assert.deepEqual(m.layers.map((l) => [l.in, l.out, l.W.out, l.W.in]), n.layers.map((l) => [l.in, l.out, l.out, l.in]));
  assert.deepEqual(m.outputs, n.outputs);
  assert.deepEqual(Array.from(m.layers[2].bias), n.weights[2].bias.map(Math.fround));
  assert.equal(StageGraph.from(viewOfLegacy(n, { task: "binary" })).task.link, "sigmoid");
});

// ---- families
test("families: story@2 members, story@1 patterns, one family without groups, coverage refused", () => {
  const names = ["a0", "a1", "b0", "b1", "c0"];
  const f2 = familiesOf({ groups: [{ id: "a", label: "A", text: "", members: ["a1", "a0"] }, { id: "b", label: "B", text: "", members: ["b0", "b1", "c0"] }], names });
  assert.deepEqual(f2.families.map((f) => [f.id, f.members, f.idx]), [["a", ["a0", "a1"], [0, 1]], ["b", ["b0", "b1", "c0"], [2, 3, 4]]]);
  assert.deepEqual(f2.problems, []);
  const hole = familiesOf({ groups: [{ id: "a", label: "A", text: "", members: ["a0", "zz"] }], names });
  assert.equal(hole.families.length, 1);
  assert.deepEqual(hole.families[0].members, names);
  assert.ok(hole.problems.some((p) => /zz is not an input/.test(p)) && hole.problems.some((p) => /a1 is in no input group/.test(p)));
  const f1 = familiesOf({ groups: [{ label: "B", match: ["^b"] }, { label: "none", match: ["^q"] }], names });
  assert.deepEqual(f1.families.map((f) => [f.id, f.members]), [["g0", ["b0", "b1"]], ["other", ["a0", "a1", "c0"]]]);
  assert.deepEqual(familiesOf({ names }).families.map((f) => f.idx), [[0, 1, 2, 3, 4]]);
  const emb = graph("embedding");
  const fe = familiesOf({ groups: [{ id: "n", label: "N", text: "", members: ["x0", "x1", "x2", "x3", "x4"] }, { id: "c", label: "C", text: "", members: ["c0", "c1"] }], names: emb.inputs.names, units: emb.featureUnits() });
  assert.deepEqual(fe.families[1].idx, [5, 6, 7, 8, 9]);
  assert.equal(familyOf(fe.families, "c1").id, "c");
  const r7 = familiesOf({ groups: R7.families.map((f, i) => ({ id: `g${i}`, label: f.label, text: "", members: f.idx.map((j) => R7.network.inputs[j]) })), names: R7.network.inputs });
  assert.deepEqual(r7.families.map((f) => f.idx), R7.families.map((f) => f.idx.slice().sort((a, b) => a - b)));
});

// ---- the reading guide
test("the guide's words are composed from the graph; a linear network has no middle step", () => {
  configureText({ variants: ["classes+row", "classes", "row"] });
  const g = StageGraph.from(viewOfLegacy(R7.network)), fam = R7.families.map((f) => ({ ...f, members: f.idx.map((i) => g.inputs.names[i]) }));
  const steps = guideSteps(g, { families: fam, k: 14 });
  assert.deepEqual(steps.map((s) => s.text), [
    "On the left, the 65 inputs of a row, gathered into 7 families. Each circle is one family.",
    "In the middle, 2 hidden layers of 256 units. Each unit mixes everything before it into one number. The 14 that matter most in each layer are drawn.",
    "Each line is a weight: how much one circle pushes the next. Teal raises it, violet lowers it; the thicker the line, the stronger the push.",
    "On the right, one score per class — 15 in all. The highest score is the network's answer."]);
  assert.deepEqual([...steps[1].focus].sort(), ["stages.0.linear", "stages.0.linear#act", "stages.0.linear.dropout", "stages.3.linear",
    "stages.3.linear#act", "stages.3.linear.dropout"]);
  configureText({ variants: ["value+row", "value", "row"] });
  const lin = guideSteps(graph("linear"), { families: [{ idx: [0] }], k: 14 });
  assert.equal(lin.length, 3);
  assert.equal(lin[0].text, "On the left, the 8 inputs of a row, as one family. The circle stands for all of them.");
  assert.equal(lin[2].text, "On the right, one number: the value the network answers.");
  configureText({ variants: ["classes+graph", "classes", "graph"] });
  const gcn = guideSteps(graph("gcn-node"), { families: [{ idx: [0] }, { idx: [1] }], k: 14 });
  assert.match(gcn[1].text, /^In the middle, 2 rounds of message passing between neighbouring nodes\.$/);
  configureText({ variants: ["reconstruction+row", "reconstruction", "row"] });
  const ae = guideSteps(graph("autoencoder"), { families: [{ idx: [0] }], k: 14 });
  assert.equal(ae[1].text.split(". ")[0], "In the middle, 5 hidden layers of 6, 4, 2 units and a waist of 2 units every row must pass through");
  configureText({ variants: ["value+row", "value", "row"] });
  const par = guideSteps(graph("parallel"), { families: [{ idx: [0] }], k: 14 });
  assert.equal(par[1].text, "In the middle, 1 hidden layer of 4 units, a block whose branches read the same numbers side by side "
    + "and 1 further step. Each unit mixes everything before it into one number. The 14 that matter most in each layer are drawn.");
  configureText({ variants: ["binary+row", "binary", "row"] });
  const res = guideSteps(graph("mlp-residual"), { families: [{ idx: [0] }, { idx: [1] }], k: 12 });
  assert.equal(res[1].text.split(". ")[0], "In the middle, 1 hidden layer of 6 units and a residual block that adds its input back to what it computes");
  assert.equal(res[3].text, "On the right, one score; when it passes the threshold the answer is yes.");
  configureText({ variants: ["classes+window", "classes", "window"] });
  const seq = guideSteps(graph("lstm-window"), { families: [{ idx: [0] }, { idx: [1] }, { idx: [2] }], k: 14 });
  assert.equal(seq[0].text, "On the left, the 5 inputs of each of the last 6 steps of a window, gathered into 3 families. Each circle is one family.");
  assert.equal(seq[1].text, "In the middle, a recurrent layer that reads the steps in order.");
  configureText({ variants: ["horizon+window", "horizon", "window"] });
  assert.equal(guideSteps(graph("forecast"), { families: [{ idx: [0] }], k: 14 }).pop().text, "On the right, one value per step ahead — 3 in all.",
    "a step's quantiles are one step");
  assert.equal(joinList(["a", "b", "c"]), "a, b and c");
  assert.equal(joinList(["a"]), "a");
  configureText({});
});

// ---- the canvas: spreading the segments, their captions, their layouts and connectors
test("spreadRow grows runs, groups and gaps up to three times and centres the rest; a row too wide shrinks", () => {
  const sizes = [{ w: 100 }, { w: 50, grow: true }, { w: 100 }];
  assert.deepEqual(spreadRow(sizes, 0, 1000, 30).map((b) => [b.x, b.w]), [[235, 100], [425, 150], [665, 100]]);
  assert.deepEqual(spreadRow(sizes, 0, 155, 30).map((b) => [b.x, b.w]), [[0, 50], [65, 25], [105, 50]]);
  assert.deepEqual(spreadRow([{ w: 40 }], 10, 100, 30).map((b) => [b.x, b.w]), [[40, 40]]);
});

test("drawCaptions puts each caption on the lowest row it clears and leaves out one that clears neither", () => {
  const calls = [], g = { measureText: (s) => ({ width: 6 * s.length }), fillText: (s, x, y) => calls.push([s, x, y]) };
  drawCaptions(g, [{ x: 300, text: "DDDD" }, { x: 50, text: "AAAAAAAAAA" }, { x: 80, text: "BBBBBBBBBB" }, { x: 100, text: "CCCCCCCCCC" }],
    { W: 400, H: 200, colors: { muted: "grey" }, captionFont: "10px mono" });
  assert.deepEqual(calls, [["AAAAAAAAAA", 50, 190], ["BBBBBBBBBB", 80, 178], ["DDDD", 300, 190]]);
  assert.equal(g.font, "10px mono");
});

test("the segments' layouts: ports through the weights, captions numbered in drawing order", async () => {
  configureText({ variants: ["binary+row", "binary", "row"] });
  const g = graph("mlp-residual"), renderers = await load(g.kinds()), segs = g.segments();
  const families = familiesOf({ groups: [{ id: "a", label: "A", text: "", members: ["x0", "x1", "x2"] }, { id: "b", label: "B", text: "", members: ["x3", "x4", "x5", "x6", "x7"] }],
    names: g.inputs.names, units: g.featureUnits() }).families;
  const ctx = { W: 1100, H: 480, narrow: false, mode: "simple", graph: g, env: { families }, K: 14, colors: {}, boxW: 88, top: 18, bottom: 36,
    ordinal: new Map([["B1", 1], ["B2.r0.0", 2]]), waist: null, rendererFor: (s) => rendererFor(renderers, g, s) };
  const boxes = spreadRow(segs.map((s) => ctx.rendererFor(s).measure(s, ctx)), 28, 1044, 30);
  const L = segs.map((s, i) => ctx.rendererFor(s).layout(s, { ...boxes[i], y: 18, h: 426 }, ctx));
  assert.deepEqual(L.flatMap((l) => l.captions.map((c) => c.text)), ["INPUT FAMILIES", "HIDDEN 1 · 6 OF 6", "HIDDEN 2 · 6 OF 6", "SCORE"]);
  const xs = L.flatMap((l) => l.captions.map((c) => c.x));
  assert.deepEqual(xs, xs.slice().sort((a, b) => a - b), "captions left to right");
  assert.deepEqual(L[0].out.map((p) => p.units), [[0, 1, 2], [3, 4, 5, 6, 7]]);
  assert.equal(L[1].inMatrix.w, g.vector("B1.weight"));
  const lines = connectorLines(L[0], L[1]);
  assert.ok(lines.every((e) => e.w !== null) && new Set(lines.map((e) => e.b)).size === 6, "every unit of the first run is fed by a family's weights");
  assert.equal(L[2].inMatrix.w, g.vector("B2.r0.0.weight"), "a residual is entered through its first stage's weights");
  assert.equal(L[3].inMatrix.w, g.vector("B3.weight"));
  const fan = connectorLines(L[2], L[3]);
  assert.ok(fan.length >= 1 && fan.every((e) => e.w === null), "⊕ → head: one line, no weight between them");
  configureText({});
});

test("every stage kind has a word for its card", () => {
  for (const k of STAGE_KINDS) if (k !== "input") assert.ok(hasKey(`box.kind.${k}`), `box.kind.${k}`);
});

// ---- the output column per link (renderers/head.js)
const headCtx = (g) => ({ W: 1100, H: 480, narrow: false, graph: g, env: {}, colors: {} });
const headLayout = (g) => { const seg = g.segments().find((s) => s.kind === "head"); return head.layout(seg, { x: 800, y: 18, w: head.measure(seg, headCtx(g)).w, h: 400 }, headCtx(g)); };

test("the output column's units per link: classes, a score, thresholds between classes, rebuilt inputs, steps ahead", () => {
  configureText({});
  const ord = graph("ordinal");
  assert.deepEqual(ord.headLabels(), [">k0", ">k1", ">k2"]);
  assert.deepEqual(ord.ordinalClasses(), ["k0", "k1", "k2", null]);
  const L = headLayout(ord);
  assert.deepEqual(L.classes.map((c) => c.text), ["k0", "k1", "k2", "above k2"]);
  assert.equal(L.pts.length, 3, "one point per threshold, between two classes");
  assert.ok(L.pts.every((p, j) => p.y > L.classes[j].y && p.y < L.classes[j + 1].y));
  assert.deepEqual([L.inMatrix.out, L.inMatrix.in], [3, 6], "the shared score row, once per threshold");
  const row = Array.from(ord.matrix("B2.weight").w);
  for (let j = 0; j < 3; j++) assert.deepEqual(Array.from(L.inMatrix.w.subarray(6 * j, 6 * j + 6)), row);
  assert.equal(ord.isPureDenseChain(), true);
  assert.equal(complete(ord), true);
  const m = ord.runModel(ord.segments()[1]);
  assert.deepEqual(Array.from(m.layers[1].bias), FIX("ordinal").nodes.find((n) => n.id === "B2").params.thresholds.map(Math.fround));

  const rec = graph("recon-nominal");
  assert.deepEqual(rec.headLabels(), ["x0", "x1", "x2", "x3", "c0[0]", "c0[1]", "c0[2]"]);
  assert.deepEqual(rec.rebuiltInputOf(), [0, 1, 2, 3, 4, 4, 4]);
  const fc = graph("forecast"), F = headLayout(fc);
  assert.equal(F.pts.length, 9);
  assert.deepEqual(F.pts.filter((p) => p.first).map((p) => p.step), ["t+1", "t+2", "t+3"]);
  assert.deepEqual([...new Set(F.pts.map((p) => p.x))].length, 3, "a step's quantiles share its column");
  assert.equal(head.measure(fc.segments().find((s) => s.kind === "head"), headCtx(fc)).w, 90);
  assert.deepEqual(graph("mlp-residual").headLabels(), ["yes"]);
  const sv = graph("svdd"), S = headLayout(sv);
  assert.deepEqual(Array.from(S.centre), Array.from(sv.vector("B4.centre")));
  assert.equal(sv.isPureDenseChain(), false, "a normalisation: drawn as segments, the sphere beside the head");
});

test("what the output column rings and colours once answered", () => {
  const o = (read, answer = {}, raw = read) => outputsOf({ outputs: { raw, read, answer } });
  assert.deepEqual([...winners("softmax", o([0.1, 0.7, 0.2]), null)], [1]);
  assert.deepEqual([...winners("sigmoid", o([0.41]), 0.42)], []);
  assert.deepEqual([...winners("sigmoid", o([0.43]), 0.42)], [0]);
  assert.deepEqual([...winners("sigmoid_each", o([0.6, 0.2, 0.9]), null)], [0, 2]);
  assert.deepEqual([...winners("identity", o([3.2]), null)], []);
  assert.deepEqual([...winners("softmax", null, null)], []);
  assert.equal(ordinalAnswer(o([0.8, 0.6, 0.3], {}, [1.2, 0.3, -0.8])), 2);
  assert.equal(ordinalAnswer(o([0.8, 0.6, 0.3], { rank: 1 }, [1.2, 0.3, -0.8])), 1);
  const rec = graph("recon-nominal");
  assert.deepEqual(unitErrors(rec, o([0, 0, 0, 0, 0, 0, 0], { per_feature: [0.1, 0.2, 0.4, 0.1, 0.8] })), [0.125, 0.25, 0.5, 0.125, 1, 1, 1]);
  assert.deepEqual(unitErrors(rec, o([0], { per_feature: { x2: 2, c0: 1 } })), [0, 0, 1, 0, 0.5, 0.5, 0.5]);
  assert.equal(unitErrors(rec, o([0])), null);
  assert.equal(distanceOf(o([1, 2], { distance: 0.5 }), null), 0.5);
  assert.equal(distanceOf(o([1, 2]), Float32Array.from([0, 1])), 2);
  assert.equal(distanceOf(o([1, 2]), null), null);
  const acts = head.acts(null, { outputs: { raw: [0, 0, 0, 0, 0, 0, 0], read: [0, 0, 0, 0, 0, 0, 0], answer: { per_feature: [1, 1, 1, 1, 2], error: 0.3 } } }, rec);
  assert.deepEqual(acts.err, [0.5, 0.5, 0.5, 0.5, 1, 1, 1]);
});

// ---- D4 §7.3's overflow rule, the groups' shares, the connectors of one answer, the head's detail
const canvasCtx = async (name, W, { mode = "simple", K = 14 } = {}) => {
  const g = graph(name), renderers = await load(g.kinds()), narrow = W < 700;
  const families = familiesOf({ names: g.inputs.names, units: g.featureUnits() }).families;
  return { g, ctx: { W, H: 400, narrow, mode, graph: g, env: { families }, K: mode === "detail" ? Infinity : K, colors: {}, boxW: narrow ? 64 : 88,
    collapse: false, compact: false, top: 18, bottom: 36, ordinal: new Map(), waist: null, rendererFor: (s) => rendererFor(renderers, g, s) } };
};
const fitAt = async (W, opts) => {
  const { g, ctx } = await canvasCtx("residual-deep", W, opts), pad = W < 700 ? 16 : 28, gap = W < 700 ? 18 : 30;
  return { g, ctx, f: fitRow(g.segments(), ctx, W - 2 * pad, gap), room: W - 2 * pad };
};

test("the overflow rule: repeated groups collapse to × n, then boxes go to 48 px, then runs to K = 8", async () => {
  assert.equal(FIT_STAGES.length, 4);
  const wide = await fitAt(1300);
  assert.equal(wide.f.stage, 0, "every repeat drawn when they fit");
  assert.deepEqual(wide.f.sizes.map((s) => s.w), [174, 64, 3 * (64 + 18 + 18) + 2 * 18 + GROUP_PAD, 88, 88, 170], "group = its children (the inside, each ⊕) + GROUP_PAD");
  const one = await fitAt(1100);
  assert.deepEqual([one.f.stage, one.f.fit.collapse, one.f.fit.boxW], [1, true, 88]);
  assert.equal(one.f.sizes[2].w, 64 + 18 + 18 + GROUP_PAD, "one repeat, titled × 3");
  const boxes = await fitAt(900);
  assert.deepEqual([boxes.f.stage, boxes.f.fit.boxW, boxes.f.sizes[3].w, boxes.f.sizes[4].w, boxes.f.fit.K], [2, 48, 48, 48, 14]);
  const runs = await fitAt(800);
  assert.deepEqual([runs.f.stage, runs.f.fit.K, runs.f.fit.compact, runs.f.sizes[1].w], [3, 8, true, 46]);
  assert.ok(runs.f.want <= runs.room);
  const tight = await fitAt(700);
  assert.ok(tight.f.stage === 3 && tight.f.want > tight.room, "past the last stage the row shrinks as a whole (spreadRow)");
  const detail = await fitAt(800, { mode: "detail" });
  assert.equal(detail.f.fit.K, Infinity, "every weight keeps every unit");
  const { g, ctx } = one, seg = g.segments()[2];
  assert.deepEqual(shownRepeats(seg, { ...ctx, ...one.f.fit }), [0]);
  assert.deepEqual(shownRepeats(seg, ctx), [0, 1, 2]);
  assert.deepEqual(rowsOf(seg, ctx)[0].map((it) => (it.merge !== undefined ? `+${it.merge}` : it.at.join("."))), ["0.0", "+0", "1.0", "+1", "2.0", "+2"]);
});

test("a group's layout: each repeat's ⊕ in the row, the ports through the first run's weights, the branches' merge", async () => {
  configureText({});
  const { g, ctx } = await canvasCtx("residual-deep", 1300), seg = g.segments()[2];
  const L = group.layout(seg, { x: 300, y: 18, w: group.measure(seg, ctx).w, h: 346 }, ctx);
  const merges = L.bands[0].filter((c) => c.merge !== undefined);
  assert.deepEqual(merges.map((m) => m.merge), [0, 1, 2]);
  assert.ok(merges.every((m, k) => k === 0 || m.x > merges[k - 1].x));
  assert.deepEqual(L.out, [{ x: merges[2].x + 9, y: merges[2].y }], "the group leaves from its last ⊕");
  assert.equal(L.inMatrix.w, g.vector("B2.r0.0.weight"));
  assert.deepEqual(L.captions.map((c) => c.text).length, 3);
  const hit = group.hit(seg, L, L.bands[0][2].L.xs[0], L.bands[0][2].L.ys[0][0]);
  assert.deepEqual([hit.at, hit.item], [[1, 0], 2]);
  assert.equal(groupChild(seg, hit.at).stages[0], "B2.r1.0");
  const p = await canvasCtx("parallel", 1100), pseg = p.g.segments()[2];
  const P = group.layout(pseg, { x: 300, y: 18, w: group.measure(pseg, p.ctx).w, h: 346 }, p.ctx);
  assert.equal(P.bands.length, 2);
  assert.deepEqual(P.in, [P.inPort], "the branches all read the group's input");
  assert.equal(P.out[0].x, P.merge.x + 9);
});

test("one answer: the skip's share of each ⊕, each branch's share of the merge", () => {
  const vec = (values) => ({ form: "vector", n: values.length, values });
  const node = (fields) => ({ level: "full", why: null, in: null, out: null, grad_out: null, relevance: null, extra: {}, recompute: null, exact: null, ...fields });
  const deep = graph("residual-deep").segments()[2];
  const ex = { schema: "vega/explanation@2", nodes: {
    "B2.r0.add": node({ extra: { skip_norm: 3, branch_norm: 1 } }),
    "B2.r1.1": node({ out: vec([0, 3, 0, 4, 0, 0]) }),
    "B2.r1.add": node({ out: vec([5, 0, 0, 0, 0, 0]) }),
    "B2.r2.1": node({ out: { form: "vector", n: 6, top: [[0, 1]], summary: { l2: 5 } } }),
  } };
  ex.nodes["B2.r0.add"].out = vec([0, 0, 0, 0, 0, 5]);
  assert.equal(skipShare(ex, deep, 0), 0.75, "the add node's own reading");
  assert.equal(skipShare(ex, deep, 1), 0.5, "‖x‖ = the ⊕ before (5), ‖f(x)‖ = the repeat's last stage (5)");
  assert.equal(skipShare(ex, deep, 2), 0.5, "the last stage's reduced vector read through its summary");
  const { "B2.r1.add": _gone, ...rest } = ex.nodes;
  assert.equal(skipShare({ ...ex, nodes: rest }, deep, 2), null, "nothing said of the ⊕ before");
  assert.equal(skipShare(null, deep, 0), null);
  assert.equal(stageNorm(ex, "B2.r2.1", "out", tensorValues), 5, "a reduced vector's summary");
  assert.equal(stageNorm({ ...ex, nodes: { a: node({ out: { form: "sequence", per_step: { norm: [3, 4], mean: [0, 0] } } }) } }, "a", "out", tensorValues), 5);
  assert.equal(stageNorm({ ...ex, nodes: { a: node({ level: "omitted", why: "budget" }) } }, "a", "out", tensorValues), null);
  const par = graph("parallel").segments()[2];
  assert.deepEqual(branchShares({ schema: "vega/explanation@2", nodes: { "B2.merge": node({ extra: { branch_norms: [1, 3] } }) } }, par), [0.25, 0.75]);
  assert.deepEqual(branchShares({ schema: "vega/explanation@2", nodes: { "B2.b0": node({ out: vec([3, 0, 4]) }), "B2.b1.1": node({ out: vec([0, 0, 15]) }) } }, par), [0.25, 0.75],
    "else each branch's last stage");
  assert.equal(branchShares({ schema: "vega/explanation@2", nodes: {} }, par), null);
  const acts = group.acts(deep, ex, graph("residual-deep"), { rendererFor: () => ({ acts: () => null }) });
  assert.deepEqual(acts.share, [0.75, 0.5, 0.5]);
  assert.equal(acts.parts.length, 3, "every repeat's children, drawn or not");
});

test("connectors: a fan takes the sign of the share it carries once answered; the head's detail joins 6 inputs", async () => {
  configureText({});
  const A = { out: [{ x: 0, y: 0, units: [0], v: -0.5 }, { x: 0, y: 10, units: [1], v: 0.9 }] }, B = { in: [{ x: 50, y: 5 }] };
  assert.deepEqual(connectorLines(A, B).map((e) => [e.b, e.w, e.n]), [[0, -0.5, 0.5], [0, 0.9, 0.9]]);
  assert.deepEqual(connectorLines({ out: [{ x: 0, y: 0 }] }, { in: [{ x: 9, y: 0, unit: 0, v: 0.3 }] }).map((e) => e.w), [0.3], "box → run: the unit it reaches");
  assert.deepEqual(connectorLines({ out: [{ x: 0, y: 0 }] }, B).map((e) => e.w), [null], "nothing answered: --muted2");
  const { g, ctx } = await canvasCtx("mlp-residual", 1100, { mode: "detail" }), segs = g.segments();
  assert.equal(head.detail, true);
  const H = head.layout(segs[3], { x: 900, y: 18, w: 170, h: 346 }, ctx);
  assert.equal(H.inTop, 6);
  assert.equal(head.layout(segs[3], { x: 900, y: 18, w: 170, h: 346 }, { ...ctx, mode: "simple" }).inTop, 3);
  const run = { out: Array.from({ length: 6 }, (_, j) => ({ x: 800, y: 20 + 40 * j, units: [j] })) };
  assert.equal(connectorLines(run, H).length, 6, "the one score joined to all 6 units of the run before");
  const plain = graph("mlp-plain"), res = graph("mlp-residual");
  assert.equal(supportOf(res, res.segments()[3], 0), null, "fed by a residual group, not a dense run");
  const lin = graph("linear");
  assert.equal(supportOf(lin, lin.segments()[1], 0), null, "fed by the inputs");
  const sup = supportOf(plain, plain.segments()[2], 1), row = Array.from(plain.matrix("B5.weight").w.subarray(30, 60));
  const best = row.map((v, i) => [i, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];
  assert.ok(sup.startsWith(`unit ${best[0] + 1} (${best[1] >= 0 ? "raises" : "lowers"})`), sup);
});

test("deferred tensors: the drawn ones are fetched and attached; a failed fetch is reported, never thrown", async () => {
  const p = FIX("mlp-plain"), q = JSON.parse(JSON.stringify(p));
  for (const id of ["B4.weight", "B1.bias"]) q.tensors[id] = { ...q.tensors[id], encoding: "deferred", data: null };
  const g = StageGraph.from(q);
  assert.deepEqual(g.drawnTensors().sort(), ["B1.bias", "B1.weight", "B4.bias", "B4.weight", "B5.bias", "B5.weight"]);
  assert.deepEqual(g.drawnTensorsMissing().sort(), ["B1.bias", "B4.weight"]);
  assert.equal(complete(g), false);
  const asked = [];
  const r = await g.fetchDeferred(async (id) => { asked.push(id); return p.tensors[id]; });
  assert.deepEqual([asked.sort(), r.fetched.sort(), r.failed], [["B1.bias", "B4.weight"], ["B1.bias", "B4.weight"], []]);
  assert.deepEqual(g.drawnTensorsMissing(), []);
  assert.equal(complete(g), true, "the pure chain is drawn whole again");
  const h = StageGraph.from(q), unhandled = [], seen = (e) => unhandled.push(e);
  process.on("unhandledRejection", seen);
  try {
    const out = await h.fetchDeferred((id) => (id === "B4.weight" ? Promise.reject(new Error("404 not found")) : { ...p.tensors[id], shape: [9] }));
    assert.deepEqual(out.fetched, []);
    assert.deepEqual(out.failed.map((f) => f.id).sort(), ["B1.bias", "B4.weight"]);
    assert.match(out.failed.find((f) => f.id === "B4.weight").error, /404/);
    assert.match(out.failed.find((f) => f.id === "B1.bias").error, /declared shape/, "a payload of another shape is refused");
    await new Promise((res) => setTimeout(res, 20));
    assert.deepEqual(unhandled, []);
  } finally { process.off("unhandledRejection", seen); }
  assert.deepEqual(h.drawnTensorsMissing().sort(), ["B1.bias", "B4.weight"]);
  const legacy = StageGraph.from(viewOfLegacy(R7.network));
  assert.deepEqual(legacy.drawnTensorsMissing(), [], "inline tensors carry their values");
  assert.deepEqual((await legacy.fetchDeferred(() => { throw new Error("never asked"); })).failed, []);
  assert.equal(graph("svdd").drawnTensors().includes("B4.centre"), true);
});

test("the group, head and run renderers draw every state without throwing", async () => {
  const saved = { gcs: globalThis.getComputedStyle, doc: globalThis.document };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
  globalThis.document = { documentElement: {} };
  const widths = [];
  const g2d = new Proxy({ measureText: (s) => ({ width: 6 * String(s).length }) }, {
    get: (o, k) => (k in o ? o[k] : () => {}),
    set: (o, k, v) => { if (k === "lineWidth") widths.push(v); o[k] = v; return true; },
  });
  try {
    configureText({});
    for (const [name, W] of [["residual-deep", 1300], ["residual-deep", 1100], ["parallel", 1100], ["mlp-residual", 900]]) {
      for (const mode of ["simple", "detail"]) {
        const { g, ctx } = await canvasCtx(name, W, { mode }), pad = 28, gap = 30, f = fitRow(g.segments(), ctx, W - 2 * pad, gap);
        const c = { ...ctx, ...f.fit, colors: { acc: "a", neg: "n", mut: "m", fg: "f", fg2: "f2", muted: "mu", alert: "al", bg: "b", line: "l" } };
        const segs = g.segments(), boxes = spreadRow(f.sizes, pad, W - 2 * pad, gap);
        const ex = { schema: "vega/explanation@2", outputs: { raw: [0.4], read: [0.4], answer: {} }, nodes: {
          "B2.r0.add": { level: "full", extra: { skip_norm: 1, branch_norm: 3 } }, "B2.merge": { level: "full", extra: { branch_norms: [2, 2] } } } };
        for (const acts of [null, segs.map((s) => c.rendererFor(s).acts(s, ex, g, c))]) {
          const L = segs.map((s, i) => c.rendererFor(s).layout(s, { ...boxes[i], y: 18, h: 346 }, { ...c, acts: acts ? acts[i] : null }));
          segs.forEach((s, i) => c.rendererFor(s).draw(g2d, s, L[i], { phase: 1, focus: new Focus(null), hover: null, acts, mode, g: g2d, n: segs.length, i, ctx: c }));
        }
      }
    }
    assert.ok(widths.includes(0.6 + 4.4 * 0.25), "an arc as wide as the skip's share");
    assert.ok(widths.includes(0.6 + 4.4 * 0.5), "a branch's line as wide as its share");
  } finally { globalThis.getComputedStyle = saved.gcs; globalThis.document = saved.doc; }
});
