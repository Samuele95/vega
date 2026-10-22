// node --test: how/stepper.js and how/explainers/** — "how the network computed it" (D4 §8.1) on the
// network-view@2 fixtures (fixtures/network-view, the stage graphs of graph.test.js) and on the R7
// bundle's v2.2 answers (fixtures/network-view/r7-legacy.json): the steps each graph gives (runs,
// boxes, groups, the notes of what does nothing while answering, the link), their names and walk
// ids, the collapse past twelve steps, the explainer registry (every step kind resolves, the
// catalog's recompute table followed, a module that fails falls back to the generic explainer and
// is named), and on R7 the stepper walked in a small DOM (dom-shim.js): each step's z is today's
// page's (the unit that carried most of the answer, its weighted sum ending on the network's own
// z), no key missing, no placeholder left, nothing failed. Last, the pooling explainer (pool mean,
// max, cls; pool1d max, avg; a reshape) and a standalone bend on hand-computed values, and what each
// says when it cannot recompute.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { install } from "./dom-shim.js";

install();
const { StageGraph } = await import("../js/net/graph.js");
const { viewOfLegacy } = await import("./legacy-view.js");
const { STEP_KINDS, QUIET_KINDS, KIND_OF } = await import("../js/net/kinds.js");
const { Stepper, deriveSteps, stepNames, collapse, flatSteps, codeId, MAX_STEPS } = await import("../js/how/stepper.js");
const { load, EXPLAINER_OF, MEMBERS, MODULES, CORE_MODULES, GENERIC, gapsOf } = await import("../js/how/explainers/index.js");
const { configureText, missingKeys, unresolved } = await import("../js/core/text.js");
const { Lexicon } = await import("../js/core/lexicon.js");
const { sg } = await import("../js/core/fmt.js");

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const FIX = (name) => JSON.parse(readFileSync(here(`./fixtures/network-view/${name}.json`), "utf8"));
const EXPLAIN = JSON.parse(readFileSync(here("../../../../vega-forge/scripts/backends/pytorch/explain-catalog.json"), "utf8"));
const R7 = FIX("r7-legacy");
const kinds = (steps) => steps.map((s) => s.kind);

// ---- steps
const STEPS = {
  "mlp-plain": ["input", "dense", "dense", "link"],
  linear: ["input", "link"],
  "mlp-residual": ["input", "dense", "dense", "residual", "link"],
  embedding: ["input", "embedding", "dense", "link"],
  autoencoder: ["input", "dense", "dense", "dense", "dense", "dense", "link"],
  parallel: ["input", "dense", "dense", "dense", "merge", "custom", "link"],
  "lstm-window": ["input", "recurrent", "link"],
  "gcn-node": ["input", "graph_conv", "activation", "graph_conv", "link"],
  ordinal: ["input", "dense", "link"],
  svdd: ["input", "dense", "link"],
  forecast: ["input", "dense", "link"],
  "recon-nominal": ["input", "embedding", "dense", "link"],
  "residual-deep": ["input", "dense", "dense", "residual", "dense", "residual", "dense", "residual", "norm", "link"],
};

test("the steps of every fixture: one per dense stage, per computing box, per group's sum or merge, the input and the link", () => {
  for (const [name, want] of Object.entries(STEPS)) assert.deepEqual(kinds(deriveSteps(StageGraph.from(FIX(name)))), want, name);
});

test("what does nothing while answering is a note on the next step, never a step", () => {
  const notes = (name) => deriveSteps(StageGraph.from(FIX(name))).map((s) => s.notes.map((n) => n.kind));
  assert.deepEqual(notes("gcn-node"), [[], ["edge_dropout"], [], ["dropout"], []]);
  assert.deepEqual(notes("lstm-window"), [[], [], ["dropout"]]);
  assert.deepEqual(notes("forecast"), [[], ["reshape"], []]);
  assert.deepEqual(notes("embedding"), [[], [], ["concat"], []]);
  assert.deepEqual(notes("residual-deep").at(-1), ["dropout"]);
  for (const name of Object.keys(STEPS)) {
    for (const s of deriveSteps(StageGraph.from(FIX(name)))) {
      assert.ok(!QUIET_KINDS.includes(s.kind), `${name}: ${s.kind} is a note`);
      for (const n of s.notes) assert.ok(QUIET_KINDS.includes(n.kind), `${name}: note ${n.kind}`);
    }
  }
});

test("each step names the step whose output it reads; a group's step names its inside", () => {
  const steps = deriveSteps(StageGraph.from(FIX("residual-deep")));
  steps.slice(1).forEach((s, i) => assert.ok(s.from, `step ${i + 1} reads a step`));
  const [r1, r2] = steps.filter((s) => s.kind === "residual");
  assert.equal(r1.projection, "B2.r0.proj");
  assert.equal(r1.add, "B2.r0.add");
  assert.deepEqual(r1.inner, ["B2.r0.0", "B2.r0.1"]);
  assert.equal(r1.from, steps[1], "the first repeat's sum adds to what the block read");
  assert.equal(r2.from, r1, "the next repeat starts from the sum before it");
  const par = deriveSteps(StageGraph.from(FIX("parallel"))), merge = par.find((s) => s.kind === "merge");
  assert.deepEqual(merge.branches.map((b) => b.ids), [["B2.b0"], ["B2.b1.0", "B2.b1.1"]]);
  assert.equal(merge.merge, "concat");
  assert.deepEqual(par.filter((s) => s.kind === "dense").map((s) => s.branch), [null, 1, 2]);
  assert.equal(par[3].from, par[1], "a branch's first step reads what the group read");
});

test("names: a layer is numbered, a word shared by several steps numbered, a branch named, the catalog's words first", () => {
  configureText({});
  const g = StageGraph.from(FIX("residual-deep")), names = stepNames(g, deriveSteps(g));
  assert.deepEqual([...names.values()], ["input", "layer 1", "layer 2", "residual 1", "layer 3", "residual 2", "layer 4", "residual 3",
    "normalisation", "output"]);
  const p = StageGraph.from(FIX("parallel"));
  assert.deepEqual([...stepNames(p, deriveSteps(p)).values()], ["input", "layer 1", "branch 1 · layer 2", "branch 2 · layer 3", "branches",
    "custom block", "output"]);
  configureText({ explain: { blocks: { dense: { step: "dense layer" }, head: { step: "the answer" } } } });
  const m = StageGraph.from(FIX("mlp-plain"));
  assert.deepEqual([...stepNames(m, deriveSteps(m)).values()], ["input", "dense layer 1", "dense layer 2", "the answer"]);
  configureText({});
});

test("past twelve steps, consecutive steps of a kind collapse into one with a picker; input and link stay", () => {
  const step = (kind, branch = null) => ({ kind, branch, notes: [] });
  const long = [step("input"), ...Array.from({ length: 11 }, () => step("dense")), step("norm"), step("dense"), step("link")];
  const c = collapse(long);
  assert.deepEqual(kinds(c), ["input", "dense", "norm", "dense", "link"]);
  assert.equal(c[1].members.length, 11);
  assert.equal(flatSteps(c).length, long.length);
  assert.deepEqual(kinds(collapse(long.slice(0, MAX_STEPS))), kinds(long.slice(0, MAX_STEPS)), "twelve steps stay as they are");
  assert.deepEqual(kinds(collapse([step("input"), step("dense", 1), step("dense", 2), step("link")], 2)), ["input", "dense", "dense", "link"],
    "two branches' steps do not merge");
});

test("walk ids of the steps' code (R-E8): shape, the block types, the group's type, link", () => {
  const g = StageGraph.from(FIX("mlp-residual")), steps = deriveSteps(g);
  assert.deepEqual(steps.map((s) => codeId(g, s)), ["shape", "dense", "dense", "residual", "link"]);
  const e = StageGraph.from(FIX("embedding"));
  assert.deepEqual(deriveSteps(e).map((s) => codeId(e, s)), ["shape", "embedding", "dense", "link"]);
});

// ---- the registry
test("every step kind has an explainer module, as the explain catalog's recompute table assigns them", () => {
  const stepKinds = [...new Set([...STEP_KINDS, ...QUIET_KINDS, "input", "link", "residual", "merge", "decoder", "head"])];
  for (const k of stepKinds) assert.ok(EXPLAINER_OF[k], `${k} has an explainer`);
  for (const [type, entry] of Object.entries(EXPLAIN.blocks)) {
    const recompute = EXPLAIN.recompute[entry.recompute], kind = KIND_OF[type];
    if (!recompute || recompute.explainer === "generic" || !EXPLAINER_OF[kind]) continue;
    const want = type === "head" ? "link.js" : `${recompute.explainer}.js`;       // the head's sum is the link step's
    assert.equal(EXPLAINER_OF[kind], want, `${type} (${kind}) → ${want}`);
  }
  assert.equal(EXPLAINER_OF.merge, `${EXPLAIN.recompute[EXPLAIN.blocks.parallel.recompute].explainer}.js`);
  for (const m of MODULES) assert.ok(CORE_MODULES.includes(m) || ["conv.js", "recurrent.js", "attention.js", "graph.js", "latent.js"].includes(m), m);
});

test("the core explainers load, each with the whole interface and the kinds the registry sends it", async () => {
  for (const file of CORE_MODULES) assert.ok(existsSync(here(`../js/how/explainers/${file}`)), file);
  const own = Object.keys(EXPLAINER_OF).filter((k) => CORE_MODULES.includes(EXPLAINER_OF[k]));
  const map = await load(own);
  assert.deepEqual(gapsOf(map), []);
  for (const k of own) {
    const x = map.get(k);
    for (const m of MEMBERS) assert.ok(m in x, `${k}: ${m}`);
    assert.ok(x.kinds.includes(k), `${EXPLAINER_OF[k]} lists ${k}`);
  }
});

test("a module that does not load, lacks a member or misses its kind falls back to the generic explainer, named", async () => {
  configureText({});
  const real = (f) => import(`../js/how/explainers/${f}`).then((m) => m.default);
  const importer = (f) => (f === "conv.js" ? Promise.reject(new Error("404")) : f === "norm.js" ? Promise.resolve({ kinds: ["norm"] })
    : f === "pool.js" ? real(f).then((m) => ({ ...m, kinds: [] })) : real(f));
  const map = await load(["conv", "norm", "pool", "dense", "no_such_kind"], "softmax", { importer });
  const generic = await real(GENERIC);
  for (const k of ["conv", "norm", "pool", "no_such_kind"]) assert.equal(map.get(k), generic, k);
  assert.notEqual(map.get("dense"), generic);
  assert.deepEqual(gapsOf(map).map((g) => [g.kind, g.module]).sort(), [["conv", "conv.js"], ["no_such_kind", GENERIC], ["norm", "norm.js"], ["pool", "pool.js"]]);
  assert.match(gapsOf(map).find((g) => g.kind === "conv").why, /did not load \(404\)/);
  assert.equal(map.link, "softmax");
  assert.deepEqual(Stepper.kinds(StageGraph.from(FIX("gcn-node"))), ["input", "graph_conv", "activation", "link"]);
});

// ---- R7: today's stepper, through the new one
const legacyChosen = (want, h) => want.hidden[h].map((_, j) => j).sort((a, b) => Math.abs(want.hidden[h][b]) - Math.abs(want.hidden[h][a]))[0];

R7.explanations.forEach((ex, n) => {
  test(`R7 answer ${n}: each step's unit and z are today's page's, recomputed within the gap rule`, async () => {
    const graph = StageGraph.from(viewOfLegacy(R7.network)), want = R7.expected[n], L = ex.pre.length;
    const st = new Stepper({ graph, ex, explainers: await load(Stepper.kinds(graph), graph.link) });
    const values = st.values();
    assert.deepEqual(values.map((v) => v.kind), ["input", "dense", "dense", "link"]);
    assert.equal(values[0].check, null);
    [1, 2].forEach((h) => {
      const v = values[h], j = legacyChosen(want, h - 1);
      assert.equal(v.unit, j, `layer ${h}: the unit that carried most`);
      assert.equal(v.check.network, ex.pre[h - 1][j]);
      assert.ok(v.check.within && v.check.ops.every((o) => o.within), `layer ${h}`);
      assert.deepEqual(v.check.ops.map((o) => o.what), ["act"], "the layer's own bend, recomputed");
    });
    assert.equal(values[3].unit, want.win);
    assert.equal(values[3].check.network, ex.pre[L - 1][want.win]);
    assert.ok(values[3].check.within);
    assert.equal(values[3].check.ops[0].what, "read", "the softmax recomputed beside the service's probability");
  });

  test(`R7 answer ${n}: walked in the DOM, each tab shows its z, the bend and the softmax table; no key missing`, async () => {
    const doc = install();
    configureText({ lex: new Lexicon(null, null), lane: { viewKey: "classes", shape: "row" } });
    const graph = StageGraph.from(viewOfLegacy(R7.network)), host = doc.createElement("div");
    doc.body.append(host);
    const st = await new Stepper({ graph, ex, onCode: () => {} }).mount(host);
    assert.equal(host.hidden, false);
    const tabs = host.querySelectorAll("button").filter((b) => b.getAttribute("role") === "tab").map((b) => b.textContent);
    assert.deepEqual(tabs, ["1 · input", "2 · layer 1", "3 · layer 2", "4 · output"]);
    const zs = [];
    for (let i = 0; i < 4; i++) {
      st.show(i);
      const body = doc.getElementById("how-body"), total = body.querySelector(".total");
      zs.push(total ? total.querySelector(".val").textContent.replace(/^=\s*/, "") : null);
      if (i === 1 || i === 2) {
        assert.ok(body.querySelector(".act"), "the bend drawn");
        assert.match(body.textContent, new RegExp(`Each of the 256 units of layer ${i} does two things`), "the step names its layer");
        if (i === 2) assert.match(body.textContent, /the \d+ outputs of layer 1/, "what it reads, by the step before's name");
      }
      if (i === 3) assert.ok(body.querySelector(".soft"), "the softmax table");
    }
    const want = R7.expected[n], L = ex.pre.length;
    assert.deepEqual(zs, [null, sg(ex.pre[0][legacyChosen(want, 0)], 3), sg(ex.pre[1][legacyChosen(want, 1)], 3), sg(ex.pre[L - 1][want.win], 3)]);
    assert.equal(doc.getElementById("how-n").textContent, "4 / 4");
    assert.deepEqual(st.reading().failed, []);
    assert.deepEqual(missingKeys(), []);
    assert.deepEqual(unresolved(), []);
  });
});

// ---- a graph node: the graph steps are the graph explainer's (W14); an answer that does not bring
// a stage's numbers has that step say which tensors it lacks, and the walk goes on
test("a graph node: the input step names the neighbourhood per hop; the graph steps name the tensors the answer lacks", async () => {
  const doc = install();
  const view = FIX("gcn-node"), graph = StageGraph.from(view), names = graph.task.outputs.names;
  configureText({ lex: new Lexicon(null, null), lane: { viewKey: graph.viewKey, shape: "graph" } });
  const ex = {
    schema: "vega/explanation@2", detail: "exact", degraded: false,
    sample: { shape: "graph", index: 0, row: 0, span: null, graph: { level: "node", nodes: 9, edges: 14, target_node: 0 } },
    target: { kind: "logit", index: 1, name: names[1], sign: 1, value: 0.7, why: "predicted", unit: "logit" },
    outputs: { kind: "classes", names, link: "softmax", raw: names.map((_, k) => (k === 1 ? 0.7 : -0.2)), read: null, answer: { class: names[1] } },
    inputs: { order: graph.inputs.names, shaped: graph.inputs.names.map((_, k) => k / 10), raw: null },
    attribution: { shape: "graph", neighbours: [{ node: 3, hop: 1 }, { node: 5, hop: 1 }, { node: 8, hop: 2 }] },
    nodes: {}, chain: null, warnings: [],
  };
  const host = doc.createElement("div");
  doc.body.append(host);
  const st = await new Stepper({ graph, ex }).mount(host);
  st.show(0);
  const input = doc.getElementById("how-body").textContent;
  assert.match(input, /9 nodes and 14 edges/);
  assert.match(input, /2 nodes at hop 1 and 1 node at hop 2/);
  const bodies = st.steps().map((_, i) => { st.show(i); return doc.getElementById("how-body").textContent; });
  assert.match(bodies[1], /cannot recompute this step for this .*: the answer did not bring B\d+\.in, B\d+\.out/, "a graph convolution: what it lacks, named");
  assert.doesNotMatch(bodies[1], /does not recompute this kind of step yet/, "not the generic explainer's");
  assert.match(bodies.at(-1), /Softmax/);
  assert.deepEqual(st.reading().gaps, [], "the graph explainer loaded: nothing fell back");
  assert.ok(st.values().filter((v) => v.kind === "graph_conv").every((v) => v.check === null), "nothing recomputed without the stage's numbers");
  assert.deepEqual(st.reading().failed, []);
  assert.deepEqual(missingKeys(), []);
});

// ---- the pooling explainer (explainers/pool.js) and a standalone bend (explainers/dense.js), on
// hand-computed values: a (L, C) = (4, 3) input, position t's channel c at t·C + c (blocks.py's
// (B, L, C) layout for one sample). Every number is exact in float32, so page and network agree to
// the last bit.
const POOL_X = [1, -2, 0.5, 3, 0, -1, -4, 5, 2, 2, 1, 7];
const f32 = (xs) => ({ shape: [xs.length], axes: ["a0"], dtype: "float32", encoding: "base64-le", data: Buffer.from(new Float32Array(xs).buffer).toString("base64") });
const stageNode = (id, kind, payload, in_shape, out_shape) => Object.freeze({ id, kind, type: kind, payload, config: {}, in_shape, out_shape, children: [] });
// A stage graph of one node and an explanation@2 at detail exact carrying its in, out and grad_out
// (a missing x or y leaves that tensor out).
function oneStage(node, x, y, grad = null) {
  const graph = { chain: null, byId: (id) => (id === node.id ? node : null) };
  const exact = { ...(x ? { in: f32(x) } : {}), ...(y ? { out: f32(y), grad_out: f32(grad || y.map(() => 1)) } : {}) };
  return { graph, ex: { schema: "vega/explanation@2", detail: "exact", nodes: { [node.id]: { level: "exact", exact } } } };
}

test("pool.js pools a (L, C) input as blocks.py does: pool mean, max, cls over every position; pool1d max and avg over groups of size", async () => {
  const { poolUnit, poolOf, poolInputs } = await import("../js/how/explainers/pool.js");
  const unitsOf = (n, count) => Array.from({ length: count }, (_, j) => poolUnit(n, POOL_X, j));
  const pool = (kind) => stageNode("P", "pool", kind === undefined ? {} : { kind }, [4, 3], [3]);
  const pool1d = (kind, size = 2) => stageNode("Q", "pool1d", kind === undefined ? { size } : { kind, size }, [4, 3], [4 / size, 3]);
  assert.deepEqual(unitsOf(pool("mean"), 3), [0.5, 1, 2.125]);
  assert.deepEqual(unitsOf(pool("max"), 3), [3, 5, 7]);
  assert.deepEqual(unitsOf(pool("cls"), 3), [1, -2, 0.5]);
  assert.deepEqual(unitsOf(pool("first"), 3), [1, -2, 0.5], "first: the [cls] token's older name");
  assert.deepEqual(unitsOf(pool(undefined), 3), [0.5, 1, 2.125], "a pool with no kind is the block's default, mean");
  // pool1d: unit j is channel j mod C of group ⌊j / C⌋ (positions 2·⌊j / C⌋ and the one after)
  assert.deepEqual(unitsOf(pool1d("max"), 6), [3, 0, 0.5, 2, 5, 7]);
  assert.deepEqual(unitsOf(pool1d("avg"), 6), [2, -1, -0.25, -1, 3, 4.5]);
  assert.deepEqual(unitsOf(pool1d(undefined), 6), [3, 0, 0.5, 2, 5, 7], "a pool1d with no kind is the block's default, max");
  assert.deepEqual(unitsOf(pool1d("max", 4), 3), [3, 5, 7], "one group of every position: the pool's max");
  assert.deepEqual(poolInputs(pool1d("avg"), POOL_X, 4), [{ t: 2, x: 5 }, { t: 3, x: 1 }]);
  assert.deepEqual([poolOf(pool1d("avg")).kind, poolOf(pool1d("avg")).name, poolOf(pool1d("avg")).size], ["mean", "avg", 2]);
  // a name the block does not take: no reduction, so no recompute
  for (const n of [pool("avg"), pool1d("cls"), pool("median")]) {
    assert.equal(poolOf(n).kind, null, `${n.kind} ${n.payload.kind}`);
    assert.equal(poolUnit(n, POOL_X, 0), null);
  }
});

test("pool.js recomputes pool and pool1d steps against the network's values, and names what stops it", async () => {
  configureText({ lex: new Lexicon(null, null), lane: { viewKey: "classes", shape: "window" } });
  const doc = install();
  const pool = (await import("../js/how/explainers/pool.js")).default;
  const env = (graph) => ({ graph, target: "the answer", choose: () => {}, nameOf: () => "pooling", pickOf: () => -1 });
  const cases = [
    [stageNode("P", "pool", { kind: "mean" }, [4, 3], [3]), [0.5, 1, 2.125]],
    [stageNode("P", "pool", { kind: "max" }, [4, 3], [3]), [3, 5, 7]],
    [stageNode("P", "pool", { kind: "cls" }, [4, 3], [3]), [1, -2, 0.5]],
    [stageNode("Q", "pool1d", { kind: "max", size: 2 }, [4, 3], [2, 3]), [3, 0, 0.5, 2, 5, 7]],
    [stageNode("Q", "pool1d", { kind: "avg", size: 2 }, [4, 3], [2, 3]), [2, -1, -0.25, -1, 3, 4.5]],
  ];
  for (const [node, y] of cases) {
    const { graph, ex } = oneStage(node, POOL_X, y), step = { kind: node.kind, stage: node.id, pick: null };
    const check = pool.recompute(step, ex, graph);
    const j = y.map((_, k) => k).sort((a, b) => Math.abs(y[b]) - Math.abs(y[a]))[0];
    assert.equal(check.unit, j, `${node.kind} ${node.payload.kind}: the unit that carried most (out ⊙ grad_out)`);
    assert.deepEqual([check.page, check.network, check.gap, check.within], [y[j], y[j], 0, true], `${node.kind} ${node.payload.kind}`);
    for (let k = 0; k < y.length; k++) assert.equal(pool.recompute({ ...step, pick: k }, ex, graph).page, y[k], `${node.kind} ${node.payload.kind} unit ${k}`);
    const body = pool.render(step, ex, env(graph)).textContent;
    assert.match(body, node.payload.kind === "max" ? /keeps the largest/ : node.payload.kind === "cls" ? /first position/ : /is the mean of its/);
    assert.doesNotMatch(body, /did not bring|does not know/);
  }
  // a wrong network value shows as a gap, never hidden
  const off = oneStage(cases[4][0], POOL_X, [2, -1, -0.25, -1, 3, 4.75]);
  const gap = pool.recompute({ kind: "pool1d", stage: "Q", pick: 5 }, off.ex, off.graph);
  assert.deepEqual([gap.page, gap.network, gap.gap, gap.within], [4.5, 4.75, 0.25, false]);
  // what stops the recompute, said as it is
  const unknown = stageNode("P", "pool", { kind: "median" }, [4, 3], [3]), u = oneStage(unknown, POOL_X, [1, 1, 1]);
  assert.equal(pool.recompute({ kind: "pool", stage: "P", pick: null }, u.ex, u.graph), null);
  const said = pool.render({ kind: "pool", stage: "P", pick: null }, u.ex, env(u.graph)).textContent;
  assert.match(said, /names this pooling median, which the page does not know/);
  assert.doesNotMatch(said, /did not bring/, "both tensors came: none is named missing");
  const noOut = oneStage(cases[3][0], POOL_X, null);
  assert.match(pool.render({ kind: "pool1d", stage: "Q", pick: null }, noOut.ex, env(noOut.graph)).textContent, /did not bring Q\.out\./);
  const noIn = oneStage(cases[0][0], null, [0.5, 1, 2.125]);
  assert.match(pool.render({ kind: "pool", stage: "P", pick: null }, noIn.ex, env(noIn.graph)).textContent, /did not bring P\.in\./);
  const short = oneStage(stageNode("Q", "pool1d", { kind: "max", size: 2 }, [4, 3], [2, 3]), POOL_X.slice(0, 3), [3, 0, 0.5, 2, 5, 7]);
  assert.match(pool.render({ kind: "pool1d", stage: "Q", pick: 5 }, short.ex, env(short.graph)).textContent, /do not fit its shape 4 × 3/);
  // a reshape step: the same numbers, or the first one that moved
  const flat = stageNode("F", "reshape", {}, [4, 3], [12]), same = oneStage(flat, POOL_X, POOL_X);
  assert.deepEqual(pool.recompute({ kind: "reshape", stage: "F" }, same.ex, same.graph).gap, 0);
  const moved = oneStage(flat, POOL_X, [...POOL_X.slice(0, 11), 8]);
  const m = pool.recompute({ kind: "reshape", stage: "F" }, moved.ex, moved.graph);
  assert.deepEqual([m.unit, m.gap, m.within], [11, 1, false]);
  assert.deepEqual(missingKeys(), []);
  assert.deepEqual(unresolved(), []);
  doc.body.replaceChildren();
});

test("dense.js recomputes a standalone bend a = f(z) against hand-computed values", async () => {
  configureText({ lex: new Lexicon(null, null), lane: { viewKey: "classes", shape: "window" } });
  install();
  const dense = (await import("../js/how/explainers/dense.js")).default;
  const z = [0.5, -2, 1.5, -0.25];
  const bends = { relu: [0.5, 0, 1.5, 0], leaky_relu: [0.5, -0.02, 1.5, -0.0025], tanh: z.map(Math.tanh), none: z };
  for (const [name, a] of Object.entries(bends)) {
    const node = stageNode("A", "activation", { name }, [4], [4]), { graph, ex } = oneStage(node, z, a);
    for (let k = 0; k < z.length; k++) {
      const c = dense.recompute({ kind: "activation", stage: "A", pick: k }, ex, graph);
      assert.equal(c.what, "a");
      assert.ok(c.within && Math.abs(c.gap) < 1e-6, `${name} unit ${k}: ${c.page} vs ${c.network}`);
    }
    const body = dense.render({ kind: "activation", stage: "A", pick: null }, ex, { graph, target: "the answer", choose: () => {}, nameOf: () => "activation", pickOf: () => -1 });
    assert.ok(body.querySelector(".act"), `${name}: the bend drawn`);
  }
  const miss = oneStage(stageNode("A", "activation", { name: "relu" }, [4], [4]), z, null);
  assert.equal(dense.recompute({ kind: "activation", stage: "A", pick: null }, miss.ex, miss.graph), null);
  assert.match(dense.render({ kind: "activation", stage: "A", pick: null }, miss.ex, { graph: miss.graph, target: "", choose: () => {}, nameOf: () => "", pickOf: () => -1 }).textContent,
    /did not bring A\.out\./);
  assert.deepEqual(missingKeys(), []);
});
