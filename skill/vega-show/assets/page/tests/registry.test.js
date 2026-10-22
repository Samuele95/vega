// node --test: the view registry (views/registry.js, D4 §6.2) and the eight task modules it resolves.
// The lane's view key picks one task module and its sample shape one shape module, each loaded
// lazily and held to its members; a gap is a RegistryGap naming the slot and the key. The task
// modules are then run on the answers the service writes: fixtures/predictions/predictions.json,
// made by the serving AnswerCodec from vega-pack's output record for every reading (R-C4: D2's
// RowPrediction keys, never a hand-typed copy), with and without the gated includes, a window row
// the service did not answer, and a graph-level answer with its graph_id. Every slot renders, with
// the answer's own values, and every word it shows is a key of the strings table. The weights axis
// of each module is read against the demo sample's attribution (vega/attribution@2), and the network
// view's weights read (net/weights.js) takes it for the columns of its map on real rows.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// ---- a document small enough for node: elements (HTML and SVG), text, fragments, class queries
class FakeNode {
  constructor(nodeType, tag = null) {
    Object.assign(this, { nodeType, tagName: tag, childNodes: [], attrs: {}, style: { cssText: "" }, value: "", title: "", parentNode: null });
    this.classList = {
      add: (...c) => { this.className = [...new Set([...this.classes(), ...c])].join(" "); },
      remove: (...c) => { this.className = this.classes().filter((x) => !c.includes(x)).join(" "); },
      toggle: (c, on) => { if (on ?? !this.classes().includes(c)) this.classList.add(c); else this.classList.remove(c); },
      contains: (c) => this.classes().includes(c),
    };
  }
  get className() { return this.attrs.class || ""; }
  set className(v) { this.attrs.class = String(v); }
  classes() { return this.className.split(/\s+/).filter(Boolean); }
  get textContent() { return this.nodeType === 3 ? this.value : this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) { if (this.nodeType === 3) this.value = String(v); else this.replaceChildren(String(v)); }
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  get firstElementChild() { return this.children[0] || null; }
  nodesOf(x) { return typeof x === "string" ? [text(x)] : x.nodeType === 11 ? x.childNodes.splice(0) : [x]; }
  append(...xs) { for (const n of xs.flatMap((x) => this.nodesOf(x))) { n.parentNode = this; this.childNodes.push(n); } }
  prepend(...xs) { const ns = xs.flatMap((x) => this.nodesOf(x)); ns.forEach((n) => { n.parentNode = this; }); this.childNodes.unshift(...ns); }
  replaceChildren(...xs) { this.childNodes = []; this.append(...xs); }
  getAttribute(k) { return k === "class" ? this.className : k in this.attrs ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
  get dataset() {
    const a = this.attrs;
    return new Proxy({}, { get: (_, k) => a[`data-${String(k)}`], set: (_, k, v) => { a[`data-${String(k)}`] = String(v); return true; } });
  }
  addEventListener() {}
  matches(sel) { return sel.startsWith(".") ? this.classes().includes(sel.slice(1)) : sel.startsWith("[") ? sel.slice(1, -1) in this.attrs : this.tagName === sel; }
  querySelectorAll(sel) { return this.childNodes.flatMap((c) => (c.nodeType === 1 ? [...(c.matches(sel) ? [c] : []), ...c.querySelectorAll(sel)] : [])); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}
const text = (s) => Object.assign(new FakeNode(3), { value: String(s) });
globalThis.document = {
  documentElement: { lang: "en-US" },
  createDocumentFragment: () => new FakeNode(11),
  createTextNode: (s) => text(s),
  createElement: (tag) => (tag === "canvas" ? { getContext: () => ({ fillStyle: "#000000" }) } : new FakeNode(1, tag)),
  createElementNS: (ns, tag) => new FakeNode(1, tag),
};

const { configureText, missingKeys, unresolved, tText } = await import("../js/core/text.js");
const { STRINGS } = await import("../js/core/strings.en.js");
const { Lane } = await import("../js/core/lane.js");
const { Lexicon } = await import("../js/core/lexicon.js");
const { RoleMap } = await import("../js/core/roles.js");
const { outputRecord } = await import("../js/views/payloads.js");
const { fmtNum } = await import("../js/core/fmt.js");
const R = await import("../js/views/registry.js");
const { anomalySides } = await import("../js/views/tasks/shared.js");
const { ANOMALY_SIDES } = await import("../js/core/roles.js");
const W = await import("../js/net/weights.js");

const PAGE = fileURLToPath(new URL("..", import.meta.url));
const FIX = JSON.parse(readFileSync(join(PAGE, "tests", "fixtures", "predictions", "predictions.json"), "utf8"));
const onDisk = (rel) => existsSync(join(PAGE, "js", "views", rel));
const txt = (n) => (n === null || n === undefined ? "" : typeof n === "string" ? n : n.textContent).replace(/\s+/g, " ").trim();
const bars = (n) => (n ? n.querySelectorAll(".bar").length : 0);
const pct = (v) => `${(100 * v).toFixed(1)} %`;
const shown = (v) => fmtNum(+v.toPrecision(4));        // a value as the task modules write it

// ---- the registry
test("every view key and every sample shape maps to one module path", () => {
  assert.deepEqual(Object.keys(R.TASK_MODULES).sort(), [...R.VIEW_KEYS].sort());
  assert.equal(R.VIEW_KEYS.length, 8);
  assert.deepEqual(Object.keys(R.SHAPE_MODULES).sort(), ["graph", "row", "window"]);
  for (const [vk, path] of Object.entries(R.TASK_MODULES)) assert.equal(path, `./tasks/${vk}.js`);
  for (const [shape, path] of Object.entries(R.SHAPE_MODULES)) assert.equal(path, `./shapes/${shape}.js`);
});

test("every task module on disk exports output, summary, weightsAxis, breakdown and headLabel with their members", async () => {
  for (const [vk, path] of Object.entries(R.TASK_MODULES)) {
    assert.ok(onDisk(path), `${path}: the eight task modules ship in this wave`);
    const mod = await R.moduleFor("task", vk);
    assert.deepEqual(R.missingMembers(mod, R.TASK_MEMBERS), [], vk);
    assert.equal(mod.headLabel, `head.caption@${vk}`);
    assert.ok(mod.headLabel in STRINGS, `${mod.headLabel} is a key of the table`);
  }
});

test("a shape module on disk exports editor, sources, attribution, inputStep and drawInput; a shape without one is a RegistryGap", async () => {
  assert.ok(onDisk(R.SHAPE_MODULES.row), "the row shape ships in this wave");
  for (const [shape, path] of Object.entries(R.SHAPE_MODULES)) {
    if (onDisk(path)) {
      assert.deepEqual(R.missingMembers(await R.moduleFor("shape", shape), R.SHAPE_MEMBERS), [], shape);
    } else {
      await assert.rejects(R.moduleFor("shape", shape), (e) => e instanceof R.RegistryGap && e.slot === "shape" && e.key === shape);
    }
  }
});

test("resolve(lane) loads the lane's two modules, for every view key on rows", async () => {
  for (const c of FIX.cases.filter((x) => x.lane.shape === "row")) {
    const lane = Lane.from(c.lane), got = await R.resolve(lane);
    assert.equal(got.task, await import(`../js/views/tasks/${lane.viewKey}.js`), c.id);
    assert.equal(got.shape, await import("../js/views/shapes/row.js"), c.id);
    assert.deepEqual(R.registryReading(lane, got), { view_key: lane.viewKey, shape: "row", task: `./tasks/${lane.viewKey}.js`,
      shape_module: "./shapes/row.js", resolved: true, view_keys: [...R.VIEW_KEYS], shapes: ["row", "window", "graph"] });
  }
});

test("a gap is a RegistryGap naming the slot, the key and why: unmapped, unloaded, members missing", async () => {
  await assert.rejects(R.moduleFor("task", "clusters"), (e) => e instanceof R.RegistryGap && e.slot === "task" && e.key === "clusters"
    && e.why === tText("registry.gap.unmapped"));
  await assert.rejects(R.moduleFor("task", "classes", { importer: () => { throw new Error("offline"); } }),
    (e) => e instanceof R.RegistryGap && e.why === tText("registry.gap.unloaded", { error: "offline" }));
  const half = { output: { verdict() {}, agree() {} }, summary: {}, weightsAxis() {}, headLabel: "" };
  assert.deepEqual(R.missingMembers(half, R.TASK_MEMBERS), ["output.detail", "output.note", "summary.sentence", "summary.left",
    "summary.right", "summary.columns", "summary.cells", "summary.miss", "breakdown", "headLabel"]);
  await assert.rejects(R.moduleFor("task", "classes", { importer: async () => half }),
    (e) => e instanceof R.RegistryGap && e.message.includes("output.detail"));
  await assert.rejects(R.moduleFor("shape", "row", { importer: async () => ({ editor: { mount() {} } }) }),
    (e) => e instanceof R.RegistryGap && e.message.includes("editor.load") && e.message.includes("drawInput"));
});

test("a sequence lane resolves its base's task module; a graph-level lane its task's", async () => {
  const seq = Lane.from(FIX.cases.find((c) => c.id === "sequence-ordinal").lane);
  assert.equal(await R.moduleFor("task", seq.viewKey), await import("../js/views/tasks/ordinal.js"));
  const graph = Lane.from(FIX.cases.find((c) => c.id === "graph-multiclass").lane);
  assert.equal(await R.moduleFor("task", graph.viewKey), await import("../js/views/tasks/classes.js"));
});

// ---- the task modules on the service's answers
test("the fixture is the serving codec's: every reading, the keys R-C4 names, the gated keys only on include", () => {
  const READINGS = ["binary", "multiclass", "multilabel", "ordinal", "regression", "quantile", "forecast", "reconstruction",
    "anomaly-reconstruction", "anomaly-svdd"];
  assert.deepEqual([...new Set(FIX.cases.map((c) => c.reading))].sort(), READINGS.sort());
  const KEYS = {
    binary: ["label", "probabilities"], multiclass: ["label", "probabilities"], multilabel: ["labels", "probabilities"],
    ordinal: ["label", "probabilities", "cumulative"], regression: ["value"], quantile: ["value", "quantiles"], forecast: ["values"],
    reconstruction: ["reconstruction_error"], "anomaly-reconstruction": ["score", "anomalous"], "anomaly-svdd": ["score", "anomalous"],
  };
  const GATED = { logits: ["binary", "multiclass", "multilabel", "ordinal"], reconstruction: ["reconstruction", "anomaly-reconstruction"],
    embedding: ["anomaly-svdd"] };
  for (const c of FIX.cases) {
    for (const k of [...KEYS[c.reading], "answered", "warnings"]) assert.ok(c.keys.includes(k), `${c.id}: ${k}`);
    for (const [k, readings] of Object.entries(GATED)) {
      assert.equal(c.keys.includes(k), false, `${c.id}: ${k} only on include`);
      assert.equal(c.keys_included.includes(k), readings.includes(c.reading), `${c.id}: ${k} on include`);
    }
    assert.equal(c.predictions.length, FIX.rows);
    c.predictions.forEach((p) => assert.deepEqual(Object.keys(p).sort(), [...c.keys].filter((k) => k in p).sort()));
  }
});

// The environment of one case, as the boot builds it (main.js): the lane, its words, the roles, the
// output record /v1/schema carries.
function envOf(c) {
  const lane = Lane.from(c.lane), lex = new Lexicon(null, null);
  const network = { schema: "vega/network-view@2", task: { kind: c.lane.task, base: c.lane.base, link: c.lane.link,
    outputs: { kind: "classes", names: c.schema.output.classes || [] } }, sample: { shape: c.lane.shape, level: c.lane.level } };
  const roles = new RoleMap(c.class_roles, lane, network);
  const story = { values: { ...c.values, primary: { value: 0.8, ci: [0.7, 0.9], n: 60 } } };
  configureText({ lane, lex, values: story.values });
  return { lane, lex, roles, graph: null, story, page: { reference: c.reference }, schema: c.schema, output: outputRecord(c.schema, null), examples: null };
}

// The demo sample's attribution of a case, in vega/attribution@2's form (schemas/attribution-2.schema.json):
// one column per stratum of the lane's weights axis (D4 §6.4: classes, labels, the target's bins, the
// groups, the steps ahead), each cell the rows' attribution over the network inputs. The class and
// label columns come in an order and with ids of their own (the reverse of the lane's, "k-<name>"), so
// an axis that took the head's order or made its ids up would not index the cells.
const INPUTS = ["a", "b", "zone"];
const AXIS_KIND = { classes: "class", binary: "class", labels: "label", ordinal: "class", value: "bin", reconstruction: "group", anomaly: "group", horizon: "horizon" };
function attributionFor(c) {
  const vk = c.lane.view_key, o = c.schema.output, classes = o.classes || [], col = (id, name, range = null) => ({ id, name, range });
  const columns = AXIS_KIND[vk] === "class" || AXIS_KIND[vk] === "label" ? classes.slice().reverse().map((n) => col(`k-${n}`, n))
    : vk === "value" ? c.values.breakdown_names.map((n, i) => col(`b${i}`, n, [i, i + 1]))
      : vk === "horizon" ? o.horizons.map((h) => col(`h${h}`, `h${h}`))
        : vk === "anomaly" && classes.length === 2 ? classes.map((n, i) => col(`g${i}`, n)) : [col("g0", "every row")];
  const cells = Object.fromEntries(columns.map((x, k) => [x.id, { n: 2, correct: null,
    rows: [[0.1 * (k + 1), -0.2, 0.05], [0.3 * (k + 1), -0.1, -0.02]] }]));
  return { schema: "vega/attribution@2", source: { file: "examples/demo.json", part: "train" }, method: { name: "grad_x_input", steps: null },
    target: "logit", inputs: INPUTS.slice(), reduce: null, axis: { kind: AXIS_KIND[vk], columns }, cells };
}

async function render(c, env, preds) {
  const mod = await R.moduleFor("task", env.lane.viewKey), out = {};
  const answered = preds.map((p, i) => [p, c.truth ? c.truth[i] : null]).filter(([p]) => p.answered !== false);
  const p0 = answered[0][0], a = { p: p0, ex: c.explanation };
  const preset = { id: "typical", group: "typical", kind: "typical_correct", expected: c.truth ? c.truth[0] : null };
  out.verdict = mod.output.verdict(a, env);
  out.agree = mod.output.agree(a, preset, env);
  out.detail = mod.output.detail(a, env);
  out.note = mod.output.note(env);
  const ps = answered.map(([p]) => p), ts = c.truth ? answered.map(([, x]) => x) : null;
  for (const truth of [ts, null]) {
    const s = mod.summary;
    out[truth ? "withTruth" : "withoutTruth"] = { sentence: s.sentence(ps, truth, env), left: s.left(ps, truth, env), right: s.right(ps, truth, env) };
  }
  out.columns = mod.summary.columns(env);
  out.cells = preds.map((p, i) => mod.summary.cells(p, c.truth ? c.truth[i] : null, env));
  out.miss = preds.map((p, i) => mod.summary.miss(p, c.truth ? c.truth[i] : null, env));
  out.axis = mod.weightsAxis(attributionFor(c), env);
  out.breakdown = mod.breakdown(c.breakdown, c.values, env);
  return { mod, out, p0 };
}

test("every task module renders every slot on every reading's answers, with and without the includes, in words of the table", async () => {
  for (const c of FIX.cases) {
    for (const preds of [c.predictions, c.predictions_included]) {
      const env = envOf(c), { out } = await render(c, env, preds);
      assert.ok(txt(out.verdict).length, `${c.id}: verdict`);
      assert.ok(out.detail, `${c.id}: detail`);
      for (const side of c.truth ? ["withTruth", "withoutTruth"] : ["withoutTruth"]) {
        assert.ok(txt(out[side].sentence).length, `${c.id}: ${side} sentence`);
        assert.ok(out[side].left && out[side].left.body && out[side].left.title, `${c.id}: ${side} left`);
      }
      assert.ok(out.columns.length >= 2 && out.columns.every((k) => k in STRINGS), `${c.id}: columns ${out.columns}`);
      out.cells.forEach((cells) => assert.equal(cells.length, out.columns.length, `${c.id}: one cell per column`));
      out.cells.forEach((cells) => cells.forEach((x) => assert.ok(!/\b(null|undefined|NaN)\b/.test(txt(x)), `${c.id}: cell ${txt(x)}`)));
      assert.ok(Array.isArray(out.axis.columns) && out.axis.title, `${c.id}: weights axis`);
      assert.ok(out.breakdown && bars(out.breakdown) === c.values.breakdown_names.length, `${c.id}: breakdown`);
      assert.equal(out.breakdown.querySelectorAll(".weak").length, c.values.breakdown_weak.filter(Boolean).length, `${c.id}: the story's weak marks`);
    }
  }
  assert.deepEqual(missingKeys(), [], "every key the modules asked for is in the table");
  assert.deepEqual(unresolved(), [], "every placeholder resolved");
});

test("classes (multiclass, and at graph level): the label, its probability, the five highest, right per true class", async () => {
  for (const id of ["multiclass", "graph-multiclass"]) {
    const c = FIX.cases.find((x) => x.id === id), env = envOf(c), { out, p0 } = await render(c, env, c.predictions);
    assert.ok(txt(out.verdict).startsWith(p0.label), id);
    assert.ok(txt(out.verdict).includes((100 * p0.probabilities[p0.label]).toFixed(1)), id);
    assert.equal(bars(out.detail), Math.min(5, Object.keys(p0.probabilities).length));
    const agree = c.predictions.filter((p, i) => p.label === c.truth[i]).length;
    assert.ok(txt(out.withTruth.sentence).startsWith(String(agree)), `${id}: ${txt(out.withTruth.sentence)}`);
    assert.deepEqual(out.cells[0].map(txt), [c.truth[0], p0.label, pct(p0.probabilities[p0.label]), ""]);
    assert.deepEqual(out.miss, c.predictions.map((p, i) => p.label !== c.truth[i]));
    assert.deepEqual(out.axis.columns.map((x) => x.name), c.schema.output.classes);
  }
});

test("binary (and a sequence of a binary base): the class at the threshold, p of the positive class, the threshold marked", async () => {
  for (const id of ["binary", "sequence-binary"]) {
    const c = FIX.cases.find((x) => x.id === id), env = envOf(c), { out, p0 } = await render(c, env, c.predictions);
    const positive = c.schema.output.positive, p = p0.probabilities[positive];
    assert.equal(p0.label === positive, p >= c.schema.output.threshold, `${id}: the fixture decides at the threshold`);
    assert.ok(txt(out.verdict).startsWith(p0.label), id);
    assert.ok(txt(out.verdict).includes(`p(${positive})`), id);
    assert.equal(out.detail.querySelectorAll(".ref").length, 1, `${id}: the threshold marker`);
    assert.ok(txt(out.note).includes(positive), `${id}: the note names the positive class`);
    assert.deepEqual(out.cells[0].map(txt).slice(1, 3), [p0.label, pct(p)]);
    assert.deepEqual(out.axis.columns.map((x) => x.name), [c.schema.output.classes.find((x) => x !== positive), positive]);
    if (id === "sequence-binary") assert.deepEqual(out.cells[FIX.rows - 1].map(txt).slice(1, 3), ["—", "—"], "an unanswered window row");
  }
});

test("labels: the label set as chips in their roles, every label's p with its own threshold", async () => {
  const c = FIX.cases.find((x) => x.id === "multilabel"), env = envOf(c), { out, p0 } = await render(c, env, c.predictions);
  assert.deepEqual(out.verdict.querySelectorAll(".views-chip").map(txt), p0.labels);
  assert.equal(bars(out.detail), Object.keys(p0.probabilities).length);
  assert.equal(out.detail.querySelectorAll(".ref").length, Object.keys(p0.probabilities).length, "one threshold per label");
  const exact = c.predictions.filter((p, i) => [...p.labels].sort().join() === [...c.truth[i]].sort().join()).length;
  assert.ok(txt(out.withTruth.sentence).startsWith(String(exact)));
  assert.deepEqual(out.cells[0].map(txt), [c.truth[0].join(", "), p0.labels.join(", "), ""]);
  assert.equal(env.roles.cls("red"), "role-neutral");
});

test("ordinal (and a sequence of an ordinal base): the class, the cumulative steps with the half-way line, distances", async () => {
  for (const id of ["ordinal", "sequence-ordinal"]) {
    const c = FIX.cases.find((x) => x.id === id), env = envOf(c), { out, p0 } = await render(c, env, c.predictions);
    const passed = Object.values(p0.cumulative).filter((v) => v >= 0.5).length;
    assert.equal(p0.label, c.schema.output.classes[passed], `${id}: cumulative_count (R-C6)`);
    assert.ok(txt(out.verdict).startsWith(p0.label));
    assert.equal(bars(out.detail), Object.keys(p0.cumulative).length + c.schema.output.classes.length);
    assert.equal(out.detail.querySelectorAll(".ref").length, Object.keys(p0.cumulative).length, "the half-way line per step");
    const d = Math.abs(c.schema.output.classes.indexOf(p0.label) - c.schema.output.classes.indexOf(c.truth[0]));
    assert.equal(txt(out.cells[0][2]), String(d));
  }
});

test("value (regression and quantile): the value, the interval of the outermost quantiles, residuals against the truth", async () => {
  for (const id of ["regression", "quantile"]) {
    const c = FIX.cases.find((x) => x.id === id), env = envOf(c), { out, p0 } = await render(c, env, c.predictions);
    assert.ok(txt(out.verdict).startsWith(shown(p0.value)), `${id}: ${txt(out.verdict)}`);
    const qs = Object.values(p0.quantiles || {});
    if (id === "quantile") {
      assert.ok(txt(out.verdict).includes(shown(Math.min(...qs))) && txt(out.verdict).includes(shown(Math.max(...qs))));
      assert.equal(bars(out.detail), qs.length);
    } else assert.equal(bars(out.detail), 0, `${id}: no training range stated, no residual context`);
    assert.equal(txt(out.cells[0][2]), shown(p0.value - c.truth[0]));
    assert.equal(out.withTruth.left.body.tagName, "svg", "answered against true");
    // the residual context (D4 §6.4): the value on the target's training range, when /v1/schema states it
    const lo = Math.min(...c.truth, p0.value), hi = Math.max(...c.truth, p0.value);
    const ranged = (range) => ({ ...env, schema: { ...env.schema, label: { column: "target", decode: {}, range_seen: range } } });
    const detail = (await R.moduleFor("task", "value")).output.detail({ p: p0, ex: c.explanation }, ranged([lo, hi]));
    assert.equal(bars(detail), 1 + qs.length, `${id}: the range bar, then the quantiles`);
    const bar = detail.querySelector(".bar");
    assert.equal(txt(bar.querySelector(".name")), tText("try.range", { lo: shown(lo), hi: shown(hi) }));
    assert.equal(txt(bar.querySelector(".val")), shown(p0.value));
    assert.equal(detail.querySelector(".views-range-beyond"), null, `${id}: within the range`);
    const beyond = (await R.moduleFor("task", "value")).output.detail({ p: p0, ex: c.explanation }, ranged([hi + 1, hi + 2]));
    assert.equal(txt(beyond.querySelector(".views-range-beyond")), tText("try.range.beyond"), `${id}: beyond the range, said`);
  }
});

test("horizon: the forecast steps ahead as a line, the quantile band, the unanswered window row", async () => {
  for (const id of ["forecast", "forecast-quantile"]) {
    const c = FIX.cases.find((x) => x.id === id), env = envOf(c), { out, p0 } = await render(c, env, c.predictions);
    const H = Object.keys(p0.values).length;
    assert.ok(txt(out.verdict).includes(String(H)), id);
    assert.equal(bars(out.detail), H);
    assert.equal(out.verdict.querySelectorAll("polygon").length, id === "forecast-quantile" ? 1 : 0, `${id}: the quantile band`);
    assert.equal(out.detail.querySelectorAll(".ci").length, id === "forecast-quantile" ? H : 0);
    assert.equal(txt(out.cells[FIX.rows - 1][0]), "", `${id}: no forecast for an unanswered row`);
    assert.equal(c.predictions[FIX.rows - 1].answered, false);
  }
});

test("reconstruction: the error, where it falls among the training rows, the inputs rebuilt worst", async () => {
  const c = FIX.cases.find((x) => x.id === "reconstruction"), env = envOf(c);
  const { out, p0 } = await render(c, env, c.predictions_included);
  assert.ok(txt(out.verdict).includes(shown(p0.reconstruction_error)));
  const q = c.reference.quantiles, e = p0.reconstruction_error;
  const band = e >= q["0.99"] ? "try.band.p99" : e >= q["0.9"] ? "try.band.p90" : e >= q["0.5"] ? "try.band.p50" : "try.band.below";
  assert.ok(txt(out.verdict).endsWith(tText(band)));
  assert.equal(bars(out.detail), c.explanation.inputs.order.length, "one bar per rebuilt input");
  assert.deepEqual(out.columns, ["many.col.error", "many.col.warnings"]);
});

test("anomaly (reconstruction and distance scores): the flag in its side's role, the score against the threshold", async () => {
  for (const id of ["anomaly-reconstruction", "anomaly-svdd"]) {
    const c = FIX.cases.find((x) => x.id === id), env = envOf(c), { out, p0 } = await render(c, env, c.predictions_included);
    assert.equal(p0.anomalous, p0.score >= c.schema.output.threshold, `${id}: SA11`);
    assert.ok(txt(out.verdict).startsWith(tText(p0.anomalous ? "anomaly.flag.on" : "anomaly.flag.off")));
    assert.ok(out.verdict.querySelector(".role-alert"), `${id}: the anomalous side is alert by task (R-E5)`);
    assert.equal(out.detail.querySelectorAll(".ref").length, 1, "the threshold marker");
    const flagged = c.predictions.filter((p) => p.anomalous).length;
    assert.ok(txt(out.withoutTruth.sentence).startsWith(String(flagged)));
    assert.equal(txt(out.cells[0][2]), tText(p0.anomalous ? "anomaly.flag.on" : "anomaly.flag.off"));
    if (id === "anomaly-reconstruction") assert.ok(bars(out.detail) > 1, "the inputs rebuilt worst");
    if (id === "anomaly-svdd") assert.deepEqual(out.miss, c.predictions.map((p, i) => p.anomalous !== (c.truth[i] === "fraud")));
  }
});

test("the anomaly sides are one reading: the task's classes when the data has a label, else the role map's two words", async () => {
  const A = await import("../js/views/tasks/anomaly.js");
  assert.equal(A.sidesOf, anomalySides, "the task module and the try view's preset dots read the same sides");
  assert.deepEqual(anomalySides(envOf(FIX.cases.find((x) => x.id === "anomaly-svdd"))), { normal: "normal", anomalous: "fraud", named: true });
  assert.deepEqual(anomalySides(envOf(FIX.cases.find((x) => x.id === "anomaly-reconstruction"))),
    { normal: ANOMALY_SIDES[0], anomalous: ANOMALY_SIDES[1], named: false });
});

// ---- the weights axis (D4 §6.4) and the weights read that takes it
const pairs = (cols) => cols.map(({ id, name }) => ({ id, name }));

test("the weights axis is the attribution's columns, by id: classes, binary sides and labels by name in the lane's order; bins, groups and steps in the attribution's", async () => {
  for (const c of FIX.cases) {
    const env = envOf(c), vk = env.lane.viewKey, mod = await R.moduleFor("task", vk), doc = attributionFor(c);
    const idOf = new Map(doc.axis.columns.map((x) => [x.name, x.id])), cs = env.output.classes;
    const named = { classes: cs, labels: cs, ordinal: cs, binary: cs && [cs.find((x) => x !== env.output.positive), env.output.positive] }[vk];
    const want = named ? named.map((n) => ({ id: idOf.get(n), name: n })) : pairs(doc.axis.columns);
    const axis = mod.weightsAxis(doc, env);
    assert.deepEqual(pairs(axis.columns), want, c.id);
    axis.columns.forEach((x) => assert.ok(doc.cells[x.id], `${c.id}: column ${x.name} indexes a cell of the attribution`));
    assert.deepEqual(mod.weightsAxis(W.attributionOf(doc), env).columns, axis.columns, `${c.id}: the page's one form of the attribution reads the same`);
    assert.ok(axis.columns.every((x) => ["neutral", "alert", "plain"].includes(x.role)), `${c.id}: every column in a role`);
  }
  const binary = FIX.cases.find((x) => x.id === "binary"), regression = FIX.cases.find((x) => x.id === "regression");
  assert.deepEqual(pairs((await R.moduleFor("task", "binary")).weightsAxis(attributionFor(binary), envOf(binary)).columns),
    [{ id: "k-ok", name: "ok" }, { id: "k-delayed", name: "delayed" }], "binary: [negative, positive], though the head has one logit");
  assert.deepEqual(pairs((await R.moduleFor("task", "value")).weightsAxis(attributionFor(regression), envOf(regression)).columns),
    ["q1", "q2", "q3", "q4", "q5"].map((name, i) => ({ id: `b${i}`, name })), "value: the target's quintile bins");
  const multi = FIX.cases.find((x) => x.id === "multiclass"), partial = attributionFor(multi);
  partial.axis.columns = partial.axis.columns.filter((x) => x.name !== "gust"); delete partial.cells["k-gust"];
  assert.deepEqual(pairs((await R.moduleFor("task", "classes")).weightsAxis(partial, envOf(multi)).columns),
    [{ id: "k-calm", name: "calm" }, { id: null, name: "gust" }, { id: "k-storm", name: "storm" }], "a class the sample has no row of keeps its column");
});

test("the weights read on real rows has the lane's axis for columns, each read from its cell; without one, the head's outputs", async () => {
  const families = [{ label: "a", members: ["a"], idx: [0], text: "" }, { label: "b, zone", members: ["b", "zone"], idx: [1, 2], text: "" }];
  const palette = { pos: "#10a080", neg: "#e05040" };
  const map = (c, head, columns) => {
    const env = envOf(c), host = document.createElement("div"), graph = { inputs: { names: INPUTS.slice() }, headLabels: () => head.slice() };
    W.drawWeightMap(host, { graph, model: null, families, roles: env.roles, attribution: W.attributionOf(attributionFor(c)), columns, palette });
    const cells = host.querySelectorAll(".c").filter((x) => !x.classList.contains("sub"));
    return { heads: host.querySelectorAll(".ch").map(txt), cells: cells.length, none: cells.filter((x) => x.classList.contains("none")).length };
  };
  for (const [id, head] of [["binary", ["delayed"]], ["regression", ["value"]], ["forecast", ["value_h1", "value_h2", "value_h3"]],
    ["anomaly-svdd", ["d0", "d1"]], ["reconstruction", INPUTS]]) {
    const c = FIX.cases.find((x) => x.id === id), env = envOf(c), mod = await R.moduleFor("task", env.lane.viewKey);
    const axis = mod.weightsAxis(W.attributionOf(attributionFor(c)), env), got = map(c, head, axis.columns);
    assert.deepEqual(got.heads, axis.columns.map((x) => x.name), `${id}: the axis's columns`);
    assert.equal(got.cells, families.length * axis.columns.length, id);
    assert.equal(got.none, 0, `${id}: every column read from its cell`);
    const bare = map(c, head, null);
    assert.deepEqual(bare.heads, head, `${id}: without an axis, the head's outputs`);
  }
  const binary = FIX.cases.find((x) => x.id === "binary");
  assert.equal(map(binary, ["delayed"], null).none, 0, "the head's outputs are read by name when the attribution has them");
  assert.equal(map(FIX.cases.find((x) => x.id === "regression"), ["value"], null).none, families.length, "and are empty when it has not");
});
