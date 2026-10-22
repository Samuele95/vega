// node --test: the sequence renderers (net/renderers/{conv, recurrent, attention}.js, over
// net/seq.js and net/seq-tokens.js) and the window shape (views/shapes/window.js) on the
// inspection zoo's dumped payloads (I10: vega-forge/scripts/tests/test_inspect_zoo.py --dump, one
// network.json / explain.json pair per case; every window case, the ft-transformer rows and the
// window extras). The dump is read from $VEGA_ZOO_DUMP when it is set, else made once by the zoo
// itself ($VEGA_PYTHON, else $PYTHON, else python3: one with torch) into a cache under the temporary directory
// keyed by the digest of the zoo and of the backend templates it renders.
//
// What is held: every sequence stage kind has its module, with the whole renderer interface; every
// segment of every case lays out inside its box and draws there — at rest, at the "every weight"
// level and with the case's own answer — and its caption resolves every word; each kind reads the
// answer it should (a conv's map is the output's steps × channels, a recurrent layer's state norms
// one list per direction over its steps, an attention map per head that is a distribution, the
// pooling weights over the steps adding up to one, an encoder's per-layer maps and token norms, the
// tokens' norms with [cls] first); the weights are split as the backend publishes them (a gate's
// rows of W_ih and W_hh, the kernel's C′ × C·k, q / k / v / o of in_proj and out_proj); the window
// shape's editor, its sources (one window end per series a request) and what it says of an answer.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync, mkdtempSync, renameSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { install } from "./dom-shim.js";

install();
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
const { StageGraph, Focus, tensorValues } = await import("../js/net/graph.js");
const { load, MEMBERS } = await import("../js/net/renderers/index.js");
const { rendererFor, fitRow, spreadRow } = await import("../js/net/canvas.js");
const { familiesOf } = await import("../js/net/families.js");
const { configureText, missingKeys, unresolved } = await import("../js/core/text.js");
const conv = (await import("../js/net/renderers/conv.js")).default;
const { convShape, kernelMatrix } = await import("../js/net/renderers/conv.js");
const { shownSteps, handedOn, recurrentShape, gateBlocks, hiddenNorms } = await import("../js/net/renderers/recurrent.js");
const { projections, poolWeights, downsampleList } = await import("../js/net/renderers/attention.js");
const { encoderLayers, shownTokens, received, tokenNames } = await import("../js/net/seq-tokens.js");
const seq = await import("../js/net/seq.js");
const win = await import("../js/views/shapes/window.js");
const { SHAPE_MEMBERS, missingMembers } = await import("../js/views/registry.js");
const { setState, S } = await import("../js/core/state.js");

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const ROOT = here("../../../../");
const ZOO = join(ROOT, "vega-forge/scripts/tests/test_inspect_zoo.py");
const TEMPLATES = join(ROOT, "vega-forge/scripts/backends/pytorch/templates");
const CASES = "*/window/*,ft-transformer/row/*,extra/*window*";
const SEQ_KINDS = ["conv", "pool1d", "recurrent", "attention", "attention_pool", "transformer", "tokenizer"];
// The renderer modules by kind, for the tests that call a kind's acts directly.
const StageGraphRenderer = Object.freeze(Object.fromEntries((await load(SEQ_KINDS)).entries()));
const COLORS = Object.freeze({ acc: "a", neg: "n", mut: "m", fg: "f", fg2: "f2", muted: "mu", alert: "al", bg: "b", line: "l" });

// ---- the dump
function filesUnder(dir) {
  return readdirSync(dir).sort().flatMap((n) => { const p = join(dir, n); return statSync(p).isDirectory() ? (n === "__pycache__" ? [] : filesUnder(p)) : [p]; });
}

function zooDump() {
  if (process.env.VEGA_ZOO_DUMP) return existsSync(join(process.env.VEGA_ZOO_DUMP, "index.json")) ? { dir: process.env.VEGA_ZOO_DUMP } : { skip: "VEGA_ZOO_DUMP holds no index.json" };
  const h = createHash("sha256");
  for (const f of [ZOO, ...filesUnder(TEMPLATES)]) h.update(f).update(readFileSync(f));
  const cache = join(tmpdir(), `vega-zoo-w10-${h.update(CASES).digest("hex").slice(0, 16)}`);
  if (existsSync(join(cache, "index.json"))) return { dir: cache };
  const work = mkdtempSync(join(tmpdir(), "vega-zoo-w10-make-"));
  const run = spawnSync(process.env.VEGA_PYTHON || process.env.PYTHON || "python3", [ZOO, "--cases", CASES, "--dump", work, "-q", "-x"],
    { encoding: "utf8", timeout: 900000, env: { ...process.env, CUDA_VISIBLE_DEVICES: "" } });
  if (run.status !== 0 || !existsSync(join(work, "index.json"))) {
    rmSync(work, { recursive: true, force: true });
    return { skip: `the zoo did not dump (exit ${run.status}): ${String(run.stderr || run.stdout || run.error).slice(-400)}` };
  }
  try { renameSync(work, cache); } catch { return { dir: work }; }
  return { dir: cache };
}

const DUMP = zooDump();
const opt = DUMP.skip ? { skip: DUMP.skip } : {};
const index = DUMP.dir ? JSON.parse(readFileSync(join(DUMP.dir, "index.json"), "utf8")).cases : [];
const pair = (c) => ({ network: JSON.parse(readFileSync(join(DUMP.dir, c.files.network), "utf8")), ex: JSON.parse(readFileSync(join(DUMP.dir, c.files.explain), "utf8")) });
const CASE_LIST = index.map((c) => ({ ...c, ...pair(c) }));
const one = (pred) => CASE_LIST.find(pred);
const variantsOf = (g) => [`${g.viewKey}+${g.sample.shape}`, g.viewKey, g.sample.shape];

// The drawing context of a graph on a canvas W wide (what NetCanvas hands its renderers).
async function canvasOf(g, W, mode) {
  const renderers = await load(g.kinds()), narrow = W < 700, families = familiesOf({ names: g.inputs.names, units: g.featureUnits() }).families;
  const ctx = { W, H: 400, narrow, mode, graph: g, env: { families }, K: mode === "detail" ? Infinity : 14, colors: COLORS, boxW: narrow ? 64 : 88,
    collapse: false, compact: false, top: 18, bottom: 36, ordinal: new Map(), waist: null, rendererFor: (s) => rendererFor(renderers, g, s) };
  const pad = narrow ? 16 : 28, gap = narrow ? 18 : 30, f = fitRow(g.segments(), ctx, W - 2 * pad, gap);
  const c = { ...ctx, ...f.fit }, boxes = spreadRow(f.sizes, pad, W - 2 * pad, gap).map((b) => ({ ...b, y: 18, h: 346 }));
  return { c, boxes, segs: g.segments() };
}

// A 2-D context that records where every mark lands.
function recorder() {
  const marks = [];
  const mark = (x, y, w = 0, h = 0) => marks.push({ x, y, w, h });
  const g = { marks, globalAlpha: 1, measureText: (s) => ({ width: 6 * String(s).length }),
    fillRect: (x, y, w, h) => mark(x, y, w, h), strokeRect: (x, y, w, h) => mark(x, y, w, h), arc: (x, y) => mark(x, y),
    moveTo: (x, y) => mark(x, y), lineTo: (x, y) => mark(x, y), bezierCurveTo: (a, b, c, d, x, y) => mark(x, y), fillText: (s, x, y) => mark(x, y) };
  return new Proxy(g, { get: (o, k) => (k in o ? o[k] : () => {}), set: (o, k, v) => { o[k] = v; return true; } });
}

const isSeq = (g, s) => s.kind === "box" && SEQ_KINDS.includes(g.byId(s.stage).kind);

// ---- the interface
test("every sequence stage kind has its module, each with the whole renderer interface", opt, async () => {
  const map = await load(SEQ_KINDS);
  for (const kind of SEQ_KINDS) {
    const r = map.get(kind);
    for (const m of MEMBERS) assert.ok(m in r, `${kind}: ${m}`);
    assert.ok(r.kinds.includes(kind) && r.detail === true, kind);
  }
  const seen = new Set(CASE_LIST.flatMap((c) => StageGraph.from(c.network).stages().map((n) => n.kind)));
  for (const kind of SEQ_KINDS) assert.ok(seen.has(kind), `the dump has a ${kind} stage`);
  assert.ok(CASE_LIST.length >= 60, `${CASE_LIST.length} cases dumped`);
});

// ---- drawing
test("every segment of every case lays out in its box and draws there: at rest, every weight, one answer", opt, async () => {
  configureText({});
  for (const c of CASE_LIST) {
    const g = StageGraph.from(c.network);
    configureText({ variants: variantsOf(g) });
    for (const mode of ["simple", "detail"]) {
      const { c: ctx, boxes, segs } = await canvasOf(g, 1100, mode);
      for (const answered of [false, true]) {
        const acts = answered ? segs.map((s) => ctx.rendererFor(s).acts(s, c.ex, g, ctx)) : null;
        segs.forEach((s, i) => {
          if (!isSeq(g, s)) return;
          const r = ctx.rendererFor(s), L = r.layout(s, boxes[i], { ...ctx, acts: acts ? acts[i] : null }), rec = recorder();
          r.draw(rec, s, L, { phase: 1, focus: new Focus(null), hover: null, acts, mode, g: rec, n: segs.length, i, ctx });
          const b = boxes[i], where = `${c.slug} ${s.stage} ${mode}${answered ? " answered" : ""}`;
          const inside = rec.marks.filter((m) => m.x >= b.x - 1 && m.x <= b.x + b.w + 1 && m.y >= b.y - 30 && m.y <= b.y + b.h + 30);
          assert.ok(inside.length >= 4, `${where}: ${inside.length} marks in its box`);
          for (const m of rec.marks) assert.ok(m.x >= b.x - 40 && m.x <= b.x + b.w + 40 && Number.isFinite(m.y), `${where}: a mark at ${m.x} outside [${b.x}, ${b.x + b.w}]`);
          for (const p of [...L.in, ...L.out]) assert.ok(p.x >= b.x - 2 && p.x <= b.x + b.w + 2, `${where}: a port outside its box`);
          if (answered) assert.notEqual(acts[i], null, `${where}: the answer says something of it`);
        });
      }
    }
  }
});

test("every caption of a sequence stage resolves its words, with and without a point hit", opt, async () => {
  for (const c of CASE_LIST) {
    const g = StageGraph.from(c.network);
    configureText({ variants: variantsOf(g) });
    const { c: ctx, boxes, segs } = await canvasOf(g, 1100, "simple");
    const acts = segs.map((s) => ctx.rendererFor(s).acts(s, c.ex, g, ctx));
    segs.forEach((s, i) => {
      if (!isSeq(g, s)) return;
      const r = ctx.rendererFor(s), L = r.layout(s, boxes[i], { ...ctx, acts: acts[i] });
      const hits = [null, { stage: s.stage }, ...[L.in[0], L.out[0]].map((p) => r.hit(s, L, p.x, p.y)),
        ...(L.rect ? [r.hit(s, L, L.rect.x + 2, L.rect.y + 2)] : []), ...(L.cells ? [r.hit(s, L, L.cells[0].x + 2, L.cells[0].y + 2)] : []),
        ...(L.rects ? [r.hit(s, L, L.rects[0].x + 2, L.rects[0].y + 2)] : []), ...(L.ys && L.xt ? [r.hit(s, L, L.xt + 2, L.ys[1] ?? L.ys[0])] : [])];
      for (const h of hits) {
        const node = r.caption(s, h, ctx), formula = node.querySelector(".formula");      // the backend's formula keeps its braces
        const text = node.textContent.replace(formula ? formula.textContent : "", "");
        assert.ok(text.length > 10 && !/\{[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?(?::[a-z]+)?\}/.test(text), `${c.slug} ${s.stage}: ${text}`);
      }
    });
  }
  assert.deepEqual(missingKeys(), []);
  assert.deepEqual(unresolved(), []);
});

// ---- what each kind reads of an answer
test("a convolution reads its output map (steps × channels), a max pooling the steps each one won", opt, () => {
  const c = one((x) => x.family === "cnn1d" && x.task.startsWith("forecast")), g = StageGraph.from(c.network);
  const convs = g.stages().filter((n) => n.kind === "conv"), pools = g.stages().filter((n) => n.kind === "pool1d");
  assert.ok(convs.length && pools.length);
  for (const n of [...convs, ...pools]) {
    const a = conv.acts({ stage: n.id }, c.ex), s = convShape(n);
    assert.deepEqual([a.map.rows, a.map.cols], [s.Lo, s.Co], n.id);
    assert.equal(a.mx, seq.maxAbs(a.map));
  }
  const maxPool = pools.find((n) => n.payload.kind === "max");
  if (maxPool) assert.equal(conv.acts({ stage: maxPool.id }, c.ex).winners.length, convShape(maxPool).L);
  const k = convs[0], M = kernelMatrix(g, k), s = convShape(k), t = c.network.tensors[k.payload.kernel];
  assert.deepEqual([M.rows, M.cols], [s.Co, s.C * s.k]);
  assert.deepEqual(t.shape, [s.Co, s.C, s.k]);
  assert.equal(M.v[1], tensorValues(t)[1]);
  const withheld = { ...k, payload: { ...k.payload, kernel: null } }, P = kernelMatrix(g, withheld);
  assert.deepEqual([P.rows, P.cols], [s.Co, s.C], "without the weights: the channel-pair norms");
  assert.equal(conv.acts({ stage: k.id }, { schema: "vega/explanation@2", nodes: {} }), null);
});

test("a recurrent layer reads one list of state norms per direction over its steps, and the steps tipped", opt, () => {
  for (const c of CASE_LIST) {
    const g = StageGraph.from(c.network);
    for (const n of g.stages().filter((x) => x.kind === "recurrent")) {
      const shape = recurrentShape(n), acts = (StageGraphRenderer.recurrent).acts({ stage: n.id }, c.ex, g);
      assert.equal(acts.raw.length, shape.dirs, `${c.slug} ${n.id}`);
      for (const d of acts.raw) assert.equal(d.length, shape.L);
      assert.equal(shape.dirs, c.family.startsWith("bi-") ? 2 : 1, c.slug);
      if (acts.perStep) assert.equal(acts.perStep.length, shape.L);
      const blocks = gateBlocks(g, n), H = shape.H;
      assert.deepEqual(blocks.map((b) => b.gate), shape.gates);
      const wih = c.network.tensors[n.payload.per_layer[0].w_ih];
      const k = blocks.length - 1;
      assert.deepEqual([blocks[k].ih.rows, blocks[k].ih.cols], [H, wih.shape[1]]);
      assert.equal(blocks[k].ih.v[0], tensorValues(wih)[k * H * wih.shape[1]], "the last gate's rows of W_ih");
      assert.equal(blocks[0].hh.cols, H);
    }
  }
  assert.deepEqual(shownSteps(12), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  assert.deepEqual(shownSteps(24), [0, 1, 2, 3, 4, 5, null, 18, 19, 20, 21, 22, 23]);
  assert.deepEqual(handedOn({ ret: "last", dirs: 2, L: 12 }), [11, 0], "a backward row's last state is its first step's");
  assert.deepEqual(handedOn({ ret: "mean", dirs: 1, L: 12 }), [null]);
});

test("attention reads one map per head, each row a distribution; pooling reads weights over the steps that add up to one", opt, () => {
  let maps = 0, pools = 0;
  for (const c of CASE_LIST) {
    const g = StageGraph.from(c.network);
    for (const n of g.stages()) {
      if (n.kind === "attention") {
        const a = StageGraphRenderer.attention.acts({ stage: n.id }, c.ex, g), heads = n.payload.heads, L = n.in_shape[0];
        assert.equal(a.maps.length, heads);
        for (const M of a.maps) {
          assert.deepEqual([M.rows, M.cols], [L, L]);
          for (let r = 0; r < M.rows; r++) assert.ok(Math.abs(Array.from(M.v.subarray(r * L, (r + 1) * L)).reduce((s, v) => s + v, 0) - 1) < 1e-5);
        }
        const P = projections(g, n.payload);
        assert.deepEqual(P.map((p) => [p.name, p.M.rows, p.M.cols]), [["q", n.payload.dim, n.payload.dim], ["k", n.payload.dim, n.payload.dim], ["v", n.payload.dim, n.payload.dim], ["o", n.payload.dim, n.payload.dim]]);
        maps++;
      }
      if (n.kind === "attention_pool") {
        const w = poolWeights(c.ex, n.id);
        assert.equal(w.length, n.in_shape[0]);
        assert.ok(Math.abs(w.reduce((s, v) => s + v, 0) - 1) < 1e-5, c.slug);
        pools++;
      }
    }
  }
  assert.ok(maps > 0 && pools > 0, `${maps} attention, ${pools} pooling stages read`);
  assert.deepEqual(downsampleList([1, 2, 3, 4], 2), [1.5, 3.5]);
  const silent = StageGraphRenderer.attention.acts({ stage: "B4" }, { schema: "vega/explanation@2", nodes: { B4: { level: "full", extra: {} } } },
    StageGraph.from(one((x) => x.slug === "extra__gru__window__forecast-attention-pool").network));
  assert.deepEqual(silent, { maps: null }, "an answer without the map: said, not drawn");
});

test("an encoder reads each layer's maps, its tokens' norms and what the tokens receive; a tokenizer its tokens, [cls] first", opt, () => {
  let encoders = 0;
  for (const c of CASE_LIST) {
    const g = StageGraph.from(c.network);
    for (const n of g.stages().filter((x) => x.kind === "transformer")) {
      const layers = encoderLayers(g, n), a = StageGraphRenderer.attention.acts({ stage: n.id }, c.ex, g);
      assert.equal(a.layers.length, layers.length);
      for (const l of a.layers) {
        assert.equal(l.maps.length, n.config.heads);
        assert.equal(l.tokens.length, n.in_shape[0]);
        assert.equal(l.query.length, n.in_shape[0]);
        assert.ok(Math.abs(l.query.reduce((s, v) => s + v, 0) - 1) < 1e-4, "what the tokens receive adds up to one");
      }
      encoders++;
    }
    for (const n of g.stages().filter((x) => x.kind === "tokenizer")) {
      const a = StageGraphRenderer.attention.acts({ stage: n.id }, c.ex, g), names = tokenNames(g, n);
      assert.equal(a.tokens.length, n.out_shape[0]);
      assert.equal(names[0], "[cls]");
      assert.equal(a.raw[0], c.network.nodes.find((x) => x.id === n.id).params.cls_norm, "the [cls] token is the first");
    }
  }
  assert.ok(encoders > 0);
  const cls = { schema: "vega/explanation@2", nodes: { L: { level: "full", extra: { cls_row: [[0.5, 0.5, 0], [0.1, 0.1, 0.8]] } } } };
  assert.deepEqual(received(cls, "L", null), [0.3, 0.3, 0.4]);
  assert.deepEqual(shownTokens(40).length, 16);
  assert.deepEqual([shownTokens(40)[0], shownTokens(40)[15]], [0, 39]);
});

// ---- the helpers
test("matrices read from every form an explanation or the network view carries", () => {
  const H = seq.heatMatrix({ rows: 2, cols: 2, values: [[1, -2], [3, 4]] });
  assert.deepEqual([H.rows, H.cols, Array.from(H.v)], [2, 2, [1, -2, 3, 4]]);
  assert.deepEqual(Array.from(seq.transpose(H).v), [1, 3, -2, 4]);
  assert.deepEqual(Array.from(seq.downsample(seq.matrix(2, 4, Float32Array.from([1, 3, 5, 7, 1, 3, 5, 7])), 1, 2).v), [2, 6]);
  const b64 = Buffer.from(Float32Array.from([1, 2, 3, 4, 5, 6]).buffer).toString("base64");
  const T = seq.tensorMatrix({ shape: [2, 1, 3], encoding: "base64-le", data: b64 });
  assert.deepEqual([T.rows, T.cols], [2, 3]);
  assert.deepEqual([seq.tensorMatrix({ shape: [2, 1, 3], encoding: "base64-le", data: b64 }, 1).rows], [2]);
  const ex = { nodes: { a: { level: "full", exact: { out: { shape: [3, 2], encoding: "json", data: [1, 2, 3, 4, 5, 6] } } },
    b: { level: "omitted" }, c: { level: "full", out: { form: "sequence", heat: { values: [[1], [2]] } } },
    d: { level: "full", exact: { alpha: { shape: [2, 1, 2], encoding: "json", data: [0.5, 0.5, 0.25, 0.75] } } } } };
  assert.deepEqual([seq.seqOut(ex, "a").rows, seq.seqOut(ex, "a").cols], [3, 2], "the exact tensor first");
  assert.equal(seq.seqOut(ex, "b"), null, "an omitted node says nothing");
  assert.deepEqual(Array.from(seq.seqOut(ex, "c").v), [1, 2]);
  assert.deepEqual(seq.attentionMaps(ex, "d").map((M) => Array.from(M.v)), [[0.5, 0.5], [0.25, 0.75]]);
  const norms = hiddenNorms({ nodes: { r: { level: "full", exact: { hidden: { shape: [2, 2, 2], encoding: "json", data: [3, 4, 0, 1, 6, 8, 0, 2] } } } } }, "r", { dirs: 2 });
  assert.deepEqual(norms, [[5, 1], [10, 2]], "the exact hidden states, one list per direction");
  assert.deepEqual(seq.normalised([2, -4, null]), [0.5, -1, null]);
});

// ---- the window shape
const COLS = (g) => [{ name: "ts", dtype: "string" }, ...g.inputs.numeric.filter((n) => n !== g.inputs.pastTarget).map((name) => ({ name, dtype: "float" })),
  ...g.inputs.nominal.map((c) => ({ name: c.name, dtype: "string" }))];

test("the window shape owes every member of the registry", () => {
  assert.deepEqual(missingMembers(win, SHAPE_MEMBERS), []);
});

test("the editor shows the series, the window marked, and reads back {rows: [last], history}", opt, () => {
  const c = one((x) => x.family === "gru" && x.task.startsWith("forecast")), g = StageGraph.from(c.network);
  configureText({ variants: variantsOf(g) });
  setState({ graph: g });
  const schema = { input: COLS(g) }, env = { graph: g, story: null }, host = document.createElement("div");
  const L = g.sample.length, rows = Array.from({ length: L + 3 }, (_, k) => Object.fromEntries(schema.input.map((col) => [col.name, col.dtype === "float" ? k + 0.5 : `v${k}`])));
  win.editor.mount(host, schema, env);
  assert.match(host.textContent, /Pick an example/);
  win.editor.load({ rows: rows.slice(-1), history: rows.slice(0, -1), name: "n", text: "t" }, env);
  assert.equal(host.querySelectorAll("tr.seq-in").length, L);
  assert.equal(host.querySelectorAll("tr.seq-ctx").length, 3);
  assert.equal(host.querySelector("tr.seq-in").children[0].textContent, c.network.inputs.axis_labels.step[0]);
  const back = win.editor.read();
  assert.deepEqual(back.rows, rows.slice(-1));
  assert.deepEqual(back.history, rows.slice(0, -1));
  const input = host.querySelector("tr.seq-in").querySelector("input");
  input.value = ""; assert.equal(win.editor.read().history[3][input.getAttribute("data-col")], null, "an empty cell is null");
  assert.deepEqual(win.editor.describe({ name: "a", text: "b" }), { title: "a", text: "b" });
  assert.deepEqual(missingKeys(), []);
});

test("many at once: a run holds one window end per series beside its context; a row opens with its context", () => {
  const item = (series, k, truth) => ({ rows: [{ s: series, k }], history: [{ s: series, k: k - 2 }, { s: series, k: k - 1 }], truth });
  const examples = { demo: { items: [item("a", 5, "x"), item("a", 9, "y"), item("b", 4, "z"), item("c", 3, "x")] } };
  const by = win.demoWindows(examples, "s");
  assert.deepEqual(win.roundOf(by, 0, "s").map((w) => w.row.k), [5, 4, 3]);
  assert.deepEqual(win.roundOf(by, 1, "s").map((w) => w.row.k), [9, 4, 3], "the next run takes each series' next window");
  assert.deepEqual(win.roundOf(win.demoWindows(examples, null), 1, null).map((w) => w.row.k), [9], "series unknown: one window a run");
  const batch = win.batchOf(win.roundOf(by, 0, "s"));
  assert.deepEqual(batch.rows.map((r) => r.k), [5, 4, 3]);
  assert.deepEqual(batch.history.map((r) => `${r.s}${r.k}`), ["a3", "a4", "b2", "b3", "c1", "c2"]);
  assert.deepEqual(batch.truth, ["x", "z", "x"]);
  assert.deepEqual(win.rowsOf({ row: batch.rows[1] }).map((r) => `${r.s}${r.k}`), ["b2", "b3", "b4"]);
  assert.deepEqual(win.rowsOf({ row: { s: "q" } }), [{ s: "q" }], "a row from elsewhere has no context");
  const shipped = { rows: [{ s: "a", k: 1 }, { s: "b", k: 1 }, { s: "a", k: 2 }, { s: "a", k: 3 }, { s: "a", k: 4 }], truth: null, history: [{ s: "a", k: 0 }] };
  win.sources({ test: shipped }, {}).find((x) => x.id === "test").run();
  assert.deepEqual(win.rowsOf({ row: shipped.rows[4] }).map((r) => `${r.s}${r.k}`), ["a0", "a1", "b1", "a2", "a3", "a4"], "no record: every row before it");
  setState({ views: { ...S.views, "window-sample": { asked: true, sample: { series_by: "s", history_rows: 2 } } } });
  win.sources({ test: shipped }, {}).find((x) => x.id === "test").run();
  assert.deepEqual(win.rowsOf({ row: shipped.rows[4] }).map((r) => `${r.s}${r.k}`), ["a2", "a3", "a4"], "its series' last history_rows");
  const forecast = win.batchOf([{ row: { k: 1 }, history: [], truth: [1, 2, 3] }]);
  assert.equal(forecast.truth, null, "a forecast's truth (the steps ahead) is not the monitor's one truth per row");
  assert.deepEqual(win.batchOf([{ row: {}, history: [], truth: ["l1", "l2"] }]).truth, [["l1", "l2"]], "a label set travels");
});

test("what tipped a window's answer: the step × feature heat adds up to the attribution's totals", opt, () => {
  for (const c of CASE_LIST.filter((x) => x.shape === "window")) {
    const g = StageGraph.from(c.network), a = c.ex.attribution, wa = win.windowAttribution(c.ex);
    configureText({ variants: variantsOf(g) });
    assert.equal(wa.M.rows, a.steps, c.slug);
    assert.deepEqual(wa.names, [...a.features.names, ...(a.nominal ? a.nominal.names : [])]);
    a.per_feature.forEach((v, j) => assert.ok(Math.abs(wa.perFeature[j] - v) < 1e-5 * Math.max(1, Math.abs(v)), `${c.slug} ${wa.names[j]}`));
    const total = wa.perFeature.reduce((s, v) => s + v, 0);
    assert.ok(Math.abs(total - a.total) < 1e-4 * Math.max(1, Math.abs(a.total)), `${c.slug}: the heat adds up to the total`);
    const node = win.attribution(c.ex, { graph: g, story: null });
    assert.ok(node.querySelectorAll("canvas").length >= 1 && node.querySelectorAll(".bar").length >= 1, c.slug);
  }
  assert.deepEqual(missingKeys(), []);
});

test("the stepper's input step names the cells furthest from zero, nominal inputs apart", opt, () => {
  for (const c of CASE_LIST.filter((x) => x.shape === "window")) {
    const g = StageGraph.from(c.network), x = win.shapedWindow(c.ex), nominal = g.inputs.nominal.map((n) => n.name);
    configureText({ variants: variantsOf(g) });
    assert.deepEqual([x.M.rows, x.names], [g.sample.length, c.ex.inputs.order], c.slug);
    const top = win.topCells(x, g.inputs.steps, nominal);
    const all = x.names.flatMap((n, j) => (nominal.includes(n) ? [] : Array.from({ length: x.M.rows }, (_, r) => Math.abs(seq.at(x.M, r, j)))));
    assert.equal(Math.abs(top[0].v), Math.max(...all));
    assert.ok(top.every((q) => !nominal.includes(q.feature) && g.inputs.steps.includes(q.step)));
    const text = win.inputStep(c.ex, null, { graph: g }).textContent;
    assert.ok(text.includes(`${top[0].feature} @ ${top[0].step}`), `${c.slug}: ${text.slice(0, 120)}`);
    if (nominal.length) assert.ok(text.includes(nominal[0]));
  }
  assert.deepEqual(missingKeys(), []);
  assert.deepEqual(unresolved(), []);
});
