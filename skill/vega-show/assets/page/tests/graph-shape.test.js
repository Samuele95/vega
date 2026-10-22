// node --test: the graph area (BUILD-PLAN W11, D4 §6.5 and §7.2, R-A3) on the inspection zoo's real
// graph payloads (fixtures/graph-shape: network-view@2 / explanation@2 pairs dumped by I10's
// test_inspect_zoo.py through make_graph_fixtures.py) — every graph convolution at node level (gcn,
// sage with its root, gat with its heads), a graph family with no convolution, and graph level with a
// sum and a mean readout.
//
//   the renderer (js/net/renderers/graph.js): the registry loads it for graph_conv and graph_pool;
//     it reads each convolution's messages as the answer states them (the ring is the eight messages
//     from other nodes that carried the most, the node's own term apart, gat's α averaged over its
//     heads), each readout's node shares, each convolution's matrices (decoded to their shapes, read
//     down to thumbnails that are the blocks' means); it draws inside its box at both levels of
//     detail, before and after an answer; a pointer on a ring node names the message, and at the
//     every-weight level the matrix under it (the ring is not drawn there).
//   the shape (js/views/shapes/graph.js): the registry's members; the editor sends {rows, at} (a node
//     alone; a whole graph, the chosen node edited); a row of many at once opens its whole graph,
//     the reader's CSV included;
//     the sources batch the demonstration items (each row its graph's truth at graph level); what tipped
//     draws the neighbourhood the answer lists, the rows past it and a reported gap apart from the
//     attributions (A31); the input step names the node's largest shaped numbers, its hops and the
//     linking columns, with the env the real stepper builds (no schema in it).
// Every word is a key of the strings table, and no placeholder is left unresolved.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { install } from "./dom-shim.js";

install();
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#101010" });
const { StageGraph } = await import("../js/net/graph.js");
const { RENDERER_OF, KIND_OF } = await import("../js/net/kinds.js");
const { load } = await import("../js/net/renderers/index.js");
const G = await import("../js/net/renderers/graph.js");
const S = await import("../js/views/shapes/graph.js");
const R = await import("../js/views/registry.js");
const { configureText, missingKeys, unresolved } = await import("../js/core/text.js");
const { Lexicon } = await import("../js/core/lexicon.js");
const { Focus } = await import("../js/net/graph.js");
const { setState } = await import("../js/core/state.js");
const { Stepper } = await import("../js/how/stepper.js");
const { indexNamed } = await import("../js/core/node-names.js");

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const INDEX = JSON.parse(readFileSync(here("./fixtures/graph-shape/index.json"), "utf8"));
const CASES = INDEX.cases.map((c) => c.slug);
const read = (slug, file) => JSON.parse(readFileSync(here(`./fixtures/graph-shape/${slug}/${file}`), "utf8"));
const FIX = Object.fromEntries(CASES.map((slug) => [slug, { case: read(slug, "case.json"), network: read(slug, "network.json"), ex: read(slug, "explain.json") }]));
const NODE = CASES.filter((s) => FIX[s].case.level === "node"), GRAPH = CASES.filter((s) => FIX[s].case.level === "graph");
const graphOf = (slug) => StageGraph.from(FIX[slug].network);
const convs = (slug) => graphOf(slug).stages().filter((n) => n.kind === "graph_conv");
const txt = (n) => (n === null || n === undefined ? "" : typeof n === "string" ? n : n.textContent).replace(/\s+/g, " ").trim();
const fresh = (lane) => { install(); setState({ views: {} }); configureText({ lex: new Lexicon(null, null), lane }); };
const clean = (what) => { assert.deepEqual(missingKeys(), [], `${what}: every word a key`); assert.deepEqual(unresolved(), [], `${what}: every placeholder resolved`); };

test("the fixtures are the zoo's graph cases: every convolution at node level, and both readouts at graph level", () => {
  assert.equal(INDEX.schema, "vega/inspect-zoo@1");
  const types = new Set(CASES.flatMap((s) => graphOf(s).stages().map((n) => n.type)));
  for (const t of ["gcn_conv", "sage_conv", "gat_conv", "gin_conv", "graph_pool", "edge_dropout"]) assert.ok(types.has(t), t);
  assert.ok(NODE.length >= 4 && GRAPH.length >= 2);
  const readouts = GRAPH.flatMap((s) => graphOf(s).stages().filter((n) => n.type === "graph_pool").map((n) => n.payload.readout));
  assert.deepEqual(new Set(readouts), new Set(["sum", "mean"]));
  for (const s of CASES) assert.equal(FIX[s].ex.sample.graph.level, FIX[s].case.level, s);
});

test("the renderer registry draws graph_conv and graph_pool with renderers/graph.js", async () => {
  for (const type of ["gcn_conv", "sage_conv", "gat_conv", "gin_conv", "graph_pool"]) assert.equal(RENDERER_OF[KIND_OF[type]], "graph.js", type);
  assert.deepEqual(G.default.kinds, ["graph_conv", "graph_pool"]);
  assert.equal(G.default.detail, true);
  for (const s of CASES) {
    const kinds = graphOf(s).kinds(), got = await load(kinds);
    for (const k of ["graph_conv", "graph_pool"]) if (kinds.has(k)) assert.equal(got.get(k), G.default, `${s} ${k}`);
  }
});

test("a convolution's answer: the eight strongest messages from other nodes on the ring, its own term apart", () => {
  for (const s of CASES) for (const n of convs(s)) {
    const raw = FIX[s].ex.nodes[n.id].extra.messages, st = G.convState(FIX[s].ex, n.id);
    assert.ok(Array.isArray(raw) && raw.length, `${s} ${n.id}: the answer carries its messages`);
    const others = raw.filter((m) => m.src !== m.dst && !m.root).map((m) => Math.abs(m.relevance)).sort((a, b) => b - a);
    assert.deepEqual(st.ring.map((m) => Math.abs(m.rel)), others.slice(0, G.RING_TRY), `${s} ${n.id}: ring`);
    assert.ok(st.ring.every((m) => !m.self && !m.root));
    const mine = raw.filter((m) => m.src === m.dst || m.root);
    assert.ok(Math.abs((st.own ? st.own.rel : 0) - mine.reduce((a, m) => a + m.relevance, 0)) < 1e-9, `${s} ${n.id}: own term`);
    assert.equal(st.n, raw.length); assert.equal(st.level, FIX[s].case.level);
    if (n.type === "sage_conv") assert.ok(st.own && st.own.root, `${s}: sage's root weight is its own term`);
    if (n.type === "gat_conv") {
      const m = G.messagesOf(FIX[s].ex, n.id)[0];
      assert.equal(m.heads.length, n.payload.heads);
      assert.ok(Math.abs(m.coef - m.heads.reduce((a, v) => a + v, 0) / m.heads.length) < 1e-12, "α averaged over the heads");
    }
  }
  const node = NODE.find((s) => convs(s).length);
  for (const n of convs(node)) assert.ok(G.messagesOf(FIX[node].ex, n.id).every((m) => m.dst === FIX[node].ex.attribution.target_node), "at node level every message goes into the explained node");
  assert.equal(G.convState(null, "B1"), null);
  assert.equal(G.convState({ ...FIX[node].ex, nodes: {} }, convs(node)[0].id), null);
});

test("a readout's answer: each node's share of the pooled vector, the largest first", () => {
  for (const s of GRAPH) {
    const pool = graphOf(s).stages().find((n) => n.type === "graph_pool"), st = G.poolState(FIX[s].ex, pool.id);
    const share = FIX[s].ex.nodes[pool.id].extra.node_share;
    assert.equal(st.nodes, share.nodes.length);
    assert.ok(st.top.length <= 8 && st.top.every((x, i, a) => i === 0 || Math.abs(a[i - 1].share) >= Math.abs(x.share)));
    if (pool.payload.readout === "mean") assert.ok(share.share.every((v) => Math.abs(v - 1 / share.nodes.length) < 1e-6), "a mean shares equally");
    assert.ok(Math.abs(share.share.reduce((a, v) => a + v, 0) - 1) < 1e-4, `${s}: the shares add up to one`);
  }
});

test("each convolution's matrices decode to their shapes, and a thumbnail is the blocks' means", () => {
  const want = { gcn_conv: ["graph.w.weight"], sage_conv: ["graph.w.root", "graph.w.neigh"], gat_conv: ["graph.w.weight", "graph.w.att_src", "graph.w.att_dst"], gin_conv: ["graph.w.mlp", "graph.w.mlp"] };
  for (const s of CASES) for (const n of convs(s)) {
    const g = graphOf(s), mats = G.matricesOf(n);
    assert.deepEqual(mats.map((m) => m.key), want[n.type], `${s} ${n.id}`);
    for (const m of mats) {
      const M = g.matrix(m.ref), shape = FIX[s].network.tensors[m.ref].shape;
      assert.equal(M.out, shape[0]); assert.equal(M.in, shape.slice(1).reduce((a, b) => a * b, 1));
      const T = G.thumbOf(M, 4), r0 = 0, r1 = Math.max(1, Math.floor(M.out / T.rows)), c1 = Math.max(1, Math.floor(M.in / T.cols));
      let sum = 0; for (let i = r0; i < r1; i++) for (let j = 0; j < c1; j++) sum += M.w[i * M.in + j];
      assert.ok(Math.abs(T.v[0] - sum / (r1 * c1)) < 1e-6, `${s} ${m.ref}: the first block's mean`);
      assert.ok(T.rows <= 4 && T.cols <= 4);
    }
  }
  assert.equal(G.thumbOf(null), null);
});

// ---- drawing: a context that records every call and the points it touched
function recorder() {
  const calls = [], pts = [];
  const g = new Proxy({ globalAlpha: 1, measureText: (s) => ({ width: String(s).length * 6 }) }, {
    get(o, k) {
      if (k in o) return o[k];
      return (...a) => { calls.push([k, a]); if (["moveTo", "lineTo", "arc", "fillRect", "strokeRect", "fillText"].includes(k)) pts.push(k === "fillText" ? [a[1], a[2]] : [a[0], a[1]]); };
    },
    set(o, k, v) { o[k] = v; return true; },
  });
  return { g, calls, pts };
}

function ctxOf(graph, mode, acts = null) {
  const colors = { acc: "#0a0", neg: "#00a", mut: "#555", fg: "#fff", fg2: "#ccc", muted: "#888", alert: "#f00", bg: "#000", line: "#333" };
  return { W: 900, H: 340, narrow: false, mode, graph, env: {}, K: 14, colors, boxW: 88, collapse: false, compact: false, top: 18, bottom: 36,
           captionFont: "10px mono", acts };
}

function drawOne(slug, id, { mode = "simple", ex = null } = {}) {
  const graph = graphOf(slug), seg = { kind: "box", stage: id }, r = G.default, acts = ex ? r.acts(seg, ex, graph) : null;
  const ctx = ctxOf(graph, mode, acts), m = r.measure(seg, ctx), box = { x: 100, w: m.w, y: 18, h: 286 };
  const L = r.layout(seg, box, ctx), rec = recorder();
  r.draw(rec.g, seg, L, { phase: 1, focus: new Focus(null), hover: null, acts: [acts], mode, g: rec.g, n: 1, i: 0, ctx });
  return { L, rec, box, acts, seg, ctx };
}

test("each graph stage draws inside its box, before an answer, with one, and its weights at the detail level", () => {
  fresh({ viewKey: "classes", shape: "graph" });
  for (const s of CASES) for (const n of graphOf(s).stages().filter((x) => ["graph_conv", "graph_pool"].includes(x.kind))) {
    for (const opt of [{}, { ex: FIX[s].ex }, { mode: "detail" }]) {
      const { rec, box } = drawOne(s, n.id, opt);
      assert.ok(rec.calls.some(([k]) => k === "fill" || k === "fillRect"), `${s} ${n.id} ${JSON.stringify(Object.keys(opt))}: something painted`);
      for (const [x, y] of rec.pts) assert.ok(x >= box.x - 40 && x <= box.x + box.w + 40 && y >= box.y - 20 && y <= box.y + box.h + 20, `${s} ${n.id}: ${x},${y} inside its box`);
    }
    if (n.kind === "graph_conv") {
      const T = G.matricesOf(n).map((m) => G.thumbOf(graphOf(s).matrix(m.ref)));
      const { rec } = drawOne(s, n.id, { mode: "detail" });
      assert.equal(rec.calls.filter(([k]) => k === "fillRect").length, T.reduce((a, t) => a + t.rows * t.cols, 0), `${s} ${n.id}: one cell per block of each matrix`);
      const { acts } = drawOne(s, n.id, { ex: FIX[s].ex });
      assert.ok(acts.conv && acts.conv.ring.length === Math.min(G.RING_TRY, acts.conv.others));
    }
  }
  clean("the canvas");
});

test("a pointer on a ring node names the message it carried; the caption says what the stage does", () => {
  fresh({ viewKey: "binary", shape: "graph" });
  for (const s of NODE.filter((x) => convs(x).length)) {
    const n = convs(s)[0], { L, seg, ctx } = drawOne(s, n.id, { ex: FIX[s].ex }), r = G.default;
    const p = G.ringPoints(L.acts.conv.ring.length, L.c, L.r)[0], hit = r.hit(seg, L, p.x, p.y);
    assert.equal(hit.k, 0); assert.equal(hit.message.src, L.acts.conv.ring[0].src);
    const cap = txt(r.caption(seg, hit, ctx));
    assert.ok(cap.includes(n.formula), `${s}: the formula`);
    assert.match(cap, /from distance 1: carried [+−]/, `${s}: the message`);
    assert.match(cap, /this node/);
    assert.equal(r.hit(seg, L, L.rect.x - 100, L.c.y), null);
    assert.equal(r.hit(seg, L, L.c.x, L.c.y).message, null);
  }
  clean("the captions");
});

test("at the every-weight level a pointer names the matrix under it, never a ring node that is not drawn", () => {
  fresh({ viewKey: "binary", shape: "graph" });
  for (const s of NODE.filter((x) => convs(x).length)) {
    const n = convs(s)[0], { L, seg, ctx } = drawOne(s, n.id, { mode: "detail", ex: FIX[s].ex }), r = G.default;
    assert.equal(L.detail, true, `${s}: the thumbnails drawn`);
    const p = G.ringPoints(L.acts.conv.ring.length, L.c, L.r)[0], ring = r.hit(seg, L, p.x, p.y);
    assert.equal(ring.message, null, `${s}: no message for a ring node off screen`);
    const m = L.thumbs[0], hit = r.hit(seg, L, m.box.x + m.box.w / 2, m.box.y + m.box.h / 2);
    assert.deepEqual(hit.thumb, { key: m.key, ref: m.ref }); assert.equal(hit.message, null);
    const M = graphOf(s).matrix(m.ref), cap = txt(r.caption(seg, hit, ctx));
    assert.ok(cap.includes(`${M.out} × ${M.in} weights`), `${s}: ${cap}`);
    assert.doesNotMatch(cap, /carried/);
    const back = drawOne(s, n.id, { ex: FIX[s].ex });
    assert.equal(back.L.detail, false, `${s}: the families level draws the ring again`);
    assert.ok(r.hit(back.seg, back.L, p.x, p.y).message, `${s}: and names its messages`);
  }
  clean("the thumbnail captions");
});

// ---- the shape module
const schemaOf = (slug) => {
  const ex = FIX[slug].ex, level = ex.sample.graph.level, raw = ex.inputs.raw;
  const cols = Object.keys(raw).map((name) => ({ name, dtype: typeof raw[name] === "string" ? "string" : "float",
    ...(name === "key" || name === "graph_id" ? { role: "graph" } : {}) }));
  const nodes = ex.sample.graph.nodes;
  return { input: cols, sample: { shape: "graph", level, graph_id: level === "graph" ? "graph_id" : null, reference: { nodes: level === "node" ? nodes - 6 : 0 } } };
};
const rowsOf = (slug, n) => Array.from({ length: n }, (_, k) => ({ ...FIX[slug].ex.inputs.raw, x0: k / 10, ...(FIX[slug].case.level === "graph" ? { graph_id: `g${k % 2}` } : { key: k }) }));
function envOf(slug, examples = null) {
  const level = FIX[slug].case.level;
  return { lane: { level, viewKey: "classes", shape: "graph" }, schema: schemaOf(slug), graph: graphOf(slug), story: null, examples };
}

test("the shape module has every member the registry asks of one", async () => {
  assert.deepEqual(R.missingMembers(await R.moduleFor("shape", "graph"), R.SHAPE_MEMBERS), []);
  assert.equal(S.drawInput.draw, (await import("../js/net/renderers/input.js")).default.draw);
});

test("node level: the editor sends the node alone, {rows: [row], at: 0}, the linking columns marked", () => {
  const s = NODE[0], env = envOf(s);
  fresh(env.lane);
  const host = document.createElement("div");
  S.editor.mount(host, env.schema, env);
  assert.equal(host.querySelector(".graph-nodes").hidden, true, "no chips for one node");
  assert.equal(host.querySelectorAll("input").length, env.schema.input.length);
  assert.equal(host.querySelectorAll(".graph-link").length, 1, "the linking column");
  const row = rowsOf(s, 1)[0];
  S.editor.load({ id: "p", rows: [row], expected: null }, env);
  const req = S.editor.read();
  assert.equal(req.at, 0); assert.equal(req.rows.length, 1);
  assert.equal(req.rows[0].x0, row.x0); assert.equal(req.rows[0].zone, row.zone);
  assert.ok(!("node" in req) && !("edges" in req), "R-A3: no edges travel");
  clean("the node editor");
});

test("graph level: the editor sends the whole graph and the chosen node, edited", () => {
  const s = GRAPH[0], env = envOf(s), rows = rowsOf(s, 5).map((r) => ({ ...r, graph_id: "g0", key: undefined }));
  fresh(env.lane);
  const host = document.createElement("div");
  S.editor.mount(host, env.schema, env);
  S.editor.load({ id: "p", rows, expected: "1" }, env);
  const chips = host.querySelectorAll(".graph-chip");
  assert.equal(chips.length, 5); assert.equal(host.querySelector(".graph-nodes").hidden, false);
  chips[3].click();
  const x0 = host.querySelectorAll("input").find((i) => i.dataset.col === "x0");
  x0.value = "7.5";
  const req = S.editor.read();
  assert.equal(req.at, 3); assert.equal(req.rows.length, 5);
  assert.equal(req.rows[3].x0, 7.5, "the chosen node edited"); assert.equal(req.rows[2].x0, rows[2].x0, "the others as they were");
  assert.match(txt(host.querySelector(".graph-nodes")), /the graph's 5 nodes/);
  clean("the graph editor");
});

test("the try view's edit toggle names a whole graph at graph level, one row at node level", () => {
  for (const [s, want, key] of [[GRAPH[0], "edit this graph", "graph.edit.toggle_graph"], [NODE[0], "edit this row", "try.edit"]]) {
    const env = envOf(s);
    fresh(env.lane);
    const toggle = document.createElement("button"), slot = document.createElement("span");
    toggle.className = "disclose caps"; toggle.setAttribute("data-for", "edit");
    slot.setAttribute("data-t", "try.edit"); slot.textContent = "edit this row";
    toggle.append(slot); document.body.append(toggle);
    S.editor.mount(document.createElement("div"), env.schema, env);
    assert.equal(slot.getAttribute("data-t"), key, `${s}: the slot keeps the key it shows`);
    assert.equal(txt(slot), want, s);
    clean(`the edit toggle ${s}`);
  }
});

test("a row of many at once opens its whole graph; the demonstration batch gives each row its graph's truth", () => {
  const s = GRAPH[0], items = [0, 1, 2].map((g) => ({ rows: rowsOf(s, 3).map((r, k) => ({ ...r, graph_id: `g${g}`, x0: g + k / 10 })), truth: String(g % 2) }));
  const examples = { schema: "vega/examples@2", demo: { items }, test: { rows: items.flatMap((it) => it.rows) }, presets: [] };
  const env = envOf(s, examples);
  assert.deepEqual(S.presetRows({ row: items[1].rows[2] }, env), items[1].rows);
  const batch = S.demoBatch(examples, "graph");
  assert.equal(batch.rows.length, 9);
  assert.deepEqual(batch.truth, ["0", "0", "0", "1", "1", "1", "0", "0", "0"], "each row with its graph's truth, as the service answers row by row");
  assert.deepEqual(batch.rows.map((r) => r.graph_id), items.flatMap((it) => it.rows.map((r) => r.graph_id)));
  const node = S.demoBatch({ demo: { items: [{ rows: [{ a: 1 }], truth: "x" }, { rows: [{ a: 2 }], truth: "y" }] } }, "node");
  assert.deepEqual(node, { rows: [{ a: 1 }, { a: 2 }], truth: ["x", "y"], history: null });
  fresh(env.lane);
  const src = S.sources(examples, env);
  assert.deepEqual(src.map((x) => x.id), ["demo", "test", "csv"]);
  assert.equal(src[1].run().rows.length, 9);
  assert.deepEqual(S.sources({}, env).map((x) => x.id), ["csv"]);
  clean("the sources");
});

test("a row of the reader's CSV opens its whole graph too, read from the file the page uploaded", async () => {
  const s = GRAPH[0], env = envOf(s, { schema: "vega/examples@2", demo: { items: [] }, presets: [] });
  const cols = env.schema.input.map((c) => c.name), rows = [7, 7, 8, 7].map((g, k) => ({ ...rowsOf(s, 1)[0], graph_id: g, x0: k + .5 }));
  const quote = (v) => (typeof v === "string" ? `"${v.replace(/"/g, '""')}"` : String(v));
  const csv = [[...cols, "label"].join(","), ...rows.map((r) => [...cols.map((c) => quote(r[c])), "1"].join(","))].join("\r\n") + "\r\n";
  const parsed = S.csvRows(csv, env.schema);
  assert.equal(parsed.length, 4); assert.ok(!("label" in parsed[0]), "the schema's input columns only");
  assert.equal(parsed[1].x0, 1.5, "typed as the editor types a cell");
  assert.deepEqual(S.csvRows('a,b\n"x, ""y""",2\n', { input: [{ name: "a", dtype: "string" }, { name: "b", dtype: "float" }] }), [{ a: 'x, "y"', b: 2 }]);
  fresh(env.lane);
  const csvSource = S.sources(env.examples, env).find((x) => x.id === "csv"), file = new Blob([csv]);
  assert.deepEqual(csvSource.run(file), { file }, "many at once uploads the file as before");
  await new Promise((r) => setTimeout(r, 0));
  const served = { ...rows[3], graph_id: "7" };                 // the service's row: its id as text
  const opened = S.presetRows({ id: null, row: served }, env);
  assert.equal(opened.length, 3, "the whole graph 7, not one node"); assert.deepEqual(opened.map((r) => r.x0), [0.5, 1.5, 3.5]);
  assert.deepEqual(S.presetRows({ id: "p", row: served }, env), [served], "a preset is not looked up in the last batch");
});

test("what tipped: the neighbourhood the answer lists, drawn and ranked, the rest apart", () => {
  for (const s of CASES) {
    const env = envOf(s), a = FIX[s].ex.attribution;
    fresh(env.lane);
    const out = S.attribution(FIX[s].ex, env), dots = out.querySelectorAll("circle"), lines = out.querySelectorAll("line");
    assert.equal(dots.length, Math.min(40, a.neighbours.length), `${s}: one dot per node listed`);
    const drawn = new Set(a.neighbours.slice(0, 40).map((n) => n.node));
    assert.equal(lines.length, (a.edges || []).filter((e) => drawn.has(e.src) && drawn.has(e.dst)).length, `${s}: the edges between them`);
    if (a.level === "node") {
      assert.equal(out.querySelectorAll(".self").length, 1, `${s}: the explained node marked`);
      assert.match(txt(out), /the node's own inputs, by family/, "the default lexicon's feature words");
    }
    const hops = Object.keys(a.hops).length;
    assert.equal(txt(out).includes("what each distance brought"), hops > 1, `${s}: the distances split only when there are several`);
    dots[1 % dots.length].onclick();
    assert.ok(out.querySelector(".graph-pick"), `${s}: a dot clicked opens its card`);
    assert.equal(out.querySelector(".graph-pick").querySelectorAll(".bar").length, (a.neighbours[1 % dots.length].top || []).length);
    clean(`what tipped ${s}`);
  }
});

test("A31: a gap the method reports is a row of its own, never folded into the bars; so is what lies beyond the nodes listed", () => {
  const s = NODE[0], env = envOf(s), ex = structuredClone(FIX[s].ex);
  fresh(env.lane);
  const bars = (n) => n.querySelectorAll(".bar").length, before = S.attribution(ex, env);
  assert.equal(before.querySelectorAll(".graph-gap").length, 0, "grad × input claims no completeness");
  ex.attribution.completeness = { sum: 1.25, target_minus_baseline: 1.5, gap: 0.25 };
  ex.attribution.outside = 0.125;
  const after = S.attribution(ex, env);
  assert.equal(after.querySelectorAll(".graph-gap").length, 1);
  assert.match(txt(after.querySelector(".graph-gap")), /add up to \+1\.250, while the score moved by \+1\.500: a gap of 0\.2500/);
  assert.match(txt(after), /tipped \+0\.125 together/);
  assert.equal(bars(after), bars(before), "the rows are not bars");
  clean("the gap");
});

// The env the try view's stepper hands the input step (try.js builds the Stepper without an env, so
// no `schema`), after the editor was mounted as the page mounts it.
function stepperEnv(s, env) {
  S.editor.mount(document.createElement("div"), env.schema, env);
  const st = new Stepper({ graph: env.graph, ex: FIX[s].ex, row: null, lane: env.lane, page: null, roles: null, inputStep: S.inputStep });
  const e = st.env();
  assert.ok(!("schema" in e), "the stepper's env carries no schema");
  return e;
}

test("the input step: the node's largest shaped numbers and what each hop brought; at graph level, the graph's cells", () => {
  for (const s of CASES) {
    const env = envOf(s), ex = FIX[s].ex;
    fresh(env.lane);
    const out = S.inputStep(ex, null, stepperEnv(s, env)), text = txt(out);
    assert.equal(out.querySelectorAll(".bar").length, 6, `${s}: the six largest`);
    if (ex.sample.graph.level === "node") {
      const per = new Map();
      ex.attribution.neighbours.filter((n) => n.hop > 0).forEach((n) => per.set(n.hop, (per.get(n.hop) || 0) + 1));
      for (const [h, n] of per) assert.ok(text.includes(`${n} neighbour${n === 1 ? "" : "s"} at distance ${h}`), `${s}: hop ${h}`);
      assert.match(text, /from the linking columns key/);
    } else {
      assert.match(text, new RegExp(`the graph's ${ex.inputs.shaped.length} nodes`));
      // a whole graph is read at once: no node alone, and its own rows do change the answer
      assert.match(text, /reads the whole graph at once/, `${s}: the whole graph`);
      assert.match(text, /between the rows of this graph: each of them can change the answer, and the rows of another graph never do/, `${s}: the edges`);
      assert.doesNotMatch(text, /alone|other rows of a request never change/, `${s}: the node-level words`);
    }
    clean(`the input step ${s}`);
  }
});

test("the answer's node ids: the explained node, the request's rows, the reference graph's nodes", () => {
  const s = NODE[0], ex = FIX[s].ex, env = envOf(s), rows = [FIX[s].ex.inputs.raw];
  fresh(env.lane);
  const Gr = S.graphOf(env.schema, env), name = S.namer(ex, rows, Gr), target = ex.attribution.target_node;
  assert.equal(name(target), "this node");
  // a reference node: never by its index in the assembled graph. Unnamed, by its rank in the answer
  // (or "another node" when the answer does not list it); named, by the key the service sent
  const near = ex.attribution.neighbours.map((n) => n.node).filter((id) => id !== target);
  assert.ok(near.length, "the fixture's answer lists neighbours");
  near.forEach((id, i) => assert.equal(name(id), `neighbour ${i + 1}`));
  const unlisted = Math.max(target, ...near) + 1000;
  assert.equal(name(unlisted), "another node");
  near.concat([unlisted]).forEach((id) => assert.doesNotMatch(name(id), new RegExp(`#|\\b${id}\\b`)));
  const keyed = { ...ex, sample: { ...ex.sample, graph: { ...ex.sample.graph, names: { column: "key", of: { [String(near[0])]: "B00840" } } } } };
  const kname = S.namer(keyed, rows, Gr);
  assert.equal(kname(near[0]), "node B00840", "a reference node by the key the service sent");
  assert.equal(kname(target), "this node");
  if (near.length > 1) assert.equal(kname(near[1]), "neighbour 2");
  // a one-column key_pair links the rows that share a value: it names no node; a two-column one
  // names each by its first column (a row's own key), knn and spatial by none
  assert.equal(Gr.key, null); assert.equal(Gr.level, "node");
  const two = (method) => ({ ...env.schema, input: [...env.schema.input, { name: "feeds", dtype: "string", role: "graph" }],
    sample: { ...env.schema.sample, method } });
  assert.equal(S.graphOf(two("key_pair"), env).key, "key");
  assert.equal(S.graphOf(two("spatial"), env).key, null);
  // the probe's rule (P-STEP node_index_named): an index written as a name is found, a rank or a key is not
  assert.deepEqual(indexNamed(ex, [`${name(near[0])} carried most`, kname(near[0]), "unit 3 · hop 1"]), []);
  assert.deepEqual(indexNamed(ex, [`node ${near[0]} · hop 1`, "reference node #12"]), [`node ${near[0]}`, "#12"]);
  assert.deepEqual(indexNamed(FIX[GRAPH[0]].ex, ["node 3"]), [], "graph level: a node's place is its name");
  const g = GRAPH[0], gx = FIX[g].ex, genv = envOf(g), grows = Array.from({ length: gx.sample.graph.nodes }, (_, k) => ({ key: `n${k}` }));
  const gname = S.namer(gx, grows, { ...S.graphOf(genv.schema, genv), key: "key" });
  assert.equal(gname(3), "n3", "a graph sent alone: node ids are its rows");
  clean("the names");
});
