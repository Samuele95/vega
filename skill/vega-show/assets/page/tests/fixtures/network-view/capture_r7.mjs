// capture_r7.mjs — the R7 fixture of relevance.test.js: the R7 bundle's own v1/network and two
// v1/explain answers (the v2.2-merge payloads the hand-built page drew), with the values today's
// code computed from them, so that net/relevance.js and the dense renderer can be held to them
// after the code that computed them is gone.
//
//   node capture_r7.mjs <dir> [--legacy <path of js/legacy/how.js>] [--groups <story.json>]
//   node capture_r7.mjs --verify <page dir>
//
// <dir> holds network.json (GET v1/network), explain-0.json and explain-1.json (POST v1/explain
// answers). --legacy is the module whose relevance() is today's (default: this template's
// js/legacy/how.js, which exists until the stepper package replaces it; at a later commit, pass a
// checkout of the commit that captured the fixture). --groups is the bundle's story.json: its
// input_groups (regular expressions, story@1) give the input families today's drawing gathered.
//
// The fixture is anonymous: the inputs are renamed i0…, the classes k0…, the families g0…; the
// numbers (weights, the explanations' vectors) are the bundle's own, untouched. It records, per
// explanation: the winner, the hidden units' relevance, the inputs' attribution, the connections
// every weight draws in the "every weight" drawing of one answer (the strongest 90 per layer, 40
// into the answer), and the families drawing's kept units, connections and circle sizes (K = 12,
// the try drawing's); and, for the network view without an answer, the families drawing's kept
// units and connections (K = 14) and the every-weight drawing's connections.
//
// The capture computes those values with transcriptions of today's code. --verify holds the
// fixture on disk to today's classes themselves: <page dir> is a checkout of vega-show/assets/page
// at a commit that still has js/legacy/net.js (fc58219, say: `git archive fc58219
// vega-show/assets/page | tar -x -C <tmp>`); the script copies that net.js beside itself with its
// Diagram and Detail exported, stubs the few browser objects their constructors touch, builds them
// on the fixture's network (the families from story@1 patterns that match the fixture's own
// families), draws the two answers, and compares every value. Exit 0: every value equal.
import { readFileSync, writeFileSync, rmSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve } from "node:path";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : dflt; };

// ---- --verify: the fixture against js/legacy/net.js's Diagram and Detail
function stubBrowser() {
  const ctx = new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => ({ width: 10 })), set: (t, k, v) => { t[k] = v; return true; } });
  const canvas = () => ({ getBoundingClientRect: () => ({ width: 1100, height: 520 }), getContext: () => ctx, addEventListener() {}, width: 0, height: 0 });
  Object.assign(globalThis, { ResizeObserver: class { observe() {} }, devicePixelRatio: 1, matchMedia: () => ({ matches: true }),
    getComputedStyle: () => ({ getPropertyValue: () => "" }),
    document: { documentElement: { lang: "en", dataset: {} }, getElementById: () => null, createElement: () => canvas(), querySelectorAll: () => [] } });
  return canvas;
}

async function verify(page) {
  const FX = JSON.parse(readFileSync(resolve(HERE, "r7-legacy.json"), "utf8")), net = FX.network, canvas = stubBrowser();
  const { S } = await import(pathToFileURL(resolve(page, "js/core/state.js")).href);
  S.story = { input_groups: FX.families.map((f) => ({ label: f.label, text: "", match: [`^(${f.idx.map((i) => net.inputs[i]).join("|")})$`] })) };
  const copy = resolve(page, "js/legacy/net.verify.js");
  writeFileSync(copy, readFileSync(resolve(page, "js/legacy/net.js"), "utf8") + "\nexport { Diagram, Detail };\n");
  let bad = 0;
  const eq = (a, b, what) => { const ok = JSON.stringify(a) === JSON.stringify(b); bad += !ok; console.log(`${ok ? "OK:  " : "DIFF:"} ${what}`); };
  try {
    const { Diagram, Detail } = await import(pathToFileURL(copy).href);
    const d = new Diagram(canvas(), net, { k: 14 });
    eq(d.keep, FX.drawing.families.keep, "network view: the families drawing's kept units");
    eq(d.edges.map((l) => l.map((e) => [e.a, e.b, e.w])), FX.drawing.families.edges, "network view: the families drawing's connections");
    eq(new Detail(canvas(), net, {}).top, FX.drawing.detail, "network view: the every-weight connections");
    FX.explanations.forEach((ex, n) => {
      const want = FX.expected[n], t = new Diagram(canvas(), net, { k: 12, static: true }), dd = new Detail(canvas(), net, { static: true });
      t.showActivations(ex); dd.showActivations(ex);
      eq([t.keep, t.edges.map((l) => l.map((e) => [e.a, e.b, e.w])), t.acts, t.winner],
        [want.families.keep, want.families.edges, want.families.acts, want.win], `answer ${n}: the families drawing (units, connections, circles, answer)`);
      const path = want.detail.map((top) => { const mx = top.length ? Math.abs(top[0][2]) : 1; return top.map(([i, j, c]) => [i, j, c / mx]); });
      eq([dd.path, [...dd.pathInputs]], [path, want.detail_inputs], `answer ${n}: the every-weight path and its named inputs`);
    });
  } finally { rmSync(copy, { force: true }); }
  return bad ? 1 : 0;
}

if (args[0] === "--verify") {
  if (!args[1]) { console.error("usage: node capture_r7.mjs --verify <page dir with js/legacy/net.js>"); process.exit(2); }
  process.exitCode = await verify(resolve(args[1]));
} else await capture(args[0]);

// ---- the capture
async function capture(dir) {
  if (!dir) { console.error("usage: node capture_r7.mjs <dir> [--legacy how.js] [--groups story.json]"); process.exit(2); }
  const legacy = resolve(opt("--legacy", resolve(HERE, "../../../js/legacy/how.js")));
  const { relevance } = await import(pathToFileURL(legacy).href);
  const { decode } = await import(pathToFileURL(resolve(HERE, "../../../js/core/decode.js")).href);
  const read = (f) => JSON.parse(readFileSync(resolve(dir, f), "utf8"));

  const net = read("network.json");
  const answers = [read("explain-0.json").explanation, read("explain-1.json").explanation];
  const story = opt("--groups") ? JSON.parse(readFileSync(opt("--groups"), "utf8")) : { input_groups: [] };

  // today's families (legacy/views.js families()): the story's regex groups in order, the rest in "other"
  function familiesOf(inputs) {
    const groups = (story.input_groups || []).map((g) => ({ re: (g.match || []).map((m) => new RegExp(m)), idx: [] }));
    const other = { idx: [] };
    inputs.forEach((name, i) => { const g = groups.find((x) => x.re.some((r) => r.test(name))); (g || other).idx.push(i); });
    const out = groups.filter((g) => g.idx.length).map((g) => g.idx);
    if (other.idx.length) out.push(other.idx);
    return out.length ? out : [inputs.map((_, i) => i)];
}

  const signedNorm = (arr) => { const m = Math.max(1e-12, ...arr.map(Math.abs)); return arr.map((v) => v / m); };

  // the "every weight" drawing of one answer (legacy Detail.showActivations)
  function detailPath(R, mats) {
    return mats.map((m, l) => {
      const cand = [], last = l === mats.length - 1;
      for (let j = 0; j < m.out; j++) {
        if (last && j !== R.win) continue;
        for (let i = 0; i < m.in; i++) { const c = R.edge(l, i, j); if (c !== 0) cand.push([Math.abs(c), i, j, c]); }
      }
      cand.sort((p, q) => q[0] - p[0]);
      return cand.slice(0, last ? 40 : 90).map(([, i, j, c]) => [i, j, c]);
    });
}

  // the families drawing of one answer (legacy Diagram.showActivations), K units per hidden layer
  function familiesDrawing(R, mats, fam, outputs, K) {
    const keep = R.hidden.map((r) => r.map((v, j) => [Math.abs(v), j]).sort((a, b) => b[0] - a[0])
      .slice(0, Math.min(K, r.length)).map((t) => t[1]).sort((a, b) => a - b));
    const cols = [fam.map((idx) => ({ kind: "family", idx })), ...keep.map((ks) => ks.map((j) => ({ kind: "unit", j }))),
      outputs.map((c, j) => ({ kind: "class", j }))];
    const L = mats.length;
    const carried = (l, a, b) => {
      const src = cols[l][a], dst = cols[l + 1][b];
      if (src.kind === "family") return src.idx.reduce((t, i) => t + R.edge(0, i, dst.j), 0);
      return R.edge(l, src.j, dst.j);
    };
    const edges = cols.slice(0, -1).map((col, l) => {
      const list = [], last = l === L - 1;
      cols[l + 1].forEach((_, b) => {
        if (last && b !== R.win) return;
        const cand = col.map((_, a) => [a, b, carried(l, a, b)]).sort((p, q) => Math.abs(q[2]) - Math.abs(p[2]));
        list.push(...cand.slice(0, last ? 6 : 3));
      });
      return list;
    });
    const famRel = signedNorm(fam.map((idx) => idx.reduce((t, i) => t + R.inputs[i], 0)));
    const acts = [famRel, ...keep.map((ks, h) => { const all = signedNorm(R.hidden[h]); return ks.map((j) => all[j]); }), R.probs];
    return { k: K, keep, edges, acts };
}

  // the families drawing of the network view (legacy Diagram constructor): the K units per hidden layer
  // that both listen and speak the most, and every drawn node's 3 strongest inputs and strongest output
  function networkDrawing(mats, fam, outputs, K) {
    const norm2 = (arr) => Math.sqrt(arr.reduce((a, v) => a + v * v, 0));
    const keep = [];
    for (let h = 1; h < mats.length; h++) {
      const inM = mats[h - 1], outM = mats[h], score = [];
      for (let j = 0; j < inM.out; j++) {
        const a = norm2(Array.from(inM.w.subarray(j * inM.in, (j + 1) * inM.in)));
        let b = 0; for (let o = 0; o < outM.out; o++) b += outM.w[o * outM.in + j] ** 2;
        score.push([a * Math.sqrt(b), j]);
      }
      score.sort((p, q) => q[0] - p[0]);
      keep.push(score.slice(0, Math.min(K, score.length)).map((s) => s[1]).sort((a, b) => a - b));
    }
    const cols = [fam.map((idx) => ({ kind: "family", idx })), ...keep.map((ks) => ks.map((j) => ({ kind: "unit", j }))),
      outputs.map((c, j) => ({ kind: "class", j }))];
    const weight = (l, a, b) => {
      const m = mats[l], src = cols[l][a], dst = cols[l + 1][b];
      if (src.kind === "family") { let s = 0, q = 0; for (const i of src.idx) { const w = m.w[dst.j * m.in + i]; s += w; q += w * w; } return Math.sign(s || 1) * Math.sqrt(q); }
      return m.w[dst.j * m.in + src.j];
    };
    const edges = [];
    for (let l = 0; l < mats.length; l++) {
      const list = [], seen = new Set();
      const add = (a, b) => { const k = a + ":" + b; if (!seen.has(k)) { seen.add(k); list.push([a, b, weight(l, a, b)]); } };
      cols[l + 1].forEach((_, b) => {
        const cand = cols[l].map((_, a) => [a, Math.abs(weight(l, a, b))]);
        cand.sort((p, q) => q[1] - p[1]);
        cand.slice(0, 3).forEach(([a]) => add(a, b));
      });
      cols[l].forEach((_, a) => {
        let best = 0, bw = -1;
        cols[l + 1].forEach((_, b) => { const w = Math.abs(weight(l, a, b)); if (w > bw) { bw = w; best = b; } });
        add(a, best);
      });
      edges.push(list);
    }
    return { k: K, keep, edges };
}

  // the "every weight" drawing of the network view (legacy Detail constructor): every unit's three
  // strongest inputs (six into a class) and every input's strongest output
  function detailTop(mats) {
    return mats.map((m, li) => {
      const k = li === mats.length - 1 ? 6 : 3, edges = [], seen = new Set();
      const add = (i, j) => { const key = i * 65536 + j; if (!seen.has(key)) { seen.add(key); edges.push([i, j, m.w[j * m.in + i]]); } };
      for (let j = 0; j < m.out; j++) {
        const row = []; for (let i = 0; i < m.in; i++) row.push([Math.abs(m.w[j * m.in + i]), i]);
        row.sort((a, b) => b[0] - a[0]);
        for (let t = 0; t < k; t++) add(row[t][1], j);
      }
      for (let i = 0; i < m.in; i++) {
        let best = 0, bw = -1;
        for (let j = 0; j < m.out; j++) { const w = Math.abs(m.w[j * m.in + i]); if (w > bw) { bw = w; best = j; } }
        add(i, best);
      }
      return edges;
    });
}

  const mats = net.weights.map(decode);
  const fam = familiesOf(net.inputs);
  const rename = (names, prefix) => new Map(names.map((n, i) => [n, `${prefix}${i}`]));
  const inName = rename(net.inputs, "i"), outName = rename(net.outputs, "k");
  const anonymous = (ex) => ({
    ...ex, label: outName.get(ex.label), explained: outName.get(ex.explained),
    probabilities: Object.fromEntries(Object.entries(ex.probabilities).map(([c, p]) => [outName.get(c), p])),
    warnings: [],
  });
  const expected = answers.map((ex) => {
    const R = relevance(ex, mats, net.outputs);
    return { win: R.win, probs: R.probs, hidden: R.hidden, inputs: R.inputs, detail: detailPath(R, mats),
             detail_inputs: R.inputs.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]).slice(0, 6).map((t) => t[1]),
             families: familiesDrawing(R, mats, fam, net.outputs, 12) };
  });
  const fixture = {
    schema: "vega/page-fixture-r7@1",
    about: "The R7 bundle's v1/network and two v1/explain answers (v2.2-merge payloads, names anonymised) with the relevance, the every-weight path and the families drawing today's code (js/legacy/how.js relevance, legacy/net.js Diagram and Detail) computed from them. Captured by capture_r7.mjs; read by relevance.test.js and graph.test.js.",
    network: { ...net, inputs: net.inputs.map((n) => inName.get(n)), outputs: net.outputs.map((n) => outName.get(n)) },
    families: fam.map((idx, g) => ({ label: `g${g}`, idx })),
    explanations: answers.map(anonymous),
    drawing: { families: networkDrawing(mats, fam, net.outputs, 14), detail: detailTop(mats) },
    expected,
  };
  writeFileSync(resolve(HERE, "r7-legacy.json"), JSON.stringify(fixture) + "\n");
  console.log(`wrote r7-legacy.json: ${net.inputs.length} inputs, ${net.outputs.length} classes, ${mats.length} layers, ${fam.length} families, ${answers.length} explanations`);
}
