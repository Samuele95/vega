// node --test: the code room (js/views/code.js, js/views/highlight/*; D4 §6.3, R-D7, R-E8) on the
// fixture vega/code-view@2 answers of fixtures/code-view (make_code_views.py): the walk page.json
// carries (page_config.build_walk's ids, the explain catalog's anchors) resolved against the served
// files exactly as the fixture's own resolution says — the steps, their files and lines, and the
// steps that drop out and why —, with the highlighter page.json#code names (python, or plain for
// any other language); the files' roles and kinds as the vega-pack catalog closes them, each with
// its words; a v2.2 answer read, another version refused; the story's notes, their labels composed
// from report.kind.<kind> with the report's number and title, linked to the report at its page or
// to a stage brief, and never to a file the slot does not serve; the Python highlighter's runs and
// extents; and the room itself in a small DOM (dom-shim.js): the lead, the files, the walk, a
// step's text, note and highlighted lines, the § lines above them, back and next, a step opened
// from another view, a file opened alone, what the probe reads — no key missing, no placeholder
// left.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { install } from "./dom-shim.js";

install();
globalThis.matchMedia = () => ({ matches: false });
const { S, setState } = await import("../js/core/state.js");
const { configureText, missingKeys, unresolved } = await import("../js/core/text.js");
const { Lexicon } = await import("../js/core/lexicon.js");
const { STRINGS } = await import("../js/core/strings.en.js");
const { dynamicMembers } = await import("../js/core/strings.keys.js");
const code = await import("../js/views/code.js");
const python = (await import("../js/views/highlight/python.js")).default;
const plain = (await import("../js/views/highlight/plain.js")).default;

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const FIX = (name) => JSON.parse(readFileSync(here(`./fixtures/code-view/${name}.json`), "utf8"));
const PACK = JSON.parse(readFileSync(here("../../../../vega-pack/scripts/catalog.json"), "utf8"));
const NAMES = ["row-linked", "graph-briefs", "plain"];
const text = (node) => (node ? node.textContent : "").replace(/\s+/g, " ").trim();
const raw = (node) => (node ? node.textContent : "");

function words(fx) {
  configureText({ lang: "en", lex: new Lexicon(fx.story.lexicon, null), values: fx.story.values });
}

const reading = (walk) => ({
  steps: walk.steps.map((s) => ({ id: s.id, path: s.path, from: s.range[0] + 1, to: s.range[1] + 1 })),
  missing: walk.missing.map((m) => ({ id: m.id, why: m.miss })),
});

// ---- the walk ---------------------------------------------------------------------------------------
test("the walk from page.json resolves on every fixture answer as the fixture says", async () => {
  for (const name of NAMES) {
    const fx = FIX(name);
    words(fx);
    const view = code.codeView(fx.code), { hl, gap } = await code.loadHighlighter(fx.page.code.language);
    assert.equal(gap, null, name);
    assert.equal(hl, fx.page.code.language === "python" ? python : plain, `${name}: the highlighter of ${fx.page.code.language}`);
    assert.deepEqual(reading(code.resolveWalk(fx.page.walk, view.files, hl.extent)), fx.expected, name);
  }
});

test("the walk ids are R-E8's: root, store, build, the block types in forward order, then the fixed tail", () => {
  assert.deepEqual(FIX("row-linked").page.walk.map((s) => s.id),
    ["root", "store", "build", "dense", "activation", "dropout", "head", "forward", "link", "shape", "run", "explain", "route"]);
  assert.deepEqual(FIX("graph-briefs").page.walk.map((s) => s.id),
    ["root", "store", "build", "gcn_conv", "activation", "dropout", "head", "forward", "link", "shape", "run", "explain", "route"]);
  const fx = FIX("row-linked"), view = code.codeView(fx.code), walk = code.resolveWalk(fx.page.walk, view.files, python.extent);
  // `after`: the forward pass is Network's, not the helper class's above it; the run is the method, not the module function
  const forward = walk.steps.find((s) => s.id === "forward"), run = walk.steps.find((s) => s.id === "run");
  const lines = (path) => view.files.find((f) => f.path === path).lines;
  assert.match(lines(forward.path)[forward.range[0]], /^ {4}def forward\(self, batch\)/);
  assert.match(lines(run.path)[run.range[0]], /^ {4}@torch\.no_grad\(\)/, "the decorator opens the step");
  assert.match(lines(run.path)[run.range[0] + 1], /^ {4}def _forward\(self, prepared\)/);
});

test("a step runs on to its `to` definition just below, and an anchor that does not compile is a miss", () => {
  const far = Array.from({ length: 12 }, (_, k) => `Y${k} = ${k}`).join("\n");
  const src = `def _predict(rows):\n    return rows\n\n\n@app.post("/v1/predict")\ndef predict(body):\n    return _predict(body)\n\n${far}\nX = 1`;
  const files = code.codeView({ schema: code.CODE_VIEW, verified: true, language: "python", files: [{ path: "serve/app.py", role: "http_entry",
    kind: "entrypoint", digest: null, lines: 21, text: src }] }).files;
  const step = (to) => ({ id: "route", title: "t", text: "x", code: { file: "(^|/)app\\.py$", at: ["^def _predict\\("], after: null, to } });
  assert.deepEqual(code.resolveStep(step(null), files, python.extent).range, [0, 1]);
  assert.deepEqual(code.resolveStep(step("^@app\\.post"), files, python.extent).range, [0, 6]);
  assert.deepEqual(code.resolveStep(step("^Y4 = "), files, python.extent).range, [0, 12], "eleven lines below its end: reached");
  assert.deepEqual(code.resolveStep(step("^Y5 = "), files, python.extent).range, [0, 1], "twelve lines below: too far");
  assert.equal(code.resolveStep({ id: "x", code: { file: "(", at: ["a"] } }, files, python.extent).miss, "pattern");
  assert.equal(code.resolveStep({ id: "x", code: { file: "app", at: ["^nothing"] } }, files, python.extent).miss, "anchor");
  assert.equal(code.resolveStep({ id: "x", code: { file: "other", at: ["^def"] } }, files, python.extent).miss, "file");
});

// ---- the answer -------------------------------------------------------------------------------------
test("roles and kinds are the vega-pack catalog's code_view, each with its words", () => {
  configureText({ lang: "en" });
  assert.deepEqual([...code.ROLES], PACK.code_view.roles);
  assert.deepEqual([...code.KINDS], PACK.code_view.kinds);
  assert.equal(code.CODE_VIEW, PACK.code_view.schema);
  assert.deepEqual(dynamicMembers("code.role."), PACK.code_view.roles.map(code.roleKey));
  assert.deepEqual(dynamicMembers("code.about."), PACK.code_view.roles.map(code.roleKey));
  assert.deepEqual(dynamicMembers("code.kind."), PACK.code_view.kinds);
  assert.deepEqual(["Model", "FeatureTransformer", "ModelStore", "composition_root", "http_entry"].map(code.roleKey),
    ["model", "feature_transformer", "model_store", "composition_root", "http_entry"]);
  for (const r of code.ROLES) assert.ok(code.roleLabel(r) && code.aboutText(r), r);
  for (const k of code.KINDS) assert.ok(code.kindLabel(k), k);
  assert.equal(code.roleLabel("Elsewhere"), "Elsewhere", "an unknown role is shown as the service sent it");
  assert.equal(code.aboutText("Elsewhere"), "");
  assert.deepEqual(missingKeys(), []);
  for (const f of FIX("row-linked").code.files) assert.equal(f.digest.length, "sha256:".length + 64);
});

test("a v2.2 answer is read, another version and a malformed answer are refused", () => {
  configureText({ lang: "en" });
  const legacy = code.codeView({ package: "pkg", verified: "every file hashed against MANIFEST.json at start-up", files: [
    { path: "pkg/nets/network.py", role: "Model", port: "Model", module: "nets", about: "the network itself", lines: 1, text: "x = 1" },
    { path: "predict.py", role: "composition root", module: null, about: "wires", lines: 1, text: "def load():" },
    { path: "serve/app.py", role: "HTTP entry point", module: null, lines: 1, text: "app = None" },
    { path: "pkg/ports.py", role: null, module: "domain", lines: 1, text: "" },
    { path: "pkg/domain.py", role: null, module: "domain", lines: 1, text: "" }] });
  assert.equal(legacy.verified, true);
  assert.equal(legacy.language, "python");
  assert.deepEqual(legacy.files.map((f) => f.role), ["Model", "composition_root", "http_entry", "ports", "domain"]);
  assert.ok(legacy.files.every((f) => f.kind === null && !("about" in f)));
  const refused = (payload) => { try { code.codeView(payload); return null; } catch (e) { return e; } };
  const other = refused({ schema: "vega/code-view@3", files: [] });
  assert.ok(other instanceof code.CodeRoomError);
  assert.equal(other.key, "app.schema_mismatch");
  assert.equal(other.params.want, code.CODE_VIEW);
  assert.equal(refused({ schema: code.CODE_VIEW }).key, "code.malformed");
  assert.equal(refused({ schema: code.CODE_VIEW, files: [{ path: "a.py" }] }).key, "code.malformed");
  assert.equal(code.codeView({ schema: code.CODE_VIEW, verified: "yes", files: [] }).verified, false, "verified is a boolean");
});

// ---- the notes --------------------------------------------------------------------------------------
test("a note's label is report.kind.<kind> with the report's number and title, linked to the report at its page", () => {
  const fx = FIX("row-linked");
  words(fx);
  const note = (id) => code.noteOf(fx.story, id), values = fx.story.values;
  const labels = Object.fromEntries(["root", "store", "dense", "head"].map((id) => [id, text(code.noteLabel(note(id), values))]));
  assert.deepEqual(labels, { root: "Figure 15", store: "§3.3 System architecture", dense: "Equation (2)", head: "Table 4" });
  for (const kind of dynamicMembers("report.kind.")) assert.ok(STRINGS[`report.kind.${kind}`].includes("{number}"), kind);
  const root = code.noteLine(note("root"), values), a = root.querySelector("a");
  assert.equal(text(root), "In the report: Figure 15 — one adapter per role, wired in this one place.");
  assert.equal(a.getAttribute("href"), "showcase/report/report.pdf#page=25");
  assert.equal(a.getAttribute("target"), "_blank");
  assert.equal(a.getAttribute("rel"), "noopener");
  assert.equal(a.getAttribute("title"), "open the report at page 25");
  assert.equal(text(a), "Figure 15");
  // a reference without a link: the label as text; the note's own values and lexicon words resolved
  const forward = code.noteLine(note("forward"), values);
  assert.equal(forward.querySelector("a"), null);
  assert.equal(text(forward), "In the report: Equation (2) — the layers in the order the equation writes them (Equation 2).");
  assert.equal(text(code.noteLine(note("shape"), values)), "Note: the steps this file replays, one per item.");
  assert.equal(code.noteOf(fx.story, "build"), null);
  assert.deepEqual(missingKeys(), []);
  assert.deepEqual(unresolved(), []);
});

test("brief links: the stage brief is the label and the link, with code.note.file", () => {
  const fx = FIX("graph-briefs");
  words(fx);
  const root = code.noteOf(fx.story, "root");
  assert.equal(root.stage, "vega-pack");
  const line = code.noteLine(root, {}), a = line.querySelector("a");
  assert.equal(text(line), "In the vega-pack brief: where the modules meet.");
  assert.equal(a.getAttribute("href"), "showcase/briefs/vega-pack-brief.md");
  assert.equal(a.getAttribute("title"), "open vega-pack-brief.md");
  assert.equal(text(code.noteLine(code.noteOf(fx.story, "run"), {})), "Note: the batch the network reads, answered.");
  const rows = code.noteRows(root, "    ", true, {});
  assert.deepEqual(rows.map(raw), ["§    # see: the vega-pack brief", "§    #   where the modules meet"]);
  assert.ok(rows.every((r) => r.className === "ln note on"));
  assert.deepEqual(missingKeys(), []);
});

test("a link the slot does not serve is never linked", () => {
  words(FIX("row-linked"));
  const ref = { kind: "figure", number: "n", page: "p", title: "t" };
  for (const href of ["javascript:alert(1)", "https://example.test/report.pdf", "showcase/report/other.pdf", "showcase/briefs/../x-brief.md",
    "/showcase/report/report.pdf"]) {
    const n = code.noteOf({ code_notes: { root: { note: "x", link: { href, kind: "report" }, ref } } }, "root");
    assert.equal(n.link, null, href);
    assert.equal(code.noteLine(n, { n: 3, p: 9 }).querySelector("a"), null, href);
  }
  assert.equal(text(code.noteLabel({ ref: null, link: { href: "showcase/report/report.pdf", kind: "report" }, stage: null })), "the report");
  assert.equal(code.noteLabel({ ref: null, link: null, stage: null }), null);
});

test("a report object is named with the chapter it sits in, when the story binds that chapter", () => {
  words(FIX("row-linked"));
  const ref = { kind: "figure", number: "fig_n", page: "fig_p", title: "Dropout", chapter: { number: "ch_n", title: "The network" } };
  assert.equal(text(code.noteLabel({ ref, link: null, stage: null }, { fig_n: "19", fig_p: 28, ch_n: "3.4" })), "§3.4 The network · Figure 19");
  const bare = { ...ref, chapter: undefined };
  assert.equal(text(code.noteLabel({ ref: bare, link: null, stage: null }, { fig_n: "19" })), "Figure 19");
});

test("the § lines wrap the note at the code's indentation", () => {
  const fx = FIX("row-linked");
  words(fx);
  const long = { ...code.noteOf(fx.story, "root"), note: "word ".repeat(40).trim() };
  const pad = "        ", rows = code.noteRows(long, pad, false, fx.story.values).map(raw);
  assert.equal(rows[0], `§${pad}# report: Figure 15`);
  assert.ok(rows.length > 3);
  for (const r of rows.slice(1)) assert.ok(r.startsWith(`§${pad}#   word`) && r.length - `§${pad}#   `.length <= 62 - pad.length, r);
  assert.deepEqual(code.wrapWords("a bb ccc", 4), ["a bb", "ccc"]);
});

// ---- the highlighters -------------------------------------------------------------------------------
test("the Python highlighter: its runs give every line back, coloured by what it is", () => {
  for (const name of NAMES) {
    for (const f of FIX(name).code.files) {
      const lines = f.text.split("\n"), runs = (name === "plain" ? plain : python).tokens(f.text);
      assert.equal(runs.length, lines.length, f.path);
      runs.forEach((r, k) => assert.equal(r.map(([, s]) => s).join(""), lines[k], `${f.path}:${k + 1}`));
    }
  }
  const src = "@torch.no_grad()\ndef load(self, x=1e-3):  # read it\n    s = f'a{x}' + \"b\"\n    '''two\n    lines'''\n    return self";
  const [dec, def, str, open, close, ret] = python.tokens(src);
  assert.deepEqual(dec, [["tk-d", "@torch.no_grad"], [null, "()"]]);
  assert.deepEqual(def, [["tk-k", "def"], [null, " "], ["tk-f", "load"], [null, "("], ["tk-self", "self"], [null, ", "], [null, "x"], [null, "="],
    ["tk-n", "1e-3"], [null, "):  "], ["tk-c", "# read it"]]);
  assert.deepEqual(str, [[null, "    "], [null, "s"], [null, " = "], ["tk-s", "f'a{x}'"], [null, " + "], ["tk-s", "\"b\""]]);
  assert.deepEqual(open, [[null, "    "], ["tk-s", "'''two"]]);
  assert.deepEqual(close, [["tk-s", "    lines'''"]]);
  assert.deepEqual(ret, [[null, "    "], ["tk-k", "return"], [null, " "], ["tk-self", "self"]]);
  assert.deepEqual(plain.tokens("a <b> & c\n\nd"), [[[null, "a <b> & c"]], [], [[null, "d"]]]);
});

test("extents: decorators, the header and the deeper-indented body; a paragraph for plain text", () => {
  const lines = ["class A:", "    @property", "    def x(self):", "        return 1", "", "        # still x", "", "", "    def y(self):",
    "        pass", "", "z = 1"];
  assert.deepEqual(python.extent(lines, 2), [1, 5], "from the decorator, the trailing blank lines left out");
  assert.deepEqual(python.extent(lines, 1), [1, 5], "anchored on the decorator itself");
  assert.deepEqual(python.extent(lines, 0), [0, 9]);
  assert.deepEqual(python.extent(lines, 11), [11, 11]);
  const long = ["def f():", ...Array.from({ length: 200 }, () => "    x = 1")];
  assert.deepEqual(python.extent(long, 0), [0, 89], "at most ninety lines");
  assert.deepEqual(plain.extent(["a", "b", "", "c"], 0), [0, 1]);
  assert.deepEqual(plain.extent(["a", "b", "", "c"], 3), [3, 3]);
});

test("the highlighter registry: python.js for python, plain.js for any other language, plain when its own does not load", async () => {
  const seen = [], importer = (file) => { seen.push(file); return import(`../js/views/highlight/${file}`).then((m) => m.default); };
  assert.equal((await code.loadHighlighter("python", importer)).hl.language, "python");
  assert.equal((await code.loadHighlighter("r", importer)).hl.language, null);
  assert.equal((await code.loadHighlighter(null, importer)).hl.language, null);
  assert.deepEqual(seen, ["python.js", "plain.js", "plain.js"]);
  const broken = (file) => (file === "python.js" ? Promise.reject(new Error("no module")) : importer(file));
  const fell = await code.loadHighlighter("python", broken);
  assert.equal(fell.hl, plain);
  assert.deepEqual(fell.gap, { file: "python.js", error: "no module" });
  const hollow = (file) => (file === "python.js" ? Promise.resolve({ language: "python" }) : importer(file));
  assert.match((await code.loadHighlighter("python", hollow)).gap.error, /python\.js: tokens, extent/);
  await assert.rejects(code.loadHighlighter("r", () => Promise.resolve({})), /plain\.js: language, tokens, extent/);
});

// ---- the room ---------------------------------------------------------------------------------------
const SHELL = ["code-back", "code-verified", "code-lead", "code-files", "code-step", "code-head", "src", "code-nav", "code-n", "code-prev", "code-next"];

function shell() {
  const doc = install();
  const make = (tag, id) => { const n = doc.createElement(tag); n.setAttribute("id", id); doc.body.append(n); return n; };
  SHELL.forEach((id) => make(id === "code-prev" || id === "code-next" || id === "code-back" ? "button" : "div", id));
  make("ol", "walk");
  const src = doc.getElementById("src");
  src.scrolled = [];
  src.scrollTo = (o) => src.scrolled.push(o);
  doc.getElementById("code-step").scrollIntoView = () => {};
  return doc;
}

async function room(name, answer = null) {
  const fx = FIX(name), doc = shell(), fetched = [];
  globalThis.fetch = async (url) => { fetched.push(url); return answer ? answer(url) : { ok: true, status: 200, json: async () => fx.code }; };
  setState({ views: {}, page: { walk: fx.page.walk, code: fx.page.code }, story: { code_notes: fx.story.code_notes, values: fx.story.values },
    tour: { current: "code", show: () => code.show(), back: () => {} } });
  words(fx);
  await code.show();
  return { fx, doc, fetched, $: (id) => doc.getElementById(id) };
}

test("the room on the linked fixture: lead, files, walk, the first step with its note and lines, then back and next", async () => {
  const { fx, $, fetched } = await room("row-linked");
  assert.deepEqual(fetched, ["v1/code"]);
  assert.equal(text($("code-lead")).startsWith("10 files, "), true, text($("code-lead")));
  assert.match(text($("code-lead")), /Lines marked § are notes/);
  assert.doesNotMatch(text($("code-lead")), /left out/);
  assert.equal(text($("code-verified")), "✓ every file checked at start-up against the digest the manifest records");
  const files = $("code-files").querySelectorAll("button");
  assert.deepEqual(files.map((b) => b.getAttribute("data-path")), fx.code.files.map((f) => f.path));
  assert.equal(text(files[0]), "pkg/model/network.pymodel");
  assert.equal(files[0].getAttribute("title"), "the network itself");
  assert.equal(files[2].getAttribute("title"), "the recipe: a raw item to the network's inputs");
  const steps = $("walk").querySelectorAll("li");
  assert.equal(steps.length, fx.expected.steps.length);
  assert.equal(text(steps[0]), "01Start-up: the composition root");
  // the first step: its words, its note, its file and lines
  assert.equal(text($("code-step").querySelector("h2")), "Start-up: the composition root");
  assert.equal(text($("code-step").querySelector(".code-ref")), "In the report: Figure 15 — one adapter per role, wired in this one place.");
  const head = $("code-head").children.map(text);
  assert.equal(head[0], "predict.py");
  assert.equal(head[1], "composition root · entry point");
  assert.equal(head[2], `lines 16–23 of ${fx.code.files.find((f) => f.path === "predict.py").lines}`);
  assert.match(head[3], /^✓ [0-9a-f]{12}$/);
  const src = $("src").firstChild, hl = src.querySelectorAll(".hl");
  assert.equal(hl.length, 8);
  assert.equal(text(hl[0]), "16def load(bundle_dir=None) -> Predictor:");
  const notes = src.querySelectorAll(".note");
  assert.equal(raw(notes[0]), "§# report: Figure 15");
  assert.equal(notes[0].querySelector("a").getAttribute("href"), "showcase/report/report.pdf#page=25");
  assert.equal(src.children.indexOf(notes[0]), 15, "the § lines stand above the step's first line");
  assert.equal(text($("code-n")), "1 / 13");
  assert.equal($("code-prev").hidden, true);
  assert.equal(text($("code-next")), "next · The weights, trusted by digest →");
  // next, and the arrows
  $("code-next").click();
  assert.equal(text($("code-step").querySelector("h2")), "The weights, trusted by digest");
  assert.equal(text($("code-head").children[2]), `lines 15–20 of ${fx.code.files.find((f) => f.path === "pkg/model/store.py").lines}`);
  code.key({ key: "ArrowLeft", preventDefault: () => {} });
  assert.equal(text($("code-n")), "1 / 13");
  code.key({ key: "ArrowLeft", preventDefault: () => {} });
  assert.equal(text($("code-n")), "13 / 13");
  assert.equal(text($("code-next")), "from the start ↺");
  assert.deepEqual(missingKeys(), []);
  assert.deepEqual(unresolved(), []);
});

test("the room opens at a step another view asks for, and a file alone from the list", async () => {
  const { fx, $ } = await room("row-linked");
  code.openCode("dense");
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(text($("code-step").querySelector("h2")), fx.page.walk.find((s) => s.id === "dense").title);
  assert.equal(text($("code-step").querySelector(".code-ref")), "In the report: Equation (2) — W x + b, the affine part of every layer.");
  assert.equal(text($("code-n")), "4 / 13");
  const blocks = $("src").firstChild;
  assert.deepEqual(blocks.querySelectorAll(".note.on").map(raw), ["§# report: Equation (2)", "§#   W x + b, the affine part of every layer"]);
  assert.equal(blocks.querySelectorAll(".note").length, 4, "every note of this file: dense and head, two lines each");
  code.openCode("nothing-here");
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(text($("code-n")), "4 / 13", "a step the walk does not hold leaves the room where it was");
  $("code-files").querySelectorAll("button")[9].click();
  assert.equal(text($("code-step").querySelector("h2")), "pkg/domain.py");
  assert.equal(text($("code-step").querySelector("p")), "the value objects");
  assert.equal(text($("code-head").children[2]), "8 lines");
  assert.equal($("src").firstChild.querySelectorAll(".hl").length, 0);
  assert.equal(text($("code-n")), "");
  assert.equal(text($("code-next")), "start the walk →");
  $("code-next").click();
  assert.equal(text($("code-n")), "1 / 13");
});

test("the room on the brief-link fixture: the steps left out are counted, the notes link the briefs", async () => {
  const { fx, $ } = await room("graph-briefs");
  assert.match(text($("code-lead")), /^6 files, .* 4 steps of the walk are not in these files and are left out\.$/);
  assert.equal($("walk").querySelectorAll("li").length, 9);
  assert.equal(text($("code-step").querySelector(".code-ref")), "In the vega-pack brief: where the modules meet.");
  assert.deepEqual(code.walkReading(), {
    read: true, language: "python", gap: null, at: 0, steps: fx.expected.steps, missing: fx.expected.missing,
    links: [{ step: "root", href: "showcase/briefs/vega-pack-brief.md", kind: "file" },
      { step: "activation", href: "showcase/briefs/vega-forge-brief.md", kind: "file" }] });
  assert.deepEqual(missingKeys(), []);
});

test("the room on a language without a highlighter: plain text, uncoloured", async () => {
  const { $ } = await room("plain");
  assert.equal(code.walkReading().language, null);
  const src = $("src").firstChild;
  assert.equal(src.querySelectorAll(".tk-k").length + src.querySelectorAll(".tk-s").length, 0);
  assert.deepEqual(src.querySelectorAll(".hl").map(raw), ["2start <- function(dir) {", "3  model <- read_model(dir)", "4  serve(model)", "5}"]);
  assert.equal(text($("code-head").children[1]), "composition root · entry point");
});

test("the room says why when the service does not show its code, and reads again on the next visit", async () => {
  const absent = await room("row-linked", async () => ({ ok: false, status: 404, statusText: "Not Found", json: async () => ({ title: "Not Found" }) }));
  assert.equal(text(absent.$("code-step").querySelector(".err")), "This service does not show its code.");
  assert.equal(absent.$("code-nav").hidden, true);
  assert.deepEqual(code.walkReading(), { read: false, error: "code.absent" });
  const fx = FIX("row-linked");
  globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => fx.code });
  await code.show();
  assert.equal(absent.$("walk").querySelectorAll("li").length, 13);
  const failed = await room("row-linked", async () => ({ ok: false, status: 500, statusText: "Server Error", json: async () => ({ title: "Boom", detail: "down" }) }));
  assert.equal(text(failed.$("code-step").querySelector(".err")), "This service does not show its code: Boom — down");
  const none = await room("row-linked", async () => ({ ok: true, status: 200, json: async () => ({ schema: "vega/code-view@2", verified: true, language: "python", files: [] }) }));
  assert.match(text(none.$("code-lead")), /^0 files, 0 lines — .* 13 steps of the walk are not in these files and are left out\.$/);
  assert.equal(none.$("code-step").children.length, 0, "no file: nothing is left reading");
  assert.equal(none.$("code-nav").hidden, true);
  const other = await room("row-linked", async () => ({ ok: true, status: 200, json: async () => ({ schema: "vega/code-view@1", files: [] }) }));
  assert.equal(text(other.$("code-step").querySelector(".err")), "v1/code is vega/code-view@1; this page reads vega/code-view@2.");
  assert.deepEqual(missingKeys(), []);
});
