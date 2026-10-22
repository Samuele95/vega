// node --test: the page's words. core/strings.en.js is one keyed table merged from the nine area
// fragments (R-F2; a key two fragments define is a load error); every key follows the grammar and
// every value the digit rule; the ids vega-show emits have their keys; strings.keys.json is the
// fresh key list; and core/text.js resolves a key against the lane (variants), a count (plurals),
// the lexicon, the story's values and page.json#explain, records what it cannot find, and wraps
// every string under the qps-ploc pseudo-locale.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { FRAGMENTS, META, STRINGS, mergeTables } from "../js/core/strings.en.js";
import { DYNAMIC_KEYS, KEYS_SCHEMA, dynamicKeys, dynamicMembers, keysDocument } from "../js/core/strings.keys.js";
import { stale } from "./strings-keys.js";
import {
  PLACEHOLDER, KEY_RE, PSEUDO, BUILD_LOCALES, WRAP, FORMATS, Text, parseTemplate, baseKey, variantsOf, pickLang,
  configureText, t, tText, tn, tnText, hasKey, data, missingKeys, unresolved, usedKeys, storyText, applyI18n,
} from "../js/core/text.js";
import { Lexicon, LEXICON_KEYS, LEXICON_PATHS } from "../js/core/lexicon.js";

const PAGE = fileURLToPath(new URL("..", import.meta.url));
const SKILLS = join(PAGE, "..", "..", "..");
const json = (...p) => JSON.parse(readFileSync(join(SKILLS, ...p), "utf8"));
const SHOW = json("vega-show", "scripts", "catalog.json");
const FORGE = json("vega-forge", "scripts", "catalog.json");
const COVERAGE = json("vega-forge", "scripts", "backends", "pytorch", "coverage.json");
const AREAS = ["core", "network", "views", "stepper", "code", "sequence", "graph", "latent", "how-advanced"];

// ---- a document small enough for node: fragments, elements, text nodes, attributes, [attr] queries
class FakeNode {
  constructor(nodeType, tag = null) {
    Object.assign(this, { nodeType, tagName: tag, childNodes: [], attrs: {}, style: {}, className: "", title: "", value: "" });
  }
  get textContent() { return this.nodeType === 3 ? this.value : this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) { if (this.nodeType === 3) this.value = String(v); else this.childNodes = [text(v)]; }
  nodesOf(x) { return typeof x === "string" ? [text(x)] : x.nodeType === 11 ? x.childNodes.splice(0) : [x]; }
  append(...xs) { xs.forEach((x) => this.childNodes.push(...this.nodesOf(x))); }
  prepend(...xs) { this.childNodes.unshift(...xs.flatMap((x) => this.nodesOf(x))); }
  replaceChildren(...xs) { this.childNodes = []; this.append(...xs); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  matches(sel) { return k(sel) in this.attrs; }
  querySelectorAll(sel) { return this.childNodes.flatMap((c) => (c.nodeType === 1 ? [...(c.matches(sel) ? [c] : []), ...c.querySelectorAll(sel)] : [])); }
}
const k = (sel) => sel.replace(/^\[|\]$/g, "");
const text = (s) => Object.assign(new FakeNode(3), { value: String(s) });
globalThis.document = {
  createDocumentFragment: () => new FakeNode(11),
  createTextNode: (s) => text(s),
  createElement: (tag) => new FakeNode(1, tag),
};
const element = (attrs = {}, ...children) => { const n = new FakeNode(1, "span"); Object.assign(n.attrs, attrs); n.append(...children); return n; };

// ---- the table
test("the table is the merge of the nine area fragments, each frozen, in R-F2's order", () => {
  assert.deepEqual(Object.keys(FRAGMENTS), AREAS);
  const onDisk = readdirSync(join(PAGE, "js", "core", "strings")).sort();
  assert.deepEqual(onDisk, AREAS.map((a) => `${a}.en.js`).sort());
  let n = 0;
  for (const [area, mod] of Object.entries(FRAGMENTS)) {
    assert.ok(Object.isFrozen(mod.STRINGS) && Object.isFrozen(mod.DYNAMIC), `${area}: STRINGS and DYNAMIC are frozen`);
    for (const [key, value] of Object.entries(mod.STRINGS)) assert.equal(STRINGS[key], value, `${area}: ${key}`);
    n += Object.keys(mod.STRINGS).length;
  }
  assert.equal(Object.keys(STRINGS).length, n);
  assert.ok(Object.isFrozen(STRINGS));
  assert.ok(Object.keys(FRAGMENTS.core.STRINGS).length > 0, "core is filled");
  assert.equal(META.lang, "en");
  assert.deepEqual(SHOW.locales, [META.lang]);
});

test("a key or a dynamic prefix two fragments define is a load error naming both areas", () => {
  const a = { STRINGS: { "x.y": "one" }, DYNAMIC: { "p.": ["a"] } }, b = { STRINGS: { "x.y": "two" }, DYNAMIC: { "p.": ["b"] } };
  assert.throws(() => mergeTables({ a, b }, "STRINGS"), /"x\.y" is defined by both a and b/);
  assert.throws(() => mergeTables({ a, b }, "DYNAMIC"), /"p\." is defined by both a and b/);
  assert.deepEqual(mergeTables({ a, c: { STRINGS: { "x.z": "three" } } }, "STRINGS"), { "x.y": "one", "x.z": "three" });
  assert.deepEqual(mergeTables({ a, e: {} }, "DYNAMIC"), { "p.": ["a"] });
});

// vega-show's digit rule (catalog digit_rule), with the catalog's name tokens as the only names:
// placeholders out, then no token with a decimal digit unless it has a letter and is a name.
function digitProblems(value) {
  const rest = value.replace(new RegExp(PLACEHOLDER.source, "g"), " ");
  const names = new Set(SHOW.name_tokens.map((s) => s.toLowerCase()));
  const out = [];
  for (const tok of rest.match(/[\p{L}\p{N}\p{M}]+(?:[-'][\p{L}\p{N}\p{M}]+)*/gu) || []) {
    if (/\p{Nd}/u.test(tok) && (!/\p{L}/u.test(tok) || !names.has(tok.toLowerCase()))) out.push(tok);
  }
  return out;
}

test("every key follows the key grammar, every value the placeholder grammar and the digit rule", () => {
  const suffixes = new Set(SHOW.placeholder.suffixes);
  for (const [key, value] of Object.entries(STRINGS)) {
    assert.match(key, KEY_RE, key);
    assert.equal(typeof value, "string", key);
    assert.deepEqual(digitProblems(value), [], `${key}: a digit outside a placeholder`);
    for (const p of parseTemplate(value).filter((x) => !("text" in x))) {
      if (p.fmt) assert.ok(suffixes.has(p.fmt), `${key}: ${p.raw}`);
      if (p.path) assert.ok(LEXICON_KEYS.includes(p.name) && LEXICON_PATHS.includes(p.path), `${key}: ${p.raw} is not a lexicon path`);
    }
    assert.ok(!/\{\{|\}\}/.test(value), `${key}: no {{…}} token`);
    if (key.endsWith("#one")) assert.ok(`${key.slice(0, -4)}#other` in STRINGS, `${key}: a #one without its #other`);
  }
  assert.deepEqual(digitProblems("top5 of F1 {n} x₁ R²"), []);
  assert.deepEqual(digitProblems("the 3 best · λ2"), ["3", "λ2"]);
});

test("the grammars are the catalog's: placeholders, format suffixes, build locales", () => {
  assert.equal(PLACEHOLDER.source, SHOW.placeholder.pattern);
  assert.deepEqual(Object.keys(FORMATS).sort(), [...SHOW.placeholder.suffixes].sort());
  assert.deepEqual([...BUILD_LOCALES], SHOW.build_locales);
  assert.equal(PSEUDO, "qps-ploc");
  for (const good of ["app.status.ready", "try.run@classes", "try.run@classes+row", "x.y#one", "model.seasonal-naive", "a@row#other"]) assert.match(good, KEY_RE);
  for (const bad of ["App.x", "x.", "x@", "x#two", "x@a+b+c", "-x", "x y"]) assert.doesNotMatch(bad, KEY_RE);
});

test("every dynamic member has its key, and the closed sets match the catalogs", () => {
  for (const [prefix, members] of Object.entries(DYNAMIC_KEYS)) {
    for (const m of members) assert.ok(Object.keys(STRINGS).some((key) => baseKey(key) === prefix + m), `${prefix}${m} has no key`);
  }
  assert.deepEqual(dynamicMembers("view."), SHOW.views.ids);
  assert.deepEqual(dynamicMembers("context.step."), SHOW.context_steps.ids);
  assert.deepEqual(dynamicMembers("model."), SHOW.result_models.names);
  assert.deepEqual(dynamicMembers("report.kind."), SHOW.label_kinds.kinds);
  assert.deepEqual([...dynamicMembers("metric.")].sort(), Object.keys(FORGE.metric_direction).sort());
  for (const m of COVERAGE.metrics) assert.ok(dynamicMembers("metric.").includes(m), `metric.${m}`);
  assert.deepEqual(dynamicMembers("no.such."), []);
  assert.equal(dynamicKeys().length, Object.values(DYNAMIC_KEYS).reduce((s, ms) => s + ms.length, 0));
});

test("strings.keys.json is the fresh key list of the table", () => {
  assert.equal(stale(), false, "run node tests/strings-keys.js --write");
  const doc = JSON.parse(readFileSync(join(PAGE, "strings.keys.json"), "utf8"));
  assert.equal(doc.schema, KEYS_SCHEMA);
  assert.deepEqual(doc.keys, Object.keys(STRINGS).sort());
  assert.deepEqual(Object.keys(doc.fragments), AREAS);
  assert.deepEqual(doc, JSON.parse(JSON.stringify(keysDocument())));
});

// Every key the shell and the scripts name exists, and every key of the table is named somewhere
// (a literal, a data-t slot, a dynamic member, or the base of a variant or plural).
test("every key the page names exists, and no key is dead", () => {
  const walk = (d) => readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(join(d, e.name)) : [join(d, e.name)]));
  const scripts = walk(join(PAGE, "js")).filter((f) => f.endsWith(".js") && !f.includes(join("core", "strings")));
  const roots = new Set(Object.keys(STRINGS).map((key) => key.split(".")[0]));
  const named = new Map();
  for (const f of scripts) {
    const src = readFileSync(f, "utf8").replace(/^\s*\/\/.*$/gm, "");
    for (const m of src.matchAll(/"([a-z][a-z0-9_]*(?:\.[a-z0-9_][a-z0-9_-]*)+(?:@[a-z_+]+)?(?:#[a-z]+)?)"/g)) {
      if (roots.has(m[1].split(".")[0]) && !/\.(pdf|json|js|css|html)$/.test(m[1])) named.set(m[1], f);
    }
  }
  const shell = readFileSync(join(PAGE, "index.html"), "utf8");
  for (const m of shell.matchAll(/data-t(?:-title|-aria)?="([^"]+)"/g)) named.set(m[1], "index.html");
  const keys = Object.keys(STRINGS);
  for (const [key, where] of named) assert.ok(keys.some((s) => s === key || baseKey(s) === key), `${where}: ${key} is not in the table`);
  const dyn = new Set(dynamicKeys());
  for (const key of keys) assert.ok(named.has(key) || named.has(baseKey(key)) || dyn.has(baseKey(key)), `${key} is named nowhere`);
});

// ---- core/text.js
const T = (table, opts = {}) => new Text({ table, ...opts });

test("variants: the lane's order, view key and shape, then the key itself; @ and # keys as written", () => {
  const table = { k: "base", "k@row": "shape", "k@classes": "view key", "k@classes+row": "both" };
  const lane = { viewKey: "classes", shape: "row" };
  assert.deepEqual(variantsOf(lane), ["classes+row", "classes", "row"]);
  assert.deepEqual(variantsOf({ view_key: "value", shape: "window" }), ["value+window", "value", "window"]);
  assert.deepEqual(variantsOf({ variants: () => ["x"] }), ["x"]);
  assert.deepEqual(variantsOf(null), []);
  const v = variantsOf(lane);
  assert.equal(T(table, { variants: v }).tText("k"), "both");
  const { "k@classes+row": _both, ...noBoth } = table;
  assert.equal(T(noBoth, { variants: v }).tText("k"), "view key");
  const { "k@classes": _vk, ...shapeOnly } = noBoth;
  assert.equal(T(shapeOnly, { variants: v }).tText("k"), "shape");
  assert.equal(T(table).tText("k"), "base");
  assert.equal(T(table, { variants: v }).tText("k@row"), "shape");
  assert.equal(baseKey("try.run@classes+row#one"), "try.run");
});

test("plurals: #zero at zero when present, then the language's category, #other, the key itself", () => {
  const table = { "x#zero": "none", "x#one": "{n} thing", "x#other": "{n} things", "y#one": "{n} unit", "y#other": "{n} units",
    "y@classes#other": "{n} classes", z: "{n} of it" };
  const tx = T(table, { variants: ["classes"] });
  assert.equal(tx.tnText("x", 0), "none");
  assert.equal(tx.tnText("x", 1), "1 thing");
  assert.equal(tx.tnText("x", 7), "7 things");
  assert.equal(tx.tnText("y", 0), "0 classes");
  assert.equal(tx.tnText("y", 1), "1 unit");
  assert.equal(tx.tnText("z", 2), "2 of it");
  assert.equal(tx.tnText("x", 2, { n: "two" }), "two things");
  assert.equal(tx.tnText("nope", 1), "nope");
  assert.deepEqual(tx.missing(), ["nope"]);
});

test("placeholders: parameters, formats, lexicon paths and the target noun", () => {
  const table = { a: "{row.a} of {n:int} with {p:pct}, {d:signed}, {c:count} and {x:num}", b: "every {target.one}, all {target.many}",
    "target.one@classes": "class", "target.many@classes": "classes", "target.one": "output", "target.many": "outputs", c: "{source.the} writes {feature.many}" };
  const lex = new Lexicon({ schema: "vega/lexicon@1", row: { one: "order", many: "orders", article: "an" } }, null);
  const tx = T(table, { lex, variants: ["classes"] });
  assert.equal(tx.tText("a", { n: 1234.4, p: 0.955, d: -0.5, c: [1, 2, 3], x: 0.123456 }), "an order of 1,234 with 95.5 %, −0.500, 3 and 0.1235");
  assert.equal(tx.tText("b"), "every class, all classes");
  assert.equal(T(table).tText("b"), "every output, all outputs");
  const withTarget = new Lexicon({ schema: "vega/lexicon@1", target: { one: "species", many: "species", article: "a" } }, null);
  assert.equal(T(table, { lex: withTarget, variants: ["classes"] }).tText("b"), "every species, all species");
  assert.equal(tx.tText("c"), "the source data writes inputs");
  const parcels = T(table, { lex: new Lexicon(null, { noun: "parcel", nouns: "parcels" }) });
  assert.equal(parcels.tText("a", { n: 2, p: 1, d: 1, c: 0, x: 2 }), "a parcel of 2 with 100 %, +1.000, 0 and 2");
});

test("what nothing resolves stays visible and is recorded; a missing key renders as itself", () => {
  const tx = T({ a: "{known} and {unknown} and {thing.one} and {row.nope}" });
  assert.equal(tx.tText("a", { known: "yes" }), "yes and {unknown} and {thing.one} and {row.nope}");
  assert.deepEqual(tx.unresolved(), ["{row.nope}", "{thing.one}", "{unknown}"]);
  assert.equal(tx.tText("no.such.key"), "no.such.key");
  assert.equal(tx.has("no.such.key"), false);
  const frag = tx.t("another.missing");
  assert.equal(frag.textContent, "another.missing");
  assert.match(frag.childNodes[0].style.cssText, /var\(--alert\)/);
  assert.deepEqual(tx.missing(), ["another.missing", "no.such.key"]);
});

test("explain.* keys read page.json#explain: blocks, activations under act., links under link.", () => {
  const explain = {
    blocks: { dense: { prose: "Each of the {units} units adds up", walk: { title: "A dense layer" } } },
    activations: { gelu: { bend: "passes positive evidence" } },
    links: { softmax: { formula: "pᵢ = exp(zᵢ) / Σ exp(z)" } },
  };
  const tx = T({}, { explain });
  assert.equal(tx.tText("explain.dense.prose", { units: 64 }), "Each of the 64 units adds up");
  assert.equal(tx.tText("explain.dense.walk.title"), "A dense layer");
  assert.equal(tx.tText("explain.act.gelu.bend"), "passes positive evidence");
  assert.equal(tx.tText("explain.link.softmax.formula"), "pᵢ = exp(zᵢ) / Σ exp(z)");
  assert.equal(tx.tText("explain.dense.walk"), "explain.dense.walk");
  assert.equal(tx.tText("explain.conv1d.prose"), "explain.conv1d.prose");
  assert.equal(T({}).tText("explain.dense.prose"), "explain.dense.prose");
  assert.deepEqual(tx.missing(), ["explain.conv1d.prose", "explain.dense.walk"]);
});

test("qps-ploc wraps the table's text ⟦…⟧, lexicon words and data ‹…›; a parameter keeps its own marks", () => {
  const table = { a: "choose {row.a}", "b#one": "{n} row", "b#other": "{n} rows", c: "{x} or {y}", d: "{label}",
    e: "pick {target.one}", "target.one": "an outcome" };
  const q = T(table, { lang: PSEUDO });
  assert.equal(q.pseudo, true);
  assert.equal(q.tText("a"), `${WRAP.chrome[0]}choose ${WRAP.chrome[1]}${WRAP.data[0]}a row${WRAP.data[1]}`);
  assert.equal(q.tnText("b", 3), "3⟦ rows⟧");
  assert.equal(q.t("a").textContent, "⟦choose ⟧‹a row›");
  assert.equal(q.tText("e"), "⟦pick ⟧⟦an outcome⟧", "the table's target noun is chrome");
  assert.equal(q.tText("c", { x: q.data("calm"), y: q.tnText("b", 1) }), "‹calm›⟦ or ⟧1⟦ row⟧");
  // a word passed in unmarked stays unmarked: the smoke sees it
  assert.equal(q.tText("c", { x: "gale", y: 2 }), "gale⟦ or ⟧2");
  const kbd = element({}, "g");
  assert.equal(q.t("d", { label: kbd }).textContent, "g");
  assert.equal(q.data("calm"), `${WRAP.data[0]}calm${WRAP.data[1]}`);
  assert.equal(q.data(42), "‹42›");
  const en = T(table);
  assert.equal(en.tText("a"), "choose a row");
  assert.equal(en.tText("e"), "pick an outcome");
  assert.equal(en.data("calm"), "calm");
});

test("the language: ?locale= when it names a build locale, else the table's; a story's locale is not read", () => {
  assert.equal(pickLang({ search: "?probe=all&locale=qps-ploc" }), PSEUDO);
  assert.equal(pickLang({ search: "?locale=fr" }), "en");
  assert.equal(pickLang({ search: "?locale=en" }), "en");
  // the language is chosen before the payloads are read, so a story cannot name it
  assert.equal(pickLang({ search: "", story: { locale: "qps-ploc" } }), "en");
  assert.equal(pickLang(), "en");
});

test("t() and tn() build fragments that keep a Node parameter as it is", () => {
  const kbd = element({}, "1");
  const frag = T({ keys: "jump with {k} or {j}" }).t("keys", { k: kbd, j: "esc" });
  assert.equal(frag.nodeType, 11);
  assert.ok(frag.childNodes.includes(kbd));
  assert.equal(frag.textContent, "jump with 1 or esc");
  assert.equal(T({ "x#other": "{n} files" }).tn("x", 3).textContent, "3 files");
  assert.equal(T({ a: "{k}" }).tText("a", { k: element({}, "node text") }), "node text");
});

test("story text: values as .num spans with the interval in the title, lexicon paths as words", () => {
  const values = { f1: { value: 0.9123, ci: [0.9, 0.93] }, runs: [1, 2, 3], n: 12 };
  const tx = T({}, { values, lex: new Lexicon(null, { noun: "parcel", nouns: "parcels" }) });
  const host = tx.story("The F1 is {f1} over {n} {row.many}, after {runs:count} runs; {missing} stays.", element(), false);
  const spans = host.childNodes.filter((c) => c.nodeType === 1);
  assert.equal(host.textContent, "The F1 is 0.9123 [0.9, 0.93] over 12 parcels, after 3 runs; {missing} stays.");
  assert.equal(spans[0].className, "num");
  assert.equal(spans[0].title, "0.9123 [0.9, 0.93]");
  assert.equal(tx.story("{f1}", element(), true).textContent, "0.9123");
  assert.deepEqual(tx.unresolved(), ["{missing}"]);
});

test("the page's words: configureText, then t/tText/tn/tnText/hasKey/data, applyI18n, missingKeys", () => {
  configureText({ lang: "en" });
  assert.equal(tText("app.status.ready"), "ready");
  assert.equal(hasKey("app.status.ready"), true);
  tText("gone.key");
  const lex = new Lexicon(null, { noun: "parcel", nouns: "parcels" });
  configureText({ lex, lane: { viewKey: "classes", shape: "row" }, values: { f1: 0.5 } });
  assert.deepEqual(missingKeys(), ["gone.key"], "the keys found missing so far are kept");
  assert.equal(tText("try.run"), "classify again");
  assert.equal(tText("try.choose"), "choose a parcel");
  assert.equal(t("try.edit").textContent, "edit this parcel");
  assert.equal(tn("legacy.experiments.variants", 1).textContent, "1 variant");
  assert.equal(tnText("legacy.experiments.variants", 4), "4 variants");
  assert.equal(data("x"), "x");
  assert.equal(storyText("{f1}").textContent, "0.5");
  const slot = element({ "data-t": "app.index" }), titled = element({ "data-t-title": "app.theme.title", "data-t-aria": "app.theme.aria" });
  const root = element({}, slot, element({}, titled));
  applyI18n(root);
  assert.equal(slot.textContent, "index");
  assert.equal(titled.title, "Light or dark (t)");
  assert.equal(titled.getAttribute("aria-label"), "Switch theme");
  configureText({ lang: PSEUDO });
  applyI18n(root);
  assert.equal(slot.textContent, "⟦index⟧");
  assert.deepEqual(unresolved(), []);
  configureText({ lang: "en" });
});

test("configureText keeps what was found so far: the missing keys and the unresolved placeholders", () => {
  configureText({ lang: "en" });
  storyText("{nobody} here");
  tText("no.such");
  assert.ok(unresolved().includes("{nobody}"));
  configureText({ lang: "en", lex: new Lexicon(null, null), lane: { viewKey: "classes", shape: "row" } });
  assert.ok(unresolved().includes("{nobody}"), "an unresolved placeholder outlives a reconfiguration");
  assert.ok(missingKeys().includes("no.such"), "a missing key outlives a reconfiguration");
  configureText({ lang: PSEUDO });
  assert.ok(unresolved().includes("{nobody}") && missingKeys().includes("no.such"), "and a change of language");
  configureText({ lang: "en" });
});

test("the keys a render resolved are recorded as resolved (variant, plural); a lookup alone records none", () => {
  const table = { k: "base", "k@classes": "view key", "n#one": "one {n}", "n#other": "{n} things", "target.one": "row" };
  const text = T(table, { variants: ["classes+row", "classes", "row"] });
  text.has("k"); text.lookup("n#one");
  assert.deepEqual(text.used(), [], "has() and lookup() render nothing");
  text.tText("k"); text.tnText("n", 3); text.t("no.such");
  assert.deepEqual(text.used(), ["k@classes", "n#other"], "a missing key is not a used one");
  configureText({ lang: "en" });
  tText("view.start");
  configureText({ lang: PSEUDO });
  assert.ok(usedKeys().includes("view.start"), "the used keys outlive a reconfiguration");
  configureText({ lang: "en" });
});
