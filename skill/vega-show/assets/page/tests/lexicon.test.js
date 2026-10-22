// node --test: core/lexicon.js — the lexicon@1 block is the page's only source of domain words
// (R-E6). Its closed sets are vega-show's catalog (lexicon); a key the block leaves out takes the
// bundle identity's row noun or the default words, which are the strings table's (D4 §11.2) and
// the catalog's; an entry that breaks the rules is refused and said, never shown; `target` has no
// default here (the strings table stands in for it). What the lexicon says is a machine code, a key
// of the strings table, with its params; the table's English is what the page shows.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  Lexicon, LEXICON_SCHEMA, LEXICON_KEYS, LEXICON_PATHS, OPTIONAL_KEYS, ENTRY_FIELDS, VALUE_RE, ARTICLES, DEFAULTS, REQUIRED,
  articleOf, entryProblem,
} from "../js/core/lexicon.js";
import { STRINGS } from "../js/core/strings.en.js";
import { dynamicMembers } from "../js/core/strings.keys.js";
import { Text } from "../js/core/text.js";

const CATALOG = JSON.parse(readFileSync(fileURLToPath(new URL("../../../scripts/catalog.json", import.meta.url)), "utf8")).lexicon;
const block = (entries) => ({ schema: LEXICON_SCHEMA, ...entries });
const plain = (o) => JSON.parse(JSON.stringify(o));
// A problem as the page says it: its code through the strings table, its params as placeholders.
const TEXT = new Text();
const say = (p) => TEXT.tText(p.code, p.params);
const said = (problems) => problems.map(say);
const LEXICON_CODES = ["block", "key", "entry", "value", "article", "path"].map((c) => `lexicon.problem.${c}`);

test("the closed sets are the catalog's: keys, optional keys, paths, entry, value rule, articles, required, defaults", () => {
  assert.equal(LEXICON_SCHEMA, CATALOG.schema);
  assert.deepEqual([...LEXICON_KEYS], CATALOG.keys);
  assert.deepEqual([...OPTIONAL_KEYS], CATALOG.optional);
  assert.deepEqual([...LEXICON_PATHS], CATALOG.paths);
  assert.deepEqual([...ENTRY_FIELDS], CATALOG.entry);
  assert.equal(VALUE_RE.source, CATALOG.value_pattern);
  assert.deepEqual(plain(ARTICLES), CATALOG.articles);
  assert.deepEqual(plain(REQUIRED), CATALOG.required);
  assert.deepEqual(plain(DEFAULTS), CATALOG.defaults);
  assert.deepEqual(Object.keys(DEFAULTS).sort(), LEXICON_KEYS.filter((k) => !OPTIONAL_KEYS.includes(k)).sort());
});

test("the default words are the strings table's (D4 §11.2), each keeping the lexicon's own rules", () => {
  const members = Object.keys(DEFAULTS).flatMap((k) => ENTRY_FIELDS.map((f) => `${k}.${f}`));
  assert.deepEqual(dynamicMembers("lexicon.default.").sort(), members.sort(), "the core fragment declares every default word");
  for (const [key, e] of Object.entries(DEFAULTS)) {
    assert.ok(Object.isFrozen(e), key);
    for (const f of ENTRY_FIELDS) assert.equal(e[f], STRINGS[`lexicon.default.${key}.${f}`], `${key}.${f}`);
    assert.equal(entryProblem(key, e), null, key);
  }
  assert.deepEqual(plain(DEFAULTS.source), { one: "source data", many: "source data", article: "the" });
  assert.deepEqual(plain(DEFAULTS.feature), { one: "input", many: "inputs", article: "an" });
  assert.deepEqual(plain(DEFAULTS.row), { one: "row", many: "rows", article: "a" });
  assert.deepEqual(plain(DEFAULTS.edge), { one: "edge", many: "edges", article: "an" });
});

test("paths: one, many, the article with one, the definite article with one", () => {
  const lex = new Lexicon(block({
    row: { one: "parcel", many: "parcels", article: "a" },
    feature: { one: "measurement", many: "measurements", article: "a" },
    source: { one: "parcel scanner", many: "parcel scanners", article: "the" },
  }));
  assert.equal(lex.get("row.one"), "parcel");
  assert.equal(lex.get("row.many"), "parcels");
  assert.equal(lex.get("row.a"), "a parcel");
  assert.equal(lex.get("row.the"), "the parcel");
  assert.equal(lex.get("feature.many"), "measurements");
  assert.equal(lex.get("source.the"), "the parcel scanner");
  assert.equal(lex.get("source.a"), "the parcel scanner");
  assert.deepEqual(lex.problems(), []);
});

test("a key the block leaves out: the identity's row noun (article by its first letter), else the default", () => {
  const lex = new Lexicon(null, { noun: "order", nouns: "orders", title: "Orders" });
  assert.equal(lex.get("row.a"), "an order");
  assert.equal(lex.get("row.many"), "orders");
  assert.equal(lex.get("feature.a"), "an input");
  assert.equal(lex.get("source.the"), "the source data");
  assert.equal(lex.get("source.many"), "source data");
  assert.equal(lex.get("edge.a"), "an edge");
  assert.equal(lex.has("row"), false, "the identity's noun is not the block's");
  assert.equal(new Lexicon(null, { noun: "parcel", nouns: "parcels" }).get("row.a"), "a parcel");
  assert.equal(new Lexicon(null, null).get("row.a"), "a row");
  assert.equal(new Lexicon(null, { noun: "Parcel 2", nouns: "parcels" }).get("row.one"), "row", "an identity noun that breaks the rule is not taken");
  assert.equal(new Lexicon(null, { noun: "parcel" }).get("row.many"), "rows", "an identity without its plural is not taken");
  const own = new Lexicon(block({ row: { one: "passenger", many: "passengers", article: "a" } }), { noun: "order", nouns: "orders" });
  assert.equal(own.get("row.a"), "a passenger", "the block's entry wins over the identity");
  assert.equal(articleOf("apple"), "an");
  assert.equal(articleOf("row"), "a");
});

test("target: the block's when it carries one, else no word here (the strings table's target.*)", () => {
  const withTarget = new Lexicon(block({ target: { one: "species", many: "species", article: "a" } }));
  assert.equal(withTarget.has("target"), true);
  assert.equal(withTarget.get("target.one"), "species");
  const without = new Lexicon(block({}));
  assert.equal(without.has("target"), false);
  assert.equal(without.get("target.one"), null);
  assert.deepEqual(plain(without.problems()), [{ code: "lexicon.problem.path", params: { path: "target.one" } }]);
  assert.deepEqual(said(without.problems()), ["lexicon: no path target.one"]);
});

test("entries that break the rules are refused, said, and replaced by the default", () => {
  const lex = new Lexicon(block({
    row: { one: "Parcel", many: "parcels", article: "a" },
    feature: { one: "measurement", many: "measurements", article: "the" },
    step: { one: "step 2", many: "steps", article: "a" },
    window: "window",
    colour: { one: "red", many: "reds", article: "a" },
    source_of: { row: "user:turn 3" },
  }));
  assert.equal(lex.get("row.one"), "row");
  assert.equal(lex.get("feature.a"), "an input");
  assert.equal(lex.get("step.one"), "step");
  assert.equal(lex.get("window.one"), "window");
  const problems = lex.problems();
  assert.deepEqual(plain(problems), [
    { code: "lexicon.problem.value", params: { key: "row", field: "one", value: '"Parcel"' } },
    { code: "lexicon.problem.article", params: { key: "feature", article: '"the"', allowed: "a, an" } },
    { code: "lexicon.problem.value", params: { key: "step", field: "one", value: '"step 2"' } },
    { code: "lexicon.problem.entry", params: { key: "window" } },
    { code: "lexicon.problem.key", params: { key: "colour" } },
  ]);
  assert.deepEqual(said(problems), [
    'lexicon.row: one "Parcel" breaks the value rule; the default stands',
    'lexicon.feature: article "the" is not one of a, an; the default stands',
    'lexicon.step: one "step 2" breaks the value rule; the default stands',
    "lexicon.window: is not an entry of one, many and article; the default stands",
    "lexicon.colour: not a lexicon key",
  ]);
  for (const k of ["row", "feature", "step", "window", "colour"]) assert.ok(said(problems).some((p) => p.startsWith(`lexicon.${k}:`)), k);
  assert.equal(entryProblem("source", { one: "api", many: "apis", article: "the" }), null);
  assert.match(say(entryProblem("row", { one: "api", many: "apis", article: "the" })), /article "the" is not one of a, an/);
  assert.match(say(entryProblem("row", { one: "x".repeat(32), many: "xs", article: "a" })), /one .* breaks the value rule/);
  assert.equal(say(entryProblem("row", { one: "parcel", article: "a" })), "lexicon.row: many undefined breaks the value rule; the default stands");
  assert.equal(entryProblem("row", { one: "o'clock reading", many: "o'clock readings", article: "an" }), null);
});

test("a block of another schema is refused whole: every word is the default", () => {
  const lex = new Lexicon({ schema: "vega/lexicon@2", row: { one: "parcel", many: "parcels", article: "a" } });
  assert.equal(lex.get("row.one"), "row");
  assert.deepEqual(plain(lex.problems()), [{ code: "lexicon.problem.block", params: { schema: LEXICON_SCHEMA } }]);
  assert.deepEqual(said(lex.problems()), ["lexicon: the block is not vega/lexicon@1; the defaults stand"]);
  assert.deepEqual(said(new Lexicon("parcels").problems()), ["lexicon: the block is not vega/lexicon@1; the defaults stand"]);
});

test("unknown keys and paths resolve to nothing, and are said once", () => {
  const lex = new Lexicon(null);
  assert.equal(lex.get("row.plural"), null);
  assert.equal(lex.get("colour.one"), null);
  assert.equal(lex.get("row"), null);
  assert.equal(lex.get("row.one.x"), null);
  assert.equal(lex.get("row.plural"), null);
  assert.deepEqual(lex.problems().map((p) => p.code), Array(4).fill("lexicon.problem.path"));
  assert.deepEqual(said(lex.problems()), ["lexicon: no path row.plural", "lexicon: no path colour.one", "lexicon: no path row", "lexicon: no path row.one.x"]);
});

test("every problem code is a key of the strings table, said with every placeholder filled", () => {
  for (const code of LEXICON_CODES) assert.equal(typeof STRINGS[code], "string", code);
  const lex = new Lexicon(block({ row: { one: "Parcel", many: "parcels", article: "a" }, colour: {}, window: 3,
    feature: { one: "input", many: "inputs", article: "the" } }));
  lex.get("row.plural");
  const seen = [...new Lexicon({ schema: "x" }).problems(), ...lex.problems()];
  assert.deepEqual([...new Set(seen.map((p) => p.code))].sort(), [...LEXICON_CODES].sort());
  for (const p of seen) {
    assert.ok(Object.isFrozen(p) && Object.isFrozen(p.params), p.code);
    assert.ok(Object.values(p.params).every((v) => typeof v === "string"), p.code);
  }
  const tx = new Text();
  seen.forEach((p) => tx.tText(p.code, p.params));
  assert.deepEqual(tx.missing(), []);
  assert.deepEqual(tx.unresolved(), []);
});

test("the keys a shape requires that the block does not carry (R-E6)", () => {
  const rowOnly = new Lexicon(block({
    row: { one: "parcel", many: "parcels", article: "a" },
    feature: { one: "measurement", many: "measurements", article: "a" },
    source: { one: "parcel scanner", many: "parcel scanners", article: "the" },
  }));
  assert.deepEqual(rowOnly.missing("row"), []);
  assert.deepEqual(rowOnly.missing("window"), ["step", "window"]);
  assert.deepEqual(rowOnly.missing("graph", "node"), ["node", "edge"]);
  assert.deepEqual(rowOnly.missing("graph", "graph"), ["node", "edge", "graph"]);
  assert.deepEqual(rowOnly.missing("graph"), ["node", "edge"], "a graph lane without a level is node level");
  assert.deepEqual(new Lexicon(null).missing("row"), ["row", "feature", "source"]);
});
