// The lexicon (vega/lexicon@1, story.json#lexicon): the page's only source of domain words. Each
// key is {one, many, article}; the strings table and the story name them by path:
//   {row.one} "parcel"   {row.many} "parcels"   {row.a} "a parcel" (the entry's article and one)
//   {source.the} "the parcel scanner"
// A key the block does not carry takes its default: the row noun from the bundle's identity
// (MANIFEST.serve.identity.noun / .nouns, the article by the first letter) when the identity has
// one, otherwise the default words of D4 §11.2 (row, input, source data, step, window, node, edge,
// graph), which are English and so live in the strings table: lexicon.default.<key>.one | .many |
// .article (the core fragment, declared under DYNAMIC). `target` has no default here: the strings
// table's target.one / target.many in the lane's variant stand in for it (text.js). An entry that
// breaks the lexicon's rules (lower case, no digit; `the` on source only) is refused, its default
// used and the refusal kept in problems().
//
// A problem is a machine code and the values of its placeholders, {code, params}: the code is a
// key of the strings table (lexicon.problem.*), whose English the page renders with
// tText(code, params). The closed sets mirror vega-show/scripts/catalog.json#lexicon
// (tests/lexicon.test.js holds them equal, the default words included).
import { META, STRINGS } from "./strings.en.js";

export const LEXICON_SCHEMA = "vega/lexicon@1";
export const LEXICON_KEYS = Object.freeze(["row", "feature", "source", "step", "window", "node", "edge", "graph", "target"]);
export const LEXICON_PATHS = Object.freeze(["one", "many", "a", "the"]);
export const OPTIONAL_KEYS = Object.freeze(["target"]);
export const ENTRY_FIELDS = Object.freeze(["one", "many", "article"]);
export const VALUE_RE = /^[a-z][a-z' -]{0,30}$/;
export const ARTICLES = Object.freeze({ default: Object.freeze(["a", "an"]), source: Object.freeze(["a", "an", "the"]) });

// The default entry of `key`, from the strings table's words.
const defaultEntry = (key) => Object.freeze(Object.fromEntries(ENTRY_FIELDS.map((f) => [f, STRINGS[`lexicon.default.${key}.${f}`]])));
export const DEFAULTS = Object.freeze(Object.fromEntries(
  LEXICON_KEYS.filter((k) => !OPTIONAL_KEYS.includes(k)).map((k) => [k, defaultEntry(k)])));

// The keys a lane's lexicon must carry, by sample shape (graphs by level).
export const REQUIRED = Object.freeze({
  row: Object.freeze(["row", "feature", "source"]),
  window: Object.freeze(["row", "feature", "source", "step", "window"]),
  "graph:node": Object.freeze(["row", "feature", "source", "node", "edge"]),
  "graph:graph": Object.freeze(["row", "feature", "source", "node", "edge", "graph"]),
});

// "a" before a consonant, "an" before a vowel: the rule for a noun the identity names.
export const articleOf = (word) => (/^[aeiou]/.test(word) ? "an" : "a");

// A problem: its code (a strings key) and its placeholders' values, as text (a value the block
// brings in its JSON form, so a string reads quoted and a missing one undefined).
const problem = (code, params) => Object.freeze({ code, params: Object.freeze(params) });
const shown = (value) => String(JSON.stringify(value));

// Why `value` is not a valid entry for `key`, as a problem (null when it is valid).
export function entryProblem(key, value) {
  if (!value || typeof value !== "object") return problem("lexicon.problem.entry", { key });
  for (const field of ["one", "many"]) {
    if (typeof value[field] !== "string" || !VALUE_RE.test(value[field])) {
      return problem("lexicon.problem.value", { key, field, value: shown(value[field]) });
    }
  }
  const allowed = ARTICLES[key] || ARTICLES.default;
  if (!allowed.includes(value.article)) {
    return problem("lexicon.problem.article", { key, article: shown(value.article), allowed: allowed.join(META.list.comma) });
  }
  return null;
}

// The row entry the bundle's identity gives, or null when it names no valid noun.
function identityRow(identity) {
  const one = identity && identity.noun, many = identity && identity.nouns;
  if (typeof one !== "string" || typeof many !== "string") return null;
  const e = { one, many, article: articleOf(one) };
  return entryProblem("row", e) ? null : Object.freeze(e);
}

function readBlock(block, problems) {
  const out = {};
  if (block === null || block === undefined) return out;
  if (typeof block !== "object" || block.schema !== LEXICON_SCHEMA) {
    problems.push(problem("lexicon.problem.block", { schema: LEXICON_SCHEMA }));
    return out;
  }
  for (const [key, value] of Object.entries(block)) {
    if (key === "schema" || key === "source_of") continue;
    if (!LEXICON_KEYS.includes(key)) { problems.push(problem("lexicon.problem.key", { key })); continue; }
    const why = entryProblem(key, value);
    if (why) problems.push(why);
    else out[key] = Object.freeze({ one: value.one, many: value.many, article: value.article });
  }
  return out;
}

const sameProblem = (a, b) => a.code === b.code && JSON.stringify(a.params) === JSON.stringify(b.params);

export class Lexicon {
  // block: story.json#lexicon (or null); identity: {noun, nouns} of MANIFEST.serve.identity, or null.
  constructor(block, identity = null) {
    this.issues = [];
    this.own = Object.freeze(readBlock(block, this.issues));
    this.row = identityRow(identity);
  }

  // True when the block carries a valid entry for `key`.
  has(key) { return Object.prototype.hasOwnProperty.call(this.own, key); }

  // The entry for `key`: the block's, else the identity's row, else the default; null for a key
  // with no default (target) or an unknown key.
  entry(key) {
    if (this.has(key)) return this.own[key];
    if (key === "row" && this.row) return this.row;
    return DEFAULTS[key] || null;
  }

  // "row.one" → "parcel"; "row.a" → "a parcel"; "source.the" → "the parcel scanner". null (and a
  // problem) for an unknown key or path, and for a key with no entry.
  get(path) {
    const [key, part, extra] = String(path).split(".");
    const e = extra === undefined && LEXICON_PATHS.includes(part) ? this.entry(key) : null;
    if (!e) { this.note(problem("lexicon.problem.path", { path: String(path) })); return null; }
    if (part === "a") return `${e.article} ${e.one}`;
    if (part === "the") return `the ${e.one}`;
    return e[part];
  }

  // The keys `shape` (and, for graphs, `level`) requires that the block does not carry.
  missing(shape, level = null) {
    const req = REQUIRED[shape === "graph" ? `graph:${level || "node"}` : shape] || REQUIRED.row;
    return req.filter((k) => !this.has(k));
  }

  // Keeps a problem once.
  note(p) { if (!this.issues.some((q) => sameProblem(p, q))) this.issues.push(p); }
  // Every problem so far, in the order found: [{code, params}].
  problems() { return this.issues.slice(); }
}
