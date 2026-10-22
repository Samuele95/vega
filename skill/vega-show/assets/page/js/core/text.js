// The page's words. Every chrome string comes from the one keyed table (strings.en.js) through
// t(key, params), resolved against the lane, the lexicon and the page's language:
//
//   variants  t("k") takes the first of k@<view key>+<shape>, k@<view key>, k@<shape>, k that the
//             table holds (the lane's variants, in that order); a key written with @ or # is looked
//             up as written
//   plurals   tn("k", n) picks k#zero (n = 0, when present), then k#<Intl plural category>, then
//             k#other, then k itself, each with the variants; {n} is passed
//   explain   explain.<block type>.<field>, explain.act.<activation>.<field> and
//             explain.link.<link>.<field> come from page.json#explain (the backend's catalog)
//   missing   a key the table does not hold renders as the key itself, in the alert colour, and is
//             kept for missingKeys() (the probe fails on any)
//   used      every key a render resolved, as resolved (its variant or plural key), is kept for
//             usedKeys(): the probe reports them, and the adaptation report (D6 §11.3 AD2) holds a
//             lane's rendered keys to the table and to the other lane's
//
// Placeholders (one grammar with the story and vega-show's digit rule, catalog.json#placeholder):
//   {name}, {name:int|num|pct|signed|count}   a runtime parameter (a string, a number, a Node);
//                                             in story text, a story value
//   {row.one}, {row.a}, {source.the}, …       a lexicon path (lexicon.js)
//   {target.one}, {target.many}               the lexicon's target when it carries one, else the
//                                             strings table's target.one / target.many in the
//                                             lane's variant
// A placeholder nothing resolves stays visible as written and is kept for unresolved().
//
// Languages: the table's META.lang, or the pseudo-locale qps-ploc, which the page takes from
// ?locale=qps-ploc on its URL (the showcase copy of the template is byte-identical, so its own
// META.lang never changes; a story's `locale` is not read: story-2 admits only "en", and the
// language must be known before the shell draws its first word). Under qps-ploc each run of a
// template's own text is wrapped ⟦…⟧, each lexicon word and each string data() marks as the
// dataset's or the service's ‹…›; a parameter keeps the marks it came with (a number none), so a
// page smoke can prove that no visible word bypasses the table, a word passed in as a parameter
// included.
import { S, setState } from "./state.js";
import { META, STRINGS } from "./strings.en.js";
import { Lexicon, LEXICON_KEYS } from "./lexicon.js";
import { fmtInt, fmtNum, fmtVal, scalar, signed } from "./fmt.js";

export const PLACEHOLDER = /\{([a-z][a-z0-9_]*)(?:\.([a-z][a-z0-9_]*))?(?::(int|num|pct|signed|count))?\}/;
// D4 §12.2's key grammar; a segment after the first may carry a hyphen, as the catalogs' ids do
// (model.seasonal-naive, model.graph-linear).
export const KEY_RE = /^[a-z][a-z0-9_]*(?:\.[a-z0-9_][a-z0-9_-]*)*(?:@[a-z_]+(?:\+[a-z_]+)?)?(?:#(?:zero|one|other))?$/;
export const PSEUDO = "qps-ploc";
export const BUILD_LOCALES = Object.freeze(["en", PSEUDO]);
export const WRAP = Object.freeze({ chrome: Object.freeze(["⟦", "⟧"]), data: Object.freeze(["‹", "›"]) });
const MISSING_STYLE = "color:var(--alert);font-family:var(--mono)";

export const FORMATS = Object.freeze({
  int: (v) => fmtInt(Math.round(Number(scalar(v)))),
  num: (v) => fmtNum(Number(scalar(v))),
  pct: (v) => `${+(100 * Number(scalar(v))).toFixed(1)} %`,
  signed: (v) => signed(Number(scalar(v))),
  count: (v) => fmtInt(Array.isArray(v) ? v.length : Number(scalar(v))),
});

const isNode = (v) => v !== null && typeof v === "object" && typeof v.nodeType === "number";
const has = (o, k) => !!o && Object.prototype.hasOwnProperty.call(o, k);

// A parameter as text: a number by fmtNum, a story value ({value, ci}, an array) by fmtVal.
function plainValue(v) {
  if (typeof v === "number") return fmtNum(v);
  if (typeof v === "string") return v;
  return fmtVal(v);
}

// "{a} and {b.c:pct}" → [{text: ""}, {name: "a", path: null, fmt: null, raw: "{a}"}, {text: " and "}, …]
export function parseTemplate(str) {
  const parts = [], re = new RegExp(PLACEHOLDER.source, "g");
  let at = 0, m;
  while ((m = re.exec(str))) {
    if (m.index > at) parts.push({ text: str.slice(at, m.index) });
    parts.push({ name: m[1], path: m[2] || null, fmt: m[3] || null, raw: m[0] });
    at = m.index + m[0].length;
  }
  if (at < str.length) parts.push({ text: str.slice(at) });
  return parts;
}

// The key a variant or plural key belongs to: "try.run@classes" → "try.run", "x#one" → "x".
export const baseKey = (key) => key.replace(/[@#].*$/, "");

// The lane's variant order: its own variants() when it has them, else from its view key and shape.
export function variantsOf(lane) {
  if (!lane) return [];
  if (typeof lane.variants === "function") return lane.variants();
  const vk = lane.viewKey || lane.view_key, shape = lane.shape;
  return [vk && shape ? `${vk}+${shape}` : null, vk, shape].filter(Boolean);
}

// The language to render in: the build locale ?locale= names on the URL, else the table's own (an
// unknown locale names nothing). It is chosen once, before the shell draws a word.
export function pickLang({ search = "" } = {}) {
  const asked = new URLSearchParams(search || "").get("locale");
  return asked && BUILD_LOCALES.includes(asked) ? asked : META.lang;
}

function pluralRules(lang) {
  try { return new Intl.PluralRules(lang === PSEUDO ? "en" : lang); } catch { return new Intl.PluralRules("en"); }
}

// A value of page.json#explain under an explain.* key, or null.
function explainValue(explain, key) {
  if (!explain) return null;
  const [, head, ...rest] = key.split(".");
  const roots = { act: explain.activations, link: explain.links };
  let node = has(roots, head) ? roots[head] : explain.blocks;
  for (const seg of has(roots, head) ? rest : [head, ...rest]) node = node && typeof node === "object" ? node[seg] : undefined;
  return typeof node === "string" ? node : null;
}

export class Text {
  // table: the strings; lang: META.lang or qps-ploc; lex: a Lexicon; variants: the lane's variant
  // order; values: the story's values; explain: page.json#explain; missing, unresolved: the Sets to
  // record into (kept across configureText).
  constructor({ table = STRINGS, lang = META.lang, lex = null, variants = [], values = null, explain = null,
    classNames = null, missing = null, unresolved = null, used = null } = {}) {
    Object.assign(this, { table, lang, variants: variants.slice(), values: values || {}, explain, classNames: classNames || {} });
    this.lex = lex || new Lexicon(null, null);
    this.missingSet = missing || new Set();
    this.unresolvedSet = unresolved || new Set();
    this.usedSet = used || new Set();
    this.rules = pluralRules(lang);
  }

  get pseudo() { return this.lang === PSEUDO; }

  // The template of a key after variant resolution, or null.
  lookup(key) {
    if (key.startsWith("explain.")) return explainValue(this.explain, key);
    if (/[@#]/.test(key)) return has(this.table, key) ? this.table[key] : null;
    for (const v of this.variants) if (has(this.table, `${key}@${v}`)) return this.table[`${key}@${v}`];
    return has(this.table, key) ? this.table[key] : null;
  }

  has(key) { return this.lookup(key) !== null; }

  // The key lookup(key) resolves to (its variant, when the lane has one), recorded as used.
  use(key) {
    const k = key.startsWith("explain.") || /[@#]/.test(key) ? key
      : this.variants.map((v) => `${key}@${v}`).find((vk) => has(this.table, vk)) || key;
    this.usedSet.add(k);
  }

  // The key tn(key, n) resolves to, or null.
  pluralKey(key, n) {
    const suffixes = [n === 0 ? "#zero" : null, `#${this.rules.select(n)}`, "#other", ""].filter((s) => s !== null);
    for (const s of suffixes) {
      for (const v of [...this.variants.map((x) => `@${x}`), ""]) if (has(this.table, key + v + s)) return key + v + s;
    }
    return null;
  }

  // One placeholder of a strings template: text, a Node, or null when nothing resolves it. With
  // `wrap`, a lexicon word is marked as the dataset's ‹…› and the table's own target noun ⟦…⟧.
  resolve(ph, params, wrap = false) {
    if (ph.path === null) {
      if (!has(params, ph.name)) return null;
      const v = params[ph.name];
      if (isNode(v)) return v;
      return ph.fmt ? FORMATS[ph.fmt](v) : plainValue(v);
    }
    const word = this.noun(ph.name, ph.path);
    if (word === null || !wrap) return word;
    return ph.name === "target" && !this.lex.has("target") ? this.wrapText(word) : this.data(word);
  }

  // A dotted placeholder: a lexicon path, or the target noun the strings table gives a lane whose
  // lexicon carries no target. Any other dotted name resolves to nothing.
  noun(name, path) {
    if (!LEXICON_KEYS.includes(name)) return null;
    if (name !== "target" || this.lex.has("target")) return this.lex.get(`${name}.${path}`);
    const tpl = this.lookup(`target.${path}`);
    if (tpl !== null) this.use(`target.${path}`);
    // the target's own words are plain; a placeholder in them is left as written (no recursion)
    return tpl;
  }

  // A template with its placeholders filled: a string (asNode false) or a DocumentFragment. With
  // `wrap` (the pseudo-locale), each run of the template's own text is wrapped ⟦…⟧ and the
  // parameters are left as they come, so a word passed in unwrapped stays visible to the smoke.
  render(tpl, params = {}, asNode = false, wrap = false) {
    const out = [];
    for (const p of parseTemplate(tpl)) {
      const v = "text" in p ? (wrap ? this.wrapText(p.text) : p.text) : this.resolve(p, params, wrap);
      if (v === null) this.unresolvedSet.add(p.raw);
      out.push(v === null ? p.raw : v);
    }
    if (!asNode) return out.map((v) => (isNode(v) ? v.textContent : v)).join("");
    const frag = document.createDocumentFragment();
    out.forEach((v) => frag.append(isNode(v) ? v : document.createTextNode(v)));
    return frag;
  }

  wrapText(s) { return this.pseudo ? `${WRAP.chrome[0]}${s}${WRAP.chrome[1]}` : s; }

  miss(key, asNode) {
    this.missingSet.add(key);
    if (!asNode) return key;
    const span = document.createElement("span");
    span.style.cssText = MISSING_STYLE; span.textContent = key;
    const frag = document.createDocumentFragment(); frag.append(span);
    return frag;
  }

  t(key, params = {}) {
    const tpl = this.lookup(key);
    if (tpl !== null) this.use(key);
    return tpl === null ? this.miss(key, true) : this.render(tpl, params, true, this.pseudo);
  }

  tText(key, params = {}) {
    const tpl = this.lookup(key);
    if (tpl !== null) this.use(key);
    return tpl === null ? this.miss(key, false) : this.render(tpl, params, false, this.pseudo);
  }

  tn(key, n, params = {}) {
    const k = this.pluralKey(key, n);
    if (k !== null) this.usedSet.add(k);
    return k === null ? this.miss(key, true) : this.render(this.table[k], { n, ...params }, true, this.pseudo);
  }

  tnText(key, n, params = {}) {
    const k = this.pluralKey(key, n);
    if (k !== null) this.usedSet.add(k);
    return k === null ? this.miss(key, false) : this.render(this.table[k], { n, ...params }, false, this.pseudo);
  }

  // A string the dataset, the service or the bundle brings (a class, a column, a file, a warning):
  // as it is, marked ‹…› under the pseudo-locale.
  data(v) { const s = String(v); return this.pseudo ? `${WRAP.data[0]}${s}${WRAP.data[1]}` : s; }

  // A class (or label) of the task in the words the story gives it (story@2 class_names: a word
  // for the class 1), else as the data names it; data either way.
  cls(v) { const s = String(v); return this.data(has(this.classNames, s) ? this.classNames[s] : s); }

  // Story text into `host`: a {value} becomes a .num span (the interval in its title; the value
  // alone when `short`), a lexicon path or the target noun its word; anything else stays as written.
  story(str, host, short = false) {
    for (const p of parseTemplate(String(str || ""))) {
      if ("text" in p) { host.append(p.text); continue; }
      if (p.path === null && has(this.values, p.name)) { host.append(this.storyValue(this.values[p.name], p.fmt, short)); continue; }
      const word = p.path === null ? null : this.noun(p.name, p.path);
      if (word === null) this.unresolvedSet.add(p.raw);
      host.append(word === null ? p.raw : word);
    }
    return host;
  }

  // Story text as a plain string, resolved as story() resolves it (values in their short form).
  storyString(str) {
    let out = "";
    for (const p of parseTemplate(String(str || ""))) {
      if ("text" in p) { out += p.text; continue; }
      if (p.path === null && has(this.values, p.name)) { const v = this.values[p.name]; out += p.fmt ? FORMATS[p.fmt](v) : fmtVal(v, true); continue; }
      const word = p.path === null ? null : this.noun(p.name, p.path);
      if (word === null) this.unresolvedSet.add(p.raw);
      out += word === null ? p.raw : word;
    }
    return out;
  }

  storyValue(v, fmt, short) {
    const span = document.createElement("span");
    span.className = "num";
    span.title = fmtVal(v);
    span.textContent = fmt ? FORMATS[fmt](v) : fmtVal(v, short);
    return span;
  }

  missing() { return [...this.missingSet].sort(); }
  used() { return [...this.usedSet].sort(); }
  unresolved() { return [...this.unresolvedSet].sort(); }
}

// ---- the page's current words (kept in the page state, core/state.js, as S.text)
export const current = () => S.text || setState({ text: new Text() }).text;

// Sets what the words are resolved against: {lang, lex, lane (or variants), values, explain}. The
// keys found missing and the placeholders found unresolved so far are kept.
export function configureText({ lane = null, variants = null, ...opts } = {}) {
  const prev = S.text;
  const text = new Text({ ...opts, variants: variants || variantsOf(lane), missing: prev ? prev.missingSet : null,
    unresolved: prev ? prev.unresolvedSet : null, used: prev ? prev.usedSet : null });
  return setState({ text }).text;
}

export const t = (key, params) => current().t(key, params);
export const tText = (key, params) => current().tText(key, params);
export const tn = (key, n, params) => current().tn(key, n, params);
export const tnText = (key, n, params) => current().tnText(key, n, params);
export const hasKey = (key) => current().has(key);
export const data = (v) => current().data(v);
export const cls = (v) => current().cls(v);
export const missingKeys = () => current().missing();
export const usedKeys = () => current().used();
export const unresolved = () => current().unresolved();

// Story text: today's text(), with lexicon paths and format suffixes.
export const storyText = (str, host, short = false) => current().story(str, host || document.createElement("span"), short);
// Story text as a plain string (a bar's name, a title attribute, a canvas label): the same
// resolution, values in their short form.
export const storyPlain = (str) => (str === null || str === undefined ? "" : current().storyString(str));

// Fills the shell's text slots under root: [data-t] (content), [data-t-title], [data-t-aria].
export function applyI18n(root = document) {
  const T = current();
  const nodes = (attr) => [...(root.matches && root.matches(`[${attr}]`) ? [root] : []), ...root.querySelectorAll(`[${attr}]`)];
  nodes("data-t").forEach((n) => n.replaceChildren(T.t(n.getAttribute("data-t"))));
  nodes("data-t-title").forEach((n) => { n.title = T.tText(n.getAttribute("data-t-title")); });
  nodes("data-t-aria").forEach((n) => n.setAttribute("aria-label", T.tText(n.getAttribute("data-t-aria"))));
}
