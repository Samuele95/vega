// What the task modules share: reading an answer's keys (D2's RowPrediction, R-C4), the page's bar
// vocabulary laid out for classes, scores and values, the small charts of many at once (histogram,
// stacked bars, confusion, scatter, sparkline) and the results breakdown. Every word is the strings
// table's; every value the service's answers or the story's values, passed through data() or a
// formatter.
import { el } from "../../core/dom.js";
import { fmtNum, fmtInt } from "../../core/fmt.js";
import { barRow } from "../../core/bars.js";
import { t, tText, tn, data, cls } from "../../core/text.js";
import { DEFAULT_ROLE, ANOMALY_SIDES } from "../../core/roles.js";

const SVG = "http://www.w3.org/2000/svg";
export const pct = (v) => (typeof v === "number" && Number.isFinite(v) ? `${(100 * v).toFixed(1)} %` : "—");
export const finite = (v) => typeof v === "number" && Number.isFinite(v);
export const shown = (v) => (finite(v) ? fmtNum(+v.toPrecision(4)) : "—");
export const warnings = (p) => ((p && p.warnings) || []);
export const roleCls = (env, name) => (env && env.roles ? env.roles.cls(name) : "");
export const isAlert = (env, name) => !!env && !!env.roles && env.roles.isAlert(name);
const median = (xs) => { const s = xs.filter(finite).slice().sort((a, b) => a - b); return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null; };
export { median };

// An answer's probability of one name ({of: number} keys: probabilities, cumulative).
export const valueOf = (map, name) => (map && finite(map[name]) ? map[name] : null);

// The top-k entries of a {name: number} map, largest first.
export const topOf = (map, k) => Object.entries(map || {}).filter(([, v]) => finite(v)).sort((a, b) => b[1] - a[1]).slice(0, k);

// Bars of {name: probability}, the chosen names selected in their role colour (try one's detail).
export function probabilityBars(map, { chosen = [], env, k = 5, thresholds = null }) {
  const host = el("div", { class: "bars" });
  topOf(map, k).forEach(([c, v], i) => {
    const on = chosen.includes(c), row = barRow({ name: cls(c), v, max: 1, label: pct(v), selected: on, i, ci: null,
      cls: on ? roleCls(env, c) : "", fillCls: on && isAlert(env, c) ? "bad" : "" });
    const tv = thresholds ? thresholds(c) : null;
    if (finite(tv)) row.querySelector(".track").append(el("div", { class: "ref", style: `left:${tv * 100}%`, title: tText("try.threshold", { t: shown(tv) }) }));
    host.append(row);
  });
  return host;
}

// One bar of a score in [lo, hi] with a threshold marker (binary, anomaly).
export function scoreBar({ name, v, lo = 0, hi = 1, threshold = null, cls = "", fillCls = "" }) {
  const span = Math.max(1e-12, hi - lo), at = (x) => Math.max(0, Math.min(1, (x - lo) / span));
  const row = barRow({ name, v: at(v) * span, max: span, label: shown(v), selected: true, i: 0, ci: null, cls, fillCls });
  if (finite(threshold)) row.querySelector(".track").append(el("div", { class: "ref", style: `left:${at(threshold) * 100}%`, title: tText("try.threshold", { t: shown(threshold) }) }));
  return row;
}

// ---- the two sides of an anomaly lane -------------------------------------------------------------
// {normal, anomalous, named}: the task's classes when the data has a label (the positive one, else
// the second, is the anomalous side), else the role map's own two words (core/roles.js
// ANOMALY_SIDES). The anomaly task module and the try view's preset dots read the sides here.
export function anomalySides(env) {
  const o = (env && env.output) || {}, cs = o.classes || [];
  const anomalous = o.positive ?? (cs.length === 2 ? cs[1] : null);
  if (anomalous === null) return { normal: ANOMALY_SIDES[0], anomalous: ANOMALY_SIDES[1], named: false };
  return { normal: cs.find((c) => c !== anomalous) ?? null, anomalous, named: true };
}

// ---- the weights read's axis (D4 §6.4, the "weights axis" column) --------------------------------
// The columns of the demo sample's attribution, [{id, name}] in its order: vega/attribution@2's
// axis, or the page's one form of it (net/weights.js attributionOf); [] without an attribution.
export function attributionColumns(attribution) {
  const cols = !attribution ? [] : attribution.axis && Array.isArray(attribution.axis.columns) ? attribution.axis.columns
    : Array.isArray(attribution.columns) ? attribution.columns : [];
  return cols.filter((c) => c && c.id !== null && c.id !== undefined && c.name !== null && c.name !== undefined)
    .map((c) => ({ id: String(c.id), name: String(c.name) }));
}

const roleOf = (env, name) => (env && env.roles ? env.roles.role(name) : DEFAULT_ROLE);

// An axis of named columns (classes, labels, the binary sides): one per name, in the lane's order,
// each with the id of the attribution's column of that name (null when the sample has none, so the
// map shows that column empty) and the name's role. Without names, the attribution's own columns.
export function namedAxis(names, attribution, env) {
  const cols = attributionColumns(attribution), byName = new Map(cols.map((c) => [c.name, c.id]));
  const list = names && names.length ? names.map(String) : cols.map((c) => c.name);
  return { columns: list.map((n) => ({ id: byName.get(n) ?? null, name: n, role: roleOf(env, n) })), title: t("net.weights.map") };
}

// An axis of the attribution's own strata (bins, groups, steps ahead), in its order; `roles` colours
// them by the role map (the anomaly sides), else every column is plain.
export function strataAxis(attribution, env, { roles = false } = {}) {
  return { columns: attributionColumns(attribution).map((c) => ({ ...c, role: roles ? roleOf(env, c.name) : DEFAULT_ROLE })),
           title: t("net.weights.map") };
}

// ---- many at once: counts and agreement ----------------------------------------------------------
// Per true name: how many rows of it the network got right (teal) and wrong (coral), worst first.
export function rightWrongBars(names, truth, right) {
  const per = new Map();
  truth.forEach((c, i) => { if (c === null || c === undefined) return; const k = String(c); const e = per.get(k) || { n: 0, ok: 0 }; e.n++; if (right(i)) e.ok++; per.set(k, e); });
  const mx = Math.max(1, ...[...per.values()].map((v) => v.n)), host = el("div", { class: "bars" });
  [...per.entries()].sort((a, b) => a[1].ok / a[1].n - b[1].ok / b[1].n).forEach(([c, v], i) => {
    const track = el("div", { class: "track" },
      el("div", { class: "fill", style: `width:${v.ok / mx * 100}%;background:var(--accent);animation-delay:${i * .04}s` }),
      el("div", { class: "stack", style: `left:${v.ok / mx * 100}%;width:${(v.n - v.ok) / mx * 100}%;background:var(--alert)` }));
    host.append(el("div", { class: "bar" + (v.ok < v.n ? " weak" : "") }, el("div", { class: "name", title: cls(c) }, cls(c)), track, el("div", { class: "val" }, `${v.ok} / ${v.n}`)));
  });
  return host;
}

// Counts of each name, largest first (what the network says, without a truth to compare with).
export function countBars(names, env) {
  const counts = new Map();
  names.forEach((c) => { if (c !== null && c !== undefined) counts.set(String(c), (counts.get(String(c)) || 0) + 1); });
  const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]), mx = entries.length ? entries[0][1] : 1, host = el("div", { class: "bars" });
  entries.forEach(([c, n], i) => host.append(barRow({ name: cls(c), v: n, max: mx, label: String(n), selected: false, i, ci: null, cls: "", fillCls: isAlert(env, c) ? "bad" : "" })));
  return host;
}

// The most frequent (true, predicted) confusions, as the mix-up list.
export function mixups(truth, predicted, env, k = 6) {
  const mix = new Map();
  truth.forEach((c, i) => { const p = predicted[i]; if (c !== null && c !== undefined && p !== c) { const key = `${c}\u0000${p}`; mix.set(key, (mix.get(key) || 0) + 1); } });
  const top = [...mix.entries()].sort((a, b) => b[1] - a[1]).slice(0, k);
  if (!top.length) return null;
  const host = el("div", { class: "mix" });
  top.forEach(([key, n]) => { const [c, p] = key.split("\u0000"); host.append(el("div", {}, cls(c), el("span", { class: "arrow" }, "  →  "), el("span", { class: isAlert(env, p) ? "bad" : "acc" }, cls(p))), el("div", { class: "n" }, n)); });
  return host;
}

// A histogram of numbers: equal-width bins between the smallest and largest, as bars; `mark` a
// value drawn as the reference line (a threshold); `split(i)` the group a value belongs to (drawn as
// its own series, e.g. per true class).
export function histogram(values, { bins = 10, mark = null, groups = null } = {}) {
  const xs = values.filter(finite), host = el("div", { class: "bars views-hist" });
  if (!xs.length) return host;
  let lo = Math.min(...xs, finite(mark) ? mark : Infinity), hi = Math.max(...xs, finite(mark) ? mark : -Infinity);
  if (hi - lo < 1e-12) { lo -= .5; hi += .5; }
  const w = (hi - lo) / bins, at = (x) => Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / w)));
  const series = groups ? [...new Set(groups.filter((g) => g !== null && g !== undefined).map(String))] : [null];
  const count = series.map((g) => { const c = new Array(bins).fill(0); values.forEach((x, i) => { if (finite(x) && (g === null || String(groups[i]) === g)) c[at(x)]++; }); return c; });
  const mx = Math.max(1, ...count.flat());
  count.forEach((c, s) => {
    if (series[s] !== null) host.append(el("div", { class: "caps muted views-hist-group" }, cls(series[s])));
    c.forEach((n, b) => {
      const row = barRow({ name: `${shown(lo + b * w)} – ${shown(lo + (b + 1) * w)}`, v: n, max: mx, label: String(n), selected: false, i: b, ci: null, cls: "" });
      if (finite(mark) && mark >= lo + b * w && mark < lo + (b + 1) * w) row.classList.add("views-hist-mark");
      host.append(row);
    });
  });
  return host;
}

// A small table of counts: rows are the true names, columns the predicted ones; the diagonal in the
// accent, the rest in the alert colour by share (confusion; ordinal draws its band).
export function confusion(names, truth, predicted, { band = 0 } = {}) {
  const idx = new Map(names.map((c, i) => [String(c), i])), K = names.length, m = names.map(() => new Array(K).fill(0));
  truth.forEach((c, i) => { const a = idx.get(String(c)), b = idx.get(String(predicted[i])); if (a !== undefined && b !== undefined) m[a][b]++; });
  const mx = Math.max(1, ...m.flat()), table = el("table", { class: "views-confusion" });
  table.append(el("thead", {}, el("tr", {}, el("th", {}, t("many.confusion.corner")), ...names.map((c) => el("th", { title: cls(c) }, cls(c))))));
  const body = el("tbody");
  m.forEach((row, a) => body.append(el("tr", {}, el("th", { title: cls(names[a]) }, cls(names[a])), ...row.map((n, b) => {
    const near = Math.abs(a - b) <= band, alpha = (.1 + .8 * n / mx).toFixed(2);
    return el("td", { class: near ? "acc" : n ? "bad" : "muted", style: n ? `opacity:${a === b || near ? 1 : alpha}` : "" }, n ? String(n) : "·");
  }))));
  table.append(body);
  return table;
}

// An SVG of points (x, y) with the y = x line (predicted against true).
export function scatter(xs, ys, { w = 320, h = 220 } = {}) {
  const pts = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => finite(x) && finite(y));
  const all = pts.flat(), lo = Math.min(...all), hi = Math.max(...all), span = Math.max(1e-12, hi - lo), p = 18;
  const X = (v) => p + (v - lo) / span * (w - 2 * p), Y = (v) => h - p - (v - lo) / span * (h - 2 * p);
  const svg = svgEl("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}`, class: "views-plot", role: "img", "aria-label": tText("many.scatter.aria", { n: pts.length }) });
  if (!pts.length) return svg;
  svg.append(svgEl("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), class: "views-plot-ref" }));
  pts.forEach(([x, y]) => svg.append(svgEl("circle", { cx: X(x).toFixed(1), cy: Y(y).toFixed(1), r: 2.4, class: "views-plot-dot" })));
  return svg;
}

// An SVG line through values, with an optional band [lo[i], hi[i]] (the forecast and its quantiles).
export function sparkline(values, { lo = null, hi = null, w = 260, h = 70 } = {}) {
  const vs = values.map((v) => (finite(v) ? v : null)), all = [...vs, ...(lo || []), ...(hi || [])].filter(finite);
  const svg = svgEl("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}`, class: "views-plot", role: "img", "aria-label": tText("try.sparkline.aria", { n: vs.length }) });
  if (!all.length) return svg;
  const a = Math.min(...all), b = Math.max(...all), span = Math.max(1e-12, b - a), p = 8, n = Math.max(1, vs.length - 1);
  const X = (i) => p + i / n * (w - 2 * p), Y = (v) => h - p - (v - a) / span * (h - 2 * p);
  if (lo && hi) {
    const up = hi.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`), down = lo.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).reverse();
    svg.append(svgEl("polygon", { points: [...up, ...down].join(" "), class: "views-plot-band" }));
  }
  const d = vs.map((v, i) => (v === null ? "" : `${i && vs[i - 1] !== null ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`)).join("");
  svg.append(svgEl("path", { d, class: "views-plot-line" }));
  vs.forEach((v, i) => { if (v !== null) svg.append(svgEl("circle", { cx: X(i).toFixed(1), cy: Y(v).toFixed(1), r: 2.6, class: "views-plot-dot" })); });
  return svg;
}

function svgEl(tag, attrs) {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  return n;
}

// ---- results: the story's breakdown@1, drawn as bars in its direction --------------------------------
// section: results.breakdown ({kind, metric, names, values, weak, direction, caption}); values: the
// story's values. The weak marks are the story's (resolve_story computed them); the page never
// decides weakness. Selected: the neutral name (the ordinary case), as the hand-built page did.
export function breakdownBars(b, values, env) {
  if (!b || !values) return null;
  const names = values[b.names], vals = values[b.values], weak = b.weak ? values[b.weak] : null;
  if (!Array.isArray(names) || !Array.isArray(vals) || names.length !== vals.length) return null;
  const hi = Math.max(1e-12, ...vals.filter(finite).map(Math.abs)), max = b.direction === "min" ? hi * 1.04 : Math.max(1, hi);
  const order = names.map((n, i) => i).sort((a, c) => (b.direction === "min" ? vals[a] - vals[c] : vals[c] - vals[a]));
  const host = el("div", { class: "bars" });
  order.forEach((i, k) => {
    const w = Array.isArray(weak) && weak[i] === true;
    host.append(barRow({ name: cls(names[i]), v: finite(vals[i]) ? Math.abs(vals[i]) : 0, max, label: shown(vals[i]),
      selected: !!env && !!env.roles && env.roles.isNeutral(String(names[i])), i: k, ci: null, cls: w ? "weak" : "", fillCls: w ? "bad" : "" }));
  });
  return host;
}

// ---- the table of many at once -----------------------------------------------------------------------
export const warningCell = (p) => (warnings(p).length ? String(warnings(p).length) : "");
export const truthCell = (truth) => (truth === null || truth === undefined ? "—" : Array.isArray(truth) ? truth.map((x) => cls(x)).join(", ") : cls(truth));
// An answer's name in a cell: "—" when the row was not answered (answered: false, its keys null).
export const answerCell = (v) => (v === null || v === undefined ? "—" : cls(v));
export const countOf = (n) => fmtInt(n);

// "{n} {row.many}"-style sentences with a highlighted count (the many view's big line).
export const sentence = (key, params) => t(key, params);
export const pluralSentence = (key, n, params) => tn(key, n, params);
