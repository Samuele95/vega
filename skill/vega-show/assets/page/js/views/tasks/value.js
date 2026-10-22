// The value view key (regression and quantile regression, and a sequence regression of a
// regression base): one number in the target's units (identity link). The answer's keys (D2's
// RowPrediction, R-C4): value, quantiles{"<q>": v} with a quantile loss, warnings. The truth is a
// number.
//   try one       the value, and the interval its outermost quantiles give; the value against the
//                 target's training range (/v1/schema label.range_seen, when the service states it);
//                 the quantiles one by one
//   many at once  the median absolute error; predicted against true with y = x (without a truth,
//                 the predictions' histogram); the mean absolute residual per true-value quintile;
//                 #, true, predicted, residual, warnings
//   results       per bin (breakdown@1 per_bin)
import { el } from "../../core/dom.js";
import { barRow } from "../../core/bars.js";
import { t, tText } from "../../core/text.js";
import { num } from "../shared.js";
import { finite, shown, median, histogram, scatter, scoreBar, breakdownBars, truthCell, warningCell, strataAxis } from "./shared.js";

export const headLabel = "head.caption@value";

// The quantiles of an answer, [[q, v]] in quantile order.
export const quantilesOf = (p) => Object.entries((p && p.quantiles) || {}).map(([q, v]) => [Number(q), v])
  .filter(([q, v]) => finite(q) && finite(v)).sort((a, b) => a[0] - b[0]);

const residual = (p, truth) => (finite(p.value) && finite(truth) ? p.value - truth : null);

// The target's training range, [lo, hi] (/v1/schema label.range_seen), or null when the service does
// not state it.
export function rangeOf(env) {
  const r = env && env.schema && env.schema.label ? env.schema.label.range_seen : null;
  return Array.isArray(r) && r.length === 2 && finite(r[0]) && finite(r[1]) && r[0] <= r[1] ? r : null;
}

// The residual context: one bar of the value on the target's training range; a value beyond it is
// said to be so.
function rangeRows(v, range) {
  const [lo, hi] = range, beyond = v < lo || v > hi;
  const bar = scoreBar({ name: tText("try.range", { lo: shown(lo), hi: shown(hi) }), v, lo, hi });
  return beyond ? [bar, el("div", { class: "hint views-range-beyond" }, t("try.range.beyond"))] : [bar];
}

export const output = Object.freeze({
  verdict(a) {
    const qs = quantilesOf(a.p);
    return el("span", {}, el("span", { class: "enter role-plain" }, shown(a.p.value)),
      qs.length > 1 ? el("span", { class: "p enter" }, t("try.interval", { lo: shown(qs[0][1]), hi: shown(qs[qs.length - 1][1]), from: qs[0][0], to: qs[qs.length - 1][0] })) : null);
  },
  agree(a, preset) {
    if (!preset || !finite(preset.expected) || !finite(a.p.value)) return null;
    return el("span", { class: "muted" }, t("try.residual", { v: shown(a.p.value - preset.expected), truth: shown(preset.expected) }));
  },
  detail(a, env) {
    const qs = quantilesOf(a.p), host = el("div", { class: "bars" }), range = rangeOf(env);
    if (range && finite(a.p.value)) host.append(...rangeRows(a.p.value, range));
    if (!qs.length) return host;
    const lo = Math.min(...qs.map((q) => q[1]), a.p.value), hi = Math.max(...qs.map((q) => q[1]), a.p.value), span = Math.max(1e-12, hi - lo);
    qs.forEach(([q, v], i) => host.append(barRow({ name: tText("try.quantile", { q }), v: v - lo + span * .02, max: span * 1.04, label: shown(v), selected: false, i, ci: null, cls: "" })));
    return host;
  },
  note() { return t("try.note.value"); },
});

// The mean absolute residual per quintile of the true values.
function byQuintile(preds, truth) {
  const pairs = preds.map((p, i) => [truth[i], residual(p, truth[i])]).filter(([y, r]) => finite(y) && finite(r)).sort((a, b) => a[0] - b[0]);
  const host = el("div", { class: "bars" });
  if (!pairs.length) return host;
  const q = Array.from({ length: 5 }, (_, k) => pairs.slice(Math.floor(k * pairs.length / 5), Math.floor((k + 1) * pairs.length / 5))).filter((g) => g.length);
  const means = q.map((g) => g.reduce((s, [, r]) => s + Math.abs(r), 0) / g.length), mx = Math.max(1e-12, ...means);
  q.forEach((g, k) => host.append(barRow({ name: `${shown(g[0][0])} – ${shown(g[g.length - 1][0])}`, v: means[k], max: mx, label: shown(means[k]), selected: false, i: k, ci: null, cls: "" })));
  return host;
}

export const summary = Object.freeze({
  sentence(preds, truth) {
    if (!truth) return t("many.sentence.no_truth", { n: num(preds.length), m: shown(median(preds.map((p) => p.value))) });
    const e = median(preds.map((p, i) => residual(p, truth[i])).filter(finite).map(Math.abs));
    return t("many.sentence.truth", { e: num(shown(e)), n: preds.length });
  },
  left(preds, truth) {
    if (!truth) return { title: t("many.left.values"), body: histogram(preds.map((p) => p.value)), caption: null };
    return { title: t("many.left.scatter"), body: scatter(truth, preds.map((p) => p.value)), caption: t("many.left.scatter.caption") };
  },
  right(preds, truth) { return truth ? { title: t("many.right.quintiles"), body: byQuintile(preds, truth) } : null; },
  columns() { return ["many.col.truth", "many.col.predicted", "many.col.residual", "many.col.warnings"]; },
  cells(p, truth) { return [truthCell(truth), shown(p.value), shown(residual(p, truth)), warningCell(p)]; },
  miss() { return false; },
});

// The weights read's columns: the quintile bins of the demo sample's target (the attribution's axis;
// the target: the value).
export const weightsAxis = (attribution, env) => strataAxis(attribution, env);

export const breakdown = (section, values, env) => breakdownBars(section, values, env);
