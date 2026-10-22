// The horizon view key (forecast): one value per step ahead, h = 1…H, each in the target's units
// (the horizon link); with a quantile loss, one band per step. The answer's keys (D2's
// RowPrediction, R-C4): values{"h<h>": v} or values{"h<h>": {"<q>": v}}, answered, warnings. The
// truth is one number per step ahead.
//   try one       the next H values as a line, with the quantile band; the values step by step
//   many at once  the mean absolute error per step; the error per step (with a truth) or the
//                 forecasts' spread per step; #, predicted, true, warnings
//   results       per step ahead (breakdown@1 per_horizon)
import { el } from "../../core/dom.js";
import { barRow } from "../../core/bars.js";
import { t, tText } from "../../core/text.js";
import { num } from "../shared.js";
import { finite, shown, median, sparkline, breakdownBars, warningCell, strataAxis } from "./shared.js";

export const headLabel = "head.caption@horizon";

// An answer's forecast: [{h, v, lo, hi}] in step order (v the median quantile when the steps carry
// quantiles, lo and hi the outermost ones).
export function stepsOf(p) {
  const values = (p && p.values) || {};
  return Object.entries(values).map(([k, x]) => [Number(String(k).replace(/^h/, "")), x]).filter(([h]) => Number.isInteger(h))
    .sort((a, b) => a[0] - b[0]).map(([h, x]) => {
      if (finite(x)) return { h, v: x, lo: null, hi: null };
      const qs = Object.entries(x || {}).map(([q, v]) => [Number(q), v]).filter(([q, v]) => finite(q) && finite(v)).sort((a, b) => a[0] - b[0]);
      if (!qs.length) return { h, v: null, lo: null, hi: null };
      const mid = qs.reduce((best, q) => (Math.abs(q[0] - .5) < Math.abs(best[0] - .5) ? q : best), qs[0]);
      return { h, v: mid[1], lo: qs[0][1], hi: qs[qs.length - 1][1] };
    });
}
const truthAt = (truth, k) => (Array.isArray(truth) ? truth[k] : k === 0 ? truth : null);

export const output = Object.freeze({
  verdict(a) {
    const s = stepsOf(a.p), band = s.every((x) => finite(x.lo) && finite(x.hi));
    return el("span", { class: "views-forecast enter" }, sparkline(s.map((x) => x.v), band ? { lo: s.map((x) => x.lo), hi: s.map((x) => x.hi) } : {}),
      el("span", { class: "p" }, t("try.steps_ahead", { n: s.length })));
  },
  // The mean absolute error over the steps ahead only when every step has its true value; a preset
  // that carries the first step's truth alone (a bundle packed before the presets carried the whole
  // horizon) says it is the error one step ahead, never a mean over steps it does not have.
  agree(a, preset) {
    if (!preset || preset.expected === null || preset.expected === undefined) return null;
    const s = stepsOf(a.p), err = (k) => (s[k] && finite(s[k].v) && finite(truthAt(preset.expected, k)) ? Math.abs(s[k].v - truthAt(preset.expected, k)) : null);
    const errs = s.map((_, k) => err(k));
    if (s.length && errs.every(finite)) return el("span", { class: "muted" }, t("try.mean_error", { e: shown(errs.reduce((u, v) => u + v, 0) / errs.length) }));
    return finite(errs[0]) ? el("span", { class: "muted" }, t("try.error_first", { e: shown(errs[0]) })) : null;
  },
  detail(a) {
    const s = stepsOf(a.p), host = el("div", { class: "bars" }), vs = s.flatMap((x) => [x.v, x.lo, x.hi]).filter(finite);
    const lo = Math.min(0, ...vs), hi = Math.max(1e-12, ...vs), span = hi - lo || 1;
    s.forEach((x, i) => host.append(barRow({ name: tText("try.step_ahead", { h: x.h }), v: finite(x.v) ? x.v - lo : 0, max: span, label: shown(x.v), selected: false, i,
      ci: finite(x.lo) && finite(x.hi) ? [x.lo - lo, x.hi - lo] : null, cls: "" })));
    return host;
  },
  note() { return t("try.note.horizon"); },
});

// The mean absolute error of each step ahead over the rows (with a truth), or the forecasts' median
// per step (without one).
function perStep(preds, truth) {
  const steps = preds.map(stepsOf), H = Math.max(0, ...steps.map((s) => s.length)), host = el("div", { class: "bars" });
  const stat = Array.from({ length: H }, (_, k) => {
    if (!truth) return median(steps.map((s) => (s[k] ? s[k].v : null)));
    const e = steps.map((s, i) => (s[k] && finite(s[k].v) && finite(truthAt(truth[i], k)) ? Math.abs(s[k].v - truthAt(truth[i], k)) : null)).filter(finite);
    return e.length ? e.reduce((u, v) => u + v, 0) / e.length : null;
  });
  const mx = Math.max(1e-12, ...stat.filter(finite).map(Math.abs));
  stat.forEach((v, k) => host.append(barRow({ name: tText("try.step_ahead", { h: k + 1 }), v: finite(v) ? Math.abs(v) : 0, max: mx, label: shown(v), selected: false, i: k, ci: null, cls: "" })));
  return { host, stat };
}

export const summary = Object.freeze({
  sentence(preds, truth) {
    if (!truth) return t("many.sentence.no_truth", { n: num(preds.length), h: Math.max(0, ...preds.map((p) => stepsOf(p).length)) });
    const { stat } = perStep(preds, truth), ok = stat.filter(finite);
    return t("many.sentence.truth", { e: num(shown(ok.length ? ok.reduce((u, v) => u + v, 0) / ok.length : null)), n: preds.length });
  },
  left(preds, truth) { return { title: t(truth ? "many.left.per_step" : "many.left.per_step.values"), body: perStep(preds, truth).host, caption: null }; },
  right() { return null; },
  columns() { return ["many.col.predicted", "many.col.truth", "many.col.warnings"]; },
  cells(p, truth) {
    return [stepsOf(p).map((x) => shown(x.v)).join(" · "), Array.isArray(truth) ? truth.map(shown).join(" · ") : finite(truth) ? shown(truth) : "—", warningCell(p)];
  },
  miss() { return false; },
});

// The weights read's columns: the attribution's steps ahead (the target: each step's value).
export const weightsAxis = (attribution, env) => strataAxis(attribution, env);

export const breakdown = (section, values, env) => breakdownBars(section, values, env);
