// The ordinal view key (ordinal, and a sequence regression of an ordinal base): K ordered classes
// read from K−1 cumulative logits, P(y > j) = σ(z_j), and the answer is the number of steps passed
// (R-C6: cumulative_count). The answer's keys (D2's RowPrediction, R-C4): label, probabilities
// {class: p}, cumulative{step: p}, logits{} on include, warnings.
//   try one       the class; the chance of passing each step with the half-way mark, then the
//                 classes' probabilities
//   many at once  exact, one class off, further; the confusion with its diagonal band; #, true,
//                 predicted, distance, warnings
//   results       per class (breakdown@1 per_class)
import { el } from "../../core/dom.js";
import { barRow } from "../../core/bars.js";
import { t, tText, data, cls } from "../../core/text.js";
import { num } from "../shared.js";
import { valueOf, pct, finite, probabilityBars, confusion, breakdownBars, truthCell, answerCell, warningCell, roleCls, namedAxis } from "./shared.js";

export const headLabel = "head.caption@ordinal";

export const classesOf = (env) => (env && env.output && env.output.classes) || [];
// How many classes apart an answer is from the truth (null when either is not a class of the lane).
export function distance(env, label, truth) {
  const cs = classesOf(env), a = cs.indexOf(label), b = cs.indexOf(truth);
  return a < 0 || b < 0 ? null : Math.abs(a - b);
}

// The cumulative bars: one per step, P(y > j), with the half-way line.
function steps(p) {
  const host = el("div", { class: "bars" });
  Object.entries(p.cumulative || {}).filter(([, v]) => finite(v)).forEach(([name, v], i) => {
    const row = barRow({ name: tText("try.step_past", { name: cls(name) }), v, max: 1, label: pct(v), selected: v >= .5, i, ci: null, cls: "" });
    row.querySelector(".track").append(el("div", { class: "ref", style: "left:50%", title: tText("try.half") }));
    host.append(row);
  });
  return host;
}

export const output = Object.freeze({
  verdict(a, env) {
    const v = valueOf(a.p.probabilities, a.p.label);
    return el("span", {}, el("span", { class: "enter " + roleCls(env, a.p.label) }, cls(a.p.label)),
      el("span", { class: "p enter" }, t("try.score", { p: v === null ? "—" : (100 * v).toFixed(1) })));
  },
  agree(a, preset, env) {
    if (!preset || preset.expected === null || preset.expected === undefined) return null;
    const d = distance(env, a.p.label, preset.expected);
    return el("span", { class: d === 0 ? "acc" : "bad" }, d === 0 ? tText("try.agree.yes") : tText("try.agree.off", { name: cls(preset.expected), d: d ?? "?" }));
  },
  detail(a, env) {
    return el("div", {}, steps(a.p), el("div", { class: "caps muted views-sub" }, t("try.classes")),
      probabilityBars(a.p.probabilities, { chosen: [a.p.label], env, k: classesOf(env).length || 12 }));
  },
  note() { return t("try.note.ordinal"); },
});

export const summary = Object.freeze({
  sentence(preds, truth, env) {
    if (!truth) return t("many.sentence.no_truth", { n: num(preds.length) });
    const d = preds.map((p, i) => distance(env, p.label, truth[i])).filter((x) => x !== null);
    return t("many.sentence.truth", { agree: num(d.filter((x) => x === 0).length), near: d.filter((x) => x === 1).length,
                                      far: d.filter((x) => x > 1).length, n: preds.length });
  },
  left(preds, truth, env) {
    if (!truth) return { title: t("many.left.predicted"), body: probabilityCounts(preds, env), caption: null };
    return { title: t("many.confusion"), body: confusion(classesOf(env), truth, preds.map((p) => p.label), { band: 1 }), caption: t("many.left.band.caption") };
  },
  right() { return null; },
  columns() { return ["many.col.truth", "many.col.predicted", "many.col.distance", "many.col.warnings"]; },
  cells(p, truth, env) { const d = distance(env, p.label, truth); return [truthCell(truth), answerCell(p.label), d === null ? "—" : String(d), warningCell(p)]; },
  miss(p, truth) { return truth !== null && truth !== undefined && p.label !== truth; },
});

// Counts of the answered classes, in the classes' order (an ordinal scale keeps its order).
function probabilityCounts(preds, env) {
  const cs = classesOf(env), counts = cs.map((c) => preds.filter((p) => p.label === c).length), mx = Math.max(1, ...counts);
  const host = el("div", { class: "bars" });
  cs.forEach((c, i) => host.append(barRow({ name: cls(c), v: counts[i], max: mx, label: String(counts[i]), selected: false, i, ci: null, cls: "" })));
  return host;
}

// The weights read's columns: the classes in their order, each the attribution's column of that true
// class (the target: the expected rank).
export const weightsAxis = (attribution, env) => namedAxis(classesOf(env), attribution, env);

export const breakdown = (section, values, env) => breakdownBars(section, values, env);
