// The binary view key (binary, and a sequence classification of a binary base): one logit, sigmoid,
// and a threshold chosen on the validation part (/v1/schema output.threshold, probability scale).
// The answer's keys (D2's RowPrediction, R-C4): label (the decided class), probabilities{negative,
// positive}, logits{positive} on include, warnings.
//   try one       the class at the threshold in its role colour, p of the positive class; one bar
//                 with the threshold marked; what the threshold is
//   many at once  how many get their true class; the positive class's probability per true class
//                 against the threshold; the confusion at the threshold; #, true, predicted,
//                 probability, warnings
//   results       per class (recall per class, breakdown@1 per_class)
import { el } from "../../core/dom.js";
import { t, tText, data, cls } from "../../core/text.js";
import { num } from "../shared.js";
import { valueOf, pct, shown, scoreBar, histogram, confusion, breakdownBars, truthCell, answerCell, warningCell, roleCls, isAlert, namedAxis } from "./shared.js";

export const headLabel = "head.caption@binary";

// [negative, positive] of the lane: the output record's classes, the positive one second.
export function sidesOf(env) {
  const o = (env && env.output) || {}, classes = o.classes || [], positive = o.positive ?? classes[1] ?? null;
  const negative = classes.find((c) => c !== positive) ?? null;
  return { negative, positive, threshold: o.threshold };
}

const positiveP = (p, env) => valueOf(p.probabilities, sidesOf(env).positive);

export const output = Object.freeze({
  verdict(a, env) {
    const { positive } = sidesOf(env), v = positiveP(a.p, env);
    return el("span", {}, el("span", { class: "enter " + roleCls(env, a.p.label) }, cls(a.p.label)),
      el("span", { class: "p enter" }, t("try.p", { name: cls(positive), p: shown(v) })));
  },
  agree(a, preset) {
    if (!preset || preset.expected === null || preset.expected === undefined) return null;
    const ok = a.p.label === preset.expected;
    return el("span", { class: ok ? "acc" : "bad" }, ok ? tText("try.agree.yes") : tText("try.agree.no", { name: cls(preset.expected) }));
  },
  detail(a, env) {
    const { positive, threshold } = sidesOf(env), v = positiveP(a.p, env);
    const alert = a.p.label === positive && isAlert(env, positive);
    return el("div", { class: "bars" }, scoreBar({ name: tText("try.p.name", { name: cls(positive) }), v, threshold, cls: roleCls(env, a.p.label), fillCls: alert ? "bad" : "" }));
  },
  note(env) {
    const { positive, threshold } = sidesOf(env);
    return typeof threshold === "number" ? t("try.note.threshold", { t: shown(threshold), positive: cls(positive) }) : null;
  },
});

export const summary = Object.freeze({
  sentence(preds, truth) {
    if (!truth) return t("many.sentence.no_truth", { n: num(preds.length) });
    return t("many.sentence.truth", { agree: num(preds.filter((p, i) => p.label === truth[i]).length), n: preds.length });
  },
  left(preds, truth, env) {
    const { positive, threshold } = sidesOf(env), ps = preds.map((p) => positiveP(p, env));
    return { title: t("many.left.scores", { name: cls(positive) }), body: histogram(ps, { bins: 10, mark: threshold, groups: truth }),
             caption: typeof threshold === "number" ? t("many.left.scores.caption", { t: shown(threshold) }) : null };
  },
  right(preds, truth, env) {
    if (!truth) return null;
    const { negative, positive } = sidesOf(env);
    return { title: t("many.confusion"), body: confusion([negative, positive], truth, preds.map((p) => p.label)) };
  },
  columns() { return ["many.col.truth", "many.col.predicted", "many.col.probability", "many.col.warnings"]; },
  cells(p, truth, env) { return [truthCell(truth), answerCell(p.label), pct(positiveP(p, env)), warningCell(p)]; },
  miss(p, truth) { return truth !== null && truth !== undefined && p.label !== truth; },
});

// The weights read's columns: the negative and the positive class, each the attribution's column of
// that true class (the target: the positive logit; the head has that one output).
export function weightsAxis(attribution, env) {
  const { negative, positive } = sidesOf(env);
  return namedAxis([negative, positive].filter((c) => c !== null), attribution, env);
}

export const breakdown = (section, values, env) => breakdownBars(section, values, env);
