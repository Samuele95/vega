// The classes view key (multiclass, and a sequence classification of a multiclass base): the answer
// is one class, softmax over the head's scores. The answer's keys (D2's RowPrediction, R-C4):
// label, probabilities{class: p}, logits{} on include, warnings.
//   try one       the class in its role colour and its score; the five highest scores; what the
//                 scores mean (they add up to one; their calibration only as the story's values
//                 state it, never a claim typed here)
//   many at once  how many get their true class; per true class right and wrong; the most frequent
//                 mix-ups; the table #, true, predicted, score, warnings
//   results       per class (breakdown@1 per_class)
import { el } from "../../core/dom.js";
import { t, tText, data, cls } from "../../core/text.js";
import { num } from "../shared.js";
import { valueOf, pct, probabilityBars, rightWrongBars, countBars, mixups, breakdownBars, truthCell, answerCell, warningCell, roleCls, namedAxis } from "./shared.js";

export const headLabel = "head.caption@classes";

export const output = Object.freeze({
  verdict(a, env) {
    const p = a.p, v = valueOf(p.probabilities, p.label);
    return el("span", {}, el("span", { class: "enter " + roleCls(env, p.label) }, cls(p.label)),
      el("span", { class: "p enter" }, t("try.score", { p: v === null ? "—" : (100 * v).toFixed(1) })));
  },
  agree(a, preset, env) {
    if (!preset || preset.expected === null || preset.expected === undefined) return null;
    const ok = a.p.label === preset.expected;
    return el("span", { class: ok ? "acc" : "bad" }, ok ? tText("try.agree.yes") : tText("try.agree.no", { name: cls(preset.expected) }));
  },
  detail(a, env) { return probabilityBars(a.p.probabilities, { chosen: [a.p.label], env }); },
  note(env) {
    const ece = env && env.story && env.story.values ? env.story.values.ece : undefined;
    return el("span", {}, t("try.note.classes"), ece === undefined ? "" : " ", ece === undefined ? "" : t("try.note.calibration", { ece }));
  },
});

export const summary = Object.freeze({
  sentence(preds, truth) {
    if (!truth) return t("many.sentence.no_truth", { n: num(preds.length) });
    const agree = preds.filter((p, i) => p.label === truth[i]).length;
    return t("many.sentence.truth", { agree: num(agree), n: preds.length });
  },
  left(preds, truth, env) {
    if (!truth) return { title: t("many.left.predicted"), body: countBars(preds.map((p) => p.label), env), caption: null };
    return { title: t("many.left.by_truth"), body: rightWrongBars(null, truth, (i) => preds[i].label === truth[i]), caption: t("many.left.by_truth.caption") };
  },
  right(preds, truth, env) {
    const body = truth ? mixups(truth, preds.map((p) => p.label), env) : null;
    return body ? { title: t("many.mix"), body } : null;
  },
  columns() { return ["many.col.truth", "many.col.predicted", "many.col.score", "many.col.warnings"]; },
  cells(p, truth) { return [truthCell(truth), answerCell(p.label), pct(valueOf(p.probabilities, p.label)), warningCell(p)]; },
  miss(p, truth) { return truth !== null && truth !== undefined && p.label !== truth; },
});

// The weights read's columns: one per class, in the network's order, coloured by role; each reads
// the attribution's column of that class (target: the class's logit).
export const weightsAxis = (attribution, env) => namedAxis((env && env.output && env.output.classes) || [], attribution, env);

export const breakdown = (section, values, env) => breakdownBars(section, values, env);
