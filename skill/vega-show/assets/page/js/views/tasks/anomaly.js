// The anomaly view key: a score (the reconstruction error, or the distance to the SVDD centre) and a
// threshold chosen on the validation part (/v1/schema output.threshold, score scale); a row at or
// above it is flagged. The answer's keys (D2's RowPrediction, R-C4): score, anomalous, label when the
// data had one, reconstruction{} or embedding[] on include, warnings. The two sides are the task's
// classes when a label exists (the positive one anomalous), else normal and anomalous; the anomalous
// side is alert by task (R-E5), the normal side neutral.
//   try one       unusual or typical in the side's role colour, the score against the threshold;
//                 the score bar; with a reconstruction score, the inputs rebuilt worst
//   many at once  how many were flagged; the scores' histogram with the threshold; the highest
//                 scores (and how many of them are truly anomalous when the truth is known);
//                 #, true, score, flag, warnings
//   results       at k (breakdown@1 at_k)
import { el } from "../../core/dom.js";
import { t, tText, data, cls } from "../../core/text.js";
import { num } from "../shared.js";
import { finite, shown, scoreBar, histogram, breakdownBars, truthCell, warningCell, anomalySides, strataAxis } from "./shared.js";
import { worstInputs } from "./reconstruction.js";

export const headLabel = "head.caption@anomaly";

// {normal, anomalous, named}: the sides' names (the task's classes, the positive one anomalous), or
// the role map's own sides when the task has no label (tasks/shared.js anomalySides).
export const sidesOf = anomalySides;
const sideCls = (env, flagged) => { const s = sidesOf(env), name = flagged ? s.anomalous : s.normal; return env && env.roles ? env.roles.cls(name) : ""; };
const flagOf = (p, env) => (typeof p.anomalous === "boolean" ? p.anomalous : finite(p.score) && finite(env.output.threshold) ? p.score >= env.output.threshold : null);
// Whether a threshold exists to flag by: the service's own flag on the answers, or the record's
// threshold (an unlabelled lane whose evaluation chose none, threshold.applies false, has neither).
const flags = (preds, env) => preds.some((p) => flagOf(p, env) !== null);
// A truth list that holds a truth (payloads.truthOrNull: a list of nulls is no truth).
const known = (truth) => Array.isArray(truth) && truth.some((v) => v !== null && v !== undefined);
// Whether a truth says anomalous (a class name of the anomalous side, or a flag).
const truthFlag = (truth, env) => (truth === null || truth === undefined ? null : typeof truth === "boolean" ? truth : String(truth) === String(sidesOf(env).anomalous));

export const output = Object.freeze({
  verdict(a, env) {
    const f = flagOf(a.p, env);
    if (f === null) return el("span", {}, el("span", { class: "p enter" }, t("anomaly.no_threshold.verdict", { s: shown(a.p.score) })));
    return el("span", {}, el("span", { class: "enter " + sideCls(env, f === true) }, t(f ? "anomaly.flag.on" : "anomaly.flag.off")),
      el("span", { class: "p enter" }, t("try.score_against", { s: shown(a.p.score), t: shown(env.output.threshold) })));
  },
  agree(a, preset, env) {
    if (!preset || preset.expected === null || preset.expected === undefined) return null;
    const want = truthFlag(preset.expected, env), got = flagOf(a.p, env), ok = want === got;
    if (got === null) return null;
    return el("span", { class: ok ? "acc" : "bad" }, ok ? tText("try.agree.yes") : tText(want ? "try.agree.missed" : "try.agree.raised"));
  },
  detail(a, env) {
    const s = a.p.score, th = env.output.threshold, hi = Math.max(s || 0, th || 0) * 1.25 || 1;
    return el("div", {}, el("div", { class: "bars" }, scoreBar({ name: tText("try.score.name"), v: s, lo: 0, hi, threshold: th, cls: sideCls(env, flagOf(a.p, env) === true),
      fillCls: flagOf(a.p, env) ? "bad" : "" })), env.lane.scoreSource === "reconstruction" ? worstInputs(a.ex) : null);
  },
  note(env) { return finite(env.output.threshold) ? t("try.note.threshold", { t: shown(env.output.threshold), positive: t("anomaly.flag.on") }) : null; },
});

// The highest scores, flagged ones in the alert colour; with a truth, how many of the top k are truly
// anomalous (precision at k).
function topScores(preds, truth, env, k = 10) {
  const order = preds.map((p, i) => [i, p.score]).filter(([, s]) => finite(s)).sort((a, b) => b[1] - a[1]).slice(0, k);
  if (!order.length) return null;
  if (!known(truth)) truth = null;
  const host = el("div", { class: "mix" });
  order.forEach(([i, s]) => host.append(el("div", { class: truth && truthFlag(truth[i], env) ? "bad" : "" }, t("many.row", { i: i + 1 })), el("div", { class: "n" }, shown(s))));
  if (truth) host.append(el("div", { class: "muted" }, t("many.precision_at_k", { k: order.length, hit: order.filter(([i]) => truthFlag(truth[i], env)).length })), el("div"));
  return host;
}

export const summary = Object.freeze({
  sentence(preds, truth, env) {
    if (!flags(preds, env)) return t("anomaly.no_threshold.sentence", { n: num(preds.length) });
    return t("many.sentence.no_truth", { flagged: num(preds.filter((p) => flagOf(p, env)).length), n: preds.length });
  },
  left(preds, truth, env) {
    const groups = known(truth) ? truth.map((x) => (truthFlag(x, env) ? sidesOf(env).anomalous : sidesOf(env).normal)) : null;
    return { title: t("many.left.scores.anomaly"), body: histogram(preds.map((p) => p.score), { mark: env.output.threshold, groups }),
             caption: finite(env.output.threshold) ? t("many.left.scores.caption", { t: shown(env.output.threshold) }) : null };
  },
  right(preds, truth, env) { const body = topScores(preds, truth, env); return body ? { title: t("many.right.top"), body } : null; },
  columns() { return ["many.col.truth", "many.col.score", "many.col.flag", "many.col.warnings"]; },
  cells(p, truth, env) { const f = flagOf(p, env); return [truthCell(truth), shown(p.score), f === null ? "—" : tText(f ? "anomaly.flag.on" : "anomaly.flag.off"), warningCell(p)]; },
  miss(p, truth, env) { const want = env ? truthFlag(truth, env) : null, f = env ? flagOf(p, env) : null; return want !== null && f !== null && want !== f; },
});

// The weights read's columns: the attribution's groups, the two sides when the data has a label,
// else one column, in their roles (the target: the error or the distance).
export const weightsAxis = (attribution, env) => strataAxis(attribution, env, { roles: true });

export const breakdown = (section, values, env) => breakdownBars(section, values, env);
export const sideName = (env, flagged) => cls(flagged ? sidesOf(env).anomalous : sidesOf(env).normal);
