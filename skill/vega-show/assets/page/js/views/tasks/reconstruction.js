// The reconstruction view key (unsupervised, goal reconstruct): the network rebuilds its inputs,
// and the answer is how far the copy lands from them. The answer's keys (D2's RowPrediction, R-C4):
// reconstruction_error, reconstruction{input: x̂} on include, warnings. The error is read against
// the demo sample's own errors (page.json#reference, the training part).
//   try one       the error and where it falls among the training rows' errors; the inputs rebuilt
//                 worst, x against x̂ (from the explanation's inputs and output)
//   many at once  how many were rebuilt and the median error; the errors' histogram; the rows
//                 rebuilt worst; #, error, warnings
//   results       per input (breakdown@1 per_feature)
import { el } from "../../core/dom.js";
import { barRow } from "../../core/bars.js";
import { t, tText, data } from "../../core/text.js";
import { num } from "../shared.js";
import { reconstructionOf } from "../payloads.js";
import { finite, shown, median, histogram, breakdownBars, warningCell, strataAxis } from "./shared.js";

export const headLabel = "head.caption@reconstruction";

// Where a row score falls among the reference quantiles (page.json#reference): the key of the band.
export function band(reference, v) {
  const q = reference && reference.quantiles;
  if (!q || !finite(v)) return null;
  const at = (k) => (finite(q[k]) ? q[k] : Infinity);
  return v >= at("0.99") ? "try.band.p99" : v >= at("0.9") ? "try.band.p90" : v >= at("0.5") ? "try.band.p50" : "try.band.below";
}

// The inputs an explanation rebuilt worst: bars of (x̂ − x)², the k largest, x and x̂ beside.
export function worstInputs(ex, k = 8) {
  const r = reconstructionOf(ex), host = el("div", { class: "bars" });
  if (!r) return host;
  const err = r.x.map((x, i) => [i, (r.xhat[i] - x) ** 2]).sort((a, b) => b[1] - a[1]).slice(0, k), mx = Math.max(1e-12, ...err.map((e) => e[1]));
  err.forEach(([i, e], n) => host.append(barRow({ name: data(r.names[i] ?? String(i)), v: e, max: mx,
    label: tText("try.rebuilt", { x: shown(r.x[i]), xhat: shown(r.xhat[i]) }), selected: n === 0, i: n, ci: null, cls: "" })));
  return host;
}

export const errorOf = (p) => (p ? (finite(p.reconstruction_error) ? p.reconstruction_error : finite(p.score) ? p.score : null) : null);

export const output = Object.freeze({
  verdict(a, env) {
    const e = errorOf(a.p), key = band(env && env.page ? env.page.reference : null, e);
    return el("span", {}, el("span", { class: "enter role-plain" }, t("try.error", { e: shown(e) })),
      key ? el("span", { class: "p enter" }, t(key)) : null);
  },
  agree() { return null; },
  detail(a) { return worstInputs(a.ex); },
  note() { return t("try.note.reconstruction"); },
});

// The rows rebuilt worst: their position in the batch and their error.
function worstRows(preds, k = 10) {
  const order = preds.map((p, i) => [i, errorOf(p)]).filter(([, e]) => finite(e)).sort((a, b) => b[1] - a[1]).slice(0, k);
  if (!order.length) return null;
  const host = el("div", { class: "mix" });
  order.forEach(([i, e]) => host.append(el("div", {}, t("many.row", { i: i + 1 })), el("div", { class: "n" }, shown(e))));
  return host;
}

export const summary = Object.freeze({
  sentence(preds) { return t("many.sentence.no_truth", { n: num(preds.length), e: shown(median(preds.map(errorOf))) }); },
  left(preds, truth, env) {
    const ref = env && env.page && env.page.reference, mark = ref && ref.quantiles ? ref.quantiles["0.99"] : null;
    return { title: t("many.left.errors"), body: histogram(preds.map(errorOf), { mark }), caption: finite(mark) ? t("many.left.errors.caption") : null };
  },
  right(preds) { const body = worstRows(preds); return body ? { title: t("many.right.worst"), body } : null; },
  columns() { return ["many.col.error", "many.col.warnings"]; },
  cells(p) { return [shown(errorOf(p)), warningCell(p)]; },
  miss() { return false; },
});

// The weights read's columns: the attribution's one group, every row (the target: the
// reconstruction error).
export const weightsAxis = (attribution, env) => strataAxis(attribution, env);

export const breakdown = (section, values, env) => breakdownBars(section, values, env);
