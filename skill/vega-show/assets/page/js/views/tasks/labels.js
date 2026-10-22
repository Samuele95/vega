// The labels view key (multilabel, and a sequence classification of a multilabel base): one logit
// per label, each through its own sigmoid and its own threshold (/v1/schema output.threshold_per_label,
// else output.threshold). The answer's keys (D2's RowPrediction, R-C4): labels[] (the labels at or
// above their threshold), probabilities{label: p}, logits{} on include, warnings. The truth is a
// list of label names.
//   try one       the predicted label set as chips in their role colours; every label's p with its
//                 threshold; the labels are scored independently
//   many at once  how many get every label right; per label hit, missed and added; the most
//                 frequent errors; #, true labels, predicted, warnings
//   results       per label (breakdown@1 per_label)
import { el } from "../../core/dom.js";
import { fmtInt } from "../../core/fmt.js";
import { t, tText, data, cls } from "../../core/text.js";
import { num } from "../shared.js";
import { probabilityBars, breakdownBars, warningCell, roleCls, finite, median, namedAxis } from "./shared.js";

export const headLabel = "head.caption@labels";

export const namesOf = (env) => (env && env.output && env.output.classes) || [];
// The threshold of one label: its own, else the lane's.
export function thresholdOf(env, name) {
  const o = (env && env.output) || {};
  return o.perLabel && finite(o.perLabel[name]) ? o.perLabel[name] : o.threshold;
}
const setOf = (v) => new Set(Array.isArray(v) ? v.map(String) : []);
const same = (a, b) => a.size === b.size && [...a].every((x) => b.has(x));
const chips = (names, env) => (names.length ? names.map((n) => el("span", { class: "views-chip " + roleCls(env, n) }, cls(n)))
  : [el("span", { class: "muted" }, t("try.labels.none"))]);

export const output = Object.freeze({
  verdict(a, env) { return el("span", { class: "views-chips enter" }, ...chips(a.p.labels || [], env)); },
  agree(a, preset) {
    if (!preset || !Array.isArray(preset.expected)) return null;
    const ok = same(setOf(a.p.labels), setOf(preset.expected));
    return el("span", { class: ok ? "acc" : "bad" }, ok ? tText("try.agree.yes") : tText("try.agree.no", { name: preset.expected.map((x) => cls(x)).join(", ") || "∅" }));
  },
  detail(a, env) {
    const names = Object.keys(a.p.probabilities || {});
    return probabilityBars(a.p.probabilities, { chosen: a.p.labels || [], env, k: names.length, thresholds: (n) => thresholdOf(env, n) });
  },
  note() { return t("try.note.independent"); },
});

// Per label: how many rows had it and the network found it (hit), had it and missed it, and had it
// not and the network added it.
function perLabel(preds, truth, env) {
  const host = el("div", { class: "bars" }), names = namesOf(env);
  const rows = names.map((n) => {
    const c = { hit: 0, missed: 0, added: 0 };
    preds.forEach((p, i) => { const got = setOf(p.labels).has(n), had = setOf(truth[i]).has(n); if (got && had) c.hit++; else if (had) c.missed++; else if (got) c.added++; });
    return [n, c];
  });
  const mx = Math.max(1, ...rows.map(([, c]) => c.hit + c.missed + c.added));
  rows.forEach(([n, c], i) => {
    const w = (v) => `${v / mx * 100}%`;
    host.append(el("div", { class: "bar" + (c.missed + c.added ? " weak" : "") }, el("div", { class: "name", title: cls(n) }, cls(n)),
      el("div", { class: "track" }, el("div", { class: "fill", style: `width:${w(c.hit)};background:var(--accent);animation-delay:${i * .04}s` }),
        el("div", { class: "stack", style: `left:${w(c.hit)};width:${w(c.missed)};background:var(--alert)` }),
        el("div", { class: "stack", style: `left:${w(c.hit + c.missed)};width:${w(c.added)};background:var(--neg)` })),
      el("div", { class: "val" }, `${c.hit} · ${c.missed} · ${c.added}`)));
  });
  return host;
}

// The most frequent single errors: a label missed or added, with how often.
function errors(preds, truth) {
  const n = new Map();
  preds.forEach((p, i) => {
    const got = setOf(p.labels), had = setOf(truth[i]);
    had.forEach((x) => { if (!got.has(x)) n.set(`− ${x}`, (n.get(`− ${x}`) || 0) + 1); });
    got.forEach((x) => { if (!had.has(x)) n.set(`+ ${x}`, (n.get(`+ ${x}`) || 0) + 1); });
  });
  const top = [...n.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  if (!top.length) return null;
  const host = el("div", { class: "mix" });
  top.forEach(([k, c]) => host.append(el("div", {}, cls(k)), el("div", { class: "n" }, c)));
  return host;
}

export const summary = Object.freeze({
  sentence(preds, truth) {
    if (!truth) return t("many.sentence.no_truth", { n: num(preds.length), mean: median(preds.map((p) => (p.labels || []).length)) ?? 0 });
    const exact = preds.filter((p, i) => same(setOf(p.labels), setOf(truth[i]))).length;
    return t("many.sentence.truth", { agree: num(exact), n: preds.length });
  },
  left(preds, truth, env) {
    if (truth) return { title: t("many.left.per_label"), body: perLabel(preds, truth, env), caption: t("many.left.per_label.caption") };
    const counts = new Map(namesOf(env).map((n) => [n, 0]));
    preds.forEach((p) => (p.labels || []).forEach((x) => counts.set(x, (counts.get(x) || 0) + 1)));
    const host = el("div", { class: "bars" }), mx = Math.max(1, ...counts.values());
    [...counts.entries()].forEach(([x, c], i) => host.append(el("div", { class: "bar" }, el("div", { class: "name", title: cls(x) }, cls(x)),
      el("div", { class: "track" }, el("div", { class: "fill", style: `width:${c / mx * 100}%;animation-delay:${i * .04}s` })), el("div", { class: "val" }, fmtInt(c)))));
    return { title: t("many.left.predicted"), body: host, caption: null };
  },
  right(preds, truth) {
    const body = truth ? errors(preds, truth) : null;
    return body ? { title: t("many.right.errors"), body } : null;
  },
  columns() { return ["many.col.truth", "many.col.predicted", "many.col.warnings"]; },
  cells(p, truth) {
    return [Array.isArray(truth) ? truth.map((x) => cls(x)).join(", ") : "—", (p.labels || []).map((x) => cls(x)).join(", "), warningCell(p)];
  },
  miss(p, truth) { return Array.isArray(truth) && !same(setOf(p.labels), setOf(truth)); },
});

// The weights read's columns: one per label, each the attribution's column of that label (the
// target: the label's own logit).
export const weightsAxis = (attribution, env) => namedAxis(namesOf(env), attribution, env);

export const breakdown = (section, values, env) => breakdownBars(section, values, env);
