// The results view (D4 §6.3): the story's headline; the models compared on the sealed test part as
// bars with their bootstrap interval brackets (model.<name>; a min-direction metric keeps the bars and
// says lower is better); the interval's level from the story (values.ci_level, never typed); the
// facts (metric.<name>); the breakdown the task module draws from the story's breakdown@1 (the weak
// marks are the story's); a verdict that is not assessable (A30: earned null), said in words with the
// evaluation's reason and never drawn as a win or a loss (no "lower is better" over scales that differ);
// the verdict in every state the evaluation records (earned, not earned, beaten by a baseline: the
// strongest baseline by name and the paired difference with its interval, never a softer word); the
// limits.
import { $, el } from "../core/dom.js";
import { S } from "../core/state.js";
import { fmtNum, fmtCi, fmtVal, scalar } from "../core/fmt.js";
import { barRow } from "../core/bars.js";
import { t, tText, hasKey, storyText, data } from "../core/text.js";
import { section } from "./shared.js";

const labelOf = (m) => tText(`model.${m.name}`);
// a fact's label: its metric's name, or for n_test the sealed test part's size in the lexicon's words
const factLabel = (f) => (f.metric === "n_test" ? tText("results.fact.n_test") : tText(`metric.${f.metric}`));

// The story's verdict is not assessable: earned null, with its reason's values key (story@2).
export const notAssessable = (sec) => !!(sec.verdict && sec.verdict.earned === null && sec.verdict.reason);

function models(sec, vals, direction) {
  const ms = (sec.models || []).map((m) => ({ ...m, label: labelOf(m), v: vals[m.value] })).filter((m) => m.v !== undefined);
  const mx = Math.max(1e-9, ...ms.map((m) => (m.v && m.v.ci ? m.v.ci[1] : scalar(m.v)))) * 1.04;
  $("res-models").replaceChildren(...ms.map((m, i) => barRow({ name: m.label, v: scalar(m.v), max: mx, label: fmtVal(m.v), selected: m.selected, i, ci: m.v && m.v.ci })));
  if (direction === "min" && !notAssessable(sec)) $("res-models").prepend(el("div", { class: "scale" }, el("span", {}, "0"), el("span", {}, t("results.scale@min"))));
}

// The state of the story's verdict: earned | beaten | not_earned | not_assessable, or null when the
// story carries none (story@2 results.verdict; resolve_story writes it as the evaluation recorded it).
export function verdictState(sec) {
  const v = sec && sec.verdict;
  if (!v) return null;
  if (notAssessable(sec)) return "not_assessable";
  if (v.earned === true) return "earned";
  if (v.earned === false) return v.beaten_by_baseline === true ? "beaten" : "not_earned";
  return null;
}

// The paired difference the verdict rests on, as words: its value and interval, or "not recorded".
function diffWords(v) {
  if (v === undefined || v === null) return tText("results.verdict.no_diff");
  const ci = fmtCi(v);
  return ci ? `${fmtNum(scalar(v))} ${ci}` : fmtNum(scalar(v));
}

// The verdict's sentence ({baseline}, {diff}) for the results view and its short form for the start
// view, or null without a verdict.
export function verdictWords(sec, vals) {
  const state = verdictState(sec);
  if (!state) return null;
  if (state === "not_assessable") {
    const reason = typeof vals[sec.verdict.reason] === "string" ? data(vals[sec.verdict.reason]) : tText("results.verdict.no_diff");
    return { state, sentence: t("results.verdict.not_assessable", { reason }), word: tText("results.verdict.word.not_assessable"), baseline: null, diff: null };
  }
  const v = sec.verdict, baseline = v.strongest_baseline ? tText(`model.${v.strongest_baseline}`) : tText("results.verdict.no_baseline");
  const diff = diffWords(v.diff ? vals[v.diff] : null);
  return { state, sentence: t(`results.verdict.${state}`, { baseline, diff }), word: tText(`results.verdict.word.${state}`), baseline, diff };
}

function verdict(sec, vals) {
  const host = $("res-verdict"), w = verdictWords(sec, vals);
  host.hidden = !w;
  if (w) host.setAttribute("data-verdict", w.state); else host.removeAttribute("data-verdict");
  host.replaceChildren(...(w ? [w.sentence] : []));
}

function render() {
  if ($("res-models").childElementCount) return;
  const sec = section("results"), vals = (S.story && S.story.values) || {}, direction = S.page && S.page.metric ? S.page.metric.direction : "max";
  if (sec.headline) storyText(sec.headline, $("res-headline"), true); else $("res-headline").textContent = tText("results.no_story");
  $("res-ci-note").removeAttribute("data-t");
  $("res-ci-note").replaceChildren(typeof vals.ci_level === "number" ? t("results.ci_note", { ci_level: vals.ci_level }) : t("results.ci_note.no_level"));
  models(sec, vals, direction);
  verdict(sec, vals);
  $("res-facts").replaceChildren(...(sec.facts || []).filter((f) => vals[f.value] !== undefined).map((f) => el("div", { class: "fact" },
    el("span", { class: "v" }, fmtNum(scalar(vals[f.value])), fmtCi(vals[f.value]) ? el("small", {}, fmtCi(vals[f.value])) : null),
    el("span", { class: "l caps" }, factLabel(f)), f.note ? storyText(f.note, el("div", { class: "note" })) : null)));
  const b = sec.breakdown || null, drawn = b && S.mods ? S.mods.task.breakdown(b, vals, S.env) : null;
  const toggle = document.querySelector(".disclose[data-for=\"res-class\"]");
  if (toggle) toggle.hidden = !drawn;
  $("res-class").hidden = !drawn;
  if (drawn) {
    // the breakdown's own metric, named by the page whatever its caption says (it may differ from
    // the primary: a per-horizon error under a MASE primary, a per-label PR-AUC under a micro F-score)
    const metric = b.metric ? (hasKey(`metric.${b.metric}`) ? tText(`metric.${b.metric}`) : data(b.metric)) : null;
    $("res-class-cap").replaceChildren(...(metric ? [el("b", { class: "res-class-metric" }, t("results.breakdown.metric", { metric })), " "] : []));
    if (b.caption) storyText(b.caption, $("res-class-cap"));
    $("res-per-class").replaceChildren(drawn);
  }
  // a limit that names the record's own words (a warning's line, the ship stance, the classes with
  // few test rows) quotes them beside the authored text, as data
  $("res-limits").replaceChildren(...(sec.limits || []).map((l) => el("div", {}, storyText(l.title, el("b")), storyText(l.text),
    l.recorded ? el("q", { class: "res-limit-recorded", "data-verbatim": "" }, data(l.recorded)) : null)));
}

export const view = Object.freeze({ id: "results", show: render });
