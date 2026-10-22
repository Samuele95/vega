// What was tried (D4 §6.3), inside the network view: the story's experiment runs in the story's order
// (D5 sorts them in the metric's direction), as bars scaled to [0, max · 1.04]; the scale's legend in
// the primary metric's direction (page.json#metric.direction: higher or lower is better); the
// reference run as a dashed line, and each other run's signed difference from it.
import { $, el } from "../core/dom.js";
import { S } from "../core/state.js";
import { fmtNum, scalar, signed } from "../core/fmt.js";
import { barRow } from "../core/bars.js";
import { t, tText, tnText, storyText, storyPlain } from "../core/text.js";
import { section } from "./shared.js";

// The legend of the scale: the direction of the lane's primary metric.
export const scaleKey = (direction) => (direction === "min" ? "experiments.scale@min" : "experiments.scale@max");

// Whether the story tried anything to show: without runs the disclosure is not offered at all.
export const hasRuns = () => (section("experiments").runs || []).length > 0;

export function renderExperiments() {
  const exp = section("experiments"), runs = exp.runs || [], toggle = document.querySelector(".disclose[data-for=\"exp\"]");
  if (toggle) toggle.hidden = !runs.length;
  $("exp").hidden = !runs.length;
  if (!runs.length || $("exp-bars").childElementCount) return;
  const values = (S.story && S.story.values) || {}, direction = (S.page && S.page.metric && S.page.metric.direction) || (exp.metric && exp.metric.direction);
  $("exp-title").removeAttribute("data-t");
  $("exp-title").textContent = `${exp.title ? storyPlain(exp.title) : tText("net.tried")} · ${tnText("legacy.experiments.variants", runs.length)}`;
  if (exp.lead) storyText(exp.lead, $("exp-lead"));
  if (exp.caption) storyText(exp.caption, $("exp-cap"));
  const vals = runs.map((r) => scalar(values[r.value])), max = Math.max(...vals.filter((v) => typeof v === "number")) * 1.04;
  const refRun = runs.find((r) => r.reference), ref = refRun ? scalar(values[refRun.value]) : null;
  const host = $("exp-bars");
  host.append(el("div", { class: "scale" }, el("span", {}, "0"), el("span", {}, t(scaleKey(direction)))));
  const bars = el("div", { class: "bars" });
  host.append(bars);
  runs.forEach((r, i) => {
    const v = vals[i], shown = typeof v === "number" ? fmtNum(+v.toFixed(3)) : "—";
    const label = ref !== null && !r.reference && typeof v === "number" ? `${shown}  ${signed(v - ref)}` : shown;
    const row = barRow({ name: storyPlain(r.label), v: typeof v === "number" ? v : 0, max, label, selected: r.selected, i, ci: null, cls: "" });
    if (ref !== null) row.querySelector(".track").append(el("div", { class: "ref", style: `left:${ref / max * 100}%`, title: tText("experiments.reference") }));
    bars.append(row);
  });
  if (ref !== null) host.append(el("p", { class: "hint" }, t("experiments.reference_note", { label: storyPlain(refRun.label) })));
}
