// The start view, the threshold (D4 §6.3): the story's headline and lead (without a story, the
// bundle's title and description, page.json#identity), three facts — the network's weights, the size
// of its answer in the lane's words, the primary metric on the sealed test part with its interval and
// the evaluation's verdict beside it —
// then begin (continue the tour, start again) and the keys to jump with.
import { $, el } from "../core/dom.js";
import { S } from "../core/state.js";
import { fmtInt, fmtNum, fmtCi, scalar } from "../core/fmt.js";
import { t, tText, tn, data, storyText } from "../core/text.js";
import { section } from "./shared.js";
import { verdictWords } from "./results.js";

// How many things the network answers, in the lane's words: its classes or labels, the inputs it
// rebuilds or measures, the steps ahead; null when the answer is one value.
export function outputCount(env) {
  const vk = env.lane.viewKey, g = env.graph, o = env.output || {};
  if (vk === "value") return null;
  if (vk === "reconstruction" || vk === "anomaly") return g ? g.inputs.names.length : null;
  if (vk === "horizon") return o.horizon || (g && g.sample ? g.sample.horizon : null);
  const names = o.classes || (g ? g.task.outputs.names : null);
  return Array.isArray(names) ? names.length : null;
}

function facts(env) {
  const out = [], st = S.story, g = env.graph, n = outputCount(env);
  const params = g ? (typeof g.params === "number" ? g.params : g.params.total) : null;
  if (typeof params === "number") out.push([fmtInt(params), t("start.fact.params")]);
  if (typeof n === "number") out.push([fmtInt(n), tn("start.fact.outputs", n)]);
  const primary = st && st.values ? st.values.primary : null, metric = S.page ? S.page.metric.primary : null;
  if (primary && metric) out.push([fmtNum(scalar(primary)), t("start.fact.sealed", { metric: tText(`metric.${metric}`) }), fmtCi(primary)]);
  // beside the primary, the evaluation's verdict in its own word (earned, not earned, beaten), the
  // strongest baseline and the paired difference: a figure is never shown without what it earned
  const w = verdictWords(section("results"), (st && st.values) || {});
  if (w) out.push([w.word, w.baseline ? t("start.fact.verdict", { baseline: w.baseline }) : t("start.fact.verdict.alone"), w.state === "not_assessable" ? "" : w.diff]);
  return out;
}

// Why a view of the tour is not shown (page.json#views.off: {view: reason key}; views_available).
export const OFF_REASONS = Object.freeze(["views.off.no_section", "views.off.no_examples", "views.off.no_route"]);
function offNotes() {
  const off = (S.page && S.page.views && S.page.views.off) || {};
  return Object.entries(off).map(([id, why]) => el("p", { class: "hint views-off" },
    t("views.off", { view: tText(`view.${id}`), why: OFF_REASONS.includes(why) ? tText(why) : data(why) })));
}

// The keys to jump with: 1 to the last tour view, the index, the code, back here. A digit keycap
// shows the digit; the letter keycaps and the escape key's name are words of the table.
export function renderKeys() {
  const kbd = (k) => el("kbd", {}, k), n = S.tour ? S.tour.views.length : 0;
  if (!n) { $("start-keys").replaceChildren(); return; }
  $("start-keys").replaceChildren(t("start.keys", { k_first: kbd("1"), k_last: kbd(String(n)),
    k_index: kbd(tText("app.kbd.index")), k_code: kbd(tText("app.kbd.code")), k_esc: kbd(tText("app.kbd.esc")) }));
}

export function show(env) {
  const st = S.story, identity = (S.page && S.page.identity) || {}, views = S.tour.views;
  $("headline").replaceChildren(); $("lead").replaceChildren();
  if (st && st.headline) storyText(st.headline, $("headline"), true); else $("headline").textContent = data(identity.title || document.title);
  if (st && st.lead) storyText(st.lead, $("lead")); else $("lead").textContent = data(identity.description || "");
  $("start-facts").replaceChildren(...facts(env).map(([v, l, ci]) => el("div", { class: "fact" },
    el("span", { class: "v" }, v, ci ? el("small", {}, ci) : null), el("span", { class: "l caps" }, l))));
  const resume = views.find((id) => !S.seen.has(id));
  $("begin").replaceChildren(t(S.seen.size ? (resume ? "start.continue" : "start.again") : "start.begin"), " ", el("span", { class: "arrow" }, "→"));
  $("begin").onclick = () => S.tour.show(resume || views[0]);
  renderKeys();
  $("start-keys").append(...offNotes());
}

export const view = Object.freeze({ id: "start", show });
