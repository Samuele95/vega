// The context view (D4 §6.3): the story's context steps one at a time on a rail (a story@2 step names
// its title by id, context.step.<id>; a story@1 step carries its own), and on the last step the
// facts, counted up from the story's values (a list counts its members). Without a story, one step
// reads the bundle's description.
import { $, el, reduced } from "../core/dom.js";
import { S } from "../core/state.js";
import { countUp, scalar } from "../core/fmt.js";
import { t, tText, data, storyText } from "../core/text.js";
import { section, viewState } from "./shared.js";

function steps() {
  const sec = section("context");
  if (Array.isArray(sec.steps) && sec.steps.length) {
    return sec.steps.map((s) => ({ title: s.id ? tText(`context.step.${s.id}`) : data(s.title || ""), body: () => storyText(s.text, el("p")) }));
  }
  const about = (S.page && S.page.identity && S.page.identity.description) || "";
  return [{ title: tText("context.about"), body: () => el("p", {}, data(about)) }];
}

function render() {
  const st = viewState("context", { at: 0 }), all = steps(), sec = section("context");
  st.at = Math.min(st.at, all.length - 1);
  $("ctx-rail").replaceChildren(...all.map((s, i) => Object.assign(el("button", { class: i === st.at ? "on" : i < st.at ? "done" : "" },
    el("span", { class: "n" }, String(i + 1).padStart(2, "0")), s.title), { onclick: () => { st.at = i; render(); } })));
  const s = all[st.at], last = st.at >= all.length - 1;
  $("ctx-step").replaceChildren(el("div", { class: "enter" }, el("h2", {}, s.title), s.body()));
  $("ctx-next").replaceChildren(t(last ? "context.to_network" : "context.next"), " ", el("span", { class: "arrow" }, "→"));
  $("ctx-next").onclick = advance;
  const facts = $("ctx-facts");
  if (last && Array.isArray(sec.facts) && sec.facts.length) {
    facts.hidden = false; facts.replaceChildren();
    sec.facts.forEach((f, i) => {
      const v = ((S.story && S.story.values) || {})[f.value], span = el("span", { class: "v" }, "");
      facts.append(el("div", { class: "fact enter", style: `animation-delay:${.1 + i * .12}s` }, span, storyText(f.label, el("span", { class: "l caps" }))));
      countUp(span, Array.isArray(v) ? v.length : scalar(v), reduced());
    });
  } else facts.hidden = true;
}

function advance() {
  const st = viewState("context", { at: 0 }), n = steps().length;
  if (st.at < n - 1) { st.at++; render(); } else S.tour.show(S.tour.views.includes("network") ? "network" : S.tour.views[0]);
}

// → and space step forward, ← back.
function key(e) {
  const st = viewState("context", { at: 0 });
  if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); advance(); } else if (e.key === "ArrowLeft" && st.at > 0) { st.at--; render(); }
}

export const view = Object.freeze({ id: "context", show: render, key });
