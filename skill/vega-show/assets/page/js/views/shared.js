// What the views share: the story's sections, each view's own remembered state (kept in the page
// state, core/state.js, the one module that may hold mutable state), a refused request as the page
// shows it, and a number as a text slot's parameter.
import { el } from "../core/dom.js";
import { S, setState } from "../core/state.js";
import { t, tText, data } from "../core/text.js";
import { sg, plain } from "../core/fmt.js";

// The story's section of that id, or {} (a story without it, or no story).
export const section = (id) => ((S.story && S.story.sections) || []).find((s) => s && s.id === id) || {};

// The remembered state of one view: an object created once from `init`, kept across its showings.
export function viewState(id, init = {}) {
  if (!S.views) setState({ views: {} });
  if (!S.views[id]) S.views[id] = { ...init };
  return S.views[id];
}

// A refused request, as the page shows it: the problem's own title and detail, then one line per
// column it names (the service's words as data; the frame of each line is the strings table's).
export function problemText(e) {
  const line = (x) => "  " + tText("problem.line", { column: x.column === null || x.column === undefined ? tText("problem.request") : data(x.column), error: data(x.error) })
    + (x.rows && x.rows.length ? " " + tText("problem.rows", { rows: x.rows.join(", ") }) : "");
  const errs = ((e && e.errors) || []).map(line).join("\n");
  return data(e && e.message ? e.message : String(e)) + (errs ? "\n" + errs : "");
}

// A number in the page's .num style, as a text slot's parameter.
export const num = (v) => el("span", { class: "num", style: "font-size:inherit" }, String(v));

// ---- the attribution's method -----------------------------------------------------------------------
// The methods the service attributes with (inspect-contract §5.4): gradient × input, a local,
// first-order reading that makes no completeness claim; integrated gradients, whose terms add up to
// the score's change from the baseline up to the gap the record states.
export const ATTRIBUTION_METHODS = Object.freeze(["grad_x_input", "integrated_gradients"]);

// The method an attribution (an explanation@2's, the demo sample's vega/attribution@2) names, or null.
export const methodOf = (a) => (!a ? null : typeof a.method === "string" ? a.method : a.method && typeof a.method.name === "string" ? a.method.name : null);

// The attribution's method in words beside what it drew, and — when the record states its
// completeness — the gap between the terms and the score's change (A31: never folded into the
// terms). `gap: false` leaves the gap to a renderer that already shows it.
export function methodNote(a, { gap = true } = {}) {
  const m = methodOf(a);
  if (!m) return null;
  const c = a.completeness, parts = [t(ATTRIBUTION_METHODS.includes(m) ? `attribution.method.${m}` : "attribution.method.other", { method: data(m) })];
  if (gap && c && Number.isFinite(c.gap)) {
    parts.push(" ", el("span", { class: "views-completeness" }, t("attribution.gap", { gap: plain(c.gap, 4), sum: sg(c.sum ?? 0, 3), change: sg(c.target_minus_baseline ?? 0, 3) })));
  }
  return el("p", { class: "hint views-method", "data-method": m }, ...parts);
}
