// The generic step (D4 §8.1): a stage the page does not recompute — user code (custom), a
// positional encoding, and while the network answers a dropout or an edge dropout when a collapsed
// network gives them a step — shown by its block's words, its formula and its shapes, with what it
// passed on for this sample when the answer carries it. It is also the stand-in for a kind whose
// explainer did not load (explainers/index.js records the gap): the step says so, and the walk goes
// on.
import { el } from "../../core/dom.js";
import { t, hasKey, data } from "../../core/text.js";
import { fmtInt, plain } from "../../core/fmt.js";
import { Answer } from "../answer.js";
import { formula, stageParams, stageProse, stageFormula } from "../sum.js";

export const OWN = Object.freeze(["custom", "positional", "edge_dropout", "dropout"]);

// The largest |value| a stage passed on, with its index, or null.
function largest(values) {
  if (!values || !values.length) return null;
  const j = values.reduce((b, v, k) => (Math.abs(v) > Math.abs(values[b]) ? k : b), 0);
  return { j, v: values[j], n: values.length };
}

function render(step, ex, env) {
  const graph = env.graph, id = step.stage || step.group || step.node, n = id ? graph.byId(id) : null;
  if (!n) return el("div", {}, el("p", {}, t("how.generic.lead")));
  const shape = (s) => data((s || []).join(" × ")), A = Answer.of(ex, graph);
  const own = stageFormula(n) || (n.formula ? formula(data(n.formula)) : null), out = largest(A.values(n.id, "out"));
  return el("div", {}, stageProse(n, "how.generic.lead", stageParams(n)), own,
    el("p", {}, t("how.generic.shapes", { from: shape(n.in_shape), to: shape(n.out_shape), params: fmtInt(n.params || 0) })),
    n.type === "custom" && !hasKey(`explain.${n.type}.prose`) ? el("p", { class: "hint" }, t("how.generic.custom")) : null,
    out ? el("p", { class: "hint" }, t("how.generic.out", { n: out.n, j: out.j + 1, v: plain(out.v, 3) })) : null,
    OWN.includes(step.kind) ? null : el("p", { class: "hint how-miss" }, t("how.generic.fallback")));
}

export default { kinds: OWN.slice(), render, recompute: () => null };
