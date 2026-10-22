// The input step (D4 §6.5, §8.1): what the network sees. The recipe fitted on the training data put
// each input on the network's scale — one clause per op page.json#recipe names (D4 §14.1:
// how.input.op.<op>; with no recipe on the page, no claim about what it did) — and the shaped
// numbers follow, each beside the raw value it came from: a row's inputs, the furthest from zero
// first; a window's matrix, its largest cells named "<feature> @ <step>"; a graph node's own vector,
// then its neighbourhood. Nothing is recomputed: the recipe runs in the service, before the network.
import { el } from "../../core/dom.js";
import { t, tText, tnText, data } from "../../core/text.js";
import { termRow } from "../../core/bars.js";
import { fmtNum, sg, plain, subs } from "../../core/fmt.js";
import { joinList } from "../../net/guide.js";
import { Answer } from "../answer.js";

export const TOP_INPUTS = 6;
// The recipe ops the page names (vega-show catalog.json#page.recipe_ops, D4 §14.1).
export const RECIPE_OPS = Object.freeze(["impute", "indicator", "encode", "clip", "log1p", "standard_scale", "robust_scale",
  "minmax_scale", "bin", "window", "graph_edges", "project", "other"]);

// The recipe's clauses, one per distinct op of page.json#recipe.ops, in the recipe's order.
export function recipeClauses(page) {
  const ops = page && page.recipe && Array.isArray(page.recipe.ops) ? page.recipe.ops.map((o) => o.op) : [];
  return [...new Set(ops)].filter((op) => RECIPE_OPS.includes(op)).map((op) => tText(`how.input.op.${op}`, { zero: 0, one: 1 }));
}

// The raw value an input came from, as the page shows it (null: the row does not carry it).
function rawOf(raw, name) {
  if (!raw || !Object.prototype.hasOwnProperty.call(raw, name)) return null;
  const v = raw[name];
  return v === null || v === "" ? tText("how.input.empty") : typeof v === "number" ? fmtNum(v) : data(v);
}

// The largest cells of the shaped numbers, as signed bars "raw → x = v".
function topBars(cells, raw) {
  const top = cells.slice().sort((a, b) => Math.abs(b.x) - Math.abs(a.x)).slice(0, TOP_INPUTS), mx = Math.max(1e-9, ...top.map((c) => Math.abs(c.x)));
  const bars = el("div", { class: "bars diverge eq" });
  top.forEach((c, k) => {
    const r = c.raw === undefined ? rawOf(raw, c.column) : c.raw;
    bars.append(termRow({ name: c.name, v: c.x, max: mx, i: k, val: [r !== null ? el("span", { class: "wx" }, `${r}  → `) : "", tText("how.input.x", { v: sg(c.x) })] }));
  });
  return { bars, top };
}

// The rest of the numbers, by their range.
function restLine(cells, top) {
  const rest = cells.filter((c) => !top.includes(c)).map((c) => c.x);
  return rest.length ? el("p", { class: "hint" }, t("how.input.rest", { n: rest.length, lo: plain(Math.min(...rest)), hi: plain(Math.max(...rest)) })) : null;
}

// A row (or a graph node's own vector): one cell per input, named by the network's input order.
function rowCells(A) {
  const i = A.inputs(), x = i.x || [], names = i.names.length === x.length ? i.names : A.graph.inputs.numeric.length === x.length ? A.graph.inputs.numeric : null;
  return x.map((v, k) => ({ name: names ? data(names[k]) : tText("how.input.nth", { k: k + 1 }), column: names ? names[k] : null, x: v }));
}

// A window: one cell per (step, feature), named "<feature> @ <step>" (the axis labels of v1/network).
function windowCells(A) {
  const grid = A.inputs().grid, L = grid.length, feats = A.graph.inputs.features || A.graph.inputs.numeric, steps = A.graph.inputs.steps;
  const featName = (f) => { const x = feats[f]; return x && typeof x === "object" ? tText("how.input.past", { name: data(x.name) }) : data(x ?? tText("how.input.nth", { k: f + 1 })); };
  const stepName = (s) => (steps && steps[s] ? data(steps[s]) : s === L - 1 ? tText("how.input.now") : tText("how.input.back", { k: L - 1 - s }));
  return grid.flatMap((row, s) => row.map((v, f) => ({ name: tText("how.input.cell", { feature: featName(f), step: stepName(s) }), raw: null, x: v })));
}

// The graph around the explained node, as the answer states it: the nodes and edges read, and how
// many neighbours each hop brought (the attribution's neighbourhood).
function graphLine(A) {
  const g = A.v2 && A.ex.sample ? A.ex.sample.graph : null;
  if (!g) return null;
  const near = A.ex.attribution && Array.isArray(A.ex.attribution.neighbours) ? A.ex.attribution.neighbours : [], hops = new Map();
  near.forEach((m) => hops.set(m.hop, (hops.get(m.hop) || 0) + 1));
  const per = [...hops].sort((a, b) => a[0] - b[0]).map(([h, n]) => tnText("how.input.hop", n, { h }));
  return el("div", {}, el("p", { class: "hint" }, t(g.level === "graph" ? "how.input.graph_level" : "how.input.graph_node", { nodes: g.nodes, edges: g.edges })),
    per.length ? el("p", { class: "hint" }, t("how.input.hops", { hops: joinList(per) })) : null);
}

function render(step, ex, env) {
  const A = Answer.of(ex, env.graph), isWindow = !!A.inputs().grid, cells = isWindow ? windowCells(A) : rowCells(A), n = cells.length;
  const clauses = recipeClauses(env.page);
  const lead = clauses.length ? t("how.input.lead", { n, n_sub: subs(n), ops: joinList(clauses) }) : t("how.input.lead_plain", { n, n_sub: subs(n) });
  if (!n) return el("div", {}, el("p", {}, lead), el("p", { class: "hint how-miss" }, t("how.input.none")));
  const { bars, top } = topBars(cells, A.inputs().raw);
  return el("div", {}, el("p", {}, lead),
    isWindow ? el("p", {}, t("how.input.window", { n: A.inputs().grid.length })) : graphLine(A),
    el("div", { class: "caps muted how-terms" }, t("how.input.top", { k: top.length, zero: 0 })),
    bars, restLine(cells, top));
}

export default { kinds: ["input"], render, recompute: () => null };
