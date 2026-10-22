// The shared parts of the advanced explainers (explainers/{conv, recurrent, attention, graph,
// latent}.js; D4 §8.1): a stage's tensors read from the stage graph with what stops them, the
// names of a sequence's positions and channels and of a graph's nodes, the picks of a unit laid
// out on a grid (a position and a channel, a node and a unit), small float64 algebra over the
// row-major float32 weights, and the recompute record the probe reads.
//
// What stops a recompute is said as it is, in three words the reader can tell apart:
//   missing   the answer did not bring a tensor the step needs (how.no_recompute, naming its ids)
//   weights   the page did not receive a weight (withheld by the service, or deferred and not
//             fetched) — how.adv.weights
//   fit       the numbers came, but they do not fit the stage's shape — how.adv.fit
import { el } from "../core/dom.js";
import { t, tText, data } from "../core/text.js";
import { sg, plain } from "../core/fmt.js";
import { Answer } from "./answer.js";
import { gapOf, noRecompute } from "./sum.js";

export const lastOf = (s) => (Array.isArray(s) && s.length ? s[s.length - 1] : 0);
export const sizeOf = (s) => (Array.isArray(s) ? s.reduce((a, b) => a * b, 1) : 0);
export const shapeText = (s) => data((s || []).join(" × "));
const has = (o, k) => !!o && Object.prototype.hasOwnProperty.call(o, k);

// ---- tensors
// A weight of the stage graph by its id: {values (Float32Array), shape} or {why: "absent" |
// "weights", ref} — absent when the network view names no such tensor, weights when it names it
// without its values (withheld, deferred).
export function tensorOf(graph, ref) {
  if (typeof ref !== "string") return { why: "absent", ref: String(ref) };
  const tsr = graph.tensor(ref), values = tsr ? graph.vector(ref) : null;
  if (!tsr) return { why: "absent", ref };
  return values ? { values, shape: tsr.shape.slice() } : { why: "weights", ref };
}

// Several weights at once: {w: {name: values}, shape: {name: shape}, lost: [refs]} (a name whose
// ref is null or undefined is optional: it is left out, not lost).
export function tensorsOf(graph, refs) {
  const w = {}, shape = {}, lost = [];
  for (const [name, ref] of Object.entries(refs)) {
    if (ref === null || ref === undefined) continue;
    const got = tensorOf(graph, ref);
    if (got.values) { w[name] = got.values; shape[name] = got.shape; } else lost.push(got.ref);
  }
  return { w, shape, lost };
}

// A bias tensor named by the convention `<weight id>_bias` (the attention projections, the encoder
// layer's linears, the attention pool's score), or null when the view has none.
export const biasRef = (graph, ref) => (typeof ref === "string" && graph.tensor(`${ref}_bias`) ? `${ref}_bias` : null);

// ---- algebra (float64 over row-major float32)
// Row r of a row-major (rows × cols) matrix times x (from offset `at` of x).
export function rowDot(w, cols, r, x, at = 0) {
  let s = 0;
  for (let i = 0; i < cols; i++) s += w[r * cols + i] * x[at + i];
  return s;
}

// W·x + b for a (rows × cols) matrix: a plain array of rows numbers.
export function affine(w, rows, cols, x, b = null) {
  return Array.from({ length: rows }, (_, r) => rowDot(w, cols, r, x) + (b ? b[r] : 0));
}

// One row of a (rows × cols) row-major array, as a plain array.
export const rowOf = (a, cols, r) => Array.from(a.slice(r * cols, (r + 1) * cols));

// ---- the answer's numbers
// A stage's field (in, out, grad_out, hidden, alpha, …) for the explained sample, flat, or null.
export const valuesOf = (ex, graph, id, field) => Answer.of(ex, graph).values(id, field);

// The fields a step reads: {have: {field: values}, missing: "<id>.<field>, …" or ""}.
export function fieldsOf(ex, graph, id, fields) {
  const have = {}, lost = [];
  for (const f of fields) { have[f] = valuesOf(ex, graph, id, f); if (!have[f]) lost.push(`${id}.${f}`); }
  return { have, missing: lost.join(", ") };
}

// The answer's record of a stage's probe (the node's `extra`, as the service reduced it), or null.
export function extraOf(ex, graph, id, key) {
  const n = Answer.of(ex, graph).node(id), x = n && n.extra;
  return x && has(x, key) ? x[key] : null;
}

// ---- the recompute record the probe reads (R-F1 state().stepper.values): the shown value's page
// and network numbers and their gap, and `ops`, each other value the step checked; `within` only
// when every one is. An op that could not be compared because the network's numbers do not fit
// the page's (`fit: false`, its page and network null, `sizes` saying what did not fit) is kept,
// never within: a check the page could not make is said, not dropped.
export function checkOf(main, ops = []) {
  const parts = ops.filter((o) => o && (o.fit === false || (Number.isFinite(o.page) && Number.isFinite(o.network))));
  return { ...main, ops: parts, within: main.within && parts.every((o) => o.within) };
}

// The worst of two equally long arrays (page, network) as one check {what, page, network, gap,
// within, at}, or null when they are not comparable.
export function worstOf(what, page, network) {
  if (!page || !network || page.length !== network.length || !page.length) return null;
  let at = 0;
  for (let i = 1; i < page.length; i++) if (Math.abs(network[i] - page[i]) > Math.abs(network[at] - page[at])) at = i;
  return { what, at, ...gapOf(page[at], network[at]) };
}

// ---- what stops a recompute
// What did not fit, as the answer's tensor ids with their lengths: "B3.in[120], B3.out[96]".
export const sizesOf = (id, have) => Object.entries(have).filter(([, v]) => v !== null && v !== undefined)
  .map(([f, v]) => `${id}.${f}[${typeof v === "number" ? v : v.length}]`).join(", ");

// The note of a step that cannot recompute: missing (the answer's tensor ids), weights (the
// weights' ids) or fit (what did not fit, and the stage's shape).
export function stopped(why, { what = "", stage = "", shape = null } = {}) {
  if (why === "weights") return el("p", { class: "hint how-miss" }, t("how.adv.weights", { what: data(what) }));
  if (why === "fit") return el("p", { class: "hint how-miss" }, t("how.adv.fit", { stage, what: data(what), shape: shapeText(shape) }));
  return noRecompute(data(what));
}

// ---- names
// The name of position t of a sequence of L positions: the network view's step labels (a window's)
// or token labels (a tokenizer's: [cls], then each input) when the stage keeps their length, else
// "position t".
export function positionName(graph, L, tt) {
  const labels = [graph.inputs.steps, graph.inputs.tokens].find((s) => Array.isArray(s) && s.length === L);
  if (labels && labels[tt] !== undefined) return data(labels[tt]);
  return tText("how.adv.position", { t: tt + 1 });
}

// Whether a step reads the network's inputs (the input step's, or an embedding joined to them).
export const readsInputs = (step) => !step.from || step.from.kind === "input" || step.from.kind === "embedding";

// The name of channel c of what a step reads: the window's features (the numeric columns, the past
// target, then each nominal column's embedding dims) when it reads the inputs, else "channel c".
export function channelName(graph, step, c) {
  if (readsInputs(step)) {
    const i = graph.inputs, feats = i.features || i.numeric || [];
    if (c < feats.length) return data(feats[c] && typeof feats[c] === "object" ? feats[c].name : feats[c]);
    let at = feats.length;
    for (const col of i.nominal || []) {
      if (c < at + (col.dim || 0)) return tText("how.adv.dim", { name: data(col.name), d: c - at + 1 });
      at += col.dim || 0;
    }
  }
  return tText("how.adv.channel", { c: c + 1 });
}

// The name of cell (position, channel) of a (L, C) stage.
export const cellName = (pos, ch) => tText("how.adv.cell", { position: pos, channel: ch });

// ---- picks: the units that moved the target most, named as cells ("t−3 · channel 5")
export function cellPicks({ rank, R, on, target, choose, nameOf }) {
  return el("div", { class: "picks" }, el("span", {}, t("how.picks", { target })),
    ...rank.map((u) => Object.assign(el("button", { class: u === on ? "on" : "", title: tText("how.pick_title", { target, v: sg(R[u]) }) },
      t("how.adv.pick", { what: nameOf(u), v: sg(R[u]) })), { onclick: () => choose(u) })));
}

// A line "over all n numbers the page and the network differ by at most gap".
export const allLine = (n, worst) => (worst ? el("p", { class: "hint" }, t("how.adv.all", { n, gap: plain(Math.abs(worst.gap), 6) })) : null);
