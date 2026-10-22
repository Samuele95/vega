// The card of what was clicked on the drawing (D4 §18: the hand-built page's unit card,
// index.html:1366–1429). On a pure dense chain a unit, an output or a family is read straight
// through the weights: the product of the matrices from the inputs up to it and from it on to the
// outputs — a linear reading that leaves out the bends between layers, and says so. That reading
// means nothing past a stage that is not a weighted sum, so on any other graph the card of a unit
// shows its own weights only, and the card of a stage its shapes, its parameters and its formula.
import { el, restart } from "../core/dom.js";
import { t, tText, tnText, data, hasKey } from "../core/text.js";
import { divergingBars } from "../core/bars.js";
import { groupChild } from "./segments.js";

// A stage's name: the explain catalog's label when the page has it (page.json#explain), else the
// strings table's word for its kind.
export function stageLabel(stage) {
  const key = `explain.${stage.type}.label`;
  return hasKey(key) ? tText(key) : tText(`box.kind.${stage.kind}`);
}

// (A.out × A.in) · (A.in × B.in)
function mul(A, B) {
  const out = new Float32Array(A.out * B.in);
  for (let i = 0; i < A.out; i++) for (let s = 0; s < A.in; s++) {
    const a = A.w[i * A.in + s]; if (!a) continue;
    for (let j = 0; j < B.in; j++) out[i * B.in + j] += a * B.w[s * B.in + j];
  }
  return { out: A.out, in: B.in, w: out };
}

// The linear reading of a pure chain's matrices: toIn[k] (stage k+1 units × inputs) and toOut[h]
// (outputs × units of hidden layer h). Computed once per model (kept on it).
export function effective(model) {
  if (model.effective) return model.effective;
  const mats = model.layers.map((l) => l.W), L = mats.length, toIn = [mats[0]], toOut = [];
  for (let h = 1; h < L; h++) toIn.push(mul(mats[h], toIn[h - 1]));
  let acc = mats[L - 1]; toOut[L - 1] = acc;
  for (let h = L - 2; h >= 1; h--) { acc = mul(acc, mats[h]); toOut[h] = acc; }
  model.effective = { toIn, toOut, L };
  return model.effective;
}

const rowOf = (m, r) => Array.from(m.w.subarray(r * m.in, (r + 1) * m.in));
const colOf = (m, c) => Array.from({ length: m.out }, (_, o) => m.w[o * m.in + c]);

// who, sub, left, right, note of a node of the families drawing (DenseRun cols)
function chainCard(model, node) {
  const E = effective(model), input = (i) => data(model.inputs[i]), out = (c) => data(model.outputs[c]);
  if (node.kind === "unit") {
    const h = node.layer;
    return { who: tText("legacy.net.cap.unit", { j: node.j + 1 }), sub: tText("legacy.net.card.unit_sub", { h, l: E.L - 1, n: model.layers[h - 1].out }),
      left: divergingBars({ title: tText("legacy.net.card.rises"), values: rowOf(E.toIn[h - 1], node.j), k: 8, tag: input }),
      right: divergingBars({ title: tText("legacy.net.card.pushes"), values: colOf(E.toOut[h], node.j), k: 6, tag: out }),
      note: tText(h === 1 ? "legacy.net.card.note_first" : "legacy.net.card.note_deep") };
  }
  if (node.kind === "class") {
    return { who: data(node.c), sub: tText("legacy.net.card.class_sub"),
      left: divergingBars({ title: tText("legacy.net.card.raised_by"), values: rowOf(E.toIn[E.L - 1], node.j), k: 10, tag: input }),
      right: null, note: tText("legacy.net.card.class_note") };
  }
  const M = E.toIn[E.L - 1], cls = model.outputs.map((_, c) => node.g.idx.reduce((s, i) => s + M.w[c * M.in + i], 0));
  const text = node.g.text ? `${node.g.text[0].toUpperCase()}${node.g.text.slice(1)}. ${tText("legacy.net.card.family_note")}` : "";
  return { who: node.g.label, sub: tnText("legacy.net.card.family_sub", node.g.idx.length),
    left: el("div", {}, el("div", { class: "caps muted net-card-title" }, t("legacy.net.card.members")),
      el("div", { class: "fg2 net-card-members" }, node.g.members.map((n) => data(n)).join(" · "))),
    right: divergingBars({ title: tText("legacy.net.card.family_pushes"), values: cls, k: 6, tag: out }), note: text };
}

// the card of a stage of a segmented graph: a run's unit (its own weights), a box, the head, a family
function stageCard(graph, pick, env) {
  let { seg, hit } = pick;
  const ordinal = pick.ordinal || new Map();
  while (seg.kind === "group" && hit && hit.at) { const kid = groupChild(seg, hit.at); hit = hit.hit; seg = kid; }
  if (seg.kind === "run") {
    const layer = seg.model.layers[hit.k], W = layer.W, names = hit.k === 0 ? seg.model.inputs : null;
    const tag = (i) => (names ? data(names[i]) : tText("legacy.net.unit", { j: i + 1 }));
    const h = ordinal.get(layer.id) || hit.k + 1, l = ordinal.size || seg.model.layers.length;
    return { who: tText("legacy.net.cap.unit", { j: hit.j + 1 }), sub: tText("legacy.net.card.unit_sub", { h, l, n: layer.out }),
      left: W ? divergingBars({ title: tText("card.own"), values: rowOf(W, hit.j), k: 8, tag }) : null, right: null, note: tText("card.own_note") };
  }
  if (seg.kind === "input") {
    const f = env.families[hit.j];
    return { who: f.label, sub: tnText("legacy.net.card.family_sub", f.members.length), note: f.text,
      left: el("div", { class: "fg2 net-card-members" }, f.members.map((n) => data(n)).join(" · ")), right: null };
  }
  const s = graph.byId(seg.kind === "group" ? seg.group : seg.stage);
  if (!s) return null;
  return { who: seg.kind === "head" ? data(graph.headLabels()[hit.j] ?? "") : stageLabel(s),
    sub: s.params ? tnText("card.stage_params", s.params) : tText("card.stage_none"),
    left: s.formula ? el("div", {}, el("div", { class: "caps muted net-card-title" }, t("card.stage_formula")), el("div", { class: "formula" }, s.formula)) : null,
    right: null, note: "" };
}

// Shows the card of `pick` (a DenseRun node, or {seg, hit} of a segmented drawing) in `host`.
// ctx: {graph, model (the pure chain's run model), families}.
export function showCard(host, pick, ctx) {
  const c = pick && pick.kind ? chainCard(ctx.model, pick) : pick ? stageCard(ctx.graph, pick, ctx) : null;
  if (!c) { host.hidden = true; return; }
  host.hidden = false; restart(host);
  const close = el("button", { class: "x caps", "aria-label": tText("legacy.net.card.close_aria") }, t("legacy.net.card.close"));
  close.onclick = () => { host.hidden = true; };
  host.replaceChildren(el("div", { class: "head" }, el("span", { class: "who" }, c.who), el("span", { class: "muted" }, c.sub), close),
    el("div", { class: "cols" }, c.left || el("div"), c.right || el("div")),
    c.note ? el("p", { class: "hint net-card-note" }, c.note) : null);
}
