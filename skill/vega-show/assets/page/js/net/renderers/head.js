// The output column (D4 §7.4), drawn by the lane's link. Its nodes are the head's units, so the
// segment before feeds them through the head's weights.
//   softmax        one node per class; once answered, the highest ringed
//   sigmoid        one score node with its threshold tick, the positive class beside it; once
//                  answered, ringed when the score passes the threshold, else the answer named
//   sigmoid_each   one node per label with its threshold tick; every label past it ringed
//   identity       one value node (one per quantile), labelled with the value once answered
//   cumulative     the k−1 thresholds between the k classes, the classes beside them (read off the
//                  thresholds' names); the answer ringed
//   reconstruction the rebuilt inputs, one node per unit; once answered each is coloured by the error
//                  of the input it rebuilds (the alert colour on the anomaly view key), the error said
//   distance       the embedding's units and the threshold sphere (radius √t), the row's point at its
//                  distance; the point in the alert colour when it falls outside
//   horizon        one node per step ahead, in a row; under the quantile loss a step's quantiles are
//                  stacked into a band
// Its detail level (D4 §7.2) is the head matrix drawn as dense detail when a dense run feeds it: at
// "every weight" each unit is joined to its 6 strongest inputs among the run's units, all drawn, and
// pointing at a unit names them. Fed by any other stage, the head matrix is read in the weights read
// (weights.js), stage by stage.
import { css, el } from "../../core/dom.js";
import { t, tText, data } from "../../core/text.js";
import { fmtNum } from "../../core/fmt.js";
import { tensorValues } from "../graph.js";
import { unitMatrix } from "../segments.js";

const W = 170, W_NARROW = 120, STEP_W = 26, BAND = 9, THRESHOLD_LINKS = Object.freeze(["sigmoid", "sigmoid_each"]);
const TOP_IN = 3, TOP_IN_DETAIL = 6, SUPPORT = 3;

// The explained outputs of an explanation@2: the head's raw scores, their reading after the link,
// the task's answer, each unit's share of the largest reading.
export function outputsOf(ex) {
  const o = ex && ex.outputs;
  if (!o || !Array.isArray(o.raw)) return null;
  const read = Array.isArray(o.read) ? o.read : o.raw, mx = Math.max(1e-12, ...read.map(Math.abs));
  return { raw: o.raw, read, answer: o.answer || {}, share: read.map((v) => Math.abs(v) / mx), err: null };
}

// The units ringed once answered: the highest class, or every score past the threshold.
export function winners(link, o, threshold) {
  if (!o) return new Set();
  if (link === "softmax") return new Set([o.read.reduce((b, v, i) => (v > o.read[b] ? i : b), 0)]);
  if (!THRESHOLD_LINKS.includes(link)) return new Set();
  const t = threshold ?? .5;
  return new Set(o.read.map((p, j) => (p >= t ? j : -1)).filter((j) => j >= 0));
}

// The class an ordinal answer lands on: the service's rank, else the thresholds passed (z ≥ 0).
export const ordinalAnswer = (o) => (Number.isInteger(o.answer.rank) ? o.answer.rank : o.raw.filter((z) => z >= 0).length);

// Each unit's error share on a reconstruction head (answer.per_feature, by the input it rebuilds).
export function unitErrors(graph, o) {
  const pf = o && o.answer.per_feature;
  if (!pf) return null;
  const names = graph.task.outputs.names, per = Array.isArray(pf) ? pf : names.map((n) => Number(pf[n]) || 0);
  const e = graph.rebuiltInputOf().map((f) => (f >= 0 ? Math.abs(Number(per[f]) || 0) : 0)), mx = Math.max(1e-12, ...e);
  return e.map((v) => v / mx);
}

// The squared distance of the row's embedding from the one-class centre: the answer's, else
// recomputed from the head's raw output and the exported centre buffer.
export function distanceOf(o, centre) {
  if (!o) return null;
  if (Number.isFinite(o.answer.distance)) return o.answer.distance;
  if (!centre || centre.length !== o.raw.length) return null;
  return o.raw.reduce((s, v, j) => s + (v - centre[j]) ** 2, 0);
}

// A unit's strongest weights on the units of the dense run that feeds the head (its last layer), as
// "unit j (raises), …"; null when no dense run feeds it or its weights are not in the payload.
export function supportOf(graph, seg, j) {
  const segs = graph.segments(), before = segs[segs.indexOf(seg) - 1], M = unitMatrix(graph, graph.byId(seg.stage));
  if (!M || !before || before.kind !== "run") return null;
  return Array.from(M.w.subarray(j * M.in, (j + 1) * M.in)).map((v, i) => [i, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, SUPPORT)
    .map(([i, v]) => tText("legacy.net.cap.pair", { name: tText("legacy.net.unit", { j: i + 1 }), verb: tText(v >= 0 ? "legacy.net.verb.raises" : "legacy.net.verb.lowers") }))
    .join(", ");
}

// ---- layouts: one point per head unit
function columnLayout(box, n, x) {
  return Array.from({ length: n }, (_, j) => ({ x, y: box.y + box.h * (n === 1 ? .5 : j / (n - 1)) }));
}

// the steps ahead left to right; Q quantiles of a step stacked around the middle
function horizonLayout(box, labels, quantiles) {
  const q = quantiles ? quantiles.length : 1, Q = q > 1 && labels.length % q === 0 ? q : 1, H = labels.length / Q, mid = box.y + box.h / 2;
  return labels.map((label, j) => {
    const h = Math.floor(j / Q), k = j % Q;
    return { x: box.x + 12 + (box.w - 24) * (H === 1 ? .5 : h / (H - 1)), y: mid + (k - (Q - 1) / 2) * BAND, step: String(label).split("/")[0], first: k === 0, Q };
  });
}

// the k classes top (lowest) to bottom, the k−1 thresholds between neighbours
function ordinalLayout(box, graph, x) {
  const names = graph.ordinalClasses(), k = names.length, below = names[k - 2];
  const classes = names.map((name, c) => ({ x: x + 14, y: box.y + box.h * (k === 1 ? .5 : c / (k - 1)), name,
    text: name === null ? tText("head.above", { name: data(below) }) : data(name) }));
  return { classes, pts: classes.slice(0, -1).map((c, j) => ({ x, y: (c.y + classes[j + 1].y) / 2 })) };
}

// ---- drawing
function unitValue(L, o, j) {
  if (!o) return null;
  return L.link === "reconstruction" && o.err ? o.err[j] : o.share[j];
}

function unitFill(L, u, st) {
  const C = st.ctx.colors, roles = st.ctx.env.roles, on = st.focus.node("head");
  if (u.hot) return C.fg;
  if (u.v === null) return st.focus.active && !st.focus.all && on ? C.acc : C.mut;
  if (L.link === "reconstruction") return st.ctx.graph.viewKey === "anomaly" ? C.alert : C.fg;
  return roles && roles.isAlert(L.labels[u.j]) ? C.alert : C.acc;
}

function unitText(L, u, o) {
  const label = L.labels[u.j];
  if (L.link === "horizon") return u.p.first ? data(u.p.step) : null;
  if (L.link === "identity" && o) return L.labels.length > 1 ? tText("head.quantile", { name: data(label), v: fmtNum(o.read[u.j]) }) : fmtNum(o.read[u.j]);
  if (L.link === "sigmoid" && o && !u.won && o.answer.class !== undefined) return data(o.answer.class);
  return data(label);
}

function drawUnit(g, L, u, st) {
  const C = st.ctx.colors, o = u.o, fill = unitFill(L, u, st), on = st.focus.node("head");
  g.globalAlpha = u.hot ? 1 : u.v !== null ? .3 + .7 * u.v : on ? .95 : .3; g.fillStyle = fill;
  g.beginPath(); g.arc(u.p.x, u.p.y, (L.link === "horizon" && u.p.Q > 1 ? 3.5 : 5.5) + (u.v !== null ? 2.5 * u.v : 0), 0, 7); g.fill();
  if (u.won && st.phase >= 1) { g.globalAlpha = .9; g.strokeStyle = fill; g.lineWidth = 1.5; g.beginPath(); g.arc(u.p.x, u.p.y, 10.5, 0, 7); g.stroke(); }
  if (THRESHOLD_LINKS.includes(L.link) && st.ctx.graph.task.threshold !== null) {
    g.globalAlpha = .7; g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(u.p.x - 9, u.p.y - 9); g.lineTo(u.p.x + 9, u.p.y - 9); g.stroke();
  }
  const text = L.classes || (!L.fits && !(u.v !== null && u.v >= .02) && !u.hot && !u.won) ? null : unitText(L, u, o);
  if (text === null) return;
  g.globalAlpha = on || u.v !== null ? .85 : .35; g.fillStyle = u.won ? fill : C.fg2;
  if (L.link === "horizon") g.fillText(text, u.p.x, L.box.y + L.box.h / 2 + (u.p.Q - 1) / 2 * BAND + 16); else g.fillText(text, u.p.x + 14, u.p.y);
}

function drawBands(g, L, st) {                         // horizon under the quantile loss: a step's quantiles joined
  const C = st.ctx.colors;
  g.globalAlpha = .6; g.strokeStyle = C.muted; g.lineWidth = 1;
  L.pts.filter((p) => p.first && p.Q > 1).forEach((p) => { g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(p.x, p.y + (p.Q - 1) * BAND); g.stroke(); });
}

function drawClasses(g, L, st, o) {                    // cumulative: the classes beside the thresholds
  const C = st.ctx.colors, answer = o ? ordinalAnswer(o) : -1, roles = st.ctx.env.roles;
  L.classes.forEach((c, j) => {
    const won = j === answer && st.phase >= 1, fill = won && roles && c.name !== null && roles.isAlert(c.name) ? C.alert : won ? C.acc : C.mut;
    g.globalAlpha = won ? 1 : .6; g.fillStyle = fill; g.beginPath(); g.arc(c.x, c.y, 4.5, 0, 7); g.fill();
    if (won) { g.strokeStyle = fill; g.lineWidth = 1.5; g.beginPath(); g.arc(c.x, c.y, 9.5, 0, 7); g.stroke(); }
    g.globalAlpha = .85; g.fillStyle = won ? fill : C.fg2; g.fillText(c.text, c.x + 13, c.y);
  });
}

function drawSphere(g, L, st, o) {                     // distance: the threshold sphere and the row's point
  const C = st.ctx.colors, thr = st.ctx.graph.task.threshold, cx = L.box.x + L.box.w * .62, cy = L.box.y + L.box.h / 2;
  const R = Math.max(8, Math.min(L.box.w * .3, L.box.h / 3)), d2 = distanceOf(o, L.centre);
  g.globalAlpha = .6; g.strokeStyle = C.line; g.lineWidth = 1; if (!thr) g.setLineDash([3, 3]);
  g.beginPath(); g.arc(cx, cy, R, 0, 7); g.stroke(); g.setLineDash([]);
  g.globalAlpha = .8; g.fillStyle = C.muted; g.beginPath(); g.arc(cx, cy, 2, 0, 7); g.fill();
  if (d2 === null || !thr) return;
  const x = cx + R * Math.min(1.6, Math.sqrt(Math.max(0, d2) / thr));
  g.globalAlpha = 1; g.fillStyle = d2 >= thr ? C.alert : C.acc; g.beginPath(); g.arc(x, cy, 5, 0, 7); g.fill();
  g.fillStyle = C.fg2; g.textAlign = "center"; g.fillText(tText("head.distance", { d: fmtNum(d2) }), cx, cy + R + 14);
}

function drawError(g, L, st, o) {                      // reconstruction: the row's error under the column
  if (!o || !Number.isFinite(o.answer.error)) return;
  g.globalAlpha = .85; g.fillStyle = st.ctx.colors.fg2; g.textAlign = "left";
  g.fillText(tText("head.error", { e: fmtNum(o.answer.error) }), L.box.x, L.box.y + L.box.h + 16);
}

export default {
  kinds: ["head"], detail: true,

  measure(seg, ctx) {
    const labels = ctx.graph.headLabels();
    if (ctx.graph.link !== "horizon") return { w: ctx.narrow ? W_NARROW : W, minH: 60 };
    const steps = horizonLayout({ x: 0, y: 0, w: 0, h: 0 }, labels, ctx.graph.task.quantiles).filter((p) => p.first).length;
    return { w: Math.max(90, STEP_W * steps), minH: 60 };
  },

  layout(seg, box, ctx) {
    const graph = ctx.graph, labels = graph.headLabels(), link = graph.link, stage = graph.byId(seg.stage);
    const ord = link === "cumulative" ? ordinalLayout(box, graph, box.x + 10) : null;
    const pts = link === "horizon" ? horizonLayout(box, labels, graph.task.quantiles) : ord ? ord.pts : columnLayout(box, labels.length, box.x + 10);
    const fits = link === "horizon" || (box.h - 10) / Math.max(1, labels.length - 1) >= 13;
    const caption = { x: link === "horizon" ? box.x + box.w / 2 : pts[0].x, text: tText("head.caption") };
    const centre = link === "distance" ? tensorValues(graph.tensor(`${stage.id}.centre`)) : null;
    return { box, labels, link, pts, fits, classes: ord ? ord.classes : null, stage, centre, inMatrix: unitMatrix(graph, stage),
             inTop: ctx.mode === "detail" ? TOP_IN_DETAIL : TOP_IN,
             in: pts.map((p, j) => ({ x: p.x - 6, y: p.y, unit: j })), out: [], captions: [caption] };
  },

  draw(g, seg, L, st) {
    const o = st.acts && st.acts[st.i], won = winners(L.link, o, st.ctx.graph.task.threshold);
    g.textBaseline = "middle"; g.font = `12.5px ${css("--sans")}`; g.textAlign = L.link === "horizon" ? "center" : "left";
    L.pts.forEach((p, j) => drawUnit(g, L, { p, j, o, won: won.has(j), v: unitValue(L, o, j), hot: !!st.hover && st.hover.i === st.i && st.hover.hit.j === j }, st));
    if (L.link === "horizon") drawBands(g, L, st);
    g.textAlign = "left";
    if (L.classes) drawClasses(g, L, st, o);
    if (L.link === "distance") drawSphere(g, L, st, o);
    if (L.link === "reconstruction") drawError(g, L, st, o);
    g.globalAlpha = 1;
  },

  hit(seg, L, x, y) {
    let best = null, bd = 12;
    L.pts.forEach((p, j) => { const d = Math.hypot(p.x - x, p.y - y), zone = L.link !== "horizon" && x > p.x && x < p.x + 160 ? Math.abs(p.y - y) : d; if (zone < bd) { bd = zone; best = { j }; } });
    return best;
  },

  caption(seg, hit, ctx) {
    const graph = ctx.graph, task = graph.task, labels = graph.headLabels(), support = supportOf(graph, seg, hit.j);
    const name = graph.link === "cumulative" ? t("head.above", { name: data(graph.ordinalClasses()[hit.j]) }) : data(labels[hit.j]);
    return el("div", {}, el("span", { class: "who" }, name), el("span", { class: "muted" }, " ", t("legacy.net.cap.class")),
      THRESHOLD_LINKS.includes(graph.link) && task.threshold !== null ? el("div", { class: "muted" }, t("head.threshold", { t: task.threshold })) : null,
      support ? el("div", {}, t("legacy.net.cap.support", { list: support })) : null);
  },

  acts(seg, ex, graph) {
    const o = outputsOf(ex);
    if (o && graph.link === "reconstruction") o.err = unitErrors(graph, o);
    return o;
  },
};
