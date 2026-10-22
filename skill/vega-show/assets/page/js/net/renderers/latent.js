// The latent stages (D4 §7.2, P7): what an autoencoder, a PCA floor or a variational autoencoder
// squeezes every sample through, and the one-class sphere a deep SVDD measures it against.
//
//   reparameterize  drawn here (stage kind `latent`): one row per number of the code z, each on an
//                   axis centred on 0. The stage reads 2·z numbers from the layer before it — the
//                   first z are the means μ, the last z the log-variances log σ² — so its left edge
//                   takes the upper half of that layer as "μ" and the lower half as "log σ²". At rest
//                   every row carries the standard normal the training pulls the code towards (a
//                   dashed whisker of ±1 around 0); one answer draws this sample's μ as a bar (the
//                   sign's colour) and its σ as a whisker around it, the code's value z = μ as the
//                   dot, and the KL term under the column. At answer time z = μ: nothing is drawn at
//                   random, so the same sample always gets the same answer.
//   the waist       bottleneckOf(graph): the one rule of which stage a latent family narrows to —
//                   the reparameterize stage of a VAE; the narrowest dense stage before the decoder
//                   of an autoencoder or a PCA floor (a PCA floor's one hidden layer is its waist).
//                   The dense drawings mark it (net.caption.bottleneck); waistLayer() gives their
//                   column index.
//   the sphere      sphereOf(graph): the deep SVDD head's reading — the exported centre buffer
//                   (<head>.centre), the threshold, and one answer's squared distance recomputed
//                   from the head's output and that centre beside the service's own.
//
// latentReading() is the probe's reading of this area (extendProbe "latent"): the waist, the latent
// stages and the sphere as each canvas of the page drew them, with the pixels drawn in each segment.
import { css, el } from "../../core/dom.js";
import { t, tn, tText } from "../../core/text.js";
import { fmtNum } from "../../core/fmt.js";
import { S } from "../../core/state.js";
import { extendProbe } from "../../core/probe.js";
import { tensorValues } from "../graph.js";
import { stageLabel } from "../card.js";

// Families that squeeze every sample through a waist (the catalog's reconstruction families).
export const LATENT_FAMILIES = Object.freeze(["autoencoder", "pca-floor", "vae"]);
const WAIST_FAMILIES = Object.freeze(["autoencoder", "pca-floor"]);
const ROWS_MAX = 16, ROW_MIN = 9, SPAN_MIN = 3, LOGVAR_CLAMP = 10, WIDTH = 1.3, W_MIN = 56;
// the box, top to bottom: the stage's name (LABEL_H), then a frame whose first and last MARK_H hold
// the μ and log σ² marks and the note, the rows between them
const LABEL_H = 14, MARK_H = 18;

const lastOf = (shape) => (Array.isArray(shape) && shape.length ? shape[shape.length - 1] : 0);
const finite = (v) => typeof v === "number" && Number.isFinite(v);

// ---- the waist
// The stage a latent family narrows every sample to: {kind: "latent", stage, units} for a
// reparameterize stage, {kind: "dense", stage, units} for the narrowest dense stage of an
// autoencoder's or a PCA floor's encoder (the dense stages before the first decoder layer; the
// last of equals, the one nearest the decoder); null for any other network.
export function bottleneckOf(graph) {
  if (!graph) return null;
  const latent = graph.stages().find((n) => n.kind === "latent");
  if (latent) return { kind: "latent", stage: latent.id, units: lastOf(latent.out_shape) };
  if (!WAIST_FAMILIES.includes(graph.family)) return null;
  const encoder = [];
  for (const run of graph.denseRuns()) {
    const stop = run.layers.findIndex((l) => l.group !== null);
    encoder.push(...run.layers.slice(0, stop < 0 ? run.layers.length : stop).map((l) => l.dense));
    if (stop >= 0) break;
  }
  if (!encoder.length) return null;
  const width = (id) => lastOf(graph.byId(id).out_shape);
  const stage = encoder.reduce((a, id) => (width(id) <= width(a) ? id : a), encoder[0]);
  return { kind: "dense", stage, units: width(stage) };
}

// The waist's column among a dense drawing's layers (graph.runModel(...).layers; the head, when the
// model carries it, is never the waist), or null.
export function waistLayer(graph, model) {
  const b = bottleneckOf(graph);
  if (!b || b.kind !== "dense" || !model) return null;
  const at = model.layers.findIndex((l, i) => l.id === b.stage && l.id !== model.head);
  return at < 0 ? null : at;
}

// ---- one answer
// The whole vector of a reduced reading ({form: vector, n, values}), an exact tensor, or null when
// the explanation kept only part of it (a top-k or a summary: a σ cannot be filled with zeros).
function wholeVector(r, n) {
  if (!r) return null;
  if (r.encoding) { const v = tensorValues(r); return v && (!n || v.length === n) ? Array.from(v) : null; }
  if (r.form === "vector" && Array.isArray(r.values) && r.values.length === r.n && (!n || r.n === n)) return r.values.slice();
  return null;
}

// This sample's reading of a reparameterize stage: {z, mu, sigma, kl, code} — the probe's own μ, σ
// and KL (inspect-contract §3.1, extra.mu / extra.sigma / extra.kl), else split from what the stage
// read (μ = the first z, σ = exp(½ · log σ²) with log σ² clamped to ±10 as the block clamps it); the
// code is what it passed on (z = μ). null when the explanation carries neither.
export function latentOf(ex, id, z) {
  const node = ex && ex.nodes ? ex.nodes[id] : null;
  if (!node || node.level === "omitted") return null;
  const extra = node.extra || {}, exact = node.exact || {};
  let mu = wholeVector(extra.mu, z), sigma = wholeVector(extra.sigma, z);
  const read = wholeVector(exact.in, 2 * z) || wholeVector(node.in, 2 * z);
  if ((!mu || !sigma) && read) {
    mu = mu || read.slice(0, z);
    sigma = sigma || read.slice(z).map((v) => Math.exp(.5 * Math.max(-LOGVAR_CLAMP, Math.min(LOGVAR_CLAMP, v))));
  }
  const code = wholeVector(exact.out, z) || wholeVector(node.out, z) || mu;
  const kl = extra.kl && extra.kl.form === "scalar" && finite(extra.kl.value) ? extra.kl.value
    : mu && sigma ? mu.reduce((s, m, j) => s - .5 * (1 + 2 * Math.log(sigma[j]) - m * m - sigma[j] * sigma[j]), 0) : null;
  if (!mu && !code) return null;
  return { z, mu, sigma, kl, code };
}

// ---- the sphere (deep SVDD, link "distance")
// {head, centre: the exported centre (Float32Array) or null, units, threshold}; null for any other link.
export function sphereOf(graph) {
  const h = graph ? graph.head() : null;
  if (!h || graph.link !== "distance") return null;
  const centre = tensorValues(graph.tensor(`${h.id}.centre`));
  return { head: h.id, centre, units: lastOf(h.out_shape), threshold: finite(graph.task.threshold) ? graph.task.threshold : null };
}

// One answer's squared distance from the centre, recomputed from the head's output, beside the
// service's own: {page, network, gap}; null without the centre or the output.
export function sphereAnswer(sphere, ex) {
  const raw = ex && ex.outputs && Array.isArray(ex.outputs.raw) ? ex.outputs.raw : null;
  if (!sphere || !sphere.centre || !raw || raw.length !== sphere.centre.length) return null;
  const page = raw.reduce((s, v, j) => s + (v - sphere.centre[j]) ** 2, 0);
  const own = ex.outputs.answer && finite(ex.outputs.answer.distance) ? ex.outputs.answer.distance : null;
  return { page, network: own, gap: own === null ? null : Math.abs(page - own) };
}

// ---- the renderer of the latent stage
// The rows drawn: every number of the code, or ROWS_MAX of them (the largest |μ| once answered, else
// evenly spread), and no more than the box's height holds at ROW_MIN apart.
function rowsOf(z, h, acts) {
  const room = Math.max(1, Math.min(ROWS_MAX, Math.floor(h / ROW_MIN) + 1));
  if (z <= room) return Array.from({ length: z }, (_, j) => j);
  if (acts && acts.mu) return acts.mu.map((m, j) => [Math.abs(m), j]).sort((a, b) => b[0] - a[0]).slice(0, room).map((p) => p[1]).sort((a, b) => a - b);
  return Array.from({ length: room }, (_, k) => Math.round(k * (z - 1) / Math.max(1, room - 1)));
}

// The axis's half-span: 3 (the prior's ±3σ) or what this sample's μ ± σ reaches.
function spanOf(acts) {
  if (!acts || !acts.mu) return SPAN_MIN;
  return Math.max(SPAN_MIN, ...acts.mu.map((m, j) => Math.abs(m) + (acts.sigma ? acts.sigma[j] : 0)));
}

function geometry(box, z, acts) {
  const frameTop = box.y + LABEL_H, frameBottom = box.y + box.h - 2, top = frameTop + MARK_H, bottom = Math.max(top, frameBottom - MARK_H);
  const rows = rowsOf(z, bottom - top, acts);
  const ys = rows.map((_, k) => (rows.length === 1 ? (top + bottom) / 2 : top + (bottom - top) * k / (rows.length - 1)));
  const x0 = box.x + 14, x1 = box.x + box.w - 8;
  return { rows, ys, x0, x1, xc: (x0 + x1) / 2, span: spanOf(acts), top, bottom, frameTop, frameBottom };
}

const xOf = (L, v) => L.xc + (L.x1 - L.xc) * Math.max(-1, Math.min(1, v / L.span));

// the left edge: the upper half reads μ, the lower half log σ² (the order the block splits them in)
function drawSplit(g, L, C) {
  const x = L.box.x + 3, mid = (L.top + L.bottom) / 2;
  g.strokeStyle = C.muted; g.lineWidth = 1; g.globalAlpha = .7;
  for (const [a, b] of [[L.top - 4, mid - 3], [mid + 3, L.bottom + 4]]) {
    g.beginPath(); g.moveTo(x + 4, a); g.lineTo(x, a); g.lineTo(x, b); g.lineTo(x + 4, b); g.stroke();
  }
  g.fillStyle = C.muted; g.font = `10px ${css("--mono")}`; g.textAlign = "left"; g.textBaseline = "top";
  g.fillText(tText("latent.mu"), x + 6, L.frameTop + 3);
  // the log σ² mark shares its line with the note (drawLabels): left out when the two do not fit
  const mark = tText("latent.logvar"), room = L.x1 - x - 6 - g.measureText(noteOf(L)).width - 6;
  g.textBaseline = "bottom";
  if (g.measureText(mark).width <= room) g.fillText(mark, x + 6, L.frameBottom - 3);
}

// under the rows: the KL term of one answer, else "z = μ"
const noteOf = (L) => (L.acts && finite(L.acts.kl) ? tText("latent.kl", { kl: fmtNum(L.acts.kl) }) : tText("latent.z_is_mu"));

// one number of the code: its axis, the prior at rest, or this sample's μ bar, σ whisker and z dot
function drawRow(g, L, k, st) {
  const C = st.ctx.colors, y = L.ys[k], j = L.rows[k], a = L.acts, on = st.focus.node(L.stage.id);
  const hot = !!st.hover && st.hover.i === st.i && st.hover.hit.j === j;
  g.globalAlpha = on ? .9 : .3; g.strokeStyle = hot ? C.fg : C.line; g.lineWidth = 1;
  g.beginPath(); g.moveTo(L.x0, y); g.lineTo(L.x1, y); g.moveTo(L.xc, y - 3); g.lineTo(L.xc, y + 3); g.stroke();
  const mu = a && a.mu ? a.mu[j] : null, sigma = a && a.sigma ? a.sigma[j] : null;
  if (mu === null) {                                   // the prior N(0, 1): a dashed whisker of ±1, its caps
    const lo = xOf(L, -1), hi = xOf(L, 1);
    g.globalAlpha = on ? .75 : .25; g.strokeStyle = C.muted; g.setLineDash([2, 2]);
    g.beginPath(); g.moveTo(lo, y); g.lineTo(hi, y); g.stroke(); g.setLineDash([]);
    g.beginPath(); g.moveTo(lo, y - 3); g.lineTo(lo, y + 3); g.moveTo(hi, y - 3); g.lineTo(hi, y + 3); g.stroke();
    g.globalAlpha = on ? .95 : .3; g.fillStyle = hot ? C.fg : on && st.focus.active && !st.focus.all ? C.acc : C.mut;
    g.beginPath(); g.arc(L.xc, y, 3.5, 0, 7); g.fill();
    return;
  }
  g.globalAlpha = on ? .55 : .2; g.fillStyle = mu >= 0 ? C.acc : C.neg;
  g.fillRect(Math.min(L.xc, xOf(L, mu)), y - 3, Math.abs(xOf(L, mu) - L.xc), 6);
  if (sigma !== null) {
    const lo = xOf(L, mu - sigma), hi = xOf(L, mu + sigma);
    g.globalAlpha = on ? .9 : .3; g.strokeStyle = C.fg2; g.beginPath();
    g.moveTo(lo, y); g.lineTo(hi, y); g.moveTo(lo, y - 4); g.lineTo(lo, y + 4); g.moveTo(hi, y - 4); g.lineTo(hi, y + 4); g.stroke();
  }
  g.globalAlpha = on ? 1 : .35; g.fillStyle = hot ? C.fg : mu >= 0 ? C.acc : C.neg;
  g.beginPath(); g.arc(xOf(L, a.code ? a.code[j] : mu), y, 3.5, 0, 7); g.fill();
}

function drawLabels(g, L, st) {
  const C = st.ctx.colors, on = st.focus.node(L.stage.id);
  g.globalAlpha = on ? .9 : .35; g.fillStyle = C.fg2; g.textAlign = "center"; g.textBaseline = "top";
  g.font = `11px ${css("--sans")}`; g.fillText(L.label, L.box.x + L.box.w / 2, L.box.y, L.box.w);
  g.fillStyle = C.muted; g.font = `10px ${css("--mono")}`; g.textBaseline = "bottom"; g.textAlign = "right";
  g.fillText(noteOf(L), L.x1, L.frameBottom - 3, L.box.w - 12);
}

export default {
  kinds: ["latent"],
  detail: false,

  measure(seg, ctx) { return { w: Math.max(W_MIN, Math.round(ctx.boxW * WIDTH)), minH: 90 }; },

  layout(seg, box, ctx) {
    const stage = ctx.graph.byId(seg.stage), z = lastOf(stage.out_shape), acts = ctx.acts || null;
    const L = { box, stage, z, acts, label: stageLabel(stage), ...geometry(box, z, acts) };
    const mid = (L.top + L.bottom) / 2, half = Math.max(1, Math.min(z, L.rows.length));
    // the ports the layer before feeds: its first z units (μ) on the upper half, the last z (log σ²)
    // on the lower half; what leaves is the code, one port per row drawn
    const side = (from, to, k) => from + (to - from) * (half === 1 ? .5 : k / (half - 1));
    L.in = [...Array.from({ length: half }, (_, k) => ({ x: box.x, y: side(L.top, mid - 3, k), unit: L.rows[k] })),
      ...Array.from({ length: half }, (_, k) => ({ x: box.x, y: side(mid + 3, L.bottom, k), unit: z + L.rows[k] }))];
    L.out = L.rows.map((j, k) => ({ x: box.x + box.w, y: L.ys[k], units: [j], v: acts && acts.code ? acts.code[j] / L.span : undefined }));
    L.captions = [{ x: box.x + box.w / 2, text: tText("latent.caption", { k: L.rows.length, z }) }];
    return L;
  },

  draw(g, seg, L, st) {
    if (Math.max(0, Math.min(1, st.phase * st.n - st.i)) <= 0) return;
    const C = st.ctx.colors, on = st.focus.node(L.stage.id), hot = !!st.hover && st.hover.i === st.i;
    g.globalAlpha = on ? .5 : .2; g.strokeStyle = hot ? C.fg : st.focus.active && !st.focus.all && on ? C.acc : C.line; g.lineWidth = 1;
    g.strokeRect(L.box.x + .5, L.frameTop + .5, L.box.w - 1, L.frameBottom - L.frameTop - 1);
    drawSplit(g, L, C);
    L.rows.forEach((_, k) => drawRow(g, L, k, st));
    drawLabels(g, L, st);
    g.globalAlpha = 1;
  },

  hit(seg, L, x, y) {
    if (x < L.box.x - 4 || x > L.box.x + L.box.w + 4 || y < L.box.y || y > L.box.y + L.box.h) return null;
    let best = null, bd = 8;
    L.ys.forEach((cy, k) => { const d = Math.abs(cy - y); if (d < bd) { bd = d; best = { j: L.rows[k] }; } });
    return best || { stage: seg.stage };
  },

  caption(seg, hit, ctx) {
    const s = ctx.graph.byId(seg.stage), z = lastOf(s.out_shape);
    return el("div", {}, el("div", {}, el("span", { class: "who" }, stageLabel(s)), el("span", { class: "muted" }, " ",
      Number.isInteger(hit.j) ? t("latent.cap.number", { j: hit.j + 1, z }) : tn("latent.cap.numbers", z))),
      el("div", { class: "latent-split" }, t("latent.cap.split", { n: lastOf(s.in_shape), z })),
      s.formula ? el("div", { class: "formula" }, s.formula) : null,
      el("div", { class: "muted" }, t("latent.cap.prior")));
  },

  acts(seg, ex, graph) { return latentOf(ex, seg.stage, lastOf(graph.byId(seg.stage).out_shape)); },
};

// ---- the probe's reading
// Pixels a segment's region of a canvas holds (alpha > 0: the canvas is cleared, not painted).
function inked(canvas, W, r) {
  if (!canvas || !W || !r || r.w <= 0 || r.h <= 0) return 0;
  const d = canvas.width / W, x = Math.max(0, Math.floor(r.x * d)), y = Math.max(0, Math.floor(r.y * d));
  const w = Math.min(canvas.width - x, Math.ceil(r.w * d)), h = Math.min(canvas.height - y, Math.ceil(r.h * d));
  if (w <= 0 || h <= 0) return 0;
  const px = canvas.getContext("2d").getImageData(x, y, w, h).data;
  let n = 0;
  for (let i = 3; i < px.length; i += 4) if (px[i] > 0) n++;
  return n;
}

// A pure dense chain drawn whole: the whole chain's ink, the column the drawing marked as its waist and its ink.
function chainReading(nc, b) {
  const d = nc.drawing, at = Number.isInteger(d.waist) ? d.waist : null, x = at === null || !d.xs ? null : d.xs[at + 1];
  return { drawn: "chain", ink: inked(nc.c, d.W, { x: 0, y: 0, w: d.W, h: d.H }), segments: [], latent: [], waist: { stage: at === null ? null : d.model.layers[at].id,
    caption: at === null ? null : d.captions()[at + 1], expected: b ? b.stage : null,
    pixels: x === null ? 0 : inked(nc.c, d.W, { x: x - 10, y: 0, w: 20, h: d.H }) } };
}

// A graph of segments: every segment's ink, the waist the runs marked, the latent stages, the sphere.
function segmentsReading(nc, graph, b) {
  const d = nc.drawing, ctx = d.ctx(), pixels = d.boxes.map((r) => inked(nc.c, d.W, { x: r.x, y: 0, w: r.w, h: d.H }));
  const segments = d.segs.map((s, i) => ({ kind: s.kind, stage: s.stage || s.group || null, pixels: pixels[i] }));
  const latent = d.segs.map((s, i) => [s, i]).filter(([s]) => s.kind === "box" && graph.byId(s.stage).kind === "latent")
    .map(([s, i]) => ({ stage: s.stage, z: d.layouts[i].z, rows: d.layouts[i].rows.length, pixels: pixels[i],
      answered: !!(d.acts && d.acts[i]), mu: d.acts && d.acts[i] ? d.acts[i].mu : null, sigma: d.acts && d.acts[i] ? d.acts[i].sigma : null,
      kl: d.acts && d.acts[i] ? d.acts[i].kl : null }));
  const captions = d.layouts.flatMap((L) => L.captions || []).map((c) => c.text);
  const at = b ? d.segs.findIndex((s) => (s.kind === "run" ? s.stages.includes(b.stage) : s.stage === b.stage)) : -1;
  const waist = { stage: b && b.kind === "dense" && ctx.waist === b.stage ? b.stage : b && b.kind === "latent" && at >= 0 ? b.stage : null,
    caption: b && b.kind === "dense" ? captions.find((c) => c === tText("net.caption.bottleneck", captionParams(d, b))) || null : null,
    expected: b ? b.stage : null, pixels: at >= 0 ? pixels[at] : 0 };
  return { drawn: "segments", ink: pixels.reduce((a, n) => a + n, 0), segments, latent, waist, captions };
}

// the parameters the dense drawings give the waist's caption ({h, k, n}): the one the run drew
function captionParams(d, b) {
  const i = d.segs.findIndex((s) => s.kind === "run" && s.stages.includes(b.stage));
  if (i < 0) return {};
  const k = d.segs[i].stages.indexOf(b.stage), L = d.layouts[i];
  return { h: d.ctx().ordinal.get(b.stage), k: L.kept[k].length, n: b.units };
}

function sphereReading(nc, sphere) {
  if (!nc || !sphere || nc.pure) return null;
  const d = nc.drawing, i = d.segs.findIndex((s) => s.kind === "head"), L = i >= 0 ? d.layouts[i] : null, o = d.acts ? d.acts[i] : null;
  const ex = o ? { outputs: { raw: o.raw, answer: o.answer } } : null;
  return { centre: L && L.centre ? L.centre.length : null, units: sphere.units, threshold: sphere.threshold,
    pixels: i >= 0 ? inked(nc.c, d.W, { x: d.boxes[i].x, y: 0, w: d.boxes[i].w, h: d.H }) : 0, answer: sphereAnswer(sphere, ex) };
}

function canvasReading(nc, graph, b, sphere) {
  if (!nc || !nc.drawing) return null;
  const out = nc.pure ? chainReading(nc, b) : segmentsReading(nc, graph, b);
  out.sphere = sphereReading(nc, sphere);
  return out;
}

// The probe's reading of the latent area on the page as it is now (null before a graph is read).
export function latentReading() {
  const graph = S.graph, D = S.draw;
  if (!graph) return null;
  const b = bottleneckOf(graph), sphere = sphereOf(graph);
  return { family: graph.family, link: graph.link, bottleneck: b,
    sphere: sphere ? { head: sphere.head, centre: sphere.centre ? Array.from(sphere.centre) : null, units: sphere.units, threshold: sphere.threshold } : null,
    canvases: { network: canvasReading(D && D.net, graph, b, sphere), try: canvasReading(D && D.tryNet, graph, b, sphere) } };
}

if (typeof window !== "undefined") extendProbe("latent", latentReading, { report: true });
