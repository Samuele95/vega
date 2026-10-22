// The network's canvas (D4 §7.3). A pure dense chain is one drawing across the whole canvas,
// today's: the families drawing (renderers/dense.js DenseRun) or every weight (DenseDetail), one of
// the two at a time. Any other graph is laid out as segments left to right — the inputs, dense runs,
// boxes, groups, the output column — each drawn by the renderer of its kind, joined by connectors:
// a run's weights where they join two columns of units, a fan of lines where a box stands between.
// NetCanvas owns the <canvas>: its size and pixel ratio, the pointer, the growing animation; the
// drawings own what is drawn.
import { css, reduced } from "../core/dom.js";
import { t } from "../core/text.js";
import { Focus } from "./graph.js";
import { rendererOf } from "./kinds.js";
import { RendererGap } from "./renderers/index.js";
import { familiesOf } from "./families.js";
import { bottleneckOf, waistLayer } from "./renderers/latent.js";

// A drawing grows in `ms` (d.phase 0 → 1); frames can stop (a background tab, a hidden pane), so a
// timer finishes the drawing and calls `done` even then. A later call supersedes an earlier one.
export function animateDrawing(d, ms, done) {
  if (reduced()) { d.phase = 1; d.draw(); if (done) done(); return; }
  const t0 = performance.now(), run = (d.run = (d.run || 0) + 1);
  let over = false;
  const finish = () => { if (over) return; over = true; clearTimeout(guard); if (done) done(); };
  const guard = setTimeout(() => { if (run === d.run && d.active) { d.phase = 1; d.draw(); } finish(); }, ms + 400);
  const tick = (s) => {
    if (over) return;
    if (!d.active || run !== d.run) { finish(); return; }            // superseded: finish quietly
    d.phase = Math.max(0, Math.min(1, (s - t0) / ms)); d.draw();
    if (d.phase < 1) requestAnimationFrame(tick); else finish();
  };
  requestAnimationFrame(tick);
}

// The theme's colours, read when a drawing is drawn (a theme change redraws).
export const colours = () => ({ acc: css("--accent"), neg: css("--neg"), mut: css("--muted2"), fg: css("--fg"), fg2: css("--fg2"),
  muted: css("--muted"), alert: css("--alert"), bg: css("--bg"), line: css("--line-hi") });

// The renderer a segment is drawn by (segments.js kinds → kinds.js RENDERER_OF).
export function rendererFor(renderers, graph, seg) {
  const kind = seg.kind === "run" ? "dense" : seg.kind === "group" ? seg.mode : seg.kind === "box" ? graph.byId(seg.stage).kind : seg.kind;
  const r = renderers.get(kind);
  if (!r) throw new RendererGap(kind, rendererOf(kind), "net.renderer.not_loaded");
  return r;
}

const BOX_W = 88, BOX_W_NARROW = 64, BOX_W_MIN = 48, K_MIN = 8, FAN = 8, GROW_MAX = 3, CAPTION_GAP = 10, CAPTION_ROW = 12;
// A group's width beyond its children's (D4 §7.3: group = Σ children + 16 px).
export const GROUP_PAD = 16;

// D4 §7.3's overflow rule, stage by stage: when the segments are wider than the room, collapse
// repeated groups to one repeat with "× n", then draw boxes 48 px wide, then draw runs with K = 8
// units a column (the "every weight" level keeps every unit) in narrow columns. A row still too wide
// after the last stage shrinks as a whole (spreadRow). Each stage keeps the ones before it.
export const FIT_STAGES = Object.freeze([
  Object.freeze({}),
  Object.freeze({ collapse: true }),
  Object.freeze({ collapse: true, boxW: BOX_W_MIN }),
  Object.freeze({ collapse: true, boxW: BOX_W_MIN, K: K_MIN, compact: true }),
]);

// The first stage of FIT_STAGES at which the segments fit in `room` (with `gap` between them), or the
// last: {stage, fit: {collapse, boxW, K, compact} (what the renderers read from ctx), sizes, want}.
export function fitRow(segs, ctx, room, gap) {
  let out = null;
  for (let stage = 0; stage < FIT_STAGES.length; stage++) {
    const f = FIT_STAGES[stage];
    const fit = { collapse: !!f.collapse, boxW: f.boxW || ctx.boxW, K: f.K && Number.isFinite(ctx.K) ? Math.min(ctx.K, f.K) : ctx.K, compact: !!f.compact };
    const c = { ...ctx, ...fit }, sizes = segs.map((s) => c.rendererFor(s).measure(s, c));
    out = { stage, fit, sizes, want: sizes.reduce((a, m) => a + m.w, 0) + gap * Math.max(0, segs.length - 1) };
    if (out.want <= room) break;
  }
  return out;
}

// The boxes [{x, w}] of a row of measured segments ({w, grow}) in [x, x + w] with `gap` between
// them: a row wider than the room shrinks as a whole; a narrower one grows its runs, its groups
// and its gaps (at most GROW_MAX times) and is centred in what is left.
export function spreadRow(sizes, x, w, gap) {
  const n = sizes.length, gaps = gap * Math.max(0, n - 1), want = sizes.reduce((a, m) => a + m.w, 0) + gaps;
  const growable = sizes.reduce((a, m) => a + (m.grow ? m.w : 0), 0) + gaps;
  const f = want > w ? w / want : growable ? Math.min(GROW_MAX, 1 + (w - want) / growable) : 1;
  const widths = sizes.map((m) => (want > w || m.grow ? m.w * f : m.w)), step = gap * f;
  let at = x + Math.max(0, (w - widths.reduce((a, v) => a + v, 0) - step * Math.max(0, n - 1)) / 2);
  return widths.map((v) => { const b = { x: at, w: v }; at += v + step; return b; });
}

// The column captions ({x, text}) under the drawing, left to right: each on the lowest of two rows
// where it clears its neighbour; one that clears neither is left out (pointing at it names it).
// ctx: {W, H, colors, captionFont}.
export function drawCaptions(g, captions, ctx) {
  g.font = ctx.captionFont; g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = ctx.colors.muted; g.globalAlpha = .9;
  const ends = [-Infinity, -Infinity];
  captions.slice().sort((a, b) => a.x - b.x).forEach((c) => {
    const w = g.measureText(c.text).width, x = Math.max(w / 2 + 2, Math.min(ctx.W - w / 2 - 2, c.x));
    const row = ends.findIndex((end) => end + CAPTION_GAP <= x - w / 2);
    if (row < 0) return;
    ends[row] = x + w / 2;
    g.fillText(c.text, x, ctx.H - 10 - CAPTION_ROW * row);
  });
  g.globalAlpha = 1;
}

// Any graph but a pure dense chain: segments left to right, each drawn by its renderer.
class SegmentDrawing {
  constructor(graph, { renderers, env, k, caption, isStatic }) {
    Object.assign(this, { graph, renderers, env, K: k, captionHost: caption, isStatic, segs: graph.segments() });
    Object.assign(this, { focus: new Focus(null), hover: null, acts: null, phase: 1, active: true, mode: "simple", W: 320, H: 180 });
    const dense = graph.denseRuns().flatMap((r) => r.stages);
    this.ordinal = new Map(dense.map((id, i) => [id, i + 1]));        // hidden layers numbered in drawing order
    const waist = bottleneckOf(graph);                                 // the latent families' waist (latent.js)
    this.waist = waist && waist.kind === "dense" ? waist.stage : null;
  }

  // What every renderer reads (D4 §7.3 ctx), before the overflow rule: fit is the stage layout() fitted.
  baseCtx() {
    const narrow = this.W < 700;
    return { W: this.W, H: this.H, narrow, mode: this.mode, graph: this.graph, env: this.env, K: this.mode === "detail" ? Infinity : this.K, colors: colours(),
             boxW: narrow ? BOX_W_NARROW : BOX_W, collapse: false, compact: false, top: 18, bottom: 36,
             ordinal: this.ordinal, waist: this.waist, captionFont: `10.5px ${css("--mono")}`,
             rendererFor: (s) => rendererFor(this.renderers, this.graph, s) };
  }
  ctx() { return this.fit ? { ...this.baseCtx(), ...this.fit.fit } : this.baseCtx(); }

  // widths from each renderer's measure under the overflow rule (fitRow), spread over the canvas
  layout(g, W, H) {
    Object.assign(this, { g, W: Math.max(320, W), H: Math.max(180, H) });
    const base = this.baseCtx(), pad = base.narrow ? 16 : 28, gap = base.narrow ? 18 : 30;
    this.fit = fitRow(this.segs, base, this.W - 2 * pad, gap);
    const ctx = this.ctx();
    this.boxes = spreadRow(this.fit.sizes, pad, this.W - 2 * pad, gap).map((b) => ({ ...b, y: ctx.top, h: this.H - ctx.top - ctx.bottom }));
    this.layouts = this.segs.map((s, i) => ctx.rendererFor(s).layout(s, this.boxes[i], { ...ctx, acts: this.acts ? this.acts[i] : null }));
  }

  st() { return { phase: this.phase, focus: this.focus, hover: this.hover, acts: this.acts, mode: this.mode, g: this.g, n: this.segs.length }; }

  draw() {
    const g = this.g, ctx = this.ctx(), st = this.st();
    g.clearRect(0, 0, this.W, this.H);
    for (let i = 0; i + 1 < this.segs.length; i++) drawConnector(g, this.layouts[i], this.layouts[i + 1], { ...st, i, ctx });
    this.segs.forEach((s, i) => { g.save(); ctx.rendererFor(s).draw(g, s, this.layouts[i], { ...st, i, ctx }); g.restore(); });
    drawCaptions(g, this.layouts.flatMap((L) => L.captions || []), ctx);
  }

  hitAt(x, y) {
    const i = this.boxes.findIndex((b) => x >= b.x - 14 && x <= b.x + b.w + 14);
    if (i < 0) return null;
    const hit = this.ctx().rendererFor(this.segs[i]).hit(this.segs[i], this.layouts[i], x, y);
    return hit ? { i, hit } : null;
  }

  point(x, y) {
    const best = this.hitAt(x, y), key = (h) => (h ? `${h.i}:${JSON.stringify(h.hit)}` : "");
    if (key(best) === key(this.hover)) return false;
    this.hover = best; this.draw(); this.caption(best);
    return true;
  }
  leave() { this.hover = null; this.draw(); this.caption(null); }
  picked() { return this.hover ? { seg: this.segs[this.hover.i], hit: this.hover.hit, ordinal: this.ordinal } : null; }

  caption(h) {
    if (!this.captionHost) return;
    const ctx = this.ctx();
    this.captionHost.replaceChildren(h ? ctx.rendererFor(this.segs[h.i]).caption(this.segs[h.i], h.hit, ctx) : this.hint());
  }
  // The caption before anything is pointed at; it says so when the weights the drawing reads are not
  // in the payload (withheld by the service, or deferred and not fetched).
  hint() {
    const d = document.createElement("span"), missing = this.graph.drawnTensorsMissing();
    d.className = "muted"; d.append(t("net.cap.stages_hint"));
    if (missing.length) d.append(" ", t(this.graph.weightsWithheld() ? "weights.withheld" : "net.cap.weights_missing"));
    return d;
  }

  animate(ms, done) { animateDrawing(this, ms, done); }

  // One answer: every segment's state from the explanation (null where it says nothing).
  showActivations(ex) {
    const ctx = this.ctx();
    this.acts = this.segs.map((s) => ctx.rendererFor(s).acts(s, ex, this.graph, ctx));
    if (this.g) this.layout(this.g, this.W, this.H);
    return this.acts.some((a) => a !== null);
  }
}

// ---- connectors between two segments' ports
// a line's colour: the sign of its weight, or of the share it carries in one answer; --muted2 when
// it has neither (a fan before anything is answered)
function lineColour(st, w) {
  const C = st.ctx.colors;
  return w === null ? C.mut : w >= 0 ? C.acc : C.neg;
}

const portValue = (p) => (p && typeof p.v === "number" && Number.isFinite(p.v) ? p.v : null);

// weights where both sides are units of one matrix: each target's strongest sources (3, or B.inTop:
// 6 into the head at the "every weight" level), each source's strongest target; otherwise a fan of
// at most FAN lines, each carrying the signed share of the unit it leaves (or reaches) once
// answered (a port's `v`), else nothing (w null)
export function connectorLines(A, B) {
  const outs = A.out || [], ins = B.in || [];
  if (!outs.length || !ins.length) return [];
  const M = B.inMatrix, top = B.inTop || 3;
  if (M && outs.every((p) => p.units) && ins.every((p) => Number.isInteger(p.unit))) {
    const w = (a, b) => { let s = 0, q = 0; for (const i of outs[a].units) { const v = M.w[ins[b].unit * M.in + i] || 0; s += v; q += v * v; } return Math.sign(s || 1) * Math.sqrt(q); };
    const seen = new Set(), lines = [];
    const add = (a, b) => { const k = a + ":" + b; if (!seen.has(k)) { seen.add(k); lines.push([a, b, w(a, b)]); } };
    ins.forEach((_, b) => outs.map((_, a) => [a, Math.abs(w(a, b))]).sort((p, q) => q[1] - p[1]).slice(0, top).forEach(([a]) => add(a, b)));
    outs.forEach((_, a) => { let best = 0, bw = -1; ins.forEach((_, b) => { const v = Math.abs(w(a, b)); if (v > bw) { bw = v; best = b; } }); add(a, best); });
    const mx = Math.max(1e-9, ...lines.map((e) => Math.abs(e[2])));
    return lines.map(([a, b, v]) => ({ a, b, w: v, n: Math.abs(v) / mx }));
  }
  const n = Math.max(outs.length, ins.length), step = Math.max(1, Math.ceil(n / FAN)), lines = [];
  for (let k = 0; k < n; k += step) {
    const a = outs.length === 1 ? 0 : Math.round(k * (outs.length - 1) / Math.max(1, n - 1));
    const b = ins.length === 1 ? 0 : Math.round(k * (ins.length - 1) / Math.max(1, n - 1));
    const v = portValue(outs[a]) ?? portValue(ins[b]);
    lines.push({ a, b, w: v, n: v === null ? .6 : Math.max(.15, Math.min(1, Math.abs(v))) });
  }
  return lines;
}

export function drawConnector(g, A, B, st) {
  // connector i joins segment i to i + 1: it grows in the second half of segment i's share of the phase
  const reach = Math.max(0, Math.min(1, (st.phase * st.n - st.i - .5) * 2)); if (reach <= 0) return;
  const lit = st.focus.active ? (st.focus.set.has("edges") || st.focus.all) : true;
  for (const e of connectorLines(A, B)) {
    const p = A.out[e.a], q = B.in[e.b], mid = (p.x + q.x) / 2;
    g.globalAlpha = (lit ? Math.min(1, e.n * 2) * .5 : .05) * (st.hover ? .5 : 1);
    g.strokeStyle = lineColour(st, e.w); g.lineWidth = .8;
    g.beginPath(); g.moveTo(p.x, p.y);
    if (reach < 1) g.lineTo(p.x + (q.x - p.x) * reach, p.y + (q.y - p.y) * reach); else g.bezierCurveTo(mid, p.y, mid, q.y, q.x, q.y);
    g.stroke();
  }
}

// The network's canvas. opts: {renderers (Map kind → renderer, renderers/index.js load()), caption
// (the caption host), onPick (a node clicked), isStatic (no pointer), k (units kept per layer),
// env: {roles, story (its input_groups)}}.
export class NetCanvas {
  constructor(canvas, graph, { renderers, caption = null, onPick = null, isStatic = false, k = 14, env = {} }) {
    Object.assign(this, { c: canvas, graph, renderers, captionHost: caption, onPick, isStatic, k });
    const groups = env.story ? env.story.input_groups : null;
    const fam = familiesOf({ groups, names: graph.inputs.names, units: graph.featureUnits() });
    this.env = { ...env, families: fam.families, problems: fam.problems };
    this.pure = graph.isPureDenseChain() && complete(graph);
    this.drawing = this.pure ? this.denseRun() : new SegmentDrawing(graph, { renderers, env: this.env, k, caption, isStatic });
    this.detail = null; this.mode = "simple";
    new ResizeObserver(() => { if (this.drawing.active) { this.resize(); this.draw(); } }).observe(canvas);
    if (!isStatic) this.listen();
    this.resize();
  }

  denseRun() {
    const dense = this.renderers.get("dense"), seg = this.graph.segments().find((s) => s.kind === "run") || { layers: [] };
    this.model = this.graph.runModel(seg);
    return new dense.Run(this.model, { families: this.env.families, k: this.k, roles: this.env.roles || null, caption: this.captionHost, waist: waistLayer(this.graph, this.model) });
  }

  listen() {
    const at = (e) => { const r = this.c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    this.c.addEventListener("click", (e) => {
      if (!this.drawing.active) return;
      this.drawing.point(...at(e));
      const node = this.drawing.picked();
      if (node && this.onPick) this.onPick(node, this);
    });
    this.c.addEventListener("mousemove", (e) => { if (this.drawing.active) this.drawing.point(...at(e)); });
    this.c.addEventListener("mouseleave", () => { if (this.drawing.active) this.drawing.leave(); });
  }

  // Whether the "every weight" level exists for this graph (the switch is shown only then): a pure
  // chain, or a graph any of whose stage kinds is drawn by a renderer that declares a detail level
  // (D4 §7.3 `detail`: a dense run's, a head's, a convolution's, a recurrent layer's, attention's, a
  // graph stage's).
  get hasDetail() { return this.pure || [...this.renderers.values()].some((r) => !!r && r.detail === true); }

  resize() {
    const r = this.c.getBoundingClientRect(), d = devicePixelRatio || 1, W = Math.max(320, r.width), H = Math.max(180, r.height);
    this.c.width = W * d; this.c.height = H * d;
    const g = this.c.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0);
    this.drawing.layout(g, W, H);
  }
  draw() { this.drawing.draw(); }
  animate(ms, done) { this.drawing.animate(ms, done); }
  caption(h = null) { this.drawing.caption(h); }

  // focus: a story focus (focus@1, or a story@1 region word), a focus Set already computed (the
  // guide's), or null
  setFocus(focus) {
    this.drawing.focus = new Focus(focus instanceof Set ? focus : this.graph.focusSet(focus));
    this.drawing.hover = null; this.draw();
  }

  // "simple" (families) or "detail" (every weight). A pure chain swaps drawings; a segmented graph
  // redraws its runs with every unit.
  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    if (!this.pure) { this.drawing.mode = mode; this.resize(); this.draw(); return; }
    this.drawing.active = false;
    if (mode === "detail") {
      if (!this.detail) this.detail = new (this.renderers.get("dense").Detail)(this.model, { roles: this.env.roles || null, caption: this.captionHost, isStatic: this.isStatic });
      this.simple = this.simple || this.drawing;
      this.drawing = this.detail;
    } else this.drawing = this.simple || this.drawing;
    this.drawing.active = true; this.drawing.hover = null;
    this.resize();
  }

  // The drawing of one answer; false when the explanation carries nothing this drawing reads.
  showActivations(ex) { this.resize(); return this.drawing.showActivations(ex); }
}

// Every weight matrix of a pure chain is in the payload (not withheld, not deferred) and joins its
// layer's widths: today's drawing needs them all; without them the chain is drawn as segments.
export function complete(graph) {
  const seg = graph.segments().find((s) => s.kind === "run") || { layers: [] };
  return graph.runModel(seg).layers.every((l) => l.W && l.W.out === l.out && l.W.in === l.in);
}

