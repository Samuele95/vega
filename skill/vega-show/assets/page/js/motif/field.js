// The field: the page's only decoration, a quiet picture on the fixed #field canvas under the veil.
// The engine sizes the canvas to the window, draws about thirty frames a second, pauses while the
// page is hidden and stays still under reduced motion (and for a motif that does not move). The
// picture is a motif of the closed set (motifs.js); the canvas carries its id in `data-motif`, as
// the vega-show catalog's `motifs.element` and `motifs.attribute` name them. A mark takes the alert
// colour only when the field was started with the lane's alert role: without it, none does.
//
// The boot starts the field with the page's motif and the lane's roles:
//   field.start(motifOf(page.motif.id), roles.hasAlert());  applyMark(headerMark, motif)
// and redraws it when the theme changes (the colours are read from the theme's tokens at each
// frame). A motif is given as a motif of the set or as its id (page.motif.id); either way it is
// looked up in the set. Until start() has run, the field draws nothing and asks for no frame: a
// resize, a theme switch or the page showing before the boot starts it leaves the canvas as it is.
// Under ?probe, window.__showcase.field() reads what the field drew (reading()).
import { css, reduced } from "../core/dom.js";
import { extendProbe } from "../core/probe.js";
import { MOTIFS, SEED, rng, drawGrid, motifOf } from "./motifs.js";

export const MOTIF_ATTRIBUTE = "data-motif";
const FRAME_MS = 33, MAX_DT = 100;          // about thirty frames a second; a long pause moves as 100 ms

// The colours a motif draws with, from the current theme's tokens.
export const palette = () => ({ grid: css("--grid"), lane: css("--lane"), acc: css("--accent"), alert: css("--alert") });

// The motif of the closed set that `motif` names: a motif object (by its id) or an id string.
// Anything else, or an id outside the set, is the grid.
const pick = (motif) => motifOf(typeof motif === "string" ? motif : motif && motif.id);

// The header mark of a motif, into the header's <svg> (its shapes only; the svg keeps its box). The
// shapes are the closed set's own template constants, looked up by the motif's id.
export function applyMark(svg, motif) {
  if (!svg) return;
  const m = pick(motif);
  svg.innerHTML = m.mark;
  svg.setAttribute(MOTIF_ATTRIBUTE, m.id);
}

export class Field {
  constructor(canvas, motif = MOTIFS.lanes) {
    this.c = canvas; this.g = canvas.getContext("2d"); this.motif = pick(motif); this.alert = false;
    this.W = 0; this.H = 0; this.scene = null; this.last = 0; this.started = false;
    this.running = false; this.pending = false; this.frames = 0; this.tinted = 0;
    this.loop = (t) => this.frame(t);
    addEventListener("resize", () => { if (this.started) { this.setup(); this.draw(0); } });
    document.addEventListener("visibilitychange", () => this.wake());
    extendProbe("field", () => this.reading(), { report: true });
  }

  // Starts the field with a motif of the closed set (a motif, or its id), or starts it again with
  // another. alert: whether the lane has an alert role (roles.hasAlert()); anything but true leaves
  // every mark in the accent.
  start(motif = this.motif, alert = false) {
    this.motif = pick(motif); this.alert = alert === true; this.tinted = 0; this.started = true;
    this.c.setAttribute(MOTIF_ATTRIBUTE, this.motif.id);
    this.setup(); this.draw(0);
    if (reduced() || this.motif.still) { this.running = false; return; }
    this.running = true; this.last = 0; this.request();
  }

  setup() {
    const d = devicePixelRatio || 1;
    this.W = innerWidth; this.H = innerHeight;
    this.c.width = this.W * d; this.c.height = this.H * d; this.g.setTransform(d, 0, 0, d, 0, 0);
    this.scene = this.motif.setup(this.W, this.H, rng(SEED));
  }

  draw(dt) {
    const g = this.g, C = palette();
    g.clearRect(0, 0, this.W, this.H);
    drawGrid(g, this.W, this.H, C);
    this.tinted += this.motif.draw(g, { W: this.W, H: this.H, scene: this.scene, dt, C, alert: this.alert });
    this.frames += 1;
  }

  // One animation frame is asked for at a time, however often the loop is woken.
  request() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(this.loop);
  }

  frame(t) {
    this.pending = false;
    if (!this.running) return;
    if (!this.last) this.last = t;
    const dt = t - this.last;
    if (dt >= FRAME_MS) { this.last = t; this.draw(Math.min(MAX_DT, dt)); }
    this.request();
  }

  // Pauses while the page is hidden; moves again when it shows (unless the picture stays still, or
  // the field has not been started).
  wake() {
    if (!this.started) return;
    this.running = !document.hidden && !reduced() && !this.motif.still;
    if (this.running) { this.last = 0; this.request(); }
  }

  redraw() { if (this.started) this.draw(0); }

  // What the field drew: its motif, the attribute on the canvas, whether it runs with an alert
  // role and whether any mark was drawn in the alert colour since it started.
  reading() {
    return {
      motif: this.motif.id, data_motif: this.c.getAttribute(MOTIF_ATTRIBUTE), alert: this.alert,
      alert_marks: this.tinted > 0, tinted: this.tinted, frames: this.frames, running: this.running,
    };
  }
}
