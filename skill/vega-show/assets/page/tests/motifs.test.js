// node --test: motif/motifs.js and motif/field.js. The six motifs of the closed set and their marks;
// `lanes` draws the hand-built page's field (0d06aea, index.html:615-655) exactly, on the fixed
// seed; the canvas carries `data-motif`; no mark takes the alert colour without an alert role.
//
// Here the field runs on a recording 2D context: every call and every state set, in order, whose
// sha256 is the canvas hash compared. With a browser ($CHROME, else google-chrome or chromium on
// PATH) the same checks run on real canvases, pixel by pixel (the last test; it is skipped when no
// browser is found and $CHROME is unset, and fails when $CHROME names none).
//
// The hand-built field is reproduced below with three names changed (its canvas id and two
// identifiers whose words the template's literal gate refuses); the first test proves it is
// 0d06aea's code up to that renaming, token by token, when the git history is reachable (this
// checkout, or $VEGA_SOURCE_REPO).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { execFileSync, spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import { MOTIFS, SEED, ALERT_SHARE, MARK_BOX, motifOf, rng } from "../js/motif/motifs.js";
import { Field, applyMark, MOTIF_ATTRIBUTE } from "../js/motif/field.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PAGE = path.resolve(HERE, "..");
const CATALOG = JSON.parse(readFileSync(path.resolve(PAGE, "../../scripts/catalog.json"), "utf8"));
const SOURCE_COMMIT = "0d06aea";
const SOURCE_PAGE = "vega-forge/scripts/backends/pytorch/templates/serving/static/index.html";
// The names the reproduction gives the three names it changes.
const RENAMED = Object.freeze(["marks", "alert", '"reference"']);

// 0d06aea index.html:615-655, with three names changed (RENAMED).
const TODAY_FIELD = `  const Field = (() => {
    const c = $("reference"), g = c.getContext("2d");
    let W = 0, H = 0, lanes = [], last = 0, running = false;
    const setup = () => {
      const d = devicePixelRatio || 1; W = innerWidth; H = innerHeight;
      c.width = W * d; c.height = H * d; g.setTransform(d, 0, 0, d, 0, 0);
      let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      lanes = Array.from({ length: 9 }, (_, i) => ({
        y: H * (.08 + .84 * i / 8) + (rnd() - .5) * 40, amp: 18 + rnd() * 46, f: .0018 + rnd() * .0026, ph: rnd() * 6.3,
        marks: Array.from({ length: 2 + Math.floor(rnd() * 4) }, () => ({ x: rnd() * W, v: .012 + rnd() * .03, alert: rnd() < .07 })),
      }));
    };
    const y = (l, x) => l.y + l.amp * Math.sin(x * l.f + l.ph);
    const draw = (dt) => {
      g.clearRect(0, 0, W, H);
      const grid = css("--grid"), lane = css("--lane"), calm = css("--accent"), alert = css("--alert");
      g.fillStyle = grid;
      for (let x = 24; x < W; x += 56) for (let yy = 24; yy < H; yy += 56) g.fillRect(x, yy, 1, 1);
      g.lineWidth = 1; g.strokeStyle = lane;
      for (const l of lanes) {
        g.beginPath(); for (let x = 0; x <= W; x += 16) { const yy = y(l, x); x ? g.lineTo(x, yy) : g.moveTo(x, yy); } g.stroke();
        for (const p of l.marks) {
          p.x += p.v * dt; if (p.x > W + 30) p.x = -30;
          g.globalAlpha = p.alert ? .34 : .2; g.strokeStyle = p.alert ? alert : calm; g.lineWidth = 1.6;
          g.beginPath(); g.moveTo(p.x - 14, y(l, p.x - 14)); g.lineTo(p.x, y(l, p.x)); g.stroke();
          g.globalAlpha = 1; g.strokeStyle = lane; g.lineWidth = 1;
        }
      }
    };
    const loop = (t) => {                       // about thirty frames a second is plenty
      if (!running) return;
      if (!last) last = t;
      const dt = t - last;
      if (dt >= 33) { last = t; draw(Math.min(100, dt)); }
      requestAnimationFrame(loop);
    };
    const start = () => { setup(); draw(0); if (reduced()) return; running = true; last = 0; requestAnimationFrame(loop); };
    addEventListener("resize", () => { setup(); draw(0); });
    document.addEventListener("visibilitychange", () => { running = !document.hidden && !reduced(); if (running) { last = 0; requestAnimationFrame(loop); } });
    return { start, redraw: () => draw(0) };
  })();`;

// 0d06aea index.html:340-341, the header mark's shapes.
const TODAY_MARK = `<path d="M2 7c5 0 5 5 10 5s5-5 10-5" opacity=".45"/><path d="M2 17c5 0 5-5 10-5s5 5 10 5" opacity=".45"/>
      <circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none"/><circle cx="5.2" cy="8.2" r="1.2" fill="currentColor" stroke="none" opacity=".7"/><circle cx="18.8" cy="15.8" r="1.2" fill="currentColor" stroke="none" opacity=".7"/>`;

const tight = (markup) => markup.replace(/>\s+</g, "><").trim();

// ---- the theme tokens, read from page.css (night and day)
function themeTokens() {
  const cssText = readFileSync(path.join(PAGE, "page.css"), "utf8"), out = {};
  for (const theme of ["night", "day"]) {
    const block = cssText.match(new RegExp(`\\[data-theme="${theme}"\\]\\{([^}]*)\\}`))[1];
    out[theme] = Object.fromEntries([...block.matchAll(/(--[\w-]+):([^;]+)/g)].map((m) => [m[1], m[2].trim()]));
  }
  return out;
}
const TOKENS = themeTokens();

// ---- a window for the field: size, pixel ratio, reduced motion, visibility, theme, frames
const ENV = { now: null };
const live = (key) => ({ get: () => ENV.now[key], configurable: true });
Object.defineProperties(globalThis, {
  window: { value: globalThis, configurable: true, writable: true },
  innerWidth: live("W"), innerHeight: live("H"), devicePixelRatio: live("dpr"),
});
const listen = (where) => (type, fn) => { (ENV.now.on[where][type] ||= []).push(fn); };
Object.assign(globalThis, {
  addEventListener: listen("window"),
  requestAnimationFrame: (cb) => ENV.now.frames.push(cb),
  matchMedia: () => ({ matches: ENV.now.reduce }),
  getComputedStyle: () => ({ getPropertyValue: (name) => ENV.now.tokens[name] ?? "" }),
  document: {
    get hidden() { return ENV.now.hidden; }, documentElement: {},
    addEventListener: listen("document"), getElementById: (id) => ENV.now.canvases[id],
  },
});

function environment({ W = 1440, H = 900, dpr = 1, reduce = false, theme = "night" } = {}) {
  ENV.now = { W, H, dpr, reduce, hidden: false, tokens: TOKENS[theme], frames: [], canvases: {}, on: { window: {}, document: {} } };
  return ENV.now;
}
const fire = (where, type) => (ENV.now.on[where][type] || []).forEach((fn) => fn());
function pump(times) {
  for (const t of times) for (const cb of ENV.now.frames.splice(0)) cb(t);
}

// Frame times with gaps under the 33 ms gate, at it, and over the 100 ms cap.
function frameTimes(n, start = 1000) {
  const gaps = [16, 17, 40, 33, 12, 250, 34, 20, 50, 66, 101, 5, 33, 48];
  const out = [start];
  for (let k = 1; k < n; k++) out.push(out[k - 1] + gaps[k % gaps.length]);
  return out;
}

// ---- the recording canvas: every 2D call and state set, in order
function recordingCanvas() {
  const log = [], attrs = new Map(), state = {};
  const g = new Proxy(state, {
    get: (t, k) => (k in t ? t[k] : (...args) => { log.push([k, ...args]); }),
    set: (t, k, v) => { t[k] = v; log.push([`${String(k)}=`, v]); return true; },
  });
  const canvas = {
    log, getContext: () => g,
    setAttribute: (k, v) => { attrs.set(k, String(v)); }, getAttribute: (k) => (attrs.has(k) ? attrs.get(k) : null),
  };
  return new Proxy(canvas, { set: (t, k, v) => { t[k] = v; log.push([`canvas.${String(k)}=`, v]); return true; } });
}
const hash = (log) => createHash("sha256").update(JSON.stringify(log)).digest("hex");

function firstDifference(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (JSON.stringify(a[i]) !== JSON.stringify(b[i])) return `entry ${i}: ${JSON.stringify(a[i])} vs ${JSON.stringify(b[i])}`;
  return a.length === b.length ? null : `lengths ${a.length} vs ${b.length}`;
}

function assertSameDrawing(got, want, what) {
  const diff = firstDifference(got, want);
  assert.equal(diff, null, `${what}: ${diff}`);
  assert.equal(hash(got), hash(want), what);
}

// The hand-built field, built over the window above.
function todayField() {
  const build = new Function("$", "css", "reduced", `${TODAY_FIELD}\nreturn Field;`);
  return build((id) => document.getElementById(id),
    (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim(),
    () => matchMedia("(prefers-reduced-motion: reduce)").matches);
}

// The field and the hand-built one side by side, through frames, a resize, a hide-and-show and a
// theme switch; their two drawings.
function sideBySide({ size, alert }) {
  const env = environment(size);
  const reference = recordingCanvas(), current = recordingCanvas();
  env.canvases = { reference, field: current };
  const today = todayField(), field = new Field(current);
  today.start(); field.start(MOTIFS.lanes, alert);
  pump(frameTimes(80));
  env.W = Math.round(size.W * .7); env.H = Math.round(size.H * 1.1); fire("window", "resize");
  pump(frameTimes(40, 9000));
  env.hidden = true; fire("document", "visibilitychange"); pump([12000, 12040]);
  env.hidden = false; fire("document", "visibilitychange");
  env.hidden = true; fire("document", "visibilitychange");
  env.hidden = false; fire("document", "visibilitychange");      // shown again before a frame came
  pump(frameTimes(40, 13000));
  env.tokens = TOKENS.day; today.redraw(); field.redraw();
  pump(frameTimes(30, 16000));
  return { today: reference.log, field: current.log, reading: field.reading() };
}

// The field alone, one motif, through frames; its drawing and its reading.
function drawAlone(motif, { size = {}, alert, times = frameTimes(80) } = {}) {
  const env = environment(size), canvas = recordingCanvas();
  env.canvases = { field: canvas };
  const field = new Field(canvas);
  if (alert === undefined) field.start(motif); else field.start(motif, alert);
  pump(times);
  return { log: canvas.log, field, env };
}

const styleSets = (log) => log.filter(([k]) => k === "strokeStyle=" || k === "fillStyle=");

function gitShow(spec) {
  for (const repo of [HERE, process.env.VEGA_SOURCE_REPO].filter(Boolean)) {
    try {
      return execFileSync("git", ["-C", repo, "show", spec], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], maxBuffer: 1 << 26 });
    } catch { /* not this repository */ }
  }
  return null;
}

// =============================================================== the hand-built page's field

// The tokens of a piece of JavaScript: names, numbers, strings, comments and single characters.
const tokens = (src) => src.match(/[A-Za-z_$][\w$]*|\d*\.?\d+(?:e[+-]?\d+)?|"(?:[^"\\]|\\.)*"|\/\/[^\n]*|\S/g);
const renamable = (tok) => /^[A-Za-z_$"]/.test(tok);

// How `ours` renames `theirs`: the same tokens but for names (or strings) replaced consistently,
// one for one; null when they differ otherwise.
function renaming(theirs, ours) {
  const a = tokens(theirs), b = tokens(ours), map = new Map(), back = new Map();
  if (a.length !== b.length) return null;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === b[i] && !renamable(a[i])) continue;
    if (!renamable(a[i]) || !renamable(b[i]) || (map.get(a[i]) ?? b[i]) !== b[i] || (back.get(b[i]) ?? a[i]) !== a[i]) return null;
    map.set(a[i], b[i]); back.set(b[i], a[i]);
  }
  return new Map([...map].filter(([from, to]) => from !== to));
}

test("the reference is 0d06aea's field and mark, up to three names", (t) => {
  const page = gitShow(`${SOURCE_COMMIT}:${SOURCE_PAGE}`);
  if (page === null) { t.skip(`git history with ${SOURCE_COMMIT} not reachable (this checkout, $VEGA_SOURCE_REPO)`); return; }
  const lines = page.split("\n"), theirs = lines.slice(614, 655).join("\n");
  const map = renaming(theirs, TODAY_FIELD);
  assert.ok(map, "the same code, token by token, but for consistently renamed names");
  assert.deepEqual([...map.values()].sort(), [...RENAMED].sort());
  assert.equal(theirs.split("\n").length, TODAY_FIELD.split("\n").length);
  assert.equal(renaming(theirs.replace("x += 56", "x += 57"), TODAY_FIELD), null, "a changed number is not a renaming");
  assert.equal(renaming("a(b); c(b)", "a(d); d(b)"), null, "a renaming is one for one");
  assert.equal(tight(lines.slice(339, 341).join("\n")), tight(TODAY_MARK));
});

test("lanes draws the hand-built field exactly when the lane has an alert role (canvas hash)", () => {
  for (const size of [{ W: 1440, H: 900, dpr: 1 }, { W: 390, H: 844, dpr: 3 }, { W: 1920, H: 1080, dpr: 1.25 }]) {
    const run = sideBySide({ size, alert: true });
    assert.ok(run.today.length > 10000, "the drawing is not trivial");
    assertSameDrawing(run.field, run.today, `lanes at ${size.W}x${size.H}@${size.dpr}`);
    assert.ok(run.reading.alert_marks, "the hand-built field tints some marks");
  }
});

test("without an alert role, lanes is the same drawing with every tinted mark in the accent", () => {
  const size = { W: 1440, H: 900, dpr: 1 };
  const run = sideBySide({ size, alert: false });
  const colours = new Map([[TOKENS.night["--alert"], TOKENS.night["--accent"]], [TOKENS.day["--alert"], TOKENS.day["--accent"]]]);
  const calm = run.today.map((entry) => {
    const [k, v] = entry;
    if (k === "globalAlpha=" && v === .34) return [k, .2];
    if (k === "strokeStyle=" && colours.has(v)) return [k, colours.get(v)];
    return entry;
  });
  assert.notEqual(hash(calm), hash(run.today), "the hand-built field has tinted marks to recolour");
  assertSameDrawing(run.field, calm, "lanes without an alert role");
  assert.equal(run.reading.alert_marks, false);
});

test("the lanes mark is the hand-built page's mark", () => {
  assert.equal(MOTIFS.lanes.mark, tight(TODAY_MARK));
});

// =============================================================== the closed set

test("the set is the catalog's six motifs, in its order, each admissible somewhere", () => {
  assert.deepEqual(Object.keys(MOTIFS), CATALOG.motifs.set);
  assert.equal(ALERT_SHARE, CATALOG.motifs.alert_share);
  assert.equal(SEED, 7);
  const admitted = new Set();
  for (const [modality, row] of Object.entries(CATALOG.motifs.by_modality)) {
    for (const id of row.admissible) { assert.equal(motifOf(id).id, id, `${modality}: ${id}`); admitted.add(id); }
    assert.equal(motifOf(row.base).id, row.base, `${modality}: base ${row.base}`);
  }
  assert.deepEqual([...admitted].sort(), Object.keys(MOTIFS).sort());
  // the element the field draws on, and the attribute it sets there
  assert.equal(CATALOG.motifs.element, "#field");
  assert.equal(CATALOG.motifs.attribute, MOTIF_ATTRIBUTE);
  assert.match(readFileSync(path.join(PAGE, "index.html"), "utf8"), /<canvas id="field"[^>]*>/);
});

test("every motif has the interface, frozen; only the grid stays still", () => {
  assert.ok(Object.isFrozen(MOTIFS));
  for (const [id, m] of Object.entries(MOTIFS)) {
    assert.ok(Object.isFrozen(m), id);
    assert.equal(m.id, id);
    assert.equal(typeof m.setup, "function", id);
    assert.equal(typeof m.draw, "function", id);
    assert.equal(typeof m.mark, "string", id);
    assert.equal(m.still, id === "grid", id);
  }
});

test("an id outside the set is the grid", () => {
  for (const id of ["nebula", "constructor", "toString", "__proto__", "", undefined, null]) assert.equal(motifOf(id), MOTIFS.grid, String(id));
  const { field, log } = drawAlone({ id: "nebula", draw: () => { throw new Error("not a motif of the set"); } });
  assert.equal(field.reading().motif, "grid");
  assert.ok(log.length > 0);
});

// Every element of a mark: its tag and attributes, and whatever text lies between the elements.
function markElements(markup) {
  const text = markup.replace(/<[^>]*>/g, "");
  const elements = [...markup.matchAll(/<(\/?)([a-zA-Z]+)([^>]*?)(\/?)>/g)].map((m) => ({
    close: m[1] === "/", tag: m[2], selfClosing: m[4] === "/",
    attrs: Object.fromEntries([...m[3].matchAll(/([\w-]+)="([^"]*)"/g)].map((a) => [a[1], a[2]])),
    rest: m[3].replace(/([\w-]+)="([^"]*)"/g, "").trim(),
  }));
  return { text, elements };
}
const SHAPES = new Set(["path", "circle", "ellipse", "line", "polyline", "polygon", "rect"]);
const ATTRS = new Set(["d", "cx", "cy", "r", "rx", "ry", "x", "y", "x1", "y1", "x2", "y2", "width", "height", "points",
  "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "opacity"]);

// The absolute points a path's data reaches (M, L, H, V, C, S, Q, T, Z in either case).
function pathPoints(d) {
  const out = [];
  let x = 0, y = 0, sx = 0, sy = 0;
  const ARITY = { m: 2, l: 2, h: 1, v: 1, c: 6, s: 4, q: 4, t: 2, z: 0 };
  for (const [, cmd, args] of d.matchAll(/([MLHVCSQTZmlhvcsqtz])([^MLHVCSQTZmlhvcsqtz]*)/g)) {
    const nums = (args.match(/-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?/gi) || []).map(Number), low = cmd.toLowerCase(), rel = cmd === low;
    if (low === "z") { x = sx; y = sy; continue; }
    for (let i = 0; i < nums.length; i += ARITY[low]) {
      const a = nums.slice(i, i + ARITY[low]);
      if (low === "h") x = rel ? x + a[0] : a[0];
      else if (low === "v") y = rel ? y + a[0] : a[0];
      else {
        for (let k = 0; k + 1 < a.length; k += 2) out.push([rel ? x + a[k] : a[k], rel ? y + a[k + 1] : a[k + 1]]);
        [x, y] = out[out.length - 1];
      }
      out.push([x, y]);
      if (low === "m" && i === 0) { sx = x; sy = y; }
    }
  }
  return out;
}

test("every mark is shapes only, in currentColor, inside the view box, and its own", () => {
  const seen = new Set();
  for (const [id, m] of Object.entries(MOTIFS)) {
    const { text, elements } = markElements(m.mark);
    assert.equal(text.trim(), "", `${id}: no text`);
    assert.ok(elements.length >= 3, `${id}: a mark of a few shapes`);
    for (const el of elements) {
      assert.ok(SHAPES.has(el.tag) && el.selfClosing && !el.close, `${id}: <${el.tag}> is a closed shape`);
      assert.equal(el.rest, "", `${id}: <${el.tag}> has only quoted attributes`);
      for (const [name, value] of Object.entries(el.attrs)) {
        assert.ok(ATTRS.has(name), `${id}: attribute ${name}`);
        if (name === "fill" || name === "stroke") assert.ok(["currentColor", "none"].includes(value), `${id}: ${name}=${value}`);
      }
      const pts = el.tag === "path" ? pathPoints(el.attrs.d)
        : [[+el.attrs.cx - +el.attrs.r, +el.attrs.cy - +el.attrs.r], [+el.attrs.cx + +el.attrs.r, +el.attrs.cy + +el.attrs.r]];
      for (const [x, y] of pts) assert.ok(x >= 0 && x <= MARK_BOX && y >= 0 && y <= MARK_BOX, `${id}: (${x}, ${y}) inside the ${MARK_BOX} box`);
    }
    assert.ok(!seen.has(m.mark), `${id}: its own mark`);
    seen.add(m.mark);
  }
});

test("applyMark puts the motif's own shapes into the header mark", () => {
  const svg = { innerHTML: "", attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } };
  for (const m of Object.values(MOTIFS)) {
    applyMark(svg, m);
    assert.equal(svg.innerHTML, m.mark);
    assert.equal(svg.attrs[MOTIF_ATTRIBUTE], m.id);
  }
  applyMark(svg, { id: "points", mark: "<text>not a shape</text>" });
  assert.equal(svg.innerHTML, MOTIFS.points.mark, "the shapes come from the set, never from the caller");
  applyMark(null, MOTIFS.lanes);
});

// =============================================================== the field

test("the canvas carries data-motif, and a restart with another motif changes it", () => {
  const env = environment(), canvas = recordingCanvas();
  env.canvases = { field: canvas };
  const field = new Field(canvas);
  assert.equal(canvas.getAttribute(MOTIF_ATTRIBUTE), null, "set when the field starts");
  for (const m of Object.values(MOTIFS)) {
    field.start(m, false);
    assert.equal(canvas.getAttribute(MOTIF_ATTRIBUTE), m.id);
    assert.equal(field.reading().data_motif, m.id);
    assert.equal(field.reading().motif, m.id);
  }
  assert.equal(env.frames.length <= 1, true, "one frame asked for at a time, however often restarted");
});

// A run long enough for the bracket of traces to reach positions that would be tinted.
const LONG_RUN = Object.freeze(Array.from({ length: 1600 }, (_, k) => 1000 + 100 * k));

test("no mark takes the alert colour without an alert role, in any motif, either theme", () => {
  const alerts = new Set([TOKENS.night["--alert"], TOKENS.day["--alert"]]);
  for (const m of Object.values(MOTIFS)) {
    for (const alert of [false, undefined, "yes", 1]) {
      for (const theme of ["night", "day"]) {
        const times = alert === false && theme === "night" ? LONG_RUN : frameTimes(80);
        const { log, field } = drawAlone(m, { size: { theme }, alert, times });
        const tinted = styleSets(log).filter(([, v]) => alerts.has(v));
        assert.deepEqual(tinted, [], `${m.id} (alert ${alert}, ${theme})`);
        assert.equal(log.some(([k, v]) => k === "globalAlpha=" && v === .34), false, `${m.id}: no alert alpha`);
        assert.equal(field.reading().alert_marks, false);
        assert.equal(field.reading().alert, false);
      }
    }
  }
});

test("with an alert role, the motifs with marks tint some of them; grid and contours none", () => {
  for (const m of Object.values(MOTIFS)) {
    const { log, field } = drawAlone(m, { alert: true, times: m.id === "traces" ? LONG_RUN : frameTimes(80) });
    const tinted = styleSets(log).filter(([, v]) => v === TOKENS.night["--alert"]).length;
    if (m.id === "grid" || m.id === "contours") {
      assert.equal(tinted, 0, m.id);
      assert.equal(field.reading().alert_marks, false, m.id);
    } else {
      assert.ok(tinted > 0, `${m.id}: some mark tinted`);
      assert.ok(field.reading().alert_marks, m.id);
    }
  }
});

test("every motif draws something finite over the grid, at every size", () => {
  const sizes = [{ W: 1440, H: 900, dpr: 1 }, { W: 390, H: 844, dpr: 3 }, { W: 2560, H: 1440, dpr: 1 }, { W: 40, H: 30, dpr: 1 }];
  for (const m of Object.values(MOTIFS)) {
    for (const size of sizes) {
      const { log } = drawAlone(m, { size, alert: true });
      for (const entry of log) for (const v of entry.slice(1)) if (typeof v === "number") assert.ok(Number.isFinite(v), `${m.id} ${size.W}x${size.H}: ${JSON.stringify(entry)}`);
      const drawn = log.filter(([k]) => k === "stroke" || k === "fill").length;
      if (m.id !== "grid" && size.W > 100) assert.ok(drawn > 0, `${m.id} ${size.W}x${size.H}: draws over the grid`);
      if (m.id === "grid") assert.equal(drawn, 0, "the grid is dots only");
    }
  }
});

test("the pictures are the seed's: the same every time, and no other randomness", () => {
  const random = Math.random;
  Math.random = () => { throw new Error("Math.random called"); };
  try {
    for (const m of Object.values(MOTIFS)) {
      const a = drawAlone(m, { alert: true }).log, b = drawAlone(m, { alert: true }).log;
      assert.equal(hash(a), hash(b), m.id);
    }
  } finally { Math.random = random; }
  const r = rng(SEED), M = 2147483647;          // Park–Miller from the seed 7, as the hand-built page
  for (let k = 0, s = 7; k < 5; k++) { s = (s * 16807) % M; assert.equal(r(), s / M); }
});

test("the field moves only when it may: reduced motion, a still motif, a hidden page", () => {
  let run = drawAlone(MOTIFS.lanes, { size: { reduce: true }, times: frameTimes(20) });
  assert.equal(run.env.frames.length, 0, "reduced motion: no frame asked for");
  assert.equal(run.field.reading().frames, 1, "reduced motion: drawn once");
  run.env.hidden = true; fire("document", "visibilitychange");
  run.env.hidden = false; fire("document", "visibilitychange");
  assert.equal(run.env.frames.length, 0, "reduced motion: none either after a hide and a show");
  run = drawAlone(MOTIFS.grid, { times: frameTimes(20) });
  assert.equal(run.env.frames.length, 0, "a still motif: no frame asked for");
  run.env.hidden = true; fire("document", "visibilitychange");
  run.env.hidden = false; fire("document", "visibilitychange");
  assert.equal(run.env.frames.length, 0, "a still motif: none either after a hide and a show");
  pump(frameTimes(20, 5000));
  assert.equal(run.field.reading().frames, 1, "a still motif: drawn once");
  assert.equal(run.field.reading().running, false);
  run = drawAlone(MOTIFS.points, { times: frameTimes(20) });
  assert.equal(run.env.frames.length, 1, "a moving motif: one frame asked for at a time");
  const drawn = run.field.reading().frames;
  run.env.hidden = true; fire("document", "visibilitychange"); pump(frameTimes(10, 20000));
  assert.equal(run.env.frames.length, 0, "hidden: the loop stops");
  assert.equal(run.field.reading().frames, drawn, "hidden: nothing drawn");
  run.env.hidden = false; fire("document", "visibilitychange"); fire("document", "visibilitychange");
  assert.equal(run.env.frames.length, 1, "shown: one loop again");
  pump(frameTimes(10, 30000));
  assert.ok(run.field.reading().frames > drawn, "shown: it moves again");
});

test("until start(), the field draws nothing and asks for no frame: a shown page, a resize, a theme switch", () => {
  for (const m of Object.values(MOTIFS)) {
    const env = environment(), canvas = recordingCanvas();
    env.canvases = { field: canvas };
    const field = new Field(canvas, m);
    env.hidden = true; fire("document", "visibilitychange");
    env.hidden = false; fire("document", "visibilitychange");      // a tab loaded in the background shows
    env.W = 800; env.H = 600; fire("window", "resize");
    env.tokens = TOKENS.day; field.redraw();                        // the theme switched before the boot
    assert.equal(env.frames.length, 0, `${m.id}: no frame asked for`);
    assert.doesNotThrow(() => pump(frameTimes(20)), m.id);
    assert.deepEqual(canvas.log, [], `${m.id}: nothing drawn, the canvas untouched`);
    assert.equal(canvas.getAttribute(MOTIF_ATTRIBUTE), null, `${m.id}: no data-motif`);
    const before = field.reading();
    assert.equal(before.frames, 0, m.id);
    assert.equal(before.running, false, m.id);
    field.start(m, true);                                           // and it starts as usual afterwards
    assert.equal(canvas.getAttribute(MOTIF_ATTRIBUTE), m.id);
    assert.equal(env.frames.length, m.still ? 0 : 1, m.id);
    pump(frameTimes(20, 5000));
    assert.ok(field.reading().frames >= 1, m.id);
    assert.equal(field.reading().frames > 1, !m.still, m.id);
  }
});

test("new Field(canvas).start() draws lanes, with no alert role", () => {
  const env = environment(), canvas = recordingCanvas();
  env.canvases = { field: canvas };
  const field = new Field(canvas);
  field.start();
  pump(frameTimes(40));
  const r = field.reading();
  assert.equal(r.motif, "lanes");
  assert.equal(r.data_motif, "lanes");
  assert.equal(r.alert, false);
  assert.equal(r.alert_marks, false);
  assert.equal(r.running, true);
});

test("a motif may be given by its id: start and applyMark look it up in the set", () => {
  const env = environment(), canvas = recordingCanvas();
  env.canvases = { field: canvas };
  const field = new Field(canvas, "contours");
  field.start();
  assert.equal(field.reading().motif, "contours");
  const svg = { innerHTML: "", attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } };
  for (const id of Object.keys(MOTIFS)) {
    field.start(id, true);
    assert.equal(field.reading().motif, id);
    assert.equal(canvas.getAttribute(MOTIF_ATTRIBUTE), id);
    assert.equal(field.reading().alert, true);
    applyMark(svg, id);
    assert.equal(svg.innerHTML, MOTIFS[id].mark);
    assert.equal(svg.attrs[MOTIF_ATTRIBUTE], id);
  }
  for (const id of ["nebula", "constructor", "__proto__", ""]) {
    field.start(id);
    assert.equal(field.reading().motif, "grid", id);
    applyMark(svg, id);
    assert.equal(svg.innerHTML, MOTIFS.grid.mark, id);
  }
});

// ---- the alert share: which marks of a scene may be tinted
const FLAGS = Object.freeze({
  points: (scene) => scene.pts.map((p) => p.alert),
  traces: (scene) => scene.hot,                                   // the bracket's positions
  lanes: (scene) => scene.flatMap((l) => l.marks.map((p) => p.alert)),
  links: (scene) => scene.nodes.map((n) => n.alert),
});
const share = (flags) => flags.filter(Boolean).length / flags.length;

// A canvas that draws nothing and records nothing, for long runs.
function quietCanvas() {
  const attrs = new Map(), g = new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } });
  return { getContext: () => g, setAttribute: (k, v) => { attrs.set(k, String(v)); }, getAttribute: (k) => (attrs.get(k) ?? null) };
}

test("the share of marks a scene may tint is the catalog's alert share, in every motif with marks", () => {
  const M = 2147483647, next = rng(20260929);
  const seeds = Array.from({ length: 400 }, () => 1 + Math.floor(next() * (M - 2)));
  for (const [id, flagsOf] of Object.entries(FLAGS)) {
    const flags = seeds.flatMap((seed) => flagsOf(MOTIFS[id].setup(1440, 900, rng(seed))));
    assert.ok(flags.length >= 10000, `${id}: ${flags.length} candidates`);
    assert.ok(Math.abs(share(flags) - ALERT_SHARE) < .01, `${id}: ${share(flags)} of ${flags.length} candidates flagged, not ${ALERT_SHARE}`);
  }
});

// How many marks a recorded drawing inked in the colour given: a filled or stroked path in it counts
// its round marks (arcs), or one when it has none (a stroke of lanes, the bracket of traces).
function inked(log, colour) {
  let fill = null, stroke = null, arcs = 0, n = 0;
  for (const [k, v] of log) {
    if (k === "fillStyle=") fill = v;
    else if (k === "strokeStyle=") stroke = v;
    else if (k === "beginPath") arcs = 0;
    else if (k === "arc") arcs += 1;
    else if ((k === "fill" && fill === colour) || (k === "stroke" && stroke === colour)) n += Math.max(arcs, 1);
  }
  return n;
}

test("with the role, the tinted marks are exactly the scene's flagged ones", () => {
  for (const id of ["points", "links", "lanes", "traces"]) {
    const { field, log } = drawAlone(MOTIFS[id], { alert: true, times: id === "traces" ? LONG_RUN : frameTimes(80) });
    const flagged = FLAGS[id](field.scene).filter(Boolean).length, r = field.reading();
    assert.ok(flagged > 0, `${id}: the seed flags some marks`);
    assert.ok(r.tinted > 0, `${id}: some mark tinted`);
    assert.equal(inked(log, TOKENS.night["--alert"]), r.tinted, `${id}: every mark drawn in the alert colour is counted, no other`);
    if (id !== "traces") assert.equal(r.tinted, r.frames * flagged, `${id}: every flagged mark tinted in every frame, no other`);
  }
  // traces: the bracket steps through its positions; over whole cycles, the frames it is tinted are
  // the flagged share of its positions (3 px a frame, 72 px a position, 50 positions: 1200 frames a cycle)
  const env = environment({ W: 200, H: 200 }), canvas = quietCanvas();
  env.canvases = { field: canvas };
  const field = new Field(canvas);
  field.start(MOTIFS.traces, true);
  pump(Array.from({ length: 12001 }, (_, k) => 1000 + 100 * k));
  const r = field.reading(), hot = share(FLAGS.traces(field.scene));
  assert.equal(r.frames, 12001);
  assert.ok(hot > 0, "the seed flags some positions");
  assert.ok(Math.abs(r.tinted / r.frames - hot) < .002, `traces: tinted in ${r.tinted} of ${r.frames} frames, flagged ${hot}`);
});

// The iso-levels a contours frame drew: each end of a segment lies on a cell edge, where the level is
// the heights of the edge's two corners interpolated linearly (marching squares); the levels found
// are grouped when they agree to 1e-9.
function contourLevels(log, scene) {
  const end = log.map(([k]) => k).lastIndexOf("stroke"), begin = log.map(([k]) => k).lastIndexOf("beginPath", end);
  const { nx, cell, v } = scene, at = (i, j) => v[j * nx + i];
  const level = ([x, y]) => {
    const u = x / cell, w = y / cell, i = Math.round(u), j = Math.round(w);
    const onColumn = Math.abs(u - i) < 1e-9, onRow = Math.abs(w - j) < 1e-9;
    if (onColumn && onRow) return at(i, j);
    if (onColumn) { const j0 = Math.floor(w); return at(i, j0) + (w - j0) * (at(i, j0 + 1) - at(i, j0)); }
    assert.ok(onRow, `(${x}, ${y}) on a cell edge`);
    const i0 = Math.floor(u);
    return at(i0, j) + (u - i0) * (at(i0 + 1, j) - at(i0, j));
  };
  const calls = log.slice(begin + 1, end), segments = [];
  for (let k = 0; k < calls.length; k += 2) {
    assert.deepEqual([calls[k][0], calls[k + 1][0]], ["moveTo", "lineTo"], "segments only");
    const a = level(calls[k].slice(1)), b = level(calls[k + 1].slice(1));
    assert.ok(Math.abs(a - b) < 1e-9, `a segment joins two crossings of one level (${a}, ${b})`);
    segments.push(a);
  }
  const groups = [];
  for (const z of segments.sort((p, q) => p - q)) {
    if (groups.length && z - groups[groups.length - 1].z < 1e-9) groups[groups.length - 1].n += 1; else groups.push({ z, n: 1 });
  }
  return groups;
}

test("contours draws five distinct iso-levels", () => {
  for (const size of [{ W: 1440, H: 900 }, { W: 390, H: 844, dpr: 3 }]) {
    const { log, field } = drawAlone(MOTIFS.contours, { size, times: frameTimes(40) });
    const groups = contourLevels(log, field.scene);
    assert.equal(groups.length, 5, `${size.W}x${size.H}: levels ${groups.map((g) => `${g.z.toFixed(3)}×${g.n}`).join(", ")}`);
    for (const g of groups) assert.ok(g.n >= 10, `level ${g.z}: ${g.n} segments`);
  }
});

// =============================================================== the modules' discipline

test("motifs.js imports nothing and the two modules hold no module-level state", () => {
  const motifs = readFileSync(path.join(PAGE, "js/motif/motifs.js"), "utf8");
  const field = readFileSync(path.join(PAGE, "js/motif/field.js"), "utf8");
  assert.equal(/^\s*import\b/m.test(motifs), false, "motifs.js is pure");
  for (const [name, src] of [["motifs.js", motifs], ["field.js", field]]) {
    assert.equal(/^(?:export\s+)?(?:let|var)\s/m.test(src), false, `${name}: no module-level let or var`);
    assert.equal(/Math\.random/.test(src), false, `${name}: no randomness but the seed's`);
    const nonBlank = src.split("\n").filter((l) => l.trim()).length;
    assert.ok(nonBlank <= 400, `${name}: ${nonBlank} non-blank lines`);
  }
});

// =============================================================== in a browser, pixel by pixel

function findChrome() {
  const named = process.env.CHROME;
  if (named) return { chrome: named, required: true };
  for (const c of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]) {
    try { return { chrome: execFileSync("which", [c], { encoding: "utf8" }).trim(), required: false }; } catch { /* next */ }
  }
  return { chrome: null, required: false };
}

const HARNESS_JS = `
import { Field, applyMark } from "./page/js/motif/field.js";
import { MOTIFS } from "./page/js/motif/motifs.js";
import { $, css, reduced } from "./page/js/core/dom.js";
const queue = [];
window.requestAnimationFrame = (cb) => queue.push(cb);          // the frames come at the times below
const times = (n, start) => { const out = [start]; for (let k = 1; k < n; k++) out.push(out[k - 1] + [16, 17, 40, 33, 250, 34, 50][k % 7]); return out; };
const pump = (ts) => { for (const t of ts) queue.splice(0).forEach((cb) => cb(t)); };
const pixels = (id) => { const c = $(id); return c.getContext("2d").getImageData(0, 0, c.width, c.height).data; };
const hex = async (data) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", data))].map((b) => b.toString(16).padStart(2, "0")).join("");
const rgbOf = (v) => { const c = document.createElement("canvas").getContext("2d"); c.fillStyle = v; c.fillRect(0, 0, 1, 1); return [...c.getImageData(0, 0, 1, 1).data]; };
function hsv([r, g, b]) {
  const mx = Math.max(r, g, b) / 255, mn = Math.min(r, g, b) / 255, d = mx - mn;
  let h = 0;
  if (d) h = mx === r / 255 ? ((g - b) / 255 / d) % 6 : mx === g / 255 ? (b - r) / 255 / d + 2 : (r - g) / 255 / d + 4;
  return [(h * 60 + 360) % 360, mx ? d / mx : 0, mx];
}
// P-ROLE: pixels whose hue is within 12 degrees of the alert colour's, saturation > .4, value > .3
function alertPixels(data, alertRgb) {
  const [ha] = hsv(alertRgb);
  let n = 0, drawn = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (!data[i + 3]) continue;
    drawn += 1;
    const [h, s, v] = hsv([data[i], data[i + 1], data[i + 2]]), dh = Math.min(Math.abs(h - ha), 360 - Math.abs(h - ha));
    if (dh <= 12 && s > .4 && v > .3) n += 1;
  }
  return { n, drawn };
}
function distinctColours(data, width, height) {
  const seen = new Set();
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
    const i = (Math.floor((y + .5) * height / 32) * width + Math.floor((x + .5) * width / 32)) * 4;
    seen.add(data.slice(i, i + 4).join(","));
  }
  return seen.size;
}
async function compare(label) {
  const a = pixels("reference"), b = pixels("field");
  let differing = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) differing += 1;
  return { label, reference: await hex(a), field: await hex(b), differing, bytes: a.length };
}
async function run() {
  const out = { size: [innerWidth, innerHeight, devicePixelRatio], lanes: [], motifs: {}, marks: {} };
  const today = window.__todayField($, css, reduced), field = new Field($("field"));
  today.start(); field.start(MOTIFS.lanes, true);
  pump(times(90, 1000));
  out.lanes.push(await compare("night"));
  document.documentElement.dataset.theme = "day"; today.redraw(); field.redraw();
  pump(times(40, 20000));
  out.lanes.push(await compare("day"));
  out.lanes_tinted = alertPixels(pixels("field"), rgbOf(css("--alert")));
  document.documentElement.dataset.theme = "night";
  for (const m of Object.values(MOTIFS)) {
    field.start(m, false);
    pump(times(90, 40000));
    const data = pixels("field"), c = $("field");
    out.motifs[m.id] = { data_motif: c.getAttribute("data-motif"), alert: alertPixels(data, rgbOf(css("--alert"))),
                         colours: distinctColours(data, c.width, c.height), reading: field.reading() };
    const svg = $("mark");
    applyMark(svg, m);
    const box = svg.getBBox();
    out.marks[m.id] = { shapes: svg.children.length, text: svg.textContent.trim(), box: [box.x, box.y, box.width, box.height] };
  }
  return out;
}
run().then((out) => { $("out").textContent = JSON.stringify(out); }, (e) => { $("out").textContent = JSON.stringify({ error: String(e && e.stack || e) }); });
`;

const HARNESS_HTML = `<!doctype html>
<html lang="en" data-theme="night"><head><meta charset="utf-8"><title>motifs</title>
<link rel="stylesheet" href="page/page.css"></head><body>
<canvas id="reference"></canvas><canvas id="field"></canvas>
<svg id="mark" class="mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"></svg>
<pre id="out"></pre>
<script>window.__todayField = function ($, css, reduced) {
${TODAY_FIELD}
return Field; };</script>
<script type="module" src="harness.js"></script>
</body></html>`;

const TYPES = { ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json" };

function serve() {
  const server = createServer((req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");
    const send = (status, type, body) => { res.writeHead(status, { "content-type": `${type}; charset=utf-8` }); res.end(body); };
    if (url.pathname === "/harness.html") return send(200, TYPES[".html"], HARNESS_HTML);
    if (url.pathname === "/harness.js") return send(200, TYPES[".js"], HARNESS_JS);
    const file = path.resolve(PAGE, "." + decodeURIComponent(url.pathname).replace(/^\/page\//, "/"));
    if (!url.pathname.startsWith("/page/") || !file.startsWith(PAGE + path.sep)) return send(404, "text/plain", "not found");
    try { return send(200, TYPES[path.extname(file)] || "application/octet-stream", readFileSync(file)); } catch { return send(404, "text/plain", "not found"); }
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

function browse(chrome, url) {
  const tmp = os.tmpdir().length > 60 ? "/tmp" : os.tmpdir();
  const profile = mkdtempSync(path.join(tmp, "motifs-"));
  const args = ["--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    `--user-data-dir=${profile}`, "--window-size=1440,900", "--force-device-scale-factor=1", "--hide-scrollbars",
    "--virtual-time-budget=60000", "--dump-dom", url];
  return new Promise((resolve, reject) => {
    const p = spawn(chrome, args, { env: { ...process.env, TMPDIR: tmp }, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    p.stdout.on("data", (d) => { stdout += d; });
    p.stderr.on("data", (d) => { stderr += d; });
    const timer = setTimeout(() => p.kill("SIGKILL"), 170000);
    p.on("error", (e) => { clearTimeout(timer); reject(e); });
    p.on("close", (code) => {
      clearTimeout(timer);
      rmSync(profile, { recursive: true, force: true });
      const m = stdout.match(/<pre id="out">([\s\S]*?)<\/pre>/);
      if (!m || !m[1]) return reject(new Error(`no result from the browser (exit ${code}): ${stderr.slice(-1500)}`));
      resolve(JSON.parse(m[1].replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&")));
    });
  });
}

test("in a browser: lanes is pixel-identical to the hand-built field; no alert pixel without the role", { timeout: 200000 }, async (t) => {
  const { chrome, required } = findChrome();
  if (!chrome) { t.skip("no browser ($CHROME unset; no google-chrome or chromium on PATH)"); return; }
  const server = await serve();
  let out;
  try { out = await browse(chrome, `http://127.0.0.1:${server.address().port}/harness.html`); } catch (e) {
    if (required) throw e;
    t.skip(`the browser did not run: ${e.message}`); return;
  } finally { server.close(); }
  assert.equal(out.error, undefined, out.error);
  t.diagnostic(`${chrome} at ${out.size.join("x")}: lanes ${out.lanes.map((c) => `${c.label} sha256:${c.field.slice(0, 16)}`).join(", ")}; `
    + `pixels drawn ${Object.entries(out.motifs).map(([id, r]) => `${id} ${r.alert.drawn}`).join(", ")}`);
  for (const c of out.lanes) {
    assert.equal(c.differing, 0, `${c.label}: ${c.differing} of ${c.bytes} bytes differ`);
    assert.equal(c.field, c.reference, `${c.label}: canvas hash`);
  }
  assert.ok(out.lanes_tinted.n > 0, "with the role, the tinted marks show in the alert hue (the detector works)");
  for (const [id, r] of Object.entries(out.motifs)) {
    assert.equal(r.data_motif, id);
    assert.equal(r.alert.n, 0, `${id}: ${r.alert.n} alert-hue pixels of ${r.alert.drawn} drawn`);
    assert.equal(r.reading.alert_marks, false, id);
    assert.ok(r.colours >= 2, `${id}: ${r.colours} distinct colours in the 32x32 sample`);
    if (id !== "grid") assert.ok(r.alert.drawn > 1000, `${id}: ${r.alert.drawn} pixels drawn`);
  }
  for (const [id, mk] of Object.entries(out.marks)) {
    assert.ok(mk.shapes >= 3, `${id}: ${mk.shapes} shapes`);
    assert.equal(mk.text, "", `${id}: no text`);
    const [x, y, w, h] = mk.box;
    assert.ok(x >= 0 && y >= 0 && x + w <= MARK_BOX && y + h <= MARK_BOX && w > 8 && h > 8, `${id}: box ${mk.box}`);
  }
});
