// The motifs: the closed set of pictures the page's field can draw (the vega-show catalog's
// `motifs.set`, R-E4), each a quiet background over the 56 px dot grid, with a mark for the header.
// A motif never encodes data. Every parameter below is a template constant and every picture comes
// from the template's seed, so nothing on the background is a fact. The share of marks that may take
// the alert colour is a constant too (the catalog's `motifs.alert_share`), and a mark takes it only
// when the lane has an alert role.
//
// A motif is { id, still, setup(W, H, rnd) -> scene, draw(g, frame) -> tinted, mark }:
//   setup  builds the scene of a W×H field from the seeded generator rnd (a moving motif keeps rnd
//          in its scene, so its motion comes from the seed too);
//   draw   draws one frame on the 2D context g, with frame = { W, H, scene, dt, C, alert }: dt is the
//          time since the last frame in ms (0 for a redraw in place, which moves nothing), C the
//          colours { grid, lane, acc, alert } and alert whether the lane has an alert role. It
//          returns how many marks it drew in the alert colour: always 0 when alert is false;
//   still  the picture does not move (the field draws it once);
//   mark   the header mark: SVG shapes only, no text, in currentColor on a 24×24 view box.
// The module is pure: it reads no page state and touches no DOM (field.js runs it).

export const SEED = 7;               // the template's seed: the same picture on every load
export const ALERT_SHARE = .07;      // the share of marks that may be tinted (catalog motifs.alert_share)
export const MARK_BOX = 24;          // the header mark's view box

const MARK_ALPHA = .2, ALERT_ALPHA = .34, TAU = 2 * Math.PI;

// Park–Miller: the hand-built page's generator, so `lanes` draws its picture exactly.
export function rng(seed) {
  let s = seed;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}

// The 56 px dot grid every motif draws under its picture.
export function drawGrid(g, W, H, C) {
  g.fillStyle = C.grid;
  for (let x = 24; x < W; x += 56) for (let yy = 24; yy < H; yy += 56) g.fillRect(x, yy, 1, 1);
}

// A mark's ink: the accent at the marks' alpha, or the alert colour at its own alpha when hot.
function ink(g, C, hot) {
  g.globalAlpha = hot ? ALERT_ALPHA : MARK_ALPHA;
  g.fillStyle = g.strokeStyle = hot ? C.alert : C.acc;
}

// Round marks at the points given, in one path; the number drawn in the alert colour.
function dots(g, pts, { r, C, hot }) {
  if (!pts.length) return 0;
  ink(g, C, hot);
  g.beginPath();
  for (const p of pts) { g.moveTo(p.x + r, p.y); g.arc(p.x, p.y, r, 0, TAU); }
  g.fill();
  g.globalAlpha = 1;
  return hot ? pts.length : 0;
}

// ---- the header marks: SVG shapes in currentColor (the svg element gives stroke and fill defaults)
const num = (v) => String(Math.round(v * 100) / 100);
const shade = (opacity) => (opacity ? ` opacity="${opacity}"` : "");
const dot = (cx, cy, r, opacity) => `<circle cx="${cx}" cy="${cy}" r="${r}" fill="currentColor" stroke="none"${shade(opacity)}/>`;
const ring = (cx, cy, r, opacity) => `<circle cx="${cx}" cy="${cy}" r="${r}" stroke-width="1.1"${shade(opacity)}/>`;
const line = (d, opacity) => `<path d="${d}"${shade(opacity)}/>`;

// A closed wavy ring around (12, 12): radius R with `waves` bumps of relative height amp, drawn
// through the midpoints of 16 samples, so the curve is smooth.
function wavyRing({ R, waves, amp, phase, opacity }) {
  const pts = Array.from({ length: 16 }, (_, k) => {
    const a = TAU * k / 16, r = R * (1 + amp * Math.sin(waves * a + phase));
    return [12 + r * Math.cos(a), 12 + r * Math.sin(a)];
  });
  const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
  const start = mid(pts[15], pts[0]);
  const d = [`M${num(start[0])} ${num(start[1])}`].concat(pts.map((p, k) => {
    const m = mid(p, pts[(k + 1) % 16]);
    return `Q${num(p[0])} ${num(p[1])} ${num(m[0])} ${num(m[1])}`;
  }));
  return line(d.join("") + "Z", opacity);
}

const MARKS = Object.freeze({
  grid: [6, 12, 18].flatMap((y) => [6, 12, 18].map((x) => (x === 12 && y === 12 ? dot(x, y, 2.2) : dot(x, y, 1.2, ".7")))).join(""),
  points: dot(12, 12, 2.2) + ring(5.5, 8, 1.7, ".7") + ring(17.5, 6.5, 1.7, ".7") + ring(7, 17.5, 1.7, ".7") + ring(18, 16, 1.7, ".7"),
  traces: line("M2 12c2 0 3-2.5 5-2.5s3 5 5 5 3-5 5-5 3 2.5 5 2.5", ".45") + line("M9.5 6H7.5v12h2") + line("M14.5 6h2v12h-2"),
  lanes: line("M2 7c5 0 5 5 10 5s5-5 10-5", ".45") + line("M2 17c5 0 5-5 10-5s5 5 10 5", ".45")
    + dot(12, 12, 2.2) + dot(5.2, 8.2, 1.2, ".7") + dot(18.8, 15.8, 1.2, ".7"),
  links: line("M5 6L18 5M5 6l7 7M18 5l-6 8M12 13l7 6", ".45")
    + dot(5, 6, 1.9) + dot(18, 5, 1.9) + dot(12, 13, 1.9) + dot(19, 19, 1.9) + dot(11.5, 5.5, 1.1, ".7"),
  contours: wavyRing({ R: 10, waves: 5, amp: .07, phase: 0, opacity: ".45" })
    + wavyRing({ R: 6.4, waves: 3, amp: .08, phase: 1.3, opacity: ".7" })
    + wavyRing({ R: 2.9, waves: 2, amp: .12, phase: .4 }),
});

// ---- grid: the dot grid alone, still
const grid = Object.freeze({
  id: "grid", still: true, mark: MARKS.grid,
  setup: () => null,
  draw: () => 0,
});

// ---- points: about sixty faint points in four soft clusters, each wandering near its home
const WALK = Object.freeze({ kick: .00018, keep: 1500, back: 8000 });

// A damped random walk: the velocity takes a small kick and loses part of itself at every frame,
// and the displacement from home decays slowly, so a point wanders and never leaves its cluster.
function wander(p, rnd, dt) {
  const kick = WALK.kick * dt, keep = Math.max(0, 1 - dt / WALK.keep), back = Math.max(0, 1 - dt / WALK.back);
  p.vx = p.vx * keep + (rnd() - .5) * kick; p.vy = p.vy * keep + (rnd() - .5) * kick;
  p.dx = p.dx * back + p.vx * dt; p.dy = p.dy * back + p.vy * dt;
  p.x = p.hx + p.dx; p.y = p.hy + p.dy;
}

const points = Object.freeze({
  id: "points", still: false, mark: MARKS.points,
  setup(W, H, rnd) {
    const clusters = Array.from({ length: 4 }, (_, k) => ({    // one cluster in each quarter of the field
      x: W * (.27 + .46 * (k % 2) + (rnd() - .5) * .24), y: H * (.3 + .4 * Math.floor(k / 2) + (rnd() - .5) * .24), s: 40 + rnd() * 60,
    }));
    const soft = () => rnd() + rnd() + rnd() - 1.5;
    const pts = Array.from({ length: 60 }, (_, i) => {
      const c = clusters[i % 4], hx = c.x + soft() * c.s, hy = c.y + soft() * c.s;
      return { hx, hy, x: hx, y: hy, dx: 0, dy: 0, vx: 0, vy: 0, alert: rnd() < ALERT_SHARE };
    });
    return { pts, rnd };
  },
  draw(g, { scene, dt, C, alert }) {
    if (dt > 0) for (const p of scene.pts) wander(p, scene.rnd, dt);
    const hot = scene.pts.filter((p) => alert && p.alert);
    dots(g, scene.pts.filter((p) => !(alert && p.alert)), { r: 2, C, hot: false });
    return dots(g, hot, { r: 2, C, hot: true });
  },
});

// ---- traces: five faint signals scrolling left, a bracket sliding along one of them
const STEP = 12;                      // px between the samples of a trace

// The height of a trace at its j-th sample: two sines and a little fixed noise, all tied to the
// sample, so the signal scrolls without changing shape.
function traceY(l, j) {
  const u = j * STEP;
  return l.y + l.a1 * Math.sin(u * l.f1 + l.p1) + l.a2 * Math.sin(u * l.f2 + l.p2) + l.noise[j % l.noise.length];
}

function tracePath(g, l, W) {
  const j0 = Math.floor(l.off / STEP), j1 = Math.ceil((l.off + W) / STEP);
  g.beginPath();
  for (let j = j0; j <= j1; j++) { const x = j * STEP - l.off, y = traceY(l, j); j === j0 ? g.moveTo(x, y) : g.lineTo(x, y); }
  g.stroke();
}

// The bracket: a window's two ends over one trace, moving right; its positions follow each other
// a bracket's width apart, and a fixed share of them is tinted when the lane has an alert role.
function drawBracket(g, { W, scene, dt, C, alert }) {
  const b = scene.bracket, l = scene.lines[b.line], serif = 5;
  b.pos += b.v * dt;
  const x = (b.pos % (W + 2 * b.w)) - b.w, top = l.y - (l.a1 + l.a2 + 10), bottom = l.y + (l.a1 + l.a2 + 10);
  const hot = alert && scene.hot[Math.floor(b.pos / b.w) % scene.hot.length];
  ink(g, C, hot); g.lineWidth = 1.6;
  g.beginPath();
  g.moveTo(x + serif, top); g.lineTo(x, top); g.lineTo(x, bottom); g.lineTo(x + serif, bottom);
  g.moveTo(x + b.w - serif, top); g.lineTo(x + b.w, top); g.lineTo(x + b.w, bottom); g.lineTo(x + b.w - serif, bottom);
  g.stroke();
  g.globalAlpha = 1; g.lineWidth = 1;
  return hot ? 1 : 0;
}

const traces = Object.freeze({
  id: "traces", still: false, mark: MARKS.traces,
  setup(W, H, rnd) {
    const lines = Array.from({ length: 5 }, (_, i) => ({
      y: H * (.18 + .64 * i / 4) + (rnd() - .5) * 30,
      a1: 10 + rnd() * 18, f1: .004 + rnd() * .006, p1: rnd() * 6.3,
      a2: 3 + rnd() * 6, f2: .016 + rnd() * .02, p2: rnd() * 6.3,
      noise: Array.from({ length: 64 }, () => (rnd() - .5) * 2.4), v: .016 + rnd() * .018, off: rnd() * 4000,
    }));
    const hot = Array.from({ length: 50 }, () => rnd() < ALERT_SHARE);
    return { lines, hot, bracket: { line: Math.floor(rnd() * 5), pos: rnd() * W, v: .03, w: 72 } };
  },
  draw(g, frame) {
    g.lineWidth = 1; g.strokeStyle = frame.C.lane;
    for (const l of frame.scene.lines) { l.off += l.v * frame.dt; tracePath(g, l, frame.W); }
    return drawBracket(g, frame);
  },
});

// ---- lanes: the hand-built page's field, exactly (nine sine lanes with a few slow marks each)
const lanes = Object.freeze({
  id: "lanes", still: false, mark: MARKS.lanes,
  setup(W, H, rnd) {
    return Array.from({ length: 9 }, (_, i) => ({
      y: H * (.08 + .84 * i / 8) + (rnd() - .5) * 40, amp: 18 + rnd() * 46, f: .0018 + rnd() * .0026, ph: rnd() * 6.3,
      marks: Array.from({ length: 2 + Math.floor(rnd() * 4) }, () => ({ x: rnd() * W, v: .012 + rnd() * .03, alert: rnd() < ALERT_SHARE })),
    }));
  },
  draw(g, { W, scene, dt, C, alert }) {
    const y = (l, x) => l.y + l.amp * Math.sin(x * l.f + l.ph);
    let tinted = 0;
    g.lineWidth = 1; g.strokeStyle = C.lane;
    for (const l of scene) {
      g.beginPath(); for (let x = 0; x <= W; x += 16) { const yy = y(l, x); x ? g.lineTo(x, yy) : g.moveTo(x, yy); } g.stroke();
      for (const p of l.marks) {
        const hot = alert && p.alert;
        p.x += p.v * dt; if (p.x > W + 30) p.x = -30;
        g.globalAlpha = hot ? ALERT_ALPHA : MARK_ALPHA; g.strokeStyle = hot ? C.alert : C.acc; g.lineWidth = 1.6;
        g.beginPath(); g.moveTo(p.x - 14, y(l, p.x - 14)); g.lineTo(p.x, y(l, p.x)); g.stroke();
        g.globalAlpha = 1; g.strokeStyle = C.lane; g.lineWidth = 1;
        if (hot) tinted += 1;
      }
    }
    return tinted;
  },
});

// ---- links: a faint random geometric graph, slow pulses travelling along its edges
function pulseOn([a, b], rnd) {
  const forward = rnd() < .5;
  return { from: forward ? a : b, to: forward ? b : a, t: rnd(), v: .018 + rnd() * .014 };
}

// Moves a pulse along its edge; at the edge's end it takes one of the node's edges at random.
function travel(p, scene, dt) {
  const a = scene.nodes[p.from], b = scene.nodes[p.to];
  p.t += p.v * dt / Math.max(1, Math.hypot(b.x - a.x, b.y - a.y));
  if (p.t < 1) return;
  p.from = p.to; p.to = b.adj[Math.floor(scene.rnd() * b.adj.length)]; p.t = 0;
}

const links = Object.freeze({
  id: "links", still: false, mark: MARKS.links,
  setup(W, H, rnd) {
    const nodes = Array.from({ length: 40 }, () => ({ x: W * (.04 + .92 * rnd()), y: H * (.06 + .88 * rnd()), alert: rnd() < ALERT_SHARE, adj: [] }));
    const reach = Math.sqrt(3 * W * H / (Math.PI * nodes.length)), edges = [];   // about three edges a node
    nodes.forEach((a, i) => nodes.forEach((b, j) => {
      if (i < j && Math.hypot(a.x - b.x, a.y - b.y) < reach) { a.adj.push(j); b.adj.push(i); edges.push([i, j]); }
    }));
    const pulses = edges.length ? Array.from({ length: 8 }, () => pulseOn(edges[Math.floor(rnd() * edges.length)], rnd)) : [];
    return { nodes, edges, pulses, rnd };
  },
  draw(g, { scene, dt, C, alert }) {
    const { nodes } = scene;
    g.lineWidth = 1; g.strokeStyle = C.lane;
    g.beginPath();
    for (const [i, j] of scene.edges) { g.moveTo(nodes[i].x, nodes[i].y); g.lineTo(nodes[j].x, nodes[j].y); }
    g.stroke();
    if (dt > 0) for (const p of scene.pulses) travel(p, scene, dt);
    const at = (p) => ({ x: nodes[p.from].x + (nodes[p.to].x - nodes[p.from].x) * p.t, y: nodes[p.from].y + (nodes[p.to].y - nodes[p.from].y) * p.t });
    dots(g, nodes.filter((n) => !(alert && n.alert)), { r: 2, C, hot: false });
    dots(g, scene.pulses.map(at), { r: 1.6, C, hot: false });
    return dots(g, nodes.filter((n) => alert && n.alert), { r: 2, C, hot: true });
  },
});

// ---- contours: five faint iso-lines of a slowly shifting sum of Gaussians (marching squares)
const CELL = 24, LEVELS = Object.freeze([.18, .34, .5, .66, .82]);
// Marching squares: corners top-left, top-right, bottom-right, bottom-left above the level set bits
// 1, 2, 4, 8; each case lists the edges (0 top, 1 right, 2 bottom, 3 left) its segments join.
const SEGMENTS = Object.freeze([[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 0], [1, 2]], [[0, 2]], [[3, 2]],
  [[2, 3]], [[0, 2]], [[0, 1], [2, 3]], [[1, 2]], [[1, 3]], [[0, 1]], [[3, 0]], []].map(Object.freeze));
const EDGE_CORNERS = Object.freeze([[0, 1], [1, 2], [3, 2], [0, 3]].map(Object.freeze));
const CORNER_AT = Object.freeze([[0, 0], [1, 0], [1, 1], [0, 1]].map(Object.freeze));

function sampleHeights(sc) {
  const hills = sc.hills.map((h) => ({
    x: h.x + h.ax * Math.sin(sc.t * h.w + h.p), y: h.y + h.ay * Math.cos(sc.t * h.w * .8 + h.p), k: -1 / (2 * h.s * h.s), a: h.a,
  }));
  for (let j = 0; j < sc.ny; j++) for (let i = 0; i < sc.nx; i++) {
    let z = 0;
    for (const h of hills) { const dx = i * CELL - h.x, dy = j * CELL - h.y; z += h.a * Math.exp((dx * dx + dy * dy) * h.k); }
    sc.v[j * sc.nx + i] = z;
  }
}

// Where the level crosses edge e of cell (i, j), whose corner heights are in sc.z.
function crossing(sc, { i, j, e, level }) {
  const [c0, c1] = EDGE_CORNERS[e], z0 = sc.z[c0], z1 = sc.z[c1];
  const t = z1 === z0 ? .5 : (level - z0) / (z1 - z0);
  const [x0, y0] = CORNER_AT[c0], [x1, y1] = CORNER_AT[c1];
  return [(i + x0 + (x1 - x0) * t) * CELL, (j + y0 + (y1 - y0) * t) * CELL];
}

function isoLines(g, sc, level) {
  const { nx, ny, v, z } = sc;
  for (let j = 0; j + 1 < ny; j++) for (let i = 0; i + 1 < nx; i++) {
    const k = j * nx + i;
    z[0] = v[k]; z[1] = v[k + 1]; z[2] = v[k + nx + 1]; z[3] = v[k + nx];
    const code = (z[0] > level) | (z[1] > level) << 1 | (z[2] > level) << 2 | (z[3] > level) << 3;
    for (const [e0, e1] of SEGMENTS[code]) {
      const [xa, ya] = crossing(sc, { i, j, e: e0, level }), [xb, yb] = crossing(sc, { i, j, e: e1, level });
      g.moveTo(xa, ya); g.lineTo(xb, yb);
    }
  }
}

const contours = Object.freeze({
  id: "contours", still: false, mark: MARKS.contours,
  setup(W, H, rnd) {
    const m = Math.min(W, H), nx = Math.ceil(W / CELL) + 1, ny = Math.ceil(H / CELL) + 1;
    const hills = Array.from({ length: 6 }, () => ({
      x: W * (.1 + .8 * rnd()), y: H * (.1 + .8 * rnd()), s: m * (.11 + .11 * rnd()), a: .55 + .45 * rnd(),
      ax: W * (.03 + .05 * rnd()), ay: H * (.03 + .05 * rnd()), w: TAU / (30000 + rnd() * 30000), p: rnd() * 6.3,
    }));
    return { hills, t: 0, nx, ny, cell: CELL, v: new Float64Array(nx * ny), z: new Float64Array(4) };
  },
  draw(g, { scene, dt, C }) {
    scene.t += dt;
    sampleHeights(scene);
    g.lineWidth = 1; g.strokeStyle = C.lane;
    g.beginPath();
    for (const level of LEVELS) isoLines(g, scene, level);
    g.stroke();
    return 0;
  },
});

// The closed set, in the catalog's order.
export const MOTIFS = Object.freeze({ grid, points, traces, lanes, links, contours });

// The motif of an id; an id outside the set draws the grid (the page's fallback).
export const motifOf = (id) => (Object.hasOwn(MOTIFS, id) ? MOTIFS[id] : MOTIFS.grid);
