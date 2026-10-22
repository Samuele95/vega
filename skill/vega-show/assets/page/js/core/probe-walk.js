// The probe's walk (js/core/probe.js; D4 §17.5, D5 §6.7.3, D6 §10), imported only under ?probe:
// the page walked as a reader walks it, by its real controls, and what it showed read back as
// vega/page-probe@1. It imports the hook's own module, the API's URL table (core/api.js, for the
// story's digest), the renderer registry (net/renderers/index.js, to know which stage kinds draw
// an "every weight" level) and the node namer's own rule (core/node-names.js indexNamed: a graph
// node named by its index), and reads the page through the DOM and the registered readings, so it
// stays outside every view's code.
//
// The DOM it reads (the contract a view keeps with its tests):
//   nav.index button[data-view], .view[data-view], #guide-n, #guide-t, #guide-go, #choices > button,
//   #net-switch button[data-mode], .disclose[data-for], #exp-bars .bar .name, #w-map .wmodes
//   button[data-m], #w-map .wmap (.ch, .rh, .c, .c.none), #wtip .v (.acc), #presets button[data-id],
//   #presets h3 + div > button, #try-verdict (.busy), #try-err, #how-box, #how-tabs button,
//   #how-body, .sources button, #many-busy, #many-err, #many-rows tr, #many-mon, #many-monitor tr,
//   #walk li, #code-step .err, #theme, #field,
//   .role-alert, #dot.
import { PROBE_SCHEMA, PROBE_VERSION, said, reading, extensions, fetched } from "./probe.js";
import { URLS } from "./api.js";
import { load as loadRenderers } from "../net/renderers/index.js";
import { indexNamed } from "./node-names.js";

const READY_MS = 60000, VIEW_MS = 5000, DRAW_MS = 10000, ANSWER_MS = 30000;
const VIEW_IDS = Object.freeze(["start", "context", "network", "try", "many", "results", "code"]);
const ALIAS = Object.freeze({ window: "many" });
const GAP_ABS = 1e-3, GAP_REL = 1e-3;
// what a view must never print: a value the code failed to turn into words (D6 P-TXT), looked for
// outside the strings table's own slots (data-t: the editor's hint names the NaN token) and the
// code room's source (the verified files, shown as they are) and a record quoted verbatim
// (data-verbatim: a recorded warning names routes such as /v1/network/tensor/{tensor_id})
// — and a placeholder token ({row.many}, {eval_diff}, {value:pct}) that reached the page unresolved:
// a string rendered as plain text instead of through the story's or the table's resolver is
// invisible to the text engine's own unresolved set
const BAD_TEXT = /\b(?:undefined|NaN|null)\b|\[object Object\]|\{[A-Za-z_]\w*(?:\.[\w@]+)*(?::\w+)?\}/g;
const NOT_DATA = "[data-t], #src, [data-verbatim]";

const q = (sel, root = document) => root.querySelector(sel);
const qa = (sel, root = document) => [...root.querySelectorAll(sel)];
const txt = (n) => (n ? n.textContent : "").replace(/\s+/g, " ").trim();
const key = (k) => document.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true }));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// the next frame, or 50 ms when the browser draws none (a headless --dump-dom run under a virtual
// time budget produces no frames)
const FRAME_MS = 50;
const frame = () => new Promise((r) => { let done = false; const go = () => { if (!done) { done = true; r(); } }; requestAnimationFrame(go); setTimeout(go, FRAME_MS); });

function waitFor(pred, ms = DRAW_MS) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    const tick = () => {
      let v = false;
      try { v = pred(); } catch { v = false; }
      if (v) resolve(true); else if (Date.now() - t0 > ms) resolve(false); else setTimeout(tick, 40);
    };
    tick();
  });
}

function openDisclosure(id) {
  const b = q(`.disclose[data-for="${id}"]`);
  if (b && !b.hidden && b.getAttribute("aria-expanded") !== "true") b.click();
  return !!b && !b.hidden;
}

async function sha256(text) {
  if (!(globalThis.crypto && crypto.subtle)) return null;
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return "sha256:" + [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ---- the walk
export async function walk(h, mode, trace) {
  const out = { schema: PROBE_SCHEMA, version: PROBE_VERSION, probe: { mode }, ready: false, views: {}, sub_views: {} };
  out.ready = !!(await Promise.race([h.ready, sleep(READY_MS).then(() => false)]));
  out.ready_ms = Math.round(performance.now());
  const ctx = { h, mode, out, trace, facts: {}, stepper: { steps: 0, max_gap: null, failed: [], failures: [], fallbacks: [] },
                explain: { asked: 0, ok: 0, degraded: false, errors: [] }, network: null };
  const tour = out.ready ? h.views() : [];
  const off = ((reading("page") || {}).views || {}).off || {};
  for (const id of VIEW_IDS) if (!tour.includes(id)) out.views[id] = { reached: false, rendered: false, visible_text: 0, off: off[id] || null, errors: [] };
  for (const id of tour) out.views[id] = await visit(ctx, id);
  if (out.ready) out.theme_walk = await themes();
  out.theme = document.documentElement.dataset.theme === "day" ? "day" : "night";
  return finish(ctx);
}

async function go(h, id) {
  if (h.current() === id) return true;
  if (id === "start") key("0");
  else if (id === "code") key("c");
  else { const b = q(`nav.index button[data-view="${id}"]`); if (b) b.click(); }
  return waitFor(() => h.current() === id, VIEW_MS);
}

async function visit(ctx, id) {
  const before = ctx.trace.errors.length, reached = await go(ctx.h, id);
  const view = q(`.view[data-view="${id}"]`), collect = COLLECT[id] || COLLECT[ALIAS[id]];
  if (reached && collect) await collect(ctx, view);
  const text = view ? view.innerText.replace(/\s+/g, " ").trim() : "";
  return { reached, rendered: !!view && !view.hidden && text.length > 20, visible_text: text.length, off: null,
           errors: ctx.trace.errors.slice(before).map(said), bad_text: view ? badText(view) : [], cut: view ? cutValues(view) : [] };
}

// The values the view cuts off: text drawn past the viewport's edge (a phone: "0.8834 [0.847, 0"
// with the interval's upper bound lost) — every bar's value, and in the start and results views any
// text at all (a figure, an interval, the verdict's difference), save a name shortened on purpose
// (text-overflow: ellipsis) and what sits in a box that scrolls. The text's own rectangle is read (a
// Range over it), not its box: a value that spills out of its grid cell into the gap beside it is
// still all on show.
function cutValues(view) {
  const edge = document.documentElement.clientWidth + 1, out = new Set(), wide = ["start", "results"].includes(view.dataset.view);
  const kept = (n) => { for (let a = n.parentElement; a && a !== view; a = a.parentElement) { const s = getComputedStyle(a); if (s.textOverflow === "ellipsis" || ["auto", "scroll"].includes(s.overflowX)) return true; } return false; };
  const texts = wide ? (() => { const w = document.createTreeWalker(view, NodeFilter.SHOW_TEXT), all = []; for (let n = w.nextNode(); n; n = w.nextNode()) all.push(n); return all; })() : qa(".bar .val", view);
  for (const v of texts) {
    if (!(v.parentElement || v).getClientRects().length || !txt(v) || (wide && kept(v))) continue;
    const range = document.createRange();
    range.selectNodeContents(v);
    const r = range.getBoundingClientRect();
    if (r.width && (r.right > edge || r.left < -1)) out.add(txt(v));
  }
  return [...out].slice(0, 8);
}

// The bad words of a view's visible text nodes (module comment above).
function badText(view) {
  const found = new Set(), walker = document.createTreeWalker(view, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const el = n.parentElement;
    if (!el || el.closest(NOT_DATA) || !el.getClientRects().length) continue;
    for (const m of n.textContent.match(BAD_TEXT) || []) found.add(m);
  }
  return [...found];
}

const COLLECT = Object.freeze({
  // the context's last step, where the facts count up (their labels are story text)
  async context() {
    const rail = qa("#ctx-rail button");
    if (rail.length > 1) { rail[rail.length - 1].click(); await waitFor(() => { const f = q("#ctx-facts"); return f && !f.hidden; }, VIEW_MS); await frame(); }
  },
  // the verdict as the view says it (its state on #res-verdict), with the limits and the breakdown
  // opened so their story text is read too
  async results(ctx) {
    openDisclosure("res-lim"); openDisclosure("res-class"); await frame();
    const v = q("#res-verdict");
    ctx.out.verdict = { shown: !!v && !v.hidden && !!txt(v), state: v ? v.getAttribute("data-verdict") : null, text: txt(v) };
  },
  async network(ctx) {
    await waitFor(() => txt(q("#guide-t")) || txt(q("#net-cap")), DRAW_MS);
    ctx.network = await drawingModes();
    ctx.detailKinds = await detailKinds();
    ctx.out.latent = { network: reading("latent"), try: null };
    const guide = await guideTexts();
    if (ctx.mode === "all") Object.assign(ctx.facts, { guide, choices: qa("#choices > button").length });
    await decisions();
    const tried = await subView("exp", () => qa("#exp-bars .bar").length > 0);
    ctx.out.sub_views.tried = tried;
    if (ctx.mode === "all") ctx.facts.experiments = { count: qa("#exp-bars .bar").length, order: qa("#exp-bars .bar .name").map(txt) };
    const weights = ctx.out.sub_views.weights = await subView("heat", () => qa("#w-map .wmap .c").length > 0 || q("#w-layers canvas, #w-layers .wmap"));
    const map = weights.rendered ? await weightsMap(ctx.mode === "all") : {};
    weights.columns = Object.fromEntries(Object.entries(map).map(([m, v]) => [m, v ? v.columns : null]));
    if (ctx.mode === "all") ctx.facts.weights_map = map;
  },
  async try(ctx) {
    await waitFor(() => qa("#presets button[data-id]").length > 0, DRAW_MS);
    const chosen = ctx.mode === "all" ? qa("#presets button[data-id]") : qa("#presets h3 + div > button[data-id]:first-child");
    const out = [];
    for (const b of chosen) out.push(await stepper(ctx, b.dataset.id));
    if (ctx.mode === "all") ctx.facts.stepper = out;
    ctx.out.latent = { network: null, ...(ctx.out.latent || {}), try: reading("latent") };
    ctx.out.sub_views.stepper = { presets: out.length, answered: out.filter((p) => p.answered).length };
  },
  async many(ctx, view) {
    const src = view && q(".sources button", view);
    if (!src) { ctx.out.sub_views.many = { source: null, rows: 0, error: null }; return; }
    src.click();
    await waitFor(() => { const b = q("#many-busy"); return b && b.hidden; }, ANSWER_MS);
    const mon = q("#many-mon");
    ctx.out.sub_views.many = { source: src.id || null, rows: qa("#many-rows tr", view).length, error: txt(q("#many-err")) || null,
                               monitor: mon && !mon.hidden ? qa("#many-monitor tr").length : null };
  },
  async code(ctx) {
    await waitFor(() => qa("#walk li").length > 0 || q("#code-step .err"), DRAW_MS);
    const items = qa("#walk li");
    let broken = 0;
    // a step is opened by its button (the li only holds it: a click on the li reaches no handler)
    for (const li of items) { (q("button", li) || li).click(); await frame(); if (q("#code-step .err")) broken++; }
    ctx.out.sub_views.code = { walk: items.length, files: qa("#code-files button").length, broken };
  },
});

// ---- the network view
// The drawing in each of its modes (#net-switch), read back by the latent area's reading: every
// segment it laid out, with the pixels it inked (or the whole chain's, drawn whole).
async function drawingModes() {
  const box = q("#net-switch"), offered = !!box && !box.hidden;
  const modes = {}, buttons = offered ? qa("#net-switch button[data-mode]").filter((b) => !b.hidden) : [];
  for (const b of buttons.length ? buttons : [null]) {
    if (b) b.click();
    await waitFor(() => { const r = reading("latent"); return r && r.canvases && r.canvases.network; }, DRAW_MS);
    await sleep(50); await frame();
    const r = reading("latent"), net = r && r.canvases ? r.canvases.network : null;
    modes[b ? b.dataset.mode : "only"] = net ? { drawn: net.drawn, ink: net.ink, segments: net.segments.map((s) => ({ kind: s.kind, stage: s.stage, pixels: s.pixels })) } : null;
  }
  if (buttons.length) buttons[0].click();
  return modes;
}

// The stage kinds of the graph whose renderer declares an "every weight" level (D4 §7.3 `detail`):
// the page must offer the switch to it whenever there is one (P-NET). Null when unread.
async function detailKinds() {
  const g = reading("graph");
  if (!g || !Array.isArray(g.kinds)) return null;
  try {
    const renderers = await loadRenderers(g.kinds);
    return g.kinds.filter((k) => { const r = renderers.get(k); return !!r && r.detail === true; });
  } catch { return null; }
}

async function guideTexts() {
  const go = q("#guide-go");
  if (!go || q("#guide").hidden) return [];
  if (txt(q("#guide-n")) === "✓") go.click();          // read again from the first step
  const out = [];
  for (let k = 0; k < 20; k++) {
    const n = txt(q("#guide-n"));
    out.push({ n, t: txt(q("#guide-t")) });
    if (n === "✓") break;
    q("#guide-go").click();
    await sleep(0);
  }
  return out;
}

async function decisions() {
  for (const b of qa("#choices > button")) { b.click(); await frame(); }
}

async function subView(disclosure, rendered) {
  const found = openDisclosure(disclosure);
  const ok = found && await waitFor(rendered, DRAW_MS);
  return { found, rendered: !!ok };
}

// The number the tooltip of a map cell states (3 significant digits), signed by the cell's colour.
function cellValue(cell) {
  cell.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, clientX: 10, clientY: 10 }));
  const v = q("#wtip .v"), nums = txt(v).match(/\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi);
  if (!v || !nums) return null;
  return (v.classList.contains("acc") ? "+" : "-") + nums[nums.length - 1];
}

// The map's column heads, and with `values` its rows and the number of each cell (the parity facts).
function readMap(values) {
  const grid = q("#w-map .wmap");
  if (!grid) return null;
  const columns = qa(".ch", grid).map(txt), rows = [];
  if (!values) return { columns, rows: null };
  for (const n of grid.children) {
    if (n.classList.contains("rh")) rows.push({ label: txt(n).replace(/[\s›⌄]+$/, ""), sub: n.classList.contains("sub"), cells: [] });
    else if (n.classList.contains("c") && rows.length) rows[rows.length - 1].cells.push(n.classList.contains("none") ? null : cellValue(n));
  }
  const last = qa(".c", grid).pop();
  if (last) last.dispatchEvent(new MouseEvent("mouseleave", { bubbles: true }));
  return { columns, rows };
}

// Each mode of the weights map (#w-map .wmodes), read in turn: {mode: {columns, rows}}.
async function weightsMap(values) {
  const out = {};
  for (const b of qa("#w-map .wmodes button[data-m]")) { b.click(); await waitFor(() => q("#w-map .wmap"), VIEW_MS); out[b.dataset.m] = readMap(values); }
  return out;
}

// ---- try one: a preset answered, its stepper walked step by step, its values checked
async function stepper(ctx, id) {
  const b = q(`#presets button[data-id="${CSS.escape(id)}"]`);
  b.click();
  ctx.explain.asked++;
  const answered = await waitFor(() => !q("#try-verdict .busy") && (txt(q("#try-verdict")) || txt(q("#try-err"))), ANSWER_MS);
  const rec = { id, answered: answered && !txt(q("#try-err")), verdict: txt(q("#try-verdict")), steps: [] };
  if (!rec.answered) { ctx.explain.errors.push(`${id}: ${txt(q("#try-err")) || "timeout"}`); return rec; }
  ctx.explain.ok++;
  const ex = (ctx.h.state() || {}).explanation;
  if (ex && ex.degraded) ctx.explain.degraded = true;
  const comp = ex && ex.attribution && ex.attribution.completeness;
  if (comp && Number.isFinite(comp.gap) && !q(".view[data-view=\"try\"] .views-completeness, .view[data-view=\"try\"] .graph-gap")) ctx.explain.errors.push(`${id}|completeness_gap_unshown`);
  if (ex && ex.attribution && typeof ex.attribution.method === "string" && q(".view[data-view=\"try\"] .views-why-lead") && !q(".view[data-view=\"try\"] .views-method")) ctx.explain.errors.push(`${id}|method_unnamed|${ex.attribution.method}`);
  const box = q("#how-box");
  if (!box || box.hidden) return rec;
  openDisclosure("how");
  await waitFor(() => qa("#how-tabs button").length > 0, DRAW_MS);
  const bodies = [txt(q(".view[data-view=\"try\"]"))];
  for (let i = 0; i < qa("#how-tabs button").length && i < 60; i++) {
    qa("#how-tabs button")[i].click(); rec.steps.push(stepFacts(txt(qa("#how-tabs button")[i]))); bodies.push(txt(q("#how-body")));
  }
  indexNamed(ex, bodies).forEach((m) => ctx.explain.errors.push(`${id}|node_index_named|${m}`));
  check(ctx, id, reading("stepper"));
  return rec;
}

function stepFacts(tab) {
  const body = q("#how-body"), total = q(".bar.total .val", body), a = q(".act .formula .num", body);
  return { tab, z: total ? txt(total).replace(/^=\s*/, "") : null, a: a ? txt(a) : null,
           values: qa(".eq .bar .val", body).map(txt), soft: qa(".soft span", body).map(txt) };
}

// Every recompute of one answer's stepper: within max(1e-3, 1e-3·|z|) of the network's number, or
// a failure. A check the page could not make (an op that did not fit, an explainer that threw, a
// step that failed) is a failure too, never dropped. `failed` names each as
// "<preset>/<step>/<kind>|<reason>" (gap, fit, error, failed, unread); `failures` holds the numbers.
function check(ctx, id, r) {
  if (!r || !Array.isArray(r.values)) { fail(ctx, `${id}|unread`, { preset: id }); return; }
  ctx.stepper.steps = Math.max(ctx.stepper.steps, r.steps || 0);
  for (const g of r.gaps || []) ctx.stepper.fallbacks.push(`${id}/${g.kind}|${g.why || g.module || ""}`);
  for (const f of r.failed || []) fail(ctx, `${id}/${f.kind}|failed`, { preset: id, ...f });
  for (const v of r.values) {
    const c = v.check, where = `${id}/${v.index}/${v.kind}`;
    if (c === null || c === undefined) continue;
    if (c.error) { fail(ctx, `${where}|error`, { preset: id, step: v.index, name: v.name, error: c.error }); continue; }
    for (const part of [c, ...(c.ops || [])]) {
      if (part.what === "decision") decision(ctx, where, part); else gap(ctx, where, part);
    }
  }
}

// The decision a step states (the labels at or above their thresholds, a binary class's side)
// against the network's answer: the same set, or a failure.
function decision(ctx, where, part) {
  const a = [...(part.page || [])].map(String).sort(), b = [...(part.network || [])].map(String).sort();
  if (a.length !== b.length || a.some((x, i) => x !== b[i])) fail(ctx, `${where}|decision`, { what: "decision", page: a, network: b });
}

function fail(ctx, name, detail) {
  ctx.stepper.failed.push(name);
  ctx.stepper.failures.push(detail);
}

function gap(ctx, where, part) {
  if (part.fit === false) { fail(ctx, `${where}|fit`, { what: part.what, sizes: part.sizes || null }); return; }
  if (!Number.isFinite(part.page) || !Number.isFinite(part.network)) return;
  const g = Math.abs(part.network - part.page), bound = Math.max(GAP_ABS, GAP_REL * Math.abs(part.network));
  ctx.stepper.max_gap = Math.max(ctx.stepper.max_gap || 0, g);
  if (g > bound) fail(ctx, `${where}|gap`, { what: part.what, page: part.page, network: part.network, gap: g, bound });
}

// ---- both themes, by the real control; back to the first
async function themes() {
  const root = document.documentElement, first = root.dataset.theme, b = q("#theme");
  if (!b) return { first, flipped: null, back: null };
  b.click(); await frame();
  const flipped = root.dataset.theme;
  b.click(); await frame();
  return { first, flipped, back: root.dataset.theme };
}

// ---- the output
async function finish(ctx) {
  const { h, out, trace } = ctx, page = reading("page") || {}, story = h.story() || null;
  const fieldReading = reading("field") || {};
  out.current = h.current();
  out.locale = new URLSearchParams(location.search).get("locale") === "qps-ploc" ? "qps-ploc" : "en";
  out.page_errors = trace.errors.map(said);
  out.placeholders_unresolved = reading("unresolvedPlaceholders") || [];
  out.missing_keys = reading("missingKeys") || [];
  out.lane = laneOf(page);
  out.motif = { id: page.motif ? page.motif.id : null, data_motif: (q("#field") || { getAttribute: () => null }).getAttribute("data-motif"),
                alert_marks: !!fieldReading.alert_marks };
  const roles = story && story.class_roles ? Object.values(story.class_roles.roles || {}) : [];
  out.palette = { alert_used: !!q(".role-alert") || !!fieldReading.alert_marks, alert_roles: roles.filter((r) => r === "alert").length,
                  role_alert_elements: qa(".role-alert").length };
  out.network = networkOf(ctx);
  const graph = reading("graph");
  out.explain = { ok: ctx.explain.asked > 0 && ctx.explain.ok === ctx.explain.asked, family: graph ? graph.family : null,
                  degraded: ctx.explain.degraded, asked: ctx.explain.asked, errors: ctx.explain.errors };
  out.stepper = ctx.stepper;
  out.verdict = { ...(out.verdict || { shown: false, state: null, text: "" }), expected: expectedVerdict(story) };
  out.links = await links(reading("codeWalk"));
  out.story_digest = story ? await storyDigest() : null;
  out.fetched = fetched();
  out.extensions = extensions();
  out.parity = ctx.mode === "all" ? ctx.facts : null;
  for (const id of ["tried", "weights"]) if (out.sub_views[id]) out.views[id] = { reached: out.sub_views[id].found, rendered: out.sub_views[id].rendered, off: null, errors: [] };
  return out;
}

// The verdict state the story records (story@2 results.verdict, as the evaluation recorded it), or
// null when it carries none: the state #res-verdict must say.
function expectedVerdict(story) {
  const sec = ((story && story.sections) || []).find((s) => s && s.id === "results"), v = sec && sec.verdict;
  if (!v) return null;
  if (v.earned === null && v.reason) return "not_assessable";
  if (v.earned === true) return "earned";
  if (v.earned === false) return v.beaten_by_baseline === true ? "beaten" : "not_earned";
  return null;
}

function laneOf(page) {
  const lane = reading("lane");
  if (lane) return { ok: true, view_key: lane.view_key, shape: lane.shape, level: lane.level, link: lane.link, mismatch: null };
  const bad = /\bbad\b/.test((q("#dot") || {}).className || "");
  return { ok: false, view_key: null, shape: null, level: null, link: page.lane ? page.lane.link || null : null,
           mismatch: bad ? txt(q("#headline")) || txt(q("#status-text")) : null };
}

function networkOf(ctx) {
  const g = reading("graph"), modes = ctx.network || {}, all = Object.values(modes).filter(Boolean);
  const segments = all.flatMap((m) => m.segments), inked = segments.filter((s) => s.pixels > 0).length;
  const nodes = ((ctx.h.network() || {}).nodes || []).filter((n) => n.parent === null);
  return { available: !!g, why: g ? null : txt(q("#net-cap")) || null, stages: nodes.length,
           segments: segments.length, segments_drawn: inked, chain_ink: all.filter((m) => m.drawn === "chain").map((m) => m.ink),
           modes, detail_kinds: ctx.detailKinds === undefined ? null : ctx.detailKinds,
           warnings: reading("netWarnings") || [], drawn: g ? g.drawn : null };
}

// The code walk's report links, each fetched: its status and the digest of what it served.
async function links(walk) {
  const out = [];
  for (const l of (walk && walk.links) || []) {
    try {
      const r = await fetch(l.href), body = r.ok ? await r.text() : null;
      out.push({ step: l.step, href: l.href, status: r.status, digest: body === null ? null : await sha256(body) });
    } catch (e) { out.push({ step: l.step, href: l.href, status: 0, digest: null }); }
  }
  return out;
}

async function storyDigest() {
  try { const r = await fetch(URLS.story); return r.ok ? await sha256(await r.text()) : null; } catch { return null; }
}
