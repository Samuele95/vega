// The page's test hook (R-F1; D4 §17.5, D5 §6.7.3, D6 §10): window.__showcase, the one read-only
// API a test reads the running page through, and the ?probe=<views|all> driver.
//
//   window.__showcase = { version, ready, lane(), story(), page(), graph(), registry(), views(),
//                         current(), state() (its stepper: the shown answer's step values), math,
//                         missingKeys(), fetched(), probe(opts) }
//
// The members are the union of the three designs (the vega-show catalog's probe.hook_members). A
// module adds its own reading without editing this file — extendProbe(name, fn, {report}) — and the
// union's members read those readings by name: lane, page and registry are the boot's, graph the
// network view's, math the stepper's, missingKeys the strings table's. The hook also answers every
// other reading a module registered (field, latent, stepper, codeWalk, netWarnings, subViews, …),
// and with {report: true} the probe's output carries that reading under `extensions`.
//
// With ?probe in the URL (inert otherwise) the page walks itself once it is ready, as a reader
// would — clicking the real controls — and writes vega/page-probe@1 into <pre id="showcase-probe">
// (js/core/probe-walk.js, imported only then). `views` walks every view of the tour, the network
// view's guide, decisions, drawing modes and disclosures, the first preset of each group with every
// step of its stepper, one many-at-once source, the code walk end to end and both themes; `all`
// walks every preset too and adds the facts the parity golden pins (`parity`). The page keeps running
// after the output is written (a headless browser dumps it only when its time budget ends), so the
// probe keeps listening: every error heard after the walk is appended to `late_errors` and the output
// written again. A clean page has both lists empty.
//
// Under ?probe the page's fetch() is also watched: each answer's status and, for a JSON answer, the
// schema it names (`fetched`); outside it, fetched() reads the browser's resource timing.

export const PROBE_VERSION = "2.3.0";
export const PROBE_SCHEMA = "vega/page-probe@1";
export const HOOK_MEMBERS = Object.freeze(["version", "ready", "lane", "story", "page", "graph", "registry", "views", "current",
  "state", "math", "missingKeys", "fetched", "probe"]);
const MODE = probeMode();
const LATE_POLL_MS = 100;
const NOT_STARTED_MS = 5000;
const READY_MS = 60000;

function probeMode() {
  if (typeof location === "undefined") return null;
  const m = new URLSearchParams(location.search).get("probe");
  return m === null ? null : m === "all" ? "all" : "views";
}

// ---- errors: every uncaught error, rejected promise, failed resource, CSP refusal and console.error
// from the moment this module loads (or earlier, when the page installed window.__probeTrace itself),
// as {kind: error | rejection | resource | csp | console, text}; and, under ?probe, every fetch().
function trace() {
  if (!window.__probeTrace) {
    const t = { errors: [], fetched: [], started: false };
    window.__probeTrace = t;
    addEventListener("error", (e) => t.errors.push(errorOf(e)), true);
    addEventListener("unhandledrejection", (e) => t.errors.push({ kind: "rejection", text: String(e.reason && (e.reason.stack || e.reason.message) || e.reason) }));
    addEventListener("securitypolicyviolation", (e) => t.errors.push({ kind: "csp", text: `${e.violatedDirective} ${e.blockedURI}` }));
    const ce = console.error.bind(console);
    console.error = (...a) => { t.errors.push({ kind: "console", text: a.map(String).join(" ") }); ce(...a); };
  }
  if (!window.__probeTrace.fetched) window.__probeTrace.fetched = [];
  return window.__probeTrace;
}

function errorOf(e) {
  const n = e.target;
  if (n && n !== window && (n.src || n.href)) return { kind: "resource", text: n.src || n.href };
  return { kind: "error", text: (e.message || String(e.error)) + (e.filename ? ` (${e.filename}:${e.lineno})` : "") };
}

// The page-relative spelling of a URL the page reached ("v1/network", "showcase/story.json"), or
// the URL itself when it is not under the page's own location.
export function relative(url) {
  const base = new URL(".", location.href).href, abs = new URL(url, location.href).href;
  return abs.startsWith(base) ? abs.slice(base.length).split("#")[0] : abs;
}

// Each fetch() the page makes, with its status and the schema its JSON answer names.
function watchFetch(t) {
  const own = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = relative(typeof input === "string" ? input : input.url), rec = { url, status: 0, schema: null };
    t.fetched.push(rec);
    const r = await own(input, init);
    rec.status = r.status;
    if ((r.headers.get("content-type") || "").includes("json")) {
      r.clone().json().then((b) => { rec.schema = b && typeof b.schema === "string" ? b.schema : null; }, () => {});
    }
    return r;
  };
}

if (MODE) {
  watchFetch(trace());
  // a page whose modules failed to load never starts the probe: the output says so
  addEventListener("load", () => setTimeout(() => {
    if (!trace().started) write({ schema: PROBE_SCHEMA, version: PROBE_VERSION, probe: { mode: MODE }, ready: false, error: "not_started", page_errors: trace().errors.map(said) });
  }, NOT_STARTED_MS));
}

// "kind: text" — the probe output's form of one error
export const said = (e) => (e && typeof e === "object" ? `${e.kind}: ${e.text}` : String(e));

// ---- the readings modules add
// extendProbe("missingKeys", () => [...]) from the strings, extendProbe("lane", …) from the boot.
// The hooks live on window, so the order of imports does not matter.
export function extendProbe(name, fn, { report = false } = {}) {
  if (!window.__probeHooks) window.__probeHooks = {};
  if (!window.__probeReported) window.__probeReported = [];
  window.__probeHooks[name] = fn;
  if (report && !window.__probeReported.includes(name)) window.__probeReported.push(name);
}

// A registered reading by name: its value (a function's result), or null when no module gave one.
export function reading(name) {
  const fn = (window.__probeHooks || {})[name];
  if (fn === undefined) return null;
  try { return typeof fn === "function" ? fn() : fn; } catch (e) { return { error: String(e && e.message ? e.message : e) }; }
}

// Every reported reading, as JSON (a reading that throws says so in its place).
export function extensions() {
  const out = {};
  for (const name of window.__probeReported || []) {
    try { out[name] = JSON.parse(JSON.stringify(reading(name))); } catch (e) { out[name] = { error: String(e) }; }
  }
  return out;
}

// What the page fetched: under ?probe each fetch() with its status and schema; without it the
// browser's resource timing (scripts and sheets included), the status where the browser keeps it.
export function fetched() {
  const t = window.__probeTrace;
  if (t && t.fetched && t.fetched.length) return t.fetched.map((r) => ({ ...r }));
  return performance.getEntriesByType("resource").map((e) => ({ url: relative(e.name), status: e.responseStatus || null, schema: null }));
}

// ---- the hook
// What a page offers through its DOM alone, the defaults of the hooks a boot does not pass (the
// parity recorder hooks this module into a page with no boot of its own): readiness when the status
// dot has turned and the start view holds its headline; the views of the index, the start and the
// code room; the view not hidden.
const q = (sel) => document.querySelector(sel);
const textOf = (sel) => (q(sel) ? q(sel).textContent : "").trim();
function domReady() {
  return new Promise((resolve) => {
    const t0 = Date.now();
    const tick = () => {
      const ok = /\b(ok|bad)\b/.test((q("#dot") || {}).className || "") && !!(textOf("#headline") || textOf("#start-facts"));
      if (ok) resolve(true); else if (Date.now() - t0 > READY_MS) resolve(false); else setTimeout(tick, 40);
    };
    tick();
  });
}

export function domHooks() {
  return {
    views: () => ["start", ...[...document.querySelectorAll("nav.index button[data-view]")].map((b) => b.dataset.view), ...(q('.view[data-view="code"]') ? ["code"] : [])],
    current: () => { const v = q(".view:not([hidden])"); return v ? v.dataset.view : null; },
    story: () => null, network: () => null, state: () => ({}),
  };
}

// Installs window.__showcase from the page's hooks (the boot passes ready, views, current, story,
// network and state; the DOM's stand in for any it does not) and, under ?probe, runs the walk once
// the page is ready and writes the result.
export function startProbe(hooks = {}) {
  const h = { ...domHooks(), ...hooks };
  h.ready = hooks.ready || domReady();
  trace().started = !!MODE;
  const base = Object.freeze({
    version: PROBE_VERSION, ready: h.ready, views: h.views, current: h.current, story: h.story, state: h.state, network: h.network,
    lane: () => reading("lane"), page: () => reading("page"), graph: () => reading("graph"), registry: () => reading("registry"),
    get math() { return reading("math"); }, missingKeys: () => reading("missingKeys") || [], fetched,
    errors: () => trace().errors.map(said), probe: (opts = {}) => run(h, opts.mode || "views"),
  });
  window.__showcase = new Proxy(base, {
    get: (t, k) => (k in t ? t[k] : (window.__probeHooks || {})[k]),
    has: (t, k) => k in t || k in (window.__probeHooks || {}),
  });
  if (MODE) run(h, MODE).then(write, (e) => write({ schema: PROBE_SCHEMA, version: PROBE_VERSION, error: String(e && e.stack || e), page_errors: trace().errors.map(said) }));
  return window.__showcase;
}

async function run(h, mode) {
  const { walk } = await import("./probe-walk.js");
  return walk(h, mode, trace());
}

// Writes the output, then rewrites it whenever an error is heard after the ones it reports (the
// watch's timer is kept on the trace object: no module-level state, PS5).
function write(out) {
  let pre = document.getElementById("showcase-probe");
  if (!pre) { pre = document.createElement("pre"); pre.id = "showcase-probe"; pre.hidden = true; document.body.append(pre); }
  const t = trace(), seen = (out.page_errors || []).length;
  const render = () => { out.late_errors = t.errors.slice(seen).map(said); pre.textContent = JSON.stringify(out); };
  render();
  clearInterval(t.lateWatch);
  t.lateWatch = setInterval(() => { if (t.errors.length > seen + out.late_errors.length) render(); }, LATE_POLL_MS);
}
