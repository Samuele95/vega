// "How the network computed it" (D4 §8.1): the steps of one answer, derived from the stage graph,
// one tab each, walked with back / next, each explained by the explainer of its kind with the
// sample's own numbers — the page recomputing every value it can from the parameters it was sent,
// the network's value beside it, and any difference shown as its own row, never hidden.
//
// Steps (deriveSteps):
//   input                  the shaped inputs and the recipe that made them
//   a dense run            one step per dense stage, its fused normalisation, bend and dropout inside
//                          it; an embedding feeding its input side a step of its own
//   a box                  one step when its kind computes (net/kinds.js STEP_KINDS; a transformer
//                          one step per encoder layer); a kind that does nothing at answer time
//                          (QUIET_KINDS: dropout, edge_dropout, flatten / reshape, concat,
//                          positional) becomes a one-line "then …" note on the next step
//   a residual group       each repeat's inside, then its sum ("residual")
//   a parallel group       each branch's inside, its steps named by the branch, then the merge
//   link                   the head's weighted sum for the explained output, and the link that
//                          reads it
// Past MAX_STEPS, consecutive steps of the same kind collapse into one step with a picker.
//
// class Stepper — the contract the try view consumes (BUILD-PLAN W08):
//   new Stepper(ctx)   ctx: {graph (StageGraph), ex (the explanation, either form — how/answer.js;
//                      the page asks v1/explain for detail "exact", R-F5), row (the raw row asked
//                      about, or null), explainers (the Map explainers/index.js load() resolves to —
//                      the boot loads it with Stepper.kinds(graph, link) so that mount renders at once
//                      — or null: mount loads what the steps need), lane (core/lane.js Lane, or null:
//                      the graph's link), page (page.json: recipe, explain), roles (RoleMap),
//                      onCode(walkId) ("in the code"; null hides the link), inputStep(ex, row, env)
//                      (the shape module's first-step body, or null: the input explainer), nodeName(id)
//                      (a graph's node by the name the shape's drawings give it, or null: the
//                      explainer's own, core/node-names.js), at (the step to show first)}; an `env`
//                      member is read as the same fields.
//   mount(host)        binds the step tabs, body and navigation inside `host` (#how-tabs, #how-body,
//                      #how-n, #how-back, #how-next, #how-code, looked up in `host`, then in the
//                      document, else created in `host`), renders the step shown — at once when the
//                      explainers are there — then shows #how-box; resolves to the stepper
//   show(i)            renders step i (the tabs, the body, the navigation), synchronously
//   steps()            the steps, in order
//   values()           per step: its kind, name, stage, unit and recompute {page, network, gap,
//                      within} (the probe's stepper reading, R-F1 state().stepper.values)
//   Stepper.kinds(graph, link)   the step kinds a network's steps need explainers for
import { $, el } from "../core/dom.js";
import { t, tText, hasKey, data } from "../core/text.js";
import { extendProbe } from "../core/probe.js";
import { STEP_KINDS, QUIET_KINDS } from "../net/kinds.js";
import { Answer } from "./answer.js";
import { MATH } from "./act.js";
import { load as loadExplainers, gapsOf, ALWAYS } from "./explainers/index.js";

export const MAX_STEPS = 12;
export const PARTS = Object.freeze({ tabs: "how-tabs", body: "how-body", count: "how-n", back: "how-back", next: "how-next", code: "how-code" });
const BOX = "how-box";

// window.__showcase.math (R-F1): the page's own arithmetic, for a probe to compare with the network's.
if (typeof window !== "undefined") extendProbe("math", MATH);

// ---- step derivation (pure). Each step names the step whose output it reads (`from`: the input
// step, the step before it, the step before its group for a branch's or a repeat's first step).
function box(graph, s, into) {
  const n = graph.byId(s.stage), kind = n.kind;
  if (QUIET_KINDS.includes(kind)) into.note({ kind, stage: n.id });
  else if (kind === "transformer" && n.children.length) n.children.forEach((c, layer) => into.push({ kind, stage: n.id, layer, node: c }));
  else into.push({ kind: STEP_KINDS.includes(kind) ? kind : "custom", stage: n.id });
}

function run(graph, s, into) {
  for (const id of s.input) {
    const kind = graph.byId(id).kind;
    if (kind === "embedding") into.push({ kind, stage: id }); else into.note({ kind, stage: id });
  }
  for (const layer of s.layers) into.push({ kind: "dense", stage: layer.dense, layer });
}

function group(graph, s, into, branch) {
  const start = into.last;
  if (s.mode === "residual") {
    s.repeats.forEach((r, repeat) => {
      const input = into.last;
      walk(graph, r.segments, into, branch);
      into.push({ kind: "residual", group: s.group, repeat, add: r.add, projection: r.projection, inner: innerIds(r.segments), input, body: into.last }, input);
    });
  } else if (s.mode === "parallel") {
    const ends = s.branches.map((b, k) => { into.last = start; walk(graph, b.segments, into, k + 1); return { ids: innerIds(b.segments), step: into.last }; });
    into.branch = branch;
    into.push({ kind: "merge", group: s.group, node: s.node, merge: s.merge, branches: ends, input: start }, start);
  } else walk(graph, s.children || [], into, branch);
}

// The stage ids a list of segments runs through, in order (a group by its own id).
function innerIds(segs) {
  return segs.flatMap((s) => (s.kind === "run" ? [...s.input, ...s.layers.flatMap((l) => [l.dense, ...l.fused])]
    : s.kind === "group" ? [s.group] : s.stage ? [s.stage] : []));
}

function walk(graph, segs, into, branch = null) {
  for (const s of segs) {
    into.branch = branch;
    if (s.kind === "run") run(graph, s, into);
    else if (s.kind === "box") box(graph, s, into);
    else if (s.kind === "group") group(graph, s, into, branch);
    else if (s.kind === "head") into.head = s.stage;
  }
}

// The steps of a graph, in order (see the header). link: the lane's link (the graph's by default).
export function deriveSteps(graph, link = graph.link) {
  const first = { kind: "input", notes: [], branch: null, from: null }, out = [first];
  const into = {
    branch: null, head: null, pending: [], last: first,
    push(step, from = this.last) { const s = { ...step, notes: this.pending, branch: this.branch, from }; out.push(s); this.pending = []; this.last = s; },
    note(n) { this.pending.push(n); },
  };
  walk(graph, graph.segments().filter((s) => s.kind !== "input"), into);
  into.branch = null;
  into.push({ kind: "link", link, stage: into.head || (graph.head() ? graph.head().id : null) });
  return collapse(out);
}

// Past MAX_STEPS, each run of two or more consecutive steps of the same kind (and branch) becomes
// one step {kind, members} with a picker; the input and link steps stay as they are.
export function collapse(steps, max = MAX_STEPS) {
  if (steps.length <= max) return steps;
  const out = [];
  for (const s of steps) {
    const last = out[out.length - 1], same = last && last.kind === s.kind && last.branch === s.branch && !ALWAYS.includes(s.kind);
    if (same && last.members) last.members.push(s);
    else if (same) out[out.length - 1] = { kind: s.kind, branch: s.branch, notes: [], members: [last, s] };
    else out.push(s);
  }
  return out;
}

// Every step, a collapsed one's members in its place.
export const flatSteps = (steps) => steps.flatMap((s) => s.members || [s]);

// The step kinds a list of steps needs explainers for.
export const stepKindsOf = (steps) => [...new Set(flatSteps(steps).map((s) => s.kind))];

// ---- names
// A step's word: the explain catalog's for its stage (page.json#explain: blocks.<type>.step; a
// residual or merge step its group's), else the table's word for the step kind.
function stepWord(graph, step) {
  const id = step.kind === "residual" || step.kind === "merge" ? step.group : step.kind === "input" ? null : step.stage;
  const n = id ? graph.byId(id) : null, key = n ? `explain.${n.type}.step` : null;
  return key && hasKey(key) ? tText(key) : tText(`how.step.${step.kind}`);
}

// Map<step, name> of every step ("input", "layer 1", "normalisation", "branch 2 · layer 1",
// "output"): a dense step is always numbered, another kind when more than one step shares its word;
// a collapsed step is named by the range of its members.
export function stepNames(graph, steps) {
  const flat = flatSteps(steps), words = flat.map((s) => stepWord(graph, s)), count = {}, seen = {};
  words.forEach((w) => { count[w] = (count[w] || 0) + 1; });
  const names = new Map();
  flat.forEach((s, i) => {
    const w = words[i], n = (seen[w] = (seen[w] || 0) + 1);
    const base = (s.kind === "dense" || count[w] > 1) && !ALWAYS.includes(s.kind) ? tText("how.step.nth", { step: w, n }) : w;
    names.set(s, s.branch ? tText("how.step.branch", { b: s.branch, step: base }) : base);
  });
  for (const s of steps) if (s.members) names.set(s, tText("how.step.range", { first: names.get(s.members[0]), last: names.get(s.members[s.members.length - 1]) }));
  return names;
}

// The walk id of a step's code (R-E8): the input step the recipe's ("shape"), the link step the
// link's, a group's step its group type, a block step its block type.
export function codeId(graph, step) {
  if (step.kind === "input") return "shape";
  if (step.kind === "link") return "link";
  const id = step.group || step.stage || (step.members && (step.members[0].group || step.members[0].stage)), n = id ? graph.byId(id) : null;
  return n ? n.type : "forward";
}

// ---- the DOM the stepper binds (the shell's, or its own when the page has none)
function bindParts(host) {
  const find = (id) => host.querySelector(`#${id}`) || $(id);
  if (!find(PARTS.tabs)) {
    host.append(el("div", { class: "wmodes caps", id: PARTS.tabs, role: "tablist", "aria-label": tText("try.how.tabs") }), el("div", { id: PARTS.body }),
      el("div", { class: "how-nav" }, el("span", { class: "caps muted", id: PARTS.count }),
        el("button", { class: "caps muted", id: PARTS.back }, "← ", t("app.back")), el("button", { class: "caps acc", id: PARTS.next }),
        el("button", { class: "caps code-link how-code", id: PARTS.code }, t("try.how.code"), " →")));
  }
  return Object.fromEntries(Object.entries(PARTS).map(([k, id]) => [k, find(id)]));
}

// The words of what an answer explained (how.target.<kind>), for the picks' lines: named by the
// target's own name, else by the output it indexes, else "the explained output".
function targetWords(answer) {
  const tg = answer.target() || {}, kind = tg.kind || "logit", out = answer.output();
  const name = tg.name ?? (Number.isInteger(tg.index) && out.names ? out.names[tg.index] : null);
  const named = ["logit", "binary_logit", "label_logit", "quantile", "threshold_logit", "feature_error", "forecast_step"].includes(kind);
  return named && (name === null || name === undefined || name === "") ? tText("how.target.unnamed") : tText(`how.target.${kind}`, { name: data(name ?? "") });
}

export class Stepper {
  constructor(ctx = {}) {
    const c = { ...(ctx.env || {}), ...ctx }, graph = c.graph || null, ex = c.ex ?? c.explanation ?? null;
    // the view's env (D4 §6.2: T, lex, roles, lane, graph, story, page, schema, fmt) goes on to the
    // explainers and to the shape module's input step as it came
    this.shared = ctx.env || {};
    this.ctx = { graph, row: c.row ?? null, lane: c.lane || null, page: c.page || null, roles: c.roles || null,
                 onCode: c.onCode || c.openCode || null, inputStep: c.inputStep || null, nodeName: c.nodeName || null };
    this.answer = graph && ex ? Answer.of(ex, graph, this.ctx.row) : null;
    this.list = graph ? deriveSteps(graph, this.link) : [];
    this.names = graph ? stepNames(graph, this.list) : new Map();
    Object.assign(this, { explainers: c.explainers || null, at: Number(c.at) || 0, picks: new Map(), members: new Map(), parts: null, ready: false, failed: new Map() });
  }

  // The step kinds a network's steps need explainers for (the boot's explainers.load(…) argument).
  static kinds(graph, link = graph.link) { return stepKindsOf(deriveSteps(graph, link)); }

  get link() { return (this.ctx.lane && this.ctx.lane.link) || (this.ctx.graph ? this.ctx.graph.link : null); }
  steps() { return this.list.slice(); }
  available() { return !!this.answer && this.list.length > 1; }

  mount(host) {
    this.parts = bindParts(host);
    const box = (host.closest && host.closest(`#${BOX}`)) || $(BOX) || host;
    box.hidden = true;
    extendProbe("stepper", () => this.reading(), { report: true });
    if (!this.available()) return Promise.resolve(this);
    const kinds = stepKindsOf(this.list), have = this.explainers, done = () => { this.ready = true; this.show(this.at); box.hidden = false; return this; };
    if (have && kinds.every((k) => have.has(k))) return Promise.resolve(done());
    return loadExplainers(kinds, this.link).then((m) => {
      this.explainers = Object.assign(new Map([...m, ...(have || [])]), { gaps: gapsOf(m), link: this.link });
      return done();
    });
  }

  show(i) {
    if (!this.list.length) return;
    this.at = Math.max(0, Math.min(Number(i) || 0, this.list.length - 1));
    if (!this.ready || !this.parts) return;
    this.renderTabs(); this.renderBody(); this.renderNav();
  }

  renderTabs() {
    this.parts.tabs.replaceChildren(...this.list.map((s, i) => Object.assign(el("button", { class: i === this.at ? "on" : "", role: "tab", "aria-selected": String(i === this.at) },
      t("how.tab", { n: i + 1, stage: this.names.get(s) })), { onclick: () => this.show(i) })));
  }

  renderBody() {
    const outer = this.list[this.at], step = this.member(outer), notes = step.notes.length ? step.notes : outer.notes;
    const banner = this.answer.degraded ? el("p", { class: "hint how-miss" }, t("how.degraded")) : null;
    this.parts.body.replaceChildren(el("div", { class: "how-step enter" }, banner, ...notes.map((n) => this.note(n)),
      outer.members ? this.memberPicker(outer) : null, this.body(step)));
  }

  // One step's body: its explainer's (the shape module's for the input step when the page has one);
  // an explainer that throws leaves the step's words and the error on the page, and the probe's
  // reading names it — the walk goes on.
  body(step) {
    try {
      if (step.kind === "input" && this.ctx.inputStep) return this.ctx.inputStep(this.answer.ex, this.ctx.row, this.env());
      return this.explainer(step.kind).render(this.stepFor(step), this.answer, this.env());
    } catch (e) {
      const why = String(e && e.message ? e.message : e);
      this.failed.set(step, why);
      return el("p", { class: "hint how-miss" }, t("how.error", { stage: this.names.get(step) || "", why: data(why) }));
    }
  }

  renderNav() {
    const P = this.parts, n = this.list.length, last = this.at === n - 1;
    P.count.textContent = `${this.at + 1} / ${n}`;
    P.back.hidden = this.at === 0; P.back.onclick = () => this.show(this.at - 1);
    P.next.replaceChildren(last ? t("how.again") : t("how.next", { step: this.names.get(this.list[this.at + 1]) }));
    P.next.onclick = () => this.show(last ? 0 : this.at + 1);
    P.code.hidden = !this.ctx.onCode;
    P.code.onclick = () => this.ctx.onCode && this.ctx.onCode(codeId(this.ctx.graph, this.member(this.list[this.at])));
  }

  // The member of a collapsed step shown (its picker's choice), or the step itself.
  member(step) { return step.members ? step.members[this.members.get(step) || 0] : step; }

  memberPicker(step) {
    const on = this.members.get(step) || 0;
    return el("div", { class: "picks how-members" }, ...step.members.map((m, k) => Object.assign(el("button", { class: k === on ? "on" : "" }, this.names.get(m)),
      { onclick: () => { this.members.set(step, k); this.show(this.at); } })));
  }

  // A "then …" note of a stage that does nothing while answering (how.then.<kind>).
  note(n) {
    const node = this.ctx.graph.byId(n.stage), shape = (s) => data((s || []).join(" × "));
    const p = node ? node.dropout || Number(node.config.p) || 0 : 0;
    return el("p", { class: "hint how-then" }, t(`how.then.${n.kind}`, { p, from: shape(node && node.in_shape), to: shape(node && node.out_shape) }));
  }

  explainer(kind) {
    const x = this.explainers && (this.explainers.get(kind) || this.explainers.get("custom"));
    if (!x) throw new Error(tText("how.gap.none", { kind: data(kind) }));
    return x;
  }

  // The step as an explainer receives it: the step with the reader's pick, and the step itself as
  // `orig` (the stepper's maps are keyed by it).
  stepFor(step) { return { ...step, pick: this.picks.has(step) ? this.picks.get(step) : null, orig: step }; }

  // The unit a step shows: the reader's pick, else its explainer's default (-1: none).
  pickOf(step) {
    if (!step || step.kind === "input") return -1;
    if (this.picks.has(step)) return this.picks.get(step);
    const x = this.explainers && this.explainers.get(step.kind);
    const u = x && x.unit ? x.unit(this.stepFor(step), this.answer, this.ctx.graph) : null;
    return Number.isInteger(u) ? u : -1;
  }

  env() {
    const step = this.member(this.list[this.at]);
    return { ...this.shared, ...this.ctx, answer: this.answer, link: this.link, steps: this.list, index: this.at, target: targetWords(this.answer),
             nameOf: (s) => (s ? this.names.get(s.orig || s) || "" : ""), pickOf: (s) => this.pickOf(s && s.orig ? s.orig : s), next: this.list[this.at + 1] || null,
             choose: (u) => { this.picks.set(step, u); this.show(this.at); } };
  }

  // Per step (a collapsed step's members each): its index, kind, name, stage, the unit it shows and
  // its recompute (null when the explainer cannot recompute it).
  values() {
    if (!this.available() || !this.explainers) return [];
    return this.list.flatMap((outer, index) => (outer.members || [outer]).map((s) => {
      const x = this.explainers.get(s.kind);
      let check = null;
      try { check = x ? x.recompute(this.stepFor(s), this.answer, this.ctx.graph, { ...this.shared, ...this.ctx }) : null; } catch (e) { check = { error: String(e && e.message ? e.message : e) }; }
      return { index, kind: s.kind, name: this.names.get(s), stage: s.stage || s.group || null, unit: this.pickOf(s), check };
    }));
  }

  // The probe's reading (extendProbe "stepper"): the steps, the step shown, the values, the
  // explainers that fell back to the generic one, and the steps whose explainer failed.
  reading() {
    return { steps: this.list.length, at: this.at, values: this.values(), gaps: gapsOf(this.explainers),
             failed: [...this.failed].map(([s, why]) => ({ name: this.names.get(s), kind: s.kind, why })) };
  }
}
