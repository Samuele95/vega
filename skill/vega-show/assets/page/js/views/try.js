// The try-one view (D4 §6.3): the presets on the left, grouped (typical, edge cases), each dot in the
// role colour of what it should get; on the right the preset's description (the shape module's
// editor), the try canvas (the network drawing with this answer's activity), the answer (the task
// module: verdict, agreement, detail, note), the service's warnings, what tipped the answer (the
// shape module), the editor (a change answers again after a pause), and how the network computed it
// (the stepper, js/how/stepper.js: class Stepper {constructor(ctx); mount(host); show(i); steps()}).
// One call per answer: POST v1/explain with the editor's request; a later answer supersedes an
// earlier one.
import { $, el, restart } from "../core/dom.js";
import { S } from "../core/state.js";
import { URLS, post } from "../core/api.js";
import { t, tText, data } from "../core/text.js";
import { openDisclose } from "../core/disclose.js";
import { NetCanvas } from "../net/canvas.js";
import { viewState, problemText } from "./shared.js";
import { presetsOf, predictionOf, explainedName } from "./payloads.js";
import { anomalySides } from "./tasks/shared.js";
import { drawReady } from "./network.js";
import { openCode } from "./code.js";

const EDIT_PAUSE_MS = 400;
const state = () => viewState("try", { preset: null, mode: "simple", timer: null, stepper: null, explainers: null, asked: 0 });
const mods = () => S.mods;

// The role class of a preset's dot: the role of the class (or side) it should get.
function presetRole(p) {
  if (!S.roles || p.expected === null || p.expected === undefined) return "";
  if (p.expectedKind === "alarm") { const sides = anomalySides(S.env); return S.roles.cls(p.expected ? sides.anomalous : sides.normal); }
  return p.expectedKind === "class" || typeof p.expected === "string" ? S.roles.cls(String(p.expected)) : "";
}

function renderPresets() {
  const host = $("presets");
  if (host.dataset.done) return;
  host.dataset.done = "1";
  const groups = new Map();
  presetsOf(S.examples).forEach((p) => {
    const k = p.group || `text:${p.groupText}`;
    if (!groups.has(k)) {
      const list = el("div");
      groups.set(k, list);
      host.append(el("h3", {}, p.group ? t(`preset.group.${p.group}`) : p.groupText), list);
    }
    groups.get(k).append(Object.assign(el("button", { "data-id": p.id, class: presetRole(p) }, p.name), { onclick: () => pickPreset(p) }));
  });
  if (!groups.size) host.append(el("p", { class: "hint" }, t("try.no_presets")));
}

// p: a preset (payloads.presetsOf), or a row of many at once ({id: null, name, text, expected, row}).
export function pickPreset(p, title = null) {
  const st = state();
  st.preset = p;
  document.querySelectorAll("#presets button").forEach((b) => b.classList.toggle("on", !!p.id && b.dataset.id === p.id));
  $("try-empty").hidden = true; $("try-out").hidden = false; restart($("try-out"));
  const d = mods().shape.editor.describe(p, S.env);
  $("try-desc-title").textContent = title || d.title;
  $("try-desc").textContent = d.text || "";
  mods().shape.editor.load(p, S.env);
  runOne(mods().shape.editor.read());
}

// A row of many at once, followed through the network: the view opens with that row.
export function tryRow(row, truth, i) {
  S.tour.show("try");
  pickPreset({ id: null, kind: null, group: null, name: tText("many.row.name", { i }), text: tText("many.row.description"),
               expected: truth === undefined ? null : truth, expectedKind: null, row }, tText("many.row.title", { i }));
}

function clearAnswer() {
  $("try-err").textContent = "";
  $("try-verdict").replaceChildren(el("span", { class: "busy" }, t("try.busy")));
  $("try-agree").replaceChildren(); $("try-warn").replaceChildren(); $("try-probs").replaceChildren(); $("how-body").replaceChildren();
  const why = $("why").firstElementChild;
  if (why) why.replaceChildren();
}

// The request of the editor, for the service it goes to: a v2.3 service explains exactly (R-F5); a
// v2.2 one takes the bare request.
const requestOf = (req) => (S.graph && !S.graph.legacy ? { ...req, detail: "exact" } : req);

export async function runOne(req) {
  const st = state(), asked = ++st.asked;
  clearAnswer();
  try {
    const ex = (await post(URLS.explain, requestOf(req))).explanation;
    if (asked !== st.asked) return;                              // a later answer supersedes this one
    Object.assign(S, { ex, row: req.row || null });
    if (S.graph) playTry(ex, () => { if (ex === S.ex) showAnswer(ex); }); else showAnswer(ex);
  } catch (e) {
    if (asked !== st.asked) return;
    $("try-verdict").replaceChildren(); $("try-err").textContent = problemText(e); openDisclose("edit");
  }
}

// The try drawing: one answer, at either level of detail; switching replays the same explanation.
function playTry(ex, done) {
  drawReady().then((D) => {
    if (!D.tryNet) D.tryNet = new NetCanvas($("net-try"), S.graph, { renderers: D.renderers, isStatic: true, k: 12, env: { roles: S.roles, story: S.story } });
    const mode = state().mode, name = explainedName(ex);
    D.tryNet.setMode(mode);
    D.tryNet.showActivations(ex); D.tryNet.drawing.phase = 0;
    $("try-read").textContent = !D.tryNet.pure ? tText("net.try.stages", { name })
      : mode === "detail" ? tText("legacy.net.try.detail", { name }) : tText("legacy.net.try.simple", { name, k: D.tryNet.k });
    D.tryNet.animate(mode === "detail" ? 2000 : 1600, done);
  }, () => { $("try-read").textContent = ""; if (done) done(); });
}

// How the network computed it: the stepper of js/how/ (loaded when first needed), mounted in
// #how-box with this answer; the explainers it loaded are kept for the next answer.
async function mountStepper(ex, row) {
  const st = state(), box = $("how-box");
  if (!S.graph) { box.hidden = true; return; }
  try {
    const { Stepper } = await import("../how/stepper.js");
    // the view's env (its output record: the thresholds the service decides with) goes on to the
    // explainers, so the output step decides as the answer above it did
    // a graph's nodes go by the names the shape gives them in its drawings (its optional nodeNames)
    const shape = mods().shape, nodeName = typeof shape.nodeNames === "function" ? shape.nodeNames(ex, S.env) : null;
    const stepper = new Stepper({ env: S.env, graph: S.graph, ex, row, explainers: st.explainers, lane: S.lane, page: S.page, roles: S.roles,
                                  onCode: (id) => openCode(id), inputStep: shape.inputStep, nodeName });
    st.stepper = stepper;
    await stepper.mount(box);
    st.explainers = stepper.explainers;
  } catch (e) {
    box.hidden = true;
    st.stepperError = String(e && e.message ? e.message : e);
  }
}

async function showAnswer(ex) {
  const task = mods().task, env = S.env, a = { p: predictionOf(ex) || {}, ex };
  await mountStepper(ex, S.row);
  if (ex !== S.ex) return;
  $("try-verdict").replaceChildren(task.output.verdict(a, env));
  const agree = task.output.agree(a, state().preset, env);
  $("try-agree").className = "agree" + (agree ? " enter" : ""); $("try-agree").replaceChildren(...(agree ? [agree] : []));
  (a.p.warnings || []).forEach((w) => $("try-warn").append(el("div", { class: "warn enter" }, data(w))));
  $("try-probs").replaceChildren(task.output.detail(a, env));
  const note = task.output.note(env), slot = $("try-probs").nextElementSibling;
  if (slot) { slot.removeAttribute("data-t"); slot.hidden = !note; slot.replaceChildren(...(note ? [note] : [])); }
  const why = $("why").firstElementChild;
  if (why) why.replaceChildren(mods().shape.attribution(ex, env));
}

function bindSwitch() {
  document.querySelectorAll("#try-switch button").forEach((b) => { b.onclick = () => {
    state().mode = b.dataset.mode;
    document.querySelectorAll("#try-switch button").forEach((x) => x.classList.toggle("on", x === b));
    if (S.ex) playTry(S.ex);
  }; });
}

// Whether a column of the schema can hold a non-finite value (the editor's hint is shown only then).
const nonFinite = (schema) => !!schema && ((Array.isArray(schema.input) && schema.input.some((c) => c.non_finite_seen)) || Array.isArray(schema.tokens));

// The editor, once: mounted from /v1/schema; a change answers again after a pause.
function mountEditor() {
  const form = $("form"), st = state();
  if (form.dataset.mounted) return;
  form.dataset.mounted = "1";
  mods().shape.editor.mount(form, S.schema, S.env);
  const hint = document.querySelector("#edit .hint");
  if (hint) hint.hidden = !nonFinite(S.schema);
  form.addEventListener("input", () => {
    clearTimeout(st.timer);
    st.timer = setTimeout(() => again(), EDIT_PAUSE_MS);
  });
}

// Answers the editor's row again (a preset's expected answer no longer applies once it is edited).
function again() {
  const st = state();
  clearTimeout(st.timer);
  st.preset = st.preset && { ...st.preset, expected: null };
  runOne(mods().shape.editor.read());
}

function show() {
  renderPresets();
  mountEditor();
  const D = S.draw;
  if (D && D.tryNet) { D.tryNet.resize(); D.tryNet.draw(); }
}

// The controls that exist from the start.
export function bind() {
  $("try-run").onclick = again;
  bindSwitch();
}

export const view = Object.freeze({ id: "try", show });
