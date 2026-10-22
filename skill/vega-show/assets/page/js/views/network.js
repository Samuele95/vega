// The network view (D4 §6.3): the story's lead; the drawing of the stage graph (js/net/**: the canvas
// and its renderers, the level switch shown only when a segment has an "every weight" level, the
// reading guide, the unit and stage card); the story's decisions, each lighting up the part of the
// network it shaped (focus@1); what a unit of the first computing stage does (the backend's explain
// catalog, page.json#explain); what was tried (experiments.js); the weights read (net/weights.js);
// and the code room's walk step of the first computing block type.
//
// The drawings' state (the graph read once, its renderers, the canvases, the guide) lives in the
// page state (S.draw); the try view's canvas reads the same graph and renderers (drawReady()).
import { $, el } from "../core/dom.js";
import { S, setState } from "../core/state.js";
import { URLS, api } from "../core/api.js";
import { t, tText, hasKey, data, storyText } from "../core/text.js";
import { extendProbe } from "../core/probe.js";
import { onDisclose } from "../core/disclose.js";
import { NetCanvas, complete } from "../net/canvas.js";
import { load as loadRenderers } from "../net/renderers/index.js";
import { Guide, guideSteps } from "../net/guide.js";
import { showCard } from "../net/card.js";
import { drawWeights, attributionOf } from "../net/weights.js";
import { familiesOf } from "../net/families.js";
import { STEP_KINDS } from "../net/kinds.js";
import { section } from "./shared.js";
import { renderExperiments } from "./experiments.js";

const draw = () => S.draw || setState({ draw: { ready: null, renderers: null, net: null, tryNet: null, guide: null, unfetched: [] } }).draw;
const netEnv = () => ({ roles: S.roles, story: S.story });

// A tensor the network view deferred (D3 §4): GET v1/network/tensor/{id}.
const getTensor = (id) => api(`${URLS.network}/tensor/${encodeURIComponent(id)}`);

// The stage graph with the deferred weights the drawings read fetched, and the renderers its
// drawing needs; every caller handles a rejection (no graph, a RendererGap).
export function drawReady() {
  const D = draw();
  if (!D.ready) {
    D.ready = Promise.resolve().then(async () => {
      if (!S.graph) throw new Error(S.graphError || tText("net.absent"));
      const { failed } = await S.graph.fetchDeferred(getTensor);
      D.unfetched = failed.map((f) => tText("net.tensor.unfetched", { tensor: data(f.id), error: data(f.error) }));
      D.renderers = await loadRenderers(S.graph.kinds());
      return D;
    });
  }
  return D.ready;
}

function setMode(mode) {
  document.querySelectorAll("#net-switch button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  const net = draw().net;
  if (!net) return;
  if (mode === "detail") {
    $("unit-card").hidden = true;
    net.setMode("detail"); net.drawing.phase = 0; net.animate(1800); net.caption(null);
    $("guide").style.visibility = "hidden";
  } else if (net.mode === "detail") {
    net.setMode("simple"); net.draw(); net.caption(null);
    $("guide").style.visibility = "";
  }
}

async function drawNetwork() {
  const D = await drawReady();
  if (D.net) { D.net.resize(); D.net.draw(); return; }
  D.net = new NetCanvas($("net"), S.graph, { renderers: D.renderers, caption: $("net-cap"), env: netEnv(),
    onPick: (pick, c) => showCard($("unit-card"), pick, { graph: S.graph, model: c.pure ? c.model : null, families: c.env.families }) });
  $("net-switch").hidden = !D.net.hasDetail;
  D.guide = new Guide(guideSteps(S.graph, { families: D.net.env.families, k: D.net.k }), { onFocus: (f) => D.net.setFocus(f) });
  D.net.drawing.phase = 0; D.net.caption(null);
  D.guide.render(); D.net.animate(1500);
  document.querySelectorAll("#net-switch button").forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
}

// A decision opened (or closed: null): the part of the network it shaped lights up.
function focusNetwork(focus) {
  const net = draw().net;
  if (!net) return;
  setMode("simple");
  net.setFocus(focus);
}

function renderChoices(sec) {
  const host = $("choices");
  if (host.childElementCount) return;
  (sec.choices || []).forEach((c, i) => {
    const btn = el("button", {}, el("span", { class: "n" }, String(i + 1).padStart(2, "0")), storyText(c.title));
    const body = el("div", { class: "reveal" }, el("div", {}, storyText(c.text, el("p", { class: "choice-text" }))));
    btn.onclick = () => {
      const on = !btn.classList.contains("on");
      host.querySelectorAll("button").forEach((b) => b.classList.remove("on"));
      host.querySelectorAll(".reveal").forEach((r) => r.classList.remove("open"));
      if (on) { btn.classList.add("on"); body.classList.add("open"); }
      focusNetwork(on ? c.focus : null);
    };
    host.append(btn, body);
  });
}

// The first stage that computes an answer (a dense layer, a convolution, a recurrent layer, …): what
// "what a unit does" and the "see it in the code" link speak of.
export function firstComputing(graph) {
  if (!graph) return null;
  return graph.topLevel().map((id) => graph.byId(id)).find((n) => n && STEP_KINDS.includes(n.kind) && n.type !== "head")
    || graph.stages().find((n) => STEP_KINDS.includes(n.kind)) || null;
}

function renderUnitHow() {
  const n = firstComputing(S.graph), key = n ? `explain.${n.type}.unit_how` : null, text = key && hasKey(key) ? tText(key) : null;
  $("unit-how-box").hidden = !text;
  if (!$("unit-how-box").hidden) $("unit-how-text").textContent = text;
  const link = document.querySelector("#main .view[data-view=\"network\"] .code-link[data-code]");
  if (link && n) link.dataset.code = n.type;
}

// The weights read needs the graph only (no renderer): a graph whose drawing has a gap still shows it.
// Its map on real rows has the lane's weights axis for columns (the task module's weightsAxis, D4
// §6.4: the classes, the binary sides, the labels, the target's bins, the groups or the steps ahead),
// each read from the attribution's cell of its id; the linear reading and the layers keep the head's
// outputs.
function drawHeatmaps() {
  if (!S.graph) return;
  const graph = S.graph, D = draw();
  const seg = graph.segments().find((x) => x.kind === "run") || { layers: [] };
  const model = D.net && D.net.pure ? D.net.model : graph.isPureDenseChain() && complete(graph) ? graph.runModel(seg) : null;
  const fam = familiesOf({ groups: S.story && S.story.input_groups, names: graph.inputs.names, units: graph.featureUnits() }).families;
  const attribution = attributionOf(S.attribution || (S.examples && S.examples.attribution));
  const axis = S.mods ? S.mods.task.weightsAxis(attribution, S.env) : null;
  if (axis && axis.title) { const b = document.querySelector("#w-switch button[data-w=\"map\"]"); if (b) { b.removeAttribute("data-t"); b.replaceChildren(axis.title); } }
  drawWeights({ graph, model, families: fam, roles: S.roles, attribution, columns: axis ? axis.columns : null });
}

function redraw() {
  const D = draw();
  if (D.net) D.net.draw();
  if (D.tryNet) D.tryNet.draw();
  if ($("w-map").childElementCount) { $("w-map").replaceChildren(); $("w-layers").replaceChildren(); if ($("heat").classList.contains("open")) drawHeatmaps(); }
}

// The weights read is offered only with a stage graph to read the weights of.
function offerWeights(on) {
  const toggle = document.querySelector(".disclose[data-for=\"heat\"]");
  if (toggle) toggle.hidden = !on;
  $("heat").hidden = !on;
}

function show() {
  const sec = section("network");
  $("net-lead").replaceChildren(); if (sec.lead) storyText(sec.lead, $("net-lead"));
  renderChoices(sec);
  renderExperiments();
  offerWeights(!!S.graph);
  if (!S.graph) {
    $("net-cap").replaceChildren(S.graphError ? t("net.unreadable", { why: data(S.graphError) }) : t("net.absent"));
    $("guide").hidden = true; $("net-switch").hidden = true;
    return;
  }
  renderUnitHow();
  drawNetwork().catch((e) => { $("net-cap").replaceChildren(t("net.unreadable", { why: data(e.message) })); $("guide").hidden = true; });
}

function key(e) { const g = draw().guide; if (g) g.key(e); }

// The probe's reading of the stage graph (R-F1 graph()): what was read, and how it is drawn.
function graphReading() {
  const g = S.graph, D = draw();
  if (!g) return null;
  return { family: g.family, task: g.task.kind, link: g.link, shape: g.sample.shape, level: g.sample.level,
           kinds: [...g.kinds()].sort(), steps: [...g.stepKinds()].sort(), pure: g.isPureDenseChain(),
           segments: g.segments().map((x) => x.kind), drawn: D.net ? (D.net.pure ? "chain" : "segments") : null };
}

// The network view's two sub-views (R-F4: disclosures inside it, not views of the tour): what was
// tried and the weights read, each offered or not, opened or not, and how much it drew.
const disclosure = (id) => document.querySelector(`.disclose[data-for="${id}"]`);
export function subViews() {
  const tried = disclosure("exp"), weights = disclosure("heat");
  return {
    tried: { offered: !!tried && !tried.hidden, open: $("exp").classList.contains("open"), runs: $("exp-bars").querySelectorAll(".bar").length },
    weights: { offered: !!weights && !weights.hidden, open: $("heat").classList.contains("open"),
               cells: $("w-map").querySelectorAll(".c").length, layers: $("w-layers").childElementCount },
  };
}

// What exists from the start: the weights read opens on demand; the probe's readings.
export function bind() {
  onDisclose("heat", drawHeatmaps);
  extendProbe("graph", graphReading);
  extendProbe("netWarnings", () => [...(draw().net ? draw().net.env.problems : []), ...draw().unfetched], { report: true });
  extendProbe("subViews", () => ({ network: subViews() }), { report: true });
}

export const view = Object.freeze({ id: "network", show: () => setTimeout(show, 0), key, redraw });
