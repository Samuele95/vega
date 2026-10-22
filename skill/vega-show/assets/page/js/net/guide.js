// The reading guide (D4 §7.5): four steps over the drawing, composed from the stage graph — the
// inputs (how many, in how many families, of what), the middle (none for a network with nothing
// between its inputs and its head; today's sentence when it is hidden layers only; otherwise one
// phrase per kind of segment, in order), the lines, the outputs (by the lane's view key). Each step
// lights its part of the drawing through the graph's focus. Guide runs the steps in the shell's
// guide slots and remembers, per browser, that it was read. The steps' words are D4's guide.* keys
// (network.en.js: guide.inputs@<shape>, guide.edges, guide.outputs@<view key>); today's middle and
// the guide's own buttons are still the core fragment's legacy.net.guide.*.
import { $, store } from "../core/dom.js";
import { tText, tnText, data } from "../core/text.js";
import { META } from "../core/strings.en.js";
import { fmtInt } from "../core/fmt.js";
import { STEP_KINDS } from "./kinds.js";
import { bottleneckOf } from "./renderers/latent.js";

const PHRASE_OF = Object.freeze({
  conv: "conv", pool1d: "conv", recurrent: "recurrent", attention: "attention", attention_pool: "attention",
  transformer: "transformer", graph_conv: "graph_conv", latent: "latent",
});
const DONE = "✓", STORE_KEY = "guide";

// "a, b and c" with the table's list words
export function joinList(items) {
  if (items.length < 2) return items.join("");
  return items.slice(0, -1).join(META.list.comma) + META.list.and + items[items.length - 1];
}

const widths = (run, graph) => [...new Set(run.layers.map((l) => graph.byId(l.dense).out_shape.slice(-1)[0]))].map(fmtInt);
const middleOf = (graph) => graph.segments().filter((s) => s.kind !== "input" && s.kind !== "head");

// The middle's phrases in drawing order: each run's hidden layers, each group once (its inside is
// what the group's phrase says it does), each other computing kind counted where it first appears.
function phrases(graph) {
  const counts = new Map(), order = [], add = (key, n, params = {}) => {
    if (!counts.has(key)) { order.push(key); counts.set(key, { n: 0, params }); }
    counts.get(key).n += n;
  };
  middleOf(graph).forEach((s) => {
    if (s.kind === "run") add(`run:${order.length}`, s.layers.length, { units: widths(s, graph).join(META.list.comma) });
    else if (s.kind === "group") add(s.mode, 1);
    else if (s.kind === "box") {
      const kind = graph.byId(s.stage).kind;
      if (PHRASE_OF[kind]) add(PHRASE_OF[kind], 1);
      else if (STEP_KINDS.includes(kind) && kind !== "activation") add("box", 1);   // a bend belongs to the layer before it
    }
  });
  return order.map((key) => {
    const { n, params } = counts.get(key), base = key.startsWith("run:") ? "run" : key;
    return tnText(`guide.phrase.${base}`, n, params);
  });
}

// The dense waist of an autoencoder or a PCA floor (latent.js bottleneckOf): the narrowest layer
// every row passes through before the decoder. A VAE's latent code has its own phrase.
function bottleneck(graph) {
  const waist = bottleneckOf(graph);
  return waist && waist.kind === "dense" ? tText("guide.phrase.bottleneck", { units: waist.units }) : null;
}

// The middle step's text, or null: hidden layers alone read as today; anything else is composed.
function middleText(graph, k) {
  const segs = middleOf(graph), waist = bottleneck(graph);
  if (segs.length === 1 && segs[0].kind === "run" && !waist) {
    const run = segs[0];
    return tnText("legacy.net.guide.hidden", run.layers.length, { units: widths(run, graph).join(META.list.comma), k });
  }
  const middle = [...phrases(graph), ...[waist].filter(Boolean)];
  if (!middle.length) return null;
  const note = graph.denseRuns().length ? " " + tText("guide.note.run", { k }) : "";
  return tText("guide.middle", { phrases: joinList(middle) }) + note;
}

// How many outputs the outputs step counts: the steps ahead of a forecast (its quantiles are bands of
// a step), the classes of an ordinal head (one more than its thresholds), else the output names.
function outputCount(graph) {
  const names = graph.task.outputs.names;
  if (graph.link === "horizon") return graph.sample.horizon || new Set(names.map((n) => String(n).split("/")[0])).size;
  return graph.link === "cumulative" ? names.length + 1 : names.length;
}

// The steps [{focus: Set | null, text}] of a graph; opts: {families (the drawing's), k (units drawn per layer)}.
export function guideSteps(graph, { families, k }) {
  const sample = graph.sample, names = graph.task.outputs.names, head = graph.head();
  const inner = new Set([...graph.focusSet({ kind: "family", ref: "architecture" })]
    .filter((m) => m !== "input" && m !== "head" && !(head && m === head.id)));
  graph.focusSet("activation").forEach((m) => inner.add(m));
  const inputs = tnText("guide.inputs", families.length, { n: graph.inputs.names.length, f: families.length, length: sample.length });
  const steps = [{ focus: graph.focusSet("inputs"), text: inputs }];
  const middle = middleText(graph, k);
  if (middle) steps.push({ focus: inner, text: middle });
  const positive = graph.task.positive || names[names.length - 1];
  steps.push({ focus: graph.focusSet("edges"), text: tText("guide.edges") });
  steps.push({ focus: graph.focusSet("outputs"), text: tText("guide.outputs", { n: outputCount(graph), positive: data(positive) }) });
  return steps;
}

// The guide in the shell (#guide-n, #guide-t, #guide-go, #guide-skip). onFocus(Set | null) lights the
// drawing; a guide read to its end stays read (browser store) and starts at its last word.
export class Guide {
  constructor(steps, { onFocus }) {
    Object.assign(this, { steps, onFocus, at: store.get(STORE_KEY) === "done" ? steps.length : 0 });
  }

  render() {
    const n = this.steps.length;
    if (this.at >= n) {
      $("guide-n").textContent = DONE; $("guide-t").textContent = tText("legacy.net.guide.done");
      $("guide-go").textContent = tText("legacy.net.guide.again"); $("guide-go").onclick = () => { this.at = 0; this.render(); };
      $("guide-skip").hidden = true;
      store.set(STORE_KEY, "done");
      this.onFocus(null);
      return;
    }
    const s = this.steps[this.at];
    $("guide-n").textContent = `${this.at + 1} / ${n}`;
    $("guide-t").textContent = s.text;
    $("guide-go").textContent = tText(this.at === n - 1 ? "legacy.net.guide.got_it" : "legacy.net.guide.next");
    $("guide-go").onclick = () => this.next();
    $("guide-skip").hidden = false; $("guide-skip").onclick = () => { this.at = n; this.render(); };
    this.onFocus(s.focus);
  }

  next() { this.at++; this.render(); }

  // → and space step the guide while it runs
  key(e) {
    if ((e.key === "ArrowRight" || e.key === " ") && this.at < this.steps.length) { e.preventDefault(); this.next(); }
  }
}
