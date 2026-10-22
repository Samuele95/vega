// The explainer registry (D4 §8.1): the module that explains each step kind of "how the network
// computed it", loaded by dynamic import, only those a network's steps need. The kinds are the
// stepper's: the stage kinds that compute (net/kinds.js STEP_KINDS), the input step, the step a
// residual group adds (its sum, "residual") and a parallel group's ("merge"), the link step (the
// head's sum and the link; "head" names it too), and the kinds that only leave a note (a dropout,
// a reshape) when a collapsed network gives them a step of their own. The module of a block kind
// follows the explain catalog's recompute table (explain-catalog.json#recompute: an activation is
// the dense explainer's, pool1d the pool explainer's, a parallel block's merge the residual
// explainer's); the head's weighted sum is the link step's.
//
// Every explainer module's default export is {kinds, render(step, ex, env) -> Node,
// recompute(step, ex, graph) -> {page, network, gap, within, …} | null}, and may add unit(step, ex,
// graph) -> the unit it shows by default. A kind whose module does not load, lacks a member or
// does not list the kind falls back to the generic explainer (formula and shapes, no recompute):
// the page still walks every step, and gaps() names what fell back, for the probe. The modules of
// the sequence, graph and latent kinds (conv, recurrent, attention, graph, latent) are W14's.
import { tText, data } from "../../core/text.js";

export const EXPLAINER_OF = Object.freeze({
  input: "input.js", link: "link.js", head: "link.js",
  dense: "dense.js", activation: "dense.js", decoder: "dense.js",
  norm: "norm.js", residual: "residual.js", merge: "residual.js",
  embedding: "embedding.js", concat: "embedding.js",
  pool: "pool.js", pool1d: "pool.js", reshape: "pool.js",
  custom: "generic.js", positional: "generic.js", edge_dropout: "generic.js", dropout: "generic.js",
  conv: "conv.js", recurrent: "recurrent.js", attention: "attention.js", attention_pool: "attention.js",
  transformer: "attention.js", tokenizer: "attention.js", latent: "latent.js",
  graph_conv: "graph.js", graph_pool: "graph.js",
});

export const MEMBERS = Object.freeze(["kinds", "render", "recompute"]);
export const GENERIC = "generic.js";
// The steps every network has: its inputs, and its link.
export const ALWAYS = Object.freeze(["input", "link"]);

export const explainerOf = (kind) => (Object.prototype.hasOwnProperty.call(EXPLAINER_OF, kind) ? EXPLAINER_OF[kind] : null);

async function loadModule(file) {
  const mod = await import(`./${file}`);
  return mod.default;
}

// Why a loaded module cannot explain `kind` (a key of the strings table), or null when it can.
function unfit(mod, kind) {
  if (!mod || MEMBERS.some((m) => !(m in mod))) return "how.gap.members";
  return mod.kinds.includes(kind) ? null : "how.gap.unlisted";
}

// Map<kind, explainer> for the kinds given (a Set or an array) plus the input and link steps (and
// `link`, the lane's link, for the record). Each module is imported once; every import settles
// before the map is built, so a failing module leaves no rejection unhandled. The map carries
// `gaps`: [{kind, module, why}] for each kind that fell back to the generic explainer.
// opts.importer: the module loader (file → its default export), replaceable in tests.
export async function load(kinds, link = null, { importer = loadModule } = {}) {
  const wanted = [...new Set([...ALWAYS, ...kinds])], fileOf = new Map(wanted.map((k) => [k, explainerOf(k) || GENERIC]));
  const files = [...new Set([GENERIC, ...fileOf.values()])];
  const settled = await Promise.allSettled(files.map((f) => Promise.resolve().then(() => importer(f))));
  const loaded = new Map(files.map((f, i) => [f, settled[i]]));
  const generic = loaded.get(GENERIC).status === "fulfilled" ? loaded.get(GENERIC).value : null;
  const out = new Map(), gaps = [];
  for (const kind of wanted) {
    const file = fileOf.get(kind), s = loaded.get(file);
    const why = !explainerOf(kind) ? "how.gap.unmapped" : s.status === "rejected" ? "how.gap.unloaded" : unfit(s.value, kind);
    if (why) {
      const error = s.status === "rejected" ? String(s.reason && s.reason.message ? s.reason.message : s.reason) : "";
      gaps.push({ kind, module: file, why: tText(why, { kind: data(kind), module: data(file), error: data(error) }) });
    }
    const chosen = why ? generic : s.value;
    if (chosen) out.set(kind, chosen);
  }
  return Object.assign(out, { gaps, link });
}

// What fell back to the generic explainer in a loaded map (the probe's reading).
export const gapsOf = (map) => (map && Array.isArray(map.gaps) ? map.gaps.slice() : []);

// Every module the registry names (the coverage test: every step kind resolves to one).
export const MODULES = Object.freeze([...new Set(Object.values(EXPLAINER_OF))]);
// The modules this package ships (the others are the advanced explainers, W14's).
export const CORE_MODULES = Object.freeze(["input.js", "dense.js", "norm.js", "residual.js", "embedding.js", "pool.js", "generic.js", "link.js"]);
