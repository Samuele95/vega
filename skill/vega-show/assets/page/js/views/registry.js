// The view registry (D4 §6.2): the lane's view key picks one task module, its sample shape one
// shape module, and the views fill their slots from the two. Eight task modules cover the ten task
// kinds (a sequence task reads its base's), three shape modules the three sample shapes. Both are
// loaded lazily, only the two a lane needs.
//
// A task module exports (every member mandatory):
//   output = {verdict(p, env), agree(p, preset, env), detail(p, env), note(env)}    try one
//   summary = {sentence(preds, truth, env), left(preds, truth, env), right(preds, truth, env),
//              columns(env), cells(pred, truth, env), miss(pred, truth)}           many at once
//   weightsAxis(attribution, env) -> {columns: [{id, name, role}], title}           the weights read
//   breakdown(block, values, env) -> Node | null     results (block: story results.breakdown@1)
//   headLabel: "head.caption@<view key>"                                            the head's caption
// where p, pred are the service's answers for one row (D2's RowPrediction: the keys of the reading,
// R-C4) and truth the true values the reader supplied (or null). left and right answer
// {title, body, caption} (right: null when the lane has nothing to put beside the chart).
// weightsAxis reads the demo sample's attribution (vega/attribution@2, or its one form in
// net/weights.js attributionOf): its columns are the attribution's own column ids, the classes, the
// binary sides and the labels matched by name in the lane's order (D4 §6.4), and the network view's
// weights read takes them for the columns of its map on real rows.
// A shape module exports:
//   editor = {mount(host, schema, env), load(preset, env), read(), describe(preset, env)}
//   sources(examples, env) -> [{id, title, sub, run}]                                many at once
//   attribution(ex, env) -> Node                                                     what tipped
//   inputStep(ex, row, env) -> Node                                                  the stepper's input
//   drawInput = {measure, layout, draw, hit, caption, acts}                          the input segment
// and may export nodeNames(ex, env) -> (id) => string (graph: the stepper names nodes as the drawings do).
// env: {lane, lex, roles, graph, story, page, schema, output, examples} (one object, PS4).
import { tText, data } from "../core/text.js";
import { VIEW_KEYS as LANE_VIEW_KEYS } from "../core/lane.js";

export const VIEW_KEYS = LANE_VIEW_KEYS;
export const TASK_MODULES = Object.freeze({
  classes: "./tasks/classes.js", binary: "./tasks/binary.js", labels: "./tasks/labels.js", value: "./tasks/value.js",
  ordinal: "./tasks/ordinal.js", reconstruction: "./tasks/reconstruction.js", anomaly: "./tasks/anomaly.js",
  horizon: "./tasks/horizon.js",
});
export const SHAPE_MODULES = Object.freeze({ row: "./shapes/row.js", window: "./shapes/window.js", graph: "./shapes/graph.js" });

// Each export a module owes, and the members an object export owes (a function export: null; the
// head's caption key: "string").
export const TASK_MEMBERS = Object.freeze({
  output: Object.freeze(["verdict", "agree", "detail", "note"]),
  summary: Object.freeze(["sentence", "left", "right", "columns", "cells", "miss"]),
  weightsAxis: null, breakdown: null, headLabel: "string",
});
export const SHAPE_MEMBERS = Object.freeze({
  editor: Object.freeze(["mount", "load", "read", "describe"]),
  sources: null, attribution: null, inputStep: null,
  drawInput: Object.freeze(["measure", "layout", "draw", "hit", "caption", "acts"]),
});
const SLOTS = Object.freeze({ task: Object.freeze({ modules: TASK_MODULES, members: TASK_MEMBERS }),
                              shape: Object.freeze({ modules: SHAPE_MODULES, members: SHAPE_MEMBERS }) });

// A lane the registry cannot serve: `slot` (task | shape) and `key` (the view key or the shape)
// name the module, `why` what is wrong with it (the words: registry.gap.*).
export class RegistryGap extends Error {
  constructor(slot, key, why) {
    super(tText("registry.gap", { slot: data(slot), key: data(key), why }));
    Object.assign(this, { name: "RegistryGap", slot, key, why });
  }
}

// The members a loaded module lacks, against a member table ("output.verdict", "headLabel", …).
export function missingMembers(mod, members) {
  const out = [];
  for (const [name, want] of Object.entries(members)) {
    const got = mod ? mod[name] : undefined;
    if (want === null) { if (typeof got !== "function") out.push(name); } else if (want === "string") {
      if (typeof got !== "string" || !got) out.push(name);
    } else if (!got || typeof got !== "object") out.push(name);
    else want.filter((m) => typeof got[m] !== "function").forEach((m) => out.push(`${name}.${m}`));
  }
  return out;
}

// The module of one slot for one key: loaded, checked, or a RegistryGap.
export async function moduleFor(slot, key, { importer = (path) => import(path) } = {}) {
  const spec = SLOTS[slot], path = spec && spec.modules[key];
  if (!path) throw new RegistryGap(slot, key, tText("registry.gap.unmapped"));
  let mod;
  try { mod = await importer(path); } catch (e) { throw new RegistryGap(slot, key, tText("registry.gap.unloaded", { error: data(e && e.message ? e.message : e) })); }
  const lacks = missingMembers(mod, spec.members);
  if (lacks.length) throw new RegistryGap(slot, key, tText("registry.gap.members", { members: data(lacks.join(", ")) }));
  return mod;
}

// {task, shape}: the two modules a lane renders with (D4 §6.2 resolve(lane)).
export async function resolve(lane, opts = {}) {
  const [task, shape] = await Promise.all([moduleFor("task", lane.viewKey, opts), moduleFor("shape", lane.shape, opts)]);
  return { task, shape };
}

// The probe's reading of the registry (R-F1 registry()): the lane's two modules and every key known.
export function registryReading(lane, resolved) {
  return { view_key: lane ? lane.viewKey : null, shape: lane ? lane.shape : null,
           task: lane ? TASK_MODULES[lane.viewKey] || null : null, shape_module: lane ? SHAPE_MODULES[lane.shape] || null : null,
           resolved: !!resolved, view_keys: VIEW_KEYS.slice(), shapes: Object.keys(SHAPE_MODULES) };
}
