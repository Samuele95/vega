// The lane (D4 §5): what the page renders is read from page.json#lane and cross-checked, at boot,
// against the network the service answers with (GET v1/network): the task and its base, the view
// key the page speaks in, the sample shape and graph level, the link that turns the head's raw
// output into the answer, the family, and (anomaly) where the score comes from. A page.json whose
// lane contradicts itself or the network is refused with a LaneMismatch: the showcase was built for
// another network (the page shows app.lane_mismatch and stops).
//
//   task kind (netspec.task.kind)            base                         view key         link
//   multiclass                               —                            classes          softmax
//   binary                                   —                            binary           sigmoid
//   multilabel                               —                            labels           sigmoid_each
//   regression                               —                            value            identity
//   ordinal                                  —                            ordinal          cumulative
//   unsupervised (goal reconstruct)          —                            reconstruction   reconstruction
//   anomaly (loss reconstruction | vae)      —                            anomaly          reconstruction
//   anomaly (loss svdd)                      —                            anomaly          distance
//   sequence-classification                  binary | multiclass | multilabel   as the base
//   sequence-regression                      regression | ordinal               as the base
//   forecast                                 —                            horizon          horizon
//
// A graph lane at level node reads the table unchanged; at level graph only binary, multiclass and
// regression occur (vega-forge catalog graph_level.tasks). The words of a mismatch are the strings
// table's (lane.mismatch.*, views fragment); the values compared pass through data().
import { tText, data } from "./text.js";

export const TASKS = Object.freeze(["binary", "multiclass", "multilabel", "regression", "ordinal", "unsupervised", "anomaly",
  "sequence-classification", "sequence-regression", "forecast"]);
export const SHAPES = Object.freeze(["row", "window", "graph"]);
export const LEVELS = Object.freeze(["node", "graph"]);
export const SCORE_SOURCES = Object.freeze(["reconstruction", "distance"]);
export const BASES = Object.freeze({
  "sequence-classification": Object.freeze(["binary", "multiclass", "multilabel"]),
  "sequence-regression": Object.freeze(["regression", "ordinal"]),
});
export const GRAPH_LEVEL_TASKS = Object.freeze(["binary", "multiclass", "regression"]);
// The effective task (the base of a sequence task) → the view key and the link (D4 §5.1, R-C1).
export const VIEW_KEY_OF = Object.freeze({
  multiclass: "classes", binary: "binary", multilabel: "labels", regression: "value", ordinal: "ordinal",
  unsupervised: "reconstruction", anomaly: "anomaly", forecast: "horizon",
});
export const LINK_OF = Object.freeze({
  multiclass: "softmax", binary: "sigmoid", multilabel: "sigmoid_each", regression: "identity", ordinal: "cumulative",
  unsupervised: "reconstruction", forecast: "horizon",
});
export const VIEW_KEYS = Object.freeze([...new Set(Object.values(VIEW_KEY_OF))]);
const GOAL_OF = Object.freeze({ unsupervised: "reconstruct", anomaly: "anomaly" });
const has = (o, k) => !!o && Object.prototype.hasOwnProperty.call(o, k);

// A lane the page cannot render for the network it was given: `field` names what differs, `page`
// and `network` the two values (the network's is null when the page's lane contradicts itself).
export class LaneMismatch extends Error {
  constructor(field, page, network = null) {
    const shown = (v) => data(JSON.stringify(v === undefined ? null : v));
    const what = network === null ? tText("lane.mismatch.self", { field: data(field), page: shown(page) })
      : tText("lane.mismatch.field", { field: data(field), page: shown(page), network: shown(network) });
    super(tText("app.lane_mismatch", { what }));
    Object.assign(this, { name: "LaneMismatch", field, page, network });
  }
}

// The effective task of a kind and its base: a sequence task reads as its base.
export const effectiveOf = (task, base) => (String(task).startsWith("sequence-") ? base : task);

// {viewKey, link, goal} of a task, its base and (anomaly) its score source; throws LaneMismatch
// when the three do not make a lane (an unknown task, a base the task does not take, a score source
// on a task without one).
export function derive({ task, base = null, scoreSource = null }) {
  if (!TASKS.includes(task)) throw new LaneMismatch("task", task);
  const bases = BASES[task] || [null];
  if (!bases.includes(base)) throw new LaneMismatch("base", base);
  if (task === "anomaly" ? !SCORE_SOURCES.includes(scoreSource) : scoreSource !== null) throw new LaneMismatch("score_source", scoreSource);
  const effective = effectiveOf(task, base);
  const link = task === "anomaly" ? (scoreSource === "distance" ? "distance" : "reconstruction") : LINK_OF[effective];
  return { viewKey: VIEW_KEY_OF[effective], link, goal: GOAL_OF[task] || "predict" };
}

// Whether a network payload is the v2.2 merge payload a bundle packed before v2.3 serves (no schema,
// a dense chain on rows): it names no task and no sample to cross-check against.
export const isLegacyNetwork = (network) => !!network && typeof network === "object" && !has(network, "schema") && Array.isArray(network.layers);

export class Lane {
  // fields: {task, base, viewKey, shape, level, link, family, goal, scoreSource}
  constructor(fields) { Object.assign(this, { fields: Object.freeze({ ...fields }) }); Object.freeze(this); }

  get task() { return this.fields.task; }
  get base() { return this.fields.base; }
  get viewKey() { return this.fields.viewKey; }
  get shape() { return this.fields.shape; }
  get level() { return this.fields.level; }
  get link() { return this.fields.link; }
  get family() { return this.fields.family; }
  get goal() { return this.fields.goal; }
  get scoreSource() { return this.fields.scoreSource; }
  get effective() { return effectiveOf(this.task, this.base); }

  // The strings table's and the registry's variant order: view key and shape, view key, shape.
  variants() { return [`${this.viewKey}+${this.shape}`, this.viewKey, this.shape]; }

  // page.json#lane's own spelling (the probe's lane()).
  toJSON() {
    const f = this.fields;
    return { task: f.task, base: f.base, view_key: f.viewKey, shape: f.shape, level: f.level, link: f.link,
             family: f.family, goal: f.goal, score_source: f.scoreSource };
  }

  // The lane of page.json#lane, held to itself (the view key and the link its task derives) and to
  // the network: GET v1/network's task and sample (vega/network-view@2). A v2.2 payload names no
  // task: its lane can only be a row lane. `network` null (inspection off, the network not read):
  // no cross-check.
  static from(pageLane, network = null) {
    const p = pageLane || {};
    const derived = derive({ task: p.task, base: p.base ?? null, scoreSource: p.score_source ?? null });
    for (const [field, want] of [["view_key", derived.viewKey], ["link", derived.link], ["goal", derived.goal]]) {
      if (p[field] !== want) throw new LaneMismatch(field, p[field]);
    }
    const shape = p.shape, level = p.level ?? null;
    if (!SHAPES.includes(shape)) throw new LaneMismatch("shape", shape);
    if (shape === "graph" ? !LEVELS.includes(level) : level !== null) throw new LaneMismatch("level", level);
    if (level === "graph" && !GRAPH_LEVEL_TASKS.includes(effectiveOf(p.task, p.base ?? null))) throw new LaneMismatch("level", level);
    if (typeof p.family !== "string" || !p.family) throw new LaneMismatch("family", p.family);
    const lane = new Lane({ task: p.task, base: p.base ?? null, viewKey: derived.viewKey, shape, level, link: derived.link,
                            family: p.family, goal: derived.goal, scoreSource: p.score_source ?? null });
    lane.check(network);
    return lane;
  }

  // Throws LaneMismatch on the first field the network states otherwise.
  check(network) {
    if (!network || network.available === false) return;
    if (isLegacyNetwork(network)) {
      if (this.shape !== "row") throw new LaneMismatch("shape", this.shape, "row");
      return;
    }
    const task = network.task || {}, sample = network.sample || {};
    const pairs = [["task", this.task, task.kind], ["base", this.base, task.base ?? null], ["link", this.link, task.link],
                   ["shape", this.shape, sample.shape], ["level", this.level, sample.level ?? null]];
    for (const [field, mine, theirs] of pairs) if (mine !== theirs) throw new LaneMismatch(field, mine, theirs);
  }
}
