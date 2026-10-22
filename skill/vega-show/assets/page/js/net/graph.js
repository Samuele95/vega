// The stage graph (D4 §7.1): the page's only reader of the names of GET v1/network. StageGraph.from
// reads vega/network-view@2 (D3 §5, R-F7), the one form the page reads: a payload of any other form
// (a bundle packed before v2.3 serves the v2.2-merge payload) is refused by name (net.error.schema),
// and the network view says why. Everything else on the page asks the graph: the task and the
// sample, the inputs, the stages and their tree, the weights as decoded matrices, the segments the
// canvas lays out (segments.js), the kinds the renderers and the stepper need, and what a decision's
// focus lights up.
import { KIND_OF, SYNTHETIC, VIEW_KEY_OF, STEP_KINDS, GROUP_STEP, MATRIX_KINDS, kindOf } from "./kinds.js";
import { segmentsOf, runsOf, isPure, runModel, headLabels, ordinalClasses, rebuiltInputOf, fusedMarkers } from "./segments.js";
import { tText, data } from "../core/text.js";

export const VIEW_SCHEMA = "vega/network-view@2";
export const LINKS = Object.freeze(["softmax", "sigmoid", "sigmoid_each", "identity", "cumulative", "reconstruction", "distance", "horizon"]);
const REQUIRED = Object.freeze(["schema", "available", "family", "task", "sample", "inputs", "params", "nodes", "edges", "chain", "tensors"]);
const TASK_REQUIRED = Object.freeze(["kind", "link", "outputs"]);
const has = (o, k) => !!o && Object.prototype.hasOwnProperty.call(o, k);

// A payload the page cannot read as a stage graph: `key` names what is wrong (a key of the strings
// table, net.error.*), the message is its text with the payload's own words as data.
export class NetworkViewError extends Error {
  constructor(key, params = {}) {
    super(tText(key, params));
    Object.assign(this, { name: "NetworkViewError", key, params });
  }
}

// ---- tensors: json lists, or base64 little-endian float32 (D3 §4)
function bytesOf(b64) {
  if (typeof atob !== "function") return new Uint8Array(Buffer.from(b64, "base64"));
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function float32Of(t) {
  if (t.encoding === "json") return Float32Array.from(t.data);
  if (t.encoding !== "base64-le") return null;                     // deferred or withheld: not here
  const bytes = bytesOf(t.data), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(4 * i, true);
  return out;
}

// Whether a tensor payload carries its values (inline), not only a summary.
const carried = (t) => !!t && (t.encoding === "json" || t.encoding === "base64-le");

// The float32 values of a tensor payload (a network tensor, or an explanation's exact tensor), or
// null when the payload carries none (deferred, withheld).
export const tensorValues = (t) => (t && typeof t === "object" ? float32Of(t) : null);

// ---- reading network-view@2
function check(cond, key, params) { if (!cond) throw new NetworkViewError(key, params); }
const quoted = (v) => data(JSON.stringify(v === undefined ? null : v));

function validate(p) {
  check(p && typeof p === "object", "net.error.not_object");
  check(p.schema === VIEW_SCHEMA, "net.error.schema", { want: data(VIEW_SCHEMA), found: quoted(p.schema) });
  check(p.available === true, "net.error.unavailable");
  for (const k of REQUIRED) check(has(p, k), "net.error.field", { field: data(k) });
  for (const k of TASK_REQUIRED) check(has(p.task, k), "net.error.task_field", { field: data(k) });
  check(LINKS.includes(p.task.link), "net.error.link", { link: quoted(p.task.link) });
  check(Array.isArray(p.nodes) && p.nodes.length > 0, "net.error.no_nodes");
  const ids = new Set();
  for (const n of p.nodes) {
    check(typeof n.id === "string" && !ids.has(n.id), "net.error.node_id", { id: quoted(n.id) });
    check(has(KIND_OF, n.type) || SYNTHETIC.includes(n.type), "net.error.node_type", { id: data(n.id), type: quoted(n.type) });
    ids.add(n.id);
  }
  for (const n of p.nodes) {
    check(n.parent === null || ids.has(n.parent), "net.error.node_parent", { id: data(n.id), parent: data(n.parent) });
    for (const c of n.children || []) check(ids.has(c), "net.error.node_child", { id: data(n.id), child: data(c) });
    for (const ref of [n.params && n.params.weight, n.params && n.params.bias]) {
      check(typeof ref !== "string" || has(p.tensors, ref), "net.error.node_tensor", { id: data(n.id), tensor: data(ref) });
    }
  }
  for (const e of p.edges) check(ids.has(e.from) && ids.has(e.to), "net.error.edge", { from: data(e.from), to: data(e.to) });
}

function nodeOf(n) {
  const params = n.params || {}, act = params.activation;
  return Object.freeze({
    id: n.id, type: n.type, kind: kindOf(n.type) || n.type, view: n.view, parent: n.parent, children: Object.freeze((n.children || []).slice()),
    role: n.role || null, path: n.path || "", in_shape: n.in_shape, out_shape: n.out_shape, per: n.per || "sample",
    config: n.config || {}, params: params.count || 0, payload: params, formula: n.formula || "", code: n.code || null,
    activation: act && act !== "none" ? act : null, dropout: 0,
    weight: typeof params.weight === "string" ? params.weight : null, bias: typeof params.bias === "string" ? params.bias : params.bias || null,
  });
}

function inputsOf(p) {
  const i = p.inputs, labels = i.axis_labels || {};
  return Object.freeze({
    names: (i.order || [...i.numeric, ...i.nominal.map((c) => c.name)]).slice(), numeric: i.numeric.slice(),
    nominal: i.nominal.map((c) => ({ ...c })), width: i.width, pastTarget: i.past_target || null, edges: i.edges || null,
    features: labels.feature || null, steps: labels.step || null, tokens: labels.token || null,
  });
}

export class StageGraph {
  // model: {family, task, sample, inputs, params, nodes (normalised, document order), edges, chain, tensors}
  constructor(model) {
    Object.assign(this, { model, byIdMap: new Map(model.nodes.map((n) => [n.id, n])), cache: new Map(), segs: null });
    this.dropouts();
  }

  static from(payload) {
    validate(payload);
    return new StageGraph({
      family: { ...payload.family }, task: { ...payload.task }, sample: { ...payload.sample }, inputs: inputsOf(payload),
      params: { ...payload.params }, nodes: payload.nodes.map(nodeOf), edges: payload.edges.slice(), chain: payload.chain,
      tensors: { ...payload.tensors },
    });
  }

  // A dropout block's rate, on the stage itself (the renderers read stage.dropout)
  dropouts() {
    for (const n of this.model.nodes) if (n.type === "dropout" || n.type === "edge_dropout") {
      this.byIdMap.set(n.id, Object.freeze({ ...n, dropout: Number(n.payload.p ?? n.config.p ?? 0) }));
    }
  }

  get task() { return this.model.task; }
  get sample() { return this.model.sample; }
  get family() { return this.model.family.name; }
  get params() { return this.model.params; }
  get inputs() { return this.model.inputs; }
  get chain() { return this.model.chain || null; }
  get link() { return this.model.task.link; }
  get viewKey() { const t = this.model.task; return VIEW_KEY_OF[t.base || t.kind] || VIEW_KEY_OF[t.kind] || null; }

  nodes() { return this.model.nodes.map((n) => this.byIdMap.get(n.id)); }
  stages() { return this.nodes().filter((n) => has(KIND_OF, n.type)); }
  byId(id) { return this.byIdMap.get(id) || null; }
  childrenOf(id) { const n = this.byId(id); return n ? n.children.map((c) => this.byId(c)) : []; }
  // The top-level stages in forward order: by their module path (stages.<i>) when every one has
  // it, else in the payload's order.
  topLevel() {
    const top = this.stages().filter((n) => n.parent === null), at = (n) => /^stages\.(\d+)$/.exec(n.path || "");
    return (top.every(at) ? top.slice().sort((a, b) => Number(at(a)[1]) - Number(at(b)[1])) : top).map((n) => n.id);
  }
  head() { const h = this.stages().filter((n) => n.type === "head"); return h.length ? h[h.length - 1] : null; }
  edges() { return this.model.edges.slice(); }

  // Every id under a node (its children, theirs, …), the node itself first.
  descendants(id) {
    const out = [id];
    for (const c of this.byId(id) ? this.byId(id).children : []) out.push(...this.descendants(c));
    return out;
  }

  // The stage kinds present, with input and head: the renderers a drawing needs (a pure dense chain
  // is drawn whole by the dense renderer, a head alone included).
  kinds() {
    const out = new Set(["input", "head", ...this.stages().map((n) => n.kind)]);
    if (this.isPureDenseChain()) out.add("dense");
    return out;
  }

  // ---- weights
  tensor(id) { return has(this.model.tensors, id) ? this.model.tensors[id] : null; }
  vector(ref) {
    if (Array.isArray(ref)) return Float32Array.from(ref);
    if (typeof ref !== "string") return null;
    if (!this.cache.has(ref)) { const t = this.tensor(ref); this.cache.set(ref, t ? float32Of(t) : null); }
    return this.cache.get(ref);
  }
  // {out, in, w: Float32Array row-major} of a weight tensor, or null when it is not in the payload.
  matrix(ref) {
    const t = this.tensor(ref), w = this.vector(ref);
    return t && w ? { out: t.shape[0], in: t.shape.slice(1).reduce((a, b) => a * b, 1), w } : null;
  }
  // Whether the service publishes summaries of its weights only (weights mode "summary").
  weightsWithheld() { return Object.values(this.model.tensors).some((t) => t.encoding === "withheld"); }
  // The tensors the payload deferred (past inline_total_max, D3 §4: fetched from
  // v1/network/tensor/{id}) and a way to add one: a tensor payload of the same id and shape that
  // carries its values.
  missingTensors() { return Object.values(this.model.tensors).filter((t) => t.encoding === "deferred").map((t) => t.id); }
  attachTensor(t) {
    check(t && has(this.model.tensors, t.id), "net.error.tensor", { tensor: data(t && t.id) });
    const own = this.model.tensors[t.id], same = Array.isArray(t.shape) && t.shape.length === own.shape.length && t.shape.every((n, i) => n === own.shape[i]);
    check(same && carried(t), "net.error.tensor_payload", { tensor: data(t.id) });
    this.model.tensors[t.id] = t; this.cache.delete(t.id);
  }
  // The tensors the drawings and the weights read decode: every dense-like stage's weight and bias,
  // and the one-class centre of the head.
  drawnTensors() {
    const ids = [];
    for (const n of this.stages()) if (MATRIX_KINDS.includes(n.kind)) ids.push(...[n.weight, n.bias].filter((r) => typeof r === "string"));
    const h = this.head();
    if (h && this.tensor(`${h.id}.centre`)) ids.push(`${h.id}.centre`);
    return ids.filter((id) => this.tensor(id));
  }
  // Those of drawnTensors() the payload does not carry (deferred, or withheld).
  drawnTensorsMissing() { return this.drawnTensors().filter((id) => !carried(this.tensor(id))); }
  // Fetches the deferred tensors the drawings read, each through get(id) (a promise of the tensor
  // payload: the caller's GET of v1/network/tensor/{id}), and attaches them. Resolves (never rejects)
  // to {fetched: [ids], failed: [{id, error}]} once every fetch has settled.
  async fetchDeferred(get) {
    const ids = this.drawnTensors().filter((id) => this.tensor(id).encoding === "deferred");
    const settled = await Promise.allSettled(ids.map((id) => Promise.resolve().then(() => get(id)).then((t) => this.attachTensor(t))));
    const failed = settled.map((s, i) => (s.status === "rejected" ? { id: ids[i], error: String(s.reason && s.reason.message ? s.reason.message : s.reason) } : null)).filter(Boolean);
    return { fetched: ids.filter((id) => !failed.some((f) => f.id === id)), failed };
  }

  // ---- the drawing's structure (segments.js)
  segments() { return this.segs || (this.segs = segmentsOf(this)); }
  denseRuns() { return runsOf(this.segments()); }
  isPureDenseChain() { return isPure(this); }
  runModel(seg, opts) { return runModel(this, seg, opts); }
  headLabels() { return headLabels(this); }
  ordinalClasses() { return ordinalClasses(this); }
  rebuiltInputOf() { return rebuiltInputOf(this); }

  // The step kinds the stepper needs an explainer for (D4 §8.1): the input step, the kinds that
  // compute an answer, a residual group's sum ("residual") and a parallel group's merge ("merge");
  // the link step is the lane's.
  stepKinds() {
    const out = new Set(["input"]);
    const walk = (segs) => segs.forEach((s) => {
      if (s.kind === "run") out.add("dense");
      else if (s.kind === "head") out.add("head");
      else if (s.kind === "box" && STEP_KINDS.includes(this.byId(s.stage).kind)) out.add(this.byId(s.stage).kind);
      else if (s.kind === "group") {
        if (GROUP_STEP[s.mode]) out.add(GROUP_STEP[s.mode]);
        (s.repeats || s.branches || []).forEach((b) => walk(b.segments || b));
      }
    });
    walk(this.segments());
    return out;
  }

  // Unit indices of every input feature (inputs.names order) in the columns of the first dense layer
  // that reads the inputs: a numeric column is one unit, a nominal column its embedding's dims.
  featureUnits() {
    const i = this.inputs, n = i.numeric.length;
    let at = n;
    return i.names.map((name) => {
      const k = i.numeric.indexOf(name);
      if (k >= 0) return [k];
      const c = i.nominal.find((x) => x.name === name), dim = c && c.dim ? c.dim : 0;
      const units = Array.from({ length: dim }, (_, d) => at + d);
      at += dim;
      return units;
    });
  }

  // What a decision's focus lights up (focus@1, D4 §15.4; a story@1 region word is read as today's
  // page read it): stage ids, "input", "head", "edges", "feature:<name>", "<id>#act|#drop|#norm"
  // for a stage fused into a dense stage, and "*" when everything is. null: no focus.
  focusSet(focus) {
    if (focus === null || focus === undefined) return null;
    const all = () => new Set(["*", "input", "head", "edges", ...this.stages().map((n) => n.id), ...fusedMarkers(this, () => true).map((m) => m.marker)]);
    const headIds = () => ["head", ...(this.head() ? [this.head().id] : [])];
    const hidden = () => this.denseRuns().flatMap((r) => r.stages);
    if (typeof focus === "string") {
      if (focus.startsWith("feature:")) return new Set(["input", focus]);
      const words = {
        all, inputs: () => new Set(["input"]), outputs: () => new Set(headIds()), edges: () => new Set(["edges"]),
        hidden: () => new Set(hidden()),
        activation: () => new Set([...fusedMarkers(this, (k) => k === "activation").map((m) => m.marker),
          ...this.stages().filter((n) => n.kind === "activation").map((n) => n.id)]),
      };
      return has(words, focus) ? words[focus]() : null;
    }
    return this.focusOf(focus, { all, headIds });
  }

  focusOf(focus, { all, headIds }) {
    const k = focus && focus.kind, ref = focus && focus.ref;
    if (k === "all") return all();
    if (k === "feature") return new Set(["input", `feature:${ref}`]);
    if (k === "block") {
      if (!this.byId(ref)) return new Set();
      const ids = this.descendants(ref), out = new Set(ids);
      fusedMarkers(this, () => true).filter((m) => ids.includes(m.fused)).forEach((m) => out.add(m.marker));
      if (this.head() && ids.includes(this.head().id)) out.add("head");
      return out;
    }
    if (k !== "family") return null;
    if (ref === "objective") return new Set(headIds());
    if (ref === "optimization") return new Set(["edges"]);
    if (ref === "architecture") return new Set(["input", ...headIds(), ...this.stages().map((n) => n.id)]);
    if (ref === "regularization") {
      const kinds = ["dropout", "norm", "edge_dropout"];
      return new Set([...this.stages().filter((n) => kinds.includes(n.kind)).map((n) => n.id),
        ...fusedMarkers(this, (kind) => kinds.includes(kind)).map((m) => m.marker)]);
    }
    return null;
  }
}

// A focus set read by the drawings: is a column (a stage id, "input" or "head") lit, is a connection
// between two columns lit, and which feature (if any) the focus names.
export class Focus {
  constructor(set) { this.set = set; }
  get active() { return !!this.set; }
  get all() { return !!this.set && this.set.has("*"); }
  get feature() { const f = this.set ? [...this.set].find((m) => m.startsWith("feature:")) : null; return f ? f.slice(8) : null; }
  node(id, fused = []) {
    const s = this.set;
    if (!s || s.has("*") || s.has("edges")) return true;
    return s.has(id) || fused.some((m) => s.has(m)) || [`${id}#act`, `${id}#drop`, `${id}#norm`].some((m) => s.has(m));
  }
  // src, dst: column ids ("input", a stage id, "head" for the output column)
  edge(src, dst) {
    const s = this.set;
    if (!s || s.has("*") || s.has("edges")) return true;
    return s.has(dst) && (s.has(src) || dst === "head");
  }
}
