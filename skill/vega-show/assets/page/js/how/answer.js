// One answer as the stepper reads it: the explanation of one sample (POST v1/explain), whichever
// form the service sends, over the stage graph that names its stages.
//
//   vega/explanation@2 (D3, inspect-contract §5.4): nodes[id] with the values each stage read and
//     passed on — exact tensors (base64 little-endian float32, int64 for ids) under detail "exact"
//     (R-F5), reduced forms under "full" (the probe keys of a block, pre / post / grad_pre of a
//     dense stage, under `extra`), nothing under "summary" / "omitted" — and, for a network whose
//     dense chain is eligible, chain {inputs, pre, post, grad_pre}; outputs {kind, names, link, raw,
//     read, answer}; target {kind, index, name, sign, value}; inputs {order, shaped, raw};
//     attribution (the row's features.values: each input's shaped value or nominal id).
//   the v2.2-merge answer a bundle packed before v2.3 serves (R7): its dense chain only — inputs,
//     pre and activations per layer, grad_pre, probabilities by class, the label — read through
//     the chain StageGraph.fromLegacy builds (its layers name the stages).
//
// values(id, field) gives one stage's numbers for the explained sample (field: in, out, pre, post,
// grad_pre, grad_out, or a probe key of `extra`), flat and row-major, or null when the answer does
// not carry them all; unitRelevance(id) what each unit of a stage's output carried of the explained
// target (the hand-built page's reading on a dense chain, net/relevance.js; out ⊙ grad_out, or the
// node's relevance, anywhere else).
import { readAnswer, relevance } from "../net/relevance.js";
import { unitMatrix } from "../net/segments.js";

export const EXPLANATION_SCHEMA = "vega/explanation@2";

const has = (o, k) => !!o && Object.prototype.hasOwnProperty.call(o, k);
// A dense stage's output is its post-activation, and the other way round.
const FALLBACK = Object.freeze({ post: "out", out: "post" });
const CHAIN_KEY = "\u0000chain";

function bytesOf(b64) {
  if (typeof atob !== "function") return new Uint8Array(Buffer.from(b64, "base64"));
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// The numbers of an exact tensor ({shape, axes, dtype, encoding: base64-le, data}) as a plain
// array, row-major (int64 ids as numbers), or null for anything else.
export function exactValues(t) {
  if (!t || typeof t !== "object" || t.encoding !== "base64-le" || typeof t.data !== "string") return null;
  const b = bytesOf(t.data), view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (t.dtype === "int64") return Array.from({ length: b.byteLength / 8 }, (_, i) => Number(view.getBigInt64(8 * i, true)));
  return Array.from({ length: b.byteLength / 4 }, (_, i) => view.getFloat32(4 * i, true));
}

// The numbers of a reduced form when it holds all of them (a scalar, a full vector, a plain list);
// null otherwise (the largest entries only, a summary, a heat map).
export function reducedValues(r) {
  if (Array.isArray(r) && r.every((v) => typeof v === "number")) return r.slice();
  if (typeof r === "number" && Number.isFinite(r)) return [r];
  if (!r || typeof r !== "object") return null;
  if (r.form === "scalar" && Number.isFinite(r.value)) return [r.value];
  if (r.form === "vector" && Array.isArray(r.values) && r.values.length === r.n) return r.values.slice();
  return null;
}

const argmax = (xs) => xs.reduce((b, v, i) => (v > xs[b] ? i : b), 0);

export class Answer {
  // ex: the explanation (either form); graph: its StageGraph; row: the raw row asked about
  constructor(ex, graph, row = null) {
    Object.assign(this, { ex, graph, row, cache: new Map() });
    this.v2 = !!ex && ex.schema === EXPLANATION_SCHEMA;
    const chain = graph && graph.chain ? graph.chain.layers || [] : [];
    this.chainAt = new Map(chain.map((l, i) => [l.node, i]));
  }

  // An Answer of an explanation (an Answer is returned as it is).
  static of(ex, graph, row = null) { return ex instanceof Answer ? ex : new Answer(ex, graph, row); }

  get legacy() { return !this.v2; }
  // Past the exact budget the service summarised some stages (R-F5): their steps say so.
  get degraded() { return this.v2 && !!this.ex.degraded; }

  // The chain arrays of the answer, when it carries the dense chain: {inputs, pre, post, grad_pre}.
  chain() {
    const ex = this.ex;
    if (!ex) return null;
    if (this.v2) return ex.chain || null;
    return ex.activations && ex.pre ? { inputs: ex.inputs, pre: ex.pre, post: ex.activations, grad_pre: ex.grad_pre || null } : null;
  }

  node(id) { return this.v2 && this.ex.nodes && has(this.ex.nodes, id) ? this.ex.nodes[id] : null; }

  // The payload level of a stage: exact | full | summary | omitted, "chain" for a chain layer, or
  // null when the answer says nothing of it.
  level(id) {
    const n = this.node(id);
    if (n) return n.level;
    return this.chainAt.has(id) && this.chain() ? "chain" : null;
  }

  values(id, field) {
    const key = `${id}\u0000${field}`;
    if (!this.cache.has(key)) this.cache.set(key, this.read(id, field) ?? (has(FALLBACK, field) ? this.read(id, FALLBACK[field]) : null));
    return this.cache.get(key);
  }

  read(id, field) {
    const n = this.node(id);
    if (n && n.exact && n.exact[field]) return exactValues(n.exact[field]);
    const c = this.chain(), l = this.chainAt.get(id);
    if (c && l !== undefined) {
      const at = { in: l === 0 ? c.inputs : c.post[l - 1], pre: c.pre[l], post: c.post[l], grad_pre: c.grad_pre ? c.grad_pre[l] : null }[field];
      if (Array.isArray(at)) return at.slice();
    }
    if (!n) return null;
    if (has(n, field)) return reducedValues(n[field]);
    return n.extra && has(n.extra, field) ? reducedValues(n.extra[field]) : null;
  }

  // The shaped inputs: {names, x (the numbers, flat), grid (a window's [step][feature]) , raw}.
  inputs() {
    const ex = this.ex || {};
    if (!this.v2) return { names: this.graph.inputs.names.slice(), x: (ex.inputs || []).slice(), grid: null, raw: this.row };
    const i = ex.inputs || {}, shaped = i.shaped;
    const grid = Array.isArray(shaped) && Array.isArray(shaped[0]) ? shaped : null;
    const x = grid ? grid.flat() : Array.isArray(shaped) ? shaped.slice() : reducedValues(shaped);
    return { names: (i.order || this.graph.inputs.names).slice(), x, grid, raw: i.raw || this.row };
  }

  // Each network input's value as the network received it, in inputs.order: the shaped number of a
  // numeric input, the id of a nominal one (the row's attribution carries both), else the shaped
  // vector when it holds as many; null when neither does.
  inputValues() {
    const names = this.inputs().names, f = this.v2 && this.ex.attribution && this.ex.attribution.features;
    if (f && Array.isArray(f.values) && f.values.length === names.length) return f.values.slice();
    const x = this.inputs().x;
    return x && x.length === names.length ? x.slice() : null;
  }

  // The head's readout: {kind, names, link, raw (the head's numbers), read (after the link), answer}.
  output() {
    const ex = this.ex || {};
    if (this.v2 && ex.outputs) return { ...ex.outputs, read: ex.outputs.read || null, answer: ex.outputs.answer || {} };
    const names = this.graph.task.outputs.names.slice(), c = this.chain();
    const raw = c ? c.pre[c.pre.length - 1].slice() : [];
    const read = ex.probabilities ? names.map((k) => ex.probabilities[k]) : null;
    return { kind: this.graph.task.outputs.kind, names, link: this.graph.link, raw, read, answer: { class: ex.label ?? null } };
  }

  // What was explained: {kind, index, name, sign, value}. The v2.2 answer explained its class.
  target() {
    const ex = this.ex || {};
    if (this.v2 && ex.target) return ex.target;
    const out = this.output(), index = out.names.indexOf(ex.label), at = index >= 0 ? index : argmax(out.raw);
    return { kind: "logit", index: at, name: ex.label ?? out.names[at] ?? null, sign: 1, value: out.raw[at] ?? null };
  }

  // What each unit of a stage's output carried of the target, or null. A dense chain layer reads it
  // as the hand-built page did (net/relevance.js); any other stage as out ⊙ grad_out, else the
  // node's relevance, from the stage whose output the unit's value is (`from`: the last stage fused
  // into a dense stage) or the stage itself.
  unitRelevance(id, from = id) {
    const l = this.chainAt.get(id), c = this.chain();
    if (c && l !== undefined && l < this.chainAt.size - 1) {
      const R = this.chainReading();
      if (R && R.hidden[l]) return R.hidden[l].slice();
    }
    for (const at of [...new Set([from, id])]) {
      const out = this.values(at, "out"), g = this.values(at, "grad_out");
      if (out && g && out.length === g.length) return out.map((v, j) => v * g[j]);
      const r = reducedValues(this.node(at) && this.node(at).relevance);
      if (r) return r;
    }
    return null;
  }

  // net/relevance.js's reading of the dense chain (computed once).
  chainReading() {
    if (this.cache.has(CHAIN_KEY)) return this.cache.get(CHAIN_KEY);
    let R = null;
    const layers = this.graph.chain ? this.graph.chain.layers : [];
    const mats = layers.map((l) => unitMatrix(this.graph, this.graph.byId(l.node)));
    const answer = this.chain() ? readAnswer(this.ex, this.output().names) : null;
    if (answer && mats.length && mats.every(Boolean)) R = relevance(answer, mats);
    this.cache.set(CHAIN_KEY, R);
    return R;
  }
}
