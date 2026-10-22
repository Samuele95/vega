// The stage graph cut into what the canvas lays out left to right (D4 §7.3):
//
//   {kind: "input"}
//   {kind: "run", stages: [dense ids], layers: [{dense, fused, activation, dropout, norm, group}], input: [ids]}
//   {kind: "box", stage: id}
//   {kind: "group", group: id, mode: "residual", repeats: [{segments, projection, add}]}
//   {kind: "group", group: id, mode: "parallel", branches: [{segments}], merge, node}
//   {kind: "head", stage: id}
//
// A dense run is a maximal sequence of dense stages; an activation, a dropout or a normalisation
// right after a dense stage is fused into it (its column carries the bend, the rate, the tick), an
// embedding and a concat right before a run are fused into the run's input side, and a sequential
// block or a decoder is laid out inline (its dense layers join the run; a decoder's keep its id in
// `group`, for the caption). Anything else is a box of its own; a residual or a parallel block is a
// group whose inside is cut the same way.
import { FUSED_KINDS, INPUT_SIDE_KINDS, MATRIX_KINDS } from "./kinds.js";

const INLINE = Object.freeze(["sequential", "decoder"]);
const PURE_KINDS = Object.freeze(["dense", "activation", "dropout", "head"]);
const MARK = Object.freeze({ activation: "act", dropout: "drop", norm: "norm" });

// The ids of a sequence with its inline composites opened, each with the composite that held it.
function flatten(graph, ids, holder = null) {
  const out = [];
  for (const id of ids) {
    const n = graph.byId(id);
    if (n && INLINE.includes(n.type)) out.push(...flatten(graph, n.children, n.type === "decoder" ? id : holder));
    else if (n) out.push({ id, holder });
  }
  return out;
}

// An embedding or a concat feeds a run's input side when the next stage past its likes is dense.
function feedsRun(graph, items, i) {
  let j = i;
  while (j < items.length && INPUT_SIDE_KINDS.includes(graph.byId(items[j].id).kind)) j++;
  return j < items.length && graph.byId(items[j].id).kind === "dense";
}

function fuse(layer, n) {
  layer.fused.push(n.id);
  if (n.kind === "activation") layer.activation = n.payload.name && n.payload.name !== "none" ? n.payload.name : (n.config.name || layer.activation);
  else if (n.kind === "dropout") layer.dropout = n.dropout;
  else layer.norm = n.id;
}

// One sequence of stage ids, cut into segments.
function sequence(graph, ids) {
  const items = flatten(graph, ids), out = [];
  let run = null, pending = [];
  items.forEach(({ id, holder }, i) => {
    const n = graph.byId(id), last = out[out.length - 1];
    if (n.kind === "dense") {
      if (!run) { run = { kind: "run", stages: [], layers: [], input: pending }; pending = []; out.push(run); }
      run.stages.push(id);
      run.layers.push({ dense: id, fused: [], activation: n.activation, dropout: n.dropout, norm: null, group: holder });
      return;
    }
    if (FUSED_KINDS.includes(n.kind) && run && last === run) { fuse(run.layers[run.layers.length - 1], n); return; }
    if (INPUT_SIDE_KINDS.includes(n.kind) && feedsRun(graph, items, i)) { pending.push(id); return; }
    run = null;
    if (n.kind === "head") out.push({ kind: "head", stage: id });
    else if (n.kind === "residual") out.push(residual(graph, n));
    else if (n.kind === "parallel") out.push(parallel(graph, n));
    else out.push({ kind: "box", stage: id });
  });
  return out;
}

function residual(graph, n) {
  const kids = n.children.map((c) => graph.byId(c));
  const declared = Array.isArray(n.payload.repeats) ? n.payload.repeats : null;
  const count = declared ? declared.length : Math.max(1, Number(n.config.repeat) || 1);
  const repeats = Array.from({ length: count }, (_, r) => {
    const mine = count === 1 ? kids : kids.filter((k) => k.id.startsWith(`${n.id}.r${r}.`));
    const inner = declared ? declared[r].inner : mine.filter((k) => k.type !== "residual_add" && k.role !== "projection").map((k) => k.id);
    const proj = declared ? declared[r].projection : (mine.find((k) => k.role === "projection") || {}).id || null;
    const add = (mine.find((k) => k.type === "residual_add") || {}).id || null;
    return { segments: sequence(graph, inner), projection: proj || null, add };
  });
  return { kind: "group", group: n.id, mode: "residual", repeats };
}

function parallel(graph, n) {
  const kids = n.children.map((c) => graph.byId(c));
  const branches = (Array.isArray(n.payload.branches) ? n.payload.branches : kids.filter((k) => k.type !== "parallel_merge").map((k) => k.id));
  const node = (kids.find((k) => k.type === "parallel_merge") || {}).id || null;
  return {
    kind: "group", group: n.id, mode: "parallel", branches: branches.map((b) => ({ segments: sequence(graph, [b]) })),
    merge: n.payload.merge || n.config.merge || "concat", node,
  };
}

export function segmentsOf(graph) {
  return Object.freeze([{ kind: "input" }, ...sequence(graph, graph.topLevel())]);
}

// The parts of a group, each a list of segments: its repeats (residual), its branches (parallel), or
// its inside (sequential, when a caller did not lay it out inline).
export function groupParts(seg) {
  if (seg.mode === "parallel") return seg.branches.map((b) => b.segments);
  if (seg.mode === "residual") return seg.repeats.map((r) => r.segments);
  return [seg.children || []];
}

// The segment at [part, index] of a group (groupParts order).
export const groupChild = (seg, at) => groupParts(seg)[at[0]][at[1]];

// The stage whose output a list of segments ends in: a run's last layer (the last stage fused into
// it, else its dense stage), a box's stage, a residual group's last add, a parallel group's merge.
export function lastStageOf(segments) {
  const s = segments[segments.length - 1];
  if (!s) return null;
  if (s.kind === "run") { const l = s.layers[s.layers.length - 1]; return l.fused.length ? l.fused[l.fused.length - 1] : l.dense; }
  if (s.kind === "box" || s.kind === "head") return s.stage;
  if (s.kind === "group" && s.mode === "residual") return s.repeats[s.repeats.length - 1].add || s.group;
  if (s.kind === "group") return s.node || s.group;
  return null;
}

// Every dense run, the ones inside groups included, in drawing order.
export function runsOf(segments) {
  const out = [];
  for (const s of segments) {
    if (s.kind === "run") out.push(s);
    else if (s.kind === "group") (s.repeats || s.branches).forEach((b) => out.push(...runsOf(b.segments)));
  }
  return out;
}

// The stages fused into dense stages, as focus markers "<dense>#act|#drop|#norm": a fused stage's
// own (fused: its id), and a dense stage's own activation and dropout (fused: the dense id). pred
// filters by the fused kind.
export function fusedMarkers(graph, pred) {
  const out = [];
  for (const run of graph.denseRuns()) for (const l of run.layers) {
    for (const f of l.fused) {
      const kind = graph.byId(f).kind;
      if (pred(kind)) out.push({ marker: `${l.dense}#${MARK[kind]}`, fused: f, kind });
    }
    const own = graph.byId(l.dense);
    if (own.activation && pred("activation")) out.push({ marker: `${l.dense}#act`, fused: l.dense, kind: "activation" });
    if (own.dropout > 0 && pred("dropout")) out.push({ marker: `${l.dense}#drop`, fused: l.dense, kind: "dropout" });
  }
  return out;
}

// input → (dense [activation] [dropout])* → head, on rows of numeric inputs: the chain today's
// drawing draws as it is (D4 §7.3).
export function isPure(graph) {
  if (graph.sample.shape !== "row" || graph.inputs.nominal.length) return false;
  if (!graph.topLevel().every((id) => PURE_KINDS.includes(graph.byId(id).kind))) return false;
  const segs = graph.segments().slice(1);
  if (!segs.length || segs[segs.length - 1].kind !== "head") return false;
  const body = segs.slice(0, -1);
  if (body.length > 1 || body.some((s) => s.kind !== "run" || s.input.length)) return false;
  // one bend at most per layer (D3 §5.4): the dense stage's own or one activation block after it
  const bends = (l) => l.fused.filter((f) => graph.byId(f).kind === "activation").length + (graph.byId(l.dense).activation ? 1 : 0);
  return body.every((run) => run.layers.every((l) => l.norm === null && bends(l) <= 1));
}

// The labels of the head's units: the task's output names when one per unit (network-view@2 names
// the units: the classes, the positive class of a score, the labels, the thresholds ">c" of an
// ordinal head, the steps ahead "t+h" or "t+h/q"), otherwise what each unit scores: a single score is
// the positive class's; a reconstruction head rebuilds the numeric inputs, then scores every value
// of each nominal input ("<column>[<k>]", recon_dim's order).
export function headLabels(graph) {
  const h = graph.head(), names = graph.task.outputs.names, out = h ? h.out_shape[h.out_shape.length - 1] : names.length;
  if (names.length === out) return names.slice();
  if (graph.link === "reconstruction") {
    const i = graph.inputs, units = [...i.numeric, ...i.nominal.flatMap((c) => Array.from({ length: c.cardinality }, (_, k) => `${c.name}[${k}]`))];
    if (units.length === out) return units;
  }
  if (out === 1) return [graph.task.positive || names[names.length - 1]];
  if (graph.link === "cumulative" && names.length === out + 1) return names.slice(1);
  return Array.from({ length: out }, (_, j) => names[j] ?? String(j));
}

// For each unit of a reconstruction head, the index of the input it rebuilds in
// task.outputs.names (the numeric inputs one unit each, a nominal input its cardinality's units).
export function rebuiltInputOf(graph) {
  const i = graph.inputs, names = graph.task.outputs.names, at = (n) => names.indexOf(n);
  return [...i.numeric.map(at), ...i.nominal.flatMap((c) => Array.from({ length: c.cardinality }, () => at(c.name)))];
}

// The classes of an ordinal head, low to high. network-view@2 names its k−1 thresholds ">c" by the
// class each is above (inspect-contract §4.5), so c_0 … c_{k−2} are read off them; the class above
// the last threshold has no name in the payload (null: the page calls it "above c_{k−2}").
export function ordinalClasses(graph) {
  return [...headLabels(graph).map((s) => String(s).replace(/^>/, "")), null];
}

// The names of the units a network's first dense stage reads: the dense chain's when the service
// sent one (D3 §5.4: numeric columns, then "<col>·e<j>" per embedding dim), else the numeric columns
// then "<col>[<j>]" per embedding dim, in featureUnits() order.
function inputUnitNames(graph, first) {
  const chain = graph.chain;
  if (chain && chain.layers[0] && first && chain.layers[0].node === first.id) return chain.inputs.slice();
  const i = graph.inputs;
  return [...i.numeric, ...i.nominal.flatMap((c) => Array.from({ length: c.dim || 0 }, (_, d) => `${c.name}[${d}]`))];
}

// The (out × in) matrix of a dense-like stage's units, or null when its weight is not in the payload:
// its weight, or, for the rank-consistent ordinal head (one shared score row w; unit j is w·x + θ_j,
// inspect-contract §3.1), that row once per threshold. Its bias: the bias tensor, else the ordinal
// head's thresholds θ.
export function unitMatrix(graph, stage) {
  const W = graph.matrix(stage.weight), out = stage.out_shape[stage.out_shape.length - 1];
  if (!W || W.out === out) return W;
  if (W.out !== 1) return null;
  const w = new Float32Array(out * W.in);
  for (let j = 0; j < out; j++) w.set(W.w, j * W.in);
  return { out, in: W.in, w };
}
export function unitBias(graph, stage) {
  const b = graph.vector(stage.bias), th = stage.payload && stage.payload.thresholds;
  return b || (Array.isArray(th) ? Float32Array.from(th) : null);
}

// What a dense drawing needs of a run (and, for the pure chain drawn whole, of the head after it:
// opts.head, by default whether the graph is a pure chain): the names of its input units (null when
// it reads another stage's units, not the network's inputs), every layer's widths, bend, rate and
// decoded weights, and the head's labels. A run drawn as a segment leaves the head to its own.
export function runModel(graph, seg, { head: withHead = graph.isPureDenseChain() } = {}) {
  const pure = graph.isPureDenseChain(), head = graph.head(), own = seg && seg.layers ? seg.layers : [];
  const ids = [...own.map((l) => l.dense), ...(withHead && head ? [head.id] : [])];
  const layers = ids.map((id, i) => {
    const n = graph.byId(id), l = own[i];
    return {
      id, kind: n.kind, in: n.in_shape[n.in_shape.length - 1], out: n.out_shape[n.out_shape.length - 1],
      activation: l ? l.activation : n.activation, dropout: l ? l.dropout : n.dropout, norm: l ? l.norm : null,
      group: l ? l.group : null, params: n.params, W: MATRIX_KINDS.includes(n.kind) ? unitMatrix(graph, n) : null,
      bias: unitBias(graph, n),
    };
  });
  const first = layers.length ? graph.byId(layers[0].id) : null, readsInputs = pure || graph.segments()[1] === seg;
  return { inputs: readsInputs ? inputUnitNames(graph, first) : null, layers, outputs: headLabels(graph), head: head ? head.id : null, pure,
           params: graph.params.total };
}
