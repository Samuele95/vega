// What carried one answer through a dense run (the hand-built page's relevance, now reading its
// target from the explanation). With g = the gradient of the explained target with respect to a
// layer's pre-activation z, a connection i → j of layer l carried  w[j][i] · source[i] · g[l][j]
// (positive: towards the target), and a hidden unit carried  a[j] · Σ_o w[o][j] · g[o]  of the next
// layer. The inputs carry their gradient × input: the "what tipped it" bars' numbers.
//
// readAnswer() turns an explanation into the vectors this needs, from either form the service
// sends: vega/explanation@2's dense chain (chain.inputs, pre, post, grad_pre; the target's index) or
// the v2.2-merge answer (inputs, activations, pre, grad_pre; the answer's class), whose last
// gradient the hand-built page set to the answer's one-hot, as it still is here.

export const EXPLANATION_SCHEMA = "vega/explanation@2";

export const signedNorm = (arr) => { const m = Math.max(1e-12, ...arr.map(Math.abs)); return arr.map((v) => v / m); };

const argmax = (xs) => xs.reduce((b, v, i) => (v > xs[b] ? i : b), 0);
const oneHot = (n, k) => Array.from({ length: n }, (_, c) => (c === k ? 1 : 0));

function fromV2(ex, outputs) {
  const c = ex.chain;
  if (!c) return null;
  // the target's unit; a target that names no unit of the head (the expected rank, a binary class's
  // index on a one-score head) points at the largest score
  const raw = ex.outputs ? ex.outputs.raw : [], index = ex.target && Number.isInteger(ex.target.index) ? ex.target.index : -1;
  const win = index >= 0 && index < outputs.length ? index : argmax(raw);
  const read = ex.outputs && Array.isArray(ex.outputs.read) ? ex.outputs.read : raw;
  const attribution = ex.attribution && ex.attribution.shape === "row" ? ex.attribution.features.attribution : null;
  return { inputs: c.inputs, pre: c.pre, post: c.post, grad_pre: c.grad_pre, win, probs: outputs.map((_, k) => read[k]), attribution };
}

function fromLegacy(ex, outputs) {
  if (!ex.activations || !ex.inputs) return null;
  const win = Math.max(0, outputs.indexOf(ex.label)), L = ex.activations.length;
  const grad = ex.grad_pre ? ex.grad_pre.map((g, l) => (l === L - 1 ? oneHot(outputs.length, win) : g)) : null;
  return { inputs: ex.inputs, pre: ex.pre, post: ex.activations, grad_pre: grad, win,
           probs: outputs.map((c) => ex.probabilities[c]), attribution: ex.attribution || null };
}

// {inputs, pre, post, grad_pre, win, probs, attribution} of an explanation, or null when it carries
// no dense chain. outputs: the head's labels in unit order.
export function readAnswer(ex, outputs) {
  if (!ex) return null;
  return ex.schema === EXPLANATION_SCHEMA ? fromV2(ex, outputs) : fromLegacy(ex, outputs);
}

// The relevance of one answer through the matrices `mats` ({out, in, w}, the run's layers then the
// head): {L, win, probs, src, gpre, hidden, edge(l, i, j), inputs}.
export function relevance(answer, mats) {
  const L = mats.length, win = answer.win;
  const src = [answer.inputs, ...answer.post.slice(0, L - 1)];
  const gpre = answer.grad_pre || mats.map((m, l) => (l === L - 1 ? oneHot(m.out, win) : new Array(m.out).fill(1)));
  const hidden = [];
  for (let h = 1; h < L; h++) {
    const m = mats[h], g = gpre[h], r = new Array(m.in).fill(0);
    for (let j = 0; j < m.in; j++) {
      let t = 0;
      for (let o = 0; o < m.out; o++) t += m.w[o * m.in + j] * g[o];
      r[j] = src[h][j] * t;
    }
    hidden.push(r);
  }
  const edge = (l, i, j) => mats[l].w[j * mats[l].in + i] * src[l][i] * gpre[l][j];
  return { L, win, probs: answer.probs, src, gpre, hidden, edge, inputs: answer.attribution };
}

// The strongest connections of every layer in one answer (the "every weight" drawing of it): the
// top `k` by |carried| (`last` into the answer only), each [i, j, carried].
export function strongest(R, mats, { k = 90, kLast = 40 } = {}) {
  return mats.map((m, l) => {
    const cand = [], last = l === mats.length - 1;
    for (let j = 0; j < m.out; j++) {
      if (last && j !== R.win) continue;
      for (let i = 0; i < m.in; i++) { const c = R.edge(l, i, j); if (c !== 0) cand.push([Math.abs(c), i, j, c]); }
    }
    cand.sort((p, q) => q[0] - p[0]);
    return cand.slice(0, last ? kLast : k).map(([, i, j, c]) => [i, j, c]);
  });
}

// One stage's output for the explained sample, from an explanation@2 (null when it carries none):
// the exact tensor when detail was exact, else the reduced vector (every value, or the largest ones
// with the rest left at zero), else a scalar summary as one value.
export function stageValues(ex, id, decode) {
  const node = ex && ex.schema === EXPLANATION_SCHEMA && ex.nodes ? ex.nodes[id] : null;
  if (!node || node.level === "omitted") return null;
  if (node.exact && node.exact.post) return Array.from(decode(node.exact.post));
  if (node.exact && node.exact.out) return Array.from(decode(node.exact.out));
  const out = node.out;
  if (!out) return null;
  if (out.form === "vector" && Array.isArray(out.values)) return out.values.slice();
  if (out.form === "vector" && Array.isArray(out.top)) { const v = new Array(out.n).fill(0); out.top.forEach(([i, x]) => { v[i] = x; }); return v; }
  if (out.form === "scalar") return [out.value];
  return null;
}

// The l2 norm of what one stage read (side "in") or passed on (side "out") for the explained sample,
// from an explanation@2 (null when it carries neither): the exact tensor, else the reduced form's
// values, its summary, a scalar, or a sequence's per-step norms.
export function stageNorm(ex, id, side, decode) {
  const node = ex && ex.schema === EXPLANATION_SCHEMA && ex.nodes ? ex.nodes[id] : null;
  if (!node || node.level === "omitted") return null;
  const l2 = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  if (node.exact && node.exact[side]) { const v = decode(node.exact[side]); return v ? l2(Array.from(v)) : null; }
  const r = node[side];
  if (!r) return null;
  if (r.form === "scalar" && Number.isFinite(r.value)) return Math.abs(r.value);
  if (r.form === "vector" && Array.isArray(r.values)) return l2(r.values);
  if (r.form === "vector" && r.summary && Number.isFinite(r.summary.l2)) return r.summary.l2;
  if (r.form === "sequence" && r.per_step && Array.isArray(r.per_step.norm)) return l2(r.per_step.norm);
  return null;
}

// The inputs that carried the most (by |gradient × input|), at most n.
export const topInputs = (R, n = 6) => (R.inputs || []).map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]).slice(0, n).map((t) => t[1]);
