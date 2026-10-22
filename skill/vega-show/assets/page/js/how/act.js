// The page's own arithmetic for "how the network computed it" (D4 §8.3): the activation functions
// by their catalog names (coverage.json activations: none, relu, gelu, silu, tanh, sigmoid,
// leaky_relu), erf, the eight links' readings of a head's raw numbers (D4 §5.2, inspect-contract
// §4.3: softmax, sigmoid, sigmoid_each, identity, cumulative, reconstruction, distance, horizon),
// and actPlot, the small curve a hidden step draws with the unit's own z on it.
//
// GELU is the erf form PyTorch uses by default (approximate="none"): 0.5·z·(1 + erf(z/√2)). JS has
// no Math.erf, so erf is Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), as the hand-built page
// had it; over [−8, 8] every function here stays within 1e-5 of torch.nn.functional
// (vega-show/scripts/tests/test_page_math.py). leaky_relu takes its slope from the catalog
// (explain-catalog activations.leaky_relu.params.slope), 0.01 by default, PyTorch's.
//
// Everything but actPlot is pure: node imports this module in the unit tests, and MATH is what the
// page's test hook exposes as window.__showcase.math (R-F1).

export const LEAKY_SLOPE = 0.01;

// Abramowitz & Stegun 7.1.26: erf(x) for any real x, |error| < 1.5e-7.
export function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return Math.sign(x) * (1 - poly * Math.exp(-x * x));
}

// σ(z) without overflow: exp of a negative number only.
export function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

// softmax(z) without overflow: every exponent shifted by the largest z.
export function softmax(z) {
  const zs = Array.from(z), top = Math.max(...zs), e = zs.map((v) => Math.exp(v - top));
  const total = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / total);
}

// log Σ exp(z), for the cross-entropy of a nominal column's logits.
export function logSumExp(z) {
  const zs = Array.from(z), top = Math.max(...zs);
  return top + Math.log(zs.reduce((s, v) => s + Math.exp(v - top), 0));
}

// The activations by catalog name; each takes z (and leaky_relu its slope).
export const ACT = Object.freeze({
  none: (z) => z,
  relu: (z) => (z > 0 ? z : 0),
  gelu: (z) => 0.5 * z * (1 + erf(z / Math.SQRT2)),
  silu: (z) => z * sigmoid(z),
  tanh: (z) => Math.tanh(z),
  sigmoid,
  leaky_relu: (z, slope = LEAKY_SLOPE) => (z > 0 ? z : slope * z),
});

// The activations whose bend silences a negative sum (the catalog's silences_negative: true):
// a unit below zero comes out near zero, "quiet".
export const SILENCES_NEGATIVE = Object.freeze(["relu", "gelu", "silu", "leaky_relu"]);

// The function of an activation name (a catalog name; null and "none" are the identity), with the
// catalog entry's parameters (leaky_relu's slope); null for a name the page does not know.
export function actOf(name, entry = null) {
  const key = name === null || name === undefined ? "none" : String(name);
  if (!Object.prototype.hasOwnProperty.call(ACT, key)) return null;
  if (key !== "leaky_relu") return ACT[key];
  const slope = entry && entry.params && Number.isFinite(entry.params.slope) ? entry.params.slope : LEAKY_SLOPE;
  return (z) => ACT.leaky_relu(z, slope);
}

// Whether an activation silences a negative sum: the catalog entry says so when the page has it.
export function silencesNegative(name, entry = null) {
  if (entry && typeof entry.silences_negative === "boolean") return entry.silences_negative;
  return SILENCES_NEGATIVE.includes(name);
}

// ---- the links (D4 §5.2): what each makes of the head's raw numbers z
// The reading after the link, as explanation@2 outputs.read states it: softmax; σ; σ per label;
// the number itself; σ per threshold, P(y > j); the rebuilt inputs x̂; none for a distance (its
// answer is the distance alone); one value per step ahead.
export const LINK_READ = Object.freeze({
  softmax: (z) => softmax(z),
  sigmoid: (z) => Array.from(z, sigmoid),
  sigmoid_each: (z) => Array.from(z, sigmoid),
  identity: (z) => Array.from(z),
  cumulative: (z) => Array.from(z, sigmoid),
  reconstruction: (z) => Array.from(z),
  distance: () => null,
  horizon: (z) => Array.from(z),
});
export const LINKS = Object.freeze(Object.keys(LINK_READ));

// The class probabilities of a rank-consistent ordinal head from its k−1 raw numbers:
// p₀ = 1 − σ(z₀), pⱼ = σ(zⱼ₋₁) − σ(zⱼ), p_(k−1) = σ(z_(k−2)).
export function ordinalProbs(z) {
  const s = Array.from(z, sigmoid), k = s.length + 1;
  return Array.from({ length: k }, (_, j) => (j === 0 ? 1 - s[0] : j === k - 1 ? s[k - 2] : s[j - 1] - s[j]));
}

// The class an ordinal answer lands on under the cumulative-count rule: the thresholds whose σ
// reaches ½; and the expected rank Σ σ(zⱼ).
export const ordinalRank = (z) => Array.from(z, sigmoid).filter((p) => p >= 0.5).length;
export const expectedRank = (z) => Array.from(z, sigmoid).reduce((a, p) => a + p, 0);

// The reconstruction error of one row (the loss's per-row form, inspect-contract §4.4): the mean
// squared error over the numeric columns plus the mean cross-entropy over the nominal ones. xhat:
// the head's numbers (the numeric columns first, then each nominal's `cardinality` logits); x: the
// numeric values then the nominal ids; cards: the nominal columns' cardinalities, in order. Returns
// {error, per (each column's term: squared error or cross-entropy), unitOf (each head unit's
// column)}, or null when the head does not lay the columns out that way.
export function reconstructionError(xhat, x, cards = []) {
  const num = x.length - cards.length, width = num + cards.reduce((a, c) => a + c, 0);
  if (num < 0 || xhat.length !== width && xhat.length !== num) return null;
  const per = [], unitOf = [];
  for (let f = 0; f < num; f++) { per.push((xhat[f] - x[f]) ** 2); unitOf.push(f); }
  let at = num;
  if (xhat.length === width) {
    cards.forEach((c, k) => {
      const slice = Array.from(xhat).slice(at, at + c), id = Math.round(x[num + k]);
      per.push(id >= 0 && id < c ? logSumExp(slice) - slice[id] : NaN);
      for (let u = 0; u < c; u++) unitOf.push(num + k);
      at += c;
    });
  }
  const mean = (xs) => (xs.length ? xs.reduce((a, v) => a + v, 0) / xs.length : 0);
  return { error: mean(per.slice(0, num)) + (xhat.length === width ? mean(per.slice(num)) : 0), per, unitOf };
}

// The one-class score: the squared distance of the network's point f from the centre c.
export const svddDistance = (f, c) => Array.from(f).reduce((a, v, j) => a + (v - c[j]) ** 2, 0);

// The page's pure math, as the test hook exposes it (window.__showcase.math).
export const MATH = Object.freeze({ erf, sigmoid, softmax, logSumExp, ACT, actOf, silencesNegative, LEAKY_SLOPE, LINK_READ, ordinalProbs,
  ordinalRank, expectedRank, reconstructionError, svddDistance });

// ---- the curve of one bend with this unit on it (the hand-built page's actPlot, 1826–1844)
const SVG_NS = "http://www.w3.org/2000/svg";

function svgNode(tag, attrs, text) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text) n.textContent = text;
  return n;
}

// The curve a = f(z) over ±R (R at least 4, past |z|), the axes, and the unit's point (z, a) with
// its drop line, coloured by the side of zero z fell on. words: {aria, axisZ, axisA} (strings of
// the table: the caller's).
export function actPlot(f, z, a, words) {
  const R = Math.max(4, Math.ceil(Math.abs(z) + 0.5)), W = 250, H = 132, p = 16;
  const xs = Array.from({ length: 161 }, (_, k) => -R + (2 * R * k) / 160), ys = xs.map((v) => f(v));
  const lo = Math.min(-0.5, ...ys, a), hi = Math.max(1, ...ys, a);
  const X = (v) => p + ((v + R) / (2 * R)) * (W - 2 * p), Y = (v) => H - p - ((v - lo) / (hi - lo)) * (H - 2 * p);
  const col = z > 0 ? "var(--accent)" : "var(--muted)", lab = "fill:var(--muted);font:11px var(--mono)";
  const svg = svgNode("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": words.aria });
  svg.append(
    svgNode("line", { x1: X(-R), x2: X(R), y1: Y(0), y2: Y(0), style: "stroke:var(--line-hi)" }),
    svgNode("line", { x1: X(0), x2: X(0), y1: Y(lo), y2: Y(hi), style: "stroke:var(--line-hi)" }),
    svgNode("path", { d: xs.map((v, k) => `${k ? "L" : "M"}${X(v).toFixed(1)},${Y(ys[k]).toFixed(1)}`).join(""), style: "fill:none;stroke:var(--fg2);stroke-width:1.5" }),
    svgNode("line", { x1: X(z), x2: X(z), y1: Y(0), y2: Y(a), style: `stroke:${col};stroke-dasharray:2 3` }),
    svgNode("circle", { cx: X(z), cy: Y(a), r: 4.5, style: `fill:${col}` }),
    svgNode("text", { x: X(-R), y: Y(0) + 14, style: lab }, `−${R}`),
    svgNode("text", { x: X(R), y: Y(0) + 14, "text-anchor": "end", style: lab }, words.axisZ(R)),
    svgNode("text", { x: X(0) + 6, y: Y(hi) + 4, style: lab }, words.axisA));
  return svg;
}
