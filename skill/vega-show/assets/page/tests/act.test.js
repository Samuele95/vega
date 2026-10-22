// node --test: how/act.js — the page's own arithmetic for "how the network computed it": erf, the
// seven activations by their catalog names (D4 §8.3; against torch in
// vega-show/scripts/tests/test_page_math.py), the links' readings of a head (D4 §5.2), the ordinal
// class probabilities, the reconstruction error and the one-class distance, and MATH, the object
// window.__showcase.math exposes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { erf, sigmoid, softmax, logSumExp, ACT, actOf, silencesNegative, SILENCES_NEGATIVE, LEAKY_SLOPE, LINK_READ, LINKS,
  ordinalProbs, ordinalRank, expectedRank, reconstructionError, svddDistance, MATH } from "../js/how/act.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const COVERAGE = JSON.parse(readFileSync(here("../../../../vega-forge/scripts/backends/pytorch/coverage.json"), "utf8"));
const EXPLAIN = JSON.parse(readFileSync(here("../../../../vega-forge/scripts/backends/pytorch/explain-catalog.json"), "utf8"));
const FORGE = JSON.parse(readFileSync(here("../../../../vega-forge/scripts/catalog.json"), "utf8"));
const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const grid = Array.from({ length: 321 }, (_, k) => -8 + k * 0.05);

test("the activations are the catalog's, keyed by their catalog names, each with a curve", () => {
  assert.deepEqual(Object.keys(ACT).sort(), [...COVERAGE.activations].sort());
  assert.deepEqual(Object.keys(EXPLAIN.activations).sort(), Object.keys(ACT).sort());
  for (const [name, entry] of Object.entries(EXPLAIN.activations)) {
    assert.ok(ACT[entry.curve], `${name}: curve ${entry.curve}`);
    assert.equal(silencesNegative(name), entry.silences_negative, `${name}: silences_negative`);
    assert.equal(silencesNegative(name, entry), entry.silences_negative);
  }
  assert.deepEqual([...SILENCES_NEGATIVE].sort(), Object.keys(EXPLAIN.activations).filter((n) => EXPLAIN.activations[n].silences_negative).sort());
});

test("erf is Abramowitz & Stegun 7.1.26: odd, bounded, within 1.5e-7 of known values", () => {
  const known = { 0: 0, 0.5: 0.5204998778130465, 1: 0.8427007929497149, 2: 0.9953222650189527, 3: 0.9999779095030014 };
  for (const [x, v] of Object.entries(known)) { close(erf(Number(x)), v, 1.5e-7, `erf(${x})`); close(erf(-Number(x)), -v, 1.5e-7, `erf(-${x})`); }
  for (const x of grid) assert.ok(Math.abs(erf(x)) <= 1);
});

test("each activation by definition: identity, relu, gelu (erf form), silu, tanh, sigmoid, leaky_relu", () => {
  for (const z of grid) {
    assert.equal(ACT.none(z), z);
    assert.equal(ACT.relu(z), Math.max(0, z));
    close(ACT.gelu(z), 0.5 * z * (1 + erf(z / Math.SQRT2)), 0, "gelu");
    close(ACT.silu(z), z / (1 + Math.exp(-z)), 1e-12, "silu");
    close(ACT.tanh(z), Math.tanh(z), 0, "tanh");
    close(ACT.sigmoid(z), 1 / (1 + Math.exp(-z)), 1e-15, "sigmoid");
    close(ACT.leaky_relu(z), z > 0 ? z : LEAKY_SLOPE * z, 0, "leaky_relu");
  }
  close(ACT.gelu(1), 0.8413447460685429, 1e-6, "gelu(1) = Φ(1)");
  assert.equal(LEAKY_SLOPE, EXPLAIN.activations.leaky_relu.params.slope);
});

test("actOf: a catalog name (null and none the identity), the leaky slope from the catalog entry, null for an unknown name", () => {
  assert.equal(actOf(null)(3), 3);
  assert.equal(actOf("none")(-2), -2);
  assert.equal(actOf("leaky_relu")(-2), -0.02);
  assert.equal(actOf("leaky_relu", { params: { slope: 0.2 } })(-2), -0.4);
  assert.equal(actOf("GELU"), null, "PyTorch class names are the v2.2 service's; the page reads catalog names");
  assert.equal(actOf("mish"), null);
});

test("sigmoid, softmax and logSumExp stay finite at any magnitude", () => {
  assert.equal(sigmoid(-1000), 0);
  assert.equal(sigmoid(1000), 1);
  const p = softmax([1000, 999, -1000]);
  close(p.reduce((a, v) => a + v, 0), 1, 1e-12, "softmax sums to one");
  close(p[0] / p[1], Math.E, 1e-9, "softmax ratios");
  close(logSumExp([1000, 1000]), 1000 + Math.log(2), 1e-9, "logSumExp");
});

test("the links' readings: one per link of the network view, as explanation@2 outputs.read states them", () => {
  assert.deepEqual([...LINKS].sort(), Object.keys(FORGE.inspection.links).sort());
  const z = [2, -1, 0.5];
  assert.deepEqual(LINK_READ.softmax(z), softmax(z));
  assert.deepEqual(LINK_READ.sigmoid([0.3]), [sigmoid(0.3)]);
  assert.deepEqual(LINK_READ.sigmoid_each(z), z.map(sigmoid));
  assert.deepEqual(LINK_READ.identity(z), z);
  assert.deepEqual(LINK_READ.cumulative(z), z.map(sigmoid));
  assert.deepEqual(LINK_READ.reconstruction(z), z);
  assert.equal(LINK_READ.distance(z), null);
  assert.deepEqual(LINK_READ.horizon(Float32Array.from(z)), z);
});

test("ordinal: the class probabilities add up to one, the rank counts the boundaries passed, the expected rank is Σ σ", () => {
  const z = [2.2, 0.3, -1.4];
  const p = ordinalProbs(z), s = z.map(sigmoid);
  assert.equal(p.length, 4);
  close(p.reduce((a, v) => a + v, 0), 1, 1e-12, "Σ p");
  close(p[0], 1 - s[0], 1e-15, "p0");
  close(p[2], s[1] - s[2], 1e-15, "p2");
  close(p[3], s[2], 1e-15, "p3");
  assert.equal(ordinalRank(z), 2);
  assert.equal(ordinalRank([-3, -4]), 0);
  close(expectedRank(z), s[0] + s[1] + s[2], 1e-15, "E[y]");
});

test("the reconstruction error: numeric mean squared error plus the nominal columns' mean cross-entropy", () => {
  const xhat = [0.5, -1, 2, 0.1, 0.3, 1, -1], x = [1, -1, 2, 1, 0];   // 3 numeric, then ids of two nominals (cardinality 2 each)
  const r = reconstructionError(xhat, x, [2, 2]);
  const mse = (0.25 + 0 + 0) / 3, ce1 = logSumExp([0.1, 0.3]) - 0.3, ce2 = logSumExp([1, -1]) - 1;
  close(r.error, mse + (ce1 + ce2) / 2, 1e-12, "error");
  assert.deepEqual(r.unitOf, [0, 1, 2, 3, 3, 4, 4]);
  close(r.per[3], ce1, 1e-12, "the first nominal's term");
  const numeric = reconstructionError([1, 2], [1, 4]);
  assert.equal(numeric.error, 2);
  assert.equal(reconstructionError([1, 2, 3], [1, 2]), null, "a head that is not laid out as the inputs");
  assert.equal(reconstructionError([0.5, 1, -1], [0.5, 1], [2]).error, logSumExp([1, -1]) - (-1), "a numeric column rebuilt exactly: the cross-entropy alone");
});

test("the one-class distance is the squared distance from the centre", () => {
  assert.equal(svddDistance([1, 2, 3], [1, 0, 1]), 8);
  assert.equal(svddDistance(Float32Array.from([0.5]), [0.5]), 0);
});

test("MATH is the pure part, as the probe exposes it", () => {
  for (const k of ["erf", "sigmoid", "softmax", "logSumExp", "ACT", "actOf", "silencesNegative", "LINK_READ", "ordinalProbs",
    "ordinalRank", "expectedRank", "reconstructionError", "svddDistance"]) assert.ok(k in MATH, k);
  assert.ok(Object.isFrozen(MATH) && Object.isFrozen(ACT) && Object.isFrozen(LINK_READ));
});
