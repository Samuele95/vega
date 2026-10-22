// node --test: how/sum.js — the stepper's number blocks as numbers (D4 §8.1; the hand-built page's
// sumBlock, 1792–1812): a weighted sum's terms, its largest ones, the rest, the bias, the page's
// total beside the network's z and the precision gap between them; the gap verdict the probe reads
// (within max(1e-3, 1e-3·|z|)); the units ranked by what they carried; the placeholders a stage's
// catalog words take. The blocks themselves build DOM and are walked by the parity probe.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { sumParts, gapOf, gapShown, rankUnits, worstOf, stageParams, TOP_TERMS, GAP_SHOWN, GAP_WITHIN } from "../js/how/sum.js";
import { StageGraph } from "../js/net/graph.js";
import { viewOfLegacy } from "./legacy-view.js";
import { runModel } from "../js/net/segments.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const R7 = JSON.parse(readFileSync(here("./fixtures/network-view/r7-legacy.json"), "utf8"));
const matrix = (rows) => ({ out: rows.length, in: rows[0].length, w: Float32Array.from(rows.flat()) });

test("a sum: the terms w·s, the largest first, the rest added up, the bias, the page's total", () => {
  const m = matrix([[1, -2, 0.5, 3, 0, 0.25, -4, 1], [0, 0, 0, 0, 0, 0, 0, 0]]), src = [1, 1, 2, 1, 5, 4, 0.5, -1];
  const p = sumParts(m, 0, src, { bias: [0.5, 0] });
  assert.deepEqual(p.terms, [1, -2, 1, 3, 0, 1, -2, -1]);
  assert.equal(p.K, TOP_TERMS);
  assert.deepEqual(p.top, [3, 1, 6, 0, 2, 5], "by magnitude, ties in index order");
  assert.equal(p.restCount, 2);
  assert.equal(p.rest, -1);
  assert.equal(p.bias, 0.5);
  assert.equal(p.total, 1.5);
  assert.equal(p.z, 1.5, "no network z: the page's own total");
  assert.equal(p.gap, 0);
  assert.equal(p.shown, false);
  assert.equal(p.weight(3), 3);
  assert.equal(p.src(4), 5);
  assert.equal(p.hasBias, true);
  const noBias = sumParts(m, 1, src);
  assert.equal(noBias.hasBias, false);
  assert.equal(noBias.total, 0);
});

test("the network's z beside the page's total: the gap row shows past half a percent", () => {
  const m = matrix([[1, 1]]);
  const small = sumParts(m, 0, [1, 1], { z: 2.004 }), big = sumParts(m, 0, [1, 1], { z: 2.2 });
  assert.equal(small.z, 2.004);
  assert.ok(Math.abs(small.gap - 0.004) < 1e-12);
  assert.equal(small.shown, false);
  assert.equal(big.shown, true);
  assert.equal(gapShown(0.004, 0.2), false, "below one the threshold is half a percent of one");
  assert.equal(gapShown(0.006, 0.2), true);
  assert.equal(gapShown(0.4, 100), false);
  assert.equal(GAP_SHOWN, 0.005);
});

test("many small terms: the block says no single term decides", () => {
  const n = 40, m = matrix([Array(n).fill(1)]), src = Array.from({ length: n }, (_, i) => (i < 6 ? 0.5 : 0.2));
  const p = sumParts(m, 0, src);
  assert.equal(p.restCount, n - 6);
  assert.equal(p.many, true);
  assert.equal(sumParts(matrix([[10, 1]]), 0, [1, 1]).many, false, "every term shown");
});

test("the gap of a recompute: network − page, within max(1e-3, 1e-3·|network|)", () => {
  assert.deepEqual(gapOf(1, 1.0005), { page: 1, network: 1.0005, gap: 1.0005 - 1, within: true });
  assert.equal(gapOf(1, 1.002).within, false);
  assert.equal(gapOf(1000, 1000.9).within, true);
  assert.equal(gapOf(1000, 1001.1).within, false);
  assert.equal(gapOf(0, -0.0009).within, true);
  assert.equal(GAP_WITHIN, 1e-3);
  assert.equal(worstOf([gapOf(1, 1.0001), null, gapOf(2, 1.99)]).network, 1.99);
  assert.equal(worstOf([null]), null);
});

test("units ranked by what they carried, the pick kept, by value without a relevance", () => {
  assert.deepEqual(rankUnits([0.1, -3, 2, 0.5], null), { rank: [1, 2, 3], j: 1, R: [0.1, -3, 2, 0.5] });
  assert.equal(rankUnits([0.1, -3, 2], null, 2).j, 2);
  assert.equal(rankUnits([0.1, -3, 2], null, 7).j, 1, "a pick out of range is ignored");
  assert.equal(rankUnits(null, [4, -9, 1]).j, 1);
  assert.deepEqual(rankUnits(null, null), { rank: [], j: 0, R: [] });
});

test("on R7, every hidden sum the page adds from the weights it was sent ends on the network's own z", () => {
  const graph = StageGraph.from(viewOfLegacy(R7.network)), model = runModel(graph, graph.segments()[1]);
  for (const ex of R7.explanations) {
    const src = [ex.inputs, ...ex.activations];
    model.layers.forEach((l, h) => {
      for (const j of [0, 7, l.out - 1]) {
        const p = sumParts(l.W, j, src[h], { bias: l.bias, z: ex.pre[h][j] });
        assert.ok(gapOf(p.total, p.z).within, `layer ${h} unit ${j}: ${p.total} vs ${p.z}`);
        assert.equal(p.shown, false, "full-precision weights: no gap row");
      }
    });
  }
});

test("a stage's placeholders: its widths, the subscript of its inputs, its rate and ε, overridable", () => {
  const node = { in_shape: [12, 65], out_shape: [256], config: { eps: 0.001 }, payload: { p: 0.2 }, dropout: 0 };
  const p = stageParams(node);
  assert.equal(p.units, 256);
  assert.equal(p.inputs, 65);
  assert.equal(p.n_sub, "₆₅");
  assert.equal(p.eps, 0.001);
  assert.equal(p.p, 0.2);
  const q = stageParams(node, { inputs: 7, v: "a" });
  assert.equal(q.n_sub, "₇", "n_sub follows the inputs given");
  assert.equal(q.v, "a");
});
