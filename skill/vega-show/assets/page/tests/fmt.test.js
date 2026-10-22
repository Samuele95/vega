// node --test: core/fmt.js keeps the hand-built page's number rules (one case each).
import { test } from "node:test";
import assert from "node:assert/strict";
import { fmtInt, fmtNum, fmtCi, fmtVal, scalar, signed, sg, plain, subs, pct, pad2, countUp } from "../js/core/fmt.js";

test("integers are grouped by the locale", () => {
  assert.equal(fmtInt(86543), "86,543");
  assert.equal(fmtNum(1234567), "1,234,567");
});

test("fmtNum: two decimals from 1, four significant places below, two digits of precision under 0.01", () => {
  assert.equal(fmtNum(123.456), "123");
  assert.equal(fmtNum(12.3456), "12.35");
  assert.equal(fmtNum(-1.5), "-1.50");
  assert.equal(fmtNum(0.123456), "0.1235");
  assert.equal(fmtNum(0.0012345), "0.0012");
  assert.equal(fmtNum("n/a"), "n/a");
});

test("a value with its interval", () => {
  const v = { value: 0.9123, ci: [0.9, 0.93] };
  assert.equal(fmtCi(v), "[0.9, 0.93]");
  assert.equal(fmtCi(0.5), "");
  assert.equal(fmtVal(v), "0.9123 [0.9, 0.93]");
  assert.equal(fmtVal(v, true), "0.9123");
  assert.equal(fmtVal([1, 2, 3]), "3");
  assert.equal(scalar(v), 0.9123);
  assert.equal(scalar(4), 4);
});

test("signs: a true minus, a plus on zero", () => {
  assert.equal(signed(-0.1234), "−0.123");
  assert.equal(signed(0), "+0.000");
  assert.equal(sg(-2.5), "−2.50");
  assert.equal(sg(1.234, 3), "+1.234");
  assert.equal(plain(-0.5), "−0.50");
  assert.equal(plain(0.5, 1), "0.5");
});

test("subscripts, percentages, two-digit counters", () => {
  assert.equal(subs(65), "₆₅");
  assert.equal(pct(0.955), "95.5 %");
  assert.equal(pct(0.5, 0), "50 %");
  assert.equal(pad2(3), "03");
  assert.equal(pad2(12), "12");
});

test("countUp writes the target at once under reduced motion", () => {
  const node = { textContent: "" };
  countUp(node, 1234, true);
  assert.equal(node.textContent, "1,234");
  countUp(node, "x", false);
  assert.equal(node.textContent, "x");
});
