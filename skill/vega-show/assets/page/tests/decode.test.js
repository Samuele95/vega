// node --test: core/decode.js reads the weight records of v1/network as the drawings need them.
import { test } from "node:test";
import assert from "node:assert/strict";
import { decode } from "../js/core/decode.js";

const b64 = (bytes) => Buffer.from(bytes).toString("base64");

test("float32, little-endian, row-major out × in", () => {
  const w = new Float32Array([0.5, -1.25, 3, 0, 1e-3, -7.5]);
  const rec = { f32: b64(new Uint8Array(w.buffer)), shape: [2, 3] };
  const m = decode(rec);
  assert.equal(m.out, 2);
  assert.equal(m.in, 3);
  assert.deepEqual(Array.from(m.w), Array.from(w));
});

test("int8 with a scale: signed bytes times the scale", () => {
  const rec = { int8: b64([0, 1, 127, 128, 255, 200]), scale: 0.5, shape: [3, 2] };
  assert.deepEqual(Array.from(decode(rec).w), [0, 0.5, 63.5, -64, -0.5, -28]);
});

test("each record is decoded once and shared", () => {
  const rec = { f32: b64(new Uint8Array(new Float32Array([1, 2]).buffer)), shape: [1, 2] };
  assert.equal(decode(rec), decode(rec));
});
