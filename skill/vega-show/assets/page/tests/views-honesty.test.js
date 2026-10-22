// node --test: what the views say must be what the records hold (gate C's review panel).
//   - a forecast preset's error is a mean over the steps ahead only when every step has its true
//     value; a preset carrying the first step's truth alone says it is the error one step ahead;
//   - a binned forecast preset is named by its bin, never by a bare first-step value;
//   - an unlabelled lane's demo sample (a truth list of nulls) is no truth: no "0 of the top 10
//     are anomalous", and without a threshold nothing is said to be flagged;
//   - the authored story strings (a context fact's label, a limit's title, a fact's note, an
//     experiment's run label, an input family's label) resolve their lexicon paths;
//   - the attribution's method is named beside what it drew, and a completeness gap is shown.
import { test } from "node:test";
import assert from "node:assert/strict";
import { install } from "./dom-shim.js";

install();
const { configureText, unresolved, storyPlain, storyText, cls, data } = await import("../js/core/text.js");
const { Lexicon } = await import("../js/core/lexicon.js");
const horizon = await import("../js/views/tasks/horizon.js");
const anomaly = await import("../js/views/tasks/anomaly.js");
const { demoOf, presetsOf, truthOrNull } = await import("../js/views/payloads.js");
const { methodNote } = await import("../js/views/shared.js");
const { familiesOf } = await import("../js/net/families.js");

const text = (n) => (n ? n.textContent : "").replace(/\s+/g, " ").trim();
const LEX = { schema: "vega/lexicon@1", row: { one: "reading", many: "readings", article: "a" }, feature: { one: "sensor", many: "sensors", article: "a" } };
const fresh = (viewKey = "horizon") => configureText({ lex: new Lexicon(LEX, null), lane: { viewKey, shape: "window" } });

// L10 typical-bin-0: predictions −0.1975 / −0.7459 / −0.8982 against the truth 0.7 / 1.7 / 1.2
const FORECAST = { values: { h1: -0.1975, h2: -0.7459, h3: -0.8982 } };

test("a forecast's mean error is over every step ahead, and only when every step has its truth", () => {
  fresh();
  const all = horizon.output.agree({ p: FORECAST }, { expected: [0.7, 1.7, 1.2] });
  const mae = (Math.abs(-0.1975 - 0.7) + Math.abs(-0.7459 - 1.7) + Math.abs(-0.8982 - 1.2)) / 3;
  assert.match(text(all), /^mean error /);
  assert.ok(text(all).includes(mae.toFixed(2).slice(0, 4)), `${text(all)} carries the three-step mean ${mae}`);
  const first = horizon.output.agree({ p: FORECAST }, { expected: 0.7 });
  assert.match(text(first), /^error .* one step ahead/, "a first-step truth alone is never called a mean");
  assert.doesNotMatch(text(first), /mean/);
  const gap = horizon.output.agree({ p: FORECAST }, { expected: [0.7, null, 1.2] });
  assert.match(text(gap), /one step ahead/, "a step without its truth: no mean over the steps");
  assert.deepEqual(unresolved(), []);
});

test("a binned forecast preset is named by its bin; a class preset keeps its stratum's name", () => {
  fresh();
  const examples = { schema: "vega/examples@2", demo: { strata: [{ id: "bin-0", name: null }, { id: "class-a", name: "tardy" }] },
    presets: [{ id: "typical-bin-0", kind: "typical_sequence", group: "typical", stratum: "bin-0", facts: { of: 4, bin: [0.5, 2] },
                expected: { kind: "value", value: [0.7, 1.7, 1.2], range: [0.5, 2] }, rows: [], history: [] },
              { id: "typical-class-a", kind: "typical_sequence", group: "typical", stratum: "class-a", facts: { of: 3 },
                expected: { kind: "class", value: "tardy" }, rows: [], history: [] }] };
  const [bin, cls] = presetsOf(examples);
  assert.equal(bin.name, "0.5 to 2 one step ahead");
  assert.deepEqual(bin.expected, [0.7, 1.7, 1.2]);
  assert.equal(cls.name, "tardy");
});

test("an unlabelled lane's demo truth is no truth, and without a threshold nothing is flagged", () => {
  configureText({ lex: new Lexicon(LEX, null), lane: { viewKey: "anomaly", shape: "row" } });
  const demo = demoOf({ demo: { items: [{ rows: [{ a: 1 }], truth: null }, { rows: [{ a: 2 }], truth: null }] } });
  assert.equal(demo.truth, null, "a list of nulls is no truth");
  assert.equal(truthOrNull([null, "anomalous"]).length, 2, "one truth makes it a truth list");
  const env = { output: { threshold: null, classes: null }, roles: null, lane: { scoreSource: "reconstruction" }, lex: null };
  const preds = Array.from({ length: 20 }, (_, i) => ({ score: i / 10, anomalous: null }));
  const said = text(anomaly.summary.sentence(preds, [null, null], env));
  assert.match(said, /No threshold was chosen, so none is flagged/);
  assert.doesNotMatch(said, /\b0 of 20\b/);
  const right = anomaly.summary.right(preds, Array(20).fill(null), env);
  assert.doesNotMatch(text(right.body), /are anomalous/, "no precision at k on data without labels");
  assert.match(text(anomaly.output.verdict({ p: { score: 0.4 } }, env)), /no threshold was chosen/);
  assert.equal(anomaly.output.agree({ p: { score: 0.4 } }, { expected: true }, env), null, "no flag, no agreement claimed");
  assert.equal(anomaly.summary.miss({ score: 0.4 }, true, env), false);
});

test("authored story strings resolve their lexicon paths", () => {
  fresh("classes");
  assert.equal(storyPlain("{row.many} in the source"), "readings in the source");
  assert.equal(text(storyText("{feature.many} the network reads", document.createElement("span"))), "sensors the network reads");
  const { families } = familiesOf({ groups: [{ id: "g", label: "the {feature.many}", text: "every {feature.one}", members: ["a"] }], names: ["a"] });
  assert.equal(families[0].label, "the sensors");
  assert.equal(families[0].text, "every sensor");
  assert.deepEqual(unresolved(), []);
});

test("the attribution's method is named, and a stated completeness gap is shown", () => {
  fresh("classes");
  const gx = methodNote({ method: "grad_x_input", completeness: null });
  assert.match(text(gx), /gradient × input: a local, first-order reading/);
  assert.match(text(gx), /do not add up to the change in score/);
  assert.equal(gx.querySelector(".views-completeness"), null);
  const ig = methodNote({ method: "integrated_gradients", completeness: { sum: 0.9, target_minus_baseline: 1, gap: 0.1 } });
  assert.match(text(ig), /integrated gradients/);
  assert.ok(ig.querySelector(".views-completeness"), "the gap is a part of its own");
  assert.match(text(ig), /a gap of 0\.1000/);
  assert.match(text(methodNote({ method: { name: "grad_x_input" } })), /gradient × input/, "the demo sample's {name, steps} form");
  assert.equal(methodNote(null), null);
  assert.equal(methodNote([0.1, 0.2]), null, "a v2.2 attribution list names no method");
});

test("a class is shown by the words the story gives it, and by its name without them", () => {
  configureText({ lex: new Lexicon(LEX, null), lane: { viewKey: "binary", shape: "row" }, classNames: { 1: "tardy", 0: "punctual" } });
  assert.equal(cls("1"), "tardy");
  assert.equal(cls(0), "punctual");
  assert.equal(cls("2"), "2", "a class without words keeps its name");
  assert.equal(data("1"), "1", "data is never renamed");
  const examples = { schema: "vega/examples@2", demo: { strata: [{ id: "class-1", name: "1" }] },
    presets: [{ id: "typical-class-1", kind: "typical_correct", group: "typical", stratum: "class-1", facts: { right: 3, of: 4 },
                expected: { kind: "class", value: "1" }, row: {} }] };
  assert.equal(presetsOf(examples)[0].name, "tardy", "the preset is named in the class's words");
  assert.equal(presetsOf(examples)[0].expected, "1", "while the answer is compared on the class itself");
  configureText({ lex: new Lexicon(LEX, null), lane: { viewKey: "binary", shape: "row" } });
  assert.equal(cls("1"), "1");
});
