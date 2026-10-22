// node --test: the results view (js/views/results.js; D4 §6.3) on the story's verdict, in a small DOM
// (dom-shim.js). A30: a verdict that is not assessable (story@2 results.verdict: earned null, with its
// reason's values key) is said in words — "not assessable" and the evaluation's own reason, marked as
// data under the pseudo-locale — and never drawn as a win or a loss: no "lower is better" over the
// models' bars, whose scores live on scales of their own. An assessed verdict is said in its own word
// in every state — earned, not earned, beaten by a baseline — with the strongest baseline by name and
// the paired difference with its interval (gate C: a beaten verdict was rendered nowhere but in the
// collapsed limits), and keeps the min-direction hint. No key missing, no placeholder left.
import { test } from "node:test";
import assert from "node:assert/strict";
import { install } from "./dom-shim.js";

const doc = install();
globalThis.matchMedia = () => ({ matches: false });
const { setState } = await import("../js/core/state.js");
const { configureText, missingKeys, unresolved, PSEUDO, WRAP } = await import("../js/core/text.js");
const { Lexicon } = await import("../js/core/lexicon.js");
const results = await import("../js/views/results.js");

const REASON = "the primary reads no label: each model's own anomaly score, on a scale of its own";
const SLOTS = ["res-headline", "res-models", "res-ci-note", "res-verdict", "res-facts", "res-class", "res-class-cap", "res-per-class", "res-limits"];
const text = (node) => (node ? node.textContent : "").replace(/\s+/g, " ").trim();

// The view's slots, fresh (the view renders once per page: it returns when its bars are drawn).
function slots() {
  doc.body.replaceChildren();
  for (const id of SLOTS) {
    const n = doc.createElement(id === "res-headline" ? "h1" : "div");
    n.setAttribute("id", id);
    n.removeAttribute = (k) => { delete n.attributes[k]; };            // the two members dom-shim.js
    n.prepend = (...kids) => n.replaceChildren(...kids, ...n.childNodes); // lacks, for this view only
    doc.body.append(n);
  }
  const query = Object.getPrototypeOf(doc).querySelector.bind(doc);
  doc.querySelector = (sel) => (sel.startsWith(".disclose[") ? null : query(sel));
  return (id) => doc.getElementById(id);
}

// A story of an unlabelled anomaly lane: three models on anomaly_score_mean (min), the verdict given.
function story(verdict) {
  const values = {
    primary: { value: 0.8868, ci: [0.7925, 0.9958], n: 300 }, ci_level: 0.95,
    eval_network_anomaly_score_mean: { value: 0.8868, ci: [0.7925, 0.9958], n: 300 },
    eval_trivial_anomaly_score_mean: { value: 0.926, ci: [0.84, 1.01], n: 300 },
    eval_tree_anomaly_score_mean: { value: 0.4493, ci: [0.41, 0.49], n: 300 },
    ...(verdict.reason ? { [verdict.reason]: REASON } : {}),
    eval_diff_tree: { value: -0.0332, ci: [-0.0644, -0.0014], n: 271 },
  };
  const results = {
    id: "results", headline: "What the sealed test said about the network.",
    models: [{ name: "network", value: "eval_network_anomaly_score_mean", selected: true },
      { name: "trivial", value: "eval_trivial_anomaly_score_mean", selected: false },
      { name: "tree", value: "eval_tree_anomaly_score_mean", selected: false }],
    facts: [], breakdown: null, verdict,
    limits: [{ id: "verdict_not_assessable", title: "The verdict is not assessable", text: "No model is ranked against another." }],
  };
  return { values, sections: [results] };
}

function render(verdict, lang = "en") {
  const $ = slots();
  const s = story(verdict);
  configureText({ lang, lex: new Lexicon(null, null), values: s.values });
  setState({ story: s, page: { metric: { name: "anomaly_score_mean", direction: "min" } }, mods: null });
  results.view.show();
  return $;
}

const NOT_ASSESSABLE = { earned: null, beaten_by_baseline: null, strongest_baseline: null, diff: null, reason: "eval_verdict_reason" };

test("a verdict that is not assessable is said in words, with the evaluation's reason, never as a win or a loss", () => {
  const $ = render(NOT_ASSESSABLE);
  assert.equal(results.notAssessable({ verdict: NOT_ASSESSABLE }), true);
  assert.equal($("res-verdict").hidden, false);
  const said = text($("res-verdict"));
  assert.match(said, /^The verdict is not assessable:/);
  assert.match(said, /neither credited nor faulted/);
  assert.ok(said.endsWith(REASON), said);
  assert.doesNotMatch(said, /\b(earned|beaten|better|worse|wins?|loses?)\b/i, "no win or loss is claimed");
  assert.doesNotMatch(text($("res-models")), /lower is better/, "no ranking hint over scales that differ");
  assert.equal($("res-models").querySelectorAll(".bar").length, 3, "every model's metric is still reported");
  assert.match(text($("res-limits")), /The verdict is not assessable/);
  assert.deepEqual(missingKeys(), []);
  assert.deepEqual(unresolved(), []);
});

test("under the pseudo-locale the reason is the evaluation's words, marked as data", () => {
  const $ = render(NOT_ASSESSABLE, PSEUDO);
  assert.ok(text($("res-verdict")).includes(`${WRAP.data[0]}${REASON}${WRAP.data[1]}`), text($("res-verdict")));
  assert.ok(text($("res-verdict")).startsWith(WRAP.chrome[0]), "the frame is the strings table's");
});

test("an assessed verdict is said in its own word, with the strongest baseline and the paired difference", () => {
  const CASES = [
    [{ earned: true, beaten_by_baseline: false }, "earned", /^Earned: .*gradient-boosted trees, by -?0\.0332 \[/],
    [{ earned: false, beaten_by_baseline: false }, "not_earned", /^Not earned: .*gradient-boosted trees.* is -0\.0332 \[-0\.0644, -0\.0014\]/],
    [{ earned: false, beaten_by_baseline: true }, "beaten", /^Beaten by a baseline: gradient-boosted trees did better .* is -0\.0332 \[-0\.0644, -0\.0014\]/],
  ];
  for (const [v, state, said] of CASES) {
    const $ = render({ ...v, strongest_baseline: "tree", diff: "eval_diff_tree" });
    assert.equal($("res-verdict").hidden, false, state);
    assert.equal($("res-verdict").dataset.verdict, state);
    assert.match(text($("res-verdict")), said, state);
    assert.doesNotMatch(text($("res-verdict")), /could not separate|does as well/, `${state}: never a softer word`);
    assert.equal(results.verdictState({ verdict: { ...v } }), state);
    assert.match(text($("res-models")), /lower is better/);
    assert.deepEqual(missingKeys(), []);
    assert.deepEqual(unresolved(), []);
  }
  const $ = render({ earned: false, beaten_by_baseline: true, strongest_baseline: "tree", diff: null });
  assert.match(text($("res-verdict")), /is not recorded\.$/, "a difference the evaluation did not record is said so, never invented");
  assert.equal(results.notAssessable({ verdict: { earned: null, reason: null } }), false, "no reason, no not-assessable reading");
  assert.equal(results.notAssessable({}), false);
  assert.equal(results.verdictState({}), null, "no verdict, no line");
});
