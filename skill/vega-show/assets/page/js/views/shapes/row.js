// The row shape (D4 §6.5): a sample is one row of raw columns.
//   editor        today's form: one field per raw input column of /v1/schema, typed by it; the
//                 request is {row}
//   sources       many at once: the demonstration sample (training part), the shipped examples
//                 (sealed test part), the reader's CSV (raw columns, an optional truth column)
//   attribution   what tipped the answer: gradient × input added up per input family, the members
//                 that weighed most listed under each family
//   inputStep     the stepper's first step: the shaped inputs furthest from zero, raw → x
//   drawInput     the input segment of the network drawing (net/renderers/input.js draws a row's
//                 families)
import { el } from "../../core/dom.js";
import { fmtNum, sg, plain } from "../../core/fmt.js";
import { termRow } from "../../core/bars.js";
import { t, tText, data, cls } from "../../core/text.js";
import { familiesOf } from "../../net/families.js";
import inputRenderer from "../../net/renderers/input.js";
import { viewState, methodNote } from "../shared.js";
import { demoOf, testOf, shapedOf, attributionOf, explainedName } from "../payloads.js";

const TOKENS = Object.freeze(["Infinity", "-Infinity", "+Infinity", "NaN"]);
const K_TOP = 6;
const columnsOf = (schema) => (schema && Array.isArray(schema.input) ? schema.input : []);

// A field's text back to a raw value: empty is null, a non-finite token stays a token (the service
// reads it), text stays text, a number becomes a number; a mistyped number stays text and the
// service refuses it by name.
export function cellOf(text, dtype) {
  const s = String(text).trim();
  if (s === "") return null;
  if (TOKENS.includes(s)) return s;
  if (dtype === "string") return s;
  return Number.isNaN(Number(s)) ? s : Number(s);
}

export const editor = Object.freeze({
  mount(host, schema) {
    const st = viewState("row-editor");
    Object.assign(st, { host, schema });
    host.replaceChildren(...columnsOf(schema).map((col) => el("label", {}, data(col.name), el("input", { "data-col": col.name, "data-type": col.dtype || "", value: "" }))));
  },
  load(preset) {
    const st = viewState("row-editor"), row = preset ? preset.row || (preset.rows && preset.rows[0]) || null : null;
    if (!st.host) return;
    for (const input of st.host.querySelectorAll("input")) {
      const v = row ? row[input.dataset.col] : null;
      input.value = v === null || v === undefined ? "" : String(v);
    }
  },
  read() {
    const st = viewState("row-editor"), row = {};
    if (!st.host) return { row };
    for (const input of st.host.querySelectorAll("input")) row[input.dataset.col] = cellOf(input.value, input.getAttribute("data-type"));
    return { row };
  },
  describe(preset) {
    if (!preset) return { title: "", text: "" };
    const typical = preset.expected !== null && preset.expected !== undefined && (preset.group === "typical" || preset.kind === null) && !Array.isArray(preset.expected);
    return { title: typical ? tText("try.typical", { name: cls(preset.expected) }) : preset.name, text: preset.text };
  },
});

// Many at once: where the rows come from. run() answers {rows, truth} (truth null when the rows
// carry none); the CSV source is a file input (file: true) whose run(file) the view calls.
export function sources(examples) {
  const demo = demoOf(examples), test = testOf(examples);
  return [
    demo ? { id: "demo", title: t("many.src.demo"), sub: t("many.src.demo_note"), run: () => demo } : null,
    test ? { id: "test", title: t("many.src.test"), sub: t("many.src.test_note"), run: () => test } : null,
    { id: "csv", file: true, title: t("many.src.csv"), sub: t("many.src.csv_note"), run: (file) => ({ file }) },
  ].filter(Boolean);
}

// What tipped the answer: per family of inputs (the story's groups), the sum of gradient × input of
// its members, largest first; under each, its three heaviest members.
export function attribution(ex, env) {
  const a = attributionOf(ex, env.graph), names = a.names, by = new Map(names.map((n, i) => [n, a.values[i]]));
  const units = names.map((_, i) => [i]), fam = familiesOf({ groups: env.story && env.story.input_groups, names, units }).families;
  const rows = fam.map((g) => {
    const parts = g.members.map((n) => [n, by.get(n) || 0]).filter((q) => q[1] !== 0).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]));
    return { g, v: parts.reduce((s, q) => s + q[1], 0), parts };
  }).filter((f) => f.parts.length).sort((x, y) => Math.abs(y.v) - Math.abs(x.v));
  const mx = Math.max(1e-9, ...rows.map((f) => Math.abs(f.v))), bars = el("div", { class: "bars diverge" });
  rows.forEach((f, i) => {
    const fill = el("div", { class: "fill " + (f.v >= 0 ? "pos" : "neg"), style: `width:${Math.abs(f.v) / mx * 50}%;animation-delay:${i * .05}s` });
    bars.append(el("div", { class: "bar" }, el("div", { class: "name" }, f.g.label), el("div", { class: "track" }, fill),
      el("div", { class: "val" }, tText(f.v >= 0 ? "try.why.raised" : "try.why.lowered", { v: Math.abs(f.v).toPrecision(2) }))),
    el("div", { class: "bar-sub" }, f.parts.slice(0, 3).map(([n, v]) => `${data(n)} ${v >= 0 ? "↑" : "↓"}`).join("   ·   ")));
  });
  return el("div", {}, el("p", { class: "hint views-why-lead" }, t("try.why.lead", { name: explainedName(ex) })), methodNote(ex && ex.attribution), bars);
}

// The stepper's input step, after the recipe's sentence: the shaped inputs furthest from zero, the
// raw value each came from beside it.
export function inputStep(ex, row, env) {
  const x = shapedOf(ex, env && env.graph), n = x.values.length, K = Math.min(K_TOP, n);
  const order = x.values.map((_, i) => i).sort((a, b) => Math.abs(x.values[b]) - Math.abs(x.values[a]));
  const top = order.slice(0, K), rest = order.slice(K).map((i) => x.values[i]), mx = Math.max(1e-9, n ? Math.abs(x.values[top[0]]) : 1);
  const raw = (name) => {
    if (!row || !(name in row)) return null;
    const v = row[name];
    return v === null || v === "" ? tText("shape.row.input.empty") : typeof v === "number" ? fmtNum(v) : data(v);
  };
  const bars = el("div", { class: "bars diverge eq" });
  top.forEach((i, k) => {
    const r = raw(x.names[i]);
    bars.append(termRow({ name: data(x.names[i]), v: x.values[i], max: mx, val: [r !== null ? el("span", { class: "wx" }, `${r}  → `) : "", tText("shape.row.input.x", { v: sg(x.values[i]) })], i: k }));
  });
  return el("div", {}, el("div", { class: "caps muted views-sub" }, t("shape.row.input.top", { k: K, zero: 0 })), bars,
    rest.length ? el("p", { class: "hint" }, t("shape.row.input.rest", { n: rest.length, lo: plain(Math.min(...rest)), hi: plain(Math.max(...rest)) })) : null);
}

export const drawInput = Object.freeze({
  measure: inputRenderer.measure, layout: inputRenderer.layout, draw: inputRenderer.draw,
  hit: inputRenderer.hit, caption: inputRenderer.caption, acts: inputRenderer.acts,
});
