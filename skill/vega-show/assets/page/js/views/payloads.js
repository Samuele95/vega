// The service's payloads as the views read them. A v2.3 bundle answers in the documents the page
// is written for (vega/examples@2, vega/explanation@2, /v1/schema with the output record); a bundle
// packed before v2.3 (the bundle the parity probe runs on) answers in v2.2's forms, and so does
// its network (StageGraph.fromLegacy). This module is the one place that tells the two apart, so the
// views and the task and shape modules read one form: an answer's RowPrediction (D2, R-C4), the
// shaped inputs and their attribution, the presets, the demonstration sample, the output record.
// The v2.2 branches leave with the v2.2 page (W15).
import { tText, data, cls } from "../core/text.js";

export const EXPLANATION = "vega/explanation@2";
export const EXAMPLES = "vega/examples@2";
const has = (o, k) => !!o && Object.prototype.hasOwnProperty.call(o, k);
const isObject = (v) => !!v && typeof v === "object" && !Array.isArray(v);

// ---- the output record (vega/output@2 in /v1/schema.output; R-C4 thresholds) --------------------
// {reading, classes, positive, threshold, thresholdScale, perLabel, quantiles, horizon}, each null
// when neither the schema nor the network states it.
export function outputRecord(schema, graph = null) {
  const out = schema && isObject(schema.output) ? schema.output : {};
  const task = graph ? graph.task : {};
  const classes = out.classes || (schema && Array.isArray(schema.classes) ? schema.classes : null)
    || (task.outputs && ["classes", "labels", "score", "thresholds"].includes(task.outputs.kind) ? task.outputs.names : null);
  const positive = out.positive ?? task.positive ?? (classes && classes.length === 2 ? classes[1] : null);
  const horizons = Array.isArray(out.horizons) ? out.horizons.length : null;
  return Object.freeze({
    reading: out.reading || null, classes: classes ? classes.slice() : null, positive: positive === null ? null : String(positive),
    threshold: typeof out.threshold === "number" ? out.threshold : typeof task.threshold === "number" ? task.threshold : null,
    thresholdScale: out.threshold_scale || task.threshold_scale || null,
    perLabel: isObject(out.threshold_per_label) ? out.threshold_per_label : null,
    quantiles: Array.isArray(out.quantiles) ? out.quantiles.slice() : Array.isArray(task.quantiles) ? task.quantiles.slice() : null,
    horizon: horizons || (graph && graph.sample && graph.sample.horizon) || null,
  });
}

// ---- one answer ----------------------------------------------------------------------------------
// The RowPrediction of an explanation: explanation@2's `prediction` (what /v1/predict answers for
// the row), or the v2.2 explanation's own answer keys.
export function predictionOf(ex) {
  if (!ex) return null;
  if (ex.schema === EXPLANATION) return ex.prediction || { warnings: ex.warnings || [] };
  return { label: ex.label, probabilities: ex.probabilities || {}, warnings: ex.warnings || [] };
}

// The shaped inputs of an explanation, {names, values} in network order (a row's vector).
export function shapedOf(ex, graph) {
  if (!ex) return { names: [], values: [] };
  if (ex.schema === EXPLANATION) {
    const shaped = ex.inputs && Array.isArray(ex.inputs.shaped) && typeof ex.inputs.shaped[0] === "number" ? ex.inputs.shaped : [];
    return { names: (ex.inputs && ex.inputs.order) || [], values: shaped };
  }
  return { names: graph ? graph.inputs.names : [], values: Array.isArray(ex.inputs) ? ex.inputs : [] };
}

// The attribution of an explanation per network input, {names, values} (gradient × input on the
// explained target; a row's vector).
export function attributionOf(ex, graph) {
  if (!ex) return { names: [], values: [] };
  if (ex.schema === EXPLANATION) {
    const f = ex.attribution && ex.attribution.features;
    return f && Array.isArray(f.attribution) ? { names: f.names, values: f.attribution } : { names: [], values: [] };
  }
  return { names: graph ? graph.inputs.names : [], values: Array.isArray(ex.attribution) ? ex.attribution : [] };
}

// The name of what an explanation explained (the answer's class, the explained target), as data.
export function explainedName(ex) {
  if (!ex) return "";
  const p = predictionOf(ex);
  const name = (p && p.label) ?? (ex.target && ex.target.name) ?? (ex.outputs && ex.outputs.answer && ex.outputs.answer.class) ?? "";
  return data(name === null ? "" : name);
}

// The explanation's reconstruction, per input: {names, x, xhat} (explanation@2 outputs.raw beside the
// shaped inputs), or null when it carries none.
export function reconstructionOf(ex) {
  if (!ex || ex.schema !== EXPLANATION || !ex.outputs || !Array.isArray(ex.outputs.raw)) return null;
  const x = ex.inputs && Array.isArray(ex.inputs.shaped) ? ex.inputs.shaped : null;
  if (!x || typeof x[0] !== "number" || x.length !== ex.outputs.raw.length) return null;
  return { names: ex.inputs.order || ex.outputs.names || [], x, xhat: ex.outputs.raw };
}

// ---- the presets (vega/examples@2 presets; v2.2's named presets) ------------------------------------
// One form: {id, kind, group, groupText, stratum, expected, row, rows, history, facts, name, text}.
// `expected` is the value a typical preset should get (a class, a label list, a number, a flag) or
// null; `group` the examples@2 group (typical | edge), or null with `groupText` the v2.2 group's own
// words; `name` and `text` its title and description, composed from the strings table (examples@2)
// or the v2.2 payload's own (data).
export function presetsOf(examples) {
  const list = (examples && Array.isArray(examples.presets)) ? examples.presets : [];
  const strata = new Map(((examples && examples.demo && examples.demo.strata) || []).map((s) => [s.id, s]));
  return list.filter((p) => p && p.id).map((p) => (examples.schema === EXAMPLES ? presetOf(p, strata) : legacyPreset(p)));
}

function factsText(facts) {
  const out = {};
  for (const [k, v] of Object.entries(facts || {})) out[k] = Array.isArray(v) ? data(v.join(", ")) : typeof v === "number" ? v : data(v);
  if (Array.isArray(facts && facts.bin)) Object.assign(out, { lo: facts.bin[0], hi: facts.bin[1] });
  return out;
}

// The words a preset is named by when its stratum has no name: a value stratum (a forecast's truth,
// one value per step ahead, binned on its first step) by its bin, never a bare first-step number;
// else the value.
function unnamed(expected, facts) {
  const bin = Array.isArray(facts && facts.bin) ? { lo: facts.bin[0], hi: facts.bin[1] } : null;
  if (bin && Array.isArray(expected) && expected.length && expected.every((v) => typeof v === "number")) return tText("preset.bin.first_step", bin);
  if (bin && typeof expected === "number") return tText("preset.typical_bin.name", bin);
  return expected === null ? "" : data(Array.isArray(expected) ? expected.join(", ") : String(expected));
}

function presetOf(p, strata) {
  const st = strata.get(p.stratum) || null, expected = p.expected ? p.expected.value : null;
  const params = { ...factsText(p.facts), name: st && st.name !== null ? cls(st.name) : unnamed(expected, p.facts) };
  return Object.freeze({
    id: p.id, kind: p.kind, group: p.group, groupText: null, stratum: st, expected, expectedKind: p.expected ? p.expected.kind : null,
    row: p.row || null, rows: p.rows || null, history: p.history || null, facts: p.facts || {},
    name: tText(`preset.${p.kind}.name`, params), text: tText(`preset.${p.kind}.text`, params),
  });
}

function legacyPreset(p) {
  return Object.freeze({
    id: p.id, kind: null, group: null, groupText: p.group === undefined || p.group === null ? "" : data(p.group), stratum: null,
    expected: p.expected ?? null, expectedKind: p.expected === null || p.expected === undefined ? null : "class",
    row: p.row || null, rows: null, history: null, facts: {},
    name: data(p.name ?? p.id), text: p.description ? data(p.description) : "",
  });
}

// ---- many at once: the demonstration sample and the shipped examples ------------------------------
// {rows, truth, history}: the demo sample's rows (one per item for a row lane) and their truth, or
// the shipped test rows (truth null); null when the payload has none.
export function demoOf(examples) {
  const d = examples && examples.demo;
  if (!d) return null;
  if (Array.isArray(d.items)) {
    const items = d.items.filter((it) => Array.isArray(it.rows) && it.rows.length);
    return { rows: items.map((it) => it.rows[it.rows.length - 1]), truth: truthOrNull(items.map((it) => it.truth ?? null)), history: null };
  }
  return Array.isArray(d.rows) ? { rows: d.rows, truth: Array.isArray(d.labels) ? truthOrNull(d.labels) : null, history: null } : null;
}

// A truth list with no truth in it (an unlabelled lane's demo items) is no truth: the views never
// count a row against a label it does not have.
export const truthOrNull = (truth) => (Array.isArray(truth) && truth.some((v) => v !== null && v !== undefined) ? truth : null);

export function testOf(examples) {
  const t = examples && examples.test;
  return t && Array.isArray(t.rows) && t.rows.length ? { rows: t.rows, truth: null, history: t.history || null } : null;
}

// The drift monitor's request: the batch, with its truth only when the bundle's requests take one
// (/v1/schema's `truth` names its type; a bundle with none, an unsupervised one, refuses `labels`
// with a 422) and a row has one.
export const monitorBody = (body, truth, schema) => (Array.isArray(truth) && truth.some((x) => x !== null && x !== undefined)
  && schema && isObject(schema.truth) && schema.truth.type ? { ...body, labels: truth } : body);

// The truth an uploaded table carried (the CSV answer's truth, or v2.2's labels), or null.
export const uploadTruth = (res) => (res && (Array.isArray(res.truth) ? res.truth : Array.isArray(res.labels) ? res.labels : null)) || null;

// Whether a document is the version the page reads, or a v2.2 form it still accepts (no schema at
// all): true for both. A document naming another schema is not.
export const readable = (doc, want) => !!doc && (!has(doc, "schema") || doc.schema === want);
