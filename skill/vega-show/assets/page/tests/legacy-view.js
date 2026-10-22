// The v2.2-merge payload of GET v1/network (a bundle packed before v2.3; the R7 fixture,
// fixtures/network-view/r7-legacy.json) as the vega/network-view@2 document D3 §5.6's mapping makes
// of it, for the node tests that hold the page to today's numbers on R7. The page itself reads
// network-view@2 only (net/graph.js StageGraph.from; BUILD-PLAN W15 removed StageGraph.fromLegacy):
// the mapping lives here, in the tests, and nowhere in the template.
//
//   viewOfLegacy(payload, {task}) -> network-view@2
//
// D3 §5.6: `inputs` -> inputs.order (and the chain's inputs); `outputs` -> task.outputs.names;
// `layers[l].{name, in, out, activation, dropout, params}` -> a dense stage per layer (the last the
// head) with its activation, and a dropout stage after it when the layer dropped units, and
// chain.layers[l]; `weights[l].{shape, f32 | int8 + scale, bias}` -> the stage's weight tensor
// (base64 little-endian float32) and its bias (a JSON list); `params` -> params.total. The task is
// the one the bundle states (its output kind; the v2.2 page drew classes), read through its link.
// Not served, not copied into a showcase.
import { decode } from "../js/core/decode.js";
import { LINK_OF, activationName } from "../js/net/kinds.js";

const OUTPUTS_OF = Object.freeze({ multiclass: "classes", binary: "score", multilabel: "labels", regression: "value" });

function base64(f32) {
  return Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString("base64");
}

function task(p, kind) {
  const link = LINK_OF[kind] || "softmax";
  return { kind, base: null, link, outputs: { kind: OUTPUTS_OF[kind] || "classes", names: p.outputs.slice() }, positive: null,
           threshold: null, threshold_scale: null, quantiles: null, decision_rule: link === "softmax" ? "argmax" : "none",
           targets: ["logit"], default_target: "logit" };
}

function stage(id, type, layer, activation) {
  return { id, type, view: "units", parent: null, children: [], role: null, path: id, in_shape: [layer.in], out_shape: [layer.out],
           per: "sample", config: {}, formula: "", code: null,
           params: { count: layer.params || 0, weight: `${id}.weight`, bias: `${id}.bias`, activation } };
}

export function viewOfLegacy(p, { task: kind = "multiclass" } = {}) {
  if (!p || p.available === false || !Array.isArray(p.layers) || !Array.isArray(p.weights)) throw new Error("not a v2.2-merge network payload");
  const nodes = [], tensors = {}, chain = [], last = p.layers.length - 1;
  p.layers.forEach((layer, l) => {
    const id = layer.name || `layer${l}`, w = p.weights[l], act = activationName(layer.activation);
    nodes.push(stage(id, l === last ? "head" : "dense", layer, act));
    tensors[`${id}.weight`] = { id: `${id}.weight`, node: id, role: "weight", shape: w.shape.slice(), encoding: "base64-le", data: base64(decode(w).w) };
    tensors[`${id}.bias`] = { id: `${id}.bias`, node: id, role: "bias", shape: [layer.out], encoding: "json", data: Array.from(w.bias || []) };
    if (layer.dropout > 0) {
      const did = `${id}.dropout`;
      nodes.push({ id: did, type: "dropout", view: "box", parent: null, children: [], role: null, path: did, in_shape: [layer.out],
                   out_shape: [layer.out], per: "sample", config: { p: layer.dropout }, formula: "", code: null, params: { count: 0, p: layer.dropout } });
    }
    chain.push({ node: id, in: layer.in, out: layer.out, activation: act === "none" ? null : act, dropout: layer.dropout || 0,
                 weight: `${id}.weight`, bias: `${id}.bias`, params: layer.params || 0 });
  });
  return {
    schema: "vega/network-view@2", available: true, family: { name: null, source: null }, task: task(p, kind),
    sample: { shape: "row", length: null, stride: null, horizon: null, autoregressive: null, level: null, directed: null,
              self_loops: null, protocol: null, batching: null },
    inputs: { order: p.inputs.slice(), numeric: p.inputs.slice(), nominal: [], width: p.inputs.length, past_target: null, edges: null, axis_labels: {} },
    params: { total: p.params || 0, trainable: p.params || 0 }, nodes, edges: [], chain: { inputs: p.inputs.slice(), layers: chain }, tensors,
  };
}
