// What the page draws each block with (D4 §7.2): the 33 block types of the forge catalog mapped to
// the page's 27 stage kinds, each kind to the renderer module that draws it (loaded lazily by
// renderers/index.js), and the kind sets the stage graph's segmentation and the stepper read.
// The coverage test holds KIND_OF to the catalog's block types; renaming a kind means changing it
// here and in the renderer that lists it.

export const KIND_OF = Object.freeze({
  dense: "dense", embedding: "embedding", concat: "concat", batchnorm: "norm", layernorm: "norm",
  dropout: "dropout", activation: "activation", residual: "residual", sequential: "sequential",
  parallel: "parallel", conv1d: "conv", pool1d: "pool1d", rnn: "recurrent", gru: "recurrent",
  lstm: "recurrent", attention: "attention", attention_pool: "attention_pool",
  transformer_encoder: "transformer", feature_tokenizer: "tokenizer", positional: "positional",
  flatten: "reshape", reshape: "reshape", pool: "pool", head: "head", decoder: "decoder",
  reparameterize: "latent", gcn_conv: "graph_conv", sage_conv: "graph_conv", gat_conv: "graph_conv",
  gin_conv: "graph_conv", graph_pool: "graph_pool", edge_dropout: "edge_dropout", custom: "custom",
});

// The stage kinds: `input` (the network's inputs, drawn first) and the kinds of KIND_OF.
export const STAGE_KINDS = Object.freeze(["input", ...new Set(Object.values(KIND_OF))]);

// Stage kind → the module under net/renderers/ that draws it. A kind fused into a dense run when one
// follows or precedes it (embedding, concat, activation, dropout, norm) is drawn by the box renderer
// when it stands alone.
export const RENDERER_OF = Object.freeze({
  input: "input.js", embedding: "box.js", concat: "box.js", dense: "dense.js", activation: "box.js",
  dropout: "box.js", norm: "box.js", residual: "group.js", sequential: "group.js", parallel: "group.js",
  conv: "conv.js", pool1d: "conv.js", recurrent: "recurrent.js", attention: "attention.js",
  attention_pool: "attention.js", transformer: "attention.js", tokenizer: "attention.js",
  positional: "box.js", reshape: "box.js", pool: "box.js", head: "head.js", decoder: "dense.js",
  latent: "latent.js", graph_conv: "graph.js", graph_pool: "graph.js", edge_dropout: "box.js",
  custom: "box.js",
});

// The node types network-view@2 adds beside the block types (D3 §3.1).
export const SYNTHETIC = Object.freeze(["input", "output", "residual_add", "parallel_merge", "encoder_layer"]);

// Composite kinds: drawn as groups, their children inside them.
export const GROUP_KINDS = Object.freeze(["residual", "sequential", "parallel"]);

// Kinds fused into the dense stage they follow inside a dense run (a tick or a caption on its column).
export const FUSED_KINDS = Object.freeze(["activation", "dropout", "norm"]);

// Kinds fused into the input side of the dense run that follows them.
export const INPUT_SIDE_KINDS = Object.freeze(["embedding", "concat"]);

// Kinds that do nothing to an answer (only while training, or a relabelling of the same numbers):
// the stepper turns them into a note on the next step (D4 §8.1).
export const QUIET_KINDS = Object.freeze(["dropout", "edge_dropout", "reshape", "concat", "positional"]);

// Kinds with a step of their own in "how the network computed it" (D4 §8.1); a dense run gives one
// step per dense stage (its fused bend included); a bend standing alone changes the numbers too, and
// a custom block is shown opaque.
export const STEP_KINDS = Object.freeze([
  "dense", "embedding", "norm", "activation", "conv", "pool1d", "recurrent", "attention", "attention_pool", "transformer",
  "tokenizer", "pool", "latent", "graph_conv", "graph_pool", "custom", "head",
]);

// The step a group adds after its inside (D4 §8.1): a residual group its sum, a parallel group its
// merge; a sequential group none (its inside is laid out inline).
export const GROUP_STEP = Object.freeze({ residual: "residual", parallel: "merge" });

// Kinds a stage's weight is drawn as a unit matrix for (the dense-like layers).
export const MATRIX_KINDS = Object.freeze(["dense", "head"]);

// The activation names the v2.2 service sent (PyTorch class names) in the catalog's own names.
export const ACTIVATION_NAMES = Object.freeze({
  Identity: "none", ReLU: "relu", GELU: "gelu", SiLU: "silu", Tanh: "tanh", Sigmoid: "sigmoid", LeakyReLU: "leaky_relu",
});

// The link each task answers through (R-C1), and the view key its page speaks in (D4 §5.1), by
// task kind; a sequence task reads its base's.
export const LINK_OF = Object.freeze({
  multiclass: "softmax", binary: "sigmoid", multilabel: "sigmoid_each", regression: "identity", ordinal: "cumulative",
  unsupervised: "reconstruction", anomaly: "reconstruction", forecast: "horizon",
});
export const VIEW_KEY_OF = Object.freeze({
  multiclass: "classes", binary: "binary", multilabel: "labels", regression: "value", ordinal: "ordinal",
  unsupervised: "reconstruction", anomaly: "anomaly", forecast: "horizon",
});

export const kindOf = (type) => (Object.prototype.hasOwnProperty.call(KIND_OF, type) ? KIND_OF[type] : null);
export const rendererOf = (kind) => (Object.prototype.hasOwnProperty.call(RENDERER_OF, kind) ? RENDERER_OF[kind] : null);
export const activationName = (name) => (name === null || name === undefined ? null
  : Object.prototype.hasOwnProperty.call(ACTIVATION_NAMES, name) ? ACTIVATION_NAMES[name] : String(name).toLowerCase());
