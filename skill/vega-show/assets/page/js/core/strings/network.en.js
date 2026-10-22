// The strings of the network area (js/net/**: the stage graph's drawings, the reading guide, the
// unit and stage card, the weights read). strings.en.js merges every fragment into the page's one
// table; a key another fragment defines is a load error, a variant of one (key@<view key>,
// key@<shape>, key#one|#other) is a key of its own. Values carry no digit outside a {placeholder};
// domain words come from the lexicon ({row.a}, {feature.many}, {node.many}, {step.many}) and the
// target's from {target.one} / {target.many}. DYNAMIC: dynamic prefix → its closed member list.
//
// Today's drawing of a pure dense chain keeps today's words. The captions of its columns, the input
// families' fallback names and the reading guide's steps carry D4's names (net.caption.*,
// head.caption@<view key>, net.family.*, guide.inputs@<shape>, guide.edges, guide.outputs@<view key>;
// A18); the rest of today's words are still the core fragment's legacy.net.* (the words
// js/legacy/net.js showed, moved there by the strings package), which the modules under js/net/
// name as they are, with their variants for the lanes today's page never drew here
// (legacy.net.<key>@<view key>). The words of the drawings that are new (segments, boxes, groups,
// the output column by link, the guide's phrases, the stage card, the per-stage weights) are here.

export const STRINGS = Object.freeze({
  // ---- reading the network (net/graph.js)
  "net.unreadable": "The network could not be read: {why}",
  // what a payload that is not a stage graph lacks (graph.js NetworkViewError; the {why} above)
  "net.error.not_object": "the network view is not an object",
  "net.error.schema": "expected {want}, found {found}",
  "net.error.unavailable": "the network view says the network is not available",
  "net.error.field": "the network view has no \"{field}\"",
  "net.error.task_field": "the network view's task has no \"{field}\"",
  "net.error.link": "unknown link {link}",
  "net.error.no_nodes": "the network view has no nodes",
  "net.error.node_id": "node id {id} is missing or repeated",
  "net.error.node_type": "node {id}: unknown type {type}",
  "net.error.node_parent": "node {id}: unknown parent {parent}",
  "net.error.node_child": "node {id}: unknown child {child}",
  "net.error.node_tensor": "node {id}: tensor {tensor} is not in tensors",
  "net.error.edge": "edge {from} → {to} names an unknown node",
  "net.error.tensor": "no tensor {tensor} in this network",
  "net.error.tensor_payload": "tensor {tensor}: the answer does not carry its values in its declared shape",
  // a deferred tensor the page could not fetch from the service (a probe warning; graph.fetchDeferred)
  "net.tensor.unfetched": "tensor {tensor} could not be fetched: {error}",
  // a stage kind the page has no drawing for (renderers/index.js RendererGap)
  "net.renderer.gap": "no renderer for stage kind {kind} ({module}): {why}",
  "net.renderer.unmapped": "kinds.js maps it to no module",
  "net.renderer.unloaded": "the module did not load ({error})",
  "net.renderer.members": "the module lacks {members}",
  "net.renderer.unlisted": "the module does not list this kind",
  "net.renderer.not_loaded": "the drawing was not given this module",
  // why the story's input groups were not used (families.js; a probe warning, one family drawn)
  "net.family.problem.unknown": "input group {group}: {name} is not an input of the network",
  "net.family.problem.twice": "input {name} is in both {first} and {second}",
  "net.family.problem.none": "input {name} is in no input group",

  // ---- dense runs inside a larger graph (renderers/run-segment.js, renderers/dense.js)
  "net.caption.decoder": "DECODER {h} · {k} OF {n}",
  "net.caption.bottleneck": "WAIST · {k} OF {n}",
  "net.cap.stages_hint": "Point at a stage to read what it does; click it for its card.",
  "net.cap.weights_missing": "Some of the weights the drawing reads were neither sent with the network nor fetched from it, so its lines leave them out.",

  // ---- the columns' captions (a dense run's inputs and hidden layers; the output column by view key)
  "net.caption.inputs": "INPUT FAMILIES",
  "net.caption.hidden": "HIDDEN {h} · {k} OF {n}",
  "head.caption@classes": "CLASSES",
  "head.caption@binary": "SCORE",
  "head.caption@labels": "LABELS",
  "head.caption@value": "VALUE",
  "head.caption@ordinal": "STEPS · CLASSES",
  "head.caption@reconstruction": "REBUILT INPUTS",
  "head.caption@anomaly": "SCORE",
  "head.caption@horizon": "STEPS AHEAD",

  // ---- the input families when the story names none, or no group takes an input (families.js)
  "net.family.all": "{feature.many}",
  "net.family.other": "other {feature.many}",

  // ---- the detail level's output caption and a score's meaning, for the lanes other than classes
  "legacy.net.detail.outputs@binary": "SCORE",
  "legacy.net.detail.outputs@labels": "LABELS",
  "legacy.net.detail.outputs@value": "VALUE",
  "legacy.net.detail.outputs@ordinal": "STEPS",
  "legacy.net.detail.outputs@reconstruction": "REBUILT INPUTS",
  "legacy.net.detail.outputs@anomaly": "SCORE",
  "legacy.net.detail.outputs@horizon": "STEPS AHEAD",
  "legacy.net.cap.class@binary": "— the network's one score; above its threshold the answer is this {target.one}.",
  "legacy.net.cap.class@labels": "— one {target.one}'s score; every {target.one} above its threshold is in the answer.",
  "legacy.net.cap.class@value": "— the number the network answers.",
  "legacy.net.cap.class@ordinal": "— the score of passing one step between neighbouring {target.many}.",
  "legacy.net.cap.class@reconstruction": "— the network's rebuilt copy of one input.",
  "legacy.net.cap.class@anomaly": "— one of the numbers the network measures a {row.one} by; the further they land from what is typical, the more unusual the {row.one}.",
  "legacy.net.cap.class@horizon": "— the value the network answers for one step ahead.",

  // ---- the output column by link (renderers/head.js)
  "head.threshold": "threshold {t}",
  "head.above": "above {name}",
  "head.quantile": "{name} {v}",
  "head.distance": "distance {d}",
  "head.error": "error {e}",

  // ---- boxes and groups (renderers/box.js, renderers/group.js, renderers/input.js)
  "box.kind.dense": "dense layer",
  "box.kind.head": "output layer",
  "box.kind.decoder": "decoder",
  "box.kind.residual": "residual block",
  "box.kind.sequential": "block sequence",
  "box.kind.parallel": "parallel branches",
  "box.kind.embedding": "embedding",
  "box.kind.concat": "join",
  "box.kind.activation": "bend",
  "box.kind.dropout": "dropout",
  "box.kind.norm": "normalisation",
  "box.kind.positional": "positions",
  "box.kind.reshape": "reshape",
  "box.kind.pool": "pooling",
  "box.kind.edge_dropout": "edge dropout",
  "box.kind.custom": "custom block",
  "box.kind.conv": "convolution",
  "box.kind.pool1d": "pooling",
  "box.kind.recurrent": "recurrent layer",
  "box.kind.attention": "attention",
  "box.kind.attention_pool": "attention pooling",
  "box.kind.transformer": "transformer encoder",
  "box.kind.tokenizer": "feature tokens",
  "box.kind.latent": "latent code",
  "box.kind.graph_conv": "graph convolution",
  "box.kind.graph_pool": "graph pooling",
  "box.note.train_only": "only while training",
  "box.unverified": "unverified",
  "box.shape": "{from} → {to}",
  "group.residual": "residual",
  "group.parallel": "branches",
  "group.repeat": "× {n}",
  "group.share": "{p:pct}",
  "input.steps": "{n} {step.many}",
  "input.neighbours": "and its neighbours",
  "input.embedded": "its nominal {feature.many} are read through an embedding: each value becomes a learned vector of numbers",

  // ---- the reading guide (net/guide.js): the inputs by shape and number of families, the middle
  // composed from phrases when it is more than today's hidden layers, the outputs by view key
  "guide.inputs#one": "On the left, the {n} {feature.many} of {row.a}, as one family. The circle stands for all of them.",
  "guide.inputs#other": "On the left, the {n} {feature.many} of {row.a}, gathered into {f} families. Each circle is one family.",
  "guide.inputs@window#one": "On the left, the {n} {feature.many} of each of the last {length} {step.many} of {window.a}, as one family. The circle stands for all of them.",
  "guide.inputs@window#other": "On the left, the {n} {feature.many} of each of the last {length} {step.many} of {window.a}, gathered into {f} families. Each circle is one family.",
  "guide.inputs@graph#one": "On the left, the {n} {feature.many} of {node.a} and of its neighbours, as one family. The circle stands for all of them.",
  "guide.inputs@graph#other": "On the left, the {n} {feature.many} of {node.a} and of its neighbours, gathered into {f} families. Each circle is one family.",
  "guide.middle": "In the middle, {phrases}.",
  "guide.phrase.run#one": "{n} hidden layer of {units} units",
  "guide.phrase.run#other": "{n} hidden layers of {units} units",
  "guide.phrase.conv#one": "a convolution sliding along the {step.many}",
  "guide.phrase.conv#other": "{n} convolutions sliding along the {step.many}",
  "guide.phrase.recurrent#one": "a recurrent layer that reads the {step.many} in order",
  "guide.phrase.recurrent#other": "{n} recurrent layers that read the {step.many} in order",
  "guide.phrase.attention#one": "an attention layer that weighs the {step.many} against each other",
  "guide.phrase.attention#other": "{n} attention layers that weigh the {step.many} against each other",
  "guide.phrase.transformer#one": "a transformer encoder",
  "guide.phrase.transformer#other": "{n} transformer encoders",
  "guide.phrase.graph_conv#one": "{n} round of message passing between neighbouring {node.many}",
  "guide.phrase.graph_conv#other": "{n} rounds of message passing between neighbouring {node.many}",
  "guide.phrase.latent#one": "a latent code the answer is read from",
  "guide.phrase.latent#other": "{n} latent codes the answer is read from",
  "guide.phrase.bottleneck": "a waist of {units} units every {row.one} must pass through",
  "guide.phrase.residual#one": "a residual block that adds its input back to what it computes",
  "guide.phrase.residual#other": "{n} residual blocks that add their input back to what they compute",
  "guide.phrase.parallel#one": "a block whose branches read the same numbers side by side",
  "guide.phrase.parallel#other": "{n} blocks whose branches read the same numbers side by side",
  "guide.phrase.box#one": "{n} further step",
  "guide.phrase.box#other": "{n} further steps",
  "guide.note.run": "Each unit mixes everything before it into one number. The {k} that matter most in each layer are drawn.",
  "guide.edges": "Each line is a weight: how much one circle pushes the next. Teal raises it, violet lowers it; the thicker the line, the stronger the push.",
  "guide.outputs@classes": "On the right, one score per {target.one} — {n} in all. The highest score is the network's answer.",
  "guide.outputs@binary": "On the right, one score; when it passes the threshold the answer is {positive}.",
  "guide.outputs@labels": "On the right, one score per {target.one} — {n} in all. Every {target.one} whose score passes its threshold is in the answer.",
  "guide.outputs@value": "On the right, one number: the value the network answers.",
  "guide.outputs@ordinal": "On the right, one score per step between neighbouring {target.many}. The answer is the {target.one} the passed steps lead to.",
  "guide.outputs@reconstruction": "On the right, the network's rebuilt copy of its {n} inputs; how far it lands from the input is its error.",
  "guide.outputs@anomaly": "On the right, the numbers the network measures a {row.one} by; the further they land from what is typical, the more unusual the {row.one}.",
  "guide.outputs@horizon": "On the right, one value per step ahead — {n} in all.",

  // ---- the stage card (net/card.js): what a unit of any dense stage, a box or a group says
  "card.own": "its own weights on what it reads (+ raises · − lowers)",
  "card.own_note": "Only this layer's own weights: past a stage that is not a plain weighted sum, a product of weights would not mean anything.",
  "card.stage_params#one": "{n} parameter",
  "card.stage_params#other": "{n} parameters",
  "card.stage_none": "no weights of its own",
  "card.stage_formula": "what it computes",

  // ---- the weights, read, on a graph that is not a pure dense chain (net/weights.js)
  "weights.map.unavailable": "No reading of the {feature.many} against the {target.many} is available for this network.",
  "weights.stages.lead": "Every weight matrix of the network, stage by stage, each in its own order: a row is one unit, a column one of the numbers it reads. Each pixel is one weight — teal raises, violet lowers. Point at a pixel to name it.",
  "weights.stages.caption": "{name} · {from} → {to} · {n} parameters",
  "weights.stages.row": "unit {r}",
  "weights.stages.col": "input {c}",
  "weights.withheld": "This service publishes summaries of its weights, not the weights themselves.",

  // ---- one answer drawn on a graph of stages (the try view's canvas)
  "net.try.stages": "Each stage glows with how strongly it answered this {row.one}; the output column shows the answer, {name}.",
});

export const DYNAMIC = Object.freeze({
  "box.kind.": Object.freeze(["dense", "head", "decoder", "residual", "sequential", "parallel", "embedding", "concat",
    "activation", "dropout", "norm", "positional", "reshape", "pool", "edge_dropout", "custom", "conv", "pool1d",
    "recurrent", "attention", "attention_pool", "transformer", "tokenizer", "latent", "graph_conv", "graph_pool"]),
  "guide.phrase.": Object.freeze(["run", "conv", "recurrent", "attention", "transformer", "graph_conv", "latent",
    "bottleneck", "residual", "parallel", "box"]),
});
