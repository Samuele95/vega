// The strings of the graph area (js/net/renderers/graph.js: the graph convolutions and the readout on
// the network's canvas; js/views/shapes/graph.js: the node or graph editor, many at once's sources,
// what tipped the answer, the stepper's input step). strings.en.js merges every fragment into the
// page's one table; a key another fragment already defines is a load error, a variant of one
// (key@<view key>, key@<shape>, key#one|#other) is a key of its own. Values carry no digit outside a
// {placeholder}; the domain's words come from the lexicon ({node.one}, {node.many}, {edge.many},
// {graph.one}, {feature.many}) and the target's from {target.one}. DYNAMIC: dynamic prefix → its
// closed member list.

export const STRINGS = Object.freeze({
  // ---- the canvas: what each convolution does, its glyph's marks, a message pointed at
  "graph.about.gcn_conv": "Each {node.one} averages what it and its neighbours hold, each weighted by how connected the two are, then mixes the result through one weight matrix.",
  "graph.about.sage_conv": "Each {node.one} keeps its own numbers through one weight matrix and adds its neighbours' numbers, gathered, through another.",
  "graph.about.gat_conv": "Each {node.one} weighs its neighbours by attention (α), learned from what both ends hold, across {heads} heads, and adds up what they send.",
  "graph.about.gin_conv": "Each {node.one} adds up its neighbours' numbers and its own, counted once plus ε, then passes the sum through a small network.",
  "graph.about.graph_pool": "Every {node.one} of the {graph.one} is gathered into one vector, by their {readout}.",
  "graph.gin.self": "({one} + ε) · self, ε = {eps}",
  "graph.sage.root": "root apart",
  "graph.gat.heads#one": "α · {n} head",
  "graph.gat.heads#other": "α · {n} heads",
  "graph.readout.sum": "sum",
  "graph.readout.mean": "mean",
  "graph.readout.max": "max",
  "graph.w.weight": "W",
  "graph.w.root": "W root",
  "graph.w.neigh": "W neighbours",
  "graph.w.att_src": "a src",
  "graph.w.att_dst": "a dst",
  "graph.w.mlp": "MLP",
  "graph.node.this": "this {node.one}",
  "graph.node.key": "{node.one} {key}",
  "graph.node.ref": "neighbour {k}",
  "graph.node.other": "another {node.one}",
  "graph.node.nth": "{node.one} {k}",
  "graph.cap.alone": "No message from another {node.one} reached this one: it answered from its own numbers alone.",
  "graph.cap.message": "{from} → {to}, from distance {hop}: carried {rel} of the answer, coefficient {coef}.",
  "graph.cap.thumb": "{name}: {rows} × {cols} weights, read down to blocks; teal positive, violet negative, deeper the larger.",
  "graph.cap.thumb_missing": "{name}: the payload does not carry this matrix.",

  // ---- the editor
  "graph.edit.link": "links this {node.one} to its neighbours",
  "graph.edit.graph_id": "names the {graph.one} this {node.one} belongs to",
  "graph.edit.nodes#one": "the {graph.one}'s {n} {node.one}; choose it to edit it",
  "graph.edit.nodes#other": "the {graph.one}'s {n} {node.many}; choose one to edit it",
  "graph.edit.toggle_graph": "edit this {graph.one}",

  // ---- many at once: the reader's CSV
  "graph.src.csv_node": "raw columns, one {node.one} per row, optional true label",
  "graph.src.csv_graph": "raw columns, one {node.one} per row, each naming its {graph.one}",

  // ---- what tipped the answer
  "graph.why.lead": "The picture puts this {node.one} in the centre and its neighbours around it, nearer ones inside. Each dot is as large as the attribution of that {node.one}'s {feature.many} to the score of {name}, teal where they pushed it up, violet where they pushed it down; each line is {edge.a}, as thick as the attribution that passed along it. Click a dot to read its {feature.many}.",
  "graph.why.lead_graph": "The picture draws the {graph.one}'s {node.many}. Each dot is as large as the attribution of that {node.one}'s {feature.many} to the score of {name}, teal where they pushed it up, violet where they pushed it down; each line is {edge.a}, as thick as the attribution that passed along it. Click a dot to read its {feature.many}.",
  "graph.why.none": "This answer does not say what its {node.many} tipped.",
  "graph.why.self": "the {node.one}'s own {feature.many}, by family",
  "graph.why.hops": "what each distance brought",
  "graph.why.hop0": "the {node.one} itself",
  "graph.why.hop": "at distance {n}",
  "graph.why.near": "the {k} neighbours that weighed most",
  "graph.why.near_graph": "the {k} {node.many} that weighed most",
  "graph.why.node": "{name}, at distance {hop}, tipped {v}: its {feature.many} that weighed most",
  "graph.why.node_tip": "{name} · distance {hop} · {v}",
  "graph.why.hidden#one": "{n} more {node.one} the answer lists is not drawn.",
  "graph.why.hidden#other": "{n} more {node.many} the answer lists are not drawn.",
  "graph.why.outside": "The {node.many} beyond those the answer lists tipped {v} together.",
  "graph.why.gap": "The method's terms add up to {sum}, while the score moved by {change}: a gap of {gap} the terms do not account for.",

  // ---- the stepper's input step
  "graph.input.lead": "What the network reads first: the {node.one}'s own {n} numbers, each shaped by the recipe fitted on the training data, and the numbers of its neighbours.",
  "graph.input.lead_graph": "What the network reads first: the {graph.one}'s {nodes} {node.many}, {f} numbers each, shaped by the recipe fitted on the training data.",
  "graph.input.cell": "{feature} @ {node.one} {k}",
  "graph.input.hop#one": "{n} neighbour at distance {h}",
  "graph.input.hop#other": "{n} neighbours at distance {h}",
  "graph.input.hops": "Its neighbourhood: {hops}.",
  "graph.input.alone": "No neighbour reaches it: the answer reads the {node.one} alone.",
  "graph.input.edges": "The service drew its {edge.many} itself, from the linking columns {columns}; the other rows of a request never change the answer.",
  "graph.input.whole_graph": "The answer reads the whole {graph.one} at once: every {node.one} of it, and what its {edge.many} carry between them.",
  "graph.input.edges_graph": "The service drew the {edge.many} itself, from the linking columns {columns}, between the {row.many} of this {graph.one}: each of them can change the answer, and the {row.many} of another {graph.one} never do.",
});

export const DYNAMIC = Object.freeze({
  "graph.about.": Object.freeze(["gcn_conv", "sage_conv", "gat_conv", "gin_conv", "graph_pool"]),
  "graph.readout.": Object.freeze(["sum", "mean", "max"]),
});
