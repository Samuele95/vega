// The strings of the sequence area (js/net/renderers/{conv, recurrent, attention}.js, js/net/seq.js,
// js/net/seq-tokens.js and js/views/shapes/window.js): the convolution, recurrent and attention
// drawings' captions and words, and the window shape's editor, sources, what tipped the answer and
// the stepper's input step. strings.en.js merges every fragment into the page's one table; a key
// another fragment defines is a load error, a variant of one (key@<view key>, key@<shape>,
// key#one|#other) is a key of its own. Values carry no digit outside a {placeholder}; domain words
// come from the lexicon ({step.many}, {window.one}, {feature.many}, {row.one}, {source.the}); every
// value the service or the network brings arrives as a parameter. DYNAMIC: dynamic prefix → its
// closed member list.

export const STRINGS = Object.freeze({
  // ---- a convolution and its pooling (renderers/conv.js)
  "conv.caption": "CONV {l} × {c}",
  "conv.caption.pool1d": "POOL {l} × {c}",
  "conv.pool1d.kind.max": "max",
  "conv.pool1d.kind.avg": "mean",
  "conv.cap.kernel": "each of its {co} channels reads {k} neighbouring {step.many} of the {c} channels before it (stride {stride}, padding {padding})",
  "conv.cap.pool": "takes the {kind} of every {size} neighbouring {step.many}, channel by channel",
  "conv.cap.cell": "channel {channel} at position {at} ({step})",

  // ---- a recurrent layer (renderers/recurrent.js)
  "recurrent.caption": "{cell} · {h}",
  "recurrent.layers": "× {n} layers",
  "recurrent.return.last": "last",
  "recurrent.return.all": "every step",
  "recurrent.return.mean": "mean",
  "recurrent.return.max": "max",
  "recurrent.cap.units#one": "{n} {cell} unit",
  "recurrent.cap.units#other": "{n} {cell} units",
  "recurrent.cap.forward": "reading the {step.many} in order,",
  "recurrent.cap.both": "reading the {step.many} both ways,",
  "recurrent.cap.return.last": "hands on its state after the last {step.one}.",
  "recurrent.cap.return.all": "hands on its state at every {step.one}.",
  "recurrent.cap.return.mean": "hands on the mean of its states over the {step.many}.",
  "recurrent.cap.return.max": "hands on the largest of its states over the {step.many}, unit by unit.",
  "recurrent.cap.memory": "how long each gate remembers (spectral radius of its state weights): {list}",
  "recurrent.cap.step": "{step.one} {step}, read forwards",
  "recurrent.cap.step_back": "{step.one} {step}, read backwards",
  "recurrent.cap.norm": "size of the state here: {v}",

  // ---- attention, attention pooling, the encoder, the feature tokens (renderers/attention.js, seq-tokens.js)
  "attention.caption.attention": "ATTENTION · {h} HEADS",
  "attention.caption.attention_pool": "ATTENTION POOL · {n}",
  "attention.caption.transformer": "ENCODER · {layers} × {h} HEADS",
  "attention.caption.tokenizer": "TOKENS · {tokens}",
  "attention.more": "+{n}",
  "attention.not_exposed": "this answer does not show its attention",
  "attention.pool.mean": "mean",
  "attention.layers": "× {n}",
  "attention.cls": "[cls]",
  "attention.cap.heads#one": "{n} head over vectors of {d} numbers: each {step.one} weighs every other {step.one}, and the weights add up to one.",
  "attention.cap.heads#other": "{n} heads over vectors of {d} numbers: each {step.one} weighs every other {step.one}, head by head, and each head's weights add up to one.",
  "attention.cap.pool": "then averages over the {step.many}.",
  "attention.cap.head": "head {h}: rows are the {step.many} that read, columns the {step.many} they read.",
  "attention.cap.pool_steps": "weighs the {n} {step.many} and adds them up: the weights add up to one.",
  "attention.cap.step": "{step.one} {step}",
  "attention.cap.encoder#one": "{n} layer, each {h} heads of attention over vectors of {d} numbers and a feed-forward of {ff} units.",
  "attention.cap.encoder#other": "{n} layers, each {h} heads of attention over vectors of {d} numbers and a feed-forward of {ff} units.",
  "attention.cap.layer": "layer {k}: tokens on the left, a map per head on the right.",
  "attention.cap.tokens#one": "{n} token of {d} numbers.",
  "attention.cap.tokens#other": "{n} tokens of {d} numbers, one per {feature.one} and one the answer is read from.",
  "attention.cap.cls": "the token the answer is read from",
  "attention.cap.token": "the token of {name}",
  "attention.cap.norm": "its size for this {row.one}: {v}",

  // ---- the window shape (views/shapes/window.js)
  "shape.window.context": "before",
  "shape.window.editor.lead": "The last {n} {step.many} form the {window.one} the network reads; the {step.many} before it are context. Edit any cell to ask again.",
  "shape.window.editor.empty": "Pick an example to fill the {window.one}.",
  "shape.window.src.demo_note": "a {window.one} of each series a run · training data",
  "shape.window.src.test": "shipped series",
  "shape.window.src.test_note": "from the sealed test part, every {step.one} answered",
  "shape.window.src.csv_note": "consecutive {row.many} of your {source.one}; the service builds the {window.many}",
  "shape.window.why.lead": "Each cell is the attribution of one {feature.one} at one {step.one} to the score of {name}: accent where it pushed the score up, the other colour where it pushed it down.",
  "shape.window.why.families": "added up by family of {feature.many}",
  "shape.window.why.steps": "added up by {step.one}, oldest to newest",
  "shape.window.why.raised": "raised {v}",
  "shape.window.why.lowered": "lowered {v}",
  "shape.window.input.title": "the last {n} {step.many}, each shaped by the recipe",
  "shape.window.input.top": "the {k} cells furthest from {zero}",
  "shape.window.input.cell": "{feature} @ {step}",
  "shape.window.input.x": "x = {v}",
  "shape.window.input.nominal#one": "{n} nominal {feature.one} enters as its value's number at every {step.one}: {names}.",
  "shape.window.input.nominal#other": "{n} nominal {feature.many} enter as their values' numbers at every {step.one}: {names}.",
});

export const DYNAMIC = Object.freeze({
  "conv.pool1d.kind.": Object.freeze(["max", "avg"]),
  "recurrent.return.": Object.freeze(["last", "all", "mean", "max"]),
  "recurrent.cap.return.": Object.freeze(["last", "all", "mean", "max"]),
  "attention.caption.": Object.freeze(["attention", "attention_pool", "transformer", "tokenizer"]),
});
