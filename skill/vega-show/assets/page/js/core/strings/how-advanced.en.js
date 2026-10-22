// The strings of the how-advanced area: the words of the advanced explainers of "how the network
// computed it" (js/how/explainers/{conv, recurrent, attention, graph, latent}.js and their shared
// parts, js/how/cells.js). strings.en.js merges every fragment into the page's one table; a key
// another fragment already defines is a load error, a variant of one (key@<view key>, key@<shape>)
// is a key of its own. Values carry no digit outside a {placeholder}: every number is a parameter
// the explainer fills from the answer, the stage or the recompute. Block prose and formulas are the
// backend's explain catalog (explain.<type>.prose / .formula), which the explainers read first; the
// .lead and .formula keys here are the fallbacks of a page without the catalog. DYNAMIC: dynamic
// prefix → its closed member list.

export const STRINGS = Object.freeze({
  // ---- shared (how/cells.js): what stops a recompute, names of positions, channels and cells
  "how.adv.weights": "The page did not receive the weights this step needs ({what}): the service withholds them or sent them apart, so it shows the formula without recomputing it.",
  "how.adv.fit": "The numbers the answer brought for {stage} do not fit its shape {shape} ({what}): the page cannot place them, so it does not recompute the step.",
  "how.adv.position": "position {t}",
  "how.adv.channel": "channel {c}",
  "how.adv.unit": "unit {u}",
  "how.adv.dim": "{name} · dim {d}",
  "how.adv.cell": "{position} · {channel}",
  "how.adv.pick": "{what}  {v}",
  "how.adv.all": "Recomputed whole, the {n} numbers this stage passed on differ from the network's by at most {gap}.",

  // ---- a convolution (explainers/conv.js)
  "how.conv.lead": "Each of the {channels} output channels slides a window of {kernel} positions along the sequence, adding up the numbers under it, each times a weight of its own, plus a bias:",
  "how.conv.formula": "y[t, c′] = b[c′] + Σ_c Σ_j W[c′, c, j] · x[t·s + j·d − pad, c]",
  "how.conv.at": "{what} reads {k} positions of each of the {c} input channels, from {first} to {last}; its largest terms:",
  "how.conv.at_pad": "{what} lies wholly in the padding: every number it reads is zero, so it is its bias.",
  "how.conv.pad": "Positions before the start or past the end of the sequence read as zero (a padding of {pad} on each side).",
  "how.conv.pad_term": "{channel} @ padding",
  "how.conv.bias": "bias of channel {c}",
  "how.conv.next": "The map of {n} numbers — {positions} positions × {channels} channels — is what the next stage reads.",

  // ---- a recurrent layer (explainers/recurrent.js)
  "how.recurrent.lead": "The layer reads the sequence one position at a time and carries a state of {hidden} numbers from each position to the next:",
  "how.recurrent.formula.rnn": "h′ = tanh(W_ih·x + b_ih + W_hh·h + b_hh)",
  "how.recurrent.formula.gru": "r = σ(W_ir·x + b_ir + W_hr·h + b_hr) ; z = σ(W_iz·x + b_iz + W_hz·h + b_hz) ; n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn)) ; h′ = ({one} − z) ⊙ n + z ⊙ h",
  "how.recurrent.formula.lstm": "i, f, o = σ(W_i·x + b_i + W_h·h + b_h) ; g = tanh(W_ig·x + b_ig + W_hg·h + b_hg) ; c′ = f ⊙ c + i ⊙ g ; h′ = o ⊙ tanh(c′)",
  "how.recurrent.unit": "unit {u}",
  "how.recurrent.unit_back": "unit {u}, backward",
  "how.recurrent.at": "Unit {u} at {position}: the gates read the {f} numbers of this position and the {h} numbers of the state before it (the top of {layer} stacked layers).",
  "how.recurrent.at_back": "Unit {u} at {position}, reading the sequence backwards: the gates read the {f} numbers of this position and the {h} numbers of the state after it (the top of {layer} stacked layers).",
  "how.recurrent.prev": "The state it starts from is the page's own, carried from the first position read with the same weights: unit {u} of it was {h}.{cell}",
  "how.recurrent.prev_c": " Its cell was {c}.",
  "how.recurrent.gate.r": "reset gate r",
  "how.recurrent.gate.z": "update gate z",
  "how.recurrent.gate.n": "candidate n",
  "how.recurrent.gate.i": "input gate i",
  "how.recurrent.gate.f": "forget gate f",
  "how.recurrent.gate.g": "candidate g",
  "how.recurrent.gate.o": "output gate o",
  "how.recurrent.gate.c": "new cell c′",
  "how.recurrent.gate.h": "new state h′",
  "how.recurrent.unrolled": "Unrolled from a zero state over all {n} positions and {layers} direction layers, the page's states and the network's differ by at most {gap}.",
  "how.recurrent.no_states": "The answer did not bring the network's states step by step ({what}): the page compares only what the layer passed on.",
  "how.recurrent.return.last": "The layer passes on only its final state, {n} numbers; the states before it served to build it.",
  "how.recurrent.return.last_both": "The layer passes on each direction's final state — the forward one after the last position, the backward one after the first — {n} numbers.",
  "how.recurrent.return.all": "The layer passes on its state at every position: {n} numbers.",
  "how.recurrent.return.mean": "The layer passes on the mean of its states over the positions: {n} numbers.",
  "how.recurrent.return.max": "The layer passes on each unit's largest state over the positions: {n} numbers.",

  // ---- attention, attention pooling, an encoder layer, a tokenizer (explainers/attention.js)
  "how.attention.lead": "Each of the {heads} heads compares every position with every other, turns the similarities into weights that add up to one, and mixes what the positions carry with them:",
  "how.attention.formula": "A = softmax(Q·Kᵀ / √d_h) per head ; y = [A₁V₁ ‖ … ‖ A_hV_h]·W_o + b_o",
  "how.attention.query": "The query at {position}: its weights over the {n} positions, the largest shown per head ({heads} heads), each head's adding up to one:",
  "how.attention.head_key": "head {h} · {key}",
  "how.attention.alpha_fit": "The network's attention weights ({what}) do not fit the page's heads × queries × keys {shape}: the page cannot compare them, so this step is not counted as within.",
  "how.attention.pooled": "The layer then takes the mean of its {n} positions' outputs.",
  "how.attention_pool.lead": "One learned query weighs the positions: each state is scored through a small tanh layer, the scores become weights that add up to one, and the output is the states' weighted sum:",
  "how.attention_pool.formula": "α = softmaxₜ(q · tanh(W·hₜ + b)) ; y = Σₜ αₜ·hₜ",
  "how.attention_pool.weights": "The weights αₜ over the {n} positions, the largest shown:",
  "how.attention_pool.sum": "Channel {c} of the output is Σₜ αₜ·hₜ over the {n} positions:",
  "how.transformer.lead": "An encoder layer lets every token attend to every other, then passes each token through a small feed-forward network; both parts add their result back:",
  "how.transformer.formula": "x₁ = x + MHA(LN₁ x) ; y = x₁ + W₂·f(W₁·LN₂ x₁ + b₁) + b₂",
  "how.transformer.parts": "At {position}, channel {c}: the input {x}, plus what attention added {a}, plus what the feed-forward network added {f}, gives {y}.",
  "how.transformer.post_norm": "This layer normalises after each part rather than before: its sums are normalised again, so the three numbers above do not simply add up.",
  "how.tokenizer.lead": "Every input becomes a token of {units} numbers, and a learned [cls] token is put first:",
  "how.tokenizer.formula": "t₀ = [cls] ; tⱼ = xⱼ·wⱼ + bⱼ ; tₖ = Eₖ[idₖ]",
  "how.tokenizer.cls": "The token {k} is learned: the same for every {row.one}. Channel {c}:",
  "how.tokenizer.num": "The token of {k} is its value {x} times a learned vector, plus a learned bias. Channel {c}:",
  "how.tokenizer.nom": "The token of {k} is row {id} of its table. Channel {c}:",

  // ---- graph convolutions and the readout (explainers/graph.js)
  "how.graph.lead": "Each {node.one} gathers the messages its {edge.many} bring and combines them through one shared set of weights:",
  "how.graph.formula": "h′ᵢ = Σ_j cⱼᵢ · W·hⱼ + b",
  "how.graph.self": "itself",
  "how.graph.root": "itself, through its own weights",
  "how.graph.from": "{node} · hop {hop}",
  "how.graph.msg": "c {coef} · size {size} ·",
  "how.graph.messages": "The {n} messages into this {node.one}, the {shown} that carried most of the target first (their shares of it):",
  "how.graph.messages_into": "The {n} messages into {node}, the {shown} that carried most of the target first (their shares of it):",
  "how.graph.sum": "{what}: the sum of its messages plus the bias, recomputed here from every {node.one}'s numbers and the layer's weights:",
  "how.graph.normalised": "The layer then divides each {node.one}'s vector by its length.",
  "how.graph.node_only": "The answer brings the numbers of {node} itself, not its neighbours': the page recomputes the message it sends itself and checks that the messages' shares add up to the output's.",
  "how.graph.shares": "Along the explained target the messages carry {sum} and the bias {b}: {page} in all, where the output itself carries {network}.",
  "how.graph.self_size": "the size of its own message",
  "how.graph.share": "the messages' shares, with the bias",
  "how.graph.gin": "The messages add up to the {n} numbers the small network reads; the page runs it on them:",
  "how.graph.pool.lead": "All the {node.many} of one {graph.one} are merged into one vector, channel by channel:",
  "how.graph.pool.formula": "h_G = Σᵢ hᵢ, their mean, or their maximum",
  "how.graph.pool.sum": "Channel {c} is the sum over the {n} {node.many}:",
  "how.graph.pool.mean": "Channel {c} is the mean over the {n} {node.many}:",
  "how.graph.pool.max": "Channel {c} is the largest over the {n} {node.many}; the {node.one} in row {row} holds it:",

  // ---- a latent code (explainers/latent.js)
  "how.latent.lead": "The layer before produced two halves, the means μ and the log-variances of a code of {z} numbers; while answering the code is the mean itself:",
  "how.latent.formula": "μ, log σ² = split(h) ; z = μ",
  "how.latent.split": "The {n} numbers read are split in two: the first {z} are the means μ, the last {z} the log-variances log σ², clipped to ±{clip}.",
  "how.latent.unit": "Code unit {j}: μ = {mu}, log σ² = {logvar}, so σ = {sigma}.",
  "how.latent.code": "code unit {j} = μ",
  "how.latent.sigma": "its spread σ",
  "how.latent.kl_row": "the KL term",
  "how.latent.kl": "In training the code was drawn around μ with spread σ; how far that range lies from the standard one, the KL term, is {kl} for this {row.one}.",
});

export const DYNAMIC = Object.freeze({
  "how.recurrent.formula.": Object.freeze(["rnn", "gru", "lstm"]),
  "how.recurrent.gate.": Object.freeze(["r", "z", "n", "i", "f", "g", "o"]),
  "how.recurrent.return.": Object.freeze(["last", "all", "mean", "max"]),
  "how.tokenizer.": Object.freeze(["cls", "num", "nom"]),
  "how.graph.pool.": Object.freeze(["sum", "mean", "max"]),
});
