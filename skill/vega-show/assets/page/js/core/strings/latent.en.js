// The strings of the latent area (strings.en.js merges every fragment into the page's one table):
// the words of the latent code's drawing (js/net/renderers/latent.js) — its column caption, the
// split of what it reads into means and log-variances, the note under it, and the caption a pointer
// on it writes. A key another fragment already defines is a load error; a variant of one
// (key@<view key>, key@<shape>) is a key of its own and may live here. DYNAMIC: dynamic prefix →
// its closed member list.
export const STRINGS = Object.freeze({
  // ---- the drawing (canvas text)
  "latent.caption": "LATENT · {k} OF {z}",
  "latent.mu": "μ",
  "latent.logvar": "log σ²",
  "latent.z_is_mu": "z = μ",
  "latent.kl": "KL {kl}",

  // ---- the caption of a pointer on it
  "latent.cap.numbers#one": "a code of {n} number",
  "latent.cap.numbers#other": "a code of {n} numbers",
  "latent.cap.number": "number {j} of the code's {z}",
  "latent.cap.split": "It reads {n} numbers: the first {z} are the means μ, the last {z} the log-variances log σ². While answering, the code is the mean itself, z = μ.",
  "latent.cap.prior": "Each line is one number of the code on an axis centred on zero. Before a {row.one} is answered it shows the standard normal the training pulls every number towards (a spread of one around zero); an answer draws the {row.one}'s mean as a bar and its spread σ as a whisker around it.",
});
export const DYNAMIC = Object.freeze({});
