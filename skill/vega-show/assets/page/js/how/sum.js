// The stepper's number blocks (D4 §8.1; the hand-built page's sumBlock, 1792–1812). A weighted sum
// z = Σ w·s + b is shown as its largest terms, the rest added up, the bias, and the total: the
// network's own z. The page adds the terms itself from the full-precision weights it was sent, and
// whatever its sum and the network's z differ by is shown as a row of its own, the precision gap —
// never hidden. The same rule holds for every recomputed value (checkRows): the page's number, the
// network's, and the gap when there is one.
//
// sumParts, gapOf, gapShown and rankUnits are pure (tests/sum.test.js); the blocks build DOM.
import { el } from "../core/dom.js";
import { termRow } from "../core/bars.js";
import { t, tText, tnText, hasKey } from "../core/text.js";
import { sg, plain, subs } from "../core/fmt.js";

export const TOP_TERMS = 6;
// A gap is shown when it passes half a percent of the value (at least of 1): below that the page's
// sum and the network's agree to the digits shown.
export const GAP_SHOWN = 0.005;
// The probe's tolerance for a recompute (D4 §17.5, BUILD-PLAN W14): max(1e-3, 1e-3·|network|).
export const GAP_WITHIN = 1e-3;

// Whether a gap between the page's value and the network's is worth its own row.
export const gapShown = (gap, z) => Math.abs(gap) > GAP_SHOWN * Math.max(1, Math.abs(z));

// The gap of a recompute: network − page, with the verdict the probe reads.
export function gapOf(page, network) {
  const gap = network - page;
  return { page, network, gap, within: Math.abs(gap) <= Math.max(GAP_WITHIN, GAP_WITHIN * Math.abs(network)) };
}

// The worst of several recomputes (the largest |gap|), or null when there is none.
export function worstOf(checks) {
  const ok = checks.filter(Boolean);
  return ok.length ? ok.reduce((w, c) => (Math.abs(c.gap) > Math.abs(w.gap) ? c : w), ok[0]) : null;
}

// One weighted sum, unit j of the matrix m ({out, in, w} row-major) over the source vector src:
// the terms w[j][i]·src[i], the largest `k` of them by magnitude (their indices, in order), the
// rest added up, the bias, the page's own total, the network's z and the gap between them.
// opts: {bias (the bias vector or null), z (the network's value, null: the page's), k}
export function sumParts(m, j, src, { bias = null, z = null, k = TOP_TERMS } = {}) {
  const n = m.in, w = (i) => m.w[j * n + i];
  const terms = Array.from({ length: n }, (_, i) => w(i) * src[i]);
  const order = terms.map((_, i) => i).sort((a, b) => Math.abs(terms[b]) - Math.abs(terms[a]));
  const K = Math.min(k, n), top = order.slice(0, K), restIdx = order.slice(K);
  const rest = restIdx.reduce((s, i) => s + terms[i], 0), b = bias ? bias[j] : 0;
  const total = terms.reduce((s, v) => s + v, 0) + b, network = z === null || z === undefined ? total : z;
  const gap = network - total;
  const many = n > K && Math.abs(rest) > top.reduce((s, i) => s + Math.abs(terms[i]), 0);
  return { n, K, terms, top, rest, restCount: restIdx.length, bias: b, hasBias: !!bias, total, z: network, gap,
           shown: gapShown(gap, network), many, weight: w, src: (i) => src[i] };
}

// The block of one sum (DOM): the top terms as signed bars "w × s = term", the rest, the bias, the
// gap row when it shows, and the total "= z". nameOf(i) names source i; mark: the source to
// highlight (the unit picked on the step before), or -1; biasName: the bias row's words.
export function sumBlock(parts, { nameOf, mark = -1, biasName = null }) {
  const P = parts, mx = Math.max(1e-9, ...P.top.map((i) => Math.abs(P.terms[i])), Math.abs(P.rest), Math.abs(P.bias), Math.abs(P.gap), Math.abs(P.z));
  const bars = el("div", { class: "bars diverge eq" });
  P.top.forEach((i, k) => bars.append(termRow({ name: nameOf(i), v: P.terms[i], max: mx, i: k, cls: i === mark ? "sel" : "",
    val: [el("span", { class: "wx" }, `${sg(P.weight(i), 3)} × ${plain(P.src(i))} = `), sg(P.terms[i])] })));
  if (P.restCount) bars.append(termRow({ name: tnText("how.sum.rest", P.restCount), v: P.rest, max: mx, val: [sg(P.rest)], i: P.K }));
  if (P.hasBias) bars.append(termRow({ name: biasName || tText("how.sum.bias"), v: P.bias, max: mx, val: [sg(P.bias)], i: P.K + 1 }));
  if (P.shown) bars.append(termRow({ name: tText("how.sum.gap"), v: P.gap, max: mx, val: [sg(P.gap)], i: P.K + 2, cls: "round" }));
  bars.append(termRow({ name: tText("how.sum.z"), v: P.z, max: mx, val: [`= ${sg(P.z, 3)}`], i: P.K + 3, cls: "total" }));
  const notes = [P.many ? tText("how.sum.many", { rest: P.restCount, k: P.K }) : null, tText(P.shown ? "how.sum.coarse" : "how.sum.exact")];
  return el("div", {}, bars, el("p", { class: "hint" }, notes.filter(Boolean).join(" ")));
}

// Rows "what — recomputed here — the network's own — gap" for values the page recomputed: rows is
// [{name, page, network}] (network null: the answer did not carry it). The gap cell shows only when
// the gap passes GAP_SHOWN.
export function checkRows(rows, digits = 3) {
  const grid = el("div", { class: "soft how-check" }, ...["how.check.what", "how.check.page", "how.check.network", "how.check.gap"]
    .map((k) => el("span", { class: "h" }, tText(k))));
  for (const r of rows) {
    const gap = r.network === null || r.network === undefined ? null : r.network - r.page;
    // a gap below the last digit shown reads as zero, not as a signed zero
    const shown = gap === null ? null : Math.abs(gap) < 0.5 * 10 ** -digits ? 0 : gap;
    grid.append(el("span", { class: "c", title: r.name }, r.name), el("span", {}, plain(r.page, digits)),
      el("span", {}, gap === null ? "—" : plain(r.network, digits)),
      el("span", { class: gap !== null && gapShown(gap, r.network) ? "how-gap" : "h" }, gap === null ? "—" : sg(shown, digits)));
  }
  return grid;
}

// The picks of a step: the units that carried most of the target, as buttons ("unit u  ±v"); the
// one shown is `on`. choose(u) re-renders with unit u. target: the target's words.
export function picksRow({ rank, R, on, target, choose }) {
  return el("div", { class: "picks" }, el("span", {}, t("how.picks", { target })),
    ...rank.map((u) => Object.assign(el("button", { class: u === on ? "on" : "", title: tText("how.pick_title", { target, v: sg(R[u]) }) },
      t("how.pick", { u: u + 1, v: sg(R[u]) })), { onclick: () => choose(u) })));
}

// The units of a vector ranked by |relevance| (the first `k`), and the one shown: the step's own
// pick when it has one, else the first. With no relevance, by |value|.
export function rankUnits(R, values, pick = null, k = 3) {
  const by = R || values || [];
  const rank = by.map((_, j) => j).sort((a, b) => Math.abs(by[b]) - Math.abs(by[a])).slice(0, k);
  const j = pick !== null && pick !== undefined && pick >= 0 && pick < by.length ? pick : (rank[0] ?? 0);
  return { rank, j, R: by };
}

// A line of notation (a formula, the catalog's or the table's) in the formula style.
export const formula = (node) => el("div", { class: "formula" }, node);

// The note of a step whose recompute the answer does not allow: what did not come with it.
export const noRecompute = (what) => el("p", { class: "hint how-miss" }, t("how.no_recompute", { what }));

// The tensor ids "<stage>.<field>" of the fields a stage's recompute needs that the answer did not
// carry: have is {field: values or null}, in the order to name them ("B2.in, B2.out").
export const missingOf = (id, have) => Object.keys(have).filter((f) => !have[f]).map((f) => `${id}.${f}`).join(", ");

// The block-level words of a stage (D4 §8.1): the explain catalog's prose for its block type when
// page.json carries it, else the table's `fallback` key; params fill the catalog's placeholders.
export function stageProse(n, fallback, params = {}) {
  const key = `explain.${n.type}.prose`;
  return el("p", {}, t(hasKey(key) ? key : fallback, params));
}

// The catalog's formula of a stage's block type, else the table's `fallback` key (null: none).
export function stageFormula(n, fallback = null, params = {}) {
  const key = `explain.${n.type}.formula`;
  return hasKey(key) ? formula(t(key, params)) : fallback ? formula(t(fallback, params)) : null;
}

// The placeholders a stage's catalog words take (explain-catalog.json#placeholders), read off its
// node; `extra` adds or overrides (v, the letter of what the stage reads, is x unless given; n_sub
// follows the inputs given).
export function stageParams(n, extra = {}) {
  const last = (s) => (Array.isArray(s) && s.length ? s[s.length - 1] : 0), c = n.config || {}, p = n.payload || {};
  const pick = (...vs) => vs.find((v) => v !== undefined && v !== null) ?? "";
  const out = { units: last(n.out_shape), inputs: last(n.in_shape), v: tText("how.sym.input"), kernel: pick(c.kernel, c.kernel_size, p.kernel_size),
                channels: pick(c.channels, c.out_channels, last(n.out_shape)), hidden: pick(c.hidden, p.hidden), heads: pick(c.heads, p.heads),
                layers: pick(c.layers, p.layers), z: pick(c.z, p.z), slope: pick(c.negative_slope, p.negative_slope, 0.01),
                eps: Number(pick(p.eps, c.eps, 1e-5)), p: n.dropout || Number(pick(c.p, p.p, 0)), ...extra };
  return { n_sub: subs(out.inputs), ...out };
}
