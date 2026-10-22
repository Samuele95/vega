// A latent step (D4 §8.1, latent; explain-catalog.json#recompute.latent_mean): the layer before
// produced 2·z numbers, which the reparameterisation splits in two — the means μ (the first z) and
// the log-variances log σ² (the last z, clipped to [−10, 10] as blocks.py clips them). While the
// network answers the code is μ itself: no noise is drawn, so the same row always gets the same
// answer. The page splits what the stage read, shows one code unit — the one that carried most of
// the explained target, or the one picked — with its μ and its spread σ = exp(½·log σ²), beside the
// network's code, and recomputes the KL term −½·Σ(1 + log σ² − μ² − σ²) the network trained with,
// beside the one the answer reports.
import { el } from "../../core/dom.js";
import { t, tText } from "../../core/text.js";
import { plain } from "../../core/fmt.js";
import { Answer, reducedValues } from "../answer.js";
import { rankUnits, gapOf, checkRows, stageParams, stageProse, stageFormula } from "../sum.js";
import { sizesOf, fieldsOf, extraOf, stopped, checkOf, worstOf, cellPicks } from "../cells.js";

export const LOGVAR_CLIP = 10;

// μ, log σ², σ and the KL term of what a reparameterisation read (2·z numbers).
export function split(x, z) {
  const mu = Array.from(x.slice(0, z)), logvar = Array.from(x.slice(z, 2 * z), (v) => Math.min(LOGVAR_CLIP, Math.max(-LOGVAR_CLIP, v)));
  const sigma = logvar.map((v) => Math.exp(0.5 * v));
  const kl = -0.5 * mu.reduce((s, m, j) => s + 1 + logvar[j] - m * m - Math.exp(logvar[j]), 0);
  return { mu, logvar, sigma, kl };
}

export function latentValues(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage), { have, missing } = fieldsOf(A, graph, n.id, ["in", "out"]);
  const z = Number(n.payload.z ?? n.config.z ?? (n.out_shape || [])[0]), base = { A, n, z, x: have.in, y: have.out, R: A.unitRelevance(n.id) };
  if (missing) return { ...base, why: "missing", what: missing };
  if (base.x.length !== 2 * z || base.y.length !== z) return { ...base, why: "fit", what: sizesOf(n.id, { in: base.x, out: base.y }) };
  const kl = reducedValues(extraOf(A, graph, n.id, "kl")), sigma = reducedValues(extraOf(A, graph, n.id, "sigma"));
  return { ...base, ...split(base.x, z), klNet: kl ? kl[0] : null, sigmaNet: sigma && sigma.length === z ? sigma : null };
}

function recompute(step, ex, graph) {
  const v = latentValues(step, ex, graph);
  if (v.why) return null;
  const j = rankUnits(v.R, v.y, step.pick).j;
  return checkOf({ ...gapOf(v.mu[j], v.y[j]), what: "mu", unit: j },
    [worstOf("code", v.mu, v.y), v.sigmaNet ? worstOf("sigma", v.sigma, v.sigmaNet) : null, v.klNet === null ? null : { what: "kl", ...gapOf(v.kl, v.klNet) }]);
}

function render(step, ex, env) {
  const v = latentValues(step, ex, env.graph), n = v.n, params = stageParams(n);
  const head = [stageProse(n, "how.latent.lead", params), stageFormula(n, "how.latent.formula", params)];
  if (v.why) return el("div", {}, ...head, stopped(v.why, { what: v.what, stage: env.nameOf(step), shape: n.in_shape }));
  const { rank, j, R } = rankUnits(v.R, v.y, step.pick), unit = (u) => tText("how.adv.unit", { u: u + 1 });
  const rows = [{ name: tText("how.latent.code", { j: j + 1 }), page: v.mu[j], network: v.y[j] },
    { name: tText("how.latent.sigma", { j: j + 1 }), page: v.sigma[j], network: v.sigmaNet ? v.sigmaNet[j] : null },
    { name: tText("how.latent.kl_row"), page: v.kl, network: v.klNet }];
  return el("div", {}, ...head, cellPicks({ rank, R, on: j, target: env.target, choose: env.choose, nameOf: unit }),
    el("p", {}, t("how.latent.split", { n: v.x.length, z: v.z, clip: LOGVAR_CLIP })),
    el("p", {}, t("how.latent.unit", { j: j + 1, mu: plain(v.mu[j], 3), logvar: plain(v.logvar[j], 3), sigma: plain(v.sigma[j], 3) })),
    checkRows(rows), el("p", { class: "hint" }, t("how.latent_off")), el("p", { class: "hint" }, t("how.latent.kl", { kl: plain(v.kl, 3) })));
}

export default {
  kinds: ["latent"],
  render,
  recompute,
  unit: (step, ex, graph) => { const v = latentValues(step, ex, graph); return v.y ? rankUnits(v.R, v.y, step.pick).j : null; },
};
