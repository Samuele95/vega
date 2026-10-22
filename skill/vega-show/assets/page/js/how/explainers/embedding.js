// An embedding step (D4 §8.1): each nominal input's value picks one row of its table — a value the
// network never saw in training the reserved last row — and the rows are laid side by side for the
// layers after it (the join computes nothing, so it is a note, or this step when a network gives it
// one). The page reads each row out of the table it was sent and holds it beside the numbers the
// network passed on.
import { el } from "../../core/dom.js";
import { t, tText, data } from "../../core/text.js";
import { Answer, reducedValues } from "../answer.js";
import { noRecompute, checkRows, gapOf, worstOf, stageParams, stageProse, stageFormula } from "../sum.js";

// The id a column's value looked up: the stage's exact input (the ids, one per table), else the
// probe's record of the column (extra[column].id or extra.id[k]).
function idOf(A, n, k, column) {
  const ids = A.values(n.id, "in");
  if (ids && ids.length > k) return Math.round(ids[k]);
  const extra = (A.node(n.id) || {}).extra || {}, own = extra[column] || (extra.columns || {})[column];
  if (own && Number.isFinite(own.id)) return own.id;
  const list = reducedValues(extra.id) || reducedValues(extra.ids);
  return list && list.length > k ? Math.round(list[k]) : null;
}

// Each table of an embedding stage, with the id this sample looked up and the row it read:
// [{column, dim, id, unseen, raw, page (the table's row), network (the stage's output slice)}].
function lookups(step, ex, graph) {
  const A = Answer.of(ex, graph), n = graph.byId(step.stage), tables = n.payload.tables || [];
  const out = A.values(n.id, "out"), raw = A.inputs().raw || {};
  let at = 0;
  return tables.map((tb, k) => {
    const E = graph.matrix(tb.table), dim = tb.dim || (E ? E.in : 0), id = idOf(A, n, k, tb.column);
    const page = E && id !== null && id >= 0 && id < E.out ? Array.from(E.w.slice(id * E.in, (id + 1) * E.in)) : null;
    const network = out && out.length >= at + dim ? out.slice(at, at + dim) : null;
    at += dim;
    return { column: tb.column, dim, id, unseen: id !== null && id === tb.unseen_row, raw: raw[tb.column], page, network, table: tb.table };
  });
}

function columnBlock(c) {
  // the raw value when the request row carries the column under the network's name
  const value = c.raw === undefined ? null : c.raw === null || c.raw === "" ? tText("how.input.empty") : data(c.raw);
  const key = c.id === null ? "how.embedding.no_id" : c.unseen ? "how.embedding.unseen" : value === null ? "how.embedding.id" : "how.embedding.row";
  const line = el("p", {}, t(key, { column: data(c.column), value: value ?? "", id: c.id ?? "" }));
  if (!c.page) return el("div", { class: "how-lookup" }, line, noRecompute(data(c.table)));
  const rows = c.page.map((v, d) => ({ name: data(`${c.column}[${d}]`), page: v, network: c.network ? c.network[d] : null }));
  return el("div", { class: "how-lookup" }, line, checkRows(rows));
}

function render(step, ex, env) {
  const n = env.graph.byId(step.stage), fallback = step.kind === "concat" ? "how.concat.lead" : "how.embedding.lead";
  const head = [stageProse(n, fallback, stageParams(n)), stageFormula(n)];
  if (step.kind === "concat") return el("div", {}, ...head);
  return el("div", {}, ...head, ...lookups(step, ex, env.graph).map(columnBlock));
}

// The worst of the looked-up numbers: the dim whose page and network values differ most.
function recompute(step, ex, graph) {
  if (step.kind === "concat") return null;
  const checks = lookups(step, ex, graph).flatMap((c) => (c.page && c.network ? c.page.map((v, d) => ({ what: `${c.column}[${d}]`, ...gapOf(v, c.network[d]) })) : []));
  const worst = worstOf(checks);
  return worst ? { ...worst, within: checks.every((c) => c.within) } : null;
}

export default { kinds: ["embedding", "concat"], render, recompute };
