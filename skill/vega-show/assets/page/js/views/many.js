// The many-at-once view (D4 §6.3): where the rows come from (the shape module: the demonstration
// sample, the shipped examples, the reader's CSV); the service's answers read by the task module (the
// sentence, the main chart, what goes beside it, the table's columns and cells, which rows it got
// wrong); the drift monitor (hidden when the service has none), with the story's note on what it
// watches. A row of the table opens in try one.
import { $, el, restart } from "../core/dom.js";
import { S } from "../core/state.js";
import { URLS, post, upload } from "../core/api.js";
import { t, tText, tnText, data, storyText } from "../core/text.js";
import { section, problemText, viewState } from "./shared.js";
import { monitorBody, uploadTruth } from "./payloads.js";
import { tryRow } from "./try.js";

const state = () => viewState("many", { asked: 0 });

// The sources of the shape module, drawn into the .sources row (a file source as a file input).
function renderSources() {
  const host = document.querySelector("#main .view[data-view=\"many\"] .sources");
  if (!host || host.dataset.done) return;
  host.dataset.done = "1";
  host.replaceChildren(...S.mods.shape.sources(S.examples, S.env).map((s) => {
    if (!s.file) return Object.assign(el("button", { id: `src-${s.id}` }, el("span", { class: "caps" }, s.title), el("small", {}, s.sub)), { onclick: () => runMany(s.run()) });
    const input = el("input", { id: s.id, type: "file", accept: ".csv,text/csv" });
    input.onchange = (e) => { const f = e.target.files[0]; if (f) runMany(s.run(f)); e.target.value = ""; };
    return el("label", {}, el("span", { class: "caps" }, s.title), el("small", {}, s.sub), input);
  }));
}

// batch: {rows, truth, history} or {file}.
async function runMany(batch) {
  const st = state(), asked = ++st.asked;
  $("many-err").textContent = ""; $("many-out").hidden = true; $("many-busy").hidden = false;
  try {
    let rows = batch.rows, truth = batch.truth || null, res, mon;
    if (batch.file) {
      res = await upload(URLS.predictCsv, batch.file);
      rows = res.rows || []; truth = uploadTruth(res); mon = res.monitor || null;
    } else {
      const body = batch.history ? { rows, history: batch.history } : { rows };
      res = await post(URLS.predict, body);
      mon = await post(URLS.monitor, monitorBody(body, truth, S.schema)).catch((e) => (e && e.status === 404 ? null : Promise.reject(e)));
    }
    if (asked !== st.asked) return;
    $("many-busy").hidden = true; $("many-out").hidden = false; restart($("many-out"));
    renderSummary(res.predictions || [], truth);
    renderTable(res.predictions || [], rows, truth);
    renderMonitor(mon);
  } catch (e) {
    if (asked !== st.asked) return;
    $("many-busy").hidden = true; $("many-err").textContent = problemText(e);
  }
}

// The summary speaks of the rows the service answered (a window row without enough history is
// answered: false, its keys null, its warning in the table); the table lists every row.
function renderSummary(all, allTruth) {
  const kept = all.map((p, i) => [p, allTruth ? allTruth[i] : null]).filter(([p]) => p && p.answered !== false);
  const preds = kept.map(([p]) => p), truth = allTruth ? kept.map(([, x]) => x) : null;
  const s = S.mods.task.summary, env = S.env;
  $("many-sentence").replaceChildren(s.sentence(preds, truth, env));
  const left = s.left(preds, truth, env), right = s.right(preds, truth, env);
  $("many-bars-title").replaceChildren(left.title || "");
  $("many-classes").replaceChildren(left.body);
  $("many-bars-cap").replaceChildren(left.caption || "");
  $("many-mix-box").hidden = !right;
  if (right) {
    const head = $("many-mix-box").firstElementChild;
    head.removeAttribute("data-t"); head.replaceChildren(right.title);
    $("many-mix").replaceChildren(right.body);
  }
}

function renderTable(preds, rows, truth) {
  const s = S.mods.task.summary, env = S.env, head = $("many-rows").closest("table").tHead.rows[0];
  head.replaceChildren(el("th", {}, "#"), ...s.columns(env).map((k) => el("th", {}, t(k))));
  $("many-rows").replaceChildren(...preds.map((p, i) => {
    const want = truth ? truth[i] : null, cells = s.cells(p, want, env), miss = p.answered !== false && s.miss(p, want, env);
    const tr = el("tr", { class: "go" }, el("td", {}, i + 1), ...cells.map((c, k) => el("td", { class: miss && k === 1 ? "miss" : "" }, c)));
    tr.onclick = () => tryRow(rows[i], truth ? truth[i] : null, i + 1);
    return tr;
  }));
  const title = $("many-table-title");
  title.removeAttribute("data-t");
  title.textContent = tText("many.table.count", { n: preds.length });
}

function renderMonitor(mon) {
  const box = $("many-mon"), toggle = document.querySelector(".disclose[data-for=\"many-mon\"]");
  const readings = (mon && mon.readings) || [];
  if (toggle) toggle.hidden = !mon;
  box.hidden = !mon;
  if (!mon) return;
  const said = (v) => (v === null || v === undefined || v === "" ? "" : data(v));
  $("many-monitor").replaceChildren(...readings.map((r) => el("tr", {}, el("td", {}, said(r.title || r.check)), el("td", {}, said(r.measure)),
    el("td", {}, r.value === null || r.value === undefined ? "—" : (+r.value).toPrecision(3)), el("td", { class: "verdict-" + r.verdict }, said(r.verdict)), el("td", { class: "muted" }, said(r.note)))));
  const moved = readings.filter((r) => r.verdict === "watch" || r.verdict === "alert").length, title = $("many-mon-title");
  title.removeAttribute("data-t");
  title.textContent = tText("many.monitor.title", { readings: tnText("many.monitor.readings", readings.length), moved });
  const note = section("many").monitor_note;
  $("many-mon-note").replaceChildren(); if (note) storyText(note, $("many-mon-note"));
}

function show() { renderSources(); }

export const view = Object.freeze({ id: "many", show });
