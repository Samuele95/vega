// The code room (D4 §6.3, P9; R-D7, R-E8), off the tour: opened with c, from the index, or by a "see
// it in the code" link at a walk step. Three parts:
//
//   the files   GET v1/code, vega/code-view@2: the files this process runs, read by the service from
//               where it runs them, in its order, each with its role (code.role.<role>), what that
//               role does (code.about.<role>), its kind, its digest and its text; the verified line
//               when the service checked every file against the manifest at start-up
//   the walk    page.json#walk (page_config.build_walk: R-E8's ids — root, store, build, one per
//               block type in forward order, forward, link, shape, run, explain, route —, each
//               step's title and text from the backend's explain catalog, and its code anchor
//               {file, at[], after, to}), resolved against the served text: the step is in the first
//               file whose path matches `file` and that holds an `at` line (from the `after` line
//               on, when there is one); it shows the definition that line opens (its extent, the
//               highlighter's), run on to a `to` definition just below. A step whose file or line is
//               not served drops out of the walk instead of pointing at a wrong line, and the lead
//               says how many did.
//   the notes   story.code_notes[<walk id>] = {note, link: {href, kind: report|file} | null,
//               ref: {kind, number, page, title} | null} (R-E3): under the step's text and as §
//               lines above its code. The label is composed from report.kind.<kind> with the
//               report's number and title (number and page are story values keys); a link opens the
//               report at its page or the stage brief it names (only the slot's own files are
//               linked: story-2's href pattern).
//
// The highlighter is page.json#code.language's (HIGHLIGHTERS; plain.js for any other language),
// loaded lazily, once. A bundle packed before v2.3 answers v1/code in v2.2's form (no schema; roles
// in words, an English `about`, `verified` a sentence): legacyView reads it until the parity probe
// runs on a v2.3 bundle. The probe reads the room through window.__showcase.codeWalk().
import { $, el, reduced } from "../core/dom.js";
import { S } from "../core/state.js";
import { URLS, api } from "../core/api.js";
import { fmtInt, pad2 } from "../core/fmt.js";
import { t, tText, tn, data, storyText, current } from "../core/text.js";
import { extendProbe } from "../core/probe.js";
import { viewState } from "./shared.js";

export const CODE_VIEW = "vega/code-view@2";
export const ROLES = Object.freeze(["Model", "FeatureTransformer", "Predictor", "ModelStore", "Monitor", "Inspection", "support",
  "composition_root", "http_entry", "ports", "domain"]);
export const KINDS = Object.freeze(["adapter", "support", "entrypoint"]);
export const HIGHLIGHTERS = Object.freeze({ python: "python.js" });
export const PLAIN = "plain.js";
export const HIGHLIGHT_MEMBERS = Object.freeze(["language", "tokens", "extent"]);
// the files a note may link: the report (at a page) and the stage briefs the slot serves
export const NOTE_HREF = /^showcase\/(?:report\/report\.pdf(?:#page=\d+)?|briefs\/([a-z-]+)-brief\.md)$/;
// a step runs on to its `to` definition when that starts fewer lines than this below its own end
const RUN_ON = 12;
const NOTE_WIDTH = 62;
const NOTE_MIN = 36;
// v2.2's role words, snake-cased
const LEGACY_ROLE = Object.freeze({ composition_root: "composition_root", http_entry_point: "http_entry" });

// A state of the room the reader sees instead of the code: a strings key and its params.
export class CodeRoomError extends Error {
  constructor(key, params = {}) {
    super(key);
    Object.assign(this, { name: "CodeRoomError", key, params });
  }
}

// ---- the payload ---------------------------------------------------------------------------------
// A role id as a strings key segment: ModelStore → model_store, composition_root as it is.
export const roleKey = (role) => String(role).replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();

function fileOf(f) {
  if (!f || typeof f.path !== "string" || typeof f.text !== "string") throw new CodeRoomError("code.malformed", { why: tText("code.malformed.file") });
  return Object.freeze({ path: f.path, role: typeof f.role === "string" ? f.role : null, kind: KINDS.includes(f.kind) ? f.kind : null,
    digest: typeof f.digest === "string" ? f.digest : null, text: f.text, lines: Object.freeze(f.text.split("\n")) });
}

// {verified, language, files: [{path, role, kind, digest, text, lines}]} of a v1/code answer.
export function codeView(payload) {
  if (!payload || typeof payload !== "object" || !Array.isArray(payload.files)) throw new CodeRoomError("code.malformed", { why: tText("code.malformed.files") });
  if (payload.schema === undefined) return legacyView(payload);
  if (payload.schema !== CODE_VIEW) throw new CodeRoomError("app.schema_mismatch", { doc: data(URLS.code), got: data(String(payload.schema)), want: data(CODE_VIEW) });
  return Object.freeze({ verified: payload.verified === true, language: typeof payload.language === "string" ? payload.language : null,
    files: Object.freeze(payload.files.map(fileOf)) });
}

// v2.2's answer (a bundle packed before v2.3): the roles in words mapped to the ids, the ports and
// the value objects by their module, and `verified` the sentence the service wrote after hashing
// every file against the manifest.
function legacyView(payload) {
  const role = (f) => {
    const id = String(f.role || "").trim().replace(/\s+/g, "_");
    if (ROLES.includes(id)) return id;
    if (LEGACY_ROLE[id.toLowerCase()]) return LEGACY_ROLE[id.toLowerCase()];
    return f.module === "domain" ? (/(^|\/)ports\.py$/.test(String(f.path)) ? "ports" : "domain") : "support";
  };
  return Object.freeze({ verified: !!payload.verified, language: "python",
    files: Object.freeze(payload.files.map((f) => fileOf({ path: f && f.path, text: f && f.text, digest: f && f.digest, role: f ? role(f) : null }))) });
}

// ---- the walk ------------------------------------------------------------------------------------
function pattern(src) {
  try { return typeof src === "string" ? new RegExp(src) : null; } catch { return null; }
}

// One walk step in the first served file that holds it: {id, title, text, path, range: [from, to]}
// (line indices), or {id, miss} — "pattern" (an anchor that does not compile), "file" (no served
// path matches) or "anchor" (no matching file holds the line).
export function resolveStep(w, files, extent) {
  const c = (w && w.code) || {}, file = pattern(c.file), at = (Array.isArray(c.at) ? c.at : []).map(pattern);
  const after = c.after ? pattern(c.after) : null, to = c.to ? pattern(c.to) : null;
  if (!file || !at.length || at.includes(null) || (c.after && !after) || (c.to && !to)) return { id: w.id, miss: "pattern" };
  const served = files.filter((f) => file.test(f.path));
  for (const f of served) {
    const lines = f.lines, from = after ? lines.findIndex((l) => after.test(l)) : 0;
    if (from < 0) continue;
    const i = at.map((r) => lines.findIndex((l, k) => k >= from && r.test(l))).find((k) => k >= 0);
    if (i === undefined) continue;
    const range = extent(lines, i);
    const j = to ? lines.findIndex((l, k) => k > range[1] && to.test(l)) : -1;
    if (j > range[1] && j - range[1] < RUN_ON) range[1] = extent(lines, j)[1];
    return { id: w.id, title: String(w.title || ""), text: String(w.text || ""), path: f.path, range };
  }
  return { id: w.id, miss: served.length ? "anchor" : "file" };
}

// page.json#walk against the served files: {steps (in the walk's order), missing}.
export function resolveWalk(walk, files, extent) {
  const all = (walk || []).map((w) => resolveStep(w, files, extent));
  return { steps: all.filter((s) => !s.miss), missing: all.filter((s) => s.miss) };
}

// ---- the highlighter -----------------------------------------------------------------------------
const importHighlighter = (file) => import(`./highlight/${file}`).then((m) => m.default);

function held(mod, file) {
  const missing = HIGHLIGHT_MEMBERS.filter((m) => !mod || !(m in mod));
  if (missing.length) throw new Error(`${file}: ${missing.join(", ")}`);
  return mod;
}

// {hl, gap}: the language's highlighter, or plain.js for a language without one; a language's own
// module that does not load falls back to plain.js, and `gap` says which and why (for the probe).
export async function loadHighlighter(language, importer = importHighlighter) {
  const file = HIGHLIGHTERS[language] || PLAIN;
  try {
    return { hl: held(await importer(file), file), gap: null };
  } catch (e) {
    if (file === PLAIN) throw e;
    return { hl: held(await importer(PLAIN), PLAIN), gap: { file, error: String((e && e.message) || e) } };
  }
}

// ---- the notes -----------------------------------------------------------------------------------
// The story's note of a walk step: {note, ref, link, stage}, the link kept only when its href is one
// of the slot's files (the stage of a brief link), or null.
export function noteOf(story, id) {
  const n = story && story.code_notes ? story.code_notes[id] : null;
  if (!n || typeof n.note !== "string") return null;
  const link = n.link && typeof n.link.href === "string" && NOTE_HREF.test(n.link.href) ? n.link : null;
  return { note: n.note, ref: n.ref && n.ref.kind ? n.ref : null, link, stage: link ? NOTE_HREF.exec(link.href)[1] || null : null };
}

// A report fact of the story (a number, or {value}) as the report writes it; its key when absent.
function reportValue(values, key) {
  const v = values ? values[key] : undefined, x = v && typeof v === "object" && "value" in v ? v.value : v;
  return x === undefined || x === null ? String(key) : String(x);
}

// The label of a note: its report reference (report.kind.<kind>: the report's number and title),
// else the brief it links to, else the report; null when it has neither.
export function noteLabel(n, values) {
  if (n.ref) {
    const object = t(`report.kind.${n.ref.kind}`, { number: data(reportValue(values, n.ref.number)), title: data(n.ref.title || "") });
    const c = n.ref.chapter;
    // where the object sits: "§n Title · Figure k" (the stage chapter's number and title)
    return c ? t("report.in_chapter", { number: data(reportValue(values, c.number)), title: data(c.title || ""), object }) : object;
  }
  if (n.stage) return t("code.note.brief", { stage: data(n.stage) });
  return n.link && n.link.kind === "report" ? t("code.note.the_report") : null;
}

// The label as a link to the note's file (a new tab), titled with the report's page or the file.
function linked(n, label, values) {
  if (!n.link) return label;
  const page = n.link.kind === "report" && n.ref ? reportValue(values, n.ref.page) : null;
  const title = page ? tText("code.note.open", { page: data(page) }) : tText("code.note.open_file", { file: data(n.link.href.split("/").pop().split("#")[0]) });
  return el("a", { href: n.link.href, target: "_blank", rel: "noopener", title }, label);
}

// The note under a step's text: "In the report: <label> — <note>." ("In <label>: …" for a file, "Note:
// …" without a label); the note is story text (its values and lexicon words resolved).
export function noteLine(n, values) {
  const label = noteLabel(n, values), note = storyText(n.note, el("span"));
  if (!label) return t("code.note.plain", { note });
  return t(n.link && n.link.kind === "file" ? "code.note.file" : "code.note.report", { label: linked(n, label, values), note });
}

export function wrapWords(text, width) {
  const out = [];
  let line = "";
  for (const w of String(text).split(/\s+/).filter(Boolean)) {
    if (line && (line + " " + w).length > width) { out.push(line); line = w; } else line = line ? `${line} ${w}` : w;
  }
  if (line) out.push(line);
  return out;
}

// The § lines of a note above the code it explains, at the code's indentation: the label line
// (linked), then the note as comment lines.
export function noteRows(n, pad, on, values) {
  const row = (...kids) => el("div", { class: on ? "ln note on" : "ln note" }, el("i", {}, "§"), el("span", {}, pad, ...kids));
  const label = noteLabel(n, values), rows = [];
  if (label) rows.push(row(t(n.link && n.link.kind === "file" ? "code.src.file" : "code.src.report", { label: linked(n, label, values) })));
  const text = storyText(n.note, el("span")).textContent;
  for (const line of wrapWords(text, Math.max(NOTE_MIN, NOTE_WIDTH - pad.length))) rows.push(row(`#   ${line}`));
  return rows;
}

// ---- words ---------------------------------------------------------------------------------------
// A walk step's title or text: the explain catalog's prose, its lexicon words resolved.
const walkWords = (str) => current().render(String(str), {}, true, current().pseudo);
const walkText = (str) => current().render(String(str), {}, false, current().pseudo);
const storyValues = () => (S.story && S.story.values) || {};
const known = (role) => ROLES.includes(role);
export const roleLabel = (role) => (known(role) ? tText(`code.role.${roleKey(role)}`) : data(role || ""));
export const aboutText = (role) => (known(role) ? tText(`code.about.${roleKey(role)}`) : "");
export const kindLabel = (kind) => (KINDS.includes(kind) ? tText(`code.kind.${kind}`) : null);
const joined = (parts) => parts.filter(Boolean).flatMap((p, i) => (i ? [" ", p] : [p]));

// ---- the room ------------------------------------------------------------------------------------
const state = () => viewState("code", { view: null, hl: null, gap: null, walk: { steps: [], missing: [] }, at: 0, file: null,
  want: null, loading: null, error: null, sources: new Map() });

async function read(st) {
  $("code-step").replaceChildren(el("p", { class: "hint" }, t("code.reading")));
  let payload;
  try { payload = await api(URLS.code); } catch (e) {
    throw e && e.status === 404 ? new CodeRoomError("code.absent") : new CodeRoomError("code.unavailable", { why: data((e && e.message) || String(e)) });
  }
  const view = codeView(payload), language = (S.page && S.page.code && S.page.code.language) || view.language;
  const { hl, gap } = await loadHighlighter(language);
  Object.assign(st, { view, hl, gap, walk: resolveWalk(S.page ? S.page.walk : [], view.files, hl.extent), error: null });
}

function fileButton(st, f) {
  return Object.assign(el("button", { "data-path": f.path, title: aboutText(f.role) }, data(f.path), el("span", { class: "role" }, roleLabel(f.role))),
    { onclick: () => { st.at = -1; st.file = f.path; draw(st); inView(); } });
}

// The parts drawn once the files are read: the lead, the verified line, the files, the walk.
function layout(st) {
  const { view, walk } = st, lines = view.files.reduce((s, f) => s + f.lines.length, 0);
  $("code-lead").replaceChildren(...joined([tn("code.lead", view.files.length, { lines: fmtInt(lines) }),
    walk.steps.some((s) => noteOf(S.story, s.id)) ? t("code.lead.notes") : null,
    walk.missing.length ? tn("code.lead.missing", walk.missing.length) : null]));
  $("code-verified").replaceChildren(...(view.verified ? [el("span", { class: "ok" }, "✓ "), t("code.verified")]
    : [el("span", { class: "code-unverified" }, t("code.unverified"))]));
  $("code-files").replaceChildren(...view.files.map((f) => fileButton(st, f)));
  $("walk").replaceChildren(...walk.steps.map((s, k) => el("li", {}, Object.assign(
    el("button", {}, el("span", { class: "k" }, pad2(k + 1)), walkWords(s.title)), { onclick: () => { st.at = k; draw(st); inView(); } }))));
  $("code-prev").onclick = () => step(st, -1);
  $("code-next").onclick = () => step(st, 1);
}

function step(st, d) {
  const n = st.walk.steps.length;
  if (!n) return;
  st.at = st.at < 0 ? 0 : (st.at + d + n) % n;
  draw(st);
}

// on a narrow screen the walk sits above the code: a pick scrolls down to what it opened
function inView() {
  if (matchMedia("(max-width:1000px)").matches) $("code-step").scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "start" });
}

function stepPanel(s) {
  const n = noteOf(S.story, s.id);
  return el("div", { class: "enter" }, el("h2", {}, walkWords(s.title)), el("p", {}, walkWords(s.text)),
    n ? el("p", { class: "code-ref" }, noteLine(n, storyValues())) : null);
}

function filePanel(f) {
  return el("div", { class: "enter" }, el("h2", {}, data(f.path)), el("p", {}, aboutText(f.role)));
}

function head(f, s) {
  return [el("span", { class: "path" }, data(f.path)),
    el("span", { class: "caps muted" }, [roleLabel(f.role), kindLabel(f.kind)].filter(Boolean).join(" · ")),
    el("span", { class: "caps muted" }, s ? t("code.lines.range", { from: s.range[0] + 1, to: s.range[1] + 1, n: f.lines.length })
      : tn("code.lines.count", f.lines.length)),
    f.digest ? el("span", { class: "caps muted", title: data(f.digest) }, "✓ ", data(f.digest.replace(/^sha256:/, "").slice(0, 12))) : null];
}

// A file's lines, highlighted once and kept: {code, rows}.
function sourceOf(st, f) {
  if (!st.sources.has(f.path)) {
    const rows = st.hl.tokens(f.text).map((runs, k) => el("div", { class: "ln" }, el("i", {}, String(k + 1)),
      el("span", {}, ...(runs.length ? runs.map(([cls, s]) => (cls ? el("span", { class: cls }, s) : s)) : [" "]))));
    st.sources.set(f.path, { code: el("code"), rows });
  }
  return st.sources.get(f.path);
}

// The § lines of every step of this file that has a note, by the line they stand above.
function notesAt(st, f) {
  const at = new Map(), values = storyValues();
  st.walk.steps.forEach((s, k) => {
    const n = s.path === f.path ? noteOf(S.story, s.id) : null;
    if (!n) return;
    const line = s.range[0], pad = f.lines[line].match(/^\s*/)[0];
    at.set(line, [...(at.get(line) || []), ...noteRows(n, pad, k === st.at, values)]);
  });
  return at;
}

function drawSource(st, f, s) {
  const src = $("src"), shown = sourceOf(st, f), notes = notesAt(st, f);
  shown.rows.forEach((r, k) => r.classList.toggle("hl", !!s && k >= s.range[0] && k <= s.range[1]));
  shown.code.replaceChildren(...shown.rows.flatMap((r, k) => [...(notes.get(k) || []), r]));
  if (src.firstChild !== shown.code) src.replaceChildren(shown.code);
  const anchor = s ? shown.code.querySelector(".ln.note.on") || shown.rows[s.range[0]] : null;
  src.scrollTo({ top: anchor ? Math.max(0, anchor.offsetTop - 24) : 0, behavior: anchor && !reduced() ? "smooth" : "auto" });
}

function nav(st, s) {
  const steps = st.walk.steps, n = steps.length;
  $("code-nav").hidden = !n;
  $("code-n").textContent = s ? `${st.at + 1} / ${n}` : "";
  $("code-prev").hidden = !s || st.at === 0;
  $("code-next").textContent = !s ? tText("code.walk.start") : st.at === n - 1 ? tText("code.walk.again")
    : tText("code.walk.next", { title: walkText(steps[st.at + 1].title) });
}

// The step on show (or, after a file was picked, the file alone), its code and its note.
function draw(st) {
  const s = st.at >= 0 ? st.walk.steps[st.at] || null : null;
  const f = st.view.files.find((x) => x.path === (s ? s.path : st.file)) || st.view.files[0];
  nav(st, s);
  if (!f) {
    ["code-step", "code-head", "src"].forEach((id) => $(id).replaceChildren());
    return;
  }
  st.file = f.path;
  [...$("walk").querySelectorAll("button")].forEach((b, k) => b.classList.toggle("on", k === st.at));
  [...$("code-files").children].forEach((b) => b.classList.toggle("on", b.dataset.path === f.path));
  $("code-step").replaceChildren(s ? stepPanel(s) : filePanel(f));
  $("code-head").replaceChildren(...head(f, s));
  drawSource(st, f, s);
}

function drawError(e) {
  $("code-nav").hidden = true;
  $("code-step").replaceChildren(el("p", { class: "err" }, e instanceof CodeRoomError ? t(e.key, e.params) : t("code.unavailable", { why: data(e.message) })));
}

// Reads the files once (again after a failure), then shows the step asked for, or where the reader was.
export async function show() {
  const st = state();
  $("code-back").onclick = () => S.tour.back();
  if (!st.view && !st.loading) {
    st.loading = read(st).then(() => layout(st), (e) => {
      st.error = e;
      if (!(e instanceof CodeRoomError)) console.error(e);
    }).finally(() => { st.loading = null; });
  }
  if (st.loading) await st.loading;
  if (!st.view) return drawError(st.error);
  if (st.want) {
    const k = st.walk.steps.findIndex((x) => x.id === st.want);
    if (k >= 0) st.at = k;
    st.want = null;
  }
  draw(st);
}

// Opens the room at a walk step (a block type, "shape", "link", …), or where the reader left it.
export function openCode(stepId = null) {
  state().want = stepId || null;
  if (S.tour.current === "code") show(); else S.tour.show("code");
}

export function key(e) {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  e.preventDefault();
  const st = state();
  if (st.view) step(st, e.key === "ArrowRight" ? 1 : -1);
}

// What the probe reads: the steps and their lines, the steps left out and why, the notes' links.
export function walkReading() {
  const st = S.views && S.views.code;
  if (!st || !st.view) return { read: false, error: st && st.error ? st.error.key || String(st.error.message) : null };
  const links = st.walk.steps.map((s) => [s.id, noteOf(S.story, s.id)]).filter(([, n]) => n && n.link);
  return { read: true, language: st.hl.language, gap: st.gap, at: st.at,
    steps: st.walk.steps.map((s) => ({ id: s.id, path: s.path, from: s.range[0] + 1, to: s.range[1] + 1 })),
    missing: st.walk.missing.map((m) => ({ id: m.id, why: m.miss })),
    links: links.map(([id, n]) => ({ step: id, href: n.link.href, kind: n.link.kind })) };
}

// The "see it in the code" links of the other views open the room at their walk step.
export function bind() {
  document.querySelectorAll(".code-link[data-code]").forEach((b) => b.addEventListener("click", () => openCode(b.dataset.code)));
  extendProbe("codeWalk", walkReading, { report: true });
}

export const view = Object.freeze({ id: "code", show, key });
