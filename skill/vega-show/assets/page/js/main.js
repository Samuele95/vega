// The page's boot (D4 §4.2). The test hook is imported first, so that under ?probe it hears every
// error from here on. Then: the theme and the shell's words (the lexicon's defaults until the story is
// read); the service's readiness; the documents in parallel (the schema, the examples, the network,
// the story, page.json, the attribution; all but the schema and page.json tolerated as absent);
// each held to the version the page reads; the lane from page.json, cross-checked against the
// network (a mismatch is the page's error state: the showcase was built for another network);
// the lexicon, the class roles and the words resolved with them; the stage graph; the lane's task
// and shape modules (views/registry.js); the field with the page's motif and the header's mark; the
// tour from page.json#views; and the first view.
import { startProbe, extendProbe } from "./core/probe.js";
import { $, store } from "./core/dom.js";
import { S, setState } from "./core/state.js";
import { URLS, EXPECT, api, health } from "./core/api.js";
import { applyStoredTheme, onTheme, toggleTheme } from "./core/theme.js";
import { Tour } from "./core/tour.js";
import { bindDisclosures } from "./core/disclose.js";
import { t, tText, data, applyI18n, configureText, pickLang, missingKeys, unresolved, usedKeys } from "./core/text.js";
import { Lexicon } from "./core/lexicon.js";
import { RoleMap } from "./core/roles.js";
import { Lane } from "./core/lane.js";
import { StageGraph } from "./net/graph.js";
import { Field, applyMark } from "./motif/field.js";
import { resolve, registryReading } from "./views/registry.js";
import { outputRecord, readable } from "./views/payloads.js";
import { view as start, renderKeys } from "./views/start.js";
import { view as context } from "./views/context.js";
import { view as network, bind as bindNetwork } from "./views/network.js";
import { view as tryOne, bind as bindTry } from "./views/try.js";
import { view as many } from "./views/many.js";
import { view as results } from "./views/results.js";
import { view as code, bind as bindCode, openCode } from "./views/code.js";

const VIEWS = Object.freeze({ start, context, network, try: tryOne, many, results, code });
const ALIASES = Object.freeze({ window: "many" });

// A document the service or the slot does not have (404) is absent: null; any other failure is the
// boot's. Each call names its route itself (api(URLS.x)), so the template's route scan sees it.
const absent = (e) => (e && e.status === 404 ? null : Promise.reject(e));

// The error state: the status line, the headline says why, and the tour stays at the start.
function fail(key, message) {
  $("dot").className = "dot bad"; $("status-text").textContent = tText(key);
  $("headline").replaceChildren(message); $("lead").replaceChildren(); $("start-facts").replaceChildren();
  return false;
}

// Each document is the version the page reads: page.json and the story strictly; the network and
// the examples of a bundle packed before v2.3 (no schema) are still read in their v2.2 form.
function mismatch(docs) {
  const strict = [["page", docs.page], ["story", docs.story], ["attribution", docs.attribution]];
  const lenient = [["network", docs.network], ["examples", docs.examples]];
  for (const [kind, doc] of strict) if (doc && doc.schema !== EXPECT[kind]) return { doc: kind, got: doc.schema ?? null, want: EXPECT[kind] };
  for (const [kind, doc] of lenient) if (doc && !readable(doc, EXPECT[kind])) return { doc: kind, got: doc.schema, want: EXPECT[kind] };
  return null;
}

// The stage graph of the network payload, or null (and why, for the network view): the page reads
// vega/network-view@2 only, and a payload of any other form (a v2.2 bundle's) is refused by name.
function graphOf(network) {
  if (!network || network.available === false) return { graph: null, error: network && network.why ? data(network.why) : null };
  try {
    return { graph: StageGraph.from(network), error: null };
  } catch (e) { return { graph: null, error: e.message }; }
}

// The words of the lane: the lexicon, the roles, the lane's variants, the story's values and the
// backend's explain catalog; then the shell's text slots again.
function words(docs, lane) {
  const lex = new Lexicon(docs.story ? docs.story.lexicon : null, docs.page.identity || null);
  const roles = new RoleMap(docs.story ? docs.story.class_roles : null, lane, docs.network);
  configureText({ lang: pickLang({ search: location.search }), lex, lane, values: docs.story && docs.story.values, explain: docs.page.explain,
                  classNames: docs.story && docs.story.class_names });
  applyI18n(document);
  $("crumb-title").textContent = docs.page.identity && docs.page.identity.title ? data(docs.page.identity.title) : "";
  return { lex, roles };
}

// The schema and page.json are the page's own ground (without them there is no editor and no lane);
// the examples, the network, the story and the attribution may be absent (D4 §4.1: try one then
// offers no preset, many at once the reader's table only, the network view says why).
async function read() {
  const [schema, examples, network, story, page, attribution] = await Promise.all([api(URLS.schema), api(URLS.examples).catch(absent),
    api(URLS.network).catch(absent), api(URLS.story).catch(absent), api(URLS.page), api(URLS.attribution).catch(absent)]);
  return { schema, examples, network, story, page, attribution };
}

async function boot(field) {
  applyStoredTheme();
  (store.get("seen") || "").split(",").filter(Boolean).forEach((v) => S.seen.add(v));
  let docs, lane;
  try {
    const h = await health();
    $("dot").className = "dot " + (h.ready ? "ok" : "bad");
    $("status-text").textContent = tText(h.ready ? "app.status.ready" : "app.status.not_ready");
    if (!h.ready) { $("headline").textContent = h.error ? data(h.error) : tText("app.not_ready"); return false; }
    docs = await read();
  } catch (e) { return fail("app.status.unreachable", data(e.message)); }
  const wrong = mismatch(docs);
  if (wrong) return fail("app.status.refused", t("app.schema_mismatch", { doc: data(wrong.doc), got: data(String(wrong.got)), want: data(wrong.want) }));
  try { lane = Lane.from(docs.page.lane, docs.network); } catch (e) { return fail("app.status.refused", e.message); }
  const { lex, roles } = words(docs, lane), { graph, error } = graphOf(docs.network);
  let mods;
  try { mods = await resolve(lane); } catch (e) { return fail("app.status.refused", e.message); }
  const output = outputRecord(docs.schema, graph);
  setState({ schema: docs.schema, examples: docs.examples, net: docs.network, story: docs.story, page: docs.page, attribution: docs.attribution,
             identity: docs.page.identity, lane, lex, roles, graph, graphError: error, mods, output });
  setState({ env: Object.freeze({ lane, lex, roles, graph, story: docs.story, page: docs.page, schema: docs.schema, output, examples: docs.examples }) });
  field.start(docs.page.motif ? docs.page.motif.id : null, roles.hasAlert());
  applyMark(document.querySelector("header .mark"), docs.page.motif ? docs.page.motif.id : null);
  document.title = docs.page.identity ? docs.page.identity.title : "";
  $("description").content.replaceChildren(docs.page.identity ? docs.page.identity.description || "" : "");
  setState({ tour: tourOf(docs.page.views ? docs.page.views.order : []) });
  S.tour.build();
  renderKeys();
  S.tour.show(location.hash.slice(1) || "start", false);
  return true;
}

function onView(id) { const v = VIEWS[id]; if (v) v.show(S.env); }

// The keys a view takes while it shows (→ steps the context and the guide, ← → the code walk).
const viewKeys = Object.freeze(Object.fromEntries(Object.entries(VIEWS).filter(([, v]) => v.key).map(([id, v]) => [id, (e) => v.key(e)])));

function redrawAll() { network.redraw(); }

// The tour of the views page.json makes available (views_available: its order), with the code room
// off the tour.
const tourOf = (order) => new Tour({ views: order, aliases: ALIASES, say: (key, params) => tText(key, params), onShow: onView,
  offTour: [{ id: "code", key: "c", open: () => openCode() }], hotkeys: { t: toggleTheme }, viewKeys });

function main() {
  configureText({ lang: pickLang({ search: location.search }) });
  applyI18n(document);
  $("status-text").textContent = tText("app.status.connecting");
  $("crumb-here").textContent = tText("view.start");
  const field = new Field($("field"));
  onTheme(() => field.redraw());
  onTheme(redrawAll);
  $("theme").onclick = toggleTheme;
  bindDisclosures();
  bindNetwork(); bindTry(); bindCode();
  extendProbe("missingKeys", missingKeys, { report: true });
  extendProbe("usedKeys", usedKeys, { report: true });
  extendProbe("unresolvedPlaceholders", unresolved, { report: true });
  extendProbe("lane", () => (S.lane ? S.lane.toJSON() : null), { report: true });
  extendProbe("page", () => S.page);
  extendProbe("registry", () => registryReading(S.lane, S.mods), { report: true });
  const ready = boot(field);
  startProbe({
    ready, views: () => ["start", ...(S.tour ? S.tour.views : []), "code"], current: () => (S.tour ? S.tour.current : "start"),
    story: () => S.story, network: () => S.net,
    state: () => ({ view: S.tour ? S.tour.current : "start", seen: [...S.seen], lane: S.lane ? S.lane.toJSON() : null, identity: S.identity, explanation: S.ex,
                    stepper: S.views && S.views.try && S.views.try.stepper ? S.views.try.stepper.values() : null }),
  });
}

main();
