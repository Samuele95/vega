// The renderer registry (D4 §7.3): the modules a drawing needs, loaded by dynamic import, only those
// the graph's stage kinds name (kinds.js RENDERER_OF). load() resolves to a Map from stage kind to
// its renderer; a kind with no module, or a module that does not load or does not draw that kind,
// rejects with RendererGap naming both, so a page never draws a stage with the wrong renderer.
import { RENDERER_OF, rendererOf } from "../kinds.js";
import { tText, data } from "../../core/text.js";

// `why`: a key of the strings table (net.renderer.*) saying what is missing, with its params.
export class RendererGap extends Error {
  constructor(kind, module, why, params = {}) {
    super(tText("net.renderer.gap", { kind: data(kind), module: data(module || "—"), why: tText(why, params) }));
    Object.assign(this, { name: "RendererGap", kind, module, why });
  }
}

// Every renderer interface member (D4 §7.3).
export const MEMBERS = Object.freeze(["kinds", "detail", "measure", "layout", "draw", "hit", "caption", "acts"]);

async function loadModule(file) {
  const mod = await import(`./${file}`);
  return mod.default;
}

// Map<kind, renderer> for the kinds given (a Set or an array), each module loaded once. Every kind is
// mapped before any module is imported, and every import settles before a gap is thrown, so a graph
// that needs several modules that do not load rejects once, with the first kind's RendererGap, and
// leaves no rejection unhandled (the probe counts one as a page error). opts.importer: the module
// loader (file → its default export), replaceable in tests.
export async function load(kinds, { importer = loadModule } = {}) {
  const wanted = [...new Set(kinds)], fileOf = new Map();
  for (const kind of wanted) {
    const file = rendererOf(kind);
    if (!file) throw new RendererGap(kind, null, "net.renderer.unmapped");
    fileOf.set(kind, file);
  }
  const files = [...new Set(fileOf.values())];
  const settled = await Promise.allSettled(files.map((file) => Promise.resolve().then(() => importer(file))));
  const loaded = new Map(files.map((file, i) => [file, settled[i]]));
  const out = new Map();
  for (const kind of wanted) {
    const file = fileOf.get(kind), s = loaded.get(file);
    if (s.status === "rejected") throw new RendererGap(kind, file, "net.renderer.unloaded", { error: data(s.reason && s.reason.message ? s.reason.message : String(s.reason)) });
    const r = s.value, missing = MEMBERS.filter((m) => !(m in (r || {})));
    if (missing.length) throw new RendererGap(kind, file, "net.renderer.members", { members: data(missing.join(", ")) });
    if (!r.kinds.includes(kind)) throw new RendererGap(kind, file, "net.renderer.unlisted");
    out.set(kind, r);
  }
  return out;
}

// Every kind a module draws, by module (for the coverage test: every RENDERER_OF entry resolves).
export const MODULES = Object.freeze([...new Set(Object.values(RENDERER_OF))]);
