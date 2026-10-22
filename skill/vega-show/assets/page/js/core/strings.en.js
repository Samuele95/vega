// The page's one keyed table of English words (R-F2): every chrome string the page shows, merged
// from the area fragments under strings/ so that the packages that own the areas write their own
// words without sharing a file. A key two fragments both define is a load error: the page does not
// start, and the page lint names the key. Keys follow text.js's grammar (a base, an optional
// @<view key>[+<shape>] variant, an optional #zero|#one|#other plural); values carry no digit
// outside a {placeholder}. Block, activation and link prose is not here: it is the backend's
// explain catalog, which page.json#explain carries and text.js looks up under explain.*.
import * as core from "./strings/core.en.js";
import * as network from "./strings/network.en.js";
import * as views from "./strings/views.en.js";
import * as stepper from "./strings/stepper.en.js";
import * as code from "./strings/code.en.js";
import * as sequence from "./strings/sequence.en.js";
import * as graph from "./strings/graph.en.js";
import * as latent from "./strings/latent.en.js";
import * as howAdvanced from "./strings/how-advanced.en.js";

// The fragments by area, in merge order (the file of each is strings/<area>.en.js).
export const FRAGMENTS = Object.freeze({
  core, network, views, stepper, code, sequence, graph, latent, "how-advanced": howAdvanced,
});

// Merges one member (STRINGS or DYNAMIC) of every fragment into one frozen table; a key defined by
// two fragments throws, naming the key and both areas.
export function mergeTables(fragments, member) {
  const out = {}, from = {};
  for (const [area, mod] of Object.entries(fragments)) {
    for (const [key, value] of Object.entries(mod[member] || {})) {
      if (key in out) throw new Error(`strings: ${member} key "${key}" is defined by both ${from[key]} and ${area}`);
      out[key] = value; from[key] = area;
    }
  }
  return Object.freeze(out);
}

// The table's language, and the words that join a list.
export const META = Object.freeze({ lang: "en", version: 1, list: Object.freeze({ and: " and ", comma: ", " }) });

export const STRINGS = mergeTables(FRAGMENTS, "STRINGS");
