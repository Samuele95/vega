// The page's state: the one module allowed module-level mutable state (page lint PS5). Everything
// the page has read, and the little it remembers between views, lives here and nowhere else, so a
// module stays a set of functions over this state and its arguments.

// What the page has read, and the objects it built from it.
//   story, page, net, examples, schema, info: the payloads, as fetched (null when absent)
//   identity: the bundle's title, description and row nouns (MANIFEST.serve.identity)
//   lane: the lane the page renders (set by the boot that reads page.json)
//   seen: the tour views already visited; tour: the running Tour
//   ex, row: the last explanation asked for, and the raw row it explains
export const S = {
  story: null, page: null, net: null, examples: null, schema: null, info: null,
  identity: null, lane: null, seen: new Set(), tour: null, ex: null, row: null,
};

// Caches that a theme change clears: colours read from the CSS tokens, parsed colours.
export const CACHE = { colours: null, rgb: new Map() };

// Listeners registered by the modules: theme changes, disclosures opened (by the id they reveal).
export const LISTENERS = { theme: [], disclose: new Map() };

// Replaces the fields of S given in `values`, leaving the others as they are.
export function setState(values) {
  Object.assign(S, values);
  return S;
}
