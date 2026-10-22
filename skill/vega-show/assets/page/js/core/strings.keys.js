// The keys the page reads through a variable rather than a literal (R-F2): each dynamic prefix with
// its closed list of members, merged from the fragments' DYNAMIC tables the same way the strings
// are (a prefix two fragments declare is a load error). The page lint checks that every member
// has its key, and counts those keys as used; keysDocument() is the key list vega-show's sync test
// reads, written to assets/page/strings.keys.json by tests/strings-keys.js --write.
import { FRAGMENTS, META, STRINGS, mergeTables } from "./strings.en.js";

export const DYNAMIC_KEYS = mergeTables(FRAGMENTS, "DYNAMIC");

export const KEYS_SCHEMA = "vega/strings-keys@1";

// The members of a dynamic prefix ([] when the prefix is not declared).
export const dynamicMembers = (prefix) => (DYNAMIC_KEYS[prefix] || []).slice();

// Every declared dynamic key, prefix + member.
export const dynamicKeys = () => Object.entries(DYNAMIC_KEYS).flatMap(([p, ms]) => ms.map((m) => p + m));

// The generated key list: every key of the table (sorted), the number each fragment brings, and the
// dynamic prefixes with their members.
export function keysDocument() {
  const fragments = {};
  for (const [area, mod] of Object.entries(FRAGMENTS)) fragments[area] = Object.keys(mod.STRINGS || {}).length;
  const dynamic = {};
  for (const prefix of Object.keys(DYNAMIC_KEYS).sort()) dynamic[prefix] = DYNAMIC_KEYS[prefix].slice();
  return { schema: KEYS_SCHEMA, lang: META.lang, version: META.version, fragments, keys: Object.keys(STRINGS).sort(), dynamic };
}
