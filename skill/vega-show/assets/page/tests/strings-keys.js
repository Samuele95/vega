// The generated key list assets/page/strings.keys.json (vega/strings-keys@1): every key of the
// page's one strings table, the number each area fragment brings, and the dynamic prefixes with
// their members — what vega-show's catalog sync test reads to prove that every id vega-show emits
// has its key. Not served, not copied into a showcase.
//
//   node tests/strings-keys.js --write     regenerate the file from js/core/strings.keys.js
//   node tests/strings-keys.js --check     exit 1 when the file differs from a fresh document
//
// tests/text.test.js runs the check, so a fragment edited without regenerating the file fails.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { keysDocument } from "../js/core/strings.keys.js";

export const KEYS_FILE = fileURLToPath(new URL("../strings.keys.json", import.meta.url));

// The file's text for a document: one key per line, so a change reads as a diff.
export const render = (doc) => JSON.stringify(doc, null, 1) + "\n";

export function fresh() {
  return render(keysDocument());
}

export function stale() {
  let onDisk = null;
  try { onDisk = readFileSync(KEYS_FILE, "utf8"); } catch { onDisk = null; }
  return onDisk !== fresh();
}

function main(argv) {
  if (argv.includes("--write")) {
    writeFileSync(KEYS_FILE, fresh());
    console.log(`OK: wrote ${KEYS_FILE} (${keysDocument().keys.length} keys)`);
    return 0;
  }
  if (stale()) {
    console.error(`FAIL: ${KEYS_FILE} is stale; run node tests/strings-keys.js --write`);
    return 1;
  }
  console.log(`OK: ${KEYS_FILE} matches the strings table`);
  return 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) process.exitCode = main(process.argv.slice(2));
