// The strings of the code area (js/views/code.js and js/views/highlight/*: the code room, D4 §6.3).
// strings.en.js merges every fragment into the page's one table; a key another fragment defines is a
// load error (the room's heading, "a walk through it", "every file" and the source panel's label are
// the shell's, in core). Values carry no digit outside a {placeholder}. The walk's own titles and
// texts are not here: they are the backend's explain catalog, which page.json#walk carries; the notes
// beside the steps are the story's (code_notes), their report labels report.kind.<kind> (core).
// A file's role and kind are ids of vega/code-view@2 (R-D7): the room says them through
// code.role.<role>, code.about.<role> and code.kind.<kind>, the role id in snake case (ModelStore →
// model_store). DYNAMIC: dynamic prefix → its closed member list.

export const STRINGS = Object.freeze({
  // ---- reading the files (GET v1/code)
  "code.reading": "Reading the files…",
  "code.absent": "This service does not show its code.",
  "code.unavailable": "This service does not show its code: {why}",
  "code.malformed": "The service's list of its files could not be read: {why}",
  "code.malformed.files": "it holds no list of files",
  "code.malformed.file": "a file without its path or its text",

  // ---- the lead, and the line beside the way back
  "code.lead#one": "{n} file, {lines} lines — read by the service from the file it runs. Follow the walk, or open the file.",
  "code.lead#other": "{n} files, {lines} lines — read by the service from the files it runs. Follow the walk, or open any file.",
  "code.lead.notes": "Lines marked § are notes tying a step to where the written account explains it; they are shown here, not written in the file, whose bytes stay the ones that were evaluated.",
  "code.lead.missing#one": "{n} step of the walk is not in these files and is left out.",
  "code.lead.missing#other": "{n} steps of the walk are not in these files and are left out.",
  "code.verified": "every file checked at start-up against the digest the manifest records",
  "code.unverified": "the service did not check these files against the manifest",

  // ---- the walk: its steps, the file and lines on show, back and next
  "code.walk.start": "start the walk →",
  "code.walk.again": "from the start ↺",
  "code.walk.next": "next · {title} →",
  "code.lines.range": "lines {from}–{to} of {n}",
  "code.lines.count#one": "{n} line",
  "code.lines.count#other": "{n} lines",

  // ---- the story's note beside a step (code_notes), under the step's text and above its code
  "code.note.report": "In the report: {label} — {note}.",
  "code.note.file": "In {label}: {note}.",
  "code.note.plain": "Note: {note}.",
  "code.note.the_report": "the report",
  "code.note.brief": "the {stage} brief",
  "code.note.open": "open the report at page {page}",
  "code.note.open_file": "open {file}",
  "code.src.report": "# report: {label}",
  "code.src.file": "# see: {label}",

  // ---- a file's role (vega/code-view@2), what that role does, and its kind
  "code.role.model": "model",
  "code.role.feature_transformer": "feature transformer",
  "code.role.predictor": "predictor",
  "code.role.model_store": "model store",
  "code.role.monitor": "monitor",
  "code.role.inspection": "inspection",
  "code.role.support": "support",
  "code.role.composition_root": "composition root",
  "code.role.http_entry": "HTTP entry point",
  "code.role.ports": "ports",
  "code.role.domain": "domain",
  "code.about.model": "the network itself",
  "code.about.feature_transformer": "the recipe: a raw {row.one} to the network's inputs",
  "code.about.predictor": "{row.many} in, answers out",
  "code.about.model_store": "loading and verifying the weights",
  "code.about.monitor": "the drift checks",
  "code.about.inspection": "the read-only view of the network the page draws",
  "code.about.support": "a helper the adapters share",
  "code.about.composition_root": "wires the adapters together",
  "code.about.http_entry": "this service",
  "code.about.ports": "the ports: the interfaces",
  "code.about.domain": "the value objects",
  "code.kind.adapter": "adapter",
  "code.kind.support": "support",
  "code.kind.entrypoint": "entry point",
});

export const DYNAMIC = Object.freeze({
  "code.role.": Object.freeze(["model", "feature_transformer", "predictor", "model_store", "monitor", "inspection", "support",
    "composition_root", "http_entry", "ports", "domain"]),
  "code.about.": Object.freeze(["model", "feature_transformer", "predictor", "model_store", "monitor", "inspection", "support",
    "composition_root", "http_entry", "ports", "domain"]),
  "code.kind.": Object.freeze(["adapter", "support", "entrypoint"]),
});
