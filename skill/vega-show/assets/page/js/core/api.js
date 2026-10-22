// Every URL the page reads, and the calls that read them. All URLs are relative to the page's own
// location (the service's root, or a reverse-proxy prefix in front of it): the bundle's service
// (health, v1/*) and the showcase slot (showcase/*). A problem+json answer becomes a ProblemError
// that carries the problem itself; the words that describe it are the caller's.

export const URLS = Object.freeze({
  health: "health",
  info: "v1/info",
  schema: "v1/schema",
  examples: "v1/examples",
  network: "v1/network",
  explain: "v1/explain",
  predict: "v1/predict",
  predictCsv: "v1/predict/csv?monitor=true&include=rows",
  monitor: "v1/monitor?all=true",
  code: "v1/code",
  serviceStory: "v1/story",
  story: "showcase/story.json",
  page: "showcase/page.json",
  attribution: "showcase/data/attribution.json",
  report: "report.pdf",
});

// The document versions this template reads; a payload naming another is a visible error state.
export const EXPECT = Object.freeze({
  story: "vega/story@2",
  page: "vega/page@1",
  network: "vega/network-view@2",
  explanation: "vega/explanation@2",
  attribution: "vega/attribution@2",
  examples: "vega/examples@2",
});

// An answer that is not 2xx. `message` is the problem's own title and detail; `problem` is its
// body (with `errors: [{column, error, rows}]` on a refused request).
export class ProblemError extends Error {
  constructor(status, problem, statusText) {
    const title = problem.title || statusText;
    super(`${title}${problem.detail ? " — " + problem.detail : ""}`);
    this.name = "ProblemError";
    this.status = status;
    this.problem = problem;
    this.errors = Array.isArray(problem.errors) ? problem.errors : [];
  }
}

export async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new ProblemError(r.status, body, r.statusText);
  return body;
}

export const post = (url, payload) => api(url, {
  method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });

export function upload(url, file) {
  const fd = new FormData();
  fd.append("file", file);
  return api(url, { method: "POST", body: fd });
}

// The readiness document: its body whatever the status (503 carries the reason).
export async function health() {
  const r = await fetch(URLS.health);
  return r.json();
}

// null when `doc` is the version the page reads, else {doc, got, want} for the error state.
export function schemaMismatch(kind, doc) {
  const want = EXPECT[kind];
  if (!want || !doc) return null;
  const got = typeof doc.schema === "string" ? doc.schema : null;
  return got === want ? null : { doc: kind, got, want };
}
