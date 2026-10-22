

## The HTTP service

The bundle runs as a service over the same predictor the CLI uses. At start-up it checks every
file against `MANIFEST.json` and answers the shipped examples again; `GET /health` answers `200`
only if they reproduce the evaluation's stored answers within the manifest's tolerance.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # if torch is not installed
pip install ".[serve]"                  # the code; this directory stays the model's data
{cmd}-serve --bundle . --port {port}       # then open http://127.0.0.1:{port}/
```

The installed commands (`{cmd}-serve`, `{cmd}-predict`) read the model from the directory they
are given with `--bundle` (or `{env}_BUNDLE`); an editable install (`pip install -e ".[serve]"`) run
inside this directory finds it by itself. The rule is: **the code runs from where it is
installed, the directory supplies the data, and both must be the packed files.** At start-up the
service checks every code file it imports against the digests in this directory's
`MANIFEST.json`, and every data file it reads (weights, spec, fitted parameters, monitor
reference, examples); a mismatch leaves it unready (`/health` 503, naming the file). An installed
copy therefore serves exactly this bundle — rebuild the distribution from the bundle you ship.

or, with Docker (CPU image, non-root, health-checked):

```bash
docker build -t {dist} .
docker run --rm -p {port}:{port} {dist}
```

| method | path | |
|---|---|---|
| GET | `/` | the showcase page when `/vega-show` installed one in `showcase/` (or `--showcase DIR`), else the console page |
| GET | `/console` | the console page: any answer shown as its key/value table |
| GET | `/health` | readiness, the start-up self-test, the showcase and inspection status |
| POST | `/v1/predict` | JSON `{{"rows": [...], "history"?: [...]}}` → one answer per row (`?include=logits`, `reconstruction`, `embedding` add the gated keys) |
| POST | `/v1/predict/csv` | a CSV in the raw schema (multipart `file`) → the same; `?monitor=true` adds the monitor's readings, `Accept: text/csv` answers a CSV |
| POST | `/v1/monitor` | `{{"rows": [...], "truth"?: [...]}}` → the monitor's readings (`?all=true` for every reading) |
| GET | `/v1/info`, `/v1/schema`, `/v1/examples`, `/v1/code` | the manifest, the input and output contract, the example rows, the code that runs |

Every route this bundle serves is listed in `MANIFEST.serve.endpoints`: {endpoints}.

```bash
curl -s -X POST http://127.0.0.1:{port}/v1/predict \
     -H 'content-type: application/json' -d @examples/request.json
```

What an answer holds is the bundle's reading (`MANIFEST.output.reading`; `/v1/schema` describes
every field): a class and its probabilities, a value, quantiles, a score and its decision, and
so on — one answer per row sent, in order, each with its `warnings`. A window bundle answers a
row only when enough preceding rows of its series came with the request (in `rows` or in
`history`); the others are answered with `answered: false` and a warning saying what is missing.

A request that breaks the input schema — a JSON body or an uploaded CSV, which is held to the same
row model — is refused with `422` and a problem body naming every offending column; values
outside the ranges seen in training are answered, with warnings. Every answer carries
`model_version` (the weights' digest), `contract_version` (the input schema's digest) and
`feature_contract_version` (the shaped feature contract's digest, the one each prediction was
computed against); the CSV answer carries them as `X-Model-Version`, `X-Contract-Version` and
`X-Feature-Contract-Version` headers. Numeric columns accept the tokens `Infinity`, `-Infinity`
and `NaN`, which JSON has no literal for. The truth column `schema.json` names (`truth.column`)
is not an input: when a row, a file or a monitor request carries it, it is held to its type — a
class (or the source data's own spelling of one, which the service decodes), a number, or label
names — and it feeds the monitor's checks that read the truth.

The service has no authentication: expose it beyond localhost only behind a proxy that adds it.
It admits at most `{env}_MAX_INFLIGHT` requests to the model at once (default: at most 2 per
process; `--max-inflight`; `--workers N` multiplies it, and the resident model with it).
Admission is decided before a request's body is read: a request that finds no place within
`{env}_INFLIGHT_WAIT` seconds (default 1) is answered `503` with `Retry-After` straight away, and
costs the service nothing. `/health`, `/v1/info`, `/v1/schema`, `/v1/examples`, `/v1/code` and
the pages are never queued behind the model. `examples/demo.json` holds training rows of every
stratum for the demonstration — an illustration, not evidence; the evaluation's figures are in
`model-card.html`.
