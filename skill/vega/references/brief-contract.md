# The stage brief — `docs/<stage>-brief.md`

Every stage skill of the Vega package leaves one Markdown file in its run directory: the **stage brief**. It is the step-by-step record of *this* run of *this* stage, written for two readers — the user who wants the story of the run without opening the HTML, and `/document-project`, which turns the briefs of a lane into the chapters of the step-by-step PDF (`/document-project vega <lane>`). It is written by the stage, after the layer is published and before the summary, and checked by `/vega` (step 7): a stage without its brief did not finish.

The brief **repeats nothing the layer does not already hold**. Every number it states — every number outside the fenced transcript of `## What ran`, below — is copied from a record — the layer's JSON, `manifest.json`, `experiments.json`, `metrics.json`, `MANIFEST.json` — and carries the key it was copied from in parentheses, so a reader (and `document-project`'s verifier) can resolve it. A brief that states a number the records do not contain is a defect, the same defect the apply scripts refuse in a reading. In the fenced transcript of `## What ran` — the commands and the marker lines they printed, quoted verbatim as the reproduction path — a number's key is the command printed above it: re-running that command resolves it, and nothing in the transcript is the brief's own claim. A number the brief relies on (a validation count, the conformance gate's finding counts by depth) is stated in `## What came out` or `## What the reader should check` with its record key, and a chapter takes its numbers from `records`, never from a transcript. Prose is connected paragraphs, motive before mechanism, in the register of the stage's own readings; no bullet carries an argument.

## Location and name

`<run-dir>/docs/<stage>-brief.md`, where `<run-dir>` is the directory the stage already writes (the forge's and lens's run directories, the shaper's output directory, `blueprint-runs/<ts>/` for `blueprint`, `vega-runs/<ts>/` for `net`, the campaign directory for `train` and `eval`, the bundle directory for `pack`, the showcase directory `<bundle dir>/showcase/` for `show`) and `<stage>` is the chain-map stage name (`forge-data`, `lens`, `shape`, `blueprint`, `net`, `train`, `eval`, `pack`, `show`). A re-run of the same stage in the same directory overwrites its brief and says so in the frontmatter (`supersedes`).

## Frontmatter

```yaml
---
stage: net                          # chain-map stage name
skill: vega-forge                   # the skill that wrote it
lane: dataset:orders                # the ledger's lane name, when invoked by /vega; else the input stem
input: orders.domain.analysis.shaped.html
output: orders.domain.analysis.shaped.vega.html
run_dir: vega-runs/2026-09-26T10-12Z
layer: architecture                 # the layer this stage appended
date: 2026-09-26T10:41Z
records:                            # the files every number below resolves to, with digests
  - path: vega-runs/2026-09-26T10-12Z/architecture-layer.json
    sha256: …
  - path: vega-runs/2026-09-26T10-12Z/smoke.json
    sha256: …
supersedes: null                    # or the date of the brief this one replaced
---
```

Two keys are written by the post-report stage only, after `supersedes` (`show`, below):

```yaml
post_report: true                   # show only: written after the report, never a chapter
report: {status: linked, pdf_digest: sha256:…, aux_digest: sha256:…, roles_digest: sha256:…}   # show only; status brief-links (the digests null) when no report was compiled
```

## Sections — in this order, these headings

**`## What this stage was for`** — two or three sentences: the question the stage answers on this lane and what the previous stage handed it (named: the layer, the key, the fork it left).

**`## What was decided`** — one paragraph per decision the stage's layer records (a retyping, an orientation, a finding admitted, a recipe phase, a decision family, a rung, the seal, the ship stance), each stating the choice, its **source** in the stage's own source vocabulary (`analysis:F3`, `blueprint:posture`, `user:stance/…`, `vega:default catalog: defaults/optimizer`), the alternative weighed, and what the choice does downstream. Forks asked and their answers (or the default applied unattended) appear here, marked.

**`## What ran`** — the exact commands the stage executed, in order, as a fenced block, with the `OK:`/`WARN:`/`ERROR:` line each returned. Nothing paraphrased: this is the reproduction path.

**`## What came out`** — the numbers, each with its record key: the shape of the data (`manifest.output_schema`, rows × columns per part), the parameter count and multiply-adds (`smoke.json.params_total`, `.madds_per_sample`), the ledger in one line per run (`experiments.json.runs[<id>].valid.<primary>`), the primary with its CI and the paired difference against the strongest baseline (`metrics.json.primary`, `.paired[<baseline>]`), the round-trip result (`MANIFEST.json.roundtrip.status`) — whatever the stage's layer holds. A number with no key does not appear.

**`## What the reader should check`** — the verifications the stage ran on its own product (validation, smoke, the compiler, the conformance gate with the tools it invoked and their finding counts by depth, each count with the layer key it is recorded under, the notebook's assertion cell, the replay, the round trip) with their results, and how to re-run each.

**`## Open`** — what the stage did not settle: forks abstained, `WARN:` lines, hypotheses the next stage must test, an open item pointed upstream — and, for `net` and `pack`, **every open item the conformance gate recorded** (a structural or divergence finding two of three audits confirmed, advisory — `vega-forge/references/code-contract.md` §5, *Advisory*), each named `<tool> <audit>:<id>` (`architect 1:A-3`) with its remedy, for `/architect` and `/improve`; `check_brief.py` reads the stage's output layer and fails a brief whose *Open* omits one. Empty is written as "Nothing open."

**`## Next`** — the hand-off in one line: the next stage's exact invocation as the stage's summary states it.

## How `/document-project` reads it

In Vega mode (`/document-project vega <lane> [<stage>]`), the orientation interview takes the briefs as its first answers: the lane's briefs in chain order, a brief with `post_report: true` skipped, are the proposed chapters (one chapter per brief, plus an introduction from the first brief's *What this stage was for* and a conclusion from the last brief's *Open* and *Next*); the register is proposed from the briefs' prose; every number in a chapter must resolve to a `records` entry of a brief — the verifier reads the frontmatter. The interactive chapter-by-chapter loop is unchanged: the briefs seed it, they do not replace it.

## Rules

- **The stage writes its brief; nobody else does.** `/vega` checks that it exists and that its numbers resolve; `/document-project` reads it. An orchestrator or a documentation pass that writes a brief for a stage produces a record nobody verified.
- **Records, not memory.** A number comes with the key it was read from; a decision comes with its source. The brief is checkable against the run directory alone.
- **After publish, before summary.** The brief describes a published, verified layer — it is written when the HTML has validated, passed its smoke test and been opened in the conversation, and the summary that follows names it beside the HTML.
- **Post-report stages.** `show` writes its brief after the report was compiled and cites the report's digests (`post_report: true`, `report`); `/document-project` never makes it a chapter, and a report that changes makes the showcase stale rather than the other way round — the dependency runs pack → document → show, one way. In brief-link mode (no report yet) the `report` key says so, and the stage is proposed again once the report exists.
- **Short.** A brief is one to three pages of Markdown; it points into the layer, it does not copy it. The interactive HTML is the record; the brief is the story a chapter needs.
