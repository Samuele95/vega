# The Vega ledger — `.claude/vega-ledger.md`

The ledger is the orchestrator's working memory across turns and sessions: **where each lane stands**, which artifact is current, what was verified, what is waiting for the user. It is deliberately separate from `.claude/domain-forge-memory.md`, which the forges own and which records *modeling decisions* (stances, retypings, partitions, shaping choices). Vega reads that file but never writes to it; the stages do. The ledger records *chain state* and nothing a stage already records elsewhere.

Created from the template below on the first run (say so in the first status line). Rewritten — not appended forever — at the end of every stage: the **Lanes** table is the current truth, the **History** section is the append-only log.

```markdown
# Vega ledger — <project name or root path>

## Sources
<!-- one line per source Vega classified: kind | path | digest (sha256 of the file, or of the file list for a codebase) | first seen -->
dataset | data/orders.csv | sha256:… | 2026-09-03
codebase | src/ | sha256:… (tree) | 2026-09-03

## Lanes
<!-- one row per lane: the objective (the destination the user chose, `+document` when the PDF was wanted, with how it was set — user | --goal | assumed), the current artifact, the stages done in order, the next stage Vega would propose, and the last verification (the stage's own check, then `published` and `brief` — the two checks every package stage passes) -->
| lane | source | objective | current artifact | stages done | next | last check |
|---|---|---|---|---|---|---|
| dataset:orders | data/orders.csv | blueprint(pipeline) +document — user, 2026-09-03 | orders.domain.analysis.html | forge-data, lens | shape | validate 19/19 · smoke_analysis 12/12 · published · brief · 2026-09-03T10:12Z |
| dataset:sensors | data/sensors.csv | pack — user, 2026-09-05 | sensors.domain.analysis.shaped.vega.train.eval.html | forge-data, lens, shape, blueprint, net, train, eval | pack | validate 19/19 · smoke_evaluation · seal exposures 1 · campaign vega-runs/2026-09-05T11-03Z · published · brief · 2026-09-05T12:41Z |
| software:src | src/ | spec — user, 2026-09-03 | spec/spec-analysis.html | spec | forge-prose | self-contained · module map 7 · 2026-09-03T09:40Z |

## Open
<!-- forks or gates a stage left pending, and the advisory open items of the conformance gate, one line each, with the stage and what it needs. Removed when resolved. -->
- shape: phase *values* — S6 impute delivered_days: group-median vs median; S8 transform unit_price: log vs none (waiting for user; answer returns via `--recipe`)
- train: GATE emit-only — run vega-runs/2026-09-05T11-03Z/run_campaign.sh on the GPU machine, then `/vega-train sensors….vega.html --import-run vega-runs/2026-09-05T11-03Z` (waiting for the user)
- net: advisory — architect 1:A-3 (structural, training/loop.py: …) confirmed by audits 1, 2 of 3 — open item for /architect and /improve: fix the backend template through /architect or /improve (apply) on backends/<name>/templates/, re-emit, and run the gate again

## Environment
<!-- prerequisites checked once, with the date; re-checked when a stage fails on one -->
python 3.12 · pandas 2.2 · scipy 1.14 · scikit-learn 1.5 · geopandas absent · torch 2.4 · nbconvert present · optuna absent · gpu absent · chromium present · latex absent · checked 2026-09-03

## History
<!-- append-only: timestamp | lane | stage | outcome | artifact | note -->
2026-09-03T09:40Z | software:src | spec | OK | spec/spec-analysis.html | 7 modules, 1 flow, 2 open questions
2026-09-03T10:12Z | dataset:orders | lens | OK | orders.domain.analysis.html | 6 findings, 4 turns, 3 shaper candidates
```

## Rules

- **Only Vega writes the ledger; only the stages write artifacts.** Vega never edits an HTML, a dataset, a recipe or the forge memory. If a stage's product is wrong, the fix is to re-run the stage (a refine, a `--continue`, a new recipe), never a hand edit by the orchestrator.
- **A lane's `current artifact` is the last one that passed its check — its own and the two package checks, `published` and `brief`.** A stage whose HTML the orchestrator had to open itself is `OK` with the omission in the History note; a stage without its brief is `FAIL` until re-run. A stage whose product failed verification is recorded in History with `FAIL` and the artifact path, and the lane's current artifact stays the predecessor; `next` becomes the same stage with the failure as the brief.
- **Digests make re-runs cheap.** A source whose digest is unchanged and whose lane has a current artifact is not re-forged; Vega proposes the next stage. A changed digest resets the lane to its first stage and says so (`WARN: data/orders.csv changed since orders.domain.html was forged`).
- **The objective is a decision, and decisions are asked once.** A lane's `objective` is the destination from `chain-map.md` § Destinations, with `+document` when the user wanted the step-by-step PDF at the end (asked in the same question, recorded in the same cell), and with how it was set: `user` (chosen at step 4), `--goal` (stated), or `assumed` (unattended, and the summary says so); a label the user named with `/vega --target <column>` goes in the same cell as `target <column> — --target`, and is the partition every later invocation of the lane passes on (`--partition` to the forge, `--target` to the shaper); a `/vega --ship-anyway "<reason>"` the user typed goes in the same cell as `ship-anyway "<reason>" — --ship-anyway`, passed to `pack` when the verdict is not earned. While it is recorded and the source digest is unchanged, Vega does not re-ask it — the same rule that keeps a partition from being re-asked. It is cleared, and asked again, when the lane reaches it, when the source changes, or when the user moves it.
- **Open items are questions for the user, never answered by Vega.** A `FORK:` a stage emitted is copied here verbatim with the stage name; when the user answers, Vega re-invokes the stage with the answer in the form the stage accepts — a flag (`--partition late` for the forge), an edited recipe (`--recipe` for the shaper), or the first message of its loop (the lens, the chat) — it does not decide in the user's place.
- **The conformance gate's advisory findings are open items too, handed to `/architect` and `/improve`.** A structural or divergence finding two of three audits confirmed on the `net` package or the `pack` bundle (`vega-forge/references/code-contract.md` §5, *Advisory*) is recorded by the stage's layer as an open item (`conformance.<tool>.advisory.open_items`) and named in its brief's *Open*; Vega copies each one here as `<stage>: advisory — <the item's text>` — unattended, and attended when the user answered *accept and record*. It never blocks the next stage: the lane advances, and the item is the brief of a later `/architect` or `/improve` pass on the templates (which Vega names as an option, step 10, and never schedules on its own). It is removed when a re-run of that stage's gate no longer confirms it.
- **Nothing that memory holds is duplicated.** A retyping, a partition, a shaping choice lives in `.claude/domain-forge-memory.md`; the ledger says only that the stage ran and what it produced.
- **The ledger is prose-free.** Tables and one-line entries; the reasoning behind a routing decision goes in the chat turn where Vega proposed it, not here.
- **The network sub-chain's chain state.** The campaign directory, the test seal's exposure count and a `--after-eval` campaign's predecessor are chain state and live in the lane's `last check` and in History (`… | train | OK | ….train.html | campaign vega-runs/<ts>, after-eval of <previous> (exposures 1)`); the memory line the stages write holds the *verdict* and the stances, naming the exposure the verdict was made at (a qualifier of the verdict, not a counter — the seal file is the count's truth, the ledger its chain state). A `train --emit-only` hand-over is an **Open** item whose answer is the `--import-run` invocation.
