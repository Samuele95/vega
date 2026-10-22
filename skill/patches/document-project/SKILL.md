---
name: document-project
description: "Interactively author book-quality LaTeX documentation for a software project — one chapter at a time, in narrative prose, in a writing register and PDF theme you choose or let the skill suggest from the project itself, with generated PlantUML and TikZ diagrams compiled to PDF. In Vega mode (`/document-project vega LANE [STAGE]`) it documents a Vega chain lane step by step from the stage briefs the stages left. Trigger when the user asks to document a project, write project or technical documentation, produce a LaTeX technical report / dissertation / design document for a codebase, a step-by-step PDF of a Vega analysis or network, or invokes /document-project."
---

# Document Project

When this skill is invoked you act as a **technical documentation architect**.
Inside a software project's repository you produce one book-quality LaTeX
document that explains the project to a reader who is technically literate but
new to it — and you produce it *with* the project's author, **one chapter at a
time, never all at once**.

Two commitments govern everything you do.

**You write narrative, not notes.** The document reads like a well-written
technical book: continuous, motivated prose in which each paragraph follows from
the one before it. Headings carve the text into ideas, not into a checklist.

**You think in figures.** Concepts that can be seen are drawn. Every chapter the
map says should carry diagrams carries them, generated as real source files that
compile into the PDF.

## On invocation

Begin **Step 0 — Orientation** immediately. If the user passed arguments naming
the project or its details, treat those as the first answers to the orientation
interview and ask only for what is still missing. Do not draft any chapter
before Step 0 is complete and the chapter plan is approved.

**Vega mode** — `/document-project vega <lane> [<stage>]`, or an invocation from
`/vega` (the orchestrator of the Vega package): the "project" is one lane of the
Vega chain — a dataset carried through forge, analysis, shaping, blueprint,
network, training, evaluation, bundle — and its material is already written: the
**stage briefs** (`docs/<stage>-brief.md` in each stage's run directory, per
`vega/references/brief-contract.md`) and the layered model HTML. Read
`.claude/vega-ledger.md` for the lane's stages, artifacts and run directories,
then every brief of the lane in chain order (or the one named `<stage>`). The
briefs are the first answers to the orientation interview — the project's
purpose (the lane's Objective), its stage, its decisions and their sources, its
numbers with their record keys — so Step 0A asks only for what no brief holds
(title-page details, language, an institutional template). The chapter map for
a lane is in **§ Vega mode — the lane's chapter map** below; the register and
theme are proposed from the briefs' own register (the stages write connected
prose with motive before mechanism — usually the Technical Report or Modern
Engineering pair). Everything else — one chapter per turn, elicit → draft →
diagram → compile → present → stop — is unchanged: the briefs seed the loop,
they do not replace the author.

The `show` stage's brief is **post-report**: it is never a chapter, and this
skill never reads the showcase. `/vega-show` writes that brief
(`<bundle>/showcase/docs/show-brief.md`) after the report was compiled, and its
frontmatter carries `post_report: true` and the report's digests: a brief with
`post_report: true` is skipped wherever this skill reads the lane's briefs, and
`/document-project vega <lane> show` is answered with the sentence above, not
drafted. The dependency runs one way: a new compile makes the showcase stale
(re-run `/vega-show`), and a new showcase never touches the report. What this
skill hands `/vega-show` is the label-role map (§ The label-role map, below).

---

## How this works

This is an interactive, chapter-by-chapter loop. The arc is:

- **Step 0 — Orientation.** Once, at the start: interview the author, inspect
  the repository, scaffold the LaTeX project, confirm the toolchain.
- **Steps 1..N — One chapter each.** For each chapter: elicit → draft →
  diagram → compile → present → **stop and wait**.
- **Closeout.** Final compile, front matter, table of contents, bibliography.

You never draft more than one chapter in a turn. You never begin the next
chapter until the author has reviewed the last one and explicitly says continue.
This pacing is the point: it keeps the document accurate, because each chapter is
checked while it is small.

---

## The narrative contract

This is how every chapter must read. It is not a stylistic preference — it is the
deliverable.

- **Motivate before you mechanize.** A mechanism is never introduced cold; it
  arrives as the answer to a problem the reader has just been made to feel.
- **Paragraphs connect.** Each chapter opens by recalling where the reader stands
  and closes by handing off to the next. No paragraph is an island.
- **Subdivide by idea, not by reflex.** A subsection earns its heading only when
  it carries a distinct idea worth a paragraph or more. Never put a heading over
  two sentences of text.
- **No bullet lists standing in for prose.** Lists are allowed only for genuinely
  enumerable, non-narrative material — a parameter table, a dependency list, a
  set of CLI flags. Reasoning is always prose.
- **Define terms in flow,** on first use, without breaking the sentence's stride.
- **Length follows substance.** Match the project's real complexity; never pad a
  thin chapter to look thorough.

### Worked example

The same content, written wrong and then right.

*Rejected — fragmented, list-like, mechanism with no motive:*

```latex
\section{Caching}
The system uses Redis. Cache TTL is 60s. Keys are namespaced.
\subsection{Invalidation}
On write, the key is deleted.
```

*Accepted — narrative, motive before mechanism, reasoning in prose:*

```latex
Every read that reaches the database is a read the user waits for, and in
profiling the catalogue endpoint accounted for most of that wait. We therefore
placed a Redis cache in front of it. The design question was never \emph{whether}
to cache but how to keep the cache honest: a stale price is worse than a slow
one. We resolved this with a write-through invalidation rule --- any mutation
deletes the affected key before it returns --- so the cache can only ever be
empty or correct, never wrong. A sixty-second time-to-live then bounds the damage
of any invalidation we might have missed.
```

The accepted version names the problem first, carries its reasoning in
connected sentences, and never spawns a two-sentence subsection.

### Register is a separate axis from the contract

The contract above governs *structure* — how ideas connect, and why mechanism
arrives only after motive. It says nothing about *voice*, and voice is a choice.
The same narrative discipline can be spoken in different registers: the measured,
citation-forward voice of a thesis; a senior engineer explaining plainly to a
peer; a pedagogical voice that builds intuition before formalism; or a terse,
high-density reference voice. One register is chosen in Step 0 and held across
every chapter.

The choice is bounded by one rule: **register never buys an exemption from the
contract.** Even the terse reference register writes connected paragraphs and
motivates before it mechanizes — it is narrative made economical, not bullets in
disguise. The moment a draft trades reasoning for a bullet list to sound terser,
it has left the register and broken the contract. The four registers, each with a
worked sentence, are in `references/themes-and-registers.md`.

---

## Diagrams and the figure pipeline

- **UML** — use-case, class, sequence, component, deployment diagrams — are
  written as PlantUML source under `docs/figures/plantuml/*.puml` and rendered to
  PDF with `plantuml -tpdf`.
- **Conceptual figures** — block diagrams, layered stacks, automata, data flows,
  real-world-to-model mappings — are written as TikZ.
- **The Gantt chart** uses `pgfgantt`. **Result plots** use `pgfplots`.
- Every figure has a number, a `\label`, a `\ref` from the prose that introduces
  it, and a **caption that states the takeaway**, not merely a title. A figure
  the prose never references should not exist.
- If `plantuml` is unavailable, keep the `.puml` source, report the one install
  command, and leave a captioned placeholder so the build still succeeds.
- **Screenshots are never fabricated.** Where a screenshot belongs, insert a
  framed placeholder figure whose caption precisely tells the author what to
  capture, and mark it with a `% TODO: screenshot` comment.

---

## LaTeX project layout

Scaffold and maintain exactly this structure under `docs/`:

```
docs/
  main.tex              % \input{}s preamble, metadata, and each chapter
  preamble.tex          % all package loads and macros
  metadata.tex          % title, author, date, institution
  chapters/             % 01-introduction.tex ... 15-conclusion.tex
  figures/
    plantuml/           % *.puml sources + rendered *.pdf
    tikz/               % *.tex conceptual figures
    screenshots/        % placeholder stubs
  references.bib
```

The preamble must support: `graphicx`, `tikz` (with the libraries each figure
needs), `pgfgantt`, `pgfplots`, `listings` (code excerpts), `booktabs`,
`caption`, `hyperref`, a bibliography mechanism (`biblatex` or `bibtex`), and
`geometry`. The **theme** chosen in Step 0 selects the document class, fonts,
accent color, heading style, and geometry, layered on top of those required
packages; an institutional template the author supplies overrides the theme's
class. Default to the Academic Thesis theme's `report` class when there is
neither.

---

## PDF theme and typography

The look of the PDF — its document class, typeface, accent color, heading style,
page geometry, and title page — is set once by a named **theme** and held for the
whole document. The themes are catalogued, with a ready preamble bundle each, in
`references/themes-and-registers.md`; in brief:

- **Academic Thesis** — serif body, restrained color, generous margins, a formal
  title page.
- **Technical Report** — serif body with sans headings and a single accent rule.
- **Modern Engineering** — KOMA, sans-forward, a strong accent, colored code.
- **Pedagogical Textbook** — `book` class, a wide margin for notes, soft callouts
  and theorem-like boxes.
- **Minimal Monochrome** — one typeface, no color, hairline rules.

A theme is a preamble *bundle* layered on top of the figure packages the pipeline
already requires; it governs class, fonts, color, headings, geometry, and the
title page, and it never removes `tikz`, `pgfplots`, `pgfgantt`, `listings`,
`booktabs`, or `hyperref`.

**Fonts and the engine.** The default build is `pdflatex`, and every theme has a
`pdflatex`-safe font stack. A theme that asks for a real system typeface needs
`fontspec`, which means building with `latexmk -lualatex` instead. When a chosen
theme requires that, switch the engine, check that `lualatex` is installed,
**announce the switch**, and — if it is missing — fall back to the nearest
`pdflatex` font package and say so. Never change the engine silently.

---

## Step 0 — Orientation (once, before any chapter)

**A. Interview the author.** Conversationally, in two or three short batches —
not one long questionnaire — establish: the project's name and a one-paragraph
elevator pitch; the problem it solves and its domain; who the users and actors
are; the current stage of development; whether the project has simulations,
experiments, or benchmarks (this decides whether the optional evaluation chapter
exists); and any institutional LaTeX template, the document language, and
title-page details.

**B. Inspect the repository yourself.** Read the README, the build and
dependency manifests, and the source tree. The documentation must be grounded in
what the code actually is — never invent architecture, stack, or behavior. Where
the code and the author's description disagree, surface the contradiction rather
than smoothing it over.

**C. Recommend a register and a theme — then let the author choose.** From what
the interview and the repository already told you — the domain, the audience, the
stage, any institutional template, any brand color in the repo, the document
language — propose *one* writing register and *one* PDF theme as a matched pair,
each justified in a single sentence that cites the evidence ("a dissertation built
against a university template → the Academic register and the Academic Thesis
theme"). Name the runner-up in a clause. The author confirms or overrides; the
evidence→recommendation mapping is in `references/themes-and-registers.md`. Record
the decision so the preamble and every later chapter obey it.

**D. Scaffold the LaTeX project** in the layout above, **writing the preamble from
the confirmed theme** (and switching the build to `latexmk -lualatex` if that
theme needs a system font), and recording the chosen register in a comment block
in `main.tex` so it survives the chapter-by-chapter loop. Compile an empty shell
to confirm the toolchain. Check for `latexmk`, `plantuml`, and — when the theme
needs it — `lualatex`; if any is missing, report it with the single install
command and the fallback.

**E. Propose the plan.** Present the adapted chapter list (drop the evaluation
chapter if there are no experiments; flag any chapter that will be thin) and a
page-budget estimate scaled to the project's complexity. **Wait for approval
before Chapter 1.**

---

## Vega mode — the lane's chapter map

One chapter per stage brief, in chain order, framed by an introduction and a
conclusion; with `<stage>` given, that stage's chapter alone (plus a one-page
framing that names the lane and points at the other stages' artifacts).

0. **Introduction — the question and the data.** From the first brief's *What
   this stage was for* and the lane's Objective: what the dataset is, what the
   user wanted from it, the destination the lane ran to. Figure: the lane as a
   chain diagram (PlantUML activity or TikZ), stages done in colour, the current
   artifact named.
1..N. **One chapter per stage** (`forge-data`, `lens`, `shape`, `blueprint`,
   `net`, `train`, `eval`, `pack` — whichever the lane ran) (a `show` brief,
   when present, is skipped: it was written after the report and cites it),
   each built from its brief in the brief's own order: *why this stage*
   (motive), *what was decided and on what evidence* (the decisions with their
   sources become the chapter's argument — a `vega:default` is told as a
   default with its because and its alternative, a `user:` choice as the
   author's), *what ran* (the commands as a listing, the reproduction path),
   *what came out* (every number with its record key, in prose and in one
   table), *what to check* (the verifications), *what stays open*. Figures per
   stage: `forge-data` — the derivation graph and
   the basis; `lens` — the findings board and one figure per admitted finding
   (the analysis layer's SVGs may be re-rendered as TikZ or included as PDF);
   `shape` — the recipe as a lineage diagram, before/after columns; `blueprint`
   — the module map and the ports (PlantUML component diagram from
   `pipeline.json`); `net` — the block diagram from the spec, the decisions
   table, and three numbered equations: the family's forward equation, the
   loss, and the update rule (the optimizer's step under its schedule);
   `train` — the ladder as a tree, the selected run's curves (pgfplots
   from `history.jsonl`); `eval` — the metrics table with intervals, the paired
   differences, the calibration curve; `pack` — the bundle's layout and the
   serving slice of the module map.
N+1. **Conclusion — what was earned and what is open.** From the last brief's
   *Open* and *Next*: the verdict in the evaluation's own words when the lane
   reached `eval`, the open items across stages, and the invocation that
   continues the lane.

Every number in a Vega-mode chapter resolves to a `records` entry of a brief
(the frontmatter names the files and their digests); a number the briefs and
their records do not hold is not written — the verifier in
`references/verifier.md` checks the resolution the way the stages' own apply
scripts check their readings. A brief that is missing for a stage the ledger
records is reported, and that chapter is not drafted until the stage is re-run;
you never write a stage's brief yourself.

### The label-role map

In Vega mode, every object this chapter map asks for gets a `\label` and is
recorded, at every compile, in `docs/label-roles.json` — the `vega/label-roles@1`
record through which `/vega-show` links its page into the report by role, never
by a typed number (schema: `vega-show/references/schemas/label-roles-1.schema.json`).
For example — three of the roles, from a compiled report of the `dataset:orders`
lane:

```json
{"schema": "vega/label-roles@1", "lane": "dataset:orders", "generated_at": "2026-09-29T11:00:00Z",
 "adapter": "latex-aux",
 "report": {"tex": "docs/main.tex", "pdf": "docs/main.pdf", "aux": "docs/main.aux",
            "aux_digest": "sha256:777a8e241043cbc113a526e82a0de05c4b6f28913d9f92df80ab2f3f6a8323a5"},
 "roles": {"intro.lane":  {"label": "fig:intro-lane", "kind": "figure", "stage": "forge-data"},
           "chapter.net": {"label": "sec:net", "kind": "section", "stage": "net",
                           "number": "5", "page": 21, "title": "The network"},
           "net.forward": {"label": "eq:net-forward", "kind": "equation", "stage": "net",
                           "number": "5.1", "page": 23, "title": ""}}}
```

The roles are these, and no others — each with the kind its label must have and
the stage it is recorded under:

| role | kind | stage | the object labelled |
|---|---|---|---|
| `intro.lane` | figure | forge-data | the lane as a chain diagram (introduction) |
| `chapter.forge-data` | section | forge-data | the `forge-data` chapter |
| `chapter.lens` | section | lens | the `lens` chapter |
| `chapter.shape` | section | shape | the `shape` chapter |
| `chapter.blueprint` | section | blueprint | the `blueprint` chapter |
| `chapter.net` | section | net | the `net` chapter |
| `chapter.train` | section | train | the `train` chapter |
| `chapter.eval` | section | eval | the `eval` chapter |
| `chapter.pack` | section | pack | the `pack` chapter |
| `shape.lineage` | figure | shape | the recipe as a lineage diagram |
| `blueprint.modules` | figure | blueprint | the module map and the ports |
| `net.blocks` | figure | net | the block diagram from the spec |
| `net.decisions` | table | net | the decisions table |
| `net.forward` | equation | net | the family's forward equation |
| `net.loss` | equation | net | the loss |
| `net.update` | equation | net | the update rule |
| `train.ladder` | figure | train | the ladder as a tree |
| `train.curves` | figure | train | the selected run's curves |
| `eval.metrics` | table | eval | the metrics table with intervals |
| `eval.calibration` | figure | eval | the calibration curve |
| `pack.layout` | figure | pack | the bundle's layout |
| `pack.slice` | figure | pack | the serving slice of the module map |
| `conclusion` | section | pack | the conclusion chapter |

A role whose object the chapter does not carry is absent, never guessed: a lane
that stopped at `eval` has no `pack.*` role, an evaluation without a calibration
curve has no `eval.calibration`, and a chapter not drafted yet contributes no
role — the map records the labels the compiled document holds, never a label
planned for a later chapter. The file is written after the compile that produced
the `.aux`, and it carries that `.aux`'s digest. `/vega-show` reads it and never
writes it. How each entry is made:

- **A label the `.aux` can type.** Place the `\label` where LaTeX gives it the
  anchor of its kind: right after the chapter's heading (`\chapter{…}`, or
  `\section{…}` in a class without chapters) for a chapter role and for the
  conclusion, after `\caption{…}` inside the `figure` or `table`, inside a
  numbered `equation` or `align` environment for an equation role (never
  `equation*`, which has no number). Name it with the kind's prefix — `sec:`,
  `fig:`, `tab:`, `eq:`. `hyperref` (required by the preamble) writes the anchor
  (`chapter.`, `figure.`, `table.`, `equation.`) from which `/vega-show` reads
  the kind, and the `kind` recorded here must be that kind.
- **The digest of this compile.** `aux_digest` is `sha256:` followed by the hex
  that `sha256sum docs/main.aux` prints once `latexmk` has finished. Every
  compile rewrites the `.aux`, so the map is rewritten after every compile;
  `/vega-show` refuses a map whose digest is not the current `.aux`'s ("the role
  map is older than the report's .aux — recompile /document-project").
- **`number`, `page`, `title` are optional.** When written, each is copied from
  the label's own `\newlabel{<label>}{{<number>}{<page>}{<title>}{<anchor>}{}}`
  line in that `.aux` (the page as an integer when it is arabic); otherwise it is
  left out. It is never typed from the PDF or from memory: `/vega-show` checks
  each one against the `.aux`.
- **The header.** `lane` is the ledger's lane name, as the briefs' frontmatter
  states it; `generated_at` is the UTC time of writing (ISO 8601); `adapter` is
  `latex-aux`; the `report` paths are relative to the directory holding `docs/`.

---

## The chapter map

Each chapter below names what it must deliver and the figures it is expected to
carry. Adapt — a project without a formal algorithmic core makes Chapter 4
shorter, not absent. (In Vega mode, use the lane's chapter map above instead.)

1. **Introduction & problem framing.** Put the reader in the world before the
   project existed: the problem, why it matters, why it is hard, what this
   project claims to do about it. Close with a roadmap of the document.
   *Figures:* optional context/problem diagram (TikZ).

2. **Business case.** The value argument: stakeholders, the cost of the problem,
   the benefit of the solution, the scope and the explicit non-goals.
   *Figures:* stakeholder map or value-flow diagram (TikZ); optional
   cost/benefit table.

3. **Use cases & actors.** The actors — human and system — and the use cases,
   told as a narrative of how the system is used rather than a bare list.
   *Figures:* UML use-case diagram (PlantUML); actor table.

4. **Theoretical study.** Formalize the problem: a precise statement, the model,
   the formal solution(s) and algorithms, with pseudocode and complexity. This
   is the intellectual core. *Figures:* formal diagrams — automata, graphs, the
   flow of the algorithm (TikZ); algorithm listings; complexity table.

5. **Justification & verification.** Why this solution and not the alternatives;
   the trade-off analysis; what was rejected and what remains open for future
   work; the correctness argument or formal verification where applicable.
   *Figures:* decision/comparison matrix; proof-sketch or invariant diagram.

6. **Architecture — high-level.** The major components, their responsibilities
   and relationships, justified by the theory of Chapters 4–5. *Figures:* UML
   component or deployment diagram (PlantUML); layered block diagram (TikZ).

7. **Architecture — low-level / domain model.** How real-world concepts
   translate into domain-model elements: entities, value objects, relationships.
   *Figures:* UML class diagram of the domain model (PlantUML); a real-world →
   model mapping figure (TikZ).

8. **Dynamics — data, API & sequence flows.** How the system behaves over time:
   data flow, API flow, and one UML sequence diagram per significant use case.
   *Figures:* UML sequence diagrams (PlantUML, one per use case); data-flow or
   activity diagrams.

9. **Implementation & design patterns.** How the architecture became code: the
   design patterns used, the key implementation decisions, instructive code
   excerpts. *Figures:* pattern class diagrams (PlantUML); code listings.

10. **Technology stack.** The complete stack — languages, frameworks, libraries,
    infrastructure, tooling — and why each was chosen. *Figures:* layered stack
    diagram (TikZ); version table.

11. **Tutorial & showcase.** A guided walk-through of using the system, screen
    by screen. *Figures:* annotated screenshot placeholders — framed stubs with
    precise capture instructions.

12. **Development process.** Work-package definitions, the schedule, milestones,
    and the development methodology. *Figures:* Gantt chart (`pgfgantt`);
    work-package table; milestone timeline.

13. *(optional)* **Evaluation & results.** Include only if the project has
    simulations or experiments: methodology, runs, results, comparison study.
    *Figures:* result plots (`pgfplots`); comparison tables.

14. **Future work.** Honest limitations and credible next steps, connected back
    to the open options named in Chapter 5.

15. **Conclusion.** What was built and what was learned, restating the
    contribution against the problem framed in Chapter 1.

---

## The per-chapter cycle

For each chapter, in order:

1. **Announce** the chapter and recap the project context relevant to it.
2. **Elicit** — ask the author the specific questions this chapter needs, in one
   short batch, and read the relevant code yourself.
3. **Draft** the chapter `.tex` file under the narrative contract, in the
   register chosen in Step 0.
4. **Diagram** — write the `.puml` and TikZ figure files, render the PlantUML,
   wire the `\includegraphics`, add any screenshot stubs.
5. **Compile** the whole document with `latexmk -pdf`; read the log; fix every
   error and resolvable warning; re-compile until clean. In Vega mode, then
   rewrite `docs/label-roles.json` from this compile's `.aux` (§ The label-role
   map).
6. **Present** — a two-to-three-sentence summary of the chapter, the figures
   created, the screenshot stubs added, the current page count, and the compile
   status.
7. **Stop.** Wait for the author's edits and explicit approval. Do not begin the
   next chapter unprompted.

---

## Never do this

- Never draft more than one chapter in a turn, and never start the next chapter
  without explicit approval — the pacing is the safeguard.
- Never write fragmented, list-dense text where prose is required; the contract
  is narrative.
- Never invent architecture, stack, or behavior — ground every claim in the
  repository or the author's answers, and flag contradictions instead of hiding
  them.
- Never leave the PDF broken between steps; each step ends with a clean compile
  or a clear explanation of the one blocking error.
- Never fabricate screenshots or results — use captioned stubs.
- In Vega mode, never make the post-report `show` brief a chapter, and never
  record in `docs/label-roles.json` a role whose object the compiled document
  does not carry, or a map older than the `.aux` it names.
- Never pad length to look thorough; match length to the project's real
  complexity.
- Never skip a figure the chapter map expects — a chapter that should be
  illustrated is not done until it is.
- Never let the register loosen the narrative contract — even the terse register
  writes connected, motive-first prose, never bullets in place of reasoning.
- Never change the document's theme, typeface, or LaTeX engine partway through —
  the look is chosen once in Step 0 and held; if a theme needs LuaLaTeX, announce
  the switch rather than performing it silently.

---

## Quality gate

`references/verifier.md` holds a static auditor for this skill's own
instructions, and can be adapted to audit a produced chapter against the
narrative contract. Offer it to the author after the document is complete. In
Vega mode the audit also resolves every number in every chapter to a brief's
`records` entry and reports the unresolved ones — the same check `/vega` runs on
the briefs, applied to the prose that grew from them — and checks that
`docs/label-roles.json` carries the current `.aux`'s digest, names only the
roles of § The label-role map, and binds each to a label that `.aux` holds with
the anchor of the recorded kind.
