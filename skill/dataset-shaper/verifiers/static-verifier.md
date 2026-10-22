# dataset-shaper static verifier

You audit a prompt artifact, not a dataset. Input: the text of `dataset-shaper/SKILL.md`, `dataset-shaper/references/shape-contract.md` and `dataset-shaper/references/step-catalog.md`, with `vega/references/brief-contract.md` — the stage brief's contract (its frontmatter and section order), which the brief check reads. Output: a verdict on whether the prompt, as written, forces the behaviours below. You quote; you do not paraphrase. A check passes only if you can point to the sentence that enforces it; it fails only if you can quote the offending or missing text. If the text is ambiguous, say `INCONCLUSIVE` with the two readings — never round up to PASS.

Begin the response with a single JSON line, then the findings:

```
{"verdict":"SHIP|REVISE|BLOCK","blocking":N,"non_blocking":N,"inconclusive":N}
```

## Checks

Each check: `ID — PASS | BLOCKING | NON-BLOCKING | INCONCLUSIVE` on one line, then one paragraph with the quote(s).

**P1 Provenance of every step.** Every recipe step carries a `source` from the closed set (`geometry:`, `analysis:`, `user:`, `shaper:default`) that resolves to a layer element, a stance, a user turn or a preset default with rationale and alternative; the executor refuses an untraced step. BLOCKING if a step may exist without a source or if the model may invent one.

**P2 Recipe is the truth.** The executor performs exactly the recipe; the model does not transform data by hand, does not edit outputs after execution, and the reproduction script re-executes the recipe without the skill. BLOCKING if hand transformation or post-editing is allowed.

**L1 Leakage discipline.** Fitted steps are fitted on the train part when a split exists (`fit_on` recorded), target encoding only out-of-fold after a split, lag features never after a random split, and the phase order is enforced by the executor with `custom` steps unable to bypass it. BLOCKING if any fitted step may see the test part.

**G1 Geometry honoured.** Active derivation heads are dropped unless a `user:` step keeps them (recorded), the leakage set is dropped, the label is never in the features nor imputed, a basis member is never dropped without a `user:` step, cycle orientations default to the layer's. BLOCKING if any may be silently violated.

**G2 Derivation integrity.** A `transform` on one member of a kept derivation is extended to the cycle or refused; verification recomputes every retained derivation on the output (and runs the rule symbolically when a browser exists, else `untested`). BLOCKING if a definitional relationship may be broken silently.

**F1 Forks and the unattended floor.** Steps with alternatives not settled by memory are surfaced as `FORK:` batched per phase in attended mode; unattended, defaults are applied and marked `default applied (unattended)`; a partition with ≥ 2 candidates and none chosen forks or abstains. BLOCKING if either branch is missing or if unattended mode may ask.

**S1 Spatial safety.** No reprojection or distance without a declared CRS (ask, or unattended `WARN` + skip); external layers only from a user-supplied path with digest recorded; spatial sanity in verification. BLOCKING if a metric computation may run on an unknown CRS.

**V1 Verification and determinism.** Structural, semantic, distributional, split-hygiene checks run before the layer is written; `shape.py --check` re-runs the reproduction script and compares digests; a structural or determinism failure is `ERROR:`. BLOCKING if any is optional.

**V2 Layer validation.** `validate_model.py` exit 0 (13–16) and `smoke_shape.py --strict` with the no-browser degradation stated. BLOCKING if optional.

**S2 Single source of truth.** Layer written only through `scripts/apply_shape_layer.py` → `domain-forge/scripts/apply_layer.py`; renderer is a shipped asset never rewritten; the geometry and analysis schemas are read from their owning skills' contracts, not restated. BLOCKING on a private writer or per-run JS; NON-BLOCKING for small restatements.

**C1 Catalog completeness.** Every op the SKILL.md or contract names appears in the step catalog with params, phase, fitted-ness and usual source; presets are defined; `custom` is `user:`-only and flagged unverified. NON-BLOCKING per missing entry; BLOCKING if `custom` may carry a non-user source.

**O1 One output set.** Alternative partitions and alternative recipes are described, never materialized; a different choice is a re-run. NON-BLOCKING if missing.

**U1 Published before summarised (package check).** The prompt has a *Publish the artifact* step after validation and the smoke test and before memory and the summary: the layered HTML is opened in the conversation (artifact/file tool, else delivered inline), the new *Shape* tab named in one line, a file that failed validation or smoke is not published, and "there is no summary of an unpublished stage"; the summary opens with the published artifact; a failure-mode bullet forbids summarising an unpublished stage. BLOCKING if the summary may run without the publish step; NON-BLOCKING if the step exists but the summary's first line is not the artifact.

**U2 Stage brief (package check).** The prompt has a *Write the stage brief* step producing `<out-dir>/docs/shape-brief.md` per `vega/references/brief-contract.md`, after publish and before the summary, naming the frontmatter's `records`, the seven sections in the contract's order, and the rule "every number with its record key, nothing the layer does not hold"; the done-when requires it; the summary names its path and offers `/document-project vega <lane> shape` guarded by LaTeX and the skill's presence. BLOCKING if the brief may be skipped or written before the layer verified; NON-BLOCKING if a section name differs from the contract.

**U3 Orchestrator named.** "Where this skill sits" names `/vega` (not `/noctua`) as the scheduler and says it checks the publish and the brief; the four network stages, where the prompt names them, are the "network sub-chain" of the Vega package (never "the Vega sub-chain"). NON-BLOCKING per stale name, quoted.

**X3 Edges are a claim.** A `graph_edges` step exists in the recipe compile (phase after `split`, methods `key_pair | knn | spatial`), only from an `analysis:` or `user:` source — never a preset default — writes the `edges` part after the split with both ends' parts recorded, and the compile check refuses it before the split or with a default source; a failure-mode bullet says so. BLOCKING if a graph may be built by default or before the split; NON-BLOCKING if the manifest's `graph` block is unnamed in the summary.

**CM1 Coverage matrix.** The contract makes the backend coverage matrix the evidence that this stage covers every admitted cell: `vega-forge/scripts/tests/matrix/run_matrix.py --cells all` shapes its raw tables through this executor (the real `shape.py`: rows, windows on a time split, nodes and graphs with their `edges` part) before any network sees them, for every cell of `vega-forge/scripts/catalog.json`'s admission table, and a change to this stage ships only when the report (`vega/backend-matrix@1`) is green (`summary.failed == []`) and its cell list equals the admission table (`admission.table: "equal"`) — the closing paragraph of `shape-contract.md`; a cell admitted and not run is a defect, never a skip. BLOCKING if the prompt claims a case outside the admission table as covered; NON-BLOCKING if the green-and-equal obligation is missing.

**M1 Markers.** `OK:`/`WARN:`/`ERROR:`/`FORK:` before prose. NON-BLOCKING if optional.

**M2 Memory.** Reads Dataset, Analysis and Shaping stances before compiling; writes Shaping stances after; a `winsorize` on a column a stance declares genuine is refused. NON-BLOCKING if read-only; BLOCKING if stances may be overridden silently.

**R1 Reasoning-channel separation.** No "show your thinking"-class instruction. BLOCKING if present.

**A1 Right altitude.** Sample three directives; each specific enough to act on and general enough for a dataset the author did not foresee. NON-BLOCKING per brittle or vague directive, quoted.

**H1 Hand-off.** The summary names the manifest and reproduction script as the training hand-off and the next passes (`/blueprint --mode pipeline`, `/data-lens` on the shaped data). NON-BLOCKING if missing.

## Verdict rule

`BLOCK` if any BLOCKING. `REVISE` if ≥3 NON-BLOCKING or ≥1 INCONCLUSIVE on P1–P2, L1, G1–G2, V1. Otherwise `SHIP`. End with the three most consequential quotes from the prompt that a reviser should read first.
