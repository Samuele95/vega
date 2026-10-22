# dataset-forge static verifier

You audit a prompt artifact, not a dataset. Input: the text of `dataset-forge/SKILL.md` and `dataset-forge/references/report-contract.md`. Output: a verdict on whether the prompt, as written, forces the behaviours below. You quote; you do not paraphrase. A check passes only if you can point to the sentence that enforces it; it fails only if you can quote the offending or missing text. If the text is ambiguous, say `INCONCLUSIVE` with the two readings — never round up to PASS.

Begin the response with a single JSON line, then the findings:

```
{"verdict":"SHIP|REVISE|BLOCK","blocking":N,"non_blocking":N,"inconclusive":N}
```

## Checks

Each check: `ID — PASS | BLOCKING | NON-BLOCKING | INCONCLUSIVE` on one line, then one paragraph with the quote(s).

**D1 Derived-column detection (done criterion 1).** The prompt requires that a numerically derived column be excluded from the basis and expressed in function of others, and names a mechanism that does it (`linear_derivability`, a rule, a formula). BLOCKING if absent.

**D2 Semantic derivation without numeric evidence (criterion 2).** The prompt requires proposing derivations from meaning *before* consulting evidence and marks such derivations as semantic when unverified. BLOCKING if the prompt lets correlation or FDs alone license a derivation.

**D3 Label candidates are determined columns (criterion 3).** The prompt defines label candidates as rule heads / FD-determined columns, requires the leakage set to be stated, and forbids choosing by column name. BLOCKING if any of the three is missing.

**D4 Disagreement visibility (criterion 4).** The prompt mandates that semantic/symbolic/empirical disagreements appear explicitly in the artifact, with a schema slot for them. BLOCKING if disagreements may be resolved silently.

**D5 Validation and engine-answerability (criterion 5).** The prompt requires `validate_model.py` exit 0 and that the primitive/derived question be answerable from the engines (rules present in `model-swrl`/`model-horn`, symbolic verification recorded). BLOCKING if validation is optional.

**P1 Provenance triple.** Three channels, each with confirmed/refuted/untested; `symbolic: confirmed` requires a headless engine run (`run_query.py`) behind the count, and a rule no engine ran is `untested`, never `confirmed`. BLOCKING if the model may assert symbolic verification from its own arithmetic or mental simulation.

**P2 Reasoner limits.** The prompt directs the reader to domain-forge's *Not implemented* list and routes string/date derivations to Horn (Prolog-verified) rather than SWRL. NON-BLOCKING if the pointer exists but the routing is vague; BLOCKING if SWRL is prescribed for operations the reasoner lacks.

**F1 Fork behaviour.** On ≥2 defensible label candidates: stop, `FORK:` marker, one targeted question naming candidates and the separating difference. Abstention floor for unattended runs (`provenance: abstained`, no question). BLOCKING if either branch is missing or the prompt instructs asking a user who may be absent.

**I1 Insight admission rule.** The prompt states the `describe()`/`corr()` test and gives one example on each side. NON-BLOCKING if the rule exists without an example.

**C1 Consequence blocks.** Every decision type listed (retyping, basis choice within a cycle, derivation kept/rejected, partition candidate) is required to carry alternatives and concrete downstream effects; padding is forbidden and the "none" escape is provided. BLOCKING if consequences are optional.

**N1 Narrative register.** The Abstract and readings are bound to the audit-chapter rubric (motive before mechanism, connected paragraphs, no bullet-list argument). NON-BLOCKING if bound by reference only.

**R1 Reasoning-channel separation (M6).** No instruction asks the model to reveal, transcribe, or narrate its internal reasoning as output. Task-mandated justification (readings, consequence prose) is fine. BLOCKING on any "show your thinking"-class instruction.

**S1 Single source of truth.** The prompt reuses domain-forge's template, validator and contracts by reference and does not restate their schemas. NON-BLOCKING for small restatements; BLOCKING if the prompt carries its own copy of the Turtle/JSON-LD/SWRL contract that could drift.

**U1 Published before summarised (package check).** The prompt has a *Publish the artifact* step after validation and the smoke test and before memory and the summary: the layered HTML is opened in the conversation (artifact/file tool, else delivered inline), the new *Geometry* tab named in one line, a file that failed validation or smoke is not published, and "there is no summary of an unpublished stage"; the summary opens with the published artifact; a failure-mode bullet forbids summarising an unpublished stage. BLOCKING if the summary may run without the publish step; NON-BLOCKING if the step exists but the summary's first line is not the artifact.

**U2 Stage brief (package check).** The prompt has a *Write the stage brief* step producing `<run-dir>/docs/forge-data-brief.md` per `vega/references/brief-contract.md`, after publish and before the summary, naming the frontmatter's `records`, the seven sections in the contract's order, and the rule "every number with its record key, nothing the layer does not hold"; the done-when requires it; the summary names its path and offers `/document-project vega <lane> forge-data` guarded by LaTeX and the skill's presence. BLOCKING if the brief may be skipped or written before the layer verified; NON-BLOCKING if a section name differs from the contract.

**U3 Orchestrator named.** "Where this skill sits" names `/vega` (not `/noctua`) as the scheduler and says it checks the publish and the brief; the four network stages, where the prompt names them, are the "network sub-chain" of the Vega package (never "the Vega sub-chain"). NON-BLOCKING per stale name, quoted.

**X2 Platform writer.** The layer is written through `scripts/apply_geometry_layer.py` → `domain-forge/scripts/apply_layer.py` with the shipped `geometry-render.js` (the model writes no JavaScript and does not append the layer by hand), then `validate_model.py` (13–16) and `smoke_geometry.py`. BLOCKING if the model may hand-write the render script or append the block itself.

**M1 Markers.** `OK:`/`WARN:`/`ERROR:`/`FORK:` are required before prose at each stage, and the script's markers are surfaced. NON-BLOCKING if present but not required to precede prose.

**A1 Right altitude.** Sample three directives at random. Each is specific enough to act on and general enough to survive a dataset the author did not foresee. NON-BLOCKING per brittle or vague directive found; quote it.

**X1 Interactive explorer.** The prompt requires the Geometry tab to be the four linked views of contract §6, forbids recomputing shipped numbers in JS, and requires consequence blocks for every cycle orientation and candidate (not only the chosen ones) so the explorer's what-if has content. BLOCKING if the explorer is optional or if consequences are required only for chosen decisions.

**H1 Hand-off.** A chosen partition is recorded in the layer (`partitions.chosen`, its provenance) and is *not* materialized here: the summary hands `/data-lens <out>` or `/dataset-shaper <out> --target <label>`; no `nn-*` skill or `nn-data` artifact is named anywhere (the family is retired; `input_dim` is the key). BLOCKING if the prompt still emits an `nn-data` artifact; NON-BLOCKING per stale `nn-*` name, quoted.

## Verdict rule

`BLOCK` if any BLOCKING. `REVISE` if ≥3 NON-BLOCKING or ≥1 INCONCLUSIVE on D1–D5. Otherwise `SHIP`. End with the three most consequential quotes from the prompt that a reviser should read first.
