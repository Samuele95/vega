"""
lanekit — build and drive the verification program's lanes (D6 §2.2, BUILD-PLAN §2.9).

A lane is one dataset driven through the whole chain — fixture, forge, train, eval, pack, the
bundle's own tests, report, show, serve, and the live checks — and it proves only what it
observed (`lanes/<id>/observed.json`), never what `lanes.json` declares.

Modules:

- `contracts`   every name consumed from another subsystem, in one place (D6 §17.2);
- `drift`       `drift()`, which compares those copies with the catalogs that own them;
- `registry`    `Lane`, `load_lanes()`, the catalogs a lane is checked against, `validate()`;
- `validation`  the registry's schema and cross-checks (T0);
- `routing`     `--changed <range>`: the lanes a change affects (D6 §3.4);
- `skilltree`   the matrix's copy of the skills (repository + domain-forge) and its pins;
- `runner`      commands with logs, budgets and the flake policy (`flakes.json`);
- `stamps`      what `--from` trusts: per-stage input and output digests, chained;
- `context`     what a lane's stages share (`MatrixEnv`, `StageContext`, `StageResult`);
- `stages`      one function per stage, per driver (acceptance, lanekit, golden);
- `lane`        `LaneRun`, the driver that runs a lane's stages in order;
- `checks`      the stage helpers that run in their own process (fixture, authoring, report,
                and the artifact assertions of `assertions`);
- `assertions`  what each stage asserts about the artifacts it produced (D6 §6.1–§6.5);
- `show_assertions`  the show stage's assertions (D6 §6.8);
- `service`     start and stop a bundle's service on a free port;
- `observe`     `observed.json` (`vega/lane-observed@1`) and its schema;
- `tiers`       the static, unit, negatives and report tiers as lists of commands;
- `timing`      `timing.json` / `timing.md` and the flake share warning.

Later work packages add `fixture`, `model_source`, `spec`, `datasets/` (T02), `authoring` and
`report_stub` (T03); `contracts.PROVIDERS` names which stage needs which of them.
"""
from __future__ import annotations

__all__ = ["assertions", "checks", "context", "contracts", "drift", "lane", "observe", "registry",
           "routing", "runner", "service", "show_assertions", "skilltree", "stages", "stamps",
           "tiers", "timing", "validation"]
