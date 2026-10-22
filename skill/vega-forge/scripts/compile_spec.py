#!/usr/bin/env python3
"""
compile_spec.py — validate a `netspec.json` against the chain that produced it.

    python3 compile_spec.py --spec netspec.json --model MODEL.html
                            [--pipeline pipeline.json] [--backend pytorch]
                            [--shaped-dir shaped/] [--memory .claude/domain-forge-memory.md]
                            [--check-only] [--out spec_resolved.json] [--json] [--explain]

The compiler is the enforcement mechanism of the rule the skill exists to keep: **no choice
without a traced source**. It is a set of named rules over one immutable context, and each
rule cites the contract clause it enforces — so a refusal can be argued with the contract
rather than with the script. `--explain` prints the rule and clause beside every marker.

Design (the same discipline `code-contract.md` §4 imposes on the emitted code, applied here):
functional core, imperative shell — every rule is a pure `Context -> Markers`, and only
`main` reads argv, prints and chooses an exit code. This file holds the context, the rule
registry, the core and the shell; the rules live by what they judge:

  rules_chain.py    P0 P1 B1 S1 D1a-D1e K1 M2   provenance, the seal, the blueprint, memory
  rules_task.py     T1a-T1e D2a D2b             the task, its fields, the head, the inputs
  rules_blocks.py   G1a-G1d G2 B2               the grammar, admission, the shape walk, coverage
  rules_sample.py   W1 W3 R1                    windows, the future, the runtime
  rules_graph.py    W2                          the graph sample
  rules_regularization.py  R2                   one regularization declaration, live paths

`--out` writes the spec **resolved**: the declared regularization values written into the live
paths the emitted code builds with (`vega_regularization.resolve`; netspec-contract §3, R2).

Contracts: references/netspec-contract.md §1-§3, references/family-catalog.md §1-§7,
references/code-contract.md §1 and §6.

Exit codes: 0 the spec compiles (warnings may still be printed); 2 it does not, or an input
could not be read.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rules_blocks as blocks_  # noqa: E402
import rules_chain as chain  # noqa: E402
import rules_graph as graph_  # noqa: E402
import rules_regularization as reg_  # noqa: E402
import rules_sample as sample_  # noqa: E402
import rules_task as task_  # noqa: E402
import vega_regularization  # noqa: E402
from vega_core import (Catalog, ChainInputs, Choice, Markers,  # noqa: E402
                       SourceResolver, error, read_json, walk_blocks, warn)

SCHEMA = chain.SCHEMA
BACKENDS = HERE / "backends"
CATALOG = HERE / "catalog.json"

CHOICE_KEYS = ("family", "loss", "metrics", "optimizer", "schedule", "early_stopping",
               "init", "training")

#: the columns of a sample that are keys: they order or group rows and are never features
KEY_FIELDS = ("series_by", "order_by", "graph_id")


# --------------------------------------------------------------------------- the context

@dataclass(frozen=True)
class Context:
    """Everything a rule may read, and nothing it may write.

    A rule takes this and returns `Markers`. It cannot reach a file, argv or stdout, which is
    what makes the rule set testable one rule at a time and the output one function's job.
    """

    spec: Mapping[str, Any]
    catalog: Catalog
    inputs: ChainInputs
    resolver: SourceResolver
    coverage: Mapping[str, Any] | None = None
    backend_name: str = "pytorch"

    @staticmethod
    def of(spec, catalog, inputs, coverage=None, backend_name="pytorch") -> "Context":
        return Context(spec=spec, catalog=catalog, inputs=inputs,
                       resolver=SourceResolver(inputs, catalog), coverage=coverage,
                       backend_name=backend_name)

    # -- the questions the rules ask, asked once and named

    @property
    def task(self) -> Mapping[str, Any]:
        return self.spec.get("task") or {}

    @property
    def kind(self):
        return self.task.get("kind")

    @property
    def effective_kind(self):
        """The kind every reader of the base uses: `task.base_kind` for a `sequence-*` task
        (R-A9), the kind itself otherwise."""
        if str(self.kind).startswith("sequence-") and self.task.get("base_kind"):
            return self.task.get("base_kind")
        return self.kind

    @property
    def task_row(self):
        return self.catalog.task(self.kind)

    @property
    def loss_name(self):
        """The spec's loss, or the task row's default when the spec names none."""
        return ((self.spec.get("loss") or {}).get("name")
                or (self.task_row or {}).get("default_loss"))

    @property
    def data_contract(self) -> Mapping[str, Any]:
        return self.spec.get("data_contract") or {}

    @property
    def sample(self) -> Mapping[str, Any]:
        return self.data_contract.get("sample") or {}

    @property
    def sample_shape(self) -> str:
        return str(self.sample.get("shape") or "row")

    @property
    def keys(self) -> tuple:
        """The derived key columns (netspec-contract §1): the sample's non-null keys."""
        return tuple(str(self.sample[k]) for k in KEY_FIELDS if self.sample.get(k))

    @property
    def features(self) -> tuple:
        return tuple(self.data_contract.get("features") or ())

    @property
    def family_name(self):
        return (self.spec.get("family") or {}).get("name")

    @property
    def family_row(self):
        return self.catalog.family(self.family_name)

    @property
    def training(self) -> Mapping[str, Any]:
        return self.spec.get("training") or {}

    @property
    def blocks(self) -> tuple:
        return tuple(walk_blocks(self.spec.get("blocks")))

    @property
    def heads(self) -> tuple:
        return tuple(b for _, b, _ in self.blocks
                     if isinstance(b, Mapping) and b.get("type") == "head")

    @property
    def embedded_columns(self) -> dict:
        out = {}
        for _, b, _ in self.blocks:
            if isinstance(b, Mapping) and b.get("type") == "embedding":
                for c in (b.get("params") or {}).get("columns") or []:
                    out[c] = b
        return out

    @property
    def declared_metrics(self) -> frozenset:
        m = self.spec.get("metrics") or {}
        names = {m.get("primary"), "loss"} | set(m.get("secondary") or ())
        return frozenset(n for n in names if n)

    @property
    def n_classes(self):
        classes = self.task.get("classes")
        if isinstance(classes, (list, tuple)):
            return len(classes)
        return self.task.get("n_classes")

    @property
    def choices(self) -> tuple:
        """Every traced decision in the spec, with where it sits."""
        out = [Choice(key, self.spec.get(key) or {}) for key in CHOICE_KEYS if key in self.spec]
        for i, reg in enumerate(self.spec.get("regularization") or []):
            label = f"regularization[{i}] ({(reg or {}).get('kind')})"
            out.append(Choice(label, reg or {}))
        for path, block, inherited in self.blocks:
            if not isinstance(block, Mapping):
                continue
            if inherited and "source" not in block:
                continue          # netspec-contract §1: nested blocks inherit provenance
            structural = bool((self.catalog.block(block.get("type")) or {}).get("structural"))
            out.append(Choice(path, block, structural=structural, inherits=inherited))
        return tuple(out)

    def needs_window(self) -> bool:
        return (str(self.kind).startswith("sequence-") or self.kind == "forecast"
                or (self.family_row or {}).get("sample_shape") == "window")


# --------------------------------------------------------------------------- the rule set

@dataclass(frozen=True)
class Rule:
    """A named check with the contract clause behind it (EJ-34/DDD: rules are values, not
    branches buried in one long function)."""

    id: str
    clause: str
    title: str
    check: Callable

    def run(self, ctx: Context) -> Markers:
        return Markers.gather([self.check(ctx)])


RULES = (
    Rule("P0", "netspec-contract §1", "the schema and the required keys", chain.schema),
    Rule("P1", "netspec-contract §1", "every choice carries a resolving source",
         chain.provenance),
    Rule("B1", "netspec-contract §3", "no framework name outside `backend`",
         chain.framework_leak),
    Rule("T1a", "family-catalog.md §1", "the task kind is a catalog row", task_.task_known),
    Rule("T1b", "family-catalog.md §1", "the head matches the task", task_.head_matches_task),
    Rule("T1c", "family-catalog.md §1", "the loss matches the task", task_.loss_matches_task),
    Rule("T1d", "family-catalog.md §1", "the monitors name declared valid metrics",
         task_.monitors),
    Rule("T1e", "netspec-contract §1", "the task carries the fields its readers need",
         task_.task_fields),
    Rule("S1", "netspec-contract §1", "the test part is sealed", chain.sealed_test),
    Rule("D2a", "netspec-contract §3", "input_dim agrees with the manifest", task_.input_dim),
    Rule("D2b", "netspec-contract §3", "nominals are embedded, one-hot or tokenized",
         task_.nominals),
    Rule("G1a", "family-catalog.md §3", "every block type is known", blocks_.known_blocks),
    Rule("G1b", "family-catalog.md §2, §7", "an unknown family needs known blocks; a catalog "
         "family is admitted", blocks_.open_family),
    Rule("G1c", "family-catalog.md §3", "residual widths or a projection",
         blocks_.residual_widths),
    Rule("G1d", "netspec-contract §2", "custom blocks are user-sourced and declared",
         blocks_.custom_blocks),
    Rule("G2", "family-catalog.md §3", "the shape walk reaches the head with the right axes",
         blocks_.shape_walk),
    Rule("W1", "netspec-contract §3", "windows and the time split", sample_.windows),
    Rule("W2", "netspec-contract §3", "graphs: the edges part, the level, the protocol, the "
         "batching", graph_.graphs),
    Rule("W3", "netspec-contract §3", "no bidirectional block reads a future it will not "
         "have", sample_.future),
    Rule("R1", "netspec-contract §3", "the batch, the budget, the normalisation",
         sample_.batch_and_budget),
    Rule("R2", "netspec-contract §3", "one regularization declaration; the live paths are "
         "derived from it", reg_.regularization),
    Rule("B2", "backend-contract.md §3", "the backend covers what the spec uses",
         blocks_.coverage),
    Rule("D1a", "code-contract.md §1", "the blueprint is required", chain.blueprint_present),
    Rule("D1b", "code-contract.md §6", "the feature contract agrees", chain.feature_contract),
    Rule("D1e", "netspec-contract §1", "input.pipeline_digest names the blueprint compiled "
         "against", chain.pipeline_digest),
    Rule("D1c", "code-contract.md §6", "serving.mode is a force on the family",
         chain.serving_mode),
    Rule("D1d", "family-catalog.md §2", "a posture overruled says so", chain.posture),
    Rule("K1", "code-contract.md §1", "the ports this stage needs are drawn",
         chain.ports_this_stage_needs),
    Rule("M2", "netspec-contract §5", "the Network stances are honoured", chain.stances),
)


# --------------------------------------------------------------------------- the core

_STANCE_HEADING = re.compile(r"^#+\s*Network stances\s*$", re.I | re.M)
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def read_stances(path) -> tuple:
    """The Network stances section of the chain's memory file, as lines. HTML comments are not
    stances: the memory template (domain-forge's `memory-template.md`) keeps its illustrative
    lines — a binary task's `loss: bce_with_logits`, a declined family — inside a comment under
    the heading, and an example read as a stance vetoes a spec it was never about (M2)."""
    if not path or not Path(path).is_file():
        return ()
    text = _COMMENT.sub("", Path(path).read_text(encoding="utf-8"))
    match = _STANCE_HEADING.search(text)
    if not match:
        return ()
    tail = text[match.end():]
    nxt = re.search(r"^#+\s", tail, re.M)
    body = tail[:nxt.start()] if nxt else tail
    return tuple(line.strip("-* \t") for line in body.splitlines()
                 if line.strip().startswith(("-", "*")) or ("|" in line and line.strip()))


def _backend_coverage(name, backends_dir) -> tuple:
    """(coverage | None, prelude markers): the named backend's coverage.json."""
    coverage, cov_err = read_json(Path(backends_dir) / name / "coverage.json",
                                  f"the {name} backend's coverage.json")
    if cov_err is not None:
        return None, [error(f"{cov_err.message} — the named backend must exist under "
                            f"scripts/backends/{name}/", rule="B2",
                            clause="backend-contract.md §3")]
    if coverage is None:
        return None, []
    return {**coverage, "name": coverage.get("name", name)}, []


def compile_spec(spec, *, model_html=None, pipeline=None, manifest=None, backend=None,
                 stances=(), catalog_path=CATALOG, backends_dir=BACKENDS) -> Markers:
    """The functional core: a spec and its chain in, markers out. No I/O beyond reading the
    catalog and the named backend's coverage, both of which are the skill's own data."""
    catalog, err = Catalog.load(catalog_path)
    if err is not None:
        return Markers.of(err)
    inputs = ChainInputs.of(model_html=model_html, manifest=manifest, pipeline=pipeline,
                            stances=stances)
    name = backend or (spec.get("backend") or {}).get("name") or "pytorch"
    coverage, prelude = _backend_coverage(name, backends_dir)
    ctx = Context.of(spec, catalog, inputs, coverage=coverage, backend_name=name)
    schema_markers = Markers.gather([chain.schema(ctx)])
    if schema_markers.errors:
        return Markers.gather([prelude, schema_markers])
    return Markers.gather([prelude] + [r.run(ctx) for r in RULES if r.id != "P0"])


# --------------------------------------------------------------------------- the shell

def _parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--model", required=True,
                    help="the chain's HTML model (its layers are read from it)")
    ap.add_argument("--pipeline", default=None, help="the blueprint run's pipeline.json")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--shaped-dir", default=None,
                    help="overrides the spec's input.shaped_dir when the parts moved")
    ap.add_argument("--memory", default=".claude/domain-forge-memory.md")
    ap.add_argument("--check-only", action="store_true", help="validate and report; write nothing")
    ap.add_argument("--out", default=None, help="write the validated spec back out, resolved "
                                                "(the regularization live paths derived)")
    ap.add_argument("--json", action="store_true", help="markers as JSON on stdout")
    ap.add_argument("--explain", action="store_true", help="print the rule and clause behind "
                                                           "every marker")
    return ap


def _read_chain(a, spec) -> tuple:
    """(manifest, manifest error, pipeline, pipeline error) for the spec's chain."""
    manifest_path = (str(Path(a.shaped_dir) / "manifest.json") if a.shaped_dir
                     else (spec.get("input") or {}).get("manifest"))
    manifest, manifest_err = (read_json(manifest_path, "the shaped manifest")
                              if manifest_path else (None, None))
    pipeline_path = a.pipeline or (spec.get("input") or {}).get("pipeline")
    pipeline, pipeline_err = (read_json(pipeline_path, "pipeline.json")
                              if pipeline_path else (None, None))
    return manifest, manifest_err, pipeline, pipeline_err


def _report(a, spec, markers) -> int:
    if a.json:
        print(json.dumps({"ok": markers.ok, "markers": markers.as_json()}, indent=2))
    else:
        for line in markers.lines(verbose=a.explain):
            print(line)
    if not markers.ok:
        print(f"ERROR: the spec does not compile ({len(markers.errors)} error(s), "
              f"{len(markers.warns)} warning(s)) — fix the spec, never the emitted code")
        return 2
    n_blocks = sum(1 for _ in walk_blocks(spec.get("blocks")))
    print(f"OK: {a.spec} compiles against {SCHEMA} — task {(spec.get('task') or {}).get('kind')}"
          f", family {(spec.get('family') or {}).get('name')}, {n_blocks} block(s), backend "
          f"{a.backend or (spec.get('backend') or {}).get('name')}, "
          f"{len(markers.warns)} warning(s)")
    if a.out and not a.check_only:
        resolved = vega_regularization.resolve(spec)
        Path(a.out).write_text(json.dumps(resolved, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"OK: wrote {a.out} (resolved: the declared regularization written into "
              f"{len(vega_regularization.live_values(spec))} live path(s))")
    return 0


def main(argv=None):
    a = _parser().parse_args(argv)
    spec, err = read_json(a.spec, "spec")
    if err is not None:
        print(err.line())
        return 2
    try:
        html = Path(a.model).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read the model: {exc}")
        return 2
    manifest, manifest_err, pipeline, pipeline_err = _read_chain(a, spec)
    if pipeline_err is not None:
        print(pipeline_err.line())
        return 2
    markers = compile_spec(spec, model_html=html, pipeline=pipeline, manifest=manifest,
                           backend=a.backend, stances=read_stances(a.memory))
    if manifest_err is not None:
        markers = Markers.gather([markers, warn(f"{manifest_err.message}; the manifest "
                                                "cross-checks were skipped", rule="IO")])
    return _report(a, spec, markers)


if __name__ == "__main__":
    sys.exit(main())
