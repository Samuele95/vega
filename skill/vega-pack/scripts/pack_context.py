"""
pack_context — what one pack run knows, read once and handed to every phase.

`pack.py` reads the chain's records into a `Campaign`, assembles the bundle's inputs, and then
builds one frozen `PackContext`. Every phase module takes that context (and, after the round
trip, the loaded predictor) and returns its manifest record(s) as a dict keyed by where the
record goes in `MANIFEST.json`:

    output_record.records(ctx)            -> {"output": …, "sample": …}
    roundtrip_plan.run(ctx, predictor)    -> {"roundtrip": …}
    serving_pack.render(ctx)              -> {"serve": …}
    inspection_pack.check(ctx, predictor) -> {"serve.inspection": …}

A dotted key nests (`serve.inspection` lands inside `serve`), and `merge_records` applies the
records in that order: every top-level key first, then the nested ones. A phase that refuses
raises `PackExit`; `pack.py` prints its lines and returns its exit code. Nothing here prints on
its own except the two marker helpers, which print what they record — the markers are the
run's account, and the same lines reach the terminal and `markers.json`.

The context is frozen: a phase reads it and never rebinds a field. Two fields are logs by
design — `markers` (the run's `OK:` / `WARN:` lines) and `stances` (the decisions a user or a
default made, `MANIFEST.stances`) — and a phase appends to them; nothing else is written
through the context.

What a phase changes, it changes from its own module, through one of these and nothing else —
the support modules (`pack_context`, `bundle_inputs`, `bundle_slice`, `bundle_map`,
`bundle_manifest`, `export_pack`) are not edited by later work:

  * a manifest field anywhere: a record under a dotted key (`blueprint.conformance.api_declared`
    lands in the gate's record, `sample.graph` in the sample record);
  * a field the Predictor reads at load: a record whose first segment is `output`, `sample` or
    `graph`, which `bundle_manifest.write_stub` also writes into the stub it loads against;
  * a data file for the bundle's code: written under `graph/` or `exports/` (data by prefix,
    beside the root files `bundle_manifest.DATA_FILES` names) and pinned by role; one written
    after the predictor loaded is pinned for its next reader with
    `bundle_manifest.refresh_stub(ctx)`;
  * a package file or an entry point it rendered: `bundle_map.add_map_entries(ctx, source,
    files, points)`, before `bundle_slice.finish` (every phase runs before it);
  * a stance or a marker: appended to `ctx.stances` / through `out_ok` and `out_warn`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from vega_pack_core import BundlePaths, read_json, serving_mode

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parent.parent

__all__ = ["Campaign", "PackContext", "PackExit", "SKILLS", "blueprint_mode", "blueprint_port",
           "merge_records", "out_ok", "out_warn", "read_pipeline"]


class PackExit(Exception):
    """A refusal: the exit code and the line(s) that say why, printed with `level` as prefix.

    `lines` is one line, or a tuple of them when a refusal has several things to say (each
    provenance disagreement, then the refusal itself). `keep_markers` asks `pack.py` to write
    `markers.json` into the bundle before it returns: a round trip that failed leaves its
    account beside the files that failed it.
    """

    def __init__(self, code: int, lines: str | tuple[str, ...], *, level: str = "ERROR",
                 keep_markers: bool = False):
        self.lines = (lines,) if isinstance(lines, str) else tuple(lines)
        super().__init__("; ".join(self.lines))
        self.code = code
        self.level = level
        self.keep_markers = keep_markers


def out_ok(msg, markers):
    markers.append(f"OK: {msg}")
    print(f"OK: {msg}")


def out_warn(msg, markers):
    markers.append(f"WARN: {msg}")
    print(f"WARN: {msg}")


# --------------------------------------------------------------------------- the blueprint

def read_pipeline(pipeline) -> dict:
    if not pipeline:
        return {}
    doc, _err = read_json(Path(pipeline), "pipeline.json")
    return doc or {}


def blueprint_mode(pipeline, campaign_doc) -> str:
    """The serving mode is the blueprint's decision; the campaign only echoes it, when at all."""
    doc = read_pipeline(pipeline)
    return ((doc.get("serving") or {}).get("mode")
            or ((campaign_doc or {}).get("serving") or {}).get("mode") or "batch")


def blueprint_port(pipeline, paths, role) -> dict:
    """The port the blueprint drew for a role — its name and responsibility from pipeline.json,
    its file from the package's map — so rendered adapters carry the baseline's names."""
    doc = read_pipeline(pipeline)
    port = next((dict(p) for p in (doc.get("ports") or []) if p.get("role") == role), {})
    bmap, _err = read_json(paths.pkg / "blueprint-map.json", "blueprint-map.json")
    entry = next((e for e in ((bmap or {}).get("entries") or []) if e.get("role") == role), {})
    port["file"] = entry.get("file")
    port["module"] = entry.get("module") or (entry.get("file") or "").split("/")[0] or None
    return port


# --------------------------------------------------------------------------- the chain's records

@dataclass(frozen=True)
class Campaign:
    """What the chain handed over: the campaign, its ledger and seal, the evaluation of the
    selected run, the shaped dataset's manifest — and, once the bundle's inputs are copied,
    the recipe, the resolved spec, the raw rows and the whole fitted file (`with_inputs`)."""

    dir: Path
    doc: dict
    ledger: dict
    seal: dict
    run_id: str
    eval_dir: Path
    metrics: dict
    threshold: dict | None
    shaped_dir: Path
    shaped_manifest: dict
    recipe: dict | None = None
    spec: dict | None = None
    raw: object = None
    fitted: dict | None = None

    @staticmethod
    def read(options) -> "Campaign":
        """The records every later step reads, or `PackExit(2)` naming the one missing."""
        camp_dir = Path(options.campaign).resolve()
        shaped_dir = Path(options.shaped).resolve()
        doc = _required(camp_dir / "campaign.json", "campaign.json")
        ledger = _required(camp_dir / "experiments.json", "experiments.json")
        seal, err = read_json(camp_dir / "test_seal.json", "test_seal.json")
        if err:
            raise PackExit(2, f"{err} — a bundle ships what /vega-eval measured, and nothing "
                              "here has been measured yet")
        run_id = options.run or seal.get("run")
        eval_dir = Path(options.eval).resolve() if options.eval else camp_dir / "eval" / run_id
        metrics = _required(eval_dir / "metrics.json", f"eval/{run_id}/metrics.json")
        threshold, _ = read_json(eval_dir / "threshold.json", "threshold.json")
        shaped_manifest = _required(shaped_dir / "manifest.json", "the shaped manifest")
        return Campaign(dir=camp_dir, doc=doc, ledger=ledger, seal=seal, run_id=run_id,
                        eval_dir=eval_dir, metrics=metrics, threshold=threshold,
                        shaped_dir=shaped_dir, shaped_manifest=shaped_manifest)

    def with_inputs(self, recipe, spec, raw, fitted) -> "Campaign":
        return replace(self, recipe=recipe, spec=spec, raw=raw, fitted=fitted)

    @property
    def run_dir(self) -> Path:
        return self.dir / "runs" / self.run_id

    @property
    def package(self) -> str:
        return (self.doc.get("input") or {}).get("package") or "pipeline"

    @property
    def src_pkg(self) -> Path:
        """The emitted package the run trained: under the run, else under the campaign."""
        inside = self.run_dir / self.package
        return inside if inside.is_dir() else self.dir / self.package

    @property
    def sources(self) -> dict:
        outputs = self.shaped_manifest.get("outputs") or {}
        return {"netspec": self.run_dir / "netspec.resolved.json",
                "weights": self.run_dir / "checkpoints" / "best.pt",
                "recipe": self.shaped_dir / (outputs.get("recipe") or "recipe.json"),
                "fitted": self.shaped_dir / (outputs.get("fitted") or "fitted.json")}

    @property
    def verdict(self) -> dict:
        return self.metrics.get("verdict") or {}

    @property
    def ledger_row(self) -> dict:
        return next((r for r in self.ledger.get("runs") or [] if r.get("id") == self.run_id),
                    {})

    @property
    def evaluation(self) -> dict:
        return {"seal": self.seal, "run": self.ledger_row, "verdict": self.verdict,
                "metrics": self.metrics}

    @property
    def backend(self) -> dict:
        return self.doc.get("backend") or {}

    @property
    def backend_name(self) -> str:
        return self.backend.get("name", "pytorch")

    @property
    def label(self):
        return self.shaped_manifest.get("label")

    @property
    def split(self) -> dict:
        """The raw-file positions of each part (`__split_raw__`), which the fitted file keeps."""
        return (self.fitted or {}).get("__split_raw__") or {}

    def part_rows(self, part):
        """The raw rows of one part, in the order the shaper recorded them (empty when the
        split records no such part)."""
        return self.raw.iloc[list(self.split.get(part) or [])]


def _required(path, what) -> dict:
    doc, err = read_json(path, what)
    if err:
        raise PackExit(2, err)
    return doc


# --------------------------------------------------------------------------- the run's context

@dataclass(frozen=True)
class PackContext:
    """One pack run, as every phase sees it.

    `options` is the parsed command line (`pack_options.PackOptions`); `paths` the directory
    being built (under `--re-emit-serving`, a staging directory with the bundle's own name);
    `campaign` the chain's records and the bundle's inputs; `package` the emitted package's
    name; `ident` the served identity (`MANIFEST.serve.identity`); `schema` the input schema as
    written to `schema.json`; `copied` the package files the serving slice copied.
    """

    options: object
    paths: BundlePaths
    campaign: Campaign
    package: str
    ident: dict
    schema: dict = field(default_factory=dict)
    copied: tuple = ()
    markers: list = field(default_factory=list)
    stances: list = field(default_factory=list)

    @property
    def spec(self) -> dict:
        return self.campaign.spec or {}

    @property
    def mode(self) -> str:
        return blueprint_mode(self.options.pipeline, self.campaign.doc)

    @property
    def serving(self) -> bool:
        """Whether the serving scaffold is written: `--serve` / `--no-serve` when given, else
        what the blueprint's mode implies (online and stream serve, batch does not)."""
        if self.options.serve is not None:
            return bool(self.options.serve)
        return bool(serving_mode(self.mode).get("serve"))

    @property
    def templates(self) -> Path:
        """The backend's serving templates: the adapters, `bundle/`, `tests/`, `serve/`."""
        return (SKILLS / "vega-forge" / "scripts" / "backends" / self.campaign.backend_name
                / "templates" / "serving")

    def port(self, role) -> dict:
        return blueprint_port(self.options.pipeline, self.paths, role)

    def pipeline(self) -> dict:
        return read_pipeline(self.options.pipeline)


def merge_records(target: dict, records: dict) -> dict:
    """Each record under its key — a dotted key nests, a dict merges into a dict already
    there — top-level keys first, so `serve` exists before `serve.inspection` lands in it."""
    ordered = sorted(records.items(), key=lambda kv: kv[0].count("."))
    for key, value in ordered:
        if value is None:
            continue
        node = target
        *parents, last = key.split(".")
        for name in parents:
            node = node.setdefault(name, {})
        if isinstance(node.get(last), dict) and isinstance(value, dict):
            node[last].update(value)
        else:
            node[last] = value
    return target
