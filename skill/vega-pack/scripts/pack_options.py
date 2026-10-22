"""
pack_options — the command line of `pack.py`, parsed once into a frozen `PackOptions`.

Every flag the skill documents is declared here, including the ones a later build fills in:
the phase modules read `options.<name>` and never the raw argv. Three flags are gone — the
story and the report moved to `/vega-show`, which builds them beside the bundle rather than
inside it — and asking for them is a refusal that says where they went, not an unknown-flag
error that leaves the caller guessing.

    options = pack_options.parse(argv)       # PackExit(2) on a removed flag
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from pack_context import PackExit
from vega_pack_core import CATALOG

__all__ = ["PackOptions", "REMOVED_FLAGS", "parse"]

# the story plan and the written report are presentation: /vega-show resolves them into the
# showcase next to the bundle, and the bundle carries neither
REMOVED_FLAGS = ("--story-plan", "--report-aux", "--report-pdf")


@dataclass(frozen=True)
class PackOptions:
    """What the packer was asked for. Field names are the flags' (`--bundle-dir` is
    `bundle_dir`); an alias (`--run-dir`, `--out-dir`) lands on the field it aliases."""

    campaign: str
    shaped: str
    bundle_dir: str
    run: str | None = None
    eval: str | None = None
    pipeline: str | None = None
    model: str | None = None
    slug: str | None = None
    examples: str | None = None
    ship_anyway: str | None = None
    include: str = "serving"
    dataset_shaper_dir: str | None = None
    serve: bool | None = None
    service_title: str | None = None
    service_port: int = 8765
    row_noun: str | None = None
    service_description: str | None = None
    demo_sample: int = 20
    monitor_plan: str | None = None
    export: tuple = ()
    re_emit_serving: bool = False
    reason: str | None = None
    inspect: bool = True
    inspect_weights: str = "full"
    unattended: bool = False


def parse(argv=None, description=None) -> PackOptions:
    """The options, or `PackExit(2)` for a removed flag; argparse's own errors exit 2 too."""
    argv = list(argv) if argv is not None else None
    _refuse_removed(argv)
    ap = _parser(description)
    a = ap.parse_args(argv)
    if a.reason and not a.re_emit_serving:
        ap.error("--reason explains a --re-emit-serving run; it means nothing without one")
    return PackOptions(**vars(a))


def _refuse_removed(argv) -> None:
    import sys
    given = sys.argv[1:] if argv is None else argv
    for token in given:
        flag = str(token).split("=", 1)[0]
        if flag in REMOVED_FLAGS:
            raise PackExit(2, f"{flag} moved to /vega-show (vega-show/SKILL.md); the bundle "
                              "carries no story")


def _parser(description=None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _inputs(ap)
    _shape(ap)
    _service(ap)
    _reemit_and_inspection(ap)
    return ap


def _inputs(ap) -> None:
    ap.add_argument("--campaign", "--run-dir", dest="campaign", required=True,
                    help="the campaign directory (--run-dir is the same flag, for a campaign "
                         "the layers' path no longer names)")
    ap.add_argument("--run", default=None, help="default: the run the seal records")
    ap.add_argument("--eval", default=None, metavar="DIR",
                    help="the evaluation of the run (default: <campaign>/eval/<run>)")
    ap.add_argument("--shaped", required=True, help="the shaped directory")
    ap.add_argument("--bundle-dir", "--out-dir", dest="bundle_dir", required=True,
                    help="where the bundle directory <slug>-<run> is written (--out-dir is "
                         "the same flag)")
    ap.add_argument("--pipeline", default=None, help="the blueprint run's pipeline.json")
    ap.add_argument("--model", default=None, help="the .eval.html the bundle came from")
    ap.add_argument("--dataset-shaper-dir", default=None)
    ap.add_argument("--ship-anyway", default=None, metavar="REASON")
    ap.add_argument("--unattended", action="store_true",
                    help="no user present: an unearned verdict without --ship-anyway stops "
                         "with WARN: and no bundle, and every fork takes its recorded "
                         "unattended answer")


def _shape(ap) -> None:
    ap.add_argument("--slug", default=None)
    ap.add_argument("--examples", default=None, help="<n> test rows, or `synthetic`")
    ap.add_argument("--include", default=CATALOG["include"]["default"],
                    choices=tuple(CATALOG["include"]["modes"]),
                    help="what of the emitted package the bundle carries: `serving` (the "
                         "default) is the serving slice and predict.py; `full` is the whole "
                         "package plus train.py and evaluate.py — the complete server-side "
                         "system, with the seal still in force")
    ap.add_argument("--export", default=(), type=_export_formats, metavar="FORMATS",
                    help="a comma list of " + ", ".join(CATALOG["export_formats"])
                         + ": each is exported under exports/ and its parity with the "
                           "predictor checked (default: none)")
    ap.add_argument("--monitor-plan", default=None,
                    help="the packer's plan: one check per pipeline.json monitoring concern "
                         "(kinds psi, rate, unseen_rate, class_recall); references are "
                         "measured here on the train part (unseen rate on valid); required "
                         "when the blueprint's Monitor port says filled_by /vega-pack")


def _service(ap) -> None:
    ap.add_argument("--serve", dest="serve", action="store_true", default=None,
                    help="write the online service (default: on when the blueprint's mode is "
                         "online)")
    ap.add_argument("--no-serve", dest="serve", action="store_false")
    ap.add_argument("--service-title", default=None,
                    help="what the service calls itself (default: the blueprint's system name)")
    ap.add_argument("--service-port", type=int, default=8765,
                    help="the port the service listens on (default 8765; 8000 is the usual "
                         "port of other local model servers, vLLM among them)")
    ap.add_argument("--row-noun", default=None,
                    help="the word for one input row on the page and in the docs (default: row)")
    ap.add_argument("--service-description", default=None,
                    help="one paragraph under the title (default: derived from the schema)")
    ap.add_argument("--demo-sample", type=int, default=20,
                    help="items per stratum of the demonstration sample, drawn from the "
                         "train part (0: none)")


def _reemit_and_inspection(ap) -> None:
    ap.add_argument("--re-emit-serving", action="store_true",
                    help="rebuild the bundle already at <bundle-dir>/<slug>-<run> from the "
                         "same inputs, changing only what the serving templates produce; a "
                         "change to anything the evaluation pinned is refused (exit 3) and "
                         "the bundle is left as it was")
    ap.add_argument("--reason", default=None,
                    help="why the serving adapters are re-emitted (recorded in "
                         "MANIFEST.reemits)")
    ap.add_argument("--no-inspect", dest="inspect", action="store_false",
                    help="leave the inspection routes (/v1/network, /v1/explain) out")
    ap.add_argument("--inspect-weights", default="full", choices=("full", "summary"),
                    help="how much of the weights the inspection routes expose (default: full)")


def _export_formats(text) -> tuple:
    formats = tuple(f.strip() for f in str(text).split(",") if f.strip())
    unknown = [f for f in formats if f not in CATALOG["export_formats"]]
    if unknown or not formats:
        raise argparse.ArgumentTypeError(
            f"{', '.join(unknown) or 'nothing'}: the backend exports "
            + ", ".join(CATALOG["export_formats"]))
    return formats
