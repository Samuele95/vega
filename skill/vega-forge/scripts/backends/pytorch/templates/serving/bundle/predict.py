#!/usr/bin/env python3
"""
predict.py — the bundle's one entry point.

    python3 predict.py --input rows.csv --output preds.csv [--format csv|parquet|json]
                       [--history earlier.csv]
    python3 predict.py --one '{"<column>": <value>, ...}' [--history earlier.csv]
    python3 predict.py --check

A thin shell over `{package}.{mod_serving}.predict`, the Predictor adapter: `from predict import
load; load("path/to/bundle").predict(df_raw, history=None)`, one row out per row in; `history`
is the raw rows before the query (a window's context, a lag recipe's past; a graph's context
is its reference graph). `load_inspector(bundle_dir)` opens the inspector on the example rows.

The composition root: `load` verifies the files against MANIFEST.json, imports the Recipe on
this bundle's fitted parameters, restores the weights (ModelStore), reads the serving records
and a graph's reference graph, and builds the `Predictor`; `check_examples` is the one example
comparison. Nothing of the skills is imported: the directory is found first (`--bundle`,
{env}_BUNDLE, this file's, the working directory), the adapters imported then (the Recipe reads
its fitted parameters at import, and they must be this bundle's).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from {package}.{mod_serving}.predict import Predictor                     # noqa: E402

__all__ = ["Predictor", "bundle_root", "check_examples", "load", "load_inspector",
           "load_monitor", "predict", "read_rows", "verify"]
# >>> shape: graph
__all__ += ["load_context"]
# <<< shape

EXAMPLE_ROWS = "examples/rows.csv"
EXAMPLE_EXPECTED = "examples/expected.parquet"
EVIDENCE = "authenticated against MANIFEST.files"


def bundle_root(bundle_dir=None) -> Path:
    """The bundle directory: the one named, else {env}_BUNDLE, else this file's directory when it
    is a bundle, else the working directory. A directory without MANIFEST.json is not one."""
    for cand in (bundle_dir, os.environ.get("{env}_BUNDLE"), ROOT, Path.cwd()):
        if cand and (Path(cand) / "MANIFEST.json").is_file():
            return Path(cand).resolve()
    named = bundle_dir or os.environ.get("{env}_BUNDLE")
    raise FileNotFoundError(f"no bundle at {named}: MANIFEST.json not found" if named else
                            "no bundle found: pass --bundle <dir> or set {env}_BUNDLE")


def _recipe(root: Path):
    """The Recipe adapter, imported against this bundle's fitted parameters (read once, at
    import, from VEGA_FITTED): one process serves one bundle, and a second one is refused."""
    fitted = (root / "fitted.json").resolve()
    name = "{package}.{mod_features}.preprocess"
    current = os.environ.get("VEGA_FITTED")
    if name in sys.modules and current and Path(current).resolve() != fitted:
        raise RuntimeError(f"this process already loaded the recipe of another bundle "
                           f"({current}); one bundle per process")
    os.environ["VEGA_FITTED"] = str(fitted)
    from {package}.{mod_features} import preprocess
    return preprocess


def _read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def verify(bundle_dir=None, roles=("data",)) -> Path:
    """The bundle directory, once every file MANIFEST.json gives one of `roles` (`data`: weights,
    spec, fitted parameters, schema, monitor reference, reference graph; `example`: the shipped
    rows and answers) hashes to its pin. An altered file is a refusal naming it."""
    import hashlib
    root = bundle_root(bundle_dir)
    manifest = _read(root, "MANIFEST.json")
    files, recorded = manifest.get("files") or {}, manifest.get("roles")
    if not recorded:
        raise ValueError("MANIFEST.json records no file roles; the bundle cannot say which of "
                         "its files to trust, so it refuses to answer")
    for role in roles:
        for rel in recorded.get(role) or ():
            path, pinned = root / rel, files.get(rel)
            if (not pinned or not path.is_file()
                    or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != pinned):
                raise ValueError(f"{rel} is not the file MANIFEST.json pinned; the bundle "
                                 "refuses to answer from altered files")
    return root


def load(bundle_dir=None) -> Predictor:
    """The bundle's adapters wired together: the Predictor. Composition lives here, not in the
    Predictor's module (an adapter importing its siblings could not be handed others)."""
    root = verify(bundle_dir)
    preprocess = _recipe(root)
    from {package}.domain import NetSpec
    manifest = _read(root, "MANIFEST.json")
    spec = NetSpec.from_dict(_read(root, "netspec.json"))
    # the version every answer carries: the weights' digest, the one the store verifies
    model_version = manifest.get("model_version")
    if not model_version:
        raise ValueError("MANIFEST.json records no model_version; the weights cannot be "
                         "verified, so the bundle refuses to answer")
    schema = _read(root, manifest.get("input_schema", "schema.json"))
    model = _restore(root, manifest, spec)
    from {package}.{mod_serving}.records import ServingRecords
    records = ServingRecords.of(spec, manifest, schema, model_version)
    context = None
    # >>> shape: graph
    context = load_context(root, preprocess) if records.sample.shape.value == "graph" else None
    # <<< shape
    return Predictor(model, preprocess.transform, records, context)


def _restore(root: Path, manifest: dict, spec):
    """The network, rebuilt from the spec and loaded through the ModelStore, handed both pins
    (the file's digest, the weights'): a swapped file refuses even if internally consistent."""
    from {package}.domain import Checkpoint
    from {package}.{mod_model} import network
    from {package}.{mod_store} import store
    files = manifest.get("files") or {}
    weights = next((root / n for n in files if n.startswith("weights.")), root / "weights.pt")
    ref = Checkpoint.of(weights, files.get(weights.name) or "", spec_digest=spec.digest,
                        weights_digest=manifest["model_version"])
    return store.ModelRegistry().load(ref, spec=spec, build=network.build_model,
                                      run_dir=str(root))


def load_inspector(bundle_dir=None):
    """`Predictor.inspector()` opened on the example rows: its `status()` says whether it is
    ready, or why not. A bundle packed with --no-inspect has none (LookupError)."""
    inspector = load(bundle_dir).inspector()
    inspector.open(_example_query(verify(bundle_dir, roles=("example",))))
    return inspector


def _example_query(root: Path):
    """`examples/rows.csv` as the query `open()` explains (every sample, up to 16, summarised)."""
    from {package}.{mod_serving}.inspection.query import ExplainQuery
    rows = read_rows(root / EXAMPLE_ROWS, root).reset_index(drop=True)
    return ExplainQuery(rows, None, max(len(rows) - 1, 0), detail="summary")


# >>> shape: graph
def load_context(root, recipe_module):
    """The reference graph of a graph bundle (`graph/*`, vega/graph-reference@1), verified, for
    the Recipe `recipe_module` whose `edges_for` / `edges_within` attach a request to it."""
    root = verify(root)
    from {package}.domain import NetSpec
    from {package}.{mod_serving}.graphs import GraphContext
    spec = NetSpec.from_dict(_read(root, "netspec.json"))
    return GraphContext.load(root, recipe_module, spec.feature_contract)
# <<< shape


def load_monitor(bundle_dir=None):
    """The Monitor over the reference this bundle carries: `load_monitor().check(window,
    load().predict(window))` on windows of served rows. "Unseen" is the Recipe's word, so it
    asks the Recipe port; classes read through the LabelSet the head was trained against."""
    root = verify(bundle_dir)
    preprocess = _recipe(root)
    from {package}.domain import LabelSet
    from {package}.{mod_monitor} import monitor
    labels = LabelSet.from_spec(_read(root, "netspec.json"))
    return monitor.Monitor(_read(root, "monitor-reference.json"), recipe=preprocess.transform,
                           labels=labels)


def check_examples(bundle_dir=None, predictor=None) -> dict:
    """`examples/rows.csv` answered again against `examples/expected.parquet` at its `__row__`
    (by position without one) — compared fields within the round-trip tolerance, decided fields
    exactly, no compared row unanswered — once the `data` and `example` files are verified."""
    root = verify(bundle_dir, roles=("data", "example"))
    manifest = _read(root, "MANIFEST.json")
    expected = read_rows(root / EXAMPLE_EXPECTED)
    got = (predictor or load(root)).predict(read_rows(root / EXAMPLE_ROWS, root))
    return _Comparison.of(manifest, got, expected).result()


class _Comparison:
    """The answers against the expected frame; each disagreement is a line of `problems`."""

    def __init__(self, output: dict, got, expected, tolerance: float):
        import numpy as np
        self.got, self.expected, self.tolerance = got, expected, tolerance
        self.positional = "__row__" not in expected.columns
        self.at = (np.arange(len(expected)) if self.positional else
                   expected["__row__"].to_numpy(dtype="int64"))
        pairs = {"compared": [f for f, _ in output.get("compared") or ()],
                 "decided": [f for f, _ in output.get("decided") or ()]}
        # a set without __row__ (packed before v2.3) is held to the fields it carries
        self.missing = [] if self.positional else [
            f for f in pairs["compared"] + pairs["decided"] if f not in expected.columns]
        self.compared = [f for f in pairs["compared"] if f in expected.columns]
        self.decided = [f for f in pairs["decided"] if f in expected.columns]
        self.problems = []

    @staticmethod
    def of(manifest: dict, got, expected) -> "_Comparison":
        tolerance = ((manifest.get("roundtrip") or {}).get("tolerance") or {}).get("value")
        if tolerance is None:
            raise ValueError("MANIFEST.json records no roundtrip.tolerance; the examples "
                             "cannot be compared without the bundle's own tolerance")
        return _Comparison(manifest.get("output") or {}, got, expected, float(tolerance))

    def result(self) -> dict:
        self._check_rows()
        worst, decisions, unanswered = None, False, []
        if not self.problems:
            worst, decisions, unanswered = self._worst(), self._decisions(), self._unanswered()
        return {"rows": int(len(self.at)), "columns": len(self.compared),
                "decisions": len(self.decided), "max_abs_diff": worst,
                "tolerance": self.tolerance, "decisions_match": decisions,
                "labels_match": decisions, "unanswered": unanswered,
                "positional": self.positional, "passed": not self.problems,
                "problems": list(self.problems), "evidence": EVIDENCE}

    def _check_rows(self) -> None:
        n, fields = len(self.got), self.compared + self.decided
        if self.missing:
            self.problems.append(f"{EXAMPLE_EXPECTED} lacks {', '.join(self.missing)}, which "
                                 "MANIFEST.output compares or decides")
        if not self.compared:
            self.problems.append(f"{EXAMPLE_EXPECTED} carries none of MANIFEST.output's "
                                 "compared fields")
        absent = [f for f in fields if f not in self.got.columns]
        if absent:
            self.problems.append(f"the answer carries no {', '.join(absent)}")
        if self.positional and len(self.expected) != n:
            self.problems.append(f"{len(self.expected)} expected row(s) for {n} example "
                                 "row(s); a set without __row__ pairs them by position")
        if len(self.at) and (self.at.min() < 0 or self.at.max() >= n):
            self.problems.append(f"__row__ points outside the {n} rows of {EXAMPLE_ROWS}")

    def _worst(self) -> float:
        """max |answer − expected| over the compared fields (missing on one side: infinite)."""
        import numpy as np
        worst, where = 0.0, None
        for field in self.compared:
            mine = self.got[field].to_numpy(dtype="float64", na_value=np.nan)[self.at]
            theirs = self.expected[field].to_numpy(dtype="float64", na_value=np.nan)
            with np.errstate(invalid="ignore"):
                diff = np.abs(mine - theirs)
            diff[(mine == theirs) | (np.isnan(mine) & np.isnan(theirs))] = 0.0
            diff[np.isnan(diff)] = np.inf
            if diff.size and float(diff.max()) > worst:
                k = int(np.argmax(diff))
                worst, where = float(diff[k]), (field, int(self.at[k]), mine[k], theirs[k])
        if worst > self.tolerance:
            field, row, mine, theirs = where
            self.problems.append(f"{field} at example row {row}: the bundle answers {mine!r}, "
                                 f"the examples expect {theirs!r} (|delta| {worst:.3g} > the "
                                 f"tolerance {self.tolerance:.3g})")
        return worst

    def _decisions(self) -> bool:
        for field in self.decided:
            mine = self.got[field].to_numpy(dtype=object)[self.at]
            for k, (a, b) in enumerate(zip(mine, self.expected[field].to_numpy(dtype=object))):
                if not _same(a, b):
                    self.problems.append(f"{field} at example row {int(self.at[k])}: the "
                                         f"bundle answers {a!r}, the examples expect {b!r}")
                    return False
        return True

    def _unanswered(self) -> list:
        if "answered" not in self.got.columns:
            return []
        done = self.got["answered"].to_numpy(dtype=bool)[self.at]
        missed = [int(r) for r, ok in zip(self.at, done) if not ok]
        if missed:
            self.problems.append(f"{len(missed)} compared example row(s) went unanswered "
                                 f"(first: row {missed[0]})")
        return missed


def _same(a, b) -> bool:
    """Numbers by value (1 == 1.0), names by text, missing (None, NaN, NA) only as missing."""
    import math
    import pandas as pd
    missing = [v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))
               for v in (a, b)]
    if any(missing):
        return all(missing)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def read_rows(path, bundle_dir=None, nrows=None):
    """Raw rows from a file, the columns schema.json declares as text read as text (a blank cell
    in a code column would turn 80 into 80.0, a category the recipe never saw); `nrows`: a cap."""
    import pandas as pd
    if str(path).endswith(".parquet"):      # by Arrow's own filesystem: no Python in its threads
        from pyarrow import fs              # (a Python file object aborts at exit, A40)
        frame = pd.read_parquet(path, filesystem=fs.LocalFileSystem())
        return frame if nrows is None else frame.head(int(nrows))
    root = bundle_root(bundle_dir)
    manifest = _read(root, "MANIFEST.json")
    schema = _read(root, manifest.get("input_schema", "schema.json"))
    text = {c["name"]: "string" for c in schema.get("input") or []
            if c.get("dtype") == "string"}
    return pd.read_csv(path, dtype=text, nrows=nrows)


def predict(rows, bundle_dir=None, history=None):
    import pandas as pd                     # the README's one-liner: rows -> the answer
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return load(bundle_dir).predict(frame, history=history)


# --------------------------------------------------------------------------- the CLI

def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=None,
                    help="the bundle directory (default: {env}_BUNDLE, this file's directory, "
                         "or the working directory)")
    ap.add_argument("--input", help="raw rows in the input schema (csv or parquet)")
    ap.add_argument("--output", help="where the answers go")
    ap.add_argument("--format", default="csv", choices=("csv", "parquet", "json"))
    ap.add_argument("--one", help="a single row as JSON")
    ap.add_argument("--history", help="the raw rows preceding the query (csv or parquet): a "
                                      "window's context, a lag recipe's past")
    ap.add_argument("--check", action="store_true",
                    help="verify every pinned file and answer the shipped examples again")
    a = ap.parse_args(argv)
    try:
        a.bundle = str(bundle_root(a.bundle))
    except FileNotFoundError as exc:
        ap.error(str(exc))
    if not (a.check or a.one or a.input):
        ap.error("one of --input, --one or --check is required")
    return a


def _run_check(a) -> int:
    """The two checks: every pinned file against its digest, then `check_examples`."""
    try:
        roles = tuple((_read(Path(a.bundle), "MANIFEST.json").get("roles") or {}).keys())
        verify(a.bundle, roles=roles)
        print("ok  every file hashes to what MANIFEST.json records")
        result = check_examples(a.bundle)
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if not result["passed"]:
        print("ERROR: the bundle does not reproduce its examples: "
              + "; ".join(result["problems"]))
        return 1
    print(f"ok  {result['rows']} example row(s) reproduce within {result['tolerance']:.3g} "
          f"(worst |delta| {result['max_abs_diff']:.3g}; {result['decisions']} decided "
          "field(s) equal)")
    print("2 checks passed")
    return 0


def _run_one(a, history) -> int:
    answer = load(a.bundle).predict_one(json.loads(a.one), history=history)
    print(json.dumps(answer, indent=2, default=str))
    return 0


def _run_input(a, history) -> int:
    out = load(a.bundle).predict(read_rows(a.input, a.bundle), history=history)
    if not a.output:
        print(out.to_string())
        return 0
    if a.format == "parquet":
        out.to_parquet(a.output, index=False)
    elif a.format == "json":
        Path(a.output).write_text(out.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        out.to_csv(a.output, index=False)
    flagged = int((out["warnings"].astype(str) != "").sum()) if "warnings" in out else 0
    missed = int((~out["answered"].astype(bool)).sum()) if "answered" in out else 0
    print(f"OK: {len(out)} answer(s) -> {a.output}"
          + (f"; {flagged} row(s) carry a warning" if flagged else "")
          + (f"; {missed} row(s) unanswered (see their warnings)" if missed else ""))
    return 0


def main(argv=None) -> int:
    a = _parse(argv)
    if a.check:
        return _run_check(a)
    history = read_rows(a.history, a.bundle) if a.history else None
    return _run_one(a, history) if a.one else _run_input(a, history)


if __name__ == "__main__":
    sys.exit(main())
