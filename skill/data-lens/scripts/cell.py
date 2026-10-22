#!/usr/bin/env python3
"""
cell.py — run ONE analysis cell for a /data-lens dialogue turn and return its result as JSON.

    python3 cell.py RUN_DIR --code cell.py [--timeout 30] [--out result.json] [--label "..."]
    python3 cell.py RUN_DIR --list          # the cells run so far, newest last

The cell is ordinary Python with four names already bound:

    df     the dataset as a pandas DataFrame (the same file analysis.json names)
    ctx    the geometry context from analysis.json — typing, basis, derivations, partition,
           time, spatial: what the forge already settled, so a turn never re-derives it
    prev   the results of the earlier cells in this run directory, oldest first
    fig()  fig(matplotlib_figure) -> registers it as this turn's SVG figure

A cell reports by assigning `result` (any JSON-able value) or by leaving its last expression
as the value; `print()` output is captured separately. Nothing else is returned, so a turn's
answer is composed from a value the analyst can read back, not from a screenful of text.

Sandbox: the process may not open a network connection, may not start another process, and
may write only inside RUN_DIR or the system temp area — never into the project, the dataset or
the model. (The temp area is allowed because ordinary libraries scribble there; the rule is
about the analysis's inputs, not about scratch.) A refusal raises inside the cell and lands in
the error field like any other exception — it is never silently ignored. The cell is also
wall-clock bounded (--timeout, default 30 s).

Exit codes: 0 the cell ran (check `ok` in the JSON for whether it raised); 2 input error.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

RUN = None
ALLOWED = None


def _audit(event, args):
    """Refuse the three things a read-only analysis cell has no business doing."""
    if event in ("socket.connect", "socket.bind", "socket.getaddrinfo", "urllib.Request"):
        raise PermissionError(f"cell sandbox: network access is not available ({event})")
    if event in ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn"):
        raise PermissionError(f"cell sandbox: starting a process is not available ({event})")
    if event == "open":
        path, mode = args[0], (args[1] or "")
        if any(m in str(mode) for m in ("w", "a", "x", "+")):
            try:
                rp = Path(os.fsdecode(path)).resolve()
            except Exception:
                raise PermissionError("cell sandbox: refused a write to an unreadable path")
            # The run directory is where a cell may leave something; the system temp area is
            # allowed because ordinary scientific libraries (threadpoolctl, matplotlib, joblib)
            # write scratch files there while doing nothing of the sort the rule is about.
            # Everything else — the project, the dataset, the model — is out of reach.
            if not any(d == rp or d in rp.parents for d in ALLOWED):
                raise PermissionError(
                    "cell sandbox: a cell may write only inside the run directory "
                    f"({ALLOWED[0]}) or the system temp area, not {rp}")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout("the cell exceeded its wall-clock budget")


def load_run(run_dir):
    a = Path(run_dir) / "analysis.json"
    if not a.is_file():
        raise SystemExit(f"ERROR: {a} not found — run analysis.py into this run directory first")
    doc = json.loads(a.read_text(encoding="utf-8"))
    return doc


def prior_cells(run_dir):
    d = Path(run_dir) / "cells"
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("cell-*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def main(argv=None):
    global RUN, ALLOWED
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the /data-lens run directory (holds analysis.json)")
    ap.add_argument("--code", help="file holding the cell's Python source")
    ap.add_argument("--timeout", type=int, default=30, help="wall-clock budget in seconds")
    ap.add_argument("--out", default=None, help="where to write the result JSON (default: stdout "
                                                "and RUN_DIR/cells/cell-NNN.json)")
    ap.add_argument("--label", default=None, help="a short name for the turn this cell serves")
    ap.add_argument("--list", action="store_true", help="list the cells already run, then exit")
    a = ap.parse_args(argv)

    RUN = Path(a.run_dir).resolve()
    import tempfile as _tf
    ALLOWED = [RUN, Path(_tf.gettempdir()).resolve()]
    for env in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
        v = os.environ.get(env)
        if v:
            try:
                ALLOWED.append(Path(v).resolve())
            except Exception:
                pass
    if not RUN.is_dir():
        print(f"ERROR: run directory {RUN} does not exist")
        return 2
    if a.list:
        for c in prior_cells(RUN):
            print(f"{c['cell']:>3}  {'ok ' if c['ok'] else 'ERR'}  {c.get('label') or ''}  "
                  f"{(c.get('code') or '').splitlines()[0][:70] if c.get('code') else ''}")
        return 0
    if not a.code:
        print("ERROR: --code is required (or --list)")
        return 2
    code_path = Path(a.code)
    if not code_path.is_file():
        print(f"ERROR: cell source {code_path} not found")
        return 2
    code = code_path.read_text(encoding="utf-8")

    doc = load_run(RUN)
    import pandas as pd  # noqa: F401  (bound into the cell namespace below)
    import numpy as np   # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analysis import read_dataset

    ds = doc.get("source", {}).get("path")
    try:
        df = read_dataset(ds)
    except Exception as e:
        print(f"ERROR: cannot read the dataset {ds} named in analysis.json: {e}")
        return 2

    figures = []

    def fig(figure=None, title=None):
        """Register this turn's figure. fig(f) with a matplotlib figure, or fig() for the
        current one. Returns the SVG string it stored."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f = figure if figure is not None else plt.gcf()
        buf = io.StringIO()
        f.savefig(buf, format="svg", metadata={"Date": None, "Creator": None},
                  bbox_inches="tight")
        plt.close(f)
        svg = buf.getvalue()
        figures.append({"title": title, "svg": svg})
        return svg

    ns = {"df": df, "ctx": doc.get("context", {}), "prev": prior_cells(RUN), "fig": fig,
          "pd": pd, "np": np, "analysis": doc, "run_dir": str(RUN), "__name__": "__cell__"}

    out, err, t0 = io.StringIO(), None, time.time()
    value = None
    sys.addaudithook(_audit)
    old_stdout = sys.stdout
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(max(1, a.timeout))
    except (AttributeError, ValueError):
        pass
    try:
        sys.stdout = out
        import ast
        tree = ast.parse(code, mode="exec")
        last = tree.body[-1] if tree.body else None
        if isinstance(last, ast.Expr):
            exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<cell>", "exec"), ns)
            value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)
        else:
            exec(compile(tree, "<cell>", "exec"), ns)
        if value is None:
            value = ns.get("result")
    except _Timeout as e:
        err = {"type": "Timeout", "message": str(e), "traceback": ""}
    except BaseException as e:  # noqa: BLE001 — the cell's failure is data, not a crash
        err = {"type": type(e).__name__, "message": str(e),
               "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__))[-4000:]}
    finally:
        sys.stdout = old_stdout
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass

    from analysis import jsonable
    n = len(prior_cells(RUN)) + 1
    rec = {"cell": n, "label": a.label, "ok": err is None, "code": code,
           "result": jsonable(value) if err is None else None,
           "figure": figures[-1]["svg"] if figures else None,
           "figures": len(figures), "stdout": out.getvalue()[-8000:], "error": err,
           "elapsed_ms": int((time.time() - t0) * 1000),
           "dataset": ds, "rows": int(len(df))}
    cells = RUN / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    (cells / f"cell-{n:03d}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    if figures:
        (cells / f"cell-{n:03d}.svg").write_text(figures[-1]["svg"], encoding="utf-8")
    payload = dict(rec)
    if a.out:
        Path(a.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    slim = dict(payload)
    if slim.get("figure"):
        slim["figure"] = f"<svg {len(slim['figure'])} bytes — saved to cells/cell-{n:03d}.svg>"
    print(json.dumps(slim, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
