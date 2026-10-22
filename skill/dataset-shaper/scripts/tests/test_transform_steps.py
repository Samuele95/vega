#!/usr/bin/env python3
"""
test_transform_steps.py — the transform-only module's step functions (`transform_steps_source`).

    python3 -m pytest test_transform_steps.py      (or: python3 test_transform_steps.py)

`--transform-only` runs each applied step's lines inside short functions (`_step_<id>`, listed
in `_STEPS`) so the module it ships meets vega-forge code-contract §5.1 (C11: at most 30
statements per function). The lines are the executor's, unchanged; the emitter only decides
where one function ends and the next begins. These tests hold that to being a pure
re-arrangement:

  equivalence   the split module and the same lines run flat, in one function (the module as it
                was emitted before the split), leave the same parts and the same side effects in
                the same order — on hand-written cases and on 300 generated step programs whose
                names collide across steps (a binder read by a later step, a module-constant
                binder `_f = FITTED[..]['fitted']` and an import read again in every function,
                a rebinding that reads the name it binds, a loop over a name it rebinds,
                comprehension and lambda names, statements sharing a line, a branch, an
                augmented assignment);
  budget        every function is at most STEP_BUDGET statements whenever no single unit is
                larger, and a step longer than that is cut into `_step_<id>`, `_step_<id>_2` …;
  layout        one function per step for short steps, `_STEPS` in recipe order, a skipped or
                comment-only step is a comment and no function, an empty recipe is
                `_STEPS = ()`, a step whose lines do not parse is refused naming the step;
  guard         `unbound_step_names` names a step function reading a name nobody binds, and
                `write_transform_module` refuses to write such a module;
  end to end    a recipe whose steps each run past 30 statements through `shape.py`,
                `--transform-only` and the twin check (`verify_shape.py --transform-only`) on
                every part, every function of the emitted module within the budget;
  module size   a module past code-contract §5's 400 lines (C13) moves its step functions to
                `<stem>_steps*.py` beside it, every module within the limit, the constants a
                reader takes by name kept, the twin check passing; a short recipe stays one
                module and an emission that no longer splits removes the siblings an earlier
                one left;
  parquet       a parquet input read through Arrow's own filesystem (A40) by the shaper, the
                reproduction script it writes (which still reproduces every part) and the
                verifier.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SHAPE, VERIFY = HERE.parent / "shape.py", HERE.parent / "verify_shape.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shape = _load(SHAPE, "shaper_shape_under_test")
BUDGET = shape.STEP_BUDGET
SIDS = [f"S{i:02d}" for i in range(1, 13)]
PRELUDE = ("import math\n"
           "FITTED = {" + ", ".join(f"{s!r}: {{'fitted': {{'k': {i}, 'v': [{i}, {i + 1}]}}}}"
                                    for i, s in enumerate(SIDS)) + "}\n"
           "LOG = []\n")


# --------------------------------------------------------------------------- harness
def flat_source(steps: list) -> str:
    """The lines run as the module ran them before the split: one function, recipe order."""
    body = [("    " + line).rstrip() for s in steps for line in s["lines"]] or ["    pass"]
    return PRELUDE + "\n\ndef run(parts):\n" + "\n".join(body) + "\n"


def split_source(steps: list) -> str:
    return (PRELUDE + shape.transform_steps_source(steps)
            + "\n\ndef run(parts):\n    for step in _STEPS:\n        step(parts)\n")


def execute(source: str, parts: dict):
    ns = {}
    exec(compile(source, "<steps>", "exec"), ns)          # noqa: S102 — the test's own code
    try:
        ns["run"](parts)
        outcome = "ok"
    except Exception as ex:                                # noqa: BLE001
        outcome = type(ex).__name__
    return outcome, parts, ns["LOG"], ns["FITTED"]


def same_run(steps: list) -> tuple:
    """(flat result, split result) over the same starting parts."""
    start = {"n": 3, "acc": [], "keys": ["a", "b"]}
    return (execute(flat_source(steps), copy.deepcopy(start)),
            execute(split_source(steps), copy.deepcopy(start)))


def functions(source: str) -> dict:
    """{function name: its statement count, code-contract §5.1's way} of the `_step_*`s."""
    tree = ast.parse(source)
    return {n.name: sum(shape._stmt_cost(b) for b in n.body) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("_step_")}


def listed(source: str) -> list:
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "_STEPS")
    return [e.id for e in node.value.elts]


def step(sid: str, *lines: str, head: str | None = None) -> dict:
    return {"id": sid, "head": [head or f"# --- {sid}  test"], "lines": list(lines)}


# --------------------------------------------------------------------------- hand-written cases
def test_one_function_per_short_step_listed_in_recipe_order():
    steps = [step("S01", "parts['n'] = parts['n'] + 1"),
             step("S02", "_f = FITTED['S02']['fitted']", "parts['acc'].append(_f['k'])"),
             step("S03", "LOG.append('S03')")]
    src = shape.transform_steps_source(steps)
    assert listed(src) == ["_step_s01", "_step_s02", "_step_s03"]
    assert all(f"# --- {s}  test" in src for s in ("S01", "S02", "S03"))
    flat, split = same_run(steps)
    assert flat == split and flat[0] == "ok"


def test_a_name_bound_in_one_step_and_read_in_a_later_one_shares_its_function():
    steps = [step("S01", "_m = parts['n'] * 2"),
             step("S02", "LOG.append('between')"),
             step("S03", "parts['acc'].append(_m)")]
    src = shape.transform_steps_source(steps)
    assert listed(src) == ["_step_s01"]                    # S02 sits between binder and reader
    body = src[src.index("def _step_s01"):]
    assert body.index("_m = parts") < body.index("LOG.append") < body.index("append(_m)")
    flat, split = same_run(steps)
    assert flat == split and flat[1]["acc"] == [6]


def test_a_long_step_is_cut_within_the_budget_and_the_fitted_binder_read_again():
    lines = ["_f = FITTED['S01']['fitted']"]
    lines += [f"parts['acc'].append(_f['k'] + {i})" for i in range(3 * BUDGET)]
    steps = [step("S01", *lines)]
    src = shape.transform_steps_source(steps)
    sizes = functions(src)
    assert list(sizes) == ["_step_s01", "_step_s01_2", "_step_s01_3", "_step_s01_4"]
    assert max(sizes.values()) <= BUDGET
    assert src.count("_f = FITTED['S01']['fitted']") == len(sizes)   # once per function
    assert shape.unbound_step_names(PRELUDE + src) == []
    flat, split = same_run(steps)
    assert flat == split and len(flat[1]["acc"]) == 3 * BUDGET


def test_an_import_is_read_again_where_it_is_used():
    lines = ["import math as _mt"] + [f"parts['acc'].append(_mt.floor({i} / 2))"
                                       for i in range(2 * BUDGET)]
    src = shape.transform_steps_source([step("S01", *lines)])
    assert max(functions(src).values()) <= BUDGET and src.count("import math as _mt") == 3
    flat, split = same_run([step("S01", *lines)])
    assert flat == split


def test_a_binder_computed_from_another_line_is_not_replayed():
    # `_g = _f['v']` reads the replayable `_f`, so it is a line's value, not a constant's: every
    # reader of `_g` stays with it, and `_f` is read again before it.
    steps = [step("S01", "_f = FITTED['S01']['fitted']", "_g = _f['v']"),
             step("S02", *[f"parts['acc'].append(_g[{i % 2}])" for i in range(5)])]
    src = shape.transform_steps_source(steps)
    assert listed(src) == ["_step_s01"]
    flat, split = same_run(steps)
    assert flat == split


def test_a_rebinding_and_an_augmented_assignment_read_the_value_in_force():
    steps = [step("S01", "_f = FITTED['S01']['fitted']", "_a = _f['k']"),
             step("S02", "_f = FITTED['S02']['fitted']", "_a += _f['k']"),
             step("S03", "_f = dict(_f)", "parts['acc'].append((_a, _f['k']))")]
    flat, split = same_run(steps)
    assert flat == split and flat[1]["acc"] == [(1, 1)]


def test_comprehension_and_lambda_names_do_not_join_steps():
    steps = [step("S01", "_c = 5", "parts['acc'].append(_c)"),
             step("S02", "parts['acc'].append([_c for _c in range(3)])"),
             step("S03", "parts['acc'].append((lambda _c: _c + 1)(1))"),
             step("S04", "parts['acc'].append({k: len(k) for k in parts['keys']})")]
    src = shape.transform_steps_source(steps)
    assert listed(src) == ["_step_s01", "_step_s02", "_step_s03", "_step_s04"]
    flat, split = same_run(steps)
    assert flat == split


def test_statements_sharing_a_line_and_a_compound_statement_stay_whole():
    steps = [step("S01", "_e = list(FITTED['S01']['fitted']['v']); _e[0] = -1",
                  "for k in parts['keys']:",
                  "    _t = k.upper()",
                  "    parts['acc'].append((_t, _e[0]))",
                  "if parts['n'] > 2:",
                  "    _b = 'big'",
                  "else:",
                  "    _b = 'small'"),
             step("S02", "parts['acc'].append(_b)")]
    src = shape.transform_steps_source(steps)
    assert "_e = list(FITTED['S01']['fitted']['v']); _e[0] = -1" in src
    flat, split = same_run(steps)
    assert flat == split and flat[1]["acc"][-1] == "big"


def test_a_skipped_or_comment_only_step_is_a_comment_and_no_function():
    steps = [step("S01", "parts['n'] = 1"),
             {"id": "S02", "head": ["# --- S02  split  — skipped: a query is one part"],
              "lines": []},
             step("S03", "# geometry declared: kind=point (no column change)"),
             step("S04", "parts['n'] += 1")]
    src = shape.transform_steps_source(steps)
    assert listed(src) == ["_step_s01", "_step_s04"]
    assert "# --- S02  split  — skipped" in src and "# geometry declared: kind=point" in src
    assert src.index("# --- S02") < src.index("def _step_s04")
    flat, split = same_run(steps)
    assert flat == split


def test_an_empty_recipe_lists_no_step():
    src = shape.transform_steps_source([])
    assert "_STEPS = ()" in src
    assert execute(split_source([]), {"n": 0})[0] == "ok"


def test_ids_that_fold_to_one_name_get_distinct_functions():
    steps = [step("S-1", "parts['n'] += 1"), step("s 1", "parts['n'] *= 3")]
    src = shape.transform_steps_source(steps)
    assert listed(src) == ["_step_s_1", "_step_s_1_2"]
    flat, split = same_run(steps)
    assert flat == split and flat[1]["n"] == 12


def test_lines_that_do_not_parse_are_refused_naming_the_step():
    with pytest.raises(ValueError, match="step S07's emitted lines do not parse"):
        shape.transform_steps_source([step("S07", "for k in parts", "    pass")])


def test_the_guard_names_a_read_nobody_binds():
    src = PRELUDE + "\ndef _step_s01(parts):\n    parts['acc'].append(_lost + len([1]))\n"
    assert shape.unbound_step_names(src) == ["_step_s01: _lost"]
    nested = PRELUDE + "\ndef _step_s01(parts):\n    parts['acc'].append((lambda q: q + _w)(1))\n"
    assert shape.unbound_step_names(nested) == ["_step_s01: _w"]
    assert shape.unbound_step_names(split_source([step("S01", "_a = 1", "LOG.append(_a)")])) == []


def test_the_emitter_refuses_to_write_a_module_with_an_unbound_read(monkeypatch, tmp_path):
    real = shape.transform_steps_source
    monkeypatch.setattr(shape, "transform_steps_source",
                        lambda steps: real(steps) + "\ndef _step_zz(parts):\n    return _gone\n")
    recipe = {"input": {"path": str(tmp_path / "t.csv")},
              "steps": [{"id": "S01", "op": "drop_identity", "columns": ["a"],
                         "source": "user:test", "rationale": "r"}]}
    code = {"S01": ["for k in parts: parts[k] = parts[k].drop(columns=[c for c in ['a'] "
                    "if c in parts[k].columns])"]}
    target = tmp_path / "preprocess.py"
    with pytest.raises(ValueError, match="_step_zz: _gone"):
        shape.write_transform_module(recipe, {}, code, target, "fitted.json", lambda m: None)
    assert not target.exists()


# --------------------------------------------------------------------------- generated programs
NAMES = ("_a", "_b", "_m", "_t")


def _program(rng: random.Random) -> list:
    """A recipe of 1-8 steps of 1-25 statement items each, valid when run flat: a name is read
    only after some earlier item bound it; `_f` only ever holds a FITTED entry, `_mt` an import."""
    bound, steps = set(), []
    for sid in rng.sample(SIDS, rng.randint(1, 8)):
        lines = []
        for _ in range(rng.randint(1, 25)):
            have = sorted(bound - {"_f", "_mt"})
            kind = rng.choice(["fitted", "import", "bind", "read", "read", "fitted_read",
                               "aug", "loop", "comp", "lambda", "shared", "branch", "log",
                               "comment", "alias", "import_read", "rebind", "loop_over"])
            v, c = rng.choice(NAMES), rng.randint(0, 9)
            if kind == "fitted":
                lines.append(f"_f = FITTED[{rng.choice(SIDS)!r}]['fitted']")
                bound.add("_f")
            elif kind == "import":
                lines.append("import math as _mt")
                bound.add("_mt")
            elif kind == "import_read" and "_mt" in bound:
                lines.append(f"parts['acc'].append(_mt.floor({c} / 2))")
            elif kind == "bind":
                lines.append(f"{v} = parts['n'] + {c}")
                bound.add(v)
            elif kind == "read" and have:
                lines.append(f"parts['n'] = (parts['n'] * 3 + {rng.choice(have)}) % 1000003")
            elif kind == "fitted_read" and "_f" in bound:
                lines.append(f"parts['acc'].append(_f['k'] + {c})")
            elif kind == "alias" and "_f" in bound:
                lines.append(f"{v} = _f['v'][{c % 2}]")
                bound.add(v)
            elif kind == "aug" and have:
                lines.append(f"{rng.choice(have)} += {c}")
            elif kind == "rebind" and have:                # reads the name before it binds it
                w = rng.choice(have)
                lines.append(f"{w} = ({w} * 2 + {c}) % 1009")
            elif kind == "loop_over" and have:             # the iterable before the target
                w = rng.choice(have)
                lines += [f"for {w} in range({w} % 4, 5):", f"    parts['acc'].append({w})"]
            elif kind == "loop":
                w = rng.choice(have) if have else c
                lines += ["for k in parts['keys']:", f"    {v} = len(k) * {c}",
                          f"    parts['acc'].append(({v}, {w}))"]
                bound.add(v)
            elif kind == "comp":
                lines.append(f"parts['acc'].append([{v} for {v} in range({c})])")
            elif kind == "lambda":
                w = rng.choice(have) if have else c
                lines.append(f"parts['acc'].append((lambda {v}: {v} + {w})({c}))")
            elif kind == "shared":
                lines.append(f"{v} = {c}; parts['acc'].append({v} * 2)")
                bound.add(v)
            elif kind == "branch":
                lines += [f"if parts['n'] % 2:", f"    {v} = {c}", "else:", f"    {v} = -{c}"]
                bound.add(v)
            elif kind == "log":
                lines.append(f"LOG.append(({sid!r}, {c}))")
            elif kind == "comment":
                lines.append(f"# a note in {sid}")
        steps.append(step(sid, *lines))
    return steps


@pytest.mark.parametrize("seed", range(300))
def test_a_generated_program_runs_the_same_split_as_flat(seed):
    steps = _program(random.Random(seed))
    flat, split = same_run(steps)
    assert flat[0] == "ok", flat                           # the generator writes valid programs
    assert flat == split
    src = shape.transform_steps_source(steps)
    assert shape.unbound_step_names(PRELUDE + src) == []
    ast.parse(src)


def test_generated_programs_stay_within_the_budget_when_no_unit_exceeds_it():
    """Every function within STEP_BUDGET unless one unit (lines that must share a function) is
    itself larger — the only case the split cannot fix, and C11 then names it."""
    over = 0
    for seed in range(300):
        steps = _program(random.Random(seed))
        items = [it for s in steps for it in (shape._step_items(s["id"], s["lines"])[0]
                                              if s["lines"] else [])]
        biggest = max((shape._chunk_lines(items, [u])[1] for u in shape._units(items)),
                      default=0)
        sizes = functions(shape.transform_steps_source(steps))
        if biggest <= BUDGET:
            assert max(sizes.values(), default=0) <= BUDGET, (seed, sizes)
        else:
            over += 1
    assert over < 300                                       # the property is exercised


# --------------------------------------------------------------------------- end to end
def _rec(csv: Path, out: Path, cols: list) -> dict:
    def s(sid, op, columns, params, source="user:test"):
        return {"id": sid, "op": op, "columns": columns, "params": params, "source": source,
                "rationale": f"{op} for the test", "consequences":
                {"rows": "unchanged", "columns": "unchanged", "downstream": "none"}}
    nominal = [f"c{i}" for i in range(12)]
    return {"schema": "dataset-shaper/recipe@1", "input": {"path": str(csv)}, "seed": 1,
            "goal": "train-ready", "target": "y",
            "outputs": {"dir": str(out), "format": "parquet", "files": []},
            "steps": [s("S01", "retype", nominal, {"to": "nominal"}),
                      s("S02", "select_partition", ["y"], {"label": "y", "task": "regression"},
                        source="geometry:partitions/y"),
                      s("S03", "split", [], {"kind": "random", "test": 0.2, "valid": 0.2}),
                      s("S04", "impute", cols, {"strategy": "median", "indicator": True}),
                      s("S05", "clip", cols, {"lower": -3.0, "upper": 3.0}),
                      s("S06", "bin", cols[:16], {"kind": "quantile", "k": 4}),
                      s("S07", "encode", nominal, {"strategy": "ordinal"}),
                      s("S08", "scale", cols, {"kind": "standard", "columns": cols})]}


def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def test_a_long_recipe_through_the_shaper_and_the_twin_check(tmp_path):
    rng = np.random.default_rng(5)
    n, cols = 240, [f"x{i}" for i in range(40)]
    df = pd.DataFrame(rng.normal(size=(n, len(cols))).round(6), columns=cols)
    df.iloc[rng.integers(0, n, 30), 3] = np.nan
    for i in range(12):
        df[f"c{i}"] = rng.choice(list("abcde"), size=n)
    df["y"] = (df["x0"] * 2 + rng.normal(size=n)).round(6)
    csv, out = tmp_path / "wide.csv", tmp_path / "shaped"
    df.to_csv(csv, index=False)
    rp = tmp_path / "recipe.json"
    rp.write_text(json.dumps(_rec(csv, out, cols), indent=1), encoding="utf-8")
    rc, log = _run(SHAPE, "--recipe", rp, "--out-dir", out)
    assert rc == 0, log[-2000:]
    module = out / "preprocess.py"                     # beside the fitted file it reads
    rc, log = _run(SHAPE, "--transform-only", "--recipe", rp, "--fitted", out / "fitted.json",
                   "--emit", module)
    assert rc == 0, log[-2000:]
    siblings = sorted(out.glob("preprocess_steps*.py"))      # C13: past the module limit
    assert siblings and "preprocess_steps*.py beside preprocess.py: one module would pass" in log
    assert all(last_line(f) <= shape.MODULE_LINES for f in (module, *siblings)), \
        [last_line(f) for f in (module, *siblings)]
    sizes = {k: v for f in siblings for k, v in functions(f.read_text(encoding="utf-8")).items()}
    assert max(sizes.values()) <= BUDGET, sizes
    assert {"_step_s04_2", "_step_s08_2"} <= set(sizes), sorted(sizes)   # cut, not merged
    shared = " = ".join(shape.STEPS_SHARED) + " = None\n"
    assert all(shape.unbound_step_names(f.read_text(encoding="utf-8") + shared) == []
               for f in siblings)
    port = module.read_text(encoding="utf-8")
    assert not functions(port) and {"REQUIRED_COLUMNS", "HISTORY_COLUMNS", "LABEL"} <= {
        t.id for n in ast.parse(port).body if isinstance(n, ast.Assign) for t in n.targets
        if isinstance(t, ast.Name)}, "the port keeps the constants a reader takes by name"
    for part in ("train", "valid", "test"):
        rc, log = _run(VERIFY, "--transform-only", "--recipe", rp, "--out-dir", out,
                       "--module", module, "--part", part)
        assert rc == 0, log[-2000:]
        assert "match byte for byte" in log, log[-2000:]


def last_line(path: Path) -> int:
    """C13's measure of a module: line 1 to its last statement."""
    return max(n.end_lineno for n in ast.parse(path.read_text(encoding="utf-8")).body)


def test_a_short_recipe_stays_one_module_and_drops_a_stale_split(tmp_path):
    """Within the line limit the module is whole (no `<stem>_steps.py`), and an emission that
    no longer splits removes the sibling an earlier one left, which nothing would import."""
    rng = np.random.default_rng(3)
    cols = ["x0", "x1", "x2"]
    df = pd.DataFrame(rng.normal(size=(120, 3)).round(6), columns=cols)
    df["y"] = (df["x0"] + rng.normal(size=120)).round(6)
    for i in range(12):
        df[f"c{i}"] = rng.choice(list("ab"), size=120)
    csv, out = tmp_path / "narrow.csv", tmp_path / "shaped"
    df.to_csv(csv, index=False)
    rp = tmp_path / "recipe.json"
    rp.write_text(json.dumps(_rec(csv, out, cols), indent=1), encoding="utf-8")
    rc, log = _run(SHAPE, "--recipe", rp, "--out-dir", out)
    assert rc == 0, log[-2000:]
    module = out / "preprocess.py"
    stale = module.with_name("preprocess_steps_2.py")
    stale.write_text("raise ImportError('stale')\n", encoding="utf-8")
    rc, log = _run(SHAPE, "--transform-only", "--recipe", rp, "--fitted", out / "fitted.json",
                   "--emit", module)
    assert rc == 0, log[-2000:]
    assert not stale.exists() and last_line(module) <= shape.MODULE_LINES
    assert functions(module.read_text(encoding="utf-8")), "the steps stay in the module"
    rc, log = _run(VERIFY, "--transform-only", "--recipe", rp, "--out-dir", out,
                   "--module", module, "--part", "test")
    assert rc == 0 and "match byte for byte" in log, log[-2000:]


def test_a_parquet_input_is_read_through_arrows_filesystem(tmp_path):
    """PLAN-AMENDMENTS A40: the shaper, its reproduction script and the verifier read parquet
    with `filesystem=pyarrow.fs.LocalFileSystem()`, never through a Python file object (an
    abort at interpreter exit under load); the script so written still reproduces every part."""
    rng = np.random.default_rng(9)
    cols = ["x0", "x1"]
    df = pd.DataFrame(rng.normal(size=(120, 2)).round(6), columns=cols)
    df["y"] = (df["x0"] + rng.normal(size=120)).round(6)
    for i in range(12):
        df[f"c{i}"] = rng.choice(list("ab"), size=120)
    src, out = tmp_path / "wide.parquet", tmp_path / "shaped"
    df.to_parquet(src, index=False)
    rp = tmp_path / "recipe.json"
    rp.write_text(json.dumps(_rec(src, out, cols), indent=1), encoding="utf-8")
    rc, log = _run(SHAPE, "--recipe", rp, "--out-dir", out)
    assert rc == 0, log[-2000:]
    script = next(out.glob("reproduce_*.py")).read_text(encoding="utf-8")
    assert "filesystem=__import__('pyarrow.fs').fs.LocalFileSystem()" in script
    rc, log = _run(SHAPE, "--check", "--out-dir", out)
    assert rc == 0, log[-2000:]
    rc, log = _run(VERIFY, "--recipe", rp, "--out-dir", out)
    assert rc == 0, log[-2000:]
    for name in ("shape.py", "verify_shape.py"):           # every call site, by the AST
        calls = [n for n in ast.walk(ast.parse((HERE.parent / name).read_text(encoding="utf-8")))
                 if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "read_parquet"]
        assert calls and all(any(k.arg == "filesystem" for k in c.keywords) for c in calls), \
            (name, [c.lineno for c in calls])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
