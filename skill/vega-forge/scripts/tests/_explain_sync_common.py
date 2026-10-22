"""
_explain_sync_common.py — the plumbing `test_explain_catalog_sync.py` stands on.

Where the files are and the JSON it reads (the schema's draft 2020-12 validator included);
`Check`, one named group of checks; the catalog's
text fields (prose, formulas, code anchors); the lexicon keys a sample shape carries; the
backend-contract §2.2 placeholder list; the digit rule (the vega-show catalog's statement of
it, and H01's `digit_rule.py` when present); and the script runner. The code-anchor side is
`_explain_sync_anchors.py`.

The leading underscore keeps the unit tier's `test_*.py` discovery from collecting it: it holds
no test of its own.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import json
import re
import sys
import unicodedata
from pathlib import Path

from _explain_sync_facts import DRAFT_2020_12

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILL = SCRIPTS.parent
SKILLS = SKILL.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
EXPLAIN_JSON = BACKEND / "explain-catalog.json"
COVERAGE_JSON = BACKEND / "coverage.json"
BLOCKS_PY = BACKEND / "blocks.py"
CATALOG_JSON = SCRIPTS / "catalog.json"
BACKEND_MD = SKILL / "references" / "backend-contract.md"
SCHEMA_JSON = SKILL / "references" / "schemas" / "explain-catalog-1.schema.json"
SHOW_SCRIPTS = SKILLS / "vega-show" / "scripts"
SHOW_CATALOG_JSON = SHOW_SCRIPTS / "catalog.json"
DIGIT_RULE_PY = SHOW_SCRIPTS / "digit_rule.py"
VERBOSE = "-v" in sys.argv

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class Check:
    """One named group of checks: failures are printed as they happen and raised at the end,
    so the CLI reports every divergence and pytest sees one failing test per group."""

    def __init__(self, name):
        self.name = name
        self.errors = []

    def fail(self, msg):
        self.errors.append(msg)
        print(f"  FAIL  {msg}")

    def ok(self, msg):
        if VERBOSE:
            print(f"  ok    {msg}")

    def expect(self, cond, msg):
        if not cond:
            self.fail(msg)
        return bool(cond)

    def close(self):
        if self.errors:
            raise AssertionError(f"{self.name}: {len(self.errors)} divergence(s)\n  "
                                 + "\n  ".join(self.errors))


# --------------------------------------------------------------------------- the JSON read

@functools.lru_cache(maxsize=None)
def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def explain():
    return _json(str(EXPLAIN_JSON))


def coverage():
    return _json(str(COVERAGE_JSON))


def catalog():
    return _json(str(CATALOG_JSON))


def show_catalog():
    return _json(str(SHOW_CATALOG_JSON))


def schema():
    return _json(str(SCHEMA_JSON))


@functools.lru_cache(maxsize=None)
def blocks_module():
    import blocks  # the backend's block library (torch)
    return blocks


def contract_placeholders():
    """The bare placeholders backend-contract §2.2 lists, as `{name}` in backticks."""
    body = BACKEND_MD.read_text(encoding="utf-8")
    section = body.split("### 2.2 The explain catalog", 1)[-1].split("\n## ", 1)[0]
    return set(re.findall(r"`\{([a-z_]+)\}`", section))


def schema_validator(c):
    """The draft 2020-12 validator of the schema file, or None with the reason recorded on
    the group `c` (jsonschema missing, the file missing, or not a draft 2020-12 schema)."""
    try:
        import jsonschema
    except ImportError:
        c.fail("jsonschema >= 4 is this test's dependency (pip install jsonschema); the "
               "catalog cannot be checked against its schema without it")
        return None
    if not c.expect(SCHEMA_JSON.is_file(), f"{SCHEMA_JSON} does not exist"):
        return None
    c.expect(schema().get("$schema") == DRAFT_2020_12, f"$schema must be {DRAFT_2020_12}")
    try:
        jsonschema.Draft202012Validator.check_schema(schema())
    except jsonschema.exceptions.SchemaError as exc:
        c.fail(f"{SCHEMA_JSON.name} is not a valid draft 2020-12 schema: {exc.message}")
        return None
    return jsonschema.Draft202012Validator(schema())


# --------------------------------------------------------------------------- the catalog's text

def variants_of(entry):
    return (entry.get("variants") or {}).items()


@functools.lru_cache(maxsize=None)
def placeholder_re():
    return re.compile(show_catalog()["placeholder"]["pattern"])


def prose_fields(cat):
    """(where, text) for every prose field: the ones the digit rule governs."""
    for t, b in cat["blocks"].items():
        for key in ("label", "label_many", "step", "prose", "unit_how"):
            yield f"blocks.{t}.{key}", b[key]
        yield f"blocks.{t}.walk.title", b["walk"]["title"]
        yield f"blocks.{t}.walk.text", b["walk"]["text"]
        for v, var in variants_of(b):
            for key in ("prose", "unit_how"):
                yield f"blocks.{t}.variants.{v}.{key}", var[key]
            yield f"blocks.{t}.variants.{v}.walk.title", var["walk"]["title"]
            yield f"blocks.{t}.variants.{v}.walk.text", var["walk"]["text"]
    for a, entry in cat["activations"].items():
        yield f"activations.{a}.name", entry["name"]
        yield f"activations.{a}.bend", entry["bend"]
    for link, entry in cat["links"].items():
        for key in ("prose", "answer"):
            yield f"links.{link}.{key}", entry[key]
        yield f"links.{link}.walk.title", entry["walk"]["title"]
        yield f"links.{link}.walk.text", entry["walk"]["text"]
    for step in cat["walk_fixed"]:
        yield f"walk_fixed.{step['id']}.title", step["title"]
        yield f"walk_fixed.{step['id']}.text", step["text"]


def formula_fields(cat):
    """(where, text) for every formula: notation, exempt from the digit rule."""
    for t, b in cat["blocks"].items():
        yield f"blocks.{t}.formula", b["formula"]
        for v, var in variants_of(b):
            yield f"blocks.{t}.variants.{v}.formula", var["formula"]
    for a, entry in cat["activations"].items():
        yield f"activations.{a}.formula", entry["formula"]
    for link, entry in cat["links"].items():
        yield f"links.{link}.formula", entry["formula"]


def code_objects(cat):
    """(where, code) for every code anchor object."""
    for t, b in cat["blocks"].items():
        yield f"blocks.{t}", b["code"]
        for v, var in variants_of(b):
            yield f"blocks.{t}.variants.{v}", var["code"]
    for step in cat["walk_fixed"]:
        yield f"walk_fixed.{step['id']}", step["code"]


# --------------------------------------------------------------------------- lexicon

def lexicon_keys_for(flags):
    """The lexicon keys a text may use when the block can appear in the sample shapes its
    catalog flags allow: the keys every one of those shapes requires, plus the optional ones."""
    lex = show_catalog()["lexicon"]
    req = lex["required"]
    if flags.get("graph_level_only"):
        shapes = ["graph:graph"]
    elif flags.get("graph_only"):
        shapes = ["graph:node", "graph:graph"]
    elif flags.get("window_only"):
        shapes = ["window"]
    else:
        shapes = list(req)
    keys = set(lex["keys"])
    for shape in shapes:
        keys &= set(req[shape])
    return keys | set(lex.get("optional") or ())


def lexicon_keys_at(where):
    """The lexicon keys the field at `where` may use: a block's by its catalog flags, any
    other field's those every sample shape carries."""
    parts = where.split(".")
    if parts[0] == "blocks":
        return lexicon_keys_for(catalog()["blocks"].get(parts[1]) or {})
    return lexicon_keys_for({})


# --------------------------------------------------------------------------- the digit rule

def catalog_digit_rule():
    """The digit rule as the vega-show catalog states it (`digit_rule`), with an empty name
    list: placeholders deleted, a token with a decimal digit is refused (a bare number, or a
    name no lane's facts carry here), and so is any other decimal digit."""
    rule = show_catalog()["digit_rule"]
    token = re.compile(rule["token_pattern"])
    pre = placeholder_re()

    def refusals(text, where):
        rest = pre.sub(" ", text)
        out = [f"token {m.group(0)!r}" for m in token.finditer(rest)
               if any(ch.isdecimal() for ch in m.group(0))]
        outside = token.sub(" ", rest)
        out += [f"digit {ch!r}" for ch in outside if unicodedata.category(ch) == "Nd"]
        return out
    return refusals


def h01_digit_rule():
    """vega-show's digit_rule.py (WP H01) as a `(text, where) -> refusals` function, or None
    when the module is not there yet. A module that is there but offers neither `check_prose`
    nor a `DigitRule` this test can call is a failure: the two must be reconciled, not
    skipped."""
    if not DIGIT_RULE_PY.is_file():
        return None
    # imported by its own name, as the vega-show scripts import each other; the path entry is
    # taken back afterwards so vega-show's module names shadow nothing in a shared session
    added = str(SHOW_SCRIPTS) not in sys.path
    if added:
        sys.path.insert(0, str(SHOW_SCRIPTS))
    try:
        mod = importlib.import_module("digit_rule")
    finally:
        if added:
            sys.path.remove(str(SHOW_SCRIPTS))
    if Path(mod.__file__).resolve() != DIGIT_RULE_PY.resolve():
        raise AssertionError(f"`import digit_rule` found {mod.__file__}, not {DIGIT_RULE_PY}")
    return _adapt(mod)


def _required(fn, skip=0):
    """The positional parameters of `fn` without a default, after the first `skip`."""
    params = list(inspect.signature(fn).parameters.values())[skip:]
    return [p for p in params if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


def _adapt(mod):
    """H01's one-call prose check (D4 §12.6: `check_prose(text, known, lexicon_keys)`), given
    this catalog's placeholders as the known bare names and every lexicon key; else its
    `DigitRule` taken with no names."""
    empty = frozenset()
    if hasattr(mod, "check_prose"):
        fn = mod.check_prose
        params = inspect.signature(fn).parameters
        given = {"known": frozenset(explain()["placeholders"]),
                 "lexicon_keys": frozenset(show_catalog()["lexicon"]["keys"])}
        kwargs = {k: v for k, v in given.items() if k in params}
        fill = [empty] * len([p for p in _required(fn, 1) if p.name not in kwargs])

        def run(text, where):
            extra = {"where": where} if "where" in params else {}
            return list(fn(text, *fill, **kwargs, **extra) or ())
        return run
    if hasattr(mod, "DigitRule"):
        rule = mod.DigitRule(*([empty] * len(_required(mod.DigitRule))))
        for meth in ("refusals", "check"):
            method = getattr(rule, meth, None)
            if callable(method):
                takes_where = len(inspect.signature(method).parameters) >= 2
                return lambda text, where, m=method, w=takes_where: list(
                    (m(text, where) if w else m(text)) or ())
    raise AssertionError(f"{DIGIT_RULE_PY} offers neither check_prose(text, …) nor a DigitRule "
                         "with refusals/check(text); reconcile this test with it")


# --------------------------------------------------------------------------- runner

def run(tests, summary):
    """Run each group as the CLI does: 0 when all agree (and `summary()` is printed), 1 with
    the diverging groups named. A catalog too damaged to compare is a divergence too."""
    failed = []
    for test in tests:
        try:
            test()
            if VERBOSE:
                print(f"  pass  {test.__name__}")
        except AssertionError:
            failed.append(test.__name__)
        except Exception as exc:
            print(f"  FAIL  {test.__name__} stopped on {type(exc).__name__}: {exc}")
            failed.append(test.__name__)
    if failed:
        print(f"\n{len(failed)} group(s) diverged: {', '.join(failed)}")
        return 1
    print(summary())
    return 0
