#!/usr/bin/env python3
"""test_prereqs.py: the prerequisites the chain needs from outside itself (work package X1).

    python3 test_prereqs.py [-v]          (or: python3 -m pytest test_prereqs.py)

1. **The blueprint patch** (`patches/blueprint/SKILL.md`, build plan R-G2). The emitted code
   inherits the module paradigms and the port signatures from the blueprint, so port-owning
   modules must be `OOP` and port methods must take at most 5 parameters (C12, C14).
   `/blueprint --extends` must accept the four serving targets. The frontmatter must still pass
   the Agent Skills upload rules, or the patched copy never replaces the installed skill.
2. **The domain-forge patch** (`patches/domain-forge/`). Against the installed
   `validate_model.py` it either applies with no offset and no fuzz, or is already applied
   locally. It changes only the cleanup of the three browser-profile temp dirs, and the result
   compiles. A changed upstream file fails with the action it calls for.
3. **The known-flake wrapper** (`known_flake.py`). It retries a seeded `Errno 39` profile race
   exactly once, never retries any other failure, and refuses a malformed signatures file
   without running the command. The forge, train and eval acceptance scripts call
   `validate_model.py` only through it.

Exit 0 when every check passes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KNOWN_FLAKE = HERE / "known_flake.py"
REAL_FLAKES = HERE / "lanekit" / "flakes.json"
BLUEPRINT_PATCH = ROOT / "patches" / "blueprint" / "SKILL.md"
BLUEPRINT_INSTALLED = ROOT / "blueprint" / "SKILL.md"
SKILL_VALIDATOR = ROOT / "skill-creator" / "scripts" / "quick_validate.py"
DF_PATCH_DIR = ROOT / "patches" / "domain-forge"
DF_PATCH = DF_PATCH_DIR / "validate_model-tempdir.patch"
DF_VALIDATOR = ROOT / "domain-forge" / "scripts" / "validate_model.py"
ACCEPTANCE = {stage: ROOT / f"vega-{stage}" / "scripts" / "tests" / "acceptance.sh"
              for stage in ("forge", "train", "eval")}

EXTENDS_TARGETS = ("serving.api", "serving.inspection", "serving.showcase", "serving.demo")
# The Agent Skills frontmatter rules, as skill-creator's quick_validate.py applies them.
SKILL_KEYS = {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}
MAX_DESCRIPTION = 1024

# The installed validate_model.py before and after the patch (patches/domain-forge/README.md).
DF_PRE_SHA256 = "86e6db2b3b606685cfba52d1e30338607a11943e58de44d6944641002c2b02d3"
DF_POST_SHA256 = "6e46edbef1ced45f085faee50b18ce6af6469a99822ee6260ea32d6f947cb18d"
# A context line of the patch's first hunk: changing it plays an upstream edit.
DF_CONTEXT = "os.path.join(td, 'instrumented.html')"

# The df-profile-race signature exactly as design D6 §6.12 words it: the command is
# validate_model.py, the output is the Errno 39 on 'Default' with tempfile in the traceback.
SEEDED_SIGNATURE = {
    "id": "df-profile-race",
    "command_regex": r"validate_model\.py",
    "output_regex": r"(?s)tempfile.*OSError: \[Errno 39\] Directory not empty: 'Default'",
    "owner": "domain-forge",
    "upstream": "patches/domain-forge/README.md",
}

# The traceback the acceptance flow printed (eval step 6 on c849f8b), paths made generic.
ERRNO39_TRACEBACK = textwrap.dedent("""\
    Traceback (most recent call last):
      File "/skills/domain-forge/scripts/validate_model.py", line 760, in check_projector_round_trip
        with tempfile.TemporaryDirectory() as td:
      File "/usr/lib/python3.12/tempfile.py", line 1085, in __exit__
        self.cleanup()
      File "/usr/lib/python3.12/tempfile.py", line 1089, in cleanup
        self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
      File "/usr/lib/python3.12/shutil.py", line 695, in _rmtree_safe_fd
        os.rmdir(entry.name, dir_fd=topfd)
    OSError: [Errno 39] Directory not empty: 'Default'
    """)
# The same errno raised by something that is not the temp-dir cleanup: not the known flake.
ERRNO39_ELSEWHERE = ("  File \"/skills/domain-forge/scripts/validate_model.py\", line 1452\n"
                     "    os.rmdir(out_dir)\nOSError: [Errno 39] Directory not empty: 'Default'\n")
OTHER_FAILURE = "AssertionError: invariant 12: projector turtle threw\n"

# A stand-in validator: attempt n plays the n-th mode of $FAKE_PLAN (the last one repeats) and
# counts itself in $FAKE_COUNTER, so a test can tell how many attempts the wrapper made.
FAKE_SCRIPT = textwrap.dedent("""\
    import json, os, sys
    counter = os.environ["FAKE_COUNTER"]
    n = (int(open(counter).read()) if os.path.exists(counter) else 0) + 1
    open(counter, "w").write(str(n))
    plan = os.environ["FAKE_PLAN"].split(",")
    mode = plan[min(n, len(plan)) - 1]
    print(f"OK: attempt {n} checked the canonical Turtle")
    sys.stdout.flush()
    if mode == "pass":
        print("OK: all invariants hold")
        sys.exit(0)
    sys.stderr.write(json.loads(os.environ["FAKE_TEXTS"])[mode])
    sys.exit(1)
    """)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body of the `## heading` section, up to the next `## ` heading."""
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    return text[start:nxt if nxt >= 0 else len(text)]


def _frontmatter(text: str):
    """The YAML frontmatter of a SKILL.md: PyYAML when installed, else a reader for the flat
    `key: value` / folded `key: >-` form the skill uses. None when there is no frontmatter."""
    match = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not match:
        return None
    try:
        import yaml
        return yaml.safe_load(match.group(1))
    except ImportError:
        doc, key = {}, None
        for line in match.group(1).splitlines():
            if line.startswith((" ", "\t")) and key:
                doc[key] = f"{doc[key]} {line.strip()}".strip()
            else:
                key, _, value = line.partition(":")
                doc[key] = "" if value.strip() in (">-", ">", "|", "|-") else value.strip()
        return doc


class BlueprintPatchTest(unittest.TestCase):
    """R-G2: OOP port modules, at most 5 parameters per port method, four --extends targets."""

    @classmethod
    def setUpClass(cls):
        cls.text = _read(BLUEPRINT_PATCH)

    def test_frontmatter_passes_the_skill_upload_rules(self):
        meta = _frontmatter(self.text)
        self.assertIsInstance(meta, dict, "the patch has no YAML frontmatter")
        self.assertLessEqual(set(meta), SKILL_KEYS, "frontmatter keys outside the allowed set")
        self.assertEqual(meta.get("name"), "blueprint", "a replacement copy of blueprint/SKILL.md")
        description = meta.get("description")
        self.assertIsInstance(description, str)
        self.assertLessEqual(len(description.strip()), MAX_DESCRIPTION,
                             "the description is over the Agent Skills limit: the upload "
                             "validator refuses the copy, and the patch never reaches /blueprint")
        self.assertNotRegex(description, "[<>]", "the description may not contain < or >")

    def test_skill_creator_validates_the_patch(self):
        if not SKILL_VALIDATOR.is_file():
            self.skipTest(f"no skill-creator at {SKILL_VALIDATOR.parents[1]}")
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(BLUEPRINT_PATCH, Path(td) / "SKILL.md")
            res = subprocess.run([sys.executable, str(SKILL_VALIDATOR), td],
                                 capture_output=True, text=True)
        if "No module named 'yaml'" in res.stderr:
            self.skipTest("quick_validate.py needs PyYAML; the frontmatter test above stands in")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

    def test_port_owning_modules_are_oop(self):
        self.assertIn('"paradigm": "OOP|FP — OOP for every module that owns a port"', self.text)
        self.assertIn("**A module that owns a port is `OOP`**", self.text)
        self.assertIn("OOP for stateful identity and for every module that owns a port",
                      self.text)
        self.assertIn("C14", self.text)

    def test_port_signatures_take_at_most_five_parameters(self):
        self.assertRegex(self.text, r'"ports": \[ \{[^\]]*"signature": \["def <method>\(self, '
                                    r'<at most 5 parameters>\)')
        for needle in ("**A port method takes at most 5 parameters**", "**parameter object**",
                       "at most 5 parameters beyond `self` / `cls`", "no `*args` / `**kwargs`",
                       "C12", "EJ-51"):
            self.assertIn(needle, self.text)

    def test_the_four_extends_targets(self):
        trigger = ("`/blueprint --extends " + " | ".join(EXTENDS_TARGETS) + " --pipeline "
                   "<run>/pipeline.json --entries '<json>' --because \"<pressure>\"`")
        self.assertIn(trigger, self.text, "the trigger list names the four targets and --because")
        body = _section(self.text, "## Extending a pipeline blueprint (`--extends`)")
        rows = re.findall(r"^\| `(serving\.[a-z]+)` \|", body, flags=re.M)
        self.assertEqual(tuple(rows), EXTENDS_TARGETS,
                         "the targets table lists exactly the four targets, in order")
        self.assertIn("exactly these; any other is refused", body)
        self.assertIn('--because "<pressure>" [--by <caller>]', body, "--because is required")
        self.assertIn("`--because` is required", body)

    def test_extends_entry_shape(self):
        body = _section(self.text, "## Extending a pipeline blueprint (`--extends`)")
        self.assertIn('{ "method": "GET|POST|PUT|PATCH|DELETE", "path": "/…", '
                      '"capability": "core|code|inspection|showcase", "calls": "<port>" }', body)
        for rule in ("`calls` that resolves to no port is **refused**",
                     "`serving.api.endpoints[]`", "`MANIFEST.serve.endpoints`",
                     "the base run is never edited", "`base_digest`",
                     "never a change of `serving.mode`", "**The fallback rule.**",
                     '"fallback": { "capability": "core", "target": "<its target>"',
                     "neither is refused"):
            self.assertIn(rule, body)
        for block in ('"inspection": { "enabled": true, "weights": "full|summary"',
                      '"showcase": { "slot": "showcase/"',
                      '"demo": { "adapter": "online", "forced_by": "--serve"',
                      '"fallback": { "capability": "core", "target": …, "bet": … }'):
            self.assertIn(block, self.text, "pipeline.json documents the block --extends writes")

    def test_stop_criteria_cover_extends(self):
        self.assertIn("For `--extends`: done when", self.text)

    def test_an_artifact_may_be_declared_external(self):
        """code-contract §3: an artifact held outside the package is declared with its owner
        and where it lives, so /vega-forge records it instead of finding it missing."""
        self.assertRegex(self.text, r'"artifacts": \[ \{ "name": "<§2 value object or entity>", '
                                    r'"kind": "value\|entity", "identity": …, "external": '
                                    r'\{ "owner": "[^"]*/vega-train[^"]*", "where": "<where it lives>" \}')
        for needle in ("**Every artifact has a home**", "`/vega-train`'s campaign directory",
                       "never `/vega-forge`", "code-contract.md` §3", "refuses, by name"):
            self.assertIn(needle, self.text)

    def test_a_held_artifact_takes_the_packages_name(self):
        """T5 round 1: a blueprint that named its held artifacts in the data's words
        (ParcelFrame, FeatureMatrix) was refused by /vega-forge and had to be re-run; the patch
        points the blueprint at the names the package can hold them under."""
        for needle in ("**A held artifact takes the package's name for it**",
                       "`python3 <vega-forge>/scripts/held_artifacts.py --shape <row|window|graph>`",
                       "never a second artifact", "declared external with the stage that holds it"):
            self.assertIn(needle, self.text)
        self.assertTrue((ROOT / "vega-forge" / "scripts" / "held_artifacts.py").is_file(),
                        "the patch names a script the package ships")
        # the parameter objects the patch gives as examples are the package's own (the round 1
        # blueprint drew FitRequest/ScoreRequest/Budget, which no package holds)
        for gone in ("FitRequest", "ScoreRequest", "`Budget`"):
            self.assertNotIn(gone, self.text)
        for held in ("`RunContext`", "`BatchPlan`", "`MetricOptions`"):
            self.assertIn(held, self.text)

    def test_a_signature_names_only_what_ports_py_imports(self):
        """T5 round 2: the first blueprint's §3 annotated with `DataFrame` and `Tensor` (and
        with held names the emitter did not import), the emitter refused the package and the
        blueprint had to be re-run; the patch states the vocabulary a signature may use."""
        for needle in ("**A signature annotates only with names the emitted `ports.py` can "
                       "import**", "a pandas `DataFrame`, a torch `Tensor`, a numpy array — "
                       "stays unannotated", "`held_artifacts.py --shape <…>` lists",
                       "first parameter is `self`"):
            self.assertIn(needle, self.text)

    def test_a_port_draws_its_adapters_methods(self):
        """T5 round 2: §3 drew ports narrower than their adapters (iter_batches without
        batch_size and columns, an Evaluator without measure), struck a value object the package
        holds, and claimed a refusal at the features boundary the package does not make; the
        patch points §3 at port_surface.py and states what §2 and the boundaries may say."""
        for needle in ("**A port's methods are its adapter's**",
                       "`python3 <vega-forge>/scripts/port_surface.py --shape <row|window|graph> --task <kind>`",
                       "**A port a later stage fills says so**", "`\"filled_by\": \"/vega-pack\"`",
                       "**The package is the library the processes wire, not the processes**",
                       "every port method of the role, each with exactly the parameters printed",
                       "which adapter class implements it",
                       "`ModelBuilder`, `TrainedNetwork`, `TrainingData`, `Callbacks`, "
                       "`ScoredModel`, `ArtifactExporter`",
                       "a read-only view of the staged network",
                       "the evaluation's baselines are scored models",
                       "so §2 never strikes one from the world model",
                       "a boundary's `contract` states what the package enforces at it",
                       "does not claim that every adapter holds state",
                       "the label set is the spec's `task.classes`, in that order",
                       "ends with the module arrows the backend's code needs"):
            self.assertIn(needle, self.text)
        self.assertNotIn("the adapter guards a state;", self.text)
        self.assertTrue((ROOT / "vega-forge" / "scripts" / "port_surface.py").is_file(),
                        "the patch names a script the package ships")

    def test_keeps_every_section_of_the_installed_skill(self):
        if not BLUEPRINT_INSTALLED.is_file():
            self.skipTest(f"no installed blueprint skill at {BLUEPRINT_INSTALLED}")
        headings = re.findall(r"^#{2,4} .+$", _read(BLUEPRINT_INSTALLED), flags=re.M)
        missing = [h for h in headings if h not in self.text]
        self.assertEqual(missing, [], "the replacement copy dropped sections of the original")


def _browser_tempdirs(source: str):
    """Every `with tempfile.TemporaryDirectory(...)` whose body starts a headless browser, with
    whether it passes ignore_cleanup_errors=True."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "TemporaryDirectory"):
                continue
            segment = ast.get_source_segment(source, node) or ""
            if "--user-data-dir" in segment or "_l16_run(" in segment:
                ignores = any(k.arg == "ignore_cleanup_errors"
                              and isinstance(k.value, ast.Constant) and k.value.value is True
                              for k in call.keywords)
                found.append((node.lineno, ignores))
    return found


def _df_tree(validator: Path, td: str) -> Path:
    """A scratch domain-forge tree holding a copy of `validator` at scripts/validate_model.py."""
    tree = Path(td) / "domain-forge"
    (tree / "scripts").mkdir(parents=True)
    shutil.copy2(validator, tree / "scripts" / "validate_model.py")
    return tree


def _run_patch(tree: Path, reverse: bool = False, dry_run: bool = False):
    """The domain-forge patch run on `tree`, never with fuzz, with the tools' output in English.
    None when neither GNU patch nor git is installed."""
    env = dict(os.environ, LC_ALL="C", LANGUAGE="", GIT_CEILING_DIRECTORIES=str(tree.parent))
    if shutil.which("patch"):
        cmd = ["patch", "-p1", "--forward", "--batch", "--fuzz=0", "-d", str(tree),
               "-i", str(DF_PATCH)] + ["-R"] * reverse + ["--dry-run"] * dry_run
    elif shutil.which("git"):
        cmd = (["git", "-C", str(tree), "apply", "-v", "-C3", "-p1"] + ["-R"] * reverse
               + ["--check"] * dry_run + [str(DF_PATCH)])
    else:
        return None
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _df_state(validator: Path) -> str:
    """`unpatched` (the pre-image, or the patch applies forward), `patched` (the post-image, or
    the patch applies in reverse: applied locally) or `changed` (neither: upstream moved)."""
    digest = hashlib.sha256(validator.read_bytes()).hexdigest()
    if digest in (DF_PRE_SHA256, DF_POST_SHA256):
        return "unpatched" if digest == DF_PRE_SHA256 else "patched"
    with tempfile.TemporaryDirectory() as td:
        tree = _df_tree(validator, td)
        for state, reverse in (("unpatched", False), ("patched", True)):
            res = _run_patch(tree, reverse=reverse, dry_run=True)
            if res is not None and res.returncode == 0:
                return state
    return "changed"


class DomainForgePatchTest(unittest.TestCase):
    """The TemporaryDirectory(ignore_cleanup_errors=True) patch, with its reason."""

    def test_readme_states_the_reason(self):
        readme = _read(DF_PATCH_DIR / "README.md")
        for needle in ("OSError: [Errno 39] Directory not empty: 'Default'",
                       "ignore_cleanup_errors=True", "df-profile-race", "G17", DF_PRE_SHA256,
                       DF_POST_SHA256, "validate_model-tempdir.patch", "known_flake.py"):
            self.assertIn(needle, readme)

    def test_patch_only_adds_ignore_cleanup_errors(self):
        diff = _read(DF_PATCH)
        self.assertIn("--- a/scripts/validate_model.py\n+++ b/scripts/validate_model.py\n", diff)
        lines = diff.splitlines()
        removed = [ln[1:] for ln in lines if ln.startswith("-") and not ln.startswith("---")]
        added = [ln[1:] for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
        self.assertEqual((len(removed), len(added)), (3, 3),
                         "three temp dirs host a browser profile")
        for old, new in zip(removed, added):
            self.assertIn("tempfile.TemporaryDirectory()", old)
            self.assertEqual(new, old.replace("tempfile.TemporaryDirectory()",
                                              "tempfile.TemporaryDirectory("
                                              "ignore_cleanup_errors=True)"))

    def _check_validator(self, validator: Path) -> str:
        """Assert what the patch's state against `validator` calls for; return the state."""
        before = _browser_tempdirs(_read(validator))
        self.assertTrue(before, "the validator starts no browser in a temp dir")
        strict = [line for line, ignores in before if not ignores]
        state = _df_state(validator)
        if state == "changed" and not (shutil.which("patch") or shutil.which("git")):
            self.skipTest("neither patch nor git is installed to compare an unknown validator")
        if state == "patched":
            self.assertEqual(strict, [], "the patch applies in reverse, yet a browser-profile "
                                         "temp dir is still cleaned up strictly")
        elif state == "changed" and not strict:
            self.fail("the validator changed upstream and already ignores the cleanup errors: "
                      "the fix has landed; delete patches/domain-forge/ and the df-profile-race "
                      "entry of lanekit/flakes.json (the known-flake list only shrinks)")
        elif state == "changed":
            self.fail("the validator changed upstream and the patch applies neither forward nor "
                      f"in reverse: regenerate the patch against it (strict at lines {strict})")
        else:
            with tempfile.TemporaryDirectory() as td:
                tree = _df_tree(validator, td)
                res = _run_patch(tree)
                if res is None:
                    self.skipTest("neither patch nor git is installed")
                out = res.stdout + res.stderr
                self.assertEqual(res.returncode, 0, f"the patch does not apply cleanly: {out}")
                for word in ("offset", "fuzz"):
                    self.assertNotIn(word, out.lower(), f"the patch applied with {word}: "
                                                        "regenerate it against the installed file")
                patched = tree / "scripts" / "validate_model.py"
                py_compile.compile(str(patched), cfile=str(Path(td) / "v.pyc"), doraise=True)
                after = _browser_tempdirs(_read(patched))
            self.assertEqual(len(after), len(before))
            self.assertEqual([line for line, ignores in after if not ignores], [],
                             "a browser-profile temp dir is still cleaned up strictly")
        return state

    def test_patch_fits_the_installed_validator(self):
        if not DF_VALIDATOR.is_file():
            self.skipTest(f"no installed domain-forge at {DF_VALIDATOR.parents[1]}")
        if self._check_validator(DF_VALIDATOR) == "patched":
            sys.stderr.write(f"\nNOTE: {DF_VALIDATOR} already carries "
                             f"{DF_PATCH.name}: the patch is applied locally\n")

    def test_the_three_states_are_told_apart(self):
        """The README's Apply step keeps the test green; only a moved upstream fails."""
        if not DF_VALIDATOR.is_file():
            self.skipTest(f"no installed domain-forge at {DF_VALIDATOR.parents[1]}")
        if not (shutil.which("patch") or shutil.which("git")):
            self.skipTest("neither patch nor git is installed")
        with tempfile.TemporaryDirectory() as td:
            tree = _df_tree(DF_VALIDATOR, td)
            copy = tree / "scripts" / "validate_model.py"
            digest = hashlib.sha256(copy.read_bytes()).hexdigest()
            if digest not in (DF_PRE_SHA256, DF_POST_SHA256):
                self.skipTest("the installed validator is neither the recorded pre-image nor "
                              "the post-image; the test above says what that calls for")
            if digest == DF_POST_SHA256:
                self.assertEqual(_run_patch(tree, reverse=True).returncode, 0)
            pre = _read(copy)
            self.assertEqual(self._check_validator(copy), "unpatched")
            self.assertEqual(_run_patch(tree).returncode, 0)   # the README's Apply step
            post = _read(copy)
            self.assertEqual(self._check_validator(copy), "patched")
            self.assertEqual(pre.count(DF_CONTEXT), 1)
            for base, action in ((pre, "regenerate the patch"),
                                 (post, "delete patches/domain-forge/")):
                copy.write_text(base.replace(DF_CONTEXT, "os.path.join(td, 'probe.html')"),
                                encoding="utf-8")
                self.assertEqual(_df_state(copy), "changed")
                with self.assertRaises(AssertionError) as caught:
                    self._check_validator(copy)
                self.assertIn(action, str(caught.exception))


class KnownFlakeTest(unittest.TestCase):
    """known_flake.py retries the seeded Errno 39 once and never retries anything else."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="known-flake-"))
        self.flakes = self._flakes_file({"schema": "vega/known-flakes@1",
                                         "flakes": [SEEDED_SIGNATURE]})
        self.record = self.tmp / "flakes.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _flakes_file(self, doc, name="flakes.json") -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def _run(self, plan, script="validate_model.py", flakes=None):
        fake = self.tmp / script
        fake.write_text(FAKE_SCRIPT, encoding="utf-8")
        counter = self.tmp / f"{script}.count"
        counter.unlink(missing_ok=True)
        env = dict(os.environ, FAKE_COUNTER=str(counter), FAKE_PLAN=plan,
                   FAKE_TEXTS=json.dumps({"flake": ERRNO39_TRACEBACK,
                                          "elsewhere": ERRNO39_ELSEWHERE,
                                          "other": OTHER_FAILURE}))
        env.pop("VEGA_FLAKE_RECORD", None)
        env.pop("VEGA_KNOWN_FLAKES", None)
        res = subprocess.run([sys.executable, str(KNOWN_FLAKE), "run",
                              "--flakes", str(flakes or self.flakes), "--stage", "eval",
                              "--record", str(self.record), "--",
                              sys.executable, str(fake), "model.html"],
                             capture_output=True, text=True, env=env, timeout=60)
        attempts = int(counter.read_text()) if counter.exists() else 0
        return res, attempts

    def _records(self):
        if not self.record.exists():
            return []
        return [json.loads(ln) for ln in self.record.read_text().splitlines() if ln.strip()]

    def test_seeded_errno39_is_retried_once(self):
        res, attempts = self._run("flake,pass")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(attempts, 2)
        self.assertIn("WARN: known flake df-profile-race", res.stderr)
        self.assertIn("OK: known flake df-profile-race: the second attempt passed", res.stderr)
        self.assertEqual(self._records(), [{"id": "df-profile-race", "stage": "eval",
                                            "command": "validate_model.py …", "retried": True,
                                            "second_attempt": "pass"}])

    def test_second_failure_is_final(self):
        res, attempts = self._run("flake,flake,pass")
        self.assertEqual(res.returncode, 1)
        self.assertEqual(attempts, 2, "a flake is retried once, never twice")
        self.assertIn("ERROR: known flake df-profile-race: the second attempt failed too",
                      res.stderr)
        self.assertEqual([r["second_attempt"] for r in self._records()], ["fail"])

    def test_nothing_else_is_retried(self):
        broad = self._flakes_file({"schema": "vega/known-flakes@1",
                                   "flakes": [dict(SEEDED_SIGNATURE, command_regex=r".")]},
                                  name="broad.json")
        for plan, script, flakes in (
                ("other,pass", "validate_model.py", None),      # another failure
                ("elsewhere,pass", "validate_model.py", None),  # Errno 39 outside the cleanup
                ("flake,pass", "apply_layer.py", None),         # the output, another command
                ("flake,pass", "run_negatives.py", broad),      # negatives, under any signature
                ("flake,pass", "test_roundtrip.py", broad)):    # round trip, under any signature
            with self.subTest(plan=plan, script=script):
                res, attempts = self._run(plan, script=script, flakes=flakes)
                self.assertEqual((res.returncode, attempts), (1, 1))
                self.assertNotIn("known flake", res.stderr)
        self.assertEqual(self._records(), [])

    def test_a_pass_runs_once_and_streams_through(self):
        res, attempts = self._run("pass")
        self.assertEqual((res.returncode, attempts), (0, 1))
        self.assertEqual(res.stdout, "OK: attempt 1 checked the canonical Turtle\n"
                                     "OK: all invariants hold\n")
        self.assertEqual(res.stderr, "")

    def test_streams_stay_apart(self):
        res, _ = self._run("other")
        self.assertIn("OK: attempt 1 checked the canonical Turtle", res.stdout)
        self.assertNotIn("AssertionError", res.stdout)
        self.assertIn("AssertionError: invariant 12", res.stderr)

    def test_a_malformed_signatures_file_is_refused_before_running(self):
        for bad in ({"schema": "vega/known-flakes@1",
                     "flakes": [dict(SEEDED_SIGNATURE, upstream="")]},
                    {"schema": "vega/known-flakes@1",
                     "flakes": [dict(SEEDED_SIGNATURE, output_regex="([")]},
                    {"schema": "vega/known-flakes@1",
                     "flakes": [SEEDED_SIGNATURE, SEEDED_SIGNATURE]},
                    {"schema": "vega/known-flakes@2", "flakes": []}):
            res, attempts = self._run("pass", flakes=self._flakes_file(bad, name="bad.json"))
            self.assertEqual((res.returncode, attempts), (2, 0), bad)
            self.assertIn("ERROR: known flakes:", res.stderr)

    def test_a_missing_signatures_file_retries_nothing(self):
        res, attempts = self._run("flake,pass", flakes=self.tmp / "absent.json")
        self.assertEqual((res.returncode, attempts), (1, 1))
        self.assertIn("WARN: no known-flakes file at", res.stderr)

    def test_check_names_the_signatures(self):
        res = subprocess.run([sys.executable, str(KNOWN_FLAKE), "check", "--flakes",
                              str(self.flakes)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("df-profile-race (owner domain-forge)", res.stdout)

    def test_the_lanekit_signatures_catch_the_seeded_race(self):
        if not REAL_FLAKES.is_file():
            self.skipTest(f"{REAL_FLAKES.relative_to(ROOT)} (work package T01) is not in "
                          "this tree yet; the seeded signature stands in for it above")
        res, attempts = self._run("flake,pass", flakes=REAL_FLAKES)
        self.assertEqual((res.returncode, attempts), (0, 2), res.stderr)
        res, attempts = self._run("other,pass", flakes=REAL_FLAKES)
        self.assertEqual((res.returncode, attempts), (1, 1))


class AcceptanceWiringTest(unittest.TestCase):
    """The forge, train and eval acceptance scripts call validate_model.py through the wrapper."""

    @staticmethod
    def _resolved_kf(script: Path) -> str:
        """The path `$KF` holds in the script, computed by bash from the script's own header."""
        lines = _read(script).splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("TESTS="))
        end = next(i for i, ln in enumerate(lines) if ln.startswith("KF="))
        header = "\n".join(lines[start:end + 1]).replace('"${BASH_SOURCE[0]}"', f'"{script}"')
        res = subprocess.run(["bash", "-c", header + '\nprintf %s "$KF"'],
                             capture_output=True, text=True)
        return res.stdout

    def test_validate_model_runs_through_known_flake(self):
        for stage, script in ACCEPTANCE.items():
            text = _read(script)
            self.assertEqual(Path(self._resolved_kf(script)).resolve(), KNOWN_FLAKE,
                             f"{stage}: $KF does not name this wrapper")
            calls = [ln for ln in text.splitlines()
                     if "validate_model.py" in ln and not ln.lstrip().startswith("#")]
            self.assertTrue(calls, f"{stage}: the acceptance no longer validates its layer")
            for ln in calls:
                self.assertRegex(ln, rf'^python3 "\$KF" run --stage {stage} -- python3 '
                                     r'"\$DF/validate_model\.py" ',
                                 f"{stage}: validate_model.py called outside the wrapper")

    def test_scripts_parse(self):
        for stage, script in ACCEPTANCE.items():
            res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"{stage}: {res.stderr}")
            self.assertTrue(os.access(script, os.X_OK), f"{stage}: acceptance.sh lost its x bit")


if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    result = unittest.main(argv=[sys.argv[0]], exit=False, verbosity=verbosity).result
    sys.exit(0 if result.wasSuccessful() else 1)
