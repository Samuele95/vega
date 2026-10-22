#!/usr/bin/env python3
"""
test_static_tier.py — the static tier's gates catch what they exist to catch (T04's closing tests).

    python3 -m pytest -q vega/scripts/tests/test_static_tier.py

- lint_templates.py flags the known CIC literals of c849f8b (D6 §15.2, D3 G7) and a seeded
  BENIGN (VS20), reads words in their dataset sense, pack sources' non-docstring literals and
  the lane VOCAB (exit 2 on a broken generator); an allowance needs a reason and its line;
- style_lock.py holds on 0d06aea's page and on the page template, fails on a one-token edit
  (PN10), a rule edit in an area sheet and a dropped breakpoint (a new one is an addition);
  style-lock.json is 0d06aea's extract;
- check_unattended.py flags a seeded unanswered FORK (RG2), a fork phrased like a reference and
  an empty chain-map cell, and passes the legend, a reference and an answered fork;
- verifier_gate.py passes a current SHIP record and fails a stale prompt digest, a non-SHIP
  verdict and a run that did not audit an Input file; --release accepts a T5 run that says
  pass-with-fixed-defects only with every product check passing and each defect's fix, with its
  test, on the released head (and says so in a NOTE), and fails every way short of that;
- check_brief.py passes a brief written to the contract and names each defect;
- test_no_planned.py fails naming a seeded planned block (RG3) and passes on a coverage that
  implements every catalog block.

Git history (c849f8b, 0d06aea) is read from this tree, else from $VEGA_SOURCE_REPO (the lane
program runs the unit tier in a copy of the skills tree that is no checkout). The computed
layer of the style lock needs a headless browser ($CHROME, google-chrome, chromium).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import check_brief  # noqa: E402
import check_unattended  # noqa: E402
import lint_templates  # noqa: E402
import style_lock  # noqa: E402
import verifier_gate  # noqa: E402

LEGACY_PAGE = "vega-forge/scripts/backends/pytorch/templates/serving/static/index.html"
DOMAIN = "vega-forge/scripts/backends/pytorch/templates/domain.py"
MODEL_CARD = "vega-pack/scripts/model_card.py"
CHROME = style_lock.find_chrome(None)
needs_browser = pytest.mark.skipif(CHROME is None, reason="no headless browser ($CHROME, google-chrome, chromium)")


def _run(main, *argv) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = main([str(a) for a in argv])
    return code, out.getvalue()


def git_tree(commit: str, dest: Path) -> Path:
    """The tree of a commit, from this checkout or $VEGA_SOURCE_REPO; a missing history fails."""
    errors = []
    for repo in [str(REPO)] + [r for r in [os.environ.get("VEGA_SOURCE_REPO")] if r]:
        try:
            blob = subprocess.run(["git", "-C", repo, "archive", commit], capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{repo}: {exc}")
            continue
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(dest, filter="data")
        return dest
    pytest.fail(f"git archive {commit} failed ({'; '.join(errors)})")


# ---- lint_templates.py -----------------------------------------------------------------------------------
KNOWN_C849 = (("vega-pack/scripts/pack.py", "Label", "Label"),                         # _class_attribution fallback
              ("vega-pack/scripts/pack.py", "flow", "a typical flow"),                  # _class_attribution
              ("vega-pack/scripts/pack.py", "flow", "The flow classifier"),             # DOCKERFILE
              ("vega-pack/scripts/pack.py", "CIDS", "useradd --create-home --uid 10001 cids"),
              ("vega-pack/scripts/pack.py", "CIDS", 'or "CIDS"'),                       # API_TEST env fallback
              ("vega-pack/scripts/pack.py", "port", "def test_port_as_number_or_text_is_one_category"),
              (LEGACY_PAGE, "extractor", "as the extractor writes them"),              # the CICFlowMeter hint
              (LEGACY_PAGE, "capture", "one capture day"),
              (LEGACY_PAGE, "traffic", "#traffic{"),                                 # the traffic field
              (DOMAIN, "traffic", "class TrafficClass:"),                               # D3 G7, D2 gap 11
              (DOMAIN, "traffic", "label: TrafficClass | None"),
              (DOMAIN, "IDS", "because an IDS answer feeds"),
              (DOMAIN, "IDS", "an IDS answer is a class"),
              (MODEL_CARD, "BENIGN", "MCC, benign vs attack"),                           # post_eval_block
              (MODEL_CARD, "attack", "missed attacks"),
              (MODEL_CARD, "BENIGN", "false-positive rate on benign flows"),
              (MODEL_CARD, "BENIGN", "benign vs attack, counts on the test part"))


def test_lint_flags_the_known_cic_literals_at_c849f8b(tmp_path):
    report = lint_templates.lint(root=git_tree("c849f8b", tmp_path / "c849"))
    for rel, token, text in KNOWN_C849:
        assert any(h.file == rel and h.token == token and text in h.text for h in report.hits), \
            f"{rel}: {token!r} in {text!r} is not flagged"


def test_lint_reads_every_pack_literal_and_allows_the_alias_line_only(tmp_path):
    (tmp_path / DOMAIN).parent.mkdir(parents=True)
    (tmp_path / DOMAIN).write_text("TrafficClass = ClassValue\nclass TrafficClass:\n    pass\n", encoding="utf-8")
    (tmp_path / MODEL_CARD).parent.mkdir(parents=True)
    (tmp_path / MODEL_CARD).write_text(
        '"""The card; no attack word here counts."""\n\n\ndef post_eval_block(b):\n    """Every attack row."""\n'
        '    return table(["benign rows", "attack_rows"], caption=f"{b} counts")  # a missed attack\n', encoding="utf-8")
    report = lint_templates.lint(root=tmp_path)
    assert [(h.file, h.line, h.token) for h in report.allowed] == [(DOMAIN, 1, "traffic")], report.allowed
    assert [(h.file, h.line, h.token) for h in report.hits] == [(DOMAIN, 2, "traffic"), (MODEL_CARD, 6, "BENIGN")]


GENERATORS_INIT = '''import importlib
class Generators(dict):  # T02's lazy mapping: a module is loaded, and checked, on access
    def __getitem__(self, name):
        module = importlib.import_module(f"{{__name__}}.{{name}}")
        if not hasattr(module, "VOCAB"):
            raise KeyError(f"generator {{name!r}} lacks VOCAB")
        return module
GENERATORS = Generators.fromkeys({names!r})
'''


def generator_tree(root: Path, generators: dict) -> Path:
    """A tree with a shared template and a lanekit.datasets holding the given {name: source}."""
    datasets = root / "vega" / "scripts" / "tests" / "lanekit" / "datasets"
    datasets.mkdir(parents=True)
    (datasets.parent / "__init__.py").write_text("", encoding="utf-8")
    (datasets / "__init__.py").write_text(GENERATORS_INIT.format(names=tuple(generators)), encoding="utf-8")
    for name, source in generators.items():
        (datasets / f"{name}.py").write_text(source, encoding="utf-8")
    page = root / "vega-show" / "assets" / "page"
    page.mkdir(parents=True)
    (page / "index.html").write_text("<p>one parcel per row</p>\n", encoding="utf-8")
    return root


def test_lint_reads_the_lane_vocab_and_stops_on_a_broken_generator(tmp_path):
    ok = generator_tree(tmp_path / "ok", {"parcels": 'VOCAB = frozenset({"parcel", "row"})\n'})
    code, out = _run(lint_templates.main, "--root", ok)
    assert code == 1 and "vega-show/assets/page/index.html:1: parcel" in out and ": row" not in out, out
    assert "lane VOCAB of 1 generator(s): parcels" in out, out
    broken = generator_tree(tmp_path / "broken", {"parcels": 'VOCAB = frozenset({"parcel"})\n',
                                                  "broken": "SCHEMA = ()\n"})
    code, out = _run(lint_templates.main, "--root", broken)
    assert code == 2 and "ERROR: lanekit.datasets cannot be read" in out and "lacks VOCAB" in out, out
    with pytest.raises(lint_templates.LintError, match="lacks VOCAB"):
        lint_templates.lint(root=broken)
    assert "lanekit" not in sys.modules or not str(sys.modules["lanekit"].__file__).startswith(str(tmp_path))


def test_lint_names_a_seeded_benign_by_file_and_line(tmp_path):
    page = tmp_path / "page"
    shutil.copytree(REPO / "vega-show" / "assets" / "page" / "js" / "core", page)
    target = page / "fmt.js"
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join(lines[:3] + ['const x = "BENIGN";'] + lines[3:]) + "\n", encoding="utf-8")
    code, out = _run(lint_templates.main, page)
    assert code == 1 and f"{target}:4: BENIGN" in out, out


def test_lint_reads_each_word_in_its_dataset_sense_only(tmp_path):
    generic = tmp_path / "generic.py"
    generic.write_text("# a CUDA-graph capture requires warm-up steps before the capture: the label column\n"
                       "for a, b in zip(xs, ys):  # the Predictor port, --port 8765, blueprint_port()\n"
                       "    pass  # a window drawn from one capture day of flows\n", encoding="utf-8")
    code, out = _run(lint_templates.main, generic)
    assert code == 1 and ":3: capture" in out and ":3: flows" in out, out
    assert ":1:" not in out and ":2:" not in out, \
        "a CUDA-graph capture, zip(, an architecture port and a service port are not dataset literals"


def test_lint_allowance_needs_a_reason_and_covers_its_sense(tmp_path):
    target = tmp_path / "t.md"
    target.write_text("the flow of the argument\nevery flow of the capture\n", encoding="utf-8")
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"allow": [{"file": str(target), "token": "flow"}]}), encoding="utf-8")
    assert _run(lint_templates.main, target, "--allow", allow)[0] == 2
    allow.write_text(json.dumps({"allow": [{"file": str(target), "token": "flow", "match": "argument",
                                            "reason": "prose about an argument's course"}]}), encoding="utf-8")
    code, out = _run(lint_templates.main, target, "--allow", allow)
    assert code == 1 and ":2: flow" in out and ":1: flow" not in out and "1 allowed" in out, out


def test_page_leaks_follow_the_lane_vocab():
    text = "Every flow of BENIGN traffic, and one parcel"
    assert lint_templates.page_leaks(text) == ["BENIGN", "flow", "traffic"]
    assert lint_templates.page_leaks(text, own_vocab=frozenset({"flow", "traffic"})) == ["BENIGN"]
    assert lint_templates.page_leaks(text, other_vocab=frozenset({"parcel", "rows"})) == \
        ["BENIGN", "flow", "parcel", "traffic"]
    assert lint_templates.page_leaks("the traffic network of an IDS", own_vocab=frozenset({"traffic"})) == ["IDS"]


# ---- style_lock.py -----------------------------------------------------------------------------------
def reference_copy(tmp_path: Path) -> Path:
    page, _, _ = style_lock.read_page(style_lock.REFERENCE)
    out = tmp_path / "reference.html"
    out.write_text(page, encoding="utf-8")
    return out


def test_style_lock_holds_statically_and_fails_on_a_one_token_edit_and_a_dropped_breakpoint(tmp_path):
    ref = reference_copy(tmp_path)
    assert _run(style_lock.main, "check", ref, "--static-only")[0] == 0
    text = ref.read_text(encoding="utf-8")
    ref.write_text(text.replace("--accent:#3fd0c2", "--accent:#3fd0c3", 1), encoding="utf-8")
    code, out = _run(style_lock.main, "check", ref, "--static-only")
    assert code == 1 and "token --accent [night]" in out and "#3fd0c3" in out, out
    ref.write_text(text.replace("max-width:600px", "max-width:640px"), encoding="utf-8")
    code, out = _run(style_lock.main, "check", ref, "--static-only")
    assert code == 1 and "breakpoints:" in out and "missing [600]" in out, out
    page = tmp_path / "template"
    shutil.copytree(style_lock.TEMPLATE, page)
    with (page / "css" / "sequence.css").open("a", encoding="utf-8") as sheet:
        sheet.write("@media (max-width: 800px){.x{color:var(--fg)}}\n")
    assert _run(style_lock.main, "check", page, "--static-only")[0] == 0, "a new breakpoint is an addition"


@needs_browser
def test_style_lock_holds_on_0d06aea_and_the_page_template():
    for target in (style_lock.REFERENCE, style_lock.TEMPLATE):
        code, out = _run(style_lock.main, "check", target, "--strict")
        assert code == 0 and "static and computed" in out, out


@needs_browser
def test_style_lock_computed_layer_catches_a_token_and_a_rule_edit(tmp_path):
    ref = reference_copy(tmp_path)
    ref.write_text(ref.read_text(encoding="utf-8").replace("--accent:#0b8a7f", "--accent:#0b8a80", 1), encoding="utf-8")
    code, out = _run(style_lock.main, "check", ref, "--strict")
    assert code == 1 and "day nav.index button.on color" in out and "day token --accent" in out, out
    page = tmp_path / "template"
    shutil.copytree(style_lock.TEMPLATE, page)
    (page / "css" / "views.css").write_text(".verdict{letter-spacing:.01em}\n", encoding="utf-8")
    code, out = _run(style_lock.main, "check", page, "--strict")
    assert code == 1 and "night .verdict letter-spacing" in out, out


def test_style_lock_json_is_the_extract_of_0d06aea():
    lock = style_lock.load_lock()
    page, base, source = style_lock.read_page(style_lock.REFERENCE)
    sheet = style_lock.Sheet.parse(style_lock.page_css(page, base))
    assert lock["source"] == source
    assert (lock["tokens"], lock["keyframes"], lock["reduced_motion"], lock["breakpoints"]) == \
        (sheet.tokens, sheet.keyframes, sheet.reduced_motion, sorted(sheet.breakpoints))
    assert len(lock["computed"]["selectors"]) >= 40 and set(lock["computed"]["themes"]) == {"night", "day"}
    assert lock["tokens"]["night"]["--str"] == "#d9b77a" and lock["breakpoints"] == [600, 1000]


# ---- check_unattended.py -------------------------------------------------------------------------------
SKILL = """# a skill

`OK:` / `WARN:` / `ERROR:` before prose at every stage; `FORK:` when you stop to ask.

**2. Gate.** One `FORK:` per decision family. Wait. Under `--unattended`: apply the defaults and mark them.

- the questions left (the `FORK:` items from step 2) go to the summary;
- a column type is ambiguous: emit the `FORK:` naming the two readings and the one difference between them.
"""
REFERENCE_LIKE = """- When the CRS is unknown, the step's `FORK:` asks the user for it. Wait.
- With two partitions, any `FORK:` names both. Wait for the answer.
- Do not answer a stage's `FORK:` yourself; copy any `FORK:` the stage left into Open.
"""


def test_check_unattended_flags_a_seeded_unanswered_fork(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(SKILL + REFERENCE_LIKE, encoding="utf-8")
    code, out = _run(check_unattended.main, skill, "--root", REPO)
    assert code == 1 and all(f"{skill}:{n}: a FORK: with no --unattended" in out for n in (8, 9, 10)), out
    assert out.count("a FORK: with no --unattended") == 3, "the legend, the references and the answered fork pass"
    skill.write_text(SKILL.replace("between them.", "between them; unattended, the first reading, marked."),
                     encoding="utf-8")
    assert _run(check_unattended.main, skill, "--root", REPO)[0] == 0


def test_check_unattended_reads_the_chain_map_rows(tmp_path):
    chain = tmp_path / "chain-map.md"
    text = (REPO / check_unattended.CHAIN_MAP).read_text(encoding="utf-8")
    assert _run(check_unattended.main, REPO / "vega-forge" / "SKILL.md", "--chain-map", chain.parent / "none.md")[0] == 2
    row = next(ln for ln in text.splitlines() if ln.startswith("| train |"))
    cells = row.split(" | ")
    chain.write_text(text.replace(row, " | ".join(cells[:-2] + ["—", cells[-1]])), encoding="utf-8")
    code, out = _run(check_unattended.main, REPO / "vega-forge" / "SKILL.md", "--chain-map", chain)
    assert code == 1 and "stage train has no unattended behaviour" in out, out


# ---- verifier_gate.py ----------------------------------------------------------------------------------------
def _copy_brief_contract(root):
    """The stage brief's contract, which every stage verifier's Input line names (W8)."""
    rel = Path("vega/references/brief-contract.md")
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / rel, root / rel)


def test_verifier_gate_passes_a_current_ship_and_fails_a_stale_digest(tmp_path):
    root, runs = tmp_path / "skills", tmp_path / "runs"
    for rel in ("SKILL.md", "verifiers/static-verifier.md", "references/eval-contract.md"):
        (root / "vega-eval" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "vega-eval" / rel, root / "vega-eval" / rel)
    _copy_brief_contract(root)          # the stage verifiers' Input names it beside the skill's files
    gate = ("--skill", "vega-eval", "--root", root, "--runs", runs)
    answer = tmp_path / "answer.md"
    answer.write_text('{"verdict":"SHIP","blocking":0,"non_blocking":2,"inconclusive":0}\n\nS1 — PASS\n', encoding="utf-8")
    assert _run(verifier_gate.main, "record", *gate, "--session", "t04", "--response", answer)[0] == 0
    record = json.loads((runs / "vega-eval.json").read_text(encoding="utf-8"))
    assert set(record["audited"]) == {"SKILL.md", "verifiers/static-verifier.md", "references/eval-contract.md",
                                      "../vega/references/brief-contract.md"}
    assert _run(verifier_gate.main, *gate)[0] == 0
    skill = root / "vega-eval" / "SKILL.md"
    skill.write_text(skill.read_text(encoding="utf-8") + "\nOne more directive.\n", encoding="utf-8")
    code, out = _run(verifier_gate.main, *gate)
    assert code == 1 and "vega-eval: SKILL.md changed after the verifier read it" in out, out


def test_verifier_gate_fails_a_revise_an_unaudited_input_and_a_release_without_rehearsal(tmp_path):
    root, runs = tmp_path / "skills", tmp_path / "runs"
    shutil.copytree(REPO / "vega-pack", root / "vega-pack", ignore=shutil.ignore_patterns("scripts", "assets"))
    _copy_brief_contract(root)
    gate = ("--skill", "vega-pack", "--root", root, "--runs", runs)
    assert _run(verifier_gate.main, "record", *gate, "--session", "t04", "--verdict", "REVISE", "--blocking", "1",
               "--audited", "SKILL.md")[0] == 0
    code, out = _run(verifier_gate.main, *gate)
    assert code == 1 and "says REVISE with 1 blocking" in out and "did not audit references/bundle-contract.md" in out
    assert _run(verifier_gate.main, "record", *gate, "--session", "t04", "--verdict", "SHIP", "--blocking", "0")[0] == 0
    assert _run(verifier_gate.main, *gate)[0] == 0
    code, out = _run(verifier_gate.main, *gate, "--release")
    assert code == 1 and "no release/rehearsal.json" in out, out
    (root / "release").mkdir()
    (root / "release" / "rehearsal.json").write_text(json.dumps(
        {"schema": "vega/rehearsal@1", "session": "s", "checks": [{"id": "check_brief", "verdict": "pass"}]}))
    assert _run(verifier_gate.main, *gate, "--release")[0] == 0


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env=env).stdout.strip()


def test_verifier_gate_release_accepts_fixed_defects_only_with_their_evidence(tmp_path):
    """The rehearsal of record may pass with defects it met, never silently: each one fixed with a
    test on the released head, the run completed and every product check passing."""
    root, runs = tmp_path / "skills", tmp_path / "runs"
    shutil.copytree(REPO / "vega-pack", root / "vega-pack", ignore=shutil.ignore_patterns("scripts", "assets"))
    _copy_brief_contract(root)
    gate = ("--skill", "vega-pack", "--root", root, "--runs", runs, "--release")
    assert _run(verifier_gate.main, "record", *gate[:-1], "--session", "t", "--verdict", "SHIP", "--blocking", "0")[0] == 0
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    test = Path("vega-pack/scripts/tests/test_fix.py")
    (root / test).parent.mkdir(parents=True)
    (root / test).write_text("def test_fix():\n    pass\n")
    (root / "vega-pack/fixed.py").write_text("FIXED = True\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix F1 with its test")
    fix = _git(root, "rev-parse", "--short", "HEAD")
    (root / "vega-pack/other.py").write_text("OTHER = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "a change with no test")
    untested = _git(root, "rev-parse", "--short", "HEAD")
    _git(root, "checkout", "-q", "-b", "side", fix)
    (root / test).write_text("def test_fix():\n    assert True\n")
    _git(root, "commit", "-q", "-am", "a fix that never reached main")
    off_head = _git(root, "rev-parse", "--short", "HEAD")
    _git(root, "checkout", "-q", "main")

    def record(run=None, product="pass", defects=None):
        run = {"id": "T5-run", "verdict": "pass-with-fixed-defects", "completed": True, "defects": ["F1"],
               **(run or {})}
        doc = {"schema": "vega/rehearsal@1", "session": "s",
               "checks": [run, {"id": "check_brief", "verdict": product}],
               "defects_fixed": defects if defects is not None else
               [{"id": "F1", "summary": "s", "fix": fix, "tests": [str(test)]}]}
        (root / "release").mkdir(exist_ok=True)
        (root / "release" / "rehearsal.json").write_text(json.dumps(doc))
        return _run(verifier_gate.main, *gate)

    code, out = record()
    assert code == 0 and f"NOTE: release: T5-run passed with a fixed defect: F1 fixed in {fix}" in out, out
    code, out = record(product="fail")
    assert code == 1 and "rehearsal check check_brief is fail" in out, out
    code, out = record(product="pass-with-fixed-defects")
    assert code == 1 and "only the run's own check (T5-run) may be" in out, out
    code, out = record(run={"completed": False})
    assert code == 1 and "not completed: true" in out, out
    code, out = record(run={"defects": ["F1", "F2"]})
    assert code == 1 and "defect F2, named by T5-run, is not in defects_fixed" in out, out
    code, out = record(defects=[])
    assert code == 1 and "has no defects_fixed" in out, out
    code, out = record(defects=[{"id": "F1", "summary": "s", "fix": fix, "tests": []}])
    assert code == 1 and "names no test" in out, out
    code, out = record(defects=[{"id": "F1", "summary": "s", "fix": untested, "tests": [str(test)]}])
    assert code == 1 and "adds or changes none of its tests" in out, out
    code, out = record(defects=[{"id": "F1", "summary": "s", "fix": off_head, "tests": [str(test)]}])
    assert code == 1 and "is not on the released head" in out, out
    code, out = record(defects=[{"id": "F1", "summary": "s", "fix": "0000000", "tests": [str(test)]}])
    assert code == 1 and "is not a commit of this repository" in out, out
    code, out = record(defects=[{"id": "F1", "summary": "s", "fix": fix, "tests": ["no/such_test.py"]}])
    assert code == 1 and "test no/such_test.py is not in the tree" in out, out
    shutil.rmtree(root / ".git")
    code, out = record()
    assert code == 1 and "is not a git checkout" in out, out


# ---- check_brief.py ----------------------------------------------------------------------------------------
def write_brief(project: Path, sections=check_brief.SECTIONS) -> Path:
    run_dir = project / "vega-runs" / "2026-09-26T10-12Z"
    (run_dir / "docs").mkdir(parents=True)
    layer = run_dir / "architecture-layer.json"
    layer.write_text('{"params": 12345}\n', encoding="utf-8")
    body = "\n".join(f"## {s}\n\n" + ("```\npython3 compile_spec.py --check-only\nOK: compiles\n```\n"
                                       if s == "What ran" else "Text (architecture-layer.json.params).\n")
                     for s in sections)
    brief = run_dir / "docs" / "net-brief.md"
    brief.write_text(f"""---
stage: net                          # chain-map stage name
skill: vega-forge
lane: dataset:orders
input: orders.shaped.html
output: orders.shaped.vega.html
run_dir: vega-runs/2026-09-26T10-12Z
layer: architecture
date: 2026-09-26T10:41Z
records:
  - path: vega-runs/2026-09-26T10-12Z/architecture-layer.json
    sha256: {hashlib.sha256(layer.read_bytes()).hexdigest()}
supersedes: null
---

{body}""", encoding="utf-8")
    return brief


def test_check_brief_passes_the_contract_and_names_each_defect(tmp_path):
    brief = write_brief(tmp_path / "ok")
    assert _run(check_brief.main, brief)[0] == 0
    (brief.parents[1] / "architecture-layer.json").write_text('{"params": 12346}\n', encoding="utf-8")
    code, out = _run(check_brief.main, brief)
    assert code == 1 and "its sha256 differs" in out, out
    swapped = list(check_brief.SECTIONS)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    code, out = _run(check_brief.main, write_brief(tmp_path / "order", swapped))
    assert code == 1 and "not the seven in order" in out, out
    assert _run(check_brief.main)[0] == 2


def test_check_brief_requires_the_layers_open_items_in_open(tmp_path):
    """A confirmed judgement finding is advisory (code-contract §5): the open item the layer
    records is named in the brief's Open section, as `<tool> <audit>:<id>`."""
    brief = write_brief(tmp_path / "advisory")
    item = {"tool": "architect", "finding": "1:A-3", "text": "architect 1:A-3 (structural, …)"}
    layer = {"conformance": {"mechanical": "pass", "architect": {
        "tool": "architect", "advisory": {"mode": "unattended", "outcome": "warn", "open_items": [item]}}}}
    (tmp_path / "advisory" / "orders.shaped.vega.html").write_text(
        '<html><script type="application/json" id="layer-architecture-data">'
        + json.dumps(layer) + "</script></html>", encoding="utf-8")
    code, out = _run(check_brief.main, brief)
    assert code == 1 and "does not name the open item architect 1:A-3" in out, out
    text = brief.read_text(encoding="utf-8").replace(
        "## Open\n\nText (architecture-layer.json.params).",
        "## Open\n\nThe advisory finding architect 1:A-3 is an open item for /architect and /improve.")
    brief.write_text(text, encoding="utf-8")
    assert _run(check_brief.main, brief)[0] == 0
    assert check_brief.layer_open_items("<html></html>", "architecture") == []


# ---- test_no_planned.py ------------------------------------------------------------------------------------
def no_planned(tmp_path: Path, edit) -> subprocess.CompletedProcess:
    backends = tmp_path / "backends" / "pytorch"
    backends.mkdir(parents=True)
    coverage = json.loads((REPO / "vega-forge/scripts/backends/pytorch/coverage.json").read_text(encoding="utf-8"))
    edit(coverage, json.loads((REPO / "vega-forge/scripts/catalog.json").read_text(encoding="utf-8")))
    (backends / "coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    env = {**os.environ, "VEGA_BACKENDS_DIR": str(backends.parent), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           str(REPO / "vega-forge/scripts/tests/test_no_planned.py")],
                          capture_output=True, text=True, env=env, timeout=300)


def _complete(coverage: dict, catalog: dict) -> None:
    coverage["planned"] = {"note": coverage["planned"].get("note", ""), "blocks": [], "families": [], "metrics": []}
    coverage["blocks"] = sorted(set(coverage["blocks"]) | set(catalog["blocks"]))


def test_no_planned_names_a_seeded_planned_block_and_passes_a_complete_backend(tmp_path):
    seeded = no_planned(tmp_path / "seeded", lambda cov, cat: (_complete(cov, cat), cov["planned"]["blocks"].append("gcn_conv"),
                                                               cov["blocks"].remove("gcn_conv")))
    assert seeded.returncode == 1 and "coverage.json still plans blocks: gcn_conv" in seeded.stdout, seeded.stdout[-3000:]
    assert re.search(r"gcn x graph/node x binary: .*'gcn_conv'", seeded.stdout), seeded.stdout[-3000:]
    complete = no_planned(tmp_path / "complete", _complete)
    assert complete.returncode == 0, complete.stdout[-3000:]
