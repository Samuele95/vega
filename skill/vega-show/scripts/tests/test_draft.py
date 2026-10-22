#!/usr/bin/env python3
"""
test_draft.py — the draft (`draft_plan.py`) and its proposers.

    python3 test_draft.py            # or: python3 -m pytest test_draft.py

- **The CIDS draft** (when the CIC-IDS2017 bundle is present, `CIDS_ROOT`, default
  `~/Documenti/cids`; *skipped* otherwise, never passed): `BENIGN` neutral by
  `structural:majority` and no alert; the 65 shaped inputs partitioned into 3–9 groups;
  `net.forward → eq:mlp` and `pack.slice → fig:serving` proposed from the hand-authored
  report's `.aux`; the row noun from `MANIFEST.serve.identity`; the mandatory choices.
- **The motif** is the modality's base (`contours` on a spatial row); a `show: motif` stance is
  applied and cited with its date, never re-asked; one outside the admissible list is a `WARN:`.
- **The class roles**: `alert` is never proposed without a `show:` stance or `task:anomaly`,
  over every admitted cell; the binary negative is neutral (`structural:negative`), the positive
  only `spec:` evidence; the anomalous side is never downgraded; a stance keeps the structural
  basis it confirms; the unattended answer downgrades an unsupported alert with the S2 `WARN:`.
- **The lexicon** proposes the row noun from `MANIFEST.serve.identity` and the required keys of
  the lane's shape; a `show: lexicon` stance replaces its entry.
- **The input groups** partition the shaped inputs within the count bounds.
- **The report roles and code steps**: emitted, proposed and absent (brief links); a report
  changed since the facts were extracted is refused (PF6).
- **Every admitted cell** drafts a plan its schema accepts; the CLI pins its inputs, appends its
  markers and refuses (exit 2) what it cannot read. The carry is `test_carry.py`'s.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _draft_fixtures as fx  # noqa: E402  (puts the scripts on the path)
import coverage_fixtures as cf  # noqa: E402
from memory_stances import MemoryStances  # noqa: E402
from propose_families import FamilyProposer  # noqa: E402
from propose_labels import LabelRoleProposer, code_steps, report_proposal  # noqa: E402
from propose_lexicon import LexiconProposer  # noqa: E402
from propose_motif import MotifProposer  # noqa: E402
from propose_roles import RoleProposer, s2_open  # noqa: E402
from report_labels import AuxLabelSource  # noqa: E402
from vega_show_core import CATALOG, digest_file  # noqa: E402


# --------------------------------------------------------------------------- the CIDS draft

def test_the_cids_draft():
    fx.need_cids()
    with tempfile.TemporaryDirectory() as tmp:
        code, plan, printed = fx.draft_cli(fx.cids_facts(Path(tmp)), Path(tmp) / "draft.json")
        assert code == 0, printed
        props = plan["proposals"]
        roles = props["class_roles"]
        assert roles["neutral"] == {"class": "BENIGN", "rule": "structural:majority",
                                    "share": "train_share_benign"}, roles["neutral"]
        assert {c: r for c, r in roles["roles"].items() if r != "plain"} == {"BENIGN": "neutral"}
        assert roles["fork"] == "S2" and roles["basis_found"] is None
        assert len(roles["alert_candidates"]) == 14
        facts = json.loads((Path(tmp) / "facts.json").read_text())
        shaped = facts["groups"]["inputs"]["shaped"]
        assert len(shaped) == 65
        assert not fx.partition_problems(props["input_groups"], shaped, 3, 9), props["input_groups"]
        bindings = props["report_roles"]["bindings"]
        assert props["report_roles"]["status"] == "proposed"
        assert bindings["net.forward"]["label"] == "eq:mlp"
        assert bindings["pack.slice"]["label"] == "fig:serving"
        assert props["lexicon"]["row"] == {"one": "flow", "many": "flows", "article": "a",
                                           "source": "bundle:MANIFEST.serve.identity.noun"}
        assert props["motif"] == {"modality": "row", "base": "points", "stance": None,
                                  "admissible": ["points", "lanes", "grid"], "fork": "S3"}
        mandatory = {c["decision"] for c in props["choices"] if c["mandatory"]}
        assert mandatory == {"family", "loss", "optimizer", "regularization[grad_clip]"}, mandatory
        runs = props["runs"]
        assert [r["id"] for r in runs if r["selected"]] == ["R7"]
        assert [r["id"] for r in runs if r["reference"]] == ["R0"]
        assert {s["step"]: s["role"] for s in props["code_steps"]}["dense"] == "net.forward"
        assert "forks open: S2, S3" in printed, printed


# --------------------------------------------------------------------------- motif

def test_the_motif_is_the_modality_base_and_contours_on_a_spatial_row():
    spatial = fx.cell("regression")
    spatial["records/analysis.json"]["context"]["spatial"] = {"lat": "lat", "lon": "lon"}
    window = {"family": "gru", "shape": "window", "level": None, "task": "forecast",
              "variant": None}
    graph = {"family": "gcn", "shape": "graph", "level": "node", "task": "multiclass",
             "variant": None}
    cases = {"row": (fx.cell("regression"), "points"), "spatial-row": (spatial, "contours"),
             "window": (cf.cell_docs(window), "traces"), "graph": (cf.cell_docs(graph), "links")}
    for modality, (docs, base) in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            code, plan, printed, _ = fx.draft(Path(tmp), docs, ["--unattended"])
            assert code == 0, printed
            motif = plan["proposals"]["motif"]
            assert plan["proposals"]["modality"] == modality and motif["base"] == base
            assert plan["authored"]["motif"] == base
            assert motif["admissible"] == CATALOG["motifs"]["by_modality"][modality]["admissible"]
            assert f"OK: motif {base} (the modality's base, unattended)" in printed


def test_a_motif_stance_is_applied_and_cited_never_asked():
    stance = MotifProposer("row", fx.stances(["show: motif lanes (user)"]))
    proposal = stance.propose()
    assert proposal["fork"] is None and stance.answer(proposal) == "lanes"
    assert proposal["stance"] == f"{fx.DAY} | demo | show: motif lanes (user)"
    wrong = MotifProposer("graph", fx.stances(["show: motif lanes (user)"]))
    assert wrong.propose()["fork"] == "S3" and "does not admit" in wrong.markers[0]
    with tempfile.TemporaryDirectory() as tmp:
        memory = fx.memory(Path(tmp), ["show: motif grid (user)"])
        code, plan, printed, _ = fx.draft(Path(tmp), fx.cell("regression"), ["--memory", memory])
        assert code == 0, printed
        assert plan["proposals"]["motif"]["stance"] == f"{fx.DAY} | demo | show: motif grid (user)"
        assert "S3" not in printed.split("forks open:")[1]


# --------------------------------------------------------------------------- class roles

def test_alert_is_never_proposed_without_a_stance_or_the_anomaly_task():
    failed, shaped = [], [n for n, _ in cf.FEATURES]
    for cell in cf.cells():
        with tempfile.TemporaryDirectory() as tmp:
            docs = cf.cell_docs(cell)
            code, plan, printed, _ = fx.draft(Path(tmp), docs)
            if code != 0:
                failed.append(f"{cell}: exit {code}: {printed}")
                continue
            roles = plan["proposals"]["class_roles"] or {"roles": {}, "source": {}}
            bad = [c for c, r in roles["roles"].items()
                   if r == "alert" and roles["source"][c] != "task:anomaly"]
            failed += [f"{cell}: alert {bad}"] if bad else []
            groups = plan["proposals"]["input_groups"]
            # the network's inputs: the features, then an autoregressive window's target channel
            inputs = shaped + (["target"] if docs["netspec.json"]["data_contract"]["sample"].get("autoregressive") else [])
            bounds = (1, len(inputs))
            failed += [f"{cell}: {p}" for p in fx.partition_problems(groups, inputs, *bounds)]
    assert len(cf.cells()) >= 120 and not failed, failed[:3]


BINARY = {"kind": "binary", "view_key": "binary", "primary": "pr_auc"}
ANOMALY = {"kind": "anomaly", "goal": "anomaly", "view_key": "anomaly", "direction": "min",
           "outcomes": ["normal", "anomalous"], "primary": "anomaly_score_mean"}


def test_the_structural_rules_and_the_stances():
    sides = {"classes": ["no", "yes"], "outcomes": ["no", "yes"]}
    binary = RoleProposer(fx.task({**BINARY, **sides}), {}, MemoryStances()).propose()
    assert binary["neutral"]["class"] == "no" and binary["neutral"]["rule"] == "structural:negative"
    assert binary["roles"]["yes"] == "plain" and binary["fork"] == "S2"
    assert binary["evidence"] == ["spec:task.positive yes"]
    line = "show: roles 0 neutral (structural:negative), 1 alert (user:lanes) (user)"
    orders = fx.task({**BINARY, "classes": ["0", "1"], "outcomes": ["0", "1"]})
    stanced = RoleProposer(orders, {}, fx.stances([line])).propose()
    assert stanced["roles"] == {"0": "neutral", "1": "alert"} and stanced["fork"] is None
    assert stanced["source"] == {"0": "structural:negative", "1": f"show:{fx.DAY}"}
    assert stanced["basis_found"] == f"show:{fx.DAY}"
    kept = RoleProposer(fx.task(ANOMALY), {}, fx.stances(["show: roles anomalous plain (user)"]))
    proposal = kept.propose()
    assert proposal["roles"] == {"normal": "neutral", "anomalous": "alert"}
    assert proposal["fork"] is None and set(proposal["source"].values()) == {"task:anomaly"}
    assert "never downgraded" in kept.markers[0]
    shares = {"a": ("train_share_a", 0.7), "b": ("train_share_b", 0.2)}
    majority = RoleProposer(fx.task({"classes": ["a", "b", "c"]}), shares, MemoryStances())
    assert majority.propose()["neutral"] == {"class": "a", "rule": "structural:majority",
                                             "share": "train_share_a"}
    halves = {"a": ("k", 0.5), "b": ("j", 0.5)}
    even = RoleProposer(fx.task({"classes": ["a", "b"]}), halves, MemoryStances()).propose()
    assert even["neutral"] is None and even["fork"] is None
    assert set(even["roles"].values()) == {"plain"}
    value = fx.task({"kind": "regression", "view_key": "value", "primary": "rmse",
                     "direction": "min"})
    assert RoleProposer(value, {}, MemoryStances()).propose() is None


def test_unattended_roles_downgrade_an_unsupported_alert():
    roles = RoleProposer(fx.task({"classes": ["a", "b"]}), {"a": ("k", 0.9)}, MemoryStances())
    proposal = roles.propose()
    answer, markers = roles.unattended(proposal)
    assert answer["roles"] == {"a": "neutral", "b": "plain"}
    assert "downgraded" in markers[0] and markers[0].endswith("— b")
    carried = {"roles": {"a": "neutral", "b": "alert"},
               "source": {"a": "structural:majority", "b": "vega:default"}}
    answer, markers = roles.unattended(proposal, carried)
    assert answer["roles"]["b"] == "plain" and answer["source"]["b"] == "vega:default" and markers
    user = {"roles": {"a": "neutral", "b": "alert"},
            "source": {"a": "structural:majority", "b": "user:turn-3"}}
    assert not s2_open(proposal, user) and roles.unattended(proposal, user) == (user, [])


# --------------------------------------------------------------------------- lexicon

def test_the_lexicon_comes_from_the_identity_and_the_shape():
    identity = {"noun": "Order", "nouns": "orders", "title": "Late orders"}
    row = LexiconProposer(fx.task(), identity, MemoryStances()).propose()
    assert list(row) == ["row", "feature", "source"]
    assert row["row"] == {"one": "order", "many": "orders", "article": "an",
                          "source": "bundle:MANIFEST.serve.identity.noun"}
    assert row["source"] == {"one": "source data", "many": "source data", "article": "the",
                             "source": "vega:default"}
    window = LexiconProposer(fx.task({"sample_shape": "window"}), {}, MemoryStances())
    assert list(window.propose()) == ["row", "feature", "source", "step", "window"]
    assert "names no row noun" in window.markers[0]
    level = fx.task({"sample_shape": "graph", "graph_level": "graph"})
    graph = LexiconProposer(level, {"noun": "mol"}, MemoryStances())
    assert list(graph.propose()) == ["row", "feature", "source", "node", "edge", "graph"]
    assert "without its plural" in graph.markers[0]
    lines = ["show: lexicon feature statistic/statistics (user)",
             "show: lexicon target price/prices (user)", "show: lexicon colour Red/reds (user)"]
    stanced = LexiconProposer(fx.task(), identity, fx.stances(lines))
    entries = stanced.propose()
    assert entries["feature"] == {"one": "statistic", "many": "statistics", "article": "a",
                                  "source": f"show:{fx.DAY}"}
    assert entries["target"]["one"] == "price" and "colour" not in entries
    assert "not a lexicon entry" in stanced.markers[0]


# --------------------------------------------------------------------------- input groups

def test_the_input_groups_partition_within_the_bounds():
    names = [f"Wind Speed {s}" for s in ("Mean", "Max", "Min")] + \
        [f"Blade Pitch {s}" for s in ("Mean", "Max")] + [f"col{c}" for c in "abcdefghijkl"] + \
        ["Site", "Model", "flag_missing", "rate_missing"]
    inputs = {"shaped": names, "created_by_recipe": ["flag_missing", "rate_missing"],
              "encoded": ["Site", "Model"],
              "relations": [{"a": "cola", "b": "colb", "r": 0.95},
                            {"a": "colb", "b": "colc", "r": -0.9},
                            {"a": "cold", "b": "cole", "r": 0.3}]}
    recipe = [{"step": "S9", "op": "impute", "columns_added": ["flag_missing", "rate_missing"]}]
    groups = FamilyProposer(inputs, recipe).propose()
    assert not fx.partition_problems(groups, names, 3, 9), groups
    kinds = {e["kind"] for g in groups for e in g["evidence"]}
    assert {"created", "encoded", "correlation", "name-token"} <= kinds, kinds
    assert groups == FamilyProposer(inputs, recipe).propose(), "deterministic"
    few = FamilyProposer({"shaped": ["a", "b"]}, []).propose()
    assert not fx.partition_problems(few, ["a", "b"], 1, 2)


def test_an_autoregressive_window_places_its_target_channel_in_a_group():
    """The network reads an autoregressive forecast's target after the spec's features (the
    network view lists it last): the facts name it an input, not one the recipe created, and the
    draft's groups place it (else the page draws one family and warns, T09's L10)."""
    docs = cf.cell_docs({"family": "gru", "shape": "window", "level": None, "task": "forecast", "variant": None})
    assert docs["netspec.json"]["data_contract"]["sample"]["autoregressive"] is True
    with tempfile.TemporaryDirectory() as tmp:
        code, plan, printed, facts = fx.draft(Path(tmp), docs, ["--unattended"])
    assert code == 0, printed
    features = [f["name"] for f in docs["netspec.json"]["data_contract"]["features"]]
    inputs = facts["groups"]["inputs"]
    assert inputs["shaped"] == features + ["target"] and inputs["origin"]["target"] == "autoregressive"
    assert "target" not in inputs["created_by_recipe"]
    placed = [m for g in plan["proposals"]["input_groups"] for m in g["members"]]
    assert sorted(placed) == sorted(inputs["shaped"]), plan["proposals"]["input_groups"]
    docs["netspec.json"]["data_contract"]["sample"]["autoregressive"] = False
    with tempfile.TemporaryDirectory() as tmp:
        _, _, _, plain = fx.draft(Path(tmp), docs, ["--unattended"])
    assert plain["groups"]["inputs"]["shaped"] == features


# --------------------------------------------------------------------------- report roles

def test_report_roles_emitted_proposed_or_absent():
    with tempfile.TemporaryDirectory() as tmp:
        labels = AuxLabelSource(cf.write_report(Path(tmp)).parent / "main.aux").labels()
    proposed = LabelRoleProposer(labels, CATALOG["report_roles"]).propose()
    assert proposed["net.forward"]["label"] == "vega:net.forward"
    assert proposed["pack.slice"]["label"] == "vega:pack.slice"
    assert all(CATALOG["report_roles"][r]["kind"] == str(labels[b["label"]].kind)
               for r, b in proposed.items())
    linked = {"status": "linked", "roles_source": "docs/label-roles.json",
              "roles": {"net.forward": {"label": "eq:x"}, "nonsense": {"label": "y"}}}
    emitted = {"net.forward": {"label": "eq:x", "source": "document-project", "evidence": None}}
    assert report_proposal(linked, None) == {"status": "emitted", "fork": None, "bindings": emitted}
    absent = report_proposal({"status": "brief-links"}, None)
    assert absent == {"status": "absent", "bindings": {}, "fork": "S1"}
    walk = ["root", "dense", "explain", "route"]
    briefs = code_steps(walk, absent, {"pack": {}, "net": {}})
    assert {s["step"]: (s["role"], s["status"]) for s in briefs} == {
        "root": (None, "no-role"), "dense": ("net.forward", "brief"),
        "explain": ("net.forward", "brief"), "route": ("pack.slice", "brief")}
    bound = code_steps(walk, report_proposal(linked, None), {})
    assert {s["step"]: s["status"] for s in bound} == {
        "root": "no-role", "dense": "bound", "explain": "bound", "route": "no-role"}


def test_a_report_the_facts_pinned_then_changed_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        docs_dir, facts = Path(tmp) / "docs", Path(tmp) / "plan" / "facts.json"
        cf.write_report(docs_dir)
        (docs_dir / "label-roles.json").unlink()  # a hand-authored report: roles from its .aux
        code, _, printed = cf.run(Path(tmp), fx.cell("binary"), extra=("--report", str(docs_dir)))
        assert code == 0, printed
        code, plan, printed = fx.draft_cli(facts, Path(tmp) / "draft.json")
        report = plan["proposals"]["report_roles"]
        assert code == 0 and report["status"] == "proposed", printed
        assert report["bindings"]["net.forward"]["label"] == "vega:net.forward"
        with (docs_dir / "main.aux").open("a", encoding="utf-8") as fh:
            fh.write("\\newlabel{late}{{9}{9}{Late}{section.9}{}}\n")
        code, _, printed = fx.draft_cli(facts, Path(tmp) / "draft.json")
        assert code == 2 and printed.startswith("ERROR: PF6 "), printed
        assert "re-run extract_facts.py" in printed


# --------------------------------------------------------------------------- the command line

def test_the_draft_pins_its_inputs_and_writes_markers():
    with tempfile.TemporaryDirectory() as tmp:
        showcase = Path(tmp) / "bundle" / "showcase.building"
        facts = Path(tmp) / "plan" / "facts.json"
        (showcase / ".work").mkdir(parents=True)
        (showcase / ".work" / "markers.json").write_text("[]", encoding="utf-8")
        cf.run(Path(tmp), fx.cell("multiclass"))
        out = showcase / "plan" / "story-plan.draft.json"
        code, plan, printed = fx.draft_cli(facts, out, ["--json"])
        assert code == 0, printed
        result = json.loads(printed)
        assert result["status"] == "ok" and result["forks"] == ["S1", "S3"], result
        assert plan["facts"] == {"file": "plan/facts.json", "digest": digest_file(facts)}
        assert plan["status"] == "draft" and set(plan["authored"].values()) == {None}
        assert plan["template"]["version"] == "2.3.0"
        assert plan["template"]["manifest_digest"].startswith("sha256:")
        markers = json.loads((showcase / ".work" / "markers.json").read_text())
        assert markers and all(m.startswith("draft_plan.py: ") for m in markers)


def test_the_command_line_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "bad.json").write_text('{"schema": "vega/show-facts@0"}', encoding="utf-8")
        for facts in (tmp / "missing.json", tmp / "bad.json"):
            code, _, printed = fx.draft_cli(facts, tmp / "out.json")
            assert code == 2 and printed.startswith("ERROR: draft "), printed
        cf.run(tmp, fx.cell("regression"))
        old = '{"schema": "vega/story-plan@1", "values": [], "sections": 3}'
        (tmp / "old.json").write_text(old, encoding="utf-8")
        (tmp / "no.json").write_text('{"schema": "other"}', encoding="utf-8")
        for carried in ("old.json", "no.json"):
            code, _, printed = fx.draft_cli(tmp / "plan" / "facts.json", tmp / "out.json",
                                            ["--carry", tmp / carried])
            assert code == 2 and printed.startswith("ERROR: draft "), printed
        assert not (tmp / "out.json").exists()


if __name__ == "__main__":
    sys.exit(fx.run_tests(globals()))
