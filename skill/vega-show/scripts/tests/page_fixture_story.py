"""
page_fixture_story — the fixed authoring of the page fixture matrix (BUILD-PLAN W15; D4 §17.4:
"story/page from the builders here plus a fixed authoring fixture"): for one fixture case, the
`vega/story@2` a showcase of that network would carry, the authored `vega/story-plan@2` that
settles its motif, and the `records` `page_config.build_page_config` assembles `page.json` from.

    story_of(case, spec, lane, metric, walk, output, sizes) -> dict        vega/story@2
    plan_of(case, modality, template) -> dict                              vega/story-plan@2
    shape_record(spec) -> dict                                             records/shape.json
    MOTIFS, ROLES                                                          the matrix's stances

Nothing here is drawn from data: a fixture belongs to no lane, so its words are the generic
lexicon of the vega-show catalog, its figures are placeholders bound to `values` (marked as the
fixture's own by their refs), and it names no dataset. The words still pass the vega-show digit
rule (`digit_rule.check_prose`), so a fixture page is held to what a real showcase is.

**The stances** the matrix varies, so that every motif and both role palettes are rendered:

- `MOTIFS`: the motif a case's plan settles when it is not the modality's base — `lanes` and
  `grid` (rows, as fork S3's answer), `contours` (a row sample whose analysis names a spatial
  context: the `spatial-row` modality, its base), the bases `points`, `traces` and `links`
  elsewhere;
- `briefs` (a matrix entry's): the stage briefs a case ships — the story links the first from
  its `build` code note and lists them under report.briefs (status `brief-links`);
- `ROLES`: the cases whose classes carry an `alert` role by a user stance (a classes lane, the
  golden path's shape) — an anomaly lane's anomalous side is alert by `task:anomaly` — and every
  other case has no alert role at all (the probe then requires no alert colour anywhere);
- `verdict_of`: the verdict each case's evaluation records, in turn earned, not earned and beaten
  by the trivial baseline (case number mod 3), with figures that agree with it — so the results
  view is held to say every state (P-TXT reads #res-verdict against the story).

The authored strings carry lexicon paths where a real plan's may (a context fact's label, a
limit's title, a fact's note, a run's label), so a string the page renders raw shows its brace.
"""
from __future__ import annotations

import copy
from typing import Mapping

from vega_show_core import CATALOG

STANCE_DATE = "2026-09-30"
SOURCE = "user:page-matrix"
FIXTURE_REF = "records/evaluation.json#fixture.{key}"
# case number -> (motif, how the plan settled it: None for the base, "S3" for the fork's answer)
MOTIFS = {1: ("lanes", "S3"), 3: ("grid", "S3"), 8: ("contours", None), 16: ("lanes", "S3"),
          20: ("grid", "S3"), 27: ("grid", "S3")}
SPATIAL = frozenset({8})            # cases whose analysis record names a spatial context
# case number -> the classes a user stance makes alert (every class but the neutral one)
ROLES = {1: "all_but_first", 5: "all_but_first", 19: "positive"}
LIMITS = (("split", "One random split of the {row.many}", "The test part was drawn at random once."),
          ("exposures", "Sealed once", "The test part was opened once, for this evaluation."),
          ("one_seed", "One seed", "Every run used one seed."))


def _estimate(value: float, n: int, width: float = 0.06) -> dict:
    return {"value": value, "ci": [round(value - width, 4), round(value + width, 4)], "n": n}


def _figure(direction: str, verdict: str = "earned") -> tuple[float, float, float]:
    """(network, strongest baseline, run zero) on the metric's scale and direction, agreeing with
    the verdict: ahead of the baseline (earned), level with it (not earned), behind it (beaten)."""
    net, base, run0 = (0.71, 0.66, 0.62) if direction == "max" else (0.84, 0.91, 0.97)
    if verdict == "not_earned":
        net = base + (0.005 if direction == "max" else -0.005)
    elif verdict == "beaten":
        net, base = base, net
    return net, base, run0


def verdict_of(case: Mapping) -> str:
    """The verdict a case's evaluation records (module docstring)."""
    return ("beaten", "earned", "not_earned")[case["n"] % 3]


VERDICT_CI = {"earned": [0.01, 0.09], "not_earned": [-0.02, 0.03], "beaten": [-0.09, -0.01]}


def _breakdown(kind: str, base: str | None, output: Mapping, values: dict, formats: dict) -> dict | None:
    """A per-unit breakdown where the task has one (the vega-show catalog's `breakdown`)."""
    rule = CATALOG["breakdown"]["by_task"].get(kind)
    if rule == "base":
        rule = CATALOG["breakdown"]["by_task"].get(base)
    if not rule:
        return None
    if rule["kind"] == "per_class" or rule["kind"] == "per_label":
        names = [str(c) for c in output.get("classes") or ()]
    elif rule["kind"] == "per_horizon":
        names = [f"h{h}" for h in output.get("horizons") or ()]
    else:
        return None
    if not names:
        return None
    values["breakdown_names"] = names
    values["breakdown_values"] = [round(0.6 + 0.05 * (k % 4), 4) for k in range(len(names))]
    values["breakdown_weak"] = [k == len(names) - 1 for k in range(len(names))]
    formats.update(breakdown_names="list", breakdown_values="list", breakdown_weak="list")
    direction = "min" if rule["kind"] == "per_horizon" else "max"
    return {"kind": rule["kind"], "metric": rule["metric"], "names": "breakdown_names",
            "values": "breakdown_values", "weak": "breakdown_weak", "direction": direction,
            "caption": "Each unit's value on the sealed test part."}


def _roles(case: Mapping, lane: Mapping, output: Mapping) -> tuple[dict | None, str]:
    """(class_roles@1 or None, polarity) of a case (module docstring)."""
    classes = [str(c) for c in output.get("classes") or ()]
    kind = lane["task"]
    polarity = CATALOG["polarity"]["by_task"][kind]
    if lane["view_key"] == "anomaly":
        if len(classes) != 2:
            return None, polarity[1]
        normal, anomalous = classes
        return ({"schema": "vega/class-roles@1", "roles": {normal: "neutral", anomalous: "alert"},
                 "source": {normal: "task:anomaly", anomalous: "task:anomaly"},
                 "proposed": {"neutral": normal, "rule": "task:anomaly", "alert": [anomalous],
                              "evidence": ["task:anomaly"]}}, polarity[0])
    if not classes or lane["view_key"] not in ("classes", "binary", "labels", "ordinal"):
        return None, polarity[1]
    stance = ROLES.get(case["n"])
    roles, source = {}, {}
    neutral = classes[0] if lane["view_key"] in ("classes", "binary") else None
    for k, name in enumerate(classes):
        alert = (stance == "all_but_first" and k > 0) or (stance == "positive" and k == len(classes) - 1)
        if alert:
            roles[name], source[name] = "alert", f"show:{STANCE_DATE}"
        elif name == neutral:
            rule = "structural:negative" if lane["view_key"] == "binary" else "structural:majority"
            roles[name], source[name] = "neutral", rule
        else:
            roles[name], source[name] = "plain", "vega:default"
    alert = [n for n, r in roles.items() if r == "alert"]
    rule = source[neutral] if neutral else "none"
    doc = {"schema": "vega/class-roles@1", "roles": roles, "source": source,
           "proposed": {"neutral": neutral, "rule": rule, "alert": alert,
                        "evidence": [f"show:{STANCE_DATE}"] if alert else []}}
    return doc, polarity[0] if alert else polarity[1]


def _lexicon(lane: Mapping) -> dict:
    lex = CATALOG["lexicon"]
    shape = lane["shape"] if lane["shape"] != "graph" else f"graph:{lane['level']}"
    out = {"schema": lex["schema"]}
    for key in lex["required"][shape]:
        out[key] = copy.deepcopy(lex["defaults"][key])
    out["source_of"] = {key: "vega:default" for key in lex["required"][shape]}
    return out


def _groups(spec: Mapping) -> list[dict]:
    """Three input families over the network's inputs (every input in exactly one)."""
    contract = spec.get("data_contract") or {}
    features = [f["name"] for f in contract.get("features") or ()]
    label = (spec.get("task") or {}).get("label")
    if (contract.get("sample") or {}).get("autoregressive") and label:
        features.append(label)            # a forecast reads its target's past as an input channel
    thirds = [features[k::3] for k in range(3)]
    words = (("first", "the first family of inputs"), ("second", "the second family of inputs"),
             ("third", "the third family of inputs"))
    return [{"id": wid, "label": f"{wid} family", "text": text, "members": members}
            for (wid, text), members in zip(words, thirds) if members]


def _choices(spec: Mapping) -> list[dict]:
    blocks = [b for b in spec.get("blocks") or () if b.get("id")]
    first, head = blocks[0], blocks[-1]
    out = [{"title": "Its first block", "text": "The block every input passes through first.",
            "focus": {"kind": "block", "ref": first["id"]}},
           {"title": "The loss", "text": "What the training pushed down.", "focus": {"kind": "family", "ref": "objective"}},
           {"title": "The head", "text": "The block that gives the answer its width.",
            "focus": {"kind": "block", "ref": head["id"]}}]
    feature = ((spec.get("data_contract") or {}).get("features") or [{}])[0].get("name")
    if feature:
        out.append({"title": "One input", "text": "The input the first decision read most.",
                    "focus": {"kind": "feature", "ref": feature}})
    return out


def _context(lane: Mapping, sizes: Mapping) -> dict:
    steps = CATALOG["context_steps"]["by_shape"][lane["shape"]]
    texts = {"problem": "A network answers each {row.one} from its {feature.many}.",
             "data": "{source_rows:count} {row.many}, each with its {feature.many}.",
             "windows": "Each answer reads a window of the {row.many} before it in its series.",
             "graph": "Each {node.one} reads its neighbours through the {edge.many} of the graph.",
             "cleaning": "Nothing was dropped.",
             "split": "{train_rows:count} to learn from, {valid_rows:count} to choose with, {test_rows:count} sealed."}
    return {"id": "context", "steps": [{"id": s, "text": texts[s]} for s in steps],
            "facts": [{"label": "{row.many} in the source", "value": "source_rows", "format": "count"},
                      {"label": "sealed test {row.many}", "value": "test_rows", "format": "count"}]}


def story_of(case: Mapping, spec: Mapping, lane: Mapping, metric: Mapping, walk: list,
             output: Mapping, sizes: Mapping) -> dict:
    """The story@2 of a fixture case (module docstring)."""
    primary, direction = metric["primary"], metric["direction"]
    state = verdict_of(case)
    net, base, run0 = _figure(direction, state)
    n_test = int(sizes["test"])
    values = {"primary": _estimate(net, n_test), "ci_level": 0.95,
              "source_rows": int(sizes["source"]), "train_rows": int(sizes["train"]),
              "valid_rows": int(sizes["valid"]), "test_rows": n_test,
              "run_r0_primary": run0, "run_r1_primary": net,
              f"eval_network_{primary}": _estimate(net, n_test),
              f"eval_trivial_{primary}": _estimate(base, n_test),
              "eval_diff_trivial": {"value": round(net - base if direction == "max" else base - net, 4),
                                    "ci": VERDICT_CI[state]}}
    formats = {"primary": "ci", "ci_level": "param", "source_rows": "count", "train_rows": "count",
               "valid_rows": "count", "test_rows": "count", "run_r0_primary": "metric",
               "run_r1_primary": "metric", f"eval_network_{primary}": "ci",
               f"eval_trivial_{primary}": "ci", "eval_diff_trivial": "ci"}
    roles, polarity = _roles(case, lane, output)
    breakdown = _breakdown(lane["task"], lane["base"], output, values, formats)
    refs = {k: FIXTURE_REF.format(key=k) for k in values}
    notes = {s["id"]: {"note": "Where this step lives in the code.", "link": None, "ref": None}
             for s in walk if s["id"] in ("root", "build", "forward", "run")}
    briefs = {stage: f"briefs/{stage}-brief.md" for stage in case.get("briefs") or ()}
    if briefs and "build" in notes:          # a case that ships a stage brief links it (P-LINKS)
        stage = next(iter(briefs))
        notes["build"]["link"] = {"href": f"showcase/{briefs[stage]}", "kind": "file"}
    return {
        "schema": "vega/story@2", "locale": "en",
        "task": {"kind": lane["task"], "base": lane["base"], "goal": lane["goal"],
                 "sample_shape": lane["shape"], "graph_level": lane["level"],
                 "family": lane["family"], "view_key": lane["view_key"],
                 "classes": [str(c) for c in output.get("classes") or ()] or None,
                 "outcomes": None, "primary": primary, "direction": direction},
        "lexicon": _lexicon(lane), "class_roles": roles, "polarity": polarity,
        "values": values, "formats": formats, "refs": refs,
        "headline": "A small network, read from {source_rows:count} {row.many}",
        "lead": "A page fixture: a network built and explained by the backend, nothing trained.",
        "sections": [
            _context(lane, sizes),
            {"id": "network", "lead": "Each choice below lights up the part of the network it shaped.",
             "choices": _choices(spec)},
            {"id": "experiments", "lead": "Each run changed one thing.",
             "caption": "Validation score of each run.",
             "metric": {"name": primary, "direction": direction},
             "runs": [{"id": "R1", "value": "run_r1_primary", "label": "one change to how {row.many} are read", "selected": True,
                       "reference": False},
                      {"id": "R0", "value": "run_r0_primary", "label": "the spec as engineered",
                       "selected": False, "reference": True}]},
            {"id": "results", "headline": "What the sealed test part says about the network.",
             "models": [{"name": "network", "value": f"eval_network_{primary}", "selected": True},
                        {"name": "trivial", "value": f"eval_trivial_{primary}", "selected": False}],
             "facts": [{"metric": primary, "value": f"eval_network_{primary}",
                        "note": "Read on the sealed test {row.many}."}],
             "breakdown": breakdown,
             "verdict": {"earned": state == "earned", "beaten_by_baseline": state == "beaten",
                         "strongest_baseline": "trivial", "diff": "eval_diff_trivial"},
             "limits": [{"id": i, "title": t, "text": x} for i, t, x in LIMITS]},
            {"id": "many", "monitor_note": None}],
        "input_groups": _groups(spec), "code_notes": notes,
        "report": {"status": "brief-links", "file": None, "pdf_digest": None, "aux_digest": None,
                   "roles_digest": None, "roles_source": None, "briefs": briefs or None},
        "bundle": {"manifest_digest": "sha256:" + "0" * 64, "slug": case["id"], "run": "R1"},
    }


def plan_of(case: Mapping, modality: str, template: Mapping) -> dict:
    """The authored story plan's parts page_config reads: the template pin and the motif."""
    motifs = CATALOG["motifs"]["by_modality"][modality]
    chosen, how = MOTIFS.get(case["n"], (motifs["base"], None))
    if chosen not in motifs["admissible"]:
        raise ValueError(f"case {case['id']}: motif {chosen} is not admissible for {modality}")
    return {"schema": "vega/story-plan@2", "status": "authored", "template": dict(template),
            "proposals": {"modality": modality,
                          "motif": {"modality": modality, "base": motifs["base"],
                                    "admissible": list(motifs["admissible"]), "stance": None,
                                    "fork": how}},
            "authored": {"motif": chosen}}


def shape_record(spec: Mapping) -> dict:
    """records/shape.json's manifest and recipe as the shaper would record them for these
    inputs: the nominal encoded, every numeric input scaled."""
    features = (spec.get("data_contract") or {}).get("features") or ()
    numeric = [f["name"] for f in features if f.get("role") == "numeric"]
    nominal = [f["name"] for f in features if f.get("role") == "nominal"]
    steps = []
    if nominal:
        steps.append({"id": "S1", "op": "encode", "columns": nominal, "params": {"strategy": "ordinal"}})
    steps.append({"id": "S2", "op": "scale", "columns": [], "params": {"kind": "standard", "columns": "numeric"}})
    if ((spec.get("data_contract") or {}).get("sample") or {}).get("shape") == "graph":
        steps.append({"id": "S3", "op": "graph_edges", "columns": [], "params": {}})
    ran = [{"id": s["id"], "op": s["op"], "columns_added": []} for s in steps]
    ran[[s["op"] for s in steps].index("scale")]["parameters_fitted"] = {
        "kind": "standard", "fitted": {n: {"mean": 0, "std": 1} for n in numeric}}
    return {"recipe": {"steps": steps}, "manifest": {"steps": ran}}


def spatial(case: Mapping) -> bool:
    return case["n"] in SPATIAL
