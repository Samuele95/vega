"""
The graph lanes' dataset generators (BUILD-PLAN T07, D6 §4): grid (L19), papers (L20, L23),
roads (L21) and molecules (L22).

Each keeps the lanekit.datasets protocol, is deterministic in (seed, rows), agrees with the
lanes that name it, plants the relation its lane's `graph_edges` step reads (a feeder forest,
homophilous citations, sensor coordinates, bond trees inside molecules), and builds the recipe
D6 §4 names: key_pair on the key columns (knn on the coordinates for roads), the split the
lane's shape asks for (by group on `molecule_id` at graph level). The shaper, the forge, train
and eval run on these generators in the lanes themselves (`matrix.sh lanes --lanes L19,...,L23
--to eval`); this file checks the generators alone, in seconds.

    python3 -m pytest -q vega/scripts/tests/test_lanekit_graph_generators.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lanekit import datasets  # noqa: E402
from lanekit.recipes import ColumnPlan  # noqa: E402
from lanekit.registry import LANES_FILE, load_lanes  # noqa: E402

GRAPH_LANES = ("L19", "L20", "L21", "L22", "L23")
GENERATORS = ("grid", "papers", "roads", "molecules")


@pytest.fixture(scope="module")
def lanes():
    every = load_lanes(LANES_FILE)
    return {i: every[i] for i in GRAPH_LANES}


@pytest.fixture(scope="module")
def frames(lanes):
    """Each graph lane's raw frame, at the lane's own (seed, rows)."""
    return {i: datasets.load(lane.raw["generator"]).make(int(lane.raw["seed"]), int(lane.raw["rows"]))
            for i, lane in lanes.items()}


# --------------------------------------------------------------------------- the protocol
@pytest.mark.parametrize("name", GENERATORS)
def test_the_generator_keeps_the_protocol_and_is_deterministic(name):
    gen = datasets.load(name)
    assert name not in datasets.pending()
    a, b, c = gen.make(3, 120), gen.make(3, 120), gen.make(4, 120)
    assert a.equals(b), "make is deterministic in (seed, rows)"
    assert not a.equals(c), "another seed draws other rows"
    assert list(a.columns) == [col.name for col in gen.SCHEMA]
    assert gen.VOCAB and all(isinstance(w, str) and w for w in gen.VOCAB)


@pytest.mark.parametrize("lane_id", GRAPH_LANES)
def test_every_graph_lane_agrees_with_its_generator(lanes, frames, lane_id):
    lane, frame = lanes[lane_id], frames[lane_id]
    gen = datasets.load(lane.raw["generator"])
    plan = ColumnPlan.of(gen.SCHEMA, lane.raw)          # raises when lane and SCHEMA disagree
    graph = lane.raw["graph"]
    assert set(graph["columns"]) <= set(frame.columns)
    if graph["method"] == "key_pair":
        assert set(graph["columns"]) <= set(plan.keys), "key_pair reads key columns, dropped as features"
    else:
        assert set(graph["columns"]) <= set(plan.numeric), "knn measures distances on numeric features"
    target = lane.raw["target"]
    labels = {str(v) for v in frame[target["column"]].unique()}
    if target.get("classes"):
        assert labels == set(target["classes"]), f"{lane_id}: the label takes exactly the declared classes"
    if target.get("positive") is not None:
        assert str(target["classes"][1]) == str(target["positive"]), "the positive class is listed second"
    if graph.get("graph_id"):
        assert plan.graph_id == graph["graph_id"]


# --------------------------------------------------------------------------- the vocabulary
# The lint-allow.json entries (vega/lint-allow@1) that let D6 §4's withheld words (each generator's
# VOCAB_WITHHELD) join VOCAB: each covers one sense a shared template gives one of them, line by
# line through `match`, so the word itself stays a leak everywhere else. lint-allow.json is T04's:
# the entries go there (a need of T07, runs/T07/needs/T04.lint-allow.graph-vocab.patch), and then
# the four generators fold VOCAB_WITHHELD into VOCAB.
_SHOW_PAGE = "vega-show/assets/page/*"
_TEMPLATES = "vega-forge/scripts/backends/pytorch/templates/*"
_INSPECTION = "vega-forge/scripts/backends/pytorch/templates/serving/package/inspection/*"
SENSE_ALLOW = (
    {"file": "vega-show/assets/page/tests/fixtures/predictions/*", "token": "zone", "match": r"""["']zone["']""",
     "reason": "the made-up nominal input column of the page's prediction fixtures (a test fixture no lane's "
               "page renders), not the grid lane's supply zone"},
    {"file": "vega-show/assets/page/tests/fixtures/graph-shape/*", "token": "zone", "match": r"""["'`]zone["'`]""",
     "reason": "the made-up nominal input column of the zoo cases behind the page's graph-shape fixtures (a "
               "test fixture no lane's page renders), not the grid lane's supply zone"},
    {"file": "vega-show/assets/page/tests/graph-shape.test.js", "token": "zone", "match": r"\.zone\b",
     "reason": "the graph-shape test reading the zoo fixture's made-up `zone` input, not the grid lane's "
               "supply zone"},
    {"file": "vega-show/assets/page/tests/fixtures/latent/*", "token": "zone", "match": r'"zone"',
     "reason": "the made-up nominal input column of the latent explainers' fixture networks (a test fixture "
               "no lane's page renders), not the grid lane's supply zone"},
    {"file": "vega-show/assets/page/tests/advanced.test.js", "token": "zone",
     "match": r"""["']zone(["']| · dim)|\.zone\b|token of zone is row""",
     "reason": "the made-up nominal input of the advanced explainers' attention and channel-name cases (its "
               "column, its embedding table, the sentence naming its token), not the grid lane's supply zone"},
    {"file": "vega-show/assets/page/tests/registry.test.js", "token": "zone", "match": r'"zone"',
     "reason": "a made-up input name of the registry test's feature families, not the grid lane's supply zone"},
    {"file": "vega-show/assets/page/js/net/renderers/head.js", "token": "zone", "match": r"\bzone = L\.link\b",
     "reason": "the head renderer's hover zone (a pointer distance), not the grid lane's supply zone"},
    {"file": "*/assets/*-render.js", "token": "cites",
     "match": r"\b(it|step|choice|default|rung) cites\b|\bcites the decision\b",
     "reason": "a recipe step, spec choice or campaign rung citing the decision (or the catalog row) behind "
               "it: provenance, not the papers lane's `cites` column"},
    {"file": _INSPECTION, "token": "cites", "match": r"\bwhich a payload cites\b",
     "reason": "an inspection payload citing a registered tensor's id, not the papers lane's `cites` column"},
    {"file": "vega-pack/scripts/model_card.py", "token": "paper",
     "match": r"--paper\b|\bon paper\b|\bpaper has page\b",
     "reason": "the printed page (the model card's print stylesheet and its `--paper` colour), not the papers "
               "lane's row noun"},
    {"file": "vega-show/assets/page/tests/fixtures/code-view/*", "token": "systems", "match": r"\brecord systems\b",
     "reason": "the code-view fixtures' lexicon source noun ('record systems'), not the papers lane's topic"},
    *({"file": glob, "token": "learning", "match": r"\blearning[ -]rates?\b",
       "reason": "the optimiser's learning rate, not the papers lane's `learning` topic"}
      for glob in ("vega-forge/assets/*", _TEMPLATES, "vega-train/assets/*")),
    {"file": _SHOW_PAGE, "token": "language",
     "match": r"\.language\b|\blanguage\s*(?:[:)\]]|={1,3})|\blanguage, (?:tokens|files)\b|\"language\""
              r"|\(language\b|json:language\b"
              r"|\blanguage's\b|\ba language\b|\b(?:other|another|code) language\b|\blanguage it reads\b"
              r"|\blanguage to render\b|\blanguage must be known\b|\blanguage is chosen\b|\bchange of language\b"
              r"|\b(?:document|table|page)'s language\b",
     "reason": "the code room's programming language (page.json#code.language, a v1/code answer's `language`, "
               "the highlighter registry) and the page's natural language (html[lang], the build locale), not "
               "the papers lane's `language` topic"},
    {"file": "vega-forge/scripts/backends/pytorch/templates/serving/*", "token": "language",
     "match": r'"language": "python"',
     "reason": "the programming language of the served code (a v1/code answer), not the papers lane's topic"},
    {"file": "vega-pack/scripts/*", "token": "language",
     "match": r"\bplaceholder language\b|\bwhat language the card\b",
     "reason": "the placeholder mini-language of the output readings and the natural language a model card is "
               "written in, not the papers lane's topic"},
    {"file": _SHOW_PAGE, "token": "lanes",
     "match": r"\b(?:MOTIFS|MARKS)\.lanes\b|\.lanes\b|`lanes`|\"lanes\"|\blanes: |\bconst lanes = "
              r"|\blanes = (?:\[|Array)|\bof lanes\)|\btraces, lanes, links\b"
              r"|\blanes (?:draws|at \$|is the same|is pixel|without an alert|mark\b)|\bdraws lanes\b"
              r"|\bstroke of lanes\b|\bfor the lanes\b|\breconstruction lanes\b",
     "reason": "the vega-show `lanes` motif (its id, mark and field drawing) and the page's own word for the "
               "lanes a string or a fixture serves, not the roads lane's `lanes` column"},
    {"file": _SHOW_PAGE, "token": "element",
     "match": r"\belement\(|\bconst element = |\bmotifs\.element\b|\bsvg element\b|\belement lookup\b"
              r"|\bevery element below\b|\bthe element the field draws on\b|\bEvery element of a mark\b",
     "reason": "a DOM or SVG element (the page's DOM helpers and test shim, the catalog's `motifs.element` "
               "selector), not the molecules lane's `element` column"},
    {"file": "data-lens/assets/*", "token": "element", "match": r"\bthe root element\b",
     "reason": "the render's root DOM element, not the molecules lane's `element` column"},
    {"file": "vega-eval/assets/*", "token": "element", "match": r"\bper-element\b",
     "reason": "a per-element (vector) metric reading, not the molecules lane's `element` column"},
    {"file": _TEMPLATES, "token": "element",
     "match": r"\bno element reaches\b|\bfirst element when a tuple\b|\bone element order\b|\bthe element count\b",
     "reason": "a tensor's or a tuple's element, not the molecules lane's `element` column"},
    {"file": "vega-pack/scripts/*", "token": "element", "match": r"\belement by element\b",
     "reason": "a truth vector read element by element, not the molecules lane's `element` column"},
    {"file": "vega-forge/scripts/backends/pytorch/templates/graphs.py", "token": "degree",
     "match": r"\bdegree = self\._indptr\b|\bdegree > 0\b|\bdegree\[(?:has|:, None)\]",
     "reason": "a node's out-degree in the graph template's neighbour sampler (a CSR row length), not the "
               "molecules lane's `degree` column"},
)


def _lint():
    import lint_templates
    return lint_templates


def _uncovered(words, extra_allow=()) -> list:
    """The hits of `words` in the shared templates that lint-allow.json plus extra_allow does not
    cover: the literal gate's default scan (a pack source's string literals, every other template's
    text) and each pack source's whole text too, the way the monitor, consumer and predictor suites
    pass their pack files to the gate."""
    lt = _lint()
    tokens = lt.vocab_tokens({"graph": frozenset(words)}, lt.read_json(lt.LITERALS, "the literals"))
    units = lt.default_units(lt.REPO)
    units += [(path, "text") for path, mode in units if mode == "pack"]
    rules = lt.load_allow(lt.ALLOW) + [lt.Allow(e["file"], e["token"], e["reason"], re.compile(e["match"]))
                                       for e in extra_allow]
    return [h.line_out(True) for h in lt.scan(units, tokens, lt.REPO)
            if not any(rule.covers(h) for rule in rules)]


def _d6_words(lane) -> set:
    """D6 §4's words of one lane: its SCHEMA's column names, its class names, its entity and its
    lexicon's row noun, less what can never be a lint token (a generic word of
    foreign-literals.json, a word shorter than three letters)."""
    literals = json.loads((HERE / "foreign-literals.json").read_text(encoding="utf-8"))
    generic = {w.lower() for w in literals.get("generic") or []}
    raw = lane.raw
    row = (raw.get("lexicon") or {}).get("row") or {}
    words = {c.name for c in datasets.load(raw["generator"]).SCHEMA}
    words |= {str(c) for c in (raw["target"].get("classes") or [])}
    words |= {str(raw["entity"]), str(row.get("one") or ""), str(row.get("many") or "")}
    return {w.lower() for w in words if len(w) >= 3 and w.lower() not in generic}


def test_no_vocab_word_is_a_generic_word_or_a_hit_in_the_shared_templates():
    """VOCAB feeds the literal lint over every shared template and the page leak check: a generic
    word is never a lane's, and what VOCAB holds today adds no hit to the literal gate."""
    literals = json.loads((HERE / "foreign-literals.json").read_text(encoding="utf-8"))
    generic = {w.lower() for w in literals.get("generic") or []}
    for name in GENERATORS:
        vocab = {w.lower() for w in datasets.load(name).VOCAB}
        assert not vocab & generic, f"{name}: a generic word in VOCAB: {sorted(vocab & generic)}"
        assert all(w == w.lower() for w in datasets.load(name).VOCAB), \
            f"{name}: a capitalised VOCAB word would shadow its lower-case form in the lint's tokens"
    assert "traffic" in datasets.load("roads").VOCAB, "L21's source is a traffic network (P-LEAK)"
    words = {w for name in GENERATORS for w in datasets.load(name).VOCAB}
    assert not _uncovered(words), _uncovered(words)[:5]


def test_vocab_and_the_withheld_words_hold_every_d6_word(lanes):
    """No silent narrowing: every D6 §4 word of a graph lane (column, class, entity, row noun) is in
    its generator's VOCAB or named in its VOCAB_WITHHELD, and a word is withheld only while a shared
    template uses it (a word no template uses belongs in VOCAB)."""
    for lane_id, lane in lanes.items():
        gen = datasets.load(lane.raw["generator"])
        kept = {w.lower() for w in gen.VOCAB}
        withheld = {w.lower() for w in gen.VOCAB_WITHHELD}
        assert isinstance(gen.VOCAB_WITHHELD, frozenset) and not kept & withheld, lane_id
        missing = _d6_words(lane) - kept - withheld
        assert not missing, f"{lane_id}: D6 §4 words in neither VOCAB nor VOCAB_WITHHELD: {sorted(missing)}"
    for name in GENERATORS:
        for word in datasets.load(name).VOCAB_WITHHELD:
            assert _uncovered({word}), f"{name}: {word!r} is withheld, yet no shared template uses it"


def test_every_template_use_of_a_withheld_word_is_another_sense():
    """SENSE_ALLOW covers every line of a shared template that uses a withheld word (pack sources
    read whole too), so the words can join VOCAB the day lint-allow.json holds those entries; and no
    entry covers the word itself, in a file its glob names."""
    words = {w for name in GENERATORS for w in datasets.load(name).VOCAB | datasets.load(name).VOCAB_WITHHELD}
    withheld = {w for name in GENERATORS for w in datasets.load(name).VOCAB_WITHHELD}
    tokens = {entry["token"] for entry in SENSE_ALLOW}
    assert withheld <= tokens <= words, (sorted(withheld - tokens), sorted(tokens - words))
    left = _uncovered(tokens, SENSE_ALLOW)
    assert not left, left[:5]
    lt = _lint()
    for entry in SENSE_ALLOW:
        assert entry["reason"].strip()
        rule = lt.Allow(entry["file"], entry["token"], entry["reason"], re.compile(entry["match"]))
        probe = entry["file"].replace("*", "x/y.js")
        for text in (f"the {entry['token']} of this lane", f"{entry['token']}", f"one {entry['token']}, then"):
            assert not rule.covers(lt.Hit(probe, 1, entry["token"], "vocab:graph", text)), \
                f"{entry['file']} {entry['token']}: covers the word itself ({text!r})"


# --------------------------------------------------------------------------- the planted relations
def test_grid_buses_form_one_feeder_tree_a_zone(frames):
    frame = frames["L19"]
    from lanekit.datasets import grid
    keys = dict(zip(frame["bus_key"], frame["zone"]))
    feeder = frame["feeder_to"].fillna("").astype(str)
    roots = frame[feeder == ""]
    assert sorted(roots["zone"]) == sorted(grid.ZONES), "one substation a zone"
    fed = frame[feeder != ""]
    assert fed["feeder_to"].isin(keys).all(), "every feeder_to names a bus"
    assert (fed["feeder_to"].map(keys) == fed["zone"]).all(), "a bus is fed from its own zone"
    parent = dict(zip(fed["bus_key"], fed["feeder_to"]))
    for bus in frame["bus_key"]:                        # every bus reaches its substation: no cycle
        seen = set()
        while bus in parent:
            assert bus not in seen
            seen.add(bus)
            bus = parent[bus]
    share = (frame["overloaded"] == "overloaded").mean()
    assert abs(share - grid.OVERLOADED_SHARE) < 0.02


def test_grid_overload_reads_the_feeder_neighbourhood(frames):
    """The label compares load and capacity summed over the bus and its feeder neighbours, so
    the neighbours' loading predicts it better than the bus's own does."""
    import numpy as np
    frame = frames["L19"]
    idx = {k: i for i, k in enumerate(frame["bus_key"])}
    net = (frame["load_mw"] - frame["gen_mw"]).to_numpy()
    own = net / frame["line_cap_mw"].to_numpy()
    total, cap = net.copy(), frame["line_cap_mw"].to_numpy().copy()
    for i, up in enumerate(frame["feeder_to"].fillna("").astype(str)):
        if up:
            j = idx[up]
            total[i] += net[j]; total[j] += net[i]
            cap[i] += cap[j]; cap[j] += frame["line_cap_mw"].iat[i]
    y = (frame["overloaded"] == "overloaded").to_numpy(float)
    corr = lambda x: abs(np.corrcoef(x, y)[0, 1])  # noqa: E731
    assert corr(total / cap) > corr(own) + 0.2


def test_papers_cite_earlier_papers_mostly_within_their_topic(frames):
    from lanekit.datasets import papers
    for lane_id in ("L20", "L23"):
        frame = frames[lane_id]
        cites = frame["cites"].fillna("").astype(str)
        pos = {k: i for i, k in enumerate(frame["paper_key"])}
        citing = frame[cites != ""]
        assert (cites.iloc[:papers.FIRST_CITING] == "").all() and len(citing) == len(frame) - papers.FIRST_CITING
        assert all(pos[c] < pos[k] for k, c in zip(citing["paper_key"], citing["cites"])), "a paper cites the past"
        topic = dict(zip(frame["paper_key"], frame["topic"]))
        homophily = (citing["cites"].map(topic) == citing["topic"]).mean()
        assert abs(homophily - papers.HOMOPHILY) < 0.05
        shares = frame[list(papers.TERMS)].sum(axis=1)
        assert ((shares - 1.0).abs() < 1e-3).all(), "a paper's term shares sum to 1"
    assert not frames["L20"].equals(frames["L23"]), "L23 is papers at another seed"


def test_roads_sensors_sit_in_the_region_with_a_speed_to_predict(frames):
    from lanekit.datasets import roads
    frame = frames["L21"]
    for axis in ("x_km", "y_km"):
        assert frame[axis].between(0.0, roads.EXTENT_KM).all()
    assert frame[["x_km", "y_km"]].drop_duplicates().shape[0] == len(frame), "no two sensors share a place"
    pairs = set(zip(frame["lanes"], frame["speed_limit"]))
    assert pairs <= set(roads.ROAD_CLASSES)
    assert (frame["mean_speed"] > 0).all() and frame["mean_speed"].std() > 5
    assert frame["sensor_id"].is_unique


def test_molecules_are_bond_trees_with_one_label_each(frames, lanes):
    from lanekit.datasets import molecules
    frame = frames["L22"]
    lo, hi = molecules.ATOMS_PER_MOLECULE
    sizes = frame.groupby("molecule_id").size()
    assert len(sizes) == int(lanes["L22"].raw["rows"]), "at graph level `rows` counts molecules"
    assert sizes.between(lo, hi).all()
    of = dict(zip(frame["atom_key"], frame["molecule_id"]))
    bonded = frame["bonded_to"].fillna("").astype(str)
    assert (bonded == "").sum() == len(sizes), "one first atom a molecule, bonded to nothing"
    linked = frame[bonded != ""]
    assert (linked["bonded_to"].map(of) == linked["molecule_id"]).all(), "no bond leaves its molecule"
    bonds = linked["bonded_to"].value_counts().reindex(frame["atom_key"], fill_value=0).to_numpy() + (bonded != "")
    assert (bonds == frame["degree"].to_numpy()).all(), "degree is the atom's bond count in the tree"
    valence = frame["element"].map(molecules.VALENCE).to_numpy()
    assert (frame["degree"].to_numpy() <= valence).all(), "no atom takes more bonds than its valence"
    assert set(frame["element"]) <= set(molecules.ELEMENTS)
    per = frame.groupby("molecule_id")["soluble"]
    assert (per.nunique() == 1).all() and 0.4 < per.first().mean() < 0.6


# --------------------------------------------------------------------------- the recipes
@pytest.mark.parametrize("lane_id, split", [("L19", {"kind": "stratified", "by": "overloaded"}),
                                            ("L20", {"kind": "stratified", "by": "topic"}),
                                            ("L21", {"kind": "random"}),
                                            ("L22", {"kind": "group", "by": "molecule_id"}),
                                            ("L23", {"kind": "stratified", "by": "topic"})])
def test_the_recipe_splits_as_the_shape_asks_and_records_the_relation(lanes, frames, tmp_path, lane_id, split):
    lane, frame = lanes[lane_id], frames[lane_id]
    gen = datasets.load(lane.raw["generator"])
    raw = tmp_path / f"{lane.raw['generator']}.csv"
    frame.to_csv(raw, index=False)
    recipe = gen.RECIPE(raw, tmp_path / "shaped", lane)
    steps = {s["op"]: s for s in recipe["steps"]}
    assert recipe["schema"] == "dataset-shaper/recipe@1" and recipe["input"]["rows"] == len(frame)
    got = steps["split"]["params"]
    assert {k: got.get(k) for k in split} == split
    edges = steps["graph_edges"]
    graph = lane.raw["graph"]
    assert edges["source"].startswith("analysis:F") and edges["params"]["method"] == graph["method"]
    assert edges["params"]["graph_id"] == graph.get("graph_id")
    if graph["method"] == "key_pair":
        assert edges["params"]["columns"] == graph["columns"]
    else:
        assert edges["params"]["features"] == graph["columns"] and edges["params"]["k"] == graph["k"]
        assert edges["params"]["weight"] == "distance"
    keys = [c.name for c in gen.SCHEMA if c.role == "key"]
    assert set(keys) <= set(steps["drop_identity"]["columns"]), "the keys never reach the network"
    nominal = [c.name for c in gen.SCHEMA if c.role == "nominal"]
    if nominal:
        assert set(nominal) <= set(steps["encode"]["columns"])
