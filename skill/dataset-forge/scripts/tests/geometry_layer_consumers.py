"""
geometry_layer_consumers.py — the X2 test stages that read the layer from the outside: the
/dataset-shaper orders recipe on the layered model, the data-lens sensors layer (the other
geometry layer the lane ships), and a generated file with every column kind. Imported by
`test_geometry_layer.py`; not a test.
"""
from __future__ import annotations

import json

from geometry_layer_cases import messy_frame, messy_layer
from geometry_layer_testkit import (BOOTSTRAP, GEOMETRY, APPLY, RECIPE, SENSORS, SHAPE, SMOKE,
                                    apply_layer, check, layer_doc, run, tail, validate_model,
                                    errors)


def _shape(work, model, recipe, name):
    """shape.py on `recipe` against `model`; (exit, log, manifest or None)."""
    out = work / name
    rc, log = run([SHAPE, "--recipe", recipe, "--model", model, "--out-dir", out, "--format", "csv"])
    man = out / "manifest.json"
    return rc, log, (json.loads(man.read_text(encoding="utf-8")) if man.is_file() else None)


def _removed(man, upto):
    steps = {s["id"]: s for s in man.get("steps") or []}
    return {c for sid in upto for c in (steps.get(sid) or {}).get("columns_removed") or []}


def stage_shaper(work, model):
    """The consumer the layer's `cycles` slot exists for: /dataset-shaper's orders recipe cites
    geometry:typing/…, geometry:cycles/<id> (naming an orientation by id), geometry:derivations
    and geometry:partitions/late; every citation resolves on the layer written here, the report's
    orientation drops exactly the kept rule heads, and the alternative one the analyst pre-wrote
    (unit-price-derived) drops unit_price in total's place — the completed `basis`/`rules` keys."""
    if not (SHAPE.is_file() and RECIPE.is_file()):
        print("  skip  the dataset-shaper orders recipe is not in the tree")
        return
    rc, log = run([SHAPE, "--recipe", RECIPE, "--model", model, "--check-only"])
    unresolved = [ln for ln in log.splitlines() if "geometry:" in ln and ("WARN" in ln or "ERROR" in ln)]
    check("shape.py --check-only resolves every geometry: source of the orders recipe",
          rc == 0 and not unresolved, tail(log, 4))
    doc = layer_doc(model.read_text(encoding="utf-8"))
    cand = next(c for c in doc["partitions"]["candidates"] if c["label"] == "late")
    cols = set(doc["source"]["column_names"])
    rc, log, man = _shape(work, model, RECIPE, "shaped")
    if not check("shape.py executes the orders recipe on the layered model", rc == 0 and man, tail(log, 4)):
        return
    check("the report's orientation drops exactly the kept rule heads",
          _removed(man, ("S5",)) == {"city", "region", "total", "subtotal"}, sorted(_removed(man, ("S5",))))
    check("the leakage set is dropped and the candidate's features remain beside the label",
          _removed(man, ("S7",)) == set(cand["dropped_for_leakage"]) and
          cols - _removed(man, ("S1", "S2", "S3", "S4", "S5", "S6", "S7")) == set(cand["features"]) | {"late"})
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    step = next(s for s in recipe["steps"] if (s.get("params") or {}).get("cycle_id") == "price-qty-total")
    step["params"]["orientation_id"] = "unit-price-derived"
    alt = work / "orders.alt-orientation.recipe.json"
    alt.write_text(json.dumps(recipe), encoding="utf-8")
    rc, log, man = _shape(work, model, alt, "shaped-alt")
    check("an alternative orientation named by id re-orients the shaper (unit_price derived, total kept)",
          rc == 0 and man and _removed(man, ("S5",)) == {"city", "region", "unit_price", "subtotal"},
          sorted(_removed(man, ("S5",))) if man else tail(log, 4))


def stage_sensors(work):
    """The other geometry layer the lane ships: hand-written, no from_run, no explore payload,
    no cycle, two candidates, a user-chosen partition — the explorer's other branches."""
    if not SENSORS.is_file():
        print("  skip  the data-lens sensors geometry layer is not in the tree")
        return
    rc, log = run([APPLY, "--data", SENSORS, "--check-only"])
    check("the data-lens fixture's hand-written geometry layer validates", rc == 0, errors(log))
    base, out = work / "sensors.base.html", work / "sensors.domain.html"
    rc, log = run([BOOTSTRAP, SENSORS.with_name("sensors.csv"), "--out", base])
    if not check("a sensors base model is written", rc == 0, tail(log)):
        return
    rc, log = apply_layer(base, SENSORS, out)
    check("the sensors layer is written", rc == 0, tail(log))
    rc, log = validate_model(out)
    check("validate_model.py exits 0 on the sensors model", rc == 0, tail(log))
    rc, log = run([SMOKE, out, "--strict"])
    check("smoke_geometry.py --strict passes on a layer without explore or cycles", rc == 0 and
          "no cycle to re-orient" in log and "A and C empty" in log, tail(log, 6))


def stage_messy(work):
    """A generated file with an identity, a constant, free text, missing values, a date, a
    two-member cycle, a confirmed symbolic count and an abstained partition."""
    rd = work / "messy"
    rd.mkdir(parents=True, exist_ok=True)
    messy_frame().to_csv(rd / "messy.csv", index=False)
    (rd / "geometry-layer.json").write_text(json.dumps(messy_layer()), encoding="utf-8")
    rc, log = run([GEOMETRY, rd / "messy.csv", "--out", rd / "geometry.json"])
    rc2, log2 = run([BOOTSTRAP, rd / "messy.csv", "--out", rd / "base.html"])
    if not check("the messy run and its base model are written", rc == 0 and rc2 == 0, tail(log + log2)):
        return
    out = rd / "messy.domain.html"
    rc, log = apply_layer(rd / "base.html", rd / "geometry-layer.json", out)
    check("the messy layer is written", rc == 0, tail(log, 6))
    rc, log = validate_model(out)
    check("validate_model.py exits 0 on the messy model", rc == 0, tail(log))
    rc, log = run([SMOKE, out, "--strict"])
    check("smoke_geometry.py --strict passes on the messy model", rc == 0, tail(log, 6))
