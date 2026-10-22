"""
housing — residential sales in one city, with where the home stands and what it sold for (lane
L05: the blocks lane — parallel, reshape, a row-shaped pool, a custom block — on the most common
real task, a row regression, with a coordinate pair the showcase draws as contours).

    frame = make(seed, rows)        # sale_id, area_m2, rooms, age_years, dist_center_km, lat, lon,
                                    # district, price_keur, notary_fee_keur

The signal is a hedonic price model, stated as a formula on the log of the price:

    log price = log base + 0.85 log(area / 80) + 0.03 rooms - 0.004 age
                + 0.35 exp(-distance / 3 km) + district premium + lakeside field + noise

- the city is a disc around a centre; a home stands at a radius drawn from a gamma and an
  angle drawn uniformly, and its district is the sector (or the old town, within 1.2 km)
  that position falls in;
- the lakeside field is a smooth bump of the price over the coordinates, south-east of the
  centre: what `lat` and `lon` carry beyond the distance and the district;
- the noise is Student-t with 3 degrees of freedom: a few sales far off the model (a
  family sale, a bidding war) — the heavy tail the lane's Huber loss is for;
- the distance is the road distance, the straight line times a detour factor, so it does not
  derive from the coordinates exactly.

`notary_fee_keur` is the notary's fee, a share of the price paid at the deed: it is known only
after the sale and gives the price away, so the chain drops it before the split as leakage.
Deterministic in (seed, rows).

The recipe goes one step beyond the standard pattern (D6 §4): the living area is log-normal and
the price moves with its log, so the generator plants the finding F7 (the column is
right-skewed; lanekit.planted) and the recipe answers it with a `log` transform of `area_m2`
before the scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..planted import skew_finding, with_transform
from ..recipes import standard_recipe

__all__ = ["SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "FINDINGS", "DISTRICTS", "CENTRE", "make"]

CENTRE = (46.0680, 11.1210)                # (lat, lon) of the city centre, degrees
KM_PER_DEG_LAT = 111.2
# the districts: the old town inside OLD_TOWN_KM, then seven sectors clockwise from north, each
# with its price premium on the log scale
OLD_TOWN_KM = 1.2
DISTRICTS = ("old_town", "university", "parkside", "lakeside", "harbour", "station", "west_end", "hillside")
PREMIUM = {"old_town": 0.25, "university": 0.05, "parkside": 0.12, "lakeside": 0.20, "harbour": -0.05,
           "station": -0.12, "west_end": -0.02, "hillside": 0.15}
LAKE_KM = (2.5, -2.0)                      # (east, north) of the lakeside field's centre, km

SCHEMA = (
    ColumnSpec("sale_id", "str", "key", comment="The sale's number in the land registry."),
    ColumnSpec("area_m2", "float", "numeric", "m2", "The home's living area."),
    ColumnSpec("rooms", "int", "numeric", None, "The number of rooms."),
    ColumnSpec("age_years", "float", "numeric", "years", "How old the building was at the sale."),
    ColumnSpec("dist_center_km", "float", "numeric", "km", "The road distance to the city centre."),
    ColumnSpec("lat", "float", "numeric", "deg", "The home's latitude, WGS 84."),
    ColumnSpec("lon", "float", "numeric", "deg", "The home's longitude, WGS 84."),
    ColumnSpec("district", "str", "nominal", comment="The city district the home stands in."),
    ColumnSpec("price_keur", "float", "label", "kEUR", "The price the home sold for."),
    ColumnSpec("notary_fee_keur", "float", "drop", "kEUR", "The notary's fee, paid at the deed after the sale."),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6). `harbour` is
# bikeshare's word too and `road` roads' (the road distance): each lane owns what its page says
VOCAB = frozenset({"housing", "sale", "sales", "sale_id", "area_m2", "rooms", "age_years", "dist_center_km",
                   "road", "lat", "lon", "district", "price_keur", "notary_fee_keur", "notary", "old_town",
                   "university", "parkside", "lakeside", "harbour", "west_end", "hillside"})
# D6 §4's words VOCAB leaves out: a shared template uses each in a sense of its own, so in VOCAB
# each would fail the literal gate there — `station` (a page fixture's series keyed by station,
# the word bikeshare withholds too) and `registry` (the land registry is the lane's source noun;
# the templates' registry is the block and catalog registry, 53 lines). They join VOCAB with a
# sense-scoped lint-allow.json entry per line (T04's file), as bikeshare's `station` does.
VOCAB_WITHHELD = frozenset({"station", "registry"})
COMPONENT = "market"
ENTITY_COMMENT = "One residential sale as the land registry recorded it."
OVERVIEW = """# housing

A home is sold, and the land registry records where it stands, what it is and what it sold
for. `price_keur` is the attribute the chain partitions on; every other property describes the
home as it was offered, before the price was agreed.

The domain has one class, Sale. The sale's number names it without describing it, and the
notary's fee is paid only at the deed — it is a share of the price, so the chain drops it as
leakage before anything is fitted.

Where a home stands carries the price twice: through the district it belongs to, and through
its coordinates, over which the prices rise toward the lake. The latitude and longitude are a
coordinate pair in WGS 84.
"""


def make(seed: int, rows: int):
    """`rows` sales, deterministic in (seed, rows): the position, the home, the price."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = int(rows)
    radius = np.clip(rng.gamma(2.0, 2.0, n), 0.1, 14.0)
    angle = rng.uniform(0.0, 2 * np.pi, n)                   # clockwise from north
    east, north = radius * np.sin(angle), radius * np.cos(angle)
    district = _district(radius, angle)
    lat = CENTRE[0] + north / KM_PER_DEG_LAT
    lon = CENTRE[1] + east / (KM_PER_DEG_LAT * np.cos(np.radians(CENTRE[0])))
    area = np.clip(rng.lognormal(np.log(85.0) + 0.02 * radius, 0.35), 25.0, 400.0)
    rooms = np.clip(np.round(area / 28.0 + rng.normal(0.0, 0.7, n)), 1, 10).astype(int)
    age = np.clip(rng.gamma(2.0, 22.0 - 1.2 * np.minimum(radius, 10.0)), 0.0, 180.0)
    road = radius * rng.uniform(1.15, 1.45, n)
    lake = 0.22 * np.exp(-((east - LAKE_KM[0]) ** 2 + (north - LAKE_KM[1]) ** 2) / 4.0)
    log_price = (np.log(420.0) + 0.85 * np.log(area / 80.0) + 0.03 * rooms - 0.004 * age
                 + 0.35 * np.exp(-road / 3.0) + np.array([PREMIUM[d] for d in district]) + lake
                 + 0.08 * rng.standard_t(3, n))
    price = np.exp(log_price)
    frame = pd.DataFrame({"sale_id": [f"S{i + 1:06d}" for i in range(n)], "area_m2": np.round(area, 1),
                          "rooms": rooms, "age_years": np.round(age, 1), "dist_center_km": np.round(road, 2),
                          "lat": np.round(lat, 6), "lon": np.round(lon, 6), "district": district,
                          "price_keur": np.round(price, 1),
                          "notary_fee_keur": np.round(price * rng.uniform(0.010, 0.016, n) + 0.8, 2)})
    return frame[[c.name for c in SCHEMA]]


def _district(radius: Any, angle: Any):
    """The old town within OLD_TOWN_KM, else one of seven equal sectors clockwise from north."""
    import numpy as np

    sector = np.floor(angle / (2 * np.pi / 7)).astype(int) % 7
    return np.where(radius < OLD_TOWN_KM, DISTRICTS[0], np.array(DISTRICTS[1:])[sector])


# the findings this generator plants beyond lanekit.recipes.findings_for (lanekit.planted)
FINDINGS = {"F7": skew_finding(
    "F7", "T7", "area_m2",
    "a home's living area is log-normal around 85 m2: small flats are many, a few villas reach 400 m2, "
    "so the column's right tail is long",
    "the price moves with the log of the area (0.85 log(area / 80) in the hedonic model): the log makes "
    "the effect linear and brings the villas in")}


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe (lanekit.recipes): the sale number dropped, `price_keur` selected and
    the notary's fee dropped as leakage, a random split, `district` encoded ordinal (the
    network embeds it), the quantities and the coordinates scaled on train — with `area_m2`
    log-transformed before the scale, citing F7/T7 (a living area is never zero)."""
    return with_transform(standard_recipe(SCHEMA, raw_path, shaped_dir, lane), FINDINGS["F7"], "log")
