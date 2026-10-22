"""
landcover — land parcels read from a ten-band satellite image, with the cover a field survey
recorded (lane L02, the domain-adaptation lane: CIC's shape — row, multiclass, numeric inputs
only — on another domain, through the real data sub-chain).

    frame = make(seed, rows)        # parcel_id, tile, b01..b10, elevation_m, slope_deg, aspect_deg, cover

The signal is the physics of a multispectral image, stated as a formula:

- each cover class has a centroid spectrum over the ten bands (water dark beyond the red,
  vegetation bright in the near infrared and dark in the red, bare soil bright in the
  shortwave infrared, built-up surfaces flat), and a class-specific terrain: water and fields
  lie low and flat, forest on the slopes, bare rock high;
- within a class the spectrum moves with the vegetation's vigour, and a field may be partly
  fallow and a built-up parcel partly green (a mixture with another centroid);
- the terrain lights the parcel: a slope facing the sun reads brighter by a factor of the
  slope and the cosine of the aspect away from the sun's azimuth, so slope and aspect carry
  what the network needs to undo the illumination;
- conifers at altitude read darker in the near infrared, so elevation carries signal twice.

Deterministic in (seed, rows): one numpy Generator seeded once, drawn in a fixed order.
`tile` is the image tile a parcel was read from — an identifier with repeats, dropped with the
parcel id before anything is fitted. Every column is a raw measurement: no column derives
another by construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..recipes import standard_recipe

__all__ = ["SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "CLASSES", "BANDS", "make"]

CLASSES = ("water", "forest", "grassland", "cropland", "urban", "bare")
PRIORS = (0.10, 0.22, 0.20, 0.24, 0.13, 0.11)
# (column, wavelength nm, what the band reads)
BANDS = (("b01", 443, "coastal aerosol"), ("b02", 490, "blue"), ("b03", 560, "green"),
         ("b04", 665, "red"), ("b05", 705, "first red edge"), ("b06", 740, "second red edge"),
         ("b07", 783, "third red edge"), ("b08", 842, "near infrared"),
         ("b09", 1610, "first shortwave infrared"), ("b10", 2190, "second shortwave infrared"))
# the centroid surface reflectance of each class, band by band
CENTROIDS = {
    "water":     (0.080, 0.070, 0.060, 0.040, 0.030, 0.022, 0.020, 0.018, 0.010, 0.008),
    "forest":    (0.030, 0.032, 0.052, 0.030, 0.090, 0.240, 0.300, 0.320, 0.150, 0.070),
    "grassland": (0.040, 0.050, 0.085, 0.060, 0.130, 0.280, 0.330, 0.350, 0.240, 0.130),
    "cropland":  (0.050, 0.060, 0.090, 0.080, 0.140, 0.240, 0.280, 0.300, 0.260, 0.160),
    "urban":     (0.100, 0.110, 0.120, 0.130, 0.150, 0.170, 0.180, 0.190, 0.220, 0.190),
    "bare":      (0.120, 0.140, 0.180, 0.220, 0.250, 0.270, 0.280, 0.290, 0.340, 0.300)}
# how the near-infrared plateau and the red respond to one unit of vigour (vegetation only)
VIGOUR = (0.000, 0.000, 0.004, -0.010, 0.010, 0.030, 0.040, 0.045, 0.010, 0.000)
# (elevation mean m, sd m, slope gamma shape, slope gamma scale) per class
TERRAIN = {"water": (150, 60, 1.0, 1.0), "forest": (900, 300, 3.0, 6.0), "grassland": (600, 250, 2.0, 4.0),
           "cropland": (300, 100, 1.5, 2.0), "urban": (250, 80, 1.5, 2.0), "bare": (1500, 400, 3.0, 8.0)}
# a class a parcel may be partly made of: (the other class, Beta a, Beta b of its share)
MIXTURE = {"cropland": ("grassland", 1.5, 2.5), "urban": ("grassland", 1.5, 3.0), "grassland": ("bare", 1.2, 5.0),
           "forest": ("grassland", 1.2, 4.0)}
SUN_AZIMUTH_DEG = 150.0
N_TILES = 24

SCHEMA = (
    ColumnSpec("parcel_id", "str", "key", comment="The parcel's cadastral identifier."),
    ColumnSpec("tile", "str", "key", comment="The image tile the parcel was read from."),
    *(ColumnSpec(name, "float", "numeric", "reflectance",
                 f"Surface reflectance in the {what} band, {nm} nm.") for name, nm, what in BANDS),
    ColumnSpec("elevation_m", "float", "numeric", "m", "The parcel's mean elevation above sea level."),
    ColumnSpec("slope_deg", "float", "numeric", "deg", "The parcel's mean terrain slope."),
    ColumnSpec("aspect_deg", "float", "numeric", "deg",
               "The compass direction the slope faces, clockwise from north."),
    ColumnSpec("cover", "str", "label", comment="The land cover a field survey recorded for the parcel."),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6)
VOCAB = frozenset({"landcover", "parcel_id", *(name for name, _, _ in BANDS), "elevation_m", "slope_deg",
                   "aspect_deg", "grassland", "cropland", "urban", "reflectance", "infrared", "satellite",
                   "survey"})
# D6 §4's words VOCAB leaves out. A shared template uses these in a sense of its own, so in VOCAB
# each would fail the literal gate there: `parcel`, `parcels` (the page lexicon's own example
# noun, lexicon.js and its tests, 22 lines), `band`, `bands` (the lane's feature noun; the model
# card's --band colour and the network renderer's bands, 53 lines), `cover` (the verb), `forest`
# (an isolation forest), `bare` (a bare string), `tile` (numpy's tile). Each joins VOCAB with a
# sense-scoped lint-allow.json entry per line (T04's file). `water` hits no template but is the
# graph lane L22's word (molecules: "water-soluble"): in VOCAB it would be a cross-lane leak on
# that page (D6 §10.6), so it stays out until the two lanes share it through the `generic` words of
# foreign-literals.json (T04's file).
VOCAB_WITHHELD = frozenset({"parcel", "parcels", "band", "bands", "cover", "forest", "bare", "tile", "water"})
COMPONENT = "survey"
ENTITY_COMMENT = "One land parcel as the satellite image and the field survey recorded it."
OVERVIEW = """# landcover

A parcel of land is read by a satellite in ten spectral bands, and a field survey records
what covers it: water, forest, grassland, cropland, urban or bare ground. `cover` is the
attribute the chain partitions on; every other property is what the image and the terrain
model say about the parcel before anyone walks it.

The domain has one class, Parcel. The image tile a parcel was read from and the parcel's own
identifier name it without describing it, so the chain drops them before anything is fitted.

The reflectances are the heart of the problem: vegetation is bright in the near infrared and
dark in the red, water is dark beyond the red, bare ground is bright in the shortwave
infrared. The terrain matters twice: a slope facing the sun reads brighter, and the cover
itself depends on altitude.
"""


def make(seed: int, rows: int):
    """`rows` parcels, deterministic in (seed, rows): a class drawn from the priors, its
    terrain, its spectrum (centroid, vigour, mixture, illumination, sensor noise)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = int(rows)
    labels = rng.choice(len(CLASSES), size=n, p=PRIORS)
    names = np.array(CLASSES)[labels]
    spectra = np.array([CENTROIDS[c] for c in CLASSES])[labels]          # (n, 10)
    elevation, slope = _terrain(rng, names, n)
    aspect = rng.uniform(0.0, 360.0, n)
    vigour = rng.normal(0.0, 1.5, n) * np.isin(names, ("forest", "grassland", "cropland"))
    spectra = spectra + np.outer(vigour, VIGOUR)
    spectra = _mixed(rng, spectra, names, n)
    conifer = (names == "forest") * np.clip((elevation - 900.0) * 8e-5, -0.15, 0.25)
    spectra[:, 5:8] *= (1.0 - conifer)[:, None]
    light = 1.0 + 0.3 * (slope / 45.0) * np.cos(np.radians(aspect - SUN_AZIMUTH_DEG))
    spectra = spectra * light[:, None] * rng.normal(1.0, 0.12, (n, 1))
    spectra = np.clip(spectra + rng.normal(0.0, 0.02, spectra.shape), 0.001, 0.95)
    frame = pd.DataFrame({"parcel_id": [f"P{i:06d}" for i in range(n)],
                          "tile": [f"T{int(t):02d}" for t in rng.integers(0, N_TILES, n)]})
    for j, (name, _, _) in enumerate(BANDS):
        frame[name] = np.round(spectra[:, j], 4)
    frame["elevation_m"] = np.round(elevation, 1)
    frame["slope_deg"] = np.round(slope, 2)
    frame["aspect_deg"] = np.round(aspect, 1)
    frame["cover"] = names
    return frame[[c.name for c in SCHEMA]]


def _terrain(rng: Any, names: Any, n: int):
    """(elevation m, slope deg) from each class's terrain: a normal elevation clipped at the sea,
    a gamma slope clipped at 60 degrees."""
    import numpy as np

    mean = np.array([TERRAIN[c][0] for c in names], dtype=float)
    sd = np.array([TERRAIN[c][1] for c in names], dtype=float)
    shape = np.array([TERRAIN[c][2] for c in names], dtype=float)
    scale = np.array([TERRAIN[c][3] for c in names], dtype=float)
    elevation = np.clip(rng.normal(mean, sd), 2.0, 3200.0)
    slope = np.clip(rng.gamma(shape, scale), 0.0, 60.0)
    return elevation, slope


def _mixed(rng: Any, spectra: Any, names: Any, n: int):
    """A cropland parcel partly fallow, an urban one partly green, a grassland partly rock: the
    spectrum moves toward the other class's centroid by a Beta-drawn share."""
    import numpy as np

    out = spectra.copy()
    for cls, (other, a, b) in MIXTURE.items():
        rows = names == cls
        share = rng.beta(a, b, n)[rows]                # drawn for every row: a fixed order of draws
        out[rows] = (1 - share)[:, None] * out[rows] + share[:, None] * np.array(CENTROIDS[other])
    return out


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe (lanekit.recipes): the two identifiers dropped, `cover` selected, a
    stratified split, every band and the terrain scaled on train.

    `aspect_deg` stays one scaled column, not the sin/cos pair D6 §4 names: the shaper's catalog
    has no op for it. `orient_cycle` picks the orientation of a derivation cycle
    (step-catalog.md phase 2), `datetime_expand`'s `cyclic` encodes the parts of a datetime, and
    a `custom` step needs a `user:` stance the lane does not hold and is marked unverified in the
    layer. The illumination the aspect carries is still learnable: the network reads it next to
    the slope."""
    return standard_recipe(SCHEMA, raw_path, shaped_dir, lane)
