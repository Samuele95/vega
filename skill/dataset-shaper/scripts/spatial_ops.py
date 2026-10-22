#!/usr/bin/env python3
"""
spatial_ops.py — the geographic half of the /dataset-shaper step catalog, imported by
shape.py ONLY when a recipe carries a `spatial_*` step.

Two implementations sit behind the same ops. When `pyproj` (and, for joins, `geopandas`) is
installed, coordinates are transformed by the real CRS machinery. When it is not, distances
and grids fall back to a **local equirectangular projection about the data's own centroid**,
in metres, which is accurate to a fraction of a percent over a city and degrades with extent —
and every result records which of the two ran, because a distance whose projection nobody can
name is a number without a unit. `spatial_join` has no fallback: an overlay needs geopandas,
and the step fails with an ERROR naming the missing package rather than approximating it.

Nothing here reprojects without a declared CRS: `parse_geometry` must have supplied one, and
shape.py refuses the recipe otherwise.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EARTH_R = 6371008.8


def have(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _coords(d, lat, lon):
    return (pd.to_numeric(d[lat], errors="coerce").values,
            pd.to_numeric(d[lon], errors="coerce").values)


def local_projection(la, lo):
    """Metres about the centroid: x east, y north. Deterministic, dependency-free, honest
    about being local."""
    lat0, lon0 = float(np.nanmean(la)), float(np.nanmean(lo))
    x = np.radians(lo - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(la - lat0) * EARTH_R
    return x, y, {"kind": "equirectangular about the centroid", "lat0": lat0, "lon0": lon0,
                  "units": "metres", "exact": False}


def project_xy(d, lat, lon, from_crs, to_crs, fitted=None):
    la, lo = _coords(d, lat, lon)
    if to_crs and have("pyproj"):
        from pyproj import Transformer
        tr = Transformer.from_crs(from_crs or "EPSG:4326", to_crs, always_xy=True)
        x, y = tr.transform(lo, la)
        return np.asarray(x), np.asarray(y), {"kind": "pyproj", "from": from_crs, "to": to_crs,
                                              "units": "crs units", "exact": True}
    if fitted:
        lat0, lon0 = fitted["lat0"], fitted["lon0"]
        x = np.radians(lo - lon0) * EARTH_R * math.cos(math.radians(lat0))
        y = np.radians(la - lat0) * EARTH_R
        return x, y, dict(fitted, reused=True)
    return local_projection(la, lo)


def op_spatial_reproject(ctx, step, geom):
    p = step.get("params") or {}
    to_crs = p.get("to_crs")
    lat, lon = geom["lat"], geom["lon"]
    fit = ctx.parts[ctx.fit_part]
    la, lo = _coords(fit, lat, lon)
    _, _, meta = project_xy(fit, lat, lon, ctx.crs, to_crs)
    names = (p.get("x", "x_m"), p.get("y", "y_m"))
    def apply(d):
        d = d.copy()
        x, y, _ = project_xy(d, lat, lon, ctx.crs, to_crs, fitted=meta if not meta.get("exact") else None)
        d[names[0]], d[names[1]] = x, y
        return d
    ctx.set_all(apply)
    for n in names:
        ctx.lineage.setdefault(n, []).append(step["id"])
    ctx.emit(f"# spatial_reproject -> {names} via {meta['kind']} (see the manifest for its parameters)",
             f"_f = FITTED[{repr(step['id'])}]",
             "for k in parts:",
             f"    _la = pd.to_numeric(parts[k][{repr(lat)}], errors='coerce').values",
             f"    _lo = pd.to_numeric(parts[k][{repr(lon)}], errors='coerce').values",
             "    if _f['projection'].get('kind') == 'pyproj':",
             "        from pyproj import Transformer",
             "        _tr = Transformer.from_crs(_f['projection']['from'] or 'EPSG:4326', _f['projection']['to'], always_xy=True)",
             "        _x, _y = _tr.transform(_lo, _la)",
             "    else:",
             "        _lat0, _lon0 = _f['projection']['lat0'], _f['projection']['lon0']",
             f"        _x = np.radians(_lo - _lon0) * {EARTH_R} * np.cos(np.radians(_lat0))",
             f"        _y = np.radians(_la - _lat0) * {EARTH_R}",
             f"    parts[k][{repr(names[0])}], parts[k][{repr(names[1])}] = _x, _y")
    return {"projection": meta, "columns_added": list(names), "to_crs": to_crs,
            "fit_on": None if meta.get("exact") else ctx.fit_part}   # the fallback's centroid


def op_spatial_distance(ctx, step, geom):
    p = step.get("params") or {}
    to = p.get("to")
    name = p.get("name") or "distance_m"
    lat, lon = geom["lat"], geom["lon"]
    fit = ctx.parts[ctx.fit_part]
    la, lo = _coords(fit, lat, lon)
    _, _, meta = project_xy(fit, lat, lon, ctx.crs, p.get("via_crs"))
    if not isinstance(to, dict) or "lat" not in to or "lon" not in to:
        raise ValueError("spatial_distance needs params.to = {lat, lon} (a reference point); an "
                         "external layer needs spatial_join and geopandas")
    def apply(d):
        d = d.copy()
        x, y, _ = project_xy(d, lat, lon, ctx.crs, p.get("via_crs"),
                             fitted=meta if not meta.get("exact") else None)
        ref = pd.DataFrame({lat: [to["lat"]], lon: [to["lon"]]})
        rx, ry, _ = project_xy(ref, lat, lon, ctx.crs, p.get("via_crs"),
                               fitted=meta if not meta.get("exact") else None)
        d[name] = np.hypot(x - rx[0], y - ry[0])
        return d
    ctx.set_all(apply)
    ctx.lineage.setdefault(name, []).append(step["id"])
    ctx.emit(f"# spatial_distance -> {name} (metres to {to})",
             f"_f = FITTED[{repr(step['id'])}]",
             "for k in parts:",
             f"    _la = pd.to_numeric(parts[k][{repr(lat)}], errors='coerce').values",
             f"    _lo = pd.to_numeric(parts[k][{repr(lon)}], errors='coerce').values",
             "    _lat0, _lon0 = _f['projection']['lat0'], _f['projection']['lon0']",
             f"    _x = np.radians(_lo - _lon0) * {EARTH_R} * np.cos(np.radians(_lat0))",
             f"    _y = np.radians(_la - _lat0) * {EARTH_R}",
             f"    _rx = np.radians({to['lon']} - _lon0) * {EARTH_R} * np.cos(np.radians(_lat0))",
             f"    _ry = np.radians({to['lat']} - _lat0) * {EARTH_R}",
             f"    parts[k][{repr(name)}] = np.hypot(_x - _rx, _y - _ry)")
    return {"projection": meta, "columns_added": [name], "to": to,
            "fit_on": None if meta.get("exact") else ctx.fit_part}


def op_spatial_grid(ctx, step, geom):
    p = step.get("params") or {}
    kind = p.get("kind", "square")
    size = float(p.get("size_m", 1000))
    name = p.get("name") or "cell_id"
    agg = p.get("aggregate") or {}
    lat, lon = geom["lat"], geom["lon"]
    fit = ctx.parts[ctx.fit_part]
    _, _, meta = project_xy(fit, lat, lon, ctx.crs, p.get("via_crs"))
    if kind == "h3":
        if not have("h3"):
            raise ValueError("spatial_grid kind=h3 needs the h3 package; use kind=square for a "
                             "metric grid with no extra dependency")
        import h3
        res = int(p.get("resolution", 8))
        def cell(d):
            la, lo = _coords(d, lat, lon)
            return [h3.latlng_to_cell(a, b, res) if np.isfinite(a) and np.isfinite(b) else None
                    for a, b in zip(la, lo)]
    else:
        def cell(d):
            x, y, _ = project_xy(d, lat, lon, ctx.crs, p.get("via_crs"),
                                 fitted=meta if not meta.get("exact") else None)
            return [f"{int(np.floor(a / size))}_{int(np.floor(b / size))}"
                    if np.isfinite(a) and np.isfinite(b) else None for a, b in zip(x, y)]
    added = [name]
    fitted_agg = {}
    if agg:
        f = fit.copy()
        f[name] = cell(f)
        for col, how in agg.items():
            t = f.groupby(name)[col].agg(how)
            fitted_agg[col] = {"how": how, "table": {str(k): float(v) for k, v in t.items()
                                                     if pd.notna(v)}}
            added.append(f"{name}_{col}_{how}")
    def apply(d):
        d = d.copy()
        d[name] = cell(d)
        for col, spec in fitted_agg.items():
            d[f"{name}_{col}_{spec['how']}"] = d[name].map(spec["table"]).astype(float)
        return d
    ctx.set_all(apply)
    for a in added:
        ctx.lineage.setdefault(a, []).append(step["id"])
    ctx.emit(f"# spatial_grid kind={kind} -> {added}",
             f"_f = FITTED[{repr(step['id'])}]")
    if kind == "h3":
        ctx.emit("import h3 as _h3",
                 "for k in parts:",
                 f"    _la = pd.to_numeric(parts[k][{repr(lat)}], errors='coerce').values",
                 f"    _lo = pd.to_numeric(parts[k][{repr(lon)}], errors='coerce').values",
                 f"    parts[k][{repr(name)}] = [_h3.latlng_to_cell(a, b, {int(p.get('resolution', 8))})"
                 "        if np.isfinite(a) and np.isfinite(b) else None for a, b in zip(_la, _lo)]")
    else:
        ctx.emit("for k in parts:",
                 f"    _la = pd.to_numeric(parts[k][{repr(lat)}], errors='coerce').values",
                 f"    _lo = pd.to_numeric(parts[k][{repr(lon)}], errors='coerce').values",
                 "    _lat0, _lon0 = _f['projection']['lat0'], _f['projection']['lon0']",
                 f"    _x = np.radians(_lo - _lon0) * {EARTH_R} * np.cos(np.radians(_lat0))",
                 f"    _y = np.radians(_la - _lat0) * {EARTH_R}",
                 f"    parts[k][{repr(name)}] = [f'{{int(np.floor(a / {size}))}}_{{int(np.floor(b / {size}))}}'"
                 "        if np.isfinite(a) and np.isfinite(b) else None for a, b in zip(_x, _y)]")
    for col, spec in fitted_agg.items():
        agg_col = f"{name}_{col}_{spec['how']}"
        ctx.emit("for k in parts: parts[k][%s] = parts[k][%s].map("
                 "_f['aggregates'][%s]['table']).astype(float)"
                 % (repr(agg_col), repr(name), repr(col)))
    return {"projection": meta, "kind": kind, "size_m": size, "columns_added": added,
            "aggregates": fitted_agg,
            "fit_on": (ctx.fit_part if fitted_agg or (kind != "h3" and not meta.get("exact"))
                       else None)}


# What `spatial_features` computes. The neighbour features read an index fitted on the train
# part — never the frame they are given — so a row's value depends on the train rows alone: a
# valid or test row never counts another non-train row, and a one-row query is answered as it
# would be inside any batch.
GEOMETRY_FEATURES = ("centroid_x", "centroid_y", "bearing_to")
NEIGHBOUR_FEATURES = ("n_neighbours_within", "nearest_distance")
# Per-geometry features of a polygon: parse_geometry declares points (kind latlon), on which an
# area or a perimeter is not a measurement, so they are refused rather than written as zeros.
POLYGON_FEATURES = ("area", "perimeter")


def _metric_xy(d, lat, lon, meta):
    """The rows' coordinates in the step's metric space: pyproj when the projection is exact,
    else the equirectangular projection about the fitted centroid — written exactly as the
    emitted programs write it, so the three compute the same floats."""
    la, lo = _coords(d, lat, lon)
    if meta.get("kind") == "pyproj":
        from pyproj import Transformer
        tr = Transformer.from_crs(meta["from"] or "EPSG:4326", meta["to"], always_xy=True)
        x, y = map(np.asarray, tr.transform(lo, la))
        return x, y
    x = np.radians(lo - meta["lon0"]) * EARTH_R * np.cos(np.radians(meta["lat0"]))
    y = np.radians(la - meta["lat0"]) * EARTH_R
    return x, y


def neighbour_index(x, y):
    """The fitted index: the fit part's distinct locations, each with how many rows stand on it
    (np.unique sorts them, so the index is the same whatever the row order)."""
    pts = np.column_stack([x, y])
    pts = pts[np.isfinite(pts).all(axis=1)]
    loc, counts = np.unique(pts, axis=0, return_counts=True)
    return {"points": loc.tolist(), "counts": counts.tolist(), "n_fit": int(len(pts))}


def op_spatial_features(ctx, step, geom):
    p = step.get("params") or {}
    want = list(p.get("features") or ["n_neighbours_within"])
    polygon = [f for f in want if f in POLYGON_FEATURES]
    if polygon:
        raise ValueError(f"spatial_features: {polygon} need polygon geometry, and parse_geometry "
                         "declares points (kind latlon) — an area or a perimeter of a point is "
                         "not a measurement")
    unknown = [f for f in want if f not in GEOMETRY_FEATURES + NEIGHBOUR_FEATURES]
    if unknown:
        raise ValueError(f"spatial_features: {unknown} is not a feature this executor computes "
                         f"({list(GEOMETRY_FEATURES + NEIGHBOUR_FEATURES)}); a feature asked for "
                         "and silently not written would leave the network a column short")
    radius = float(p.get("radius_m", 1000))
    to = p.get("to")
    if "bearing_to" in want and not (isinstance(to, dict) and {"lat", "lon"} <= set(to)):
        raise ValueError("spatial_features bearing_to needs params.to = {lat, lon}, the point "
                         "the bearing is taken to")
    lat, lon = geom["lat"], geom["lon"]
    fit = ctx.parts[ctx.fit_part]
    _, _, meta = project_xy(fit, lat, lon, ctx.crs, p.get("via_crs"))
    within, nearest = f"n_within_{int(radius)}m", "nearest_neighbour_m"
    index = None
    if any(f in NEIGHBOUR_FEATURES for f in want):
        index = neighbour_index(*_metric_xy(fit, lat, lon, meta))
        if not index["n_fit"]:
            raise ValueError(f"spatial_features: the {ctx.fit_part} part has no finite "
                             "coordinate to fit the neighbour index on")
        index["fit_on"] = ctx.fit_part
    added = ([n for f, n in (("centroid_x", "centroid_x_m"), ("centroid_y", "centroid_y_m"),
                             ("bearing_to", "bearing_to_deg")) if f in want] + ([within] if "n_neighbours_within" in want else [])
             + ([nearest] if "nearest_distance" in want else []))

    def apply(d):
        d = d.copy()
        x, y = _metric_xy(d, lat, lon, meta)
        P = np.column_stack([x, y])
        ok = np.isfinite(P).all(axis=1)
        Q = P[ok]
        if "centroid_x" in want:
            d["centroid_x_m"] = x
        if "centroid_y" in want:
            d["centroid_y_m"] = y
        if "bearing_to" in want:
            # the initial great-circle bearing from the row to `to`, degrees clockwise from north
            la, lo = _coords(d, lat, lon)
            p1, p2, dl = np.radians(la), np.radians(to["lat"]), np.radians(to["lon"] - lo)
            d["bearing_to_deg"] = np.degrees(np.arctan2(
                np.sin(dl) * np.cos(p2),
                np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))) % 360
        if index is None:
            return d
        from sklearn.neighbors import KDTree
        tree, w = KDTree(np.asarray(index["points"], dtype=float)), np.asarray(index["counts"])
        # A train location at distance 0 is the row's own place: not a neighbour, so a train
        # row does not count itself and a served row on a train location reads what it does.
        if "n_neighbours_within" in want:
            col = np.full(len(d), np.nan)
            if len(Q):
                ii, dd = tree.query_radius(Q, r=radius, return_distance=True)
                col[ok] = [int(w[i[r > 0]].sum()) for i, r in zip(ii, dd)]
            d[within] = col
        if "nearest_distance" in want:
            col = np.full(len(d), np.nan)
            if len(Q):
                dist, _ = tree.query(Q, k=min(2, len(w)))
                col[ok] = np.where(dist[:, 0] > 0, dist[:, 0],
                                   dist[:, 1] if dist.shape[1] > 1 else np.nan)
            d[nearest] = col
        return d
    ctx.set_all(apply)
    for a in added:
        ctx.lineage.setdefault(a, []).append(step["id"])
    ctx.emit(f"# spatial_features {want} radius={radius} m"
             + (f", the neighbour index fitted on {ctx.fit_part}" if index else ""),
             f"_f = FITTED[{repr(step['id'])}]",
             "_pj = _f['projection']",
             "_tr = (__import__('pyproj').Transformer.from_crs(_pj['from'] or 'EPSG:4326', "
             "_pj['to'], always_xy=True) if _pj.get('kind') == 'pyproj' else None)")
    if index:
        ctx.emit("from sklearn.neighbors import KDTree as _KDT",
                 "_tree = _KDT(np.asarray(_f['index']['points'], dtype=float)); "
                 "_w = np.asarray(_f['index']['counts'])")
    ctx.emit("for k in parts:",
             f"    _la = pd.to_numeric(parts[k][{repr(lat)}], errors='coerce').values; "
             f"_lo = pd.to_numeric(parts[k][{repr(lon)}], errors='coerce').values",
             "    if _tr is not None:",
             "        _x, _y = map(np.asarray, _tr.transform(_lo, _la))",
             "    else:",
             f"        _x = np.radians(_lo - _pj['lon0']) * {EARTH_R} * np.cos(np.radians(_pj['lat0'])); "
             f"_y = np.radians(_la - _pj['lat0']) * {EARTH_R}",
             "    _P = np.column_stack([_x, _y]); _ok = np.isfinite(_P).all(axis=1); _Q = _P[_ok]")
    if "centroid_x" in want:
        ctx.emit("    parts[k]['centroid_x_m'] = _x")
    if "centroid_y" in want:
        ctx.emit("    parts[k]['centroid_y_m'] = _y")
    if "bearing_to" in want:
        ctx.emit(f"    _p1, _p2, _dl = np.radians(_la), np.radians({to['lat']!r}), np.radians({to['lon']!r} - _lo)",
                 "    parts[k]['bearing_to_deg'] = np.degrees(np.arctan2(np.sin(_dl) * np.cos(_p2), "
                 "np.cos(_p1) * np.sin(_p2) - np.sin(_p1) * np.cos(_p2) * np.cos(_dl))) % 360")
    if index and "n_neighbours_within" in want:
        ctx.emit("    _col = np.full(len(parts[k]), np.nan)",
                 "    if len(_Q):",
                 f"        _ii, _dd = _tree.query_radius(_Q, r={radius!r}, return_distance=True)",
                 "        _col[_ok] = [int(_w[_i[_r > 0]].sum()) for _i, _r in zip(_ii, _dd)]",
                 f"    parts[k][{repr(within)}] = _col")
    if index and "nearest_distance" in want:
        ctx.emit("    _col2 = np.full(len(parts[k]), np.nan)",
                 "    if len(_Q):",
                 "        _d2, _ = _tree.query(_Q, k=min(2, len(_w)))",
                 "        _col2[_ok] = np.where(_d2[:, 0] > 0, _d2[:, 0], "
                 "_d2[:, 1] if _d2.shape[1] > 1 else np.nan)",
                 f"    parts[k][{repr(nearest)}] = _col2")
    return {"projection": meta, "columns_added": sorted(added), "radius_m": radius,
            "features": want, **({"to": to} if "bearing_to" in want else {}),
            **({"index": index} if index else {}),
            "fit_on": ctx.fit_part if (index or not meta.get("exact")) else None}


def op_spatial_join(ctx, step, geom):
    if not have("geopandas"):
        raise ValueError("spatial_join needs geopandas (and an external layer file); there is no "
                         "honest fallback for an overlay, so this step fails rather than "
                         "approximating it")
    import geopandas as gpd
    from shapely.geometry import Point
    p = step.get("params") or {}
    layer_path = p.get("layer")
    how, pred = p.get("how", "left"), p.get("predicate", "within")
    cols = p.get("columns") or []
    lat, lon = geom["lat"], geom["lon"]
    ext = gpd.read_file(layer_path)
    def apply(d):
        g = gpd.GeoDataFrame(d.copy(), geometry=[Point(b, a) for a, b in zip(
            pd.to_numeric(d[lat], errors="coerce"), pd.to_numeric(d[lon], errors="coerce"))],
            crs=ctx.crs or "EPSG:4326").to_crs(ext.crs)
        j = gpd.sjoin(g, ext[cols + ["geometry"]] if cols else ext, how=how, predicate=pred)
        return pd.DataFrame(j.drop(columns=["geometry", "index_right"], errors="ignore"))
    before = ctx.cols()
    ctx.set_all(apply)
    added = [c for c in ctx.cols() if c not in before]
    for a in added:
        ctx.lineage.setdefault(a, []).append(step["id"])
    import hashlib
    dig = "sha256:" + hashlib.sha256(open(layer_path, "rb").read()).hexdigest()
    ctx.emit(f"# spatial_join with {layer_path} ({dig}) — needs geopandas")
    return {"layer": layer_path, "layer_digest": dig, "how": how, "predicate": pred,
            "columns_added": added}


OPS = {"spatial_reproject": op_spatial_reproject, "spatial_distance": op_spatial_distance,
       "spatial_grid": op_spatial_grid, "spatial_features": op_spatial_features,
       "spatial_join": op_spatial_join}
