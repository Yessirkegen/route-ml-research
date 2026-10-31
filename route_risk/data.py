"""Data loading (STATS19) and count-based, exposure-normalised dataset construction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

from .config import BBOX, YEAR, SEVERITY_WEIGHT, TRAFFIC_PROXY, TRAFFIC_PROXY_DEFAULT
from .geo import nearest_edge_idx
from .features import _season

if TYPE_CHECKING:
    from .weather import WeatherClient


# ---------------------------------------------------------------------------
# STATS19 loader
# ---------------------------------------------------------------------------

def _pick(cols, options):
    for name in options:
        if name in cols:
            return name
    raise KeyError(f"None of {options} found in CSV columns: {list(cols)[:20]} ...")


def load_accidents_csv(csv_path: str,
                       bbox=BBOX,
                       years: list[int] | int | None = None) -> pd.DataFrame:
    """Load one or more DfT STATS19 Collisions CSVs, keeping enriched columns.

    *csv_path* may be a single file or a glob pattern / comma-separated list.
    *years* filters to specific year(s); ``None`` keeps all years found.

    Returns a DataFrame filtered to *bbox* with at least:
    ``longitude, latitude, dt_utc, collision_severity, ...``
    """
    import glob as _glob

    paths = []
    for part in csv_path.split(","):
        part = part.strip()
        found = _glob.glob(part)
        paths.extend(found if found else [part])

    frames = []
    for p in paths:
        frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]

    lon_col = _pick(df.columns, ["longitude", "lon", "lng"])
    lat_col = _pick(df.columns, ["latitude", "lat"])
    date_col = _pick(df.columns, ["date", "accident_date"])

    time_col = None
    for candidate in ("time", "accident_time"):
        if candidate in df.columns:
            time_col = candidate
            break

    df = df[df[lon_col].notna() & df[lat_col].notna()]

    def parse_dt(row):
        d = str(row[date_col]).strip()
        t = str(row[time_col]).strip() if (time_col and pd.notna(row.get(time_col))) else "00:00"
        for fmt in ("%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"):
            try:
                return datetime.strptime(f"{d} {t}", fmt).replace(tzinfo=timezone.utc)
            except Exception:
                pass
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(d, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                pass
        return pd.NaT

    df["dt_utc"] = df.apply(parse_dt, axis=1)
    df = df[df["dt_utc"].notna()]
    df["year"] = df["dt_utc"].dt.year

    if years is not None:
        if isinstance(years, int):
            years = [years]
        df = df[df["year"].isin(years)]

    north, south, east, west = bbox
    df = df[(df[lat_col] <= north) & (df[lat_col] >= south)
            & (df[lon_col] <= east) & (df[lon_col] >= west)]

    keep = {lon_col: "longitude", lat_col: "latitude"}
    extra_cols = [
        "collision_severity", "speed_limit", "light_conditions",
        "road_surface_conditions", "weather_conditions",
        "number_of_vehicles", "number_of_casualties",
        "junction_detail", "day_of_week",
    ]
    for c in extra_cols:
        if c in df.columns:
            keep[c] = c

    out = df.rename(columns=keep)
    out_cols = list(keep.values()) + ["dt_utc", "year"]
    out_cols = [c for c in out_cols if c in out.columns]
    return out[out_cols].copy()


# ---------------------------------------------------------------------------
# DfT AADF (real traffic volumes)
# ---------------------------------------------------------------------------

def load_aadf(csv_path: str, bbox=BBOX, year=YEAR) -> pd.DataFrame:
    """Load DfT AADF traffic counts, filtered to *bbox* and *year*.

    Returns a DataFrame with ``latitude, longitude, all_motor_vehicles``.
    Download from https://storage.googleapis.com/dft-statistics/road-traffic/
    downloads/data-gov-uk/dft_traffic_counts_aadf.zip
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["year"] == year]
    north, south, east, west = bbox
    df = df[(df["latitude"] >= south) & (df["latitude"] <= north)
            & (df["longitude"] >= west) & (df["longitude"] <= east)]
    df = df[["latitude", "longitude", "all_motor_vehicles"]].dropna()
    print(f"  AADF: {len(df)} count points in bbox for {year}")
    return df


def _snap_aadf_to_edges(df_aadf: pd.DataFrame,
                        gdf_edges: gpd.GeoDataFrame) -> dict[int, float]:
    """Snap AADF count points to nearest edges, return {edge_idx: AADT}."""
    aadf_gdf = gpd.GeoDataFrame(
        df_aadf,
        geometry=gpd.points_from_xy(df_aadf["longitude"], df_aadf["latitude"]),
        crs="EPSG:4326",
    ).to_crs(gdf_edges.crs)

    edge_aadf: dict[int, list[float]] = {}
    for _, row in aadf_gdf.iterrows():
        idx = nearest_edge_idx(row.geometry, gdf_edges)
        edge_aadf.setdefault(idx, []).append(float(row["all_motor_vehicles"]))

    return {idx: np.mean(vals) for idx, vals in edge_aadf.items()}


def _idw_interpolate_aadf(direct_aadf: dict[int, float],
                          gdf_edges: gpd.GeoDataFrame,
                          k: int = 5, power: float = 2.0) -> dict[int, float]:
    """For edges without a direct AADF match, interpolate from the k nearest
    matched edges of the same highway class using inverse-distance weighting."""
    if not direct_aadf:
        return {}

    def _hw(idx):
        hw = gdf_edges.iloc[idx].get("highway", "unclassified")
        return hw[0] if isinstance(hw, list) and hw else hw

    matched_by_hw: dict[str, list[tuple[int, float, float, float]]] = {}
    for idx, aadt in direct_aadf.items():
        geom = gdf_edges.iloc[idx]["geometry"]
        cx, cy = geom.centroid.x, geom.centroid.y
        hw = _hw(idx)
        matched_by_hw.setdefault(hw, []).append((idx, cx, cy, aadt))

    result = dict(direct_aadf)
    for idx in range(len(gdf_edges)):
        if idx in result:
            continue
        hw = _hw(idx)
        pool = matched_by_hw.get(hw, [])
        if not pool:
            all_pools = [v for vals in matched_by_hw.values() for v in vals]
            pool = all_pools
        if not pool:
            continue
        geom = gdf_edges.iloc[idx]["geometry"]
        cx, cy = geom.centroid.x, geom.centroid.y
        dists = [(np.hypot(cx - px, cy - py), aadt) for _, px, py, aadt in pool]
        dists.sort(key=lambda x: x[0])
        top = dists[:k]
        w_sum = 0.0
        v_sum = 0.0
        for d, aadt in top:
            w = 1.0 / max(d, 1e-9) ** power
            w_sum += w
            v_sum += w * aadt
        if w_sum > 0:
            result[idx] = v_sum / w_sum
    return result


# ---------------------------------------------------------------------------
# Count-based, exposure-normalised dataset
# ---------------------------------------------------------------------------

def _snap_accidents_to_edges(df_acc: pd.DataFrame,
                             gdf_edges: gpd.GeoDataFrame) -> dict[str, dict]:
    """Snap accidents to nearest edges and return per-edge aggregates.

    Returns a dict with keys:
      crash_count, severity_score, night, wet, junction, severities
    Each is a ``dict[int, ...]`` keyed by edge positional index.
    """
    acc_gdf = gpd.GeoDataFrame(
        df_acc,
        geometry=gpd.points_from_xy(df_acc["longitude"], df_acc["latitude"]),
        crs="EPSG:4326",
    ).to_crs(gdf_edges.crs)

    edge_crash_count: dict[int, int] = {}
    edge_severity_score: dict[int, float] = {}
    edge_night: dict[int, int] = {}
    edge_wet: dict[int, int] = {}
    edge_junction: dict[int, int] = {}
    edge_severities: dict[int, list[int]] = {}

    has_light = "light_conditions" in df_acc.columns
    has_surface = "road_surface_conditions" in df_acc.columns
    has_junction = "junction_detail" in df_acc.columns

    for _, row in tqdm(acc_gdf.iterrows(), total=len(acc_gdf), desc="Snapping accidents"):
        pt = row.geometry
        idx = nearest_edge_idx(pt, gdf_edges)
        edge_crash_count[idx] = edge_crash_count.get(idx, 0) + 1

        sev = row.get("collision_severity", 3)
        try:
            sev = int(sev)
        except (ValueError, TypeError):
            sev = 3
        w = SEVERITY_WEIGHT.get(sev, 1.0)
        edge_severity_score[idx] = edge_severity_score.get(idx, 0.0) + w
        edge_severities.setdefault(idx, []).append(sev)

        if has_light:
            try:
                lc = int(row["light_conditions"])
                if lc in (4, 5, 6, 7):
                    edge_night[idx] = edge_night.get(idx, 0) + 1
            except (ValueError, TypeError):
                pass

        if has_surface:
            try:
                sc = int(row["road_surface_conditions"])
                if sc >= 2:
                    edge_wet[idx] = edge_wet.get(idx, 0) + 1
            except (ValueError, TypeError):
                pass

        if has_junction:
            try:
                jd_val = int(row["junction_detail"])
                if jd_val > 0:
                    edge_junction[idx] = edge_junction.get(idx, 0) + 1
            except (ValueError, TypeError):
                pass

    return {
        "crash_count": edge_crash_count,
        "severity_score": edge_severity_score,
        "night": edge_night,
        "wet": edge_wet,
        "junction": edge_junction,
        "severities": edge_severities,
    }


def _compute_graph_features(G, gdf_edges, edge_crash_count: dict) -> dict[int, dict]:
    """Compute graph-topology features for each edge."""
    import networkx as nx

    print("  Computing betweenness centrality...", flush=True)
    edge_bc = nx.edge_betweenness_centrality(G, weight="travel_time", k=min(300, len(G.nodes)))

    bc_by_uv: dict[tuple, float] = {}
    for (u, v, k_), val in edge_bc.items():
        bc_by_uv[(u, v)] = max(bc_by_uv.get((u, v), 0.0), val)

    hw_col = gdf_edges["highway"].apply(lambda h: h[0] if isinstance(h, list) and h else h)

    uv_to_idx: dict[tuple, int] = {}
    for i, row in enumerate(gdf_edges.itertuples(index=False)):
        uv_to_idx[(row.u, row.v)] = i

    result: dict[int, dict] = {}
    for i, row in enumerate(gdf_edges.itertuples(index=False)):
        u, v = row.u, row.v
        bc = bc_by_uv.get((u, v), bc_by_uv.get((v, u), 0.0))

        neighbor_crashes = []
        hw_i = hw_col.iloc[i]
        for nbr in set(G.neighbors(u)) | set(G.neighbors(v)):
            for target in (u, v):
                if nbr == target:
                    continue
                nidx = uv_to_idx.get((target, nbr), uv_to_idx.get((nbr, target)))
                if nidx is not None:
                    neighbor_crashes.append(edge_crash_count.get(nidx, 0))

        avg_neighbor_crashes = float(np.mean(neighbor_crashes)) if neighbor_crashes else 0.0

        hw_neighbors = set()
        for nbr in set(G.neighbors(u)) | set(G.neighbors(v)):
            for target in (u, v):
                nidx = uv_to_idx.get((target, nbr), uv_to_idx.get((nbr, target)))
                if nidx is not None:
                    hw_neighbors.add(hw_col.iloc[nidx])

        road_continuity = 1.0 if hw_i in hw_neighbors else 0.0

        result[i] = {
            "betweenness": bc,
            "avg_neighbor_crashes": avg_neighbor_crashes,
            "road_continuity": road_continuity,
        }
    return result


def build_dataset(df_acc: pd.DataFrame,
                  gdf_edges: gpd.GeoDataFrame,
                  weather: "WeatherClient | None" = None,
                  aadf_csv: str | None = None,
                  G=None,
                  df_acc_historical: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a per-edge regression dataset with *crash_count* as target.

    When *aadf_csv* is provided, real DfT AADF counts replace the highway-class
    proxy for edges that have a nearby count point.  Exposure is computed as
    ``length_km * traffic * n_years * 365`` so that multi-year data is
    properly normalised.

    *df_acc_historical*, when provided, is used exclusively for computing
    edge-level descriptive features (``pct_night``, ``pct_wet``,
    ``pct_junction``, ``mean_severity``, ``avg_neighbor_crashes``) so that
    these features are derived from a separate time period and do **not** leak
    the target variable.  When ``None``, these features are set to safe
    defaults (0.0 / 3.0).
    """
    n_years = max(df_acc["year"].nunique(), 1) if "year" in df_acc.columns else 1
    print(f"  {n_years} year(s) of accident data")

    # -- load real AADF if available, with IDW interpolation ----------------
    real_aadf: dict[int, float] = {}
    if aadf_csv:
        df_aadf = load_aadf(aadf_csv)
        if not df_aadf.empty:
            direct = _snap_aadf_to_edges(df_aadf, gdf_edges)
            print(f"  AADF direct match: {len(direct)} / {len(gdf_edges)} edges")
            real_aadf = _idw_interpolate_aadf(direct, gdf_edges)
            print(f"  AADF after IDW interpolation: {len(real_aadf)} / {len(gdf_edges)} edges")

    # -- snap target accidents to edges (crash_count / severity_score) ------
    print("  Snapping target accidents...")
    target_agg = _snap_accidents_to_edges(df_acc, gdf_edges)
    edge_crash_count = target_agg["crash_count"]
    edge_severity_score = target_agg["severity_score"]

    # -- historical descriptive features (no leakage) ----------------------
    if df_acc_historical is not None and not df_acc_historical.empty:
        print("  Snapping historical accidents for descriptive features...")
        hist_agg = _snap_accidents_to_edges(df_acc_historical, gdf_edges)
        hist_crash_count = hist_agg["crash_count"]
        hist_night = hist_agg["night"]
        hist_wet = hist_agg["wet"]
        hist_junction = hist_agg["junction"]
        hist_severities = hist_agg["severities"]
    else:
        hist_crash_count = {}
        hist_night = {}
        hist_wet = {}
        hist_junction = {}
        hist_severities = {}

    # -- graph-based features (use historical counts if available) ----------
    graph_feats: dict[int, dict] = {}
    if G is not None:
        counts_for_graph = hist_crash_count if hist_crash_count else edge_crash_count
        graph_feats = _compute_graph_features(G, gdf_edges, counts_for_graph)

    # -- build feature rows for every edge ---------------------------------
    def _hw_norm(hw):
        return hw[0] if isinstance(hw, list) and hw else hw

    rows = []
    for idx in range(len(gdf_edges)):
        edge = gdf_edges.iloc[idx]
        hw = _hw_norm(edge.get("highway", "unclassified"))
        length_km = edge.get("length", 0.0) / 1000.0
        curvature = edge.get("curvature", 0.0)

        lanes_n = 0
        if "lanes_n" in edge.index:
            lanes_n = int(edge["lanes_n"]) if pd.notna(edge.get("lanes_n")) else 0
        elif "lanes" in edge.index:
            try:
                val = edge["lanes"]
                lanes_n = int(float(val[0]) if isinstance(val, list) else float(val))
            except Exception:
                lanes_n = 0

        if idx in real_aadf:
            tp = real_aadf[idx]
        else:
            tp = TRAFFIC_PROXY.get(hw, TRAFFIC_PROXY_DEFAULT)
            if lanes_n > 0:
                tp *= lanes_n

        speed_limit = 0
        if "speed_limit" in edge.index:
            try:
                speed_limit = int(edge["speed_limit"])
            except Exception:
                pass
        elif "maxspeed" in edge.index:
            try:
                ms = edge["maxspeed"]
                speed_limit = int(float(ms[0]) if isinstance(ms, list) else float(ms))
            except Exception:
                pass

        jd = edge.get("junction_density", 0.0)
        if pd.isna(jd):
            jd = 0.0

        cc = edge_crash_count.get(idx, 0)
        ss = edge_severity_score.get(idx, 0.0)

        exposure = max(length_km * tp * n_years * 365, 1e-9)
        log_exposure = np.log(exposure)
        crash_rate = cc / exposure
        severity_rate = ss / exposure

        geom = edge["geometry"]
        cx, cy = geom.centroid.x, geom.centroid.y
        pt_ll = gpd.GeoSeries([Point(cx, cy)], crs=gdf_edges.crs).to_crs(4326).iloc[0]

        hcc = hist_crash_count.get(idx, 0)
        pct_night = hist_night.get(idx, 0) / hcc if hcc > 0 else 0.0
        pct_wet = hist_wet.get(idx, 0) / hcc if hcc > 0 else 0.0
        pct_junction = hist_junction.get(idx, 0) / hcc if hcc > 0 else 0.0
        mean_severity = float(np.mean(hist_severities[idx])) if idx in hist_severities else 3.0

        gf = graph_feats.get(idx, {})

        row_dict = {
            "edge_idx": idx,
            "highway": hw,
            "curvature": curvature,
            "length_km": length_km,
            "speed_limit": speed_limit,
            "lanes_n": lanes_n,
            "traffic_proxy": tp,
            "junction_density": jd,
            "log_exposure": log_exposure,
            "pct_night": pct_night,
            "pct_wet": pct_wet,
            "pct_junction": pct_junction,
            "mean_severity": mean_severity,
            "betweenness": gf.get("betweenness", 0.0),
            "avg_neighbor_crashes": gf.get("avg_neighbor_crashes", 0.0),
            "road_continuity": gf.get("road_continuity", 1.0),
            "centroid_lat": float(pt_ll.y),
            "centroid_lon": float(pt_ll.x),
            "crash_count": cc,
            "severity_score": ss,
            "crash_rate": crash_rate,
            "severity_rate": severity_rate,
        }

        if weather is not None:
            lat_w, lon_w = float(pt_ll.y), float(pt_ll.x)
            season_dts = [
                datetime(YEAR, 1, 15, 12, tzinfo=timezone.utc),
                datetime(YEAR, 4, 15, 12, tzinfo=timezone.utc),
                datetime(YEAR, 7, 15, 12, tzinfo=timezone.utc),
                datetime(YEAR, 10, 15, 12, tzinfo=timezone.utc),
            ]
            season_names = ["winter", "spring", "summer", "autumn"]
            for sdt, sname in zip(season_dts, season_names):
                w = weather.get(lat_w, lon_w, sdt)
                for k, v in w.items():
                    row_dict[f"{k}_{sname}"] = v

        rows.append(row_dict)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Edge-hour dataset (STATS19 condition codes as features)
# ---------------------------------------------------------------------------

NEGATIVE_SAMPLE_SLOTS = [
    datetime(YEAR, 1, 15, 2, tzinfo=timezone.utc),
    datetime(YEAR, 3, 20, 8, tzinfo=timezone.utc),
    datetime(YEAR, 5, 10, 12, tzinfo=timezone.utc),
    datetime(YEAR, 7, 15, 14, tzinfo=timezone.utc),
    datetime(YEAR, 9, 5, 18, tzinfo=timezone.utc),
    datetime(YEAR, 11, 20, 22, tzinfo=timezone.utc),
]


def _safe_int(val, default=0):
    try:
        v = int(val)
        return v if v >= 0 else default
    except (ValueError, TypeError):
        return default


def build_dataset_edge_hour(
    df_acc: pd.DataFrame,
    gdf_edges: gpd.GeoDataFrame,
    aadf_csv: str | None = None,
    G=None,
    df_acc_historical: pd.DataFrame | None = None,
    neg_slots: list | None = None,
) -> pd.DataFrame:
    """Build an **edge-hour** Poisson dataset using STATS19 condition codes.

    Each row represents a (road segment, time-slot).  Positive rows come from
    actual STATS19 crashes with their recorded ``weather_conditions``,
    ``light_conditions`` and ``road_surface_conditions``.  Negative rows are
    generated by sampling every edge at a fixed set of representative
    time-slots with default "fine/daylight/dry" conditions.

    The resulting feature columns align exactly with what
    ``features.edge_features_eh()`` produces at routing time, so the model
    can respond to the user's departure hour and live weather.
    """
    if neg_slots is None:
        neg_slots = NEGATIVE_SAMPLE_SLOTS

    # -- AADF ---------------------------------------------------------------
    real_aadf: dict[int, float] = {}
    if aadf_csv:
        df_aadf = load_aadf(aadf_csv)
        if not df_aadf.empty:
            direct = _snap_aadf_to_edges(df_aadf, gdf_edges)
            real_aadf = _idw_interpolate_aadf(direct, gdf_edges)
            print(f"  AADF: {len(direct)} direct -> {len(real_aadf)} after IDW")

    # -- historical descriptive features ------------------------------------
    hist_crash_count: dict[int, int] = {}
    hist_night: dict[int, int] = {}
    hist_wet: dict[int, int] = {}
    hist_junction: dict[int, int] = {}
    hist_severities: dict[int, list[int]] = {}
    if df_acc_historical is not None and not df_acc_historical.empty:
        print("  Snapping historical accidents for descriptive features...")
        h_agg = _snap_accidents_to_edges(df_acc_historical, gdf_edges)
        hist_crash_count = h_agg["crash_count"]
        hist_night = h_agg["night"]
        hist_wet = h_agg["wet"]
        hist_junction = h_agg["junction"]
        hist_severities = h_agg["severities"]

    # -- graph features -----------------------------------------------------
    graph_feats: dict[int, dict] = {}
    if G is not None:
        counts_for_graph = hist_crash_count if hist_crash_count else {}
        graph_feats = _compute_graph_features(G, gdf_edges, counts_for_graph)

    # -- pre-compute static per-edge dict -----------------------------------
    def _hw_norm(hw):
        return hw[0] if isinstance(hw, list) and hw else hw

    static: dict[int, dict] = {}
    for idx in range(len(gdf_edges)):
        edge = gdf_edges.iloc[idx]
        hw = _hw_norm(edge.get("highway", "unclassified"))
        length_km = edge.get("length", 0.0) / 1000.0

        lanes_n = 0
        if "lanes_n" in edge.index:
            lanes_n = int(edge["lanes_n"]) if pd.notna(edge.get("lanes_n")) else 0

        if idx in real_aadf:
            tp = real_aadf[idx]
        else:
            tp = TRAFFIC_PROXY.get(hw, TRAFFIC_PROXY_DEFAULT)
            if lanes_n > 0:
                tp *= lanes_n

        speed_limit = 0
        if "speed_limit" in edge.index:
            try:
                speed_limit = int(edge["speed_limit"])
            except Exception:
                pass

        jd = edge.get("junction_density", 0.0)
        if pd.isna(jd):
            jd = 0.0

        exposure_hour = max(length_km * tp / 8760.0, 1e-12)

        hcc = hist_crash_count.get(idx, 0)
        gf = graph_feats.get(idx, {})

        geom = edge["geometry"]
        cx, cy = geom.centroid.x, geom.centroid.y
        pt_ll = gpd.GeoSeries([Point(cx, cy)], crs=gdf_edges.crs).to_crs(4326).iloc[0]

        static[idx] = {
            "edge_idx": idx,
            "highway": hw,
            "curvature": edge.get("curvature", 0.0),
            "length_km": length_km,
            "speed_limit": speed_limit,
            "lanes_n": lanes_n,
            "traffic_proxy": tp,
            "junction_density": jd,
            "log_exposure": np.log(exposure_hour),
            "pct_night": hist_night.get(idx, 0) / hcc if hcc > 0 else 0.0,
            "pct_wet": hist_wet.get(idx, 0) / hcc if hcc > 0 else 0.0,
            "pct_junction": hist_junction.get(idx, 0) / hcc if hcc > 0 else 0.0,
            "mean_severity": float(np.mean(hist_severities[idx])) if idx in hist_severities else 3.0,
            "betweenness": gf.get("betweenness", 0.0),
            "avg_neighbor_crashes": gf.get("avg_neighbor_crashes", 0.0),
            "road_continuity": gf.get("road_continuity", 1.0),
            "centroid_lat": float(pt_ll.y),
            "centroid_lon": float(pt_ll.x),
        }

    # -- positive rows: one per accident ------------------------------------
    print("  Building positive edge-hour rows...")
    acc_gdf = gpd.GeoDataFrame(
        df_acc,
        geometry=gpd.points_from_xy(df_acc["longitude"], df_acc["latitude"]),
        crs="EPSG:4326",
    ).to_crs(gdf_edges.crs)

    pos_rows: list[dict] = []
    for _, row in tqdm(acc_gdf.iterrows(), total=len(acc_gdf), desc="Edge-hour positives"):
        idx = nearest_edge_idx(row.geometry, gdf_edges)
        dt = row["dt_utc"]
        s = static[idx]

        pos_rows.append({
            **s,
            "hour": dt.hour,
            "dow": dt.weekday(),
            "month": dt.month,
            "season": _season(dt.month),
            "weather_conditions": _safe_int(row.get("weather_conditions"), 1),
            "light_conditions": _safe_int(row.get("light_conditions"), 1),
            "road_surface_conditions": _safe_int(row.get("road_surface_conditions"), 1),
            "crash_count": 1,
        })

    # -- negative rows: every edge × fixed slots ----------------------------
    print(f"  Building negative edge-hour rows ({len(gdf_edges)} edges × {len(neg_slots)} slots)...")
    neg_rows: list[dict] = []
    for idx in range(len(gdf_edges)):
        s = static[idx]
        for dt in neg_slots:
            neg_rows.append({
                **s,
                "hour": dt.hour,
                "dow": dt.weekday(),
                "month": dt.month,
                "season": _season(dt.month),
                "weather_conditions": 1,
                "light_conditions": 1 if 7 <= dt.hour <= 17 else 4,
                "road_surface_conditions": 1,
                "crash_count": 0,
            })

    # -- merge and aggregate duplicates -------------------------------------
    df = pd.DataFrame(pos_rows + neg_rows)
    agg_cols = [c for c in df.columns if c != "crash_count"]
    df = df.groupby(agg_cols, as_index=False, dropna=False)["crash_count"].sum()

    print(f"  Edge-hour dataset: {len(df)} rows, "
          f"{int(df['crash_count'].sum())} crashes, "
          f"{(df['crash_count'] > 0).sum()} positive rows")
    return df
