"""Risk-aware routing: cost functions, A* search, path metrics."""

from __future__ import annotations

import math
import os
import csv
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd

from .features import edge_features, edge_features_eh, weather_api_to_stats19, _season

if TYPE_CHECKING:
    from .weather import WeatherClient


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def make_cost_fn(G, dt_utc, pipe, alpha: float, delta: float, weather: "WeatherClient",
                 edge_lookup: pd.DataFrame | None = None):
    """Return *(weight_fn, ensure_cost_fn)* for A* routing.

    ``cost = alpha * travel_time + delta * risk_per_km * length_km``

    *risk_per_km* is the model prediction (exposure-normalised crash rate).
    Multiplying by ``length_km`` makes the risk contribution proportional to
    edge length, so that many short edges are not penalised more than a single
    long edge at the same risk density.
    """
    cache_token = dt_utc.strftime("%Y%m%d%H")

    # Pre-build KD-tree for fast edge lookup matching
    _tree = None
    _lookup_rows = None
    if edge_lookup is not None and not edge_lookup.empty:
        from scipy.spatial import cKDTree
        coords = edge_lookup[["centroid_lat", "centroid_lon"]].values
        _tree = cKDTree(coords)
        _lookup_rows = edge_lookup

    def _find_lookup_row(lat, lon):
        if _tree is None:
            return None
        dist_sq, idx = _tree.query([lat, lon])
        if dist_sq < 0.01:  # ~1 km match radius
            return _lookup_rows.iloc[idx]
        return None

    try:
        _model_features = set()
        for _, _, cols in pipe.named_steps["pre"].transformers_:
            _model_features.update(cols)
        _is_eh_model = "weather_conditions" in _model_features
    except Exception:
        _is_eh_model = False

    _TEMPORAL_KEYS = {"hour", "dow", "month", "season",
                      "weather_conditions", "light_conditions",
                      "road_surface_conditions"}

    def ensure_cost(u, v, data):
        if data.get("_cache_ts") == cache_token and "_cost" in data:
            return data["_cost"]

        eta = data.get("travel_time")
        if eta is None:
            eta = data.get("length", 1.0) / (30_000 / 3600)

        lat = data.get("centroid_lat", G.nodes[u]["y"])
        lon = data.get("centroid_lon", G.nodes[u]["x"])

        lookup_row = _find_lookup_row(lat, lon)

        if _is_eh_model:
            if lookup_row is not None:
                feat = {k: v for k, v in
                        lookup_row.drop(["centroid_lat", "centroid_lon"],
                                        errors="ignore").to_dict().items()
                        if k not in _TEMPORAL_KEYS}
            else:
                feat = edge_features_eh(data | {"u": u, "v": v}, G, dt_utc, weather)
                for extra in ["pct_night", "pct_wet", "pct_junction",
                              "mean_severity", "betweenness",
                              "avg_neighbor_crashes", "road_continuity"]:
                    feat.setdefault(extra, 0.0)

            feat["hour"] = dt_utc.hour
            feat["dow"] = dt_utc.weekday()
            feat["month"] = dt_utc.month
            feat["season"] = _season(dt_utc.month)
            api_w = weather.get(lat, lon, dt_utc)
            feat.update(weather_api_to_stats19(api_w, dt_utc.hour))

        elif lookup_row is not None:
            feat = lookup_row.drop(["centroid_lat", "centroid_lon"],
                                   errors="ignore").to_dict()
        else:
            feat = edge_features(data | {"u": u, "v": v}, G, dt_utc, weather)
            length_km = data.get("length_km", data.get("length", 1.0) / 1000.0)
            traffic_proxy = data.get("traffic_proxy", 500.0)
            exposure = max(length_km * traffic_proxy * 365, 1e-9)
            feat["log_exposure"] = math.log(exposure)
            for extra in ["pct_night", "pct_wet", "pct_junction", "mean_severity",
                          "betweenness", "avg_neighbor_crashes", "road_continuity"]:
                feat.setdefault(extra, 0.0)
            if _lookup_rows is not None:
                for col in _lookup_rows.columns:
                    if col not in ("centroid_lat", "centroid_lon") and col not in feat:
                        feat[col] = 0.0

        X = pd.DataFrame([feat])
        predicted_crashes = max(float(pipe.predict(X)[0]), 0.0)
        length_km = data.get("length_km", data.get("length", 1.0) / 1000.0)
        risk_per_km = predicted_crashes / max(length_km, 1e-6)
        risk_contribution = predicted_crashes

        cost = alpha * eta + delta * risk_contribution

        data["_cache_ts"] = cache_token
        data["_risk_per_km"] = risk_per_km
        data["_risk_contribution"] = risk_contribution
        data["_cost"] = cost
        data["_eta"] = eta
        return cost

    def weight(u, v, data):
        return ensure_cost(u, v, data)

    return weight, ensure_cost


# ---------------------------------------------------------------------------
# Heuristic for A*
# ---------------------------------------------------------------------------

V_REF = 70_000 / 3600  # m/s -- conservative lower bound


def _haversine_m(G, u, v):
    y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
    y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
    R = 6_371_000.0
    dlon = math.radians(x2 - x1)
    dlat = math.radians(y2 - y1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(y1)) * math.cos(math.radians(y2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def make_heuristic(G, alpha: float):
    def h(n, goal):
        return (_haversine_m(G, n, goal) / V_REF) * alpha
    return h


# ---------------------------------------------------------------------------
# Edge attribute helpers (Graph / MultiGraph)
# ---------------------------------------------------------------------------

def edge_attr_candidates(G, u, v) -> list[dict]:
    ed = G.get_edge_data(u, v)
    if ed is None:
        return []
    if isinstance(ed, dict) and any(isinstance(val, dict) for val in ed.values()):
        return [d for d in ed.values() if isinstance(d, dict)]
    if isinstance(ed, dict):
        return [ed]
    return []


def pick_edge(G, u, v, ensure_cost=None, by_cost=False):
    cands = edge_attr_candidates(G, u, v)
    if not cands:
        return None
    if by_cost and ensure_cost is not None:
        best = min(cands, key=lambda d: ensure_cost(u, v, d))
    else:
        best = min(cands, key=lambda d: d.get("travel_time", float("inf")))
    if ensure_cost is not None:
        ensure_cost(u, v, best)
    return best


# ---------------------------------------------------------------------------
# Path statistics
# ---------------------------------------------------------------------------

def path_stats(G, path, ensure_cost=None, for_cost=False):
    """Return ``(dist_m, eta_s, risk_sum)`` for a path."""
    dist = eta = risk = 0.0
    for u, v in zip(path[:-1], path[1:]):
        candidates = edge_attr_candidates(G, u, v)
        if not candidates:
            continue
        if for_cost and ensure_cost is not None:
            best = min(candidates, key=lambda d: ensure_cost(u, v, d))
        else:
            best = min(candidates, key=lambda d: d.get("travel_time", float("inf")))
        if ensure_cost is not None:
            ensure_cost(u, v, best)
        dist += best.get("length", 0.0)
        eta += best.get("travel_time", 0.0)
        risk += best.get("_risk_contribution", 0.0)
    return dist, eta, risk


# ---------------------------------------------------------------------------
# CSV results logger
# ---------------------------------------------------------------------------

def append_csv(out_csv: str, row: dict):
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Main routing pipeline
# ---------------------------------------------------------------------------

def route_with_ml(
    orig, dest, depart_iso, *,
    alpha=1.0, delta=0.6,
    bbox_pad=0.05,
    roads="motorway,trunk,primary,secondary,tertiary",
    model_path="risk_model.joblib",
    weather_cache="weather_cache_route.json",
    weather_round=0.05,
    weather_timeout=60, weather_retries=5, no_weather=False,
    max_eta_penalty_min=None,
    out_csv="",
    geojson_path="", geojson_fast_path="", geojson_risk_path="",
    map_html_path="", png_path="",
):
    from .geo import build_graph, enrich_edges_static, nearest_node_safe
    from .model import load_model
    from .weather import WeatherClient
    from .viz import path_to_gdf, save_folium_map

    dt_utc = datetime.fromisoformat(depart_iso).astimezone(timezone.utc).replace(tzinfo=timezone.utc)
    olat, olon = orig
    dlat, dlon = dest

    north = max(olat, dlat) + bbox_pad
    south = min(olat, dlat) - bbox_pad
    east = max(olon, dlon) + bbox_pad
    west = min(olon, dlon) - bbox_pad

    road_classes = []
    if roads and roads.lower() != "all":
        road_classes = [r.strip() for r in roads.split(",") if r.strip()]

    print("[1/6] Load graph")
    G = build_graph(north, south, east, west, road_classes)
    G = enrich_edges_static(G)

    print("[2/6] Load model")
    pipe = load_model(model_path)

    import os
    lookup_path = model_path.replace(".joblib", "_edge_features.csv")
    edge_lookup = None
    if os.path.exists(lookup_path):
        edge_lookup = pd.read_csv(lookup_path)
        print(f"  Loaded edge feature lookup ({len(edge_lookup)} rows)")

    weather = WeatherClient(cache_path=weather_cache, round_deg=weather_round,
                            timeout=weather_timeout, retries=weather_retries,
                            no_weather=no_weather)

    print("[3/6] Prepare cost function")
    cost_weight, ensure_cost = make_cost_fn(G, dt_utc, pipe, alpha, delta, weather,
                                            edge_lookup=edge_lookup)
    h = make_heuristic(G, alpha)

    print("[4/6] Shortest paths")
    on = nearest_node_safe(G, olat, olon)
    dn = nearest_node_safe(G, dlat, dlon)

    path_fast = nx.astar_path(G, on, dn, heuristic=h,
                              weight=lambda u, v, d: d.get("travel_time", 1.0))
    path_risk = nx.astar_path(G, on, dn, heuristic=h,
                              weight=lambda u, v, d: cost_weight(u, v, d))

    d1, t1, r1 = path_stats(G, path_fast, ensure_cost=ensure_cost, for_cost=False)
    d2, t2, r2 = path_stats(G, path_risk, ensure_cost=ensure_cost, for_cost=True)

    # -- risk-reversal guard ------------------------------------------------
    if r2 > r1:
        print(f"[!] Risk-aware route has HIGHER risk ({r2:.4f}) than fastest "
              f"({r1:.4f}). Falling back to fastest route.")
        path_risk = path_fast
        d2, t2, r2 = d1, t1, r1

    # -- max ETA penalty guard ---------------------------------------------
    if max_eta_penalty_min is not None:
        eta_penalty_min = (t2 - t1) / 60.0
        if eta_penalty_min > max_eta_penalty_min:
            print(f"[!] Risk-aware route exceeds ETA penalty limit "
                  f"({eta_penalty_min:+.1f} > {max_eta_penalty_min} min). "
                  f"Falling back to fastest route.")
            path_risk = path_fast
            d2, t2, r2 = d1, t1, r1

    print("[5/6] Save weather cache")
    weather.save()

    deta_min = round((t2 - t1) / 60, 1)
    ddist_km = round((d2 - d1) / 1000, 2)
    drisk = r1 - r2
    prisk = (drisk / r1 * 100.0) if r1 > 0 else 0.0

    print("\n=== RESULTS ===")
    print(f"Fastest:   dist={d1 / 1000:.2f} km  ETA={int(t1 // 60)}m  risk_sum={r1:.4f}")
    print(f"RiskAware: dist={d2 / 1000:.2f} km  ETA={int(t2 // 60)}m  risk_sum={r2:.4f}")
    print(f"dETA: {deta_min:+.1f} min  dDist: {ddist_km:+.2f} km  "
          f"dRisk: {drisk:+.4f} ({prisk:+.1f}%)")

    if out_csv:
        append_csv(out_csv, {
            "orig": f"{orig[0]:.4f},{orig[1]:.4f}",
            "dest": f"{dest[0]:.4f},{dest[1]:.4f}",
            "depart": depart_iso, "alpha": alpha, "delta": delta,
            "bbox_pad": bbox_pad, "roads": roads,
            "dist_fast_km": round(d1 / 1000, 3), "eta_fast_s": round(t1),
            "risk_fast": round(r1, 6),
            "dist_risk_km": round(d2 / 1000, 3), "eta_risk_s": round(t2),
            "risk_risk": round(r2, 6),
            "deta_s": round(t2 - t1), "drisk": round(drisk, 6),
            "no_weather": no_weather, "weather_round": weather_round,
        })

    # -- visualisations ----------------------------------------------------
    print("[6/6] Export")
    gdf_fast = path_to_gdf(G, path_fast, "fastest", ensure_cost=ensure_cost, by_cost=False)
    gdf_risk = path_to_gdf(G, path_risk, "riskaware", ensure_cost=ensure_cost, by_cost=True)

    if geojson_path:
        both = pd.concat([gdf_fast.assign(type="fastest"),
                          gdf_risk.assign(type="risk")], ignore_index=True)
        both.to_file(geojson_path, driver="GeoJSON")
        print(f"  GeoJSON (both): {geojson_path}")
    if geojson_fast_path:
        gdf_fast.to_file(geojson_fast_path, driver="GeoJSON")
    if geojson_risk_path:
        gdf_risk.to_file(geojson_risk_path, driver="GeoJSON")
    if map_html_path:
        save_folium_map(gdf_fast, gdf_risk, (olat, olon), (dlat, dlon), map_html_path)
    if png_path:
        try:
            import osmnx as ox
            fig, ax = ox.plot_graph_routes(
                G, [path_fast, path_risk],
                route_colors=["#1f77b4", "#d62728"],
                route_linewidth=4, node_size=0, bgcolor="w", show=False, close=False,
            )
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            print(f"  PNG: {png_path}")
        except Exception as e:
            print(f"  PNG export failed: {e}")


# ---------------------------------------------------------------------------
# Pareto-front routing
# ---------------------------------------------------------------------------

def route_pareto(
    orig, dest, depart_iso, *,
    deltas=(0, 50, 100, 200, 500, 1000),
    alpha=1.0,
    bbox_pad=0.05,
    roads="motorway,trunk,primary,secondary,tertiary",
    model_path="risk_model.joblib",
    weather_cache="weather_cache_route.json",
    weather_round=0.05,
    weather_timeout=60, weather_retries=5, no_weather=False,
    map_html_path="",
):
    """Compute multiple risk-aware routes with varying delta and export a Pareto map."""
    from .geo import build_graph, enrich_edges_static, nearest_node_safe
    from .model import load_model
    from .weather import WeatherClient
    from .viz import save_pareto_map

    dt_utc = datetime.fromisoformat(depart_iso).astimezone(timezone.utc).replace(tzinfo=timezone.utc)
    olat, olon = orig
    dlat, dlon = dest

    north = max(olat, dlat) + bbox_pad
    south = min(olat, dlat) - bbox_pad
    east = max(olon, dlon) + bbox_pad
    west = min(olon, dlon) - bbox_pad

    road_classes = []
    if roads and roads.lower() != "all":
        road_classes = [r.strip() for r in roads.split(",") if r.strip()]

    G = build_graph(north, south, east, west, road_classes)
    G = enrich_edges_static(G)
    pipe = load_model(model_path)

    import os
    lookup_path = model_path.replace(".joblib", "_edge_features.csv")
    edge_lookup = None
    if os.path.exists(lookup_path):
        edge_lookup = pd.read_csv(lookup_path)

    weather = WeatherClient(cache_path=weather_cache, round_deg=weather_round,
                            timeout=weather_timeout, retries=weather_retries,
                            no_weather=no_weather)

    on = nearest_node_safe(G, olat, olon)
    dn = nearest_node_safe(G, dlat, dlon)
    h = make_heuristic(G, alpha)

    results = []
    for d in deltas:
        w_fn, ec_fn = make_cost_fn(G, dt_utc, pipe, alpha, float(d), weather,
                                   edge_lookup=edge_lookup)
        if d == 0:
            path = nx.astar_path(G, on, dn, heuristic=h,
                                 weight=lambda u, v, data: data.get("travel_time", 1.0))
        else:
            path = nx.astar_path(G, on, dn, heuristic=h,
                                 weight=lambda u, v, data: w_fn(u, v, data))
        dist, eta, risk = path_stats(G, path, ensure_cost=ec_fn, for_cost=(d > 0))
        results.append({"delta": d, "path": path, "dist_m": dist,
                         "eta_s": eta, "risk": risk})
        print(f"  delta={d:5d}  ETA={eta / 60:.1f}m  risk={risk:.4f}")

    weather.save()

    if map_html_path:
        save_pareto_map(G, results, (olat, olon), (dlat, dlon), map_html_path)

    return results
