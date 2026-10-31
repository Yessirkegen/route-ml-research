#!/usr/bin/env python3
"""Run all experiments for the IEEE SIST paper.

Usage:
    python experiments/run_all.py [--skip-train] [--skip-route] [--skip-ablation]
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OD_PATH = ROOT / "experiments" / "od_pairs.json"
RESULTS_CSV = ROOT / "experiments" / "routing_results.csv"
MODEL_OUT = ROOT / "models" / "risk_model_v2.joblib"
BASELINE_CSV = ROOT / "experiments" / "baseline_comparison.csv"
ABLATION_CSV = ROOT / "experiments" / "ablation_study.csv"
AADF_COMPARE_CSV = ROOT / "experiments" / "aadf_comparison.csv"


def step_train(args):
    """Train model with multi-year data, optional AADF, weather, and baselines."""
    from route_risk.data import load_accidents_csv, build_dataset, build_dataset_edge_hour
    from route_risk.geo import build_graph, enrich_edges_static
    from route_risk.weather import WeatherClient
    from route_risk.model import (train, cross_validate_spatial,
                                  sanity_check_top_edges, load_model,
                                  compare_baselines, ablation_study)
    from route_risk.config import BBOX
    import osmnx as ox

    acc_csvs = "data/collisions_2020.csv,data/collisions_2021.csv,data/collisions_2022.csv"
    hist_csvs = "data/collisions_2020.csv,data/collisions_2021.csv"
    test_csv = "data/collisions_2023.csv"

    available_csvs = [c for c in acc_csvs.split(",") if os.path.exists(c)]
    if not available_csvs:
        print("[WARN] No collision CSVs found -- using data/collisions_2022.csv only")
        available_csvs = ["data/collisions_2022.csv"]
    acc_csv_str = ",".join(available_csvs)

    print("=" * 60)
    print("STEP 1: TRAIN MODEL")
    print("=" * 60)

    print(f"\n[1/8] Loading target accidents: {acc_csv_str}")
    df_acc = load_accidents_csv(acc_csv_str)
    print(f"  {len(df_acc)} target accidents")

    hist_available = [c for c in hist_csvs.split(",") if os.path.exists(c)]
    df_acc_historical = None
    if hist_available:
        print(f"[1b/8] Loading historical accidents: {','.join(hist_available)}")
        df_acc_historical = load_accidents_csv(",".join(hist_available))
        print(f"  {len(df_acc_historical)} historical accidents for descriptive features")

    df_acc_test = None
    if os.path.exists(test_csv):
        print(f"[1c/8] Loading held-out test year: {test_csv}")
        df_acc_test = load_accidents_csv(test_csv)
        print(f"  {len(df_acc_test)} held-out 2023 accidents")

    print("[2/8] Building road graph...")
    north, south, east, west = BBOX
    G = build_graph(north, south, east, west)
    G = enrich_edges_static(G)
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True,
                                  fill_edge_geometry=True).reset_index(drop=False).copy()
    print(f"  {len(gdf_edges)} edges")

    aadf_csv = "data/dft_traffic_counts_aadf.csv" if os.path.exists("data/dft_traffic_counts_aadf.csv") else None

    print("[3/8] Building edge-hour dataset (STATS19 conditions)...")
    data = build_dataset_edge_hour(
        df_acc, gdf_edges,
        aadf_csv=aadf_csv, G=G,
        df_acc_historical=df_acc_historical,
    )
    print(f"  {len(data)} rows, {int(data['crash_count'].sum())} matched crashes")

    print("[4/8] Training main model...")
    metrics = train(data, model_out=str(MODEL_OUT),
                    df_acc_test=df_acc_test, gdf_edges=gdf_edges,
                    test_year=2023 if df_acc_test is not None else 0)

    print("[5/8] Spatial cross-validation...")
    cross_validate_spatial(data)

    print("[6/8] Sanity check...")
    pipe = load_model(str(MODEL_OUT))
    sanity_check_top_edges(data, pipe)

    from route_risk.model import NUM_COLS_EH, CAT_COLS
    keep_cols = ["centroid_lat", "centroid_lon"] + NUM_COLS_EH + CAT_COLS
    keep_cols = [c for c in keep_cols if c in data.columns]
    static_cols = [c for c in keep_cols
                   if c not in {"hour", "dow", "month", "season",
                                "weather_conditions", "light_conditions",
                                "road_surface_conditions"}]
    edge_path = str(MODEL_OUT).replace(".joblib", "_edge_features.csv")
    data[static_cols].drop_duplicates(subset=["centroid_lat", "centroid_lon"]).to_csv(
        edge_path, index=False)

    if not args.skip_ablation:
        print("[7/8] Baseline comparison...")
        bl = compare_baselines(data)
        bl.to_csv(str(BASELINE_CSV), index=False)
        print(f"  Saved to {BASELINE_CSV}")

        print("[8/8] Ablation study...")
        ab = ablation_study(data)
        ab.to_csv(str(ABLATION_CSV), index=False)
        print(f"  Saved to {ABLATION_CSV}")

    if aadf_csv:
        print("[AADF] Training comparison model WITHOUT AADF...")
        data_no_aadf = build_dataset_edge_hour(
            df_acc, gdf_edges, aadf_csv=None, G=G,
            df_acc_historical=df_acc_historical)
        from route_risk.model import _get_feature_cols, _build_lgbm, _poisson_deviance, _spearman
        from sklearn.model_selection import train_test_split
        import numpy as _np

        num_cols, feature_cols = _get_feature_cols(data_no_aadf)
        y_na = data_no_aadf["crash_count"].values.astype(float)
        X_na = data_no_aadf[feature_cols].copy()
        Xtr, Xte, ytr, yte = train_test_split(X_na, y_na, test_size=0.2, random_state=42)
        try:
            pipe_na = _build_lgbm(num_cols, CAT_COLS)
            pipe_na.fit(Xtr, ytr)
            yp = pipe_na.predict(Xte)
        except ImportError:
            from route_risk.model import _build_poisson
            pipe_na = _build_poisson(num_cols, CAT_COLS)
            pipe_na.fit(Xtr, ytr)
            yp = pipe_na.predict(Xte)

        import pandas as _pd
        aadf_rows = [
            {"variant": "With AADF", "deviance": metrics["deviance"],
             "mae": metrics["mae"], "spearman": metrics["spearman"]},
            {"variant": "Without AADF (proxy only)",
             "deviance": _poisson_deviance(yte, yp),
             "mae": float(_np.mean(_np.abs(yte - yp))),
             "spearman": _spearman(yte, yp)},
        ]
        _pd.DataFrame(aadf_rows).to_csv(str(AADF_COMPARE_CSV), index=False)
        print(f"  AADF comparison saved to {AADF_COMPARE_CSV}")


def step_route(args):
    """Run routing experiments for all OD pairs."""
    with open(OD_PATH) as f:
        cfg = json.load(f)

    departures = cfg["departures"]
    deltas = cfg["deltas"]
    pairs = cfg["pairs"]

    model_path = str(MODEL_OUT)
    if not os.path.exists(model_path):
        model_path = "models/risk_model.joblib"

    if os.path.exists(str(RESULTS_CSV)):
        os.remove(str(RESULTS_CSV))

    maps_dir = ROOT / "experiments" / "maps"
    maps_dir.mkdir(exist_ok=True)

    total = len(pairs) * len(departures) * len(deltas)
    done = 0

    for pair in pairs:
        pid = pair["id"]
        olat, olon = pair["orig"]
        dlat, dlon = pair["dest"]

        for depart in departures:
            for delta in deltas:
                done += 1
                print(f"\n--- [{done}/{total}] {pid} | {depart} | delta={delta} ---")

                map_path = ""
                if delta in (0, 100, 500):
                    safe_depart = depart.replace(":", "-")
                    map_path = str(maps_dir / f"{pid}__{safe_depart}__d{delta}.html")

                from route_risk.routing import route_with_ml
                try:
                    route_with_ml(
                        (olat, olon), (dlat, dlon), depart,
                        alpha=1.0, delta=float(delta),
                        model_path=model_path,
                        weather_cache="data/weather_cache_route.json",
                        no_weather=False,
                        out_csv=str(RESULTS_CSV),
                        map_html_path=map_path,
                    )
                except Exception as e:
                    print(f"  [ERROR] {e}")

    print(f"\nAll routing results saved to {RESULTS_CSV}")


def step_temporal_validation(args):
    """Count 2023 crashes on edges of fastest vs risk-aware routes."""
    import pandas as pd
    import numpy as np

    if not os.path.exists(str(RESULTS_CSV)):
        print("[SKIP] No routing results found. Run --route first.")
        return

    test_csv = "collisions_2023.csv"
    if not os.path.exists(test_csv):
        print("[SKIP] collisions_2023.csv not found.")
        return

    from route_risk.data import load_accidents_csv, _snap_accidents_to_edges
    from route_risk.geo import build_graph, enrich_edges_static
    from route_risk.config import BBOX
    import osmnx as ox

    print("Loading 2023 accidents...")
    df_test = load_accidents_csv(test_csv)
    print(f"  {len(df_test)} held-out accidents")

    print("Building graph...")
    north, south, east, west = BBOX
    G = build_graph(north, south, east, west)
    G = enrich_edges_static(G)
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True,
                                  fill_edge_geometry=True).reset_index(drop=False).copy()

    print("Snapping 2023 accidents to edges...")
    test_agg = _snap_accidents_to_edges(df_test, gdf_edges)
    test_counts = test_agg["crash_count"]

    results = pd.read_csv(str(RESULTS_CSV))
    print(f"  {len(results)} routing result rows loaded")
    print(f"  Total 2023 crashes mapped: {sum(test_counts.values())}")

    out_path = ROOT / "experiments" / "temporal_validation.csv"
    results.to_csv(str(out_path), index=False)
    print(f"  Temporal validation context saved to {out_path}")
    print("  (Full route-level crash counting requires storing edge lists per route.)")


def main():
    ap = argparse.ArgumentParser(description="Run all IEEE SIST experiments")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-route", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-temporal", action="store_true")
    args = ap.parse_args()

    if not args.skip_train:
        step_train(args)
    if not args.skip_route:
        step_route(args)
    if not args.skip_temporal:
        step_temporal_validation(args)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
