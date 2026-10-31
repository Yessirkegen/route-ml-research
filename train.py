#!/usr/bin/env python3
"""CLI entry-point: train the risk model."""

import argparse

from route_risk.config import BBOX, YEAR, WEATHER_ROUND_DEG


def main():
    ap = argparse.ArgumentParser(description="Train a Poisson risk model on STATS19 + OSM data")
    ap.add_argument("--accidents-csv", required=True,
                    help="Path(s) to STATS19 CSV -- comma-separated or glob "
                         "(e.g. 'collisions_*.csv' or 'collisions_last5y.csv')")
    ap.add_argument("--model-out", default="models/risk_model.joblib")
    ap.add_argument("--weather-cache", default="data/weather_cache.json")
    ap.add_argument("--weather-round", type=float, default=WEATHER_ROUND_DEG)
    ap.add_argument("--no-weather", action="store_true",
                    help="Skip weather API calls; use default weather values")
    ap.add_argument("--aadf-csv", default="",
                    help="Path to DfT AADF CSV (dft_traffic_counts_aadf.csv); "
                         "replaces highway-class proxy with real traffic volumes")
    ap.add_argument("--cross-validate", action="store_true",
                    help="Run spatial cross-validation after training")
    ap.add_argument("--historical-csv", default="",
                    help="Separate STATS19 CSV(s) for descriptive features "
                         "(pct_night, pct_wet, etc.) to prevent target leakage. "
                         "When empty, these features default to 0.")
    ap.add_argument("--test-year", type=int, default=0,
                    help="Hold out this year for temporal validation "
                         "(e.g. 2023). 0 = use random split only.")
    ap.add_argument("--edge-hour", action="store_true",
                    help="Use edge-hour dataset with STATS19 condition codes "
                         "instead of edge-level with seasonal weather.")
    args = ap.parse_args()

    from route_risk.data import load_accidents_csv, build_dataset, build_dataset_edge_hour
    from route_risk.geo import build_graph, enrich_edges_static
    from route_risk.weather import WeatherClient
    from route_risk.model import train, cross_validate_spatial, sanity_check_top_edges, load_model
    from route_risk.config import BBOX
    import osmnx as ox

    print("[1/6] Loading accidents...")
    df_acc = load_accidents_csv(args.accidents_csv)
    if df_acc.empty:
        raise SystemExit("No accidents in bbox/year. Adjust BBOX/YEAR or CSV path.")

    df_acc_test = None
    if args.test_year > 0 and "year" in df_acc.columns:
        df_acc_test = df_acc[df_acc["year"] == args.test_year].copy()
        df_acc = df_acc[df_acc["year"] != args.test_year].copy()
        print(f"  Temporal split: train={len(df_acc)} accidents, "
              f"held-out {args.test_year}={len(df_acc_test)} accidents")
    print(f"  {len(df_acc)} training accidents in bbox")

    df_acc_historical = None
    if args.historical_csv:
        df_acc_historical = load_accidents_csv(args.historical_csv)
        print(f"  {len(df_acc_historical)} historical accidents for descriptive features")

    print("[2/6] Building road graph...")
    north, south, east, west = BBOX
    G = build_graph(north, south, east, west)
    G = enrich_edges_static(G)
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True,
                                  fill_edge_geometry=True).reset_index(drop=False).copy()
    print(f"  {len(gdf_edges)} edges")

    if args.edge_hour:
        print("[3/6] Building edge-hour dataset (STATS19 conditions)...")
        data = build_dataset_edge_hour(
            df_acc, gdf_edges,
            aadf_csv=args.aadf_csv or None, G=G,
            df_acc_historical=df_acc_historical,
        )
    else:
        print("[3/6] Building count-based dataset...")
        weather = WeatherClient(cache_path=args.weather_cache,
                                round_deg=args.weather_round,
                                no_weather=args.no_weather)
        data = build_dataset(df_acc, gdf_edges, weather=weather,
                             aadf_csv=args.aadf_csv or None, G=G,
                             df_acc_historical=df_acc_historical)
        weather.save()
    print(f"  {len(data)} rows, {int(data['crash_count'].sum())} matched crashes")

    print("[4/6] Training Poisson model...")
    metrics = train(data, model_out=args.model_out,
                    test_year=args.test_year if df_acc_test is not None else 0,
                    df_acc_test=df_acc_test, gdf_edges=gdf_edges)

    if args.cross_validate:
        print("[4b/6] Spatial cross-validation...")
        cross_validate_spatial(data)

    print("[5/6] Sanity check...")
    pipe = load_model(args.model_out)
    sanity_check_top_edges(data, pipe)

    edge_lookup_path = args.model_out.replace(".joblib", "_edge_features.csv")
    from route_risk.model import NUM_COLS, NUM_COLS_EH, CAT_COLS, _available_weather_cols
    if args.edge_hour:
        keep_cols = ["centroid_lat", "centroid_lon"] + NUM_COLS_EH + CAT_COLS
    else:
        weather_cols = _available_weather_cols(data)
        keep_cols = ["centroid_lat", "centroid_lon"] + NUM_COLS + weather_cols + CAT_COLS
    keep_cols = [c for c in keep_cols if c in data.columns]

    if args.edge_hour:
        static_cols = [c for c in keep_cols
                       if c not in {"hour", "dow", "month", "season",
                                    "weather_conditions", "light_conditions",
                                    "road_surface_conditions"}]
        lookup_df = data[static_cols].drop_duplicates(subset=["centroid_lat", "centroid_lon"])
        lookup_df.to_csv(edge_lookup_path, index=False)
    else:
        data[keep_cols].to_csv(edge_lookup_path, index=False)
    print(f"  [6/6] Edge feature lookup saved to {edge_lookup_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
