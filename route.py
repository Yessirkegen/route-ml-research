#!/usr/bin/env python3
"""CLI entry-point: compute fastest vs risk-aware routes."""

import argparse

from route_risk.config import WEATHER_ROUND_DEG


def main():
    ap = argparse.ArgumentParser(description="Risk-aware routing on OSM")
    ap.add_argument("--orig", required=True, help="lat,lon")
    ap.add_argument("--dest", required=True, help="lat,lon")
    ap.add_argument("--depart", required=True,
                    help="YYYY-MM-DDTHH:MM (UTC or local; converted to UTC)")

    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Weight of travel-time (seconds) in cost function")
    ap.add_argument("--delta", type=float, default=100.0,
                    help="Penalty for risk_per_km * length_km")

    ap.add_argument("--bbox-pad", type=float, default=0.05)
    ap.add_argument("--roads",
                    default="motorway,trunk,primary,secondary,tertiary",
                    help='Comma-separated highway types; "all" disables filter')
    ap.add_argument("--model", default="models/risk_model.joblib")

    ap.add_argument("--weather-cache", default="data/weather_cache_route.json")
    ap.add_argument("--weather-round", type=float, default=WEATHER_ROUND_DEG)
    ap.add_argument("--weather-timeout", type=int, default=60)
    ap.add_argument("--weather-retries", type=int, default=5)
    ap.add_argument("--no-weather", action="store_true")

    ap.add_argument("--max-eta-penalty", type=float, default=None,
                    help="Max extra ETA (minutes) for risk-aware route; "
                         "fallback to fastest if exceeded")

    ap.add_argument("--out", default="", help="CSV path for metrics (append)")
    ap.add_argument("--geojson", default="")
    ap.add_argument("--geojson-fast", default="")
    ap.add_argument("--geojson-risk", default="")
    ap.add_argument("--map-html", default="")
    ap.add_argument("--png", default="")

    # Pareto-front mode
    ap.add_argument("--pareto", action="store_true",
                    help="Run Pareto-front mode with multiple delta values")
    ap.add_argument("--pareto-deltas", default="0,50,100,200,500,1000",
                    help="Comma-separated delta values for Pareto mode")

    args = ap.parse_args()
    olat, olon = map(float, args.orig.split(","))
    dlat, dlon = map(float, args.dest.split(","))

    if args.pareto:
        from route_risk.routing import route_pareto
        deltas = tuple(int(d) for d in args.pareto_deltas.split(","))
        route_pareto(
            (olat, olon), (dlat, dlon), args.depart,
            deltas=deltas, alpha=args.alpha,
            bbox_pad=args.bbox_pad, roads=args.roads,
            model_path=args.model,
            weather_cache=args.weather_cache,
            weather_round=args.weather_round,
            weather_timeout=args.weather_timeout,
            weather_retries=args.weather_retries,
            no_weather=args.no_weather,
            map_html_path=args.map_html,
        )
    else:
        from route_risk.routing import route_with_ml
        route_with_ml(
            (olat, olon), (dlat, dlon), args.depart,
            alpha=args.alpha, delta=args.delta,
            bbox_pad=args.bbox_pad, roads=args.roads,
            model_path=args.model,
            weather_cache=args.weather_cache,
            weather_round=args.weather_round,
            weather_timeout=args.weather_timeout,
            weather_retries=args.weather_retries,
            no_weather=args.no_weather,
            max_eta_penalty_min=args.max_eta_penalty,
            out_csv=args.out,
            geojson_path=args.geojson,
            geojson_fast_path=args.geojson_fast,
            geojson_risk_path=args.geojson_risk,
            map_html_path=args.map_html,
            png_path=args.png,
        )


if __name__ == "__main__":
    main()
