"""Feature extraction for a single edge at inference time."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .weather import WeatherClient

from .config import WEATHER_VARS


def edge_features(data: dict, G, dt_utc, weather: "WeatherClient") -> dict:
    """Build the feature dict that the trained model expects (edge-level model).

    *data* is the edge-attribute dict (already enriched by
    ``enrich_edges_static``).  *G* is the networkx graph (used as fallback
    for centroid coordinates).
    """
    curv = data.get("curvature", 0.0)
    length_km = data.get("length_km", data.get("length", 1.0) / 1000.0)
    hw = data.get("highway1", data.get("highway"))
    if isinstance(hw, list) and hw:
        hw = hw[0]

    speed_limit = data.get("speed_limit", 0)
    lanes_n = data.get("lanes_n", 0)
    traffic_proxy = data.get("traffic_proxy", 500.0)
    junction_density = data.get("junction_density", 0.0)

    lat = data.get("centroid_lat")
    lon = data.get("centroid_lon")
    if lat is None or lon is None:
        u = data.get("u")
        if u is not None and u in G.nodes:
            lat = G.nodes[u]["y"]
            lon = G.nodes[u]["x"]
        else:
            lat, lon = 51.5, -0.1

    w = weather.get(lat, lon, dt_utc)

    month = dt_utc.month
    season = _season(month)

    return {
        "curvature": curv,
        "length_km": length_km,
        "hour": dt_utc.hour,
        "dow": dt_utc.weekday(),
        "month": month,
        "season": season,
        "speed_limit": speed_limit,
        "lanes_n": lanes_n,
        "traffic_proxy": traffic_proxy,
        "junction_density": junction_density,
        **w,
        "highway": hw,
    }


# ---------------------------------------------------------------------------
# Edge-hour feature extraction (uses STATS19 condition codes)
# ---------------------------------------------------------------------------

def weather_api_to_stats19(api_data: dict, hour: int) -> dict:
    """Convert Open-Meteo API response to STATS19 categorical condition codes.

    Returns ``weather_conditions``, ``light_conditions``,
    ``road_surface_conditions`` as integers matching STATS19 encoding.
    """
    precip = api_data.get("precipitation", 0.0) or 0.0
    wind = api_data.get("wind_speed_10m", 0.0) or 0.0
    temp = api_data.get("temperature_2m", 10.0) or 10.0

    if temp < 0 and precip > 0:
        weather_cat = 3   # sleet / snow
    elif precip > 0.5 and wind > 10:
        weather_cat = 6   # rain + high wind
    elif precip > 0.5:
        weather_cat = 2   # rain (no high wind)
    elif wind > 15:
        weather_cat = 5   # fine + high wind
    else:
        weather_cat = 1   # fine, no high wind

    if 7 <= hour <= 18:
        light_cat = 1     # daylight
    else:
        light_cat = 4     # darkness, street lights present

    if temp < 0:
        surface_cat = 4   # frost / ice
    elif precip > 0.2:
        surface_cat = 2   # wet / damp
    else:
        surface_cat = 1   # dry

    return {
        "weather_conditions": weather_cat,
        "light_conditions": light_cat,
        "road_surface_conditions": surface_cat,
    }


def edge_features_eh(data: dict, G, dt_utc, weather: "WeatherClient") -> dict:
    """Build the feature dict for the **edge-hour** model.

    Uses STATS19-style condition codes derived from the Open-Meteo weather
    API, so the feature format matches the training data exactly.
    """
    curv = data.get("curvature", 0.0)
    length_km = data.get("length_km", data.get("length", 1.0) / 1000.0)
    hw = data.get("highway1", data.get("highway"))
    if isinstance(hw, list) and hw:
        hw = hw[0]

    speed_limit = data.get("speed_limit", 0)
    lanes_n = data.get("lanes_n", 0)
    traffic_proxy = data.get("traffic_proxy", 500.0)
    junction_density = data.get("junction_density", 0.0)

    lat = data.get("centroid_lat")
    lon = data.get("centroid_lon")
    if lat is None or lon is None:
        u = data.get("u")
        if u is not None and u in G.nodes:
            lat = G.nodes[u]["y"]
            lon = G.nodes[u]["x"]
        else:
            lat, lon = 51.5, -0.1

    api_weather = weather.get(lat, lon, dt_utc)
    conditions = weather_api_to_stats19(api_weather, dt_utc.hour)

    import math
    exposure_hour = max(length_km * traffic_proxy / 8760.0, 1e-12)

    month = dt_utc.month

    return {
        "curvature": curv,
        "length_km": length_km,
        "speed_limit": speed_limit,
        "lanes_n": lanes_n,
        "traffic_proxy": traffic_proxy,
        "junction_density": junction_density,
        "log_exposure": math.log(exposure_hour),
        "hour": dt_utc.hour,
        "dow": dt_utc.weekday(),
        "month": month,
        "season": _season(month),
        **conditions,
        "highway": hw,
    }


def _season(month: int) -> int:
    """Map month number (1-12) to season code: 0=winter, 1=spring, 2=summer, 3=autumn."""
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3
