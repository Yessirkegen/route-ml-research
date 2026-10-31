"""Centralised configuration constants."""

# Central London bounding box (north, south, east, west).
# Covers ~15x10 km from Camden to Brixton, Tower Bridge to Kensington.
# Increase for broader coverage (but slower OSM download).
BBOX = (51.55, 51.45, -0.01, -0.20)
YEAR = 2022

# Weather variables requested from Open-Meteo ERA5.
# visibility is excluded -- ERA5 returns null for the London region.
WEATHER_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "cloudcover"]

DEFAULT_WEATHER = {
    "temperature_2m": 10.0,
    "precipitation": 0.0,
    "wind_speed_10m": 3.0,
    "cloudcover": 50.0,
}

# Unified quantisation step for weather tile cache (degrees).
# ~5 km -- a compromise between the old 0.01 (train) and 0.2 (route).
WEATHER_ROUND_DEG = 0.05

# Severity weights for STATS19 collision_severity codes.
SEVERITY_WEIGHT = {1: 10.0, 2: 3.0, 3: 1.0}

# Approximate annual average daily flow (vehicles/day) per highway class.
# Used as an exposure proxy when real AADF data is unavailable.
TRAFFIC_PROXY = {
    "motorway": 50_000,
    "motorway_link": 25_000,
    "trunk": 30_000,
    "trunk_link": 15_000,
    "primary": 15_000,
    "primary_link": 8_000,
    "secondary": 8_000,
    "secondary_link": 4_000,
    "tertiary": 4_000,
    "tertiary_link": 2_000,
    "residential": 1_000,
    "living_street": 200,
    "unclassified": 500,
    "service": 100,
}
TRAFFIC_PROXY_DEFAULT = 500
