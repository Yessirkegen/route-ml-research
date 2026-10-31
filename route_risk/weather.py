"""WeatherClient -- stateful, cached access to Open-Meteo ERA5 hourly data."""

import json
import math

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import WEATHER_VARS, DEFAULT_WEATHER, WEATHER_ROUND_DEG


class WeatherClient:
    """Fetches and caches hourly weather from the Open-Meteo ERA5 archive."""

    def __init__(self, cache_path: str = "weather_cache.json",
                 round_deg: float = WEATHER_ROUND_DEG,
                 timeout: int = 60, retries: int = 5,
                 no_weather: bool = False):
        self.cache_path = cache_path
        self.round_deg = round_deg
        self.timeout = timeout
        self.retries = retries
        self.no_weather = no_weather
        self._cache: dict = self._load()
        self._hits_since_flush = 0

    # -- persistence --------------------------------------------------------

    def _load(self) -> dict:
        try:
            with open(self.cache_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self):
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f)
        except Exception:
            pass

    # -- public API ---------------------------------------------------------

    def get(self, lat: float, lon: float, dt_utc, flush_every: int = 200) -> dict:
        """Return a dict of weather variables for *(lat, lon)* at *dt_utc*."""
        if self.no_weather:
            return DEFAULT_WEATHER.copy()
        key = self._key(lat, lon, dt_utc)
        if key not in self._cache:
            self._cache[key] = self._fetch(lat, lon, dt_utc)
            self._hits_since_flush += 1
            if self._hits_since_flush % flush_every == 0:
                self.save()
        return self._cache[key]

    # -- internals ----------------------------------------------------------

    def _key(self, lat: float, lon: float, dt_utc) -> str:
        step = self.round_deg
        if step <= 0:
            qlat = round(lat, 2)
            qlon = round(lon, 2)
        else:
            qlat = round(round(lat / step) * step, 3)
            qlon = round(round(lon / step) * step, 3)
        return f"{qlat:.3f}_{qlon:.3f}_{dt_utc.strftime('%Y%m%d%H')}"

    def _fetch(self, lat, lon, dt, backoff=1.5):
        base = "https://archive-api.open-meteo.com/v1/era5"
        date_str = dt.strftime("%Y-%m-%d")
        url = (
            f"{base}?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&start_date={date_str}&end_date={date_str}"
            f"&hourly={','.join(WEATHER_VARS)}&timezone=UTC"
        )
        sess = requests.Session()
        retry = Retry(
            total=self.retries, connect=self.retries, read=self.retries,
            status=self.retries, backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        sess.mount("https://", HTTPAdapter(max_retries=retry))
        try:
            r = sess.get(url, timeout=self.timeout)
            r.raise_for_status()
            js = r.json()
            times = js["hourly"]["time"]
            target = dt.strftime("%Y-%m-%dT%H:00")
            try:
                idx = times.index(target)
            except ValueError:
                hours = [int(t.split("T")[1].split(":")[0]) for t in times]
                idx = int(np.argmin([abs(h - dt.hour) for h in hours]))
            result = {}
            for v in WEATHER_VARS:
                val = js["hourly"].get(v, [None] * (idx + 1))[idx]
                result[v] = val if val is not None else DEFAULT_WEATHER.get(v, np.nan)
            return result
        except Exception:
            return DEFAULT_WEATHER.copy()
