"""Tests for route_risk.weather -- WeatherClient caching and key generation."""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from route_risk.weather import WeatherClient
from route_risk.config import DEFAULT_WEATHER, WEATHER_VARS


@pytest.fixture
def tmp_cache(tmp_path):
    return str(tmp_path / "weather_test.json")


class TestWeatherKey:
    def test_quantisation_at_default(self, tmp_cache):
        wc = WeatherClient(cache_path=tmp_cache, round_deg=0.05, no_weather=True)
        dt = datetime(2022, 6, 15, 10, tzinfo=timezone.utc)
        key = wc._key(51.512, -0.127, dt)
        assert "2022061510" in key
        lat_part = float(key.split("_")[0])
        assert lat_part == pytest.approx(51.50, abs=0.05)

    def test_different_hours_give_different_keys(self, tmp_cache):
        wc = WeatherClient(cache_path=tmp_cache, round_deg=0.05, no_weather=True)
        dt1 = datetime(2022, 6, 15, 10, tzinfo=timezone.utc)
        dt2 = datetime(2022, 6, 15, 11, tzinfo=timezone.utc)
        assert wc._key(51.5, -0.1, dt1) != wc._key(51.5, -0.1, dt2)


class TestWeatherClientNoWeather:
    def test_returns_defaults_when_no_weather(self, tmp_cache):
        wc = WeatherClient(cache_path=tmp_cache, no_weather=True)
        dt = datetime(2022, 1, 1, 12, tzinfo=timezone.utc)
        result = wc.get(51.5, -0.1, dt)
        for var in WEATHER_VARS:
            assert var in result
        assert result["temperature_2m"] == DEFAULT_WEATHER["temperature_2m"]


class TestWeatherClientCaching:
    def test_cache_persists_to_disk(self, tmp_cache):
        wc = WeatherClient(cache_path=tmp_cache, no_weather=False)
        dt = datetime(2022, 6, 15, 10, tzinfo=timezone.utc)
        # Manually insert into cache to avoid real HTTP call
        key = wc._key(51.5, -0.1, dt)
        wc._cache[key] = DEFAULT_WEATHER.copy()
        wc.save()
        assert os.path.exists(tmp_cache)
        with open(tmp_cache) as f:
            data = json.load(f)
        assert len(data) >= 1

    def test_cache_loaded_on_init(self, tmp_cache):
        wc1 = WeatherClient(cache_path=tmp_cache, no_weather=False)
        dt = datetime(2022, 6, 15, 10, tzinfo=timezone.utc)
        key = wc1._key(51.5, -0.1, dt)
        wc1._cache[key] = DEFAULT_WEATHER.copy()
        wc1.save()

        wc2 = WeatherClient(cache_path=tmp_cache, no_weather=False)
        assert len(wc2._cache) >= 1
