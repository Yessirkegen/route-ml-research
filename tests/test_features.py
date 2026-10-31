"""Tests for route_risk.features -- edge feature extraction."""

from datetime import datetime, timezone

import pytest

from route_risk.features import edge_features, _season
from route_risk.weather import WeatherClient
from route_risk.config import WEATHER_VARS


@pytest.fixture
def dummy_weather(tmp_path):
    return WeatherClient(cache_path=str(tmp_path / "w.json"), no_weather=True)


class TestEdgeFeatures:
    def test_returns_all_expected_keys(self, dummy_weather):
        data = {
            "curvature": 0.5, "length_km": 1.2, "highway1": "primary",
            "speed_limit": 30, "lanes_n": 2, "traffic_proxy": 15000.0,
            "junction_density": 1.0,
            "centroid_lat": 51.5, "centroid_lon": -0.1,
            "u": 1, "v": 2,
        }

        class FakeG:
            nodes = {1: {"y": 51.5, "x": -0.1}, 2: {"y": 51.6, "x": -0.2}}

        dt = datetime(2022, 7, 15, 14, tzinfo=timezone.utc)
        feat = edge_features(data, FakeG(), dt, dummy_weather)

        assert "curvature" in feat
        assert "length_km" in feat
        assert "highway" in feat
        assert "hour" in feat
        assert "dow" in feat
        assert "month" in feat
        assert "season" in feat
        assert "speed_limit" in feat
        assert "lanes_n" in feat
        assert "traffic_proxy" in feat
        assert "junction_density" in feat
        for wv in WEATHER_VARS:
            assert wv in feat

    def test_hour_and_dow_from_dt(self, dummy_weather):
        data = {"curvature": 0, "length_km": 1, "highway1": "residential",
                "speed_limit": 20, "lanes_n": 0, "traffic_proxy": 1000,
                "junction_density": 0, "centroid_lat": 51.5, "centroid_lon": -0.1,
                "u": 1, "v": 2}

        class FakeG:
            nodes = {1: {"y": 51.5, "x": -0.1}, 2: {"y": 51.6, "x": -0.2}}

        dt = datetime(2022, 3, 14, 9, tzinfo=timezone.utc)  # Monday
        feat = edge_features(data, FakeG(), dt, dummy_weather)
        assert feat["hour"] == 9
        assert feat["dow"] == 0  # Monday


class TestSeason:
    @pytest.mark.parametrize("month,expected", [
        (1, 0), (2, 0), (12, 0),   # winter
        (3, 1), (4, 1), (5, 1),    # spring
        (6, 2), (7, 2), (8, 2),    # summer
        (9, 3), (10, 3), (11, 3),  # autumn
    ])
    def test_season_mapping(self, month, expected):
        assert _season(month) == expected
