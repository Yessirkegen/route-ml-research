"""Tests for route_risk.data -- CSV loading and dataset construction."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString

from route_risk.data import load_accidents_csv, build_dataset


@pytest.fixture
def mini_csv(tmp_path):
    """Create a minimal STATS19-like CSV with 3 rows inside the default BBOX."""
    csv_path = tmp_path / "collisions_mini.csv"
    rows = [
        "collision_index,longitude,latitude,date,time,collision_severity,speed_limit,"
        "light_conditions,road_surface_conditions,weather_conditions,"
        "number_of_vehicles,number_of_casualties",
        "1,-0.12,51.50,15/03/2022,08:30,3,30,1,1,1,2,1",
        "2,-0.10,51.52,20/06/2022,17:00,2,40,4,2,2,1,2",
        "3,-0.15,51.48,01/12/2022,02:15,1,60,7,3,6,3,1",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    return str(csv_path)


@pytest.fixture
def mini_edges():
    """A tiny GeoDataFrame pretending to be OSM edges."""
    lines = [
        LineString([(-0.13, 51.49), (-0.11, 51.51)]),
        LineString([(-0.11, 51.51), (-0.09, 51.53)]),
        LineString([(-0.16, 51.47), (-0.14, 51.49)]),
    ]
    gdf = gpd.GeoDataFrame({
        "geometry": lines,
        "highway": ["primary", "secondary", "residential"],
        "length": [2500.0, 3000.0, 1500.0],
        "curvature": [0.1, 0.05, 0.2],
        "lanes_n": [2, 1, 0],
        "speed_limit": [30, 40, 20],
        "junction_density": [1.0, 0.5, 2.0],
    }, crs="EPSG:4326")
    return gdf


class TestLoadAccidentsCsv:
    def test_loads_correct_count(self, mini_csv):
        df = load_accidents_csv(mini_csv)
        assert len(df) == 3

    def test_has_required_columns(self, mini_csv):
        df = load_accidents_csv(mini_csv)
        assert "longitude" in df.columns
        assert "latitude" in df.columns
        assert "dt_utc" in df.columns
        assert "collision_severity" in df.columns

    def test_filters_by_year(self, mini_csv):
        df = load_accidents_csv(mini_csv, year=2023)
        assert len(df) == 0


class TestBuildDataset:
    def test_returns_all_edges(self, mini_csv, mini_edges):
        df_acc = load_accidents_csv(mini_csv)
        data = build_dataset(df_acc, mini_edges, weather=None)
        assert len(data) == len(mini_edges)

    def test_has_crash_rate_column(self, mini_csv, mini_edges):
        df_acc = load_accidents_csv(mini_csv)
        data = build_dataset(df_acc, mini_edges, weather=None)
        assert "crash_rate" in data.columns
        assert "severity_rate" in data.columns
        assert (data["crash_rate"] >= 0).all()

    def test_crash_counts_sum_to_accidents(self, mini_csv, mini_edges):
        df_acc = load_accidents_csv(mini_csv)
        data = build_dataset(df_acc, mini_edges, weather=None)
        assert data["crash_count"].sum() == len(df_acc)
