"""Tests for route_risk.routing -- cost function, heuristic, path stats."""

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from route_risk.routing import (
    make_cost_fn,
    make_heuristic,
    edge_attr_candidates,
    path_stats,
    _haversine_m,
)


def _build_tiny_graph():
    """A simple 3-node graph: A --1km-- B --1km-- C, plus shortcut A--C (1.5km)."""
    G = nx.MultiGraph()
    G.add_node("A", y=51.50, x=-0.10)
    G.add_node("B", y=51.51, x=-0.10)
    G.add_node("C", y=51.52, x=-0.10)
    G.add_edge("A", "B", 0, length=1000, travel_time=60.0,
               curvature=0.0, length_km=1.0, highway1="primary",
               speed_limit=30, lanes_n=2, traffic_proxy=15000,
               junction_density=0.5,
               centroid_lat=51.505, centroid_lon=-0.10,
               geometry=None)
    G.add_edge("B", "C", 0, length=1000, travel_time=60.0,
               curvature=0.0, length_km=1.0, highway1="primary",
               speed_limit=30, lanes_n=2, traffic_proxy=15000,
               junction_density=0.5,
               centroid_lat=51.515, centroid_lon=-0.10,
               geometry=None)
    G.add_edge("A", "C", 0, length=1500, travel_time=80.0,
               curvature=0.3, length_km=1.5, highway1="residential",
               speed_limit=20, lanes_n=1, traffic_proxy=1000,
               junction_density=2.0,
               centroid_lat=51.51, centroid_lon=-0.10,
               geometry=None)
    return G


class TestEdgeAttrCandidates:
    def test_multi_graph(self):
        G = _build_tiny_graph()
        cands = edge_attr_candidates(G, "A", "B")
        assert len(cands) == 1
        assert cands[0]["length"] == 1000

    def test_missing_edge(self):
        G = _build_tiny_graph()
        assert edge_attr_candidates(G, "A", "Z") == []


class TestPathStats:
    def test_simple_path(self):
        G = _build_tiny_graph()
        path = ["A", "B", "C"]
        dist, eta, risk = path_stats(G, path)
        assert dist == pytest.approx(2000.0)
        assert eta == pytest.approx(120.0)

    def test_single_node_path(self):
        G = _build_tiny_graph()
        dist, eta, risk = path_stats(G, ["A"])
        assert dist == 0.0
        assert eta == 0.0


class TestHeuristic:
    def test_same_node_zero(self):
        G = _build_tiny_graph()
        h = make_heuristic(G, alpha=1.0)
        assert h("A", "A") == 0.0

    def test_positive_for_different_nodes(self):
        G = _build_tiny_graph()
        h = make_heuristic(G, alpha=1.0)
        assert h("A", "C") > 0

    def test_alpha_scales_heuristic(self):
        G = _build_tiny_graph()
        h1 = make_heuristic(G, alpha=1.0)
        h2 = make_heuristic(G, alpha=2.0)
        assert h2("A", "C") == pytest.approx(2 * h1("A", "C"))


class TestHaversine:
    def test_same_point(self):
        G = _build_tiny_graph()
        assert _haversine_m(G, "A", "A") == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        G = _build_tiny_graph()
        d = _haversine_m(G, "A", "B")
        assert 1000 < d < 1200  # ~1.11 km for 0.01 deg lat


class TestMakeCostFn:
    def test_cost_includes_risk_contribution(self):
        G = _build_tiny_graph()
        dt = datetime(2022, 6, 15, 12, tzinfo=timezone.utc)

        mock_pipe = MagicMock()
        mock_pipe.predict.return_value = np.array([0.5])

        mock_weather = MagicMock()
        mock_weather.get.return_value = {
            "temperature_2m": 10, "precipitation": 0,
            "wind_speed_10m": 3, "cloudcover": 50,
        }

        w_fn, ec_fn = make_cost_fn(G, dt, mock_pipe, alpha=1.0, delta=100.0,
                                    weather=mock_weather)

        data = dict(G["A"]["B"][0])
        cost = w_fn("A", "B", data)

        assert cost > data["travel_time"]
        assert data["_risk_per_km"] == pytest.approx(0.5)
        assert data["_risk_contribution"] == pytest.approx(0.5 * 1.0)
