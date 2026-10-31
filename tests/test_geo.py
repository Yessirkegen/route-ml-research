"""Tests for route_risk.geo -- curvature, edge helpers, graph enrichment."""

import math

import pytest
from shapely.geometry import LineString

from route_risk.geo import curvature_index, _turning_angle, _hw_norm, _lanes_int


class TestCurvatureIndex:
    def test_straight_line_has_zero_curvature(self):
        line = LineString([(0, 0), (1, 0), (2, 0)])
        assert curvature_index(line) == pytest.approx(0.0, abs=1e-9)

    def test_right_angle_has_positive_curvature(self):
        line = LineString([(0, 0), (1, 0), (1, 1)])
        assert curvature_index(line) > 0

    def test_s_curve_has_higher_curvature_than_single_bend(self):
        single = LineString([(0, 0), (1, 0), (1, 1)])
        s_curve = LineString([(0, 0), (1, 0), (1, 1), (2, 1), (2, 0)])
        assert curvature_index(s_curve) > curvature_index(single)

    def test_two_point_line_returns_zero(self):
        line = LineString([(0, 0), (5, 5)])
        assert curvature_index(line) == 0.0

    def test_empty_geometry_returns_zero(self):
        assert curvature_index(None) == 0.0


class TestTurningAngle:
    def test_straight(self):
        assert _turning_angle((0, 0), (1, 0), (2, 0)) == pytest.approx(0.0, abs=1e-9)

    def test_right_angle(self):
        assert _turning_angle((0, 0), (1, 0), (1, 1)) == pytest.approx(math.pi / 2, abs=1e-6)

    def test_zero_length_segment(self):
        assert _turning_angle((0, 0), (0, 0), (1, 1)) == 0.0


class TestHwNorm:
    def test_list_returns_first(self):
        assert _hw_norm(["primary", "trunk"]) == "primary"

    def test_string_unchanged(self):
        assert _hw_norm("residential") == "residential"

    def test_empty_list_unchanged(self):
        assert _hw_norm([]) == []


class TestLanesInt:
    def test_int_passthrough(self):
        assert _lanes_int(2) == 2

    def test_string(self):
        assert _lanes_int("3") == 3

    def test_list(self):
        assert _lanes_int(["4", "2"]) == 4

    def test_none(self):
        assert _lanes_int(None) == 0

    def test_nan(self):
        assert _lanes_int(float("nan")) == 0

    def test_non_numeric(self):
        assert _lanes_int("no") == 0
