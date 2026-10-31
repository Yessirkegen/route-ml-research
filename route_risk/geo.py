"""Geometry helpers, OSM graph construction, and static edge enrichment."""

import math

import osmnx as ox
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from .config import TRAFFIC_PROXY, TRAFFIC_PROXY_DEFAULT


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def _turning_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = (bx - ax, by - ay)
    v2 = (cx - bx, cy - by)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.acos(dot)


def curvature_index(geom: LineString) -> float:
    """Sum of turning angles divided by length (rad/m in projected CRS)."""
    try:
        coords = list(geom.coords)
    except Exception:
        return 0.0
    if len(coords) < 3:
        return 0.0
    ang = sum(abs(_turning_angle(coords[i - 1], coords[i], coords[i + 1]))
              for i in range(1, len(coords) - 1))
    length = geom.length
    return 0.0 if length == 0 else ang / length


# ---------------------------------------------------------------------------
# Nearest node / edge helpers
# ---------------------------------------------------------------------------

def nearest_node_safe(G, lat: float, lon: float):
    """Find the nearest graph node; works even without a projected CRS."""
    try:
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except ImportError:
        Gp = ox.project_graph(G)
        pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(Gp.graph["crs"]).iloc[0]
        return ox.distance.nearest_nodes(Gp, X=pt.x, Y=pt.y)


def nearest_edge_idx(pt, gdf_edges: gpd.GeoDataFrame) -> int:
    """Robust nearest-edge lookup across different rtree/sindex versions."""
    try:
        idxs = list(gdf_edges.sindex.nearest(pt))
        if idxs:
            return int(idxs[0])
    except Exception:
        pass
    try:
        cand = list(gdf_edges.sindex.query(pt.buffer(0.001)))
        if cand:
            sub = gdf_edges.iloc[cand]
            return int(sub.geometry.distance(pt).idxmin())
    except Exception:
        pass
    return int(gdf_edges.geometry.distance(pt).idxmin())


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _hw_norm(hw):
    return hw[0] if isinstance(hw, list) and hw else hw


def build_graph(north, south, east, west, road_classes=None):
    """Download/cache an OSM driving graph for the given bbox.

    osmnx 2.x expects ``bbox=(left, bottom, right, top)`` i.e.
    ``(west, south, east, north)``.
    """
    bbox_lbrt = (west, south, east, north)
    if road_classes:
        cf = f'["highway"~"{ "|".join(road_classes) }"]["access"!~"private"]'
        G = ox.graph_from_bbox(bbox=bbox_lbrt,
                               network_type="drive", simplify=True, custom_filter=cf)
    else:
        G = ox.graph_from_bbox(bbox=bbox_lbrt,
                               network_type="drive", simplify=True)
    G = ox.convert.to_undirected(G)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def _parse_maxspeed(val) -> int:
    """Extract numeric speed from OSM maxspeed (may be '30', '20 mph', list)."""
    if val is None:
        return 0
    if isinstance(val, list):
        val = val[0]
    val = str(val).strip()
    import re
    m = re.match(r"(\d+)", val)
    return int(m.group(1)) if m else 0


def _lanes_int(val) -> int:
    """Parse the OSM ``lanes`` tag (may be a list, string, or int)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    if isinstance(val, list):
        val = val[0]
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _junction_density(G, gdf):
    """Junctions per km: count of high-degree (>2) nodes along each edge."""
    node_deg = dict(G.degree())
    densities = []
    for row in gdf.itertuples(index=False):
        junctions = sum(1 for n in (row.u, row.v) if node_deg.get(n, 0) > 2)
        length_km = row.length / 1000.0
        densities.append(junctions / length_km if length_km > 0 else 0.0)
    return densities


def enrich_edges_static(G):
    """Pre-compute static features and write them into edge data dicts.

    Features added: curvature, length_km, highway1, lanes, traffic_proxy,
    junction_density, speed_limit, centroid_lat, centroid_lon.
    """
    gdf = (ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=True)
           .reset_index(drop=False).copy())

    gdf_proj_curv = ox.projection.project_gdf(gdf)
    gdf["curvature"] = gdf_proj_curv["geometry"].apply(curvature_index)
    gdf["length_km"] = gdf["length"] / 1000.0
    gdf["highway1"] = gdf["highway"].apply(_hw_norm)

    gdf["lanes_n"] = gdf.get("lanes", 0)
    if "lanes" in gdf.columns:
        gdf["lanes_n"] = gdf["lanes"].apply(_lanes_int)
    else:
        gdf["lanes_n"] = 0

    gdf["traffic_proxy"] = gdf["highway1"].map(TRAFFIC_PROXY).fillna(TRAFFIC_PROXY_DEFAULT)
    lanes_mult = gdf["lanes_n"].clip(lower=1).where(gdf["lanes_n"] > 0, 1)
    gdf["traffic_proxy"] = gdf["traffic_proxy"] * lanes_mult

    gdf["junction_density"] = _junction_density(G, gdf)

    if "maxspeed" in gdf.columns:
        gdf["speed_limit"] = gdf["maxspeed"].apply(_parse_maxspeed)
    else:
        gdf["speed_limit"] = 0

    gdf_proj = ox.projection.project_gdf(gdf)
    cent_proj = gdf_proj.geometry.centroid
    cent_ll = gpd.GeoSeries(cent_proj, crs=gdf_proj.crs).to_crs(4326)
    gdf["centroid_lat"] = cent_ll.y.values
    gdf["centroid_lon"] = cent_ll.x.values

    for row in gdf.itertuples(index=False):
        data = G[row.u][row.v][row.key]
        data["curvature"] = float(row.curvature)
        data["length_km"] = float(row.length_km)
        data["highway1"] = row.highway1
        data["lanes_n"] = int(row.lanes_n)
        data["traffic_proxy"] = float(row.traffic_proxy)
        data["junction_density"] = float(row.junction_density)
        data["speed_limit"] = int(row.speed_limit)
        data["centroid_lat"] = float(row.centroid_lat)
        data["centroid_lon"] = float(row.centroid_lon)
    return G
