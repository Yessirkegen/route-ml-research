"""Visualisation utilities: Folium maps, GeoJSON, PNG."""

from __future__ import annotations
from typing import TYPE_CHECKING

import geopandas as gpd
from shapely.geometry import LineString

if TYPE_CHECKING:
    pass


def path_to_gdf(G, path, label, ensure_cost=None, by_cost=False):
    """Build a GeoDataFrame of edge geometries for *path*."""
    from .routing import pick_edge

    lines = []
    for u, v in zip(path[:-1], path[1:]):
        d = pick_edge(G, u, v, ensure_cost=ensure_cost, by_cost=by_cost)
        geom = None if d is None else d.get("geometry")
        if geom is None:
            p1 = (G.nodes[u]["x"], G.nodes[u]["y"])
            p2 = (G.nodes[v]["x"], G.nodes[v]["y"])
            geom = LineString([p1, p2])
        lines.append({"label": label, "geometry": geom})
    return gpd.GeoDataFrame(lines, crs="EPSG:4326")


def save_folium_map(gdf_fast, gdf_risk, orig, dest, html_path):
    import folium

    ctr_lat = (orig[0] + dest[0]) / 2
    ctr_lon = (orig[1] + dest[1]) / 2
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB positron")

    if gdf_fast is not None and not gdf_fast.empty:
        folium.GeoJson(
            gdf_fast.__geo_interface__, name="Fastest",
            style_function=lambda _: {"color": "#1f77b4", "weight": 5, "opacity": 0.9},
        ).add_to(m)

    if gdf_risk is not None and not gdf_risk.empty:
        folium.GeoJson(
            gdf_risk.__geo_interface__, name="Risk-aware",
            style_function=lambda _: {"color": "#d62728", "weight": 5, "opacity": 0.9},
        ).add_to(m)

    folium.Marker([orig[0], orig[1]], icon=folium.Icon(color="green"),
                  tooltip="Origin").add_to(m)
    folium.Marker([dest[0], dest[1]], icon=folium.Icon(color="red"),
                  tooltip="Destination").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(html_path)
    print(f"Saved interactive map: {html_path}")


_PARETO_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def save_pareto_map(G, results: list[dict], orig, dest, html_path):
    """Save a Folium map with multiple routes (one per delta value)."""
    import folium

    ctr_lat = (orig[0] + dest[0]) / 2
    ctr_lon = (orig[1] + dest[1]) / 2
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB positron")

    for i, r in enumerate(results):
        color = _PARETO_COLORS[i % len(_PARETO_COLORS)]
        gdf = path_to_gdf(G, r["path"], f"delta={r['delta']}")
        name = f"delta={r['delta']} (ETA={r['eta_s'] / 60:.1f}m, risk={r['risk']:.4f})"
        folium.GeoJson(
            gdf.__geo_interface__, name=name,
            style_function=lambda _, c=color: {"color": c, "weight": 4, "opacity": 0.8},
        ).add_to(m)

    folium.Marker([orig[0], orig[1]], icon=folium.Icon(color="green"),
                  tooltip="Origin").add_to(m)
    folium.Marker([dest[0], dest[1]], icon=folium.Icon(color="red"),
                  tooltip="Destination").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(html_path)
    print(f"Saved Pareto map: {html_path}")
