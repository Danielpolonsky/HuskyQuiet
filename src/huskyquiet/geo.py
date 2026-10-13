"""Distances and point-in-polygon tests for permit locations.

Distances are great-circle lengths on the WGS84 sphere. Neighborhood membership uses
the ray-casting test on the polygons shipped in ``uw_areas.geojson``. Nothing here
invents a coordinate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from huskyquiet.config import SEATTLE_BBOX
from huskyquiet.features.engineering import haversine_km


def haversine_m(lat1, lon1, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance in meters. Inputs may be scalars or arrays."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1000.0


def in_seattle(latitude: float, longitude: float) -> bool:
    return (
        SEATTLE_BBOX["lat_min"] <= latitude <= SEATTLE_BBOX["lat_max"]
        and SEATTLE_BBOX["lon_min"] <= longitude <= SEATTLE_BBOX["lon_max"]
    )


def valid_coordinates(frame: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.Series:
    """True where both coordinates are finite and inside the Seattle bounding box."""
    lat = pd.to_numeric(frame[lat_col], errors="coerce")
    lon = pd.to_numeric(frame[lon_col], errors="coerce")
    return (
        lat.notna()
        & lon.notna()
        & lat.between(SEATTLE_BBOX["lat_min"], SEATTLE_BBOX["lat_max"])
        & lon.between(SEATTLE_BBOX["lon_min"], SEATTLE_BBOX["lon_max"])
    )


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_geometry(longitude: float, latitude: float, geometry: dict) -> bool:
    """True if the point is inside a GeoJSON Polygon or MultiPolygon, respecting holes."""
    kind = geometry["type"]
    if kind == "Polygon":
        return _polygon_contains(longitude, latitude, geometry["coordinates"])
    if kind == "MultiPolygon":
        return any(_polygon_contains(longitude, latitude, poly) for poly in geometry["coordinates"])
    raise ValueError(f"unsupported geometry type {kind}")


def _polygon_contains(lon: float, lat: float, rings: list) -> bool:
    if not rings or not _point_in_ring(lon, lat, rings[0]):
        return False
    return not any(_point_in_ring(lon, lat, hole) for hole in rings[1:])


def geometry_bounds(geometry: dict) -> tuple[float, float, float, float]:
    """Return ``(min_lon, min_lat, max_lon, max_lat)``."""
    coords = _flatten_coords(geometry)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def _flatten_coords(geometry: dict) -> list:
    if geometry["type"] == "Polygon":
        return [pt for ring in geometry["coordinates"] for pt in ring]
    if geometry["type"] == "MultiPolygon":
        return [pt for poly in geometry["coordinates"] for ring in poly for pt in ring]
    raise ValueError(f"unsupported geometry type {geometry['type']}")


def view_for_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Center latitude, longitude, and a zoom level that frames the bounding box."""
    min_lon, min_lat, max_lon, max_lat = bounds
    lat = (min_lat + max_lat) / 2
    lon = (min_lon + max_lon) / 2
    span = max(max_lat - min_lat, (max_lon - min_lon) * np.cos(np.radians(lat)))
    if span < 0.008:
        zoom = 15.0
    elif span < 0.02:
        zoom = 14.0
    elif span < 0.045:
        zoom = 13.0
    else:
        zoom = 12.0
    return lat, lon, zoom


def with_distance(
    frame: pd.DataFrame,
    latitude: float,
    longitude: float,
    radius_m: float | None = None,
) -> pd.DataFrame:
    """Copy ``frame`` with ``distance_m`` for rows that have real Seattle coordinates.

    Rows without usable coordinates are dropped. When ``radius_m`` is set, rows farther
    than that distance are dropped too. The result is sorted nearest-first.
    """
    keep = frame.loc[valid_coordinates(frame)].copy()
    if keep.empty:
        keep["distance_m"] = pd.Series(dtype="float64")
        return keep
    keep["distance_m"] = haversine_m(
        keep["latitude"].to_numpy(), keep["longitude"].to_numpy(), latitude, longitude
    )
    if radius_m is not None:
        keep = keep.loc[keep["distance_m"] <= radius_m]
    return keep.sort_values("distance_m", ignore_index=True)
