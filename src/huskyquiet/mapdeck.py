"""PyDeck maps of scored permits. Only rows with real Seattle coordinates are drawn.

Markers are small circles built as polygons around each recorded coordinate.
ScatterplotLayer draws nothing in the deck.gl build Streamlit 1.65 ships
(instanced quads are discarded), so the map uses SolidPolygonLayer instead.
The circle is a symbol at the coordinate, not a parcel or a noise contour.
"""

from __future__ import annotations

import html
import math

import pandas as pd
import pydeck as pdk

from huskyquiet.analysis import format_probability, probability_color
from huskyquiet.config import UW_LAT, UW_LON
from huskyquiet.geo import valid_coordinates, with_distance

CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
_EARTH_M_PER_DEG = 111_320.0


def mappable_permits(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[valid_coordinates(frame)].copy()
    out["color"] = [probability_color(p) for p in out["probability_issued_90d"]]
    out["p_label"] = [format_probability(p) for p in out["probability_issued_90d"]]
    if "address_key" in out.columns:
        out["tip_address"] = out["address_key"].map(_tip_address)
    else:
        out["tip_address"] = "Address not recorded"
    out["tip_type"] = _tip_type(out)
    return out


def city_deck(
    permits: pd.DataFrame,
    *,
    latitude: float = UW_LAT,
    longitude: float = UW_LON,
    zoom: float = 13,
) -> pdk.Deck:
    """City map. Marker color is predicted P(issued within 90 days)."""
    data = mappable_permits(permits)
    layers = [
        _permit_layer(data, zoom),
        _marker_layer(
            UW_LON,
            UW_LAT,
            zoom,
            color=(196, 163, 90, 230),
            tip_address="University of Washington",
            tip_type="Reference point (Red Square)",
            p_label="",
            layer_id="uw",
        ),
    ]
    return _deck(layers, latitude, longitude, zoom)


def location_deck(
    permits: pd.DataFrame,
    latitude: float,
    longitude: float,
    radius_m: float,
) -> pdk.Deck:
    """Apartment-centered map with a radius ring and the permits inside it."""
    nearby = with_distance(permits, latitude, longitude, radius_m)
    data = mappable_permits(nearby)
    zoom = 15 if radius_m <= 250 else 14 if radius_m <= 500 else 13
    layers = [
        _ring_layer(longitude, latitude, radius_m, zoom),
        _permit_layer(data, zoom),
        _marker_layer(
            longitude,
            latitude,
            zoom,
            color=(196, 163, 90, 230),
            tip_address="Selected location",
            tip_type=f"Search radius {int(radius_m)} m",
            p_label="",
            layer_id="home",
        ),
    ]
    return _deck(layers, latitude, longitude, zoom)


def _permit_layer(data: pd.DataFrame, zoom: float) -> pdk.Layer:
    radius_m = _marker_radius_m(zoom)
    records = []
    if not data.empty:
        colors = data["color"].tolist()
        for row, color in zip(data.itertuples(index=False), colors, strict=True):
            records.append({
                "polygon": _circle(float(row.longitude), float(row.latitude), radius_m),
                "r": int(color[0]),
                "g": int(color[1]),
                "b": int(color[2]),
                "tip_address": row.tip_address,
                "tip_type": row.tip_type,
                "p_label": row.p_label,
                "permit_num": row.permit_num,
            })
    return pdk.Layer(
        "SolidPolygonLayer",
        data=records,
        id="permits",
        pickable=True,
        auto_highlight=True,
        filled=True,
        stroked=False,
        extruded=False,
        get_polygon="polygon",
        get_fill_color="[r, g, b, 220]",
    )


def _marker_layer(
    longitude: float,
    latitude: float,
    zoom: float,
    *,
    color: tuple[int, int, int, int],
    tip_address: str,
    tip_type: str,
    p_label: str,
    layer_id: str,
) -> pdk.Layer:
    return pdk.Layer(
        "SolidPolygonLayer",
        data=[{
            "polygon": _circle(longitude, latitude, _marker_radius_m(zoom) * 0.7),
            "tip_address": tip_address,
            "tip_type": tip_type,
            "p_label": p_label,
        }],
        id=layer_id,
        pickable=True,
        filled=True,
        stroked=False,
        extruded=False,
        get_polygon="polygon",
        get_fill_color=list(color),
    )


def _ring_layer(longitude: float, latitude: float, radius_m: float, zoom: float) -> pdk.Layer:
    # A hole in one polygon is filled solid by this deck.gl build, so the ring
    # is a chain of quads between an outer and inner circle.
    thickness = max(_marker_radius_m(zoom) * 0.45, 10.0)
    outer = _circle(longitude, latitude, radius_m, steps=72)
    inner = _circle(longitude, latitude, max(radius_m - thickness, 1.0), steps=72)
    quads = []
    for index in range(72):
        quads.append({
            "polygon": [outer[index], outer[index + 1], inner[index + 1], inner[index]],
        })
    return pdk.Layer(
        "SolidPolygonLayer",
        data=quads,
        id="radius",
        pickable=False,
        filled=True,
        stroked=False,
        extruded=False,
        get_polygon="polygon",
        get_fill_color=[14, 77, 92, 180],
    )


def _circle(longitude: float, latitude: float, radius_m: float, steps: int = 20) -> list[list[float]]:
    ring: list[list[float]] = []
    lat_scale = _EARTH_M_PER_DEG
    lon_scale = _EARTH_M_PER_DEG * math.cos(math.radians(latitude))
    for step in range(steps):
        angle = 2 * math.pi * step / steps
        ring.append([
            longitude + (radius_m * math.sin(angle)) / lon_scale,
            latitude + (radius_m * math.cos(angle)) / lat_scale,
        ])
    ring.append(ring[0])
    return ring


def _marker_radius_m(zoom: float) -> float:
    """Geographic radius that stays about eight pixels across typical zooms."""
    meters_per_pixel = 156_543.03 * math.cos(math.radians(UW_LAT)) / (2**zoom)
    return meters_per_pixel * 8


def _deck(layers: list, latitude: float, longitude: float, zoom: float) -> pdk.Deck:
    return pdk.Deck(
        map_style=CARTO_POSITRON,
        initial_view_state=pdk.ViewState(
            latitude=latitude, longitude=longitude, zoom=zoom, pitch=0, bearing=0,
        ),
        layers=layers,
        tooltip={
            "html": "<b>{tip_address}</b><br/>{tip_type}<br/>{p_label}",
            "style": {
                "backgroundColor": "#1C2421",
                "color": "#F7F5F0",
                "fontSize": "12px",
            },
        },
    )


def _tip_address(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "Address not recorded"
    return html.escape(str(value))


def _tip_type(frame: pd.DataFrame) -> pd.Series:
    kind = frame["permit_type_mapped"].fillna("Type not recorded").astype(str) if "permit_type_mapped" in frame else ""
    desc = frame["permit_type_desc"].fillna("").astype(str) if "permit_type_desc" in frame else ""
    return (kind + " · " + desc).str.strip(" ·").map(html.escape)
