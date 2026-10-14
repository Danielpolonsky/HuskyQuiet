"""Named UW-area extents used to filter the map.

Neighborhood polygons are a simplified copy of the Seattle City Clerk Neighborhood Map
Atlas. That atlas is an unofficial indexing boundary, not an official neighborhood map
and not a risk score. University Village is the OpenStreetMap footprint of the shopping
center. Permit locations themselves always come from the City permit file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

from huskyquiet.geo import geometry_bounds, point_in_geometry, valid_coordinates, view_for_bounds

_AREAS_PATH = Path(__file__).with_name("uw_areas.geojson")


@dataclass(frozen=True)
class Area:
    name: str
    kind: str
    source: str
    note: str
    geometry: dict

    @property
    def label(self) -> str:
        return f"{self.name} (shopping center)" if self.kind == "place" else self.name

    @property
    def scope_note(self) -> str:
        if self.kind == "place":
            return (
                "University Village is the shopping center's footprint from OpenStreetMap, not a neighborhood. "
                "It sits inside the Ravenna extent, so its permits are also counted in Ravenna."
            )
        return (
            f"{self.name} uses the City Clerk's Neighborhood Map Atlas extent, "
            "an unofficial indexing boundary rather than a legal neighborhood line."
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return geometry_bounds(self.geometry)

    @property
    def view(self) -> tuple[float, float, float]:
        return view_for_bounds(self.bounds)


@cache
def load_areas() -> tuple[Area, ...]:
    payload = json.loads(_AREAS_PATH.read_text(encoding="utf-8"))
    areas = []
    for feature in payload["features"]:
        props = feature["properties"]
        areas.append(Area(
            name=props["name"],
            kind=props["kind"],
            source=props["source"],
            note=props["note"],
            geometry=feature["geometry"],
        ))
    return tuple(areas)


def area_by_name(name: str) -> Area:
    for area in load_areas():
        if area.name == name:
            return area
    known = ", ".join(area.name for area in load_areas())
    raise KeyError(f"unknown area {name!r}; expected one of {known}")


def area_by_label(label: str) -> Area:
    for area in load_areas():
        if area.label == label:
            return area
    return area_by_name(label)


def area_scope(area: Area) -> str:
    if area.kind == "place":
        return "inside the shopping-center footprint"
    return f"inside the {area.name} extent"


def permits_in_area(frame: pd.DataFrame, area: Area) -> pd.DataFrame:
    """Permits whose recorded coordinates fall inside ``area``. Others are omitted."""
    mask = valid_coordinates(frame)
    if not mask.any():
        return frame.iloc[0:0].copy()
    inside = [
        point_in_geometry(lon, lat, area.geometry)
        for lon, lat in zip(frame.loc[mask, "longitude"], frame.loc[mask, "latitude"], strict=True)
    ]
    chosen = mask.copy()
    chosen.loc[mask] = inside
    return frame.loc[chosen].copy()
