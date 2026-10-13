"""Download UW-area boundaries used by the map.

Neighborhood polygons come from the Seattle City Clerk's Neighborhood Map Atlas
(unofficial indexing boundaries, not a legal neighborhood map). University Village
is the OpenStreetMap retail footprint, because the atlas does not name it.

Run: python scripts/fetch_uw_areas.py
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "reference" / "uw_areas.geojson"
CLERK = "https://gisdata.seattle.gov/spugis/rest/services/pub/SP_UTIL_Locate_Print/MapServer/263/query"
UA = "HuskyQuiet/0.2 (educational local project)"


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    where = "S_HOOD IN ('University District','Wallingford','Ravenna','Roosevelt')"
    url = CLERK + "?" + urllib.parse.urlencode(
        {"where": where, "outFields": "S_HOOD", "outSR": "4326", "f": "geojson"}
    )
    data = get_json(url)
    features = []
    for feat in data["features"]:
        name = feat["properties"]["S_HOOD"]
        geom = feat["geometry"]
        ring = geom["coordinates"][0]
        lon, lat = ring[0]
        if not (-123 < lon < -122 and 47 < lat < 48):
            raise SystemExit(f"{name} coordinates are not WGS84: {lon}, {lat}")
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "kind": "neighborhood",
                    "source": "Seattle City Clerk Neighborhood Map Atlas",
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    time.sleep(1.1)
    q = urllib.parse.urlencode(
        {
            "q": "University Village, Seattle, Washington",
            "format": "jsonv2",
            "polygon_geojson": 1,
            "limit": 1,
        }
    )
    hit = get_json("https://nominatim.openstreetmap.org/search?" + q)[0]
    geom = hit["geojson"]
    if geom["type"] != "Polygon":
        raise SystemExit(f"unexpected University Village geometry {geom['type']}")
    features.append(
        {
            "type": "Feature",
            "properties": {
                "name": "University Village",
                "kind": "place",
                "source": f"OpenStreetMap via Nominatim (osm {hit.get('osm_type')} {hit.get('osm_id')})",
            },
            "geometry": geom,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(features)} features)")


if __name__ == "__main__":
    main()
