"""
CycloSense — Map / GIS Routes
Exposes geographic data for the Leaflet dashboard.

Endpoints
---------
GET  /api/map/track/<cyclone_id>   – Return predicted track as GeoJSON LineString
POST /api/map/impact               – Return risk-zone GeoJSON from forecast + intensity
GET  /api/map/basins               – Return Indian Ocean basin boundary GeoJSON
"""

import logging
import math

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)
map_bp = Blueprint("map", __name__, url_prefix="/api/map")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _uncertainty_radius_km(hours: int) -> float:
    """
    Approximate uncertainty cone radius at a given forecast horizon.
    Grows linearly — fine-tuned from Model 3 multi-step errors.
    """
    error_map = {3: 130, 6: 160, 12: 240, 24: 400}
    if hours in error_map:
        return float(error_map[hours])
    # Linear interpolation for other horizons
    return 130.0 + (hours / 24.0) * (400.0 - 130.0)


def _km_to_deg(km: float) -> float:
    """Rough conversion: 1 degree ≈ 111 km."""
    return km / 111.0


def _intensity_to_risk(category: str) -> str:
    """Map IMD category to risk level."""
    mapping = {
        "D":    "LOW",
        "DD":   "LOW",
        "CS":   "MODERATE",
        "SCS":  "HIGH",
        "VSCS": "VERY HIGH",
    }
    return mapping.get(category, "UNKNOWN")


def _intensity_to_color(risk: str) -> str:
    colors = {
        "LOW":       "#2ecc71",
        "MODERATE":  "#f1c40f",
        "HIGH":      "#e67e22",
        "VERY HIGH": "#e74c3c",
        "UNKNOWN":   "#95a5a6",
    }
    return colors.get(risk, "#95a5a6")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@map_bp.post("/track")
def track_geojson():
    """
    Convert a forecast result into a GeoJSON FeatureCollection for Leaflet.

    Request JSON:
    {
      "current": {"lat": float, "lon": float},
      "forecast": [
        {"hours": 3,  "lat": ..., "lon": ..., "wind_speed": ...},
        {"hours": 6,  ...},
        {"hours": 12, ...},
        {"hours": 24, ...}
      ],
      "category": "VSCS"   // optional — from Model 2
    }

    Response: GeoJSON FeatureCollection with:
      - LineString of the predicted track
      - Point markers for each forecast position (with uncertainty radius)
    """
    body = request.get_json(silent=True)
    if not body or "forecast" not in body or "current" not in body:
        return jsonify({"error": "JSON with 'current' and 'forecast' required."}), 400

    current  = body["current"]
    forecast = body["forecast"]
    category = body.get("category", "CS")
    risk     = _intensity_to_risk(category)
    color    = _intensity_to_color(risk)

    # Build coordinate list: current → forecast points
    coords = [[current["lon"], current["lat"]]]
    coords += [[pt["lon"], pt["lat"]] for pt in forecast]

    features = []

    # ── Main track LineString ──────────────────────────────────────────────
    features.append({
        "type": "Feature",
        "geometry": {
            "type":        "LineString",
            "coordinates": coords,
        },
        "properties": {
            "type":     "predicted_track",
            "category": category,
            "risk":     risk,
            "color":    color,
            "dash":     "dashed",
        },
    })

    # ── Current position marker ────────────────────────────────────────────
    features.append({
        "type": "Feature",
        "geometry": {
            "type":        "Point",
            "coordinates": [current["lon"], current["lat"]],
        },
        "properties": {
            "type":       "current_position",
            "label":      "Current Position",
            "category":   category,
            "risk":       risk,
            "color":      color,
            "hours":      0,
        },
    })

    # ── Forecast point markers + uncertainty circles ───────────────────────
    for pt in forecast:
        radius_km  = _uncertainty_radius_km(pt["hours"])
        radius_deg = _km_to_deg(radius_km)

        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": [pt["lon"], pt["lat"]],
            },
            "properties": {
                "type":             "forecast_point",
                "label":            f"+{pt['hours']}h",
                "hours":            pt["hours"],
                "wind_speed_kts":   round(pt["wind_speed"] * 0.539957, 1),  # km/h → kts
                "wind_speed_kmh":   pt["wind_speed"],
                "uncertainty_km":   radius_km,
                "uncertainty_deg":  round(radius_deg, 4),
                "color":            color,
            },
        })

    return jsonify({
        "type":     "FeatureCollection",
        "features": features,
        "meta": {
            "category":   category,
            "risk_level": risk,
            "color":      color,
        }
    }), 200


@map_bp.post("/impact")
def impact_zones():
    """
    Generate a simple impact risk zone GeoJSON around the forecast track.

    Request JSON:
    {
      "forecast": [ {"lat": ..., "lon": ..., "wind_speed": ..., "hours": ...} ],
      "category": "SCS"
    }

    Response: GeoJSON FeatureCollection with colour-coded impact polygons
    (approximated as bounding boxes around each forecast point).
    """
    body = request.get_json(silent=True)
    if not body or "forecast" not in body:
        return jsonify({"error": "JSON with 'forecast' required."}), 400

    forecast = body["forecast"]
    category = body.get("category", "CS")
    risk     = _intensity_to_risk(category)
    color    = _intensity_to_color(risk)

    features = []
    for pt in forecast:
        radius_deg = _km_to_deg(_uncertainty_radius_km(pt["hours"]))
        lat, lon   = pt["lat"], pt["lon"]

        # Simple bounding-box polygon for the impact zone
        polygon_coords = [[
            [lon - radius_deg, lat - radius_deg],
            [lon + radius_deg, lat - radius_deg],
            [lon + radius_deg, lat + radius_deg],
            [lon - radius_deg, lat + radius_deg],
            [lon - radius_deg, lat - radius_deg],
        ]]

        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "Polygon",
                "coordinates": polygon_coords,
            },
            "properties": {
                "type":       "impact_zone",
                "hours":      pt["hours"],
                "risk":       risk,
                "color":      color,
                "opacity":    0.15,
                "wind_speed": pt["wind_speed"],
            },
        })

    return jsonify({
        "type":     "FeatureCollection",
        "features": features,
        "meta":     {"risk_level": risk, "category": category},
    }), 200


@map_bp.get("/basins")
def basins():
    """
    Return a simple GeoJSON bounding box for the North Indian Ocean monitoring domain.
    Clients use this to set the initial Leaflet map view.
    """
    domain = {
        "lat_min":  0.0,  "lat_max":  35.0,
        "lon_min": 40.0,  "lon_max": 100.0,
        "centre":  {"lat": 15.0, "lon": 80.0},
        "zoom":    5,
    }
    return jsonify({"domain": domain}), 200
