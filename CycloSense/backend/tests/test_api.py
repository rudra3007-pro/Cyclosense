"""
CycloSense — Quick backend smoke test
Run from backend/ directory: python tests/test_api.py
Starts the Flask app and hits each endpoint.
"""

import sys
import os
import json
import threading
import time
import io

# ── Make sure backend/ is on sys.path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app

# ── Create app ────────────────────────────────────────────────────────────────
print("Creating app and loading models …")
app = create_app()
client = app.test_client()
print("Done.\n")

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []


def check(label, resp, expected_status=200, check_keys=None):
    ok = resp.status_code == expected_status
    data = resp.get_json()
    if check_keys:
        for k in check_keys:
            if k not in data:
                ok = False
                errors.append(f"{label}: missing key '{k}' in response")
    icon = PASS if ok else FAIL
    print(f"  {icon} {label} — HTTP {resp.status_code}")
    if not ok and not check_keys:
        errors.append(f"{label}: expected {expected_status}, got {resp.status_code}")
    return data


print("=" * 60)
print("SMOKE TESTS")
print("=" * 60)

# ── 1. Root ───────────────────────────────────────────────────────────────────
print("\n[1] Root endpoint")
r = client.get("/")
check("GET /", r, 200, ["service", "endpoints"])

# ── 2. Health ─────────────────────────────────────────────────────────────────
print("\n[2] Model health check")
r = client.get("/api/predict/health")
d = check("GET /api/predict/health", r, 200, ["status", "models_loaded"])
print(f"     models_loaded = {d.get('models_loaded')}")

# ── 3. Basins ─────────────────────────────────────────────────────────────────
print("\n[3] Map basins")
r = client.get("/api/map/basins")
check("GET /api/map/basins", r, 200, ["domain"])

# ── 4. Forecast (Model 3) — uses only tabular data, easiest to test ───────────
print("\n[4] Forecast endpoint (Model 3 — LSTM)")
track = [
    {"lat": 10.2 + i * 0.3, "lon": 80.1 + i * 0.4, "wind_speed": 65.0 + i * 3}
    for i in range(8)
]
r = client.post(
    "/api/predict/forecast",
    data=json.dumps({"track": track}),
    content_type="application/json",
)
d = check("POST /api/predict/forecast", r, 200, ["forecast", "all_steps"])
if d.get("forecast"):
    print("     Forecast steps:")
    for pt in d["forecast"]:
        print(f"       +{pt['hours']}h → lat={pt['lat']}, lon={pt['lon']}, wind={pt['wind_speed']} km/h")

# ── 5. Forecast — bad input ───────────────────────────────────────────────────
print("\n[5] Forecast — bad input (too few points)")
r = client.post(
    "/api/predict/forecast",
    data=json.dumps({"track": track[:4]}),
    content_type="application/json",
)
check("POST /api/predict/forecast (bad input)", r, 400, ["error"])

# ── 6. Map track GeoJSON ──────────────────────────────────────────────────────
print("\n[6] Map track GeoJSON")
forecast_pts = [
    {"hours": 3,  "lat": 11.5, "lon": 81.4, "wind_speed": 80},
    {"hours": 6,  "lat": 12.2, "lon": 82.1, "wind_speed": 88},
    {"hours": 12, "lat": 13.5, "lon": 83.5, "wind_speed": 100},
    {"hours": 24, "lat": 15.8, "lon": 85.2, "wind_speed": 118},
]
r = client.post(
    "/api/map/track",
    data=json.dumps({
        "current":  {"lat": 10.8, "lon": 80.6},
        "forecast": forecast_pts,
        "category": "SCS",
    }),
    content_type="application/json",
)
d = check("POST /api/map/track", r, 200, ["type", "features"])
if d.get("features"):
    print(f"     GeoJSON features: {len(d['features'])} (track + markers)")

# ── 7. Map impact zones ───────────────────────────────────────────────────────
print("\n[7] Map impact zones")
r = client.post(
    "/api/map/impact",
    data=json.dumps({"forecast": forecast_pts, "category": "SCS"}),
    content_type="application/json",
)
check("POST /api/map/impact", r, 200, ["features"])

# ── 8. Detect — no image (error) ──────────────────────────────────────────────
print("\n[8] Detect — missing image (expect 400)")
r = client.post("/api/predict/detect")
check("POST /api/predict/detect (no image)", r, 400, ["error"])

# ── 9. Classify — no image (error) ───────────────────────────────────────────
print("\n[9] Classify — missing image (expect 400)")
r = client.post("/api/predict/classify")
check("POST /api/predict/classify (no image)", r, 400, ["error"])

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"RESULT: {len(errors)} FAILURE(S)")
    for e in errors:
        print(f"  {FAIL} {e}")
    sys.exit(1)
else:
    print("RESULT: ALL TESTS PASSED [OK]")
    sys.exit(0)
