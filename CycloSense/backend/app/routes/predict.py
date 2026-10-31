"""
CycloSense — Prediction Routes
Exposes the three-model inference pipeline via a Flask Blueprint.

Endpoints
---------
POST /api/predict/detect       – Model 1: YOLO cyclone detection
POST /api/predict/classify     – Model 2: EfficientNet intensity classification
POST /api/predict/forecast     – Model 3: LSTM track & intensity forecasting
POST /api/predict/full         – Full pipeline: image → detect → classify → forecast
GET  /api/predict/health       – Liveness check
GET  /api/predict/test-models  – Diagnostics: verify all 3 models are loaded and generating outputs
"""

import base64
import io
import logging
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from flask import Blueprint, request, jsonify

from app.services.model_loader import (
    get_model1, get_model2, get_model3, CLASSES, LSTM_SEQ_LEN
)
from app.utils.image_utils import (
    bytes_to_pil, pil_to_tensor, crop_from_bbox, bbox_to_geocentre
)

logger = logging.getLogger(__name__)
predict_bp = Blueprint("predict", __name__, url_prefix="/api/predict")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run YOLO on a PIL image
# ──────────────────────────────────────────────────────────────────────────────

def _run_detection(img, basin: str = "bay_of_bengal"):
    """
    Run YOLOv8 on a PIL image.

    Returns a list of detections, each as:
        {
          "bbox":       [x_center, y_center, width, height],  # pixels
          "confidence": float,
          "geo_centre": {"lat": float, "lon": float}
        }
    """
    model1 = get_model1()
    results = model1(img, verbose=False)[0]

    detections = []
    if results.boxes is None or len(results.boxes) == 0:
        return detections, results

    for box in results.boxes:
        xywh = box.xywh[0].tolist()          # [x_c, y_c, w, h] in pixels
        conf = float(box.conf[0])
        geo  = bbox_to_geocentre(xywh, img.width, img.height, basin=basin)
        detections.append({
            "bbox":       [round(v, 2) for v in xywh],
            "confidence": round(conf, 4),
            "geo_centre": geo,
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, results


def _annotate_image(img, results):
    """
    Draw YOLO bounding boxes on the image.
    Uses results.plot() for the authentic YOLO training output look,
    with a PIL fallback if needed.
    Returns a base64-encoded PNG string ready for an <img src=...> tag.
    """
    from PIL import Image

    annotated_pil = None
    try:
        if results is not None and results.boxes is not None and len(results.boxes) > 0:
            # Ultralytics results.plot() returns BGR numpy array
            plot_bgr = results.plot()
            plot_rgb = plot_bgr[:, :, ::-1]
            annotated_pil = Image.fromarray(plot_rgb)
    except Exception as e:
        logger.warning("Ultralytics plot() failed, falling back to PIL: %s", e)

    if annotated_pil is None:
        annotated_pil = img.copy()
        draw = ImageDraw.Draw(annotated_pil)
        w, h = annotated_pil.size
        box_color = (0, 212, 255)
        thick = max(2, int(min(w, h) / 200))
        if results is not None and results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=thick)
                label = f"Cyclone {conf*100:.1f}%"
                try:
                    font = ImageFont.truetype("arial.ttf", size=max(12, int(min(w, h) / 40)))
                except Exception:
                    font = ImageFont.load_default()
                bbox_txt = draw.textbbox((x1, y1), label, font=font)
                pad = 3
                draw.rectangle(
                    [bbox_txt[0]-pad, bbox_txt[1]-pad, bbox_txt[2]+pad, bbox_txt[3]+pad],
                    fill=(0, 212, 255)
                )
                draw.text((x1, y1), label, fill=(5, 13, 26), font=font)

    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run EfficientNet on a PIL image
# ──────────────────────────────────────────────────────────────────────────────

def _run_classification(img):
    """
    Run EfficientNet-B0 on a cropped PIL image.

    Returns:
        {
          "category":     str,          # e.g. "VSCS"
          "category_index": int,
          "probabilities": {class: prob, ...},
          "confidence":   float         # probability of predicted class
        }
    """
    model2 = get_model2()
    tensor = pil_to_tensor(img).to(DEVICE)

    with torch.no_grad():
        logits = model2(tensor)                        # (1, 5)
        probs  = torch.softmax(logits, dim=1)[0]       # (5,)

    pred_idx  = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    prob_dict  = {cls: round(float(probs[i]), 4) for i, cls in enumerate(CLASSES)}

    return {
        "category":       pred_class,
        "category_index": pred_idx,
        "probabilities":  prob_dict,
        "confidence":     round(float(probs[pred_idx]), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run LSTM forecasting
# ──────────────────────────────────────────────────────────────────────────────

def _run_forecast(track_sequence: list):
    """
    Run the 2-layer LSTM to generate a multi-step recursive forecast.

    Args:
        track_sequence: list of 8 dicts, each {"lat": float, "lon": float, "wind_speed": float}
                        Ordered oldest → newest (3-hr intervals).

    Returns:
        {
          "forecast": [
            {"step": 1, "hours": 3,  "lat": float, "lon": float, "wind_speed": float},
            {"step": 2, "hours": 6,  ...},
            {"step": 4, "hours": 12, ...},
            {"step": 8, "hours": 24, ...},
          ],
          "all_steps": [ ... ]   # all 8 recursive steps
        }
    """
    model3, scaler = get_model3()

    # Build (8, 3) numpy array
    seq_np = np.array(
        [[p["lat"], p["lon"], p["wind_speed"]] for p in track_sequence],
        dtype=np.float32,
    )  # (8, 3)

    # Normalise with the training scaler
    seq_scaled = scaler.transform(seq_np)           # (8, 3)

    # Recursive multi-step forecast: 8 steps = +3h … +24h
    current_seq = seq_scaled.copy()                 # (8, 3)
    all_steps   = []

    with torch.no_grad():
        for step in range(1, 9):
            inp    = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 8, 3)
            out    = model3(inp)                                  # (1, 3) — normalised
            pred_n = out[0].cpu().numpy()                         # (3,)

            # Inverse-transform to real-world units
            pred_real = scaler.inverse_transform(pred_n.reshape(1, -1))[0]  # (3,)
            lat, lon, wind = float(pred_real[0]), float(pred_real[1]), float(pred_real[2])

            all_steps.append({
                "step":       step,
                "hours":      step * 3,
                "lat":        round(lat, 4),
                "lon":        round(lon, 4),
                "wind_speed": round(max(wind, 0.0), 2),   # clip negative wind
            })

            # Slide the window: drop oldest, append new prediction
            current_seq = np.vstack([current_seq[1:], pred_n])

    # Key horizons only
    key_steps = [s for s in all_steps if s["step"] in {1, 2, 4, 8}]

    return {"forecast": key_steps, "all_steps": all_steps}


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@predict_bp.get("/health")
def health():
    """Liveness check — confirms models are loaded. Pass ?test=true to also run dummy inference."""
    if request.args.get("test", "false").lower() in ("true", "1", "yes"):
        return test_models()
    try:
        get_model1(); get_model2(); get_model3()
        return jsonify({"status": "ok", "models_loaded": True}), 200
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@predict_bp.get("/test-models")
def test_models():
    """
    Diagnostic route to verify that all 3 models are loaded and actively generating outputs.
    Feeds synthetic test inputs to:
      1. Model 1 (YOLOv8 Detection)
      2. Model 2 (EfficientNet-B0 Classification)
      3. Model 3 (LSTM Forecast)
    Returns latency, status, and sample outputs for each model.
    """
    results = {
        "status": "operational",
        "timestamp": time.time(),
        "models": {}
    }
    all_ok = True

    # 1. Test Model 1 (YOLO)
    t0 = time.perf_counter()
    try:
        m1 = get_model1()
        dummy_img = Image.new("RGB", (640, 640), color=(15, 25, 35))
        det, yolo_res = _run_detection(dummy_img)
        t_m1 = round((time.perf_counter() - t0) * 1000, 2)
        results["models"]["model1_yolo"] = {
            "loaded": True,
            "generating_output": True,
            "latency_ms": t_m1,
            "details": f"Inference executed successfully. Detections found: {len(det)}"
        }
    except Exception as e:
        all_ok = False
        results["models"]["model1_yolo"] = {
            "loaded": False,
            "generating_output": False,
            "error": str(e)
        }

    # 2. Test Model 2 (EfficientNet)
    t0 = time.perf_counter()
    try:
        m2 = get_model2()
        dummy_crop = Image.new("RGB", (224, 224), color=(30, 40, 50))
        cls_res = _run_classification(dummy_crop)
        t_m2 = round((time.perf_counter() - t0) * 1000, 2)
        results["models"]["model2_efficientnet"] = {
            "loaded": True,
            "generating_output": True,
            "latency_ms": t_m2,
            "sample_output": {
                "predicted_category": cls_res["category"],
                "confidence": cls_res["confidence"],
                "probabilities": cls_res["probabilities"]
            }
        }
    except Exception as e:
        all_ok = False
        results["models"]["model2_efficientnet"] = {
            "loaded": False,
            "generating_output": False,
            "error": str(e)
        }

    # 3. Test Model 3 (LSTM)
    t0 = time.perf_counter()
    try:
        m3 = get_model3()
        dummy_track = [
            {"lat": round(12.0 + i * 0.3, 2), "lon": round(85.0 - i * 0.2, 2), "wind_speed": round(65.0 + i * 5, 1)}
            for i in range(LSTM_SEQ_LEN)
        ]
        fcst_res = _run_forecast(dummy_track)
        t_m3 = round((time.perf_counter() - t0) * 1000, 2)
        results["models"]["model3_lstm"] = {
            "loaded": True,
            "generating_output": True,
            "latency_ms": t_m3,
            "sample_output": {
                "forecast_steps": len(fcst_res["all_steps"]),
                "key_horizons": [
                    {f"+{s['hours']}h": f"{s['lat']}°N, {s['lon']}°E ({s['wind_speed']} km/h)"}
                    for s in fcst_res["forecast"]
                ]
            }
        }
    except Exception as e:
        all_ok = False
        results["models"]["model3_lstm"] = {
            "loaded": False,
            "generating_output": False,
            "error": str(e)
        }

    results["all_operational"] = all_ok
    results["status"] = "operational" if all_ok else "degraded"
    status_code = 200 if all_ok else 500
    return jsonify(results), status_code


# ── Route 1: Detection ────────────────────────────────────────────────────────

@predict_bp.post("/detect")
def detect():
    """
    Detect cyclones in a satellite image.

    Request:  multipart/form-data  with field 'image' (JPEG / PNG).
    Response: JSON with list of detections.

    Example response:
    {
      "cyclone_detected": true,
      "count": 1,
      "detections": [
        {
          "bbox": [320.5, 240.1, 150.0, 120.0],
          "confidence": 0.812,
          "geo_centre": {"lat": 14.2, "lon": 82.6}
        }
      ]
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as 'image' field."}), 400

    basin = request.form.get("basin", request.args.get("basin", "bay_of_bengal"))
    img_bytes = request.files["image"].read()
    try:
        img              = bytes_to_pil(img_bytes)
        detections, yolo_results = _run_detection(img, basin=basin)
        annotated        = _annotate_image(img.copy(), yolo_results)
    except Exception as e:
        logger.exception("Detection failed")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "cyclone_detected":  len(detections) > 0,
        "count":             len(detections),
        "detections":        detections,
        "annotated_image":   annotated,   # base64 PNG with bbox drawn
    }), 200


# ── Route 2: Classification ───────────────────────────────────────────────────

@predict_bp.post("/classify")
def classify():
    """
    Classify the intensity category of a cropped cyclone image.

    Request:  multipart/form-data  with field 'image' (cropped cyclone region).
    Response: JSON with predicted IMD category and per-class probabilities.

    Example response:
    {
      "category":       "VSCS",
      "category_index": 4,
      "confidence":     0.812,
      "probabilities":  {"CS": 0.02, "D": 0.01, "DD": 0.03, "SCS": 0.14, "VSCS": 0.81}
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as 'image' field."}), 400

    img_bytes = request.files["image"].read()
    try:
        img    = bytes_to_pil(img_bytes)
        result = _run_classification(img)
    except Exception as e:
        logger.exception("Classification failed")
        return jsonify({"error": str(e)}), 500

    return jsonify(result), 200


# ── Route 3: Forecast ─────────────────────────────────────────────────────────

@predict_bp.post("/forecast")
def forecast():
    """
    Forecast future cyclone track and intensity.

    Request: JSON body:
    {
      "track": [
        {"lat": 10.2, "lon": 80.1, "wind_speed": 65.0},
        ...   // exactly 8 points, oldest first, 3-hr intervals
      ]
    }

    Response:
    {
      "forecast": [
        {"step": 1, "hours": 3,  "lat": ..., "lon": ..., "wind_speed": ...},
        {"step": 2, "hours": 6,  ...},
        {"step": 4, "hours": 12, ...},
        {"step": 8, "hours": 24, ...}
      ],
      "all_steps": [ ... ]
    }
    """
    body = request.get_json(silent=True)
    if not body or "track" not in body:
        return jsonify({"error": "JSON body with 'track' array required."}), 400

    track = body["track"]
    if len(track) != LSTM_SEQ_LEN:
        return jsonify({
            "error": f"'track' must contain exactly {LSTM_SEQ_LEN} points "
                     f"(got {len(track)}). Each point: {{lat, lon, wind_speed}}."
        }), 400

    required_keys = {"lat", "lon", "wind_speed"}
    for i, pt in enumerate(track):
        if not required_keys.issubset(pt.keys()):
            return jsonify({
                "error": f"Point {i} missing keys. Expected: {required_keys}"
            }), 400

    try:
        result = _run_forecast(track)
    except Exception as e:
        logger.exception("Forecasting failed")
        return jsonify({"error": str(e)}), 500

    return jsonify(result), 200


# ── Route 4: Full Pipeline ────────────────────────────────────────────────────

@predict_bp.post("/full")
def full_pipeline():
    """
    Run the complete 3-model pipeline on a satellite image + track history.

    Request: multipart/form-data
        - 'image'  (required): full satellite image file
        - 'track'  (required): JSON string — list of 8 track points
                               [{"lat":..., "lon":..., "wind_speed":...}, ...]

    Response:
    {
      "detection":      { ... },   // Model 1 output (best detection)
      "classification": { ... },   // Model 2 output on cropped region
      "forecast":       { ... }    // Model 3 output
    }
    """
    import json

    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file."}), 400
    if "track" not in request.form:
        return jsonify({"error": "Missing 'track' JSON string in form data."}), 400

    img_bytes = request.files["image"].read()
    try:
        track = json.loads(request.form["track"])
    except Exception:
        return jsonify({"error": "'track' must be valid JSON."}), 400

    try:
        basin = request.form.get("basin", "bay_of_bengal")
        img   = bytes_to_pil(img_bytes)

        # ── Step 1: Detection ──────────────────────────────────────────────
        detections, yolo_results = _run_detection(img, basin=basin)
        annotated_img = _annotate_image(img.copy(), yolo_results)
        detection_result = {
            "cyclone_detected": len(detections) > 0,
            "count":            len(detections),
            "detections":       detections,
            "annotated_image":  annotated_img,
        }

        # ── Step 2: Classification (on best bbox crop, if detected) ────────
        classification_result = None
        if detections:
            best_bbox = detections[0]["bbox"]         # highest-confidence detection
            crop      = crop_from_bbox(img, best_bbox)
            classification_result = _run_classification(crop)

        # ── Step 3: Forecast ───────────────────────────────────────────────
        forecast_result = None
        if len(track) == LSTM_SEQ_LEN:
            forecast_result = _run_forecast(track)
        else:
            forecast_result = {
                "error": f"Track must have {LSTM_SEQ_LEN} points for forecasting."
            }

    except Exception as e:
        logger.exception("Full pipeline failed")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "detection":      detection_result,
        "classification": classification_result,
        "forecast":       forecast_result,
    }), 200
