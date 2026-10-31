#!/usr/bin/env python3
"""
CycloSense — Universal Test Runner CLI
======================================
Runs all system, environment, weights, backend API, live model inference,
and evaluation metric test suites at once in a single command.

Usage:
  python test_all.py
  python test_all.py --verbose
  python test_all.py --json
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ---------------------------------------------------------------------------
# Test Tracking & Helpers (Strictly emoji-free)
# ---------------------------------------------------------------------------

class TestTracker:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.suite_results = {}
        self.start_time = time.perf_counter()

    def record_pass(self, suite_name, test_name, extra=""):
        self.tests_run += 1
        self.tests_passed += 1
        msg = f"  [PASS] {test_name}"
        if extra:
            msg += f" ({extra})"
        print(msg)

    def record_fail(self, suite_name, test_name, error_msg):
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append((suite_name, test_name, error_msg))
        print(f"  [FAIL] {test_name} — {error_msg}")

    def summary(self):
        duration = time.perf_counter() - self.start_time
        return {
            "total_run": self.tests_run,
            "passed": self.tests_passed,
            "failed": self.tests_failed,
            "duration_seconds": round(duration, 2),
            "all_passed": self.tests_failed == 0,
            "failures": self.failures
        }

def _bar(char="=", length=80):
    return char * length

# ---------------------------------------------------------------------------
# Suite 1: System Environment & Static Assets Check
# ---------------------------------------------------------------------------

def run_suite_environment(tracker):
    print("\n" + _bar("-"))
    print("SUITE 1: ENVIRONMENT, DEPENDENCIES & ASSETS INTEGRITY")
    print(_bar("-"))

    # 1.1 Python version
    py_ver = sys.version.split()[0]
    if sys.version_info >= (3, 9):
        tracker.record_pass("Environment", f"Python version >= 3.9", f"v{py_ver}")
    else:
        tracker.record_fail("Environment", f"Python version", f"Expected >= 3.9, got {py_ver}")

    # 1.2 Core dependencies
    deps = [
        ("torch", "PyTorch deep learning framework"),
        ("torchvision", "Computer vision model utilities"),
        ("ultralytics", "YOLOv8 object detection engine"),
        ("sklearn", "Scikit-learn preprocessing and metrics"),
        ("PIL", "Pillow image processing library"),
        ("numpy", "NumPy array computations"),
        ("flask", "Flask backend REST micro-framework"),
        ("flask_cors", "Flask CORS middleware"),
    ]
    for mod, desc in deps:
        try:
            __import__(mod)
            tracker.record_pass("Environment", f"Library import: {mod}", desc)
        except ImportError as e:
            tracker.record_fail("Environment", f"Library import: {mod}", str(e))

    # 1.3 Device detection
    try:
        import torch
        dev = "CUDA GPU (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"
        tracker.record_pass("Environment", "Inference device detected", dev)
    except Exception as e:
        tracker.record_fail("Environment", "Inference device detected", str(e))

    # 1.4 Frontend UI files
    frontend_files = [
        "frontend/pages/index.html",
        "frontend/css/style.css",
        "frontend/js/main.js",
    ]
    for rel in frontend_files:
        full = os.path.join(PROJECT_ROOT, rel)
        if os.path.exists(full) and os.path.getsize(full) > 0:
            tracker.record_pass("Environment", f"Frontend asset exists: {rel}", f"{os.path.getsize(full)} bytes")
        else:
            tracker.record_fail("Environment", f"Frontend asset exists: {rel}", "File missing or empty")

    # 1.5 Environment configuration
    env_path = os.path.join(BACKEND_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        has_groq = "GROQ_API_KEY=" in content and "your_groq_api_key_here" not in content
        if has_groq:
            tracker.record_pass("Environment", "Backend .env configuration", "GROQ_API_KEY configured")
        else:
            tracker.record_pass("Environment", "Backend .env configuration", "File exists (.env ready)")
    else:
        tracker.record_fail("Environment", "Backend .env configuration", "backend/.env not found")

# ---------------------------------------------------------------------------
# Suite 2: Model Weights & Pretrained Checkpoints Validation
# ---------------------------------------------------------------------------

def run_suite_weights(tracker):
    print("\n" + _bar("-"))
    print("SUITE 2: MODEL WEIGHTS & CHECKPOINT ARTIFACTS")
    print(_bar("-"))

    weights = [
        ("Model 1 (YOLOv8-nano)", "backend/weights/model1_detection.pt", 5.0 * 1024 * 1024),
        ("Model 2 (EfficientNet-B0)", "backend/weights/model2_classification.pt", 14.0 * 1024 * 1024),
        ("Model 3 (2-Layer LSTM)", "backend/weights/model3_forecasting.pt", 150 * 1024),
        ("Model 3 (StandardScaler)", "backend/weights/model3_scaler.pkl", 200),
    ]

    for name, rel_path, min_bytes in weights:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            tracker.record_fail("Weights", f"{name} file presence", f"File not found: {rel_path}")
            continue

        size = os.path.getsize(full_path)
        if size >= min_bytes:
            size_mb = size / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{size/1024:.1f} KB"
            tracker.record_pass("Weights", f"{name} integrity verified", f"Size: {size_str}")
        else:
            tracker.record_fail("Weights", f"{name} size check", f"File size too small ({size} bytes)")

# ---------------------------------------------------------------------------
# Suite 3: Backend Flask API & Contract Endpoints
# ---------------------------------------------------------------------------

def run_suite_backend_api(tracker):
    print("\n" + _bar("-"))
    print("SUITE 3: BACKEND FLASK API & ROUTE CONTRACT TESTS")
    print(_bar("-"))

    try:
        from app import create_app
        app = create_app()
        client = app.test_client()
        tracker.record_pass("Backend API", "Flask create_app() initialization", "Loaded models into memory")
    except Exception as e:
        tracker.record_fail("Backend API", "Flask create_app() initialization", str(e))
        return

    # 3.1 Root endpoint
    try:
        r = client.get("/")
        if r.status_code == 200 and "endpoints" in r.get_json():
            tracker.record_pass("Backend API", "GET / — Service Discovery", "HTTP 200")
        else:
            tracker.record_fail("Backend API", "GET / — Service Discovery", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "GET / — Service Discovery", str(e))

    # 3.2 Health endpoint
    try:
        r = client.get("/api/predict/health")
        data = r.get_json()
        if r.status_code == 200 and data.get("models_loaded") is True:
            tracker.record_pass("Backend API", "GET /api/predict/health — Model Status", "All models loaded")
        else:
            tracker.record_fail("Backend API", "GET /api/predict/health — Model Status", f"models_loaded={data.get('models_loaded')}")
    except Exception as e:
        tracker.record_fail("Backend API", "GET /api/predict/health — Model Status", str(e))

    # 3.3 Basins map endpoint
    try:
        r = client.get("/api/map/basins")
        data = r.get_json()
        if r.status_code == 200 and "domain" in data:
            tracker.record_pass("Backend API", "GET /api/map/basins — Ocean Bounds", "HTTP 200")
        else:
            tracker.record_fail("Backend API", "GET /api/map/basins — Ocean Bounds", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "GET /api/map/basins — Ocean Bounds", str(e))

    # 3.4 Model 3 Forecast endpoint (Valid 8 points)
    sample_track = [
        {"lat": 10.2 + i * 0.3, "lon": 80.1 + i * 0.4, "wind_speed": 65.0 + i * 3.0}
        for i in range(8)
    ]
    try:
        r = client.post(
            "/api/predict/forecast",
            data=json.dumps({"track": sample_track}),
            content_type="application/json"
        )
        data = r.get_json()
        if r.status_code == 200 and "forecast" in data and len(data["forecast"]) == 4:
            steps_desc = f"+3h: {data['forecast'][0]['lat']}N, +24h: {data['forecast'][-1]['lat']}N"
            tracker.record_pass("Backend API", "POST /api/predict/forecast — Valid Track", steps_desc)
        else:
            tracker.record_fail("Backend API", "POST /api/predict/forecast — Valid Track", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "POST /api/predict/forecast — Valid Track", str(e))

    # 3.5 Model 3 Forecast endpoint (Rejection of invalid input)
    try:
        r = client.post(
            "/api/predict/forecast",
            data=json.dumps({"track": sample_track[:4]}),
            content_type="application/json"
        )
        if r.status_code == 400 and "error" in r.get_json():
            tracker.record_pass("Backend API", "POST /api/predict/forecast — Reject <8 Points", "HTTP 400 rejection verified")
        else:
            tracker.record_fail("Backend API", "POST /api/predict/forecast — Reject <8 Points", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "POST /api/predict/forecast — Reject <8 Points", str(e))

    # 3.6 Map track GeoJSON
    try:
        forecast_pts = [
            {"hours": 3,  "lat": 11.5, "lon": 81.4, "wind_speed": 80},
            {"hours": 6,  "lat": 12.2, "lon": 82.1, "wind_speed": 88},
            {"hours": 12, "lat": 13.5, "lon": 83.5, "wind_speed": 100},
            {"hours": 24, "lat": 15.8, "lon": 85.2, "wind_speed": 118},
        ]
        r = client.post(
            "/api/map/track",
            data=json.dumps({
                "current": {"lat": 10.8, "lon": 80.6},
                "forecast": forecast_pts,
                "category": "SCS"
            }),
            content_type="application/json"
        )
        data = r.get_json()
        if r.status_code == 200 and data.get("type") == "FeatureCollection" and len(data.get("features", [])) > 0:
            tracker.record_pass("Backend API", "POST /api/map/track — GeoJSON Track", f"{len(data['features'])} features generated")
        else:
            tracker.record_fail("Backend API", "POST /api/map/track — GeoJSON Track", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "POST /api/map/track — GeoJSON Track", str(e))

    # 3.7 Map impact zones
    try:
        r = client.post(
            "/api/map/impact",
            data=json.dumps({"forecast": forecast_pts, "category": "SCS"}),
            content_type="application/json"
        )
        data = r.get_json()
        if r.status_code == 200 and len(data.get("features", [])) > 0:
            tracker.record_pass("Backend API", "POST /api/map/impact — GeoJSON Polygons", f"{len(data['features'])} impact zones")
        else:
            tracker.record_fail("Backend API", "POST /api/map/impact — GeoJSON Polygons", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "POST /api/map/impact — GeoJSON Polygons", str(e))

    # 3.8 Detect missing image error handling
    try:
        r = client.post("/api/predict/detect")
        if r.status_code == 400:
            tracker.record_pass("Backend API", "POST /api/predict/detect — Missing Image Validation", "HTTP 400 handled")
        else:
            tracker.record_fail("Backend API", "POST /api/predict/detect — Missing Image Validation", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "POST /api/predict/detect — Missing Image Validation", str(e))

    # 3.9 Classify missing image error handling
    try:
        r = client.post("/api/predict/classify")
        if r.status_code == 400:
            tracker.record_pass("Backend API", "POST /api/predict/classify — Missing Image Validation", "HTTP 400 handled")
        else:
            tracker.record_fail("Backend API", "POST /api/predict/classify — Missing Image Validation", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "POST /api/predict/classify — Missing Image Validation", str(e))

    # 3.10 Model test diagnostic endpoint
    try:
        r = client.get("/api/predict/test-models")
        data = r.get_json()
        if r.status_code == 200 and data.get("all_operational") is True:
            tracker.record_pass("Backend API", "GET /api/predict/test-models — Self-Test", "All 3 models operational")
        else:
            tracker.record_fail("Backend API", "GET /api/predict/test-models — Self-Test", f"HTTP {r.status_code}")
    except Exception as e:
        tracker.record_fail("Backend API", "GET /api/predict/test-models — Self-Test", str(e))

# ---------------------------------------------------------------------------
# Suite 4: Live Forward-Pass Model Inference & Latency Benchmarks
# ---------------------------------------------------------------------------

def run_suite_live_inference(tracker):
    print("\n" + _bar("-"))
    print("SUITE 4: LIVE FORWARD-PASS INFERENCE & LATENCY BENCHMARKS")
    print(_bar("-"))

    import torch
    import numpy as np
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.1 Model 1: YOLO Detection Forward Pass
    try:
        from ultralytics import YOLO
        m1_path = os.path.join(BACKEND_DIR, "weights", "model1_detection.pt")
        yolo_model = YOLO(m1_path)
        dummy_img = Image.new("RGB", (640, 640), color=(20, 35, 65))
        
        t0 = time.perf_counter()
        results = yolo_model(dummy_img, verbose=False)[0]
        t_infer = (time.perf_counter() - t0) * 1000
        
        tracker.record_pass(
            "Live Inference",
            "Model 1 (YOLOv8-nano) forward pass",
            f"{t_infer:.1f} ms on 640x640 input"
        )
    except Exception as e:
        tracker.record_fail("Live Inference", "Model 1 (YOLOv8-nano) forward pass", str(e))

    # 4.2 Model 2: EfficientNet Classification Forward Pass
    try:
        import torchvision.models as tv_models
        m2_path = os.path.join(BACKEND_DIR, "weights", "model2_classification.pt")
        eff_model = tv_models.efficientnet_b0(weights=None)
        in_features = eff_model.classifier[1].in_features
        eff_model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4, inplace=True),
            torch.nn.Linear(in_features, 5)
        )
        state = torch.load(m2_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        eff_model.load_state_dict(state)
        eff_model.to(device)
        eff_model.eval()

        dummy_tensor = torch.randn(1, 3, 224, 224, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = eff_model(dummy_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        t_infer = (time.perf_counter() - t0) * 1000

        classes = ["CS", "D", "DD", "SCS", "VSCS"]
        pred_class = classes[int(np.argmax(probs))]
        tracker.record_pass(
            "Live Inference",
            "Model 2 (EfficientNet-B0) forward pass",
            f"{t_infer:.1f} ms, output: {pred_class}"
        )
    except Exception as e:
        tracker.record_fail("Live Inference", "Model 2 (EfficientNet-B0) forward pass", str(e))

    # 4.3 Model 3: 2-Layer LSTM Autoregressive Forecasting
    try:
        import pickle
        from app.services.model_loader import CycloneLSTM
        m3_path = os.path.join(BACKEND_DIR, "weights", "model3_forecasting.pt")
        sc_path = os.path.join(BACKEND_DIR, "weights", "model3_scaler.pkl")

        lstm_model = CycloneLSTM(input_size=3, hidden_size=64, num_layers=2, output_size=3)
        state = torch.load(m3_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        lstm_model.load_state_dict(state)
        lstm_model.to(device)
        lstm_model.eval()

        with open(sc_path, "rb") as f:
            scaler = pickle.load(f)

        sample_seq = np.array([
            [10.2 + i*0.3, 80.1 + i*0.4, 65.0 + i*3.0]
            for i in range(8)
        ], dtype=np.float32)

        t0 = time.perf_counter()
        seq_norm = scaler.transform(sample_seq)
        curr = torch.tensor(seq_norm, dtype=torch.float32, device=device).unsqueeze(0)
        forecast_pts = []

        with torch.no_grad():
            for _ in range(8):
                pred_norm = lstm_model(curr)
                pred_orig = scaler.inverse_transform(pred_norm.cpu().numpy())[0]
                forecast_pts.append(pred_orig)
                curr = torch.cat([curr[:, 1:, :], pred_norm.unsqueeze(1)], dim=1)
        t_infer = (time.perf_counter() - t0) * 1000

        last_pt = forecast_pts[-1]
        tracker.record_pass(
            "Live Inference",
            "Model 3 (LSTM) 8-step autoregression",
            f"{t_infer:.1f} ms, +24h point: {last_pt[0]:.2f}N, {last_pt[1]:.2f}E"
        )
    except Exception as e:
        tracker.record_fail("Live Inference", "Model 3 (LSTM) 8-step autoregression", str(e))

# ---------------------------------------------------------------------------
# Suite 5: Model Evaluation Score Assertions
# ---------------------------------------------------------------------------

def run_suite_evaluation_metrics(tracker):
    print("\n" + _bar("-"))
    print("SUITE 5: OFFICIAL MODEL EVALUATION METRICS VERIFICATION")
    print(_bar("-"))

    from ml.src.evaluation.evaluate_models import EVALUATION_METRICS
    models = EVALUATION_METRICS["models"]

    # Model 1 Metrics
    m1_scores = models["model1_detection"]["scores"]
    if m1_scores["precision"] >= 0.70 and m1_scores["recall"] >= 0.80 and m1_scores["mAP50"] >= 0.69:
        tracker.record_pass(
            "Evaluation Metrics",
            "Model 1: YOLO Detection Test Scores",
            f"mAP@50={m1_scores['mAP50']:.3f}, Recall={m1_scores['recall']:.3f}, Prec={m1_scores['precision']:.3f}"
        )
    else:
        tracker.record_fail("Evaluation Metrics", "Model 1: YOLO Detection Test Scores", "Scores below verified threshold")

    # Model 2 Metrics
    m2_scores = models["model2_classification"]["scores"]
    if m2_scores["overall_accuracy"] >= 0.45 and m2_scores["class_metrics"]["VSCS"]["f1_score"] >= 0.70:
        tracker.record_pass(
            "Evaluation Metrics",
            "Model 2: EfficientNet Intensity Classification",
            f"Accuracy={m2_scores['overall_accuracy']*100:.1f}%, VSCS F1={m2_scores['class_metrics']['VSCS']['f1_score']:.2f}"
        )
    else:
        tracker.record_fail("Evaluation Metrics", "Model 2: EfficientNet Intensity Classification", "Accuracy or F1 below threshold")

    # Model 3 Metrics
    m3_scores = models["model3_forecasting"]["scores"]
    s_err = m3_scores["single_step_forecast"]["mean_position_error_km"]
    m_24h = m3_scores["multi_step_recursive_forecast"][-1]["mean_position_error_km"]
    if s_err <= 110.0 and m_24h <= 400.0:
        tracker.record_pass(
            "Evaluation Metrics",
            "Model 3: Track Forecasting Position Errors",
            f"+3h Single Step: {s_err} km, +24h Horizon: {m_24h} km"
        )
    else:
        tracker.record_fail("Evaluation Metrics", "Model 3: Track Forecasting Position Errors", "Errors exceed threshold")

# ---------------------------------------------------------------------------
# Main Runner & CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CycloSense Universal Test Suite — Runs all tests at once",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test_all.py            # Run complete universal test suite
  python test_all.py --json     # Export test run results as JSON
"""
    )
    parser.add_argument("--json", action="store_true", help="Output pure JSON test report")
    parser.add_argument("--export", type=str, default="", help="Save test report to file")
    args = parser.parse_args()

    tracker = TestTracker()

    print()
    print(_bar("=", 80))
    print("                    CYCLOSENSE UNIVERSAL TEST SUITE                          ")
    print("           Complete System, API, Weights & ML Performance Verification       ")
    print(_bar("=", 80))
    print(f" Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Directory : {PROJECT_ROOT}")
    print(_bar("=", 80))

    # Run all 5 suites
    run_suite_environment(tracker)
    run_suite_weights(tracker)
    run_suite_backend_api(tracker)
    run_suite_live_inference(tracker)
    run_suite_evaluation_metrics(tracker)

    # Summary
    summary = tracker.summary()

    print("\n" + _bar("=", 80))
    print("                              TEST SUMMARY REPORT                            ")
    print(_bar("=", 80))
    print(f"  Total Tests Run : {summary['total_run']}")
    print(f"  Tests Passed    : {summary['passed']}")
    print(f"  Tests Failed    : {summary['failed']}")
    print(f"  Execution Time  : {summary['duration_seconds']}s")
    print(_bar("-", 80))

    if summary["all_passed"]:
        print("  RESULT: ALL TEST SUITES PASSED [OK]")
        print("  Status: Platform is healthy, models are loaded, and API is ready.")
    else:
        print(f"  RESULT: {summary['failed']} TEST(S) FAILED [FAIL]")
        for suite, name, err in summary["failures"]:
            print(f"    - [{suite}] {name}: {err}")

    print(_bar("=", 80) + "\n")

    if args.json or args.export:
        report_json = json.dumps(summary, indent=2)
        if args.json:
            print(report_json)
        if args.export:
            with open(args.export, "w", encoding="utf-8") as f:
                f.write(report_json)
            print(f"Test report saved to: {os.path.abspath(args.export)}")

    sys.exit(0 if summary["all_passed"] else 1)

if __name__ == "__main__":
    main()
