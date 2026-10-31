"""
CycloSense — Flask Application Factory
"""

import logging
from flask import Flask
from flask_cors import CORS


def create_app():
    """
    Create and configure the Flask application.
    - Loads all ML models once at startup.
    - Registers all Blueprints.
    - Enables CORS for the frontend dev server.
    """
    app = Flask(__name__)

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ── CORS (allow requests from any origin during development) ──────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Load ML models (runs once; cached as module globals) ─────────────────
    logger.info("Loading CycloSense models …")
    from app.services.model_loader import load_all_models
    load_all_models()
    logger.info("All models ready.")

    # ── Register Blueprints ───────────────────────────────────────────────────
    from app.routes.predict import predict_bp
    from app.routes.map import map_bp
    from app.routes.ai import ai_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(map_bp)
    app.register_blueprint(ai_bp)

    # ── Root health check ─────────────────────────────────────────────────────
    @app.get("/")
    def index():
        from flask import jsonify
        return jsonify({
            "service": "CycloSense API",
            "version": "1.0.0",
            "endpoints": {
                "health":      "GET  /api/predict/health",
                "test_models": "GET  /api/predict/test-models",
                "detect":      "POST /api/predict/detect",
                "classify":    "POST /api/predict/classify",
                "forecast":    "POST /api/predict/forecast",
                "full":        "POST /api/predict/full",
                "track_geo":   "POST /api/map/track",
                "impact":      "POST /api/map/impact",
                "basins":      "GET  /api/map/basins",
                "ai_status":   "GET  /api/ai/status",
                "ask_ai":      "POST /api/ai/ask",
            }
        }), 200

    logger.info("CycloSense Flask app created successfully.")
    return app
