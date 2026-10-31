"""
CycloSense — AI Action Planner & Cyclone Assistant Route
Powered by Groq API (Llama-3.3-70B / Llama-3.1-8B)

Endpoints
---------
GET  /api/ai/status   – Liveness & Groq API configuration status
POST /api/ai/ask      – Decision assistant query grounded in cyclone telemetry
"""

import os
import logging
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)
ai_bp = Blueprint("ai", __name__, url_prefix="/api/ai")

DEFAULT_MODEL = "qwen/qwen3.8-27b"
FALLBACK_MODEL = "openai/gpt-oss-120b"


def _get_api_key(client_provided_key: str = None) -> str:
    """Retrieve Groq API key from environment or backend/.env file."""
    if client_provided_key and client_provided_key.strip():
        return client_provided_key.strip()

    # Check environment variable
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    # Absolute path to backend directory (ai.py -> routes -> app -> backend)
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root_dir = os.path.dirname(backend_dir)

    candidate_paths = [
        os.path.join(backend_dir, ".env"),
        os.path.join(root_dir, ".env"),
        os.path.join(root_dir, "backend", ".env"),
        "backend/.env",
        ".env",
    ]

    for env_path in candidate_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GROQ_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass

    return ""


def _build_system_prompt(role: str, context: dict) -> str:
    """Ground the LLM with live CycloSense model outputs and selected user role."""
    detected = context.get("cyclone_detected", False)
    category = context.get("category", "Unknown")
    confidence = context.get("confidence", 0)
    pos = context.get("current_position", {})
    lat = pos.get("lat") or context.get("lat", "Unknown")
    lon = pos.get("lon") or context.get("lon", "Unknown")
    wind_kmh = context.get("wind_speed_kmh") or context.get("wind_speed", "Unknown")
    risk_level = context.get("risk_level", "MODERATE")
    forecast = context.get("forecast", [])

    forecast_summary = ""
    if forecast:
        steps = []
        for pt in forecast:
            h = pt.get("hours")
            p_lat = pt.get("lat")
            p_lon = pt.get("lon")
            w = pt.get("wind_speed")
            steps.append(f"  • +{h}h: Lat {p_lat}°N, Lon {p_lon}°E, Wind: {w} km/h")
        forecast_summary = "\n".join(steps)
    else:
        forecast_summary = "  • No trajectory forecast loaded yet."

    role_guidelines = {
        "resident": (
            "Target audience: Coastal and inland residents. Focus on household safety, "
            "evacuation timelines, shelter locations, emergency supply kits (food, potable water, batteries), "
            "and immediate precautions against flooding, flying debris, and storm surges."
        ),
        "fisherman": (
            "Target audience: Fishermen, boat owners, and maritime operators. Emphasize total suspension "
            "of fishing operations, immediate return to harbour/shore, securing boats and fishing gear, "
            "monitoring wave heights and marine gale warnings."
        ),
        "port_operator": (
            "Target audience: Port, shipping, and coastal infrastructure authorities. Focus on berthing "
            "suspension, cargo crane lockdown, barge anchoring, vessel de-berthing, communication backup, "
            "and structural protection against gale-force winds and surges."
        ),
        "disaster_official": (
            "Target audience: NDRF, SDRF, district collectors, and emergency response teams. Provide high-level "
            "operational guidance: priority evacuation zones, hospital backup power readiness, deployment of rescue boats, "
            "resource pre-positioning along landfall corridors, and multi-agency coordination."
        ),
        "general": (
            "Target audience: General public and media. Provide clear, calm, authoritative cyclone summary, "
            "severity explanation, expected impacts, and verified emergency helplines advice."
        )
    }.get(role.lower(), "Target audience: General public. Provide authoritative, safety-first cyclone guidance.")

    prompt = f"""You are the CycloSense AI Action Planner & Decision Assistant, an expert meteorological emergency intelligence system for the North Indian Ocean basin (Bay of Bengal and Arabian Sea), compliant with India Meteorological Department (IMD) standards.

CURRENT CYCLOSENSE REAL-TIME TELEMETRY:
----------------------------------------
- Cyclone Detected: {detected}
- IMD Category: {category} (Confidence: {confidence*100:.1f}%)
- Current Center Position: {lat}°N, {lon}°E
- Estimated Surface Wind Speed: {wind_kmh} km/h
- Assessed Risk Level: {risk_level}
- Predicted Trajectory Horizons (LSTM Model 3):
{forecast_summary}

USER ROLE & PERSPECTIVE:
------------------------
{role_guidelines}

INSTRUCTIONS:
1. Always base your analysis and recommendations strictly on the live telemetry provided above.
2. Structure your response with clear Markdown headings, bullet points, and high-priority action items.
3. Include:
   - 🚨 **Current Situation & Threat Window**: Category severity and projected path.
   - 🎯 **Critical Immediate Actions**: 3-5 high-priority steps tailored to the user's role.
   - ⏱️ **Timeline & Landfall Precautions**: What to do right now vs. in the next 6-24 hours.
   - ⚠️ **Key Hazard Warnings**: Wind gust damage, storm surges, inland flooding, or communication outage.
4. Keep the tone calm, urgent, highly professional, and actionable. Do not speculate beyond meteorological consensus.
"""
    return prompt


@ai_bp.get("/status")
def ai_status():
    """Check if Groq API key is available and list supported models."""
    key = _get_api_key()
    return jsonify({
        "groq_configured": bool(key),
        "default_model": DEFAULT_MODEL,
        "available_models": [
            DEFAULT_MODEL,
            FALLBACK_MODEL,
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
    }), 200


@ai_bp.post("/ask")
def ask_ai():
    """
    Query the AI Action Planner / Cyclone Assistant.

    Request JSON:
    {
      "message": "What should residents in Odisha do right now?",
      "role": "resident" | "fisherman" | "port_operator" | "disaster_official" | "general",
      "context": {
         "cyclone_detected": true,
         "category": "VSCS",
         "confidence": 0.91,
         "lat": 14.2,
         "lon": 82.6,
         "wind_speed_kmh": 175,
         "forecast": [...],
         "risk_level": "VERY HIGH"
      },
      "api_key": "optional groq api key override",
      "model": "optional groq model name",
      "conversation_history": [ {"role": "user"|"assistant", "content": "..."} ]
    }
    """
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    role = data.get("role", "general")
    context = data.get("context", {})
    client_key = data.get("api_key", "")
    model_name = data.get("model", DEFAULT_MODEL)
    history = data.get("conversation_history", [])

    api_key = _get_api_key(client_key)
    if not api_key:
        return jsonify({
            "error": "Groq API key not configured.",
            "instructions": (
                "Please provide your Groq API key in the request payload as 'api_key', "
                "or set the GROQ_API_KEY environment variable on the backend server."
            )
        }), 401

    if not user_msg:
        user_msg = f"Generate an immediate emergency action and preparedness plan for a {role} based on the latest cyclone telemetry."

    system_prompt = _build_system_prompt(role, context)

    # Prepare message list for Groq
    messages = [{"role": "system", "content": system_prompt}]

    # Append recent conversation history (up to last 6 turns)
    for turn in history[-6:]:
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            if turn["role"] in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_msg})

    # Call Groq API directly via requests
    try:
        import requests
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Try specified model first, then fallback to other active models if needed
        candidate_models = [model_name]
        for fallback in [DEFAULT_MODEL, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        ai_response = None
        used_model = None
        last_error = ""

        for m in candidate_models:
            payload = {
                "model": m,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1024
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                res_data = resp.json()
                ai_response = res_data["choices"][0]["message"]["content"]
                used_model = res_data.get("model", m)
                break
            else:
                last_error = resp.text
                logger.warning(f"Groq model {m} failed ({resp.status_code}): {resp.text}")

        if ai_response is None:
            return jsonify({
                "error": f"Groq API error: {last_error}"
            }), 502

    except Exception as e_http:
        logger.exception("Groq API request failed")
        return jsonify({
            "error": f"Failed to communicate with Groq API: {str(e_http)}"
        }), 500

    return jsonify({
        "response": ai_response,
        "model": used_model,
        "role": role,
        "status": "success"
    }), 200
