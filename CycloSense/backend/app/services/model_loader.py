"""
CycloSense — Model Loader Service
Loads all three trained models and the scaler once at application startup.
Models are stored as module-level singletons so they are shared across requests.
"""

import os
import pickle
import logging

import torch
import torchvision.models as tv_models
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "weights")

WEIGHTS = {
    "model1": os.path.join(_BASE, "model1_detection.pt"),
    "model2": os.path.join(_BASE, "model2_classification.pt"),
    "model3": os.path.join(_BASE, "model3_forecasting.pt"),
    "scaler":  os.path.join(_BASE, "model3_scaler.pkl"),
}

# ── IMD intensity classes (3-class: D+DD→D_DD, SCS+SuCS→VSCS) ──────────────────────
CLASSES = ["CS", "D_DD", "VSCS"]   # sorted alphabetically — must match training
NUM_CLASSES = len(CLASSES)

# ── LSTM hyper-parameters (must match training config) ───────────────────────
LSTM_INPUT_SIZE  = 3    # (lat, lon, wind_speed)
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS  = 2
LSTM_OUTPUT_SIZE = 3    # predict (lat, lon, wind_speed)
LSTM_SEQ_LEN     = 8   # 8 × 3-hour steps = 24 hours of history

# ── Singleton holders ─────────────────────────────────────────────────────────
_model1 = None   # YOLOv8-nano  (ultralytics.YOLO)
_model2 = None   # EfficientNet-B0, 3-class (torch.nn.Module, eval mode)
_model3 = None   # 2-layer LSTM    (torch.nn.Module, eval mode)
_scaler = None   # sklearn StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Internal LSTM definition — must match the architecture used during training
# ──────────────────────────────────────────────────────────────────────────────
class CycloneLSTM(torch.nn.Module):
    """2-layer LSTM that predicts next (lat, lon, wind_speed) from a sequence."""

    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        return self.fc(out[:, -1, :])  # take last time-step → (batch, output)


# ──────────────────────────────────────────────────────────────────────────────
# Loader functions
# ──────────────────────────────────────────────────────────────────────────────

def _load_model1():
    """Load YOLOv8-nano detection model."""
    path = WEIGHTS["model1"]
    logger.info("Loading Model 1 (YOLO detection) from %s", path)
    model = YOLO(path)
    logger.info("Model 1 loaded successfully.")
    return model


def _load_model2():
    """Load EfficientNet-B0 3-class classification model (CS, D_DD, VSCS)."""
    path = WEIGHTS["model2"]
    logger.info("Loading Model 2 (EfficientNet 3-class classification) from %s", path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the exact architecture used during training
    model = tv_models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4, inplace=True),
        torch.nn.Linear(in_features, NUM_CLASSES),
    )

    state = torch.load(path, map_location=device, weights_only=True)
    # Support both raw state-dict and wrapped checkpoint
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Model 2 loaded on %s — classes: %s", device, CLASSES)
    return model


def _load_model3_and_scaler():
    """Load 2-layer LSTM forecasting model and its StandardScaler."""
    m3_path  = WEIGHTS["model3"]
    sc_path  = WEIGHTS["scaler"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading Model 3 (LSTM forecasting) from %s", m3_path)
    model = CycloneLSTM(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        output_size=LSTM_OUTPUT_SIZE,
    )
    state = torch.load(m3_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logger.info("Loading scaler from %s", sc_path)
    with open(sc_path, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Model 3 + scaler loaded on %s.", device)
    return model, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_all_models():
    """
    Load all models at application startup.
    Call this once from the Flask app factory (create_app).
    Subsequent calls are no-ops — models are cached in module globals.
    """
    global _model1, _model2, _model3, _scaler

    if _model1 is None:
        _model1 = _load_model1()

    if _model2 is None:
        _model2 = _load_model2()

    if _model3 is None or _scaler is None:
        _model3, _scaler = _load_model3_and_scaler()

    logger.info("All CycloSense models loaded and ready.")


def get_model1() -> YOLO:
    """Return the loaded YOLOv8 detection model."""
    if _model1 is None:
        raise RuntimeError("Models not loaded. Call load_all_models() first.")
    return _model1


def get_model2() -> torch.nn.Module:
    """Return the loaded EfficientNet-B0 classification model."""
    if _model2 is None:
        raise RuntimeError("Models not loaded. Call load_all_models() first.")
    return _model2


def get_model3():
    """Return (lstm_model, scaler) tuple for forecasting."""
    if _model3 is None or _scaler is None:
        raise RuntimeError("Models not loaded. Call load_all_models() first.")
    return _model3, _scaler
