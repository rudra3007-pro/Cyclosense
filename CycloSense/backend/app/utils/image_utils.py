"""
CycloSense — Image Preprocessing Utilities
Shared transforms for Model 1 (YOLO) and Model 2 (EfficientNet).
"""

import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# ── EfficientNet input transform (matches training augmentation) ──────────────
CLASSIFY_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw image bytes to an RGB PIL Image."""
    img = Image.open(io.BytesIO(image_bytes))
    return img.convert("RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Apply EfficientNet preprocessing to a PIL image.
    Returns a (1, 3, 224, 224) tensor.
    """
    return CLASSIFY_TRANSFORM(img).unsqueeze(0)


def crop_from_bbox(img: Image.Image, bbox: list) -> Image.Image:
    """
    Crop a PIL image using a YOLO bounding box.

    Args:
        img:  original PIL Image (full satellite frame).
        bbox: [x_center, y_center, width, height] in pixel coordinates
              (as returned by Ultralytics results.boxes.xywh).

    Returns:
        Cropped PIL Image of the cyclone region.
    """
    x_c, y_c, w, h = bbox
    x1 = max(0, int(x_c - w / 2))
    y1 = max(0, int(y_c - h / 2))
    x2 = min(img.width,  int(x_c + w / 2))
    y2 = min(img.height, int(y_c + h / 2))
    return img.crop((x1, y1, x2, y2))


# ── Realistic oceanic bounds for North Indian Ocean basins ────────────────────
BASIN_BOUNDS = {
    "bay_of_bengal": {"lat_min": 10.0, "lat_max": 19.5, "lon_min": 83.5, "lon_max": 93.0},
    "arabian_sea":   {"lat_min": 10.0, "lat_max": 20.0, "lon_min": 62.0, "lon_max": 72.0},
    "full_domain":   {"lat_min": 2.0,  "lat_max": 30.0, "lon_min": 50.0, "lon_max": 100.0},
}


def bbox_to_geocentre(bbox: list, img_width: int, img_height: int,
                       basin: str = "bay_of_bengal") -> dict:
    """
    Convert pixel bounding-box centre to approximate geographic coordinates
    grounded in the specified oceanic basin (Bay of Bengal / Arabian Sea).
    Prevents satellite crops from erroneously projecting onto continental landmass.

    Returns:
        {"lat": float, "lon": float}
    """
    bounds = BASIN_BOUNDS.get((basin or "").lower(), BASIN_BOUNDS["bay_of_bengal"])
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]

    x_c, y_c, _, _ = bbox
    norm_x = min(max(x_c / img_width, 0.0), 1.0)
    norm_y = min(max(y_c / img_height, 0.0), 1.0)

    lat = lat_max - norm_y * (lat_max - lat_min)
    lon = lon_min + norm_x * (lon_max - lon_min)
    return {"lat": round(float(lat), 4), "lon": round(float(lon), 4)}
