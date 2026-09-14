"""Safe, CPU-first single-image inference."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from .data import build_transforms
from .models import build_model


def load_predictor(checkpoint_path: str | Path, device: str = "cpu"):
    """Load a checkpoint and return `(model, classes, image_size)` on the requested device."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file was not found: {path}. Train a model or set ECOSORT_MODEL_PATH.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state", "model_name", "classes", "image_size"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Invalid EcoSort checkpoint; missing: {', '.join(sorted(missing))}")
    model = build_model(checkpoint["model_name"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
    return model, checkpoint["classes"], checkpoint["image_size"]


def open_uploaded_image(uploaded_file) -> Image.Image:
    """Validate and return an RGB image from a Streamlit uploaded file or file-like object."""
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        return Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, AttributeError) as error:
        raise ValueError("The upload is not a valid readable JPG, JPEG, or PNG image.") from error


def predict_image(model, image: Image.Image, classes: list[str], image_size: int, device: str = "cpu") -> dict:
    """Return the predicted class, confidence, and class probabilities."""
    tensor = build_transforms(image_size, training=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().tolist()
    indexed = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    index, confidence = indexed[0]
    return {"label": classes[index], "confidence": confidence,
            "probabilities": {classes[i]: probability for i, probability in indexed}}
