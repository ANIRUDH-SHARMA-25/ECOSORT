from io import BytesIO
from pathlib import Path

from PIL import Image

from ecosort.inference import load_predictor, open_uploaded_image, predict_image


EXPECTED_CLASSES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
    "metal", "paper", "plastic", "shoes", "trash", "white-glass",
]


def test_deployment_checkpoint_loads_on_cpu_and_preserves_label_mapping():
    checkpoint = Path(__file__).resolve().parents[1] / "models" / "best_model.pt"
    model, classes, image_size = load_predictor(checkpoint, device="cpu")
    assert classes == EXPECTED_CLASSES
    assert image_size == 224
    raw = BytesIO(); Image.new("RGB", (32, 32), "green").save(raw, format="PNG"); raw.seek(0)
    result = predict_image(model, open_uploaded_image(raw), classes, image_size)
    assert result["label"] in EXPECTED_CLASSES
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
