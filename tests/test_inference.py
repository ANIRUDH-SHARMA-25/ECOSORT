from io import BytesIO

import pytest
import torch
from PIL import Image

from ecosort.inference import open_uploaded_image, predict_image


def test_open_uploaded_image_and_predict():
    raw = BytesIO(); Image.new("RGB", (20, 20), "green").save(raw, format="PNG"); raw.seek(0)
    image = open_uploaded_image(raw)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 2))
    result = predict_image(model, image, ["metal", "paper"], 224)
    assert result["label"] in {"metal", "paper"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5


def test_invalid_upload_raises_value_error():
    with pytest.raises(ValueError):
        open_uploaded_image(BytesIO(b"not an image"))
