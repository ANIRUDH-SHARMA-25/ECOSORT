from pathlib import Path
from io import BytesIO

from PIL import Image
from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_errors():
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "♻️ EcoSort"
    assert app.file_uploader[0].label == "Upload a waste image"


def test_streamlit_app_renders_prediction_for_png_upload():
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    image_bytes = BytesIO()
    Image.new("RGB", (48, 48), "green").save(image_bytes, format="PNG")
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    app.file_uploader[0].set_value(("waste.png", image_bytes.getvalue(), "image/png"))
    app.run(timeout=30)
    assert not app.exception
    assert app.metric[0].label == "Confidence"
    assert app.dataframe[0]
