"""Streamlit UI for EcoSort inference."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecosort.inference import load_predictor, open_uploaded_image, predict_image  # noqa: E402

st.set_page_config(page_title="EcoSort | Waste classifier", page_icon="♻️", layout="centered")

st.markdown("""
<style>
    .stApp { background: linear-gradient(145deg, #f7fbf6 0%, #ffffff 55%); }
    [data-testid="stMetricValue"] { color: #207044; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the EcoSort model…")
def cached_predictor(path: str):
    return load_predictor(path, device="cpu")

def main() -> None:
    st.title("♻️ EcoSort")
    st.caption("Waste-material recognition powered by the selected MobileNetV3 model.")
    model_path = os.getenv("ECOSORT_MODEL_PATH", str(ROOT / "models" / "best_model.pt"))
    with st.sidebar:
        st.header("Model")
        st.write("MobileNetV3-Small • 12 material classes • CPU-ready inference")
        st.caption("Held-out test accuracy: 95.96% • Macro F1: 94.93%")
        st.divider()
        st.header("How to use")
        st.write("Upload a clear image containing one item of household waste.")
        st.caption("Accepted formats: JPG, JPEG, PNG • Maximum upload: 10 MB")
    upload = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])
    if not upload:
        st.info("Upload an image to get a material prediction.")
        return
    try:
        image = open_uploaded_image(upload)
    except ValueError as error:
        st.error(str(error)); return
    st.image(image, caption="Uploaded image", use_container_width=True)
    try:
        model, classes, image_size = cached_predictor(model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(f"Model unavailable: {error}")
        st.info("For local use, train a model then copy/rename its selected checkpoint to `models/best_model.pt`, or set `ECOSORT_MODEL_PATH`.")
        return
    result = predict_image(model, image, classes, image_size)
    left, right = st.columns(2)
    left.success(f"Material: **{result['label']}**")
    right.metric("Confidence", f"{result['confidence']:.1%}")
    table = pd.DataFrame(result["probabilities"].items(), columns=["Class", "Probability"])
    st.subheader("Class probabilities")
    st.bar_chart(table.set_index("Class"), color="#2E7D32")
    st.dataframe(table.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
