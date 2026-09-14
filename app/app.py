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

@st.cache_resource(show_spinner="Loading the EcoSort model…")
def cached_predictor(path: str):
    return load_predictor(path, device="cpu")

def main() -> None:
    st.title("♻️ EcoSort")
    st.caption("A computer-vision assistant for sorting waste material images.")
    model_path = os.getenv("ECOSORT_MODEL_PATH", str(ROOT / "models" / "best_model.pt"))
    with st.sidebar:
        st.header("About")
        st.write("EcoSort compares a custom CNN with transfer learning during training and serves the selected checkpoint on CPU.")
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
    st.success(f"Prediction: **{result['label']}**")
    st.metric("Confidence", f"{result['confidence']:.1%}")
    table = pd.DataFrame(result["probabilities"].items(), columns=["Class", "Probability"])
    st.subheader("Class probabilities")
    st.bar_chart(table.set_index("Class"))
    st.dataframe(table.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
