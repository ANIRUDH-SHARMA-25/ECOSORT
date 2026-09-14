# EcoSort ♻️

EcoSort is an end-to-end PyTorch computer-vision project that classifies waste-material images. It trains a compact custom CNN and a MobileNetV3 transfer-learning model, evaluates both comprehensively, and serves the chosen checkpoint with a CPU-friendly Streamlit app.

## Highlights

- Dataset validation plus reproducible stratified train/validation/test splits
- Training augmentation and ImageNet-normalized inference preprocessing
- Custom CNN baseline and pretrained MobileNetV3-Small transfer-learning model
- Best-validation checkpointing, configurable YAML hyperparameters, and reproducible seeds
- Accuracy, macro precision/recall/F1, per-class CSV, confusion matrix, and learning curves
- A comparison command which selects the best test-accuracy checkpoint for the app
- Safe JPG/JPEG/PNG uploads, invalid-image messaging, and missing-model messaging

## Repository layout

```text
app/                 Streamlit UI
config/default.yaml  training settings
models/              optional small deployment checkpoint (not committed)
src/ecosort/         production package
tests/               unit tests
artifacts/           generated checkpoints and evaluation reports (ignored)
data/                downloaded dataset (ignored)
```

## Dataset

Use the public [Kaggle Garbage Classification dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) or another waste image dataset with one directory per class. The project discovers class names dynamically, so it supports the approximately 12-category Kaggle variant without source-code changes.

After downloading and extracting it, arrange images as follows (the folder can have any number of classes):

```text
data/raw/
  battery/
  biological/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
  ...
```

Do not commit `data/`: it is deliberately ignored. Validate the extraction first:

```powershell
python -m ecosort validate --data-dir data/raw
```

## Local setup

Python 3.10+ is recommended. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

On macOS/Linux, activate with `source .venv/bin/activate` and use `export PYTHONPATH=src`.

## Train and evaluate

Settings live in `config/default.yaml`; update `data_dir`, epochs, batch size, or device there. A GPU is used automatically if available; inference always runs on CPU.

Train a single model:

```powershell
python -m ecosort train --config config/default.yaml --model custom_cnn
python -m ecosort train --config config/default.yaml --model mobilenet_v3_small
```

Run the complete comparison and select the winner:

```powershell
python -m ecosort compare --config config/default.yaml
```

Each run creates `artifacts/best_<model>.pt`, plus `artifacts/<model>/summary_metrics.csv`, `per_class_metrics.csv`, `confusion_matrix.png`, and `training_curves.png`. The comparison creates `artifacts/model_comparison.csv` and copies the best evaluated model to `models/best_model.pt` for Streamlit. Checkpoints are intentionally gitignored; attach a small chosen checkpoint to a GitHub release, cloud storage, or rebuild it during deployment if it is too large for practical repository use.

## Run inference app

First run `compare` or place a compatible checkpoint at `models/best_model.pt`. Then:

```powershell
streamlit run app/app.py
```

To use another checkpoint, set `ECOSORT_MODEL_PATH` to its path before starting Streamlit. The app accepts JPG, JPEG, and PNG uploads, shows the image, top class and confidence, and all class probabilities. It gracefully explains malformed uploads and absent/invalid checkpoints.

## Streamlit Community Cloud deployment

1. Push this repository, excluding datasets and large artifacts.
2. Make `models/best_model.pt` available in the repository only if it is comfortably within GitHub and Cloud limits. Otherwise have a deployment build step retrieve a trusted release asset, or deploy after adding the selected small checkpoint through your hosting workflow. Do not hardcode personal file paths.
3. In Community Cloud select `app/app.py` as the entry point and use `requirements.txt`.
4. If the checkpoint lives elsewhere in the deployed filesystem, configure `ECOSORT_MODEL_PATH`; the app will display an actionable message when it is unavailable.

## Tests

```powershell
pytest -q
```

The tests cover folder-dataset validation and safe single-image inference. For an end-to-end smoke test, validate a real dataset, train for one epoch with a small batch size, and upload an image in the app.
