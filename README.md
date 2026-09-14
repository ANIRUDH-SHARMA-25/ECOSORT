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

### Reproducible Kaggle download

EcoSort includes a preparation script for the selected [Garbage Classification (12 classes) dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification), which has 15,150 images across paper, cardboard, biological, metal, plastic, green/brown/white glass, clothes, shoes, batteries, and trash. It downloads the archive and normalizes any nested Kaggle extraction directory into `data/raw/<class>/...`.

1. With `.venv` activated, run `kaggle auth login` and complete the browser-based Kaggle authorization. Credentials are stored outside this repository.
2. With dependencies installed, run:

```powershell
python scripts/download_dataset.py
python -m ecosort validate --data-dir data/raw
```

If browser authorization is unavailable, create a Kaggle API token from [Kaggle Settings](https://www.kaggle.com/settings/api) and provide it through the client’s supported `KAGGLE_API_TOKEN` environment variable. Never commit credentials.

The script will not replace a non-empty `data/raw` directory. Review it first; only use `--force` when intentionally re-preparing the dataset. This workflow downloads data only—it does not start training.

### Dataset EDA and split checks

Generate class-distribution CSV/plot, a leakage-free split report, and an augmentation preview before training:

```powershell
python scripts/eda.py --data-dir data/raw --output-dir reports/eda
```

`reports/eda/class_distribution.csv` records the class counts in each stratified split. The training code uses inverse-frequency class-weighted cross-entropy by default (`class_weighted_loss: true`) so large classes such as `clothes` do not dominate optimization. Use balanced accuracy, macro precision/recall/F1, and the per-class report—not raw accuracy alone—to compare models.

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

### Smoke training

Before a full experiment, run the one-epoch integration smoke test. It uses an equal cap of 10 images per class at 64px and is deliberately unsuitable for comparing real model quality:

```powershell
python -m ecosort compare --config config/smoke.yaml
```

It verifies both architectures, training/backpropagation, checkpoint save/reload, evaluation CSVs/plots, and app-compatible inference. Outputs live in ignored `artifacts/smoke/`.

On a CUDA-enabled machine, use `config/smoke-gpu.yaml` instead. Training automatically stores any pretrained-weight cache below the ignored configured `artifact_dir`, rather than a user-profile cache.

### GPU training recommendation

This project’s default dependencies are CPU-compatible for reliable Streamlit deployment. On the checked development machine, an NVIDIA GeForce RTX 4060 Laptop GPU is present, but the installed PyTorch build is CPU-only. For full training, use this local GPU after installing the CUDA-enabled PyTorch/torchvision wheel selected for your Windows driver on the official [PyTorch installation page](https://pytorch.org/get-started/locally/), then confirm `torch.cuda.is_available()` returns `True`.

That local RTX 4060 should be substantially faster and more convenient than CPU training for this dataset. Kaggle or Colab GPU is a good alternative only if you prefer a cloud notebook, need a longer unattended run, or cannot enable CUDA locally.

For this RTX 4060 machine, the NVIDIA driver exposes CUDA 13.3 and EcoSort includes the matching official PyTorch CUDA 13.0 wheel pins in `requirements-gpu.txt`. Keep `requirements.txt` unchanged for CPU Streamlit deployments. To enable the local training environment, run the following in a normal PowerShell window (the download is approximately 2 GB):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip uninstall -y torch torchvision
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected verification includes a `+cu130` Torch version, `True`, and `NVIDIA GeForce RTX 4060 Laptop GPU`. Then rerun `pytest -q` and `python -m ecosort compare --config config/smoke.yaml` before beginning a full run.

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
