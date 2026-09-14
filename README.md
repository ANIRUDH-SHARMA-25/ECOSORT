# EcoSort ♻️

EcoSort is an end-to-end PyTorch computer-vision project that classifies waste-material images. It trains a compact custom CNN and a MobileNetV3 transfer-learning model, evaluates both comprehensively, and serves the chosen checkpoint with a CPU-friendly Streamlit app.

**Live app:** [https://ecosort-app.streamlit.app](https://ecosort-app.streamlit.app)

**Deployment model:** MobileNetV3-Small, selected from validation performance and served with CPU-compatible inference.

| Model | Test accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Custom CNN | 67.74% | 66.76% | 62.69% |
| **MobileNetV3-Small (selected)** | **95.96%** | **95.21%** | **94.93%** |

## Highlights

- Dataset validation plus reproducible stratified train/validation/test splits
- Training augmentation and ImageNet-normalized inference preprocessing
- Custom CNN baseline and pretrained MobileNetV3-Small transfer-learning model
- Best-validation checkpointing, configurable YAML hyperparameters, and reproducible seeds
- Accuracy, macro precision/recall/F1, per-class CSV, confusion matrix, and learning curves
- Validation-selected deployment checkpoint committed for straightforward Streamlit deployment
- Safe JPG/JPEG/PNG uploads, invalid-image messaging, and missing-model messaging

## Repository layout

```text
app/                 Streamlit UI
config/default.yaml  training settings
models/              selected 6.26 MB deployment checkpoint
src/ecosort/         production package
tests/               unit tests
artifacts/           generated checkpoints and evaluation reports (ignored)
data/                downloaded dataset (ignored)
```

## Dataset

### Dataset summary

The prepared Kaggle dataset contains **15,515 valid images** in 12 classes: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, and white-glass. Validation found no unreadable files. The fixed seed creates a stratified **70%/15%/15%** train/validation/test split with no path overlap.

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

Each run creates `artifacts/best_<model>.pt`, plus `artifacts/<model>/summary_metrics.csv`, `per_class_metrics.csv`, `confusion_matrix.png`, and `training_curves.png`. The comparison creates `artifacts/model_comparison.csv` and copies the best evaluated model to `models/best_model.pt` for Streamlit. Experiment checkpoints remain gitignored; the selected 6.26 MB deployment checkpoint is the intentional exception.

### Final measured results

Both models were trained on the same fixed stratified split with inverse-frequency weighted cross-entropy. The test set was kept untouched until one final evaluation of the validation-selected checkpoint.

| Model | Best epoch | Training duration | Test accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom CNN | 22 / 25 | 20.0 min | 67.74% | 66.76% | 62.69% |
| MobileNetV3-Small | 13 (stopped at 18 / 20) | 11.6 min | 95.96% | 95.21% | 94.93% |

MobileNetV3-Small is the selected deployment model because it achieved the strongest validation result (96.61%) and materially better balanced and macro test metrics. Its 6.26 MB CPU-loadable checkpoint is committed at `models/best_model.pt`; the dataset, experiment checkpoints, reports, and caches remain ignored.

The final Custom CNN configuration used 224px images, batch size 64, a learning rate of `1e-3`, CUDA mixed precision, weighted loss, and patience-6 early stopping. MobileNetV3-Small used pretrained weights, 224px images, batch size 32, a learning rate of `3e-4`, two frozen-backbone epochs, CUDA mixed precision, weighted loss, and patience-5 early stopping.

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

The selected checkpoint is already included in `models/best_model.pt`. Start locally:

```powershell
streamlit run app/app.py
```

To use another checkpoint, set `ECOSORT_MODEL_PATH` to its path before starting Streamlit. The app accepts JPG, JPEG, and PNG uploads, shows the image, top class and confidence, and all class probabilities. It gracefully explains malformed uploads and absent/invalid checkpoints.

## Streamlit Community Cloud deployment

1. Push this repository; `models/best_model.pt` is only 6.26 MB and is intentionally committed. Datasets, artifacts, and caches remain excluded.
2. In Streamlit Community Cloud, create an app from this repository and select `app/app.py` as the entry point.
3. Use the default Python version supported by `requirements.txt`; Community Cloud installs CPU PyTorch automatically, which is supported by EcoSort inference.
4. Deploy. No secrets are required. Leave `ECOSORT_MODEL_PATH` unset to use the committed checkpoint; set it only when deliberately deploying a different compatible checkpoint.

## Tests

```powershell
pytest -q
```

The tests cover folder-dataset validation and safe single-image inference. For an end-to-end smoke test, validate a real dataset, train for one epoch with a small batch size, and upload an image in the app.
