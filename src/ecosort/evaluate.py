"""Classification reports, plots, and model comparison utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def predict_loader(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return np.array(targets), np.array(predictions)


def classification_metrics(targets: np.ndarray, predictions: np.ndarray, classes: list[str]) -> dict:
    """Calculate overall and per-class metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, labels=range(len(classes)), zero_division=0)
    return {"accuracy": float(accuracy_score(targets, predictions)), "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)), "macro_f1": float(np.mean(f1)),
            "per_class": pd.DataFrame({"class": classes, "precision": precision, "recall": recall, "f1": f1})}


def save_evaluation(targets: np.ndarray, predictions: np.ndarray, classes: list[str], history: dict[str, list[float]], output_dir: str | Path) -> dict:
    """Write metrics CSV/JSON-style summary, confusion matrix, and training curves."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(targets, predictions, classes)
    metrics["per_class"].to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame([{key: value for key, value in metrics.items() if key != "per_class"}]).to_csv(output / "summary_metrics.csv", index=False)
    fig, axis = plt.subplots(figsize=(max(7, len(classes)), max(6, len(classes))))
    sns.heatmap(confusion_matrix(targets, predictions), annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=axis)
    axis.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix"); fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Validation"); axes[0].set(title="Loss", xlabel="Epoch"); axes[0].legend()
    axes[1].plot(history["train_accuracy"], label="Train"); axes[1].plot(history["val_accuracy"], label="Validation"); axes[1].set(title="Accuracy", xlabel="Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / "training_curves.png", dpi=160); plt.close(fig)
    return metrics


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Build a sortable comparison table from model evaluation summaries."""
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)
