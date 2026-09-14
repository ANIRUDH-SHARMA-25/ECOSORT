"""Command line entry points for validation, training, and evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import get_device, load_config, set_seed
from .data import (ImagePathsDataset, build_transforms, class_weights, collect_samples,
                   limit_samples_per_class, stratified_split, validate_dataset)
from .evaluate import predict_loader, save_evaluation
from .models import build_model, freeze_backbone
from .train import fit


def _loaders(config, classes):
    samples = collect_samples(config["data_dir"], classes)
    samples = limit_samples_per_class(samples, config.get("max_samples_per_class"), config["seed"])
    train, val, test = stratified_split(samples, config["validation_fraction"], config["test_fraction"], config["seed"])
    args = {"batch_size": config["batch_size"], "num_workers": config["num_workers"], "pin_memory": torch.cuda.is_available()}
    return (DataLoader(ImagePathsDataset(train, build_transforms(config["image_size"], True)), shuffle=True, **args),
            DataLoader(ImagePathsDataset(val, build_transforms(config["image_size"])), **args),
            DataLoader(ImagePathsDataset(test, build_transforms(config["image_size"])), **args))


def train_command(config_path: str, model_override: str | None = None) -> tuple[dict, Path]:
    config = load_config(config_path); config["model"] = model_override or config["model"]
    # Keep pretrained-weight downloads with ignored experiment artifacts, not in a user-profile cache.
    os.environ.setdefault("TORCH_HOME", str(Path(config["artifact_dir"]) / "torch-cache"))
    set_seed(config["seed"])
    status = validate_dataset(config["data_dir"])
    if not status["valid"]:
        raise RuntimeError("Dataset validation failed:\n- " + "\n- ".join(status["errors"]))
    classes = status["classes"]; device = get_device(config.get("device", "auto"))
    train_loader, val_loader, test_loader = _loaders(config, classes)
    model = build_model(config["model"], len(classes), config.get("pretrained", True)).to(device)
    freeze_backbone(model, config["model"], config.get("freeze_backbone_epochs", 0) > 0)
    # Include all parameters from the start. Frozen parameters simply have no gradients;
    # this keeps optimizer and scheduler parameter groups stable when unfreezing.
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    checkpoint = Path(config["artifact_dir"]) / f"best_{config['model']}.pt"
    weights = class_weights(train_loader.dataset.samples, len(classes)).to(device) if config.get("class_weighted_loss", False) else None
    def epoch_hook(epoch, history, _best):
        if config["model"] == "mobilenet_v3_small" and epoch == config.get("freeze_backbone_epochs", 0):
            freeze_backbone(model, config["model"], False)
        print(f"Epoch {epoch}/{config['epochs']} | train acc {history['train_accuracy'][-1]:.3f} | val acc {history['val_accuracy'][-1]:.3f}")
    started_at = time.perf_counter()
    history = fit(model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(weight=weights), device, config["epochs"], checkpoint,
                  config["model"], classes, config["image_size"], ReduceLROnPlateau(optimizer, patience=2), epoch_hook,
                  early_stopping_patience=config.get("early_stopping_patience"), mixed_precision=config.get("mixed_precision", False))
    training_duration_seconds = time.perf_counter() - started_at
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(checkpoint_data["model_state"])
    targets, predictions = predict_loader(model, test_loader, device)
    metrics = save_evaluation(targets, predictions, classes, history, Path(config["artifact_dir"]) / config["model"])
    metrics["best_epoch"] = checkpoint_data["epoch"]
    metrics["best_validation_accuracy"] = checkpoint_data["val_accuracy"]
    metrics["training_duration_seconds"] = training_duration_seconds
    import pandas as pd
    pd.DataFrame([{key: value for key, value in metrics.items() if key != "per_class"}]).to_csv(
        Path(config["artifact_dir"]) / config["model"] / "summary_metrics.csv", index=False)
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))
    print(f"Best checkpoint: {checkpoint}")
    return metrics, checkpoint


def compare_command(config_path: str) -> None:
    """Train both models and copy the strongest test-accuracy checkpoint for the app."""
    rows, checkpoints = [], []
    for name in ("custom_cnn", "mobilenet_v3_small"):
        metrics, checkpoint = train_command(config_path, name)
        rows.append({"model": name, **{key: value for key, value in metrics.items() if key != "per_class"}})
        checkpoints.append(checkpoint)
    winner_index = max(range(len(rows)), key=lambda index: rows[index]["accuracy"])
    output = Path(load_config(config_path)["artifact_dir"])
    import pandas as pd
    pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_csv(output / "model_comparison.csv", index=False)
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(checkpoints[winner_index], "models/best_model.pt")
    print(f"Selected {rows[winner_index]['model']} for app inference: models/best_model.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="EcoSort training utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate folder dataset")
    validate.add_argument("--data-dir", default="data/raw")
    train = sub.add_parser("train", help="Train and evaluate one model")
    train.add_argument("--config", default="config/default.yaml")
    train.add_argument("--model", choices=["custom_cnn", "mobilenet_v3_small"])
    compare = sub.add_parser("compare", help="Train both models, evaluate, and select the winner")
    compare.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_dataset(args.data_dir); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["valid"] else 1)
    if args.command == "compare":
        compare_command(args.config)
    else:
        train_command(args.config, args.model)


if __name__ == "__main__":
    main()
