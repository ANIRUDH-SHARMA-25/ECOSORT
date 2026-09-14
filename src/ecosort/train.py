"""Training loop, checkpoints, and histories."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm


def run_epoch(model: nn.Module, loader, criterion, device: torch.device, optimizer=None,
              scaler: torch.amp.GradScaler | None = None) -> tuple[float, float]:
    """Run one train epoch when optimizer is supplied, otherwise evaluate."""
    training = optimizer is not None
    model.train(training)
    loss_total = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    amp_enabled = scaler is not None
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if training:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_total += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_total / max(total, 1), correct / max(total, 1)


def save_checkpoint(path: str | Path, model: nn.Module, model_name: str, classes: list[str], image_size: int, epoch: int, val_accuracy: float) -> None:
    """Save portable model metadata and CPU tensors."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.cpu().state_dict(), "model_name": model_name, "classes": classes,
                "image_size": image_size, "epoch": epoch, "val_accuracy": val_accuracy}, path)


def fit(model: nn.Module, train_loader, val_loader, optimizer, criterion, device: torch.device,
        epochs: int, checkpoint_path: str | Path, model_name: str, classes: list[str], image_size: int,
        scheduler=None, on_epoch=None, early_stopping_patience: int | None = None,
        mixed_precision: bool = False) -> dict[str, list[float]]:
    """Train, validate, retain the best validation-accuracy checkpoint, and return history."""
    history: dict[str, list[float]] = {key: [] for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy")}
    best_accuracy = -1.0
    epochs_without_improvement = 0
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision and device.type == "cuda")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        for key, value in zip(history, (train_loss, val_loss, train_accuracy, val_accuracy)):
            history[key].append(value)
        if scheduler:
            scheduler.step(val_loss)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, model_name, classes, image_size, epoch, val_accuracy)
            model.to(device)
        else:
            epochs_without_improvement += 1
        if on_epoch:
            on_epoch(epoch, history, best_accuracy)
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best validation accuracy: {best_accuracy:.4f}")
            break
    return history
