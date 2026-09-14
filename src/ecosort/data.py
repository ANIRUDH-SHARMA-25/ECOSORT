"""Dataset validation, deterministic splits, and image transforms."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def discover_classes(data_dir: str | Path) -> list[str]:
    """Return alphabetically ordered class directory names."""
    root = Path(data_dir)
    return sorted(item.name for item in root.iterdir() if item.is_dir()) if root.exists() else []


def validate_dataset(data_dir: str | Path, min_images_per_class: int = 3) -> dict[str, object]:
    """Validate a class-per-folder image dataset without loading it into memory."""
    root = Path(data_dir)
    errors: list[str] = []
    counts: dict[str, int] = {}
    invalid_files: list[str] = []
    if not root.is_dir():
        return {"valid": False, "classes": [], "counts": {}, "invalid_files": [],
                "errors": [f"Dataset directory does not exist: {root}"]}
    classes = discover_classes(root)
    if len(classes) < 2:
        errors.append("At least two class folders are required.")
    for class_name in classes:
        files = [p for p in (root / class_name).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        counts[class_name] = len(files)
        if len(files) < min_images_per_class:
            errors.append(f"'{class_name}' has {len(files)} images; need at least {min_images_per_class}.")
        for image_path in files:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, ValueError):
                invalid_files.append(str(image_path))
    if invalid_files:
        errors.append(f"Found {len(invalid_files)} unreadable image file(s).")
    return {"valid": not errors, "classes": classes, "counts": counts,
            "invalid_files": invalid_files, "errors": errors}


def build_transforms(image_size: int = 224, training: bool = False) -> transforms.Compose:
    """Create augmentation for training or deterministic preprocessing for evaluation."""
    base = [transforms.Resize((image_size, image_size))]
    if training:
        base = [transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)]
    return transforms.Compose(base + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ImagePathsDataset(Dataset):
    """Image dataset from `(path, label)` samples, supporting separate split transforms."""
    def __init__(self, samples: Sequence[tuple[str, int]], transform: transforms.Compose | None = None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return (self.transform(image) if self.transform else image), label


def collect_samples(data_dir: str | Path, classes: Sequence[str]) -> list[tuple[str, int]]:
    """Collect supported image paths and numeric labels."""
    root = Path(data_dir)
    return [(str(path), label) for label, class_name in enumerate(classes)
            for path in sorted((root / class_name).rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]


def stratified_split(samples: Sequence[tuple[str, int]], val_fraction: float, test_fraction: float, seed: int):
    """Create reproducible stratified train, validation, and test samples."""
    from sklearn.model_selection import train_test_split
    paths, labels = zip(*samples)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=val_fraction + test_fraction, stratify=labels, random_state=seed)
    relative_test = test_fraction / (val_fraction + test_fraction)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=relative_test, stratify=temp_labels, random_state=seed)
    return (list(zip(train_paths, train_labels)), list(zip(val_paths, val_labels)), list(zip(test_paths, test_labels)))


def class_distribution(samples: Sequence[tuple[str, int]]) -> dict[int, int]:
    return dict(Counter(label for _, label in samples))


def limit_samples_per_class(samples: Sequence[tuple[str, int]], limit: int | None, seed: int) -> list[tuple[str, int]]:
    """Select an equal, reproducible cap per class for smoke tests; `None` preserves all data."""
    if limit is None:
        return list(samples)
    if limit < 3:
        raise ValueError("max_samples_per_class must be at least 3 for stratified splits.")
    generator = torch.Generator().manual_seed(seed)
    grouped: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)
    selected: list[tuple[str, int]] = []
    for label, group in grouped.items():
        if len(group) < limit:
            raise ValueError(f"Class {label} has only {len(group)} samples; cannot select {limit}.")
        order = torch.randperm(len(group), generator=generator).tolist()
        selected.extend(group[index] for index in order[:limit])
    return selected


def class_weights(samples: Sequence[tuple[str, int]], num_classes: int) -> torch.Tensor:
    """Return normalized inverse-frequency weights derived exclusively from training samples."""
    counts = torch.tensor([sum(label == index for _, label in samples) for index in range(num_classes)], dtype=torch.float32)
    if torch.any(counts == 0):
        raise ValueError("Every class must occur in the training split before computing class weights.")
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()
