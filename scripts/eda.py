"""Create non-training dataset EDA and preprocessing/augmentation visual checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from ecosort.data import (build_transforms, class_distribution, collect_samples,
                          discover_classes, stratified_split, validate_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EcoSort dataset EDA outputs")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="reports/eda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    status = validate_dataset(args.data_dir)
    if not status["valid"]:
        raise RuntimeError("Dataset validation failed: " + "; ".join(status["errors"]))
    classes = discover_classes(args.data_dir)
    samples = collect_samples(args.data_dir, classes)
    train, validation, test = stratified_split(samples, 0.15, 0.15, args.seed)
    split_sets = [set(path for path, _ in split) for split in (train, validation, test)]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise RuntimeError("Leakage detected: an image appears in more than one split.")
    rows = []
    for index, name in enumerate(classes):
        rows.append({"class": name, "all": status["counts"][name], "train": class_distribution(train).get(index, 0),
                     "validation": class_distribution(validation).get(index, 0), "test": class_distribution(test).get(index, 0)})
    distribution = pd.DataFrame(rows); distribution.to_csv(output / "class_distribution.csv", index=False)
    axis = distribution.set_index("class")[["all", "train", "validation", "test"]].plot(kind="bar", figsize=(13, 6), color=["#236B3A", "#5BA66C", "#97C99C", "#D2E8D4"])
    axis.set(title="EcoSort class and split distribution", xlabel="Class", ylabel="Image count"); axis.figure.tight_layout(); axis.figure.savefig(output / "class_distribution.png", dpi=180); plt.close(axis.figure)
    # Display one deterministic source image with four independent augmented variants.
    source_path = samples[0][0]
    with Image.open(source_path) as source:
        source = source.convert("RGB")
    transform = build_transforms(args.image_size, training=True)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    axes[0].imshow(source); axes[0].set_title("Original")
    for index in range(4):
        tensor = transform(source)
        rendered = tensor.permute(1, 2, 0).numpy()
        rendered = rendered * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
        axes[index + 1].imshow(rendered.clip(0, 1)); axes[index + 1].set_title(f"Augmentation {index + 1}")
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(output / "augmentation_preview.png", dpi=180); plt.close(fig)
    pd.DataFrame([{"total_images": len(samples), "classes": len(classes), "leakage_free": True,
                   "min_class_count": min(status["counts"].values()), "max_class_count": max(status["counts"].values())}]).to_csv(output / "dataset_summary.csv", index=False)
    print(f"EDA reports written to {output}")


if __name__ == "__main__":
    main()
