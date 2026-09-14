from pathlib import Path

from PIL import Image

from ecosort.data import class_weights, discover_classes, limit_samples_per_class, stratified_split, validate_dataset


def test_validate_dataset_accepts_small_valid_folder_dataset(tmp_path: Path):
    for name in ("glass", "paper"):
        folder = tmp_path / name; folder.mkdir()
        for index in range(3):
            Image.new("RGB", (8, 8), "white").save(folder / f"{index}.jpg")
    result = validate_dataset(tmp_path)
    assert result["valid"] is True
    assert discover_classes(tmp_path) == ["glass", "paper"]


def test_validate_dataset_rejects_missing_path(tmp_path: Path):
    assert validate_dataset(tmp_path / "missing")["valid"] is False


def test_stratified_split_is_disjoint_and_class_weights_favor_rare_classes():
    samples = [(f"class-{label}-{index}.jpg", label) for label, count in enumerate((20, 40)) for index in range(count)]
    train, validation, test = stratified_split(samples, 0.2, 0.2, seed=7)
    paths = [set(path for path, _ in split) for split in (train, validation, test)]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    weights = class_weights(train, 2)
    assert weights[0] > weights[1]
    assert len(limit_samples_per_class(samples, 10, 7)) == 20
