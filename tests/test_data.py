from pathlib import Path

from PIL import Image

from ecosort.data import discover_classes, validate_dataset


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
