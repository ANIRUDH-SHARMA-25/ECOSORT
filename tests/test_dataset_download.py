from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_dataset import find_dataset_root


def test_find_dataset_root_handles_nested_kaggle_directory(tmp_path: Path):
    root = tmp_path / "archive" / "garbage_classification"
    for class_name in ("paper", "plastic"):
        folder = root / class_name
        folder.mkdir(parents=True)
        Image.new("RGB", (5, 5)).save(folder / "sample.jpg")
    assert find_dataset_root(tmp_path) == root
