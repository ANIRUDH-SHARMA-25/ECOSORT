"""Download and normalize the EcoSort Kaggle dataset without committing it."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


DEFAULT_SLUG = "mostafaabla/garbage-classification"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_dataset_root(extracted: Path) -> Path:
    """Find the nested directory whose immediate children are image class folders."""
    candidates = [extracted, *[path for path in extracted.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        class_dirs = [item for item in candidate.iterdir() if item.is_dir()]
        if len(class_dirs) >= 2 and all(any(file.suffix.lower() in IMAGE_EXTENSIONS for file in folder.rglob("*")) for folder in class_dirs):
            return candidate
    raise RuntimeError("Could not locate class folders after extraction. Inspect data/downloads manually.")


def download_and_prepare(slug: str, download_dir: Path, output_dir: Path, force: bool) -> None:
    """Download Kaggle ZIP and place class folders in the project-standard `data/raw` layout."""
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty. Use --force only after reviewing its contents.")
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise RuntimeError("Kaggle CLI was not found. Activate .venv and run `python -m pip install -r requirements.txt`.")
    download_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([kaggle, "datasets", "download", "--dataset", slug, "--path", str(download_dir)], check=True)
    archives = sorted(download_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not archives:
        raise FileNotFoundError(f"No ZIP archive was downloaded to {download_dir}.")
    with tempfile.TemporaryDirectory(prefix="ecosort-kaggle-") as temp_dir:
        extracted = Path(temp_dir)
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(extracted)
        dataset_root = find_dataset_root(extracted)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(dataset_root, output_dir)
    print(f"Dataset ready at: {output_dir}")
    print("Validate it next with: python -m ecosort validate --data-dir data/raw")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize EcoSort's Kaggle dataset")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Replace an existing prepared output directory")
    args = parser.parse_args()
    download_and_prepare(args.slug, args.download_dir, args.output_dir, args.force)


if __name__ == "__main__":
    main()
