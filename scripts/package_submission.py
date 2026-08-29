"""Create the compact assignment submission ZIP from an explicit allowlist."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DIRECTORIES = (
    ".github",
    "configs",
    "data/manifests",
    "docs",
    "kaggle/kernel",
    "models/production",
    "scripts",
    "src",
    "tests",
    "artifacts/training/simple-cnn-baseline-50pct",
    "artifacts/post_deployment",
    "mlartifacts",
    "mlruns",
)

FILES = (
    ".dvc/.gitignore",
    ".dvc/config",
    ".dvcignore",
    ".dockerignore",
    ".gitignore",
    "Dockerfile",
    "README.md",
    "compose.yaml",
    "pytest.ini",
    "requirements-api.txt",
    "requirements.txt",
    "data/downloads/PetImages.dvc",
    "data/processed/PetImages224.dvc",
)

EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".gitkeep"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def iter_submission_files(root: Path):
    candidates = [root / relative for relative in FILES]
    for directory in DIRECTORIES:
        path = root / directory
        if path.exists():
            candidates.extend(item for item in path.rglob("*") if item.is_file())
    for path in sorted(set(candidates)):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_NAMES for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        yield path, relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/mlops-cats-dogs-assignment2.zip"),
    )
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_submission_files(root))
    largest = sorted(files, key=lambda item: item[0].stat().st_size, reverse=True)[:10]
    total_mb = sum(path.stat().st_size for path, _ in files) / (1024 * 1024)
    print(f"Allowlist: {len(files)} files ({total_mb:.2f} MiB uncompressed)")
    for path, relative in largest:
        print(f"  {path.stat().st_size / (1024 * 1024):8.2f} MiB  {relative.as_posix()}")
    if args.inspect_only:
        return
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in files:
            archive.write(path, relative.as_posix())
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Created {output} with {len(files)} files ({size_mb:.2f} MiB)")


if __name__ == "__main__":
    main()
