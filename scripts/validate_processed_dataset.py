"""Verify processed split counts and the canonical 224x224 RGB image contract."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from PIL import Image

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    images = sorted(args.data_dir.glob("*/*/*.jpg"))
    counts = Counter(path.parts[-3] for path in images)
    for path in images:
        with Image.open(path) as image:
            if image.size != (224, 224) or image.mode != "RGB":
                raise ValueError(f"Invalid processed image: {path}")
    print(f"Validated {len(images)} images: {dict(sorted(counts.items()))}")

if __name__ == "__main__":
    main()
