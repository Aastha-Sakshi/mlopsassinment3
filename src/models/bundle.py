"""Persist and restore versioned model bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.models.artifact import ModelArtifactMetadata, load_metadata, metadata_to_dict
from src.models.factory import build_model


MODEL_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"
CHECKSUM_FILENAME = "SHA256SUMS"


@dataclass(frozen=True)
class ModelBundlePaths:
    model_path: Path
    metadata_path: Path
    checksum_path: Path


@dataclass(frozen=True)
class LoadedModelBundle:
    model: nn.Module
    metadata: ModelArtifactMetadata
    model_path: Path


def save_model_bundle(
    bundle_dir: str | Path,
    model: nn.Module,
    metadata: ModelArtifactMetadata,
) -> ModelBundlePaths:
    target_dir = Path(bundle_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / MODEL_FILENAME
    metadata_path = target_dir / METADATA_FILENAME
    checksum_path = target_dir / CHECKSUM_FILENAME

    torch.save({"state_dict": model.state_dict()}, model_path)
    metadata_path.write_text(
        json.dumps(metadata_to_dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_path.write_text(
        f"{sha256_file(model_path)}  {MODEL_FILENAME}\n"
        f"{sha256_file(metadata_path)}  {METADATA_FILENAME}\n",
        encoding="utf-8",
    )
    return ModelBundlePaths(
        model_path=model_path,
        metadata_path=metadata_path,
        checksum_path=checksum_path,
    )


def load_model_bundle(bundle_dir: str | Path, map_location: str | torch.device = "cpu") -> LoadedModelBundle:
    bundle_path = Path(bundle_dir)
    validate_model_bundle(bundle_path)
    metadata = load_metadata(bundle_path / METADATA_FILENAME)
    payload = torch.load(bundle_path / MODEL_FILENAME, map_location=map_location, weights_only=True)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model = build_model(metadata.model_type, num_classes=len(metadata.class_labels))
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedModelBundle(model=model, metadata=metadata, model_path=bundle_path / MODEL_FILENAME)


def validate_model_bundle(bundle_dir: str | Path) -> None:
    """Validate required files, checksums, and metadata before model loading."""
    bundle_path = Path(bundle_dir)
    checksum_path = bundle_path / CHECKSUM_FILENAME
    if not checksum_path.is_file():
        raise ValueError(f"Model bundle is missing {CHECKSUM_FILENAME}.")
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        try:
            expected, filename = line.split("  ", maxsplit=1)
        except ValueError as error:
            raise ValueError("Malformed SHA256SUMS entry.") from error
        entries[filename] = expected
    if set(entries) != {MODEL_FILENAME, METADATA_FILENAME}:
        raise ValueError("SHA256SUMS must cover model.pt and metadata.json exactly.")
    for filename, expected in entries.items():
        path = bundle_path / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Invalid bundle checksum: {filename}")
    load_metadata(bundle_path / METADATA_FILENAME)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
