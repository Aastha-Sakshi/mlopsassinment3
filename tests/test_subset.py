import json

import yaml

from src.data.splits import ManifestRow, write_manifests
from src.data.subset import build_balanced_subset, validate_subset


def test_balanced_subset_is_exact_and_reproducible(tmp_path):
    source = tmp_path / "source"
    splits = {
        "train": [ManifestRow(f"Cat/{index}.jpg", "cat") for index in range(5000)]
        + [ManifestRow(f"Dog/{index}.jpg", "dog") for index in range(5000)],
        "validation": [ManifestRow(f"Cat/v{index}.jpg", "cat") for index in range(630)]
        + [ManifestRow(f"Dog/v{index}.jpg", "dog") for index in range(630)],
        "test": [ManifestRow(f"Cat/t{index}.jpg", "cat") for index in range(630)]
        + [ManifestRow(f"Dog/t{index}.jpg", "dog") for index in range(630)],
    }
    write_manifests(
        source,
        splits,
        {"split_seed": 42, "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1}, "dataset_fingerprint_sha256": "abc"},
    )
    dvc_path = tmp_path / "processed.dvc"
    dvc_path.write_text(yaml.safe_dump({"outs": [{"md5": "hash.dir", "size": 1, "nfiles": 2, "hash": "md5"}]}), encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_balanced_subset(source, dvc_path, first)
    build_balanced_subset(source, dvc_path, second)
    assert validate_subset(first)["total"] == 12480
    for name in ("train.csv", "validation.csv", "test.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    first_metadata = json.loads((first / "subset_metadata.json").read_text(encoding="utf-8"))
    second_metadata = json.loads((second / "subset_metadata.json").read_text(encoding="utf-8"))
    assert first_metadata == second_metadata


def test_project_subset_rebuild_matches_recorded_manifests(tmp_path):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    recorded = project_root / "data" / "manifests" / "baseline_50"
    rebuilt = tmp_path / "rebuilt"
    build_balanced_subset(
        project_root / "data" / "manifests",
        project_root / "data" / "processed" / "PetImages224.dvc",
        rebuilt,
        selection_seed=20260829,
    )
    for name in ("train.csv", "validation.csv", "test.csv"):
        assert (recorded / name).read_text(encoding="utf-8").splitlines() == (
            rebuilt / name
        ).read_text(encoding="utf-8").splitlines()
    assert (recorded / "subset_metadata.json").read_bytes() == (
        rebuilt / "subset_metadata.json"
    ).read_bytes()
    assert validate_subset(recorded)["combined_sha256"] == (
        "55c208fea1fb4fea2043dab00db62ab012cf1fa851a761b01526c96fd6d4b8a9"
    )
