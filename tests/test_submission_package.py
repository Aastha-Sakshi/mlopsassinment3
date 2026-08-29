from pathlib import Path

from scripts.package_submission import iter_submission_files


def test_submission_allowlist_excludes_large_and_secret_locations() -> None:
    root = Path(__file__).resolve().parents[1]
    relatives = {relative.as_posix() for _, relative in iter_submission_files(root)}
    assert "data/downloads/PetImages.dvc" in relatives
    assert "data/processed/PetImages224.dvc" in relatives
    assert "models/production/model.pt" in relatives
    assert not any(path.startswith("data/downloads/PetImages/") for path in relatives)
    assert not any(path.startswith("data/processed/PetImages224/") for path in relatives)
    assert not any("kaggle.json" in path.lower() for path in relatives)
    assert not any(".venv/" in path for path in relatives)
    assert not any(path.startswith(".dvc/cache/") for path in relatives)
