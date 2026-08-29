# Cats vs Dogs MLOps Assignment

This repository implements the Assignment 2 MLOps lifecycle for binary Cats vs
Dogs image classification in a pet-adoption context.

## Current status

The required local M1-M5 path now includes:

- the configuration, preprocessing, dataset, and artifact contracts,
- dataset validation and deterministic split-manifest tooling,
- a Kaggle-trained repository-native SimpleCNN baseline and production bundle,
- persistent MLflow tracking with browser UI, metrics, loss/confusion plots,
- FastAPI health, prediction, request metrics, and post-deployment evaluation,
- a tested Docker image, Compose deployment, and GitHub CI/CD workflows.

Phase 2C is complete. The full processed dataset contains 24,966 validated
224 x 224 RGB images and is tracked by DVC. Baseline training uses a
deterministic, balanced manifest-only subset of 12,480 images.

The friend-provided prototype is preserved unchanged in
`prototype_original/mlops-cats-dogs-assignment2.zip`. It is a reference, not
the active implementation.

## Local toolchain status

A project-local Python 3.11 environment now exists in `.venv`, and
`requirements.txt` captures the intended local dependency contract. Kaggle or
Colab notebooks remain optional execution environments, but the repository is
the source of truth for local development, testing, MLflow tooling, Docker,
and the inference service.

## Architecture direction

```text
Kaggle dataset
  -> DVC-tracked raw data
  -> deterministic split manifests and processed 224 x 224 RGB data
  -> baseline SimpleCNN training and MLflow tracking
  -> versioned model bundle
  -> FastAPI inference service
  -> Docker Compose deployment
```

Kaggle Compute is the preferred optional remote environment for the baseline
because the selected dataset is already hosted there. Google Colab remains an
optional alternative, and the same repository modules must also run locally as
a slower fallback. No remote environment is an inference dependency: the
deployed FastAPI service loads a finished model bundle from its own filesystem.

## Foundation contracts

- `configs/base.yaml` is the readable central configuration file.
- `src/preprocessing/image_contract.py` defines deterministic 224 x 224 RGB
  image preparation shared by later training, evaluation, and inference code.
- `src/data/contracts.py` defines the small records used for dataset identity,
  samples, and split manifests.
- `src/models/artifact.py` defines metadata required to safely identify a
  production model bundle.

## Data and model policy

The selected source is Kaggle dataset
`bhavikjikadara/dog-and-cat-classification-dataset`. Raw and processed images
remain out of Git and are versioned through DVC. Small deterministic manifests,
configuration, and validation metadata remain in Git.

The full Kaggle dataset is the authoritative DVC-versioned dataset, while a
deterministic balanced 50% subset is used for the assignment's baseline
training experiment to reduce iteration time. The subset contains 6,240 cats
and 6,240 dogs: 9,984 training, 1,248 validation, and 1,248 test images. Exact
membership and hashes are recorded under `data/manifests/baseline_50/`; no
second physical image copy is created locally.

The selected, small production model bundle may be committed to Git so a Docker
image can contain a known valid artifact. Training checkpoints and experiment
output will not be committed.

## Development philosophy

Clarity over cleverness is a standing project requirement. Code should be easy
to read, explicit about assumptions, and simple enough to explain in a review.
Complexity is added only when it protects correctness, reproducibility, or a
real assignment requirement.

## Core commands

```powershell
python -m scripts.validate_baseline_subset --manifests-dir data/manifests/baseline_50
python -m src.training.train --config configs/base.yaml --promote-to-production
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root mlartifacts --host 127.0.0.1 --port 5000 --workers 1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
docker compose up --build -d
python -m scripts.smoke_test
python -m scripts.post_deployment_evaluate
python -m scripts.package_submission
```

The Kaggle notebook at `kaggle/kernel/training_kaggle.ipynb` calls these same
repository modules. It contains no second preprocessing, model, or training
implementation.

## Verified baseline

The production SimpleCNN bundle is traceable to MLflow run
`462a190825b548c7b6d5600724b42cca`. Its held-out test accuracy is 0.80048 and
macro F1 is 0.79990. The local Docker Compose service has passed both `/health`
and real-image `/predict` smoke checks. See `docs/submission-guide.md` for the
evidence map, reproduction commands, deployment flow, and remaining external
GitHub setup.
