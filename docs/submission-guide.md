# Submission and demonstration guide

## Evidence map

| Requirement | Repository evidence |
| --- | --- |
| M1 data versioning | `data/downloads/PetImages.dvc`, `data/processed/PetImages224.dvc`, manifests, validation scripts |
| M1 baseline and tracking | `src/training/`, `models/production/`, `mlruns/mlflow.db`, `mlartifacts/`, training plots |
| M2 API and container | `src/api/main.py`, `Dockerfile`, `requirements-api.txt`, `compose.yaml` |
| M3 testing and CI | `tests/`, `.github/workflows/ci.yml` |
| M4 deployment and smoke gate | `.github/workflows/cd.yml`, `scripts/smoke_test.py` |
| M5 monitoring and evaluation | `/metrics`, structured prediction logs, `scripts/post_deployment_evaluate.py` |

## Verified dataset and baseline

The full Kaggle dataset is the authoritative DVC-versioned dataset. It produced
24,966 valid processed 224 x 224 RGB images. Baseline training uses a
manifest-only deterministic balanced subset; it does not create another image
copy.

- subset: 12,480 images, 6,240 cats and 6,240 dogs
- split: 9,984 train, 1,248 validation, 1,248 test
- subset manifest SHA-256: `55c208fea1fb4fea2043dab00db62ab012cf1fa851a761b01526c96fd6d4b8a9`
- source fingerprint: `b17077327cf872b4dff77102d6e5463f97053445c7e7e2452895aac045bfef82`
- processed DVC hash: `6fa0f52466ef6aa1583102e520c47980.dir`
- MLflow run: `462a190825b548c7b6d5600724b42cca`
- test accuracy: 0.80048; macro F1: 0.79990

## Local demonstration

Use the project Python 3.11 environment. The MLflow server is deliberately
localhost-only and uses one worker for reliable Windows execution.

```powershell
.\.venv\Scripts\Activate.ps1
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root mlartifacts --host 127.0.0.1 --port 5000 --workers 1
docker compose up -d --no-build
python -m scripts.smoke_test --base-url http://127.0.0.1:8000
python -m scripts.post_deployment_evaluate --base-url http://127.0.0.1:8000
```

Open `http://127.0.0.1:5000` for the experiment/run/metrics/artifact UI,
`http://127.0.0.1:8000/docs` for interactive API evidence, and
`http://127.0.0.1:8000/metrics` for lightweight service monitoring.

## CI/CD contract

CI runs tests, validates the committed production bundle, builds the image,
and publishes immutable commit-SHA and `main` tags to GHCR on a main-branch
push. CD starts only after successful CI, checks out the exact successful SHA,
pulls that immutable image on a persistent self-hosted runner, deploys through
Compose, and fails if health or prediction smoke testing fails.

The runner label must be `mlops-deploy`. Register the runner only on a trusted,
persistent host; repository collaborators can modify workflows that execute on
it. Do not place GitHub, Kaggle, or MLflow credentials in the repository.

Final verified automation evidence:

- CI run: `33260462683` — tests, model validation, image build, and GHCR publish passed
- CD run: `33260564239` — immutable image pull, Compose deployment, and smoke test passed
- deployed image: `ghcr.io/aastha-sakshi/cats-dogs-api:87ada8b46344e7370c8b36b52fefe02b5c44bd84`

## Demonstration video

The verified final file is `docs/demo.mp4`: H.264, 1920 x 1080, 264.80
seconds (4:24.80), with no black section lasting two seconds or longer. It is
a silent browser recording with burned-in, human-written subtitles and subtle
deterministic cursor and scrolling motion. It contains no credentials or
desktop notifications.

## Packaging

```powershell
python -m scripts.package_submission
```

This creates `dist/mlops-cats-dogs-assignment2.zip` from an explicit allowlist.
It includes source, configurations, DVC pointers/manifests, tests, workflows,
the production bundle, and compact experiment evidence. Raw/processed images,
the DVC cache, virtual environment, credentials, and transient Kaggle downloads
are excluded.
