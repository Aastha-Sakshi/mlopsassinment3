# Phase 2 architecture reassessment

## Compute environment decision

The repository remains the source of truth. Remote environments execute the
same repository code; they are not part of the deployed application.

| Capability | Local Windows / MX150 | Google Colab | Kaggle Compute |
| --- | --- | --- | --- |
| GPU availability | Fixed local MX150, 4 GB VRAM | Variable and not guaranteed | GPU option is available, subject to quota/queue |
| Training practicality | Suitable for small/local fallback runs | Suitable when an assigned GPU is available | Preferred for baseline remote runs because the dataset is a native input |
| Session persistence | Local disk persists | VM is ephemeral | Notebook versions and `/kaggle/working` outputs can be saved; sessions remain ephemeral |
| Dataset access | Download through authenticated Kaggle tooling | Authenticated download required | Attach the selected Kaggle dataset as a Notebook input |
| VS Code workflow | Native | Existing extension/workflow | Experimental external-editor connection to Kaggle Jupyter Server is documented |
| Git integration | Native | Clone/install repository in notebook | Clone/install repository in notebook; no notebook is the source of truth |
| DVC compatibility | Full local DVC workflow | Validate Kaggle download against repository fingerprint | Validate attached Kaggle data against repository fingerprint |
| MLflow compatibility | Direct local persistent server | Result-package import unless secure remote server exists | Same result-package import approach; Kaggle does not make MLflow persistent |
| Artifact export | Native filesystem | Explicit download/transfer required | Save output, then explicitly retrieve/verify it |
| Cost/free-tier practicality | No extra service cost | Availability and limits vary | Free GPU quota is documented as weekly and demand-dependent |
| Demonstration role | Final Docker/API deployment | Optional completed-run evidence | Optional completed-run evidence |

**Primary training environment: Kaggle Compute.** It provides direct access to
the selected dataset, versioned notebook outputs, documented GPU options, and
an experimental route to a Kaggle Jupyter Server from VS Code. This reduces
credential and data-transfer friction for baseline training.

**Required fallback: local Windows.** It is the only environment that remains
available without a remote-session quota. The MX150 is not the preferred
full-dataset trainer, but the same command must work there for reproducibility.

**Optional alternative: Google Colab.** Use it only if Kaggle GPU capacity is
unavailable or a specific experiment needs it. Colab has variable accelerator
availability and lifetime, and it needs a separate authenticated dataset
download. It is not the primary path for this Kaggle-hosted dataset.

Kaggle's published documentation lists notebook GPU options, including a Tesla
P100 and T4 x2, but the actual assigned accelerator and usable memory must be
recorded at runtime rather than assumed. The current Kaggle documentation also
describes a weekly GPU quota that can vary with demand. The exact Colab GPU and
limits are likewise variable.

## Recommended execution model

| Location | Responsibility |
| --- | --- |
| Local VS Code | Source code, configuration, data validation, DVC, tests, local training fallback, MLflow UI, Docker inference, documentation |
| Kaggle Compute | Attach the selected dataset, clone the repository, run the same training/evaluation modules, and produce a validated model bundle |
| Google Colab | Optional alternative when a suitable GPU is available |
| GitHub Actions | Fast tests, code checks, Docker build and later image publishing; no full training or Kaggle download |
| Docker / Compose | Load a finished model bundle and run FastAPI independently of Colab |

The MX150 has only 4 GB of VRAM. It is adequate for development checks and a
small baseline with conservative settings, but it is not the preferred machine
for repeated full-dataset experimentation. Kaggle Compute is sensible for those
runs, provided a GPU is actually available. The local command path remains a
required fallback.

## Colab notebook strategy

A future `notebooks/training_kaggle.ipynb` will be a thin orchestration notebook:

```text
clone repository -> install pinned project environment -> load config
-> obtain/verify data -> call project training module -> evaluate
-> write model bundle -> transfer verified bundle
```

It will not duplicate preprocessing, model, training, or evaluation code in
notebook cells. The notebook will attach the Kaggle dataset as input and clone
the repository before calling the same modules as the local fallback. A future
Colab notebook, if needed, will use the same shape.

## Selected dataset and validation boundary

The selected source is Kaggle dataset
`bhavikjikadara/dog-and-cat-classification-dataset`, Version 1. Kaggle's
published metadata reports about 848.87 MB and 24,998 images under
`PetImages/Cat` and `PetImages/Dog`, with 12,499 images per class. The published
metadata does not establish a labelled train/validation/test layout that we can
use without inspection.

Therefore, Phase 2 will validate the downloaded files and create our own seeded,
class-stratified 80/10/10 split. It will record stable image identifiers and
reject duplicate split membership. File formats, corrupt files, duplicate image
content, and any source-supplied metadata remain unverified until the dataset is
actually downloaded.

## Dataset and DVC lifecycle

```text
Kaggle versioned download
  -> raw source directory, DVC-tracked
  -> validation report and content fingerprint
  -> Git-tracked split manifests
  -> deterministic processed split, DVC-tracked
  -> training/evaluation
```

Git will store source code, DVC pointer files, `dvc.yaml`, `dvc.lock`, validation
rules, and small manifests. It will not store raw or processed images. DVC will
track both raw and processed data so the assignment's data and preprocessing
versioning requirement is real rather than decorative.

For local development, use a local DVC remote outside the Git repository. A
future shared DVC remote is optional; it is not required before the local
pipeline works. Neither Kaggle nor Colab must rely on this local-only remote.
Kaggle Compute will validate its attached dataset against the repository's
accepted fingerprint. Colab, if used, will download the specified Kaggle version
with user-supplied credentials and run the same validation.

## MLflow architecture

MLflow has two separate parts: the tracking server, which stores run metadata
and artifacts, and the browser UI, which is served by that same server.

For local training, the initial design is a persistent local MLflow server with
a SQLite backend and an artifact directory on the local disk. The browser opens
the local UI. This is persistent across local commands and does not require a
cloud service.

Neither Kaggle nor Colab can securely reach a laptop-only `localhost` MLflow
server. We will not expose the laptop through an unauthenticated public tunnel.
Until a persistent authenticated HTTPS endpoint is deliberately configured,
remote runs will save their exact metrics, plots, and model bundle in a result
package. After controlled transfer, an explicit import step will record those
artifacts in the local MLflow server with `execution_environment=kaggle` or
`execution_environment=colab`. This is a traceable post-run import, not a claim
of live remote tracking.

If live Colab logging becomes necessary later, the required upgrade is a
persistent MLflow server on a deliberately configured private/HTTPS-accessible
host with durable backend and artifact storage. That is an external deployment
decision, not a Phase 2 prerequisite.

## Model artifact lifecycle

```text
same project training code (Kaggle preferred, local fallback, Colab optional)
  -> evaluation and immutable run result
  -> MLflow logging or explicit verified import
  -> models/production/model.pt + metadata.json + SHA256SUMS
  -> Git promotion of the selected small baseline bundle
  -> Docker image validation
  -> FastAPI inference
```

The bundle metadata will link model architecture, preprocessing contract,
configuration hash, dataset fingerprint, seed, MLflow run identifier, and
evaluation results. A Colab-produced bundle is transferred with its checksum
and validation report; it is never accepted merely because a file exists in
Google Drive.

## Inference and deployment

The inference container will only load a validated local model bundle. Neither
FastAPI, Docker, CI, nor deployment will call an active Colab runtime. Docker
Compose on a persistent local/self-hosted target remains the preferred later
deployment option.

## Video implications

The final under-five-minute recording will show the actual promoted artifact,
Docker deployment, API health/prediction, metrics, and online evaluation. It
will not require an active Colab session. If useful, a short recorded browser
view of the completed MLflow run can show the remote-training evidence without
making the video depend on live GPU availability.

## Deferred Phase 2 implementation

## Phase 2A results and next increments

Phase 2A validated the user-provided local extraction at
`data/downloads/PetImages`. It found exactly 24,998 readable JPEG files: 12,499
under `Cat` and 12,499 under `Dog`. There were no invalid images. The report
found 28 exact-byte duplicate groups (58 files) and recorded the accepted
fingerprint
`b17077327cf872b4dff77102d6e5463f97053445c7e7e2452895aac045bfef82`.
The source layout has class directories only; it does not supply usable
train/validation/test splits.

Phase 2B configures DVC for raw data and creates deterministic, duplicate-aware
80/10/10 manifests plus invariant tests. Its detailed lineage contract is in
`docs/phase-2b-data-lineage.md`. Phase 2C will materialize only the necessary
224×224 RGB processed dataset and DVC-track that output. No training, MLflow,
API, Docker, CI/CD, or deployment work is part of either phase.
