# Foundation decisions

## Approach and prototype preservation

We are taking a hybrid approach. The friend's prototype provides a useful
starting reference, especially its baseline model and MLOps coverage, but it
has reproducibility and deployment-artifact gaps. The original ZIP is preserved
unchanged under `prototype_original/` so it remains clearly separate from this
implementation.

## Technical direction

- **Baseline model:** retain a simple CNN as the first model because it meets
  the assignment requirement and is feasible on local hardware. Transfer
  learning is deferred.
- **Python environment:** use a project-local Python 3.11 environment once
  dependencies are approved. The currently installed global Python 3.14 is not
  the compatible target for the prototype's pinned PyTorch version.
- **Dataset versioning:** plan to use DVC for the real dataset and keep small,
  deterministic split manifests in Git.
- **Experiment tracking:** plan to use local MLflow because it visibly records
  parameters, metrics, plots, and artifacts without requiring a cloud service.
- **API:** plan to use FastAPI for explicit request validation, typed responses,
  and easy endpoint testing.
- **Deployment:** Docker Compose on a persistent local self-hosted runner is
  currently preferred over Kubernetes because it is simpler to reproduce and
  demonstrate.

## Deliberate unknowns and contracts

The assignment does not specify one Kaggle dataset URL or slug. The source is
therefore intentionally `UNRESOLVED` in configuration until it is supplied or
confirmed. No synthetic data will be treated as final training data.

Before Docker or CI exists, the project defines a production model metadata
contract. A later bundle will contain `model.pt`, `metadata.json`, and a
checksum file. This prevents the API and container from silently depending on
an absent or unexplained model.

## Engineering principle

Clarity over cleverness is a standing requirement. We favour direct functions,
small data structures, explicit validation, and standard ML patterns over
unnecessary frameworks or abstraction. Optimisation is deferred until a real
measured bottleneck requires it.

## Deferred work

Dataset acquisition, DVC, splitting, training, MLflow, FastAPI, Docker, CI/CD,
deployment, monitoring, online evaluation, and video automation are deferred to
later phases. They will be implemented only after the local data-to-model path
is working and verifiable.

The Phase 2 compute and data architecture is documented separately in
`phase-2-architecture.md`. It keeps the repository as the source of truth,
allows Colab as an optional training environment, and keeps inference local and
independent of Colab.
