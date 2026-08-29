# Phase 2B — raw-data lineage and split manifests

This phase starts from the locally validated extraction at
`data/downloads/PetImages`. It does not resize images, create processed data,
or train a model.

## Accepted source

- Kaggle source: `bhavikjikadara/dog-and-cat-classification-dataset`
- Kaggle dataset version: `1`
- locally verified fingerprint:
  `b17077327cf872b4dff77102d6e5463f97053445c7e7e2452895aac045bfef82`
- validation report: `artifacts/dataset_validation.json` (local artifact)

The fingerprint is a SHA-256 over each sorted relative path, byte size, and
per-file SHA-256. It identifies this exact accepted extraction, not merely the
Kaggle dataset name.

## Versioning boundaries

Git stores the generator, configuration, DVC metadata, and the small split
manifests. DVC stores the large raw image directory through
`data/downloads/PetImages.dvc`; the images themselves remain ignored by Git.
The local DVC cache uses hardlinks on this volume so raw tracking does not keep
an unnecessary permanent duplicate beside the working extraction.

No remote is configured in this phase. A future remote, if needed, must be
configured deliberately outside Git and must not contain credentials in tracked
configuration.

## Duplicate policy

The Phase 2A validator found 28 exact-byte duplicate groups containing 58
files. The deterministic split generator applies this policy before splitting:

- Same-label exact duplicates: keep only the lexicographically first relative
  path and record every omitted duplicate in `manifest_metadata.json`.
- Cross-label exact duplicates: exclude every path in the group. These samples
  have contradictory labels and must not enter any split.
- Near/perceptual duplicates: not measured in Phase 2A and not claimed to be
  resolved.

This removes 32 paths from split eligibility: 26 same-label duplicate copies
and 6 paths from two cross-label groups. The 24,966 remaining samples are split
independently per class with seed `42`.

## Reproducible commands

Run from the repository root after `dvc pull` (or when the accepted raw
extraction is already present):

```powershell
python -m scripts.create_split_manifests `
  --data-dir data/downloads/PetImages `
  --validation-report artifacts/dataset_validation.json `
  --output-dir data/manifests `
  --seed 42 `
  --dataset-source bhavikjikadara/dog-and-cat-classification-dataset `
  --dataset-version 1

python -m scripts.verify_split_manifests `
  --data-dir data/downloads/PetImages `
  --manifests-dir data/manifests
```

The first command writes `train.csv`, `validation.csv`, `test.csv`, and
`manifest_metadata.json`. The second verifies that all paths exist, paths are
unique, and splits do not overlap.

## Intentional Phase 2C boundary

Phase 2C will use these committed manifests to build one canonical 224×224 RGB
processed dataset, DVC-track it, and avoid creating normalized or augmented
copies. Augmentation remains a future training-time transformation. Nothing in
this phase creates processed images.
