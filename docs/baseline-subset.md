# Baseline training population

The full Kaggle dataset remains the authoritative dataset. DVC tracks both the
validated raw dataset and the complete 24,966-image processed 224 x 224 RGB
artifact.

The assignment baseline uses a deterministic balanced 50% subset to reduce
iteration time while retaining enough data for a credible CNN experiment. No
subset image directory is created. Training reads the Git-tracked manifests in
`data/manifests/baseline_50/` and resolves those identifiers against the full
processed DVC artifact.

## Contract

- selection algorithm: `sha256-ranked-per-existing-split-class-v1`
- selection seed: `20260829`
- source split seed: `42`
- train: 9,984 images, 4,992 per class
- validation: 1,248 images, 624 per class
- test: 1,248 images, 624 per class
- combined subset manifest SHA-256:
  `55c208fea1fb4fea2043dab00db62ab012cf1fa851a761b01526c96fd6d4b8a9`
- processed DVC directory hash:
  `6fa0f52466ef6aa1583102e520c47980.dir`

Selection ranks each existing split/class independently by SHA-256 of the
selection seed, split, class, and relative image path. The fixed quota is then
taken from that ordering. This preserves the approved split boundaries and
guarantees exact class balance without data overlap.
