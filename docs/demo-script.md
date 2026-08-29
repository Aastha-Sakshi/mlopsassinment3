# Demonstration storyline

The generated browser recording is designed to remain below five minutes and
show completed evidence instead of long-running training or installation.

1. Project and end-to-end workflow
2. Full DVC-versioned dataset and deterministic 12,480-image baseline subset
3. MLflow run, confusion matrix, loss curve, and measured test results
4. Public GitHub repository and successful CI run
5. GHCR image and successful self-hosted CD run
6. FastAPI documentation
7. Known cat, known dog, and invalid-image requests
8. Request counts, errors, latency, and access logs
9. Labelled post-deployment evaluation
10. M1–M5 summary

The recorder uses a fixed 1920×1080 Playwright viewport and captures browser
content only. It does not record the desktop, terminals, notifications,
credentials, GitHub settings, or runner registration pages. The final MP4 is
validated with FFprobe and FFmpeg.
