"""Fail-fast deployed API health and prediction smoke test."""

from __future__ import annotations

import argparse
import json
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-seconds", type=float, default=5.0)
    args = parser.parse_args()
    health_payload = None
    last_error: Exception | None = None
    for attempt in range(1, args.attempts + 1):
        try:
            health = requests.get(f"{args.base_url}/health", timeout=15)
            health.raise_for_status()
            health_payload = health.json()
            if health_payload.get("model_loaded"):
                break
            last_error = RuntimeError(f"Service is not ready: {health_payload}")
        except (requests.RequestException, ValueError) as error:
            last_error = error
        if attempt < args.attempts:
            time.sleep(args.retry_seconds)
    else:
        raise RuntimeError(f"Service did not become ready after {args.attempts} attempts") from last_error

    ppm_image = b"P6\n1 1\n255\n" + bytes((128, 128, 128))
    prediction = requests.post(
        f"{args.base_url}/predict",
        files={"file": ("smoke.ppm", ppm_image, "image/x-portable-pixmap")},
        timeout=30,
    )
    prediction.raise_for_status()
    payload = prediction.json()
    if payload.get("label") not in {"cat", "dog"} or set(payload.get("probabilities", {})) != {"cat", "dog"}:
        raise RuntimeError(f"Unexpected prediction response: {payload}")
    print(json.dumps({"health": health_payload, "prediction": payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
