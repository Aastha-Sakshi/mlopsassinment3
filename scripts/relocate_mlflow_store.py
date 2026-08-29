"""Relocate filesystem artifact URIs in an exported MLflow SQLite store."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("mlruns/mlflow.db"))
    parser.add_argument("--artifact-root", type=Path, default=Path("mlartifacts"))
    parser.add_argument(
        "--old-prefix",
        default="file:///kaggle/working/mlops_binary_class/mlartifacts",
    )
    args = parser.parse_args()
    new_prefix = args.artifact_root.resolve().as_uri()
    with sqlite3.connect(args.database) as connection:
        experiment_updates = connection.execute(
            "UPDATE experiments SET artifact_location = REPLACE(artifact_location, ?, ?) "
            "WHERE artifact_location LIKE ?",
            (args.old_prefix, new_prefix, f"{args.old_prefix}%"),
        ).rowcount
        run_updates = connection.execute(
            "UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, ?) WHERE artifact_uri LIKE ?",
            (args.old_prefix, new_prefix, f"{args.old_prefix}%"),
        ).rowcount
        connection.commit()
    print(f"Relocated {experiment_updates} experiment URI(s) and {run_updates} run URI(s) to {new_prefix}")


if __name__ == "__main__":
    main()
