from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

def write_history_csv(path: str | Path, history: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("history is empty")

    fieldnames = list(history[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in history:
            w.writerow(row)



def write_snapshots_npz(path, snapshots: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = {}
    for label, snap in snapshots.items():
        for key, val in snap.items():
            flat[f"{label}__{key}"] = val
    np.savez(path, **flat)


def write_metadata_json(path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metadata, f, indent=2)
