from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(metrics_path: str | Path, out_path: str | Path, title: str) -> None:
    with Path(metrics_path).open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    points = metrics["points"]
    xs = [p["alpha"] for p in points]
    losses = [p["loss"] for p in points]
    baseline = [p["baseline_loss"] for p in points]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, losses, marker="o", label="path loss")
    plt.plot(xs, baseline, linestyle="--", label="endpoint linear baseline")
    plt.xlabel("path position")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
