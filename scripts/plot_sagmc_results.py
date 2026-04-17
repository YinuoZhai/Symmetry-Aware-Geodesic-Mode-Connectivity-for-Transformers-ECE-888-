from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _curve_from_coeff_losses(losses_by_coeff: dict[str, float]) -> tuple[list[float], list[float]]:
    rows = []
    for coeff, loss in losses_by_coeff.items():
        # GLMC convention: coeff=1 is theta_A, coeff=0 is aligned theta_B.
        alpha = 1.0 - float(coeff)
        rows.append((alpha, float(loss)))
    rows.sort(key=lambda item: item[0])
    return [row[0] for row in rows], [row[1] for row in rows]


def _curve_from_sagmc_metrics(metrics: dict) -> tuple[list[float], list[float]]:
    points = sorted(metrics["points"], key=lambda point: float(point["alpha"]))
    return [float(point["alpha"]) for point in points], [float(point["loss"]) for point in points]


def _barrier(xs: list[float], ys: list[float]) -> float:
    start = ys[0]
    end = ys[-1]
    return max(loss - ((1.0 - alpha) * start + alpha * end) for alpha, loss in zip(xs, ys))


def _load_energy_log(path: str | Path) -> tuple[list[int], list[float]]:
    steps = []
    energies = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            steps.append(int(row["step"]))
            energies.append(float(row["train_jsd_energy"]))
    return steps, energies


def _style_axes(ax) -> None:
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_loss_comparison(glmc_json: Path, sagmc_metrics: Path, out_dir: Path) -> dict[str, float]:
    glmc = _load_json(glmc_json)
    sagmc = _load_json(sagmc_metrics)

    curves = {
        "Naive linear": _curve_from_coeff_losses(glmc["coeff_losses_vanilla"]),
        "GLMC weight": _curve_from_coeff_losses(glmc["coeff_losses_weight_matching"]),
        "GLMC learned": _curve_from_coeff_losses(glmc["coeff_losses_learned_matching"]),
        "SA-GMC medium": _curve_from_sagmc_metrics(sagmc),
    }
    barriers = {name: _barrier(xs, ys) for name, (xs, ys) in curves.items()}

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    styles = {
        "Naive linear": {"color": "#b43c3c", "marker": "o", "linestyle": "--"},
        "GLMC weight": {"color": "#d08a25", "marker": "s", "linestyle": "-."},
        "GLMC learned": {"color": "#26734d", "marker": "^", "linestyle": "-"},
        "SA-GMC medium": {"color": "#2458a6", "marker": "D", "linestyle": "-"},
    }
    for name, (xs, ys) in curves.items():
        ax.plot(
            xs,
            ys,
            linewidth=2.0,
            markersize=5,
            label=f"{name} (B={max(0.0, barriers[name]):.3f})",
            **styles[name],
        )

    ax.set_xlabel("Path position: theta_A to aligned theta_B")
    ax.set_ylabel("Test loss")
    ax.set_title("Tiny Shakespeare Mode Connectivity")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "tiny_shakespeare_loss_comparison.png", dpi=220)
    fig.savefig(out_dir / "tiny_shakespeare_loss_comparison.pdf")
    plt.close(fig)
    return barriers


def plot_energy(energy_log: Path, out_dir: Path) -> None:
    steps, energies = _load_energy_log(energy_log)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(steps, energies, color="#2458a6", marker="o", linewidth=2.0, markersize=4)
    ax.set_xlabel("SA-GMC optimization step")
    ax.set_ylabel("Train JSD energy")
    ax.set_title("SA-GMC Energy During Optimization")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "sa_gmc_jsd_energy.png", dpi=220)
    fig.savefig(out_dir / "sa_gmc_jsd_energy.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Tiny Shakespeare GLMC and SA-GMC results.")
    parser.add_argument(
        "--glmc-json",
        default="eval_merge_seed_0_1/merged_coeff_losses_weight_learned_vanilla.json",
    )
    parser.add_argument(
        "--sagmc-metrics",
        default="runs/glmc_tiny_shakespeare/sa_gmc_medium_lr1e-3/eval/metrics.json",
    )
    parser.add_argument(
        "--energy-log",
        default="runs/glmc_tiny_shakespeare/sa_gmc_medium_lr1e-3/geodesic_metrics.jsonl",
    )
    parser.add_argument("--out-dir", default="runs/glmc_tiny_shakespeare/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    barriers = plot_loss_comparison(Path(args.glmc_json), Path(args.sagmc_metrics), out_dir)
    plot_energy(Path(args.energy_log), out_dir)

    summary = {
        "barriers": {name: max(0.0, value) for name, value in barriers.items()},
        "glmc_json": args.glmc_json,
        "sagmc_metrics": args.sagmc_metrics,
        "energy_log": args.energy_log,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {out_dir / 'tiny_shakespeare_loss_comparison.png'}")
    print(f"wrote {out_dir / 'sa_gmc_jsd_energy.png'}")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
