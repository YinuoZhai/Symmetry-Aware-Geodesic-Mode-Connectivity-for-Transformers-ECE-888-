from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.config import load_config
from ece888_sagmc.data import build_dataset
from ece888_sagmc.geodesic import evaluate_waypoint_states, load_geodesic_path, write_geodesic_eval
from ece888_sagmc.model import GPT, GPTConfig
from ece888_sagmc.plotting import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved geodesic path.")
    parser.add_argument("--config", default="configs/tiny_shakespeare.yaml")
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-iters", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(cfg["data"])
    path_obj = load_geodesic_path(args.path)
    model = GPT(GPTConfig.from_dict(path_obj["model_config"]))
    eval_iters = args.eval_iters or int(cfg["eval"]["eval_iters"])
    barrier, rows = evaluate_waypoint_states(
        model=model,
        states=path_obj["waypoints"],
        dataset=dataset,
        split=str(cfg["eval"]["split"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        eval_iters=eval_iters,
        device=args.device,
    )
    write_geodesic_eval(args.out, barrier, rows, extra={"path": args.path})
    plot_metrics(Path(args.out) / "metrics.json", Path(args.out) / "loss_curve.png", "geodesic")
    print(f"geodesic barrier: {barrier:.6f}")


if __name__ == "__main__":
    main()
