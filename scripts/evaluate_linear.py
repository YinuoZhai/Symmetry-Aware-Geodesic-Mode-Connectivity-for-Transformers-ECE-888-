from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.checkpoint import load_model_and_state, load_state_dict
from ece888_sagmc.config import load_config
from ece888_sagmc.data import build_dataset
from ece888_sagmc.interpolate import evaluate_linear_path, write_path_metrics
from ece888_sagmc.plotting import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a linear interpolation path.")
    parser.add_argument("--config", default="configs/tiny_shakespeare.yaml")
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="linear")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(cfg["data"])
    model, state_a, model_config, _meta = load_model_and_state(args.ckpt_a)
    state_b = load_state_dict(args.ckpt_b)

    num_points = args.num_points or int(cfg["eval"]["num_points"])
    eval_iters = args.eval_iters or int(cfg["eval"]["eval_iters"])
    barrier, points = evaluate_linear_path(
        model=model,
        state_a=state_a,
        state_b=state_b,
        dataset=dataset,
        split=str(cfg["eval"]["split"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        eval_iters=eval_iters,
        num_points=num_points,
        device=args.device,
    )
    write_path_metrics(
        args.out,
        barrier,
        points,
        extra={"method": args.name, "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b},
    )
    plot_metrics(Path(args.out) / "metrics.json", Path(args.out) / "loss_curve.png", args.name)
    print(f"{args.name} barrier: {barrier:.6f}")
    print(f"model_config: {model_config}")


if __name__ == "__main__":
    main()
