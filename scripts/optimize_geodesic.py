from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.checkpoint import load_model_and_state, load_state_dict
from ece888_sagmc.config import load_config
from ece888_sagmc.data import build_dataset
from ece888_sagmc.geodesic import (
    evaluate_waypoint_states,
    load_geodesic_path,
    optimize_geodesic,
    write_geodesic_eval,
)
from ece888_sagmc.model import GPT, GPTConfig
from ece888_sagmc.plotting import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a GMC/SA-GMC geodesic path.")
    parser.add_argument("--config", default="configs/tiny_shakespeare.yaml")
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", default="gmc", choices=["gmc", "sa_gmc"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-waypoints", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.num_waypoints is not None:
        cfg["geodesic"]["num_waypoints"] = args.num_waypoints
    if args.iterations is not None:
        cfg["geodesic"]["iterations"] = args.iterations
    if args.batch_size is not None:
        cfg["geodesic"]["batch_size"] = args.batch_size

    dataset = build_dataset(cfg["data"])
    model, state_a, model_config, _meta = load_model_and_state(args.ckpt_a)
    state_b = load_state_dict(args.ckpt_b)

    path_file = optimize_geodesic(
        model=model,
        state_a=state_a,
        state_b=state_b,
        dataset=dataset,
        cfg=cfg["geodesic"],
        out_dir=args.out_dir,
        device=args.device,
        metadata={"method": args.method, "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b},
    )

    path_obj = load_geodesic_path(path_file)
    eval_model = GPT(GPTConfig.from_dict(model_config))
    eval_iters = args.eval_iters or int(cfg["eval"]["eval_iters"])
    barrier, rows = evaluate_waypoint_states(
        model=eval_model,
        states=path_obj["waypoints"],
        dataset=dataset,
        split=str(cfg["eval"]["split"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        eval_iters=eval_iters,
        device=args.device,
    )
    write_geodesic_eval(
        Path(args.out_dir) / "eval",
        barrier,
        rows,
        extra={"method": args.method, "path": str(path_file)},
    )
    plot_metrics(
        Path(args.out_dir) / "eval" / "metrics.json",
        Path(args.out_dir) / "eval" / "loss_curve.png",
        args.method,
    )
    print(f"{args.method} path: {path_file}")
    print(f"{args.method} barrier: {barrier:.6f}")


if __name__ == "__main__":
    main()
