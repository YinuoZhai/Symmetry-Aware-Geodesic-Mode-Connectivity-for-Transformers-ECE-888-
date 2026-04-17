from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.config import load_glmc_config
from ece888_sagmc.hf_gpt2 import (
    build_hf_data,
    complete_state_dict,
    evaluate_hf_waypoint_path,
    load_hf_geodesic_path,
    load_hf_model_and_state,
    load_hf_state_dict,
    optimize_hf_geodesic,
    write_hf_metrics,
)
from ece888_sagmc.plotting import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a geodesic for official-GLMC-style GPT-2 dirs.")
    parser.add_argument("--config", default="configs/glmc_tiny_shakespeare.yaml")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", default="gmc", choices=["gmc", "sa_gmc"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-waypoints", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_glmc_config(args.config)
    if args.iterations is not None:
        cfg["geodesic"]["iterations"] = args.iterations
    if args.num_waypoints is not None:
        cfg["geodesic"]["num_waypoints"] = args.num_waypoints
    if args.batch_size is not None:
        cfg["geodesic"]["batch_size"] = args.batch_size
    if args.sequence_length is not None:
        cfg["geodesic"]["sequence_length"] = args.sequence_length
    if args.learning_rate is not None:
        cfg["geodesic"]["learning_rate"] = args.learning_rate
    if args.momentum is not None:
        cfg["geodesic"]["momentum"] = args.momentum
    if args.grad_clip is not None:
        cfg["geodesic"]["grad_clip"] = args.grad_clip
    if args.log_interval is not None:
        cfg["geodesic"]["log_interval"] = args.log_interval

    model, state_a = load_hf_model_and_state(args.model_a)
    raw_b = load_hf_state_dict(args.model_b)
    state_b = complete_state_dict(model, raw_b)
    data = build_hf_data(cfg["data"], block_size=int(cfg["model"]["block_size"]))

    path_file = optimize_hf_geodesic(
        model=model,
        state_a=state_a,
        state_b=state_b,
        data=data,
        cfg=cfg["geodesic"],
        out_dir=args.out_dir,
        device=args.device,
        metadata={"method": args.method, "model_a": args.model_a, "model_b": args.model_b},
    )
    path_obj = load_hf_geodesic_path(path_file)
    eval_model = GPT2LMHeadModel(GPT2Config.from_dict(path_obj["model_config"]))
    eval_model.config.use_cache = False
    max_batches = args.max_eval_batches
    if max_batches is None:
        max_batches = cfg["eval"].get("max_batches")
    barrier, rows = evaluate_hf_waypoint_path(
        model=eval_model,
        states=path_obj["waypoints"],
        data=data,
        split=str(cfg["eval"]["split"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        device=args.device,
        max_batches=max_batches,
    )
    eval_dir = Path(args.out_dir) / "eval"
    write_hf_metrics(eval_dir, barrier, rows, extra={"method": args.method, "path": str(path_file)})
    plot_metrics(eval_dir / "metrics.json", eval_dir / "loss_curve.png", f"HF {args.method}")
    print(f"{args.method} path: {path_file}")
    print(f"{args.method} barrier: {barrier:.6f}")


if __name__ == "__main__":
    main()
