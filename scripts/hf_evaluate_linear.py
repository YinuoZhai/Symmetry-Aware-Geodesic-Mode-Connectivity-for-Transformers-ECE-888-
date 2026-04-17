from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.config import load_glmc_config
from ece888_sagmc.hf_gpt2 import (
    build_hf_data,
    evaluate_hf_linear_path,
    load_hf_model_and_state,
    load_hf_state_dict,
    complete_state_dict,
    write_hf_metrics,
)
from ece888_sagmc.plotting import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official-GLMC-style GPT-2 linear path.")
    parser.add_argument("--config", default="configs/glmc_tiny_shakespeare.yaml")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_glmc_config(args.config)
    model, state_a = load_hf_model_and_state(args.model_a)
    raw_b = load_hf_state_dict(args.model_b)
    state_b = complete_state_dict(model, raw_b)
    data = build_hf_data(cfg["data"], block_size=int(cfg["model"]["block_size"]))
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = cfg["eval"].get("max_batches")
    barrier, rows = evaluate_hf_linear_path(
        model=model,
        state_a=state_a,
        state_b=state_b,
        data=data,
        split=str(cfg["eval"]["split"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        coeff_start=float(cfg["eval"]["coeff_start"]),
        coeff_end=float(cfg["eval"]["coeff_end"]),
        coeff_step=float(cfg["eval"]["coeff_step"]),
        device=args.device,
        max_batches=max_batches,
    )
    write_hf_metrics(
        args.out_dir,
        barrier,
        rows,
        extra={"model_a": args.model_a, "model_b": args.model_b, "method": "hf_linear"},
    )
    plot_metrics(Path(args.out_dir) / "metrics.json", Path(args.out_dir) / "loss_curve.png", "HF linear")
    print(f"hf linear barrier: {barrier:.6f}")


if __name__ == "__main__":
    main()
