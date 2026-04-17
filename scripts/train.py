from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.config import load_config
from ece888_sagmc.data import build_dataset
from ece888_sagmc.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Tiny Shakespeare GPT checkpoint.")
    parser.add_argument("--config", default="configs/tiny_shakespeare.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-iters", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_iters is not None:
        cfg["train"]["max_iters"] = args.max_iters
    dataset = build_dataset(cfg["data"])
    best_path = train_model(
        dataset=dataset,
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
    )
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
