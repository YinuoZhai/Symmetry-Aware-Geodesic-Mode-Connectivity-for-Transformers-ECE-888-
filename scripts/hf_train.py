from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.config import load_glmc_config
from ece888_sagmc.hf_gpt2 import train_hf_gpt2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HuggingFace GPT-2 model like the GLMC repo.")
    parser.add_argument("--config", default="configs/glmc_tiny_shakespeare.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_glmc_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    out_dir = args.output_dir or (
        f"gpt2_tinyshakespeare_seed{args.seed}_nembd{cfg['model']['n_embd']}"
    )
    result = train_hf_gpt2(
        data_cfg=cfg["data"],
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
        seed=args.seed,
        output_dir=out_dir,
    )
    print(f"saved model dir: {result}")


if __name__ == "__main__":
    main()
