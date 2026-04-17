from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.config import load_glmc_config
from ece888_sagmc.hf_gpt2 import (
    compute_token_frequencies,
    create_contiguous_splits,
    download_tiny_shakespeare,
    ensure_gpt2_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official-GLMC-style Tiny Shakespeare data.")
    parser.add_argument("--config", default="configs/glmc_tiny_shakespeare.yaml")
    parser.add_argument("--tokenizer-source", default="gpt2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_glmc_config(args.config)
    data_cfg = cfg["data"]
    data_file = download_tiny_shakespeare(data_cfg["data_file"], data_cfg["source_url"])
    tokenizer_dir = ensure_gpt2_tokenizer(data_cfg["tokenizer_dir"], args.tokenizer_source)
    splits_dir = create_contiguous_splits(
        data_file=data_file,
        out_dir=data_cfg["splits_dir"],
        val_frac=float(data_cfg["val_frac"]),
        test_frac=float(data_cfg["test_frac"]),
    )
    freqs = compute_token_frequencies(splits_dir, tokenizer_dir)
    print(f"data_file: {data_file}")
    print(f"tokenizer_dir: {tokenizer_dir}")
    print(f"splits_dir: {splits_dir}")
    print(f"token_freqs: {freqs}")


if __name__ == "__main__":
    main()
