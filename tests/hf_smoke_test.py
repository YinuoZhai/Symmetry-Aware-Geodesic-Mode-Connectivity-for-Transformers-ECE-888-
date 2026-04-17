from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.hf_gpt2 import (
    HFGPT2Data,
    evaluate_hf_linear_path,
    evaluate_hf_waypoint_path,
    load_hf_geodesic_path,
    optimize_hf_geodesic,
)


def _toy_split(vocab_size: int, rows: int = 8, block_size: int = 8):
    g = torch.Generator().manual_seed(0)
    input_ids = [
        torch.randint(0, vocab_size, (block_size,), generator=g).tolist()
        for _ in range(rows)
    ]
    ds = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": [[1] * block_size for _ in range(rows)],
        }
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="sagmc_hf_smoke_"))
    try:
        cfg = GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_ctx=8,
            n_embd=16,
            n_layer=1,
            n_head=1,
            n_inner=32,
            tie_word_embeddings=False,
        )
        torch.manual_seed(0)
        model_a = GPT2LMHeadModel(cfg)
        torch.manual_seed(1)
        model_b = GPT2LMHeadModel(cfg)
        data = HFGPT2Data(
            train=_toy_split(cfg.vocab_size),
            validation=_toy_split(cfg.vocab_size),
            test=_toy_split(cfg.vocab_size),
            tokenizer=None,
            block_size=8,
        )
        barrier, rows = evaluate_hf_linear_path(
            model=GPT2LMHeadModel(cfg),
            state_a=model_a.state_dict(),
            state_b=model_b.state_dict(),
            data=data,
            split="test",
            batch_size=2,
            coeff_start=0.0,
            coeff_end=1.0,
            coeff_step=0.5,
            device="cpu",
            max_batches=1,
        )
        assert isinstance(barrier, float)
        assert len(rows) == 3

        path_file = optimize_hf_geodesic(
            model=GPT2LMHeadModel(cfg),
            state_a=model_a.state_dict(),
            state_b=model_b.state_dict(),
            data=data,
            cfg={
                "num_waypoints": 3,
                "iterations": 2,
                "batch_size": 2,
                "sequence_length": 8,
                "learning_rate": 0.01,
                "momentum": 0.0,
                "grad_clip": 1.0,
                "log_interval": 1,
            },
            out_dir=root / "gmc",
            device="cpu",
        )
        obj = load_hf_geodesic_path(path_file)
        barrier, rows = evaluate_hf_waypoint_path(
            model=GPT2LMHeadModel(cfg),
            states=obj["waypoints"],
            data=data,
            split="test",
            batch_size=2,
            device="cpu",
            max_batches=1,
        )
        assert len(rows) == 3
        print("hf smoke test passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
