from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.checkpoint import save_checkpoint
from ece888_sagmc.data import TinyShakespeare
from ece888_sagmc.geodesic import optimize_geodesic
from ece888_sagmc.interpolate import evaluate_linear_path
from ece888_sagmc.model import GPT, GPTConfig
from ece888_sagmc.training import seed_everything


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="sagmc_smoke_"))
    try:
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        text = ("First Citizen:\nBefore we proceed any further, hear me speak.\n" * 80)
        (data_dir / "input.txt").write_text(text, encoding="utf-8")
        dataset = TinyShakespeare(
            data_dir=data_dir,
            url="unused",
            train_fraction=0.8,
            download=False,
        )

        cfg = GPTConfig(
            vocab_size=dataset.vocab.size,
            block_size=16,
            n_layer=1,
            n_head=1,
            n_embd=16,
            dropout=0.0,
            bias=True,
        )
        seed_everything(0)
        model_a = GPT(cfg)
        seed_everything(1)
        model_b = GPT(cfg)
        ckpt_a = root / "a.pt"
        ckpt_b = root / "b.pt"
        save_checkpoint(ckpt_a, model_a, None, {"seed": 0, "vocab": dataset.vocab.to_dict()})
        save_checkpoint(ckpt_b, model_b, None, {"seed": 1, "vocab": dataset.vocab.to_dict()})

        barrier, points = evaluate_linear_path(
            model=GPT(cfg),
            state_a=model_a.state_dict(),
            state_b=model_b.state_dict(),
            dataset=dataset,
            split="val",
            batch_size=2,
            eval_iters=2,
            num_points=3,
            device="cpu",
        )
        assert len(points) == 3
        assert isinstance(barrier, float)

        out_dir = root / "gmc"
        path_file = optimize_geodesic(
            model=GPT(cfg),
            state_a=model_a.state_dict(),
            state_b=model_b.state_dict(),
            dataset=dataset,
            cfg={
                "num_waypoints": 3,
                "iterations": 2,
                "batch_size": 2,
                "learning_rate": 0.01,
                "momentum": 0.0,
                "grad_clip": 1.0,
                "log_interval": 1,
            },
            out_dir=out_dir,
            device="cpu",
        )
        assert path_file.exists()
        print("smoke test passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
