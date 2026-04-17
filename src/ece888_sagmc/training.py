from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .config import ensure_dir
from .data import TinyShakespeare
from .interpolate import evaluate_model_loss
from .model import GPT, GPTConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(model: GPT, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not name.endswith("bias"):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(train_cfg["weight_decay"])},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(train_cfg["learning_rate"]),
        betas=(float(train_cfg["beta1"]), float(train_cfg["beta2"])),
    )


def train_model(
    dataset: TinyShakespeare,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    out_dir: str | Path,
    seed: int,
    device: torch.device | str,
) -> Path:
    seed_everything(seed)
    out = ensure_dir(out_dir)
    cfg = GPTConfig(vocab_size=dataset.vocab.size, **model_cfg)
    model = GPT(cfg).to(device)
    optimizer = create_optimizer(model, train_cfg)
    metrics_path = out / "train_metrics.jsonl"
    best_val = float("inf")
    best_path = out / "best.pt"

    pbar = tqdm(range(1, int(train_cfg["max_iters"]) + 1), desc=f"train seed={seed}")
    for step in pbar:
        model.train()
        x, y = dataset.get_batch(
            "train",
            batch_size=int(train_cfg["batch_size"]),
            block_size=cfg.block_size,
            device=device,
        )
        _logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Model did not return a training loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(train_cfg["grad_clip"]) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        optimizer.step()

        if step == 1 or step % int(train_cfg["eval_interval"]) == 0 or step == int(train_cfg["max_iters"]):
            train_loss = evaluate_model_loss(
                model,
                dataset,
                split="train",
                batch_size=int(train_cfg["batch_size"]),
                block_size=cfg.block_size,
                eval_iters=int(train_cfg["eval_iters"]),
                device=device,
            )
            val_loss = evaluate_model_loss(
                model,
                dataset,
                split="val",
                batch_size=int(train_cfg["batch_size"]),
                block_size=cfg.block_size,
                eval_iters=int(train_cfg["eval_iters"]),
                device=device,
            )
            row = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seed": seed,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            pbar.set_postfix(train=f"{train_loss:.3f}", val=f"{val_loss:.3f}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    metadata={
                        "seed": seed,
                        "step": step,
                        "best_val_loss": best_val,
                        "vocab": dataset.vocab.to_dict(),
                    },
                )

    save_checkpoint(
        out / "last.pt",
        model,
        optimizer,
        metadata={
            "seed": seed,
            "step": int(train_cfg["max_iters"]),
            "best_val_loss": best_val,
            "vocab": dataset.vocab.to_dict(),
        },
    )
    return best_path
