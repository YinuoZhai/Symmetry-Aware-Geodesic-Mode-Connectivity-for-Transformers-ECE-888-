from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .checkpoint import assert_compatible_states
from .data import TinyShakespeare
from .metrics import PathPoint, compute_barrier
from .model import GPT


def interpolate_state_dict(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    alpha: float,
) -> OrderedDict[str, torch.Tensor]:
    assert_compatible_states(state_a, state_b)
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in state_a.keys():
        a = state_a[key]
        b = state_b[key]
        if torch.is_floating_point(a):
            out[key] = (1.0 - alpha) * a + alpha * b
        else:
            out[key] = a if alpha < 0.5 else b
    return out


@torch.no_grad()
def evaluate_model_loss(
    model: GPT,
    dataset: TinyShakespeare,
    split: str,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device | str,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch(split, batch_size, block_size, device)
        _logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Model did not return a loss.")
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(sum(losses) / len(losses))


@torch.no_grad()
def evaluate_state_loss(
    model: GPT,
    state: dict[str, torch.Tensor],
    dataset: TinyShakespeare,
    split: str,
    batch_size: int,
    eval_iters: int,
    device: torch.device | str,
) -> float:
    model.load_state_dict(state, strict=True)
    model.to(device)
    return evaluate_model_loss(
        model=model,
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        block_size=model.config.block_size,
        eval_iters=eval_iters,
        device=device,
    )


def evaluate_linear_path(
    model: GPT,
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    dataset: TinyShakespeare,
    split: str,
    batch_size: int,
    eval_iters: int,
    num_points: int,
    device: torch.device | str,
) -> tuple[float, list[PathPoint]]:
    alphas = [i / (num_points - 1) for i in range(num_points)]
    losses: list[float] = []
    cpu_a = {k: v.detach().cpu() for k, v in state_a.items()}
    cpu_b = {k: v.detach().cpu() for k, v in state_b.items()}
    for alpha in alphas:
        state = interpolate_state_dict(cpu_a, cpu_b, alpha)
        loss = evaluate_state_loss(
            model=model,
            state=state,
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            eval_iters=eval_iters,
            device=device,
        )
        losses.append(loss)
    return compute_barrier(alphas, losses)


def write_path_metrics(
    out_dir: str | Path,
    barrier: float,
    points: list[PathPoint],
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [point.__dict__ for point in points]
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"barrier": barrier, "points": rows, **(extra or {})}, f, indent=2)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "alpha", "loss", "baseline_loss", "barrier"],
        )
        writer.writeheader()
        writer.writerows(rows)
