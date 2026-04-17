from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.func import functional_call

from .checkpoint import assert_compatible_states
from .data import TinyShakespeare
from .interpolate import evaluate_state_loss, interpolate_state_dict
from .metrics import compute_barrier, jsd_from_logits
from .model import GPT


def initialize_waypoint_states(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    num_waypoints: int,
) -> list[OrderedDict[str, torch.Tensor]]:
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least 2.")
    assert_compatible_states(state_a, state_b)
    return [
        interpolate_state_dict(state_a, state_b, i / (num_waypoints - 1))
        for i in range(num_waypoints)
    ]


def _states_to_trainable_waypoints(
    states: list[dict[str, torch.Tensor]],
    device: torch.device | str,
) -> list[OrderedDict[str, torch.Tensor]]:
    waypoints: list[OrderedDict[str, torch.Tensor]] = []
    last_idx = len(states) - 1
    for idx, state in enumerate(states):
        params: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, tensor in state.items():
            value = tensor.detach().to(device)
            if 0 < idx < last_idx and torch.is_floating_point(value):
                params[key] = torch.nn.Parameter(value.clone())
            else:
                params[key] = value.clone().requires_grad_(False)
        waypoints.append(params)
    return waypoints


def _waypoint_parameters(
    waypoints: list[OrderedDict[str, torch.Tensor]],
) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for waypoint in waypoints[1:-1]:
        params.extend([p for p in waypoint.values() if isinstance(p, torch.nn.Parameter)])
    return params


def _functional_logits(
    model: GPT,
    params: OrderedDict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    logits, _loss = functional_call(model, params, (x,))
    return logits


def geodesic_energy(
    model: GPT,
    waypoints: list[OrderedDict[str, torch.Tensor]],
    x: torch.Tensor,
) -> torch.Tensor:
    logits = [_functional_logits(model, params, x) for params in waypoints]
    energy = torch.zeros((), device=x.device)
    for idx in range(len(logits) - 1):
        energy = energy + jsd_from_logits(logits[idx], logits[idx + 1])
    return energy


def _cpu_waypoint_states(
    waypoints: list[OrderedDict[str, torch.Tensor]],
) -> list[OrderedDict[str, torch.Tensor]]:
    return [
        OrderedDict((k, v.detach().cpu()) for k, v in waypoint.items())
        for waypoint in waypoints
    ]


def save_geodesic_path(
    path: str | Path,
    model: GPT,
    waypoints: list[OrderedDict[str, torch.Tensor]],
    metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "waypoints": _cpu_waypoint_states(waypoints),
            **metadata,
        },
        path,
    )


def load_geodesic_path(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def optimize_geodesic(
    model: GPT,
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    dataset: TinyShakespeare,
    cfg: dict[str, Any],
    out_dir: str | Path,
    device: torch.device | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    init_states = initialize_waypoint_states(
        {k: v.detach().cpu() for k, v in state_a.items()},
        {k: v.detach().cpu() for k, v in state_b.items()},
        int(cfg["num_waypoints"]),
    )
    waypoints = _states_to_trainable_waypoints(init_states, device=device)
    optimizer = torch.optim.SGD(
        _waypoint_parameters(waypoints),
        lr=float(cfg["learning_rate"]),
        momentum=float(cfg["momentum"]),
    )

    log_path = out / "geodesic_metrics.jsonl"
    for step in range(1, int(cfg["iterations"]) + 1):
        x, _y = dataset.get_batch(
            "train",
            batch_size=int(cfg["batch_size"]),
            block_size=model.config.block_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        energy = geodesic_energy(model, waypoints, x)
        energy.backward()
        if float(cfg["grad_clip"]) > 0:
            torch.nn.utils.clip_grad_norm_(_waypoint_parameters(waypoints), float(cfg["grad_clip"]))
        optimizer.step()

        if (
            step == 1
            or step % int(cfg["log_interval"]) == 0
            or step == int(cfg["iterations"])
        ):
            row = {"step": step, "train_jsd_energy": float(energy.detach().cpu())}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[geodesic] step={step} train_jsd_energy={row['train_jsd_energy']:.6f}")

    path_file = out / "geodesic_path.pt"
    save_geodesic_path(
        path_file,
        model,
        waypoints,
        metadata={"geodesic_config": cfg, **(metadata or {})},
    )
    return path_file


def evaluate_waypoint_states(
    model: GPT,
    states: list[dict[str, torch.Tensor]],
    dataset: TinyShakespeare,
    split: str,
    batch_size: int,
    eval_iters: int,
    device: torch.device | str,
) -> tuple[float, list[dict[str, float]]]:
    losses: list[float] = []
    alphas = [i / (len(states) - 1) for i in range(len(states))]
    for state in states:
        losses.append(
            evaluate_state_loss(
                model=model,
                state=state,
                dataset=dataset,
                split=split,
                batch_size=batch_size,
                eval_iters=eval_iters,
                device=device,
            )
        )
    barrier, points = compute_barrier(alphas, losses)
    return barrier, [point.__dict__ for point in points]


def write_geodesic_eval(
    out_dir: str | Path,
    barrier: float,
    rows: list[dict[str, float]],
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"barrier": barrier, "points": rows, **(extra or {})}, f, indent=2)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "alpha", "loss", "baseline_loss", "barrier"],
        )
        writer.writeheader()
        writer.writerows(rows)
