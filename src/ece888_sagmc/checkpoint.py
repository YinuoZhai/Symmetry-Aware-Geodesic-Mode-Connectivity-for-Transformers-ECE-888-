from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .model import GPT, GPTConfig


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        clean_key = key.removeprefix("module.")
        out[clean_key] = value
    return out


def extract_state_dict(obj: Any) -> OrderedDict[str, torch.Tensor]:
    if isinstance(obj, dict) and "model_state" in obj:
        return _strip_module_prefix(obj["model_state"])
    if isinstance(obj, dict) and "state_dict" in obj:
        return _strip_module_prefix(obj["state_dict"])
    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        return _strip_module_prefix(obj)
    raise ValueError(
        "Unsupported checkpoint format. Expected {'model_state': ...}, "
        "{'state_dict': ...}, or a raw state_dict."
    )


def extract_model_config(obj: Any, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(obj, dict) and "model_config" in obj:
        return dict(obj["model_config"])
    if fallback is not None:
        return dict(fallback)
    raise ValueError("Checkpoint has no model_config and no fallback config was supplied.")


def load_checkpoint_object(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> OrderedDict[str, torch.Tensor]:
    return extract_state_dict(load_checkpoint_object(path, map_location=map_location))


def load_model_and_state(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    fallback_model_config: dict[str, Any] | None = None,
) -> tuple[GPT, OrderedDict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    obj = load_checkpoint_object(path, map_location=map_location)
    state = extract_state_dict(obj)
    model_config = extract_model_config(obj, fallback=fallback_model_config)
    model = GPT(GPTConfig.from_dict(model_config))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint {path} is incompatible. Missing={missing}, unexpected={unexpected}"
        )
    return model, state, model_config, obj if isinstance(obj, dict) else {}


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    metadata: dict[str, Any],
) -> None:
    ckpt: dict[str, Any] = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model.config.to_dict(),
        **metadata,
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    torch.save(ckpt, path)


def assert_compatible_states(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> None:
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    if keys_a != keys_b:
        raise ValueError(
            f"State dict keys differ. Only in A={sorted(keys_a - keys_b)[:5]}, "
            f"only in B={sorted(keys_b - keys_a)[:5]}"
        )
    shape_errors = [
        (k, tuple(state_a[k].shape), tuple(state_b[k].shape))
        for k in state_a
        if state_a[k].shape != state_b[k].shape
    ]
    if shape_errors:
        raise ValueError(f"State dict tensor shapes differ: {shape_errors[:5]}")
