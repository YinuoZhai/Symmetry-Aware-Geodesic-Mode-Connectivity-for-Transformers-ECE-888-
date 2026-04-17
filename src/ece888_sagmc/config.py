from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "data_dir": "data/tiny_shakespeare",
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "train_fraction": 0.9,
        "download": True,
    },
    "model": {
        "block_size": 128,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "dropout": 0.1,
        "bias": True,
    },
    "train": {
        "batch_size": 64,
        "max_iters": 5000,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "eval_interval": 250,
        "eval_iters": 50,
    },
    "eval": {
        "batch_size": 64,
        "eval_iters": 100,
        "num_points": 25,
        "split": "val",
    },
    "geodesic": {
        "num_waypoints": 25,
        "iterations": 3000,
        "batch_size": 256,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "grad_clip": 10.0,
        "log_interval": 50,
        "eval_interval": 250,
    },
    "experiment": {
        "seed_a": 0,
        "seed_b": 1,
        "out_dir": "runs/tiny_shakespeare",
    },
}


DEFAULT_GLMC_CONFIG: dict[str, Any] = {
    "data": {
        "data_file": "data/tiny_shakespeare/tinyshakespeare.txt",
        "source_url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "splits_dir": "splits_tiny_contig",
        "tokenizer_dir": "gpt2_tokenizer",
        "val_frac": 0.05,
        "test_frac": 0.05,
        "method": "contiguous",
        "split_seed": 123,
    },
    "model": {
        "block_size": 256,
        "n_layer": 6,
        "n_embd": 256,
        "n_inner": 1024,
        "n_head": 4,
        "tie_word_embeddings": False,
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
    },
    "train": {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 3e-4,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "eval_steps": 50,
        "logging_steps": 25,
        "save_total_limit": 2,
        "early_stop": True,
        "early_stop_patience": 5,
        "fp16": True,
    },
    "eval": {
        "batch_size": 64,
        "split": "test",
        "coeff_start": 0.0,
        "coeff_end": 1.0,
        "coeff_step": 0.1,
        "max_batches": None,
    },
    "geodesic": {
        "num_waypoints": 25,
        "iterations": 3000,
        "batch_size": 8,
        "sequence_length": 64,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "grad_clip": 10.0,
        "log_interval": 50,
    },
    "experiment": {
        "seed_a": 0,
        "seed_b": 1,
        "out_dir": "runs/glmc_tiny_shakespeare",
    },
}


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_config)


def load_glmc_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_GLMC_CONFIG)
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_GLMC_CONFIG, user_config)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
