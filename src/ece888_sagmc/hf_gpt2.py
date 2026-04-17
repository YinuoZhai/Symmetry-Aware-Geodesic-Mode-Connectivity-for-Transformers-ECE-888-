from __future__ import annotations

import csv
import json
import math
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from safetensors.torch import load_file as load_safetensors_file
from torch.func import functional_call
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .metrics import compute_barrier, jsd_from_logits


def download_tiny_shakespeare(data_file: str | Path, source_url: str) -> Path:
    path = Path(data_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    try:
        with urllib.request.urlopen(source_url, timeout=30) as response:
            text = response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download Tiny Shakespeare to {path}. "
            "Place the text file there manually or rerun with network access."
        ) from exc
    path.write_text(text, encoding="utf-8")
    return path


def create_contiguous_splits(
    data_file: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> Path:
    with Path(data_file).open("r", encoding="utf-8") as f:
        text = f.read()
    n_chars = len(text)
    if n_chars == 0:
        raise ValueError("Input text file is empty.")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.")

    test_n = int(round(n_chars * test_frac))
    val_n = int(round(n_chars * val_frac))
    holdout_n = test_n + val_n
    train_text = text[: n_chars - holdout_n]
    val_text = text[n_chars - holdout_n : n_chars - test_n] if val_n > 0 else ""
    test_text = text[n_chars - test_n :] if test_n > 0 else ""

    ds = DatasetDict(
        {
            "train": Dataset.from_dict({"text": [train_text]}),
            "validation": Dataset.from_dict({"text": [val_text]}),
            "test": Dataset.from_dict({"text": [test_text]}),
        }
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    metadata = {
        "method": "contiguous",
        "data_file": str(Path(data_file).resolve()),
        "val_frac": val_frac,
        "test_frac": test_frac,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "test_chars": len(test_text),
    }
    with (out / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out


def ensure_gpt2_tokenizer(tokenizer_dir: str | Path, source: str = "gpt2") -> Path:
    out = Path(tokenizer_dir)
    if (out / "tokenizer_config.json").exists() or (out / "vocab.json").exists():
        return out
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.pad_token = tokenizer.eos_token
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out))
    return out


def compute_token_frequencies(
    splits_dir: str | Path,
    tokenizer_dir: str | Path,
    out_name: str = "token_freqs.pt",
) -> Path:
    ds = load_from_disk(str(splits_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [str(t) for t in ds["train"]["text"]]
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=False,
    )
    all_ids = list(chain.from_iterable(enc["input_ids"]))
    if not all_ids:
        raise ValueError("No train tokens found while computing token frequencies.")
    freqs = torch.bincount(torch.tensor(all_ids, dtype=torch.long), minlength=tokenizer.vocab_size)
    out = Path(splits_dir) / out_name
    torch.save(freqs, out)
    return out


def build_gpt2_config(tokenizer, model_cfg: dict[str, Any]) -> GPT2Config:
    return GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=int(model_cfg["block_size"]),
        n_ctx=int(model_cfg["block_size"]),
        n_embd=int(model_cfg["n_embd"]),
        n_layer=int(model_cfg["n_layer"]),
        n_head=int(model_cfg["n_head"]),
        n_inner=int(model_cfg["n_inner"]),
        tie_word_embeddings=bool(model_cfg.get("tie_word_embeddings", False)),
        activation_function="gelu_new",
        resid_pdrop=float(model_cfg.get("resid_pdrop", 0.1)),
        embd_pdrop=float(model_cfg.get("embd_pdrop", 0.1)),
        attn_pdrop=float(model_cfg.get("attn_pdrop", 0.1)),
    )


def tokenize_and_chunk_splits(
    splits_dir: str | Path,
    tokenizer,
    block_size: int,
) -> tuple[Any, Any, Any]:
    ds = load_from_disk(str(splits_dir))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=True)

    def group_texts(examples):
        ids = list(chain(*examples["input_ids"]))
        masks = list(chain(*examples["attention_mask"]))
        chunk_len = (len(ids) // block_size) * block_size
        ids = ids[:chunk_len]
        masks = masks[:chunk_len]
        return {
            "input_ids": [ids[i : i + block_size] for i in range(0, chunk_len, block_size)],
            "attention_mask": [masks[i : i + block_size] for i in range(0, chunk_len, block_size)],
        }

    def prep(split_ds):
        tokenized = split_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        chunked = tokenized.map(group_texts, batched=True)
        chunked.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return chunked

    return prep(ds["train"]), prep(ds["validation"]), prep(ds["test"])


@dataclass
class HFGPT2Data:
    train: Any
    validation: Any
    test: Any
    tokenizer: Any
    block_size: int

    def split(self, name: str):
        if name == "train":
            return self.train
        if name in {"val", "validation"}:
            return self.validation
        if name == "test":
            return self.test
        raise ValueError(f"Unknown split {name!r}.")

    def get_batch(
        self,
        split: str,
        batch_size: int,
        sequence_length: int,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        ds = self.split(split)
        if len(ds) == 0:
            raise ValueError(f"Split {split!r} has no token chunks.")
        indices = torch.randint(len(ds), (batch_size,)).tolist()
        input_rows = []
        mask_rows = []
        for idx in indices:
            row = ds[idx]
            input_ids = row["input_ids"]
            attention_mask = row["attention_mask"]
            if sequence_length < input_ids.numel():
                start = torch.randint(input_ids.numel() - sequence_length + 1, ()).item()
                input_ids = input_ids[start : start + sequence_length]
                attention_mask = attention_mask[start : start + sequence_length]
            input_rows.append(input_ids)
            mask_rows.append(attention_mask)
        return {
            "input_ids": torch.stack(input_rows).to(device),
            "attention_mask": torch.stack(mask_rows).to(device),
        }


def build_hf_data(data_cfg: dict[str, Any], block_size: int) -> HFGPT2Data:
    tokenizer = AutoTokenizer.from_pretrained(str(data_cfg["tokenizer_dir"]))
    tokenizer.pad_token = tokenizer.eos_token
    train_ds, val_ds, test_ds = tokenize_and_chunk_splits(
        splits_dir=data_cfg["splits_dir"],
        tokenizer=tokenizer,
        block_size=block_size,
    )
    return HFGPT2Data(
        train=train_ds,
        validation=val_ds,
        test=test_ds,
        tokenizer=tokenizer,
        block_size=block_size,
    )


def load_hf_state_dict(model_dir_or_file: str | Path) -> OrderedDict[str, torch.Tensor]:
    path = Path(model_dir_or_file)
    if path.is_file():
        if path.suffix == ".safetensors":
            state = load_safetensors_file(str(path))
        else:
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
    else:
        sft = path / "model.safetensors"
        ptb = path / "pytorch_model.bin"
        if sft.exists():
            state = load_safetensors_file(str(sft))
        elif ptb.exists():
            state = torch.load(ptb, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")
    return OrderedDict((k.removeprefix("module."), v.detach().cpu()) for k, v in state.items())


def complete_state_dict(
    model: GPT2LMHeadModel,
    loaded: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    base = model.state_dict()
    completed: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in base.items():
        completed[key] = loaded.get(key, value).detach().cpu()
    extras = sorted(set(loaded.keys()) - set(base.keys()))
    if extras:
        raise ValueError(f"Loaded state has unexpected keys: {extras[:5]}")
    return completed


def load_hf_model_and_state(model_dir: str | Path) -> tuple[GPT2LMHeadModel, OrderedDict[str, torch.Tensor]]:
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.config.use_cache = False
    try:
        model.config.attn_implementation = "eager"
        model._attn_implementation = "eager"
    except Exception:
        pass
    state = complete_state_dict(model, load_hf_state_dict(model_dir))
    return model, state


def assert_hf_compatible(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    if set(a.keys()) != set(b.keys()):
        raise ValueError("State dict keys differ; cannot interpolate.")
    bad = [(k, tuple(a[k].shape), tuple(b[k].shape)) for k in a if a[k].shape != b[k].shape]
    if bad:
        raise ValueError(f"State dict shapes differ: {bad[:5]}")


def interpolate_hf_state(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    coeff: float,
) -> OrderedDict[str, torch.Tensor]:
    # Match official GLMC convention: coeff=1 is theta_A, coeff=0 is theta_B.
    assert_hf_compatible(state_a, state_b)
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in state_a:
        a = state_a[key]
        b = state_b[key]
        if torch.is_floating_point(a):
            out[key] = coeff * a + (1.0 - coeff) * b
        else:
            out[key] = a if coeff >= 0.5 else b
    return out


@torch.no_grad()
def evaluate_hf_model_loss(
    model: GPT2LMHeadModel,
    data: HFGPT2Data,
    split: str,
    batch_size: int,
    device: torch.device | str,
    max_batches: int | None = None,
) -> float:
    model = model.to(device)
    model.eval()
    loader = DataLoader(data.split(split), batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_examples = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
        )
        batch_size = int(input_ids.shape[0])
        total_loss += float(outputs.loss.detach().cpu()) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError(f"No evaluation batches for split {split!r}.")
    return float(total_loss / total_examples)


def evaluate_hf_state_loss(
    model: GPT2LMHeadModel,
    state: dict[str, torch.Tensor],
    data: HFGPT2Data,
    split: str,
    batch_size: int,
    device: torch.device | str,
    max_batches: int | None = None,
) -> float:
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith(".attn.masked_bias")]
    if unexpected:
        raise ValueError(f"Unexpected keys while loading interpolated state: {unexpected[:5]}")
    if missing:
        pass
    return evaluate_hf_model_loss(model, data, split, batch_size, device, max_batches)


def evaluate_hf_linear_path(
    model: GPT2LMHeadModel,
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    data: HFGPT2Data,
    split: str,
    batch_size: int,
    coeff_start: float,
    coeff_end: float,
    coeff_step: float,
    device: torch.device | str,
    max_batches: int | None = None,
) -> tuple[float, list[dict[str, float]]]:
    coeffs = []
    coeff = coeff_start
    while coeff <= coeff_end + 1e-9:
        coeffs.append(float(round(coeff, 10)))
        coeff += coeff_step

    losses = []
    for coeff in coeffs:
        state = interpolate_hf_state(state_a, state_b, coeff)
        loss = evaluate_hf_state_loss(model, state, data, split, batch_size, device, max_batches)
        print(f"[hf-linear] coeff={coeff:.6f} loss={loss:.6f}")
        losses.append(loss)

    # compute_barrier expects the path ordered from endpoint A to endpoint B.
    # Official GLMC coeffs increase from B to A, so sort by alpha=1-coeff.
    triples = sorted((1.0 - c, loss, c) for c, loss in zip(coeffs, losses))
    barrier, points = compute_barrier(
        [alpha for alpha, _loss, _coeff in triples],
        [loss for _alpha, loss, _coeff in triples],
    )
    return barrier, [p.__dict__ | {"coeff": triples[p.index][2]} for p in points]


def write_hf_metrics(
    out_dir: str | Path,
    barrier: float,
    rows: list[dict[str, float]],
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"barrier": barrier, "points": rows, **(extra or {})}
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    fieldnames = ["index", "alpha", "coeff", "loss", "baseline_loss", "barrier"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _states_to_waypoints(
    states: list[dict[str, torch.Tensor]],
    device: torch.device | str,
) -> list[OrderedDict[str, torch.Tensor]]:
    waypoints: list[OrderedDict[str, torch.Tensor]] = []
    last_idx = len(states) - 1
    for idx, state in enumerate(states):
        waypoint: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, tensor in state.items():
            value = tensor.detach().to(device)
            if 0 < idx < last_idx and torch.is_floating_point(value):
                waypoint[key] = torch.nn.Parameter(value.clone())
            else:
                waypoint[key] = value.clone().requires_grad_(False)
        waypoints.append(waypoint)
    return waypoints


def _waypoint_parameters(waypoints: list[OrderedDict[str, torch.Tensor]]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for waypoint in waypoints[1:-1]:
        params.extend(v for v in waypoint.values() if isinstance(v, torch.nn.Parameter))
    return params


def _functional_hf_logits(
    model: GPT2LMHeadModel,
    state: OrderedDict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = functional_call(
        model,
        state,
        args=(),
        kwargs={
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        },
        tie_weights=False,
        strict=False,
    )
    return outputs.logits


def initialize_hf_waypoints(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    num_waypoints: int,
) -> list[OrderedDict[str, torch.Tensor]]:
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least 2.")
    return [
        interpolate_hf_state(state_a, state_b, coeff=1.0 - i / (num_waypoints - 1))
        for i in range(num_waypoints)
    ]


def optimize_hf_geodesic(
    model: GPT2LMHeadModel,
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    data: HFGPT2Data,
    cfg: dict[str, Any],
    out_dir: str | Path,
    device: torch.device | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    init_states = initialize_hf_waypoints(state_a, state_b, int(cfg["num_waypoints"]))
    waypoints = _states_to_waypoints(init_states, device)
    params = _waypoint_parameters(waypoints)
    optimizer = torch.optim.SGD(
        params,
        lr=float(cfg["learning_rate"]),
        momentum=float(cfg["momentum"]),
    )
    sequence_length = int(cfg.get("sequence_length") or data.block_size)
    log_path = out / "geodesic_metrics.jsonl"

    for step in range(1, int(cfg["iterations"]) + 1):
        batch = data.get_batch(
            "train",
            batch_size=int(cfg["batch_size"]),
            sequence_length=sequence_length,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        total_energy = 0.0

        # Pairwise backward keeps memory practical with a 50k GPT-2 vocabulary.
        for idx in range(len(waypoints) - 1):
            logits_i = _functional_hf_logits(model, waypoints[idx], batch)
            logits_j = _functional_hf_logits(model, waypoints[idx + 1], batch)
            energy = jsd_from_logits(logits_i, logits_j)
            total_energy += float(energy.detach().cpu())
            energy.backward()
            del logits_i, logits_j, energy

        if float(cfg["grad_clip"]) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
        optimizer.step()

        if step == 1 or step % int(cfg["log_interval"]) == 0 or step == int(cfg["iterations"]):
            row = {"step": step, "train_jsd_energy": total_energy}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[hf-geodesic] step={step} train_jsd_energy={total_energy:.6f}")

    path_file = out / "geodesic_path.pt"
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "waypoints": [
                OrderedDict((k, v.detach().cpu()) for k, v in waypoint.items())
                for waypoint in waypoints
            ],
            "geodesic_config": cfg,
            **(metadata or {}),
        },
        path_file,
    )
    return path_file


def load_hf_geodesic_path(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_hf_waypoint_path(
    model: GPT2LMHeadModel,
    states: list[dict[str, torch.Tensor]],
    data: HFGPT2Data,
    split: str,
    batch_size: int,
    device: torch.device | str,
    max_batches: int | None = None,
) -> tuple[float, list[dict[str, float]]]:
    losses = []
    for idx, state in enumerate(states):
        loss = evaluate_hf_state_loss(model, state, data, split, batch_size, device, max_batches)
        print(f"[hf-path] waypoint={idx} loss={loss:.6f}")
        losses.append(loss)
    alphas = [i / (len(states) - 1) for i in range(len(states))]
    barrier, points = compute_barrier(alphas, losses)
    return barrier, [p.__dict__ | {"coeff": 1.0 - p.alpha} for p in points]


def train_hf_gpt2(
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    seed: int,
    output_dir: str | Path,
) -> Path:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(data_cfg["tokenizer_dir"]))
    tokenizer.pad_token = tokenizer.eos_token
    block_size = int(model_cfg["block_size"])
    train_ds, val_ds, test_ds = tokenize_and_chunk_splits(data_cfg["splits_dir"], tokenizer, block_size)

    model = GPT2LMHeadModel(build_gpt2_config(tokenizer, model_cfg))
    model.config.use_cache = False
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    use_fp16 = bool(train_cfg.get("fp16", False) and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=str(out),
        evaluation_strategy="steps",
        eval_steps=int(train_cfg["eval_steps"]),
        logging_strategy="steps",
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["eval_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=float(train_cfg["epochs"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=1,
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        fp16=use_fp16,
        report_to="none",
        run_name=f"gpt2-tinyshakespeare-seed{seed}-n_embd{model_cfg['n_embd']}",
    )
    callbacks = []
    if bool(train_cfg.get("early_stop", False)):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=int(train_cfg["early_stop_patience"]))
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is not None:
        (out / "BEST_CHECKPOINT.txt").write_text(best_ckpt + "\n", encoding="utf-8")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    with (out / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    return out
