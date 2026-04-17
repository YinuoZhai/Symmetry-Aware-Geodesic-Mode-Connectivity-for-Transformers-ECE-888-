from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_dict(self) -> dict[str, object]:
        return {"itos": self.itos, "stoi": self.stoi}

    @classmethod
    def from_dict(cls, obj: dict[str, object]) -> "Vocab":
        itos = list(obj["itos"])
        stoi = {str(k): int(v) for k, v in dict(obj["stoi"]).items()}
        return cls(stoi=stoi, itos=itos)


class TinyShakespeare:
    def __init__(
        self,
        data_dir: str | Path,
        url: str,
        train_fraction: float = 0.9,
        download: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.url = url
        self.train_fraction = train_fraction
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.input_path = self.data_dir / "input.txt"
        self.meta_path = self.data_dir / "meta.json"

        if not self.input_path.exists():
            if not download:
                raise FileNotFoundError(
                    f"{self.input_path} does not exist and data.download=false. "
                    "Place Tiny Shakespeare input.txt there or enable download."
                )
            self._download()

        self.text = self.input_path.read_text(encoding="utf-8")
        self.vocab = self._build_or_load_vocab(self.text)
        ids = torch.tensor(self.vocab.encode(self.text), dtype=torch.long)
        split_idx = int(len(ids) * train_fraction)
        self.train_ids = ids[:split_idx]
        self.val_ids = ids[split_idx:]

    def _download(self) -> None:
        try:
            with urllib.request.urlopen(self.url, timeout=30) as response:
                text = response.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(
                "Could not download Tiny Shakespeare. If this machine is offline, "
                f"manually place input.txt at {self.input_path}."
            ) from exc
        self.input_path.write_text(text, encoding="utf-8")

    def _build_or_load_vocab(self, text: str) -> Vocab:
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                return Vocab.from_dict(json.load(f))

        chars = sorted(set(text))
        vocab = Vocab(stoi={ch: i for i, ch in enumerate(chars)}, itos=chars)
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(vocab.to_dict(), f, ensure_ascii=False, indent=2)
        return vocab

    def get_split(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.train_ids
        if split == "val":
            return self.val_ids
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'val'.")

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.get_split(split)
        if len(data) <= block_size + 1:
            raise ValueError(
                f"{split} split has {len(data)} tokens, not enough for block_size={block_size}."
            )
        starts = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
        return x.to(device), y.to(device)


def build_dataset(config: dict[str, object]) -> TinyShakespeare:
    return TinyShakespeare(
        data_dir=str(config["data_dir"]),
        url=str(config["url"]),
        train_fraction=float(config.get("train_fraction", 0.9)),
        download=bool(config.get("download", True)),
    )
