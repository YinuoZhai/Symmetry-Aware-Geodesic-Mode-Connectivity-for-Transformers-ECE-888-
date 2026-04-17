from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms) + self.bias


def load_merge_meta(merge_dir: str | Path) -> dict[str, Any]:
    path = Path(merge_dir) / "merge_meta.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_merge_state(merge_dir: str | Path) -> dict[str, torch.Tensor]:
    path = Path(merge_dir) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"No model.safetensors found in {merge_dir}")
    return load_safetensors_file(str(path))


def _project_orthogonal(a: torch.Tensor) -> torch.Tensor:
    device = a.device
    dtype = a.dtype
    u, _, vh = torch.linalg.svd(a.detach().float().cpu(), full_matrices=False)
    return (u @ vh).to(device=device, dtype=dtype)


def _project_permutation_greedy(a: torch.Tensor) -> torch.Tensor:
    # The learned GLMC matrices are already near permutations. This fallback avoids
    # a hard scipy dependency while preserving one-to-one row/column assignments.
    n_rows, n_cols = a.shape
    flat = torch.argsort(a.detach().flatten().cpu(), descending=True)
    out = torch.zeros(n_rows, n_cols, dtype=a.dtype, device=a.device)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for item in flat.tolist():
        row = item // n_cols
        col = item % n_cols
        if row in used_rows or col in used_cols:
            continue
        out[row, col] = 1
        used_rows.add(row)
        used_cols.add(col)
        if len(used_rows) == min(n_rows, n_cols):
            break
    if len(used_rows) != min(n_rows, n_cols):
        raise RuntimeError("Could not project matrix to a permutation.")
    return out


def _project_permutation(a: torch.Tensor) -> torch.Tensor:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return _project_permutation_greedy(a)

    row_ind, col_ind = linear_sum_assignment(-a.detach().cpu().numpy())
    out = torch.zeros_like(a)
    out[row_ind.tolist(), col_ind.tolist()] = 1
    return out


def _interpolate(a: torch.Tensor, b: torch.Tensor, coeff: float) -> torch.Tensor:
    return coeff * a + (1.0 - coeff) * b


def _head_view(x: torch.Tensor, num_heads: int, width: int) -> torch.Tensor:
    return x.reshape(num_heads, -1, width)


class GLMCMaterializer:
    """Materializes fixed-coeff states from the official Tiny Shakespeare GPTMerger."""

    def __init__(
        self,
        merge_dir: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.merge_dir = Path(merge_dir)
        self.meta = load_merge_meta(self.merge_dir)
        raw_state = load_merge_state(self.merge_dir)
        self.state = {k: v.detach().to(device) for k, v in raw_state.items()}
        self.permutations_only = bool(self.meta.get("permutations_only", False))

        config = GPT2Config.from_pretrained(str(self.merge_dir))
        self.n_layer = int(config.n_layer)
        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.internal_attn_dim = self.n_head * (self.n_embd + 1)
        self._p_res_cache: torch.Tensor | None = None
        self._p_mlp_cache: dict[int, torch.Tensor] = {}
        self._p_head_cache: dict[int, torch.Tensor] = {}

    def _key(self, name: str) -> torch.Tensor:
        try:
            return self.state[name]
        except KeyError as exc:
            raise KeyError(f"Missing GLMC tensor {name!r} in {self.merge_dir}") from exc

    def residual_projection(self) -> torch.Tensor:
        if self._p_res_cache is not None:
            return self._p_res_cache
        proj = self._key("proj.residual")
        if self.permutations_only:
            self._p_res_cache = _project_permutation(proj)
        else:
            self._p_res_cache = _project_orthogonal(proj)
        return self._p_res_cache

    def mlp_projection(self, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self._p_mlp_cache:
            self._p_mlp_cache[layer_idx] = _project_permutation(self._key(f"proj.mlp_{layer_idx}"))
        return self._p_mlp_cache[layer_idx]

    def head_projection(self, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self._p_head_cache:
            self._p_head_cache[layer_idx] = _project_permutation(
                self._key(f"proj.attention_heads_{layer_idx}")
            )
        return self._p_head_cache[layer_idx]

    def _aligned_c_attn(
        self,
        layer_idx: int,
        p_res: torch.Tensor,
        p_heads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"model.transformer.h.{layer_idx}.attn.c_attn"
        weight = self._key(f"{prefix}.conv1d_1_weight")
        bias = self._key(f"{prefix}.conv1d_1_bias")
        combined = torch.cat((weight.t(), bias.reshape(-1, 1)), dim=1)
        q, k, v = combined.chunk(3, dim=0)

        p_in = p_res.t()
        q = torch.cat((q[:, :-1] @ p_in.t(), q[:, -1:]), dim=-1)

        width = self.n_embd + 1
        q = _head_view(q, self.n_head, width)
        k = _head_view(k, self.n_head, width)
        v = _head_view(v, self.n_head, width)

        q = torch.cat(
            (
                torch.bmm(q.transpose(1, 2)[:, :, :-1], p_in.t().expand(self.n_head, -1, -1)),
                q.transpose(1, 2)[:, :, -1:],
            ),
            dim=-1,
        ).transpose(1, 2)

        def permute_heads(x: torch.Tensor) -> torch.Tensor:
            return (p_heads @ x.reshape(x.shape[0], -1)).reshape_as(x)

        q = permute_heads(q)
        k = permute_heads(k)
        v = permute_heads(v)

        combined = torch.cat(
            (q.reshape(-1, width), k.reshape(-1, width), v.reshape(-1, width)),
            dim=0,
        )
        return combined[:, :-1].t().contiguous(), combined[:, -1].contiguous()

    def _aligned_c_proj(
        self,
        layer_idx: int,
        p_res: torch.Tensor,
        p_heads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"model.transformer.h.{layer_idx}.attn.c_proj"
        weight = self._key(f"{prefix}.conv1d_1_weight") @ p_res
        bias = self._key(f"{prefix}.conv1d_1_bias") @ p_res

        out = weight.t().reshape(self.n_embd, self.n_head, -1).permute(1, 2, 0)
        out = torch.cat(
            (
                out.transpose(1, 2)[:, :, :-1] @ p_res.expand(self.n_head, -1, -1),
                out.transpose(1, 2)[:, :, -1:],
            ),
            dim=-1,
        ).transpose(1, 2)
        out = (p_heads @ out.reshape(out.shape[0], -1)).reshape_as(out)
        out = out.permute(2, 0, 1).reshape(self.n_embd, -1).t().contiguous()
        return out, bias.contiguous()

    def materialize_state(self, coeff: float) -> OrderedDict[str, torch.Tensor]:
        """Return a GLMC-materialized GPT-2 state. coeff=1 is model0, coeff=0 is aligned model1."""
        p_res = self.residual_projection()
        out: OrderedDict[str, torch.Tensor] = OrderedDict()

        out["transformer.wte.weight"] = _interpolate(
            self._key("model.transformer.wte.embedding_0.weight"),
            self._key("model.transformer.wte.embedding_1.weight") @ p_res,
            coeff,
        ).contiguous()
        out["transformer.wpe.weight"] = _interpolate(
            self._key("model.transformer.wpe.embedding_0.weight"),
            self._key("model.transformer.wpe.embedding_1.weight") @ p_res,
            coeff,
        ).contiguous()

        for layer_idx in range(self.n_layer):
            p_heads = self.head_projection(layer_idx)
            p_mlp = self.mlp_projection(layer_idx)
            prefix = f"model.transformer.h.{layer_idx}"
            target = f"transformer.h.{layer_idx}"

            for ln_name in ("ln_1", "ln_2"):
                ln_prefix = f"{prefix}.{ln_name}"
                out[f"{target}.{ln_name}.weight"] = self._key(f"{ln_prefix}.norm.weight").contiguous()
                out[f"{target}.{ln_name}.bias"] = _interpolate(
                    self._key(f"{ln_prefix}.bias_0"),
                    p_res.t() @ self._key(f"{ln_prefix}.bias_1"),
                    coeff,
                ).contiguous()

            c_attn_w1, c_attn_b1 = self._aligned_c_attn(layer_idx, p_res, p_heads)
            c_attn_prefix = f"{prefix}.attn.c_attn"
            out[f"{target}.attn.c_attn.weight"] = _interpolate(
                self._key(f"{c_attn_prefix}.conv1d_0_weight"),
                c_attn_w1,
                coeff,
            ).contiguous()
            out[f"{target}.attn.c_attn.bias"] = _interpolate(
                self._key(f"{c_attn_prefix}.conv1d_0_bias"),
                c_attn_b1,
                coeff,
            ).contiguous()

            c_proj_w1, c_proj_b1 = self._aligned_c_proj(layer_idx, p_res, p_heads)
            c_proj_prefix = f"{prefix}.attn.c_proj"
            out[f"{target}.attn.c_proj.weight"] = _interpolate(
                self._key(f"{c_proj_prefix}.conv1d_0_weight"),
                c_proj_w1,
                coeff,
            ).contiguous()
            out[f"{target}.attn.c_proj.bias"] = _interpolate(
                self._key(f"{c_proj_prefix}.conv1d_0_bias"),
                c_proj_b1,
                coeff,
            ).contiguous()

            c_fc_prefix = f"{prefix}.mlp.c_fc"
            out[f"{target}.mlp.c_fc.weight"] = _interpolate(
                self._key(f"{c_fc_prefix}.conv1d_0_weight"),
                p_res.t() @ self._key(f"{c_fc_prefix}.conv1d_1_weight") @ p_mlp,
                coeff,
            ).contiguous()
            out[f"{target}.mlp.c_fc.bias"] = _interpolate(
                self._key(f"{c_fc_prefix}.conv1d_0_bias"),
                self._key(f"{c_fc_prefix}.conv1d_1_bias") @ p_mlp,
                coeff,
            ).contiguous()

            c_mlp_proj_prefix = f"{prefix}.mlp.c_proj"
            out[f"{target}.mlp.c_proj.weight"] = _interpolate(
                self._key(f"{c_mlp_proj_prefix}.conv1d_0_weight"),
                p_mlp.t() @ self._key(f"{c_mlp_proj_prefix}.conv1d_1_weight") @ p_res,
                coeff,
            ).contiguous()
            out[f"{target}.mlp.c_proj.bias"] = _interpolate(
                self._key(f"{c_mlp_proj_prefix}.conv1d_0_bias"),
                self._key(f"{c_mlp_proj_prefix}.conv1d_1_bias") @ p_res,
                coeff,
            ).contiguous()

        ln_f_prefix = "model.transformer.ln_f"
        out["transformer.ln_f.weight"] = self._key(f"{ln_f_prefix}.norm.weight").contiguous()
        out["transformer.ln_f.bias"] = _interpolate(
            self._key(f"{ln_f_prefix}.bias_0"),
            p_res.t() @ self._key(f"{ln_f_prefix}.bias_1"),
            coeff,
        ).contiguous()

        lm_weight = _interpolate(
            self._key("model.lm_head.conv1d_0_weight"),
            p_res.t() @ self._key("model.lm_head.conv1d_1_weight"),
            coeff,
        )
        out["lm_head.weight"] = lm_weight.t().contiguous()
        return out


def build_glmc_materialized_model(merge_dir: str | Path) -> GPT2LMHeadModel:
    config = GPT2Config.from_pretrained(str(merge_dir))
    config.use_cache = False
    try:
        config.attn_implementation = "eager"
    except Exception:
        pass
    model = GPT2LMHeadModel(config)
    model.config.use_cache = False

    hidden = int(config.n_embd)
    n_head = int(config.n_head)
    internal = n_head * (hidden + 1)
    for block in model.transformer.h:
        block.ln_1 = RMSNorm(hidden, eps=float(config.layer_norm_epsilon))
        block.ln_2 = RMSNorm(hidden, eps=float(config.layer_norm_epsilon))
        block.attn.c_attn = Conv1D(3 * internal, hidden)
        block.attn.c_proj = Conv1D(hidden, internal)
        block.attn.split_size = internal
        block.attn.head_dim = hidden + 1
        block.attn.embed_dim = hidden
        block.attn.scaling = (hidden + 1) ** -0.5
    model.transformer.ln_f = RMSNorm(hidden, eps=float(config.layer_norm_epsilon))
    return model


def load_materialized_state(
    model: GPT2LMHeadModel,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> None:
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        raise ValueError(f"Could not load materialized GLMC state: missing={missing}, unexpected={unexpected}")
