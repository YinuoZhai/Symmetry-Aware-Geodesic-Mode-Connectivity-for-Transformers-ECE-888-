from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def jsd_from_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean Jensen-Shannon divergence between categorical output distributions."""
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = (m + eps).log()
    jsd = 0.5 * (p * (log_p - log_m)).sum(dim=-1)
    jsd = jsd + 0.5 * (q * (log_q - log_m)).sum(dim=-1)
    return jsd.mean()


def path_energy_from_logits(logits: list[torch.Tensor]) -> torch.Tensor:
    if len(logits) < 2:
        raise ValueError("Need at least two waypoints to compute path energy.")
    return sum(jsd_from_logits(logits[i], logits[i + 1]) for i in range(len(logits) - 1))


@dataclass
class PathPoint:
    index: int
    alpha: float
    loss: float
    baseline_loss: float
    barrier: float


def compute_barrier(alphas: list[float], losses: list[float]) -> tuple[float, list[PathPoint]]:
    if len(alphas) != len(losses):
        raise ValueError("alphas and losses must have the same length.")
    if len(losses) < 2:
        raise ValueError("Need at least two losses to compute a barrier.")
    start = losses[0]
    end = losses[-1]
    points: list[PathPoint] = []
    max_barrier = float("-inf")
    for i, (alpha, loss) in enumerate(zip(alphas, losses)):
        baseline = (1.0 - alpha) * start + alpha * end
        barrier = loss - baseline
        max_barrier = max(max_barrier, barrier)
        points.append(
            PathPoint(
                index=i,
                alpha=float(alpha),
                loss=float(loss),
                baseline_loss=float(baseline),
                barrier=float(barrier),
            )
        )
    return float(max_barrier), points
