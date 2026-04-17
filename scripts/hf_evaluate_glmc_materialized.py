from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.config import load_glmc_config
from ece888_sagmc.glmc_materialize import GLMCMaterializer, build_glmc_materialized_model
from ece888_sagmc.hf_gpt2 import build_hf_data, evaluate_hf_state_loss, write_hf_metrics
from ece888_sagmc.metrics import compute_barrier
from ece888_sagmc.plotting import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a materialized official GLMC GPTMerger path.")
    parser.add_argument("--config", default="configs/glmc_tiny_shakespeare.yaml")
    parser.add_argument("--merge-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--coeff-start", type=float, default=None)
    parser.add_argument("--coeff-end", type=float, default=None)
    parser.add_argument("--coeff-step", type=float, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--reference-json",
        default=None,
        help="Optional eval_merge_* JSON used to print max abs diff against coeff_losses_learned_matching.",
    )
    return parser.parse_args()


def _coeff_grid(start: float, end: float, step: float) -> list[float]:
    coeffs = []
    coeff = start
    while coeff <= end + 1e-9:
        coeffs.append(float(round(coeff, 10)))
        coeff += step
    return coeffs


def _print_reference_diff(reference_json: str | Path, rows: list[dict[str, float]]) -> None:
    with Path(reference_json).open("r", encoding="utf-8") as f:
        ref = json.load(f)
    ref_losses = ref.get("coeff_losses_learned_matching")
    if not ref_losses:
        print(f"reference {reference_json} has no coeff_losses_learned_matching")
        return
    diffs = []
    for row in rows:
        key = f"{row['coeff']:.6f}"
        if key in ref_losses:
            diffs.append(abs(float(row["loss"]) - float(ref_losses[key])))
    if diffs:
        print(f"reference max_abs_diff: {max(diffs):.6f}")
        print(f"reference mean_abs_diff: {sum(diffs) / len(diffs):.6f}")


def main() -> None:
    args = parse_args()
    cfg = load_glmc_config(args.config)
    coeff_start = float(args.coeff_start if args.coeff_start is not None else cfg["eval"]["coeff_start"])
    coeff_end = float(args.coeff_end if args.coeff_end is not None else cfg["eval"]["coeff_end"])
    coeff_step = float(args.coeff_step if args.coeff_step is not None else cfg["eval"]["coeff_step"])
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = cfg["eval"].get("max_batches")

    materializer = GLMCMaterializer(args.merge_dir)
    model = build_glmc_materialized_model(args.merge_dir)
    data = build_hf_data(cfg["data"], block_size=int(cfg["model"]["block_size"]))

    coeffs = _coeff_grid(coeff_start, coeff_end, coeff_step)
    losses = []
    for coeff in coeffs:
        state = materializer.materialize_state(coeff)
        loss = evaluate_hf_state_loss(
            model=model,
            state=state,
            data=data,
            split=str(cfg["eval"]["split"]),
            batch_size=int(cfg["eval"]["batch_size"]),
            device=args.device,
            max_batches=max_batches,
        )
        print(f"[glmc-materialized] coeff={coeff:.6f} loss={loss:.6f}")
        losses.append(loss)

    triples = sorted((1.0 - c, loss, c) for c, loss in zip(coeffs, losses))
    barrier, points = compute_barrier(
        [alpha for alpha, _loss, _coeff in triples],
        [loss for _alpha, loss, _coeff in triples],
    )
    rows = [p.__dict__ | {"coeff": triples[p.index][2]} for p in points]
    write_hf_metrics(
        args.out_dir,
        barrier,
        rows,
        extra={"method": "glmc_materialized", "merge_dir": args.merge_dir},
    )
    plot_metrics(Path(args.out_dir) / "metrics.json", Path(args.out_dir) / "loss_curve.png", "GLMC materialized")
    print(f"glmc materialized barrier: {barrier:.6f}")
    if args.reference_json:
        _print_reference_diff(args.reference_json, rows)


if __name__ == "__main__":
    main()
