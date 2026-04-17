from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ece888_sagmc.checkpoint import load_model_and_state, load_state_dict
from ece888_sagmc.config import ensure_dir, load_config
from ece888_sagmc.data import build_dataset
from ece888_sagmc.geodesic import evaluate_waypoint_states, load_geodesic_path, optimize_geodesic, write_geodesic_eval
from ece888_sagmc.interpolate import evaluate_linear_path, write_path_metrics
from ece888_sagmc.model import GPT, GPTConfig
from ece888_sagmc.plotting import plot_metrics
from ece888_sagmc.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tiny Shakespeare SA-GMC experiment.")
    parser.add_argument("--config", default="configs/tiny_shakespeare.yaml")
    parser.add_argument("--ckpt-a", default=None)
    parser.add_argument("--ckpt-b", default=None)
    parser.add_argument("--ckpt-b-aligned", default=None, help="External GLMC-aligned endpoint.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-geodesic", action="store_true")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--geodesic-iters", type=int, default=None)
    parser.add_argument("--num-waypoints", type=int, default=None)
    return parser.parse_args()


def _evaluate_linear(name: str, model: GPT, state_a, state_b, dataset, cfg, out_dir: Path, device: str) -> None:
    barrier, points = evaluate_linear_path(
        model=model,
        state_a=state_a,
        state_b=state_b,
        dataset=dataset,
        split=str(cfg["eval"]["split"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        eval_iters=int(cfg["eval"]["eval_iters"]),
        num_points=int(cfg["eval"]["num_points"]),
        device=device,
    )
    method_dir = out_dir / name
    write_path_metrics(method_dir, barrier, points, extra={"method": name})
    plot_metrics(method_dir / "metrics.json", method_dir / "loss_curve.png", name)
    print(f"{name} barrier: {barrier:.6f}")


def _run_geodesic(name: str, model: GPT, model_config, state_a, state_b, dataset, cfg, out_dir: Path, device: str) -> None:
    method_dir = out_dir / name
    path_file = optimize_geodesic(
        model=model,
        state_a=state_a,
        state_b=state_b,
        dataset=dataset,
        cfg=cfg["geodesic"],
        out_dir=method_dir,
        device=device,
        metadata={"method": name},
    )
    path_obj = load_geodesic_path(path_file)
    eval_model = GPT(GPTConfig.from_dict(model_config))
    barrier, rows = evaluate_waypoint_states(
        model=eval_model,
        states=path_obj["waypoints"],
        dataset=dataset,
        split=str(cfg["eval"]["split"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        eval_iters=int(cfg["eval"]["eval_iters"]),
        device=device,
    )
    write_geodesic_eval(method_dir / "eval", barrier, rows, extra={"method": name})
    plot_metrics(method_dir / "eval" / "metrics.json", method_dir / "eval" / "loss_curve.png", name)
    print(f"{name} barrier: {barrier:.6f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_iters is not None:
        cfg["train"]["max_iters"] = args.max_iters
    if args.geodesic_iters is not None:
        cfg["geodesic"]["iterations"] = args.geodesic_iters
    if args.num_waypoints is not None:
        cfg["geodesic"]["num_waypoints"] = args.num_waypoints

    out_dir = ensure_dir(args.out_dir or cfg["experiment"]["out_dir"])
    dataset = build_dataset(cfg["data"])

    ckpt_a = Path(args.ckpt_a) if args.ckpt_a else out_dir / "seed0" / "best.pt"
    ckpt_b = Path(args.ckpt_b) if args.ckpt_b else out_dir / "seed1" / "best.pt"

    if not args.skip_train:
        if not ckpt_a.exists():
            ckpt_a = train_model(
                dataset,
                cfg["model"],
                cfg["train"],
                out_dir / "seed0",
                seed=int(cfg["experiment"]["seed_a"]),
                device=args.device,
            )
        if not ckpt_b.exists():
            ckpt_b = train_model(
                dataset,
                cfg["model"],
                cfg["train"],
                out_dir / "seed1",
                seed=int(cfg["experiment"]["seed_b"]),
                device=args.device,
            )

    model, state_a, model_config, _meta = load_model_and_state(ckpt_a)
    state_b = load_state_dict(ckpt_b)
    _evaluate_linear("naive_linear", model, state_a, state_b, dataset, cfg, out_dir, args.device)

    if args.ckpt_b_aligned:
        aligned_b = load_state_dict(args.ckpt_b_aligned)
        _evaluate_linear("glmc_linear_external", model, state_a, aligned_b, dataset, cfg, out_dir, args.device)
    else:
        aligned_b = None
        print("No --ckpt-b-aligned supplied; skipping GLMC-linear and SA-GMC.")

    if not args.skip_geodesic:
        _run_geodesic("gmc", model, model_config, state_a, state_b, dataset, cfg, out_dir, args.device)
        if aligned_b is not None:
            model_for_sa = GPT(GPTConfig.from_dict(model_config))
            _run_geodesic("sa_gmc", model_for_sa, model_config, state_a, aligned_b, dataset, cfg, out_dir, args.device)


if __name__ == "__main__":
    main()
