# ECE 888 SA-GMC Tiny Shakespeare Experiments

This repository implements the non-GLMC parts of the report experiment for **Tiny Shakespeare only**.
There are now two tracks:

- `configs/glmc_tiny_shakespeare.yaml`: HuggingFace GPT-2 setup aligned with the official GLMC Tiny Shakespeare repo.
- `configs/tiny_shakespeare.yaml`: lightweight char-level GPT smoke/prototyping setup.

For results that need to compare with the official GLMC baseline, use the HuggingFace GPT-2 track.

The experiment code can:

- train two independent GPT-style language models from different seeds;
- evaluate naive linear mode connectivity;
- run geodesic mode connectivity by minimizing the discrete JSD energy over model waypoints;
- run SA-GMC from an externally produced GLMC `GPTMergerWrapper` checkpoint by materializing fixed-coefficient endpoints;
- evaluate and plot loss barriers for linear and geodesic paths.

The GLMC alignment training algorithm itself is intentionally not reimplemented here. The official repository represents learned GLMC as a `GPTMergerWrapper` interpolation model, not as a plain `pi(theta_B)` HuggingFace checkpoint. This code therefore includes a materialization path for teammate-produced `merge_seed_*` checkpoints and runs geodesic optimization in the GLMC-aligned parameterization.

Official GLMC repo: https://github.com/alexandertheus/Generalized-LMC-for-Transformers

## References Reflected In Code

- GMC follows Tan et al.: initialize `N` waypoints on the linear path, keep endpoints fixed, optimize the interior waypoints with SGD, and minimize `sum_i JSD(p_i || p_{i+1})` using logits on unlabeled minibatches.
- The report's Tiny Shakespeare setting uses GPT-2-style autoregressive language modeling. The default model here is a compact GPT decoder so the experiment is runnable locally; the model size can be increased in `configs/tiny_shakespeare.yaml`.
- GLMC is treated as a preprocessing stage that outputs a functionally equivalent, symmetry-aligned version of checkpoint B.

## Setup

```bash
pip install -r requirements.txt
```

## Full Tiny Shakespeare Run

## Official-GLMC-Aligned GPT-2 Track

Prepare the same contiguous Tiny Shakespeare splits and fixed GPT-2 tokenizer:

```bash
python scripts/hf_prepare_tiny.py --config configs/glmc_tiny_shakespeare.yaml
```

Train two official-style HuggingFace GPT-2 models:

```bash
python scripts/hf_train.py --config configs/glmc_tiny_shakespeare.yaml --seed 0
python scripts/hf_train.py --config configs/glmc_tiny_shakespeare.yaml --seed 1
```

This matches the official repo defaults: GPT-2 tokenizer, contiguous 90/5/5 split, `block_size=256`, `n_layer=6`, `n_embd=256`, `n_head=4`, `n_inner=1024`, `epochs=100`, and early stopping.

Evaluate naive linear interpolation:

```bash
python scripts/hf_evaluate_linear.py \
  --config configs/glmc_tiny_shakespeare.yaml \
  --model-a gpt2_tinyshakespeare_seed0_nembd256 \
  --model-b gpt2_tinyshakespeare_seed1_nembd256 \
  --out-dir runs/glmc_tiny_shakespeare/naive_linear
```

Run GMC on those same official-style model directories:

```bash
python scripts/hf_optimize_geodesic.py \
  --config configs/glmc_tiny_shakespeare.yaml \
  --model-a gpt2_tinyshakespeare_seed0_nembd256 \
  --model-b gpt2_tinyshakespeare_seed1_nembd256 \
  --out-dir runs/glmc_tiny_shakespeare/gmc
```

For a quick A100 sanity pass before the full run:

```bash
python scripts/hf_optimize_geodesic.py \
  --config configs/glmc_tiny_shakespeare.yaml \
  --model-a gpt2_tinyshakespeare_seed0_nembd256 \
  --model-b gpt2_tinyshakespeare_seed1_nembd256 \
  --out-dir runs/glmc_tiny_shakespeare/gmc_smoke \
  --num-waypoints 5 \
  --iterations 10 \
  --batch-size 2 \
  --sequence-length 32 \
  --max-eval-batches 2
```

If your teammate gives you a plain exported GLMC-aligned endpoint directory, use the same command and set `--method sa_gmc`:

```bash
python scripts/hf_optimize_geodesic.py \
  --config configs/glmc_tiny_shakespeare.yaml \
  --model-a gpt2_tinyshakespeare_seed0_nembd256 \
  --model-b path/to/exported_glmc_aligned_b \
  --method sa_gmc \
  --out-dir runs/glmc_tiny_shakespeare/sa_gmc
```

If your teammate gives you an official GLMC `GPTMergerWrapper` checkpoint directory such as `merge_seed_0_1`, first verify that materialization reproduces the GLMC baseline:

```bash
python scripts/hf_evaluate_glmc_materialized.py \
  --config configs/glmc_tiny_shakespeare.yaml \
  --merge-dir merge_seed_0_1 \
  --out-dir runs/glmc_tiny_shakespeare/glmc_materialized_check \
  --reference-json eval_merge_seed_0_1/merged_coeff_losses_weight_learned_vanilla.json
```

Then run the medium SA-GMC experiment:

```bash
python scripts/hf_optimize_sa_gmc_from_glmc.py \
  --config configs/glmc_tiny_shakespeare.yaml \
  --merge-dir merge_seed_0_1 \
  --out-dir runs/glmc_tiny_shakespeare/sa_gmc_medium_lr1e-3 \
  --num-waypoints 10 \
  --iterations 500 \
  --batch-size 8 \
  --sequence-length 64 \
  --learning-rate 0.001 \
  --momentum 0.0 \
  --grad-clip 1.0 \
  --log-interval 50 \
  --max-eval-batches 5
```

Generate the comparison figures:

```bash
python scripts/plot_sagmc_results.py
```

By default this command reads the lightweight curated metrics in `results/tiny_shakespeare/`
and writes regenerated figures to `runs/glmc_tiny_shakespeare/figures/`. Curated final
figures and metrics are stored under `results/tiny_shakespeare/`.

## Lightweight Char-Level Track

From the repository root:

```bash
python scripts/run_tiny_shakespeare.py --config configs/tiny_shakespeare.yaml
```

This will train two checkpoints, evaluate the naive linear path, and run GMC. To also run SA-GMC, provide the GLMC-aligned checkpoint from your teammate:

```bash
python scripts/run_tiny_shakespeare.py \
  --config configs/tiny_shakespeare.yaml \
  --ckpt-b-aligned path/to/glmc_aligned_b.pt
```

Outputs are written under `runs/tiny_shakespeare/` by default.

## Individual Commands

Train one model:

```bash
python scripts/train.py --config configs/tiny_shakespeare.yaml --seed 0 --out-dir runs/tiny_shakespeare/seed0
python scripts/train.py --config configs/tiny_shakespeare.yaml --seed 1 --out-dir runs/tiny_shakespeare/seed1
```

Evaluate a linear path:

```bash
python scripts/evaluate_linear.py \
  --config configs/tiny_shakespeare.yaml \
  --ckpt-a runs/tiny_shakespeare/seed0/best.pt \
  --ckpt-b runs/tiny_shakespeare/seed1/best.pt \
  --out runs/tiny_shakespeare/naive_linear
```

Optimize a geodesic path:

```bash
python scripts/optimize_geodesic.py \
  --config configs/tiny_shakespeare.yaml \
  --ckpt-a runs/tiny_shakespeare/seed0/best.pt \
  --ckpt-b runs/tiny_shakespeare/seed1/best.pt \
  --out-dir runs/tiny_shakespeare/gmc
```

Optimize SA-GMC using an externally aligned endpoint:

```bash
python scripts/optimize_geodesic.py \
  --config configs/tiny_shakespeare.yaml \
  --ckpt-a runs/tiny_shakespeare/seed0/best.pt \
  --ckpt-b path/to/glmc_aligned_b.pt \
  --out-dir runs/tiny_shakespeare/sa_gmc
```

Evaluate a saved geodesic path:

```bash
python scripts/evaluate_geodesic.py \
  --config configs/tiny_shakespeare.yaml \
  --path runs/tiny_shakespeare/gmc/geodesic_path.pt \
  --out runs/tiny_shakespeare/gmc_eval
```

## Fast Smoke Test

This checks the code path without doing a real experiment:

```bash
python tests/smoke_test.py
```

## Checkpoint Format

Native checkpoints contain:

- `model_state`: PyTorch state dict;
- `model_config`: GPT model configuration;
- `vocab`: character vocabulary metadata;
- training metadata such as seed and validation loss.

For external GLMC output, either pass a native checkpoint or a raw PyTorch state dict. If the aligned file has no `model_config`, the scripts reuse checkpoint A's model config.

## Public Release Notes

Large generated artifacts are intentionally not tracked in Git:

- Tiny Shakespeare data, GPT-2 tokenizer, and prepared splits can be regenerated with `scripts/hf_prepare_tiny.py`.
- Model checkpoints such as `gpt2_tinyshakespeare_seed*/`, `merge_seed_*/`, `*.safetensors`, and `*.pt` files are excluded from the public repository.
- Full experiment runs under `runs/` are excluded because geodesic paths can be several GB.
- The public repository keeps only lightweight curated artifacts in `results/tiny_shakespeare/`.
