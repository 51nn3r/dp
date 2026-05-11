# 01 — Vanilla Dreamer vs Shapley decomposition

Comparison of two reward regimes on the SoT arithmetic task with a frozen Qwen backend.

## Task

Multi-step parenthesised arithmetic over six operands with operators `+ - *` and values in `0..20`. Reward is binary: 1 if the final cell content equals the ground-truth value, 0 otherwise.

## Setup

- Grid: K=3 cells, T_max=4 steps, N=24 trajectories per training step.
- World model: 30 epochs of pretraining.
- Actor-critic: 300 imagination updates.
- Evaluation: 500 rollouts of the trained policy.
- Random-policy success baseline: 0.116.
- Full run settings are stored in `curves.json` under `settings`.

## Files

- `colab_dreamer.ipynb` — the Colab notebook that produced this run.
- `curves.json` — settings, per-epoch world-model loss, and per-update actor/critic curves for both regimes.
- `curves.png` — vanilla vs Shapley training-success curves.
- `rollouts.jsonl` — sample episode traces (one episode per line).

## Result

Peak training success: vanilla 0.133, Shapley 0.300 (random baseline 0.116). Both regimes collapse late in training near step 300; this run does not include entropy regularisation or learning-rate scheduling.