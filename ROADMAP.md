# Roadmap

## Done

- I follow TDD throughout the project.
- The agent's environment is ready: text-procedural tasks with discrete actions and terminal reward.
- The full Dreamer architecture is heavier than my setting needs (stochastic latent, KL balancing, symlog, twohot, continue head — redundant under a deterministic environment, a frozen language model and a binary terminal reward), so I built a compact version: deterministic latent, Transformer encoder over frozen embeddings, GRU dynamics, reward head, attention actor-critic trained by REINFORCE in imagination.
- Exact Shapley estimator: enumerates all 2^T coalitions; the reference for short episodes.
- Monte Carlo Shapley estimator (permutation sampling, Štrumbelj-Kononenko): linear in the number of steps.
- Both estimators are verified against the Shapley axioms.
- First experiment shows that the Shapley signal helps on a terminal-reward task: see `experiments/01_vanilla_vs_shapley/`.
- Google Colab is the testing environment; the first end-to-end run was performed there.
- The thesis skeleton is ready in English and Slovak.

## Planned

- Fill in the remaining thesis chapters.
- Compare against established credit-assignment baselines (HCA, RUDDER, AgeMem) and against manual reward shaping; sweep the number of Shapley samples and the action-segmentation strategy.
- Validate the world-model Shapley estimator against direct language-model rollouts as a ground-truth oracle.
- Stabilize training (entropy bonus, gradient clipping, learning-rate schedule, checkpointing) and run longer.
- Upgrade the world model toward a full DreamerV3 RSSM (categorical latent, KL balancing).
- Move from a frozen language model to fine-tuning the language model itself with the Shapley signal (LoRA + REINFORCE).
- Scale the task family: longer episodes, multi-hop QA, planning, memory-management tasks similar to those targeted by AgeMem.

Priorities may shift as results come in.