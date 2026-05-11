# SCAR: Shapley Credit Assignment for More Efficient RLHF

[arXiv:2505.20417](https://arxiv.org/abs/2505.20417) · 2025 · BibTeX: `scar2025`

## Problem

In RLHF the reward model produces a scalar over the whole generated sequence. One scalar over a long token sequence gives every token the same gradient signal — high variance, slow convergence, poor discrimination between important and filler tokens.

The authors treat generation as a cooperative game. Tokens or token groups (units in the paper's terminology) are players, and the terminal reward is the value of the grand coalition. The Shapley contribution of each unit is its fair share of the total value under the axioms of efficiency, symmetry, dummy player and additivity.

## Method

The coalition value $v(S)$ is built as follows: only the units in $S$ are kept in their natural order from the original sequence, the missing positions are filled with blanks, and the resulting partial sequence $y_S$ is fed to the reward model. The reward model returns a scalar — this is $v(S)$.

The exact computation of $\phi_i$ is exponential in the number of units. SCAR sidesteps this with Owen-value approximation under adaptive segmentation: units are grouped into meaningful segments, and Owen values are computed over this hierarchy (the SHAP package is used for the underlying computation). The authors claim a reduction in complexity from exponential to quadratic in the number of units.

SCAR does not replace the terminal signal entirely; it mixes the Shapley signal with the original at every step:

$$R_t = R_t^{KL} + \alpha\, R_t^{Shapley} + (1 - \alpha)\, \mathbb{1}_{t = T}\, R^{terminal}.$$

Here $R_t^{Shapley}$ is the dense Shapley signal distributed over all steps, $R^{terminal}$ is delivered only at the last step via the indicator $\mathbb{1}_{t = T}$, and $R_t^{KL}$ is the usual RLHF KL penalty against the reference model. The coefficient $\alpha$ balances the dense Shapley signal against the original terminal reward.

## Results

SCAR shows faster convergence and higher final reward than sparse RLHF (a single terminal scalar) and attention-weighted baselines.

## Limitations

The authors highlight the computational overhead of multiple reward-model calls for evaluating partial sequences. The whole method also depends on how well the reward model evaluates partial sequences — it was trained on complete ones, and its behaviour on partials needs verification.

## Connection to my work

I take the same cooperative-game formulation as SCAR, but compute the coalition value through a world model in latent space, without a separate reward model and without LLM re-runs. On my short episodes, exact enumeration of $2^T$ coalitions is available, which removes the approximation error of Owen segmentation.