# Proximal Policy Optimization Algorithms (PPO)

[arXiv:1707.06347](https://arxiv.org/abs/1707.06347) · Schulman, Wolski, Dhariwal, Radford, Klimov · 2017 · BibTeX: `ppo2017`

## Problem

REINFORCE estimates the gradient as

$$\hat g = \hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \, \hat A_t\right],$$

where $\hat{\mathbb{E}}_t$ is sample-mean estimation and $\hat A_t$ is the advantage estimate. In practice $\hat g$ is obtained by differentiating the loss

$$L^{PG}(\theta) = \hat{\mathbb{E}}_t\left[\log \pi_\theta(a_t \mid s_t) \, \hat A_t\right].$$

The algorithm is unstable: a single large gradient step takes the policy far from the current one, and performance collapses. TRPO bounds this by an explicit KL-divergence constraint between the new and the old policy, but requires second-order optimisation, which is expensive and awkward on large networks.

PPO offers a first-order approximation to the same objective.

## Method

The probability ratio between the new and the old policy:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}.$$

The clipped surrogate loss:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta) \hat A_t, \; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t\right)\right].$$

A small $\epsilon$ limits how far the policy can move in a single step. Clipping zeroes the gradient only when the ratio leaves the window **in the direction that would improve the objective**: when $\hat A_t > 0$ and $r_t > 1 + \epsilon$ (the actor wants to push up an already good action even harder), or when $\hat A_t < 0$ and $r_t < 1 - \epsilon$ (the actor wants to push down an already bad action even harder). In the opposite direction the gradient is preserved, so the policy can still correct mistakes.

The outer $\min$ gives a pessimistic lower bound: for each step we take the smaller of the raw and the clipped ratio, so improvements to the objective cannot be overestimated either at positive or at negative advantage.

The advantage is estimated via GAE (Schulman et al. 2016):

$$\hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \ldots + (\gamma\lambda)^{T-t-1}\delta_{T-1}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$

$\lambda$ trades bias against variance: at $\lambda = 0$ we get pure TD targets (low variance, high bias), at $\lambda = 1$ we get a Monte-Carlo-like return over a full trajectory; on a truncated segment, $\lambda = 1$ may still include a bootstrap $V(s_T)$ at the end.

PPO performs several epochs of updates on the same batch of experience. Clipping guarantees that even after several epochs the policy does not drift too far.

The full loss combines actor, critic and entropy bonuses:

$$L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb{E}}_t\left[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t)\right],$$

where $L^{VF}_t$ is the squared value loss, $S[\pi_\theta](s_t)$ is policy entropy at state $s_t$, and $c_1$, $c_2$ are weighting coefficients.

## Results

On MuJoCo (Hopper, Walker, HalfCheetah) PPO matches TRPO at much lower implementation cost. On Atari the paper compares to A2C and ACER: PPO wins on average training reward across 30 games, while ACER wins on average reward over the last 100 episodes on 28 games versus 19 for PPO.

PPO became mainstream: OpenAI Five was trained with PPO, and InstructGPT and SCAR use PPO as the outer loop in RLHF.

## Limitations

PPO is on-policy: every update requires fresh rollouts from the current policy. Off-policy methods (SAC, TD3) are more sample-efficient on expensive environments. The clipped objective is a first-order approximation to the trust region, and on tasks with a narrow optimal-policy region PPO can be unstable.

## Connection to my work

My actor-critic update is REINFORCE with a baseline and without clipping. PPO is the natural extension for longer horizons, and switching to it would make my setup directly comparable with SCAR, where PPO is the outer loop with per-token Shapley rewards.