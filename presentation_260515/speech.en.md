# Speaker notes — Shapley credit assignment for Dreamer in terminal-reward tasks

Projektový seminár 1 · 15.05.2026 · Aleksandr Bukhtoiarov

## Slide 2 — Main motivation

I'll start with something that isn't directly my work but follows from it. The goal many LLM researchers chase is to squeeze the quality of a larger model out of a smaller one. In my work I approach this through Shapley and Dreamer.

## Slide 3 — Setting: RL with terminal reward

Now what I actually do. This is reinforcement learning in a terminal-reward setting: the episode is scored by a single scalar at the very end, no intermediate feedback. The standard problem is how to distribute this one signal across T actions.

## Slide 4 — Tool 1: Shapley decomposition

From SCAR 2025 I take the cooperative-game formulation. Actions are players, the terminal reward is the value of the full coalition, and Shapley decomposition gives each step its fair share. One sparse R turns into a dense per-action signal φᵢ.

## Slide 5 — Tool 2: world model as oracle v(S)

Shapley decomposition needs to evaluate the coalition value v(S). In RLHF this role is played by a reward model trained on preference pairs. In my case it's the Dreamer world model: encoder + GRU + reward head, trained on rollouts in the environment by MSE against the observed R. On the counterfactual trajectory, out-of-coalition actions are replaced by a baseline, we roll forward in latent space, and the reward head produces v(S).

## Slide 6 — Agent: cell-based working memory

The agent setup is non-standard. Not one growing text context, but K short cells, each a rewritable memory. Each action — the LLM picks a cell and writes to it; its own context receives only the current contents of the cells. Why exactly this — I'll come back to it at the end.

## Slide 7 — Architecture

The frozen LLM only runs during rollout collection. The encoder runs on each cell separately, the GRU updates state under an action, the reward head produces a scalar. Actor and critic are self-attention over K+1 state tokens, trained entirely in the world model's imagination via REINFORCE.

## Slide 8 — First experiment

The task is multi-step arithmetic with parentheses, LLM frozen. Vanilla barely beats the random policy. Shapley boosts the result by 2.3× — the dense signal on cell memory delivers the gain.

## Slide 9 — Back to the LLM: CoT bloat

Back to the motivation from slide one. The standard path to quality is long context or chain-of-thought; the LLM stuffs intermediate computation into its own input. Each step the context grows, but only a small part of it is actually load-bearing — current variables and active hypotheses. The rest is bloat, and a large LLM handles it through sheer capacity.

## Slide 10 — Long-term hypothesis: smaller LLM with cells

In my setup the LLM-call context stays short — only the current cells. This gives a long-term hypothesis: on tasks with long reasoning chains a smaller LLM with cell-based memory could catch up to a larger LLM without memory. I want to stress — this is a hypothesis and a direction, not today's result; the first experiment only tests credit assignment, the LLM size is fixed for now.

## Slide 11 — Near-term steps

First — push the arithmetic experiment to a stable solution. The current 0.30 from Shapley is a first run, no entropy bonus, no lr schedule; in parallel I need to ground-truth-validate φ by rolling the real LLM forward. Second — plug in XGBLoRA, the LLM fine-tuning method from my bachelor thesis, so the dense Shapley signal trains not only the actor but the LLM itself. Third — in parallel, finish the remaining chapters of the thesis.

## Slide 12 — Next: compare against competing methods

Once the pipeline is stable and the LLM starts fine-tuning, the goal is to beat the three closest credit-assignment methods on comparable tasks. AgeMem trains an agent over six memory operations through step-level GRPO. MemAct proposes a Prune&Write operator over cells and trains via DCPO. RUDDER is return decomposition through an LSTM return-predictor. All three are credit assignment in sparse-reward episodes, but without the cooperative-game formulation and without a world model as oracle. Long-term behind this stands the hypothesis from slide one: to empirically test whether a smaller LLM with cell-based memory can catch up to a larger one without memory.

## Slide 13 — Summary

Thank you. Code, tests, notes on key papers in EN and SK, LaTeX source — at the link.