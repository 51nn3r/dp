# The Shapley Value in Machine Learning

[arXiv:2202.05594](https://arxiv.org/abs/2202.05594) · Rozemberczki, Watson, Bayer, Yang, Kiss, Nilsson, Sarkar · 2022 (IJCAI survey track) · BibTeX: `shapleyml2022`

## Theoretical foundation

Players $N = \{1, \ldots, n\}$, value function $v: 2^N \to \mathbb{R}$ with $v(\emptyset) = 0$. The Shapley value of player $i$:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,[v(S \cup \{i\}) - v(S)].$$

The weights $\frac{|S|!(n-|S|-1)!}{n!}$ are the probability that $S$ is the set of players coming before $i$ in a uniformly random permutation. This is the "fair share" of $i$ in $v(N) - v(\emptyset)$.

## Four axioms

The Shapley value is the unique distribution satisfying all of the following simultaneously:

**Efficiency**: $\sum_{i \in N} \phi_i(v) = v(N) - v(\emptyset)$.

**Symmetry**: if $v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S$, then $\phi_i = \phi_j$. Players that contribute identically receive equal credit.

**Dummy (null) player**: if $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i = 0$. A player with no effect gets zero.

**Linearity / Additivity**: for two games $v_1, v_2$ on the same set of players, $\phi_i(v_1 + v_2) = \phi_i(v_1) + \phi_i(v_2)$.

The survey emphasises that these properties characterise the Shapley value — there is no other distribution satisfying all four.

## Applications in ML

The survey covers feature selection, data valuation, federated learning (the contribution of data silos), explainable ML (instance-level attribution through the SHAP family), multi-agent RL (distributing the global reward among agents), and model valuation in ensembles.

RL is treated only in the multi-agent variant, where players are agents. Single-agent credit assignment at the level of individual actions of one trajectory is not discussed among the listed applications.

## Computational methods

The exact computation requires $2^n$ evaluations of $v$, which becomes intractable already on dozens of players. The survey discusses three approximation methods.

Monte Carlo over permutations — a universal estimator, $O(M \cdot n)$ complexity, unbiased. The general formulation for cooperative games is from Castro et al. 2009; in feature attribution a close approach is due to Štrumbelj & Kononenko 2014.

Multilinear extension — a move from permutation sums to integrals over probability distributions, with sampling-based approximation.

Linear-regression approximation — the KernelSHAP method (Lundberg & Lee 2017), solving a weighted least-squares problem on a subsample of coalitions.

## Connection to my work

The four axioms (efficiency, symmetry, dummy player, additivity) are verified by tests on my Shapley estimators. The application to single-agent credit assignment is a setting that is absent from the survey's list of applications.