# Explaining Prediction Models and Individual Predictions with Feature Contributions

[doi:10.1007/s10115-013-0679-x](https://doi.org/10.1007/s10115-013-0679-x) · Štrumbelj, Kononenko · *Knowledge and Information Systems* · 2014 · BibTeX: `strumbelj2014`

## Problem

Given a predictive model $f$ (classifier, regressor) and an example $x = (x_1, \ldots, x_n)$, the goal is to explain $f(x)$ via the contributions of individual features. The method treats features as players in a cooperative game and $f$ as the value function. The coalition value $v(S)$ is theoretically defined as the conditional expectation $\mathbb{E}[f(X) \mid X_S = x_S]$; in practice the paper estimates it through sampling approximation under an independence assumption. The Shapley contribution

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,[v(S \cup \{i\}) - v(S)]$$

is uniquely determined by the axioms of efficiency, symmetry, dummy player and additivity. The challenge is exponential complexity: $2^n$ evaluations of $v$.

## Method

Monte Carlo over permutations. We sample $M$ random permutations $\pi^{(1)}, \ldots, \pi^{(M)}$ of $\{1, \ldots, n\}$. For each permutation and each feature $i$, we compute the marginal contribution

$$\Delta_i^{(m)} = v(\text{Pre}_i^{\pi^{(m)}} \cup \{i\}) - v(\text{Pre}_i^{\pi^{(m)}}),$$

where $\text{Pre}_i^{\pi}$ is the set of features coming before $i$ in the permutation. Averaging gives an unbiased estimate:

$$\hat\phi_i = \frac{1}{M}\sum_{m=1}^M \Delta_i^{(m)}.$$

The complexity drops to $O(M \cdot n)$. The estimate converges to the true Shapley value as $M \to \infty$.

For my full-permutation implementation (where each permutation is traversed entirely), the per-permutation efficiency property holds:

$$\sum_{i=1}^n \Delta_i^{\pi} = v(N) - v(\emptyset)$$

by telescoping. The efficiency axiom is preserved not only in expectation but also for each individual permutation. This follows from the algorithm's structure rather than being proven as a separate property in the paper.

## Results

The method applies to classification and regression, accounts for subsets, interactions and redundancies among features. It became the standard for feature attribution and the basis for SHAP (Lundberg & Lee 2017) and its successors.

## Limitations

The sampling approximation in the paper relies on an independence assumption between features. As a consequence, on strongly correlated features the estimate is biased in the distribution of credit between them. Variance grows with $n$.

## Connection to my work

I use the permutation Monte Carlo estimator from this paper as one of two ways of computing Shapley values in my environment; the other is exact enumeration over $2^T$ coalitions, which serves as a baseline for tests on short episodes. Convergence of the permutation estimate to the exact one and per-permutation efficiency are verified by tests on synthetic value functions.