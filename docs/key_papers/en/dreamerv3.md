# Mastering Diverse Domains through World Models (DreamerV3)

[arXiv:2301.04104](https://arxiv.org/abs/2301.04104) · Hafner, Pasukonis, Ba, Lillicrap · 2023 · BibTeX: `dreamerv3`

## Problem

Model-based RL has long promised efficiency in environment interactions, but historically each domain required manual hyper-parameter tuning; an algorithm that worked well on one benchmark would underperform on another. V3 claims that one and the same set of hyper-parameters works across a broad range of domains from Atari and DMC to Minecraft and navigation, and confirms this with an extensive experimental part.

## Method

Architecturally, V3 continues the RSSM (Recurrent State-Space Model) line. The latent state has two parts: a deterministic part $h_t$ updated by GRU, plus a discrete stochastic part $z_t$ sampled from softmax distributions with straight-through gradients — the gradient passes through the discrete sample directly, as if it were continuous. The encoder yields a posterior $q(z_t \mid h_t, o_t)$ from the observation, and the dynamics yields a prior $p(z_t \mid h_{t-1}, a_{t-1})$ without the observation.

V3 uses **both KL balancing and free bits simultaneously**. KL balancing is a two-sided weighting scheme on the KL terms in the ELBO that prevents either posterior or prior from dominating. Free bits clip the dynamics and representation losses from below by a constant threshold; once the KL is already well minimised, these terms switch off and learning focuses on prediction.

The reward passes through **symlog**:

$$\mathrm{symlog}(x) = \mathrm{sign}(x) \ln(|x| + 1), \quad \mathrm{symexp}(x) = \mathrm{sign}(x)(\exp(|x|) - 1).$$

This non-linear compression makes learning insensitive to the scale of the reward. The critic output is encoded via **twohot symlog regression**: discrete regression in the symlog-transformed space with a softmax target across two nearest bins. This decouples the target scale from the gradient scale and removes loss explosions on the first non-zero reward in an episode.

The actor and critic are trained entirely in imagination. The world model is rolled out over a fixed horizon, the critic learns from $\lambda$-returns, and an EMA copy of the critic is used as a regulariser similar to a target network.

## Results

In the Minecraft Diamond Challenge, V3 agents find a diamond — for the first time an algorithm solves this from scratch, without human demonstrations and without curriculum. On Atari 200M, V3 surpasses DreamerV2, Rainbow and IQN; results across a broad set of tasks are obtained with fixed hyper-parameters.

## Limitations

The main strength of V3 is universality, and the paper lists few explicit limitations. The world model with a large discrete latent is expensive to train. Ablations in the paper show that the components (KL balancing, free bits, symlog, twohot, EMA critic regularisation) substantially affect both stability and learning speed.

## Connection to my work

I take the V3 skeleton — world model plus actor-critic in imagination — and remove the stochastic latent, KL balancing, free bits, symlog, twohot, EMA critic regularisation and continue head. None of these are needed under a deterministic LLM and a binary reward. The encoder uses a Transformer over frozen Qwen embeddings instead of a CNN; the actor update is REINFORCE over discrete actions rather than reparameterisation.