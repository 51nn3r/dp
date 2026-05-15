# Roadmap

## Done

- TDD throughout.
- Environment ready: short text tasks with one final reward.
- Compact Dreamer-like world model — full DreamerV3 would be overkill.
- Actor and critic trained inside the world model's imagined rollouts.
- Two Shapley estimators: exact for short episodes, Monte Carlo linear in the number of steps.
- Both checked against the Shapley axioms.
- First experiment: Shapley beats vanilla and random on a terminal-reward task.
- Google Colab as the testing environment.
- Thesis skeleton ready in English and Slovak.

## Planned

- Polish the arithmetic experiment.
- Cross-check the credit-assignment scores against direct LLM rollouts.
- Plug in XGBLoRA to train the LLM itself.
- Try harder tasks (multi-hop QA, planning, memory).
- Compare against competing methods (RUDDER, AgeMem, MemAct).
- Sweep the Shapley estimator.
- Upgrade the world model to a full DreamerV3 RSSM.
- Write the missing thesis chapters.

Priorities may shift as results come in.