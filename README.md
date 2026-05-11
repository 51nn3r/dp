# Shapley credit assignment for the Dreamer algorithm in terminal-reward tasks

Master's thesis project, Faculty of Mathematics, Physics and Informatics, Comenius University in Bratislava (Applied Informatics, 2026).

Author: Bc. Aleksandr Bukhtoiarov.
Supervisor: Mgr. Marek Šuppa.

## Idea

In a reinforcement-learning episode with only a terminal reward, the agent receives one scalar at the end of a long sequence and no intermediate feedback. This thesis splits that terminal reward across the actions of the episode using Shapley values, with coalition values estimated by latent rollouts of a Dreamer-style world model. The world model already trained inside Dreamer plays the role of the coalition-value oracle, so no separate reward model and no extra environment interaction are needed. The per-action contributions are then used as a dense training signal for the policy and for the dynamics model.

## Layout

- `thesis/en/` — English version.
- `thesis/sk/` — Slovak version.
- `thesis/images/`, `thesis/references.bib` — shared assets.
- `code/` — Python package (`shapley_dreamer`), configs, scripts.
- `tests/` — unit tests.
- `notebooks/` — Colab notebooks.

## Building the thesis

From `thesis/en/` (or `thesis/sk/`):

    pdflatex main && bibtex main && pdflatex main && pdflatex main

## License

Not yet decided.
