import importlib

MODULES = [
    "shapley_dreamer",
    "shapley_dreamer.settings",
    "shapley_dreamer.envs",
    "shapley_dreamer.envs.sot",
    "shapley_dreamer.llm",
    "shapley_dreamer.llm.base",
    "shapley_dreamer.llm.factory",
    "shapley_dreamer.llm.mock",
    "shapley_dreamer.llm.prompts",
    "shapley_dreamer.tasks",
    "shapley_dreamer.tasks.base",
    "shapley_dreamer.tasks.factory",
    "shapley_dreamer.tasks.word_problem",
    "shapley_dreamer.shapley",
    "shapley_dreamer.shapley.base",
    "shapley_dreamer.shapley.factory",
    "shapley_dreamer.shapley.exact",
    "shapley_dreamer.shapley.permutation_mc",
    "shapley_dreamer.shapley.counterfactual",
    "shapley_dreamer.training",
    "shapley_dreamer.training.trajectory",
    "shapley_dreamer.training.rollout",
    "shapley_dreamer.training.metrics",
    "shapley_dreamer.training.dreamer_loop",
    "shapley_dreamer.world_model",
    "shapley_dreamer.world_model.encoder",
    "shapley_dreamer.world_model.model",
    "shapley_dreamer.world_model.training",
    "shapley_dreamer.agent",
    "shapley_dreamer.agent.actor_critic",
    "shapley_dreamer.agent.imagination",
    "shapley_dreamer.shapley.imagined",
]


def test_modules_import():
    for name in MODULES:
        importlib.import_module(name)
