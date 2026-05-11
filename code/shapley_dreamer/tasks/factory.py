from __future__ import annotations

from importlib import import_module

from .base import TaskGenerator, get_task


_MODULE_FOR_TYPE = {
    "word_problem": "shapley_dreamer.tasks.word_problem",
}


def build_task(settings=None) -> TaskGenerator:
    if settings is None:
        from shapley_dreamer import settings as default_settings

        settings = default_settings

    name = settings.TASK_TYPE
    if name not in _MODULE_FOR_TYPE:
        raise ValueError(
            f"Unknown TASK_TYPE={name!r}. Known: {sorted(_MODULE_FOR_TYPE)}"
        )

    import_module(_MODULE_FOR_TYPE[name])
    cls = get_task(name)
    return cls.from_settings(settings)