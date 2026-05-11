from __future__ import annotations

from importlib import import_module

from .base import LLMBackend, get_backend


_MODULE_FOR_TYPE = {
    "mock": "shapley_dreamer.llm.mock",
    "hf": "shapley_dreamer.llm.hf",
}


def build_llm(settings=None) -> LLMBackend:
    if settings is None:
        from shapley_dreamer import settings as default_settings

        settings = default_settings

    name = settings.LLM_TYPE
    if name not in _MODULE_FOR_TYPE:
        raise ValueError(
            f"Unknown LLM_TYPE={name!r}. Known: {sorted(_MODULE_FOR_TYPE)}"
        )

    import_module(_MODULE_FOR_TYPE[name])
    cls = get_backend(name)
    return cls.from_settings(settings)