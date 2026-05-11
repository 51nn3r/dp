from __future__ import annotations

from importlib import import_module

from .base import ShapleyEstimator, get_estimator


_MODULE_FOR_TYPE = {
    "perm_mc": "shapley_dreamer.shapley.permutation_mc",
    "exact": "shapley_dreamer.shapley.exact",
}


def build_shapley(settings=None) -> ShapleyEstimator:
    if settings is None:
        from shapley_dreamer import settings as default_settings

        settings = default_settings

    name = settings.SHAPLEY_ALGORITHM
    if name not in _MODULE_FOR_TYPE:
        raise ValueError(
            f"Unknown SHAPLEY_ALGORITHM={name!r}. Known: {sorted(_MODULE_FOR_TYPE)}"
        )

    import_module(_MODULE_FOR_TYPE[name])
    cls = get_estimator(name)
    return cls.from_settings(settings)