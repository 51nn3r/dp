from types import SimpleNamespace

import pytest

from shapley_dreamer.llm.base import (
    LLMBackend,
    _REGISTRY,
    get_backend,
    register,
    registered_backends,
)
from shapley_dreamer.llm.factory import build_llm


def _settings(**overrides):
    base = dict(LLM_TYPE="mock", LLM_MAX_NEW_TOKENS=8, RANDOM_SEED=0)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        LLMBackend()  # type: ignore[abstract]


def test_register_rejects_non_subclass():
    with pytest.raises(TypeError):
        @register("not_a_backend")
        class _Bogus:  # type: ignore[misc]
            pass


def test_register_rejects_duplicates():
    class _Dummy(LLMBackend):
        def generate(self, cells, target_cell, max_new_tokens):
            return ""

        @classmethod
        def from_settings(cls, settings):
            return cls()

    register("dup_test_x")(_Dummy)
    try:
        with pytest.raises(ValueError):
            register("dup_test_x")(_Dummy)
    finally:
        _REGISTRY.pop("dup_test_x", None)


def test_get_backend_unknown_raises():
    with pytest.raises(KeyError):
        get_backend("definitely_not_registered")


def test_factory_builds_mock_via_settings():
    backend = build_llm(_settings(LLM_TYPE="mock"))
    assert backend.__class__.__name__ == "MockBackend"
    assert "mock" in registered_backends()


def test_factory_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_llm(_settings(LLM_TYPE="claude_xyz"))


def test_mock_backend_generates_deterministic_string():
    backend = build_llm(_settings(LLM_TYPE="mock"))
    out1 = backend.generate(["q", ""], target_cell=1, max_new_tokens=8)
    out2 = backend.generate(["q", ""], target_cell=1, max_new_tokens=8)
    assert isinstance(out1, str) and out1
    assert out1 != out2  # internal counter advances


def test_mock_backend_drops_into_sotenv():
    from shapley_dreamer.envs.sot import SoTEnv

    class _Task:
        def sample(self):
            return ("q", "a")

        def eval(self, answer, gt):
            return 1.0 if answer == gt else 0.0

    backend = build_llm(_settings(LLM_TYPE="mock"))
    env = SoTEnv(K=2, N=8, T_max=2, llm=backend, task_gen=_Task())
    env.reset()
    cells, _, _, _ = env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    assert cells[1].startswith("mock[")