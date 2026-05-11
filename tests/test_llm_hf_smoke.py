from types import SimpleNamespace

import pytest


pytest.importorskip("transformers")
pytest.importorskip("torch")


def _settings():
    return SimpleNamespace(
        LLM_TYPE="hf",
        LLM_MAX_NEW_TOKENS=4,
        RANDOM_SEED=0,
        HF_MODEL_NAME="sshleifer/tiny-gpt2",
        HF_DEVICE="cpu",
    )


def test_hf_backend_loads_and_generates_string():
    from shapley_dreamer.llm.factory import build_llm

    backend = build_llm(_settings())
    out = backend.generate(["3 + 5 = ?", ""], target_cell=1, max_new_tokens=4)
    assert isinstance(out, str)


def test_hf_resolve_device_falls_back_when_no_cuda():
    import torch

    from shapley_dreamer.llm.hf import _resolve_device

    if torch.cuda.is_available():
        pytest.skip("CUDA available; cannot test fallback path here.")
    with pytest.warns(UserWarning):
        assert _resolve_device("cuda") == "cpu"
    assert _resolve_device("cpu") == "cpu"