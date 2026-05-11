from shapley_dreamer import settings


def test_experiment_fields_are_ints():
    assert isinstance(settings.K, int)
    assert isinstance(settings.T_MAX, int)
    assert isinstance(settings.N, int)
    assert isinstance(settings.RANDOM_SEED, int)


def test_llm_fields_have_expected_types():
    assert settings.LLM_TYPE in {"hf", "openai", "gemini", "mock"}
    assert isinstance(settings.LLM_MAX_NEW_TOKENS, int)
    assert isinstance(settings.HF_MODEL_NAME, str)
    assert isinstance(settings.HF_DEVICE, str)


def test_optional_api_keys_default_to_none():
    # При пустом или отсутствующем ключе в .env значение должно быть None,
    # не пустая строка — иначе бэкенды решат что ключ задан.
    if not settings.OPENAI_API_KEY:
        assert settings.OPENAI_API_KEY is None
    if not settings.GEMINI_API_KEY:
        assert settings.GEMINI_API_KEY is None


def test_task_fields():
    assert isinstance(settings.TASK_TYPE, str)
    assert isinstance(settings.TASK_NUM_OPERANDS, int)
    assert isinstance(settings.TASK_MAX_VALUE, int)
    assert isinstance(settings.TASK_OPS, str)
    assert settings.TASK_NUM_OPERANDS >= 2


def test_shapley_fields():
    assert settings.SHAPLEY_ALGORITHM in {"perm_mc", "exact"}
    assert isinstance(settings.SHAPLEY_M, int)
    assert isinstance(settings.SHAPLEY_STEP_PENALTY, float)


def test_training_fields_are_ints():
    assert isinstance(settings.TRAINING_BUFFER_SIZE, int)
    assert isinstance(settings.TRAINING_BATCH_SIZE, int)
    assert isinstance(settings.TRAINING_TOTAL_ENV_STEPS, int)
