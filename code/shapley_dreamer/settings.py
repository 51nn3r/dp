from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)


def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _opt_str(name: str) -> str | None:
    val = os.getenv(name)
    return val if val else None


# === Эксперимент ===
K: int = _int("K", 2)
T_MAX: int = _int("T_MAX", 4)
N: int = _int("N", 16)
RANDOM_SEED: int = _int("RANDOM_SEED", 0)

# === LLM (общее) ===
LLM_TYPE: str = _str("LLM_TYPE", "hf")           # hf | openai | gemini | mock
LLM_MAX_NEW_TOKENS: int = _int("LLM_MAX_NEW_TOKENS", 16)

# === HuggingFace ===
HF_MODEL_NAME: str = _str("HF_MODEL_NAME", "gpt2")
HF_DEVICE: str = _str("HF_DEVICE", "cuda")

# === OpenAI ===
OPENAI_API_KEY: str | None = _opt_str("OPENAI_API_KEY")
OPENAI_MODEL: str = _str("OPENAI_MODEL", "gpt-4o-mini")

# === Gemini ===
GEMINI_API_KEY: str | None = _opt_str("GEMINI_API_KEY")
GEMINI_MODEL: str = _str("GEMINI_MODEL", "gemini-1.5-flash")

# === Задачи ===
TASK_TYPE: str = _str("TASK_TYPE", "word_problem")
TASK_NUM_OPERANDS: int = _int("TASK_NUM_OPERANDS", 3)
TASK_MAX_VALUE: int = _int("TASK_MAX_VALUE", 9)
TASK_OPS: str = _str("TASK_OPS", "+,-")  # comma-separated subset of '+,-,*'

# === Shapley ===
SHAPLEY_ALGORITHM: str = _str("SHAPLEY_ALGORITHM", "perm_mc")  # perm_mc | exact
SHAPLEY_M: int = _int("SHAPLEY_M", 32)
SHAPLEY_STEP_PENALTY: float = _float("SHAPLEY_STEP_PENALTY", 0.05)

# === Тренировка ===
TRAINING_BUFFER_SIZE: int = _int("TRAINING_BUFFER_SIZE", 1024)
TRAINING_BATCH_SIZE: int = _int("TRAINING_BATCH_SIZE", 16)
TRAINING_TOTAL_ENV_STEPS: int = _int("TRAINING_TOTAL_ENV_STEPS", 10000)