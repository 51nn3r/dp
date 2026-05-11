from __future__ import annotations

import warnings

from .base import LLMBackend, register
from .prompts import flatten_messages, messages_generate, messages_think


def _resolve_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    warnings.warn("CUDA requested but unavailable; falling back to CPU.", stacklevel=2)
    return "cpu"


@register("hf")
class HuggingFaceBackend(LLMBackend):
    def __init__(
        self,
        model_name: str,
        device: str,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _resolve_device(device)
        self.max_new_tokens = max_new_tokens

        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @classmethod
    def from_settings(cls, settings) -> "HuggingFaceBackend":
        return cls(
            model_name=settings.HF_MODEL_NAME,
            device=settings.HF_DEVICE,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            seed=settings.RANDOM_SEED,
        )

    def format_prompt(self, cells: list[str], target_cell: int) -> str:
        K = len(cells)
        messages = (
            messages_generate(cells)
            if target_cell == K - 1
            else messages_think(cells)
        )
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return flatten_messages(messages)

    def generate(
        self,
        cells: list[str],
        target_cell: int,
        max_new_tokens: int,
    ) -> str:
        import torch

        prompt = self.format_prompt(cells, target_cell)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip().splitlines()[0] if text.strip() else ""