from __future__ import annotations


def _format_notes(cells: list[str]) -> str:
    K = len(cells)
    notes = [c.strip() for c in cells[1:K - 1] if c.strip()]
    return "\n".join(f"- {n}" for n in notes) if notes else "(none yet)"


def messages_generate(cells: list[str]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You solve simple arithmetic word problems. "
                "When asked for the final answer, output ONLY the number, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {cells[0]}\n"
                f"Working notes:\n{_format_notes(cells)}\n\n"
                "Final answer (number only):"
            ),
        },
    ]


def messages_think(cells: list[str]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You solve arithmetic word problems step by step. "
                "Write ONE concise intermediate computation that helps reach the final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {cells[0]}\n"
                f"Working notes so far:\n{_format_notes(cells)}\n\n"
                "Next step (one short line):"
            ),
        },
    ]


def flatten_messages(messages: list[dict]) -> str:
    """Plain-text fallback for tokenizers without a chat template."""
    return "\n\n".join(m["content"] for m in messages) + "\n\n"