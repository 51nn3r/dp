from __future__ import annotations

import re


_INT_RE = re.compile(r"-?\d+")


def _parses_int(s: str) -> bool:
    return _INT_RE.search(s or "") is not None


def _is_substantive(s: str) -> bool:
    s = (s or "").strip()
    return bool(s) and s != "?"


def summarize(trajs: list[dict]) -> dict:
    """Return a dict of headline metrics for a list of rollout records.

    Each record must have: cells_history, actions, rewards, gt, success,
    optionally elapsed_s.
    """
    n = len(trajs)
    if n == 0:
        return {"n_episodes": 0}

    K = len(trajs[0]["cells_history"][0])
    answer_idx = K - 1

    successes = sum(1 for t in trajs if t["success"] >= 1.0)
    answers = [t["cells_history"][-1][answer_idx] for t in trajs]
    non_empty = sum(1 for a in answers if _is_substantive(a))
    parses_int = sum(1 for a in answers if _parses_int(a))
    mean_answer_chars = sum(len(a) for a in answers) / n

    action_counts = [0] * K
    step_lengths: list[int] = []
    for t in trajs:
        for i, (_, idx) in enumerate(t["actions"]):
            action_counts[idx] += 1
            after = t["cells_history"][i + 1][idx]
            step_lengths.append(len(after))
    total_actions = sum(action_counts)
    action_dist = (
        [c / total_actions for c in action_counts] if total_actions else [0.0] * K
    )
    mean_chars_per_step = (
        sum(step_lengths) / len(step_lengths) if step_lengths else 0.0
    )

    elapsed = [t["elapsed_s"] for t in trajs if "elapsed_s" in t]
    mean_elapsed = sum(elapsed) / len(elapsed) if elapsed else None
    total_elapsed = sum(elapsed) if elapsed else None

    return {
        "n_episodes": n,
        "K": K,
        "success_rate": successes / n,
        "successes": successes,
        "non_empty_answer_rate": non_empty / n,
        "answer_parses_int_rate": parses_int / n,
        "mean_answer_chars": mean_answer_chars,
        "mean_chars_per_step": mean_chars_per_step,
        "action_dist": action_dist,
        "mean_elapsed_s": mean_elapsed,
        "total_elapsed_s": total_elapsed,
    }


def format_summary(s: dict) -> str:
    if s.get("n_episodes", 0) == 0:
        return "=== Rollout summary ===\n  no episodes"

    rows: list[tuple[str, str]] = [
        ("episodes", f"{s['n_episodes']}"),
        ("success_rate", f"{s['success_rate']:.3f} ({s['successes']}/{s['n_episodes']})"),
        ("non_empty_answer_rate", f"{s['non_empty_answer_rate']:.3f}"),
        ("answer_parses_int_rate", f"{s['answer_parses_int_rate']:.3f}"),
        ("mean answer length (chars)", f"{s['mean_answer_chars']:.1f}"),
        ("mean output per step (chars)", f"{s['mean_chars_per_step']:.1f}"),
        ("action distribution", str([round(x, 3) for x in s["action_dist"]])),
    ]
    if s["mean_elapsed_s"] is not None:
        rows.append(("mean wall-time per episode (s)", f"{s['mean_elapsed_s']:.2f}"))
        rows.append(("total wall-time (s)", f"{s['total_elapsed_s']:.1f}"))

    width = max(len(k) for k, _ in rows)
    lines = ["=== Rollout summary ==="]
    lines.extend(f"  {k:<{width}}  {v}" for k, v in rows)
    return "\n".join(lines)