from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from shapley_dreamer import settings
from shapley_dreamer.envs.sot import SoTEnv
from shapley_dreamer.llm.factory import build_llm
from shapley_dreamer.tasks.factory import build_task
from shapley_dreamer.training.metrics import format_summary, summarize
from shapley_dreamer.training.rollout import collect_rollout, random_policy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="Number of rollouts to collect.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/rollouts.jsonl"),
        help="Output JSONL path (relative to code/).",
    )
    p.add_argument(
        "--policy-seed",
        type=int,
        default=settings.RANDOM_SEED,
        help="Seed for the random policy (independent of LLM/task seeds).",
    )
    return p.parse_args()


def _trajectory_to_dict(traj, episode: int, success: float, elapsed_s: float) -> dict:
    d = asdict(traj)
    d["episode"] = episode
    d["success"] = success
    d["elapsed_s"] = elapsed_s
    return d


def main() -> None:
    args = _parse_args()
    out_path = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"LLM={settings.LLM_TYPE} model={settings.HF_MODEL_NAME} device={settings.HF_DEVICE}")
    print(f"K={settings.K} T_MAX={settings.T_MAX} N={settings.N}")
    print(f"Task={settings.TASK_TYPE} operands={settings.TASK_NUM_OPERANDS} max_value={settings.TASK_MAX_VALUE}")
    print(f"Writing -> {out_path}")

    t0 = time.perf_counter()
    llm = build_llm(settings)
    task = build_task(settings)
    print(f"Built llm+task in {time.perf_counter() - t0:.1f}s")

    rng = random.Random(args.policy_seed)
    policy = random_policy(rng, K=settings.K)

    collected: list[dict] = []
    with out_path.open("a") as f:
        for ep in range(args.n):
            env = SoTEnv(
                K=settings.K,
                N=settings.N,
                T_max=settings.T_MAX,
                llm=llm,
                task_gen=task,
            )
            t_ep = time.perf_counter()
            traj = collect_rollout(env, policy)
            elapsed = time.perf_counter() - t_ep
            record = _trajectory_to_dict(traj, ep, traj.terminal_reward, elapsed)
            collected.append(record)
            f.write(json.dumps(record) + "\n")
            print(
                f"[ep {ep + 1}/{args.n}] gt={traj.gt!r} "
                f"final={traj.cells_history[-1][settings.K - 1]!r} "
                f"r={traj.terminal_reward} ({elapsed:.1f}s)"
            )

    print()
    print(format_summary(summarize(collected)))


if __name__ == "__main__":
    main()