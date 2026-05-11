from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch
from torch import optim

from shapley_dreamer.agent.actor_critic import ActorCritic
from shapley_dreamer.agent.imagination import imagine_rollout
from shapley_dreamer.envs.sot import SoTEnv
from shapley_dreamer.shapley.base import ShapleyEstimator
from shapley_dreamer.shapley.exact import ExactShapley
from shapley_dreamer.shapley.imagined import compute_phi_imagined
from shapley_dreamer.world_model.encoder import WMEncoder
from shapley_dreamer.world_model.model import WorldModel
from shapley_dreamer.world_model.training import train_wm


@dataclass
class StepStats:
    actor_loss: float
    critic_loss: float
    imagined_terminal: float


def imagined_train_step(
    wm: WorldModel,
    ac: ActorCritic,
    optimiser: optim.Optimizer,
    initial_cells: list[str],
    T_max: int,
    reward_mode: str,
    estimator: ShapleyEstimator | None = None,
    critic_weight: float = 0.5,
) -> StepStats:
    roll = imagine_rollout(wm, ac, initial_cells, T_max)

    if reward_mode == "terminal":
        # Sparse reward at terminal: r_t = 0 for t < T-1, r_{T-1} = R.
        # Yields G_t = R for all t (every step shares the same return).
        zero = torch.zeros_like(roll.terminal_reward)
        per_step = [zero] * (T_max - 1) + [roll.terminal_reward.detach()]
    elif reward_mode == "shapley":
        est = estimator or ExactShapley()
        phi = compute_phi_imagined(wm, initial_cells, roll.actions, est)
        per_step = [
            torch.tensor(float(p), dtype=roll.terminal_reward.dtype,
                         device=roll.terminal_reward.device)
            for p in phi
        ]
    else:
        raise ValueError(f"reward_mode must be 'terminal' or 'shapley', got {reward_mode!r}")

    returns: list[torch.Tensor] = []
    G = torch.zeros_like(roll.terminal_reward)
    for r in reversed(per_step):
        G = r + G
        returns.insert(0, G)
    returns_t = torch.stack(returns).detach()

    values = torch.stack([ac.value(s) for s in roll.states[:-1]])
    advantages = returns_t - values.detach()
    log_probs = torch.stack(roll.log_probs)

    actor_loss = -(log_probs * advantages).mean()
    critic_loss = ((values - returns_t) ** 2).mean()
    loss = actor_loss + critic_weight * critic_loss

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return StepStats(
        actor_loss=float(actor_loss.item()),
        critic_loss=float(critic_loss.item()),
        imagined_terminal=float(roll.terminal_reward.item()),
    )


def evaluate_on_env(
    ac: ActorCritic,
    wm: WorldModel,
    env: SoTEnv,
    n_episodes: int,
) -> float:
    successes = 0
    for _ in range(n_episodes):
        cells = env.reset()
        with torch.no_grad():
            state = wm.encode(cells, step=0)
        done = False
        reward = 0.0
        while not done:
            with torch.no_grad():
                idx = int(torch.argmax(ac.logits(state)).item())
            cells, reward, done, _ = env.step((SoTEnv.ACTION_LLM_WRITE, idx))
            with torch.no_grad():
                state = wm.dynamics_step(state, idx)
        if reward >= 1.0:
            successes += 1
    return successes / max(1, n_episodes)


def collect_random_episodes(
    env: SoTEnv,
    n_episodes: int,
    rng: random.Random,
) -> list[tuple[list[str], list[int], float]]:
    out = []
    for _ in range(n_episodes):
        cells = env.reset()
        initial = list(cells)
        actions: list[int] = []
        done = False
        terminal_r = 0.0
        while not done:
            idx = rng.randrange(env.K)
            actions.append(idx)
            cells, terminal_r, done, _ = env.step((SoTEnv.ACTION_LLM_WRITE, idx))
        out.append((initial, actions, float(terminal_r)))
    return out


def train_actor_critic(
    *,
    reward_mode: str,
    wm: WorldModel,
    env: SoTEnv,
    task,
    K: int,
    T_max: int,
    d_model: int,
    nhead: int,
    ac_updates: int,
    eval_every: int,
    eval_episodes: int,
    device: torch.device,
    lr: float = 1e-3,
    log_prefix: str = "",
) -> list[dict]:
    ac = ActorCritic(K=K, T_max=T_max, d_model=d_model, nhead=nhead, num_layers=1).to(device)
    opt = optim.Adam(ac.parameters(), lr=lr)
    curve: list[dict] = []
    for step in range(ac_updates):
        q, _ = task.sample()
        initial_cells = [q] + [""] * (K - 1)
        imagined_train_step(wm, ac, opt, initial_cells, T_max, reward_mode)
        if (step + 1) % eval_every == 0:
            success = evaluate_on_env(ac, wm, env, eval_episodes)
            curve.append({"step": step + 1, "success_rate": success})
            print(f"{log_prefix}[{reward_mode} {step + 1}/{ac_updates}] success={success:.3f}")
    return curve


def run_comparison(
    *,
    settings_module,
    n_collect: int = 200,
    wm_epochs: int = 50,
    ac_updates: int = 200,
    eval_every: int = 20,
    eval_episodes: int = 20,
    d_model: int = 64,
    nhead: int = 4,
) -> dict:
    """Build env+WM+AC, collect, train WM, train AC twice (terminal vs shapley).

    Returns dict with `settings`, `wm_loss_curve`, `curves` (per-mode list of
    {step, success_rate}). LLM, task, env are constructed from settings_module.
    """
    from shapley_dreamer.llm.factory import build_llm
    from shapley_dreamer.tasks.factory import build_task

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K, T_max, N = settings_module.K, settings_module.T_MAX, settings_module.N

    llm = build_llm(settings_module)
    task = build_task(settings_module)
    env = SoTEnv(K=K, N=N, T_max=T_max, llm=llm, task_gen=task)

    print(f"device={device} K={K} T_MAX={T_max} model={settings_module.HF_MODEL_NAME}")
    print(f"\nCollecting {n_collect} random episodes...")
    t0 = time.perf_counter()
    rng = random.Random(settings_module.RANDOM_SEED)
    dataset = collect_random_episodes(env, n_collect, rng)
    rand_success = sum(1 for _, _, r in dataset if r >= 1.0) / max(1, n_collect)
    print(f"  Done in {time.perf_counter() - t0:.1f}s; random_success={rand_success:.3f}")

    print(f"\nBuilding WM (d_model={d_model}, nhead={nhead})...")
    encoder = WMEncoder(
        tokenizer=llm.tokenizer,
        token_embedding=llm.model.get_input_embeddings(),
        K=K, T_max=T_max,
        d_model=d_model, nhead=nhead, num_layers=1,
    ).to(device)
    wm = WorldModel(encoder, K=K).to(device)

    print(f"\nTraining WM ({wm_epochs} epochs)...")
    t0 = time.perf_counter()
    losses = train_wm(wm, dataset, epochs=wm_epochs, lr=1e-3, log_every=10)
    print(f"  Done in {time.perf_counter() - t0:.1f}s; final mse={losses[-1]:.4f}")

    curves: dict[str, list[dict]] = {}
    for mode in ("terminal", "shapley"):
        print(f"\nTraining AC mode={mode} ({ac_updates} updates)...")
        t0 = time.perf_counter()
        curves[mode] = train_actor_critic(
            reward_mode=mode, wm=wm, env=env, task=task,
            K=K, T_max=T_max, d_model=d_model, nhead=nhead,
            ac_updates=ac_updates, eval_every=eval_every,
            eval_episodes=eval_episodes, device=device,
        )
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

    return {
        "settings": {
            "K": K, "T_max": T_max, "N": N,
            "model": settings_module.HF_MODEL_NAME,
            "task_num_operands": settings_module.TASK_NUM_OPERANDS,
            "task_max_value": settings_module.TASK_MAX_VALUE,
            "task_ops": settings_module.TASK_OPS,
            "n_collect": n_collect,
            "wm_epochs": wm_epochs,
            "ac_updates": ac_updates,
            "random_baseline_success": rand_success,
        },
        "wm_loss_curve": losses,
        "curves": curves,
    }