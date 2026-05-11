import pytest

from shapley_dreamer.envs.sot import SoTEnv


class _ScriptedLLM:
    """LLM, выдающий заранее заготовленную последовательность строк."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, cells, target_cell, max_new_tokens):
        self.calls.append((list(cells), target_cell))
        return self.outputs.pop(0)


class _FixedTask:
    def __init__(self, question, gt):
        self.question = question
        self.gt = gt

    def sample(self):
        return (self.question, self.gt)

    def eval(self, answer, gt):
        return 1.0 if answer.strip() == gt.strip() else 0.0


def test_reset_initialises_input_cell():
    env = SoTEnv(K=2, N=16, T_max=4, llm=_ScriptedLLM([]), task_gen=_FixedTask("q", "a"))
    cells = env.reset()
    assert cells == ["q", ""]
    assert env.t == 0


def test_step_writes_into_target_cell_and_advances_clock():
    llm = _ScriptedLLM(["x"])
    env = SoTEnv(K=2, N=16, T_max=4, llm=llm, task_gen=_FixedTask("q", "a"))
    env.reset()

    cells, reward, done, info = env.step((SoTEnv.ACTION_LLM_WRITE, 1))

    assert cells[1] == "x"
    assert reward == 0.0
    assert not done
    assert info["t"] == 1


def test_terminal_reward_matches_gt_in_answer_cell():
    llm = _ScriptedLLM(["a", "b", "c", "8"])
    env = SoTEnv(K=2, N=16, T_max=4, llm=llm, task_gen=_FixedTask("3 + 5 = ?", "8"))
    env.reset()
    rewards = []
    for _ in range(4):
        _, r, done, _ = env.step((SoTEnv.ACTION_LLM_WRITE, 1))
        rewards.append(r)
    assert rewards[:-1] == [0.0, 0.0, 0.0]
    assert rewards[-1] == 1.0
    assert done


def test_overwrite_keeps_only_last_value():
    llm = _ScriptedLLM(["x", "y", "z", "answer"])
    env = SoTEnv(K=2, N=16, T_max=4, llm=llm, task_gen=_FixedTask("q", "answer"))
    env.reset()
    for _ in range(4):
        cells, _, _, _ = env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    assert cells[1] == "answer"


def test_invalid_action_code_raises():
    env = SoTEnv(K=2, N=16, T_max=4, llm=_ScriptedLLM([]), task_gen=_FixedTask("q", "a"))
    env.reset()
    with pytest.raises(ValueError):
        env.step((2, 0))


def test_invalid_cell_index_raises():
    env = SoTEnv(K=2, N=16, T_max=4, llm=_ScriptedLLM([]), task_gen=_FixedTask("q", "a"))
    env.reset()
    with pytest.raises(ValueError):
        env.step((SoTEnv.ACTION_LLM_WRITE, 5))


def test_K_below_two_rejected():
    with pytest.raises(ValueError):
        SoTEnv(K=1, N=16, T_max=4, llm=_ScriptedLLM([]), task_gen=_FixedTask("q", "a"))


def test_step_before_reset_raises():
    env = SoTEnv(K=2, N=16, T_max=4, llm=_ScriptedLLM([]), task_gen=_FixedTask("q", "a"))
    with pytest.raises(RuntimeError):
        env.step((SoTEnv.ACTION_LLM_WRITE, 1))


def test_step_after_done_raises():
    llm = _ScriptedLLM(["x", "y"])
    env = SoTEnv(K=2, N=16, T_max=2, llm=llm, task_gen=_FixedTask("q", "a"))
    env.reset()
    env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    _, _, done, _ = env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    assert done
    with pytest.raises(RuntimeError):
        env.step((SoTEnv.ACTION_LLM_WRITE, 1))


def test_reset_after_done_resumes():
    llm = _ScriptedLLM(["x", "y", "z"])
    env = SoTEnv(K=2, N=16, T_max=2, llm=llm, task_gen=_FixedTask("q", "a"))
    env.reset()
    env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    env.reset()
    cells, _, done, _ = env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    assert cells[1] == "z"
    assert not done


def test_info_carries_gt():
    llm = _ScriptedLLM(["x"])
    env = SoTEnv(K=2, N=16, T_max=2, llm=llm, task_gen=_FixedTask("q", "the_answer"))
    env.reset()
    _, _, _, info = env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    assert info["gt"] == "the_answer"
