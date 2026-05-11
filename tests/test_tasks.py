from types import SimpleNamespace

import pytest

from shapley_dreamer.tasks.base import (
    TaskGenerator,
    _REGISTRY,
    get_task,
    register,
    registered_tasks,
)
from shapley_dreamer.tasks.factory import build_task
from shapley_dreamer.tasks.word_problem import WordProblemGenerator


def _settings(**overrides):
    base = dict(
        TASK_TYPE="word_problem",
        RANDOM_SEED=0,
        TASK_NUM_OPERANDS=3,
        TASK_MAX_VALUE=9,
        TASK_OPS="+,-",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        TaskGenerator()  # type: ignore[abstract]


def test_register_rejects_duplicates():
    class _Dummy(TaskGenerator):
        def sample(self):
            return ("", "")

        def eval(self, answer, gt):
            return 0.0

        @classmethod
        def from_settings(cls, settings):
            return cls()

    register("dup_task_x")(_Dummy)
    try:
        with pytest.raises(ValueError):
            register("dup_task_x")(_Dummy)
    finally:
        _REGISTRY.pop("dup_task_x", None)


def test_get_task_unknown_raises():
    with pytest.raises(KeyError):
        get_task("definitely_not_registered")


def test_factory_builds_word_problem():
    task = build_task(_settings())
    assert isinstance(task, WordProblemGenerator)
    assert "word_problem" in registered_tasks()


def test_factory_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_task(_settings(TASK_TYPE="riddles"))


def test_word_problem_deterministic_with_seed():
    a = WordProblemGenerator(seed=42).sample()
    b = WordProblemGenerator(seed=42).sample()
    c = WordProblemGenerator(seed=43).sample()
    assert a == b
    assert a != c


def test_word_problem_question_format_and_gt_correct():
    task = WordProblemGenerator(seed=0, num_operands=3, max_value=9)
    for _ in range(20):
        q, gt = task.sample()
        assert q.endswith(" = ?")
        # gt must equal the result of evaluating the LHS expression
        lhs = q[: -len(" = ?")]
        assert int(gt) == eval(lhs)  # noqa: S307 - safe; only digits/+/-/spaces


def test_word_problem_rejects_bad_args():
    with pytest.raises(ValueError):
        WordProblemGenerator(num_operands=1)
    with pytest.raises(ValueError):
        WordProblemGenerator(max_value=0)
    with pytest.raises(ValueError):
        WordProblemGenerator(ops=("/",))
    with pytest.raises(ValueError):
        WordProblemGenerator(ops=())


def test_word_problem_supports_multiplication():
    task = WordProblemGenerator(seed=0, num_operands=4, max_value=20, ops=("+", "-", "*"))
    seen_ops = set()
    for _ in range(40):
        q, gt = task.sample()
        lhs = q[: -len(" = ?")]
        assert int(gt) == eval(lhs)  # noqa: S307
        for op in ("+", "-", "*"):
            if f" {op} " in lhs:
                seen_ops.add(op)
    assert "*" in seen_ops


def test_word_problem_factory_parses_ops():
    task = build_task(_settings(TASK_OPS="+,-,*"))
    assert task.ops == ("+", "-", "*")


def test_word_problem_uses_parens_when_complex_enough():
    task = WordProblemGenerator(seed=0, num_operands=6, max_value=9, ops=("+", "-", "*"))
    seen_paren = False
    for _ in range(40):
        q, _ = task.sample()
        if "(" in q:
            seen_paren = True
            break
    assert seen_paren


def test_word_problem_n_leaves_varies():
    # Count of operators ≈ leaves - 1; should vary across samples.
    task = WordProblemGenerator(seed=0, num_operands=8, max_value=9, ops=("+", "-"))
    op_counts = set()
    for _ in range(40):
        q, _ = task.sample()
        lhs = q[: -len(" = ?")]
        op_counts.add(sum(lhs.count(o) for o in ("+", "-", "*")))
    assert len(op_counts) >= 2  # we observe multiple distinct lengths


def test_word_problem_expression_contains_only_allowed_chars():
    task = WordProblemGenerator(seed=0, num_operands=6, max_value=99, ops=("+", "-", "*"))
    allowed = set("0123456789+-* ()=?-")
    for _ in range(20):
        q, _ = task.sample()
        assert set(q) <= allowed, f"Disallowed char in {q!r}"


def test_word_problem_eval_extracts_first_int_from_answer():
    task = WordProblemGenerator(seed=0)
    assert task.eval("8", "8") == 1.0
    assert task.eval("the answer is 8", "8") == 1.0
    assert task.eval("8 because 3+5", "8") == 1.0
    assert task.eval("9", "8") == 0.0
    assert task.eval("no number here", "8") == 0.0
    assert task.eval("-3 yes", "-3") == 1.0


def test_word_problem_drops_into_sotenv():
    from shapley_dreamer.envs.sot import SoTEnv
    from shapley_dreamer.llm.factory import build_llm

    llm_settings = SimpleNamespace(LLM_TYPE="mock", LLM_MAX_NEW_TOKENS=8, RANDOM_SEED=0)
    llm = build_llm(llm_settings)
    task = build_task(_settings())

    env = SoTEnv(K=2, N=8, T_max=2, llm=llm, task_gen=task)
    cells = env.reset()
    assert cells[0].endswith(" = ?")
    _, reward, done, _ = env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    _, reward, done, _ = env.step((SoTEnv.ACTION_LLM_WRITE, 1))
    assert done
    assert reward in (0.0, 1.0)  # mock backend won't actually solve arithmetic