import pytest

from shapley_dreamer.training.metrics import format_summary, summarize


def _record(cells_history, actions, rewards, success, gt="", elapsed_s=None):
    rec = dict(
        cells_history=cells_history,
        actions=actions,
        rewards=rewards,
        gt=gt,
        success=success,
    )
    if elapsed_s is not None:
        rec["elapsed_s"] = elapsed_s
    return rec


def test_summarize_empty():
    s = summarize([])
    assert s == {"n_episodes": 0}


def test_summarize_single_success():
    rec = _record(
        cells_history=[["q", ""], ["q", "8"]],
        actions=[[1, 1]],
        rewards=[1.0],
        success=1.0,
        gt="8",
    )
    s = summarize([rec])
    assert s["n_episodes"] == 1
    assert s["success_rate"] == 1.0
    assert s["non_empty_answer_rate"] == 1.0
    assert s["answer_parses_int_rate"] == 1.0
    assert s["mean_answer_chars"] == 1.0
    assert s["action_dist"] == [0.0, 1.0]


def test_summarize_question_marks_count_as_empty():
    rec = _record(
        cells_history=[["q", ""], ["q", "?"]],
        actions=[[1, 1]],
        rewards=[0.0],
        success=0.0,
    )
    s = summarize([rec])
    assert s["non_empty_answer_rate"] == 0.0
    assert s["answer_parses_int_rate"] == 0.0


def test_summarize_action_distribution_is_uniform_for_random_policy():
    # 3 episodes, each writes once into a different cell of K=3
    recs = [
        _record([["q", "", ""], ["q", "x", ""]], [[1, 1]], [0.0], 0.0),
        _record([["q", "", ""], ["q", "", "y"]], [[1, 2]], [0.0], 0.0),
        _record([["q", "", ""], ["z", "", ""]], [[1, 0]], [0.0], 0.0),
    ]
    s = summarize(recs)
    assert s["action_dist"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_summarize_extracts_int_from_noisy_answer():
    rec = _record(
        cells_history=[["q", ""], ["q", "the answer is 42 maybe"]],
        actions=[[1, 1]],
        rewards=[0.0],
        success=0.0,
    )
    s = summarize([rec])
    assert s["answer_parses_int_rate"] == 1.0
    assert s["non_empty_answer_rate"] == 1.0


def test_summarize_aggregates_elapsed_when_present():
    recs = [
        _record([["q", ""], ["q", "1"]], [[1, 1]], [0.0], 0.0, elapsed_s=2.0),
        _record([["q", ""], ["q", "2"]], [[1, 1]], [0.0], 0.0, elapsed_s=4.0),
    ]
    s = summarize(recs)
    assert s["mean_elapsed_s"] == 3.0
    assert s["total_elapsed_s"] == 6.0


def test_summarize_omits_elapsed_when_absent():
    rec = _record([["q", ""], ["q", "1"]], [[1, 1]], [0.0], 0.0)
    s = summarize([rec])
    assert s["mean_elapsed_s"] is None
    assert s["total_elapsed_s"] is None


def test_summarize_chars_per_step_counts_only_written_cell():
    rec = _record(
        cells_history=[["q", "", ""], ["q", "abcd", ""], ["q", "abcd", "xy"]],
        actions=[[1, 1], [1, 2]],
        rewards=[0.0, 0.0],
        success=0.0,
    )
    s = summarize([rec])
    assert s["mean_chars_per_step"] == 3.0  # (4 + 2) / 2


def test_format_summary_renders_known_fields():
    rec = _record(
        cells_history=[["q", ""], ["q", "8"]],
        actions=[[1, 1]],
        rewards=[1.0],
        success=1.0,
        elapsed_s=1.5,
    )
    text = format_summary(summarize([rec]))
    assert "success_rate" in text
    assert "1.000" in text
    assert "wall-time" in text


def test_format_summary_handles_empty():
    assert "no episodes" in format_summary({"n_episodes": 0})