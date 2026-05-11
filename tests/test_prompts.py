from shapley_dreamer.llm.prompts import (
    flatten_messages,
    messages_generate,
    messages_think,
)


def test_generate_and_think_differ():
    cells = ["3 + 5 = ?", "", ""]
    g = messages_generate(cells)
    t = messages_think(cells)
    assert g != t
    assert g[0]["role"] == "system"
    assert t[0]["role"] == "system"


def test_generate_asks_for_number_only():
    cells = ["3 + 5 = ?", "", ""]
    g = messages_generate(cells)
    assert "number" in g[0]["content"].lower()
    assert cells[0] in g[1]["content"]


def test_think_asks_for_intermediate_step():
    cells = ["3 + 5 = ?", "", ""]
    t = messages_think(cells)
    assert "step" in t[0]["content"].lower() or "step" in t[1]["content"].lower()
    assert cells[0] in t[1]["content"]


def test_notes_show_only_non_empty_scratch():
    cells = ["q", "scratch_a", "", "scratch_b", "answer"]
    g = messages_generate(cells)
    user = g[1]["content"]
    assert "scratch_a" in user
    assert "scratch_b" in user
    assert "answer" not in user.split("Working notes:")[1].split("Final answer")[0]


def test_notes_empty_when_no_scratch():
    cells = ["q", "", "", ""]
    g = messages_generate(cells)
    assert "(none yet)" in g[1]["content"]


def test_flatten_messages_concatenates_content():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    out = flatten_messages(msgs)
    assert "S" in out and "U" in out


def test_K_2_uses_generate_path_only():
    # K=2 has no scratch cells. Generate prompt should still work.
    cells = ["q", ""]
    g = messages_generate(cells)
    assert "(none yet)" in g[1]["content"]