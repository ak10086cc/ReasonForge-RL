import pytest
from reasonforge.rewards import correctness_reward, format_reward


@pytest.mark.parametrize("text,gold,correct,fmt", [
    ("", "3", 0, 0), ("3", "3", 0, 0), ("Final Answer: ???", "3", 0, 0),
    ("Final Answer: 4", "3", 0, 1), ("Final Answer: 3", "3", 1, 1),
    ("work 3\nFinal Answer: -1,234.5", "-1234.50", 1, 1),
    ("Final Answer: 3\nthis is wrong", "3", 0, 0),
])
def test_rewards(text, gold, correct, fmt):
    assert correctness_reward([text], [gold]) == [correct]
    assert format_reward([text]) == [fmt]
    assert correctness_reward([[{"role": "assistant", "content": text}]], [gold]) == [correct]


def test_label_mismatch_fails():
    with pytest.raises(ValueError):
        correctness_reward(["Final Answer: 1"], [])
