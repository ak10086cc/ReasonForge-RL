import pytest
from reasonforge.metrics import extract_answer, normalize_number, summarize


@pytest.mark.parametrize("text,value", [
    ("work 2+3=5\nFinal Answer: 5", "5"), ("Final Answer: -1,234.50", "-1234.5"),
    ("Final Answer: −0.5", "-0.5"), ("Final Answer: 25%", "25"),
    ("Final Answer: .25", "0.25"), ("Final Answer: -0.00", "0"),
    ("Final Answer: 2\nFinal Answer: 3", "3"), ("Final Answer: 1.000", "1"),
    ("We computed 42.", None), ("", None), ("Final Answer: 1,23", None),
    ("Final Answer: 2\nextra discussion 8", None), ("Final Answer: NaN", None),
    ("Final Answer: 1/2", None), ("Final Answer: 4 or 5", None),
    ("#### 42", None), ("work\nFinal Answer: 2 apples", None),
])
def test_parser(text, value):
    assert extract_answer(text) == value


def test_gold_and_decimal_exactness():
    assert extract_answer("work 5\n#### 1,234", allow_gold=True) == "1234"
    assert normalize_number("123456789123456789.1") == "123456789123456789.1"
    assert normalize_number("1234567890123456789012345678901234567890.10") == "1234567890123456789012345678901234567890.1"
    assert normalize_number("invalid") is None


def test_metric_denominator_includes_failed_parses():
    rows = [{"correct": True, "parse_success": True, "generated_tokens": 10},
            {"correct": False, "parse_success": False, "generated_tokens": 20}]
    assert summarize(rows) == {"n": 2, "correct": 1, "accuracy": .5, "parse_rate": .5, "mean_generated_tokens": 15}
    with pytest.raises(ValueError):
        summarize([])
