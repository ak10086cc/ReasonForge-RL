"""Separate correctness and strict final-line format rewards for TRL."""
from .metrics import extract_answer, normalize_number


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    return "".join(m.get("content", "") for m in completion)


def correctness_reward(completions, ground_truth, **kwargs):
    if len(completions) != len(ground_truth):
        raise ValueError("Completion/label count mismatch")
    return [float((p := extract_answer(completion_text(c))) is not None
                  and p == normalize_number(g)) for c, g in zip(completions, ground_truth)]


def format_reward(completions, **kwargs):
    return [float(extract_answer(completion_text(c)) is not None) for c in completions]
