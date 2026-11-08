"""One prompt and one native chat template for all stages."""
import re
from .metrics import extract_answer

SYSTEM = (
    "Solve the math problem. Show clear reasoning steps. "
    "End with exactly one line: Final Answer: <number>"
)


def messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def render_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(messages(question), tokenize=False, add_generation_prompt=True)


def clean_solution(answer):
    gold = extract_answer(answer, allow_gold=True)
    if gold is None:
        raise ValueError("GSM8K answer lacks a valid final #### value")
    reasoning = re.sub(r"<<[^<>]*>>", "", answer.rsplit("####", 1)[0]).strip()
    return f"{reasoning}\nFinal Answer: {gold}"


def sft_record(row, tokenizer):
    prompt = render_prompt(tokenizer, row["question"])
    completion = clean_solution(row["answer"]) + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


def grpo_record(row, tokenizer):
    if row["split"] != "train":
        raise ValueError("GRPO rewards accept training rows only")
    return {"prompt": render_prompt(tokenizer, row["question"]),
            "ground_truth": row["ground_truth"], "question_id": row["question_id"]}
