"""Conservative, exact numeric parsing; no intermediate-number fallback."""
import re
from decimal import Decimal, InvalidOperation

NUMBER = r"[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)%?"
FINAL = re.compile(rf"^Final Answer:\s*({NUMBER})\s*$", re.IGNORECASE)
GOLD = re.compile(rf"^####\s*({NUMBER})\s*$")


def normalize_number(value):
    text = str(value).strip().replace("−", "-")
    if not re.fullmatch(NUMBER, text):
        return None
    # Percent signs denote percentage points, matching GSM8K scalar labels.
    try:
        number = Decimal(text.rstrip("%").replace(",", ""))
        if not number:
            return "0"
        # Decimal.normalize() applies the active context precision and can round
        # very long integers. Fixed-point formatting preserves every digit.
        result = format(number, "f")
        return result.rstrip("0").rstrip(".") if "." in result else result
    except InvalidOperation:
        return None


def extract_answer(text, *, allow_gold=False):
    lines = str(text).strip().splitlines()
    if not lines:
        return None
    match = FINAL.fullmatch(lines[-1].strip())
    if match is None and allow_gold:
        match = GOLD.fullmatch(lines[-1].strip())
    return normalize_number(match.group(1)) if match else None


def summarize(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty split")
    n = len(rows)
    return {
        "n": n,
        "correct": sum(bool(r["correct"]) for r in rows),
        "accuracy": sum(bool(r["correct"]) for r in rows) / n,
        "parse_rate": sum(bool(r["parse_success"]) for r in rows) / n,
        "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / n,
    }
