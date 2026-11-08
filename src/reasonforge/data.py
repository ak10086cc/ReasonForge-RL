"""Fixed split membership, deduplication and training/test boundaries."""
import random
import re
from pathlib import Path
from .metrics import extract_answer
from .utils import digest, write_json, write_jsonl, read_jsonl


def question_hash(question):
    return digest(re.sub(r"\s+", " ", question).strip().casefold())


def split_records(train, test, seed=42, validation_size=128):
    def convert(rows, source):
        result, seen, duplicates = [], set(), []
        for index, row in enumerate(rows):
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError("Missing/empty question")
            if not isinstance(row.get("answer"), str):
                raise ValueError("Missing answer")
            qh = question_hash(row["question"])
            gold = extract_answer(row["answer"], allow_gold=True)
            if gold is None:
                raise ValueError(f"Invalid answer in {source}:{index}")
            if qh in seen:
                duplicates.append(f"{source}:{index}")
                continue
            seen.add(qh)
            result.append({**row, "question_id": f"{source}:{index}", "question_hash": qh,
                           "ground_truth": gold, "source": source})
        return result, duplicates

    train, td = convert(train, "official_train")
    test, vd = convert(test, "official_test")
    test_hashes = {r["question_hash"] for r in test}
    overlap = [r["question_id"] for r in train if r["question_hash"] in test_hashes]
    # Never move test rows into training; remove any overlap from official train.
    train = [r for r in train if r["question_hash"] not in test_hashes]
    if not 0 < validation_size < len(train):
        raise ValueError("validation_size must leave a nonempty training split")
    random.Random(seed).shuffle(train)
    splits = {"validation": train[:validation_size], "train": train[validation_size:], "test": test}
    for split, rows in splits.items():
        for row in rows:
            row["split"] = split
    return splits, {"train_duplicates": td, "test_duplicates": vd, "removed_train_test_overlap": overlap}


def select_subset(rows, n):
    if n is None:
        return rows
    if not 0 < n <= len(rows):
        raise ValueError(f"Requested {n} samples from split of size {len(rows)}")
    return rows[:n]


def prepare(config, run):
    from datasets import load_dataset
    d = config["data"]
    raw = load_dataset("openai/gsm8k", "main", revision=d["revision"])
    # Datasets may fall back to the newest cache while offline. Reject a cache
    # from a different revision instead of silently labelling it as the pin.
    for split in ("train", "test"):
        cache_files = raw[split].cache_files
        if not cache_files or any(d["revision"] not in Path(f["filename"]).parts for f in cache_files):
            raise RuntimeError("Cannot verify pinned GSM8K revision from actual Arrow cache files")
    splits, audit = split_records(raw["train"], raw["test"], config["seed"], d["validation_size"])
    manifest = {"dataset": "openai/gsm8k", "subset": "main", "revision": d["revision"],
                "seed": config["seed"], "audit": audit,
                "dataset_fingerprints": {s: raw[s]._fingerprint for s in ("train", "test")},
                "actual_cache_revision_verified": True, "splits": {}}
    for split, rows in splits.items():
        selected = select_subset(rows, d[f"{split}_samples"])
        write_jsonl(Path(run) / "data" / f"{split}.jsonl", selected)
        manifest["splits"][split] = {"total_count": len(rows), "selected_count": len(selected),
                                    "ids": [r["question_id"] for r in rows],
                                    "selected_ids": [r["question_id"] for r in selected],
                                    "hash": digest(rows), "selected_hash": digest(selected)}
    write_json(Path(run) / "data" / "manifest.json", manifest)
    return manifest


def training_rows(run, split="train"):
    if split not in ("train", "validation"):
        raise ValueError("Test data cannot enter training or training validation")
    rows = read_jsonl(Path(run) / "data" / f"{split}.jsonl")
    if any(r["split"] != split or r["source"] != "official_train" for r in rows):
        raise ValueError("Training data provenance violation")
    return rows
