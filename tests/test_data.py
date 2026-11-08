import pytest
from reasonforge.data import split_records, training_rows, select_subset
from reasonforge.prompts import grpo_record
from reasonforge.utils import write_jsonl


def fixtures():
    return [{"question": f"Training question {i}?", "answer": f"step\n#### {i}"} for i in range(12)], [
        {"question": f"Testing question {i}?", "answer": f"step\n#### {i}"} for i in range(4)]


def test_deterministic_disjoint_split():
    train, test = fixtures()
    a, audit = split_records(train, test, 42, 3)
    b, _ = split_records(train, test, 42, 3)
    c, _ = split_records(train, test, 43, 3)
    assert a == b and a["train"] != c["train"]
    assert [len(a[s]) for s in ("train", "validation", "test")] == [9, 3, 4]
    hashes = {s: {r["question_hash"] for r in rows} for s, rows in a.items()}
    assert not hashes["train"] & hashes["validation"]
    assert not (hashes["train"] | hashes["validation"]) & hashes["test"]
    assert not any(audit.values())


def test_duplicates_and_overlap_removed_from_train():
    train, test = fixtures()
    train += [dict(train[0]), dict(test[0])]
    splits, audit = split_records(train, test, 42, 3)
    assert len(audit["train_duplicates"]) == 1
    assert len(audit["removed_train_test_overlap"]) == 1
    assert len(splits["test"]) == 4


def test_train_boundary(tmp_path):
    train, test = fixtures()
    splits, _ = split_records(train, test, 42, 3)
    write_jsonl(tmp_path / "data/train.jsonl", splits["test"])
    with pytest.raises(ValueError, match="provenance"):
        training_rows(tmp_path)
    with pytest.raises(ValueError, match="Test"):
        training_rows(tmp_path, "test")
    with pytest.raises(ValueError):
        grpo_record(splits["test"][0], None)


def test_bad_data_and_counts():
    train, test = fixtures()
    with pytest.raises(ValueError):
        split_records(train, test, validation_size=12)
    with pytest.raises(ValueError):
        split_records([{"question": "?", "answer": "no label"}] + train, test, validation_size=2)
    with pytest.raises(ValueError):
        select_subset(train, 20)
    assert select_subset(train, None) == train
