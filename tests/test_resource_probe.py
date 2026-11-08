from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
from reasonforge.data import split_records
from reasonforge.utils import digest, load_config

spec = importlib.util.spec_from_file_location("resource_probe", Path(__file__).parents[1] / "scripts/resource_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_probe_preserves_full_resource_settings_and_original_config():
    full = load_config(Path(__file__).parents[1] / "configs/full.yaml")
    original = deepcopy(full)
    config = probe.probe_config(full)
    assert full == original
    for section in ("sft", "grpo"):
        for key in full[section].keys() - {"max_steps", "save_steps"}:
            assert config[section][key] == full[section][key]
    for key in ("model", "seed", "device", "lora", "evaluation"):
        assert config[key] == full[key]
    assert config["name"] != "full" and config["grpo"]["max_steps"] == 8


def test_length_selection_includes_extremes_without_duplicate_ids():
    rows = [{"question_id": str(i)} for i in range(12)]
    chosen = probe.choose_probe_rows(rows, list(range(12)), list(reversed(range(12))), 8, 9)
    assert len(chosen) == len({r["question_id"] for r in chosen}) == 8
    assert rows[9] in chosen and rows[0] in chosen


def training_fixture():
    raw = [{"question": f"q{i}", "answer": "reason\n#### 3"} for i in range(8)]
    splits, _ = split_records(raw, [{"question": "test", "answer": "#### 4"}], validation_size=2)
    manifest = {"revision": "pin", "seed": 42, "splits": {s: {
        "ids": [r["question_id"] for r in rows], "hash": digest(rows), "total_count": len(rows)}
        for s, rows in splits.items() if s != "test"}}
    return raw, manifest, {"seed": 42, "data": {"revision": "pin", "validation_size": 2}}


def test_probe_reconstructs_exact_recorded_splits_without_test_rows():
    raw, manifest, config = training_fixture()
    splits = probe.recover_training_splits(raw, manifest, config)
    assert set(splits) == {"train", "validation"}
    assert all(row["source"] == "official_train" for rows in splits.values() for row in rows)
    raw[0]["answer"] = "changed\n#### 5"
    with pytest.raises(ValueError, match="hash differs"):
        probe.recover_training_splits(raw, manifest, config)


def test_probe_rejects_test_membership_before_reading_it():
    raw, manifest, config = training_fixture()
    manifest["splits"]["train"]["ids"][0] = "official_test:999999"
    with pytest.raises(ValueError, match="official training data only"):
        probe.recover_training_splits(raw, manifest, config)
