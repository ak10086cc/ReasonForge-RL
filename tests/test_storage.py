import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from reasonforge import storage


def test_disk_and_run_limits_preserve_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))
    result = storage.prepare_storage_policy(tmp_path)
    assert result["max_run_bytes"] == 5 * 2**30
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=49 * 2**30))
    with pytest.raises(storage.StorageLimitError):
        storage.check_storage(tmp_path)
    assert evidence.read_text(encoding="utf-8") == "keep"
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))
    policy_path = tmp_path / "storage_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["max_run_bytes"] = 1
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(storage.StorageLimitError):
        storage.check_storage(tmp_path)


def test_policy_cannot_silently_change(tmp_path):
    storage.prepare_storage_policy(tmp_path, min_free_gib=0.0001)
    with pytest.raises(ValueError, match="differs"):
        storage.prepare_storage_policy(tmp_path, max_run_gib=6, min_free_gib=0.0001)
    assert storage.check_storage(tmp_path / "legacy") is None


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_storage_limits(tmp_path, value):
    with pytest.raises(ValueError):
        storage.prepare_storage_policy(tmp_path, max_run_gib=value)
    assert not (tmp_path / "storage_policy.json").exists()


def test_directory_counts_nested_files_and_missing_path(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "payload").write_bytes(b"12345")
    assert storage.directory_bytes(tmp_path) == 5
    assert storage.directory_bytes(tmp_path / "absent") == 0


def test_training_guard_uses_recorded_policy(tmp_path, monkeypatch):
    storage.prepare_storage_policy(tmp_path, min_free_gib=0.0001)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    callback = storage.storage_callback(tmp_path)
    with pytest.raises(storage.StorageLimitError):
        callback.on_step_begin(None, None, None)


def test_inventory_uses_effective_cache_paths_without_double_counting(tmp_path, monkeypatch):
    from huggingface_hub import constants
    from datasets import config
    environment, hub, pip, outputs = [tmp_path / name for name in ("environment", "hub", "pip", "outputs")]
    datasets = hub / "datasets"
    xet = hub / "xet"
    for path in (environment, hub, datasets, xet, pip, outputs):
        path.mkdir(parents=True, exist_ok=True)
        (path / "content").write_bytes(b"12345")
    monkeypatch.setattr(storage.sys, "prefix", str(environment))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(constants, "HF_XET_CACHE", str(xet))
    monkeypatch.setattr(config, "HF_DATASETS_CACHE", str(datasets))
    monkeypatch.setattr(storage.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=str(pip)))
    result = storage.storage_inventory(outputs)
    assert result["unique_total_bytes"] == 30
    assert all(Path(row["path"]).is_absolute() for row in result["directories"])
    assert result["temporary_bytes"] is None


def test_volume_root_is_not_scanned(tmp_path):
    with pytest.raises(ValueError, match="entire volume"):
        storage.directory_bytes(Path(tmp_path.anchor))
