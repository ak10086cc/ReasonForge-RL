import json
from pathlib import Path
import pytest
from reasonforge.recovery import prepare_stage
from reasonforge.utils import load_config, write_json


def make_checkpoint(run, config, stage="sft", step=1):
    checkpoint = run / stage / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    write_json(run / "config.json", config)
    write_json(checkpoint / "trainer_state.json", {"global_step":step})
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).touch()
    return checkpoint


def test_recovery_protects_completed_and_newer_checkpoints(tmp_path):
    config = load_config("configs/small.yaml")
    run = tmp_path/"run"
    checkpoint = make_checkpoint(run, config)
    assert prepare_stage(run, "sft", config, checkpoint) == run/"sft"
    (run/"sft/checkpoint-2").mkdir()
    with pytest.raises(FileExistsError, match="newer"):
        prepare_stage(run, "sft", config, checkpoint)
    (run/"sft/final").mkdir()
    with pytest.raises(FileExistsError, match="Completed"):
        prepare_stage(run, "sft", config, checkpoint)


def test_recovery_rejects_changed_config_and_incomplete_state(tmp_path):
    config = load_config("configs/small.yaml")
    run = tmp_path/"run"
    checkpoint = make_checkpoint(run, config)
    config["seed"] = 7
    with pytest.raises(ValueError, match="seed"):
        prepare_stage(run, "sft", config, checkpoint)
    (checkpoint/"optimizer.pt").unlink()
    with pytest.raises(ValueError, match="missing"):
        prepare_stage(run, "sft", config, checkpoint)


def test_fresh_stage_cannot_replace_existing(tmp_path):
    config = load_config("configs/small.yaml")
    prepare_stage(tmp_path, "sft", config)
    with pytest.raises(FileExistsError):
        prepare_stage(tmp_path, "sft", config)
