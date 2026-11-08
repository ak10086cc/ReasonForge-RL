import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from reasonforge.utils import write_json

spec = importlib.util.spec_from_file_location("grpo_stage", Path(__file__).parents[1] / "scripts/run_grpo_stage.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def setup_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "verify_inputs", lambda *args: {})
    monkeypatch.setattr(worker, "storage_inventory", lambda *args: {})
    (tmp_path / "stage5").mkdir()
    return tmp_path / "evaluation/grpo_test/progress.json"


def test_grpo_only_and_training_resume_used_once(tmp_path, monkeypatch):
    progress = setup_worker(tmp_path, monkeypatch)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            write_json(progress, {"completed": 12})
            write_json(tmp_path / "status.json", {"status": "paused", "phase": "grpo"})
        else:
            write_json(tmp_path / "status.json", {"status": "completed", "completed_phase": "grpo"})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, "run", execute)
    worker.run_batches(Path("full.yaml"), tmp_path, Path("checkpoint-100"))
    assert len(commands) == 2
    assert all(c[c.index("--phase")+1] == "grpo" and c[c.index("--eval-max-seconds")+1] == "3600" for c in commands)
    assert "--resume-checkpoint" in commands[0] and "--resume-checkpoint" not in commands[1]
    assert len(list((tmp_path / "stage5").glob("grpo-*.log"))) == 2


@pytest.mark.parametrize("returncode,status", [
    (1, {"status": "failed"}),
    (0, {"status": "paused", "phase": "grpo"}),
    (0, {"status": "completed", "completed_phase": "ablation"}),
])
def test_failure_no_progress_or_wrong_phase_is_never_retried(tmp_path, monkeypatch, returncode, status):
    setup_worker(tmp_path, monkeypatch)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        write_json(tmp_path / "status.json", status)
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(worker.subprocess, "run", execute)
    with pytest.raises(RuntimeError):
        worker.run_batches(Path("full.yaml"), tmp_path)
    assert len(calls) == 1


@pytest.fixture
def training_fixture(tmp_path):
    from reasonforge.utils import write_jsonl
    config = {"seed": 42, "grpo": {"max_steps": 256, "num_generations": 4, "reward_weights": [2.0, 0.2]}}
    parent = {"final_adapter_sha256": "parent-file", "training": {"weight_update": {"after_hash": "sft-state"}}}
    summary = {"parent_checkpoint": "sft/final", "parent_adapter_sha256": "parent-file",
               "sft_start_hash": "sft-state", "weight_update": {"before_hash": "sft-state", "parameters_changed": True},
               "resumed": False, "correctness_only": False, "train_samples": 7345, "generations_seen": 4}
    rollout = {"question_id": "train:1", "ground_truth": "2", "prediction": "Final Answer: 2",
               "extracted_answer": "2", "stage": "grpo", "seed": 42,
               "correctness_reward": 1.0, "format_reward": 1.0, "weighted_reward": 2.2}
    write_json(tmp_path / "stage4/acceptance.json", parent)
    write_json(tmp_path / "grpo/summary.json", summary)
    write_jsonl(tmp_path / "grpo/train_log.jsonl", [{"step": 256, "loss": 0.1}])
    write_jsonl(tmp_path / "grpo/rollout_audit.jsonl", [{"n": 4}])
    write_jsonl(tmp_path / "grpo/rollouts.jsonl", [rollout] * 4)
    write_jsonl(tmp_path / "data/train.jsonl", [{"question_id": "train:1", "ground_truth": "2"}])
    return config, {"grpo": {"adapter_reload_verified": True}}


def test_acceptance_recomputes_rollout_rewards(tmp_path, training_fixture):
    from reasonforge.utils import read_jsonl, write_jsonl
    config, results = training_fixture
    assert worker.validate_training(config, tmp_path, results)[1] == 256
    path = tmp_path / "grpo/rollouts.jsonl"
    rows = read_jsonl(path)
    rows[0]["weighted_reward"] = 0.0
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="rewards disagree"):
        worker.validate_training(config, tmp_path, results)


@pytest.mark.parametrize("field,value", [("parent_adapter_sha256", "wrong-file"), ("sft_start_hash", "wrong-policy")])
def test_acceptance_rejects_wrong_sft_parent(tmp_path, training_fixture, field, value):
    config, results = training_fixture
    path = tmp_path / "grpo/summary.json"
    summary = worker.read(path)
    summary[field] = value
    write_json(path, summary)
    with pytest.raises(ValueError, match="continuity"):
        worker.validate_training(config, tmp_path, results)


def test_acceptance_rejects_nontraining_rollout(tmp_path, training_fixture):
    from reasonforge.utils import read_jsonl, write_jsonl
    config, results = training_fixture
    path = tmp_path / "grpo/rollouts.jsonl"
    rows = read_jsonl(path)
    rows[0]["question_id"] = "official_test:1"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="membership"):
        worker.validate_training(config, tmp_path, results)
