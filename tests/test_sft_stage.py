import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from reasonforge.utils import write_json

spec = importlib.util.spec_from_file_location("sft_stage", Path(__file__).parents[1] / "scripts/run_sft_stage.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def setup_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "verify_inputs", lambda *args: {})
    monkeypatch.setattr(worker, "storage_inventory", lambda *args: {})
    (tmp_path / "stage4").mkdir()
    return tmp_path / "evaluation/sft_test/progress.json"


def test_sft_only_and_training_resume_used_once(tmp_path, monkeypatch):
    progress = setup_worker(tmp_path, monkeypatch)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            write_json(progress, {"completed": 12})
            write_json(tmp_path / "status.json", {"status": "paused", "phase": "sft"})
        else:
            write_json(tmp_path / "status.json", {"status": "completed", "completed_phase": "sft"})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, "run", execute)
    worker.run_batches(Path("full.yaml"), tmp_path, Path("checkpoint-100"))
    assert len(commands) == 2
    assert all(c[c.index("--phase")+1] == "sft" and c[c.index("--eval-max-seconds")+1] == "3600" for c in commands)
    assert "--resume-checkpoint" in commands[0] and "--resume-checkpoint" not in commands[1]
    assert len(list((tmp_path / "stage4").glob("sft-*.log"))) == 2


@pytest.mark.parametrize("returncode,status", [
    (1, {"status": "failed"}),
    (0, {"status": "paused", "phase": "sft"}),
    (0, {"status": "completed", "completed_phase": "grpo"}),
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
