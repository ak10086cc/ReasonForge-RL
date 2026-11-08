import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from reasonforge.utils import write_json

spec = importlib.util.spec_from_file_location("baseline_stage", Path(__file__).parents[1] / "scripts/run_baseline_stage.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def setup_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "verify_inputs", lambda *args: {})
    monkeypatch.setattr(worker, "storage_inventory", lambda *args: {})
    write_json(tmp_path / "evaluation/base_validation/progress.json", {"completed": 0})
    return tmp_path / "evaluation/base_validation/progress.json"


def test_only_planned_pauses_are_resumed_then_stop_at_baseline(tmp_path, monkeypatch):
    progress = setup_worker(tmp_path, monkeypatch)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            write_json(progress, {"completed": 1})
            write_json(tmp_path / "status.json", {"status": "paused"})
        else:
            write_json(tmp_path / "status.json", {"status": "completed", "completed_phase": "baseline"})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, "run", execute)
    worker.run_batches(Path("full.yaml"), tmp_path)
    assert len(commands) == 2
    assert all(c[c.index("--phase")+1] == "baseline" and c[-1] == "3600" for c in commands)
    assert len(list((tmp_path / "stage3").glob("baseline-*.log"))) == 2


@pytest.mark.parametrize("returncode,status", [(1, {"status": "failed"}), (0, {"status": "paused"})])
def test_failure_or_no_progress_stops_without_retry(tmp_path, monkeypatch, returncode, status):
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
