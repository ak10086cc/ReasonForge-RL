"""No synthetic experiment metrics: test the CLI and artifact integrity."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from reasonforge.utils import load_config, create_run, write_jsonl, file_hash
from reasonforge.metrics import summarize
from reasonforge.evaluate import freeze_evaluation


def test_configs_and_unique_run_directories(tmp_path):
    for name in ("smoke", "small", "full"):
        c = load_config(f"configs/{name}.yaml")
        c["output_dir"] = str(tmp_path)
        a, b = create_run(c), create_run(c)
        assert a != b and (a / "provenance.json").exists()
        import zipfile, hashlib
        provenance = json.loads((a/"provenance.json").read_text(encoding="utf-8"))
        assert provenance["source_snapshot_sha256"] == file_hash(a/"source_snapshot.zip")
        with zipfile.ZipFile(a/"source_snapshot.zip") as archive:
            assert all(hashlib.sha256(archive.read(name)).hexdigest() == value
                       for name,value in provenance["source_hashes"].items())


def test_cli_help():
    result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--help"], capture_output=True)
    assert result.returncode == 0 and b"--resume-checkpoint" in result.stdout


def test_resume_requires_an_explicit_training_phase():
    result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--resume-checkpoint", "unused"], capture_output=True)
    assert result.returncode == 2 and b"individual training phase" in result.stderr


def test_contract_rejects_changed_data_and_generation(tmp_path):
    c = load_config("configs/smoke.yaml")
    for split in ("test", "validation"):
        write_jsonl(tmp_path / "data" / f"{split}.jsonl", [{"question_id": split}])
    freeze_evaluation(c, tmp_path)
    c["evaluation"]["max_new_tokens"] += 1
    with pytest.raises(ValueError, match="changed"):
        freeze_evaluation(c, tmp_path)
    c["evaluation"]["max_new_tokens"] -= 1
    write_jsonl(tmp_path / "data/test.jsonl", [{"question_id": "other"}])
    with pytest.raises(ValueError, match="changed"):
        freeze_evaluation(c, tmp_path)
