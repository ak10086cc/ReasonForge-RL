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


@pytest.mark.parametrize("custom_hub", [False, True])
def test_runtime_cache_locations_in_fresh_process(tmp_path, custom_hub):
    import os
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    names = ("HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE")
    for name in names:
        env.pop(name, None)
    env["PYTHONPATH"] = str(root / "src")
    if custom_hub:
        env["HF_HUB_CACHE"] = str(tmp_path / "custom-hub")
    code = """
import json
from reasonforge.utils import configure_runtime
configure_runtime()
from huggingface_hub import constants as hub
from datasets import config as data
print(json.dumps([hub.HF_HUB_CACHE, str(data.HF_DATASETS_CACHE),
                  hub.HF_XET_CACHE, hub.HF_ASSETS_CACHE]))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=env, capture_output=True, text=True, check=True)
    actual = [Path(p) for p in json.loads(result.stdout)]
    cache = root / ".cache" / "huggingface"
    assert actual == [tmp_path / "custom-hub" if custom_hub else cache / "hub",
                      cache / "datasets", cache / "xet", cache / "assets"]


def test_create_run_without_private_documents(tmp_path, monkeypatch):
    config = load_config("configs/smoke.yaml")
    workspace = tmp_path / "public-checkout"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/fixture.py").write_text("# archive fixture\n", encoding="utf-8")
    (workspace / "requirements.txt").write_text("# fixture manifest\n", encoding="utf-8")
    config["output_dir"] = str(tmp_path / "runs")
    monkeypatch.chdir(workspace)
    run = create_run(config)
    import zipfile
    with zipfile.ZipFile(run / "source_snapshot.zip") as archive:
        assert "src/fixture.py" in archive.namelist()
        assert "requirements.txt" in archive.namelist()
        assert "AGENTS.md" not in archive.namelist()
        assert "docs/项目实施方案.md" not in archive.namelist()
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_snapshot_sha256"] == file_hash(run / "source_snapshot.zip")


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
