import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_output(args):
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() if result.returncode == 0 else f"unavailable (exit {result.returncode})"


def versions():
    names = ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "tokenizers",
             "numpy", "pandas", "matplotlib", "pytest", "PyYAML"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def configure_runtime():
    # Prevent tracking integrations and unsolicited telemetry/network logging.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TRL_HUB_DISABLE_TELEMETRY", "1")


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["model"]["id"] not in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
        raise ValueError("Use an approved Qwen2.5 model in a separate pinned comparison")
    g = config["grpo"]
    if g["num_generations"] < 2 or (g["batch_size"] * g["gradient_accumulation_steps"]) % g["num_generations"]:
        raise ValueError("GRPO effective batch must be divisible by num_generations >= 2")
    if g["reward_weights"][0] <= g["reward_weights"][1]:
        raise ValueError("Correctness must outweigh format")
    if config["evaluation"]["temperature"] != 0:
        raise ValueError("Frozen comparison uses greedy decoding")
    return config


def create_run(config):
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = Path(config["output_dir"]) / f"{config['name']}-{now}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    provenance = {
        "created_at_utc": now, "command": [Path(sys.executable).name, *sys.argv],
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]), "dependencies": versions(),
        "source_hashes": {str(p).replace('\\', '/'): file_hash(p) for base in ("src", "scripts", "configs", "tests")
                          for p in Path(base).rglob("*") if p.is_file() and p.suffix in (".py", ".yaml")},
        "all_dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
        "offline_mode": os.environ.get("HF_HUB_OFFLINE", "0"),
    }
    import torch
    provenance["hardware"] = {"cuda_available": torch.cuda.is_available(), "torch_cuda_runtime": torch.version.cuda,
                              "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                              "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else None}
    import zipfile
    with zipfile.ZipFile(run / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in provenance["source_hashes"]:
            archive.write(name, name)
        for name in ("requirements.txt", "AGENTS.md", "docs/项目实施方案.md"):
            archive.write(name, name)
    provenance["source_snapshot_sha256"] = file_hash(run / "source_snapshot.zip")
    write_json(run / "provenance.json", provenance)
    return run
