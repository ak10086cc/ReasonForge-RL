"""Environment inspection without credentials or private absolute paths."""
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from .utils import versions, command_output


def inspect_environment():
    disk = shutil.disk_usage(Path.cwd())
    home = Path.home()
    cache = home / ".cache" / "huggingface"
    import os
    cache = Path(os.environ.get("HF_HOME", cache))
    model_cache = Path(os.environ.get("HF_HUB_CACHE", cache / "hub"))
    dataset_cache = Path(os.environ.get("HF_DATASETS_CACHE", cache / "datasets"))
    result = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "os": platform.platform(), "python": platform.python_version(),
        "python_architecture": platform.architecture()[0],
        "cpu": platform.processor(), "disk_free_gib": round(disk.free / 2**30, 2),
        "disk_total_gib": round(disk.total / 2**30, 2),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_remote_configured": bool(command_output(["git", "remote"])),
        "git_status": command_output(["git", "status", "--short"]),
        "dependencies": versions(),
        "model_cache_names": [p.name for p in model_cache.glob("models--*")],
        "dataset_cache_names": [p.name for p in dataset_cache.glob("openai*")],
        "project_checkpoints": [str(p.relative_to(Path.cwd())) for p in Path.cwd().glob("outputs/runs/*/*/final")],
    }
    try:
        result["nvidia_smi"] = command_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"])
    except FileNotFoundError:
        result["nvidia_smi"] = "not available"
    try:
        with urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct", timeout=20) as r:
            result["huggingface_network"] = r.status
    except Exception as e:
        result["huggingface_network"] = type(e).__name__
    try:
        import torch
        result["torch_cuda_runtime"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["cuda_compute_capability"] = list(torch.cuda.get_device_capability())
            result["bf16_supported"] = torch.cuda.is_bf16_supported()
            x = torch.ones(16, device="cuda")
            result["cuda_tensor_test"] = float((x * x).sum().item()) == 16
        import psutil
        result["ram_available_gib"] = round(psutil.virtual_memory().available / 2**30, 2)
    except ImportError:
        result["cuda_available"] = "PyTorch not installed"
    return result


def write_environment(path="reports/environment.md"):
    result = inspect_environment()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# 实际环境检测\n\n"
        "初始状态：目录仅有 AGENTS.md 和 docs/项目实施方案.md；非 Git 仓库，未发现模型/数据缓存或 checkpoint。"
        "默认 Python 3.14.5；已有 Python 3.12.10 未安装项目依赖。已在仓库外建立专用 Python 3.12 环境，"
        "本地初始化 main 分支，未配置远程、未上传。\n\n"
        "下列为本命令的实际检测快照。nvidia-smi 中 CUDA 上限与 torch CUDA runtime 是不同概念；"
        "采用官方 cu128 wheel 以支持现有 Ampere GPU。\n\n```json\n"
        + json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n\n"
        "4 GB 显存配置：bf16、单样本 microbatch、梯度累积、gradient checkpointing、SDPA、"
        "不启用 vLLM。完整规模运行前必须先检查 smoke/small 的耗时、峰值显存与奖励诊断。\n",
        encoding="utf-8",
    )
    return result
