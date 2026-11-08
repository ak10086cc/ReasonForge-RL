"""Local storage inventory and bounded experiment output, without cleanup."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from .utils import write_json


class StorageLimitError(RuntimeError):
    pass


def directory_bytes(path):
    """Logical bytes; do not follow symlinks/junctions or scan an entire drive."""
    path = Path(path)
    if path.resolve() == Path(path.resolve().anchor):
        raise ValueError("Refusing to scan an entire volume")
    if not path.exists():
        return 0
    total = 0
    def failed(error):
        raise error
    for directory, dirs, files in os.walk(path, followlinks=False, onerror=failed):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()
                   and not (Path(directory) / d).is_junction()]
        for name in files:
            file = Path(directory) / name
            if not file.is_symlink():
                total += file.stat().st_size
    return total


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def storage_inventory(output_dir):
    from huggingface_hub import constants as hub
    from datasets import config as datasets_config
    pip = subprocess.run([sys.executable, "-m", "pip", "cache", "dir"],
                         capture_output=True, text=True, check=True)
    specs = [("environment", Path(sys.prefix), "required_environment"),
             ("model_hub", Path(hub.HF_HUB_CACHE), "reusable_shared_cache"),
             ("datasets", Path(datasets_config.HF_DATASETS_CACHE), "reusable_shared_cache"),
             ("xet", Path(hub.HF_XET_CACHE), "reusable_shared_cache"),
             ("pip", Path(pip.stdout.strip()), "shared_download_cache"),
             ("outputs", Path(output_dir), "experiment_artifacts")]
    rows = []
    for name, path, category in specs:
        path = path.resolve()
        rows.append({"name": name, "path": str(path), "category": category,
                     "bytes": directory_bytes(path),
                     "disk_free_bytes": shutil.disk_usage(existing_parent(path)).free})
    # Report overlapping paths separately but count their bytes only once.
    roots = []
    total = 0
    for row in sorted(rows, key=lambda row: len(Path(row["path"]).parts)):
        path = Path(row["path"])
        if not any(path.is_relative_to(parent) for parent in roots):
            roots.append(path)
            total += row["bytes"]
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "directories": rows,
            "unique_total_bytes": total,
            "temporary_directory": str(Path(tempfile.gettempdir()).resolve()),
            "temporary_bytes": None,
            "note": "Logical bytes, not allocated disk blocks. Shared caches are not attributed entirely to this project; system temporary directory is not scanned."}


def prepare_storage_policy(run, max_run_gib=5.0, min_free_gib=50.0):
    if not all(math.isfinite(v) and v > 0 for v in (max_run_gib, min_free_gib)):
        raise ValueError("Storage limits must be positive finite GiB values")
    path = Path(run) / "storage_policy.json"
    policy = {"max_run_bytes": int(max_run_gib * 2**30), "min_free_bytes": int(min_free_gib * 2**30)}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != policy:
            raise ValueError("Storage policy differs from this run; keep the recorded limits")
    else:
        write_json(path, policy)
    return check_storage(run)


def check_storage(run):
    path = Path(run) / "storage_policy.json"
    if not path.exists():
        return None  # Older experiments have no retrospectively invented policy.
    policy = json.loads(path.read_text(encoding="utf-8"))
    free = shutil.disk_usage(existing_parent(run)).free
    used = directory_bytes(run)
    if free < policy["min_free_bytes"] or used >= policy["max_run_bytes"]:
        raise StorageLimitError("Storage limit reached: stop this invocation; preserve outputs and inspect the last checkpoint")
    return {"run_bytes": used, "disk_free_bytes": free, **policy}


def storage_callback(run):
    from transformers import TrainerCallback

    class StorageGuard(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            check_storage(run)

        def on_save(self, args, state, control, **kwargs):
            check_storage(run)
    return StorageGuard()
