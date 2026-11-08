"""Run only the authorized Base stage unattended, then verify and stop before SFT."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reasonforge.utils import (configure_runtime, digest, file_hash, load_config,
                              read_jsonl, write_json)
from reasonforge.storage import check_storage, storage_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def completed_questions(run):
    return sum(read(p)["completed"] for p in (Path(run) / "evaluation").glob("base_*/progress.json"))


def verify_inputs(config_path, run):
    config = load_config(config_path)
    if config != read(run / "config.json") or config["name"] != "full":
        raise ValueError("Use the unchanged Full config belonging to this run")
    provenance = read(run / "provenance.json")
    for name, expected in provenance["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Run source/config changed: {name}; preserve results and inspect before resuming")
    manifest = read(run / "data/manifest.json")
    if manifest["revision"] != config["data"]["revision"] or manifest["seed"] != config["seed"]:
        raise ValueError("Data revision or seed differs")
    hashes = []
    for split, count in (("train", 7345), ("validation", 128), ("test", 1319)):
        rows = read_jsonl(run / "data" / f"{split}.jsonl")
        spec = manifest["splits"][split]
        source = "official_test" if split == "test" else "official_train"
        if (len(rows) != count or spec["selected_count"] != count
                or digest(rows) != spec["selected_hash"]
                or [r["question_id"] for r in rows] != spec["selected_ids"]
                or any(r["split"] != split or r["source"] != source for r in rows)):
            raise ValueError(f"Invalid Full {split} membership or hash")
        hashes.append({r["question_hash"] for r in rows})
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i)):
        raise ValueError("Training, validation and test questions overlap")
    if any((run / name).exists() for name in ("sft", "grpo", "grpo_correctness_only")):
        raise ValueError("This runner must stop before training; use a Base-only run")
    check_storage(run)
    return config


def run_batches(config_path, run):
    """Repeat only planned pauses. Failures and no-progress pauses stop, never retry."""
    stage = run / "stage3"
    stage.mkdir(exist_ok=True)
    invocation = len(list(stage.glob("baseline-*.log")))
    while True:
        verify_inputs(config_path, run)
        before = completed_questions(run)
        invocation += 1
        command = [sys.executable, "-u", str(ROOT / "scripts/run_pipeline.py"), "--config", str(config_path),
                   "--phase", "baseline", "--run-dir", str(run), "--eval-max-seconds", "3600"]
        log = stage / f"baseline-{invocation:04d}.log"
        write_json(stage / "worker_status.json", {"stage": 3, "status": "running", "pid": os.getpid(),
            "invocation": invocation, "command": command, "log": str(log.relative_to(run)),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "stops_before_stage": 4})
        print(f"START baseline invocation {invocation}; completed questions={before}", flush=True)
        with log.open("x", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Baseline exited {result.returncode}; inspect {log.name}; no automatic failure retry")
        status = read(run / "status.json")
        if status.get("status") == "completed" and status.get("completed_phase") == "baseline":
            return
        if status.get("status") != "paused" or completed_questions(run) <= before:
            raise RuntimeError("Unexpected baseline status or no progress; preserve outputs and inspect")
        write_json(stage / "storage_latest.json", storage_inventory(run.parent))


def finish(config_path, run):
    from reasonforge.evaluate import evaluate, evaluation_contract
    from reasonforge.analyze import collect_results
    config = verify_inputs(config_path, run)
    if read(run / "evaluation_contract.json") != evaluation_contract(config, run):
        raise ValueError("Frozen evaluation contract changed")
    # Completed evaluations are verified read-only by evaluate(); no new inference.
    for split in ("validation", "test"):
        if not (run / "evaluation" / f"base_{split}" / "metrics.json").exists():
            raise ValueError("Full validation and test metrics are required before acceptance")
        evaluate(config, run, "base", split)
    results = collect_results(run)
    if set(results) != {"base"} or results["base"]["n"] != 1319:
        raise ValueError("Acceptance requires exactly the complete Base evaluation")
    final = storage_inventory(run.parent)
    write_json(run / "stage3/storage_final.json", final)
    before = read(run / "stage3/storage_before.json")
    deltas = [{"name": row["name"], "path": row["path"], "after_bytes": row["bytes"],
               "delta_bytes": row["bytes"] - next(old["bytes"] for old in before["directories"]
                                                   if old["name"] == row["name"])}
              for row in final["directories"]]
    record = {"stage": 3, "status": "complete", "stopped_before_stage": 4,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "base": results["base"], "validation": read(run / "evaluation/base_validation/metrics.json"),
              "configuration_sha256": file_hash(run / "config.json"),
              "data_manifest_sha256": file_hash(run / "data/manifest.json"),
              "evaluation_contract_sha256": file_hash(run / "evaluation_contract.json"),
              "checks": {"full_data_hashes": True, "source_unchanged": True,
                         "complete_ordered_predictions_and_recomputed_metrics": True,
                         "sft_started": False, "grpo_started": False},
              "storage_deltas": deltas, "storage": check_storage(run),
              "tests": read(run / "stage3/preflight_tests.json"),
              "temporary_cleanup": "Preflight pytest fixture directory only; preserve formal data, predictions, logs and contract",
              "next_stage": "Stage 4 SFT on this same run; requires separate authorization"}
    write_json(run / "stage3/acceptance.json", record)
    write_json(run / "stage3/worker_status.json", {"stage": 3, "status": "completed", "pid": os.getpid(),
               "stopped_before_stage": 4, "acceptance": "stage3/acceptance.json"})
    print("DONE Stage 3; complete verified Base; stopped before SFT", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    configure_runtime()
    run = args.run_dir.resolve()
    config_path = args.config.resolve()
    stage = run / "stage3"
    stage.mkdir(exist_ok=True)
    with FileLock(run / ".baseline_stage.lock", timeout=0):
        try:
            verify_inputs(config_path, run)
            if not (stage / "storage_before.json").exists():
                write_json(stage / "storage_before.json", storage_inventory(run.parent))
            if (stage / "acceptance.json").exists():
                print("Stage 3 acceptance already exists; preserve it", flush=True)
                return
            run_batches(config_path, run)
            finish(config_path, run)
        except BaseException as error:
            write_json(stage / "worker_status.json", {"stage": 3, "status": "failed", "pid": os.getpid(),
                "error_type": type(error).__name__, "message": str(error).replace(str(Path.home()), "<user-home>"),
                "recovery": "Inspect worker/baseline logs and fix the cause before repeating this runner on the same run; no all/SFT phase is launched"})
            raise


if __name__ == "__main__":
    main()
