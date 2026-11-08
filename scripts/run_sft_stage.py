"""Run the authorized Full SFT stage unattended; verify and stop before GRPO."""
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
                              read_jsonl, versions, write_json)
from reasonforge.storage import check_storage, storage_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_inputs(config_path, run):
    from reasonforge.analyze import collect_results
    from reasonforge.evaluate import evaluation_contract
    config = load_config(config_path)
    if config != read(run / "config.json") or config["name"] != "full":
        raise ValueError("Use this run's unchanged Full configuration")
    acceptance = read(run / "stage3/acceptance.json")
    if acceptance["status"] != "complete" or acceptance["stopped_before_stage"] != 4:
        raise ValueError("Complete Base acceptance is required")
    for name, key in (("config.json", "configuration_sha256"),
                      ("data/manifest.json", "data_manifest_sha256"),
                      ("evaluation_contract.json", "evaluation_contract_sha256")):
        if file_hash(run / name) != acceptance[key]:
            raise ValueError(f"Base input changed: {name}")
    original = read(run / "provenance.json")
    additions = read(run / "stage4/provenance.json")
    if (file_hash(run / "provenance.json") != additions["parent_provenance_sha256"]
            or file_hash(run / "source_snapshot.zip") != original["source_snapshot_sha256"]
            or file_hash(run / "stage4/source_additions.zip") != additions["source_snapshot_sha256"]):
        raise ValueError("Source provenance changed")
    for name, expected in {**original["source_hashes"], **additions["source_hashes"]}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Run source/config changed: {name}")
    if versions() != original["dependencies"]:
        raise ValueError("Dependencies changed since Base")
    manifest = read(run / "data/manifest.json")
    question_sets = []
    for split, count in (("train", 7345), ("validation", 128), ("test", 1319)):
        rows = read_jsonl(run / "data" / f"{split}.jsonl")
        expected = manifest["splits"][split]
        source = "official_test" if split == "test" else "official_train"
        if (len(rows) != count or expected["selected_count"] != count
                or digest(rows) != expected["selected_hash"]
                or [r["question_id"] for r in rows] != expected["selected_ids"]
                or any(r["split"] != split or r["source"] != source for r in rows)):
            raise ValueError(f"Full {split} membership or contents changed")
        question_sets.append({r["question_hash"] for r in rows})
    if any(question_sets[i] & question_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Training, validation and test overlap")
    if read(run / "evaluation_contract.json") != evaluation_contract(config, run):
        raise ValueError("Frozen evaluation contract changed")
    if collect_results(run).get("base") != acceptance["base"]:
        raise ValueError("Base predictions or metrics changed")
    if any((run / name).exists() for name in ("grpo", "grpo_correctness_only")):
        raise ValueError("This runner stops before GRPO")
    check_storage(run)
    return config


def completed_questions(run):
    path = run / "evaluation/sft_test/progress.json"
    return read(path)["completed"] if path.exists() else 0


def run_batches(config_path, run, resume=None):
    stage = run / "stage4"
    invocation = len(list(stage.glob("sft-*.log")))
    while True:
        verify_inputs(config_path, run)
        before = completed_questions(run)
        invocation += 1
        command = [sys.executable, "-u", str(ROOT / "scripts/run_pipeline.py"), "--config", str(config_path),
                   "--phase", "sft", "--run-dir", str(run), "--eval-max-seconds", "3600"]
        if resume is not None:
            command.extend(["--resume-checkpoint", str(resume)])
        log = stage / f"sft-{invocation:04d}.log"
        write_json(stage / "worker_status.json", {"stage": 4, "status": "running", "pid": os.getpid(),
            "invocation": invocation, "command": command, "log": str(log.relative_to(run)),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "stops_before_stage": 5})
        print(f"START SFT invocation {invocation}; completed test questions={before}", flush=True)
        with log.open("x", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"SFT exited {result.returncode}; inspect {log.name}; no automatic failure retry")
        status = read(run / "status.json")
        if status.get("status") == "completed" and status.get("completed_phase") == "sft":
            return
        if (status.get("status") != "paused" or status.get("phase") != "sft"
                or completed_questions(run) <= before):
            raise RuntimeError("Unexpected SFT status or no progress; preserve outputs and inspect")
        resume = None  # Training is complete; subsequent invocations reuse final weights.
        write_json(stage / "storage_latest.json", storage_inventory(run.parent))


def finish(config_path, run):
    from reasonforge.analyze import collect_results
    from reasonforge.evaluate import evaluate
    config = verify_inputs(config_path, run)
    if not (run / "evaluation/sft_test/metrics.json").exists():
        raise ValueError("Complete SFT test evaluation is required")
    evaluate(config, run, "sft")  # Verify completed evaluation without inference or rewriting.
    results = collect_results(run)
    if set(results) != {"base", "sft"} or results["sft"]["n"] != 1319:
        raise ValueError("Acceptance requires complete Base and SFT only")
    summary = read(run / "sft/summary.json")
    mask = read(run / "sft/loss_mask_audit.json")
    logs = read_jsonl(run / "sft/train_log.jsonl")
    exclusions = read(run / "sft/length_exclusions.json")
    steps = [row["step"] for row in logs if "loss" in row]
    if (not mask["passed"] or not summary["weight_update"]["parameters_changed"]
            or not results["sft"]["adapter_reload_verified"]
            or not steps or max(steps) != config["sft"]["max_steps"]
            or summary["train_samples"] != 7345 - len(exclusions["train"])
            or summary["validation_samples"] != 128 - len(exclusions["validation"])):
        raise ValueError("SFT training, loss mask, sample counts or adapter checks failed")
    final = storage_inventory(run.parent)
    write_json(run / "stage4/storage_final.json", final)
    before = read(run / "stage4/storage_before.json")
    deltas = [{"name": row["name"], "path": row["path"], "after_bytes": row["bytes"],
               "delta_bytes": row["bytes"] - next(old["bytes"] for old in before["directories"]
                                                   if old["name"] == row["name"])}
              for row in final["directories"]]
    record = {"stage": 4, "status": "complete", "stopped_before_stage": 5,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "base": results["base"], "sft": results["sft"], "training": summary,
              "loss_mask": mask, "maximum_training_step": max(steps),
              "configuration_sha256": file_hash(run / "config.json"),
              "data_manifest_sha256": file_hash(run / "data/manifest.json"),
              "evaluation_contract_sha256": file_hash(run / "evaluation_contract.json"),
              "final_adapter_sha256": file_hash(run / "sft/final/adapter_model.safetensors"),
              "checks": {"original_sources_and_data_unchanged": True, "complete_ordered_predictions_and_recomputed_metrics": True,
                         "adapter_saved_and_reloaded": True, "grpo_started": False},
              "storage_deltas": deltas, "storage": check_storage(run),
              "tests": read(run / "stage4/preflight_tests.json"),
              "next_stage": "Stage 5 GRPO from this same SFT adapter; requires separate authorization"}
    write_json(run / "stage4/acceptance.json", record)
    write_json(run / "stage4/worker_status.json", {"stage": 4, "status": "completed", "pid": os.getpid(),
               "stopped_before_stage": 5, "acceptance": "stage4/acceptance.json"})
    print("DONE Stage 4; verified Full SFT; stopped before GRPO", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    configure_runtime()
    run, config_path = args.run_dir.resolve(), args.config.resolve()
    stage = run / "stage4"
    if not (stage / "preflight_tests.json").exists() or not (stage / "storage_before.json").exists():
        raise ValueError("Stage 4 preflight tests, provenance and storage baseline are required")
    with FileLock(run / ".sft_stage.lock", timeout=0):
        try:
            verify_inputs(config_path, run)
            if (stage / "acceptance.json").exists():
                print("Stage 4 acceptance already exists; preserve it", flush=True)
                return
            run_batches(config_path, run, args.resume_checkpoint)
            finish(config_path, run)
        except BaseException as error:
            write_json(stage / "worker_status.json", {"stage": 4, "status": "failed", "pid": os.getpid(),
                "error_type": type(error).__name__, "message": str(error).replace(str(Path.home()), "<user-home>"),
                "recovery": "Inspect SFT logs and last complete checkpoint; fix the cause before repeating. Evaluation resumes with final weights; incomplete training needs --resume-checkpoint. GRPO is never launched."})
            raise


if __name__ == "__main__":
    main()
