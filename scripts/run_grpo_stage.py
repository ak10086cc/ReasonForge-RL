"""Run the authorized Full GRPO stage unattended; verify and stop before ablation."""
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
    acceptance = read(run / "stage4/acceptance.json")
    if acceptance["status"] != "complete" or acceptance["stopped_before_stage"] != 5:
        raise ValueError("Complete SFT acceptance is required")
    for name, key in (("config.json", "configuration_sha256"),
                      ("data/manifest.json", "data_manifest_sha256"),
                      ("evaluation_contract.json", "evaluation_contract_sha256")):
        if file_hash(run / name) != acceptance[key]:
            raise ValueError(f"SFT input changed: {name}")
    original = read(run / "provenance.json")
    sft_additions = read(run / "stage4/provenance.json")
    additions = read(run / "stage5/provenance.json")
    if (file_hash(run / "provenance.json") != sft_additions["parent_provenance_sha256"]
            or file_hash(run / "stage4/provenance.json") != additions["parent_provenance_sha256"]
            or file_hash(run / "source_snapshot.zip") != original["source_snapshot_sha256"]
            or file_hash(run / "stage4/source_additions.zip") != sft_additions["source_snapshot_sha256"]
            or file_hash(run / "stage5/source_additions.zip") != additions["source_snapshot_sha256"]):
        raise ValueError("Source provenance changed")
    for name, expected in {**original["source_hashes"], **sft_additions["source_hashes"],
                           **additions["source_hashes"]}.items():
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
    results = collect_results(run)
    if any(results.get(key) != acceptance[key] for key in ("base", "sft")):
        raise ValueError("Base or SFT predictions or metrics changed")
    if (file_hash(run / "sft/final/adapter_model.safetensors") != acceptance["final_adapter_sha256"]
            or read(run / "sft/summary.json") != acceptance["training"]):
        raise ValueError("Accepted SFT parent weights or training summary changed")
    if (run / "grpo_correctness_only").exists():
        raise ValueError("This runner stops before ablation")
    check_storage(run)
    return config


def completed_questions(run):
    path = run / "evaluation/grpo_test/progress.json"
    return read(path)["completed"] if path.exists() else 0


def run_batches(config_path, run, resume=None):
    stage = run / "stage5"
    invocation = len(list(stage.glob("grpo-*.log")))
    while True:
        verify_inputs(config_path, run)
        before = completed_questions(run)
        invocation += 1
        command = [sys.executable, "-u", str(ROOT / "scripts/run_pipeline.py"), "--config", str(config_path),
                   "--phase", "grpo", "--run-dir", str(run), "--eval-max-seconds", "3600"]
        if resume is not None:
            command.extend(["--resume-checkpoint", str(resume)])
        log = stage / f"grpo-{invocation:04d}.log"
        write_json(stage / "worker_status.json", {"stage": 5, "status": "running", "pid": os.getpid(),
            "invocation": invocation, "command": command, "log": str(log.relative_to(run)),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "stops_before_stage": 6})
        print(f"START GRPO invocation {invocation}; completed test questions={before}", flush=True)
        with log.open("x", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"GRPO exited {result.returncode}; inspect {log.name}; no automatic failure retry")
        status = read(run / "status.json")
        if status.get("status") == "completed" and status.get("completed_phase") == "grpo":
            return
        if (status.get("status") != "paused" or status.get("phase") != "grpo"
                or completed_questions(run) <= before):
            raise RuntimeError("Unexpected GRPO status or no progress; preserve outputs and inspect")
        resume = None  # Training is complete; subsequent invocations reuse final weights.
        write_json(stage / "storage_latest.json", storage_inventory(run.parent))


def validate_training(config, run, results):
    summary = read(run / "grpo/summary.json")
    parent = read(run / "stage4/acceptance.json")
    logs = read_jsonl(run / "grpo/train_log.jsonl")
    steps = [row["step"] for row in logs if "loss" in row]
    audits = read_jsonl(run / "grpo/rollout_audit.jsonl")
    rollouts = read_jsonl(run / "grpo/rollouts.jsonl")
    if (summary["parent_checkpoint"] != "sft/final"
            or summary["parent_adapter_sha256"] != parent["final_adapter_sha256"]
            or summary["sft_start_hash"] != parent["training"]["weight_update"]["after_hash"]
            or (not summary["resumed"] and summary["weight_update"]["before_hash"] != summary["sft_start_hash"])
            or not summary["weight_update"]["parameters_changed"]
            or summary["correctness_only"] or summary["train_samples"] != 7345
            or not results["grpo"]["adapter_reload_verified"]
            or not steps or max(steps) != config["grpo"]["max_steps"]
            or not audits or summary["generations_seen"] != sum(a["n"] for a in audits)
            or len(rollouts) != summary["generations_seen"]
            or any(a["n"] != config["grpo"]["num_generations"] for a in audits)):
        raise ValueError("GRPO SFT continuity, update, rollout or completion checks failed")
    train_rows = {r["question_id"]: r for r in read_jsonl(run / "data/train.jsonl")}
    from reasonforge.metrics import extract_answer
    from reasonforge.rewards import correctness_reward, format_reward
    for row in rollouts:
        if (row["question_id"] not in train_rows
                or row["ground_truth"] != train_rows[row["question_id"]]["ground_truth"]
                or row["stage"] != "grpo" or row["seed"] != config["seed"]
                or row["extracted_answer"] != extract_answer(row["prediction"])):
            raise ValueError("Rollout training membership or prediction mismatch")
        correct = correctness_reward([row["prediction"]], [row["ground_truth"]])[0]
        formatted = format_reward([row["prediction"]])[0]
        weighted = correct * config["grpo"]["reward_weights"][0] + formatted * config["grpo"]["reward_weights"][1]
        if (row["correctness_reward"] != correct or row["format_reward"] != formatted
                or abs(row["weighted_reward"] - weighted) > 1e-8):
            raise ValueError("Saved rollout rewards disagree with reward functions")
    return summary, max(steps), audits


def finish(config_path, run):
    from reasonforge.analyze import collect_results
    from reasonforge.evaluate import evaluate
    config = verify_inputs(config_path, run)
    if not (run / "evaluation/grpo_test/metrics.json").exists():
        raise ValueError("Complete GRPO test evaluation is required")
    evaluate(config, run, "grpo")
    results = collect_results(run)
    if set(results) != {"base", "sft", "grpo"} or results["grpo"]["n"] != 1319:
        raise ValueError("Acceptance requires complete Base, SFT and GRPO only")
    summary, step, audits = validate_training(config, run, results)
    final = storage_inventory(run.parent)
    write_json(run / "stage5/storage_final.json", final)
    before = read(run / "stage5/storage_before.json")
    deltas = [{"name": row["name"], "path": row["path"], "after_bytes": row["bytes"],
               "delta_bytes": row["bytes"] - next(old["bytes"] for old in before["directories"]
                                                   if old["name"] == row["name"])}
              for row in final["directories"]]
    record = {"stage": 5, "status": "complete", "stopped_before_stage": 6,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              **results, "training": summary, "maximum_training_step": step,
              "rollout_batches": len(audits),
              "configuration_sha256": file_hash(run / "config.json"),
              "data_manifest_sha256": file_hash(run / "data/manifest.json"),
              "evaluation_contract_sha256": file_hash(run / "evaluation_contract.json"),
              "final_adapter_sha256": file_hash(run / "grpo/final/adapter_model.safetensors"),
              "checks": {"original_and_stage4_sources_data_dependencies_unchanged": True,
                         "complete_ordered_predictions_and_recomputed_metrics": True,
                         "sft_parent_and_adapter_reload_verified": True, "rollout_rewards_recomputed": True,
                         "ablation_started": False},
              "storage_deltas": deltas, "storage": check_storage(run),
              "tests": read(run / "stage5/preflight_tests.json"),
              "next_stage": "Stage 6 ablation and final analysis/delivery; requires separate authorization"}
    write_json(run / "stage5/acceptance.json", record)
    write_json(run / "stage5/worker_status.json", {"stage": 5, "status": "completed", "pid": os.getpid(),
               "stopped_before_stage": 6, "acceptance": "stage5/acceptance.json"})
    print("DONE Stage 5; verified Full GRPO; stopped before ablation", flush=True)


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
    stage = run / "stage5"
    if not (stage / "preflight_tests.json").exists() or not (stage / "storage_before.json").exists():
        raise ValueError("Stage 5 preflight tests, provenance and storage baseline are required")
    with FileLock(run / ".grpo_stage.lock", timeout=0):
        try:
            verify_inputs(config_path, run)
            if (stage / "acceptance.json").exists():
                print("Stage 5 acceptance already exists; preserve it", flush=True)
                return
            run_batches(config_path, run, args.resume_checkpoint)
            finish(config_path, run)
        except BaseException as error:
            write_json(stage / "worker_status.json", {"stage": 5, "status": "failed", "pid": os.getpid(),
                "error_type": type(error).__name__, "message": str(error).replace(str(Path.home()), "<user-home>"),
                "recovery": "Inspect GRPO logs and last complete checkpoint; fix the cause before repeating. Evaluation resumes with final weights; incomplete training needs --resume-checkpoint. Ablation is never launched."})
            raise


if __name__ == "__main__":
    main()
