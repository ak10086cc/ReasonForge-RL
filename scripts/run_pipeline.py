"""Single CLI for preparation, training, evaluation, ablation and reports."""
import argparse
import json
from pathlib import Path
import sys
import time
from filelock import FileLock, Timeout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasonforge.utils import configure_runtime, load_config, create_run, write_json
from reasonforge.storage import prepare_storage_policy, check_storage, storage_inventory


class PhasePaused(Exception):
    """A planned evaluation boundary, not an experiment failure."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--phase", choices=["all", "env", "storage", "prepare", "baseline", "sft", "grpo", "evaluate", "ablation", "analyze"], default="all")
    parser.add_argument("--run-dir", type=Path, help="Existing run for an individual phase; its config must match")
    parser.add_argument("--resume-checkpoint", help="Trainer checkpoint (optimizer/scheduler/RNG included)")
    parser.add_argument("--eval-max-seconds", type=float, help="Evaluation time budget for this invocation; finish the current question before pausing")
    parser.add_argument("--eval-max-samples", type=int, help="Maximum new questions per evaluation target before pausing")
    parser.add_argument("--max-run-gib", type=float, default=5.0, help="Maximum logical output size per run (GiB)")
    parser.add_argument("--min-free-gib", type=float, default=50.0, help="Minimum free space on the output volume (GiB)")
    args = parser.parse_args()
    if args.resume_checkpoint and args.phase not in ("sft", "grpo", "ablation"):
        parser.error("--resume-checkpoint requires an individual training phase: sft, grpo or ablation")
    import math
    if args.eval_max_seconds is not None and (not math.isfinite(args.eval_max_seconds) or args.eval_max_seconds <= 0):
        parser.error("--eval-max-seconds must be positive and finite")
    if args.eval_max_samples is not None and args.eval_max_samples <= 0:
        parser.error("--eval-max-samples must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.max_run_gib, args.min_free_gib)):
        parser.error("Storage limits must be positive and finite")
    configure_runtime()
    if args.phase == "env":
        from reasonforge.hardware import write_environment
        print(json.dumps(write_environment(), ensure_ascii=True, indent=2))
        return
    config = load_config(args.config)
    if args.phase == "storage":
        print(json.dumps(storage_inventory(config["output_dir"]), ensure_ascii=True, indent=2))
        return
    run = args.run_dir
    if run is None:
        if args.phase not in ("all", "prepare"):
            parser.error("This phase requires --run-dir")
        run = create_run(config)
    else:
        if args.phase in ("all", "prepare"):
            parser.error("Use a fresh run for all/prepare to protect prior outputs")
        saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if saved != config:
            parser.error("--config must match the existing run config.json")
    print(f"RUN_DIR={run}", flush=True)
    last_success = "run initialized"
    evaluation_seconds = 0.0

    def execute(label, function, *pos, **kw):
        nonlocal last_success
        check_storage(run)
        print(f"START {label}", flush=True)
        value = function(*pos, **kw)
        if isinstance(value, dict) and value.get("status") == "paused":
            print(f"PAUSED {label}", flush=True)
            return value
        last_success = label
        write_json(run / "status.json", {"last_success": label, "status": "running"})
        print(f"DONE {label}", flush=True)
        return value

    def evaluate_stage(stage="base", split="test"):
        nonlocal evaluation_seconds
        remaining = None if args.eval_max_seconds is None else args.eval_max_seconds - evaluation_seconds
        if remaining is not None and remaining <= 0:
            raise PhasePaused({"stage": stage, "split": split, "reason": "evaluation time budget exhausted"})
        started = time.monotonic()
        result = execute(f"{stage}_{split}", evaluate.evaluate, config, run, stage, split,
                         max_seconds=remaining, max_samples=args.eval_max_samples)
        evaluation_seconds += time.monotonic() - started
        if result.get("status") == "paused":
            raise PhasePaused(result)
        return result

    def train_stage(stage, function, **kwargs):
        required = {"sft": ("base",), "grpo": ("base", "sft"),
                    "grpo_correctness_only": ("base", "sft", "grpo")}[stage]
        verified = analyze.collect_results(run)
        if any(name not in verified for name in required):
            raise ValueError(f"Stage {stage} requires completed, verified evaluations: {required}")
        directory = run / stage
        if args.resume_checkpoint is None and (directory / "summary.json").exists() and (directory / "final").is_dir():
            print(f"REUSE {stage} final; continue evaluation without retraining", flush=True)
            return
        execute(f"{stage}_training", function, config, run, resume=args.resume_checkpoint, **kwargs)

    lock = FileLock(run / ".run.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout:
        parser.error("This run is already active in another process; do not start a second phase")
    try:
        print(json.dumps({"storage": prepare_storage_policy(run, args.max_run_gib, args.min_free_gib)}), flush=True)
        from reasonforge import data, evaluate, sft, grpo, analyze
        if args.phase in ("all", "prepare"):
            execute("prepare", data.prepare, config, run)
        if args.phase in ("all", "baseline"):
            evaluate_stage("base", "validation")
            execute("freeze_evaluation", evaluate.freeze_evaluation, config, run)
            evaluate_stage()
        if args.phase in ("all", "sft"):
            train_stage("sft", sft.train)
            evaluate_stage("sft")
        if args.phase in ("all", "grpo"):
            train_stage("grpo", grpo.train)
            evaluate_stage("grpo")
        if args.phase == "ablation" or (args.phase == "all" and config["ablation"]):
            train_stage("grpo_correctness_only", grpo.train, correctness_only=True)
            evaluate_stage("grpo_correctness_only")
        if args.phase == "evaluate":
            for stage in ("base", "sft", "grpo", "grpo_correctness_only"):
                if stage == "base" or (run / stage / "final").exists():
                    evaluate_stage(stage)
        if args.phase in ("all", "analyze"):
            execute("analysis", analyze.analyze_run, run)
        storage = check_storage(run)
        write_json(run / "status.json", {"last_success": last_success, "status": "completed", "storage": storage,
                                         "completed_phase": args.phase})
    except PhasePaused as paused:
        record = {"status": "paused", "phase": args.phase, "last_success": last_success,
                  "evaluation": paused.args[0], "recovery": "Repeat the same individual phase and run directory; evaluation resumes automatically"}
        write_json(run / "status.json", record)
        print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)
    except KeyboardInterrupt:
        write_json(run / "status.json", {"status": "paused", "phase": args.phase, "last_success": last_success,
                   "recovery": "Evaluation resumes from saved predictions; training requires its last complete checkpoint"})
        raise SystemExit(130) from None
    except Exception as error:
        message = str(error).replace(str(Path.home()), "<user-home>").replace(str(Path.cwd()), "<project>")
        import traceback
        frames = [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                  for f in traceback.extract_tb(error.__traceback__)]
        record = {"status": "failed", "phase": args.phase, "last_success": last_success,
                  "error_type": type(error).__name__, "message": message, "frames": frames,
                  "command": ["python", *sys.argv], "recovery": "Inspect failure and last checkpoint; rerun a fixed phase in a fresh run or use --resume-checkpoint."}
        # Preserve prior failures across phase retries.
        path = run / "failures.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(run / "status.json", record)
        print(json.dumps(record, ensure_ascii=True, indent=2), file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        lock.release()


if __name__ == "__main__":
    main()
