"""Stage 2 only: Full-length resource measurements, never an official experiment."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasonforge.utils import (configure_runtime, create_run, digest, file_hash,
                              load_config, read_jsonl, write_json, write_jsonl)


def probe_config(full):
    config = deepcopy(full)
    config["name"] = "stage2-resource-probe"
    config["data"].update(train_samples=8, validation_samples=4)
    config["sft"].update(max_steps=4, save_steps=2)
    config["grpo"].update(max_steps=8, save_steps=4)
    config["ablation"] = False
    return config


def recover_training_splits(raw, manifest, config):
    """Reuse verified split membership without opening any test examples."""
    from reasonforge.data import question_hash
    from reasonforge.metrics import extract_answer
    if (manifest["revision"] != config["data"]["revision"]
            or manifest["seed"] != config["seed"]
            or manifest["splits"]["validation"]["total_count"] != config["data"]["validation_size"]):
        raise ValueError("Source split manifest differs from Full configuration")
    result = {}
    for split in ("train", "validation"):
        rows = []
        for qid in manifest["splits"][split]["ids"]:
            source, index = qid.split(":")
            if source != "official_train":
                raise ValueError("Resource probe accepts official training data only")
            row = dict(raw[int(index)])
            rows.append({**row, "question_id": qid, "question_hash": question_hash(row["question"]),
                         "ground_truth": extract_answer(row["answer"], allow_gold=True),
                         "source": source, "split": split})
        if digest(rows) != manifest["splits"][split]["hash"]:
            raise ValueError("Reconstructed training split hash differs from recorded split")
        result[split] = rows
    if {r["question_id"] for r in result["train"]} & {r["question_id"] for r in result["validation"]}:
        raise ValueError("Training and validation membership overlaps")
    return result


def choose_probe_rows(rows, lengths, prompt_lengths, count, max_length):
    """Include ordinary prefix examples and real length extremes, not test scores."""
    fits = [i for i, length in enumerate(lengths) if length <= max_length]
    if not fits or len(rows) < count or len(rows) != len(lengths) or len(rows) != len(prompt_lengths):
        raise ValueError("Insufficient or inconsistent probe candidates")
    indices = list(range(count - 2))
    indices += [max(fits, key=lambda i: lengths[i]), max(range(len(rows)), key=lambda i: prompt_lengths[i])]
    indices = list(dict.fromkeys(indices))
    indices += [i for i in range(len(rows)) if i not in indices][:count - len(indices)]
    return [rows[i] for i in indices]


def validation_probe(config, run, stage):
    """Same greedy generation as evaluate.py; validation only, no test contract."""
    import torch
    from transformers import set_seed
    from reasonforge.modeling import (load_model, load_tokenizer, adapter_state,
                                     state_hash, ResourceMeter, release)
    from reasonforge.prompts import render_prompt
    from reasonforge.storage import check_storage
    set_seed(config["seed"])
    tokenizer = load_tokenizer(config)
    adapter = None if stage == "base" else run / stage / "final"
    model = load_model(config, adapter)
    if adapter:
        expected = json.loads((adapter.parent / "summary.json").read_text(encoding="utf-8"))
        if state_hash(adapter_state(model)) != expected["weight_update"]["after_hash"]:
            raise RuntimeError("Probe adapter reload differs from saved parameters")
    model.eval()
    predictions = []
    try:
        with ResourceMeter() as meter, torch.inference_mode():
            for row in read_jsonl(run / "data" / "validation.jsonl"):
                check_storage(run)
                batch = tokenizer(render_prompt(tokenizer, row["question"]), return_tensors="pt",
                                  add_special_tokens=False).to(model.device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = model.generate(**batch, do_sample=False,
                    max_new_tokens=config["evaluation"]["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                ids = output[0, batch["input_ids"].shape[1]:].tolist()
                predictions.append({"question_id": row["question_id"], "split": "validation",
                    "source": "official_train", "stage": stage, "seconds": time.perf_counter() - start,
                    "prompt_tokens": batch["input_ids"].shape[1], "generated_tokens": len(ids),
                    "reached_eos": tokenizer.eos_token_id in ids,
                    "prediction": tokenizer.decode(ids, skip_special_tokens=True)})
                write_jsonl(run / "probe_validation" / stage / "predictions.jsonl", predictions)
                print(f"PROBE {stage} validation {len(predictions)}/4", flush=True)
        result = {"stage": stage, "samples": len(predictions), "resources": meter.result,
                  "adapter_reload_verified": adapter is not None,
                  "mean_generated_tokens": sum(r["generated_tokens"] for r in predictions) / len(predictions),
                  "scope": "resource probe only; no accuracy estimate or test evaluation"}
        write_json(run / "probe_validation" / stage / "resources.json", result)
        return result
    finally:
        del model
        release()


def length_stress(run):
    """One GRPO step at the real longest prompt + 512 forced completion tokens.

    Forcing a minimum length changes sampling and is diagnostic only. It must
    never be treated as the normal training setup or included in quality metrics.
    """
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig
    from reasonforge.grpo import ResumeSafeGRPOTrainer
    from reasonforge.modeling import (load_model, load_tokenizer, adapter_state,
                                     state_hash, ResourceMeter, release)
    from reasonforge.prompts import grpo_record
    from reasonforge.rewards import correctness_reward, format_reward
    from reasonforge.training import common_args, log_callback
    from reasonforge.storage import storage_callback, check_storage
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["name"] != "stage2-resource-probe":
        raise ValueError("Length stress accepts only a Stage 2 probe run")
    directory = run / "length_stress"
    directory.mkdir(exist_ok=False)
    config["grpo"].update(max_steps=1)
    write_json(directory / "config.json", config)
    write_json(directory / "identity.json", {"script_sha256": file_hash(__file__),
        "command": [Path(sys.executable).name, *sys.argv],
        "diagnostic_generation_override": {"min_new_tokens": config["grpo"]["max_completion_length"]},
        "quality_experiment": False})
    # Preserve the exact diagnostic implementation separately from the earlier
    # ordinary probe's immutable source_snapshot.zip.
    import shutil
    shutil.copyfile(__file__, directory / "resource_probe_source.py")
    set_seed(config["seed"])
    tokenizer = load_tokenizer(config)
    rows = read_jsonl(run / "data" / "train.jsonl")
    records = [grpo_record(row, tokenizer) for row in rows]
    record = max(records, key=lambda r: len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]))
    write_jsonl(directory / "input.jsonl", [record])
    model = load_model(config, run / "sft" / "final", trainable=True)
    expected = json.loads((run / "sft" / "summary.json").read_text(encoding="utf-8"))
    if state_hash(adapter_state(model)) != expected["weight_update"]["after_hash"]:
        raise RuntimeError("Stress model is not the recorded SFT adapter")
    c = config["grpo"]
    args = common_args(config, directory, "grpo")
    args["save_strategy"] = "no"
    args = GRPOConfig(**args, num_generations=c["num_generations"],
        max_completion_length=c["max_completion_length"], temperature=c["temperature"], top_p=1., top_k=0,
        beta=c["beta"], loss_type="grpo", scale_rewards="group", use_vllm=False,
        reward_weights=c["reward_weights"], mask_truncated_completions=False,
        generation_kwargs={"min_new_tokens": c["max_completion_length"]}, log_completions=False)
    trainer = ResumeSafeGRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list([record]),
        processing_class=tokenizer, reward_funcs=[correctness_reward, format_reward],
        callbacks=[log_callback(directory / "train_log.jsonl"), storage_callback(run)])
    try:
        with ResourceMeter() as meter:
            trainer.train()
        logs = read_jsonl(directory / "train_log.jsonl")
        step = next(row for row in logs if "completions/max_length" in row)
        if not (step["completions/min_length"] == step["completions/max_length"] == c["max_completion_length"]
                and trainer.state.global_step == 1):
            raise RuntimeError("Stress did not reach the configured completion boundary")
        write_json(directory / "resources.json", {"status": "completed", "resources": meter.result,
            "prompt_tokens": len(tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]),
            "completion_tokens_each": c["max_completion_length"], "generations": c["num_generations"],
            "optimizer_steps": trainer.state.global_step, "sampling_modified_for_length_stress_only": True,
            "storage": check_storage(run)})
    finally:
        del trainer, model
        release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--split-run", type=Path,
                        help="Existing run containing the unchanged seed-42 full split membership")
    parser.add_argument("--stress-run", type=Path, help="Completed Stage 2 probe for one diagnostic length-boundary step")
    args = parser.parse_args()
    # Process-local offline settings prevent downloads; no global cache relocation.
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    configure_runtime()
    if args.stress_run:
        if args.split_run:
            parser.error("Choose --stress-run or --split-run")
        length_stress(args.stress_run)
        return
    if args.split_run is None:
        parser.error("--split-run is required for the ordinary probe")
    from reasonforge.storage import storage_inventory, prepare_storage_policy, check_storage
    from reasonforge.modeling import device_dtype, load_tokenizer
    full = load_config(args.config)
    if device_dtype(full)[0] != "cuda":
        parser.error("This hardware probe requires the existing CUDA environment")
    before = storage_inventory(full["output_dir"])
    config = probe_config(full)
    run = create_run(config)
    console = (run / "console.log").open("x", encoding="utf-8", buffering=1)
    class Tee:
        def __init__(self, stream):
            self.stream = stream

        def write(self, value):
            console.write(value)
            return self.stream.write(value)

        def flush(self):
            console.flush()
            self.stream.flush()
    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
    print(f"RUN_DIR={run}", flush=True)
    write_json(run / "full_config.json", full)
    write_json(run / "storage_before.json", before)
    prepare_storage_policy(run, max_run_gib=1, min_free_gib=50)
    started = time.perf_counter()
    phase = "prepare_training_validation_only"
    wall_seconds = {}
    try:
        from datasets import load_dataset
        from reasonforge.prompts import sft_record, render_prompt
        from reasonforge import sft, grpo
        raw = load_dataset("openai/gsm8k", "main", revision=full["data"]["revision"], split="train")
        if not raw.cache_files or any(full["data"]["revision"] not in Path(f["filename"]).parts
                                     for f in raw.cache_files):
            raise RuntimeError("Cannot verify pinned training Arrow cache revision")
        manifest_path = args.split_run / "data" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = recover_training_splits(raw, manifest, full)
        tokenizer = load_tokenizer(config)
        audit = {"purpose": "stage2_resource_probe_not_full_experiment", "test_loaded": False,
                 "source_manifest_sha256": file_hash(manifest_path), "splits": {}}
        for split, rows in splits.items():
            records = [sft_record(row, tokenizer) for row in rows]
            lengths = [len(tokenizer(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"])
                       for r in records]
            prompts = [len(tokenizer(render_prompt(tokenizer, row["question"]),
                                    add_special_tokens=False)["input_ids"]) for row in rows]
            selected = choose_probe_rows(rows, lengths, prompts, config["data"][f"{split}_samples"],
                                         config["sft"]["max_length"])
            write_jsonl(run / "data" / f"{split}.jsonl", selected)
            indices = [rows.index(row) for row in selected]
            audit["splits"][split] = {"pool_count": len(rows), "pool_hash": digest(rows),
                "selected_count": len(selected), "selected_hash": digest(selected),
                "selected_ids": [r["question_id"] for r in selected],
                "selected_sft_tokens": [lengths[i] for i in indices],
                "selected_prompt_tokens": [prompts[i] for i in indices],
                "sft_eligible_pool": sum(n <= config["sft"]["max_length"] for n in lengths),
                "longest_eligible_sft_tokens": max(n for n in lengths if n <= config["sft"]["max_length"]),
                "longest_prompt_tokens": max(prompts)}
        write_json(run / "data" / "manifest.json", audit)
        for phase, function in (("base_validation", lambda: validation_probe(config, run, "base")),
                                ("sft_training", lambda: sft.train(config, run)),
                                ("sft_validation", lambda: validation_probe(config, run, "sft")),
                                ("grpo_training", lambda: grpo.train(config, run)),
                                ("grpo_validation", lambda: validation_probe(config, run, "grpo"))):
            check_storage(run)
            write_json(run / "status.json", {"stage": 2, "status": "running", "phase": phase})
            print(f"START {phase}", flush=True)
            begin = time.perf_counter()
            function()
            wall_seconds[phase] = time.perf_counter() - begin
            print(f"DONE {phase}: {wall_seconds[phase]:.2f}s", flush=True)
        write_json(run / "probe_summary.json", {"stage": 2, "status": "completed", "formal_full": False,
            "test_evaluated": False, "wall_seconds": wall_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "storage": check_storage(run), "next_stage_started": False})
        write_json(run / "status.json", {"stage": 2, "status": "completed", "stopped_before_stage": 3})
    except BaseException as error:
        write_json(run / "failure.json", {"phase": phase, "type": type(error).__name__,
            "message": str(error).replace(str(Path.home()), "<user-home>"),
            "frames": [{"file": Path(f.filename).name, "line": f.lineno} for f in
                       traceback.extract_tb(error.__traceback__)], "completed_wall_seconds": wall_seconds,
            "recovery": "Preserve this probe; diagnose the error before a new run. Do not start Full."})
        write_json(run / "status.json", {"stage": 2, "status": "failed", "phase": phase})
        raise
    finally:
        write_json(run / "storage_after.json", storage_inventory(full["output_dir"]))


if __name__ == "__main__":
    main()
