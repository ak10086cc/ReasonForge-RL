"""Deterministic, identical evaluation across stages, with immutable inputs."""
import json
import math
import os
from pathlib import Path
import tempfile
import time
from .metrics import extract_answer, summarize
from .modeling import load_model, load_tokenizer, adapter_state, state_hash, ResourceMeter, release, device_dtype
from .prompts import SYSTEM, render_prompt
from .utils import write_json, write_jsonl, read_jsonl, digest, file_hash, versions
from .storage import check_storage


def evaluation_contract(config, run):
    tokenizer = load_tokenizer(config)
    contract = {"model": config["model"], "generation": config["evaluation"], "seed": config["seed"],
            "system_prompt": SYSTEM, "chat_template_hash": digest(tokenizer.chat_template),
            "parser_source_sha256": file_hash(Path(__file__).with_name("metrics.py")),
            "test_hash": file_hash(Path(run) / "data" / "test.jsonl"),
            "validation_hash": file_hash(Path(run) / "data" / "validation.jsonl")}
    saved = Path(run) / "evaluation_contract.json"
    # Preserve readable historical contracts; new experiments also freeze the
    # inference implementation across Base/SFT/GRPO, not just within a batch.
    if not saved.exists() or "implementation_hashes" in json.loads(saved.read_text(encoding="utf-8")):
        contract["implementation_hashes"] = {name: file_hash(Path(__file__).with_name(name))
                                              for name in ("evaluate.py", "modeling.py", "prompts.py")}
        device, dtype = device_dtype(config)
        contract["inference_runtime"] = {"device": device, "dtype": str(dtype),
                    "dependencies": {k: v for k, v in versions().items() if k in ("torch", "transformers", "peft", "tokenizers")}}
    return contract


def freeze_evaluation(config, run):
    path = Path(run) / "evaluation_contract.json"
    contract = evaluation_contract(config, run)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Frozen evaluation contract changed")
    else:
        write_json(path, contract)
    return contract


def _atomic_json(path, value):
    """A power interruption must not turn a progress file into partial JSON."""
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".evaluation-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_predictions(predictions, rows, config, stage, split):
    if len(predictions) > len(rows):
        raise ValueError("More predictions than evaluation questions")
    for saved, row in zip(predictions, rows):
        expected = {"question_id": row["question_id"], "question": row["question"],
                    "ground_truth": row["ground_truth"], "stage": stage, "split": split, "seed": config["seed"]}
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError("Saved predictions are not the unchanged ordered evaluation prefix")
        extracted = extract_answer(saved["prediction"])
        if (saved["extracted_answer"] != extracted or saved["parse_success"] != (extracted is not None)
                or saved["correct"] != (extracted == row["ground_truth"])
                or type(saved["generated_tokens"]) is not int or saved["generated_tokens"] < 0
                or type(saved["reached_eos"]) is not bool):
            raise ValueError("Saved prediction fields disagree with the parser or token count")


def _resources(sessions):
    return {"seconds": sum(s["seconds"] for s in sessions),
            **{key: max((s[key] for s in sessions if s[key] is not None), default=None)
               for key in ("peak_allocated_mib", "peak_reserved_mib")}}


def evaluate(config, run, stage="base", split="test", *, max_seconds=None, max_samples=None):
    import torch
    from transformers import set_seed
    started = time.monotonic()
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    if stage not in ("base", "sft", "grpo", "grpo_correctness_only"):
        raise ValueError("Unknown stage")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples must be a positive integer")
    contract_file = Path(run) / "evaluation_contract.json"
    if split == "test" and not contract_file.exists():
        raise ValueError("Freeze evaluation after validation before reading test")
    if contract_file.exists() and json.loads(contract_file.read_text(encoding="utf-8")) != evaluation_contract(config, run):
        raise ValueError("Evaluation data or generation settings differ from frozen contract")
    contract = evaluation_contract(config, run)
    directory = Path(run) / "evaluation" / f"{stage}_{split}"
    rows = read_jsonl(Path(run) / "data" / f"{split}.jsonl")
    if not rows or any(r["split"] != split for r in rows):
        raise ValueError("Evaluation split provenance violation")
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation question IDs")
    adapter = None if stage == "base" else Path(run) / stage / "final"
    binding = {"contract": contract, "stage": stage, "split": split,
               "data_sha256": file_hash(Path(run) / "data" / f"{split}.jsonl"),
               "adapter_sha256": file_hash(adapter / "adapter_model.safetensors") if adapter else None,
               "evaluator_source_sha256": file_hash(__file__),
               "dependencies": {k: v for k, v in versions().items() if k in ("torch", "transformers", "peft", "tokenizers")}}
    prediction_file = directory / "predictions.jsonl"
    predictions = read_jsonl(prediction_file) if prediction_file.exists() else []
    _validate_predictions(predictions, rows, config, stage, split)
    metrics_path = directory / "metrics.json"
    if (directory / "resume_manifest.json").exists():
        if json.loads((directory / "resume_manifest.json").read_text(encoding="utf-8")) != binding:
            raise ValueError("Evaluation resume identity changed (model, data, configuration, code or dependencies)")
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (len(predictions) != len(rows) or saved["prediction_sha256"] != file_hash(prediction_file)
                or saved["evaluation_contract_hash"] != digest(contract)
                or any(saved[k] != v for k, v in summarize(predictions).items())):
            raise ValueError("Completed evaluation failed integrity checks")
        # Also validate that the current adapter is the one previously evaluated.
        if adapter:
            from safetensors.torch import load_file
            expected = json.loads((adapter.parent / "summary.json").read_text(encoding="utf-8"))["weight_update"]["after_hash"]
            current = state_hash({k: v.float() for k, v in load_file(str(adapter / "adapter_model.safetensors")).items()})
            if current != expected:
                raise ValueError("Completed evaluation adapter changed")
        return saved
    manifest_path = directory / "resume_manifest.json"
    progress_path = directory / "progress.json"
    if directory.exists():
        if not manifest_path.exists() or not progress_path.exists():
            raise ValueError("Legacy/incomplete evaluation has no resume manifest; preserve it and use a fresh run")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != binding:
            raise ValueError("Evaluation resume identity changed (model, data, configuration, code or dependencies)")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        n = progress["completed"]
        if not 0 <= n <= len(predictions) or digest(predictions[:n]) != progress["prediction_prefix_hash"]:
            raise ValueError("Saved prediction prefix changed since the last progress record")
        if len(predictions) - n > 1:
            raise ValueError("Unexpected unrecorded prediction suffix")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        progress = {"completed": 0, "prediction_prefix_hash": digest([]), "sessions": [],
                    "status": "paused", "resource_scope": "recorded invocations only; hard-kill intervals may be missing"}
        _atomic_json(manifest_path, binding)
        _atomic_json(progress_path, progress)
    check_storage(run)
    set_seed(config["seed"])
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter)
    adapter_reload_verified = None
    if adapter:
        summary = json.loads((Path(run) / stage / "summary.json").read_text(encoding="utf-8"))
        adapter_reload_verified = state_hash(adapter_state(model)) == summary["weight_update"]["after_hash"]
        if not adapter_reload_verified:
            raise RuntimeError("Saved/reloaded adapter parameters differ")
    model.eval()
    initial_count = len(predictions)
    meter = ResourceMeter()
    try:
        with meter, torch.inference_mode():
            for row in rows[initial_count:]:
                check_storage(run)
                prompt = render_prompt(tokenizer, row["question"])
                batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                output = model.generate(**batch, do_sample=False,
                                        max_new_tokens=config["evaluation"]["max_new_tokens"],
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                ids = output[0, batch["input_ids"].shape[1]:].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True)
                predicted = extract_answer(text)
                record = {"question_id": row["question_id"], "question": row["question"],
                                    "ground_truth": row["ground_truth"], "prediction": text,
                                    "extracted_answer": predicted, "correct": predicted == row["ground_truth"],
                                    "parse_success": predicted is not None, "stage": stage, "split": split,
                                    "seed": config["seed"], "generated_tokens": len(ids),
                                    "reached_eos": tokenizer.eos_token_id in ids}
                # Append completed predictions immediately so interrupted runs retain evidence.
                with prediction_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                predictions.append(record)
                progress.update(completed=len(predictions), prediction_prefix_hash=digest(predictions), status="running")
                _atomic_json(progress_path, progress)
                if ((max_samples is not None and len(predictions) - initial_count >= max_samples)
                        or (max_seconds is not None and time.monotonic() - started >= max_seconds)):
                    break
    finally:
        if hasattr(meter, "result"):
            progress["sessions"].append({**meter.result, "new_predictions": len(predictions) - initial_count})
        progress.update(completed=len(predictions), prediction_prefix_hash=digest(predictions), status="paused")
        _atomic_json(progress_path, progress)
        del model
        release()
    if len(predictions) < len(rows):
        return {"status": "paused", "completed": len(predictions), "total": len(rows), "stage": stage, "split": split}
    metrics = {**summarize(predictions), "stage": stage, "split": split, "resources": _resources(progress["sessions"]),
               "resource_scope": progress["resource_scope"],
               "adapter_reload_verified": adapter_reload_verified, "model_revision": config["model"]["revision"],
               "prediction_sha256": file_hash(prediction_file),
               "evaluation_contract_hash": digest(contract),
               "completion_rate": sum(r["reached_eos"] for r in predictions) / len(predictions)}
    write_jsonl(directory / "errors.jsonl", [r for r in predictions if not r["correct"]])
    _atomic_json(metrics_path, metrics)
    progress["status"] = "completed"
    _atomic_json(progress_path, progress)
    return metrics
