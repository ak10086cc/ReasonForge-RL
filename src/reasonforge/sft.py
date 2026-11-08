from pathlib import Path
from .data import training_rows
from .modeling import load_model, load_tokenizer, adapter_state, weight_delta, ResourceMeter, release
from .prompts import sft_record
from .training import common_args, log_callback
from .utils import write_json, write_jsonl
from .recovery import prepare_stage
from .storage import storage_callback


def audit_loss_mask(trainer, prompt, tokenizer):
    """Inspect the actual TRL-prepared dataset and actual training collator."""
    import torch
    example = trainer.train_dataset[0]
    batch = trainer.data_collator([example])
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if example["input_ids"][:len(prefix)] != prefix:
        raise RuntimeError("Prepared SFT prompt does not match inference prompt tokens")
    labels = batch["labels"][0][:len(example["input_ids"])]
    prompt_positions = torch.arange(len(labels)) < len(prefix)
    if not prompt_positions.any() or not (labels[prompt_positions] == -100).all():
        raise RuntimeError("Prompt tokens contribute to SFT loss")
    if not (labels != -100).any():
        raise RuntimeError("All completion labels are masked")
    if not torch.equal(labels[~prompt_positions], batch["input_ids"][0][:len(labels)][~prompt_positions]):
        raise RuntimeError("Completion supervision differs from completion tokens")
    return {"prompt_masked_tokens": int(prompt_positions.sum()),
            "supervised_tokens": int((labels != -100).sum()), "passed": True}


def train(config, run, resume=None):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    set_seed(config["seed"])
    directory = prepare_stage(run, "sft", config, resume)
    tokenizer = load_tokenizer(config)
    prepared, exclusions = {}, {}
    for split in ("train", "validation"):
        rows = training_rows(run, split)
        records = [sft_record(row, tokenizer) for row in rows]
        # Keep complete solutions; do not silently cut away final answer supervision.
        keep = [len(tokenizer(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"])
                <= config["sft"]["max_length"] for r in records]
        exclusions[split] = [r["question_id"] for r, ok in zip(rows, keep) if not ok]
        records = [r for r, ok in zip(records, keep) if ok]
        if not records:
            raise ValueError(f"No complete {split} solution fits max_length")
        prepared[split] = Dataset.from_list(records)
        write_jsonl(directory / f"{split}_formatted.jsonl", records[:2])
    write_json(directory / "length_exclusions.json", exclusions)
    model = load_model(config, resume, trainable=True) if resume else load_model(config)
    lora = LoraConfig(r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
                      lora_dropout=config["lora"]["dropout"], target_modules=config["lora"]["target_modules"],
                      task_type="CAUSAL_LM", bias="none")
    args = SFTConfig(**common_args(config, directory, "sft"),
                     max_length=config["sft"]["max_length"], completion_only_loss=True,
                     packing=False, shuffle_dataset=False,
                     eval_strategy="steps", eval_steps=config["sft"]["save_steps"])
    trainer = SFTTrainer(model=model, args=args, train_dataset=prepared["train"],
                         eval_dataset=prepared["validation"], processing_class=tokenizer,
                         peft_config=None if resume else lora,
                         callbacks=[log_callback(directory / "train_log.jsonl"), storage_callback(run)])
    write_json(directory / "loss_mask_audit.json", audit_loss_mask(trainer, prepared["train"][0]["prompt"], tokenizer))
    before = adapter_state(trainer.model)
    with ResourceMeter() as meter:
        result = trainer.train(resume_from_checkpoint=resume)
        validation = trainer.evaluate()
    after = adapter_state(trainer.model)
    changes = weight_delta(before, after)
    if not changes["parameters_changed"]:
        raise RuntimeError("SFT finished without any adapter parameter update")
    trainer.save_model(str(directory / "final"))
    tokenizer.save_pretrained(directory / "final")
    summary = {"stage": "sft", "resources": meter.result, "train": result.metrics,
               "validation": validation, "weight_update": changes,
               "checkpoint": "sft/final", "train_samples": len(prepared["train"]),
               "validation_samples": len(prepared["validation"]), "resumed": resume is not None,
               "resource_scope": "current invocation only"}
    write_json(directory / "summary.json", summary)
    del trainer, model
    release()
    return summary
